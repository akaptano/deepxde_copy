"""
Fixed pedestal profile for pPINN training.

Key changes from original:
1. Gentler pedestal transition (larger WIDTH, removed extra STEEPNESS factor)
2. Better alpha parameter ranges (bounded ratio)
3. Pedestal location moved further inside plasma
4. Optional pressure scaling for numerical stability

Usage:
    python profile_pedestal_fixed.py --num_params 5

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from contextlib import nullcontext

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument('--num_params', type=int, default=5,
                    help='Number of parameters per dimension')
parser.add_argument('--epochs_adam', type=int, default=1000,
                    help='Adam epochs')
parser.add_argument('--epochs_lbfgs', type=int, default=10000,
                    help='L-BFGS epochs')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: auto-generated)')
args = parser.parse_args()

# DeepXDE setup
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)

import deepxde as dde
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

try:
    tf.config.optimizer.set_jit(True)
except:
    pass

# Configure GPU
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPU devices:", gpus)
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')

print("Using DeepXDE from:", dde.__file__)
dde.config.set_default_float("float64")


# ============================================================
# Shape parameter ranges (same as original)
# ============================================================
eps_deviation = 0.2
kappa_deviation = 0.75
delta_deviation = 0.5

eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)
delta0 = (0 - delta_deviation, 0 + delta_deviation)

num_param = args.num_params
eps_vals = np.linspace(eps0[0], eps0[1], num_param)
kappa_vals = np.linspace(kappa0[0], kappa0[1], num_param)
delta_vals = np.linspace(delta0[0], delta0[1], num_param)


# ============================================================
# FIXED Pedestal Profile Parameters
# ============================================================

# Key insight: For the Solov'ev solution with A=-0.155, psi ranges roughly from
# psi_min ≈ -0.1 to -0.3 (at magnetic axis) to psi=0 (at boundary).
# The pedestal should be located between the edge and the core.

PSI_PED = -0.08         # Pedestal location (closer to edge for H-mode)
WIDTH = 0.06            # Pedestal width in psi units (wider = gentler transition)

# Pressure ranges - use parameters that give reasonable gradients
# α0 = base pressure scale (controls overall magnitude)
# α1 = pedestal-to-core pressure ratio (controls gradient strength)

NUM_ALPHA = 2
INPUT_DIM = 2 + NUM_ALPHA + 3  # R, Z, α0, α1, eps, kappa, delta

# Alpha ranges:
# α0 (p_scale): Overall pressure magnitude [0.1, 0.5]
# α1 (p_ratio): Core-to-edge pressure ratio [1.5, 4.0]
#   - p_edge = p_scale
#   - p_core = p_scale * p_ratio
#   - This bounds the gradient: dp ~ p_scale * (p_ratio - 1) / WIDTH

alpha_ranges = [
    np.linspace(0.1, 0.5, num_param),    # α0: pressure scale
    np.linspace(1.5, 4.0, num_param),    # α1: core/edge ratio
]


def p_of_psi_pedestal(psi, alpha):
    """
    H-mode pedestal pressure profile.
    
    p(ψ) = p_core + (p_edge - p_core) * H(ψ)
    
    where H(ψ) is a smooth step function from 0 (core) to 1 (edge).
    
    Parameters:
    -----------
    psi : tensor
        Poloidal flux values, shape (N, 1)
    alpha : tensor
        Profile parameters, shape (N, 2)
        alpha[:, 0] = p_scale (base pressure)
        alpha[:, 1] = p_ratio (core/edge ratio)
    
    Returns:
    --------
    p : tensor
        Pressure values, shape (N, 1)
    """
    p_scale = alpha[:, 0:1]
    p_ratio = alpha[:, 1:2]
    
    p_edge = p_scale
    p_core = p_scale * p_ratio
    
    # Smooth step function: H(ψ) = 0.5 * (1 + tanh((ψ - ψ_ped) / Δ))
    # H → 0 as ψ → -∞ (deep core)
    # H → 1 as ψ → 0 (edge)
    arg = (psi - PSI_PED) / WIDTH
    H = 0.5 * (1.0 + tf.tanh(arg))
    
    # Pressure: high in core, low at edge
    p = p_core * (1.0 - H) + p_edge * H
    
    return p


def dp_dpsi_pedestal(psi, alpha):
    """
    Compute dp/dψ using automatic differentiation.
    """
    with tf.GradientTape() as tape:
        tape.watch(psi)
        p_val = p_of_psi_pedestal(psi, alpha)
    return tape.gradient(p_val, psi)


def pde_pedestal(x, u):
    """
    Grad-Shafranov equation with pedestal pressure profile.
    
    Δ*ψ = -μ₀R²(dp/dψ) - F(dF/dψ)
    
    For the simplified case (F = const), this becomes:
    ψ_RR - ψ_R/R + ψ_ZZ = -R² * dp/dψ
    
    Note: We use the standard GS convention where the RHS is the source.
    """
    psi = u[:, 0:1]
    R = x[:, 0:1]
    
    # Spatial derivatives
    psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
    psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
    psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
    
    # Extract pressure parameters
    alpha = x[:, 2:2+NUM_ALPHA]
    
    # Pressure gradient
    dpdpsi = dp_dpsi_pedestal(psi, alpha)
    
    # GS equation residual
    # Note: Standard form is Δ*ψ + R²p'(ψ) = 0
    # So residual = ψ_RR - ψ_R/R + ψ_ZZ + R² * dp/dψ
    GS = psi_RR - psi_R/R + psi_ZZ + R**2 * dpdpsi
    
    return GS


def gen_boundary_data(num_boundary_pts):
    """
    Generate boundary training data (ψ = 0 on plasma boundary).
    """
    N = num_boundary_pts
    tau = np.linspace(0, 2*np.pi, N)
    
    R_list, Z_list = [], []
    alpha_list, eps_list, kappa_list, delta_list = [], [], [], []
    
    for eps_val in eps_vals:
        for kappa_val in kappa_vals:
            for delta_val in delta_vals:
                # Boundary parametric curve
                Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val) * np.sin(tau))
                Zb = eps_val * kappa_val * np.sin(tau)
                
                # Loop over pressure parameters
                for a0 in alpha_ranges[0]:
                    for a1 in alpha_ranges[1]:
                        R_list.append(Rb)
                        Z_list.append(Zb)
                        alpha_list.append(np.column_stack([
                            a0 * np.ones(N),
                            a1 * np.ones(N)
                        ]))
                        eps_list.append(eps_val * np.ones((N, 1)))
                        kappa_list.append(kappa_val * np.ones((N, 1)))
                        delta_list.append(delta_val * np.ones((N, 1)))
    
    R_flat = np.concatenate(R_list)[:, None]
    Z_flat = np.concatenate(Z_list)[:, None]
    alpha_flat = np.concatenate(alpha_list)
    eps_flat = np.concatenate(eps_list)
    kappa_flat = np.concatenate(kappa_list)
    delta_flat = np.concatenate(delta_list)
    
    x_boundary = np.hstack([
        R_flat, Z_flat, alpha_flat, eps_flat, kappa_flat, delta_flat
    ])
    
    # Boundary condition: ψ = 0
    u_boundary = np.zeros((x_boundary.shape[0], 1))
    
    print(f"Generated {len(x_boundary)} boundary points")
    print(f"  Shape: {x_boundary.shape}")
    
    return x_boundary, u_boundary


# ============================================================
# Import the HyperEllipticalToroid geometry class
# ============================================================

# The HyperEllipticalToroid class from geometry_nd.py properly handles:
# - Boundary point generation for all parameter combinations
# - Inside/outside checking for the D-shaped domain
# - Random point sampling within the geometry

# Note: We need to make sure deepxde can find the geometry_nd module
# It should be in the deepxde/geometry/ directory


# ============================================================
# Main training script
# ============================================================

if __name__ == "__main__":
    
    TIME = time.strftime("%m%d%Y_%H%M")
    
    if args.output_dir:
        PATH = args.output_dir
    else:
        PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}_pedestal_fixed"
    
    os.makedirs(PATH, exist_ok=True)
    print(f"Output directory: {PATH}")
    
    # --------------------------------------------------------
    # Generate training data
    # --------------------------------------------------------
    print("\nGenerating boundary data...")
    x_bc, u_bc = gen_boundary_data(num_boundary_pts=201)
    
    # Subsample if too large
    MAX_BC_POINTS = 200000
    if len(x_bc) > MAX_BC_POINTS:
        idx = np.random.choice(len(x_bc), MAX_BC_POINTS, replace=False)
        x_bc = x_bc[idx]
        u_bc = u_bc[idx]
        print(f"Subsampled to {MAX_BC_POINTS} boundary points")
    
    bc = dde.PointSetBC(x_bc, u_bc)
    
    # --------------------------------------------------------
    # Create geometry using HyperEllipticalToroid
    # --------------------------------------------------------
    print("\nSetting up geometry and PDE...")
    
    # The HyperEllipticalToroid class expects alpha_ranges as a list of arrays
    geom = dde.geometry.HyperEllipticalToroid(
        eps_range=eps0,
        kappa_range=kappa0,
        delta_range=delta0,
        alpha_ranges=alpha_ranges,
        num_param=num_param,
        psi_boundary_points=200
    )
    
    data = dde.data.PDE(
        geom,
        pde_pedestal,
        [bc],
        num_domain=2048,
        num_boundary=0,
        num_test=100,
        train_distribution="LHS"
    )
    
    # --------------------------------------------------------
    # Network architecture
    # --------------------------------------------------------
    DEPTH = 4
    BREADTH = 40
    AF = "swish"
    LR = 2e-2
    
    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    model = dde.Model(data, net)
    
    # --------------------------------------------------------
    # Training: Adam phase
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: Adam optimizer")
    print("="*60)
    
    decay_rate = ("inverse time", 100, 0.1)
    model.compile("adam", lr=LR, decay=decay_rate, loss_weights=[1, 100])
    
    loss_history, train_state = model.train(
        epochs=args.epochs_adam,
        display_every=100
    )
    
    dde.saveplot(loss_history, train_state, issave=True, isplot=True,
                output_dir=PATH, output_fname="loss_adam")
    
    # --------------------------------------------------------
    # Training: L-BFGS phase
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: L-BFGS optimizer")
    print("="*60)
    
    from deepxde.optimizers import set_LBFGS_options
    set_LBFGS_options(
        maxiter=20000,
        maxcor=50,
        ftol=0,
        gtol=1e-10,
        maxfun=15000,
        maxls=50,
    )
    
    model.compile("L-BFGS-B", loss_weights=[1, 100])
    
    loss_history, train_state = model.train(
        epochs=args.epochs_lbfgs,
        display_every=100
    )
    
    dde.saveplot(loss_history, train_state, issave=True, isplot=True,
                output_dir=PATH, output_fname="loss_lbfgs")
    
    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    model.save(f"{PATH}/pedestal_model", protocol="backend", verbose=1)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {PATH}")
    print("="*60)
    
    # --------------------------------------------------------
    # Quick validation plot
    # --------------------------------------------------------
    print("\nGenerating validation plot...")
    
    # Test at center of parameter space
    eps_test = 0.32
    kap_test = 2.0
    delt_test = 0.0
    a0_test = 0.3
    a1_test = 2.5
    
    # Create evaluation grid
    n_grid = 100
    r = np.linspace(1 - 1.5*eps_test, 1 + 1.5*eps_test, n_grid)
    z = np.linspace(-1.5*eps_test*kap_test, 1.5*eps_test*kap_test, n_grid)
    RR, ZZ = np.meshgrid(r, z, indexing='ij')
    
    X_test = np.zeros((n_grid**2, INPUT_DIM))
    X_test[:, 0] = RR.ravel()
    X_test[:, 1] = ZZ.ravel()
    X_test[:, 2] = a0_test
    X_test[:, 3] = a1_test
    X_test[:, 4] = eps_test
    X_test[:, 5] = kap_test
    X_test[:, 6] = delt_test
    
    psi_pred = model.predict(X_test).reshape(n_grid, n_grid)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax1 = axes[0]
    pcm = ax1.pcolormesh(RR, ZZ, psi_pred, shading='auto', cmap='RdBu_r')
    plt.colorbar(pcm, ax=ax1, label='ψ')
    
    # Boundary
    tau = np.linspace(0, 2*np.pi, 200)
    R_bnd = 1 + eps_test * np.cos(tau + np.arcsin(delt_test) * np.sin(tau))
    Z_bnd = eps_test * kap_test * np.sin(tau)
    ax1.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Boundary')
    
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title(f'ψ field (eps={eps_test}, kappa={kap_test}, delta={delt_test})')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Contours
    ax2 = axes[1]
    levels = np.linspace(psi_pred.min(), psi_pred.max(), 20)
    cs = ax2.contour(RR, ZZ, psi_pred, levels=levels, cmap='coolwarm')
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    
    # Try ψ=0 contour
    try:
        cs0 = ax2.contour(RR, ZZ, psi_pred, levels=[0.0], colors='green', linewidths=3)
        ax2.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Boundary')
    except:
        pass
    
    ax2.set_xlabel('R')
    ax2.set_ylabel('Z')
    ax2.set_title('ψ contours (green = ψ=0)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'validation_plot.png'), dpi=150)
    plt.close()
    
    print(f"Validation plot saved to: {PATH}/validation_plot.png")
    print(f"\nψ statistics: min={psi_pred.min():.4f}, max={psi_pred.max():.4f}")
    
    if psi_pred.min() < 0 < psi_pred.max():
        print("✓ ψ field has proper zero crossing")
    else:
        print("⚠ WARNING: ψ field may not have proper zero crossing!")