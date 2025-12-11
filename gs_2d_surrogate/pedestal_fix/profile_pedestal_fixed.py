"""
Fixed pedestal profile for pPINN training.

Key changes from original:
1. Gentler pedestal transition (larger WIDTH, removed extra STEEPNESS factor)
2. Better alpha parameter ranges (bounded ratio)
3. Pedestal location moved further inside plasma
4. Optional pressure scaling for numerical stability

Usage:
    python profile_pedestal_fixed.py --num_params 5
    
    # With scaled collocation points (num_domain * num_params)
    python profile_pedestal_fixed.py --num_params 7 --scale_domain
    
    # Custom base domain points + scaling
    python profile_pedestal_fixed.py --num_params 9 --num_domain 4096 --scale_domain
    
    # Disable scaling (default)
    python profile_pedestal_fixed.py --num_params 9
    
    # Limit boundary points (e.g., for memory constraints)
    python profile_pedestal_fixed.py --num_params 9 --max_bc_points 500000
    
    # Scale BC limit with num_params: 100000 * 9 = 900000
    python profile_pedestal_fixed.py --num_params 9 --max_bc_points 100000 --scale_bc_points
    
    # No limit on boundary points (default)
    python profile_pedestal_fixed.py --num_params 9


CORRECTED pedestal profile for pPINN training.

Key fix: Pressure profile orientation
- Core (ψ << 0): HIGH pressure (p_core)
- Edge (ψ → 0): LOW pressure (p_edge)
- Pedestal is a DECREASE from core to edge

This matches the physical H-mode profile where:
- Hot, high-pressure plasma in the core
- Steep gradient at the pedestal
- Low pressure at the edge/SOL

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
import os

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
parser.add_argument('--num_domain', type=int, default=2048,
                    help='Base number of PDE collocation points')
parser.add_argument('--scale_domain', action='store_true', default=False,
                    help='Scale num_domain with num_params (num_domain * num_params)')
parser.add_argument('--max_bc_points', type=int, default=None,
                    help='Base maximum boundary points (default: None = no limit)')
parser.add_argument('--scale_bc_points', action='store_true', default=False,
                    help='Scale max_bc_points with num_params')
args = parser.parse_args()

print("Number of parameters:", args.num_params)

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
# Shape parameter ranges
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
# CORRECTED Pedestal Profile Parameters
# ============================================================

# ψ convention in this code:
#   ψ_axis < 0  (most negative at magnetic axis, i.e., core)
#   ψ_boundary = 0  (at plasma edge)
#
# For H-mode pedestal:
#   p_core = HIGH (in the core where ψ << 0)
#   p_edge = LOW  (at the edge where ψ → 0)
#   Pedestal location: ψ_ped ~ -0.08 (between core and edge)

PSI_PED = -0.08         # Pedestal location
WIDTH = 0.06            # Pedestal width in psi units

NUM_ALPHA = 2
INPUT_DIM = 2 + NUM_ALPHA + 3  # R, Z, α0, α1, eps, kappa, delta

# Alpha ranges:
# α0 (p_edge): Edge pressure (LOW) [0.05, 0.2]
# α1 (p_core): Core pressure (HIGH) [0.3, 1.0]
#
# Physical constraint: p_core > p_edge always
# Gradient: dp/dψ ~ (p_edge - p_core) / WIDTH < 0 (pressure decreases outward)

alpha_ranges = [
    np.linspace(0.05, 0.2, num_param),   # α0: p_edge (low values)
    np.linspace(0.3, 1.0, num_param),    # α1: p_core (high values)
]


def p_of_psi_pedestal(psi, alpha):
    """
    CORRECTED H-mode pedestal pressure profile.
    
    Physical behavior:
    - p = p_core (HIGH) in core where ψ << 0
    - p = p_edge (LOW) at edge where ψ → 0
    - Smooth tanh transition at pedestal location ψ_ped
    
    Parameters:
    -----------
    psi : tensor
        Poloidal flux values, shape (N, 1)
        Convention: ψ < 0 in core, ψ = 0 at boundary
    alpha : tensor
        Profile parameters, shape (N, 2)
        alpha[:, 0] = p_edge (edge pressure, LOW)
        alpha[:, 1] = p_core (core pressure, HIGH)
    
    Returns:
    --------
    p : tensor
        Pressure values, shape (N, 1)
    """
    p_edge = alpha[:, 0:1]  # LOW pressure at edge
    p_core = alpha[:, 1:2]  # HIGH pressure in core
    
    # Smooth step function:
    # H(ψ) = 0.5 * (1 + tanh((ψ - ψ_ped) / Δ))
    # 
    # When ψ << ψ_ped (deep core): H → 0
    # When ψ >> ψ_ped (near edge): H → 1
    arg = (psi - PSI_PED) / WIDTH
    H = 0.5 * (1.0 + tf.tanh(arg))
    
    # CORRECTED pressure formula:
    # p = p_core when H=0 (core)
    # p = p_edge when H=1 (edge)
    # p = p_core * (1 - H) + p_edge * H
    p = p_core * (1.0 - H) + p_edge * H
    
    return p


def dp_dpsi_pedestal(psi, alpha):
    """
    Compute dp/dψ using automatic differentiation.
    
    Note: dp/dψ should be NEGATIVE (pressure decreases as ψ increases toward edge)
    """
    with tf.GradientTape() as tape:
        tape.watch(psi)
        p_val = p_of_psi_pedestal(psi, alpha)
    return tape.gradient(p_val, psi)


def pde_pedestal(x, u):
    """
    Grad-Shafranov equation with pedestal pressure profile.
    
    Δ*ψ = -μ₀R²(dp/dψ) - F(dF/dψ)
    
    For simplified case (F = const):
    ψ_RR - ψ_R/R + ψ_ZZ = -R² * dp/dψ
    
    Since dp/dψ < 0 (pressure decreases outward), the RHS is positive.
    """
    psi = u[:, 0:1]
    R = x[:, 0:1]
    
    # Spatial derivatives
    psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
    psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
    psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
    
    # Extract pressure parameters
    alpha = x[:, 2:2+NUM_ALPHA]
    
    # Pressure gradient (should be negative)
    dpdpsi = dp_dpsi_pedestal(psi, alpha)
    
    # GS equation residual
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
                Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val) * np.sin(tau))
                Zb = eps_val * kappa_val * np.sin(tau)
                
                for a0 in alpha_ranges[0]:  # p_edge
                    for a1 in alpha_ranges[1]:  # p_core
                        # Only include if p_core > p_edge (physical constraint)
                        if a1 > a0:
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
    
    u_boundary = np.zeros((x_boundary.shape[0], 1))
    
    print(f"Generated {len(x_boundary)} boundary points")
    
    return x_boundary, u_boundary


# ============================================================
# Visualization of CORRECTED profile
# ============================================================

def plot_pressure_profile():
    """
    Plot the corrected pressure profile to verify orientation.
    """
    import matplotlib.pyplot as plt
    
    psi_vals = np.linspace(-0.5, 0.1, 500)
    
    # Test parameters
    p_edge = 0.1   # LOW at edge
    p_core = 0.8   # HIGH in core
    
    # Compute H
    H = 0.5 * (1.0 + np.tanh((psi_vals - PSI_PED) / WIDTH))
    
    # Compute pressure (CORRECTED)
    p = p_core * (1.0 - H) + p_edge * H
    
    # Compute gradient
    dpdpsi = (p_edge - p_core) / WIDTH * 0.5 * (1 - np.tanh((psi_vals - PSI_PED) / WIDTH)**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pressure profile
    ax1 = axes[0]
    ax1.plot(psi_vals, p, 'b-', lw=2)
    ax1.axvline(x=0, color='r', ls='--', label='ψ=0 (boundary)')
    ax1.axvline(x=PSI_PED, color='orange', ls='--', label=f'ψ_ped={PSI_PED}')
    ax1.axhline(y=p_core, color='green', ls=':', alpha=0.5, label=f'p_core={p_core}')
    ax1.axhline(y=p_edge, color='purple', ls=':', alpha=0.5, label=f'p_edge={p_edge}')
    ax1.set_xlabel('ψ', fontsize=12)
    ax1.set_ylabel('p(ψ)', fontsize=12)
    ax1.set_title(f'CORRECTED Pressure Profile\np_edge={p_edge}, p_core={p_core}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('Core\n(HIGH p)', xy=(-0.4, p_core-0.05), fontsize=11, ha='center')
    ax1.annotate('Edge\n(LOW p)', xy=(0.05, p_edge+0.05), fontsize=11, ha='center')
    ax1.annotate('Pedestal\n(steep gradient)', xy=(PSI_PED, 0.5*(p_core+p_edge)), 
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'),
                xytext=(PSI_PED-0.15, 0.5*(p_core+p_edge)))
    
    # Gradient profile
    ax2 = axes[1]
    ax2.plot(psi_vals, dpdpsi, 'r-', lw=2)
    ax2.axvline(x=0, color='r', ls='--', label='ψ=0 (boundary)')
    ax2.axvline(x=PSI_PED, color='orange', ls='--', label=f'ψ_ped={PSI_PED}')
    ax2.axhline(y=0, color='gray', ls='-', alpha=0.5)
    ax2.set_xlabel('ψ', fontsize=12)
    ax2.set_ylabel('dp/dψ', fontsize=12)
    ax2.set_title('Pressure Gradient', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for gradient sign
    ax2.annotate('dp/dψ < 0\n(pressure decreases\ntoward edge)', 
                xy=(PSI_PED, dpdpsi.min()/2), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('pressure_profile_corrected.png', dpi=150)
    plt.close()
    print("Saved: pressure_profile_corrected.png")
    
    # Print verification
    print(f"\nVerification:")
    print(f"  At ψ = -0.4 (core):  p = {p_core * (1 - 0.5*(1+np.tanh((-0.4-PSI_PED)/WIDTH))) + p_edge * 0.5*(1+np.tanh((-0.4-PSI_PED)/WIDTH)):.3f}")
    print(f"  At ψ = 0.0 (edge):   p = {p_core * (1 - 0.5*(1+np.tanh((0-PSI_PED)/WIDTH))) + p_edge * 0.5*(1+np.tanh((0-PSI_PED)/WIDTH)):.3f}")
    print(f"  Expected: core ~ {p_core}, edge ~ {p_edge}")


if __name__ == "__main__":
    
    # First, generate and show the corrected pressure profile
    plot_pressure_profile()
    
    TIME = time.strftime("%m%d%Y_%H%M")
    
    if args.output_dir:
        PATH = args.output_dir
    else:
        PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}_pedestal_corrected_{args.num_params}"
    
    os.makedirs(PATH, exist_ok=True)
    print(f"Output directory: {PATH}")
    
    # --------------------------------------------------------
    # Generate training data
    # --------------------------------------------------------
    print("\nGenerating boundary data...")
    x_bc, u_bc = gen_boundary_data(num_boundary_pts=201)
    
    # Optionally subsample boundary points
    if args.max_bc_points is not None:
        if args.scale_bc_points:
            actual_max_bc = args.max_bc_points * num_param
        else:
            actual_max_bc = args.max_bc_points
        
        if len(x_bc) > actual_max_bc:
            idx = np.random.choice(len(x_bc), actual_max_bc, replace=False)
            x_bc = x_bc[idx]
            u_bc = u_bc[idx]
            print(f"Subsampled to {actual_max_bc} boundary points")
    
    bc = dde.PointSetBC(x_bc, u_bc)
    
    # --------------------------------------------------------
    # Create geometry
    # --------------------------------------------------------
    print("\nSetting up geometry and PDE...")
    
    geom = dde.geometry.HyperEllipticalToroid(
        eps_range=eps0,
        kappa_range=kappa0,
        delta_range=delta0,
        alpha_ranges=alpha_ranges,
        num_param=num_param,
        psi_boundary_points=200
    )
    
    if args.scale_domain:
        actual_num_domain = args.num_domain * num_param
    else:
        actual_num_domain = args.num_domain
    
    data = dde.data.PDE(
        geom,
        pde_pedestal,
        [bc],
        num_domain=actual_num_domain,
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
        maxiter=50000,
        maxcor=50,
        ftol=0,
        gtol=1e-10,
        maxfun=100000,
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
    # Validation plot
    # --------------------------------------------------------
    print("\nGenerating validation plot...")
    
    eps_test = 0.32
    kap_test = 2.0
    delt_test = 0.0
    a0_test = 0.1   # p_edge (LOW)
    a1_test = 0.6   # p_core (HIGH)
    
    n_grid = 100
    r = np.linspace(1 - 1.5*eps_test, 1 + 1.5*eps_test, n_grid)
    z = np.linspace(-1.5*eps_test*kap_test, 1.5*eps_test*kap_test, n_grid)
    RR, ZZ = np.meshgrid(r, z, indexing='ij')
    
    X_test = np.zeros((n_grid**2, INPUT_DIM))
    X_test[:, 0] = RR.ravel()
    X_test[:, 1] = ZZ.ravel()
    X_test[:, 2] = a0_test  # p_edge
    X_test[:, 3] = a1_test  # p_core
    X_test[:, 4] = eps_test
    X_test[:, 5] = kap_test
    X_test[:, 6] = delt_test
    
    psi_pred = model.predict(X_test).reshape(n_grid, n_grid)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    pcm = ax1.pcolormesh(RR, ZZ, psi_pred, shading='auto', cmap='RdBu_r')
    plt.colorbar(pcm, ax=ax1, label='ψ')
    
    tau = np.linspace(0, 2*np.pi, 200)
    R_bnd = 1 + eps_test * np.cos(tau + np.arcsin(delt_test) * np.sin(tau))
    Z_bnd = eps_test * kap_test * np.sin(tau)
    ax1.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Boundary')
    
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title(f'ψ field (eps={eps_test}, kappa={kap_test}, delta={delt_test}, p_edge={a0_test}, p_core={a1_test})')
    ax1.set_aspect('equal')
    ax1.legend()
    
    ax2 = axes[1]
    levels = np.linspace(psi_pred.min(), psi_pred.max(), 20)
    cs = ax2.contour(RR, ZZ, psi_pred, levels=levels, cmap='coolwarm')
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    
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