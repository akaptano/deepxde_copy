"""
Diagnostic script for the CORRECTED pedestal profile pPINN models.

This script helps analyze the trained pedestal pPINN model and verify
that the pressure profile and psi field are correct.

Usage:
    python diagnose_pedestal_corrected.py --model_path /path/to/checkpoint.ckpt

It will generate diagnostic plots for:
1. Psi field heatmap with contour lines
2. Psi values along the midplane (Z=0)
3. Pressure profile p(psi) for the given alpha parameters
4. dp/dpsi profile (the source term driver)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Add DeepXDE path
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import deepxde as dde
dde.config.set_default_float("float64")

# ============================================================
# CORRECTED Pedestal profile parameters (must match profile_pedestal_fixed.py)
# ============================================================
PSI_PED = -0.08    # Pedestal location (CORRECTED)
WIDTH = 0.06       # Pedestal width (CORRECTED)
# Note: No STEEPNESS factor in the new version!

# Shape parameter ranges
eps_deviation = 0.2
kappa_deviation = 0.75
delta_deviation = 0.5
eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)
delta0 = (0 - delta_deviation, 0 + delta_deviation)


def p_of_psi_pedestal_numpy(psi, p_edge, p_core):
    """
    CORRECTED NumPy version of the pedestal pressure profile.
    
    Matches the TensorFlow version in profile_pedestal_fixed.py:
    - p = p_core when psi << 0 (core)
    - p = p_edge when psi -> 0 (edge)
    """
    arg = (psi - PSI_PED) / WIDTH
    H = 0.5 * (1.0 + np.tanh(arg))
    return p_core * (1.0 - H) + p_edge * H


def dp_dpsi_pedestal_numpy(psi, p_edge, p_core):
    """
    CORRECTED NumPy version of dp/dpsi for pedestal profile.
    
    Analytical derivative of p_of_psi_pedestal_numpy.
    """
    arg = (psi - PSI_PED) / WIDTH
    sech2 = 1.0 / np.cosh(arg)**2
    # d/dpsi [p_core * (1-H) + p_edge * H] = (p_edge - p_core) * dH/dpsi
    # dH/dpsi = 0.5 * (1/WIDTH) * sech^2(arg)
    return (p_edge - p_core) * 0.5 * (1.0 / WIDTH) * sech2


def create_diagnostic_plots(model, eps, kappa, delta, alpha, save_dir, grid_size=200):
    """
    Create comprehensive diagnostic plots for a pedestal pPINN model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    p_edge, p_core = alpha[0], alpha[1]
    
    # Create evaluation grid
    margin = 0.15
    r = np.linspace(1 - eps - margin, 1 + eps + margin, grid_size)
    z = np.linspace(-eps * kappa - margin, eps * kappa + margin, grid_size)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    
    # Build input tensor: [R, Z, p_edge, p_core, eps, kappa, delta]
    N = grid_size * grid_size
    X_in = np.zeros((N, 7))
    X_in[:, 0] = RR.ravel()
    X_in[:, 1] = ZZ.ravel()
    X_in[:, 2] = p_edge
    X_in[:, 3] = p_core
    X_in[:, 4] = eps
    X_in[:, 5] = kappa
    X_in[:, 6] = delta
    
    # Predict psi
    psi_pred = model.predict(X_in).reshape(grid_size, grid_size)
    
    # Analytic boundary for reference
    tau = np.linspace(0, 2*np.pi, 400)
    R_bnd = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    Z_bnd = eps * kappa * np.sin(tau)
    
    # ============================================================
    # Plot 1: Psi field and contours
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax1 = axes[0]
    pcm = ax1.pcolormesh(RR, ZZ, psi_pred, shading='auto', cmap='RdBu_r')
    ax1.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Boundary')
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title(f'psi field (p_edge={p_edge}, p_core={p_core})')
    ax1.set_aspect('equal')
    ax1.legend()
    plt.colorbar(pcm, ax=ax1, label='psi')
    
    # Contours
    ax2 = axes[1]
    psi_min, psi_max = psi_pred.min(), psi_pred.max()
    
    levels = np.linspace(psi_min, psi_max, 20)
    cs = ax2.contour(RR, ZZ, psi_pred, levels=levels, cmap='coolwarm')
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    
    # Plot psi=0 contour
    if psi_min < 0 < psi_max:
        cs0 = ax2.contour(RR, ZZ, psi_pred, levels=[0.0], colors='green', linewidths=3)
    
    ax2.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Boundary')
    ax2.set_xlabel('R')
    ax2.set_ylabel('Z')
    ax2.set_title('psi contours (green = psi=0)')
    ax2.set_aspect('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psi_field_diagnostics.png'), dpi=150)
    plt.close()
    
    print(f"\nPsi statistics:")
    print(f"  min(psi) = {psi_min:.6f}")
    print(f"  max(psi) = {psi_max:.6f}")
    if psi_min < 0 < psi_max:
        print("  [OK] Zero crossing exists")
    else:
        print("  [WARNING] No zero crossing!")
    
    # ============================================================
    # Plot 2: Midplane cut (Z=0)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    z_idx = np.argmin(np.abs(z))
    psi_midplane = psi_pred[:, z_idx]
    
    ax.plot(r, psi_midplane, 'b-', lw=2, label='psi(R, Z=0)')
    ax.axhline(y=0, color='r', linestyle='--', lw=1, label='psi=0')
    ax.axvline(x=1-eps, color='g', linestyle=':', lw=1, label=f'R_inner={1-eps:.3f}')
    ax.axvline(x=1+eps, color='g', linestyle=':', lw=1, label=f'R_outer={1+eps:.3f}')
    
    ax.set_xlabel('R')
    ax.set_ylabel('psi')
    ax.set_title(f'Midplane cut (Z=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psi_midplane_cut.png'), dpi=150)
    plt.close()
    
    # ============================================================
    # Plot 3: CORRECTED Pressure profile analysis
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    psi_range = np.linspace(-0.5, 0.1, 500)
    
    # p(psi) - CORRECTED
    ax1 = axes[0]
    p_vals = p_of_psi_pedestal_numpy(psi_range, p_edge, p_core)
    ax1.plot(psi_range, p_vals, 'b-', lw=2)
    ax1.axvline(x=0, color='r', linestyle='--', lw=1, label='psi=0 (boundary)')
    ax1.axvline(x=PSI_PED, color='orange', linestyle='--', lw=1, 
                label=f'psi_ped={PSI_PED}')
    ax1.axhline(y=p_core, color='green', linestyle=':', alpha=0.7, label=f'p_core={p_core}')
    ax1.axhline(y=p_edge, color='purple', linestyle=':', alpha=0.7, label=f'p_edge={p_edge}')
    ax1.set_xlabel('psi')
    ax1.set_ylabel('p(psi)')
    ax1.set_title(f'CORRECTED Pressure profile\np_edge={p_edge:.2f}, p_core={p_core:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate core and edge
    ax1.annotate('Core\n(HIGH p)', xy=(-0.35, p_core*0.95), fontsize=10, ha='center', color='green')
    ax1.annotate('Edge\n(LOW p)', xy=(0.05, p_edge*1.5), fontsize=10, ha='center', color='purple')
    
    # dp/dpsi - CORRECTED
    ax2 = axes[1]
    dp_vals = dp_dpsi_pedestal_numpy(psi_range, p_edge, p_core)
    ax2.plot(psi_range, dp_vals, 'r-', lw=2)
    ax2.axvline(x=0, color='r', linestyle='--', lw=1, label='psi=0 (boundary)')
    ax2.axvline(x=PSI_PED, color='orange', linestyle='--', lw=1,
                label=f'psi_ped={PSI_PED}')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('psi')
    ax2.set_ylabel('dp/dpsi')
    ax2.set_title(f'Pressure gradient\nMax |dp/dpsi| = {np.max(np.abs(dp_vals)):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotate sign
    ax2.annotate('dp/dpsi < 0\n(p decreases toward edge)', 
                xy=(PSI_PED, dp_vals.min()*0.5), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pressure_profile_analysis.png'), dpi=150)
    plt.close()
    
    print(f"\nPressure profile (CORRECTED formula):")
    print(f"  p_edge = {p_edge}")
    print(f"  p_core = {p_core}")
    print(f"  PSI_PED = {PSI_PED}")
    print(f"  WIDTH = {WIDTH}")
    print(f"  Max |dp/dpsi| = {np.max(np.abs(dp_vals)):.2f}")
    
    # Verify pressure values
    p_at_core = p_of_psi_pedestal_numpy(-0.4, p_edge, p_core)
    p_at_edge = p_of_psi_pedestal_numpy(0.0, p_edge, p_core)
    print(f"\nVerification:")
    print(f"  p(psi=-0.4) [core] = {p_at_core:.4f} (expected ~ {p_core})")
    print(f"  p(psi=0.0) [edge] = {p_at_edge:.4f} (expected ~ {p_edge})")
    
    if p_at_core > p_at_edge:
        print("  [OK] Pressure profile orientation correct (core > edge)")
    else:
        print("  [ERROR] Pressure profile inverted!")
    
    # ============================================================
    # Plot 4: Compare with what the psi field actually produces
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get psi values from the model at midplane
    psi_from_model = psi_midplane
    
    # Compute what pressure would be at those psi values
    p_from_model_psi = p_of_psi_pedestal_numpy(psi_from_model, p_edge, p_core)
    
    ax.plot(r, p_from_model_psi, 'b-', lw=2, label='p(psi_model(R, Z=0))')
    ax.axvline(x=1-eps, color='g', linestyle=':', lw=1, label='Inner boundary')
    ax.axvline(x=1+eps, color='g', linestyle=':', lw=1, label='Outer boundary')
    ax.axvline(x=1.0, color='orange', linestyle='--', lw=1, label='Magnetic axis (R=1)')
    
    ax.set_xlabel('R')
    ax.set_ylabel('p')
    ax.set_title('Pressure profile along midplane (from model psi)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pressure_along_midplane.png'), dpi=150)
    plt.close()
    
    print(f"\nAll plots saved to: {save_dir}")
    
    return psi_pred, RR, ZZ


def load_model_for_pedestal(checkpoint_path, num_alpha=2):
    """Load a trained pedestal pPINN model."""
    
    DEPTH = 4
    BREADTH = 40
    AF = "swish"
    INPUT_DIM = 2 + num_alpha + 3
    
    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    
    # Minimal geometry for model compilation
    xmin = np.array([0.5, -1.5, 0.05, 0.3, 0.12, 1.25, -0.5])
    xmax = np.array([1.5, 1.5, 0.2, 1.0, 0.52, 2.75, 0.5])
    geom = dde.geometry.Hypercube(xmin, xmax)
    
    def dummy_pde(x, u):
        return dde.grad.hessian(u, x, i=0, j=0)
    
    data = dde.data.PDE(geom, dummy_pde, [], num_domain=1, num_boundary=0, num_test=1)
    
    model = dde.Model(data=data, net=net)
    model.compile("adam", lr=1e-3)
    model.restore(checkpoint_path, verbose=1)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Diagnose CORRECTED pedestal pPINN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eps', type=float, default=0.32,
                        help='Inverse aspect ratio')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Elongation')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Triangularity')
    parser.add_argument('--p_edge', type=float, default=0.1,
                        help='Edge pressure (alpha0)')
    parser.add_argument('--p_core', type=float, default=0.6,
                        help='Core pressure (alpha1)')
    parser.add_argument('--output_dir', type=str, default='./pedestal_diagnostics_corrected',
                        help='Output directory for plots')
    parser.add_argument('--grid_size', type=int, default=200,
                        help='Grid resolution')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CORRECTED Pedestal Model Diagnostics")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Shape: eps={args.eps}, kappa={args.kappa}, delta={args.delta}")
    print(f"Pressure: p_edge={args.p_edge}, p_core={args.p_core}")
    
    print("\nLoading model...")
    model = load_model_for_pedestal(args.model_path)
    
    print("\nCreating diagnostic plots...")
    alpha = [args.p_edge, args.p_core]
    
    create_diagnostic_plots(
        model=model,
        eps=args.eps,
        kappa=args.kappa,
        delta=args.delta,
        alpha=alpha,
        save_dir=args.output_dir,
        grid_size=args.grid_size
    )


if __name__ == "__main__":
    main()