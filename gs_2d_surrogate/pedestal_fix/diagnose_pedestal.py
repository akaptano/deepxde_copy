"""
Diagnostic script for pedestal profile pPINN models.

This script helps identify why no ψ=0 contour is found during shape optimization.

Usage:
    python pedestal_fix/diagnose_pedestal.py --model_path /path/to/checkpoint.ckpt

It will generate several diagnostic plots:
1. ψ field heatmap with contour lines
2. ψ values along the midplane (Z=0)
3. Histogram of ψ values
4. Pressure profile p(ψ) for the given alpha parameters
5. dp/dψ profile (the source term driver)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
# Pedestal profile parameters (must match profile.py)
# ============================================================
PSI_PED_CONST = -0.1
WIDTH_CONST = 0.05
STEEPNESS_CONST = 4.0

# Shape parameter ranges
eps_deviation = 0.2
kappa_deviation = 0.75
delta_deviation = 0.5
eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)
delta0 = (0 - delta_deviation, 0 + delta_deviation)


def p_of_psi_pedestal_numpy(psi, p_ped, p_core):
    """NumPy version of the pedestal pressure profile."""
    arg = STEEPNESS_CONST * (psi - PSI_PED_CONST) / WIDTH_CONST
    return p_core + (p_ped - p_core) * 0.5 * (1 - np.tanh(arg))


def dp_dpsi_pedestal_numpy(psi, p_ped, p_core):
    """NumPy version of dp/dpsi for pedestal profile."""
    arg = STEEPNESS_CONST * (psi - PSI_PED_CONST) / WIDTH_CONST
    sech2 = 1.0 / np.cosh(arg)**2
    return -(p_ped - p_core) * 0.5 * (STEEPNESS_CONST / WIDTH_CONST) * sech2


def create_diagnostic_plots(model, eps, kappa, delta, alpha, save_dir, grid_size=200, zoom=1.5):
    """
    Create comprehensive diagnostic plots for a pedestal pPINN model.
    
    Parameters:
    -----------
    model : dde.Model
        The trained DeepXDE model
    eps, kappa, delta : float
        Shape parameters
    alpha : array-like
        Pressure profile parameters [p_ped, p_core]
    save_dir : str
        Directory to save plots
    grid_size : int
        Grid resolution for plotting
    zoom : float
        Zoom factor for the grid extent
    """
    os.makedirs(save_dir, exist_ok=True)
    
    p_ped, p_core = alpha[0], alpha[1]
    
    # Create evaluation grid
    inner_point = 1 - 1.1 * eps * (1 + zoom)
    outer_point = 1 + 1.1 * eps * (1 + zoom)
    low_point = -1.1 * kappa * eps * (1 + zoom)
    high_point = 1.1 * kappa * eps * (1 + zoom)
    
    r = np.linspace(inner_point, outer_point, grid_size)
    z = np.linspace(low_point, high_point, grid_size)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    
    # Build input tensor: [R, Z, alpha0, alpha1, eps, kappa, delta]
    N = grid_size * grid_size
    X_in = np.zeros((N, 7))
    X_in[:, 0] = RR.ravel()
    X_in[:, 1] = ZZ.ravel()
    X_in[:, 2] = p_ped
    X_in[:, 3] = p_core
    X_in[:, 4] = eps
    X_in[:, 5] = kappa
    X_in[:, 6] = delta
    
    # Predict ψ
    psi_pred = model.predict(X_in).reshape(grid_size, grid_size)
    
    # Analytic boundary for reference
    tau = np.linspace(0, 2*np.pi, 400)
    R_bnd = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    Z_bnd = eps * kappa * np.sin(tau)
    
    # ============================================================
    # Plot 1: ψ field heatmap with contours
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Heatmap
    ax1 = axes[0]
    pcm = ax1.pcolormesh(RR, ZZ, psi_pred, shading='auto', cmap='RdBu_r')
    ax1.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Analytic boundary')
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title(f'ψ field\neps={eps:.3f}, kappa={kappa:.3f}, delta={delta:.3f}\nalpha=[{p_ped:.2f}, {p_core:.2f}]')
    ax1.set_aspect('equal')
    ax1.legend()
    plt.colorbar(pcm, ax=ax1, label='ψ')
    
    # Contours
    ax2 = axes[1]
    psi_min, psi_max = psi_pred.min(), psi_pred.max()
    print(f"\nψ statistics:")
    print(f"  min(ψ) = {psi_min:.6f}")
    print(f"  max(ψ) = {psi_max:.6f}")
    print(f"  mean(ψ) = {psi_pred.mean():.6f}")
    
    # Check if zero is in the range
    if psi_min < 0 < psi_max:
        print("  ✓ Zero crossing exists in ψ field")
        levels = np.linspace(psi_min, psi_max, 20)
        # Ensure 0 is in the levels
        if 0 not in levels:
            levels = np.sort(np.append(levels, 0))
    else:
        print(f"  ✗ NO zero crossing! ψ range is [{psi_min:.4f}, {psi_max:.4f}]")
        levels = np.linspace(psi_min, psi_max, 20)
    
    cs = ax2.contour(RR, ZZ, psi_pred, levels=levels, cmap='coolwarm')
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    
    # Try to plot ψ=0 contour specifically
    try:
        cs0 = ax2.contour(RR, ZZ, psi_pred, levels=[0.0], colors='green', linewidths=3)
        if cs0.collections and cs0.collections[0].get_paths():
            print("  ✓ ψ=0 contour extracted successfully")
            n_paths = len(cs0.collections[0].get_paths())
            print(f"    Found {n_paths} contour path(s)")
        else:
            print("  ✗ ψ=0 contour extraction failed (no paths)")
    except Exception as e:
        print(f"  ✗ Error extracting ψ=0 contour: {e}")
    
    ax2.plot(R_bnd, Z_bnd, 'k--', lw=2, label='Analytic boundary')
    ax2.set_xlabel('R')
    ax2.set_ylabel('Z')
    ax2.set_title('ψ contours (green = ψ=0)')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Histogram
    ax3 = axes[2]
    ax3.hist(psi_pred.ravel(), bins=100, density=True, alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='ψ=0')
    ax3.axvline(x=psi_min, color='b', linestyle=':', lw=1, label=f'min={psi_min:.4f}')
    ax3.axvline(x=psi_max, color='g', linestyle=':', lw=1, label=f'max={psi_max:.4f}')
    ax3.set_xlabel('ψ')
    ax3.set_ylabel('Density')
    ax3.set_title('ψ value distribution')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psi_field_diagnostics.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {save_dir}/psi_field_diagnostics.png")
    
    # ============================================================
    # Plot 2: Midplane cut (Z=0)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find Z=0 index
    z_idx = np.argmin(np.abs(z))
    psi_midplane = psi_pred[:, z_idx]
    
    ax.plot(r, psi_midplane, 'b-', lw=2, label='ψ(R, Z=0)')
    ax.axhline(y=0, color='r', linestyle='--', lw=1, label='ψ=0')
    
    # Mark the expected boundary locations
    R_inner = 1 - eps
    R_outer = 1 + eps
    ax.axvline(x=R_inner, color='g', linestyle=':', lw=1, label=f'R_inner={R_inner:.3f}')
    ax.axvline(x=R_outer, color='g', linestyle=':', lw=1, label=f'R_outer={R_outer:.3f}')
    
    ax.set_xlabel('R')
    ax.set_ylabel('ψ')
    ax.set_title(f'Midplane cut (Z=0)\nalpha=[{p_ped:.2f}, {p_core:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psi_midplane_cut.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/psi_midplane_cut.png")
    
    # ============================================================
    # Plot 3: Pressure profile analysis
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    psi_range = np.linspace(-0.5, 0.1, 500)
    
    # p(ψ)
    ax1 = axes[0]
    p_vals = p_of_psi_pedestal_numpy(psi_range, p_ped, p_core)
    ax1.plot(psi_range, p_vals, 'b-', lw=2)
    ax1.axvline(x=0, color='r', linestyle='--', lw=1, label='ψ=0 (boundary)')
    ax1.axvline(x=PSI_PED_CONST, color='orange', linestyle='--', lw=1, 
                label=f'ψ_ped={PSI_PED_CONST}')
    ax1.set_xlabel('ψ')
    ax1.set_ylabel('p(ψ)')
    ax1.set_title(f'Pressure profile\np_ped={p_ped:.2f}, p_core={p_core:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # dp/dψ
    ax2 = axes[1]
    dp_vals = dp_dpsi_pedestal_numpy(psi_range, p_ped, p_core)
    ax2.plot(psi_range, dp_vals, 'r-', lw=2)
    ax2.axvline(x=0, color='r', linestyle='--', lw=1, label='ψ=0 (boundary)')
    ax2.axvline(x=PSI_PED_CONST, color='orange', linestyle='--', lw=1,
                label=f'ψ_ped={PSI_PED_CONST}')
    ax2.set_xlabel('ψ')
    ax2.set_ylabel('dp/dψ')
    ax2.set_title(f'Pressure gradient (GS source term)\nMax |dp/dψ| = {np.max(np.abs(dp_vals)):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pressure_profile_analysis.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/pressure_profile_analysis.png")
    
    # ============================================================
    # Plot 4: 3D surface plot of ψ
    # ============================================================
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for cleaner 3D plot
    stride = max(1, grid_size // 50)
    surf = ax.plot_surface(RR[::stride, ::stride], ZZ[::stride, ::stride], 
                           psi_pred[::stride, ::stride],
                           cmap='RdBu_r', alpha=0.8)
    
    # Add ψ=0 plane
    if psi_min < 0 < psi_max:
        ax.plot_surface(RR[::stride, ::stride], ZZ[::stride, ::stride],
                       np.zeros_like(RR[::stride, ::stride]),
                       alpha=0.3, color='green')
    
    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.set_zlabel('ψ')
    ax.set_title('3D ψ field (green plane = ψ=0)')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='ψ')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psi_3d_surface.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/psi_3d_surface.png")
    
    # ============================================================
    # Print summary
    # ============================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Shape parameters: eps={eps}, kappa={kappa}, delta={delta}")
    print(f"Pressure params: p_ped={p_ped}, p_core={p_core}")
    print(f"Pedestal constants: PSI_PED={PSI_PED_CONST}, WIDTH={WIDTH_CONST}, STEEP={STEEPNESS_CONST}")
    print(f"\nψ range: [{psi_min:.6f}, {psi_max:.6f}]")
    
    if psi_min > 0:
        print("\n⚠️  PROBLEM: ψ is entirely positive!")
        print("   The boundary condition ψ=0 is not being satisfied.")
        print("   Possible causes:")
        print("   1. Network not trained long enough")
        print("   2. Pressure profile parameters too extreme")
        print("   3. Boundary condition weight too low")
    elif psi_max < 0:
        print("\n⚠️  PROBLEM: ψ is entirely negative!")
        print("   The ψ=0 level set doesn't exist in the domain.")
    else:
        print("\n✓ ψ field has proper zero crossing")
    
    # Check pressure gradient magnitude
    max_dp = np.max(np.abs(dp_dpsi_pedestal_numpy(np.linspace(-0.3, 0, 100), p_ped, p_core)))
    print(f"\nMax |dp/dψ| near boundary: {max_dp:.2f}")
    if max_dp > 100:
        print("⚠️  WARNING: Very large pressure gradient!")
        print("   This can cause numerical instability.")
        print("   Consider reducing (p_core - p_ped) or increasing WIDTH_CONST")
    
    return psi_pred, RR, ZZ


def load_model_for_pedestal(checkpoint_path, num_alpha=2):
    """Load a trained pedestal pPINN model."""
    
    # Model architecture (must match training)
    DEPTH = 4
    BREADTH = 40
    AF = "swish"
    INPUT_DIM = 2 + num_alpha + 3  # R, Z, alpha0, alpha1, eps, kappa, delta
    
    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    
    # Create minimal geometry and data for model compilation
    # This is just for DeepXDE to work - we won't use it for anything
    alpha_ranges = [
        np.linspace(0.1, 1.0, 2),
        np.linspace(1.5, 5.0, 2),
    ]
    
    # Import the geometry class
    sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')
    
    # Minimal PDE for compilation
    def dummy_pde(x, u):
        return dde.grad.hessian(u, x, i=0, j=0)
    
    # Create a simple hypercube geometry for dummy data
    xmin = np.array([0.5, -1.5, 0.1, 1.5, 0.12, 1.25, -0.5])
    xmax = np.array([1.5, 1.5, 1.0, 5.0, 0.52, 2.75, 0.5])
    geom = dde.geometry.Hypercube(xmin, xmax)
    
    data = dde.data.PDE(
        geom,
        dummy_pde,
        [],
        num_domain=1,
        num_boundary=0,
        num_test=1
    )
    
    model = dde.Model(data=data, net=net)
    model.compile("adam", lr=1e-3)
    model.restore(checkpoint_path, verbose=1)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Diagnose pedestal pPINN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eps', type=float, default=0.32,
                        help='Inverse aspect ratio')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Elongation')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Triangularity')
    parser.add_argument('--p_ped', type=float, default=0.55,
                        help='Pedestal pressure (alpha0)')
    parser.add_argument('--p_core', type=float, default=3.25,
                        help='Core pressure (alpha1)')
    parser.add_argument('--output_dir', type=str, default='./pedestal_diagnostics',
                        help='Output directory for plots')
    parser.add_argument('--grid_size', type=int, default=200,
                        help='Grid resolution')
    parser.add_argument('--zoom', type=float, default=1.5,
                        help='Grid zoom factor')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_model_for_pedestal(args.model_path)
    
    print("\nCreating diagnostic plots...")
    alpha = [args.p_ped, args.p_core]
    
    create_diagnostic_plots(
        model=model,
        eps=args.eps,
        kappa=args.kappa,
        delta=args.delta,
        alpha=alpha,
        save_dir=args.output_dir,
        grid_size=args.grid_size,
        zoom=args.zoom
    )
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
