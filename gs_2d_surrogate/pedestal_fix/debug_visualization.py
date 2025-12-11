"""
Debug visualization functions to add to shape_optimization_general.py

Copy this entire block and paste it AFTER the imports in shape_optimization_general.py
(around line 30, after `from matplotlib.path import Path`)

Then, in the objective functions (make_objective, make_beta_p_volume_objective),
you can call `debug_psi_field(...)` when the contour extraction fails.
"""

import os
import time as time_module

# ============================================================
# DEBUG: Visualization for failed contour extraction
# ============================================================

DEBUG_PLOT_DIR = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/debug_plots"


def debug_psi_field(R, Z, psi_pred_grid, eps, kappa, delta, alpha_vals=None,
                    iteration=None, save=True):
    """
    Create a diagnostic plot when ψ=0 contour extraction fails.
    
    Call this inside your objective function when `c.collections[0].get_paths()` 
    returns empty.
    
    Parameters:
    -----------
    R, Z : 2D arrays
        Meshgrid of R and Z coordinates
    psi_pred_grid : 2D array
        Predicted ψ values on the grid
    eps, kappa, delta : float
        Shape parameters
    alpha_vals : array, optional
        Pressure profile parameters
    iteration : int, optional
        Optimization iteration number
    save : bool
        Whether to save the plot
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(DEBUG_PLOT_DIR, exist_ok=True)
    
    psi_min = psi_pred_grid.min()
    psi_max = psi_pred_grid.max()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Heatmap
    ax1 = axes[0]
    pcm = ax1.pcolormesh(R, Z, psi_pred_grid, shading='auto', cmap='RdBu_r')
    plt.colorbar(pcm, ax=ax1, label='ψ')
    
    # Analytic boundary
    tau = np.linspace(0, 2*np.pi, 200)
    R_bnd = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    Z_bnd = eps * kappa * np.sin(tau)
    ax1.plot(R_bnd, Z_bnd, 'k--', lw=2)
    
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title(f'ψ field\neps={eps:.3f}, κ={kappa:.3f}, δ={delta:.3f}')
    ax1.set_aspect('equal')
    
    # 2. Contours (including ψ=0 attempt)
    ax2 = axes[1]
    
    if psi_min < 0 < psi_max:
        levels = np.sort(np.unique(np.append(
            np.linspace(psi_min, psi_max, 15), 0
        )))
    else:
        levels = np.linspace(psi_min, psi_max, 15)
    
    cs = ax2.contour(R, Z, psi_pred_grid, levels=levels, cmap='coolwarm')
    ax2.clabel(cs, inline=True, fontsize=7, fmt='%.3f')
    
    # Try to highlight ψ=0
    if psi_min < 0 < psi_max:
        try:
            cs0 = ax2.contour(R, Z, psi_pred_grid, levels=[0], colors='green', linewidths=3)
        except:
            pass
    
    ax2.plot(R_bnd, Z_bnd, 'k--', lw=2)
    ax2.set_xlabel('R')
    ax2.set_ylabel('Z')
    ax2.set_title(f'Contours\nψ ∈ [{psi_min:.4f}, {psi_max:.4f}]')
    ax2.set_aspect('equal')
    
    # 3. Midplane cut
    ax3 = axes[2]
    z_idx = R.shape[1] // 2  # Middle Z index
    r_vals = R[:, z_idx]
    psi_mid = psi_pred_grid[:, z_idx]
    
    ax3.plot(r_vals, psi_mid, 'b-', lw=2)
    ax3.axhline(y=0, color='r', linestyle='--', label='ψ=0')
    ax3.axvline(x=1-eps, color='g', linestyle=':', alpha=0.7)
    ax3.axvline(x=1+eps, color='g', linestyle=':', alpha=0.7)
    ax3.set_xlabel('R')
    ax3.set_ylabel('ψ')
    ax3.set_title('Midplane cut (Z≈0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add alpha info if available
    alpha_str = ""
    if alpha_vals is not None and len(alpha_vals) > 0:
        alpha_str = f"\nα={alpha_vals}"
    
    fig.suptitle(f"DEBUG: ψ=0 contour extraction failed{alpha_str}", fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save:
        timestamp = time_module.strftime("%H%M%S")
        iter_str = f"_iter{iteration}" if iteration is not None else ""
        filename = f"debug_psi{iter_str}_{timestamp}.png"
        filepath = os.path.join(DEBUG_PLOT_DIR, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"DEBUG: Saved diagnostic plot to {filepath}")
    
    plt.close()
    
    # Also print diagnostic info
    print(f"\n{'='*50}")
    print("DEBUG: ψ=0 CONTOUR EXTRACTION FAILED")
    print(f"{'='*50}")
    print(f"ψ range: [{psi_min:.6f}, {psi_max:.6f}]")
    print(f"Shape: eps={eps:.4f}, kappa={kappa:.4f}, delta={delta:.4f}")
    if alpha_vals is not None:
        print(f"Alpha: {alpha_vals}")
    
    if psi_min > 0:
        print("\n⚠️  PROBLEM: ψ is entirely POSITIVE")
        print("   The boundary condition ψ=0 is not satisfied.")
    elif psi_max < 0:
        print("\n⚠️  PROBLEM: ψ is entirely NEGATIVE")
        print("   The ψ=0 level set doesn't exist in the domain.")
    else:
        print("\n⚠️  ψ has zero crossing but contour extraction failed")
        print("   Try increasing grid resolution or zoom factor.")
    print(f"{'='*50}\n")


# ============================================================
# How to integrate into your objective functions
# ============================================================

"""
In make_beta_p_volume_objective() and make_objective(), modify the contour
extraction section like this:

ORIGINAL CODE:
    c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
    
    if use_fallback:
        if c.collections and c.collections[0].get_paths():
            vertices = c.collections[0].get_paths()[0].vertices
        else:
            vertices = vertices_analytic
    else:
        vertices = c.collections[0].get_paths()[0].vertices
    plt.close(c.figure)

MODIFIED CODE (with debug):
    c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
    
    contour_found = (c.collections and c.collections[0].get_paths() 
                     and len(c.collections[0].get_paths()[0].vertices) > 3)
    
    if not contour_found:
        # DEBUG: Save diagnostic plot
        debug_psi_field(R, Z, psi_pred_grid, eps, kappa, delta, 
                       alpha_vals=alpha_vals if 'alpha_vals' in dir() else None,
                       iteration=len(metrics["obj"]))
    
    if use_fallback:
        if contour_found:
            vertices = c.collections[0].get_paths()[0].vertices
        else:
            vertices = vertices_analytic
    else:
        if contour_found:
            vertices = c.collections[0].get_paths()[0].vertices
        else:
            plt.close(c.figure)
            raise ValueError(
                f"No ψ=0 contour found! ψ range: [{psi_pred_grid.min():.4f}, {psi_pred_grid.max():.4f}]"
            )
    
    plt.close(c.figure)
"""
