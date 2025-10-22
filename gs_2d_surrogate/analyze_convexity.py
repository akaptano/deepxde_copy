"""
Analyze the convexity of the optimization problem by plotting f(epsilon, kappa)
with fixed delta and A values.
"""

import time
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Setup paths
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)
sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import deepxde as dde
dde.config.set_default_float("float64")

from utils.utils import *
from utils.gs_solovev_sol import GS_Linear

# Import necessary functions from shape_optimization
from shape_optimization import (
    predict_psi, area, compute_beta_p, compute_beta_p_and_qstar,
    Cp, qstar_integral, pde_solovev
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DELTA_FIXED = 0.33
A_FIXED = -0.155

# Grid resolution for the landscape plot
N_EPS = 100      # Number of epsilon points
N_KAPPA = 100    # Number of kappa points

# Parameter ranges (from original code)
eps_deviation = 0.2
kappa_deviation = 0.75
eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)

# Target values for objective function
TARGET_BETA_P = 1.2
TARGET_VOLUME = 1.2
TARGET_QSTAR = 1.57

# Weights for multi-objective
LAMBDA_BETA_P = 1.0
LAMBDA_VOLUME = 10.0
LAMBDA_QSTAR = 3.0

# Grid and zoom settings
ZOOM = 1.2
N_BOUNDARY = 400

# Model checkpoint path
CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09152025_2103_parametrized/ITER-16001.ckpt"

# Objective type: "beta_p", "volume", or "beta_p_and_qstar"
OBJECTIVE_TYPE = "beta_p_and_qstar"

# ----------------------------------------------------------------------------
# Load the pretrained model
# ----------------------------------------------------------------------------

print("Loading pretrained model...")

delta_deviation = 0.5
delta0 = (0 - delta_deviation, 0 + delta_deviation)
Amax = 0.2

DEPTH = 4
BREADTH = 40
LR = 2e-2
AF = "swish"

net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

# Create dummy data for inference
spatial_domain = dde.geometry.HyperEllipticalToroid(
    eps0, kappa0, delta0, Amax=Amax
)
data = dde.data.PDE(
    spatial_domain,
    pde_solovev,
    [],
    num_domain=1,
    num_boundary=0,
    num_test=1
)
model = dde.Model(data=data, net=net)
model.compile("adam", lr=LR, loss_weights=[1, 100])
model.restore(CHECKPOINT_PATH, verbose=1)

print("Model loaded successfully!")

# ----------------------------------------------------------------------------
# Define objective function
# ----------------------------------------------------------------------------

tau = np.linspace(0.0, 2 * np.pi, N_BOUNDARY, endpoint=False)

def evaluate_objective(eps, kappa, delta, A):
    try:
        # Get predictions
        R, Z, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat = predict_psi(
            model, eps=eps, kappa=kappa, delta=delta, A=A, 
            return_X=True, plot_psi=False, zoom=ZOOM
        )
        
        # Try to extract ψ = 0 contour
        # Use matplotlib.contour which creates a figure in the background
        fig, ax = plt.subplots()
        c = ax.contour(R, Z, psi_pred_grid, levels=[0.0])
        
        # Extract paths - compatible with different matplotlib versions
        try:
            # Try newer API first
            paths = c.get_paths()
        except AttributeError:
            # Fall back to older API
            if hasattr(c, 'collections') and c.collections:
                paths = c.collections[0].get_paths()
            else:
                plt.close(fig)
                return np.inf
        
        if not paths or len(paths) == 0:
            plt.close(fig)
            return np.inf
            
        vertices = paths[0].vertices
        plt.close(fig)
        
        # Compute volume
        volume = area(vertices)
        
        # Compute objective based on type
        if OBJECTIVE_TYPE == "volume":
            obj = (volume - TARGET_VOLUME) ** 2
            
        elif OBJECTIVE_TYPE == "beta_p":
            beta_p = compute_beta_p(model, X_in, psi_pred_flat, vertices, A=A)
            obj = (beta_p - TARGET_BETA_P) ** 2 + LAMBDA_VOLUME * (volume - TARGET_VOLUME) ** 2
            
        elif OBJECTIVE_TYPE == "beta_p_and_qstar":
            beta_p, qstar = compute_beta_p_and_qstar(model, X_in, psi_pred_flat, vertices, A=A)
            obj = (LAMBDA_BETA_P * (beta_p - TARGET_BETA_P) ** 2 + 
                   LAMBDA_VOLUME * (volume - TARGET_VOLUME) ** 2 + 
                   LAMBDA_QSTAR * (qstar - TARGET_QSTAR) ** 2)
        else:
            raise ValueError(f"Unknown objective type: {OBJECTIVE_TYPE}")
        
        return float(obj)
        
    except Exception as e:
        print(f"Error at eps={eps:.3f}, kappa={kappa:.3f}: {e}")
        return np.inf

# ----------------------------------------------------------------------------
# Create parameter grid and evaluate objective
# ----------------------------------------------------------------------------

print(f"\nCreating {N_EPS}x{N_KAPPA} grid for epsilon and kappa...")
print(f"Fixed: delta={DELTA_FIXED}, A={A_FIXED}")
print(f"Objective type: {OBJECTIVE_TYPE}\n")

eps_grid = np.linspace(eps0[0], eps0[1], N_EPS)
kappa_grid = np.linspace(kappa0[0], kappa0[1], N_KAPPA)

EPS, KAPPA = np.meshgrid(eps_grid, kappa_grid)
OBJ = np.zeros_like(EPS)

print("Evaluating objective function at grid points...")
start_time = time.time()

for i in range(N_EPS):
    for j in range(N_KAPPA):
        eps_val = eps_grid[i]
        kappa_val = kappa_grid[j]
        
        obj_val = evaluate_objective(eps_val, kappa_val, DELTA_FIXED, A_FIXED)
        OBJ[j, i] = obj_val
        
        if (i * N_KAPPA + j + 1) % 10 == 0:
            print(f"Progress: {i * N_KAPPA + j + 1}/{N_EPS * N_KAPPA} points completed")

elapsed_time = time.time() - start_time
print(f"\nEvaluation completed in {elapsed_time:.1f} seconds")

# ----------------------------------------------------------------------------
# Plot the results
# ----------------------------------------------------------------------------

print("\nCreating plots...")

# Create output directory
output_dir = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/convexity_analysis/{OBJECTIVE_TYPE}_N{N_EPS}/"
os.makedirs(output_dir, exist_ok=True)

# Filter finite values for plotting
finite_mask = np.isfinite(OBJ)
finite_obj = OBJ[finite_mask]

if len(finite_obj) == 0:
    print("ERROR: All objective values are infinite! Cannot create plots.")
    print("This suggests the contour extraction failed for all parameter combinations.")
    exit(1)

print(f"Valid objective values: {len(finite_obj)} out of {OBJ.size} ({100*len(finite_obj)/OBJ.size:.1f}%)")

# 1. Contour plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create levels for contour plot
obj_min_val = finite_obj.min()
obj_max_val = finite_obj.max()

# Use log scale if values span more than 2 orders of magnitude
if obj_max_val / obj_min_val > 100:
    levels = np.logspace(np.log10(obj_min_val), np.log10(obj_max_val), 20)
    norm = plt.matplotlib.colors.LogNorm(vmin=obj_min_val, vmax=obj_max_val)
else:
    levels = np.linspace(obj_min_val, obj_max_val, 20)
    norm = None

# Replace inf with a large value for plotting
OBJ_plot = np.copy(OBJ)
OBJ_plot[~finite_mask] = obj_max_val * 10

contour = ax.contourf(EPS, KAPPA, OBJ_plot, levels=levels, cmap='viridis', norm=norm, extend='max')
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Objective value (log scale)', fontsize=12)

# Add contour lines
contour_lines = ax.contour(EPS, KAPPA, OBJ, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

# Mark the minimum
min_idx = np.unravel_index(np.argmin(OBJ), OBJ.shape)
eps_min = EPS[min_idx]
kappa_min = KAPPA[min_idx]
obj_min = OBJ[min_idx]
ax.plot(eps_min, kappa_min, 'r*', markersize=20, label=f'Min: ({eps_min:.3f}, {kappa_min:.3f})')

ax.set_xlabel('ε (epsilon)', fontsize=14)
ax.set_ylabel('κ (kappa)', fontsize=14)
ax.set_title(f'Objective Landscape: f(ε, κ)\nδ={DELTA_FIXED}, A={A_FIXED}\nObjective={OBJECTIVE_TYPE}', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'objective_contour.png', dpi=150, bbox_inches='tight')
print(f"Saved contour plot to {output_dir}objective_contour.png")
plt.close()

# 2. 3D surface plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Use log scale for z-axis if values span many orders of magnitude
# Cap infinite values for visualization
OBJ_3d = np.copy(OBJ)
OBJ_3d[~finite_mask] = obj_max_val * 10  # Replace inf with large value

# Apply log transform for better visualization
OBJ_3d_plot = np.log10(OBJ_3d + 1e-10)  # Add small value to avoid log(0)

surf = ax.plot_surface(EPS, KAPPA, OBJ_3d_plot, cmap='viridis', 
                       edgecolor='none', alpha=0.8, antialiased=True)

# Mark the minimum
ax.plot([eps_min], [kappa_min], [np.log10(obj_min + 1e-10)], 
        'r*', markersize=15, label=f'Min: ({eps_min:.3f}, {kappa_min:.3f})')

ax.set_xlabel('ε (epsilon)', fontsize=12)
ax.set_ylabel('κ (kappa)', fontsize=12)
ax.set_zlabel('log₁₀(Objective)', fontsize=12)
ax.set_title(f'3D Objective Landscape\nδ={DELTA_FIXED}, A={A_FIXED}', fontsize=14)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(output_dir + 'objective_3d.png', dpi=150, bbox_inches='tight')
print(f"Saved 3D plot to {output_dir}objective_3d.png")
plt.close()

# 3. Cross-sections at minimum
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cross-section along epsilon (fixed kappa at minimum)
kappa_idx = min_idx[0]
ax1.plot(eps_grid, OBJ[kappa_idx, :], 'b-', linewidth=2)
ax1.axvline(eps_min, color='r', linestyle='--', label=f'Min at ε={eps_min:.3f}')
ax1.set_xlabel('ε (epsilon)', fontsize=12)
ax1.set_ylabel('Objective value', fontsize=12)
ax1.set_title(f'Cross-section at κ={kappa_min:.3f}', fontsize=12)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Cross-section along kappa (fixed epsilon at minimum)
eps_idx = min_idx[1]
ax2.plot(kappa_grid, OBJ[:, eps_idx], 'g-', linewidth=2)
ax2.axvline(kappa_min, color='r', linestyle='--', label=f'Min at κ={kappa_min:.3f}')
ax2.set_xlabel('κ (kappa)', fontsize=12)
ax2.set_ylabel('Objective value', fontsize=12)
ax2.set_title(f'Cross-section at ε={eps_min:.3f}', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(output_dir + 'objective_crosssections.png', dpi=150, bbox_inches='tight')
print(f"Saved cross-section plot to {output_dir}objective_crosssections.png")
plt.close()

# ----------------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------------

print("\n" + "="*60)
print("CONVEXITY ANALYSIS SUMMARY")
print("="*60)
print(f"Fixed parameters: δ={DELTA_FIXED}, A={A_FIXED}")
print(f"Objective type: {OBJECTIVE_TYPE}")
print(f"\nParameter ranges:")
print(f"  ε: [{eps0[0]:.3f}, {eps0[1]:.3f}]")
print(f"  κ: [{kappa0[0]:.3f}, {kappa0[1]:.3f}]")
print(f"\nObjective statistics (finite values only):")
print(f"  Minimum: {obj_min:.6e} at (ε={eps_min:.4f}, κ={kappa_min:.4f})")
print(f"  Maximum: {obj_max_val:.6e}")
print(f"  Mean: {finite_obj.mean():.6e}")
print(f"  Std: {finite_obj.std():.6e}")
print(f"  Valid points: {len(finite_obj)}/{OBJ.size} ({100*len(finite_obj)/OBJ.size:.1f}%)")

# Count local minima (rough estimate)
# A point is a local minimum if it's smaller than all 4 neighbors
local_minima = 0
for i in range(1, N_EPS-1):
    for j in range(1, N_KAPPA-1):
        if (OBJ[j, i] < OBJ[j-1, i] and OBJ[j, i] < OBJ[j+1, i] and
            OBJ[j, i] < OBJ[j, i-1] and OBJ[j, i] < OBJ[j, i+1] and
            OBJ[j, i] < np.inf):
            local_minima += 1

print(f"\nEstimated number of local minima: {local_minima}")
if local_minima > 1:
    print("⚠️  Multiple local minima detected - problem appears NON-CONVEX")
else:
    print("✓ Single minimum detected - problem may be convex in this region")

# Save results to file
results_file = output_dir + 'analysis_results.txt'
with open(results_file, 'w') as f:
    f.write("CONVEXITY ANALYSIS RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Fixed parameters: δ={DELTA_FIXED}, A={A_FIXED}\n")
    f.write(f"Objective type: {OBJECTIVE_TYPE}\n")
    f.write(f"Grid resolution: {N_EPS} x {N_KAPPA}\n\n")
    f.write(f"Minimum objective: {obj_min:.6e}\n")
    f.write(f"  at ε = {eps_min:.6f}\n")
    f.write(f"  at κ = {kappa_min:.6f}\n\n")
    f.write(f"Estimated local minima: {local_minima}\n")
    f.write(f"Evaluation time: {elapsed_time:.1f} seconds\n")

print(f"\nResults saved to {results_file}")
print("="*60)
print("\nAnalysis complete! Check the plots in:")
print(f"  {output_dir}")

