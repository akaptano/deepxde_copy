"""
Benchmark script to measure PINN-based shape optimization speed.

Purpose: Demonstrate the computational advantage of pPINN for rapid equilibrium evaluation.

This script:
1. Benchmarks individual components (model inference, contour extraction, objective evaluation)
2. Compares different profiles and checkpoints
3. Generates timing statistics and comparison plots

Usage:
    python benchmark_optimization_speed.py --profile solovev
    python benchmark_optimization_speed.py --profile pedestal
    python benchmark_optimization_speed.py --all  # Run all benchmarks
"""

import time
import os
import json
import argparse
import numpy as np
from typing import Callable, Sequence, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Setup TensorFlow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)
import deepxde as dde

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt

# Try to import skimage for fast contour extraction
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not available. Using matplotlib for contour extraction.")

dde.config.set_default_float("float64")
tf.keras.backend.set_floatx("float64")

sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')
from utils.gs_solovev_sol import GS_Linear


# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_CONFIGS = {
    "solovev": {
        "path": "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09152025_2103_parametrized/ITER-16001.ckpt",
        "input_dim": 6,
        "uses_A": True,
        "num_alpha": 0,
    },
    "pedestal": {
        "path": "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12112025_0657_pedestal_fixed_6/pedestal_model-16001.ckpt",
        "input_dim": 7,
        "uses_A": False,
        "num_alpha": 2,
    },
}

# ITER-like parameters
eps0 = (0.12, 0.52)
kappa0 = (1.25, 2.75)
delta0 = (-0.5, 0.5)
Amax = 0.2
num_param = 5


# ============================================================================
# Timing utilities
# ============================================================================
@dataclass
class TimingResult:
    """Store timing results for a benchmark."""
    name: str
    n_runs: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    total_time_s: float


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def benchmark_function(func: Callable, n_runs: int = 100, warmup: int = 5, name: str = "function") -> TimingResult:
    """Benchmark a function over multiple runs."""
    # Warmup runs
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    return TimingResult(
        name=name,
        n_runs=n_runs,
        mean_time_ms=float(np.mean(times)),
        std_time_ms=float(np.std(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        total_time_s=float(np.sum(times) / 1000),
    )


# ============================================================================
# Optimized geometric functions (vectorized)
# ============================================================================
def area_vectorized(vs: np.ndarray) -> float:
    """Compute polygon area using shoelace formula (vectorized)."""
    x = vs[:, 0]
    y = vs[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def Cp_vectorized(vs: np.ndarray) -> float:
    """Chord length integral (vectorized)."""
    dx = np.diff(vs[:, 0])
    dy = np.diff(vs[:, 1])
    # Avoid division by zero
    dy_dx = np.where(np.abs(dx) > 1e-12, dy / dx, 0.0)
    return float(np.sum(np.sqrt(1.0 + dy_dx ** 2) * np.abs(dx)))


def qstar_integral_vectorized(vs: np.ndarray) -> float:
    """q* integral (vectorized)."""
    x = vs[1:, 0]
    dy = np.diff(vs[:, 1])
    M = -1.0 / np.where(np.abs(x) > 1e-12, x, 1e-12)
    return float(np.sum(M * dy))


# Original loop-based versions for comparison
def area_loop(vs):
    """Original loop-based area calculation."""
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * abs(y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


def Cp_loop(vs: np.ndarray) -> float:
    """Original loop-based Cp calculation."""
    a = 0.0
    x0, y0 = vs[0]
    for x1, y1 in vs[1:]:
        dx, dy = x1 - x0, y1 - y0
        dy_dx = dy / dx if dx != 0 else 0.0
        a += np.sqrt(1.0 + dy_dx ** 2) * abs(dx)
        x0, y0 = x1, y1
    return a


# ============================================================================
# Contour extraction methods
# ============================================================================
def extract_contour_matplotlib(R, Z, psi_grid, level=0.0):
    """Extract contour using matplotlib (original method)."""
    c = plt.contour(R, Z, psi_grid, levels=[level])
    if c.collections and c.collections[0].get_paths():
        vertices = c.collections[0].get_paths()[0].vertices
    else:
        vertices = None
    plt.close(c.figure)
    return vertices


def extract_contour_skimage(R, Z, psi_grid, level=0.0):
    """Extract contour using skimage (faster)."""
    if not HAS_SKIMAGE:
        return extract_contour_matplotlib(R, Z, psi_grid, level)
    
    contours = measure.find_contours(psi_grid, level)
    if not contours:
        return None
    
    # Get the largest contour
    largest = max(contours, key=len)
    
    # Convert from grid indices to physical coordinates
    # Note: find_contours returns (row, col) indices
    r_vals = np.interp(largest[:, 1], np.arange(R.shape[1]), R[0, :])
    z_vals = np.interp(largest[:, 0], np.arange(Z.shape[0]), Z[:, 0])
    
    return np.column_stack((r_vals, z_vals))


# ============================================================================
# Model setup
# ============================================================================
def setup_model(profile: str, checkpoint_path: str = None):
    """Initialize and load a trained PINN model."""
    config = CHECKPOINT_CONFIGS[profile]
    
    if checkpoint_path is None:
        checkpoint_path = config["path"]
    
    DEPTH = 4
    BREADTH = 40
    LR = 2e-2
    AF = "swish"
    INPUT_DIM = config["input_dim"]
    
    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    
    # Create minimal geometry for inference
    if profile == "solovev":
        spatial_domain = dde.geometry.HyperEllipticalToroid_old(
            eps_range=eps0,
            kappa_range=kappa0,
            delta_range=delta0,
            Amax=Amax,
            num_param=num_param,
        )
    else:
        alpha_ranges = [
            np.linspace(0.1, 0.5, num_param),
            np.linspace(1.5, 3.5, num_param),
        ]
        spatial_domain = dde.geometry.HyperEllipticalToroid(
            eps_range=eps0,
            kappa_range=kappa0,
            delta_range=delta0,
            alpha_ranges=alpha_ranges,
            num_param=num_param,
            psi_boundary_points=200,
        )
    
    # Dummy PDE for inference
    def dummy_pde(x, u):
        return u[:, 0:1]
    
    data = dde.data.PDE(
        spatial_domain,
        dummy_pde,
        [],
        num_domain=1,
        num_boundary=0,
        num_test=1
    )
    
    model = dde.Model(data=data, net=net)
    model.compile("adam", lr=LR, loss_weights=[1, 100])
    model.restore(checkpoint_path, verbose=0)
    
    return model, config


def create_input_grid(eps, kappa, delta, A=-0.155, alpha=None, grid_size=300, zoom=1.2, config=None):
    """Create input grid for model prediction."""
    inner_point = 1 - 1.1 * eps * (1 + zoom)
    outer_point = 1 + 1.1 * eps * (1 + zoom)
    low_point = -1.1 * kappa * eps * (1 + zoom)
    high_point = 1.1 * kappa * eps * (1 + zoom)
    
    r = np.linspace(inner_point, outer_point, grid_size)
    z = np.linspace(low_point, high_point, grid_size)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    
    spatial = np.column_stack((RR.ravel(), ZZ.ravel()))
    features = [spatial]
    
    if config and config["uses_A"]:
        A_col = np.full((spatial.shape[0], 1), A)
        features.append(A_col)
    
    if config and config["num_alpha"] > 0:
        if alpha is None:
            alpha = [0.3, 2.5]  # Default values
        alpha_col = np.tile(np.array(alpha).reshape(1, -1), (spatial.shape[0], 1))
        features.append(alpha_col)
    
    param_col = np.tile(np.array([[eps, kappa, delta]]), (spatial.shape[0], 1))
    features.append(param_col)
    
    X_in = np.hstack(features)
    return RR, ZZ, X_in


# ============================================================================
# Benchmark functions
# ============================================================================
def benchmark_model_inference(model, config, n_runs=50, grid_sizes=[100, 200, 300]):
    """Benchmark model inference at different grid sizes."""
    results = []
    
    # Random parameters within bounds
    eps = 0.32
    kappa = 1.7
    delta = 0.33
    
    for grid_size in grid_sizes:
        RR, ZZ, X_in = create_input_grid(
            eps, kappa, delta,
            grid_size=grid_size,
            config=config
        )
        
        def run_inference():
            model.predict(X_in)
        
        result = benchmark_function(run_inference, n_runs=n_runs, warmup=3,
                                     name=f"inference_grid{grid_size}")
        result.grid_size = grid_size
        result.n_points = grid_size ** 2
        results.append(result)
        
        print(f"  Grid {grid_size}x{grid_size} ({grid_size**2} pts): "
              f"{result.mean_time_ms:.2f} ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_contour_extraction(n_runs=100):
    """Benchmark contour extraction methods."""
    # Create a sample psi field
    n = 200
    r = np.linspace(0.5, 1.5, n)
    z = np.linspace(-0.5, 0.5, n)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    
    # Simple elliptical psi for testing
    eps, kappa = 0.32, 1.7
    psi = ((RR - 1) / eps) ** 2 + (ZZ / (eps * kappa)) ** 2 - 1
    
    print("\n  Contour extraction methods:")
    
    # Matplotlib method
    def run_matplotlib():
        extract_contour_matplotlib(RR, ZZ, psi, level=0.0)
    
    mpl_result = benchmark_function(run_matplotlib, n_runs=n_runs, warmup=5,
                                    name="contour_matplotlib")
    print(f"    Matplotlib: {mpl_result.mean_time_ms:.2f} ± {mpl_result.std_time_ms:.2f} ms")
    
    # Skimage method (if available)
    if HAS_SKIMAGE:
        def run_skimage():
            extract_contour_skimage(RR, ZZ, psi, level=0.0)
        
        ski_result = benchmark_function(run_skimage, n_runs=n_runs, warmup=5,
                                        name="contour_skimage")
        print(f"    Skimage:    {ski_result.mean_time_ms:.2f} ± {ski_result.std_time_ms:.2f} ms")
        print(f"    Speedup:    {mpl_result.mean_time_ms / ski_result.mean_time_ms:.1f}x")
        return [mpl_result, ski_result]
    
    return [mpl_result]


def benchmark_geometric_calculations(n_runs=1000):
    """Benchmark geometric calculations."""
    # Create sample contour vertices
    tau = np.linspace(0, 2 * np.pi, 500)
    eps, kappa, delta = 0.32, 1.7, 0.33
    x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    y = eps * kappa * np.sin(tau)
    vertices = np.column_stack((x, y))
    
    print("\n  Geometric calculations (500 vertices):")
    
    results = []
    
    # Area - loop vs vectorized
    loop_result = benchmark_function(lambda: area_loop(vertices), n_runs=n_runs,
                                     warmup=10, name="area_loop")
    vec_result = benchmark_function(lambda: area_vectorized(vertices), n_runs=n_runs,
                                    warmup=10, name="area_vectorized")
    print(f"    Area (loop):       {loop_result.mean_time_ms:.4f} ± {loop_result.std_time_ms:.4f} ms")
    print(f"    Area (vectorized): {vec_result.mean_time_ms:.4f} ± {vec_result.std_time_ms:.4f} ms")
    print(f"    Speedup:           {loop_result.mean_time_ms / vec_result.mean_time_ms:.1f}x")
    results.extend([loop_result, vec_result])
    
    # Cp - loop vs vectorized
    loop_result = benchmark_function(lambda: Cp_loop(vertices), n_runs=n_runs,
                                     warmup=10, name="Cp_loop")
    vec_result = benchmark_function(lambda: Cp_vectorized(vertices), n_runs=n_runs,
                                    warmup=10, name="Cp_vectorized")
    print(f"    Cp (loop):         {loop_result.mean_time_ms:.4f} ± {loop_result.std_time_ms:.4f} ms")
    print(f"    Cp (vectorized):   {vec_result.mean_time_ms:.4f} ± {vec_result.std_time_ms:.4f} ms")
    print(f"    Speedup:           {loop_result.mean_time_ms / vec_result.mean_time_ms:.1f}x")
    results.extend([loop_result, vec_result])
    
    return results


def benchmark_full_objective_evaluation(model, config, profile, n_runs=20, grid_size=300):
    """Benchmark a full objective function evaluation."""
    from shape_optimization_with_general_profile import (
        build_profile_config, predict_psi, compute_beta_p, area
    )
    
    # Setup profile config
    global PROFILE_CONFIG
    PROFILE_CONFIG = build_profile_config(profile)
    
    eps, kappa, delta = 0.32, 1.7, 0.33
    A = -0.155
    alpha = [0.3, 2.5] if config["num_alpha"] > 0 else None
    
    print(f"\n  Full objective evaluation (grid={grid_size}):")
    
    # Time individual components
    components = {}
    
    # 1. Grid creation
    def create_grid():
        return create_input_grid(eps, kappa, delta, A=A, alpha=alpha,
                                 grid_size=grid_size, config=config)
    components["grid_creation"] = benchmark_function(create_grid, n_runs=n_runs,
                                                     warmup=3, name="grid_creation")
    
    RR, ZZ, X_in = create_grid()
    
    # 2. Model inference
    def run_inference():
        return model.predict(X_in).reshape(-1)
    components["model_inference"] = benchmark_function(run_inference, n_runs=n_runs,
                                                       warmup=3, name="model_inference")
    
    psi_pred = run_inference()
    psi_grid = psi_pred.reshape(grid_size, grid_size)
    
    # 3. Contour extraction
    def extract_contour():
        return extract_contour_matplotlib(RR, ZZ, psi_grid, level=0.0)
    components["contour_extraction"] = benchmark_function(extract_contour, n_runs=n_runs,
                                                          warmup=3, name="contour_extraction")
    
    vertices = extract_contour()
    
    # 4. Geometric calculations
    if vertices is not None:
        def compute_geometry():
            area(vertices)
            Cp_vectorized(vertices)
            qstar_integral_vectorized(vertices)
        components["geometry"] = benchmark_function(compute_geometry, n_runs=n_runs*5,
                                                    warmup=10, name="geometry")
    
    # Print results
    total_ms = 0
    for name, result in components.items():
        print(f"    {name:20s}: {result.mean_time_ms:8.2f} ± {result.std_time_ms:.2f} ms")
        total_ms += result.mean_time_ms
    
    print(f"    {'TOTAL':20s}: {total_ms:8.2f} ms")
    print(f"    Evaluations/sec:    {1000/total_ms:.1f}")
    
    return components


def benchmark_optimization_run(model, config, profile, n_iterations=50):
    """Benchmark a short optimization run."""
    from scipy.optimize import minimize
    from shape_optimization_with_general_profile import (
        build_profile_config, predict_psi, compute_beta_p_and_qstar, area
    )
    
    global PROFILE_CONFIG
    PROFILE_CONFIG = build_profile_config(profile)
    
    print(f"\n  Optimization run ({n_iterations} iterations):")
    
    eval_times = []
    
    def objective(params):
        start = time.perf_counter()
        
        eps, kappa, delta = params[:3]
        A = -0.155
        alpha = params[3:5].tolist() if config["num_alpha"] > 0 else None
        
        # Create grid and predict
        RR, ZZ, X_in = create_input_grid(eps, kappa, delta, A=A, alpha=alpha,
                                         grid_size=150, config=config)  # Smaller grid for speed
        psi_pred = model.predict(X_in).reshape(-1)
        psi_grid = psi_pred.reshape(150, 150)
        
        # Extract contour and compute objective
        vertices = extract_contour_matplotlib(RR, ZZ, psi_grid, level=0.0)
        if vertices is None:
            return 1e10
        
        vol = area_vectorized(vertices)
        obj = (vol - 1.0) ** 2  # Simple volume objective
        
        eval_times.append((time.perf_counter() - start) * 1000)
        return obj
    
    # Initial guess
    x0 = [0.32, 1.7, 0.33]
    if config["num_alpha"] > 0:
        x0.extend([0.3, 2.5])
    x0 = np.array(x0)
    
    bounds = [(eps0[0], eps0[1]), (kappa0[0], kappa0[1]), (delta0[0], delta0[1])]
    if config["num_alpha"] > 0:
        bounds.extend([(0.1, 0.5), (1.5, 3.5)])
    
    start_total = time.perf_counter()
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': n_iterations, 'disp': False})
    total_time = time.perf_counter() - start_total
    
    eval_times = np.array(eval_times)
    print(f"    Total time:         {total_time:.2f} s")
    print(f"    Function evals:     {len(eval_times)}")
    print(f"    Avg eval time:      {np.mean(eval_times):.2f} ± {np.std(eval_times):.2f} ms")
    print(f"    Min/Max eval time:  {np.min(eval_times):.2f} / {np.max(eval_times):.2f} ms")
    print(f"    Final objective:    {result.fun:.6f}")
    
    return {
        "total_time_s": total_time,
        "n_evals": len(eval_times),
        "mean_eval_ms": float(np.mean(eval_times)),
        "std_eval_ms": float(np.std(eval_times)),
        "final_obj": float(result.fun),
    }


# ============================================================================
# Main benchmark runner
# ============================================================================
def run_all_benchmarks(profiles=None, save_results=True):
    """Run all benchmarks and optionally save results."""
    if profiles is None:
        profiles = ["solovev", "pedestal"]
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": {},
    }
    
    print("=" * 70)
    print("PINN Shape Optimization Benchmark")
    print("=" * 70)
    
    # Benchmark geometric calculations (profile-independent)
    print("\n[1] Geometric Calculations Benchmark")
    geo_results = benchmark_geometric_calculations()
    all_results["geometric_calculations"] = [asdict(r) for r in geo_results]
    
    # Benchmark contour extraction (profile-independent)
    print("\n[2] Contour Extraction Benchmark")
    contour_results = benchmark_contour_extraction()
    all_results["contour_extraction"] = [asdict(r) for r in contour_results]
    
    # Profile-specific benchmarks
    for profile in profiles:
        print(f"\n{'=' * 70}")
        print(f"[3] Profile: {profile.upper()}")
        print("=" * 70)
        
        profile_results = {}
        
        # Load model
        print(f"\nLoading {profile} model...")
        model, config = setup_model(profile)
        print(f"  Checkpoint: {CHECKPOINT_CONFIGS[profile]['path']}")
        print(f"  Input dim: {config['input_dim']}")
        
        # Inference benchmark
        print("\n[3.1] Model Inference Benchmark")
        inference_results = benchmark_model_inference(model, config, n_runs=30,
                                                      grid_sizes=[100, 150, 200, 300])
        profile_results["inference"] = [
            {**asdict(r), "grid_size": r.grid_size, "n_points": r.n_points}
            for r in inference_results
        ]
        
        # Full objective evaluation
        print("\n[3.2] Full Objective Evaluation Benchmark")
        obj_results = benchmark_full_objective_evaluation(model, config, profile,
                                                          n_runs=20, grid_size=200)
        profile_results["objective_components"] = {
            name: asdict(result) for name, result in obj_results.items()
        }
        
        # Short optimization run
        print("\n[3.3] Optimization Run Benchmark")
        opt_results = benchmark_optimization_run(model, config, profile, n_iterations=30)
        profile_results["optimization_run"] = opt_results
        
        all_results["profiles"][profile] = profile_results
        
        # Clear TF session to free memory before next profile
        tf.reset_default_graph()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for profile in profiles:
        pr = all_results["profiles"][profile]
        inference_200 = next(r for r in pr["inference"] if r["grid_size"] == 200)
        opt = pr["optimization_run"]
        
        print(f"\n{profile.upper()}:")
        print(f"  Single inference (200x200): {inference_200['mean_time_ms']:.1f} ms")
        print(f"  Objective evaluation:       {sum(r['mean_time_ms'] for r in pr['objective_components'].values()):.1f} ms")
        print(f"  Optimization (30 iter):     {opt['total_time_s']:.1f} s ({opt['n_evals']} evals)")
        print(f"  Throughput:                 {1000/opt['mean_eval_ms']:.0f} evals/sec")
    
    # Save results
    if save_results:
        save_dir = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/benchmark_results"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"benchmark_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    return all_results


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PINN shape optimization speed")
    parser.add_argument("--profile", type=str, choices=["solovev", "pedestal"],
                        help="Run benchmarks for a specific profile")
    parser.add_argument("--all", action="store_true",
                        help="Run benchmarks for all profiles")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    args = parser.parse_args()
    
    if args.all or args.profile is None:
        profiles = ["solovev", "pedestal"]
    else:
        profiles = [args.profile]
    
    run_all_benchmarks(profiles=profiles, save_results=not args.no_save)

