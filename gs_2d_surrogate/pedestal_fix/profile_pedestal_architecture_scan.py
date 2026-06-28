"""
Pedestal profile training with configurable network architecture.

This script is designed for architecture hyperparameter scanning:
- Configurable network depth (number of hidden layers)
- Configurable network width (neurons per layer)
- Saves detailed training metrics for comparison

Usage:
    # Test different depths
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 2 --width 40
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 3 --width 40
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 4 --width 40
    
    # Test different widths
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 4 --width 20
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 4 --width 40
    python profile_pedestal_architecture_scan.py --num_params 5 --depth 4 --width 60

Based on: profile_pedestal_fixed.py
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import argparse

# Parse arguments first
parser = argparse.ArgumentParser(description='Pedestal pPINN with configurable architecture')
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
                    help='Maximum boundary points (default: None = no limit)')

# NEW: Architecture arguments
parser.add_argument('--depth', type=int, default=4,
                    help='Number of hidden layers (default: 4)')
parser.add_argument('--width', type=int, default=40,
                    help='Neurons per hidden layer (default: 40)')
parser.add_argument('--activation', type=str, default='swish',
                    choices=['swish', 'tanh', 'sigmoid', 'relu'],
                    help='Activation function (default: swish)')
parser.add_argument('--lr', type=float, default=2e-2,
                    help='Learning rate for Adam (default: 0.02)')
parser.add_argument('--bc_weight', type=float, default=100,
                    help='Boundary condition loss weight (default: 100)')

# Experiment naming
parser.add_argument('--experiment_name', type=str, default=None,
                    help='Custom experiment name for organizing runs')

args = parser.parse_args()

print("="*60)
print("ARCHITECTURE SCAN CONFIGURATION")
print("="*60)
print(f"  num_params: {args.num_params}")
print(f"  depth (hidden layers): {args.depth}")
print(f"  width (neurons/layer): {args.width}")
print(f"  activation: {args.activation}")
print(f"  learning rate: {args.lr}")
print(f"  BC weight: {args.bc_weight}")
print(f"  epochs_adam: {args.epochs_adam}")
print(f"  epochs_lbfgs: {args.epochs_lbfgs}")
print("="*60)

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
# Pedestal Profile Parameters
# ============================================================
PSI_PED = -0.08
WIDTH = 0.06

NUM_ALPHA = 2
INPUT_DIM = 2 + NUM_ALPHA + 3  # R, Z, α0, α1, eps, kappa, delta

alpha_ranges = [
    np.linspace(0.05, 0.2, num_param),   # α0: p_edge (low values)
    np.linspace(0.3, 1.0, num_param),    # α1: p_core (high values)
]


def p_of_psi_pedestal(psi, alpha):
    """CORRECTED H-mode pedestal pressure profile."""
    p_edge = alpha[:, 0:1]
    p_core = alpha[:, 1:2]
    
    arg = (psi - PSI_PED) / WIDTH
    H = 0.5 * (1.0 + tf.tanh(arg))
    p = p_core * (1.0 - H) + p_edge * H
    
    return p


def dp_dpsi_pedestal(psi, alpha):
    """Compute dp/dψ using automatic differentiation."""
    with tf.GradientTape() as tape:
        tape.watch(psi)
        p_val = p_of_psi_pedestal(psi, alpha)
    return tape.gradient(p_val, psi)


def pde_pedestal(x, u):
    """Grad-Shafranov equation with pedestal pressure profile."""
    psi = u[:, 0:1]
    R = x[:, 0:1]
    
    psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
    psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
    psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
    
    alpha = x[:, 2:2+NUM_ALPHA]
    dpdpsi = dp_dpsi_pedestal(psi, alpha)
    
    GS = psi_RR - psi_R/R + psi_ZZ + R**2 * dpdpsi
    
    return GS


def gen_boundary_data(num_boundary_pts):
    """Generate boundary training data (ψ = 0 on plasma boundary)."""
    N = num_boundary_pts
    tau = np.linspace(0, 2*np.pi, N)
    
    R_list, Z_list = [], []
    alpha_list, eps_list, kappa_list, delta_list = [], [], [], []
    
    for eps_val in eps_vals:
        for kappa_val in kappa_vals:
            for delta_val in delta_vals:
                Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val) * np.sin(tau))
                Zb = eps_val * kappa_val * np.sin(tau)
                
                for a0 in alpha_ranges[0]:
                    for a1 in alpha_ranges[1]:
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


class LossHistoryCallback(dde.callbacks.Callback):
    """Callback to record detailed loss history."""
    
    def __init__(self, record_every=10):
        super().__init__()
        self.record_every = record_every
        self.history = []
        self.start_time = None
        
    def on_train_begin(self):
        self.start_time = time.time()
        
    def on_epoch_end(self):
        if self.model.train_state.step % self.record_every == 0:
            elapsed = time.time() - self.start_time
            train_loss = self.model.train_state.loss_train
            test_loss = self.model.train_state.loss_test
            
            # Total loss (sum of components)
            total_train = sum(train_loss) if isinstance(train_loss, (list, tuple)) else train_loss
            total_test = sum(test_loss) if isinstance(test_loss, (list, tuple)) else test_loss
            
            self.history.append({
                'step': int(self.model.train_state.step),
                'elapsed_time': elapsed,
                'train_loss_pde': float(train_loss[0]) if isinstance(train_loss, (list, tuple)) else float(train_loss),
                'train_loss_bc': float(train_loss[1]) if isinstance(train_loss, (list, tuple)) and len(train_loss) > 1 else 0,
                'train_loss_total': float(total_train),
                'test_loss_total': float(total_test),
            })


if __name__ == "__main__":
    
    TIME = time.strftime("%m%d%Y_%H%M%S")
    
    # Create descriptive output directory name
    if args.output_dir:
        PATH = args.output_dir
    else:
        exp_name = args.experiment_name or "arch_scan"
        PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/{exp_name}_d{args.depth}_w{args.width}_p{args.num_params}_{TIME}"
    
    os.makedirs(PATH, exist_ok=True)
    print(f"Output directory: {PATH}")
    
    # Save configuration
    config = {
        'num_params': args.num_params,
        'depth': args.depth,
        'width': args.width,
        'activation': args.activation,
        'lr': args.lr,
        'bc_weight': args.bc_weight,
        'epochs_adam': args.epochs_adam,
        'epochs_lbfgs': args.epochs_lbfgs,
        'num_domain': args.num_domain,
        'scale_domain': args.scale_domain,
        'max_bc_points': args.max_bc_points,
        'timestamp': TIME,
    }
    
    with open(os.path.join(PATH, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # --------------------------------------------------------
    # Generate training data
    # --------------------------------------------------------
    print("\nGenerating boundary data...")
    x_bc, u_bc = gen_boundary_data(num_boundary_pts=201)
    
    if args.max_bc_points is not None and len(x_bc) > args.max_bc_points:
        idx = np.random.choice(len(x_bc), args.max_bc_points, replace=False)
        x_bc = x_bc[idx]
        u_bc = u_bc[idx]
        print(f"Subsampled to {args.max_bc_points} boundary points")
    
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
    # Network architecture - CONFIGURABLE
    # --------------------------------------------------------
    DEPTH = args.depth
    WIDTH = args.width
    AF = args.activation
    LR = args.lr
    
    layer_sizes = [INPUT_DIM] + DEPTH * [WIDTH] + [1]
    print(f"\nNetwork architecture: {layer_sizes}")
    print(f"Total parameters: ~{sum(layer_sizes[i]*layer_sizes[i+1] + layer_sizes[i+1] for i in range(len(layer_sizes)-1))}")
    
    net = dde.maps.FNN(layer_sizes, AF, "Glorot normal")
    model = dde.Model(data, net)
    
    # --------------------------------------------------------
    # Training: Adam phase
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: Adam optimizer")
    print("="*60)
    
    loss_callback = LossHistoryCallback(record_every=10)
    
    decay_rate = ("inverse time", 100, 0.1)
    model.compile("adam", lr=LR, decay=decay_rate, loss_weights=[1, args.bc_weight])
    
    loss_history, train_state = model.train(
        epochs=args.epochs_adam,
        display_every=100,
        callbacks=[loss_callback]
    )
    
    # Mark end of Adam phase
    adam_end_step = model.train_state.step
    
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
        maxiter=100000,
        maxcor=50,
        ftol=0,
        gtol=1e-10,
        maxfun=120000,
        maxls=50,
    )
    
    model.compile("L-BFGS-B", loss_weights=[1, args.bc_weight])
    
    loss_history, train_state = model.train(
        epochs=args.epochs_lbfgs,
        display_every=100,
        callbacks=[loss_callback]
    )
    
    dde.saveplot(loss_history, train_state, issave=True, isplot=True,
                output_dir=PATH, output_fname="loss_lbfgs")
    
    # --------------------------------------------------------
    # Save detailed loss history
    # --------------------------------------------------------
    loss_data = {
        'config': config,
        'adam_end_step': adam_end_step,
        'history': loss_callback.history
    }
    
    with open(os.path.join(PATH, 'loss_history.json'), 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"\nLoss history saved to: {PATH}/loss_history.json")
    
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
    a0_test = 0.1
    a1_test = 0.6
    
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
    ax1.set_title(f'ψ field (depth={args.depth}, width={args.width})')
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
    
    # Save final statistics
    final_stats = {
        'psi_min': float(psi_pred.min()),
        'psi_max': float(psi_pred.max()),
        'final_train_loss': float(sum(model.train_state.loss_train)),
        'final_test_loss': float(sum(model.train_state.loss_test)),
        'total_steps': int(model.train_state.step),
        'has_zero_crossing': bool(psi_pred.min() < 0 < psi_pred.max()),
    }
    
    with open(os.path.join(PATH, 'final_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"Validation plot saved to: {PATH}/validation_plot.png")
    print(f"\nψ statistics: min={psi_pred.min():.4f}, max={psi_pred.max():.4f}")
    
    if psi_pred.min() < 0 < psi_pred.max():
        print("✓ ψ field has proper zero crossing")
    else:
        print("⚠ WARNING: ψ field may not have proper zero crossing!")
