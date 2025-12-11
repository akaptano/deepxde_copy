# adapted from gs_2d_surrogate/general_solovev_equil_parametrized_shape.py FINAL VERSION 04/16/2025
# 8/13/2025 added float64 to solve l-bfgs stopping early issue
# Before running, in terminal run:
# export TF_USE_LEGACY_KERAS=1 
# Nov 2025 adapted to use Chebyshev polynomial pressure profile
# Dec 2025 adapted to use Pedestal pressure profile

"""
How to run:

python profile.py --profile pedestal --num_params 6

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from contextlib import nullcontext
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # Suppress TF logging
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import site
import multiprocessing
import time
import psutil
import argparse
import wandb
print("Initializing wandb")
wandb.init(project="shape_parametrized_pedestal", config={"system": True})
print("Wandb initialized")

parser = argparse.ArgumentParser()
parser.add_argument('--num_params', type=str, default=2,
                    help='The number of parameters for each dimension')
parser.add_argument('--profile', type=str, default="pedestal",
                    help='The pressure profile to use')
parser.add_argument('--max_bc_points', type=int, default=500000,
                    help='Max boundary-condition samples to keep (-1 keeps all)')
parser.add_argument('--max_test_bc_points', type=int, default=20000,
                    help='Max boundary samples for eval (-1 keeps all)')
parser.add_argument('--bc_seed', type=int, default=0,
                    help='Seed used when subsampling boundary points')

args = parser.parse_args()

print("Running with profile:", args.profile, "and num_params:", args.num_params)

# Use this path to import customized DeepXDE, instead of the pip installed version
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
# This ensures it's searched before the system packages
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)

import deepxde as dde
from deepxde.backend import backend_name
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.config.optimizer.set_jit(True)  # Enable XLA compilation to use all GPUs

TRAIN_BC_LIMIT = None if args.max_bc_points < 0 else int(args.max_bc_points)
TEST_BC_LIMIT = None if args.max_test_bc_points < 0 else int(args.max_test_bc_points)
rng = np.random.default_rng(args.bc_seed)


def subsample_bc(points, values, max_points, label="boundary"):
    """Subsample boundary-condition points to keep GPU memory in check."""
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points, values
    idx = rng.choice(len(points), size=max_points, replace=False)
    print(f"{label}: subsampled from {len(points)} to {max_points} points")
    return points[idx], values[idx]

# ----------------------------------------------------------------------------
# Check GPU availability
# ----------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPU devices:", gpus)            # empty list → TF cannot see a GPU
print("Built with CUDA support:",
      tf.test.is_built_with_cuda())             # True only if your TF build is GPU–enabled

strategy = None
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) > 1:
            if backend_name == "tensorflow":
                strategy = tf.distribute.MirroredStrategy()
                print(f"Enabled MirroredStrategy on {strategy.num_replicas_in_sync} GPUs")
            else:
                print(
                    "Multiple GPUs detected but backend "
                    f"{backend_name} does not support tf.distribute; "
                    "falling back to single-GPU graph mode."
                )
        else:
            print("Single GPU detected; training on the only available device.")
    except RuntimeError as exc:
        print("Failed to configure GPU memory growth/distribution:", exc)
else:
    print("No GPUs detected; training will fall back to CPU.")

sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')
from utils.gs_solovev_sol import GS_Linear

print("Using DeepXDE from:", dde.__file__)


# this solves l-bfgs stopping early issue
dde.config.set_default_float("float64")


######################
# ITER shape #
######################

# same parameters as paper
eps_deviation = 0.2
kappa_deviation = 0.75
delta_deviation = 0.5
eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)
delta0 = (0 - delta_deviation, 0 + delta_deviation)
Amax = 0.2
num_param = int(args.num_params)
Arange = np.linspace(-Amax, Amax, num_param)
eps = np.linspace(eps0[0], eps0[1], num_param)
kappa = np.linspace(kappa0[0], kappa0[1], num_param)
delta = np.linspace(delta0[0], delta0[1], num_param)


if args.profile == "solovev":
    def pde_solovev(x, u):
        psi = u[:, 0:1]
        psi_r = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_rr = dde.grad.hessian(psi, x, i=0, j=0)
        psi_zz = dde.grad.hessian(psi, x, i=1, j=1)
        A = x[:, 2:3]
        GS = psi_rr - psi_r / x[:, 0:1] + psi_zz - (1 - A) * x[:, 0:1] ** 2 - A
        return GS

elif args.profile == "polynomial":
    NUM_ALPHA = 4
    INPUT_DIM = 2 + NUM_ALPHA + 3
    alpha_ranges = [
        np.linspace(-1.0, 1.0, num_param),   # α0 range
        np.linspace(-1.0, 1.0, num_param),   # α1 range
        np.linspace(-1.0, 1.0, num_param),   # α2 range
    ]

    # ============================================================
    # Generate boundary training data with polynomial pressure
    # ============================================================

    def gen_traindata(num_boundary_pts, max_points=None, label="boundary"):
        N = num_boundary_pts
        tau = np.linspace(0, 2*np.pi, N)

        # Allocate arrays
        R_list = []
        Z_list = []
        alpha_list = []
        eps_list = []
        kappa_list = []
        delta_list = []

        # Loop over all combinations: alpha0,alpha1,alpha2, eps,kappa,delta
        for i0 in range(num_param):
            for i1 in range(num_param):
                for i2 in range(num_param):
                    for j in range(num_param):
                        for k in range(num_param):
                            for kk in range(num_param):

                                # get α's
                                alpha0 = alpha_ranges[0][i0]
                                alpha1 = alpha_ranges[1][i1]
                                alpha2 = alpha_ranges[2][i2]

                                # get shape parameters
                                eps_val = eps[j]
                                kappa_val = kappa[k]
                                delta_val = delta[kk]

                                # boundary parametric equation
                                Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val)*np.sin(tau))
                                Zb = eps_val * kappa_val * np.sin(tau)

                                # append lists
                                R_list.append(Rb)
                                Z_list.append(Zb)
                                alpha_list.append(np.stack((alpha0*np.ones(N),
                                                            alpha1*np.ones(N),
                                                            alpha2*np.ones(N)), axis=1))
                                eps_list.append(eps_val*np.ones((N,1)))
                                kappa_list.append(kappa_val*np.ones((N,1)))
                                delta_list.append(delta_val*np.ones((N,1)))

        # flatten all
        R_flat = np.concatenate(R_list, axis=0)[:, None]
        Z_flat = np.concatenate(Z_list, axis=0)[:, None]
        alpha_flat = np.concatenate(alpha_list, axis=0)
        eps_flat = np.concatenate(eps_list, axis=0)
        kappa_flat = np.concatenate(kappa_list, axis=0)
        delta_flat = np.concatenate(delta_list, axis=0)

        # concat final input array
        x_boundary = np.hstack((R_flat, Z_flat, alpha_flat, eps_flat, kappa_flat, delta_flat))

        # boundary condition ψ = 0
        uvals = np.zeros((len(x_boundary), 1))

        return subsample_bc(x_boundary, uvals, max_points, label)


    # ============================================================
    # Polynomial pressure p(psi) = sum alpha_k * psi^k
    # ============================================================

    def p_of_psi(psi, alpha):
        """
        psi: tensor shape (N,1)
        alpha: tensor shape (N, NUM_ALPHA)
        returns: p(psi), same shape as psi
        """
        p = 0
        for k in range(alpha.shape[1]):   # alpha_k * psi^k
            p += alpha[:, k:k+1] * psi**k
        return p

    def dp_dpsi(psi, alpha):
        """
        Compute derivative dp/dpsi analytically.
        """
        dp = 0
        for k in range(1, alpha.shape[1]):
            dp += k * alpha[:, k:k+1] * psi**(k-1)
        return dp


    # ============================================================
    # General PDE with polynomial pressure profile
    # ============================================================

    def pde_general_polynomial(x, u):
        psi = u[:, 0:1]

        # Spatial derivatives
        psi_R  = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)

        # Extract polynomial coefficients from x
        # x = [R, Z, alpha0, alpha1, ..., alpha_{K}, eps, kappa, delta]
        # Assuming: first 2 entries are R,Z. Next NUM_ALPHA entries are α.
        alpha = x[:, 2:2+NUM_ALPHA]

        # Construct dp/dpsi
        dpdpsi = dp_dpsi(psi, alpha)   # shape (N,1)

        # FULL GSE (F(ψ) profile omitted for now → you can add later)
        GS = psi_RR - psi_R/x[:, 0:1] + psi_ZZ + (x[:, 0:1]**2) * dpdpsi

        return GS



elif args.profile == "chebyshev":
    # ============================================================
    # Chebyshev polynomial pressure profile
    # ============================================================

    NUM_ALPHA = 4  # Example: up to T3(psi). Adjust as needed.
    INPUT_DIM = 2 + NUM_ALPHA + 3

    # --- Chebyshev polynomials of the first kind ---
    def cheb_T(k, psi):
        """Returns T_k(psi) evaluated elementwise."""
        if k == 0:
            return tf.ones_like(psi)
        elif k == 1:
            return psi
        else:
            T0 = tf.ones_like(psi)
            T1 = psi
            for n in range(2, k + 1):
                Tn = 2 * psi * T1 - T0
                T0, T1 = T1, Tn
            return Tn

    # --- Normalize psi to [-1,1] for Chebyshev ---
    def normalize_psi(psi, psi_min=-1.0, psi_max=0.0):
        """
        Normalize psi to [-1,1] dynamically.
        psi_min, psi_max = typical GS range (outer boundary at psi=0).
        """
        return 2.0 * (psi - psi_min) / (psi_max - psi_min) - 1.0

    # ============================================================
    # p(psi) and dp/dpsi using Chebyshev polynomials
    # ============================================================

    def p_of_psi_cheb(psi, alpha):
        """
        psi: (N,1)
        alpha: (N, NUM_ALPHA)
        Returns p(psi) = sum alpha_k * T_k(norm_psi)
        """
        psi_norm = normalize_psi(psi)
        p = 0
        for k in range(NUM_ALPHA):
            p += alpha[:, k:k+1] * cheb_T(k, psi_norm)
        return p


    def dp_dpsi_cheb(psi, alpha):
        """
        Compute dp/dpsi by automatic differentiation of p(psi) with tf.gradients.
        No analytic derivative required.
        """

        with tf.GradientTape() as tape:
            tape.watch(psi)
            p_val = p_of_psi_cheb(psi, alpha)

        dp = tape.gradient(p_val, psi)
        return dp

    # ============================================================
    # Full GSE with Chebyshev pressure profile
    # ============================================================

    def pde_general_cheb(x, u):
        psi = u[:, 0:1]

        R = x[:, 0:1]

        psi_R  = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)

        # Extract α0..α{NUM_ALPHA-1}
        alpha = x[:, 2 : 2 + NUM_ALPHA]  # shape (N, NUM_ALPHA)

        # dp/dψ
        dp_dpsi = dp_dpsi_cheb(psi, alpha)

        # No current term yet (F(ψ)); we can add it later
        GS = psi_RR - psi_R/R + psi_ZZ + R**2 * dp_dpsi

        return GS

    # ============================================================
    # Pressure coefficient ranges (Chebyshev α_k)
    # ============================================================

    alpha_ranges = [
        np.linspace(-1.0, 1.0, num_param),  # α0
        np.linspace(-1.0, 1.0, num_param),  # α1
        np.linspace(-0.5, 0.5, num_param),  # α2
        np.linspace(-0.2, 0.2, num_param),  # α3
    ]

    # ============================================================
    # Boundary training data generator (integrated Chebyshev version)
    # ============================================================

    def gen_traindata(num_boundary_pts, max_points=None, label="boundary"):
        N = num_boundary_pts
        tau = np.linspace(0, 2*np.pi, N)

        R_list = []
        Z_list = []
        alpha_list = []
        eps_list = []
        kappa_list = []
        delta_list = []

        for j_eps in range(num_param):
            for j_kappa in range(num_param):
                for j_delta in range(num_param):

                    eps_val = eps[j_eps]
                    kappa_val = kappa[j_kappa]
                    delta_val = delta[j_delta]

                    # boundary curve
                    Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val)*np.sin(tau))
                    Zb = eps_val * kappa_val * np.sin(tau)

                    for a0 in alpha_ranges[0]:
                        for a1 in alpha_ranges[1]:
                            for a2 in alpha_ranges[2]:
                                for a3 in alpha_ranges[3]:

                                    R_list.append(Rb)
                                    Z_list.append(Zb)

                                    alpha_list.append(
                                        np.column_stack((
                                            a0*np.ones(N),
                                            a1*np.ones(N),
                                            a2*np.ones(N),
                                            a3*np.ones(N),
                                        ))
                                    )

                                    eps_list.append(eps_val*np.ones((N,1)))
                                    kappa_list.append(kappa_val*np.ones((N,1)))
                                    delta_list.append(delta_val*np.ones((N,1)))

        R_flat = np.concatenate(R_list)[:, None]
        Z_flat = np.concatenate(Z_list)[:, None]
        alpha_flat = np.concatenate(alpha_list)
        eps_flat = np.concatenate(eps_list)
        kappa_flat = np.concatenate(kappa_list)
        delta_flat = np.concatenate(delta_list)

        x_boundary = np.hstack((
            R_flat,
            Z_flat,
            alpha_flat,
            eps_flat,
            kappa_flat,
            delta_flat
        ))

        # Boundary condition ψ = 0
        uvals = np.zeros((x_boundary.shape[0], 1))

        return subsample_bc(x_boundary, uvals, max_points, label)


elif args.profile == "pedestal":
    # ============================================================
    # Pedestal Profile with Frozen Width, Location, Steepness
    # ============================================================

    NUM_ALPHA = 2                    # Only pedestal height + core pressure are free
    INPUT_DIM = 2 + NUM_ALPHA + 3   # R,Z + αs + eps,kappa,delta

    # --------------------------
    # Fixed pedestal parameters
    # --------------------------
    PSI_PED_CONST = -0.1             # pedestal location (in psi)
    WIDTH_CONST    = 0.05            # pedestal width
    STEEPNESS_CONST = 4.0            # tanh slope/sharpness

    # --------------------------
    # Trainable parameters (2D sweep)
    # --------------------------
    alpha_ranges = [
        np.linspace(0.1, 1.0, num_param),   # α0: pedestal height
        np.linspace(1.5, 5.0, num_param),   # α1: core height pressure
    ]

    def p_of_psi_pedestal(psi, alpha):
        """
        alpha[:,0] = pedestal height p_ped
        alpha[:,1] = core pressure p_core

        PSI_PED_CONST: pedestal location (fixed)
        WIDTH_CONST: pedestal width (fixed)
        STEEPNESS_CONST: tanh steepness (fixed)
        """
        p_ped   = alpha[:, 0:1]
        p_core  = alpha[:, 1:1+1]

        arg =  (psi - PSI_PED_CONST) / WIDTH_CONST

        # H-mode pedestal profile
        p = p_core + (p_ped - p_core) * 0.5 * (1 - tf.tanh(arg))

        return p


    def dp_dpsi_pedestal(psi, alpha):
        with tf.GradientTape() as tape:
            tape.watch(psi)
            p_val = p_of_psi_pedestal(psi, alpha)
        return tape.gradient(p_val, psi)

    def pde_general_pedestal(x, u):
        psi = u[:, 0:1]
        R   = x[:, 0:1]

        # derivatives
        psi_R  = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)

        # α0, α1 (only)
        alpha = x[:, 2:2+NUM_ALPHA]

        dpdpsi = dp_dpsi_pedestal(psi, alpha)

        GS = psi_RR - psi_R/R + psi_ZZ + R**2 * dpdpsi
        return GS

    def gen_traindata(num_boundary_pts, max_points=None, label="boundary"):
        N = num_boundary_pts
        tau = np.linspace(0, 2*np.pi, N)

        R_list = []
        Z_list = []
        alpha_list = []
        eps_list = []
        kappa_list = []
        delta_list = []

        for eps_val in eps:
            for kappa_val in kappa:
                for delta_val in delta:

                    Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val)*np.sin(tau))
                    Zb = eps_val * kappa_val * np.sin(tau)

                    # free parameters α0, α1 sweep
                    for a0 in alpha_ranges[0]:
                        for a1 in alpha_ranges[1]:
                            R_list.append(Rb)
                            Z_list.append(Zb)
                            alpha_list.append(
                                np.column_stack((a0*np.ones(N),
                                                 a1*np.ones(N)))
                            )
                            eps_list.append(eps_val*np.ones((N,1)))
                            kappa_list.append(kappa_val*np.ones((N,1)))
                            delta_list.append(delta_val*np.ones((N,1)))

        R_flat = np.concatenate(R_list)[:, None]
        Z_flat = np.concatenate(Z_list)[:, None]
        alpha_flat = np.concatenate(alpha_list)
        eps_flat = np.concatenate(eps_list)
        kappa_flat = np.concatenate(kappa_list)
        delta_flat = np.concatenate(delta_list)

        x_boundary = np.hstack((
            R_flat,
            Z_flat,
            alpha_flat,   # α0, α1 only
            eps_flat,
            kappa_flat,
            delta_flat
        ))

        uvals = np.zeros((x_boundary.shape[0], 1))
        return subsample_bc(x_boundary, uvals, max_points, label)







def psi_r(x,u):
    return dde.grad.jacobian(u, x, i=0, j=0)
def psi_z(x,u):
    return  dde.grad.jacobian(u, x, i=0, j=1)
def psi_rr(x, u):
    return dde.grad.hessian(u, x, i=0, j=0)
def psi_zz(x, u):
    return dde.grad.hessian(u, x, i=1, j=1)

print("Generating spatial domain...")
spatial_domain = dde.geometry.HyperEllipticalToroid(
    eps_range=eps0,
    kappa_range=kappa0,
    delta_range=delta0,
    alpha_ranges=alpha_ranges,
    num_param=num_param,
    psi_boundary_points=200
) 
print("Spatial domain generation complete")
print("Generating training data...")
x, u = gen_traindata(1001, max_points=TRAIN_BC_LIMIT, label="train BC")
print("Training data generation complete")

n_test = 100
x_test, u_test = gen_traindata(n_test, max_points=TEST_BC_LIMIT, label="test BC")
print("Test data generation complete")
x_domain = spatial_domain.random_points(n_test)
x_test = np.concatenate((x_test, x_domain))
u_test = np.concatenate((u_test, np.zeros((n_test, 1))))
print("Random test data generation complete")

bc135 = dde.PointSetBC(x,u)
print("Generating PDE data...")
start_time = time.time()


if args.profile == "solovev":
    data = dde.data.PDE(
        spatial_domain,
        pde_solovev,
        [bc135],
        num_domain=1028,
        num_boundary=0,
        num_test=n_test,
        train_distribution="LHS"
    )
elif args.profile == "polynomial":
    data = dde.data.PDE(
        spatial_domain,
        pde_general_polynomial,
        [bc135],
        num_domain=1028,
        num_boundary=0,
        num_test=n_test,
        train_distribution="LHS"
    )
elif args.profile == "chebyshev":
    data = dde.data.PDE(
        spatial_domain,
        pde_general_cheb,
        [bc135],
        num_domain=1028,
        num_boundary=0,
        num_test=n_test,
        train_distribution="LHS"
    )

elif args.profile == "pedestal":
    data = dde.data.PDE(
        spatial_domain,
        pde_general_pedestal,
        [bc135],
        num_domain=1028,
        num_boundary=0,
        num_test=n_test,
        train_distribution="LHS"
    )


print(f"PDE data generation complete. Time taken: {time.time() - start_time:.2f} seconds")

# Save data for future use
np.save('pde_data.npy', {
    'train_x': data.train_x,
    'train_y': data.train_y,
    'train_aux_vars': data.train_aux_vars,
    'train_x_all': data.train_x_all,
    'train_x_bc': data.train_x_bc,
    'num_bcs': data.num_bcs,
    'test_x': data.test_x,
    'test_y': data.test_y,
    'test_aux_vars': data.test_aux_vars
})


import time
TIME = time.strftime("%m%d%Y_%H%M")
CONFIG = "ITER"
LR = 2e-2
DEPTH = 4
BREADTH = 40
run = "01_100Adam_BFGS"
AF = "swish"
# RUN_NAME = f"network_sweep_{DATE}_depth0{DEPTH}_breadth{BREADTH}_{AF}_lr{LR}-varying-short_lw1-10_{run}"
# PATH = f"./cefron/{CONFIG}/runs/{RUN_NAME}"
PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}_parametrized{args.num_params}_{args.profile}"



# Check whether the specified path exists or not
isExist = os.path.exists(PATH)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(PATH)
  print("The new directory is created!")
  print("Will save model to:", PATH)

# Plot collocation points for visual check
# %matplotlib

fig,ax=plt.subplots(1, figsize=(5, 5))
ax.scatter(data.train_x_bc[:, 0], data.train_x_bc[:, 1], s=2, color='r')
ax.set_title('Collocation Points')
ax.set_xlabel('R/R_0')
ax.set_ylabel(r'$u(r,z=0)$')

fig =plt.figure(2, figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    data.train_x[:, 0], 
    data.train_x[:, 1], 
    data.train_x[:, 2], 
    s=2, 
    color='r'
)
ax.set_title('Collocation Points')
ax.set_xlabel('R/R_0')
ax.set_ylabel(r'$u(r,z=0)$')
plt.savefig(os.path.join(PATH, "collocation_points.pdf"))
plt.close()





strategy_scope = strategy.scope() if strategy is not None else nullcontext()
with strategy_scope:
    print("Before BFGS\n")
    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

    model = dde.Model(data, net)
    decay_rate = ("inverse time", 100, 0.1)

    # - `InverseTimeDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: 
    # ("inverse time", decay_steps, decay_rate)
    # - `CosineDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: 
    # ("cosine", decay_steps, alpha)


    # Compile, train and save model
    model.compile("adam", lr = LR, decay = decay_rate,loss_weights=[1, 100]) # changed loss weights from 1,100 to 1,1


    # loss_history, train_state = model.train(epochs=1000, display_every=10)
    loss_history, train_state = model.train(epochs=10, display_every=10)

    dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=PATH, output_fname="loss_history")


    # model.save(f"{PATH}/ITER_adam", protocol="backend", verbose=1)

    print("Done with Adam\n")


    print("After BFGS\n")

    # this one is not working
    # options = {
    #     "disp": None,
    #     "maxcor": 50,
    #     "ftol": np.finfo(float).eps,
    #     "gtol": 1e-8,
    #     "eps": 1e-8,
    #     "maxfun": 15000,
    #     "maxiter": 1000,
    #     "iprint": -1,
    #     "maxls": 50,    
    # }

    # this one is the new added one
    from deepxde.optimizers import set_LBFGS_options


    # set_LBFGS_options(
    #     maxiter = 500,
    #     maxcor  = 50,
    #     ftol    = 0,
    #     gtol    = 1e-10,
    #     maxfun  = 500,
    #     maxls   = 50,
    # )
    # original
    set_LBFGS_options(
        maxiter = 20000,
        maxcor  = 50,
        ftol    = 0,
        gtol    = 1e-10,
        maxfun  = 15000,
        maxls   = 50,
    )


    # Compile, train and save model
    model.compile("L-BFGS-B", loss_weights=[1, 100])

    loss_history, train_state = model.train(
        # epochs=10000,
        epochs=50,
        display_every=10, 
    )
    dde.saveplot(
        loss_history, 
        train_state, 
        issave=True, 
        isplot=True,
        output_dir=PATH,
        output_fname="loss_history_bfgs"
    )

    model.save(f"{PATH}/ITER", protocol="backend", verbose=1)






# ########################################################
# # Plots and Evaluations
# ########################################################

# def evaluate_grid(model, alpha, eps, kappa, delta, 
#                   Rmin=0.5, Rmax=1.5, Zmin=-1.0, Zmax=1.0, 
#                   NR=200, NZ=200):

#     R = np.linspace(Rmin, Rmax, NR)
#     Z = np.linspace(Zmin, Zmax, NZ)

#     RR, ZZ = np.meshgrid(R, Z)

#     N = NR * NZ
#     X = np.zeros((N, 2 + NUM_ALPHA + 3))

#     X[:,0] = RR.reshape(-1)
#     X[:,1] = ZZ.reshape(-1)

#     # α parameters
#     X[:,2:2+NUM_ALPHA] = np.array(alpha)[None,:]

#     # shape
#     X[:,-3] = eps
#     X[:,-2] = kappa
#     X[:,-1] = delta

#     psi_pred = model.predict(X).reshape(NZ, NR)

#     return R, Z, psi_pred



# import matplotlib.pyplot as plt

# def plot_flux_surfaces(R, Z, psi, levels=20, title="Flux Surfaces"):
#     plt.figure(figsize=(6,6))
#     CS = plt.contour(R, Z, psi, levels=levels, colors='black')
#     plt.clabel(CS, inline=True)
#     plt.xlabel("R")
#     plt.ylabel("Z")
#     plt.title(title)
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()


# def plot_pressure_profile(alpha, N=200):
#     psi = np.linspace(-1, 0, N).reshape(-1,1)

#     # convert psi → normalized domain for Chebyshev evaluation
#     psi_norm = normalize_psi(psi)

#     p = np.zeros_like(psi)
#     for k in range(NUM_ALPHA):
#         p += alpha[k] * cheb_T(k, psi_norm)

#     plt.figure(figsize=(6,4))
#     plt.plot(psi, p, linewidth=2)
#     plt.xlabel("ψ")
#     plt.ylabel("p(ψ)")
#     plt.title("Pressure Profile")
#     plt.grid(True)
#     plt.show()


# def compare_pressure_profiles(alpha_list, labels=None):
#     psi = np.linspace(-1,0,400).reshape(-1,1)
#     psi_norm = normalize_psi(psi)

#     if labels is None:
#         labels = [f"Profile {i}" for i in range(len(alpha_list))]

#     plt.figure(figsize=(6,4))
#     for alpha, label in zip(alpha_list, labels):
#         p = np.zeros_like(psi)
#         for k in range(NUM_ALPHA):
#             p += alpha[k] * cheb_T(k, psi_norm)
#         plt.plot(psi, p, label=label)

#     plt.xlabel("ψ")
#     plt.ylabel("p(ψ)")
#     plt.grid(True)
#     plt.legend()
#     plt.title("Comparison of Pressure Profiles")
#     plt.show()


# def compare_flux_surfaces(R, Z, psiA, psiB, labels=("A","B")):
#     plt.figure(figsize=(12,5))

#     plt.subplot(121)
#     plt.contour(R, Z, psiA, levels=20)
#     plt.title(f"ψ for {labels[0]}")
#     plt.xlabel("R"); plt.ylabel("Z"); plt.axis("equal")

#     plt.subplot(122)
#     plt.contour(R, Z, psiB, levels=20)
#     plt.title(f"ψ for {labels[1]}")
#     plt.xlabel("R"); plt.ylabel("Z"); plt.axis("equal")

#     plt.tight_layout()
#     plt.show()


# def plot_error(R, Z, psi_true, psi_pred, rel=False):
#     if rel:
#         err = np.abs((psi_pred - psi_true) / (psi_true + 1e-8))
#         title="Relative Error"
#     else:
#         err = np.abs(psi_pred - psi_true)
#         title="Absolute Error"

#     plt.figure(figsize=(6,5))
#     plt.contourf(R, Z, err, 50, cmap='inferno')
#     plt.colorbar(label="Error")
#     plt.title(title)
#     plt.xlabel("R"); plt.ylabel("Z")
#     plt.axis("equal")
#     plt.show()






# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from pathlib import Path



# # You MUST have: NUM_ALPHA, normalize_psi, cheb_T imported or defined


# # ============================================================
# # Helper: ensure directory exists
# # ============================================================
# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


# # ============================================================
# # Save plot helper
# # ============================================================
# def savefig(path):
#     plt.savefig(path, dpi=200, bbox_inches='tight')
#     plt.close()


# # ============================================================
# # 1) Single-shot evaluation (one set of parameters)
# # ============================================================
# def evaluate_single(model, outdir, alpha, eps, kappa, delta):
#     ensure_dir(outdir)

#     print("\n[1] Evaluating single pressure/shape combination...")

#     # Compute ψ grid
#     R, Z, psi = evaluate_grid(model, alpha, eps, kappa, delta)

#     # Flux surfaces
#     plot_flux_surfaces(R, Z, psi, levels=40,
#                        title=f"Flux Surfaces: α={alpha}, ε={eps}, κ={kappa}, δ={delta}")
#     savefig(f"{outdir}/flux_surfaces_single.png")

#     # Pressure profile
#     plot_pressure_profile(alpha)
#     savefig(f"{outdir}/pressure_profile_single.png")

#     print(" → Saved single-evaluation images.")


# # ============================================================
# # 2) Pressure sweep images
# # ============================================================
# def evaluate_pressure_sweep(model, outdir, eps, kappa, delta,
#                             alpha_sweep):
#     ensure_dir(outdir)

#     print("\n[2] Pressure profile sweep...")

#     # Plot pressure profiles together
#     labels = [f"a{i}" for i in range(len(alpha_sweep))]
#     compare_pressure_profiles(alpha_sweep, labels)
#     savefig(f"{outdir}/pressure_profiles_compare.png")

#     # Plot flux surfaces for each α
#     for idx, alpha in enumerate(alpha_sweep):
#         print(f"    → Computing ψ for alpha set #{idx}: {alpha}")

#         R, Z, psi = evaluate_grid(model, alpha, eps, kappa, delta)

#         plot_flux_surfaces(
#             R, Z, psi, levels=40,
#             title=f"Flux Surfaces for α={alpha} (same shape)"
#         )
#         savefig(f"{outdir}/flux_alpha_{idx}.png")


# # ============================================================
# # 3) Shape sweep images
# # ============================================================
# def evaluate_shape_sweep(model, outdir, alpha, shape_list):
#     ensure_dir(outdir)

#     print("\n[3] Shape sweep evaluation...")

#     for idx, (eps, kappa, delta) in enumerate(shape_list):
#         print(f"   Shape #{idx}: eps={eps}, kappa={kappa}, delta={delta}")

#         R, Z, psi = evaluate_grid(model, alpha, eps, kappa, delta)

#         plot_flux_surfaces(
#             R, Z, psi, levels=40,
#             title=f"Flux Surfaces α={alpha}, eps={eps}, kappa={kappa}, delta={delta}"
#         )
#         savefig(f"{outdir}/flux_shape_{idx}.png")


# # ============================================================
# # 4) Grid comparison between two different α or two shapes
# # ============================================================
# def evaluate_comparison(model, outdir, alphaA, alphaB, eps, kappa, delta):
#     ensure_dir(outdir)

#     print("\n[4] Comparing two α configurations...")

#     R, Z, psiA = evaluate_grid(model, alphaA, eps, kappa, delta)
#     R, Z, psiB = evaluate_grid(model, alphaB, eps, kappa, delta)

#     compare_flux_surfaces(R, Z, psiA, psiB,
#                           labels=("A: "+str(alphaA), "B: "+str(alphaB)))
#     savefig(f"{outdir}/flux_compare_AB.png")

#     # Error
#     plot_error(R, Z, psiA, psiB, rel=True)
#     savefig(f"{outdir}/error_AB_relative.png")


# # ============================================================
# # 5) Line‐cut comparison along R or Z for deeper inspection
# # ============================================================
# def linecut_plot(model, outdir, alpha, eps, kappa, delta,
#                  line="midplane"):
#     ensure_dir(outdir)

#     print("\n[5] Generating line-cuts...")

#     R, Z, psi = evaluate_grid(model, alpha, eps, kappa, delta)

#     if line == "midplane":
#         idx = np.argmin(np.abs(Z[:,0]))  # find Z=0 row
#         psi_line = psi[idx,:]
#         x = R[0,:]
#         xlabel = "R"

#     elif line == "axis":
#         idx = np.argmin(np.abs(R[0,:] - 1))  # R=1 cut
#         psi_line = psi[:,idx]
#         x = Z[:,0]
#         xlabel = "Z"

#     else:
#         raise ValueError("line must be 'midplane' or 'axis'")

#     plt.figure(figsize=(6,4))
#     plt.plot(x, psi_line, linewidth=2)
#     plt.xlabel(xlabel)
#     plt.ylabel("ψ")
#     plt.grid(True)
#     plt.title(f"Linecut ({line}) for α={alpha}")
#     savefig(f"{outdir}/linecut_{line}.png")

#     print(" → Linecut saved.")


# # ============================================================
# # MASTER: Run all evaluations
# # ============================================================
# def run_all_experiments(model, outdir):

#     ensure_dir(outdir)

#     # --------------------------
#     # Choose some example params
#     # --------------------------
#     alpha_single = [0.5, -0.3, 0.1, 0.0]
#     alpha_sweep = [
#         [0.5, -0.2, 0.0, 0.0],
#         [0.2, +0.1, 0.2, -0.1],
#         [-0.3, 0.2, 0.0, 0.0],
#     ]

#     shape_list = [
#         (0.30, 2.0, 0.0),
#         (0.35, 2.2, +0.2),
#         (0.25, 2.5, -0.3),
#     ]

#     eps = 0.32
#     kappa = 2.0
#     delta = 0.0

#     # -----------------------------------------------------
#     # Run each evaluation (creates dozens of images)
#     # -----------------------------------------------------
#     evaluate_single(
#         model, f"{outdir}/01_single",
#         alpha_single, eps, kappa, delta
#     )

#     evaluate_pressure_sweep(
#         model, f"{outdir}/02_pressure_sweep",
#         eps, kappa, delta,
#         alpha_sweep
#     )

#     evaluate_shape_sweep(
#         model, f"{outdir}/03_shape_sweep",
#         alpha_single, shape_list
#     )

#     evaluate_comparison(
#         model, f"{outdir}/04_comparison",
#         alpha_sweep[0], alpha_sweep[1], eps, kappa, delta
#     )

#     linecut_plot(
#         model, f"{outdir}/05_linecuts",
#         alpha_single, eps, kappa, delta, line="midplane"
#     )

#     print("\n✔ ALL evaluation images created!")


# out_path = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/evaluation_profiles"
# run_all_experiments(model, out_path)
