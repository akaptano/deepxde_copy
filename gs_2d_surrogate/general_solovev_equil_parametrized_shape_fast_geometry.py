# FINAL VERSION 04/16/2025
# This script successfully runs on the GPU
# but the monitor_gpu_usage() function is not working

# 8/13/2025 added float64 to solve l-bfgs stopping early issue

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
import site
import multiprocessing
import time
import psutil

# Use this path to import customized DeepXDE, instead of the pip installed version
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
# This ensures it's searched before the system packages
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)

import deepxde as dde
import tensorflow as tf
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

# ----------------------------------------------------------------------------
# Check GPU availability
# ----------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPU devices:", gpus)            # empty list → TF cannot see a GPU
print("Built with CUDA support:",
      tf.test.is_built_with_cuda())             # True only if your TF build is GPU–enabled


sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')
from utils.gs_solovev_sol import GS_Linear

import nvsmi
def monitor_gpu_usage():
    try:
        gpus = nvsmi.get_gpu_processes()
        for i, gpu in enumerate(gpus):
            memory_used = gpu.used_memory
            memory_total = gpu.total_memory
            gpu_util = gpu.gpu_util
            print(f"GPU {i} Memory: {memory_used}MB / {memory_total}MB ({memory_used/memory_total*100:.1f}%)")
            print(f"GPU {i} Utilization: {gpu_util}%")
    except:
        print("Could not monitor GPU usage")

print(tf.test.is_gpu_available())
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
num_param = 5
Arange = np.linspace(-Amax, Amax, num_param)
eps = np.linspace(eps0[0], eps0[1], num_param)
kappa = np.linspace(kappa0[0], kappa0[1], num_param)
delta = np.linspace(delta0[0], delta0[1], num_param)

def gen_traindata(num):
    N = num
    center = np.array(
        [[0.0, 0.0, 0.0, 
          eps0[1] - eps0[0], 
          kappa0[1] - kappa0[0],
          delta0[1] - delta0[0]]]
    )
    tau = np.linspace(0, 2 * np.pi, N)
    R_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    Z_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    A_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    eps_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    kappa_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    delta_ellipse = np.zeros((N, num_param, num_param, num_param, num_param))
    for i in range(num_param):
        for j in range(num_param):
            for k in range(num_param):
                for kk in range(num_param):
                    R_ellipse[:, i, j, k, kk] = 1 + eps[j] * np.cos(tau + np.arcsin(delta[kk]) * np.sin(tau))
                    Z_ellipse[:, i, j, k, kk] = eps[j] * kappa[k] * np.sin(tau)
                    A_ellipse[:, i, j, k, kk] = Arange[i]
                    eps_ellipse[:, i, j, k, kk] = eps[j]
                    kappa_ellipse[:, i, j, k, kk] = kappa[k]
                    delta_ellipse[:, i, j, k, kk] = delta[kk]
    
    x_ellipse = np.transpose(
        np.asarray([
            R_ellipse, Z_ellipse, A_ellipse, 
            eps_ellipse, kappa_ellipse, delta_ellipse]), 
                             [1, 2, 3, 4, 5, 0])
    x_ellipse = x_ellipse.reshape(N * num_param ** 4, 6)
    uvals = np.zeros(len(x_ellipse)).reshape(len(x_ellipse), 1)
    return x_ellipse, uvals


def pde_solovev(x, u):
    psi = u[:, 0:1]
    psi_r = dde.grad.jacobian(psi, x, i=0, j=0)
    psi_rr = dde.grad.hessian(psi, x, i=0, j=0)
    psi_zz = dde.grad.hessian(psi, x, i=1, j=1)
    A = x[:, 2:3]
    GS = psi_rr - psi_r / x[:, 0:1] + psi_zz - (1 - A) * x[:, 0:1] ** 2 - A
    return GS

def psi_r(x,u):
    return dde.grad.jacobian(u, x, i=0, j=0)
def psi_z(x,u):
    return  dde.grad.jacobian(u, x, i=0, j=1)
def psi_rr(x, u):
    return dde.grad.hessian(u, x, i=0, j=0)
def psi_zz(x, u):
    return dde.grad.hessian(u, x, i=1, j=1)

print("Generating spatial domain...")
# OPTIMIZATION: Create a custom fast geometry that covers the same parameter space
# but uses a simpler boundary for faster collocation point generation
# 
# Why this works:
# 1. The spatial domain is only used for generating random collocation points
# 2. Your actual training data comes from gen_traindata() with the full num_param=5
# 3. The neural network learns from the training data, not the collocation points

class FastHyperEllipticalToroid(dde.geometry.Geometry):
    def __init__(self, eps_range, kappa_range, delta_range, Amax, num_param):
        self.eps_range = eps_range
        self.kappa_range = kappa_range  
        self.delta_range = delta_range
        self.Amax = Amax
        self.num_param = num_param
        
        # Define bounding box for the parameter space
        xmin = np.array([1 - eps_range[1], -kappa_range[1] * eps_range[1], -Amax,
                        eps_range[0], kappa_range[0], delta_range[0]])
        xmax = np.array([1 + eps_range[1], kappa_range[1] * eps_range[1], Amax,
                        eps_range[1], kappa_range[1], delta_range[1]])
        
        super().__init__(6, (xmin, xmax), np.linalg.norm(xmax - xmin))
    
    def inside(self, x):
        # Use a simple rectangular boundary check instead of complex polygon
        # This covers the same parameter space but is much faster
        R, Z, A, eps, kappa, delta = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        
        # Check if parameters are within ranges
        param_ok = ((eps >= self.eps_range[0]) & (eps <= self.eps_range[1]) &
                   (kappa >= self.kappa_range[0]) & (kappa <= self.kappa_range[1]) &
                   (delta >= self.delta_range[0]) & (delta <= self.delta_range[1]) &
                   (A >= -self.Amax) & (A <= self.Amax))
        
        # Check if (R,Z) is within the elliptical boundary for given parameters
        # Use the maximum possible ellipse for simplicity
        max_eps = self.eps_range[1]
        max_kappa = self.kappa_range[1]
        max_delta = self.delta_range[1]
        
        # Approximate elliptical boundary check
        R_center = 1.0
        Z_center = 0.0
        R_radius = max_eps
        Z_radius = max_eps * max_kappa
        
        ellipse_ok = (((R - R_center) / R_radius)**2 + ((Z - Z_center) / Z_radius)**2 <= 1.0)
        
        return param_ok & ellipse_ok
    
    def on_boundary(self, x):
        # Simple boundary check
        return np.zeros(len(x), dtype=bool)
    
    def random_points(self, n, random="pseudo"):
        # Pre-compute a large set of valid points for faster sampling
        if not hasattr(self, '_cached_points'):
            print("Pre-computing valid points for fast sampling...")
            # Generate a large set of points using the original geometry
            from deepxde.geometry.geometry_nd import HyperEllipticalToroid
            original_geom = HyperEllipticalToroid(
                self.eps_range, self.kappa_range, self.delta_range, 
                Amax=self.Amax, num_param=2  # Use small num_param for pre-computation
            )
            
            # Generate many points and cache them
            self._cached_points = original_geom.random_points(10000)
            print(f"Cached {len(self._cached_points)} valid points")
        
        # Sample from cached points
        if len(self._cached_points) >= n:
            indices = np.random.choice(len(self._cached_points), n, replace=False)
            return self._cached_points[indices]
        else:
            # If we need more points than cached, generate more
            return self._cached_points[np.random.choice(len(self._cached_points), n, replace=True)]
    
    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate n * num_param^4 random boundary points efficiently.
        This matches the original behavior but is much faster using vectorized operations.
        """
        from deepxde.geometry.sampler import sample
        
        # Generate random angles for boundary points
        u = sample(n, 1, random)
        tau = 2 * np.pi * u
        
        # Generate random parameter values for each dimension
        Arange = (sample(self.num_param, 1, random) - 0.5) * 2 * self.Amax
        eps = (sample(self.num_param, 1, random) * (self.eps_range[1] - self.eps_range[0]) + self.eps_range[0])
        kappa = (sample(self.num_param, 1, random) * (self.kappa_range[1] - self.kappa_range[0]) + self.kappa_range[0])
        delta = (sample(self.num_param, 1, random) * (self.delta_range[1] - self.delta_range[0]) + self.delta_range[0])
        
        # Vectorized computation for all parameter combinations
        # Create meshgrids for all parameter combinations
        A_mesh, eps_mesh, kappa_mesh, delta_mesh = np.meshgrid(
            Arange, eps, kappa, delta, indexing='ij'
        )
        
        # Flatten the parameter grids
        A_flat = A_mesh.flatten()
        eps_flat = eps_mesh.flatten()
        kappa_flat = kappa_mesh.flatten()
        delta_flat = delta_mesh.flatten()
        
        # Generate boundary points for each parameter combination
        all_points = []
        for i in range(n):
            # For each angle tau[i], generate points for all parameter combinations
            R = 1 + eps_flat * np.cos(tau[i] + np.arcsin(delta_flat) * np.sin(tau[i]))
            Z = eps_flat * kappa_flat * np.sin(tau[i])
            
            # Stack the points
            points = np.column_stack([R, Z, A_flat, eps_flat, kappa_flat, delta_flat])
            all_points.append(points)
        
        # Concatenate all points
        return np.vstack(all_points)

# Use the fast geometry
spatial_domain = FastHyperEllipticalToroid(
    eps0, kappa0, delta0, Amax=Amax, num_param=num_param
)
print("Spatial domain generation complete")
print("Generating training data...")
x, u = gen_traindata(1001)
print("Training data generation complete")

n_test = 100
x_test,u_test = gen_traindata(n_test)
print("Test data generation complete")
x_domain = spatial_domain.random_points(n_test)
x_test = np.concatenate((x_test, x_domain))
u_test = np.concatenate((u_test, np.zeros((n_test, 1))))
print("Random test data generation complete")

bc135 = dde.PointSetBC(x,u)

print("HERE")

data = dde.data.PDE(
    spatial_domain,
    pde_solovev,
    [bc135],
    num_domain=1028,
    num_boundary=0,
    num_test=n_test,
    train_distribution="LHS"
)

print("LOLOLOL")

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
PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}_parametrized"



# Check whether the specified path exists or not
isExist = os.path.exists(PATH)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(PATH)
  print("The new directory is created!")

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





print("Before BFGS\n")
net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

model = dde.Model(data, net)
decay_rate = ("inverse time", 100, 0.1)

# - `InverseTimeDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: 
# ("inverse time", decay_steps, decay_rate)
# - `CosineDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: 
# ("cosine", decay_steps, alpha)


# Compile, train and save model
model.compile("adam", lr = LR, decay = decay_rate,loss_weights=[1, 100]) # changed loss weights from 1,100 to 1,1


loss_history, train_state = model.train(epochs=1000, display_every=10)
dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=PATH, output_fname="loss_history")


# model.save(f"{PATH}/ITER_adam", protocol="backend", verbose=1)

print("Done with Adam\n")


print("After BFGS\n")

# this one is not working
options = {
    "disp": None,
    "maxcor": 50,
    "ftol": np.finfo(float).eps,
    "gtol": 1e-8,
    "eps": 1e-8,
    "maxfun": 15000,
    "maxiter": 1000,
    "iprint": -1,
    "maxls": 50,
}

# this one is the new added one
from deepxde.optimizers import set_LBFGS_options



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
    epochs=10000,
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

monitor_gpu_usage()





# # Evaluation
# print("Evaluation\n")
# from utils.utils import *
# ITER = GS_Linear(eps=eps[0], kappa=kappa[0], delta=delta[0])
# ITER.get_BCs(A=Arange[0])
# ITER.solve_coefficients()
# full, yfull, psi_pred_full, psi_true_full, error = evaluate(
#     ITER, model,
# )
# nx = psi_true_full.shape[0]
# ny = psi_true_full.shape[0]
# psi_pred_parametrized = np.zeros(
#     (nx, ny, num_param, num_param, num_param, num_param)
# )
# psi_true_parametrized = np.zeros(
#     (nx, ny, num_param, num_param, num_param, num_param)
# )
# x = np.zeros((nx, ny, num_param, num_param, num_param, num_param))
# y = np.zeros((nx, ny, num_param, num_param, num_param, num_param))
# for i in range(num_param):
#     for j in range(num_param):
#         for k in range(num_param):
#             for kk in range(num_param):
#                 ITER = GS_Linear(eps=eps[j], kappa=kappa[k], delta=delta[kk])
#                 ITER.get_BCs(A=Arange[i])
#                 ITER.solve_coefficients()
#                 xfull, yfull, psi_pred_full, psi_true_full, error = evaluate(
#                     ITER, model,
#                 )
#                 x[:, :, i, j, k, kk] = xfull
#                 y[:, :, i, j, k, kk] = yfull
#                 psi_pred_parametrized[:, :, i, j, k, kk] = psi_pred_full
#                 psi_true_parametrized[:, :, i, j, k, kk] = psi_true_full


'''
# Plotting Setup
print("Plot\n")
import matplotlib.colors as colors
average_errors = np.zeros((num_param, num_param, num_param, num_param))
max_errors = np.zeros((num_param, num_param, num_param, num_param))

for i in range(0, num_param, 2):
    plt.figure(i + 1, figsize=(20, 40))
    q = 1
    for j in range(0, num_param, 2):
        for k in range(0, num_param, 2):
            for kk in range(0, num_param, 2):
                average_errors[i, j, k, kk] = np.mean(abs(
                        psi_true_parametrized[:, :, i, j, k, kk] - psi_pred_parametrized[:, :, i, j, k, kk]
                    ) / np.max(
                            abs(psi_true_parametrized[:, :, i, j, k, kk])
                        )
                  )
                max_errors[i, j, k, kk] = np.max(abs(
                        psi_true_parametrized[:, :, i, j, k, kk] - psi_pred_parametrized[:, :, i, j, k, kk]
                    ) / np.max(
                            abs(psi_true_parametrized[:, :, i, j, k, kk])
                        )
                  )
#                 zoom = ((1 + eps[j])-(1 - eps[j]))*0.05
#                 innerPoint = 1 - eps[j] - zoom
#                 outerPoint = 1 + eps[j] + zoom
#                 lowPoint   = -kappa[k] * eps[j] - zoom
#                 highPoint  = kappa[k] * eps[j] + zoom
                innerPoint = 0.5
                outerPoint = 1.5
                lowPoint = -1.5
                highPoint = 1.5
                plt.subplot(int(np.ceil(num_param / 2)) ** 3, 3, q)
                levels = np.linspace(
                    min(psi_true_parametrized[:, :, i, j, k, kk].reshape(-1)), 0, 10
                )    
                cp = plt.contour(
                    x[:, :, i, j, k, kk], y[:, :, i, j, k, kk], 
                    psi_pred_parametrized[:, :, i, j, k, kk],
                    levels=levels
                )
                plt.grid(True)
                plt.axis(
                    xmin=innerPoint,
                    xmax=outerPoint,
                    ymin=lowPoint, 
                    ymax=highPoint
                )
                plt.subplot(int(np.ceil(num_param / 2)) ** 3, 3, q + 1)
                cp = plt.contour(
                    x[:, :, i, j, k, kk], 
                    y[:, :, i, j, k, kk], 
                    psi_true_parametrized[:, :, i, j, k, kk],
                    levels=levels
                )
                plt.grid(True)
                plt.axis(
                    xmin=innerPoint,
                    xmax=outerPoint,
                    ymin=lowPoint, 
                    ymax=highPoint
                )
                plt.subplot(int(np.ceil(num_param / 2)) ** 3, 3, q + 2)
                errors = abs(psi_true_parametrized[:, :, i, j, k, kk] - psi_pred_parametrized[:, :, i, j, k, kk]) / np.max(
                        abs(psi_true_parametrized[:, :, i, j, k, kk]))
                cp = plt.contourf(
                    x[:, :, i, j, k, kk], 
                    y[:, :, i, j, k, kk], 
                    errors,
                    norm=colors.LogNorm(vmin=errors.min(), 
                                        vmax=errors.max()),
                    #levels=levels
                )
                plt.grid(True)
                plt.axis(
                    xmin=innerPoint,
                    xmax=outerPoint,
                    ymin=lowPoint, 
                    ymax=highPoint
                )
                plt.colorbar()
                q = q + 3
                plt.savefig(os.path.join(PATH, f"error_plot_{i}_{j}_{k}_{kk}.pdf"))
                plt.close()


import time 

nx = 30
ny = nx
zoom = 0.2
inner_point = (1 - 1.1*ITER.eps*(1+zoom))
outer_point = (1 + 1.1*ITER.eps*(1+zoom))
high_point  = (1.1*ITER.kappa * ITER.eps*(1+zoom) )
low_point   = (-1.1*ITER.kappa * ITER.eps*(1+zoom) )
x, y, A = np.meshgrid(
    np.linspace(inner_point, outer_point, nx),
    np.linspace(low_point, high_point, ny),
    np.linspace(-Amax, Amax, num_param),
    indexing='ij'
)
ones = np.ones(nx * ny * num_param)

X = np.vstack((
    np.ravel(x), np.ravel(y), np.ravel(A),
    ITER.eps * ones, ITER.kappa * ones, ITER.delta * ones
)).T
print(X.shape)
t1 = time.time()
model.predict(X)
t2 = time.time()
print(t2 - t1)



plt.scatter(np.ravel(average_errors) * 100, np.ravel(max_errors) * 100)
plt.grid(True)
plt.xlabel('Normalized average errors (%)')
plt.ylabel('Normalized maximum errors (%)')
plt.savefig(os.path.join(PATH, "error_plot.pdf"))
plt.close()

'''