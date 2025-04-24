# # %%

# FINAL VERSION 04/16/2025
# This script successfully runs on the GPU
# but the monitor_gpu_usage() function is not working

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
print(tf.config.list_physical_devices('GPU'))



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


spatial_domain = dde.geometry.HyperEllipticalToroid(
    eps0, kappa0, delta0, Amax=Amax
) 

x, u = gen_traindata(1000)

n_test = 100
x_test,u_test = gen_traindata(n_test)
x_domain = spatial_domain.random_points(n_test)
x_test = np.concatenate((x_test, x_domain))
u_test = np.concatenate((u_test, np.zeros((n_test, 1))))

bc135 = dde.PointSetBC(x,u)

data = dde.data.PDE(
    spatial_domain,
    pde_solovev,
    [bc135],
    num_domain=1028,
    num_boundary=0,
    num_test=n_test,
    train_distribution="LHS"
)

# # %%

import time
DATE = time.strftime("%m%d%Y_%H%M")
CONFIG = "ITER"
LR = 1e-3
DEPTH = 4
BREADTH = 40
run = "01_100Adam_BFGS"
AF = "swish"
RUN_NAME = f"network_sweep_{DATE}_depth0{DEPTH}_breadth{BREADTH}_{AF}_lr{LR}-varying-short_lw1-10_{run}"
PATH = f"./cefron/{CONFIG}/runs/{RUN_NAME}"



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




# %%


print("Before BFGS\n")
net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

model = dde.Model(data, net)
decay_rate = ("inverse time", 100, 0.1)

# - `InverseTimeDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: 
# ("inverse time", decay_steps, decay_rate)
# - `CosineDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: 
# ("cosine", decay_steps, alpha)


# Compile, train and save model
model.compile("adam", lr = LR, decay = decay_rate,loss_weights=[1, 100])
loss_history, train_state = model.train(epochs=100, display_every=10)
dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=PATH, output_fname="loss_history")

monitor_gpu_usage()

# %%


print("After BFGS\n")


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

# Compile, train and save model
model.compile("L-BFGS-B", loss_weights=[1, 100])

loss_history, train_state = model.train(
    epochs=1,
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

monitor_gpu_usage()


# %%


# Evaluation
print("Evaluation\n")
from utils.utils import *
ITER = GS_Linear(eps=eps[0], kappa=kappa[0], delta=delta[0])
ITER.get_BCs(A=Arange[0])
ITER.solve_coefficients()
full, yfull, psi_pred_full, psi_true_full, error = evaluate(
    ITER, model,
)
nx = psi_true_full.shape[0]
ny = psi_true_full.shape[0]
psi_pred_parametrized = np.zeros(
    (nx, ny, num_param, num_param, num_param, num_param)
)
psi_true_parametrized = np.zeros(
    (nx, ny, num_param, num_param, num_param, num_param)
)
x = np.zeros((nx, ny, num_param, num_param, num_param, num_param))
y = np.zeros((nx, ny, num_param, num_param, num_param, num_param))
for i in range(num_param):
    for j in range(num_param):
        for k in range(num_param):
            for kk in range(num_param):
                ITER = GS_Linear(eps=eps[j], kappa=kappa[k], delta=delta[kk])
                ITER.get_BCs(A=Arange[i])
                ITER.solve_coefficients()
                xfull, yfull, psi_pred_full, psi_true_full, error = evaluate(
                    ITER, model,
                )
                x[:, :, i, j, k, kk] = xfull
                y[:, :, i, j, k, kk] = yfull
                psi_pred_parametrized[:, :, i, j, k, kk] = psi_pred_full
                psi_true_parametrized[:, :, i, j, k, kk] = psi_true_full



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