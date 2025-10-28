import numpy as np
import matplotlib.pyplot as plt
import time
from utils.gs_solovev_sol import GS_Linear
from shape_optimization import predict_psi
from shape_optimization import pde_solovev, psi_r, psi_z, psi_rr, psi_zz

import sys
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
# This ensures it's searched before the system packages
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)
import deepxde as dde
print("Using DeepXDE from:", dde.__file__)
sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')
dde.config.set_default_float("float64")


def check_boundary_prediction(model, eps, kappa, delta, n_boundary=400):

    tau = np.linspace(0.0, 2*np.pi, n_boundary, endpoint=False)
    x_anal = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    y_anal = eps * kappa * np.sin(tau)
    
    R, Z, psi_pred_grid, psi_true_grid = predict_psi(
        model, eps=eps, kappa=kappa, delta=delta, plot_psi=False
    )
    
    # Extract psi=0 contour
    c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
    if c.collections and c.collections[0].get_paths():
        vertices = c.collections[0].get_paths()[0].vertices
        x_pred = vertices[:,0]
        y_pred = vertices[:,1]
    else:
        print("Warning: No closed contour found in prediction")
        x_pred = []
        y_pred = []
    plt.close()


    plt.figure(figsize=(8,8))
    plt.plot(x_anal, y_anal, 'b-', label='Analytic')
    plt.plot(x_pred, y_pred, 'r--', label='PINN')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'Plasma Boundary Comparison\nε={eps:.3f}, κ={kappa:.3f}, δ={delta:.3f}')
    
    timestamp = time.strftime("%m%d_%H%M%S")
    plt.savefig(f'boundary_comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()




eps=7.825e-02
kappa=1.719e+00
delta=3.117e-01

eps = 0.32
kappa = 1.7
delta = 0.33

eps = 0.17
kappa = 1.5
delta = -0.52


A = -0.155
eps_deviation = 0.2
kappa_deviation = 0.2
delta_deviation = 0.2
eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
kappa0 = (1.7 - kappa_deviation, 1.7 + kappa_deviation)
delta0 = (0.33 - delta_deviation, 0.33 + delta_deviation)
Amax = 0.2
num_param = 5
Arange = np.linspace(-Amax, Amax, num_param)

CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_07092025_2333_parametrized/ITER-539.ckpt"
CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_08142025_0149_parametrized/ITER-388.ckpt"
CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_08142025_1323_parametrized/ITER-11293.ckpt"
CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_08142025_2320_parametrized/ITER-16001.ckpt"

LR = 2e-3
AF = "swish"
DEPTH = 4
BREADTH = 40
Amax = 0.2

net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

spatial_domain = dde.geometry.HyperEllipticalToroid(
    eps0, kappa0, delta0, Amax=Amax
) 
data = dde.data.PDE(
    spatial_domain,
    pde_solovev,
    [],  # No BCs needed for inference
    num_domain=1,  # Minimal points needed
    num_boundary=0,
    num_test=1
)
model = dde.Model(data=data, net=net)
model.compile("adam", lr=LR, loss_weights=[1,100])
model.restore(CHECKPOINT_PATH, verbose=1)

check_boundary_prediction(model, eps, kappa, delta)