import numpy as np
import matplotlib.pyplot as plt
import time
import os
from utils.gs_solovev_sol import GS_Linear
from shape_optimization import predict_psi, gen_traindata
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

from utils.utils import *


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


run_name = "run_07092025_2333_parametrized/ITER-539"
run_name = "run_08142025_0149_parametrized/ITER-388"
run_name = "run_08142025_1323_parametrized/ITER-11293"
# run_name = "run_08142025_2320_parametrized/ITER-16001"

# after modified spatial domain
run_name = "run_09092025_2320_parametrized/ITER-16001"  # param 2
run_name = "run_09102025_0206_parametrized/ITER-16001"  # param 3
run_name = "run_09102025_0818_parametrized/ITER-16001"  # param 4
run_name = "run_09102025_0843_parametrized/ITER-16001"  # param 5
CHECKPOINT_PATH = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/{run_name}.ckpt"

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



# x,u = gen_traindata(1001)
# n_test = 100

# x_test,u_test = gen_traindata(n_test)
# x_domain = spatial_domain.random_points(n_test)
# x_test = np.concatenate((x_test, x_domain))
# u_test = np.concatenate((u_test, np.zeros((n_test, 1))))

# bc135 = dde.PointSetBC(x,u)

# data = dde.data.PDE(
#     spatial_domain,
#     pde_solovev,
#     [bc135],
#     num_domain=1028,
#     num_boundary=0,
#     num_test=n_test,
#     train_distribution="LHS"
# )


model = dde.Model(data=data, net=net)
model.compile("adam", lr=LR, loss_weights=[1,100])
model.restore(CHECKPOINT_PATH, verbose=1)





save_path = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/check_plot_{run_name}_2"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)
print(f"Created/verified directory: {save_path}")

ITER = GS_Linear(eps=eps, kappa=kappa, delta=delta)
# for i in range(num_param):
#     ITER.get_BCs(A=Arange[i])
#     ITER.solve_coefficients()
#     xfull, yfull, psi_pred_full, psi_true_full, error = evaluate(ITER, model)
#     X_test = spatial_domain.random_points(333)
#     plot_summary_figure_kaltsas(ITER, model, X_test, save_path)
#     plt.savefig(f'{save_path}/check_plot_{i}.png')
#     plt.close()

ITER.get_BCs(A=Arange[0])
ITER.solve_coefficients()
xfull, yfull, psi_pred_full, psi_true_full, error = evaluate(ITER, model)
X_test = spatial_domain.random_points(333)
plot_summary_figure_kaltsas(ITER, model, X_test, save_path)
plt.savefig(f'{save_path}/check_plot_{0}.png')
plt.close()