
import os
from typing import Callable, Sequence, Tuple
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import sys
deepxde_path = '/scratch/yx3044/Projects/deepxde_copy'
# This ensures it's searched before the system packages
if deepxde_path not in sys.path:
    sys.path.insert(0, deepxde_path)
import deepxde as dde
print("Using DeepXDE from:", dde.__file__)
sys.path.append('/scratch/yx3044/Projects/deepxde_copy/gs-2d-surrogate')

dde.config.set_default_float("float64")
from shape_optimization import *


def check_normalization(model, Amax, eps0, kappa0, delta0, A=-0.155):
    num_param = 6
    Arange = np.linspace(-Amax, Amax, num_param)
    eps = np.linspace(eps0[0], eps0[1], num_param)
    kappa = np.linspace(kappa0[0], kappa0[1], num_param)
    delta = np.linspace(delta0[0], delta0[1], num_param)
    x, u = gen_traindata(1)
    psi_bnd = model.predict(x).flatten()
    print("ψ on analytic boundary: min =", psi_bnd.min(),
          "max =", psi_bnd.max(), "mean =", psi_bnd.mean())
    return psi_bnd


if __name__ == "__main__":
    CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09152025_2103_parametrized/ITER-16001.ckpt"
    LR = 2e-2
    AF = "swish"
    DEPTH = 4
    BREADTH = 40
    Amax = 0.2
    eps0 = (0.32 - eps_deviation, 0.32 + eps_deviation)
    kappa0 = (2 - kappa_deviation, 2 + kappa_deviation)
    delta0 = (0 - delta_deviation, 0 + delta_deviation)
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
    check_normalization(model, Amax, eps0, kappa0, delta0)