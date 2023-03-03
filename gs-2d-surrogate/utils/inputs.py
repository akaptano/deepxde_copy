from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .gs_solovev_sol import GS_Linear
import deepxde as dde

######################
# ITER Configuration #
######################
A = -0.155
eps = 0.32
kappa = 1.7
delta = 0.33

#######################
# Model Configuration #
#######################
n_domain = 1000
n_boundary = 1000
n_test = 200
AF = "swish"
LR = 2e-2 # for BFGS this doesn't apply
DEPTH = 3
BREADTH = 20
LOSSRATIO = 100 # Domain vs. Boudanry loss weight. This ratio is assuming domain loss weight is zero.
DROPOUT = 0.1

######################
# Optimization PARAM #
######################
PARAM = "sample_size"
sample_sizes = 5*np.logspace(1, 3, 6, endpoint=True).astype(int)
n_domain = sample_sizes
n_boundary = sample_sizes

######################
# File Configuration #
######################
DATE = "03032023"
CONFIG = "ITER"
OPTIMIZER = f"{LOSSRATIO}BFGS"
run = "01"
DIR = f"./sweep{run}_{DATE}_{CONFIG}_{OPTIMIZER}/{PARAM}/"



N1 = - (1 + np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
N2 = (1 - np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
N3 = - kappa / (eps * np.cos(np.arcsin(delta)) ** 2)


def gen_traindata(num,eps, kappa, delta):
    ######################
    # ITER Configuration #
    ######################
    N = num
    center, eps, kappa, delta = np.array([[0.0,0.0]]), eps, kappa, delta
    tau = np.linspace(0, 2 * np.pi, N)
    # Define boundary of ellipse
    x_ellipse = np.asarray([1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)), 
                    eps * kappa * np.sin(tau)]).T
    xvals = x_ellipse
    uvals = np.zeros(len(xvals)).reshape(len(xvals), 1)
    return xvals, uvals

def pde_solovev(x, u):
    psi = u[:, 0:1]
    psi_r = dde.grad.jacobian(psi, x, i=0, j=0)
    psi_rr = dde.grad.hessian(psi, x, i=0, j=0)
    psi_zz = dde.grad.hessian(psi, x, i=1, j=1)
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

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 + eps, 0]).all()
def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 - eps, 0]).all()
def boundary_high(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 - delta * eps, kappa * eps]).all()


# Test function for module  
def _test():
    assert int("1") == 1

if __name__ == '__main__':
    _test()