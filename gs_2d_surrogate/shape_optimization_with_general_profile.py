"""cd gs_2d_surrogate/
source /scratch/yx3044/Projects/deepxde_copy/venv/bin/activate
conda activate deepxde_copy
python -u shape_optimization_with_general_profile.py  --profile solovev --lambda_vol=125 --zoom=1.2 --plot True >> shape_solovev_saved_metric_param6.txt
python -u shape_optimization_with_general_profile.py  --profile solovev --lambda_vol=125 --zoom=1.2 --objective_type beta_p --plot True >> shape_solovev_beta_p_saved_metric_param6.txt
python -u shape_optimization_with_general_profile.py  --profile pedestal --plot True >> shape_pedestal_saved_metric_param6.txt
python -u shape_optimization_with_general_profile.py  --profile pedestal --objective_type beta_p --plot True >> shape_pedestal_beta_p_saved_metric_param6.txt


Choose objective type:
1. beta_p
2. volume
3. beta_p_and_qstar


Debugging:
1. Error:
vertices = c.collections[0].get_paths()[0].vertices
IndexError: list index out of range

Solution:
try using a larger zoom, e.g. zoom=2.2

2. Error: AttributeError: `dense` is not available with Keras 3.

Solution:
export TF_USE_LEGACY_KERAS=1 
or
pip install tensorflow[and-cuda]==2.15.0

"""

import time
import os
import json
import itertools
from typing import Callable, Sequence, Tuple
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_USE_LEGACY_KERAS"] = "1"
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

import argparse
import matplotlib.pyplot as plt
from matplotlib.path import Path

dde.config.set_default_float("float64")
tf.keras.backend.set_floatx("float64")
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

# ----------------------------
# Global store for optimisation diagnostics
# ----------------------------
metrics = {
    "beta_p_pred": [],  # predicted β_p each function evaluation
    "volume_pred": [],  # predicted volume each function evaluation
    "qstar_pred": [],  # predicted qstar each function evaluation
    "eps": [],          # ε value proposed
    "kappa": [],        # κ value proposed
    "delta": [],         # δ value proposed
    "A": [],            # A value proposed
    "alpha": [],        # pressure profile coefficients
    "obj": []           # objective values
}

from utils.utils import *
from utils.gs_solovev_sol import GS_Linear



# ----------------------------------------------------------------------------
# ITER Configuration
# ----------------------------------------------------------------------------
A = -0.155
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
n_test = 100


def _gen_boundary_solovev(num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boundary points for the Solov'ev profile."""
    N = num
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
                    R_ellipse[:, i, j, k, kk] = 1 + eps[j] * np.cos(
                        tau + np.arcsin(delta[kk]) * np.sin(tau)
                    )
                    Z_ellipse[:, i, j, k, kk] = eps[j] * kappa[k] * np.sin(tau)
                    A_ellipse[:, i, j, k, kk] = Arange[i]
                    eps_ellipse[:, i, j, k, kk] = eps[j]
                    kappa_ellipse[:, i, j, k, kk] = kappa[k]
                    delta_ellipse[:, i, j, k, kk] = delta[kk]

    x_ellipse = np.transpose(
        np.asarray(
            [
                R_ellipse,
                Z_ellipse,
                A_ellipse,
                eps_ellipse,
                kappa_ellipse,
                delta_ellipse,
            ]
        ),
        [1, 2, 3, 4, 5, 0],
    )
    x_ellipse = x_ellipse.reshape(N * num_param**4, 6)
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


def _make_profile_boundary_generator(alpha_ranges: Sequence[np.ndarray]):
    """Create a boundary generator for general pressure profiles."""
    alpha_values = np.array(list(itertools.product(*alpha_ranges)), dtype=np.float64)
    num_alpha = len(alpha_ranges)

    def _generator(num_boundary_pts: int) -> Tuple[np.ndarray, np.ndarray]:
        N = num_boundary_pts
        tau = np.linspace(0, 2 * np.pi, N)

        R_list, Z_list = [], []
        alpha_list, eps_list, kappa_list, delta_list = [], [], [], []

        for eps_val in eps:
            for kappa_val in kappa:
                for delta_val in delta:
                    Rb = 1 + eps_val * np.cos(tau + np.arcsin(delta_val) * np.sin(tau))
                    Zb = eps_val * kappa_val * np.sin(tau)
                    eps_tile = np.full((N, 1), eps_val)
                    kappa_tile = np.full((N, 1), kappa_val)
                    delta_tile = np.full((N, 1), delta_val)

                    for alpha_vec in alpha_values:
                        R_list.append(Rb)
                        Z_list.append(Zb)
                        alpha_list.append(np.tile(alpha_vec.reshape(1, num_alpha), (N, 1)))
                        eps_list.append(eps_tile)
                        kappa_list.append(kappa_tile)
                        delta_list.append(delta_tile)

        R_flat = np.concatenate(R_list)[:, None]
        Z_flat = np.concatenate(Z_list)[:, None]
        alpha_flat = np.concatenate(alpha_list)
        eps_flat = np.concatenate(eps_list)
        kappa_flat = np.concatenate(kappa_list)
        delta_flat = np.concatenate(delta_list)

        x_boundary = np.hstack((R_flat, Z_flat, alpha_flat, eps_flat, kappa_flat, delta_flat))
        uvals = np.zeros((x_boundary.shape[0], 1))
        return x_boundary, uvals

    return _generator


def _polynomial_dp_dpsi(psi, alpha, num_alpha: int):
    """dp/dpsi for polynomial profiles."""
    dp = 0.0
    for k in range(1, num_alpha):
        dp += k * alpha[:, k:k+1] * tf.pow(psi, k - 1)
    return dp


def _build_polynomial_pde(num_alpha: int):
    def _pde(x, u):
        psi = u[:, 0:1]
        psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
        alpha = x[:, 2 : 2 + num_alpha]
        dp_dpsi = _polynomial_dp_dpsi(psi, alpha, num_alpha)
        GS = psi_RR - psi_R / x[:, 0:1] + psi_ZZ + (x[:, 0:1] ** 2) * dp_dpsi
        return GS

    return _pde


def _cheb_T(k: int, psi):
    if k == 0:
        return tf.ones_like(psi)
    if k == 1:
        return psi
    T0 = tf.ones_like(psi)
    T1 = psi
    for _ in range(2, k + 1):
        Tn = 2 * psi * T1 - T0
        T0, T1 = T1, Tn
    return T1


def _normalize_psi(psi, psi_min: float = -1.0, psi_max: float = 0.0):
    return 2.0 * (psi - psi_min) / (psi_max - psi_min) - 1.0


def _p_of_psi_cheb(psi, alpha, num_alpha: int):
    psi_norm = _normalize_psi(psi)
    p = 0.0
    for k in range(num_alpha):
        p += alpha[:, k:k+1] * _cheb_T(k, psi_norm)
    return p


def _dp_dpsi_cheb(psi, alpha, num_alpha: int):
    with tf.GradientTape() as tape:
        tape.watch(psi)
        p_val = _p_of_psi_cheb(psi, alpha, num_alpha)
    return tape.gradient(p_val, psi)


def _build_chebyshev_pde(num_alpha: int):
    def _pde(x, u):
        psi = u[:, 0:1]
        R = x[:, 0:1]
        psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
        alpha = x[:, 2 : 2 + num_alpha]
        dp_dpsi = _dp_dpsi_cheb(psi, alpha, num_alpha)
        GS = psi_RR - psi_R / R + psi_ZZ + R**2 * dp_dpsi
        return GS

    return _pde


def _build_polynomial_alpha_ranges(n: int) -> Sequence[np.ndarray]:
    return [
        np.linspace(-1.0, 1.0, n),
        np.linspace(-1.0, 1.0, n),
        np.linspace(-1.0, 1.0, n),
    ]


def _build_chebyshev_alpha_ranges(n: int) -> Sequence[np.ndarray]:
    return [
        np.linspace(-1.0, 1.0, n),
        np.linspace(-1.0, 1.0, n),
        np.linspace(-0.5, 0.5, n),
        np.linspace(-0.2, 0.2, n),
    ]


PEDESTAL_PSI = -0.08
PEDESTAL_WIDTH = 0.08


def _build_pedestal_alpha_ranges(n: int) -> Sequence[np.ndarray]:
    return [
        np.linspace(0.1, 0.5, n),  # pressure scale (edge pressure)
        np.linspace(1.5, 3.5, n),  # core/edge ratio
    ]


def _p_of_psi_pedestal(psi, alpha):
    """
    Fixed pedestal profile with bounded gradients.
    alpha[:, 0] = p_scale (edge pressure)
    alpha[:, 1] = ratio (core = p_scale * ratio)
    """
    p_scale = alpha[:, 0:1]
    ratio = alpha[:, 1:2]
    
    p_edge = p_scale
    p_core = p_scale * ratio
    
    # Smooth step (no extra steepness factor!)
    x = (psi - PEDESTAL_PSI) / PEDESTAL_WIDTH
    H = 0.5 * (1.0 + tf.tanh(x))
    
    return p_core * (1.0 - H) + p_edge * H


def _dp_dpsi_pedestal(psi, alpha):
    with tf.GradientTape() as tape:
        tape.watch(psi)
        p_val = _p_of_psi_pedestal(psi, alpha)
    return tape.gradient(p_val, psi)


def _build_pedestal_pde(num_alpha: int):
    def _pde(x, u):
        psi = u[:, 0:1]
        R = x[:, 0:1]
        psi_R = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_RR = dde.grad.hessian(psi, x, i=0, j=0)
        psi_ZZ = dde.grad.hessian(psi, x, i=1, j=1)
        alpha = x[:, 2 : 2 + num_alpha]
        dp_dpsi = _dp_dpsi_pedestal(psi, alpha)
        GS = psi_RR - psi_R / R + psi_ZZ + R**2 * dp_dpsi
        return GS

    return _pde


PROFILE_CONFIG = None


def build_profile_config(profile_name: str) -> dict:
    profile = profile_name.lower()
    if profile == "solovev":
        return {
            "name": profile,
            "uses_A": True,
            "num_alpha": 0,
            "alpha_ranges": None,
            "alpha_bounds": [],
            "alpha_default": np.array([], dtype=np.float64),
            "input_dim": 6,
            "pde_fn": pde_solovev,
            "boundary_generator": _gen_boundary_solovev,
            "alpha_labels": [],
        }
    if profile == "polynomial":
        alpha_ranges = _build_polynomial_alpha_ranges(num_param)
        alpha_bounds = [(arr.min(), arr.max()) for arr in alpha_ranges]
        return {
            "name": profile,
            "uses_A": False,
            "num_alpha": len(alpha_ranges),
            "alpha_ranges": alpha_ranges,
            "alpha_bounds": alpha_bounds,
            "alpha_default": np.array(
                [0.5 * (low + high) for low, high in alpha_bounds], dtype=np.float64
            ),
            "input_dim": 2 + len(alpha_ranges) + 3,
            "pde_fn": _build_polynomial_pde(len(alpha_ranges)),
            "boundary_generator": _make_profile_boundary_generator(alpha_ranges),
            "alpha_labels": [f"alpha_{i}" for i in range(len(alpha_ranges))],
        }
    if profile == "chebyshev":
        alpha_ranges = _build_chebyshev_alpha_ranges(num_param)
        alpha_bounds = [(arr.min(), arr.max()) for arr in alpha_ranges]
        return {
            "name": profile,
            "uses_A": False,
            "num_alpha": len(alpha_ranges),
            "alpha_ranges": alpha_ranges,
            "alpha_bounds": alpha_bounds,
            "alpha_default": np.array(
                [0.5 * (low + high) for low, high in alpha_bounds], dtype=np.float64
            ),
            "input_dim": 2 + len(alpha_ranges) + 3,
            "pde_fn": _build_chebyshev_pde(len(alpha_ranges)),
            "boundary_generator": _make_profile_boundary_generator(alpha_ranges),
            "alpha_labels": [f"alpha_{i}" for i in range(len(alpha_ranges))],
        }
    if profile == "pedestal":
        alpha_ranges = _build_pedestal_alpha_ranges(num_param)
        alpha_bounds = [(arr.min(), arr.max()) for arr in alpha_ranges]
        return {
            "name": profile,
            "uses_A": False,
            "num_alpha": len(alpha_ranges),
            "alpha_ranges": alpha_ranges,
            "alpha_bounds": alpha_bounds,
            "alpha_default": np.array(
                [0.5 * (low + high) for low, high in alpha_bounds], dtype=np.float64
            ),
            "input_dim": 2 + len(alpha_ranges) + 3,
            "pde_fn": _build_pedestal_pde(len(alpha_ranges)),
            "boundary_generator": _make_profile_boundary_generator(alpha_ranges),
            "alpha_labels": ["pedestal_height", "core_pressure"],
        }
    raise ValueError(f"Unsupported profile '{profile_name}'")


def gen_traindata(num: int) -> Tuple[np.ndarray, np.ndarray]:
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")
    return PROFILE_CONFIG["boundary_generator"](num)

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



def area(vs):
    """Compute a contour integral"""
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * abs(y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a




def Cp(vs: np.ndarray) -> float:
    """Chord length integral used in β_p and q* calculations."""
    a = 0.0
    x0, y0 = vs[0]
    for x1, y1 in vs[1:]:
        dx, dy = x1 - x0, y1 - y0
        dy_dx = dy / dx if dx != 0 else 0.0
        a += np.sqrt(1.0 + dy_dx ** 2) * abs(dx)
        x0, y0 = x1, y1
    return a


def qstar_integral(vs: np.ndarray) -> float:
    """Integral appearing in the q* expression (Green's theorem)."""
    a = 0.0
    x0, y0 = vs[0]
    for x1, y1 in vs[1:]:
        dx, dy = x1 - x0, y1 - y0
        M = -1.0 / x1
        a += M * dy
        x0, y0 = x1, y1
    return a


def inside_contour_mask(R: np.ndarray, Z: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return a mask of points inside the contour."""
    print(R.shape, Z.shape, vertices.shape) # (200, 200) (200, 200) (403, 2)
    return np.all(np.cross(vertices[1:] - vertices[0], R.ravel() - vertices[0, 0]) * np.cross(vertices[1:] - vertices[0], Z.ravel() - vertices[0, 1]) <= 0, axis=1)


def _extract_shape_params_from_input(X: np.ndarray) -> Tuple[float, float, float]:
    """Read (eps, kappa, delta) from the network input tensor."""
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")
    offset = 2 + (1 if PROFILE_CONFIG["uses_A"] else 0) + PROFILE_CONFIG["num_alpha"]
    eps_val = float(X[0, offset])
    kappa_val = float(X[0, offset + 1])
    delta_val = float(X[0, offset + 2])
    return eps_val, kappa_val, delta_val


def compute_beta_p(model: dde.Model,
                   X: np.ndarray,
                   psi_pred: np.ndarray,
                   vertices: np.ndarray,
                   A: float = -0.155) -> float:
    """Compute poloidal beta β_p from the predicted stream-function.

    The implementation follows the same procedure used in
    ``utils.utils.compute_params`` for the *predicted* β_p value.  The
    required geometric integrals are evaluated directly from the
    (R, Z) grid and the *ψ* field supplied via *X* and *psi_pred*.
    """

    # ------------------------------------------------------------------
    # Reconstruct structured (R, Z, ψ) grids from the flattened input
    # ------------------------------------------------------------------
    N = psi_pred.shape[0]
    n = int(np.sqrt(N))
    if n * n != N:
        raise ValueError("X and psi_pred must correspond to a square R-Z grid.")

    R = X[:, 0].reshape(n, n)
    Z = X[:, 1].reshape(n, n)
    psi = psi_pred.reshape(n, n)



    # ------------------------------------------------------------------
    # Geometric integrals (now using shared helpers)
    # ------------------------------------------------------------------
    area_cs = area(vertices)
    Cp_val = Cp(vertices)
    q_int = qstar_integral(vertices)

    # ------------------------------------------------------------------
    # Physical constants and shape parameters (ITER defaults)
    # ------------------------------------------------------------------
    mu0 = 4.0 * np.pi * 1e-7
    Itor = 15e6          # Plasma current [A]
    a_minor = 2.0        # Minor radius [m]
    R0 = 6.2             # Major radius [m]
    B0 = 5.3             # Toroidal field on axis [T]

    eps, _, _ = _extract_shape_params_from_input(X)

    # ------------------------------------------------------------------
    # Compute q* and β_p following utils.utils.compute_params
    #    (integrate over the full R–Z grid; psi is ~0 outside the plasma)
    # ------------------------------------------------------------------
    psi_average = np.trapz(
        np.trapz(psi * R[:, 0], R[:, 0], axis=0), Z[0, :]
    )

    denom = area_cs * (A * q_int + 1.115 * area_cs)

    beta_p = (2.0 * 1.155 * Cp_val ** 2 * psi_average) / (denom ** 2)

    return float(beta_p)



def compute_beta_p_and_qstar(model: dde.Model,
                   X: np.ndarray,
                   psi_pred: np.ndarray,
                   vertices: np.ndarray,
                   A: float = -0.155) -> float:
    """Compute beta_p and q* from the predicted stream-function.
    """
    N = psi_pred.shape[0]
    n = int(np.sqrt(N))
    if n * n != N:
        raise ValueError("X and psi_pred must correspond to a square R-Z grid.")

    R = X[:, 0].reshape(n, n)
    Z = X[:, 1].reshape(n, n)
    psi = psi_pred.reshape(n, n)


    area_cs = area(vertices)
    Cp_val = Cp(vertices)
    q_int = qstar_integral(vertices)


    # ------------------------------------------------------------------
    # (ITER defaults)
    # ------------------------------------------------------------------
    mu0 = 4.0 * np.pi * 1e-7
    Itor = 15e6          # Plasma current [A]
    a_minor = 2.0        # Minor radius [m]
    R0 = 6.2             # Major radius [m]
    B0 = 5.3             # Toroidal field on axis [T]

    eps, _, _ = _extract_shape_params_from_input(X)

    psi_average = np.trapz(
        np.trapz(psi * R[:, 0], R[:, 0], axis=0), Z[0, :]
    )
    psi0 = - mu0 * Itor * a_minor / eps / (A * q_int + 1.115 * area_cs)
    qstar = - (a_minor * R0 * B0 * Cp_val) / (psi0 * (A * q_int + 1.115 * area_cs))

    denom = area_cs * (A * q_int + 1.115 * area_cs)

    beta_p = (2.0 * 1.155 * Cp_val ** 2 * psi_average) / (denom ** 2)

    return float(beta_p), float(qstar)
    




grid_size = 300

# if zoom = 1, would get RuntimeWarning: invalid value encountered in arcsin 
#                          self.N1 = - (1 + np.arcsin(self.delta)) ** 2 / (self.eps * self.kappa ** 2)

# True beta_p: 1.0, Predicted beta_p: 1.005661103548024, True volume: 0.5, Predicted volume: 0.48012887330828863, Objective: 0.00042690976937949485
# eps: 0.1535847754093088, kappa: 1.583470712716403, delta: 0.6605555004375785

# if zoom = 2.2, would not get warning


def predict_psi(
    model,
    eps: float,
    kappa: float,
    delta: float,
    A: float = -0.155,
    n: int = grid_size,
    zoom: float = 1.2,
    return_X: bool = False,
    plot_psi: bool = False,
    alpha: Sequence[float] | None = None,
):
    """Evaluate the PINN (predicted) and analytic (true) ψ on a structured grid.

    The grid bounds follow the `utils.utils.evaluate` routine so that the mesh
    grows with ε and κ.  A zoom factor >0 enlarges the domain slightly.

    Returns
    -------
    If *return_X* is False:
        RR, ZZ, psi_pred_grid, psi_true_grid
    If *return_X* is True:
        RR, ZZ, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat

    The optional *alpha* argument supplies pressure-profile coefficients when the
    selected model expects them.  If omitted, the midpoint of the training range
    is used.
    """

    # ------------------------------------------------------------------
    # 1. Build a grid adapted to the plasma shape
    # ------------------------------------------------------------------
    inner_point = 1 - 1.1 * eps * (1 + zoom)
    outer_point = 1 + 1.1 * eps * (1 + zoom)
    low_point   = -1.1 * kappa * eps * (1 + zoom)
    high_point  =  1.1 * kappa * eps * (1 + zoom)

    r = np.linspace(inner_point, outer_point, n)
    z = np.linspace(low_point,   high_point,  n)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")

    

    # 2. Assemble network input tensor X_in (n², 6)
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")

    spatial = np.column_stack((RR.ravel(), ZZ.ravel()))
    features = [spatial]

    if PROFILE_CONFIG["uses_A"]:
        A_col = np.full((spatial.shape[0], 1), A)
        features.append(A_col)

    if PROFILE_CONFIG["num_alpha"] > 0:
        alpha_vec = (
            np.asarray(alpha, dtype=np.float64)
            if alpha is not None
            else PROFILE_CONFIG["alpha_default"]
        )
        if alpha_vec.size != PROFILE_CONFIG["num_alpha"]:
            raise ValueError(
                f"Expected {PROFILE_CONFIG['num_alpha']} alpha coefficients, received {alpha_vec.size}"
            )
        alpha_col = np.tile(alpha_vec.reshape(1, -1), (spatial.shape[0], 1))
        features.append(alpha_col)

    param_col = np.tile(np.array([[eps, kappa, delta]]), (spatial.shape[0], 1))
    features.append(param_col)
    X_in = np.hstack(features)

    # 3. Predict ψ with the PINN
    psi_pred_flat = model.predict(X_in).reshape(-1)
    # psi_pred_flat -= np.min(psi_pred_flat)    # enforce psi=0 at boundary
    psi_pred_grid = np.copy(psi_pred_flat.reshape(n, n))

    # 4. Compute analytic ψ_true via GS_Linear
    # from utils.gs_solovev_sol import GS_Linear  # local import to avoid circular

    # gs = GS_Linear(eps=eps, kappa=kappa, delta=delta)
    # gs.get_BCs(A)
    # gs.solve_coefficients()

    # # psi_true_list = [gs.psi_func(p[0], p[1]) for p in spatial]
    # # psi_true_grid = np.copy(np.reshape(np.array(psi_true_list), (n, n)))

    psi_true_list = "placeholder"
    psi_true_grid = "placeholder"


    if plot_psi:
        TIME0 = time.strftime("%M%S")
        DATE = time.strftime("%m%d")
        from mpl_toolkits.mplot3d import Axes3D

        # Create 3D figure
        fig = plt.figure(figsize=(12, 5))

        # Plot predicted psi
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(RR, ZZ, psi_pred_grid, cmap='viridis')
        ax1.set_xlabel('R')
        ax1.set_ylabel('Z') 
        ax1.set_title('Predicted ψ')
        fig.colorbar(surf1, ax=ax1)

        # Plot true psi
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(RR, ZZ, psi_true_grid, cmap='viridis')
        ax2.set_xlabel('R')
        ax2.set_ylabel('Z')
        ax2.set_title('True ψ')
        fig.colorbar(surf2, ax=ax2)

        plt.tight_layout()
        plt.savefig(f'/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/all{DATE}/psi_3d_comparison{zoom}_{TIME0}.png')
        plt.close()

    if return_X:
        return RR, ZZ, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat
    else:
        return RR, ZZ, psi_pred_grid, psi_true_grid






def visualize_contours(R, Z, psi_grid, eps, kappa, delta, A=-0.155, savepath=None):
    """
    Visualize all ψ=0 contours extracted from a PINN prediction grid,
    and overlay the analytic boundary for comparison.
    """

    # Extract ψ=0 contours
    c = plt.contour(R, Z, psi_grid, levels=[0.0], colors="blue")
    paths = c.collections[0].get_paths()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the full ψ field for context
    pcm = ax.pcolormesh(R, Z, psi_grid, shading="auto", cmap="RdBu")
    fig.colorbar(pcm, ax=ax, label="ψ")
    
    # Overlay each ψ=0 contour
    for i, p in enumerate(paths):
        v = p.vertices
        ax.plot(v[:, 0], v[:, 1], "-", lw=2, label=f"ψ=0 loop {i}")

    # Overlay analytic boundary for comparison
    tau = np.linspace(0, 2*np.pi, 400)
    R_bnd = 1 + eps*np.cos(tau + np.arcsin(delta)*np.sin(tau))
    Z_bnd = eps*kappa*np.sin(tau)
    ax.plot(R_bnd, Z_bnd, "k--", lw=2, label="Analytic boundary")

    ax.set_aspect("equal")
    ax.set_title("ψ=0 contour extraction")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.legend()

    if savepath:
        plt.savefig(savepath, dpi=150)
        print(f"Saved contour visualization to {savepath}")
        plt.close(fig)
    else:
        plt.show()

    # Report how many loops we found and their areas
    print(f"Found {len(paths)} ψ=0 loops")
    for i, p in enumerate(paths):
        v = p.vertices
        # signed polygon area
        area_val = 0.5 * np.abs(np.dot(v[:,0], np.roll(v[:,1], 1)) -
                                np.dot(v[:,1], np.roll(v[:,0], 1)))
        print(f"  Loop {i}: {len(v)} points, area ≈ {area_val:.4f}")



# ----------------------------------------------------------------------------
# Parameter helpers
# ----------------------------------------------------------------------------

def _unpack_params(params: Sequence[float],
                   optimize_A: bool,
                   optimize_alpha: bool,
                   A_fixed: float) -> Tuple[float, float, float, float, np.ndarray]:
    """Decode optimiser parameter vector into physical quantities."""
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")

    params = np.asarray(params, dtype=np.float64)
    if params.size < 3:
        raise ValueError("Parameter vector must contain at least (eps, kappa, delta).")

    idx = 0
    eps_val, kappa_val, delta_val = params[idx:idx + 3]
    idx += 3

    if optimize_A:
        A_cur = params[idx]
        idx += 1
    else:
        A_cur = A_fixed

    if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
        alpha_needed = PROFILE_CONFIG["num_alpha"]
        if params.size < idx + alpha_needed:
            raise ValueError(
                f"Expected {alpha_needed} alpha variables, received {params.size - idx}."
            )
        alpha_vals = params[idx: idx + alpha_needed]
    else:
        alpha_vals = PROFILE_CONFIG["alpha_default"]

    return float(eps_val), float(kappa_val), float(delta_val), float(A_cur), np.asarray(alpha_vals, dtype=np.float64)


def _prepare_initial_guess(initial_guess: Sequence[float] | None,
                           optimize_A: bool,
                           optimize_alpha: bool,
                           A_fixed: float) -> np.ndarray:
    """Ensure the initial guess matches the active optimisation variables."""
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")

    if initial_guess is None:
        guess_values = [0.32, 1.7, 0.33]
    else:
        guess_values = list(initial_guess)

    if len(guess_values) < 3:
        raise ValueError("Initial guess must contain at least [eps, kappa, delta].")

    prepared = guess_values[:3]
    cursor = 3

    if optimize_A:
        if cursor < len(guess_values):
            prepared.append(guess_values[cursor])
            cursor += 1
        else:
            prepared.append(A_fixed)

    if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
        needed = PROFILE_CONFIG["num_alpha"]
        alpha_vals = []
        for i in range(needed):
            if cursor < len(guess_values):
                alpha_vals.append(guess_values[cursor])
                cursor += 1
            else:
                alpha_vals.append(PROFILE_CONFIG["alpha_default"][i])
        prepared.extend(alpha_vals)

    return np.asarray(prepared, dtype=np.float64)


def _build_bounds(bounds: Tuple[Sequence[float], Sequence[float]] | None,
                  optimize_A: bool,
                  optimize_alpha: bool) -> Sequence[Tuple[float, float]]:
    """Compose optimisation bounds for all active variables."""
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")

    if bounds is None:
        lower = [eps0[0], kappa0[0], delta0[0]]
        upper = [eps0[1], kappa0[1], delta0[1]]
        shape_bounds = list(zip(lower, upper))
    else:
        if (
            isinstance(bounds, (list, tuple))
            and len(bounds) == 2
            and isinstance(bounds[0], Sequence)
            and isinstance(bounds[1], Sequence)
        ):
            shape_bounds = list(zip(bounds[0], bounds[1]))
        else:
            shape_bounds = list(bounds)

    full_bounds = shape_bounds

    if optimize_A and PROFILE_CONFIG["uses_A"]:
        full_bounds = full_bounds + [(-Amax, Amax)]

    if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
        full_bounds = full_bounds + PROFILE_CONFIG["alpha_bounds"]

    return full_bounds


def _sample_alpha_initial() -> np.ndarray:
    if PROFILE_CONFIG is None or PROFILE_CONFIG["num_alpha"] == 0:
        return np.array([], dtype=np.float64)
    return np.array(
        [np.random.uniform(low, high) for low, high in PROFILE_CONFIG["alpha_bounds"]],
        dtype=np.float64,
    )

# ----------------------------------------------------------------------------
# Objective function with beta_p and volume
# ----------------------------------------------------------------------------

def make_beta_p_volume_objective(model: dde.Model,
                   target_beta_p: float,
                   target_volume: float,
                   lambda_vol: float = 1.0,
                   n_boundary: int = 400,
                   n_grid: int = 32,
                   major_radius: float = 1.0,
                   zoom: float = 1.2,
                   A_fixed: float = -0.155,
                   optimize_A: bool = False,
                   optimize_alpha: bool = False,
                   print_metrics: bool = False) -> Callable[[Sequence[float]], float]:
    """Return *f([eps, kappa, delta])* for optimisation.

    Args:
        model: Pretrained DeepXDE model representing psi.
        target_beta_p: Desired β_p.
        target_volume: Desired plasma volume.
        lambda_vol: Weight for volume term.
        n_boundary: Number of boundary points used to compute geometric props.
        n_grid: Cartesian grid resolution (per dimension) for model queries.
        major_radius: Major radius R0.
    """

    tau = np.linspace(0.0, 2 * np.pi, n_boundary, endpoint=False)

    def _objective(params: Sequence[float]) -> float:
        eps, kappa, delta, A_cur, alpha_vals = _unpack_params(
            params, optimize_A, optimize_alpha, A_fixed
        )
        R, Z, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat = predict_psi(
            model,
            eps=eps,
            kappa=kappa,
            delta=delta,
            A=A_cur,
            return_X=True,
            plot_psi=False,
            zoom=zoom,
            alpha=alpha_vals,
        )

        # Build analytic Solov'ev boundary (available as fallback for debugging purposes, not used for actual optimization!! We always set fallback to False for actual optimization.)
        use_fallback = args.use_fallback
        if use_fallback:
            x_anal = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
            y_anal = eps * kappa * np.sin(tau)
            vertices_analytic = np.column_stack((x_anal, y_anal))

        # Try to extract ψ = 0 contour from the network prediction. If it fails
        # (no closed contour found on the current grid) fall back to the
        # analytic boundary to keep the optimiser running.
        c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
        
        if use_fallback:
            if c.collections and c.collections[0].get_paths():
                vertices = c.collections[0].get_paths()[0].vertices
            else:
                vertices = vertices_analytic
        else:
            vertices = c.collections[0].get_paths()[0].vertices
        plt.close(c.figure)

        volume = area(vertices)
        beta_p = compute_beta_p(model, X_in, psi_pred_flat, vertices, A=A_cur)

        # obj = lambda_vol * (volume - target_volume) ** 2
        obj = (beta_p - target_beta_p) ** 2 + lambda_vol * (volume - target_volume) ** 2
        # obj = ((beta_p - target_beta_p)/target_beta_p) ** 2 + lambda_vol * ((volume - target_volume)/target_volume) ** 2

        # Record diagnostics for later plotting
        metrics["beta_p_pred"].append(beta_p)
        metrics["volume_pred"].append(volume)
        metrics["eps"].append(eps)
        metrics["kappa"].append(kappa)
        metrics["delta"].append(delta)
        metrics["A"].append(A_cur)
        metrics["alpha"].append(alpha_vals.tolist())
        metrics["obj"].append(obj)

        if print_metrics:
            print(f"True beta_p: {target_beta_p}, Predicted beta_p: {beta_p}, True volume: {target_volume}, Predicted volume: {volume}, Objective: {obj}")
            detail = f"eps: {eps}, kappa: {kappa}, delta: {delta}, A: {A_cur}"
            if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
                detail += f", alpha: {alpha_vals.tolist()}"
            print(detail + "\n")

        return float(obj)
    
    return _objective


# ----------------------------------------------------------------------------
# Objective function with volume
# ----------------------------------------------------------------------------

def make_volume_objective(model: dde.Model,
                         ITER: GS_Linear,
                         target_volume: float,
                         n_boundary: int = 400,
                         n_grid: int = 32,
                         major_radius: float = 1.0,
                         optimize_A: bool = False,
                         optimize_alpha: bool = False,
                         A_fixed: float = -0.155,
                         print_metrics: bool = False) -> Callable[[Sequence[float]], float]:
    """Return *f([eps, kappa, delta])* for optimisation."""

    tau = np.linspace(0.0, 2 * np.pi, n_boundary, endpoint=False)    

    def _volume_objective(params: Sequence[float]) -> float:
        eps, kappa, delta, A_cur, alpha_vals = _unpack_params(
            params, optimize_A, optimize_alpha, A_fixed
        )

        # x, y, psi_pred, psi_true, error = evaluate(ITER, model)

        x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
        y = eps * kappa * np.sin(tau)
        contour = np.column_stack((x, y))

        pred_volume = area(contour)


        obj = (pred_volume - target_volume) ** 2

        # Record diagnostics for later plotting (β_p not evaluated here)
        metrics["beta_p_pred"].append(float('nan'))
        metrics["volume_pred"].append(pred_volume)
        metrics["eps"].append(eps)
        metrics["kappa"].append(kappa)
        metrics["delta"].append(delta)
        metrics["A"].append(A_cur if optimize_A else float('nan'))
        metrics["alpha"].append(alpha_vals.tolist())
        metrics["obj"].append(obj)

        if print_metrics:
            print(f"True volume: {target_volume}, Predicted volume: {pred_volume}, Objective: {obj}")
            detail = f"eps: {eps}, kappa: {kappa}, delta: {delta}"
            if optimize_A:
                detail += f", A: {A_cur}"
            if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
                detail += f", alpha: {alpha_vals.tolist()}"
            print(detail)
        return float(obj)
    return _volume_objective


# ----------------------------------------------------------------------------
# Objective function with beta_p, volume and qstar
# ----------------------------------------------------------------------------


def make_objective(model: dde.Model,
                  target_beta_p: float,
                  target_volume: float,
                  target_qstar: float,
                  lambda_beta_p: float = 1.0,
                  lambda_vol: float = 1.0,
                  lambda_qstar: float = 1.0,
                  n_boundary: int = 400,
                  n_grid: int = 32,
                  major_radius: float = 1.0,
                  zoom: float = 1.2,
                  A_fixed: float = -0.155,
                  optimize_A: bool = False,
                  optimize_alpha: bool = False,
                  print_metrics: bool = False) -> Callable[[Sequence[float]], float]:
    """Return *f([eps, kappa, delta])* for optimisation.

    Args:
        model: Pretrained DeepXDE model representing psi.
        target_beta_p: Desired β_p.
        target_volume: Desired plasma volume.
        target_qstar: Desired qstar.
        lambda_beta_p: Weight for beta_p term.
        lambda_vol: Weight for volume term.
        lambda_qstar: Weight for qstar term.
        n_boundary: Number of boundary points used to compute geometric props.
        n_grid: Cartesian grid resolution (per dimension) for model queries.
        major_radius: Major radius R0.
    """

    tau = np.linspace(0.0, 2 * np.pi, n_boundary, endpoint=False)

    def _objective(params: Sequence[float]) -> float:
        eps, kappa, delta, A_cur, alpha_vals = _unpack_params(
            params, optimize_A, optimize_alpha, A_fixed
        )
        R, Z, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat = predict_psi(
            model,
            eps=eps,
            kappa=kappa,
            delta=delta,
            A=A_cur,
            return_X=True,
            plot_psi=False,
            zoom=zoom,
            alpha=alpha_vals,
        )
        # visualize_contours(R, Z, psi_pred_grid, eps, kappa, delta, savepath=f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/psi_contour/{eps}_{kappa}_{delta}.png")

        use_fallback = args.use_fallback
        if use_fallback:
            x_anal = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
            y_anal = eps * kappa * np.sin(tau)
            vertices_analytic = np.column_stack((x_anal, y_anal))

        # Try to extract ψ = 0 contour from the network prediction. If it fails
        # (no closed contour found on the current grid) fall back to the
        # analytic boundary to keep the optimiser running.
        c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
        
        if use_fallback:
            if c.collections and c.collections[0].get_paths():
                vertices = c.collections[0].get_paths()[0].vertices
            else:
                # Fallback: use analytic boundary when the predicted contour is
                # not available. This prevents IndexError and provides a
                # meaningful, smooth objective for the optimiser.
                vertices = vertices_analytic
        else:
            vertices = c.collections[0].get_paths()[0].vertices
        plt.close(c.figure)


        volume = area(vertices)
        beta_p, qstar = compute_beta_p_and_qstar(model, X_in, psi_pred_flat, vertices, A=A_cur)

        # obj = lambda_beta_p * ((beta_p - target_beta_p)/target_beta_p) ** 2 + lambda_vol * ((volume - target_volume)/target_volume) ** 2 + lambda_qstar * ((qstar - target_qstar)/target_qstar) ** 2
        obj = lambda_beta_p * (beta_p - target_beta_p) ** 2 + lambda_vol * (volume - target_volume) ** 2 + lambda_qstar * (qstar - target_qstar) ** 2

        # Record diagnostics for later plotting
        metrics["beta_p_pred"].append(beta_p)
        metrics["volume_pred"].append(volume)
        metrics["qstar_pred"].append(qstar)
        metrics["eps"].append(eps)
        metrics["kappa"].append(kappa)
        metrics["delta"].append(delta)
        metrics["A"].append(A_cur)
        metrics["alpha"].append(alpha_vals.tolist())
        metrics["obj"].append(obj)

        if print_metrics:
            print(f"True beta_p: {target_beta_p}, Predicted beta_p: {beta_p}, True volume: {target_volume}, Predicted volume: {volume}, True qstar: {target_qstar}, Predicted qstar: {qstar}, Objective: {obj}")
            detail = f"eps: {eps}, kappa: {kappa}, delta: {delta}, A: {A_cur}"
            if optimize_alpha and PROFILE_CONFIG["num_alpha"] > 0:
                detail += f", alpha: {alpha_vals.tolist()}"
            print(detail + "\n")

        return float(obj)
    
    return _objective






# ----------------------------------------------------------------------------
# Top-level optimisation 
# ----------------------------------------------------------------------------

def optimise_shape(model: dde.Model,
                  ITER: GS_Linear,
                  target_beta_p: float,
                  target_volume: float,
                  target_qstar: float,
                  lambda_beta_p: float = 1.0,
                  lambda_volume: float = 1.0,
                  lambda_qstar: float = 1.0,
                  initial_guess: Sequence[float] | None = None,
                  bounds: Tuple[Sequence[float], Sequence[float]] | None = None,
                  method: str = "L-BFGS-B",
                  maxiter: int = 200,
                  zoom: float = 1.2,
                  objective_type: str = "volume",
                  A_fixed: float = -0.155,
                  optimize_A: bool = False,
                  optimize_alpha: bool = False,
                  print_metrics: bool = False) -> OptimizeResult:
    """Optimise (eps, kappa, delta) to minimise f(psi_pred).

    Args:
        model_path: Directory containing saved DeepXDE model.
        target_beta_p: Desired β_p.
        target_volume: Desired toroidal volume.
        target_qstar: Desired qstar.
        lambda_beta_p: Weight λ in objective.
        lambda_volume: Weight λ in objective.
        lambda_qstar: Weight λ in objective.
        initial_guess: Starting [eps, kappa, delta].
        bounds: Tuple (lower, upper) for each parameter.
        method: SciPy optimisation method.
        maxiter: Maximum iterations.
        objective_type: Type of objective function to use.
    """

    # ------------------ Initial guess ----------------------------
    if initial_guess is None:
        initial_guess = [0.32, 1.7, 0.33]  # ITER-like defaults
    if PROFILE_CONFIG is None:
        raise RuntimeError("Profile configuration has not been initialised.")

    if optimize_A and not PROFILE_CONFIG["uses_A"]:
        print("Selected profile does not use A. Disabling optimise_A flag.")
        optimize_A = False

    if optimize_alpha and PROFILE_CONFIG["num_alpha"] == 0:
        print("No pressure coefficients defined for this profile. Disabling optimise_alpha flag.")
        optimize_alpha = False

    initial_guess = _prepare_initial_guess(initial_guess, optimize_A, optimize_alpha, A_fixed)

    if objective_type == "volume":
        objective = make_volume_objective(model,
                                        ITER=ITER,
                                        target_volume=target_volume,
                                        n_boundary=400,
                                        n_grid=32,
                                        major_radius=1.0,
                                        optimize_A=optimize_A,
                                        optimize_alpha=optimize_alpha,
                                        A_fixed=A_fixed,
                                        print_metrics=print_metrics)

    elif objective_type == "beta_p":
        objective = make_beta_p_volume_objective(model,
                                   target_beta_p=target_beta_p,
                                   target_volume=target_volume,
                                   lambda_vol=lambda_volume,
                                   n_boundary=400,
                                   n_grid=32,
                                   major_radius=1.0,
                                   zoom=zoom,
                                   A_fixed=A_fixed,
                                   optimize_A=optimize_A,
                                   optimize_alpha=optimize_alpha,
                                   print_metrics=print_metrics)

    
    elif objective_type == "beta_p_and_qstar":
        objective = make_objective(model,
                                   target_beta_p=target_beta_p,
                                   target_volume=target_volume,
                                   target_qstar=target_qstar,
                                   lambda_beta_p=lambda_beta_p,
                                   lambda_vol=lambda_volume,
                                   lambda_qstar=lambda_qstar,
                                   n_boundary=400,
                                   n_grid=32,
                                   major_radius=1.0,
                                   zoom=zoom,
                                   A_fixed=A_fixed,
                                   optimize_A=optimize_A,
                                   optimize_alpha=optimize_alpha,
                                   print_metrics=print_metrics)
    else:
        raise ValueError(f"Invalid objective type: {objective_type}")


    options = dict(maxiter=maxiter, disp=True, ftol=1e-09, gtol=1e-05)

    bounds_used = _build_bounds(bounds, optimize_A, optimize_alpha)

    res = minimize(objective,
                x0=initial_guess,
                method=method,
                bounds=bounds_used,
                options=options)
    
    
    return res



def optimise_shape_with_restarts(
    model: dde.Model,
    ITER: GS_Linear,
    target_beta_p: float,
    target_volume: float,
    target_qstar: float,
    lambda_beta_p: float = 1.0,
    lambda_volume: float = 1.0,
    lambda_qstar: float = 1.0,
    n_restarts: int = 10,
    bounds: Tuple[Sequence[float], Sequence[float]] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    zoom: float = 1.2,
    objective_type: str = "beta_p_and_qstar",
    A_fixed: float = -0.155,
    optimize_A: bool = False,
    optimize_alpha: bool = False,
):
    """
    Perform shape optimization with multiple random restarts.
    Each restart samples a random initial guess within the bounds
    and records the best result (lowest objective value).
    """
    best_result = None
    best_obj = np.inf

    for i in range(n_restarts):

        x0 = np.array([
            np.random.uniform(0.12, 0.52),  # ε
            np.random.uniform(1.25, 2.75),  # κ
            np.random.uniform(-0.5, 0.5),   # δ
        ])
        if optimize_A:
            x0 = np.append(x0, np.random.uniform(-Amax, Amax))
        if optimize_alpha and PROFILE_CONFIG is not None and PROFILE_CONFIG["num_alpha"] > 0:
            x0 = np.append(x0, _sample_alpha_initial())

        print(f"\n=== Restart {i+1}/{n_restarts} ===")
        print(f"Initial guess: {x0}")

        result = optimise_shape(
            model=model,
            ITER=ITER,
            target_beta_p=target_beta_p,
            target_volume=target_volume,
            target_qstar=target_qstar,
            lambda_beta_p=lambda_beta_p,
            lambda_volume=lambda_volume,
            lambda_qstar=lambda_qstar,
            initial_guess=x0,
            bounds=bounds,
            method=method,
            maxiter=maxiter,
            zoom=zoom,
            objective_type=objective_type,
            A_fixed=A_fixed,
            optimize_A=optimize_A,
            optimize_alpha=optimize_alpha,
        )

        obj_val = result.fun
        print(f"Restart {i+1} objective: {obj_val}")
        print(f"Parameters: {result.x}")

        if obj_val < best_obj:
            best_obj = obj_val
            best_result = result

    print("\n=== Best result across all restarts ===")
    print(f"Objective: {best_obj}")
    print(f"Parameters: {best_result.x}")
    return best_result




def save_metrics_to_jsonl(metrics: dict, save_dir: str, filename: str = None) -> str:
    """Save optimization metrics to a JSONL file.
    
    Each line in the file represents one function evaluation with all parameters.
    
    Args:
        metrics: Dictionary containing lists of metric values.
        save_dir: Directory to save the file.
        filename: Optional filename. If None, generates timestamped name.
    
    Returns:
        Path to the saved file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_metrics_{timestamp}.jsonl"
    
    filepath = os.path.join(save_dir, filename)
    
    # Get the number of evaluations
    n_evals = len(metrics["obj"])
    
    with open(filepath, 'w') as f:
        for i in range(n_evals):
            record = {
                "eval_idx": i,
                "beta_p_pred": metrics["beta_p_pred"][i] if i < len(metrics["beta_p_pred"]) else None,
                "volume_pred": metrics["volume_pred"][i] if i < len(metrics["volume_pred"]) else None,
                "qstar_pred": metrics["qstar_pred"][i] if i < len(metrics["qstar_pred"]) else None,
                "eps": metrics["eps"][i] if i < len(metrics["eps"]) else None,
                "kappa": metrics["kappa"][i] if i < len(metrics["kappa"]) else None,
                "delta": metrics["delta"][i] if i < len(metrics["delta"]) else None,
                "A": metrics["A"][i] if i < len(metrics["A"]) else None,
                "alpha": metrics["alpha"][i] if i < len(metrics["alpha"]) else None,
                "obj": metrics["obj"][i] if i < len(metrics["obj"]) else None,
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved {n_evals} evaluation records to {filepath}")
    return filepath


def save_optimization_summary(metrics: dict, result, args, save_dir: str, filename: str = None) -> str:
    """Save a summary JSON file with optimization results and configuration.
    
    Args:
        metrics: Dictionary containing lists of metric values.
        result: OptimizeResult from scipy.optimize.minimize.
        args: Command line arguments.
        save_dir: Directory to save the file.
        filename: Optional filename. If None, generates timestamped name.
    
    Returns:
        Path to the saved file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_summary_{timestamp}.json"
    
    filepath = os.path.join(save_dir, filename)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "profile": args.profile,
            "objective_type": args.objective_type,
            "target_beta_p": args.target_beta_p,
            "target_volume": args.target_volume,
            "target_qstar": args.target_qstar,
            "lambda_beta_p": args.lambda_beta_p,
            "lambda_volume": args.lambda_volume,
            "lambda_qstar": args.lambda_qstar,
            "zoom": args.zoom,
            "A": args.A,
            "optimize_A": args.optimize_A,
            "optimize_alpha": args.optimize_alpha,
            "n_restarts": args.n_restarts,
            "maxiter": args.maxiter,
            "method": args.method,
        },
        "result": {
            "optimal_params": result.x.tolist() if hasattr(result.x, 'tolist') else list(result.x),
            "optimal_objective": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message) if hasattr(result, 'message') else None,
            "n_iterations": int(result.nit) if hasattr(result, 'nit') else None,
            "n_function_evals": int(result.nfev) if hasattr(result, 'nfev') else None,
        },
        "final_metrics": {
            "beta_p_pred": metrics["beta_p_pred"][-1] if metrics["beta_p_pred"] else None,
            "volume_pred": metrics["volume_pred"][-1] if metrics["volume_pred"] else None,
            "qstar_pred": metrics["qstar_pred"][-1] if metrics["qstar_pred"] else None,
        },
        "total_evaluations": len(metrics["obj"]),
    }
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved optimization summary to {filepath}")
    return filepath


def tune_lambdas(model, ITER, base_args, n_trials=10):
    """
    Randomly sample combinations of lambda_beta_p, lambda_volume, lambda_qstar
    and pick the one yielding the lowest objective after short optimization runs.
    """
    best_lambdas = None
    best_obj = np.inf

    print("Initial guess: ", [0.398, 2.264, 0.444, 0.199])

    for i in range(n_trials):
        lam_beta = 10 ** np.random.uniform(-1, 2)     # 0.1 – 100
        lam_vol  = 10 ** np.random.uniform(0, 2)      # 1 – 100
        lam_q    = 10 ** np.random.uniform(-1, 1.5)   # 0.1 – 30

        print(f"\n>>> Lambda trial {i+1}: βp={lam_beta:.2f}, V={lam_vol:.2f}, q*={lam_q:.2f}")

        result = optimise_shape(
            model=model,
            ITER=ITER,
            target_beta_p=base_args.target_beta_p,
            target_volume=base_args.target_volume,
            target_qstar=base_args.target_qstar,
            lambda_beta_p=lam_beta,
            lambda_volume=lam_vol,
            lambda_qstar=lam_q,
            initial_guess=[0.398, 2.264, 0.444, 0.199],
            bounds=base_args.bounds,
            method=base_args.method,
            maxiter=200,    # shorter run for testing
            zoom=base_args.zoom,
            objective_type="beta_p_and_qstar",
            A_fixed=base_args.A,
            optimize_A=base_args.optimize_A,
            optimize_alpha=base_args.optimize_alpha,
            print_metrics=base_args.print_metrics
        )

        print("Objective value: ", result.fun)
        print("Parameters: ", result.x)

        if result.fun < best_obj:
            best_obj = result.fun
            best_lambdas = (lam_beta, lam_vol, lam_q)

    print("\n=== Best λ combination ===")
    print(f"λβp={best_lambdas[0]:.2f}, λV={best_lambdas[1]:.2f}, λq*={best_lambdas[2]:.2f}")
    print(f"Objective={best_obj:.4e}")
    return best_lambdas




if __name__ == "__main__":
    TIME = time.strftime("%m%d%H%M")

    # ----------------------------------------------------------------------------
    # Parse command line arguments
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--target_beta_p", type=float, default=1.2)
    parser.add_argument("--target_volume", type=float, default=1.2)
    parser.add_argument("--target_qstar", type=float, default=1.57)
    parser.add_argument("--lambda_beta_p", type=float, default=48.9)
    parser.add_argument("--lambda_volume", type=float, default=1.61)
    parser.add_argument("--lambda_qstar", type=float, default=0.11)
    parser.add_argument("--profile", type=str, default="solovev",
                        choices=["solovev", "polynomial", "chebyshev", "pedestal"])
    parser.add_argument("--initial_guess", type=list, default=[0.32, 1.7, 0.33])
    parser.add_argument("--bounds", type=list, default=([eps0[0], kappa0[0], delta0[0]], [eps0[1], kappa0[1], delta0[1]]))
    parser.add_argument("--method", type=str, default="L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=15000)
    parser.add_argument("--train_new", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--use_fallback", type=bool, default=False)
    parser.add_argument("--zoom", type=float, default=1.2)
    parser.add_argument("--A", type=float, default=-0.155)
    parser.add_argument("--optimize_A", type=bool, default=True)
    parser.add_argument(
        "--optimize_alpha",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable optimisation of pressure-profile coefficients.",
    )
    parser.add_argument(
        "--alpha_init",
        type=float,
        nargs="+",
        default=None,
        help="Optional initial guess for pressure-profile coefficients.",
    )
    parser.add_argument("--objective_type", type=str, default="beta_p_and_qstar")
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--print_metrics", type=bool, default=False)
    args = parser.parse_args()

    PROFILE_CONFIG = build_profile_config(args.profile)

    if args.optimize_alpha is None:
        args.optimize_alpha = PROFILE_CONFIG["num_alpha"] > 0
    elif PROFILE_CONFIG["num_alpha"] == 0 and args.optimize_alpha:
        print("Selected profile has no alpha parameters; disabling optimise_alpha.")
        args.optimize_alpha = False

    if args.optimize_A and not PROFILE_CONFIG["uses_A"]:
        print("Selected profile ignores A parameter; disabling optimise_A.")
        args.optimize_A = False

    args.initial_guess = list(args.initial_guess)
    if args.alpha_init:
        args.initial_guess.extend(args.alpha_init)

    print(f"\n\nObjective type: {args.objective_type}")
    print(f"Targets: beta_p: {args.target_beta_p}, volume: {args.target_volume}, qstar: {args.target_qstar}")
    print(f"Lambdas: beta_p: {args.lambda_beta_p}, volume: {args.lambda_volume}, qstar: {args.lambda_qstar}")
    print(f"Zoom: {args.zoom}")
    print(f"A: {args.A}, optimize_A: {args.optimize_A}")
    print(f"Profile: {args.profile}, optimize_alpha: {args.optimize_alpha}\n\n")

    # ----------------------------------------------------------------------------
    # Define model
    # ----------------------------------------------------------------------------

    start_run_time = time.time()
    
    if args.model_path is not None:
        CHECKPOINT_PATH = args.model_path
    else:
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_08142025_1323_parametrized/ITER-11293.ckpt"
        # CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_08142025_2320_parametrized/ITER-16001.ckpt"
        # param 5
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09102025_0843_parametrized/ITER-16001.ckpt"
        # param 6
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09152025_2103_parametrized/ITER-16001.ckpt"
        # param 7
        # CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_09162025_0731_parametrized/ITER-16001.ckpt"
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12042025_0525_parametrized2_pedestal/ITER-511.ckpt"
        # Pedestal fixed params 4
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12112025_0647_pedestal_fixed4/pedestal_model-16001.ckpt"
        # Pedestal fixed params 5
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12112025_0657_pedestal_fixed_5/pedestal_model-16001.ckpt"
        # Pedestal fixed params 6 !!performance possibly decreased due to MAX_BOUNDARY_POINTS, which later will be disabled
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12112025_1356_pedestal_corrected_6/pedestal_model-16001.ckpt"
        # Pedestal fixed params 9 !!performance vastly decreased due to MAX_BOUNDARY_POINTS, which later will be disabled
        # CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_12112025_0710_pedestal_fixed_9/pedestal_model-16001.ckpt"


    DEPTH = 4
    BREADTH = 40
    # LR = 2e-3
    LR = 2e-2
    AF = "swish"

    INPUT_DIM = PROFILE_CONFIG["input_dim"]

    net = dde.maps.FNN([INPUT_DIM] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

    geometry_kwargs = dict(
        eps_range=eps0,
        kappa_range=kappa0,
        delta_range=delta0,
        num_param=num_param,
        psi_boundary_points=200,
    )
    if PROFILE_CONFIG["alpha_ranges"] is not None:
        geometry_kwargs["alpha_ranges"] = PROFILE_CONFIG["alpha_ranges"]
    

    if args.train_new:
        if args.profile == "solovev":
            spatial_domain = dde.geometry.HyperEllipticalToroid_old(
                eps_range=eps0,
                kappa_range=kappa0,
                delta_range=delta0,
                Amax=Amax,
                num_param=num_param,
            )
        elif args.profile == "pedestal":
            spatial_domain = dde.geometry.HyperEllipticalToroid(**geometry_kwargs)
        x, u = gen_traindata(1001)
        bc135 = dde.PointSetBC(x, u)
        data = dde.data.PDE(spatial_domain, PROFILE_CONFIG["pde_fn"], [bc135],
                            num_domain=1028, num_boundary=100,
                            num_test=n_test, train_distribution="LHS")
        model = dde.Model(data=data, net=net)
        model.compile(args.method, lr=LR, loss_weights=[1,100])
        loss_history, train_state = model.train(epochs=1000, display_every=10)
        model.save(f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}/ITER", protocol="backend", verbose=1)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir= f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_plots_new/run_{TIME}")
    else:
        # Create dummy data for inference only
        if args.profile == "solovev":
            spatial_domain = dde.geometry.HyperEllipticalToroid_old(
                eps_range=eps0,
                kappa_range=kappa0,
                delta_range=delta0,
                Amax=Amax,
                num_param=num_param,
            )
        elif args.profile == "pedestal":
            spatial_domain = dde.geometry.HyperEllipticalToroid(**geometry_kwargs)

        data = dde.data.PDE(
            spatial_domain,
            PROFILE_CONFIG["pde_fn"],
            [],  # No BCs needed for inference
            num_domain=1,  # Minimal points needed
            num_boundary=0,
            num_test=1
        )
        model = dde.Model(data=data, net=net)
        model.compile("adam", lr=LR, loss_weights=[1,100])
        model.restore(CHECKPOINT_PATH, verbose=1)


    ITER = GS_Linear(eps=eps0[0], kappa=kappa0[0], delta=delta0[0])
    ITER.get_BCs(args.A)
    ITER.solve_coefficients()

    result = optimise_shape_with_restarts(model=model,
                            ITER=ITER,
                            target_beta_p=args.target_beta_p,
                            target_volume=args.target_volume,
                            target_qstar=args.target_qstar,
                            lambda_beta_p=args.lambda_beta_p,
                            lambda_volume=args.lambda_volume,
                            lambda_qstar=args.lambda_qstar,
                            n_restarts=args.n_restarts,
                            # initial_guess=args.initial_guess,
                            # bounds=args.bounds,
                            bounds = None,
                            method=args.method,
                            maxiter=args.maxiter,
                            zoom=args.zoom,
                            objective_type=args.objective_type,
                            A_fixed=args.A,
                            optimize_A=args.optimize_A,
                            optimize_alpha=args.optimize_alpha)


    # result = tune_lambdas(model, ITER, args, 50)


    print("\nOptimisation finished:\n", result)

    end_run_time = time.time()
    print(f"Run time: {end_run_time - start_run_time} seconds")

    # ------------------------------------------------------------
    # Save metrics to files for later analysis
    # ------------------------------------------------------------
    metrics_save_dir = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/optimization_metrics"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed metrics (JSONL - one record per evaluation)
    metrics_filepath = save_metrics_to_jsonl(
        metrics, 
        save_dir=metrics_save_dir,
        filename=f"metrics_{args.profile}_{args.objective_type}_{timestamp}.jsonl"
    )
    
    # Save summary (single JSON with config and final results)
    summary_filepath = save_optimization_summary(
        metrics, 
        result, 
        args,
        save_dir=metrics_save_dir,
        filename=f"summary_{args.profile}_{args.objective_type}_{timestamp}.json"
    )

    # ------------------------------------------------------------
    # Plot optimisation diagnostics collected in *metrics*
    # ------------------------------------------------------------
    
    if args.plot:
        plot_save_path = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/run_plots/{args.objective_type}/plots_lambda_{args.lambda_volume}_{args.lambda_beta_p}_{args.lambda_qstar}/"
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
            
        if metrics["beta_p_pred"]:
            # Take every 4th item
            iters = range(0, len(metrics["beta_p_pred"]), 4)
            beta_p_pred = [metrics["beta_p_pred"][i] for i in iters]
            volume_pred = [metrics["volume_pred"][i] for i in iters]
            eps = [metrics["eps"][i] for i in iters]
            kappa = [metrics["kappa"][i] for i in iters] 
            delta = [metrics["delta"][i] for i in iters]
            obj = [metrics["obj"][i] for i in iters]
            iters = range(len(obj))

            # 1. β_p and volume on shared x-axis with twin y-axes
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("Function evaluation")
            ax1.set_ylabel("Predicted β_p", color="tab:red")
            ax1.plot(iters, beta_p_pred, color="tab:red", label="β_p (pred)")
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel("Predicted volume", color="tab:blue")
            ax2.plot(iters, volume_pred, color="tab:blue", label="Volume (pred)")
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            fig.tight_layout()
            fig.savefig(plot_save_path + "metrics_beta_volume.png", dpi=150)
            plt.close(fig)

            # 2. ε, κ, δ evolution
            plt.figure()
            plt.plot(iters, eps, label="eps (ε)")
            plt.plot(iters, kappa, label="kappa (κ)")
            plt.plot(iters, delta, label="delta (δ)")
            plt.xlabel("Function evaluation")
            plt.ylabel("Parameter value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_save_path + "params.png", dpi=150)
            plt.close()

            if PROFILE_CONFIG["num_alpha"] > 0 and metrics["alpha"]:
                alpha_series = [
                    metrics["alpha"][i]
                    for i in range(0, len(metrics["alpha"]), 4)
                    if len(metrics["alpha"][i]) == PROFILE_CONFIG["num_alpha"]
                ]
                if alpha_series:
                    alpha_arr = np.array(alpha_series)
                    plt.figure()
                    alpha_iters = range(alpha_arr.shape[0])
                    for idx in range(alpha_arr.shape[1]):
                        label = (
                            PROFILE_CONFIG["alpha_labels"][idx]
                            if idx < len(PROFILE_CONFIG["alpha_labels"])
                            else f"alpha_{idx}"
                        )
                        plt.plot(alpha_iters, alpha_arr[:, idx], label=label)
                    plt.xlabel("Function evaluation (downsampled)")
                    plt.ylabel("Alpha value")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_save_path + "alpha_params.png", dpi=150)
                    plt.close()

            # 3. Objective value
            plt.figure()
            plt.yscale("log")
            plt.plot(iters, obj, label="Objective")
            plt.xlabel("Function evaluation")
            plt.ylabel("Objective value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_save_path + "obj.png", dpi=150)
            plt.close()


