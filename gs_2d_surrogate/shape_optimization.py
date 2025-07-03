import os
from typing import Callable, Sequence, Tuple
import tensorflow as tf
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
import time
import argparse

from utils.utils import *
from utils.gs_solovev_sol import GS_Linear

# ----------------------------------------------------------------------------
# Geometry helper routines specific to Solovev parameterisation
# ----------------------------------------------------------------------------

def solov_ev_boundary(tau: np.ndarray,
                      eps: float,
                      kappa: float,
                      delta: float,
                      r0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R(tau), Z(tau)) for the ITER-like Solovev shape.

    The parametrisation matches that used in *gen_traindata* of
    ``gs-2d-surrogate/general_solovev_equil_parametrized_shape.py``::

        R = R0 + eps * cos(tau + asin(delta) * sin(tau))
        Z = eps * kappa * sin(tau)

    Args:
        tau: 1-D array of poloidal angles (0..2π).
        eps: Inverse aspect ratio ϵ.
        kappa: Elongation κ.
        delta: Triangularity δ.
        r0: Major radius R0 (default 1.0).
    """
    tau = np.asarray(tau)
    R = r0 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    Z = eps * kappa * np.sin(tau)
    return R, Z


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Polygon area via the shoelace formula (positive orientation)."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def torus_volume(area_cross_section: float, major_radius: float = 1.0) -> float:
    """Toroidal volume *V = 2π R0 A_cs*."""
    return 2 * np.pi * major_radius * area_cross_section

# ----------------------------------------------------------------------------
# Physics metrics
# ----------------------------------------------------------------------------

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




def compute_beta_p(model: dde.Model,
                   X: np.ndarray,
                   psi_pred: np.ndarray) -> float:
    """Compute poloidal beta β_p from predicted stream-function.

    A domain-specific implementation is required here.  For the purpose of
    setting up the optimisation pipeline we simply return a dummy value.
    """
    # TODO: Implement real β_p calculation using model gradients.
    return 0.0

# ----------------------------------------------------------------------------
# Total objective function
# ----------------------------------------------------------------------------

def make_objective(model: dde.Model,
                   target_beta_p: float,
                   target_volume: float,
                   lambda_vol: float = 1.0,
                   n_boundary: int = 400,
                   n_grid: int = 32,
                   major_radius: float = 1.0) -> Callable[[Sequence[float]], float]:
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
        eps, kappa, delta = params

        # --------------------------------------------------------------
        # 1. Geometry (area, volume)
        # --------------------------------------------------------------
        R_bnd, Z_bnd = solov_ev_boundary(tau, eps, kappa, delta, r0=major_radius)
        area_cs = polygon_area(R_bnd, Z_bnd)
        volume = torus_volume(area_cs, major_radius)

        # --------------------------------------------------------------
        # 2. Model evaluation inside domain
        # --------------------------------------------------------------
        r_min, r_max = R_bnd.min(), R_bnd.max()
        z_min, z_max = Z_bnd.min(), Z_bnd.max()
        r_lin = np.linspace(r_min, r_max, n_grid)
        z_lin = np.linspace(z_min, z_max, n_grid)
        rr, zz = np.meshgrid(r_lin, z_lin, indexing="ij")
        pts = np.column_stack((rr.ravel(), zz.ravel()))  # (N,2)

        # Assemble input tensor: [R, Z, A, eps, kappa, delta]
        A = np.zeros((pts.shape[0], 1))  # A=0 for now (can be varied)
        params_block = np.tile(np.array([[eps, kappa, delta]]), (pts.shape[0], 1))
        X_in = np.hstack((pts, A, params_block))

        psi_pred = model.predict(X_in)
        beta_p = compute_beta_p(model, X_in, psi_pred)

        # --------------------------------------------------------------
        # 3. Objective value
        # --------------------------------------------------------------
        obj = (beta_p - target_beta_p) ** 2 + lambda_vol * (volume - target_volume) ** 2
        return float(obj)
    
    # def _volume_objective(params: Sequence[float]) -> float:
    #     eps, kappa, delta = params
    #     R_bnd, Z_bnd = solov_ev_boundary(tau, eps, kappa, delta, r0=major_radius)
    #     area_cs = polygon_area(R_bnd, Z_bnd)
    #     volume = torus_volume(area_cs, major_radius)
        
    #     obj = (volume - target_volume) ** 2
    #     return float(obj)
    return _objective


# ----------------------------------------------------------------------------
# Volume objective function
# ----------------------------------------------------------------------------

def make_volume_objective(model: dde.Model,
                          ITER: GS_Linear,
                          target_volume: float,
                          n_boundary: int = 400,
                          n_grid: int = 32,
                          major_radius: float = 1.0) -> Callable[[Sequence[float]], float]:
    """Return *f([eps, kappa, delta])* for optimisation."""

    tau = np.linspace(0.0, 2 * np.pi, n_boundary, endpoint=False)    

    def _volume_objective(params: Sequence[float]) -> float:
        eps, kappa, delta = params

        # x, y, psi_pred, psi_true, error = evaluate(ITER, model)

        x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
        y = eps * kappa * np.sin(tau)
        contour = np.column_stack((x, y))

        pred_volume = area(contour)


        obj = (pred_volume - target_volume) ** 2
        print(f"True volume: {target_volume}, Predicted volume: {pred_volume}, Objective: {obj}")
        return float(obj)
    return _volume_objective



        # c = plt.contour(x, y, psi_pred, [0])
        # pred_volume = c.collections[0].get_paths()[0].vertices.shape[0]

        # R_bnd, Z_bnd = solov_ev_boundary(tau, eps, kappa, delta, r0=major_radius)
        # area_cs = polygon_area(R_bnd, Z_bnd)
        # volume = torus_volume(area_cs, major_radius)


# ----------------------------------------------------------------------------
# Top-level optimisation 
# ----------------------------------------------------------------------------

def optimise_shape(model: dde.Model,
                   ITER: GS_Linear,
                   target_beta_p: float,
                   target_volume: float,
                   lambda_volume: float = 1.0,
                   initial_guess: Sequence[float] | None = None,
                   bounds: Tuple[Sequence[float], Sequence[float]] | None = None,
                   method: str = "L-BFGS-B",
                   maxiter: int = 200) -> OptimizeResult:
    """Optimise (eps, kappa, delta) to minimise f(\u03A6̄).

    Args:
        model_path: Directory containing saved DeepXDE model.
        target_beta_p: Desired β_p.
        target_volume: Desired toroidal volume.
        lambda_volume: Weight λ in objective.
        initial_guess: Starting [eps, kappa, delta].
        bounds: Tuple (lower, upper) for each parameter.
        method: SciPy optimisation method.
        maxiter: Maximum iterations.
    """
    # ------------------ Load pretrained model --------------------
    # model = dde.models.Model.load(model_path) # gibberish

    # ------------------ Initial guess ----------------------------
    if initial_guess is None:
        initial_guess = [0.32, 1.7, 0.33]  # ITER-like defaults
    initial_guess = np.asarray(initial_guess, dtype=float)

    objective = make_volume_objective(model,
                                      ITER=ITER,
                                      target_volume=target_volume,
                                      n_boundary=400,
                                      n_grid=32,
                                      major_radius=1.0)

    options = dict(maxiter=maxiter, disp=True)

    res = minimize(objective,
                   x0=initial_guess,
                   method=method,
                   bounds=None if bounds is None else list(zip(*bounds)),
                   options=options)

    return res



if __name__ == "__main__":
    TIME = time.strftime("%m%d%H%M")
    # Path where *model.ckpt* (or SavedModel) resides.
    MODEL_DIR = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_07030318/ITER-01.ckpt-812.ckpt.index"  # pretrained model checkpoint


    ######################
    # ITER Configuration #
    ######################
    A = -0.155
    eps = 0.32
    kappa = 1.7
    delta = 0.33
    # A = -0.155
    # eps = 0.2  
    # kappa = 1.7 
    # delta = 0.5 

    N1 = - (1 + np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N2 = (1 - np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N3 = - kappa / (eps * np.cos(np.arcsin(delta)) ** 2)

    def gen_traindata(num):
        ######################
        # ITER Configuration #
        ######################
        eps = 0.32
        kappa = 1.7
        delta = 0.33
        N = num
        center, eps, kappa, delta = np.array([[0.0,0.0]]), eps, kappa, delta
        tau = np.linspace(0, 2 * np.pi, N)
        # Define boundary of ellipse
        # these are the R and Z in equantion (8) in paper
        x_ellipse = np.asarray([1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)), 
                        eps * kappa * np.sin(tau)]).T
        xvals = x_ellipse
        uvals = np.zeros(len(xvals)).reshape(len(xvals), 1)
        return xvals, uvals


    def pde_solovev(x, u):
        # computes the residual of the Grad-Shafranov equation
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

    # changed center to (0,0)
    spatial_domain = dde.geometry.Ellipse(eps, kappa, delta) 

    x,u = gen_traindata(1001)   #  red dots in the boundary

    n_test = 100

    # specify psi, psi_r, psi_z, psi_rr, psi_zz at four locations 

    observe_x = np.asarray([[1 + eps, 0], 
                            [1 - eps, 0], 
                            [1 - delta * eps, kappa * eps],
                            [1 - delta * eps, -kappa * eps]]
                        )
    observe_y = np.asarray([0.0, 0.0, 0.0,0.0]).reshape(4, 1)

    observe_x = np.concatenate((x,observe_x))
    observe_y = np.concatenate((u,observe_y))

    bc135 = dde.PointSetBC(x,u)

    data = dde.data.PDE(
        spatial_domain,
        pde_solovev,
        [bc135],
        num_domain=1024,        # blue dots in paper
        num_boundary=100,        
        num_test=n_test,
        train_distribution="LHS"
    )

    # ----------------------------------------------------------------------------
    # Define model
    # ----------------------------------------------------------------------------
    

    DEPTH = 6  # 3
    BREADTH = 64  # 20
    LR = 2e-3
    AF = "swish"


    net = dde.maps.FNN([2] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

    model = dde.model.Model(data, net)

    print(type(model))


    # Target specifications (user-defined)
    TARGET_BETA_P = 1.0
    TARGET_VOLUME = 0.5  # arbitrary units



    # model.compile("adam", lr=LR)
    model.compile("L-BFGS-B", loss_weights=[1,100])

    # model.restore(MODEL_DIR, verbose=1)
    # model.print_model()

    loss_history, train_state = model.train(epochs=1000, display_every=10)

    if args.save:

        model.save(f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}/ITER", protocol="backend", verbose=1)

        dde.saveplot(
        loss_history, 
        train_state, 
        issave=True, 
        isplot=True,
        output_dir= f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_plots_new/run_{TIME}"
    )


    ITER = GS_Linear(eps=0.32, kappa=1.7, delta=0.33)
    ITER.get_BCs(A)
    ITER.solve_coefficients()
    x, y, psi_pred, psi_true, error = evaluate(ITER, model)

    result = optimise_shape(model=model,
                            ITER=ITER,
                            target_beta_p=TARGET_BETA_P,
                            target_volume=TARGET_VOLUME,
                            lambda_volume=1.0,
                            initial_guess=[0.3, 2.1, 0.1],
                            bounds=([0.1, 1.5, -0.5], [0.5, 3.0, 0.5]),
                            method="L-BFGS-B",
                            maxiter=300)

    # result = optimise_shape(model_path=MODEL_DIR,
    #                         target_beta_p=TARGET_BETA_P,
    #                         target_volume=TARGET_VOLUME,
    #                         lambda_volume=1.0,
    #                         initial_guess=[0.3, 2.1, 0.1],
    #                         bounds=([0.1, 1.5, -0.5], [0.5, 3.0, 0.5]),
    #                         method="L-BFGS-B",
    #                         maxiter=300)

    print("\nOptimisation finished:\n", result)



    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--target_beta_p", type=float, default=1.0)
    parser.add_argument("--target_volume", type=float, default=0.5)
    parser.add_argument("--lambda_volume", type=float, default=1.0)
    parser.add_argument("--initial_guess", type=list, default=[0.3, 2.1, 0.1])
    parser.add_argument("--bounds", type=list, default=([0.1, 1.5, -0.5], [0.5, 3.0, 0.5]))
    parser.add_argument("--method", type=str, default="L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()


"""
Optimisation finished:
   message: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
  success: True
   status: 0
      fun: 144.0
        x: [ 3.000e-01  2.100e+00  1.000e-01]
      nit: 0
      jac: [ 0.000e+00  0.000e+00  0.000e+00]
     nfev: 4
     njev: 1
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
"""