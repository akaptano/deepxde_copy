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
import time
import argparse
import matplotlib.pyplot as plt

# ----------------------------
# Global store for optimisation diagnostics
# ----------------------------
metrics = {
    "beta_p_pred": [],  # predicted β_p each function evaluation
    "volume_pred": [],  # predicted volume each function evaluation
    "eps": [],          # ε value proposed
    "kappa": [],        # κ value proposed
    "delta": [],         # δ value proposed
    "obj": []           # objective values
}

from utils.utils import *
from utils.gs_solovev_sol import GS_Linear



# ----------------------------------------------------------------------------
# ITER Configuration
# ----------------------------------------------------------------------------
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

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 + eps, 0]).all()
def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 - eps, 0]).all()
def boundary_high(x, on_boundary):
    return on_boundary and np.isclose([x[0], x[1]], [1 - delta * eps, kappa * eps]).all()

# changed center to (0,0)
spatial_domain = dde.geometry.HyperEllipticalToroid(
    eps0, kappa0, delta0, Amax=Amax
) 


x,u = gen_traindata(1001)   #  red dots in the boundary
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


# # specify psi, psi_r, psi_z, psi_rr, psi_zz at four locations 
# observe_x = np.asarray([[1 + eps, 0], 
#                         [1 - eps, 0], 
#                         [1 - delta * eps, kappa * eps],
#                         [1 - delta * eps, -kappa * eps]]
#                     )
# observe_y = np.asarray([0.0, 0.0, 0.0,0.0]).reshape(4, 1)
# observe_x = np.concatenate((x,observe_x))
# observe_y = np.concatenate((u,observe_y))

# bc135 = dde.PointSetBC(x,u)

# data = dde.data.PDE(
#     spatial_domain,
#     pde_solovev,
#     [bc135],
#     num_domain=1024,        # blue dots in paper
#     num_boundary=100,        
#     num_test=n_test,
#     train_distribution="LHS"
# )






# ----------------------------------------------------------------------------
# Geometry helper routines specific to Solovev parameterisation
# ----------------------------------------------------------------------------

# def solov_ev_boundary(tau: np.ndarray,
#                       eps: float,
#                       kappa: float,
#                       delta: float,
#                       r0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
#     """Return (R(tau), Z(tau)) for the ITER-like Solovev shape.

#     The parametrisation matches that used in *gen_traindata* of
#     ``gs-2d-surrogate/general_solovev_equil_parametrized_shape.py``::

#         R = R0 + eps * cos(tau + asin(delta) * sin(tau))
#         Z = eps * kappa * sin(tau)

#     Args:
#         tau: 1-D array of poloidal angles (0..2π).
#         eps: Inverse aspect ratio ϵ.
#         kappa: Elongation κ.
#         delta: Triangularity δ.
#         r0: Major radius R0 (default 1.0).
#     """
#     tau = np.asarray(tau)
#     R = r0 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
#     Z = eps * kappa * np.sin(tau)
#     return R, Z


# def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
#     """Polygon area via the shoelace formula (positive orientation)."""
#     return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# def torus_volume(area_cross_section: float, major_radius: float = 1.0) -> float:
#     """Toroidal volume *V = 2π R0 A_cs*."""
#     return 2 * np.pi * major_radius * area_cross_section

# ----------------------------------------------------------------------------
# Physics metrics helper functions
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


def compute_beta_p(model: dde.Model,
                   X: np.ndarray,
                   psi_pred: np.ndarray,
                   vertices: np.ndarray) -> float:
    """Compute poloidal beta β_p from the predicted stream-function.

    The implementation follows the same procedure used in
    ``utils.utils.compute_params`` for the *predicted* β_p value.  The
    required geometric integrals are evaluated directly from the
    (R, Z) grid and the *ψ* field supplied via *X* and *psi_pred*.
    """

    import numpy as _np
    import matplotlib.pyplot as _plt

    # ------------------------------------------------------------------
    # 1. Reconstruct structured (R, Z, ψ) grids from the flattened input
    # ------------------------------------------------------------------
    N = psi_pred.shape[0]
    n = int(_np.sqrt(N))
    if n * n != N:
        raise ValueError("X and psi_pred must correspond to a square R-Z grid.")

    R = X[:, 0].reshape(n, n)
    Z = X[:, 1].reshape(n, n)
    psi = psi_pred.reshape(n, n)

    # ------------------------------------------------------------------
    # 2. Extract the ψ = 0 contour to obtain the plasma boundary
    # ------------------------------------------------------------------
    # c = _plt.contour(R, Z, psi, levels=[0.0])
    # if not c.collections or not c.collections[0].get_paths():
    #     # Fallback: if no closed contour is found return a large penalty
    #     # print("No closed contour found!!!!!!!!!!!!!!!")
    #     return _np.inf
    # vertices = c.collections[0].get_paths()[0].vertices  # (N_v, 2)
    # _plt.close(c.figure)  # prevent accumulation of hidden figures

    # ------------------------------------------------------------------
    # 3. Geometric integrals (now using shared helpers)
    # ------------------------------------------------------------------
    area_cs = area(vertices)
    Cp_val = Cp(vertices)
    q_int = qstar_integral(vertices)
    # print("area_cs", area_cs, "Cp_val", Cp_val)

    # ------------------------------------------------------------------
    # 4. Physical constants and shape parameters (ITER defaults)
    # ------------------------------------------------------------------
    mu0 = 4.0 * _np.pi * 1e-7
    Itor = 15e6          # Plasma current [A]
    a_minor = 2.0        # Minor radius [m]
    R0 = 6.2             # Major radius [m]
    B0 = 5.3             # Toroidal field on axis [T]

    eps = float(X[0, 3])  # inverse aspect-ratio (ϵ) from the input tensor

    # ------------------------------------------------------------------
    # 5. Compute q* and β_p following utils.utils.compute_params
    #    (integrate over the full R–Z grid; psi is ~0 outside the plasma)
    # ------------------------------------------------------------------
    psi_average = _np.trapz(
        _np.trapz(psi * R[0, :], R[0, :], axis=0), Z[:, 0]
    )
    # print("psi", psi, R.shape, Z.shape)
    # print("psi_average", psi_average)
    # psi0 = - mu0 * Itor * a_minor / eps / (-0.155 * q_int + 1.115 * area_cs)
    # qstar = - (a_minor * R0 * B0 * Cp_val) / (psi0 * (-0.155 * q_int + 1.115 * area_cs))

    # beta_p = (2.0 * 1.155 * Cp_val ** 2 * psi_average) / (
    #     area_cs * (-0.155 * q_int + 1.115 * area_cs) ** 2
    # )
    beta_p = 2 * 1.155 * Cp_val ** 2 * 0.018170271593863394 / (
        area_cs * (-0.155 * q_int + 1.115 * area_cs) ** 2
    )

    return float(beta_p)


# ----------------------------------------------------------------------------
# Predict psi
# ----------------------------------------------------------------------------

def predict_psi(
    model,
    eps, kappa, delta,
    A=-0.155,
    r_range=(0.5, 1.5),
    z_range=(-1.5, 1.5),
    n=200,                      # grid resolution
    return_X: bool = False,    # optionally return the input tensor
):
    r = np.linspace(*r_range, n)
    z = np.linspace(*z_range, n)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")

    # (n², 2) spatial part
    spatial = np.column_stack((RR.ravel(), ZZ.ravel()))

    # constant columns that carry the parameters
    A_col     = np.full((spatial.shape[0], 1), A)
    param_col = np.tile(np.array([[eps, kappa, delta]]),
                        (spatial.shape[0], 1))

    X_in = np.hstack((spatial, A_col, param_col))   # (n², 6)

    psi_flat = model.predict(X_in)
    psi_grid = psi_flat.reshape(n, n)
    if return_X:
        return RR, ZZ, psi_grid, X_in, psi_flat  # both grid and flat forms
    else:
        return RR, ZZ, psi_grid


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
        # 1. Model evaluation inside domain
        # --------------------------------------------------------------
        R, Z, psi_grid, X_in, psi_flat = predict_psi(model, eps=eps, kappa=kappa, delta=delta, return_X=True)

        # --------------------------------------------------------------
        # 2. Geometry
        # --------------------------------------------------------------
        # x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
        # y = eps * kappa * np.sin(tau)
        # vertices = np.column_stack((x, y))

        # Compute vertices from psi = 0 flux surface for predicted psi
        print(psi_grid.min(), psi_grid.max())      # do you straddle 0?
        plt.figure()
        plt.imshow(psi_grid.T, origin='lower')
        plt.colorbar()
        plt.savefig("psi_grid.png")
        plt.show()

        c = plt.contour(R, Z, psi_grid, [0.0])
        if not c.collections or not c.collections[0].get_paths():
        # No closed ψ = 0 contour – treat as unacceptable shape
            plt.close(c.figure)            # avoid hidden figures
            return np.inf                  # or another large penalty value

        vertices = c.collections[0].get_paths()[0].vertices

        volume = area(vertices)
        beta_p = compute_beta_p(model, X_in, psi_flat, vertices)

        # --------------------------------------------------------------
        # 3. Objective value
        # --------------------------------------------------------------
        obj = (beta_p - target_beta_p) ** 2 + lambda_vol * (volume - target_volume) ** 2

        # Record diagnostics for later plotting
        metrics["beta_p_pred"].append(beta_p)
        metrics["volume_pred"].append(volume)
        metrics["eps"].append(eps)
        metrics["kappa"].append(kappa)
        metrics["delta"].append(delta)
        metrics["obj"].append(obj)

        print(f"True beta_p: {target_beta_p}, Predicted beta_p: {beta_p}, True volume: {target_volume}, Predicted volume: {volume}, Objective: {obj}")
        print(f"eps: {eps}, kappa: {kappa}, delta: {delta}\n")
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

        # Record diagnostics for later plotting (β_p not evaluated here)
        metrics["beta_p_pred"].append(float('nan'))
        metrics["volume_pred"].append(pred_volume)
        metrics["eps"].append(eps)
        metrics["kappa"].append(kappa)
        metrics["delta"].append(delta)
        metrics["obj"].append(obj)

        print(f"True volume: {target_volume}, Predicted volume: {pred_volume}, Objective: {obj}")
        print(f"eps: {eps}, kappa: {kappa}, delta: {delta}")
        return float(obj)
    return _volume_objective




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
                   maxiter: int = 200,
                   objective_type: str = "volume") -> OptimizeResult:
    """Optimise (eps, kappa, delta) to minimise f(psi_pred).

    Args:
        model_path: Directory containing saved DeepXDE model.
        target_beta_p: Desired β_p.
        target_volume: Desired toroidal volume.
        lambda_volume: Weight λ in objective.
        initial_guess: Starting [eps, kappa, delta].
        bounds: Tuple (lower, upper) for each parameter.
        method: SciPy optimisation method.
        maxiter: Maximum iterations.
        objective_type: Type of objective function to use.
    """
    # ------------------ Load pretrained model --------------------
    # model = dde.models.Model.restore(model_path) # gibberish

    # ------------------ Initial guess ----------------------------
    if initial_guess is None:
        initial_guess = [0.32, 1.7, 0.33]  # ITER-like defaults
    initial_guess = np.asarray(initial_guess, dtype=float)

    if objective_type == "volume":
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


    if objective_type == "beta_p":
        objective = make_objective(model,
                                   target_beta_p=target_beta_p,
                                   target_volume=target_volume,
                                   lambda_vol=lambda_volume,
                                   n_boundary=400,
                                   n_grid=32,
                                   major_radius=1.0)

        options = dict(maxiter=maxiter, disp=True, ftol=1e-09, gtol=1e-05)

        res = minimize(objective,
                    x0=initial_guess,
                    method=method,
                    # bounds=None if bounds is None else list(zip(*bounds)),
                    options=options)

    else:
        raise ValueError(f"Invalid objective type: {objective_type}")


    return res



if __name__ == "__main__":
    TIME = time.strftime("%m%d%H%M")

    # ----------------------------------------------------------------------------
    # Parse command line arguments
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--target_beta_p", type=float, default=1.0)
    parser.add_argument("--target_volume", type=float, default=0.5)
    parser.add_argument("--lambda_volume", type=float, default=1.0)
    parser.add_argument("--initial_guess", type=list, default=[0.32, 1.7, 0.33])
    parser.add_argument("--bounds", type=list, default=([eps0[0], kappa0[0], delta0[0]], [eps0[1], kappa0[1], delta0[1]]))
    parser.add_argument("--method", type=str, default="L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=15000)
    parser.add_argument("--train_new", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # Define model
    # ----------------------------------------------------------------------------
    
    if args.model_path is not None:
        CHECKPOINT_PATH = args.model_path
    else:
        CHECKPOINT_PATH = "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_07092025_2333_parametrized/ITER-539.ckpt"

    DEPTH = 4
    BREADTH = 40
    LR = 2e-3
    AF = "swish"


    net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    model = dde.Model(data=data, net=net)

    if args.train_new:
        model.compile(args.method, lr=LR, loss_weights=[1,100])
        loss_history, train_state = model.train(epochs=1000, display_every=10)
        model.save(f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}/ITER", protocol="backend", verbose=1)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir= f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_plots_new/run_{TIME}")
    else:
        model.compile("adam", lr=LR, loss_weights=[1,100])
        model.restore(CHECKPOINT_PATH, verbose=1)


    ITER = GS_Linear(eps=0.32, kappa=1.7, delta=0.33)
    ITER.get_BCs(A)
    ITER.solve_coefficients()
    # x, y, psi_pred, psi_true, error = evaluate(ITER, model)

    result = optimise_shape(model=model,
                            ITER=ITER,
                            target_beta_p=args.target_beta_p,
                            target_volume=args.target_volume,
                            lambda_volume=args.lambda_volume,
                            initial_guess=args.initial_guess,
                            bounds=args.bounds,
                            method=args.method,
                            maxiter=args.maxiter,
                            objective_type="beta_p")


    print("\nOptimisation finished:\n", result)

    # ------------------------------------------------------------
    # Plot optimisation diagnostics collected in *metrics*
    # ------------------------------------------------------------
    if args.plot:
        if metrics["beta_p_pred"]:
            iters = range(len(metrics["beta_p_pred"]))

            # 1. β_p and volume on shared x-axis with twin y-axes
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("Function evaluation")
            ax1.set_ylabel("Predicted β_p", color="tab:red")
            ax1.plot(iters, metrics["beta_p_pred"], color="tab:red", label="β_p (pred)")
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel("Predicted volume", color="tab:blue")
            ax2.plot(iters, metrics["volume_pred"], color="tab:blue", label="Volume (pred)")
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            fig.tight_layout()
            fig.savefig(f"optim_metrics_beta_volume_{TIME}.png", dpi=150)
            plt.close(fig)

            # 2. ε, κ, δ evolution
            plt.figure()
            plt.plot(iters, metrics["eps"],   label="eps (ε)")
            plt.plot(iters, metrics["kappa"], label="kappa (κ)")
            plt.plot(iters, metrics["delta"], label="delta (δ)")
            plt.xlabel("Function evaluation")
            plt.ylabel("Parameter value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"optim_params_{TIME}.png", dpi=150)
            plt.close()

            # 3. Objective value
            plt.figure()
            plt.plot(iters, metrics["obj"], label="Objective")
            plt.xlabel("Function evaluation")
            plt.ylabel("Objective value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"optim_obj_{TIME}.png", dpi=150)
            plt.close()



"""
------------------- Optimization for volume ----------------------
Optimisation finished:
   message: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
  success: True
   status: 0
      fun: 6.893893867343462e-13
        x: [ 1.525e-01  1.630e+00 -2.778e-02]
      nit: 6
      jac: [ 5.678e-06  5.102e-07  4.095e-08]
     nfev: 28
     njev: 7
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
"""



"""
------------------ Optimization for beta_p -----------------------
Optimisation finished:
   message: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
  success: True
   status: 0
      fun: 1.0000000000003406
        x: [ 1.597e-01  1.534e+00  2.568e-01]
      nit: 5
      jac: [-3.686e-06 -3.775e-07 -2.220e-08]
     nfev: 24
     njev: 6
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
 """