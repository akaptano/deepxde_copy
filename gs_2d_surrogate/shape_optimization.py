"""cd gs_2d_surrogate/
source /scratch/yx3044/Projects/deepxde_copy/venv/bin/activate
conda activate deepxde_copy
python -u shape_optimization.py --lambda_vol=125 --plot True >> shape_125.txt"""

import time
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

import argparse
import matplotlib.pyplot as plt
from matplotlib.path import Path

dde.config.set_default_float("float64")

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
# spatial_domain = dde.geometry.HyperEllipticalToroid(
#     eps0, kappa0, delta0, Amax=Amax
# ) 


# x,u = gen_traindata(1001)   #  red dots in the boundary
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




# def torus_volume(area_cross_section: float, major_radius: float = 1.0) -> float:
#     """Toroidal volume *V = 2π R0 A_cs*."""
#     return 2 * np.pi * major_radius * area_cross_section




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


# def area(vs):
#     a = 0
#     x0, y0 = vs[0]
#     for x1, y1 in vs[1:] + [vs[0]]:   # ← loop back to the start
#         dx = x1 - x0
#         dy = y1 - y0
#         a += 0.5 * abs(y0 * dx - x0 * dy)
#         x0, y0 = x1, y1
#     return a




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

    # ------------------------------------------------------------------
    # 1. Reconstruct structured (R, Z, ψ) grids from the flattened input
    # ------------------------------------------------------------------
    N = psi_pred.shape[0]
    n = int(np.sqrt(N))
    if n * n != N:
        raise ValueError("X and psi_pred must correspond to a square R-Z grid.")

    R = X[:, 0].reshape(n, n)
    Z = X[:, 1].reshape(n, n)
    psi = psi_pred.reshape(n, n)

    # ------------------------------------------------------------------
    # 2. Extract the ψ = 0 contour to obtain the plasma boundary
    # ------------------------------------------------------------------
    # c = plt.contour(R, Z, psi, levels=[0.0])
    # if not c.collections or not c.collections[0].get_paths():
    #     # Fallback: if no closed contour is found return a large penalty
    #     # print("No closed contour found!!!!!!!!!!!!!!!")
    #     return np.inf
    # vertices = c.collections[0].get_paths()[0].vertices  # (N_v, 2)
    # plt.close(c.figure)  # prevent accumulation of hidden figures

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
    mu0 = 4.0 * np.pi * 1e-7
    Itor = 15e6          # Plasma current [A]
    a_minor = 2.0        # Minor radius [m]
    R0 = 6.2             # Major radius [m]
    B0 = 5.3             # Toroidal field on axis [T]

    eps = float(X[0, 3])  # inverse aspect-ratio (ϵ) from the input tensor

    # ------------------------------------------------------------------
    # 5. Compute q* and β_p following utils.utils.compute_params
    #    (integrate over the full R–Z grid; psi is ~0 outside the plasma)
    # ------------------------------------------------------------------
    psi_average = np.trapz(
        np.trapz(psi * R[:, 0], R[:, 0], axis=0), Z[0, :]
    )
    # psi_average = np.trapz(
    #     np.trapz(psi * R[0, :], R[0, :], axis=0), Z[:, 0]
    # )
    # print("R[0, :]", R[0, :])
    # print("R[:, 0]", R[:, 0])
    # print("Z[:, 0]", Z[:, 0])
    # print("psi_average!!!!!!!!!!!!", psi_average)
    # print("psi", psi, R.shape, Z.shape)
    # print("psi_average", psi_average)
    # psi0 = - mu0 * Itor * a_minor / eps / (-0.155 * q_int + 1.115 * area_cs)
    # qstar = - (a_minor * R0 * B0 * Cp_val) / (psi0 * (-0.155 * q_int + 1.115 * area_cs))

    denom = area_cs * (-0.155 * q_int + 1.115 * area_cs)

    beta_p = (2.0 * 1.155 * Cp_val ** 2 * psi_average) / (denom ** 2)

    return float(beta_p)


# ----------------------------------------------------------------------------
# Predict psi
# ----------------------------------------------------------------------------

grid_size = 300

# if zoom = 1, would get RuntimeWarning: invalid value encountered in arcsin 
#                          self.N1 = - (1 + np.arcsin(self.delta)) ** 2 / (self.eps * self.kappa ** 2)

# True beta_p: 1.0, Predicted beta_p: 1.005661103548024, True volume: 0.5, Predicted volume: 0.48012887330828863, Objective: 0.00042690976937949485
# eps: 0.1535847754093088, kappa: 1.583470712716403, delta: 0.6605555004375785

# if zoom = 2.2, would not get warning


"""plot the pretrained pinns boundary vs the analytic boundary, see if the pinns is well-trained

"""

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
    spatial = np.column_stack((RR.ravel(), ZZ.ravel()))
    A_col   = np.full((spatial.shape[0], 1), A)
    param_col = np.tile(np.array([[eps, kappa, delta]]), (spatial.shape[0], 1))
    X_in = np.hstack((spatial, A_col, param_col))

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

    #-------------- plot the grid and psi solution in 3D --------------!!!!!

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

        # # Add 2D heatmap plots
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # # Plot predicted psi heatmap
        # im1 = ax1.pcolormesh(RR, ZZ, psi_pred_grid, cmap='viridis', shading='auto')
        # ax1.set_xlabel('R')
        # ax1.set_ylabel('Z')
        # ax1.set_title('Predicted ψ')
        # fig.colorbar(im1, ax=ax1)

        # # Plot true psi heatmap  
        # im2 = ax2.pcolormesh(RR, ZZ, psi_true_grid, cmap='viridis', shading='auto')
        # ax2.set_xlabel('R') 
        # ax2.set_ylabel('Z')
        # ax2.set_title('True ψ')
        # fig.colorbar(im2, ax=ax2)

        # plt.tight_layout()
        # plt.savefig('/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/psi_2d_comparison1.png')
        # plt.close()

    # assert False

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
        R, Z, psi_pred_grid, psi_true_grid, X_in, psi_pred_flat = predict_psi(model, eps=eps, kappa=kappa, delta=delta, return_X=True, plot_psi=False)
        # visualize_contours(R, Z, psi_pred_grid, eps, kappa, delta, savepath=f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/psi_contour/{eps}_{kappa}_{delta}.png")

        # offset = psi_pred_flat.mean()
        # psi_pred_flat = psi_pred_flat - offset
        

        # plot analytic vs c.collections[0].get_paths()[0].vertices

        # Build analytic Solov'ev boundary (always available as fallback)
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

        # def analytical():
            # x_anal = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
            # y_anal = eps * kappa * np.sin(tau)
            # vertices = np.column_stack((x_anal, y_anal))

            # plt.close(c.figure)


        # c = plt.contour(R, Z, psi_pred_grid, levels=[0.0])
        # paths = c.collections[0].get_paths()
        # plt.close(c.figure)

        # Keep only closed loops
        # closed_paths = [p for p in paths if np.allclose(p.vertices[0], p.vertices[-1])]
        # closed_paths = []
        # tol = 1
        # for p in paths:
        #     v0, v1 = p.vertices[0], p.vertices[-1]
        #     if np.linalg.norm(v0 - v1) < tol:
        #         closed_paths.append(p)

        # closed_paths = [p for p in paths if p.codes is not None and Path.CLOSEPOLY in p.codes]
        # if not closed_paths:
        #     print("No closed ψ=0 contour found!")
        #     centroids = [p.vertices.mean(axis=0) for p in paths]
        #     dists = [np.linalg.norm(c - np.array([1.0, 0.0])) for c in centroids]
        #     vertices = paths[np.argmin(dists)].vertices
        # else:

            # Pick the loop closest to the magnetic axis (default axis at R=1, Z=0)
        # centroids = [p.vertices.mean(axis=0) for p in closed_paths]
        # dists = [np.linalg.norm(c - np.array([1.0, 0.0])) for c in centroids]
        # vertices = closed_paths[np.argmin(dists)].vertices

        volume = area(vertices)
        beta_p = compute_beta_p(model, X_in, psi_pred_flat, vertices)

        # obj = lambda_vol * (volume - target_volume) ** 2
        obj = (beta_p - target_beta_p) ** 2 + lambda_vol * (volume - target_volume) ** 2
        # obj = ((beta_p - target_beta_p)/target_beta_p) ** 2 + lambda_vol * ((volume - target_volume)/target_volume) ** 2

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

    # def _volume_objective(params: Sequence[float]) -> float:
    #     eps, kappa, delta = params
    #     R_bnd, Z_bnd = solov_ev_boundary(tau, eps, kappa, delta, r0=major_radius)
    #     area_cs = polygon_area(R_bnd, Z_bnd)
    #     volume = torus_volume(area_cs, major_radius)
        
    #     obj = (volume - target_volume) ** 2
    #     return float(obj)



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
    parser.add_argument("--lambda_volume", type=float, default=125)
    parser.add_argument("--initial_guess", type=list, default=[0.32, 1.7, 0.33])
    parser.add_argument("--bounds", type=list, default=([eps0[0], kappa0[0], delta0[0]], [eps0[1], kappa0[1], delta0[1]]))
    parser.add_argument("--method", type=str, default="L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=15000)
    parser.add_argument("--train_new", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--use_fallback", type=bool, default=False)
    args = parser.parse_args()

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


    DEPTH = 4
    BREADTH = 40
    # LR = 2e-3
    LR = 2e-2
    AF = "swish"


    net = dde.maps.FNN([6] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
    

    if args.train_new:
        spatial_domain = dde.geometry.HyperEllipticalToroid(
            eps0, kappa0, delta0, Amax=Amax
        ) 
        x, u = gen_traindata(1001)
        bc135 = dde.PointSetBC(x, u)
        data = dde.data.PDE(spatial_domain, pde_solovev, [bc135],
                            num_domain=1028, num_boundary=100,
                            num_test=n_test, train_distribution="LHS")
        model = dde.Model(data=data, net=net)
        model.compile(args.method, lr=LR, loss_weights=[1,100])
        loss_history, train_state = model.train(epochs=1000, display_every=10)
        model.save(f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new/run_{TIME}/ITER", protocol="backend", verbose=1)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir= f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_plots_new/run_{TIME}")
    else:
        # Create dummy data for inference only
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


    ITER = GS_Linear(eps=eps0[0], kappa=kappa0[0], delta=delta0[0])
    ITER.get_BCs(A)
    ITER.solve_coefficients()
    # x, y, psi_pred, psi_true, error = evaluate(ITER, model)

    # print(x.shape, y.shape, psi_pred.shape, psi_true.shape, error.shape)
    # print(psi_true.min(), psi_true.max())

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

    end_run_time = time.time()
    print(f"Run time: {end_run_time - start_run_time} seconds")

    # ------------------------------------------------------------
    # Plot optimisation diagnostics collected in *metrics*
    # ------------------------------------------------------------
    
    if args.plot:
        plot_save_path = f"/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/plots/run_plots/plots_lambda_{args.lambda_volume}/"
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
            
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
            fig.savefig(plot_save_path + "metrics_beta_volume.png", dpi=150)
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
            plt.savefig(plot_save_path + "params.png", dpi=150)
            plt.close()

            # 3. Objective value
            plt.figure()
            plt.yscale("log")
            plt.plot(iters, metrics["obj"], label="Objective")
            plt.xlabel("Function evaluation")
            plt.ylabel("Objective value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_save_path + "obj.png", dpi=150)
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

"""
 Optimisation finished:
   message: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
  success: True
   status: 0
      fun: 1.032508804376499e-05
        x: [-3.513e-01  1.067e+00  6.993e-01]
      nit: 17
      jac: [ 3.986e-02  4.243e-03  3.104e-03]
     nfev: 148
     njev: 37
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
"""


"""
Changed zoom to 2.5, avoided log out of bounds error
Optimisation finished:
   message: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
  success: True
   status: 0
      fun: 0.011574628038271993
        x: [ 7.825e-02  1.719e+00  3.117e-01]
      nit: 9
      jac: [ 3.378e+00 -8.408e-02  2.261e-02]
     nfev: 92
     njev: 23
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
 """