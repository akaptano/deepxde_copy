#!/usr/bin/env python3
"""boundary_difference_plot.py

Visualise and quantify the difference between the ψ = 0 contour predicted by a
pre-trained PINN and the analytic Solov'ev boundary for a single set of plasma
shape parameters (ε, κ, δ).

Usage (all arguments optional::

    python boundary_difference_plot.py \
        --eps 0.32 --kappa 1.7 --delta 0.33 \
        --checkpoint PATH/TO/ITER-XXX.ckpt \
        --n_boundary 400

If no checkpoint is given the default used in *check_pretrain.py* is loaded.
The script produces a two-panel figure:
  • left: predicted vs analytic boundary in the R–Z plane;
  • right: point-wise Euclidean distance (error) versus poloidal angle τ.
The mean and maximum errors are printed to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import time

# -----------------------------------------------------------------------------
# DeepXDE import (ensure local version has priority)
# -----------------------------------------------------------------------------
DEEPEXDE_PATH = Path(__file__).resolve().parents[1]  # /gs_2d_surrogate -> project
if str(DEEPEXDE_PATH) not in sys.path:
    sys.path.insert(0, str(DEEPEXDE_PATH))
import deepxde as dde  # noqa: E402  pylint: disable=wrong-import-position

dde.config.set_default_float("float64")

# -----------------------------------------------------------------------------
# Local helpers
# -----------------------------------------------------------------------------
from shape_optimization import (
    pde_solovev,
    predict_psi,
)  # noqa: E402  pylint: disable=wrong-import-position
from utils.gs_solovev_sol import GS_Linear  # noqa: E402  pylint: disable=wrong-import-position

A_CONST = -0.155  # fixed in current data-set

# Network hyper-parameters must match those used for the checkpoint
DEPTH = 4
WIDTH = 40
ACTIVATION = "swish"
LOSS_WEIGHTS = [1, 100]

# Reasonable default checkpoint path (can be overridden via CLI)
DEFAULT_CKPT = (
    "/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/"
    "saved_models_new/run_08142025_2320_parametrized/ITER-16001.ckpt"
)


def _build_inference_model() -> dde.Model:
    """Return a DeepXDE model ready for inference (weights loaded later)."""
    net = dde.maps.FNN([6] + DEPTH * [WIDTH] + [1], ACTIVATION, "Glorot normal")

    # Dummy geometry/data – only one point is required for restore + predict
    # (DeepXDE enforces presence of *data* even for inference-only usage).
    eps0 = (0.12, 0.52)  # loose bounds
    kappa0 = (1.5, 1.9)
    delta0 = (0.0, 0.6)
    Amax = 0.2
    geom = dde.geometry.HyperEllipticalToroid(eps0, kappa0, delta0, Amax=Amax)

    data = dde.data.PDE(
        geom,
        pde_solovev,
        [],
        num_domain=1,
        num_boundary=0,
        num_test=1,
    )

    model = dde.Model(data, net)
    # Compile with external optimizer to avoid creating Adam variables that
    # are not present in the checkpoint. Learning-rate parameter is ignored.
    model.compile("L-BFGS-B", loss_weights=LOSS_WEIGHTS)
    return model


# -----------------------------------------------------------------------------
# Distance metric helpers
# -----------------------------------------------------------------------------

def _nearest_distance(set_a: np.ndarray, set_b: np.ndarray) -> np.ndarray:
    """For each point in *set_a* return distance to the nearest point in *set_b*."""
    from scipy.spatial import cKDTree

    tree = cKDTree(set_b)
    dists, _ = tree.query(set_a)
    return dists


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    TIME = time.strftime("%m%d_%H%M")
    parser = argparse.ArgumentParser(description="Compare PINN and analytic boundaries.")
    parser.add_argument(
        "--eps", type=float, default=0.32, help="Inverse aspect ratio ε.")
    parser.add_argument("--kappa", type=float, default=1.7, help="Elongation κ.")
    parser.add_argument("--delta", type=float, default=0.33, help="Triangularity δ.")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to model checkpoint (*.ckpt)."
    )
    parser.add_argument("--n_boundary", type=int, default=400, help="Resolution of analytic boundary.")
    parser.add_argument("--out", type=str, default=f"boundary_comparison_{TIME}.png", help="Output figure file.")

    args = parser.parse_args(argv)

    

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = _build_inference_model()
    print("Restoring weights from", args.checkpoint)
    model.restore(args.checkpoint, verbose=1)

    # ------------------------------------------------------------------
    # 2. Predict ψ field and extract PINN boundary
    # ------------------------------------------------------------------
    Rg, Zg, psi_pred_grid, _ = predict_psi(
        model, eps=args.eps, kappa=args.kappa, delta=args.delta, A=A_CONST, plot_psi=False
    )

    cont = plt.contour(Rg, Zg, psi_pred_grid, levels=[0.0])
    if not cont.collections or not cont.collections[0].get_paths():
        raise RuntimeError("No closed ψ = 0 contour found in PINN prediction.")
    vertices_pred = cont.collections[0].get_paths()[0].vertices  # (N_pred, 2)
    plt.close(cont.figure)

    # ------------------------------------------------------------------
    # 3. Analytic boundary (ground truth)
    # ------------------------------------------------------------------
    tau = np.linspace(0.0, 2 * np.pi, args.n_boundary, endpoint=False)
    x_anal = 1 + args.eps * np.cos(tau + np.arcsin(args.delta) * np.sin(tau))
    y_anal = args.eps * args.kappa * np.sin(tau)
    vertices_anal = np.column_stack((x_anal, y_anal))

    # ------------------------------------------------------------------
    # 4. Distance error evaluation
    # ------------------------------------------------------------------
    # For each analytic vertex find nearest PINN vertex (symmetrised error is optional)
    err_a_to_p = _nearest_distance(vertices_anal, vertices_pred)
    mean_err = float(err_a_to_p.mean())
    max_err = float(err_a_to_p.max())

    print(f"Mean boundary error  = {mean_err:.4e} R0")
    print(f"Max  boundary error  = {max_err:.4e} R0")

    # Calculate percent error relative to plasma minor radius (eps)
    # Calculate relative error as percentage of max predicted flux
    psi_max = np.max(np.abs(psi_pred_grid))
    mean_pct_err = 100 * mean_err / args.eps
    max_pct_err = 100 * max_err / args.eps

    print(f"Mean boundary error  = {mean_pct_err:.2f}% (of minor radius)")
    print(f"Max  boundary error  = {max_pct_err:.2f}% (of minor radius)")

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: overlay of boundaries
    ax1.plot(vertices_anal[:, 0], vertices_anal[:, 1], "b-", label="Analytic")
    ax1.plot(vertices_pred[:, 0], vertices_pred[:, 1], "r--", label="PINN")
    ax1.set_aspect("equal")
    ax1.set_xlabel("R / R₀")
    ax1.set_ylabel("Z / R₀")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(
        "Plasma boundary (ε={:.3f}, κ={:.3f}, δ={:.3f})".format(args.eps, args.kappa, args.delta)
    )

    # Right: point-wise error
    ax2.plot(tau, err_a_to_p, color="purple")
    ax2.set_xlabel("Poloidal angle τ [rad]")
    ax2.set_ylabel("|ΔRZ| (distance)")
    ax2.set_title("Distance between analytic and PINN contour")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Figure saved to", args.out)


if __name__ == "__main__":
    main()
