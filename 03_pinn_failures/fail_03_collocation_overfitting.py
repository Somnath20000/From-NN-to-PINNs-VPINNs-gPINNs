r"""
==============================================================================
STAGE 3 - Script 3:  PINN failure #3 - "overfitting" the collocation points
==============================================================================

PROBLEM  (same as 02_pinns/pinn_03, but with the sharp peak of the gPINN paper)
    -(u_xx + u_yy) = f   in (0,1)^2,   u = 0 on the boundary
    Exact: u(x, y) = [16 x (1-x) y (1-y)]^a,   a = 10  (narrow, steep peak)
    Boundary condition built in exactly (hard constraint), 400 random
    collocation points - exactly the setting that works well for a = 4.

WHAT WE MEASURE
    * the PDE residual loss at the 400 TRAINING collocation points
      (this is what the optimiser sees and minimises),
    * the PDE residual on 10 000 NEW points the optimiser never saw,
    * the true relative L2 error of u.

WHAT YOU SHOULD SEE
    The residual at the training points becomes small, but the residual at
    unseen points stays thousands of times larger, and the solution error
    stays in the 10-20 % range. The residual map shows "holes" around the collocation points:
    the network satisfies the PDE where it is checked, not in between.
    This is the PINN analogue of overfitting in machine learning.

REMEDIES (discussed in the README)
    more / better-placed points, resampling the points during training,
    gradient-enhanced PINNs (Stage 5 revisits this exact problem), or
    weak/variational formulations that test the PDE in an integral sense
    (Stage 4).

HOW TO RUN
    python 03_pinn_failures/fail_03_collocation_overfitting.py
    python 03_pinn_failures/fail_03_collocation_overfitting.py --lbfgs 2000
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import (COLORS, FIELD_FIGSIZE, domain_ticks, finish,  # noqa: E402
                             new_figure, plot_fields_2d, plot_history, style_colorbar)
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("a", int, 10, "sharpness exponent of the exact solution"),
    ("n_col", int, 400, "number of random collocation points"),
])
LAYERS = [2, 20, 20, 20, 1]

setup(args.seed)
out = output_dir(__file__)
u_exact = lambda xy: P.peak2d_u(xy, a=args.a)   # noqa: E731
f = lambda xy: P.peak2d_f(xy, a=args.a)         # noqa: E731


def hard_bc(xy, raw):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return x * (1 - x) * y * (1 - y) * raw


model = MLP(LAYERS, activation="tanh", output_transform=hard_bc)


def pde_residual(xy):
    """r = u_xx + u_yy + f   (zero for the exact solution)."""
    u = model(xy)
    g = grad(u, xy)
    u_xx = grad(g[:, 0:1], xy)[:, 0:1]
    u_yy = grad(g[:, 1:2], xy)[:, 1:2]
    return u_xx + u_yy + f(xy)


xy_train = torch.rand(args.n_col, 2).requires_grad_(True)   # what the optimiser sees
xy_unseen = torch.rand(10000, 2).requires_grad_(True)       # never used in training

n = 101
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
xy_test = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
u_test = u_exact(xy_test)

unseen_history = []


def loss_fn():
    loss = torch.mean(pde_residual(xy_train) ** 2)
    return {"total": loss, "pde (training points)": loss}


def error_fn():
    # also record the residual at unseen points, to compare with the training loss
    unseen_history.append(float(torch.mean(pde_residual(xy_unseen) ** 2).detach()))
    with torch.no_grad():
        return relative_l2(model(xy_test), u_test)


history = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                record_every=500, lbfgs_iters=args.lbfgs)
history["pde (unseen points)"] = unseen_history

# -----------------------------------------------------------------------------
# 2. Plots
# -----------------------------------------------------------------------------
fig, ax = new_figure()
ax.semilogy(history["step"], history["pde (training points)"], color=COLORS[1],
            label=f"PDE residual at the {args.n_col} training points")
ax.semilogy(history["step"], history["pde (unseen points)"], "--", color=COLORS[0],
            label="PDE residual at 10 000 unseen points")
finish(fig, ax, out / "train_vs_unseen_residual.png", "iteration", "mean squared residual",
       "Training residual vs. residual where nobody looked")

xy_grid = xy_test.clone().requires_grad_(True)
res_map = pde_residual(xy_grid).detach().cpu().numpy().reshape(n, n)
log_res = np.log10(np.abs(res_map) + 1e-12)
# clip the colour range: a few exact zero-crossings would otherwise squash the scale
levels = np.linspace(np.percentile(log_res, 2), log_res.max(), 60)
fig, ax = new_figure(FIELD_FIGSIZE)
im = ax.contourf(X, Y, np.clip(log_res, levels[0], None), levels, cmap="magma")
pts = xy_train.detach().cpu().numpy()
ax.plot(pts[:, 0], pts[:, 1], "o", ms=5, mfc="cyan", mec="k", mew=0.6,
        label="collocation points")
cb = fig.colorbar(im, ax=ax, shrink=1, pad=0.02)
style_colorbar(cb)
cb.set_label(r"$\log_{10}$ |PDE residual|", fontsize=24)
ax.set_aspect("equal")
domain_ticks(ax, X, Y)
finish(fig, ax, out / "residual_map.png", "x", "y", "Residual is small only near the points",
       grid=False)

with torch.no_grad():
    u_pred = model(xy_test).cpu().numpy().reshape(n, n)
U = u_test.cpu().numpy().reshape(n, n)
plot_fields_2d(X, Y, {"exact u": U, "PINN u": u_pred, "|error|": np.abs(u_pred - U)},
               out / "fields.png")
plot_history({"PINN": history}, out / "error_history.png")

save_summary(out, {
    "PDE loss at training points": history["pde (training points)"][-1],
    "PDE loss at unseen points": history["pde (unseen points)"][-1],
    "ratio unseen / training": history["pde (unseen points)"][-1]
    / history["pde (training points)"][-1],
    "rel. L2 error of u": relative_l2(u_pred, U),
})
