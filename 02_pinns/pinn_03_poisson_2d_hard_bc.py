r"""
==============================================================================
STAGE 2 - Script 3:  PINN in 2-D with "hard" boundary conditions
==============================================================================

PROBLEM  (gPINN paper, Yu et al. 2022)
    -(u_xx + u_yy) = f(x, y)   in  (0,1) x (0,1),      u = 0 on the boundary
    Exact: u(x, y) = [16 x (1-x) y (1-y)]^a       (a peak at the centre)
    We use a = 4 here (moderately peaked). The paper's a = 10 (very sharp) is
    revisited in 03_pinn_failures/fail_03 and 05_gradient_enhanced/gpinn_03.

HARD BOUNDARY CONDITIONS (output transform)
    Instead of penalising u on the boundary, we build the condition into the
    network:
            u_NN(x, y) = x (1-x) y (1-y) * N(x, y)
    where N is the raw network. Whatever the weights are, u_NN = 0 on the
    boundary EXACTLY. The loss then contains only the PDE residual:

            Loss = mean_j [ u_NN,xx + u_NN,yy + f ]^2   at random interior points

    Advantages: no boundary weight to tune, no boundary error.
    Limitation: you need a function that vanishes on the boundary; easy for
    boxes, harder for complex geometries or Neumann conditions.

WHAT YOU SHOULD SEE
    An accurate solution (about 0.1-0.2 % relative error) with only 400 random
    collocation points and no boundary loss at all.

HOW TO RUN
    python 02_pinns/pinn_03_poisson_2d_hard_bc.py
    python 02_pinns/pinn_03_poisson_2d_hard_bc.py --a 10      # sharper peak

THINGS TO TRY
    * --a 2 (smooth) vs --a 10 (very sharp): the error grows a lot with sharpness.
    * --n_col 100 / 2000 : effect of the number of collocation points.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_fields_2d, plot_history, plot_loss_terms  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("a", int, 4, "sharpness exponent of the exact solution"),
    ("n_col", int, 400, "number of random interior collocation points"),
])
LAYERS = [2, 20, 20, 20, 1]

setup(args.seed)
out = output_dir(__file__)


def u_exact(xy):
    return P.peak2d_u(xy, a=args.a)


def f(xy):
    return P.peak2d_f(xy, a=args.a)


# -----------------------------------------------------------------------------
# 2. Network with the boundary condition built in
# -----------------------------------------------------------------------------
def hard_bc(xy, raw):
    """u_NN = x(1-x) y(1-y) * raw  ->  exactly zero on the four edges."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return x * (1 - x) * y * (1 - y) * raw


model = MLP(LAYERS, activation="tanh", output_transform=hard_bc)

# Random interior points (uniform in the unit square). Column 0 = x, column 1 = y.
xy_col = torch.rand(args.n_col, 2).requires_grad_(True)
f_col = f(xy_col).detach()

# Test grid for measuring the error
n = 101
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
xy_test = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
u_test = u_exact(xy_test)


# -----------------------------------------------------------------------------
# 3. Loss: only the PDE residual (the boundary condition is automatic)
# -----------------------------------------------------------------------------
def laplacian(u, xy):
    g = grad(u, xy)                         # [u_x, u_y]
    u_xx = grad(g[:, 0:1], xy)[:, 0:1]      # d/dx of u_x
    u_yy = grad(g[:, 1:2], xy)[:, 1:2]      # d/dy of u_y
    return u_xx + u_yy


def loss_fn():
    u = model(xy_col)
    residual = laplacian(u, xy_col) + f_col      # u_xx + u_yy + f = 0
    loss_pde = torch.mean(residual ** 2)
    return {"total": loss_pde, "pde": loss_pde}


def error_fn():
    with torch.no_grad():
        return relative_l2(model(xy_test), u_test)


history = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                lbfgs_iters=args.lbfgs)

# -----------------------------------------------------------------------------
# 4. Plots and summary
# -----------------------------------------------------------------------------
with torch.no_grad():
    u_pred = model(xy_test).cpu().numpy().reshape(n, n)
U = u_test.cpu().numpy().reshape(n, n)
plot_fields_2d(X, Y, {"exact u": U, "PINN u": u_pred, "|error|": np.abs(u_pred - U)},
               out / "fields.png")
plot_history({"PINN (hard BC)": history}, out / "error_history.png")
plot_loss_terms(history, out / "loss.png")

save_summary(out, {
    "PINN rel. L2 error": relative_l2(u_pred, U),
    "max |error|": float(np.abs(u_pred - U).max()),
    "boundary error (exactly zero by construction)": float(np.abs(u_pred[0, :]).max()),
})
