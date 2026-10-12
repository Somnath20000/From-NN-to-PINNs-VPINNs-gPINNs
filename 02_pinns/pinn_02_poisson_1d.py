r"""
==============================================================================
STAGE 2 - Script 2:  PINN for a boundary-value problem (1-D Poisson)
==============================================================================

PROBLEM  (benchmark from the gPINN paper, Yu et al., CMAME 2022)
    -u''(x) = f(x),   x in (0, pi),     u(0) = 0,  u(pi) = pi
    f(x) = sum_{i=1}^{4} i sin(i x) + 8 sin(8x)
    Exact: u(x) = x + sum_{i=1}^{4} sin(i x)/i + sin(8x)/8

PINN LOSS
    Loss = w_pde * mean_j [ -u_NN''(x_j) - f(x_j) ]^2               (PDE residual)
         + w_bc  * { [u_NN(0) - 0]^2 + [u_NN(pi) - pi]^2 } / 2        (boundary)

    The second derivative u_NN'' is obtained by applying autograd twice.
    Boundary conditions are imposed "softly" (as a penalty), the most common
    and simplest choice. (Script 3 shows the "hard" alternative.)

COMPARISON
    * PINN: N_COLLOCATION unlabeled points + the 2 boundary values.
    * Plain NN: N_DATA labelled samples of the true solution.

WHAT YOU SHOULD SEE
    The PINN reaches a small error without a single interior measurement,
    and its derivative u'(x) is accurate too, because the physics constrains
    the curvature everywhere. The plain NN with sparse data does not.

HOW TO RUN
    python 02_pinns/pinn_02_poisson_1d.py
    python 02_pinns/pinn_02_poisson_1d.py --n_col 15     # very few points

THINGS TO TRY
    * --n_col 15 : with only 15 collocation points the PINN gets worse.
      Stage 5 (gradient-enhanced PINN) revisits exactly this setting.
    * --w_bc 100 : put more weight on the boundary conditions.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import (plot_curves, plot_history, plot_loss_terms,  # noqa: E402
                             plot_pointwise_error)
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("n_col", int, 50, "number of collocation points of the PINN"),
    ("n_data", int, 15, "number of labelled points of the plain NN"),
    ("w_bc", float, 1.0, "weight of the boundary loss"),
])
X_MIN, X_MAX = 0.0, math.pi
LAYERS = [1, 50, 50, 50, 1]

setup(args.seed)
out = output_dir(__file__)

u_exact, du_exact, f = P.poisson1d_u, P.poisson1d_du, P.poisson1d_f
x_test = torch.linspace(X_MIN, X_MAX, 1000).reshape(-1, 1)

# -----------------------------------------------------------------------------
# 2. PINN
# -----------------------------------------------------------------------------
x_col = torch.linspace(X_MIN, X_MAX, args.n_col).reshape(-1, 1).requires_grad_(True)
x_bc = torch.tensor([[X_MIN], [X_MAX]])
u_bc = u_exact(x_bc)                  # = [0, pi]
f_col = f(x_col).detach()             # forcing at the collocation points (fixed)

pinn = MLP(LAYERS, activation="tanh")


def pinn_loss():
    u = pinn(x_col)
    u_x = grad(u, x_col)
    u_xx = grad(u_x, x_col)
    residual = -u_xx - f_col                       # PDE residual, should be 0
    loss_pde = torch.mean(residual ** 2)
    loss_bc = torch.mean((pinn(x_bc) - u_bc) ** 2)
    return {"total": loss_pde + args.w_bc * loss_bc, "pde": loss_pde, "bc": loss_bc}


def error_of(model):
    def fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))
    return fn


print(f"\n----- PINN with {args.n_col} collocation points -----")
hist_pinn = train(pinn.parameters(), pinn_loss, args.epochs, args.lr, error_fn=error_of(pinn))

# -----------------------------------------------------------------------------
# 3. Plain NN fitted to labelled data
# -----------------------------------------------------------------------------
x_data = torch.linspace(X_MIN, X_MAX, args.n_data).reshape(-1, 1)
u_data = u_exact(x_data)
nn_model = MLP(LAYERS, activation="tanh")


def nn_loss():
    return {"total": torch.mean((nn_model(x_data) - u_data) ** 2)}


print(f"\n----- Plain NN with {args.n_data} labelled points -----")
hist_nn = train(nn_model.parameters(), nn_loss, args.epochs, args.lr, error_fn=error_of(nn_model))

# -----------------------------------------------------------------------------
# 4. Evaluation and plots
# -----------------------------------------------------------------------------
def derivative(model, x):
    x = x.clone().requires_grad_(True)
    return grad(model(x), x).detach()


with torch.no_grad():
    u_pinn, u_nn = pinn(x_test), nn_model(x_test)
du_pinn, du_nn = derivative(pinn, x_test), derivative(nn_model, x_test)

plot_curves(x_test, u_exact(x_test),
            {"PINN": u_pinn, f"plain NN ({args.n_data} data)": u_nn},
            out / "solution.png", title="1-D Poisson: -u'' = f",
            points=(x_data, u_data), points_label="NN training data")
plot_curves(x_test, du_exact(x_test), {"PINN": du_pinn, "plain NN": du_nn},
            out / "derivative.png", title="Derivative u'(x)", ylabel="u'(x)")
plot_pointwise_error(x_test, {"PINN": u_pinn - u_exact(x_test),
                              "plain NN": u_nn - u_exact(x_test)},
                     out / "pointwise_error.png")
plot_history({"PINN": hist_pinn, "plain NN": hist_nn}, out / "error_history.png")
plot_loss_terms(hist_pinn, out / "pinn_loss_terms.png")

save_summary(out, {
    "PINN rel. L2 error of u": relative_l2(u_pinn, u_exact(x_test)),
    "PINN rel. L2 error of u'": relative_l2(du_pinn, du_exact(x_test)),
    "plain NN rel. L2 error of u": relative_l2(u_nn, u_exact(x_test)),
    "plain NN rel. L2 error of u'": relative_l2(du_nn, du_exact(x_test)),
})
