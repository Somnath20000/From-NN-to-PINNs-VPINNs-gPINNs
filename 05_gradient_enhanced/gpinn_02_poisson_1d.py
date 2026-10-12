r"""
==============================================================================
STAGE 5 - Script 2:  Gradient-enhanced PINN (gPINN) for the 1-D Poisson problem
==============================================================================

THE IDEA  (Yu, Lu, Meng & Karniadakis, CMAME 393, 114823, 2022)
    If the PDE residual  r(x) = -u''(x) - f(x)  is zero EVERYWHERE, then its
    derivative is zero too:  r'(x) = -u'''(x) - f'(x) = 0.
    A gPINN adds this "free" information to the loss:

        Loss = mean r(x_j)^2  +  w_g * mean r'(x_j)^2  +  boundary loss

    No new data are needed - only the derivative of the equation. Intuitively,
    each collocation point now also constrains the SLOPE of the residual,
    so the residual cannot easily "bounce" between points (compare
    03_pinn_failures/fail_03 where it did).

    f'(x) is obtained here with autograd from the torch expression of f, so
    you never have to differentiate f by hand.

PROBLEM  (same as 02_pinns/pinn_02, gPINN paper setting)
    -u'' = f on (0, pi),  u(0) = 0, u(pi) = pi,  with only N_COL = 15 points.

WHAT YOU SHOULD SEE
    With so few collocation points the PINN is noticeably less accurate than
    with 50 points (02_pinns/pinn_02); the gPINN reduces the error of u and
    especially of u'. The weight w_g matters: try --w_g 1 or --w_g 1e-4.

HOW TO RUN
    python 05_gradient_enhanced/gpinn_02_poisson_1d.py
    python 05_gradient_enhanced/gpinn_02_poisson_1d.py --w_g 0.1 --n_col 30
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves, plot_history, plot_pointwise_error  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("n_col", int, 15, "number of collocation points"),
    ("w_g", float, 0.01, "weight of the gradient-of-residual loss"),
])
X_MIN, X_MAX = 0.0, math.pi
LAYERS = [1, 50, 50, 50, 1]

setup(args.seed)
out = output_dir(__file__)
u_exact, du_exact, f = P.poisson1d_u, P.poisson1d_du, P.poisson1d_f

x_col = torch.linspace(X_MIN, X_MAX, args.n_col).reshape(-1, 1).requires_grad_(True)
f_col = f(x_col)
f_x_col = grad(f_col, x_col).detach()        # f'(x) by autograd (no hand derivation)
f_col = f_col.detach()
x_bc = torch.tensor([[X_MIN], [X_MAX]])
u_bc = u_exact(x_bc)
x_test = torch.linspace(X_MIN, X_MAX, 1000).reshape(-1, 1)


def run(w_g):
    setup(args.seed)                          # same initial network for both methods
    model = MLP(LAYERS, activation="tanh")

    def loss_fn():
        u = model(x_col)
        u_x = grad(u, x_col)
        u_xx = grad(u_x, x_col)
        r = -u_xx - f_col                     # PDE residual
        terms = {"pde": torch.mean(r ** 2),
                 "bc": torch.mean((model(x_bc) - u_bc) ** 2)}
        total = terms["pde"] + terms["bc"]
        if w_g > 0:
            u_xxx = grad(u_xx, x_col)
            r_x = -u_xxx - f_x_col            # derivative of the residual
            terms["grad-pde"] = torch.mean(r_x ** 2)
            total = total + w_g * terms["grad-pde"]
        terms["total"] = total
        return terms

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))

    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                 log_every=5000, lbfgs_iters=args.lbfgs)
    return model, hist


def derivative(model, x):
    x = x.clone().requires_grad_(True)
    return grad(model(x), x).detach()


print(f"\n===== PINN ({args.n_col} points) =====")
pinn, hist_pinn = run(0.0)
print(f"\n===== gPINN ({args.n_col} points, w_g = {args.w_g}) =====")
gpinn, hist_gpinn = run(args.w_g)

with torch.no_grad():
    u_p, u_g = pinn(x_test), gpinn(x_test)
du_p, du_g = derivative(pinn, x_test), derivative(gpinn, x_test)
u_ex, du_ex = u_exact(x_test), du_exact(x_test)

plot_curves(x_test, u_ex, {"PINN": u_p, "gPINN": u_g}, out / "solution.png",
            title=f"PINN vs gPINN, {args.n_col} collocation points",
            points=(x_col.detach(), u_exact(x_col.detach())), points_label="collocation points")
plot_curves(x_test, du_ex, {"PINN": du_p, "gPINN": du_g}, out / "derivative.png",
            title="Derivative u'(x)", ylabel="u'(x)")
plot_pointwise_error(x_test, {"PINN": u_p - u_ex, "gPINN": u_g - u_ex}, out / "pointwise_error.png")
plot_history({"PINN": hist_pinn, "gPINN": hist_gpinn}, out / "error_history.png")

save_summary(out, {
    "PINN rel. L2 error of u": relative_l2(u_p, u_ex),
    "gPINN rel. L2 error of u": relative_l2(u_g, u_ex),
    "PINN rel. L2 error of u'": relative_l2(du_p, du_ex),
    "gPINN rel. L2 error of u'": relative_l2(du_g, du_ex),
})
