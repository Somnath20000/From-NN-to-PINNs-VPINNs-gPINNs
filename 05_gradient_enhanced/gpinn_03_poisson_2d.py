r"""
==============================================================================
STAGE 5 - Script 3:  gPINN in 2-D - revisiting the sharp peak of fail_03
==============================================================================

PROBLEM  (gPINN paper; same problem as 03_pinn_failures/fail_03, but a
          different random draw of the 400 collocation points)
    -(u_xx + u_yy) = f in (0,1)^2,  u = 0 on the boundary (hard constraint)
    Exact: u = [16 x (1-x) y (1-y)]^10,   400 random collocation points

gPINN LOSS IN 2-D
    r = u_xx + u_yy + f                         (PDE residual)
    r_x = u_xxx + u_yyx + f_x,   r_y = u_xxy + u_yyy + f_y
    Loss = mean r^2 + w_g * ( mean r_x^2 + mean r_y^2 )

    Third derivatives of the network are computed by applying autograd three
    times; f_x and f_y come from autograd on the torch expression of f.

    Note on the weight: for this problem the derivatives of f are about 10x
    larger than f itself (root-mean-square 258 vs 23), so the gradient terms
    are about 100x larger than the residual term. A weight around
    w_g = 1e-2 keeps the two contributions comparable; try other values.

WHAT YOU SHOULD SEE
    Compare the final PINN and gPINN errors in the summary (our numbers are in
    the README). The extra term makes the residual flatter between the
    collocation points, at the price of ~2-3x more work per iteration
    (third derivatives).

HOW TO RUN
    python 05_gradient_enhanced/gpinn_03_poisson_2d.py            # ~10 min on a laptop CPU
    python 05_gradient_enhanced/gpinn_03_poisson_2d.py --w_g 1e-3
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_fields_2d, plot_history  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("a", int, 10, "sharpness exponent of the exact solution"),
    ("n_col", int, 400, "number of random collocation points"),
    ("w_g", float, 1e-2, "weight of the gradient-of-residual terms"),
])
LAYERS = [2, 20, 20, 20, 1]

setup(args.seed)
out = output_dir(__file__)
u_exact = lambda xy: P.peak2d_u(xy, a=args.a)   # noqa: E731
f = lambda xy: P.peak2d_f(xy, a=args.a)         # noqa: E731


def hard_bc(xy, raw):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return x * (1 - x) * y * (1 - y) * raw


xy_col = torch.rand(args.n_col, 2).requires_grad_(True)
f_col = f(xy_col)
f_grad = grad(f_col, xy_col).detach()            # [f_x, f_y] by autograd
f_col = f_col.detach()

n = 101
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
xy_test = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
U = u_exact(xy_test).cpu().numpy().reshape(n, n)


def run(w_g):
    setup(args.seed)                              # identical initial weights
    model = MLP(LAYERS, activation="tanh", output_transform=hard_bc)

    def loss_fn():
        u = model(xy_col)
        g = grad(u, xy_col)
        u_xx = grad(g[:, 0:1], xy_col)[:, 0:1]
        u_yy = grad(g[:, 1:2], xy_col)[:, 1:2]
        r = u_xx + u_yy + f_col
        terms = {"pde": torch.mean(r ** 2)}
        total = terms["pde"]
        if w_g > 0:
            lap_grad = grad(u_xx + u_yy, xy_col)          # [d/dx, d/dy] of the Laplacian
            r_x = lap_grad[:, 0:1] + f_grad[:, 0:1]
            r_y = lap_grad[:, 1:2] + f_grad[:, 1:2]
            terms["grad-pde"] = torch.mean(r_x ** 2) + torch.mean(r_y ** 2)
            total = total + w_g * terms["grad-pde"]
        terms["total"] = total
        return terms

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(xy_test).cpu().numpy().reshape(n, n), U)

    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                 log_every=2000, record_every=500, lbfgs_iters=args.lbfgs)
    with torch.no_grad():
        return model(xy_test).cpu().numpy().reshape(n, n), hist


print("\n===== PINN =====")
u_pinn, hist_pinn = run(0.0)
print(f"\n===== gPINN (w_g = {args.w_g}) =====")
u_gpinn, hist_gpinn = run(args.w_g)

plot_fields_2d(X, Y, {"exact u": U, "PINN |error|": np.abs(u_pinn - U),
                      "gPINN |error|": np.abs(u_gpinn - U)}, out / "fields.png")
plot_history({"PINN": hist_pinn, "gPINN": hist_gpinn}, out / "error_history.png")
save_summary(out, {
    "PINN rel. L2 error": relative_l2(u_pinn, U),
    f"gPINN (w_g={args.w_g}) rel. L2 error": relative_l2(u_gpinn, U),
})
