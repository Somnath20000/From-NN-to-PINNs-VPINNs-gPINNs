r"""
==============================================================================
STAGE 4 - Script 2:  hp-VPINN in 2-D (domain decomposition) vs. PINN
==============================================================================

PROBLEM  (hp-VPINN paper, Kharazmi, Zhang & Karniadakis, CMAME 2021)
    -(u_xx + u_yy) = f   in (-1, 1)^2,    u = exact solution on the boundary
    Exact: u(x, y) = (0.1 sin(2 pi x) + tanh(10 x)) sin(2 pi y)
    (a steep layer along the line x = 0)

hp-VPINN
    "h": the square is split into Nx x Ny elements (sub-squares).
    "p": in every element we use K x K test functions
             v_rs(x, y) = v_r(xi) v_s(eta),  v_r = P_{r+1} - P_{r-1}
         where (xi, eta) are the local coordinates of the element in [-1, 1]^2.
    One network covers the whole domain; only the TESTING is local.

    Weak form (integrate by parts once; v vanishes on the element boundary):
        R_e,rs = \int_e grad(u_NN) . grad(v_rs) dx dy  -  \int_e f v_rs dx dy
    Each integral uses a Q x Q tensor-product Gauss-Lobatto rule.

    Loss = sum_e mean_rs R_e,rs^2  +  w_bc * mean (u_NN - g)^2 on boundary points

BASELINE PINN
    Same network, same boundary points/weight, same optimiser, with the
    strong residual  -(u_xx + u_yy) - f  at N_RES random interior points.

WHAT YOU SHOULD SEE  (our numbers; see the README)
    * With the settings of the original hp-VPINN code (100 PINN residual
      points, 4 x 4 elements for the VPINN): hp-VPINN ~2 % error, PINN ~60 %
      although its residual loss is small.
    * BUT with --method pinn --n_res 1600 the PINN reaches ~1 % - better than
      the hp-VPINN - at a similar wall-clock time. Lesson: compare methods at
      matched COST, not only at matched settings. For this smooth 2-D problem
      brute-force collocation is a strong baseline.
    Fairness notes: the VPINN evaluates the network at 16 x 20 x 20 = 6400
    quadrature points but needs only FIRST derivatives; the PINN needs second
    derivatives. Runtime: about 5 minutes for both methods on a laptop CPU.

HOW TO RUN
    python 04_vpinns/vpinn_02_hp_poisson_2d.py                  # both methods
    python 04_vpinns/vpinn_02_hp_poisson_2d.py --method vpinn
    python 04_vpinns/vpinn_02_hp_poisson_2d.py --elements 2 --n_test 15

THINGS TO TRY
    * --elements 1 (a single element, plain VPINN) vs --elements 4 (hp-VPINN).
    * --n_res 2000 for the PINN: how many points does it need to match?
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
from common.quadrature import gauss_lobatto_legendre, test_functions  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("method", str, "both", "vpinn, pinn or both"),
    ("elements", int, 4, "number of elements per direction (Nx = Ny)"),
    ("n_test", int, 10, "test functions per direction per element (K)"),
    ("n_quad", int, 20, "quadrature points per direction per element (Q)"),
    ("n_res", int, 100, "random residual points for the PINN baseline"),
    ("n_bound", int, 80, "random boundary points per edge"),
    ("w_bc", float, 50.0, "weight of the boundary loss"),
])
LAYERS = [2, 20, 20, 20, 20, 1]
LO, HI = -1.0, 1.0

setup(args.seed)
out = output_dir(__file__)
u_exact, f = P.layer2d_u, P.layer2d_f

# -----------------------------------------------------------------------------
# 2. Boundary points (shared by both methods): N_BOUND random points per edge
# -----------------------------------------------------------------------------
t = LO + (HI - LO) * torch.rand(4, args.n_bound, 1)
ones = torch.ones(args.n_bound, 1)
xy_bc = torch.cat([
    torch.cat([t[0], HI * ones], 1),    # top edge    y = 1
    torch.cat([t[1], LO * ones], 1),    # bottom edge y = -1
    torch.cat([HI * ones, t[2]], 1),    # right edge  x = 1
    torch.cat([LO * ones, t[3]], 1),    # left edge   x = -1
])
u_bc = u_exact(xy_bc)

# Test grid for errors and plots
n = 201
X, Y = np.meshgrid(np.linspace(LO, HI, n), np.linspace(LO, HI, n))
xy_test = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
u_test = u_exact(xy_test)

# -----------------------------------------------------------------------------
# 3. hp-VPINN pre-computation (independent of the network)
# -----------------------------------------------------------------------------
Ne, K, Q = args.elements, args.n_test, args.n_quad
xi, w = gauss_lobatto_legendre(Q)
V, dV, _ = test_functions(K, xi)                       # (K, Q) each
xi, w, V, dV = (torch.tensor(a) for a in (xi, w, V, dV))

edges = torch.linspace(LO, HI, Ne + 1)
# element (ex, ey) -> index e = ex * Ne + ey ; lower-left corner and half-widths
ax = edges[:-1].repeat_interleave(Ne)                  # x of lower-left corner, (E,)
ay = edges[:-1].repeat(Ne)                             # y of lower-left corner, (E,)
Jx = Jy = (HI - LO) / Ne / 2                           # all elements are equal squares
E = Ne * Ne

# Quadrature points of every element, shape (E, Q, Q): index [e, i, j] with
# i = point along x, j = point along y
xq = ax[:, None, None] + Jx * (xi[None, :, None] + 1) + 0 * xi[None, None, :]
yq = ay[:, None, None] + Jy * (xi[None, None, :] + 1) + 0 * xi[None, :, None]
xy_quad = torch.stack([xq.reshape(-1), yq.reshape(-1)], dim=1).requires_grad_(True)

# Right-hand side F[e, r, s] = Jx Jy sum_ij w_i w_j f(x_eij) v_r(xi_i) v_s(eta_j)
f_quad = f(xy_quad).detach().reshape(E, Q, Q)
Vw, dVw = V * w, dV * w                                # fold the weights in
F = Jx * Jy * torch.einsum("ri,sj,eij->ers", Vw, Vw, f_quad)


def vpinn_loss_fn(model):
    def loss_fn():
        u = model(xy_quad)
        g = grad(u, xy_quad)                           # first derivatives only
        ux, uy = g[:, 0].reshape(E, Q, Q), g[:, 1].reshape(E, Q, Q)
        # \int u_x v_x = Jy sum w w u_x dv_r v_s ;  \int u_y v_y = Jx sum w w u_y v_r dv_s
        R = (Jy * torch.einsum("ri,sj,eij->ers", dVw, Vw, ux)
             + Jx * torch.einsum("ri,sj,eij->ers", Vw, dVw, uy) - F)
        loss_var = torch.sum(torch.mean(R.reshape(E, -1) ** 2, dim=1))
        loss_bc = torch.mean((model(xy_bc) - u_bc) ** 2)
        return {"total": loss_var + args.w_bc * loss_bc, "variational": loss_var, "bc": loss_bc}
    return loss_fn


# -----------------------------------------------------------------------------
# 4. PINN baseline
# -----------------------------------------------------------------------------
xy_res = (LO + (HI - LO) * torch.rand(args.n_res, 2)).requires_grad_(True)
f_res = f(xy_res).detach()


def pinn_loss_fn(model):
    def loss_fn():
        u = model(xy_res)
        g = grad(u, xy_res)
        u_xx = grad(g[:, 0:1], xy_res)[:, 0:1]
        u_yy = grad(g[:, 1:2], xy_res)[:, 1:2]
        loss_pde = torch.mean((-(u_xx + u_yy) - f_res) ** 2)
        loss_bc = torch.mean((model(xy_bc) - u_bc) ** 2)
        return {"total": loss_pde + args.w_bc * loss_bc, "pde": loss_pde, "bc": loss_bc}
    return loss_fn


# -----------------------------------------------------------------------------
# 5. Train the requested method(s)
# -----------------------------------------------------------------------------
methods = ["vpinn", "pinn"] if args.method == "both" else [args.method]
labels = {"vpinn": f"hp-VPINN ({Ne}x{Ne} elements, K={K})", "pinn": f"PINN ({args.n_res} points)"}
summary, histories, fields = {}, {}, {"exact u": u_test.cpu().numpy().reshape(n, n)}
for method in methods:
    setup(args.seed)                                   # identical initial weights
    model = MLP(LAYERS, activation="tanh")
    loss_fn = vpinn_loss_fn(model) if method == "vpinn" else pinn_loss_fn(model)

    def error_fn(model=model):
        with torch.no_grad():
            return relative_l2(model(xy_test), u_test)

    print(f"\n===== {labels[method]} =====")
    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                 log_every=2000, lbfgs_iters=args.lbfgs)
    with torch.no_grad():
        u_pred = model(xy_test).cpu().numpy().reshape(n, n)
    histories[labels[method]] = hist
    fields[f"{method.upper()} u"] = u_pred
    fields[f"{method.upper()} |error|"] = np.abs(u_pred - fields["exact u"])
    summary[f"{labels[method]}: rel. L2 error"] = relative_l2(u_pred, fields["exact u"])

plot_fields_2d(X, Y, fields, out / "fields.png")
plot_history(histories, out / "error_history.png", title="Relative L2 error during training")
save_summary(out, summary)
