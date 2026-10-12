r"""
==============================================================================
STAGE 4 - Script 3:  VPINN with Neumann conditions - "natural" boundary conditions
==============================================================================

PROBLEM  (the pure-Neumann problem of 03_pinn_failures/fail_02, plus an anchor)
    -u''(x) = 2  on (0, 1),    u'(0) = g_L = 0,   u'(1) = g_R = -2,   u(0) = 1
    Exact: u(x) = 1 - x^2

WHY THE WEAK FORM IS ELEGANT FOR NEUMANN DATA
    Multiply -u'' = f by a test function v and integrate by parts:

        \int_0^1 u' v' dx  -  [u' v]_0^1  =  \int_0^1 f v dx
    =>  \int_0^1 u' v' dx  =  \int_0^1 f v dx  +  g_R v(1)  -  g_L v(0)

    The prescribed derivatives g_L, g_R appear directly in the equation.
    Neumann conditions are therefore called NATURAL boundary conditions in
    the finite-element method: no extra penalty term is needed. But the test
    functions must NOT vanish at the boundary, otherwise the boundary term
    (and with it the Neumann data) disappears. So we add two linear "hat"
    functions to the Legendre bubbles v_k:
        phi_L(xi) = (1 - xi)/2   (= 1 at x = 0, 0 at x = 1)
        phi_R(xi) = (1 + xi)/2   (= 0 at x = 0, 1 at x = 1)

THREE RUNS
    (a) "penalty":  bubbles only (vanish at the ends) + penalty terms
                    (u'(0) - g_L)^2 + (u'(1) - g_R)^2 + anchor (u(0) - 1)^2
    (b) "natural":  bubbles + hats, Neumann data inside the weak form,
                    only the anchor is penalised
    (c) "natural, no anchor": like (b) without u(0) = 1. The PDE still does
                    not determine the constant -> expect an arbitrary shift,
                    exactly as for the PINN in fail_02. The weak form fixes
                    HOW the Neumann data enter, not the missing uniqueness.

WHAT YOU SHOULD SEE
    (a) and (b) are both accurate; (b) needs one loss weight fewer and
    never differentiates the network at the boundary. (c) shows a constant offset.

HOW TO RUN
    python 04_vpinns/vpinn_03_neumann_natural_bc.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves  # noqa: E402
from common.quadrature import gauss_lobatto_legendre, test_functions  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=10000, lr=1e-3, extra=[
    ("n_test", int, 20, "number of Legendre bubble test functions"),
    ("n_quad", int, 40, "number of quadrature points"),
])
G_L, G_R = 0.0, -2.0           # Neumann data u'(0), u'(1)
ANCHOR_X, ANCHOR_U = 0.0, 1.0  # Dirichlet anchor u(0) = 1
LAYERS = [1, 20, 20, 20, 1]
A, B = 0.0, 1.0                # one element = the whole interval
J = (B - A) / 2

setup(args.seed)
out = output_dir(__file__)
u_exact, f = P.neumann_u, P.neumann_f
x_test = torch.linspace(A, B, 500).reshape(-1, 1)

# -----------------------------------------------------------------------------
# 2. Quadrature and test functions
# -----------------------------------------------------------------------------
xi, w = gauss_lobatto_legendre(args.n_quad)
V, dV, _ = test_functions(args.n_test, xi)
xi, w, V, dV = (torch.tensor(a) for a in (xi, w, V, dV))

# bubbles + two hat functions (rows = test functions, columns = quadrature points)
V_hat = torch.stack([(1 - xi) / 2, (1 + xi) / 2])           # phi_L, phi_R
dV_hat = torch.stack([-0.5 * torch.ones_like(xi), 0.5 * torch.ones_like(xi)])
V_all, dV_all = torch.cat([V, V_hat]), torch.cat([dV, dV_hat])
# values of every test function at x = 0 (xi = -1) and x = 1 (xi = +1)
v_at_0 = torch.cat([torch.zeros(args.n_test), torch.tensor([1.0, 0.0])])
v_at_1 = torch.cat([torch.zeros(args.n_test), torch.tensor([0.0, 1.0])])

x_quad = (A + J * (xi + 1)).reshape(-1, 1).requires_grad_(True)
f_quad = f(x_quad).detach().ravel()


def weak_residual(model, test_V, test_dV, rhs_boundary):
    r"""R_k = \int u' v_k' dx - \int f v_k dx - (boundary term)."""
    u_x = grad(model(x_quad), x_quad).ravel()
    lhs = test_dV @ (w * u_x)                 # sum_q w_q u'(x_q) dv_k/dxi   (J cancels)
    rhs = J * (test_V @ (w * f_quad))         # J sum_q w_q f(x_q) v_k(xi_q)
    return lhs - rhs - rhs_boundary


x_anchor = torch.tensor([[ANCHOR_X]])
x_ends = torch.tensor([[A], [B]], requires_grad=True)
g_ends = torch.tensor([[G_L], [G_R]])


def run(mode):
    setup(args.seed)
    model = MLP(LAYERS, activation="tanh")

    def loss_fn():
        terms = {}
        if mode == "penalty":
            R = weak_residual(model, V, dV, 0.0)               # bubbles only
            du_ends = grad(model(x_ends), x_ends)              # u'(0), u'(1)
            terms["neumann penalty"] = torch.mean((du_ends - g_ends) ** 2)
        else:
            boundary = G_R * v_at_1 - G_L * v_at_0             # natural BC term
            R = weak_residual(model, V_all, dV_all, boundary)
        terms["variational"] = torch.mean(R ** 2)
        if mode != "natural, no anchor":
            terms["anchor"] = torch.mean((model(x_anchor) - ANCHOR_U) ** 2)
        terms["total"] = sum(terms.values())
        return terms

    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, log_every=5000,
                 lbfgs_iters=args.lbfgs)
    with torch.no_grad():
        return model(x_test), hist


summary, preds = {}, {}
for mode in ["penalty", "natural", "natural, no anchor"]:
    print(f"\n===== VPINN, Neumann treatment: {mode} =====")
    u_pred, hist = run(mode)
    preds[f"VPINN ({mode})"] = u_pred
    summary[f"({mode}) rel. L2 error"] = relative_l2(u_pred, u_exact(x_test))
    summary[f"({mode}) final loss"] = hist["total"][-1]
    if mode == "natural, no anchor":
        summary[f"({mode}) constant offset C"] = round(float(torch.mean(u_pred - u_exact(x_test))), 4)

plot_curves(x_test, u_exact(x_test), preds, out / "solutions.png",
            title="Neumann conditions: penalty vs. natural (weak form)")
save_summary(out, summary)
