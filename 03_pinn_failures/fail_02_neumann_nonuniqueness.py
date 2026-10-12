r"""
==============================================================================
STAGE 3 - Script 2:  PINN failure #2 - a tiny loss does not mean the right answer
==============================================================================

PROBLEM  (pure Neumann boundary-value problem)
    -u''(x) = 2,   x in (0, 1),    u'(0) = 0,   u'(1) = -2

    Every function  u(x) = 1 - x^2 + C  solves this problem, for ANY constant C:
    only derivatives of u appear, so adding a constant changes nothing.
    The problem is ill-posed (no unique solution).
    (Compatibility check: integral of f = 2 = -(u'(1) - u'(0)) = 2, so solutions exist.)

    The "intended" solution is u = 1 - x^2, which needs ONE extra condition,
    e.g. the anchor u(0) = 1.

EXPERIMENT
    (a) PINN with only the two Neumann conditions, three different seeds.
    (b) PINN with the Neumann conditions + the anchor u(0) = 1.

WHAT YOU SHOULD SEE
    (a) Every run drives the loss to ~1e-7 - it "converges" - but each run
        ends at a different, arbitrary constant C (nothing in the loss prefers
        C = 0). The relative error vs. 1 - x^2 is huge although the loss is tiny.
    (b) With the anchor, the problem is well-posed and the PINN is accurate.

LESSON
    A PINN only minimises a loss; it does not check well-posedness. Before
    trusting a small loss, make sure the continuous problem has a unique
    solution (here: add an anchor, or a zero-mean condition).
    Stage 4 (vpinn_03) shows how the weak form treats Neumann conditions
    "naturally"; the uniqueness issue, however, is a property of the PDE
    and must be fixed in the same way there.

HOW TO RUN
    python 03_pinn_failures/fail_02_neumann_nonuniqueness.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=10000, lr=1e-3)
SEEDS = [1, 2, 3]
N_COLLOCATION = 50
LAYERS = [1, 20, 20, 20, 1]

setup(args.seed)
out = output_dir(__file__)
u_exact, f = P.neumann_u, P.neumann_f
x_test = torch.linspace(0, 1, 500).reshape(-1, 1)


def run(seed, use_anchor):
    setup(seed)
    x_col = torch.linspace(0, 1, N_COLLOCATION).reshape(-1, 1).requires_grad_(True)
    f_col = f(x_col).detach()
    # Neumann points and data: u'(0) = 0, u'(1) = -2
    x_n = torch.tensor([[0.0], [1.0]], requires_grad=True)
    g_n = torch.tensor([[0.0], [-2.0]])
    # Dirichlet anchor: u(0) = 1
    x_d, g_d = torch.tensor([[0.0]]), torch.tensor([[1.0]])

    model = MLP(LAYERS, activation="tanh")

    def loss_fn():
        u = model(x_col)
        u_xx = grad(grad(u, x_col), x_col)
        loss_pde = torch.mean((-u_xx - f_col) ** 2)
        du_n = grad(model(x_n), x_n)                    # u' at the two end points
        loss_neu = torch.mean((du_n - g_n) ** 2)
        terms = {"pde": loss_pde, "neumann": loss_neu}
        total = loss_pde + loss_neu
        if use_anchor:
            terms["anchor"] = torch.mean((model(x_d) - g_d) ** 2)
            total = total + terms["anchor"]
        terms["total"] = total
        return terms

    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, log_every=5000,
                 lbfgs_iters=args.lbfgs)
    with torch.no_grad():
        u_pred = model(x_test)
    return u_pred, hist


summary, preds = {}, {}
for seed in SEEDS:
    print(f"\n===== (a) Neumann only, seed {seed} =====")
    u_pred, hist = run(seed, use_anchor=False)
    C = float(torch.mean(u_pred - u_exact(x_test)))       # the constant the PINN picked
    preds[f"Neumann only, seed {seed} (C = {C:+.2f})"] = u_pred
    summary[f"(a) seed {seed}: final total loss"] = hist["total"][-1]
    summary[f"(a) seed {seed}: constant C picked"] = round(C, 4)
    summary[f"(a) seed {seed}: rel. L2 error vs 1 - x^2"] = relative_l2(u_pred, u_exact(x_test))

print("\n===== (b) Neumann + anchor u(0) = 1 =====")
u_pred, hist = run(SEEDS[0], use_anchor=True)
preds["Neumann + anchor u(0)=1"] = u_pred
summary["(b) with anchor: final total loss"] = hist["total"][-1]
summary["(b) with anchor: rel. L2 error"] = relative_l2(u_pred, u_exact(x_test))

plot_curves(x_test, u_exact(x_test), preds, out / "solutions.png",
            title="Tiny loss in every run - but which solution?")
save_summary(out, summary)
