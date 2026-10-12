r"""
==============================================================================
STAGE 3 - Script 1:  PINN failure #1 - multi-scale solutions (unreliable training)
==============================================================================

PROBLEM  (multi-scale benchmark of the VPINN papers, Kharazmi et al. 2019/2021)
    -u''(x) = f(x),   x in (-1, 1),   Dirichlet values at x = -1 and x = 1
    Exact:  u(x) = 0.1 sin(8 pi x) + tanh(5 pi x)
              = small, fast wave   +  steep layer at x = 0

EXPERIMENT
    (a) "single-scale" version: only the layer, u = tanh(5 pi x)
    (b) "multi-scale" version: layer + fast wave, trained with 4 different
        random seeds. A seed fixes the initial weights AND where the random
        collocation points are placed - nothing else changes.

WHAT YOU SHOULD SEE
    * (a) The PINN solves the single-scale problem very well (error < 0.1 %).
    * (b) For the multi-scale problem the result depends wildly on the seed:
      the best run reaches a few % error, others are completely wrong (> 50 %).
    * Look at "final PDE loss": the worst run can have a SMALLER residual
      loss than a much better run. The loss is only measured AT the
      collocation points; between them the network is free to do anything.

WHY
    * Spectral bias (01_neural_networks/nn_02): the fast wave is learned last.
    * The forcing f = -u'' contains 0.1 (8 pi)^2 ~ 63 sin(8 pi x) PLUS a huge
      spike ~ (5 pi)^2 inside the thin layer. With ~100 random points, a few
      points dominate the loss and the optimiser can settle in a wrong solution
      that happens to fit those points.
    A practitioner cannot tell which run is good without the exact solution!
    Stage 4 (VPINNs) tests the residual against smooth test functions,
    integrated accurately over the domain, and is far more reliable here.

HOW TO RUN
    python 03_pinn_failures/fail_01_multiscale.py
    python 03_pinn_failures/fail_01_multiscale.py --n_col 1000     # many more points

THINGS TO TRY
    * --n_col 1000 : with 10x more collocation points the PINN usually becomes
      accurate again, but each iteration costs more. Brute force works in 1-D,
      it becomes very expensive in 2-D/3-D.
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
    ("n_col", int, 100, "number of random collocation points"),
    ("w_bc", float, 10.0, "weight of the boundary loss"),
    ("seeds", str, "1,2,3,4", "comma-separated list of random seeds"),
])
SEEDS = [int(s) for s in args.seeds.split(",")]          # the same seeds are used in 04_vpinns/vpinn_01
R = 5 * math.pi               # steepness of the layer tanh(R x)
OMEGA = 8 * math.pi           # frequency of the small wave
LAYERS = [1, 20, 20, 20, 20, 1]
X_MIN, X_MAX = -1.0, 1.0

setup(args.seed)              # float64 (the seed is re-set before every run)
out = output_dir(__file__)
x_test = torch.linspace(X_MIN, X_MAX, 2001).reshape(-1, 1)


def run_pinn(seed, omega):
    """Train one PINN. omega = 0 gives the single-scale problem."""
    setup(seed)
    u_exact = lambda x: P.sharp_u(x, r=R, omega=omega)      # noqa: E731
    f = lambda x: P.sharp_f(x, r=R, omega=omega)            # noqa: E731

    # random collocation points in (-1, 1); NO solution values are used here
    x_col = (X_MIN + (X_MAX - X_MIN) * torch.rand(args.n_col, 1)).requires_grad_(True)
    f_col = f(x_col).detach()
    x_bc = torch.tensor([[X_MIN], [X_MAX]])
    u_bc = u_exact(x_bc)

    model = MLP(LAYERS, activation="tanh")

    def loss_fn():
        u = model(x_col)
        u_xx = grad(grad(u, x_col), x_col)            # second derivative via autograd
        loss_pde = torch.mean((-u_xx - f_col) ** 2)   # residual of -u'' = f
        loss_bc = torch.mean((model(x_bc) - u_bc) ** 2)
        return {"total": loss_pde + args.w_bc * loss_bc, "pde": loss_pde, "bc": loss_bc}

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))

    history = train(model.parameters(), loss_fn, args.epochs, args.lr,
                    error_fn=error_fn, lbfgs_iters=args.lbfgs, log_every=5000)
    with torch.no_grad():
        u_pred = model(x_test)
    return u_pred, u_exact(x_test), history


summary = {}

# -----------------------------------------------------------------------------
# 2. (a) Single-scale problem: u = tanh(5 pi x)
# -----------------------------------------------------------------------------
print("\n===== (a) single-scale problem  u = tanh(5 pi x) =====")
u_pred, u_ex, hist = run_pinn(SEEDS[0], omega=0.0)
summary["(a) single-scale: rel. L2 error"] = relative_l2(u_pred, u_ex)
plot_curves(x_test, u_ex, {"PINN": u_pred}, out / "single_scale_solution.png",
            title="PINN, single-scale u = tanh(5 pi x)")

# -----------------------------------------------------------------------------
# 3. (b) Multi-scale problem, several seeds
# -----------------------------------------------------------------------------
preds, errors, histories = {}, {}, {}
for seed in SEEDS:
    print(f"\n===== (b) multi-scale problem, seed {seed} =====")
    u_pred, u_ex, hist = run_pinn(seed, omega=OMEGA)
    label = f"seed {seed}"
    preds[label], errors[label], histories[label] = u_pred, u_pred - u_ex, hist
    summary[f"(b) multi-scale, {label}: rel. L2 error"] = relative_l2(u_pred, u_ex)
    summary[f"(b) multi-scale, {label}: final PDE loss"] = hist["pde"][-1]

plot_curves(x_test, u_ex, preds, out / "multiscale_solutions.png",
            title="PINN on the multi-scale problem: same setup, different seeds")
plot_pointwise_error(x_test, errors, out / "multiscale_pointwise_error.png")
plot_history(histories, out / "multiscale_error_history.png",
             title="PINN error during training (multi-scale)")
plot_history(histories, out / "multiscale_pde_loss_history.png", key="pde",
             ylabel="PDE residual loss", title="PDE loss: similar for good and bad runs")

errs = [summary[f"(b) multi-scale, seed {s}: rel. L2 error"] for s in SEEDS]
summary["(b) best / worst rel. L2 error over seeds"] = f"{min(errs):.2e} / {max(errs):.2e}"
save_summary(out, summary)
