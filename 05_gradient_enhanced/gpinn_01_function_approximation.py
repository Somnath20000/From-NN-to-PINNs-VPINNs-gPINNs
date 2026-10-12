r"""
==============================================================================
STAGE 5 - Script 1:  Gradient-enhanced NN (gNN) - learning from derivatives too
==============================================================================

A DIFFERENT PERSPECTIVE
    Stages 2-4 changed WHERE/HOW the equation is enforced. Gradient
    enhancement instead adds MORE INFORMATION to the loss: if the derivative
    u'(x) is known (or, in Stage 5 script 2, if the derivative of the PDE
    residual must vanish), we also penalise its mismatch:

        Loss = mean (u_NN - u)^2  +  w * mean (u_NN' - u')^2

    This is also called Sobolev training (Czarnecki et al., NeurIPS 2017):
    the loss is now an H^1 norm instead of an L^2 norm. Each sample carries
    two pieces of information (value and slope), so sparse data go further.

PROBLEM  (function-approximation examples of the gPINN paper, Yu et al. 2022)
    --target oscillatory (default):  u = -(1.4 - 3x) sin(18x), 12 equispaced points
    --target damped_sine:           u = exp(-x) sin(30x),     100 random points

RUNS
    w = 0 (plain NN), 0.01, 0.1, 1   and two "learnable weight" variants:
    * naive learnable:  Loss = L_u + exp(s) L_g  minimised over the network AND s.
      Problem: d(Loss)/ds = exp(s) L_g > 0 ALWAYS, so gradient descent can
      only push s down. The minimum over s is at exp(s) = 0, i.e. the
      optimiser lowers the loss by switching the gradient term OFF. The
      weight drifts monotonically towards 0 (slowly - how far it gets
      depends on the learning rate and the number of iterations).
    * uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018):
          Loss = 0.5 exp(-s_u) L_u + 0.5 exp(-s_g) L_g + 0.5 (s_u + s_g)
      The + s terms prevent the collapse; the effective weight
      w_eff = exp(-s_g)/exp(-s_u) settles near L_u / L_g, which balances the two terms.

WHAT YOU SHOULD SEE
    * With derivative information the error of BOTH u and u' drops, often by
      an order of magnitude, for the same number of samples.
    * The best fixed weight is problem dependent (try --target damped_sine):
      w trades accuracy of u against accuracy of u'.
    * The naive learnable weight only decreases during training (it started
      at 1); it still helps here simply because it has not reached 0 yet.
      The uncertainty-weighted version is the most accurate in our run.

HOW TO RUN
    python 05_gradient_enhanced/gpinn_01_function_approximation.py
    python 05_gradient_enhanced/gpinn_01_function_approximation.py --target damped_sine
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves, plot_history  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=10000, lr=1e-3, extra=[
    ("target", str, "oscillatory", "oscillatory or damped_sine"),
])
WEIGHTS = [0.0, 0.01, 0.1, 1.0]

setup(args.seed)
out = output_dir(__file__)
if args.target == "oscillatory":
    u_exact, du_exact = P.oscillatory_u, P.oscillatory_du
    x_train = torch.linspace(0, 1, 12).reshape(-1, 1)     # 10 interior + 2 end points
    LAYERS = [1, 20, 20, 20, 1]
else:
    u_exact, du_exact = P.damped_sine_u, P.damped_sine_du
    x_train = torch.rand(100, 1)                          # random samples
    LAYERS = [1, 50, 50, 50, 1]

x_train.requires_grad_(True)
u_train = u_exact(x_train).detach()      # measured values
du_train = du_exact(x_train).detach()    # measured slopes (the extra information)
x_test = torch.linspace(0, 1, 1000).reshape(-1, 1)


def derivative(model, x):
    x = x.clone().requires_grad_(True)
    return grad(model(x), x).detach()


def run(mode, weight=0.0):
    """mode: 'fixed' (given weight), 'naive' or 'uncertainty' (learned weights)."""
    setup(args.seed)
    model = MLP(LAYERS, activation="tanh")
    params = list(model.parameters())
    s = torch.zeros(2, requires_grad=True)          # log-weights for learnable modes
    if mode != "fixed":
        params.append(s)

    def loss_fn():
        u = model(x_train)
        du = grad(u, x_train)
        L_u = torch.mean((u - u_train) ** 2)           # value mismatch
        L_g = torch.mean((du - du_train) ** 2)         # slope mismatch
        if mode == "fixed":
            total = L_u + weight * L_g
        elif mode == "naive":
            total = L_u + torch.exp(s[1]) * L_g        # collapses: s[1] -> -inf
        else:
            total = (0.5 * torch.exp(-s[0]) * L_u + 0.5 * torch.exp(-s[1]) * L_g
                     + 0.5 * (s[0] + s[1]))
        return {"total": total, "value": L_u, "gradient": L_g}

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))

    hist = train(params, loss_fn, args.epochs, args.lr, error_fn=error_fn, log_every=5000,
                 lbfgs_iters=args.lbfgs)
    if mode == "naive":
        learned = math.exp(float(s[1]))
    elif mode == "uncertainty":
        learned = math.exp(float(s[0]) - float(s[1]))   # effective weight ratio
    else:
        learned = weight
    return model, hist, learned


summary, preds, dpreds, histories = {}, {}, {}, {}
runs = [("fixed", w) for w in WEIGHTS] + [("naive", None), ("uncertainty", None)]
for mode, weight in runs:
    label = ("plain NN (w = 0)" if weight == 0 else f"gNN, w = {weight}") if mode == "fixed" \
        else f"gNN, learnable ({mode})"
    print(f"\n===== {label} =====")
    model, hist, learned = run(mode, weight or 0.0)
    with torch.no_grad():
        preds[label] = model(x_test)
    dpreds[label] = derivative(model, x_test)
    histories[label] = hist
    summary[f"{label}: rel. L2 error of u"] = relative_l2(preds[label], u_exact(x_test))
    summary[f"{label}: rel. L2 error of u'"] = relative_l2(dpreds[label], du_exact(x_test))
    if mode != "fixed":
        summary[f"{label}: final (effective) weight"] = learned

show = ["plain NN (w = 0)", "gNN, w = 0.1", "gNN, learnable (uncertainty)"]
plot_curves(x_test, u_exact(x_test), {k: preds[k] for k in show}, out / "solution.png",
            title=f"Function approximation ({args.target})",
            points=(x_train.detach(), u_train), points_label="training samples")
plot_curves(x_test, du_exact(x_test), {k: dpreds[k] for k in show}, out / "derivative.png",
            title="Derivative of the fitted networks", ylabel="u'(x)")
plot_history(histories, out / "error_history.png", title="Relative L2 error of u")
save_summary(out, summary)
