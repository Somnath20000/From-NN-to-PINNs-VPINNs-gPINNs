"""
==============================================================================
STAGE 1 - Script 1:  A plain neural network is "data hungry"
==============================================================================

QUESTION
    A neural network (NN) is a universal function approximator. So why not
    simply fit the solution of a differential equation with a NN?

PROBLEM
    Learn  u(x) = -(1.4 - 3x) sin(18x)  on [0, 1]  from labelled samples (x_i, u_i).
    We train the SAME network twice:
        (a) with 100 samples  -> plenty of data
        (b) with  10 samples  -> sparse data (the realistic case in science:
                                  measurements are expensive)
    and then look at
        * the error inside [0, 1],
        * the error of the DERIVATIVE u'(x) (physics often needs derivatives),
        * the error OUTSIDE the data range, on [1, 1.3] (extrapolation).

WHAT YOU SHOULD SEE
    * With 100 points the fit is excellent.
    * With 10 points the network still passes through the 10 points
      (training loss ~ 0), but between them it is wrong, and its derivative
      is much worse. A small training loss does NOT mean a correct model.
    * Neither network extrapolates: outside the data the NN knows nothing.
    -> A plain NN has no knowledge of the physics; it can only interpolate data.
       This motivates Physics-Informed Neural Networks (Stage 2).

HOW TO RUN  (from the repository root folder)
    python 01_neural_networks/nn_01_data_hungry.py
    python 01_neural_networks/nn_01_data_hungry.py --epochs 20000 --seed 7

THINGS TO TRY
    * Change N_SPARSE below to 20 or 30. How many points are "enough"?
    * Change the hidden layers in LAYERS. Does a bigger network help with
      10 points? (Usually not: the information is simply not in the data.)
"""

# -----------------------------------------------------------------------------
# 0. Imports. The two lines with sys.path let Python find the "common" folder.
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves, plot_history  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings you may want to change
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=10000, lr=1e-3)
N_DENSE = 100              # number of labelled samples in the "rich data" case
N_SPARSE = 10              # number of labelled samples in the "sparse data" case
LAYERS = [1, 20, 20, 20, 1]  # network size: input, hidden..., output
X_MIN, X_MAX = 0.0, 1.0    # training interval
X_EXTRAP = 1.3             # we also test on [X_MAX, X_EXTRAP]

setup(args.seed)
out = output_dir(__file__)

# -----------------------------------------------------------------------------
# 2. The function we want to learn (exact solution and its derivative)
# -----------------------------------------------------------------------------
u_exact, du_exact = P.oscillatory_u, P.oscillatory_du

# Dense test grid used ONLY to measure the error (never used for training)
x_test = torch.linspace(X_MIN, X_MAX, 1000).reshape(-1, 1)
x_extra = torch.linspace(X_MAX, X_EXTRAP, 300).reshape(-1, 1)
x_plot = torch.linspace(X_MIN, X_EXTRAP, 1300).reshape(-1, 1)


def fit(n_points):
    """Train a fresh network on n_points equally spaced labelled samples."""
    print(f"\n----- Training a plain NN with {n_points} labelled points -----")
    # Training data: x_i equally spaced, u_i = exact value (a "measurement")
    x_train = torch.linspace(X_MIN, X_MAX, n_points).reshape(-1, 1)
    u_train = u_exact(x_train)

    model = MLP(LAYERS, activation="tanh")

    # Loss = mean squared error between network output and data
    def loss_fn():
        mse = torch.mean((model(x_train) - u_train) ** 2)
        return {"total": mse}

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))

    history = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn)
    return model, x_train, u_train, history


def derivative_of(model, x):
    """u'(x) of the trained network, via automatic differentiation."""
    x = x.clone().requires_grad_(True)
    return grad(model(x), x).detach()


# -----------------------------------------------------------------------------
# 3. Train both networks
# -----------------------------------------------------------------------------
model_dense, xd, ud, hist_dense = fit(N_DENSE)
model_sparse, xs, us, hist_sparse = fit(N_SPARSE)

# -----------------------------------------------------------------------------
# 4. Evaluate: function error, derivative error, extrapolation error
# -----------------------------------------------------------------------------
results = {}
for name, model in [("dense", model_dense), ("sparse", model_sparse)]:
    with torch.no_grad():
        results[f"{name}: rel. L2 error of u on [0,1]"] = relative_l2(model(x_test), u_exact(x_test))
        results[f"{name}: rel. L2 error of u on [1,1.3] (extrap.)"] = relative_l2(
            model(x_extra), u_exact(x_extra))
    results[f"{name}: rel. L2 error of u' on [0,1]"] = relative_l2(
        derivative_of(model, x_test), du_exact(x_test))
results["sparse: final training loss"] = hist_sparse["total"][-1]

# -----------------------------------------------------------------------------
# 5. Plots (saved to 01_neural_networks/figures/nn_01_data_hungry/)
# -----------------------------------------------------------------------------
with torch.no_grad():
    plot_curves(x_plot, u_exact(x_plot),
                {f"NN, {N_DENSE} points": model_dense(x_plot),
                 f"NN, {N_SPARSE} points": model_sparse(x_plot)},
                out / "fit_and_extrapolation.png",
                title="Plain NN: fit inside [0,1], extrapolation beyond x=1",
                points=(xs, us), points_label=f"the {N_SPARSE} sparse samples")
plot_curves(x_test, du_exact(x_test),
            {f"NN, {N_DENSE} points": derivative_of(model_dense, x_test),
             f"NN, {N_SPARSE} points": derivative_of(model_sparse, x_test)},
            out / "derivative.png", title="Derivative u'(x) of the fitted networks",
            ylabel="u'(x)")
plot_history({f"{N_DENSE} points": hist_dense, f"{N_SPARSE} points": hist_sparse},
             out / "error_history.png", title="Test error during training")
plot_history({f"{N_DENSE} points": hist_dense, f"{N_SPARSE} points": hist_sparse},
             out / "training_loss_history.png", key="total", ylabel="training loss (MSE)",
             title="Training loss: both go to ~0, but only one model is right")

save_summary(out, results)
