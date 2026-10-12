r"""
==============================================================================
STAGE 2 - Script 1:  Your first PINN - an ODE with (almost) no data
==============================================================================

IDEA OF A PINN  (Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 2019)
    Instead of fitting data, we ask the network to SATISFY THE EQUATION.
    For an ODE  y'(t) = F(t, y)  we define the residual

        r(t) = y_NN'(t) - F(t, y_NN(t)),

    where y_NN' is computed exactly by automatic differentiation, and we
    minimise the residual at many "collocation points" t_j, plus the error
    in the initial condition:

        Loss = mean_j r(t_j)^2  +  (y_NN(0) - y0)^2

    The collocation points need NO labels (we never need to know y(t_j)).
    Only the equation and one initial value are used.

PROBLEM
    y'(t) = -5 y(t),   y(0) = 1,   t in [0, 1].       Exact: y(t) = exp(-5 t)

    For comparison we also train a plain NN on N_DATA labelled samples.
    The PINN uses ZERO labelled samples inside the domain.

WHAT YOU SHOULD SEE
    The PINN recovers exp(-5t) accurately from the equation alone, while the
    plain NN with a few data points cannot know what happens between them.

HOW TO RUN
    python 02_pinns/pinn_01_first_order_ode.py
    python 02_pinns/pinn_01_first_order_ode.py --epochs 3000

THINGS TO TRY
    * Change LAMBDA to 20 (a stiffer, faster decay). Is it harder?
    * Remove the initial-condition term from the loss. What does the PINN
      learn now? (Hint: y = C exp(-5t) solves the ODE for EVERY C.)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves, plot_history, plot_loss_terms  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=5000, lr=1e-3)
LAMBDA = 5.0                  # decay rate in y' = -LAMBDA * y
Y0 = 1.0                      # initial value y(0)
T_END = 1.0
N_COLLOCATION = 100           # points where the ODE residual is enforced (no labels!)
N_DATA = 4                    # labelled points for the plain-NN comparison
LAYERS = [1, 32, 32, 1]

setup(args.seed)
out = output_dir(__file__)


def y_exact(t):
    return Y0 * torch.exp(-LAMBDA * t)


t_test = torch.linspace(0, T_END, 500).reshape(-1, 1)

# -----------------------------------------------------------------------------
# 2. PINN
# -----------------------------------------------------------------------------
# Collocation points. requires_grad_(True) tells PyTorch we will need dy/dt.
t_col = torch.linspace(0, T_END, N_COLLOCATION).reshape(-1, 1).requires_grad_(True)
t0 = torch.zeros(1, 1)

pinn = MLP(LAYERS, activation="tanh")


def pinn_loss():
    y = pinn(t_col)
    dy_dt = grad(y, t_col)                    # exact derivative of the network
    residual = dy_dt + LAMBDA * y             # r = y' + lambda y   (should be 0)
    loss_ode = torch.mean(residual ** 2)
    loss_ic = torch.mean((pinn(t0) - Y0) ** 2)   # initial condition y(0) = Y0
    return {"total": loss_ode + loss_ic, "ode": loss_ode, "ic": loss_ic}


def pinn_error():
    with torch.no_grad():
        return relative_l2(pinn(t_test), y_exact(t_test))


print("\n----- Training the PINN (equation + initial condition only) -----")
hist_pinn = train(pinn.parameters(), pinn_loss, args.epochs, args.lr, error_fn=pinn_error)

# -----------------------------------------------------------------------------
# 3. Plain NN with a handful of labelled samples (for comparison)
# -----------------------------------------------------------------------------
t_data = torch.linspace(0, T_END, N_DATA).reshape(-1, 1)
y_data = y_exact(t_data)
nn_model = MLP(LAYERS, activation="tanh")


def nn_loss():
    return {"total": torch.mean((nn_model(t_data) - y_data) ** 2)}


def nn_error():
    with torch.no_grad():
        return relative_l2(nn_model(t_test), y_exact(t_test))


print(f"\n----- Training a plain NN on {N_DATA} labelled samples -----")
hist_nn = train(nn_model.parameters(), nn_loss, args.epochs, args.lr, error_fn=nn_error)

# -----------------------------------------------------------------------------
# 4. Plots and summary
# -----------------------------------------------------------------------------
with torch.no_grad():
    plot_curves(t_test, y_exact(t_test),
                {"PINN (0 labelled points)": pinn(t_test),
                 f"plain NN ({N_DATA} labelled points)": nn_model(t_test)},
                out / "solution.png", title="y' = -5y, y(0) = 1", ylabel="y(t)",
                points=(t_data, y_data), points_label="NN training data", xlabel="t")
plot_history({"PINN": hist_pinn, "plain NN": hist_nn}, out / "error_history.png",
             title="Relative L2 error during training")
plot_loss_terms(hist_pinn, out / "pinn_loss_terms.png", title="PINN loss terms")

save_summary(out, {
    "PINN rel. L2 error": hist_pinn["error"][-1],
    f"plain NN ({N_DATA} points) rel. L2 error": hist_nn["error"][-1],
    "PINN final ODE residual loss": hist_pinn["ode"][-1],
})
