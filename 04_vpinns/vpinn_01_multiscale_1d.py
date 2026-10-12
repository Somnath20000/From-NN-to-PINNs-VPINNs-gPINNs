r"""
==============================================================================
STAGE 4 - Script 1:  Variational PINN (VPINN) - the weak form to the rescue
==============================================================================

THE IDEA  (Kharazmi, Zhang & Karniadakis, arXiv:1912.00873, 2019;
           hp-VPINNs: CMAME 374, 113547, 2021)
    A PINN asks the residual  r(x) = -u_NN''(x) - f(x)  to vanish at
    individual points. A VPINN asks it to vanish "on average" against a set of
    smooth TEST FUNCTIONS v_k, exactly like the Galerkin / finite-element method:

        R_k = \int_a^b  r(x) v_k(x) dx  = 0,           k = 1, ..., K.

    The integrals are computed with Gauss-Lobatto quadrature (Q points).

THREE VARIATIONAL FORMS  (integration by parts moves derivatives from u_NN
onto v_k; boundary terms with v_k vanish because v_k(a) = v_k(b) = 0)
    form 1:  R_k = \int -u'' v_k dx               - \int f v_k dx   (no integration by parts)
    form 2:  R_k = \int  u'  v_k' dx              - \int f v_k dx   (once)
    form 3:  R_k = -\int u v_k'' dx + [u v_k']_a^b - \int f v_k dx   (twice)
    Forms 2 and 3 need fewer derivatives of the network -> cheaper, and they
    are well defined even when u is not twice differentiable.

LOSS
    Loss = sum_elements mean_k R_k^2  +  w_bc * mean (u_NN - g)^2 on the boundary

TEST FUNCTIONS AND QUADRATURE  (see common/quadrature.py)
    v_k(xi) = P_{k+1}(xi) - P_{k-1}(xi)  (Legendre polynomials), on the reference
    interval [-1, 1], mapped to each element [a, b] by x = a + (b-a)(xi+1)/2.

EXPERIMENT
    The multi-scale problem of 03_pinn_failures/fail_01:
        -u'' = f on (-1, 1),  u = 0.1 sin(8 pi x) + tanh(5 pi x)
    SAME network, SAME optimiser, SAME seeds (1, 2, 3, 4), and a comparable
    number of points: Q = 80 quadrature points (the PINN used 100).
    By default forms 1 and 3 are trained (8 runs, roughly 5-6 minutes on a laptop CPU).

WHAT YOU SHOULD SEE  (numbers from our run are in the README)
    * Form 3: every seed reaches 0.03-0.3 % relative error. Compare with the PINN,
      whose error ranged from a few % to > 70 % depending on the seed.
    * Form 1: three seeds are just as good, but ONE seed fails badly although
      its variational loss is small. Form 1 still evaluates u_NN'' only at the
      quadrature points; the network can develop wiggles BETWEEN the points
      that the quadrature cannot see (aliasing) - the same disease as the
      PINN, in milder form.
    * Integrating by parts (forms 2 and 3) removes this failure here: the
      network enters through u' or u itself, so hidden wiggles in u'' no longer
      go unnoticed. More quadrature points (--n_quad 160) or elements also help.
    -> VPINNs are not magic, but the weak form gives a much more reliable loss.

HOW TO RUN
    python 04_vpinns/vpinn_01_multiscale_1d.py
    python 04_vpinns/vpinn_01_multiscale_1d.py --forms 1,2,3
    python 04_vpinns/vpinn_01_multiscale_1d.py --forms 1 --seeds 2 --n_quad 160
    python 04_vpinns/vpinn_01_multiscale_1d.py --forms 3 --elements 4 --n_test 15 --n_quad 40

THINGS TO TRY
    * --forms 1,2,3 : which form is most accurate, and which is cheapest
      (look at "milliseconds per iteration" in the summary)?
    * --n_test 10 : too few test functions cannot "see" the fast wave.
    * --elements 4 : hp-VPINN = split the domain into elements, each with its
      own set of test functions (h-refinement).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.networks import MLP  # noqa: E402
from common.plotting import plot_curves, plot_history, plot_pointwise_error  # noqa: E402
from common.quadrature import gauss_lobatto_legendre, test_functions  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3, extra=[
    ("forms", str, "1,3", "comma-separated variational forms to run (1, 2, 3)"),
    ("elements", int, 1, "number of equal sub-intervals (elements)"),
    ("n_test", int, 60, "number of test functions per element"),
    ("n_quad", int, 80, "number of quadrature points per element"),
    ("w_bc", float, 10.0, "weight of the boundary loss"),
    ("seeds", str, "1,2,3,4", "comma-separated list of random seeds"),
])
SEEDS = [int(s) for s in args.seeds.split(",")]    # identical to 03_pinn_failures/fail_01
FORMS = [int(k) for k in args.forms.split(",")]
R, OMEGA = 5 * math.pi, 8 * math.pi
LAYERS = [1, 20, 20, 20, 20, 1]    # identical to the PINN of fail_01
X_MIN, X_MAX = -1.0, 1.0

setup(args.seed)
out = output_dir(__file__)
u_exact = lambda x: P.sharp_u(x, r=R, omega=OMEGA)   # noqa: E731
f = lambda x: P.sharp_f(x, r=R, omega=OMEGA)         # noqa: E731
x_test = torch.linspace(X_MIN, X_MAX, 2001).reshape(-1, 1)

# -----------------------------------------------------------------------------
# 2. Everything that does NOT depend on the network is computed once:
#    quadrature, test functions, element geometry and the right-hand side.
# -----------------------------------------------------------------------------
E, K, Q = args.elements, args.n_test, args.n_quad

# Reference quadrature on [-1, 1] and test functions at those points
xi, w = gauss_lobatto_legendre(Q)                    # shapes (Q,), (Q,)
V, dV, d2V = test_functions(K, xi)                   # each (K, Q), derivatives in xi
_, dV_end, _ = test_functions(K, np.array([-1.0, 1.0]))   # v_k'(-1), v_k'(1): (K, 2)
xi, w, V, dV, d2V, dV_end = (torch.tensor(a) for a in (xi, w, V, dV, d2V, dV_end))

# Elements [a_e, b_e] and Jacobians J_e = (b_e - a_e)/2   (shapes (E, 1))
nodes = torch.linspace(X_MIN, X_MAX, E + 1)
a_e, b_e = nodes[:-1, None], nodes[1:, None]
J = (b_e - a_e) / 2

# Physical quadrature points in every element: x_eq = a_e + J_e (xi_q + 1), shape (E, Q)
x_quad = a_e + J * (xi[None, :] + 1)

# Right-hand side F_ek = \int_element f v_k dx = J_e * sum_q w_q f(x_eq) v_k(xi_q)
#   einsum('kq,eq->ek', A, B) means: for every e and k, sum over q of A[k,q]*B[e,q]
F = J * torch.einsum("kq,eq->ek", V * w, f(x_quad))

# Flattened quadrature points for the network (need gradients for forms 1, 2)
x_quad_flat = x_quad.reshape(-1, 1).requires_grad_(True)
x_ends = torch.cat([a_e, b_e], dim=1).reshape(-1, 1)   # element end points (form 3)
x_bc = torch.tensor([[X_MIN], [X_MAX]])
u_bc = u_exact(x_bc)


def variational_residual(model, form):
    """R[e, k] for every element e and test function k  (shape (E, K))."""
    u = model(x_quad_flat)                                   # (E*Q, 1)
    if form == 3:
        # form 3: -(1/J) sum_q w_q u v_k''  +  (1/J) [u(b) v_k'(1) - u(a) v_k'(-1)]
        U = u.reshape(E, Q)
        u_ends = model(x_ends).reshape(E, 2)                 # [u(a_e), u(b_e)]
        boundary = u_ends[:, 1:2] * dV_end[:, 1] - u_ends[:, 0:1] * dV_end[:, 0]
        return (-torch.einsum("kq,eq->ek", d2V * w, U) + boundary) / J - F
    u_x = grad(u, x_quad_flat)                               # du/dx at quadrature pts
    if form == 2:
        # form 2: \int u' v_k' dx = sum_q w_q u'(x_q) dv_k/dxi  (the J's cancel)
        return torch.einsum("kq,eq->ek", dV * w, u_x.reshape(E, Q)) - F
    # form 1: \int -u'' v_k dx = -J sum_q w_q u''(x_q) v_k(xi_q)
    u_xx = grad(u_x, x_quad_flat)
    return -J * torch.einsum("kq,eq->ek", V * w, u_xx.reshape(E, Q)) - F


def run_vpinn(seed, form):
    setup(seed)
    model = MLP(LAYERS, activation="tanh")

    def loss_fn():
        R_ek = variational_residual(model, form)
        loss_var = torch.sum(torch.mean(R_ek ** 2, dim=1))   # mean over k, sum over e
        loss_bc = torch.mean((model(x_bc) - u_bc) ** 2)
        return {"total": loss_var + args.w_bc * loss_bc, "variational": loss_var, "bc": loss_bc}

    def error_fn():
        with torch.no_grad():
            return relative_l2(model(x_test), u_exact(x_test))

    hist = train(model.parameters(), loss_fn, args.epochs, args.lr, error_fn=error_fn,
                 log_every=5000, lbfgs_iters=args.lbfgs)
    with torch.no_grad():
        return model(x_test), hist


# -----------------------------------------------------------------------------
# 3. Train every requested form with the same seeds as the PINN experiment
# -----------------------------------------------------------------------------
summary = {}
for form in FORMS:
    preds, errors, histories = {}, {}, {}
    for seed in SEEDS:
        print(f"\n===== VPINN form {form} ({E} element(s), K={K}, Q={Q}), seed {seed} =====")
        t0 = time.time()
        u_pred, hist = run_vpinn(seed, form)
        ms_per_it = 1000 * (time.time() - t0) / max(args.epochs, 1)
        label = f"seed {seed}"
        preds[label], histories[label] = u_pred, hist
        errors[label] = u_pred - u_exact(x_test)
        summary[f"form {form}, {label}: rel. L2 error"] = relative_l2(u_pred, u_exact(x_test))
        summary[f"form {form}, {label}: final variational loss"] = hist["variational"][-1]
    summary[f"form {form}: milliseconds per iteration (last seed)"] = round(ms_per_it, 2)
    errs = [summary[f"form {form}, seed {s}: rel. L2 error"] for s in SEEDS]
    summary[f"form {form}: best / worst error over seeds"] = f"{min(errs):.2e} / {max(errs):.2e}"

    plot_curves(x_test, u_exact(x_test), preds, out / f"form{form}_solutions.png",
                title=f"VPINN form {form} on the multi-scale problem")
    plot_pointwise_error(x_test, errors, out / f"form{form}_pointwise_error.png")
    plot_history(histories, out / f"form{form}_error_history.png",
                 title=f"VPINN form {form}: error during training")

summary["settings"] = f"elements={E}, n_test={K}, n_quad={Q}, epochs={args.epochs}"
save_summary(out, summary)
