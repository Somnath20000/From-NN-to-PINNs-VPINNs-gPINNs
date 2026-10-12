"""
==============================================================================
STAGE 1 - Script 2:  Spectral bias - NNs learn low frequencies first
==============================================================================

QUESTION
    Even with LOTS of data, are all functions equally easy for a NN?

PROBLEM
    Fit  u(x) = sin(2 pi x) + 0.5 sin(2 pi K x)   on [0, 1]
    with K = 10 (a slow wave plus a fast wave), using 256 samples - plenty.

    During training we decompose the error  e(x) = NN(x) - u(x)  into Fourier
    modes and track separately
        * the error in the LOW-frequency mode  (frequency 1),
        * the error in the HIGH-frequency mode (frequency K).

WHAT YOU SHOULD SEE
    The low-frequency error drops quickly; the high-frequency error stays
    large for much longer. Snapshots of the prediction show a smooth curve
    first, and the fast wiggles only later. This "spectral bias" (Rahaman et
    al., 2019) is a property of gradient-descent training of NNs.

WHY THIS MATTERS LATER
    PINNs inherit this bias: solutions with sharp layers or high frequencies
    are hard for PINNs (see 03_pinn_failures). Many remedies (Fourier features,
    domain decomposition, variational/hp formulations) are attempts to fight it.

HOW TO RUN
    python 01_neural_networks/nn_02_spectral_bias.py
    python 01_neural_networks/nn_02_spectral_bias.py --K 20

THINGS TO TRY
    * Increase K (option --K). The high-frequency mode becomes even slower.
    * Change the activation in MLP(...) from "tanh" to "sin".
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.networks import MLP  # noqa: E402
from common.plotting import COLORS, finish, new_figure, plot_curves  # noqa: E402
from common.training import train  # noqa: E402
from common.utils import get_args, output_dir, relative_l2, save_summary, setup  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
args = get_args(__doc__.split("\n")[2], epochs=20000, lr=1e-3,
                extra=[("K", int, 10, "frequency of the fast wave")])
K = args.K
N_DATA = 256                      # number of (equally spaced) samples
LAYERS = [1, 50, 50, 50, 1]
SNAPSHOTS = [200, 1000, 5000, args.epochs]   # iterations at which we save the prediction

setup(args.seed)
out = output_dir(__file__)


def u_exact(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.sin(2 * math.pi * K * x)


# Samples on [0, 1) - a periodic grid, so that the discrete Fourier transform
# of the error is meaningful.
x = (torch.arange(N_DATA) / N_DATA).reshape(-1, 1)
u = u_exact(x)

model = MLP(LAYERS, activation="tanh")


def loss_fn():
    return {"total": torch.mean((model(x) - u) ** 2)}


# -----------------------------------------------------------------------------
# 2. During training, record the Fourier amplitude of the error at
#    frequency 1 (slow) and frequency K (fast), plus a few snapshots.
# -----------------------------------------------------------------------------
low_err, high_err, steps, snapshots = [], [], [], {}
counter = {"step": 0}


def error_fn():
    with torch.no_grad():
        pred = model(x)
    err = (pred - u).cpu().numpy().ravel()
    spectrum = np.abs(np.fft.rfft(err)) / (N_DATA / 2)   # amplitude of each mode
    low_err.append(spectrum[1])
    high_err.append(spectrum[K])
    steps.append(counter["step"])
    if counter["step"] in SNAPSHOTS:
        snapshots[counter["step"]] = pred.clone()
    counter["step"] += 100                               # train() records every 100 its
    return relative_l2(pred, u)


history = train(model.parameters(), loss_fn, args.epochs, args.lr,
                error_fn=error_fn, record_every=100)

# -----------------------------------------------------------------------------
# 3. Plots
# -----------------------------------------------------------------------------
fig, ax = new_figure()
ax.semilogy(steps, low_err, color=COLORS[1], label="error in frequency 1 (slow wave)")
ax.semilogy(steps, high_err, "--", color=COLORS[0], label=f"error in frequency {K} (fast wave)")
finish(fig, ax, out / "spectral_bias.png", "iteration", "Fourier amplitude of the error",
       "Spectral bias: the slow wave is learned first")

plot_curves(x, u, {f"iteration {s}": p for s, p in sorted(snapshots.items())},
            out / "snapshots.png", title="NN prediction at different iterations")


def first_step_below(series, tol=0.05):
    """First iteration at which the error amplitude drops below tol (or None)."""
    for s, v in zip(steps, series):
        if v < tol:
            return s
    return "never"


save_summary(out, {
    "final rel. L2 error": history["error"][-1],
    "final error amplitude, frequency 1": float(low_err[-1]),
    f"final error amplitude, frequency {K}": float(high_err[-1]),
    "first iteration with low-freq error < 0.05": first_step_below(low_err),
    "first iteration with high-freq error < 0.05": first_step_below(high_err),
})
