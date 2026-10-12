# Figures — Stage 1: plain neural networks

This folder holds the output of the two Stage 1 scripts. Each script writes to its own subfolder: the figures (PNG, 300 dpi) and a `summary.json` with the final numbers.

The numbers quoted below come from the run on **2026-10-12**, on the CPU with the default settings. If you rerun a script, its figures and `summary.json` are overwritten, but this README is not, so the numbers here can then be out of date.

**How to read every plot**
* The **solid black line** is always the exact (true) solution. Coloured broken lines are network predictions.
* **Relative L2 error** = ‖prediction − exact‖ / ‖exact‖ on a fine test grid. A value of 0.01 means 1 % error.
* Plots against "Iteration" use a **log scale** on the y-axis. Each step down of one grid line is a factor of 10.

---

## `nn_01_data_hungry/` — a plain network needs lots of data

Script: `python 01_neural_networks/nn_01_data_hungry.py`

The task is plain curve fitting, with no physics. Learn `u(x) = −(1.4 − 3x) sin(18x)` on [0, 1] from labelled samples (x, u(x)). The **same network** is trained twice: once with **100** samples ("dense") and once with only **10** samples ("sparse").

| Figure | What it shows | What it means |
|---|---|---|
| `fit_and_extrapolation.png` | The exact u(x) (black) on [0, 1.3], the two trained networks (red = 100 points, blue = 10 points) and the 10 sparse samples (open circles). Training only covers [0, 1]; the part beyond x = 1 is **extrapolation**. | **Inside [0, 1]:** the 100-point network lies on top of the exact curve (1.2 % error). The 10-point network passes through all 10 circles but is wrong between them (29 % error). **Beyond x = 1:** *both* networks fail completely (110 % and 166 % error). A network knows nothing outside its data. |
| `derivative.png` | The derivative u′(x) of each trained network (computed by automatic differentiation) vs. the exact u′(x). | Derivatives are much less accurate than the values: 7.3 % (dense) and 54 % (sparse). This matters because physics equations are built from derivatives. |
| `error_history.png` | Relative L2 error on the test grid during training, one curve per network (log scale). | The dense network keeps improving. The sparse one levels off at a high error, because more training cannot add information that is not in the 10 points. |
| `training_loss_history.png` | The **training loss** (mean squared misfit at the training points) during training. | Both losses go almost to zero; the sparse one ends at 4.2e-5. A tiny training loss does **not** mean the model is correct: compare it with the 29 % test error above. |

**Key message:** a plain network only interpolates its data. With little data it is wrong between the points and outside them, and its derivatives are worse still. This motivates PINNs (Stage 2), which use the equation instead of data.

---

## `nn_02_spectral_bias/` — networks learn slow waves before fast waves

Script: `python 01_neural_networks/nn_02_spectral_bias.py`

Fit `u(x) = sin(2πx) + 0.5 sin(2π·10·x)`, a slow wave plus a fast wave (frequency K = 10), using 256 samples, which is plenty of data. During training, the error is split into Fourier modes so the slow and fast parts can be tracked separately.

| Figure | What it shows | What it means |
|---|---|---|
| `spectral_bias.png` | The size (Fourier amplitude) of the error in **frequency 1** (slow wave, blue) and in **frequency 10** (fast wave, red dashed) against iteration (log scale). | The slow-wave error drops below 0.05 after about **300** iterations; the fast-wave error only after about **3,400**, more than 10× later. This is **spectral bias**: gradient-descent training learns smooth, low-frequency features first. |
| `snapshots.png` | The network's prediction at iterations 200, 1,000, 5,000 and 20,000 vs. the exact function. | Early snapshots are smooth curves that follow only the slow wave. The fast wiggles appear later. At the end the fit is good (0.48 % error). |

**Key message:** high-frequency or sharp features are slow and hard for networks to learn, even with plenty of data. PINNs inherit this, which is one cause of the failures in Stage 3.
