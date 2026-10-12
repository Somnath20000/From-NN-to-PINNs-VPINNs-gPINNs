# Figures — Stage 5: gradient enhancement (gNN / gPINN)

This folder holds the output of the three Stage 5 scripts. Each script writes to its own subfolder: the figures (PNG, 300 dpi) and a `summary.json` with the final numbers.

The numbers quoted below come from the run on **2026-10-12**, on the CPU with the default settings. If you rerun a script, its figures and `summary.json` are overwritten, but this README is not.

**How to read every plot**
* The **solid black line** is the exact solution; coloured broken lines are predictions; open circles are training or collocation points.
* **Relative L2 error** = ‖prediction − exact‖ / ‖exact‖ (0.01 = 1 %). Error and history plots use a **log scale**.
* **2-D field plots** have one figure per field. Error maps use `hot` (black = 0), each with its own range; compare the numbers on the colour bars.
* **Gradient enhancement** adds *derivative* information to the loss. With data, that means also matching the slope u′ (gNN, "Sobolev training"). With physics, it means also requiring the *derivative of the PDE residual* to vanish (gPINN).

---

## `gpinn_01_function_approximation/` — learning from values and slopes

Script: `python 05_gradient_enhanced/gpinn_01_function_approximation.py`

Fit `u = −(1.4 − 3x) sin(18x)` from **12** samples, the same function as in Stage 1. Six runs are compared: a plain NN (values only); gNN with fixed weights w = 0.01, 0.1 and 1 on the slope term; and two variants where the weight is learned.

| Figure | What it shows | What it means |
|---|---|---|
| `solution.png` | Exact u, the plain NN, gNN with w = 0.1 and gNN with uncertainty weighting, plus the 12 samples. | The plain NN misses the curve between samples (**22 %** error). With slopes added, both gNNs follow the exact curve (**2.8 %** and **2.2 %**), about 10× better from the same 12 points. |
| `derivative.png` | The derivative u′ of the same three models vs. the exact u′. | The plain NN's derivative is poor (**47 %**). gNN, w = 0.1 reaches **6.6 %** and the uncertainty-weighted gNN **1.6 %**. |
| `error_history.png` | Relative L2 error of u during training for **all six** runs. | Every run with derivative information ends near 2–3 %, far below the plain NN. The exact weight matters less than including the slope at all. |

About the learnable weights:
* **Naive learnable weight** (`L_u + e^s · L_g`): gradient descent can only push this weight down, since switching the slope term off lowers the loss. It drifted from 1 to **0.38**. It still helps here only because it has not reached 0 yet.
* **Uncertainty weighting** (Kendall et al.) prevents that collapse and balances the two terms. It gave the best result in this run.

---

## `gpinn_02_poisson_1d/` — gPINN for the 1-D Poisson problem with few points

Script: `python 05_gradient_enhanced/gpinn_02_poisson_1d.py`

This is the problem of `02_pinns/pinn_02` (`−u″ = f` on (0, π)), but with only **15** collocation points. The gPINN also penalises the derivative of the residual, r′ = −u‴ − f′, with weight w_g = 0.01.

| Figure | What it shows | What it means |
|---|---|---|
| `solution.png` | Exact u, PINN and gPINN, with the 15 collocation points marked. | With so few points the PINN is visibly off (**3.0 %**). The gPINN is close to exact (**0.31 %**), about 10× better. |
| `derivative.png` | u′ of both models vs. the exact u′. | PINN **9.9 %** vs. gPINN **0.75 %**: the derivative improves even more than the solution. |
| `pointwise_error.png` | \|prediction − exact\| along x (log scale). | The gPINN's error is about an order of magnitude below the PINN's almost everywhere. |
| `error_history.png` | Relative L2 error during training. | The gPINN reaches about 0.4 % within 5,000 iterations. At that point the PINN is still at 10 %, and it levels off at about 3 % from iteration 10,000. |

---

## `gpinn_03_poisson_2d/` — gPINN on the sharp 2-D peak

Script: `python 05_gradient_enhanced/gpinn_03_poisson_2d.py`

This is the problem from `03_pinn_failures/fail_03`: the sharp peak `u = [16x(1−x)y(1−y)]^10`, hard boundary conditions and 400 random collocation points (a different random draw). The gPINN adds the x- and y-derivatives of the residual with weight w_g = 0.01. This requires third derivatives of the network.

| Figure | What it shows | What it means |
|---|---|---|
| `fields_exact_u.png` | The exact sharp peak. | A narrow peak of height 1 at the centre. |
| `fields_pinn_error.png` | Pointwise error of the plain PINN. | Large errors on the right flank of the peak (up to about 0.45, half the peak height). The relative L2 error is **50 %**. |
| `fields_gpinn_error.png` | Pointwise error of the gPINN (note the ×10⁻² on its colour bar). | The error is in the same place but at most about 0.012. The relative L2 error is **1.4 %**, about 35× better. |
| `error_history.png` | Relative L2 error during training for PINN and gPINN. | The PINN stalls at about 50 % from the first 2,000 iterations on. The gPINN keeps improving all the way (25 % → 1.4 %), because the extra term stops the residual from "bouncing" between collocation points (the failure seen in `fail_03`). |

**Key message of Stage 5:** instead of changing *how* the PDE is tested (Stage 4), add *more information*: derivatives of the data or of the residual. This helps most when points are scarce or the solution is sharp. The cost is more work per iteration (higher derivatives).
