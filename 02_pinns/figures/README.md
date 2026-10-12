# Figures — Stage 2: Physics-Informed Neural Networks (PINNs)

This folder holds the output of the three Stage 2 scripts. Each script writes to its own subfolder: the figures (PNG, 300 dpi) and a `summary.json` with the final numbers.

The numbers quoted below come from the run on **2026-10-12**, on the CPU with the default settings. If you rerun a script, its figures and `summary.json` are overwritten, but this README is not.

**How to read every plot**
* The **solid black line** is the exact solution. Coloured broken lines are predictions. **Open circles** mark the data points used to train the plain network that serves as a comparison.
* **Relative L2 error** = ‖prediction − exact‖ / ‖exact‖ (0.01 = 1 %). History and error plots use a **log scale**.
* A **PINN** is trained on the *equation*: its loss is the PDE/ODE residual (computed by automatic differentiation) at unlabelled "collocation" points, plus the boundary or initial conditions.
* **2-D field plots** have one figure per field. The exact solution and the prediction share the same `viridis` colour scale, so they can be compared by eye. Error maps use the `hot` scale (black = 0) with their own range; read the values on the colour bar. An arrow at the end of a colour bar means some values fall outside its range.

---

## `pinn_01_first_order_ode/` — the first PINN: an ODE without data

Script: `python 02_pinns/pinn_01_first_order_ode.py`

The problem is `y′(t) = −5 y(t)`, `y(0) = 1`, on [0, 1], with exact solution `y = e^(−5t)`. The PINN uses **zero** labelled points inside the domain: only the ODE at 100 collocation points and the initial value. For comparison, a plain network is fitted to **4** labelled samples.

| Figure | What it shows | What it means |
|---|---|---|
| `solution.png` | Exact y(t) (black), PINN (red), plain NN (blue) and the plain NN's 4 data points (circles). | The PINN lies on the exact curve (**0.097 %** error) using only the equation. The plain NN with 4 points is clearly worse (**5.0 %**). |
| `error_history.png` | Relative L2 error during training for both models. | The PINN's error falls well below the plain NN's and keeps decreasing. |
| `pinn_loss_terms.png` | The PINN's loss terms: `ode` (residual of y′ + 5y), `ic` (initial-condition error) and `total`. | Both terms go down together; the ODE residual ends at 3.2e-5. If the `ic` term were dropped, any C·e^(−5t) would satisfy the ODE, so this term is what pins down the answer. |

---

## `pinn_02_poisson_1d/` — a boundary-value problem: −u″ = f

Script: `python 02_pinns/pinn_02_poisson_1d.py`

The problem is `−u″ = f` on (0, π) with `u(0) = 0`, `u(π) = π`. The exact solution is `u = x + Σ sin(ix)/i + sin(8x)/8`. The PINN uses 50 collocation points and the 2 boundary values. The plain NN uses 15 labelled samples.

| Figure | What it shows | What it means |
|---|---|---|
| `solution.png` | Exact u (black), PINN, plain NN, and the plain NN's 15 data points. | Both look close at this scale, but the PINN's error is **0.0036 %** against **0.90 %** for the plain NN, 250× smaller. |
| `derivative.png` | u′(x) of both models vs. the exact u′. | The PINN's derivative is accurate (**0.015 %**); the plain NN's is visibly off (**17 %**). The equation constrains curvature everywhere, data constrain only values at points. |
| `pointwise_error.png` | \|prediction − exact\| at every x (log scale). | The PINN's error stays near 1e-4 along the whole interval. The plain NN's error swings between about 1e-3 and 1e-1: it drops sharply at its 15 data points and peaks between them. |
| `error_history.png` | Relative L2 error during training. | The PINN's curve is noisy: Adam makes it jump between about 1e-5 and 3e-3. It still stays below the plain NN almost all the time, and ends at 3.6e-5. The plain NN plateaus near 2e-2 and ends at 9e-3. |
| `pinn_loss_terms.png` | The PINN's `pde`, `bc` and `total` loss. | The boundary loss becomes tiny quickly; the PDE residual dominates the total. |

---

## `pinn_03_poisson_2d_hard_bc/` — a 2-D PINN with "hard" boundary conditions

Script: `python 02_pinns/pinn_03_poisson_2d_hard_bc.py`

The problem is `−(u_xx + u_yy) = f` on the unit square with `u = 0` on the boundary. The exact solution is `u = [16x(1−x)y(1−y)]^4`, a smooth peak in the centre. The network output is multiplied by `x(1−x)y(1−y)`, so the boundary condition holds **exactly** for any weights. The loss is only the PDE residual at 400 random points.

| Figure | What it shows | What it means |
|---|---|---|
| `fields_exact_u.png` | The exact solution u(x, y). | A single peak of height 1 at the centre, zero on the boundary. |
| `fields_pinn_u.png` | The PINN prediction, on the same colour scale as the exact field. | It looks identical to the exact field. |
| `fields_error.png` | Pointwise absolute error \|PINN − exact\|. | The maximum error is only **1.3e-3** (largest spots near (0.2, 0.45) and (0.4, 0.8)). The relative L2 error is **0.15 %**. The error goes to 0 at the boundary because of the hard constraint. |
| `error_history.png` | Relative L2 error during training. | A fast drop to about 0.7 % within 2,000 iterations, then a slow improvement with small bumps to 0.15 %. |
| `loss.png` | The PDE residual loss during training (`pde` and `total` coincide, because there is no boundary term). | No boundary weight had to be tuned. |

**Key message of Stage 2:** the PDE residual, computed exactly by autograd, can replace labelled data. PINNs get accurate solutions *and derivatives* from the equation alone.
