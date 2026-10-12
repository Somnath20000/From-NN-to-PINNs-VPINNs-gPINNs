# Figures — Stage 3: where PINNs fail

This folder holds the output of the three Stage 3 scripts. Each script writes to its own subfolder: the figures (PNG, 300 dpi) and a `summary.json` with the final numbers.

The numbers quoted below come from the run on **2026-10-12**, on the CPU with the default settings. If you rerun a script, its figures and `summary.json` are overwritten, but this README is not.

**How to read every plot**
* The **solid black line** is the exact solution; coloured broken lines are predictions.
* **Relative L2 error** = ‖prediction − exact‖ / ‖exact‖ (0.01 = 1 %). Loss and error plots use a **log scale**.
* **2-D field plots** have one figure per field. Exact and predicted fields share the `viridis` colour scale. Error maps use `hot` (black = 0) with their own range.
* A **seed** fixes the random initial weights and the random collocation points. Everything else is identical between seeds.

---

## `fail_01_multiscale/` — failure 1: multi-scale solutions make training unreliable

Script: `python 03_pinn_failures/fail_01_multiscale.py`

The problem is `−u″ = f` on (−1, 1). The exact solution is `u = 0.1 sin(8πx) + tanh(5πx)`: a steep layer at x = 0 plus a small, fast wave. Each run uses 100 random collocation points.

| Figure | What it shows | What it means |
|---|---|---|
| `single_scale_solution.png` | The PINN on the easier **single-scale** problem `u = tanh(5πx)`, which has the layer but no wave. | The PINN solves it well (**0.07 %** error). The layer alone is not the problem. |
| `multiscale_solutions.png` | The exact multi-scale solution (black) and PINNs trained with seeds 1–4. | Same setup, very different outcomes. Seed 4 is good (**2.3 %**), seed 1 fair (**4.6 %**), seed 3 shifted (**15 %**), and seed 2 completely wrong (**71 %**, the blue curve far above the exact one). |
| `multiscale_pointwise_error.png` | \|prediction − exact\| for each seed (log scale). | The bad seeds are wrong across the whole domain, not just at the layer. |
| `multiscale_error_history.png` | Relative L2 error during training for each seed. | Training is erratic. Seed 4 drops to 33 %, jumps back up to 86 % and only then falls to 2.3 %. Seed 1 is still at 95 % after 5,000 iterations. Seed 2 gets steadily *worse* (53 % → 71 %). |
| `multiscale_pde_loss_history.png` | The **PDE residual loss** (what the optimiser actually minimises) for each seed. | All four losses end at similar values (1.6e-2 to 2.8e-2). Seed 2, the worst solution, does **not** have the largest loss. **The loss cannot tell you which run is wrong.** |

**Why:** spectral bias (Stage 1) makes the fast wave hard to learn. With only 100 points, a few points inside the steep layer dominate the loss, so the network can fit the points without fitting the function.

---

## `fail_02_neumann_nonuniqueness/` — failure 2: a tiny loss, but which solution?

Script: `python 03_pinn_failures/fail_02_neumann_nonuniqueness.py`

The problem is `−u″ = 2` on (0, 1) with **only Neumann** conditions `u′(0) = 0`, `u′(1) = −2`. Every function `u = 1 − x² + C` solves it, for any constant C. Without an extra condition, the problem has no unique solution.

| Figure | What it shows | What it means |
|---|---|---|
| `solutions.png` | The intended solution `1 − x²` (black); three PINNs trained with Neumann conditions only (seeds 1–3, each labelled with the constant C it picked); and one PINN with the extra anchor `u(0) = 1`. | The three Neumann-only runs have the correct *shape* but are shifted up by **C = 3.57, 3.37 and 3.72**: arbitrary values from the random start. Their losses are tiny (2–4 × 10⁻⁷), yet the error against `1 − x²` is **460–510 %**. With the anchor, the PINN lands on the right curve (**0.004 %** error). |

**Key message:** a PINN minimises a loss; it does not check whether the problem is well posed. Before trusting a small loss, make sure the continuous problem has a unique solution.

---

## `fail_03_collocation_overfitting/` — failure 3: overfitting the collocation points

Script: `python 03_pinn_failures/fail_03_collocation_overfitting.py`

The problem is the 2-D Poisson problem of Stage 2 (`pinn_03`), but with a much **sharper** peak: `u = [16x(1−x)y(1−y)]^10`. It uses hard boundary conditions and 400 random collocation points, the setting that worked well for the smooth peak.

| Figure | What it shows | What it means |
|---|---|---|
| `train_vs_unseen_residual.png` | Mean squared PDE residual at the **400 training points** (blue) and at **10,000 unseen points** (red dashed) during training. | The training residual falls to **1.1e-2**. The residual where the optimiser never looked stays at **68**, about **6,300×** larger. This is the PINN version of overfitting. |
| `residual_map.png` | log₁₀ \|PDE residual\| over the whole square, with the collocation points marked (cyan dots). Dark = small residual, bright = large. | Each point sits in a small dark "hole": the PDE is satisfied **at** the points and violated **between** them, especially around the peak in the centre. |
| `fields_exact_u.png` | The exact sharp peak. | A narrow peak of height 1 at (0.5, 0.5). |
| `fields_pinn_u.png` | The PINN prediction on the same colour scale. | It has a peak in the right place, but it is broader and more irregular than the exact one. |
| `fields_error.png` | Pointwise absolute error. | The error is concentrated on the flanks of the peak, up to about **0.13**, which is 13 % of the peak height. The relative L2 error is **18 %**, despite the small training loss. |
| `error_history.png` | Relative L2 error during training. | The error reaches about 14 % within 2,000 iterations, then gets *worse* (22 % at 7,000) and ends at 18 %. All the while, the training residual keeps falling: more training fits the 400 points better, not the solution. |

**Key message of Stage 3:** a small PINN loss is **not** a certificate of a correct solution. Stage 4 (VPINNs) and Stage 5 (gradient-enhanced PINNs) address these failures.
