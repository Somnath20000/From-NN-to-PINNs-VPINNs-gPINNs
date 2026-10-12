# Figures — Stage 4: Variational PINNs (VPINN / hp-VPINN)

This folder holds the output of the three Stage 4 scripts. Each script writes to its own subfolder: the figures (PNG, 300 dpi) and a `summary.json` with the final numbers.

The numbers quoted below come from the run on **2026-10-12**, on the CPU with the default settings. If you rerun a script, its figures and `summary.json` are overwritten, but this README is not.

**How to read every plot**
* The **solid black line** is the exact solution; coloured broken lines are predictions.
* **Relative L2 error** = ‖prediction − exact‖ / ‖exact‖ (0.01 = 1 %). Error and history plots use a **log scale**.
* **2-D field plots** have one figure per field. The exact solution and all predictions share the colour scale of the exact solution (`viridis`). An **arrow at the end of the colour bar** means the prediction goes outside the exact solution's range. Error maps use `hot` (black = 0), each with its own range; compare the numbers on their colour bars.
* A **VPINN** does not require the residual to vanish at points. It requires the residual's integrals against smooth test functions to vanish (the weak form, as in finite elements). The integrals are computed by Gauss–Lobatto quadrature.

---

## `vpinn_01_multiscale_1d/` — the weak form on the multi-scale problem

Script: `python 04_vpinns/vpinn_01_multiscale_1d.py`

This is the same multi-scale problem as `03_pinn_failures/fail_01`: `u = 0.1 sin(8πx) + tanh(5πx)`. It uses the **same** network, optimiser and seeds (1–4), with 60 Legendre test functions and 80 quadrature points; the PINN used 100 points.

Two variational forms are compared:
* **Form 1:** `∫ −u″ v dx = ∫ f v dx`. No integration by parts; it still needs u″ of the network.
* **Form 3:** integration by parts twice. Only u itself enters the integral, plus boundary terms.

| Figure | What it shows | What it means |
|---|---|---|
| `form1_solutions.png` | Exact u and the form-1 VPINN for seeds 1–4. | Three seeds are accurate (0.04–0.22 %). **Seed 2 fails** (101 % error), although its variational loss is as small as the others'. Form 1 still checks u″ only at the quadrature points, so wiggles between them go unnoticed. |
| `form1_pointwise_error.png` | \|prediction − exact\| for each seed (log scale). | Seed 2's error is orders of magnitude above the other three seeds across the domain. |
| `form1_error_history.png` | Relative L2 error during training for each seed. | Seed 2 stays above 100 % for the whole run. The other three seeds reach 1e-3 or below within 5,000 iterations. |
| `form3_solutions.png` | Exact u and the form-3 VPINN for seeds 1–4. | **All four seeds are accurate** (0.028–0.24 % error), all on top of the exact curve. Compare the PINN in `fail_01`: 2.3–71 % for the same seeds. |
| `form3_pointwise_error.png` | \|prediction − exact\| for each seed. | Small, smooth errors for every seed. |
| `form3_error_history.png` | Relative L2 error during training. | Every seed converges. Form 3 is also the cheaper form: 0.62 ms per iteration against 0.89 ms for form 1. |

**Message:** testing the equation in an integral (weak) sense, especially after integrating by parts, gives a much more reliable loss than pointwise collocation.

---

## `vpinn_02_hp_poisson_2d/` — hp-VPINN in 2-D vs. PINN

Script: `python 04_vpinns/vpinn_02_hp_poisson_2d.py`

The problem is `−Δu = f` on (−1, 1)². The exact solution is `u = (0.1 sin(2πx) + tanh(10x)) sin(2πy)`, with a steep layer along x = 0.
* **hp-VPINN:** the square is split into 4 × 4 elements, each with 10 × 10 test functions ("h" = elements, "p" = polynomial degree). One network covers the whole domain.
* **PINN:** the same network and boundary points, with the strong residual at 100 random points.

| Figure | What it shows | What it means |
|---|---|---|
| `fields_exact_u.png` | The exact solution. | Four horizontal bands whose sign flips across the steep layer at x = 0. |
| `fields_vpinn_u.png` | hp-VPINN prediction (same colour scale). | Visually identical to the exact solution. |
| `fields_pinn_u.png` | PINN prediction (same colour scale; the arrow at −1.2 means it undershoots the exact minimum). | The structure is badly distorted, especially near the layer and in the lower left. |
| `fields_vpinn_error.png` | hp-VPINN pointwise error. | At most about **0.05** (see the ×10⁻² colour bar). The relative L2 error is **2.1 %**. |
| `fields_pinn_error.png` | PINN pointwise error. | Up to about **2**, i.e. errors as large as the solution itself. The relative L2 error is **62 %**. |
| `error_history.png` | Relative L2 error during training for both methods. | The hp-VPINN falls steadily to about 2 %. The PINN improves only slowly, from 100 % to 62 %, and is still far behind after 20,000 iterations. |

**Fairness note:** this compares the methods at the *settings of the original paper*, not at equal cost. The PINN uses only 100 residual points, while the VPINN evaluates the network at 6,400 quadrature points. The main README reports that the PINN reaches about 1 % with 1,600 points (`--method pinn --n_res 1600`).

---

## `vpinn_03_neumann_natural_bc/` — Neumann conditions enter the weak form "naturally"

Script: `python 04_vpinns/vpinn_03_neumann_natural_bc.py`

This is the pure-Neumann problem of `03_pinn_failures/fail_02`: `−u″ = 2`, `u′(0) = 0`, `u′(1) = −2`, exact solution `1 − x²`.

| Figure | What it shows | What it means |
|---|---|---|
| `solutions.png` | The exact solution and three VPINN runs. **Penalty:** Neumann data as extra loss terms, plus the anchor u(0) = 1. **Natural:** Neumann data inside the weak form's boundary term, plus the anchor. **Natural, no anchor:** the same without u(0) = 1. | *Penalty* (**0.33 %**) and *natural* (**0.0085 %**) are both on the exact curve, and the natural treatment is about 40× more accurate here with one loss weight fewer. *Natural, no anchor* has the correct shape but is shifted by **C = 3.0**, the same non-uniqueness as the PINN in `fail_02`. The weak form fixes *how* Neumann data enter, not the missing uniqueness. |

**Key message of Stage 4:** the variational (weak) form makes training more reliable (`vpinn_01`), lets domain decomposition handle hard 2-D layers (`vpinn_02`), and treats Neumann conditions naturally (`vpinn_03`). It cannot fix an ill-posed problem.
