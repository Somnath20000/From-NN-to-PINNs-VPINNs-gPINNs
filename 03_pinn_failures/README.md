# Stage 3 — Where PINNs fail

| Script | Failure mode |
|---|---|
| `fail_01_multiscale.py` | Multi-scale solution (steep layer + fast wave): the outcome depends wildly on the random seed, and the loss does not reveal which run is wrong. |
| `fail_02_neumann_nonuniqueness.py` | Pure Neumann problem: the loss goes to ~1e-7 in every run, but each run picks a different constant. Ill-posedness is invisible to the optimiser. |
| `fail_03_collocation_overfitting.py` | Sharp 2-D peak: the residual is small at the training points but much larger at unseen points; the solution error stays large. |

```bash
python 03_pinn_failures/fail_01_multiscale.py
python 03_pinn_failures/fail_01_multiscale.py --n_col 1000
python 03_pinn_failures/fail_02_neumann_nonuniqueness.py
python 03_pinn_failures/fail_03_collocation_overfitting.py
```
Key message: a small PINN loss is **not** a certificate of a correct solution.
