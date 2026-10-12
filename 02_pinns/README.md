# Stage 2 — Physics-Informed Neural Networks (PINNs)

| Script | What it shows |
|---|---|
| `pinn_01_first_order_ode.py` | `y' = -5y, y(0)=1` solved from the equation alone (no interior data). |
| `pinn_02_poisson_1d.py` | Boundary-value problem `-u'' = f` with soft (penalty) boundary conditions; accurate `u` **and** `u'`. |
| `pinn_03_poisson_2d_hard_bc.py` | 2-D Poisson with the boundary condition built into the network (output transform). |

```bash
python 02_pinns/pinn_01_first_order_ode.py
python 02_pinns/pinn_02_poisson_1d.py --n_col 15
python 02_pinns/pinn_03_poisson_2d_hard_bc.py --a 10
```
Key message: the PDE residual (computed by automatic differentiation) replaces labelled data.
