# Stage 4 — Variational PINNs (VPINN / hp-VPINN)

| Script | What it shows |
|---|---|
| `vpinn_01_multiscale_1d.py` | The weak form on the multi-scale problem of `fail_01`, same seeds. Form 3 (integrated by parts twice) is accurate for every seed; form 1 still fails for one seed (aliasing between quadrature points). |
| `vpinn_02_hp_poisson_2d.py` | hp-VPINN (domain decomposition, 4×4 elements) vs. PINN in 2-D. |
| `vpinn_03_neumann_natural_bc.py` | Neumann data enter the weak form *naturally* through the boundary term; comparison with a penalty approach. |

```bash
python 04_vpinns/vpinn_01_multiscale_1d.py --forms 1,2,3
python 04_vpinns/vpinn_02_hp_poisson_2d.py --elements 2
python 04_vpinns/vpinn_03_neumann_natural_bc.py
```
Implementation notes: quadrature and Legendre test functions are in `common/quadrature.py`
(verified by `tests/test_core.py`). All element integrals are vectorised with `torch.einsum`.
