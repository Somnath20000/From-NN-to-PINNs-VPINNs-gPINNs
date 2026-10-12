# Stage 5 — Gradient enhancement: a different perspective

| Script | What it shows |
|---|---|
| `gpinn_01_function_approximation.py` | Sobolev/gradient-enhanced fitting with derivative data; fixed weights vs. learnable weights (and why a naive learnable weight collapses to zero). |
| `gpinn_02_poisson_1d.py` | gPINN: also penalise the derivative of the PDE residual. 1-D Poisson with only 15 points. |
| `gpinn_03_poisson_2d.py` | gPINN on the sharp 2-D peak of `fail_03`. |

```bash
python 05_gradient_enhanced/gpinn_01_function_approximation.py --target damped_sine
python 05_gradient_enhanced/gpinn_02_poisson_1d.py --w_g 0.1
python 05_gradient_enhanced/gpinn_03_poisson_2d.py --w_g 1e-3
```
Key message: instead of changing *how* the PDE is tested, add *more information* (derivatives) to the loss.
