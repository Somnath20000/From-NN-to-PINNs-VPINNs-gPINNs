# Stage 1 — Plain neural networks: why they are not enough

| Script | What it shows |
|---|---|
| `nn_01_data_hungry.py` | A network fits 100 samples well, but with 10 samples it is wrong between the points, its derivative is poor and it cannot extrapolate — even though the training loss is ~0. |
| `nn_02_spectral_bias.py` | Even with plenty of data, gradient descent learns the slow wave first and the fast wave much later (*spectral bias*). |

```bash
python 01_neural_networks/nn_01_data_hungry.py
python 01_neural_networks/nn_02_spectral_bias.py --K 20
```
Key message: a plain network knows nothing about the physics; it can only interpolate data.
