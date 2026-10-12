"""
Automatic differentiation helper.

PINNs need derivatives of the network output u(x) with respect to its INPUT x
(not with respect to the weights). PyTorch can compute them exactly
(up to round-off) with torch.autograd.grad — no finite differences needed.

Usage
-----
    x = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    u = model(x)
    u_x  = grad(u, x)        # du/dx
    u_xx = grad(u_x, x)      # d2u/dx2

For 2-D inputs x = (x, y) stored as columns, grad returns both partial
derivatives; pick the one you need:
    g = grad(u, xy);  u_x = g[:, 0:1];  u_y = g[:, 1:2]
"""

import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    d(outputs)/d(inputs), keeping the graph so we can differentiate again.

    `inputs` must have been created with requires_grad=True.
    `create_graph=True` is what allows 2nd, 3rd, ... derivatives AND lets the
    optimiser back-propagate through the derivative during training.
    """
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]
