"""
The neural network used everywhere: a fully connected network (multilayer perceptron).

    input x  ->  [Linear -> activation] x (hidden layers)  ->  Linear  ->  output u(x)

Why tanh (or sin) and not ReLU?
    PINNs need 2nd (and sometimes 3rd) derivatives of the network output.
    ReLU is piecewise linear, so its second derivative is zero almost everywhere
    and the PDE residual would carry no information. tanh and sin are smooth.
"""

import torch
import torch.nn as nn


class Sine(nn.Module):
    """sin(x) activation (used in the original VPINN papers)."""

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """
    Fully connected network.

    Parameters
    ----------
    layers : list of int
        Width of every layer including input and output,
        e.g. [1, 20, 20, 20, 1] = 1 input, three hidden layers of 20 neurons, 1 output.
    activation : "tanh" or "sin"
    output_transform : optional function g(x, y) -> u
        Lets us build boundary conditions directly into the network
        ("hard constraints"). Example for u(0)=u(1)=0:
            output_transform = lambda x, y: x * (1 - x) * y
        Then u is exactly zero at x=0 and x=1 whatever the weights are.
    """

    def __init__(self, layers, activation="tanh", output_transform=None):
        super().__init__()
        act = {"tanh": nn.Tanh, "sin": Sine}[activation]
        modules = []
        for i in range(len(layers) - 2):
            modules += [nn.Linear(layers[i], layers[i + 1]), act()]
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        self.output_transform = output_transform

        # Xavier (Glorot) initialisation: keeps the size of signals roughly
        # constant from layer to layer at the start of training.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.net(x)
        if self.output_transform is not None:
            y = self.output_transform(x, y)
        return y
