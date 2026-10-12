"""
Shared building blocks used by every tutorial script.

You normally do NOT need to edit anything in this folder.
Each tutorial script (in the numbered folders) imports what it needs from here:

    utils.py       -> random seeds, CPU/GPU device, command-line options, output folders, error metrics
    networks.py    -> the fully connected neural network (MLP)
    derivatives.py -> a one-line helper for automatic differentiation
    training.py    -> the training loop (Adam, optionally followed by L-BFGS)
    quadrature.py  -> Gauss-Lobatto quadrature + Legendre test functions (for VPINNs)
    problems.py    -> exact solutions and forcing terms of every benchmark problem
    plotting.py    -> small plotting helpers that save .png figures
"""
