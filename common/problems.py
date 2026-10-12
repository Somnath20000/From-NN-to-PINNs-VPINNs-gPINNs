r"""
Benchmark problems used across the tutorial.

Every function takes a torch tensor x (shape [N, 1] in 1-D, [N, 2] in 2-D)
and returns a torch tensor of shape [N, 1]. Because they are written with
torch operations, we can also differentiate them with autograd when we need
e.g. df/dx for gradient-enhanced PINNs (see 05_gradient_enhanced).

All exact solutions were checked by hand AND numerically
(tests/test_core.py differentiates them with autograd).
"""

import math

import torch

PI = math.pi


# ---------------------------------------------------------------------------
# 1. Pure function-approximation targets (Stage 1 and Stage 5)
# ---------------------------------------------------------------------------
def oscillatory_u(x):
    """u(x) = -(1.4 - 3x) sin(18x) on [0, 1]  (gPINN paper, function approx.)"""
    return -(1.4 - 3 * x) * torch.sin(18 * x)


def oscillatory_du(x):
    """u'(x) = 3 sin(18x) + 18 (3x - 1.4) cos(18x)"""
    return 3 * torch.sin(18 * x) + 18 * (3 * x - 1.4) * torch.cos(18 * x)


def damped_sine_u(x):
    """u(x) = exp(-x) sin(30x) on [0, 1]  (higher frequency target)"""
    return torch.exp(-x) * torch.sin(30 * x)


def damped_sine_du(x):
    """u'(x) = exp(-x) (30 cos(30x) - sin(30x))"""
    return torch.exp(-x) * (30 * torch.cos(30 * x) - torch.sin(30 * x))


# ---------------------------------------------------------------------------
# 2. 1-D Poisson problem from the gPINN paper:  -u'' = f  on [0, pi]
#    u(x) = x + sum_{i=1}^{4} sin(i x)/i + sin(8x)/8,  u(0) = 0, u(pi) = pi
# ---------------------------------------------------------------------------
def poisson1d_u(x):
    u = x + torch.sin(8 * x) / 8
    for i in range(1, 5):
        u = u + torch.sin(i * x) / i
    return u


def poisson1d_du(x):
    du = 1 + torch.cos(8 * x)
    for i in range(1, 5):
        du = du + torch.cos(i * x)
    return du


def poisson1d_f(x):
    """f = -u'' = sum_{i=1}^{4} i sin(i x) + 8 sin(8x)"""
    f = 8 * torch.sin(8 * x)
    for i in range(1, 5):
        f = f + i * torch.sin(i * x)
    return f


# ---------------------------------------------------------------------------
# 3. Multi-scale problem with a sharp layer:  -u'' = f  on [-1, 1]
#    u(x) = 0.1 sin(omega x) + tanh(r x)
#    r controls the steepness of the layer at x = 0 (larger r = harder).
# ---------------------------------------------------------------------------
def sharp_u(x, r=5 * PI, omega=8 * PI):
    return 0.1 * torch.sin(omega * x) + torch.tanh(r * x)


def sharp_du(x, r=5 * PI, omega=8 * PI):
    return 0.1 * omega * torch.cos(omega * x) + r / torch.cosh(r * x) ** 2


def sharp_f(x, r=5 * PI, omega=8 * PI):
    """f = -u'' = 0.1 omega^2 sin(omega x) + 2 r^2 tanh(r x) sech^2(r x)"""
    return (0.1 * omega ** 2 * torch.sin(omega * x)
            + 2 * r ** 2 * torch.tanh(r * x) / torch.cosh(r * x) ** 2)


# ---------------------------------------------------------------------------
# 4. Pure-Neumann problem:  -u'' = 2 on [0, 1],  u'(0) = 0,  u'(1) = -2
#    Solutions: u(x) = 1 - x^2 + C  for ANY constant C (not unique!).
#    Adding u(0) = 1 picks C = 0.
# ---------------------------------------------------------------------------
def neumann_u(x):
    return 1 - x ** 2


def neumann_du(x):
    return -2 * x


def neumann_f(x):
    return 2 * torch.ones_like(x)


# ---------------------------------------------------------------------------
# 5. 2-D Poisson with a localised peak:  -Laplace(u) = f  on [0, 1]^2, u = 0 on boundary
#    u(x, y) = g(x, y)^a,  g = 16 x (1-x) y (1-y)     (gPINN paper, a = 10)
# ---------------------------------------------------------------------------
def peak2d_u(xy, a=10):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return (16 * x * (1 - x) * y * (1 - y)) ** a


def peak2d_f(xy, a=10):
    r"""
    f = -(u_xx + u_yy) with  u = g^a:
        u_xx = a(a-1) g^(a-2) g_x^2 + a g^(a-1) g_xx   (same for y)
        g_x = 16 (1-2x) y(1-y),   g_xx = -32 y(1-y)
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    g = 16 * x * (1 - x) * y * (1 - y)
    gx, gy = 16 * (1 - 2 * x) * y * (1 - y), 16 * (1 - 2 * y) * x * (1 - x)
    gxx, gyy = -32 * y * (1 - y), -32 * x * (1 - x)
    uxx = a * (a - 1) * g ** (a - 2) * gx ** 2 + a * g ** (a - 1) * gxx
    uyy = a * (a - 1) * g ** (a - 2) * gy ** 2 + a * g ** (a - 1) * gyy
    return -(uxx + uyy)


# ---------------------------------------------------------------------------
# 6. 2-D Poisson with a steep layer (hp-VPINN paper):  -Laplace(u) = f on [-1, 1]^2
#    u(x, y) = (0.1 sin(2 pi x) + tanh(r x)) sin(2 pi y),  r = 10
# ---------------------------------------------------------------------------
def layer2d_u(xy, r=10.0):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return (0.1 * torch.sin(2 * PI * x) + torch.tanh(r * x)) * torch.sin(2 * PI * y)


def layer2d_f(xy, r=10.0):
    x, y = xy[:, 0:1], xy[:, 1:2]
    X = 0.1 * torch.sin(2 * PI * x) + torch.tanh(r * x)
    Xxx = (-0.1 * (2 * PI) ** 2 * torch.sin(2 * PI * x)
           - 2 * r ** 2 * torch.tanh(r * x) / torch.cosh(r * x) ** 2)
    Y = torch.sin(2 * PI * y)
    Yyy = -(2 * PI) ** 2 * Y
    return -(Xxx * Y + X * Yyy)
