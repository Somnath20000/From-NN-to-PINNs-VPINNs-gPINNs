"""
Unit tests for the shared code. Run with:   pytest -q tests
They check the mathematics (quadrature exactness, test functions, exact
solutions vs. their forcing terms) so that the tutorials rest on solid ground.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import problems as P  # noqa: E402
from common.derivatives import grad  # noqa: E402
from common.quadrature import gauss_lobatto_legendre  # noqa: E402
from common.quadrature import test_functions as legendre_test_functions  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("n", [3, 5, 20, 80, 150])
def test_gll_exact_for_polynomials(n):
    nodes, w = gauss_lobatto_legendre(n)
    assert np.isclose(w.sum(), 2.0)
    assert nodes[0] == -1.0 and nodes[-1] == 1.0
    for deg in range(0, 2 * n - 2):          # exact up to degree 2n-3
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert np.isclose(np.sum(w * nodes ** deg), exact, atol=1e-11), deg


def test_test_functions_vanish_and_derivatives():
    xi = np.linspace(-1, 1, 2001)
    v, dv, d2v = legendre_test_functions(10, xi)
    assert np.allclose(v[:, [0, -1]], 0.0, atol=1e-12)
    h = xi[1] - xi[0]
    assert np.allclose(np.gradient(v, h, axis=1)[:, 5:-5], dv[:, 5:-5], rtol=1e-3, atol=1e-2)
    assert np.allclose(np.gradient(dv, h, axis=1)[:, 5:-5], d2v[:, 5:-5], rtol=1e-2, atol=1e-1)


def _second_derivative(u_fn, x):
    x = x.clone().requires_grad_(True)
    u = u_fn(x)
    du = grad(u, x)
    return du, grad(du, x)


@pytest.mark.parametrize("u, du, f, a, b", [
    (P.poisson1d_u, P.poisson1d_du, P.poisson1d_f, 0.0, np.pi),
    (P.sharp_u, P.sharp_du, P.sharp_f, -1.0, 1.0),
    (P.neumann_u, P.neumann_du, P.neumann_f, 0.0, 1.0),
])
def test_1d_problems_consistent(u, du, f, a, b):
    x = torch.linspace(a, b, 301).reshape(-1, 1)
    du_auto, d2u_auto = _second_derivative(u, x)
    assert torch.allclose(du_auto, du(x), atol=1e-8)
    assert torch.allclose(-d2u_auto, f(x), atol=1e-6 * float(f(x).abs().max()))


@pytest.mark.parametrize("u, du", [(P.oscillatory_u, P.oscillatory_du),
                                   (P.damped_sine_u, P.damped_sine_du)])
def test_function_targets(u, du):
    x = torch.linspace(0, 1, 201).reshape(-1, 1).requires_grad_(True)
    assert torch.allclose(grad(u(x), x), du(x), atol=1e-8)


@pytest.mark.parametrize("u, f, lo, hi", [(P.peak2d_u, P.peak2d_f, 0.0, 1.0),
                                          (P.layer2d_u, P.layer2d_f, -1.0, 1.0)])
def test_2d_problems_consistent(u, f, lo, hi):
    torch.manual_seed(0)
    xy = (lo + (hi - lo) * torch.rand(200, 2)).requires_grad_(True)
    g = grad(u(xy), xy)
    uxx = grad(g[:, 0:1], xy)[:, 0:1]
    uyy = grad(g[:, 1:2], xy)[:, 1:2]
    ref = f(xy)
    assert torch.allclose(-(uxx + uyy), ref, atol=1e-6 * float(ref.abs().max()))
