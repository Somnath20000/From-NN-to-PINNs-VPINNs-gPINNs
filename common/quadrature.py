r"""
Numerical integration (quadrature) and test functions for Variational PINNs.

A VPINN needs integrals such as  \int_a^b u'(x) v'(x) dx.
We approximate them with Gauss-Lobatto-Legendre (GLL) quadrature:

    \int_{-1}^{1} g(\xi) d\xi  ~=  sum_q  w_q g(\xi_q)

which is EXACT when g is a polynomial of degree <= 2Q-3 (Q = number of points).
The GLL points include the two end points -1 and +1.

Integrals over a physical element [a, b] are mapped to the reference interval
[-1, 1] with  x = a + (b - a)/2 * (\xi + 1),   dx = J d\xi,   J = (b - a)/2.

Test functions
--------------
We use  v_k(\xi) = P_{k+1}(\xi) - P_{k-1}(\xi),  k = 1, 2, ..., K
where P_n is the Legendre polynomial of degree n. Because P_n(+-1) = (+-1)^n,
every v_k vanishes at both ends of the element:  v_k(-1) = v_k(1) = 0.
That is why boundary terms disappear after integration by parts.
(Legendre polynomials are the Jacobi polynomials P^{(0,0)} used in the
original hp-VPINN code.)
"""

import numpy as np
from numpy.polynomial import legendre as L


def gauss_lobatto_legendre(n_points: int):
    """
    Nodes and weights of the n-point Gauss-Lobatto-Legendre rule on [-1, 1].

    Nodes  : -1, +1 and the roots of P'_{n-1}
    Weights: w_i = 2 / ( n (n-1) [P_{n-1}(x_i)]^2 )
    """
    if n_points < 2:
        raise ValueError("Need at least 2 GLL points")
    N = n_points - 1
    coeffs = np.zeros(N + 1)
    coeffs[N] = 1.0                                  # coefficients of P_N
    interior = L.legroots(L.legder(coeffs))          # roots of P'_N
    nodes = np.concatenate(([-1.0], np.sort(interior.real), [1.0]))
    weights = 2.0 / (N * (N + 1) * L.legval(nodes, coeffs) ** 2)
    return nodes, weights


def test_functions(n_test: int, xi: np.ndarray):
    """
    Evaluate v_k = P_{k+1} - P_{k-1} (k = 1..n_test) and their first two
    derivatives (w.r.t. the reference coordinate xi) at the points xi.

    Returns three arrays of shape (n_test, len(xi)):  v, dv/dxi, d2v/dxi2
    """
    xi = np.asarray(xi, dtype=float).ravel()
    v, dv, d2v = [], [], []
    for k in range(1, n_test + 1):
        c = np.zeros(k + 2)
        c[k + 1], c[k - 1] = 1.0, -1.0               # P_{k+1} - P_{k-1}
        v.append(L.legval(xi, c))
        dv.append(L.legval(xi, L.legder(c, 1)))
        d2v.append(L.legval(xi, L.legder(c, 2)))
    return np.array(v), np.array(dv), np.array(d2v)
