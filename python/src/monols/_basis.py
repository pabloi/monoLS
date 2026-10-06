"""The basis z = A @ w of spec/ALGORITHM.md section 4, applied in O(N k) without forming A.

Coefficient layout: [s_0 (intercept), s_1 .. s_{n_start-1} (start values), d_0 .. (knots)].
"""
import numpy as np


def n_start(N, k):
    """Number of start coefficients (intercept included) for N samples and order k."""
    return min(k, N - 1) + 1


def _rev_tail_sums(u):
    """t[i] = sum(u[i+1:]) for i = 0 .. len(u)-2."""
    return np.cumsum(u[::-1])[::-1][1:]


def apply(x, k, w):
    N = len(x)
    ns = n_start(N, k)
    v = np.asarray(w[ns:], dtype=float)  # knot coefficients, or empty when N <= k+1
    for m in range(ns - 1, -1, -1):
        h = x[m + 1:] - x[:N - m - 1]
        v = w[m] + np.concatenate(([0.0], np.cumsum(h * v)))
    return v


def adjoint(x, k, r):
    N = len(x)
    ns = n_start(N, k)
    g = np.empty(N)
    u = np.asarray(r, dtype=float)
    for m in range(ns):
        g[m] = u.sum()
        u = (x[m + 1:] - x[:N - m - 1]) * _rev_tail_sums(u)
    g[ns:] = u
    return g


def column(x, k, j):
    e = np.zeros(len(x))
    e[j] = 1.0
    return apply(x, k, e)


def norm_proxy(x, k, v):
    """Weighted column norms: exact for start columns, a cheap estimate for knot columns.

    Knot column j is a degree-k polynomial on its support i >= j+k+1 that grows to |A[N-1, j]|,
    so its weighted norm is about |A[N-1, j]| * sqrt(sum(v over support) / (2k+1)).
    """
    N = len(x)
    ns = n_start(N, k)
    p = np.empty(N)
    for m in range(ns):
        p[m] = np.sqrt(v @ column(x, k, m) ** 2)
    if N > ns:
        e = np.zeros(N)
        e[-1] = 1.0
        last_row = adjoint(x, k, e)[ns:]
        support_w = np.cumsum(v[::-1])[::-1][k + 1:]
        p[ns:] = np.abs(last_row) * np.sqrt(support_w / (2 * k + 1))
    return p
