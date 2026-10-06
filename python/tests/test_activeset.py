import time

import numpy as np
import pytest
from scipy.optimize import nnls

from monols._activeset import solve_canonical
from monols._basis import column


def reference(x, y, w, k):
    """Dense column-scaled NNLS with the intercept as a +1/-1 pair."""
    A = np.column_stack([column(x, k, j) for j in range(len(x))])
    A = np.column_stack([A[:, :1], -A[:, :1], A[:, 1:]])
    B = np.sqrt(w)[:, None] * A
    s = np.linalg.norm(B, axis=0)
    c, _ = nnls(B / s, np.sqrt(w) * y, maxiter=100 * B.shape[1])
    return A @ (c / s)


def data(n, seed, irregular=True):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n)) if irregular else np.linspace(0, 1, n)
    x = (x - x[0]) / (x[-1] - x[0])
    y = np.exp(3 * x) + rng.normal(0, 0.5, n)  # increasing, accelerating: canonical shape
    return x, y, rng.uniform(0.3, 3, n)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_matches_dense_reference(k):
    x, y, w = data(200, k)
    sol = solve_canonical(x, y, w, k)
    np.testing.assert_allclose(sol.z, reference(x, y, w, k), atol=1e-8 * np.ptp(y))
    assert sol.converged and sol.kkt <= 1e-10


@pytest.mark.parametrize("k", [0, 1, 2])
def test_order_zero_through_activeset_also_matches(k):
    x, y, w = data(80, 10 + k, irregular=False)
    np.testing.assert_allclose(solve_canonical(x, y, w, k).z, reference(x, y, w, k), atol=1e-8 * np.ptp(y))


def test_constant_data_returns_the_mean():
    sol = solve_canonical(np.linspace(0, 1, 10), np.full(10, 3.0), np.ones(10), 2)
    np.testing.assert_allclose(sol.z, 3.0)
    assert sol.converged


def test_data_in_the_cone_is_reproduced():
    x = np.linspace(0, 1, 30)
    y = 1 + x + x ** 3
    np.testing.assert_allclose(solve_canonical(x, y, np.ones(30), 2).z, y, atol=1e-10)


def test_large_n_is_fast_and_does_not_need_dense_matrix():
    x, y, w = data(20000, 99, irregular=False)
    t = time.perf_counter()
    sol = solve_canonical(x, y, w, 2)
    assert time.perf_counter() - t < 10
    assert sol.converged
