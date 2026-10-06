import numpy as np
import pytest

from monols._basis import adjoint, apply, column, n_start, norm_proxy

RNG = np.random.default_rng(7)


def grid(n, irregular=True):
    x = np.sort(RNG.uniform(0, 1, n)) if irregular else np.linspace(0, 1, n)
    return (x - x[0]) / (x[-1] - x[0])


def dense(x, k):
    return np.column_stack([column(x, k, j) for j in range(len(x))])


def divdiff(x, z, m):
    d = z.copy()
    for q in range(1, m + 1):
        d = (d[1:] - d[:-1]) / (x[q:] - x[:-q])
    return d


def legacy_get_matrix(n, k):
    """Faithful port of v1 incLS.m getMatrix: A(:,i+1:end) is cumsummed for i = 1..k."""
    A = np.tril(np.ones((n, n)))
    for i in range(1, k + 1):
        A[:, i:] = np.fliplr(np.cumsum(np.fliplr(A[:, i:]), axis=1))
    return A


@pytest.mark.parametrize("k", range(5))
def test_apply_matches_dense_and_adjoint_is_its_transpose(k):
    x = grid(25)
    A = dense(x, k)
    w, r = RNG.normal(size=25), RNG.normal(size=25)
    np.testing.assert_allclose(apply(x, k, w), A @ w, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(adjoint(x, k, r), A.T @ r, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("k", range(5))
def test_columns_match_closed_form_newton_polynomials(k):
    x = grid(20)
    N, K = len(x), k + 1
    for m in range(n_start(N, k)):
        expected = np.prod([x - x[l] for l in range(m)], axis=0) if m else np.ones(N)
        np.testing.assert_allclose(column(x, k, m), expected, atol=1e-13)
    for j in range(N - K):
        expected = np.zeros(N)
        i = np.arange(j + K, N)
        expected[i] = (x[j + K] - x[j]) * np.prod([x[i] - x[l] for l in range(j + 1, j + K)], axis=0)
        np.testing.assert_allclose(column(x, k, n_start(N, k) + j), expected, atol=1e-13)


@pytest.mark.parametrize("k", range(4))
def test_knot_columns_have_a_single_unit_divided_difference(k):
    x = grid(15)
    s = n_start(15, k)
    for j in range(15 - k - 1):
        e = np.zeros(15 - k - 1)
        e[j] = 1
        np.testing.assert_allclose(divdiff(x, column(x, k, s + j), k + 1), e, atol=1e-8)


@pytest.mark.parametrize("k", range(4))
def test_even_grid_columns_equal_v1_matrix_up_to_scaling(k):
    n = 12
    A, L = dense(np.linspace(0, 1, n), k), legacy_get_matrix(n, k)
    for j in range(n):
        ratio = A[:, j][L[:, j] != 0] / L[:, j][L[:, j] != 0]
        assert np.all(ratio > 0) and np.ptp(ratio) < 1e-10 * ratio.max()
        assert np.all(A[:, j][L[:, j] == 0] == 0)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_grids_have_n_columns(n):
    x = grid(n) if n > 1 else np.zeros(1)
    assert dense(x, 3).shape == (n, n)
    assert np.linalg.matrix_rank(dense(x, 3)) == n


@pytest.mark.parametrize("k", range(4))
def test_norm_proxy_is_exact_for_start_columns_and_close_for_knots(k):
    x, v = grid(60), RNG.uniform(0.5, 2, 60)
    A = dense(x, k)
    exact = np.sqrt(v @ A ** 2)
    p = norm_proxy(x, k, v)
    s = n_start(60, k)
    np.testing.assert_allclose(p[:s], exact[:s], rtol=1e-12)
    ratio = p[s:] / exact[s:]
    assert ratio.min() > 0.2 and ratio.max() < 5
