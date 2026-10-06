import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from monols._pava import pava


@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn_with_weights(seed):
    rng = np.random.default_rng(seed)
    n = 300
    y = np.linspace(0, 2, n) + rng.normal(0, 1, n)
    w = rng.uniform(0.1, 5, n)
    expected = IsotonicRegression().fit_transform(np.arange(n), y, sample_weight=w)
    np.testing.assert_allclose(pava(y, w), expected, atol=1e-12)


def test_monotone_input_is_unchanged():
    y = np.array([0.0, 0.5, 0.5, 2.0, 7.0])
    np.testing.assert_array_equal(pava(y, np.ones(5)), y)


def test_single_and_empty():
    np.testing.assert_array_equal(pava(np.array([3.0]), np.array([2.0])), [3.0])
    assert pava(np.zeros(0), np.zeros(0)).size == 0
