import numpy as np
import pytest

from monols._preprocess import merge, prepare


def test_nan_samples_are_dropped_keeping_positions():
    p = prepare(np.array([1.0, np.nan, 3.0, 4.0]), None, None)
    assert p.valid.tolist() == [True, False, True, True]
    np.testing.assert_allclose(p.xu, [0.0, 2 / 3, 1.0])
    np.testing.assert_allclose(p.yu, [1.0, 3.0, 4.0])


def test_ties_are_merged_with_weighted_mean():
    p = prepare(np.array([1.0, 3.0, 5.0]), np.array([2.0, 2.0, 7.0]), np.array([1.0, 3.0, 1.0]))
    np.testing.assert_allclose(p.yu, [2.5, 5.0])
    np.testing.assert_allclose(p.wu, [4.0, 1.0])
    assert p.inverse.tolist() == [0, 0, 1]


def test_x_is_sorted_and_rescaled_to_unit_interval():
    p = prepare(np.array([1.0, 2.0, 3.0]), np.array([30.0, 10.0, 20.0]), None)
    np.testing.assert_allclose(p.xu, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(p.yu, [2.0, 3.0, 1.0])
    assert p.x_min == 10.0 and p.x_span == 20.0


def test_all_nan_gives_empty_grid():
    p = prepare(np.full(4, np.nan), None, None)
    assert len(p.xu) == 0 and not p.valid.any()


def test_single_sample_maps_to_zero():
    p = prepare(np.array([np.nan, 5.0]), None, None)
    np.testing.assert_allclose(p.xu, [0.0])


def test_constant_y_is_accepted():
    p = prepare(np.full(5, 2.0), None, None)
    np.testing.assert_allclose(p.yu, 2.0)


@pytest.mark.parametrize("w", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0]])
def test_weights_must_be_positive_and_finite(w):
    with pytest.raises(ValueError):
        prepare(np.array([1.0, 2.0, 3.0]), None, np.array(w))


def test_merge_recomputes_group_means_for_new_weights():
    yu, wu = merge(np.array([1.0, 3.0, 5.0]), np.array([1.0, 1.0, 2.0]), np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(yu, [2.0, 5.0])
    np.testing.assert_allclose(wu, [2.0, 2.0])
