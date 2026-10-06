import time

import numpy as np
import pytest

import monols

RNG = np.random.default_rng(3)


@pytest.mark.parametrize("order", range(5))
@pytest.mark.parametrize("curvature", ["saturating", "accelerating"])
def test_straight_line_is_reproduced(order, curvature):
    y = 2 + 0.5 * np.arange(30.0)
    f = monols.fit(y, order=order, direction="increasing", curvature=curvature)
    np.testing.assert_allclose(f.fitted, y, atol=1e-9)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_negating_y_mirrors_the_direction(order):
    y = np.log1p(np.arange(40.0)) + RNG.normal(0, 0.1, 40)
    up = monols.fit(y, order=order, direction="increasing", curvature="saturating")
    down = monols.fit(-y, order=order, direction="decreasing", curvature="saturating")
    np.testing.assert_allclose(down.fitted, -up.fitted, atol=1e-9)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_affine_change_of_x_does_not_change_the_fit(order):
    x = np.sort(RNG.uniform(0, 5, 30))
    y = np.sqrt(x) + RNG.normal(0, 0.1, 30)
    a = monols.fit(y, x, order=order)
    b = monols.fit(y, 3 * x + 7, order=order)
    np.testing.assert_allclose(a.fitted, b.fitted, atol=1e-9)


def test_fit_lies_in_the_requested_cone():
    x = np.sort(RNG.uniform(0, 1, 50))
    f = monols.fit(np.exp(-4 * x) + RNG.normal(0, 0.05, 50), x, order=2,
                   direction="decreasing", curvature="saturating")
    z = f.fitted
    d1 = np.diff(z) / np.diff(x)
    d2 = np.diff(d1) / (x[2:] - x[:-2])
    assert d1.max() <= 1e-9 and d2.min() >= -1e-6


def test_constant_y_gives_constant_fit():
    f = monols.fit(np.full(12, 4.2), order=2)
    np.testing.assert_allclose(f.fitted, 4.2)
    assert f.converged


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_inputs_at_high_order(n):
    y = np.array([1.0, 3.0, 2.0])[:n]
    f = monols.fit(y, order=3)
    assert np.all(np.isfinite(f.fitted)) and f.converged
    if n == 1:
        assert f.fitted[0] == 1.0


def test_all_nan_input():
    f = monols.fit(np.full(5, np.nan), order=1)
    assert np.all(np.isnan(f.fitted)) and f.loss_value == 0.0 and f.direction == "auto"
    assert np.all(np.isnan(f.predict([0.0, 1.0])))


def test_large_n_runs_quickly():
    y = np.exp(np.linspace(0, 3, 20000)) + RNG.normal(0, 0.5, 20000)
    t = time.perf_counter()
    f = monols.fit(y, order=2, direction="increasing", curvature="accelerating")
    assert time.perf_counter() - t < 10 and f.converged


def test_knots_are_reported_in_original_x():
    x = np.arange(11.0)
    f = monols.fit(np.maximum(0, x - 5), x, order=1, direction="increasing", curvature="accelerating")
    np.testing.assert_allclose(f.knots, [5.0])


def test_predict_interpolates_and_extrapolates():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    f0 = monols.fit(np.array([0.0, 1.0, 2.0, 3.0]), x, order=0, direction="increasing")
    np.testing.assert_allclose(f0.predict([-1.0, 1.5, 9.0]), [0.0, 1.5, 3.0])
    f1 = monols.fit(np.array([0.0, 1.0, 2.0, 3.0]), x, order=1, direction="increasing")
    np.testing.assert_allclose(f1.predict([-1.0, 1.5, 5.0]), [-1.0, 1.5, 5.0])
    single = monols.fit(np.array([np.nan, 2.0]), order=1)
    np.testing.assert_allclose(single.predict([0.0, 10.0]), [2.0, 2.0])


def test_matrix_input_is_fit_column_by_column():
    Y = np.column_stack([np.arange(10.0), -np.arange(10.0)])
    fits = monols.fit(Y, order=0)
    assert len(fits) == 2
    np.testing.assert_allclose(fits[1].fitted, monols.fit(Y[:, 1], order=0).fitted)


@pytest.mark.parametrize("bad", [dict(order=-1), dict(direction="up"), dict(curvature="flat"),
                                 dict(loss="l3"), dict(boundary=-2)])
def test_invalid_options_raise(bad):
    with pytest.raises(ValueError):
        monols.fit(np.arange(5.0), **bad)


@pytest.mark.parametrize("order,b", [(0, 3), (1, 2), (2, 4)])
def test_boundary_zeroes_the_last_highest_order_differences(order, b):
    x = np.linspace(0, 1, 40)
    y = np.exp(3 * x) + np.random.default_rng(order).normal(0, 0.3, 40)  # canonical: steep at the end
    y[-1] += 3  # end spike, so the unconstrained fit has a knot near the boundary
    z = monols.fit(y, x, order=order, direction="increasing", curvature="accelerating", boundary=b).fitted
    d = z
    for q in range(1, order + 2):
        d = (d[1:] - d[:-1]) / (x[q:] - x[:-q])
    np.testing.assert_allclose(d[-b:], 0, atol=1e-8 * np.abs(d).max())
    free = monols.fit(y, x, order=order, direction="increasing", curvature="accelerating").fitted
    assert not np.allclose(z, free)  # the option actually changed the fit
