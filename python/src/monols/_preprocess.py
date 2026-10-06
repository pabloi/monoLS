"""Preprocessing (spec/ALGORITHM.md section 2): drop NaN, sort, merge ties, rescale x."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Prepared:
    xu: np.ndarray       # unique x rescaled to [0, 1], increasing
    yu: np.ndarray       # weighted mean of y per unique x
    wu: np.ndarray       # summed weights per unique x
    inverse: np.ndarray  # for each valid sample, its index into xu
    valid: np.ndarray    # bool mask over the input samples (x and y not NaN)
    x_min: float         # original x of xu == 0
    x_span: float        # original-x length of the unit interval
    y: np.ndarray        # input y (float)
    w: np.ndarray        # input weights (float)


def merge(y_valid, w_valid, inverse, m):
    """Weighted mean of y and summed weight per group (groups given by inverse, m groups)."""
    wu = np.bincount(inverse, weights=w_valid, minlength=m)
    yu = np.bincount(inverse, weights=w_valid * y_valid, minlength=m) / wu
    return yu, wu


def prepare(y, x=None, weights=None):
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    x = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float).ravel()
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if x.size != n or w.size != n:
        raise ValueError("x and weights must have the same length as y")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and > 0")
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, inverse = np.unique(x[valid], return_inverse=True)
    inverse = inverse.ravel()
    if xs.size == 0:
        empty = np.zeros(0)
        return Prepared(empty, empty, empty, inverse, valid, 0.0, 1.0, y, w)
    yu, wu = merge(y[valid], w[valid], inverse, xs.size)
    x_min = float(xs[0])
    x_span = float(xs[-1] - xs[0]) if xs.size > 1 else 1.0
    return Prepared((xs - x_min) / x_span, yu, wu, inverse, valid, x_min, x_span, y, w)
