"""Public API: fit() and MonoFit (spec/ALGORITHM.md sections 1, 3, 6)."""
from dataclasses import dataclass, field

import numpy as np

from ._activeset import solve_canonical
from ._basis import n_start
from ._pava import pava
from ._preprocess import prepare

DIRECTIONS = ("increasing", "decreasing")
CURVATURES = ("saturating", "accelerating")
CANDIDATES = [(d, c) for d in DIRECTIONS for c in CURVATURES]


@dataclass
class MonoFit:
    fitted: np.ndarray
    x: np.ndarray
    knots: np.ndarray
    coef: np.ndarray
    order: int
    direction: str
    curvature: str
    loss: str
    loss_value: float
    converged: bool
    kkt_residual: float
    n_iter: int
    _xu: np.ndarray = field(repr=False, default=None)
    _zu: np.ndarray = field(repr=False, default=None)

    def predict(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        xu, zu = self._xu, self._zu
        if xu is None or xu.size == 0:
            return np.full(x_new.shape, np.nan)
        out = np.interp(x_new, xu, zu)
        if self.order >= 1 and xu.size >= 2:
            lo, hi = x_new < xu[0], x_new > xu[-1]
            out[lo] = zu[0] + (x_new[lo] - xu[0]) * (zu[1] - zu[0]) / (xu[1] - xu[0])
            out[hi] = zu[-1] + (x_new[hi] - xu[-1]) * (zu[-1] - zu[-2]) / (xu[-1] - xu[-2])
        return out


def _canonical_flags(direction, curvature, order):
    """(negate y, reverse x) mapping a shape onto the canonical cone (spec section 3)."""
    if order == 0:
        return direction == "decreasing", False
    negate = (direction, curvature) in (("decreasing", "accelerating"), ("increasing", "saturating"))
    return negate, curvature == "saturating"


def _solve_shape(prep, yu, wu, order, direction, curvature, boundary, tol, max_iter):
    """Weighted LS fit for one shape. Returns per-unique-x fit and solver details."""
    negate, reverse = _canonical_flags(direction, curvature, order)
    sign = -1.0 if negate else 1.0
    perm = np.arange(prep.xu.size)[::-1] if reverse else np.arange(prep.xu.size)
    xc = 1.0 - prep.xu[perm] if reverse else prep.xu
    yc, wc = sign * yu[perm], wu[perm]
    if order == 0 and boundary == 0:
        zc = pava(yc, wc)
        coef = np.concatenate(([zc[0]], np.diff(zc) / np.diff(xc)))
        converged, kkt, n_iter = True, 0.0, 0
    else:
        sol = solve_canonical(xc, yc, wc, order, excluded_tail=boundary, tol=tol, max_iter=max_iter)
        zc, coef, converged, kkt, n_iter = sol.z, sol.coef, sol.converged, sol.kkt, sol.n_iter
    zu = np.empty_like(zc)
    zu[perm] = sign * zc
    ns, K = n_start(xc.size, order), order + 1
    d = coef[ns:]
    knot_idx = np.flatnonzero(d > 1e-9 * np.max(d, initial=0.0))
    centers = np.array([xc[j:j + K + 1].mean() for j in knot_idx])
    centers = 1.0 - centers if reverse else centers
    knots = np.sort(prep.x_min + prep.x_span * centers)
    return zu, coef, knots, converged, kkt, n_iter


def _loss(prep, zu, loss):
    r = zu[prep.inverse] - prep.y[prep.valid]
    w = prep.w[prep.valid]
    return float(w @ r ** 2) if loss == "l2" else float(w @ np.abs(r))


def _validate(order, direction, curvature, loss, boundary):
    if not (isinstance(order, (int, np.integer)) and order >= 0):
        raise ValueError("order must be a non-negative integer")
    if direction not in DIRECTIONS + ("auto",):
        raise ValueError(f"direction must be one of {DIRECTIONS + ('auto',)}")
    if curvature not in CURVATURES + ("auto",):
        raise ValueError(f"curvature must be one of {CURVATURES + ('auto',)}")
    if loss not in ("l2", "l1"):
        raise ValueError("loss must be 'l2' or 'l1'")
    if not (isinstance(boundary, (int, np.integer)) and boundary >= 0):
        raise ValueError("boundary must be a non-negative integer")


def fit(y, x=None, *, order=0, direction="auto", curvature="saturating", loss="l2",
        weights=None, boundary=0, tol=1e-10, max_iter=None):
    """Shape-constrained least-squares (or L1) fit of y against x.

    order k constrains divided differences of orders 1..k+1 (0: monotone; 1: monotone and
    convex/concave; ...). direction is 'increasing', 'decreasing' or 'auto'; curvature is
    'saturating' (e.g. decaying exponentials), 'accelerating' or 'auto' and is ignored for
    order 0. A 2-D y (n x p) is fit column by column and returns a list of MonoFit.
    """
    _validate(order, direction, curvature, loss, boundary)
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        w2 = None if weights is None else np.asarray(weights, dtype=float)
        return [fit(y[:, i], x, order=order, direction=direction, curvature=curvature, loss=loss,
                    weights=None if w2 is None else (w2[:, i] if w2.ndim == 2 else w2),
                    boundary=boundary, tol=tol, max_iter=max_iter) for i in range(y.shape[1])]

    prep = prepare(y, x, weights)
    x_out = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float).ravel()
    if prep.xu.size == 0:
        return MonoFit(np.full(y.size, np.nan), x_out, np.zeros(0), np.zeros(0), order, direction,
                       curvature, loss, 0.0, True, 0.0, 0, prep.xu, prep.xu)

    if loss == "l1":
        from ._irls import solve_l1 as solver
    else:
        def solver(prep, d, c):
            return _solve_shape(prep, prep.yu, prep.wu, order, d, c, boundary, tol, max_iter)

    dirs = DIRECTIONS if direction == "auto" else (direction,)
    curvs = ("saturating",) if order == 0 else (CURVATURES if curvature == "auto" else (curvature,))
    best = None
    for d, c in CANDIDATES:
        if d not in dirs or c not in curvs:
            continue
        res = solver(prep, d, c) if loss == "l2" else solver(prep, order, d, c, boundary, tol, max_iter)
        lv = _loss(prep, res[0], loss)
        if best is None or lv < best[0] * (1 - 1e-9) - 1e-15:
            best = (lv, d, c, res)
    lv, d, c, (zu, coef, knots, converged, kkt, n_iter) = best
    fitted = np.full(y.size, np.nan)
    fitted[prep.valid] = zu[prep.inverse]
    xu_orig = prep.x_min + prep.x_span * prep.xu
    return MonoFit(fitted, x_out, knots, coef, order, d, c, loss, lv, converged, kkt, n_iter, xu_orig, zu)
