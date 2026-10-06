"""L1 loss by iteratively reweighted least squares (spec/ALGORITHM.md section 5)."""
import numpy as np

from ._preprocess import merge


def solve_l1(prep, order, direction, curvature, boundary, tol, max_iter):
    from .api import _solve_shape  # local import: api imports this module lazily too

    yv, wv, inv, m = prep.y[prep.valid], prep.w[prep.valid], prep.inverse, prep.xu.size
    res = _solve_shape(prep, prep.yu, prep.wu, order, direction, curvature, boundary, tol, max_iter)
    med = np.median(yv)
    scale = np.max(np.abs(yv - med))
    if scale == 0:
        return res
    mad = np.median(np.abs(yv - med))
    eps = 1e-3 * (mad if mad > 0 else scale)
    r = res[0][inv] - yv
    obj = wv @ np.abs(r)
    best = (obj, res)
    irls_converged = False
    for _ in range(50):
        yu, wu = merge(yv, wv / np.maximum(np.abs(r), eps), inv, m)
        res = _solve_shape(prep, yu, wu, order, direction, curvature, boundary, tol, max_iter)
        r = res[0][inv] - yv
        new_obj = wv @ np.abs(r)
        # relative stall test, plus an absolute floor so an exact fit (obj = 0) counts as done
        done = abs(obj - new_obj) <= 1e-8 * obj + 1e-15 * scale * wv.sum()
        obj = new_obj
        if obj < best[0]:
            best = (obj, res)
        eps = max(eps / 2, 1e-7 * scale)  # a lower floor makes IRLS stall above the optimum
        if done:
            irls_converged = True
            break
    zu, coef, knots, converged, kkt, n_iter = best[1]
    return zu, coef, knots, converged and irls_converged, kkt, n_iter
