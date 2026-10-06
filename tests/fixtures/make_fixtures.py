"""Generate golden fixtures for monoLS (see spec/ALGORITHM.md).

This reference solver is deliberately independent of the fast implementations: it builds the
basis explicitly from closed-form Newton polynomials (checked against divided differences),
solves L2 with dense column-scaled NNLS (intercept as a +1/-1 column pair) and L1 with a linear
program. It is slow and only meant for small cases.

Requires numpy and scipy. Run: python make_fixtures.py  (writes cases.json next to this file)
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, nnls

CANDIDATES = [("increasing", "saturating"), ("increasing", "accelerating"),
              ("decreasing", "saturating"), ("decreasing", "accelerating")]


def divided_differences(x, z, m):
    d = np.asarray(z, float)
    for q in range(1, m + 1):
        d = (d[1:] - d[:-1]) / (x[q:] - x[:-q])
    return d


def basis(x, k, boundary=0):
    """Explicit A (N x ncols) and a mask of constrained columns, canonical coordinates."""
    N, K = len(x), k + 1
    cols, constrained = [], []
    for m in range(0, min(k, N - 1) + 1):
        c = np.ones(N)
        for l in range(m):
            c = c * (x - x[l])
        cols.append(c)
        constrained.append(m > 0)
    for j in range(max(0, N - K - boundary)):
        c = np.zeros(N)
        i = np.arange(j + K, N)
        c[i] = x[j + K] - x[j]
        for l in range(j + 1, j + k + 1):
            c[i] = c[i] * (x[i] - x[l])
        cols.append(c)
        constrained.append(True)
    A = np.column_stack(cols)
    # self-check: each knot column has a unit (k+1)-th divided difference at its own index only
    for j in range(max(0, N - K - boundary)):
        dd = divided_differences(x, A[:, min(k, N - 1) + 1 + j], K)
        e = np.zeros_like(dd)
        e[j] = 1.0
        assert np.allclose(dd, e, atol=1e-8 * max(1.0, np.abs(dd).max())), "knot column check failed"
    return A, np.array(constrained)


def preprocess(y, x, w):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    valid = ~(np.isnan(x) | np.isnan(y))
    xu, inv = np.unique(x[valid], return_inverse=True)
    wu = np.bincount(inv, weights=w[valid], minlength=len(xu))
    yu = np.bincount(inv, weights=w[valid] * y[valid], minlength=len(xu)) / np.where(wu > 0, wu, 1)
    span = xu[-1] - xu[0] if len(xu) > 1 else 1.0
    xs = (xu - xu[0]) / span if len(xu) > 1 else np.zeros(1)
    return valid, inv, xs, yu, wu, y, w


def canon(direction, curvature, k):
    if k == 0:
        return (direction == "decreasing"), False
    neg = (direction, curvature) in [("decreasing", "accelerating"), ("increasing", "saturating")]
    rev = curvature == "saturating"
    return neg, rev


def solve_l2(xs, yu, wu, k, boundary):
    A, cons = basis(xs, k, boundary)
    A = np.column_stack([A[:, :1], -A[:, :1], A[:, 1:]])  # free intercept as +1/-1 pair
    sw = np.sqrt(wu)
    B = sw[:, None] * A
    s = np.linalg.norm(B, axis=0)
    s[s == 0] = 1
    c, _ = nnls(B / s, sw * yu, maxiter=200 * B.shape[1])
    z = A @ (c / s)
    g = (B / s).T @ (sw * yu - (B / s) @ c)
    scale = max(np.linalg.norm(sw * (yu - np.average(yu, weights=wu))), 1e-300)
    assert g.max() <= 1e-9 * scale + 1e-14, f"reference KKT violated: {g.max()}"
    return z


def solve_l1(xs, inv, yv, wv, k, boundary):
    A, cons = basis(xs, k, boundary)
    Ar = A[inv]  # one row per valid sample
    n, p = Ar.shape
    # variables [c (p), t (n)]: min sum w t  s.t.  A c - y <= t,  -(A c - y) <= t
    cost = np.concatenate([np.zeros(p), wv])
    A_ub = np.block([[Ar, -np.eye(n)], [-Ar, -np.eye(n)]])
    b_ub = np.concatenate([yv, -yv])
    bounds = [(None, None) if not cons[i] else (0, None) for i in range(p)] + [(0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return A @ res.x[:p]


def reference_fit(y, x, w, order, direction, curvature, loss, boundary):
    valid, inv, xs, yu, wu, yall, wall = preprocess(y, x, w)
    fitted = np.full(len(yall), np.nan)
    if valid.sum() == 0:
        return fitted, direction, curvature, 0.0
    dirs = ["increasing", "decreasing"] if direction == "auto" else [direction]
    curvs = ["saturating"] if order == 0 else (["saturating", "accelerating"] if curvature == "auto" else [curvature])
    best = None
    for d, c in CANDIDATES:
        if d not in dirs or c not in curvs:
            continue
        neg, rev = canon(d, c, order)
        sgn = -1.0 if neg else 1.0
        if rev:
            xc, perm = 1.0 - xs[::-1], np.arange(len(xs))[::-1]
        else:
            xc, perm = xs, np.arange(len(xs))
        if loss == "l2":
            zc = solve_l2(xc, sgn * yu[perm], wu[perm], order, boundary)
        else:
            pos = np.empty_like(perm)
            pos[perm] = np.arange(len(perm))
            zc = solve_l1(xc, pos[inv], sgn * yall[valid], wall[valid], order, boundary)
        zu = np.empty_like(zc)
        zu[perm] = sgn * zc
        # the fit must lie in the cone in canonical coordinates
        for m in range(1, order + 2):
            if len(xc) > m:
                dd = divided_differences(xc, zc, m)
                assert dd.min() >= -1e-7 * max(1.0, np.abs(dd).max()), "fit outside the cone"
        r = zu[inv] - yall[valid]
        lv = float(np.sum(wall[valid] * r ** 2) if loss == "l2" else np.sum(wall[valid] * np.abs(r)))
        if best is None or lv < best[0] * (1 - 1e-9):
            best = (lv, zu, d, c)
    lv, zu, d, c = best
    fitted[valid] = zu[inv]
    return fitted, d, c, lv


def nan_to_none(a):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in a]


def cases():
    rng = np.random.default_rng(20261006)
    t = np.arange(40.0)
    sat = 3 - 2 * np.exp(-t / 12) - np.exp(-t / 3)  # increasing, saturating
    out = []

    def add(name, y, x=None, weights=None, **opt):
        o = dict(order=0, direction="auto", curvature="saturating", loss="l2", boundary=0)
        o.update(opt)
        out.append(dict(name=name, x=None if x is None else nan_to_none(x), y=nan_to_none(y),
                        weights=None if weights is None else nan_to_none(weights), options=o))

    noisy = sat + rng.normal(0, 0.15, t.size)
    for k in range(5):
        add(f"even_order{k}", noisy, order=k, direction="increasing")
    xi = np.sort(rng.uniform(0, 10, 35))
    yi = np.log1p(xi) + rng.normal(0, 0.1, xi.size)
    for k in (1, 2):
        add(f"irregular_order{k}", yi, x=xi, order=k, direction="increasing")
    xt = np.repeat(np.arange(12.0), 3)
    add("ties_order1", np.sqrt(xt) + rng.normal(0, 0.2, xt.size), x=xt, order=1, direction="increasing")
    yn = noisy.copy()
    yn[[3, 17, 30]] = np.nan
    add("nan_order1", yn, order=1)
    add("weights_order1", noisy, weights=rng.uniform(0.2, 3, t.size), order=1, direction="increasing")
    add("boundary_order1", noisy, order=1, direction="increasing", boundary=2)
    add("boundary_order0", -noisy[::-1], order=0, direction="increasing", boundary=3)
    base = 2 * np.exp(-t / 10) + rng.normal(0, 0.1, t.size)  # decreasing, saturating (convex)
    for d, c in CANDIDATES:
        y = {"decreasing": base, "increasing": -base}[d]
        y = y if c == "saturating" else y[::-1].copy() * -1
        add(f"shape_{d}_{c}", y, order=1, direction=d, curvature=c)
    add("auto_direction_order0", base, order=0)
    add("auto_curvature_order2", base[::-1].copy(), order=2, direction="increasing", curvature="auto")
    yo = sat + rng.normal(0, 0.05, t.size)
    yo[[5, 22]] += 3
    add("l1_outliers_order0", yo, order=0, direction="increasing", loss="l1")
    add("l1_outliers_order1", yo, order=1, direction="increasing", loss="l1")
    for n in (1, 2, 3):
        add(f"tiny_n{n}_order3", noisy[:n], order=3, direction="increasing")
    add("constant_order2", np.full(10, 4.2), order=2)
    add("all_nan", np.full(5, np.nan), order=1)
    return out


def main():
    data = cases()
    for c in data:
        y = np.array([np.nan if v is None else v for v in c["y"]])
        x = None if c["x"] is None else np.array(c["x"])
        w = None if c["weights"] is None else np.array(c["weights"])
        f, d, cv, lv = reference_fit(y, x, w, **c["options"])
        c["expected"] = dict(fitted=nan_to_none(f), direction=d, curvature=cv, loss_value=lv)
        print(f"{c['name']:36s} {d:10s} {cv:12s} loss={lv:.6g}")
    path = Path(__file__).with_name("cases.json")
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(data)} cases to {path}")


if __name__ == "__main__":
    main()
