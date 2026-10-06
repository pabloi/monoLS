"""Structured Lawson-Hanson NNLS over the implicit basis (spec/ALGORITHM.md section 5)."""
from dataclasses import dataclass

import numpy as np

from ._basis import adjoint, apply, column, n_start, norm_proxy


@dataclass
class Solution:
    z: np.ndarray          # fitted values (canonical coordinates)
    coef: np.ndarray       # coefficients in the canonical basis
    active: np.ndarray     # indices of non-zero coefficients (intercept included)
    converged: bool
    kkt: float             # scaled KKT residual
    n_iter: int


def solve_canonical(x, y, w, k, *, excluded_tail=0, tol=1e-10, max_iter=None):
    N = len(x)
    ns = n_start(N, k)
    max_iter = 10 * N if max_iter is None else max_iter
    ybar = np.average(y, weights=w)
    coef = np.zeros(N)
    coef[0] = ybar
    sigma = np.sqrt(w @ (y - ybar) ** 2)
    if sigma <= 1e-13 * np.max(np.abs(y)) * np.sqrt(w.sum()):  # constant up to rounding
        return Solution(np.full(N, ybar), coef, np.array([0]), True, 0.0, 0)

    allowed = np.ones(N, dtype=bool)
    allowed[0] = False  # the intercept is always active
    if excluded_tail > 0:
        allowed[max(ns, N - excluded_tail):] = False
    p = norm_proxy(x, k, w)
    allowed &= p > 0
    sw = np.sqrt(w)
    cache = {}

    def unit_column(j):
        if j not in cache:
            c = column(x, k, j)
            cache[j] = (c, np.linalg.norm(sw * c))
        return cache[j]

    def least_squares(P):
        cols = [unit_column(j) for j in P]
        B = np.column_stack([sw * c / n for c, n in cols])
        u = np.linalg.lstsq(B, sw * y, rcond=None)[0]
        return u / np.array([n for _, n in cols])

    P = [0]
    z = np.full(N, ybar)
    banned = set()
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        score = adjoint(x, k, w * (y - z)) / np.where(p > 0, p, 1)
        candidates = allowed.copy()
        candidates[P] = False
        if banned:
            candidates[list(banned)] = False
        if not candidates.any():
            break
        j = int(np.argmax(np.where(candidates, score, -np.inf)))
        if score[j] <= tol * sigma:
            break
        P.append(j)
        while True:
            u = least_squares(P)
            current = coef[P]
            bad = [i for i in range(1, len(P)) if u[i] <= 0]
            if not bad:
                coef[P] = u
                break
            ratios = {i: current[i] / (current[i] - u[i]) for i in bad}
            blocking = min(ratios, key=ratios.get)
            coef[P] = current + ratios[blocking] * (u - current)
            coef[P[blocking]] = 0.0  # exact zero: rounding can leave ~1e-16 and cycle forever
            floor = 1e-14 * np.max(np.abs(coef[P[1:]]), initial=0.0)
            keep = [P[0]] + [q for q in P[1:] if coef[q] > floor]
            for q in set(P) - set(keep):
                coef[q] = 0.0
            P = keep
        if j in P:
            banned.clear()
        else:
            banned.add(j)  # entered and left immediately: numerically degenerate, skip it for now
        z = apply(x, k, coef)

    score = adjoint(x, k, w * (y - z)) / np.where(p > 0, p, 1)
    outside = allowed.copy()
    outside[P] = False
    inside = np.zeros(N, dtype=bool)
    inside[P[1:]] = True
    kkt = max(np.max(np.maximum(score[outside], 0), initial=0.0),
              np.max(np.abs(score[inside]), initial=0.0)) / sigma
    return Solution(z, coef, np.array(sorted(P)), bool(kkt <= tol), float(kkt), n_iter)
