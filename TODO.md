# TODO

## S-shaped (sigmoid / psychometric) fits — idea, not started

Target: monotone curves whose slope rises then falls (convex left of an inflection c, concave
right of it), e.g. logistic psychometric functions. Order 0 ignores this shape; order 1 cannot
express it because the curvature changes sign.

- **Fixed c is still NNLS.** A non-negative slope that peaks at c is a non-negative sum of plateau
  indicators 1[a ≤ i ≤ b] with a ≤ c ≤ b. The generators are their integrals (flat, then a
  linear rise, then flat): z = intercept + Σ w_ab·clipramp_ab, w ≥ 0. No extra constraints are needed.
- **Pricing stays O(n).** g(a,b) = g_ramp(a) − g_ramp(b+1) uses the order-1 adjoint, and the best
  pair splits into max over a ≤ c and min over b ≥ c. The O(n²) generators are never formed.
- **Unknown c:** profile the loss over c, with warm starts and a coarse-to-fine grid. The profile
  is not convex in c, so a full scan is the safe default.
- **Psychometric specifics:**
  - binomial loss (weights = trials as a first step; an exact binomial fit via IRLS);
  - guess/lapse bounds [γ, 1−λ] (two endpoint constraints);
  - log-concavity is a further option, but not convex in least squares.
- **Before building:** do a literature check on S-shaped / convex–concave regression with an
  unknown inflection (the estimator itself is likely known). Compare on simulated psychometric data
  against isotonic, centered isotonic (Oron & Flournoy 2017), local-linear (Zychaluk & Foster 2009)
  and parametric logistic fits.

## Deferred from the v2 review

- Incremental QR updates in the active-set loop (many-knot fits are slow at large n).
- Exact L1 (polish or certify after IRLS; currently up to ~0.3% above the optimal loss).
- Reject ±Inf in x/y; validate weights only on the samples that are kept; document `tol`.
- Choose a LICENSE.
