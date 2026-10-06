# Changelog

## 2.0.0 (unreleased, branch `v2`)

### New
- Python package `monols` with the same API and results as the MATLAB/Octave package `+monols`
  (`monols.fit`, `monols.predict`), both following `spec/ALGORITHM.md`.
- Irregular and tied x, per-sample weights, `predict` for new x, reported knots and diagnostics
  (`converged`, KKT residual, iterations).
- `boundary` option (formerly `regN`), now also available for order 0.
- Golden fixtures from an independent reference solver; CI on Python, Octave and MATLAB.

### Faster
- No n×n matrix: Aᵀr is computed with cumulative sums in O(n·k), and only active columns are built.
  Order 0 uses PAVA. n = 100,000 fits take about 0.1–4 s; v1 needed an 80 GB matrix at that size.

### Fixed (in v1 before the rewrite, then carried over)
- Orders ≥ 2 failed on Octave (the `quadprog`/`optimoptions` branch), and order ≥ 3 was refused
  because `quadprog` on A'A (which squares an already huge condition number) did not converge. All
  orders now use column-normalized NNLS, and the order limit is gone.
- Matrix input ignored `oddSign`/`evenSign`.
- Automatic direction detection picked a decreasing fit when the data contained NaN.

### Changed
- `monoLS(y, normP, derN, regN, oddSign, evenSign)` is a wrapper around `monols.fit`. Only `normP` = 1
  or 2 is supported (general p-norms were unreliable). `oddSign = 0` now picks the direction with
  the best fit instead of the sign of the covariance. `incLS` and `monoLS2` are removed.
- The intercept is a free variable; v1 shifted y to keep it non-negative.

### Note
- The v1 basis (`getMatrix`) was already correct for every order. Fits on evenly spaced data match
  v1 up to solver tolerance.
