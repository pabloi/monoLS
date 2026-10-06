# monoLS

Shape-constrained least-squares fitting for Python and MATLAB/Octave. You give it samples (x, y),
and it returns the closest curve that is

- **order 0**: monotone (isotonic regression),
- **order 1**: monotone and convex/concave,
- **order k**: additionally has divided differences of every order up to k+1 of constant sign
  (discrete *k-monotone* fits; for example, a decaying exponential at any order).

It is nonparametric: no functional form is assumed beyond the shape. Fits are piecewise
polynomials of degree k with data-chosen knots. The Python and MATLAB implementations are
twins: they follow the same [algorithm spec](spec/ALGORITHM.md) and are tested against the
same golden fixtures.

## Install

**Python** (≥ 3.9, needs numpy only):

```bash
pip install "git+https://github.com/pabloi/monoLS#subdirectory=python"
```

**MATLAB / Octave** (no toolboxes needed): add the `matlab` folder to the path.

```matlab
addpath('monoLS/matlab')
```

## Usage

```python
import monols

f = monols.fit(y, x, order=1, direction="decreasing")  # monotone + convex
f.fitted          # fitted values (NaN where y was NaN)
f.knots           # where the fit bends
f.predict(x_new)  # evaluate between/beyond the samples
```

```matlab
F = monols.fit(y, 'x', x, 'order', 1, 'direction', 'decreasing');
F.fitted, F.knots
yq = monols.predict(F, xq);
```

| option | values | default |
|---|---|---|
| `order` | 0, 1, 2, … | 0 |
| `direction` | `increasing`, `decreasing`, `auto` (best fit) | `auto` |
| `curvature` | `saturating` (levels off, like decaying exponentials), `accelerating`, `auto` | `saturating` |
| `loss` | `l2` (least squares, exact), `l1` (robust to outliers; approximate, solved by IRLS and typically within a fraction of a percent of the optimal L1 loss) | `l2` |
| `weights` | positive per-sample weights | all 1 |
| `boundary` | number of samples at the steep end where the highest-order difference is held at 0 (reduces boundary over-fitting) | 0 |

`x` may be irregularly spaced and may contain ties. A matrix `y` is fit column by column.
See `examples/` for complete scripts in both languages.

**v1 code** still runs: `z = monoLS(y, normP, derN, regN, oddSign, evenSign)` (MATLAB) is now a thin
wrapper around `monols.fit`. Only `normP` = 1 or 2 is supported. See [CHANGELOG](CHANGELOG.md).

## How it works

The v1 idea is kept: fitted values are z = A·w with w ≥ 0, so the fit is a non-negative least
squares (NNLS) problem. The columns of A are the start value, slope, …, and the "knots" of the
(k+1)-th divided difference. v2 builds A for any x spacing and **never forms it**: Aᵀr is computed
with k+1 cumulative sums in O(n·k), and only the columns in the solution (typically 5–50) are ever
built. Order 0 uses the pool-adjacent-violators algorithm (PAVA). Higher orders use a structured
Lawson–Hanson active-set method, and L1 uses iteratively reweighted least squares. Details:
[spec/ALGORITHM.md](spec/ALGORITHM.md).

## Speed

Time to fit n samples of a noisy saturating curve (Apple M5 Pro, single thread):

| n | order 0 | order 1 | order 2 | order 3 |
|---|---|---|---|---|
| **Python** 1,000 | 0.001 s | 0.005 s | 0.011 s | 0.004 s |
| 10,000 | 0.007 s | 0.107 s | 0.061 s | 0.037 s |
| 100,000 | 0.067 s | 2.84 s | 1.10 s | 0.69 s |
| **Octave 11** 1,000 | 0.019 s | 0.024 s | 0.023 s | 0.016 s |
| 10,000 | 0.115 s | 0.353 s | 0.157 s | 0.123 s |
| 100,000 | 1.23 s | 3.72 s | 1.53 s | 0.95 s |

For comparison, v1 (dense n×n matrix with `lsqnonneg`) took 0.73 s for order 0 at n = 3,000 in
Octave. At n = 100,000 its matrix alone would need 80 GB. Reproduce with `benchmarks/bench.py`
and `benchmarks/bench.m`.

## Tests

```bash
pip install -e "python[test]" && pytest python/tests
octave-cli --eval "cd matlab/tests; runAll"     # or the same command in MATLAB
```

CI runs both suites on Python 3.9/3.13, Octave and MATLAB. The expected values in
`tests/fixtures/cases.json` come from an independent dense reference solver
(`tests/fixtures/make_fixtures.py`).

## Related work

Isotonic regression (PAVA; e.g. scikit-learn's `IsotonicRegression`, R `isotone`), convex regression
(Hildreth 1954), and discrete k-monotone least squares (R package `pkmon`; Giguelay 2017) cover the
same family of estimators. The cone-generator (NNLS) formulation is also behind R's `coneproj`
(Meyer). monoLS offers that family in a single API for both Python and MATLAB, with irregular x,
weights, an L1 loss, and an O(n·k)-memory solver.
