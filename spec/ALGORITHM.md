# monoLS algorithm (normative)

Both implementations (Python `monols`, MATLAB/Octave `+monols`) follow this document. Indices are 0-based.
Design rationale is in `docs/design/2026-10-06-v2-design.md`.

## 1. Inputs and defaults

`y` (n samples), optional `x` (default 0..n−1), `weights` (default 1), `order` k ≥ 0 (default 0),
`direction` ∈ {increasing, decreasing, auto} (default auto), `curvature` ∈ {saturating, accelerating, auto}
(default saturating; ignored when k = 0), `loss` ∈ {l2, l1} (default l2), `boundary` m ≥ 0 (default 0),
`tol` (default 1e-10), `max_iter` (default 10·n).

## 2. Preprocessing

1. Weights must be finite and > 0 (otherwise raise an error). Drop samples where x or y is NaN.
2. Sort by x. Merge tied x into one sample: weight = sum of weights, y = weighted mean.
3. Rescale x affinely to [0, 1] (a single unique sample gets x = 0).

The result is the unique grid x_0 < … < x_{N−1}, with values ȳ and weights v.

## 3. Canonical form

| direction  | curvature    | y transform | x transform |
|------------|--------------|-------------|-------------|
| increasing | accelerating | y           | x           |
| decreasing | accelerating | −y          | x           |
| increasing | saturating   | −y          | reversed    |
| decreasing | saturating   | y           | reversed    |

For k = 0, curvature is ignored: increasing → y, decreasing → −y, x not reversed. "Reversed" means
x'_i = 1 − x_{N−1−i}, with y and v re-indexed accordingly.

In canonical form the fit z satisfies δ^m z ≥ 0 for m = 1..k+1, where δ^m are divided differences:
δ^0 z = z and δ^m z_i = (δ^{m−1} z_{i+1} − δ^{m−1} z_i) / (x_{i+m} − x_i).

## 4. Basis: z = A·w

Let K = k+1 and h^{(m)}_i = x_{i+m} − x_i.

Coefficients (N in total):
- s_0: intercept, free;
- s_m, m = 1..min(k, N−1): start values δ^m z_0, constrained ≥ 0;
- d_j, j = 0..N−K−1 (absent if N ≤ K): knot coefficients δ^K z_j, constrained ≥ 0.

**Apply** (z = A·w), O(N·k). Start from v^{(K)} = d. For m = k, k−1, …, 0 (only for m ≤ N−1), build
v^{(m)} of length N−m:

    v^{(m)}_0 = s_m,   v^{(m)}_{i+1} = v^{(m)}_i + h^{(m+1)}_i · v^{(m+1)}_i,   i = 0..N−m−2.

Then z = v^{(0)}. If N ≤ K, take v^{(m)} = 0 for the m with no room left.

**Adjoint** (g = Aᵀ·r), O(N·k). u^{(0)} = r. For m = 0..min(k, N−1):

    g[s_m] = Σ_i u^{(m)}_i,
    u^{(m+1)}_i = h^{(m+1)}_i · Σ_{l ≥ i+1} u^{(m)}_l,   i = 0..N−m−2.

Then g[d] = u^{(K)}.

**Columns** are apply() of a unit coefficient. Closed forms, used by the reference solver only:
- s_m column: the Newton polynomial Π_{l<m}(x_i − x_l);
- d_j column: (x_{j+K} − x_j) · Π_{l=j+1}^{j+k}(x_i − x_l) for i ≥ j+K, and 0 otherwise.

**Knots** are the d_j with d_j > 1e-9·max(d) (smaller values are rounding noise). The **location** of d_j is the mean of x_j … x_{j+K} in canonical coordinates, mapped back to original x.

**boundary = b**: drop the last b knot coefficients (d_j with j ≥ N−K−b), so the (k+1)-th divided
difference is zero on the last b stencils.

## 5. Solvers (canonical, weighted least squares with weights v)

**k = 0, no boundary**: weighted PAVA on ȳ, giving the non-decreasing fit.

**k ≥ 1, or boundary > 0**: Lawson–Hanson NNLS over A with these specifics:
- The objective is Σ v_i (z_i − ȳ_i)²; W = diag(v).
- Scale σ = sqrt(Σ v_i (ȳ_i − ȳ_w)²) with ȳ_w the weighted mean. If σ ≤ 1e-13·max|ȳ|·sqrt(Σv) (constant data up to rounding), return z = ȳ_w (converged).
- Column scales p: exact weighted norms ‖√v ⊙ a‖ for the s columns. For d_j:
  |A[N−1, d_j]| · sqrt(Σ_{i ≥ j+K} v_i / (2k+1)), where row N−1 of A is adjoint(e_{N−1}).
- Start with the active set P = {s_0} and z = ȳ_w.
- Outer loop: g = Aᵀ W (ȳ − z); score_j = g_j / p_j for allowed j ∉ P. Stop if max score ≤ tol·σ.
  Otherwise add the argmax to P.
- Inner loop: solve the unconstrained weighted LS on the columns in P (built with apply, normalized to
  unit weighted norm, `lstsq`). If every constrained coefficient is > 0, accept it. Otherwise step from
  the current w toward the new solution by α = min over violating q of w_q/(w_q − u_q), drop the
  coefficients that reach ≤ 0, and repeat. If the column just added is dropped immediately, exclude it for
  the rest of this outer iteration.
- Report `kkt_residual` = max(max_{j∉P} score_j⁺, max_{j∈P, constrained} |score_j|) / σ,
  `converged` = (kkt_residual ≤ tol), and `n_iter` = number of outer iterations. Stop at max_iter.

**L1 loss**: IRLS over the original (unmerged) valid samples. ε₀ = 1e-3·MAD(y) (if MAD = 0, use
1e-3·max(|y − median(y)|), and if that is 0 too, return the median). Iterate: per-sample weights
w_i / max(|r_i|, ε), then merge ties (§2.2) and run the L2 solver. Set ε ← max(ε/2, 1e-7·scale), where
scale = max(|y − median(y)|). Stop when the change in Σ w_i|r_i| is ≤ 1e-8·Σ w_i|r_i| + 1e-15·scale·Σw_i,
or after 50 iterations, and return the iterate with the lowest Σ w_i|r_i|.
**The L1 solution is approximate.** IRLS converges only linearly, and on some problems it stops
several tenths of a percent above the optimal L1 loss. For L1, `converged` means the iterations
settled (and the inner L2 solves converged); it does not certify L1 optimality. The initial r comes from the L2 fit.
(A lower ε floor makes the weights span ~1e9 and IRLS stalls above the optimum.)

## 6. Output

Undo the x reversal and the y negation, assign each merged group's value to its samples, and put NaN at
the dropped samples. `loss_value` is Σ w_i (z_i − y_i)² (l2) or Σ w_i |z_i − y_i| (l1) over valid samples.

With no valid samples, the output is all NaN, `loss_value` = 0, and direction/curvature are
returned as requested.

**Auto choices**: candidates in the order (increasing, saturating), (increasing, accelerating),
(decreasing, saturating), (decreasing, accelerating), restricted to the requested ones (for k = 0,
direction only). Pick the lowest `loss_value`. Values within a relative 1e-9 of the minimum count as ties, and the
first tied candidate wins (this keeps both implementations deterministic).

**predict(x_new)**: linear interpolation over (unique x, fitted values). Outside the data range: constant
for k = 0; for k ≥ 1, extend the end segment linearly (a single sample gives a constant).
