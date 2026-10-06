"""Monotone and convex fits to a noisy line with outliers, on an irregular x grid.

Run: python examples/line_with_outliers.py
"""
import numpy as np

import monols

rng = np.random.default_rng(2)
x = np.sort(rng.uniform(-1, 1, 80))
truth = 0.7 * x + 0.3
y = truth + rng.normal(0, 0.1, x.size)
y[rng.choice(x.size, 5, replace=False)] = rng.normal(0, 1, 5)  # outliers

for loss in ("l2", "l1"):
    for order in (0, 1):
        f = monols.fit(y, x, order=order, loss=loss)
        rmse = np.sqrt(np.mean((f.fitted - truth) ** 2))
        print(f"loss={loss} order={order}: chose {f.direction}/{f.curvature}, RMSE vs truth {rmse:.3f}")
print("prediction at x = 0.5 (L1, order 1):", monols.fit(y, x, order=1, loss="l1").predict([0.5]))
