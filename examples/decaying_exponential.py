"""Fit a noisy two-timescale decay (e.g. a learning curve) with increasingly strict shapes.

Run: python examples/decaying_exponential.py   (plots if matplotlib is installed)
"""
import numpy as np

import monols

rng = np.random.default_rng(1)
t = np.arange(60.0)
truth = 1.5 * np.exp(-t / 4) + 0.8 * np.exp(-t / 25) + 0.2
y = truth + rng.normal(0, 0.08, t.size)
y[[7, 31]] += 0.8  # two outliers

fits = {
    "monotone (order 0)": monols.fit(y, t, order=0, direction="decreasing"),
    "+ convex (order 1)": monols.fit(y, t, order=1, direction="decreasing"),
    "+ convex, L1 loss": monols.fit(y, t, order=1, direction="decreasing", loss="l1"),
    "order 2, boundary=2": monols.fit(y, t, order=2, direction="decreasing", boundary=2),
}
for name, f in fits.items():
    rmse = np.sqrt(np.mean((f.fitted - truth) ** 2))
    print(f"{name:22s} error vs truth (RMSE) = {rmse:.4f}   knots at {np.round(f.knots, 1)}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit
plt.plot(t, y, ".", color="gray", label="data")
plt.plot(t, truth, "k--", label="truth")
for name, f in fits.items():
    plt.plot(t, f.fitted, label=name)
plt.legend()
plt.title("Shape-constrained fits of a decaying curve")
plt.show()
