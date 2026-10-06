"""Timing of monols.fit for growing n and order (prints a markdown table).

Usage: python benchmarks/bench.py [max_n]   (default max_n = 100000)
"""
import sys
import time

import numpy as np

import monols


def main(max_n=100_000):
    rng = np.random.default_rng(0)
    print("| n | order 0 | order 1 | order 2 | order 3 |")
    print("|---|---|---|---|---|")
    for n in (1_000, 10_000, 100_000):
        if n > max_n:
            break
        x = np.linspace(0, 1, n)
        y = 1 - np.exp(-5 * x) + rng.normal(0, 0.1, n)  # increasing, saturating
        cells = []
        for k in range(4):
            t = time.perf_counter()
            f = monols.fit(y, x, order=k, direction="increasing", curvature="saturating")
            dt = time.perf_counter() - t
            cells.append(f"{dt:.3f} s" + ("" if f.converged else " (!)"))
        print(f"| {n:,} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
