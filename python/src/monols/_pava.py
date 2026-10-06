"""Weighted pool-adjacent-violators algorithm (non-decreasing fit), O(n)."""
import numpy as np


def pava(y, w):
    n = len(y)
    level = np.empty(n)  # block means
    weight = np.empty(n)
    count = np.empty(n, dtype=int)
    b = -1
    for i in range(n):
        b += 1
        level[b], weight[b], count[b] = y[i], w[i], 1
        while b > 0 and level[b - 1] > level[b]:
            wsum = weight[b - 1] + weight[b]
            level[b - 1] = (weight[b - 1] * level[b - 1] + weight[b] * level[b]) / wsum
            weight[b - 1] = wsum
            count[b - 1] += count[b]
            b -= 1
    return np.repeat(level[:b + 1], count[:b + 1])
