"""Cluster bootstrap confidence intervals (Part 06B).

Customers' transactions are correlated, so CI's resample **customers** (clusters),
not transactions: resampling rows would understate uncertainty. Per-customer index
lists are precomputed once so each resample is a concatenation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cluster_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for ``metric_fn`` resampling whole clusters."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups)
    if not (len(y) == len(score) == len(groups)):
        raise ValueError("y, score and groups must have equal length")

    point = float(metric_fn(y, score))

    unique = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in unique]
    n_groups = len(unique)
    rng = np.random.default_rng(seed)

    stats = np.empty(n, dtype=float)
    for i in range(n):
        chosen = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        stats[i] = metric_fn(y[idx], score[idx])

    finite = stats[np.isfinite(stats)]
    if len(finite) < 2:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
