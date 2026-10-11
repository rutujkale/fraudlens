"""Cluster bootstrap tests (Part 06B)."""

from __future__ import annotations

import numpy as np

from fraudlens.evaluation.bootstrap import cluster_bootstrap
from fraudlens.evaluation.metrics import pr_auc


def _synthetic(n_groups: int, per_group: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, score, groups = [], [], []
    for g in range(n_groups):
        base = rng.normal(0.0, 1.0)
        for _ in range(per_group):
            s = base + rng.normal(0.0, 0.3)
            score.append(s)
            y.append(int(s + rng.normal(0, 1) > 1.0))
            groups.append(g)
    return np.array(y), np.array(score), np.array(groups)


def test_ci_contains_point_estimate():
    y, score, groups = _synthetic(300)
    point, lo, hi = cluster_bootstrap(y, score, groups, pr_auc, n=200, seed=1)
    assert lo <= point <= hi


def test_ci_shrinks_with_more_data():
    y_small, s_small, g_small = _synthetic(150, seed=2)
    y_big, s_big, g_big = _synthetic(2000, seed=2)
    _, lo_s, hi_s = cluster_bootstrap(y_small, s_small, g_small, pr_auc, n=200, seed=3)
    _, lo_b, hi_b = cluster_bootstrap(y_big, s_big, g_big, pr_auc, n=200, seed=3)
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        cluster_bootstrap(np.array([0, 1]), np.array([0.1]), np.array([0, 1]), pr_auc)
