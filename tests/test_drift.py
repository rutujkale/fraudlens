"""Tests for PSI-based drift detection (Part 08)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.monitoring.drift import psi, psi_flag, psi_table


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_identical_samples_psi_zero():
    x = _rng().normal(size=5000)
    assert psi(x, x) < 1e-9


def test_shifted_distribution_flags_shift():
    rng = _rng()
    ref = rng.normal(0.0, 1.0, size=5000)
    cur = rng.normal(3.0, 1.0, size=5000)
    value = psi(ref, cur)
    assert value > 0.25
    assert psi_flag(value) == "shift"


def test_psi_increases_with_shift_size():
    rng = _rng()
    ref = rng.normal(0.0, 1.0, size=4000)
    small = rng.normal(0.2, 1.0, size=4000)
    large = rng.normal(1.5, 1.0, size=4000)
    assert psi(ref, small) < psi(ref, large)


def test_missing_rate_shift_is_detected():
    rng = _rng()
    ref = rng.normal(size=2000)
    cur = ref.copy()
    cur[:1000] = np.nan
    assert psi(ref, cur) > 0.25


def test_eps_keeps_psi_finite_for_disjoint_support():
    rng = _rng()
    ref = rng.normal(0.0, 0.01, size=1000)
    cur = rng.normal(1.0, 0.01, size=1000)
    value = psi(ref, cur)
    assert np.isfinite(value)
    assert value > 0.25


def test_psi_table_sorted_with_flags():
    rng = _rng()
    ref = pd.DataFrame({"stable": rng.normal(size=3000), "shifted": rng.normal(size=3000)})
    cur = pd.DataFrame({"stable": rng.normal(size=3000), "shifted": rng.normal(4.0, size=3000)})
    table = psi_table(ref, cur, ["stable", "shifted"])
    assert list(table["feature"]) == ["shifted", "stable"]
    assert table.iloc[0]["flag"] == "shift"
    assert table.iloc[1]["flag"] == "stable"
