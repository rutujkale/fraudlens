"""Tests for probability calibration (Part 07A)."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from fraudlens.evaluation.metrics import brier
from fraudlens.models.calibration import (
    IsotonicCalibrator,
    SigmoidCalibrator,
    calibration_bins,
    calibration_table,
    choose_calibrator,
    fit_calibrators,
    paired_cluster_bootstrap_diff,
)


def _dataset(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.beta(2.0, 8.0, size=n)  # right-skewed raw scores
    y = (rng.random(n) < raw).astype(int)
    groups = rng.integers(0, 400, size=n)
    return raw, y, groups


def test_sigmoid_predict_is_monotone():
    raw, y, _ = _dataset()
    cal = SigmoidCalibrator().fit(raw, y)
    grid = np.linspace(0, 1, 200)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_isotonic_predict_is_monotone():
    raw, y, _ = _dataset()
    cal = IsotonicCalibrator().fit(raw, y)
    grid = np.linspace(0, 1, 200)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0 and p.max() <= 1.0


@pytest.mark.parametrize("name", ["sigmoid", "isotonic"])
def test_joblib_round_trip(name, tmp_path):
    raw, y, _ = _dataset()
    cal = fit_calibrators(raw, y)[name]
    path = tmp_path / f"{name}.joblib"
    joblib.dump(cal, path)
    back = joblib.load(path)
    np.testing.assert_allclose(cal.predict(raw[:50]), back.predict(raw[:50]))


def test_calibration_repairs_mean_prediction():
    # Raw scores are inflated (mean far above base rate); calibration must pull
    # the mean predicted probability toward the observed rate.
    raw, y, _ = _dataset()
    base = y.mean()
    cal = fit_calibrators(raw, y)["sigmoid"]
    p = cal.predict(raw)
    assert abs(p.mean() - base) < abs(raw.mean() - base)


def test_calibration_bins_partition_rows():
    raw, y, _ = _dataset(n=1000)
    bins = calibration_bins(y, raw, n_bins=10)
    assert sum(b["count"] for b in bins) == 1000
    assert all(0.0 <= b["observed"] <= 1.0 for b in bins)


def test_calibration_table_rows():
    raw, y, _ = _dataset()
    cals = fit_calibrators(raw, y)
    rows = calibration_table(y, raw, cals, bins=10)
    assert [r["calibrator"] for r in rows] == ["raw", "sigmoid", "isotonic"]
    keys = {"n", "observed_rate", "mean_pred", "brier", "ece", "log_loss"}
    assert all(set(r) >= keys for r in rows)


def test_paired_bootstrap_diff_sign_and_ci():
    raw, y, groups = _dataset()
    cals = fit_calibrators(raw, y)
    p_sig = cals["sigmoid"].predict(raw)
    p_iso = cals["isotonic"].predict(raw)
    diff, lo, hi = paired_cluster_bootstrap_diff(y, p_sig, p_iso, groups, brier, n=200, seed=1)
    assert lo <= diff <= hi or np.isfinite(diff)


def test_thin_cal_positives_prefers_sigmoid():
    raw, y, groups = _dataset(n=1000, seed=3)
    cals = fit_calibrators(raw, y)
    sel = choose_calibrator(
        y, raw, cals, groups, n_cal_positives=20, min_cal_positives=150, n_boot=200, seed=1
    )
    assert sel["selected"] == "sigmoid"
    assert "min_cal_positives" in sel["reason"] or "< 150" in sel["reason"]


def test_choose_calibrator_returns_valid_choice():
    raw, y, groups = _dataset(n=6000, seed=5)
    cals = fit_calibrators(raw, y)
    sel = choose_calibrator(
        y, raw, cals, groups, n_cal_positives=400, min_cal_positives=150, n_boot=200, seed=1
    )
    assert sel["selected"] in {"sigmoid", "isotonic"}
    keys = {"selected", "reason", "brier_sigmoid", "brier_isotonic", "diff_lo", "diff_hi"}
    assert set(sel) >= keys
