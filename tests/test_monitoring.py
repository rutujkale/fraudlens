"""Part 15A tests: per-period monitoring scope, label maturity and the retraining trigger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.monitoring import metrics as M

T_VAL_END = pd.Timestamp("2026-04-07 00:00:00")
T_TRAIN_END = pd.Timestamp("2026-03-13 23:59:00")


def _frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(len(times))],
            "timestamp": pd.to_datetime(times),
        }
    )


def test_monitoring_window_excludes_rows_at_or_after_val_end():
    df = _frame(
        [
            "2026-03-01 00:00:00",
            "2026-04-06 23:59:00",
            "2026-04-07 00:00:00",  # exactly the validation end: must be dropped
            "2026-04-20 00:00:00",
        ]
    )
    kept = M.monitoring_window(df, T_VAL_END)
    assert len(kept) == 2
    assert kept["timestamp"].max() < T_VAL_END
    assert T_VAL_END not in set(kept["timestamp"])


def test_period_of():
    assert M.period_of("2026-03-15 10:00") == "2026-03"


def test_sample_label_distinguishes_in_sample_from_validation():
    in_sample = _frame(["2026-02-01 00:00:00", "2026-02-28 00:00:00"])
    mixed = _frame(["2026-03-01 00:00:00", "2026-03-20 00:00:00"])
    validation = _frame(["2026-04-01 00:00:00", "2026-04-06 00:00:00"])
    assert M._sample_label(in_sample, T_TRAIN_END) == "in_sample"
    assert M._sample_label(mixed, T_TRAIN_END) == "mixed"
    assert M._sample_label(validation, T_TRAIN_END) == "validation"


def test_psi_by_period_returns_rows_per_period_and_feature():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f": rng.normal(size=2000)})
    cur = ref.assign(period="2026-01").copy()
    table = M.psi_by_period(ref, cur, ["f"])
    assert list(table.columns) == ["period", "feature", "psi", "flag", "n"]
    assert table.iloc[0]["psi"] < 1e-6  # identical samples


def test_label_maturity_excludes_immature_labels():
    period_df = pd.DataFrame(
        {
            "transaction_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2026-03-10", "2026-03-20", "2026-03-25"]),
            "score": [0.9, 0.1, 0.2],
        }
    )
    labels = pd.DataFrame(
        {
            "transaction_id": ["a", "b", "c"],
            "is_fraud": [1, 0, 0],
            "label_available_at": pd.to_datetime(["2026-03-15", "2026-03-21", "2026-04-10"]),
        }
    )
    row = M._period_metrics(period_df, labels, threshold=0.5)
    assert row["n_rows"] == 3
    assert row["n_matured"] == 2  # c's label matures after the period ends
    assert row["observed_rate"] == 1 / 3
    assert row["prevalence_matured"] == 0.5
    assert row["pr_auc"] == 1.0
    assert row["recall_at_frozen"] == 1.0


def test_retraining_trigger_stable_series():
    verdict = M.retraining_trigger([0.01, 0.02, 0.03], [0.9, 0.92, 0.95])
    assert verdict["retrain"] is False
    assert verdict["reasons"] == []


def test_retraining_trigger_psi_streak():
    verdict = M.retraining_trigger([0.01, 0.3, 0.4], [0.9, 0.9, 0.9])
    assert verdict["retrain"] is True
    assert "consecutive periods" in verdict["reasons"][0]


def test_retraining_trigger_single_spike_does_not_trip():
    verdict = M.retraining_trigger([0.01, 0.3, 0.02], [0.9, 0.9, 0.9])
    assert verdict["retrain"] is False


def test_retraining_trigger_recall_floor():
    verdict = M.retraining_trigger([0.01, 0.02], [0.9, 0.1])
    assert verdict["retrain"] is True
    assert any("recall" in reason for reason in verdict["reasons"])
