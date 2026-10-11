"""Tests for the separability gate condition logic (Part 03C)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.config import BoostedConfig, SeparabilityGateConfig
from fraudlens.data_quality.gate import gate_markdown, run_separability_gate
from fraudlens.data_quality.signals import SIGNAL_COLUMNS


def _gate_cfg() -> SeparabilityGateConfig:
    return SeparabilityGateConfig(
        max_single_signal_roc_auc=0.85,
        max_single_signal_pr_auc=0.20,
        tree3_vs_boosted_min_pr_auc_gap=0.05,
        min_hard_negative_overlap_signals=2,
        boosted=BoostedConfig(
            n_estimators=20, max_depth=3, learning_rate=0.3, subsample=0.8, colsample_bytree=0.8
        ),
    )


def _toy(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tids = [f"t{i:05d}" for i in range(n)]
    signals = pd.DataFrame({"transaction_id": tids})
    for col in SIGNAL_COLUMNS:
        signals[col] = rng.random(n)
    labels = pd.DataFrame({"transaction_id": tids})
    # fraud driven purely by a single signal (perfectly separable) -> gate must FAIL
    labels["is_fraud"] = signals["amount_ratio"] > 0.9
    labels["hard_negative_type"] = pd.NA
    hn_mask = (labels["is_fraud"] == 0) & (rng.random(n) < 0.1)
    labels.loc[hn_mask, "hard_negative_type"] = "toy_hn"
    return signals, labels


def test_single_signal_gate_flags_perfect_separator() -> None:
    signals, labels = _toy()
    report = run_separability_gate(
        signals, labels, gate_cfg=_gate_cfg(), train_end_fraction=0.7, seed=0, profile="tiny"
    )
    assert set(report.conditions) == {
        "single_signal_roc_auc",
        "single_signal_pr_auc",
        "tree3_below_boosted",
        "hard_negative_overlap",
    }
    assert report.best_single_signal_roc_auc > 0.85
    assert report.conditions["single_signal_roc_auc"]["passed"] is False
    assert report.passed is False


def test_gate_report_structure_and_markdown() -> None:
    signals, labels = _toy()
    report = run_separability_gate(
        signals, labels, gate_cfg=_gate_cfg(), train_end_fraction=0.7, seed=0, profile="tiny"
    )
    assert report.n_rows == len(signals)
    assert report.n_fraud == int(labels["is_fraud"].sum())
    assert len(report.top_gain_importances) <= 3
    md = gate_markdown(report)
    assert "Separability Gate" in md
    assert "FAIL" in md
