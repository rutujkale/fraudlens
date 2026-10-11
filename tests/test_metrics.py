"""Hand-computed metric tests (Part 06B)."""

from __future__ import annotations

import numpy as np

from fraudlens.evaluation.metrics import (
    best_f1_threshold,
    brier,
    confusion_at_threshold,
    ece,
    evaluate_scores,
    fpr_at_threshold,
    lift_over_base_rate,
    pr_auc,
    precision_at_top_k,
    precision_recall_f1_at_threshold,
    recall_at_precision,
    recall_at_top_k,
    roc_auc,
)


def test_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, score) == 1.0
    assert roc_auc(y, score) == 1.0


def test_constant_score_pr_auc_equals_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(int)
    score = np.full(len(y), 0.5)
    assert abs(pr_auc(y, score) - y.mean()) < 0.02


def test_confusion_and_prf_at_threshold():
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.4, 0.8, 0.2])
    c = confusion_at_threshold(y, score, threshold=0.5)
    assert c == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    prf = precision_recall_f1_at_threshold(y, score, 0.5)
    assert prf["precision"] == 0.5
    assert prf["recall"] == 0.5
    assert prf["f1"] == 0.5
    assert fpr_at_threshold(y, score, 0.5) == 0.5


def test_recall_and_precision_at_top_k():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    # top 40% (k=2) captures both positives
    assert recall_at_top_k(y, score, 0.4) == 1.0
    assert precision_at_top_k(y, score, 0.4) == 1.0
    # top 20% (k=1) captures one of two positives
    assert recall_at_top_k(y, score, 0.2) == 0.5


def test_recall_at_precision_edges():
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1])  # perfect separation
    assert recall_at_precision(y, score, 0.9) == 1.0
    # unattainable precision -> NaN (negative outranks the single positive)
    y2 = np.array([1, 0])
    score2 = np.array([0.8, 0.9])
    assert np.isnan(recall_at_precision(y2, score2, 0.9))


def test_ties_in_scores():
    y = np.array([1, 0, 0, 0])
    score = np.ones(4)  # all tied
    # top 25% (k=1) picks the first row; deterministic via stable argsort
    assert recall_at_top_k(y, score, 0.25) == 1.0
    assert recall_at_top_k(y, score, 1.0) == 1.0


def test_brier_and_ece_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    assert brier(y, y.astype(float)) == 0.0
    assert ece(y, y.astype(float)) == 0.0
    # scores inverted vs labels -> brier large, ece large
    assert brier(y, 1 - y.astype(float)) == 1.0


def test_best_f1_threshold_finds_separation():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.85, 0.9])
    thr, f1 = best_f1_threshold(y, score)
    assert f1 == 1.0
    assert 0.2 < thr <= 0.85


def test_lift_over_base_rate():
    y = np.array([0, 0, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9])
    # top 25% = the single positive -> precision 1.0, base 0.25 -> lift 4.0
    assert lift_over_base_rate(y, score, 0.25) == 4.0


def test_evaluate_scores_bundle_keys():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    score = np.linspace(0, 1, 8)
    out = evaluate_scores(y, score, ks=[0.125, 0.25], precision_targets=[0.9, 0.8])
    for key in ("pr_auc", "roc_auc", "brier", "ece"):
        assert key in out
    assert "recall_at_top_0.125" in out
    assert "recall_at_precision_0.9" in out
