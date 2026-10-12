"""Evaluation metrics for imbalanced fraud scoring (Part 06B).

PR-AUC (average precision) is the primary metric; accuracy is never a headline.
All functions take ``y`` (0/1) and a continuous ``score`` (higher = more fraud).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as log_loss_metric,
)


def _arrays(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y).astype(int), np.asarray(score, dtype=float)


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    y, score = _arrays(y, score)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y, _ = _arrays(y, score)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def confusion_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    """TP/FP/TN/FN at ``score >= threshold``."""
    y, score = _arrays(y, score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def precision_recall_f1_at_threshold(
    y: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, float]:
    c = confusion_at_threshold(y, score, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def fpr_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> float:
    c = confusion_at_threshold(y, score, threshold)
    fp, tn = c["fp"], c["tn"]
    return float(fp / (fp + tn)) if (fp + tn) else float("nan")


def _top_k_indices(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top ``k`` scores (ties broken by original order)."""
    return np.argsort(-score, kind="stable")[:k]


def recall_at_top_k(y: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Fraction of positives captured in the top ``k_frac`` of rows (k >= 1)."""
    y, score = _arrays(y, score)
    n = len(y)
    k = max(1, int(np.ceil(k_frac * n)))
    top = _top_k_indices(score, k)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    return float(y[top].sum() / total_pos)


def precision_at_top_k(y: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    y, score = _arrays(y, score)
    n = len(y)
    k = max(1, int(np.ceil(k_frac * n)))
    top = _top_k_indices(score, k)
    return float(y[top].mean())


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """Max recall achievable with precision >= ``target`` (NaN if unattainable)."""
    y, score = _arrays(y, score)
    if y.sum() == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, score)
    # Ignore the terminal (recall=0, precision=1) point sklearn appends: making no
    # positive predictions is not an operating point.
    mask = (precision >= target) & (recall > 0)
    if not mask.any():
        return float("nan")
    return float(recall[mask].max())


def brier(y: np.ndarray, score: np.ndarray) -> float:
    """Brier score (mean squared error of the score). Lower is better."""
    y, score = _arrays(y, score)
    return float(brier_score_loss(y, score))


def log_loss(y: np.ndarray, score: np.ndarray) -> float:
    """Binary log-loss of the (clipped) probabilities. Lower is better."""
    y, score = _arrays(y, score)
    clipped = np.clip(score, 1e-15, 1 - 1e-15)
    return float(log_loss_metric(y, clipped, labels=[0, 1]))


def ece(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with **equal-frequency** bins.

    Rows are sorted by score and split into ``n_bins`` equal-count bins; the
    contribution of each bin is ``|mean(score) - mean(y)| * bin_fraction``.
    """
    y, score = _arrays(y, score)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, n_bins)
    total = len(y)
    err = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        err += (len(b) / total) * abs(score[b].mean() - y[b].mean())
    return float(err)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 and the F1 value (threshold from PR curve)."""
    y, score = _arrays(y, score)
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # precision/recall have one more element than thresholds; compare on thresholds.
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    thr = float(thresholds[best]) if len(thresholds) else float("inf")
    return thr, float(f1[best])


def lift_over_base_rate(y: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Precision@top-k divided by the base prevalence."""
    y, _ = _arrays(y, score)
    base = y.mean()
    if base == 0:
        return float("nan")
    return float(precision_at_top_k(y, score, k_frac) / base)


def evaluate_scores(
    y: np.ndarray,
    score: np.ndarray,
    ks: list[float],
    precision_targets: list[float],
) -> dict[str, float]:
    """Convenience bundle of the scalar validation metrics."""
    y, score = _arrays(y, score)
    thr, f1 = best_f1_threshold(y, score)
    out: dict[str, float] = {
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "brier": brier(y, score),
        "ece": ece(y, score),
        "base_rate": float(y.mean()),
        "best_f1": f1,
        "best_f1_threshold": thr,
    }
    for k in ks:
        out[f"recall_at_top_{k:g}"] = recall_at_top_k(y, score, k)
        out[f"lift_at_top_{k:g}"] = lift_over_base_rate(y, score, k)
    for p in precision_targets:
        out[f"recall_at_precision_{p:g}"] = recall_at_precision(y, score, p)
    return out
