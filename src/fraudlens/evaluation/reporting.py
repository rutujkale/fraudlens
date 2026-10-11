"""Validation comparison reporting: tables, per-scenario recall, holdout, figures (Part 06G).

Labels (``is_fraud`` for scoring; ``fraud_scenario`` for the per-scenario breakdown)
are used **only** here, at evaluation time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.evaluation.bootstrap import cluster_bootstrap
from fraudlens.evaluation.metrics import evaluate_scores, pr_auc


def build_comparison(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    groups: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """One row per model: PR-AUC (+cluster-bootstrap CI), ROC-AUC, recall/lift, F1."""
    sim = cfg.simulation
    rows = []
    for name, score in score_map.items():
        metrics = evaluate_scores(y, score, sim.ks, sim.precision_targets)
        _, lo, hi = cluster_bootstrap(
            y, score, groups, pr_auc, n=sim.bootstrap.n, seed=sim.seed
        )
        row = {
            "model": name,
            "pr_auc": metrics["pr_auc"],
            "pr_auc_lo": lo,
            "pr_auc_hi": hi,
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "best_f1": metrics["best_f1"],
        }
        for key, val in metrics.items():
            if key.startswith(("recall_at_top", "recall_at_precision", "lift_at_top")):
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


def per_scenario_recall(
    scenario: pd.Series, score: np.ndarray, k_frac: float
) -> pd.DataFrame:
    """Recall per fraud scenario at the top-``k_frac`` alert budget (val window)."""
    scenario = scenario.to_numpy()
    n = len(score)
    k = max(1, int(np.ceil(k_frac * n)))
    alert = np.zeros(n, dtype=bool)
    alert[np.argsort(-score, kind="stable")[:k]] = True

    rows = []
    for s in sorted(pd.Series(scenario).dropna().unique()):
        idx = np.flatnonzero(scenario == s)
        if len(idx) == 0:
            continue
        captured = int(alert[idx].sum())
        rows.append(
            {
                "scenario": str(s),
                "n_val_fraud": int(len(idx)),
                "captured_at_top": captured,
                "recall": captured / len(idx),
            }
        )
    return pd.DataFrame(rows)


def group_metrics(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    ks: list[float],
) -> dict[str, float]:
    """PR-AUC / ROC-AUC / recall@K for the subset selected by ``mask``."""
    y_m, score_m = y[mask], score[mask]
    out = {"n": int(mask.sum()), "n_fraud": int(y_m.sum())}
    if y_m.sum() == 0 or len(y_m) == 0:
        out.update({"pr_auc": float("nan"), "roc_auc": float("nan")})
        return out
    metrics = evaluate_scores(y_m, score_m, ks, [])
    out["pr_auc"] = metrics["pr_auc"]
    out["roc_auc"] = metrics["roc_auc"]
    for k in ks:
        out[f"recall_at_top_{k:g}"] = metrics[f"recall_at_top_{k:g}"]
    return out


def holdout_comparison(
    y: np.ndarray,
    score: np.ndarray,
    customers: np.ndarray,
    heldout: np.ndarray,
    ks: list[float],
) -> pd.DataFrame:
    """Validation metrics for seen (Setting A) vs held-out (Setting B) customers."""
    in_holdout = np.isin(customers, heldout)
    seen = group_metrics(y, score, ~in_holdout, ks)
    unseen = group_metrics(y, score, in_holdout, ks)
    return pd.DataFrame(
        [
            {"setting": "A_seen_customers", **seen},
            {"setting": "B_unseen_customers", **unseen},
        ]
    )


def to_markdown(comparison: pd.DataFrame) -> str:
    """Render the comparison table (and key columns) to markdown."""
    cols = [
        "model",
        "pr_auc",
        "pr_auc_lo",
        "pr_auc_hi",
        "roc_auc",
        "best_f1",
        "recall_at_precision_0.9",
        "recall_at_top_0.01",
    ]
    cols = [c for c in cols if c in comparison.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in comparison.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)
