"""Validation figures: PR curves and recall@K curves (Part 06G)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraudlens.evaluation.metrics import pr_auc, recall_at_top_k  # noqa: E402


def plot_pr_curves(y: np.ndarray, score_map: dict[str, np.ndarray], path: str | Path) -> Path:
    """Precision-recall curves for every model (with PR-AUC in the legend)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, score in score_map.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=f"{name} (AP={pr_auc(y, score):.3f})")
    ax.axhline(y.mean(), color="grey", ls="--", lw=0.8, label=f"base rate {y.mean():.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation PR curves")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_recall_at_k(
    y: np.ndarray, score_map: dict[str, np.ndarray], ks: list[float], path: str | Path
) -> Path:
    """Recall@top-K curves for every model over a grid of K fractions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, score in score_map.items():
        recalls = [recall_at_top_k(y, score, k) for k in ks]
        ax.plot(ks, recalls, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Alert budget (fraction of transactions, top-K)")
    ax.set_ylabel("Recall")
    ax.set_title("Validation recall@top-K")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
