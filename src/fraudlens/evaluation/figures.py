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


def plot_reliability(prob_map: dict[str, tuple[np.ndarray, list[dict]]], path: str | Path) -> Path:
    """Reliability diagram: mean predicted vs observed per model/calibrator."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_base_rate, bins) in prob_map.items():
        xs = [b["mean_pred"] for b in bins]
        ys = [b["observed"] for b in bins]
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, max(ax.get_xlim()[1], 1e-3)], [0, max(ax.get_xlim()[1], 1e-3)],
            color="grey", ls="--", lw=0.8, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration reliability (val_thr)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_expected_loss_curve(scan, f1_threshold: float, cost_threshold: float,
                             capacity: float, path: str | Path) -> Path:
    """Expected-cost vs threshold with optima marked and the feasible band shaded."""
    fig, ax = plt.subplots(figsize=(7, 5))
    t = scan["threshold"].to_numpy()
    ax.plot(t, scan["total_expected_cost"], color="tab:blue", label="expected cost")
    ax.plot(t, scan["total_realized_cost"], color="tab:orange", ls="--", label="realized cost")
    feasible = scan["flagged_per_day"] <= capacity
    if feasible.any():
        ax.axvspan(t[feasible].min(), t[feasible].max(), color="green", alpha=0.10,
                   label="capacity-feasible")
    ax.axvline(f1_threshold, color="tab:red", ls=":", label=f"F1-optimal {f1_threshold:.4f}")
    ax.axvline(
        cost_threshold, color="tab:green", ls=":", label=f"cost-optimal {cost_threshold:.4f}"
    )
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Total cost (₹)")
    ax.set_title("Expected-loss curve vs threshold")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_cost_breakdown(comparison, path: str | Path) -> Path:
    """Stacked cost bars (fraud loss / review / friction+step-up) per policy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    policies = comparison["policy"].tolist()
    ax.bar(policies, comparison["fraud_loss"], label="fraud loss")
    ax.bar(policies, comparison["review_cost"], bottom=comparison["fraud_loss"], label="review")
    ax.bar(
        policies,
        comparison["friction_cost"],
        bottom=comparison["fraud_loss"] + comparison["review_cost"],
        label="friction + step-up",
    )
    ax.set_ylabel("Realized cost (₹)")
    ax.set_title("Realized cost breakdown by policy")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_flagged_vs_capacity(scan, capacity: float, path: str | Path) -> Path:
    """Flagged-per-day vs threshold with the capacity line."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scan["threshold"], scan["flagged_per_day"], color="tab:blue")
    ax.axhline(capacity, color="tab:red", ls="--", label=f"capacity/day = {capacity:g}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Flagged per day")
    ax.set_title("Review demand vs analyst capacity")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_amount_threshold(params, amount_grid: np.ndarray, path: str | Path) -> Path:
    """Closed-form amount-dependent threshold p*(A) for APPROVE vs REVIEW (c=1)."""
    threshold = (params.review_cost + params.friction_cost) / (
        params.alpha * amount_grid + params.fixed_fee + params.friction_cost
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(amount_grid, threshold, color="tab:purple")
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount A (₹, log scale)")
    ax.set_ylabel("Cost-optimal probability threshold p*(A)")
    ax.set_title("Amount-dependent threshold (APPROVE vs REVIEW, c=1)")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sensitivity(sensitivity, path: str | Path) -> Path:
    """Threshold and realized cost per cost scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(sensitivity["scenario"], sensitivity["cost_optimal_threshold"], color="tab:blue")
    ax1.set_title("Cost-optimal threshold by scenario")
    ax1.set_ylabel("threshold")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.bar(sensitivity["scenario"], sensitivity["pct_saving_vs_f1"], color="tab:green")
    ax2.set_title("Realized saving vs F1-optimal (%)")
    ax2.set_ylabel("% saving")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

