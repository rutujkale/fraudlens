"""Threshold search and policy comparison on ``val_thr`` (Part 07C).

Global-threshold policies are scanned over a fine grid; the amount-aware and
ranking policies live in :mod:`fraudlens.risk.policy`. All selection uses the
**validation** window only. Headline costs are **realized** (labels on
``val_thr``); expected cost is kept as a diagnostic (and a calibration sanity
check: expected ≈ realized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.evaluation.cost_model import (
    CostParams,
    cost_breakdown,
    review_all,
)
from fraudlens.evaluation.metrics import best_f1_threshold
from fraudlens.risk.policy import (
    amount_aware_actions,
    enforce_capacity,
    rank_key,
    top_k_actions,
)


def daily_capacity(dates: np.ndarray, cfg: Config) -> tuple[int, float]:
    """Return ``(capacity_per_day, avg_daily_volume)`` for the val_thr window."""
    dates = np.asarray(dates)
    days = len(np.unique(dates))
    avg_daily = len(dates) / days if days else 0.0
    costs = cfg.simulation.costs
    if costs.analyst_daily_capacity is not None:
        capacity = int(costs.analyst_daily_capacity)
    else:
        capacity = int(np.ceil(costs.capacity_pct_of_daily_volume * avg_daily))
    return max(capacity, 1), avg_daily


def _rates(y: np.ndarray, review_mask: np.ndarray) -> dict[str, float]:
    tp = int(((y == 1) & review_mask).sum())
    fp = int(((y == 0) & review_mask).sum())
    fn = int(((y == 1) & ~review_mask).sum())
    tn = int(((y == 0) & ~review_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr}


def scan_global_thresholds(
    y: np.ndarray,
    prob: np.ndarray,
    amount: np.ndarray,
    dates: np.ndarray,
    params: CostParams,
    grid: np.ndarray,
) -> pd.DataFrame:
    """Metrics + costs for ``REVIEW if p >= t else APPROVE`` over a threshold grid."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    n_days = len(np.unique(dates))
    rows = []
    for t in grid:
        mask = prob >= t
        r = _rates(y, mask)
        action = np.where(mask, 2, 0)
        real = cost_breakdown(action, y, amount, params, realized=True)
        exp = cost_breakdown(action, prob, amount, params, realized=False)
        rows.append(
            {
                "threshold": float(t),
                "flagged": int(mask.sum()),
                "flagged_per_day": float(mask.sum() / n_days) if n_days else float("nan"),
                **r,
                "expected_loss": exp["fraud_loss"],
                "total_expected_cost": exp["total"],
                "fraud_loss": real["fraud_loss"],
                "review_cost": real["review_cost"],
                "friction_cost": real["friction_cost"] + real["stepup_cost"],
                "total_realized_cost": real["total"],
            }
        )
    return pd.DataFrame(rows)


def op_at_threshold(
    threshold: float, y, prob, amount, dates, params: CostParams, label: str
) -> dict:
    """Single operating-point summary at a fixed global threshold."""
    action = np.where(np.asarray(prob, dtype=float) >= threshold, 2, 0)
    return _summary(label, action, y, amount, prob, dates, params)


def _summary(label, action, y, amount, prob, dates, params: CostParams) -> dict:
    y = np.asarray(y).astype(int)
    action = np.asarray(action).astype(int)
    intervene = action != 0  # any non-APPROVE action counts as flagged for recall
    review_mask = action == 2  # REVIEW is the capacity-limited queue
    r = _rates(y, intervene)
    real = cost_breakdown(action, y, amount, params, realized=True)
    exp = cost_breakdown(action, np.asarray(prob, dtype=float), amount, params, realized=False)
    n_days = len(np.unique(dates))
    return {
        "policy": label,
        "flagged_per_day": float(review_mask.sum() / n_days) if n_days else float("nan"),
        "recall": r["recall"],
        "precision": r["precision"],
        "fpr": r["fpr"],
        "fraud_loss": real["fraud_loss"],
        "review_cost": real["review_cost"],
        "friction_cost": real["friction_cost"] + real["stepup_cost"],
        "total_realized_cost": real["total"],
        "total_expected_cost": exp["total"],
    }


def build_policy_comparison(
    y: np.ndarray,
    prob: np.ndarray,
    amount: np.ndarray,
    dates: np.ndarray,
    params: CostParams,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All nine policies + the capacity used; realized cost is the headline column."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    capacity, avg_daily = daily_capacity(dates, cfg)

    grid = np.unique(np.quantile(prob, np.linspace(0.80, 1.0, 201)))
    scan = scan_global_thresholds(y, prob, amount, dates, params, grid)

    f1_thr, _ = best_f1_threshold(y, prob)
    f1_thr = float(np.clip(f1_thr, prob.min(), prob.max()))
    cost_row = scan.loc[scan["total_realized_cost"].idxmin()]
    feasible = scan[scan["flagged_per_day"] <= capacity]
    feas_row = feasible.loc[feasible["total_realized_cost"].idxmin()]

    policies = [
        _summary("approve_all", np.zeros(len(y), dtype=int), y, amount, prob, dates, params),
        _summary("review_all", review_all(len(y)), y, amount, prob, dates, params),
        op_at_threshold(f1_thr, y, prob, amount, dates, params, "f1_optimal"),
        op_at_threshold(
            float(cost_row["threshold"]), y, prob, amount, dates, params, "cost_optimal_global"
        ),
        op_at_threshold(
            float(feas_row["threshold"]), y, prob, amount, dates, params,
            "cost_optimal_global_capacity",
        ),
    ]

    aware = amount_aware_actions(prob, amount, params)
    policies.append(_summary("amount_aware", aware, y, amount, prob, dates, params))
    capped = enforce_capacity(dates, prob, amount, aware, params, capacity)
    policies.append(
        _summary("amount_aware_capacity", capped.actions, y, amount, prob, dates, params)
    )

    for key in ("p", "p_times_amount", "expected_saving"):
        actions = top_k_actions(dates, rank_key(prob, amount, params, key), capacity)
        policies.append(_summary(f"top_k_by_{key}", actions, y, amount, prob, dates, params))

    df = pd.DataFrame(policies)
    f1_cost = float(df.loc[df["policy"] == "f1_optimal", "total_realized_cost"].iloc[0])
    df["pct_cost_reduction_vs_f1"] = (
        (f1_cost - df["total_realized_cost"]) / f1_cost * 100.0 if f1_cost else 0.0
    )
    df["feasible"] = df["flagged_per_day"] <= capacity

    meta = {
        "capacity_per_day": int(capacity),
        "avg_daily_volume": float(avg_daily),
        "f1_threshold": float(f1_thr),
        "cost_optimal_threshold": float(cost_row["threshold"]),
        "cost_optimal_threshold_capacity": float(feas_row["threshold"]),
        "n_demoted": int(capped.n_demoted),
        "n_days": int(len(np.unique(dates))),
    }
    return df, scan, meta
