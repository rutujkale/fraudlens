"""Thin recompute wrappers for the Review Queue page (Part 14).

The sliders on Page 2 must change the decision in real time without the app
re-implementing any cost or policy logic. These functions call the **Part 07
package code** (``evaluation.cost_model``, ``evaluation.threshold_search``,
``risk.policy``) on the precomputed ``val_thr`` sample and return plain dicts the
page renders. Tests assert these results equal calling the package directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.evaluation.cost_model import (
    ACTION_NAMES,
    APPROVE,
    CostParams,
    cost_breakdown,
    expected_cost,
    realized_cost,
    resolve_cost_params,
)
from fraudlens.evaluation.metrics import best_f1_threshold
from fraudlens.evaluation.threshold_search import op_at_threshold, scan_global_thresholds
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity, expected_saving
from fraudlens.risk.scoring import action_name


def sample_arrays(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(is_fraud, prob, amount, dates)`` from a scored-sample frame."""
    y = sample["is_fraud"].to_numpy().astype(int)
    prob = sample["fraud_probability"].to_numpy(dtype=float)
    amount = sample["amount"].to_numpy(dtype=float)
    dates = np.array(pd.to_datetime(sample["timestamp"]).dt.date.tolist(), dtype=object)
    return y, prob, amount, dates


def threshold_grid(prob: np.ndarray) -> np.ndarray:
    """The Part 07 grid: 201 quantiles of the upper score range."""
    return np.unique(np.quantile(np.asarray(prob, dtype=float), np.linspace(0.80, 1.0, 201)))


def scan(sample: pd.DataFrame, params: CostParams) -> pd.DataFrame:
    """Global-threshold scan over the sample (delegates to ``scan_global_thresholds``)."""
    y, prob, amount, dates = sample_arrays(sample)
    return scan_global_thresholds(y, prob, amount, dates, params, threshold_grid(prob))


def best_thresholds(sample: pd.DataFrame, params: CostParams, capacity: int) -> dict[str, Any]:
    """Best global thresholds (F1 / cost / cost-feasible) with their realized cost."""
    y, prob, amount, dates = sample_arrays(sample)
    scanned = scan(sample, params)
    cost_row = scanned.loc[scanned["total_realized_cost"].idxmin()]
    feasible = scanned[scanned["flagged_per_day"] <= capacity]
    feas_row = feasible.loc[feasible["total_realized_cost"].idxmin()]
    f1_thr, _ = best_f1_threshold(y, prob)
    f1_thr = float(np.clip(f1_thr, prob.min(), prob.max()))
    f1 = op_at_threshold(f1_thr, y, prob, amount, dates, params, "f1_optimal")
    return {
        "f1_threshold": f1_thr,
        "f1_realized_cost": float(f1["total_realized_cost"]),
        "cost_optimal_threshold": float(cost_row["threshold"]),
        "cost_optimal_realized_cost": float(cost_row["total_realized_cost"]),
        "cost_optimal_feasible_threshold": float(feas_row["threshold"]),
        "cost_optimal_feasible_realized_cost": float(feas_row["total_realized_cost"]),
        "capacity_per_day": int(capacity),
    }


def policy_metrics(sample: pd.DataFrame, params: CostParams, capacity: int) -> dict[str, Any]:
    """Amount-aware policy under ``params`` + capacity, on the sample."""
    y, prob, amount, dates = sample_arrays(sample)
    actions = amount_aware_actions(prob, amount, params)
    capped = enforce_capacity(dates, prob, amount, actions, params, capacity)
    chosen = capped.actions
    n_days = len(np.unique(dates)) or 1

    realized = cost_breakdown(chosen, y, amount, params, realized=True)
    expected = cost_breakdown(chosen, prob, amount, params, realized=False)
    approve_expected = float(expected_cost(APPROVE, prob, amount, params).sum())
    approve_realized = float(
        realized_cost(np.full(len(y), APPROVE, dtype=int), y, amount, params).sum()
    )

    counts = {name: int((chosen == code).sum()) for code, name in enumerate(ACTION_NAMES)}
    return {
        "n": int(len(y)),
        "counts_by_action": counts,
        "flagged": int((chosen != APPROVE).sum()),
        "review_workload": int(counts["REVIEW"]),
        "review_per_day": float(counts["REVIEW"] / n_days),
        "n_demoted": int(capped.n_demoted),
        "expected_loss": float(expected["total"]),
        "expected_loss_avoided": float(approve_expected - expected["total"]),
        "realized_cost": float(realized["total"]),
        "realized_loss_avoided": float(approve_realized - realized["total"]),
        "capacity_per_day": int(capacity),
    }


def queue_table(sample: pd.DataFrame, params: CostParams, capacity: int) -> pd.DataFrame:
    """Sample plus per-row capacity-capped action and expected saving under ``params``."""
    _, prob, amount, dates = sample_arrays(sample)
    actions = amount_aware_actions(prob, amount, params)
    capped = enforce_capacity(dates, prob, amount, actions, params, capacity)
    out = sample.copy()
    out["recommended_action"] = action_name(capped.actions)
    out["expected_saving"] = expected_saving(prob, amount, params)
    return out


def scenario_table(sample: pd.DataFrame, cfg: Config, capacity: int) -> pd.DataFrame:
    """Compare the preset cost scenarios on the same sample."""
    rows = []
    for name, scenario in cfg.simulation.cost_scenarios.items():
        params = resolve_cost_params(cfg.simulation.costs, scenario)
        best = best_thresholds(sample, params, capacity)
        metrics = policy_metrics(sample, params, capacity)
        rows.append(
            {
                "scenario": name,
                "cost_optimal_threshold": best["cost_optimal_threshold"],
                "realized_cost": metrics["realized_cost"],
                "pct_reduction_vs_f1": (
                    (best["f1_realized_cost"] - metrics["realized_cost"])
                    / best["f1_realized_cost"]
                    * 100.0
                    if best["f1_realized_cost"]
                    else 0.0
                ),
                "review_per_day": metrics["review_per_day"],
                "feasible": metrics["review_per_day"] <= capacity,
            }
        )
    return pd.DataFrame(rows)
