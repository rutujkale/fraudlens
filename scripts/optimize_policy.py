"""Thin CLI: cost-optimal decision policy on the val_thr slice (Part 07C–07F).

Loads the calibrator chosen in Part 07A, calibrates the XGBoost scores on
``val_thr``, then compares global-threshold, amount-aware and ranking policies
under analyst capacity. Headline costs are realized; expected cost is a
diagnostic. Writes ``policy.json`` into the model artifact and logs E05.
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from fraudlens.config import config_hash, load_config
from fraudlens.evaluation.cost_model import cost_breakdown, resolve_cost_params
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.experiments import log_experiment
from fraudlens.evaluation.figures import (
    plot_amount_threshold,
    plot_cost_breakdown,
    plot_expected_loss_curve,
    plot_flagged_vs_capacity,
    plot_sensitivity,
)
from fraudlens.evaluation.scores import load_model_versions, raw_val_scores
from fraudlens.evaluation.splits import chronological_fraction_split, make_temporal_split
from fraudlens.evaluation.threshold_search import build_policy_comparison, op_at_threshold
from fraudlens.features.registry import FEATURE_VERSION
from fraudlens.models.io import assert_artifact_profile
from fraudlens.paths import artifacts_dir, figures_dir, results_dir
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)
POLICY_VERSION = "cost_policy_v1"


def _val_thr(df, split, cfg, raw):
    """Return the calibrated val_thr arrays (y, p, amount, dates) and selection info."""
    versions = load_model_versions(cfg.simulation.profile)
    assert_artifact_profile(versions["xgb"], cfg.simulation.profile)
    n_val = len(split.val_idx)
    _, thr_pos = chronological_fraction_split(
        np.arange(n_val), cfg.simulation.calibration.cal_fraction
    )
    calibrator = joblib.load(artifacts_dir(versions["xgb"]) / "calibrator.joblib")
    selection = json.loads(
        (artifacts_dir(versions["xgb"]) / "calibration.json").read_text(encoding="utf-8")
    )
    thr = df.iloc[split.val_idx].iloc[thr_pos]
    p = calibrator.predict(raw["xgb"][thr_pos])
    dates = thr["timestamp"].dt.date.to_numpy()
    return (
        thr["is_fraud"].to_numpy(int),
        p,
        thr["amount"].to_numpy(float),
        dates,
        selection,
        versions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize the cost-based decision policy.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--scenario", default="balanced")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    df, labels = load_feature_dataset(args.profile)
    df["is_fraud"] = df["transaction_id"].map(
        labels.set_index("transaction_id")["is_fraud"]
    )
    split = make_temporal_split(df, cfg)
    raw = raw_val_scores(df, split, cfg)
    y, p, amount, dates, selection, versions = _val_thr(df, split, cfg, raw)

    scenario = cfg.simulation.cost_scenarios.get(args.scenario)
    if scenario is None:
        raise ValueError(f"unknown scenario {args.scenario!r}")
    params = resolve_cost_params(cfg.simulation.costs, scenario)

    comparison, scan, meta = build_policy_comparison(y, p, amount, dates, params, cfg)
    out_dir = results_dir()
    suffix = "" if args.scenario == "balanced" else f"_{args.scenario}"
    comparison.to_csv(out_dir / f"policy_comparison_{args.profile}{suffix}.csv", index=False)
    scan.to_csv(out_dir / f"threshold_scan_{args.profile}{suffix}.csv", index=False)
    logger.info(
        "capacity/day=%d (avg daily volume %.1f)",
        meta["capacity_per_day"],
        meta["avg_daily_volume"],
    )
    logger.info("policy comparison:\n%s", comparison.to_string(index=False))

    plot_expected_loss_curve(
        scan, meta["f1_threshold"], meta["cost_optimal_threshold"], meta["capacity_per_day"],
        figures_dir() / f"expected_loss_curve_{args.profile}.png",
    )
    plot_cost_breakdown(comparison, figures_dir() / f"cost_breakdown_{args.profile}.png")
    plot_flagged_vs_capacity(
        scan, meta["capacity_per_day"], figures_dir() / f"flagged_vs_capacity_{args.profile}.png"
    )
    amount_grid = np.logspace(2, 6, 100)
    plot_amount_threshold(
        params, amount_grid, figures_dir() / f"amount_threshold_{args.profile}.png"
    )

    sens = _sensitivity(y, p, amount, dates, cfg, args.profile, meta)
    sens.to_csv(out_dir / f"sensitivity_{args.profile}.csv", index=False)
    plot_sensitivity(sens, figures_dir() / f"sensitivity_{args.profile}.png")
    logger.info("sensitivity:\n%s", sens.to_string(index=False))

    rates = _rate_sensitivity(y, p, amount, dates, cfg)
    rates.to_csv(out_dir / f"sensitivity_rates_{args.profile}.csv", index=False)
    logger.info("rate sensitivity:\n%s", rates.to_string(index=False))

    risk = _risk_distribution(p, y.mean())
    risk.to_csv(out_dir / f"risk_distribution_{args.profile}.csv", index=False)

    f1_row = comparison.loc[comparison["policy"] == "f1_optimal"].iloc[0]
    aware_row = comparison.loc[comparison["policy"] == "amount_aware_capacity"].iloc[0]
    policy = {
        "policy_version": POLICY_VERSION,
        "profile": args.profile,
        "scenario": args.scenario,
        "costs": params.__dict__,
        "capacity_per_day": meta["capacity_per_day"],
        "avg_daily_volume": meta["avg_daily_volume"],
        "f1_threshold": meta["f1_threshold"],
        "cost_optimal_threshold": meta["cost_optimal_threshold"],
        "cost_optimal_threshold_capacity": meta["cost_optimal_threshold_capacity"],
        "selected_calibrator": selection["selected"],
        "n_demoted": meta["n_demoted"],
        "validation": {
            "f1_realized_cost": float(f1_row["total_realized_cost"]),
            "amount_aware_capacity_realized_cost": float(aware_row["total_realized_cost"]),
            "amount_aware_capacity_pct_saving_vs_f1": float(aware_row["pct_cost_reduction_vs_f1"]),
        },
    }
    (artifacts_dir(versions["xgb"]) / "policy.json").write_text(
        json.dumps(policy, indent=2, default=str), encoding="utf-8"
    )

    if np.unique(y).size < 2:
        logger.warning("val_thr has a single class (%d positives); skipping E05 log", int(y.sum()))
        return 0
    log_experiment(
        {
            "experiment_id": "E05",
            "part": "07",
            "dataset_profile": args.profile,
            "model": "xgboost+calibration+cost_policy",
            "features_version": FEATURE_VERSION,
            "train_window": {
                "start": split.boundaries["t0"],
                "end": split.boundaries["t_train_end"],
            },
            "val_window": {
                "start": split.boundaries["t_train_end"],
                "end": split.boundaries["t_val_end"],
            },
            "hyperparameters": {
                "scenario": args.scenario,
                "capacity_per_day": meta["capacity_per_day"],
                "policy_version": POLICY_VERSION,
            },
            "calibration": selection["selected"],
            "threshold_policy": "cost_optimal_capacity",
            "metrics": {
                "realized_cost_f1": round(float(f1_row["total_realized_cost"]), 4),
                "realized_cost_amount_aware": round(float(aware_row["total_realized_cost"]), 4),
                "pct_saving_vs_f1": round(float(aware_row["pct_cost_reduction_vs_f1"]), 4),
                "recall": round(float(aware_row["recall"]), 6),
                "flagged_per_day": round(float(aware_row["flagged_per_day"]), 4),
            },
            "notes": "amount-aware policy under daily review capacity; realized cost headline",
        },
        config_hash=chash,
    )
    return 0


def _sensitivity(y, p, amount, dates, cfg, profile, meta) -> pd.DataFrame:
    """Cost-optimal threshold + saving vs F1-optimal, per named cost scenario."""
    from fraudlens.evaluation.metrics import best_f1_threshold
    from fraudlens.evaluation.threshold_search import scan_global_thresholds

    grid = np.unique(np.quantile(p, np.linspace(0.80, 1.0, 201)))
    base_scenario = cfg.simulation.cost_scenarios["balanced"]
    base_params = resolve_cost_params(cfg.simulation.costs, base_scenario)
    f1_thr = float(np.clip(best_f1_threshold(y, p)[0], p.min(), p.max()))
    f1_cost = op_at_threshold(f1_thr, y, p, amount, dates, base_params, "f1")["total_realized_cost"]

    rows = []
    for name, scenario in cfg.simulation.cost_scenarios.items():
        params = resolve_cost_params(cfg.simulation.costs, scenario)
        scan = scan_global_thresholds(y, p, amount, dates, params, grid)
        best = scan.loc[scan["total_realized_cost"].idxmin()]
        op = op_at_threshold(float(best["threshold"]), y, p, amount, dates, params, name)
        rows.append(
            {
                "scenario": name,
                "cost_optimal_threshold": float(best["threshold"]),
                "realized_cost": op["total_realized_cost"],
                "recall": op["recall"],
                "flagged_per_day": op["flagged_per_day"],
                "pct_saving_vs_f1": (f1_cost - op["total_realized_cost"]) / f1_cost * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _rate_sensitivity(y, p, amount, dates, cfg) -> pd.DataFrame:
    """Second sensitivity (7E): vary review catch-rate and step-up stop-rate."""
    from dataclasses import replace

    from fraudlens.evaluation.threshold_search import scan_global_thresholds
    from fraudlens.risk.policy import amount_aware_actions

    grid = np.unique(np.quantile(p, np.linspace(0.80, 1.0, 201)))
    base = resolve_cost_params(cfg.simulation.costs, cfg.simulation.cost_scenarios["balanced"])
    rows = []
    for label, field, values in (
        ("review_catch_rate", "catch_rate", [0.7, 0.8, 0.9, 1.0]),
        ("stepup_stop_rate", "stop_rate", [0.5, 0.6, 0.7, 0.8]),
    ):
        for value in values:
            params = replace(base, **{field: value})
            scan = scan_global_thresholds(y, p, amount, dates, params, grid)
            best = scan.loc[scan["total_realized_cost"].idxmin()]
            aware = amount_aware_actions(p, amount, params)
            aware_cost = cost_breakdown(aware, y, amount, params, realized=True)["total"]
            rows.append(
                {
                    "varied": label,
                    "value": value,
                    "cost_optimal_threshold": float(best["threshold"]),
                    "global_realized_cost": float(best["total_realized_cost"]),
                    "amount_aware_realized_cost": float(aware_cost),
                }
            )
    return pd.DataFrame(rows)


def _risk_distribution(p, base_rate) -> pd.DataFrame:
    """Empirical distribution of calibrated probabilities and action-derived levels."""
    edges = [0.0, 0.01, 0.05, 0.10, 0.25, 1.0]
    labels = ["<1%", "1-5%", "5-10%", "10-25%", ">=25%"]
    bucket = pd.cut(p, bins=edges, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        mask = np.asarray(bucket == label)
        rows.append(
            {
                "prob_band": label,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "mean_pred": float(p[mask].mean()) if mask.any() else float("nan"),
                "mean_lift": (
                    float(p[mask].mean() / base_rate) if mask.any() and base_rate else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
