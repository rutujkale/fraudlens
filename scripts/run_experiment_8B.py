"""Experiment 8B (E11): feature-group ablation, broken out by fraud scenario.

Ten groups mapped from the feature registry. For each group we retrain the same
fixed Part 06 XGBoost config twice: **drop-one** (all features except the group)
and **only-one** (just the group). We report overall PR-AUC and per-scenario
recall@top-1% on validation, then read the pattern against how each scenario is
*simulated* (burst = velocity windows; account_takeover = new device + far
location + odd hour; amount_anomaly = amount vs the customer's own baseline;
geo_anomaly = far location; structuring = sub-threshold splitting).

3 seeds; mean ± std. Test window never touched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.experiments import log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_xgb_params,
    fit_xgb,
    load_supervised,
    mean_std,
    predict_scores,
    score_row,
)
from fraudlens.evaluation.reporting import per_scenario_recall  # noqa: E402
from fraudlens.features.registry import SPECS  # noqa: E402
from fraudlens.paths import figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEEDS = [42, 43, 44]
TOP_K = 0.01


def _feature_groups() -> dict[str, list[str]]:
    """Registry groups with ``recency``+``frequency`` merged (10 groups)."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for s in SPECS:
        by_group[s.group].append(s.name)
    return {
        "transaction": by_group["transaction"],
        "recency_frequency": by_group["recency"] + by_group["frequency"],
        "amount_sums": by_group["amount_sums"],
        "amount_deviation": by_group["amount_deviation"],
        "merchant_category": by_group["merchant_category"],
        "device": by_group["device"],
        "recipient": by_group["recipient"],
        "geographic": by_group["geographic"],
        "temporal": by_group["temporal"],
        "trend": by_group["trend"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 8B: feature-group ablation.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    params = best_xgb_params(args.profile)
    params["max_depth"] = int(params["max_depth"])

    groups = _feature_groups()
    all_features = list(sup.x.columns)
    logger.info("profile=%s features=%d groups=%d", args.profile, len(all_features), len(groups))

    x_tr, y_tr = sup.x_train(), sup.y_train()
    x_va, y_va = sup.x_val(), sup.y_val()
    scenario = sup.val_scenario()

    configs: list[tuple[str, str, list[str]]] = [("full", "full", all_features)]
    for g, cols in groups.items():
        complement = [c for c in all_features if c not in cols]
        configs.append((f"drop_{g}", "drop", complement))
        configs.append((f"only_{g}", "only", cols))

    rows: list[dict] = []
    for name, mode, cols in configs:
        for seed in seeds:
            model = fit_xgb(x_tr[cols], y_tr, x_va[cols], y_va, params, seed)
            score = predict_scores(model, x_va[cols])
            row = score_row(name, y_va, score, cfg, with_ci=False)
            row["mode"] = mode
            row["seed"] = seed
            row["n_features"] = len(cols)
            scen = per_scenario_recall(pd.Series(scenario), score, TOP_K)
            for _, s in scen.iterrows():
                row[f"recall_{s['scenario']}"] = float(s["recall"])
            rows.append(row)
        logger.info("8B %-20s PR-AUC=%.4f", name, rows[-1]["pr_auc"])

    per_seed = pd.DataFrame(rows)
    scen_cols = [c for c in per_seed.columns if c.startswith("recall_") and c != "recall_at_1pct"]
    numeric = ["pr_auc", "recall_at_1pct", *scen_cols]
    summary = mean_std(per_seed, "config", numeric)
    summary = summary.merge(
        per_seed[["config", "mode", "n_features"]].drop_duplicates("config"), on="config"
    )
    per_seed.to_csv(results_dir() / f"exp_8B_ablation_{args.profile}.csv", index=False)

    # Delta vs full (mean recall per scenario) for the heatmap/doc.
    full = summary[summary["config"] == "full"].iloc[0]
    delta_rows = []
    for _, r in summary.iterrows():
        if r["config"] == "full":
            continue
        d = {"config": r["config"], "mode": r["mode"], "pr_auc_mean": r["pr_auc_mean"]}
        for sc in scen_cols:
            key = sc.replace("recall_", "")
            d[f"delta_{key}"] = float(r[f"{sc}_mean"] - full[f"{sc}_mean"])
            d[f"recall_{key}"] = float(r[f"{sc}_mean"])
        delta_rows.append(d)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(results_dir() / f"exp_8B_scenario_delta_{args.profile}.csv", index=False)
    logger.info("wrote %s", results_dir() / f"exp_8B_scenario_delta_{args.profile}.csv")

    # Figure 1: PR-AUC by configuration.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    drop = summary[summary["mode"].isin(["full", "drop"])]
    only = summary[summary["mode"] == "only"]
    ax1.bar(drop["config"], drop["pr_auc_mean"], color="tab:red")
    ax1.axhline(float(full["pr_auc_mean"]), color="grey", ls="--", lw=0.8, label="full")
    ax1.set_title("Drop-one-group PR-AUC")
    ax1.set_ylabel("PR-AUC")
    ax1.tick_params(axis="x", rotation=60, labelsize=8)
    ax1.legend(fontsize=8)
    ax2.bar(only["config"], only["pr_auc_mean"], color="tab:blue")
    ax2.axhline(float(full["pr_auc_mean"]), color="grey", ls="--", lw=0.8, label="full")
    ax2.set_title("Only-one-group PR-AUC")
    ax2.set_ylabel("PR-AUC")
    ax2.tick_params(axis="x", rotation=60, labelsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8B_ablation_{args.profile}.png", dpi=120)
    plt.close(fig)

    # Figure 2: scenario x group delta-recall heatmap (drop and only).
    scen_names = [sc.replace("recall_", "") for sc in scen_cols]
    group_names = list(groups.keys())
    for mode in ("drop", "only"):
        mat = np.full((len(group_names), len(scen_names)), np.nan)
        for i, g in enumerate(group_names):
            row = delta[(delta["config"] == f"{mode}_{g}")]
            if row.empty:
                continue
            for j, sc in enumerate(scen_names):
                mat[i, j] = row.iloc[0][f"delta_{sc}"]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(scen_names)))
        ax.set_xticklabels(scen_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(group_names)))
        ax.set_yticklabels(group_names, fontsize=8)
        for i in range(len(group_names)):
            for j in range(len(scen_names)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{mode.capitalize()}-group delta recall@1% by scenario")
        fig.colorbar(im, label="Δrecall vs full")
        fig.tight_layout()
        fig.savefig(figures_dir() / f"exp_8B_scenario_delta_{mode}_{args.profile}.png", dpi=120)
        plt.close(fig)

    for _, r in summary.iterrows():
        log_experiment(
            {
                "experiment_id": "E11",
                "part": "08",
                "dataset_profile": args.profile,
                "model": "xgboost",
                "features_version": "features_v1",
                "hyperparameters": {
                    "ablation": r["config"],
                    "mode": r["mode"],
                    "n_features": int(r["n_features"]),
                },
                "calibration": "none",
                "threshold_policy": "rank_by_score",
                "metrics": {
                    "pr_auc_mean": round(float(r["pr_auc_mean"]), 6),
                    "pr_auc_std": round(float(r["pr_auc_std"]), 6),
                },
                "notes": f"feature-group {r['mode']}; {args.profile}",
            },
            config_hash=chash,
        )
    logger.info("logged %d E11 rows", len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
