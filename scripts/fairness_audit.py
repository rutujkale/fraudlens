"""Part 10E: subgroup fairness audit on validation data.

Two views (see ``fraudlens.explain.fairness``): the primary ``val_thr`` slice the
policy was selected on, and an exploratory full-validation view. Audit attributes
are never model inputs; they are joined for analysis only. The test window is
never touched.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fraudlens.config import config_hash, load_config
from fraudlens.evaluation.cost_model import resolve_cost_params
from fraudlens.evaluation.experiments import drop_experiment_rows, log_experiment
from fraudlens.evaluation.experiments_lib import load_supervised
from fraudlens.evaluation.scores import load_model_versions
from fraudlens.evaluation.splits import chronological_fraction_split
from fraudlens.explain.fairness import (
    AUDIT_ATTRIBUTES,
    MIN_ROWS,
    flag_rate_metric,
    fpr_metric,
    group_table,
    load_audit_attributes,
    metric_ratios,
    paired_group_diff,
    proxy_check,
    recall_metric,
)
from fraudlens.explain.report import render_fairness_markdown
from fraudlens.features.registry import FEATURE_VERSION
from fraudlens.models import io as model_io
from fraudlens.paths import docs_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)

PAIRED_METRICS = (
    ("recall", recall_metric),
    ("fpr_any", fpr_metric),
    ("flag_rate_any", flag_rate_metric),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 10E fairness / subgroup audit.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    sup = load_supervised(args.profile)
    version = load_model_versions(args.profile)["xgb"]
    bundle = model_io.load_serving_bundle(version)
    model, calibrator, policy = bundle["model"], bundle["calibrator"], bundle["policy"]
    params = resolve_cost_params(
        cfg.simulation.costs, cfg.simulation.cost_scenarios[policy["scenario"]]
    )
    capacity = int(policy["capacity_per_day"])
    n_boot = int(cfg.simulation.bootstrap.n)
    seed = int(cfg.simulation.seed)

    x_val = sup.x_val().reset_index(drop=True)
    y_val = sup.y_val()
    val = sup.df.iloc[sup.split.val_idx].reset_index(drop=True)
    p = calibrator.predict(model.predict_proba(x_val)[:, 1])

    n_val = len(val)
    _, thr_pos = chronological_fraction_split(
        np.arange(n_val), cfg.simulation.calibration.cal_fraction
    )
    actions_all = enforce_capacity(
        val["timestamp"].dt.date.to_numpy(),
        p,
        val["amount"].to_numpy(float),
        amount_aware_actions(p, val["amount"].to_numpy(float), params),
        params,
        capacity,
    ).actions

    audit = load_audit_attributes(args.profile)
    audit_rows = val[["customer_id"]].merge(audit, on="customer_id", how="left")
    base = pd.DataFrame(
        {
            "customer_id": val["customer_id"].to_numpy(),
            "y": y_val,
            "p": p,
            "action": actions_all,
        }
    )
    for attribute in AUDIT_ATTRIBUTES:
        base[attribute] = audit_rows[attribute].to_numpy()
    primary = base.iloc[thr_pos].reset_index(drop=True)
    exploratory = base
    logger.info(
        "profile=%s val=%d val_thr=%d | audit attrs: %s",
        args.profile, n_val, len(thr_pos), ", ".join(AUDIT_ATTRIBUTES),
    )

    primary_table = group_table(
        primary, calibration_metrics=True, n_boot=n_boot, seed=seed
    )
    exploratory_table = group_table(
        exploratory, calibration_metrics=False, n_boot=n_boot, seed=seed
    )

    ratios = [
        {"attribute": attribute, **metric_ratios(sub, m)}
        for attribute in AUDIT_ATTRIBUTES
        for m in ("recall", "fpr_any", "flag_rate_any")
        for sub in [primary_table.metrics[primary_table.metrics["attribute"] == attribute]]
    ]

    pairwise_rows = []
    for attribute in AUDIT_ATTRIBUTES:
        groups = primary_table.metrics[primary_table.metrics["attribute"] == attribute]
        if groups.empty:
            continue
        reference = groups.sort_values("n", ascending=False).iloc[0]["group"]
        ref_mask = (primary[attribute] == reference).to_numpy()
        for _, row in groups.sort_values("n", ascending=False).iterrows():
            if row["group"] == reference or row["n"] < MIN_ROWS:
                continue
            mask = (primary[attribute] == row["group"]).to_numpy()
            for metric_name, metric_fn in PAIRED_METRICS:
                point, lo, hi = paired_group_diff(
                    primary["y"].to_numpy(),
                    primary["action"].to_numpy(),
                    primary["customer_id"].to_numpy(),
                    mask,
                    ref_mask,
                    metric_fn,
                    n_boot=n_boot,
                    seed=seed,
                )
                pairwise_rows.append(
                    {
                        "attribute": attribute,
                        "group": row["group"],
                        "reference": reference,
                        "metric": metric_name,
                        "point": point,
                        "lo": lo,
                        "hi": hi,
                        "distinguishable": bool(
                            np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
                        ),
                    }
                )
    pairwise = pd.DataFrame(pairwise_rows)
    logger.info(
        "pairwise differences distinguishable from noise: %d of %d",
        int(pairwise["distinguishable"].sum()) if len(pairwise) else 0, len(pairwise),
    )

    proxy = proxy_check(x_val, audit_rows[list(AUDIT_ATTRIBUTES)].reset_index(drop=True))
    logger.info(
        "proxy pairs over threshold: %d of %d", int(proxy["strong"].sum()), len(proxy)
    )

    out = results_dir()
    primary_table.metrics.to_csv(out / f"fairness_groups_primary_{args.profile}.csv", index=False)
    exploratory_table.metrics.to_csv(
        out / f"fairness_groups_exploratory_{args.profile}.csv", index=False
    )
    primary_table.suppression.to_csv(out / f"fairness_suppression_{args.profile}.csv", index=False)
    pd.DataFrame(ratios).to_csv(out / f"fairness_ratios_{args.profile}.csv", index=False)
    pairwise.to_csv(out / f"fairness_pairwise_{args.profile}.csv", index=False)
    proxy.to_csv(out / f"fairness_proxy_{args.profile}.csv", index=False)

    policy_line = (
        f"amount-aware policy `{policy['policy_version']}` (scenario `{policy['scenario']}`), "
        f"calibrator `{policy['selected_calibrator']}`, capacity {capacity}/day. Alert = "
        f"`action != APPROVE`; this fires on APPROVE vs STEP_UP/REVIEW/BLOCK."
    )
    primary_note = (
        "Primary view — this is the slice the policy and its capacity were selected on "
        f"({len(thr_pos):,} rows, {int(primary['y'].sum())} frauds), so per-group metrics are "
        "the honest operating numbers of the frozen policy."
    )
    (docs_dir() / "fairness_audit.md").write_text(
        render_fairness_markdown(
            profile=args.profile,
            primary_note=primary_note,
            primary=primary_table.metrics,
            primary_suppression=primary_table.suppression,
            secondary=exploratory_table.metrics,
            ratios=ratios,
            pairwise=pairwise,
            proxy=proxy,
            policy_line=policy_line,
        ),
        encoding="utf-8",
    )

    drop_experiment_rows("E18", args.profile)
    log_experiment(
        {
            "experiment_id": "E18",
            "part": "10",
            "dataset_profile": args.profile,
            "model": f"xgboost+calibration ({version})",
            "features_version": FEATURE_VERSION,
            "hyperparameters": {"n_boot": n_boot, "audit_attributes": len(AUDIT_ATTRIBUTES)},
            "calibration": policy["selected_calibrator"],
            "threshold_policy": f"amount_aware_capacity_{capacity}",
            "metrics": {
                "primary_rows": int(len(primary)),
                "primary_n_fraud": int(primary["y"].sum()),
                "strong_proxy_pairs": int(proxy["strong"].sum()),
                "distinguishable_pairs": int(pairwise["distinguishable"].sum())
                if len(pairwise)
                else 0,
            },
            "notes": "subgroup audit on val_thr (primary) + full validation (exploratory)",
        },
        config_hash=chash,
    )
    logger.info("wrote Part 10 fairness outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
