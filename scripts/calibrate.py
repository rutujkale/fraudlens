"""Thin CLI: fit calibrators on val_cal, select on val_thr (Part 07A).

The validation window is split chronologically (never randomly): the first
``calibration.cal_fraction`` fits the calibrators, the rest selects the calibrator
and evaluates calibration quality. The test window is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from fraudlens.config import config_hash, load_config
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.experiments import log_experiment
from fraudlens.evaluation.figures import plot_reliability
from fraudlens.evaluation.metrics import evaluate_scores
from fraudlens.evaluation.scores import load_model_versions, raw_val_scores
from fraudlens.evaluation.splits import chronological_fraction_split, make_temporal_split
from fraudlens.features.registry import FEATURE_VERSION
from fraudlens.models.calibration import (
    calibration_bins,
    calibration_table,
    choose_calibrator,
    fit_calibrators,
)
from fraudlens.models.io import assert_artifact_profile
from fraudlens.paths import artifacts_dir, figures_dir, results_dir
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)

CALIBRATED_MODELS = ("xgb", "logreg")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and select probability calibrators.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    df, labels = load_feature_dataset(args.profile)
    label_cols = labels.set_index("transaction_id")
    df["is_fraud"] = df["transaction_id"].map(label_cols["is_fraud"])
    split = make_temporal_split(df, cfg)

    raw = raw_val_scores(df, split, cfg)
    y_val = df.iloc[split.val_idx]["is_fraud"].to_numpy(int)
    cust_val = df.iloc[split.val_idx]["customer_id"].to_numpy()
    n_val = len(split.val_idx)
    cal_pos, thr_pos = chronological_fraction_split(
        np.arange(n_val), cfg.simulation.calibration.cal_fraction
    )
    logger.info(
        "val_cal rows=%d frauds=%d | val_thr rows=%d frauds=%d",
        len(cal_pos), int(y_val[cal_pos].sum()), len(thr_pos), int(y_val[thr_pos].sum()),
    )

    bins = cfg.simulation.calibration.bins
    versions = load_model_versions(args.profile)
    rows: list[dict] = []
    selections: dict[str, dict] = {}
    reliability: dict[str, tuple[float, list[dict]]] = {}

    for model in CALIBRATED_MODELS:
        raw_cal, raw_thr = raw[model][cal_pos], raw[model][thr_pos]
        y_cal, y_thr = y_val[cal_pos], y_val[thr_pos]
        calibrators = fit_calibrators(raw_cal, y_cal)
        selection = choose_calibrator(
            y_thr, raw_thr, calibrators, cust_val[thr_pos],
            n_cal_positives=int(y_cal.sum()),
            min_cal_positives=cfg.simulation.calibration.min_cal_positives_flag,
            n_boot=cfg.simulation.bootstrap.n,
            seed=cfg.simulation.seed,
        )
        selections[model] = selection
        logger.info("%s calibrator -> %s (%s)", model, selection["selected"], selection["reason"])

        for row in calibration_table(y_thr, raw_thr, calibrators, bins):
            rows.append({"model": model, **row})

        chosen = calibrators[selection["selected"]]
        reliability[f"{model}_raw"] = (float(y_thr.mean()), calibration_bins(y_thr, raw_thr, bins))
        reliability[f"{model}_{selection['selected']}"] = (
            float(y_thr.mean()), calibration_bins(y_thr, chosen.predict(raw_thr), bins)
        )

        artifact = artifacts_dir(versions[model])
        assert_artifact_profile(versions[model], args.profile)
        joblib.dump(chosen, artifact / "calibrator.joblib")
        (artifact / "calibration.json").write_text(
            json.dumps({"model": model, "selected": selection["selected"], **selection},
                       indent=2, default=str),
            encoding="utf-8",
        )

    out = pd.DataFrame(rows)
    out.to_csv(results_dir() / f"calibration_{args.profile}.csv", index=False)
    (results_dir() / f"calibration_selection_{args.profile}.json").write_text(
        json.dumps(selections, indent=2, default=str), encoding="utf-8"
    )
    plot_reliability(reliability, figures_dir() / f"calibration_{args.profile}.png")
    logger.info("calibration table:\n%s", out.to_string(index=False))

    xgb_cal = joblib.load(artifacts_dir(versions["xgb"]) / "calibrator.joblib")
    p_xgb_thr = xgb_cal.predict(raw["xgb"][thr_pos])
    y_thr = y_val[thr_pos]
    if np.unique(y_thr).size < 2:
        logger.warning(
            "val_thr has a single class (%d positives); skipping E04 metrics", int(y_thr.sum())
        )
        return 0
    metrics = evaluate_scores(
        y_thr, p_xgb_thr, cfg.simulation.ks, cfg.simulation.precision_targets
    )
    log_experiment(
        {
            "experiment_id": "E04",
            "part": "07",
            "dataset_profile": args.profile,
            "model": "xgboost+calibration",
            "features_version": FEATURE_VERSION,
            "train_window": {
                "start": split.boundaries["t0"],
                "end": split.boundaries["t_train_end"],
            },
            "val_window": {
                "start": split.boundaries["t_train_end"],
                "end": split.boundaries["t_val_end"],
            },
            "hyperparameters": {"calibrator": selections["xgb"]["selected"]},
            "calibration": selections["xgb"]["selected"],
            "threshold_policy": "f1_optimal",
            "metrics": {
                "pr_auc": round(float(metrics["pr_auc"]), 6),
                "roc_auc": round(float(metrics["roc_auc"]), 6),
                "brier": round(float(metrics["brier"]), 6),
                "ece": round(float(metrics["ece"]), 6),
                "best_f1": round(float(metrics["best_f1"]), 6),
                "best_f1_threshold": round(float(metrics["best_f1_threshold"]), 6),
            },
            "notes": "xgb calibrated on val_cal, F1 threshold selected on val_thr",
        },
        config_hash=chash,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
