"""Thin CLI: train the three core models and select on validation (Part 06G).

Trains on the training window, tunes/selects on the validation window; the test
window is never touched. Writes the split, the rules doc, logreg coefficients,
the XGBoost search log, model artifacts and experiment rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from fraudlens.config import Config, config_hash, load_config
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.experiments import log_experiment
from fraudlens.evaluation.metrics import evaluate_scores
from fraudlens.evaluation.splits import Split, make_temporal_split, save_split
from fraudlens.features.registry import FEATURE_COLUMNS, FEATURE_VERSION
from fraudlens.models import io as model_io
from fraudlens.models.logreg import save_logreg_coefficients, tune_logreg
from fraudlens.models.rules import RuleEngine, write_rules_doc
from fraudlens.models.xgb import (
    ablation_pr_auc,
    encode_features,
    gain_importances,
    random_search_xgb,
)
from fraudlens.paths import docs_dir, results_dir
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)

SUSPICIOUS_PR_AUC = 0.95
DOMINANT_GAIN_SHARE = 0.60


def _attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_fraud"] = df["transaction_id"].map(
        labels.set_index("transaction_id")["is_fraud"]
    )
    return df


def _windows(split: Split) -> tuple[dict[str, str], dict[str, str]]:
    b = split.boundaries
    train = {"start": b["t0"], "end": b["t_train_end"]}
    val = {"start": b["t_train_end"], "end": b["t_val_end"]}
    return train, val


def _compact(metrics: dict) -> dict:
    keys = ("pr_auc", "roc_auc", "brier", "ece", "best_f1")
    out = {k: round(float(metrics[k]), 6) for k in keys if k in metrics}
    for k, v in metrics.items():
        if k.startswith("recall_at_top") and not np.isnan(v):
            out[k] = round(float(v), 6)
    return out


def _suspicious_check(
    model, x_val: pd.DataFrame, y_val: np.ndarray, best_pr: float, profile: str
) -> dict:
    """Top-5 gain importances + is_new_device ablation, written to results/."""
    imp = gain_importances(model).head(5)
    top5 = imp.to_dict(orient="records")
    full = float(best_pr)
    ablated = float(ablation_pr_auc(model, x_val, y_val, "is_new_device"))
    diagnostic = {
        "profile": profile,
        "val_pr_auc": full,
        "top5_gain": top5,
        "dominant_feature": next(
            (t["feature"] for t in top5 if t["gain_share"] > DOMINANT_GAIN_SHARE), None
        ),
        "is_new_device_ablation": {"with": full, "without": ablated, "drop": full - ablated},
    }
    out = results_dir()
    (out / f"xgb_importance_{profile}.json").write_text(
        json.dumps(diagnostic, indent=2, default=str), encoding="utf-8"
    )
    imp.assign(**{}).to_csv(out / f"xgb_importance_{profile}.csv", index=False)
    logger.warning(
        "SUSPICIOUS: val PR-AUC %.4f > %.2f. Top-5 gain: %s",
        full,
        SUSPICIOUS_PR_AUC,
        ", ".join(f"{t['feature']}={t['gain_share']:.3f}" for t in top5),
    )
    logger.warning(
        "is_new_device ablation: %.4f -> %.4f (drop %.4f)",
        full,
        ablated,
        full - ablated,
    )
    return diagnostic


def train_rules(df, split, cfg: Config, chash: str) -> np.ndarray:
    train, val = _windows(split)
    y = df["is_fraud"].to_numpy(int)
    train_legit = df.iloc[split.train_idx][df.iloc[split.train_idx]["is_fraud"] == 0]
    engine = RuleEngine(cfg.simulation.rules).fit(
        train_legit[FEATURE_COLUMNS]
    )
    write_rules_doc(engine, cfg.simulation.profile, docs_dir() / "rules_baseline.md")
    score = engine.score(df.iloc[split.val_idx][FEATURE_COLUMNS]).to_numpy()
    metrics = evaluate_scores(
        y[split.val_idx], score, cfg.simulation.ks, cfg.simulation.precision_targets
    )
    log_experiment(
        {
            "experiment_id": "E01",
            "part": "06",
            "dataset_profile": cfg.simulation.profile,
            "model": "rules",
            "features_version": FEATURE_VERSION,
            "train_window": train,
            "val_window": val,
            "calibration": "none",
            "threshold_policy": "rank_by_score",
            "metrics": _compact(metrics),
            "notes": "weighted rule score in [0,1]; thresholds from training-legit percentiles",
        },
        config_hash=chash,
    )
    logger.info("rules: val PR-AUC=%.4f", metrics["pr_auc"])
    return score


def train_logreg(df, split, cfg: Config, chash: str) -> tuple[np.ndarray, str]:
    train, val = _windows(split)
    y = df["is_fraud"].to_numpy(int)
    x = df[FEATURE_COLUMNS]
    lr_cfg = cfg.simulation.models.logreg
    model, _tuning = tune_logreg(
        x.iloc[split.train_idx],
        y[split.train_idx],
        x.iloc[split.val_idx],
        y[split.val_idx],
        lr_cfg.C_grid,
        lr_cfg.class_weights,
    )
    score = model.predict_proba(x.iloc[split.val_idx])[:, 1]
    metrics = evaluate_scores(
        y[split.val_idx], score, cfg.simulation.ks, cfg.simulation.precision_targets
    )
    save_logreg_coefficients(model, results_dir() / "logreg_coefficients.csv")
    version = f"logreg_{cfg.simulation.profile}_{datetime.now():%Y%m%d}_v1"
    metadata = model_io.build_metadata(
        model_version=version,
        feature_version=FEATURE_VERSION,
        profile=cfg.simulation.profile,
        train_window=train,
        val_window=val,
        calibration="none",
        hyperparameters={
            "C": float(model.named_steps["clf"].C),
            "class_weight": model.named_steps["clf"].class_weight,
        },
        seed=cfg.simulation.seed,
        metrics=_compact(metrics),
        config_hash=chash,
    )
    val_scores = df.iloc[split.val_idx][["transaction_id", "customer_id"]].copy()
    val_scores["raw_score"] = score
    model_io.save_model(model, version, metadata, val_scores)
    log_experiment(
        {
            "experiment_id": "E02",
            "part": "06",
            "dataset_profile": cfg.simulation.profile,
            "model": "logreg",
            "features_version": FEATURE_VERSION,
            "train_window": train,
            "val_window": val,
            "hyperparameters": metadata["hyperparameters"],
            "calibration": "none",
            "threshold_policy": "rank_by_score",
            "metrics": _compact(metrics),
            "notes": "L2 logistic regression; impute+log1p+scale+one-hot",
        },
        config_hash=chash,
    )
    logger.info("logreg: val PR-AUC=%.4f", metrics["pr_auc"])
    return score, version


def train_xgb(df, split, cfg: Config, chash: str) -> tuple[np.ndarray, str, dict | None]:
    train, val = _windows(split)
    y = df["is_fraud"].to_numpy(int)
    x = encode_features(df)

    model, search = random_search_xgb(
        x.iloc[split.train_idx],
        y[split.train_idx],
        x.iloc[split.val_idx],
        y[split.val_idx],
        cfg.simulation.models.xgb,
        seed=cfg.simulation.seed,
    )
    search.to_csv(results_dir() / f"xgb_search_{cfg.simulation.profile}.csv", index=False)
    score = model.predict_proba(x.iloc[split.val_idx])[:, 1]
    best_pr = float(search.iloc[0]["val_pr_auc"])
    metrics = evaluate_scores(
        y[split.val_idx], score, cfg.simulation.ks, cfg.simulation.precision_targets
    )
    version = f"xgb_{cfg.simulation.profile}_{datetime.now():%Y%m%d}_v1"
    params = search.drop(columns=["trial", "val_pr_auc", "best_iteration"]).iloc[0].to_dict()
    metadata = model_io.build_metadata(
        model_version=version,
        feature_version=FEATURE_VERSION,
        profile=cfg.simulation.profile,
        train_window=train,
        val_window=val,
        calibration="none",
        hyperparameters={k: (int(v) if k == "max_depth" else float(v)) for k, v in params.items()},
        seed=cfg.simulation.seed,
        metrics=_compact(metrics),
        config_hash=chash,
    )
    val_scores = df.iloc[split.val_idx][["transaction_id", "customer_id"]].copy()
    val_scores["raw_score"] = score
    model_io.save_model(model, version, metadata, val_scores)
    log_experiment(
        {
            "experiment_id": "E03",
            "part": "06",
            "dataset_profile": cfg.simulation.profile,
            "model": "xgboost",
            "features_version": FEATURE_VERSION,
            "train_window": train,
            "val_window": val,
            "hyperparameters": metadata["hyperparameters"],
            "calibration": "none",
            "threshold_policy": "rank_by_score",
            "metrics": _compact(metrics),
            "notes": f"random search n={len(search)}; early stop on val aucpr",
        },
        config_hash=chash,
    )
    logger.info("xgb: val PR-AUC=%.4f", best_pr)

    diagnostic = None
    if best_pr > SUSPICIOUS_PR_AUC:
        diagnostic = _suspicious_check(
            model, x.iloc[split.val_idx], y[split.val_idx], best_pr, cfg.simulation.profile
        )
    return score, version, diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FraudLens core models.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--model", default="all", choices=["rules", "logreg", "xgb", "all"])
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    df, labels = load_feature_dataset(args.profile)
    df = _attach_labels(df, labels)
    split = make_temporal_split(df, cfg)
    save_split(split, args.profile)
    logger.info(
        "split: train=%d val=%d test=%d (test untouched)",
        len(split.train_idx),
        len(split.val_idx),
        len(split.test_idx),
    )

    todo = ["rules", "logreg", "xgb"] if args.model == "all" else [args.model]
    diagnostic = None
    versions: dict[str, str] = {}
    if "rules" in todo:
        train_rules(df, split, cfg, chash)
    if "logreg" in todo:
        _, versions["logreg"] = train_logreg(df, split, cfg, chash)
    if "xgb" in todo:
        _, versions["xgb"], diagnostic = train_xgb(df, split, cfg, chash)
        logger.info("xgb artifact: %s", versions["xgb"])
    if versions:
        (results_dir() / f"models_{args.profile}.json").write_text(
            json.dumps(versions, indent=2), encoding="utf-8"
        )

    if diagnostic is not None:
        logger.warning(
            "STOP: XGBoost val PR-AUC %.4f > %.2f — inspect %s and %s before continuing.",
            diagnostic["val_pr_auc"],
            SUSPICIOUS_PR_AUC,
            f"results/xgb_importance_{args.profile}.json",
            "docs/what_went_wrong.md",
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
