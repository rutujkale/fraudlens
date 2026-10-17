"""Shared plumbing for the Part 08 experiments (E10–E14).

Each ``scripts/run_experiment_8X.py`` holds the experiment logic; this module
holds only what all of them repeat: loading the feature matrix once, fitting the
*fixed* Part 06 XGBoost configuration (so the only thing that changes is the
factor under study), reducing multi-seed runs to mean ± std, and the
before/after-sigmoid calibration block for the imbalance experiment.

Labels (``is_fraud``, ``fraud_scenario``, ``label_available_at``) are used here
for training targets and evaluation only — never as features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraudlens.config import Config, load_config
from fraudlens.evaluation.bootstrap import cluster_bootstrap
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.metrics import brier, ece, evaluate_scores, pr_auc, recall_at_top_k
from fraudlens.evaluation.splits import Split, make_temporal_split
from fraudlens.models.calibration import fit_calibrators
from fraudlens.models.xgb import encode_features
from fraudlens.paths import artifacts_dir, results_dir

LABEL_COLUMNS = ["transaction_id", "is_fraud", "fraud_scenario", "label_available_at"]

# Part 06 best dev XGBoost config, used when a profile has no tuned artifact
# (e.g. drift_lab): keeps every experiment on an identical, already-chosen model.
FALLBACK_XGB_PARAMS: dict[str, float | int] = {
    "scale_pos_weight": 50.9847,
    "max_depth": 7,
    "learning_rate": 0.04228,
    "min_child_weight": 3.0,
    "subsample": 0.971,
    "colsample_bytree": 0.699,
    "reg_alpha": 0.602,
    "reg_lambda": 2.443,
}


@dataclass
class Supervised:
    """Feature matrix + encoded design matrix + labels + temporal split."""

    profile: str
    cfg: Config
    df: pd.DataFrame
    x: pd.DataFrame
    y: np.ndarray
    split: Split

    @property
    def groups(self) -> np.ndarray:
        return self.df["customer_id"].to_numpy()

    def x_train(self) -> pd.DataFrame:
        return self.x.iloc[self.split.train_idx]

    def y_train(self) -> np.ndarray:
        return self.y[self.split.train_idx]

    def x_val(self) -> pd.DataFrame:
        return self.x.iloc[self.split.val_idx]

    def y_val(self) -> np.ndarray:
        return self.y[self.split.val_idx]

    def val_scenario(self) -> np.ndarray:
        return self.df["fraud_scenario"].to_numpy()[self.split.val_idx]

    def val_groups(self) -> np.ndarray:
        return self.groups[self.split.val_idx]


def load_supervised(profile: str) -> Supervised:
    """Load features+keys+labels, build the temporal split, encode features once."""
    cfg = load_config(profile)
    df, labels = load_feature_dataset(profile)
    df = df.merge(labels[LABEL_COLUMNS], on="transaction_id", how="left")
    split = make_temporal_split(df, cfg)
    x = encode_features(df)
    y = df["is_fraud"].to_numpy(int)
    return Supervised(profile=profile, cfg=cfg, df=df, x=x, y=y, split=split)


def _tuned_metadata(profile: str, model: str) -> dict | None:
    """Metadata for ``model`` on ``profile`` if the artifact is present locally.

    ``results/models_<profile>.json`` is tracked but ``artifacts/`` is gitignored, so a
    fresh clone (e.g. CI) has the pointer without the file; fall through to the search
    log / default space rather than raising.
    """
    versions_path = results_dir() / f"models_{profile}.json"
    if not versions_path.exists():
        return None
    versions = json.loads(versions_path.read_text(encoding="utf-8"))
    if model not in versions:
        return None
    meta_path = artifacts_dir(versions[model]) / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def best_xgb_params(profile: str) -> dict[str, float | int]:
    """The Part 06 selected XGBoost config for ``profile`` (artifact, else search log).

    Falls back to :data:`FALLBACK_XGB_PARAMS` for profiles with no trained model.
    """
    metadata = _tuned_metadata(profile, "xgb")
    if metadata is not None:
        return dict(metadata["hyperparameters"])
    search_path = results_dir() / f"xgb_search_{profile}.csv"
    if search_path.exists():
        row = pd.read_csv(search_path).sort_values("val_pr_auc", ascending=False).iloc[0]
        keys = [
            "scale_pos_weight",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        ]
        params = {k: row[k] for k in keys}
        params["max_depth"] = int(params["max_depth"])
        return params
    return dict(FALLBACK_XGB_PARAMS)


def best_logreg_params(profile: str) -> dict:
    """The Part 06 selected logistic-regression config for ``profile``."""
    metadata = _tuned_metadata(profile, "logreg")
    if metadata is not None:
        return dict(metadata["hyperparameters"])
    return {"C": 1.0, "class_weight": None}


def fit_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, float | int],
    seed: int,
    n_estimators: int = 800,
    early_stopping_rounds: int = 30,
) -> XGBClassifier:
    """Fit XGBoost with a fixed config + early stopping (auprc) on the val window."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def predict_scores(model: XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def score_row(
    config: str,
    y: np.ndarray,
    score: np.ndarray,
    cfg: Config,
    groups: np.ndarray | None = None,
    *,
    with_ci: bool = False,
    seed: int = 42,
) -> dict:
    """One metrics row (PR-AUC, ROC-AUC, Brier, ECE, best-F1, recall@1%) + optional CI."""
    m = evaluate_scores(y, score, cfg.simulation.ks, cfg.simulation.precision_targets)
    row: dict = {
        "config": config,
        "n": int(len(y)),
        "n_fraud": int(np.asarray(y).sum()),
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "brier": m["brier"],
        "ece": m["ece"],
        "best_f1": m["best_f1"],
        "recall_at_1pct": recall_at_top_k(y, score, 0.01),
    }
    if with_ci and groups is not None:
        _, lo, hi = cluster_bootstrap(
            y, score, groups, pr_auc, n=cfg.simulation.bootstrap.n, seed=seed
        )
        row["pr_auc_lo"], row["pr_auc_hi"] = lo, hi
    return row


def mean_std(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    """Group ``df`` by ``config`` and return mean/std for ``cols`` as flat columns."""
    agg = df.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def calibration_before_after(
    y_cal: np.ndarray,
    raw_cal: np.ndarray,
    y_thr: np.ndarray,
    raw_thr: np.ndarray,
    bins: int = 10,
) -> dict:
    """Sigmoid fit on ``(raw_cal, y_cal)``; Brier/ECE on ``(raw_thr, y_thr)`` before/after."""
    calibrator = fit_calibrators(raw_cal, y_cal, names=("sigmoid",))["sigmoid"]
    calibrated = calibrator.predict(raw_thr)
    return {
        "brier_before": brier(y_thr, raw_thr),
        "brier_after": brier(y_thr, calibrated),
        "ece_before": ece(y_thr, raw_thr, bins),
        "ece_after": ece(y_thr, calibrated, bins),
    }
