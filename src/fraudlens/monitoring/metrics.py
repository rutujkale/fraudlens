"""Part 15A — per-period monitoring over the **train + validation** timeline only.

The test window is closed: the monitoring run reads no test-window features or
labels. Every row used here satisfies ``timestamp < t_val_end`` (the embargo seat
included), and ``monitoring_window`` is the single gate that enforces it. The
test-window drift figure that *is* published is quoted from
``results/final_results.json`` (``drift.*``), not recomputed here.

Alert bands (conventional rules of thumb, same as ``drift.py``): PSI < 0.1 stable,
0.1-0.25 monitor, > 0.25 significant shift. They are heuristics, not thresholds
derived from this data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraudlens.config import load_config
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.metrics import (
    fpr_at_threshold,
    pr_auc,
    precision_recall_f1_at_threshold,
)
from fraudlens.evaluation.splits import make_temporal_split
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.models.xgb import encode_features
from fraudlens.monitoring.drift import PSI_MONITOR, PSI_SHIFT, psi, psi_flag
from fraudlens.paths import results_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

# Named key features (Part 15A) — the ones the business reasons about.
KEY_FEATURES = [
    "amount_ratio_to_30d_mean",
    "tx_count_1h",
    "is_new_device",
    "distance_from_home_km",
    "amount_sum_24h",
]


def period_of(ts: object) -> str:
    """Calendar month bucket, ``YYYY-MM``."""
    return pd.Timestamp(ts).strftime("%Y-%m")


def monitoring_window(df: pd.DataFrame, t_val_end: object) -> pd.DataFrame:
    """Restrict ``df`` to ``timestamp < t_val_end`` (train + validation only).

    This is the only place the monitoring scope is decided; if a future caller
    widened it, this function would have to change, and the test that the max
    timestamp stays before the validation end would fail.
    """
    t_val_end = pd.Timestamp(t_val_end)
    kept = df[df["timestamp"] < t_val_end]
    if len(kept) and kept["timestamp"].max() >= t_val_end:
        raise AssertionError("monitoring scope leaked rows at/after the validation end")
    return kept


def split_boundaries(profile: str) -> dict[str, str]:
    """Temporal-split boundaries (``t0`` / ``t_train_end`` / ``t_val_end`` / ...)."""
    df, _ = load_feature_dataset(profile)
    return make_temporal_split(df, load_config(profile)).boundaries


def top_shap_features(profile: str, k: int = 15) -> list[str]:
    """Top-``k`` mean|SHAP| names from ``results/shap_global_<profile>.csv``.

    Falls back to :data:`KEY_FEATURES` when the SHAP table is absent, so the
    monitoring run degrades gracefully instead of inventing a feature list.
    """
    path = results_dir() / f"shap_global_{profile}.csv"
    if not path.exists():
        logger.warning("%s missing; falling back to KEY_FEATURES", path.name)
        return list(KEY_FEATURES)
    table = pd.read_csv(path).sort_values("mean_abs_shap", ascending=False)
    names = [f for f in table["feature"].tolist() if f in FEATURE_COLUMNS]
    return names[:k]


def psi_by_period(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    *,
    period_col: str = "period",
) -> pd.DataFrame:
    """Long-format PSI per (period, feature) against a fixed ``reference`` window."""
    rows: list[dict[str, object]] = []
    for period, chunk in current.groupby(period_col, sort=True):
        for feature in features:
            value = psi(
                reference[feature].to_numpy(dtype=float),
                chunk[feature].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "period": str(period),
                    "feature": feature,
                    "psi": value,
                    "flag": psi_flag(value),
                    "n": int(len(chunk)),
                }
            )
    return pd.DataFrame(rows)


def _frozen_threshold(profile: str) -> float:
    """Frozen global cutoff (the capacity threshold) from ``results/freeze.json``."""
    path = results_dir() / "freeze.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        thresholds = frozen.get("frozen_thresholds", {})
        value = thresholds.get("cost_optimal_threshold_capacity") or thresholds.get(
            "cost_optimal_threshold"
        )
        if value is not None:
            return float(value)
    final = results_dir() / "final_results.json"
    if final.exists():
        meta = json.loads(final.read_text(encoding="utf-8"))["decision"]["meta"]
        return float(meta["cost_optimal_threshold_capacity"])
    raise FileNotFoundError("no frozen threshold found (results/freeze.json or final_results.json)")


def _calibrated_scores(profile: str, df: pd.DataFrame, model_name: str) -> tuple[np.ndarray, str]:
    """Frozen model + calibrator probabilities for ``df`` (raw scores if uncalibrated)."""
    versions = json.loads((results_dir() / f"models_{profile}.json").read_text("utf-8"))
    version = versions[model_name]
    bundle = model_io.load_serving_bundle(version)
    raw = bundle["model"].predict_proba(encode_features(df))[:, 1]
    calibrator = bundle["calibrator"]
    prob = calibrator.predict(raw) if calibrator is not None else raw
    return np.asarray(prob, dtype=float), version


def _period_metrics(period_df: pd.DataFrame, labels: pd.DataFrame, threshold: float) -> dict:
    """One period's label-matured metrics (prevalence, calibration, ranking, operating point)."""
    merged = period_df[["transaction_id"]].merge(labels, on="transaction_id", how="left")
    period_end = pd.Timestamp(period_df["timestamp"].max())
    matured = merged["label_available_at"] <= period_end
    n_rows = int(len(period_df))
    n_matured = int(matured.sum())
    y_all = merged["is_fraud"].to_numpy(dtype=float)
    y_mature = y_all[matured.to_numpy()]
    prob = period_df["score"].to_numpy(dtype=float)
    prob_mature = prob[matured.to_numpy()]

    out: dict[str, object] = {
        "n_rows": n_rows,
        "n_matured": n_matured,
        "maturity_share": n_matured / n_rows if n_rows else float("nan"),
        "prevalence_matured": float(np.nanmean(y_mature)) if n_matured else float("nan"),
        "mean_pred": float(np.mean(prob)) if n_rows else float("nan"),
        "observed_rate": float(np.nanmean(y_all)) if n_rows else float("nan"),
        "pr_auc": float("nan"),
        "recall_at_frozen": float("nan"),
        "precision_at_frozen": float("nan"),
        "fpr_at_frozen": float("nan"),
    }
    if n_matured and len(np.unique(y_mature)) == 2:
        out["pr_auc"] = pr_auc(y_mature, prob_mature)
        prf = precision_recall_f1_at_threshold(y_mature, prob_mature, threshold)
        out["recall_at_frozen"] = prf["recall"]
        out["precision_at_frozen"] = prf["precision"]
        out["fpr_at_frozen"] = fpr_at_threshold(y_mature, prob_mature, threshold)
    return out


def build_dashboard_metrics(
    profile: str,
    *,
    model_name: str = "xgb",
    top_shap: int = 15,
) -> dict[str, object]:
    """Per-period monitoring tables over train + validation only.

    Returns a dict with:
    * ``periods`` — one row per month (prevalence, calibration, delayed-label performance),
    * ``psi_by_period`` — long PSI per (period, feature),
    * ``score_psi_by_period`` — score (calibrated probability) PSI per period,
    * ``reference`` — the training-window reference summary,
    * ``threshold`` / ``model_version`` / ``t_val_end`` / ``features`` metadata.
    """
    cfg = load_config(profile)
    df, labels = load_feature_dataset(profile)
    split = make_temporal_split(df, cfg)
    t_val_end = split.boundaries["t_val_end"]

    window = monitoring_window(df, t_val_end).copy()
    prob, model_version = _calibrated_scores(profile, window, model_name)
    window["score"] = prob
    window["period"] = window["timestamp"].map(period_of)

    train = window[window["timestamp"] < pd.Timestamp(split.boundaries["t_train_end"])]
    train_labels = train[["transaction_id"]].merge(labels, on="transaction_id", how="left")
    threshold = _frozen_threshold(profile)

    features = [f for f in FEATURE_COLUMNS if f in window.columns]
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(window[f])]
    monitor_features = list(dict.fromkeys(KEY_FEATURES + top_shap_features(profile, top_shap)))

    psi_long = psi_by_period(train, window, [f for f in monitor_features if f in numeric_features])

    score_rows = []
    for period, chunk in window.groupby("period", sort=True):
        score_rows.append(
            {
                "period": str(period),
                "score_psi": psi(train["score"].to_numpy(), chunk["score"].to_numpy()),
                "n": int(len(chunk)),
            }
        )
    score_psi = pd.DataFrame(score_rows)
    if len(score_psi):
        score_psi["flag"] = score_psi["score_psi"].map(psi_flag)

    period_rows = []
    for period, chunk in window.groupby("period", sort=True):
        row = {"period": str(period), **_period_metrics(chunk, labels, threshold)}
        row["score_psi"] = float(
            score_psi.loc[score_psi["period"] == str(period), "score_psi"].iloc[0]
        )
        period_rows.append(row)
    periods = pd.DataFrame(period_rows)

    return {
        "periods": periods,
        "psi_by_period": psi_long,
        "score_psi_by_period": score_psi,
        "threshold": threshold,
        "model_version": model_version,
        "t_val_end": t_val_end,
        "features": monitor_features,
        "reference": {
            "window": f"train (timestamp < {split.boundaries['t_train_end']})",
            "n_rows": int(len(train)),
            "prevalence": float(train_labels["is_fraud"].mean()) if len(train) else float("nan"),
        },
        "bands": {"stable": PSI_MONITOR, "monitor": PSI_SHIFT},
        "note": (
            "train + validation timeline only; the published test-window PSI is quoted "
            "from results/final_results.json, not recomputed here."
        ),
    }


def retraining_trigger(
    score_psi: list[float] | pd.Series,
    recall: list[float] | pd.Series,
    *,
    psi_threshold: float = PSI_SHIFT,
    consecutive: int = 2,
    recall_floor: float = 0.5,
) -> dict[str, object]:
    """Pre-stated retraining rules (Part 15A), evaluated on a per-period series.

    Rules (documented before looking at any run):
    1. **score PSI** above ``psi_threshold`` for ``consecutive`` periods in a row;
    2. **recall at the frozen policy** below ``recall_floor`` in any period.

    Returns ``{"retrain": bool, "reasons": [...]}`` and never mutates inputs.
    """
    psi_series = [float(v) for v in score_psi]
    recall_series = [float(v) for v in recall]
    reasons: list[str] = []

    streak = 0
    for value in psi_series:
        streak = streak + 1 if value > psi_threshold else 0
        if streak >= consecutive:
            reasons.append(
                f"score PSI > {psi_threshold:.2f} for {consecutive} consecutive periods"
            )
            break

    violations = [
        i for i, value in enumerate(recall_series) if value == value and value < recall_floor
    ]
    if violations:
        reasons.append(f"recall at frozen policy < {recall_floor:.2f} in period(s) {violations}")

    return {"retrain": bool(reasons), "reasons": reasons}
