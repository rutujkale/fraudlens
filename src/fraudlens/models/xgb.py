"""XGBoost model: seeded random search + early stopping (Part 06F).

Selection is by **validation PR-AUC**; early stopping also uses the validation
``aucpr`` (so the validation window is used for both — stated in the progress
file). ``scale_pos_weight`` candidates include the discrete tokens one/sqrt/full
(1, sqrt(neg/pos), neg/pos) plus a continuous range. Native NaN handling; the
categorical features are label-encoded with a fixed, train/val-consistent map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraudlens.config import XgbConfig
from fraudlens.data_gen.archetypes import CATEGORIES, CHANNELS, TRANSACTION_TYPES
from fraudlens.evaluation.metrics import pr_auc
from fraudlens.features.registry import FEATURE_COLUMNS

# IEEE-CIS ``ProductCD`` levels (Part 11). Appended after the synthetic category
# codes so the synthetic encoding is unchanged; only the external baseline (R1)
# and native reference (R4) use them. R2 excludes categoricals entirely.
IEEE_PRODUCT_CD = ["W", "C", "R", "H", "S"]

_CAT_MAPS: dict[str, dict[str, int]] = {
    "transaction_type": {v: i for i, v in enumerate(TRANSACTION_TYPES)},
    "channel": {v: i for i, v in enumerate(CHANNELS)},
    "merchant_category": {v: i for i, v in enumerate(CATEGORIES)}
    | {v: i + len(CATEGORIES) for i, v in enumerate(IEEE_PRODUCT_CD)},
}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``FEATURE_COLUMNS`` with categoricals label-encoded (unknown -> NaN)."""
    x = df[FEATURE_COLUMNS].copy()
    for col, mapping in _CAT_MAPS.items():
        x[col] = x[col].map(mapping).astype(float)
    return x


def scale_pos_weight_value(mode: str, neg_pos: float) -> float:
    """Map a mode token to its weight: one=1, sqrt=sqrt(neg/pos), full=neg/pos."""
    if mode == "one":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(neg_pos))
    if mode == "full":
        return float(neg_pos)
    raise ValueError(f"unknown scale_pos_weight mode {mode!r}")


def _sample_params(
    rng: np.random.Generator, space: dict[str, list[float]]
) -> dict[str, float | int]:
    params: dict[str, float] = {}
    for name, (lo, hi) in space.items():
        if name == "max_depth":
            params[name] = int(rng.integers(int(lo), int(hi) + 1))
        elif name == "min_child_weight":
            params[name] = float(rng.integers(int(lo), int(hi) + 1))
        else:
            params[name] = float(rng.uniform(lo, hi))
    return params


def _scale_pos_weight_plan(
    rng: np.random.Generator, cfg: XgbConfig, neg_pos: float, n_trials: int
) -> list[float]:
    """First trials pin every discrete mode; the rest mix discrete and continuous."""
    modes = [scale_pos_weight_value(m, neg_pos) for m in cfg.scale_pos_weight_modes]
    lo, hi = cfg.scale_pos_weight_continuous
    plan = list(modes[:n_trials])
    while len(plan) < n_trials:
        if rng.random() < 0.5:
            plan.append(float(rng.choice(modes)))
        else:
            plan.append(float(rng.uniform(lo, hi)))
    return plan[:n_trials]


def random_search_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: XgbConfig,
    seed: int = 42,
) -> tuple[XGBClassifier, pd.DataFrame]:
    """Random search; returns the best model (by val PR-AUC) and every trial."""
    neg_pos = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    rng = np.random.default_rng(seed)
    weights = _scale_pos_weight_plan(rng, cfg, neg_pos, cfg.n_trials)

    rows, best_model, best_pr = [], None, -np.inf
    for i in range(cfg.n_trials):
        params = _sample_params(rng, cfg.param_space)
        spw = weights[i]
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            early_stopping_rounds=cfg.early_stopping_rounds,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=spw,
            random_state=seed + i,
            n_jobs=-1,
            **params,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        pr = pr_auc(y_val, model.predict_proba(x_val)[:, 1])
        rows.append(
            {
                "trial": i,
                "scale_pos_weight": spw,
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "val_pr_auc": pr,
                **params,
            }
        )
        if pr > best_pr:
            best_pr, best_model = pr, model
    results = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    return best_model, results


def gain_importances(model: XGBClassifier) -> pd.DataFrame:
    """Top gain importances with shares (descending)."""
    gains = model.get_booster().get_score(importance_type="gain")
    total = sum(gains.values()) or 1.0
    rows = [
        {"feature": feat, "gain": float(val), "gain_share": float(val / total)}
        for feat, val in gains.items()
    ]
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def ablation_pr_auc(
    model: XGBClassifier, x_val: pd.DataFrame, y_val: np.ndarray, feature: str
) -> float:
    """Val PR-AUC after setting ``feature`` to NaN for all rows (no retraining)."""
    ablated = x_val.copy()
    ablated[feature] = np.nan
    return pr_auc(y_val, model.predict_proba(ablated)[:, 1])
