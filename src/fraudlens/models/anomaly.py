"""Isolation Forest anomaly scoring (Part 09).

An unsupervised detector over a curated numeric **behavioural** subset (no IDs,
no labels, no tenure/clock features). Used two ways in the unseen-fraud
experiment: as a standalone ranker, and as an extra supervised feature — which
requires **customer-grouped cross-fitting** so a training row is never scored by
a forest fit on its own customer (a customer's rows are highly correlated, so an
in-sample score would leak the label direction into training).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from fraudlens.evaluation.metrics import pr_auc

# Curated behavioural subset: amount deviation, velocity/counts, amount sums,
# near-threshold counts, distance/travel, hour deviation, device/recipient
# novelty. Excludes identifiers, categoricals, and monotonic tenure features
# (history_days, n_prior_tx, device_age_days, tx_from_device_prior) since the
# last two would let the detector trivially separate old from new customers.
ANOMALY_FEATURES: list[str] = [
    "log_amount",
    "amount_ratio_to_7d_mean",
    "amount_ratio_to_30d_mean",
    "amount_zscore_30d",
    "amount_vs_lifetime_median",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_6h",
    "tx_count_24h",
    "tx_count_7d",
    "velocity_ratio_1h_vs_24h",
    "minutes_since_prev_tx",
    "amount_sum_1h",
    "amount_sum_24h",
    "max_amount_24h",
    "count_near_threshold_24h",
    "distance_from_home_km",
    "distance_from_prev_km",
    "travel_speed_kmh",
    "hour_deviation",
    "is_new_device",
    "device_age_days",
    "tx_from_device_prior",
    "accounts_seen_on_device",
    "device_count_30d",
    "is_new_recipient",
    "unique_recipients_1h",
    "unique_recipients_24h",
]

# Small hyperparameter grid; selected on validation PR-AUC of the IF-alone score
# on non-held-out fraud only (never on the held-out scenario).
IF_GRID: list[dict[str, int]] = [
    {"n_estimators": 200, "max_samples": 256},
    {"n_estimators": 200, "max_samples": 512},
    {"n_estimators": 300, "max_samples": 256},
    {"n_estimators": 300, "max_samples": 512},
    {"n_estimators": 400, "max_samples": 1024},
]


@dataclass
class AnomalyModel:
    """Median-impute + Isolation Forest with a rank-normalised score in ``[0, 1]``."""

    features: list[str]
    imputer: SimpleImputer
    forest: IsolationForest
    train_raw_sorted: np.ndarray
    params: dict
    seed: int
    n_fit_rows: int

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        """Negated ``score_samples`` on the imputed design (higher = more anomalous)."""
        x = self.imputer.transform(df[self.features])
        return -self.forest.score_samples(x)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Empirical-CDF rank of ``raw_score`` within the training distribution."""
        raw = self.raw_score(df)
        rank = np.searchsorted(self.train_raw_sorted, raw, side="right")
        return rank / len(self.train_raw_sorted)


def fit_anomaly(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    n_estimators: int = 300,
    max_samples: int = 256,
    contamination: str | float = "auto",
    seed: int = 42,
    exclude_mask: np.ndarray | None = None,
) -> AnomalyModel:
    """Fit the IF on ``df`` (minus ``exclude_mask`` rows) and cache train scores.

    ``exclude_mask`` is how the strict variant is built: held-out (structuring)
    rows are dropped from the *fit* but the returned model can still score them.
    """
    features = list(features) if features is not None else list(ANOMALY_FEATURES)
    fit_df = df if exclude_mask is None else df[~np.asarray(exclude_mask, dtype=bool)]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x = imputer.fit_transform(fit_df[features])
    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(x)
    raw = -forest.score_samples(x)
    return AnomalyModel(
        features=features,
        imputer=imputer,
        forest=forest,
        train_raw_sorted=np.sort(raw),
        params={
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
        },
        seed=seed,
        n_fit_rows=int(len(x)),
    )


def customer_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Assign every row to a fold; all rows of one customer share the same fold."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_of = {g: i % n_splits for i, g in enumerate(shuffled)}
    return np.array([fold_of[g] for g in groups], dtype=int)


def cross_fit_scores(
    df: pd.DataFrame,
    groups: np.ndarray,
    *,
    features: list[str] | None = None,
    n_estimators: int = 300,
    max_samples: int = 256,
    contamination: str | float = "auto",
    seed: int = 42,
    n_splits: int = 5,
    fit_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-fold IF scores: each row is scored by a forest that never saw its customer.

    ``fit_exclude_mask`` additionally removes rows (e.g. held-out structuring
    episodes) from every fold's fit — the strict variant.
    """
    groups = np.asarray(groups)
    folds = customer_folds(groups, n_splits=n_splits, seed=seed)
    exclude = None if fit_exclude_mask is None else np.asarray(fit_exclude_mask, dtype=bool)
    scores = np.empty(len(df), dtype=float)
    for f in range(n_splits):
        fit = folds != f
        if exclude is not None:
            fit = fit & ~exclude
        apply = folds == f
        model = fit_anomaly(
            df.iloc[fit],
            features=features,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            seed=seed,
        )
        scores[apply] = model.score(df.iloc[apply])
    return scores


def grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    grid: list[dict[str, int]] | None = None,
    features: list[str] | None = None,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Pick IF hyperparameters by val PR-AUC of the IF-alone score.

    ``y_val`` must encode **non-held-out** fraud only (held-out rows dropped), so
    the selection never touches the held-out scenario's labels.
    """
    grid = grid or IF_GRID
    rows = []
    for params in grid:
        model = fit_anomaly(df_train, features=features, seed=seed, **params)
        rows.append({**params, "val_pr_auc": pr_auc(y_val, model.score(df_val))})
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {
        "n_estimators": int(table.iloc[0]["n_estimators"]),
        "max_samples": int(table.iloc[0]["max_samples"]),
    }
    return best, table
