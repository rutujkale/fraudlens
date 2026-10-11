"""Logistic-regression baseline (Part 06E).

Pipeline: median impute + missing indicators (numeric) -> ``log1p`` on heavy-tailed
non-negative columns -> standardise, one-hot low-cardinality categoricals, then L2
logistic regression. ``C`` and ``class_weight`` are tuned on validation PR-AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraudlens.evaluation.metrics import pr_auc
from fraudlens.features.registry import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, get_spec

# Non-negative, heavy-tailed groups: safe to log1p (z-scores can be negative and
# are excluded; `amount` is excluded because `log_amount` already exists).
_LOG_GROUPS = {"amount_sums", "frequency"}
_LOG_EXTRA = {
    "avg_amount_7d",
    "avg_amount_30d",
    "std_amount_30d",
    "amount_ratio_to_7d_mean",
    "amount_ratio_to_30d_mean",
    "amount_vs_lifetime_median",
}


def heavy_tail_columns() -> list[str]:
    """Numeric columns that get a ``log1p`` transform."""
    return [
        c for c in NUMERIC_COLUMNS if get_spec(c).group in _LOG_GROUPS or c in _LOG_EXTRA
    ]


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, a_min=0.0, a_max=None))


def _split_numeric_columns() -> tuple[list[str], list[str]]:
    heavy = set(heavy_tail_columns())
    plain = [c for c in NUMERIC_COLUMNS if c not in heavy]
    return plain, sorted(heavy)


def make_logreg_pipeline(C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    """Build the (unfitted) logistic-regression pipeline."""
    plain, heavy = _split_numeric_columns()

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    heavy_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numeric, plain),
            ("heavy", heavy_pipe, heavy),
            ("cat", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def tune_logreg(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    C_grid: list[float],
    class_weights: list[str | None],
) -> tuple[Pipeline, pd.DataFrame]:
    """Grid-search ``C`` x ``class_weight`` on validation PR-AUC."""
    rows = []
    best_pipe, best_pr = None, -np.inf
    for cw in class_weights:
        for C in C_grid:
            pipe = make_logreg_pipeline(C=C, class_weight=cw)
            pipe.fit(x_train, y_train)
            score = pipe.predict_proba(x_val)[:, 1]
            pr = pr_auc(y_val, score)
            rows.append({"C": C, "class_weight": cw, "val_pr_auc": pr})
            if pr > best_pr:
                best_pr, best_pipe = pr, pipe
    return best_pipe, pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)


def logreg_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Standardised coefficients (already on the scaled/one-hot feature space)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs_coefficient"] = df["coefficient"].abs()
    df["direction"] = np.where(df["coefficient"] >= 0, "increases_risk", "decreases_risk")
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_logreg_coefficients(pipeline: Pipeline, path) -> pd.DataFrame:
    df = logreg_coefficients(pipeline)
    df.to_csv(path, index=False)
    return df
