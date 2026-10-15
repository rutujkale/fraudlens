"""SHAP explanations for the production XGBoost model (Part 10A).

TreeSHAP explains the **raw model margin** (log-odds), not the calibrated
probability. Calibration is a monotone but non-linear map of that margin, so a
SHAP value is a push on the margin and must never be plotted as a point on the
0-100 risk score (see ``docs/methodology.md``).

The explainer is rebuilt on load rather than pickled: it is cheap to construct
and pickling third-party explainer objects couples the artifact to the shap
version. ``shap.TreeExplainer(model, model_output="raw")`` also honours the
model's ``best_iteration``, so the attributions sum to the same margin that
``predict_proba`` (and therefore the calibrator) sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from fraudlens.features.registry import FEATURE_COLUMNS


@dataclass(frozen=True)
class Contribution:
    """One feature's signed SHAP push on the raw margin for a single row."""

    feature: str
    value: float
    shap_value: float
    direction: str  # "increases" | "decreases"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "shap_value": self.shap_value,
            "direction": self.direction,
        }


class ShapExplainer:
    """Margin-space TreeSHAP wrapper around a fitted ``XGBClassifier``."""

    def __init__(self, model, feature_names: list[str] | None = None) -> None:
        self._model = model
        self.feature_names = list(feature_names if feature_names is not None else FEATURE_COLUMNS)
        self._explainer = shap.TreeExplainer(model, model_output="raw")

    @property
    def expected_value(self) -> float:
        """Base margin (``E[f(x)]`` over the tree-path background)."""
        return float(np.asarray(self._explainer.expected_value).ravel()[0])

    def raw_margin(self, x: pd.DataFrame) -> np.ndarray:
        """Raw margin per row, on the same tree range as ``predict_proba``."""
        booster = self._model.get_booster()
        dm = xgb.DMatrix(x, feature_names=self.feature_names)
        best = getattr(self._model, "best_iteration", None)
        iteration_range = (0, int(best) + 1) if best is not None else None
        return np.asarray(booster.predict(dm, output_margin=True, iteration_range=iteration_range))

    def shap_values(self, x: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_features)`` signed SHAP values in margin space."""
        return np.asarray(self._explainer.shap_values(x), dtype=float)

    def interaction_values(self, x: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_features, n_features)`` TreeSHAP interaction values."""
        return np.asarray(self._explainer.shap_interaction_values(x), dtype=float)

    def check_additivity(self, x: pd.DataFrame) -> float:
        """Max ``|sum(shap) + expected_value - raw_margin|`` over ``x``."""
        total = self.shap_values(x).sum(axis=1) + self.expected_value
        return float(np.max(np.abs(total - self.raw_margin(x))))

    def explain_global(self, x: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| and mean signed SHAP per feature, ranked descending."""
        values = self.shap_values(x)
        out = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False, kind="stable")
        out = out.reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def explain_local(self, x_row: pd.DataFrame, top_k: int | None = None) -> list[Contribution]:
        """Signed contributions for one row (already feature-encoded), largest first."""
        if len(x_row) != 1:
            raise ValueError("explain_local expects exactly one row")
        values = self.shap_values(x_row)[0]
        row = x_row.iloc[0]
        contributions = [
            Contribution(
                feature=feature,
                value=float(row[feature]) if pd.notna(row[feature]) else float("nan"),
                shap_value=float(sv),
                direction="increases" if sv >= 0 else "decreases",
            )
            for feature, sv in zip(self.feature_names, values, strict=True)
        ]
        contributions.sort(key=lambda c: (-abs(c.shap_value), c.feature))
        return contributions[:top_k] if top_k is not None else contributions

    def top_drivers(
        self, x_row: pd.DataFrame, k: int = 5
    ) -> tuple[list[Contribution], list[Contribution]]:
        """Top ``k`` margin-raising and margin-lowering contributions."""
        ordered = self.explain_local(x_row)
        raising = [c for c in ordered if c.shap_value > 0][:k]
        lowering = [c for c in ordered if c.shap_value < 0][:k]
        return raising, lowering


def stratified_val_sample(
    y: np.ndarray, max_frauds: int = 5000, max_legit: int = 15000, seed: int = 42
) -> np.ndarray:
    """Positional indices: all frauds up to ``max_frauds`` + sampled legit rows."""
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    if len(pos) > max_frauds:
        pos = np.sort(rng.choice(pos, size=max_frauds, replace=False))
    if len(neg) > max_legit:
        neg = np.sort(rng.choice(neg, size=max_legit, replace=False))
    return np.sort(np.concatenate([pos, neg]))
