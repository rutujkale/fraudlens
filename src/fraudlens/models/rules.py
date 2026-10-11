"""Rules baseline: a transparent human-designed risk policy (Part 06D).

Rules are defined **before** looking at validation labels. Each rule has a name,
a condition and a weight (from ``cfg.rules``); the score is the weighted fraction
of fired rules, normalised to [0, 1]. Ties are broken by amount (larger first).

Thresholds for the "unusual amount/velocity" rules come from percentiles of the
**training-legit** feature distribution (fitted in :meth:`RuleEngine.fit`); the
rest are hand-set domain thresholds. No label ever touches the rule logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import RulesConfig

# Feature column used to derive a percentile threshold per rule.
_PERCENTILE_COLUMN = {
    "velocity_10m": "tx_count_10m",
    "amount_ratio_30d": "amount_ratio_to_30d_mean",
}
_EXPLICIT_KEYS = ("min_tx_count_10m", "min_ratio", "min_speed_kmh", "min_count_24h")


def _fires(name: str, df: pd.DataFrame, threshold: float | None) -> pd.Series:
    """Boolean Series of whether the named rule fires on each row."""
    if name == "velocity_10m":
        return df["tx_count_10m"] >= threshold
    if name == "amount_ratio_30d":
        return df["amount_ratio_to_30d_mean"] >= threshold
    if name == "new_device_amount":
        return (df["is_new_device"] == 1) & (df["amount_ratio_to_30d_mean"] >= threshold)
    if name == "travel_speed":
        return df["travel_speed_kmh"] >= threshold
    if name == "night_new_category":
        return (df["is_night"] == 1) & (df["new_category_flag"] == 1)
    if name == "near_threshold":
        return df["count_near_threshold_24h"] >= threshold
    raise KeyError(f"unknown rule {name!r}")


class RuleEngine:
    """Weighted rule score in [0, 1] plus per-row explanations."""

    def __init__(self, cfg: RulesConfig) -> None:
        self.cfg = cfg
        self.thresholds: dict[str, float | None] = {}

    def fit(self, train_legit: pd.DataFrame) -> RuleEngine:
        """Resolve thresholds from training-legit percentiles / domain values."""
        for name, params in self.cfg.rules.items():
            self.thresholds[name] = self._resolve_threshold(name, params, train_legit)
        return self

    def _resolve_threshold(
        self, name: str, params: dict, train_legit: pd.DataFrame
    ) -> float | None:
        if "percentile" in params:
            col = _PERCENTILE_COLUMN[name]
            vals = train_legit[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return float(params.get("fallback_min", np.inf))
            thr = float(np.percentile(vals, float(params["percentile"]) * 100.0))
            return max(thr, float(params.get("fallback_min", -np.inf)))
        for key in _EXPLICIT_KEYS:
            if key in params:
                return float(params[key])
        return None

    def _fired_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        fired = {
            name: _fires(name, df, self.thresholds.get(name))
            for name in self.cfg.rules
        }
        return pd.DataFrame(fired, index=df.index)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Weighted fraction of fired rules in [0, 1]."""
        if not self.thresholds:
            raise RuntimeError("RuleEngine.fit must be called before score")
        fired = self._fired_matrix(df)
        weights = pd.Series({name: float(p["weight"]) for name, p in self.cfg.rules.items()})
        total = float(weights.sum())
        return (fired * weights).sum(axis=1) / total

    def explain(self, row: pd.Series) -> list[str]:
        """Names of the rules that fire for a single row."""
        frame = row.to_frame().T
        fired = self._fired_matrix(frame)
        return [name for name in fired.columns if bool(fired.iloc[0][name])]

    def definitions(self) -> list[dict[str, object]]:
        """Rule definitions (name, weight, threshold, condition) for docs."""
        out = []
        for name, params in self.cfg.rules.items():
            out.append(
                {
                    "name": name,
                    "weight": float(params["weight"]),
                    "threshold": self.thresholds.get(name),
                    "condition": _CONDITION_TEXT[name],
                }
            )
        return out


_CONDITION_TEXT = {
    "velocity_10m": "tx_count_10m >= threshold",
    "amount_ratio_30d": "amount_ratio_to_30d_mean >= threshold",
    "new_device_amount": "is_new_device == 1 and amount_ratio_to_30d_mean >= threshold",
    "travel_speed": "travel_speed_kmh >= threshold",
    "night_new_category": "is_night == 1 and new_category_flag == 1",
    "near_threshold": "count_near_threshold_24h >= threshold",
}


def rules_markdown(engine: RuleEngine, profile: str) -> str:
    """Render the fitted rule definitions to markdown for ``docs/rules_baseline.md``."""
    lines = [
        f"# Rules baseline — profile `{profile}`",
        "",
        "Generated by `scripts/train.py`; thresholds fitted on **training-legit** rows "
        "only. The score is the weighted fraction of fired rules in [0, 1]; ties are "
        "broken by amount (larger first).",
        "",
        "| rule | weight | threshold | condition | source |",
        "|---|---|---|---|---|",
    ]
    for d in engine.definitions():
        params = engine.cfg.rules[str(d["name"])]
        source = "training-legit percentile" if "percentile" in params else "domain"
        thr = "-" if d["threshold"] is None else f"{d['threshold']:.4g}"
        lines.append(
            f"| {d['name']} | {d['weight']:g} | {thr} | {d['condition']} | {source} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_rules_doc(engine: RuleEngine, profile: str, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(rules_markdown(engine, profile), encoding="utf-8")
    return out
