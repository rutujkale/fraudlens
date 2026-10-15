"""Subgroup fairness audit (Part 10E).

Audit attributes (``age_band``, ``income_band``, ``archetype``, ``home_region_group``)
are **never model inputs**; they are joined onto validation rows for analysis only.
All metrics are computed on validation data with the frozen policy.

Two views are produced:

* **primary** — the ``val_thr`` slice the policy and thresholds were selected on
  (122 frauds). Calibration metrics are out-of-sample here (the calibrator was fit
  on the earlier ``val_cal`` half).
* **exploratory** — the full validation window (342 frauds). Calibration gap and
  Brier are omitted: the calibrator saw half of this window when it was fitted.

Because the data is simulated, group differences reflect the simulator's
assumptions about age/income/archetype, not real-world demographics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from fraudlens.evaluation.bootstrap import cluster_bootstrap
from fraudlens.evaluation.cost_model import APPROVE, BLOCK, REVIEW, STEP_UP
from fraudlens.evaluation.metrics import pr_auc
from fraudlens.features.registry import CATEGORICAL_COLUMNS
from fraudlens.paths import raw_data_dir

AUDIT_ATTRIBUTES = ("age_band", "income_band", "archetype", "home_region_group")
ORDINAL_ATTRIBUTES = ("age_band", "income_band")
NOMINAL_ATTRIBUTES = ("archetype", "home_region_group")
AGE_BANDS = (
    (18, 25, "<25"),
    (25, 35, "25-34"),
    (35, 50, "35-49"),
    (50, 65, "50-64"),
    (65, 200, "65+"),
)

MIN_POSITIVES = 30
MIN_ROWS = 1000
STRONG: dict[str, float] = {"spearman": 0.30, "eta_squared": 0.14, "cramers_v": 0.30}


def age_band(age: float) -> str:
    """Bucket an age into the pre-registered bands."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def load_audit_attributes(profile: str) -> pd.DataFrame:
    """Per-customer audit attributes (derives ``home_region_group``)."""
    customers = pd.read_parquet(raw_data_dir(profile) / "customers.parquet")
    locations = pd.read_parquet(raw_data_dir(profile) / "locations.parquet")
    region = locations.set_index("location_id")["region_group"]
    out = pd.DataFrame(
        {
            "customer_id": customers["customer_id"],
            "age_band": customers["age"].map(age_band),
            "income_band": customers["income_band"],
            "archetype": customers["archetype"],
            "home_region_group": customers["home_location_id"].map(region),
        }
    )
    return out


@dataclass
class GroupTable:
    """Per-group metrics plus the suppression summary."""

    metrics: pd.DataFrame
    suppression: pd.DataFrame


def recall_metric(y: np.ndarray, actions: np.ndarray) -> float:
    """Fraud recall with the alert defined as ``action != APPROVE``."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(actions) != APPROVE
    if not y.any():
        return float("nan")
    return float((y & flagged).sum() / y.sum())


def precision_metric(y: np.ndarray, actions: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(actions) != APPROVE
    if not flagged.any():
        return float("nan")
    return float((y & flagged).sum() / flagged.sum())


def fpr_metric(y: np.ndarray, actions: np.ndarray) -> float:
    """False-positive rate among legitimate rows, alert = ``action != APPROVE``."""
    y = np.asarray(y).astype(bool)
    legit = ~y
    flagged = np.asarray(actions) != APPROVE
    if not legit.any():
        return float("nan")
    return float((legit & flagged).sum() / legit.sum())


def flag_rate_metric(y: np.ndarray, actions: np.ndarray) -> float:
    return float((np.asarray(actions) != APPROVE).mean())


def _group_block(
    y: np.ndarray,
    p: np.ndarray,
    actions: np.ndarray,
    groups: np.ndarray,
    calibration_metrics: bool,
    n_boot: int,
    seed: int,
) -> dict:
    """All metrics for one group's rows (already subset)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    actions = np.asarray(actions)
    n = len(y)
    n_fraud = int(y.sum())
    strict = (actions == REVIEW) | (actions == BLOCK)

    row = {
        "n": n,
        "n_fraud": n_fraud,
        "prevalence": float(n_fraud / n) if n else float("nan"),
        "flag_rate_any": flag_rate_metric(y, actions),
        "flag_rate_strict": float((strict).mean()) if n else float("nan"),
        "fpr_any": fpr_metric(y, actions),
        "fpr_strict": (
            float((~y.astype(bool) & strict).sum() / max(1, (~y.astype(bool)).sum()))
            if (~y.astype(bool)).any()
            else float("nan")
        ),
        "recall": recall_metric(y, actions) if n_fraud else float("nan"),
        "precision": precision_metric(y, actions) if n_fraud else float("nan"),
        "pr_auc": (
            pr_auc(y, p) if 0 < n_fraud < n else float("nan")
        ),
        "n_approve": int((actions == APPROVE).sum()),
        "n_step_up": int((actions == STEP_UP).sum()),
        "n_review": int((actions == REVIEW).sum()),
        "n_block": int((actions == BLOCK).sum()),
    }
    row["mean_p"] = float(p.mean()) if n else float("nan")
    row["calibration_gap"] = (
        float(p.mean() - row["prevalence"]) if n else float("nan")
    )
    row["brier"] = float(np.mean((p - y) ** 2)) if n else float("nan")

    reasons: list[str] = []
    if n < MIN_ROWS:
        reasons.append(f"rows={n}<{MIN_ROWS}")
        for key in ("calibration_gap", "brier"):
            row[key] = float("nan")
    if n_fraud < MIN_POSITIVES:
        reasons.append(f"positives={n_fraud}<{MIN_POSITIVES}")
        for key in ("recall", "precision", "pr_auc"):
            row[key] = float("nan")

    if n >= MIN_ROWS:
        for name, metric in (
            ("flag_rate_any", flag_rate_metric),
            ("fpr_any", fpr_metric),
        ):
            _, lo, hi = cluster_bootstrap(
                y, actions, groups, metric, n=n_boot, seed=seed
            )
            row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
        if n_fraud >= MIN_POSITIVES:
            _, lo, hi = cluster_bootstrap(
                y, actions, groups, recall_metric, n=n_boot, seed=seed
            )
            row["recall_lo"], row["recall_hi"] = lo, hi

    if not calibration_metrics:
        for key in ("mean_p", "calibration_gap", "brier"):
            row[key] = float("nan")
    row["suppressed_reason"] = "; ".join(reasons)
    return row


def group_table(
    frame: pd.DataFrame,
    *,
    calibration_metrics: bool,
    n_boot: int = 500,
    seed: int = 42,
) -> GroupTable:
    """Metrics per (attribute, group) for the rows in ``frame``.

    ``frame`` needs ``y``, ``p``, ``action``, ``customer_id`` and the audit columns.
    """
    rows = []
    for attribute in AUDIT_ATTRIBUTES:
        for group in sorted(frame[attribute].dropna().unique()):
            mask = (frame[attribute] == group).to_numpy()
            if not mask.any():
                continue
            block = _group_block(
                frame.loc[mask, "y"].to_numpy(),
                frame.loc[mask, "p"].to_numpy(),
                frame.loc[mask, "action"].to_numpy(),
                frame.loc[mask, "customer_id"].to_numpy(),
                calibration_metrics,
                n_boot,
                seed,
            )
            rows.append({"attribute": attribute, "group": group, **block})
    metrics = pd.DataFrame(rows)

    supp = []
    for attribute, sub in metrics.groupby("attribute", sort=False):
        by_pos = int((sub["n_fraud"] < MIN_POSITIVES).sum())
        by_rows = int((sub["n"] < MIN_ROWS).sum())
        supp.append(
            {
                "attribute": attribute,
                "n_groups": int(len(sub)),
                "suppressed_positives": by_pos,
                "suppressed_rows": by_rows,
                "reported": int((sub["suppressed_reason"] == "").sum()),
            }
        )
    return GroupTable(metrics=metrics, suppression=pd.DataFrame(supp))


def metric_ratios(metrics: pd.DataFrame, metric: str = "fpr_any") -> dict:
    """Max/min ratio for ``metric`` across reported groups (with group names)."""
    clean = metrics.dropna(subset=[metric])
    clean = clean[clean["suppressed_reason"] == ""]
    if len(clean) < 2:
        return {"metric": metric, "ratio": float("nan"), "hi_group": None, "lo_group": None}
    hi = clean.loc[clean[metric].idxmax()]
    lo = clean.loc[clean[metric].idxmin()]
    ratio = float(hi[metric] / lo[metric]) if lo[metric] else float("inf")
    return {
        "metric": metric,
        "ratio": ratio,
        "hi_group": f"{hi['attribute']}={hi['group']}",
        "lo_group": f"{lo['attribute']}={lo['group']}",
    }


def paired_group_diff(
    y: np.ndarray,
    actions: np.ndarray,
    groups: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-bootstrap CI of ``metric(A) - metric(B)`` (shared customer resample).

    Both groups are resampled from the same draws, so the CI answers "is this
    difference distinguishable from sampling noise?" rather than comparing two
    independent samples.
    """
    y = np.asarray(y)
    actions = np.asarray(actions)
    groups = np.asarray(groups)
    mask_a = np.asarray(mask_a)
    mask_b = np.asarray(mask_b)
    point = float(metric_fn(y[mask_a], actions[mask_a]) - metric_fn(y[mask_b], actions[mask_b]))
    unique = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        a, b = idx[mask_a[idx]], idx[mask_b[idx]]
        if len(a) == 0 or len(b) == 0:
            stats[i] = float("nan")
            continue
        stats[i] = metric_fn(y[a], actions[a]) - metric_fn(y[b], actions[b])
    finite = stats[np.isfinite(stats)]
    if len(finite) < 2:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return point, float(lo), float(hi)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Share of variance in ``values`` explained by group membership."""
    frame = pd.DataFrame({"v": values, "g": labels}).dropna()
    if frame.empty:
        return float("nan")
    grand = frame["v"].mean()
    ss_total = float(((frame["v"] - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = float(
        (frame.groupby("g")["v"].agg(["mean", "count"]).assign(
            ss=lambda d: d["count"] * (d["mean"] - grand) ** 2
        )["ss"]).sum()
    )
    return ss_between / ss_total


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if frame.empty:
        return float("nan")
    table = pd.crosstab(frame["a"], frame["b"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = float(chi2_contingency(table.to_numpy())[0])
    n = float(table.to_numpy().sum())
    phi2 = chi2 / n
    r, k = table.shape
    return float(np.sqrt(phi2 / max(1, min(r - 1, k - 1))))


def proxy_check(features: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Correlation between each feature and each audit attribute.

    Ordinal attributes (``age_band``, ``income_band``) use Spearman rho against the
    band rank; nominal attributes use eta-squared for numeric features and Cramer's V
    for the three categorical features. ``strong`` uses the pre-registered thresholds.
    """
    ordinal_rank = {
        "age_band": {label: i for i, (_lo, _hi, label) in enumerate(AGE_BANDS)},
        "income_band": {"low": 0, "mid": 1, "mid_high": 2, "high": 3},
    }
    rows = []
    for attribute in AUDIT_ATTRIBUTES:
        for feature in features.columns:
            x = features[feature].to_numpy(dtype=float)
            if feature in CATEGORICAL_COLUMNS:
                codes = np.asarray(features[feature])
                value = _cramers_v(codes, audit[attribute].to_numpy())
                statistic = "cramers_v"
            elif attribute in ORDINAL_ATTRIBUTES:
                codes = audit[attribute].map(ordinal_rank[attribute]).to_numpy(dtype=float)
                if len(np.unique(codes[~np.isnan(codes)])) < 2:
                    value = float("nan")
                else:
                    value = float(spearmanr(x, codes, nan_policy="omit").statistic)
                statistic = "spearman"
            else:
                value = _eta_squared(x, audit[attribute].to_numpy())
                statistic = "eta_squared"
            rows.append(
                {
                    "attribute": attribute,
                    "feature": feature,
                    "statistic": statistic,
                    "value": value,
                    "abs_value": abs(value) if value == value else float("nan"),
                    "threshold": STRONG[statistic],
                    "strong": bool(np.isfinite(value) and abs(value) >= STRONG[statistic]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["attribute", "abs_value"], ascending=[True, False], kind="stable"
    )
