"""Population Stability Index for feature drift (Part 08 monitoring).

PSI compares a *reference* distribution (e.g. the training window) with a
*current* one on the same bins, using the reference quantiles as bin edges:

    PSI = Σ (cur% − ref%) · ln(cur% / ref%)

Bins are equal-frequency on the reference, so each reference bin starts near
``1/bins``. Missing values are kept as their own bucket — several features here
are >80% NaN by construction (no prior device/recipient), and a change in the
missing *rate* is drift too. A small ``eps`` floor keeps empty bins finite.

Rule of thumb: PSI < 0.1 stable, 0.1–0.25 monitor, > 0.25 significant shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MONITOR = 0.10
PSI_SHIFT = 0.25

# Tenure-style features dominate naive PSI (Part 08E): a new account legitimately
# shows null device/history features, so PSI is reported with and without these.
TENURE_FEATURES = ["history_days", "device_age_days", "n_prior_tx", "tx_from_device_prior"]


def _bin_edges(valid: np.ndarray, bins: int) -> np.ndarray:
    if len(valid) == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(valid, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        edges = np.array([valid[0] - 1.0, valid[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray, eps: float) -> np.ndarray:
    """Bin proportions plus a trailing missing-value proportion, floored by ``eps``."""
    nan_frac = float(np.isnan(values).mean()) if len(values) else 0.0
    valid = values[~np.isnan(values)]
    counts, _ = np.histogram(valid, bins=edges)
    total = counts.sum()
    props = (counts / total) * (1.0 - nan_frac) if total > 0 else np.zeros(len(counts))
    props = np.append(props, nan_frac)
    props = np.clip(props, eps, None)
    return props / props.sum()


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two 1-D samples (0 = identical)."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    edges = _bin_edges(ref[~np.isnan(ref)], bins)
    ref_p = _proportions(ref, edges, eps)
    cur_p = _proportions(cur, edges, eps)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_flag(value: float) -> str:
    """Bucket a PSI value into ``stable`` / ``monitor`` / ``shift``."""
    if value < PSI_MONITOR:
        return "stable"
    if value < PSI_SHIFT:
        return "monitor"
    return "shift"


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bins: int = 10,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """PSI per feature, sorted descending, with a stability flag column."""
    rows = [
        {"feature": f, "psi": psi(reference[f].to_numpy(), current[f].to_numpy(), bins, eps)}
        for f in features
    ]
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    table["flag"] = table["psi"].map(psi_flag)
    return table
