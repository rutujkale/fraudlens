"""Reusable point-in-time leakage checks.

The truncation-invariance property (a feature for the row at ``T`` depends only on
rows strictly before ``T``) is asserted by the Part 05 test suite and re-used by
the Part 11 external validation on IEEE rows, so the logic lives in exactly one
place. Callers supply canonical ``transactions`` + ``entities`` and run the real
``build_features``; nothing here re-implements feature computation.
"""

from __future__ import annotations

import pandas as pd

from fraudlens.features.build_features import build_features
from fraudlens.features.registry import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

_TOL = 1e-9


def truncation_invariance(tx: pd.DataFrame, entities, cfg, cut_ts) -> dict:
    """Compare features built on the full frame vs a frame truncated at ``cut_ts``.

    Rows with ``feature_as_of_timestamp <= cut_ts`` must be identical. Returns a
    dict with ``n_rows``, ``max_abs_diff`` (numeric columns; string columns count
    as ``inf`` if unequal) and ``passed``.
    """
    cut = pd.Timestamp(cut_ts)
    full = build_features(tx, entities, cfg)
    sub = tx[tx["timestamp"] <= cut]
    trunc = build_features(sub, entities, cfg)

    a = (
        full[full["feature_as_of_timestamp"] <= cut]
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )
    b = trunc.sort_values("transaction_id").reset_index(drop=True)
    if set(a["transaction_id"]) != set(b["transaction_id"]):
        return {
            "n_rows": int(len(a)),
            "max_abs_diff": float("inf"),
            "passed": False,
            "note": "transaction_id sets differ",
        }

    max_diff = 0.0
    for col in NUMERIC_COLUMNS:
        d = (a[col].astype("float64") - b[col].astype("float64")).abs()
        if d.notna().any():
            max_diff = max(max_diff, float(d.max()))
    for col in CATEGORICAL_COLUMNS:
        if not a[col].astype("object").equals(b[col].astype("object")):
            max_diff = float("inf")
    return {
        "n_rows": int(len(a)),
        "max_abs_diff": max_diff,
        "passed": bool(max_diff <= _TOL),
        "note": "",
    }
