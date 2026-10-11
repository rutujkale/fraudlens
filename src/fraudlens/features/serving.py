"""Training/serving parity: score a live transaction with the offline logic (Part 05A).

There is no separate serving implementation. We append the incoming transaction
to its history and call the exact same ``build_features`` used in training, then
return the last row.
"""

from __future__ import annotations

import pandas as pd

from fraudlens.features.build_features import build_features


def compute_features_for_transaction(
    tx_row: dict, history: pd.DataFrame, entities, cfg
) -> pd.Series:
    """Return the feature row for ``tx_row`` given strictly-prior ``history``.

    ``history`` must contain only transactions strictly earlier than ``tx_row``.
    ``tx_row`` and ``history`` share the raw transaction columns.
    """
    row = pd.DataFrame([tx_row])
    combined = pd.concat([history, row], ignore_index=True) if len(history) else row
    features = build_features(combined, entities, cfg)
    match = features[features["transaction_id"] == tx_row["transaction_id"]]
    return match.iloc[-1]
