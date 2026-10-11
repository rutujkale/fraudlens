"""Load the feature matrix joined with keys and labels (Part 06G).

The features parquet has no ``customer_id``/``timestamp`` (Part 05 keeps feature
logic free of identifiers), so training and evaluation rejoin them from the raw
transactions table. Labels are returned separately and must only ever be used for
training targets or evaluation — never as features.
"""

from __future__ import annotations

import pandas as pd

from fraudlens.features.registry import FEATURE_VERSION
from fraudlens.paths import features_data_dir, raw_data_dir

KEY_COLUMNS = ["transaction_id", "customer_id", "timestamp"]


def load_feature_dataset(profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(features_with_keys, labels)`` sorted by (timestamp, transaction_id)."""
    feats = pd.read_parquet(features_data_dir(profile) / f"{FEATURE_VERSION}.parquet")
    tx = pd.read_parquet(raw_data_dir(profile) / "transactions.parquet")[KEY_COLUMNS]
    labels = pd.read_parquet(raw_data_dir(profile) / "labels.parquet")

    df = feats.merge(tx, on="transaction_id", how="left").sort_values(
        ["timestamp", "transaction_id"], kind="stable"
    )
    return df.reset_index(drop=True), labels
