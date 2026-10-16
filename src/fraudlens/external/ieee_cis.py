"""IEEE-CIS external-validation adapter (Part 11).

The IEEE-CIS Fraud Detection dataset (Kaggle) has no customer id and no
coordinates, so the adapter makes two documented, **pre-registered** choices
(fixed before the data is loaded, in ``config.common.external``):

* **uid** — a pseudo-customer built by string-joining ``card1, card2, card3,
  card5``. ``addr1`` is *not* a uid field; it is used as the location proxy. This
  heuristic is noisy (shared cards, missing values) and is never treated as a
  true customer.
* **pseudo-timestamp** — ``start_date + TransactionDT seconds``. The absolute
  reference is arbitrary; only ordering and spacing matter.

``build_features`` is reused unchanged (it tolerates the coordinate columns being
absent): the ``locations`` frame here carries no ``latitude``/``longitude``, so
distance/travel-speed features are structurally NaN, as is the whole
``STRUCTURALLY_MISSING`` set below. ``prepare_dataset`` returns the canonical
feature matrix aligned with FraudLens' ``df`` so the evaluation code is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fraudlens.config import Config, ExternalIeeeCisConfig
from fraudlens.data_gen.archetypes import CHANNELS, TRANSACTION_TYPES
from fraudlens.evaluation.splits import Split, chronological_window_split, sort_transactions
from fraudlens.features.build_features import build_features
from fraudlens.features.registry import FEATURE_COLUMNS, SPECS
from fraudlens.paths import external_data_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

# --- pre-registered availability contract (Part 11) -------------------------
# Raw columns the adapter never supplies (no merchant id, no recipient).
UNAVAILABLE_RAW: tuple[str, ...] = ("merchant_id", "recipient_id")
# Available location_id proxy (addr1) but no coordinates -> these go NaN.
GEO_METRIC_FEATURES: tuple[str, ...] = (
    "distance_from_home_km",
    "distance_from_prev_km",
    "travel_speed_kmh",
)
# Computable on IEEE but deliberately EXCLUDED from the R2 shared subset S:
# time-of-day is meaningless under the arbitrary TransactionDT origin, and the
# categoricals have different semantics/base rates (ProductCD != merchant_category).
TRANSFER_EXCLUDED: tuple[str, ...] = (
    "merchant_category",
    "transaction_type",
    "channel",
    "hour",
    "is_night",
    "is_weekend",
    "day_of_week",
)

TX_TYPE_CONSTANT = TRANSACTION_TYPES[0]
CHANNEL_CONSTANT = CHANNELS[0]


def _features_requiring(columns: tuple[str, ...]) -> set[str]:
    wanted = set(columns)
    return {s.name for s in SPECS if wanted.intersection(s.requires_columns)}


STRUCTURALLY_MISSING: tuple[str, ...] = tuple(
    sorted(_features_requiring(UNAVAILABLE_RAW) | set(GEO_METRIC_FEATURES))
)
#: R2 shared feature subset: computable AND semantically comparable.
SHARED_FEATURES: list[str] = [
    f for f in FEATURE_COLUMNS if f not in STRUCTURALLY_MISSING and f not in TRANSFER_EXCLUDED
]


class IeeeCisMissingError(FileNotFoundError):
    """Raised when the IEEE-CIS CSVs are absent; message explains how to get them."""


def ieee_cis_paths(cfg: Config) -> tuple[Path, Path]:
    """Return ``(transaction_csv, identity_csv)`` paths for the configured subdir."""
    ext = cfg.simulation.external
    base = external_data_dir(ext.subdir)
    return base / ext.transaction_file, base / ext.identity_file


def missing_message(tx_path: Path, missing: list[str]) -> str:
    """Human-readable download instructions for the missing files."""
    return (
        f"IEEE-CIS files missing ({', '.join(missing)}) in {tx_path.parent}.\n"
        "Download 'train_transaction.csv' and 'train_identity.csv' from the Kaggle "
        "competition 'IEEE-CIS Fraud Detection' (free account; accept the rules) and "
        f"place them in {tx_path.parent}.\n"
        "The data is not redistributed and must never be committed (data/ is gitignored).\n"
        "The rest of FraudLens works without it."
    )


def _tx_usecols(ext: ExternalIeeeCisConfig) -> list[str]:
    base = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "addr1",
        "addr2",
    ]
    for field in ext.uid.fields:
        if field not in base:
            base.append(field)
    return base


_TX_DTYPES: dict[str, str] = {
    "TransactionID": "int64",
    "isFraud": "int8",
    "TransactionDT": "int64",
    "TransactionAmt": "float32",
    "ProductCD": "string",
    "addr1": "string",
    "addr2": "string",
    "card1": "string",
    "card2": "string",
    "card3": "string",
    "card5": "string",
}


def load_ieee_cis(cfg: Config) -> pd.DataFrame:
    """Load the IEEE-CIS transaction + identity tables (left join on ``TransactionID``)."""
    tx_path, id_path = ieee_cis_paths(cfg)
    missing = [p.name for p in (tx_path, id_path) if not p.exists()]
    if missing:
        raise IeeeCisMissingError(missing_message(tx_path, missing))

    cols = _tx_usecols(cfg.simulation.external)
    tx = pd.read_csv(
        tx_path,
        usecols=cols,
        dtype={c: _TX_DTYPES[c] for c in cols if c in _TX_DTYPES},
        low_memory=False,
    )
    identity = pd.read_csv(
        id_path,
        usecols=["TransactionID", "DeviceInfo", "DeviceType"],
        dtype={"TransactionID": "int64", "DeviceInfo": "string", "DeviceType": "string"},
        low_memory=False,
    )
    raw = tx.merge(identity, on="TransactionID", how="left")
    logger.info(
        "IEEE-CIS loaded: %d rows, %d cols, fraud rate %.5f",
        len(raw),
        raw.shape[1],
        float(raw["isFraud"].mean()),
    )
    return raw


def make_uid(raw: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Join the configured stable fields into a pseudo-customer id."""
    cols = [f for f in fields if f in raw.columns]
    if not cols:
        raise ValueError(f"none of the configured uid fields {fields} are present in the frame")
    parts = [raw[c].astype("string").fillna("NA") for c in cols]
    uid = parts[0]
    for part in parts[1:]:
        uid = uid.str.cat(part, sep="|")
    return uid


def uid_stats(uid: pd.Series) -> dict[str, float]:
    """Count and transactions-per-uid distribution for the pre-registered uid."""
    counts = uid.value_counts()
    single = counts[counts == 1]
    return {
        "n_transactions": int(len(uid)),
        "n_uids": int(counts.size),
        "tx_per_uid_mean": float(counts.mean()),
        "tx_per_uid_p50": float(counts.quantile(0.50)),
        "tx_per_uid_p90": float(counts.quantile(0.90)),
        "tx_per_uid_p99": float(counts.quantile(0.99)),
        "tx_per_uid_max": int(counts.max()),
        "single_tx_uid_share": float(single.size / counts.size),
        "rows_in_single_tx_uids_share": float(single.sum() / len(uid)),
    }


def _object_col(series: pd.Series) -> np.ndarray:
    """Object array whose missing values are ``np.nan`` (never ``pd.NA``).

    The loader reads these columns as pandas ``"string"``, so ``astype("object")``
    yields ``pd.NA``. ``build_features`` compares object columns with ``==``; ``pd.NA``
    propagates through that comparison and breaks ``np.count_nonzero``. Convert to
    ``np.nan`` so the external path matches the synthetic one (Part 11 finding).
    """
    return series.astype("object").where(series.notna(), np.nan).to_numpy()


def load_ieee_cis_native(cfg: Config) -> pd.DataFrame:
    """Load **all** native IEEE-CIS columns (V/C/D/M/dist/email + identity).

    R4 is an upper-bound reference that deliberately uses columns outside the
    FraudLens schema, so this is the only loader that reads the full table
    (~590k x ~400). Identifiers and the label are dropped by ``native_matrix``;
    nothing is selected on R4. ``load_ieee_cis`` stays the memory-light path for
    the adapter.
    """
    tx_path, id_path = ieee_cis_paths(cfg)
    missing = [p.name for p in (tx_path, id_path) if not p.exists()]
    if missing:
        raise IeeeCisMissingError(missing_message(tx_path, missing))
    tx = pd.read_csv(tx_path, low_memory=False)
    identity = pd.read_csv(id_path, low_memory=False)
    return tx.merge(identity, on="TransactionID", how="left")


def adapt(raw: pd.DataFrame, ext: ExternalIeeeCisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the raw IEEE frame to the canonical transactions schema + a labels table.

    Labels are returned separately and never appear in the transaction frame.
    """
    uid = make_uid(raw, ext.uid.fields)
    ts = pd.Timestamp(ext.start_date) + pd.to_timedelta(raw["TransactionDT"].to_numpy(), unit="s")
    tx = pd.DataFrame(
        {
            "transaction_id": raw["TransactionID"].to_numpy(),
            "customer_id": uid.to_numpy(),
            "timestamp": ts.to_numpy(),
            "amount": raw["TransactionAmt"].astype("float64").to_numpy(),
            "merchant_category": _object_col(raw["ProductCD"]),
            "device_id": _object_col(raw["DeviceInfo"]),
            "location_id": raw["addr1"].astype("string").fillna("NA").to_numpy(),
            "recipient_id": np.nan,
            "merchant_id": np.nan,
            "transaction_type": TX_TYPE_CONSTANT,
            "channel": CHANNEL_CONSTANT,
        }
    )
    labels = pd.DataFrame(
        {
            "transaction_id": raw["TransactionID"].to_numpy(),
            "is_fraud": raw["isFraud"].astype(int).to_numpy(),
            "fraud_scenario": "external",
            "label_available_at": ts.to_numpy(),
        }
    )
    return tx, labels


def adapt_entities(tx: pd.DataFrame) -> SimpleNamespace:
    """Synthesise ``customers``/``locations`` so ``build_features`` can run.

    ``account_open_date`` is each uid's first timestamp (so ``history_days >= 0``);
    ``home_location_id`` is its first location. ``locations`` has no coordinates,
    which makes the geographic distance features structurally NaN.
    """
    ordered = tx.sort_values(["timestamp", "transaction_id"], kind="stable")
    first = (
        ordered.groupby("customer_id", sort=False)
        .agg(
            account_open_date=("timestamp", "min"),
            home_location_id=("location_id", "first"),
        )
        .reset_index()
    )
    customers = first[["customer_id", "home_location_id", "account_open_date"]]
    locs = tx["location_id"].dropna().unique()
    locations = pd.DataFrame({"location_id": locs, "region_group": locs})
    return SimpleNamespace(customers=customers, locations=locations)


def _missing_reason(name: str) -> str:
    if name in GEO_METRIC_FEATURES:
        return "no coordinates (addr1 is a region proxy and lat/lon are absent)"
    return "requires merchant_id/recipient_id (absent in IEEE-CIS)"


def availability_rows() -> list[dict[str, object]]:
    """Per-feature availability rows (heatmap + ``results/external_feature_availability.csv``)."""
    rows: list[dict[str, object]] = []
    for name in FEATURE_COLUMNS:
        if name in STRUCTURALLY_MISSING:
            status, reason = "missing", _missing_reason(name)
        elif name in TRANSFER_EXCLUDED:
            status, reason = "excluded_from_transfer", "available but semantically incomparable"
        else:
            status, reason = "available", ""
        rows.append(
            {
                "feature": name,
                "status": status,
                "reason": reason,
                "in_shared_subset": name in SHARED_FEATURES,
            }
        )
    return rows


def external_split(tx_sorted: pd.DataFrame, cfg: Config) -> Split:
    """Chronological split of the external dataset (own axes, own embargo)."""
    ext = cfg.simulation.external
    return chronological_window_split(
        tx_sorted,
        train_end_fraction=ext.splits.train_end_fraction,
        val_end_fraction=ext.splits.val_end_fraction,
        embargo_days=ext.splits.embargo_days,
        profile=f"external:{ext.subdir}",
    )


@dataclass
class ExternalDataset:
    """Adapted IEEE-CIS data: canonical transactions, labels, features, split."""

    tx: pd.DataFrame
    labels: pd.DataFrame
    entities: SimpleNamespace
    features: pd.DataFrame
    df: pd.DataFrame
    split: Split
    uid_stats: dict[str, float]
    raw: pd.DataFrame


def prepare_dataset(cfg: Config, raw: pd.DataFrame | None = None) -> ExternalDataset:
    """Load (unless ``raw`` given), adapt, featurise and split the external data."""
    ext = cfg.simulation.external
    if raw is None:
        raw = load_ieee_cis(cfg)
    tx, labels = adapt(raw, ext)
    tx = sort_transactions(tx)
    entities = adapt_entities(tx)
    features = build_features(tx, entities, cfg)
    # Enforce the availability contract: features that are structurally missing on
    # IEEE (e.g. unique_recipients_* compute as 0 with no recipient column) are set
    # to NaN so the transfer model cannot read a misleading constant.
    for name in STRUCTURALLY_MISSING:
        features[name] = np.nan
    df = features.merge(tx[["transaction_id", "customer_id"]], on="transaction_id", how="left")
    df = df.merge(labels[["transaction_id", "is_fraud"]], on="transaction_id", how="left")
    df["timestamp"] = df["feature_as_of_timestamp"]
    df = df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    split = external_split(df, cfg)
    stats = uid_stats(tx["customer_id"])
    logger.info(
        "external split: train=%d val=%d test=%d (uids=%d, single-tx uid share %.3f)",
        len(split.train_idx),
        len(split.val_idx),
        len(split.test_idx),
        stats["n_uids"],
        stats["single_tx_uid_share"],
    )
    return ExternalDataset(
        tx=tx,
        labels=labels,
        entities=entities,
        features=features,
        df=df,
        split=split,
        uid_stats=stats,
        raw=raw,
    )
