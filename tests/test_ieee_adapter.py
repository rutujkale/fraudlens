"""IEEE-CIS adapter + external-runner tests (Part 11).

Uses a small synthetic fake IEEE-shaped frame so CI never needs the real Kaggle
files. The runner end-to-end test writes to ``tmp_path`` only.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from fraudlens.config import load_config
from fraudlens.data_gen.archetypes import CATEGORIES, CHANNELS, TRANSACTION_TYPES
from fraudlens.evaluation.splits import chronological_window_split, make_temporal_split
from fraudlens.external import ieee_cis
from fraudlens.external.runner import SyntheticRefs, native_matrix, run
from fraudlens.features.registry import FEATURE_COLUMNS, FORBIDDEN_COLUMNS
from fraudlens.models.xgb import encode_features
from fraudlens.paths import REPO_ROOT

CFG = load_config("dev")

_CANONICAL = {
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant_category",
    "device_id",
    "location_id",
    "recipient_id",
    "merchant_id",
    "transaction_type",
    "channel",
}

_SYNTH_PARAMS = {
    "scale_pos_weight": 5.0,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def _fake_raw(n: int = 600, n_uids: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uid = rng.integers(0, n_uids, size=n)
    dt = np.sort(rng.integers(0, 180 * 86400, size=n)).astype("int64")
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1, dtype="int64"),
            "isFraud": (rng.random(n) < 0.06).astype("int8"),
            "TransactionDT": dt,
            "TransactionAmt": np.round(rng.lognormal(4.0, 1.0, size=n), 2).astype("float32"),
            "ProductCD": pd.Series(rng.choice(["W", "C", "R", "H", "S"], size=n), dtype="string"),
            "addr1": pd.Series(rng.choice([100.0, 110.0, 120.0, 130.0], size=n), dtype="float64"),
            "addr2": pd.Series(rng.choice([87.0, 60.0], size=n), dtype="float64"),
            "card1": pd.Series(uid, dtype="string"),
            "card2": pd.Series(1, index=range(n), dtype="string"),
            "card3": pd.Series(2, index=range(n), dtype="string"),
            "card5": pd.Series(3, index=range(n), dtype="string"),
            "DeviceInfo": pd.Series(
                np.where(rng.random(n) < 0.5, "Windows", None), dtype="object"
            ),
            "DeviceType": pd.Series(
                np.where(rng.random(n) < 0.5, "desktop", None), dtype="object"
            ),
            "V1": rng.normal(size=n).astype("float32"),
            "V2": rng.normal(size=n).astype("float32"),
        }
    )


def _fake_synth(n: int = 400, seed: int = 1) -> SyntheticRefs:
    rng = np.random.default_rng(seed)
    cols = ieee_cis.SHARED_FEATURES
    x = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    y = (rng.random(n) < 0.15).astype(int)
    x.iloc[:, 0] = x.iloc[:, 0] + y * 2.5
    half = n // 2
    return SyntheticRefs(
        profile="dev",
        x_train=x.iloc[:half].reset_index(drop=True),
        y_train=y[:half],
        x_val=x.iloc[half:].reset_index(drop=True),
        y_val=y[half:],
        params=dict(_SYNTH_PARAMS),
    )


def test_adapter_schema_and_labels() -> None:
    raw = _fake_raw(n=50)
    tx, labels = ieee_cis.adapt(raw, CFG.simulation.external)
    assert set(tx.columns) == _CANONICAL
    assert tx["amount"].dtype == np.float64
    assert tx["transaction_id"].tolist() == raw["TransactionID"].tolist()
    assert labels["is_fraud"].tolist() == raw["isFraud"].astype(int).tolist()
    assert tx["recipient_id"].isna().all()
    assert tx["merchant_id"].isna().all()
    assert set(tx["transaction_type"].unique()) == {TRANSACTION_TYPES[0]}


def test_pseudo_timestamp_matches_transactiondt() -> None:
    raw = _fake_raw(n=20)
    tx, _ = ieee_cis.adapt(raw, CFG.simulation.external)
    base = pd.Timestamp(CFG.simulation.external.start_date)
    expected = base + pd.to_timedelta(raw["TransactionDT"], unit="s")
    assert (pd.to_datetime(tx["timestamp"]).to_numpy() == expected.to_numpy()).all()
    assert pd.to_datetime(tx["timestamp"]).is_monotonic_increasing


def test_uid_construction_and_stats() -> None:
    raw = _fake_raw(n=600, n_uids=50, seed=0)
    fields = CFG.simulation.external.uid.fields
    uid = ieee_cis.make_uid(raw, fields)
    assert uid.nunique() == raw["card1"].nunique() == 50
    stats = ieee_cis.uid_stats(uid)
    assert stats["n_transactions"] == 600
    assert stats["n_uids"] == 50
    assert 0.0 <= stats["single_tx_uid_share"] <= 1.0
    # addr1 is NOT a uid field (it is the location proxy)
    assert "addr1" not in fields


def test_availability_mask_sets() -> None:
    assert set(ieee_cis.STRUCTURALLY_MISSING) == {
        "new_merchant_flag",
        "merchant_freq_30d",
        "unique_recipients_1h",
        "unique_recipients_24h",
        "is_new_recipient",
        "distance_from_home_km",
        "distance_from_prev_km",
        "travel_speed_kmh",
    }
    assert len(ieee_cis.SHARED_FEATURES) == (
        len(FEATURE_COLUMNS) - len(ieee_cis.STRUCTURALLY_MISSING) - len(ieee_cis.TRANSFER_EXCLUDED)
    )
    for name in (*ieee_cis.STRUCTURALLY_MISSING, *ieee_cis.TRANSFER_EXCLUDED):
        assert name not in ieee_cis.SHARED_FEATURES
    rows = ieee_cis.availability_rows()
    assert {r["feature"] for r in rows} == set(FEATURE_COLUMNS)


def test_prepare_dataset_masks_missing_features() -> None:
    ds = ieee_cis.prepare_dataset(CFG, _fake_raw(n=400))
    for name in ieee_cis.STRUCTURALLY_MISSING:
        assert ds.features[name].isna().all(), name
    # entities have no coordinates -> the geographic features stay NaN
    assert "latitude" not in ds.entities.locations.columns
    # device features are NaN exactly where DeviceInfo is missing
    device = ds.tx.set_index("transaction_id")["device_id"]
    is_new = ds.features.set_index("transaction_id")["is_new_device"]
    missing_device = device.reindex(is_new.index).isna()
    assert is_new[missing_device].isna().all()
    assert (ds.features["history_days"] >= 0).all()


def test_no_label_leakage() -> None:
    ds = ieee_cis.prepare_dataset(CFG, _fake_raw(n=300))
    assert not (FORBIDDEN_COLUMNS & set(ds.features.columns))
    assert "is_fraud" not in ds.features.columns
    assert "is_fraud" not in FEATURE_COLUMNS


def test_encode_features_preserves_synthetic_and_adds_ieee() -> None:
    rows = [
        {"transaction_type": TRANSACTION_TYPES[0], "channel": CHANNELS[0], "merchant_category": cat}
        for cat in CATEGORIES
    ]
    rows += [
        {"transaction_type": TRANSACTION_TYPES[0], "channel": CHANNELS[0], "merchant_category": pc}
        for pc in ["W", "C", "R", "H", "S"]
    ]
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    encoded = encode_features(df[FEATURE_COLUMNS])
    synthetic = encoded.iloc[: len(CATEGORIES)]["merchant_category"].to_numpy()
    assert (synthetic == np.arange(len(CATEGORIES))).all()
    ieee = encoded.iloc[len(CATEGORIES) :]["merchant_category"].to_numpy()
    assert not np.isnan(ieee).any()
    assert (ieee >= len(CATEGORIES)).all()


def test_default_split_boundaries_unchanged() -> None:
    ts = pd.date_range("2026-01-01", periods=1000, freq="h")
    tx = pd.DataFrame({"timestamp": ts, "transaction_id": np.arange(1000)})
    a = make_temporal_split(tx, CFG)
    b = chronological_window_split(
        tx,
        train_end_fraction=CFG.simulation.splits.train_end_fraction,
        val_end_fraction=CFG.simulation.splits.val_end_fraction,
        embargo_days=CFG.embargo_days,
        profile=CFG.simulation.profile,
    )
    assert a.boundaries == b.boundaries
    assert (a.train_idx == b.train_idx).all()
    assert (a.val_idx == b.val_idx).all()
    assert (a.test_idx == b.test_idx).all()
    assert CFG.simulation.splits.train_end_fraction == 0.60
    assert CFG.simulation.splits.val_end_fraction == 0.80


def test_native_matrix_drops_identifiers_and_labels() -> None:
    native = native_matrix(_fake_raw(n=80))
    assert "isFraud" not in native.columns
    assert "TransactionID" not in native.columns
    assert native.shape[1] > 0


def test_cli_exits_cleanly_without_files() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_external_validation.py", "--profile", "dev"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "skipped: data missing" in result.stdout


def test_runner_end_to_end_on_fake_frame(tmp_path) -> None:
    ds = ieee_cis.prepare_dataset(CFG, _fake_raw(n=600, n_uids=50, seed=3))
    result = run(
        CFG,
        ds,
        synthetic=_fake_synth(),
        out_dir=tmp_path,
        log_experiments=False,
        save_artifacts=False,
    )
    for name in (
        "external_validation.csv",
        "external_gap.csv",
        "external_calibration.csv",
        "external_policy.csv",
        "external_feature_availability.csv",
        "external_feature_nan_share.csv",
        "test_access_log_external.csv",
    ):
        assert (tmp_path / name).exists(), name
    assert "primary_gap_pr_auc_r1_minus_r2_on_ieee_test" in result["gap"]
    assert len(result["policy"]) == 4
    # the external guard log is separate from the synthetic one
    assert "external" in (tmp_path / "test_access_log_external.csv").read_text(encoding="utf-8")


def test_no_tracked_files_under_external_data() -> None:
    result = subprocess.run(
        ["git", "ls-files", "data/external"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
