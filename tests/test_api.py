"""Part 13B: FastAPI service tests + training/serving parity on the validation window.

The app runs the frozen dev bundle with the in-memory history backend. Parity
samples and batch fixtures come from the **validation** window only; the test
hard-fails if any sampled row falls outside ``[t_train_end, t_val_end)`` and never
skips. The test window is never read.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.api.main import create_app
from fraudlens.config import load_config
from fraudlens.db import queries
from fraudlens.db.loader import get_engine
from fraudlens.evaluation import locked
from fraudlens.evaluation.cost_model import ACTION_NAMES, REVIEW
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.freeze import sha256_file
from fraudlens.evaluation.splits import make_temporal_split
from fraudlens.features.serving import compute_features_for_transaction
from fraudlens.models.xgb import encode_features
from fraudlens.paths import artifacts_dir, raw_data_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions
from fraudlens.risk.scoring import RISK_LEVELS

VALIDATION_DAY = "2026-04-06"


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app(profile="dev", history_backend="memory", verify_freeze=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def scorer(client: TestClient):
    return client.app.state.scorer


@pytest.fixture(scope="module")
def raw_transactions() -> pd.DataFrame:
    return pd.read_parquet(raw_data_dir("dev") / "transactions.parquet")


def _val_frame(scorer) -> tuple[pd.DataFrame, dict[str, str]]:
    df, _ = load_feature_dataset("dev")
    split = make_temporal_split(df, load_config("dev"))
    val = df.iloc[split.val_idx].reset_index(drop=True)
    return val, split.boundaries


def _opt(value):
    return None if value is None or pd.isna(value) else value


def _request_from_row(
    row: dict, include_explanation: bool = False, transaction_id: str | None = None
) -> dict:
    return {
        "customer_id": row["customer_id"],
        "amount": float(row["amount"]),
        "timestamp": str(row["timestamp"]),
        "merchant_category": _opt(row.get("merchant_category")),
        "merchant_id": _opt(row.get("merchant_id")),
        "device_id": _opt(row.get("device_id")),
        "location_id": _opt(row.get("location_id")),
        "transaction_type": row["transaction_type"],
        "channel": row["channel"],
        "recipient_id": _opt(row.get("recipient_id")),
        "transaction_id": transaction_id or row.get("transaction_id"),
        "include_explanation": include_explanation,
    }


# --- functional ---------------------------------------------------------------


def test_predict_valid_schema(client, scorer):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["recommended_action"] in ACTION_NAMES
    assert body["risk_level"] == RISK_LEVELS[ACTION_NAMES.index(body["recommended_action"])]
    assert body["model_version"] == scorer.model_version
    assert body["base_rate"] == pytest.approx(scorer.base_rate)
    assert body["lift_over_base_rate"] == pytest.approx(
        body["fraud_probability"] / body["base_rate"]
    )
    assert set(body["timing_ms"]) == {
        "history_fetch",
        "features",
        "model",
        "calibration",
        "policy",
        "explanation",
    }
    assert all(v >= 0 for v in body["timing_ms"].values())
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"amount": -5.0},
        {"amount": 0.0},
        {"timestamp": "not-a-timestamp"},
        {"customer_id": None},
        {"merchant_category": "crypto_swap"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_requests_422(client, mutation):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    payload.update(mutation)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text


def test_timezone_aware_timestamp_is_normalised(client):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    payload["timestamp"] = "2026-03-29T02:19:00+05:30"
    payload["include_explanation"] = False
    with_tz = client.post("/predict", json=payload).json()["fraud_probability"]
    payload["timestamp"] = "2026-03-29 02:19:00"
    without_tz = client.post("/predict", json=payload).json()["fraud_probability"]
    assert with_tz == pytest.approx(without_tz)


def test_cold_start_customer_is_valid(client, scorer):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    payload["customer_id"] = "C_UNSEEN_0001"
    payload["include_explanation"] = True
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0

    # The shared feature code marks the account cold-start (no prior rows).
    tx_row = {
        "transaction_id": "cold-1",
        "customer_id": payload["customer_id"],
        "timestamp": pd.Timestamp(payload["timestamp"]),
        "amount": float(payload["amount"]),
        "merchant_id": payload.get("merchant_id"),
        "merchant_category": None,
        "device_id": payload.get("device_id"),
        "location_id": payload.get("location_id"),
        "recipient_id": payload.get("recipient_id"),
        "transaction_type": payload["transaction_type"],
        "channel": payload["channel"],
    }
    history = scorer.provider.history(
        payload["customer_id"], payload.get("device_id"), tx_row["timestamp"]
    )
    feat = compute_features_for_transaction(tx_row, history, scorer.entities, scorer.cfg)
    assert feat["is_cold_start"] == 1
    assert feat["n_prior_tx"] == 0


def test_unknown_device_is_new_device(client):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    payload["device_id"] = "D_UNSEEN_9999"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text


def test_batch_size_limit_enforced(client, scorer):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    max_batch = int(scorer.cfg.simulation.api.max_batch)
    over = {"transactions": [payload] * (max_batch + 1)}
    response = client.post("/predict/batch", json=over)
    assert response.status_code == 422, response.text
    assert "max_batch" in response.text


def test_batch_capacity_caps_review_only(client, scorer):
    # A deterministic pool of validation rows whose uncapped action is REVIEW.
    val, _ = _val_frame(scorer)
    x = encode_features(val)
    p = scorer.calibrator.predict(scorer.model.predict_proba(x)[:, 1])
    actions = amount_aware_actions(p, val["amount"].to_numpy(float), scorer.params)
    day = pd.Timestamp(VALIDATION_DAY).date()
    ids = val.loc[(actions == REVIEW) & (val["timestamp"].dt.date == day), "transaction_id"]
    raw = pd.read_parquet(raw_data_dir("dev") / "transactions.parquet").set_index("transaction_id")
    requests_payload = [
        _request_from_row(raw.loc[tid].to_dict(), transaction_id=tid) for tid in ids
    ]
    assert len(requests_payload) >= 2, "fixture expected multiple REVIEW rows on the day"

    original = scorer.capacity
    scorer.capacity = 1
    try:
        body = client.post("/predict/batch", json={"transactions": requests_payload}).json()
    finally:
        scorer.capacity = original
    assert body["summary"]["review_workload"] == 1
    assert body["summary"]["n_demoted"] == len(requests_payload) - 1


def test_capacity_never_touches_block_or_step_up(client, scorer, raw_transactions):
    day = pd.Timestamp(VALIDATION_DAY).date()
    rows = raw_transactions[raw_transactions["timestamp"].dt.date == day].head(60)
    payload = {"transactions": [_request_from_row(r) for r in rows.to_dict("records")]}
    body = client.post("/predict/batch", json=payload).json()
    by_id = {row["transaction_id"]: row for row in body["rows"]}
    ids = [r["transaction_id"] for r in rows.to_dict("records")]
    probs = np.array([by_id[tid]["fraud_probability"] for tid in ids])
    amounts = np.array([by_id[tid]["amount"] for tid in ids])
    uncapped = amount_aware_actions(probs, amounts, scorer.params)
    for tid, action in zip(ids, uncapped, strict=True):
        if action != REVIEW:
            assert by_id[tid]["recommended_action"] == ACTION_NAMES[int(action)], tid
    assert body["summary"]["review_workload"] <= scorer.capacity
    assert "REVIEW" in body["summary"]["capacity_note"]


def test_batch_ordered_by_expected_saving(client):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    body = client.post("/predict/batch", json={"transactions": [payload] * 5}).json()
    savings = [row["expected_saving"] for row in body["rows"]]
    assert savings == sorted(savings, reverse=True)


def test_predict_is_deterministic(client):
    payload = json.loads(open("examples/sample_request.json", encoding="utf-8").read())
    payload["include_explanation"] = True
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first["fraud_probability"] == second["fraud_probability"]
    assert first["recommended_action"] == second["recommended_action"]
    assert first["reasons"] == second["reasons"]


# --- leakage ------------------------------------------------------------------


def test_history_provider_never_returns_future_rows(client, scorer, raw_transactions):
    day = pd.Timestamp(VALIDATION_DAY).date()
    row = raw_transactions[raw_transactions["timestamp"].dt.date == day].iloc[0]
    cutoff = row["timestamp"]
    history = scorer.provider.history(row["customer_id"], row["device_id"], cutoff)
    assert len(history), "expected prior history for a validation customer"
    assert (history["timestamp"] < cutoff).all()
    assert (history["transaction_id"] != row["transaction_id"]).all()


@pytest.mark.db
def test_history_provider_strictly_prior_postgres():
    from sqlalchemy import text

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT customer_id, MAX(timestamp) AS cutoff FROM raw.transactions "
                "GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 1"
            )
        ).first()
    if row is None:
        pytest.skip("raw.transactions is empty; load a profile first")

    cutoff = row.cutoff
    history = queries.fetch_customer_history(row.customer_id, cutoff, engine=engine)
    assert history and all(r["timestamp"] < cutoff for r in history)
    device = queries.fetch_transactions(limit=1, engine=engine)[0].get("device_id")
    if device:
        device_rows = queries.fetch_device_history(device, cutoff, engine=engine)
        assert all(r["timestamp"] < cutoff for r in device_rows)


@pytest.mark.db
def test_postgres_history_matches_memory_features():
    """PostgreSQL and in-memory providers must produce the same feature row.

    Feature-level equality is the right check because the ``tiny`` profile has no
    calibrator/policy: identical strictly-prior history => identical features =>
    identical calibrated probability. The 1e-6 calibrated-probability parity claim
    stays scoped to the dev/memory path (``test_parity_matches_offline_pipeline``).
    The ``tiny`` tables are reloaded (this test owns the DB state, like the rest of
    the ``db``-marked suite).
    """
    from sqlalchemy import text

    from fraudlens.api.deps import load_entities
    from fraudlens.api.history import InMemoryHistoryProvider, PostgresHistoryProvider
    from fraudlens.db.loader import load_all
    from fraudlens.features.registry import FEATURE_COLUMNS

    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    tiny_path = raw_data_dir("tiny") / "transactions.parquet"
    if not tiny_path.exists():
        pytest.skip("tiny parquet data missing; run scripts/generate_data.py --profile tiny")
    load_all("tiny", reset=True, write_explain=False)

    tx = pd.read_parquet(tiny_path).sort_values(
        ["timestamp", "transaction_id"], kind="stable"
    ).reset_index(drop=True)
    postgres = PostgresHistoryProvider()
    memory = InMemoryHistoryProvider("tiny")
    cfg = load_config("tiny")
    entities = load_entities("tiny")

    pool = tx[tx["timestamp"] < tx["timestamp"].max()]
    assert len(pool) > 20, "tiny window too small to sample prior history"
    rng = np.random.default_rng(42)
    sample = pool.iloc[np.sort(rng.choice(len(pool), size=25, replace=False))]

    empty = 0
    for _, row in sample.iterrows():
        tx_row = row.to_dict()
        before = tx_row["timestamp"]
        pg_history = postgres.history(tx_row["customer_id"], tx_row.get("device_id"), before)
        mem_history = memory.history(tx_row["customer_id"], tx_row.get("device_id"), before)
        assert len(pg_history) == len(mem_history), (
            f"provider history length mismatch for {tx_row['transaction_id']}: "
            f"{len(pg_history)} != {len(mem_history)}"
        )
        if not len(pg_history):
            empty += 1
            continue
        assert (pg_history["timestamp"] < before).all()
        assert sorted(pg_history["transaction_id"]) == sorted(mem_history["transaction_id"]), (
            f"provider history mismatch for {tx_row['transaction_id']}"
        )

        pg_features = compute_features_for_transaction(tx_row, pg_history, entities, cfg)
        mem_features = compute_features_for_transaction(tx_row, mem_history, entities, cfg)
        for column in FEATURE_COLUMNS:
            left, right = pg_features[column], mem_features[column]
            if pd.isna(left) and pd.isna(right):
                continue
            assert left == right, (
                f"postgres/memory feature mismatch on {column} "
                f"for {tx_row['transaction_id']}: {left!r} != {right!r}"
            )
    assert empty < len(sample), "no sampled row had prior history"


# --- parity -------------------------------------------------------------------


def test_parity_matches_offline_pipeline(client, scorer):
    val, boundaries = _val_frame(scorer)
    t_train_end = pd.Timestamp(boundaries["t_train_end"])
    t_val_end = pd.Timestamp(boundaries["t_val_end"])

    raw = pd.read_parquet(raw_data_dir("dev") / "transactions.parquet").set_index(
        "transaction_id"
    )
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(len(val), size=220, replace=False))
    sample = val.iloc[sample_idx]

    # Hard-fail: every sampled row must be inside the validation window.
    outside = sample[(sample["timestamp"] < t_train_end) | (sample["timestamp"] >= t_val_end)]
    assert outside.empty, f"{len(outside)} sampled rows fall outside the validation window"

    x = encode_features(sample)
    offline_p = scorer.calibrator.predict(scorer.model.predict_proba(x)[:, 1])

    worst = 0.0
    for position, row in enumerate(sample.itertuples(index=False)):
        request = _request_from_row(
            raw.loc[row.transaction_id].to_dict(), transaction_id=row.transaction_id
        )
        response = client.post("/predict", json=request)
        assert response.status_code == 200, (row.transaction_id, response.text)
        diff = abs(response.json()["fraud_probability"] - float(offline_p[position]))
        worst = max(worst, diff)

    print(f"parity: n={len(sample)} max_abs_diff={worst:.3e}")
    assert worst <= 1e-6, f"training/serving parity broken: max abs diff {worst:.3e}"


# --- startup guard ------------------------------------------------------------


def test_startup_refuses_tampered_artifact(tmp_path, monkeypatch):
    source = artifacts_dir("xgb_dev_20261015_v1")
    destination = tmp_path / "artifacts" / "xgb_dev_20261015_v1"
    shutil.copytree(source, destination)

    freeze = json.loads((results_dir() / "freeze.json").read_text(encoding="utf-8"))
    tampered = {
        "models": {"xgb": dict(freeze["models"]["xgb"])},
        "data_files": {},
    }
    tampered["models"]["xgb"]["files"] = dict(freeze["models"]["xgb"]["files"])
    tampered["models"]["xgb"]["files"]["model.joblib"] = "0" * 64
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")

    monkeypatch.setattr(locked, "artifacts_dir", lambda version: tmp_path / "artifacts" / version)
    monkeypatch.setattr(locked, "resolve_freeze_path", lambda key: tmp_path / key)

    app = create_app(
        profile="dev", history_backend="memory", freeze_path=str(freeze_path)
    )
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass

    # Control: the untampered copy verifies (the refusal was the tampered hash).
    locked.verify_artifact_hashes({"models": {"xgb": freeze["models"]["xgb"]}, "data_files": {}})

    # The real artifact on disk was never touched.
    real_hash = sha256_file(artifacts_dir("xgb_dev_20261015_v1") / "model.joblib")
    assert real_hash == freeze["models"]["xgb"]["files"]["model.joblib"]
