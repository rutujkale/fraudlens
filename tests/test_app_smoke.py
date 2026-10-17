"""Smoke tests for the Streamlit analyst console (Part 14).

The pages are exercised with ``streamlit.testing.v1.AppTest`` against the ``tiny``
artifacts (built into a temp dir, never committed) and a fake API client. The fake
responses are built from ``fraudlens.api.schemas`` and ``.model_dump()``ed, so any
schema drift breaks the test rather than the page.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

_TMP = Path(tempfile.mkdtemp(prefix="fl_app_smoke_"))
os.environ["FRAUDLENS_PROFILE"] = "tiny"
os.environ["FRAUDLENS_APP_ARTIFACTS_DIR"] = str(_TMP)

from components import api_client  # noqa: E402
from components.artifacts import _load as _artifacts_load  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from fraudlens.api import schemas  # noqa: E402

ENTRY = ROOT / "app" / "streamlit_app.py"
PAGE1 = APP / "pages" / "1_Transaction_Analyzer.py"
PAGE2 = APP / "pages" / "2_Review_Queue.py"
PAGE3 = APP / "pages" / "3_Model_Performance.py"

TIMEOUT = 300


@pytest.fixture(scope="session", autouse=True)
def tiny_artifacts():
    from fraudlens.appdata.artifacts import build_app_artifacts

    build_app_artifacts("tiny", out_dir=_TMP / "tiny", n_scored=2000, n_shap=200)


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    api_client.set_client(None)


def _predict_response() -> dict:
    return schemas.PredictResponse(
        transaction_id="tx_demo",
        fraud_probability=0.8731,
        raw_score=1.9,
        risk_score=87.3,
        base_rate=0.00765,
        lift_over_base_rate=114.1,
        risk_level="HIGH",
        recommended_action="REVIEW",
        reasons=["amount far above the customer's usual spend"],
        mitigating_factors=["known device"],
        contributions=[
            schemas.ShapContribution(
                feature="amount_vs_lifetime_median",
                value=12.0,
                shap_value=0.9,
                direction="increases",
            ),
            schemas.ShapContribution(
                feature="is_known_device", value=1.0, shap_value=-0.2, direction="decreases"
            ),
        ],
        model_version="xgb_demo",
        feature_version="features_v1",
        policy_version="cost_policy_v1",
        latency_ms=42.0,
        timing_ms=schemas.TimingMs(
            history_fetch=20.0,
            features=15.0,
            model=5.0,
            calibration=1.0,
            policy=0.5,
            explanation=0.5,
        ),
    ).model_dump()


class FakeClient:
    """Minimal stand-in whose payloads are validated by the real Pydantic models."""

    def predict(self, payload: dict) -> dict:
        return _predict_response()

    def customer_summary(self, customer_id: str, before: str | None = None) -> dict:
        return schemas.CustomerSummaryResponse(
            customer_id=customer_id,
            before=before,
            n_prior_tx=7,
            typical_amount=1800.0,
            usual_hours=[13, 19],
            known_devices=["dev_1"],
            recent=[],
        ).model_dump()

    def batch_csv(self, data: bytes, filename: str = "batch.csv") -> dict:
        return schemas.BatchResponse(
            rows=[],
            summary=schemas.BatchSummary(
                n=0,
                counts_by_action={},
                review_workload=0,
                n_demoted=0,
                total_expected_loss_avoided=0.0,
                capacity_note="capacity caps REVIEW only",
            ),
        ).model_dump()


class DownClient:
    def predict(self, payload: dict) -> dict:
        raise api_client.ApiError("connection refused")

    def customer_summary(self, customer_id: str, before: str | None = None) -> dict:
        raise api_client.ApiError("connection refused")


def _run(script: Path, times: int = 1) -> AppTest:
    at = AppTest.from_file(str(script), default_timeout=TIMEOUT)
    for _ in range(times):
        at.run()
    return at


def _text(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_pages_compile():
    scripts = [ENTRY, PAGE1, PAGE2, PAGE3]
    for path in scripts:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_entry_page_renders():
    at = _run(ENTRY)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("FraudLens" in t.value for t in at.title)
    assert "Synthetic data demo" in _text(at.caption)


def test_page1_scores_and_shows_contributions():
    api_client.set_client(FakeClient())
    at = _run(PAGE1)
    at.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Recommended action" in [m.label for m in at.metric]
    assert "Customer context" in [s.value for s in at.subheader]
    assert "log-odds (not points on the risk score)" in _text(at.caption)


def test_page1_api_down_message_names_start_command():
    api_client.set_client(DownClient())
    at = _run(PAGE1)
    at.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    text = _text(at.info) + " " + _text(at.error)
    assert "scripts/run_api.py" in text


def test_page2_hides_labels_by_default():
    at = _run(PAGE2)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert all("is_fraud" not in set(df.value.columns) for df in at.dataframe)
    assert not any("Realized" in m.label for m in at.metric)
    assert at.toggle[0].value is False

    at.toggle[0].set_value(True).run()
    assert any("is_fraud" in set(df.value.columns) for df in at.dataframe)
    assert any("Realized" in m.label for m in at.metric)


def _sample_with_positives(n_legit: int = 280, n_fraud: int = 20, seed: int = 42) -> pd.DataFrame:
    """A scored sample that contains frauds (the tiny validation slice has none)."""
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "timestamp": pd.date_range("2026-03-20", periods=n, freq="30min"),
            "amount": np.round(rng.uniform(100.0, 50_000.0, n), 2),
            "fraud_probability": np.concatenate(
                [rng.uniform(0.0, 0.05, n_legit), rng.uniform(0.30, 0.95, n_fraud)]
            ),
            "is_fraud": np.array([0] * n_legit + [1] * n_fraud),
        }
    )


def test_page2_recompute_uses_package_code():
    from fraudlens.appdata import recompute
    from fraudlens.appdata.artifacts import load_app_artifacts
    from fraudlens.evaluation.cost_model import CostParams
    from fraudlens.evaluation.threshold_search import scan_global_thresholds

    sample = _sample_with_positives()
    assert sample["is_fraud"].sum() >= 1, "the equality must hold on data with positives"

    params = CostParams(**load_app_artifacts("tiny").manifest["costs"])
    capacity = int(load_app_artifacts("tiny").manifest["capacity_per_day"])
    y, prob, amount, dates = recompute.sample_arrays(sample)

    expected = scan_global_thresholds(
        y, prob, amount, dates, params, recompute.threshold_grid(prob)
    )
    got = recompute.scan(sample, params)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))

    best = recompute.best_thresholds(sample, params, capacity)
    cheapest = float(expected.loc[expected["total_realized_cost"].idxmin(), "threshold"])
    assert best["cost_optimal_threshold"] == cheapest


def test_missing_artifacts_names_build_script(monkeypatch):
    empty = Path(tempfile.mkdtemp(prefix="fl_app_empty_"))
    monkeypatch.setenv("FRAUDLENS_APP_ARTIFACTS_DIR", str(empty))
    _artifacts_load.clear()
    try:
        at = _run(ENTRY)
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "scripts/make_app_artifacts.py" in _text(at.error)
    finally:
        monkeypatch.undo()
        _artifacts_load.clear()
