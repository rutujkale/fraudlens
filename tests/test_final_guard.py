"""Locked-evaluation guard tests (Part 12E).

Covers the freeze verifier: hash checks (both path separators), the clean-tree /
HEAD-contains-freeze precondition, and that a disclosure row (``final=False``) is
not mistaken for a completed evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json

import pytest

from fraudlens.evaluation import locked
from fraudlens.evaluation.freeze import sha256_file
from fraudlens.evaluation.guard import TestSetGuard


def _write_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "caller", "model_version", "final", "git_commit", "reason"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_guard_denies_without_final(tmp_path):
    log = tmp_path / "test_access_log.csv"
    with pytest.raises(PermissionError, match="denied"):
        TestSetGuard(log_path=log).access("someone", final=False)
    with open(log, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and rows[0]["final"] == "False"  # the attempt is still logged


def test_guard_allows_final(tmp_path):
    log = tmp_path / "test_access_log.csv"
    assert TestSetGuard(log_path=log).access("final_eval", final=True) is True


def test_resolve_accepts_both_separators():
    assert locked.resolve_freeze_path("data\\features\\x.parquet") == locked.resolve_freeze_path(
        "data/features/x.parquet"
    )


def test_verify_hashes_passes_then_aborts_on_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(locked, "REPO_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    f = d / "x.parquet"
    f.write_bytes(b"frozen")
    freeze = {"models": {}, "data_files": {"data\\x.parquet": sha256_file(f)}}
    locked.verify_artifact_hashes(freeze)  # no raise
    f.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        locked.verify_artifact_hashes(freeze)


def test_verify_artifact_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(locked, "artifacts_dir", lambda v: tmp_path / v)
    d = tmp_path / "xgb_x"
    d.mkdir()
    (d / "model.joblib").write_bytes(b"model")
    freeze = {"models": {"xgb": {"model_version": "xgb_x", "files": {"model.joblib": "0" * 64}}}}
    with pytest.raises(RuntimeError, match="hash mismatch"):
        locked.verify_artifact_hashes(freeze)


def test_final_false_row_is_not_an_evaluation(tmp_path):
    log = tmp_path / "test_access_log.csv"
    _write_log(log, [{"timestamp": "t", "caller": "disclosure", "final": "False"}])
    assert locked.final_evaluation_count(log) == 0
    _write_log(
        log,
        [
            {"timestamp": "t", "caller": "disclosure", "final": "False"},
            {"timestamp": "t", "caller": "final_eval", "final": "True"},
        ],
    )
    assert locked.final_evaluation_count(log) == 1


def test_reconstructed_hash_matches_frozen_training_hash():
    from fraudlens.config import load_config
    from fraudlens.paths import results_dir

    freeze_path = results_dir() / "freeze.json"
    if not freeze_path.exists():
        pytest.skip("freeze.json not present")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = frozen["models"]["xgb"]["config_hash_at_training"]
    assert locked.reconstructed_training_config_hash(load_config("dev")) == expected


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_headline_builder_requires_keys(tmp_path):
    from fraudlens.evaluation.headlines import load_final_results

    p = tmp_path / "final_results.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="missing required keys"):
        load_final_results(p)
