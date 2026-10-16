"""Freeze tooling tests (Part 12A).

The pure helpers are tested directly; the dirty-tree and cross-profile guards are
exercised by monkeypatching the git/artifact lookups so no real data is needed.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from fraudlens.evaluation import freeze


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"fraudlens" * 1000)
    assert freeze.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_artifact_hashes_covers_every_file(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "artifacts_dir", lambda v: tmp_path / v)
    d = tmp_path / "xgb_dev_v1"
    d.mkdir()
    (d / "model.joblib").write_bytes(b"model")
    (d / "calibrator.joblib").write_bytes(b"cal")
    (d / "policy.json").write_text("{}", encoding="utf-8")

    hashes = freeze.artifact_hashes("xgb_dev_v1")
    assert set(hashes) == {"model.joblib", "calibrator.joblib", "policy.json"}
    assert all(len(h) == 64 for h in hashes.values())


def test_missing_artifact_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "artifacts_dir", lambda v: tmp_path / v)
    (tmp_path / "empty_v1").mkdir()
    with pytest.raises(FileNotFoundError):
        freeze.artifact_hashes("empty_v1")


def test_pre_registered_success_has_h1_to_h4():
    keys = set(freeze.PRE_REGISTERED_SUCCESS)
    assert keys == {"H1_model", "H2_decision", "H3_unseen", "H4_engineering"}
    assert "Part 13" in freeze.PRE_REGISTERED_SUCCESS["H4_engineering"]


def test_freeze_refuses_dirty_tree(monkeypatch):
    monkeypatch.setattr(freeze, "git_status_porcelain", lambda: " M src/x.py\n")
    with pytest.raises(RuntimeError, match="dirty"):
        freeze.build_freeze("tiny")


def test_force_requires_a_reason(monkeypatch):
    monkeypatch.setattr(freeze, "git_status_porcelain", lambda: " M src/x.py\n")
    with pytest.raises(ValueError, match="requires"):
        freeze.build_freeze("tiny", force=True, force_reason=None)


def test_freeze_refuses_cross_profile_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "git_status_porcelain", lambda: "")
    monkeypatch.setattr(freeze, "results_dir", lambda: tmp_path)
    (tmp_path / "models_dev.json").write_text(
        json.dumps({"logreg": "logreg_x", "xgb": "xgb_x"}), encoding="utf-8"
    )
    monkeypatch.setattr(freeze, "load_metadata", lambda v: {"profile": "tiny"})
    with pytest.raises(ValueError, match="cross-profile"):
        freeze.build_freeze("dev")
