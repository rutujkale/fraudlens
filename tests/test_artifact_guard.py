"""Artifact writes must not cross profiles (Part 10 S0).

A ``tiny`` calibration run once overwrote the ``dev`` model's calibrator and
policy because both profiles mapped to ``xgb_20261011_v1``; see
``docs/what_went_wrong.md``.
"""

from __future__ import annotations

import json

import pytest

from fraudlens.models import io


def _fake_artifact(tmp_path, monkeypatch, version: str, profile: str) -> None:
    monkeypatch.setattr(io, "artifacts_dir", lambda v: tmp_path / v)
    d = tmp_path / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps({"model_version": version, "profile": profile}), encoding="utf-8"
    )


def test_guard_blocks_mismatched_profile(tmp_path, monkeypatch):
    _fake_artifact(tmp_path, monkeypatch, "xgb_20261011_v1", "dev")
    with pytest.raises(ValueError, match="belongs to profile"):
        io.assert_artifact_profile("xgb_20261011_v1", "tiny")


def test_guard_allows_matching_profile(tmp_path, monkeypatch):
    _fake_artifact(tmp_path, monkeypatch, "xgb_dev_v1", "dev")
    io.assert_artifact_profile("xgb_dev_v1", "dev")


def test_guard_requires_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "artifacts_dir", lambda v: tmp_path / v)
    (tmp_path / "xgb_x_v1").mkdir()
    with pytest.raises(FileNotFoundError):
        io.assert_artifact_profile("xgb_x_v1", "dev")
