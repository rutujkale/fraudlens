"""Experiment log tests (Part 06C / Part 02b)."""

from __future__ import annotations

import csv

from fraudlens.evaluation.experiments import (
    EXPERIMENT_COLUMNS,
    backfill_dataset_version,
    drop_experiment_rows,
    log_experiment,
)
from fraudlens.evaluation.experiments_lib import FALLBACK_XGB_PARAMS, best_xgb_params


def test_best_xgb_params_falls_back_when_artifact_is_not_local(tmp_path, monkeypatch):
    """A fresh clone has results/models_<p>.json but no (gitignored) artifacts/."""
    from fraudlens.evaluation import experiments_lib as lib

    (tmp_path / "models_ci.json").write_text('{"xgb": "xgb_ci_missing"}', encoding="utf-8")
    monkeypatch.setattr(lib, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(lib, "artifacts_dir", lambda version: tmp_path / version)
    assert best_xgb_params("ci") == FALLBACK_XGB_PARAMS


def test_log_experiment_appends_and_injects_metadata(tmp_path):
    out = tmp_path / "experiments.csv"
    log_experiment(
        {"experiment_id": "E01", "model": "rules", "metrics": {"pr_auc": 0.1}},
        path=out,
        config_hash="abc123",
    )
    log_experiment({"experiment_id": "E02", "model": "logreg"}, path=out, config_hash="abc123")

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert rows[0]["git_commit"]  # injected
    assert rows[0]["timestamp"]
    assert rows[0]["config_hash"] == "abc123"
    assert rows[0]["metrics"] == '{"pr_auc": 0.1}'  # dicts serialized
    assert "experiment_id" in EXPERIMENT_COLUMNS


def test_drop_is_scoped_to_dataset_version(tmp_path):
    out = tmp_path / "experiments.csv"
    log_experiment(
        {"experiment_id": "E14", "dataset_profile": "drift_lab",
         "dataset_version": "pre_coldstart_fix", "model": "a"},
        path=out,
    )
    log_experiment(
        {"experiment_id": "E14", "dataset_profile": "drift_lab",
         "dataset_version": "post_coldstart_fix", "model": "b"},
        path=out,
    )
    removed = drop_experiment_rows(
        "E14", "drift_lab", path=out, dataset_version="post_coldstart_fix"
    )
    assert removed == 1
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["dataset_version"] for r in rows] == ["pre_coldstart_fix"]


def test_backfill_stamps_rows_without_a_version(tmp_path):
    out = tmp_path / "experiments.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment_id", "part"])
        writer.writeheader()
        writer.writerow({"experiment_id": "E01", "part": "06"})

    assert backfill_dataset_version("pre_coldstart_fix", path=out) == 1
    assert backfill_dataset_version("pre_coldstart_fix", path=out) == 0  # idempotent
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["dataset_version"] == "pre_coldstart_fix"


def test_appending_to_legacy_header_stays_aligned(tmp_path):
    """A header written before the column existed is migrated, not misaligned."""
    out = tmp_path / "experiments.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment_id", "part", "model"])
        writer.writeheader()
        writer.writerow({"experiment_id": "E01", "part": "06", "model": "rules"})

    log_experiment(
        {"experiment_id": "E02", "part": "06", "model": "logreg",
         "dataset_version": "post_coldstart_fix"},
        path=out,
    )
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["experiment_id"] for r in rows] == ["E01", "E02"]
    assert rows[1]["model"] == "logreg"  # not shifted by the added column
    assert rows[1]["dataset_version"] == "post_coldstart_fix"
