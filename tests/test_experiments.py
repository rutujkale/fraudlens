"""Experiment log tests (Part 06C)."""

from __future__ import annotations

import csv

from fraudlens.evaluation.experiments import EXPERIMENT_COLUMNS, log_experiment


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
