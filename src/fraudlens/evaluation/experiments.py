"""Experiment log (Part 06C).

Every experiment appends one row to ``results/experiments.csv`` so results are
traceable to a commit, a timestamp and a config hash. Columns follow AGENTS.md;
extra keys are appended at the end rather than dropped.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from fraudlens.paths import results_dir
from fraudlens.utils import git_commit

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "part",
    "dataset_profile",
    "model",
    "features_version",
    "train_window",
    "val_window",
    "test_window",
    "hyperparameters",
    "calibration",
    "threshold_policy",
    "metrics",
    "notes",
    "git_commit",
    "timestamp",
    "config_hash",
]


def log_experiment(
    row: dict[str, Any], path: str | Path | None = None, config_hash: str | None = None
) -> Path:
    """Append ``row`` to the experiment CSV (auto-filling commit/time/hash)."""
    out = Path(path) if path is not None else Path(results_dir()) / "experiments.csv"
    row = dict(row)
    row.setdefault("git_commit", git_commit())
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    if config_hash is not None:
        row.setdefault("config_hash", config_hash)

    existing = _read_existing_columns(out)
    fieldnames = list(EXPERIMENT_COLUMNS)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    for col in existing:
        if col not in fieldnames:
            fieldnames.append(col)

    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: _stringify(v) for k, v in row.items()})
    return out


def _read_existing_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, default=str)
    return value
