"""Experiment log (Part 06C).

Every experiment appends one row to ``results/experiments.csv`` so results are
traceable to a commit, a timestamp and a config hash. Columns follow the project
conventions in ``docs/methodology.md``; extra keys are appended at the end rather
than dropped.
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
    "dataset_version",
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


def drop_experiment_rows(
    experiment_id: str,
    profile: str | None = None,
    path: str | Path | None = None,
    dataset_version: str | None = None,
) -> int:
    """Remove prior rows for ``experiment_id`` so re-runs stay idempotent.

    ``log_experiment`` appends blindly (see the Part 09 duplicate-E15 finding in
    ``docs/what_went_wrong.md``); call this before logging a re-runnable experiment.
    Returns the number of rows removed.

    ``dataset_version`` narrows the match to one data generation (Part 02b): rows
    from a *different* dataset version are history and are kept, so regenerated
    runs replace their own rows instead of erasing the previous dataset's evidence.
    """
    out = Path(path) if path is not None else Path(results_dir()) / "experiments.csv"
    if not out.exists():
        return 0
    import pandas as pd

    df = pd.read_csv(out)
    dupe = df["experiment_id"] == experiment_id
    if profile is not None:
        dupe &= df["dataset_profile"] == profile
    if dataset_version is not None:
        version = df["dataset_version"].astype(str) if "dataset_version" in df.columns else ""
        dupe &= version == dataset_version
    if not dupe.any():
        return 0
    df[~dupe].to_csv(out, index=False)
    return int(dupe.sum())


def backfill_dataset_version(version: str, path: str | Path | None = None) -> int:
    """Stamp ``version`` on experiment rows that have no ``dataset_version`` yet.

    Existing rows predate the column, so they describe the pre-fix dataset; this
    attaches that label rather than leaving them ambiguous. Returns rows stamped.
    """
    import pandas as pd

    out = Path(path) if path is not None else Path(results_dir()) / "experiments.csv"
    if not out.exists():
        return 0
    df = pd.read_csv(out)
    if "dataset_version" not in df.columns:
        df["dataset_version"] = version
        stamped = len(df)
    else:
        missing = df["dataset_version"].isna() | (df["dataset_version"].astype(str) == "")
        df.loc[missing, "dataset_version"] = version
        stamped = int(missing.sum())
    other = [c for c in df.columns if c not in EXPERIMENT_COLUMNS]
    order = [c for c in EXPERIMENT_COLUMNS if c in df.columns] + other
    df[order].to_csv(out, index=False)
    return stamped


def _ensure_schema(path: Path) -> None:
    """Rewrite the CSV if its header predates a column, so appended rows stay aligned."""
    if not path.exists():
        return
    existing = _read_existing_columns(path)
    if not existing:
        return
    missing = [c for c in EXPERIMENT_COLUMNS if c not in existing]
    if not missing:
        return
    import pandas as pd

    df = pd.read_csv(path)
    for col in missing:
        df[col] = ""
    other = [c for c in df.columns if c not in EXPERIMENT_COLUMNS]
    order = [c for c in EXPERIMENT_COLUMNS if c in df.columns] + other
    df[order].to_csv(path, index=False)


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
    row.setdefault("dataset_version", _config_dataset_version(row.get("dataset_profile")))
    _ensure_schema(out)

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


def _config_dataset_version(profile: Any) -> str:
    """Best-effort lookup of the profile's ``dataset_version`` label."""
    if not isinstance(profile, str):
        return "unknown"
    try:
        from fraudlens.config import load_config

        return load_config(profile).simulation.dataset_version
    except Exception:  # pragma: no cover - unknown profile / config unavailable
        return "unknown"


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
