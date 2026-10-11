"""Model artifact I/O (Part 06F).

Each artifact directory holds the fitted model (joblib), a ``metadata.json``
(model_version, feature_version, windows, hyperparameters, seed, git commit,
config hash, validation metrics) and the raw validation scores.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraudlens.paths import artifacts_dir
from fraudlens.utils import git_commit


def build_metadata(
    *,
    model_version: str,
    feature_version: str,
    profile: str,
    train_window: dict[str, str],
    val_window: dict[str, str],
    calibration: str,
    hyperparameters: dict[str, Any],
    seed: int,
    metrics: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    """Assemble the artifact metadata dict (all fields required by AGENTS.md)."""
    return {
        "model_version": model_version,
        "feature_version": feature_version,
        "profile": profile,
        "train_window": train_window,
        "val_window": val_window,
        "calibration": calibration,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "metrics": metrics,
        "git_commit": git_commit(),
        "config_hash": config_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_model(
    model: Any,
    model_version: str,
    metadata: dict[str, Any],
    val_scores: pd.DataFrame | None = None,
) -> Path:
    """Persist the model + metadata (+ optional val scores); return the dir."""
    out = artifacts_dir(model_version)
    joblib.dump(model, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if val_scores is not None:
        val_scores.to_parquet(out / "val_scores.parquet", index=False)
    return out


def load_model(model_version: str) -> Any:
    """Load a fitted model by version."""
    return joblib.load(artifacts_dir(model_version) / "model.joblib")


def load_metadata(model_version: str) -> dict[str, Any]:
    return json.loads((artifacts_dir(model_version) / "metadata.json").read_text(encoding="utf-8"))
