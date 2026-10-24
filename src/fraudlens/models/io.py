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
    """Assemble the artifact metadata dict (all fields required by project conventions)."""
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


def load_serving_bundle(model_version: str) -> dict[str, Any]:
    """Load everything needed to score: model, calibrator, policy, metadata.

    The calibrator and policy are written by ``scripts/calibrate.py`` and
    ``scripts/optimize_policy.py`` into the same artifact directory.
    """
    version_dir = artifacts_dir(model_version)
    bundle: dict[str, Any] = {
        "model": load_model(model_version),
        "metadata": load_metadata(model_version),
    }
    calibrator_path = version_dir / "calibrator.joblib"
    policy_path = version_dir / "policy.json"
    bundle["calibrator"] = joblib.load(calibrator_path) if calibrator_path.exists() else None
    bundle["policy"] = (
        json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else None
    )
    return bundle


def load_metadata(model_version: str) -> dict[str, Any]:
    return json.loads((artifacts_dir(model_version) / "metadata.json").read_text(encoding="utf-8"))


def assert_artifact_profile(model_version: str, profile: str) -> None:
    """Refuse to write into an artifact whose metadata belongs to another profile.

    ``results/models_<profile>.json`` can map two profiles to the same version when
    version names omit the profile, so a run for profile B could silently overwrite
    profile A's calibrator/policy (see docs/what_went_wrong.md, 2026-10-12).
    """
    meta_path = artifacts_dir(model_version) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing; cannot verify profile of {model_version!r}")
    found = json.loads(meta_path.read_text(encoding="utf-8")).get("profile")
    if found != profile:
        raise ValueError(
            f"artifact {model_version!r} belongs to profile {found!r}; refusing to write "
            f"{profile!r} calibrator/policy into it"
        )
