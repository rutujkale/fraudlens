"""Part 12B locked-evaluation verification helpers.

Before the test window is opened, ``final_evaluation.py`` must prove that nothing
frozen has moved: every artifact/data hash matches ``results/freeze.json``, the
tree is clean and HEAD already contains the freeze, and the test access log shows
no completed evaluation. These helpers are unit-tested in
``tests/test_final_guard.py``; the script stays thin.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from fraudlens.config import Config
from fraudlens.evaluation.freeze import sha256_file
from fraudlens.paths import REPO_ROOT, artifacts_dir, results_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

# Fields added to the config *after* the frozen artifacts were trained. Removing
# them reconstructs the training-time config (see
# ``reconstructed_training_config_hash`` and docs/progress/part_12.md).
POST_TRAINING_SIMULATION_KEYS = ("external",)
POST_TRAINING_XGB_KEYS = ("suspicious_pr_auc_threshold", "suspicious_pr_auc_bypass_min_positives")


def resolve_freeze_path(key: str) -> Path:
    """Resolve a freeze key to an absolute path, accepting ``/`` or ``\\`` separators."""
    return REPO_ROOT / Path(key.replace("\\", "/"))


def verify_artifact_hashes(freeze: dict) -> None:
    """Raise if any frozen artifact or data file hash differs from ``freeze``."""
    mismatches: list[str] = []
    for info in freeze.get("models", {}).values():
        for fname, expected in info["files"].items():
            path = artifacts_dir(info["model_version"]) / fname
            if not path.exists() or sha256_file(path) != expected:
                mismatches.append(str(path))
    for key, expected in freeze.get("data_files", {}).items():
        path = resolve_freeze_path(key)
        if not path.exists() or sha256_file(path) != expected:
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError("freeze hash mismatch for: " + "; ".join(mismatches))


def final_evaluation_count(log_path: str | Path | None = None) -> int:
    """Number of completed (``final=True``) evaluations recorded in the access log.

    A disclosure row (``final=False``) is not an evaluation, so it does not count.
    """
    path = Path(log_path) if log_path else results_dir() / "test_access_log.csv"
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sum(1 for r in rows if str(r.get("final", "")).strip().lower() in {"true", "1"})


def assert_head_has_freeze(rel_path: str = "results/freeze.json") -> None:
    """Raise unless the working tree is clean and HEAD already tracks the freeze file."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if status.strip():
        raise RuntimeError(f"working tree is dirty:\n{status}")
    found = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if found.returncode != 0:
        raise RuntimeError(f"HEAD does not contain {rel_path!r}; commit the freeze first")


def reconstructed_training_config_hash(cfg: Config) -> str:
    """Hash the config with the post-training keys removed.

    Proves the dev-relevant sections (simulation, splits, costs, models, rules,
    features, scenario mixes, seed) are byte-identical to the config the frozen
    artifacts were trained under. Only ``external`` (added in Part 11) and the two
    ``xgb`` leakage-guard flags (added in Part 02b) are dropped.
    """
    sim = cfg.model_dump(mode="json")["simulation"]
    sim = {k: v for k, v in sim.items() if k not in POST_TRAINING_SIMULATION_KEYS}
    for field in POST_TRAINING_XGB_KEYS:
        sim["models"]["xgb"].pop(field, None)
    canonical = json.dumps({"simulation": sim}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
