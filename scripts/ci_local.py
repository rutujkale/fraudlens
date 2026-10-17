"""Thin CLI: reproduce ``.github/workflows/ci.yml`` locally without dirtying tracked files.

The tiny smoke run rewrites a few tracked, profile-suffixed files
(``results/models_tiny.json``, ``results/experiments.csv``, ``reports/db_explain_tiny.txt``,
``docs/data_dictionary_features.md``). This wrapper snapshots them, runs the exact CI
steps, then restores the snapshots — so ``make ci-local`` never leaves the repo dirty
(and never touches any other edit you had open). The dev profile and the local DB are
not reloaded; run ``python scripts/load_db.py --profile dev`` afterwards if you need the
API's postgres backend on ``dev``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fraudlens.paths import REPO_ROOT

PY = sys.executable

# Files the tiny pipeline rewrites that are tracked by git.
TOUCHED = [
    "results/models_tiny.json",
    "results/experiments.csv",
    "reports/db_explain_tiny.txt",
    "docs/data_dictionary_features.md",
]

STEPS: list[list[str]] = [
    [PY, "-m", "ruff", "check", "."],
    [PY, "scripts/generate_data.py", "--profile", "tiny"],
    [PY, "scripts/validate_data.py", "--profile", "tiny"],
    [PY, "scripts/build_features.py", "--profile", "tiny"],
    [PY, "scripts/train.py", "--profile", "tiny", "--model", "all"],
    [PY, "scripts/load_db.py", "--profile", "tiny"],
    [PY, "-m", "pytest", "-m", "not slow and not needs_dev"],
]


def _snapshot() -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    for rel in TOUCHED:
        path = REPO_ROOT / rel
        out[rel] = path.read_bytes() if path.exists() else None
    return out


def _restore(snapshot: dict[str, bytes | None]) -> None:
    for rel, content in snapshot.items():
        path: Path = REPO_ROOT / rel
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def main() -> int:
    snapshot = _snapshot()
    status = 0
    try:
        for step in STEPS:
            sys.stdout.write("+ " + " ".join(step) + "\n")
            proc = subprocess.run(step, cwd=REPO_ROOT)
            if proc.returncode != 0:
                status = proc.returncode
                break
    finally:
        _restore(snapshot)
        sys.stdout.write("restored tracked files touched by the tiny smoke run\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
