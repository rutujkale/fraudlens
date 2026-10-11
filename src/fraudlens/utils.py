"""Small cross-cutting helpers: seeds, logging, git, directories."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (when imported) torch-free RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Module logger with a single stream handler at INFO level."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def git_commit() -> str:
    """Current short/long HEAD hash, or 'unknown' when not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        return "unknown"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p