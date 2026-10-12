"""Raw validation scores for the fitted models (Part 07).

Shared by ``evaluate_val.py``, ``calibrate.py`` and ``optimize_policy.py`` so the
"load the versions, predict the validation rows" logic exists in one place.
Rules scores are recomputed (the engine is deterministic and cheap); logreg/xgb
are loaded from their artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.evaluation.splits import Split
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.models.rules import RuleEngine
from fraudlens.models.xgb import encode_features
from fraudlens.paths import results_dir


def load_model_versions(profile: str) -> dict[str, str]:
    """Read ``results/models_<profile>.json`` (written by ``train.py``)."""
    path = Path(results_dir()) / f"models_{profile}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/train.py --profile {profile} first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def raw_val_scores(df: pd.DataFrame, split: Split, cfg: Config) -> dict[str, np.ndarray]:
    """Validation-window scores for ``rules``, ``logreg`` and ``xgb`` models."""
    versions = load_model_versions(cfg.simulation.profile)
    val = df.iloc[split.val_idx]
    train_legit = df.iloc[split.train_idx][df.iloc[split.train_idx]["is_fraud"] == 0]
    engine = RuleEngine(cfg.simulation.rules).fit(train_legit[FEATURE_COLUMNS])
    return {
        "rules": engine.score(val[FEATURE_COLUMNS]).to_numpy(),
        "logreg": model_io.load_model(versions["logreg"]).predict_proba(val[FEATURE_COLUMNS])[:, 1],
        "xgb": model_io.load_model(versions["xgb"]).predict_proba(encode_features(val))[:, 1],
    }
