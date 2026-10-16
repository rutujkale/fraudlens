"""Part 12A — freeze the evaluation inputs before the test window is opened.

Writes ``results/freeze.json``: the git commit, config hash, feature version,
model versions, a sha256 of every artifact and data file, the split boundaries,
the cost config + capacity, the fitted rules thresholds, the evaluated systems
and the pre-registered headline success criteria. ``scripts/final_evaluation.py``
verifies every hash against this file and aborts on any mismatch.

A dirty tree is refused (``--force`` overrides but must record a reason). The
recorded ``git_commit`` is the commit *before* ``freeze.json`` was added, so the
freeze file is a faithful snapshot of the code that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from fraudlens.config import Config, config_hash, load_config
from fraudlens.evaluation.cost_model import resolve_cost_params
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.guard import LOG_COLUMNS
from fraudlens.evaluation.splits import make_temporal_split
from fraudlens.features.registry import FEATURE_COLUMNS, FEATURE_VERSION
from fraudlens.models.io import load_metadata
from fraudlens.models.rules import RuleEngine
from fraudlens.paths import REPO_ROOT, artifacts_dir, features_data_dir, raw_data_dir, results_dir
from fraudlens.utils import get_logger, git_commit

logger = get_logger(__name__)

# Systems frozen for the locked evaluation. The Part 09 isolation forest was NOT
# adopted (post cold-start fix: S1-S0 and S1s-S0 bagged CIs do not exclude 0), so
# it is absent and no IF artifact exists to hash.
SYSTEMS = ["rules", "logreg", "xgb"]
PRIMARY_COST_SCENARIO = "balanced"

PRE_REGISTERED_SUCCESS = {
    "H1_model": (
        "XGB PR-AUC CI lower bound exceeds the Rules CI upper bound. "
        "LogReg ordering is reported either way."
    ),
    "H2_decision": (
        "The amount-aware capacity policy's realized-cost reduction versus "
        "F1-optimal has a CI excluding 0."
    ),
    "H3_unseen": (
        'The post-fix Part 09 "not adopted" decision stands, with no new Part 09 claim.'
    ),
    "H4_engineering": (
        "Counts come from code, and the latency figure comes from Part 13."
    ),
}


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file, streamed so large parquet files do not load into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_status_porcelain() -> str:
    """``git status --porcelain`` output (empty string means a clean tree)."""
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout


def artifact_hashes(model_version: str) -> dict[str, str]:
    """sha256 of every file in an artifact directory (models, calibrator, policy)."""
    d = artifacts_dir(model_version)
    files = sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files in artifact {model_version!r}")
    return {p.name: sha256_file(p) for p in files}


def data_file_hashes(profile: str) -> dict[str, str]:
    """sha256 of the three inputs the pipeline reads (features, tx, labels)."""
    feats = features_data_dir(profile) / f"{FEATURE_VERSION}.parquet"
    tx = raw_data_dir(profile) / "transactions.parquet"
    labels = raw_data_dir(profile) / "labels.parquet"
    missing = [str(p) for p in (feats, tx, labels) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing data files for freeze: {missing}")
    return {
        str(feats.relative_to(REPO_ROOT)): sha256_file(feats),
        str(tx.relative_to(REPO_ROOT)): sha256_file(tx),
        str(labels.relative_to(REPO_ROOT)): sha256_file(labels),
    }


def fitted_rule_thresholds(cfg: Config, df: pd.DataFrame, split) -> dict:
    """Rule thresholds resolved from the training-legit distribution (rules are unfitted)."""
    train_legit = df.iloc[split.train_idx]
    train_legit = train_legit[train_legit["is_fraud"] == 0]
    engine = RuleEngine(cfg.simulation.rules).fit(train_legit[FEATURE_COLUMNS])
    weights = {name: float(p["weight"]) for name, p in cfg.simulation.rules.rules.items()}
    return {
        "system": "rules",
        "fitted_on": "train_legit rows only (no labels)",
        "thresholds": {k: (None if v is None else float(v)) for k, v in engine.thresholds.items()},
        "weights": weights,
    }


def _part09_note() -> dict:
    """Part 09 adoption decision (post cold-start fix), read from its own result file."""
    path = results_dir() / "exp_09_unseen_decision_unseen_lab_structuring.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    return {
        "adopted": bool(decision["adopted"]),
        "profile": decision["profile"],
        "holdout": decision["holdout"],
        "dataset_version": "post_coldstart_fix",
        "s1_minus_s0_bagged": decision["paired"]["S1"]["diff_bagged"],
        "s1s_minus_s0_bagged": decision["paired"]["S1s"]["diff_bagged"],
        "reason": decision["reason"],
        "source": str(path.relative_to(REPO_ROOT)),
        "note": (
            "Part 09 (isolation forest) was re-run post cold-start fix; the decision is "
            "unchanged (not adopted) and no isolation-forest artifact was saved, so no "
            "anomaly system is evaluated."
        ),
    }


def _log_force_reason(reason: str) -> None:
    """Append a non-evaluation row so a forced freeze is auditable in the access log."""
    import csv

    path = results_dir() / "test_access_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "caller": "freeze_force",
        "model_version": "",
        "final": False,
        "git_commit": git_commit(),
        "reason": reason,
    }
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_freeze(
    profile: str, *, force: bool = False, force_reason: str | None = None
) -> dict:
    """Assemble the freeze dict; refuses a dirty tree unless forced with a reason."""
    cfg = load_config(profile)
    chash = config_hash(cfg)

    status = git_status_porcelain()
    if status.strip() and not force:
        raise RuntimeError(f"refusing to freeze: working tree is dirty:\n{status}")
    if force and not force_reason:
        raise ValueError("--force requires --reason so the override is recorded")
    if force:
        _log_force_reason(force_reason)

    models_path = results_dir() / f"models_{profile}.json"
    versions = json.loads(models_path.read_text(encoding="utf-8"))

    models: dict[str, dict] = {}
    for name in ("logreg", "xgb"):
        version = versions[name]
        meta = load_metadata(version)
        if meta.get("profile") != profile:
            raise ValueError(
                f"artifact {version!r} metadata profile is {meta.get('profile')!r}, "
                f"expected {profile!r}; refusing to freeze a cross-profile artifact"
            )
        models[name] = {
            "model_version": version,
            "feature_version": meta.get("feature_version"),
            "calibration": meta.get("calibration"),
            "git_commit": meta.get("git_commit"),
            "config_hash_at_training": meta.get("config_hash"),
            "files": artifact_hashes(version),
        }

    df, labels = load_feature_dataset(profile)
    df["is_fraud"] = df["transaction_id"].map(labels.set_index("transaction_id")["is_fraud"])
    split = make_temporal_split(df, cfg)

    split_json_path = results_dir() / f"split_{profile}.json"
    if split_json_path.exists():
        recorded = json.loads(split_json_path.read_text(encoding="utf-8"))
        if recorded["boundaries"].get("t_val_end") != split.boundaries.get("t_val_end"):
            raise RuntimeError(
                "recomputed split boundaries differ from results/"
                f"split_{profile}.json; regenerated data or config changed"
            )

    policy = json.loads((artifacts_dir(models["xgb"]["model_version"]) / "policy.json").read_text())
    params = resolve_cost_params(
        cfg.simulation.costs, cfg.simulation.cost_scenarios[PRIMARY_COST_SCENARIO]
    )

    return {
        "part": "12A",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "git_commit": git_commit(),
        "git_commit_note": "commit before results/freeze.json was added (Commit A)",
        "tree_clean": not status.strip(),
        "forced": bool(force),
        "force_reason": force_reason if force else None,
        "dataset_version": cfg.simulation.dataset_version,
        "seed": cfg.simulation.seed,
        "feature_version": FEATURE_VERSION,
        "config_hash": chash,
        "retrain": False,
        "retrain_rationale": (
            "Part 06-07 artifacts (xgb/logreg) are used as-is; no retraining on train+val. "
            "A retrain would fork a new timestamped artifact and is out of scope."
        ),
        "systems": SYSTEMS,
        "anomaly": _part09_note(),
        "cost_config": cfg.simulation.costs.model_dump(mode="json"),
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "cost_params": params.__dict__,
        "capacity_per_day": int(policy["capacity_per_day"]),
        "avg_daily_volume": float(policy["avg_daily_volume"]),
        "capacity_source": f"{models['xgb']['model_version']}/policy.json (val_thr)",
        "frozen_thresholds": {
            "f1_threshold": policy["f1_threshold"],
            "cost_optimal_threshold": policy["cost_optimal_threshold"],
            "cost_optimal_threshold_capacity": policy["cost_optimal_threshold_capacity"],
            "selected_calibrator": policy["selected_calibrator"],
        },
        "split_boundaries": split.boundaries,
        "split_sizes": {
            "n_train": int(len(split.train_idx)),
            "n_val": int(len(split.val_idx)),
            "n_test": int(len(split.test_idx)),
        },
        "rules": fitted_rule_thresholds(cfg, df, split),
        "models": models,
        "data_files": data_file_hashes(profile),
        "pre_registered_success": PRE_REGISTERED_SUCCESS,
    }


def write_freeze(
    profile: str, *, force: bool = False, force_reason: str | None = None
) -> Path:
    """Build and write ``results/freeze.json``; return its path."""
    freeze = build_freeze(profile, force=force, force_reason=force_reason)
    out = results_dir() / "freeze.json"
    out.write_text(json.dumps(freeze, indent=2, default=str), encoding="utf-8")
    return out
