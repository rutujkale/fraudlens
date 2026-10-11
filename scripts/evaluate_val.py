"""Thin CLI: validation comparison of rules/logreg/xgb + holdout (Part 06G).

Never touches the test window. Labels are joined here for evaluation only:
``is_fraud`` to score, ``fraud_scenario`` for the per-scenario breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import Config, load_config
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.figures import plot_pr_curves, plot_recall_at_k
from fraudlens.evaluation.reporting import (
    build_comparison,
    holdout_comparison,
    per_scenario_recall,
    to_markdown,
)
from fraudlens.evaluation.splits import customer_holdout, make_temporal_split
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.models.rules import RuleEngine
from fraudlens.models.xgb import encode_features
from fraudlens.paths import figures_dir, results_dir
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def _versions(profile: str) -> dict[str, str]:
    path = Path(results_dir()) / f"models_{profile}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/train.py --profile {profile} first")
    return json.loads(path.read_text(encoding="utf-8"))


def _rules_scores(df, split, cfg: Config) -> np.ndarray:
    train_legit = df.iloc[split.train_idx][df.iloc[split.train_idx]["is_fraud"] == 0]
    engine = RuleEngine(cfg.simulation.rules).fit(train_legit[FEATURE_COLUMNS])
    return engine.score(df.iloc[split.val_idx][FEATURE_COLUMNS]).to_numpy()


def compute_val_scores(df, split, cfg: Config) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return validation scores per model and the loaded model versions."""
    versions = _versions(cfg.simulation.profile)
    val = df.iloc[split.val_idx]
    scores = {"rules": _rules_scores(df, split, cfg)}
    for name, key in (("logreg", "logreg"), ("xgb", "xgb")):
        model = model_io.load_model(versions[key])
        x = val[FEATURE_COLUMNS] if name == "logreg" else encode_features(val)
        scores[name] = model.predict_proba(x)[:, 1]
    return scores, versions


def holdout_for_xgb(df, split, cfg: Config, versions: dict[str, str]) -> pd.DataFrame:
    """Refit XGBoost without 20% held-out customers; report seen vs unseen."""
    from xgboost import XGBClassifier

    y = df["is_fraud"].to_numpy(int)
    x = encode_features(df)
    _, heldout = customer_holdout(
        df["customer_id"], fraction=0.2, seed=cfg.simulation.seed
    )
    train_customers = df.iloc[split.train_idx]["customer_id"].to_numpy()
    keep = ~np.isin(train_customers, heldout)
    train_idx = split.train_idx[keep]

    meta = model_io.load_metadata(versions["xgb"])
    params = dict(meta["hyperparameters"])
    model = XGBClassifier(
        n_estimators=cfg.simulation.models.xgb.n_estimators,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=cfg.simulation.seed,
        n_jobs=-1,
        **params,
    )
    model.fit(x.iloc[train_idx], y[train_idx])

    val_customers = df.iloc[split.val_idx]["customer_id"].to_numpy()
    score = model.predict_proba(x.iloc[split.val_idx])[:, 1]
    comparison = holdout_comparison(
        y[split.val_idx], score, val_customers, heldout, cfg.simulation.ks
    )
    logger.info(
        "holdout (xgb refit): seen PR-AUC=%.4f unseen PR-AUC=%.4f",
        comparison.loc[0, "pr_auc"],
        comparison.loc[1, "pr_auc"],
    )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate models on the validation window.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    df, labels = load_feature_dataset(args.profile)
    label_cols = labels.set_index("transaction_id")
    df["is_fraud"] = df["transaction_id"].map(label_cols["is_fraud"])
    df["fraud_scenario"] = df["transaction_id"].map(label_cols["fraud_scenario"])
    split = make_temporal_split(df, cfg)

    y_val = df.iloc[split.val_idx]["is_fraud"].to_numpy(int)
    groups_val = df.iloc[split.val_idx]["customer_id"].to_numpy()
    scores, versions = compute_val_scores(df, split, cfg)

    comparison = build_comparison(y_val, scores, groups_val, cfg)
    out_dir = Path(results_dir())
    comparison.to_csv(out_dir / f"val_comparison_{args.profile}.csv", index=False)
    (out_dir / f"val_comparison_{args.profile}.md").write_text(
        f"# Validation comparison — profile `{args.profile}`\n\n"
        f"Train window used for fitting; validation window for selection. "
        f"PR-AUC CI = cluster bootstrap over customers (n={cfg.simulation.bootstrap.n}).\n\n"
        + to_markdown(comparison)
        + "\n",
        encoding="utf-8",
    )
    logger.info("comparison written:\n%s", comparison.to_string(index=False))

    best_model = comparison.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_k = cfg.simulation.ks[2] if len(cfg.simulation.ks) > 2 else cfg.simulation.ks[-1]
    scenario_table = per_scenario_recall(
        df.iloc[split.val_idx]["fraud_scenario"], scores[best_model], best_k
    )
    scenario_table.to_csv(out_dir / f"per_scenario_recall_{args.profile}.csv", index=False)
    logger.info("per-scenario recall (best=%s, top %.3g):\n%s",
                best_model, best_k, scenario_table.to_string(index=False))

    holdout = holdout_for_xgb(df, split, cfg, versions)
    holdout.to_csv(out_dir / f"holdout_{args.profile}.csv", index=False)

    plot_pr_curves(y_val, scores, figures_dir() / f"val_pr_curves_{args.profile}.png")
    ks_grid = np.logspace(-3, -1, 15).tolist()
    plot_recall_at_k(y_val, scores, ks_grid, figures_dir() / f"val_recall_at_k_{args.profile}.png")
    logger.info("figures written to %s", figures_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
