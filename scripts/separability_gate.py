"""Thin CLI: run the separability gate for a profile (Part 03C)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fraudlens.config import load_config
from fraudlens.data_quality.gate import gate_markdown, run_separability_gate, write_gate_report
from fraudlens.data_quality.signals import compute_crude_signals
from fraudlens.data_quality.validation import load_raw_tables
from fraudlens.paths import docs_dir, results_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FraudLens separability gate.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile).simulation
    tables = load_raw_tables(args.profile)
    signals = compute_crude_signals(
        tables["transactions"], tables["customers"], tables["locations"], tables["customer_devices"]
    )
    report = run_separability_gate(
        signals,
        tables["labels"],
        gate_cfg=cfg.separability_gate,
        train_end_fraction=cfg.splits.train_end_fraction,
        seed=cfg.seed,
        profile=args.profile,
    )
    out = Path(results_dir()) / f"separability_gate_{args.profile}.json"
    write_gate_report(report, str(out))
    md_path = Path(docs_dir()) / "separability_gate.md"
    md_path.write_text(gate_markdown(report), encoding="utf-8")

    logger.info("Gate report written to %s and %s", out, md_path)
    for name, cond in report.conditions.items():
        logger.info("  %-24s observed=%.4f threshold=%.4f %s",
                    name, float(cond["observed"]), float(cond["threshold"]),
                    "PASS" if cond["passed"] else "FAIL")
    logger.info("best single-signal ROC-AUC=%.4f PR-AUC=%.4f", report.best_single_signal_roc_auc,
                report.best_single_signal_pr_auc)
    logger.info("tree(d=1) PR-AUC=%.4f  tree(d=3) PR-AUC=%.4f  boosted PR-AUC=%.4f (gap=%.4f)",
                report.tree_stump_pr_auc, report.tree_depth3_pr_auc, report.boosted_pr_auc,
                report.tree3_vs_boosted_gap)
    logger.info("overlap signals passed=%d/%d", report.overlap_signals_passed, 5)
    if report.dominant_feature:
        logger.info("NOTE dominant feature: %s (gain share %.2f)",
                    report.dominant_feature["feature"], report.dominant_feature["gain_share"])

    overall = "PASS" if report.passed else "FAIL"
    logger.info("SEPARABILITY GATE: %s", overall)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
