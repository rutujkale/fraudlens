"""Thin CLI: validate raw data for a profile (Part 03A). Exit != 0 on failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fraudlens.config import load_config
from fraudlens.data_quality.validation import load_raw_tables, validate_dataset
from fraudlens.paths import reports_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FraudLens raw data.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile).simulation
    window_start = pd.Timestamp(cfg.start_date)
    window_end = window_start + pd.to_timedelta(cfg.days, unit="D")
    tables = load_raw_tables(args.profile)
    if not tables:
        logger.error("No raw tables found for profile %s; run the simulator first.", args.profile)
        return 2

    report = validate_dataset(
        tables,
        window_start=window_start,
        window_end=window_end,
        max_amount=cfg.validation.max_amount,
        max_fraud_rate=cfg.validation.max_fraud_rate,
    )
    out = Path(reports_dir()) / f"data_validation_{args.profile}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    logger.info("Validation report written to %s", out)
    logger.info("%s", report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
