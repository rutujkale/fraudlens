"""Thin CLI: build point-in-time features for a profile (Part 05B).

Reads raw parquet tables, computes ``features_v1`` with the shared feature
module and writes ``data/features/<profile>/features_v1.parquet``. Reports the
feature count, runtime and per-column NaN rates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace

import pandas as pd

from fraudlens.config import load_config
from fraudlens.data_quality.validation import load_raw_tables
from fraudlens.features.build_features import build_features
from fraudlens.features.registry import FEATURE_COLUMNS, FEATURE_VERSION, write_data_dictionary
from fraudlens.paths import features_data_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def _nan_rates(features: pd.DataFrame) -> pd.Series:
    return features[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FraudLens point-in-time features.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument(
        "--no-write", action="store_true", help="compute only, do not write parquet"
    )
    args = parser.parse_args()

    cfg = load_config(args.profile)
    tables = load_raw_tables(args.profile)
    if "transactions" not in tables:
        logger.error(
            "no raw transactions for profile %s; run scripts/generate_data.py first", args.profile
        )
        return 2

    entities = SimpleNamespace(customers=tables["customers"], locations=tables["locations"])
    tx = tables["transactions"]
    logger.info(
        "Building %s for profile=%s over %d transactions", FEATURE_VERSION, args.profile, len(tx)
    )

    start = time.perf_counter()
    features = build_features(tx, entities, cfg)
    runtime = time.perf_counter() - start

    nan_rates = _nan_rates(features)
    logger.info(
        "Built %d features for %d rows in %.1fs", len(FEATURE_COLUMNS), len(features), runtime
    )
    logger.info("NaN rate: max=%.3f median=%.3f", float(nan_rates.max()), float(nan_rates.median()))
    logger.info("highest-NaN columns:\n%s", nan_rates.head(8).to_string())

    if not args.no_write:
        out_dir = features_data_dir(args.profile)
        parquet = out_dir / f"{FEATURE_VERSION}.parquet"
        features.to_parquet(parquet, index=False)
        nan_rates.rename("nan_rate").to_frame().to_csv(out_dir / f"{FEATURE_VERSION}_nan_rates.csv")
        stats = {
            "profile": args.profile,
            "feature_version": FEATURE_VERSION,
            "n_rows": int(len(features)),
            "n_features": len(FEATURE_COLUMNS),
            "runtime_seconds": round(runtime, 3),
            "max_nan_rate": round(float(nan_rates.max()), 6),
        }
        (out_dir / f"{FEATURE_VERSION}_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        dd = write_data_dictionary()
        logger.info("Wrote %s, NaN rates, stats and data dictionary (%s)", parquet, dd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
