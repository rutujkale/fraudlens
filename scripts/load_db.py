"""Thin CLI: load raw parquet data into PostgreSQL (Part 04)."""

from __future__ import annotations

import argparse
import json
import sys

from fraudlens.db.loader import load_all
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load FraudLens raw data into PostgreSQL.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--reset", action="store_true", help="drop and recreate schemas first")
    args = parser.parse_args()

    result = load_all(args.profile, reset=args.reset)
    print(json.dumps(result, indent=2, default=str))
    total = sum(row["rows"] for row in result["loads"])
    logger.info(
        "Loaded %d rows across %d tables in %.2fs (profile=%s)",
        total,
        len(result["loads"]),
        result["total_seconds"],
        args.profile,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
