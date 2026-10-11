"""Thin CLI: generate EDA figures and docs for a profile (Part 03B)."""

from __future__ import annotations

import argparse

from fraudlens.data_quality.eda import run_eda
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FraudLens EDA outputs.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()
    run_eda(args.profile)
    logger.info("EDA complete for profile %s", args.profile)


if __name__ == "__main__":
    main()
