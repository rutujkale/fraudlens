"""Thin CLI: generate raw data for a profile.

Business logic lives in ``fraudlens.data_gen.generate``; this only parses args.
"""

from __future__ import annotations

import argparse

from fraudlens.data_gen.generate import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic FraudLens data.")
    parser.add_argument("--profile", default="dev", help="config profile (default: dev)")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.profile, seed=args.seed)