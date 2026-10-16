"""Thin CLI: freeze the Part 12 evaluation inputs into ``results/freeze.json``.

Run once on a clean tree, commit the result, then run ``final_evaluation.py``
(which verifies every hash). No model, feature or threshold may change after this.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.evaluation.freeze import write_freeze
from fraudlens.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Part 12 evaluation inputs.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--force", action="store_true", help="freeze despite a dirty tree")
    parser.add_argument("--reason", default=None, help="required with --force; recorded")
    args = parser.parse_args()

    out = write_freeze(args.profile, force=args.force, force_reason=args.reason)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
