"""Thin CLI: Part 12B locked test evaluation (frozen artifacts only).

``--dry-run`` exercises the identical evaluation + reporting path on the
``val_thr`` slice (no guard, no final files) so crashes surface before the single
guarded test access. The real run verifies every frozen hash, opens the test
window once via ``TestSetGuard(final=True)`` and writes ``results/final_*``.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.evaluation.final_evaluation import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Locked Part 12B test evaluation.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--dry-run", action="store_true", help="run on val_thr, no guard")
    parser.add_argument("--allow-rerun", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.allow_rerun and not args.reason:
        parser.error("--allow-rerun requires --reason")
    evaluate(args.profile, final=not args.dry_run, allow_rerun=args.allow_rerun,
             reason=args.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
