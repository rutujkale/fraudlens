"""Thin CLI: Part 12C headline numbers from ``results/final_results.json``."""

from __future__ import annotations

import sys

from fraudlens.evaluation.headlines import load_final_results, write_headline_numbers


def main() -> int:
    results = load_final_results()
    path = write_headline_numbers(results)
    sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
