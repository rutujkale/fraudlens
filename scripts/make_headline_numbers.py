"""Thin CLI: Part 12C headline numbers from ``results/final_results.json``."""

from __future__ import annotations

import sys

from fraudlens.evaluation.headlines import (
    load_final_results,
    write_final_results_doc,
    write_headline_numbers,
)


def main() -> int:
    results = load_final_results()
    path = write_headline_numbers(results)
    doc = write_final_results_doc(results)
    sys.stdout.write(path.read_text(encoding="utf-8"))
    sys.stdout.write(f"\n--- wrote {doc} ---\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
