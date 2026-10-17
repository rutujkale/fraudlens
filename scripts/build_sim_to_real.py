"""Thin CLI: project the primary sim-to-real gap into ``results/sim_to_real.json``.

The external runner writes the full record to ``results/external_gap.csv``. That file
also carries a *secondary* cross-dataset figure (selection-optimistic; two different
prevalences) which must **not** be presented as a result. This CLI keeps only the
primary R1 - R2 gap on the IEEE test window (both PR-AUCs on the same dataset, so
they are comparable) and drops every cross-dataset / differenced number.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from fraudlens.paths import results_dir

PRIMARY = "primary_gap_pr_auc_r1_minus_r2_on_ieee_test"
SAME_DATASET = ["r1_ieee_test_pr_auc", "r2_ieee_test_pr_auc", "ieee_test_prevalence"]

NOTE = (
    "Primary gap only: R1 - R2 PR-AUC on the IEEE-CIS test window (both numbers on the "
    "same dataset, same prevalence). Any cross-dataset (dev-val vs IEEE) or differenced "
    "lift is deliberately excluded as selection-optimistic and not comparable."
)


def build(gap_csv: str | None = None, out: str | None = None) -> dict:
    src = pd.read_csv(gap_csv or (results_dir() / "external_gap.csv")).iloc[0]
    record: dict[str, object] = {PRIMARY: float(src[PRIMARY])}
    record.update({key: float(src[key]) for key in SAME_DATASET})
    record["note"] = NOTE
    record["source"] = "results/external_gap.csv"
    path = out or (results_dir() / "sim_to_real.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    record = build(args.gap_csv, args.out)
    sys.stdout.write(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
