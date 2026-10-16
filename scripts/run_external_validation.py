"""Thin CLI: IEEE-CIS external validation (Part 11).

If the Kaggle files are missing it prints ``skipped: data missing`` plus download
instructions and exits 0 (``--strict`` -> 2), so the rest of the project keeps
working. Otherwise it delegates R1–R5 to :mod:`fraudlens.external.runner`.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import load_config
from fraudlens.external import ieee_cis
from fraudlens.external.runner import run
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS external validation (Part 11).")
    parser.add_argument("--profile", default="dev", help="synthetic profile for feature config")
    parser.add_argument("--strict", action="store_true", help="exit 2 when the data is missing")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    tx_path, id_path = ieee_cis.ieee_cis_paths(cfg)
    missing = [p.name for p in (tx_path, id_path) if not p.exists()]
    if missing:
        print("skipped: data missing")
        logger.warning(ieee_cis.missing_message(tx_path, missing))
        return 2 if args.strict else 0

    run(cfg, ieee_cis.prepare_dataset(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
