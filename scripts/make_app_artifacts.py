"""Thin CLI: build the Streamlit app's precomputed artifacts (Part 14).

    python scripts/make_app_artifacts.py --profile dev

Writes ``artifacts/app/<profile>/`` (gitignored). ``--out-dir`` redirects the
output so the smoke test can build ``tiny`` into a temp directory.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.appdata.artifacts import build_app_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FraudLens app artifacts.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument(
        "--out-dir", default=None, help="output dir (default artifacts/app/<profile>)"
    )
    parser.add_argument("--n-scored", type=int, default=100_000)
    parser.add_argument("--n-shap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = build_app_artifacts(
        args.profile,
        out_dir=args.out_dir,
        seed=args.seed,
        n_scored=args.n_scored,
        n_shap=args.n_shap,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
