"""Thin CLI: render README, model card and interview notes from results (Part 15D).

Exits non-zero (and writes nothing) if any ``{{key}}`` in a template has no value.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.reporting.render import (
    UnresolvedPlaceholder,
    build_context,
    render,
    render_templates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render FraudLens docs from results.")
    parser.add_argument("--template", default=None, help="render one template file instead")
    parser.add_argument("--out", default=None, help="output path for --template")
    args = parser.parse_args()

    try:
        if args.template:
            if not args.out:
                parser.error("--out is required with --template")
            from pathlib import Path

            rendered, _ = render(
                Path(args.template).read_text(encoding="utf-8"), build_context()
            )
            Path(args.out).write_text(rendered, encoding="utf-8")
            sys.stdout.write(f"wrote {args.out}\n")
            return 0
        written = render_templates()
    except UnresolvedPlaceholder as exc:
        sys.stderr.write(f"unresolved placeholder: {exc}\n")
        return 2
    for name, path in written.items():
        sys.stdout.write(f"{name}: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
