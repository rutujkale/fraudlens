"""Thin CLI: run the FraudLens scoring API with uvicorn (Part 13)."""

from __future__ import annotations

import argparse
import sys

from fraudlens.api.main import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FraudLens scoring API.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--model-version", default=None)
    parser.add_argument(
        "--history-backend", default=None, choices=["postgres", "memory"]
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-verify-freeze", action="store_true")
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        profile=args.profile,
        model_version=args.model_version,
        history_backend=args.history_backend,
        verify_freeze=not args.no_verify_freeze,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
