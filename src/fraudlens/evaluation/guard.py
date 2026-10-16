"""Test-set access guard (Part 06A).

The test window is evaluated exactly once, in Part 12. Every access must go
through :class:`TestSetGuard`, which appends an audit row to
``results/test_access_log.csv`` and raises unless the caller explicitly passes
``final=True``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from fraudlens.paths import results_dir
from fraudlens.utils import git_commit

LOG_COLUMNS = ["timestamp", "caller", "model_version", "final", "git_commit"]


def suspicious_pr_auc_action(
    val_pr_auc: float,
    n_val_positives: int,
    *,
    threshold: float,
    bypass_min_positives: int,
) -> str:
    """Classify a suspiciously high XGB validation PR-AUC (Part 02b).

    A PR-AUC above ``threshold`` usually means leakage and must stop the run —
    but on a small validation window a handful of positives can reach 1.0 by
    chance. When ``n_val_positives`` is below ``bypass_min_positives`` the trip
    is recorded as ``"bypassed: small-sample"`` instead of stopping; larger
    profiles keep the hard stop. Returns ``"ok"`` when not suspicious.
    """
    if val_pr_auc <= threshold:
        return "ok"
    if n_val_positives < bypass_min_positives:
        return "bypassed: small-sample"
    return "stop"


class TestSetGuard:
    """Audited gate for any function that touches the test window."""

    # Not a pytest test class (name starts with "Test").
    __test__ = False

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path else Path(results_dir()) / "test_access_log.csv"

    def access(self, caller: str, model_version: str = "", final: bool = False) -> bool:
        """Log the access attempt; raise unless ``final=True``. Returns True on success."""
        self._append(caller, model_version, final)
        if not final:
            raise PermissionError(
                f"test-set access by {caller!r} denied: pass final=True only in Part 12"
            )
        return True

    def _append(self, caller: str, model_version: str, final: bool) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "caller": caller,
            "model_version": model_version,
            "final": bool(final),
            "git_commit": git_commit(),
        }
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
