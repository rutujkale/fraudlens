"""Automated data-quality validation for generated raw tables (Part 03).

Every check returns pass/fail, the number of offending rows, and a few
examples.  ``validate_dataset`` is pure (takes already-loaded frames) so tests
can build tiny frames and inject defects; ``load_raw_tables`` is the thin
convenience wrapper used by the CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from fraudlens.data_gen.archetypes import (
    ARCHETYPES,
    CATEGORIES,
    CHANNELS,
    INCOME_BANDS,
    TRANSACTION_TYPES,
)
from fraudlens.data_gen.generate import FORBIDDEN_COLUMNS
from fraudlens.paths import raw_data_dir

RAW_TABLES = (
    "transactions",
    "labels",
    "customers",
    "accounts",
    "devices",
    "customer_devices",
    "merchants",
    "locations",
)

# Columns that must be present and non-null per table. Columns that are
# legitimately null (device_id for non-APP, merchant_id for P2P, ...) are omitted.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "transactions": [
        "transaction_id",
        "customer_id",
        "account_id",
        "timestamp",
        "amount",
        "currency",
        "location_id",
        "transaction_type",
        "channel",
        "created_at",
    ],
    "labels": ["transaction_id", "is_fraud", "label_available_at"],
    "customers": [
        "customer_id",
        "age",
        "income_band",
        "home_location_id",
        "account_open_date",
        "archetype",
        "baseline_amount",
        "preferred_categories",
        "active_hour_mean",
    ],
    "accounts": ["account_id", "customer_id", "account_type", "opening_date", "status"],
    "devices": ["device_id", "first_seen", "device_type"],
    "customer_devices": ["customer_id", "device_id"],
    "merchants": ["merchant_id", "merchant_category", "merchant_risk_group"],
    "locations": ["location_id", "latitude", "longitude"],
}

_ENUM_CHECKS: dict[str, tuple[str, str, list[str]]] = {
    # (table, column, allowed values) — nulls allowed unless noted in the check.
    "category": ("transactions", "merchant_category", list(CATEGORIES)),
    "transaction_type": ("transactions", "transaction_type", list(TRANSACTION_TYPES)),
    "channel": ("transactions", "channel", list(CHANNELS)),
    "currency": ("transactions", "currency", []),
    "income_band": ("customers", "income_band", list(INCOME_BANDS)),
    "archetype": ("customers", "archetype", list(ARCHETYPES)),
    "merchant_risk_group": ("merchants", "merchant_risk_group", ["low", "medium", "high"]),
}

_FK_CHECKS: list[tuple[str, str, str, str, bool]] = [
    # (child table, child col, parent table, parent col, nullable)
    ("transactions", "customer_id", "customers", "customer_id", False),
    ("transactions", "account_id", "accounts", "account_id", False),
    ("transactions", "location_id", "locations", "location_id", False),
    ("transactions", "merchant_id", "merchants", "merchant_id", True),
    ("transactions", "device_id", "devices", "device_id", True),
    ("transactions", "recipient_id", "customers", "customer_id", True),
    ("accounts", "customer_id", "customers", "customer_id", False),
    ("customer_devices", "customer_id", "customers", "customer_id", False),
    ("customer_devices", "device_id", "devices", "device_id", False),
    ("labels", "transaction_id", "transactions", "transaction_id", False),
]


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    n_offending: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Collection of checks for one dataset."""

    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": len(self.failed),
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        if self.passed:
            return f"PASS: all {len(self.checks)} checks passed"
        lines = [f"FAIL: {len(self.failed)}/{len(self.checks)} checks failed"]
        for c in self.failed:
            lines.append(f"  - {c.name}: {c.n_offending} offending rows")
        return "\n".join(lines)


def load_raw_tables(profile: str) -> dict[str, pd.DataFrame]:
    """Load every raw parquet table for ``profile`` into a dict."""
    out = raw_data_dir(profile)
    tables: dict[str, pd.DataFrame] = {}
    for name in RAW_TABLES:
        path = out / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    return tables


def _examples(
    df: pd.DataFrame, mask: pd.Series, cols: list[str] | None = None
) -> list[dict[str, Any]]:
    head = df.loc[mask]
    if cols:
        head = head[[c for c in cols if c in head.columns]]
    head = head.head(3)
    return json.loads(head.to_json(orient="records", date_format="iso"))


def _check_unique(tables: dict[str, pd.DataFrame], table: str, column: str) -> CheckResult:
    name = f"unique_{table}.{column}"
    df = tables[table]
    dup = df[column].duplicated(keep=False)
    return CheckResult(
        name=name,
        passed=not dup.any(),
        n_offending=int(dup.sum()),
        examples=_examples(df, dup, [column]),
    )


def _check_required(tables: dict[str, pd.DataFrame]) -> CheckResult:
    offenders: list[dict[str, Any]] = []
    total = 0
    for table, cols in REQUIRED_COLUMNS.items():
        if table not in tables:
            continue
        df = tables[table]
        for col in cols:
            if col not in df.columns:
                offenders.append({"table": table, "column": col, "issue": "missing_column"})
                total += 1
                continue
            miss = df[col].isna()
            n = int(miss.sum())
            if n:
                total += n
                offenders.append({"table": table, "column": col, "issue": "null", "n": n})
    return CheckResult(
        name="no_null_required_columns",
        passed=total == 0,
        n_offending=total,
        examples=offenders[:5],
    )


def _check_amount(
    tables: dict[str, pd.DataFrame], max_amount: float
) -> tuple[CheckResult, CheckResult]:
    df = tables["transactions"]
    amount = pd.to_numeric(df["amount"], errors="coerce")
    non_positive = amount <= 0
    too_large = amount > max_amount
    return (
        CheckResult(
            name="amount_positive",
            passed=not non_positive.any(),
            n_offending=int(non_positive.sum()),
            examples=_examples(df, non_positive, ["transaction_id", "amount"]),
        ),
        CheckResult(
            name="amount_not_absurd",
            passed=not too_large.any(),
            n_offending=int(too_large.sum()),
            examples=_examples(df, too_large, ["transaction_id", "amount"]),
        ),
    )


def _check_timestamps(
    tables: dict[str, pd.DataFrame],
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
) -> list[CheckResult]:
    df = tables["transactions"]
    results: list[CheckResult] = []
    raw = df["timestamp"]
    parsed = pd.to_datetime(raw, errors="coerce")
    unparsable = parsed.isna()
    results.append(
        CheckResult(
            name="transactions.timestamp_parsable",
            passed=not unparsable.any(),
            n_offending=int(unparsable.sum()),
            examples=_examples(df, unparsable, ["transaction_id", "timestamp"]),
        )
    )
    if window_start is not None and window_end is not None:
        out = (parsed < window_start) | (parsed > window_end)
        out = out & parsed.notna()
        results.append(
            CheckResult(
                name="transactions.timestamp_in_window",
                passed=not out.any(),
                n_offending=int(out.sum()),
                examples=_examples(df, out, ["transaction_id", "timestamp"]),
            )
        )
    return results


def _check_enums(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, (table, column, allowed) in _ENUM_CHECKS.items():
        if table not in tables:
            continue
        df = tables[table]
        if column not in df.columns:
            results.append(CheckResult(name=f"valid_{name}", passed=False, n_offending=1,
                                       examples=[{"missing_column": column}]))
            continue
        if not allowed:
            continue
        bad = df[column].notna() & ~df[column].isin(allowed)
        results.append(
            CheckResult(
                name=f"valid_{name}",
                passed=not bad.any(),
                n_offending=int(bad.sum()),
                examples=_examples(df, bad, [column]),
            )
        )
    return results


def _check_foreign_keys(tables: dict[str, pd.DataFrame]) -> CheckResult:
    offenders: list[dict[str, Any]] = []
    total = 0
    for child, child_col, parent, parent_col, nullable in _FK_CHECKS:
        if child not in tables or parent not in tables:
            continue
        cdf, pdf = tables[child], tables[parent]
        if child_col not in cdf.columns or parent_col not in pdf.columns:
            continue
        keys = set(pdf[parent_col].dropna())
        values = cdf[child_col]
        orphan = ~values.isin(keys)
        if nullable:
            orphan &= values.notna()
        n = int(orphan.sum())
        if n:
            total += n
            offenders.append(
                {"child": f"{child}.{child_col}", "parent": f"{parent}.{parent_col}", "n": n}
            )
    return CheckResult(
        name="no_orphan_foreign_keys",
        passed=total == 0,
        n_offending=total,
        examples=offenders[:5],
    )


def _check_transactions_after_account_open(tables: dict[str, pd.DataFrame]) -> CheckResult:
    """No transaction may precede its customer's account opening (Part 02b).

    Violations make ``history_days`` negative and hand the model a clock that runs
    backwards; the simulator now gates on this, and this check makes a regression
    fail the data-quality run instead of silently reaching the features.
    """
    tx = tables["transactions"][["transaction_id", "customer_id", "timestamp"]].copy()
    cust = tables["customers"][["customer_id", "account_open_date"]]
    merged = tx.merge(cust, on="customer_id", how="left")
    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    opened = pd.to_datetime(merged["account_open_date"], errors="coerce")
    bad = (ts < opened) & ts.notna() & opened.notna()
    return CheckResult(
        name="transactions_after_account_open",
        passed=not bad.any(),
        n_offending=int(bad.sum()),
        examples=_examples(
            merged, bad, ["transaction_id", "customer_id", "timestamp", "account_open_date"]
        ),
    )


def _check_labels(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results: list[CheckResult] = []
    tx_ids = set(tables["transactions"]["transaction_id"])
    labels = tables["labels"]
    label_ids = labels["transaction_id"]
    missing = tx_ids - set(label_ids)
    extra = set(label_ids) - tx_ids
    cover_ok = not missing and not extra
    results.append(
        CheckResult(
            name="labels_cover_transactions_exactly_once",
            passed=cover_ok,
            n_offending=len(missing) + len(extra),
            examples=[{"missing": len(missing), "extra": len(extra)}],
        )
    )
    leaked = [c for c in FORBIDDEN_COLUMNS if c in tables["transactions"].columns]
    results.append(
        CheckResult(
            name="no_label_columns_in_transactions",
            passed=not leaked,
            n_offending=len(leaked),
            examples=[{"leaked_columns": leaked}],
        )
    )
    merged = labels.merge(
        tables["transactions"][["transaction_id", "timestamp"]], on="transaction_id", how="inner"
    )
    avail = pd.to_datetime(merged["label_available_at"], errors="coerce")
    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    bad = avail < ts
    results.append(
        CheckResult(
            name="label_available_at_ge_timestamp",
            passed=not bad.any(),
            n_offending=int(bad.sum()),
            examples=_examples(merged, bad, ["transaction_id", "timestamp", "label_available_at"]),
        )
    )
    return results


def _check_fraud_rate(tables: dict[str, pd.DataFrame], max_fraud_rate: float) -> CheckResult:
    labels = tables["labels"]
    rate = float(labels["is_fraud"].astype(bool).mean()) if len(labels) else 0.0
    ok = 0.0 < rate < max_fraud_rate
    return CheckResult(
        name="fraud_rate_sanity",
        passed=ok,
        n_offending=0 if ok else 1,
        examples=[{"fraud_rate": rate, "max_fraud_rate": max_fraud_rate}],
    )


def validate_dataset(
    tables: dict[str, pd.DataFrame],
    *,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    max_amount: float = 1.0e7,
    max_fraud_rate: float = 0.05,
) -> ValidationReport:
    """Run every named check over the raw tables and return a report."""
    checks: list[CheckResult] = []
    checks.append(_check_required(tables))
    for table, column in (
        ("transactions", "transaction_id"),
        ("customers", "customer_id"),
        ("accounts", "account_id"),
        ("devices", "device_id"),
        ("merchants", "merchant_id"),
        ("locations", "location_id"),
        ("labels", "transaction_id"),
    ):
        if table in tables:
            checks.append(_check_unique(tables, table, column))
    if "transactions" in tables:
        pos, absurd = _check_amount(tables, max_amount)
        checks.extend([pos, absurd])
        checks.extend(_check_timestamps(tables, window_start, window_end))
        checks.extend(_check_enums(tables))
    if "labels" in tables and "transactions" in tables:
        checks.extend(_check_labels(tables))
        checks.append(_check_fraud_rate(tables, max_fraud_rate))
    if "transactions" in tables and "customers" in tables:
        checks.append(_check_transactions_after_account_open(tables))
    checks.append(_check_foreign_keys(tables))
    return ValidationReport(checks=checks)
