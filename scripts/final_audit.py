"""Thin CLI: Part 15F final audit. Prints PASS/FAIL per check; exit 1 on any FAIL."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from fraudlens.evaluation import freeze as freeze_mod
from fraudlens.evaluation import locked
from fraudlens.evaluation.headlines import (
    headline_markdown,
    load_final_results,
)
from fraudlens.features.registry import FEATURE_COLUMNS, FORBIDDEN_COLUMNS
from fraudlens.paths import REPO_ROOT, results_dir
from fraudlens.reporting.render import PLACEHOLDER, TEMPLATES, resume_bullet_problems

RENDERED_DOCS = [out for _t, out in TEMPLATES.values()]

# Documentation claims that must never appear (headline-honesty guardrails). The
# scan covers README.md and every tracked *.md under docs/.
FORBIDDEN_PHRASES = [
    (re.compile(r"parity verified on postgres", re.I), "parity verified on postgres"),
    (
        re.compile(r"(percent|percentage)\s+(CI|confidence interval)", re.I),
        "percent CI wording",
    ),
    (re.compile(r"\bCI\s*[:(]?\s*\d+\.?\d*\s*%"), "percent value inside a CI"),
    (re.compile(r"\d+\.?\d*\s*%\s*(?:to|[-–])\s*\d+\.?\d*\s*%"), "percent range as a CI"),
    (
        re.compile(r"\bsecondary\b[^\n]{0,30}(?:gap|dev[- ]val|PR-AUC|0\.819)", re.I),
        "secondary IEEE gap",
    ),
    (re.compile(r"`main`[^\n]{0,60}\d|\d[^\n]{0,20}`main`"), "main quoted with numbers"),
]

# DoD items that are honestly not met on this build (never silently skipped).
KNOWN_UNMET = ("main profile", "screenshots", "ci green")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        suffix = f" — {detail}" if detail else ""
        sys.stdout.write(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}\n")
        return ok


def check_lint(a: Audit) -> None:
    proc = _run([sys.executable, "-m", "ruff", "check", "."])
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
    a.check("ruff clean", proc.returncode == 0, last)


def check_tests(a: Audit) -> None:
    proc = _run([sys.executable, "-m", "pytest", "-q", "-m", "not db and not slow"])
    tail = [line for line in proc.stdout.splitlines() if "passed" in line or "failed" in line]
    detail = tail[-1] if tail else "see output"
    a.check("tests pass (non-db, non-slow)", proc.returncode == 0, detail)


def check_freeze(a: Audit) -> None:
    freeze_path = results_dir() / "freeze.json"
    try:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        locked.verify_artifact_hashes(frozen)
        n_artifacts = sum(len(info["files"]) for info in frozen["models"].values())
        hashes = freeze_mod.artifact_hashes(frozen["models"]["xgb"]["model_version"])
        ok = bool(hashes)
        detail = (
            f"{n_artifacts} model files + {len(frozen['data_files'])} data files matched; "
            f"xgb artifact has {len(hashes)} hashed files"
        )
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    a.check("freeze.json hash verification", ok, detail)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_test_logs(a: Audit) -> None:
    synth = _read_csv(results_dir() / "test_access_log.csv")
    final_rows = [r for r in synth if str(r.get("final")).lower() == "true"]
    composition = [r for r in synth if str(r.get("final")).lower() == "false"]
    ok = len(final_rows) == 1 and len(composition) == 1
    a.check(
        "synthetic test log: exactly one final eval + one composition row",
        ok,
        f"final={len(final_rows)} composition={len(composition)}",
    )
    ext = _read_csv(results_dir() / "test_access_log_external.csv")
    ok_ext = len(ext) == 2 and all(r.get("reason") for r in ext)
    a.check(
        "external log: two runs, both with reasons",
        ok_ext,
        f"rows={len(ext)}",
    )


def check_readme_placeholders(a: Audit) -> None:
    unresolved = []
    for path in RENDERED_DOCS:
        if not path.exists():
            unresolved.append(f"{path.name}: MISSING")
            continue
        text = path.read_text(encoding="utf-8")
        found = PLACEHOLDER.findall(text)
        if found:
            unresolved.append(f"{path.name}: {found[:3]}")
    a.check(
        "rendered docs have no unresolved placeholders",
        not unresolved,
        "; ".join(unresolved) or f"{len(RENDERED_DOCS)} docs clean",
    )


def check_headlines_traceable(a: Audit) -> None:
    committed = (REPO_ROOT / "reports" / "headline_numbers.md").read_text(encoding="utf-8")
    fresh = headline_markdown(load_final_results())
    # The only lines permitted to differ are the pytest collection counts.
    norm = lambda t: [ln for ln in t.splitlines() if "automated tests" not in ln]  # noqa: E731
    a.check(
        "headline_numbers.md matches results (test-count lines excluded)",
        norm(committed) == norm(fresh),
    )


def check_resume_bullets(a: Audit) -> None:
    path = REPO_ROOT / "docs" / "interview_notes.md"
    if not path.exists():
        a.check("resume bullets keep their qualifiers", False, "docs/interview_notes.md missing")
        return
    problems = resume_bullet_problems(path.read_text(encoding="utf-8"))
    a.check(
        "resume bullets keep their qualifiers",
        not problems,
        "; ".join(problems) or "all bullets carry dev (not main) + latency conditions",
    )


def check_no_tracked_data(a: Audit) -> None:
    tracked = _run(["git", "ls-files"]).stdout.splitlines()
    offenders = [f for f in tracked if f == ".env" or f.startswith("data/")]
    a.check("no .env or data/ files tracked by git", not offenders, ", ".join(offenders))


def check_ci_credentials(a: Audit) -> None:
    env_path = REPO_ROOT / ".env"
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    local_pw = ""
    if env_path.exists():
        match = re.search(r"POSTGRES_PASSWORD=(\S+)", env_path.read_text(encoding="utf-8"))
        local_pw = match.group(1) if match else ""
    ci_text = ci_path.read_text(encoding="utf-8") if ci_path.exists() else ""
    ci_pw_match = re.search(r"POSTGRES_PASSWORD:\s*(\S+)", ci_text)
    ci_pw = ci_pw_match.group(1) if ci_pw_match else ""
    differs = not local_pw or ci_pw != local_pw
    not_reused = not local_pw or local_pw not in ci_text
    ok = bool(ci_pw) and differs and not_reused
    a.check(
        "CI service password differs from local .env and is not the local password",
        ok,
        f"ci={'set' if ci_pw else 'missing'} local={'set' if local_pw else 'absent'}",
    )


def check_feature_guard(a: Audit) -> None:
    leaked = set(FEATURE_COLUMNS) & set(FORBIDDEN_COLUMNS)
    a.check("no label columns in feature columns", not leaked, ", ".join(sorted(leaked)))


def check_leakage_parity_tests(a: Audit) -> None:
    text = (REPO_ROOT / "tests" / "test_leakage.py").read_text(encoding="utf-8")
    api = (REPO_ROOT / "tests" / "test_api.py").read_text(encoding="utf-8")
    has_canary = "canary" in text.lower() or "test_8" in text
    has_parity = "test_parity_matches_offline_pipeline" in api
    a.check("leakage suite (incl. canary) present", has_canary)
    a.check("training/serving parity test present", has_parity)


def _doc_files() -> list[Path]:
    """README.md plus every tracked docs/**/*.md (local-only build logs excluded)."""
    tracked = _run(["git", "ls-files", "README.md", "docs"]).stdout.splitlines()
    files = [REPO_ROOT / p for p in tracked if p.endswith(".md")]
    return [f for f in files if f.exists()]


def check_forbidden_phrases(a: Audit) -> None:
    docs = _doc_files()
    hits: list[str] = []
    for path in docs:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, label in FORBIDDEN_PHRASES:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{i} [{label}]")
    detail = (
        "; ".join(hits[:5]) if hits else f"{len(docs)} tracked docs scanned"
    )
    a.check("no forbidden doc phrases", not hits, detail)


def check_sim_to_real(a: Audit) -> None:
    path = results_dir() / "sim_to_real.json"
    if not path.exists():
        a.check("sim_to_real.json is primary-gap-only", False, "results/sim_to_real.json missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    primary = "primary_gap_pr_auc_r1_minus_r2_on_ieee_test"
    bad_keys = [
        k for k in data if re.search(r"secondary|dev_val|synthetic", k, re.I)
    ]
    ok = primary in data and not bad_keys
    a.check(
        "sim_to_real.json is primary-gap-only (no differenced lift, no cross-dataset raw PR-AUC)",
        ok,
        f"keys={sorted(data)}" if not ok else f"primary={data[primary]:.4f}",
    )


def check_dod(a: Audit) -> None:
    path = REPO_ROOT / "docs" / "definition_of_done.md"
    if not path.exists():
        a.check("Definition-of-Done checklist present", False, "docs/definition_of_done.md missing")
        return
    text = path.read_text(encoding="utf-8")
    unchecked = re.findall(r"- \[ \] (.+)", text)
    checked = re.findall(r"- \[x\] (.+)", text, flags=re.IGNORECASE)
    unexpected = [
        item for item in unchecked if not any(k in item.lower() for k in KNOWN_UNMET)
    ]
    a.check(
        f"Definition-of-Done: {len(checked)} met, {len(unchecked)} not met",
        not unexpected,
        "unexpected unmet: " + "; ".join(unexpected) if unexpected else "",
    )
    for item in unchecked:
        sys.stdout.write(f"    [NOT MET] {item.strip()}\n")


def main() -> int:
    sys.stdout.write("FraudLens final audit\n=====================\n")
    a = Audit()
    check_lint(a)
    check_tests(a)
    check_freeze(a)
    check_test_logs(a)
    check_readme_placeholders(a)
    check_resume_bullets(a)
    check_headlines_traceable(a)
    check_sim_to_real(a)
    check_forbidden_phrases(a)
    check_no_tracked_data(a)
    check_ci_credentials(a)
    check_feature_guard(a)
    check_leakage_parity_tests(a)
    check_dod(a)

    failed = [name for name, ok, _ in a.results if not ok]
    sys.stdout.write("\n")
    if failed:
        sys.stdout.write(f"AUDIT FAILED ({len(failed)}): " + "; ".join(failed) + "\n")
        return 1
    sys.stdout.write(f"AUDIT PASSED ({len(a.results)} checks)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
