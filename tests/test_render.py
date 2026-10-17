"""Part 15D tests: the doc renderer fails on unresolved keys and resolves every template."""

from __future__ import annotations

import subprocess
import sys

import pytest

from fraudlens.paths import REPO_ROOT
from fraudlens.reporting.render import (
    PLACEHOLDER,
    TEMPLATES,
    UnresolvedPlaceholder,
    build_context,
    render,
    resolve,
    resume_bullet_problems,
)

# Deterministic test counts keep the suite fast; the real counts are produced by
# pytest collection when the renderer runs for real.
COUNTS = {"n_tests": 999, "n_leakage_tests": 12}


def test_unresolved_placeholder_raises():
    with pytest.raises(UnresolvedPlaceholder, match="no value"):
        render("hello {{nowhere.at.all}}", {})


def test_none_value_raises():
    with pytest.raises(UnresolvedPlaceholder, match="None"):
        render("{{k}}", {"k": None})


def test_format_spec_applied():
    assert resolve("meta.n_features:03d", {"meta.n_features": 53}) == "053"


def test_derived_keys_are_built_from_results():
    ctx = build_context(test_counts=COUNTS)
    assert ctx["headline.model.xgb_ci"].startswith("0.9")
    assert "CI" in ctx["headline.decision.reduction"]
    assert ctx["headline.drift.line"].startswith("no degradation observed")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_real_template_fully_resolves(name):
    template, _out = TEMPLATES[name]
    ctx = build_context(test_counts=COUNTS)
    rendered, used = render(template.read_text(encoding="utf-8"), ctx)
    assert used, "template has no placeholders"
    assert not PLACEHOLDER.findall(rendered), "unresolved placeholder left in output"


GOOD_BULLETS = """## Resume bullet candidates

- Model on dev (not main); frozen XGBoost PR-AUC 0.9619 vs Rules 0.2036.
- Decision on dev (not main); a point estimate with the interval in INR (CI 361,480 INR).
- Serving latency 127.031 ms on dev (not main); backend postgres, concurrency 1,
  n=2000 fraud-enriched, chosen after the matrix was measured.
- Drift on dev (not main); 79% of test frauds post-switch.
"""


def test_resume_bullets_pass_when_qualifiers_present():
    assert resume_bullet_problems(GOOD_BULLETS) == []


def test_resume_bullets_fail_when_profile_qualifier_removed():
    bad = GOOD_BULLETS.replace("Decision on dev (not main)", "Decision")
    problems = resume_bullet_problems(bad)
    assert any("dev (not main)" in p for p in problems)


def test_resume_bullets_fail_when_latency_condition_removed():
    bad = GOOD_BULLETS.replace("fraud-enriched, ", "")
    problems = resume_bullet_problems(bad)
    assert any("latency conditions" in p for p in problems)


def test_cli_exits_nonzero_on_unresolved_placeholder(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("value = {{nope}}", encoding="utf-8")
    out = tmp_path / "out.md"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_readme.py",
            "--template",
            str(template),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "unresolved placeholder" in proc.stderr
    assert not out.exists()
