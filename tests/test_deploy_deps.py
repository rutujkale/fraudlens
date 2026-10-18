"""Shipped code must declare every third-party import as a runtime dependency.

Regression guard for a real deploy failure: ``app/components/api_client.py`` imported
``httpx`` at module level, but ``httpx`` lived only in the ``dev`` optional extra of
``pyproject.toml`` and was missing from ``requirements.txt``. Local development never
noticed (the venv installed ``.[dev]``); Streamlit Community Cloud installs
``requirements.txt`` + ``-e .``, so the deployed app died with ``ModuleNotFoundError``
while importing the first page.

The test scans the shipped surfaces (``app/``, ``src/fraudlens/``, ``scripts/``) and
asserts every third-party top-level import is declared either in ``requirements.txt`` or
in ``[project].dependencies``. ``tests/`` is intentionally excluded: tests may legitimately
depend on dev-only tooling (``pytest``, and ``httpx`` via FastAPI's ``TestClient``).

The first-party set is derived from the scanned trees (``fraudlens``, ``components`` and
any local module/directory name), so a helper module added under ``scripts/`` later does
not trip the check.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_ROOTS = ("app", "src/fraudlens", "scripts")

_ALIASES = {"sklearn": "scikit-learn", "yaml": "pyyaml", "dotenv": "python-dotenv"}


def _normalise(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


def _name_from_requirement(requirement: str) -> str:
    name = requirement.split("[", 1)[0]
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", ";", " "):
        name = name.split(separator, 1)[0]
    return name.strip()


def _declared_dependencies() -> set[str]:
    declared = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        declared.add(_normalise(_name_from_requirement(line)))
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dependency in pyproject["project"].get("dependencies", []):
        declared.add(_normalise(_name_from_requirement(dependency)))
    return declared


def _first_party() -> set[str]:
    names = {"fraudlens", "components"}
    for root in SHIPPED_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            names.add(path.stem)
        for path in (REPO_ROOT / root).rglob("*"):
            if path.is_dir():
                names.add(path.name)
    return names


def _imports_with_files() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for root in SHIPPED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.level == 0 and node.module else []
                else:
                    continue
                for module in modules:
                    found.setdefault(module.split(".")[0], set()).add(
                        str(path.relative_to(REPO_ROOT))
                    )
    return found


def test_shipped_third_party_imports_are_declared_runtime_dependencies() -> None:
    declared = _declared_dependencies()
    first_party = _first_party()
    offenders = []
    for module, files in sorted(_imports_with_files().items()):
        if module in sys.stdlib_module_names or module in first_party or module.startswith("_"):
            continue
        if _normalise(_ALIASES.get(module, module)) not in declared:
            offenders.append(f"{module} (imported by {', '.join(sorted(files))})")
    assert not offenders, (
        "shipped code imports packages that are not declared runtime dependencies "
        "(requirements.txt or [project].dependencies): " + "; ".join(offenders)
    )
