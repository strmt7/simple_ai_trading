"""Small dependency invariants; no venue, account or captured data access."""

from pathlib import Path
import re
import tomllib

from simple_ai_trading.polymarket_live_settlement import (
    POLYMARKET_UNIFIED_SDK_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def test_settlement_dependency_matches_the_audited_adapter() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["optional-dependencies"]["polymarket-live"]
    assert f"polymarket-client=={POLYMARKET_UNIFIED_SDK_VERSION}" in requirements


def test_ruff_dependency_matches_workflow_and_explicit_rule_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(
        item
        for item in project["dependency-groups"]["test"]
        if item.startswith("ruff==")
    )
    version = requirement.split("==", 1)[1]
    workflow = (ROOT / ".github/workflows/ruff.yml").read_text(encoding="utf-8")
    assert f"version: {version}\n" in workflow
    assert project["tool"]["ruff"]["lint"]["select"] == ["E4", "E7", "E9", "F"]
    assert project["tool"]["ruff"]["force-exclude"] is True
    assert "docs/model-research/**/raw/**" in project["tool"]["ruff"]["extend-exclude"]
    assert "data/**/raw/**" in project["tool"]["ruff"]["extend-exclude"]


def test_super_linter_excludes_both_raw_roots_not_authored_research() -> None:
    workflow = (ROOT / ".github/workflows/super-linter.yml").read_text(encoding="utf-8")
    line = next(
        line for line in workflow.splitlines() if "FILTER_REGEX_EXCLUDE:" in line
    )
    pattern = line.split(":", 1)[1].strip()
    for path in (
        "data/example/raw/source.md",
        "docs/model-research/polymarket/raw/example/source.md",
        "/runner/project/data/example/raw/source.md",
    ):
        assert re.search(pattern, path)
    for path in (
        "docs/review/2026-09-04/model-evidence-reassessment.md",
        "docs/model-research/polymarket/review.md",
        "src/simple_ai_trading/model_lab.py",
    ):
        assert not re.search(pattern, path)
