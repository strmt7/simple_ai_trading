"""Small dependency invariants; no venue, account or captured data access."""

from pathlib import Path
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
