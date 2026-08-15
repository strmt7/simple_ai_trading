from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "consolidate_main_only.ps1"


def test_main_only_consolidation_is_dry_run_first_and_loss_averse() -> None:
    source = TOOL.read_text(encoding="ascii")

    assert "[switch]$Apply" in source
    assert "if (-not $Apply)" in source
    assert "merge-base --is-ancestor" in source
    assert "Unique history must be consolidated before cleanup" in source
    assert "Apply requires a clean primary worktree" in source
    assert "Secondary worktree is unavailable or dirty" in source
    assert 'current branch is $currentBranch' in source
    assert "main and $Remote/main to be identical" in source


def test_main_only_consolidation_installs_no_bypass_creation_ruleset() -> None:
    source = TOOL.read_text(encoding="ascii")

    assert 'rulesetName = "main-only-branch-creation"' in source
    assert 'include = @("~ALL")' in source
    assert 'exclude = @("refs/heads/main")' in source
    assert "bypass_actors = @()" in source
    assert 'rules = @([ordered]@{ type = "creation" })' in source
    assert "Non-main branches remain after cleanup" in source
