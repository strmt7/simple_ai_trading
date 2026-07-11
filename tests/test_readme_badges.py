from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BADGES = (
    "License",
    "CI",
    "super-linter",
    "Ruff",
    "Vulture",
    "cocoindex-code",
    "andrej-karpathy-skills",
)


def test_readme_badge_block_matches_generator() -> None:
    result = subprocess.run(
        [sys.executable, "tools/update_readme_badges.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_readme_exposes_imported_tooling_badges_in_canonical_order() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED BADGES -->" in readme
    assert "<!-- END GENERATED BADGES -->" in readme
    positions = [readme.index(f"[![{badge}](") for badge in EXPECTED_BADGES]
    assert positions == sorted(positions)
    assert "actions/workflows/super-linter.yml" in readme
    assert "actions/workflows/ruff.yml" in readme
    assert "actions/workflows/vulture.yml" in readme
    assert "https://github.com/cocoindex-io/cocoindex-code" in readme
    assert "https://github.com/multica-ai/andrej-karpathy-skills" in readme
    assert "https://github.com/forrestchang/andrej-karpathy-skills" not in readme
