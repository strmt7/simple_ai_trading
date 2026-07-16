from __future__ import annotations

from pathlib import Path

from tools.release_metadata import resolve_metadata, validate_beta_version


REPO = Path(__file__).resolve().parents[1]


def test_release_metadata_converts_pep440_beta_to_semver(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "simple-ai-trading"\nversion = "0.1.0b1"\n',
        encoding="utf-8",
    )

    metadata = resolve_metadata(requested_version="", repo_root=tmp_path, existing_tags=set())

    assert metadata.release_version == "0.1.0-beta.1"
    assert metadata.release_tag == "v0.1.0-beta.1"
    assert metadata.package_base == "SimpleAITrading-0.1.0-beta.1-win64-beta"


def test_release_metadata_rejects_non_beta_versions() -> None:
    try:
        validate_beta_version("0.1.0")
    except ValueError as exc:
        assert "only publishes beta" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("stable release unexpectedly accepted")


def test_release_metadata_rejects_existing_tag_without_override(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "simple-ai-trading"\nversion = "0.1.0b1"\n',
        encoding="utf-8",
    )

    try:
        resolve_metadata(
            requested_version="",
            repo_root=tmp_path,
            existing_tags={"v0.1.0-beta.1"},
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("existing release tag unexpectedly accepted")


def test_beta_release_workflow_is_manual_prerelease_with_replacement_guard() -> None:
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    action = (REPO / ".github" / "actions" / "windows-beta-release" / "action.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "Hosted Windows beta release validation" in workflow
    assert "replace_existing" in workflow
    assert "--prerelease" in action
    assert "replace_existing=true requires replacement_acknowledgement exactly" in action
    assert "tools\\build_native_windows.ps1" in action
    assert "tools\\smoke_native_windows_ui.ps1" in action
    assert "tools\\validate_native_windows_layout.ps1" in action
    assert "python -m coverage report" in action
    assert "coverage report --fail-under" not in action
    assert "SimpleAITrading.exe" in action
