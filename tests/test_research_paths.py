from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading.research_paths import (
    HISTORICAL_ROOT_ENV,
    historical_research_path,
)


def test_default_follows_current_repository_and_preserves_legacy_suffix(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(HISTORICAL_ROOT_ENV, raising=False)
    assert historical_research_path(tmp_path, r"evidence\round60\report.json") == (
        tmp_path / "data/historical-research/evidence/round60/report.json"
    )
    assert historical_research_path(tmp_path, Path("round12/report.json")) == (
        tmp_path / "data/historical-research/round12/report.json"
    )


def test_explicit_storage_override_is_used_without_touching_files(
    tmp_path, monkeypatch
):
    selected = tmp_path / "ssd-data"
    monkeypatch.setenv(HISTORICAL_ROOT_ENV, str(selected))
    assert (
        historical_research_path(tmp_path / "repo", "report.json")
        == selected / "report.json"
    )
    assert not selected.exists()


@pytest.mark.parametrize("override", ["", " ", "relative-data"])
def test_ambiguous_root_override_rejects(tmp_path, monkeypatch, override):
    monkeypatch.setenv(HISTORICAL_ROOT_ENV, override)
    with pytest.raises(ValueError, match="override"):
        historical_research_path(tmp_path, "report.json")


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "/outside",
        r"\outside",
        "../outside",
        r"a\..\outside",
        r"\\fixture.invalid\share\file",
        "a/stream:alternate",
    ],
)
def test_absolute_traversal_and_stream_paths_cannot_escape(
    tmp_path, monkeypatch, relative
):
    monkeypatch.delenv(HISTORICAL_ROOT_ENV, raising=False)
    with pytest.raises(ValueError, match="relative path"):
        historical_research_path(tmp_path, relative)


def test_host_absolute_path_is_not_a_relative_input(tmp_path, monkeypatch):
    monkeypatch.delenv(HISTORICAL_ROOT_ENV, raising=False)
    with pytest.raises(ValueError, match="relative path"):
        historical_research_path(tmp_path, tmp_path / "outside")


def test_existing_directory_link_cannot_escape_historical_storage(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(HISTORICAL_ROOT_ENV, raising=False)
    root = tmp_path / "data/historical-research"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    try:
        (root / "linked-input").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        historical_research_path(tmp_path, "linked-input/report.json")
