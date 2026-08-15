from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_round74_segmented_development as subject


def test_segmented_tool_delegates_to_installed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    observed: list[str] = []
    monkeypatch.setattr(
        subject,
        "cli_main",
        lambda argv: observed.extend(argv) or 17,
    )

    result = subject.main(
        [
            "--repository",
            str(repository),
            "--database",
            "one.duckdb",
            "--database",
            "two.duckdb",
            "--target-assemblies",
            "targets",
            "--source-artifacts",
            "sources",
            "--model-output",
            "model",
            "--qualification-output",
            "qualification",
        ]
    )

    assert result == 17
    assert observed == [
        "binance-round74-develop",
        "--repository",
        str(repository),
        "--database",
        "one.duckdb",
        "--database",
        "two.duckdb",
        "--target-assemblies",
        "targets",
        "--source-artifacts",
        "sources",
        "--model-output",
        "model",
        "--qualification-output",
        "qualification",
    ]


def test_segmented_tool_injects_repository_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        subject,
        "cli_main",
        lambda argv: observed.extend(argv) or 0,
    )

    assert subject.main(["--database", "one.duckdb"]) == 0
    assert observed[:3] == [
        "binance-round74-develop",
        "--repository",
        str(subject.REPOSITORY),
    ]
