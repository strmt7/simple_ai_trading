from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.run_round74_segmented_development as subject


def test_segmented_tool_reports_dependency_failure_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    database = repository / "cohort.duckdb"
    database.write_bytes(b"database")
    assemblies = repository / "assemblies"
    assemblies.mkdir()
    sources = repository / "sources"
    sources.mkdir()
    model_output = repository / "model-output"
    qualification_output = repository / "qualification-output"
    monkeypatch.setattr(
        subject,
        "run_round74_segmented_development",
        lambda **_kwargs: (_ for _ in ()).throw(
            ImportError("model dependency unavailable")
        ),
    )

    result = subject.main(
        [
            "--repository",
            str(repository),
            "--database",
            str(database),
            "--target-assemblies",
            str(assemblies),
            "--source-artifacts",
            str(sources),
            "--model-output",
            str(model_output),
            "--qualification-output",
            str(qualification_output),
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ImportError"
    assert payload["sealed_test_accessed"] is False
    assert payload["trading_authority"] is False
    assert payload["profitability_claim"] is False
