from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.run_round74_event_development as subject


def test_main_reports_dependency_failure_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    database = repository / "microstructure.duckdb"
    database.write_bytes(b"database")
    assemblies = repository / "assemblies"
    assemblies.mkdir()
    output = repository / "output"
    monkeypatch.setattr(
        subject,
        "run_round74_event_development",
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
            "--output",
            str(output),
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ImportError"
    assert payload["trading_authority"] is False
    assert payload["profitability_claim"] is False
