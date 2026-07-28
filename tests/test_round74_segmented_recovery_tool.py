from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.build_round74_segmented_recovery_outcomes as subject


_REPOSITORY = Path(__file__).resolve().parents[1]
_PLAN = (
    _REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-segmented-event-cohort-plan-v3.json"
)


def test_recovery_tool_refuses_to_classify_slots_before_campaign_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = repository / "state"
    state.mkdir()
    output = repository / "recovery"

    result = subject.main(
        [
            "--repository",
            str(repository),
            "--plan",
            str(_PLAN),
            "--state-root",
            str(state),
            "--output",
            str(output),
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "recovery outcome differs" in payload["error"]
    assert payload["admitted_data_created"] is False
    assert payload["database_opened"] is False
    assert payload["trading_authority"] is False
