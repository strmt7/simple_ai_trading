from __future__ import annotations

import json

import pytest

from tools.run_polymarket_round27_ai_cases import (
    _parser,
    _progress,
    _selection_interval,
)


def test_round27_target_free_ai_operator_exposes_no_target_or_economic_path() -> None:
    destinations = {
        action.dest
        for action in _parser()._actions
        if action.dest != "help"
    }

    assert "feature_store" in destinations
    assert "selection_source_database" in destinations
    assert "selection_claim" in destinations
    assert all(
        fragment not in destination
        for destination in destinations
        for fragment in ("target", "outcome", "resolution", "economic_report")
    )


def test_round27_target_free_ai_operator_requires_one_selection_interval() -> None:
    interval = _selection_interval(
        {
            "partitions": [
                {
                    "role": "selection",
                    "slot_id": "stage1-b",
                    "start_ms": 1,
                    "end_ms": 2,
                }
            ]
        }
    )
    assert interval["slot_id"] == "stage1-b"

    with pytest.raises(ValueError, match="selection interval differs"):
        _selection_interval({"partitions": []})


def test_round27_target_free_ai_progress_is_bounded_and_machine_readable(
    capsys,
) -> None:
    progress = _progress("model-id")
    progress(1, 7)
    progress(5, 7)
    progress(7, 7)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["completed_cases"] for line in lines] == [5, 7]
    assert all(json.loads(line)["targets_accessed"] is False for line in lines)
