from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
    Round74EventWindowAccumulator,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v71.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v71_binds_one_online_and_replay_window_path() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v71"
    source = design["source_binding"]
    assert source["event_sequence_normalized_lf_sha256"] == _git_file_sha256(
        design["implementation_git_commit"],
        source["event_sequence_path"],
    )
    assert source["event_sequence_schema_version"] == (
        ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    )
    assert Round74EventWindowAccumulator.__module__.endswith(
        "impact_absorption_event_sequence"
    )

    parity = design["implemented_parity_contract"]
    assert parity["single_incremental_window_accumulator"] is True
    assert parity["offline_iterator_delegates_to_incremental_accumulator"] is True
    assert parity["target_or_pnl_accessed"] is False
    assert parity["order_authority_added"] is False
    assert len(design["remaining_latency_boundaries"]) == 4
    assert all(value is False for value in design["authority"].values())
