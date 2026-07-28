from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_sequence import (
    Round74GlobalEventWindowAccumulator,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v72.json"
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


def test_round74_v72_binds_target_free_cross_asset_window_primitive() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v72"
    source = design["source_binding"]
    assert source["event_sequence_normalized_lf_sha256"] == _git_file_sha256(
        design["implementation_git_commit"],
        source["event_sequence_path"],
    )
    assert source["event_sequence_schema_version"] == (
        "round-074-causal-event-sequence-v4"
    )
    assert source["feature_count"] == 66
    assert Round74GlobalEventWindowAccumulator.__module__.endswith(
        "impact_absorption_event_sequence"
    )

    primitive = design["implemented_cross_asset_primitive"]
    assert primitive["global_receipt_order_preserved"] is True
    assert primitive["stride_counter_is_per_endpoint_symbol"] is True
    assert primitive["future_target_or_pnl_accessed"] is False
    assert primitive["per_symbol_baseline_replaced"] is False
    assert primitive["training_candidate_selectable"] is False
    assert design["preregistered_hypothesis"][
        "promotion_before_matched_evidence_permitted"
    ] is False
    assert all(value is False for value in design["authority"].values())
