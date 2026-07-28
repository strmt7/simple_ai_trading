from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    build_round74_event_model,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_CLOCK_PERIODS_SECONDS,
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
    ROUND74_EVENT_SCALER_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v86.json"
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
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_v86_binds_causal_exchange_clock_features() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    source = design["source_binding"]
    feature_contract = design["exchange_clock_feature_contract"]
    delta = design["model_contract_delta"]

    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v86"
    commit = design["implementation_git_commit"]
    sequence = source["event_sequence"]
    assert sequence["sha256"] == _git_file_sha256(commit, sequence["path"])
    scaler = source["event_scaler"]
    assert scaler["sha256"] == _git_file_sha256(commit, scaler["path"])
    for test in source["tests"]:
        assert test["sha256"] == _git_file_sha256(commit, test["path"])

    assert sequence["schema_version"] == ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    assert sequence["feature_count"] == len(ROUND74_EVENT_FEATURE_NAMES) == 75
    assert sequence["feature_names_sha256"] == ROUND74_EVENT_FEATURE_NAMES_SHA256
    assert scaler["schema_version"] == ROUND74_EVENT_SCALER_SCHEMA_VERSION
    assert (
        scaler["binary_feature_count"]
        == feature_contract["binary_feature_count_after"]
        == ROUND74_EVENT_BINARY_FEATURE_COUNT
        == 11
    )
    assert tuple(feature_contract["periods_seconds"]) == (
        ROUND74_EVENT_CLOCK_PERIODS_SECONDS
    )
    assert feature_contract["phase_resolution"] == "exchange event milliseconds"
    assert feature_contract["host_received_wall_time_used_for_clock_phase"] is False
    assert feature_contract["feature_available_only_after_event_receipt"] is True
    assert feature_contract["future_event_or_target_accessed"] is False
    assert feature_contract["missing_timestamp_policy"] == "reject_event"
    assert feature_contract["opening_indicators_are_leading_binary_features"] is True
    assert (
        feature_contract["robust_scaler_may_standardize_opening_indicators"] is False
    )

    parameter_counts = {
        candidate_id: sum(
            parameter.numel()
            for parameter in build_round74_event_model(candidate_id).parameters()
        )
        for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES
    }
    assert delta["model_parameter_counts_after"] == parameter_counts
    assert delta["feature_ablation_required_before_selection"] is True
    assert delta["automatic_candidate_promotion_permitted"] is False
    assert design["evidence_boundary"]["financial_backtest_performed"] is False
    assert design["authority"]["profitability_claim"] is False
