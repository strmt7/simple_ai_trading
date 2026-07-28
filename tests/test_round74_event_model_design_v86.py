from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v86.json"
)
V86_EVENT_SEQUENCE_SCHEMA_VERSION = "round-074-causal-event-sequence-v5"
V86_EVENT_FEATURE_COUNT = 75
V86_EVENT_FEATURE_NAMES_SHA256 = (
    "1487503fff95883204aa0a34a8141a126d72300f85cc0d72d005144a5f2d225e"
)
V86_EVENT_SCALER_SCHEMA_VERSION = "round-074-event-feature-scaler-v5"
V86_BINARY_FEATURE_COUNT = 11
V86_CLOCK_PERIODS_SECONDS = (60, 300, 900)
V86_MODEL_PARAMETER_COUNTS = {
    "event_pooling_linear": 22600,
    "event_pooling_mlp": 43684,
    "causal_event_tcn": 131108,
    "causal_event_attention": 154180,
}


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

    assert sequence["schema_version"] == V86_EVENT_SEQUENCE_SCHEMA_VERSION
    assert sequence["feature_count"] == V86_EVENT_FEATURE_COUNT
    assert sequence["feature_names_sha256"] == V86_EVENT_FEATURE_NAMES_SHA256
    assert scaler["schema_version"] == V86_EVENT_SCALER_SCHEMA_VERSION
    assert (
        scaler["binary_feature_count"]
        == feature_contract["binary_feature_count_after"]
        == V86_BINARY_FEATURE_COUNT
    )
    assert tuple(feature_contract["periods_seconds"]) == V86_CLOCK_PERIODS_SECONDS
    assert feature_contract["phase_resolution"] == "exchange event milliseconds"
    assert feature_contract["host_received_wall_time_used_for_clock_phase"] is False
    assert feature_contract["feature_available_only_after_event_receipt"] is True
    assert feature_contract["future_event_or_target_accessed"] is False
    assert feature_contract["missing_timestamp_policy"] == "reject_event"
    assert feature_contract["opening_indicators_are_leading_binary_features"] is True
    assert (
        feature_contract["robust_scaler_may_standardize_opening_indicators"] is False
    )

    assert delta["model_parameter_counts_after"] == V86_MODEL_PARAMETER_COUNTS
    assert delta["feature_ablation_required_before_selection"] is True
    assert delta["automatic_candidate_promotion_permitted"] is False
    assert design["evidence_boundary"]["financial_backtest_performed"] is False
    assert design["authority"]["profitability_claim"] is False
