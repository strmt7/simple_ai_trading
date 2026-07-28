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
    / "round-074-event-sequence-model-design-v90.json"
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_v90_binds_state_first_clock_control_then_order_flow() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v90"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert base["design_sha256"] == json.loads(
        base_path.read_text(encoding="ascii")
    )["design_sha256"]

    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    for section in ("model", "pretraining", "training", "tests"):
        assert source[section]["sha256"] == _git_blob_sha256(
            commit,
            source[section]["path"],
        )
    for historical_test in source["historical_test_hardening"]:
        assert historical_test["sha256"] == _git_blob_sha256(
            commit,
            historical_test["path"],
        )
    assert source["pretraining"]["schema_version"] == (
        "round-074-causal-next-event-pretraining-v3"
    )
    assert source["training"]["training_schema_version"] == (
        "round-074-event-training-v25"
    )
    assert source["training"]["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v24"
    )
    assert source["training"]["feature_view_schema_version"] == (
        "round-074-feature-view-v2"
    )

    layers = design["feature_layer_contract"]
    market_state = layers["market_state"]
    order_flow = layers["order_flow"]
    clock = layers["clock_and_intraday_phase"]
    assert len(market_state["feature_names"]) == market_state["feature_count"] == 33
    assert len(order_flow["feature_names"]) == order_flow["feature_count"] == 31
    assert len(clock["feature_names"]) == clock["feature_count"] == 11
    feature_sets = [
        set(market_state["feature_names"]),
        set(order_flow["feature_names"]),
        set(clock["feature_names"]),
    ]
    assert sum(len(values) for values in feature_sets) == layers["feature_count"]
    assert not feature_sets[0] & feature_sets[1]
    assert not feature_sets[0] & feature_sets[2]
    assert not feature_sets[1] & feature_sets[2]
    for layer in (market_state, order_flow, clock):
        assert layer["feature_names_sha256"] == _canonical_sha256(
            layer["feature_names"]
        )
    assert {
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    }.issubset(feature_sets[2])
    assert not {
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    } & feature_sets[0]

    selection = design["selection_contract"]
    assert selection["stage_1_architecture"]["feature_view"] == (
        "market_state_clock_neutral"
    )
    assert selection["stage_2_clock_control"]["challenger"] == (
        "market_state_with_clock"
    )
    assert selection["stage_3_order_flow"]["challenger_if_clock_rejected"] == (
        "clock_neutral"
    )
    assert selection["stage_3_order_flow"]["challenger_if_clock_promoted"] == "full"
    assert selection["stage_3_order_flow"][
        "order_flow_tested_after_available_clock_control"
    ]
    assert selection["stage_4_initialization"][
        "pretraining_winner_cannot_rewrite_feature_layer_metrics"
    ]

    urls = [source["url"] for source in design["primary_source_review"]]
    assert len(urls) == len(set(urls)) == 6
    assert all(
        source["reported_performance_transferred"] is False
        for source in design["primary_source_review"]
    )
    directml = design["verification"]["directml_preflight"]
    directml_path = REPOSITORY / directml["path"]
    assert directml["file_sha256"] == _file_sha256(directml_path)
    directml_evidence = json.loads(directml_path.read_text(encoding="ascii"))
    assert directml_evidence["implementation_git_commit"] == commit
    assert directml_evidence["source_binding"]["pretraining_sha256"] == (
        source["pretraining"]["sha256"]
    )
    assert directml_evidence["source_binding"]["training_sha256"] == (
        source["training"]["sha256"]
    )
    assert directml_evidence["probe"]["causal_encoder_backward_finite"] is True
    assert directml_evidence["authority"]["financial_edge_tested"] is False
    assert design["verification"]["connected_event_pipeline_tests_passed"] == 186
    assert design["verification"]["historical_design_tests_passed"] == 25
    assert design["evidence_boundary"][
        "state_first_layering_improves_predictive_accuracy_claim"
    ] is False
    assert design["authority"]["financial_edge_tested"] is False
