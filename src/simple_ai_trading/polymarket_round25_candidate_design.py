"""Strict loader for the target-blind Round 25 candidate-selection design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    POLYMARKET_ROUND25_TWAP_FEATURE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_CANDIDATE_DESIGN_SCHEMA_VERSION = (
    "polymarket-round25-twap-native-candidate-selection-design-v1"
)
POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256 = (
    "a9ae152730cc4d931f3d5846f339143e67dbeca17449ca28950d0a8d3d129494"
)
POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SCHEMA_VERSION = (
    "polymarket-round25-twap-native-candidate-selection-amendment-v2"
)
POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256 = (
    "9bd48b5ca12d717bd14ef8c00754ff4166c75a854fa78ee3abf2fd46af2384e2"
)
POLYMARKET_ROUND25_CANDIDATE_IDS = (
    "market-prior-v1",
    "phase-isotonic-market-prior-v1",
    "l2-logistic-residual-v1",
    "lightgbm-residual-depth3-v1",
    "lightgbm-residual-depth5-v1",
    "causal-multitask-tcn-residual-v1",
)
POLYMARKET_ROUND25_AI_CANDIDATE_IDS = (
    "qwen3.5-9b-risk-veto-v1",
    "fin-r1-8b-risk-veto-v1",
)
_MAXIMUM_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 25 candidate design contains duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 candidate design contains {value}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _names_sha256(names: tuple[str, ...]) -> str:
    return _canonical_sha256(list(names))


def validate_round25_candidate_design(
    value: Mapping[str, object],
) -> dict[str, object]:
    design = dict(value)
    claimed = design.pop("design_sha256", None)
    parents = design.get("parents")
    features = design.get("feature_contracts")
    population = design.get("population_and_splits")
    candidates = design.get("finite_candidate_ledger")
    selection = design.get("candidate_selection")
    ai = design.get("ai_candidate_audit")
    execution = design.get("execution_gate")
    truth = design.get("truth_state")
    if (
        claimed != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
        or claimed != _canonical_sha256(design)
        or design.get("schema_version")
        != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SCHEMA_VERSION
        or design.get("round") != 25
        or design.get("status")
        != "frozen_target_and_outcome_blind_before_round25_v2_capture_start"
        or not isinstance(parents, Mapping)
        or parents.get("twap_native_model_design_sha256")
        != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
        or not isinstance(features, Mapping)
        or features.get("twap_schema_version")
        != POLYMARKET_ROUND25_TWAP_FEATURE_SCHEMA_VERSION
        or features.get("twap_width") != len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)
        or features.get("twap_names_sha256")
        != _names_sha256(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)
        or features.get("clob_schema_version")
        != POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION
        or features.get("clob_width") != len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)
        or features.get("clob_names_sha256")
        != _names_sha256(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)
        or features.get("joint_schema_version")
        != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
        or features.get("joint_width")
        != len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        or features.get("joint_names_sha256")
        != _names_sha256(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        or features.get("future_receipts_allowed") is not False
        or features.get("partial_source_rows_allowed") is not False
        or features.get("structural_probability_allowed") is not False
        or features.get("resolution_or_target_in_features_allowed") is not False
        or not isinstance(population, Mapping)
        or population.get("unit") != "whole_five_minute_condition"
        or population.get("order") != "chronological"
        or population.get("training_endpoints_per_condition") != 16
        or population.get("condition_weight") != 1.0
        or population.get("endpoint_weight_within_condition") != 0.0625
        or population.get("live_decision_cadence_ms") != 250
        or population.get("target") != "official_polymarket_resolved_outcome_only"
        or population.get("constructed_twap_terminal_target_allowed") is not False
        or not isinstance(candidates, list)
        or tuple(
            item.get("candidate_id") if isinstance(item, Mapping) else None
            for item in candidates
        )
        != POLYMARKET_ROUND25_CANDIDATE_IDS
        or not isinstance(selection, Mapping)
        or selection.get("market_prior_control_must_be_beaten") is not True
        or selection.get("economic_replay_after_predictive_gate_only") is not True
        or selection.get("negative_or_null_result_published") is not True
        or not isinstance(ai, Mapping)
        or ai.get("role") != "asynchronous_risk_veto_or_position_reduction_only"
        or ai.get("direct_250ms_trade_generation_allowed") is not False
        or ai.get("entry_expansion_allowed") is not False
        or ai.get("safety_or_exit_override_allowed") is not False
        or tuple(
            item.get("candidate_id") if isinstance(item, Mapping) else None
            for item in ai.get("candidates", ())
        )
        != POLYMARKET_ROUND25_AI_CANDIDATE_IDS
        or tuple(
            item.get("candidate_id") if isinstance(item, Mapping) else None
            for item in ai.get("rejected_candidates", ())
        )
        != ("fino1-8b",)
        or not isinstance(execution, Mapping)
        or execution.get("event_time_l2_book_replay") is not True
        or execution.get("measured_collector_latency_distribution") is not True
        or execution.get("midpoint_or_zero_latency_execution_allowed") is not False
        or execution.get("paper_or_live_authority_from_predictive_result_allowed")
        is not False
        or not isinstance(truth, Mapping)
        or any(value is not False for value in truth.values())
    ):
        raise ValueError("Round 25 candidate selection design differs")
    return {**design, "design_sha256": claimed}


def load_round25_candidate_design(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 candidate selection design is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 candidate selection design is unavailable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 candidate selection design is not an object")
    return validate_round25_candidate_design(value)


def validate_round25_candidate_amendment(
    value: Mapping[str, object],
) -> dict[str, object]:
    amendment = dict(value)
    claimed = amendment.pop("design_sha256", None)
    development = amendment.get("development_campaign")
    test = amendment.get("separate_sealed_test_campaign")
    superseded = amendment.get("superseded_parent_fields")
    truth = amendment.get("truth_state")
    if (
        claimed != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
        or claimed != _canonical_sha256(amendment)
        or amendment.get("schema_version")
        != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SCHEMA_VERSION
        or amendment.get("round") != 25
        or amendment.get("status") != "frozen_before_round25_v2_capture_start"
        or amendment.get("parent_candidate_design_sha256")
        != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
        or not isinstance(development, Mapping)
        or development.get("capture_plan_sha256")
        != "a0b5525697c3c1e1b175bd0f0ac724fdb62845638d2040e9964221031d3e7b20"
        or development.get("scheduled_start_ms") != 1_786_406_400_000
        or development.get("train_end_ms") != 1_787_443_200_000
        or development.get("calibration_end_ms") != 1_787_745_600_000
        or development.get("selection_end_ms") != 1_788_046_800_000
        or tuple(development.get("roles", ()))
        != ("train", "calibration", "selection")
        or development.get("minimum_train_conditions") != 2000
        or development.get("minimum_calibration_conditions") != 400
        or development.get("minimum_selection_conditions") != 400
        or development.get("boundary_purge_conditions_each_side") != 1
        or development.get("test_role_present") is not False
        or development.get("edge_profitability_or_ai_uplift_claim_allowed")
        is not False
        or not isinstance(test, Mapping)
        or test.get("required") is not True
        or test.get("plan_available") is not False
        or test.get("must_start_after_model_and_ai_claim_freeze") is not True
        or test.get("minimum_conditions") != 500
        or test.get("single_use_target_access_required") is not True
        or test.get("development_refit_after_test_access_allowed") is not False
        or test.get("candidate_reselection_after_test_access_allowed") is not False
        or test.get("ai_reselection_after_test_access_allowed") is not False
        or not isinstance(superseded, Mapping)
        or superseded.get("population_and_splits.minimum_training_conditions")
        != 2000
        or superseded.get("population_and_splits.minimum_calibration_conditions")
        != 400
        or superseded.get("population_and_splits.minimum_selection_conditions")
        != 400
        or superseded.get("population_and_splits.minimum_sealed_test_conditions")
        != 500
        or not isinstance(truth, Mapping)
        or any(item is not False for item in truth.values())
    ):
        raise ValueError("Round 25 candidate selection amendment differs")
    return {**amendment, "design_sha256": claimed}


def load_round25_candidate_amendment(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 candidate selection amendment is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Round 25 candidate selection amendment is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 candidate selection amendment is not an object")
    return validate_round25_candidate_amendment(value)


__all__ = [
    "POLYMARKET_ROUND25_AI_CANDIDATE_IDS",
    "POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256",
    "POLYMARKET_ROUND25_CANDIDATE_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256",
    "POLYMARKET_ROUND25_CANDIDATE_IDS",
    "load_round25_candidate_amendment",
    "load_round25_candidate_design",
    "validate_round25_candidate_amendment",
    "validate_round25_candidate_design",
]
