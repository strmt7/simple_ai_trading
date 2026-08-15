"""Fail-closed validation for the frozen Round 27 model experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .polymarket_round27_features import (
    POLYMARKET_ROUND27_DECISION_STEP_MS,
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND27_LONG_CONTEXT_MAXIMUM_RECEIPT_GAP_MS,
    POLYMARKET_ROUND27_LONG_CONTEXT_MINIMUM_COVERAGE,
    POLYMARKET_ROUND27_LONG_CONTEXT_WINDOW_MS,
    POLYMARKET_ROUND27_TRADE_WINDOWS_MS,
)
from .polymarket_round27_model import (
    POLYMARKET_ROUND27_CORRECTION_SCALES,
    POLYMARKET_ROUND27_L2_PENALTIES,
    Round27RoleInterval,
)
from .polymarket_round27_stage1_capture import load_round27_stage1_contract


POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round27-stage1-model-contract-v1"
)
POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256 = (
    "cb314e958b1b2e693780ab3ed699ed126acfc7279a1ec84e6dec17bc79b3fac6"
)
POLYMARKET_ROUND27_MODEL_CONTRACT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-stage1-model-contract-v1.json"
)
_STAGE1_CONTRACT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-stage1-campaign-contract-v1.json"
)
_SUPPLEMENTAL_HYPOTHESES_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-supplemental-hypothesis-preregistration-v1.json"
)
_SUPPLEMENTAL_HYPOTHESES_SHA256 = (
    "2cb8f7068fb9673c1d1c94af0a8a6d1d53a4ac382c3b5e336278a9a10543b7df"
)
_STAGE1_CONTRACT_SHA256 = (
    "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
)
_FIRST_CAPTURE_START_MS = 1_786_784_400_000
_SHA256 = frozenset("0123456789abcdef")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 model contract contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 27 model contract contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: object) -> str:
    selected = str(value or "").lower()
    if len(selected) != 64 or set(selected) - _SHA256:
        raise ValueError("Round 27 model contract SHA-256 differs")
    return selected


def validate_round27_model_contract(
    value: Mapping[str, object],
    *,
    repository: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve()
    payload = dict(value)
    claimed = _sha256(payload.pop("contract_sha256", ""))
    if (
        claimed != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION
        or payload.get("campaign_contract_sha256") != _STAGE1_CONTRACT_SHA256
        or payload.get("supplemental_hypothesis_preregistration_sha256")
        != _SUPPLEMENTAL_HYPOTHESES_SHA256
        or payload.get("status") != "frozen_before_stage1_capture_and_outcome_access"
        or type(payload.get("created_at_ms")) is not int
        or int(payload["created_at_ms"]) >= _FIRST_CAPTURE_START_MS
    ):
        raise ValueError("Round 27 model contract identity differs")
    authority = payload.get("authority")
    knowledge = payload.get("knowledge_at_freeze")
    data = payload.get("data_policy")
    minimum = payload.get("minimum_population")
    prediction = payload.get("prediction_evaluation")
    economics = payload.get("economic_evaluation")
    ai = payload.get("ai_assist")
    models = payload.get("model_candidates")
    partitions = payload.get("partitions")
    sources = payload.get("source_text_sha256")
    if (
        not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
        or not isinstance(knowledge, Mapping)
        or any(value is not False for value in knowledge.values())
        or not isinstance(data, Mapping)
        or data.get("assets") != ["BTC"]
        or data.get("decision_cadence_ms") != POLYMARKET_ROUND27_DECISION_STEP_MS
        or data.get("feature_count") != len(POLYMARKET_ROUND27_FEATURE_NAMES)
        or data.get("feature_names_sha256") != POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
        or data.get("trade_windows_ms") != list(POLYMARKET_ROUND27_TRADE_WINDOWS_MS)
        or data.get("long_context_window_ms")
        != POLYMARKET_ROUND27_LONG_CONTEXT_WINDOW_MS
        or data.get("long_context_minimum_coverage")
        != POLYMARKET_ROUND27_LONG_CONTEXT_MINIMUM_COVERAGE
        or data.get("long_context_maximum_receipt_gap_ms")
        != POLYMARKET_ROUND27_LONG_CONTEXT_MAXIMUM_RECEIPT_GAP_MS
        or data.get("target_access_during_capture_or_feature_materialization")
        is not False
        or data.get("official_outcome_access_before_contract_freeze") is not False
        or data.get("selection_unit") != "whole_condition"
        or not isinstance(minimum, Mapping)
        or minimum.get("campaign_eligible_conditions") != 300
        or any(
            type(minimum.get(name)) is not int or int(minimum[name]) < floor
            for name, floor in (
                ("train_conditions", 75),
                ("calibration_conditions", 25),
                ("selection_conditions", 90),
                ("sealed_conditions", 90),
            )
        )
        or not isinstance(prediction, Mapping)
        or prediction.get("primary_metric") != "condition_weighted_log_loss"
        or prediction.get("row_level_accuracy_alone_can_promote") is not False
        or not isinstance(economics, Mapping)
        or economics.get("candidate_selection")
        != "first_target_blind_positive_after_cost_candidate_per_condition"
        or economics.get("entry_price")
        != "fok_walk_of_actual_polymarket_clob_ask_depth_after_observed_delay"
        or economics.get("exit_value")
        != "official_binary_condition_resolution_payout"
        or economics.get("fixed_delay_scenarios_ms") != [250, 500, 1000, 2000]
        or economics.get("primary_delay_ms") != 500
        or economics.get("maximum_execution_observation_delay_ms") != 500
        or economics.get("maximum_decision_book_age_ms") != 1500
        or economics.get("markout_horizon_ms") != 1000
        or economics.get("maximum_conditions_per_book_batch") != 32
        or economics.get("minimum_expected_edge_per_contract") != "0.01"
        or economics.get("maximum_entry_cost_quote") != "10"
        or economics.get("position_quantity")
        != "captured_condition_minimum_order_size_shares"
        or economics.get("order_type") != "fill_or_kill_taker_buy"
        or economics.get("minimum_selection_executed_trades") != 100
        or economics.get("minimum_sealed_executed_trades") != 100
        or economics.get("minimum_profitable_conditions") != 20
        or economics.get("rate_limit_or_account_authority") is not False
        or economics.get("replay_residency")
        != "condition_batched_single_pass_all_delay_scenarios"
        or economics.get("sealed_access_requires")
        != [
            "exact_persisted_selected_model_payload",
            "source_bound_selection_economic_report",
            "persisted_passing_selection_economic_claim",
        ]
        or not isinstance(ai, Mapping)
        or ai.get("maximum_authority") != "veto_or_reduce"
        or ai.get("may_create_or_increase_position") is not False
        or ai.get("may_override_hard_risk_gate") is not False
        or ai.get("latency_critical_probability_prediction") is not False
        or not isinstance(models, list)
        or [item.get("model_name") for item in models if isinstance(item, Mapping)]
        != ["market_prior", "l2_offset_logistic", "shallow_lightgbm_offset"]
        or models[1].get("l2_penalties") != list(POLYMARKET_ROUND27_L2_PENALTIES)
        or models[1].get("correction_scales")
        != list(POLYMARKET_ROUND27_CORRECTION_SCALES)
        or models[2].get("selection_claim_payload")
        != "full_model_text_and_sha256"
        or not isinstance(partitions, list)
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("Round 27 model contract policy differs")
    supplemental_path = root / _SUPPLEMENTAL_HYPOTHESES_RELATIVE_PATH
    try:
        supplemental = json.loads(
            supplemental_path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 supplemental hypotheses are unavailable") from exc
    if not isinstance(supplemental, dict):
        raise ValueError("Round 27 supplemental hypotheses differ")
    supplemental_claim = _sha256(
        supplemental.pop("preregistration_sha256", "")
    )
    supplemental_authority = supplemental.get("authority")
    supplemental_source = supplemental.get("source")
    if (
        supplemental_claim != _SUPPLEMENTAL_HYPOTHESES_SHA256
        or supplemental_claim != _canonical_sha256(supplemental)
        or supplemental.get("status")
        != "frozen_before_stage1_market_state_access"
        or not isinstance(supplemental_authority, Mapping)
        or any(value is not False for value in supplemental_authority.values())
        or not isinstance(supplemental_source, Mapping)
        or supplemental_source.get("classification")
        != "untrusted_reddit_food_for_thought"
        or supplemental_source.get("performance_claims_accepted") is not False
    ):
        raise ValueError("Round 27 supplemental hypotheses differ")
    intervals = tuple(
        Round27RoleInterval.from_mapping(item)
        for item in partitions
        if isinstance(item, Mapping)
    )
    if len(intervals) != len(partitions) or [item.role for item in intervals] != [
        "train",
        "purged",
        "calibration",
        "purged",
        "selection",
        "purged",
        "sealed",
        "purged",
    ]:
        raise ValueError("Round 27 model contract partitions differ")
    previous_end = 0
    for interval in sorted(intervals, key=lambda item: item.start_ms):
        if interval.start_ms < previous_end:
            raise ValueError("Round 27 model contract partitions overlap")
        previous_end = interval.end_ms
    for relative, expected in sources.items():
        relative_path = Path(str(relative))
        path = (root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or root not in path.parents
            or not path.is_file()
            or _file_sha256(path) != _sha256(expected)
        ):
            raise ValueError("Round 27 model contract source binding differs")
    stage1 = load_round27_stage1_contract(
        root / _STAGE1_CONTRACT_RELATIVE_PATH,
        repository=root,
    )
    if stage1.contract_sha256 != _STAGE1_CONTRACT_SHA256:
        raise ValueError("Round 27 model contract campaign binding differs")
    return {**payload, "contract_sha256": claimed}


def load_round27_model_contract(
    repository: str | Path,
    path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected = (
        root / POLYMARKET_ROUND27_MODEL_CONTRACT_RELATIVE_PATH
        if path is None
        else Path(path).resolve()
    )
    try:
        text = selected.read_text(encoding="ascii")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 model contract is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 model contract must be an object")
    return validate_round27_model_contract(value, repository=root)


__all__ = [
    "POLYMARKET_ROUND27_MODEL_CONTRACT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256",
    "load_round27_model_contract",
    "validate_round27_model_contract",
]
