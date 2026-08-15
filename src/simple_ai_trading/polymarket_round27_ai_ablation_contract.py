"""Exact validation for the frozen Round 27 matched AI ablation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256 = (
    "ac264326f28e501ce355795dbd4bb89ac0c6ac2ca1b454d2a290b12167b93d15"
)
POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-ai-matched-ablation-contract-v1.json"
)
POLYMARKET_ROUND27_AI_ABLATION_SCHEMA_VERSION = (
    "polymarket-round27-ai-matched-ablation-contract-v1"
)
POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS = 5.0
POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES = 60
POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS = 20
POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS = 30
POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION = 0.5
POLYMARKET_ROUND27_AI_REASON_CODES = (
    "cross_venue_disagreement",
    "insufficient_evidence",
    "late_horizon_risk",
    "liquidity_thin",
    "model_market_disagreement",
    "no_material_risk",
    "source_stale_or_gapped",
    "spread_or_cost_risk",
    "volatility_or_jump_risk",
)
_FIRST_CAPTURE_START_MS = 1_786_784_400_000
_MODEL_CONTRACT_SHA256 = (
    "3e18856b1f526655a514fd524378a92a878c6ec0a1857772d503b9bd7e77d439"
)
_ECONOMIC_AMENDMENT_SHA256 = (
    "b977ba0e7e199edff1bfbb95163d4efcec49f9e92c6c4fe04bdb5e3dd80698de"
)
_HOST_QUALIFICATION_SHA256 = (
    "9fc177849bf5e62e3d3a625b26c0c84d5fa48dc14f97b6fc5a5c3b4b37ca34de"
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 AI ablation contract has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 27 AI ablation contract contains {value}")


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


def validate_round27_ai_ablation_contract(
    value: Mapping[str, object],
) -> dict[str, object]:
    contract = dict(value)
    claimed = str(contract.pop("contract_sha256", "")).lower()
    authority = contract.get("authority")
    case_policy = contract.get("case_materialization")
    evaluation = contract.get("evaluation")
    prompt = contract.get("prompt_contract")
    runtime = contract.get("inference_runtime")
    reduce_semantics = contract.get("reduce_semantics")
    candidates = contract.get("candidate_program")
    parents = contract.get("parents")
    if (
        claimed != POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != POLYMARKET_ROUND27_AI_ABLATION_SCHEMA_VERSION
        or contract.get("status")
        != "frozen_before_stage1_capture_market_state_or_outcome_access"
        or type(contract.get("created_at_ms")) is not int
        or int(contract["created_at_ms"]) >= _FIRST_CAPTURE_START_MS
        or contract.get("economic_amendment_sha256")
        != _ECONOMIC_AMENDMENT_SHA256
        or contract.get("host_qualification_evidence_sha256")
        != _HOST_QUALIFICATION_SHA256
        or not isinstance(authority, Mapping)
        or set(authority.values()) != {False}
        or not isinstance(candidates, list)
        or len(candidates) != 2
        or {item.get("model_id") for item in candidates if isinstance(item, Mapping)}
        != {"Qwen/Qwen3.5-9B", "OpenDataArena/ODA-Fin-SFT-8B"}
        or not isinstance(case_policy, Mapping)
        or case_policy.get("case_source_process_may_open_target_store") is not False
        or case_policy.get("case_source_process_may_receive_outcome_or_economic_report_path")
        is not False
        or case_policy.get("prompt_bytes_identical_between_candidates_for_each_case")
        is not True
        or case_policy.get("one_case_per_condition") is not True
        or not isinstance(prompt, Mapping)
        or prompt.get("action") != "risk_veto_only"
        or prompt.get("target_allowed") is not False
        or prompt.get("outcome_or_resolution_allowed") is not False
        or prompt.get("future_book_allowed") is not False
        or prompt.get("pnl_or_profitability_allowed") is not False
        or tuple(prompt.get("allowed_reason_codes", ()))
        != POLYMARKET_ROUND27_AI_REASON_CODES
        or not isinstance(runtime, Mapping)
        or runtime.get("case_timeout_seconds")
        != int(POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS)
        or runtime.get("temperature") != 0
        or runtime.get("seed") != 27
        or runtime.get("think") is not False
        or runtime.get("models_resident_concurrently") is not False
        or not isinstance(reduce_semantics, Mapping)
        or reduce_semantics.get("lower_valid_quantity_exists") is not False
        or reduce_semantics.get("reduce_execution_semantics") != "abstain"
        or not isinstance(evaluation, Mapping)
        or evaluation.get("minimum_baseline_candidate_conditions")
        != POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES
        or evaluation.get("minimum_changed_action_conditions")
        != POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS
        or evaluation.get("minimum_ai_filled_conditions_each_delay")
        != POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS
        or evaluation.get("maximum_rejected_fraction")
        != POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION
        or evaluation.get("delay_scenarios_ms") != [250, 500, 1000, 2000]
        or evaluation.get("ai_latency_treatment")
        != "base_delay_ms_plus_case_measured_wall_latency_ms_ceiling"
        or evaluation.get("invalid_timeout_or_provider_error_policy")
        != "risk_reject_and_candidate_disqualification"
        or not isinstance(parents, Mapping)
        or parents.get("model_contract_sha256") != _MODEL_CONTRACT_SHA256
    ):
        raise ValueError("Round 27 AI ablation contract differs")
    return {**contract, "contract_sha256": claimed}


def load_round27_ai_ablation_contract(
    repository: str | Path,
    path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected = (
        root / POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_RELATIVE_PATH
        if path is None
        else Path(path).resolve()
    )
    try:
        value = json.loads(
            selected.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 AI ablation contract is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 AI ablation contract must be an object")
    return validate_round27_ai_ablation_contract(value)


__all__ = [
    "POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256",
    "POLYMARKET_ROUND27_AI_ABLATION_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS",
    "POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION",
    "POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES",
    "POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS",
    "POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS",
    "POLYMARKET_ROUND27_AI_REASON_CODES",
    "load_round27_ai_ablation_contract",
    "validate_round27_ai_ablation_contract",
]
