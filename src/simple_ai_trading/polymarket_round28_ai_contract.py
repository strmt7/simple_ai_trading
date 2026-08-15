"""Strict loader for the target-free Round 28 AI risk-veto experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


POLYMARKET_ROUND28_AI_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round28-ai-risk-veto-preregistration-v1"
)
POLYMARKET_ROUND28_AI_CONTRACT_SHA256 = (
    "6f0084370cd8302fe09df5f9d1caec627aaea2fe219ef435216f95a213fcf117"
)
POLYMARKET_ROUND28_AI_CONTRACT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-028-ai-risk-veto-preregistration-v1.json"
)
POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS = 5.0
POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES = 60
POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS = 20
POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS = 30
POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION = 0.5
POLYMARKET_ROUND28_AI_REASON_CODES = (
    "bbo_source_stale_or_gapped",
    "cross_venue_disagreement",
    "insufficient_evidence",
    "late_horizon_risk",
    "liquidity_thin",
    "model_market_disagreement",
    "no_material_risk",
    "spread_or_cost_risk",
    "spot_usdm_disagreement",
    "volatility_or_jump_risk",
)
POLYMARKET_ROUND28_AI_MODEL_IDS = (
    "Qwen/Qwen3.5-9B",
    "OpenDataArena/ODA-Fin-SFT-8B",
    "OpenDataArena/ODA-Fin-RL-8B",
)
_ROUND28_FEATURE_NAMES_SHA256 = (
    "5f6d5b0f963e385c2ea67b057975bcd7f4cf60c8c60525927f2a9bff050d066d"
)
_PARENT_BINDINGS = {
    "round27_ai_ablation_contract_sha256": (
        "ac264326f28e501ce355795dbd4bb89ac0c6ac2ca1b454d2a290b12167b93d15"
    ),
    "round27_campaign_contract_sha256": (
        "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
    ),
    "round28_bbo_preregistration_sha256": (
        "8239488145f0ffe331cf9823e5517120dda0d12eb5f366cf00c5e106318d4668"
    ),
    "round28_economic_implementation_amendment_sha256": (
        "9fc117a479f931d54e31bd2db494b039d081ded9ce6c4478d2c08c7d39d61d04"
    ),
    "round28_operator_implementation_amendment_sha256": (
        "0cec84ec6dd50ee8f14d6ab236e2ae886351886eccbae696a2b73d0cbcb7f826"
    ),
    "round28_selection_implementation_amendment_sha256": (
        "005caf15e94b5f43faaa451b9f12754b7f5c6dd88b1b09fea5a8d3182eb7e306"
    ),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 28 AI contract has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 28 AI contract contains {value}")


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


def _all_false(value: object, *, fields: set[str]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == fields
        and all(item is False for item in value.values())
    )


def _candidate_program_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    required = {
        "artifact_revision",
        "artifact_sha256",
        "artifact_size_bytes",
        "artifact_source",
        "host_qualification",
        "model_id",
        "quantization",
        "role",
        "runtime_digest",
        "runtime_model",
        "upstream_revision",
    }
    if any(not isinstance(item, Mapping) or set(item) != required for item in value):
        return False
    by_id = {str(item["model_id"]): item for item in value}
    if tuple(by_id) != POLYMARKET_ROUND28_AI_MODEL_IDS:
        return False
    if any(
        _SHA256.fullmatch(str(item["artifact_sha256"])) is None
        or _COMMIT.fullmatch(str(item["upstream_revision"])) is None
        or type(item["artifact_size_bytes"]) is not int
        or int(item["artifact_size_bytes"]) <= 0
        or not str(item["runtime_model"])
        or not str(item["artifact_source"])
        or not str(item["quantization"])
        for item in value
    ):
        return False
    qualified = value[:2]
    challenger = value[2]
    return bool(
        all(
            item["host_qualification"] == "passed_round27_exact_artifact"
            and isinstance(item["runtime_digest"], str)
            and _SHA256.fullmatch(str(item["runtime_digest"])) is not None
            for item in qualified
        )
        and challenger["model_id"] == "OpenDataArena/ODA-Fin-RL-8B"
        and challenger["host_qualification"]
        == "pending_exact_artifact_download_and_amd_gpu_probe"
        and challenger["runtime_digest"] is None
    )


def validate_round28_ai_contract(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact pre-target experiment and its semantic safeguards."""

    contract = dict(value)
    claimed = str(contract.pop("contract_sha256", "")).lower()
    authority = contract.get("authority")
    cases = contract.get("case_materialization")
    evaluation = contract.get("evaluation")
    runtime = contract.get("inference_runtime")
    knowledge = contract.get("knowledge_at_freeze")
    parents = contract.get("parents")
    prompt = contract.get("prompt_contract")
    reduce_semantics = contract.get("reduce_semantics")
    sealed = contract.get("sealed_evaluation")
    if (
        claimed != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != POLYMARKET_ROUND28_AI_CONTRACT_SCHEMA_VERSION
        or contract.get("status")
        != "frozen_during_stage1_a_before_round28_feature_or_target_access"
        or type(contract.get("created_at_ms")) is not int
        or not _candidate_program_valid(contract.get("candidate_program"))
        or not _all_false(
            authority,
            fields={
                "account_connected",
                "credentials_used",
                "edge_claim",
                "execution_connected",
                "live_trading_authority",
                "orders_submitted",
                "paper_trading_authority",
                "profitability_claim",
            },
        )
        or not isinstance(knowledge, Mapping)
        or knowledge
        != {
            "official_outcomes_accessed": False,
            "round28_feature_rows_accessed_or_materialized": False,
            "round28_model_fitted": False,
            "sealed_partition_accessed": False,
            "selection_partition_accessed": False,
            "stage1_a_capture_running": True,
        }
        or parents != _PARENT_BINDINGS
        or not isinstance(cases, Mapping)
        or cases.get("feature_count") != 278
        or cases.get("feature_names_sha256") != _ROUND28_FEATURE_NAMES_SHA256
        or cases.get("one_case_per_condition") is not True
        or cases.get("prompt_bytes_identical_between_candidates_for_each_case")
        is not True
        or any(
            cases.get(field) is not False
            for field in (
                "case_source_process_may_open_target_store",
                "case_source_process_may_receive_outcome_resolution_or_economic_report_path",
                "future_books_accessed",
                "outcomes_accessed",
                "target_accessed",
            )
        )
        or not isinstance(prompt, Mapping)
        or prompt.get("action") != "risk_veto_only"
        or prompt.get("allowed_decisions") != ["reject", "reduce", "unchanged"]
        or tuple(prompt.get("allowed_reason_codes", ()))
        != POLYMARKET_ROUND28_AI_REASON_CODES
        or prompt.get("unchanged_requires_reason_codes") != ["no_material_risk"]
        or any(
            prompt.get(field) is not False
            for field in (
                "credentials_allowed",
                "free_form_external_text_allowed",
                "future_book_allowed",
                "model_id_in_prompt",
                "outcome_or_resolution_allowed",
                "pnl_or_profitability_allowed",
                "response_additional_properties",
                "target_allowed",
            )
        )
        or not isinstance(runtime, Mapping)
        or runtime.get("case_timeout_seconds")
        != int(POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS)
        or runtime.get("endpoint") != "http://127.0.0.1:11434/api/generate"
        or runtime.get("num_ctx") != 8192
        or runtime.get("seed") != 28
        or runtime.get("temperature") != 0
        or runtime.get("think") is not False
        or runtime.get("models_resident_concurrently") is not False
        or runtime.get("unload_after_each_candidate") is not True
        or not isinstance(evaluation, Mapping)
        or evaluation.get("minimum_baseline_candidate_conditions")
        != POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES
        or evaluation.get("minimum_changed_action_conditions")
        != POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS
        or evaluation.get("minimum_ai_filled_conditions_each_delay")
        != POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS
        or evaluation.get("maximum_rejected_fraction")
        != POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION
        or evaluation.get("delay_scenarios_ms") != [250, 500, 1000, 2000]
        or evaluation.get("ai_latency_treatment")
        != "base_delay_ms_plus_case_measured_wall_latency_ms_ceiling"
        or not isinstance(reduce_semantics, Mapping)
        or reduce_semantics.get("lower_valid_quantity_exists") is not False
        or reduce_semantics.get("reduce_execution_semantics") != "abstain"
        or not isinstance(sealed, Mapping)
        or sealed.get("development_retuning_after_nomination") is not False
        or sealed.get("model_or_prompt_changed_after_selection") is not False
    ):
        raise ValueError("Round 28 AI contract differs")
    return {**contract, "contract_sha256": claimed}


def load_round28_ai_contract(
    repository: str | Path,
    path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected = (
        root / POLYMARKET_ROUND28_AI_CONTRACT_RELATIVE_PATH
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
        raise ValueError("Round 28 AI contract is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 28 AI contract must be an object")
    return validate_round28_ai_contract(value)


__all__ = [
    "POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS",
    "POLYMARKET_ROUND28_AI_CONTRACT_RELATIVE_PATH",
    "POLYMARKET_ROUND28_AI_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_CONTRACT_SHA256",
    "POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION",
    "POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES",
    "POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS",
    "POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS",
    "POLYMARKET_ROUND28_AI_MODEL_IDS",
    "POLYMARKET_ROUND28_AI_REASON_CODES",
    "load_round28_ai_contract",
    "validate_round28_ai_contract",
]
