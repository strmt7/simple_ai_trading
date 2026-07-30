"""Strict preregistration boundary for the Round 21 Polymarket edge study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


POLYMARKET_ROUND21_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round21-independent-matched-edge-contract-v1"
)
POLYMARKET_ROUND21_CONTRACT_SHA256 = (
    "6aadbce31c175438c40c6a1204383d828fd78ddef93b280aa2f999f347669116"
)
POLYMARKET_ROUND21_PARENT_CONTRACT_SHA256 = (
    "90e837edc9bb8071f966f9d27335983e24c6060c4ba0d36fc3d2060913c421ad"
)
POLYMARKET_ROUND21_PARENT_QUALIFICATION_SHA256 = (
    "5260a5b6c11e8acfb1343d25c593a1af21d3a239e6cf81d1430296f4a63ee05d"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTRACT_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 contract contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 contract contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _section(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    selected = value.get(name)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Round 21 {name} section is unavailable")
    return selected


@dataclass(frozen=True, slots=True)
class PolymarketRound21Program:
    contract_sha256: str
    parent_contract_sha256: str
    parent_qualification_sha256: str
    decision_cadence_ms: int
    train_calendar_days: int
    tune_calendar_days: int
    sealed_test_calendar_days: int
    minimum_resolved_test_conditions: int
    minimum_matched_ai_decisions: int


def validate_round21_contract(
    value: Mapping[str, object],
) -> PolymarketRound21Program:
    """Reject hash tampering and changes to the frozen experiment semantics."""

    contract = dict(value)
    claimed = str(contract.pop("contract_sha256", "")).strip().lower()
    expected_top_level = {
        "schema_version",
        "round",
        "status",
        "created_at_ms",
        "parents",
        "venue_independence",
        "data_roles",
        "causal_features",
        "prediction_tasks",
        "candidate_program",
        "execution_evaluation",
        "ai_assist",
        "sealed_evaluation",
        "promotion",
        "research_basis",
        "authority",
    }
    parents = _section(contract, "parents")
    independence = _section(contract, "venue_independence")
    roles = _section(contract, "data_roles")
    features = _section(contract, "causal_features")
    tasks = _section(contract, "prediction_tasks")
    candidates = _section(contract, "candidate_program")
    execution = _section(contract, "execution_evaluation")
    ai = _section(contract, "ai_assist")
    evaluation = _section(contract, "sealed_evaluation")
    promotion = _section(contract, "promotion")
    authority = _section(contract, "authority")
    passive = _section(tasks, "passive_fill_survival")
    if (
        set(contract) != expected_top_level
        or claimed != POLYMARKET_ROUND21_CONTRACT_SHA256
        or claimed != _canonical_sha256(contract)
        or _SHA256.fullmatch(claimed) is None
        or contract["schema_version"]
        != POLYMARKET_ROUND21_CONTRACT_SCHEMA_VERSION
        or contract["round"] != 21
        or contract["status"]
        != (
            "preregistered_during_target_blind_round20_capture_"
            "before_outcome_or_model_access"
        )
        or type(contract["created_at_ms"]) is not int
        or int(contract["created_at_ms"]) <= 0
        or parents["round20_contract_sha256"]
        != POLYMARKET_ROUND21_PARENT_CONTRACT_SHA256
        or parents["round20_capture_qualification_result_sha256"]
        != POLYMARKET_ROUND21_PARENT_QUALIFICATION_SHA256
        or parents["round20_capture_is_target_blind"] is not True
        or parents["round20_capture_is_model_blind"] is not True
        or independence["execution_venue"] != "polymarket"
        or independence["binance_process_and_storage"] != "independent_sidecar"
        or independence["binance_credentials_allowed"] is not False
        or independence["binance_account_or_order_state_allowed"] is not False
        or independence["binance_execution_allowed"] is not False
        or independence["binance_failure_blocks_core_capture"] is not False
        or independence["binance_failure_blocks_polymarket_risk_or_close"]
        is not False
        or independence["binance_failure_blocks_polymarket_stop_or_recovery"]
        is not False
        or roles["no_outcome_access_during_capture"] is not True
        or roles["no_model_access_during_capture"] is not True
        or roles["optional_source_coverage_never_changes_core_condition_admission"]
        is not True
        or features["polymarket_public_trade_direction"]
        != "forbidden_as_ground_truth"
        or features["trade_direction_enrichment"]
        != "post_capture_onchain_orderfilled_join_only"
        or passive["public_book_depletion_as_fill_truth"] is not False
        or passive["maker_fill_or_rebate_credit_before_qualification"] is not False
        or candidates["no_test_refit_or_threshold_change"] is not True
        or candidates["profitability_target_used_for_parameter_selection"] is not False
        or execution["maker_rebate_credit"] is not False
        or execution["reinvestment"] is not False
        or ai["per_tick_direction_or_order_generation"] is not False
        or ai["increase_risk_or_override_safety"] is not False
        or ai["block_close_stop_or_recovery"] is not False
        or promotion["automatic"] is not False
        or authority
        != {
            "model_data_eligible": False,
            "model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 contract differs")
    return PolymarketRound21Program(
        contract_sha256=claimed,
        parent_contract_sha256=POLYMARKET_ROUND21_PARENT_CONTRACT_SHA256,
        parent_qualification_sha256=POLYMARKET_ROUND21_PARENT_QUALIFICATION_SHA256,
        decision_cadence_ms=int(roles["decision_cadence_ms"]),
        train_calendar_days=int(roles["train_calendar_days"]),
        tune_calendar_days=int(roles["tune_calendar_days"]),
        sealed_test_calendar_days=int(roles["sealed_test_calendar_days"]),
        minimum_resolved_test_conditions=int(
            evaluation["minimum_resolved_test_conditions"]
        ),
        minimum_matched_ai_decisions=int(ai["minimum_matched_decisions"]),
    )


def load_round21_contract(path: str | Path) -> PolymarketRound21Program:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_CONTRACT_BYTES
    ):
        raise ValueError("Round 21 contract is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 contract is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 contract is not an object")
    return validate_round21_contract(value)


__all__ = [
    "POLYMARKET_ROUND21_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_CONTRACT_SHA256",
    "POLYMARKET_ROUND21_PARENT_CONTRACT_SHA256",
    "POLYMARKET_ROUND21_PARENT_QUALIFICATION_SHA256",
    "PolymarketRound21Program",
    "load_round21_contract",
    "validate_round21_contract",
]
