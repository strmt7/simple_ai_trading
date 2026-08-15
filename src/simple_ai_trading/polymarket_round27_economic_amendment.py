"""Validation for the Round 27 economic population correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256 = (
    "b977ba0e7e199edff1bfbb95163d4efcec49f9e92c6c4fe04bdb5e3dd80698de"
)
POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-economic-population-amendment-v1.json"
)
POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES = 60
POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES = 60
POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD = (
    "economic_population_amendment_sha256"
)
_BASE_MODEL_CONTRACT_SHA256 = (
    "3e18856b1f526655a514fd524378a92a878c6ec0a1857772d503b9bd7e77d439"
)
_CAMPAIGN_CONTRACT_SHA256 = (
    "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
)
_FIRST_CAPTURE_START_MS = 1_786_784_400_000
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_ARTIFACT_HASH_FIELDS = frozenset(
    {"claim_sha256", "report_sha256", "result_sha256"}
)
_EXPECTED_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "profitability_claim": False,
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 economic amendment has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 27 economic amendment contains {value}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object) -> str:
    selected = str(value or "").lower()
    if (
        len(selected) != 64
        or set(selected) - _SHA256_CHARACTERS
    ):
        raise ValueError("Round 27 economic amendment SHA-256 differs")
    return selected


def validate_round27_economic_amendment(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    authority = payload.get("authority")
    audit = payload.get("mathematical_audit")
    replacement = payload.get("replacement")
    semantics = payload.get("evidence_gate_semantics")
    if (
        claimed != POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-economic-population-amendment-v1"
        or payload.get("status")
        != "frozen_before_stage1_capture_market_state_or_outcome_access"
        or type(payload.get("created_at_ms")) is not int
        or int(payload["created_at_ms"]) >= _FIRST_CAPTURE_START_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256")
        != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("superseded_fields")
        != [
            "economic_evaluation.minimum_selection_executed_trades",
            "economic_evaluation.minimum_sealed_executed_trades",
        ]
        or authority != _EXPECTED_AUTHORITY
        or not isinstance(audit, Mapping)
        or audit.get("candidate_selection")
        != "first_target_blind_positive_after_cost_candidate_per_condition"
        or audit.get("maximum_candidates_per_condition") != 1
        or audit.get("selection_conditions") != 90
        or audit.get("sealed_conditions") != 90
        or audit.get("original_minimum_selection_executed_trades") != 100
        or audit.get("original_minimum_sealed_executed_trades") != 100
        or audit.get("selection_original_gate_reachable") is not False
        or audit.get("sealed_original_gate_reachable") is not False
        or not isinstance(replacement, Mapping)
        or replacement.get("minimum_selection_executed_trades")
        != POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES
        or replacement.get("minimum_sealed_executed_trades")
        != POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES
        or not isinstance(semantics, Mapping)
        or semantics.get("minimum_executed_trades_is_a_trade_quota") is not False
        or semantics.get("may_force_or_relax_a_risk_gate_to_reach_minimum")
        is not False
        or semantics.get("minimum_profitable_conditions") != 20
        or semantics.get("failure_behavior")
        != "insufficient_evidence_no_edge_or_profitability_claim"
    ):
        raise ValueError("Round 27 economic amendment differs")
    return {**payload, "amendment_sha256": claimed}


def load_round27_economic_amendment(
    repository: str | Path,
    path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected = (
        root / POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_RELATIVE_PATH
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
        raise ValueError("Round 27 economic amendment is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 economic amendment must be an object")
    return validate_round27_economic_amendment(value)


def bind_round27_economic_amendment(
    value: Mapping[str, object],
    *,
    hash_field: str,
) -> dict[str, object]:
    """Bind an intact Round 27 artifact to the frozen population amendment."""

    if hash_field not in _ARTIFACT_HASH_FIELDS:
        raise ValueError("Round 27 economic artifact hash field differs")
    body = dict(value)
    claimed = _sha256(body.pop(hash_field, ""))
    if claimed != _canonical_sha256(body):
        raise ValueError("Round 27 economic artifact integrity differs")
    existing = body.get(POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD)
    if existing not in {None, POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256}:
        raise ValueError("Round 27 economic artifact amendment differs")
    body[POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD] = (
        POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
    )
    body[hash_field] = _canonical_sha256(body)
    return body


def validate_round27_economic_amendment_binding(
    value: Mapping[str, object],
    *,
    hash_field: str,
) -> dict[str, object]:
    """Require an intact artifact already bound to the frozen amendment."""

    if hash_field not in _ARTIFACT_HASH_FIELDS:
        raise ValueError("Round 27 economic artifact hash field differs")
    body = dict(value)
    claimed = _sha256(body.pop(hash_field, ""))
    if (
        claimed != _canonical_sha256(body)
        or body.get(POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD)
        != POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
    ):
        raise ValueError("Round 27 economic artifact amendment binding differs")
    return {**body, hash_field: claimed}


__all__ = [
    "POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD",
    "POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256",
    "POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES",
    "POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES",
    "bind_round27_economic_amendment",
    "load_round27_economic_amendment",
    "validate_round27_economic_amendment_binding",
    "validate_round27_economic_amendment",
]
