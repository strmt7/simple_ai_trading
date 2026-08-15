"""Validation for the Round 27 LightGBM offset correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256 = (
    "52942735f5cd2b7fc56312e87349ba6dc8e65b1b3de0860b19ed5a4655840a09"
)
POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-lightgbm-offset-correction-amendment-v1.json"
)
POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD = (
    "model_implementation_amendment_sha256"
)
_BASE_MODEL_CONTRACT_SHA256 = (
    "3e18856b1f526655a514fd524378a92a878c6ec0a1857772d503b9bd7e77d439"
)
_CAMPAIGN_CONTRACT_SHA256 = (
    "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
)
_FIRST_CAPTURE_START_MS = 1_786_784_400_000
_FIRST_CAPTURE_END_MS = 1_786_811_400_000
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_EXPECTED_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "profitability_claim": False,
}
_EXPECTED_KNOWLEDGE = {
    "ai_assist_economic_metrics_computed": False,
    "model_fitted_on_stage1": False,
    "official_outcomes_accessed": False,
    "performance_metrics_computed": False,
    "sealed_partition_accessed": False,
    "selection_partition_accessed": False,
    "stage1_capture_started": True,
    "stage1_feature_rows_accessed_or_materialized": False,
}
_EXPECTED_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "aef524a2a1e986946d007fcaf1290c81428a2a4e820809d2f7f6bcffb7c83653",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "a035e6b1cb777a83e396aa2aae66e3dc48ce4712b3c2209d62804405243f85c1",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 model amendment has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 27 model amendment contains {value}")


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
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError("Round 27 model amendment SHA-256 differs")
    return selected


def validate_round27_model_amendment(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    replacements = payload.get("superseded_source_text_sha256")
    created_at_ms = payload.get("created_at_ms")
    if (
        claimed != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-lightgbm-offset-correction-amendment-v1"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or replacements != _EXPECTED_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("lightgbm_init_score") != "market_prior_logit"
        or correction.get("inference")
        != "sigmoid(market_prior_logit + correction_scale * lightgbm_raw_margin)"
        or correction.get("model_schema_version")
        != "polymarket-round27-offset-model-v2"
        or correction.get("selection_and_economic_gates_changed") is not False
        or not isinstance(discovery, Mapping)
        or discovery.get("old_training_used_market_prior_as_init_score") is not False
        or discovery.get("old_tree_prediction_was_a_market_prior_residual")
        is not False
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
    ):
        raise ValueError("Round 27 model amendment differs")
    return {**payload, "amendment_sha256": claimed}


def load_round27_model_amendment(
    repository: str | Path,
    path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected = (
        root / POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH
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
        raise ValueError("Round 27 model amendment is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 model amendment must be an object")
    return validate_round27_model_amendment(value)


__all__ = [
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256",
    "load_round27_model_amendment",
    "validate_round27_model_amendment",
]
