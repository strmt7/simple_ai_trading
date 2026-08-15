"""Validation for the loaded Round 27 contract used by Round 28."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD,
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
)
from .polymarket_round27_model_contract import (
    POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION,
    POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
)


POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_SCHEMA_VERSION = (
    "polymarket-round28-loaded-contract-binding-correction-v1"
)
POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v1.json"
)
_SELECTION_AMENDMENT_SHA256 = (
    "005caf15e94b5f43faaa451b9f12754b7f5c6dd88b1b09fea5a8d3182eb7e306"
)
_OPERATOR_AMENDMENT_SHA256 = (
    "0cec84ec6dd50ee8f14d6ab236e2ae886351886eccbae696a2b73d0cbcb7f826"
)
_FIRST_STAGE1_CAPTURE_START_MS = 1_786_784_400_000
_FIRST_STAGE1_FEATURE_ACCESS_BOUNDARY_MS = 1_786_811_400_000
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "paper_trading_authority": False,
    "profitability_claim": False,
}
_KNOWLEDGE_AT_FREEZE = {
    "ai_assist_economic_metrics_computed": False,
    "model_fitted_on_stage1": False,
    "official_outcomes_accessed": False,
    "performance_metrics_computed": False,
    "sealed_partition_accessed": False,
    "selection_partition_accessed": False,
    "stage1_capture_started": True,
    "stage1_feature_rows_accessed_or_materialized": False,
}
_CORRECTION = {
    "base_contract_hash_recomputation": (
        "exclude_loader_added_model_implementation_amendment_sha256"
    ),
    "defect": (
        "Round 28 recomputed the frozen base contract hash after the Round 27 "
        "loader appended the current implementation-amendment identity."
    ),
    "feature_model_target_or_economic_logic_changed": False,
    "input_manifest_schema_version_from": (
        "polymarket-round28-selection-input-manifest-v1"
    ),
    "input_manifest_schema_version_to": (
        "polymarket-round28-selection-input-manifest-v2"
    ),
    "selection_claim_schema_version_from": "polymarket-round28-selection-v1",
    "selection_claim_schema_version_to": "polymarket-round28-selection-v2",
    "separate_model_amendment_binding_required": True,
}
_DISCOVERY_AUDIT = {
    "official_outcomes_accessed": False,
    "real_loaded_contract_preflight_failed_before_correction": True,
    "stage1_feature_rows_accessed_or_materialized": False,
    "synthetic_fixture_omitted_loader_added_amendment": True,
}
_TEST_SCOPE = {
    "financial_result": False,
    "real_loaded_contract_preflight": True,
    "stage1_data_used": False,
    "synthetic_mechanics_only": False,
}
_EXPECTED_FROZEN_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round28_operator.py": (
        "24045ce7cf206b81567dedfa36f63d835ecbfc69f3ea4eeebf33c25105c845fb"
    ),
    "src/simple_ai_trading/polymarket_round28_selection.py": (
        "a21b1ce02f5db70e58c10ae6d9724cfda6ad1ba01492ebdf9ab344cacaa2e5d6"
    ),
    "tests/test_polymarket_round28_operator.py": (
        "d543a72afcace576e709c5c9f08bee9f20c9e81d5b79b917e913ecbfd6f4e215"
    ),
    "tests/test_polymarket_round28_preregistration.py": (
        "4f936d19de6a69d18c1659b948347b1e5ef8b2eb87e3d0bda5af51dff543ef2a"
    ),
    "tests/test_polymarket_round28_selection.py": (
        "e2eef23457b5cacca38f6bc18ff4a16a75416ce638f3cc8ec5020275c300a3fc"
    ),
    "tools/run_polymarket_round28_selection.py": (
        "d429045c84f9551c98456b5c25181ebb90eb91d12489a1710d360e90136c4008"
    ),
}
_EXPECTED_SOURCE_PATHS = frozenset(
    {
        "src/simple_ai_trading/polymarket_round28_contract_binding.py",
        "src/simple_ai_trading/polymarket_round28_operator.py",
        "src/simple_ai_trading/polymarket_round28_selection.py",
        "tests/test_polymarket_round28_contract_binding.py",
        "tests/test_polymarket_round28_operator.py",
        "tests/test_polymarket_round28_operator_amendment.py",
        "tests/test_polymarket_round28_preregistration.py",
        "tests/test_polymarket_round28_selection.py",
        "tools/run_polymarket_round28_selection.py",
    }
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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 {name} SHA-256 differs")
    return selected


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 28 contract correction has duplicate keys")
        output[key] = value
    return output


def validate_loaded_round27_model_contract(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the base hash separately from its loader-added amendment binding."""

    payload = dict(value)
    amendment_sha256 = _sha256(
        payload.pop(POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD, ""),
        name="Round 27 model amendment",
    )
    contract_sha256 = _sha256(
        payload.pop("contract_sha256", ""),
        name="Round 27 model contract",
    )
    if (
        contract_sha256 != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
        or contract_sha256 != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION
        or amendment_sha256 != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
    ):
        raise ValueError("Round 28 inherited model contract binding differs")
    return {
        **payload,
        "contract_sha256": contract_sha256,
        POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD: amendment_sha256,
    }


def validate_round28_contract_binding_correction(
    value: Mapping[str, object],
    *,
    repository: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve(strict=True)
    payload = dict(value)
    claimed = _sha256(
        payload.pop("amendment_sha256", ""),
        name="contract binding correction",
    )
    sources = payload.get("source_text_sha256")
    replacements = payload.get("superseded_source_text_sha256")
    created_at_ms = payload.get("created_at_ms")
    if (
        set(payload)
        != {
            "authority",
            "correction",
            "created_at_ms",
            "discovery_audit",
            "knowledge_at_freeze",
            "predecessor_operator_amendment_sha256",
            "predecessor_selection_amendment_sha256",
            "round27_model_amendment_sha256",
            "round27_model_contract_sha256",
            "schema_version",
            "source_text_sha256",
            "status",
            "superseded_source_text_sha256",
            "test_scope",
        }
        or
        claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_SCHEMA_VERSION
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or payload.get("predecessor_selection_amendment_sha256")
        != _SELECTION_AMENDMENT_SHA256
        or payload.get("predecessor_operator_amendment_sha256")
        != _OPERATOR_AMENDMENT_SHA256
        or payload.get("round27_model_contract_sha256")
        != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
        or payload.get("round27_model_amendment_sha256")
        != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
        or payload.get("authority") != _AUTHORITY
        or payload.get("knowledge_at_freeze") != _KNOWLEDGE_AT_FREEZE
        or payload.get("correction") != _CORRECTION
        or payload.get("discovery_audit") != _DISCOVERY_AUDIT
        or payload.get("test_scope") != _TEST_SCOPE
        or type(created_at_ms) is not int
        or not (
            _FIRST_STAGE1_CAPTURE_START_MS
            < int(created_at_ms)
            < _FIRST_STAGE1_FEATURE_ACCESS_BOUNDARY_MS
        )
        or not isinstance(sources, Mapping)
        or set(str(relative) for relative in sources) != _EXPECTED_SOURCE_PATHS
        or not isinstance(replacements, Mapping)
        or set(str(relative) for relative in replacements)
        != set(_EXPECTED_FROZEN_REPLACEMENTS)
    ):
        raise ValueError("Round 28 contract binding correction differs")
    for relative, expected in sources.items():
        relative_path = Path(str(relative))
        source = (root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or root not in source.parents
            or not source.is_file()
            or hashlib.sha256(
                source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            ).hexdigest()
            != _sha256(expected, name="corrected source")
        ):
            raise ValueError("Round 28 corrected source binding differs")
    for relative, frozen in _EXPECTED_FROZEN_REPLACEMENTS.items():
        replacement = replacements.get(relative)
        if (
            not isinstance(replacement, Mapping)
            or replacement.get("frozen") != frozen
            or replacement.get("corrected") != sources[relative]
        ):
            raise ValueError("Round 28 superseded source binding differs")
    return {**payload, "amendment_sha256": claimed}


def load_round28_contract_binding_correction(
    repository: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve(strict=True)
    path = root / POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_RELATIVE_PATH
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {raw}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 28 contract binding correction is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 28 contract binding correction must be an object")
    return validate_round28_contract_binding_correction(value, repository=root)


__all__ = [
    "POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_RELATIVE_PATH",
    "POLYMARKET_ROUND28_CONTRACT_BINDING_CORRECTION_SCHEMA_VERSION",
    "load_round28_contract_binding_correction",
    "validate_loaded_round27_model_contract",
    "validate_round28_contract_binding_correction",
]
