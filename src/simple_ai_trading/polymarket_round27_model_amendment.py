"""Validation for the cumulative Round 27 model implementation amendments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256 = (
    "a2ee7a5c3f89b1bca66ca3f8dd673c760719b3f47154f678b5186ee825ae3b1e"
)
POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-automatic-block-length-correction-amendment-v8.json"
)
POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD = (
    "model_implementation_amendment_sha256"
)
_PREDECESSOR_AMENDMENT_SHA256 = (
    "4efe95538114bfd814a25867b8a933b2c19b01433b953a3ee7cd57ac019c8a81"
)
_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-dependent-bootstrap-correction-amendment-v5.json"
)
_V6_PREDECESSOR_AMENDMENT_SHA256 = (
    "3d23b811f964df8d91f2f08fc5e5088293770ec2d758f4aff252173d30a425c0"
)
_V6_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-opportunity-bootstrap-correction-amendment-v6.json"
)
_V7_PREDECESSOR_AMENDMENT_SHA256 = (
    "6dc1fd872ade724df0af5fc8f41382cb4681baefba01dd220b25f98ccec10fb6"
)
_V7_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-boundary-return-and-brier-confidence-correction-amendment-v7.json"
)
_WALK_FORWARD_PREDECESSOR_AMENDMENT_SHA256 = (
    "e3ce6285cea10337f50383cdd2b89dd048d8f015f889adaa9cc0045088a44833"
)
_WALK_FORWARD_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-embargoed-walk-forward-correction-amendment-v4.json"
)
_ACTIVE_TICK_PREDECESSOR_AMENDMENT_SHA256 = (
    "e4890d02d355f8a4f5f3054232b24cdf08d3348031826415ef5a8bc9b210f4d8"
)
_ACTIVE_TICK_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-active-tick-execution-correction-amendment-v3.json"
)
_CALIBRATION_PREDECESSOR_AMENDMENT_SHA256 = (
    "8c4c7e48062446d9b6d87c716c22004fa729be094388ce6202480cc6e2098afd"
)
_CALIBRATION_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-calibration-identity-correction-amendment-v2.json"
)
_ORIGINAL_PREDECESSOR_AMENDMENT_SHA256 = (
    "52942735f5cd2b7fc56312e87349ba6dc8e65b1b3de0860b19ed5a4655840a09"
)
_ORIGINAL_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-lightgbm-offset-correction-amendment-v1.json"
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
_EXPECTED_FINAL_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
        "corrected": "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a",
        "frozen": "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2",
    },
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
        "corrected": "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b",
        "frozen": "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6",
    },
    "src/simple_ai_trading/polymarket_round27_economics.py": {
        "corrected": "913fd020f65a66c69dc6e0ff36d99c3842f82d17d17e73907974cd54fbd0fbff",
        "frozen": "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54",
    },
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "2c124d761045d787014580852b55e82416738f1316e8928df66e5e5f799cd9fc",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}
_EXPECTED_FINAL_PREDECESSOR_SOURCES = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
        "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
    ),
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
        "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae"
    ),
    "src/simple_ai_trading/polymarket_round27_economics.py": (
        "764f912b0d97134c732c023ccb7c81f14bfc6ce6c6252de0aa20cee0a2857b47"
    ),
    "src/simple_ai_trading/polymarket_round27_experiment.py": (
        "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c"
    ),
    "src/simple_ai_trading/polymarket_round27_model.py": (
        "b01a98cfa846aa98882ba381610256029a9d4e05aaec4d4a4c4c0531142987c8"
    ),
}
_EXPECTED_LATEST_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
        "corrected": "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a",
        "frozen": "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2",
    },
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
        "corrected": "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae",
        "frozen": "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6",
    },
    "src/simple_ai_trading/polymarket_round27_economics.py": {
        "corrected": "764f912b0d97134c732c023ccb7c81f14bfc6ce6c6252de0aa20cee0a2857b47",
        "frozen": "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54",
    },
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "b01a98cfa846aa98882ba381610256029a9d4e05aaec4d4a4c4c0531142987c8",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}
_EXPECTED_LATEST_PREDECESSOR_SOURCES = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
        "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
    ),
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
        "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae"
    ),
    "src/simple_ai_trading/polymarket_round27_economics.py": (
        "e7d465cdbca29b5f3d94d7f3c3d4be80409a961ef31139f846b757ac6ebf4714"
    ),
    "src/simple_ai_trading/polymarket_round27_experiment.py": (
        "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c"
    ),
    "src/simple_ai_trading/polymarket_round27_model.py": (
        "ad7d9ef2d9cdd44671ea2dc5cd8cd1f09d134d722b03f5ba8f0f78abf8412fd6"
    ),
}
_EXPECTED_CURRENT_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
        "corrected": "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a",
        "frozen": "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2",
    },
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
        "corrected": "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae",
        "frozen": "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6",
    },
    "src/simple_ai_trading/polymarket_round27_economics.py": {
        "corrected": "e7d465cdbca29b5f3d94d7f3c3d4be80409a961ef31139f846b757ac6ebf4714",
        "frozen": "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54",
    },
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "ad7d9ef2d9cdd44671ea2dc5cd8cd1f09d134d722b03f5ba8f0f78abf8412fd6",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}
_EXPECTED_CURRENT_PREDECESSOR_SOURCES = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
        "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
    ),
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
        "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae"
    ),
    "src/simple_ai_trading/polymarket_round27_economics.py": (
        "e7d465cdbca29b5f3d94d7f3c3d4be80409a961ef31139f846b757ac6ebf4714"
    ),
    "src/simple_ai_trading/polymarket_round27_experiment.py": (
        "9f75fe4546de0350a5fb0cee9ff7652daf0bdb77373f9bdb9faf471878161865"
    ),
    "src/simple_ai_trading/polymarket_round27_model.py": (
        "e2d0ebb3055529d45e18593f4d9e006ec1bcc9724b05ef9461daa4803efc51f7"
    ),
}
_EXPECTED_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
        "corrected": "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a",
        "frozen": "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2",
    },
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
        "corrected": "372564ee247d0211adcdc5a112ac7bbce1e9a9fb5057e63a24430fe42a953aae",
        "frozen": "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6",
    },
    "src/simple_ai_trading/polymarket_round27_economics.py": {
        "corrected": "e7d465cdbca29b5f3d94d7f3c3d4be80409a961ef31139f846b757ac6ebf4714",
        "frozen": "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54",
    },
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "9f75fe4546de0350a5fb0cee9ff7652daf0bdb77373f9bdb9faf471878161865",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "e2d0ebb3055529d45e18593f4d9e006ec1bcc9724b05ef9461daa4803efc51f7",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}
_EXPECTED_PREDECESSOR_SOURCES = {
    "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
        "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2"
    ),
    "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
        "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6"
    ),
    "src/simple_ai_trading/polymarket_round27_economics.py": (
        "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
    ),
    "src/simple_ai_trading/polymarket_round27_experiment.py": (
        "faf2e36ca24273d413adbdd64ec062a426ba22464bc4aeb5561c9f6f428053c6"
    ),
    "src/simple_ai_trading/polymarket_round27_model.py": (
        "e2d0ebb3055529d45e18593f4d9e006ec1bcc9724b05ef9461daa4803efc51f7"
    ),
}
_EXPECTED_V2_PREDECESSOR_SOURCES = {
    "src/simple_ai_trading/polymarket_round27_experiment.py": (
        "aef524a2a1e986946d007fcaf1290c81428a2a4e820809d2f7f6bcffb7c83653"
    ),
    "src/simple_ai_trading/polymarket_round27_model.py": (
        "a035e6b1cb777a83e396aa2aae66e3dc48ce4712b3c2209d62804405243f85c1"
    ),
}
_EXPECTED_PREDECESSOR_REPLACEMENTS = {
    "src/simple_ai_trading/polymarket_round27_experiment.py": {
        "corrected": "faf2e36ca24273d413adbdd64ec062a426ba22464bc4aeb5561c9f6f428053c6",
        "frozen": "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e",
    },
    "src/simple_ai_trading/polymarket_round27_model.py": {
        "corrected": "e2d0ebb3055529d45e18593f4d9e006ec1bcc9724b05ef9461daa4803efc51f7",
        "frozen": "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740",
    },
}
_EXPECTED_ORIGINAL_PREDECESSOR_REPLACEMENTS = {
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


def _load_strict(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 model amendment is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 model amendment must be an object")
    return value


def _validate_original_predecessor(value: Mapping[str, object]) -> None:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    if (
        claimed != _ORIGINAL_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-lightgbm-offset-correction-amendment-v1"
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256")
        != _EXPECTED_ORIGINAL_PREDECESSOR_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("lightgbm_init_score") != "market_prior_logit"
        or correction.get("selection_and_economic_gates_changed") is not False
        or not isinstance(discovery, Mapping)
        or discovery.get("old_tree_prediction_was_a_market_prior_residual")
        is not False
    ):
        raise ValueError("Round 27 predecessor model amendment differs")


def _validate_calibration_predecessor(value: Mapping[str, object]) -> None:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    if (
        claimed != _CALIBRATION_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-calibration-identity-correction-amendment-v2"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _ORIGINAL_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != _EXPECTED_V2_PREDECESSOR_SOURCES
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256")
        != _EXPECTED_PREDECESSOR_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("model_schema_version")
        != "polymarket-round27-offset-model-v2"
        or correction.get(
            "all_allowed_correction_scales_receive_distinct_bound_model_identities"
        )
        is not True
        or correction.get("selection_and_economic_gates_changed") is not False
        or not isinstance(discovery, Mapping)
        or discovery.get("old_scale_change_recomputed_model_sha256") is not False
        or discovery.get("old_selection_identity_bound_scaled_predictions")
        is not False
        or discovery.get(
            "corrected_non_unit_prediction_is_byte_identical_after_reload"
        )
        is not True
    ):
        raise ValueError("Round 27 predecessor model amendment differs")


def _validate_active_tick_predecessor(
    value: Mapping[str, object],
) -> None:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    research = payload.get("research_basis")
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    if (
        set(payload) != expected_fields
        or claimed != _ACTIVE_TICK_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-active-tick-execution-correction-amendment-v3"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _CALIBRATION_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != _EXPECTED_PREDECESSOR_SOURCES
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256") != _EXPECTED_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("ai_case_schema_version")
        != "polymarket-round27-ai-case-v2"
        or correction.get("ai_prompt_fields_changed") is not False
        or correction.get("candidate_limit_uses_decision_book_active_tick_size")
        is not True
        or correction.get("economic_gate_thresholds_changed") is not False
        or correction.get("economic_report_schema_version")
        != "polymarket-round27-economic-replay-v2"
        or correction.get(
            "execution_limit_revalidated_against_execution_book_active_tick_size"
        )
        is not True
        or correction.get("prediction_model_or_threshold_changed") is not False
        or correction.get("recorded_book_prices_must_align_to_active_tick_size")
        is not True
        or not isinstance(discovery, Mapping)
        or discovery.get("frozen_candidate_limit_used_static_market_tick_size")
        is not True
        or discovery.get(
            "frozen_execution_revalidated_limit_against_active_tick_size"
        )
        is not False
        or discovery.get("official_market_channel_emits_tick_size_change")
        is not True
        or discovery.get("official_order_price_requires_active_tick_alignment")
        is not True
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
        or research
        != [
            {
                "purpose": "official_dynamic_tick_size_event",
                "url": "https://docs.polymarket.com/market-data/websocket/market-channel",
            },
            {
                "purpose": "official_order_tick_size_rejection_rule",
                "url": "https://docs.polymarket.com/trading/orders/create",
            },
            {
                "purpose": "official_orderbook_active_tick_field",
                "url": "https://docs.polymarket.com/trading/orderbook",
            },
        ]
    ):
        raise ValueError("Round 27 model amendment differs")
    return None


def _validate_walk_forward_predecessor(
    value: Mapping[str, object],
) -> None:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    research = payload.get("research_basis")
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    if (
        set(payload) != expected_fields
        or claimed != _WALK_FORWARD_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-embargoed-walk-forward-correction-amendment-v4"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _ACTIVE_TICK_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != _EXPECTED_CURRENT_PREDECESSOR_SOURCES
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256")
        != _EXPECTED_CURRENT_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("candidate_families_changed") is not False
        or correction.get("economic_or_prediction_gate_thresholds_changed")
        is not False
        or correction.get("future_conditions_may_train_a_past_validation_fold")
        is not False
        or correction.get("l2_penalty_selection")
        != "five_fold_expanding_condition_grouped_walk_forward"
        or correction.get("model_payload_schema_changed") is not False
        or correction.get("pre_validation_embargo_ms") != 600_000
        or correction.get("validation_block_count") != 5
        or correction.get("validation_loss_weighting")
        != (
            "equal_weight_per_condition_across_all_walk_forward_validation_blocks"
        )
        or not isinstance(discovery, Mapping)
        or discovery.get("frozen_condition_hash_folds_grouped_rows_by_condition")
        is not True
        or discovery.get("frozen_condition_hash_folds_preserved_temporal_direction")
        is not False
        or discovery.get("frozen_condition_hash_folds_permitted_future_training_conditions")
        is not True
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
        or research
        != [
            {
                "purpose": "official_time_series_cross_validation_temporal_direction",
                "url": (
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.model_selection.TimeSeriesSplit.html"
                ),
            },
            {
                "purpose": "primary_financial_backtest_overfitting_analysis",
                "url": "https://doi.org/10.21314/JCF.2016.322",
            },
        ]
    ):
        raise ValueError("Round 27 model amendment differs")
    return None


def _validate_predecessor(
    value: Mapping[str, object],
) -> None:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    research = payload.get("research_basis")
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    if (
        set(payload) != expected_fields
        or claimed != _PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-dependent-bootstrap-correction-amendment-v5"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _WALK_FORWARD_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != _EXPECTED_LATEST_PREDECESSOR_SOURCES
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256")
        != _EXPECTED_LATEST_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("ai_matched_uplift_uses_same_corrected_bootstrap")
        is not True
        or correction.get("candidate_families_or_ai_prompts_changed") is not False
        or correction.get("condition_order")
        != "event_start_ms_then_condition_id"
        or correction.get("confidence_interval_method")
        != "stationary_bootstrap_block_length_sensitivity_envelope"
        or correction.get("economic_or_prediction_gate_thresholds_changed")
        is not False
        or correction.get("economic_report_schema_version")
        != "polymarket-round27-economic-replay-v3"
        or correction.get("expected_block_durations_ms")
        != [300_000, 1_200_000, 3_600_000]
        or correction.get("expected_block_lengths_conditions") != [1, 4, 12]
        or correction.get("lower_bound_aggregation")
        != "minimum_across_block_lengths"
        or correction.get("minimum_bootstrap_conditions") != 20
        or correction.get("upper_bound_aggregation")
        != "maximum_across_block_lengths"
        or not isinstance(discovery, Mapping)
        or discovery.get(
            "frozen_economic_bootstrap_resampled_conditions_independently"
        )
        is not True
        or discovery.get(
            "frozen_prediction_bootstrap_ordered_conditions_by_opaque_id"
        )
        is not True
        or discovery.get(
            "frozen_prediction_bootstrap_resampled_conditions_independently"
        )
        is not True
        or discovery.get(
            "iid_condition_resampling_was_justified_for_adjacent_five_minute_markets"
        )
        is not False
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
        or research
        != [
            {
                "purpose": (
                    "primary_stationary_bootstrap_for_weakly_dependent_observations"
                ),
                "url": "https://doi.org/10.1080/01621459.1994.10476870",
            },
            {
                "purpose": "primary_financial_backtest_overfitting_analysis",
                "url": "https://doi.org/10.21314/JCF.2016.322",
            },
        ]
    ):
        raise ValueError("Round 27 model amendment differs")
    return None


def _validate_v6_predecessor(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    research = payload.get("research_basis")
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    if (
        set(payload) != expected_fields
        or claimed != _V6_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-opportunity-bootstrap-correction-amendment-v6"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != _EXPECTED_FINAL_PREDECESSOR_SOURCES
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256")
        != _EXPECTED_FINAL_REPLACEMENTS
        or not isinstance(correction, Mapping)
        or correction.get("ai_replay_uses_same_all_condition_population")
        is not True
        or correction.get("block_length_unit")
        != "chronologically_ordered_observed_conditions"
        or correction.get("candidate_families_or_ai_prompts_changed") is not False
        or correction.get("economic_bootstrap_no_fill_value_quote") != "0"
        or correction.get("economic_bootstrap_population")
        != "all_evaluated_conditions"
        or correction.get("economic_or_prediction_gate_thresholds_changed")
        is not False
        or correction.get("economic_report_schema_version")
        != "polymarket-round27-economic-replay-v4"
        or correction.get(
            "fixed_elapsed_duration_claim_for_irregular_condition_sequence"
        )
        is not False
        or correction.get("minimum_fill_and_profitable_condition_gates_retained")
        is not True
        or correction.get("prediction_bootstrap_population")
        != "all_eligible_labeled_conditions"
        or correction.get("stationary_bootstrap_expected_block_lengths_conditions")
        != [1, 4, 12]
        or not isinstance(discovery, Mapping)
        or discovery.get(
            "frozen_economic_bootstrap_dropped_abstained_and_unfilled_conditions"
        )
        is not True
        or discovery.get(
            "frozen_expected_block_duration_assumed_no_missing_conditions"
        )
        is not True
        or discovery.get("frozen_fill_rate_gate_prevented_low_activity_from_passing")
        is not True
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
        or research
        != [
            {
                "purpose": (
                    "primary_stationary_bootstrap_for_weakly_dependent_observations"
                ),
                "url": "https://doi.org/10.1080/01621459.1994.10476870",
            },
            {
                "purpose": "primary_financial_backtest_overfitting_analysis",
                "url": "https://doi.org/10.21314/JCF.2016.322",
            },
        ]
    ):
        raise ValueError("Round 27 model amendment differs")
    return {**payload, "amendment_sha256": claimed}


def _validate_v7_predecessor(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    correction = payload.get("correction")
    discovery = payload.get("discovery_audit")
    research = payload.get("research_basis")
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
            "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
        ),
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
            "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
        ),
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "913fd020f65a66c69dc6e0ff36d99c3842f82d17d17e73907974cd54fbd0fbff"
        ),
        "src/simple_ai_trading/polymarket_round27_experiment.py": (
            "2573966ecf39a5a05a34050ceed436f8e91f7e3aac90bcab0125cbd09d6dfc0c"
        ),
        "src/simple_ai_trading/polymarket_round27_features.py": (
            "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
        ),
        "src/simple_ai_trading/polymarket_round27_model.py": (
            "2c124d761045d787014580852b55e82416738f1316e8928df66e5e5f799cd9fc"
        ),
    }
    expected_replacements = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
            "corrected": (
                "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
            ),
            "frozen": (
                "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
            "corrected": (
                "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
            ),
            "frozen": (
                "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_economics.py": {
            "corrected": (
                "913fd020f65a66c69dc6e0ff36d99c3842f82d17d17e73907974cd54fbd0fbff"
            ),
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": (
                "123e2f1955b612ed16e88dcfa9fa6277c06f062498bda54cfa2fcfdd658a4ba9"
            ),
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": (
                "854fdbfe0a2ba0e8e914f3b8d6381f15222baace2b31e46c7e9cc966fc5d32f0"
            ),
            "frozen": (
                "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_model.py": {
            "corrected": (
                "200b95a41b984625e80da840b0db3695a44896763581ce1b59cc93bac7cd3177"
            ),
            "frozen": (
                "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740"
            ),
        },
    }
    if (
        set(payload) != expected_fields
        or claimed != _V7_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != (
            "polymarket-round27-boundary-return-and-brier-confidence-"
            "correction-amendment-v7"
        )
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V6_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != expected_predecessor_sources
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("superseded_source_text_sha256") != expected_replacements
        or not isinstance(correction, Mapping)
        or correction.get("ai_prompts_or_candidate_families_changed") is not False
        or correction.get("brier_score_confidence_gate_added") is not True
        or correction.get("economic_gate_changed") is not False
        or correction.get("feature_count_or_names_changed") is not False
        or correction.get("fixed_window_flow_population")
        != "receipts_at_or_after_window_start_through_decision"
        or correction.get("fixed_window_price_path_anchor")
        != "last_receipt_at_or_before_window_start_when_available"
        or correction.get("log_loss_confidence_gate_retained") is not True
        or correction.get("model_payload_schema_changed") is not False
        or correction.get("prediction_gate_became_stricter") is not True
        or correction.get("prediction_gate_numeric_thresholds_changed") is not False
        or correction.get("realized_variance_includes_boundary_to_first_in_window_return")
        is not True
        or correction.get("trade_count_notional_and_imbalance_exclude_boundary_anchor")
        is not True
        or not isinstance(discovery, Mapping)
        or discovery.get("frozen_brier_improvement_had_dependence_aware_confidence_bound")
        is not False
        or discovery.get("frozen_fixed_window_return_included_boundary_to_first_trade_move")
        is not False
        or discovery.get("frozen_trade_count_or_notional_included_pre_window_receipts")
        is not False
        or discovery.get("official_outcomes_accessed") is not False
        or discovery.get("synthetic_host_check_is_edge_or_profitability_evidence")
        is not False
        or research
        != [
            {
                "purpose": "primary_short_horizon_price_change_and_order_flow_measurement",
                "url": "https://arxiv.org/abs/1011.6402",
            },
            {
                "purpose": "primary_probability_forecast_scoring_rule_evaluation",
                "url": "https://arxiv.org/abs/1202.5140",
            },
            {
                "purpose": (
                    "primary_stationary_bootstrap_for_weakly_dependent_observations"
                ),
                "url": "https://doi.org/10.1080/01621459.1994.10476870",
            },
        ]
    ):
        raise ValueError("Round 27 model amendment differs")
    return {**payload, "amendment_sha256": claimed}


def validate_round27_model_amendment(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    expected_fields = {
        "authority",
        "base_model_contract_sha256",
        "campaign_contract_sha256",
        "correction",
        "created_at_ms",
        "discovery_audit",
        "knowledge_at_freeze",
        "predecessor_amendment_sha256",
        "predecessor_source_text_sha256",
        "rationale",
        "research_basis",
        "schema_version",
        "status",
        "superseded_source_text_sha256",
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
            "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
        ),
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
            "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
        ),
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "913fd020f65a66c69dc6e0ff36d99c3842f82d17d17e73907974cd54fbd0fbff"
        ),
        "src/simple_ai_trading/polymarket_round27_experiment.py": (
            "123e2f1955b612ed16e88dcfa9fa6277c06f062498bda54cfa2fcfdd658a4ba9"
        ),
        "src/simple_ai_trading/polymarket_round27_features.py": (
            "854fdbfe0a2ba0e8e914f3b8d6381f15222baace2b31e46c7e9cc966fc5d32f0"
        ),
        "src/simple_ai_trading/polymarket_round27_model.py": (
            "200b95a41b984625e80da840b0db3695a44896763581ce1b59cc93bac7cd3177"
        ),
    }
    expected_replacements = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
            "corrected": (
                "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
            ),
            "frozen": (
                "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
            "corrected": (
                "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
            ),
            "frozen": (
                "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_economics.py": {
            "corrected": (
                "913fd020f65a66c69dc6e0ff36d99c3842f82d17d17e73907974cd54fbd0fbff"
            ),
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": (
                "123e2f1955b612ed16e88dcfa9fa6277c06f062498bda54cfa2fcfdd658a4ba9"
            ),
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": (
                "854fdbfe0a2ba0e8e914f3b8d6381f15222baace2b31e46c7e9cc966fc5d32f0"
            ),
            "frozen": (
                "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_model.py": {
            "corrected": (
                "73de58ec5c5a1c1b79119779ff2035c7d73eabca3807aff83c07755f14123774"
            ),
            "frozen": (
                "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740"
            ),
        },
    }
    expected_correction = {
        "ai_prompts_or_candidate_families_changed": False,
        "automatic_block_length_cap_population_fraction": 0.25,
        "automatic_block_length_method": "corrected_politis_white_2004_2009",
        "automatic_block_length_rounded_up": True,
        "automatic_selector_can_only_widen_or_match_fixed_envelope": True,
        "economic_gate_changed": False,
        "economic_pnl_bootstrap_uses_same_selector": True,
        "fixed_expected_block_lengths_conditions_retained": [1, 4, 12],
        "model_or_feature_payload_schema_changed": False,
        "prediction_gate_became_stricter": True,
        "prediction_gate_numeric_thresholds_changed": False,
        "proper_score_bootstraps_use_independent_automatic_selection": True,
    }
    expected_discovery = {
        "frozen_bootstrap_had_data_adaptive_dependence_horizon": False,
        "official_outcomes_accessed": False,
        "stage1_feature_rows_accessed_or_materialized": False,
        "synthetic_host_check_is_edge_or_profitability_evidence": False,
    }
    expected_research = [
        {
            "purpose": "primary_automatic_block_length_selection",
            "url": "https://doi.org/10.1081/ETC-120028836",
        },
        {
            "purpose": "primary_stationary_bootstrap_variance_correction",
            "url": "https://doi.org/10.1080/07474930802459016",
        },
        {
            "purpose": "independent_open_source_reference_implementation",
            "url": (
                "https://bashtage.github.io/arch/bootstrap/generated/"
                "arch.bootstrap.optimal_block_length.html"
            ),
        },
    ]
    if (
        set(payload) != expected_fields
        or claimed != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-automatic-block-length-correction-amendment-v8"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V7_PREDECESSOR_AMENDMENT_SHA256
        or payload.get("predecessor_source_text_sha256")
        != expected_predecessor_sources
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("knowledge_at_freeze") != _EXPECTED_KNOWLEDGE
        or payload.get("correction") != expected_correction
        or payload.get("discovery_audit") != expected_discovery
        or payload.get("research_basis") != expected_research
        or payload.get("superseded_source_text_sha256") != expected_replacements
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
    _validate_original_predecessor(
        _load_strict(root / _ORIGINAL_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_calibration_predecessor(
        _load_strict(root / _CALIBRATION_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_active_tick_predecessor(
        _load_strict(root / _ACTIVE_TICK_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_walk_forward_predecessor(
        _load_strict(root / _WALK_FORWARD_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_predecessor(_load_strict(root / _PREDECESSOR_AMENDMENT_RELATIVE_PATH))
    _validate_v6_predecessor(
        _load_strict(root / _V6_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_v7_predecessor(
        _load_strict(root / _V7_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    return validate_round27_model_amendment(_load_strict(selected))


__all__ = [
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256",
    "load_round27_model_amendment",
    "validate_round27_model_amendment",
]
