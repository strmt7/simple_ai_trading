"""Validation for the cumulative Round 27 model implementation amendments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256 = (
    "819fcce228d873005defec3cabe7ac6033f9f53fa9e810d9a65b69dc24a2a590"
)
POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-campaign-admission-gate-correction-amendment-v16.json"
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
_V8_PREDECESSOR_AMENDMENT_SHA256 = (
    "a2ee7a5c3f89b1bca66ca3f8dd673c760719b3f47154f678b5186ee825ae3b1e"
)
_V8_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-automatic-block-length-correction-amendment-v8.json"
)
_V9_PREDECESSOR_AMENDMENT_SHA256 = (
    "79a4282de6aaa42802dadeeee2c2405aa74f2f27974f90067e89eb722a258e83"
)
_V9_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-strict-decision-cutoff-correction-amendment-v9.json"
)
_V10_PREDECESSOR_AMENDMENT_SHA256 = (
    "538526dafd9d84a831a57a456aa35d50cedb387c33684fbaac4770ea6ced456b"
)
_V10_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-effective-source-ledger-amendment-v10.json"
)
_V11_PREDECESSOR_AMENDMENT_SHA256 = (
    "5dc338fbd521e02bdecd6e90df185e6d8276276556fa6c4d2425faccf809c731"
)
_V11_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-economic-config-binding-correction-amendment-v11.json"
)
_V12_PREDECESSOR_AMENDMENT_SHA256 = (
    "7cbc5e39f8a7c663282ca2a6b34ec5a219477faed9ba7d03230cfd655f7aa8ca"
)
_V12_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-autocorrelation-normalization-correction-amendment-v12.json"
)
_V13_PREDECESSOR_AMENDMENT_SHA256 = (
    "759bedfbc395dab37f32d78c54433b6441ff4a231a71f9a018d1e9e80d922369"
)
_V13_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-source-recomputation-correction-amendment-v13.json"
)
_V14_PREDECESSOR_AMENDMENT_SHA256 = (
    "a7015cfe099a287e96f9399b9305ad99a5a41b4619b468798fb94fed8e7b9526"
)
_V14_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-settlement-hazard-correction-amendment-v14.json"
)
_V15_PREDECESSOR_AMENDMENT_SHA256 = (
    "754bec3c86d36a1f88feaa806780c65ecf71e815d90dd149be8cef2cb8c6367a"
)
_V15_PREDECESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/"
    "round-027-execution-settlement-hazard-correction-amendment-v15.json"
)
_V1_SOURCE_LEDGER_SHA256 = (
    "af847fbe265d58dc0a40f6d011a8060822fdf5a98719880d041398a527d27d92"
)
_V1_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v1.json"
)
_V2_SOURCE_LEDGER_SHA256 = (
    "75e0f74d68e1cbf87c9edd23f55bf9e79512b17b2c284ef58c01d7e89da72d91"
)
_V2_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v2.json"
)
_V3_SOURCE_LEDGER_SHA256 = (
    "972ef3e49f16ced1706a3ff0b91dae72033ae48dee9f4b15794585de26fa9493"
)
_V3_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v3.json"
)
_V4_SOURCE_LEDGER_SHA256 = (
    "850cfb612d86e1aebafb271ffbea3281d771f922153bd2a6d41e2d567a844475"
)
_V4_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v4.json"
)
_EFFECTIVE_SOURCE_LEDGER_SHA256 = (
    "f38396df1bb3f8dba662370401b562ab431f6514f0fad58210079e7d6a059581"
)
_EFFECTIVE_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v7.json"
)
_HISTORICAL_PROVENANCE_FILES = frozenset(
    {".gitattributes", "pyproject.toml", "uv.lock"}
)
_V6_SOURCE_LEDGER_SHA256 = (
    "bf2231376f0e4748e164bdf5b828d451b9f4ed00e9f3794b3906d98611dd7539"
)
_V6_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v6.json"
)
_V5_SOURCE_LEDGER_SHA256 = (
    "700d89d8220f4a888d38ce67546fa7726083547c9507b7fd15f47beaef9472f2"
)
_V5_SOURCE_LEDGER_RELATIVE_PATH = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v5.json"
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


def _validate_v8_predecessor(
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
        or claimed != _V8_PREDECESSOR_AMENDMENT_SHA256
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


def _validate_v9_predecessor(
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
            "73de58ec5c5a1c1b79119779ff2035c7d73eabca3807aff83c07755f14123774"
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
                "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
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
                "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
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
        "decision_book_receipt_cutoff": "strictly_before_decision_wall_ms",
        "economic_execution_cutoff_became_stricter": True,
        "economic_gate_numeric_thresholds_changed": False,
        "execution_and_markout_receipt_cutoff": "strictly_after_target_wall_ms",
        "feature_count_or_names_changed": False,
        "model_payload_schema_changed": False,
        "prediction_gate_numeric_thresholds_changed": False,
        "same_millisecond_receipts_treated_as_ordering_ambiguous": True,
        "trade_and_twap_receipt_cutoff": "strictly_before_decision_wall_ms",
        "twap_source_age_measured_at_decision_wall_ms": True,
    }
    expected_discovery = {
        "frozen_decision_features_included_same_wall_millisecond_receipts": True,
        "frozen_execution_replay_included_same_target_millisecond_books": True,
        "official_outcomes_accessed": False,
        "stage1_feature_rows_accessed_or_materialized": False,
        "synthetic_host_check_is_edge_or_profitability_evidence": False,
    }
    expected_research = [
        {
            "purpose": "official_polymarket_market_channel_millisecond_timestamps",
            "url": "https://docs.polymarket.com/market-data/websocket/market-channel",
        },
        {
            "purpose": "official_polymarket_rtds_millisecond_timestamps",
            "url": "https://docs.polymarket.com/market-data/websocket/rtds",
        },
        {
            "purpose": "official_binance_timestamp_resolution",
            "url": (
                "https://developers.binance.com/en/docs/"
                "binance-spot-api-docs/web-socket-streams"
            ),
        },
    ]
    if (
        set(payload) != expected_fields
        or claimed != _V9_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-strict-decision-cutoff-correction-amendment-v9"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V8_PREDECESSOR_AMENDMENT_SHA256
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


def _validate_source_ledger(
    repository: Path,
    reference: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    expected_schema_version: str,
    predecessor_sha256: str | None,
    verify_current_files: bool,
) -> dict[str, object]:
    operator_ledger = expected_schema_version in {
        "polymarket-round27-effective-source-ledger-v4",
        "polymarket-round27-effective-source-ledger-v5",
        "polymarket-round27-effective-source-ledger-v6",
        "polymarket-round27-effective-source-ledger-v7",
    }
    expected_file_count = (
        87
        if expected_schema_version == "polymarket-round27-effective-source-ledger-v7"
        else 85
        if operator_ledger
        else 68
    )
    expected_dependency_resolution = (
        "recursive_local_project_import_closure"
        if operator_ledger
        else "recursive_local_relative_import_closure"
    )
    expected_entrypoint_selection = (
        "all_round27_model_feature_target_resolution_economic_ai_and_"
        "operator_entrypoints"
        if operator_ledger
        else "all_round27_model_feature_target_resolution_economic_and_ai_modules"
    )
    expected_reference = {
        "relative_path": expected_path.as_posix(),
        "sha256": expected_sha256,
    }
    if reference != expected_reference:
        raise ValueError("Round 27 effective source ledger reference differs")
    selected = (repository / expected_path).resolve()
    if repository not in selected.parents or not selected.is_file():
        raise ValueError("Round 27 effective source ledger is unavailable")
    ledger = _load_strict(selected)
    payload = dict(ledger)
    claimed = _sha256(payload.pop("source_ledger_sha256", ""))
    files = payload.get("files_sha256")
    scope = payload.get("scope")
    exclusions = payload.get("excluded_files")
    expected_fields = {
            "authority",
            "base_model_contract_sha256",
            "campaign_contract_sha256",
            "created_at_ms",
            "excluded_files",
            "files_sha256",
            "schema_version",
            "scope",
            "status",
    }
    if predecessor_sha256 is not None:
        expected_fields.add("predecessor_source_ledger_sha256")
    if (
        set(payload) != expected_fields
        or claimed != expected_sha256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version") != expected_schema_version
        or (
            payload.get("predecessor_source_ledger_sha256")
            if predecessor_sha256 is not None
            else None
        )
        != predecessor_sha256
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or payload.get("authority") != _EXPECTED_AUTHORITY
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or type(payload.get("created_at_ms")) is not int
        or not _FIRST_CAPTURE_START_MS
        < int(payload["created_at_ms"])
        < _FIRST_CAPTURE_END_MS
        or exclusions
        != {
            "src/simple_ai_trading/polymarket_round27_model_amendment.py": (
                "self_referential_validator_excluded; exact amendment identity "
                "remains canonical-hash and commit bound"
            )
        }
        or not isinstance(files, Mapping)
        or len(files) != expected_file_count
        or not isinstance(scope, Mapping)
        or scope.get("dependency_resolution")
        != expected_dependency_resolution
        or scope.get("entrypoint_selection")
        != expected_entrypoint_selection
        or scope.get("hash_normalization")
        != "replace_crlf_with_lf_before_sha256"
        or scope.get("included_dependency_lockfiles")
        != ["pyproject.toml", "uv.lock"]
        or scope.get("locked_file_count") != expected_file_count
        or ("operator_entrypoints_included" in scope) is not operator_ledger
        or (
            operator_ledger
            and scope.get("operator_entrypoints_included") is not True
        )
        or scope.get("stage1_capture_code_included_through_model_contract") is not True
        or scope.get("unlocked_local_dependencies_per_static_import_audit") != []
    ):
        raise ValueError("Round 27 effective source ledger differs")
    if verify_current_files:
        for relative, expected in files.items():
            relative_path = Path(str(relative))
            source = (repository / relative_path).resolve()
            if (
                relative_path.is_absolute()
                or repository not in source.parents
                or not source.is_file()
            ):
                raise ValueError("Round 27 effective source ledger file differs")
            if relative_path.as_posix() in _HISTORICAL_PROVENANCE_FILES:
                continue
            if hashlib.sha256(
                source.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest() != _sha256(expected):
                raise ValueError("Round 27 effective source ledger file differs")
    return {**payload, "source_ledger_sha256": claimed}


def _validate_v10_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
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
                "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
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
                "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
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
    expected_predecessor_sources = {
        relative: replacement["corrected"]
        for relative, replacement in expected_replacements.items()
    }
    expected_correction = {
        "ai_prompts_or_candidate_families_changed": False,
        "dependency_lockfiles_bound": ["pyproject.toml", "uv.lock"],
        "economic_gate_changed": False,
        "effective_source_dependency_closure_added": True,
        "effective_source_files_bound": 68,
        "model_or_feature_payload_schema_changed": False,
        "prediction_gate_changed": False,
        "source_ledger_validator_self_hash_excluded_and_disclosed": True,
    }
    expected_discovery = {
        "official_outcomes_accessed": False,
        "prior_contract_bound_only_top_level_modules": True,
        "stage1_feature_rows_accessed_or_materialized": False,
        "synthetic_host_check_is_edge_or_profitability_evidence": False,
        "transitive_dependencies_previously_hash_bound": False,
    }
    expected_research = [
        {
            "purpose": "official_verifiable_software_artifact_provenance_definition",
            "url": "https://slsa.dev/spec/v1.2/provenance",
        }
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V10_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-effective-source-ledger-amendment-v10"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V9_PREDECESSOR_AMENDMENT_SHA256
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
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V1_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V1_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v1",
        predecessor_sha256=None,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def _validate_v11_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
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
                "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
            ),
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": (
                "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
            ),
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": (
                "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
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
    expected_predecessor_sources = {
        relative: replacement["corrected"]
        for relative, replacement in expected_replacements.items()
    }
    expected_predecessor_sources[
        "src/simple_ai_trading/polymarket_round27_experiment.py"
    ] = "123e2f1955b612ed16e88dcfa9fa6277c06f062498bda54cfa2fcfdd658a4ba9"
    expected_correction = {
        "ai_prompts_or_candidate_families_changed": False,
        "economic_gate_numeric_thresholds_changed": False,
        "economic_report_config_exactly_contract_bound": True,
        "feature_or_model_payload_schema_changed": False,
        "noncontract_config_can_reach_selection_claim": False,
        "simulator_unit_config_ranges_changed": False,
        "source_ledger_advanced": True,
    }
    expected_discovery = {
        "official_outcomes_accessed": False,
        "persisted_tool_reload_previously_checked_expected_config": True,
        "selection_claim_builder_previously_checked_expected_config": False,
        "stage1_feature_rows_accessed_or_materialized": False,
        "synthetic_host_check_is_edge_or_profitability_evidence": False,
    }
    expected_research = [
        {
            "purpose": "official_verifiable_software_artifact_provenance_definition",
            "url": "https://slsa.dev/spec/v1.2/provenance",
        }
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V11_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-economic-config-binding-correction-amendment-v11"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V10_PREDECESSOR_AMENDMENT_SHA256
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
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V2_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V2_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v2",
        predecessor_sha256=_V1_SOURCE_LEDGER_SHA256,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def _validate_v12_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
            "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
        ),
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
            "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
        ),
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
        ),
        "src/simple_ai_trading/polymarket_round27_experiment.py": (
            "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
        ),
        "src/simple_ai_trading/polymarket_round27_features.py": (
            "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
        ),
        "src/simple_ai_trading/polymarket_round27_model.py": (
            "73de58ec5c5a1c1b79119779ff2035c7d73eabca3807aff83c07755f14123774"
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
                "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
            ),
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": (
                "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
            ),
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": (
                "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
            ),
            "frozen": (
                "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_model.py": {
            "corrected": (
                "76360e4541ab7118e9ea29561d20d18dcc97dd32fffff07fce6d11af2452d4bf"
            ),
            "frozen": (
                "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740"
            ),
        },
    }
    expected_correction = {
        "ai_prompts_or_candidate_families_changed": False,
        "automatic_autocorrelation_definition": (
            "absolute_sample_autocovariance_at_lag_divided_by_"
            "sample_autocovariance_at_zero"
        ),
        "automatic_no_insignificant_run_fallback": (
            "largest_significant_positive_lag_or_one"
        ),
        "economic_or_prediction_gate_numeric_thresholds_changed": False,
        "feature_or_model_payload_schema_changed": False,
        "fixed_expected_block_lengths_conditions_retained": [1, 4, 12],
        "lag_zero_excluded_from_insignificance_scan": True,
        "source_ledger_advanced": True,
    }
    expected_discovery = {
        "deterministic_target_free_short_population_scan_found_"
        "material_block_length_differences": True,
        "inherited_pairwise_denominator_used_lag_plus_one_vectors": True,
        "official_outcomes_accessed": False,
        "primary_definition_requires_r_hat_lag_divided_by_r_hat_zero": True,
        "stage1_feature_rows_accessed_or_materialized": False,
        "synthetic_host_check_is_edge_or_profitability_evidence": False,
    }
    expected_research = [
        {
            "purpose": (
                "primary_automatic_block_length_selection_and_"
                "autocorrelation_definition"
            ),
            "url": "https://doi.org/10.1081/ETC-120028836",
        },
        {
            "purpose": "primary_stationary_bootstrap_variance_correction",
            "url": "https://doi.org/10.1080/07474930802459016",
        },
        {
            "purpose": "authors_maintained_positive_lag_reference_implementation",
            "url": "https://public.econ.duke.edu/~ap172/ppw.R.txt",
        },
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V12_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-autocorrelation-normalization-correction-amendment-v12"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256")
        != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V11_PREDECESSOR_AMENDMENT_SHA256
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
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V3_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V3_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v3",
        predecessor_sha256=_V2_SOURCE_LEDGER_SHA256,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def _validate_v13_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    """Validate the immutable v13 source-recomputation correction."""

    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    expected_correction = {
        "ai_economic_restart_checkpoints_must_match_source_recomputation": True,
        "ai_economic_reports_recomputed_from_source_before_acceptance": True,
        "ai_prompts_or_candidate_families_changed": False,
        "economic_or_prediction_gate_numeric_thresholds_changed": False,
        "feature_or_model_payload_schema_changed": False,
        "operational_round27_entrypoints_added_to_source_ledger": True,
        "source_ledger_advanced": True,
    }
    expected_discovery = {
        "existing_ai_economic_report_path_previously_skipped_source_replay": True,
        "official_outcomes_accessed": False,
        "self_consistent_json_hash_alone_does_not_prove_source_derivation": True,
        "stage1_feature_rows_accessed_or_materialized": False,
        "v3_source_ledger_omitted_round27_operator_entrypoints": True,
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": (
            "8cc090b9d95b1493c8535b6d44ecceab81a89fa6c08ef55c3e7a3a04363f641a"
        ),
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": (
            "a4763089881c6475dce2ee56bb4e38ddcc4e71c89871e147d83b3eeaf0fb556b"
        ),
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
        ),
        "src/simple_ai_trading/polymarket_round27_experiment.py": (
            "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
        ),
        "src/simple_ai_trading/polymarket_round27_features.py": (
            "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
        ),
        "src/simple_ai_trading/polymarket_round27_model.py": (
            "76360e4541ab7118e9ea29561d20d18dcc97dd32fffff07fce6d11af2452d4bf"
        ),
        "src/simple_ai_trading/polymarket_round27_operator.py": (
            "acf3a666db7b220e2cfe74b3c9c4d8bfcda845731d7f62fbae43c7baefec87e0"
        ),
        "tools/run_polymarket_round27_ai_sealed.py": (
            "a3bb98f4807de0247b848b8a5b0356baef7a2385830553783c70e3374d435519"
        ),
        "tools/run_polymarket_round27_ai_selection.py": (
            "cbbb4d523fe7a7fe169ecd38b3cd928b6843a7b0f4c76ee36311a90d4c26b2c0"
        ),
    }
    expected_replacements = {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_ai_cases.py"
            ],
            "frozen": (
                "2e95562f3611842ecb801920f9cf6876eba2d11b2e0b89a76625f3a59be97bc2"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_ai_economics.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_ai_economics.py"
            ],
            "frozen": (
                "a222dd9c4d6246aeccf90e62ff7157697c52636aed4261c532337f5016e78fe6"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_economics.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_economics.py"
            ],
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_experiment.py"
            ],
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_features.py"
            ],
            "frozen": (
                "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_model.py": {
            "corrected": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_model.py"
            ],
            "frozen": (
                "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_operator.py": {
            "corrected": (
                "184be01f8c3f45e2f225c07201608290f0bed37c6177c5f0b48b8fe922a07a13"
            ),
            "frozen": expected_predecessor_sources[
                "src/simple_ai_trading/polymarket_round27_operator.py"
            ],
        },
        "tools/run_polymarket_round27_ai_sealed.py": {
            "corrected": (
                "915afba087c2b36749a2dded499efa69d4190aa85d7b2c45903056c5e802a86b"
            ),
            "frozen": expected_predecessor_sources[
                "tools/run_polymarket_round27_ai_sealed.py"
            ],
        },
        "tools/run_polymarket_round27_ai_selection.py": {
            "corrected": (
                "fa094c5d0d06f289f0303e75ae3bbb68ba10f7f73856104992cee1c4dd078b7d"
            ),
            "frozen": expected_predecessor_sources[
                "tools/run_polymarket_round27_ai_selection.py"
            ],
        },
    }
    expected_research = [
        {
            "purpose": (
                "verifiable_artifact_provenance_tracks_outputs_to_source_and_process"
            ),
            "url": "https://slsa.dev/spec/v1.2/provenance",
        },
        {
            "purpose": "bit_exact_recomputation_and_cryptographic_comparison",
            "url": "https://reproducible-builds.org/docs/definition/",
        },
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V13_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-source-recomputation-correction-amendment-v13"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256") != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V12_PREDECESSOR_AMENDMENT_SHA256
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
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V4_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V4_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v4",
        predecessor_sha256=_V3_SOURCE_LEDGER_SHA256,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def _validate_v14_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    """Validate the immutable v14 decision-time settlement-hazard correction."""

    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    expected_correction = {
        "economic_report_schema_version_from": (
            "polymarket-round27-economic-replay-v4"
        ),
        "economic_report_schema_version_to": (
            "polymarket-round27-economic-replay-v5"
        ),
        "feature_model_ai_candidates_changed": False,
        "minimum_new_entry_time_to_settlement_ms": 60_000,
        "settlement_hazard_gate_may_be_overridden": False,
        "source_ledger_advanced": True,
    }
    expected_discovery = {
        "official_outcomes_accessed": False,
        "prior_policy_permitted_new_entries_until_final_five_seconds": True,
        "research_revision_published_at": "2026-08-11",
        "stage1_feature_rows_accessed_or_materialized": False,
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "17743f3b178d656d88dd35e4614900e0bbacfe0e4decf494bb4fbd3127bffa8a"
        )
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
                "a9f78fac647caf5eafcc7221d498545f39a7bb1fa8d3ad5bc9458f3bb2c861db"
            ),
            "frozen": (
                "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_experiment.py": {
            "corrected": (
                "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
            ),
            "frozen": (
                "9a97a253668e9ef2487c042c3574b4bea2f5cf7e6fcd5267a1f6e6fc1ed5321e"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_features.py": {
            "corrected": (
                "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
            ),
            "frozen": (
                "032f249028418d7a479c014874a374b1dc6e68de80350b68dad83ca5aae58316"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_model.py": {
            "corrected": (
                "76360e4541ab7118e9ea29561d20d18dcc97dd32fffff07fce6d11af2452d4bf"
            ),
            "frozen": (
                "5eedf0a5e6f7c0317c795d99ad7425ff2e262c2d527c519d4f9d9cee7f8e8740"
            ),
        },
        "src/simple_ai_trading/polymarket_round27_operator.py": {
            "corrected": (
                "184be01f8c3f45e2f225c07201608290f0bed37c6177c5f0b48b8fe922a07a13"
            ),
            "frozen": (
                "acf3a666db7b220e2cfe74b3c9c4d8bfcda845731d7f62fbae43c7baefec87e0"
            ),
        },
        "tools/run_polymarket_round27_ai_sealed.py": {
            "corrected": (
                "915afba087c2b36749a2dded499efa69d4190aa85d7b2c45903056c5e802a86b"
            ),
            "frozen": (
                "a3bb98f4807de0247b848b8a5b0356baef7a2385830553783c70e3374d435519"
            ),
        },
        "tools/run_polymarket_round27_ai_selection.py": {
            "corrected": (
                "fa094c5d0d06f289f0303e75ae3bbb68ba10f7f73856104992cee1c4dd078b7d"
            ),
            "frozen": (
                "cbbb4d523fe7a7fe169ecd38b3cd928b6843a7b0f4c76ee36311a90d4c26b2c0"
            ),
        },
    }
    expected_research = [
        {
            "purpose": "five_minute_bitcoin_settlement_manipulation_hazard",
            "url": "https://arxiv.org/abs/2606.31675",
        },
        {
            "purpose": "live_source_contract_for_binance_and_chainlink",
            "url": "https://docs.polymarket.com/market-data/websocket/rtds",
        },
        {
            "purpose": "chainlink_btc_usd_stream_identity",
            "url": "https://data.chain.link/streams/btc-usd-cexprice-streams",
        },
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V14_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-settlement-hazard-correction-amendment-v14"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256") != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V13_PREDECESSOR_AMENDMENT_SHA256
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
    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V5_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V5_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v5",
        predecessor_sha256=_V4_SOURCE_LEDGER_SHA256,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def _validate_v15_predecessor(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    """Validate the immutable v15 execution-time settlement-hazard correction."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    predecessor = _validate_v14_predecessor(
        _load_strict(root / _V14_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    predecessor_replacements = predecessor.get("superseded_source_text_sha256")
    if not isinstance(predecessor_replacements, Mapping):
        raise ValueError("Round 27 predecessor replacement map differs")
    expected_replacements = dict(predecessor_replacements)
    expected_replacements["src/simple_ai_trading/polymarket_round27_economics.py"] = {
        "corrected": (
            "fd34be8bb07bf16a528d1daae67a46dedc62e98c8fc865a6c240471a3234ec24"
        ),
        "frozen": (
            "539daa52e4d5bd1f4a03b15cb81951c587aa668ec6d91cb18a2a09209e8f7f54"
        ),
    }
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    expected_correction = {
        "decision_time_settlement_hazard_gate_retained": True,
        "economic_report_schema_version_from": (
            "polymarket-round27-economic-replay-v5"
        ),
        "economic_report_schema_version_to": (
            "polymarket-round27-economic-replay-v6"
        ),
        "execution_receipt_settlement_hazard_gate_added": True,
        "feature_model_ai_candidates_changed": False,
        "minimum_new_entry_time_to_settlement_ms": 60_000,
        "settlement_hazard_gate_may_be_overridden": False,
        "source_ledger_advanced": True,
    }
    expected_discovery = {
        "execution_latency_could_cross_blocked_window": True,
        "official_outcomes_accessed": False,
        "stage1_feature_rows_accessed_or_materialized": False,
        "v14_checked_decision_timestamp_only": True,
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_economics.py": (
            "a9f78fac647caf5eafcc7221d498545f39a7bb1fa8d3ad5bc9458f3bb2c861db"
        )
    }
    expected_research = [
        {
            "purpose": "five_minute_bitcoin_settlement_manipulation_hazard",
            "url": "https://arxiv.org/abs/2606.31675",
        },
        {
            "purpose": "live_source_contract_for_binance_and_chainlink",
            "url": "https://docs.polymarket.com/market-data/websocket/rtds",
        },
        {
            "purpose": "chainlink_btc_usd_stream_identity",
            "url": "https://data.chain.link/streams/btc-usd-cexprice-streams",
        },
    ]
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != _V15_PREDECESSOR_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-execution-settlement-hazard-correction-amendment-v15"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256") != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V14_PREDECESSOR_AMENDMENT_SHA256
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
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_V6_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_V6_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v6",
        predecessor_sha256=_V5_SOURCE_LEDGER_SHA256,
        verify_current_files=False,
    )
    return {**payload, "amendment_sha256": claimed}


def validate_round27_model_amendment(
    value: Mapping[str, object],
    *,
    repository: str | Path | None = None,
) -> dict[str, object]:
    """Validate the current campaign-admission target-access correction."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else Path(repository).resolve()
    )
    predecessor = _validate_v15_predecessor(
        _load_strict(root / _V15_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    predecessor_replacements = predecessor.get("superseded_source_text_sha256")
    if not isinstance(predecessor_replacements, Mapping):
        raise ValueError("Round 27 predecessor replacement map differs")
    expected_replacements = dict(predecessor_replacements)
    expected_replacements.update(
        {
            "src/simple_ai_trading/polymarket_round27_target_store.py": {
                "corrected": (
                    "7a4fd1a48ce06bc0a5ce785090457f83563514f8daec03eb8f063a8531aa553b"
                ),
                "frozen": (
                    "8d8b3bca60d89dba108d3ebd379bddf465a481bdd36418587a1b43a78b8e088e"
                ),
            },
            "tools/collect_polymarket_round27_targets.py": {
                "corrected": (
                    "5a308c2b6aebbec6950e1da94eb35e2a89873804e0331c52830468dba0863db7"
                ),
                "frozen": (
                    "3591a9b483c7ed0b658e5825152d19e5a43e25e8d15fb2d77e20772a780b46fc"
                ),
            },
        }
    )
    expected_correction = {
        "all_primary_target_free_audits_required": True,
        "campaign_admission_artifact_required": True,
        "contingency_role_assignment_changed": False,
        "exact_role_feature_populations_bound": True,
        "feature_model_economic_or_ai_candidates_changed": False,
        "minimum_campaign_eligible_conditions": 300,
        "model_role_minima_required": True,
        "source_ledger_advanced": True,
        "target_access_schema_version_from": (
            "polymarket-round27-role-target-access-v1"
        ),
        "target_access_schema_version_to": (
            "polymarket-round27-role-target-access-v2"
        ),
        "target_store_schema_version_from": (
            "polymarket-round27-role-gated-target-store-v1"
        ),
        "target_store_schema_version_to": (
            "polymarket-round27-role-gated-target-store-v2"
        ),
    }
    expected_discovery = {
        "campaign_contract_gate_was_not_enforced_at_target_store_boundary": True,
        "official_outcomes_accessed": False,
        "single_role_target_access_was_possible_before_complete_campaign_admission": (
            True
        ),
        "stage1_feature_rows_accessed_or_materialized": False,
    }
    expected_predecessor_sources = {
        "src/simple_ai_trading/polymarket_round27_target_store.py": (
            "8d8b3bca60d89dba108d3ebd379bddf465a481bdd36418587a1b43a78b8e088e"
        ),
        "tools/collect_polymarket_round27_targets.py": (
            "3591a9b483c7ed0b658e5825152d19e5a43e25e8d15fb2d77e20772a780b46fc"
        ),
    }
    expected_research = [
        {
            "purpose": "frozen_campaign_gate",
            "relative_path": (
                "docs/model-research/polymarket/"
                "round-027-stage1-campaign-contract-v1.json"
            ),
        },
        {
            "purpose": "frozen_model_population_gate",
            "relative_path": (
                "docs/model-research/polymarket/"
                "round-027-stage1-model-contract-v1.json"
            ),
        },
    ]
    payload = dict(value)
    claimed = _sha256(payload.pop("amendment_sha256", ""))
    created_at_ms = payload.get("created_at_ms")
    if (
        set(payload)
        != {
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
            "source_ledger",
            "status",
            "superseded_source_text_sha256",
        }
        or claimed != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != "polymarket-round27-campaign-admission-gate-correction-amendment-v16"
        or payload.get("status")
        != "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
        or type(created_at_ms) is not int
        or not _FIRST_CAPTURE_START_MS < int(created_at_ms) < _FIRST_CAPTURE_END_MS
        or payload.get("base_model_contract_sha256") != _BASE_MODEL_CONTRACT_SHA256
        or payload.get("campaign_contract_sha256") != _CAMPAIGN_CONTRACT_SHA256
        or payload.get("predecessor_amendment_sha256")
        != _V15_PREDECESSOR_AMENDMENT_SHA256
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
    _validate_source_ledger(
        root,
        payload.get("source_ledger"),
        expected_path=_EFFECTIVE_SOURCE_LEDGER_RELATIVE_PATH,
        expected_sha256=_EFFECTIVE_SOURCE_LEDGER_SHA256,
        expected_schema_version="polymarket-round27-effective-source-ledger-v7",
        predecessor_sha256=_V6_SOURCE_LEDGER_SHA256,
        verify_current_files=True,
    )
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
    _validate_v8_predecessor(
        _load_strict(root / _V8_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_v9_predecessor(
        _load_strict(root / _V9_PREDECESSOR_AMENDMENT_RELATIVE_PATH)
    )
    _validate_v10_predecessor(
        _load_strict(root / _V10_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    _validate_v11_predecessor(
        _load_strict(root / _V11_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    _validate_v12_predecessor(
        _load_strict(root / _V12_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    _validate_v13_predecessor(
        _load_strict(root / _V13_PREDECESSOR_AMENDMENT_RELATIVE_PATH),
        repository=root,
    )
    return validate_round27_model_amendment(
        _load_strict(selected),
        repository=root,
    )


__all__ = [
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_RELATIVE_PATH",
    "POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256",
    "load_round27_model_amendment",
    "validate_round27_model_amendment",
]
