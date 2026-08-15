from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-operator-implementation-amendment-v1.json"
)
PREREGISTRATION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
)
SELECTION_AMENDMENT = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-selection-implementation-amendment-v1.json"
)
ECONOMIC_AMENDMENT = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-economic-implementation-amendment-v1.json"
)
CONTRACT_BINDING_CORRECTION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v1.json"
)
CURRENT_CONTRACT_BINDING_CORRECTION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v2.json"
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


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()


def test_round28_operator_amendment_is_hash_and_superseded_source_bound() -> None:
    value = json.loads(AMENDMENT.read_text(encoding="ascii"))
    correction = json.loads(CONTRACT_BINDING_CORRECTION.read_text(encoding="ascii"))
    current = json.loads(
        CURRENT_CONTRACT_BINDING_CORRECTION.read_text(encoding="ascii")
    )
    claimed = value.pop("amendment_sha256")

    assert claimed == _canonical_sha256(value)
    assert (
        value["base_preregistration_sha256"]
        == json.loads(PREREGISTRATION.read_text(encoding="ascii"))[
            "preregistration_sha256"
        ]
    )
    assert (
        value["selection_implementation_amendment_sha256"]
        == json.loads(SELECTION_AMENDMENT.read_text(encoding="ascii"))[
            "amendment_sha256"
        ]
    )
    assert (
        value["economic_implementation_amendment_sha256"]
        == json.loads(ECONOMIC_AMENDMENT.read_text(encoding="ascii"))[
            "amendment_sha256"
        ]
    )
    assert value["status"] == "frozen_before_stage1_feature_or_outcome_access"
    assert value["knowledge_at_freeze"] == {
        "ai_assist_economic_metrics_computed": False,
        "model_fitted_on_stage1": False,
        "official_outcomes_accessed": False,
        "performance_metrics_computed": False,
        "round27_stage1_feature_rows_accessed_or_materialized": False,
        "sealed_partition_accessed": False,
        "selection_partition_accessed": False,
    }
    assert value["authority"] == {
        "credentials_used": False,
        "edge_claim": False,
        "execution_connected": False,
        "live_trading_authority": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }
    replacements = correction["superseded_source_text_sha256"]
    assert value["source_text_sha256"] == {
        relative: replacements[relative]["frozen"]
        for relative in value["source_text_sha256"]
    }
    current_replacements = current["superseded_source_text_sha256"]
    for relative in value["source_text_sha256"]:
        corrected = replacements[relative]["corrected"]
        if relative in current_replacements:
            assert current_replacements[relative] == {
                "frozen": corrected,
                "corrected": _text_sha256(ROOT / relative),
            }
        else:
            assert corrected == _text_sha256(ROOT / relative)
    assert value["test_scope"] == {
        "financial_result": False,
        "stage1_data_used": False,
        "synthetic_mechanics_only": True,
    }
