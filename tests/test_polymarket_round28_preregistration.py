from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND28_FEATURE_NAMES,
    POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
)
IMPLEMENTATION_AMENDMENT = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-selection-implementation-amendment-v1.json"
)
CONTRACT_BINDING_CORRECTION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v1.json"
)
SUCCESSOR_CONTRACT_BINDING_CORRECTIONS = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v2.json",
    ROOT / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v3.json",
)
STATIC_ANALYSIS_REMEDIATION = (
    ROOT / "docs/model-research/polymarket/"
    "round-028-static-analysis-remediation-amendment-v2.json"
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


def test_round28_preregistration_is_hash_and_source_bound_before_targets() -> None:
    value = json.loads(PREREGISTRATION.read_text(encoding="ascii"))
    amendment = json.loads(IMPLEMENTATION_AMENDMENT.read_text(encoding="ascii"))
    remediation = json.loads(STATIC_ANALYSIS_REMEDIATION.read_text(encoding="ascii"))
    claimed = value.pop("preregistration_sha256")
    remediation_claimed = remediation.pop("amendment_sha256")

    assert claimed == _canonical_sha256(value)
    assert value["status"] == (
        "frozen_after_capture_start_before_stage1_feature_or_outcome_access"
    )
    assert value["knowledge_at_freeze"]["official_outcomes_accessed"] is False
    assert (
        value["knowledge_at_freeze"][
            "round27_stage1_feature_rows_accessed_or_materialized"
        ]
        is False
    )
    assert value["authority"] == {
        "credentials_used": False,
        "edge_claim": False,
        "execution_connected": False,
        "live_trading_authority": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }
    superseded = amendment["superseded_source_text_sha256"]
    remediated = remediation["source_text_sha256"]
    assert remediation_claimed == _canonical_sha256(remediation)
    assert remediation["base_preregistration_sha256"] == claimed
    assert not set(superseded) & set(remediated)
    assert set(superseded) <= set(value["source_text_sha256"])
    assert all(
        value["source_text_sha256"][relative] == expected
        for relative, expected in superseded.items()
    )
    for relative, expected in value["source_text_sha256"].items():
        if relative in superseded:
            continue
        if relative in remediated:
            assert remediated[relative] == {
                "corrected": _text_sha256(ROOT / relative),
                "frozen": expected,
            }
        else:
            assert expected == _text_sha256(ROOT / relative)


def test_round28_preregistration_binds_exact_incremental_feature_contract() -> None:
    value = json.loads(PREREGISTRATION.read_text(encoding="ascii"))
    feature = value["feature_contract"]

    assert feature["base_feature_count"] == len(POLYMARKET_ROUND27_FEATURE_NAMES)
    assert feature["incremental_bbo_feature_count"] == len(
        POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES
    )
    assert feature["augmented_feature_count"] == len(POLYMARKET_ROUND28_FEATURE_NAMES)
    assert (
        feature["round27_base_feature_names_sha256"]
        == POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
    )
    assert (
        feature["round28_bbo_feature_names_sha256"]
        == POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256
    )
    assert (
        feature["round28_augmented_feature_names_sha256"]
        == POLYMARKET_ROUND28_FEATURE_NAMES_SHA256
    )
    assert value["matched_ablation"]["view_under_test"] == (
        "the 96 incremental BBO fields only"
    )
    assert value["promotion_gates"]["accuracy"].endswith("neither can promote alone.")


def test_round28_selection_amendment_is_hash_and_superseded_source_bound() -> None:
    value = json.loads(IMPLEMENTATION_AMENDMENT.read_text(encoding="ascii"))
    correction = json.loads(CONTRACT_BINDING_CORRECTION.read_text(encoding="ascii"))
    successor_replacements = [
        json.loads(path.read_text(encoding="ascii"))["superseded_source_text_sha256"]
        for path in SUCCESSOR_CONTRACT_BINDING_CORRECTIONS
    ]
    claimed = value.pop("amendment_sha256")

    assert claimed == _canonical_sha256(value)
    assert (
        value["base_preregistration_sha256"]
        == json.loads(PREREGISTRATION.read_text(encoding="ascii"))[
            "preregistration_sha256"
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
    for relative, expected in value["source_text_sha256"].items():
        if relative in replacements:
            assert replacements[relative]["frozen"] == expected
            expected = replacements[relative]["corrected"]
        for successor in successor_replacements:
            if relative not in successor:
                continue
            assert successor[relative]["frozen"] == expected
            expected = successor[relative]["corrected"]
        assert expected == _text_sha256(ROOT / relative)
