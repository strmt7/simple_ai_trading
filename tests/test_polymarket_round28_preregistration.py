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
    claimed = value.pop("preregistration_sha256")

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
    assert value["source_text_sha256"] == {
        relative: _text_sha256(ROOT / relative)
        for relative in value["source_text_sha256"]
    }


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
