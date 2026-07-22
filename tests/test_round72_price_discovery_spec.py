from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.price_discovery_spec import (
    FEATURE_LAYERS,
    HORIZONS_SECONDS,
    PRIMARY_LOSS_METRICS,
    ROUND72_IMPLEMENTATION_V1_SHA256,
    ROUND72_IMPLEMENTATION_V2_SHA256,
    build_round72_implementation_spec,
    layer_feature_names,
    validate_layer_prefixes,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "docs/model-research/action-value/round-072-price-discovery-implementation.json"


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_round72_implementation_artifact_is_exactly_reproducible() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    canonical = dict(artifact)
    observed_hash = canonical.pop("implementation_sha256")

    assert observed_hash == _sha256(canonical)
    assert artifact == build_round72_implementation_spec(
        design_sha256=artifact["design_sha256"],
        inventory_sha256=artifact["inventory_sha256"],
        inventory_file_sha256=artifact["inventory_file_sha256"],
        frozen_at_utc=artifact["frozen_at_utc"],
    )


def test_round72_feature_layers_are_exact_unique_nested_prefixes() -> None:
    names = layer_feature_names("cross_asset")
    widths = validate_layer_prefixes(names)

    assert FEATURE_LAYERS == ("perpetual_only", "spot_perpetual", "cross_asset")
    assert widths == (123, 287, 336)
    assert len(names) == len(set(names)) == 336
    for layer, width in zip(FEATURE_LAYERS, widths, strict=True):
        assert names[:width] == layer_feature_names(layer)


def test_round72_freeze_excludes_holdout_profit_and_leverage_authority() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    split = artifact["split_contract"]

    assert HORIZONS_SECONDS == (30, 60, 300)
    assert split["folds"][-1]["test_months"][-1] == "2026-03"
    assert split["terminal_holdout_months_never_read"] == [
        "2026-04",
        "2026-05",
        "2026-06",
    ]
    assert artifact["profitability_claim"] is False
    assert artifact["execution_or_fill_claim"] is False
    assert artifact["trading_authority"] is False
    assert artifact["leverage_authority"] is False
    assert artifact["freeze_evidence"]["post_result_changes_permitted"] is False


def test_round72_pre_result_amendment_removes_evaluation_and_session_ambiguity() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    evaluation = artifact["evaluation_contract"]

    assert artifact["amendment"] == {
        "predecessor_implementation_sha256": ROUND72_IMPLEMENTATION_V2_SHA256,
        "original_implementation_sha256": ROUND72_IMPLEMENTATION_V1_SHA256,
        "reason": "final pre-result clarification of metric definitions, tied scores, undefined day metrics, and exact gate inequalities",
        "round72_price_or_return_result_available_before_amendment": False,
        "round72_model_result_available_before_amendment": False,
        "data_feature_target_split_or_model_parameter_changed": False,
    }
    assert evaluation["primary_loss_metrics"] == {
        head: list(metrics) for head, metrics in PRIMARY_LOSS_METRICS.items()
    }
    assert evaluation["fdr_family_cardinality"] == 36
    assert evaluation["minimum_finite_days_per_bootstrap_metric"] == 10
    assert evaluation["metric_definitions"]["MCC"].startswith(
        "standard confusion-matrix"
    )
    assert "continuous" in artifact["anchor_contract"]["market_session_semantics"]
    assert "never a formal market close" in artifact["anchor_contract"][
        "market_session_semantics"
    ]
    assert "ETF" in artifact["anchor_contract"]["listed_product_session_semantics"]
