from __future__ import annotations

import hashlib
import json

import pytest

from simple_ai_trading.polymarket_round21_binance_features import (
    POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    Round21OptionalBinanceFeatures,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    Round28BookTickerOverlayRow,
    Round28FeatureRow,
)
from simple_ai_trading.polymarket_round29_source import (
    compose_round29_feature_pairs,
    validate_round29_source_report,
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


def _base(decision_time_ms: int, *, marker: float = 0.0) -> Round27FeatureRow:
    values = [0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("phase.elapsed_fraction")] = 0.5
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("phase.remaining_seconds")] = 150.0
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("twap.variance_rate_per_second")] = (
        0.000_001
    )
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("twap.path_efficiency")] = 0.5
    values[0] += marker
    return Round27FeatureRow.create(
        run_id="round29-source-test",
        condition_id="0x" + "1" * 64,
        event_start_ms=1_800_000,
        decision_time_ms=decision_time_ms,
        market_prior_probability=0.55,
        values=values,
        maximum_receipt_wall_ms=decision_time_ms - 1,
        source_chain_sha256="2" * 64,
    )


def _round28(base: Round27FeatureRow) -> Round28FeatureRow:
    decision = base.decision_time_ms
    features = Round21OptionalBinanceFeatures(
        decision_time_ms=decision,
        spot_values=(0.1,) * len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES),
        usdm_values=(0.2,) * len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES),
        spot_available=True,
        usdm_available=True,
        spot_source_chain_sha256="3" * 64,
        usdm_source_chain_sha256="4" * 64,
        spot_maximum_receipt_ms=decision - 2,
        usdm_maximum_receipt_ms=decision - 1,
    )
    return Round28FeatureRow.create(
        base,
        Round28BookTickerOverlayRow.create(features),
    )


def _report() -> dict[str, object]:
    body: dict[str, object] = {"schema_version": "round28-fixture"}
    body["report_sha256"] = _canonical_sha256(body)
    return body


def test_round29_source_composes_only_matched_bbo_population() -> None:
    rows = (_base(1_830_000), _base(1_831_000))

    pairs, report = compose_round29_feature_pairs(
        base_rows=rows,
        round28_rows=(_round28(rows[1]),),
        round28_overlay_report=_report(),
    )

    assert len(pairs) == 1
    assert pairs[0][0].decision_time_ms == 1_831_000
    assert pairs[0][1].decision_time_ms == 1_831_000
    assert report["source_base_row_count"] == 2
    assert report["matched_row_count"] == 1
    assert report["excluded_without_bbo_count"] == 1
    assert report["official_outcomes_accessed"] is False
    assert validate_round29_source_report(report) == report


def test_round29_source_rejects_base_identity_and_rehashed_report_drift() -> None:
    base = _base(1_830_000)
    other = _base(1_830_000, marker=0.1)
    with pytest.raises(ValueError, match="identities differ"):
        compose_round29_feature_pairs(
            base_rows=(other,),
            round28_rows=(_round28(base),),
            round28_overlay_report=_report(),
        )

    _pairs, report = compose_round29_feature_pairs(
        base_rows=(base,),
        round28_rows=(_round28(base),),
        round28_overlay_report=_report(),
    )
    tampered = dict(report)
    tampered["matched_row_count"] = 2
    tampered["report_sha256"] = _canonical_sha256(
        {key: value for key, value in tampered.items() if key != "report_sha256"}
    )
    with pytest.raises(ValueError, match="source report differs"):
        validate_round29_source_report(tampered)

    hash_tampered = dict(report)
    hash_tampered["matched_row_count"] = 2
    with pytest.raises(ValueError, match="source report hash differs"):
        validate_round29_source_report(hash_tampered)
