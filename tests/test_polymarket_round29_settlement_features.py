from __future__ import annotations

from dataclasses import replace
import math

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
    POLYMARKET_ROUND28_FEATURE_NAMES,
    Round28BookTickerOverlayRow,
    Round28FeatureRow,
)
from simple_ai_trading.polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_BASE_FEATURE_NAMES,
    POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES,
    POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
    Round29FeatureRow,
    Round29SettlementOverlayRow,
)


def _base_row(
    *,
    margin: float = 0.004,
    variance_rate: float = 0.000004,
    path_efficiency: float = 0.75,
    remaining_seconds: float = 100.0,
    elapsed_fraction: float = 2.0 / 3.0,
    source_chain_sha256: str = "b" * 64,
) -> Round27FeatureRow:
    values = [0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    updates = {
        "phase.elapsed_fraction": elapsed_fraction,
        "phase.remaining_seconds": remaining_seconds,
        "twap.log_distance_from_open": margin,
        "twap.variance_rate_per_second": variance_rate,
        "twap.path_efficiency": path_efficiency,
    }
    for name, value in updates.items():
        values[POLYMARKET_ROUND27_FEATURE_NAMES.index(name)] = value
    return Round27FeatureRow.create(
        run_id="round29-test",
        condition_id="0x" + "a" * 64,
        event_start_ms=1_800_000,
        decision_time_ms=2_000_000,
        market_prior_probability=0.6,
        values=values,
        maximum_receipt_wall_ms=1_999_999,
        source_chain_sha256=source_chain_sha256,
    )


def _round28_row(base: Round27FeatureRow) -> Round28FeatureRow:
    decision = base.decision_time_ms
    feature = Round21OptionalBinanceFeatures(
        decision_time_ms=decision,
        spot_values=(0.1,) * len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES),
        usdm_values=(0.2,) * len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES),
        spot_available=True,
        usdm_available=True,
        spot_source_chain_sha256="c" * 64,
        usdm_source_chain_sha256="d" * 64,
        spot_maximum_receipt_ms=decision - 2,
        usdm_maximum_receipt_ms=decision - 1,
    )
    return Round28FeatureRow.create(base, Round28BookTickerOverlayRow.create(feature))


def test_round29_settlement_interactions_are_exact_and_target_blind() -> None:
    base = _base_row()
    overlay = Round29SettlementOverlayRow.create(base)

    assert overlay.values == pytest.approx(
        (
            1.0,
            0.0004,
            0.00004,
            math.asinh(0.004 / math.sqrt(0.000004 * 100.0)),
            0.75,
            0.004 * (2.0 / 3.0),
        )
    )
    assert overlay.target_accessed is False
    assert overlay.trading_authority is False
    assert len(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES) == 6


def test_round29_zero_variance_and_tie_remain_finite() -> None:
    zero_variance = Round29SettlementOverlayRow.create(
        _base_row(margin=-0.001, variance_rate=0.0)
    )
    tie = Round29SettlementOverlayRow.create(
        _base_row(margin=0.0, variance_rate=0.0, path_efficiency=0.0)
    )

    assert all(math.isfinite(value) for value in zero_variance.values)
    assert zero_variance.values[0] == -1.0
    assert zero_variance.values[3] < 0.0
    assert tie.values == pytest.approx((0.0,) * 6)


def test_round29_composes_matched_base_and_bbo_views_without_duplication() -> None:
    base = _base_row()
    overlay = Round29SettlementOverlayRow.create(base)
    base_view = Round29FeatureRow.from_round27(base, overlay)
    round28 = _round28_row(base)
    combined_view = Round29FeatureRow.from_round28(round28, overlay)

    assert len(base_view.values) == len(POLYMARKET_ROUND29_BASE_FEATURE_NAMES)
    assert base_view.values[: len(base.values)] == base.values
    assert base_view.values[-6:] == overlay.values
    assert len(combined_view.values) == len(POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES)
    assert len(round28.values) == len(POLYMARKET_ROUND28_FEATURE_NAMES)
    assert combined_view.values[: len(round28.values)] == round28.values
    assert combined_view.values[-6:] == overlay.values
    assert combined_view.base_row_sha256 == base.row_sha256
    assert combined_view.bbo_row_sha256 == round28.row_sha256


def test_round29_rejects_source_field_and_provenance_tampering() -> None:
    with pytest.raises(ValueError, match="source fields"):
        Round29SettlementOverlayRow.create(_base_row(variance_rate=-0.1))

    base = _base_row()
    overlay = Round29SettlementOverlayRow.create(base)
    with pytest.raises(ValueError, match="overlay row differs"):
        replace(overlay, values=(*overlay.values[:-1], 999.0)).validated()
    other_base = _base_row(source_chain_sha256="e" * 64)
    with pytest.raises(ValueError, match="identities differ"):
        Round29FeatureRow.from_round27(other_base, overlay)
    valid = Round29FeatureRow.from_round27(base, overlay)
    with pytest.raises(ValueError, match="feature view differs"):
        replace(valid, feature_view="unknown").validated()  # type: ignore[arg-type]
