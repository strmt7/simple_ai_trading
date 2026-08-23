from __future__ import annotations

import hashlib

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
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
    Round28BookTickerOverlayRow,
    Round28FeatureRow,
)
from simple_ai_trading.polymarket_round29_model import (
    Round29ModelSample,
    Round29Partition,
    build_round29_model_samples,
    fit_round29_l2_offset,
    round29_matched_ablation_report,
    round29_model_from_payload,
)
from simple_ai_trading.polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
    Round29FeatureRow,
    Round29SettlementOverlayRow,
)


def _sample(index: int, *, role: str, target: int) -> Round29ModelSample:
    diagnostic_base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    settlement = (1.0 if target else -1.0,) + (0.0,) * (
        len(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES) - 1
    )
    bbo = (0.0,) * len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
    primary_base = (*diagnostic_base, *bbo)
    event_start_ms = 1_800_000 + index * 300_000
    digest = hashlib.sha256(str(index).encode("ascii")).hexdigest()
    return Round29ModelSample(
        slot_id="stage1-a",
        role=role,
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=0.5,
        diagnostic_base_values=diagnostic_base,
        diagnostic_augmented_values=(*diagnostic_base, *settlement),
        primary_base_values=primary_base,
        primary_augmented_values=(*primary_base, *settlement),
        target_up=target,
        condition_weight=1.0,
        diagnostic_feature_row_sha256=digest,
        primary_feature_row_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
    ).validated()


def _feature_pair(
    decision_time_ms: int,
) -> tuple[Round29FeatureRow, Round29FeatureRow]:
    values = [0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("phase.elapsed_fraction")] = 0.5
    values[POLYMARKET_ROUND27_FEATURE_NAMES.index("phase.remaining_seconds")] = 150.0
    base = Round27FeatureRow.create(
        run_id="round29-model-test",
        condition_id="0x" + "a" * 64,
        event_start_ms=1_800_000,
        decision_time_ms=decision_time_ms,
        market_prior_probability=0.6,
        values=values,
        maximum_receipt_wall_ms=decision_time_ms - 3,
        source_chain_sha256="b" * 64,
    )
    binance = Round21OptionalBinanceFeatures(
        decision_time_ms=decision_time_ms,
        spot_values=(0.1,) * len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES),
        usdm_values=(0.2,) * len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES),
        spot_available=True,
        usdm_available=True,
        spot_source_chain_sha256="c" * 64,
        usdm_source_chain_sha256="d" * 64,
        spot_maximum_receipt_ms=decision_time_ms - 2,
        usdm_maximum_receipt_ms=decision_time_ms - 1,
    )
    round28 = Round28FeatureRow.create(
        base,
        Round28BookTickerOverlayRow.create(binance),
    )
    settlement = Round29SettlementOverlayRow.create(base)
    return (
        Round29FeatureRow.from_round27(base, settlement),
        Round29FeatureRow.from_round28(round28, settlement),
    )


def test_round29_builder_binds_both_views_and_condition_weights() -> None:
    first = _feature_pair(1_830_000)
    second = _feature_pair(1_831_000)

    samples = build_round29_model_samples(
        rows_by_slot={"stage1-a": (first, second)},
        outcomes_up={first[0].condition_id: 1},
        role_intervals=(
            {
                "role": "train",
                "slot_id": "stage1-a",
                "start_ms": 1_800_000,
                "end_ms": 2_100_000,
            },
        ),
    )

    assert [sample.condition_weight for sample in samples] == [0.5, 0.5]
    assert all(
        sample.diagnostic_augmented_values[-6:] == sample.primary_augmented_values[-6:]
        for sample in samples
    )
    with pytest.raises(ValueError, match="matched feature pair differs"):
        build_round29_model_samples(
            rows_by_slot={"stage1-a": ((first[0], _feature_pair(1_832_000)[1]),)},
            outcomes_up={first[0].condition_id: 1},
            role_intervals=(
                {
                    "role": "train",
                    "slot_id": "stage1-a",
                    "start_ms": 1_800_000,
                    "end_ms": 2_100_000,
                },
            ),
        )


@pytest.mark.parametrize("pair_name", ["diagnostic", "primary"])
def test_round29_synthetic_settlement_signal_improves_pair(
    pair_name: str,
) -> None:
    samples = tuple(
        _sample(index, role="train" if index < 40 else "selection", target=index % 2)
        for index in range(60)
    )
    train = Round29Partition.from_samples(samples, role="train")
    selection = Round29Partition.from_samples(samples, role="selection")
    views = {
        "diagnostic": ("round27_base", "round29_settlement_augmented"),
        "primary": (
            "round28_bbo_augmented",
            "round29_bbo_settlement_augmented",
        ),
    }[pair_name]
    base = fit_round29_l2_offset(train, feature_view=views[0], penalty=0.1)
    augmented = fit_round29_l2_offset(train, feature_view=views[1], penalty=0.1)

    report = round29_matched_ablation_report(
        selection,
        pair_name=pair_name,  # type: ignore[arg-type]
        base_model=base,
        augmented_model=augmented,
    )

    assert report["base_metrics"]["roc_auc"] == pytest.approx(0.5)
    assert report["augmented_metrics"]["roc_auc"] == pytest.approx(1.0)
    assert report["augmented_minus_base"]["log_loss"] < 0.0
    assert report["promotion_controlling_pair"] is (pair_name == "primary")
    assert report["edge_claim"] is False


def test_round29_l2_persistence_and_view_identity_fail_closed() -> None:
    samples = tuple(
        _sample(index, role="train", target=index % 2) for index in range(12)
    )
    partition = Round29Partition.from_samples(samples, role="train")
    model = fit_round29_l2_offset(
        partition,
        feature_view="round29_bbo_settlement_augmented",
        penalty=1.0,
        correction_scale=0.5,
    )
    payload = model.asdict()

    assert round29_model_from_payload(payload) == model
    tampered = {**payload, "coefficients": [*payload["coefficients"]]}
    tampered["coefficients"][0] += 0.01
    with pytest.raises(ValueError, match="L2 model differs"):
        round29_model_from_payload(tampered)
    with pytest.raises(ValueError, match="matched model views differ"):
        round29_matched_ablation_report(
            partition,
            pair_name="primary",
            base_model=model,
            augmented_model=model,
        )
