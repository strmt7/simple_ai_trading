from __future__ import annotations

from dataclasses import replace
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
    POLYMARKET_ROUND28_FEATURE_NAMES,
    Round28BookTickerOverlayRow,
    Round28FeatureRow,
)
from simple_ai_trading.polymarket_round28_model import (
    Round28ModelSample,
    Round28Partition,
    build_round28_model_samples,
    fit_round28_l2_offset,
    round28_matched_ablation_report,
    round28_model_from_payload,
    round28_probability_metrics,
)


def _sample(index: int, *, role: str, target: int) -> Round28ModelSample:
    base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    bbo = (1.0 if target else -1.0,) + (0.0,) * (
        len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES) - 1
    )
    event_start_ms = 1_800_000 + index * 300_000
    return Round28ModelSample(
        slot_id="stage1-a",
        role=role,
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=0.5,
        base_values=base,
        augmented_values=(*base, *bbo),
        target_up=target,
        condition_weight=1.0,
        feature_row_sha256=hashlib.sha256(str(index).encode("ascii")).hexdigest(),
    ).validated()


def _round28_row(decision_time_ms: int) -> Round28FeatureRow:
    base = Round27FeatureRow.create(
        run_id="round28-model-test",
        condition_id="0x" + "a" * 64,
        event_start_ms=1_800_000,
        decision_time_ms=decision_time_ms,
        market_prior_probability=0.6,
        values=(0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES),
        maximum_receipt_wall_ms=decision_time_ms - 3,
        source_chain_sha256="b" * 64,
    )
    feature = Round21OptionalBinanceFeatures(
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
    return Round28FeatureRow.create(base, Round28BookTickerOverlayRow.create(feature))


def test_round28_builder_preserves_condition_weight_on_matched_rows() -> None:
    rows = (_round28_row(1_830_000), _round28_row(1_831_000))
    samples = build_round28_model_samples(
        rows_by_slot={"stage1-a": rows},
        outcomes_up={rows[0].condition_id: 1},
        role_intervals=(
            {
                "role": "train",
                "slot_id": "stage1-a",
                "start_ms": 1_800_000,
                "end_ms": 2_100_000,
            },
        ),
    )

    assert len(samples) == 2
    assert [sample.condition_weight for sample in samples] == [0.5, 0.5]
    assert all(
        sample.augmented_values[: len(POLYMARKET_ROUND27_FEATURE_NAMES)]
        == sample.base_values
        for sample in samples
    )


def test_round28_synthetic_bbo_signal_improves_matched_probability_metrics() -> None:
    samples = tuple(
        _sample(index, role="train" if index < 40 else "selection", target=index % 2)
        for index in range(60)
    )
    train = Round28Partition.from_samples(samples, role="train")
    selection = Round28Partition.from_samples(samples, role="selection")
    base_model = fit_round28_l2_offset(
        train,
        feature_view="round27_base",
        penalty=0.1,
    )
    augmented_model = fit_round28_l2_offset(
        train,
        feature_view="round28_bbo_augmented",
        penalty=0.1,
    )

    report = round28_matched_ablation_report(
        selection,
        base_model=base_model,
        augmented_model=augmented_model,
    )

    assert report["row_count"] == 20
    assert report["base_metrics"]["roc_auc"] == pytest.approx(0.5)
    assert report["augmented_metrics"]["roc_auc"] == pytest.approx(1.0)
    assert report["augmented_minus_base"]["log_loss"] < 0.0
    assert report["augmented_minus_base"]["brier_score"] < 0.0
    assert report["same_rows_targets_weights_offsets"] is True
    assert report["after_cost_economic_evaluation_required"] is True
    assert report["edge_claim"] is False
    assert report["profitability_claim"] is False


def test_round28_metrics_reject_single_class_and_ablation_rejects_wrong_views() -> None:
    samples = tuple(_sample(index, role="selection", target=1) for index in range(4))
    partition = Round28Partition.from_samples(samples, role="selection")
    with pytest.raises(ValueError, match="class is absent"):
        round28_probability_metrics(partition, [0.6] * len(samples))

    balanced = tuple(
        _sample(index, role="train", target=index % 2) for index in range(10)
    )
    train = Round28Partition.from_samples(balanced, role="train")
    base = fit_round28_l2_offset(
        train,
        feature_view="round27_base",
        penalty=1.0,
    )
    augmented = fit_round28_l2_offset(
        train,
        feature_view="round28_bbo_augmented",
        penalty=1.0,
    )
    with pytest.raises(ValueError, match="model views"):
        round28_matched_ablation_report(
            train,
            base_model=replace(base, feature_view="round28_bbo_augmented"),
            augmented_model=augmented,
        )


def test_round28_calibration_bins_include_exact_decimal_boundaries() -> None:
    samples = tuple(
        replace(
            _sample(index, role="selection", target=index % 2),
            market_prior_probability=0.6,
        ).validated()
        for index in range(20)
    )
    partition = Round28Partition.from_samples(samples, role="selection")

    metrics = round28_probability_metrics(partition, [0.6] * len(samples))

    assert metrics.expected_calibration_error == pytest.approx(0.1)


def test_round28_model_sample_rejects_base_augmented_drift() -> None:
    sample = _sample(0, role="train", target=0)
    with pytest.raises(ValueError, match="model sample differs"):
        replace(
            sample,
            augmented_values=(1.0,) + sample.augmented_values[1:],
        ).validated()
    assert len(sample.augmented_values) == len(POLYMARKET_ROUND28_FEATURE_NAMES)


def test_round28_l2_persistence_round_trip_rejects_identity_drift() -> None:
    samples = tuple(
        _sample(index, role="train", target=index % 2) for index in range(12)
    )
    model = fit_round28_l2_offset(
        Round28Partition.from_samples(samples, role="train"),
        feature_view="round28_bbo_augmented",
        penalty=1.0,
        correction_scale=0.5,
    )
    payload = model.asdict()

    restored = round28_model_from_payload(payload)

    assert restored == model
    tampered = {**payload, "coefficients": [*payload["coefficients"]]}
    tampered["coefficients"][0] += 0.01
    with pytest.raises(ValueError, match="Round 28 L2 model differs"):
        round28_model_from_payload(tampered)
    with pytest.raises(ValueError, match="fields differ"):
        round28_model_from_payload({**payload, "unregistered": True})
