from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round29_model import (
    Round29ModelSample,
    Round29Partition,
    fit_round29_l2_offset,
)
from simple_ai_trading.polymarket_round29_selection import (
    load_round29_selected_pair,
    paired_round29_condition_bootstrap,
    round29_chronological_condition_folds,
    round29_pair_selection_report,
    run_round29_matched_selection,
    scale_round29_probability_model,
    select_round29_correction_scale,
    select_round29_l2_penalty,
)
from simple_ai_trading.polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _sample(index: int, *, role: str) -> Round29ModelSample:
    target = index % 2
    diagnostic_base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    settlement = (1.0 if target else -1.0,) + (0.0,) * (
        len(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES) - 1
    )
    bbo = (0.0,) * len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
    primary_base = (*diagnostic_base, *bbo)
    event_start_ms = 1_800_000 + index * 1_200_000
    digest = hashlib.sha256(str(index).encode("ascii")).hexdigest()
    return Round29ModelSample(
        slot_id="stage1-a",
        role=role,
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=0.6,
        diagnostic_base_values=diagnostic_base,
        diagnostic_augmented_values=(*diagnostic_base, *settlement),
        primary_base_values=primary_base,
        primary_augmented_values=(*primary_base, *settlement),
        target_up=target,
        condition_weight=1.0,
        diagnostic_feature_row_sha256=digest,
        primary_feature_row_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
    ).validated()


def _partition(count: int, *, role: str, offset: int = 0) -> Round29Partition:
    return Round29Partition.from_samples(
        tuple(_sample(index, role=role) for index in range(offset, offset + count)),
        role=role,
    )


def test_round29_training_controls_are_causal_and_grid_bound() -> None:
    partition = _partition(30, role="train")

    folds = round29_chronological_condition_folds(partition, fold_count=2)
    penalty, penalty_scores = select_round29_l2_penalty(
        partition,
        feature_view="round27_base",
    )
    model = fit_round29_l2_offset(
        partition,
        feature_view="round29_settlement_augmented",
        penalty=1.0,
    )
    selected_scale, scale_scores = select_round29_correction_scale(model, partition)
    zero_scale = scale_round29_probability_model(model, 0.0)

    assert len(folds) == 2
    assert all(
        set(train.conditions).isdisjoint(set(validation.conditions))
        for train, validation in folds
    )
    assert penalty in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert set(penalty_scores) == {"0.01", "0.1", "1.0", "10.0", "100.0"}
    assert selected_scale == 1.0
    assert set(scale_scores) == {"0.0", "0.25", "0.5", "0.75", "1.0"}
    assert zero_scale.predict(
        partition.diagnostic_augmented_features,
        partition.offsets,
    ) == pytest.approx(np.full(partition.targets.shape, 0.6))


def test_round29_primary_pair_passes_only_with_matched_incremental_signal() -> None:
    partition = _partition(40, role="selection")
    base = fit_round29_l2_offset(
        partition,
        feature_view="round28_bbo_augmented",
        penalty=1.0,
    )
    augmented = fit_round29_l2_offset(
        partition,
        feature_view="round29_bbo_settlement_augmented",
        penalty=0.01,
    )
    report = round29_pair_selection_report(
        partition,
        pair_name="primary",
        base_model=base,
        augmented_model=augmented,
        prediction_evaluation={
            "bootstrap_draws": 1_000,
            "balanced_accuracy_floor": 0.51,
            "calibration_ece_maximum_degradation": 0.01,
        },
        training_detail={"fixture": "mechanics_only"},
    )
    bootstrap = paired_round29_condition_bootstrap(
        partition,
        np.full(partition.targets.shape, 0.5),
        np.where(partition.targets == 1.0, 0.8, 0.2),
        draws=1_000,
    )

    assert report["probability_gate_passed"] is True
    assert report["promotion_controlling_pair"] is True
    assert all(report["gate_checks"].values())
    assert bootstrap["log_loss"]["ci95_upper"] < 0.0
    assert bootstrap["brier_score"]["ci95_upper"] < 0.0
    assert report["edge_claim"] is False


def test_round29_selection_persists_only_primary_pair_and_rejects_tampering() -> None:
    samples = tuple(
        [*(_sample(index, role="train") for index in range(75))]
        + [*(_sample(index, role="calibration") for index in range(75, 100))]
        + [*(_sample(index, role="selection") for index in range(100, 190))]
    )
    contract = load_round27_model_contract(ROOT)
    preregistration = json.loads(
        (
            ROOT / "docs/model-research/polymarket/"
            "round-029-settlement-state-matched-ablation-preregistration-v1.json"
        ).read_text(encoding="ascii")
    )
    claim, expected = run_round29_matched_selection(
        samples=samples,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest_sha256="a" * 64,
        claim_writer=lambda value: str(value["claim_sha256"]),
    )

    selected = load_round29_selected_pair(
        claim,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest_sha256="a" * 64,
    )

    assert selected == expected
    assert selected is not None
    assert claim["status"] == "primary_probability_candidate_selected"
    assert claim["diagnostic_pair_can_promote"] is False
    tampered = json.loads(json.dumps(claim))
    primary = next(
        report
        for report in tampered["candidate_pairs"]
        if report["pair_name"] == "primary"
    )
    primary["matched_ablation"]["augmented_model"]["coefficients"][0] += 0.1
    primary["matched_ablation"].pop("report_sha256")
    primary["matched_ablation"]["report_sha256"] = _canonical_sha256(
        primary["matched_ablation"]
    )
    primary.pop("pair_report_sha256")
    primary["pair_report_sha256"] = _canonical_sha256(primary)
    tampered.pop("claim_sha256")
    tampered["claim_sha256"] = _canonical_sha256(tampered)
    with pytest.raises(ValueError, match="Round 29 L2 model differs"):
        load_round29_selected_pair(
            tampered,
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="selection claim differs"):
        load_round29_selected_pair(
            claim,
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest_sha256="b" * 64,
        )
