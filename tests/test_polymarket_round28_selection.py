from __future__ import annotations

from dataclasses import replace
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
from simple_ai_trading.polymarket_round28_model import (
    Round28ModelSample,
    Round28Partition,
    fit_round28_l2_offset,
)
from simple_ai_trading.polymarket_round28_selection import (
    load_round28_selected_pair,
    paired_round28_condition_bootstrap,
    round28_pair_selection_report,
    round28_walk_forward_condition_folds,
    run_round28_matched_selection,
    scale_round28_probability_model,
    select_round28_correction_scale,
    select_round28_l2_penalty,
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


def _sample(
    index: int,
    *,
    role: str,
    target: int,
    market_prior_probability: float = 0.5,
) -> Round28ModelSample:
    base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    bbo = (1.0 if target else -1.0,) + (0.0,) * (
        len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES) - 1
    )
    event_start_ms = 1_800_000 + index * 1_200_000
    return Round28ModelSample(
        slot_id="stage1-a",
        role=role,
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=market_prior_probability,
        base_values=base,
        augmented_values=(*base, *bbo),
        target_up=target,
        condition_weight=1.0,
        feature_row_sha256=hashlib.sha256(str(index).encode("ascii")).hexdigest(),
    ).validated()


def _partition(
    count: int,
    *,
    role: str,
    market_prior_probability: float = 0.5,
) -> Round28Partition:
    return Round28Partition.from_samples(
        tuple(
            _sample(
                index,
                role=role,
                target=index % 2,
                market_prior_probability=market_prior_probability,
            )
            for index in range(count)
        ),
        role=role,
    )


def _artifact(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="ascii"))


def test_round28_scale_rebinding_preserves_fit_and_zero_scale_returns_prior() -> None:
    partition = _partition(20, role="train")
    model = fit_round28_l2_offset(
        partition,
        feature_view="round28_bbo_augmented",
        penalty=1.0,
    )

    scaled = scale_round28_probability_model(model, 0.0)
    selected_scale, scores = select_round28_correction_scale(model, partition)

    assert scaled.coefficients == model.coefficients
    assert scaled.mean == model.mean
    assert scaled.scale == model.scale
    assert scaled.model_sha256 != model.model_sha256
    assert scaled.predict(
        partition.augmented_features, partition.offsets
    ) == pytest.approx(np.full(partition.targets.shape, 0.5))
    assert selected_scale == 1.0
    assert set(scores) == {"0.0", "0.25", "0.5", "0.75", "1.0"}


def test_round28_walk_forward_folds_are_condition_isolated_and_embargoed() -> None:
    partition = _partition(30, role="train")

    folds = round28_walk_forward_condition_folds(partition, fold_count=2)

    assert len(folds) == 2
    for train, validation in folds:
        assert set(train.conditions).isdisjoint(set(validation.conditions))
        assert max(
            sample.event_start_ms + 300_000 for sample in train.samples
        ) + 600_000 <= min(sample.event_start_ms for sample in validation.samples)


def test_round28_l2_penalty_search_uses_declared_grid() -> None:
    selected, scores = select_round28_l2_penalty(
        _partition(24, role="train"),
        feature_view="round27_base",
        fold_count=2,
    )

    assert selected in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert set(scores) == {"0.01", "0.1", "1.0", "10.0", "100.0"}
    assert all(np.isfinite(value) for value in scores.values())


def test_round28_condition_bootstrap_detects_uniform_matched_improvement() -> None:
    partition = _partition(40, role="selection")
    baseline = np.full(partition.targets.shape, 0.5)
    candidate = np.where(partition.targets == 1.0, 0.8, 0.2)

    result = paired_round28_condition_bootstrap(
        partition,
        baseline,
        candidate,
        draws=1_000,
    )

    assert result["condition_count"] == 40
    assert result["mean_candidate_minus_baseline_log_loss"] < 0.0
    assert result["log_loss"]["ci95_upper"] < 0.0
    assert result["brier_score"]["ci95_upper"] < 0.0
    claimed = result.pop("bootstrap_sha256")
    assert claimed == _canonical_sha256(result)


def test_round28_pair_report_passes_only_matched_incremental_signal() -> None:
    partition = _partition(
        40,
        role="selection",
        market_prior_probability=0.6,
    )
    base = fit_round28_l2_offset(
        partition,
        feature_view="round27_base",
        penalty=1.0,
    )
    augmented = fit_round28_l2_offset(
        partition,
        feature_view="round28_bbo_augmented",
        penalty=0.01,
    )

    report = round28_pair_selection_report(
        partition,
        base_model=base,
        augmented_model=augmented,
        prediction_evaluation={
            "bootstrap_draws": 1_000,
            "balanced_accuracy_floor": 0.51,
            "calibration_ece_maximum_degradation": 0.01,
        },
        training_detail={"fixture": "mechanics_only"},
    )

    assert report["probability_gate_passed"] is True
    assert all(report["gate_checks"].values())
    assert report["edge_claim"] is False
    assert report["profitability_claim"] is False
    with pytest.raises(ValueError, match="model families"):
        round28_pair_selection_report(
            partition,
            base_model=replace(base, model_name="not-the-same-family"),
            augmented_model=augmented,
            prediction_evaluation={
                "bootstrap_draws": 1_000,
                "balanced_accuracy_floor": 0.51,
                "calibration_ece_maximum_degradation": 0.01,
            },
            training_detail={},
        )


def test_round28_selected_pair_loader_rejects_model_tampering() -> None:
    samples = tuple(
        [
            _sample(
                index,
                role="train",
                target=index % 2,
                market_prior_probability=0.6,
            )
            for index in range(75)
        ]
        + [
            _sample(
                index,
                role="calibration",
                target=index % 2,
                market_prior_probability=0.6,
            )
            for index in range(75, 100)
        ]
        + [
            _sample(
                index,
                role="selection",
                target=index % 2,
                market_prior_probability=0.6,
            )
            for index in range(100, 190)
        ]
    )
    contract = load_round27_model_contract(ROOT)
    preregistration = _artifact(
        "docs/model-research/polymarket/"
        "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
    )
    claim, expected = run_round28_matched_selection(
        samples=samples,
        contract=contract,
        preregistration=preregistration,
        claim_writer=lambda value: str(value["claim_sha256"]),
        compute_backend="cpu",
    )

    selected = load_round28_selected_pair(
        claim,
        contract=contract,
        preregistration=preregistration,
    )

    assert selected == expected
    assert selected is not None
    assert claim["round27_model_implementation_amendment_sha256"] == contract[
        "model_implementation_amendment_sha256"
    ]
    tampered = json.loads(json.dumps(claim))
    selected_report = next(
        report
        for report in tampered["candidate_pairs"]
        if report["model_family"] == tampered["selected_model_family"]
    )
    selected_report["matched_ablation"]["augmented_model"]["coefficients"][0] += 0.1
    selected_report["matched_ablation"].pop("report_sha256")
    selected_report["matched_ablation"]["report_sha256"] = _canonical_sha256(
        selected_report["matched_ablation"]
    )
    selected_report.pop("pair_report_sha256")
    selected_report["pair_report_sha256"] = _canonical_sha256(selected_report)
    tampered.pop("claim_sha256")
    tampered["claim_sha256"] = _canonical_sha256(tampered)
    with pytest.raises(ValueError, match="Round 28 L2 model differs"):
        load_round28_selected_pair(
            tampered,
            contract=contract,
            preregistration=preregistration,
        )
