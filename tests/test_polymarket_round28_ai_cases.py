from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round28_ai_cases import (
    materialize_round28_ai_cases,
    round28_ai_case_panel_from_mapping,
    round28_ai_case_prompt,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
    POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION,
    Round28FeatureRow,
)
from test_polymarket_round27_economics import _population


_MODEL_SHA256 = "b" * 64
_SELECTION_SHA256 = "e" * 64


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


class _FixedAugmentedModel:
    model_name = "fixed-test-model"
    feature_view = "round28_bbo_augmented"

    def predict(self, features, offsets):
        del offsets
        assert features.shape[1] == len(POLYMARKET_ROUND28_FEATURE_NAMES)
        return np.full(features.shape[0], 0.80, dtype=np.float64)

    def asdict(self):
        return {
            "model_name": self.model_name,
            "feature_view": self.feature_view,
            "model_sha256": _MODEL_SHA256,
        }


def _rows(partition) -> tuple[Round28FeatureRow, ...]:
    rows: list[Round28FeatureRow] = []
    for sample in partition.samples:
        base_sha = hashlib.sha256(
            f"base-{sample.condition_id}".encode("ascii")
        ).hexdigest()
        overlay_sha = hashlib.sha256(
            f"overlay-{sample.condition_id}".encode("ascii")
        ).hexdigest()
        source_sha = _canonical_sha256(
            {
                "base_row_sha256": base_sha,
                "overlay_row_sha256": overlay_sha,
            }
        )
        payload = {
            "schema_version": POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION,
            "run_id": "round28-stage1-test",
            "condition_id": sample.condition_id,
            "event_start_ms": sample.event_start_ms,
            "decision_time_ms": sample.decision_time_ms,
            "market_prior_probability": sample.market_prior_probability,
            "values": (*sample.values, *(0.0 for _ in range(96))),
            "feature_names_sha256": POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
            "maximum_receipt_wall_ms": sample.decision_time_ms - 1,
            "base_row_sha256": base_sha,
            "overlay_row_sha256": overlay_sha,
            "source_chain_sha256": source_sha,
            "target_accessed": False,
            "trading_authority": False,
        }
        rows.append(
            Round28FeatureRow(
                **payload,
                row_sha256=_canonical_sha256(payload),
            ).validated()
        )
    return tuple(rows)


def test_round28_ai_cases_match_augmented_candidate_population() -> None:
    markets, partition, probabilities, books, outcomes = _population(2)
    config = Round27EconomicConfig(
        minimum_executed_trades=1,
        minimum_profitable_conditions=1,
        bootstrap_draws=1_000,
    )
    baseline = evaluate_round27_economic_scenarios(
        partition=partition,
        predictions=probabilities,
        markets=markets,
        books=books,
        outcomes_up=outcomes,
        model_name=_FixedAugmentedModel.model_name,
        model_sha256=_MODEL_SHA256,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
    )
    panel = materialize_round28_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256=_SELECTION_SHA256,
        markets=markets,
        books=books,
        source_audit_sha256="c" * 64,
        config=config,
    )

    assert len(panel.cases) == 2
    assert (
        panel.baseline_candidate_population_sha256
        == baseline["candidate_population_sha256"]
    )
    assert len(panel.cases[0].causal_features) == 278
    assert panel.asdict()["target_accessed"] is False
    prompt = round28_ai_case_prompt(panel.cases[0])
    assert "binance_spot.bbo_" in prompt
    assert "binance_usdm.bbo_" in prompt


def test_round28_ai_case_batches_are_byte_equivalent_and_round_trip() -> None:
    markets, partition, _probabilities, books, _outcomes = _population(3)
    rows = _rows(partition)
    config = Round27EconomicConfig(
        minimum_executed_trades=1,
        minimum_profitable_conditions=1,
        bootstrap_draws=1_000,
    )
    direct = materialize_round28_ai_cases(
        role="selection",
        rows=rows,
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256=_SELECTION_SHA256,
        markets=markets,
        books=books,
        source_audit_sha256="c" * 64,
        config=config,
    )
    batches = tuple(
        Round27EconomicBookBatch(
            condition_ids=(market.condition_id,),
            books=tuple(
                book
                for book in books
                if book.market.condition_id == market.condition_id
            ),
        )
        for market in markets
    )
    batched = materialize_round28_ai_cases(
        role="selection",
        rows=rows,
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256=_SELECTION_SHA256,
        markets=markets,
        book_batches=(batch for batch in batches),
        source_audit_sha256="c" * 64,
        config=config,
    )

    assert batched.asdict() == direct.asdict()
    assert round28_ai_case_panel_from_mapping(direct.asdict()) == direct


def test_round28_ai_case_rejects_feature_or_model_view_tampering() -> None:
    markets, partition, _probabilities, books, _outcomes = _population(1)
    panel = materialize_round28_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256=_SELECTION_SHA256,
        markets=markets,
        books=books,
        source_audit_sha256="c" * 64,
        config=Round27EconomicConfig(),
    )
    case = panel.cases[0]
    features = list(case.causal_features)
    name, value = features[-1]
    features[-1] = (name, value + 1.0)

    with pytest.raises(ValueError, match="case differs"):
        replace(case, causal_features=tuple(features)).validated()
    with pytest.raises(ValueError, match="case differs"):
        replace(case, model_feature_view="round27_base").validated()


def test_round28_ai_cases_reject_base_only_model() -> None:
    markets, partition, _probabilities, books, _outcomes = _population(1)
    model = _FixedAugmentedModel()
    model.feature_view = "round27_base"

    with pytest.raises(ValueError, match="selected augmented model"):
        materialize_round28_ai_cases(
            role="selection",
            rows=_rows(partition),
            selected_model=model,
            selection_claim_sha256=_SELECTION_SHA256,
            markets=markets,
            books=books,
            source_audit_sha256="c" * 64,
            config=Round27EconomicConfig(),
        )
