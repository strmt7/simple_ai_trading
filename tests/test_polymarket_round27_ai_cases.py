from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from simple_ai_trading.polymarket_round27_ai_cases import (
    materialize_round27_ai_cases,
    round27_ai_case_prompt,
)
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round27_features import Round27FeatureRow
from test_polymarket_round27_economics import _population


_MODEL_SHA256 = "b" * 64


class _FixedModel:
    model_name = "fixed-test-model"

    def predict(self, features, offsets):
        del offsets
        return np.full(features.shape[0], 0.80, dtype=np.float64)

    def asdict(self):
        return {
            "model_name": self.model_name,
            "model_sha256": _MODEL_SHA256,
        }


def _rows(partition) -> tuple[Round27FeatureRow, ...]:
    return tuple(
        Round27FeatureRow.create(
            run_id="round27-stage1-test",
            condition_id=sample.condition_id,
            event_start_ms=sample.event_start_ms,
            decision_time_ms=sample.decision_time_ms,
            market_prior_probability=sample.market_prior_probability,
            values=sample.values,
            maximum_receipt_wall_ms=sample.decision_time_ms,
            source_chain_sha256=hashlib.sha256(
                f"source-{sample.condition_id}".encode("ascii")
            ).hexdigest(),
        )
        for sample in partition.samples
    )


def test_round27_ai_cases_match_frozen_baseline_candidate_population() -> None:
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
        model_name=_FixedModel.model_name,
        model_sha256=_MODEL_SHA256,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
    )
    panel = materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256=_MODEL_SHA256,
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
    assert panel.asdict()["target_accessed"] is False
    prompt = round27_ai_case_prompt(panel.cases[0]).lower()
    assert all(
        forbidden not in prompt
        for forbidden in ("target", "resolution", "pnl", "profit")
    )


def test_round27_ai_case_batches_are_byte_equivalent() -> None:
    markets, partition, _probabilities, books, _outcomes = _population(3)
    config = Round27EconomicConfig(
        minimum_executed_trades=1,
        minimum_profitable_conditions=1,
        bootstrap_draws=1_000,
    )
    direct = materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256=_MODEL_SHA256,
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
    batched = materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256=_MODEL_SHA256,
        markets=markets,
        book_batches=(batch for batch in batches),
        source_audit_sha256="c" * 64,
        config=config,
    )

    assert batched.asdict() == direct.asdict()


def test_round27_ai_case_integrity_rejects_feature_tampering() -> None:
    markets, partition, _probabilities, books, _outcomes = _population(1)
    panel = materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256=_MODEL_SHA256,
        markets=markets,
        books=books,
        source_audit_sha256="c" * 64,
        config=Round27EconomicConfig(),
    )
    case = panel.cases[0]
    tampered_features = list(case.causal_features)
    name, value = tampered_features[0]
    tampered_features[0] = (name, value + 1.0)

    with pytest.raises(ValueError, match="case differs"):
        replace(case, causal_features=tuple(tampered_features)).validated()
