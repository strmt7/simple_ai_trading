from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_ai_economics as ai_economics
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round28_ai_cases import (
    materialize_round28_ai_cases,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_economics import (
    evaluate_round28_ai_matched_economics,
    validate_round28_ai_economic_report,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    round28_ai_candidate_from_contract,
)
from test_polymarket_round27_economics import _book, _population
from test_polymarket_round28_ai_cases import (
    _FixedAugmentedModel,
    _rows,
)
from test_polymarket_round28_ai_inference import _host_report


ROOT = Path(__file__).resolve().parents[1]


def _batches(markets, books) -> tuple[Round27EconomicBookBatch, ...]:
    output: list[Round27EconomicBookBatch] = []
    for start in range(0, len(markets), 20):
        selected = markets[start : start + 20]
        condition_ids = tuple(market.condition_id for market in selected)
        condition_set = set(condition_ids)
        output.append(
            Round27EconomicBookBatch(
                condition_ids=condition_ids,
                books=tuple(
                    book
                    for book in books
                    if book.market.condition_id in condition_set
                ),
            ).validated()
        )
    return tuple(output)


def test_round28_ai_economics_vetoes_losing_cases_and_beats_exact_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets, partition, probabilities, raw_books, original_outcomes = _population(60)
    del original_outcomes
    books = (
        *raw_books,
        *(
            _book(market, outcome="Up", offset_ms=offset)
            for market in markets
            for offset in (33_002, 33_502)
        ),
    )
    outcomes = {
        market.condition_id: (0 if index % 3 == 0 else 1)
        for index, market in enumerate(markets)
    }
    config = Round27EconomicConfig(
        minimum_executed_trades=20,
        minimum_profitable_conditions=20,
        bootstrap_draws=1_000,
    )
    batches = _batches(markets, books)
    panel = materialize_round28_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256="e" * 64,
        markets=markets,
        book_batches=iter(batches),
        source_audit_sha256="c" * 64,
        config=config,
    )
    baseline = evaluate_round27_economic_scenarios(
        partition=partition,
        predictions=probabilities,
        markets=markets,
        book_batches=iter(batches),
        outcomes_up=outcomes,
        model_name="fixed-test-model:round28_bbo_augmented",
        model_sha256="b" * 64,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
    )
    assert baseline["economic_edge_gate_passed"] is True
    parent = {
        "report_sha256": "f" * 64,
        "economic_uplift_gate_passed": True,
        "augmented_economic_report": baseline,
    }
    responses = tuple(
        SimpleNamespace(
            abstains=index % 3 == 0,
            decision="reject" if index % 3 == 0 else "unchanged",
            wall_latency_ms=1,
        )
        for index, _case in enumerate(panel.cases)
    )
    fake_inference = SimpleNamespace(
        responses=responses,
        candidate_eligible_for_matched_evaluation=True,
        report_sha256="1" * 64,
    )
    monkeypatch.setattr(
        ai_economics,
        "validate_round28_ai_inference_report",
        lambda *_args, **_kwargs: fake_inference,
    )
    monkeypatch.setattr(
        ai_economics,
        "validate_round28_economic_report",
        lambda *_args, **_kwargs: parent,
    )
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    host_report = _host_report(contract, candidate)

    report = evaluate_round28_ai_matched_economics(
        panel=panel,
        inference_report={},
        contract=contract,
        host_qualification_report=host_report,
        round28_economic_report={},
        input_manifest={},
        selection_claim={},
        markets=markets,
        outcomes_up=outcomes,
        resolution_evidence_sha256="d" * 64,
        book_batches=iter(batches),
    )

    assert report["matched_after_cost_uplift_gate_passed"] is True, [
        json.dumps(
            {
                key: value
                for key, value in scenario["gate_checks"].items()
                if value is not True
            },
            sort_keys=True,
        )
        for scenario in report["paired_scenarios"]
    ]
    assert report["matched_candidate_condition_count"] == 60
    assert report["candidate"]["model_id"] == candidate.model_id
    assert report["edge_claim"] is False
    for scenario in report["paired_scenarios"]:
        assert scenario["ai"]["filled_order_count"] == 40
        assert scenario["augmented_baseline"]["filled_order_count"] == 60
        assert scenario["gate_checks"][
            "ai_net_pnl_strictly_greater_than_augmented_baseline"
        ] is True
        assert scenario["paired_condition_bootstrap"]["ci95_lower"] > 0
    assert validate_round28_ai_economic_report(report) == report


def test_round28_ai_economic_validator_rejects_rehashed_gate_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="economic report"):
        validate_round28_ai_economic_report({"report_sha256": "x"})

    # The module-level minimum remains the preregistered institutional gate.
    assert ai_economics.POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS == 30
