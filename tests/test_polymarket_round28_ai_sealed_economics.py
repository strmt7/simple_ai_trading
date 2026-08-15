from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_ai_sealed as sealed_ai
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
from simple_ai_trading.polymarket_round28_ai_host import (
    round28_ai_candidate_from_contract,
)
from simple_ai_trading.polymarket_round28_ai_sealed import (
    evaluate_round28_ai_sealed_economics,
)
from test_polymarket_round27_economics import _book, _population
from test_polymarket_round28_ai_cases import _FixedAugmentedModel, _rows
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


def test_round28_ai_sealed_economics_replays_exact_frozen_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets, partition, probabilities, raw_books, _ = _population(60)
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
        role="sealed",
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
        model_name=f"{panel.model_name}:round28_bbo_augmented",
        model_sha256=panel.model_sha256,
        source_audit_sha256=panel.source_audit_sha256,
        resolution_evidence_sha256="d" * 64,
        config=config,
    )
    parent = {
        "report_sha256": "f" * 64,
        "selection_claim_sha256": panel.selection_claim_sha256,
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
        sealed_ai,
        "validate_round28_ai_inference_report",
        lambda *_args, **_kwargs: fake_inference,
    )
    monkeypatch.setattr(
        sealed_ai,
        "validate_round28_sealed_economic_report",
        lambda _value: parent,
    )
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )

    report = evaluate_round28_ai_sealed_economics(
        panel=panel,
        inference_report={},
        contract=contract,
        host_qualification_report=_host_report(contract, candidate),
        sealed_round28_economic_report={},
        markets=markets,
        outcomes_up=outcomes,
        resolution_evidence_sha256="d" * 64,
        book_batches=iter(batches),
    )

    assert report["matched_after_cost_uplift_gate_passed"] is True
    assert report["partition_role"] == "sealed"
    assert report["case_panel_sha256"] == panel.panel_sha256
    assert report["augmented_baseline_economic_report_sha256"] == baseline[
        "report_sha256"
    ]
    assert all(
        scenario["augmented_baseline"]["filled_order_count"] == 60
        and scenario["ai"]["filled_order_count"] == 40
        for scenario in report["paired_scenarios"]
    )
    assert report["edge_claim"] is False
    assert report["profitability_claim"] is False
    assert report["orders_submitted"] is False
