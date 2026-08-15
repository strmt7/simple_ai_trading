from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_ai_batch_economics as batch_module
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round28_ai_batch_economics import (
    evaluate_round28_ai_candidate_batch,
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
from test_polymarket_round27_economics import _book, _population
from test_polymarket_round28_ai_cases import (
    _FixedAugmentedModel,
    _rows,
)
from test_polymarket_round28_ai_economics import _batches
from test_polymarket_round28_ai_inference import _host_report


ROOT = Path(__file__).resolve().parents[1]


def test_round28_ai_batch_economics_scans_each_book_batch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets, partition, probabilities, raw_books, _outcomes = _population(60)
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
    parent = {
        "report_sha256": "f" * 64,
        "economic_uplift_gate_passed": True,
        "augmented_economic_report": baseline,
    }
    contract = load_round28_ai_contract(ROOT)
    qwen = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    oda = round28_ai_candidate_from_contract(
        contract,
        model_id="OpenDataArena/ODA-Fin-SFT-8B",
    )
    hosts = (_host_report(contract, qwen), _host_report(contract, oda))
    inference_raw = (
        {"model_id": qwen.model_id},
        {"model_id": oda.model_id},
    )
    inferences = {
        qwen.model_id: SimpleNamespace(
            responses=tuple(
                SimpleNamespace(
                    abstains=index % 3 == 0,
                    decision="reject" if index % 3 == 0 else "unchanged",
                    wall_latency_ms=1,
                )
                for index, _case in enumerate(panel.cases)
            ),
            candidate_eligible_for_matched_evaluation=True,
            report_sha256="1" * 64,
        ),
        oda.model_id: SimpleNamespace(
            responses=tuple(
                SimpleNamespace(
                    abstains=index % 3 == 1,
                    decision="reject" if index % 3 == 1 else "unchanged",
                    wall_latency_ms=1,
                )
                for index, _case in enumerate(panel.cases)
            ),
            candidate_eligible_for_matched_evaluation=True,
            report_sha256="2" * 64,
        ),
    }
    monkeypatch.setattr(
        batch_module,
        "validate_round28_ai_inference_report",
        lambda value, **_kwargs: inferences[value["model_id"]],
    )
    monkeypatch.setattr(
        batch_module,
        "validate_round28_economic_report",
        lambda *_args, **_kwargs: parent,
    )
    yielded = 0

    def tracked_batches():
        nonlocal yielded
        for batch in batches:
            yielded += 1
            yield batch

    reports = evaluate_round28_ai_candidate_batch(
        panel=panel,
        candidate_evidence=tuple(zip(hosts, inference_raw, strict=True)),
        contract=contract,
        round28_economic_report={},
        input_manifest={},
        selection_claim={},
        markets=markets,
        outcomes_up=outcomes,
        resolution_evidence_sha256="d" * 64,
        book_batches=tracked_batches(),
    )

    assert yielded == len(batches)
    assert [report["candidate"]["model_id"] for report in reports] == [
        qwen.model_id,
        oda.model_id,
    ]
    assert reports[0]["matched_after_cost_uplift_gate_passed"] is True
    assert reports[1]["matched_after_cost_uplift_gate_passed"] is False
