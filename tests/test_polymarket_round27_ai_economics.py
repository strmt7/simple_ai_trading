from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round27_ai import (
    POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
    POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    materialize_round27_ai_cases,
)
from simple_ai_trading.polymarket_round27_ai_economics import (
    evaluate_round27_ai_matched_economics,
    select_round27_ai_candidate,
)
from simple_ai_trading.polymarket_round27_ai_inference import (
    run_round27_ai_inference,
)
from simple_ai_trading.polymarket_round27_economic_amendment import (
    bind_round27_economic_amendment,
)
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from test_polymarket_round27_ai_cases import _FixedModel, _rows
from test_polymarket_round27_economics import _book, _population


def _batches(markets, books):
    return tuple(
        Round27EconomicBookBatch(
            condition_ids=tuple(
                market.condition_id for market in markets[start : start + 30]
            ),
            books=tuple(
                book
                for book in books
                if book.market.condition_id
                in {
                    market.condition_id
                    for market in markets[start : start + 30]
                }
            ),
        )
        for start in range(0, len(markets), 30)
    )


def _raw(model: str, decision: str, reason: str) -> dict[str, object]:
    return {
        "model": model,
        "response": json.dumps(
            {"decision": decision, "reason_codes": [reason]},
            separators=(",", ":"),
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 100_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 500,
        "prompt_eval_duration": 50_000_000,
        "eval_count": 12,
        "eval_duration": 40_000_000,
    }


def _rehash(report: dict[str, object]) -> None:
    body = dict(report)
    body.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _inference(panel, candidate):
    case_count = 0

    def post_json(_url, payload, _timeout):
        nonlocal case_count
        prompt = payload.get("prompt")
        if prompt is None:
            return {}
        if str(prompt).startswith("Runtime conformance probe"):
            return _raw(candidate.runtime_model, "reject", "missing_liquidity")
        case_count += 1
        return _raw(
            candidate.runtime_model,
            "reject" if case_count <= 20 else "unchanged",
            "liquidity_thin" if case_count <= 20 else "no_material_risk",
        )

    residency_calls = 0

    def inspect_residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        loaded = residency_calls == 1
        return OllamaResidencyReport(
            requested_model=candidate.runtime_model,
            status="gpu_resident" if loaded else "unloaded",
            loaded_model=candidate.runtime_model if loaded else None,
            digest=candidate.runtime_digest if loaded else None,
            size_bytes=100 if loaded else None,
            size_vram_bytes=100 if loaded else None,
            vram_to_model_ratio=1.0 if loaded else None,
        ).validated()

    clock = 0

    def monotonic_ns():
        nonlocal clock
        clock += 100_000_000
        return clock

    return run_round27_ai_inference(
        panel=panel,
        candidate=candidate,
        post_json=post_json,
        residency_inspector=inspect_residency,
        inventory_getter=lambda _url, _timeout: {
            "models": [{"digest": candidate.runtime_digest}]
        },
        monotonic_ns=monotonic_ns,
    )


def _matched_fixture(candidate=POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE):
    markets, partition, probabilities, books, outcomes = _population(60)
    outcomes = {
        market.condition_id: int(index >= 20)
        for index, market in enumerate(markets)
    }
    extra_books = tuple(
        _book(market, outcome="Up", offset_ms=offset)
        for market in markets
        for offset in (30_350, 30_600, 31_100, 32_100, 33_100)
    )
    all_books = tuple(books) + extra_books
    config = Round27EconomicConfig(
        minimum_executed_trades=60,
        minimum_profitable_conditions=20,
        bootstrap_draws=1_000,
    )
    baseline = evaluate_round27_economic_scenarios(
        partition=partition,
        predictions=probabilities,
        markets=markets,
        outcomes_up=outcomes,
        model_name=_FixedModel.model_name,
        model_sha256="b" * 64,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
        book_batches=(batch for batch in _batches(markets, all_books)),
    )
    baseline = bind_round27_economic_amendment(
        baseline,
        hash_field="report_sha256",
    )
    panel = materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256="b" * 64,
        markets=markets,
        source_audit_sha256="c" * 64,
        config=config,
        book_batches=(batch for batch in _batches(markets, all_books)),
    )
    inference = _inference(panel, candidate)
    report = evaluate_round27_ai_matched_economics(
        panel=panel,
        inference_report=inference,
        baseline_economic_report=baseline,
        markets=markets,
        outcomes_up=outcomes,
        resolution_evidence_sha256="d" * 64,
        book_batches=(batch for batch in _batches(markets, all_books)),
    )
    return report


def test_round27_ai_matched_economics_charges_measured_inference_latency() -> None:
    report = _matched_fixture()

    assert report["matched_candidate_condition_count"] == 60
    assert report["matched_after_cost_uplift_gate_passed"] is True
    assert report["edge_claim"] is False
    assert report["orders_submitted"] is False
    for scenario in report["paired_scenarios"]:
        assert scenario["scenario_uplift_gate_passed"] is True
        assert scenario["ai"]["filled_order_count"] == 40
        assert scenario["paired_condition_bootstrap"]["ci95_lower"] > 0
    primary = next(
        scenario
        for scenario in report["paired_scenarios"]
        if scenario["base_delay_ms"] == 500
    )
    assert {trade["delay_ms"] for trade in primary["ai"]["trades"]} == {600}


def test_round27_ai_selection_uses_frozen_tie_break_and_grants_no_authority() -> None:
    qwen = _matched_fixture(POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE)
    oda = copy.deepcopy(qwen)
    oda["candidate"] = asdict(POLYMARKET_ROUND27_ODA_HOST_CANDIDATE)
    _rehash(oda)

    selection = select_round27_ai_candidate((qwen, oda))

    assert selection.nominated_model_id == "OpenDataArena/ODA-Fin-SFT-8B"
    assert selection.asdict()["sealed_partition_accessed"] is False
    assert selection.asdict()["trading_authority"] is False


def test_round27_ai_matched_economics_rejects_baseline_drift() -> None:
    report = _matched_fixture()
    tampered = copy.deepcopy(report)
    tampered["baseline_economic_report_sha256"] = "0" * 64

    # Selection must not accept reports from different baseline populations.
    tampered["candidate"] = asdict(POLYMARKET_ROUND27_ODA_HOST_CANDIDATE)
    _rehash(tampered)

    with pytest.raises(ValueError, match="matched selection population"):
        select_round27_ai_candidate((report, tampered))
