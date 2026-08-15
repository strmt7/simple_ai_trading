from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import polymarket_round21_economic_operator as operator
from simple_ai_trading.polymarket_round21_comparison import (
    POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
    compare_round21_optional_replay_matrices,
)
from simple_ai_trading.polymarket_round21_corpus import (
    Round21ConditionSource,
    Round21CoreCondition,
)
from simple_ai_trading.polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
)
from polymarket_round21_support import round21_replay_condition


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_round21_economic_observer_streams_exact_matched_paths_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_source = round21_replay_condition(layer="core")
    challenger_source = round21_replay_condition(layer="core_spot")
    market = baseline_source.market
    condition = Round21CoreCondition(
        run_id="1" * 32,
        segment_index=1,
        snapshot_sha256=baseline_source.market_evidence.snapshot_sha256,
        snapshot_observed_wall_ms=(baseline_source.market_evidence.observed_wall_ms),
        condition_id=market.condition_id,
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        up_token_id=market.up_token_id,
        down_token_id=market.down_token_id,
    ).validated()
    source = Round21ConditionSource(
        union_events=(),
        chainlink_records=(),
        lane_event_wall_ms={"clob-a": (), "clob-b": ()},
        stream_gaps=(),
    ).validated()
    monkeypatch.setattr(
        operator,
        "build_round21_execution_books",
        lambda **_kwargs: baseline_source.books,
    )
    baseline_accumulator = Round21EconomicMatrixAccumulator()
    challenger_accumulator = Round21EconomicMatrixAccumulator()
    selected_conditions = []
    observer = operator._Round21EconomicObserver(  # noqa: SLF001
        contexts={
            market.condition_id: operator._MarketContext(  # noqa: SLF001
                condition,
                market,
                baseline_source.market_evidence,
            )
        },
        outcomes={market.condition_id: baseline_source.outcome},
        baseline_envelopes={market.condition_id: baseline_source.envelopes},
        challenger_envelopes={market.condition_id: challenger_source.envelopes},
        terminal_manifest_sha256=_sha("terminal"),
        core_publication_manifest_sha256=_sha("publication"),
        baseline_accumulator=baseline_accumulator,
        challenger_accumulator=challenger_accumulator,
        selected_condition_sinks=(selected_conditions.append,),
    )

    observer.observe(condition, source)
    observer.finish()
    assert len(selected_conditions) == 1
    assert selected_conditions[0].envelopes[0].model_layer == "core_spot"
    assert selected_conditions[0].books == baseline_source.books
    assert selected_conditions[0].outcome == baseline_source.outcome
    with pytest.raises(ValueError, match="source condition differs"):
        observer.observe(condition, source)
    baseline_matrix = baseline_accumulator.finish()
    challenger_matrix = challenger_accumulator.finish()
    matched_sha = operator._canonical_sha256(  # noqa: SLF001
        {
            "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
            "condition_sha256": observer.matched_condition_sha256,
        }
    )
    comparison = compare_round21_optional_replay_matrices(
        baseline_matrix=baseline_matrix,
        challenger_matrix=challenger_matrix,
        challenger_layer="core_spot",
        matched_population_sha256=matched_sha,
    )
    provisional = operator.Round21DevelopmentEconomicResult(
        selected_population_layer="core_spot",
        terminal_transport_manifest_sha256=_sha("terminal"),
        core_publication_manifest_sha256=_sha("publication"),
        model_artifact_sha256=_sha("model"),
        terminal_receipt_audit_sha256=_sha("audit"),
        source_condition_set_sha256=_sha("condition-set"),
        source_condition_count=1,
        selected_matrix=challenger_matrix,
        optional_comparison=comparison,
        development_gate_passed=False,
        result_sha256=hashlib.sha256(b"").hexdigest(),
    )
    result = replace(
        provisional,
        result_sha256=operator._canonical_sha256(  # noqa: SLF001
            provisional.identity_payload()
        ),
    ).validated()

    assert len(result.selected_matrix) == 81
    assert result.optional_comparison == comparison
    assert result.development_gate_passed is False
    assert result.profitability_claim is False
    report = result.asdict()
    assert len(report["selected_matrix"]) == 81
    assert report["optional_comparison"]["comparison_sha256"] == (
        comparison.comparison_sha256
    )
    false_pass = replace(
        result,
        development_gate_passed=True,
        result_sha256=hashlib.sha256(b"").hexdigest(),
    )
    false_pass = replace(
        false_pass,
        result_sha256=operator._canonical_sha256(  # noqa: SLF001
            false_pass.identity_payload()
        ),
    )
    with pytest.raises(ValueError, match="development economic result differs"):
        false_pass.validated()
    with pytest.raises(ValueError, match="development economic result differs"):
        replace(result, live_trading_authority=True).validated()


def test_round21_outcome_check_uses_exact_optional_condition_subset() -> None:
    condition = round21_replay_condition(layer="core")
    panel = SimpleNamespace(
        condition_ids=np.asarray(
            [condition.market.condition_id, "0x" + "8" * 64],
            dtype=object,
        ),
        event_start_ms=np.asarray(
            [
                condition.market.event_start_ms,
                condition.market.event_start_ms + 300_000,
            ],
            dtype=np.int64,
        ),
        labels=np.asarray([1.0, 0.0], dtype=np.float64),
    )

    operator._verify_panel_outcomes(  # noqa: SLF001
        (panel,),
        {condition.market.condition_id: condition.outcome},
    )
    with pytest.raises(ValueError, match="official outcomes"):
        operator._verify_panel_outcomes(  # noqa: SLF001
            (panel,),
            {
                condition.market.condition_id: replace(
                    condition.outcome,
                    resolved_up=False,
                )
            },
        )


def test_round21_ai_case_wrapper_enables_one_pass_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_replay(**kwargs):
        calls.append(kwargs)
        return "economic", ("case",)

    monkeypatch.setattr(
        operator,
        "_replay_round21_development_economics",
        fake_replay,
    )
    result = operator.replay_round21_development_economics_with_ai_cases(
        source_database="source.duckdb",
        terminal_transport_manifest={},
        partition_policy=SimpleNamespace(),
        development_panels=(),
        development_model_artifact={},
        core_publication_manifest_sha256=_sha("publication"),
    )

    assert result == ("economic", ("case",))
    assert len(calls) == 1
    assert calls[0]["collect_historical_ai_cases"] is True
    assert calls[0]["selected_condition_sinks"] == ()
