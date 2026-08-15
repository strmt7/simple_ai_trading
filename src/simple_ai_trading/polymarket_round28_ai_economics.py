"""Latency-aware matched economics for the Round 28 AI risk veto."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json

from . import polymarket_round27_economics as _economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    Round27EconomicTrade,
)
from .polymarket_round28_ai_cases import Round28AICase, Round28AICasePanel
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS,
)
from .polymarket_round28_ai_host import validate_round28_ai_host_report
from .polymarket_round28_ai_inference import validate_round28_ai_inference_report
from .polymarket_round28_operator import validate_round28_economic_report


POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION = (
    "polymarket-round28-ai-matched-economic-report-v1"
)
_UPLIFT_GATE_NAMES = frozenset(
    {
        "round28_augmented_economic_gate_passed",
        "inference_candidate_eligible",
        "minimum_ai_filled_conditions_met",
        "minimum_profitable_ai_conditions_met",
        "ai_net_pnl_positive",
        "ai_net_pnl_strictly_greater_than_augmented_baseline",
        "paired_mean_net_pnl_delta_positive",
        "paired_condition_bootstrap_lower_bound_positive",
        "maximum_drawdown_not_worse",
        "mean_adverse_markout_not_worse",
        "no_unknown_execution_state",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 28 AI {name} SHA-256 differs")
    return selected


def _decimal(value: object, *, name: str) -> Decimal:
    try:
        selected = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001 - normalized evidence error
        raise ValueError(f"Round 28 AI {name} decimal differs") from exc
    if not selected.is_finite():
        raise ValueError(f"Round 28 AI {name} decimal differs")
    return selected


def _candidate_from_case(case: Round28AICase) -> object:
    selected = case.validated()
    return _economics._DecisionCandidate(  # noqa: SLF001
        sample_index=0,
        condition_id=selected.condition_id,
        event_start_ms=selected.event_start_ms,
        market_end_ms=selected.market_end_ms,
        decision_time_ms=selected.decision_time_ms,
        outcome=selected.proposed_side,
        token_id=selected.token_id,
        predicted_probability=selected.predicted_probability,
        quantity=Decimal(selected.quantity),
        limit_price=Decimal(selected.limit_price),
        decision_tick_size=Decimal(selected.decision_tick_size),
        decision_average_price=Decimal(selected.decision_average_price),
        decision_fee_quote=Decimal(selected.decision_fee_quote),
        expected_edge_per_contract=Decimal(selected.expected_edge_per_contract),
        segment_id=selected.segment_id,
        connection_id=selected.connection_id,
        decision_book_event_id=selected.decision_book_event_id,
        decision_source_payload_sha256=(
            selected.decision_source_payload_sha256
        ),
    )


def _trade_pnl_by_condition(
    trades: Sequence[Round27EconomicTrade],
) -> dict[str, Decimal]:
    return {
        trade.condition_id: (
            trade.net_pnl_quote
            if trade.execution_state == "FILLED"
            else Decimal("0")
        )
        for trade in trades
    }


def _paired_scenario(
    *,
    delay_ms: int,
    cases: Sequence[Round28AICase],
    baseline: Mapping[str, object],
    ai: Mapping[str, object],
    baseline_trades: Sequence[Round27EconomicTrade],
    ai_trades: Sequence[Round27EconomicTrade],
    inference_eligible: bool,
    config: Round27EconomicConfig,
) -> dict[str, object]:
    baseline_pnl = _trade_pnl_by_condition(baseline_trades)
    ai_pnl = _trade_pnl_by_condition(ai_trades)
    deltas = tuple(
        ai_pnl.get(case.condition_id, Decimal("0"))
        - baseline_pnl.get(case.condition_id, Decimal("0"))
        for case in cases
    )
    bootstrap = _economics._condition_bootstrap(  # noqa: SLF001
        deltas,
        draws=config.bootstrap_draws,
        seed=28_028 + delay_ms,
    )
    baseline_markout = baseline.get("mean_markout_pnl_per_contract")
    ai_markout = ai.get("mean_markout_pnl_per_contract")
    markout_not_worse = bool(
        baseline_markout is not None
        and ai_markout is not None
        and _decimal(ai_markout, name="AI markout")
        >= _decimal(baseline_markout, name="baseline markout")
    )
    baseline_drawdown = _decimal(
        baseline["maximum_drawdown_fraction"],
        name="baseline drawdown",
    )
    ai_drawdown = _decimal(
        ai["maximum_drawdown_fraction"],
        name="AI drawdown",
    )
    baseline_net_pnl = _decimal(
        baseline["net_pnl_quote"],
        name="baseline net PnL",
    )
    ai_net_pnl = _decimal(ai["net_pnl_quote"], name="AI net PnL")
    mean_delta = _decimal(
        bootstrap["mean_net_pnl_quote"],
        name="paired mean net PnL",
    )
    checks = {
        "round28_augmented_economic_gate_passed": (
            baseline.get("scenario_edge_gate_passed") is True
        ),
        "inference_candidate_eligible": inference_eligible,
        "minimum_ai_filled_conditions_met": int(ai["filled_order_count"])
        >= POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS,
        "minimum_profitable_ai_conditions_met": int(ai["profitable_condition_count"])
        >= config.minimum_profitable_conditions,
        "ai_net_pnl_positive": ai_net_pnl > 0,
        "ai_net_pnl_strictly_greater_than_augmented_baseline": (
            ai_net_pnl > baseline_net_pnl
        ),
        "paired_mean_net_pnl_delta_positive": mean_delta > 0,
        "paired_condition_bootstrap_lower_bound_positive": bool(
            bootstrap["eligible"]
        )
        and _decimal(bootstrap["ci95_lower"], name="paired lower bound") > 0,
        "maximum_drawdown_not_worse": ai_drawdown <= baseline_drawdown,
        "mean_adverse_markout_not_worse": markout_not_worse,
        "no_unknown_execution_state": int(ai["unknown_order_count"]) == 0,
    }
    body: dict[str, object] = {
        "base_delay_ms": delay_ms,
        "augmented_baseline": dict(baseline),
        "ai": dict(ai),
        "paired_condition_count": len(cases),
        "paired_mean_net_pnl_delta_quote": format(mean_delta, "f"),
        "paired_condition_bootstrap": bootstrap,
        "maximum_drawdown_delta_fraction": format(
            ai_drawdown - baseline_drawdown,
            "f",
        ),
        "gate_checks": checks,
        "scenario_uplift_gate_passed": all(checks.values()),
    }
    body["paired_scenario_sha256"] = _canonical_sha256(body)
    return body


def _economic_config(value: Mapping[str, object]) -> Round27EconomicConfig:
    return Round27EconomicConfig(
        **{
            key: (
                tuple(raw)
                if key == "delays_ms" and isinstance(raw, list)
                else Decimal(str(raw))
                if key
                in {
                    "minimum_expected_edge_per_contract",
                    "maximum_entry_cost_quote",
                    "initial_capital_quote",
                }
                else raw
            )
            for key, raw in value.items()
        }
    ).validated()


def evaluate_round28_ai_matched_economics(
    *,
    panel: Round28AICasePanel,
    inference_report: Mapping[str, object],
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
    round28_economic_report: Mapping[str, object],
    input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    resolution_evidence_sha256: str,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> dict[str, object]:
    """Compare one AI candidate with the exact augmented execution baseline."""

    selected_panel = panel.validated()
    selected_inference = validate_round28_ai_inference_report(
        inference_report,
        contract=contract,
        host_qualification_report=host_qualification_report,
        panel=selected_panel,
    )
    baseline_parent = validate_round28_economic_report(
        round28_economic_report,
        input_manifest=input_manifest,
        selection_claim=selection_claim,
        resolution_evidence_sha256=resolution_evidence_sha256,
    )
    if baseline_parent.get("economic_uplift_gate_passed") is not True:
        raise ValueError("Round 28 AI requires a passed augmented economic baseline")
    baseline_report = baseline_parent.get("augmented_economic_report")
    if not isinstance(baseline_report, Mapping):
        raise ValueError("Round 28 augmented baseline report differs")
    config = _economic_config(selected_panel.economic_config)
    ai_config = replace(
        config,
        minimum_executed_trades=POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS,
    ).validated()
    expected_model_name = f"{selected_panel.model_name}:round28_bbo_augmented"
    if (
        baseline_report.get("model_name") != expected_model_name
        or baseline_report.get("model_sha256") != selected_panel.model_sha256
        or baseline_report.get("source_audit_sha256")
        != selected_panel.source_audit_sha256
        or baseline_report.get("config") != selected_panel.economic_config
        or baseline_report.get("candidate_condition_count")
        != len(selected_panel.cases)
        or baseline_report.get("candidate_population_sha256")
        != selected_panel.baseline_candidate_population_sha256
    ):
        raise ValueError("Round 28 AI augmented baseline lineage differs")
    conditions = {market.condition_id for market in markets}
    if (
        len(conditions) != selected_panel.evaluated_condition_count
        or _canonical_sha256(sorted(conditions))
        != selected_panel.evaluated_condition_ids_sha256
        or set(outcomes_up) != conditions
        or any(type(value) is not int or value not in {0, 1} for value in outcomes_up.values())
    ):
        raise ValueError("Round 28 AI economic condition population differs")
    market_by_condition = {market.condition_id: market for market in markets}
    ordered_conditions = tuple(
        condition_id
        for condition_id, _market in sorted(
            market_by_condition.items(),
            key=lambda item: (item[1].event_start_ms, item[0]),
        )
    )
    if (books is None) == (book_batches is None):
        raise ValueError("Round 28 AI economics requires exactly one book source")
    batches: Iterable[Round27EconomicBookBatch] = (
        (
            Round27EconomicBookBatch(
                condition_ids=tuple(sorted(conditions)),
                books=tuple(books or ()),
            ),
        )
        if books is not None
        else book_batches or ()
    )
    case_by_condition = {
        case.condition_id: case for case in selected_panel.cases
    }
    response_by_condition = {
        case.condition_id: response
        for case, response in zip(
            selected_panel.cases,
            selected_inference.responses,
            strict=True,
        )
    }
    seen_conditions: set[str] = set()
    baseline_by_delay: dict[int, list[Round27EconomicTrade]] = {
        delay: [] for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    ai_by_delay: dict[int, list[Round27EconomicTrade]] = {
        delay: [] for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    baseline_reasons = {
        delay: Counter(selected_panel.selection_reason_counts)
        for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    ai_reasons = {
        delay: Counter(selected_panel.selection_reason_counts)
        for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    for raw_batch in batches:
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > config.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 28 AI economic book batch scope differs")
        seen_conditions.update(batch_conditions)
        batch_cases = tuple(
            case_by_condition[condition_id]
            for condition_id in batch.condition_ids
            if condition_id in case_by_condition
        )
        candidates = tuple(_candidate_from_case(case) for case in batch_cases)
        batch_markets = {
            condition_id: market_by_condition[condition_id]
            for condition_id in batch.condition_ids
        }
        index = _economics._BookIndex(batch.books)  # noqa: SLF001
        for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
            baseline_trades, reasons = _economics._execute_candidate_trades(  # noqa: SLF001
                candidates=candidates,
                delay_ms=delay,
                market_by_condition=batch_markets,
                outcomes_up=outcomes_up,
                index=index,
                config=config,
            )
            baseline_by_delay[delay].extend(baseline_trades)
            baseline_reasons[delay].update(reasons)
            unchanged_by_latency: dict[int, list[object]] = {}
            for case, candidate in zip(batch_cases, candidates, strict=True):
                response = response_by_condition[case.condition_id]
                if response.abstains:
                    ai_reasons[delay][f"ai_{response.decision}"] += 1
                else:
                    unchanged_by_latency.setdefault(
                        response.wall_latency_ms,
                        [],
                    ).append(candidate)
            for latency_ms, latency_candidates in sorted(
                unchanged_by_latency.items()
            ):
                ai_trades, reasons = _economics._execute_candidate_trades(  # noqa: SLF001
                    candidates=latency_candidates,
                    delay_ms=delay + latency_ms,
                    market_by_condition=batch_markets,
                    outcomes_up=outcomes_up,
                    index=index,
                    config=ai_config,
                )
                ai_by_delay[delay].extend(ai_trades)
                ai_reasons[delay].update(reasons)
    if seen_conditions != conditions:
        raise ValueError("Round 28 AI economic book batches do not cover the role")
    baseline_scenarios = {
        int(item["delay_ms"]): item for item in baseline_report["scenarios"]
    }
    paired_scenarios: list[dict[str, object]] = []
    for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
        baseline_trades = tuple(
            sorted(
                baseline_by_delay[delay],
                key=lambda trade: (
                    trade.event_start_ms,
                    trade.condition_id,
                    trade.decision_time_ms,
                ),
            )
        )
        ai_trades = tuple(
            sorted(
                ai_by_delay[delay],
                key=lambda trade: (
                    trade.event_start_ms,
                    trade.condition_id,
                    trade.decision_time_ms,
                ),
            )
        )
        replayed_baseline = _economics._scenario_report(  # noqa: SLF001
            trades=baseline_trades,
            candidate_count=len(selected_panel.cases),
            delay_ms=delay,
            evaluated_conditions=ordered_conditions,
            reasons=dict(sorted(baseline_reasons[delay].items())),
            config=config,
        )
        if replayed_baseline != baseline_scenarios.get(delay):
            raise ValueError("Round 28 AI replayed augmented baseline differs")
        ai_scenario = _economics._scenario_report(  # noqa: SLF001
            trades=ai_trades,
            candidate_count=len(ai_trades),
            delay_ms=delay,
            evaluated_conditions=ordered_conditions,
            reasons=dict(sorted(ai_reasons[delay].items())),
            config=ai_config,
        )
        paired_scenarios.append(
            _paired_scenario(
                delay_ms=delay,
                cases=selected_panel.cases,
                baseline=replayed_baseline,
                ai=ai_scenario,
                baseline_trades=baseline_trades,
                ai_trades=ai_trades,
                inference_eligible=(
                    selected_inference.candidate_eligible_for_matched_evaluation
                ),
                config=ai_config,
            )
        )
    all_pass = all(
        scenario["scenario_uplift_gate_passed"] is True
        for scenario in paired_scenarios
    )
    host_report, candidate = validate_round28_ai_host_report(
        host_qualification_report,
        contract=contract,
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "partition_role": selected_panel.partition_role,
        "candidate": asdict(candidate),
        "host_qualification_report_sha256": host_report["report_sha256"],
        "case_panel_sha256": selected_panel.panel_sha256,
        "inference_report_sha256": selected_inference.report_sha256,
        "round28_economic_report_sha256": baseline_parent["report_sha256"],
        "augmented_baseline_economic_report_sha256": baseline_report[
            "report_sha256"
        ],
        "selection_claim_sha256": selected_panel.selection_claim_sha256,
        "source_audit_sha256": selected_panel.source_audit_sha256,
        "resolution_evidence_sha256": _sha256(
            resolution_evidence_sha256,
            name="resolution evidence",
        ),
        "matched_candidate_condition_count": len(selected_panel.cases),
        "paired_scenarios": paired_scenarios,
        "primary_500ms_uplift_gate_passed": next(
            scenario["scenario_uplift_gate_passed"]
            for scenario in paired_scenarios
            if scenario["base_delay_ms"] == 500
        ),
        "all_delay_scenarios_uplift_gate_passed": all_pass,
        "matched_after_cost_uplift_gate_passed": all_pass,
        "sealed_partition_accessed": selected_panel.partition_role == "sealed",
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_ai_economic_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    report = dict(value)
    claimed = _sha256(report.pop("report_sha256", None), name="economic report")
    candidate = report.get("candidate")
    scenarios = report.get("paired_scenarios")
    valid_scenarios = bool(
        isinstance(scenarios, list)
        and len(scenarios) == len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
    )
    if valid_scenarios:
        for scenario, delay in zip(
            scenarios,
            POLYMARKET_ROUND27_FIXED_DELAYS_MS,
            strict=True,
        ):
            if not isinstance(scenario, Mapping):
                valid_scenarios = False
                break
            body = dict(scenario)
            scenario_claimed = str(body.pop("paired_scenario_sha256", ""))
            checks = scenario.get("gate_checks")
            baseline = scenario.get("augmented_baseline")
            ai = scenario.get("ai")
            if (
                scenario.get("base_delay_ms") != delay
                or scenario_claimed != _canonical_sha256(body)
                or not isinstance(checks, Mapping)
                or set(checks) != _UPLIFT_GATE_NAMES
                or any(type(item) is not bool for item in checks.values())
                or scenario.get("scenario_uplift_gate_passed")
                is not all(checks.values())
                or not isinstance(baseline, Mapping)
                or not isinstance(ai, Mapping)
            ):
                valid_scenarios = False
                break
            for nested in (baseline, ai):
                nested_body = dict(nested)
                nested_claim = str(nested_body.pop("scenario_sha256", ""))
                if nested_claim != _canonical_sha256(nested_body):
                    valid_scenarios = False
                    break
    all_pass = bool(
        valid_scenarios
        and all(item["scenario_uplift_gate_passed"] is True for item in scenarios)
    )
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version")
        != POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION
        or report.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or report.get("partition_role") not in {"selection", "sealed"}
        or not isinstance(candidate, Mapping)
        or not valid_scenarios
        or report.get("primary_500ms_uplift_gate_passed")
        is not bool(scenarios[1]["scenario_uplift_gate_passed"])
        or report.get("all_delay_scenarios_uplift_gate_passed") is not all_pass
        or report.get("matched_after_cost_uplift_gate_passed") is not all_pass
        or report.get("sealed_partition_accessed")
        is not (report.get("partition_role") == "sealed")
        or any(
            report.get(field) is not False
            for field in (
                "edge_claim",
                "profitability_claim",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 AI matched economic report differs")
    return {**report, "report_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION",
    "evaluate_round28_ai_matched_economics",
    "validate_round28_ai_economic_report",
]
