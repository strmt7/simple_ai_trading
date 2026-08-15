"""Single-scan matched economics for all qualified Round 28 AI candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

from . import polymarket_round27_economics as _economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
    Round27EconomicBookBatch,
    Round27EconomicTrade,
)
from .polymarket_round28_ai_cases import Round28AICasePanel
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS,
)
from .polymarket_round28_ai_economics import (
    POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION,
    _candidate_from_case,
    _canonical_sha256,
    _economic_config,
    _paired_scenario,
    _sha256,
    validate_round28_ai_economic_report,
)
from .polymarket_round28_ai_host import validate_round28_ai_host_report
from .polymarket_round28_ai_inference import validate_round28_ai_inference_report
from .polymarket_round28_operator import validate_round28_economic_report


def evaluate_round28_ai_candidate_batch(
    *,
    panel: Round28AICasePanel,
    candidate_evidence: Sequence[
        tuple[Mapping[str, object], Mapping[str, object]]
    ],
    contract: Mapping[str, object],
    round28_economic_report: Mapping[str, object],
    input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    resolution_evidence_sha256: str,
    book_batches: Iterable[Round27EconomicBookBatch],
) -> tuple[dict[str, object], ...]:
    """Replay the baseline once and all host-qualified vetoes in one DB scan."""

    selected_panel = panel.validated()
    validated_candidates: list[tuple[dict[str, object], Any, Any]] = []
    model_ids: set[str] = set()
    for host_raw, inference_raw in candidate_evidence:
        host_report, candidate = validate_round28_ai_host_report(
            host_raw,
            contract=contract,
        )
        inference = validate_round28_ai_inference_report(
            inference_raw,
            contract=contract,
            host_qualification_report=host_report,
            panel=selected_panel,
        )
        if candidate.model_id in model_ids:
            raise ValueError("Round 28 AI batch candidate is duplicated")
        model_ids.add(candidate.model_id)
        validated_candidates.append((host_report, candidate, inference))
    if not validated_candidates:
        raise ValueError("Round 28 AI batch candidate population is empty")
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
        raise ValueError("Round 28 AI batch condition population differs")
    market_by_condition = {market.condition_id: market for market in markets}
    ordered_conditions = tuple(
        condition_id
        for condition_id, _market in sorted(
            market_by_condition.items(),
            key=lambda item: (item[1].event_start_ms, item[0]),
        )
    )
    case_by_condition = {
        case.condition_id: case for case in selected_panel.cases
    }
    responses_by_model = {
        candidate.model_id: {
            case.condition_id: response
            for case, response in zip(
                selected_panel.cases,
                inference.responses,
                strict=True,
            )
        }
        for _host, candidate, inference in validated_candidates
    }
    seen_conditions: set[str] = set()
    baseline_by_delay: dict[int, list[Round27EconomicTrade]] = {
        delay: [] for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    baseline_reasons = {
        delay: Counter(selected_panel.selection_reason_counts)
        for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    }
    ai_by_model = {
        candidate.model_id: {
            delay: [] for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
        }
        for _host, candidate, _inference in validated_candidates
    }
    ai_reasons_by_model = {
        candidate.model_id: {
            delay: Counter(selected_panel.selection_reason_counts)
            for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
        }
        for _host, candidate, _inference in validated_candidates
    }
    for raw_batch in book_batches:
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > config.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 28 AI batch book scope differs")
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
            for _host, candidate, _inference in validated_candidates:
                unchanged_by_latency: dict[int, list[object]] = {}
                response_by_condition = responses_by_model[candidate.model_id]
                candidate_reasons = ai_reasons_by_model[candidate.model_id][delay]
                for case, decision_candidate in zip(
                    batch_cases,
                    candidates,
                    strict=True,
                ):
                    response = response_by_condition[case.condition_id]
                    if response.abstains:
                        candidate_reasons[f"ai_{response.decision}"] += 1
                    else:
                        unchanged_by_latency.setdefault(
                            response.wall_latency_ms,
                            [],
                        ).append(decision_candidate)
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
                    ai_by_model[candidate.model_id][delay].extend(ai_trades)
                    candidate_reasons.update(reasons)
    if seen_conditions != conditions:
        raise ValueError("Round 28 AI batch books do not cover the role")
    baseline_scenarios = {
        int(item["delay_ms"]): item for item in baseline_report["scenarios"]
    }
    replayed_baseline_by_delay: dict[int, dict[str, object]] = {}
    baseline_trades_by_delay: dict[int, tuple[Round27EconomicTrade, ...]] = {}
    for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
        trades = tuple(
            sorted(
                baseline_by_delay[delay],
                key=lambda trade: (
                    trade.event_start_ms,
                    trade.condition_id,
                    trade.decision_time_ms,
                ),
            )
        )
        replayed = _economics._scenario_report(  # noqa: SLF001
            trades=trades,
            candidate_count=len(selected_panel.cases),
            delay_ms=delay,
            evaluated_conditions=ordered_conditions,
            reasons=dict(sorted(baseline_reasons[delay].items())),
            config=config,
        )
        if replayed != baseline_scenarios.get(delay):
            raise ValueError("Round 28 AI batch replayed baseline differs")
        baseline_trades_by_delay[delay] = trades
        replayed_baseline_by_delay[delay] = replayed

    output: list[dict[str, object]] = []
    for host_report, candidate, inference in validated_candidates:
        paired_scenarios: list[dict[str, object]] = []
        for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
            ai_trades = tuple(
                sorted(
                    ai_by_model[candidate.model_id][delay],
                    key=lambda trade: (
                        trade.event_start_ms,
                        trade.condition_id,
                        trade.decision_time_ms,
                    ),
                )
            )
            ai_scenario = _economics._scenario_report(  # noqa: SLF001
                trades=ai_trades,
                candidate_count=len(ai_trades),
                delay_ms=delay,
                evaluated_conditions=ordered_conditions,
                reasons=dict(
                    sorted(ai_reasons_by_model[candidate.model_id][delay].items())
                ),
                config=ai_config,
            )
            paired_scenarios.append(
                _paired_scenario(
                    delay_ms=delay,
                    cases=selected_panel.cases,
                    baseline=replayed_baseline_by_delay[delay],
                    ai=ai_scenario,
                    baseline_trades=baseline_trades_by_delay[delay],
                    ai_trades=ai_trades,
                    inference_eligible=(
                        inference.candidate_eligible_for_matched_evaluation
                    ),
                    config=ai_config,
                )
            )
        all_pass = all(
            scenario["scenario_uplift_gate_passed"] is True
            for scenario in paired_scenarios
        )
        body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "partition_role": selected_panel.partition_role,
            "candidate": asdict(candidate),
            "host_qualification_report_sha256": host_report["report_sha256"],
            "case_panel_sha256": selected_panel.panel_sha256,
            "inference_report_sha256": inference.report_sha256,
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
        output.append(validate_round28_ai_economic_report(body))
    return tuple(output)


__all__ = ["evaluate_round28_ai_candidate_batch"]
