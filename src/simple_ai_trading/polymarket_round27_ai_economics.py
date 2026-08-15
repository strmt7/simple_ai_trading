"""Latency-aware matched economics for the Round 27 AI risk-veto candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import hashlib
import json
from typing import Iterable, Mapping, Sequence

from . import polymarket_round27_economics as _economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_ai_ablation_contract import (
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
    POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS,
)
from .polymarket_round27_ai import POLYMARKET_ROUND27_AI_HOST_CANDIDATES
from .polymarket_round27_ai_cases import Round27AICase, Round27AICasePanel
from .polymarket_round27_ai_inference import Round27AIInferenceReport
from .polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD,
    POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256,
)
from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    Round27EconomicTrade,
)


POLYMARKET_ROUND27_AI_ECONOMIC_REPORT_SCHEMA_VERSION = (
    "polymarket-round27-ai-matched-economic-report-v1"
)
POLYMARKET_ROUND27_AI_SELECTION_SCHEMA_VERSION = (
    "polymarket-round27-ai-candidate-selection-v1"
)
_FIXED_DELAYS_MS = (250, 500, 1_000, 2_000)
_UPLIFT_GATE_NAMES = frozenset(
    {
        "inference_candidate_eligible",
        "baseline_scenario_gate_passed",
        "minimum_ai_filled_conditions_met",
        "minimum_profitable_ai_conditions_met",
        "ai_net_pnl_positive",
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


def _strict_baseline_report(
    value: Mapping[str, object],
    panel: Round27AICasePanel,
) -> dict[str, object]:
    report = dict(value)
    claimed = str(report.pop("report_sha256", ""))
    scenarios = report.get("scenarios")
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version") != POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION
        or report.get("partition_role") != panel.partition_role
        or report.get("model_name") != panel.model_name
        or report.get("model_sha256") != panel.model_sha256
        or report.get("source_audit_sha256") != panel.source_audit_sha256
        or report.get("config") != panel.economic_config
        or report.get("candidate_condition_count") != len(panel.cases)
        or report.get("candidate_population_sha256")
        != panel.baseline_candidate_population_sha256
        or report.get(POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD)
        != POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
        or not isinstance(scenarios, list)
        or [item.get("delay_ms") for item in scenarios if isinstance(item, Mapping)]
        != list(_FIXED_DELAYS_MS)
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
        raise ValueError("Round 27 AI baseline economic report differs")
    return {**report, "report_sha256": claimed}


def _candidate_from_case(case: Round27AICase) -> object:
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


def _decimal(value: object) -> Decimal:
    selected = Decimal(str(value))
    if not selected.is_finite():
        raise ValueError("Round 27 AI economic decimal differs")
    return selected


def _paired_scenario(
    *,
    delay_ms: int,
    cases: Sequence[Round27AICase],
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
        seed=27_027 + delay_ms,
    )
    baseline_markout = baseline.get("mean_markout_pnl_per_contract")
    ai_markout = ai.get("mean_markout_pnl_per_contract")
    markout_not_worse = bool(
        baseline_markout is not None
        and ai_markout is not None
        and _decimal(ai_markout) >= _decimal(baseline_markout)
    )
    baseline_drawdown = _decimal(baseline["maximum_drawdown_fraction"])
    ai_drawdown = _decimal(ai["maximum_drawdown_fraction"])
    mean_delta = _decimal(bootstrap["mean_net_pnl_quote"])
    checks = {
        "inference_candidate_eligible": inference_eligible,
        "baseline_scenario_gate_passed": (
            baseline.get("scenario_edge_gate_passed") is True
        ),
        "minimum_ai_filled_conditions_met": int(ai["filled_order_count"])
        >= POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS,
        "minimum_profitable_ai_conditions_met": int(ai["profitable_condition_count"])
        >= config.minimum_profitable_conditions,
        "ai_net_pnl_positive": _decimal(ai["net_pnl_quote"]) > 0,
        "paired_mean_net_pnl_delta_positive": mean_delta > 0,
        "paired_condition_bootstrap_lower_bound_positive": bool(
            bootstrap["eligible"]
        )
        and _decimal(bootstrap["ci95_lower"]) > 0,
        "maximum_drawdown_not_worse": ai_drawdown <= baseline_drawdown,
        "mean_adverse_markout_not_worse": markout_not_worse,
        "no_unknown_execution_state": int(ai["unknown_order_count"]) == 0,
    }
    body: dict[str, object] = {
        "base_delay_ms": delay_ms,
        "baseline": dict(baseline),
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


def evaluate_round27_ai_matched_economics(
    *,
    panel: Round27AICasePanel,
    inference_report: Round27AIInferenceReport,
    baseline_economic_report: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    resolution_evidence_sha256: str,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> dict[str, object]:
    """Compare one frozen AI veto report with its exact economic baseline."""

    selected_panel = panel.validated()
    selected_inference = inference_report.validated()
    baseline_report = _strict_baseline_report(
        baseline_economic_report,
        selected_panel,
    )
    if (
        selected_inference.case_panel_sha256 != selected_panel.panel_sha256
        or [response.case_sha256 for response in selected_inference.responses]
        != [case.case_sha256 for case in selected_panel.cases]
    ):
        raise ValueError("Round 27 AI inference population differs")
    config = Round27EconomicConfig(
        **{
            key: (
                tuple(value)
                if key == "delays_ms" and isinstance(value, list)
                else Decimal(value)
                if key
                in {
                    "minimum_expected_edge_per_contract",
                    "maximum_entry_cost_quote",
                    "initial_capital_quote",
                }
                else value
            )
            for key, value in selected_panel.economic_config.items()
        }
    ).validated()
    ai_config = replace(
        config,
        minimum_executed_trades=(
            POLYMARKET_ROUND27_AI_MINIMUM_FILLED_CONDITIONS
        ),
    ).validated()
    conditions = {market.condition_id for market in markets}
    if (
        len(conditions) != selected_panel.evaluated_condition_count
        or _canonical_sha256(sorted(conditions))
        != selected_panel.evaluated_condition_ids_sha256
        or set(outcomes_up) != conditions
        or any(type(value) is not int or value not in {0, 1} for value in outcomes_up.values())
    ):
        raise ValueError("Round 27 AI economic condition population differs")
    market_by_condition = {market.condition_id: market for market in markets}
    ordered_conditions = tuple(
        condition_id
        for condition_id, _market in sorted(
            market_by_condition.items(),
            key=lambda item: (item[1].event_start_ms, item[0]),
        )
    )
    if (books is None) == (book_batches is None):
        raise ValueError("Round 27 AI economics requires exactly one book source")
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
        delay: [] for delay in _FIXED_DELAYS_MS
    }
    ai_by_delay: dict[int, list[Round27EconomicTrade]] = {
        delay: [] for delay in _FIXED_DELAYS_MS
    }
    baseline_reasons: dict[int, Counter[str]] = {
        delay: Counter(selected_panel.selection_reason_counts)
        for delay in _FIXED_DELAYS_MS
    }
    ai_reasons: dict[int, Counter[str]] = {
        delay: Counter(selected_panel.selection_reason_counts)
        for delay in _FIXED_DELAYS_MS
    }
    for raw_batch in batches:
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > config.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 27 AI economic book batch scope differs")
        seen_conditions.update(batch_conditions)
        batch_cases = tuple(
            case_by_condition[condition_id]
            for condition_id in batch.condition_ids
            if condition_id in case_by_condition
        )
        batch_candidates = tuple(_candidate_from_case(case) for case in batch_cases)
        batch_markets = {
            condition_id: market_by_condition[condition_id]
            for condition_id in batch.condition_ids
        }
        index = _economics._BookIndex(batch.books)  # noqa: SLF001
        for delay in _FIXED_DELAYS_MS:
            baseline_trades, reasons = _economics._execute_candidate_trades(  # noqa: SLF001
                candidates=batch_candidates,
                delay_ms=delay,
                market_by_condition=batch_markets,
                outcomes_up=outcomes_up,
                index=index,
                config=config,
            )
            baseline_by_delay[delay].extend(baseline_trades)
            baseline_reasons[delay].update(reasons)
            unchanged_by_latency: dict[int, list[object]] = {}
            for case, candidate in zip(batch_cases, batch_candidates, strict=True):
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
        raise ValueError("Round 27 AI economic book batches do not cover the role")
    baseline_scenarios = {
        int(item["delay_ms"]): item for item in baseline_report["scenarios"]
    }
    paired_scenarios: list[dict[str, object]] = []
    for delay in _FIXED_DELAYS_MS:
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
            raise ValueError("Round 27 AI replayed baseline scenario differs")
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
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_AI_ECONOMIC_REPORT_SCHEMA_VERSION,
        "ablation_contract_sha256": (
            POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        ),
        POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD: (
            POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
        ),
        "partition_role": selected_panel.partition_role,
        "candidate": dict(selected_inference.candidate),
        "case_panel_sha256": selected_panel.panel_sha256,
        "inference_report_sha256": selected_inference.report_sha256,
        "baseline_economic_report_sha256": baseline_report["report_sha256"],
        "source_audit_sha256": selected_panel.source_audit_sha256,
        "resolution_evidence_sha256": str(resolution_evidence_sha256),
        "matched_candidate_condition_count": len(selected_panel.cases),
        "paired_scenarios": paired_scenarios,
        "primary_500ms_uplift_gate_passed": next(
            scenario["scenario_uplift_gate_passed"]
            for scenario in paired_scenarios
            if scenario["base_delay_ms"] == 500
        ),
        "all_delay_scenarios_uplift_gate_passed": all_pass,
        "matched_after_cost_uplift_gate_passed": all_pass,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round27_ai_economic_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate one complete persisted matched economic report."""

    report = dict(value)
    claimed = str(report.pop("report_sha256", ""))
    candidate = report.get("candidate")
    scenarios = report.get("paired_scenarios")
    expected_candidates = {
        item.model_id: asdict(item)
        for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
    }
    scenarios_valid = bool(
        isinstance(scenarios, list)
        and len(scenarios) == len(_FIXED_DELAYS_MS)
    )
    if scenarios_valid:
        for item, delay in zip(scenarios, _FIXED_DELAYS_MS, strict=True):
            if not isinstance(item, Mapping):
                scenarios_valid = False
                break
            paired_body = dict(item)
            paired_claimed = str(paired_body.pop("paired_scenario_sha256", ""))
            checks = item.get("gate_checks")
            baseline = item.get("baseline")
            ai = item.get("ai")
            if not isinstance(baseline, Mapping) or not isinstance(ai, Mapping):
                scenarios_valid = False
                break
            baseline_body = dict(baseline)
            baseline_claimed = str(baseline_body.pop("scenario_sha256", ""))
            ai_body = dict(ai)
            ai_claimed = str(ai_body.pop("scenario_sha256", ""))
            if (
                item.get("base_delay_ms") != delay
                or not isinstance(checks, Mapping)
                or set(checks) != _UPLIFT_GATE_NAMES
                or any(type(value) is not bool for value in checks.values())
                or item.get("scenario_uplift_gate_passed")
                is not all(value is True for value in checks.values())
                or paired_claimed != _canonical_sha256(paired_body)
                or baseline_claimed != _canonical_sha256(baseline_body)
                or ai_claimed != _canonical_sha256(ai_body)
            ):
                scenarios_valid = False
                break
    all_pass = bool(
        scenarios_valid
        and all(item["scenario_uplift_gate_passed"] is True for item in scenarios)
    )
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version")
        != POLYMARKET_ROUND27_AI_ECONOMIC_REPORT_SCHEMA_VERSION
        or report.get("ablation_contract_sha256")
        != POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        or report.get(POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_BINDING_FIELD)
        != POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
        or report.get("partition_role") not in {"selection", "sealed"}
        or not isinstance(candidate, Mapping)
        or candidate != expected_candidates.get(str(candidate.get("model_id")))
        or not scenarios_valid
        or [
            item.get("base_delay_ms")
            for item in scenarios
            if isinstance(item, Mapping)
        ]
        != list(_FIXED_DELAYS_MS)
        or report.get("primary_500ms_uplift_gate_passed")
        is not bool(scenarios[1].get("scenario_uplift_gate_passed"))
        or report.get("all_delay_scenarios_uplift_gate_passed") is not all_pass
        or report.get("matched_after_cost_uplift_gate_passed") is not all_pass
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
        raise ValueError("Round 27 AI matched economic report differs")
    return {**report, "report_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round27AICandidateSelection:
    case_panel_sha256: str
    baseline_economic_report_sha256: str
    candidate_report_sha256: tuple[str, ...]
    nominated_model_id: str | None
    nominated_runtime_digest: str | None
    nominated_report_sha256: str | None
    selection_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_AI_SELECTION_SCHEMA_VERSION,
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
            "case_panel_sha256": self.case_panel_sha256,
            "baseline_economic_report_sha256": (
                self.baseline_economic_report_sha256
            ),
            "candidate_report_sha256": list(self.candidate_report_sha256),
            "nominated_model_id": self.nominated_model_id,
            "nominated_runtime_digest": self.nominated_runtime_digest,
            "nominated_report_sha256": self.nominated_report_sha256,
            "selection_partition_only": True,
            "sealed_partition_accessed": False,
            "post_selection_retuning_allowed": False,
            "edge_claim": False,
            "profitability_claim": False,
            "orders_submitted": False,
            "trading_authority": False,
        }

    def validated(self) -> "Round27AICandidateSelection":
        candidates = {
            item.model_id: item for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
        }
        nominated_candidate = candidates.get(self.nominated_model_id or "")
        if (
            len(self.case_panel_sha256) != 64
            or len(self.baseline_economic_report_sha256) != 64
            or len(self.candidate_report_sha256) != 2
            or len(set(self.candidate_report_sha256)) != 2
            or any(len(value) != 64 for value in self.candidate_report_sha256)
            or (self.nominated_model_id is None)
            != (self.nominated_runtime_digest is None)
            or (self.nominated_model_id is None)
            != (self.nominated_report_sha256 is None)
            or (
                self.nominated_model_id is not None
                and (
                    nominated_candidate is None
                    or self.nominated_runtime_digest
                    != nominated_candidate.runtime_digest
                    or self.nominated_report_sha256
                    not in self.candidate_report_sha256
                )
            )
            or self.selection_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 27 AI candidate selection differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "selection_sha256": self.selection_sha256,
        }


def round27_ai_candidate_selection_from_mapping(
    value: Mapping[str, object],
) -> Round27AICandidateSelection:
    """Reconstruct one strict persisted development nomination."""

    payload = dict(value)
    expected = {
        "schema_version",
        "ablation_contract_sha256",
        "case_panel_sha256",
        "baseline_economic_report_sha256",
        "candidate_report_sha256",
        "nominated_model_id",
        "nominated_runtime_digest",
        "nominated_report_sha256",
        "selection_partition_only",
        "sealed_partition_accessed",
        "post_selection_retuning_allowed",
        "edge_claim",
        "profitability_claim",
        "orders_submitted",
        "trading_authority",
        "selection_sha256",
    }
    false_fields = (
        "sealed_partition_accessed",
        "post_selection_retuning_allowed",
        "edge_claim",
        "profitability_claim",
        "orders_submitted",
        "trading_authority",
    )
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_AI_SELECTION_SCHEMA_VERSION
        or payload.get("ablation_contract_sha256")
        != POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        or payload.get("selection_partition_only") is not True
        or any(payload.get(field) is not False for field in false_fields)
        or not isinstance(payload.get("candidate_report_sha256"), list)
        or any(
            value is not None and not isinstance(value, str)
            for value in (
                payload.get("nominated_model_id"),
                payload.get("nominated_runtime_digest"),
                payload.get("nominated_report_sha256"),
            )
        )
    ):
        raise ValueError("Round 27 persisted AI candidate selection differs")
    return Round27AICandidateSelection(
        case_panel_sha256=str(payload["case_panel_sha256"]),
        baseline_economic_report_sha256=str(
            payload["baseline_economic_report_sha256"]
        ),
        candidate_report_sha256=tuple(payload["candidate_report_sha256"]),
        nominated_model_id=payload["nominated_model_id"],
        nominated_runtime_digest=payload["nominated_runtime_digest"],
        nominated_report_sha256=payload["nominated_report_sha256"],
        selection_sha256=str(payload["selection_sha256"]),
    ).validated()


def _candidate_rank(report: Mapping[str, object]) -> tuple[Decimal, Decimal, Decimal, str]:
    scenarios = report["paired_scenarios"]
    primary = next(item for item in scenarios if item["base_delay_ms"] == 500)
    lower = _decimal(primary["paired_condition_bootstrap"]["ci95_lower"])
    worst_mean = min(
        _decimal(item["paired_mean_net_pnl_delta_quote"]) for item in scenarios
    )
    worst_drawdown = max(
        _decimal(item["maximum_drawdown_delta_fraction"]) for item in scenarios
    )
    return (
        -lower,
        -worst_mean,
        worst_drawdown,
        str(report["candidate"]["model_id"]),
    )


def select_round27_ai_candidate(
    reports: Sequence[Mapping[str, object]],
) -> Round27AICandidateSelection:
    """Nominate at most one development candidate under the frozen ranking."""

    selected = tuple(validate_round27_ai_economic_report(report) for report in reports)
    if len(selected) != 2:
        raise ValueError("Round 27 AI selection requires both candidate reports")
    if any(report.get("partition_role") != "selection" for report in selected):
            raise ValueError("Round 27 AI selection report differs")
    expected_candidates = {
        candidate.model_id: asdict(candidate)
        for candidate in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
    }
    if any(not isinstance(report.get("candidate"), Mapping) for report in selected):
        raise ValueError("Round 27 AI matched selection candidate differs")
    if (
        len({report["candidate"]["model_id"] for report in selected}) != 2
        or any(
            report["candidate"]
            != expected_candidates.get(str(report["candidate"].get("model_id")))
            for report in selected
        )
        or len({report["case_panel_sha256"] for report in selected}) != 1
        or len({report["baseline_economic_report_sha256"] for report in selected})
        != 1
    ):
        raise ValueError("Round 27 AI matched selection population differs")
    qualified = tuple(
        report
        for report in selected
        if report["matched_after_cost_uplift_gate_passed"] is True
    )
    nominated = None if not qualified else min(qualified, key=_candidate_rank)
    provisional = Round27AICandidateSelection(
        case_panel_sha256=str(selected[0]["case_panel_sha256"]),
        baseline_economic_report_sha256=str(
            selected[0]["baseline_economic_report_sha256"]
        ),
        candidate_report_sha256=tuple(
            sorted(str(report["report_sha256"]) for report in selected)
        ),
        nominated_model_id=(
            None if nominated is None else str(nominated["candidate"]["model_id"])
        ),
        nominated_runtime_digest=(
            None
            if nominated is None
            else str(nominated["candidate"]["runtime_digest"])
        ),
        nominated_report_sha256=(
            None if nominated is None else str(nominated["report_sha256"])
        ),
        selection_sha256="",
    )
    return replace(
        provisional,
        selection_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND27_AI_ECONOMIC_REPORT_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_AI_SELECTION_SCHEMA_VERSION",
    "Round27AICandidateSelection",
    "evaluate_round27_ai_matched_economics",
    "round27_ai_candidate_selection_from_mapping",
    "select_round27_ai_candidate",
    "validate_round27_ai_economic_report",
]
