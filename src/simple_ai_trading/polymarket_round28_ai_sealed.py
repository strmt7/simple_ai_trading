"""One-use sealed economics for the nominated Round 28 AI risk veto."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json

from . import polymarket_round27_economics as _economics
from . import polymarket_round28_ai_economics as _ai_economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
    Round27EconomicBookBatch,
    Round27EconomicTrade,
)
from .polymarket_round28_ai_cases import Round28AICasePanel
from .polymarket_round28_ai_contract import POLYMARKET_ROUND28_AI_CONTRACT_SHA256
from .polymarket_round28_ai_economics import (
    POLYMARKET_ROUND28_AI_ECONOMIC_REPORT_SCHEMA_VERSION,
    validate_round28_ai_economic_report,
)
from .polymarket_round28_ai_host import validate_round28_ai_host_report
from .polymarket_round28_ai_inference import validate_round28_ai_inference_report
from .polymarket_round28_ai_selection import (
    Round28AICandidateSelection,
)
from .polymarket_round28_sealed import (
    validate_round28_sealed_economic_report,
)


POLYMARKET_ROUND28_AI_SEALED_TERMINAL_SCHEMA_VERSION = (
    "polymarket-round28-ai-terminal-sealed-result-v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 sealed AI {name} SHA-256 differs")
    return selected


def _trade_sort_key(trade: Round27EconomicTrade) -> tuple[int, str, int]:
    return trade.event_start_ms, trade.condition_id, trade.decision_time_ms


def evaluate_round28_ai_sealed_economics(
    *,
    panel: Round28AICasePanel,
    inference_report: Mapping[str, object],
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
    sealed_round28_economic_report: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    resolution_evidence_sha256: str,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> dict[str, object]:
    """Compare one nominated AI veto with the exact sealed augmented baseline."""

    selected_panel = panel.validated()
    if selected_panel.partition_role != "sealed":
        raise ValueError("Round 28 sealed AI case role differs")
    selected_inference = validate_round28_ai_inference_report(
        inference_report,
        contract=contract,
        host_qualification_report=host_qualification_report,
        panel=selected_panel,
    )
    baseline_parent = validate_round28_sealed_economic_report(
        sealed_round28_economic_report
    )
    baseline_report = baseline_parent.get("augmented_economic_report")
    if not isinstance(baseline_report, Mapping):
        raise ValueError("Round 28 sealed augmented baseline differs")
    config = _ai_economics._economic_config(  # noqa: SLF001
        selected_panel.economic_config
    )
    ai_config = replace(
        config,
        minimum_executed_trades=(
            _ai_economics.POLYMARKET_ROUND28_AI_MINIMUM_FILLED_CONDITIONS
        ),
    ).validated()
    expected_model_name = f"{selected_panel.model_name}:round28_bbo_augmented"
    if (
        baseline_parent.get("selection_claim_sha256")
        != selected_panel.selection_claim_sha256
        or baseline_report.get("model_name") != expected_model_name
        or baseline_report.get("model_sha256") != selected_panel.model_sha256
        or baseline_report.get("config") != selected_panel.economic_config
        or baseline_report.get("candidate_condition_count")
        != len(selected_panel.cases)
        or baseline_report.get("candidate_population_sha256")
        != selected_panel.baseline_candidate_population_sha256
    ):
        raise ValueError("Round 28 sealed AI baseline lineage differs")
    conditions = {market.condition_id for market in markets}
    if (
        len(conditions) != len(markets)
        or len(conditions) != selected_panel.evaluated_condition_count
        or _canonical_sha256(sorted(conditions))
        != selected_panel.evaluated_condition_ids_sha256
        or set(outcomes_up) != conditions
        or any(
            type(value) is not int or value not in {0, 1}
            for value in outcomes_up.values()
        )
    ):
        raise ValueError("Round 28 sealed AI economic population differs")
    market_by_condition = {market.condition_id: market for market in markets}
    ordered_conditions = tuple(
        condition_id
        for condition_id, _market in sorted(
            market_by_condition.items(),
            key=lambda item: (item[1].event_start_ms, item[0]),
        )
    )
    if (books is None) == (book_batches is None):
        raise ValueError("Round 28 sealed AI economics requires one book source")
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
            raise ValueError("Round 28 sealed AI book batch scope differs")
        seen_conditions.update(batch_conditions)
        batch_cases = tuple(
            case_by_condition[condition_id]
            for condition_id in batch.condition_ids
            if condition_id in case_by_condition
        )
        candidates = tuple(
            _ai_economics._candidate_from_case(case)  # noqa: SLF001
            for case in batch_cases
        )
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
        raise ValueError("Round 28 sealed AI book batches do not cover the role")
    baseline_scenarios = {
        int(item["delay_ms"]): item for item in baseline_report["scenarios"]
    }
    paired_scenarios: list[dict[str, object]] = []
    for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
        baseline_trades = tuple(
            sorted(baseline_by_delay[delay], key=_trade_sort_key)
        )
        ai_trades = tuple(sorted(ai_by_delay[delay], key=_trade_sort_key))
        replayed_baseline = _economics._scenario_report(  # noqa: SLF001
            trades=baseline_trades,
            candidate_count=len(selected_panel.cases),
            delay_ms=delay,
            evaluated_conditions=ordered_conditions,
            reasons=dict(sorted(baseline_reasons[delay].items())),
            config=config,
        )
        if replayed_baseline != baseline_scenarios.get(delay):
            raise ValueError("Round 28 sealed AI replayed baseline differs")
        ai_scenario = _economics._scenario_report(  # noqa: SLF001
            trades=ai_trades,
            candidate_count=len(ai_trades),
            delay_ms=delay,
            evaluated_conditions=ordered_conditions,
            reasons=dict(sorted(ai_reasons[delay].items())),
            config=ai_config,
        )
        paired_scenarios.append(
            _ai_economics._paired_scenario(  # noqa: SLF001
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
        "partition_role": "sealed",
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
        "sealed_partition_accessed": True,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return validate_round28_ai_economic_report(body)


def build_round28_ai_sealed_terminal_result(
    *,
    ai_selection: Round28AICandidateSelection,
    panel: Round28AICasePanel,
    inference_report: Mapping[str, object],
    sealed_round28_economic_report: Mapping[str, object],
    sealed_ai_economic_report: Mapping[str, object],
) -> dict[str, object]:
    """Bind the exact nominated model to its one-use sealed uplift result."""

    selection = ai_selection.validated()
    selected_panel = panel.validated()
    baseline = validate_round28_sealed_economic_report(
        sealed_round28_economic_report
    )
    ai_report = validate_round28_ai_economic_report(
        sealed_ai_economic_report
    )
    candidate = ai_report.get("candidate")
    inference_candidate = inference_report.get("candidate")
    if (
        selection.nominated_model_id is None
        or selection.nominated_runtime_digest is None
        or selected_panel.partition_role != "sealed"
        or not isinstance(candidate, Mapping)
        or not isinstance(inference_candidate, Mapping)
        or candidate.get("model_id") != selection.nominated_model_id
        or candidate.get("runtime_digest")
        != selection.nominated_runtime_digest
        or inference_candidate.get("model_id") != selection.nominated_model_id
        or inference_candidate.get("runtime_digest")
        != selection.nominated_runtime_digest
        or ai_report.get("partition_role") != "sealed"
        or ai_report.get("case_panel_sha256") != selected_panel.panel_sha256
        or ai_report.get("inference_report_sha256")
        != inference_report.get("report_sha256")
        or ai_report.get("round28_economic_report_sha256")
        != baseline["report_sha256"]
        or ai_report.get("selection_claim_sha256")
        != selected_panel.selection_claim_sha256
    ):
        raise ValueError("Round 28 sealed AI terminal binding differs")
    passed = ai_report["matched_after_cost_uplift_gate_passed"] is True
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_SEALED_TERMINAL_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "ai_selection_sha256": selection.selection_sha256,
        "nominated_model_id": selection.nominated_model_id,
        "nominated_runtime_digest": selection.nominated_runtime_digest,
        "sealed_case_panel_sha256": selected_panel.panel_sha256,
        "sealed_inference_report_sha256": inference_report["report_sha256"],
        "sealed_round28_economic_report_sha256": baseline["report_sha256"],
        "sealed_ai_economic_report_sha256": ai_report["report_sha256"],
        "sealed_matched_after_cost_uplift_gate_passed": passed,
        "observed_after_cost_ai_uplift": passed,
        "model_prompt_or_threshold_changed_after_selection": False,
        "sealed_partition_accessed": True,
        "edge_claim": False,
        "profitability_claim": False,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "POLYMARKET_ROUND28_AI_SEALED_TERMINAL_SCHEMA_VERSION",
    "build_round28_ai_sealed_terminal_result",
    "evaluate_round28_ai_sealed_economics",
]
