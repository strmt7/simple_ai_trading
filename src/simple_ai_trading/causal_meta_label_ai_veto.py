"""Local finance-LLM veto ablation for a viable Round 40 meta-label policy."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .ai_trade_veto import (
    _canonical_sha256,
    _failed_decision,
    _model_metadata,
    _request_json,
)
from .causal_meta_label_model import CausalMetaCandidate
from .cross_asset_cost_data import SYMBOLS
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .rolling_refit_ai_veto import (
    RollingAICaseResult,
    RollingAIModelReport,
    RollingAITradeCase,
    _binomial_upper_tail,
    _decision_schema,
    _feature_indices,
    _max_drawdown,
    _parse_batch_response,
    _profit_factor,
    _stationary_bootstrap_mean,
)


BATCH_SIZE = 12
MAX_CASES = 180
HASH_SAMPLE_MODULUS = 2
EVALUATION_START_MS = int(
    datetime(2024, 7, 1, tzinfo=UTC).timestamp() * 1000
)
EVALUATION_DAYS = 184
DEFAULT_MODEL = "dianjin-r1:7b"


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def case_is_sampled(case_id: str) -> bool:
    """Apply the frozen outcome-independent 50% case sampling rule."""

    if len(case_id) != 64:
        raise ValueError("Round 40 AI case identity is not SHA-256")
    return int(case_id[:16], 16) % HASH_SAMPLE_MODULUS == 0


def meta_case_set_sha256(cases: Sequence[RollingAITradeCase]) -> str:
    return _canonical_sha256([case.identity_payload() for case in cases])


def build_meta_label_ai_cases(
    dataset: DerivativesHurdleDataset,
    candidate: CausalMetaCandidate | None,
) -> tuple[RollingAITradeCase, ...]:
    """Build chronological, outcome-independent sampled cases from live actions."""

    if candidate is None:
        return ()
    feature_indices = _feature_indices(dataset.feature_names)
    rows = candidate.evaluation.selected_indices
    order = np.argsort(dataset.decision_time_ms[rows], kind="stable")
    rows = rows[order]
    directions = candidate.evaluation.selected_direction[order]
    outcomes = candidate.evaluation.net_return_bps[order]
    completed: list[tuple[int, int, float]] = []
    cases: list[RollingAITradeCase] = []
    for raw_row, raw_direction, raw_outcome in zip(
        rows, directions, outcomes, strict=True
    ):
        row = int(raw_row)
        direction = int(raw_direction)
        outcome = float(raw_outcome)
        timestamp_ms = int(dataset.decision_time_ms[row])
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
        relative_day = int((timestamp_ms - EVALUATION_START_MS) // 86_400_000)
        month_index = (timestamp.year - 2024) * 12 + timestamp.month - 7
        if not 0 <= relative_day < EVALUATION_DAYS or not 0 <= month_index < 6:
            raise ValueError("Round 40 AI action lies outside 2024-H2")
        month = f"{timestamp.year:04d}-{timestamp.month:02d}"
        threshold = candidate.monthly_thresholds.get(month)
        if threshold is None:
            raise ValueError("Round 40 AI action came from an unselected month")
        symbol_index = int(dataset.symbol_index[row])
        prior = [
            value
            for settled_time, prior_symbol, value in completed
            if prior_symbol == symbol_index and settled_time < timestamp_ms
        ][-20:]
        loss_streak = 0
        for value in reversed(prior):
            if value >= 0.0:
                break
            loss_streak += 1
        prior_symbol_counts = {
            symbol: sum(
                prior_symbol == index
                for _, prior_symbol, _ in completed
            )
            for index, symbol in enumerate(SYMBOLS)
        }
        p_short = float(candidate.primary_probabilities[row, 0])
        p_abstain = float(candidate.primary_probabilities[row, 1])
        p_long = float(candidate.primary_probabilities[row, 2])
        action_probability = p_long if direction > 0 else p_short
        meta_probability = float(candidate.meta_probabilities[row])
        payload = {
            "schema_version": "causal-meta-label-ai-veto-case-v1",
            "relative_day_index": relative_day,
            "refit_sequence_index": month_index,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": candidate.horizon_minutes,
            "proposed_direction": "long" if direction > 0 else "short",
            "primary_probabilities": {
                "short": round(p_short, 6),
                "abstain": round(p_abstain, 6),
                "long": round(p_long, 6),
            },
            "primary_direction_margin": round(abs(p_long - p_short), 6),
            "meta_profitability_probability": round(meta_probability, 6),
            "meta_probability_threshold": threshold[0],
            "primary_margin_threshold": threshold[1],
            "market_and_contract_state": {
                dataset.feature_names[index]: round(
                    float(dataset.features[row, index]), 5
                )
                for index in feature_indices
            },
            "risk_state": {
                "open_position_count": 0,
                "prior_completed_symbol_actions": len(prior),
                "trailing_symbol_action_mean_net_bps": round(
                    float(np.mean(prior)) if prior else 0.0, 4
                ),
                "trailing_symbol_action_positive_rate": round(
                    float(np.mean(np.asarray(prior) > 0.0)) if prior else 0.0,
                    4,
                ),
                "consecutive_symbol_losses": loss_streak,
                "prior_action_counts_by_symbol": prior_symbol_counts,
                "maximum_risk_multiplier": 1.0,
            },
        }
        identity = {
            "relative_day_index": relative_day,
            "refit_sequence_index": month_index,
            "asset": SYMBOLS[symbol_index],
            "dataset_row": row,
            "candidate_id": candidate.candidate_id,
            "prompt": payload,
        }
        case_id = _canonical_sha256(identity)
        if case_is_sampled(case_id) and len(cases) < MAX_CASES:
            cases.append(
                RollingAITradeCase(
                    case_id=case_id,
                    candidate_id=candidate.candidate_id,
                    dataset_row=row,
                    relative_day_index=relative_day,
                    refit_sequence_index=month_index,
                    decision_time_ms=timestamp_ms,
                    symbol=SYMBOLS[symbol_index],
                    horizon_minutes=candidate.horizon_minutes,
                    direction="long" if direction > 0 else "short",
                    action_probability=action_probability,
                    direction_probability_margin=abs(p_long - p_short),
                    outcome_net_bps=outcome,
                    prompt_payload=payload,
                )
            )
        completed.append(
            (
                timestamp_ms + (candidate.horizon_minutes + 1) * 60_000,
                symbol_index,
                outcome,
            )
        )
    return tuple(cases)


def _batch_prompt(cases: Sequence[RollingAITradeCase]) -> str:
    payload = [
        {"case_index": index, "evidence": dict(case.prompt_payload)}
        for index, case in enumerate(cases)
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return (
        "You are a fail-closed risk coordinator for autonomous crypto day-trading research. "
        "A causal machine-learning stack has already proposed each action and estimated its "
        "after-cost profitability probability. Evaluate every case using only the structured "
        "evidence. Outcomes are hidden. You cannot create a trade, reverse direction, increase "
        "risk, infer dates or prices, retrieve news, or treat missing evidence as favorable. "
        "Approve only coherent evidence; veto weak or contradictory evidence; use cooldown for "
        "unstable regimes or loss control. Return every case_index exactly once.\n"
        f"CASES={serialized}"
    )


def benchmark_meta_label_ai_model(
    cases: Sequence[RollingAITradeCase],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
    progress: ProgressCallback | None = None,
) -> RollingAIModelReport:
    """Run the frozen finance model and measure matched daily veto uplift."""

    if not cases or len(cases) > MAX_CASES:
        raise ValueError("Round 40 AI case count is empty or exceeds its maximum")
    digest, metadata_sha = _model_metadata(
        base_url, model, timeout_seconds=timeout_seconds
    )
    frozen_case_sha = meta_case_set_sha256(cases)
    results: list[RollingAICaseResult] = []
    batch_latencies: list[float] = []
    batches = [
        cases[index : index + BATCH_SIZE]
        for index in range(0, len(cases), BATCH_SIZE)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        expected_ids = [case.case_id for case in batch]
        prompt = _batch_prompt(batch)
        request = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON matching the schema. Never create, reverse, "
                        "or increase an action."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": _decision_schema(expected_ids),
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 12_288,
                "num_predict": 900,
                "seed": 4001,
            },
        }
        started = time.perf_counter()
        raw: object = {}
        failure = ""
        try:
            raw = _request_json(
                f"{base_url.rstrip('/')}/api/chat",
                request,
                timeout_seconds=timeout_seconds,
            )
            decisions = {
                case_id: replace(
                    decision,
                    summary="Compact structured Round 40 finance-model veto decision.",
                )
                for case_id, decision in _parse_batch_response(
                    raw, expected_ids
                ).items()
            }
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            decisions = {case.case_id: _failed_decision(failure) for case in batch}
        latency = time.perf_counter() - started
        batch_latencies.append(latency)
        response_sha = _canonical_sha256(raw)
        for case in batch:
            decision = decisions[case.case_id]
            results.append(
                RollingAICaseResult(
                    case_id=case.case_id,
                    model=model,
                    batch_index=batch_index,
                    batch_latency_seconds=latency,
                    prompt_sha256=_canonical_sha256(prompt),
                    response_sha256=response_sha,
                    decision=decision,
                    baseline_net_bps=case.outcome_net_bps,
                    ai_net_bps=case.outcome_net_bps * decision.risk_multiplier,
                )
            )
        if progress is not None:
            progress(
                "round40_ai_veto",
                {
                    "model": model,
                    "batch": batch_index,
                    "batches": len(batches),
                    "cases": len(batch),
                    "valid": not failure,
                    "latency_seconds": round(latency, 3),
                },
            )
    by_case = {case.case_id: case for case in cases}
    baseline_daily = np.zeros(EVALUATION_DAYS, dtype=np.float64)
    ai_daily = np.zeros(EVALUATION_DAYS, dtype=np.float64)
    baseline_active = np.zeros(EVALUATION_DAYS, dtype=bool)
    retained_active = np.zeros(EVALUATION_DAYS, dtype=bool)
    retained_by_symbol = {symbol: 0 for symbol in SYMBOLS}
    month_totals = np.zeros(6, dtype=np.float64)
    retained_values: list[float] = []
    for result in results:
        case = by_case[result.case_id]
        baseline_daily[case.relative_day_index] += result.baseline_net_bps
        ai_daily[case.relative_day_index] += result.ai_net_bps
        baseline_active[case.relative_day_index] = True
        if result.decision.risk_multiplier > 0.0:
            retained_active[case.relative_day_index] = True
            retained_by_symbol[case.symbol] += 1
            retained_values.append(result.ai_net_bps)
            month_totals[case.refit_sequence_index] += result.ai_net_bps
    retained = np.asarray(retained_values, dtype=np.float64)
    delta = ai_daily - baseline_daily
    nonzero_delta = delta[delta != 0.0]
    positive = int(np.count_nonzero(nonzero_delta > 0.0))
    negative = int(np.count_nonzero(nonzero_delta < 0.0))
    trials = positive + negative
    positive_rate = positive / trials if trials else 0.0
    lower, median, upper = _stationary_bootstrap_mean(
        delta, seed=4021
    )
    baseline_values = np.asarray(
        [result.baseline_net_bps for result in results], dtype=np.float64
    )
    ai_values = np.asarray(
        [result.ai_net_bps for result in results], dtype=np.float64
    )
    retained_total = int(retained.size)
    concentration = (
        max(retained_by_symbol.values()) / retained_total if retained_total else 1.0
    )
    active_days = int(np.count_nonzero(retained_active))
    profit_factor = _profit_factor(retained)
    valid = sum(result.decision.valid for result in results)
    sign_p = _binomial_upper_tail(trials, positive)
    reasons: list[str] = []
    if retained_total < 90:
        reasons.append("ai_retained_trades<90")
    for symbol in SYMBOLS:
        if retained_by_symbol[symbol] < 20:
            reasons.append(f"{symbol}_ai_retained_trades<20")
    if active_days < 45:
        reasons.append("ai_retained_active_days<45")
    if concentration > 0.50:
        reasons.append("ai_retained_symbol_fraction>0.50")
    if valid != len(cases):
        reasons.append("ai_provider_or_schema_failures")
    if positive_rate < 0.55:
        reasons.append("ai_positive_daily_delta_rate<0.55")
    if sign_p > 0.05:
        reasons.append("ai_exact_sign_test_p_value>0.05")
    if lower <= 0.0:
        reasons.append("ai_block_bootstrap_delta_lower_95<=0")
    if retained_total == 0 or float(np.mean(retained)) <= 0.0:
        reasons.append("ai_mean_retained_net_bps<=0")
    if profit_factor is None or profit_factor < 1.05:
        reasons.append("ai_profit_factor<1.05")
    if float(np.median(month_totals)) <= 0.0:
        reasons.append("ai_median_monthly_net_bps<=0")
    negative_month_fraction = float(np.mean(month_totals < 0.0))
    if negative_month_fraction > 0.45:
        reasons.append("ai_negative_month_fraction>0.45")
    baseline_drawdown = _max_drawdown(baseline_daily)
    ai_drawdown = _max_drawdown(ai_daily)
    if ai_drawdown > baseline_drawdown + 1e-9:
        reasons.append("ai_max_drawdown_worse_than_baseline")
    return RollingAIModelReport(
        model=model,
        model_digest=digest,
        model_metadata_sha256=metadata_sha,
        case_set_sha256=frozen_case_sha,
        cases=len(cases),
        batches=len(batches),
        valid_responses=valid,
        approvals=sum(result.decision.action == "approve" for result in results),
        vetoes=sum(result.decision.action == "veto" for result in results),
        cooldowns=sum(result.decision.action == "cooldown" for result in results),
        provider_failures=len(cases) - valid,
        average_batch_latency_seconds=float(np.mean(batch_latencies)),
        average_case_latency_seconds=float(sum(batch_latencies) / len(cases)),
        retained_trades_by_symbol=retained_by_symbol,
        retained_active_days=active_days,
        maximum_retained_symbol_fraction=concentration,
        baseline_total_net_bps=float(np.sum(baseline_values)),
        ai_total_net_bps=float(np.sum(ai_values)),
        baseline_mean_case_net_bps=float(np.mean(baseline_values)),
        ai_mean_retained_case_net_bps=(
            float(np.mean(retained)) if retained_total else 0.0
        ),
        ai_profit_factor=profit_factor,
        ai_median_monthly_net_bps=float(np.median(month_totals)),
        ai_negative_month_fraction=negative_month_fraction,
        baseline_max_drawdown_bps=baseline_drawdown,
        ai_max_drawdown_bps=ai_drawdown,
        matched_days=int(np.count_nonzero(baseline_active)),
        positive_daily_delta_count=positive,
        negative_daily_delta_count=negative,
        positive_daily_delta_rate=positive_rate,
        exact_sign_test_p_value=sign_p,
        mean_daily_delta_bps=float(np.mean(delta)),
        bootstrap_delta_lower_95_bps=lower,
        bootstrap_delta_median_bps=median,
        bootstrap_delta_upper_95_bps=upper,
        uplift_gate_passed=not reasons,
        uplift_gate_reasons=tuple(reasons),
        results=tuple(results),
    )


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_MODEL",
    "HASH_SAMPLE_MODULUS",
    "MAX_CASES",
    "benchmark_meta_label_ai_model",
    "build_meta_label_ai_cases",
    "case_is_sampled",
    "meta_case_set_sha256",
]
