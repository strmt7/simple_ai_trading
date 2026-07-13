"""Batched causal local-LLM veto ablation for Round 39 rolling candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .ai_trade_veto import (
    AI_MODELS,
    AIVetoDecision,
    _bounded_float,
    _canonical_sha256,
    _failed_decision,
    _model_metadata,
    _request_json,
)
from .cross_asset_cost_data import SYMBOLS
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .rolling_refit_model import RollingSupportCandidate


BATCH_SIZE = 12
MAX_CASES = 180
EVALUATION_START_MS = int(
    datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000
)
EVALUATION_DAYS = 181


@dataclass(frozen=True)
class RollingAITradeCase:
    case_id: str
    candidate_id: str
    dataset_row: int
    relative_day_index: int
    refit_sequence_index: int
    decision_time_ms: int
    symbol: str
    horizon_minutes: int
    direction: str
    action_probability: float
    direction_probability_margin: float
    outcome_net_bps: float
    prompt_payload: Mapping[str, object]

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "relative_day_index": self.relative_day_index,
            "refit_sequence_index": self.refit_sequence_index,
            "symbol": self.symbol,
            "horizon_minutes": self.horizon_minutes,
            "direction": self.direction,
            "action_probability": self.action_probability,
            "direction_probability_margin": self.direction_probability_margin,
            "prompt_payload": dict(self.prompt_payload),
        }

    def evidence_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "outcome_net_bps": self.outcome_net_bps}


@dataclass(frozen=True)
class RollingAICaseResult:
    case_id: str
    model: str
    batch_index: int
    batch_latency_seconds: float
    prompt_sha256: str
    response_sha256: str
    decision: AIVetoDecision
    baseline_net_bps: float
    ai_net_bps: float

    def asdict(self) -> dict[str, object]:
        return {**asdict(self), "decision": self.decision.asdict()}


@dataclass(frozen=True)
class RollingAIModelReport:
    model: str
    model_digest: str
    model_metadata_sha256: str
    case_set_sha256: str
    cases: int
    batches: int
    valid_responses: int
    approvals: int
    vetoes: int
    cooldowns: int
    provider_failures: int
    average_batch_latency_seconds: float
    average_case_latency_seconds: float
    retained_trades_by_symbol: Mapping[str, int]
    retained_active_days: int
    maximum_retained_symbol_fraction: float
    baseline_total_net_bps: float
    ai_total_net_bps: float
    baseline_mean_case_net_bps: float
    ai_mean_retained_case_net_bps: float
    ai_profit_factor: float | None
    ai_median_monthly_net_bps: float
    ai_negative_month_fraction: float
    baseline_max_drawdown_bps: float
    ai_max_drawdown_bps: float
    matched_days: int
    positive_daily_delta_count: int
    negative_daily_delta_count: int
    positive_daily_delta_rate: float
    exact_sign_test_p_value: float
    mean_daily_delta_bps: float
    bootstrap_delta_lower_95_bps: float
    bootstrap_delta_median_bps: float
    bootstrap_delta_upper_95_bps: float
    uplift_gate_passed: bool
    uplift_gate_reasons: tuple[str, ...]
    results: tuple[RollingAICaseResult, ...]

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "retained_trades_by_symbol": dict(self.retained_trades_by_symbol),
            "uplift_gate_reasons": list(self.uplift_gate_reasons),
            "results": [item.asdict() for item in self.results],
        }


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _feature_indices(feature_names: Sequence[str]) -> tuple[int, ...]:
    wanted = (
        "target_return_5m_bps",
        "target_return_15m_bps",
        "target_return_60m_bps",
        "target_realized_volatility_60m_bps",
        "target_realized_volatility_240m_bps",
        "target_intrabar_range_bps",
        "target_path_efficiency_60m",
        "target_quote_volume_vs_60m_mean",
        "target_trade_count_vs_60m_mean",
        "target_signed_taker_flow_15m",
        "target_signed_taker_flow_60m",
        "target_return_zscore_240m",
        "target_beta_residual_return_60m_bps",
        "cross_asset_return_dispersion_15m_bps",
        "cross_asset_taker_flow_mean",
        "cross_asset_taker_flow_agreement",
        "target_to_btc_volatility_ratio_60m",
        "target_same_minute_of_week_liquidity_ratio",
        "target_premium_close_bps",
        "target_premium_zscore_240m",
        "target_premium_age_minutes",
        "target_premium_observed_fraction_240m",
        "cross_asset_premium_dispersion_bps",
        "target_last_settled_funding_rate_bps",
        "target_funding_interval_hours",
        "target_minutes_since_funding",
        "target_settled_funding_sum_24h_bps",
        "target_settled_funding_sum_168h_bps",
        "target_funding_event_zscore_30",
        "cross_asset_funding_dispersion_bps",
        "weekend_flag",
    )
    index = {name: position for position, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise ValueError(f"Round 39 AI feature contract is incomplete: {missing}")
    return tuple(index[name] for name in wanted)


def rolling_case_set_sha256(cases: Sequence[RollingAITradeCase]) -> str:
    return _canonical_sha256([case.identity_payload() for case in cases])


def _month_index(timestamp_ms: int) -> int:
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    return timestamp.month - 1


def build_rolling_ai_cases(
    dataset: DerivativesHurdleDataset,
    candidates: Sequence[RollingSupportCandidate],
) -> tuple[RollingAITradeCase, ...]:
    """Freeze one highest-confidence actual ML action per symbol and UTC day."""

    if not candidates:
        return ()
    proposals: dict[
        tuple[int, int],
        tuple[float, int, RollingSupportCandidate, int, float],
    ] = {}
    for candidate in candidates:
        direction_by_row = {
            int(row): int(direction)
            for row, direction in zip(
                candidate.evaluation.selected_indices,
                candidate.evaluation.selected_direction,
                strict=True,
            )
        }
        net_by_row = {
            int(row): float(net)
            for row, net in zip(
                candidate.evaluation.selected_indices,
                candidate.evaluation.net_return_bps,
                strict=True,
            )
        }
        for row in candidate.evaluation.selected_indices:
            row = int(row)
            timestamp = int(dataset.decision_time_ms[row])
            day_index = int((timestamp - EVALUATION_START_MS) // 86_400_000)
            if not 0 <= day_index < EVALUATION_DAYS:
                raise ValueError("Round 39 AI proposal lies outside 2025-H1")
            symbol_index = int(dataset.symbol_index[row])
            direction = direction_by_row[row]
            confidence = float(
                candidate.probabilities[row, 2 if direction > 0 else 0]
            )
            proposal = (confidence, row, candidate, direction, net_by_row[row])
            key = (symbol_index, day_index)
            current = proposals.get(key)
            if current is None or confidence > current[0] or (
                confidence == current[0]
                and (candidate.candidate_id, row)
                < (current[2].candidate_id, current[1])
            ):
                proposals[key] = proposal
    ordered = [
        (
            row,
            candidate,
            direction,
            net,
            symbol_index,
            day_index,
            confidence,
        )
        for (symbol_index, day_index), (
            confidence,
            row,
            candidate,
            direction,
            net,
        ) in proposals.items()
    ]
    grouped: dict[tuple[int, int], list[tuple[object, ...]]] = {}
    for item in ordered:
        timestamp = datetime.fromtimestamp(
            int(dataset.decision_time_ms[int(item[0])]) / 1000.0, UTC
        )
        grouped.setdefault((int(item[4]), timestamp.month - 1), []).append(item)
    selected: list[tuple[object, ...]] = []
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                -float(item[6]),
                str(item[1].candidate_id),
                int(item[0]),
            )
        )
        selected.extend(group[:10])
    ordered = selected
    ordered.sort(key=lambda item: (int(dataset.decision_time_ms[int(item[0])]), item[4]))
    if len(ordered) > MAX_CASES:
        raise ValueError("Round 39 AI case count exceeds the frozen maximum")
    feature_indices = _feature_indices(dataset.feature_names)
    completed: list[tuple[int, int, float]] = []
    cases: list[RollingAITradeCase] = []
    for (
        row,
        candidate,
        direction,
        outcome,
        symbol_index,
        relative_day,
        action_probability,
    ) in ordered:
        timestamp = int(dataset.decision_time_ms[row])
        prior = [
            value
            for settled_time, completed_symbol, value in completed
            if completed_symbol == symbol_index
            and settled_time < timestamp
        ][-20:]
        loss_streak = 0
        for value in reversed(prior):
            if value >= 0.0:
                break
            loss_streak += 1
        prior_symbol_counts = {
            symbol: sum(
                completed_symbol == index
                for _, completed_symbol, _ in completed
            )
            for index, symbol in enumerate(SYMBOLS)
        }
        p_short = float(candidate.probabilities[row, 0])
        p_long = float(candidate.probabilities[row, 2])
        month_index = _month_index(timestamp)
        threshold = candidate.monthly_thresholds[f"2025-{month_index + 1:02d}"]
        if threshold is None:
            raise ValueError("AI proposal came from a month without a threshold")
        payload = {
            "schema_version": "causal-rolling-ai-veto-case-v1",
            "relative_day_index": relative_day,
            "refit_sequence_index": month_index,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": candidate.horizon_minutes,
            "proposed_direction": "long" if direction > 0 else "short",
            "ml_architecture": candidate.architecture,
            "ml_weighting": candidate.weighting,
            "action_probability": round(action_probability, 6),
            "opposing_action_probability": round(
                p_short if direction > 0 else p_long, 6
            ),
            "action_probability_threshold": threshold[0],
            "direction_probability_margin_threshold": threshold[1],
            "market_and_contract_state": {
                dataset.feature_names[index]: round(
                    float(dataset.features[row, index]), 5
                )
                for index in feature_indices
            },
            "risk_state": {
                "open_position_count": 0,
                "prior_completed_symbol_cases": len(prior),
                "trailing_symbol_case_mean_net_bps": round(
                    float(np.mean(prior)) if prior else 0.0, 4
                ),
                "trailing_symbol_case_positive_rate": round(
                    float(np.mean(np.asarray(prior) > 0.0)) if prior else 0.0,
                    4,
                ),
                "consecutive_symbol_losses": loss_streak,
                "prior_case_counts_by_symbol": prior_symbol_counts,
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
        cases.append(
            RollingAITradeCase(
                case_id=_canonical_sha256(identity),
                candidate_id=candidate.candidate_id,
                dataset_row=row,
                relative_day_index=relative_day,
                refit_sequence_index=month_index,
                decision_time_ms=timestamp,
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
                timestamp + (candidate.horizon_minutes + 1) * 60_000,
                symbol_index,
                outcome,
            )
        )
    return tuple(cases)


SHORT_REASON_CODES = (
    "cost_ok",
    "weak_edge",
    "unstable",
    "flow_risk",
    "volatility",
    "liquidity",
    "cross_asset",
    "calibration",
    "loss_lock",
    "insufficient",
)
REASON_CODE_MAP = {
    "cost_ok": "edge_covers_cost",
    "weak_edge": "weak_cost_margin",
    "unstable": "unstable_analogs",
    "flow_risk": "adverse_taker_flow",
    "volatility": "volatility_shock",
    "liquidity": "liquidity_stress",
    "cross_asset": "cross_asset_disagreement",
    "calibration": "model_calibration_risk",
    "loss_lock": "loss_cooldown",
    "insufficient": "insufficient_evidence",
}


def _decision_schema(expected_case_ids: Sequence[str]) -> dict[str, object]:
    if not expected_case_ids or len(set(expected_case_ids)) != len(expected_case_ids):
        raise ValueError("AI schema requires unique expected case identities")
    expected_indices = list(range(len(expected_case_ids)))
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "case_index": {
                            "type": "integer",
                            "enum": expected_indices,
                        },
                        "action": {
                            "type": "string",
                            "enum": ["approve", "veto", "cooldown"],
                        },
                        "risk_percent": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "confidence_percent": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "reason_codes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(SHORT_REASON_CODES),
                            },
                            "minItems": 1,
                            "maxItems": 2,
                        },
                    },
                    "required": [
                        "case_index",
                        "action",
                        "risk_percent",
                        "confidence_percent",
                        "reason_codes",
                    ],
                    "additionalProperties": False,
                },
                "minItems": len(expected_case_ids),
                "maxItems": len(expected_case_ids),
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _parse_single_decision(value: object) -> AIVetoDecision:
    if not isinstance(value, Mapping):
        raise ValueError("AI batch decision is not an object")
    action = str(value.get("action") or "").strip().lower()
    if action not in {"approve", "veto", "cooldown"}:
        raise ValueError("AI action is invalid")
    risk = float(
        np.clip(_bounded_float(value.get("risk_percent"), 0.0), 0.0, 100.0)
        / 100.0
    )
    confidence = float(
        np.clip(
            _bounded_float(value.get("confidence_percent"), 0.0), 0.0, 100.0
        )
        / 100.0
    )
    raw_codes = value.get("reason_codes")
    if not isinstance(raw_codes, list) or not raw_codes:
        raise ValueError("AI reason_codes are missing")
    codes = tuple(
        dict.fromkeys(
            REASON_CODE_MAP[str(item)]
            for item in raw_codes
            if str(item) in REASON_CODE_MAP
        )
    )
    if not codes or len(codes) > 2:
        raise ValueError("AI reason_codes are invalid")
    if action != "approve":
        risk = 0.0
    if action == "approve" and risk <= 0.0:
        action = "veto"
        codes = tuple(dict.fromkeys((*codes, "insufficient_evidence")))[:2]
    return AIVetoDecision(
        action=action,
        risk_multiplier=risk,
        confidence=confidence,
        reason_codes=codes,
        summary="Compact structured Round 39 veto decision.",
        valid=True,
        failure_reason="",
    )


def _parse_batch_response(
    response: object,
    expected_case_ids: Sequence[str],
) -> dict[str, AIVetoDecision]:
    if not isinstance(response, Mapping):
        raise ValueError("AI response is not an object")
    message = response.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("AI response message content is missing")
    content = json.loads(str(message["content"]))
    if not isinstance(content, Mapping) or not isinstance(content.get("decisions"), list):
        raise ValueError("AI response decisions are missing")
    values = content["decisions"]
    if len(values) != len(expected_case_ids):
        raise ValueError("AI response decision count differs from request")
    parsed: dict[str, AIVetoDecision] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("AI response contains a non-object decision")
        raw_index = value.get("case_index")
        if not isinstance(raw_index, int) or not 0 <= raw_index < len(expected_case_ids):
            raise ValueError("AI response case identity is missing or duplicated")
        case_id = expected_case_ids[raw_index]
        if case_id in parsed:
            raise ValueError("AI response case identity is missing or duplicated")
        parsed[case_id] = _parse_single_decision(value)
    if set(parsed) != set(expected_case_ids):
        raise ValueError("AI response case identities are incomplete")
    return parsed


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
        "You are the fail-closed risk coordinator for an autonomous crypto day-trading research system. "
        "Evaluate every independent ML-proposed action using only its causal structured evidence. Outcomes "
        "are hidden. You cannot create a trade, reverse direction, increase risk, infer calendar dates or "
        "exact prices, retrieve news, or assume missing information is favorable. Premium and settled "
        "funding fields are contract-risk context, not guaranteed alpha. Approve only coherent calibrated "
        "evidence; veto weak or contradictory evidence; use cooldown for unstable regimes or loss control. "
        "Return one decision for every case_index and no additional cases.\n"
        f"CASES={serialized}"
    )


def _profit_factor(values: np.ndarray) -> float | None:
    positive = float(np.sum(values[values > 0.0]))
    negative = float(np.sum(values[values < 0.0]))
    return positive / abs(negative) if negative < 0.0 else None


def _max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return float(np.max(np.maximum.accumulate(equity) - equity))


def _binomial_upper_tail(trials: int, successes: int) -> float:
    if trials <= 0:
        return 1.0
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials)
    )


def _stationary_bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int = 2000,
    mean_block_length: int = 5,
    seed: int = 3921,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    count = values.size
    results = np.empty(samples, dtype=np.float64)
    restart = 1.0 / mean_block_length
    for sample in range(samples):
        index = int(generator.integers(0, count))
        total = 0.0
        for _ in range(count):
            total += float(values[index])
            index = (
                int(generator.integers(0, count))
                if generator.random() < restart
                else (index + 1) % count
            )
        results[sample] = total / count
    return tuple(float(item) for item in np.quantile(results, (0.025, 0.5, 0.975)))


def benchmark_rolling_ai_model(
    cases: Sequence[RollingAITradeCase],
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
    progress: ProgressCallback | None = None,
) -> RollingAIModelReport:
    """Run one frozen local model over the immutable case set in batches of 12."""

    if model not in AI_MODELS:
        raise ValueError(f"model is not frozen for Round 39: {model}")
    if not cases or len(cases) > MAX_CASES:
        raise ValueError("Round 39 AI case count is empty or exceeds its maximum")
    digest, metadata_sha = _model_metadata(
        base_url,
        model,
        timeout_seconds=timeout_seconds,
    )
    frozen_case_sha = rolling_case_set_sha256(cases)
    results: list[RollingAICaseResult] = []
    batch_latencies: list[float] = []
    batches = [cases[index : index + BATCH_SIZE] for index in range(0, len(cases), BATCH_SIZE)]
    for batch_index, batch in enumerate(batches, start=1):
        expected_case_ids = [case.case_id for case in batch]
        schema = _decision_schema(expected_case_ids)
        prompt = _batch_prompt(batch)
        request = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON matching the supplied schema. Never increase risk.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 12_288,
                "num_predict": 900,
                "seed": 3901,
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
            decisions = _parse_batch_response(raw, expected_case_ids)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            decisions = {
                case.case_id: _failed_decision(failure) for case in batch
            }
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
                "round39_ai_veto",
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
    baseline_day_has_case = np.zeros(EVALUATION_DAYS, dtype=bool)
    retained_day_has_case = np.zeros(EVALUATION_DAYS, dtype=bool)
    retained_by_symbol = {symbol: 0 for symbol in SYMBOLS}
    month_totals = np.zeros(6, dtype=np.float64)
    retained_values: list[float] = []
    for result in results:
        case = by_case[result.case_id]
        baseline_daily[case.relative_day_index] += result.baseline_net_bps
        ai_daily[case.relative_day_index] += result.ai_net_bps
        baseline_day_has_case[case.relative_day_index] = True
        if result.decision.risk_multiplier > 0.0:
            retained_day_has_case[case.relative_day_index] = True
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
    lower, median, upper = _stationary_bootstrap_mean(delta)
    baseline_values = np.asarray([result.baseline_net_bps for result in results])
    ai_values = np.asarray([result.ai_net_bps for result in results])
    approvals = sum(result.decision.action == "approve" for result in results)
    vetoes = sum(result.decision.action == "veto" for result in results)
    cooldowns = sum(result.decision.action == "cooldown" for result in results)
    valid = sum(result.decision.valid for result in results)
    retained_total = int(retained.size)
    concentration = (
        max(retained_by_symbol.values()) / retained_total if retained_total else 1.0
    )
    active_days = int(np.count_nonzero(retained_day_has_case))
    profit_factor = _profit_factor(retained)
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
    sign_p = _binomial_upper_tail(trials, positive)
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
        approvals=approvals,
        vetoes=vetoes,
        cooldowns=cooldowns,
        provider_failures=len(cases) - valid,
        average_batch_latency_seconds=float(np.mean(batch_latencies)),
        average_case_latency_seconds=float(
            sum(batch_latencies) / len(cases)
        ),
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
        matched_days=int(np.count_nonzero(baseline_day_has_case)),
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
    "MAX_CASES",
    "RollingAICaseResult",
    "RollingAIModelReport",
    "RollingAITradeCase",
    "benchmark_rolling_ai_model",
    "build_rolling_ai_cases",
    "rolling_case_set_sha256",
]
