"""Causal local-LLM veto ablation over frozen ML candidate trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np

from .cross_asset_cost_data import CrossAssetDataset, SYMBOLS, role_by_name
from .cross_asset_cost_model import (
    CandidateResult,
    EXECUTION_CHARGE_BPS,
    TrainedCandidates,
    calibrated_predictions,
)


AI_MODELS = ("qwen3:8b", "fino1:8b")
MAX_CASES_PER_MODEL = 270
AI_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "veto", "cooldown"]},
        "risk_multiplier": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_codes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "edge_covers_cost",
                    "weak_cost_margin",
                    "unstable_analogs",
                    "adverse_taker_flow",
                    "volatility_shock",
                    "liquidity_stress",
                    "cross_asset_disagreement",
                    "model_calibration_risk",
                    "loss_cooldown",
                    "insufficient_evidence",
                ],
            },
            "maxItems": 4,
        },
        "summary": {"type": "string", "maxLength": 180},
    },
    "required": [
        "action",
        "risk_multiplier",
        "confidence",
        "reason_codes",
        "summary",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AITradeCase:
    case_id: str
    dataset_row: int
    relative_day_index: int
    decision_time_ms: int
    symbol: str
    horizon_minutes: int
    direction: str
    calibrated_prediction_bps: float
    threshold_bps: float
    outcome_net_bps: float
    prompt_payload: Mapping[str, object]

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "relative_day_index": self.relative_day_index,
            "symbol": self.symbol,
            "horizon_minutes": self.horizon_minutes,
            "direction": self.direction,
            "calibrated_prediction_bps": self.calibrated_prediction_bps,
            "threshold_bps": self.threshold_bps,
            "prompt_payload": dict(self.prompt_payload),
        }

    def evidence_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "outcome_net_bps": self.outcome_net_bps}


@dataclass(frozen=True)
class AIVetoDecision:
    action: str
    risk_multiplier: float
    confidence: float
    reason_codes: tuple[str, ...]
    summary: str
    valid: bool
    failure_reason: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass(frozen=True)
class AICaseResult:
    case_id: str
    model: str
    latency_seconds: float
    response_sha256: str
    decision: AIVetoDecision
    baseline_net_bps: float
    ai_net_bps: float

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "decision": self.decision.asdict(),
        }


@dataclass(frozen=True)
class AIModelVetoReport:
    model: str
    model_digest: str
    model_metadata_sha256: str
    case_set_sha256: str
    cases: int
    valid_responses: int
    approvals: int
    vetoes: int
    cooldowns: int
    provider_failures: int
    average_latency_seconds: float
    baseline_total_net_bps: float
    ai_total_net_bps: float
    baseline_mean_case_net_bps: float
    ai_mean_approved_case_net_bps: float
    ai_profit_factor: float
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
    results: tuple[AICaseResult, ...]

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "uplift_gate_reasons": list(self.uplift_gate_reasons),
            "results": [item.asdict() for item in self.results],
        }


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _key_feature_indices(feature_names: Sequence[str]) -> tuple[int, ...]:
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
        "target_signed_taker_flow_5m",
        "target_signed_taker_flow_15m",
        "target_signed_taker_flow_60m",
        "target_return_zscore_240m",
        "target_quote_volume_zscore_240m",
        "target_beta_residual_return_60m_bps",
        "cross_asset_return_dispersion_15m_bps",
        "cross_asset_taker_flow_mean",
        "cross_asset_taker_flow_agreement",
        "target_to_btc_volatility_ratio_60m",
        "target_same_minute_of_week_liquidity_ratio",
        "weekend_flag",
    )
    index = {name: position for position, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise ValueError(f"AI veto feature contract missing: {missing}")
    return tuple(index[name] for name in wanted)


def _analogs(
    dataset: CrossAssetDataset,
    *,
    row: int,
    horizon: int,
    direction: float,
    threshold: float,
    prediction: np.ndarray,
    feature_indices: Sequence[int],
) -> dict[str, object]:
    symbol_index = int(dataset.symbol_index[row])
    calibration = (
        dataset.role_masks[horizon]["calibration"]
        & (dataset.symbol_index == symbol_index)
        & np.isfinite(prediction)
        & (np.sign(prediction) == direction)
        & (np.abs(prediction) >= threshold)
    )
    indices = np.flatnonzero(calibration)
    if indices.size == 0:
        return {
            "samples": 0,
            "mean_net_bps": 0.0,
            "median_net_bps": 0.0,
            "positive_rate": 0.0,
            "mean_absolute_prediction_bps": 0.0,
        }
    calibration_features = dataset.features[
        np.ix_(indices, np.asarray(feature_indices, dtype=np.int64))
    ].astype(np.float64)
    current = dataset.features[row, list(feature_indices)].astype(np.float64)
    means = np.mean(calibration_features, axis=0)
    scales = np.std(calibration_features, axis=0)
    scales[scales < 1e-6] = 1.0
    standardized = (calibration_features - means) / scales
    current_standardized = (current - means) / scales
    distance = np.mean((standardized - current_standardized) ** 2, axis=1)
    nearest_count = min(8, indices.size)
    nearest = indices[np.argpartition(distance, nearest_count - 1)[:nearest_count]]
    net = (
        direction * dataset.gross_return_bps[horizon][nearest].astype(np.float64)
        - EXECUTION_CHARGE_BPS
    )
    return {
        "samples": int(nearest_count),
        "mean_net_bps": round(float(np.mean(net)), 4),
        "median_net_bps": round(float(np.median(net)), 4),
        "positive_rate": round(float(np.mean(net > 0.0)), 4),
        "mean_absolute_prediction_bps": round(
            float(np.mean(np.abs(prediction[nearest]))),
            4,
        ),
    }


def build_ai_trade_cases(
    dataset: CrossAssetDataset,
    trained: TrainedCandidates,
    candidate_results: Sequence[CandidateResult],
) -> tuple[AITradeCase, ...]:
    """Freeze one highest-conviction shared-model case per symbol/day."""

    shared = {
        item.horizon_minutes: item
        for item in candidate_results
        if item.family == "shared_cross_asset_lightgbm"
        and item.selected_threshold_bps is not None
    }
    if not shared:
        return ()
    predictions: dict[int, np.ndarray] = {}
    for horizon in shared:
        predictions[horizon] = calibrated_predictions(
            dataset,
            trained,
            family="shared_cross_asset_lightgbm",
            horizon=horizon,
        )[0]
    viability = role_by_name("viability")
    last_day = datetime.fromisoformat(viability.end).replace(tzinfo=UTC)
    first_day = last_day - timedelta(days=89)
    first_ms = int(first_day.timestamp() * 1000)
    end_exclusive_ms = viability.end_exclusive_ms
    by_symbol_day: dict[tuple[int, int], tuple[float, int, int]] = {}
    for horizon, result in shared.items():
        threshold = float(result.selected_threshold_bps)
        prediction = predictions[horizon]
        mask = (
            dataset.role_masks[horizon]["viability"]
            & (dataset.decision_time_ms >= first_ms)
            & (dataset.decision_time_ms < end_exclusive_ms)
            & np.isfinite(prediction)
            & (np.abs(prediction) >= threshold)
        )
        for row in np.flatnonzero(mask):
            day = int((int(dataset.decision_time_ms[row]) - first_ms) // 86_400_000)
            symbol_index = int(dataset.symbol_index[row])
            conviction = float(abs(prediction[row]) / threshold)
            key = (symbol_index, day)
            current = by_symbol_day.get(key)
            proposal = (conviction, int(row), horizon)
            if current is None or proposal > current:
                by_symbol_day[key] = proposal
    selected = sorted(
        (value[1], value[2], key[0], key[1]) for key, value in by_symbol_day.items()
    )
    selected.sort(key=lambda item: (int(dataset.decision_time_ms[item[0]]), item[2]))
    feature_indices = _key_feature_indices(dataset.feature_names)
    completed_by_symbol: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(SYMBOLS))
    }
    cases: list[AITradeCase] = []
    for row, horizon, symbol_index, relative_day in selected[:MAX_CASES_PER_MODEL]:
        result = shared[horizon]
        threshold = float(result.selected_threshold_bps)
        prediction = predictions[horizon]
        direction_value = float(np.sign(prediction[row]))
        direction = "long" if direction_value > 0.0 else "short"
        outcome = (
            direction_value * float(dataset.gross_return_bps[horizon][row])
            - EXECUTION_CHARGE_BPS
        )
        prior = [
            value
            for day, value in completed_by_symbol[symbol_index]
            if day < relative_day
        ][-20:]
        trailing_mean = float(np.mean(prior)) if prior else 0.0
        trailing_positive = float(np.mean(np.asarray(prior) > 0.0)) if prior else 0.0
        loss_streak = 0
        for value in reversed(prior):
            if value >= 0.0:
                break
            loss_streak += 1
        feature_payload = {
            dataset.feature_names[index]: round(float(dataset.features[row, index]), 5)
            for index in feature_indices
        }
        payload = {
            "schema_version": "causal-ai-trade-veto-case-v1",
            "relative_day_index": relative_day,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": horizon,
            "proposed_direction": direction,
            "calibrated_ml_prediction_bps": round(float(prediction[row]), 4),
            "execution_cost_hurdle_bps": EXECUTION_CHARGE_BPS,
            "route_threshold_bps": threshold,
            "market_state": feature_payload,
            "past_only_nearest_regimes": _analogs(
                dataset,
                row=row,
                horizon=horizon,
                direction=direction_value,
                threshold=threshold,
                prediction=prediction,
                feature_indices=feature_indices,
            ),
            "risk_state": {
                "open_position_count": 0,
                "prior_completed_cases": len(prior),
                "trailing_case_mean_net_bps": round(trailing_mean, 4),
                "trailing_case_positive_rate": round(trailing_positive, 4),
                "consecutive_losses": loss_streak,
                "maximum_risk_multiplier": 1.0,
            },
        }
        case_identity = {
            "relative_day_index": relative_day,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": horizon,
            "dataset_row": int(row),
            "prompt": payload,
        }
        case_id = _canonical_sha256(case_identity)
        cases.append(
            AITradeCase(
                case_id=case_id,
                dataset_row=int(row),
                relative_day_index=relative_day,
                decision_time_ms=int(dataset.decision_time_ms[row]),
                symbol=SYMBOLS[symbol_index],
                horizon_minutes=horizon,
                direction=direction,
                calibrated_prediction_bps=float(prediction[row]),
                threshold_bps=threshold,
                outcome_net_bps=outcome,
                prompt_payload=payload,
            )
        )
        completed_by_symbol[symbol_index].append((relative_day, outcome))
    return tuple(cases)


def case_set_sha256(cases: Sequence[AITradeCase]) -> str:
    return _canonical_sha256([item.identity_payload() for item in cases])


def _request_json(
    url: str,
    payload: Mapping[str, object] | None,
    *,
    timeout_seconds: float,
    method: str = "POST",
) -> object:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _model_metadata(
    base_url: str,
    model: str,
    *,
    timeout_seconds: float,
) -> tuple[str, str]:
    tags = _request_json(
        f"{base_url.rstrip('/')}/api/tags",
        None,
        timeout_seconds=timeout_seconds,
        method="GET",
    )
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Ollama tags response is invalid")
    selected = None
    for item in tags["models"]:
        if isinstance(item, Mapping) and str(item.get("name")) == model:
            selected = item
            break
    if selected is None:
        raise ValueError(f"Ollama model is not installed: {model}")
    digest = str(selected.get("digest") or "")
    if len(digest) != 64:
        raise ValueError(f"Ollama model digest is invalid for {model}")
    show = _request_json(
        f"{base_url.rstrip('/')}/api/show",
        {"model": model, "verbose": False},
        timeout_seconds=timeout_seconds,
    )
    return digest, _canonical_sha256(show)


def _prompt(case: AITradeCase) -> str:
    payload = json.dumps(
        case.prompt_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return (
        "You are the fail-closed risk coordinator for an autonomous crypto day-trading research system. "
        "Evaluate one ML-proposed trade using only the causal structured evidence below. The ML prediction "
        "already includes amplitude calibration, but realized taker cost is material. You cannot create a "
        "trade, reverse direction, increase risk, assume missing information is favorable, or use knowledge "
        "of historical dates and prices. Approve only when cost coverage, analog stability, liquidity, "
        "cross-asset state, and recent risk state are coherent. Veto weak or contradictory evidence; choose "
        "cooldown for unstable regimes or loss control. Return only the required JSON.\n"
        f"CASE={payload}"
    )


def _bounded_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _decision(payload: object) -> AIVetoDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("AI response is not an object")
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("AI response message is missing")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("AI response content is missing")
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise ValueError("AI response content is not an object")
    action = str(parsed.get("action") or "").strip().lower()
    if action not in {"approve", "veto", "cooldown"}:
        raise ValueError("AI action is invalid")
    risk = float(np.clip(_bounded_float(parsed.get("risk_multiplier"), 0.0), 0.0, 1.0))
    confidence = float(np.clip(_bounded_float(parsed.get("confidence"), 0.0), 0.0, 1.0))
    raw_codes = parsed.get("reason_codes")
    allowed = set(AI_SCHEMA["properties"]["reason_codes"]["items"]["enum"])
    if not isinstance(raw_codes, list) or not raw_codes:
        raise ValueError("AI reason_codes are missing")
    codes = tuple(dict.fromkeys(str(value) for value in raw_codes if str(value) in allowed))
    if not codes or len(codes) > 4:
        raise ValueError("AI reason_codes are invalid")
    summary = str(parsed.get("summary") or "").strip()[:180]
    if not summary:
        raise ValueError("AI summary is missing")
    if action != "approve":
        risk = 0.0
    if action == "approve" and risk <= 0.0:
        action = "veto"
        codes = tuple(dict.fromkeys((*codes, "insufficient_evidence")))[:4]
    return AIVetoDecision(
        action=action,
        risk_multiplier=risk,
        confidence=confidence,
        reason_codes=codes,
        summary=summary,
        valid=True,
        failure_reason="",
    )


def _failed_decision(reason: str) -> AIVetoDecision:
    return AIVetoDecision(
        action="veto",
        risk_multiplier=0.0,
        confidence=0.0,
        reason_codes=("insufficient_evidence",),
        summary="Provider or schema failure; fail-closed veto.",
        valid=False,
        failure_reason=reason[:240],
    )


def _profit_factor(values: np.ndarray) -> float:
    positive = float(np.sum(values[values > 0.0]))
    negative = float(np.sum(values[values < 0.0]))
    if negative < 0.0:
        return positive / abs(negative)
    return float("inf") if positive > 0.0 else 0.0


def _max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    running_maximum = np.maximum.accumulate(equity)
    return float(np.max(running_maximum - equity))


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
    seed: int = 3721,
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
    return tuple(float(value) for value in np.quantile(results, (0.025, 0.5, 0.975)))


def benchmark_ai_veto_model(
    cases: Sequence[AITradeCase],
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 60.0,
    progress: ProgressCallback | None = None,
) -> AIModelVetoReport:
    """Replay one installed local model over an immutable case set."""

    if model not in AI_MODELS:
        raise ValueError(f"model is not frozen for Round 37: {model}")
    if not cases:
        raise ValueError("AI veto benchmark requires at least one case")
    if len(cases) > MAX_CASES_PER_MODEL:
        raise ValueError("AI veto benchmark case count exceeds frozen maximum")
    digest, metadata_sha = _model_metadata(
        base_url,
        model,
        timeout_seconds=timeout_seconds,
    )
    frozen_case_sha = case_set_sha256(cases)
    results: list[AICaseResult] = []
    for position, case in enumerate(cases, start=1):
        request = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON matching the supplied schema. Never increase risk.",
                },
                {"role": "user", "content": _prompt(case)},
            ],
            "stream": False,
            "format": AI_SCHEMA,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 220,
                "seed": 3701,
            },
        }
        started = time.perf_counter()
        raw: object = {}
        try:
            raw = _request_json(
                f"{base_url.rstrip('/')}/api/chat",
                request,
                timeout_seconds=timeout_seconds,
            )
            decision = _decision(raw)
        except Exception as exc:
            decision = _failed_decision(f"{type(exc).__name__}: {exc}")
        latency = time.perf_counter() - started
        response_sha = _canonical_sha256(raw)
        ai_net = case.outcome_net_bps * decision.risk_multiplier
        current = AICaseResult(
            case_id=case.case_id,
            model=model,
            latency_seconds=latency,
            response_sha256=response_sha,
            decision=decision,
            baseline_net_bps=case.outcome_net_bps,
            ai_net_bps=ai_net,
        )
        results.append(current)
        if progress is not None:
            progress(
                "ai_veto",
                {
                    "model": model,
                    "case": position,
                    "cases": len(cases),
                    "action": decision.action,
                    "valid": decision.valid,
                    "latency_seconds": round(latency, 3),
                },
            )

    baseline_daily = np.zeros(90, dtype=np.float64)
    ai_daily = np.zeros(90, dtype=np.float64)
    for case, result in zip(cases, results, strict=True):
        baseline_daily[case.relative_day_index] += result.baseline_net_bps
        ai_daily[case.relative_day_index] += result.ai_net_bps
    delta = ai_daily - baseline_daily
    nonzero = delta[delta != 0.0]
    positive = int(np.count_nonzero(nonzero > 0.0))
    negative = int(np.count_nonzero(nonzero < 0.0))
    trials = positive + negative
    positive_rate = positive / trials if trials else 0.0
    sign_p = _binomial_upper_tail(trials, positive)
    lower, median, upper = _stationary_bootstrap_mean(delta)
    baseline_values = np.asarray([item.baseline_net_bps for item in results])
    ai_values = np.asarray([item.ai_net_bps for item in results])
    approved_values = ai_values[ai_values != 0.0]
    valid = sum(item.decision.valid for item in results)
    approvals = sum(item.decision.action == "approve" for item in results)
    vetoes = sum(item.decision.action == "veto" for item in results)
    cooldowns = sum(item.decision.action == "cooldown" for item in results)
    reasons: list[str] = []
    if len(cases) < 90:
        reasons.append("ai_cases<90")
    if approvals < 30:
        reasons.append("ai_approvals<30")
    if valid != len(cases):
        reasons.append("ai_provider_or_schema_failures")
    if positive_rate < 0.55:
        reasons.append("ai_positive_daily_delta_rate<0.55")
    if sign_p > 0.05:
        reasons.append("ai_exact_sign_test_p_value>0.05")
    if lower <= 0.0:
        reasons.append("ai_block_bootstrap_delta_lower_95<=0")
    if float(np.sum(ai_values)) <= 0.0:
        reasons.append("ai_total_net_bps<=0")
    baseline_drawdown = _max_drawdown(baseline_daily)
    ai_drawdown = _max_drawdown(ai_daily)
    if ai_drawdown > baseline_drawdown + 1e-9:
        reasons.append("ai_max_drawdown_worse_than_baseline")
    return AIModelVetoReport(
        model=model,
        model_digest=digest,
        model_metadata_sha256=metadata_sha,
        case_set_sha256=frozen_case_sha,
        cases=len(cases),
        valid_responses=valid,
        approvals=approvals,
        vetoes=vetoes,
        cooldowns=cooldowns,
        provider_failures=len(cases) - valid,
        average_latency_seconds=float(
            np.mean([item.latency_seconds for item in results])
        ),
        baseline_total_net_bps=float(np.sum(baseline_values)),
        ai_total_net_bps=float(np.sum(ai_values)),
        baseline_mean_case_net_bps=float(np.mean(baseline_values)),
        ai_mean_approved_case_net_bps=(
            float(np.mean(approved_values)) if approved_values.size else 0.0
        ),
        ai_profit_factor=_profit_factor(approved_values),
        baseline_max_drawdown_bps=baseline_drawdown,
        ai_max_drawdown_bps=ai_drawdown,
        matched_days=90,
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
    "AI_MODELS",
    "AI_SCHEMA",
    "AICaseResult",
    "AIModelVetoReport",
    "AITradeCase",
    "AIVetoDecision",
    "benchmark_ai_veto_model",
    "build_ai_trade_cases",
    "case_set_sha256",
]
