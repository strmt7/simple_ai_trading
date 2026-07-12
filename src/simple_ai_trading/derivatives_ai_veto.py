"""Causal local-LLM veto ablation for passed Round 38 ML actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from .ai_trade_veto import (
    AI_MODELS,
    AIModelVetoReport,
    AITradeCase,
    benchmark_ai_veto_model,
)
from .cross_asset_cost_data import SYMBOLS, role_by_name
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .derivatives_hurdle_model import PassedCandidate


MAX_CASES_PER_MODEL = 270


@dataclass(frozen=True)
class DerivativesAITradeCase:
    case_id: str
    dataset_row: int
    relative_day_index: int
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
            "relative_day_index": self.relative_day_index,
            "symbol": self.symbol,
            "horizon_minutes": self.horizon_minutes,
            "direction": self.direction,
            "action_probability": self.action_probability,
            "direction_probability_margin": self.direction_probability_margin,
            "prompt_payload": dict(self.prompt_payload),
        }

    def evidence_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "outcome_net_bps": self.outcome_net_bps}


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def derivatives_case_set_sha256(
    cases: Sequence[DerivativesAITradeCase],
) -> str:
    return _canonical_sha256([item.identity_payload() for item in cases])


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
        raise ValueError(f"Round 38 AI feature contract is incomplete: {missing}")
    return tuple(index[name] for name in wanted)


def _routed_action(
    probabilities: np.ndarray,
    *,
    probability_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_short = probabilities[:, 0]
    p_long = probabilities[:, 2]
    confidence = np.maximum(p_short, p_long)
    margin = np.abs(p_long - p_short)
    routed = (confidence >= probability_threshold) & (margin >= margin_threshold)
    direction = np.where(p_long > p_short, 1, -1).astype(np.int8)
    return routed, direction


def _past_analogs(
    dataset: DerivativesHurdleDataset,
    candidate: PassedCandidate,
    *,
    row: int,
    direction: int,
    feature_indices: Sequence[int],
) -> dict[str, object]:
    routed, candidate_direction = _routed_action(
        candidate.probabilities,
        probability_threshold=candidate.maximum_action_probability,
        margin_threshold=candidate.direction_probability_margin,
    )
    calibration = (
        dataset.role_masks[candidate.horizon_minutes]["calibration"]
        & (dataset.symbol_index == dataset.symbol_index[row])
        & routed
        & (candidate_direction == direction)
    )
    indices = np.flatnonzero(calibration)
    if indices.size == 0:
        return {
            "samples": 0,
            "mean_net_bps": 0.0,
            "median_net_bps": 0.0,
            "positive_rate": 0.0,
            "mean_action_probability": 0.0,
        }
    selected_features = np.asarray(feature_indices, dtype=np.int64)
    history = dataset.features[np.ix_(indices, selected_features)].astype(np.float64)
    current = dataset.features[row, selected_features].astype(np.float64)
    means = np.mean(history, axis=0)
    scales = np.std(history, axis=0)
    scales[scales < 1e-6] = 1.0
    distance = np.mean(
        (((history - means) / scales) - ((current - means) / scales)) ** 2,
        axis=1,
    )
    count = min(8, indices.size)
    nearest = indices[np.argpartition(distance, count - 1)[:count]]
    net = (
        dataset.long_net_utility_bps[candidate.horizon_minutes][nearest]
        if direction > 0
        else dataset.short_net_utility_bps[candidate.horizon_minutes][nearest]
    ).astype(np.float64)
    action_probability = (
        candidate.probabilities[nearest, 2]
        if direction > 0
        else candidate.probabilities[nearest, 0]
    )
    return {
        "samples": int(count),
        "mean_net_bps": round(float(np.mean(net)), 4),
        "median_net_bps": round(float(np.median(net)), 4),
        "positive_rate": round(float(np.mean(net > 0.0)), 4),
        "mean_action_probability": round(float(np.mean(action_probability)), 5),
    }


def build_derivatives_ai_cases(
    dataset: DerivativesHurdleDataset,
    candidates: Sequence[PassedCandidate],
) -> tuple[DerivativesAITradeCase, ...]:
    """Freeze at most one highest-confidence passed action per symbol/day."""

    if not candidates:
        return ()
    viability = role_by_name("viability")
    last_day = datetime.fromisoformat(viability.end).replace(tzinfo=UTC)
    first_day = last_day - timedelta(days=89)
    first_ms = int(first_day.timestamp() * 1000)
    proposals: dict[
        tuple[int, int], tuple[float, int, PassedCandidate, int, float]
    ] = {}
    for candidate in candidates:
        outcome = candidate.viability
        net_by_row = {
            int(row): float(net)
            for row, net in zip(
                outcome.selected_indices, outcome.net_return_bps, strict=True
            )
        }
        for row in outcome.selected_indices:
            row = int(row)
            timestamp = int(dataset.decision_time_ms[row])
            if timestamp < first_ms or timestamp >= viability.end_exclusive_ms:
                continue
            symbol_index = int(dataset.symbol_index[row])
            day_index = int((timestamp - first_ms) // 86_400_000)
            p_short = float(candidate.probabilities[row, 0])
            p_long = float(candidate.probabilities[row, 2])
            direction = 1 if p_long > p_short else -1
            confidence = max(p_short, p_long)
            proposal = (confidence, row, candidate, direction, net_by_row[row])
            key = (symbol_index, day_index)
            current = proposals.get(key)
            if current is None or (proposal[0], candidate.candidate_id, -row) > (
                current[0],
                current[2].candidate_id,
                -current[1],
            ):
                proposals[key] = proposal
    ordered = list(
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
    )
    ordered.sort(key=lambda item: (int(dataset.decision_time_ms[item[0]]), item[4]))
    feature_indices = _feature_indices(dataset.feature_names)
    completed: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(SYMBOLS))
    }
    cases: list[DerivativesAITradeCase] = []
    for (
        row,
        candidate,
        direction,
        outcome,
        symbol_index,
        relative_day,
        action_probability,
    ) in ordered[:MAX_CASES_PER_MODEL]:
        prior = [
            value
            for day, value in completed[symbol_index]
            if day < relative_day
        ][-20:]
        loss_streak = 0
        for value in reversed(prior):
            if value >= 0.0:
                break
            loss_streak += 1
        payload = {
            "schema_version": "causal-derivatives-ai-veto-case-v1",
            "relative_day_index": relative_day,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": candidate.horizon_minutes,
            "proposed_direction": "long" if direction > 0 else "short",
            "ml_architecture": candidate.architecture,
            "ml_feature_set": candidate.feature_set,
            "action_probability": round(action_probability, 6),
            "opposing_action_probability": round(
                float(
                    candidate.probabilities[row, 0 if direction > 0 else 2]
                ),
                6,
            ),
            "action_probability_threshold": candidate.maximum_action_probability,
            "direction_probability_margin_threshold": (
                candidate.direction_probability_margin
            ),
            "market_state": {
                dataset.feature_names[index]: round(
                    float(dataset.features[row, index]), 5
                )
                for index in feature_indices
            },
            "past_only_nearest_regimes": _past_analogs(
                dataset,
                candidate,
                row=row,
                direction=direction,
                feature_indices=feature_indices,
            ),
            "risk_state": {
                "open_position_count": 0,
                "prior_completed_cases": len(prior),
                "trailing_case_mean_net_bps": round(
                    float(np.mean(prior)) if prior else 0.0, 4
                ),
                "trailing_case_positive_rate": round(
                    float(np.mean(np.asarray(prior) > 0.0)) if prior else 0.0,
                    4,
                ),
                "consecutive_losses": loss_streak,
                "maximum_risk_multiplier": 1.0,
            },
        }
        identity = {
            "relative_day_index": relative_day,
            "asset": SYMBOLS[symbol_index],
            "horizon_minutes": candidate.horizon_minutes,
            "dataset_row": row,
            "candidate_id": candidate.candidate_id,
            "prompt": payload,
        }
        case = DerivativesAITradeCase(
            case_id=_canonical_sha256(identity),
            dataset_row=row,
            relative_day_index=relative_day,
            decision_time_ms=int(dataset.decision_time_ms[row]),
            symbol=SYMBOLS[symbol_index],
            horizon_minutes=candidate.horizon_minutes,
            direction="long" if direction > 0 else "short",
            action_probability=action_probability,
            direction_probability_margin=abs(
                float(candidate.probabilities[row, 2])
                - float(candidate.probabilities[row, 0])
            ),
            outcome_net_bps=outcome,
            prompt_payload=payload,
        )
        cases.append(case)
        completed[symbol_index].append((relative_day, outcome))
    return tuple(cases)


def _prompt(case: DerivativesAITradeCase) -> str:
    payload = json.dumps(
        case.prompt_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return (
        "You are the fail-closed risk coordinator for an autonomous crypto day-trading research system. "
        "Evaluate one ML-proposed action using only the causal structured evidence below. Exact taker "
        "charges and historical funding cash flows are already included in matched ML replay, but the "
        "current outcome is hidden. You cannot create a trade, reverse direction, increase risk, infer a "
        "calendar date, retrieve news, or assume missing information is favorable. Approve only when the "
        "action probability, past-only analogs, liquidity, volatility, cross-asset state, premium, settled "
        "funding, and recent risk state are coherent. Veto weak or contradictory evidence; choose cooldown "
        "for unstable regimes or loss control. Return only the required JSON.\n"
        f"CASE={payload}"
    )


def benchmark_derivatives_ai_model(
    cases: Sequence[DerivativesAITradeCase],
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 60.0,
    progress: ProgressCallback | None = None,
) -> AIModelVetoReport:
    """Run one frozen local model on the immutable derivatives case set."""

    generic_cases = cast(Sequence[AITradeCase], cases)
    return benchmark_ai_veto_model(
        generic_cases,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        progress=progress,
        prompt_builder=cast(Callable[[AITradeCase], str], _prompt),
        seed=3801,
    )


__all__ = [
    "AI_MODELS",
    "DerivativesAITradeCase",
    "benchmark_derivatives_ai_model",
    "build_derivatives_ai_cases",
    "derivatives_case_set_sha256",
]
