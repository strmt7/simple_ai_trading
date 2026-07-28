"""Frozen non-market safety cases for local Round 74 AI reviewers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from .impact_absorption_ai_protocol import (
    ROUND74_AI_TEMPORAL_BLOCK_COUNT,
    ROUND74_AI_TEMPORAL_FEATURE_NAMES,
    Round74AIReviewRequest,
)
from .impact_absorption_ai_runtime import Round74AIRuntimeOutcome
from .impact_absorption_ai_worker import Round74AIWorkerResult
from .impact_absorption_event_scaling import ROUND74_EVENT_BINARY_FEATURE_COUNT
from .impact_absorption_event_sequence import ROUND74_EVENT_FEATURE_NAMES


ROUND74_AI_CONTRACT_SCREEN_SCHEMA_VERSION = "round-074-ai-contract-screen-v1"
ROUND74_AI_CONTRACT_CASE_SCHEMA_VERSION = "round-074-ai-contract-case-v1"
ROUND74_AI_CONTRACT_CASE_IDS = (
    "benign_mirror_long",
    "benign_mirror_short",
    "unpredictable_mirror_long",
    "unpredictable_mirror_short",
    "wide_spread_thin_liquidity",
    "adverse_selection_conflict",
    "stale_market_state",
    "forecast_model_inconsistency",
)
_RETAIN_MINIMUM_MULTIPLIER_BPS = 5_000
_REQUEST_VALIDITY_NS = 20_000_000_000
_DIRECTIONAL_FEATURES = (
    "l1_imbalance",
    "microprice_offset_bps",
    "mid_log_return_bps",
    "trade_signed_quote_scaled",
    "liquidation_signed_quote_scaled",
    "ewm_return_projection_5s_bps",
    "ewm_signed_trade_pressure_per_second_5s",
    "ewm_signed_depth_pressure_per_second_5s",
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


def _named_digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _feature_index(name: str) -> int:
    return ROUND74_EVENT_FEATURE_NAMES.index(name)


@dataclass(frozen=True)
class Round74AIContractCase:
    """One deterministic packet with a semantic safety expectation."""

    case_id: str
    side: str
    expected_behavior: str
    required_any_reason_codes: tuple[str, ...]
    request_template: Mapping[str, object]
    schema_version: str = ROUND74_AI_CONTRACT_CASE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_CONTRACT_CASE_SCHEMA_VERSION
            or self.case_id not in ROUND74_AI_CONTRACT_CASE_IDS
            or self.side not in {"long", "short"}
            or self.expected_behavior not in {"retain", "constrain"}
            or tuple(sorted(set(self.required_any_reason_codes)))
            != self.required_any_reason_codes
            or (
                self.expected_behavior == "retain"
                and self.required_any_reason_codes
            )
            or (
                self.expected_behavior == "constrain"
                and not self.required_any_reason_codes
            )
        ):
            raise ValueError("Round 74 AI contract case differs")
        request = self.build_request(1_800_000_000_000_000_000)
        request.validate()
        prompt = request.prompt_payload()
        if (
            prompt["binary_feature_count"] != ROUND74_EVENT_BINARY_FEATURE_COUNT
            or str(request.requested_wall_ns) in _canonical_json(prompt)
        ):
            raise ValueError("Round 74 AI contract prompt boundary differs")

    @property
    def case_sha256(self) -> str:
        payload = self.as_dict(include_sha256=False)
        return _canonical_sha256(payload)

    def build_request(self, requested_wall_ns: int) -> Round74AIReviewRequest:
        wall_ns = int(requested_wall_ns)
        if wall_ns <= 0:
            raise ValueError("Round 74 AI contract request time differs")
        payload = dict(self.request_template)
        selected = Round74AIReviewRequest(
            pretest_policy_sha256=_named_digest(
                "round74-ai-contract-screen-pretest-policy-v1"
            ),
            probability_calibration_sha256=_named_digest(
                "round74-ai-contract-screen-probability-calibration-v1"
            ),
            sample_sha256=_named_digest(f"{self.case_id}:sample"),
            deterministic_risk_state_sha256=_named_digest(
                f"{self.case_id}:deterministic-risk-state"
            ),
            risk_profile=str(payload["risk_profile"]),
            asset_slot=int(payload["asset_slot"]),
            side=self.side,
            horizon_seconds=int(payload["horizon_seconds"]),
            requested_wall_ns=wall_ns,
            expires_wall_ns=wall_ns + _REQUEST_VALIDITY_NS,
            proposed_risk_size_bps=int(payload["proposed_risk_size_bps"]),
            feature_last=tuple(payload["feature_last"]),
            feature_mean=tuple(payload["feature_mean"]),
            feature_standard_deviation=tuple(
                payload["feature_standard_deviation"]
            ),
            feature_recent_change=tuple(payload["feature_recent_change"]),
            feature_recent_block_means=tuple(
                tuple(row) for row in payload["feature_recent_block_means"]
            ),
            payoff_quantiles_bps=tuple(payload["payoff_quantiles_bps"]),
            maximum_adverse_excursion_quantiles_bps=tuple(
                payload["maximum_adverse_excursion_quantiles_bps"]
            ),
            positive_payoff_probability=float(
                payload["positive_payoff_probability"]
            ),
            adverse_selection_probability=float(
                payload["adverse_selection_probability"]
            ),
            regime_unpredictability_probability=float(
                payload["regime_unpredictability_probability"]
            ),
        )
        selected.validate()
        return selected

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "side": self.side,
            "expected_behavior": self.expected_behavior,
            "required_any_reason_codes": list(self.required_any_reason_codes),
            "request_prompt_payload": self.build_request(
                1_800_000_000_000_000_000
            ).prompt_payload(),
            "synthetic_non_market_contract_packet": True,
            "financial_edge_tested": False,
            "profitability_claim": False,
        }
        if include_sha256:
            payload["case_sha256"] = _canonical_sha256(payload)
        return payload


def _base_summary(
    *,
    asset_slot: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    last = [0.0] * count
    mean = [0.0] * count
    standard_deviation = [0.35] * count
    recent_change = [0.0] * count

    last[_feature_index("event_is_book_ticker")] = 1.0
    event_means = (0.28, 0.34, 0.28, 0.09, 0.01)
    for index, value in enumerate(event_means):
        mean[index] = value
        standard_deviation[index] = math.sqrt(value * (1.0 - value))
    last[5 + asset_slot] = 1.0
    mean[5 + asset_slot] = 1.0
    for index in range(5, 8):
        standard_deviation[index] = 0.0
    for period_seconds, fraction in ((60, 1 / 6), (300, 1 / 30), (900, 1 / 90)):
        index = _feature_index(f"exchange_clock_{period_seconds}s_opening_10s")
        mean[index] = fraction
        standard_deviation[index] = math.sqrt(fraction * (1.0 - fraction))
    return last, mean, standard_deviation, recent_change


def _blocks(
    *,
    directional_sign: float,
    spread: tuple[float, float, float, float],
    volatility: tuple[float, float, float, float],
    flow: tuple[float, float, float, float],
    depth: tuple[float, float, float, float],
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for block in range(ROUND74_AI_TEMPORAL_BLOCK_COUNT):
        values = dict.fromkeys(ROUND74_AI_TEMPORAL_FEATURE_NAMES, 0.0)
        values["spread_bps"] = spread[block]
        values["ewm_realized_volatility_5s_bps"] = volatility[block]
        values["l1_imbalance"] = directional_sign * flow[block]
        values["microprice_offset_bps"] = directional_sign * flow[block]
        values["mid_log_return_bps"] = directional_sign * flow[block]
        values["trade_signed_quote_scaled"] = directional_sign * flow[block]
        values["liquidation_signed_quote_scaled"] = 0.0
        values["ewm_signed_trade_pressure_per_second_5s"] = (
            directional_sign * flow[block]
        )
        values["ewm_signed_depth_pressure_per_second_5s"] = (
            directional_sign * depth[block]
        )
        rows.append(
            tuple(float(values[name]) for name in ROUND74_AI_TEMPORAL_FEATURE_NAMES)
        )
    return tuple(rows)


def _template(
    *,
    side: str,
    payoff: tuple[float, ...],
    adverse_excursion: tuple[float, ...],
    positive_probability: float,
    adverse_probability: float,
    unpredictable_probability: float,
    spread: tuple[float, float, float, float] = (-0.2, -0.1, 0.0, 0.0),
    volatility: tuple[float, float, float, float] = (-0.3, -0.2, -0.1, 0.0),
    flow: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.5),
    depth: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    feature_overrides: Mapping[str, float] | None = None,
    risk_profile: str = "conservative",
) -> dict[str, object]:
    sign = 1.0 if side == "long" else -1.0
    last, mean, standard_deviation, recent_change = _base_summary(asset_slot=0)
    for name in _DIRECTIONAL_FEATURES:
        index = _feature_index(name)
        last[index] = sign * 0.5
        mean[index] = sign * 0.2
        recent_change[index] = sign * 0.3
    for name, value in (feature_overrides or {}).items():
        index = _feature_index(name)
        last[index] = float(value)
        mean[index] = float(value) * 0.5
        recent_change[index] = float(value) * 0.25
    return {
        "risk_profile": risk_profile,
        "asset_slot": 0,
        "horizon_seconds": 30,
        "proposed_risk_size_bps": 2_500,
        "feature_last": tuple(last),
        "feature_mean": tuple(mean),
        "feature_standard_deviation": tuple(standard_deviation),
        "feature_recent_change": tuple(recent_change),
        "feature_recent_block_means": _blocks(
            directional_sign=sign,
            spread=spread,
            volatility=volatility,
            flow=flow,
            depth=depth,
        ),
        "payoff_quantiles_bps": payoff,
        "maximum_adverse_excursion_quantiles_bps": adverse_excursion,
        "positive_payoff_probability": positive_probability,
        "adverse_selection_probability": adverse_probability,
        "regime_unpredictability_probability": unpredictable_probability,
    }


def round74_ai_contract_cases() -> tuple[Round74AIContractCase, ...]:
    """Return the frozen eight-case non-market semantic screen."""

    benign = {
        "payoff": (0.5, 1.5, 3.0, 5.0, 8.0),
        "adverse_excursion": (0.5, 1.0, 2.0, 3.0, 5.0),
        "positive_probability": 0.68,
        "adverse_probability": 0.12,
        "unpredictable_probability": 0.10,
    }
    unpredictable = {
        "payoff": (-8.0, -3.0, 1.0, 6.0, 12.0),
        "adverse_excursion": (2.0, 5.0, 10.0, 18.0, 30.0),
        "positive_probability": 0.52,
        "adverse_probability": 0.46,
        "unpredictable_probability": 0.84,
        "spread": (0.5, 1.0, 2.0, 3.0),
        "volatility": (1.0, 2.0, 3.5, 4.5),
        "flow": (2.0, -2.5, 3.0, -3.5),
        "depth": (-1.5, 2.0, -2.5, 3.0),
    }
    cases = (
        Round74AIContractCase(
            case_id="benign_mirror_long",
            side="long",
            expected_behavior="retain",
            required_any_reason_codes=(),
            request_template=_template(side="long", **benign),
        ),
        Round74AIContractCase(
            case_id="benign_mirror_short",
            side="short",
            expected_behavior="retain",
            required_any_reason_codes=(),
            request_template=_template(side="short", **benign),
        ),
        Round74AIContractCase(
            case_id="unpredictable_mirror_long",
            side="long",
            expected_behavior="constrain",
            required_any_reason_codes=(
                "flow_instability",
                "forecast_uncertainty",
                "regime_unpredictability",
            ),
            request_template=_template(side="long", **unpredictable),
        ),
        Round74AIContractCase(
            case_id="unpredictable_mirror_short",
            side="short",
            expected_behavior="constrain",
            required_any_reason_codes=(
                "flow_instability",
                "forecast_uncertainty",
                "regime_unpredictability",
            ),
            request_template=_template(side="short", **unpredictable),
        ),
        Round74AIContractCase(
            case_id="wide_spread_thin_liquidity",
            side="long",
            expected_behavior="constrain",
            required_any_reason_codes=("liquidity_thin", "spread_wide"),
            request_template=_template(
                side="long",
                payoff=(-4.0, -1.0, 1.0, 3.0, 7.0),
                adverse_excursion=(2.0, 4.0, 7.0, 12.0, 20.0),
                positive_probability=0.53,
                adverse_probability=0.58,
                unpredictable_probability=0.35,
                spread=(2.0, 3.0, 4.0, 5.0),
                volatility=(0.5, 1.0, 1.5, 2.0),
                flow=(0.2, 0.1, 0.0, -0.2),
                depth=(-2.0, -2.5, -3.0, -3.5),
                feature_overrides={
                    "spread_bps": 5.0,
                    "log1p_bid_depth_quote_20": -4.0,
                    "log1p_ask_depth_quote_20": -4.0,
                },
            ),
        ),
        Round74AIContractCase(
            case_id="adverse_selection_conflict",
            side="long",
            expected_behavior="constrain",
            required_any_reason_codes=(
                "adverse_selection",
                "model_inconsistency",
            ),
            request_template=_template(
                side="long",
                payoff=(-2.0, 1.0, 4.0, 8.0, 15.0),
                adverse_excursion=(4.0, 8.0, 14.0, 22.0, 35.0),
                positive_probability=0.72,
                adverse_probability=0.89,
                unpredictable_probability=0.28,
                flow=(1.0, 1.5, 2.0, -2.5),
                depth=(0.5, 0.0, -1.0, -2.0),
            ),
        ),
        Round74AIContractCase(
            case_id="stale_market_state",
            side="long",
            expected_behavior="constrain",
            required_any_reason_codes=("forecast_uncertainty", "stale_state"),
            request_template=_template(
                side="long",
                payoff=(-1.0, 1.0, 3.0, 6.0, 10.0),
                adverse_excursion=(1.0, 3.0, 6.0, 10.0, 18.0),
                positive_probability=0.62,
                adverse_probability=0.32,
                unpredictable_probability=0.44,
                feature_overrides={
                    "depth_update_is_stale": 5.0,
                    "log1p_ms_since_depth_update": 5.0,
                    "log1p_ms_since_book_ticker": 5.0,
                    "log1p_ms_since_aggregate_trade": 5.0,
                    "log1p_ms_since_mark_price": 5.0,
                },
            ),
        ),
        Round74AIContractCase(
            case_id="forecast_model_inconsistency",
            side="short",
            expected_behavior="constrain",
            required_any_reason_codes=(
                "forecast_uncertainty",
                "model_inconsistency",
            ),
            request_template=_template(
                side="short",
                payoff=(-14.0, -8.0, -2.0, 4.0, 16.0),
                adverse_excursion=(1.0, 2.0, 4.0, 8.0, 16.0),
                positive_probability=0.82,
                adverse_probability=0.10,
                unpredictable_probability=0.12,
                flow=(0.2, -0.2, 0.2, -0.2),
                depth=(-0.2, 0.2, -0.2, 0.2),
            ),
        ),
    )
    if tuple(case.case_id for case in cases) != ROUND74_AI_CONTRACT_CASE_IDS:
        raise ValueError("Round 74 AI contract case order differs")
    for case in cases:
        case.validate()
    return cases


def evaluate_round74_ai_contract_outcome(
    case: Round74AIContractCase,
    outcome: Round74AIRuntimeOutcome,
) -> dict[str, object]:
    """Score schema/runtime acceptance separately from safety semantics."""

    case.validate()
    outcome.validate()
    decision = None
    if outcome.worker_result is not None:
        decision = Round74AIWorkerResult.from_dict(outcome.worker_result).decision
    runtime_accepted = outcome.status == "accepted" and decision is not None
    fail_closed = outcome.approved_risk_size_bps == 0 if not runtime_accepted else True
    semantic_passed = False
    if runtime_accepted and decision is not None:
        if case.expected_behavior == "retain":
            semantic_passed = (
                decision.verdict in {"allow_unchanged", "reduce"}
                and decision.size_multiplier_bps >= _RETAIN_MINIMUM_MULTIPLIER_BPS
            )
        else:
            semantic_passed = (
                decision.verdict in {"reduce", "veto", "abstain"}
                and bool(
                    set(decision.reason_codes).intersection(
                        case.required_any_reason_codes
                    )
                )
            )
    return {
        "case_id": case.case_id,
        "case_sha256": case.case_sha256,
        "prompt_payload_sha256": case.build_request(
            1_800_000_000_000_000_000
        ).prompt_payload()["prompt_payload_sha256"],
        "expected_behavior": case.expected_behavior,
        "required_any_reason_codes": list(case.required_any_reason_codes),
        "runtime_status": outcome.status,
        "runtime_accepted": runtime_accepted,
        "runtime_failure_class": outcome.failure_class,
        "fail_closed": fail_closed,
        "semantic_passed": semantic_passed,
        "approved_risk_size_bps": outcome.approved_risk_size_bps,
        "elapsed_ns": outcome.elapsed_ns,
        "decision": None if decision is None else decision.as_dict(),
        "synthetic_non_market_contract_packet": True,
        "financial_edge_tested": False,
        "profitability_claim": False,
    }


__all__ = [
    "ROUND74_AI_CONTRACT_CASE_IDS",
    "ROUND74_AI_CONTRACT_CASE_SCHEMA_VERSION",
    "ROUND74_AI_CONTRACT_SCREEN_SCHEMA_VERSION",
    "Round74AIContractCase",
    "evaluate_round74_ai_contract_outcome",
    "round74_ai_contract_cases",
]
