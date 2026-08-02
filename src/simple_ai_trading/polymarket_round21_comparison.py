"""Exact-population economic comparison for optional Round 21 predictors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import re
from typing import Sequence

from .polymarket_round21_replay import (
    POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
    Round21EconomicReplay,
    Round21ReplayCondition,
    replay_round21_full_matrix,
    round21_daily_lower_95,
)


POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION = (
    "polymarket-round21-matched-economic-comparison-v1"
)
POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256 = (
    "ede0d3cd989cf62116c1882b84b7fd3bdc211c34b8f3ea98dffd6bda48b20c10"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPTIONAL_LAYERS = {"core_spot", "core_spot_usdm"}
_DAY_MS = 86_400_000


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class Round21MatchedReplayDelta:
    profile: str
    scenario: str
    baseline_replay_sha256: str
    challenger_replay_sha256: str
    matched_condition_count: int
    net_pnl_delta_quote: Decimal
    mean_condition_utility_delta_quote: Decimal
    daily_mean_delta_lower_95_quote: Decimal | None
    maximum_drawdown_delta_fraction: Decimal
    tail_mean_delta_quote: Decimal
    accepted: bool
    reasons: tuple[str, ...]
    delta_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256,
            "profile": self.profile,
            "scenario": self.scenario,
            "baseline_replay_sha256": self.baseline_replay_sha256,
            "challenger_replay_sha256": self.challenger_replay_sha256,
            "matched_condition_count": self.matched_condition_count,
            "net_pnl_delta_quote": format(self.net_pnl_delta_quote, "f"),
            "mean_condition_utility_delta_quote": format(
                self.mean_condition_utility_delta_quote,
                "f",
            ),
            "daily_mean_delta_lower_95_quote": (
                None
                if self.daily_mean_delta_lower_95_quote is None
                else format(self.daily_mean_delta_lower_95_quote, "f")
            ),
            "maximum_drawdown_delta_fraction": format(
                self.maximum_drawdown_delta_fraction,
                "f",
            ),
            "tail_mean_delta_quote": format(self.tail_mean_delta_quote, "f"),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }

    def validated(self) -> Round21MatchedReplayDelta:
        decimals = (
            self.net_pnl_delta_quote,
            self.mean_condition_utility_delta_quote,
            self.maximum_drawdown_delta_fraction,
            self.tail_mean_delta_quote,
        )
        if (
            self.matched_condition_count < 1
            or any(not value.is_finite() for value in decimals)
            or (
                self.daily_mean_delta_lower_95_quote is not None
                and not self.daily_mean_delta_lower_95_quote.is_finite()
            )
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.baseline_replay_sha256,
                    self.challenger_replay_sha256,
                )
            )
            or len(set(self.reasons)) != len(self.reasons)
            or self.accepted != (not self.reasons)
            or self.delta_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 matched replay delta differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21MatchedEconomicComparison:
    challenger_layer: str
    matched_population_sha256: str
    baseline_matrix_sha256: str
    challenger_matrix_sha256: str
    deltas: tuple[Round21MatchedReplayDelta, ...]
    all_replays_accepted: bool
    optional_layer_selected: bool
    comparison_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256,
            "economic_replay_design_sha256": (
                POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256
            ),
            "challenger_layer": self.challenger_layer,
            "matched_population_sha256": self.matched_population_sha256,
            "baseline_matrix_sha256": self.baseline_matrix_sha256,
            "challenger_matrix_sha256": self.challenger_matrix_sha256,
            "delta_sha256": [value.delta_sha256 for value in self.deltas],
            "all_replays_accepted": self.all_replays_accepted,
            "optional_layer_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21MatchedEconomicComparison:
        deltas = tuple(value.validated() for value in self.deltas)
        if (
            self.challenger_layer not in _OPTIONAL_LAYERS
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.matched_population_sha256,
                    self.baseline_matrix_sha256,
                    self.challenger_matrix_sha256,
                )
            )
            or len(deltas) != 81
            or len({(value.profile, value.scenario) for value in deltas}) != 81
            or self.all_replays_accepted != all(value.accepted for value in deltas)
            or self.optional_layer_selected
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.comparison_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 matched economic comparison differs")
        return self


def round21_replay_matrix_sha256(
    values: Sequence[Round21EconomicReplay],
) -> str:
    selected = tuple(value.validated() for value in values)
    return _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
            "replay_sha256": [value.replay_sha256 for value in selected],
        }
    )


def paired_round21_replay_delta(
    baseline: Round21EconomicReplay,
    challenger: Round21EconomicReplay,
) -> Round21MatchedReplayDelta:
    base = baseline.validated()
    challenge = challenger.validated()
    base_conditions = base.conditions
    challenge_conditions = challenge.conditions
    if (
        base.profile != challenge.profile
        or base.scenario != challenge.scenario
        or base.initial_capital_quote != challenge.initial_capital_quote
        or len(base_conditions) != len(challenge_conditions)
        or any(
            (
                left.condition_id,
                left.event_start_ms,
                left.outcome_sha256,
            )
            != (
                right.condition_id,
                right.event_start_ms,
                right.outcome_sha256,
            )
            for left, right in zip(
                base_conditions,
                challenge_conditions,
                strict=True,
            )
        )
    ):
        raise ValueError("Round 21 matched replay population differs")
    condition_deltas = tuple(
        right.utility_quote - left.utility_quote
        for left, right in zip(base_conditions, challenge_conditions, strict=True)
    )
    daily: dict[int, Decimal] = {}
    for condition, delta in zip(base_conditions, condition_deltas, strict=True):
        day = condition.event_start_ms // _DAY_MS
        daily[day] = daily.get(day, Decimal("0")) + delta
    lower = round21_daily_lower_95(
        tuple(daily.values()),
        identity=f"{base.replay_sha256}:{challenge.replay_sha256}",
    )
    net_delta = challenge.metrics.net_pnl_quote - base.metrics.net_pnl_quote
    mean_delta = (
        sum(condition_deltas, start=Decimal("0")) / len(condition_deltas)
        if condition_deltas
        else Decimal("0")
    )
    drawdown_delta = (
        challenge.metrics.maximum_drawdown_fraction
        - base.metrics.maximum_drawdown_fraction
    )
    tail_delta = (
        challenge.metrics.tail_mean_worst_five_percent_quote
        - base.metrics.tail_mean_worst_five_percent_quote
    )
    reasons: list[str] = []
    if not base.economic_gate_passed:
        reasons.append("baseline_core_economic_gate_not_passed")
    if not challenge.economic_gate_passed:
        reasons.append("challenger_economic_gate_not_passed")
    if base.unknown_state_count or challenge.unknown_state_count:
        reasons.append("unknown_post_submit_state")
    if base.risk_violation_count or challenge.risk_violation_count:
        reasons.append("risk_limit_violation")
    if net_delta <= 0:
        reasons.append("net_pnl_delta_not_positive")
    if mean_delta <= 0:
        reasons.append("mean_condition_utility_delta_not_positive")
    if lower is None or lower <= 0:
        reasons.append("daily_delta_lower_95_not_positive")
    if drawdown_delta > 0:
        reasons.append("maximum_drawdown_worse")
    if tail_delta < 0:
        reasons.append("tail_loss_worse")
    provisional = Round21MatchedReplayDelta(
        profile=base.profile,
        scenario=base.scenario,
        baseline_replay_sha256=base.replay_sha256,
        challenger_replay_sha256=challenge.replay_sha256,
        matched_condition_count=len(condition_deltas),
        net_pnl_delta_quote=net_delta,
        mean_condition_utility_delta_quote=mean_delta,
        daily_mean_delta_lower_95_quote=lower,
        maximum_drawdown_delta_fraction=drawdown_delta,
        tail_mean_delta_quote=tail_delta,
        accepted=not reasons,
        reasons=tuple(reasons),
        delta_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        delta_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def compare_round21_optional_full_matrix(
    *,
    baseline_conditions: Sequence[Round21ReplayCondition],
    challenger_conditions: Sequence[Round21ReplayCondition],
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
) -> Round21MatchedEconomicComparison:
    """Compare optional public Binance features without any Binance execution."""

    baseline = tuple(value.validated() for value in baseline_conditions)
    challenger = tuple(value.validated() for value in challenger_conditions)
    if len(baseline) != len(challenger) or not baseline:
        raise ValueError("Round 21 matched input population differs")
    challenger_layers = {
        envelope.model_layer
        for condition in challenger
        for envelope in condition.envelopes
    }
    if (
        any(
            envelope.model_layer != "core"
            for condition in baseline
            for envelope in condition.envelopes
        )
        or len(challenger_layers) != 1
        or not challenger_layers.issubset(_OPTIONAL_LAYERS)
        or any(
            left.matched_population_sha256()
            != right.matched_population_sha256()
            for left, right in zip(baseline, challenger, strict=True)
        )
    ):
        raise ValueError("Round 21 matched input population differs")
    challenger_layer = next(iter(challenger_layers))
    matched_population_sha = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
            "condition_sha256": [
                value.matched_population_sha256() for value in baseline
            ],
        }
    )
    baseline_matrix = replay_round21_full_matrix(
        baseline,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
    )
    challenger_matrix = replay_round21_full_matrix(
        challenger,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
    )
    deltas = tuple(
        paired_round21_replay_delta(left, right)
        for left, right in zip(baseline_matrix, challenger_matrix, strict=True)
    )
    provisional = Round21MatchedEconomicComparison(
        challenger_layer=challenger_layer,
        matched_population_sha256=matched_population_sha,
        baseline_matrix_sha256=round21_replay_matrix_sha256(baseline_matrix),
        challenger_matrix_sha256=round21_replay_matrix_sha256(challenger_matrix),
        deltas=deltas,
        all_replays_accepted=all(value.accepted for value in deltas),
        optional_layer_selected=False,
        comparison_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        comparison_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256",
    "POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION",
    "Round21MatchedEconomicComparison",
    "Round21MatchedReplayDelta",
    "compare_round21_optional_full_matrix",
    "paired_round21_replay_delta",
    "round21_replay_matrix_sha256",
]
