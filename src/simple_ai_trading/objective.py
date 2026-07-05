"""Objective functions for ranking backtest / walk-forward results.

Three presets ship out of the box: ``conservative``, ``regular``, and ``aggressive``.
each expressing a different risk-adjusted view of a candidate strategy.  They
are pure functions of a ``BacktestResult`` so they can be reused from the CLI,
the training suite, and the autonomous loop's sanity gates without import loops.

The scorers all return values in roughly the same range so the training suite
can compare across them when the user wants to see "how would the conservative
pick score under the aggressive lens".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .assets import (
    DEFAULT_AGGRESSIVE_LEVERAGE,
    DEFAULT_CONSERVATIVE_LEVERAGE,
    DEFAULT_REGULAR_LEVERAGE,
)
from .backtest import BacktestResult, closed_trades_per_day, trade_activity_satisfies


@dataclass(frozen=True)
class ObjectiveSpec:
    """A named scoring function with operator-facing metadata."""

    name: str
    label: str
    summary: str
    long_description: str
    scorer: Callable[[BacktestResult], float]
    min_closed_trades: int = 3
    min_trades_per_day: float = 0.0
    target_trades_per_day: float = 0.0
    min_realized_pnl: float | None = None
    min_edge_vs_buy_hold: float | None = 0.0
    min_market_edge_pct: float | None = 0.001
    max_drawdown_rejection: float = 1.0  # 1.0 = never reject on drawdown alone
    min_profit_factor: float | None = None
    min_expectancy: float | None = None
    max_consecutive_losses_allowed: int | None = None
    max_single_trade_profit_share_allowed: float | None = None
    training: "ObjectiveTraining | None" = None

    def score(self, result: BacktestResult) -> float:
        return float(self.scorer(result))

    def accepts(self, result: BacktestResult) -> bool:
        """Return False when a candidate fails hard gates — used by tuning."""

        return not self.rejection_reasons(result)

    def rejection_reasons(self, result: BacktestResult) -> list[str]:
        """Return stable machine-readable reasons for hard-gate rejection."""

        reasons: list[str] = []
        if bool(getattr(result, "stopped_by_liquidation", False)) or int(getattr(result, "liquidation_events", 0)) > 0:
            reasons.append("liquidation_events>0")
        activity_targeted = self.min_closed_trades > 0 or self.min_trades_per_day > 0.0
        if activity_targeted and not trade_activity_satisfies(
            result,
            min_closed_trades=self.min_closed_trades,
            min_trades_per_day=self.min_trades_per_day,
        ):
            if result.closed_trades < max(0, int(self.min_closed_trades)):
                reasons.append(f"closed_trades<{self.min_closed_trades}")
            elif self.min_trades_per_day > 0.0:
                reasons.append(f"trades_per_day<{self.min_trades_per_day}")
        if self.min_realized_pnl is not None and result.realized_pnl <= self.min_realized_pnl:
            reasons.append(f"realized_pnl<={self.min_realized_pnl}")
        if self.min_edge_vs_buy_hold is not None and result.edge_vs_buy_hold < self.min_edge_vs_buy_hold:
            reasons.append(f"edge_vs_buy_hold<{self.min_edge_vs_buy_hold}")
        edge_ratio = _market_edge_ratio(result)
        if self.min_market_edge_pct is not None and edge_ratio < self.min_market_edge_pct:
            reasons.append(f"market_edge_pct<{self.min_market_edge_pct}")
        if self.max_drawdown_rejection < 1.0 and result.max_drawdown > self.max_drawdown_rejection:
            reasons.append(f"max_drawdown>{self.max_drawdown_rejection}")
        if result.stopped_by_drawdown and self.max_drawdown_rejection < 0.5:
            reasons.append("stopped_by_drawdown")
        path_quality = _path_quality_evidence(result)
        if path_quality is not None:
            profit_factor = path_quality.profit_factor
            expectancy = path_quality.expectancy
            max_consecutive_losses = path_quality.max_consecutive_losses
            if self.min_profit_factor is not None and profit_factor < self.min_profit_factor:
                reasons.append(f"profit_factor<{self.min_profit_factor}")
            if self.min_expectancy is not None and expectancy <= self.min_expectancy:
                reasons.append(f"expectancy<={self.min_expectancy}")
            if (
                self.max_consecutive_losses_allowed is not None
                and max_consecutive_losses > self.max_consecutive_losses_allowed
            ):
                reasons.append(f"max_consecutive_losses>{self.max_consecutive_losses_allowed}")
            if (
                self.max_single_trade_profit_share_allowed is not None
                and path_quality.max_single_trade_profit_share > self.max_single_trade_profit_share_allowed
            ):
                reasons.append(
                    f"single_trade_profit_share>{self.max_single_trade_profit_share_allowed}"
                )
        return reasons

    def reject_reason(self, result: BacktestResult) -> str | None:
        """Return a semicolon-delimited hard-gate reason for artifact payloads."""

        reasons = self.rejection_reasons(result)
        return "; ".join(reasons) if reasons else None


@dataclass(frozen=True)
class ObjectiveTraining:
    """Per-objective hyperparameter + strategy defaults used by the training suite.

    These presets describe the intent of each objective — for example, the
    conservative profile trains on fewer epochs with stronger L2 regularization
    because it preferences model stability over ultimate score, while the
    aggressive profile pushes more epochs and a softer penalty to chase marginal
    gains.
    """

    epochs: int
    learning_rate: float
    l2_penalty: float
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade: float
    max_position_pct: float
    max_trades_per_day: int
    leverage: float
    cooldown_minutes: int
    calibrate_threshold: bool
    walk_forward_train: int
    walk_forward_test: int
    walk_forward_step: int
    polynomial_degree: int = 2
    polynomial_top_features: int = 6
    extra_lookback_windows: tuple[int, ...] = (5, 20, 60)


def _safe(value: float, default: float = 0.0) -> float:
    return default if not math.isfinite(value) else float(value)


def _loss_streak(values: list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


@dataclass(frozen=True)
class _PathQualityEvidence:
    profit_factor: float
    expectancy: float
    max_consecutive_losses: int
    max_single_trade_profit_share: float


def _single_trade_profit_share(trade_pnls: object, gross_profit: float, closed_trades: int) -> float:
    if gross_profit <= 0.0 or not isinstance(trade_pnls, (tuple, list)):
        return 0.0
    if int(closed_trades) > 0 and len(trade_pnls) != int(closed_trades):
        return 0.0
    positive = [
        float(value)
        for value in trade_pnls
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not positive:
        return 0.0
    return max(positive) / gross_profit


def _path_quality_evidence(result: BacktestResult) -> _PathQualityEvidence | None:
    gross_profit = _safe(float(getattr(result, "gross_profit", 0.0)))
    gross_loss = _safe(float(getattr(result, "gross_loss", 0.0)))
    profit_factor = _safe(float(getattr(result, "profit_factor", 0.0)))
    expectancy = _safe(float(getattr(result, "expectancy", 0.0)))
    max_consecutive_losses = int(getattr(result, "max_consecutive_losses", 0))
    trade_pnls = getattr(result, "trade_pnls", ())
    closed_trades = int(getattr(result, "closed_trades", 0))
    if (
        gross_profit > 0.0
        or gross_loss > 0.0
        or profit_factor > 0.0
        or expectancy != 0.0
        or max_consecutive_losses > 0
    ):
        return _PathQualityEvidence(
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_consecutive_losses=max_consecutive_losses,
            max_single_trade_profit_share=_single_trade_profit_share(
                trade_pnls,
                gross_profit,
                closed_trades,
            ),
        )

    if not isinstance(trade_pnls, (tuple, list)) or len(trade_pnls) == 0:
        return None
    clean_pnls = [float(value) for value in trade_pnls if math.isfinite(float(value))]
    if not clean_pnls:
        return None
    gross_profit = sum(value for value in clean_pnls if value > 0.0)
    gross_loss = abs(sum(value for value in clean_pnls if value < 0.0))
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    expectancy = sum(clean_pnls) / len(clean_pnls)
    profit_factor = min(999.0, max(0.0, profit_factor))
    return _PathQualityEvidence(
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_consecutive_losses=_loss_streak(clean_pnls),
        max_single_trade_profit_share=_single_trade_profit_share(
            clean_pnls,
            gross_profit,
            closed_trades,
        ),
    )


def _return_ratio(result: BacktestResult) -> float:
    if result.starting_cash <= 0:
        return 0.0
    return _safe(result.realized_pnl / result.starting_cash)


def _trade_frequency_fit(result: BacktestResult, target_trades_per_day: float) -> float:
    target = max(0.0, _safe(target_trades_per_day))
    if target <= 0.0:
        return 0.0
    trades_per_day = closed_trades_per_day(result)
    return max(0.0, min(1.0, trades_per_day / target))


def _cap_hit_penalty(result: BacktestResult, weight: float) -> float:
    cap_hits = max(0.0, _safe(float(getattr(result, "trades_per_day_cap_hit", 0))))
    return min(0.10, cap_hits * max(0.0, weight))


def _market_edge_ratio(result: BacktestResult) -> float:
    """Return net edge over the same-notional buy/hold baseline as capital pct."""

    starting_cash = _safe(float(getattr(result, "starting_cash", 0.0)))
    if starting_cash <= 0.0:
        return 0.0
    return _safe(float(getattr(result, "edge_vs_buy_hold", 0.0))) / starting_cash


def _conservative_scorer(result: BacktestResult) -> float:
    """Reward steady, low-drawdown returns with a modest preference for win rate.

    Formula (roughly Calmar-like):
        return_ratio − 3·max_drawdown − 0.005·trades + 0.1·win_rate
    plus a hard penalty for runs that tripped the drawdown circuit breaker.
    """

    penalty = 0.25 if result.stopped_by_drawdown else 0.0
    if bool(getattr(result, "stopped_by_liquidation", False)) or int(getattr(result, "liquidation_events", 0)) > 0:
        penalty += 1.00
    return (
        _return_ratio(result)
        - 3.0 * _safe(result.max_drawdown)
        + 0.10 * _safe(result.win_rate)
        + 0.04 * _trade_frequency_fit(result, CONSERVATIVE.target_trades_per_day)
        - _cap_hit_penalty(result, 0.0015)
        - penalty
    )


def _default_scorer(result: BacktestResult) -> float:
    """Balanced risk-adjusted return scorer.

    Formula:
        return_ratio − 1.5·max_drawdown − 0.0005·trades + 0.15·win_rate
    """

    penalty = 0.10 if result.stopped_by_drawdown else 0.0
    if bool(getattr(result, "stopped_by_liquidation", False)) or int(getattr(result, "liquidation_events", 0)) > 0:
        penalty += 0.75
    return (
        _return_ratio(result)
        - 1.5 * _safe(result.max_drawdown)
        + 0.15 * _safe(result.win_rate)
        + 0.04 * _trade_frequency_fit(result, REGULAR.target_trades_per_day)
        - _cap_hit_penalty(result, 0.0010)
        - penalty
    )


def _risky_scorer(result: BacktestResult) -> float:
    """Reward raw return more heavily and tolerate bigger drawdowns / more trades."""

    penalty = 0.05 if result.stopped_by_drawdown else 0.0
    if bool(getattr(result, "stopped_by_liquidation", False)) or int(getattr(result, "liquidation_events", 0)) > 0:
        penalty += 0.50
    return (
        1.25 * _return_ratio(result)
        - 0.8 * _safe(result.max_drawdown)
        + 0.05 * _safe(result.win_rate)
        + 0.03 * _trade_frequency_fit(result, AGGRESSIVE.target_trades_per_day)
        - _cap_hit_penalty(result, 0.0008)
        - penalty
    )


CONSERVATIVE = ObjectiveSpec(
    name="conservative",
    label="Conservative",
    summary="Prioritize capital preservation while requiring enough risk-gated day trades.",
    long_description=(
        "Training uses extra regularization and calibrated thresholds. Strategy "
        "defaults favor small position sizes, longer cooldowns, and a higher signal "
        "threshold so the bot stays out of coin-flip regimes."
    ),
    scorer=_conservative_scorer,
    min_closed_trades=5,
    min_trades_per_day=2.0,
    target_trades_per_day=12.0,
    min_realized_pnl=0.0,
    min_market_edge_pct=0.0020,
    max_drawdown_rejection=0.15,
    min_profit_factor=1.10,
    min_expectancy=0.0,
    max_consecutive_losses_allowed=3,
    max_single_trade_profit_share_allowed=0.55,
    training=ObjectiveTraining(
        epochs=400,
        learning_rate=0.02,
        l2_penalty=5e-3,
        signal_threshold=0.66,
        stop_loss_pct=0.010,
        take_profit_pct=0.022,
        risk_per_trade=0.005,
        max_position_pct=0.10,
        max_trades_per_day=24,
        leverage=DEFAULT_CONSERVATIVE_LEVERAGE,
        cooldown_minutes=15,
        calibrate_threshold=True,
        walk_forward_train=400,
        walk_forward_test=90,
        walk_forward_step=30,
        polynomial_degree=2,
        polynomial_top_features=5,
        extra_lookback_windows=(10, 30, 90),
    ),
)

REGULAR = ObjectiveSpec(
    name="regular",
    label="Regular",
    summary="Balanced risk-adjusted return - the middle preset for most operators.",
    long_description=(
        "Balanced training budget and strategy defaults. Targets a middle-of-the-road "
        "Sharpe-like profile on walk-forward evaluation."
    ),
    scorer=_default_scorer,
    min_closed_trades=3,
    min_trades_per_day=4.0,
    target_trades_per_day=20.0,
    min_realized_pnl=0.0,
    min_market_edge_pct=0.0030,
    max_drawdown_rejection=0.25,
    min_profit_factor=1.05,
    min_expectancy=0.0,
    max_consecutive_losses_allowed=5,
    max_single_trade_profit_share_allowed=0.65,
    training=ObjectiveTraining(
        epochs=600,
        learning_rate=0.03,
        l2_penalty=1e-3,
        signal_threshold=0.58,
        stop_loss_pct=0.018,
        take_profit_pct=0.030,
        risk_per_trade=0.010,
        max_position_pct=0.20,
        max_trades_per_day=48,
        leverage=DEFAULT_REGULAR_LEVERAGE,
        cooldown_minutes=7,
        calibrate_threshold=True,
        walk_forward_train=500,
        walk_forward_test=120,
        walk_forward_step=40,
        polynomial_degree=2,
        polynomial_top_features=13,
        extra_lookback_windows=(5, 20, 60),
    ),
)

AGGRESSIVE = ObjectiveSpec(
    name="aggressive",
    label="Aggressive",
    summary="Seek higher returns with strict capital-preservation gates.",
    long_description=(
        "Uses a longer training budget with a softer L2 and a lower signal threshold so "
        "the model takes more shots. Strategy defaults remain capped so this preset is "
        "aggressive relative to Regular without allowing all-in exposure. Keep this preset "
        "on testnet until its live behavior is understood."
    ),
    scorer=_risky_scorer,
    min_closed_trades=2,
    min_trades_per_day=6.0,
    target_trades_per_day=30.0,
    min_realized_pnl=0.0,
    min_market_edge_pct=0.0050,
    max_drawdown_rejection=0.30,
    min_profit_factor=1.00,
    min_expectancy=0.0,
    max_consecutive_losses_allowed=8,
    max_single_trade_profit_share_allowed=0.75,
    training=ObjectiveTraining(
        epochs=720,
        learning_rate=0.04,
        l2_penalty=5e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.024,
        take_profit_pct=0.040,
        risk_per_trade=0.012,
        max_position_pct=0.25,
        max_trades_per_day=72,
        leverage=DEFAULT_AGGRESSIVE_LEVERAGE,
        cooldown_minutes=4,
        calibrate_threshold=True,
        walk_forward_train=600,
        walk_forward_test=150,
        walk_forward_step=50,
        polynomial_degree=3,
        polynomial_top_features=9,
        extra_lookback_windows=(3, 15, 45, 120),
    ),
)

DEFAULT = CONSERVATIVE
RISKY = AGGRESSIVE


_REGISTRY: dict[str, ObjectiveSpec] = {
    CONSERVATIVE.name: CONSERVATIVE,
    REGULAR.name: REGULAR,
    AGGRESSIVE.name: AGGRESSIVE,
}
_ALIASES: dict[str, str] = {
    "balanced": REGULAR.name,
    "default": CONSERVATIVE.name,
    "risky": AGGRESSIVE.name,
}


def available_objectives() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def get_objective(name: str) -> ObjectiveSpec:
    """Look up an objective by case-insensitive name."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("Objective name cannot be empty")
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        allowed = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown objective {name!r}. Known: {allowed}")
    return _REGISTRY[key]


def describe_objectives() -> list[dict[str, str]]:
    """Return a display-friendly description for every registered objective."""

    return [
        {
            "name": spec.name,
            "label": spec.label,
            "summary": spec.summary,
            "long_description": spec.long_description,
        }
        for spec in _REGISTRY.values()
    ]


def rank_candidates(
    candidates: list[tuple[dict[str, object], BacktestResult]],
    objective: ObjectiveSpec,
) -> list[dict[str, object]]:
    """Sort ``candidates`` by objective score, annotate with reasons for rejection.

    Each entry is ``(params_dict, backtest_result)``.  Rejections are kept in
    the output but scored as ``-inf`` so they sink to the bottom; the reason is
    placed under ``"reject_reason"``.
    """

    ranked: list[dict[str, object]] = []
    for params, result in candidates:
        score = objective.score(result)
        accepted = objective.accepts(result)
        reject_reason: str | None = None
        if not accepted:
            reasons = (
                objective.rejection_reasons(result)
                if hasattr(objective, "rejection_reasons")
                else []
            )
            reject_reason = "; ".join(reasons) or "hard-gate-failed"
        ranked.append({
            "params": params,
            "score": score if accepted else float("-inf"),
            "raw_score": score,
            "result": result,
            "accepted": accepted,
            "reject_reason": reject_reason,
        })
    def _rank_score(entry: dict[str, object]) -> float:
        value = entry["score"]
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    ranked.sort(key=_rank_score, reverse=True)
    return ranked
