"""Shared supplied-snapshot cash-loss evaluation; no order or account authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .binance_open_intents import OpenIntentError
from .positions import ClosedTrade, OpenPosition, compute_snapshot_stats


@dataclass(frozen=True)
class CapitalGuard:
    """Hard local capital-at-risk decision for autonomous execution."""

    allowed: bool
    reason: str
    daily_loss: float
    session_loss: float
    consecutive_losses: int
    force_close: bool = False
    realized_peak_drawdown: float | None = None
    capital_policy_sha256: str | None = None


def finite_real_number(value: object) -> float | None:
    """Accept finite real decision values without boolean or string coercion."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class CapitalLossLimits:
    max_daily_loss_pct: float
    max_session_loss_pct: float
    max_drawdown_limit: float
    max_consecutive_losses: int
    reference_cash_quote: float

    def loss_limits_valid(self) -> bool:
        values = tuple(
            finite_real_number(v)
            for v in (
                self.max_daily_loss_pct,
                self.max_session_loss_pct,
                self.max_drawdown_limit,
            )
        )
        return (
            all(v is not None and v >= 0 for v in values)
            and type(self.max_consecutive_losses) is int
            and self.max_consecutive_losses >= 0
        )


def utc_day_index(timestamp_ms: int) -> int:
    return int(timestamp_ms // (24 * 60 * 60 * 1000))


def evaluate_capital_snapshot(
    snapshot: tuple[list[OpenPosition], list[ClosedTrade]],
    mark_price: float | None,
    limits: CapitalLossLimits,
    *,
    now_ms_value: int,
    mark_symbol: str = "",
    mark_market_type: str = "",
    policy_digest: str | None = None,
    policy_blocked: bool = False,
) -> CapitalGuard:
    """Evaluate a strictly decoded ledger snapshot, without loading or mutating storage.

    Supplied policy context is diagnostic/admission input from the caller, never
    independent evidence of native account, margin or dispatch qualification.
    """
    if not limits.loss_limits_valid():
        return CapitalGuard(False, "risk-policy-loss-limits-unqualified", 0.0, 0.0, 0)
    mark = finite_real_number(mark_price)
    if mark is not None and mark <= 0:
        mark = None
    opens, closed = snapshot
    consecutive_losses = 0
    for trade in reversed(closed):
        if trade.realized_pnl >= 0.0:
            break
        consecutive_losses += 1
    try:
        reference = float(limits.reference_cash_quote)
    except (TypeError, ValueError, OverflowError):
        reference = float("nan")
    if (
        isinstance(limits.reference_cash_quote, bool)
        or not math.isfinite(reference)
        or reference <= 0
    ):
        return CapitalGuard(
            False, "cash-valuation-reference-unqualified", 0.0, 0.0, consecutive_losses
        )
    # A scalar mark can value only its exact instrument. Closing inventory does
    # not erase the denomination of realized cash, including legacy records.
    if opens or closed:
        if (
            not mark_symbol
            or mark_market_type not in {"spot", "futures"}
            or opens
            and (
                mark is None
                or any(
                    (position.symbol, position.market_type)
                    != (mark_symbol, mark_market_type)
                    for position in opens
                )
            )
        ):
            return CapitalGuard(
                False,
                "cash-valuation-instrument-unqualified",
                0.0,
                0.0,
                consecutive_losses,
            )
        from .assets import symbol_base_for_supported_quote

        base = symbol_base_for_supported_quote(mark_symbol)
        quote = mark_symbol[len(base) :] if base else ""
        if not quote or any(not trade.symbol.endswith(quote) for trade in closed):
            return CapitalGuard(
                False, "cash-valuation-quote-unqualified", 0.0, 0.0, consecutive_losses
            )
    try:
        stats = compute_snapshot_stats(
            snapshot,
            mark_price=mark,
            starting_reference_cash=limits.reference_cash_quote,
            include_native_entry_costs=True,
        )
    except OpenIntentError:
        return CapitalGuard(
            False, "cash-valuation-fees-unqualified", 0.0, 0.0, consecutive_losses
        )
    day = utc_day_index(now_ms_value)
    realized_today = sum(
        trade.realized_pnl
        for trade in closed
        if utc_day_index(trade.closed_at_ms) == day
    )
    unrealized = float(stats.unrealized_pnl) if mark is not None else 0.0
    daily_cash = realized_today + unrealized
    session_cash = float(stats.realized_pnl) + unrealized
    # Finite records can overflow when aggregated. NaN must never become zero
    # loss through max(), nor can infinite profit offset real losses.
    if not all(math.isfinite(value) for value in (daily_cash, session_cash)):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    daily_loss = max(0.0, -daily_cash / reference)
    session_loss = max(0.0, -session_cash / reference)
    if not math.isfinite(daily_loss) or not math.isfinite(session_loss):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    from .capital_drawdown import settled_cash_drawdown

    try:
        cash_peak = settled_cash_drawdown(
            closed,
            unrealized_pnl_quote=unrealized,
            reference_cash_quote=reference,
            observed_at_ms=now_ms_value,
        )
        drawdown = float(cash_peak.reference_fraction)
    except (ValueError, TypeError):
        return CapitalGuard(
            False, "cash-valuation-history-unqualified", 0.0, 0.0, consecutive_losses
        )
    except OverflowError:
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    if not math.isfinite(drawdown):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    if limits.max_daily_loss_pct > 0.0 and daily_loss >= limits.max_daily_loss_pct:
        return CapitalGuard(
            False,
            f"daily-loss-lockout:{daily_loss:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if (
        limits.max_session_loss_pct > 0.0
        and session_loss >= limits.max_session_loss_pct
    ):
        return CapitalGuard(
            False,
            f"session-loss-lockout:{session_loss:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if limits.max_drawdown_limit > 0.0 and drawdown >= limits.max_drawdown_limit:
        return CapitalGuard(
            False,
            f"drawdown-lockout:{drawdown:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if (
        limits.max_consecutive_losses > 0
        and consecutive_losses >= limits.max_consecutive_losses
    ):
        return CapitalGuard(
            False,
            f"loss-streak-lockout:{consecutive_losses}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=False,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    return CapitalGuard(
        not policy_blocked,
        "risk-policy-activation-unqualified" if policy_blocked else "allowed",
        daily_loss,
        session_loss,
        consecutive_losses,
        realized_peak_drawdown=drawdown,
        capital_policy_sha256=policy_digest,
    )
