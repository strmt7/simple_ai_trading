"""Restart-derived autonomous Polymarket daily and drawdown risk state."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

from .polymarket_live import (
    PolymarketConditionAccounting,
    PolymarketLedgerRevision,
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
    PolymarketRealizedPnlEvent,
)


_DAY_MS = 86_400_000
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketLiveRiskProfile:
    name: str
    maximum_event_loss_capital_fraction: Decimal
    maximum_daily_loss_capital_fraction: Decimal
    maximum_drawdown_capital_fraction: Decimal
    loss_cluster_cooldown_minutes: int

    def __post_init__(self) -> None:
        name = str(self.name or "").strip().lower()
        fractions = tuple(
            _decimal(value, name=f"{name} risk fraction")
            for value in (
                self.maximum_event_loss_capital_fraction,
                self.maximum_daily_loss_capital_fraction,
                self.maximum_drawdown_capital_fraction,
            )
        )
        cooldown = int(self.loss_cluster_cooldown_minutes)
        if (
            name not in {"conservative", "regular", "aggressive"}
            or any(not Decimal("0") < value < Decimal("1") for value in fractions)
            or not fractions[0] < fractions[1] < fractions[2]
            or not 1 <= cooldown <= 1_440
        ):
            raise ValueError("Polymarket live risk profile differs")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "maximum_event_loss_capital_fraction",
            fractions[0],
        )
        object.__setattr__(
            self,
            "maximum_daily_loss_capital_fraction",
            fractions[1],
        )
        object.__setattr__(
            self,
            "maximum_drawdown_capital_fraction",
            fractions[2],
        )
        object.__setattr__(self, "loss_cluster_cooldown_minutes", cooldown)


POLYMARKET_LIVE_RISK_PROFILES = tuple(
    PolymarketLiveRiskProfile(name, event, daily, drawdown, cooldown)
    for name, event, daily, drawdown, cooldown in (
        (
            "conservative",
            Decimal("0.001"),
            Decimal("0.005"),
            Decimal("0.02"),
            30,
        ),
        (
            "regular",
            Decimal("0.002"),
            Decimal("0.01"),
            Decimal("0.04"),
            15,
        ),
        (
            "aggressive",
            Decimal("0.0035"),
            Decimal("0.015"),
            Decimal("0.06"),
            5,
        ),
    )
)


def polymarket_live_risk_profile(
    name: str = "conservative",
) -> PolymarketLiveRiskProfile:
    selected = str(name or "conservative").strip().lower()
    matches = tuple(
        profile for profile in POLYMARKET_LIVE_RISK_PROFILES if profile.name == selected
    )
    if len(matches) != 1:
        raise ValueError("Polymarket live risk profile is unknown")
    return matches[0]


@dataclass(frozen=True, slots=True)
class PolymarketLiveRiskState:
    """Hash-bound entry risk reconstructed only from Polymarket evidence."""

    condition_id: str
    risk_profile: str
    risk_capital_quote: Decimal
    observed_at_ms: int
    utc_day_index: int
    ledger_revision: PolymarketLedgerRevision
    realized_event_count: int
    realized_condition_count: int
    daily_realized_pnl_quote: Decimal
    lifetime_realized_pnl_quote: Decimal
    settled_equity_quote: Decimal
    settled_peak_equity_quote: Decimal
    drawdown_capital_fraction: Decimal
    consecutive_losing_conditions: int
    cooldown_until_ms: int
    cooldown_active: bool
    current_condition_inventory_downside_quote: Decimal
    other_condition_inventory_downside_quote: Decimal
    total_inventory_downside_quote: Decimal
    maximum_current_condition_downside_quote: Decimal
    entry_allowed: bool
    entry_block_reasons: tuple[str, ...]
    risk_state_sha256: str = ""

    def __post_init__(self) -> None:
        condition = str(self.condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(condition) is None:
            raise ValueError("Polymarket live risk condition is invalid")
        profile = polymarket_live_risk_profile(self.risk_profile)
        capital = _decimal(self.risk_capital_quote, name="risk capital")
        observed = int(self.observed_at_ms)
        day = int(self.utc_day_index)
        event_count = int(self.realized_event_count)
        condition_count = int(self.realized_condition_count)
        consecutive = int(self.consecutive_losing_conditions)
        cooldown = int(self.cooldown_until_ms)
        if (
            capital <= 0
            or observed <= 0
            or day != observed // _DAY_MS
            or event_count < 0
            or condition_count < 0
            or condition_count > event_count
            or consecutive < 0
            or cooldown < 0
            or bool(self.cooldown_active) != (observed < cooldown)
        ):
            raise ValueError("Polymarket live risk state differs")
        decimals = {
            "daily_realized_pnl_quote": _decimal(
                self.daily_realized_pnl_quote,
                name="daily realized PnL",
            ),
            "lifetime_realized_pnl_quote": _decimal(
                self.lifetime_realized_pnl_quote,
                name="lifetime realized PnL",
            ),
            "settled_equity_quote": _decimal(
                self.settled_equity_quote,
                name="settled equity",
            ),
            "settled_peak_equity_quote": _decimal(
                self.settled_peak_equity_quote,
                name="settled peak equity",
            ),
            "drawdown_capital_fraction": _decimal(
                self.drawdown_capital_fraction,
                name="drawdown fraction",
            ),
            "current_condition_inventory_downside_quote": _decimal(
                self.current_condition_inventory_downside_quote,
                name="current condition downside",
            ),
            "other_condition_inventory_downside_quote": _decimal(
                self.other_condition_inventory_downside_quote,
                name="other condition downside",
            ),
            "total_inventory_downside_quote": _decimal(
                self.total_inventory_downside_quote,
                name="total inventory downside",
            ),
            "maximum_current_condition_downside_quote": _decimal(
                self.maximum_current_condition_downside_quote,
                name="maximum current condition downside",
            ),
        }
        if any(
            decimals[name] < 0
            for name in (
                "settled_peak_equity_quote",
                "drawdown_capital_fraction",
                "current_condition_inventory_downside_quote",
                "other_condition_inventory_downside_quote",
                "total_inventory_downside_quote",
                "maximum_current_condition_downside_quote",
            )
        ):
            raise ValueError("Polymarket live risk amount is invalid")
        if decimals["total_inventory_downside_quote"] != (
            decimals["current_condition_inventory_downside_quote"]
            + decimals["other_condition_inventory_downside_quote"]
        ):
            raise ValueError("Polymarket live aggregate downside differs")
        expected_drawdown = (
            max(
                Decimal("0"),
                decimals["settled_peak_equity_quote"]
                - decimals["settled_equity_quote"],
            )
            / capital
        )
        if decimals["drawdown_capital_fraction"] != expected_drawdown:
            raise ValueError("Polymarket live drawdown accounting differs")
        expected_maximum_current = min(
            capital * profile.maximum_event_loss_capital_fraction,
            max(
                Decimal("0"),
                capital * profile.maximum_daily_loss_capital_fraction
                - max(Decimal("0"), -decimals["daily_realized_pnl_quote"])
                - decimals["other_condition_inventory_downside_quote"],
            ),
            max(
                Decimal("0"),
                capital * profile.maximum_drawdown_capital_fraction
                - max(
                    Decimal("0"),
                    decimals["settled_peak_equity_quote"]
                    - decimals["settled_equity_quote"],
                )
                - decimals["other_condition_inventory_downside_quote"],
            ),
        )
        if (
            decimals["maximum_current_condition_downside_quote"]
            != expected_maximum_current
        ):
            raise ValueError("Polymarket live risk headroom differs")
        reasons = tuple(str(item or "").strip() for item in self.entry_block_reasons)
        if any(not item or len(item) > 96 for item in reasons):
            raise ValueError("Polymarket live risk reason is invalid")
        expected_reasons: list[str] = []
        if (
            max(Decimal("0"), -decimals["daily_realized_pnl_quote"])
            + decimals["total_inventory_downside_quote"]
            >= capital * profile.maximum_daily_loss_capital_fraction
        ):
            expected_reasons.append("daily_loss_limit_reached")
        if (
            max(
                Decimal("0"),
                decimals["settled_peak_equity_quote"]
                - decimals["settled_equity_quote"],
            )
            + decimals["total_inventory_downside_quote"]
            >= capital * profile.maximum_drawdown_capital_fraction
        ):
            expected_reasons.append("drawdown_limit_reached")
        if observed < cooldown:
            expected_reasons.append("loss_cluster_cooldown_active")
        if (
            decimals["current_condition_inventory_downside_quote"]
            >= decimals["maximum_current_condition_downside_quote"]
        ):
            expected_reasons.append("event_or_portfolio_risk_headroom_exhausted")
        if reasons != tuple(expected_reasons):
            raise ValueError("Polymarket live risk reasons differ")
        if bool(self.entry_allowed) != (not reasons):
            raise ValueError("Polymarket live entry gate differs")
        for name, value in decimals.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "condition_id", condition)
        object.__setattr__(self, "risk_profile", profile.name)
        object.__setattr__(self, "risk_capital_quote", capital)
        object.__setattr__(self, "observed_at_ms", observed)
        object.__setattr__(self, "utc_day_index", day)
        object.__setattr__(self, "realized_event_count", event_count)
        object.__setattr__(self, "realized_condition_count", condition_count)
        object.__setattr__(self, "consecutive_losing_conditions", consecutive)
        object.__setattr__(self, "cooldown_until_ms", cooldown)
        object.__setattr__(self, "cooldown_active", bool(self.cooldown_active))
        object.__setattr__(self, "entry_allowed", bool(self.entry_allowed))
        object.__setattr__(self, "entry_block_reasons", reasons)
        actual = _canonical_sha256(self.body())
        claimed = str(self.risk_state_sha256 or "").strip().lower()
        if claimed and claimed != actual:
            raise ValueError("Polymarket live risk state hash differs")
        object.__setattr__(self, "risk_state_sha256", actual)

    def body(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-live-risk-state-v1",
            "condition_id": self.condition_id,
            "risk_profile": self.risk_profile,
            "risk_capital_quote": format(self.risk_capital_quote, "f"),
            "observed_at_ms": self.observed_at_ms,
            "utc_day_index": self.utc_day_index,
            "ledger_revision": {
                "sequence": self.ledger_revision.sequence,
                "record_sha256": self.ledger_revision.record_sha256,
            },
            "realized_event_count": self.realized_event_count,
            "realized_condition_count": self.realized_condition_count,
            "daily_realized_pnl_quote": format(
                self.daily_realized_pnl_quote,
                "f",
            ),
            "lifetime_realized_pnl_quote": format(
                self.lifetime_realized_pnl_quote,
                "f",
            ),
            "settled_equity_quote": format(self.settled_equity_quote, "f"),
            "settled_peak_equity_quote": format(
                self.settled_peak_equity_quote,
                "f",
            ),
            "drawdown_capital_fraction": format(
                self.drawdown_capital_fraction,
                "f",
            ),
            "consecutive_losing_conditions": self.consecutive_losing_conditions,
            "cooldown_until_ms": self.cooldown_until_ms,
            "cooldown_active": self.cooldown_active,
            "current_condition_inventory_downside_quote": format(
                self.current_condition_inventory_downside_quote,
                "f",
            ),
            "other_condition_inventory_downside_quote": format(
                self.other_condition_inventory_downside_quote,
                "f",
            ),
            "total_inventory_downside_quote": format(
                self.total_inventory_downside_quote,
                "f",
            ),
            "maximum_current_condition_downside_quote": format(
                self.maximum_current_condition_downside_quote,
                "f",
            ),
            "entry_allowed": self.entry_allowed,
            "entry_block_reasons": list(self.entry_block_reasons),
            "binance_risk_dependency": False,
        }


@dataclass(frozen=True, slots=True)
class PolymarketLiveRiskCapture:
    state: PolymarketLiveRiskState
    condition_accounting: PolymarketConditionAccounting

    def __post_init__(self) -> None:
        if self.state.condition_id != self.condition_accounting.condition_id:
            raise ValueError("Polymarket live risk capture condition differs")


@dataclass(frozen=True, slots=True)
class _CachedRiskEvidence:
    revision: PolymarketLedgerRevision
    accountings: tuple[PolymarketConditionAccounting, ...]
    events: tuple[PolymarketRealizedPnlEvent, ...]


class PolymarketLiveRiskService:
    """Cache immutable evidence and recompute time-sensitive entry gates."""

    def __init__(
        self,
        ledger: PolymarketLiveOrderLedger,
        *,
        risk_capital_quote: Decimal,
        risk_profile: str = "conservative",
    ) -> None:
        if not isinstance(ledger, PolymarketLiveOrderLedger):
            raise TypeError(
                "Polymarket live risk service requires the ownership ledger"
            )
        capital = _decimal(risk_capital_quote, name="risk capital")
        if capital <= 0:
            raise ValueError("Polymarket risk capital must be positive")
        self.ledger = ledger
        self.risk_capital_quote = capital
        self.profile = polymarket_live_risk_profile(risk_profile)
        self._cache: _CachedRiskEvidence | None = None

    def _load_evidence(self) -> _CachedRiskEvidence:
        revision = self.ledger.revision()
        if self._cache is not None and self._cache.revision == revision:
            return self._cache
        for _attempt in range(3):
            before = self.ledger.revision()
            records = self.ledger.records()
            redemptions = self.ledger.redemption_records()
            conditions = {record.intent.market_id for record in records} | {
                record.condition_id for record in redemptions
            }
            events = self.ledger.realized_pnl_events()
            accountings = tuple(
                self.ledger.condition_accounting(condition_id)
                for condition_id in sorted(conditions)
            )
            after = self.ledger.revision()
            if before == after:
                cached = _CachedRiskEvidence(after, accountings, events)
                self._cache = cached
                return cached
        raise PolymarketLiveBlocked(
            "Polymarket ownership changed while live risk was reconstructed"
        )

    @staticmethod
    def _empty_accounting(condition_id: str) -> PolymarketConditionAccounting:
        return PolymarketConditionAccounting(
            condition_id=condition_id,
            gross_buy_cost_quote=Decimal("0"),
            gross_sell_proceeds_quote=Decimal("0"),
            confirmed_redemption_payout_quote=Decimal("0"),
            up_quantity=Decimal("0"),
            down_quantity=Decimal("0"),
            up_cost_basis_quote=Decimal("0"),
            down_cost_basis_quote=Decimal("0"),
            confirmed_fill_count=0,
        )

    def capture(
        self,
        condition_id: str,
        *,
        observed_at_ms: int,
    ) -> PolymarketLiveRiskCapture:
        condition = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(condition) is None:
            raise ValueError("Polymarket live risk condition is invalid")
        observed = int(observed_at_ms)
        if observed <= 0:
            raise ValueError("Polymarket live risk observation time is invalid")
        evidence = self._load_evidence()
        accounting_by_condition = {
            item.condition_id: item for item in evidence.accountings
        }
        current = accounting_by_condition.get(
            condition,
            self._empty_accounting(condition),
        )
        total_downside = sum(
            (item.inventory_downside_quote for item in evidence.accountings),
            start=Decimal("0"),
        )
        current_downside = current.inventory_downside_quote
        other_downside = total_downside - current_downside

        daily_realized = sum(
            (
                event.pnl_quote
                for event in evidence.events
                if event.observed_at_ms // _DAY_MS == observed // _DAY_MS
            ),
            start=Decimal("0"),
        )
        lifetime_realized = sum(
            (event.pnl_quote for event in evidence.events),
            start=Decimal("0"),
        )
        equity = self.risk_capital_quote
        peak = equity
        for event in evidence.events:
            equity += event.pnl_quote
            peak = max(peak, equity)
        drawdown_quote = max(Decimal("0"), peak - equity)
        drawdown_fraction = drawdown_quote / self.risk_capital_quote

        condition_pnl: dict[str, Decimal] = {}
        condition_time: dict[str, int] = {}
        for event in evidence.events:
            condition_pnl[event.condition_id] = (
                condition_pnl.get(event.condition_id, Decimal("0")) + event.pnl_quote
            )
            condition_time[event.condition_id] = max(
                condition_time.get(event.condition_id, 0),
                event.observed_at_ms,
            )
        consecutive = 0
        cooldown_until = 0
        for realized_condition in sorted(
            condition_pnl,
            key=lambda item: (condition_time[item], item),
        ):
            if condition_pnl[realized_condition] < 0:
                consecutive += 1
                if consecutive >= 2:
                    cooldown_until = (
                        condition_time[realized_condition]
                        + self.profile.loss_cluster_cooldown_minutes * 60_000
                    )
            else:
                consecutive = 0

        event_cap = (
            self.risk_capital_quote * self.profile.maximum_event_loss_capital_fraction
        )
        daily_cap = (
            self.risk_capital_quote * self.profile.maximum_daily_loss_capital_fraction
        )
        drawdown_cap = (
            self.risk_capital_quote * self.profile.maximum_drawdown_capital_fraction
        )
        daily_loss = max(Decimal("0"), -daily_realized)
        maximum_current = min(
            event_cap,
            max(Decimal("0"), daily_cap - daily_loss - other_downside),
            max(Decimal("0"), drawdown_cap - drawdown_quote - other_downside),
        )
        reasons: list[str] = []
        if daily_loss + total_downside >= daily_cap:
            reasons.append("daily_loss_limit_reached")
        if drawdown_quote + total_downside >= drawdown_cap:
            reasons.append("drawdown_limit_reached")
        if observed < cooldown_until:
            reasons.append("loss_cluster_cooldown_active")
        if current_downside >= maximum_current:
            reasons.append("event_or_portfolio_risk_headroom_exhausted")
        state = PolymarketLiveRiskState(
            condition_id=condition,
            risk_profile=self.profile.name,
            risk_capital_quote=self.risk_capital_quote,
            observed_at_ms=observed,
            utc_day_index=observed // _DAY_MS,
            ledger_revision=evidence.revision,
            realized_event_count=len(evidence.events),
            realized_condition_count=len(condition_pnl),
            daily_realized_pnl_quote=daily_realized,
            lifetime_realized_pnl_quote=lifetime_realized,
            settled_equity_quote=equity,
            settled_peak_equity_quote=peak,
            drawdown_capital_fraction=drawdown_fraction,
            consecutive_losing_conditions=consecutive,
            cooldown_until_ms=cooldown_until,
            cooldown_active=observed < cooldown_until,
            current_condition_inventory_downside_quote=current_downside,
            other_condition_inventory_downside_quote=other_downside,
            total_inventory_downside_quote=total_downside,
            maximum_current_condition_downside_quote=maximum_current,
            entry_allowed=not reasons,
            entry_block_reasons=tuple(reasons),
        )
        return PolymarketLiveRiskCapture(state, current)


__all__ = [
    "POLYMARKET_LIVE_RISK_PROFILES",
    "PolymarketLiveRiskCapture",
    "PolymarketLiveRiskProfile",
    "PolymarketLiveRiskService",
    "PolymarketLiveRiskState",
    "polymarket_live_risk_profile",
]
