"""Positions ledger + realized/unrealized P&L accounting for the autonomous loop.

The existing backtest and live-loop code keep the single-position contract
``max_open_positions=1`` in memory only.  Autonomous mode needs durable state
so a restart (or an operator running ``autonomous status`` from another shell)
can see exactly what is open and what has been realized.

Two JSON files live under ``data/autonomous/``:

* ``open_positions.json`` — a list of currently open positions.
* ``ledger.json`` — a list of every closed trade in chronological order.

Entries are small and human-readable.  No credentials, no raw order IDs beyond
what the exchange already returned, and all numeric fields are plain floats so
the file loads fine with ``python -m json.tool``.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from .storage import write_json_atomic

BOT_OWNER = "simple_ai_trading"
BOT_CLIENT_ORDER_PREFIX = "sait"
_LIVE_FILLED_STATUSES = {"FILLED", "PARTIALLY_FILLED"}
_LIVE_UNSAFE_STATUSES = {
    "",
    "LOCAL",
    "PAPER",
    "PENDING_OPEN",
    "PENDING_CLOSE",
    "PENDING_CANCEL",
    "CANCELED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
    "EXPIRED_IN_MATCH",
}
_LIVE_FUTURES_ACK_STATUSES = {"ACCEPTED", "NEW"}


@dataclass
class OpenPosition:
    """An open position on the exchange as tracked by this process."""

    id: str
    symbol: str
    market_type: str
    side: str  # "LONG" or "SHORT"
    qty: float
    entry_price: float
    leverage: float
    opened_at_ms: int
    notional: float
    strategy_profile: str = ""
    objective: str = ""
    dry_run: bool = True
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    owner: str = BOT_OWNER
    open_client_order_id: str = ""
    open_exchange_order_id: str = ""
    exchange_status: str = "local"

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == "LONG":
            return (mark_price - self.entry_price) * self.qty
        return (self.entry_price - mark_price) * self.qty

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "LONG":
            return (mark_price - self.entry_price) / self.entry_price
        return (self.entry_price - mark_price) / self.entry_price


@dataclass
class ClosedTrade:
    """A completed round-trip recorded in the ledger."""

    id: str
    symbol: str
    market_type: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    leverage: float
    opened_at_ms: int
    closed_at_ms: int
    realized_pnl: float
    realized_pnl_pct: float
    fees: float = 0.0
    reason: str = ""
    strategy_profile: str = ""
    objective: str = ""
    dry_run: bool = True
    owner: str = BOT_OWNER
    open_client_order_id: str = ""
    open_exchange_order_id: str = ""
    close_client_order_id: str = ""
    close_exchange_order_id: str = ""
    exchange_status: str = "local"


_OPEN_REQUIRED_FIELDS = frozenset({
    "id",
    "symbol",
    "market_type",
    "side",
    "qty",
    "entry_price",
    "leverage",
    "opened_at_ms",
    "notional",
})
_CLOSED_REQUIRED_FIELDS = frozenset({
    "id",
    "symbol",
    "market_type",
    "side",
    "qty",
    "entry_price",
    "exit_price",
    "leverage",
    "opened_at_ms",
    "closed_at_ms",
    "realized_pnl",
    "realized_pnl_pct",
})
_OPEN_KNOWN_FIELDS = frozenset(item.name for item in fields(OpenPosition))
_OPEN_REQUIRED_TEXT_FIELDS = frozenset({"id", "symbol", "market_type", "side"})
_OPEN_OPTIONAL_TEXT_FIELDS = frozenset({
    "strategy_profile",
    "objective",
    "owner",
    "open_client_order_id",
    "open_exchange_order_id",
    "exchange_status",
})
_OPEN_REQUIRED_FINITE_FIELDS = frozenset({"qty", "entry_price", "leverage", "notional"})
_OPEN_OPTIONAL_FINITE_FIELDS = frozenset({"stop_loss_pct", "take_profit_pct"})
_OPEN_POSITIVE_FIELDS = frozenset({"qty", "entry_price", "leverage", "notional"})
_OPEN_MARKET_TYPES = frozenset({"spot", "futures"})
_OPEN_SIDES = frozenset({"LONG", "SHORT"})


@dataclass
class LedgerStats:
    """Aggregate statistics across closed trades + open unrealized exposure."""

    closed_trades: int
    wins: int
    losses: int
    realized_pnl: float
    realized_pnl_pct: float
    win_rate: float
    total_fees: float
    largest_win: float
    largest_loss: float
    open_positions: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    starting_reference_cash: float

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearningFeedbackReport:
    """Bounded post-trade feedback for safe retraining and review loops."""

    generated_at_ms: int
    lookback_trades: int
    closed_trades: int
    wins: int
    losses: int
    net_realized_pnl: float
    win_rate: float
    max_consecutive_losses: int
    worst_trade_pnl: float
    recurring_loss_reasons: dict[str, int]
    loss_by_symbol: dict[str, int]
    loss_by_side: dict[str, int]
    recommendations: tuple[str, ...]
    promotion_safe: bool
    notes: tuple[str, ...] = ()

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PositionsStore:
    """Durable storage for open positions + closed trades ledger."""

    root: Path = field(default_factory=lambda: Path("data/autonomous"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def open_path(self) -> Path:
        return self.root / "open_positions.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.json"

    # ---- low-level I/O ------------------------------------------------------

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, list):
            return []
        return [entry for entry in payload if isinstance(entry, dict)]

    def _write(self, path: Path, payload: list[dict[str, Any]]) -> None:
        write_json_atomic(path, payload, indent=2, sort_keys=True)

    # ---- public API ---------------------------------------------------------

    def load_open(self) -> list[OpenPosition]:
        return [OpenPosition(**entry) for entry in self._load(self.open_path)
                if self._valid_open_entry(entry)]

    def load_ledger(self) -> list[ClosedTrade]:
        return [ClosedTrade(**entry) for entry in self._load(self.ledger_path)
                if self._valid_closed_entry(entry)]

    def record_open(self, position: OpenPosition) -> OpenPosition:
        existing = self.load_open()
        existing = [p for p in existing if p.id != position.id]
        existing.append(position)
        self._write(self.open_path, [asdict(p) for p in existing])
        return position

    def record_close(self, trade: ClosedTrade) -> ClosedTrade:
        existing = self.load_ledger()
        existing.append(trade)
        self._write(self.ledger_path, [asdict(t) for t in existing])
        # also drop the matching open entry if present
        opens = [p for p in self.load_open() if p.id != trade.id]
        self._write(self.open_path, [asdict(p) for p in opens])
        write_learning_feedback(self)
        return trade

    def record_close_result(self, position: OpenPosition, trade: ClosedTrade) -> ClosedTrade:
        """Record a close fill while preserving any unfilled open remainder."""

        open_qty = max(0.0, float(position.qty))
        close_qty = max(0.0, float(trade.qty))
        tolerance = max(1e-12, open_qty * 1e-8)
        if open_qty <= 0.0 or close_qty >= open_qty - tolerance:
            return self.record_close(trade)

        existing = self.load_ledger()
        existing.append(trade)
        self._write(self.ledger_path, [asdict(t) for t in existing])

        remaining_qty = max(0.0, open_qty - close_qty)
        remaining = replace(
            position,
            qty=remaining_qty,
            notional=max(0.0, remaining_qty * float(position.entry_price)),
            exchange_status="PARTIALLY_FILLED",
        )
        opens = [p for p in self.load_open() if p.id != position.id]
        opens.append(remaining)
        self._write(self.open_path, [asdict(p) for p in opens])
        write_learning_feedback(self)
        return trade

    def remove_open(self, position_id: str) -> bool:
        opens = self.load_open()
        filtered = [p for p in opens if p.id != position_id]
        if len(filtered) == len(opens):
            return False
        self._write(self.open_path, [asdict(p) for p in filtered])
        return True

    def find_open(self, position_id: str) -> OpenPosition | None:
        for position in self.load_open():
            if position.id == position_id:
                return position
        return None

    def open_integrity_errors(self) -> tuple[str, ...]:
        """Return lossless validation errors for the durable open-position ledger."""

        return self._open_file_integrity_errors(self.open_path)

    # ---- validation helpers -------------------------------------------------

    @staticmethod
    def _valid_open_entry(entry: dict[str, Any]) -> bool:
        return _OPEN_REQUIRED_FIELDS.issubset(entry.keys())

    @staticmethod
    def _valid_closed_entry(entry: dict[str, Any]) -> bool:
        return _CLOSED_REQUIRED_FIELDS.issubset(entry.keys())

    @classmethod
    def _open_file_integrity_errors(cls, path: Path) -> tuple[str, ...]:
        if not path.exists():
            return ()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return (f"open_positions_json_invalid:line={exc.lineno}:column={exc.colno}",)
        except OSError as exc:
            return (f"open_positions_unreadable:{exc.__class__.__name__}",)
        if not isinstance(payload, list):
            return ("open_positions_payload_not_list",)

        errors: list[str] = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                errors.append(f"open_positions_entry_{index}_not_object")
                continue
            errors.extend(cls._open_entry_integrity_errors(index, entry))
        return tuple(errors)

    @staticmethod
    def _open_entry_integrity_errors(index: int, entry: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        keys = set(entry)
        missing = sorted(_OPEN_REQUIRED_FIELDS - keys)
        if missing:
            errors.append(f"open_positions_entry_{index}_missing_fields={','.join(missing)}")
        unknown = sorted(keys - _OPEN_KNOWN_FIELDS)
        if unknown:
            errors.append(f"open_positions_entry_{index}_unknown_fields={','.join(unknown)}")

        for name in sorted(_OPEN_REQUIRED_TEXT_FIELDS):
            if name not in entry:
                continue
            value = entry.get(name)
            if not isinstance(value, str):
                errors.append(f"open_positions_entry_{index}_non_string_field={name}")
                continue
            if not value.strip():
                errors.append(f"open_positions_entry_{index}_empty_text_field={name}")
        for name in sorted(_OPEN_OPTIONAL_TEXT_FIELDS):
            if name in entry and not isinstance(entry.get(name), str):
                errors.append(f"open_positions_entry_{index}_non_string_field={name}")

        market_type = str(entry.get("market_type") or "").lower()
        if "market_type" in entry and market_type not in _OPEN_MARKET_TYPES:
            errors.append(f"open_positions_entry_{index}_invalid_market_type={market_type}")
        side = str(entry.get("side") or "").upper()
        if "side" in entry and side not in _OPEN_SIDES:
            errors.append(f"open_positions_entry_{index}_invalid_side={side}")

        for name in sorted(_OPEN_REQUIRED_FINITE_FIELDS | _OPEN_OPTIONAL_FINITE_FIELDS):
            if name not in entry:
                continue
            value = entry.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                errors.append(f"open_positions_entry_{index}_non_finite_number={name}")
                continue
            if name in _OPEN_POSITIVE_FIELDS and float(value) <= 0.0:
                errors.append(f"open_positions_entry_{index}_non_positive_number={name}")

        opened_at_ms = entry.get("opened_at_ms")
        if (
            "opened_at_ms" in entry
            and (
                isinstance(opened_at_ms, bool)
                or not isinstance(opened_at_ms, int)
                or opened_at_ms < 0
            )
        ):
            errors.append(f"open_positions_entry_{index}_invalid_opened_at_ms")
        if "dry_run" in entry and not isinstance(entry.get("dry_run"), bool):
            errors.append(f"open_positions_entry_{index}_non_boolean_field=dry_run")
        if not errors:
            try:
                OpenPosition(**entry)
            except TypeError as exc:
                errors.append(f"open_positions_entry_{index}_constructor_failed:{exc.__class__.__name__}")
        return errors

    @property
    def learning_feedback_path(self) -> Path:
        return self.root / "learning_feedback.json"


def new_position_id() -> str:
    """Generate a short, collision-resistant identifier for a position."""

    return uuid.uuid4().hex[:12]


def bot_client_order_id(position_id: str, action: str) -> str:
    """Return a Binance-safe bot-owned client order id.

    Binance accepts a client order id on spot and USD-M futures orders.  The
    short deterministic id lets recovery code query an ambiguous order after a
    network failure without ever using a broad account-wide close.
    """

    clean_id = "".join(ch for ch in str(position_id) if ch.isalnum())[:18] or new_position_id()
    clean_action = "o" if str(action).lower().startswith("open") else "c"
    return f"{BOT_CLIENT_ORDER_PREFIX}-{clean_action}-{clean_id}"[:36]


def _normalized_exchange_status(status: object) -> str:
    return str(status or "").strip().upper().replace("-", "_").replace(" ", "_")


def bot_ownership_rejection_reason(position: OpenPosition) -> str | None:
    """Return why a position lacks enough proof for bot-managed live closing."""

    if position.dry_run:
        return None
    if str(position.owner or "") != BOT_OWNER:
        return "owner-unverified"
    client_id = str(position.open_client_order_id or "")
    if not client_id.startswith(f"{BOT_CLIENT_ORDER_PREFIX}-o-"):
        return "open-client-order-unverified"

    status = _normalized_exchange_status(position.exchange_status)
    if status in _LIVE_FILLED_STATUSES:
        return None
    if status in _LIVE_UNSAFE_STATUSES:
        return "exchange-fill-unverified"

    exchange_order_id = str(position.open_exchange_order_id or "").strip()
    market_type = str(position.market_type or "").strip().lower()
    if market_type == "futures" and exchange_order_id and status in _LIVE_FUTURES_ACK_STATUSES:
        return None
    return f"exchange-status-unverified:{status.lower() or 'missing'}"


def is_bot_owned_position(position: OpenPosition) -> bool:
    """Return whether a live position has enough proof to be exchange-closed."""

    return bot_ownership_rejection_reason(position) is None


def now_ms(clock=time.time) -> int:
    """Current wall clock in milliseconds (injected for deterministic tests)."""

    return int(clock() * 1000)


def unrealized_pnl_usd(position: OpenPosition, mark_price: float) -> float:
    """Return the USD-denominated unrealized P&L for ``position`` at ``mark_price``."""

    return position.unrealized_pnl(mark_price)


def unrealized_pnl_pct(position: OpenPosition, mark_price: float) -> float:
    """Return the unrealized P&L as a fraction of entry notional."""

    return position.unrealized_pnl_pct(mark_price)


def compute_stats(
    store: PositionsStore,
    *,
    mark_price: float | None,
    starting_reference_cash: float = 1000.0,
) -> LedgerStats:
    """Compute aggregate statistics from the ledger + open positions.

    ``mark_price`` is the current symbol mark used to value open positions;
    pass ``None`` if the mark is unavailable — unrealized fields will be zero
    rather than raising.
    """

    closed = store.load_ledger()
    opens = store.load_open()
    wins = sum(1 for t in closed if t.realized_pnl > 0)
    losses = sum(1 for t in closed if t.realized_pnl < 0)
    realized = sum(t.realized_pnl for t in closed)
    total_fees = sum(t.fees for t in closed)
    largest_win = max((t.realized_pnl for t in closed if t.realized_pnl > 0), default=0.0)
    largest_loss = min((t.realized_pnl for t in closed if t.realized_pnl < 0), default=0.0)
    unrealized = 0.0
    unrealized_pct = 0.0
    if mark_price is not None and opens:
        unrealized = sum(p.unrealized_pnl(float(mark_price)) for p in opens)
        entry_notional = sum(abs(p.entry_price * p.qty) for p in opens)
        if entry_notional > 0:
            unrealized_pct = unrealized / entry_notional
    realized_pct = (realized / starting_reference_cash) if starting_reference_cash > 0 else 0.0
    return LedgerStats(
        closed_trades=len(closed),
        wins=wins,
        losses=losses,
        realized_pnl=realized,
        realized_pnl_pct=realized_pct,
        win_rate=(wins / len(closed)) if closed else 0.0,
        total_fees=total_fees,
        largest_win=largest_win,
        largest_loss=largest_loss,
        open_positions=len(opens),
        unrealized_pnl=unrealized,
        unrealized_pnl_pct=unrealized_pct,
        starting_reference_cash=starting_reference_cash,
    )


def _max_consecutive_losses(trades: list[ClosedTrade]) -> int:
    longest = 0
    current = 0
    for trade in trades:
        if trade.realized_pnl < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _top_counter(counter: Counter[str], limit: int = 5) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit) if key}


def build_learning_feedback(
    trades: list[ClosedTrade],
    *,
    lookback_trades: int = 100,
    generated_at_ms: int | None = None,
) -> LearningFeedbackReport:
    """Build a bounded mistake-learning artifact from closed trades only."""

    lookback = max(1, int(lookback_trades))
    recent = sorted(trades, key=lambda trade: int(trade.closed_at_ms))[-lookback:]
    generated = int(generated_at_ms if generated_at_ms is not None else now_ms())
    wins = sum(1 for trade in recent if trade.realized_pnl > 0.0)
    losses = sum(1 for trade in recent if trade.realized_pnl < 0.0)
    net = sum(float(trade.realized_pnl) for trade in recent)
    loss_reasons: Counter[str] = Counter()
    loss_symbols: Counter[str] = Counter()
    loss_sides: Counter[str] = Counter()
    pnl_by_reason: defaultdict[str, float] = defaultdict(float)
    for trade in recent:
        reason = str(trade.reason or "unspecified")
        pnl_by_reason[reason] += float(trade.realized_pnl)
        if trade.realized_pnl < 0.0:
            loss_reasons[reason] += 1
            loss_symbols[str(trade.symbol or "unknown")] += 1
            loss_sides[str(trade.side or "unknown")] += 1

    max_loss_streak = _max_consecutive_losses(recent)
    worst = min((float(trade.realized_pnl) for trade in recent), default=0.0)
    recurring = Counter({key: value for key, value in loss_reasons.items() if value >= 2})
    recommendations: list[str] = []
    notes: list[str] = []
    if not recent:
        recommendations.append("collect_more_closed_trade_outcomes_before_self_improvement")
        notes.append("no_closed_trades")
    if max_loss_streak >= 2:
        recommendations.append("trigger_cooldown_and_replay_recent_loss_streak_before_new_promotion")
    if net <= 0.0 and recent:
        recommendations.append("require_retraining_or_model_lab_replay_before_promoting_this_profile")
    if recurring:
        recommendations.append("increase_penalty_for_recurring_exit_reason_or_market_mode")
    if loss_symbols:
        symbol, count = loss_symbols.most_common(1)[0]
        if count >= 2:
            recommendations.append(f"review_symbol_specific_edge:{symbol}")
    if loss_sides:
        side, count = loss_sides.most_common(1)[0]
        if count >= 2:
            recommendations.append(f"review_side_specific_edge:{side}")
    if not recommendations:
        recommendations.append("continue_monitoring_no_retraining_change_required")
    promotion_safe = bool(recent) and net > 0.0 and max_loss_streak < 2 and not recurring
    return LearningFeedbackReport(
        generated_at_ms=generated,
        lookback_trades=lookback,
        closed_trades=len(recent),
        wins=wins,
        losses=losses,
        net_realized_pnl=float(net),
        win_rate=(wins / len(recent)) if recent else 0.0,
        max_consecutive_losses=max_loss_streak,
        worst_trade_pnl=float(worst),
        recurring_loss_reasons=_top_counter(recurring),
        loss_by_symbol=_top_counter(loss_symbols),
        loss_by_side=_top_counter(loss_sides),
        recommendations=tuple(dict.fromkeys(recommendations)),
        promotion_safe=promotion_safe,
        notes=tuple(notes),
    )


def write_learning_feedback(
    store: PositionsStore,
    *,
    lookback_trades: int = 100,
    generated_at_ms: int | None = None,
) -> LearningFeedbackReport:
    """Persist the latest bounded learning feedback beside the ledger."""

    report = build_learning_feedback(
        store.load_ledger(),
        lookback_trades=lookback_trades,
        generated_at_ms=generated_at_ms,
    )
    write_json_atomic(store.learning_feedback_path, report.asdict(), indent=2, sort_keys=True)
    return report


def _int_counter_payload(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _string_tuple_payload(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item) for item in raw)


def learning_feedback_from_mapping(payload: Mapping[str, Any]) -> LearningFeedbackReport:
    """Parse a persisted learning-feedback artifact with bounded defaults."""

    return LearningFeedbackReport(
        generated_at_ms=int(payload.get("generated_at_ms") or 0),
        lookback_trades=int(payload.get("lookback_trades") or 100),
        closed_trades=int(payload.get("closed_trades") or 0),
        wins=int(payload.get("wins") or 0),
        losses=int(payload.get("losses") or 0),
        net_realized_pnl=float(payload.get("net_realized_pnl") or 0.0),
        win_rate=float(payload.get("win_rate") or 0.0),
        max_consecutive_losses=int(payload.get("max_consecutive_losses") or 0),
        worst_trade_pnl=float(payload.get("worst_trade_pnl") or 0.0),
        recurring_loss_reasons=_int_counter_payload(payload.get("recurring_loss_reasons")),
        loss_by_symbol=_int_counter_payload(payload.get("loss_by_symbol")),
        loss_by_side=_int_counter_payload(payload.get("loss_by_side")),
        recommendations=_string_tuple_payload(payload.get("recommendations")),
        promotion_safe=bool(payload.get("promotion_safe")),
        notes=_string_tuple_payload(payload.get("notes")),
    )


def load_learning_feedback_file(path: Path) -> LearningFeedbackReport | None:
    """Load a learning-feedback file without mutating ledgers or models."""

    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return learning_feedback_from_mapping(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def load_learning_feedback(store: PositionsStore) -> LearningFeedbackReport:
    """Load persisted feedback or rebuild it from the ledger."""

    loaded = load_learning_feedback_file(store.learning_feedback_path)
    if loaded is not None:
        return loaded
    return write_learning_feedback(store)


def render_learning_feedback(report: LearningFeedbackReport) -> list[str]:
    """Render post-trade learning feedback for operators and reports."""

    lines = [
        "Learning feedback",
        (
            f"closed={report.closed_trades} wins={report.wins} losses={report.losses} "
            f"win_rate={report.win_rate:.0%} net_pnl={report.net_realized_pnl:+.2f}"
        ),
        (
            f"max_loss_streak={report.max_consecutive_losses} "
            f"worst_trade={report.worst_trade_pnl:+.2f} promotion_safe={report.promotion_safe}"
        ),
    ]
    if report.recurring_loss_reasons:
        reasons = ", ".join(f"{key}:{value}" for key, value in report.recurring_loss_reasons.items())
        lines.append(f"recurring_loss_reasons={reasons}")
    if report.loss_by_symbol:
        symbols = ", ".join(f"{key}:{value}" for key, value in report.loss_by_symbol.items())
        lines.append(f"loss_by_symbol={symbols}")
    for item in report.recommendations:
        lines.append(f"- {item}")
    return lines


def render_positions_table(
    positions: list[OpenPosition],
    *,
    mark_price: float | None,
) -> list[str]:
    """Return a 2D-aligned textual table of open positions.

    Returns an empty list if there is nothing to show so callers can branch on
    truthiness without peeking at the length.
    """

    if not positions:
        return []
    header = (
        f"{'#':>2} {'id':<12} {'own':<9} {'side':<5} {'qty':>10} "
        f"{'entry':>12} {'mark':>12} {'pnl$':>12} {'pnl%':>7}"
    )
    rows = [header]
    for idx, pos in enumerate(positions, start=1):
        mark_value = mark_price if mark_price is not None else pos.entry_price
        pnl_usd = pos.unrealized_pnl(mark_value) if mark_price is not None else 0.0
        pnl_pct = pos.unrealized_pnl_pct(mark_value) if mark_price is not None else 0.0
        ownership = "paper" if pos.dry_run else ("verified" if is_bot_owned_position(pos) else "unknown")
        rows.append(
            f"{idx:>2} {pos.id:<12} {ownership:<9} {pos.side:<5} "
            f"{pos.qty:>10.6f} {pos.entry_price:>12.2f} "
            f"{mark_value:>12.2f} {pnl_usd:>+12.2f} {pnl_pct:>+7.2%}"
        )
    return rows


def render_stats_lines(stats: LedgerStats) -> list[str]:
    """Human-friendly rendering of ``LedgerStats`` suitable for the shell."""

    lines = [
        f"Closed trades  : {stats.closed_trades}  (wins {stats.wins}, losses {stats.losses})",
        f"Realized P&L   : {stats.realized_pnl:+.2f} USDC  ({stats.realized_pnl_pct:+.2%})",
        f"Win rate       : {stats.win_rate:.0%}",
        f"Total fees     : {stats.total_fees:.2f} USDC",
        f"Largest win    : {stats.largest_win:+.2f} USDC",
        f"Largest loss   : {stats.largest_loss:+.2f} USDC",
        f"Open positions : {stats.open_positions}",
        f"Unrealized P&L : {stats.unrealized_pnl:+.2f} USDC  ({stats.unrealized_pnl_pct:+.2%})",
    ]
    return lines
