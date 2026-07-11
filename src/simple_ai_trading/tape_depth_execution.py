"""Causal taker execution diagnostics for tape/depth forecast evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np

from .assets import is_supported_major_symbol, normalize_symbol
from .microstructure_warehouse import MicrostructureWarehouse
from .tape_depth_model import (
    TAPE_DEPTH_MODEL_SCHEMA_VERSION,
    TapeDepthPredictionBatch,
)


TAPE_DEPTH_EXECUTION_SCHEMA_VERSION = "tape-depth-taker-diagnostic-v2"
_SIGNAL_TABLE_PREFIX = "tape_depth_execution_signals"
_CONFIRMATION_DESIGN_SCHEMA = "tape-depth-execution-confirmation-design-v1"


@dataclass(frozen=True)
class TapeDepthExecutionAssumptions:
    taker_fee_bps_per_side: float = 5.0
    additional_slippage_bps_per_side: float = 0.0
    max_quote_age_ms: int = 1_000
    reference_order_notional_quote: float = 1_000.0
    max_l1_participation: float = 0.10
    suppress_overlapping_positions: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.taker_fee_bps_per_side,
            self.additional_slippage_bps_per_side,
            self.reference_order_notional_quote,
            self.max_l1_participation,
        )
        if (
            not all(math.isfinite(float(value)) for value in numeric)
            or self.taker_fee_bps_per_side < 0.0
            or self.additional_slippage_bps_per_side < 0.0
            or self.max_quote_age_ms <= 0
            or self.reference_order_notional_quote <= 0.0
            or not 0.0 < self.max_l1_participation <= 1.0
            or self.suppress_overlapping_positions is not True
        ):
            raise ValueError("tape/depth execution assumptions are invalid")


@dataclass(frozen=True)
class TapeDepthExecutionRow:
    signal_index: int
    side: int
    decision_time_ms: int
    target_entry_time_ms: int
    target_exit_time_ms: int
    trade_reference_gross_bps: float
    status: str
    rejection_reason: str
    entry_bid: float | None
    entry_ask: float | None
    exit_bid: float | None
    exit_ask: float | None
    entry_quote_age_ms: int | None
    exit_quote_age_ms: int | None
    maximum_l1_participation: float | None
    quote_path_gross_bps: float | None
    spread_cost_bps: float | None
    fee_cost_bps: float
    slippage_cost_bps: float
    net_return_bps: float | None


@dataclass(frozen=True)
class TapeDepthExecutionMetrics:
    prediction_rows: int
    selected_signal_rows: int
    overlap_suppressed_rows: int
    scheduled_signal_rows: int
    executable_rows: int
    rejected_quote_rows: int
    rejected_participation_rows: int
    executable_long_rows: int
    executable_short_rows: int
    mean_trade_reference_gross_bps: float
    mean_quote_path_gross_bps: float
    mean_spread_cost_bps: float
    mean_net_return_bps: float
    median_net_return_bps: float
    minimum_net_return_bps: float
    maximum_net_return_bps: float
    positive_net_rate: float
    mean_maximum_l1_participation: float
    maximum_l1_participation: float


@dataclass(frozen=True)
class TapeDepthExecutionReport:
    schema_version: str
    status: str
    rejection_reasons: tuple[str, ...]
    trading_authority: bool
    execution_claim: bool
    profitability_claim: bool
    symbol: str
    prediction_fingerprint: str
    source_evidence: Mapping[str, object]
    assumptions: TapeDepthExecutionAssumptions
    metrics: TapeDepthExecutionMetrics
    created_at: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload


def _finite_positive(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def load_tape_depth_execution_confirmation_design(
    path: str | Path,
    *,
    availability_path: str | Path,
) -> tuple[dict[str, object], str]:
    """Load a precommitted execution design and reject any post-hoc mutation."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tape/depth execution design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("tape/depth execution design must be a JSON object")
    expected_sha256 = str(payload.get("design_sha256") or "").lower()
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    actual_sha256 = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    candidate = payload.get("candidate")
    execution = payload.get("execution")
    acceptance = payload.get("acceptance")
    periods = payload.get("confirmation_periods")
    if (
        payload.get("schema_version") != _CONFIRMATION_DESIGN_SCHEMA
        or expected_sha256 != actual_sha256
        or not isinstance(candidate, dict)
        or not isinstance(execution, dict)
        or not isinstance(acceptance, dict)
        or not isinstance(periods, list)
        or len(periods) != int(acceptance.get("required_completed_periods", -1))
        or len(periods) != len(set(str(value) for value in periods))
        or candidate.get("model_schema_version") != TAPE_DEPTH_MODEL_SCHEMA_VERSION
        or execution.get("schema_version") != TAPE_DEPTH_EXECUTION_SCHEMA_VERSION
        or float(execution.get("leverage", 0.0)) != 1.0
        or execution.get("maker_fill_claim") is not False
        or execution.get("suppress_overlapping_positions") is not True
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("terminal_holdout") is not False
    ):
        raise ValueError("tape/depth execution design failed its immutable contract")
    try:
        for period in periods:
            datetime.strptime(str(period), "%Y-%m-%d")
        TapeDepthExecutionAssumptions(
            taker_fee_bps_per_side=float(execution["taker_fee_bps_per_side"]),
            additional_slippage_bps_per_side=float(
                execution["additional_slippage_bps_per_side"]
            ),
            max_quote_age_ms=int(execution["max_quote_age_ms"]),
            reference_order_notional_quote=float(
                execution["reference_order_notional_quote"]
            ),
            max_l1_participation=float(execution["max_l1_participation"]),
            suppress_overlapping_positions=execution[
                "suppress_overlapping_positions"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tape/depth execution design fields are invalid") from exc
    try:
        availability_sha256 = hashlib.sha256(
            Path(availability_path).read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise ValueError("tape/depth availability evidence is unreadable") from exc
    if availability_sha256 != payload.get("availability_sha256"):
        raise ValueError("tape/depth availability evidence differs from the design")
    return payload, actual_sha256


def evaluate_tape_depth_taker_execution(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    predictions: TapeDepthPredictionBatch,
    assumptions: TapeDepthExecutionAssumptions | None = None,
) -> tuple[TapeDepthExecutionReport, tuple[TapeDepthExecutionRow, ...]]:
    """Replay selected forecasts against causally available 100 ms best quotes."""

    normalized_symbol = normalize_symbol(symbol)
    if not is_supported_major_symbol(normalized_symbol):
        raise ValueError(f"unsupported tape/depth execution symbol: {normalized_symbol}")
    if (
        predictions.rows <= 0
        or np.any(np.diff(predictions.decision_time_ms) <= 0)
        or np.any(predictions.target_entry_time_ms <= predictions.decision_time_ms)
        or np.any(predictions.target_exit_time_ms <= predictions.target_entry_time_ms)
    ):
        raise ValueError("tape/depth execution prediction clock is invalid")
    config = assumptions or TapeDepthExecutionAssumptions()
    sides = predictions.action_sides()
    selected = np.flatnonzero(sides)
    scheduled: list[int] = []
    overlap_suppressed = 0
    previous_exit = -1
    for raw_index in selected:
        index = int(raw_index)
        entry_time = int(predictions.target_entry_time_ms[index])
        if entry_time < previous_exit:
            overlap_suppressed += 1
            continue
        scheduled.append(index)
        previous_exit = int(predictions.target_exit_time_ms[index])

    evidence_start = int(
        predictions.target_entry_time_ms[scheduled[0]]
        if scheduled
        else predictions.target_entry_time_ms[0]
    )
    evidence_end = int(
        predictions.target_exit_time_ms[scheduled[-1]]
        if scheduled
        else predictions.target_exit_time_ms[-1]
    )
    source_evidence = warehouse.require_corpus_certificate(
        normalized_symbol,
        required_data_types=("bookTicker",),
        required_start_ms=evidence_start,
        required_end_ms=evidence_end,
        require_full_history_inventory=True,
    )
    connection = warehouse.connect()
    signal_table = f"{_SIGNAL_TABLE_PREFIX}_{uuid4().hex}"
    connection.execute(
        f"""
        CREATE TEMP TABLE {signal_table} (
            signal_index UINTEGER,
            symbol VARCHAR,
            side TINYINT,
            decision_time_ms BIGINT,
            target_entry_time_ms BIGINT,
            target_exit_time_ms BIGINT
        )
        """
    )
    try:
        if scheduled:
            connection.executemany(
                f"INSERT INTO {signal_table} VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        index,
                        normalized_symbol,
                        int(sides[index]),
                        int(predictions.decision_time_ms[index]),
                        int(predictions.target_entry_time_ms[index]),
                        int(predictions.target_exit_time_ms[index]),
                    )
                    for index in scheduled
                ],
            )
        quote_rows = connection.execute(
            f"""
            SELECT
                s.signal_index, s.side, s.decision_time_ms,
                s.target_entry_time_ms, s.target_exit_time_ms,
                entry.close_bid, entry.close_ask,
                entry.close_bid_qty, entry.close_ask_qty,
                entry.last_transaction_time_ms,
                exit.close_bid, exit.close_ask,
                exit.close_bid_qty, exit.close_ask_qty,
                exit.last_transaction_time_ms
            FROM {signal_table} s
            ASOF LEFT JOIN current_book_ticker_100ms entry
              ON s.symbol = entry.symbol
             AND entry.available_time_ms <= s.target_entry_time_ms
            ASOF LEFT JOIN current_book_ticker_100ms exit
              ON s.symbol = exit.symbol
             AND exit.available_time_ms <= s.target_exit_time_ms
            ORDER BY s.signal_index
            """
        ).fetchall()
    finally:
        connection.execute(f"DROP TABLE IF EXISTS {signal_table}")

    if (
        len(quote_rows) != len(scheduled)
        or [int(row[0]) for row in quote_rows] != scheduled
    ):
        raise RuntimeError("tape/depth quote replay lost or reordered scheduled signals")

    output_rows: list[TapeDepthExecutionRow] = []
    quote_rejections = 0
    participation_rejections = 0
    for raw in quote_rows:
        index, side, decision_ms, entry_ms, exit_ms = (int(value) for value in raw[:5])
        entry_bid, entry_ask, entry_bid_qty, entry_ask_qty = (
            _finite_positive(value) for value in raw[5:9]
        )
        entry_transaction_ms = None if raw[9] is None else int(raw[9])
        exit_bid, exit_ask, exit_bid_qty, exit_ask_qty = (
            _finite_positive(value) for value in raw[10:14]
        )
        exit_transaction_ms = None if raw[14] is None else int(raw[14])
        entry_age = (
            None if entry_transaction_ms is None else entry_ms - entry_transaction_ms
        )
        exit_age = None if exit_transaction_ms is None else exit_ms - exit_transaction_ms
        rejection = ""
        if any(
            value is None
            for value in (
                entry_bid,
                entry_ask,
                entry_bid_qty,
                entry_ask_qty,
                exit_bid,
                exit_ask,
                exit_bid_qty,
                exit_ask_qty,
                entry_age,
                exit_age,
            )
        ):
            rejection = "missing_quote"
        elif entry_bid >= entry_ask or exit_bid >= exit_ask:
            rejection = "invalid_or_crossed_quote"
        elif not 0 <= entry_age <= config.max_quote_age_ms:
            rejection = "stale_entry_quote"
        elif not 0 <= exit_age <= config.max_quote_age_ms:
            rejection = "stale_exit_quote"

        participation: float | None = None
        quote_gross: float | None = None
        spread_cost: float | None = None
        fee_cost = 0.0
        slippage_cost = 0.0
        net_return: float | None = None
        if not rejection:
            assert entry_bid is not None and entry_ask is not None
            assert exit_bid is not None and exit_ask is not None
            assert entry_bid_qty is not None and entry_ask_qty is not None
            assert exit_bid_qty is not None and exit_ask_qty is not None
            if side == 1:
                order_qty = config.reference_order_notional_quote / entry_ask
                participation = max(
                    order_qty / entry_ask_qty,
                    order_qty / exit_bid_qty,
                )
                exit_notional_ratio = exit_bid / entry_ask
                quote_gross = (exit_notional_ratio - 1.0) * 10_000.0
            else:
                order_qty = config.reference_order_notional_quote / entry_bid
                participation = max(
                    order_qty / entry_bid_qty,
                    order_qty / exit_ask_qty,
                )
                exit_notional_ratio = exit_ask / entry_bid
                quote_gross = (1.0 - exit_notional_ratio) * 10_000.0
            entry_mid = (entry_bid + entry_ask) / 2.0
            exit_mid = (exit_bid + exit_ask) / 2.0
            midpoint_gross = (
                (exit_mid / entry_mid - 1.0) * 10_000.0
                if side == 1
                else (1.0 - exit_mid / entry_mid) * 10_000.0
            )
            spread_cost = midpoint_gross - quote_gross
            fee_cost = config.taker_fee_bps_per_side * (1.0 + exit_notional_ratio)
            slippage_cost = config.additional_slippage_bps_per_side * (
                1.0 + exit_notional_ratio
            )
            if participation > config.max_l1_participation:
                rejection = "l1_participation_exceeded"
            else:
                net_return = quote_gross - fee_cost - slippage_cost
        if rejection == "l1_participation_exceeded":
            participation_rejections += 1
        elif rejection:
            quote_rejections += 1
        output_rows.append(
            TapeDepthExecutionRow(
                signal_index=index,
                side=side,
                decision_time_ms=decision_ms,
                target_entry_time_ms=entry_ms,
                target_exit_time_ms=exit_ms,
                trade_reference_gross_bps=float(
                    sides[index] * predictions.actual_gross_return_bps[index]
                ),
                status="executable" if not rejection else "rejected",
                rejection_reason=rejection,
                entry_bid=entry_bid,
                entry_ask=entry_ask,
                exit_bid=exit_bid,
                exit_ask=exit_ask,
                entry_quote_age_ms=entry_age,
                exit_quote_age_ms=exit_age,
                maximum_l1_participation=participation,
                quote_path_gross_bps=quote_gross,
                spread_cost_bps=spread_cost,
                fee_cost_bps=fee_cost,
                slippage_cost_bps=slippage_cost,
                net_return_bps=net_return,
            )
        )

    executable = [row for row in output_rows if row.status == "executable"]
    net = np.asarray([float(row.net_return_bps) for row in executable], dtype=np.float64)
    quote_gross = np.asarray(
        [float(row.quote_path_gross_bps) for row in executable],
        dtype=np.float64,
    )
    reference_gross = np.asarray(
        [row.trade_reference_gross_bps for row in executable],
        dtype=np.float64,
    )
    spreads = np.asarray(
        [float(row.spread_cost_bps) for row in executable],
        dtype=np.float64,
    )
    participation = np.asarray(
        [float(row.maximum_l1_participation) for row in executable],
        dtype=np.float64,
    )

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    metrics = TapeDepthExecutionMetrics(
        prediction_rows=predictions.rows,
        selected_signal_rows=len(selected),
        overlap_suppressed_rows=overlap_suppressed,
        scheduled_signal_rows=len(scheduled),
        executable_rows=len(executable),
        rejected_quote_rows=quote_rejections,
        rejected_participation_rows=participation_rejections,
        executable_long_rows=sum(row.side == 1 for row in executable),
        executable_short_rows=sum(row.side == -1 for row in executable),
        mean_trade_reference_gross_bps=mean(reference_gross),
        mean_quote_path_gross_bps=mean(quote_gross),
        mean_spread_cost_bps=mean(spreads),
        mean_net_return_bps=mean(net),
        median_net_return_bps=float(np.median(net)) if len(net) else 0.0,
        minimum_net_return_bps=float(np.min(net)) if len(net) else 0.0,
        maximum_net_return_bps=float(np.max(net)) if len(net) else 0.0,
        positive_net_rate=float(np.mean(net > 0.0)) if len(net) else 0.0,
        mean_maximum_l1_participation=mean(participation),
        maximum_l1_participation=(
            float(np.max(participation)) if len(participation) else 0.0
        ),
    )
    reasons: list[str] = []
    if metrics.selected_signal_rows <= 0:
        reasons.append("no_selected_forecasts")
    if metrics.executable_rows < 5:
        reasons.append("executable_action_count_below_five")
    if metrics.rejected_quote_rows > 0:
        reasons.append("quote_path_incomplete_or_stale")
    if metrics.rejected_participation_rows > 0:
        reasons.append("l1_participation_rejection_present")
    if metrics.mean_net_return_bps <= 0.0:
        reasons.append("mean_after_cost_return_not_positive")
    if metrics.positive_net_rate <= 0.5:
        reasons.append("after_cost_hit_rate_not_above_half")
    report = TapeDepthExecutionReport(
        schema_version=TAPE_DEPTH_EXECUTION_SCHEMA_VERSION,
        status="after_cost_diagnostic_candidate" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
        trading_authority=False,
        execution_claim=False,
        profitability_claim=False,
        symbol=normalized_symbol,
        prediction_fingerprint=predictions.fingerprint(),
        source_evidence=source_evidence,
        assumptions=config,
        metrics=metrics,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    return report, tuple(output_rows)


__all__ = [
    "TAPE_DEPTH_EXECUTION_SCHEMA_VERSION",
    "TapeDepthExecutionAssumptions",
    "TapeDepthExecutionMetrics",
    "TapeDepthExecutionReport",
    "TapeDepthExecutionRow",
    "evaluate_tape_depth_taker_execution",
    "load_tape_depth_execution_confirmation_design",
]
