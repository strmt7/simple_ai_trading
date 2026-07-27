"""Empirical execution evidence for Round 74 latency and residual slippage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Mapping, Sequence

from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_ENVIRONMENTS,
    ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS,
    ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE,
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_latency_evidence_claims,
    round74_slippage_evidence_claims,
)


ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION = (
    "round-074-execution-calibration-source-v1"
)
ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL = 300
ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE = 100
ROUND74_EXECUTION_CALIBRATION_QUANTILE = 0.99
ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE = 0.95
ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION = Decimal("0.05")
ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX = "sat-r74-cal-"
ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS = 250_000_000
_RELATIVE_RECONCILIATION_TOLERANCE = Decimal("0.00000001")
_SENSITIVE_KEYS = frozenset(
    {"apikey", "secret", "secretkey", "signature", "xmbxapikey"}
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Round 74 execution evidence is not canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    selected = str(value)
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 74 execution {label} digest is invalid")
    return selected


def _reject_sensitive_keys(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = (
                str(key).strip().lower().replace("-", "").replace("_", "")
            )
            if normalized in _SENSITIVE_KEYS:
                raise ValueError(
                    f"Round 74 execution {path} contains credential material"
                )
            _reject_sensitive_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_sensitive_keys(nested, f"{path}[{index}]")


def _normalized_payload(value: object) -> object:
    _reject_sensitive_keys(value)
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _strict_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Round 74 execution {label} is invalid")
    return value


def _strict_decimal(
    value: object,
    label: str,
    *,
    minimum: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Round 74 execution {label} must be a decimal string"
        )
    try:
        selected = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Round 74 execution {label} is invalid") from exc
    if (
        not selected.is_finite()
        or (minimum is not None and selected < minimum)
    ):
        raise ValueError(f"Round 74 execution {label} is invalid")
    return selected


def _close_decimal(actual: Decimal, expected: Decimal) -> bool:
    scale = max(abs(actual), abs(expected), Decimal("1"))
    return abs(actual - expected) <= scale * _RELATIVE_RECONCILIATION_TOLERANCE


def _upper_confidence_quantile(
    values: Sequence[int | float],
    *,
    quantile: float,
    confidence: float,
) -> int | float:
    if not values:
        raise ValueError("Round 74 execution quantile sample is empty")
    ordered = sorted(values)
    sample_count = len(ordered)
    cumulative = 0.0
    for below_count in range(sample_count):
        cumulative += (
            math.comb(sample_count, below_count)
            * quantile**below_count
            * (1.0 - quantile) ** (sample_count - below_count)
        )
        if cumulative >= confidence:
            return ordered[below_count]
    raise ValueError(
        "Round 74 execution sample cannot bound the requested tail confidence"
    )


@dataclass(frozen=True)
class _ValidatedExecution:
    calibration_run_id: str
    round_trip_id: str
    path: str
    symbol: str
    side: str
    order_id: int
    client_order_id: str
    reduce_only: bool
    submission_monotonic_ns: int
    terminal_receipt_monotonic_ns: int
    latency_ns: int
    expected_book_walk_price: Decimal
    actual_vwap: Decimal
    executed_quantity: Decimal
    executed_quote_notional: Decimal
    residual_slippage_bps: float
    position_amount: Decimal
    normalized_source: object


@dataclass(frozen=True)
class Round74ExecutionEvidenceBundle:
    """Symbol-specific empirical tail inputs and their source records."""

    reference_quote_notional: float
    decision_to_entry_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    decision_to_exit_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    additional_slippage_bps_per_side_by_symbol: tuple[
        tuple[str, float],
        ...,
    ]
    entry_exit_latency_evidence: Round74EventTargetEvidence
    residual_slippage_evidence: Round74EventTargetEvidence

    def entry_latency_mapping(self) -> dict[str, int]:
        return dict(self.decision_to_entry_latency_ns_by_symbol)

    def exit_latency_mapping(self) -> dict[str, int]:
        return dict(self.decision_to_exit_latency_ns_by_symbol)

    def slippage_mapping(self) -> dict[str, float]:
        return dict(self.additional_slippage_bps_per_side_by_symbol)


def _validated_position_payload(
    value: object,
    *,
    symbol: str,
    label: str,
) -> Decimal:
    normalized = _normalized_payload(value)
    if (
        not isinstance(normalized, Mapping)
        or str(normalized.get("symbol", "")).strip().upper() != symbol
        or normalized.get("positionSide") != "BOTH"
    ):
        raise ValueError(f"Round 74 execution {label} position differs")
    amount = _strict_decimal(
        normalized.get("positionAmt"),
        f"{label} position amount",
    )
    if amount != Decimal("0"):
        raise ValueError(f"Round 74 execution {label} position is not flat")
    return amount


def _book_walk_average_price(
    value: object,
    *,
    symbol: str,
    side: str,
    quantity: Decimal,
    submission_monotonic_ns: int,
) -> Decimal:
    normalized = _normalized_payload(value)
    if (
        not isinstance(normalized, Mapping)
        or normalized.get("schema_version")
        != "round-074-execution-book-state-v1"
        or str(normalized.get("symbol", "")).strip().upper() != symbol
    ):
        raise ValueError("Round 74 execution book state differs")
    _sha256_digest(
        normalized.get("source_payload_sha256"),
        "book state source",
    )
    _strict_integer(normalized.get("update_id"), "book update ID", minimum=1)
    receipt = _strict_integer(
        normalized.get("received_monotonic_ns"),
        "book receipt monotonic time",
        minimum=1,
    )
    age = submission_monotonic_ns - receipt
    if not 0 <= age <= ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS:
        raise ValueError("Round 74 execution book state is stale or future")
    parsed_sides: dict[str, tuple[tuple[Decimal, Decimal], ...]] = {}
    for name, descending in (("bids", True), ("asks", False)):
        raw_levels = normalized.get(name)
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("Round 74 execution book levels differ")
        levels: list[tuple[Decimal, Decimal]] = []
        for raw_level in raw_levels:
            if not isinstance(raw_level, list) or len(raw_level) != 2:
                raise ValueError("Round 74 execution book level differs")
            price = _strict_decimal(
                raw_level[0],
                "book price",
                minimum=Decimal("0.000000000000000001"),
            )
            level_quantity = _strict_decimal(
                raw_level[1],
                "book quantity",
                minimum=Decimal("0.000000000000000001"),
            )
            levels.append((price, level_quantity))
        prices = [price for price, _level_quantity in levels]
        if (
            len(prices) != len(set(prices))
            or prices != sorted(prices, reverse=descending)
        ):
            raise ValueError("Round 74 execution book ordering differs")
        parsed_sides[name] = tuple(levels)
    if parsed_sides["bids"][0][0] >= parsed_sides["asks"][0][0]:
        raise ValueError("Round 74 execution book is crossed")
    levels = parsed_sides["asks" if side == "BUY" else "bids"]
    remaining = quantity
    quote = Decimal("0")
    for price, available in levels:
        filled = min(remaining, available)
        quote += price * filled
        remaining -= filled
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError("Round 74 execution book capacity differs")
    return quote / quantity


def _validated_execution(
    value: Mapping[str, object],
    *,
    reference_quote_notional: Decimal,
) -> _ValidatedExecution:
    normalized = _normalized_payload(value)
    if not isinstance(normalized, Mapping):
        raise ValueError("Round 74 execution record differs")
    if normalized.get("schema_version") != ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Round 74 execution record schema differs")
    calibration_run_id = str(
        normalized.get("calibration_run_id", "")
    ).strip()
    round_trip_id = str(normalized.get("round_trip_id", "")).strip()
    path = str(normalized.get("path", "")).strip()
    symbol = str(normalized.get("symbol", "")).strip().upper()
    side = str(normalized.get("side", "")).strip().upper()
    client_order_id = str(normalized.get("client_order_id", "")).strip()
    if (
        not calibration_run_id
        or not round_trip_id
        or path not in {"entry", "exit"}
        or symbol not in ROUND74_EVENT_TARGET_SYMBOLS
        or side not in {"BUY", "SELL"}
        or not client_order_id.startswith(
            ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX
        )
        or len(client_order_id) > 36
        or normalized.get("terminal_source") != "ORDER_TRADE_UPDATE"
    ):
        raise ValueError("Round 74 execution identity differs")
    submission = _strict_integer(
        normalized.get("submission_monotonic_ns"),
        "submission monotonic time",
        minimum=1,
    )
    terminal_receipt = _strict_integer(
        normalized.get("terminal_receipt_monotonic_ns"),
        "terminal receipt monotonic time",
        minimum=1,
    )
    if terminal_receipt <= submission:
        raise ValueError("Round 74 execution latency interval differs")
    latency = terminal_receipt - submission
    terminal = normalized.get("terminal_order_payload")
    trades = normalized.get("account_trade_payloads")
    if not isinstance(terminal, Mapping) or not isinstance(trades, list) or not trades:
        raise ValueError("Round 74 execution source payload differs")
    if (
        terminal.get("e") != "ORDER_TRADE_UPDATE"
        or not isinstance(terminal.get("o"), Mapping)
    ):
        raise ValueError("Round 74 execution terminal event differs")
    _strict_integer(terminal.get("E"), "terminal event time", minimum=1)
    _strict_integer(terminal.get("T"), "terminal transaction time", minimum=1)
    order = terminal["o"]
    assert isinstance(order, Mapping)
    order_id = _strict_integer(order.get("i"), "order ID", minimum=1)
    reduce_only = order.get("R")
    if (
        str(order.get("s", "")).strip().upper() != symbol
        or str(order.get("c", "")).strip() != client_order_id
        or str(order.get("S", "")).strip().upper() != side
        or order.get("ps") != "BOTH"
        or order.get("o") != "MARKET"
        or order.get("x") != "TRADE"
        or order.get("X") != "FILLED"
        or order.get("m") is not False
        or not isinstance(reduce_only, bool)
        or reduce_only != (path == "exit")
    ):
        raise ValueError("Round 74 execution terminal order differs")
    terminal_quantity = _strict_decimal(
        order.get("z"),
        "terminal executed quantity",
        minimum=Decimal("0.000000000000000001"),
    )
    original_quantity = _strict_decimal(
        order.get("q"),
        "terminal original quantity",
        minimum=Decimal("0.000000000000000001"),
    )
    terminal_average = _strict_decimal(
        order.get("ap"),
        "terminal average price",
        minimum=Decimal("0.000000000000000001"),
    )
    if not _close_decimal(original_quantity, terminal_quantity):
        raise ValueError("Round 74 execution terminal quantity differs")
    expected_price = _book_walk_average_price(
        normalized.get("expected_book_walk_source"),
        symbol=symbol,
        side=side,
        quantity=terminal_quantity,
        submission_monotonic_ns=submission,
    )

    quantity = Decimal("0")
    quote = Decimal("0")
    trade_ids: set[int] = set()
    for trade in trades:
        if not isinstance(trade, Mapping):
            raise ValueError("Round 74 execution account trade differs")
        trade_id = _strict_integer(trade.get("id"), "trade ID", minimum=1)
        trade_order_id = _strict_integer(
            trade.get("orderId"),
            "trade order ID",
            minimum=1,
        )
        price = _strict_decimal(
            trade.get("price"),
            "trade price",
            minimum=Decimal("0.000000000000000001"),
        )
        trade_quantity = _strict_decimal(
            trade.get("qty"),
            "trade quantity",
            minimum=Decimal("0.000000000000000001"),
        )
        trade_quote = _strict_decimal(
            trade.get("quoteQty"),
            "trade quote quantity",
            minimum=Decimal("0.000000000000000001"),
        )
        if (
            trade_id in trade_ids
            or trade_order_id != order_id
            or str(trade.get("symbol", "")).strip().upper() != symbol
            or str(trade.get("side", "")).strip().upper() != side
            or trade.get("buyer") is not (side == "BUY")
            or trade.get("maker") is not False
            or not _close_decimal(trade_quote, price * trade_quantity)
        ):
            raise ValueError("Round 74 execution account trade reconciliation differs")
        trade_ids.add(trade_id)
        quantity += trade_quantity
        quote += trade_quote
    actual_vwap = quote / quantity
    if (
        not _close_decimal(quantity, terminal_quantity)
        or not _close_decimal(actual_vwap, terminal_average)
        or abs(quote - reference_quote_notional)
        > (
            reference_quote_notional
            * ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION
        )
    ):
        raise ValueError("Round 74 execution fill reconciliation differs")
    position_payload_key = (
        "pre_pair_position_payload"
        if path == "entry"
        else "post_pair_position_payload"
    )
    position_amount = _validated_position_payload(
        normalized.get(position_payload_key),
        symbol=symbol,
        label="pre-pair" if path == "entry" else "post-pair",
    )
    forbidden_position_key = (
        "post_pair_position_payload"
        if path == "entry"
        else "pre_pair_position_payload"
    )
    if forbidden_position_key in normalized:
        raise ValueError("Round 74 execution position evidence placement differs")
    side_sign = Decimal("1") if side == "BUY" else Decimal("-1")
    residual = (
        side_sign
        * (actual_vwap / expected_price - Decimal("1"))
        * Decimal("10000")
    )
    residual_float = float(residual)
    if (
        not math.isfinite(residual_float)
        or abs(residual_float)
        > ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE
    ):
        raise ValueError("Round 74 execution residual slippage differs")
    return _ValidatedExecution(
        calibration_run_id=calibration_run_id,
        round_trip_id=round_trip_id,
        path=path,
        symbol=symbol,
        side=side,
        order_id=order_id,
        client_order_id=client_order_id,
        reduce_only=reduce_only,
        submission_monotonic_ns=submission,
        terminal_receipt_monotonic_ns=terminal_receipt,
        latency_ns=latency,
        expected_book_walk_price=expected_price,
        actual_vwap=actual_vwap,
        executed_quantity=quantity,
        executed_quote_notional=quote,
        residual_slippage_bps=residual_float,
        position_amount=position_amount,
        normalized_source=normalized,
    )


def validate_round74_execution_calibration_record(
    *,
    record: Mapping[str, object],
    reference_quote_notional: float,
) -> dict[str, object]:
    """Validate one source record without weakening full-panel requirements."""

    reference = Decimal(str(reference_quote_notional))
    if not reference.is_finite() or reference <= 0:
        raise ValueError("Round 74 execution reference notional differs")
    selected = _validated_execution(
        record,
        reference_quote_notional=reference,
    )
    normalized = selected.normalized_source
    if not isinstance(normalized, dict):
        raise ValueError("Round 74 execution normalized record differs")
    return dict(normalized)


def build_round74_execution_calibration_evidence(
    *,
    records: Sequence[Mapping[str, object]],
    environment: str,
    observed_wall_ns: int,
    reference_quote_notional: float,
) -> Round74ExecutionEvidenceBundle:
    """Derive empirical p99 latency and residual shortfall from flat pairs."""

    selected_environment = str(environment).strip()
    if selected_environment not in ROUND74_EVENT_TARGET_ENVIRONMENTS:
        raise ValueError("Round 74 execution environment differs")
    observed = _strict_integer(
        observed_wall_ns,
        "observation wall time",
        minimum=1,
    )
    reference = Decimal(str(reference_quote_notional))
    if not reference.is_finite() or reference <= 0:
        raise ValueError("Round 74 execution reference notional differs")
    selected = tuple(
        _validated_execution(
            record,
            reference_quote_notional=reference,
        )
        for record in records
    )
    if not selected or len({item.calibration_run_id for item in selected}) != 1:
        raise ValueError("Round 74 execution calibration run differs")
    if len({item.order_id for item in selected}) != len(selected) or len(
        {item.client_order_id for item in selected}
    ) != len(selected):
        raise ValueError("Round 74 execution order identity is duplicated")

    pairs: dict[tuple[str, str], list[_ValidatedExecution]] = defaultdict(list)
    for item in selected:
        pairs[(item.symbol, item.round_trip_id)].append(item)
    pair_counts = {symbol: 0 for symbol in ROUND74_EVENT_TARGET_SYMBOLS}
    side_pair_counts = {
        (symbol, side): 0
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        for side in ("BUY", "SELL")
    }
    for (symbol, _round_trip_id), pair in pairs.items():
        ordered = sorted(pair, key=lambda item: item.path)
        by_path = {item.path: item for item in ordered}
        if (
            len(pair) != 2
            or set(by_path) != {"entry", "exit"}
            or by_path["entry"].side == by_path["exit"].side
            or not _close_decimal(
                by_path["entry"].executed_quantity,
                by_path["exit"].executed_quantity,
            )
            or by_path["exit"].submission_monotonic_ns
            < by_path["entry"].terminal_receipt_monotonic_ns
        ):
            raise ValueError("Round 74 execution flat round trip differs")
        pair_counts[symbol] += 1
        side_pair_counts[(symbol, by_path["entry"].side)] += 1
    if any(
        count < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
        for count in pair_counts.values()
    ) or any(
        count < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
        for count in side_pair_counts.values()
    ):
        raise ValueError("Round 74 execution calibration sample is incomplete")

    entry_latencies: dict[str, int] = {}
    exit_latencies: dict[str, int] = {}
    slippage: dict[str, float] = {}
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        entries = [
            item.latency_ns
            for item in selected
            if item.symbol == symbol and item.path == "entry"
        ]
        exits = [
            item.latency_ns
            for item in selected
            if item.symbol == symbol and item.path == "exit"
        ]
        residuals = [
            item.residual_slippage_bps
            for item in selected
            if item.symbol == symbol
        ]
        entry_latencies[symbol] = int(
            _upper_confidence_quantile(
                entries,
                quantile=ROUND74_EXECUTION_CALIBRATION_QUANTILE,
                confidence=ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
            )
        )
        exit_latencies[symbol] = int(
            _upper_confidence_quantile(
                exits,
                quantile=ROUND74_EXECUTION_CALIBRATION_QUANTILE,
                confidence=ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
            )
        )
        slippage[symbol] = max(
            0.0,
            float(
                _upper_confidence_quantile(
                    residuals,
                    quantile=ROUND74_EXECUTION_CALIBRATION_QUANTILE,
                    confidence=(
                        ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE
                    ),
                )
            ),
        )
    if any(
        latency > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
        for latency in (*entry_latencies.values(), *exit_latencies.values())
    ):
        raise ValueError("Round 74 execution calibrated latency exceeds target bound")

    normalized_sources = [item.normalized_source for item in selected]
    source_payload_sha256 = _canonical_sha256(normalized_sources)
    protocol = {
        "schema_version": ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
        "environment": selected_environment,
        "symbols": list(ROUND74_EVENT_TARGET_SYMBOLS),
        "new_order_path": "/fapi/v1/order",
        "terminal_source": "ORDER_TRADE_UPDATE",
        "fill_reconciliation_path": "/fapi/v1/userTrades",
        "flat_position_confirmation_path": "/fapi/v2/positionRisk",
        "client_order_id_prefix": ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX,
        "minimum_completed_flat_pairs_per_symbol": (
            ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
        ),
        "minimum_completed_flat_pairs_per_symbol_entry_side": (
            ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
        ),
        "entry_and_exit_sampled_separately": True,
        "latency_semantics": "local submission to terminal execution receipt",
        "tail_estimator": (
            "distribution-free one-sided 95 percent upper confidence "
            "order statistic for p99"
        ),
        "tail_quantile": ROUND74_EXECUTION_CALIBRATION_QUANTILE,
        "tail_confidence": ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
        "reference_quote_notional": float(reference),
        "maximum_notional_deviation_fraction": float(
            ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION
        ),
        "residual_shortfall_semantics": (
            "signed actual VWAP shortfall after exact captured L2 book walk"
        ),
        "negative_residual_tail_floor_bps": 0.0,
        "credential_material_persisted": False,
        "calibration_places_orders": False,
        "trading_authority": False,
    }
    protocol_sha256 = _canonical_sha256(protocol)
    latency_claims = round74_latency_evidence_claims(
        decision_to_entry_latency_ns_by_symbol=entry_latencies,
        decision_to_exit_latency_ns_by_symbol=exit_latencies,
    )
    slippage_claims = round74_slippage_evidence_claims(
        reference_quote_notional=float(reference),
        additional_slippage_bps_per_side_by_symbol=slippage,
    )
    return Round74ExecutionEvidenceBundle(
        reference_quote_notional=float(reference),
        decision_to_entry_latency_ns_by_symbol=tuple(
            (symbol, entry_latencies[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        decision_to_exit_latency_ns_by_symbol=tuple(
            (symbol, exit_latencies[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        additional_slippage_bps_per_side_by_symbol=tuple(
            (symbol, slippage[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        entry_exit_latency_evidence=Round74EventTargetEvidence.create(
            kind="entry_exit_latency",
            environment=selected_environment,
            observed_wall_ns=observed,
            record_count=len(selected),
            source_query_or_protocol_sha256=protocol_sha256,
            source_payload_sha256=source_payload_sha256,
            claims=latency_claims,
        ),
        residual_slippage_evidence=Round74EventTargetEvidence.create(
            kind="residual_slippage",
            environment=selected_environment,
            observed_wall_ns=observed,
            record_count=len(selected),
            source_query_or_protocol_sha256=protocol_sha256,
            source_payload_sha256=source_payload_sha256,
            claims=slippage_claims,
        ),
    )


__all__ = [
    "ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX",
    "ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS",
    "ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL",
    "ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE",
    "ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION",
    "ROUND74_EXECUTION_CALIBRATION_QUANTILE",
    "ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE",
    "ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION",
    "Round74ExecutionEvidenceBundle",
    "build_round74_execution_calibration_evidence",
    "validate_round74_execution_calibration_record",
]
