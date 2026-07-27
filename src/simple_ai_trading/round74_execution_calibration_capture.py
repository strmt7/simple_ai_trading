"""Canonical source records for non-mainnet Round 74 execution calibration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Mapping, Sequence

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX,
    ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
    validate_round74_execution_calibration_record,
)


ROUND74_EXECUTION_CAPTURE_PAIR_SCHEMA_VERSION = "round-074-execution-capture-pair-v1"
ROUND74_EXECUTION_CAPTURE_ENVIRONMENT = "binance_usdm_testnet"
ROUND74_EXECUTION_CAPTURE_PATHS = ("entry", "exit")
_SENSITIVE_KEYS = frozenset(
    {"apikey", "secret", "secretkey", "signature", "xmbxapikey"}
)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Round 74 execution capture value is not canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _reject_sensitive_keys(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "").replace("_", "")
            if normalized in _SENSITIVE_KEYS:
                raise ValueError(
                    f"Round 74 execution capture {path} contains credentials"
                )
            _reject_sensitive_keys(nested, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_sensitive_keys(nested, path=f"{path}[{index}]")


def _normalized_mapping(
    value: Mapping[str, object], *, label: str
) -> dict[str, object]:
    _reject_sensitive_keys(value, path=label)
    normalized = json.loads(_canonical_json(dict(value)))
    if not isinstance(normalized, dict):
        raise ValueError(f"Round 74 execution capture {label} differs")
    return normalized


def _positive_decimal(value: object, *, label: str) -> Decimal:
    try:
        selected = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Round 74 execution capture {label} differs") from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError(f"Round 74 execution capture {label} differs")
    return selected


def _strict_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Round 74 execution capture {label} differs")
    return value


@dataclass(frozen=True)
class Round74ExecutionCaptureLeg:
    """Raw exchange sources for one market-order leg."""

    path: str
    symbol: str
    side: str
    client_order_id: str
    submission_monotonic_ns: int
    terminal_receipt_monotonic_ns: int
    expected_book_walk_source: Mapping[str, object]
    terminal_order_payload: Mapping[str, object]
    account_trade_payloads: tuple[Mapping[str, object], ...]

    def validate(self) -> None:
        if (
            self.path not in ROUND74_EXECUTION_CAPTURE_PATHS
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or self.side not in {"BUY", "SELL"}
            or not self.client_order_id.startswith(
                ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX
            )
            or len(self.client_order_id) > 36
            or _strict_positive_integer(
                self.submission_monotonic_ns,
                label="submission monotonic time",
            )
            >= _strict_positive_integer(
                self.terminal_receipt_monotonic_ns,
                label="terminal receipt monotonic time",
            )
            or not self.account_trade_payloads
        ):
            raise ValueError("Round 74 execution capture leg identity differs")
        _normalized_mapping(
            self.expected_book_walk_source,
            label="book source",
        )
        terminal = _normalized_mapping(
            self.terminal_order_payload,
            label="terminal source",
        )
        trades = tuple(
            _normalized_mapping(trade, label="account trade")
            for trade in self.account_trade_payloads
        )
        if (
            terminal.get("e") != "ORDER_TRADE_UPDATE"
            or not isinstance(terminal.get("o"), Mapping)
            or terminal["o"].get("c") != self.client_order_id
            or terminal["o"].get("s") != self.symbol
            or terminal["o"].get("S") != self.side
            or terminal["o"].get("R") is not (self.path == "exit")
            or any(
                trade.get("symbol") != self.symbol or trade.get("side") != self.side
                for trade in trades
            )
        ):
            raise ValueError("Round 74 execution capture leg source differs")

    def as_record(
        self,
        *,
        calibration_run_id: str,
        round_trip_id: str,
        environment: str,
        flat_position_payload: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate()
        run_id = str(calibration_run_id).strip()
        pair_id = str(round_trip_id).strip()
        if not run_id or not pair_id:
            raise ValueError("Round 74 execution capture pair identity differs")
        record: dict[str, object] = {
            "schema_version": ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
            "environment": str(environment),
            "calibration_run_id": run_id,
            "round_trip_id": pair_id,
            "path": self.path,
            "symbol": self.symbol,
            "side": self.side,
            "client_order_id": self.client_order_id,
            "submission_monotonic_ns": self.submission_monotonic_ns,
            "terminal_receipt_monotonic_ns": (self.terminal_receipt_monotonic_ns),
            "terminal_source": "ORDER_TRADE_UPDATE",
            "expected_book_walk_source": _normalized_mapping(
                self.expected_book_walk_source,
                label="book source",
            ),
            "terminal_order_payload": _normalized_mapping(
                self.terminal_order_payload,
                label="terminal source",
            ),
            "account_trade_payloads": [
                _normalized_mapping(trade, label="account trade")
                for trade in self.account_trade_payloads
            ],
            (
                "pre_pair_position_payload"
                if self.path == "entry"
                else "post_pair_position_payload"
            ): _normalized_mapping(
                flat_position_payload,
                label="flat position source",
            ),
        }
        _reject_sensitive_keys(record)
        return record


@dataclass(frozen=True)
class Round74ExecutionCapturePair:
    """One complete, flat-to-flat pair admitted for later panel assembly."""

    calibration_run_id: str
    round_trip_id: str
    symbol: str
    entry: Round74ExecutionCaptureLeg
    exit: Round74ExecutionCaptureLeg
    pre_pair_position_payload: Mapping[str, object]
    post_pair_position_payload: Mapping[str, object]
    reference_quote_notional: str
    environment: str = ROUND74_EXECUTION_CAPTURE_ENVIRONMENT
    schema_version: str = ROUND74_EXECUTION_CAPTURE_PAIR_SCHEMA_VERSION

    def records(self) -> tuple[dict[str, object], dict[str, object]]:
        reference = _positive_decimal(
            self.reference_quote_notional,
            label="reference quote notional",
        )
        if (
            self.schema_version != ROUND74_EXECUTION_CAPTURE_PAIR_SCHEMA_VERSION
            or self.environment != ROUND74_EXECUTION_CAPTURE_ENVIRONMENT
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or self.entry.path != "entry"
            or self.exit.path != "exit"
            or self.entry.symbol != self.symbol
            or self.exit.symbol != self.symbol
            or self.entry.side == self.exit.side
            or not str(self.calibration_run_id).strip()
            or not str(self.round_trip_id).strip()
        ):
            raise ValueError("Round 74 execution capture pair differs")
        entry_record = self.entry.as_record(
            calibration_run_id=self.calibration_run_id,
            round_trip_id=self.round_trip_id,
            environment=self.environment,
            flat_position_payload=self.pre_pair_position_payload,
        )
        exit_record = self.exit.as_record(
            calibration_run_id=self.calibration_run_id,
            round_trip_id=self.round_trip_id,
            environment=self.environment,
            flat_position_payload=self.post_pair_position_payload,
        )
        normalized_entry = validate_round74_execution_calibration_record(
            record=entry_record,
            reference_quote_notional=float(reference),
        )
        normalized_exit = validate_round74_execution_calibration_record(
            record=exit_record,
            reference_quote_notional=float(reference),
        )
        entry_order = normalized_entry["terminal_order_payload"]["o"]
        exit_order = normalized_exit["terminal_order_payload"]["o"]
        if (
            not isinstance(entry_order, Mapping)
            or not isinstance(exit_order, Mapping)
            or entry_order.get("z") != exit_order.get("z")
            or self.exit.submission_monotonic_ns
            < self.entry.terminal_receipt_monotonic_ns
        ):
            raise ValueError("Round 74 execution capture flat round trip differs")
        return normalized_entry, normalized_exit

    @property
    def pair_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        entry, exit_record = self.records()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "environment": self.environment,
            "calibration_run_id": self.calibration_run_id,
            "round_trip_id": self.round_trip_id,
            "symbol": self.symbol,
            "reference_quote_notional": format(
                _positive_decimal(
                    self.reference_quote_notional,
                    label="reference quote notional",
                ),
                "f",
            ),
            "records": [entry, exit_record],
            "authority": {
                "source_records_only": True,
                "model_training": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
                "mainnet_orders_submitted": False,
                "mainnet_trading_authority": False,
            },
        }
        if include_sha256:
            value["pair_sha256"] = _canonical_sha256(value)
        return value


__all__ = [
    "ROUND74_EXECUTION_CAPTURE_ENVIRONMENT",
    "ROUND74_EXECUTION_CAPTURE_PAIR_SCHEMA_VERSION",
    "ROUND74_EXECUTION_CAPTURE_PATHS",
    "Round74ExecutionCaptureLeg",
    "Round74ExecutionCapturePair",
]
