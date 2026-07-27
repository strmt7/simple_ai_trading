from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json

import pytest

from simple_ai_trading.impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
    validate_round74_execution_calibration_record,
)
from simple_ai_trading.round74_execution_calibration_capture import (
    Round74ExecutionCaptureLeg,
    Round74ExecutionCapturePair,
)


def _record(*, path: str, side: str, submission_ns: int) -> dict[str, object]:
    order_id = 101 if path == "entry" else 102
    client_order_id = f"sat-r74-cal-pair-0-{'i' if path == 'entry' else 'o'}"
    expected_price = Decimal("100.01") if side == "BUY" else Decimal("99.99")
    actual_price = expected_price + (
        Decimal("0.01") if side == "BUY" else Decimal("-0.01")
    )
    position_key = (
        "pre_pair_position_payload" if path == "entry" else "post_pair_position_payload"
    )
    return {
        "schema_version": ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
        "environment": "binance_usdm_testnet",
        "calibration_run_id": "round74-capture-test",
        "round_trip_id": "BTCUSDT-0",
        "path": path,
        "symbol": "BTCUSDT",
        "side": side,
        "client_order_id": client_order_id,
        "submission_monotonic_ns": submission_ns,
        "terminal_receipt_monotonic_ns": submission_ns + 100_000_000,
        "expected_book_walk_source": {
            "schema_version": "round-074-execution-book-state-v1",
            "symbol": "BTCUSDT",
            "update_id": order_id,
            "received_monotonic_ns": submission_ns - 10_000_000,
            "bids": [["99.99", "10"]],
            "asks": [["100.01", "10"]],
            "source_payload_sha256": f"{order_id:064x}",
        },
        "terminal_source": "ORDER_TRADE_UPDATE",
        "terminal_order_payload": {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1_800_000_000_000 + order_id,
            "T": 1_800_000_000_000 + order_id,
            "o": {
                "s": "BTCUSDT",
                "i": order_id,
                "c": client_order_id,
                "S": side,
                "ps": "BOTH",
                "o": "MARKET",
                "x": "TRADE",
                "X": "FILLED",
                "m": False,
                "R": path == "exit",
                "q": "1",
                "z": "1",
                "ap": str(actual_price),
            },
        },
        "account_trade_payloads": [
            {
                "id": order_id + 1000,
                "orderId": order_id,
                "symbol": "BTCUSDT",
                "side": side,
                "buyer": side == "BUY",
                "maker": False,
                "price": str(actual_price),
                "qty": "1",
                "quoteQty": str(actual_price),
            }
        ],
        position_key: {
            "symbol": "BTCUSDT",
            "positionSide": "BOTH",
            "positionAmt": "0",
        },
    }


def _leg(record: dict[str, object]) -> Round74ExecutionCaptureLeg:
    return Round74ExecutionCaptureLeg(
        path=str(record["path"]),
        symbol=str(record["symbol"]),
        side=str(record["side"]),
        client_order_id=str(record["client_order_id"]),
        submission_monotonic_ns=int(record["submission_monotonic_ns"]),
        terminal_receipt_monotonic_ns=int(record["terminal_receipt_monotonic_ns"]),
        expected_book_walk_source=record["expected_book_walk_source"],
        terminal_order_payload=record["terminal_order_payload"],
        account_trade_payloads=tuple(record["account_trade_payloads"]),
    )


def _pair() -> Round74ExecutionCapturePair:
    entry = _record(path="entry", side="BUY", submission_ns=1_000_000_000)
    exit_record = _record(
        path="exit",
        side="SELL",
        submission_ns=1_200_000_000,
    )
    return Round74ExecutionCapturePair(
        calibration_run_id="round74-capture-test",
        round_trip_id="BTCUSDT-0",
        symbol="BTCUSDT",
        entry=_leg(entry),
        exit=_leg(exit_record),
        pre_pair_position_payload=entry["pre_pair_position_payload"],
        post_pair_position_payload=exit_record["post_pair_position_payload"],
        reference_quote_notional="100",
    )


def test_capture_pair_emits_two_parser_valid_secret_free_records() -> None:
    pair = _pair()
    entry, exit_record = pair.records()

    assert [entry["path"], exit_record["path"]] == ["entry", "exit"]
    assert (
        entry["environment"] == exit_record["environment"] == ("binance_usdm_testnet")
    )
    assert entry["pre_pair_position_payload"]["positionAmt"] == "0"
    assert exit_record["post_pair_position_payload"]["positionAmt"] == "0"
    artifact = pair.as_dict()
    assert artifact["pair_sha256"] == pair.pair_sha256
    assert artifact["authority"]["mainnet_orders_submitted"] is False
    assert "signature" not in json.dumps(artifact, sort_keys=True)


def test_public_single_record_validator_preserves_exact_source() -> None:
    record = _record(path="entry", side="BUY", submission_ns=1_000_000_000)

    normalized = validate_round74_execution_calibration_record(
        record=record,
        reference_quote_notional=100.0,
    )

    assert normalized == record
    assert normalized is not record


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("credential", "contains credentials"),
        ("exit_quantity", "flat round trip differs"),
        ("stale_book", "book state is stale"),
        ("wrong_environment", "capture pair differs"),
    ],
)
def test_capture_pair_rejects_non_admissible_sources(
    mutation: str,
    match: str,
) -> None:
    pair = _pair()
    if mutation == "credential":
        terminal = deepcopy(dict(pair.entry.terminal_order_payload))
        terminal["signature"] = "never-persist"
        pair = Round74ExecutionCapturePair(
            **{
                **pair.__dict__,
                "entry": Round74ExecutionCaptureLeg(
                    **{**pair.entry.__dict__, "terminal_order_payload": terminal}
                ),
            }
        )
    elif mutation == "exit_quantity":
        terminal = deepcopy(dict(pair.exit.terminal_order_payload))
        terminal["o"]["q"] = "0.99"
        terminal["o"]["z"] = "0.99"
        trades = deepcopy(list(pair.exit.account_trade_payloads))
        trades[0]["qty"] = "0.99"
        trades[0]["quoteQty"] = str(Decimal(trades[0]["price"]) * Decimal("0.99"))
        pair = Round74ExecutionCapturePair(
            **{
                **pair.__dict__,
                "exit": Round74ExecutionCaptureLeg(
                    **{
                        **pair.exit.__dict__,
                        "terminal_order_payload": terminal,
                        "account_trade_payloads": tuple(trades),
                    }
                ),
            }
        )
    elif mutation == "stale_book":
        book = deepcopy(dict(pair.entry.expected_book_walk_source))
        book["received_monotonic_ns"] = 1
        pair = Round74ExecutionCapturePair(
            **{
                **pair.__dict__,
                "entry": Round74ExecutionCaptureLeg(
                    **{**pair.entry.__dict__, "expected_book_walk_source": book}
                ),
            }
        )
    else:
        pair = Round74ExecutionCapturePair(
            **{**pair.__dict__, "environment": "binance_usdm_mainnet"}
        )

    with pytest.raises(ValueError, match=match):
        pair.records()
