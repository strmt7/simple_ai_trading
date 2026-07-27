from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS,
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL,
    ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
    build_round74_execution_calibration_evidence,
)
from simple_ai_trading.impact_absorption_event_targets import (
    round74_latency_evidence_claims,
    round74_slippage_evidence_claims,
)
from simple_ai_trading.impact_absorption_event_evidence import (
    Round74BinanceClockProbe,
)
from simple_ai_trading.impact_absorption_target_assembly import (
    assemble_round74_source_target,
)
from simple_ai_trading.impact_absorption_targets import (
    Round73MarketQuantityRules,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WALL_NS = 1_784_000_000_000_000_000
REFERENCE_NOTIONAL = Decimal("100")
MONOTONIC_NS = 1_000_000_000_000
EXCHANGE_MS = 1_784_000_000_000


def _record(
    *,
    symbol: str,
    pair_index: int,
    path: str,
    side: str,
    submission_ns: int,
    latency_ns: int,
    residual_bps: Decimal,
) -> dict[str, object]:
    symbol_index = SYMBOLS.index(symbol)
    order_id = 1_000_000 + symbol_index * 10_000 + pair_index * 2
    if path == "exit":
        order_id += 1
    path_code = "i" if path == "entry" else "o"
    client_order_id = f"sat-r74-cal-{symbol_index}-{pair_index}-{path_code}"
    expected_price = Decimal("100.01") if side == "BUY" else Decimal("99.99")
    side_sign = Decimal("1") if side == "BUY" else Decimal("-1")
    actual_price = expected_price * (
        Decimal("1") + side_sign * residual_bps / Decimal("10000")
    )
    quantity = Decimal("1")
    quote = actual_price * quantity
    position_key = (
        "pre_pair_position_payload"
        if path == "entry"
        else "post_pair_position_payload"
    )
    return {
        "schema_version": ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
        "calibration_run_id": "round74-execution-contract-test",
        "round_trip_id": f"{symbol}-{pair_index}",
        "path": path,
        "symbol": symbol,
        "side": side,
        "client_order_id": client_order_id,
        "submission_monotonic_ns": submission_ns,
        "terminal_receipt_monotonic_ns": submission_ns + latency_ns,
        "expected_book_walk_source": {
            "schema_version": "round-074-execution-book-state-v1",
            "symbol": symbol,
            "update_id": order_id,
            "received_monotonic_ns": submission_ns - 10_000_000,
            "bids": [["99.99", "10"]],
            "asks": [["100.01", "10"]],
            "source_payload_sha256": f"{order_id:064x}",
        },
        "terminal_source": "ORDER_TRADE_UPDATE",
        "terminal_order_payload": {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1_784_000_000_001 + order_id,
            "T": 1_784_000_000_000 + order_id,
            "o": {
                "s": symbol,
                "i": order_id,
                "c": client_order_id,
                "S": side,
                "ps": "BOTH",
                "o": "MARKET",
                "x": "TRADE",
                "X": "FILLED",
                "m": False,
                "R": path == "exit",
                "q": str(quantity),
                "z": str(quantity),
                "ap": str(actual_price),
            },
        },
        "account_trade_payloads": [
            {
                "id": order_id + 10_000_000,
                "orderId": order_id,
                "symbol": symbol,
                "side": side,
                "buyer": side == "BUY",
                "maker": False,
                "price": str(actual_price),
                "qty": str(quantity),
                "quoteQty": str(quote),
            }
        ],
        position_key: {
            "symbol": symbol,
            "positionSide": "BOTH",
            "positionAmt": "0",
        },
    }


def _records(
    pairs_per_symbol: int = (
        ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
    ),
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    clock = 1_000_000_000_000
    for symbol_index, symbol in enumerate(SYMBOLS):
        for pair_index in range(pairs_per_symbol):
            entry_latency = 100_000_000 + pair_index * 1_000 + symbol_index
            exit_latency = 120_000_000 + pair_index * 2_000 + symbol_index
            residual = Decimal(pair_index) / Decimal("100")
            entry_side = "BUY" if pair_index % 2 == 0 else "SELL"
            exit_side = "SELL" if entry_side == "BUY" else "BUY"
            entry = _record(
                symbol=symbol,
                pair_index=pair_index,
                path="entry",
                side=entry_side,
                submission_ns=clock,
                latency_ns=entry_latency,
                residual_bps=residual,
            )
            exit_submission = (
                int(entry["terminal_receipt_monotonic_ns"]) + 1_000_000
            )
            exit_record = _record(
                symbol=symbol,
                pair_index=pair_index,
                path="exit",
                side=exit_side,
                submission_ns=exit_submission,
                latency_ns=exit_latency,
                residual_bps=residual,
            )
            records.extend((entry, exit_record))
            clock = (
                int(exit_record["terminal_receipt_monotonic_ns"])
                + 1_000_000
            )
    return records


def _bundle(records: list[dict[str, object]] | None = None):
    return build_round74_execution_calibration_evidence(
        records=_records() if records is None else records,
        environment="binance_usdm_testnet",
        observed_wall_ns=WALL_NS,
        reference_quote_notional=float(REFERENCE_NOTIONAL),
    )


def _exchange_info_payload() -> dict[str, object]:
    return {
        "timezone": "UTC",
        "serverTime": EXCHANGE_MS,
        "symbols": [
            {
                "symbol": symbol,
                "pair": symbol,
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "orderTypes": ["LIMIT", "MARKET"],
                "filters": [
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "1000",
                        "stepSize": "0.001",
                    },
                    {
                        "filterType": "MIN_NOTIONAL",
                        "notional": "10",
                    },
                ],
            }
            for symbol in SYMBOLS
        ],
    }


def _commission_payloads() -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "symbol": symbol,
            "makerCommissionRate": "0.0002",
            "takerCommissionRate": "0.0005",
            "rpiCommissionRate": "0.00005",
        }
        for symbol in SYMBOLS
    }


def _clock_probes() -> tuple[Round74BinanceClockProbe, ...]:
    return tuple(
        Round74BinanceClockProbe(
            capture_run_id="round74-source-assembly-test",
            capture_contract_sha256="a" * 64,
            capture_audit_sha256="b" * 64,
            frame_index=index,
            message_index=index,
            request_started_wall_ns=WALL_NS + index * 60_000_000_000,
            received_wall_ns=WALL_NS + index * 60_000_000_000 + 20_000_000,
            request_started_monotonic_ns=(
                MONOTONIC_NS + index * 60_000_000_000
            ),
            received_monotonic_ns=(
                MONOTONIC_NS + index * 60_000_000_000 + 20_000_000
            ),
            exchange_time_ms=EXCHANGE_MS + index * 60_000,
            source_payload_sha256=f"{index + 1:064x}",
        )
        for index in range(6)
    )


def _funding_payloads() -> dict[str, list[dict[str, object]]]:
    return {
        symbol: [
            {
                "symbol": symbol,
                "fundingRate": "0.0001",
                "fundingTime": EXCHANGE_MS + 150_000,
                "markPrice": "100.0",
                "rateType": "Regular",
            }
        ]
        for symbol in SYMBOLS
    }


def _assembly(**overrides: object):
    values = {
        "exchange_info_payload": _exchange_info_payload(),
        "commission_payload_by_symbol": _commission_payloads(),
        "funding_payload_by_symbol": _funding_payloads(),
        "execution_calibration_records": _records(),
        "funding_clock_probes": _clock_probes(),
        "environment": "binance_usdm_testnet",
        "exchange_info_observed_wall_ns": WALL_NS,
        "commission_observed_wall_ns": WALL_NS + 1,
        "funding_observed_wall_ns": WALL_NS + 2,
        "execution_observed_wall_ns": WALL_NS + 3,
        "funding_start_time_ms": EXCHANGE_MS - 3_600_000,
        "funding_end_time_ms": EXCHANGE_MS + 3_600_000,
        "funding_limit": 1000,
        "reference_quote_notional": float(REFERENCE_NOTIONAL),
    }
    values.update(overrides)
    return assemble_round74_source_target(**values)


def test_source_target_assembly_derives_every_configured_value() -> None:
    assembly = _assembly()
    spec = assembly.spec

    assert dict(spec.taker_fee_bps_by_symbol) == {
        symbol: 5.0 for symbol in SYMBOLS
    }
    assert dict(spec.decision_to_entry_latency_ns_by_symbol) == {
        "BTCUSDT": 100_299_000,
        "ETHUSDT": 100_299_001,
        "SOLUSDT": 100_299_002,
    }
    assert spec.execution_environment == "binance_usdm_testnet"
    restored = type(assembly).from_dict(assembly.as_dict())
    assert restored.as_dict() == assembly.as_dict()
    assert restored.assembly_sha256 == assembly.assembly_sha256

    tampered = assembly.as_dict()
    tampered["quantity_rules_by_symbol"]["BTCUSDT"]["maximum_quantity"] = "999"
    with pytest.raises(ValueError, match="assembly"):
        type(assembly).from_dict(tampered)
    malformed = assembly.as_dict()
    del malformed["quantity_rules_by_symbol"]["SOLUSDT"]
    with pytest.raises(ValueError, match="payload differs"):
        type(assembly).from_dict(malformed)
    malformed = assembly.as_dict()
    del malformed["quantity_rules_by_symbol"]["ETHUSDT"]["minimum_notional"]
    with pytest.raises(ValueError, match="quantity rules differ"):
        type(assembly).from_dict(malformed)
    malformed = assembly.as_dict()
    malformed["assembly_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="digest differs"):
        type(assembly).from_dict(malformed)
    assert len(assembly.assembly_sha256) == 64
    assert assembly.create_engine(anchors=[]).spec.spec_sha256 == (
        spec.spec_sha256
    )


def test_source_target_assembly_rejects_tampered_rules_and_sources() -> None:
    assembly = _assembly()
    rules = assembly.quantity_rules_mapping()
    rules["BTCUSDT"] = Round73MarketQuantityRules.create(
        symbol="BTCUSDT",
        step_size="0.01",
        minimum_quantity="0.01",
        maximum_quantity="1000",
        minimum_notional="10",
    )
    with pytest.raises(ValueError, match="evidence claims differ"):
        replace(
            assembly,
            quantity_rules_by_symbol=tuple(sorted(rules.items())),
        )

    commissions = _commission_payloads()
    commissions["BTCUSDT"]["api_key"] = "must-not-persist"
    with pytest.raises(ValueError, match="credential material"):
        _assembly(commission_payload_by_symbol=commissions)


def test_execution_calibration_derives_symbol_path_tail_evidence() -> None:
    bundle = _bundle()

    assert bundle.entry_latency_mapping() == {
        "BTCUSDT": 100_299_000,
        "ETHUSDT": 100_299_001,
        "SOLUSDT": 100_299_002,
    }
    assert bundle.exit_latency_mapping() == {
        "BTCUSDT": 120_598_000,
        "ETHUSDT": 120_598_001,
        "SOLUSDT": 120_598_002,
    }
    assert bundle.slippage_mapping() == {
        symbol: pytest.approx(2.99) for symbol in SYMBOLS
    }
    assert bundle.entry_exit_latency_evidence.record_count == 1800
    assert bundle.entry_exit_latency_evidence.binds(
        round74_latency_evidence_claims(
            decision_to_entry_latency_ns_by_symbol=(
                bundle.entry_latency_mapping()
            ),
            decision_to_exit_latency_ns_by_symbol=(
                bundle.exit_latency_mapping()
            ),
        )
    )
    assert bundle.residual_slippage_evidence.binds(
        round74_slippage_evidence_claims(
            reference_quote_notional=float(REFERENCE_NOTIONAL),
            additional_slippage_bps_per_side_by_symbol=(
                bundle.slippage_mapping()
            ),
        )
    )


def test_execution_calibration_rejects_incomplete_or_nonflat_pairs() -> None:
    records = _records(
        ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL - 1
    )
    with pytest.raises(ValueError, match="sample is incomplete"):
        _bundle(records)

    records = _records()
    records[1]["post_pair_position_payload"]["positionAmt"] = "0.001"
    with pytest.raises(ValueError, match="not flat"):
        _bundle(records)


@pytest.mark.parametrize(
    "mutation, match",
    (
        ("foreign_client_id", "identity differs"),
        ("maker_fill", "reconciliation differs"),
        ("quantity_mismatch", "terminal quantity differs"),
        ("same_side_exit", "flat round trip differs"),
        ("stale_book", "stale or future"),
        ("buyer_mismatch", "reconciliation differs"),
        ("credential", "credential material"),
    ),
)
def test_execution_calibration_rejects_untrusted_sources(
    mutation: str,
    match: str,
) -> None:
    records = _records()
    selected = records[1]
    if mutation == "foreign_client_id":
        selected["client_order_id"] = "manual-order"
        selected["terminal_order_payload"]["clientOrderId"] = "manual-order"
    elif mutation == "maker_fill":
        selected["account_trade_payloads"][0]["maker"] = True
    elif mutation == "quantity_mismatch":
        selected["terminal_order_payload"]["o"]["z"] = "2"
    elif mutation == "same_side_exit":
        selected["side"] = "BUY"
        selected["terminal_order_payload"]["o"]["S"] = "BUY"
        selected["account_trade_payloads"][0]["side"] = "BUY"
        selected["account_trade_payloads"][0]["buyer"] = True
    elif mutation == "stale_book":
        selected["expected_book_walk_source"]["received_monotonic_ns"] = (
            int(selected["submission_monotonic_ns"])
            - ROUND74_EXECUTION_CALIBRATION_MAXIMUM_BOOK_AGE_NS
            - 1
        )
    elif mutation == "buyer_mismatch":
        selected["account_trade_payloads"][0]["buyer"] = True
    else:
        selected["terminal_order_payload"]["signature"] = "must-not-persist"

    with pytest.raises(ValueError, match=match):
        _bundle(records)


def test_execution_calibration_digest_changes_with_exact_source() -> None:
    records = _records()
    original = _bundle(records)
    changed_records = deepcopy(records)
    changed_records[0]["terminal_order_payload"]["updateTime"] = 1
    changed = _bundle(changed_records)

    assert (
        original.entry_exit_latency_evidence.source_payload_sha256
        != changed.entry_exit_latency_evidence.source_payload_sha256
    )
