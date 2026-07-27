from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib

import pytest
import zstandard

from simple_ai_trading.impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from simple_ai_trading.impact_absorption_event_evidence import (
    ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS,
    Round74BinanceClockProbe,
    build_round74_commission_evidence,
    build_round74_funding_evidence,
    build_round74_quantity_rules_evidence,
    load_round74_binance_clock_probes,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
    ImpactCaptureAudit,
)
from simple_ai_trading.impact_absorption_event_targets import (
    round74_commission_evidence_claims,
    round74_funding_schedule_evidence_claims,
    round74_quantity_rules_evidence_claims,
)
from simple_ai_trading.impact_capture_frame import (
    ImpactCaptureFrameRecord,
    encode_impact_capture_frame,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WALL_NS = 1_784_000_000_000_000_000
MONOTONIC_NS = 1_000_000_000_000
EXCHANGE_MS = 1_784_000_000_000
CAPTURE_CONTRACT_SHA256 = "a" * 64


class _Result:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class _ClockConnection:
    def __init__(
        self,
        *,
        run_row: tuple[object, ...],
        context_rows: list[tuple[object, ...]],
        frame_rows: list[tuple[object, ...]],
    ) -> None:
        self.run_row = run_row
        self.context_rows = context_rows
        self.frame_rows = frame_rows

    def execute(
        self,
        query: str,
        _parameters: list[object],
    ) -> _Result:
        if "FROM impact_capture_run" in query:
            return _Result([self.run_row])
        if "SELECT DISTINCT f.frame_index" in query:
            return _Result(self.frame_rows)
        if "event_type = 'serverTime'" in query:
            return _Result(self.context_rows)
        raise AssertionError(f"unexpected query: {query}")


def _commission_payloads() -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "symbol": symbol,
            "makerCommissionRate": "0.0002",
            "takerCommissionRate": rate,
            "rpiCommissionRate": "0.00005",
        }
        for symbol, rate in zip(
            SYMBOLS,
            ("0.0004", "0.00045", "0.0005"),
            strict=True,
        )
    }


def _exchange_info_payload() -> dict[str, object]:
    rows = []
    for symbol, step, minimum, maximum, notional in zip(
        SYMBOLS,
        ("0.001", "0.001", "0.1"),
        ("0.001", "0.001", "0.1"),
        ("1000", "10000", "100000"),
        ("100", "20", "5"),
        strict=True,
    ):
        rows.append(
            {
                "symbol": symbol,
                "pair": symbol,
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "quantityPrecision": 3,
                "orderTypes": ["LIMIT", "MARKET", "STOP_MARKET"],
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01",
                        "maxPrice": "1000000",
                        "tickSize": "0.01",
                    },
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": minimum,
                        "maxQty": maximum,
                        "stepSize": step,
                    },
                    {
                        "filterType": "MIN_NOTIONAL",
                        "notional": notional,
                    },
                ],
            }
        )
    rows.append({"symbol": "DOGEUSDT", "status": "TRADING"})
    return {
        "timezone": "UTC",
        "serverTime": EXCHANGE_MS,
        "rateLimits": [],
        "symbols": rows,
    }


def _clock_probes(count: int = 6) -> tuple[Round74BinanceClockProbe, ...]:
    probes = []
    for index in range(count):
        started = MONOTONIC_NS + index * 60_000_000_000
        probes.append(
            Round74BinanceClockProbe(
                capture_run_id="round74-contract-test",
                capture_contract_sha256=CAPTURE_CONTRACT_SHA256,
                capture_audit_sha256="b" * 64,
                frame_index=index,
                message_index=index + 1,
                request_started_wall_ns=WALL_NS + index * 60_000_000_000,
                received_wall_ns=WALL_NS + index * 60_000_000_000 + 20_000_000,
                request_started_monotonic_ns=started,
                received_monotonic_ns=started + 20_000_000,
                exchange_time_ms=EXCHANGE_MS + index * 60_000,
                source_payload_sha256=f"{index + 1:064x}",
            )
        )
    return tuple(probes)


def _clock_capture_fixture() -> tuple[
    _ClockConnection,
    ImpactCaptureAudit,
]:
    context_rows: list[tuple[object, ...]] = []
    frame_rows: list[tuple[object, ...]] = []
    compressor = zstandard.ZstdCompressor(level=1)
    for index in range(6):
        exchange_time_ms = EXCHANGE_MS + index * 60_000
        started_wall_ns = WALL_NS + index * 60_000_000_000
        started_monotonic_ns = MONOTONIC_NS + index * 60_000_000_000
        record = ImpactCaptureFrameRecord(
            stream="binance_futures_rest",
            connection_id="binance-rest:round74-capture-test",
            sequence_number=index,
            received_wall_ns=started_wall_ns + 20_000_000,
            received_monotonic_ns=started_monotonic_ns + 20_000_000,
            raw_text=f'{{"serverTime":{exchange_time_ms}}}',
        )
        uncompressed, _located = encode_impact_capture_frame([record])
        compressed = compressor.compress(uncompressed)
        context_rows.append(
            (
                index,
                0,
                "/fapi/v1/time",
                "{}",
                200,
                started_wall_ns,
                started_monotonic_ns,
                exchange_time_ms,
            )
        )
        frame_rows.append(
            (
                index,
                1,
                len(uncompressed),
                hashlib.sha256(uncompressed).hexdigest(),
                len(compressed),
                hashlib.sha256(compressed).hexdigest(),
                compressed,
            )
        )
    last_frame_sha256 = "c" * 64
    audit = ImpactCaptureAudit(
        run_id="round74-capture-test",
        passed=True,
        errors=(),
        frame_count=6,
        message_count=6,
        compressed_payload_bytes=sum(int(row[4]) for row in frame_rows),
        last_frame_sha256=last_frame_sha256,
        capture_contract_sha256=IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    )
    connection = _ClockConnection(
        run_row=(
            IMPACT_CAPTURE_V10_SCHEMA_VERSION,
            ROUND74_CAPTURE_DESIGN_SHA256,
            IMPACT_CAPTURE_V10_CONTRACT_SHA256,
            "completed",
            last_frame_sha256,
        ),
        context_rows=context_rows,
        frame_rows=frame_rows,
    )
    return connection, audit


def _funding_payloads(
    funding_time_ms: int | None = None,
) -> dict[str, list[dict[str, object]]]:
    selected_time = (
        EXCHANGE_MS + 150_000
        if funding_time_ms is None
        else funding_time_ms
    )
    return {
        symbol: [
            {
                "symbol": symbol,
                "fundingRate": "0.0001",
                "fundingTime": selected_time,
                "markPrice": "100.0",
                "rateType": "Regular",
            }
        ]
        for symbol in SYMBOLS
    }


def _funding_bundle(
    *,
    payloads: dict[str, list[dict[str, object]]] | None = None,
    probes: tuple[Round74BinanceClockProbe, ...] | None = None,
    limit: int = 1000,
):
    return build_round74_funding_evidence(
        payload_by_symbol=(
            _funding_payloads() if payloads is None else payloads
        ),
        environment="binance_usdm_mainnet",
        observed_wall_ns=WALL_NS,
        start_time_ms=EXCHANGE_MS - 3_600_000,
        end_time_ms=EXCHANGE_MS + 3_600_000,
        limit=limit,
        clock_probes=_clock_probes() if probes is None else probes,
    )


def test_commission_evidence_is_derived_from_exact_symbol_payloads() -> None:
    bundle = build_round74_commission_evidence(
        payload_by_symbol=_commission_payloads(),
        environment="binance_usdm_mainnet",
        observed_wall_ns=WALL_NS,
    )

    assert bundle.as_mapping() == {
        "BTCUSDT": 4.0,
        "ETHUSDT": 4.5,
        "SOLUSDT": 5.0,
    }
    assert bundle.evidence.record_count == 3
    assert bundle.evidence.binds(
        round74_commission_evidence_claims(bundle.as_mapping())
    )


def test_quantity_rules_are_derived_from_executable_exchange_filters() -> None:
    bundle = build_round74_quantity_rules_evidence(
        payload=_exchange_info_payload(),
        environment="binance_usdm_mainnet",
        observed_wall_ns=WALL_NS,
    )
    rules = bundle.as_mapping()

    assert rules["BTCUSDT"].step_size == Decimal("0.001")
    assert rules["ETHUSDT"].minimum_notional == Decimal("20")
    assert rules["SOLUSDT"].maximum_quantity == Decimal("100000")
    assert bundle.evidence.record_count == 3
    assert bundle.evidence.binds(
        round74_quantity_rules_evidence_claims(rules)
    )


def test_quantity_rules_source_digest_is_canonical() -> None:
    payload = _exchange_info_payload()
    reordered = dict(reversed(tuple(payload.items())))

    first = build_round74_quantity_rules_evidence(
        payload=payload,
        environment="binance_usdm_mainnet",
        observed_wall_ns=WALL_NS,
    )
    second = build_round74_quantity_rules_evidence(
        payload=reordered,
        environment="binance_usdm_mainnet",
        observed_wall_ns=WALL_NS,
    )

    assert (
        first.evidence.source_query_or_protocol_sha256
        == second.evidence.source_query_or_protocol_sha256
    )
    assert (
        first.evidence.source_payload_sha256
        == second.evidence.source_payload_sha256
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_symbol",
        "nontrading",
        "duplicate_filter",
        "absent_market_filter",
        "precision_only",
        "wrong_assets",
        "credential",
    ),
)
def test_quantity_rules_reject_ambiguous_exchange_metadata(
    mutation: str,
) -> None:
    payload = deepcopy(_exchange_info_payload())
    rows = payload["symbols"]
    assert isinstance(rows, list)
    if mutation == "missing_symbol":
        rows.pop(2)
    elif mutation == "nontrading":
        rows[0]["status"] = "SETTLING"
    elif mutation == "duplicate_filter":
        rows[1]["filters"].append(deepcopy(rows[1]["filters"][1]))
    elif mutation in {"absent_market_filter", "precision_only"}:
        rows[2]["filters"] = [
            item
            for item in rows[2]["filters"]
            if item["filterType"] != "MARKET_LOT_SIZE"
        ]
        if mutation == "absent_market_filter":
            rows[2].pop("quantityPrecision")
    elif mutation == "wrong_assets":
        rows[1]["marginAsset"] = "BUSD"
    else:
        rows[0]["api_key"] = "must-not-persist"

    with pytest.raises(ValueError):
        build_round74_quantity_rules_evidence(
            payload=payload,
            environment="binance_usdm_mainnet",
            observed_wall_ns=WALL_NS,
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_symbol", "wrong_symbol", "numeric_rate", "credential"),
)
def test_commission_evidence_rejects_untrustworthy_payloads(
    mutation: str,
) -> None:
    payloads = _commission_payloads()
    if mutation == "missing_symbol":
        payloads.pop("SOLUSDT")
    elif mutation == "wrong_symbol":
        payloads["SOLUSDT"]["symbol"] = "DOGEUSDT"
    elif mutation == "numeric_rate":
        payloads["ETHUSDT"]["takerCommissionRate"] = 0.00045
    else:
        payloads["BTCUSDT"]["X-MBX-APIKEY"] = "must-not-persist"

    with pytest.raises(ValueError):
        build_round74_commission_evidence(
            payload_by_symbol=payloads,
            environment="binance_usdm_mainnet",
            observed_wall_ns=WALL_NS,
        )


def test_funding_evidence_uses_noninterpolated_clock_brackets() -> None:
    bundle = _funding_bundle()
    expected_interval = (
        MONOTONIC_NS + 120_000_000_000,
        MONOTONIC_NS + 180_020_000_000,
    )

    assert bundle.boundary_mapping() == {
        symbol: (expected_interval,) for symbol in SYMBOLS
    }
    assert bundle.coverage_mapping() == {
        symbol: (
            MONOTONIC_NS + 60_000_000_000,
            MONOTONIC_NS + 240_020_000_000,
        )
        for symbol in SYMBOLS
    }
    assert bundle.evidence.binds(
        round74_funding_schedule_evidence_claims(
            funding_boundary_intervals_monotonic_ns=(
                bundle.boundary_mapping()
            ),
            funding_schedule_coverage_monotonic_ns=(
                bundle.coverage_mapping()
            ),
        )
    )


def test_clock_probes_load_only_hash_verified_time_frames() -> None:
    connection, audit = _clock_capture_fixture()

    probes = load_round74_binance_clock_probes(
        connection,
        run_id=audit.run_id,
        capture_audit=audit,
    )

    assert len(probes) == 6
    assert probes[0].exchange_time_ms == EXCHANGE_MS
    assert probes[-1].request_started_monotonic_ns == (
        MONOTONIC_NS + 300_000_000_000
    )
    assert len({probe.capture_audit_sha256 for probe in probes}) == 1


def test_clock_probe_loader_rejects_tampered_selected_frame() -> None:
    connection, audit = _clock_capture_fixture()
    tampered = list(connection.frame_rows[2])
    tampered[5] = "d" * 64
    connection.frame_rows[2] = tuple(tampered)

    with pytest.raises(ValueError, match="compressed frame"):
        load_round74_binance_clock_probes(
            connection,
            run_id=audit.run_id,
            capture_audit=audit,
        )


def test_funding_at_clock_probe_trims_ambiguous_coverage_edge() -> None:
    bundle = _funding_bundle(
        payloads=_funding_payloads(EXCHANGE_MS + 60_000)
    )

    assert bundle.boundary_mapping() == {symbol: () for symbol in SYMBOLS}
    assert bundle.coverage_mapping() == {
        symbol: (
            MONOTONIC_NS + 120_020_000_001,
            MONOTONIC_NS + 240_020_000_000,
        )
        for symbol in SYMBOLS
    }


def test_funding_evidence_rejects_full_page_and_nonascending_rows() -> None:
    with pytest.raises(ValueError, match="may be truncated"):
        _funding_bundle(limit=1)

    payloads = _funding_payloads()
    payloads["BTCUSDT"] = [
        {
            **payloads["BTCUSDT"][0],
            "fundingTime": EXCHANGE_MS + 150_001,
        },
        payloads["BTCUSDT"][0],
    ]
    with pytest.raises(ValueError, match="order"):
        _funding_bundle(payloads=payloads)

    payloads = _funding_payloads()
    payloads["ETHUSDT"][0]["fundingTime"] = str(
        payloads["ETHUSDT"][0]["fundingTime"]
    )
    with pytest.raises(ValueError, match="funding time"):
        _funding_bundle(payloads=payloads)


def test_funding_evidence_rejects_clock_gap_and_query_undercoverage() -> None:
    probes = list(_clock_probes())
    probes[3] = replace(
        probes[3],
        request_started_monotonic_ns=(
            probes[2].request_started_monotonic_ns
            + ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS
            + 1
        ),
        received_monotonic_ns=(
            probes[2].request_started_monotonic_ns
            + ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS
            + 20_000_001
        ),
    )
    with pytest.raises(ValueError, match="clock probe order"):
        _funding_bundle(probes=tuple(probes))

    with pytest.raises(ValueError, match="does not cover capture clocks"):
        build_round74_funding_evidence(
            payload_by_symbol=_funding_payloads(),
            environment="binance_usdm_mainnet",
            observed_wall_ns=WALL_NS,
            start_time_ms=EXCHANGE_MS + 1,
            end_time_ms=EXCHANGE_MS + 3_600_000,
            limit=1000,
            clock_probes=_clock_probes(),
        )
