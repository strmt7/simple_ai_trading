from __future__ import annotations

from dataclasses import replace

import pytest

from simple_ai_trading.impact_absorption_event_evidence import (
    ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS,
    Round74BinanceClockProbe,
    build_round74_commission_evidence,
    build_round74_funding_evidence,
)
from simple_ai_trading.impact_absorption_event_targets import (
    round74_commission_evidence_claims,
    round74_funding_schedule_evidence_claims,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WALL_NS = 1_784_000_000_000_000_000
MONOTONIC_NS = 1_000_000_000_000
EXCHANGE_MS = 1_784_000_000_000
CAPTURE_CONTRACT_SHA256 = "a" * 64


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


def _clock_probes(count: int = 6) -> tuple[Round74BinanceClockProbe, ...]:
    probes = []
    for index in range(count):
        started = MONOTONIC_NS + index * 60_000_000_000
        probes.append(
            Round74BinanceClockProbe(
                capture_run_id="round74-contract-test",
                capture_contract_sha256=CAPTURE_CONTRACT_SHA256,
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
