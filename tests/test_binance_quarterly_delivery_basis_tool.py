from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_binance_quarterly_delivery_basis",
    ROOT / "tools" / "audit_binance_quarterly_delivery_basis.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
FIRST_DELIVERY_MS = 1_700_000_040_000
DELIVERY_STEP_MS = 7_776_000_000
AUDIT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-delivery-basis-audit-v1-2026-08-25.json"
)
ADJUDICATION = AUDIT.with_name(
    "binance-quarterly-delivery-basis-timestamp-adjudication-v1.json"
)
CARRY = AUDIT.with_name("binance-quarterly-carry-snapshot-v1-2026-08-25.json")
OFFICIAL_TIMING = AUDIT.with_name("binance-quarterly-delivery-time-semantics-v1.json")


def _deliveries(pair: str) -> list[dict[str, object]]:
    base = Decimal("100") if pair == "BTCUSDT" else Decimal("2000")
    return [
        {
            "deliveryTime": FIRST_DELIVERY_MS + index * DELIVERY_STEP_MS,
            "deliveryPrice": str(base + index),
        }
        for index in range(8)
    ]


def _klines(pair: str, start_ms: int) -> list[list[object]]:
    delivery_index = (start_ms - FIRST_DELIVERY_MS) // DELIVERY_STEP_MS
    base = (Decimal("100") if pair == "BTCUSDT" else Decimal("2000")) + Decimal(
        delivery_index
    )
    return [
        [
            start_ms + minute * 60_000,
            str(base),
            str(base * Decimal("1.001")),
            str(base * Decimal("0.999")),
            str(base * Decimal("1.0005")),
            "1",
            start_ms + (minute + 1) * 60_000 - 1,
        ]
        for minute in range(5)
    ]


class _Response:
    def __init__(
        self,
        payload: object,
        *,
        url: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.url = url
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.content = TOOL._canonical_json(payload).encode("ascii")

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, *, rate_limited: bool = False) -> None:
        self.rate_limited = rate_limited
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: int,
    ) -> _Response:
        assert timeout == 30
        self.calls.append((url, params))
        if self.rate_limited:
            return _Response(
                {},
                url=url,
                status_code=429,
                headers={"Retry-After": "19"},
            )
        if url == TOOL.FUTURES_URL:
            payload: object = _deliveries(str(params["pair"]))
        else:
            payload = _klines(str(params["symbol"]), int(params["startTime"]))
        return _Response(payload, url=url)


def test_run_uses_exact_eighteen_requests_and_never_accepts_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_640_900_000_000_000)
    session = _Session()
    ledger: list[dict[str, object]] = []

    report = TOOL.run(session=session, ledger=ledger)

    assert len(session.calls) == 18
    assert len(ledger) == 18
    assert [row["pair"] for row in report["pair_results"]] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert all(row["delivery_count"] == 8 for row in report["pair_results"])
    assert all(
        Decimal(row["minimum_low_mismatch_bips"]) == Decimal("-10.000")
        for row in report["pair_results"]
    )
    assert len(report["current_next_quarter_stress"]) == 6
    assert report["verdict"]["all_next_quarter_sizes_stress_positive"] is True
    assert report["verdict"]["accepted_edge"] is False
    assert report["safety"]["orders_placed"] is False


def test_rate_limit_is_retained_in_terminal_failure_receipt() -> None:
    ledger: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="without retry; Retry-After=19") as error:
        TOOL.run(session=_Session(rate_limited=True), ledger=ledger)

    receipt = TOOL._failure_payload(error=error.value, ledger=ledger)
    expected_hash = receipt.pop("result_sha256")

    assert len(ledger) == 1
    assert ledger[0]["status_code"] == 429
    assert ledger[0]["retry_after"] == "19"
    assert receipt["status"] == "terminal_failure_without_retry"
    assert receipt["authority"]["accepted_edge"] is False
    assert (
        hashlib.sha256(TOOL._canonical_json(receipt).encode("ascii")).hexdigest()
        == expected_hash
    )


def test_delivery_selection_rejects_insufficient_and_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="lacks the frozen"):
        TOOL._deliveries(
            _deliveries("BTCUSDT")[:-1],
            cutoff_ms=1_787_640_808_532,
            count=8,
        )

    conflict = _deliveries("BTCUSDT")
    conflict.append({**conflict[0], "deliveryPrice": "999"})
    with pytest.raises(ValueError, match="conflicting"):
        TOOL._deliveries(
            conflict,
            cutoff_ms=1_787_640_808_532,
            count=8,
        )


def test_canonical_request_payload_hash_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_640_900_000_000_000)
    ledger: list[dict[str, object]] = []
    TOOL._get(
        _Session(),
        TOOL.FUTURES_URL,
        params={"pair": "BTCUSDT"},
        ledger=ledger,
    )

    entry = ledger[0]
    assert (
        hashlib.sha256(
            TOOL._canonical_json(entry["decoded_payload"]).encode("ascii")
        ).hexdigest()
        == entry["canonical_payload_sha256"]
    )


def test_live_audit_hash_sources_and_stress_reconstruct_as_historical_record() -> None:
    report = json.loads(AUDIT.read_text(encoding="ascii"))
    expected_hash = report.pop("result_sha256")

    assert (
        hashlib.sha256(TOOL._canonical_json(report).encode("ascii")).hexdigest()
        == expected_hash
    )
    source = report["source_contract"]
    contract_path = ROOT / source["contract_path"]
    assert (
        hashlib.sha256(contract_path.read_bytes()).hexdigest()
        == source["contract_file_sha256"]
    )
    implementation = source["implementation"]
    for path_field, hash_field in (
        ("tool_path", "tool_sha256"),
        ("module_path", "module_sha256"),
    ):
        assert (
            hashlib.sha256((ROOT / implementation[path_field]).read_bytes()).hexdigest()
            == implementation[hash_field]
        )
    ledger = source["request_ledger"]
    assert len(ledger) == 18
    for entry in ledger:
        assert (
            hashlib.sha256(
                TOOL._canonical_json(entry["decoded_payload"]).encode("ascii")
            ).hexdigest()
            == entry["canonical_payload_sha256"]
        )
    worst = {
        row["pair"]: Decimal(row["minimum_low_mismatch_bips"])
        for row in report["pair_results"]
    }
    for row in report["current_next_quarter_stress"]:
        reconstructed = TOOL.stressed_after_hurdle_basis_bips(
            after_hurdle_basis_bips=Decimal(row["after_35_bps_hurdle_basis_bips"]),
            worst_observed_mismatch_bips=worst[row["pair"]],
        )
        assert str(reconstructed) == row["stressed_basis_bips"]
        assert (reconstructed > 0) is row["stress_positive"]
    assert report["verdict"]["all_next_quarter_sizes_stress_positive"] is False
    assert report["verdict"]["accepted_edge"] is False


def test_timestamp_adjudication_reconstructs_and_invalidates_prior_rejection() -> None:
    adjudication = json.loads(ADJUDICATION.read_text(encoding="ascii"))
    expected_hash = adjudication.pop("result_sha256")
    audit = json.loads(AUDIT.read_text(encoding="ascii"))
    carry = json.loads(CARRY.read_text(encoding="ascii"))

    assert (
        hashlib.sha256(TOOL._canonical_json(adjudication).encode("ascii")).hexdigest()
        == expected_hash
    )
    assert (
        hashlib.sha256(AUDIT.read_bytes()).hexdigest()
        == adjudication["adjudicated_claim"]["artifact_raw_file_sha256"]
    )
    assert (
        audit["result_sha256"]
        == adjudication["adjudicated_claim"]["artifact_result_sha256"]
    )
    assert (
        hashlib.sha256(CARRY.read_bytes()).hexdigest()
        == adjudication["current_exchange_catalog_evidence"]["artifact_raw_file_sha256"]
    )
    assert (
        carry["result_sha256"]
        == adjudication["current_exchange_catalog_evidence"]["artifact_result_sha256"]
    )

    historical_times = {
        observation["delivery_time_ms"]
        for pair in audit["pair_results"]
        for observation in pair["observations"]
    }
    current_contracts = carry["source_contract"]["selected_contracts"]
    assert historical_times == set(
        adjudication["historical_settlement_endpoint_evidence"][
            "distinct_delivery_times_ms"
        ]
    )
    assert {timestamp % 86_400_000 for timestamp in historical_times} == {0}
    assert {
        contract["delivery_time_ms"] % 86_400_000 for contract in current_contracts
    } == {28_800_000}
    assert [
        {
            "delivery_time_ms": contract["delivery_time_ms"],
            "symbol": contract["symbol"],
        }
        for contract in current_contracts
    ] == adjudication["current_exchange_catalog_evidence"]["selected_contracts"]
    assert adjudication["verdict"] == {
        "new_prices_requested": False,
        "original_one_use_artifact_retained_for_provenance": True,
        "previous_mismatch_values_authoritative": False,
        "previous_stressed_current_basis_authoritative": False,
        "resampling_authorized": False,
        "timestamp_substitution_authorized": False,
    }
    assert adjudication["authority"]["accepted_edge"] is False


def test_official_timing_source_resolves_normal_schedule_without_assuming_cutoff() -> (
    None
):
    timing = json.loads(OFFICIAL_TIMING.read_text(encoding="ascii"))
    expected_hash = timing.pop("result_sha256")
    audit = json.loads(AUDIT.read_text(encoding="ascii"))

    assert (
        hashlib.sha256(TOOL._canonical_json(timing).encode("ascii")).hexdigest()
        == expected_hash
    )
    assert timing["official_rules"] == {
        "delivery_method": "cash settlement",
        "delivery_time_utc": "last Friday of each calendar quarter at 08:00:00",
        "extreme_condition_postponement_possible": True,
        "quarterly_contract_families": [
            "COIN-Margined Futures",
            "USD-Margined Futures",
        ],
        "reduce_only_window_before_delivery_minutes": 10,
        "settlement_price_window_end_utc": "08:00:00",
        "settlement_price_window_minutes": 30,
        "settlement_price_window_start_utc": "07:30:00",
    }
    historical_times = {
        observation["delivery_time_ms"]
        for pair in audit["pair_results"]
        for observation in pair["observations"]
    }
    scheduled_times = {
        timestamp
        + timing["historical_timestamp_resolution"][
            "normal_scheduled_delivery_intraday_offset_ms"
        ]
        for timestamp in historical_times
    }
    assert len(scheduled_times) == 8
    assert {timestamp % 86_400_000 for timestamp in scheduled_times} == {28_800_000}
    assert (
        timing["historical_timestamp_resolution"][
            "actual_delivery_postponement_checked"
        ]
        is False
    )
    assert timing["authority"]["accepted_edge"] is False
