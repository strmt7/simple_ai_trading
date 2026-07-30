from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
)
from simple_ai_trading.polymarket_round20_capture import (
    _LaneTimeline,
    _intersection_milliseconds,
    _unhealthy_intervals,
    create_round20_capture_manifest,
    create_round20_recorder,
    evaluate_round20_capture,
    validate_round20_capture_manifest,
    validate_round20_qualification,
)
from simple_ai_trading.polymarket_round20_contract import (
    POLYMARKET_ROUND20_CONTRACT_SHA256,
    POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
    PolymarketRound20Program,
)


def _manifest() -> dict[str, object]:
    files = {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in (
            "docs/model-research/polymarket/"
            "round-020-independent-redundant-corpus-contract-v1.json",
            "src/simple_ai_trading/polymarket_recorder.py",
            "src/simple_ai_trading/polymarket_redundant_union.py",
            "src/simple_ai_trading/polymarket_round20_capture.py",
            "src/simple_ai_trading/polymarket_round20_contract.py",
            "tests/test_polymarket_recorder.py",
            "tests/test_polymarket_redundant_union.py",
            "tests/test_polymarket_round20_capture.py",
            "tests/test_polymarket_round20_contract.py",
            "tests/test_polymarket_round20_recorder.py",
            "tools/qualify_round20_capture.py",
        )
    }
    return create_round20_capture_manifest(
        run_id="a" * 32,
        created_at_ms=1_800_000_000_000,
        repository_commit_oid="b" * 40,
        repository_tree_oid="c" * 40,
        repository_file_sha256=files,
    )


def _rehash(value: dict[str, object], hash_name: str) -> dict[str, object]:
    body = dict(value)
    body.pop(hash_name, None)
    body[hash_name] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round20_capture_manifest_is_independent_attested_and_non_authoritative() -> None:
    manifest = validate_round20_capture_manifest(_manifest())

    assert manifest["round20_contract_sha256"] == POLYMARKET_ROUND20_CONTRACT_SHA256
    assert manifest["capture_duration_seconds"] == 1200
    assert manifest["required_streams"] == ["clob_market", "polymarket_rtds"]
    assert manifest["required_clob_lanes"] == ["clob-a", "clob-b"]
    assert manifest["required_rtds_topics"] == ["crypto_prices_chainlink"]
    assert manifest["optional_predictor_sources_captured"] == []
    assert manifest["binance_credentials_used"] is False
    assert manifest["binance_execution_connected"] is False
    assert manifest["model_data_eligible"] is False
    assert manifest["live_trading_authority"] is False


def test_round20_capture_manifest_rejects_rehashed_scope_or_authority_drift() -> None:
    for key, value in (
        ("required_streams", ["binance_spot", "clob_market", "polymarket_rtds"]),
        ("required_clob_lanes", ["clob-a"]),
        ("live_trading_authority", True),
    ):
        changed = _manifest()
        changed[key] = value
        with pytest.raises(ValueError, match="manifest differs"):
            validate_round20_capture_manifest(_rehash(changed, "manifest_sha256"))


def test_round20_recorder_factory_is_polymarket_core_only(tmp_path: Path) -> None:
    recorder = create_round20_recorder(tmp_path / "capture.duckdb")

    assert recorder.required_streams == ("clob_market", "polymarket_rtds")
    assert recorder.clob_lane_ids == ("clob-a", "clob-b")
    assert recorder.rtds_topics == ("crypto_prices_chainlink",)
    assert recorder.include_binance_spot is False
    assert recorder.include_binance_futures is False


def test_joint_unhealthy_time_uses_overlap_not_sum() -> None:
    first = _LaneTimeline(
        event_wall_ms=(1_000, 15_000),
        gaps=(),
    )
    second = _LaneTimeline(
        event_wall_ms=(2_000, 14_000),
        gaps=(),
    )
    first_unhealthy = _unhealthy_intervals(
        first,
        started_at_ms=0,
        ended_at_ms=20_000,
    )
    second_unhealthy = _unhealthy_intervals(
        second,
        started_at_ms=0,
        ended_at_ms=20_000,
    )

    assert first_unhealthy == ((0, 1_000), (11_000, 15_000))
    assert second_unhealthy == ((0, 2_000), (12_000, 14_000))
    assert _intersection_milliseconds(first_unhealthy, second_unhealthy) == 3_000


def test_round20_qualification_validator_rejects_rehashed_authority_drift() -> None:
    audit_body: dict[str, object] = {
        "schema_version": "polymarket-redundant-union-audit-v1",
        "pairing_window_ms": 2000,
        "union_event_count": 1,
        "shared_event_count": 1,
        "single_lane_event_count": 0,
        "lane_event_counts": {"clob-a": 1, "clob-b": 1},
        "lane_coverage_fraction": {"clob-a": 1.0, "clob-b": 1.0},
        "shared_fraction": 1.0,
        "event_type_counts": {"book": 1},
        "receipt_difference_ms": {
            "median": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
        },
        "maximum_pending_events_observed": 1,
        "terminal_pending_event_count": 0,
    }
    audit = _rehash(audit_body, "audit_sha256")
    gates = {
        "duration_complete": True,
        "terminal_status_usable": True,
        "zero_integrity_errors": True,
        "zero_recorder_errors": True,
        "exact_clob_receipt_count_reconciled": True,
        "both_clob_lanes_observed": True,
        "both_clob_lanes_connected": True,
        "multiple_market_conditions_observed": True,
        "minimum_lane_coverage": True,
        "minimum_shared_fraction": True,
        "joint_unhealthy_within_limit": True,
        "chainlink_rtds_observed": True,
        "union_count_reconciled": True,
        "union_pending_bound_preserved": True,
    }
    payload: dict[str, object] = {
        "schema_version": "polymarket-round20-capture-qualification-v1",
        "round20_contract_sha256": POLYMARKET_ROUND20_CONTRACT_SHA256,
        "parent_round19_result_sha256": (
            "61a7a6fe2cebd3ddc8ba6d4f59c52d6c19b91fe895353fda1bb066e86ecbc5be"
        ),
        "capture_manifest_sha256": "b" * 64,
        "run_id": "c" * 32,
        "recorder_report_sha256": "d" * 64,
        "started_at_ms": 1,
        "ended_at_ms": 1_200_001,
        "duration_seconds": 1200.0,
        "recorder_status": "complete",
        "database_bytes": 1,
        "reported_clob_receipts": 1,
        "decoded_clob_receipts": 1,
        "rtds_chainlink_receipts": 1,
        "market_condition_count": 4,
        "emitted_union_events": 1,
        "lane_connection_counts": {"clob-a": 1, "clob-b": 1},
        "lane_gap_counts": {"clob-a": 0, "clob-b": 0},
        "joint_unhealthy_ms": 0,
        "integrity_errors": [],
        "recorder_errors": [],
        "union_event_chain_sha256": "e" * 64,
        "union_audit": audit,
        "gates": gates,
        "qualified": True,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    valid = validate_round20_qualification(_rehash(payload, "result_sha256"))
    assert valid["qualified"] is True

    changed = dict(valid)
    changed["live_trading_authority"] = True
    with pytest.raises(ValueError, match="result differs"):
        validate_round20_qualification(_rehash(changed, "result_sha256"))


def _market(index: int, started_at_ms: int) -> PolymarketFiveMinuteMarket:
    condition_id = "0x" + f"{index + 1:x}" * 64
    up_token = str(index + 1) * 40
    down_token = str(index + 5) * 40
    gamma = json.dumps(
        {"conditionId": condition_id, "tokens": [up_token, down_token]},
        separators=(",", ":"),
        sort_keys=True,
    )
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id=f"market-{index}",
        condition_id=condition_id,
        slug=f"btc-updown-5m-{index}",
        question="Bitcoin Up or Down",
        event_start_ms=started_at_ms + index * 300_000,
        end_ms=started_at_ms + (index + 1) * 300_000,
        up_token_id=up_token,
        down_token_id=down_token,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.2"),
        ),
        liquidity_quote=Decimal("10000"),
        volume_quote=Decimal("20000"),
        resolution_source="https://data.chain.link/streams/btc-usd",
        gamma_payload_sha256=hashlib.sha256(gamma.encode("ascii")).hexdigest(),
        gamma_payload_json=gamma,
    )


def _market_evidence(
    market: PolymarketFiveMinuteMarket,
    observed_at_ms: int,
) -> MarketEvidence:
    clob = json.dumps(
        {"c": market.condition_id, "t": list(market.token_ids)},
        separators=(",", ":"),
        sort_keys=True,
    )
    fee = '{"base_fee":1000}'
    return MarketEvidence(
        market=market,
        observed_wall_ms=observed_at_ms,
        observed_monotonic_ns=observed_at_ms * 1_000_000,
        clob_info_json=clob,
        clob_info_sha256=hashlib.sha256(clob.encode("ascii")).hexdigest(),
        up_fee_rate_json=fee,
        up_fee_rate_sha256=hashlib.sha256(fee.encode("ascii")).hexdigest(),
        down_fee_rate_json=fee,
        down_fee_rate_sha256=hashlib.sha256(fee.encode("ascii")).hexdigest(),
        maker_base_fee=1000,
        taker_base_fee=1000,
        taker_order_delay_enabled=True,
        minimum_order_age_seconds=0,
    )


def test_round20_terminal_evaluator_rebuilds_real_store_frames_once(
    tmp_path: Path,
) -> None:
    database = tmp_path / "round20-integration.duckdb"
    started_at_ms = 1_800_000_000_000
    run_id = "f" * 32
    manifest = _manifest()
    manifest["run_id"] = run_id
    manifest["created_at_ms"] = started_at_ms
    manifest = _rehash(manifest, "manifest_sha256")
    messages: list[RawStreamMessage] = []
    sequence = {"clob-a": 0, "clob-b": 0}
    for offset_seconds in range(0, 1_200, 5):
        event = json.dumps(
            {
                "asset_id": "1" * 40,
                "event_type": "best_bid_ask",
                "market": "0x" + "1" * 64,
                "timestamp": str(started_at_ms + offset_seconds * 1_000),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        for lane, receipt_offset_ms in (("clob-a", 100), ("clob-b", 110)):
            sequence[lane] += 1
            received_wall_ms = (
                started_at_ms + offset_seconds * 1_000 + receipt_offset_ms
            )
            messages.append(
                RawStreamMessage(
                    stream="clob_market",
                    connection_id=f"{lane}:" + lane[-1] * 32,
                    sequence_number=sequence[lane],
                    received_wall_ms=received_wall_ms,
                    received_monotonic_ns=received_wall_ms * 1_000_000,
                    raw_text=event,
                )
            )
    messages.append(
        RawStreamMessage(
            stream="polymarket_rtds",
            connection_id="rtds:chainlink:btc:" + "c" * 32,
            sequence_number=1,
            received_wall_ms=started_at_ms + 1_000,
            received_monotonic_ns=(started_at_ms + 1_000) * 1_000_000,
            raw_text=json.dumps(
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                    "timestamp": started_at_ms + 1_000,
                    "payload": {
                        "symbol": "btc/usd",
                        "timestamp": started_at_ms + 1_000,
                        "value": "60000",
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    )
    messages.sort(key=lambda message: message.received_monotonic_ns)
    with PolymarketEvidenceStore(database) as store:
        store.start_run(
            run_id,
            started_at_ms,
            preregistration_manifest=manifest,
        )
        for index in range(4):
            market = _market(index, started_at_ms)
            store.record_market_evidence(
                run_id,
                _market_evidence(market, started_at_ms + index),
            )
        store.append_messages(run_id, messages)
        report = store.finish_run(
            run_id,
            started_at_ms=started_at_ms,
            ended_at_ms=started_at_ms + 1_200_000,
            database=str(database),
            errors=(),
        )
    program = PolymarketRound20Program(
        contract_sha256=POLYMARKET_ROUND20_CONTRACT_SHA256,
        parent_result_sha256=POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
        capture_unit_seconds=1_200,
        total_capture_units=2_160,
        pairing_window_ms=2_000,
        maximum_joint_unhealthy_ms=2_000,
    )
    result = evaluate_round20_capture(
        database=database,
        report=report,
        program=program,
        capture_manifest_sha256=str(manifest["manifest_sha256"]),
    )

    assert result["qualified"] is True
    assert result["reported_clob_receipts"] == 480
    assert result["decoded_clob_receipts"] == 480
    assert result["emitted_union_events"] == 240
    assert result["rtds_chainlink_receipts"] == 1
    assert result["joint_unhealthy_ms"] == 0
    assert result["union_audit"]["shared_fraction"] == 1.0
    assert result["model_data_eligible"] is False
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False
