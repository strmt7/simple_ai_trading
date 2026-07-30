from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading import polymarket_round18_transport as transport
from simple_ai_trading.polymarket_round18_transport import (
    POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256,
    RedundantClobLaneEvidence,
    evaluate_round18_transport_qualification,
    load_round18_transport_contract,
    validate_round18_transport_result,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-018-redundant-clob-transport-qualification-v1.json"
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _lane(lane_id: str, *, digest: str = "a" * 64) -> RedundantClobLaneEvidence:
    lane = RedundantClobLaneEvidence(lane_id, 1_000_000)
    lane.event_counts[digest] = 25_000
    lane.first_receipt_monotonic_ns[digest] = (
        1_000_000 if lane_id == "clob-a" else 2_000_000
    )
    lane.event_type_counts["price_change"] = 25_000
    lane.frame_count = 25_000
    lane.market_event_count = 25_000
    lane.connection_count = 1
    lane.connected_seconds = 600.0
    lane.last_market_event_monotonic = 600.0
    return lane


def _markets() -> list[dict[str, object]]:
    payload = {
        "asset": "BTC",
        "market_id": "1",
        "condition_id": "0x" + "1" * 64,
        "slug": "btc-updown-5m-1800000000",
        "event_start_ms": 1_800_000_000_000,
        "end_ms": 1_800_000_300_000,
        "token_ids": ["1" * 70, "2" * 70],
    }
    return [{**payload, "identity_sha256": _sha256(payload)}]


def test_round18_transport_contract_is_frozen_before_feed_access() -> None:
    raw = json.loads(CONTRACT.read_text(encoding="utf-8"))
    claimed = raw.pop("contract_sha256")

    assert _sha256(raw) == claimed
    assert claimed == POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256
    contract = load_round18_transport_contract(CONTRACT)
    assert contract["status"] == "preregistered_before_qualification_feed_access"
    assert contract["motivation"]["existing_campaign_or_gate_modified"] is False
    assert contract["scope"]["order_submission"] is False
    assert contract["transport"]["lane_count"] == 2
    assert (
        contract["interpretation"]["qualification_proves_venue_source_completeness"]
        is False
    )
    assert contract["authority"]["live_trading_authority"] is False


def test_round18_lane_evidence_bounds_and_classifies_frames() -> None:
    lane = RedundantClobLaneEvidence("clob-a", 1)
    lane.mark_connected(10.0)
    lane.record_frame("PONG", 10.1)
    lane.record_frame(
        json.dumps(
            [
                {
                    "event_type": "price_change",
                    "asset_id": "1",
                    "timestamp": "1800000000000",
                }
            ]
        ),
        10.2,
    )
    lane.record_frame("{invalid", 10.3)
    lane.record_frame('{"event_type":"book","asset_id":"1"}', 10.4)
    lane.mark_disconnected(11.0, reason="ConnectionClosedError:fixture")

    assert lane.pong_count == 1
    assert lane.market_event_count == 1
    assert lane.json_parse_error_count == 1
    assert lane.memory_bound_exceeded is True
    assert lane.transport_gap_count == 1
    assert lane.connected_seconds == 1.0


def test_round18_redundant_qualification_requires_both_lane_coverage() -> None:
    contract = load_round18_transport_contract(CONTRACT)
    monitor = transport._MonitorEvidence(10.0, 0.1)
    monitor.sample_count_after_warmup = 5_900
    monitor.observed_seconds_after_warmup = 590.0
    first = _lane("clob-a")
    second = _lane("clob-b")

    passed = evaluate_round18_transport_qualification(
        contract,
        market_identities=_markets(),
        lanes=(first, second),
        started_at_ms=1_800_000_000_000,
        ended_at_ms=1_800_000_600_000,
        monitor=monitor,
    )

    assert passed["qualified"] is True
    assert all(passed["gates"].values())
    assert passed["redundancy"]["counted_overlap_fraction"] == 1.0
    assert passed["venue_source_completeness_proven"] is False
    assert passed["model_data_eligible"] is False
    assert passed["live_trading_authority"] is False
    assert validate_round18_transport_result(passed) == passed

    tampered = json.loads(json.dumps(passed))
    tampered["model_data_eligible"] = True
    tampered.pop("result_sha256")
    tampered["result_sha256"] = _sha256(tampered)
    try:
        validate_round18_transport_result(tampered)
    except ValueError as exc:
        assert "terminal transport result differs" in str(exc)
    else:
        raise AssertionError("Round 18 accepted a rehashed authority escalation")

    second.event_counts = type(second.event_counts)({"b" * 64: 25_000})
    second.first_receipt_monotonic_ns = {"b" * 64: 2_000_000}
    failed = evaluate_round18_transport_qualification(
        contract,
        market_identities=_markets(),
        lanes=(first, second),
        started_at_ms=1_800_000_000_000,
        ended_at_ms=1_800_000_600_000,
        monitor=monitor,
    )

    assert failed["qualified"] is False
    assert failed["gates"]["minimum_event_coverage_each_lane"] is False
    assert failed["gates"]["minimum_counted_overlap"] is False
