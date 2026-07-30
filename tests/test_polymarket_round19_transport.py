from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from simple_ai_trading import polymarket_round19_transport as transport
from simple_ai_trading.polymarket_round18_transport import (
    RedundantClobLaneEvidence,
)
from simple_ai_trading.polymarket_round19_transport import (
    POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256,
    evaluate_round19_transport_qualification,
    load_round19_transport_contract,
    validate_round19_transport_result,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-019-rotating-redundant-clob-qualification-v1.json"
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


def _market(index: int, *, start_ms: int) -> SimpleNamespace:
    return SimpleNamespace(
        asset="BTC",
        market_id=str(index),
        condition_id="0x" + f"{index:064x}",
        slug=f"btc-updown-5m-{start_ms // 1000}",
        event_start_ms=start_ms,
        end_ms=start_ms + 300_000,
        token_ids=(f"{index * 2:070d}", f"{index * 2 + 1:070d}"),
    )


def _lane(lane_id: str, registry) -> transport._RotatingLane:
    evidence = RedundantClobLaneEvidence(lane_id, 1_000_000)
    evidence.event_counts["a" * 64] = 25_000
    evidence.first_receipt_monotonic_ns["a" * 64] = (
        1_000_000 if lane_id == "clob-a" else 2_000_000
    )
    evidence.event_type_counts["price_change"] = 25_000
    evidence.frame_count = 25_000
    evidence.market_event_count = 25_000
    evidence.connection_count = 1
    evidence.connected_seconds = 600.0
    evidence.last_market_event_monotonic = 600.0
    return transport._RotatingLane(
        evidence=evidence,
        applied_revision=registry.revision,
        subscribed_token_ids=registry.desired_token_ids,
        subscription_update_count=1,
    )


def test_round19_contract_freezes_only_subscription_rotation() -> None:
    raw = json.loads(CONTRACT.read_text(encoding="utf-8"))
    claimed = raw.pop("contract_sha256")

    assert _sha256(raw) == claimed
    assert claimed == POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256
    contract = load_round19_transport_contract(CONTRACT)
    assert contract["parent"]["round18_qualified"] is False
    assert contract["parent"]["mechanism_changed"] == (
        "rotate_discovered_current_and_next_market_subscriptions"
    )
    assert contract["rotation"]["gamma_discovery_interval_seconds"] == 30
    assert (
        contract["rotation"]["subscribe_additions_before_unsubscribe_removals"] is True
    )
    assert contract["authority"]["live_trading_authority"] is False


def test_round19_registry_is_append_only_and_applies_additions_first() -> None:
    registry = transport._RotatingRegistry()
    first = _market(1, start_ms=1_800_000_000_000)
    second = _market(2, start_ms=1_800_000_300_000)
    third = _market(3, start_ms=1_800_000_600_000)
    assert registry.update(
        (first, second),
        now_ms=1_800_000_100_000,
    )
    prior_tokens = registry.desired_token_ids
    assert registry.update(
        (second, third),
        now_ms=1_800_000_400_000,
    )
    assert len(registry.identities) == 3
    assert registry.revision == 2

    class _Socket:
        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []

        async def send(self, message: str) -> None:
            self.sent.append(json.loads(message))

    socket = _Socket()
    lane = transport._RotatingLane(
        RedundantClobLaneEvidence("clob-a", 100),
        applied_revision=1,
        subscribed_token_ids=prior_tokens,
    )
    asyncio.run(
        transport._apply_subscription(
            socket,
            lane,
            registry,
            custom_feature_enabled=True,
        )
    )

    assert [item["operation"] for item in socket.sent] == [
        "subscribe",
        "unsubscribe",
    ]
    assert lane.applied_revision == registry.revision
    assert lane.subscribed_token_ids == registry.desired_token_ids


def test_round19_terminal_result_requires_rotation_and_both_lanes() -> None:
    contract = load_round19_transport_contract(CONTRACT)
    registry = transport._RotatingRegistry()
    first = _market(1, start_ms=1_800_000_000_000)
    second = _market(2, start_ms=1_800_000_300_000)
    third = _market(3, start_ms=1_800_000_600_000)
    registry.update((first, second), now_ms=1_800_000_100_000)
    registry.update((second, third), now_ms=1_800_000_400_000)
    lanes = (_lane("clob-a", registry), _lane("clob-b", registry))
    monitor = transport._Monitor(10.0, 0.1)
    monitor.sample_count_after_warmup = 5_900
    monitor.observed_seconds_after_warmup = 590.0

    result = evaluate_round19_transport_qualification(
        contract,
        registry=registry,
        lanes=lanes,
        started_at_ms=1_800_000_000_000,
        ended_at_ms=1_800_000_600_000,
        monitor=monitor,
    )

    assert result["qualified"] is True
    assert all(result["gates"].values())
    assert result["rotation"]["final_revision"] == 2
    assert validate_round19_transport_result(result) == result
    assert result["live_trading_authority"] is False

    lanes[1].applied_revision = 1
    failed = evaluate_round19_transport_qualification(
        contract,
        registry=registry,
        lanes=lanes,
        started_at_ms=1_800_000_000_000,
        ended_at_ms=1_800_000_600_000,
        monitor=monitor,
    )
    assert failed["qualified"] is False
    assert failed["gates"]["both_lanes_applied_final_revision"] is False
