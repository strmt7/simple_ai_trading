from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round25_materialization as materialization
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading.polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_END_MS,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    POLYMARKET_ROUND25_START_MS,
    build_round25_active_segment_manifest,
    load_round25_active_campaign_plan,
)
from simple_ai_trading.polymarket_round25_materialization import (
    Round25ReceiptCondition,
    Round25ReceiptMaterializerObserver,
    load_round25_receipt_conditions,
    materialize_round25_round24_core,
    round24_role_for_event_start,
)
from simple_ai_trading.polymarket_round25_terminal import (
    build_round25_terminal_transport_manifest,
)


EVENT_START_MS = POLYMARKET_ROUND25_START_MS + 300_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
RUN_ID = "3" * 32
ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-core-campaign-plan-publication-v2-2026-08-10.json"
)
PARTITIONS = (
    {
        "role": "train",
        "start_ms": POLYMARKET_ROUND25_START_MS,
        "end_ms": 1787441400000,
    },
    {
        "role": "tune_calibration",
        "start_ms": 1787443200000,
        "end_ms": 1787745600000,
    },
    {
        "role": "tune_selection",
        "start_ms": 1787745600000,
        "end_ms": 1788046800000,
    },
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _write_hashed(path: Path, body: dict[str, object]) -> None:
    value = {**body, "artifact_sha256": _canonical_sha256(body)}
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _condition() -> Round25ReceiptCondition:
    return Round25ReceiptCondition(
        run_id=RUN_ID,
        segment_index=0,
        snapshot_sha256="4" * 64,
        snapshot_observed_wall_ms=EVENT_START_MS - 10_000,
        condition_id=CONDITION_ID,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        role="train",
    ).validated()


def _market_evidence(*, resolution_source: str) -> MarketEvidence:
    start = datetime.fromtimestamp(EVENT_START_MS / 1_000, tz=UTC)
    end = datetime.fromtimestamp((EVENT_START_MS + 300_000) / 1_000, tz=UTC)
    payload: dict[str, object] = {
        "id": "round25-materialization-BTC",
        "question": "Bitcoin Up or Down",
        "conditionId": CONDITION_ID,
        "slug": f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        "eventStartTime": start.isoformat().replace("+00:00", "Z"),
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([UP_TOKEN, DOWN_TOKEN]),
        "outcomes": '["Up", "Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000.5,
        "volumeNum": 50_000.25,
        "resolutionSource": resolution_source,
    }
    if resolution_source == POLYMARKET_ROUND25_RESOLUTION_SOURCE:
        payload.update(
            {
                "cryptoMarketConfig": {
                    "asset": "btc",
                    "duration": "5m",
                    "id": "btc-5m-twap-30",
                    "twapEnabled": True,
                    "twapLookbackSeconds": 30,
                },
                "cryptoMarketConfigId": "btc-5m-twap-30",
            }
        )
    market = parse_polymarket_five_minute_market(payload)
    clob = json.dumps(
        {"condition": market.condition_id, "tokens": list(market.token_ids)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    fee = '{"base_fee":1000}'
    return MarketEvidence(
        market=market,
        observed_wall_ms=EVENT_START_MS - 10_000,
        observed_monotonic_ns=100,
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


def _metadata_database(path: Path, *, resolution_source: str) -> None:
    with PolymarketEvidenceStore(path) as store:
        store.start_run(RUN_ID, POLYMARKET_ROUND25_START_MS)
        store.record_market_evidence(
            RUN_ID,
            _market_evidence(resolution_source=resolution_source),
        )
        store.connect().execute(
            "UPDATE polymarket_recorder_run SET status = 'complete' WHERE run_id = ?",
            [RUN_ID],
        )


def _chainlink(offset_ms: int) -> dict[str, object]:
    sequence = offset_ms // 1_000 + 1
    return {
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": EVENT_START_MS + offset_ms,
        "payload": {
            "symbol": "btc/usd",
            "timestamp": EVENT_START_MS + offset_ms,
            "value": format(60_000.0 + sequence * 0.05, ".8f"),
        },
    }


def _book(token: str, *, offset_ms: int, bid: str, ask: str) -> dict[str, object]:
    return {
        "event_type": "book",
        "market": CONDITION_ID,
        "asset_id": token,
        "timestamp": str(EVENT_START_MS + offset_ms),
        "hash": hashlib.sha256(
            f"{token}:{offset_ms}:{bid}:{ask}".encode("ascii")
        ).hexdigest(),
        "bids": [
            {"price": bid, "size": "8"},
            {"price": format(float(bid) - 0.01, ".2f"), "size": "7"},
        ],
        "asks": [
            {"price": ask, "size": "9"},
            {"price": format(float(ask) + 0.01, ".2f"), "size": "6"},
        ],
    }


def _messages(*, extra_future_book: bool = False) -> tuple[RawStreamMessage, ...]:
    pending: list[tuple[int, str, dict[str, object]]] = []
    for offset_ms in range(0, 46_000, 1_000):
        pending.append(
            (EVENT_START_MS + offset_ms + 10, "polymarket_rtds", _chainlink(offset_ms))
        )
    for offset_ms in range(29_000, 46_000, 1_000):
        pending.extend(
            (
                (
                    EVENT_START_MS + offset_ms + 20,
                    "clob_market",
                    _book(UP_TOKEN, offset_ms=offset_ms, bid="0.49", ask="0.51"),
                ),
                (
                    EVENT_START_MS + offset_ms + 21,
                    "clob_market",
                    _book(
                        DOWN_TOKEN,
                        offset_ms=offset_ms + 1,
                        bid="0.48",
                        ask="0.52",
                    ),
                ),
            )
        )
    if extra_future_book:
        pending.append(
            (
                EVENT_START_MS + 35_500,
                "clob_market",
                _book(UP_TOKEN, offset_ms=35_480, bid="0.59", ask="0.61"),
            )
        )
    sequence = {"clob_market": 0, "polymarket_rtds": 0}
    output: list[RawStreamMessage] = []
    for ordinal, (wall_ms, stream, payload) in enumerate(sorted(pending), start=1):
        sequence[stream] += 1
        output.append(
            RawStreamMessage(
                stream=stream,
                connection_id=(
                    "clob:" + "a" * 32
                    if stream == "clob_market"
                    else "polymarket-rtds:" + "b" * 32
                ),
                sequence_number=sequence[stream],
                received_wall_ms=wall_ms,
                received_monotonic_ns=wall_ms * 1_000_000 + ordinal,
                raw_text=json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
    return tuple(output)


def _run_observer(
    *,
    gaps: tuple[StreamGap, ...] = (),
    extra_future_book: bool = False,
) -> Round25ReceiptMaterializerObserver:
    observer = Round25ReceiptMaterializerObserver((_condition(),))
    segment = {
        "run_id": RUN_ID,
        "started_at_ms": POLYMARKET_ROUND25_START_MS,
        "ended_at_ms": EVENT_START_MS + 600_000,
    }
    observer.start_run(segment, gaps)
    for message in _messages(extra_future_book=extra_future_book):
        observer.observe_message(segment, message)
    observer.finish_run(segment)
    return observer


def test_round25_explicit_partition_policy_preserves_purge() -> None:
    assert (
        round24_role_for_event_start(POLYMARKET_ROUND25_START_MS, PARTITIONS) == "train"
    )
    assert round24_role_for_event_start(1787441400000, PARTITIONS) is None
    assert round24_role_for_event_start(1787443199999, PARTITIONS) is None
    assert round24_role_for_event_start(1787443200000, PARTITIONS) == "tune_calibration"
    assert round24_role_for_event_start(1787745600000, PARTITIONS) == "tune_selection"
    assert round24_role_for_event_start(1788046800000, PARTITIONS) is None


def test_round25_single_lane_materializer_builds_causal_core_snapshots() -> None:
    observer = _run_observer()

    assert observer.snapshots
    assert observer.rejection_counts == {}
    assert all(
        item.maximum_receipt_ms <= item.decision_time_ms for item in observer.snapshots
    )
    assert {item.condition_id for item in observer.snapshots} == {CONDITION_ID}
    assert min(item.decision_time_ms for item in observer.snapshots) >= (
        EVENT_START_MS + 30_000
    )


def test_round25_future_receipt_cannot_change_prior_snapshot() -> None:
    baseline = _run_observer()
    with_future = _run_observer(extra_future_book=True)
    decision = EVENT_START_MS + 35_000
    baseline_snapshot = next(
        item for item in baseline.snapshots if item.decision_time_ms == decision
    )
    future_snapshot = next(
        item for item in with_future.snapshots if item.decision_time_ms == decision
    )

    assert future_snapshot.values == baseline_snapshot.values
    assert future_snapshot.market_prior_probability == (
        baseline_snapshot.market_prior_probability
    )
    assert future_snapshot.source_chain_sha256 == baseline_snapshot.source_chain_sha256


def test_round25_gap_rejects_entire_affected_condition() -> None:
    observer = _run_observer(
        gaps=(
            StreamGap(
                stream="clob_market",
                connection_id="clob:" + "a" * 32,
                opened_at_ms=EVENT_START_MS + 35_000,
                reason="fixture_disconnect",
                last_sequence_number=12,
            ),
        )
    )

    assert observer.snapshots == []
    assert observer.rejection_counts == {"stream_gap:clob_market": 1}


def test_round25_metadata_loader_rejects_legacy_point_price_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = {
        "segments": [
            {
                "run_id": RUN_ID,
                "segment_index": 0,
                "status": "complete",
                "eligible_for_condition_rebuild": True,
            }
        ]
    }
    monkeypatch.setattr(
        materialization,
        "validate_round25_terminal_transport_manifest",
        lambda _value: transport,
    )
    exact_database = tmp_path / "exact.duckdb"
    _metadata_database(
        exact_database,
        resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    )

    conditions, counts = load_round25_receipt_conditions(
        database=exact_database,
        terminal_transport_manifest=transport,
        partitions=PARTITIONS,
    )

    assert [item.condition_id for item in conditions] == [CONDITION_ID]
    assert counts == {
        "source_condition_count": 1,
        "partition_condition_count": 1,
        "outside_partition_condition_count": 0,
    }

    legacy_database = tmp_path / "legacy.duckdb"
    _metadata_database(
        legacy_database,
        resolution_source="https://data.chain.link/streams/btc-usd",
    )
    with pytest.raises(ValueError, match="snapshot columns differ"):
        load_round25_receipt_conditions(
            database=legacy_database,
            terminal_transport_manifest=transport,
            partitions=PARTITIONS,
        )


def test_round25_terminal_database_materializes_in_one_exact_receipt_scan(
    tmp_path: Path,
) -> None:
    database = tmp_path / "round25-materialization.duckdb"
    state_root = tmp_path / "state"
    state_root.mkdir()
    plan = load_round25_active_campaign_plan(PLAN)
    ended_at_ms = EVENT_START_MS + 600_000
    manifest = build_round25_active_segment_manifest(
        plan,
        run_id=RUN_ID,
        created_at_ms=POLYMARKET_ROUND25_START_MS,
        capture_duration_seconds=(ended_at_ms - POLYMARKET_ROUND25_START_MS) // 1_000,
        segment_index=0,
    )
    (state_root / "segment-0000-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with PolymarketEvidenceStore(database) as store:
        store.start_run(
            RUN_ID,
            POLYMARKET_ROUND25_START_MS,
            preregistration_manifest=manifest,
        )
        store.record_market_evidence(
            RUN_ID,
            _market_evidence(resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE),
        )
        store.append_messages(RUN_ID, _messages())
        recorder_report = store.finish_run(
            RUN_ID,
            started_at_ms=POLYMARKET_ROUND25_START_MS,
            ended_at_ms=ended_at_ms,
            database=str(database),
            errors=(),
        )
    assert recorder_report.status == "complete"
    _write_hashed(
        state_root / "segment-0000-result.json",
        {
            "condition_admission_pending": True,
            "details": {
                "condition_count": len(recorder_report.conditions),
                "duration_seconds": recorder_report.duration_seconds,
                "ended_at_ms": recorder_report.ended_at_ms,
                "errors": list(recorder_report.errors),
                "integrity_errors": list(recorder_report.integrity_errors),
                "manifest_sha256": manifest["manifest_sha256"],
                "raw_message_count": recorder_report.raw_message_count,
                "report_sha256": recorder_report.report_sha256,
                "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
                "run_id": recorder_report.run_id,
                "started_at_ms": recorder_report.started_at_ms,
                "stream_counts": dict(recorder_report.stream_counts),
                "stream_gap_count": recorder_report.stream_gap_count,
            },
            "live_trading_authority": False,
            "model_data_eligible": False,
            "observed_at_ms": ended_at_ms + 1,
            "paper_trading_authority": False,
            "plan_sha256": plan.plan_sha256,
            "profitability_claim": False,
            "schema_version": POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION,
            "segment_index": 0,
            "status": recorder_report.status,
        },
    )
    _write_hashed(
        state_root / "campaign-state.json",
        {
            "condition_admission_pending": True,
            "live_trading_authority": False,
            "model_data_eligible": False,
            "paper_trading_authority": False,
            "plan_sha256": plan.plan_sha256,
            "profitability_claim": False,
            "schema_version": POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
            "status": "campaign_window_ended",
            "status_counts": {"complete": 1},
            "terminal_segment_count": 1,
        },
    )
    transport = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )

    with pytest.raises(RuntimeError, match="point-stream materialization is retired"):
        materialize_round25_round24_core(
            database=database,
            terminal_transport_manifest=transport,
            partitions=PARTITIONS,
            round24_specification_sha256="a" * 64,
            observed_at_ms=POLYMARKET_ROUND25_END_MS + 1,
        )
