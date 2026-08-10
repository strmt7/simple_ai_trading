from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round25_terminal as round25_terminal
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
)
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_END_MS,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_START_MS,
    build_round25_segment_manifest,
    load_round25_campaign_plan,
)
from simple_ai_trading.polymarket_round25_terminal import (
    POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    audit_round25_terminal_receipts,
    build_round25_terminal_transport_manifest,
    load_round25_terminal_design,
    load_round25_terminal_receipt_audit,
    validate_round25_terminal_receipt_audit,
    validate_round25_terminal_transport_manifest,
    write_round25_terminal_receipt_audit,
)


ROOT = Path(__file__).parents[1]
PLAN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-core-campaign-plan-publication-2026-08-10.json"
)
RUN_ID = "1" * 32


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_hashed(path: Path, body: dict[str, object]) -> None:
    value = {**body, "artifact_sha256": _canonical_sha256(body)}
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _terminal_state_root(tmp_path: Path, *, status: str = "complete") -> Path:
    state_root = tmp_path / "state"
    state_root.mkdir()
    plan = load_round25_campaign_plan(PLAN)
    if status in {"complete", "degraded"}:
        gap_count = 0 if status == "complete" else 1
        manifest = build_round25_segment_manifest(
            plan,
            run_id=RUN_ID,
            created_at_ms=POLYMARKET_ROUND25_START_MS,
            capture_duration_seconds=600,
            segment_index=0,
        )
        (state_root / "segment-0000-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        details: dict[str, object] = {
            "condition_count": 1,
            "duration_seconds": 600.0,
            "ended_at_ms": POLYMARKET_ROUND25_START_MS + 600_000,
            "errors": [],
            "integrity_errors": [],
            "manifest_sha256": manifest["manifest_sha256"],
            "raw_message_count": 10,
            "report_sha256": "2" * 64,
            "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
            "run_id": RUN_ID,
            "started_at_ms": POLYMARKET_ROUND25_START_MS,
            "stream_counts": {"clob_market": 6, "polymarket_rtds": 4},
            "stream_gap_count": gap_count,
        }
    else:
        details = {"failure": "bounded failure", "failure_type": "RuntimeError"}
    _write_hashed(
        state_root / "segment-0000-result.json",
        {
            "condition_admission_pending": True,
            "details": details,
            "live_trading_authority": False,
            "model_data_eligible": False,
            "observed_at_ms": POLYMARKET_ROUND25_START_MS + 600_001,
            "paper_trading_authority": False,
            "plan_sha256": plan.plan_sha256,
            "profitability_claim": False,
            "schema_version": POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION,
            "segment_index": 0,
            "status": status,
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
            "schema_version": "polymarket-round25-twap-core-campaign-state-v1",
            "status": "campaign_window_ended",
            "status_counts": {status: 1},
            "terminal_segment_count": 1,
        },
    )
    return state_root


def _market_evidence(observed_at_ms: int) -> MarketEvidence:
    start = datetime.fromtimestamp(POLYMARKET_ROUND25_START_MS / 1_000, tz=UTC)
    end = datetime.fromtimestamp(
        (POLYMARKET_ROUND25_START_MS + 300_000) / 1_000,
        tz=UTC,
    )
    payload = {
        "id": "round25-market-BTC",
        "question": "Bitcoin Up or Down",
        "conditionId": "0x" + "7" * 64,
        "slug": f"btc-updown-5m-{POLYMARKET_ROUND25_START_MS // 1_000}",
        "eventStartTime": start.isoformat().replace("+00:00", "Z"),
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps(["7" * 40, "7" * 39 + "1"]),
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
        "resolutionSource": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "cryptoMarketConfig": {
            "asset": "btc",
            "duration": "5m",
            "id": "btc-5m-twap-30",
            "twapEnabled": True,
            "twapLookbackSeconds": 30,
        },
        "cryptoMarketConfigId": "btc-5m-twap-30",
    }
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
        observed_wall_ms=observed_at_ms,
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


def _public_messages(received_at_ms: int) -> list[RawStreamMessage]:
    clob = json.dumps(
        [
            {
                "asset_id": "7" * 40,
                "event_type": "book",
                "market": "0x" + "7" * 64,
                "timestamp": received_at_ms,
            }
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    rtds = json.dumps(
        {
            "payload": {
                "symbol": "btcusdt",
                "timestamp": received_at_ms,
                "value": "100000.0",
            },
            "timestamp": received_at_ms,
            "topic": "crypto_prices_chainlink",
            "type": "update",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        RawStreamMessage(
            stream="clob_market",
            connection_id="clob:" + "a" * 32,
            sequence_number=1,
            received_wall_ms=received_at_ms,
            received_monotonic_ns=1_000,
            raw_text=clob,
        ),
        RawStreamMessage(
            stream="polymarket_rtds",
            connection_id="polymarket-rtds:" + "b" * 32,
            sequence_number=1,
            received_wall_ms=received_at_ms + 1,
            received_monotonic_ns=2_000,
            raw_text=rtds,
        ),
    ]


def _terminal_database_fixture(tmp_path: Path) -> tuple[Path, Path]:
    database = tmp_path / "round25-terminal.duckdb"
    state_root = tmp_path / "database-state"
    state_root.mkdir()
    plan = load_round25_campaign_plan(PLAN)
    manifest = build_round25_segment_manifest(
        plan,
        run_id=RUN_ID,
        created_at_ms=POLYMARKET_ROUND25_START_MS,
        capture_duration_seconds=600,
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
            _market_evidence(POLYMARKET_ROUND25_START_MS + 100),
        )
        store.append_messages(
            RUN_ID,
            _public_messages(POLYMARKET_ROUND25_START_MS + 200),
        )
        report = store.finish_run(
            RUN_ID,
            started_at_ms=POLYMARKET_ROUND25_START_MS,
            ended_at_ms=POLYMARKET_ROUND25_START_MS + 600_000,
            database=str(database),
            errors=(),
        )
    assert report.status == "complete"
    _write_hashed(
        state_root / "segment-0000-result.json",
        {
            "condition_admission_pending": True,
            "details": {
                "condition_count": len(report.conditions),
                "duration_seconds": report.duration_seconds,
                "ended_at_ms": report.ended_at_ms,
                "errors": list(report.errors),
                "integrity_errors": list(report.integrity_errors),
                "manifest_sha256": manifest["manifest_sha256"],
                "raw_message_count": report.raw_message_count,
                "report_sha256": report.report_sha256,
                "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
                "run_id": report.run_id,
                "started_at_ms": report.started_at_ms,
                "stream_counts": dict(report.stream_counts),
                "stream_gap_count": report.stream_gap_count,
            },
            "live_trading_authority": False,
            "model_data_eligible": False,
            "observed_at_ms": report.ended_at_ms + 1,
            "paper_trading_authority": False,
            "plan_sha256": plan.plan_sha256,
            "profitability_claim": False,
            "schema_version": POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION,
            "segment_index": 0,
            "status": report.status,
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
            "schema_version": "polymarket-round25-twap-core-campaign-state-v1",
            "status": "campaign_window_ended",
            "status_counts": {"complete": 1},
            "terminal_segment_count": 1,
        },
    )
    return database, state_root


class _ReceiptObserver:
    def __init__(self) -> None:
        self.starts: list[tuple[str, int]] = []
        self.messages: list[tuple[str, str]] = []
        self.finishes: list[str] = []

    def start_run(self, segment: dict[str, object], gaps: tuple[object, ...]) -> None:
        self.starts.append((str(segment["run_id"]), len(gaps)))

    def observe_message(
        self,
        segment: dict[str, object],
        message: RawStreamMessage,
    ) -> None:
        self.messages.append((str(segment["run_id"]), message.stream))

    def finish_run(self, segment: dict[str, object]) -> None:
        self.finishes.append(str(segment["run_id"]))


def test_terminal_design_is_frozen_before_capture() -> None:
    design = load_round25_terminal_design(ROOT)

    assert design["design_sha256"] == POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256
    assert design["frozen_at_ms"] < POLYMARKET_ROUND25_START_MS
    assert design["materialization"]["official_resolution_accessed"] is False


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("condition_admission", "twap_outcome_reconstructed"),
        ("materialization", "future_receipts_permitted"),
        ("materialization", "round24_twap_settlement_label_constructed"),
        ("authority", "credentials_used"),
    ],
)
def test_terminal_design_rejects_relaxed_causal_or_authority_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    value = json.loads(
        (
            ROOT
            / "docs"
            / "model-research"
            / "polymarket"
            / "round-025-terminal-receipt-materialization-design-v1.json"
        ).read_text(encoding="ascii")
    )
    body = dict(value)
    body.pop("design_sha256")
    body[section] = {**body[section], field: True}
    changed_sha256 = _canonical_sha256(body)
    body["design_sha256"] = changed_sha256
    destination = (
        tmp_path
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-terminal-receipt-materialization-design-v1.json"
    )
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        round25_terminal,
        "POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256",
        changed_sha256,
    )

    with pytest.raises(ValueError, match="terminal design differs"):
        load_round25_terminal_design(tmp_path)


def test_terminal_manifest_round_trips_and_excludes_uncovered_time(
    tmp_path: Path,
) -> None:
    state_root = _terminal_state_root(tmp_path)
    manifest = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )

    assert validate_round25_terminal_transport_manifest(manifest) == manifest
    assert manifest["eligible_run_ids"] == [RUN_ID]
    assert manifest["condition_admission_pending"] is True
    assert manifest["all_scheduled_transport_interval_covered"] is False
    assert manifest["known_ineligible_or_unobserved_intervals"]
    assert manifest["outcomes_consulted"] is False
    assert manifest["live_trading_authority"] is False


def test_terminal_manifest_rejects_early_access_and_tampering(tmp_path: Path) -> None:
    state_root = _terminal_state_root(tmp_path)
    with pytest.raises(RuntimeError, match="before campaign end"):
        build_round25_terminal_transport_manifest(
            ROOT,
            plan_path=PLAN,
            state_root=state_root,
            observed_at_ms=POLYMARKET_ROUND25_END_MS - 1,
        )
    manifest = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )
    changed = dict(manifest)
    changed["resolution_source"] = "https://example.invalid/point-price"
    with pytest.raises(ValueError, match="manifest"):
        validate_round25_terminal_transport_manifest(changed)

    changed = json.loads(json.dumps(manifest))
    changed_segment = changed["segments"][0]
    changed_segment["stream_counts"]["clob_market"] = True
    changed_segment["raw_message_count"] = 5
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = _canonical_sha256(changed)
    with pytest.raises(ValueError, match="stream count"):
        validate_round25_terminal_transport_manifest(changed)


def test_failed_segment_is_preserved_but_ineligible(tmp_path: Path) -> None:
    state_root = _terminal_state_root(tmp_path, status="failed")
    manifest = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )

    assert manifest["eligible_run_ids"] == []
    assert manifest["condition_admission_pending"] is False
    assert manifest["segments"][0]["exclusion_reasons"] == ["segment_status_failed"]
    assert manifest["segments"][0]["source_manifest_sha256"] is None


def test_terminal_receipt_audit_reconciles_exact_synthetic_database(
    tmp_path: Path,
) -> None:
    database, state_root = _terminal_database_fixture(tmp_path)
    transport = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )
    observer = _ReceiptObserver()

    audit = audit_round25_terminal_receipts(
        database=database,
        terminal_transport_manifest=transport,
        observed_at_ms=POLYMARKET_ROUND25_END_MS + 1,
        observer=observer,
    )

    assert audit["database_run_count"] == 1
    assert audit["receipt_replay_complete"] is True
    assert audit["condition_admission_pending"] is True
    assert audit["eligible_runs"][0]["receipt_count"] == 2
    assert audit["eligible_runs"][0]["stream_counts"] == {
        "clob_market": 1,
        "polymarket_rtds": 1,
    }
    assert (
        audit["eligible_runs"][0]["receipt_chain_sha256"]
        != hashlib.sha256(b"").hexdigest()
    )
    assert audit["eligible_runs"][0]["gap_count"] == 0
    assert observer.starts == [(RUN_ID, 0)]
    assert observer.messages == [
        (RUN_ID, "clob_market"),
        (RUN_ID, "polymarket_rtds"),
    ]
    assert observer.finishes == [RUN_ID]
    assert audit["outcomes_consulted"] is False
    assert audit["live_trading_authority"] is False

    path = tmp_path / "receipt-audit.json"
    write_round25_terminal_receipt_audit(
        path,
        audit,
        terminal_transport_manifest=transport,
    )
    assert (
        load_round25_terminal_receipt_audit(
            path,
            terminal_transport_manifest=transport,
        )
        == audit
    )


def test_terminal_receipt_audit_rejects_rehashed_extra_eligible_field(
    tmp_path: Path,
) -> None:
    database, state_root = _terminal_database_fixture(tmp_path)
    transport = build_round25_terminal_transport_manifest(
        ROOT,
        plan_path=PLAN,
        state_root=state_root,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )
    audit = audit_round25_terminal_receipts(
        database=database,
        terminal_transport_manifest=transport,
        observed_at_ms=POLYMARKET_ROUND25_END_MS + 1,
    )
    changed = json.loads(json.dumps(audit))
    changed["eligible_runs"][0]["unexpected"] = True
    changed.pop("audit_sha256")
    changed["audit_sha256"] = _canonical_sha256(changed)

    with pytest.raises(ValueError, match="eligible receipt run"):
        validate_round25_terminal_receipt_audit(
            changed,
            terminal_transport_manifest=transport,
        )
