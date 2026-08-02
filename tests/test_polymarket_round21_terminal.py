from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round20_campaign as round20_campaign
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    RecorderReport,
    StreamGap,
)
from simple_ai_trading.polymarket_round20_campaign import (
    POLYMARKET_ROUND20_CAMPAIGN_SECONDS,
    POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
    POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
    build_round20_segment_manifest,
    create_round20_campaign_plan,
    validate_round20_campaign_plan,
)
from simple_ai_trading.polymarket_round21_terminal import (
    POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256,
    audit_round21_terminal_receipts,
    build_round21_terminal_transport_manifest,
    load_round21_terminal_receipt_audit,
    load_round21_terminal_transport_design,
    load_round21_terminal_transport_manifest,
    validate_round21_terminal_transport_manifest,
    validate_round21_terminal_receipt_audit,
    write_round21_terminal_receipt_audit,
    write_round21_terminal_transport_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CREATED_MS = 1_800_000_000_000
START_MS = 1_800_001_200_000
END_MS = START_MS + POLYMARKET_ROUND20_CAMPAIGN_SECONDS * 1_000
INTERRUPTION_MS = START_MS + 3 * 86_400_000


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


def _hashed(value: dict[str, object], hash_name: str) -> dict[str, object]:
    body = dict(value)
    body.pop(hash_name, None)
    body[hash_name] = _canonical_sha256(body)
    return body


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def _plan_value() -> dict[str, object]:
    files = {
        relative: hashlib.sha256(relative.encode("ascii")).hexdigest()
        for relative in round20_campaign._REQUIRED_FILES
    }
    return create_round20_campaign_plan(
        created_at_ms=CREATED_MS,
        scheduled_start_ms=START_MS,
        repository_commit_oid="a" * 40,
        repository_tree_oid="b" * 40,
        repository_file_sha256=files,
    )


def _interrupted_segment(plan_sha256: str) -> dict[str, object]:
    return _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "segment_index": 0,
            "status": "interrupted",
            "observed_at_ms": INTERRUPTION_MS + 50,
            "condition_admission_pending": False,
            "details": {
                "run_id": "1" * 32,
                "report_sha256": "2" * 64,
                "started_at_ms": START_MS + 600,
                "ended_at_ms": INTERRUPTION_MS,
                "raw_message_count": 16_000_000,
                "integrity_errors": ["terminal_integrity_audit_incomplete"],
                "errors": ["campaign_restart_interrupted_segment"],
            },
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )


def _complete_segment(
    plan_sha256: str,
    *,
    index: int,
    started_at_ms: int,
    ended_at_ms: int,
    status: str = "complete",
) -> dict[str, object]:
    gap_count = 1 if status == "degraded" else 0
    return _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "segment_index": index,
            "status": status,
            "observed_at_ms": ended_at_ms + 100,
            "condition_admission_pending": True,
            "details": {
                "run_id": f"{index + 3:x}" * 32,
                "manifest_sha256": f"{index + 4:x}" * 64,
                "report_sha256": f"{index + 5:x}" * 64,
                "started_at_ms": started_at_ms,
                "ended_at_ms": ended_at_ms,
                "duration_seconds": (ended_at_ms - started_at_ms) / 1_000.0,
                "raw_message_count": 20_000_000,
                "stream_gap_count": gap_count,
                "stream_counts": {
                    "clob_market": 19_000_000,
                    "clob_rest_book": 30,
                    "polymarket_rtds": 999_970,
                },
                "condition_count": 8_000,
                "integrity_errors": [],
                "errors": [],
            },
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )


def _stale_state(plan_sha256: str, *, observed_at_ms: int) -> dict[str, object]:
    return _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "observed_at_ms": observed_at_ms,
            "phase": "capturing",
            "segment_index": 1,
            "details": {"run_id": "4" * 32},
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )


def _terminal_state(plan_sha256: str, *, segment_count: int) -> dict[str, object]:
    return _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "status": "campaign_window_ended",
            "terminal_segment_count": segment_count,
            "status_counts": {"complete": segment_count},
            "recovered_interrupted_segment_count": 0,
            "condition_admission_pending": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    plan = _plan_value()
    plan_path = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    _write_json(plan_path, plan)
    return plan_path, state_root, plan


def _market_evidence(observed_at_ms: int) -> MarketEvidence:
    start = datetime.fromtimestamp(START_MS / 1_000, tz=UTC)
    end = datetime.fromtimestamp((START_MS + 300_000) / 1_000, tz=UTC)
    payload = {
        "id": "market-BTC",
        "question": "BTC Up or Down",
        "conditionId": "0x" + "7" * 64,
        "slug": f"btc-updown-5m-{START_MS // 1_000}",
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
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
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
            connection_id="clob-a:" + "a" * 32,
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


def _segment_from_report(
    *,
    plan_sha256: str,
    segment_index: int,
    manifest_sha256: str | None,
    report: RecorderReport,
    interrupted: bool,
) -> dict[str, object]:
    if interrupted:
        details = {
            "run_id": report.run_id,
            "report_sha256": report.report_sha256,
            "started_at_ms": report.started_at_ms,
            "ended_at_ms": report.ended_at_ms,
            "raw_message_count": report.raw_message_count,
            "integrity_errors": list(report.integrity_errors),
            "errors": list(report.errors),
        }
        status = "interrupted"
    else:
        details = {
            "run_id": report.run_id,
            "manifest_sha256": manifest_sha256,
            "report_sha256": report.report_sha256,
            "started_at_ms": report.started_at_ms,
            "ended_at_ms": report.ended_at_ms,
            "duration_seconds": report.duration_seconds,
            "raw_message_count": report.raw_message_count,
            "stream_gap_count": report.stream_gap_count,
            "stream_counts": dict(report.stream_counts),
            "condition_count": len(report.conditions),
            "integrity_errors": list(report.integrity_errors),
            "errors": list(report.errors),
        }
        status = report.status
    return _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "segment_index": segment_index,
            "status": status,
            "observed_at_ms": report.ended_at_ms + 100,
            "condition_admission_pending": not interrupted,
            "details": details,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )


def test_terminal_design_is_hash_bound_and_non_authoritative() -> None:
    design = load_round21_terminal_transport_design(ROOT)

    assert design["design_sha256"] == (
        POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
    )
    assert design["purpose"] == (
        "transport_run_admission_only_not_condition_or_model_admission"
    )
    assert design["authority"]["model_data_eligible"] is False
    assert design["causal_blinding"]["outcomes_consulted"] is False


def test_terminal_manifest_preserves_interruption_and_role_coverage(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    _write_json(
        state_root / "segments" / "segment-0000.json",
        _interrupted_segment(str(plan["plan_sha256"])),
    )
    _write_json(
        state_root / "segments" / "segment-0001.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=1,
            started_at_ms=INTERRUPTION_MS + 100,
            ended_at_ms=END_MS + 1_000,
        ),
    )
    _write_json(
        state_root / "campaign-state.json",
        _stale_state(str(plan["plan_sha256"]), observed_at_ms=END_MS - 1_000),
    )

    manifest = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS + 2_000,
    )

    assert manifest["campaign_state_mode"] == "superseded_active_heartbeat"
    assert manifest["eligible_run_ids"] == ["4" * 32]
    assert manifest["segments"][0]["eligible_for_condition_rebuild"] is False
    assert manifest["segments"][0]["exclusion_reasons"] == [
        "segment_status_interrupted",
        "condition_admission_not_pending",
        "recorder_integrity_errors_present",
        "recorder_errors_present",
    ]
    assert manifest["segments"][1]["eligible_for_condition_rebuild"] is True
    gap = manifest["known_ineligible_or_unobserved_intervals"][0]
    assert gap["start_ms"] == START_MS
    assert gap["end_ms"] == INTERRUPTION_MS + 100
    train = manifest["role_transport_coverage"][0]
    assert train["role"] == "train"
    assert train["known_uncovered_transport_ms"] == 3 * 86_400_000 + 100
    assert manifest["all_scheduled_transport_interval_covered"] is False
    assert manifest["model_data_eligible"] is False

    written = tmp_path / "terminal.json"
    write_round21_terminal_transport_manifest(written, manifest)
    assert load_round21_terminal_transport_manifest(written) == manifest


def test_terminal_manifest_accepts_persisted_terminal_state(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    _write_json(
        state_root / "segments" / "segment-0000.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=0,
            started_at_ms=START_MS,
            ended_at_ms=END_MS,
        ),
    )
    _write_json(
        state_root / "campaign-state.json",
        _terminal_state(str(plan["plan_sha256"]), segment_count=1),
    )

    manifest = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )

    assert manifest["campaign_state_mode"] == "persisted_terminal"
    assert manifest["known_ineligible_or_unobserved_intervals"] == []
    assert manifest["all_scheduled_transport_interval_covered"] is True
    assert all(
        row["complete_transport_interval_coverage"]
        for row in manifest["role_transport_coverage"]
    )


def test_terminal_manifest_rejects_premature_or_unsuperseded_access(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    segment = _complete_segment(
        str(plan["plan_sha256"]),
        index=0,
        started_at_ms=START_MS,
        ended_at_ms=END_MS - 1,
    )
    _write_json(state_root / "segments" / "segment-0000.json", segment)
    stale = _stale_state(str(plan["plan_sha256"]), observed_at_ms=END_MS - 2_000)
    stale["segment_index"] = 0
    stale["details"] = {"run_id": "3" * 32}
    _write_json(state_root / "campaign-state.json", _hashed(stale, "artifact_sha256"))

    with pytest.raises(RuntimeError, match="before campaign end"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS - 1,
        )
    with pytest.raises(ValueError, match="heartbeat is not superseded"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )


def test_terminal_manifest_rejects_source_and_derived_tampering(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    segment = _complete_segment(
        str(plan["plan_sha256"]),
        index=0,
        started_at_ms=START_MS,
        ended_at_ms=END_MS,
    )
    _write_json(state_root / "segments" / "segment-0000.json", segment)
    _write_json(
        state_root / "campaign-state.json",
        _terminal_state(str(plan["plan_sha256"]), segment_count=1),
    )
    manifest = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )

    changed = dict(manifest)
    changed["all_scheduled_transport_interval_covered"] = False
    with pytest.raises(ValueError, match="derivation differs"):
        validate_round21_terminal_transport_manifest(
            _hashed(changed, "manifest_sha256")
        )

    source_changed = dict(segment)
    source_changed["condition_admission_pending"] = False
    _write_json(state_root / "segments" / "segment-0000.json", source_changed)
    with pytest.raises(ValueError, match="source segment differs"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )


def test_terminal_manifest_rejects_overlapping_eligible_lifecycles(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    first_end = START_MS + 20 * 86_400_000
    _write_json(
        state_root / "segments" / "segment-0000.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=0,
            started_at_ms=START_MS,
            ended_at_ms=first_end,
        ),
    )
    _write_json(
        state_root / "segments" / "segment-0001.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=1,
            started_at_ms=first_end - 1,
            ended_at_ms=END_MS,
        ),
    )
    _write_json(
        state_root / "campaign-state.json",
        _terminal_state(str(plan["plan_sha256"]), segment_count=2),
    )

    with pytest.raises(ValueError, match="lifecycles overlap"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )


def test_terminal_json_reader_rejects_missing_malformed_and_non_object(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="unavailable"):
        load_round21_terminal_transport_manifest(missing)

    malformed = tmp_path / "manifest.json"
    for content, message in (
        ('{"a":1,"a":2}', "duplicate keys"),
        ('{"a":NaN}', "contains NaN"),
        ("{x", "not strict JSON"),
        ("[]", "not an object"),
        ("", "size differs"),
    ):
        malformed.write_text(content, encoding="ascii")
        with pytest.raises(ValueError, match=message):
            load_round21_terminal_transport_manifest(malformed)


def test_terminal_source_rejects_malformed_segment_evidence(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    complete = _complete_segment(
        str(plan["plan_sha256"]),
        index=0,
        started_at_ms=START_MS,
        ended_at_ms=END_MS,
    )
    interrupted = _interrupted_segment(str(plan["plan_sha256"]))
    failed = _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "segment_index": 0,
            "status": "failed",
            "observed_at_ms": END_MS + 1,
            "condition_admission_pending": False,
            "details": {"failure_type": "RuntimeError", "failure": "fixture"},
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )
    _write_json(
        state_root / "campaign-state.json",
        _terminal_state(str(plan["plan_sha256"]), segment_count=1),
    )

    cases: list[tuple[dict[str, object], str]] = []
    changed = json.loads(json.dumps(complete))
    changed["details"].pop("condition_count")
    cases.append((_hashed(changed, "artifact_sha256"), "report details differ"))
    changed = json.loads(json.dumps(complete))
    changed["details"]["run_id"] = "invalid"
    cases.append((_hashed(changed, "artifact_sha256"), "run ID differs"))
    changed = json.loads(json.dumps(complete))
    changed["details"]["started_at_ms"] = START_MS - 1
    cases.append((_hashed(changed, "artifact_sha256"), "segment timing differs"))
    changed = json.loads(json.dumps(complete))
    changed["details"]["errors"] = ["fixture_error"]
    cases.append((_hashed(changed, "artifact_sha256"), "eligible report differs"))
    changed = json.loads(json.dumps(interrupted))
    changed["details"].pop("errors")
    cases.append(
        (_hashed(changed, "artifact_sha256"), "interrupted report details differ")
    )
    changed = json.loads(json.dumps(interrupted))
    changed["details"]["errors"] = []
    cases.append((_hashed(changed, "artifact_sha256"), "interrupted report differs"))
    changed = json.loads(json.dumps(failed))
    changed["details"]["failure"] = ""
    cases.append((_hashed(changed, "artifact_sha256"), "failure details differ"))
    changed = json.loads(json.dumps(failed))
    changed["status"] = "complete"
    changed["condition_admission_pending"] = True
    cases.append((_hashed(changed, "artifact_sha256"), "failure status differs"))
    changed = json.loads(json.dumps(complete))
    changed["observed_at_ms"] = END_MS - 1
    cases.append(
        (_hashed(changed, "artifact_sha256"), "observation predates its end")
    )

    segment_path = state_root / "segments" / "segment-0000.json"
    for source, message in cases:
        _write_json(segment_path, source)
        with pytest.raises(ValueError, match=message):
            build_round21_terminal_transport_manifest(
                ROOT,
                plan_path=plan_path,
                state_root=state_root,
                observed_at_ms=END_MS + 2,
            )


def test_terminal_manifest_preserves_failed_segment_and_trailing_gap(
    tmp_path: Path,
) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    failed = _hashed(
        {
            "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "segment_index": 0,
            "status": "failed",
            "observed_at_ms": START_MS + 1,
            "condition_admission_pending": False,
            "details": {"failure_type": "RuntimeError", "failure": "fixture"},
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
        "artifact_sha256",
    )
    complete = _complete_segment(
        str(plan["plan_sha256"]),
        index=1,
        started_at_ms=START_MS,
        ended_at_ms=END_MS - 1,
    )
    _write_json(state_root / "segments" / "segment-0000.json", failed)
    _write_json(state_root / "segments" / "segment-0001.json", complete)
    state = {
        "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "campaign_window_ended",
        "terminal_segment_count": 2,
        "status_counts": {"complete": 1, "failed": 1},
        "recovered_interrupted_segment_count": 0,
        "condition_admission_pending": True,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    _write_json(
        state_root / "campaign-state.json",
        _hashed(state, "artifact_sha256"),
    )

    manifest = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )

    assert manifest["segments"][0]["details_kind"] == "failure"
    assert manifest["segments"][0]["run_id"] is None
    assert manifest["segments"][0]["exclusion_reasons"] == [
        "segment_status_failed",
        "condition_admission_not_pending",
    ]
    assert manifest["known_ineligible_or_unobserved_intervals"][-1] == {
        "start_ms": END_MS - 1,
        "end_ms": END_MS,
        "duration_ms": 1,
        "role_overlaps_ms": [{"role": "test", "duration_ms": 1}],
    }


def test_terminal_manifest_validator_rejects_summary_tampering(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    _write_json(
        state_root / "segments" / "segment-0000.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=0,
            started_at_ms=START_MS,
            ended_at_ms=END_MS,
        ),
    )
    _write_json(
        state_root / "campaign-state.json",
        _terminal_state(str(plan["plan_sha256"]), segment_count=1),
    )
    manifest = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )

    cases: list[tuple[dict[str, object], str]] = []
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0].pop("run_id")
    cases.append((changed, "manifest segment differs"))
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0]["status"] = "unknown"
    cases.append((changed, "manifest segment status differs"))
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0]["exclusion_reasons"] = ["invented"]
    cases.append((changed, "manifest exclusion differs"))
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0]["run_id"] = "invalid"
    cases.append((changed, "manifest run ID differs"))
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0]["started_at_ms"] += 1
    cases.append((changed, "manifest timing differs"))
    changed = json.loads(json.dumps(manifest))
    changed["segments"][0]["errors"] = ["fixture_error"]
    changed["segments"][0]["exclusion_reasons"] = [
        "recorder_errors_present"
    ]
    cases.append((changed, "eligible segment differs"))
    changed = json.loads(json.dumps(manifest))
    changed["source_plan_sha256"] = "not-a-digest"
    cases.append((changed, "transport manifest differs"))

    for changed, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_round21_terminal_transport_manifest(
                _hashed(changed, "manifest_sha256")
            )


def test_terminal_source_requires_segment_set_and_consistent_state(tmp_path: Path) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    state = _terminal_state(str(plan["plan_sha256"]), segment_count=1)
    _write_json(state_root / "campaign-state.json", state)
    with pytest.raises(ValueError, match="segment directory is unavailable"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )
    (state_root / "segments").mkdir()
    with pytest.raises(ValueError, match="segment set is empty"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )
    _write_json(
        state_root / "segments" / "segment-0001.json",
        _complete_segment(
            str(plan["plan_sha256"]),
            index=1,
            started_at_ms=START_MS,
            ended_at_ms=END_MS,
        ),
    )
    with pytest.raises(ValueError, match="segment set is not contiguous"):
        build_round21_terminal_transport_manifest(
            ROOT,
            plan_path=plan_path,
            state_root=state_root,
            observed_at_ms=END_MS,
        )


def test_terminal_receipt_audit_requires_at_least_one_eligible_run(
    tmp_path: Path,
) -> None:
    plan_path, state_root, plan = _fixture_paths(tmp_path)
    interrupted = _interrupted_segment(str(plan["plan_sha256"]))
    _write_json(state_root / "segments" / "segment-0000.json", interrupted)
    state = {
        "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "campaign_window_ended",
        "terminal_segment_count": 1,
        "status_counts": {"interrupted": 1},
        "recovered_interrupted_segment_count": 1,
        "condition_admission_pending": True,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    _write_json(
        state_root / "campaign-state.json",
        _hashed(state, "artifact_sha256"),
    )
    transport = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )
    assert transport["condition_admission_pending"] is False

    with pytest.raises(RuntimeError, match="has no eligible run"):
        audit_round21_terminal_receipts(
            database=tmp_path / "absent.duckdb",
            terminal_transport_manifest=transport,
            observed_at_ms=END_MS,
        )


def test_terminal_receipt_audit_reconciles_database_and_skips_interruption(
    tmp_path: Path,
) -> None:
    plan_path, state_root, raw_plan = _fixture_paths(tmp_path)
    plan = validate_round20_campaign_plan(raw_plan)
    database = tmp_path / "terminal.duckdb"
    interrupted_run = "1" * 32
    eligible_run = "4" * 32
    interrupted_manifest = build_round20_segment_manifest(
        plan,
        run_id=interrupted_run,
        created_at_ms=START_MS + 600,
        duration_seconds=3 * 86_400,
        segment_index=0,
    )
    eligible_manifest = build_round20_segment_manifest(
        plan,
        run_id=eligible_run,
        created_at_ms=INTERRUPTION_MS + 100,
        duration_seconds=(END_MS - INTERRUPTION_MS - 100) // 1_000,
        segment_index=1,
    )
    with PolymarketEvidenceStore(database) as store:
        store.start_run(
            interrupted_run,
            START_MS + 600,
            preregistration_manifest=interrupted_manifest,
        )
        store.record_market_evidence(
            interrupted_run,
            _market_evidence(START_MS + 700),
        )
        store.append_messages(interrupted_run, _public_messages(START_MS + 800))
        interrupted_report = store.fail_run(
            interrupted_run,
            started_at_ms=START_MS + 600,
            ended_at_ms=INTERRUPTION_MS,
            database=str(database),
            errors=("campaign_restart_interrupted_segment",),
        )

        store.start_run(
            eligible_run,
            INTERRUPTION_MS + 100,
            preregistration_manifest=eligible_manifest,
        )
        store.record_market_evidence(
            eligible_run,
            _market_evidence(INTERRUPTION_MS + 200),
        )
        store.append_messages(
            eligible_run,
            _public_messages(INTERRUPTION_MS + 300),
        )
        store.record_gap(
            eligible_run,
            StreamGap(
                stream="clob_market",
                connection_id="clob-b:" + "c" * 32,
                opened_at_ms=INTERRUPTION_MS + 400,
                reason="fixture_disconnect",
                last_sequence_number=1,
            ),
        )
        eligible_report = store.finish_run(
            eligible_run,
            started_at_ms=INTERRUPTION_MS + 100,
            ended_at_ms=END_MS,
            database=str(database),
            errors=(),
        )
    assert interrupted_report.status == "failed"
    assert eligible_report.status == "degraded"
    _write_json(
        state_root / "segments" / "segment-0000.json",
        _segment_from_report(
            plan_sha256=plan.plan_sha256,
            segment_index=0,
            manifest_sha256=None,
            report=interrupted_report,
            interrupted=True,
        ),
    )
    _write_json(
        state_root / "segments" / "segment-0001.json",
        _segment_from_report(
            plan_sha256=plan.plan_sha256,
            segment_index=1,
            manifest_sha256=str(eligible_manifest["manifest_sha256"]),
            report=eligible_report,
            interrupted=False,
        ),
    )
    terminal_state = {
        "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "status": "campaign_window_ended",
        "terminal_segment_count": 2,
        "status_counts": {"degraded": 1, "interrupted": 1},
        "recovered_interrupted_segment_count": 1,
        "condition_admission_pending": True,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    _write_json(
        state_root / "campaign-state.json",
        _hashed(terminal_state, "artifact_sha256"),
    )
    transport = build_round21_terminal_transport_manifest(
        ROOT,
        plan_path=plan_path,
        state_root=state_root,
        observed_at_ms=END_MS,
    )

    audit = audit_round21_terminal_receipts(
        database=database,
        terminal_transport_manifest=transport,
        observed_at_ms=END_MS + 1,
    )

    assert audit["database_run_count"] == 2
    assert audit["eligible_runs"][0]["run_id"] == eligible_run
    assert audit["eligible_runs"][0]["receipt_count"] == 2
    assert audit["eligible_runs"][0]["stream_counts"] == {
        "clob_market": 1,
        "polymarket_rtds": 1,
    }
    assert audit["eligible_runs"][0]["receipt_chain_sha256"] != (
        hashlib.sha256(b"").hexdigest()
    )
    assert audit["eligible_runs"][0]["gap_count"] == 1
    assert audit["eligible_runs"][0]["gap_chain_sha256"] != (
        hashlib.sha256(b"").hexdigest()
    )
    assert audit["ineligible_runs"] == [
        {
            "segment_index": 0,
            "run_id": interrupted_run,
            "segment_status": "interrupted",
            "database_status": "failed",
            "report_sha256": interrupted_report.report_sha256,
            "preregistration_manifest_sha256": interrupted_manifest[
                "manifest_sha256"
            ],
            "receipts_replayed": False,
        }
    ]
    assert "asset_id" not in json.dumps(audit, sort_keys=True)
    assert audit["model_data_eligible"] is False

    path = tmp_path / "receipt-audit.json"
    write_round21_terminal_receipt_audit(
        path,
        audit,
        terminal_transport_manifest=transport,
    )
    assert load_round21_terminal_receipt_audit(
        path,
        terminal_transport_manifest=transport,
    ) == audit

    changed = dict(audit)
    changed_runs = [dict(audit["eligible_runs"][0])]
    changed_runs[0]["receipt_count"] = 3
    changed["eligible_runs"] = changed_runs
    with pytest.raises(ValueError, match="receipt run accounting differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )

    changed = json.loads(json.dumps(audit))
    changed["eligible_runs"][0].pop("receipt_count")
    with pytest.raises(ValueError, match="receipt run differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )
    changed = json.loads(json.dumps(audit))
    changed["ineligible_runs"][0].pop("report_sha256")
    with pytest.raises(ValueError, match="ineligible run differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )
    changed = json.loads(json.dumps(audit))
    changed["ineligible_runs"][0]["receipts_replayed"] = True
    with pytest.raises(ValueError, match="ineligible run accounting differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )
    changed = json.loads(json.dumps(audit))
    changed.pop("database_run_count")
    with pytest.raises(ValueError, match="receipt audit differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )
    changed = json.loads(json.dumps(audit))
    changed["receipt_replay_complete"] = False
    with pytest.raises(ValueError, match="receipt authority differs"):
        validate_round21_terminal_receipt_audit(
            _hashed(changed, "audit_sha256"),
            terminal_transport_manifest=transport,
        )

    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            """
            UPDATE polymarket_recorder_run
            SET started_at_ms = started_at_ms + 1
            WHERE run_id = ?
            """,
            [eligible_run],
        )
    with pytest.raises(ValueError, match="database report differs"):
        audit_round21_terminal_receipts(
            database=database,
            terminal_transport_manifest=transport,
            observed_at_ms=END_MS + 2,
        )
    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            """
            UPDATE polymarket_recorder_run
            SET started_at_ms = started_at_ms - 1
            WHERE run_id = ?
            """,
            [eligible_run],
        )
        store.connect().execute(
            """
            UPDATE polymarket_preregistration_manifest
            SET manifest_sha256 = ? WHERE run_id = ?
            """,
            ["f" * 64, eligible_run],
        )
    with pytest.raises(ValueError, match="database manifest differs"):
        audit_round21_terminal_receipts(
            database=database,
            terminal_transport_manifest=transport,
            observed_at_ms=END_MS + 2,
        )
    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            """
            UPDATE polymarket_preregistration_manifest
            SET manifest_sha256 = ? WHERE run_id = ?
            """,
            [eligible_manifest["manifest_sha256"], eligible_run],
        )
        store.connect().execute(
            """
            INSERT INTO polymarket_stream_gap
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "e" * 64,
                eligible_run,
                "clob_market",
                "clob-a:" + "d" * 32,
                INTERRUPTION_MS + 500,
                "post_report_fixture_gap",
                1,
            ],
        )
    with pytest.raises(ValueError, match="gap accounting differs"):
        audit_round21_terminal_receipts(
            database=database,
            terminal_transport_manifest=transport,
            observed_at_ms=END_MS + 2,
        )
    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            "DELETE FROM polymarket_stream_gap WHERE reason = ?",
            ["post_report_fixture_gap"],
        )

    with PolymarketEvidenceStore(database) as store:
        store.start_run("9" * 32, END_MS + 1)
    with pytest.raises(ValueError, match="database run set differs"):
        audit_round21_terminal_receipts(
            database=database,
            terminal_transport_manifest=transport,
            observed_at_ms=END_MS + 2,
        )
