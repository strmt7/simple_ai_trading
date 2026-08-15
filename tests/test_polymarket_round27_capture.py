from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import RecorderReport
from simple_ai_trading.polymarket_round27_capture import (
    ROUND27_CAPTURE_DURATION_SECONDS,
    ROUND27_DATABASE_CAP_BYTES,
    ROUND27_STAGE0_DURATION_SECONDS,
    ROUND27_STAGE0_MAXIMUM_RESOLVED_MARKETS,
    Round27CaptureConfig,
    _create_recorder,
    _database_footprint_bytes,
    _manifest,
    create_round27_capture_contract,
    create_round27_stage0_capture_contract,
    run_round27_capture,
    run_round27_stage0_capture,
    validate_round27_capture_contract,
    validate_round27_stage0_capture_contract,
    write_round27_capture_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RESULT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-027-documented-source-smoke-result-v1-2026-08-15.json"
)
PUBLISHED_STAGE0_RESULT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-027-stage0-mechanics-capture-result-v1-2026-08-15.json"
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


def _report(*, status: str = "complete") -> RecorderReport:
    return RecorderReport(
        schema_version="test",
        run_id="a" * 32,
        status=status,
        database="test.duckdb",
        started_at_ms=1_000,
        ended_at_ms=601_000,
        duration_seconds=600.0,
        market_snapshot_count=4,
        raw_message_count=100,
        normalized_event_count=90,
        stream_gap_count=0,
        stream_counts={
            "binance_futures": 20,
            "binance_spot": 20,
            "clob_market": 20,
            "polymarket_rtds": 20,
        },
        assets=("BTC",),
        conditions=("condition",),
        integrity_errors=(),
        errors=(),
        evidence_manifest_sha256="b" * 64,
        report_sha256="c" * 64,
    )


def test_round27_capture_contract_is_hash_bound_and_non_authorizing() -> None:
    payload = create_round27_capture_contract(ROOT, created_at_ms=1_000)
    contract = validate_round27_capture_contract(payload, repository=ROOT)
    manifest = _manifest(contract, run_id="a" * 32, started_at_ms=2_000)
    manifest_claim = manifest.pop("manifest_sha256")

    assert contract.capture_duration_seconds == ROUND27_CAPTURE_DURATION_SECONDS
    assert payload["capture_scope"]["binance_futures_profile"] == (
        "documented_aggregate_trades"
    )
    assert payload["authority"]["model_data_eligible"] is False
    assert payload["authority"]["live_trading_authority"] is False
    assert manifest_claim == _canonical_sha256(manifest)


def test_round27_capture_contract_rejects_rehashed_semantic_drift() -> None:
    payload = create_round27_capture_contract(ROOT, created_at_ms=1_000)
    payload["resource_policy"]["database_cap_bytes"] += 1
    unhashed = dict(payload)
    unhashed.pop("contract_sha256")
    payload["contract_sha256"] = _canonical_sha256(unhashed)

    with pytest.raises(ValueError, match="contract differs"):
        validate_round27_capture_contract(payload, repository=ROOT)


def test_round27_stage0_contract_caps_analysis_without_granting_authority() -> None:
    payload = create_round27_stage0_capture_contract(ROOT, created_at_ms=1_000)
    contract = validate_round27_stage0_capture_contract(payload, repository=ROOT)

    assert contract.phase == "mechanics_stage0"
    assert contract.capture_duration_seconds == ROUND27_STAGE0_DURATION_SECONDS
    assert payload["analysis_policy"] == {
        "maximum_resolved_markets": ROUND27_STAGE0_MAXIMUM_RESOLVED_MARKETS,
        "target_access_during_capture": False,
        "condition_local_transport_audit_required": True,
        "parameter_selection_allowed": False,
        "economic_claim_allowed": False,
    }
    assert payload["success_gate"]["transport_gap_policy"] == (
        "condition_local_exclusion"
    )
    assert payload["authority"]["model_data_eligible"] is False


def test_round27_published_source_smoke_is_self_hashed_and_non_authorizing() -> None:
    payload = json.loads(PUBLISHED_RESULT.read_text(encoding="ascii"))
    claim = payload.pop("result_sha256")
    source_quality = dict(payload["source_quality"])
    source_claim = source_quality.pop("source_quality_sha256")

    assert claim == _canonical_sha256(payload)
    assert source_claim == _canonical_sha256(source_quality)
    assert payload["status"] == "passed"
    assert payload["failure_reasons"] == []
    assert payload["capture_report"]["raw_message_count"] == 194_980
    assert payload["capture_report"]["stream_gap_count"] == 0
    assert payload["source_quality"]["streams"]["binance_futures"][
        "accepted_trade_count"
    ] == 1_014
    assert payload["authority"]["model_data_eligible"] is False
    assert payload["authority"]["edge_claim"] is False
    assert payload["authority"]["live_trading_authority"] is False


def test_round27_published_stage0_capture_is_self_hashed_and_non_authorizing() -> None:
    payload = json.loads(PUBLISHED_STAGE0_RESULT.read_text(encoding="ascii"))
    claim = payload.pop("result_sha256")
    source_quality = dict(payload["source_quality"])
    source_claim = source_quality.pop("source_quality_sha256")

    assert claim == _canonical_sha256(payload)
    assert source_claim == _canonical_sha256(source_quality)
    assert payload["status"] == "passed"
    assert payload["capture_report"]["status"] == "degraded"
    assert payload["capture_report"]["raw_message_count"] == 6_099_812
    assert payload["capture_report"]["stream_gap_count"] == 9
    assert payload["analysis_policy"]["captured_condition_count"] == 62
    assert payload["analysis_policy"]["maximum_resolved_markets"] == 60
    assert payload["analysis_policy"][
        "model_data_eligible_before_condition_audit"
    ] is False
    assert payload["source_quality"]["streams"]["binance_futures"][
        "accepted_trade_count"
    ] == 61_304
    assert payload["authority"]["edge_claim"] is False
    assert payload["authority"]["profitability_claim"] is False
    assert payload["authority"]["live_trading_authority"] is False


def test_round27_recorder_uses_documented_aggregate_trade_profile() -> None:
    recorder = _create_recorder(ROOT / "data" / "unused-round27-test.duckdb")

    assert recorder.assets == ("BTC",)
    assert recorder.binance_futures_aggregate_trades is True
    assert recorder.binance_book_ticker_profile is False
    assert recorder.include_binance_futures is True
    assert recorder.include_binance_spot is True
    assert recorder.chainlink_price_mode == "twap_60s"
    assert recorder.memory_limit == "1GB"
    assert recorder.database_threads == 2


def test_round27_database_footprint_includes_wal_and_temp(tmp_path: Path) -> None:
    database = tmp_path / "capture.duckdb"
    database.write_bytes(b"a" * 3)
    Path(f"{database}.wal").write_bytes(b"b" * 5)
    temporary = Path(f"{database}.tmp")
    temporary.mkdir()
    (temporary / "spill.bin").write_bytes(b"c" * 7)

    assert _database_footprint_bytes(database) == 15
    assert _database_footprint_bytes(database) < ROUND27_DATABASE_CAP_BYTES


def test_round27_run_rejects_stale_database_before_recorder(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.json"
    database_path = tmp_path / "capture.duckdb"
    write_round27_capture_contract(
        contract_path,
        create_round27_capture_contract(ROOT, created_at_ms=1_000),
    )
    database_path.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="requires fresh"):
        asyncio.run(
            run_round27_capture(
                Round27CaptureConfig(
                    repository=ROOT,
                    contract_path=contract_path,
                    database_path=database_path,
                    result_path=tmp_path / "result.json",
                    lock_path=tmp_path / "capture.lock",
                ),
                recorder_factory=lambda _path: pytest.fail("recorder was invoked"),
            )
        )


def test_round27_run_passes_only_after_automatic_source_gate(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.json"
    write_round27_capture_contract(
        contract_path,
        create_round27_capture_contract(ROOT, created_at_ms=1_000),
    )
    calls: dict[str, object] = {}

    class Recorder:
        async def run(self, **options):
            calls.update(options)
            return _report()

    result = asyncio.run(
        run_round27_capture(
            Round27CaptureConfig(
                repository=ROOT,
                contract_path=contract_path,
                database_path=tmp_path / "capture.duckdb",
                result_path=tmp_path / "result.json",
                lock_path=tmp_path / "capture.lock",
            ),
            recorder_factory=lambda _path: Recorder(),
            source_audit=lambda _path, run_id: {
                "passed": run_id == "a" * 32,
                "source_quality_sha256": "d" * 64,
            },
        )
    )

    assert calls["duration_seconds"] == ROUND27_CAPTURE_DURATION_SECONDS
    assert calls["progress_interval_seconds"] == 30
    assert callable(calls["stop_requested"])
    assert result["status"] == "passed"
    assert result["gate_checks"][
        "documented_spot_and_futures_trade_quality_passed"
    ] is True
    assert result["authority"]["edge_claim"] is False


def test_round27_run_fails_on_gap_even_when_source_types_pass(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.json"
    write_round27_capture_contract(
        contract_path,
        create_round27_capture_contract(ROOT, created_at_ms=1_000),
    )
    report = replace(_report(status="degraded"), stream_gap_count=1)

    class Recorder:
        async def run(self, **_options):
            return report

    result = asyncio.run(
        run_round27_capture(
            Round27CaptureConfig(
                repository=ROOT,
                contract_path=contract_path,
                database_path=tmp_path / "capture.duckdb",
                result_path=tmp_path / "result.json",
                lock_path=tmp_path / "capture.lock",
            ),
            recorder_factory=lambda _path: Recorder(),
            source_audit=lambda _path, _run_id: {"passed": True},
        )
    )

    assert result["status"] == "failed"
    assert "terminal_recorder_status_accepted" in result["failure_reasons"]
    assert "stream_gap_count_zero" in result["failure_reasons"]


def test_round27_stage0_accepts_gap_degraded_capture_for_local_audit(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.json"
    write_round27_capture_contract(
        contract_path,
        create_round27_stage0_capture_contract(ROOT, created_at_ms=1_000),
    )
    report = replace(_report(status="degraded"), stream_gap_count=1)

    class Recorder:
        async def run(self, **options):
            assert options["duration_seconds"] == ROUND27_STAGE0_DURATION_SECONDS
            return report

    result = asyncio.run(
        run_round27_stage0_capture(
            Round27CaptureConfig(
                repository=ROOT,
                contract_path=contract_path,
                database_path=tmp_path / "capture.duckdb",
                result_path=tmp_path / "result.json",
                lock_path=tmp_path / "capture.lock",
            ),
            recorder_factory=lambda _path: Recorder(),
            source_audit=lambda _path, _run_id: {"passed": True},
        )
    )

    assert result["status"] == "passed"
    assert "stream_gap_count_zero" not in result["gate_checks"]
    assert result["analysis_policy"] == {
        "maximum_resolved_markets": 60,
        "captured_condition_count": 1,
        "condition_local_transport_audit_required": True,
        "stream_gap_count": 1,
        "model_data_eligible_before_condition_audit": False,
    }
    assert result["authority"]["model_data_eligible"] is False
