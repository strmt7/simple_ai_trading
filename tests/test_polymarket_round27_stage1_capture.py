from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import RecorderReport
from simple_ai_trading import polymarket_round27_stage1_capture as stage1


ROOT = Path(__file__).resolve().parents[1]
BASE_MS = 1_800_000_000_000
PUBLISHED_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-027-stage1-campaign-contract-v1.json"
)


def _slots() -> tuple[stage1.Round27Stage1Slot, ...]:
    duration_ms = stage1.ROUND27_STAGE1_SLOT_DURATION_SECONDS * 1_000
    starts = (
        BASE_MS,
        BASE_MS + 31 * 60 * 60 * 1_000,
        BASE_MS + 62 * 60 * 60 * 1_000,
        BASE_MS + 93 * 60 * 60 * 1_000,
    )
    return tuple(
        stage1.Round27Stage1Slot(
            slot_id=f"stage1-{chr(ord('a') + index)}",
            role="primary" if index < 3 else "contingency",
            scheduled_start_ms=start,
            scheduled_end_ms=start + duration_ms,
        )
        for index, start in enumerate(starts)
    )


def _contract(tmp_path: Path) -> tuple[Path, stage1.Round27Stage1Contract]:
    path = tmp_path / "contract.json"
    value = stage1.create_round27_stage1_contract(
        ROOT,
        created_at_ms=BASE_MS - 60 * 60 * 1_000,
        slots=_slots(),
    )
    stage1.write_round27_stage1_contract(path, value)
    return path, stage1.load_round27_stage1_contract(path, repository=ROOT)


def _report(slot: stage1.Round27Stage1Slot) -> RecorderReport:
    return RecorderReport(
        schema_version="test",
        run_id="a" * 32,
        status="degraded",
        database="test.duckdb",
        started_at_ms=slot.scheduled_start_ms,
        ended_at_ms=slot.scheduled_end_ms,
        duration_seconds=float(stage1.ROUND27_STAGE1_SLOT_DURATION_SECONDS),
        market_snapshot_count=128,
        raw_message_count=1_000,
        normalized_event_count=900,
        stream_gap_count=1,
        stream_counts={
            "binance_futures": 200,
            "binance_spot": 200,
            "clob_market": 200,
            "polymarket_rtds": 200,
        },
        assets=("BTC",),
        conditions=tuple(f"condition-{index}" for index in range(126)),
        integrity_errors=(),
        errors=(),
        evidence_manifest_sha256="b" * 64,
        report_sha256="c" * 64,
    )


def test_stage1_contract_freezes_three_dates_and_contingency() -> None:
    value = stage1.create_round27_stage1_contract(
        ROOT,
        created_at_ms=BASE_MS - 60 * 60 * 1_000,
        slots=_slots(),
    )
    contract = stage1.validate_round27_stage1_contract(value, repository=ROOT)

    assert len([slot for slot in contract.slots if slot.role == "primary"]) == 3
    assert contract.slots[-1].role == "contingency"
    assert value["campaign_policy"]["primary_scheduled_five_minute_intervals"] == 378
    assert (
        value["campaign_policy"]["minimum_eligible_markets_after_target_free_audit"]
        == 300
    )
    assert (
        value["campaign_policy"]["target_access_during_capture_or_admission"] is False
    )
    assert value["authority"]["profitability_claim"] is False


def test_published_stage1_contract_revalidates_exact_live_schedule() -> None:
    value = json.loads(PUBLISHED_CONTRACT.read_text(encoding="ascii"))
    contract = stage1.validate_round27_stage1_contract(value, repository=ROOT)

    assert contract.contract_sha256 == (
        "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
    )
    assert [slot.slot_id for slot in contract.slots] == [
        "stage1-a",
        "stage1-b",
        "stage1-c",
        "stage1-d",
    ]
    assert [slot.role for slot in contract.slots] == [
        "primary",
        "primary",
        "primary",
        "contingency",
    ]


def test_stage1_contract_rejects_rehashed_schedule_drift() -> None:
    value = stage1.create_round27_stage1_contract(
        ROOT,
        created_at_ms=BASE_MS - 60 * 60 * 1_000,
        slots=_slots(),
    )
    value["slots"][0]["scheduled_end_ms"] += 300_000
    body = dict(value)
    body.pop("contract_sha256")
    value["contract_sha256"] = stage1._canonical_sha256(body)

    with pytest.raises(ValueError, match="slot"):
        stage1.validate_round27_stage1_contract(value, repository=ROOT)


def test_stage1_slot_uses_remaining_fixed_window_and_passes_source_gate(
    tmp_path: Path,
) -> None:
    contract_path, contract = _contract(tmp_path)
    slot = contract.slots[0]
    calls: dict[str, object] = {}

    class Recorder:
        async def run(self, **options):
            calls.update(options)
            return _report(slot)

    result = asyncio.run(
        stage1.run_round27_stage1_slot(
            stage1.Round27Stage1SlotConfig(
                repository=ROOT,
                contract_path=contract_path,
                slot_id=slot.slot_id,
                database_path=tmp_path / "capture.duckdb",
                result_path=tmp_path / "result.json",
                lock_path=tmp_path / "slot.lock",
            ),
            recorder_factory=lambda _path: Recorder(),
            source_audit=lambda _path, _run_id: {"passed": True},
            clock_ms=lambda: slot.scheduled_start_ms,
        )
    )

    assert calls["duration_seconds"] == stage1.ROUND27_STAGE1_SLOT_DURATION_SECONDS
    assert calls["progress_interval_seconds"] == 30
    assert result["status"] == "passed"
    assert result["analysis_policy"]["captured_condition_count"] == 126
    assert result["analysis_policy"]["model_data_eligible_before_audit"] is False
    assert result["authority"]["live_trading_authority"] is False


def test_stage1_slot_rejects_late_launch_before_recorder(
    tmp_path: Path,
) -> None:
    contract_path, contract = _contract(tmp_path)
    slot = contract.slots[0]

    with pytest.raises(RuntimeError, match="launch tolerance"):
        asyncio.run(
            stage1.run_round27_stage1_slot(
                stage1.Round27Stage1SlotConfig(
                    repository=ROOT,
                    contract_path=contract_path,
                    slot_id=slot.slot_id,
                    database_path=tmp_path / "capture.duckdb",
                    result_path=tmp_path / "result.json",
                    lock_path=tmp_path / "slot.lock",
                ),
                recorder_factory=lambda _path: pytest.fail("recorder was invoked"),
                clock_ms=lambda: (
                    slot.scheduled_start_ms
                    + stage1.ROUND27_STAGE1_START_TOLERANCE_MS
                    + 1
                ),
            )
        )


def test_supervisor_advances_fixed_primary_slots_and_stops_before_contingency(
    tmp_path: Path,
) -> None:
    contract_path, contract = _contract(tmp_path)
    clock = [contract.slots[0].scheduled_start_ms - 60_000]
    observed: list[str] = []

    def sleeper(seconds: float) -> None:
        clock[0] += max(1, round(seconds * 1_000))

    def runner(config: stage1.Round27Stage1SlotConfig, _progress):
        slot = contract.slot(config.slot_id)
        observed.append(slot.slot_id)
        clock[0] = slot.scheduled_end_ms
        return stage1._slot_result(
            contract,
            slot,
            _report(slot),
            {"passed": True},
            footprint=1_000,
            resource_stop_reason="",
        )

    result = stage1.supervise_round27_stage1_primary(
        repository=ROOT,
        contract_path=contract_path,
        data_root=tmp_path / "data",
        state_path=tmp_path / "state.json",
        lease_path=tmp_path / "service.lock",
        clock_ms=lambda: clock[0],
        sleeper=sleeper,
        slot_runner=runner,
    )

    assert observed == ["stage1-a", "stage1-b", "stage1-c"]
    assert result["phase"] == "awaiting_primary_target_free_audits"
    assert result["outcomes"] == {
        "stage1-a": "passed",
        "stage1-b": "passed",
        "stage1-c": "passed",
    }
    state = json.loads((tmp_path / "state.json").read_text(encoding="ascii"))
    assert state["phase"] == "awaiting_primary_target_free_audits"
    assert state["trading_authority"] is False


def test_preexisting_slot_result_must_revalidate_before_skip(tmp_path: Path) -> None:
    contract_path, contract = _contract(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    slot = contract.slots[0]
    result = stage1._slot_result(
        contract,
        slot,
        _report(slot),
        {"passed": True},
        footprint=1_000,
        resource_stop_reason="",
    )
    result["contract_sha256"] = "0" * 64
    (data / "round27-stage1-a-result.json").write_text(
        json.dumps(result),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="slot result differs"):
        stage1.supervise_round27_stage1_primary(
            repository=ROOT,
            contract_path=contract_path,
            data_root=data,
            state_path=tmp_path / "state.json",
            lease_path=tmp_path / "service.lock",
            clock_ms=lambda: slot.scheduled_start_ms,
            sleeper=lambda _seconds: None,
            slot_runner=lambda _config, _progress: pytest.fail("slot was invoked"),
        )
