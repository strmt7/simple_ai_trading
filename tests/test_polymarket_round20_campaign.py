from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import (
    POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
)
from simple_ai_trading.polymarket_round20_campaign import (
    POLYMARKET_ROUND20_CAMPAIGN_SECONDS,
    POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT,
    POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS,
    build_round20_segment_manifest,
    create_round20_campaign_plan,
    validate_round20_campaign_plan,
    validate_round20_segment_manifest,
)
from simple_ai_trading.polymarket_round20_capture import create_round20_recorder
from simple_ai_trading.polymarket_round20_contract import (
    POLYMARKET_ROUND20_CONTRACT_SHA256,
)


CREATED_MS = 1_800_000_000_000
START_MS = 1_800_001_200_000


def _required_files() -> dict[str, str]:
    from simple_ai_trading import polymarket_round20_campaign as campaign

    return {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in campaign._REQUIRED_FILES
    }


def _plan_value() -> dict[str, object]:
    return create_round20_campaign_plan(
        created_at_ms=CREATED_MS,
        scheduled_start_ms=START_MS,
        repository_commit_oid="a" * 40,
        repository_tree_oid="b" * 40,
        repository_file_sha256=_required_files(),
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


def test_campaign_plan_is_exactly_thirty_days_and_polymarket_only() -> None:
    raw = _plan_value()
    plan = validate_round20_campaign_plan(raw)

    assert POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS == 1_200
    assert POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT == 2_160
    assert POLYMARKET_ROUND20_CAMPAIGN_SECONDS == 30 * 86_400
    assert plan.scheduled_end_ms - plan.scheduled_start_ms == 30 * 86_400_000
    assert raw["required_streams"] == ["clob_market", "polymarket_rtds"]
    assert raw["planned_transport_restarts"] is False
    assert raw["binance_required"] is False
    assert raw["binance_captured"] is False
    assert raw["live_trading_authority"] is False


def test_campaign_plan_rejects_rehashed_scope_resource_or_authority_drift() -> None:
    for key, value in (
        ("required_streams", ["binance_spot", "clob_market"]),
        ("minimum_free_bytes", 1),
        ("planned_transport_restarts", True),
        ("live_trading_authority", True),
    ):
        changed = _plan_value()
        changed[key] = value
        with pytest.raises(ValueError, match="campaign plan differs"):
            validate_round20_campaign_plan(_rehash(changed, "plan_sha256"))


def test_segment_manifest_is_continuous_independent_and_non_authoritative() -> None:
    plan = validate_round20_campaign_plan(_plan_value())
    manifest = build_round20_segment_manifest(
        plan,
        run_id="c" * 32,
        created_at_ms=START_MS + 1,
        duration_seconds=30 * 86_400 - 1,
        segment_index=0,
    )
    validated = validate_round20_segment_manifest(manifest, plan)

    assert validated["round20_contract_sha256"] == (POLYMARKET_ROUND20_CONTRACT_SHA256)
    assert validated["first_logical_unit_index"] == 0
    assert validated["purpose"] == "prospective_corpus"
    assert validated["required_clob_lanes"] == ["clob-a", "clob-b"]
    assert validated["optional_predictor_sources_captured"] == []
    assert validated["binance_execution_connected"] is False
    assert validated["model_data_eligible"] is False


def test_segment_manifest_rejects_rehashed_binance_or_duration_drift() -> None:
    plan = validate_round20_campaign_plan(_plan_value())
    original = build_round20_segment_manifest(
        plan,
        run_id="c" * 32,
        created_at_ms=START_MS + 1,
        duration_seconds=1_200,
        segment_index=0,
    )
    for key, value in (
        ("required_streams", ["binance_spot", "clob_market"]),
        ("binance_execution_connected", True),
        ("capture_duration_seconds", 30 * 86_400 + 1),
    ):
        changed = dict(original)
        changed[key] = value
        with pytest.raises(ValueError, match="segment manifest differs"):
            validate_round20_segment_manifest(
                _rehash(changed, "manifest_sha256"),
                plan,
            )


def test_campaign_plan_requires_aligned_future_start() -> None:
    with pytest.raises(ValueError, match="campaign plan differs"):
        create_round20_campaign_plan(
            created_at_ms=CREATED_MS,
            scheduled_start_ms=START_MS + 1,
            repository_commit_oid="a" * 40,
            repository_tree_oid="b" * 40,
            repository_file_sha256=_required_files(),
        )


def test_campaign_design_hash_is_canonical() -> None:
    from simple_ai_trading import polymarket_round20_campaign as campaign

    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-020-continuous-campaign-design-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("design_sha256")
    assert claimed == campaign.POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256
    assert (
        claimed
        == hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    )


def test_recorder_rejects_duration_above_campaign_bound(
    tmp_path: Path,
) -> None:
    recorder = create_round20_recorder(tmp_path / "capture.duckdb")

    with pytest.raises(ValueError, match=r"\[5, 2592000\]"):
        asyncio.run(
            recorder.run(
                duration_seconds=POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS + 1,
            )
        )


def test_recorder_external_stop_probe_fails_closed(tmp_path: Path) -> None:
    recorder = create_round20_recorder(tmp_path / "capture.duckdb")
    stop = asyncio.Event()
    output: asyncio.Queue[object] = asyncio.Queue()

    asyncio.run(
        recorder._progress_loop(
            None,
            stop,
            output,
            run_id="a" * 32,
            started_at_ms=START_MS,
            duration_seconds=1_200,
            interval_seconds=30,
            stop_requested=lambda: "minimum_free_space_not_met",
        )
    )

    assert stop.is_set()
    assert recorder.errors == ["external_stop:minimum_free_space_not_met"]


def test_recorder_external_stop_probe_exception_fails_closed(
    tmp_path: Path,
) -> None:
    recorder = create_round20_recorder(tmp_path / "capture.duckdb")
    stop = asyncio.Event()
    output: asyncio.Queue[object] = asyncio.Queue()

    def unavailable() -> str | None:
        raise OSError("probe unavailable")

    asyncio.run(
        recorder._progress_loop(
            None,
            stop,
            output,
            run_id="a" * 32,
            started_at_ms=START_MS,
            duration_seconds=1_200,
            interval_seconds=30,
            stop_requested=unavailable,
        )
    )

    assert stop.is_set()
    assert recorder.errors == ["external_stop_check:OSError:probe unavailable"]
