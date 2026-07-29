from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from simple_ai_trading import polymarket_round14_capture as capture_module
from simple_ai_trading.polymarket_recorder import PolymarketPublicRecorder
from simple_ai_trading.polymarket_round14_capture import (
    POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
    create_round14_capture_manifest,
    validate_round14_capture_manifest,
)


NOW_MS = 1_800_000_000_000
CONTRACT_PATH = (
    "docs/model-research/polymarket/"
    "round-014-btc-5m-prospective-contract-v1.json"
)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _files() -> dict[str, str]:
    return {
        **{
            path: hashlib.sha256(path.encode("ascii")).hexdigest()
            for path in capture_module._REQUIRED_REPOSITORY_FILES
        },
        CONTRACT_PATH: "f" * 64,
    }


def _manifest(
    *,
    purpose: str = "qualification",
    duration: int = 120,
    slot_index: int | None = None,
    scheduled_start_ms: int | None = None,
    campaign_plan_sha256: str | None = None,
) -> dict[str, object]:
    return create_round14_capture_manifest(
        run_id="round14-capture-fixture",
        created_at_ms=NOW_MS,
        capture_duration_seconds=duration,
        purpose=purpose,
        repository_commit="a" * 40,
        repository_tree="b" * 40,
        contract_repository_path=CONTRACT_PATH,
        contract_sha256="c" * 64,
        required_file_sha256=_files(),
        slot_index=slot_index,
        scheduled_start_ms=scheduled_start_ms,
        campaign_plan_sha256=campaign_plan_sha256,
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output.pop("manifest_sha256", None)
    return {**output, "manifest_sha256": _canonical_sha(output)}


def test_round14_qualification_manifest_is_non_model_and_btc_scoped() -> None:
    manifest = _manifest()

    assert manifest["purpose"] == "qualification"
    assert manifest["model_data_eligible"] is False
    assert manifest["required_assets"] == ["BTC"]
    assert manifest["required_streams"] == [
        "binance_futures",
        "binance_spot",
        "clob_market",
        "polymarket_rtds",
    ]
    assert manifest["outcome_endpoints_queried"] is False
    assert manifest["paper_trading_authority"] is False
    assert manifest["live_trading_authority"] is False


def test_round14_prospective_manifest_binds_slot_and_plan() -> None:
    manifest = _manifest(
        purpose="prospective",
        duration=POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
        slot_index=7,
        scheduled_start_ms=NOW_MS,
        campaign_plan_sha256="d" * 64,
    )

    assert manifest["model_data_eligible"] is True
    assert manifest["slot_index"] == 7
    assert manifest["scheduled_start_ms"] == NOW_MS
    assert manifest["campaign_plan_sha256"] == "d" * 64


def test_round14_capture_manifest_rejects_tampering_or_scope_drift() -> None:
    manifest = _manifest()
    manifest["live_trading_authority"] = True
    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round14_capture_manifest(manifest)

    manifest = _manifest()
    manifest["required_assets"] = ["BTC", "ETH"]
    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round14_capture_manifest(_rehash(manifest))

    manifest = _manifest()
    manifest["capture_duration_seconds"] = 59
    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round14_capture_manifest(_rehash(manifest))


def test_scoped_recorder_refuses_unattested_capture(tmp_path) -> None:
    recorder = PolymarketPublicRecorder(
        tmp_path / "unattested.duckdb",
        assets=("BTC",),
        include_binance_futures=True,
    )

    with pytest.raises(ValueError, match="requires a preregistration manifest"):
        asyncio.run(recorder.run(duration_seconds=5))


def test_scoped_recorder_rejects_manifest_scope_drift(tmp_path) -> None:
    recorder = PolymarketPublicRecorder(
        tmp_path / "scope-drift.duckdb",
        assets=("BTC",),
        include_binance_futures=True,
    )

    def factory(run_id: str, started_at_ms: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "scope-drift-fixture-v1",
            "run_id": run_id,
            "created_at_ms": started_at_ms,
            "capture_duration_seconds": 5,
            "required_assets": ["BTC"],
            "required_streams": [
                "binance_spot",
                "clob_market",
                "polymarket_rtds",
            ],
        }
        payload["manifest_sha256"] = _canonical_sha(payload)
        return payload

    with pytest.raises(ValueError, match="scope differs from recorder"):
        asyncio.run(
            recorder.run(
                duration_seconds=5,
                preregistration_manifest_factory=factory,
            )
        )
