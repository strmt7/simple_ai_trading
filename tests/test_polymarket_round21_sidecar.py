from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_recorder as recorder_module
from simple_ai_trading.polymarket_recorder import (
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading.polymarket_round21_sidecar import (
    POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES,
    POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256,
    POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES,
    create_round21_sidecar_manifest,
    create_round21_sidecar_recorder,
    round21_sidecar_state,
    validate_round21_sidecar_manifest,
)

DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-binance-sidecar-capture-design-v1.json"
)


def _files() -> dict[str, str]:
    paths = (
        "docs/model-research/polymarket/"
        "round-021-independent-matched-edge-contract-v1.json",
        "docs/model-research/polymarket/"
        "round-021-binance-sidecar-capture-design-v1.json",
        "src/simple_ai_trading/polymarket_recorder.py",
        "src/simple_ai_trading/polymarket_round21_contract.py",
        "src/simple_ai_trading/polymarket_round21_sidecar.py",
        "tests/test_polymarket_recorder.py",
        "tests/test_polymarket_round21_contract.py",
        "tests/test_polymarket_round21_sidecar.py",
        "tools/run_polymarket_round21_sidecar.py",
    )
    return {
        path: hashlib.sha256(path.encode("ascii")).hexdigest() for path in paths
    }


def _manifest() -> dict[str, object]:
    return create_round21_sidecar_manifest(
        run_id="a" * 32,
        created_at_ms=1_800_000_000_000,
        capture_duration_seconds=60,
        scheduled_end_ms=1_788_046_800_000,
        repository_commit_oid="b" * 40,
        repository_tree_oid="c" * 40,
        repository_file_sha256=_files(),
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("manifest_sha256", None)
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round21_sidecar_is_public_independent_and_bounded(tmp_path: Path) -> None:
    recorder = create_round21_sidecar_recorder(tmp_path / "sidecar.duckdb")

    assert recorder.required_assets == ()
    assert recorder.required_streams == ("binance_futures", "binance_spot")
    assert recorder.assets == ("BTC",)
    assert recorder.include_polymarket_core is False
    assert recorder.include_binance_spot is True
    assert recorder.include_binance_futures is True
    assert recorder.include_rtds_binance is False
    assert recorder.binance_book_ticker_profile is True
    assert POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES == 64 * 1024**3
    assert POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES == 256 * 1024**3


def test_round21_sidecar_design_hash_is_canonical() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256
    assert claimed == hashlib.sha256(
        json.dumps(
            design,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert design["scope"]["credentials"] is False
    assert design["scope"]["execution"] is False
    assert design["independence"]["failure_blocks_polymarket_capture"] is False


def test_round21_sidecar_selects_exact_public_book_and_trade_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = create_round21_sidecar_recorder(tmp_path / "sidecar.duckdb")
    selected: list[dict[str, object]] = []

    async def fake_simple_stream(**kwargs: object) -> None:
        selected.append(kwargs)

    monkeypatch.setattr(recorder, "_simple_stream", fake_simple_stream)
    queue: asyncio.Queue[RawStreamMessage | StreamGap | None] = asyncio.Queue()
    stop = asyncio.Event()

    async def invoke() -> None:
        await recorder._binance_stream(queue, stop)
        await recorder._binance_futures_stream(queue, stop)

    asyncio.run(invoke())

    assert [item["stream"] for item in selected] == [
        "binance_spot",
        "binance_futures",
    ]
    assert selected[0]["url"] == recorder_module.ROUND21_BINANCE_SPOT_WEBSOCKET
    assert (
        selected[1]["url"]
        == recorder_module.ROUND21_BINANCE_FUTURES_WEBSOCKET
    )
    assert selected[0]["subscription"] is None
    assert selected[1]["subscription"] is None


def test_round21_sidecar_manifest_is_non_authoritative() -> None:
    manifest = validate_round21_sidecar_manifest(_manifest())

    assert manifest["required_assets"] == []
    assert manifest["required_streams"] == ["binance_futures", "binance_spot"]
    assert manifest["spot_streams"] == ["btcusdt@bookTicker", "btcusdt@trade"]
    assert manifest["usdm_streams"] == ["btcusdt@bookTicker", "btcusdt@trade"]
    assert manifest["binance_credentials_used"] is False
    assert manifest["binance_execution_connected"] is False
    assert manifest["model_data_eligible"] is False
    assert manifest["live_trading_authority"] is False


def test_round21_sidecar_rejects_rehashed_execution_or_scope_drift() -> None:
    for key, value in (
        ("required_assets", ["BTC"]),
        ("required_streams", ["binance_spot"]),
        ("binance_execution_connected", True),
        ("live_trading_authority", True),
    ):
        changed = _manifest()
        changed[key] = value
        with pytest.raises(ValueError, match="manifest differs"):
            validate_round21_sidecar_manifest(_rehash(changed))


def test_round21_source_only_coverage_requires_exact_false_authorities(
    tmp_path: Path,
) -> None:
    database = tmp_path / "coverage.duckdb"
    run_id = "a" * 32
    manifest = _manifest()
    with PolymarketEvidenceStore(database) as store:
        store.start_run(
            run_id,
            int(manifest["created_at_ms"]),
            preregistration_manifest=manifest,
        )
        assert store._coverage_requirements(run_id) == (
            (),
            ("binance_futures", "binance_spot"),
        )

    changed = _manifest()
    changed["profitability_claim"] = True
    changed = _rehash(changed)
    second_database = tmp_path / "coverage-drift.duckdb"
    with PolymarketEvidenceStore(second_database) as store:
        store.start_run(
            run_id,
            int(changed["created_at_ms"]),
            preregistration_manifest=changed,
        )
        with pytest.raises(ValueError, match="capture scope is invalid"):
            store._coverage_requirements(run_id)


def test_round21_sidecar_state_remains_non_authoritative() -> None:
    state = round21_sidecar_state(
        phase="capturing",
        observed_at_ms=1_800_000_000_000,
        database_bytes=123,
        wal_bytes=456,
        free_bytes=789,
        details={"received_message_count": 10},
    )

    claimed = state.pop("artifact_sha256")
    assert claimed == hashlib.sha256(
        json.dumps(
            state,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert state["model_data_eligible"] is False
    assert state["profitability_claim"] is False
    assert state["paper_trading_authority"] is False
    assert state["live_trading_authority"] is False
