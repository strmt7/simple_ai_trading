from __future__ import annotations

from pathlib import Path

from simple_ai_trading.market_store import MarketDataStore


def _capture_payload(
    *, capture_id: str, status: str, completed_at_ms: int
) -> dict[str, object]:
    return {
        "capture_id": capture_id,
        "provider": "binance",
        "market_type": "futures",
        "schema_version": "binance-usdm-l2-v3",
        "status": status,
        "started_at_ms": 100,
        "completed_at_ms": completed_at_ms,
        "output_dir": f"data/microstructure/{capture_id}",
        "manifest_path": f"data/microstructure/{capture_id}/manifest.json",
        "evidence": [
            {
                "symbol": "BTCUSDT",
                "raw_path": "raw.gz",
                "normalized_path": "data.npz",
                "initial_snapshot_path": "snapshot.npz",
                "raw_sha256": "a" * 64,
                "normalized_sha256": "b" * 64,
                "raw_messages": 100,
                "normalized_rows": 1000,
                "depth_messages": 50,
                "trade_messages": 25,
                "book_ticker_messages": 25,
                "sequence_gap_count": 0,
                "crossed_book_count": 0,
                "invalid_event_count": 0,
                "replay_smoke_passed": True,
                "error": "",
            }
        ],
    }


def test_latest_capture_uses_bound_symbol_and_static_pass_filter(
    tmp_path: Path,
) -> None:
    passed = _capture_payload(
        capture_id="capture-pass",
        status="pass",
        completed_at_ms=200,
    )
    failed = _capture_payload(
        capture_id="capture-fail",
        status="fail",
        completed_at_ms=300,
    )

    with MarketDataStore(tmp_path / "microstructure.sqlite") as store:
        assert store.record_microstructure_capture(passed) > 0
        assert store.record_microstructure_capture(failed) > 0

        passed_result = store.latest_microstructure_capture("btcusdt")
        all_result = store.latest_microstructure_capture(
            "BTCUSDT",
            require_passed=False,
        )
        injected_result = store.latest_microstructure_capture(
            "BTCUSDT' OR 1=1 --",
            require_passed=False,
        )

    assert passed_result is not None
    assert passed_result["capture_id"] == "capture-pass"
    assert all_result is not None
    assert all_result["capture_id"] == "capture-fail"
    assert injected_result is None
