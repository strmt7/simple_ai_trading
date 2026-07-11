from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.microstructure_data import (
    BINANCE_FUTURES_MARKET_STREAM_URL,
    BINANCE_FUTURES_PUBLIC_STREAM_URL,
    MICROSTRUCTURE_SCHEMA_VERSION,
    ClockSyncEvidence,
    MicrostructureCaptureResult,
    SymbolMicrostructureEvidence,
)
from simple_ai_trading.microstructure_features import MICROSTRUCTURE_FEATURE_NAMES
from simple_ai_trading.microstructure_live import (
    LiveMicrostructurePrediction,
    LiveTopOfBook,
)
from simple_ai_trading.microstructure_model import (
    MicrostructureActionPrediction,
    TradingMetrics,
)
from simple_ai_trading.microstructure_runtime import StreamingFeatureRow
from simple_ai_trading.microstructure_shadow import (
    PROMOTION_SHADOW_CONFIG,
    ShadowReplayResult,
    VirtualShadowTrade,
    _VirtualShadowLedger,
    evaluate_shadow_capture,
    replay_shadow_capture,
)


class _AlwaysLongScorer:
    symbol = "BTCUSDT"
    decision_cadence_seconds = 1
    total_latency_ms = 500
    max_quote_age_ms = 1_000
    reference_order_notional_quote = 100.0
    max_l1_participation = 0.05
    horizon_seconds = 2
    stop_loss_bps = 5_000.0
    take_profit_bps = 5_000.0
    trigger_execution_slippage_bps = 1.0
    taker_fee_bps = 0.1
    additional_slippage_bps_per_side = 0.2

    def score(self, _features, **_kwargs) -> MicrostructureActionPrediction:
        return MicrostructureActionPrediction(
            side="LONG",
            long_expected_net_bps=5.0,
            short_expected_net_bps=-5.0,
            long_profitable_probability=0.75,
            short_profitable_probability=0.25,
            minimum_predicted_edge_bps=1.0,
            minimum_profitable_probability=0.6,
            long_l1_participation=0.01,
            short_l1_participation=0.01,
            reason="long_expected_value",
        )


def _quote(event_time_ms: int, update_id: int, *, mid: float = 100.0) -> LiveTopOfBook:
    return LiveTopOfBook(
        symbol="BTCUSDT",
        event_time_ms=event_time_ms,
        transaction_time_ms=event_time_ms - 1,
        update_id=update_id,
        bid=mid - 0.01,
        ask=mid + 0.01,
        bid_qty=1_000.0,
        ask_qty=1_000.0,
    )


def test_virtual_shadow_entry_waits_for_latency_deadline_and_closes_on_horizon() -> None:
    scorer = _AlwaysLongScorer()
    ledger = _VirtualShadowLedger(scorer, entry_cutoff_ms=None)
    feature_row = StreamingFeatureRow(
        symbol="BTCUSDT",
        feature_version="fixture",
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        source_second_ms=0,
        decision_time_ms=1_000,
        close_bid=99.99,
        close_ask=100.01,
        close_bid_qty=1_000.0,
        close_ask_qty=1_000.0,
        features=np.zeros(len(MICROSTRUCTURE_FEATURE_NAMES), dtype=np.float64),
    )
    prediction = LiveMicrostructurePrediction(
        feature_row=feature_row,
        execution_quote=_quote(1_100, 1),
        prediction=scorer.score(feature_row.features),
        observed_exchange_time_ms=1_100,
        signal_deadline_ms=1_500,
        remaining_latency_budget_ms=400,
    )

    ledger.observe_quote(_quote(1_100, 1))
    ledger.observe_prediction(prediction)
    assert ledger.pending_entry is not None
    assert ledger.open_position is None

    ledger.observe_quote(_quote(1_400, 2, mid=100.05))
    assert ledger.pending_entry is not None
    assert ledger.open_position is None

    ledger.observe_quote(_quote(1_500, 3, mid=100.10))
    assert ledger.pending_entry is None
    assert ledger.open_position is not None
    assert ledger.open_position.entry_time_ms == 1_500

    ledger.observe_quote(_quote(3_500, 4, mid=100.30))
    assert ledger.open_position is None
    assert len(ledger.trades) == 1
    assert ledger.trades[0].exit_reason == "horizon"
    trade = ledger.trades[0]
    exit_ratio = trade.exit_price / trade.entry_price
    assert trade.realized_net_bps == pytest.approx(
        (exit_ratio - 1.0) * 10_000.0
        - (scorer.taker_fee_bps + scorer.additional_slippage_bps_per_side)
        * (1.0 + exit_ratio)
    )


def _write_replay_capture(path: Path, *, seconds: int = 945) -> tuple[int, int]:
    base_ms = 1_800_000_000_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for index in range(seconds):
            second_ms = base_ms + index * 1_000
            mid = 100.0 + index * 0.05
            book_time = second_ms + 100
            trade_time = second_ms + 200
            book = {
                "e": "bookTicker",
                "E": book_time,
                "T": book_time - 5,
                "u": index + 1,
                "s": "BTCUSDT",
                "b": f"{mid - 0.01:.8f}",
                "B": "1000.0",
                "a": f"{mid + 0.01:.8f}",
                "A": "1000.0",
            }
            trade = {
                "e": "aggTrade",
                "E": trade_time,
                "T": trade_time - 5,
                "s": "BTCUSDT",
                "a": index + 1,
                "f": index + 1,
                "l": index + 1,
                "p": f"{mid:.8f}",
                "q": "1.0",
                "m": bool(index % 2),
            }
            for event_time, payload in ((book_time, book), (trade_time, trade)):
                received_at_ns = (event_time + 20) * 1_000_000
                handle.write(
                    f"{received_at_ns} "
                    + json.dumps(payload, separators=(",", ":"))
                    + "\n"
                )
    return base_ms + 100, base_ms + (seconds - 1) * 1_000 + 200


def test_shadow_replay_uses_live_coordinator_and_never_submits_orders(tmp_path) -> None:
    capture_path = tmp_path / "capture.jsonl.gz"
    first_ms, last_ms = _write_replay_capture(capture_path)
    scorer = _AlwaysLongScorer()

    result = replay_shadow_capture(
        scorer,
        capture_path,
        settlement_delay_ms=100,
        clock_offset_ms=0.0,
        entry_cutoff_ms=last_ms - scorer.horizon_seconds * 1_000 - 2_000,
    )

    assert result.started_at_ms == first_ms
    assert result.completed_at_ms == last_ms
    assert result.decisions > 0
    assert result.metrics.trades > 0
    assert result.metrics.total_net_bps > 0.0
    assert result.feed_sequence_gaps == 0
    assert result.invalid_events == 0
    assert result.feature_gap_resets == 0
    assert result.deadline_misses == 0
    assert result.inference_failures == 0
    assert result.pending_entries_at_end == 0
    assert result.forced_closes == 0
    assert result.orders_submitted == 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _ShadowCandidate:
    status: str
    rejection_reasons: tuple[str, ...]
    deployment_refit: object
    deployment_model_strings: dict[str, str]
    shadow_validation: object | None
    symbol: str = "BTCUSDT"
    risk_level: str = "conservative"

    def asdict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
            "deployment_refit": None,
            "deployment_model_strings": None,
            "shadow_validation": (
                None if self.shadow_validation is None else asdict(self.shadow_validation)
            ),
            "symbol": self.symbol,
            "risk_level": self.risk_level,
        }


def _capture_fixture(tmp_path: Path, *, started_at_ms: int, completed_at_ms: int):
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    raw_path = output_dir / "btcusdt.raw.jsonl.gz"
    synchronized_path = output_dir / "btcusdt.synchronized.jsonl.gz"
    snapshot_path = output_dir / "btcusdt.initial-depth.json"
    manifest_path = output_dir / "manifest.json"
    for path in (raw_path, synchronized_path):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("fixture\n")
    snapshot_path.write_text('{"lastUpdateId":1}\n', encoding="utf-8")
    evidence = SymbolMicrostructureEvidence(
        symbol="BTCUSDT",
        raw_path=str(raw_path),
        synchronized_raw_path=str(synchronized_path),
        snapshot_json_path=str(snapshot_path),
        initial_snapshot_path="",
        normalized_path="",
        raw_sha256=_sha256(raw_path),
        synchronized_raw_sha256=_sha256(synchronized_path),
        snapshot_json_sha256=_sha256(snapshot_path),
        normalized_sha256="",
        raw_bytes=raw_path.stat().st_size,
        normalized_bytes=0,
        snapshot_last_update_id=1,
        tick_size=0.1,
        lot_size=0.001,
        raw_messages=1_000,
        synchronized_messages=1_000,
        normalized_rows=0,
        depth_messages=100,
        depth_rows=1_000,
        trade_messages=400,
        trade_fill_count=400,
        ignored_non_market_trade_messages=0,
        book_ticker_messages=500,
        sequence_gap_count=0,
        crossed_book_count=0,
        invalid_event_count=0,
        first_exchange_time_ms=started_at_ms,
        last_exchange_time_ms=completed_at_ms,
        feed_latency_p50_ms=5.0,
        feed_latency_p95_ms=10.0,
        feed_latency_p99_ms=20.0,
        feed_latency_max_ms=30.0,
        replay_smoke_passed=False,
        replay_first_bid=None,
        replay_first_ask=None,
    )
    lower = "btcusdt"
    capture = MicrostructureCaptureResult(
        status="pass",
        capture_id="fixture-capture",
        schema_version=MICROSTRUCTURE_SCHEMA_VERSION,
        provider="binance",
        market_type="futures",
        stream_urls=(
            f"{BINANCE_FUTURES_PUBLIC_STREAM_URL}?streams="
            f"{lower}@depth@100ms/{lower}@bookTicker",
            f"{BINANCE_FUTURES_MARKET_STREAM_URL}?streams={lower}@aggTrade",
        ),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
        requested_duration_seconds=21_660.0,
        clock_sync=ClockSyncEvidence(0.5, 10.0, 5.0, 5, started_at_ms),
        symbols=("BTCUSDT",),
        evidence=(evidence,),
        errors=(),
    )
    manifest_path.write_text(
        json.dumps(capture.asdict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return capture


def test_shadow_evaluation_binds_capture_and_returns_accepted_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    now_ms = int(time.time() * 1_000)
    completed_at_ms = now_ms - 1_000
    started_at_ms = completed_at_ms - 21_660_000
    fitted_at = datetime.fromtimestamp(
        (started_at_ms - 1_000) / 1_000.0,
        tz=UTC,
    ).isoformat()
    refit = SimpleNamespace(
        fitted_at=fitted_at,
        training_cutoff_ms=started_at_ms - 2_000,
        expires_at_ms=now_ms + 86_400_000,
        deployment_model_sha256="d" * 64,
    )
    artifact = _ShadowCandidate(
        status="shadow_candidate",
        rejection_reasons=(),
        deployment_refit=refit,
        deployment_model_strings={"fixture": "model"},
        shadow_validation=None,
    )
    artifact_path = tmp_path / "candidate.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    capture = _capture_fixture(
        tmp_path,
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
    )
    scorer = _AlwaysLongScorer()
    replay_start_ms = started_at_ms + 30_000
    trades = tuple(
        VirtualShadowTrade(
            side="LONG" if index % 2 == 0 else "SHORT",
            entry_time_ms=replay_start_ms + index * 1_000,
            exit_time_ms=replay_start_ms + index * 1_000 + 500,
            entry_price=100.0,
            exit_price=100.1,
            predicted_net_bps=3.0,
            realized_net_bps=5.0,
            exit_reason="horizon",
        )
        for index in range(20)
    )
    replay = ShadowReplayResult(
        started_at_ms=replay_start_ms,
        completed_at_ms=replay_start_ms + 21_600_000,
        decisions=150,
        actionable_decisions=50,
        rejected_while_open=30,
        execution_liquidity_rejections=0,
        expired_entries=0,
        pending_entries_at_end=0,
        end_censored_signals=1,
        trades=trades,
        metrics=TradingMetrics(
            20,
            100.0,
            5.0,
            5.0,
            1.0,
            2.0,
            10.0,
            5.0,
            5.0,
            10,
            10,
            1,
            20.0,
        ),
        feed_sequence_gaps=0,
        invalid_events=0,
        late_event_resets=0,
        feature_gap_resets=0,
        deadline_misses=0,
        inference_failures=0,
        forced_closes=0,
        orders_submitted=0,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_shadow.load_microstructure_model_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_shadow.load_microstructure_action_scorer",
        lambda *_args, **_kwargs: scorer,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_shadow.replay_shadow_capture",
        lambda *_args, **_kwargs: replay,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_shadow._validated_shadow_binding",
        lambda accepted: accepted.shadow_validation,
    )

    report, accepted = evaluate_shadow_capture(
        artifact,  # type: ignore[arg-type]
        artifact_path,
        capture,
        report_path=tmp_path / "report.json",
        trades_path=tmp_path / "trades.csv",
        config=PROMOTION_SHADOW_CONFIG,
    )

    assert report.passed is True
    assert report.trading_authority is True
    assert report.replay["orders_submitted"] == 0
    assert accepted is not None
    assert accepted.status == "accepted"
    assert accepted.shadow_validation is not None
    assert accepted.shadow_validation.orders_submitted == 0
    assert len(accepted.shadow_validation.report_sha256) == 64
