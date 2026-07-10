from __future__ import annotations

import asyncio
import json
import argparse
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import cli
from simple_ai_trading.advanced_model import (
    advanced_feature_signature,
    default_config_for,
    make_advanced_rows,
)
from simple_ai_trading.api import BinanceAPIError, Candle, CommissionRates, SymbolConstraints
from simple_ai_trading.assets import DEFAULT_AGGRESSIVE_LEVERAGE
from simple_ai_trading.config import RuntimeConfig, load_runtime, load_strategy, save_runtime, save_strategy
from simple_ai_trading.model import (
    ModelFeatureMismatchError,
    ModelLoadError,
    TemporalValidationSplit,
    TrainedModel,
    serialize_model,
)
from simple_ai_trading.features import ModelRow, feature_signature
from simple_ai_trading.market_store import MarketDataStore
from simple_ai_trading.positions import OpenPosition, PositionsStore, bot_client_order_id
from simple_ai_trading.types import StrategyConfig


@pytest.fixture(autouse=True)
def _isolate_repo_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _exchange_account(usdc: str = "1000") -> dict[str, object]:
    return {
        "updateTime": 123,
        "canTrade": True,
        "accountType": "MARGIN",
        "positions": [],
        "balances": [{"asset": "USDC", "free": usdc, "locked": "0"}],
        "assets": [{"asset": "USDC", "availableBalance": usdc}],
    }


def _live_data_coverage(
    symbol: str = "BTCUSDC",
    *,
    market_type: str = "spot",
    interval: str = "1s",
    years: float = 2.0,
) -> dict[str, object]:
    rows = int(365.25 * 24 * 60 * 60 * max(0.1, years))
    return {
        "symbol": symbol,
        "market_type": market_type,
        "interval": interval,
        "source_scope": "sqlite_market_data",
        "expected_interval_ms": 1000 if interval == "1s" else 60_000,
        "integrity_status": "ok",
        "integrity_warnings": [],
        "truth_basis": [
            "prices_from_timestamped_closed_candles",
            "coverage_measured_from_candle_close_time",
            "execution_results_are_simulated_not_exchange_fills",
        ],
        "full_history_requested": True,
        "full_available_history_used": True,
        "candles_available": rows,
        "candles_used": rows,
        "rows_used": rows - 100,
        "requested_start_ms": None,
        "requested_end_ms": None,
        "available_start_ms": 0,
        "available_end_ms": rows * 1000,
        "used_start_ms": 0,
        "used_end_ms": rows * 1000,
        "available_start_utc": "2024-01-01T00:00:00Z",
        "available_end_utc": "2026-01-01T00:00:00Z",
        "used_start_utc": "2024-01-01T00:00:00Z",
        "used_end_utc": "2026-01-01T00:00:00Z",
        "used_duration_days": years * 365.25,
        "used_duration_years": years,
        "gap_count": 0,
        "largest_gap_ms": 1000,
        "largest_gap_intervals": 1.0,
        "coverage_ratio": 1.0,
        "notes": [],
    }


def _promoted_execution_validation(
    symbol: str = "BTCUSDC",
    *,
    market_type: str = "spot",
    interval: str = "1s",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "passed": True,
        "symbol": symbol,
        "market_type": market_type,
        "interval": interval,
        "walk_forward_gate": {
            "passed": True,
            "reason": None,
            "fold_count": 3,
            "accepted_folds": 3,
            "worst_score": 0.08,
            "worst_realized_pnl": 1.2,
            "worst_max_drawdown": 0.025,
        },
        "stress": {"accepted": True},
        "temporal_robustness": {"accepted": True},
        "portfolio": {"accepted": True},
        "data_coverage": _live_data_coverage(symbol, market_type=market_type, interval=interval),
    }
    if market_type == "futures":
        payload["microstructure_replay"] = {
            "passed": True,
            "strategy_replay_passed": True,
            "replay_smoke_passed": True,
            "artifact_hashes_verified": True,
            "immutable_market_data": True,
            "engine": "hftbacktest",
            "engine_version": "2.4.4",
            "schema_version": "binance-usdm-l2-v3",
            "symbol": symbol,
            "queue_model": "risk_adverse_queue_model",
            "latency_model": "empirical_feed_and_order_latency",
            "captured_seconds": 20 * 86_400,
            "span_days": 400,
            "unique_days": 20,
            "normalized_rows": 20_000_000,
            "sequence_gap_count": 0,
            "crossed_book_count": 0,
            "invalid_event_count": 0,
            "clock_sync_samples": 100,
        }
    return payload


class _SignedCommissionClientEvidence:
    def get_commission_rates(self, symbol: str):
        return CommissionRates(
            symbol=symbol,
            market_type="futures",
            maker_rate=0.0002,
            taker_rate=0.0004,
            source="test_fixture",
        )


class _FakeClient(_SignedCommissionClientEvidence):
    def __init__(self) -> None:
        self.base_url = "https://api.testnet.binance.vision"
        self.orders = []

    def ping(self):
        return {"ok": True}

    def ensure_btcusdc(self):
        return {"symbol": "BTCUSDC"}

    def get_exchange_time(self):
        return {"serverTime": 123}

    def get_account(self):
        return _exchange_account()

    def get_max_leverage(self, symbol: str) -> int:
        return 10

    def get_symbol_constraints(self, symbol: str):
        return SimpleNamespace()  # placeholder unused in tests

    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
        return []

    def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
        self.orders.append((symbol, side, size, dry_run, leverage))
        return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage}


class _SignedFeeModelEvidence:
    strategy_overrides = {"taker_fee_bps": 4.0}


def _simple_candles(n: int = 320) -> list[Candle]:
    out: list[Candle] = []
    for i in range(n):
        base = 100.0 + i
        out.append(
            Candle(
                open_time=i * 60_000,
                open=base,
                high=base * 1.001,
                low=base * 0.999,
                close=base,
                volume=1.0,
                close_time=(i + 1) * 60_000,
            )
        )
    return out


def _live_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    payload: dict[str, object] = {
        "steps": 1,
        "sleep": 0,
        "paper": True,
        "live": False,
        "model": str(tmp_path / "missing-model.json"),
        "leverage": None,
        "retrain_interval": 0,
        "retrain_window": 300,
        "retrain_min_rows": 240,
        "external_signals": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_threshold_classification_guard_modes() -> None:
    baseline = SimpleNamespace(accuracy=0.50, precision=0.40, f1=0.40)
    stable = SimpleNamespace(accuracy=0.49, precision=0.39, f1=0.36)
    conservative = SimpleNamespace(accuracy=0.55, precision=0.39, f1=0.10)
    rejected = SimpleNamespace(accuracy=0.45, precision=0.20, f1=0.10)

    assert cli._threshold_classification_guard(baseline, stable)["mode"] == "f1_stable"
    assert cli._threshold_classification_guard(baseline, conservative)["mode"] == "accuracy_precision"
    zero_hit = SimpleNamespace(accuracy=0.90, precision=0.00, f1=0.00, true_positive=0, false_negative=3)
    assert cli._threshold_classification_guard(baseline, zero_hit)["mode"] == "zero_true_positive"
    guard = cli._threshold_classification_guard(baseline, rejected)
    assert guard["mode"] == "rejected"
    assert guard["passed"] is False
    assert guard["precision_floor"] == 0.38


def test_threshold_capital_preservation_guard_modes() -> None:
    accepted = SimpleNamespace(
        accepted=True,
        best_score=0.10,
        baseline_score=-0.90,
        realized_pnl=0.02,
        baseline_realized_pnl=-0.50,
        closed_trades=1,
        baseline_closed_trades=30,
    )
    guard = cli._threshold_capital_preservation_guard(accepted)
    assert guard["passed"] is True
    assert guard["mode"] == "capital_preservation"
    assert guard["closed_trades"] == 1

    weak = SimpleNamespace(
        accepted=True,
        best_score=-0.88,
        baseline_score=-0.90,
        realized_pnl=-0.49,
        baseline_realized_pnl=-0.50,
        closed_trades=1,
        baseline_closed_trades=30,
    )
    rejected = cli._threshold_capital_preservation_guard(weak)
    assert rejected["passed"] is False
    assert rejected["mode"] == "rejected"

    zero_hit = SimpleNamespace(
        accepted=True,
        best_score=0.20,
        baseline_score=-0.90,
        realized_pnl=0.20,
        baseline_realized_pnl=-0.50,
        closed_trades=1,
        baseline_closed_trades=30,
    )
    candidate = SimpleNamespace(true_positive=0, false_negative=2)
    zero_guard = cli._threshold_capital_preservation_guard(zero_hit, candidate)
    assert zero_guard["passed"] is False
    assert zero_guard["mode"] == "zero_true_positive"


def test_parse_args_and_main_dispatch(monkeypatch) -> None:
    args = cli._parse_args(["status"])
    assert callable(args.func)
    doctor_args = cli._parse_args(["doctor", "--input", "i.json", "--model", "m.json", "--online"])
    assert doctor_args.input == "i.json"
    assert doctor_args.model == "m.json"
    assert doctor_args.online is True
    audit_args = cli._parse_args(["audit", "--input", "i.json", "--model", "m.json"])
    assert audit_args.input == "i.json"
    assert audit_args.model == "m.json"
    roundtrip_args = cli._parse_args(["spot-roundtrip", "--quantity", "0.0002", "--mode", "sell-buy", "--yes"])
    assert roundtrip_args.quantity == 0.0002
    assert roundtrip_args.mode == "sell-buy"
    assert roundtrip_args.yes is True
    train_args = cli._parse_args(["train", "--preset", "quick", "--compute-backend", "directml", "--batch-size", "128"])
    assert train_args.preset == "quick"
    assert train_args.compute_backend == "directml"
    assert train_args.batch_size == 128
    compute_args = cli._parse_args(["compute", "--backend", "auto"])
    assert compute_args.backend == "auto"
    micro_candidate = cli._parse_args(["microstructure-train"])
    assert micro_candidate.evaluate_terminal is False
    assert micro_candidate.deployment_calibration_days == 14
    assert micro_candidate.maximum_model_age_seconds == 86_400
    micro_terminal = cli._parse_args(["microstructure-train", "--evaluate-terminal"])
    assert micro_terminal.evaluate_terminal is True
    micro_prequential = cli._parse_args(
        ["microstructure-prequential", "--max-folds", "2"]
    )
    assert micro_prequential.training_window_days == 180
    assert micro_prequential.evaluation_block_days == 7
    assert micro_prequential.max_folds == 2
    micro_promote = cli._parse_args(
        ["microstructure-promote", "--input", "candidate.json"]
    )
    assert micro_promote.input == "candidate.json"
    assert micro_promote.output is None
    assert micro_promote.prequential_report == "data/microstructure-prequential.json"
    micro_refit = cli._parse_args(["microstructure-refit", "--input", "validated.json"])
    assert micro_refit.input == "validated.json"
    assert micro_refit.output is None
    micro_shadow = cli._parse_args(
        ["microstructure-shadow", "--input", "shadow-candidate.json"]
    )
    assert micro_shadow.input == "shadow-candidate.json"
    assert micro_shadow.output is None
    assert micro_shadow.seconds == 21_660.0
    tick_plan = cli._parse_args(
        ["tick-archive-sync", "--full-history", "--plan-only"]
    )
    assert tick_plan.full_history is True
    assert tick_plan.plan_only is True
    assert tick_plan.start_date is None
    assert tick_plan.end_date is None
    tape_train = cli._parse_args(["tape-depth-train", "--window-days", "365"])
    assert tape_train.window_days == 365
    assert tape_train.horizon_seconds == 60
    assert tape_train.compute_backend == "auto"
    signals_args = cli._parse_args([
        "signals",
        "--compute-backend",
        "directml",
        "--short-reaction-refresh",
        "5",
        "--min-providers",
        "7",
        "--news-provider-limit",
        "30",
        "--provider-parallelism",
        "6",
        "--provider-jitter",
        "0.2",
        "--ollama-news",
    ])
    assert signals_args.compute_backend == "directml"
    assert signals_args.short_reaction_refresh == 5
    assert signals_args.min_providers == 7
    assert signals_args.news_provider_limit == 30
    assert signals_args.provider_parallelism == 6
    assert signals_args.provider_jitter == 0.2
    assert signals_args.ollama_news is True
    grade_args = cli._parse_args(["source-grades", "--window-hours", "12", "--no-ollama"])
    assert grade_args.window_hours == 12
    assert grade_args.ollama is False
    benchmark_args = cli._parse_args(["signals-benchmark", "--provider-limit", "30", "--parallelism", "8"])
    assert benchmark_args.provider_limit == [30]
    assert benchmark_args.parallelism == [8]
    suite_args = cli._parse_args(["train-suite", "--max-workers", "2", "--compute-backend", "auto", "--batch-size", "64"])
    assert suite_args.max_workers == 2
    assert suite_args.compute_backend == "auto"
    assert suite_args.batch_size == 64
    report_default = cli._parse_args(["report"])
    assert report_default.doctor is True
    report_no_doctor = cli._parse_args(["report", "--no-doctor"])
    assert report_no_doctor.doctor is False
    report_args = cli._parse_args(["report", "--account", "--doctor", "--online"])
    assert report_args.account is True
    assert report_args.doctor is True
    prepare_args = cli._parse_args(
        [
            "prepare",
            "--preset",
            "thorough",
            "--epochs",
            "77",
            "--learning-rate",
            "0.02",
            "--l2-penalty",
            "0.002",
            "--batch-size",
            "250",
            "--no-walk-forward",
            "--walk-forward-train",
            "120",
            "--walk-forward-test",
            "30",
            "--walk-forward-step",
            "10",
            "--no-calibrate-threshold",
            "--online-doctor",
        ]
    )
    assert prepare_args.preset == "thorough"
    assert prepare_args.epochs == 77
    assert prepare_args.learning_rate == 0.02
    assert prepare_args.l2_penalty == 0.002
    assert prepare_args.batch_size == 250
    assert prepare_args.walk_forward is False
    assert prepare_args.walk_forward_train == 120
    assert prepare_args.calibrate_threshold is False
    assert prepare_args.online_doctor is True
    strategy_args = cli._parse_args(["strategy", "--profile", "active"])
    assert strategy_args.profile == "active"

    marker = []

    def fake_status(ns: argparse.Namespace) -> int:
        marker.append("status")
        return 9

    monkeypatch.setattr(cli, "command_status", fake_status)
    assert cli.main(["status"]) == 9
    assert marker == ["status"]

    live = cli._parse_args(["live", "--steps", "3"])
    assert live.model == "data/model.json"
    assert live.retrain_interval == 0
    assert live.retrain_window == 300
    assert live.retrain_min_rows == 240
    assert live.paper is False


def test_microstructure_terminal_evaluation_requires_protective_exits(capsys) -> None:
    result = cli.command_microstructure_train(
        argparse.Namespace(
            evaluate_terminal=True,
            stop_loss_bps=None,
            take_profit_bps=None,
            risk_level="conservative",
            max_l1_participation=None,
            json=False,
        )
    )

    assert result == 2
    assert "requires explicit path-aware" in capsys.readouterr().err


def test_command_microstructure_prequential_rebuilds_exact_candidate_contract(
    monkeypatch,
    capsys,
) -> None:
    from types import SimpleNamespace

    artifact = SimpleNamespace(
        status="candidate",
        rejection_reasons=(),
        symbol="BTCUSDT",
        horizon_seconds=60,
        total_latency_ms=250,
        taker_fee_bps=5.0,
        max_quote_age_ms=1_000,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=0.05,
        decision_cadence_seconds=5,
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
    )
    dataset = object()
    calls: dict[str, object] = {}

    class Warehouse:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def rebuild_causal_feature_bars(self, symbol: str) -> None:
            calls["rebuild"] = symbol

    def build(_warehouse, **kwargs):
        calls["build"] = kwargs
        return dataset

    report = SimpleNamespace(
        status="passed",
        passed=True,
        reasons=(),
        aggregate={
            "metrics": {"trades": 10, "total_net_bps": 20.0, "max_drawdown_bps": 2.0},
            "confidence": {
                "mean_daily_net_bps_ci_lower": 0.1,
                "mean_daily_net_bps_ci_upper": 1.0,
            },
        },
        coverage={
            "complete_folds": 3,
            "planned_folds": 3,
            "evaluated_rows": 100,
        },
        predictions_path="predictions.csv",
        chart_path="chart.svg",
        asdict=lambda: {"status": "passed"},
    )

    def evaluate(candidate, source, **kwargs):
        calls["evaluate"] = (candidate, source, kwargs)
        return report

    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.load_microstructure_model_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_warehouse.MicrostructureWarehouse",
        Warehouse,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_features.build_executable_microstructure_dataset",
        build,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_prequential.evaluate_prequential_microstructure_model",
        evaluate,
    )

    result = cli.command_microstructure_prequential(
        argparse.Namespace(
            input="candidate.json",
            warehouse="warehouse.duckdb",
            cache_root="cache",
            output="report.json",
            predictions="predictions.csv",
            chart="chart.svg",
            compute_backend="cpu",
            training_window_days=30,
            minimum_training_days=10,
            calibration_days=3,
            policy_days=3,
            evaluation_block_days=2,
            minimum_segment_rows=32,
            minimum_class_rows=16,
            bootstrap_samples=1_000,
            max_folds=0,
            memory_limit="1GB",
            threads=1,
            json=False,
        )
    )

    assert result == 0
    assert calls["rebuild"] == "BTCUSDT"
    assert calls["build"]["reference_order_notional_quote"] == 1_000.0  # type: ignore[index]
    assert calls["evaluate"][0] is artifact  # type: ignore[index]
    assert calls["evaluate"][1] is dataset  # type: ignore[index]
    assert "terminal_holdout=not_accessed trading_authority=false" in capsys.readouterr().out


def test_command_microstructure_promote_binds_evidence_before_terminal_and_refit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from types import SimpleNamespace

    split = SimpleNamespace(terminal_start_ms=200 * 86_400_000)
    candidate = SimpleNamespace(
        status="candidate",
        rejection_reasons=(),
        prequential_validation=None,
        symbol="BTCUSDT",
        horizon_seconds=60,
        total_latency_ms=250,
        taker_fee_bps=5.0,
        max_quote_age_ms=1_000,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=0.05,
        decision_cadence_seconds=5,
        stop_loss_bps=25.0,
        take_profit_bps=40.0,
        trigger_execution_slippage_bps=1.0,
        feature_version="fixture-v1",
        schema_version="fixture-model-v1",
        split=split,
    )
    attached = SimpleNamespace(
        **{**vars(candidate), "prequential_validation": SimpleNamespace(report_sha256="e" * 64)}
    )
    terminal_metrics = SimpleNamespace(trades=40, total_net_bps=120.0)
    validated = SimpleNamespace(
        **{
            **vars(attached),
            "status": "validated",
            "terminal_metrics": terminal_metrics,
            "rejection_reasons": (),
        }
    )
    shadow_candidate = SimpleNamespace(
        **{
            **vars(validated),
            "status": "shadow_candidate",
            "asdict": lambda: {
                "status": "shadow_candidate",
                "model_strings": {"hidden": "value"},
                "deployment_model_strings": {"hidden": "value"},
            },
        }
    )
    dataset = SimpleNamespace(
        symbol="BTCUSDT",
        decision_time_ms=np.asarray([200 * 86_400_000, 201 * 86_400_000]),
        source_evidence={"manifest_fingerprint": "a" * 64, "build_id": "b" * 64},
    )
    calls: list[tuple[str, object]] = []

    class Warehouse:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def require_causal_feature_bars(self, symbol: str) -> None:
            calls.append(("require", symbol))

        def reserve_terminal_holdout(self, **kwargs):
            calls.append(("reserve", kwargs))
            return {"reservation_id": "f" * 64, "status": "reserved"}

        def finalize_terminal_holdout(self, reservation_id: str, **kwargs):
            calls.append(("finalize", (reservation_id, kwargs)))
            return {"status": "complete", "result_status": kwargs["result_status"]}

    def save(artifact, path):
        calls.append(("save", (artifact.status, Path(path))))
        return "9" * 64

    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.load_microstructure_model_artifact",
        lambda _path: candidate,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.microstructure_candidate_sha256",
        lambda _artifact: "c" * 64,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.evaluate_microstructure_model_terminal",
        lambda _artifact, _dataset, **_kwargs: validated,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.refit_validated_microstructure_model",
        lambda _artifact, _dataset, **_kwargs: shadow_candidate,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.save_microstructure_model_artifact",
        save,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_warehouse.MicrostructureWarehouse",
        Warehouse,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_features.build_executable_microstructure_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_features.apply_path_aware_lifecycle_targets",
        lambda _warehouse, value, **_kwargs: (value, SimpleNamespace()),
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_prequential.attach_verified_prequential_evidence",
        lambda _artifact, _dataset, **_kwargs: attached,
    )
    output = tmp_path / "shadow-candidate.json"

    result = cli.command_microstructure_promote(
        argparse.Namespace(
            input="candidate.json",
            output=str(output),
            prequential_report="report.json",
            prequential_predictions="predictions.csv",
            prequential_chart="chart.svg",
            warehouse="warehouse.duckdb",
            cache_root="cache",
            compute_backend="cpu",
            memory_limit="1GB",
            threads=1,
            json=False,
        )
    )

    assert result == 0
    reserve = next(value for name, value in calls if name == "reserve")
    assert reserve["candidate_sha256"] == "c" * 64
    assert reserve["prequential_report_sha256"] == "e" * 64
    finalize = next(value for name, value in calls if name == "finalize")
    assert finalize[1]["result_status"] == "validated"
    saved_statuses = [value[0] for name, value in calls if name == "save"]
    assert saved_statuses == [
        "candidate",
        "validated",
        "shadow_candidate",
        "shadow_candidate",
    ]
    output_text = capsys.readouterr().out
    assert "status=shadow_candidate" in output_text
    assert "trading_authority=false" in output_text
    assert "shadow_required=true" in output_text


def test_command_microstructure_shadow_is_no_order_and_saves_only_acceptance(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    refit = SimpleNamespace(expires_at_ms=9_999_999_999_999)
    artifact = SimpleNamespace(
        status="shadow_candidate",
        rejection_reasons=(),
        deployment_refit=refit,
        symbol="BTCUSDT",
    )
    accepted = SimpleNamespace(status="accepted")
    capture = SimpleNamespace(asdict=lambda: {"capture_id": "fixture"})
    report = SimpleNamespace(
        status="passed",
        symbol="BTCUSDT",
        replay={"decisions": 150, "orders_submitted": 0},
        metrics={"trades": 20, "total_net_bps": 25.0},
        trading_authority=True,
        reasons=(),
        passed=True,
        asdict=lambda: {
            "status": "passed",
            "trading_authority": True,
            "orders_submitted": 0,
        },
    )
    calls: dict[str, object] = {}

    class Store:
        def __init__(self, path: str) -> None:
            calls["db"] = path

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def record_microstructure_capture(self, payload) -> int:
            calls["catalog_payload"] = payload
            return 1

    def capture_feed(symbols, **kwargs):
        calls["symbols"] = symbols
        calls["capture_kwargs"] = kwargs
        return capture

    def evaluate(candidate, artifact_path, captured, **kwargs):
        calls["evaluate"] = (candidate, artifact_path, captured, kwargs)
        return report, accepted

    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.load_microstructure_model_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model.save_microstructure_model_artifact",
        lambda value, path: (
            calls.update({"saved": (value, Path(path))}) or "a" * 64
        ),
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_shadow.evaluate_shadow_capture",
        evaluate,
    )
    monkeypatch.setattr(cli, "capture_binance_futures_microstructure", capture_feed)
    monkeypatch.setattr(cli, "MarketDataStore", Store)
    output = tmp_path / "accepted.json"

    result = cli.command_microstructure_shadow(
        argparse.Namespace(
            input="shadow-candidate.json",
            output=str(output),
            seconds=21_660.0,
            output_root=str(tmp_path / "captures"),
            report=str(tmp_path / "report.json"),
            trades=str(tmp_path / "trades.csv"),
            db=str(tmp_path / "catalog.sqlite"),
            timeout=10.0,
            json=False,
        )
    )

    assert result == 0
    assert calls["symbols"] == ("BTCUSDT",)
    capture_kwargs = calls["capture_kwargs"]
    assert capture_kwargs["convert"] is False  # type: ignore[index]
    assert capture_kwargs["duration_seconds"] == 21_660.0  # type: ignore[index]
    assert "orders" not in capture_kwargs  # type: ignore[operator]
    assert calls["saved"] == (accepted, output)
    rendered = capsys.readouterr().out
    assert "trading_authority=true" in rendered
    assert "net_bps=+25.0000" in rendered


def test_tick_archive_full_history_plan_uses_independent_official_coverage(
    monkeypatch,
    capsys,
) -> None:
    def item(data_type: str, period: str, size: int):
        return SimpleNamespace(
            period=period,
            size_bytes=size,
            url=(
                "https://data.binance.vision/data/futures/um/daily/"
                f"{data_type}/BTCUSDT/BTCUSDT-{data_type}-{period}.zip"
            ),
        )

    listings = {
        "bookTicker": [
            item("bookTicker", "2023-05-16", 100),
            item("bookTicker", "2024-03-30", 200),
        ],
        "trades": [
            item("trades", "2019-09-08", 300),
            item("trades", "2026-07-09", 400),
        ],
    }
    monkeypatch.setattr(
        cli,
        "list_archive_items",
        lambda **kwargs: listings[kwargs["data_type"]],
    )

    result = cli.command_tick_archive_sync(
        argparse.Namespace(
            symbols="BTCUSDT",
            data_types="bookTicker,trades",
            start_date=None,
            end_date=None,
            full_history=True,
            available_only=False,
            plan_only=True,
            max_planned_gb=500.0,
            warehouse="unused.duckdb",
            cache_root="unused-cache",
            memory_limit="1GB",
            threads=1,
            timeout=10.0,
            no_retain_archive=True,
            json=True,
        )
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["full_history"] is True
    assert payload["available_only"] is True
    assert payload["planned_files"] == 4
    assert payload["missing"] == []
    by_type = {item["data_type"]: item for item in payload["coverage"]}
    assert by_type["bookTicker"]["selected_last_period"] == "2024-03-30"
    assert by_type["trades"]["selected_last_period"] == "2026-07-09"


def test_tape_depth_train_remains_research_only(tmp_path, monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}
    dataset = SimpleNamespace()
    metrics = SimpleNamespace(
        rows=10_000,
        direction_auc=0.60,
        spearman_information_coefficient=0.10,
        top_decile_signed_gross_bps=50.0,
    )
    artifact = SimpleNamespace(
        status="research_candidate",
        rejection_reasons=(),
        trading_authority=False,
        execution_claim=False,
        evaluation_metrics=metrics,
        asdict=lambda: {
            "status": "research_candidate",
            "trading_authority": False,
            "execution_claim": False,
            "model_strings": {"hidden": "value"},
        },
    )

    class Connection:
        def execute(self, _query, parameters):
            calls["latest_symbol"] = parameters[0]
            return self

        def fetchone(self):
            return (1_800_000_000_000,)

    class Warehouse:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def connect(self):
            return Connection()

    def build(_warehouse, **kwargs):
        calls["build"] = kwargs
        return dataset

    def train(value, **kwargs):
        calls["train"] = (value, kwargs)
        return artifact

    def save(value, path):
        calls["save"] = (value, Path(path))
        Path(path).write_text('{"research":true}\n', encoding="utf-8")

    monkeypatch.setattr(
        "simple_ai_trading.microstructure_warehouse.MicrostructureWarehouse",
        Warehouse,
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_features.build_tape_depth_forecast_dataset",
        build,
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_model.train_tape_depth_forecaster",
        train,
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_model.save_tape_depth_model_artifact",
        save,
    )
    output = tmp_path / "tape-depth.json"

    result = cli.command_tape_depth_train(
        argparse.Namespace(
            symbol="BTCUSDT",
            warehouse="ticks.duckdb",
            cache_root="cache",
            output=str(output),
            window_days=180,
            end_date=None,
            horizon_seconds=60,
            total_latency_ms=750,
            decision_cadence_seconds=5,
            maximum_depth_age_ms=60_000,
            risk_level="conservative",
            compute_backend="cpu",
            minimum_segment_rows=2_000,
            maximum_rows=5_000_000,
            memory_limit="1GB",
            threads=1,
            json=False,
        )
    )

    assert result == 0
    assert calls["latest_symbol"] == "BTCUSDT"
    assert calls["build"]["symbol"] == "BTCUSDT"  # type: ignore[index]
    assert calls["train"][0] is dataset  # type: ignore[index]
    assert calls["save"] == (artifact, output)
    rendered = capsys.readouterr().out
    assert "trading_authority=false" in rendered
    assert "execution_claim=false" in rendered


def test_command_report_renders_dashboard_and_readiness(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_readiness_report", lambda **_kwargs: (False, ["[fix] training data: missing"]))
    monkeypatch.setattr(cli, "_account_overview_lines", lambda _runtime: ["Account overview", "USDC: free=100 locked=0"])

    assert cli.command_report(
        argparse.Namespace(
            account=True,
            doctor=True,
            online=True,
            input="data/historical_market.json",
            model="data/model.json",
        )
    ) == 0
    output = capsys.readouterr().out
    assert "Session" in output
    assert "USDC: free=100 locked=0" in output
    assert "Readiness report (fix)" in output
    assert "secret-key" not in output
    assert "secret-value" not in output

    save_runtime(RuntimeConfig())
    assert cli.command_report(
        argparse.Namespace(
            account=True,
            doctor=False,
            online=False,
            input="data/historical_market.json",
            model="data/model.json",
        )
    ) == 2
    assert "Full report account section requires Binance API key" in capsys.readouterr().err

    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    monkeypatch.setattr(cli, "_account_overview_lines", lambda _runtime: ["Account balances failed: bad signature"])
    assert cli.command_report(
        argparse.Namespace(
            account=True,
            doctor=False,
            online=False,
            input="data/historical_market.json",
            model="data/model.json",
        )
    ) == 2
    assert "Account balances failed: bad signature" in capsys.readouterr().out

    plain = cli._render_operator_report(
        with_account=False,
        doctor=False,
        online=False,
        input_path="data/historical_market.json",
        model_path="data/model.json",
    )
    assert "Readiness report" not in plain


def test_command_audit_success_and_load_failure(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    class _Report:
        ok = True
        checks = ()
        raw_candles = 0
        clean_candles = 0
        feature_rows = 0
        duplicate_open_times = 0
        gap_count = 0
        max_feature_delta = 0.0

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: _simple_candles())
    monkeypatch.setattr("simple_ai_trading.audit.build_audit_report", lambda *_args, **_kwargs: _Report())
    monkeypatch.setattr("simple_ai_trading.audit.render_audit_report", lambda _report: "audit ok")

    assert cli.command_audit(argparse.Namespace(input="i.json", model="m.json")) == 0
    assert "audit ok" in capsys.readouterr().out

    class _BadReport(_Report):
        ok = False

    monkeypatch.setattr("simple_ai_trading.audit.build_audit_report", lambda *_args, **_kwargs: _BadReport())
    assert cli.command_audit(argparse.Namespace(input="i.json", model="m.json")) == 2

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: None)
    assert cli.command_audit(argparse.Namespace(input="bad.json", model="m.json")) == 2


def test_command_prepare_success_failure_and_validation(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="15m"))
    calls: list[tuple[str, argparse.Namespace]] = []

    def step(name: str, code: int = 0):
        def _step(args: argparse.Namespace) -> int:
            calls.append((name, args))
            return code

        return _step

    monkeypatch.setattr(cli, "command_fetch", step("fetch"))
    monkeypatch.setattr(cli, "command_train", step("train"))
    monkeypatch.setattr(cli, "command_evaluate", step("evaluate"))
    monkeypatch.setattr(cli, "command_backtest", step("backtest"))
    monkeypatch.setattr(cli, "command_audit", step("audit"))
    monkeypatch.setattr(cli, "command_doctor", step("doctor"))

    args = argparse.Namespace(
        historical=str(tmp_path / "history.json"),
        model=str(tmp_path / "model.json"),
        limit=25,
        preset="quick",
        epochs=9,
        seed=3,
        start_cash=500.0,
        online_doctor=True,
    )
    assert cli.command_prepare(args) == 0
    assert [name for name, _args in calls] == ["fetch", "train", "evaluate", "backtest", "audit", "doctor"]
    assert calls[1][1].preset == "custom"
    assert calls[1][1].requested_preset == "quick"
    assert calls[1][1].epochs == 9
    assert calls[1][1].learning_rate == 0.05
    assert calls[1][1].l2_penalty == 1e-4
    assert calls[1][1].walk_forward is False
    assert calls[2][1].calibrate_threshold is False
    assert calls[-2][1].model == str(tmp_path / "model.json")
    assert calls[-1][1].online is True

    calls.clear()
    monkeypatch.setattr(cli, "command_train", step("train", 2))
    assert cli.command_prepare(args) == 2
    assert [name for name, _args in calls] == ["fetch", "train"]
    assert "Prepare stopped at Train AI model" in capsys.readouterr().err

    bad = argparse.Namespace(**{**vars(args), "limit": 0})
    assert cli.command_prepare(bad) == 2
    assert "Prepare settings invalid" in capsys.readouterr().err
    for field, value in (
        ("batch_size", 0),
        ("epochs", 0),
        ("seed", -1),
        ("learning_rate", 0.0),
        ("l2_penalty", -0.1),
        ("start_cash", 0.0),
        ("walk_forward_train", 1),
        ("walk_forward_test", 0),
        ("walk_forward_step", 0),
    ):
        bad = argparse.Namespace(**{**vars(args), field: value})
        assert cli.command_prepare(bad) == 2
        assert "Prepare settings invalid" in capsys.readouterr().err


def test_training_preset_helper_and_invalid_command(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli._parse_optional_form_bool("") is None
    assert cli._parse_optional_form_bool("yes") is True
    assert cli._parse_optional_form_bool("no") is False
    with pytest.raises(ValueError, match="Expected yes/no/blank"):
        cli._parse_optional_form_bool("maybe")

    custom = cli._apply_training_preset(
        argparse.Namespace(
            preset="custom",
            epochs=999,
            walk_forward=True,
            walk_forward_train=1,
            walk_forward_test=1,
            walk_forward_step=1,
            calibrate_threshold=True,
        )
    )
    assert custom.epochs == 999
    quick = cli._apply_training_preset(argparse.Namespace(preset="quick", epochs=999, walk_forward=True, calibrate_threshold=True))
    assert quick.epochs == 80
    assert quick.walk_forward is False
    balanced = cli._apply_training_preset(argparse.Namespace(preset="balanced"))
    assert balanced.walk_forward is True
    assert balanced.walk_forward_test == 60
    thorough = cli._apply_training_preset(argparse.Namespace(preset="thorough"))
    assert thorough.epochs == 350
    assert thorough.walk_forward_train == 360
    with pytest.raises(ValueError):
        cli._apply_training_preset(argparse.Namespace(preset="wild"))

    assert cli.command_train(argparse.Namespace(input="x", output="y", preset="wild")) == 2
    assert "Training settings invalid" in capsys.readouterr().err


def test_clamp_and_direction_helpers() -> None:
    assert cli._clamp(1.2, 0.0, 1.0) == 1.0
    assert cli._clamp(-0.2, 0.0, 1.0) == 0.0
    cfg = StrategyConfig(signal_threshold=0.55)
    assert cli._score_to_direction(0.60, cfg, "spot") == 1
    assert cli._score_to_direction(0.40, cfg, "spot") == 0
    assert cli._score_to_direction(0.56, cfg, "futures") == 1
    assert cli._score_to_direction(0.40, cfg, "futures") == -1
    assert cli._score_to_direction(0.50, cfg, "futures") == 0
    assert cli._score_to_direction(0.50, cfg, "futures", threshold=0.50, short_threshold=0.50) == 0
    assert cli._score_to_direction(0.10, cfg, "futures") == -1
    assert cli._score_to_direction(
        0.10,
        cfg,
        "futures",
        threshold=0.55,
        short_threshold=None,
        side_thresholds_explicit=True,
    ) == 0


def test_resolve_futures_leverage(monkeypatch) -> None:
    runtime = RuntimeConfig(market_type="futures", api_key="k", api_secret="s", symbol="BTCUSDC")
    cfg = StrategyConfig(leverage=50.0)

    fake = _FakeClient()

    def build_client(_runtime):
        return fake

    monkeypatch.setattr(cli, "_build_client", build_client)
    assert cli._resolve_futures_leverage(runtime, cfg) == 10.0

    runtime_no_key = RuntimeConfig(market_type="futures", api_key="", api_secret="")
    assert cli._resolve_futures_leverage(runtime_no_key, cfg) == 20.0

    runtime_spot = RuntimeConfig(market_type="spot")
    assert cli._resolve_futures_leverage(runtime_spot, cfg) == 1.0


def test_entry_leverage_for_notional_uses_live_notional_bracket() -> None:
    class _BracketClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float]] = []

        def get_max_leverage_for_notional(self, symbol: str, notional: float) -> int:
            self.calls.append((symbol, notional))
            return 8

    client = _BracketClient()
    runtime = RuntimeConfig(market_type="futures", symbol="BTCUSDC")

    assert cli._entry_leverage_for_notional(client, runtime, 20.0, 2_500.0, effective_dry_run=False) == 8.0
    assert client.calls == [("BTCUSDC", 2_500.0)]
    assert cli._entry_leverage_for_notional(client, runtime, 20.0, 2_500.0, effective_dry_run=True) == 20.0
    assert client.calls == [("BTCUSDC", 2_500.0)]
    assert cli._entry_leverage_for_notional(client, RuntimeConfig(market_type="spot"), 20.0, 2_500.0, effective_dry_run=False) == 1.0
    with pytest.raises(BinanceAPIError, match="positive entry notional"):
        cli._entry_leverage_for_notional(client, runtime, 20.0, 0.0, effective_dry_run=False)


def test_build_client_forwards_runtime_request_window(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli, "BinanceClient", _Client)
    cli._build_client(RuntimeConfig(api_key="k", api_secret="s", recv_window_ms=9000, demo=True))
    assert captured["recv_window_ms"] == 9000
    assert captured["max_calls_per_minute"] == 1100
    assert captured["demo"] is True


def test_command_api_budget_auto_refreshes_stale_cache(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "market.sqlite"
    with MarketDataStore(db) as store:
        store.insert_api_rate_limit_snapshot(
            "binance",
            "spot",
            {
                "status": "ok",
                "generated_at_ms": 1_000,
                "market_type": "spot",
                "lines": [],
            },
            ts_ms=1_000,
        )
    save_runtime(RuntimeConfig(market_type="spot"))

    class BudgetClient:
        last_request_info = {"rate_limit_headers": {"X-MBX-USED-WEIGHT-1M": "12"}}

        def get_exchange_info(self):
            return {"rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1200}]}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: BudgetClient())
    monkeypatch.setattr(cli.time, "time", lambda: 1_000_000.0)

    args = argparse.Namespace(db=str(db), market=None, refresh=False, cached_only=False, max_age_seconds=90, compact=True, json=False)
    assert cli.command_api_budget(args) == 0

    assert "remaining=1188/1200" in capsys.readouterr().out
    with MarketDataStore(db) as store:
        latest = store.latest_api_rate_limit_snapshot("binance", "spot")
    assert latest is not None
    assert "1188" in str(latest)


def test_command_archive_sync_delegates_archive_listing_and_ingestion(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))
    monkeypatch.setattr(cli, "list_archive_urls", lambda **_kwargs: ["https://data.binance.vision/x/BTCUSDC-1s-2026-01.zip"])

    captured: dict[str, object] = {}

    class Result:
        status = "complete"
        rows_read = 2
        rows_inserted = 2
        bytes_downloaded = 123
        url = "https://data.binance.vision/x/BTCUSDC-1s-2026-01.zip"
        error = ""

        def asdict(self):
            return {
                "status": self.status,
                "rows_read": self.rows_read,
                "rows_inserted": self.rows_inserted,
                "bytes_downloaded": self.bytes_downloaded,
                "url": self.url,
            }

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return [Result()]

    monkeypatch.setattr(cli, "ingest_archive_urls", fake_ingest)
    args = argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        interval=None,
        market="spot",
        cadence="monthly",
        max_files=None,
        timeout=120,
        force=False,
        json=False,
    )

    assert cli.command_archive_sync(args) == 0

    assert captured["symbol"] == "BTCUSDC"
    assert captured["interval"] == "1s"
    assert captured["data_type"] == "klines"
    assert "rows_inserted=2" in capsys.readouterr().out


def test_command_archive_sync_defaults_futures_one_second_to_agg_trades(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDT", interval="1s", market_type="futures", quote_asset="USDT"))
    list_kwargs: dict[str, object] = {}
    ingest_kwargs: dict[str, object] = {}

    class Result:
        status = "complete"
        rows_read = 2
        rows_inserted = 2
        bytes_downloaded = 123
        url = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip"
        error = ""

        def asdict(self):
            return {
                "status": self.status,
                "rows_read": self.rows_read,
                "rows_inserted": self.rows_inserted,
                "bytes_downloaded": self.bytes_downloaded,
                "url": self.url,
                "data_type": "aggTrades",
            }

    def fake_list(**kwargs):
        list_kwargs.update(kwargs)
        return [Result.url]

    def fake_ingest(**kwargs):
        ingest_kwargs.update(kwargs)
        return [Result()]

    monkeypatch.setattr(cli, "list_archive_urls", fake_list)
    monkeypatch.setattr(cli, "ingest_archive_urls", fake_ingest)

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval=None,
        market="futures",
        cadence="daily",
        data_type=None,
        max_files=1,
        timeout=120,
        force=False,
        json=False,
    )) == 0

    assert list_kwargs["data_type"] == "aggTrades"
    assert ingest_kwargs["data_type"] == "aggTrades"
    assert "data_type=aggTrades" in capsys.readouterr().out


def test_command_archive_sync_plan_only_filters_period_window(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDT", interval="1s", market_type="futures", quote_asset="USDT"))
    listed = [
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-05-31.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-02.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-07-01.zip",
    ]
    sizes = {url: (index + 1) * 100 for index, url in enumerate(listed)}
    monkeypatch.setattr(cli, "list_archive_urls", lambda **_kwargs: listed)
    monkeypatch.setattr(
        cli,
        "archive_listing_items_by_url",
        lambda urls: {url: SimpleNamespace(size_bytes=sizes[url]) for url in urls if url in sizes},
    )
    monkeypatch.setattr(
        cli,
        "ingest_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("plan-only should not ingest")),
    )

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval=None,
        market="futures",
        cadence="daily",
        data_type=None,
        max_files=None,
        start_period="2024-06-01",
        end_period="2024-06-30",
        plan_only=True,
        timeout=120,
        force=False,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["plan_only"] is True
    assert payload["planned_files"] == 2
    assert payload["planned_bytes"] == 500
    assert payload["files"] == 0
    assert payload["archive_plans"][0]["listed_files"] == 4
    assert payload["archive_plans"][0]["listed_bytes"] == 1000
    assert payload["archive_plans"][0]["filtered_files"] == 2
    assert payload["archive_plans"][0]["filtered_bytes"] == 500
    assert payload["archive_plans"][0]["selected_bytes"] == 500
    assert payload["archive_plans"][0]["size_metadata_available"] is True
    assert payload["archive_plans"][0]["first_period"] == "2024-06-01"
    assert payload["archive_plans"][0]["last_period"] == "2024-06-02"


def test_command_archive_sync_rejects_invalid_period_window_before_listing(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDT", interval="1s", market_type="futures", quote_asset="USDT"))
    monkeypatch.setattr(
        cli,
        "list_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("invalid window should not list")),
    )

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval=None,
        market="futures",
        cadence="daily",
        data_type=None,
        max_files=None,
        start_period="2024/06/01",
        end_period="2024-06-30",
        plan_only=True,
        timeout=120,
        force=False,
        json=True,
    )) == 2

    assert "invalid period window" in capsys.readouterr().err


def test_command_archive_sync_blocks_oversized_non_plan_backfill(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDT", interval="1s", market_type="futures", quote_asset="USDT"))
    listed = [
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-02.zip",
    ]
    sizes = {url: 40_000_000 for url in listed}
    monkeypatch.setattr(cli, "list_archive_urls", lambda **_kwargs: listed)
    monkeypatch.setattr(
        cli,
        "archive_listing_items_by_url",
        lambda urls: {url: SimpleNamespace(size_bytes=sizes[url]) for url in urls if url in sizes},
    )
    monkeypatch.setattr(
        cli,
        "ingest_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("oversized plan should not ingest")),
    )

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval=None,
        market="futures",
        cadence="daily",
        data_type=None,
        max_files=None,
        start_period="2024-06-01",
        end_period="2024-06-02",
        plan_only=False,
        max_planned_gb=0.05,
        timeout=120,
        force=False,
        json=True,
    )) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_bytes"] == 80_000_000
    assert payload["files"] == 0
    assert payload["errors"] == [
        {"symbol": "*", "error": "planned_bytes_exceeds_max:80000000/50000000"}
    ]


def test_command_archive_sync_accepts_explicit_symbol_batch(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))
    list_calls: list[str] = []
    ingest_calls: list[str] = []

    class Result:
        status = "complete"
        rows_read = 1
        rows_inserted = 1
        bytes_downloaded = 10
        error = ""

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.url = f"https://data.binance.vision/x/{symbol}-1s-2026-01.zip"

        def asdict(self):
            return {
                "status": self.status,
                "symbol": self.symbol,
                "rows_read": self.rows_read,
                "rows_inserted": self.rows_inserted,
                "bytes_downloaded": self.bytes_downloaded,
                "url": self.url,
            }

    def fake_list(**kwargs):
        list_calls.append(str(kwargs["symbol"]))
        return [f"https://data.binance.vision/x/{kwargs['symbol']}-1s-2026-01.zip"]

    def fake_ingest(**kwargs):
        ingest_calls.append(str(kwargs["symbol"]))
        return [Result(str(kwargs["symbol"]))]

    monkeypatch.setattr(cli, "list_archive_urls", fake_list)
    monkeypatch.setattr(cli, "ingest_archive_urls", fake_ingest)

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols="btcusdc, ethusdc",
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval="1s",
        market="spot",
        cadence="monthly",
        max_files=1,
        timeout=120,
        force=False,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["BTCUSDC", "ETHUSDC"]
    assert payload["symbol_count"] == 2
    assert payload["rows_inserted"] == 2
    assert list_calls == ["BTCUSDC", "ETHUSDC"]
    assert ingest_calls == ["BTCUSDC", "ETHUSDC"]


def test_command_archive_sync_rejects_non_major_symbols(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))
    monkeypatch.setattr(
        cli,
        "list_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported symbol should not be listed")),
    )

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols="ALTUSDC",
        top_symbols=0,
        quote_asset=None,
        max_scan=250,
        interval="1s",
        market="spot",
        cadence="monthly",
        max_files=1,
        timeout=120,
        force=False,
        json=True,
    )) == 2

    assert "supports only BTC, ETH, and SOL" in capsys.readouterr().err


def test_command_archive_sync_can_auto_rank_high_liquidity_symbols(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))
    ranked: dict[str, object] = {}

    class Result:
        status = "skipped"
        rows_read = 0
        rows_inserted = 0
        bytes_downloaded = 0
        error = ""

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.url = f"https://data.binance.vision/x/{symbol}-1s-2026-01.zip"

        def asdict(self):
            return {"status": self.status, "symbol": self.symbol, "url": self.url}

    def fake_select(client, strategy, *, quote_asset, count, max_scan, strict_only):
        assert client.testnet is False
        assert client.market_type == "spot"
        ranked.update({
            "client": client,
            "risk_level": strategy.risk_level,
            "quote_asset": quote_asset,
            "count": count,
            "max_scan": max_scan,
            "strict_only": strict_only,
        })
        return [SimpleNamespace(symbol="BTCUSDT"), SimpleNamespace(symbol="ETHUSDT")]

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli, "select_top_liquidity_symbols", fake_select)
    monkeypatch.setattr(cli, "list_archive_urls", lambda **kwargs: [f"https://data.binance.vision/x/{kwargs['symbol']}.zip"])
    monkeypatch.setattr(cli, "ingest_archive_urls", lambda **kwargs: [Result(str(kwargs["symbol"]))])

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=2,
        quote_asset="USDT",
        max_scan=50,
        interval="1s",
        market="spot",
        cadence="daily",
        max_files=1,
        timeout=120,
        force=False,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["files"] == 2
    assert ranked["quote_asset"] == "USDT"
    assert ranked["count"] == 50
    assert ranked["max_scan"] == 50
    assert ranked["strict_only"] is True


def test_command_archive_sync_filters_auto_ranked_symbols_by_history_depth(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1m", market_type="futures"))
    ingested: list[str] = []

    class Result:
        status = "skipped"
        rows_read = 0
        rows_inserted = 0
        bytes_downloaded = 0
        error = ""

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.url = f"https://data.binance.vision/x/{symbol}.zip"

        def asdict(self):
            return {"status": self.status, "symbol": self.symbol, "url": self.url}

    def fake_select(client, _strategy, *, quote_asset, count, max_scan, strict_only):
        assert client.testnet is False
        assert client.market_type == "futures"
        assert quote_asset == "USDT"
        assert count == 10
        assert max_scan == 10
        assert strict_only is True
        return [
            SimpleNamespace(symbol="BTCUSDT"),
            SimpleNamespace(symbol="SOLUSDT"),
            SimpleNamespace(symbol="ETHUSDT"),
        ]

    def fake_list(**kwargs):
        symbol = str(kwargs["symbol"])
        counts = {"BTCUSDT": 3, "SOLUSDT": 1, "ETHUSDT": 2}
        return [f"https://data.binance.vision/x/{symbol}-{index}.zip" for index in range(counts[symbol])]

    def fake_ingest(**kwargs):
        symbol = str(kwargs["symbol"])
        ingested.append(symbol)
        return [Result(symbol)]

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli, "select_top_liquidity_symbols", fake_select)
    monkeypatch.setattr(cli, "list_archive_urls", fake_list)
    monkeypatch.setattr(cli, "ingest_archive_urls", fake_ingest)

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=2,
        quote_asset="USDT",
        max_scan=10,
        min_history_months=2,
        interval="1m",
        market="futures",
        cadence="monthly",
        max_files=None,
        timeout=120,
        force=False,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["requested_top_symbols"] == 2
    assert payload["history_rejections"] == [
        {"symbol": "SOLUSDT", "error": "history_months_below_min:1/2"}
    ]
    assert ingested == ["BTCUSDT", "ETHUSDT"]


def test_command_archive_sync_max_files_zero_dry_runs_ranked_symbols(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1m", market_type="futures"))

    def fake_select(client, _strategy, *, quote_asset, count, max_scan, strict_only):
        assert client.testnet is False
        assert client.market_type == "futures"
        assert quote_asset == "USDT"
        assert count == 2
        assert max_scan == 2
        assert strict_only is True
        return [
            SimpleNamespace(symbol="BTCUSDT"),
            SimpleNamespace(symbol="ETHUSDT"),
        ]

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli, "select_top_liquidity_symbols", fake_select)
    monkeypatch.setattr(cli, "list_archive_urls", lambda **kwargs: [f"https://data.binance.vision/x/{kwargs['symbol']}.zip"])
    monkeypatch.setattr(
        cli,
        "ingest_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dry run should not ingest")),
    )

    assert cli.command_archive_sync(argparse.Namespace(
        db=str(tmp_path / "m.sqlite"),
        symbol=None,
        symbols=None,
        top_symbols=2,
        quote_asset="USDT",
        max_scan=2,
        min_history_months=1,
        interval="1m",
        market="futures",
        cadence="monthly",
        max_files=0,
        timeout=120,
        force=False,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["files"] == 0
    assert payload["errors"] == []


def test_command_data_health_reports_verified_archive_coverage(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "market.sqlite"
    candles = [
        Candle(
            open_time=index * 1000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            close_time=index * 1000 + 999,
        )
        for index in range(3)
    ]
    with MarketDataStore(db) as store:
        store.upsert_candles("BTCUSDC", "spot", "1s", candles, source="binance_public_archive")
        url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip"
        store.begin_archive_file(url=url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01-01")
        store.complete_archive_file(
            url=url,
            status="complete",
            rows_inserted=3,
            bytes_downloaded=123,
            sha256="a" * 64,
            checksum_sha256="a" * 64,
            checksum_status="verified",
        )
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))

    assert cli.command_data_health(argparse.Namespace(
        db=str(db),
        symbol=None,
        symbols=None,
        interval="1s",
        market="spot",
        min_rows=3,
        min_coverage_ratio=1.0,
        max_gap_count=0,
        require_verified_checksum=True,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["items"][0]["checksum_status_counts"] == {"verified": 1}
    assert payload["items"][0]["coverage_ratio"] == 1.0


def test_command_data_health_warns_when_archive_error_is_superseded_by_verified_coverage(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    db = tmp_path / "market.sqlite"
    candles = [
        Candle(
            open_time=index * 1000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            close_time=index * 1000 + 999,
        )
        for index in range(3)
    ]
    with MarketDataStore(db) as store:
        store.upsert_candles("BTCUSDC", "spot", "1s", candles, source="binance_public_archive")
        good_url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip"
        bad_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip"
        store.begin_archive_file(url=good_url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01-01")
        store.complete_archive_file(
            url=good_url,
            status="complete",
            rows_inserted=3,
            bytes_downloaded=123,
            sha256="a" * 64,
            checksum_sha256="a" * 64,
            checksum_status="verified",
        )
        store.begin_archive_file(url=bad_url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01")
        store.complete_archive_file(
            url=bad_url,
            status="error",
            rows_inserted=0,
            bytes_downloaded=0,
            sha256="",
            checksum_status="missing",
            error="missing checksum sidecar",
        )
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))

    assert cli.command_data_health(argparse.Namespace(
        db=str(db),
        symbol="BTCUSDC",
        symbols=None,
        interval="1s",
        market="spot",
        min_rows=3,
        min_coverage_ratio=1.0,
        max_gap_count=0,
        require_verified_checksum=True,
        json=True,
    )) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["reasons"] == []
    assert payload["items"][0]["warnings"] == ["superseded_archive_errors:1"]


def test_command_data_health_blocks_missing_rows_gaps_and_checksum(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "market.sqlite"
    candles = [
        Candle(open_time=0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0, close_time=999),
        Candle(open_time=3000, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0, close_time=3999),
    ]
    with MarketDataStore(db) as store:
        store.upsert_candles("BTCUSDC", "spot", "1s", candles, source="binance_public_archive")
        url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip"
        store.begin_archive_file(url=url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01-01")
        store.complete_archive_file(url=url, status="complete", rows_inserted=2, bytes_downloaded=123, sha256="a" * 64)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(symbol="BTCUSDC", interval="1s", market_type="spot"))

    assert cli.command_data_health(argparse.Namespace(
        db=str(db),
        symbol="BTCUSDC",
        symbols=None,
        interval="1s",
        market="spot",
        min_rows=3,
        min_coverage_ratio=1.0,
        max_gap_count=0,
        require_verified_checksum=True,
        json=True,
    )) == 2

    payload = json.loads(capsys.readouterr().out)
    reasons = payload["items"][0]["reasons"]
    assert "rows_below_min:2/3" in reasons
    assert "gap_count_above_max:2/0" in reasons
    assert "no_verified_archive_checksum" in reasons


def test_live_startup_blocks_when_cached_api_budget_is_over_eighty_percent(tmp_path, monkeypatch) -> None:
    db = tmp_path / "market.sqlite"
    now_ms = 1_000_000
    with MarketDataStore(db) as store:
        store.insert_api_rate_limit_snapshot(
            "binance",
            "spot",
            {
                "status": "critical",
                "generated_at_ms": now_ms,
                "market_type": "spot",
                "retry_after_seconds": None,
                "lines": [
                    {
                        "rate_limit_type": "REQUEST_WEIGHT",
                        "interval_num": 1,
                        "interval_letter": "M",
                        "interval_label": "1M",
                        "interval_ms": 60_000,
                        "used": 960,
                        "limit": 1200,
                        "remaining": 240,
                        "remaining_pct": 0.2,
                        "status": "ok",
                    }
                ],
            },
            ts_ms=now_ms,
        )
    runtime = RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=True, market_type="spot")
    client = _FakeClient()
    monkeypatch.setattr(cli.time, "time", lambda: now_ms / 1000.0)

    with pytest.raises(BinanceAPIError, match="blocked startup"):
        cli._ensure_api_budget_startup_safe(runtime, client, db_path=db)


def test_validate_runtime_connection_skips_account_without_keys() -> None:
    class _Client:
        def __init__(self) -> None:
            self.account_calls = 0

        def ping(self):
            return {"ok": True}

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_account(self):  # pragma: no cover - assertion checks it stays unused
            self.account_calls += 1

    client = _Client()
    cli._validate_runtime_connection(RuntimeConfig(api_key="", api_secret=""), client)
    assert client.account_calls == 0


def test_connection_status_line_branches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    class NoAuthClient(_FakeClient):
        pass

    save_runtime(RuntimeConfig(api_key="", api_secret="", dry_run=True, testnet=True, market_type="spot"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: NoAuthClient())
    line = cli._connection_status_line()
    assert "public endpoint reachable spot/testnet paper-default" in line
    assert "credentials missing" in line

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=True, market_type="futures"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    line = cli._connection_status_line()
    assert "authenticated futures/testnet testnet-live-default" in line

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=False, demo=True, market_type="spot"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    assert "authenticated spot/demo demo-live-default" in cli._connection_status_line()

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=False, demo=False, market_type="spot"))
    assert "signed validation locked" in cli._connection_status_line()

    class OddAuthClient(_FakeClient):
        def get_account(self):
            return "accepted"

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=True, market_type="spot"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OddAuthClient())
    assert "auth response ok" in cli._connection_status_line()

    class OfflineClient(_FakeClient):
        def ping(self):
            raise BinanceAPIError("timeout")

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=True, market_type="futures"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OfflineClient())
    assert "public endpoint unreachable futures/testnet" in cli._connection_status_line()

    class AuthFailClient(_FakeClient):
        def get_account(self):
            raise BinanceAPIError("bad key")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: AuthFailClient())
    assert "authentication failed futures/testnet" in cli._connection_status_line()


def test_readiness_report_and_command_doctor(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=False, api_key="", api_secret=""))
    save_strategy(StrategyConfig())
    ok, lines = cli._readiness_report(input_path=str(tmp_path / "missing.json"), model_path=str(tmp_path / "missing-model.json"))
    assert ok is False
    assert any(line.startswith("[fix] safety target") for line in lines)
    assert any("training data" in line for line in lines)

    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text("[]", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, api_key="", api_secret=""))
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()] * 80)
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: SimpleNamespace(feature_dim=3))
    monkeypatch.setattr(cli, "_connection_status_line", lambda: "Connection 00: public endpoint reachable spot/testnet paper-default; credentials missing")
    assert cli.command_doctor(argparse.Namespace(input=str(data_file), model=str(model_file), online=True)) == 2
    output = capsys.readouterr().out
    assert "Readiness report" in output
    assert "[fix] exchange connectivity" in output

    monkeypatch.setattr(
        cli,
        "_load_runtime_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            feature_dim=3,
            quality_score=0.30,
            quality_warnings=["weak validation"],
        ),
    )
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("[fix] model quality" in line and "weak validation" in line for line in lines)

    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModelLoadError("bad model")))
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("bad model" in line for line in lines)


def test_readiness_report_includes_feature_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True))
    save_strategy(StrategyConfig())
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text("[]", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    drift_row = SimpleNamespace(features=(9.0,), label=1)
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()] * 80)
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [drift_row])
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: model)

    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)

    assert ok is False
    assert any("[fix] feature drift" in line and "hard threshold" in line for line in lines)

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: None)
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("[ok] model artifact" in line for line in lines)
    assert not any("feature drift" in line for line in lines)

    model.probability_brier_after = 0.20
    model.probability_ece_after = 0.12
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert any("[ok] probability calibration" in line and "ece=0.120" in line for line in lines)

    model.probability_brier_after = 0.42
    model.probability_ece_after = None
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("[fix] probability calibration" in line and "brier=0.420" in line for line in lines)

    model.probability_brier_after = 0.20
    model.probability_ece_after = 0.24
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("[fix] probability calibration" in line and "ece=0.240" in line for line in lines)


def test_readiness_report_accepts_train_suite_advanced_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True))
    strategy = StrategyConfig()
    save_strategy(strategy)

    candles = _simple_candles(320)
    feature_cfg = default_config_for("default", strategy.enabled_features)
    rows = make_advanced_rows(candles, feature_cfg)
    assert rows
    columns = list(zip(*(row.features for row in rows), strict=True))
    means = [sum(values) / len(values) for values in columns]
    stds = [
        max(1e-6, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))
        for values, mean in zip(columns, means, strict=True)
    ]

    model = TrainedModel(
        weights=[0.0] * len(rows[0].features),
        bias=0.0,
        feature_dim=len(rows[0].features),
        epochs=1,
        feature_means=means,
        feature_stds=stds,
        feature_signature=advanced_feature_signature(feature_cfg),
        strategy_overrides={"risk_per_trade": 0.005, "signal_threshold": 0.64},
    )

    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model_default.json"
    data_file.write_text("[]", encoding="utf-8")
    serialize_model(model, model_file)
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: candles)

    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)

    assert ok is True
    assert any("[ok] model artifact" in line and "kind=advanced:regular" in line for line in lines)
    assert any("[ok] model strategy overlay" in line and "risk=0.0050" in line for line in lines)
    assert any("[ok] feature drift" in line for line in lines)


def test_readiness_model_rejects_unknown_advanced_signature(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = StrategyConfig()
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        feature_signature="advanced_version=unknown",
    )
    model_file = tmp_path / "model_unknown.json"
    serialize_model(model, model_file)

    assert cli._advanced_objective_for_model(model, strategy) is None

    monkeypatch.setattr(
        cli,
        "_load_runtime_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ModelFeatureMismatchError("runtime mismatch")),
    )
    with pytest.raises(ModelFeatureMismatchError, match="runtime mismatch"):
        cli._load_readiness_model(model_file, strategy)


def test_cli_recognizes_model_lab_candidate_advanced_signature() -> None:
    strategy = StrategyConfig()
    candles = _simple_candles(320)
    feature_cfg = replace(
        default_config_for("regular", strategy.enabled_features),
        label_lookahead=7,
        label_threshold=0.00168,
        label_stop_threshold=0.00168,
        confluence_windows=(5, 13, 34),
    )
    rows = make_advanced_rows(candles, feature_cfg)
    assert rows
    dim = len(rows[0].features)
    model = TrainedModel(
        weights=[0.0] * dim,
        bias=0.0,
        feature_dim=dim,
        epochs=1,
        feature_means=[0.0] * dim,
        feature_stds=[1.0] * dim,
        feature_signature=advanced_feature_signature(feature_cfg),
    )

    recognized = cli._advanced_objective_for_model(model, strategy)
    backtest_rows = cli._backtest_rows_for_model(candles, strategy, model)

    assert recognized == ("custom", feature_cfg)
    assert backtest_rows
    assert len(backtest_rows[-1].features) == dim


def test_live_helpers_accept_train_suite_advanced_model(tmp_path) -> None:
    strategy = StrategyConfig()
    candles = _simple_candles(320)
    feature_cfg = default_config_for("default", strategy.enabled_features)
    advanced_rows = make_advanced_rows(candles, feature_cfg)
    assert advanced_rows

    model = TrainedModel(
        weights=[0.0] * len(advanced_rows[0].features),
        bias=0.0,
        feature_dim=len(advanced_rows[0].features),
        epochs=1,
        feature_means=[0.0] * len(advanced_rows[0].features),
        feature_stds=[1.0] * len(advanced_rows[0].features),
        feature_signature=advanced_feature_signature(feature_cfg),
        selection_risk={
            "passed": True,
            "effective_trials": 24,
            "selected_score": 0.12,
            "trial_penalty": 0.01,
            "deflated_score": 0.11,
        },
        execution_validation=_promoted_execution_validation(market_type="spot", interval="1s"),
        probability_calibration_size=128,
        probability_log_loss_before=0.62,
        probability_log_loss_after=0.58,
        probability_brier_before=0.24,
        probability_brier_after=0.22,
        probability_ece_before=0.10,
        probability_ece_after=0.08,
        probability_calibration_backend_requested="directml",
        probability_calibration_backend_kind="directml",
        probability_calibration_backend_device="privateuseone:0",
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
        training_backend_vendor="DirectML",
        model_candidate_count=3,
        model_selected_candidate="triple_barrier_base",
        model_selection_score=0.42,
        strategy_overrides={"taker_fee_bps": 4.0},
    )
    model_file = tmp_path / "model_default.json"
    serialize_model(model, model_file)

    loaded, error, notice = cli._load_live_start_model(model_file, strategy, effective_dry_run=False)

    assert error is None
    assert notice is None
    assert loaded is not None
    assert loaded.feature_signature == model.feature_signature
    live_rows = cli._live_rows_for_model(candles, strategy, loaded)
    assert live_rows
    assert len(live_rows[0].features) == loaded.feature_dim
    assert cli._live_model_feature_signature(loaded, strategy) == model.feature_signature

    base_rows = cli._live_rows_for_model(candles, strategy, None)
    assert base_rows
    assert len(base_rows[0].features) == len(strategy.enabled_features)
    assert cli._live_model_feature_signature(None, strategy) == cli._strategy_feature_signature(strategy)


def test_evaluate_and_backtest_accept_train_suite_advanced_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    strategy = StrategyConfig()
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(strategy)
    candles = _simple_candles(320)
    feature_cfg = default_config_for("default", strategy.enabled_features)
    rows = make_advanced_rows(candles, feature_cfg)
    assert rows
    data_file = tmp_path / "history.json"
    data_file.write_text(
        json.dumps([
            {
                "open_time": candle.open_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "close_time": candle.close_time,
            }
            for candle in candles
        ]),
        encoding="utf-8",
    )
    model_file = tmp_path / "model_default.json"
    serialize_model(
        TrainedModel(
            weights=[0.0] * len(rows[0].features),
            bias=0.0,
            feature_dim=len(rows[0].features),
            epochs=1,
            feature_means=[0.0] * len(rows[0].features),
            feature_stds=[1.0] * len(rows[0].features),
            feature_signature=advanced_feature_signature(feature_cfg),
            strategy_overrides={"risk_per_trade": 0.005, "signal_threshold": 0.64},
        ),
        model_file,
    )

    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(data_file),
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 0
    assert cli.command_backtest(
        argparse.Namespace(input=str(data_file), model=str(model_file), start_cash=1000.0)
    ) == 0


def test_backtest_rows_include_latest_inference_window() -> None:
    cfg = StrategyConfig(feature_windows=(3, 5))
    candles = _simple_candles(80)
    model = SimpleNamespace(feature_signature="")
    readiness_rows = cli._readiness_model_rows(candles, cfg, model)
    backtest_rows = cli._backtest_rows_for_model(candles, cfg, model)

    assert readiness_rows
    assert backtest_rows
    assert backtest_rows[-1].timestamp > readiness_rows[-1].timestamp
    assert len(backtest_rows) >= len(readiness_rows)


def test_command_evaluate_reports_no_rows_after_model_specific_row_build(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    row = ModelRow(1, 100.0, (0.0,), 1)
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_load_readiness_model", lambda *_args, **_kwargs: (model, "runtime"))
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_args, **_kwargs: [])

    assert cli.command_evaluate(
        argparse.Namespace(
            input="history.json",
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 2
    assert "No rows available" in capsys.readouterr().out


def test_command_live_applies_model_strategy_overrides_before_risk_policy(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.02, signal_threshold=0.58, take_profit_pct=0.03))
    model = TrainedModel(
        weights=[0.0] * 13,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
        strategy_overrides={
            "risk_per_trade": 0.005,
            "signal_threshold": 0.64,
            "take_profit_pct": 0.04,
        },
    )
    captured: dict[str, float] = {}

    def fake_policy(runtime, strategy, **kwargs):
        captured["risk"] = strategy.risk_per_trade
        captured["threshold"] = strategy.signal_threshold
        captured["take"] = strategy.take_profit_pct
        return SimpleNamespace(allowed=True, warning_count=0)

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (model, None, None))
    monkeypatch.setattr(cli, "_resolve_symbol_constraints", lambda *_args, **_kwargs: SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 5.0, 1000.0))
    monkeypatch.setattr(cli, "build_risk_policy_report", fake_policy)
    monkeypatch.setattr(cli, "build_live_run_payload", lambda **kwargs: {"events": []})
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda *_args, **_kwargs: tmp_path / "live.json")

    assert cli.command_live(
        argparse.Namespace(
            steps=0,
            sleep=0,
            paper=True,
            live=False,
            model=str(tmp_path / "model.json"),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 0

    assert captured == {"risk": 0.005, "threshold": 0.64, "take": 0.04}


def test_load_live_start_model_handles_unexpected_model_loader_failure(tmp_path, monkeypatch) -> None:
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_readiness_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    model, error, notice = cli._load_live_start_model(model_file, StrategyConfig(), effective_dry_run=True)
    assert model is None
    assert error is None
    assert notice is None

    model, error, notice = cli._load_live_start_model(model_file, StrategyConfig(), effective_dry_run=False)
    assert model is None
    assert "requires a readable model" in str(error)
    assert notice is None


def test_load_live_start_model_rejects_unpromoted_model_for_signed_mode(tmp_path) -> None:
    model_file = tmp_path / "model.json"
    serialize_model(
        TrainedModel(
            weights=[0.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
            feature_signature=cli._strategy_feature_signature(StrategyConfig(enabled_features=("momentum_1",))),
        ),
        model_file,
    )
    strategy = StrategyConfig(enabled_features=("momentum_1",))

    model, error, notice = cli._load_live_start_model(model_file, strategy, effective_dry_run=False)
    assert model is None
    assert "requires a promoted model" in str(error)
    assert notice is None

    model, error, notice = cli._load_live_start_model(model_file, strategy, effective_dry_run=True)
    assert model is not None
    assert error is None
    assert "promotion warning" in str(notice)


def test_load_live_start_model_can_require_live_grade_candidate_and_gpu_evidence(tmp_path) -> None:
    strategy = StrategyConfig(enabled_features=("momentum_1",))
    model_file = tmp_path / "model.json"
    serialize_model(
        TrainedModel(
            weights=[0.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
            feature_signature=cli._strategy_feature_signature(strategy),
            selection_risk={
                "passed": True,
                "effective_trials": 12,
                "selected_score": 0.12,
                "trial_penalty": 0.03,
                "deflated_score": 0.09,
            },
            execution_validation=_promoted_execution_validation(market_type="futures", interval="1s"),
            probability_calibration_size=128,
            probability_log_loss_before=0.62,
            probability_log_loss_after=0.58,
            probability_brier_before=0.24,
            probability_brier_after=0.22,
            probability_ece_before=0.10,
            probability_ece_after=0.08,
        ),
        model_file,
    )

    model, error, notice = cli._load_live_start_model(
        model_file,
        strategy,
        effective_dry_run=False,
        require_model_candidate_search=True,
        require_accelerator_evidence=True,
    )

    assert model is None
    assert "model candidate search" in str(error)
    assert "training accelerator" in str(error)
    assert notice is None


def test_build_order_notional_paths() -> None:
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.4,
        max_asset_allocation_pct=1.0,
        leverage=3.0,
    )
    client = SimpleNamespace(
        normalize_quantity=lambda symbol, qty: (
            0.0 if qty < 0.001 else qty,
            type(
                "Constraint",
                (),
                {"symbol": "BTCUSDC", "min_qty": 0.001, "step_size": 0.001, "min_notional": 5.0, "max_notional": 300.0},
            )(),
        ),
    )
    notional, qty = cli._build_order_notional(1000.0, 10_000.0, cfg, "futures", 3.0, client)
    assert math.isclose(notional, 1000.0)
    assert math.isclose(qty, 0.10)

    bad_constraints = type(
        "C",
        (),
        {"symbol": "BTCUSDC", "min_qty": 10.0, "step_size": 1.0, "min_notional": 5000.0, "max_notional": 0.0},
    )()

    client_too_small = SimpleNamespace(
        normalize_quantity=lambda symbol, qty: (0.0 if qty < bad_constraints.min_qty else qty, bad_constraints)
    )
    assert cli._build_order_notional(
        1000.0,
        10_000.0,
        cfg,
        "futures",
        3.0,
        client_too_small,
        constraints=bad_constraints,
    ) == (0.0, 0.0)

    assert cli._build_order_notional(0.0, 10000.0, cfg, "futures", 3.0, client) == (0.0, 0.0)
    assert cli._build_order_notional(1000.0, -1.0, cfg, "futures", 3.0, client) == (0.0, 0.0)


def test_live_inference_rows_include_latest_closed_candle() -> None:
    candles = _simple_candles(340)
    strategy = StrategyConfig()
    base_rows = cli._live_rows_for_model(candles, strategy, None)
    assert base_rows[-1].timestamp == candles[-1].close_time

    feature_cfg = default_config_for("default", strategy.enabled_features)
    advanced_model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        feature_signature=advanced_feature_signature(feature_cfg),
    )
    advanced_rows = cli._live_rows_for_model(candles, strategy, advanced_model)
    labeled_advanced_rows = make_advanced_rows(candles, feature_cfg)
    assert advanced_rows[-1].timestamp == candles[-1].close_time
    assert labeled_advanced_rows[-1].timestamp < candles[-1].close_time


def test_futures_direction_clamps_low_threshold_to_avoid_overlap() -> None:
    cfg = StrategyConfig(signal_threshold=0.40)
    assert cli._score_to_direction(0.45, cfg, "futures", threshold=0.40) == -1
    assert cli._score_to_direction(0.55, cfg, "futures", threshold=0.40) == 1


def test_paper_order_is_logged(tmp_path, monkeypatch) -> None:
    client = _FakeClient()
    cfg = StrategyConfig()

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    cli._paper_or_live_order(client, RuntimeConfig(), cfg, side="BUY", size=0.1, dry_run=True, leverage=3.0)
    cli._paper_or_live_order(client, RuntimeConfig(), cfg, side="SELL", size=0.2, dry_run=False, leverage=2.0)
    assert client.orders[0][0] == "BTCUSDC"


def test_live_order_forwards_entry_notional_to_bracket_aware_client() -> None:
    class _BracketAwareClient(_FakeClient):
        def get_max_leverage_for_notional(self, _symbol: str, _notional: float) -> int:
            return 5

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            notional: float | None = None,
        ):
            self.orders.append((symbol, side, size, dry_run, leverage, notional))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "notional": notional}

    client = _BracketAwareClient()

    cli._paper_or_live_order(
        client,
        RuntimeConfig(market_type="futures"),
        StrategyConfig(),
        side="BUY",
        size=0.2,
        dry_run=False,
        leverage=10.0,
        notional=2_500.0,
    )

    assert client.orders[-1][-1] == pytest.approx(2_500.0)


def test_roundtrip_helpers_cover_balances_and_sizing() -> None:
    account = {"balances": [{"asset": "USDC", "free": "10"}, {"asset": "USDC", "free": "2.5"}, {"asset": "BTC", "free": "bad"}]}
    assert cli._asset_free_balance(account, "USDC") == 12.5
    assert cli._asset_free_balance([], "USDC") == 0.0
    assert cli._order_executed_qty({"executedQty": "0.01"}) == 0.01
    assert cli._order_executed_qty({"origQty": "0.02"}) == 0.0
    assert cli._order_executed_qty({"fills": [{"qty": "0.01"}, {"qty": "0.02"}]}) == pytest.approx(0.03)
    assert cli._order_executed_qty([]) == 0.0
    qty, average, notional = cli._order_fill_details(
        {"fills": [{"qty": "0.01", "price": "50000"}, {"qty": "0.02", "price": "51000"}]},
        fallback_qty=0.5,
        fallback_price=1.0,
    )
    assert qty == pytest.approx(0.03)
    assert average == pytest.approx((0.01 * 50000 + 0.02 * 51000) / 0.03)
    assert notional == pytest.approx(1520.0)
    assert cli._order_fill_details({"executedQty": "0.1", "cummulativeQuoteQty": "12"}, fallback_qty=0.0, fallback_price=0.0) == (0.1, 120.0, 12.0)
    assert cli._order_fill_details({}, fallback_qty=0.2, fallback_price=50.0) == (0.2, 50.0, 10.0)
    assert cli._order_fill_details(
        {"status": "NEW", "origQty": "0.2"},
        fallback_qty=0.2,
        fallback_price=50.0,
        allow_quantity_fallback=False,
    ) == (0.0, 50.0, 0.0)
    assert cli._order_fill_details(
        {"fills": [object(), {"qty": "0", "price": "1"}, {"qty": "0.1", "price": "0"}], "executedQty": "0.2", "avgPrice": "77"},
        fallback_qty=0.0,
        fallback_price=0.0,
    ) == (0.2, 77.0, 15.4)
    assert cli._order_fill_details(None, fallback_qty=0.3, fallback_price=10.0) == (0.3, 10.0, 3.0)

    constraints = SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 5.0, 1000.0)

    class _Client:
        def normalize_quantity(self, symbol: str, quantity: float):
            assert symbol == "BTCUSDC"
            rounded = math.floor(quantity * 100000) / 100000
            return rounded, constraints

    quantity, parsed, notional = cli._roundtrip_quantity(_Client(), "BTCUSDC", 0.00001, 76000.0)
    assert parsed == constraints
    assert quantity >= 0.00008
    assert notional >= 5.0
    assert (
        cli._roundtrip_second_quantity(
            _Client(),
            "BTCUSDC",
            "USDC",
            "BTC",
            "SELL",
            0.0002,
            {"balances": [{"asset": "BTC", "free": "0.0001"}]},
            76000.0,
        )
        == 0.0001
    )
    assert (
        cli._roundtrip_second_quantity(
            _Client(),
            "BTCUSDC",
            "USDC",
            "BTC",
            "BUY",
            0.0002,
            {"balances": [{"asset": "USDC", "free": "5"}]},
            76000.0,
        )
        > 0
    )

    with pytest.raises(ValueError):
        cli._roundtrip_quantity(_Client(), "BTCUSDC", 0.0, 76000.0)
    with pytest.raises(BinanceAPIError, match="positive"):
        cli._roundtrip_quantity(_Client(), "BTCUSDC", 0.0001, 0.0)

    tight = SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 0.0, 1.0)

    class _TightClient:
        def normalize_quantity(self, _symbol: str, quantity: float):
            return quantity, tight

    with pytest.raises(BinanceAPIError, match="exceeds"):
        cli._roundtrip_quantity(_TightClient(), "BTCUSDC", 0.1, 76000.0)

    zero = SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 0.0, 1000.0)

    class _ZeroClient:
        def normalize_quantity(self, _symbol: str, _quantity: float):
            return 0.0, zero

    with pytest.raises(BinanceAPIError, match="below BTCUSDC exchange filters"):
        cli._roundtrip_quantity(_ZeroClient(), "BTCUSDC", 0.00001, 76000.0)

    class _StickyMinNotionalClient:
        def normalize_quantity(self, _symbol: str, _quantity: float):
            return 0.00001, constraints

    with pytest.raises(BinanceAPIError, match="below exchange minimum"):
        cli._roundtrip_quantity(_StickyMinNotionalClient(), "BTCUSDC", 0.00001, 76000.0)


def test_resolved_order_fill_queries_exchange_when_live_ack_has_no_fill() -> None:
    class _OrderQueryClient:
        def __init__(self) -> None:
            self.queries: list[tuple[str, object, str | None]] = []

        def get_order(self, symbol: str, *, order_id=None, orig_client_order_id=None):
            self.queries.append((symbol, order_id, orig_client_order_id))
            return {
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": "0.4",
                "cummulativeQuoteQty": "44",
            }

    client = _OrderQueryClient()
    runtime = RuntimeConfig(symbol="ETHUSDC")
    qty, average, notional, source = cli._resolved_order_fill_details(
        client,  # type: ignore[arg-type]
        runtime,
        {"status": "NEW", "orderId": 123, "clientOrderId": "abc", "origQty": "0.4"},
        fallback_qty=0.4,
        fallback_price=100.0,
        dry_run=False,
    )

    assert qty == pytest.approx(0.4)
    assert average == pytest.approx(110.0)
    assert notional == pytest.approx(44.0)
    assert source == "order_query"
    assert client.queries == [("ETHUSDC", 123, "abc")]


def test_resolved_order_fill_does_not_query_or_trust_orig_qty_for_unidentified_live_order() -> None:
    client = _FakeClient()
    qty, average, notional, source = cli._resolved_order_fill_details(
        client,  # type: ignore[arg-type]
        RuntimeConfig(symbol="BTCUSDC"),
        {"status": "NEW", "origQty": "0.4"},
        fallback_qty=0.4,
        fallback_price=100.0,
        dry_run=False,
    )

    assert qty == 0.0
    assert average == 100.0
    assert notional == 0.0
    assert source == "unresolved_no_order_id"


def test_command_spot_roundtrip_validation_and_success(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="futures", testnet=True, api_key="k", api_secret="s"))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.0001, mode="auto", yes=True)) == 2
    save_runtime(RuntimeConfig(market_type="spot", testnet=False, api_key="k", api_secret="s"))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.0001, mode="auto", yes=True)) == 2
    save_runtime(RuntimeConfig(market_type="spot", testnet=True, api_key="", api_secret=""))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.0001, mode="auto", yes=True)) == 2
    save_runtime(RuntimeConfig(market_type="spot", testnet=True, api_key="k", api_secret="s"))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.0001, mode="auto", yes=False)) == 2

    constraints = SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 5.0, 1000.0)

    class _RoundtripClient:
        def __init__(self, *, usdc: float = 20.0, btc: float = 0.5, fail_order: bool = False) -> None:
            self.usdc = usdc
            self.btc = btc
            self.fail_order = fail_order
            self.orders: list[str] = []

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_symbol_price(self, _symbol: str):
            return 76000.0, 1

        def normalize_quantity(self, _symbol: str, quantity: float):
            rounded = math.floor(quantity * 100000) / 100000
            return rounded, constraints

        def get_account(self):
            return {
                "balances": [
                    {"asset": "USDC", "free": str(self.usdc), "locked": "0"},
                    {"asset": "BTC", "free": str(self.btc), "locked": "0"},
                ]
            }

        def place_order(self, _symbol: str, side: str, quantity: float, *, dry_run: bool, leverage: float = 1.0):
            assert dry_run is False
            if self.fail_order:
                raise BinanceAPIError("order failed")
            self.orders.append(side)
            if side == "BUY":
                self.btc += quantity
                self.usdc -= quantity * 76000.0
            else:
                self.btc -= quantity
                self.usdc += quantity * 76000.0
            return {"status": "FILLED", "orderId": len(self.orders), "executedQty": f"{quantity:.8f}"}

    persisted: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=20.0, btc=0.5))
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, output_dir, payload: persisted.append(payload) or output_dir / "roundtrip.json")
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="auto", yes=True)) == 0
    output = capsys.readouterr().out
    assert "Spot testnet roundtrip complete." in output
    assert persisted[-1]["mode"] == "buy-sell"
    assert persisted[-1]["runtime"]["api_key"] == "<redacted>"

    class _EthRoundtripClient:
        def __init__(self) -> None:
            self.usdt = 100.0
            self.eth = 1.0
            self.orders: list[str] = []

        def ensure_symbol(self, symbol: str):
            return {"symbol": symbol}

        def get_symbol_price(self, _symbol: str):
            return 3000.0, 1

        def normalize_quantity(self, _symbol: str, quantity: float):
            rounded = math.floor(quantity * 1000) / 1000
            return rounded, SymbolConstraints("ETHUSDT", 0.001, 10.0, 0.001, 5.0, 1000.0)

        def get_account(self):
            return {
                "balances": [
                    {"asset": "USDT", "free": str(self.usdt), "locked": "0"},
                    {"asset": "ETH", "free": str(self.eth), "locked": "0"},
                ]
            }

        def place_order(self, _symbol: str, side: str, quantity: float, *, dry_run: bool, leverage: float = 1.0):
            assert dry_run is False
            self.orders.append(side)
            if side == "BUY":
                self.eth += quantity
                self.usdt -= quantity * 3000.0
            else:
                self.eth -= quantity
                self.usdt += quantity * 3000.0
            return {"status": "FILLED", "orderId": len(self.orders), "executedQty": f"{quantity:.8f}"}

    save_runtime(RuntimeConfig(symbol="ETHUSDT", quote_asset="USDT", market_type="spot", testnet=True, api_key="k", api_secret="s"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _EthRoundtripClient())
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.01, mode="auto", yes=True)) == 0
    assert "ETH" in persisted[-1]["balances_before"]
    assert "USDT" in persisted[-1]["balances_before"]
    assert persisted[-1]["mode"] == "buy-sell"

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, api_key="k", api_secret="s"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=0.0, btc=0.5))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="auto", yes=True)) == 0
    assert persisted[-1]["mode"] == "sell-buy"

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=0.0, btc=0.5))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="buy-sell", yes=True)) == 2

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=0.0, btc=0.0))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="sell-buy", yes=True)) == 2

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=20.0, btc=0.5))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="bad-mode", yes=True)) == 2

    class _NoSecondLegBalanceClient(_RoundtripClient):
        def place_order(self, _symbol: str, side: str, quantity: float, *, dry_run: bool, leverage: float = 1.0):
            assert dry_run is False
            self.orders.append(side)
            return {"status": "FILLED", "orderId": len(self.orders), "executedQty": f"{quantity:.8f}"}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _NoSecondLegBalanceClient(usdc=20.0, btc=0.0))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="buy-sell", yes=True)) == 2
    assert persisted[-1]["status"] == "partial_failed"
    assert persisted[-1]["first_order"]["side"] == "BUY"
    assert persisted[-1]["second_order"]["status"] == "not_completed"

    class _AckOnlyFirstLegClient(_RoundtripClient):
        def place_order(self, _symbol: str, side: str, quantity: float, *, dry_run: bool, leverage: float = 1.0):
            assert dry_run is False
            self.orders.append(side)
            return {"status": "NEW", "orderId": len(self.orders), "origQty": f"{quantity:.8f}"}

    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, output_dir, payload: persisted.append(payload) or output_dir / "roundtrip.json")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _AckOnlyFirstLegClient(usdc=20.0, btc=0.0))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="buy-sell", yes=True)) == 2
    assert "First roundtrip order response did not include executed quantity" in capsys.readouterr().err
    assert persisted[-1]["status"] == "partial_failed"
    assert persisted[-1]["first_order"]["fill_source"] == "unresolved_no_order_query"
    assert persisted[-1]["second_order"]["status"] == "not_completed"

    def raise_persist(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cli, "_persist_run_artifact", raise_persist)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _NoSecondLegBalanceClient(usdc=20.0, btc=0.0))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="buy-sell", yes=True)) == 2
    assert "Partial spot roundtrip could not be recorded" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RoundtripClient(usdc=20.0, btc=0.5, fail_order=True))
    assert cli.command_spot_roundtrip(argparse.Namespace(quantity=0.00008, mode="buy-sell", yes=True)) == 2
    assert persisted[-1]["status"] == "partial_failed"


def test_command_status_prints_masked_secret(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="visible-key", api_secret="super-secret"))
    save_strategy(StrategyConfig())
    assert cli.command_status(argparse.Namespace()) == 0
    output = capsys.readouterr().out
    assert "<redacted>" in output
    assert "super-secret" not in output
    assert "visible-key" not in output


def test_command_status_compact_reports_execution_and_lossless_ledger_state(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=True,
            market_type="futures",
            quote_asset="USDT",
            symbol="BTCUSDT",
            symbols=("BTCUSDT",),
        )
    )
    save_strategy(StrategyConfig(risk_level="conservative", leverage=5.0))

    assert cli.command_status(argparse.Namespace(compact=True)) == 0
    line = capsys.readouterr().out.strip()
    assert "environment=testnet" in line
    assert "execution=paper" in line
    assert "positions=0 ledger=clear" in line
    assert "BTCUSDT" in line

    open_path = tmp_path / "data" / "autonomous" / "open_positions.json"
    open_path.parent.mkdir(parents=True, exist_ok=True)
    open_path.write_text("{invalid", encoding="utf-8")
    assert cli.command_status(argparse.Namespace(compact=True)) == 0
    assert "positions=0 ledger=invalid" in capsys.readouterr().out


def test_command_compute_shows_and_saves_backend(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(compute_backend="cpu"))
    assert cli.command_compute(argparse.Namespace(backend=None)) == 0
    assert "compute=cpu" in capsys.readouterr().out

    assert cli.command_compute(argparse.Namespace(backend="directml")) == 0
    assert load_runtime().compute_backend == "directml"
    assert "compute=" in capsys.readouterr().out

    assert cli.command_compute(argparse.Namespace(backend="bad")) == 2
    assert "Unknown compute backend" in capsys.readouterr().err


def test_command_configure_validation_failure_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    class _BadClient(_FakeClient):
        def ensure_btcusdc(self):
            raise BinanceAPIError("no symbol")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(api_key="k", api_secret="s", validate_account=True))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadClient())
    assert cli.command_configure(argparse.Namespace()) == 2
    assert "validation failed" in capsys.readouterr().err

    class _BadAccountClient(_FakeClient):
        def get_account(self):
            raise BinanceAPIError("bad key")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadAccountClient())
    assert cli.command_configure(argparse.Namespace()) == 2
    assert "bad key" in capsys.readouterr().err


def test_command_configure_futures_prints_mode_line(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(market_type="futures", validate_account=False))
    assert cli.command_configure(argparse.Namespace()) == 0
    assert "futures-mode enabled" in capsys.readouterr().out


def test_command_connect_spot_and_futures(tmp_path, monkeypatch, capsys) -> None:
    fake = _FakeClient()
    monkeypatch.setenv("HOME", str(tmp_path))
    runtime = RuntimeConfig(api_key="k", api_secret="s", market_type="futures", symbol="BTCUSDC")
    save_runtime(runtime)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: fake)
    assert cli.command_connect(argparse.Namespace()) == 0
    text = capsys.readouterr().out
    assert "exchange: connected" in text
    assert "max leverage on BTCUSDC: 10x" in text


def test_command_connect_failure_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    class _BadClient(_FakeClient):
        def get_exchange_time(self):
            raise BinanceAPIError("offline")

    class _BadAccountClient(_FakeClient):
        def get_account(self):
            raise BinanceAPIError("bad key")

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadClient())
    assert cli.command_connect(argparse.Namespace()) == 2
    assert "Connect requires Binance API key" in capsys.readouterr().err

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    assert cli.command_connect(argparse.Namespace()) == 2
    assert "Connection failed: offline" in capsys.readouterr().err

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", testnet=False, demo=False))
    assert cli.command_connect(argparse.Namespace()) == 2
    assert "only on Binance testnet or demo" in capsys.readouterr().err

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadAccountClient())
    assert cli.command_connect(argparse.Namespace()) == 2
    assert "Connect requires valid Binance API credentials" in capsys.readouterr().err


def test_command_risk_reports_json_text_and_conflicting_modes(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    assert cli.command_risk(argparse.Namespace(model=str(tmp_path / "missing.json"), paper=True, live=True, leverage=None, json=False)) == 2
    assert "Choose either" in capsys.readouterr().out

    assert cli.command_risk(argparse.Namespace(model=str(tmp_path / "missing.json"), paper=True, live=False, leverage=None, json=False)) == 0
    assert "Risk policy report" in capsys.readouterr().out

    assert cli.command_risk(argparse.Namespace(model=str(tmp_path / "missing.json"), paper=False, live=False, leverage=3.0, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["effective_dry_run"] is True
    assert payload["leverage"] == 1.0

    save_runtime(RuntimeConfig(dry_run=False, api_key="", api_secret=""))
    assert cli.command_risk(argparse.Namespace(model=str(tmp_path / "missing.json"), paper=False, live=True, leverage=3.0, json=True)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["allowed"] is False
    assert payload["block_count"] >= 1


def test_command_risk_live_requests_strict_model_evidence(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_trading import risk_workflows

    captured: dict[str, object] = {}

    class _Report:
        allowed = True

        def asdict(self):
            return {"allowed": True}

    def fake_report(*_args, **kwargs):
        captured.update(kwargs)
        return _Report()

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, api_key="k", api_secret="s", compute_backend="directml"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(
        risk_workflows,
        "resolve_backend",
        lambda _backend: cli.BackendInfo("directml", "directml", "privateuseone:0", "DirectML", ""),
    )
    monkeypatch.setattr(risk_workflows, "build_risk_policy_report", fake_report)

    assert cli.command_risk(
        argparse.Namespace(model=str(tmp_path / "model.json"), paper=False, live=True, leverage=None, json=True)
    ) == 0
    assert json.loads(capsys.readouterr().out)["allowed"] is True
    assert captured["require_model_candidate_search"] is True
    assert captured["require_accelerator_evidence"] is True
    assert captured["require_live_data_evidence"] is True
    assert captured["require_microstructure_evidence"] is False
    assert captured["expected_symbol"] == "BTCUSDC"
    assert captured["expected_market_type"] == "spot"
    assert captured["expected_interval"] == "15m"


def test_command_live_risk_policy_and_generic_entry_gate(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    save_runtime(RuntimeConfig(managed_usdc=0.0, dry_run=False, api_key="k", api_secret="s"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")

    class _LoadedModel(_SignedFeeModelEvidence):
        feature_signature = "sig"

        def predict_proba(self, _features) -> float:
            return 0.5

    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: _LoadedModel())
    args = argparse.Namespace(
        steps=1,
        sleep=0,
        paper=False,
        live=False,
        model=str(model_file),
        leverage=None,
        retrain_interval=0,
        retrain_window=1,
        retrain_min_rows=1,
        external_signals=None,
    )
    assert cli.command_live(args) == 2
    assert "positive USDC trading cap" in capsys.readouterr().err

    class _RiskModel(_SignedFeeModelEvidence):
        feature_signature = "sig"

        def predict_proba(self, _features) -> float:
            return 0.99

    save_runtime(RuntimeConfig(managed_usdc=1000.0, dry_run=True))
    save_strategy(StrategyConfig(enabled_features=("momentum_1",), max_regime_unpredictability=1.0))
    args.paper = True
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [SimpleNamespace(timestamp=1, close=0.0, features=(0.0,))])
    monkeypatch.setattr(cli, "_build_live_model", lambda *_args, **_kwargs: _RiskModel())
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    assert cli.command_live(args) == 0
    assert "risk gate blocked entry (price)" in capsys.readouterr().out
    artifact = next(tmp_path.glob("live_*.json"))
    events = json.loads(artifact.read_text(encoding="utf-8"))["events"]
    assert any(event["status"] == "skip_risk_price" for event in events)


def test_command_live_requires_exchange_filters_before_signed_loop(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    save_runtime(RuntimeConfig(managed_usdc=1000.0, dry_run=False, api_key="k", api_secret="s", testnet=True))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (SimpleNamespace(strategy_overrides={}), None, None))
    monkeypatch.setattr(cli, "_resolve_symbol_constraints", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "build_risk_policy_report", lambda *_args, **_kwargs: SimpleNamespace(allowed=True, warning_count=0))

    assert cli.command_live(_live_args(tmp_path, steps=0, paper=False, live=True)) == 2
    assert "requires BTCUSDC exchange filters" in capsys.readouterr().err


def test_command_fetch_handles_binar_errors(tmp_path, monkeypatch) -> None:
    from simple_ai_trading.api import BinanceAPIError

    class _ErrorClient(_FakeClient):
        def ensure_btcusdc(self):
            raise BinanceAPIError("bad")

    runtime = RuntimeConfig()
    output = tmp_path / "candles.json"
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(runtime)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ErrorClient())
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, output=str(output))) == 2


def test_command_fetch_accepts_non_default_symbol(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    output = tmp_path / "candles.json"
    assert cli.command_fetch(argparse.Namespace(symbol="ETHUSDC", interval=None, limit=10, output=str(output))) == 0
    assert output.exists()


def test_command_fetch_rejects_non_major_symbol_before_client_build(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    monkeypatch.setattr(
        cli,
        "_build_client",
        lambda _runtime: (_ for _ in ()).throw(AssertionError("unsupported symbol should not build a client")),
    )

    output = tmp_path / "candles.json"
    assert cli.command_fetch(argparse.Namespace(symbol="ALTUSDC", interval=None, limit=10, output=str(output))) == 2
    assert "only BTC, ETH, and SOL" in capsys.readouterr().err
    assert not output.exists()


def test_command_train_workflow(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(model_file),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0


def test_command_train_accepts_profit_threshold_candidate(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    class _ProfitCalibration:
        accepted = True
        threshold = 0.7
        best_threshold = 0.7
        best_score = 3.0
        baseline_score = 1.0
        realized_pnl = 2.0
        baseline_realized_pnl = -1.0
        closed_trades = 4
        baseline_closed_trades = 18

        def asdict(self) -> dict[str, object]:
            return {
                "score": 3.0,
                "realized_pnl": 2.0,
                "closed_trades": 4,
                "best_threshold": 0.7,
                "best_score": self.best_score,
                "baseline_score": self.baseline_score,
                "baseline_realized_pnl": self.baseline_realized_pnl,
                "baseline_closed_trades": self.baseline_closed_trades,
            }

    reports = [
        SimpleNamespace(
            accuracy=0.50,
            precision=0.40,
            recall=0.60,
            f1=0.48,
            threshold=0.5,
            true_positive=4,
            false_positive=6,
            true_negative=10,
            false_negative=3,
        ),
        SimpleNamespace(
            accuracy=0.55,
            precision=0.42,
            recall=0.50,
            f1=0.46,
            threshold=0.58,
            true_positive=4,
            false_positive=5,
            true_negative=11,
            false_negative=4,
        ),
        SimpleNamespace(
            accuracy=0.80,
            precision=0.50,
            recall=0.20,
            f1=0.29,
            threshold=0.7,
            true_positive=3,
            false_positive=3,
            true_negative=15,
            false_negative=12,
        ),
    ]
    monkeypatch.setattr(cli, "calibrate_threshold_for_backtest", lambda *_a, **_k: _ProfitCalibration())
    monkeypatch.setattr(cli, "evaluate_classification", lambda *_a, **_k: reports.pop(0))

    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(model_file),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=True,
        )
    ) == 0
    output = capsys.readouterr().out
    model_payload = json.loads(model_file.read_text(encoding="utf-8"))
    assert "profit threshold candidate: accepted" in output
    assert model_payload["decision_threshold"] == 0.7
    assert model_payload["threshold_source"] == "profit_backtest"
    assert model_payload["threshold_calibration_score"] == 3.0
    assert model_payload["threshold_calibration_trades"] == 4


def test_command_train_keeps_strategy_threshold_when_profit_candidate_rejected(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(signal_threshold=0.61))
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    class _RejectedProfitCalibration:
        accepted = False
        threshold = 0.2
        best_threshold = 0.2
        best_score = -1.0
        baseline_score = 0.0
        realized_pnl = -4.0
        baseline_realized_pnl = 0.0
        closed_trades = 3
        baseline_closed_trades = 2

        def asdict(self) -> dict[str, object]:
            return {
                "score": -1.0,
                "realized_pnl": -4.0,
                "closed_trades": 3,
                "best_threshold": 0.2,
                "best_score": self.best_score,
                "baseline_score": self.baseline_score,
                "baseline_realized_pnl": self.baseline_realized_pnl,
                "baseline_closed_trades": self.baseline_closed_trades,
            }

    reports = [
        SimpleNamespace(
            accuracy=0.55,
            precision=0.45,
            recall=0.50,
            f1=0.47,
            threshold=0.61,
            true_positive=5,
            false_positive=6,
            true_negative=10,
            false_negative=5,
        ),
        SimpleNamespace(
            accuracy=0.65,
            precision=0.50,
            recall=0.70,
            f1=0.58,
            threshold=0.2,
            true_positive=7,
            false_positive=7,
            true_negative=9,
            false_negative=3,
        ),
        SimpleNamespace(
            accuracy=0.40,
            precision=0.20,
            recall=0.20,
            f1=0.20,
            threshold=0.2,
            true_positive=2,
            false_positive=8,
            true_negative=5,
            false_negative=8,
        ),
    ]
    monkeypatch.setattr(cli, "calibrate_threshold", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(cli, "calibrate_threshold_for_backtest", lambda *_args, **_kwargs: _RejectedProfitCalibration())
    monkeypatch.setattr(cli, "evaluate_classification", lambda *_args, **_kwargs: reports.pop(0))

    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(model_file),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=True,
        )
    ) == 0
    output = capsys.readouterr().out
    model_payload = json.loads(model_file.read_text(encoding="utf-8"))
    assert "profit threshold candidate: rejected" in output
    assert model_payload["decision_threshold"] == 0.61
    assert model_payload["threshold_source"] == "strategy"
    assert model_payload["threshold_calibration_score"] == -1.0


def test_command_train_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    bad_input = tmp_path / "bad.json"
    bad_input.write_text("{", encoding="utf-8")
    assert cli.command_train(
        argparse.Namespace(
            input=str(bad_input),
            output=str(tmp_path / "model.json"),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_train_walk_forward_unavailable_still_succeeds(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(260)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    monkeypatch.setattr(cli, "walk_forward_report", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("not enough rows")))
    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(tmp_path / "model.json"),
            epochs=12,
            walk_forward=True,
            walk_forward_train=1000,
            walk_forward_test=1000,
            walk_forward_step=10,
            calibrate_threshold=False,
        )
    ) == 0
    assert "walk-forward unavailable" in capsys.readouterr().out


def test_command_train_rejects_invalid_optimizer_parameters(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [object()])

    base = dict(
        input="history.json",
        output=str(tmp_path / "model.json"),
        preset="custom",
        epochs=10,
        seed=7,
        walk_forward=False,
        walk_forward_train=10,
        walk_forward_test=5,
        walk_forward_step=1,
        calibrate_threshold=False,
    )
    assert cli.command_train(argparse.Namespace(**base, learning_rate=0.0, l2_penalty=0.0)) == 2
    assert "learning_rate must be > 0" in capsys.readouterr().err
    assert cli.command_train(argparse.Namespace(**base, learning_rate=0.1, l2_penalty=-0.1)) == 2
    assert "l2_penalty must be >= 0" in capsys.readouterr().err
    save_runtime(RuntimeConfig(compute_backend="not-real"))
    assert cli.command_train(argparse.Namespace(**base, learning_rate=0.1, l2_penalty=0.0)) == 2
    assert "unknown compute backend" in capsys.readouterr().err


def test_command_train_handles_no_calibration_split(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    rows = [
        SimpleNamespace(timestamp=1, close=100.0, features=(0.0,), label=0),
        SimpleNamespace(timestamp=2, close=101.0, features=(1.0,), label=1),
    ]
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: rows)

    assert cli.command_train(
        argparse.Namespace(
            input="history.json",
            output=str(tmp_path / "model.json"),
            preset="custom",
            epochs=3,
            learning_rate=0.05,
            l2_penalty=0.0,
            seed=7,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0
    output = capsys.readouterr().out
    assert "probability calibration: fail" in output


def test_command_train_artifact_includes_signature(tmp_path, monkeypatch) -> None:
    strategy = StrategyConfig(feature_windows=(4, 20), label_threshold=0.002, training_epochs=7)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(strategy)

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(300)
    ]
    history = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "artifact.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

    assert cli.command_train(
        argparse.Namespace(
            input=str(history),
            output=str(model_file),
            epochs=11,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "train"
    assert payload["seed"] == 7
    expected_signature = feature_signature(
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        strategy.label_threshold,
        feature_version=strategy.feature_version,
    )
    assert payload["model"]["feature_signature"] == expected_signature
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_train_written_artifact_does_not_leak_credentials(tmp_path, monkeypatch) -> None:
    strategy = StrategyConfig()
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(strategy)
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(model_file),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0

    artifact_text = next(tmp_path.glob("train_run_*.json")).read_text(encoding="utf-8")
    assert "secret-key" not in artifact_text
    assert "secret-value" not in artifact_text
    assert "<redacted>" in artifact_text


def test_command_evaluate_runs_with_model_file(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file.write_text(
        json.dumps(
            {
                "weights": [0.1] + [0.0] * 12,
                "bias": 0.01,
                "feature_version": "v1",
                "feature_dim": 13,
                "epochs": 5,
                "feature_means": [1.0] * 13,
                "feature_stds": [1.0] * 13,
                "feature_signature": cli._strategy_feature_signature(StrategyConfig()),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(data_file),
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 0


def test_command_evaluate_artifact_is_emitted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig())
    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "evaluate.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file.write_text(
        json.dumps(
            {
                "weights": [0.1] + [0.0] * 12,
                "bias": 0.01,
                "feature_version": "v1",
                "feature_dim": 13,
                "epochs": 5,
                "feature_means": [1.0] * 13,
                "feature_stds": [1.0] * 13,
                "feature_signature": cli._strategy_feature_signature(StrategyConfig()),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert (
        cli.command_evaluate(
            argparse.Namespace(
                input=str(data_file),
                model=str(model_file),
                threshold=None,
                calibrate_threshold=False,
            )
        )
        == 0
    )

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "evaluate"
    assert payload["command"] == "evaluate"
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_evaluate_prints_zero_train_metrics_when_split_has_no_train_rows(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    row = SimpleNamespace(timestamp=1, features=(0.0,), label=1)
    model_file = tmp_path / "model.json"
    model_file.write_text(
        json.dumps(
            {
                "weights": [0.0],
                "bias": 0.0,
                "feature_dim": 1,
                "epochs": 1,
                "feature_version": "v1",
                "feature_means": [0.0],
                "feature_stds": [1.0],
                "feature_signature": cli._strategy_feature_signature(StrategyConfig()),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg, **_kwargs: [row])
    monkeypatch.setattr(cli, "temporal_validation_split", lambda _rows: TemporalValidationSplit([], [], [row]))

    assert cli.command_evaluate(
        argparse.Namespace(input="x", model=str(model_file), threshold=None, calibrate_threshold=False)
    ) == 0
    assert "train_accuracy: 0.000 precision=0.000 recall=0.000 f1=0.000" in capsys.readouterr().out


def test_command_evaluate_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    bad_input = tmp_path / "history.json"
    bad_input.write_text("{", encoding="utf-8")
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(bad_input),
            model=str(tmp_path / "model.json"),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_evaluate_rejects_invalid_model_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file = tmp_path / "model.json"
    model_file.write_text("{", encoding="utf-8")
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(data_file),
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_backtest_rejects_invalid_model_payload(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setenv("HOME", str(tmp_path))
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(220)
    ]
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file.write_text("{", encoding="utf-8")
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 2


def test_command_backtest_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setenv("HOME", str(tmp_path))
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text("{", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 2


def test_command_tune_needs_more_rows(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(80)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    assert cli.command_tune(
        argparse.Namespace(
            input=str(data_file),
            save_best=False,
            min_risk=0.002,
            max_risk=0.02,
            steps=2,
            min_leverage=1.0,
            max_leverage=2.0,
            min_threshold=0.52,
            max_threshold=0.6,
            min_take=0.01,
            max_take=0.02,
            min_stop=0.008,
            max_stop=0.02,
        )
    ) == 2
    assert "Need more data rows" in capsys.readouterr().out


def test_command_tune_uses_fallback_and_can_save_best(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda *_a, **_k: SimpleNamespace(
            realized_pnl=-5.0,
            total_fees=1.0,
            max_drawdown=0.1,
            closed_trades=1,
            stopped_by_drawdown=True,
        ),
    )

    assert cli.command_tune(
        argparse.Namespace(
            input=str(data_file),
            save_best=True,
            min_risk=0.002,
            max_risk=0.003,
            steps=1,
            min_leverage=1.0,
            max_leverage=1.0,
            min_threshold=0.52,
            max_threshold=0.52,
            min_take=0.01,
            max_take=0.01,
            min_stop=0.008,
            max_stop=0.008,
        )
    ) == 0
    output = capsys.readouterr().out
    assert "all tune candidates hit drawdown limit" in output
    assert "Saved tuned strategy." in output


def test_command_live_paper_flag_overrides_runtime_live_without_credentials(tmp_path, monkeypatch) -> None:
    class _LiveClient:
        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="", api_secret=""))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=True, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0


def test_command_live_spot_leverage_override_is_inactive(tmp_path, monkeypatch, capsys) -> None:
    class _LiveClient:
        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=20.0, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    assert "Leverage override is spot-inactive" in capsys.readouterr().out


def test_command_backtest_artifact_is_emitted(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_trading.model import serialize_model

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig())
    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "backtest.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(220)
    ]
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    serialize_model(
        TrainedModel(
            weights=[0.0] * 13,
            bias=0.0,
            feature_dim=13,
                epochs=5,
                feature_means=[0.0] * 13,
                feature_stds=[1.0] * 13,
                feature_signature=cli._strategy_feature_signature(StrategyConfig()),
            ),
        model_file,
    )

    assert (
        cli.command_backtest(
            argparse.Namespace(
                input=str(input_file),
                model=str(model_file),
                start_cash=1000.0,
                compute_backend="cpu",
                score_batch_size=4,
            )
        )
        == 0
    )

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "backtest"
    assert payload["command"] == "backtest"
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"
    assert payload["scoring_backend"]["kind"] == "cpu"
    assert payload["scoring_backend"]["score_batch_size"] == 4

    def fake_run_backtest(*_args, **kwargs):
        assert kwargs["compute_backend"] == "directml"
        assert kwargs["score_batch_size"] == 16
        return SimpleNamespace(
            trades=0,
            win_rate=0.0,
            realized_pnl=0.0,
            total_fees=0.0,
            max_exposure=0.0,
            starting_cash=1000.0,
            ending_cash=1000.0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=0.0,
            max_drawdown=0.0,
            stopped_by_drawdown=False,
            trades_per_day_cap_hit=0,
            closed_trades=0,
            gross_exposure=0.0,
            scoring_backend_requested="directml",
            scoring_backend_kind="cpu",
            scoring_backend_device="cpu",
            scoring_backend_reason="DirectML unavailable in test",
        )

    monkeypatch.setattr(cli, "run_backtest", fake_run_backtest)
    assert (
        cli.command_backtest(
            argparse.Namespace(
                input=str(input_file),
                model=str(model_file),
                start_cash=1000.0,
                compute_backend="directml",
                score_batch_size=16,
            )
        )
        == 0
    )
    assert "scoring_backend_reason: DirectML unavailable in test" in capsys.readouterr().out


def test_command_backtest_uses_top_of_book_execution_db(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_trading.market_store import MarketDataStore
    from simple_ai_trading.model import serialize_model

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(symbol="ETHUSDC", market_type="spot"))
    save_strategy(StrategyConfig(max_spread_bps=5.0))
    captured: dict[str, object] = {}

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 10.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(220)
    ]
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    db_file = tmp_path / "market.sqlite"
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    serialize_model(
        TrainedModel(
            weights=[0.0] * 13,
            bias=0.0,
            feature_dim=13,
            epochs=5,
            feature_means=[0.0] * 13,
            feature_stds=[1.0] * 13,
            feature_signature=cli._strategy_feature_signature(StrategyConfig()),
        ),
        model_file,
    )
    with MarketDataStore(db_file) as store:
        store.insert_top_of_book_snapshot(
            "binance",
            "ETHUSDC",
            "spot",
            {"bidPrice": "2500.00", "bidQty": "12", "askPrice": "2500.50", "askQty": "10"},
            ts_ms=int(cli.time.time() * 1000),
        )

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured["artifact"] = payload
        return output_dir / f"{kind}.json"

    def fake_run_backtest(*_args, **kwargs):
        captured["profile"] = kwargs.get("symbol_profile")
        return SimpleNamespace(
            trades=0,
            win_rate=0.0,
            realized_pnl=0.0,
            total_fees=0.0,
            max_exposure=0.0,
            starting_cash=1000.0,
            ending_cash=1000.0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=0.0,
            max_drawdown=0.0,
            stopped_by_drawdown=False,
            trades_per_day_cap_hit=0,
            closed_trades=0,
            gross_exposure=0.0,
            scoring_backend_requested="cpu",
            scoring_backend_kind="cpu",
            scoring_backend_device="cpu",
            scoring_backend_reason="",
        )

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli, "run_backtest", fake_run_backtest)

    assert (
        cli.command_backtest(
            argparse.Namespace(
                input=str(input_file),
                model=str(model_file),
                start_cash=1000.0,
                compute_backend="cpu",
                score_batch_size=4,
                execution_db=str(db_file),
            )
        )
        == 0
    )

    profile = captured["profile"]
    assert getattr(profile, "symbol") == "ETHUSDC"
    assert getattr(profile, "spread_bps") > 0.0
    artifact = captured["artifact"]
    assert artifact["execution_profile"]["profile"]["symbol"] == "ETHUSDC"
    assert artifact["execution_profile"]["source"] == "top_of_book:binance"
    assert "execution_profile: top_of_book:binance" in capsys.readouterr().out


def test_command_live_paper_path_runs_a_tick(tmp_path, monkeypatch) -> None:
    class _LiveClient:
        def __init__(self):
            self.calls = 0

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.calls += 1
            return {"side": side, "symbol": symbol, "size": size, "dry_run": dry_run}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            normalized = max(constraints.min_qty, round(quantity, 4))
            return normalized, constraints

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=5,
                paper=False,
                retrain_interval=-1,
                retrain_window=0,
                retrain_min_rows=0,
            )
        )
        == 0
    )


def test_command_live_applies_data_probed_liquidity_guard_to_entry(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str, str, dict[str, object]]] = []

    class _LiveClient:
        def __init__(self) -> None:
            self.orders: list[tuple[str, str, float]] = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

    base_ts = 1_767_621_600_000
    rows = [
        ModelRow(
            timestamp=base_ts + index * 60_000,
            close=100.0,
            features=(1.0,),
            label=1,
            volume=10.0 if index == 8 else 1000.0,
        )
        for index in range(9)
    ]
    client = _LiveClient()

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.01,
            max_position_pct=0.50,
            stop_loss_pct=0.01,
            take_profit_pct=0.50,
            signal_threshold=0.70,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            liquidity_lookback_bars=8,
            low_liquidity_volume_ratio=0.50,
            low_liquidity_size_multiplier=0.25,
            low_liquidity_signal_threshold_add=0.0,
            dynamic_liquidity_session_enabled=False,
            max_regime_unpredictability=1.0,
        )
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0

    assert client.orders == [("BTCUSDC", "BUY", pytest.approx(1.25))]
    assert captured and captured[0][0] == "live"
    statuses = [event.get("status") for event in captured[0][2]["events"] if isinstance(event, dict)]
    assert "liquidity_session_guard" in statuses


def test_command_live_artifact_is_emitted(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str, str, dict[str, object]]] = []

    class _LiveClient:
        def __init__(self) -> None:
            self.orders: list[tuple[str, str, float]] = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=i * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=True,
            market_type="spot",
            api_key="secret-key",
            api_secret="secret-value",
            managed_usdc=2500.0,
        )
    )
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0
    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "live"
    assert payload["command"] == "live"
    assert payload["starting_cash"] == 2500.0
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_live_retries_market_data_and_observes_before_entry(tmp_path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def candles_for(close: float, n: int = 320) -> list[Candle]:
        return [
            Candle(
                open_time=index * 60_000,
                open=close,
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=1000.0,
                close_time=(index + 1) * 60_000,
            )
            for index in range(n)
        ]

    def rows_from_candles(candles: list[Candle], *_args, **_kwargs) -> list[ModelRow]:
        close = candles[-1].close
        return [
            ModelRow(
                timestamp=index * 60_000,
                close=close,
                features=(0.1,),
                label=1,
                volume=1000.0,
            )
            for index in range(20)
        ]

    class _RecoveringClient:
        def __init__(self) -> None:
            self.kline_calls = 0
            self.orders: list[tuple[str, str, float]] = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            self.kline_calls += 1
            if self.kline_calls == 1:
                raise BinanceAPIError("temporary network outage")
            return candles_for(100.0 if self.kline_calls == 2 else 125.0, n=limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

    client = _RecoveringClient()

    def fake_persist(_kind: str, _output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append(payload)
        return _output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
            StrategyConfig(
                risk_per_trade=0.001,
                max_position_pct=0.2,
                stop_loss_pct=0.01,
            signal_threshold=0.70,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            cooldown_minutes=0,
            recovery_cooldown_seconds=0,
            max_regime_unpredictability=1.0,
            dynamic_liquidity_session_enabled=False,
        )
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_live_rows_for_model", rows_from_candles)
    monkeypatch.setattr(cli, "_build_model_rows", rows_from_candles)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=3, sleep=0)) == 0

    assert client.kline_calls == 3
    assert client.orders == [("BTCUSDC", "BUY", pytest.approx(0.5338, rel=1e-4))]
    statuses = [event.get("status") for event in captured[0]["events"] if isinstance(event, dict)]
    assert "market_error_retry" in statuses
    assert "skip_risk_recovery_pending" in statuses
    assert "recovery_observation" in statuses
    assert captured[0]["result"]["status"] == "completed"


def test_command_live_market_recovery_pending_exits_nonzero(tmp_path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class _OfflineClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            raise BinanceAPIError("offline")

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

    def fake_persist(_kind: str, _output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append(payload)
        return _output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(max_network_errors=1, recovery_cooldown_seconds=0))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _OfflineClient())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=2, sleep=0)) == 2

    assert captured[0]["result"]["status"] == "market_recovery_pending"
    assert captured[0]["result"]["steps_executed"] == 0
    statuses = [event.get("status") for event in captured[0]["events"] if isinstance(event, dict)]
    assert statuses == ["market_error_retry_limit", "market_error_retry_limit"]


def test_command_live_daily_trade_cap_counts_entries_not_closures(tmp_path, monkeypatch) -> None:
    class _CappedClient:
        def __init__(self) -> None:
            self.iteration = 0
            self.orders: list[tuple[str, str, float, bool]] = []

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            day = 24 * 60 * 60 * 1000
            closes_by_call = [
                [100.0] * (limit - 1) + [100.0],
                [100.0] * (limit - 1) + [110.0],
                [100.0] * (limit - 1) + [120.0],
            ]
            closes = closes_by_call[min(self.iteration, len(closes_by_call) - 1)]
            candles = []
            base_time = 0 if self.iteration < 2 else day
            for i, close in enumerate(closes):
                candles.append(
                    Candle(
                        open_time=base_time + i * 60_000,
                        open=close,
                        high=close * 1.001,
                        low=close * 0.999,
                        close=close,
                        volume=1.0,
                        close_time=base_time + (i + 1) * 60_000,
                    )
                )
            self.iteration += 1
            return candles

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size, dry_run))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _StepModel(_SignedFeeModelEvidence):
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.95, 0.0, 0.95]
            score = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return score

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(self.predict_proba(_features) >= threshold)

    captured: list[dict[str, object]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.01,
            max_position_pct=0.2,
            max_trades_per_day=1,
            cooldown_minutes=0,
            max_regime_unpredictability=1.0,
        )
    )
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _StepModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _CappedClient())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=3, sleep=0, paper=False, model=str(tmp_path / "missing-model.json"), leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    assert captured[0]["result"]["entries"] == 2


def test_command_live_rejects_non_testnet_runtime(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 2
    assert "Real-money execution is disabled" in capsys.readouterr().out


def test_command_live_rejects_missing_credentials_for_live_mode(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="", api_secret=""))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 2
    assert "Authenticated live mode requires Binance API key" in capsys.readouterr().err


def test_command_live_requests_strict_readiness_for_signed_gpu_mode(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_live_start_model(model_path, strategy, **kwargs):
        captured.update(kwargs)
        return None, "captured strict readiness", None

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="spot",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(StrategyConfig())
    monkeypatch.setattr(
        cli,
        "_workflow_compute_backend",
        lambda *_args, **_kwargs: (
            "directml",
            cli.BackendInfo("directml", "directml", "privateuseone:0", "DirectML", ""),
        ),
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_load_live_start_model", fake_load_live_start_model)

    assert cli.command_live(_live_args(tmp_path, steps=1, sleep=1, paper=False, live=True)) == 2
    assert captured["effective_dry_run"] is False
    assert captured["require_model_candidate_search"] is True
    assert captured["require_accelerator_evidence"] is True
    assert captured["require_live_data_evidence"] is True
    assert captured["require_microstructure_evidence"] is False
    assert captured["expected_symbol"] == "BTCUSDC"
    assert captured["expected_market_type"] == "spot"
    assert captured["expected_interval"] == "15m"


def test_command_live_futures_startup_does_not_call_set_leverage(tmp_path, monkeypatch, capsys) -> None:
    class _NoStartupLeverageClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            raise AssertionError("startup must not mutate futures leverage")

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="futures",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    strategy = StrategyConfig(leverage=5.0)
    save_strategy(strategy)
    model_file = tmp_path / "model.json"
    serialize_model(
        TrainedModel(
            weights=[0.0] * 13,
            bias=0.0,
            feature_dim=13,
            epochs=1,
            feature_means=[0.0] * 13,
            feature_stds=[1.0] * 13,
            feature_signature=cli._strategy_feature_signature(strategy),
            selection_risk={
                "passed": True,
                "effective_trials": 12,
                "selected_score": 0.12,
                "trial_penalty": 0.03,
                "deflated_score": 0.09,
            },
            execution_validation=_promoted_execution_validation(market_type="futures", interval="1s"),
            probability_calibration_size=128,
            probability_log_loss_before=0.62,
            probability_log_loss_after=0.58,
            probability_brier_before=0.24,
            probability_brier_after=0.22,
            probability_ece_before=0.10,
            probability_ece_after=0.08,
            probability_calibration_backend_requested="directml",
            probability_calibration_backend_kind="directml",
            probability_calibration_backend_device="privateuseone:0",
            training_backend_requested="directml",
            training_backend_kind="directml",
            training_backend_device="privateuseone:0",
            training_backend_vendor="DirectML",
            model_candidate_count=3,
            model_selected_candidate="triple_barrier_base",
            model_selection_score=0.42,
            strategy_overrides={"taker_fee_bps": 4.0},
        ),
        model_file,
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _NoStartupLeverageClient())
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda _runtime, _cfg: 5.0)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, model=str(model_file), leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    assert "Failed to set leverage" not in capsys.readouterr().err


def test_command_live_recovers_from_invalid_saved_model(tmp_path, monkeypatch, capsys) -> None:
    class _LiveClient:
        def __init__(self) -> None:
            self.orders: list[tuple[str, str, float, bool]] = []

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size, dry_run))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            normalized = max(constraints.min_qty, round(quantity, 4))
            return normalized, constraints

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=100.0))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    model_dir = tmp_path / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text("{}", encoding="utf-8")

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: (_ for _ in ()).throw(cli.ModelLoadError("bad model")))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    captured = capsys.readouterr()
    assert "Model load failed; regenerating" in captured.err


def test_command_live_strictly_requires_valid_model_for_authenticated_live(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())

    base_args = {
        "steps": 0,
        "sleep": 0,
        "paper": False,
        "live": False,
        "leverage": None,
        "retrain_interval": 0,
        "retrain_window": 300,
        "retrain_min_rows": 240,
    }

    missing = tmp_path / "missing-model.json"
    assert cli.command_live(argparse.Namespace(**{**base_args, "model": str(missing)})) == 2
    assert "Live mode needs model file" in capsys.readouterr().err

    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: (_ for _ in ()).throw(ModelLoadError("signature mismatch")))
    assert cli.command_live(argparse.Namespace(**{**base_args, "model": str(model_file)})) == 2
    assert "Live mode requires a compatible model" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("disk unavailable")))
    assert cli.command_live(argparse.Namespace(**{**base_args, "model": str(model_file)})) == 2
    assert "Live mode requires a readable model" in capsys.readouterr().err


def test_command_live_loaded_model_signature_and_authenticated_sleep_floor(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.5

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=0,
                sleep=0,
                paper=False,
                live=True,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert "minimum sleep=1s" in capsys.readouterr().out
    assert captured[0]["model_signature"] == "runtime-signature"


def test_command_live_records_feature_drift_warning(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []
    feature_count = 13
    row = SimpleNamespace(
        timestamp=60_000,
        close=100.0,
        features=(5.0,) + (0.0,) * (feature_count - 1),
        label=1,
    )
    model = TrainedModel(
        weights=[0.0] * feature_count,
        bias=0.0,
        feature_dim=feature_count,
        epochs=1,
        feature_means=[0.0] * feature_count,
        feature_stds=[1.0] * feature_count,
    )

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=True,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert "feature drift warning" in capsys.readouterr().out
    drift_events = [event for event in captured[0]["events"] if event["status"] == "feature_drift"]
    assert drift_events[0]["drift_status"] == "warn"


def test_command_live_blocks_failed_feature_drift(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []
    feature_count = 13
    row = SimpleNamespace(
        timestamp=60_000,
        close=100.0,
        features=(13.0,) + (0.0,) * (feature_count - 1),
        label=1,
    )
    model = TrainedModel(
        weights=[0.0] * feature_count,
        bias=0.0,
        feature_dim=feature_count,
        epochs=1,
        feature_means=[0.0] * feature_count,
        feature_stds=[1.0] * feature_count,
    )

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=True,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert "feature drift block" in capsys.readouterr().out
    drift_events = [event for event in captured[0]["events"] if event["status"] == "feature_drift"]
    assert drift_events[0]["drift_status"] == "fail"
    assert captured[0]["result"]["entries"] == 0


def test_command_live_halts_on_authenticated_feature_drift_check_failure(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []
    row = SimpleNamespace(timestamp=60_000, close=100.0, features=(1.0, 2.0), label=1)
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        selection_risk={
            "passed": True,
            "effective_trials": 12,
            "selected_score": 0.12,
            "trial_penalty": 0.03,
            "deflated_score": 0.09,
        },
        execution_validation=_promoted_execution_validation(market_type="spot", interval="1s"),
        probability_calibration_size=128,
        probability_log_loss_before=0.62,
        probability_log_loss_after=0.58,
        probability_brier_before=0.24,
        probability_brier_after=0.22,
        probability_ece_before=0.10,
        probability_ece_after=0.08,
        probability_calibration_backend_requested="directml",
        probability_calibration_backend_kind="directml",
        probability_calibration_backend_device="privateuseone:0",
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
        training_backend_vendor="DirectML",
        model_candidate_count=3,
        model_selected_candidate="triple_barrier_base",
        model_selection_score=0.42,
        strategy_overrides={"taker_fee_bps": 4.0},
    )

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="spot",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=False,
                live=True,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 2
    )
    assert "Live feature drift check failed" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "feature_drift_check_failed"
    assert captured[0]["events"][-1]["status"] == "feature_drift_check_failed"


def test_command_live_paper_recovers_after_feature_drift_check_failure(tmp_path, monkeypatch, capsys) -> None:
    row = SimpleNamespace(timestamp=60_000, close=100.0, features=(1.0, 2.0), label=1)
    incompatible = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    recovered = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=1,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
    )
    train_calls: list[int] = []

    def fake_train(*_args, **_kwargs):
        train_calls.append(1)
        return recovered

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: incompatible)
    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=True,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert "paper model incompatible; retraining" in capsys.readouterr().err
    assert train_calls == [1]


def test_paper_or_live_order_passes_reduce_only_for_authenticated_futures(capsys) -> None:
    class _OrderClient:
        def __init__(self) -> None:
            self.kwargs = {}

        def place_order(self, symbol, side, size, **kwargs):
            self.kwargs = kwargs
            return {"status": "FILLED", "symbol": symbol, "side": side, "size": size}

    client = _OrderClient()
    cli._paper_or_live_order(
        client,
        RuntimeConfig(market_type="futures"),
        StrategyConfig(),
        side="SELL",
        size=0.1,
        dry_run=False,
        leverage=2.0,
        reduce_only=True,
    )
    assert client.kwargs == {"dry_run": False, "leverage": 2.0, "reduce_only": True}
    assert "live order: SELL" in capsys.readouterr().out


def test_command_live_detects_existing_positions_and_failures(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    class _PositionClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {
                "positions": [
                    {
                        "symbol": "BTCUSDC",
                        "positionAmt": "0.2",
                        "entryPrice": "50000",
                        "positionInitialMargin": "100",
                    }
                ],
                "assets": [{"asset": "USDC", "availableBalance": "1000"}],
            }

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _PositionClient())
    live_args = argparse.Namespace(
        steps=0,
        sleep=1,
        paper=False,
        live=True,
        model=str(model_file),
        leverage=None,
        retrain_interval=0,
        retrain_window=300,
        retrain_min_rows=240,
    )
    assert cli.command_live(live_args) == 2
    assert "exchange exposure does not match the bot-owned local ledger" in capsys.readouterr().err

    position_id = "livepos001"
    PositionsStore().record_open(
        OpenPosition(
            id=position_id,
            symbol="BTCUSDC",
            market_type="futures",
            side="LONG",
            qty=0.2,
            entry_price=50_000.0,
            leverage=1.0,
            opened_at_ms=1,
            notional=10_000.0,
            dry_run=False,
            open_client_order_id=bot_client_order_id(position_id, "open"),
            exchange_status="FILLED",
        )
    )
    assert cli.command_live(live_args) == 0
    assert "Resuming bot-owned ledger position: long" in capsys.readouterr().out

    class _FailingPositionClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            raise BinanceAPIError("rate limit")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FailingPositionClient())
    assert cli.command_live(live_args) == 2
    assert "Account balance check failed: rate limit" in capsys.readouterr().err


def test_command_live_signed_entry_records_bot_owned_position(tmp_path, monkeypatch) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0, features=(0.1,), label=1, volume=1000.0)
        for index in range(20)
    ]

    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.95

    class _SignedEntryClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.submitted_client_ids: list[str] = []

        def get_max_leverage(self, symbol: str) -> int:
            return 10

        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {"positions": [], "assets": [{"asset": "USDC", "availableBalance": "1000"}]}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            reduce_only: bool = False,
            client_order_id: str | None = None,
        ):
            assert client_order_id is not None and client_order_id.startswith("sait-o-")
            self.submitted_client_ids.append(client_order_id)
            return {
                "status": "FILLED",
                "orderId": "101",
                "clientOrderId": client_order_id,
                "executedQty": str(size),
                "avgPrice": "100",
                "cummulativeQuoteQty": str(size * 100.0),
            }

    client = _SignedEntryClient()
    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="futures",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.001,
            max_position_pct=0.2,
            stop_loss_pct=0.01,
            signal_threshold=0.70,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            max_regime_unpredictability=1.0,
            dynamic_liquidity_session_enabled=False,
        )
    )
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_a, **_k: rows)
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_a, **_k: rows)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=1, sleep=1, paper=False, live=True, model=str(model_file))) == 0

    open_positions = PositionsStore().load_open()
    assert len(open_positions) == 1
    assert open_positions[0].dry_run is False
    assert open_positions[0].open_client_order_id == client.submitted_client_ids[0]
    assert open_positions[0].open_client_order_id.startswith("sait-o-")


def test_command_live_signed_close_records_ledger_and_removes_open(tmp_path, monkeypatch) -> None:
    position_id = "ownedclose001"
    open_client_id = bot_client_order_id(position_id, "open")
    PositionsStore().record_open(
        OpenPosition(
            id=position_id,
            symbol="BTCUSDC",
            market_type="futures",
            side="LONG",
            qty=0.2,
            entry_price=100.0,
            leverage=1.0,
            opened_at_ms=1,
            notional=20.0,
            dry_run=False,
            open_client_order_id=open_client_id,
            exchange_status="FILLED",
        )
    )
    rows = [
        ModelRow(timestamp=index * 60_000, close=102.0, features=(0.1,), label=1, volume=1000.0)
        for index in range(20)
    ]

    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.95

    class _SignedCloseClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.close_client_id = ""

        def get_max_leverage(self, symbol: str) -> int:
            return 10

        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {
                "positions": [{"symbol": "BTCUSDC", "positionAmt": "0.2", "entryPrice": "100"}],
                "assets": [{"asset": "USDC", "availableBalance": "1000"}],
            }

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            reduce_only: bool = False,
            client_order_id: str | None = None,
        ):
            assert side == "SELL"
            assert reduce_only is True
            assert client_order_id is not None and client_order_id.startswith("sait-c-")
            self.close_client_id = client_order_id
            return {
                "status": "FILLED",
                "orderId": "202",
                "clientOrderId": client_order_id,
                "executedQty": str(size),
                "avgPrice": "102",
                "cummulativeQuoteQty": str(size * 102.0),
            }

    client = _SignedCloseClient()
    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="futures",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(
        StrategyConfig(
            take_profit_pct=0.01,
            stop_loss_pct=0.10,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            max_regime_unpredictability=1.0,
            dynamic_liquidity_session_enabled=False,
        )
    )
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_a, **_k: rows)
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_a, **_k: rows)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=1, sleep=1, paper=False, live=True, model=str(model_file))) == 0

    store = PositionsStore()
    assert store.load_open() == []
    ledger = store.load_ledger()
    assert len(ledger) == 1
    assert ledger[0].id == position_id
    assert ledger[0].open_client_order_id == open_client_id
    assert ledger[0].close_client_order_id == client.close_client_id
    assert ledger[0].realized_pnl > 0.0


def test_command_live_caps_managed_cash_to_exchange_balance(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    class _BalanceClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {"positions": [], "assets": [{"asset": "USDC", "availableBalance": "25"}]}

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=100.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BalanceClient())
    args = argparse.Namespace(
        steps=0,
        sleep=1,
        paper=False,
        live=True,
        model=str(model_file),
        leverage=None,
        retrain_interval=0,
        retrain_window=300,
        retrain_min_rows=240,
    )

    assert cli.command_live(args) == 0
    assert "Authenticated live cash capped to exchange free USDC=25.00" in capsys.readouterr().out


def test_command_live_uses_runtime_quote_asset_for_cash_cap(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    class _UsdtBalanceClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {"positions": [], "assets": [{"asset": "USDT", "availableBalance": "30"}]}

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(
        RuntimeConfig(
            symbol="ETHUSDT",
            quote_asset="USDT",
            testnet=True,
            dry_run=False,
            market_type="futures",
            api_key="k",
            api_secret="s",
            managed_usdc=100.0,
        )
    )
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _UsdtBalanceClient())
    args = argparse.Namespace(
        steps=0,
        sleep=1,
        paper=False,
        live=True,
        model=str(model_file),
        leverage=None,
        retrain_interval=0,
        retrain_window=300,
        retrain_min_rows=240,
        external_signals=None,
    )

    assert cli.command_live(args) == 0
    assert "Authenticated live cash capped to exchange free USDT=30.00" in capsys.readouterr().out


def test_command_live_blocks_when_exchange_has_no_available_usdc(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    class _NoBalanceClient(_FakeClient):
        def get_account(self):
            return {"positions": [], "assets": [{"asset": "USDC", "availableBalance": "0"}]}

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=100.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _NoBalanceClient())

    assert cli.command_live(
        argparse.Namespace(
            steps=0,
            sleep=1,
            paper=False,
            live=True,
            model=str(model_file),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 2
    assert "requires available USDC" in capsys.readouterr().err


def test_command_live_reports_risk_policy_after_balance_check(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=100.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    monkeypatch.setattr(cli, "build_risk_policy_report", lambda *_a, **_k: SimpleNamespace(allowed=False, warning_count=0))
    monkeypatch.setattr(cli, "render_risk_policy_report", lambda _report: "Risk policy report denied")

    assert cli.command_live(
        argparse.Namespace(
            steps=0,
            sleep=1,
            paper=False,
            live=True,
            model=str(model_file),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 2
    assert "Risk policy report denied" in capsys.readouterr().err


def test_command_live_reconciliation_mismatch_after_balance(tmp_path, monkeypatch, capsys) -> None:
    class _SignedModel(_SignedFeeModelEvidence):
        feature_signature = "runtime-signature"

        def predict_proba(self, _features):
            return 0.5

    class _SignedClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def get_account(self):
            return {
                "positions": [
                    {
                        "symbol": "BTCUSDC",
                        "positionAmt": "0.1",
                        "entryPrice": "50000",
                        "positionInitialMargin": "100",
                    }
                ],
                "assets": [{"asset": "USDC", "availableBalance": "100"}],
            }

    monkeypatch.setenv("HOME", str(tmp_path))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=100.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _SignedModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _SignedClient())
    monkeypatch.setattr(cli, "_resolve_symbol_constraints", lambda *_a, **_k: SymbolConstraints("BTCUSDC", 0.00001, 1.0, 0.00001, 5.0, 1000.0))

    assert cli.command_live(
        argparse.Namespace(
            steps=0,
            sleep=1,
            paper=False,
            live=True,
            model=str(model_file),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 2
    assert "exchange exposure does not match the bot-owned local ledger" in capsys.readouterr().err


def test_command_tune_saves_candidate(monkeypatch, tmp_path) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(training_epochs=80))

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]

    data = tmp_path / "history.json"
    data.write_text(json.dumps(candles), encoding="utf-8")

    class _Model(_SignedFeeModelEvidence):
        def __init__(self, score: float) -> None:
            self.score = score

        def predict_proba(self, features: tuple[float, ...]) -> float:  # pragma: no cover
            return self.score

    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda rows, model, cfg, **_kwargs: SimpleNamespace(realized_pnl=1.0, total_fees=0.0),
    )
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model(0.8))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setenv("HOME", str(tmp_path))

    args = argparse.Namespace(
        input=str(data),
        save_best=True,
        min_risk=0.002,
        max_risk=0.003,
        steps=2,
        min_leverage=1.0,
        max_leverage=2.0,
        min_threshold=0.55,
        max_threshold=0.65,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )
    assert cli.command_tune(args) == 0


def test_loaders_handle_invalid_inputs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected candle list"):
        cli._load_json_candles(str(bad))

    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(
            [
                {
                    "open_time": 0,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                    "close_time": 60_000,
                },
                "bad-entry",
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = cli._rows_from_json(str(mixed))
    assert len(rows) == 1


def test_resolve_symbol_constraints_error_returns_none() -> None:
    runtime = RuntimeConfig(testnet=True)

    class _ErrorClient:
        def get_symbol_constraints(self, symbol: str):
            raise BinanceAPIError("boom")

    assert cli._resolve_symbol_constraints(runtime, _ErrorClient()) is None


def test_resolve_live_retrain_rows() -> None:
    rows = list(range(5))
    assert cli._resolve_live_retrain_rows(rows, retrain_window=10, retrain_min_rows=10) == []
    assert cli._resolve_live_retrain_rows(rows, retrain_window=10, retrain_min_rows=3) == rows
    assert cli._resolve_live_retrain_rows(list(range(15)), retrain_window=10, retrain_min_rows=3) == list(range(5, 15))


def test_build_live_model_retrain_interval(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=100)

    calls: list[tuple[int, int, str | None]] = []

    def fake_train(rows, epochs: int = 100, **_kwargs):
        calls.append((len(rows), epochs, _kwargs.get("feature_signature")))
        return f"model-{len(calls)}"

    monkeypatch.setattr(cli, "train", fake_train)

    base_rows = list(range(200))
    model = cli._build_live_model(
        base_rows,
        model=None,
        retrain_every=2,
        step=1,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-1"
    assert len(calls) == 1
    assert calls[-1][0] == 50
    assert calls[-1][2] == cli._strategy_feature_signature(cfg)

    model = cli._build_live_model(
        base_rows,
        model=model,
        retrain_every=2,
        step=3,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-1"
    assert len(calls) == 1

    model = cli._build_live_model(
        base_rows,
        model=model,
        retrain_every=2,
        step=4,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-2"
    assert len(calls) == 2
    assert calls[-1][0] == 50


def test_command_configure_save_and_validation_paths(tmp_path, monkeypatch, capsys) -> None:
    class _ValidClient(_FakeClient):
        def ping(self):
            return {"ok": True}

    class _FailingClient(_FakeClient):
        def ping(self):
            raise BinanceAPIError("bad")

        def ensure_btcusdc(self):
            raise BinanceAPIError("bad")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(api_key="k", api_secret="s", validate_account=True))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ValidClient())
    assert cli.command_configure(argparse.Namespace()) == 0
    first = capsys.readouterr()
    assert "Runtime config saved to" in first.out

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FailingClient())
    assert cli.command_configure(argparse.Namespace()) == 2
    captured = capsys.readouterr()
    assert "Configuration saved, but validation failed" in captured.err


def test_command_connect_paths_for_errors_and_leverage_exception(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="k", api_secret="s", market_type="futures", symbol="BTCUSDC"))

    class _ConnectionClient(_FakeClient):
        def get_exchange_time(self):
            return {"serverTime": 999}

        def get_account(self):
            return super().get_account()

        def get_max_leverage(self, symbol: str) -> int:
            return 10

    class _LeverageFailClient(_ConnectionClient):
        def get_max_leverage(self, symbol: str) -> int:
            raise BinanceAPIError("cant fetch")

    class _ExchangeErrorClient(_ConnectionClient):
        def get_exchange_time(self):
            raise BinanceAPIError("offline")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ConnectionClient())
    assert cli.command_connect(argparse.Namespace()) == 0

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LeverageFailClient())
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "unable to fetch leverage bracket" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ExchangeErrorClient())
    assert cli.command_connect(argparse.Namespace()) == 2


def test_command_strategy_covers_full_update_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig())

    class _LeverageClient(_SignedCommissionClientEvidence):
        def get_max_leverage(self, symbol: str) -> int:
            return 3

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LeverageClient())
    args = argparse.Namespace(
        leverage=12.0,
        risk=0.0,
        max_position=-1.0,
        stop=-0.5,
        take=99.0,
        max_open=0,
        max_trades_per_day=-5,
        cooldown=-10,
        signal_threshold=2.0,
        max_drawdown=-1.0,
        taker_fee_bps=-1.0,
        slippage_bps=-1.0,
        label_threshold=-1.0,
    )
    assert cli.command_strategy(args) == 0

    updated = load_strategy()
    assert updated.leverage == 3.0
    assert updated.risk_per_trade == 0.0001
    assert updated.max_position_pct == 0.0
    assert updated.stop_loss_pct == 0.0
    assert updated.take_profit_pct == 0.99
    assert updated.max_open_positions == 0
    assert updated.max_trades_per_day == 0
    assert updated.cooldown_minutes == 0
    assert updated.signal_threshold == 0.99
    assert updated.max_drawdown_limit == 0.0
    assert updated.taker_fee_bps == 0.0
    assert updated.slippage_bps == 0.0
    assert updated.label_threshold == 0.0001


def test_command_strategy_profiles_apply_and_explicit_args_override(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="spot"))
    save_strategy(StrategyConfig())

    def args_for(profile: str, **overrides):
        defaults = {
            "profile": profile,
            "leverage": None,
            "risk": None,
            "max_position": None,
            "stop": None,
            "take": None,
            "cooldown": None,
            "max_open": None,
            "max_trades_per_day": None,
            "signal_threshold": None,
            "max_drawdown": None,
            "taker_fee_bps": None,
            "slippage_bps": None,
            "label_threshold": None,
            "model_lookback": None,
            "training_epochs": None,
            "confidence_beta": None,
            "feature_window_short": None,
            "feature_window_long": None,
            "set_features": None,
            "enable_feature": None,
            "disable_feature": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    assert cli.command_strategy(args_for("conservative")) == 0
    conservative = load_strategy()
    assert conservative.risk_per_trade == 0.003
    assert conservative.signal_threshold == 0.66
    assert conservative.training_epochs == 180

    assert cli.command_strategy(args_for("active", risk=0.003, signal_threshold=0.7)) == 0
    active = load_strategy()
    assert active.leverage == pytest.approx(DEFAULT_AGGRESSIVE_LEVERAGE)
    assert active.risk_per_trade == 0.003
    assert active.signal_threshold == 0.7

    assert cli.command_strategy(args_for("bad-profile")) == 2
    assert "Invalid strategy profile" in capsys.readouterr().err


def test_tui_strategy_profile_uses_unchanged_fields_as_profile_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = StrategyConfig()
    save_strategy(cfg)

    payload = {
        "profile": "active",
        "leverage": str(cfg.leverage),
        "risk": str(cfg.risk_per_trade),
        "max_position": str(cfg.max_position_pct),
        "stop": str(cfg.stop_loss_pct),
        "take": str(cfg.take_profit_pct),
        "cooldown": str(cfg.cooldown_minutes),
        "max_open": str(cfg.max_open_positions),
        "max_trades_per_day": str(cfg.max_trades_per_day),
        "signal_threshold": str(cfg.signal_threshold),
        "max_drawdown": str(cfg.max_drawdown_limit),
        "max_daily_loss": str(cfg.max_daily_loss_pct),
        "max_session_loss": str(cfg.max_session_loss_pct),
        "max_consecutive_losses": str(cfg.max_consecutive_losses),
        "max_network_errors": str(cfg.max_network_errors),
        "recovery_cooldown_seconds": str(cfg.recovery_cooldown_seconds),
        "taker_fee_bps": str(cfg.taker_fee_bps),
        "slippage_bps": str(cfg.slippage_bps),
        "label_threshold": str(cfg.label_threshold),
        "model_lookback": str(cfg.model_lookback),
        "training_epochs": str(cfg.training_epochs),
        "confidence_beta": str(cfg.confidence_beta),
        "feature_window_short": str(cfg.feature_windows[0]),
        "feature_window_long": str(cfg.feature_windows[1]),
        "external_signals": str(cfg.external_signals_enabled),
        "external_signal_max_adjustment": str(cfg.external_signal_max_adjustment),
        "external_signal_min_providers": str(cfg.external_signal_min_providers),
        "external_signal_ttl": str(cfg.external_signal_ttl_seconds),
        "external_signal_timeout": str(cfg.external_signal_timeout_seconds),
        "external_news_ai": str(cfg.external_news_ai_enabled),
        "external_news_ai_model": str(cfg.external_news_ai_model),
        "external_news_provider_limit": str(cfg.external_signal_news_provider_limit),
        "external_provider_parallelism": str(cfg.external_signal_provider_parallelism),
        "external_provider_jitter": str(cfg.external_signal_provider_jitter_seconds),
        "external_poll_jitter": str(cfg.external_signal_poll_jitter_seconds),
        "telemetry_db": str(cfg.telemetry_db_path),
        "source_grading": str(cfg.source_grading_enabled),
        "source_grading_interval": str(cfg.source_grading_interval_seconds),
        "source_grade_max_age_hours": "72",
    }

    class _UI:
        async def multi_select(self, *_args, **_kwargs):
            return ["momentum_1", "rsi"]

        async def form(self, *_args, **_kwargs):
            return payload

    args = asyncio.run(cli._ui_edit_strategy_args(_UI(), StrategyConfig()))
    assert args.profile == "active"
    assert args.leverage is None
    assert args.risk is None
    assert args.feature_window_short is None
    assert args.set_features == "momentum_1,rsi"

    save_runtime(RuntimeConfig(market_type="spot"))
    assert cli.command_strategy(args) == 0
    updated = load_strategy()
    assert updated.leverage == pytest.approx(DEFAULT_AGGRESSIVE_LEVERAGE)
    assert updated.risk_per_trade == 0.010
    assert updated.signal_threshold == 0.55
    assert updated.external_signals_enabled is True
    assert updated.source_grade_max_age_hours == 72.0


def test_existing_position_detection_helpers() -> None:
    assert cli._safe_float("1.25") == 1.25
    assert cli._safe_float(object()) == 0.0
    assert cli._detect_existing_position(RuntimeConfig(), object(), leverage=1.0) is None

    class NonDictAccount:
        def get_account(self):
            return []

    assert cli._detect_existing_position(RuntimeConfig(), NonDictAccount(), leverage=1.0) is None

    class FuturesAccount:
        def get_account(self):
            return {
                "positions": [
                    object(),
                    {"symbol": "BTCUSDC", "positionAmt": "0", "entryPrice": "50000"},
                    {"symbol": "ETHUSDC", "positionAmt": "1"},
                    {"symbol": "BTCUSDC", "positionAmt": "-0.2", "entryPrice": "50000", "positionInitialMargin": "100"},
                ]
            }

    futures = cli._detect_existing_position(
        RuntimeConfig(market_type="futures", symbol="BTCUSDC"),
        FuturesAccount(),
        leverage=5.0,
    )
    assert futures == {
        "market": "futures",
        "side": -1,
        "qty": 0.2,
        "entry_price": 50000.0,
        "notional": 10000.0,
        "margin": 100.0,
    }
    assert (
        cli._detect_existing_position(
            RuntimeConfig(market_type="futures", symbol="BTCUSDC"),
            type("FlatFutures", (), {"get_account": lambda self: {"positions": [{"symbol": "BTCUSDC", "positionAmt": "0"}]}})(),
            leverage=5.0,
        )
        is None
    )

    class SpotAccount:
        def get_account(self):
            return {"balances": [object(), {"asset": "ETH", "free": "1", "locked": "0"}, {"asset": "BTC", "free": "0", "locked": "0"}, {"asset": "BTC", "free": "0.01", "locked": "0.02"}]}

        def get_symbol_price(self, symbol: str):
            return 60000.0, 123

    assert cli._detect_existing_position(RuntimeConfig(market_type="spot"), SpotAccount(), leverage=1.0, reference_price=60000.0) is None

    spot = cli._detect_existing_position(
        RuntimeConfig(market_type="spot", managed_btc=0.02),
        SpotAccount(),
        leverage=1.0,
        reference_price=60000.0,
    )
    assert spot["market"] == "spot"
    assert spot["qty"] == 0.02
    assert spot["entry_price"] == 60000.0
    assert (
        cli._detect_existing_position(RuntimeConfig(market_type="spot", managed_btc=0.01), SpotAccount(), leverage=1.0)["entry_price"]
        == 60000.0
    )
    eth_spot = cli._detect_existing_position(
        RuntimeConfig(market_type="spot", symbol="ETHUSDT", quote_asset="USDT", managed_btc=0.5),
        SpotAccount(),
        leverage=1.0,
        reference_price=3000.0,
    )
    assert eth_spot["qty"] == 0.5
    assert eth_spot["entry_price"] == 3000.0

    assert (
        cli._detect_existing_position(
            RuntimeConfig(market_type="spot", managed_btc=1.0),
            type("NoBtcAccount", (), {"get_account": lambda self: {"balances": [{"asset": "BTC", "free": "0", "locked": "0"}]}})(),
            leverage=1.0,
        )
        is None
    )


def test_command_fetch_success_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _FetchClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FetchClient())
    out = tmp_path / "history.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, output=str(out))) == 0
    assert out.exists()
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 10


def test_command_fetch_batches_large_requests(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _PagedFetchClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[int, int | None]] = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            self.calls.append((limit, end_time))
            latest_index = 1204 if end_time is None else end_time // 60_000
            first_index = max(0, latest_index - limit + 1)
            candles = []
            for index in range(first_index, latest_index + 1):
                candles.append(
                    Candle(
                        open_time=index * 60_000,
                        open=100.0 + index,
                        high=101.0 + index,
                        low=99.0 + index,
                        close=100.0 + index,
                        volume=1.0,
                        close_time=(index + 1) * 60_000,
                    )
                )
            return candles

    client = _PagedFetchClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    out = tmp_path / "history.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=1205, batch_size=500, output=str(out))) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1205
    assert [call[0] for call in client.calls] == [500, 500, 205]
    assert client.calls[0][1] is None
    assert client.calls[1][1] == 705 * 60_000 - 1


def test_command_fetch_stops_on_empty_or_short_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _EmptyFetchClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return []

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _EmptyFetchClient())
    empty_out = tmp_path / "empty.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, batch_size=5, output=str(empty_out))) == 0
    assert json.loads(empty_out.read_text(encoding="utf-8")) == []

    class _ShortFetchClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(2)

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ShortFetchClient())
    short_out = tmp_path / "short.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, batch_size=5, output=str(short_out))) == 0
    assert len(json.loads(short_out.read_text(encoding="utf-8"))) == 2


def test_command_fetch_stops_when_batch_adds_no_new_candles(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _DuplicateFetchClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            self.calls += 1
            if self.calls > 2:
                raise AssertionError("fetch did not stop after duplicate page")
            return [
                Candle(
                    open_time=(100 + i) * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(101 + i) * 60_000,
                )
                for i in range(limit)
            ]

    client = _DuplicateFetchClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    out = tmp_path / "duplicate.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=12, batch_size=6, output=str(out))) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 6
    assert client.calls == 2


def test_command_fetch_exits_after_exact_limit_and_after_short_late_page(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _ExactLimitClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=(50 + i) * 60_000,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1.0,
                    close_time=(51 + i) * 60_000,
                )
                for i in range(limit)
            ]

    exact_out = tmp_path / "exact.json"
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ExactLimitClient())
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=6, batch_size=6, output=str(exact_out))) == 0
    assert len(json.loads(exact_out.read_text(encoding="utf-8"))) == 6

    class _ShortLateClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=(80 + i) * 60_000,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1.0,
                    close_time=(81 + i) * 60_000,
                )
                for i in range(2)
            ]

    short_out = tmp_path / "short-late.json"
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ShortLateClient())
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=6, batch_size=5, output=str(short_out))) == 0
    assert len(json.loads(short_out.read_text(encoding="utf-8"))) == 2


def test_command_train_empty_rows_and_walk_forward_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert cli.command_train(
        argparse.Namespace(
            input=str(empty),
            output=str(tmp_path / "model.json"),
            epochs=2,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 2

    candles = []
    for i in range(220):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")
    model = tmp_path / "model.json"

    wf_calls = []

    def fake_wf(*_args, **_kwargs):
        wf_calls.append(1)
        return {
            "folds": [0.1, 0.2],
            "scores": [0.3, 0.5],
            "average_score": 0.4,
            "train_window": 10,
            "test_window": 5,
            "step": 1,
        }

    monkeypatch.setattr(cli, "walk_forward_report", fake_wf)
    assert (
        cli.command_train(
            argparse.Namespace(
                input=str(history),
                output=str(model),
                epochs=4,
                walk_forward=True,
                walk_forward_train=10,
                walk_forward_test=5,
                walk_forward_step=1,
                calibrate_threshold=True,
            )
        )
        == 0
    )
    assert wf_calls == [1]


def test_command_train_walk_forward_errors_do_not_abort_training(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setattr(cli, "walk_forward_report", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unavailable")))

    assert (
        cli.command_train(
            argparse.Namespace(
                input=str(history),
                output=str(tmp_path / "model.json"),
                epochs=4,
                walk_forward=True,
                walk_forward_train=10,
                walk_forward_test=5,
                walk_forward_step=1,
                calibrate_threshold=True,
            )
        )
        == 0
    )


def test_command_tune_uses_default_leverage_when_no_api_keys(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="futures", api_key="", api_secret=""))
    save_strategy(StrategyConfig(training_epochs=80))

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(220)
    ]
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    class _Model(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.6

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model())
    monkeypatch.setattr(cli, "run_backtest", lambda *a, **k: SimpleNamespace(realized_pnl=0.0, total_fees=0.0))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    args = argparse.Namespace(
        input=str(history),
        save_best=True,
        min_risk=0.002,
        max_risk=0.003,
        steps=2,
        min_leverage=2.0,
        max_leverage=3.0,
        min_threshold=0.52,
        max_threshold=0.53,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )

    assert cli.command_tune(args) == 0
    assert "tune best score" in capsys.readouterr().out


def test_tune_score_penalizes_drawdown_stops() -> None:
    bad = SimpleNamespace(realized_pnl=200.0, total_fees=1.0, max_drawdown=0.5, stopped_by_drawdown=True, closed_trades=1)
    good = SimpleNamespace(realized_pnl=120.0, total_fees=1.0, max_drawdown=0.01, stopped_by_drawdown=False, closed_trades=2)
    assert cli._tune_score(bad, starting_cash=1000.0) < cli._tune_score(good, starting_cash=1000.0)


def test_command_tune_falls_back_to_drawdown_fallback(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(training_epochs=80))

    candles = []
    for i in range(240):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    class _Model(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.6

    calls = []

    def _mock_run_backtest(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(realized_pnl=float(len(calls)), total_fees=0.0, max_drawdown=0.9, stopped_by_drawdown=True, closed_trades=1)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model())
    monkeypatch.setattr(cli, "run_backtest", _mock_run_backtest)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    args = argparse.Namespace(
        input=str(history),
        save_best=False,
        min_risk=0.002,
        max_risk=0.004,
        steps=2,
        min_leverage=2.0,
        max_leverage=3.0,
        min_threshold=0.52,
        max_threshold=0.53,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )
    assert cli.command_tune(args) == 0
    assert len(calls) > 0
    assert "all tune candidates hit drawdown limit" in capsys.readouterr().out


def test_command_live_uses_generated_model_when_saved_model_is_invalid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=100.0))
    save_strategy(StrategyConfig(risk_per_trade=0.002, max_position_pct=0.2))

    model_file = tmp_path / "data" / "model.json"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("{invalid-json}", encoding="utf-8")

    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(open_time=i * 60_000, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i, volume=1.0, close_time=(i + 1) * 60_000)
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return (max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol))

        def place_order(self, symbol: str, side: str, quantity: float, dry_run: bool, leverage: float):
            return {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "dry_run": dry_run,
                "leverage": leverage,
            }

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0


def test_command_live_skips_entry_when_cash_is_insufficient_after_fees(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=1.0, max_position_pct=1.0, taker_fee_bps=20000.0))

    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=0,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1.0,
                    close_time=60_000,
                )
                for _ in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.01,
                max_qty=100.0,
                step_size=0.01,
                min_notional=10.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return (max(0.01, round(quantity, 2)), self.get_symbol_constraints(symbol))

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0


def test_command_live_persists_model_incompatibility_in_authenticated_live(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _FlowClient(_SignedCommissionClientEvidence):
        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _BadRuntimeModel(_SignedFeeModelEvidence):
        feature_signature = "test-signature"

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            raise ValueError("feature vector changed")

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="spot",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _BadRuntimeModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=False,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 2
    )
    assert "Live model incompatible with current rows" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "model_incompatible"
    assert captured[0]["events"][-1]["status"] == "model_incompatible"


def test_command_live_paper_retrains_incompatible_model(tmp_path, monkeypatch, capsys) -> None:
    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _BadRuntimeModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            raise ValueError("feature vector changed")

    class _RecoveredModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.5

    train_calls: list[int] = []

    def fake_train(*_args, **_kwargs):
        train_calls.append(1)
        return _RecoveredModel()

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _BadRuntimeModel())
    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=True,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert train_calls == [1]
    assert "paper model incompatible; retraining" in capsys.readouterr().err


def test_command_live_reports_no_training_rows_when_model_cannot_be_built(tmp_path, monkeypatch, capsys) -> None:
    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

    row = ModelRow(timestamp=60_000, close=100.0, features=(0.1,), label=0)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_build_live_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=1)) == 0
    assert "not enough labeled historical data to train a live model" in capsys.readouterr().out


def test_command_live_paper_incompatible_model_without_training_rows_halts(tmp_path, monkeypatch, capsys) -> None:
    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

    class _BadRuntimeModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            raise ValueError("feature vector changed")

    row = ModelRow(timestamp=60_000, close=100.0, features=(0.1,), label=0)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: _BadRuntimeModel())
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, model=str(model_file), steps=1)) == 2
    captured = capsys.readouterr()
    assert "paper model incompatible; retraining" in captured.err
    assert "not enough labeled historical data to retrain a live model" in captured.err


def test_command_live_persists_entry_order_error(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _RejectEntryClient(_SignedCommissionClientEvidence):
        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            client_order_id: str | None = None,
        ):
            raise BinanceAPIError("Filter failure: NOTIONAL")

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RejectEntryClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=False,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 2
    )
    assert "order error" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"
    assert captured[0]["result"]["entries"] == 0
    assert captured[0]["events"][-1]["status"] == "order_error"


def test_command_live_persists_close_order_error(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _RejectCloseClient(_SignedCommissionClientEvidence):
        def __init__(self) -> None:
            self.order_calls = 0

        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            client_order_id: str | None = None,
        ):
            self.order_calls += 1
            if self.order_calls == 2:
                raise BinanceAPIError("close rejected")
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100", "dry_run": dry_run}

    class _OpenThenCloseModel(_SignedFeeModelEvidence):
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            return 0.99 if self.calls == 1 else 0.0

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2, cooldown_minutes=0))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RejectCloseClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _OpenThenCloseModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=2,
                sleep=0,
                paper=False,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 2
    )
    assert "order error" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"
    assert captured[0]["result"]["entries"] == 1
    assert captured[0]["result"]["closes"] == 0
    assert captured[0]["events"][-1]["side"] == "SELL"


def test_command_live_persists_emergency_close_order_error(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _RejectEmergencyCloseClient(_SignedCommissionClientEvidence):
        def __init__(self) -> None:
            self.market_calls = 0
            self.order_calls = 0

        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            close = 100.0 if self.market_calls == 0 else 80.0
            self.market_calls += 1
            return [
                Candle(
                    open_time=i * 60_000,
                    open=close,
                    high=close * 1.001,
                    low=close * 0.999,
                    close=close,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            client_order_id: str | None = None,
        ):
            self.order_calls += 1
            if self.order_calls == 2:
                raise BinanceAPIError("emergency close rejected")
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100", "dry_run": dry_run}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            market_type="spot",
            interval="1s",
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        )
    )
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=0.2,
            max_drawdown_limit=0.01,
            stop_loss_pct=0.02,
            cooldown_minutes=0,
            max_regime_unpredictability=1.0,
        )
    )
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _RejectEmergencyCloseClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=2,
                sleep=0,
                paper=False,
                live=False,
                model=str(model_file),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 2
    )
    assert "order error" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"
    assert captured[0]["result"]["entries"] == 1
    assert captured[0]["events"][-1]["side"] == "SELL"


def test_command_live_records_malformed_entry_fill_response(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _EntryClient(_FakeClient):
        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            client_order_id: str | None = None,
        ):
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100"}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _EntryClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_order_fill_details", lambda *_a, **_k: (0.0, 0.0, 0.0))
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, _output_dir, payload: captured.append(payload) or tmp_path / "live.json")
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(
        argparse.Namespace(
            steps=1,
            sleep=1,
            paper=False,
            live=True,
            model=str(model_file),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 2
    assert "Order response did not include executed quantity" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"


def test_command_live_records_malformed_regular_close_fill_response(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _CloseClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100"}

    class _OpenThenFlatModel(_SignedFeeModelEvidence):
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            return 0.99 if self.calls == 1 else 0.01

    fill_calls = {"count": 0}

    def fake_fill(*_args, **_kwargs):
        fill_calls["count"] += 1
        return (1.0, 100.0, 100.0) if fill_calls["count"] == 1 else (0.0, 0.0, 0.0)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2, cooldown_minutes=0))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _CloseClient())
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: _OpenThenFlatModel())
    monkeypatch.setattr(cli, "_order_fill_details", fake_fill)
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, _output_dir, payload: captured.append(payload) or tmp_path / "live.json")
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=2, paper=False, model=str(tmp_path / "missing.json"))) == 2
    assert "Close order response did not include executed quantity" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"


def test_command_live_records_malformed_emergency_close_fill_response(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _EmergencyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.market_calls = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            close = 100.0 if self.market_calls == 0 else 10.0
            self.market_calls += 1
            return [
                Candle(
                    open_time=i * 60_000,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100"}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    fill_calls = {"count": 0}

    def fake_fill(*_args, **_kwargs):
        fill_calls["count"] += 1
        return (1.0, 100.0, 100.0) if fill_calls["count"] == 1 else (0.0, 0.0, 0.0)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=0.5,
            take_profit_pct=0.95,
            stop_loss_pct=0.99,
            max_drawdown_limit=0.01,
            max_regime_unpredictability=1.0,
        )
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _EmergencyClient())
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_order_fill_details", fake_fill)
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, _output_dir, payload: captured.append(payload) or tmp_path / "live.json")
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(_live_args(tmp_path, steps=2, paper=False, model=str(tmp_path / "missing.json"))) == 2
    assert "Emergency close order response did not include executed quantity" in capsys.readouterr().err
    assert captured[0]["result"]["status"] == "order_error"


def test_command_live_handles_partial_regular_close(tmp_path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class _PartialCloseClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.order_calls = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.order_calls += 1
            executed = size if self.order_calls == 1 else size / 2.0
            return {"symbol": symbol, "side": side, "executedQty": str(executed), "avgPrice": "100"}

    class _OpenThenFlatModel(_SignedFeeModelEvidence):
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            return 0.99 if self.calls == 1 else 0.01

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2, cooldown_minutes=0))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _PartialCloseClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _OpenThenFlatModel())
    monkeypatch.setattr(cli, "_build_order_notional", lambda *_a, **_k: (100.0, 1.0))
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, _output_dir, payload: captured.append(payload) or tmp_path / "live.json")
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(
        argparse.Namespace(
            steps=2,
            sleep=0,
            paper=False,
            live=False,
            model=str(tmp_path / "missing.json"),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 0
    close_events = [event for event in captured[0]["events"] if event.get("status") == "close"]
    assert close_events
    assert close_events[0]["qty_closed"] == 0.5


def test_command_live_handles_partial_emergency_close(tmp_path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class _PartialEmergencyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.market_calls = 0
            self.order_calls = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            close = 100.0 if self.market_calls == 0 else 10.0
            self.market_calls += 1
            return [
                Candle(
                    open_time=i * 60_000,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(symbol=symbol, min_qty=0.0001, max_qty=100.0, step_size=0.0001, min_notional=1.0, max_notional=0.0)

        def normalize_quantity(self, symbol: str, quantity: float):
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.order_calls += 1
            executed = size if self.order_calls == 1 else size / 2.0
            return {"symbol": symbol, "side": side, "executedQty": str(executed), "avgPrice": "100"}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.01,
            max_position_pct=0.2,
            take_profit_pct=0.95,
            stop_loss_pct=0.99,
            max_drawdown_limit=0.01,
            max_regime_unpredictability=1.0,
        )
    )
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _PartialEmergencyClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_order_notional", lambda *_a, **_k: (100.0, 1.0))
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_persist_run_artifact", lambda _kind, _output_dir, payload: captured.append(payload) or tmp_path / "live.json")
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(
        argparse.Namespace(
            steps=2,
            sleep=0,
            paper=False,
            live=False,
            model=str(tmp_path / "missing.json"),
            leverage=None,
            retrain_interval=0,
            retrain_window=300,
            retrain_min_rows=240,
            external_signals=None,
        )
    ) == 0
    emergency_events = [event for event in captured[0]["events"] if event.get("status") == "emergency_close"]
    assert emergency_events
    assert captured[0]["result"]["drawdown_seen"] >= 0.01


def test_command_live_skips_entry_when_cash_is_insufficient_before_fill(tmp_path, monkeypatch, capsys) -> None:
    captured: list[dict[str, object]] = []

    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        assert kind == "live"
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=100.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_order_notional", lambda *_a, **_k: (1000.0, 10.0))
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert (
        cli.command_live(
            argparse.Namespace(
                steps=1,
                sleep=0,
                paper=False,
                live=False,
                model=str(tmp_path / "missing-model.json"),
                leverage=None,
                retrain_interval=0,
                retrain_window=300,
                retrain_min_rows=240,
            )
        )
        == 0
    )
    assert "insufficient cash for leverage-adjusted entry" in capsys.readouterr().out
    assert captured[0]["result"]["skipped_entries"] == 1
    assert captured[0]["events"][-1]["status"] == "skip_insufficient_cash_pre_fill"


def test_command_tune_data_insufficient(tmp_path) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    no_data = Path(tmp_path) / "small.json"
    no_data.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        input=str(no_data),
        save_best=False,
        min_risk=0.002,
        max_risk=0.02,
        steps=5,
        min_leverage=1.0,
        max_leverage=2.0,
        min_threshold=0.52,
        max_threshold=0.88,
        min_take=0.01,
        max_take=0.06,
        min_stop=0.008,
        max_stop=0.04,
    )
    assert cli.command_tune(args) == 2


def test_command_live_retrain_interval_rebuilds_model_in_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _FlowClient(_SignedCommissionClientEvidence):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def __init__(self):
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            return 0.99

    train_calls = []
    train_signatures = []

    def fake_train(rows, epochs: int, **_kwargs):
        train_calls.append(len(rows))
        train_signatures.append(_kwargs.get("feature_signature"))
        return _AlwaysLongModel()

    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", fake_train)

    assert cli.command_live(
        argparse.Namespace(
            steps=3,
            sleep=5,
            paper=False,
            model=str(tmp_path / "missing-model.json"),
            retrain_interval=2,
            retrain_window=120,
            retrain_min_rows=100,
        )
    ) == 0

    # initial build at step 1 + rebuild at step 2 (interval=2), no rebuild at step 3
    assert train_calls == [120, 120]
    assert train_signatures == [cli._strategy_feature_signature(load_strategy())] * 2



def test_command_live_rejects_live_when_not_testnet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=False, api_key="k", api_secret="s"))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_allows_demo_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, demo=True, dry_run=False, api_key="k", api_secret="s"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda runtime, cfg: 1.0)
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *args, **kwargs: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False, live=False, leverage=None)) == 2


def test_command_live_futures_does_not_set_leverage_before_market_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig(leverage=5.0))

    class _NoStartupLeverageClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            raise BinanceAPIError("market data unavailable")

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.001,
                max_qty=100.0,
                step_size=0.001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            raise AssertionError("should not normalize if leverage fails")

        def set_leverage(self, _symbol: str, _leverage: int):
            raise AssertionError("startup must not mutate futures leverage")

        def get_max_leverage(self, symbol: str) -> int:
            return 10

        def get_account(self):
            return {"assets": [{"asset": "USDC", "availableBalance": "1000"}]}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _NoStartupLeverageClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_futures_preflight_blocks_before_leverage_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=0.0))
    save_strategy(StrategyConfig(leverage=5.0))
    calls: list[str] = []

    class _PreflightClient:
        def get_account(self):
            calls.append("get_account")
            return {"assets": [{"asset": "USDC", "availableBalance": "1000"}]}

        def set_leverage(self, _symbol: str, _leverage: int):
            calls.append("set_leverage")
            raise AssertionError("leverage must not be changed before live preflight passes")

        def get_max_leverage(self, symbol: str) -> int:
            return 10

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _PreflightClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2
    assert calls == ["get_account"]


def test_command_backtest_model_missing_and_success(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_trading.model import serialize_model

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    input_file = tmp_path / "hist.json"
    input_file.write_text("[]", encoding="utf-8")
    missing_model = tmp_path / "missing.json"
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(missing_model), start_cash=1000.0)) == 2

    candles = []
    for i in range(120):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file = tmp_path / "model.json"
    serialize_model(
        TrainedModel(
            weights=[0.0] * 13,
            bias=0.0,
            feature_dim=13,
            epochs=1,
            feature_means=[0.0] * 13,
            feature_stds=[1.0] * 13,
            feature_signature=cli._strategy_feature_signature(StrategyConfig()),
        ),
        model_file,
    )
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 0
    assert "trades:" in capsys.readouterr().out


def test_command_live_rejects_non_testnet_and_missing_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_strategy(StrategyConfig())

    save_runtime(RuntimeConfig(testnet=False))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2

    save_runtime(RuntimeConfig(testnet=True, dry_run=False, api_key="", api_secret=""))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_detailed_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _FlowClient(_SignedCommissionClientEvidence):
        def __init__(self):
            self.orders = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            base = 100.0
            return [
                Candle(open_time=i * 60_000, open=base + i, high=base + i + 1, low=base + i - 1, close=base + i, volume=1.0, close_time=(i + 1) * 60_000)
                for i in range(320)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((side, size))
            return {"symbol": symbol, "side": side, "size": size}

    class _ScoredModel(_SignedFeeModelEvidence):
        def __init__(self):
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            if self.calls == 1:
                return 0.95
            if self.calls == 2:
                return 0.5
            return 0.95

    client = _FlowClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _ScoredModel())

    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", max_rate_calls_per_minute=1))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2, max_trades_per_day=1, training_epochs=40, cooldown_minutes=1))
    assert cli.command_live(argparse.Namespace(steps=3, sleep=5, paper=False, model=str(tmp_path / "missing-model.json"))) == 0

    # futures startup should not mutate leverage before a fresh entry exists.
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=1000.0, max_rate_calls_per_minute=1))

    class _SetLeverageErrorClient(_FlowClient):
        def get_max_leverage(self, symbol: str) -> int:
            return 10

        def get_account(self):
            return {"assets": [{"asset": "USDC", "availableBalance": "1000"}]}

        def set_leverage(self, symbol: str, leverage: int):
            raise AssertionError("startup must not mutate futures leverage")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _SetLeverageErrorClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_futures_leverage_override(tmp_path, monkeypatch, capsys) -> None:
    class _LeverageClient(_SignedCommissionClientEvidence):
        def __init__(self) -> None:
            self.set_calls: list[int] = []
            self.bracket_queries: list[tuple[str, float]] = []
            self.orders: list[tuple[str, float, bool, float, float | None]] = []

        def get_account(self):
            return _exchange_account()

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=i * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def get_max_leverage(self, symbol: str) -> int:
            return 20

        def get_max_leverage_for_notional(self, symbol: str, notional: float) -> int:
            self.bracket_queries.append((symbol, notional))
            return 8

        def set_leverage(self, symbol: str, leverage: int):
            self.set_calls.append(leverage)
            return {"symbol": symbol, "leverage": leverage}

        def place_order(
            self,
            symbol: str,
            side: str,
            size: float,
            *,
            dry_run: bool,
            leverage: float = 1.0,
            notional: float | None = None,
            reduce_only: bool = False,
            client_order_id: str | None = None,
        ):
            self.orders.append((side, size, dry_run, leverage, notional))
            return {"symbol": symbol, "side": side, "executedQty": str(size), "avgPrice": "100"}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", managed_usdc=1000.0))
    save_strategy(StrategyConfig(risk_per_trade=0.005, max_position_pct=0.5))
    model_file = tmp_path / "model.json"
    model_file.write_text("{}", encoding="utf-8")

    client = _LeverageClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())

    assert (
        cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False, model=str(model_file), leverage=12.0))
        == 0
    )
    assert client.set_calls == []
    assert client.orders
    assert client.bracket_queries
    assert client.orders[0][3] == 8.0
    assert client.orders[0][4] == pytest.approx(client.bracket_queries[0][1])
    open_positions = PositionsStore().load_open()
    assert open_positions
    assert open_positions[0].leverage == pytest.approx(8.0)
    assert "effective leverage" in capsys.readouterr().out


def test_command_live_spot_leverage_override_is_logged(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _SpotClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=i * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size}

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _SpotClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False, leverage=10.0)) == 0
    assert "Leverage override is spot-inactive" in capsys.readouterr().out


def test_command_live_drawdown_limit_forces_emergency_close(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", managed_usdc=1000.0))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=1.0,
            max_asset_allocation_pct=1.0,
            take_profit_pct=0.95,
            stop_loss_pct=0.99,
                max_drawdown_limit=0.20,
                feature_windows=(4, 20),
                max_regime_unpredictability=1.0,
            )
        )

    class _DrawdownClient:
        call_count = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            if interval != "15m":
                return []

            self.call_count += 1
            if self.call_count == 1:
                close = 100.0
            else:
                close = 10.0

            candles = [
                Candle(
                    open_time=i * 60_000,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(60)
            ]
            return candles

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=1000.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {
                "symbol": symbol,
                "side": side,
                "size": size,
                "dry_run": dry_run,
                "leverage": leverage,
            }

    class _AlwaysLongModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 1.0

    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _DrawdownClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())

    assert cli.command_live(argparse.Namespace(steps=3, sleep=5, paper=False, model=str(tmp_path / "missing-model.json"))) == 0
    output = capsys.readouterr().out
    assert "emergency close from drawdown" in output
    assert "drawdown limit reached" in output


def test_build_autonomous_decision_fn_error_and_flat_no_row_paths(tmp_path, monkeypatch) -> None:
    class _Client:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_price(self, symbol: str):
            return 101.0, 123

    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (None, "bad model", "notice"))
    decision_fn, error, notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=StrategyConfig(),
        effective_dry_run=True,
    )
    assert decision_fn is None
    assert error == "bad model"
    assert notice == "notice"

    class _Model(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (_Model(), None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_args, **_kwargs: [])
    decision_fn, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=StrategyConfig(),
        effective_dry_run=True,
    )
    assert error is None and decision_fn is not None
    decision = decision_fn(_Client(), RuntimeConfig(testnet=True), StrategyConfig(), None)
    assert decision.side == "FLAT"
    assert decision.mark_price == 101.0


def test_build_autonomous_decision_fn_trains_and_applies_external_signals(tmp_path, monkeypatch) -> None:
    class _Client:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

    class _Model(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.90

    row = ModelRow(timestamp=60_000, close=100.0, features=(0.1,), label=1)
    train_calls: list[int] = []
    signal_calls: list[str] = []
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (None, None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "train", lambda rows, **_kwargs: train_calls.append(len(rows)) or _Model())
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: signal_calls.append(str(_kwargs["symbol"])) or object())
    monkeypatch.setattr(cli, "_apply_external_signal_to_score", lambda score, strategy, report: (0.99, strategy, 0.09))
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *_args, **_kwargs: None)

    strategy = StrategyConfig(external_signals_enabled=True)
    decision_fn, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=strategy,
        effective_dry_run=True,
    )
    assert error is None and decision_fn is not None
    decision = decision_fn(_Client(), RuntimeConfig(testnet=True), strategy, None)

    assert decision.side == "LONG"
    assert decision.confidence == pytest.approx(0.99)
    assert train_calls == [1]
    assert signal_calls == ["BTCUSDC"]


def test_build_autonomous_decision_fn_applies_liquidity_guard(tmp_path, monkeypatch) -> None:
    class _Client:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

    class _Model(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

    base_ts = 1_767_621_600_000
    rows = [
        ModelRow(
            timestamp=base_ts + index * 60_000,
            close=100.0,
            features=(0.1,),
            label=1,
            volume=10.0 if index == 8 else 1000.0,
        )
        for index in range(9)
    ]
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (_Model(), None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *_args, **_kwargs: None)

    strategy = StrategyConfig(
        signal_threshold=0.70,
        liquidity_lookback_bars=8,
        low_liquidity_volume_ratio=0.50,
        low_liquidity_size_multiplier=0.25,
        low_liquidity_signal_threshold_add=0.0,
        dynamic_liquidity_session_enabled=False,
    )
    decision_fn, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=strategy,
        effective_dry_run=True,
    )
    assert error is None and decision_fn is not None
    decision = decision_fn(_Client(), RuntimeConfig(testnet=True), strategy, None)

    assert decision.side == "LONG"
    assert decision.size_multiplier == pytest.approx(0.25)
    assert decision.meta_label_action == "downsize"
    assert "low_liquidity_requires_stronger_signal" in decision.meta_label_reason


def test_build_autonomous_decision_fn_loaded_model_short_without_external_signals(tmp_path, monkeypatch) -> None:
    class _Client:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

    class _ShortModel(_SignedFeeModelEvidence):
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.01

    row = ModelRow(timestamp=60_000, close=100.0, features=(0.1,), label=0)
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (_ShortModel(), None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_readiness_model_rows", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: pytest.fail("external signals are disabled"))

    decision_fn, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=StrategyConfig(external_signals_enabled=False),
        effective_dry_run=True,
    )
    assert error is None and decision_fn is not None
    decision = decision_fn(
        _Client(),
        RuntimeConfig(testnet=True, market_type="futures"),
        StrategyConfig(external_signals_enabled=False),
        None,
    )
    assert decision.side == "SHORT"


def test_build_autonomous_decision_fn_returns_flat_without_training_rows(tmp_path, monkeypatch) -> None:
    class _Client:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def get_symbol_price(self, symbol: str):
            return 102.0, 123

    row = ModelRow(timestamp=60_000, close=100.0, features=(0.1,), label=1)
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (None, None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "_build_model_rows", lambda *_args, **_kwargs: [])

    decision_fn, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=StrategyConfig(),
        effective_dry_run=True,
    )
    assert error is None and decision_fn is not None
    decision = decision_fn(_Client(), RuntimeConfig(testnet=True), StrategyConfig(), None)
    assert decision.side == "FLAT"
    assert decision.mark_price == 102.0


def test_command_autonomous_start_success_error_and_client_failure(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_trading import autonomous

    base_args = {
        "action": "start",
        "objective": "default",
        "model": str(tmp_path / "model.json"),
        "live": False,
        "paper": True,
        "poll_seconds": 0.0,
        "iterations": 1,
        "heartbeat_every": 1,
        "starting_cash": 1000.0,
    }
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot"))
    save_strategy(StrategyConfig())

    def fake_decision(*_args, **_kwargs):
        return autonomous.Decision(side="FLAT", confidence=0.0, mark_price=100.0)

    def fake_run_loop(client, runtime, strategy, cfg, *, decision_fn):
        assert cfg.dry_run is True
        return autonomous.LoopResult(
            iterations=1,
            final_state=autonomous.STATE_STOPPED,
            heartbeats_written=1,
            closed_trades=0,
            opened_trades=0,
            skipped_entries=0,
            exit_reason="max-iterations",
        )

    monkeypatch.setattr(cli, "_build_autonomous_decision_fn", lambda **_kwargs: (fake_decision, None, "Model load failed; regenerating: old"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(autonomous, "run_loop", fake_run_loop)

    assert cli.command_autonomous(argparse.Namespace(**base_args)) == 0
    captured = capsys.readouterr()
    assert "Model load failed; regenerating" in captured.err
    assert "autonomous: max-iterations iterations=1" in captured.out

    monkeypatch.setattr(cli, "_build_autonomous_decision_fn", lambda **_kwargs: (None, "model missing", None))
    assert cli.command_autonomous(argparse.Namespace(**base_args)) == 2
    assert "model missing" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_build_autonomous_decision_fn", lambda **_kwargs: (fake_decision, None, None))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: (_ for _ in ()).throw(BinanceAPIError("startup down")))
    assert cli.command_autonomous(argparse.Namespace(**base_args)) == 2
    assert "Autonomous startup blocked: commission-rate verification failed: startup down" in capsys.readouterr().err
