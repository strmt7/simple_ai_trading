from __future__ import annotations

from types import SimpleNamespace

import pytest

import simple_ai_trading.backtest as backtest_module
from simple_ai_trading.backtest import (
    calibrate_threshold_for_backtest,
    risk_adjusted_backtest_score,
    run_backtest,
)
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.execution_simulation import SymbolExecutionProfile
from simple_ai_trading.features import ModelRow
from simple_ai_trading.model import HybridExpert, HybridPrototype, TrainedModel
from simple_ai_trading.types import StrategyConfig


def test_backtest_runs() -> None:
    rows = [
        ModelRow(
            timestamp=i,
            close=100 + i,
            features=(0.1, 0.0, 0.01, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            label=1,
        )
        for i in range(20)
    ]
    model = TrainedModel(
        weights=[0.0] * 13,
        bias=10.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig()
    result = run_backtest(rows, model, cfg, starting_cash=1000.0)
    assert result.trades >= 0
    assert result.starting_cash == 1000.0
    assert result.buy_hold_pnl > 0.0
    assert result.edge_vs_buy_hold == result.realized_pnl - result.buy_hold_pnl
    assert result.scoring_backend_requested == "auto"
    assert result.scoring_backend_kind in {"directml", "cuda", "rocm", "mps", "cpu"}
    if result.scoring_backend_kind == "cpu":
        assert result.scoring_backend_reason
    assert 1 <= len(result.equity_curve) <= len(rows)
    assert result.equity_curve[0]["equity"] == pytest.approx(1000.0)


def test_backtest_empty_rows_preserves_requested_scoring_backend() -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    result = run_backtest([], model, StrategyConfig(), compute_backend="cpu")
    assert result.scoring_backend_requested == "cpu"
    assert result.scoring_backend_kind == "cpu"


def test_backtest_applies_meta_label_skip_and_downsize_policy() -> None:
    rows = [
        ModelRow(timestamp=i * 60_000, close=100.0 + i, features=(1.0,), label=1)
        for i in range(8)
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.01,
        max_position_pct=0.5,
        stop_loss_pct=0.02,
        take_profit_pct=0.50,
        signal_threshold=0.60,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )
    base_model = TrainedModel(
        weights=[0.0],
        bias=5.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )

    baseline = run_backtest(rows, base_model, cfg, starting_cash=1000.0)
    assert baseline.closed_trades >= 1
    baseline_notional = float(baseline.trade_log[0]["gross_notional"])

    downsize_model = TrainedModel(
        **{
            **base_model.__dict__,
            "meta_label_policy": {
                "enabled": True,
                "mode": "take_downsize_skip",
                "take_threshold": 0.90,
                "downsize_threshold": 0.20,
                "downsize_fraction": 0.25,
            },
        }
    )
    downsized = run_backtest(rows, downsize_model, cfg, starting_cash=1000.0)
    assert downsized.closed_trades >= 1
    assert downsized.meta_label_downsizes >= 1
    assert float(downsized.trade_log[0]["gross_notional"]) == pytest.approx(baseline_notional * 0.25)
    assert downsized.trade_log[0]["meta_label_action"] == "downsize"

    skip_model = TrainedModel(
        **{
            **base_model.__dict__,
            "meta_label_policy": {
                "enabled": True,
                "mode": "take_downsize_skip",
                "take_threshold": 1.0,
                "downsize_threshold": 0.90,
                "downsize_fraction": 0.25,
            },
        }
    )
    skipped = run_backtest(rows, skip_model, cfg, starting_cash=1000.0)
    assert skipped.closed_trades == 0
    assert skipped.meta_label_skips >= 1
    assert skipped.trades_per_day_cap_hit == 0
    assert skipped.trade_log == ()


def test_backtest_applies_regime_unpredictability_entry_gate() -> None:
    closes = [100.0, 106.0, 94.0, 107.0, 93.0, 108.0, 92.0, 109.0, 91.0, 110.0, 90.0, 111.0]
    rows = [
        ModelRow(timestamp=i * 60_000, close=close, features=(1.0 if i >= 7 else -1.0,), label=1)
        for i, close in enumerate(closes)
    ]
    model = TrainedModel(
        weights=[10.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )
    base_cfg = StrategyConfig(
        cooldown_minutes=0,
        max_regime_unpredictability=1.0,
        unpredictability_cooldown_minutes=0,
        liquidity_lookback_bars=8,
        signal_threshold=0.60,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )
    baseline = run_backtest(rows, model, base_cfg, starting_cash=1000.0)
    gated = run_backtest(
        rows,
        model,
        StrategyConfig(**{**base_cfg.asdict(), "max_regime_unpredictability": 0.40, "unpredictability_cooldown_minutes": 5}),
        starting_cash=1000.0,
    )

    assert baseline.closed_trades >= 1
    assert gated.closed_trades == 0
    assert gated.regime_entry_skips > 0
    assert gated.trade_log == ()


def test_backtest_downsizes_entries_after_measured_low_liquidity_signal() -> None:
    base_ts = 1_767_621_600_000  # 2026-01-05 14:00:00 UTC.
    rows = [
        ModelRow(
            timestamp=base_ts + i * 60_000,
            close=100.0 + i * 0.4,
            features=(1.0 if i >= 8 else 0.0,),
            label=1,
            volume=10.0 if i == 8 else 1000.0,
        )
        for i in range(14)
    ]
    model = TrainedModel(
        weights=[6.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.70,
    )
    common = dict(
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
    )

    normal = run_backtest(rows, model, StrategyConfig(**common, liquidity_risk_enabled=False), starting_cash=1000.0)
    adjusted = run_backtest(rows, model, StrategyConfig(**common, liquidity_risk_enabled=True), starting_cash=1000.0)

    assert normal.closed_trades == 1
    assert adjusted.closed_trades == 1
    assert float(adjusted.trade_log[0]["gross_notional"]) == pytest.approx(float(normal.trade_log[0]["gross_notional"]) * 0.25)
    assert adjusted.trade_log[0]["meta_label_action"] == "downsize"
    assert "low_liquidity_requires_stronger_signal" in str(adjusted.trade_log[0]["meta_label_reason"])


def test_backtest_does_not_apply_fixed_utc_session_penalty() -> None:
    base_ts = 1_767_578_400_000  # 2026-01-05 02:00:00 UTC.
    rows = [
        ModelRow(
            timestamp=base_ts + i * 60_000,
            close=100.0 + i * 0.4,
            features=(1.0 if i >= 8 else 0.0,),
            label=1,
            volume=1000.0,
        )
        for i in range(14)
    ]
    model = TrainedModel(
        weights=[6.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.70,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.01,
        max_position_pct=0.50,
        stop_loss_pct=0.01,
        take_profit_pct=0.50,
        signal_threshold=0.70,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_lookback_bars=8,
        off_session_signal_threshold_add=0.10,
        off_session_size_multiplier=0.10,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0)

    assert result.closed_trades == 1
    assert result.trade_log[0]["meta_label_action"] == "take"
    assert "outside_preferred" not in str(result.trade_log[0]["meta_label_reason"])


def test_backtest_downsizes_entries_after_data_probed_low_liquidity_session() -> None:
    base_ts = 1_767_621_600_000  # 2026-01-05 14:00:00 UTC.
    rows = [
        ModelRow(
            timestamp=base_ts + i * 60_000,
            close=100.0 + i * 0.4,
            features=(1.0 if i >= 8 else 0.0,),
            label=1,
            volume=10.0 if i == 8 else 1000.0,
        )
        for i in range(14)
    ]
    model = TrainedModel(
        weights=[6.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.70,
    )
    common = dict(
        risk_per_trade=0.01,
        max_position_pct=0.50,
        stop_loss_pct=0.01,
        take_profit_pct=0.50,
        signal_threshold=0.70,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_lookback_bars=8,
        low_liquidity_volume_ratio=0.01,
        low_liquidity_signal_threshold_add=0.0,
        dynamic_liquidity_bucket_minutes=15,
        dynamic_liquidity_session_min_samples=3,
        low_session_liquidity_volume_ratio=0.50,
        low_session_signal_threshold_add=0.0,
        low_session_size_multiplier=0.40,
    )

    normal = run_backtest(rows, model, StrategyConfig(**common, dynamic_liquidity_session_enabled=False), starting_cash=1000.0)
    adjusted = run_backtest(rows, model, StrategyConfig(**common, dynamic_liquidity_session_enabled=True), starting_cash=1000.0)

    assert normal.closed_trades == 1
    assert adjusted.closed_trades == 1
    assert float(adjusted.trade_log[0]["gross_notional"]) == pytest.approx(float(normal.trade_log[0]["gross_notional"]) * 0.40)
    assert adjusted.trade_log[0]["meta_label_action"] == "downsize"
    assert "data_probed_liquidity_session_below_history" in str(adjusted.trade_log[0]["meta_label_reason"])


def test_backtest_gpu_batch_scoring_path(monkeypatch) -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=110.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    backend = BackendInfo("directml", "directml", "privateuseone:0", "DirectML", "")
    monkeypatch.setattr(backtest_module, "resolve_backend", lambda _requested: backend)
    monkeypatch.setattr(backtest_module, "_batch_probabilities_torch", lambda *_a, **_k: [0.99, 0.99])

    result = run_backtest(rows, model, StrategyConfig(risk_per_trade=0.1), compute_backend="directml")

    assert result.closed_trades == 1
    assert result.scoring_backend_kind == "directml"
    assert result.scoring_backend_device == "privateuseone:0"


def test_backtest_gpu_batch_scoring_falls_back_to_cpu(monkeypatch) -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=110.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=4.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    backend = BackendInfo("cuda", "cuda", "cuda:0", "NVIDIA CUDA", "")
    monkeypatch.setattr(backtest_module, "resolve_backend", lambda _requested: backend)

    def fail_gpu_score(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backtest_module, "_batch_probabilities_torch", fail_gpu_score)

    result = run_backtest(rows, model, StrategyConfig(risk_per_trade=0.1), compute_backend="cuda")

    assert result.closed_trades == 1
    assert result.scoring_backend_requested == "cuda"
    assert result.scoring_backend_kind == "cpu"
    assert "fell back to CPU" in result.scoring_backend_reason


def test_backtest_hybrid_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(20):
        sign = 1.0 if index % 2 else -1.0
        rows.append(ModelRow(
            timestamp=index * 60_000,
            close=100.0 + index,
            features=(sign, 0.2 * sign, 0.1, 0.05, 0.01, 0.45, 0.02, 0.01, 0.02, 2.0, 0.03, 0.01, 0.1),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.3] * 13,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    model.hybrid_base_weight = 0.4
    model.hybrid_experts = [
        HybridExpert(
            name="near",
            kind="lorentzian_knn",
            weight=0.3,
            prototypes=[
                HybridPrototype(features=list(rows[1].features), label=1),
                HybridPrototype(features=list(rows[0].features), label=0),
            ],
            k=1,
            feature_count=13,
        ),
        HybridExpert(
            name="kernel",
            kind="rational_quadratic_kernel",
            weight=0.2,
            prototypes=[
                HybridPrototype(features=list(rows[1].features), label=1),
                HybridPrototype(features=list(rows[0].features), label=0),
            ],
            bandwidth=1.0,
            alpha=1.0,
            feature_count=13,
        ),
        HybridExpert(name="tech", kind="technical_confluence", weight=0.1, feature_count=13),
    ]

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=7)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5

    result = run_backtest(
        rows,
        model,
        StrategyConfig(signal_threshold=0.5, risk_per_trade=0.1),
        compute_backend="directml",
        score_batch_size=7,
    )
    assert result.scoring_backend_kind == "directml"
    assert not result.scoring_backend_reason


def test_backtest_tracks_fees_and_cap_hits() -> None:
    rows = [
        ModelRow(
            timestamp=i * 60_000,
            close=1000.0 + (50 if i % 2 == 0 else -20),
            features=(1.0 if i % 2 == 0 else -1.0, *[0.0] * 12),
            label=1 if i % 2 == 0 else 0,
        )
        for i in range(20)
    ]
    # alternating long/short signal, forcing multiple close/open cycles in futures mode
    model = TrainedModel(
        weights=[10.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        leverage=5.0,
        risk_per_trade=0.15,
        max_position_pct=0.5,
        signal_threshold=0.6,
        take_profit_pct=0.02,
        stop_loss_pct=0.02,
        max_trades_per_day=1,
        taker_fee_bps=10.0,
    )
    result = run_backtest(rows, model, cfg, starting_cash=10_000.0, market_type="futures")
    assert result.total_fees >= 0.0
    assert result.trades_per_day_cap_hit >= 0
    assert result.gross_exposure >= 0.0


def test_backtest_unlimited_trades_when_disabled() -> None:
    rows = [
        ModelRow(
            timestamp=i * 60_000,
            close=1000.0 + (10 if i % 2 == 0 else -10),
            features=(1.0 if i % 2 == 0 else -1.0, *[0.0] * 12),
            label=1,
        )
        for i in range(30)
    ]
    model = TrainedModel(
        weights=[10.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        leverage=2.0,
        risk_per_trade=0.05,
        max_position_pct=0.5,
        signal_threshold=0.6,
        take_profit_pct=0.01,
        stop_loss_pct=0.01,
        max_trades_per_day=0,
    )
    result = run_backtest(rows, model, cfg, starting_cash=5000.0, market_type="futures")
    assert result.trades_per_day_cap_hit == 0


def test_backtest_exit_fee_uses_exit_notional() -> None:
    rows = [
        ModelRow(
            timestamp=0,
            close=100.0,
            features=(10.0, *[0.0] * 12),
            label=1,
        ),
        ModelRow(
            timestamp=60_000,
            close=100.0,
            features=(0.0, *[0.0] * 12),
            label=0,
        ),
        ModelRow(
            timestamp=120_000,
            close=120.0,
            features=(0.0, *[0.0] * 12),
            label=0,
        ),
    ]
    model = TrainedModel(
        weights=[1.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        taker_fee_bps=100.0,
        slippage_bps=0.0,
        signal_threshold=0.55,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 1
    assert result.total_fees == pytest.approx(4.389046741493509)


def test_backtest_uses_intrabar_stop_when_stop_and_take_both_touch() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1, high=100.0, low=100.0),
        ModelRow(timestamp=60_000, close=100.0, features=(1.0,), label=1, high=100.0, low=100.0),
        ModelRow(timestamp=120_000, close=100.0, features=(1.0,), label=1, high=103.0, low=98.0),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=10.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        risk_per_trade=0.01,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        signal_threshold=0.55,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")

    assert result.closed_trades == 1
    assert result.trade_log[0]["exit_reason"] == "intrabar_stop_loss_ambiguous"
    assert result.trade_log[0]["exit_mark_price"] == pytest.approx(
        float(result.trade_log[0]["entry_price"]) * (1.0 - cfg.stop_loss_pct)
    )
    assert result.realized_pnl < 0.0


def test_backtest_futures_liquidates_on_intrabar_adverse_mark() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(-1.0,), label=0, high=100.0, low=100.0),
        ModelRow(timestamp=60_000, close=100.0, features=(-1.0,), label=0, high=100.0, low=100.0),
        ModelRow(timestamp=120_000, close=100.0, features=(-1.0,), label=0, high=200.0, low=100.0),
    ]
    model = TrainedModel(
        weights=[10.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        leverage=10.0,
        risk_per_trade=0.02,
        max_position_pct=1.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.50,
        liquidation_buffer_pct=0.01,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="futures")

    assert result.stopped_by_liquidation is True
    assert result.liquidation_events == 1
    assert result.trade_log[0]["exit_mark_price"] == pytest.approx(200.0)
    assert result.trade_log[0]["exit_reason"] == "liquidation"


def test_backtest_signal_enters_on_next_bar_close() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(10.0, *[0.0] * 12), label=1),
        ModelRow(timestamp=60_000, close=120.0, features=(0.0, *[0.0] * 12), label=0),
    ]
    model = TrainedModel(
        weights=[1.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        signal_threshold=0.55,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")

    assert result.closed_trades == 1
    assert result.realized_pnl == pytest.approx(-2.281928855519027)
    assert len(result.trade_log) == 1
    assert len(result.trade_returns) == 1
    assert result.trade_pnls[0] == pytest.approx(result.realized_pnl)
    assert result.trade_log[0]["opened_at"] == 60_000
    assert result.trade_log[0]["closed_at"] == 60_000
    assert result.equity_curve[-1]["equity"] == pytest.approx(result.ending_cash)
    assert result.gross_loss == pytest.approx(abs(result.realized_pnl))
    assert result.profit_factor == pytest.approx(0.0)
    assert result.expectancy == pytest.approx(result.realized_pnl)
    assert result.average_trade_return == pytest.approx(result.trade_returns[0])
    assert result.max_consecutive_losses == 1


def test_backtest_sizes_positions_from_stop_loss_risk_budget() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=120_000, close=105.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=10.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    spot_cfg = StrategyConfig(
        risk_per_trade=0.01,
        max_position_pct=0.5,
        stop_loss_pct=0.02,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
        signal_threshold=0.55,
        take_profit_pct=0.50,
    )
    futures_cfg = StrategyConfig(**{**spot_cfg.asdict(), "leverage": 5.0, "max_position_pct": 0.2})

    spot = run_backtest(rows, model, spot_cfg, starting_cash=1000.0, market_type="spot")
    futures = run_backtest(rows, model, futures_cfg, starting_cash=1000.0, market_type="futures")

    assert spot.max_exposure == pytest.approx(500.0)
    assert futures.max_exposure == pytest.approx(500.0)


def test_backtest_path_quality_metrics_cover_profit_factor_and_loss_streak() -> None:
    class _StepModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.99, 0.0, 0.99, 0.0, 0.99, 0.0]
            value = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return value

    rows = [
        ModelRow(timestamp=i * 60_000, close=price, features=(0.0,), label=1)
        for i, price in enumerate([100.0, 100.0, 110.0, 110.0, 100.0, 100.0, 95.0])
    ]
    result = run_backtest(
        rows,
        _StepModel(),
        StrategyConfig(
            risk_per_trade=0.1,
            max_position_pct=0.2,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            signal_threshold=0.55,
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
            cooldown_minutes=0,
            max_trades_per_day=10,
        ),
        starting_cash=1000.0,
        market_type="spot",
    )

    assert result.closed_trades == 3
    assert len(result.trade_pnls) == 3
    assert result.gross_profit > 0.0
    assert result.gross_loss > 0.0
    assert result.profit_factor == pytest.approx(result.gross_profit / result.gross_loss)
    assert result.expectancy == pytest.approx(sum(result.trade_pnls) / 3)
    assert result.average_trade_return == pytest.approx(sum(result.trade_returns) / 3)
    assert result.trade_return_stdev > 0.0
    assert result.max_consecutive_losses == 2


def test_backtest_cooldown_prevents_immediate_reentry() -> None:
    class _StepModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.99, 0.0, 0.99, 0.0, 0.0]
            score = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return score

    rows = [
        ModelRow(timestamp=i * 60_000, close=100.0, features=(0.0,), label=0)
        for i in range(5)
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        signal_threshold=0.55,
        cooldown_minutes=5,
        max_trades_per_day=10,
    )
    no_cooldown = StrategyConfig(**{**cfg.asdict(), "cooldown_minutes": 0})

    assert run_backtest(rows, _StepModel(), no_cooldown, starting_cash=1000.0).closed_trades == 2
    assert run_backtest(rows, _StepModel(), cfg, starting_cash=1000.0).closed_trades == 1


def test_backtest_uses_model_threshold_and_confidence_shrinkage() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=105.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.5,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )
    active = StrategyConfig(signal_threshold=0.90, confidence_beta=1.0, risk_per_trade=0.1)
    conservative = StrategyConfig(signal_threshold=0.90, confidence_beta=0.1, risk_per_trade=0.1)

    assert run_backtest(rows, model, active, starting_cash=1000.0).closed_trades == 1
    assert run_backtest(rows, model, conservative, starting_cash=1000.0).closed_trades == 0


def test_backtest_applies_symbol_execution_profile_costs() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=130.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=10.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        risk_per_trade=0.2,
        max_position_pct=0.2,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
    )
    clean = run_backtest(rows, model, cfg, starting_cash=1000.0)
    stressed = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=1000.0,
        symbol_profile=SymbolExecutionProfile(
            symbol="TESTUSDC",
            spread_bps=500.0,
            quote_volume=10_000.0,
            trade_count=100,
            liquidity_score=0.05,
            latency_ms=3000,
            liquidity_haircut=0.9,
        ),
    )

    assert clean.closed_trades == stressed.closed_trades == 1
    assert stressed.realized_pnl < clean.realized_pnl
    assert stressed.buy_hold_pnl < clean.buy_hold_pnl


def test_backtest_uses_row_volume_for_participation_impact() -> None:
    low_volume_rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1, volume=0.0),
        ModelRow(timestamp=60_000, close=100.0, features=(1.0,), label=1, volume=0.0),
        ModelRow(timestamp=120_000, close=110.0, features=(1.0,), label=1, volume=0.0),
    ]
    high_volume_rows = [
        ModelRow(timestamp=row.timestamp, close=row.close, features=row.features, label=row.label, volume=1_000_000.0)
        for row in low_volume_rows
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=10.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        risk_per_trade=0.5,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
        signal_threshold=0.55,
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
    )

    low_volume = run_backtest(low_volume_rows, model, cfg, starting_cash=1000.0)
    high_volume = run_backtest(high_volume_rows, model, cfg, starting_cash=1000.0)

    assert low_volume.closed_trades == high_volume.closed_trades == 1
    assert high_volume.realized_pnl > low_volume.realized_pnl


def test_backtest_buy_hold_baseline_handles_invalid_inputs() -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig()
    zero_cash = run_backtest(
        [ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0)],
        model,
        cfg,
        starting_cash=0.0,
    )
    zero_price = run_backtest(
        [
            ModelRow(timestamp=0, close=0.0, features=(0.0,), label=0),
            ModelRow(timestamp=1, close=100.0, features=(0.0,), label=0),
        ],
        model,
        cfg,
        starting_cash=1000.0,
    )
    impossible_exit = run_backtest(
        [
            ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0),
            ModelRow(timestamp=1, close=100.0, features=(0.0,), label=0),
        ],
        model,
        StrategyConfig(slippage_bps=20_000.0),
        starting_cash=1000.0,
    )

    assert zero_cash.buy_hold_pnl == 0.0
    assert zero_price.buy_hold_pnl == 0.0
    assert impossible_exit.buy_hold_pnl == 0.0


def test_profit_threshold_calibration_accepts_profitable_threshold() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(0.02,), label=1),
        ModelRow(timestamp=60_000, close=90.0, features=(-0.02,), label=0),
        ModelRow(timestamp=120_000, close=90.0, features=(0.14,), label=1),
        ModelRow(timestamp=180_000, close=105.0, features=(0.14,), label=1),
    ]
    model = TrainedModel(
        weights=[10.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        risk_per_trade=0.5,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        signal_threshold=0.5,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )

    baseline = run_backtest(
        rows,
        TrainedModel(**{**model.__dict__, "decision_threshold": 0.5}),
        cfg,
        starting_cash=1000.0,
    )
    report = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=0.45,
        end=0.85,
        steps=9,
        starting_cash=1000.0,
    )

    assert report.accepted is True
    assert report.threshold != 0.5
    assert report.realized_pnl > baseline.realized_pnl
    assert report.baseline_realized_pnl == baseline.realized_pnl
    assert report.asdict()["evaluated_thresholds"] == 9


def test_profit_threshold_calibration_rejects_non_improvement_and_score_edges() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=105.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=2.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(risk_per_trade=0.1, max_position_pct=0.1, signal_threshold=0.5)

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=float("nan"),
        start=0.9,
        end=0.1,
        steps=1,
        min_score_delta=float("nan"),
    )
    assert report.accepted is False
    assert report.threshold == 0.5
    assert report.evaluated_thresholds == 1
    nan_grid = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=float("nan"),
        end=0.1,
        steps=2,
    )
    assert nan_grid.evaluated_thresholds == 2
    reversed_grid = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=0.9,
        end=0.1,
        steps=2,
    )
    assert reversed_grid.evaluated_thresholds == 3

    stopped = SimpleNamespace(
        realized_pnl=10.0,
        total_fees=1.0,
        max_drawdown=0.5,
        stopped_by_drawdown=True,
        closed_trades=0,
    )
    stable = SimpleNamespace(
        realized_pnl=10.0,
        total_fees=1.0,
        max_drawdown=0.0,
        stopped_by_drawdown=False,
        closed_trades=1,
    )
    assert risk_adjusted_backtest_score(stopped, starting_cash=1000.0) < risk_adjusted_backtest_score(
        stable,
        starting_cash=1000.0,
    )
    assert risk_adjusted_backtest_score(SimpleNamespace(realized_pnl="bad"), starting_cash=float("nan")) < 0.0
    assert risk_adjusted_backtest_score(SimpleNamespace(realized_pnl=object()), starting_cash=1000.0) < 0.0


def test_threshold_calibration_reuses_one_probability_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=1)
        for index in range(12)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=3.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    calls = {"count": 0}
    regime_calls = {"count": 0}
    liquidity_calls = {"count": 0}

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        calls["count"] += 1
        return [0.95 for _ in scored_rows], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    def fake_regime_scores(scored_rows, _cfg):
        regime_calls["count"] += 1
        return [0.0 for _ in scored_rows]

    def fake_liquidity_adjustments(scored_rows, _cfg):
        liquidity_calls["count"] += 1
        return [(0.0, 1.0, False, False) for _ in scored_rows]

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)
    monkeypatch.setattr(backtest_module, "precompute_backtest_regime_scores", fake_regime_scores)
    monkeypatch.setattr(backtest_module, "precompute_backtest_liquidity_adjustments", fake_liquidity_adjustments)

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(risk_per_trade=0.1, signal_threshold=0.5),
        baseline_threshold=0.5,
        start=0.5,
        end=0.9,
        steps=7,
        compute_backend="directml",
    )

    assert calls["count"] == 1
    assert regime_calls["count"] == 1
    assert liquidity_calls["count"] == 1
    assert report.evaluated_thresholds == 7


def test_run_backtest_rejects_bad_precomputed_probability_length() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=101.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    with pytest.raises(ValueError, match="precomputed_probabilities length mismatch"):
        run_backtest(rows, model, StrategyConfig(), precomputed_probabilities=[0.5])


def test_run_backtest_rejects_bad_precomputed_regime_score_length() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=101.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    with pytest.raises(ValueError, match="precomputed_regime_scores length mismatch"):
        run_backtest(rows, model, StrategyConfig(), precomputed_regime_scores=[0.0])


def test_run_backtest_rejects_bad_precomputed_liquidity_adjustment_length() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=101.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    with pytest.raises(ValueError, match="precomputed_liquidity_adjustments length mismatch"):
        run_backtest(rows, model, StrategyConfig(), precomputed_liquidity_adjustments=[(0.0, 1.0, False, False)])


def test_precomputed_regime_scores_match_slice_classifier() -> None:
    closes = [
        100.0,
        101.0,
        99.0,
        102.0,
        98.0,
        103.0,
        97.0,
        104.0,
        103.5,
        104.5,
        105.2,
        104.8,
    ]
    rows = [
        ModelRow(timestamp=index * 60_000, close=close, features=(1.0,), label=1)
        for index, close in enumerate(closes)
    ]
    cfg = StrategyConfig(liquidity_lookback_bars=8)

    fast = backtest_module.precompute_backtest_regime_scores(rows, cfg)
    slow = backtest_module._precompute_backtest_regime_scores_slow(rows, cfg)

    assert fast == pytest.approx(slow)
