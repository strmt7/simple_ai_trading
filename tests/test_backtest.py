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
from simple_ai_trading.execution_simulation import SymbolExecutionProfile, execution_assumptions_for_symbol
from simple_ai_trading.features import ModelRow
from simple_ai_trading.model import HybridExpert, HybridPrototype, TrainedModel
from simple_ai_trading.risk_controls import stop_loss_sized_notional_pct
from simple_ai_trading.types import StrategyConfig


def test_high_liquidity_symbol_profile_reduces_generic_spread_floor() -> None:
    strategy = StrategyConfig(slippage_bps=5.0, max_spread_bps=5.0, testnet_liquidity_haircut=0.5)
    generic = execution_assumptions_for_symbol(strategy)
    profile = SymbolExecutionProfile(
        "BTCUSDT",
        spread_bps=0.02,
        quote_volume=10_000_000_000.0,
        trade_count=4_000_000,
        liquidity_score=1.0,
        latency_ms=strategy.latency_buffer_ms,
        liquidity_haircut=strategy.testnet_liquidity_haircut,
    )
    specific = execution_assumptions_for_symbol(strategy, profile)

    assert specific.spread_bps < generic.spread_bps
    assert specific.testnet_to_live_buffer_bps < generic.testnet_to_live_buffer_bps
    assert specific.spread_bps >= 0.50
    assert specific.testnet_to_live_buffer_bps >= 2.0


def test_neutral_probability_never_becomes_a_trade_direction() -> None:
    direction = backtest_module._normalize_market_direction

    assert direction(0.5, 0.5, "futures", short_threshold=0.5) == 0
    assert direction(float("nan"), 0.5, "futures", short_threshold=0.5) == 0
    assert direction(0.500001, 0.5, "futures", short_threshold=0.5) == 1
    assert direction(0.499999, 0.5, "futures", short_threshold=0.5) == -1
    assert direction(0.5, 0.5, "spot") == 0


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
        precomputed_regime_scores=[0.0] * 7 + [0.95] * (len(rows) - 7),
        precomputed_liquidity_adjustments=[(0.0, 1.0, False, False)] * len(rows),
    )

    assert baseline.closed_trades >= 1
    assert gated.closed_trades == 0
    assert gated.regime_entry_skips > 0
    assert gated.trade_log == ()


def test_backtest_downsizes_borderline_regime_instead_of_hard_blocking() -> None:
    rows = [
        ModelRow(timestamp=i * 60_000, close=100.0 + i * 0.25, features=(1.0,), label=1)
        for i in range(18)
    ]
    model = TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.01,
        max_position_pct=0.50,
        stop_loss_pct=0.01,
        take_profit_pct=0.005,
        cooldown_minutes=0,
        max_regime_unpredictability=0.60,
        unpredictability_cooldown_minutes=90,
        liquidity_lookback_bars=8,
        signal_threshold=0.60,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_risk_enabled=False,
        dynamic_liquidity_session_enabled=False,
        max_trades_per_day=10,
    )
    probabilities = [0.50] * 8 + [0.95] * (len(rows) - 8)
    neutral_regime = [0.0] * len(rows)
    borderline_regime = [0.0] * 8 + [0.65] * (len(rows) - 8)
    liquidity = [(0.0, 1.0, False, False)] * len(rows)

    normal = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=1000.0,
        precomputed_probabilities=probabilities,
        precomputed_regime_scores=neutral_regime,
        precomputed_liquidity_adjustments=liquidity,
    )
    borderline = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=1000.0,
        precomputed_probabilities=probabilities,
        precomputed_regime_scores=borderline_regime,
        precomputed_liquidity_adjustments=liquidity,
    )

    assert normal.closed_trades >= 1
    assert borderline.closed_trades >= 1
    assert borderline.regime_entry_downsizes >= 1
    assert borderline.regime_entry_skips == 0
    assert borderline.trade_log[0]["gross_notional"] < normal.trade_log[0]["gross_notional"]


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


def test_rule_alpha_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(18):
        sign = 1.0 if index % 2 else -1.0
        base = [
            0.0014 * sign,
            0.0012 * sign,
            0.0007 * sign,
            0.0003 * sign,
            0.0,
            0.55 if sign > 0 else 0.45,
            0.0005 * sign,
            0.0002,
            0.0002,
            1.3,
            0.0008 * sign,
            0.0001 * sign,
            0.2,
        ]
        flow = [
            0.71 if sign > 0 else 0.29,
            0.58 * sign,
            0.55 * sign,
            0.36,
            0.34,
            0.22,
            0.0,
            0.62,
            0.24 * sign,
            0.44,
            0.36,
            0.28 * sign,
            0.02 * sign,
        ]
        rows.append(ModelRow(
            timestamp=index * 1000,
            close=100.0 + index * sign,
            features=tuple(base + flow),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.0] * 26,
        bias=0.0,
        feature_dim=26,
        epochs=0,
        feature_means=[0.0] * 26,
        feature_stds=[1.0] * 26,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="rule-alpha",
                kind="rule_alpha",
                weight=1.0,
                feature_count=26,
                params={
                    "family": "volume_synchronized_flow",
                    "sensitivity": 8.0,
                    "deadband": 0.01,
                    "order_flow_start": 13,
                    "order_flow_width": 13,
                    "order_flow_window_count": 1,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=5)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_trade_tape_rule_alpha_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(18):
        sign = 1.0 if index % 2 else -1.0
        base = [
            0.0014 * sign,
            0.0012 * sign,
            0.0007 * sign,
            0.0003 * sign,
            0.0,
            0.55 if sign > 0 else 0.45,
            0.0005 * sign,
            0.0002,
            0.0002,
            1.3,
            0.0008 * sign,
            0.0001 * sign,
            0.2,
        ]
        tape: list[float] = []
        for _ in range(3):
            tape.extend([
                0.76 if sign > 0 else 0.24,
                0.62 * sign,
                0.55 * sign,
                0.36 * sign,
                0.34 * sign,
                0.22,
                0.08 * sign,
                0.02,
                0.10 * sign,
                0.0,
                0.24 * sign,
                0.30 * sign,
            ])
        rows.append(ModelRow(
            timestamp=index * 1000,
            close=100.0 + index * sign,
            features=tuple(base + tape),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.0] * 49,
        bias=0.0,
        feature_dim=49,
        epochs=0,
        feature_means=[0.0] * 49,
        feature_stds=[1.0] * 49,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="tape-rule-alpha",
                kind="rule_alpha",
                weight=1.0,
                feature_count=49,
                params={
                    "family": "micro_flow_scalp",
                    "sensitivity": 8.0,
                    "deadband": 0.01,
                    "trade_tape_start": 13,
                    "trade_tape_width": 12,
                    "trade_tape_window_count": 3,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=5)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_mlp_hybrid_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [
        ModelRow(timestamp=index * 1000, close=100.0 + index, features=(float(index % 3), float(2 - index % 3)), label=1)
        for index in range(9)
    ]
    model = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=0,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="dense",
                kind="dense_mlp",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "output_activation": "sigmoid",
                    "layers": [
                        {"weights": [[1.0], [-1.0]], "bias": [0.0], "activation": "sigmoid"},
                    ],
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=4)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_signed_payoff_mlp_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [
        ModelRow(timestamp=index * 1000, close=100.0 + index, features=(float(index % 4), float(3 - index % 4)), label=1)
        for index in range(12)
    ]
    model = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=0,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="payoff-mlp",
                kind="signed_payoff_mlp_ranker",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "layers": [
                        {"weights": [[1.0], [-1.0]], "bias": [0.0], "activation": "tanh"},
                    ],
                    "clip_bps": 20.0,
                    "deadband_bps": 0.0,
                    "sensitivity": 6.0,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=4)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_signed_payoff_lightgbm_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [
        ModelRow(
            timestamp=index * 1000,
            close=100.0 + index,
            features=(value, 0.25 - value),
            label=0,
        )
        for index, value in enumerate((-1.0, -0.2, 0.0, 0.1, 0.8, 1.4))
    ]
    model = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=0,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="payoff-tree",
                kind="signed_payoff_lightgbm_ranker",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "tree_info": [
                        {
                            "tree_structure": {
                                "split_feature": 0,
                                "threshold": 0.0,
                                "decision_type": "<=",
                                "default_left": True,
                                "left_child": {"leaf_value": -0.2},
                                "right_child": {"leaf_value": 0.3},
                            }
                        },
                        {"tree_structure": {"leaf_value": 0.1}},
                    ],
                    "clip_bps": 1.0,
                    "deadband_bps": 0.0,
                    "sensitivity": 3.0,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=3)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_action_payoff_lightgbm_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [
        ModelRow(
            timestamp=index * 1000,
            close=100.0 + index,
            features=(value, 0.25 - value),
            label=0,
        )
        for index, value in enumerate((-1.0, -0.2, 0.0, 0.1, 0.8, 1.4))
    ]
    model = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=0,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="action-payoff-tree",
                kind="signed_payoff_lightgbm_ranker",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "payoff_tree_schema": "action_value_v1",
                    "long_tree_info": [
                        {
                            "tree_structure": {
                                "split_feature": 0,
                                "threshold": 0.0,
                                "decision_type": "<=",
                                "default_left": True,
                                "left_child": {"leaf_value": -0.2},
                                "right_child": {"leaf_value": 0.3},
                            }
                        }
                    ],
                    "short_tree_info": [
                        {
                            "tree_structure": {
                                "split_feature": 0,
                                "threshold": 0.0,
                                "decision_type": "<=",
                                "default_left": True,
                                "left_child": {"leaf_value": 0.25},
                                "right_child": {"leaf_value": -0.15},
                            }
                        }
                    ],
                    "clip_bps": 20.0,
                    "deadband_bps": 0.5,
                    "sensitivity": 4.0,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=3)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5
    assert cpu[0] < 0.5
    assert cpu[-1] > 0.5


def test_action_hurdle_lightgbm_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [
        ModelRow(
            timestamp=index * 1000,
            close=100.0 + index,
            features=(value, 0.25 - value),
            label=0,
        )
        for index, value in enumerate((-1.0, -0.2, 0.0, 0.1, 0.8, 1.4))
    ]
    long_tree = {
        "tree_structure": {
            "split_feature": 0,
            "threshold": 0.0,
            "decision_type": "<=",
            "default_left": True,
            "left_child": {"leaf_value": -2.0},
            "right_child": {"leaf_value": 2.0},
        }
    }
    short_tree = {
        "tree_structure": {
            "split_feature": 0,
            "threshold": 0.0,
            "decision_type": "<=",
            "default_left": True,
            "left_child": {"leaf_value": 2.0},
            "right_child": {"leaf_value": -2.0},
        }
    }
    model = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=0,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="action-hurdle-tree",
                kind="signed_payoff_lightgbm_ranker",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "payoff_tree_schema": "action_value_hurdle_v1",
                    "long_classifier_tree_info": [long_tree],
                    "short_classifier_tree_info": [short_tree],
                    "long_enabled": True,
                    "short_enabled": True,
                    "long_calibration_slope": 1.0,
                    "long_calibration_intercept": 0.0,
                    "short_calibration_slope": 1.0,
                    "short_calibration_intercept": 0.0,
                    "long_positive_mean": 0.6,
                    "long_nonpositive_mean": -0.4,
                    "short_positive_mean": 0.6,
                    "short_nonpositive_mean": -0.4,
                    "clip_bps": 20.0,
                    "deadband_bps": 0.5,
                    "sensitivity": 4.0,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=3)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5
    assert cpu[0] < 0.5
    assert cpu[-1] > 0.5


def test_accelerated_scoring_rejects_unknown_positive_weight_expert() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows = [ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0)]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=0,
        feature_means=[0.0],
        feature_stds=[1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="unsupported",
                kind="future_unknown_expert",
                weight=1.0,
                feature_count=1,
            )
        ],
    )

    with pytest.raises(ValueError, match="does not support"):
        backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=1)


def test_higher_timeframe_rule_alpha_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(18):
        sign = 1.0 if index % 2 else -1.0
        base = [
            0.0015 * sign,
            0.0012 * sign,
            0.0009 * sign,
            0.0004 * sign,
            0.0,
            0.55 if sign > 0 else 0.45,
            0.0007 * sign,
            0.0002,
            0.0002,
            1.2,
            0.0007 * sign,
            0.0001 * sign,
            0.2,
        ]
        htf = [
            0.004 * sign,
            0.002 * sign,
            0.0002,
            0.006,
            -0.002 if sign > 0 else -0.005,
            0.005 if sign > 0 else 0.002,
            0.30 * sign,
            0.20 * sign,
        ]
        flow = [
            0.70 if sign > 0 else 0.30,
            0.42 * sign,
            0.35 * sign,
            0.12,
            0.10,
            0.05,
            0.0,
            0.25 * sign,
            0.18 * sign,
            0.30,
            0.12 * sign,
            0.22 * sign,
            0.05 * sign,
        ]
        rows.append(ModelRow(
            timestamp=index * 1000,
            close=100.0 + index * sign,
            features=tuple(base + htf + flow),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.0] * 34,
        bias=0.0,
        feature_dim=34,
        epochs=0,
        feature_means=[0.0] * 34,
        feature_stds=[1.0] * 34,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="htf-rule-alpha",
                kind="rule_alpha",
                weight=1.0,
                feature_count=34,
                params={
                    "family": "higher_timeframe_alignment",
                    "sensitivity": 8.0,
                    "deadband": 0.01,
                    "higher_timeframe_start": 13,
                    "higher_timeframe_width": 8,
                    "higher_timeframe_window_count": 1,
                    "order_flow_start": 21,
                    "order_flow_width": 13,
                    "order_flow_window_count": 1,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=5)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_directional_regime_rule_alpha_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(18):
        sign = 1.0 if index % 2 else -1.0
        base = [
            0.0015 * sign,
            0.0012 * sign,
            0.0009 * sign,
            0.0005 * sign,
            0.0,
            0.55 if sign > 0 else 0.45,
            0.0007 * sign,
            0.0002,
            0.0002,
            1.2,
            0.0007 * sign,
            0.0001 * sign,
            0.2,
        ]
        htf = [
            0.006 * sign,
            0.003 * sign,
            0.0002,
            0.006,
            -0.002 if sign > 0 else -0.005,
            0.005 if sign > 0 else 0.002,
            0.36 * sign,
            0.24 * sign,
        ]
        flow = [
            0.72 if sign > 0 else 0.28,
            0.54 * sign,
            0.45 * sign,
            0.12,
            0.10,
            0.05,
            0.0,
            0.25 * sign,
            0.24 * sign,
            0.30,
            0.28 * sign,
            0.30 * sign,
            0.05 * sign,
        ]
        rows.append(ModelRow(
            timestamp=index * 1000,
            close=100.0 + index * sign,
            features=tuple(base + htf + flow),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.0] * 34,
        bias=0.0,
        feature_dim=34,
        epochs=0,
        feature_means=[0.0] * 34,
        feature_stds=[1.0] * 34,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="directional-rule-alpha",
                kind="rule_alpha",
                weight=1.0,
                feature_count=34,
                params={
                    "family": "directional_regime_rider",
                    "sensitivity": 8.0,
                    "deadband": 0.01,
                    "higher_timeframe_start": 13,
                    "higher_timeframe_width": 8,
                    "higher_timeframe_window_count": 1,
                    "order_flow_start": 21,
                    "order_flow_width": 13,
                    "order_flow_window_count": 1,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=5)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


def test_empirical_rule_alpha_gpu_batch_scoring_matches_cpu_when_available() -> None:
    backend = backtest_module.resolve_backend("directml")
    if backend.kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    rows: list[ModelRow] = []
    for index in range(18):
        sign = 1.0 if index % 2 else -1.0
        features = [0.0] * 26
        features[15] = sign
        features[16] = sign
        rows.append(ModelRow(
            timestamp=index * 1000,
            close=100.0 + index,
            features=tuple(features),
            label=1 if sign > 0 else 0,
        ))
    model = TrainedModel(
        weights=[0.0] * 26,
        bias=0.0,
        feature_dim=26,
        epochs=0,
        feature_means=[0.0] * 26,
        feature_stds=[1.0] * 26,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="empirical-rule-alpha",
                kind="rule_alpha",
                weight=1.0,
                feature_count=26,
                params={
                    "family": "empirical_feature_edge",
                    "sensitivity": 8.0,
                    "deadband": 0.0,
                    "feature_index": 15,
                    "feature_threshold": 0.0,
                    "feature_scale": 1.0,
                    "tail_direction": 1.0,
                    "second_feature_index": 16,
                    "second_feature_threshold": 0.0,
                    "second_feature_scale": 1.0,
                    "second_tail_direction": 1.0,
                    "trade_side": 1.0,
                    "edge_confidence": 0.75,
                    "edge_slope": 1.0,
                },
            )
        ],
    )

    gpu = backtest_module._batch_probabilities_torch(rows, model, backend=backend, batch_size=5)
    cpu = [model.predict_proba(row.features) for row in rows]

    assert max(abs(left - right) for left, right in zip(gpu, cpu, strict=True)) < 1e-5


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
    trade = result.trade_log[0]
    assert result.total_fees == pytest.approx(float(trade["entry_fee"]) + float(trade["exit_fee"]))
    assert float(trade["exit_fee"]) > float(trade["entry_fee"])


def test_backtest_win_rate_uses_net_pnl_after_fees() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(10.0, *[0.0] * 12), label=1, volume=1_000_000.0),
        ModelRow(timestamp=60_000, close=100.0, features=(0.0, *[0.0] * 12), label=0, volume=1_000_000.0),
        ModelRow(timestamp=120_000, close=101.0, features=(0.0, *[0.0] * 12), label=0, volume=1_000_000.0),
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
        max_spread_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
        signal_threshold=0.55,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")

    assert result.closed_trades == 1
    assert result.trade_log[0]["realized_pnl"] > 0.0
    assert result.trade_log[0]["net_pnl"] < 0.0
    assert result.win_rate == 0.0


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
        max_asset_allocation_pct=0.5,
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
    assert result.trade_log[0]["exit_reason"] == "final_mark"
    assert result.equity_curve[-1]["equity"] == pytest.approx(result.ending_cash)
    assert result.gross_loss == pytest.approx(abs(result.realized_pnl))
    assert result.profit_factor == pytest.approx(0.0)
    assert result.expectancy == pytest.approx(result.realized_pnl)
    assert result.average_trade_return == pytest.approx(result.trade_returns[0])
    assert result.max_consecutive_losses == 1


def test_backtest_drawdown_stop_closes_at_adverse_intrabar_mark() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(10.0,), label=1, high=100.0, low=100.0),
        ModelRow(timestamp=60_000, close=100.0, features=(10.0,), label=1, high=100.0, low=100.0),
        ModelRow(timestamp=120_000, close=100.0, features=(10.0,), label=1, high=100.0, low=80.0),
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
        risk_per_trade=0.50,
        max_position_pct=1.0,
        max_asset_allocation_pct=1.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        signal_threshold=0.55,
        stop_loss_pct=0.90,
        take_profit_pct=0.90,
        max_drawdown_limit=0.05,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")

    assert result.stopped_by_drawdown is True
    assert result.closed_trades == 1
    trade = result.trade_log[0]
    assert trade["exit_reason"] == "drawdown_limit"
    assert trade["exit_mark_price"] == pytest.approx(80.0)
    assert result.ending_cash == pytest.approx(result.starting_cash + float(trade["net_pnl"]))
    assert result.ending_cash < 900.0
    assert result.equity_curve[-1]["equity"] == pytest.approx(result.ending_cash)


def test_backtest_entry_bar_does_not_use_pre_entry_intrabar_extreme() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(10.0,), label=1, high=100.0, low=100.0),
        ModelRow(timestamp=60_000, close=100.0, features=(10.0,), label=1, high=100.0, low=40.0),
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
        risk_per_trade=0.50,
        max_position_pct=1.0,
        max_asset_allocation_pct=1.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        signal_threshold=0.55,
        stop_loss_pct=0.90,
        take_profit_pct=0.90,
        max_drawdown_limit=0.05,
    )

    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")

    assert result.stopped_by_drawdown is False
    assert result.closed_trades == 1
    assert result.ending_cash == pytest.approx(result.starting_cash + float(result.trade_log[0]["net_pnl"]))
    assert result.ending_cash > 990.0
    assert result.trade_log[0]["opened_at"] == 60_000
    assert result.trade_log[0]["closed_at"] == 60_000


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

    assert spot.max_exposure == pytest.approx(1000.0 * stop_loss_sized_notional_pct(spot_cfg, "spot"))
    assert futures.max_exposure == pytest.approx(1000.0 * stop_loss_sized_notional_pct(futures_cfg, "futures"))
    assert spot.max_exposure < 500.0


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


def test_backtest_uses_row_microstructure_range_and_trade_sparsity_for_fills() -> None:
    calm_rows = [
        ModelRow(
            timestamp=0,
            close=100.0,
            features=(1.0,),
            label=1,
            volume=10_000.0,
            high=100.02,
            low=99.98,
            quote_volume=1_000_000.0,
            trade_count=500,
        ),
        ModelRow(
            timestamp=60_000,
            close=100.0,
            features=(0.0,),
            label=0,
            volume=10_000.0,
            high=100.02,
            low=99.98,
            quote_volume=1_000_000.0,
            trade_count=500,
        ),
        ModelRow(
            timestamp=120_000,
            close=110.0,
            features=(0.0,),
            label=0,
            volume=10_000.0,
            high=110.02,
            low=109.98,
            quote_volume=1_000_000.0,
            trade_count=500,
        ),
    ]
    stressed_rows = [
        ModelRow(
            timestamp=row.timestamp,
            close=row.close,
            features=row.features,
            label=row.label,
            volume=row.volume,
            high=row.close * 1.04,
            low=row.close * 0.96,
            quote_volume=5_000.0,
            trade_count=1,
        )
        for row in calm_rows
    ]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.55,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.10,
        max_position_pct=0.20,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        max_spread_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
        stop_loss_pct=0.50,
        take_profit_pct=0.50,
    )

    calm = run_backtest(calm_rows, model, cfg, starting_cash=1000.0)
    stressed = run_backtest(stressed_rows, model, cfg, starting_cash=1000.0)

    assert calm.closed_trades == stressed.closed_trades == 1
    assert stressed.realized_pnl < calm.realized_pnl
    assert stressed.trade_log[0]["entry_execution_cost_bps"] > calm.trade_log[0]["entry_execution_cost_bps"]
    assert stressed.trade_log[0]["exit_execution_cost_bps"] > calm.trade_log[0]["exit_execution_cost_bps"]


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
    assert report.asdict()["evaluated_thresholds"] == 7


def test_futures_threshold_calibration_can_disable_bad_short_side() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=120_000, close=130.0, features=(-1.0,), label=0),
        ModelRow(timestamp=180_000, close=130.0, features=(-1.0,), label=0),
        ModelRow(timestamp=240_000, close=130.0, features=(0.0,), label=0),
        ModelRow(timestamp=300_000, close=170.0, features=(0.0,), label=0),
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
    cfg = StrategyConfig(
        risk_per_trade=0.5,
        max_position_pct=0.5,
        signal_threshold=0.60,
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
        cooldown_minutes=0,
        max_trades_per_day=0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_risk_enabled=False,
        max_regime_unpredictability=1.0,
    )

    symmetric = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="futures", compute_backend="cpu")
    report = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.60,
        start=0.55,
        end=0.75,
        steps=5,
        starting_cash=1000.0,
        market_type="futures",
        min_closed_trades=1,
        compute_backend="cpu",
    )
    selected = run_backtest(
        rows,
        TrainedModel(
            **{
                **model.__dict__,
                "decision_threshold": report.best_threshold,
                "long_decision_threshold": report.best_long_threshold,
                "short_decision_threshold": report.best_short_threshold,
            }
        ),
        cfg,
        starting_cash=1000.0,
        market_type="futures",
        compute_backend="cpu",
    )

    assert symmetric.liquidation_events == 1
    assert report.best_realized_pnl > symmetric.realized_pnl
    assert report.best_long_threshold is not None
    assert report.best_short_threshold is None
    assert report.best_liquidation_events == 0
    assert selected.trade_log
    assert {int(trade["side"]) for trade in selected.trade_log} == {1}


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
    assert reversed_grid.evaluated_thresholds == 1

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


def test_threshold_calibration_can_add_probability_rank_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index * 0.05, features=(1.0,), label=1)
        for index in range(240)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    calls = {"count": 0}

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        calls["count"] += 1
        return [0.50 + 0.45 * index / max(1, len(scored_rows) - 1) for index, _row in enumerate(scored_rows)], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(risk_per_trade=0.1, signal_threshold=0.5),
        baseline_threshold=0.5,
        start=0.5,
        end=0.95,
        steps=3,
        compute_backend="directml",
        adaptive_probability_thresholds=True,
        max_adaptive_thresholds=32,
    )

    assert calls["count"] == 1
    assert report.evaluated_thresholds > 3
    assert report.evaluated_thresholds <= 32


def test_probability_adaptive_threshold_grid_cannot_loop_on_duplicate_kept_values() -> None:
    probabilities = [0.50 + index * 0.01 for index in range(60)]
    dense_base = tuple(0.50 + index * 0.01 for index in range(46))

    thresholds = backtest_module._probability_adaptive_threshold_grid(
        probabilities,
        start=0.50,
        end=0.95,
        baseline=0.50,
        market_type="futures",
        max_thresholds=10,
        base_thresholds=dense_base,
    )

    assert len(thresholds) == 10
    assert thresholds == sorted(set(thresholds))
    assert thresholds[0] >= 0.50
    assert thresholds[-1] <= 0.95


def test_threshold_calibration_reports_pre_replay_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index * 0.05, features=(1.0,), label=1)
        for index in range(120)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        return [0.80 for _row in scored_rows], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)
    events: list[tuple[str, dict[str, object]]] = []

    calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(risk_per_trade=0.1, signal_threshold=0.5, max_trades_per_day=10),
        baseline_threshold=0.5,
        start=0.5,
        end=0.7,
        steps=3,
        compute_backend="directml",
        status_callback=lambda phase, payload: events.append((phase, dict(payload))),
    )

    phases = [phase for phase, _payload in events]
    assert "threshold_probability_scoring_complete" in phases
    assert "threshold_precompute_complete" in phases
    assert "threshold_baseline_complete" in phases
    assert "threshold_grid_complete" in phases
    assert "threshold_calibration_progress" in phases


def test_threshold_calibration_can_lock_futures_to_short_side(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0, features=(1.0,), label=1)
        for index in range(120)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        long_decision_threshold=None,
        short_decision_threshold=0.44,
    )

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        return [0.20 for _row in scored_rows], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)
    events: list[tuple[str, dict[str, object]]] = []

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(risk_per_trade=0.1, signal_threshold=0.56, max_trades_per_day=10),
        baseline_threshold=0.56,
        start=0.5,
        end=0.7,
        steps=3,
        market_type="futures",
        compute_backend="directml",
        allowed_sides="short",
        status_callback=lambda phase, payload: events.append((phase, dict(payload))),
    )

    progress = [payload for phase, payload in events if phase == "threshold_calibration_progress"]
    grid = next(payload for phase, payload in events if phase == "threshold_grid_complete")
    assert grid["allowed_sides"] == "short"
    assert report.evaluated_thresholds <= 4
    assert progress
    assert all(payload["long_threshold"] is None for payload in progress)
    assert all(float(payload["short_threshold"]) <= 0.5 for payload in progress)


def test_threshold_calibration_prescreen_uses_regime_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 1_000, close=100.0 + index * 0.01, features=(1.0,), label=1)
        for index in range(180)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        return [0.95 for _ in scored_rows], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)
    monkeypatch.setattr(backtest_module, "precompute_backtest_regime_scores", lambda scored_rows, _cfg: [0.95 for _ in scored_rows])
    monkeypatch.setattr(
        backtest_module,
        "precompute_backtest_liquidity_adjustments",
        lambda scored_rows, _cfg: [(0.0, 1.0, False, False) for _ in scored_rows],
    )

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(
            risk_per_trade=0.1,
            signal_threshold=0.5,
            max_regime_unpredictability=0.60,
            unpredictability_cooldown_minutes=90,
            cooldown_minutes=0,
        ),
        baseline_threshold=0.5,
        start=0.5,
        end=0.9,
        steps=7,
        min_closed_trades=3,
        compute_backend="directml",
    )

    assert report.evaluated_thresholds == 0
    assert report.accepted is False


def test_threshold_calibration_prescreen_allows_borderline_regime_soft_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        ModelRow(timestamp=index * 1_000, close=100.0 + index * 0.02, features=(1.0,), label=1)
        for index in range(180)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    def fake_probabilities(scored_rows, *_args, **_kwargs):
        return [0.95 for _ in scored_rows], BackendInfo(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    monkeypatch.setattr(backtest_module, "_backtest_probabilities", fake_probabilities)
    monkeypatch.setattr(backtest_module, "precompute_backtest_regime_scores", lambda scored_rows, _cfg: [0.70 for _ in scored_rows])
    monkeypatch.setattr(
        backtest_module,
        "precompute_backtest_liquidity_adjustments",
        lambda scored_rows, _cfg: [(0.0, 1.0, False, False) for _ in scored_rows],
    )

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        StrategyConfig(
            risk_per_trade=0.1,
            signal_threshold=0.5,
            max_regime_unpredictability=0.60,
            unpredictability_cooldown_minutes=90,
            cooldown_minutes=0,
        ),
        baseline_threshold=0.5,
        start=0.5,
        end=0.9,
        steps=3,
        min_closed_trades=3,
        compute_backend="directml",
    )

    assert report.evaluated_thresholds > 0
    assert report.best_closed_trades > 0


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
