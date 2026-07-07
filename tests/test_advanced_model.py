"""Branch-coverage tests for the advanced model / feature-expansion module."""

from __future__ import annotations

import math

import pytest

from simple_ai_trading import advanced_model as am
from simple_ai_trading.api import Candle
from simple_ai_trading.compute import resolve_backend
from simple_ai_trading.features import FEATURE_NAMES, ModelRow


def _candles(n: int = 220) -> list[Candle]:
    out = []
    for i in range(n):
        price = 100.0 + (i % 7) * 0.5 + (i * 0.01)
        volume = 1.0 + (i % 3)
        close = price + 0.1
        trade_count = 4 + (i % 5)
        taker_buy_base = volume * (0.35 + 0.1 * (i % 4))
        out.append(Candle(
            open_time=i * 60_000,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=close,
            volume=volume,
            close_time=i * 60_000 + 59_000,
            quote_volume=close * volume,
            trade_count=trade_count,
            taker_buy_base_volume=taker_buy_base,
            taker_buy_quote_volume=close * taker_buy_base,
        ))
    return out


def test_tanh_overflow_branch_positive():
    # math.tanh raises OverflowError for very large inputs on some platforms; the
    # helper must handle it by clamping to Â±1.0.
    # We monkeypatch math.tanh via the module to force the overflow branch.
    original = am.math.tanh
    try:
        am.math.tanh = lambda _x: (_ for _ in ()).throw(OverflowError)
        assert am._tanh(1.0) == 1.0
        assert am._tanh(-1.0) == -1.0
    finally:
        am.math.tanh = original


def test_tanh_happy_path():
    assert am._tanh(0.0) == 0.0
    assert -1.0 < am._tanh(0.5) < 1.0


def test_log1p_signed_zero_and_values():
    assert am._log1p_signed(0.0) == 0.0
    assert am._log1p_signed(1.0) > 0.0
    assert am._log1p_signed(-1.0) < 0.0


def test_sma_and_rsi_guards():
    assert math.isnan(am._sma([1.0, 2.0], 0))
    assert math.isnan(am._sma([1.0], 5))
    assert am._sma([1.0, 2.0, 3.0], 3) == 2.0
    assert math.isnan(am._rsi([1.0], 0))
    assert math.isnan(am._rsi([1.0], 5))
    # all losses zero â†’ 100
    assert am._rsi([1.0, 1.0, 1.0, 1.0, 1.0], 3) == 100.0
    # alternating produces a number strictly between 0 and 100
    val = am._rsi([1.0, 2.0, 1.0, 2.0, 1.0], 3)
    assert math.isfinite(val)


def test_volatility_guards():
    assert math.isnan(am._volatility([1.0], 0))
    assert math.isnan(am._volatility([1.0], 5))
    assert am._volatility([1.0, 2.0, 3.0, 4.0], 3) > 0.0


def test_safe_helper_and_extras():
    assert am._safe(float("inf")) == 0.0
    assert am._safe(1.2) == pytest.approx(1.2)
    closes = [100.0 + i for i in range(40)]
    features = am._extra_window_features(closes, (5, 10))
    assert len(features) == 14
    assert am._extra_window_features([], (5, 10)) == [0.0] * 14
    cache = am._build_window_cache(closes)
    assert math.isnan(am._window_mean(cache.close_prefix, 2, 1))


def test_confluence_features_are_finite_and_dimensioned():
    candles = _candles(80)
    cache = am._build_confluence_cache(candles)
    features = am._confluence_features_at(cache, 60, (5, 20))

    assert len(features) == 18
    assert all(math.isfinite(value) for value in features)
    assert am._confluence_features_at(cache, -1, (5,)) == [0.0] * 9

    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        confluence_windows=(5, 20),
    )
    assert am.advanced_feature_dimension(cfg) == 4 + 7 + 18 + 8


def test_market_quality_features_are_finite_and_live_inference_safe():
    candles = _candles(160)
    cache = am._build_confluence_cache(candles)
    features = am._market_quality_features_at(cache, 120, (20, 60))

    assert len(features) == 20
    assert all(math.isfinite(value) for value in features)
    assert am._market_quality_features_at(cache, -1, (20,)) == [0.0] * 10

    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(),
        confluence_windows=(),
        market_quality_windows=(20, 60),
    )
    train_rows = am.make_advanced_rows(candles, cfg)
    inference_rows = am.make_advanced_inference_rows(candles, cfg)

    assert train_rows
    assert inference_rows
    assert len(train_rows[0].features) == am.advanced_feature_dimension(cfg)
    assert len(inference_rows[-1].features) == am.advanced_feature_dimension(cfg)


def test_higher_timeframe_context_uses_only_closed_context_bars():
    candles = [
        Candle(
            open_time=i * 1_000,
            open=100.0 + i * 0.01,
            high=100.2 + i * 0.01,
            low=99.8 + i * 0.01,
            close=100.05 + i * 0.01,
            volume=1.0 + (i % 4),
            close_time=i * 1_000 + 999,
            quote_volume=(100.05 + i * 0.01) * (1.0 + (i % 4)),
            trade_count=2 + (i % 3),
        )
        for i in range(120)
    ]
    context = am._aggregate_higher_timeframe_candles(candles, 60_000)
    close_times = [candle.close_time for candle in context]

    assert len(context) == 2
    assert close_times == [59_999, 119_999]
    assert am._higher_timeframe_context_features_at(context, close_times, 119_998, (2,)) == [0.0] * 8
    features = am._higher_timeframe_context_features_at(context, close_times, 119_999, (2,))
    assert len(features) == 8
    assert all(math.isfinite(value) for value in features)
    assert features[0] > 0.0


def test_higher_timeframe_context_is_dimensioned_for_train_and_inference():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(),
        confluence_windows=(),
        market_quality_windows=(),
        higher_timeframe_windows=(2, 5),
        higher_timeframe_bucket_ms=60_000,
    )
    candles = _candles(80)
    rows = am.make_advanced_rows(candles, cfg)
    inference_rows = am.make_advanced_inference_rows(candles, cfg)

    assert am.advanced_feature_dimension(cfg) == 4 + 16 + 8
    assert rows
    assert inference_rows
    assert len(rows[-1].features) == am.advanced_feature_dimension(cfg)
    assert len(inference_rows[-1].features) == am.advanced_feature_dimension(cfg)


def test_make_advanced_rows_directml_matches_cpu_when_available() -> None:
    if resolve_backend("directml").kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:8]),
        polynomial_top_features=4,
        extra_lookback_windows=(5, 20),
        confluence_windows=(5,),
        market_quality_windows=(10,),
    )
    candles = _candles(180)

    cpu_rows = am.make_advanced_rows(candles, cfg)
    gpu_rows = am.make_advanced_rows(candles, cfg, compute_backend="directml")
    cpu_inference = am.make_advanced_inference_rows(candles, cfg)
    gpu_inference = am.make_advanced_inference_rows(candles, cfg, compute_backend="directml")

    assert len(gpu_rows) == len(cpu_rows)
    assert len(gpu_inference) == len(cpu_inference)
    for left, right in zip(cpu_rows[:5], gpu_rows[:5], strict=True):
        assert right.timestamp == left.timestamp
        assert right.label == left.label
        assert right.features == pytest.approx(left.features, abs=1e-5)
    assert gpu_inference[-1].features == pytest.approx(cpu_inference[-1].features, abs=1e-5)


def test_market_quality_prefix_features_match_naive_reference() -> None:
    candles = _candles(240)
    cache = am._build_confluence_cache(candles)
    end = 200
    windows = (20, 60, 120)

    def naive(window: int) -> list[float]:
        start = end + 1 - window
        closes = cache.closes[start:end + 1]
        volumes = cache.volumes[start:end + 1]
        returns = am._window_returns(closes)
        abs_returns = [abs(value) for value in returns]
        path = sum(abs_returns)
        net_return = am._safe_ratio(closes[-1] - closes[0], closes[0])
        efficiency = am._safe_ratio(abs(net_return), path)
        downside_pressure = am._safe_ratio(sum(abs(value) for value in returns if value < 0.0), path)
        autocorr = am._correlation(returns[:-1], returns[1:]) if len(returns) >= 3 else 0.0
        mean_abs_return = sum(abs_returns) / len(abs_returns) if abs_returns else 0.0
        abs_return_stdev = 0.0
        if len(abs_returns) >= 2:
            abs_return_stdev = math.sqrt(
                max(0.0, sum((value - mean_abs_return) ** 2 for value in abs_returns) / (len(abs_returns) - 1))
            )
        tail_ratio = am._safe_ratio(max(abs_returns) if abs_returns else 0.0, mean_abs_return)
        return_volumes = volumes[1:] if len(volumes) > 1 else []
        total_return_volume = sum(max(0.0, value) for value in return_volumes)
        volume_return_pressure = am._safe_ratio(
            sum(ret * max(0.0, vol) for ret, vol in zip(returns, return_volumes, strict=True)),
            total_return_volume,
        )
        volume_abs_return_corr = am._correlation(return_volumes, abs_returns) if return_volumes else 0.0
        avg_true_range = am._window_mean(cache.true_range_prefix, start, end)
        avg_volume = sum(max(0.0, value) for value in volumes) / len(volumes)
        current_volume = cache.volumes[end]
        close = cache.closes[end]
        return [
            am._safe(math.copysign(efficiency, net_return)),
            am._safe(efficiency),
            am._safe(downside_pressure),
            am._safe(autocorr),
            am._safe_ratio(abs_return_stdev, mean_abs_return),
            am._safe(math.tanh(tail_ratio / 5.0)),
            am._safe(math.tanh(volume_return_pressure * 120.0)),
            am._safe(volume_abs_return_corr),
            am._safe_ratio(avg_true_range, close),
            am._safe(math.tanh(am._safe_ratio(current_volume - avg_volume, avg_volume))),
        ]

    expected = [value for window in windows for value in naive(window)]
    actual = am._market_quality_features_at(cache, end, windows)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_order_flow_features_match_naive_reference_and_dimensioned() -> None:
    candles = _candles(240)
    cache = am._build_confluence_cache(candles)
    end = 200
    windows = (20, 60)

    def naive(window: int) -> list[float]:
        start = end + 1 - window
        sum_volume = sum(cache.nonnegative_volumes[start:end + 1])
        sum_quote_volume = sum(cache.quote_volumes[start:end + 1])
        sum_trade_count = sum(cache.trade_counts[start:end + 1])
        sum_taker_base = sum(cache.taker_buy_base_volumes[start:end + 1])
        sum_taker_quote = sum(cache.taker_buy_quote_volumes[start:end + 1])
        signed_base = sum(cache.signed_base_flows[start:end + 1])
        avg_trade_count = sum_trade_count / float(window)
        avg_quote_volume = sum_quote_volume / float(window)
        current_quote_volume = cache.quote_volumes[end]
        current_trade_count = cache.trade_counts[end]
        avg_quote_per_trade = am._safe_ratio(sum_quote_volume, sum_trade_count)
        current_quote_per_trade = am._safe_ratio(current_quote_volume, current_trade_count)
        signed_ratios = cache.signed_flow_ratios[start:end + 1]
        returns = cache.returns[start:end + 1]
        midpoint = start + max(1, window // 2)
        first_signed_ratio = am._safe_ratio(
            sum(cache.signed_flow_ratios[start:midpoint]),
            float(max(1, midpoint - start)),
        )
        second_signed_ratio = am._safe_ratio(
            sum(cache.signed_flow_ratios[midpoint:end + 1]),
            float(max(1, end - midpoint + 1)),
        )
        net_return = am._safe_ratio(cache.closes[end] - cache.closes[start], cache.closes[start])
        mean_signed_ratio = sum(signed_ratios) / float(window)
        return [
            am._safe_ratio(sum_taker_base, sum_volume),
            am._safe_ratio(signed_base, sum_volume),
            am._safe_ratio((2.0 * sum_taker_quote) - sum_quote_volume, sum_quote_volume),
            am._safe(math.tanh(am._safe_ratio(current_trade_count - avg_trade_count, avg_trade_count))),
            am._safe(math.tanh(am._safe_ratio(current_quote_volume - avg_quote_volume, avg_quote_volume))),
            am._safe(math.tanh(math.log1p(current_quote_per_trade) - math.log1p(avg_quote_per_trade))),
            am._safe_ratio(sum(cache.no_trade_flags[start:end + 1]), float(window)),
            am._safe(am._correlation(signed_ratios, returns)),
            am._safe(cache.signed_flow_ratios[end] - mean_signed_ratio),
            am._safe(sum(abs(value) for value in signed_ratios) / float(window)),
            am._safe(am._correlation(signed_ratios[:-1], signed_ratios[1:])),
            am._safe(second_signed_ratio - first_signed_ratio),
            am._safe(math.tanh((mean_signed_ratio * 2.0) - math.tanh(net_return * 250.0))),
        ]

    expected = [value for window in windows for value in naive(window)]
    actual = am._order_flow_features_at(cache, end, windows)

    assert len(actual) == 26
    assert actual == pytest.approx(expected, abs=1e-12)
    assert all(math.isfinite(value) for value in actual)
    assert am._order_flow_features_at(cache, -1, (20,)) == [0.0] * 13

    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(),
        confluence_windows=(),
        market_quality_windows=(),
        order_flow_windows=windows,
    )
    rows = am.make_advanced_rows(candles, cfg)
    assert rows
    assert len(rows[0].features) == am.advanced_feature_dimension(cfg)
    assert am.advanced_feature_dimension(cfg) == 4 + 26 + 8


def test_nonlinear_expand_unknown_raises():
    with pytest.raises(ValueError):
        am._nonlinear_expand([0.1], ["unknown"])


def test_polynomial_pairs_branches():
    assert am._polynomial_pairs([1.0, 2.0], top_k=1, degree=2) == []
    assert am._polynomial_pairs([1.0, 2.0], top_k=2, degree=1) == []
    pairs = am._polynomial_pairs([1.0, 2.0, 3.0], top_k=3, degree=2)
    # upper triangle: (i*i pairs): 3+2+1 = 6 values
    assert len(pairs) == 6
    cubes = am._polynomial_pairs([1.0, 2.0, 3.0], top_k=3, degree=3)
    assert len(cubes) == 10  # 6 pairs + 1 triple + 3 cubes


def test_advanced_feature_dimension_skips_polynomial_when_disabled():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
    )
    dim = am.advanced_feature_dimension(cfg)
    # polynomial disabled: base(4) + extras(7) + transforms(4*2=8) = 19
    assert dim == 19


def test_advanced_feature_dimension_degree3_branch():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:6]),
        polynomial_degree=3,
        polynomial_top_features=5,
        extra_lookback_windows=(5,),
    )
    dim = am.advanced_feature_dimension(cfg)
    # base(6) + extras(7) + transforms(6*2=12) + pairs (k=5 -> 15) + cube bonus (4) = 44
    assert dim == 6 + 7 + 12 + 15 + 4


def test_advanced_feature_dimension_matches_expand():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:6]),
        polynomial_degree=2,
        polynomial_top_features=4,
        extra_lookback_windows=(5, 20),
    )
    dim = am.advanced_feature_dimension(cfg)
    row = ModelRow(timestamp=0, close=100.0, features=(0.1,) * 6, label=1, volume=42.0)
    candles = _candles(60)
    expanded = am.expand_row(row, candles, cfg, at_index=59)
    assert len(expanded.features) == dim
    assert expanded.volume == pytest.approx(42.0)


def test_advanced_feature_group_spans_cover_dimension_in_order() -> None:
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:5]),
        polynomial_degree=3,
        polynomial_top_features=4,
        extra_lookback_windows=(5, 20),
        confluence_windows=(8,),
        market_quality_windows=(13,),
        higher_timeframe_windows=(34,),
        order_flow_windows=(21,),
        nonlinear_transforms=("tanh", "log1p"),
    )

    spans = am.advanced_feature_group_spans(cfg)

    assert [span.name for span in spans] == [
        "base_features",
        "extra_lookback_windows",
        "technical_confluence",
        "market_quality_regime",
        "higher_timeframe_context",
        "order_flow_microstructure",
        "nonlinear_transforms",
        "polynomial_interactions",
    ]
    assert spans[0].start == 0
    assert spans[-1].end == am.advanced_feature_dimension(cfg)
    assert all(left.end == right.start for left, right in zip(spans[:-1], spans[1:], strict=True))
    assert spans[-1].asdict()["size"] == spans[-1].size


def test_advanced_feature_signature_stable():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    assert am.advanced_feature_signature(cfg) == am.advanced_feature_signature(cfg)
    triple = am.AdvancedFeatureConfig(
        base_features=FEATURE_NAMES,
        label_mode="triple_barrier",
        label_stop_threshold=0.002,
    )
    assert am.advanced_feature_signature(cfg) != am.advanced_feature_signature(triple)


def test_advanced_config_from_signature_round_trips_candidate_specific_fields() -> None:
    cfg = am.AdvancedFeatureConfig(
        base_features=FEATURE_NAMES[:8],
        polynomial_degree=3,
        polynomial_top_features=7,
        extra_lookback_windows=(4, 12, 48),
        confluence_windows=(5, 13, 34),
        market_quality_windows=(10, 30),
        higher_timeframe_windows=(60, 240),
        higher_timeframe_bucket_ms=60_000,
        order_flow_windows=(6, 18),
        nonlinear_transforms=("tanh", "log1p"),
        short_window=8,
        long_window=34,
        label_threshold=0.00168,
        label_lookahead=7,
        label_mode="triple_barrier",
        label_stop_threshold=0.00125,
    )

    parsed = am.advanced_config_from_signature(am.advanced_feature_signature(cfg), FEATURE_NAMES)

    assert parsed is not None
    assert parsed == cfg
    assert am.advanced_feature_signature(parsed) == am.advanced_feature_signature(cfg)
    assert am.advanced_config_from_signature("feature_version=v1") is None


def test_legacy_v8_signature_keeps_nine_order_flow_fields_per_window() -> None:
    cfg = am.AdvancedFeatureConfig(
        base_features=FEATURE_NAMES[:4],
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(),
        confluence_windows=(),
        market_quality_windows=(),
        order_flow_windows=(6, 18),
    )
    current_parts = am.advanced_feature_signature(cfg).split("|")
    legacy_parts = [
        part.replace("advanced_version=v10-higher-timeframe-context", "advanced_version=v8-information-event")
        for part in current_parts
        if not part.startswith("order_flow_features_per_window=")
        and not part.startswith("higher_timeframe_windows=")
        and not part.startswith("higher_timeframe_bucket_ms=")
    ]
    parsed = am.advanced_config_from_signature("|".join(legacy_parts), FEATURE_NAMES)

    assert parsed is not None
    assert parsed.order_flow_features_per_window == 9
    assert am.advanced_feature_dimension(parsed) == 4 + (2 * 9) + 8
    rows = am.make_advanced_rows(_candles(240), parsed)
    assert rows
    assert len(rows[0].features) == am.advanced_feature_dimension(parsed)


def test_default_config_for_branches():
    a = am.default_config_for("conservative", FEATURE_NAMES)
    b = am.default_config_for("risky", FEATURE_NAMES)
    c = am.default_config_for("default", FEATURE_NAMES)
    d = am.default_config_for("nothing", ())
    assert a.polynomial_top_features == 5
    assert a.confluence_windows == (12, 36, 96)
    assert a.market_quality_windows == (30, 90, 180)
    assert a.higher_timeframe_windows == (60, 240, 720)
    assert a.order_flow_windows == (15, 45, 120)
    assert a.label_lookahead == 8
    assert b.polynomial_degree == 3
    assert b.confluence_windows == (5, 13, 34, 89)
    assert b.market_quality_windows == (10, 30, 90)
    assert b.higher_timeframe_windows == (10, 30, 120)
    assert b.order_flow_windows == (5, 15, 45, 90)
    assert b.label_threshold == pytest.approx(0.0005)
    assert c.higher_timeframe_windows == (20, 60, 180)
    assert c.polynomial_top_features == len(FEATURE_NAMES)
    assert c.confluence_windows == (8, 21, 55)
    assert c.market_quality_windows == (20, 60, 120)
    assert c.order_flow_windows == (10, 30, 90)
    assert c.label_lookahead == 4
    assert d.polynomial_top_features == len(FEATURE_NAMES)


def test_make_advanced_rows_happy_path():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    candles = _candles(250)
    rows = am.make_advanced_rows(candles, cfg)
    short_horizon_rows = am.make_advanced_rows(candles, cfg, lookahead=1)
    volume_by_timestamp = {candle.close_time: candle.volume for candle in candles}
    assert rows
    assert len(short_horizon_rows) > len(rows)
    assert len(rows[0].features) == am.advanced_feature_dimension(cfg)
    assert rows[0].volume == pytest.approx(volume_by_timestamp[rows[0].timestamp])


def test_make_advanced_rows_can_use_triple_barrier_labels() -> None:
    candles = _candles(120)
    cfg = am.AdvancedFeatureConfig(
        base_features=FEATURE_NAMES,
        label_threshold=0.002,
        label_stop_threshold=0.001,
        label_lookahead=4,
        label_mode="triple_barrier",
    )

    rows = am.make_advanced_rows(candles, cfg)

    assert rows
    assert {row.label for row in rows} <= {0, 1}
    assert "label_mode=triple_barrier" in am.advanced_feature_signature(cfg)


def test_make_advanced_rows_can_use_downside_labels() -> None:
    candles = []
    for index in range(80):
        close = 120.0 - index * 0.08
        candles.append(Candle(
            open_time=index * 1000,
            open=close + 0.02,
            high=close + 0.04,
            low=close - 0.12,
            close=close,
            volume=2.0,
            close_time=index * 1000 + 999,
            quote_volume=close * 2.0,
            trade_count=5,
            taker_buy_base_volume=0.6,
            taker_buy_quote_volume=close * 0.6,
        ))
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_lookahead=4,
        label_mode="downside_forward_return",
    )
    barrier_cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_lookahead=4,
        label_mode="downside_triple_barrier",
        label_stop_threshold=0.002,
    )

    forward_rows = am.make_advanced_rows(candles, cfg)
    barrier_rows = am.make_advanced_rows(candles, barrier_cfg)

    assert forward_rows
    assert barrier_rows
    assert sum(row.label for row in forward_rows) > 0
    assert sum(row.label for row in barrier_rows) > 0
    assert "label_mode=downside_forward_return" in am.advanced_feature_signature(cfg)
    assert "label_mode=downside_triple_barrier" in am.advanced_feature_signature(barrier_cfg)


def test_volatility_adjusted_threshold_uses_trailing_returns_only() -> None:
    candles: list[Candle] = []
    close = 100.0
    for index in range(120):
        if index < 60:
            close *= 1.00001
        else:
            close *= 1.004 if index % 2 else 0.996
        candles.append(Candle(
            open_time=index * 1000,
            open=close,
            high=close * 1.0005,
            low=close * 0.9995,
            close=close,
            volume=5.0,
            close_time=index * 1000 + 999,
            quote_volume=close * 5.0,
            trade_count=20,
            taker_buy_base_volume=2.5,
            taker_buy_quote_volume=close * 2.5,
        ))
    cache = am._build_confluence_cache(candles)

    calm = am._volatility_adjusted_label_threshold_pct(
        cache,
        50,
        base_threshold=0.0005,
        volatility_window=20,
        volatility_multiplier=2.5,
    )
    noisy = am._volatility_adjusted_label_threshold_pct(
        cache,
        100,
        base_threshold=0.0005,
        volatility_window=20,
        volatility_multiplier=2.5,
    )
    no_lookahead = am._volatility_adjusted_label_threshold_pct(
        cache,
        50,
        base_threshold=0.0005,
        volatility_window=20,
        volatility_multiplier=2.5,
    )

    assert calm == pytest.approx(no_lookahead)
    assert calm == pytest.approx(0.0005)
    assert noisy > calm


def test_make_advanced_rows_can_use_volatility_barrier_labels() -> None:
    candles = _candles(180)
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_stop_threshold=0.0007,
        label_lookahead=12,
        label_mode="volatility_triple_barrier",
        label_volatility_window=20,
        label_volatility_multiplier=2.0,
    )
    downside_cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_stop_threshold=0.0007,
        label_lookahead=12,
        label_mode="downside_volatility_triple_barrier",
        label_volatility_window=20,
        label_volatility_multiplier=2.0,
    )

    rows = am.make_advanced_rows(candles, cfg)
    downside_rows = am.make_advanced_rows(candles, downside_cfg)
    signature = am.advanced_feature_signature(cfg)
    restored = am.advanced_config_from_signature(signature)

    assert rows
    assert downside_rows
    assert {row.label for row in rows} <= {0, 1}
    assert {row.label for row in downside_rows} <= {0, 1}
    assert "label_mode=volatility_triple_barrier" in signature
    assert "label_volatility_window=20" in signature
    assert "label_volatility_multiplier=2" in signature
    assert restored is not None
    assert restored.label_mode == "volatility_triple_barrier"
    assert restored.label_volatility_window == 20
    assert restored.label_volatility_multiplier == pytest.approx(2.0)


def test_trailing_cusum_event_direction_uses_only_past_returns() -> None:
    candles: list[Candle] = []
    close = 100.0
    for index in range(12):
        if index <= 5:
            close *= 1.0001
        else:
            close *= 1.01
        candles.append(Candle(
            open_time=index * 1000,
            open=close,
            high=close * 1.0002,
            low=close * 0.9998,
            close=close,
            volume=10.0,
            close_time=index * 1000 + 999,
            quote_volume=close * 10.0,
            trade_count=50,
            taker_buy_base_volume=5.0,
            taker_buy_quote_volume=close * 5.0,
        ))
    cache = am._build_confluence_cache(candles)

    before_jump = am._trailing_cusum_event_direction(cache, 5, window=5, threshold_pct=0.005)
    after_jump = am._trailing_cusum_event_direction(cache, 8, window=5, threshold_pct=0.005)

    assert before_jump == 0
    assert after_jump == 1


def test_make_advanced_rows_can_use_information_event_barrier_labels() -> None:
    candles: list[Candle] = []
    close = 100.0
    for index in range(140):
        if 45 <= index < 55:
            close *= 1.0015
        elif 80 <= index < 90:
            close *= 0.9985
        elif index >= 55 and index < 65:
            close *= 1.0010
        elif index >= 90 and index < 100:
            close *= 0.9990
        else:
            close *= 1.00001
        candles.append(Candle(
            open_time=index * 1000,
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=20.0,
            close_time=index * 1000 + 999,
            quote_volume=close * 20.0,
            trade_count=80,
            taker_buy_base_volume=10.0,
            taker_buy_quote_volume=close * 10.0,
        ))
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_stop_threshold=0.0007,
        label_lookahead=10,
        label_mode="information_event_triple_barrier",
        label_volatility_window=12,
        label_volatility_multiplier=0.0,
    )
    downside_cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
        label_threshold=0.0005,
        label_stop_threshold=0.0007,
        label_lookahead=10,
        label_mode="downside_information_event_triple_barrier",
        label_volatility_window=12,
        label_volatility_multiplier=0.0,
    )

    rows = am.make_advanced_rows(candles, cfg)
    downside_rows = am.make_advanced_rows(candles, downside_cfg)
    signature = am.advanced_feature_signature(cfg)
    restored = am.advanced_config_from_signature(signature)

    assert sum(row.label for row in rows) > 0
    assert sum(row.label for row in downside_rows) > 0
    assert "label_mode=event_volatility_triple_barrier" in signature
    assert restored is not None
    assert restored.label_mode == "event_volatility_triple_barrier"


def test_make_advanced_rows_handles_missing_index(monkeypatch):
    cfg = am.default_config_for("default", FEATURE_NAMES)
    # Force every base row to carry a timestamp not present in index_by_time
    from simple_ai_trading import advanced_model as mod

    def fake_base(*_args, **_kwargs):
        return [ModelRow(timestamp=-999, close=100.0, features=(0.0,) * len(FEATURE_NAMES), label=0)]

    monkeypatch.setattr(mod, "make_base_rows", fake_base)
    rows = am.make_advanced_rows(_candles(60), cfg)
    assert rows == []


def test_make_advanced_rows_empty_input():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    assert am.make_advanced_rows([], cfg) == []


def test_make_advanced_inference_rows_empty_and_missing_index(monkeypatch):
    cfg = am.default_config_for("default", FEATURE_NAMES)
    assert am.make_advanced_inference_rows([], cfg) == []

    def fake_base(*_args, **_kwargs):
        return [ModelRow(timestamp=-999, close=100.0, features=(0.0,) * len(FEATURE_NAMES), label=0)]

    monkeypatch.setattr(am, "make_base_inference_rows", fake_base)
    assert am.make_advanced_inference_rows(_candles(60), cfg) == []


def test_filter_valid_rejects_bad_candles():
    bad = [
        Candle(open_time=0, open=float("nan"), high=1, low=1, close=1, volume=1, close_time=1),
        Candle(open_time=0, open=-1, high=1, low=1, close=1, volume=1, close_time=1),
        Candle(open_time=0, open=1, high=1, low=2, close=1, volume=1, close_time=1),  # low>high
        Candle(open_time=0, open=5, high=2, low=1, close=1, volume=1, close_time=1),  # open out of band
        Candle(open_time=0, open=1, high=2, low=1, close=3, volume=1, close_time=1),  # close out of band
        Candle(open_time=10, open=1, high=2, low=1, close=1, volume=1, close_time=5),  # close_time<open_time
        Candle(open_time=0, open=1, high=2, low=1, close=1, volume=-1, close_time=1),  # negative volume
        Candle(open_time=-1, open=1, high=2, low=1, close=1, volume=1, close_time=1),
    ]
    assert am._filter_valid(bad) == []


def test_train_advanced_empty_rows():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    with pytest.raises(ValueError):
        am.train_advanced([], cfg, epochs=10, learning_rate=0.05, l2_penalty=1e-3)


def test_train_advanced_produces_matching_dim():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    rows = am.make_advanced_rows(_candles(260), cfg)
    assert rows
    model, report = am.train_advanced(rows, cfg, epochs=3, learning_rate=0.05, l2_penalty=1e-3)
    assert model.feature_dim == len(rows[0].features)
    assert report.row_count == len(rows)
    assert 0.0 <= report.positive_rate <= 1.0
    assert report.feature_signature == am.advanced_feature_signature(cfg)


def test_train_advanced_can_build_seed_ensemble():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    rows = am.make_advanced_rows(_candles(260), cfg)
    model, report = am.train_advanced(
        rows,
        cfg,
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-3,
        seed=5,
        ensemble_seeds=(5, 7, 5),
    )

    assert report.seed == 5
    assert model.seed == 5
    assert len(model.ensemble_members) == 2
    assert {member.seed for member in model.ensemble_members} == {5, 7}
    assert 0.0 <= model.predict_proba(rows[-1].features) <= 1.0
