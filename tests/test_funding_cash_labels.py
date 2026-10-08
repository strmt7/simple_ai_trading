from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.barrier_payoff_data import (
    BarrierSpecification,
    REALIZED_VOLATILITY_FEATURE,
    _simulate_side as simulate_barrier,
    build_barrier_payoff_dataset,
)
from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, MinuteSeries, SYMBOLS
from simple_ai_trading.derivatives_hurdle_data import FundingState
from simple_ai_trading.funding_cash import (
    FundingEntitlement,
    LinearFundingSettlement,
    funding_cash_bounds_for_events,
)
from simple_ai_trading.funding_cash_labels import (
    FundingCashLabelSeries,
    bind_funding_cash_panel,
    outward_float32,
)
from simple_ai_trading.stop_time_payoff_data import (
    StopTimeSpecification,
    _simulate_side as simulate_stop,
    build_stop_time_payoff_dataset,
)


def cash_source(
    times: tuple[int, ...] = (120_000,),
    rates: tuple[str, ...] = ("0.005",),
    *,
    symbol: str = "BTCUSDT",
    mark: str = "102",
) -> FundingCashLabelSeries:
    events = tuple(
        LinearFundingSettlement(
            symbol, "USDT", time, Fraction(rate), Fraction(mark), "a" * 64
        )
        for time, rate in zip(times, rates, strict=True)
    )
    return FundingCashLabelSeries(
        symbol,
        events,
        tuple((event.funding_time_ms, event.rate) for event in events),
        0,
        2_000_000,
        "b" * 64,
    )


def rate_state(source: FundingCashLabelSeries) -> FundingState:
    zeros = np.zeros(40)
    return FundingState(
        event_time_ms=np.asarray(
            [e.funding_time_ms for e in source.settlements], dtype=np.int64
        ),
        event_rate=np.asarray([float(e.rate) for e in source.settlements]),
        event_interval_hours=np.full(len(source.settlements), 8),
        last_rate_bps=zeros,
        last_interval_hours=zeros,
        age_minutes=zeros,
        settled_sum_24h_bps=zeros,
        settled_sum_72h_bps=zeros,
        settled_sum_168h_bps=zeros,
        event_mean_30_bps=zeros,
        event_zscore_30=zeros,
    )


def price_series(symbol: str = "BTCUSDT", *, stop: bool = False) -> MinuteSeries:
    count = 25
    opens = np.full(count, 100.0)
    lows = np.full(count, 99.9)
    if stop:
        lows[2] = 98.0
    return MinuteSeries(
        symbol=symbol,
        open_time_ms=np.arange(count, dtype=np.int64) * MINUTE_MS,
        open=opens,
        high=np.full(count, 100.1),
        low=lows,
        close=opens.copy(),
        volume=np.ones(count),
        quote_volume=np.ones(count),
        trade_count=np.ones(count),
        taker_buy_base_volume=np.ones(count),
        taker_buy_quote_volume=np.ones(count),
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("rate", ["0.005", "-0.005", "0"])
def test_mark_scaled_labels_match_exact_cash_law(side: int, rate: str) -> None:
    source = cash_source(rates=(rate,))
    bounds = source.holding_bounds(
        np.array([60_000]),
        np.array([240_000]),
        np.array([240_000]),
        np.array([100.0]),
        side=side,
    )
    exact = funding_cash_bounds_for_events(
        source.settlements,
        (FundingEntitlement.HELD,),
        expected_symbol=source.symbol,
        signed_base_quantity=Fraction(side * 7),
        entry_price=Fraction(100),
    )
    assert bounds.lower_bps[0] == pytest.approx(float(exact.entry_relative_lower_bps))
    assert bounds.upper_bps[0] == pytest.approx(float(exact.entry_relative_upper_bps))
    assert bounds.uncertain_events.tolist() == [0]
    if rate == "0.005":
        assert bounds.lower_bps[0] == pytest.approx(-side * 51)


@pytest.mark.parametrize("side", [-1, 1])
def test_vectorized_uncertainty_matches_per_event_exact_enclosure(side: int) -> None:
    source = cash_source(
        (60_000, 120_000, 150_000, 180_000, 240_000),
        ("0.001", "-0.002", "0.003", "-0.004", "0.005"),
    )
    entries = np.array([0, 60_000, 120_000, 150_000, 180_000])
    early = np.array([180_000, 120_000, 120_000, 150_000, 240_000])
    late = np.array([240_000, 180_000, 150_000, 150_000, 240_000])
    bounds = source.holding_bounds(entries, early, late, np.full(5, 100.0), side=side)
    for i, (entry, first_exit, last_exit) in enumerate(
        zip(entries, early, late, strict=True)
    ):
        entitlements = tuple(
            FundingEntitlement.HELD
            if entry < e.funding_time_ms < first_exit
            else FundingEntitlement.UNKNOWN
            if entry <= e.funding_time_ms <= last_exit
            else FundingEntitlement.NOT_HELD
            for e in source.settlements
        )
        exact = funding_cash_bounds_for_events(
            source.settlements,
            entitlements,
            expected_symbol=source.symbol,
            signed_base_quantity=Fraction(side),
            entry_price=Fraction(100),
        )
        assert bounds.lower_bps[i] == pytest.approx(
            float(exact.entry_relative_lower_bps)
        )
        assert bounds.upper_bps[i] == pytest.approx(
            float(exact.entry_relative_upper_bps)
        )
        assert bounds.uncertain_events[i] == exact.uncertain_entitlement_events
        assert (
            Fraction.from_float(bounds.lower_bps[i]) <= exact.entry_relative_lower_bps
        )
        assert (
            Fraction.from_float(bounds.upper_bps[i]) >= exact.entry_relative_upper_bps
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"coverage_start_ms": True},
        {"coverage_end_exclusive_ms": 0},
        {"coverage_end_exclusive_ms": 2**63},
        {"coverage_start_ms": 120_001},
        {"population_certificate_sha256": "missing"},
        {"symbol": "ETHUSDT"},
        {"expected_events": ()},
        {"expected_events": ((120_000, Fraction("0.006")),)},
        {"settlements": []},
    ],
)
def test_cash_source_rejects_missing_conflicting_or_mutable_evidence(
    changes: dict,
) -> None:
    with pytest.raises(ValueError):
        replace(cash_source(), **changes)


@pytest.mark.parametrize(
    "entry,early,late,price,side",
    [
        ([0.5], [240_000], [240_000], [100], 1),
        ([-1], [240_000], [240_000], [100], 1),
        ([60_000], [30_000], [240_000], [100], 1),
        ([60_000], [240_000], [180_000], [100], 1),
        ([60_000], [240_000], [2_000_000], [100], 1),
        ([60_000], [240_000], [240_000], [0], 1),
        ([60_000], [240_000], [240_000], [float("nan")], 1),
        ([60_000], [240_000], [240_000], [100], True),
        ([60_000], [240_000], [240_000], [100, 100], 1),
    ],
)
def test_holding_clock_and_price_fail_closed(entry, early, late, price, side) -> None:
    with pytest.raises(ValueError):
        cash_source().holding_bounds(
            np.asarray(entry),
            np.asarray(early),
            np.asarray(late),
            np.asarray(price),
            side=side,
        )


def test_empty_certified_population_and_empty_label_batch() -> None:
    source = cash_source((), ())
    bounds = source.holding_bounds(
        np.array([60_000]),
        np.array([240_000]),
        np.array([240_000]),
        np.array([100]),
        side=1,
    )
    assert bounds.lower_bps.tolist() == bounds.upper_bps.tolist() == [0.0]
    empty = np.array([], dtype=np.int64)
    assert source.holding_bounds(empty, empty, empty, empty, side=1).lower_bps.size == 0


@pytest.mark.parametrize(
    "change", ["missing", "rate", "time", "nan", "duplicate", "float_clock"]
)
def test_binding_rejects_consumer_population_conflict(change: str) -> None:
    source = cash_source()
    state = rate_state(source)
    if change == "missing":
        state = replace(
            state, event_time_ms=np.array([], dtype=np.int64), event_rate=np.array([])
        )
    if change == "rate":
        state = replace(state, event_rate=np.array([0.006]))
    if change == "time":
        state = replace(state, event_time_ms=np.array([120_001]))
    if change == "nan":
        state = replace(state, event_rate=np.array([np.nan]))
    if change == "duplicate":
        state = replace(
            state,
            event_time_ms=np.array([120_000, 120_000]),
            event_rate=np.array([0.005, 0.005]),
        )
    if change == "float_clock":
        state = replace(state, event_time_ms=np.array([120_000.1]))
    with pytest.raises(ValueError):
        source.validate_rate_state(state)


def test_provenance_hash_and_complete_panel_binding() -> None:
    source = cash_source()
    provenance = source.provenance()
    claimed = provenance.pop("funding_cash_label_sha256")
    assert (
        hashlib.sha256(
            json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        ).hexdigest()
        == claimed
    )
    assert (
        replace(source, population_certificate_sha256="c" * 64).provenance()[
            "funding_cash_label_sha256"
        ]
        != claimed
    )
    assert bind_funding_cash_panel(None, {}) is None
    assert (
        bind_funding_cash_panel(
            {source.symbol: source}, {source.symbol: rate_state(source)}
        )[source.symbol]["funding_cash_label_sha256"]
        == claimed
    )
    with pytest.raises(ValueError):
        bind_funding_cash_panel({}, {source.symbol: rate_state(source)})
    with pytest.raises(ValueError):
        bind_funding_cash_panel({"ETHUSDT": source}, {"ETHUSDT": rate_state(source)})


@pytest.mark.parametrize("kind", ["stop", "barrier"])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("rate", ["0.005", "-0.005"])
def test_simulators_apply_mark_weighted_timeout_cash(
    kind: str, side: int, rate: str
) -> None:
    source = cash_source(rates=(rate,))
    common = (price_series(), rate_state(source), np.array([0]), np.array([100.0]))
    if kind == "stop":
        outputs = simulate_stop(
            *common,
            side=side,
            specification=StopTimeSpecification(3, 1, 100, 100, 10),
            funding_cash=source,
        )
        actual = outputs[4][0]
    else:
        outputs = simulate_barrier(
            *common,
            np.array([200.0]),
            side=side,
            specification=BarrierSpecification(3, 1, 2, 100, 100, 10),
            funding_cash=source,
        )
        actual = -side * outputs[3][0]
    assert actual == pytest.approx(-side * (51 if rate == "0.005" else -51))
    assert outputs[5 if kind == "stop" else 4][0] == pytest.approx(actual - 10)


def test_intrabar_stop_does_not_credit_unknown_positive_funding() -> None:
    source = cash_source(rates=("-0.005",))
    outputs = simulate_stop(
        price_series(stop=True),
        rate_state(source),
        np.array([0]),
        np.array([100.0]),
        side=1,
        specification=StopTimeSpecification(3, 1, 100, 100, 10),
        funding_cash=source,
    )
    assert outputs[4].tolist() == [0.0]
    assert outputs[5][0] == pytest.approx(-110.0)
    # The payment at the stop minute's start may or may not be held.
    direct = source.holding_bounds(
        np.array([60_000]),
        np.array([120_000]),
        np.array([180_000]),
        np.array([100.0]),
        side=1,
    )
    assert direct.upper_bps[0] == pytest.approx(51.0)
    assert direct.uncertain_events.tolist() == [1]


@pytest.mark.parametrize("kind", ["stop", "barrier"])
def test_builders_bind_cash_provenance_without_changing_legacy_identity(
    kind: str,
) -> None:
    sources = {symbol: cash_source(symbol=symbol) for symbol in SYMBOLS}
    funding = {symbol: rate_state(source) for symbol, source in sources.items()}
    panel = {symbol: price_series(symbol) for symbol in SYMBOLS}
    if kind == "stop":

        def build(cash=None):
            return build_stop_time_payoff_dataset(
                panel,
                funding,
                np.array([0]),
                np.ones((1, 3)),
                source_dataset_sha256="d" * 64,
                specification=StopTimeSpecification(3, 1, 100, 100, 10),
                funding_cash=cash,
            )
    else:
        masks = {
            role: np.arange(4) == i
            for i, role in enumerate(
                ("training", "early_stop", "calibration", "viability")
            )
        }
        predecessor = SimpleNamespace(
            rows=12,
            role_masks={3: {role: np.tile(mask, 3) for role, mask in masks.items()}},
        )
        temporal = SimpleNamespace(
            timestamps=4,
            timestamps_ms=np.array([0, 300_000, 600_000, 900_000]),
            feature_names=(REALIZED_VOLATILITY_FEATURE,),
            features=np.ones((4, 3, 1)),
            dataset_sha256="d" * 64,
        )

        def build(cash=None):
            return build_barrier_payoff_dataset(
                panel,
                funding,
                predecessor,
                temporal,
                BarrierSpecification(3, 1, 2, 100, 100, 10),
                funding_cash=cash,
            )

    legacy = build()
    marked = build(sources)
    assert legacy.funding_cash_provenance is None
    assert marked.funding_cash_provenance == {
        symbol: source.provenance() for symbol, source in sources.items()
    }
    assert legacy.funding_cash_upper_bps is legacy.funding_cash_uncertain_events is None
    assert (
        marked.funding_cash_upper_bps.shape
        == marked.funding_cash_uncertain_events.shape
    )
    assert marked.funding_cash_upper_bps.shape[-1] == 2
    assert np.all(marked.funding_cash_uncertain_events >= 0)
    assert legacy.dataset_sha256 != marked.dataset_sha256
    assert build().dataset_sha256 == legacy.dataset_sha256
    with pytest.raises(ValueError):
        build({"BTCUSDT": sources["BTCUSDT"]})
    with pytest.raises(ValueError):
        build(
            {
                **sources,
                "BTCUSDT": replace(
                    sources["BTCUSDT"], coverage_end_exclusive_ms=180_000
                ),
            }
        )


@pytest.mark.parametrize("kind", ["stop", "barrier"])
def test_simulators_reject_wrong_cash_symbol(kind: str) -> None:
    source = cash_source(symbol="ETHUSDT")
    args = (price_series(), rate_state(source), np.array([0]), np.array([100.0]))
    with pytest.raises(ValueError, match="symbol"):
        if kind == "stop":
            simulate_stop(
                *args,
                side=1,
                specification=StopTimeSpecification(3, 1, 100, 100, 10),
                funding_cash=source,
            )
        else:
            simulate_barrier(
                *args,
                np.array([200.0]),
                side=1,
                specification=BarrierSpecification(3, 1, 2, 100, 100, 10),
                funding_cash=source,
            )


@pytest.mark.parametrize("lower", [False, True])
def test_float32_storage_remains_outward_including_subnormal_values(
    lower: bool,
) -> None:
    values = np.array([-51.00000000000001, -1e-60, 0, 1e-60, 51.00000000000001])
    stored = outward_float32(values, lower=lower)
    assert np.all(stored <= values if lower else stored >= values)
    assert stored[2] == 0
    with pytest.raises(ValueError):
        outward_float32(np.array([np.inf]), lower=lower)
    with pytest.raises(ValueError):
        outward_float32(np.array([-1e300, 1e300]), lower=lower)


@pytest.mark.parametrize("mark", ["1" + "0" * 400, "1e-400"])
def test_cash_precision_is_outward_or_rejects_unrepresentable_prefix(mark: str) -> None:
    source = cash_source(mark=mark)
    if "e" not in mark:
        with pytest.raises(ValueError, match="dtype"):
            source.holding_bounds(
                np.array([60_000]),
                np.array([240_000]),
                np.array([240_000]),
                np.array([100.0]),
                side=1,
            )
    else:
        bounds = source.holding_bounds(
            np.array([60_000]),
            np.array([240_000]),
            np.array([240_000]),
            np.array([100.0]),
            side=1,
        )
        exact = -Fraction(mark) * Fraction("0.005") * 100
        assert (
            Fraction.from_float(bounds.lower_bps[0])
            <= exact
            <= Fraction.from_float(bounds.upper_bps[0])
        )


@pytest.mark.parametrize("rates", [np.array([0.005 + 0j]), np.array(["0.005"])])
def test_consumer_rate_state_rejects_nonreal_numeric_schema(rates: np.ndarray) -> None:
    source = cash_source()
    with pytest.raises(ValueError):
        source.validate_rate_state(replace(rate_state(source), event_rate=rates))


@pytest.mark.parametrize("kind", ["stop", "barrier"])
def test_explicit_cash_simulation_never_calls_legacy_proxy(
    monkeypatch, kind: str
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("cash path called the legacy rate proxy")

    monkeypatch.setattr(
        f"simple_ai_trading.{kind if kind == 'barrier' else 'stop_time'}_payoff_data._funding_in_holding_window",
        forbidden,
    )
    source = cash_source()
    common = (price_series(), rate_state(source), np.array([0]), np.array([100.0]))
    if kind == "stop":
        simulate_stop(
            *common,
            side=1,
            specification=StopTimeSpecification(3, 1, 100, 100, 10),
            funding_cash=source,
        )
    else:
        simulate_barrier(
            *common,
            np.array([200.0]),
            side=1,
            specification=BarrierSpecification(3, 1, 2, 100, 100, 10),
            funding_cash=source,
        )
