from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import derivatives_hurdle_data as hurdle
from simple_ai_trading import second_flow_execution_model as timing
from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, MinuteSeries, SYMBOLS
from simple_ai_trading.derivatives_hurdle_data import (
    DerivativesHurdleDataset,
    DerivativesSourceEvidence,
    FundingState,
)
from simple_ai_trading.derivatives_hurdle_model import replay_actions
from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_labels import FundingCashLabelSeries
from simple_ai_trading.second_flow_data import START_MS, SecondFlowSeries


def cash(
    symbol: str, start: int, event: int, *, rate: str = "0.005", mark: int = 102
) -> FundingCashLabelSeries:
    settlement = LinearFundingSettlement(
        symbol, "USDT", event, Fraction(rate), Fraction(mark), "a" * 64
    )
    return FundingCashLabelSeries(
        symbol,
        (settlement,),
        ((event, Fraction(rate)),),
        start,
        start + 20_000_000,
        "b" * 64,
    )


def state(source: FundingCashLabelSeries) -> FundingState:
    zeros = np.zeros(250)
    return FundingState(
        np.array([source.settlements[0].funding_time_ms]),
        np.array([float(source.settlements[0].rate)]),
        np.array([8]),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )


def source_evidence() -> DerivativesSourceEvidence:
    return DerivativesSourceEvidence(
        "synthetic", "c" * 64, SimpleNamespace(asdict=lambda: {}), (), "d" * 64, False
    )


def minute_series(symbol: str, start: int) -> MinuteSeries:
    rows = 250
    price = np.full(rows, 100.0)
    return MinuteSeries(
        symbol,
        start + np.arange(rows, dtype=np.int64) * MINUTE_MS,
        price,
        price,
        price,
        price,
        np.ones(rows),
        np.ones(rows),
        np.ones(rows),
        np.ones(rows),
        np.ones(rows),
    )


def hurdle_inputs(monkeypatch, *, rate: str = "0.005", event_offset: int = 70):
    start = hurdle.ROLES[2].start_ms
    panel = {symbol: minute_series(symbol, start) for symbol in SYMBOLS}
    zeros, ones = np.zeros(250), np.ones(250)
    premium = {
        symbol: SimpleNamespace(
            age_minutes=zeros,
            close_bps=zeros,
            rolling_observed_fraction_15m=ones,
            rolling_observed_fraction_60m=ones,
            rolling_observed_fraction_240m=ones,
            rolling_observed_fraction_1440m=ones,
        )
        for symbol in SYMBOLS
    }
    sources = {
        symbol: cash(symbol, start, start + event_offset * MINUTE_MS, rate=rate)
        for symbol in SYMBOLS
    }
    funding = {symbol: state(source) for symbol, source in sources.items()}
    monkeypatch.setattr(
        hurdle, "_feature_arrays", lambda *args: (("price_toy",), (zeros,))
    )
    monkeypatch.setattr(
        hurdle, "_derivatives_feature_arrays", lambda *args: (("rate_toy",), (zeros,))
    )
    return panel, premium, funding, sources


@pytest.mark.parametrize("rate", ["0.005", "-0.005"])
def test_hurdle_builder_uses_paired_mark_cash_and_preserves_features(
    monkeypatch, rate: str
) -> None:
    panel, premium, funding, sources = hurdle_inputs(monkeypatch, rate=rate)
    evidence = source_evidence()
    legacy = hurdle.build_derivatives_hurdle_dataset(panel, premium, funding, evidence)
    marked = hurdle.build_derivatives_hurdle_dataset(
        panel, premium, funding, evidence, funding_cash=sources
    )
    np.testing.assert_array_equal(marked.features, legacy.features)
    assert marked.source_evidence is not evidence
    assert evidence.funding_cash_provenance is None
    assert "funding_cash_provenance" not in evidence.asdict()
    assert marked.source_evidence.asdict()["funding_cash_provenance"] == {
        symbol: value.provenance() for symbol, value in sources.items()
    }
    for horizon in hurdle.HORIZONS_MINUTES:
        low, high = (
            marked.funding_cash_lower_bps[horizon],
            marked.funding_cash_upper_bps[horizon],
        )
        assert low.shape == high.shape == (marked.rows, 2)
        assert np.all(low <= high)
        assert marked.long_net_utility_bps[horizon][0] == pytest.approx(
            -12 + (-51 if rate == "0.005" else 51), abs=1e-4
        )
        assert marked.short_net_utility_bps[horizon][0] == pytest.approx(
            -12 + (51 if rate == "0.005" else -51), abs=1e-4
        )
        assert marked.target_class[horizon][0] == (0 if rate == "0.005" else 2)
        assert marked.funding_cash_uncertain_events[horizon][0].tolist() == [0, 0]


def test_hurdle_explicit_cash_never_uses_proxy_and_never_backfills_missing_sources(
    monkeypatch,
) -> None:
    panel, premium, funding, sources = hurdle_inputs(monkeypatch, event_offset=61)

    def forbidden(*args):
        raise AssertionError("cash builder used rate proxy")

    monkeypatch.setattr(hurdle, "_funding_in_holding_window", forbidden)
    dataset = hurdle.build_derivatives_hurdle_dataset(
        panel, premium, funding, source_evidence(), funding_cash=sources
    )
    assert dataset.funding_cash_lower_bps[15][0, 0] == 0
    assert dataset.funding_cash_lower_bps[15][0, 1] == pytest.approx(-51, abs=1e-4)
    assert dataset.funding_cash_uncertain_events[15][0].tolist() == [1, 1]
    with pytest.raises(ValueError):
        hurdle.build_derivatives_hurdle_dataset(
            panel, premium, funding, source_evidence(), funding_cash={}
        )
    with pytest.raises(ValueError):
        hurdle.build_derivatives_hurdle_dataset(
            {**panel, "BTCUSDT": replace(panel["BTCUSDT"], symbol="ETHUSDT")},
            premium,
            funding,
            source_evidence(),
            funding_cash=sources,
        )
    with pytest.raises(ValueError):
        hurdle.build_derivatives_hurdle_dataset(
            panel,
            premium,
            funding,
            source_evidence(),
            funding_cash={
                **sources,
                "BTCUSDT": replace(
                    sources["BTCUSDT"],
                    coverage_end_exclusive_ms=panel["BTCUSDT"].open_time_ms[80],
                ),
            },
        )


def replay_dataset() -> DerivativesHurdleDataset:
    evidence = replace(
        source_evidence(), funding_cash_provenance={"fixture": "not_certified"}
    )
    low = np.tile([-7.0, -11.0], (3, 1)).astype(np.float32)
    return DerivativesHurdleDataset(
        ("fixture",),
        1,
        np.zeros((3, 1)),
        np.full(3, hurdle.ROLES[2].start_ms),
        np.arange(3),
        {15: np.zeros(3)},
        {15: np.full(3, 10)},
        {15: np.full(3, 12)},
        {15: np.full(3, 11)},
        {15: {"calibration": np.ones(3, dtype=bool)}},
        evidence,
        {},
        {15: low},
        {15: np.zeros((3, 2))},
        {15: np.ones((3, 2), dtype=np.int64)},
    )


def replay(dataset: DerivativesHurdleDataset):
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    return replay_actions(
        dataset,
        probabilities,
        horizon=15,
        role="calibration",
        maximum_action_probability=0.6,
        direction_probability_margin=0.2,
        bootstrap_samples=0,
        bootstrap_seed=1,
    )


def test_action_replay_retains_both_side_debits_instead_of_sign_flipping_long_cash() -> (
    None
):
    outcome = replay(replay_dataset())
    assert outcome.metrics.total_funding_cash_flow_bps == -25
    assert outcome.net_return_bps.tolist() == [12, 10, 12]


@pytest.mark.parametrize(
    "change", ["missing", "shape", "nonfinite", "order", "count", "provenance"]
)
def test_action_replay_rejects_incomplete_or_conflicting_cash_bounds(
    change: str,
) -> None:
    dataset = replay_dataset()
    if change == "missing":
        dataset = replace(dataset, funding_cash_upper_bps=None)
    if change == "shape":
        dataset = replace(dataset, funding_cash_upper_bps={15: np.zeros(3)})
    if change == "nonfinite":
        dataset = replace(dataset, funding_cash_upper_bps={15: np.full((3, 2), np.nan)})
    if change == "order":
        dataset = replace(dataset, funding_cash_upper_bps={15: np.full((3, 2), -99)})
    if change == "count":
        dataset = replace(
            dataset, funding_cash_uncertain_events={15: np.full((3, 2), -1)}
        )
    if change == "provenance":
        dataset = replace(dataset, source_evidence=source_evidence())
    with pytest.raises(ValueError):
        replay(dataset)


def timing_inputs(monkeypatch, *, rate: str = "0.005", mark: int = 102):
    entry_times = START_MS + np.tile(np.array([2_000_000, 2_500_000]), 3)
    minutes = SimpleNamespace(
        decision_time_ms=entry_times - MINUTE_MS,
        symbol_index=np.repeat(np.arange(3), 2),
    )
    probabilities = np.tile([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], (3, 1))
    monkeypatch.setattr(
        timing, "_primary_probabilities", lambda *args, **kwargs: (probabilities, ())
    )
    monkeypatch.setattr(timing, "_base_feature_names", lambda: ("toy",))
    monkeypatch.setattr(timing, "_proposal_features", lambda **kwargs: (0.0,))
    series = {}
    for symbol in SYMBOLS:
        rows = 5_000
        price, volume = np.full(rows, 100.0), np.ones(rows)
        series[symbol] = SecondFlowSeries(
            symbol,
            START_MS + np.arange(rows) * 1000,
            price,
            price,
            price,
            price,
            volume,
            volume * 100,
            volume,
            volume * 0.5,
            volume * 50,
            ("synthetic",),
        )
    sources = {
        symbol: cash(symbol, START_MS, START_MS + 2_600_000, rate=rate, mark=mark)
        for symbol in SYMBOLS
    }
    funding = {symbol: state(value) for symbol, value in sources.items()}
    return minutes, series, funding, sources


def build_timing(inputs, *, cash_inputs=True):
    minutes, series, funding, sources = inputs
    return timing.build_timing_dataset(
        minutes,
        series,
        funding,
        round41_report={},
        round41_evidence_root=Path("unused"),
        funding_cash=sources if cash_inputs else None,
    )


@pytest.mark.parametrize("rate", ["0.005", "-0.005"])
def test_timing_builder_batches_mark_cash_without_future_feature_leak(
    monkeypatch, rate: str
) -> None:
    inputs = timing_inputs(monkeypatch, rate=rate)
    legacy = build_timing(inputs, cash_inputs=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("cash timing used rate proxy")

    monkeypatch.setattr(timing, "_funding_bps", forbidden)
    marked = build_timing(inputs)
    np.testing.assert_array_equal(marked.features, legacy.features)
    assert marked.option_funding_cash_lower_bps.shape == (24,)
    assert np.all(marked.option_funding_uncertain_events == 0)
    first, second = (51, -51) if rate == "0.005" else (-51, 51)
    assert marked.option_base_net_bps[0] == pytest.approx(-12 + first, abs=1e-4)
    assert marked.option_base_net_bps[4] == pytest.approx(-12 + second, abs=1e-4)
    assert marked.option_stress_net_bps[0] == pytest.approx(-16 + first, abs=1e-4)
    assert marked.funding_cash_provenance == {
        symbol: source.provenance() for symbol, source in inputs[3].items()
    }
    monkeypatch.setattr(timing, "FOLDS", ())
    timing._validate_timing_dataset(marked)
    changed = build_timing(timing_inputs(monkeypatch, rate=rate, mark=110))
    np.testing.assert_array_equal(marked.features, changed.features)
    assert not np.array_equal(marked.option_base_net_bps, changed.option_base_net_bps)


@pytest.mark.parametrize(
    "change", ["missing", "shape", "nonfinite", "order", "count", "provenance", "debit"]
)
def test_timing_validator_rejects_cash_metadata_conflicts(
    monkeypatch, change: str
) -> None:
    dataset = build_timing(timing_inputs(monkeypatch))
    if change == "missing":
        dataset = replace(dataset, option_funding_cash_upper_bps=None)
    if change == "shape":
        dataset = replace(dataset, option_funding_cash_upper_bps=np.zeros(2))
    if change == "nonfinite":
        dataset = replace(dataset, option_funding_cash_upper_bps=np.full(24, np.nan))
    if change == "order":
        dataset = replace(dataset, option_funding_cash_upper_bps=np.full(24, -99))
    if change == "count":
        dataset = replace(dataset, option_funding_uncertain_events=np.full(24, -1))
    if change == "provenance":
        dataset = replace(dataset, funding_cash_provenance=None)
    if change == "debit":
        dataset = replace(dataset, option_funding_bps=np.full(24, 11))
    with pytest.raises(ValueError, match="cash"):
        timing._validate_timing_dataset(dataset)


def test_timing_builder_rejects_missing_cash_and_wrong_price_symbol(
    monkeypatch,
) -> None:
    minutes, series, funding, sources = timing_inputs(monkeypatch)
    with pytest.raises(ValueError):
        timing.build_timing_dataset(
            minutes,
            series,
            funding,
            round41_report={},
            round41_evidence_root=Path("unused"),
            funding_cash={},
        )
    with pytest.raises(ValueError):
        timing.build_timing_dataset(
            minutes,
            {**series, "BTCUSDT": replace(series["BTCUSDT"], symbol="ETHUSDT")},
            funding,
            round41_report={},
            round41_evidence_root=Path("unused"),
            funding_cash=sources,
        )
