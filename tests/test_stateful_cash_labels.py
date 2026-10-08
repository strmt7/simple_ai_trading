from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, MinuteSeries, SYMBOLS
from simple_ai_trading.derivatives_hurdle_data import (
    DerivativesSourceEvidence,
    FundingState,
)
from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_labels import FundingCashLabelSeries
from simple_ai_trading.stateful_cash_labels import (
    _float_enclosure,
    build_stateful_cash_labels,
)
from simple_ai_trading.funding_cash_inventory import (
    InventoryCashPath,
    replay_fixed_base_inventory,
)
from simple_ai_trading.stateful_turnover_model import (
    StatefulHourlyDataset,
    train_stateful_forecasts,
)

HOUR = 60 * MINUTE_MS


def fixture(*, boundary: bool = False, rate: str = "0.005", exit_price: float = 100.0):
    decisions = np.array([0, HOUR], dtype=np.int64)
    features = np.zeros((6, 1), dtype=np.float32)
    dataset = StatefulHourlyDataset(
        ("fixture",),
        features,
        features.copy(),
        np.repeat(decisions, 3),
        np.tile(np.arange(3, dtype=np.int8), 2),
        np.zeros(6),
        np.zeros(6),
        cast(DerivativesSourceEvidence, None),
        "a" * 64,
    )
    panel, funding, cash = {}, {}, {}
    for symbol in SYMBOLS:
        count = 122
        opens = np.full(count, 100.0)
        opens[61:] = exit_price
        panel[symbol] = MinuteSeries(
            symbol,
            np.arange(count, dtype=np.int64) * MINUTE_MS,
            opens,
            opens.copy(),
            opens.copy(),
            opens.copy(),
            *[np.ones(count) for _ in range(5)],
        )
        times = (MINUTE_MS, 61 * MINUTE_MS) if boundary else (2 * MINUTE_MS,)
        rates = (Fraction(rate), -Fraction(rate)) if boundary else (Fraction(rate),)
        events = tuple(
            LinearFundingSettlement(symbol, "USDT", t, r, Fraction(102), "b" * 64)
            for t, r in zip(times, rates, strict=True)
        )
        cash[symbol] = FundingCashLabelSeries(
            symbol,
            events,
            tuple(zip(times, rates, strict=True)),
            0,
            123 * MINUTE_MS,
            "c" * 64,
        )
        zeros = np.zeros(count)
        funding[symbol] = FundingState(
            event_time_ms=np.array(times, dtype=np.int64),
            event_rate=np.array([float(r) for r in rates]),
            event_interval_hours=np.full(len(times), 8),
            last_rate_bps=zeros,
            last_interval_hours=zeros,
            age_minutes=zeros,
            settled_sum_24h_bps=zeros,
            settled_sum_72h_bps=zeros,
            settled_sum_168h_bps=zeros,
            event_mean_30_bps=zeros,
            event_zscore_30=zeros,
        )
    kwargs = dict(
        price_panel=panel,
        funding=funding,
        funding_cash=cash,
        one_way_cost_fraction=Fraction(6, 10_000),
        price_source_sha256="d" * 64,
    )
    return dataset, kwargs


@pytest.mark.parametrize("rate", ["0.005", "-0.005", "0"])
def test_mark_scaled_both_sides_keep_entry_exit_costs(rate):
    dataset, kwargs = fixture(rate=rate)
    labels = build_stateful_cash_labels(dataset, **kwargs)
    for side, low, high in (
        (1, labels.long_net_lower_bps, labels.long_net_upper_bps),
        (-1, labels.short_net_lower_bps, labels.short_net_upper_bps),
    ):
        expected = float(-side * Fraction(rate) * 102 / 100 * 10_000 - 12)
        assert low[0] <= expected <= high[0]
        assert low[0] == pytest.approx(expected)
        assert low[3] <= -12 <= high[3]
    assert labels.uncertain_funding_events.tolist() == [0] * 6
    assert not labels.financially_qualified


def test_both_adverse_cash_labels_are_not_opposite_signed_targets():
    dataset, kwargs = fixture(boundary=True)
    labels = build_stateful_cash_labels(dataset, **kwargs)
    assert labels.long_net_lower_bps[0] == pytest.approx(-63)
    assert labels.short_net_lower_bps[0] == pytest.approx(-63)
    assert labels.long_net_upper_bps[0] == pytest.approx(39)
    assert labels.short_net_upper_bps[0] == pytest.approx(39)
    assert labels.uncertain_funding_events.tolist() == [2, 2, 2, 1, 1, 1]
    assert (
        labels.long_net_lower_bps[0] - labels.short_net_lower_bps[0]
    ) / 2 == pytest.approx(0)


def test_exit_trade_cost_uses_exit_quote_value_not_two_entry_charges():
    dataset, kwargs = fixture(exit_price=200)
    labels = build_stateful_cash_labels(dataset, **kwargs)
    assert labels.long_net_lower_bps[0] == pytest.approx(10_000 - 51 - 18)
    assert labels.short_net_lower_bps[0] == pytest.approx(-10_000 + 51 - 18)
    assert labels.long_net_lower_bps[3] == pytest.approx(-12)


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("rate", ["0.005", "-0.005", "0"])
@pytest.mark.parametrize("exit_price", [97.125, 100.0, 200.0])
def test_each_side_encloses_independent_exact_inventory_ledger(
    boundary, rate, exit_price
):
    dataset, kwargs = fixture(boundary=boundary, rate=rate, exit_price=exit_price)
    labels = build_stateful_cash_labels(dataset, **kwargs)
    source = kwargs["funding_cash"][SYMBOLS[0]]
    for side, low, high in (
        (1, labels.long_net_lower_bps, labels.long_net_upper_bps),
        (-1, labels.short_net_lower_bps, labels.short_net_upper_bps),
    ):
        ledger = replay_fixed_base_inventory(
            InventoryCashPath(
                SYMBOLS[0],
                (MINUTE_MS, 61 * MINUTE_MS),
                (Fraction(100), Fraction.from_float(exit_price)),
                (side,),
                Fraction(100),
                kwargs["one_way_cost_fraction"],
                kwargs["price_source_sha256"],
                source.settlements,
                source.expected_events,
                source.coverage_start_ms,
                source.coverage_end_exclusive_ms,
                source.population_certificate_sha256,
            )
        )
        exact_low = ledger.total_net_cash_lower * 100
        exact_high = ledger.total_net_cash_upper * 100
        assert (
            Fraction.from_float(low[0])
            <= exact_low
            <= exact_high
            <= Fraction.from_float(high[0])
        )


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(-1, 3), Fraction(1, 8)])
def test_price_cash_float_enclosure_retains_exact_fraction(value):
    low, high = _float_enclosure(value)
    assert Fraction.from_float(low) <= value <= Fraction.from_float(high)


def test_unrepresentable_price_cash_rejects():
    with pytest.raises(ValueError, match="exceeds label dtype"):
        _float_enclosure(Fraction(10**400))


def test_legacy_targets_are_excluded_and_feature_price_mark_bindings_change():
    dataset, kwargs = fixture()
    labels = build_stateful_cash_labels(dataset, **kwargs)
    changed = replace(
        dataset,
        signed_pre_transition_utility_bps=np.full(6, np.nan),
        funding_cash_flow_bps=np.full(6, 999),
    )
    assert (
        build_stateful_cash_labels(changed, **kwargs).label_sha256
        == labels.label_sha256
    )
    assert (
        build_stateful_cash_labels(
            replace(dataset, baseline_features=np.ones((6, 1))), **kwargs
        ).label_sha256
        != labels.label_sha256
    )
    assert (
        build_stateful_cash_labels(
            dataset, **(kwargs | {"price_source_sha256": "e" * 64})
        ).label_sha256
        != labels.label_sha256
    )
    new_cash = dict(kwargs["funding_cash"])
    old = new_cash[SYMBOLS[0]]
    new_cash[SYMBOLS[0]] = replace(
        old, settlements=(replace(old.settlements[0], settlement_mark=Fraction(103)),)
    )
    assert (
        build_stateful_cash_labels(
            dataset, **(kwargs | {"funding_cash": new_cash})
        ).label_sha256
        != labels.label_sha256
    )
    for name in ("long_net_lower_bps", "entry_time_ms", "symbol_index"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(labels, name)[0] = 0


def test_cash_labels_cannot_enter_legacy_signed_trainer(tmp_path: Path):
    dataset, kwargs = fixture()
    labels = build_stateful_cash_labels(dataset, **kwargs)
    directory = tmp_path / "models"
    with pytest.raises(ValueError, match="legacy signed trainer"):
        train_stateful_forecasts(labels, model_dir=directory, compute_backend="cpu")
    assert not directory.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "cash_none",
        "cash_missing",
        "price_missing",
        "funding_missing",
        "cost_zero",
        "cost_float",
        "hash_bad",
        "reference_bad",
        "symbol_bad",
        "coverage",
        "rate_conflict",
        "price_missing_clock",
        "price_nan",
        "grid_gap",
        "grid_duplicate",
        "grid_symbol",
        "clock_bool",
        "clock_overflow",
        "feature_nan",
        "feature_shape",
        "augmented_shape",
    ],
)
def test_invalid_evidence_rejects_without_proxy_fallback(mutation):
    dataset, kwargs = fixture()
    if mutation == "cash_none":
        kwargs["funding_cash"] = None
    elif mutation in ("cash_missing", "price_missing", "funding_missing"):
        name = {
            "cash_missing": "funding_cash",
            "price_missing": "price_panel",
            "funding_missing": "funding",
        }[mutation]
        kwargs[name].pop(SYMBOLS[0])
    elif mutation.startswith("cost_"):
        kwargs["one_way_cost_fraction"] = (
            Fraction(0) if mutation == "cost_zero" else 0.0006
        )
    elif mutation == "hash_bad":
        kwargs["price_source_sha256"] = "bad"
    elif mutation == "reference_bad":
        dataset = replace(dataset, dataset_sha256="bad")
    elif mutation == "symbol_bad":
        kwargs["price_panel"][SYMBOLS[0]] = replace(
            kwargs["price_panel"][SYMBOLS[0]], symbol=SYMBOLS[1]
        )
    elif mutation == "coverage":
        kwargs["funding_cash"][SYMBOLS[0]] = replace(
            kwargs["funding_cash"][SYMBOLS[0]], coverage_end_exclusive_ms=70 * MINUTE_MS
        )
    elif mutation == "rate_conflict":
        kwargs["funding"][SYMBOLS[0]].event_rate[0] = 0.01
    elif mutation == "price_missing_clock":
        kwargs["price_panel"][SYMBOLS[0]].open_time_ms[1] = 59_999
    elif mutation == "price_nan":
        kwargs["price_panel"][SYMBOLS[0]].open[1] = np.nan
    elif mutation in ("grid_gap", "grid_duplicate"):
        dataset.decision_time_ms[3:] = 2 * HOUR if mutation == "grid_gap" else 0
    elif mutation == "grid_symbol":
        dataset.symbol_index[0] = 2
    elif mutation == "clock_bool":
        dataset = replace(dataset, decision_time_ms=np.zeros(6, dtype=bool))
    elif mutation == "clock_overflow":
        dataset = replace(
            dataset, decision_time_ms=np.full(6, 2**64 - 1, dtype=np.uint64)
        )
    elif mutation == "feature_nan":
        dataset.baseline_features[0] = np.nan
    elif mutation == "feature_shape":
        dataset = replace(dataset, baseline_features=np.zeros((6, 2)))
    elif mutation == "augmented_shape":
        dataset = replace(dataset, augmented_features=np.zeros((5, 1)))
    with pytest.raises(ValueError):
        build_stateful_cash_labels(dataset, **kwargs)
