"""Hash-bound consumers must not mint fresh targets from changed source bytes."""

from __future__ import annotations

from dataclasses import fields, replace
from decimal import Decimal

import numpy as np
import pytest

from simple_ai_trading.make_take_scenario_entries import (
    _scenario_entry_payload,
    _sha256,
    build_make_take_scenario_entries,
)
from simple_ai_trading.make_take_entry_integrity import (
    validate_make_take_scenario_entry_batch,
)
from simple_ai_trading.make_take_targets import build_make_take_targets
from simple_ai_trading.make_take_payoff_panel import (
    build_make_take_conditional_payoff_panel,
)
from simple_ai_trading.queue_fill_survival import build_passive_fill_survival_panel
from simple_ai_trading.paper_execution import (
    AggressiveTradePrint,
    PassiveQueueState,
    apply_passive_trade_print,
)
from simple_ai_trading.queue_censored_actions import build_passive_fill_result


def _inputs(printed_quantity: float = 120.0) -> dict[str, object]:
    common = {
        "arrival_time_ms": [1750],
        "queue_ahead_quantity": [100.0],
        "order_notional_quote": 1000.0,
        "trade_id": [1],
        "trade_time_ms": [2750],
        "trade_price": [100.0],
        "trade_quantity": [printed_quantity],
        "trade_buyer_is_maker": [True],
    }
    return {
        "scenario": "base",
        "decision_time_ms": [1000],
        "bid_price": [100.0],
        "ask_price": [100.1],
        "bid_quantity": [100.0],
        "ask_quantity": [100.0],
        "long_fill": build_passive_fill_result(
            placement_price=[100.0], buyer_is_maker=True, **common
        ),
        "short_fill": build_passive_fill_result(
            placement_price=[100.1], buyer_is_maker=False, **common
        ),
    }


def _path(_day_start_ms: int) -> dict[str, np.ndarray]:
    times = np.arange(0, 320_000, 100, dtype=np.int64)
    bid = np.full(times.size, 100.0)
    ask = np.full(times.size, 100.1)
    return {
        "path_time_ms": times,
        "path_min_bid": bid,
        "path_max_bid": bid,
        "path_close_bid": bid,
        "path_min_ask": ask,
        "path_max_ask": ask,
        "path_close_ask": ask,
    }


def _targets(entries, loader=_path):
    return build_make_take_targets(
        symbol="BTCUSDT",
        source_dataset_sha256="a" * 64,
        entries=entries,
        event_stop_bps=[80.0],
        event_take_bps=[120.0],
        load_day_path=loader,
    )


@pytest.mark.parametrize("side", ["long_fill", "short_fill"])
@pytest.mark.parametrize(
    "field", ["result_sha256", "source_trade_sha256", "printed_quantity_through_fill"]
)
def test_entry_builder_rejects_changed_fill_content_with_unchanged_binding(side, field):
    inputs = _inputs()
    fill = inputs[side]
    value = "b" * 64 if field.endswith("sha256") else np.asarray([0.25])
    inputs[side] = replace(fill, **{field: value})
    with pytest.raises(ValueError):
        build_make_take_scenario_entries(**inputs)


@pytest.mark.parametrize(
    "field",
    [
        "entry_cost_bps",
        "exit_cost_bps",
        "entry_price",
        "passive_entry_fee_bps",
        "batch_sha256",
    ],
)
def test_target_builder_rejects_changed_entry_before_loading_price_paths(field):
    entries = build_make_take_scenario_entries(**_inputs())
    value = (
        "b" * 64
        if field == "batch_sha256"
        else 0.0
        if field == "passive_entry_fee_bps"
        else np.full(4, 102.0)
        if field == "entry_price"
        else np.zeros(4)
    )
    loaded = []

    def loader(day):
        loaded.append(day)
        return _path(day)

    with pytest.raises(ValueError):
        _targets(replace(entries, **{field: value}), loader)
    assert not loaded


def test_legacy_censored_zero_does_not_reconstruct_partial_inventory_cash():
    inputs = _inputs(104.0)
    fill = inputs["long_fill"]
    assert not fill.filled[0]
    assert fill.printed_quantity_through_fill[0] == 0.0
    entries = build_make_take_scenario_entries(**inputs)
    targets = _targets(entries)
    assert targets.realized_valid[0] and targets.realized_net_bps[0] == 0.0

    # Same explicit no-cancellation FIFO assumption: four units can have filled.
    marker = PassiveQueueState(
        "partial",
        "BTCUSDT",
        "BUY",
        Decimal("100"),
        Decimal("100"),
        Decimal("10"),
        Decimal("0"),
        1750,
        16750,
    )
    state, partial = apply_passive_trade_print(
        marker,
        AggressiveTradePrint(
            "BTCUSDT", "SELL", Decimal("100"), Decimal("104"), 2750, "a" * 64
        ),
    )
    assert partial == 4 and state.remaining_quantity == 6
    acquisition = partial * Decimal("100")
    liquidation = partial * Decimal("98")
    entry_cost = acquisition * Decimal("0.0003")
    exit_cost = liquidation * Decimal("0.0006")
    assert liquidation - acquisition - entry_cost - exit_cost == Decimal("-8.3552")


def test_valid_entry_and_target_bytes_match_the_pre_repair_goldens():
    entries = build_make_take_scenario_entries(**_inputs())
    validate_make_take_scenario_entry_batch(entries)
    assert (
        entries.batch_sha256
        == "d6400e580a375f2bf74204f36055a9247dca4e66bee479417851614eb8294837"
    )
    assert (
        _targets(entries).target_sha256
        == "c84b90b973ad7fd871051f4725d4916ed24a5a5cf25b23dfeb54626f2bbb40f5"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_rows", True),
        ("event_rows", 0),
        ("schema_version", "wrong"),
        ("scenario", "unsupported"),
        ("placement_latency_ms", True),
        ("placement_latency_ms", 750.0),
        ("exit_fee_bps", np.asarray([5.0])),
        ("scenario", np.asarray(["base"])),
        ("exit_fee_bps", 0.0),
        ("long_fill_sha256", None),
        ("short_fill_sha256", "z" * 64),
        ("batch_sha256", "short"),
        ("action_code", None),
        ("entry_price", np.ones(3)),
        ("order_start_time_ms", np.zeros(4, dtype=np.int32)),
    ],
)
def test_entry_metadata_shape_and_dtype_failures_reject(field, value):
    batch = build_make_take_scenario_entries(**_inputs())
    with pytest.raises(ValueError):
        validate_make_take_scenario_entry_batch(replace(batch, **{field: value}))


def test_non_entry_object_rejects_cleanly():
    with pytest.raises(ValueError):
        validate_make_take_scenario_entry_batch(object())


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("action_code", 0, 1),
        ("action_side", 0, -1),
        ("passive", 0, False),
        ("order_start_time_ms", 0, 0),
        ("order_start_time_ms", 0, np.iinfo(np.int64).max),
        ("order_start_time_ms", 1, 1800),
        ("entry_price", 0, 0.0),
        ("entry_price", 0, 101.0),
        ("entry_price", 3, 99.0),
        ("entry_price", 2, 102.0),
        ("entry_price", 0, np.nan),
        ("displayed_l1_participation", 0, 0.0),
        ("displayed_l1_participation", 0, 0.2),
        ("displayed_l1_participation", 1, 0.2),
        ("eligible", 0, False),
        ("entry_cost_bps", 0, 0.0),
        ("exit_cost_bps", 0, 0.0),
        ("filled", 2, False),
        ("entry_time_ms", 2, 1800),
        ("unfilled_expiry_time_ms", 2, 1800),
        ("fill_bucket", 2, 1),
        ("unfilled_expiry_time_ms", 0, 1800),
        ("entry_time_ms", 1, 1800),
        ("fill_bucket", 1, 1),
        ("entry_time_ms", 0, 1750),
        ("entry_time_ms", 0, 17000),
        ("fill_bucket", 0, 2),
    ],
)
def test_semantically_invalid_entries_reject_even_after_rehash(field, index, value):
    batch = build_make_take_scenario_entries(**_inputs())
    changed = getattr(batch, field).copy()
    changed[index] = value
    batch = replace(batch, **{field: changed})
    batch = replace(batch, batch_sha256=_sha256(_scenario_entry_payload(batch)))
    with pytest.raises(ValueError):
        validate_make_take_scenario_entry_batch(batch)


@pytest.mark.parametrize("consumer", ["survival", "payoff"])
def test_model_panels_reject_changed_entries_before_feature_access(consumer):
    entries = build_make_take_scenario_entries(**_inputs())
    changed = replace(entries, entry_cost_bps=np.zeros(4))
    with pytest.raises(ValueError):
        if consumer == "survival":
            build_passive_fill_survival_panel(None, changed, symbol="BTCUSDT")
        else:
            build_make_take_conditional_payoff_panel(
                symbol="BTCUSDT",
                action_features=None,
                entries=changed,
                targets=_targets(entries),
            )


def test_payoff_panel_rejects_changed_target_before_feature_access():
    entries = build_make_take_scenario_entries(**_inputs())
    targets = _targets(entries)
    changed = replace(targets, target_sha256="b" * 64)
    with pytest.raises(ValueError):
        build_make_take_conditional_payoff_panel(
            symbol="BTCUSDT", action_features=None, entries=entries, targets=changed
        )


def test_repeated_entry_events_reject_even_with_a_matching_recomputed_hash():
    batch = build_make_take_scenario_entries(**_inputs())
    arrays = {
        item.name: np.tile(getattr(batch, item.name), 2)
        for item in fields(batch)
        if isinstance(getattr(batch, item.name), np.ndarray)
    }
    batch = replace(batch, event_rows=2, **arrays)
    batch = replace(batch, batch_sha256=_sha256(_scenario_entry_payload(batch)))
    with pytest.raises(ValueError):
        validate_make_take_scenario_entry_batch(batch)
