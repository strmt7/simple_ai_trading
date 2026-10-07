"""Validate supplied entry bytes and lifecycle, not venue origin or partial cash."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from .make_take_scenario_entries import (
    MAKE_TAKE_MAX_L1_PARTICIPATION,
    MAKE_TAKE_ORDER_NOTIONAL_QUOTE,
    MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION,
    MakeTakeScenarioEntryBatch,
    _scenario_contract,
    _scenario_entry_payload,
    _sha256,
)
from .queue_censored_actions import PASSIVE_FILL_BUCKETS_MS


def validate_make_take_scenario_entry_batch(batch: MakeTakeScenarioEntryBatch) -> None:
    """Reject altered configuration, arrays and lifecycle before price-path access."""
    if (
        not isinstance(batch, MakeTakeScenarioEntryBatch)
        or type(batch.event_rows) is not int
        or batch.event_rows <= 0
        or batch.schema_version != MAKE_TAKE_SCENARIO_ENTRY_SCHEMA_VERSION
        or not isinstance(batch.scenario, str)
    ):
        raise ValueError("make/take entry source contract is invalid")
    contract = _scenario_contract(batch.scenario)
    expected_config = contract | {
        "passive_expiry_ms": PASSIVE_FILL_BUCKETS_MS[-1],
        "order_notional_quote": MAKE_TAKE_ORDER_NOTIONAL_QUOTE,
        "max_l1_participation": MAKE_TAKE_MAX_L1_PARTICIPATION,
    }
    if any(
        isinstance(getattr(batch, name), (bool, np.bool_))
        or not isinstance(
            getattr(batch, name), Integral if isinstance(value, int) else Real
        )
        or getattr(batch, name) != value
        for name, value in expected_config.items()
    ):
        raise ValueError("make/take entry configuration differs")
    for name in ("long_fill_sha256", "short_fill_sha256", "batch_sha256"):
        value = getattr(batch, name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("make/take entry source hash is invalid")
    integer = {
        "action_code": np.uint8,
        "action_side": np.int8,
        "fill_bucket": np.uint8,
        "order_start_time_ms": np.int64,
        "entry_time_ms": np.int64,
        "unfilled_expiry_time_ms": np.int64,
    }
    boolean = ("passive", "eligible", "filled")
    floating = (
        "entry_price",
        "displayed_l1_participation",
        "entry_cost_bps",
        "exit_cost_bps",
    )
    dtypes = (
        integer
        | {name: np.bool_ for name in boolean}
        | {name: np.float64 for name in floating}
    )
    for name, dtype in dtypes.items():
        value = getattr(batch, name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (batch.event_rows * 4,)
            or value.dtype != np.dtype(dtype)
        ):
            raise ValueError("make/take entry vector contract is invalid")
    if any(not np.isfinite(getattr(batch, name)).all() for name in floating):
        raise ValueError("make/take entry numeric evidence is invalid")
    rows = batch.event_rows
    code = np.tile(np.arange(4, dtype=np.uint8), rows)
    side = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), rows)
    starts = batch.order_start_time_ms
    prices = batch.entry_price.reshape(rows, 4)
    participation = batch.displayed_l1_participation.reshape(rows, 4)
    passive = code < 2
    unfilled = passive & ~batch.filled
    executed = passive & batch.filled
    delay = batch.entry_time_ms[executed] - starts[executed]
    expected_bucket = np.searchsorted(PASSIVE_FILL_BUCKETS_MS, delay, side="left") + 1
    if (
        not np.array_equal(batch.action_code, code)
        or not np.array_equal(batch.action_side, side)
        or not np.array_equal(batch.passive, passive)
        or np.any(starts < batch.placement_latency_ms)
        or np.any(starts > np.iinfo(np.int64).max - batch.passive_expiry_ms)
        or not np.array_equal(starts, np.repeat(starts[::4], 4))
        or np.any(np.diff(starts[::4]) <= 0)
        or np.any(prices <= 0)
        or np.any(prices[:, 0] >= prices[:, 1])
        or not np.array_equal(prices[:, 0], prices[:, 3])
        or not np.array_equal(prices[:, 1], prices[:, 2])
        or np.any(participation <= 0)
        or not np.array_equal(participation[:, 0], participation[:, 3])
        or not np.array_equal(participation[:, 1], participation[:, 2])
        or not np.array_equal(
            batch.eligible,
            batch.displayed_l1_participation <= batch.max_l1_participation,
        )
        or not np.array_equal(
            batch.entry_cost_bps,
            np.where(
                passive, batch.passive_entry_fee_bps, batch.aggressive_entry_fee_bps
            )
            + batch.additional_slippage_bps_per_side,
        )
        or np.any(
            batch.exit_cost_bps
            != batch.exit_fee_bps + batch.additional_slippage_bps_per_side
        )
        or np.any(~batch.filled[~passive])
        or np.any(batch.entry_time_ms[~passive] != starts[~passive])
        or np.any(batch.unfilled_expiry_time_ms[~passive] != -1)
        or np.any(batch.fill_bucket[~passive] != 0)
        or np.any(
            batch.unfilled_expiry_time_ms[passive]
            != starts[passive] + batch.passive_expiry_ms
        )
        or np.any(batch.entry_time_ms[unfilled] != -1)
        or np.any(batch.fill_bucket[unfilled] != 0)
        or np.any(delay <= 0)
        or np.any(delay > batch.passive_expiry_ms)
        or not np.array_equal(batch.fill_bucket[executed], expected_bucket)
        or batch.batch_sha256 != _sha256(_scenario_entry_payload(batch))
    ):
        raise ValueError("make/take entry identity or lifecycle is invalid")
