from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_grid import ROUND73_GRID_FEATURE_NAMES
from simple_ai_trading.impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_MODEL_FEATURE_LAYERS,
    ROUND73_MODEL_FEATURE_NAMES_BY_LAYER,
    action_align_round73_features,
    select_round73_feature_layer,
)


def _raw_values() -> np.ndarray:
    values = np.arange(1, len(ROUND73_GRID_FEATURE_NAMES) + 1, dtype=np.float64)
    for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES):
        if name.endswith("buyer_taker_share"):
            values[index] = 0.8
    return values


def _index(names: tuple[str, ...], name: str) -> int:
    return names.index(name)


def test_round73_action_alignment_is_symmetric_causal_and_read_only() -> None:
    raw = _raw_values()
    before = raw.copy()
    common = {
        "shock_ratio": 5.0,
        "shock_direction": 1,
        "shock_direction_taker_share": 0.8,
    }
    long = action_align_round73_features(raw, side="long", **common)
    short = action_align_round73_features(raw, side="short", **common)

    assert np.array_equal(raw, before)
    assert not long.flags.writeable
    assert not short.flags.writeable
    assert long.shape == short.shape == (len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES),)
    raw_bid = _index(ROUND73_GRID_FEATURE_NAMES, "bid_quote_notional")
    raw_ask = _index(ROUND73_GRID_FEATURE_NAMES, "ask_quote_notional")
    support = _index(ROUND73_ACTION_ALIGNED_FEATURE_NAMES, "support_quote_notional")
    opposing = _index(ROUND73_ACTION_ALIGNED_FEATURE_NAMES, "opposing_quote_notional")
    assert long[support] == raw[raw_bid]
    assert long[opposing] == raw[raw_ask]
    assert short[support] == raw[raw_ask]
    assert short[opposing] == raw[raw_bid]
    imbalance = _index(ROUND73_ACTION_ALIGNED_FEATURE_NAMES, "aligned_l1_imbalance")
    raw_imbalance = _index(ROUND73_GRID_FEATURE_NAMES, "l1_imbalance")
    assert long[imbalance] == raw[raw_imbalance]
    assert short[imbalance] == -raw[raw_imbalance]
    taker = _index(
        ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
        "w100ms_aligned_taker_share",
    )
    assert long[taker] == pytest.approx(0.8)
    assert short[taker] == pytest.approx(0.2)
    aligned_trade = _index(
        ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
        "w100ms_aligned_aggressive_quote",
    )
    raw_buy = _index(
        ROUND73_GRID_FEATURE_NAMES,
        "w100ms_buy_aggressive_quote",
    )
    raw_sell = _index(
        ROUND73_GRID_FEATURE_NAMES,
        "w100ms_sell_aggressive_quote",
    )
    assert long[aligned_trade] == raw[raw_buy]
    assert short[aligned_trade] == raw[raw_sell]
    assert tuple(long[-4:]) == (5.0, 1.0, 0.8, 1.0)
    assert tuple(short[-4:]) == (5.0, -1.0, 0.8, -1.0)


def test_round73_model_feature_layers_are_nested_and_complete() -> None:
    names = ROUND73_MODEL_FEATURE_NAMES_BY_LAYER
    assert tuple(names) == ROUND73_MODEL_FEATURE_LAYERS
    assert set(names["l1_tape"]) < set(names["l2_state"])
    assert set(names["l2_state"]) < set(names["impact_absorption"])
    assert names["impact_absorption"] == ROUND73_ACTION_ALIGNED_FEATURE_NAMES
    raw = _raw_values()
    aligned = action_align_round73_features(
        raw,
        side="long",
        shock_ratio=4.0,
        shock_direction=-1,
        shock_direction_taker_share=0.75,
    )
    for layer in ROUND73_MODEL_FEATURE_LAYERS:
        projected = select_round73_feature_layer(aligned, layer=layer)
        assert projected.shape == (len(names[layer]),)
        assert not projected.flags.writeable


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda values: values[:-1], "raw model feature vector"),
        (
            lambda values: np.where(
                np.arange(values.size) == 0,
                np.nan,
                values,
            ),
            "raw model feature vector",
        ),
    ],
)
def test_round73_action_alignment_rejects_bad_vectors(mutation, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        action_align_round73_features(
            mutation(_raw_values()),
            side="long",
            shock_ratio=4.0,
            shock_direction=1,
            shock_direction_taker_share=0.8,
        )


def test_round73_action_alignment_rejects_bad_action_metadata() -> None:
    raw = _raw_values()
    with pytest.raises(ValueError, match="side"):
        action_align_round73_features(
            raw,
            side="flat",
            shock_ratio=4.0,
            shock_direction=1,
            shock_direction_taker_share=0.8,
        )
    with pytest.raises(ValueError, match="metadata"):
        action_align_round73_features(
            raw,
            side="long",
            shock_ratio=4.0,
            shock_direction=0,
            shock_direction_taker_share=0.8,
        )
    bad_share = raw.copy()
    bad_share[
        _index(ROUND73_GRID_FEATURE_NAMES, "w100ms_buyer_taker_share")
    ] = 1.1
    with pytest.raises(ValueError, match="buyer-taker share"):
        action_align_round73_features(
            bad_share,
            side="long",
            shock_ratio=4.0,
            shock_direction=1,
            shock_direction_taker_share=0.8,
        )
