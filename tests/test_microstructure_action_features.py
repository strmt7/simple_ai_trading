from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_features import (
    ACTION_CANONICALIZATION_SHA256,
    ACTION_CONDITIONAL_FEATURE_NAMES,
    ACTION_FEATURE_SCHEMA_VERSION,
    build_action_conditional_features,
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_features import MICROSTRUCTURE_FEATURE_NAMES


def test_action_features_are_event_major_symmetric_and_depth_aware() -> None:
    source = np.arange(
        2 * len(MICROSTRUCTURE_FEATURE_NAMES), dtype=np.float32
    ).reshape(2, -1)
    batch = build_action_conditional_features(source)

    assert batch.schema_version == ACTION_FEATURE_SCHEMA_VERSION
    assert batch.canonicalization_sha256 == ACTION_CANONICALIZATION_SHA256
    assert len(batch.canonicalization_sha256) == 64
    assert batch.event_rows == 2
    assert batch.features.shape == (4, len(MICROSTRUCTURE_FEATURE_NAMES))
    assert batch.feature_names == ACTION_CONDITIONAL_FEATURE_NAMES
    assert batch.action_side.tolist() == [1, -1, 1, -1]

    return_index = MICROSTRUCTURE_FEATURE_NAMES.index("return_60s_bps")
    bid_index = MICROSTRUCTURE_FEATURE_NAMES.index("log_bid_l1_depth_quote")
    ask_index = MICROSTRUCTURE_FEATURE_NAMES.index("log_ask_l1_depth_quote")
    spread_index = MICROSTRUCTURE_FEATURE_NAMES.index("spread_bps")
    assert batch.features[0, return_index] == source[0, return_index]
    assert batch.features[1, return_index] == -source[0, return_index]
    assert batch.features[0, bid_index] == source[0, bid_index]
    assert batch.features[0, ask_index] == source[0, ask_index]
    assert batch.features[1, bid_index] == source[0, ask_index]
    assert batch.features[1, ask_index] == source[0, bid_index]
    assert batch.features[1, spread_index] == source[0, spread_index]


def test_action_feature_directional_mirror_is_exactly_equivariant() -> None:
    rng = np.random.default_rng(32)
    source = rng.normal(size=(17, len(MICROSTRUCTURE_FEATURE_NAMES))).astype(
        np.float32
    )
    original = build_action_conditional_features(source)
    mirrored = build_action_conditional_features(
        mirror_microstructure_direction(source)
    )

    np.testing.assert_array_equal(original.features[0::2], mirrored.features[1::2])
    np.testing.assert_array_equal(original.features[1::2], mirrored.features[0::2])


def test_action_features_fail_closed_on_contract_or_numeric_drift() -> None:
    source = np.ones((2, len(MICROSTRUCTURE_FEATURE_NAMES)), dtype=np.float32)
    with pytest.raises(ValueError, match="source contract"):
        build_action_conditional_features(source[:, :-1], MICROSTRUCTURE_FEATURE_NAMES[:-1])
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="source matrix"):
        build_action_conditional_features(source)
