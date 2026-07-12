"""Direction-canonical long/short action features for shared models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .microstructure_features import MICROSTRUCTURE_FEATURE_NAMES


ACTION_FEATURE_SCHEMA_VERSION = "direction-canonical-action-features-v1"
_SIGNED_FEATURES = frozenset(
    {
        "return_1s_bps",
        "return_5s_bps",
        "return_15s_bps",
        "return_30s_bps",
        "return_60s_bps",
        "return_120s_bps",
        "return_300s_bps",
        "return_900s_bps",
        "l1_imbalance",
        "close_l1_imbalance",
        "imbalance_10s_mean",
        "imbalance_60s_mean",
        "imbalance_300s_mean",
        "microprice_offset_bps",
        "normalized_ofi",
        "ofi_10s_mean",
        "ofi_60s_mean",
        "ofi_300s_mean",
        "ofi_delta_5s",
        "ofi_delta_15s",
        "ofi_delta_30s",
        "ofi_delta_60s",
        "trade_imbalance",
        "trade_imbalance_10s_mean",
        "trade_imbalance_60s_mean",
        "trade_imbalance_300s_mean",
        "trade_imbalance_delta_5s",
        "trade_imbalance_delta_15s",
        "trade_imbalance_delta_30s",
        "trade_imbalance_delta_60s",
        "signed_flow_10s",
        "signed_flow_60s",
        "signed_flow_300s",
        "trade_close_vs_mid_bps",
        "l1_imbalance_delta_5s",
        "l1_imbalance_delta_15s",
        "l1_imbalance_delta_30s",
        "l1_imbalance_delta_60s",
        "microprice_delta_5s_bps",
        "microprice_delta_15s_bps",
        "microprice_delta_30s_bps",
        "microprice_delta_60s_bps",
        "return_60s_vol_units",
        "return_300s_vol_units",
        "return_900s_vol_units",
        "intrasecond_close_location",
        "return_1800s_bps",
        "return_3600s_bps",
        "return_1800s_vol_units",
        "return_3600s_vol_units",
        "signed_pressure_to_opposing_depth_10s",
        "signed_pressure_to_opposing_depth_60s",
        "signed_pressure_to_opposing_depth_300s",
    }
)
_BID_DEPTH_FEATURE = "log_bid_l1_depth_quote"
_ASK_DEPTH_FEATURE = "log_ask_l1_depth_quote"
_SUPPORTING_DEPTH_FEATURE = "log_supporting_l1_depth_quote"
_OPPOSING_DEPTH_FEATURE = "log_opposing_l1_depth_quote"


def _canonical_feature_names() -> tuple[str, ...]:
    output: list[str] = []
    for name in MICROSTRUCTURE_FEATURE_NAMES:
        if name in _SIGNED_FEATURES:
            output.append(f"action_aligned_{name}")
        elif name == _BID_DEPTH_FEATURE:
            output.append(_SUPPORTING_DEPTH_FEATURE)
        elif name == _ASK_DEPTH_FEATURE:
            output.append(_OPPOSING_DEPTH_FEATURE)
        else:
            output.append(name)
    return tuple(output)


ACTION_CONDITIONAL_FEATURE_NAMES = _canonical_feature_names()


def _canonicalization_sha256() -> str:
    payload = {
        "schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "source_feature_names": MICROSTRUCTURE_FEATURE_NAMES,
        "action_feature_names": ACTION_CONDITIONAL_FEATURE_NAMES,
        "signed_features": sorted(_SIGNED_FEATURES),
        "depth_mapping": {
            "long": {
                "supporting": _BID_DEPTH_FEATURE,
                "opposing": _ASK_DEPTH_FEATURE,
            },
            "short": {
                "supporting": _ASK_DEPTH_FEATURE,
                "opposing": _BID_DEPTH_FEATURE,
            },
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


ACTION_CANONICALIZATION_SHA256 = _canonicalization_sha256()


@dataclass(frozen=True)
class ActionConditionalFeatureBatch:
    """Paired event-major action rows: long then short for every event."""

    schema_version: str
    canonicalization_sha256: str
    feature_names: tuple[str, ...]
    action_side: np.ndarray
    features: np.ndarray

    @property
    def event_rows(self) -> int:
        return len(self.action_side) // 2


def _validate_source_features(
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    names = tuple(str(value) for value in feature_names)
    values = np.asarray(features, dtype=np.float32)
    if names != MICROSTRUCTURE_FEATURE_NAMES:
        raise ValueError("action feature source contract is unsupported")
    if not _SIGNED_FEATURES < set(names):
        raise RuntimeError("action feature signed contract is incomplete")
    if (
        values.ndim != 2
        or values.shape[0] <= 0
        or values.shape[1] != len(names)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("action feature source matrix is invalid")
    return values


def build_action_conditional_features(
    features: np.ndarray,
    feature_names: Sequence[str] = MICROSTRUCTURE_FEATURE_NAMES,
) -> ActionConditionalFeatureBatch:
    """Map each event to parameter-shared long and short action rows."""

    values = _validate_source_features(features, feature_names)
    rows, columns = values.shape
    paired = np.empty((rows * 2, columns), dtype=np.float32)
    paired[0::2] = values
    paired[1::2] = values
    names = tuple(str(value) for value in feature_names)
    signed_indexes = np.asarray(
        [index for index, name in enumerate(names) if name in _SIGNED_FEATURES],
        dtype=np.int64,
    )
    paired[1::2, signed_indexes] *= -1.0
    bid_index = names.index(_BID_DEPTH_FEATURE)
    ask_index = names.index(_ASK_DEPTH_FEATURE)
    paired[1::2, bid_index] = values[:, ask_index]
    paired[1::2, ask_index] = values[:, bid_index]
    action_side = np.tile(np.asarray([1, -1], dtype=np.int8), rows)
    return ActionConditionalFeatureBatch(
        schema_version=ACTION_FEATURE_SCHEMA_VERSION,
        canonicalization_sha256=ACTION_CANONICALIZATION_SHA256,
        feature_names=ACTION_CONDITIONAL_FEATURE_NAMES,
        action_side=action_side,
        features=paired,
    )


def mirror_microstructure_direction(
    features: np.ndarray,
    feature_names: Sequence[str] = MICROSTRUCTURE_FEATURE_NAMES,
) -> np.ndarray:
    """Construct the exact directional mirror used by equivariance tests."""

    values = _validate_source_features(features, feature_names)
    mirrored = np.array(values, dtype=np.float32, copy=True)
    names = tuple(str(value) for value in feature_names)
    signed_indexes = np.asarray(
        [index for index, name in enumerate(names) if name in _SIGNED_FEATURES],
        dtype=np.int64,
    )
    mirrored[:, signed_indexes] *= -1.0
    bid_index = names.index(_BID_DEPTH_FEATURE)
    ask_index = names.index(_ASK_DEPTH_FEATURE)
    mirrored[:, bid_index] = values[:, ask_index]
    mirrored[:, ask_index] = values[:, bid_index]
    return mirrored


__all__ = [
    "ACTION_CANONICALIZATION_SHA256",
    "ACTION_CONDITIONAL_FEATURE_NAMES",
    "ACTION_FEATURE_SCHEMA_VERSION",
    "ActionConditionalFeatureBatch",
    "build_action_conditional_features",
    "mirror_microstructure_direction",
]
