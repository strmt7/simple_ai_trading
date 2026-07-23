"""Frozen action-aligned feature views for the Round 73 shallow models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .impact_absorption_grid import (
    ROUND73_GRID_FEATURE_NAMES,
    ROUND73_GRID_FEATURE_NAMES_SHA256,
)


ROUND73_EVALUATION_CONTRACT_SHA256 = (
    "1ac9b5fabd78138caddb296497d5b4885187198f3214338c323c3ed2fe2d388a"
)
ROUND73_MODEL_FEATURE_CONTRACT_SHA256 = (
    "8f027cb2459baa0a864e31dc7ab26a287e07e7853e8e35cc73b474cbcdff1750"
)
ROUND73_MODEL_RAW_FEATURE_NAMES_SHA256 = ROUND73_GRID_FEATURE_NAMES_SHA256
ROUND73_ACTION_SIDES = ("long", "short")
ROUND73_MODEL_FEATURE_LAYERS = ("l1_tape", "l2_state", "impact_absorption")
ROUND73_MODEL_ANCHOR_FEATURE_NAMES = (
    "shock_ratio",
    "aligned_shock_direction",
    "shock_direction_taker_share",
    "action_side",
)

_WINDOW_PREFIX = re.compile(r"^w[0-9]+ms_(.+)$")
_L1_STATE_FEATURES = frozenset(
    {
        "spread_bps",
        "bid_quote_notional",
        "ask_quote_notional",
        "l1_imbalance",
        "microprice_offset_bps",
        "bbo_age_ms",
        "bbo_corrected_event_latency_ms",
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    }
)
_L2_STATE_FEATURES = frozenset(
    {
        "bid_depth_quote_5",
        "ask_depth_quote_5",
        "bid_depth_quote_10",
        "ask_depth_quote_10",
        "bid_depth_quote_20",
        "ask_depth_quote_20",
        "imbalance_5",
        "imbalance_10",
        "imbalance_20",
        "bid_depth_5_share_of_20",
        "ask_depth_5_share_of_20",
        "bid_distance_weighted_depth_20",
        "ask_distance_weighted_depth_20",
        "bid_depth_concentration_20",
        "ask_depth_concentration_20",
        "l2_age_ms",
        "l2_corrected_event_latency_ms",
    }
)
_TAPE_WINDOW_FEATURES = frozenset(
    {
        "buy_aggressive_quote",
        "sell_aggressive_quote",
        "signed_aggressive_quote",
        "absolute_aggressive_quote",
        "aggregate_trade_count",
        "buyer_taker_share",
        "mid_log_return",
        "mid_realized_variance",
        "bbo_update_count",
        "mean_spread_bps",
        "maximum_spread_bps",
    }
)
_EXACT_DIRECTIONAL_FEATURES = frozenset(
    {
        "l1_imbalance",
        "microprice_offset_bps",
        "imbalance_5",
        "imbalance_10",
        "imbalance_20",
        "mark_to_mid_bps",
        "index_to_mid_bps",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _replace_token(name: str, old: str, new: str) -> str:
    tokens = name.split("_")
    try:
        index = tokens.index(old)
    except ValueError as exc:
        raise ValueError(f"Round 73 feature token is missing: {old}") from exc
    tokens[index] = new
    return "_".join(tokens)


def _paired_name(name: str) -> tuple[str, str] | None:
    tokens = name.split("_")
    if "bid" in tokens:
        return "book", "support"
    if "ask" in tokens:
        return "book", "opposing"
    if "buy" in tokens:
        return "trade", "aligned"
    if "sell" in tokens:
        return "trade", "opposing"
    return None


def _counterpart_name(name: str, group: str) -> str:
    if group == "book":
        if "bid" in name.split("_"):
            return _replace_token(name, "bid", "ask")
        return _replace_token(name, "ask", "bid")
    if "buy" in name.split("_"):
        return _replace_token(name, "buy", "sell")
    return _replace_token(name, "sell", "buy")


def _action_name(name: str) -> str:
    paired = _paired_name(name)
    if paired is not None:
        group, output_token = paired
        input_token = (
            "bid"
            if group == "book" and output_token == "support"
            else "ask"
            if group == "book"
            else "buy"
            if output_token == "aligned"
            else "sell"
        )
        return _replace_token(name, input_token, output_token)
    if name.endswith("buyer_taker_share"):
        return name.removesuffix("buyer_taker_share") + "aligned_taker_share"
    if _is_directional(name):
        return "aligned_" + name
    return name


def _is_directional(name: str) -> bool:
    return bool(
        name in _EXACT_DIRECTIONAL_FEATURES
        or name.endswith("_signed_aggressive_quote")
        or "_normalized_order_flow_imbalance_" in name
        or name.endswith("_mid_log_return")
    )


def _raw_layer(name: str) -> str:
    if name in _L1_STATE_FEATURES:
        return "l1_tape"
    if name in _L2_STATE_FEATURES:
        return "l2_state"
    match = _WINDOW_PREFIX.fullmatch(name)
    if match is not None and match.group(1) in _TAPE_WINDOW_FEATURES:
        return "l1_tape"
    return "impact_absorption"


_RAW_NAME_TO_INDEX = MappingProxyType(
    {name: index for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES)}
)
if len(_RAW_NAME_TO_INDEX) != len(ROUND73_GRID_FEATURE_NAMES):
    raise RuntimeError("Round 73 grid feature names are not unique")

ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES = tuple(
    _action_name(name) for name in ROUND73_GRID_FEATURE_NAMES
)
if len(set(ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES)) != len(
    ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES
):
    raise RuntimeError("Round 73 action-aligned feature names are not unique")

ROUND73_ACTION_ALIGNED_FEATURE_NAMES = (
    ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES + ROUND73_MODEL_ANCHOR_FEATURE_NAMES
)
ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256 = _sha256(
    list(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)
)


def _layer_names(layer: str) -> tuple[str, ...]:
    selected = str(layer)
    if selected not in ROUND73_MODEL_FEATURE_LAYERS:
        raise ValueError(f"unknown Round 73 model feature layer: {layer!r}")
    maximum = ROUND73_MODEL_FEATURE_LAYERS.index(selected)
    grid_names = tuple(
        transformed
        for raw, transformed in zip(
            ROUND73_GRID_FEATURE_NAMES,
            ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES,
            strict=True,
        )
        if ROUND73_MODEL_FEATURE_LAYERS.index(_raw_layer(raw)) <= maximum
    )
    return grid_names + ROUND73_MODEL_ANCHOR_FEATURE_NAMES


ROUND73_MODEL_FEATURE_NAMES_BY_LAYER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {layer: _layer_names(layer) for layer in ROUND73_MODEL_FEATURE_LAYERS}
)
ROUND73_MODEL_FEATURE_SHA256_BY_LAYER: Mapping[str, str] = MappingProxyType(
    {
        layer: _sha256(list(names))
        for layer, names in ROUND73_MODEL_FEATURE_NAMES_BY_LAYER.items()
    }
)


def action_align_round73_features(
    feature_values: np.ndarray,
    *,
    side: str,
    shock_ratio: float,
    shock_direction: int,
    shock_direction_taker_share: float,
) -> np.ndarray:
    """Create one causal action view without fitting or target access."""

    values = np.asarray(feature_values, dtype=np.float64)
    selected_side = str(side).strip().lower()
    ratio = float(shock_ratio)
    direction = int(shock_direction)
    direction_share = float(shock_direction_taker_share)
    if (
        values.ndim != 1
        or values.shape != (len(ROUND73_GRID_FEATURE_NAMES),)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Round 73 raw model feature vector is invalid")
    if selected_side not in ROUND73_ACTION_SIDES:
        raise ValueError("Round 73 action side must be long or short")
    if (
        not math.isfinite(ratio)
        or ratio < 0.0
        or direction not in {-1, 1}
        or not math.isfinite(direction_share)
        or not 0.0 <= direction_share <= 1.0
    ):
        raise ValueError("Round 73 action anchor metadata is invalid")
    action_sign = 1.0 if selected_side == "long" else -1.0
    output = np.empty(len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES), dtype=np.float64)
    for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES):
        paired = _paired_name(name)
        if paired is not None:
            group, output_token = paired
            source_name = name
            if selected_side == "short":
                source_name = _counterpart_name(name, group)
            output[index] = values[_RAW_NAME_TO_INDEX[source_name]]
        elif name.endswith("buyer_taker_share"):
            share = values[index]
            if not 0.0 <= share <= 1.0:
                raise ValueError("Round 73 buyer-taker share is outside [0, 1]")
            output[index] = share if selected_side == "long" else 1.0 - share
        elif _is_directional(name):
            output[index] = values[index] * action_sign
        else:
            output[index] = values[index]
    output[-4:] = (
        ratio,
        float(direction) * action_sign,
        direction_share,
        action_sign,
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("Round 73 action-aligned feature vector is nonfinite")
    output.setflags(write=False)
    return output


def action_align_round73_feature_batch(
    feature_values: np.ndarray,
    *,
    side: np.ndarray,
    shock_ratio: np.ndarray,
    shock_direction: np.ndarray,
    shock_direction_taker_share: np.ndarray,
    dtype: object = np.float32,
) -> np.ndarray:
    """Vectorize the exact scalar action transform for bounded model batches."""

    output_dtype = np.dtype(dtype)
    if output_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(
            "Round 73 action feature batch dtype must be float32 or float64"
        )
    # Preserve the scalar contract's float64 operation order before the single
    # requested output cast. Casting inputs first changes complement rounding.
    values = np.asarray(feature_values, dtype=np.float64)
    sides = np.asarray(side, dtype=np.int8)
    ratios = np.asarray(shock_ratio, dtype=np.float64)
    directions = np.asarray(shock_direction, dtype=np.int8)
    direction_shares = np.asarray(
        shock_direction_taker_share,
        dtype=np.float64,
    )
    rows = len(values)
    taker_share_indexes = tuple(
        index
        for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES)
        if name.endswith("buyer_taker_share")
    )
    if (
        values.shape != (rows, len(ROUND73_GRID_FEATURE_NAMES))
        or sides.shape != (rows,)
        or ratios.shape != (rows,)
        or directions.shape != (rows,)
        or direction_shares.shape != (rows,)
        or np.any((sides != 1) & (sides != -1))
        or np.any((directions != 1) & (directions != -1))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(ratios))
        or np.any(ratios < 0.0)
        or not np.all(np.isfinite(direction_shares))
        or np.any((direction_shares < 0.0) | (direction_shares > 1.0))
        or np.any(
            (values[:, taker_share_indexes] < 0.0)
            | (values[:, taker_share_indexes] > 1.0)
        )
    ):
        raise ValueError("Round 73 action feature batch is invalid")

    output = np.empty(
        (rows, len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)),
        dtype=np.float64,
    )
    short_rows = sides == -1
    for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES):
        paired = _paired_name(name)
        if paired is not None:
            counterpart = _counterpart_name(name, paired[0])
            output[:, index] = values[:, index]
            output[short_rows, index] = values[
                short_rows, _RAW_NAME_TO_INDEX[counterpart]
            ]
        elif name.endswith("buyer_taker_share"):
            output[:, index] = values[:, index]
            output[short_rows, index] = 1.0 - values[short_rows, index]
        elif _is_directional(name):
            output[:, index] = values[:, index] * sides
        else:
            output[:, index] = values[:, index]
    output[:, -4] = ratios
    output[:, -3] = directions * sides
    output[:, -2] = direction_shares
    output[:, -1] = sides
    if not np.all(np.isfinite(output)):
        raise ValueError("Round 73 action-aligned feature batch is nonfinite")
    result = np.ascontiguousarray(output, dtype=output_dtype)
    result.setflags(write=False)
    return result


def select_round73_feature_layer(
    action_aligned_values: np.ndarray,
    *,
    layer: str,
) -> np.ndarray:
    """Project an action-aligned vector onto one frozen nested layer."""

    values = np.asarray(action_aligned_values, dtype=np.float64)
    names = ROUND73_MODEL_FEATURE_NAMES_BY_LAYER.get(str(layer))
    if (
        names is None
        or values.shape != (len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES),)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Round 73 action feature layer input is invalid")
    index_by_name = {
        name: index for index, name in enumerate(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)
    }
    output = np.ascontiguousarray(
        values[[index_by_name[name] for name in names]],
        dtype=np.float64,
    )
    output.setflags(write=False)
    return output


__all__ = [
    "ROUND73_ACTION_ALIGNED_FEATURE_NAMES",
    "ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256",
    "ROUND73_ACTION_ALIGNED_GRID_FEATURE_NAMES",
    "ROUND73_ACTION_SIDES",
    "ROUND73_EVALUATION_CONTRACT_SHA256",
    "ROUND73_MODEL_ANCHOR_FEATURE_NAMES",
    "ROUND73_MODEL_FEATURE_CONTRACT_SHA256",
    "ROUND73_MODEL_FEATURE_LAYERS",
    "ROUND73_MODEL_FEATURE_NAMES_BY_LAYER",
    "ROUND73_MODEL_RAW_FEATURE_NAMES_SHA256",
    "ROUND73_MODEL_FEATURE_SHA256_BY_LAYER",
    "action_align_round73_feature_batch",
    "action_align_round73_features",
    "select_round73_feature_layer",
]
