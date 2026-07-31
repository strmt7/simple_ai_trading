"""Matched residual-probability candidates for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256


POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION = (
    "polymarket-round21-matched-residual-development-v3"
)
POLYMARKET_ROUND21_PROBABILITY_BATCH_SCHEMA_VERSION = (
    "polymarket-round21-probability-batch-v1"
)
POLYMARKET_ROUND21_MODEL_SEED = 21_021
POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS = 30
POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND21_DATASET_DESIGN_SHA256 = (
    "089f046fd611e32950381ec4f33a1e6b54a0a0a4d6be161de0643ade94590eba"
)
POLYMARKET_ROUND21_MODEL_DESIGN_SHA256 = (
    "6dd59a429c1013ef737086da9fbd9a59cb93bcbb616fb7a97cc0ddf4dbb1e7be"
)
POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256 = (
    "f502c28b7a152e8eb6e3d49e2a23c145e2a90118040e79c17a7adfb0e543857c"
)
_ROLES = ("train", "tune_calibration", "tune_selection", "test")
_LAYERS = ("core", "core_spot", "core_spot_usdm")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_PROBABILITY_FLOOR = 1e-6
_CONDITION_DURATION_MS = 300_000
_TRAIN_TO_TUNE_MINIMUM_START_GAP_MS = 2_100_000
_LOGISTIC_L2 = (0.01, 0.1, 1.0)
_LIGHTGBM_GRID = (
    {"max_depth": 2, "num_leaves": 3, "min_data_in_leaf": 20},
    {"max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 40},
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _array_sha256(value: np.ndarray, *, dtype: str) -> str:
    selected = np.asarray(value, dtype=dtype, order="C")
    return hashlib.sha256(selected.tobytes(order="C")).hexdigest()


def _probability(values: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)
    if selected.ndim != 1 or not np.all(np.isfinite(selected)):
        raise ValueError("Round 21 probability vector is invalid")
    return np.clip(selected, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)


def _float_matrix(value: object, *, rows: int) -> np.ndarray:
    selected = np.asarray(value)
    if (
        selected.ndim != 2
        or selected.shape[0] != rows
        or selected.shape[1] < 1
        or selected.dtype.kind != "f"
        or selected.dtype.itemsize not in (4, 8)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 21 feature matrix is invalid")
    return np.asarray(selected, dtype=np.float32, order="C")


def _condition_groups(
    condition_ids: np.ndarray,
) -> tuple[tuple[str, int, int], ...]:
    if condition_ids.ndim != 1 or len(condition_ids) < 1:
        raise ValueError("Round 21 condition identities are invalid")
    boundaries = np.flatnonzero(condition_ids[1:] != condition_ids[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(condition_ids)])))
    groups = tuple(
        (str(condition_ids[start]), int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    )
    if len({condition for condition, _start, _end in groups}) != len(groups):
        raise ValueError("Round 21 condition identities are not contiguous")
    return groups


@dataclass(frozen=True, slots=True)
class Round21DevelopmentPanel:
    role: str
    condition_ids: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    labels: np.ndarray
    structural_probability: np.ndarray
    market_prior_probability: np.ndarray
    core_features: np.ndarray
    spot_features: np.ndarray
    usdm_features: np.ndarray
    spot_available: np.ndarray
    usdm_available: np.ndarray
    core_feature_names_sha256: str
    spot_feature_names_sha256: str
    usdm_feature_names_sha256: str
    dataset_sha256: str
    target_manifest_sha256: str
    dataset_design_sha256: str

    def validate(self) -> Round21DevelopmentPanel:
        role = str(self.role or "").strip()
        condition_ids = np.asarray(self.condition_ids, dtype=object)
        event_starts = np.asarray(self.event_start_ms, dtype=np.int64)
        decisions = np.asarray(self.decision_time_ms, dtype=np.int64)
        labels = np.asarray(self.labels)
        structural = np.asarray(self.structural_probability)
        market_prior = np.asarray(self.market_prior_probability)
        rows = len(condition_ids)
        spot_available = np.asarray(self.spot_available)
        usdm_available = np.asarray(self.usdm_available)
        if (
            role not in _ROLES
            or rows < 1
            or event_starts.shape != (rows,)
            or decisions.shape != (rows,)
            or labels.shape != (rows,)
            or labels.dtype.kind != "f"
            or structural.shape != (rows,)
            or structural.dtype.kind != "f"
            or market_prior.shape != (rows,)
            or market_prior.dtype.kind != "f"
            or spot_available.shape != (rows,)
            or spot_available.dtype.kind != "b"
            or usdm_available.shape != (rows,)
            or usdm_available.dtype.kind != "b"
            or any(
                _SHA256.fullmatch(str(value or "").strip()) is None
                or str(value).strip() == _EMPTY_SHA256
                for value in (
                    self.core_feature_names_sha256,
                    self.spot_feature_names_sha256,
                    self.usdm_feature_names_sha256,
                    self.dataset_sha256,
                    self.target_manifest_sha256,
                    self.dataset_design_sha256,
                )
            )
            or self.dataset_design_sha256 != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        ):
            raise ValueError("Round 21 development panel is invalid")
        core = _float_matrix(self.core_features, rows=rows)
        spot = _float_matrix(self.spot_features, rows=rows)
        usdm = _float_matrix(self.usdm_features, rows=rows)
        labels64 = np.asarray(labels, dtype=np.float64)
        structural64 = _probability(np.asarray(structural, dtype=np.float64))
        market_prior64 = _probability(np.asarray(market_prior, dtype=np.float64))
        if (
            not np.all(np.isin(labels64, (0.0, 1.0)))
            or set(labels64.tolist()) != {0.0, 1.0}
            or np.any(np.asarray(structural, dtype=np.float64) <= 0.0)
            or np.any(np.asarray(structural, dtype=np.float64) >= 1.0)
            or np.any(np.asarray(market_prior, dtype=np.float64) <= 0.0)
            or np.any(np.asarray(market_prior, dtype=np.float64) >= 1.0)
            or np.any(event_starts <= 0)
            or np.any(event_starts % 300_000)
            or np.any(decisions < event_starts)
            or np.any(decisions >= event_starts + 300_000)
            or np.any((decisions - event_starts) % 250)
            or np.any(event_starts[1:] < event_starts[:-1])
            or np.any(
                (event_starts[1:] == event_starts[:-1])
                & (decisions[1:] < decisions[:-1])
            )
            or np.any(spot[~spot_available] != 0.0)
            or np.any(usdm[~usdm_available] != 0.0)
            or np.any(usdm_available & ~spot_available)
        ):
            raise ValueError("Round 21 development panel is invalid")
        groups = _condition_groups(condition_ids)
        for condition, start, end in groups:
            if (
                _CONDITION_ID.fullmatch(condition) is None
                or event_starts[start] != event_starts[end - 1]
                or not np.all(labels64[start:end] == labels64[start])
                or np.any(decisions[start + 1 : end] <= decisions[start : end - 1])
            ):
                raise ValueError("Round 21 condition target identity differs")
        group_event_starts = np.asarray(
            [event_starts[start] for _condition, start, _end in groups],
            dtype=np.int64,
        )
        if np.any(group_event_starts[1:] <= group_event_starts[:-1]):
            raise ValueError("Round 21 condition target identity differs")
        return replace(
            self,
            role=role,
            condition_ids=condition_ids,
            event_start_ms=event_starts,
            decision_time_ms=decisions,
            labels=labels64,
            structural_probability=structural64,
            market_prior_probability=market_prior64,
            core_features=core,
            spot_features=spot,
            usdm_features=usdm,
            spot_available=np.asarray(spot_available, dtype=np.bool_),
            usdm_available=np.asarray(usdm_available, dtype=np.bool_),
        )


@dataclass(frozen=True, slots=True)
class Round21InferencePanel:
    condition_ids: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    structural_probability: np.ndarray
    market_prior_probability: np.ndarray
    core_features: np.ndarray
    spot_features: np.ndarray
    usdm_features: np.ndarray
    spot_available: np.ndarray
    usdm_available: np.ndarray
    core_feature_names_sha256: str
    spot_feature_names_sha256: str
    usdm_feature_names_sha256: str
    source_dataset_sha256: str
    feature_batch_sha256: str
    dataset_design_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def from_development(
        cls,
        panel: Round21DevelopmentPanel,
    ) -> Round21InferencePanel:
        selected = panel.validate()
        return cls.create(
            condition_ids=selected.condition_ids,
            event_start_ms=selected.event_start_ms,
            decision_time_ms=selected.decision_time_ms,
            structural_probability=selected.structural_probability,
            market_prior_probability=selected.market_prior_probability,
            core_features=selected.core_features,
            spot_features=selected.spot_features,
            usdm_features=selected.usdm_features,
            spot_available=selected.spot_available,
            usdm_available=selected.usdm_available,
            core_feature_names_sha256=selected.core_feature_names_sha256,
            spot_feature_names_sha256=selected.spot_feature_names_sha256,
            usdm_feature_names_sha256=selected.usdm_feature_names_sha256,
            source_dataset_sha256=selected.dataset_sha256,
            dataset_design_sha256=selected.dataset_design_sha256,
        )

    @classmethod
    def create(
        cls,
        *,
        condition_ids: np.ndarray,
        event_start_ms: np.ndarray,
        decision_time_ms: np.ndarray,
        structural_probability: np.ndarray,
        market_prior_probability: np.ndarray,
        core_features: np.ndarray,
        spot_features: np.ndarray,
        usdm_features: np.ndarray,
        spot_available: np.ndarray,
        usdm_available: np.ndarray,
        core_feature_names_sha256: str,
        spot_feature_names_sha256: str,
        usdm_feature_names_sha256: str,
        source_dataset_sha256: str,
        dataset_design_sha256: str,
    ) -> Round21InferencePanel:
        normalized_arrays = (
            np.array(condition_ids, dtype=str, order="C", copy=True),
            np.array(event_start_ms, dtype="<i8", order="C", copy=True),
            np.array(decision_time_ms, dtype="<i8", order="C", copy=True),
            np.array(structural_probability, dtype="<f8", order="C", copy=True),
            np.array(market_prior_probability, dtype="<f8", order="C", copy=True),
            np.array(core_features, dtype="<f4", order="C", copy=True),
            np.array(spot_features, dtype="<f4", order="C", copy=True),
            np.array(usdm_features, dtype="<f4", order="C", copy=True),
            np.array(spot_available, dtype=np.bool_, order="C", copy=True),
            np.array(usdm_available, dtype=np.bool_, order="C", copy=True),
        )
        for array in normalized_arrays:
            array.setflags(write=False)
        provisional = cls(
            condition_ids=normalized_arrays[0],
            event_start_ms=normalized_arrays[1],
            decision_time_ms=normalized_arrays[2],
            structural_probability=normalized_arrays[3],
            market_prior_probability=normalized_arrays[4],
            core_features=normalized_arrays[5],
            spot_features=normalized_arrays[6],
            usdm_features=normalized_arrays[7],
            spot_available=normalized_arrays[8],
            usdm_available=normalized_arrays[9],
            core_feature_names_sha256=str(core_feature_names_sha256).strip().lower(),
            spot_feature_names_sha256=str(spot_feature_names_sha256).strip().lower(),
            usdm_feature_names_sha256=str(usdm_feature_names_sha256).strip().lower(),
            source_dataset_sha256=str(source_dataset_sha256).strip().lower(),
            feature_batch_sha256=_EMPTY_SHA256,
            dataset_design_sha256=str(dataset_design_sha256).strip().lower(),
        )
        return replace(
            provisional,
            feature_batch_sha256=provisional._computed_feature_batch_sha256(),
        ).validate()

    def _computed_feature_batch_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION,
                "model_design_sha256": POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
                "condition_ids": np.asarray(
                    self.condition_ids,
                    dtype=object,
                ).tolist(),
                "event_start_ms_sha256": _array_sha256(
                    self.event_start_ms,
                    dtype="<i8",
                ),
                "decision_time_ms_sha256": _array_sha256(
                    self.decision_time_ms,
                    dtype="<i8",
                ),
                "structural_probability_sha256": _array_sha256(
                    self.structural_probability,
                    dtype="<f8",
                ),
                "market_prior_probability_sha256": _array_sha256(
                    self.market_prior_probability,
                    dtype="<f8",
                ),
                "core_features_sha256": _array_sha256(
                    self.core_features,
                    dtype="<f4",
                ),
                "spot_features_sha256": _array_sha256(
                    self.spot_features,
                    dtype="<f4",
                ),
                "usdm_features_sha256": _array_sha256(
                    self.usdm_features,
                    dtype="<f4",
                ),
                "spot_available_sha256": _array_sha256(
                    self.spot_available,
                    dtype="|b1",
                ),
                "usdm_available_sha256": _array_sha256(
                    self.usdm_available,
                    dtype="|b1",
                ),
                "core_feature_names_sha256": self.core_feature_names_sha256,
                "spot_feature_names_sha256": self.spot_feature_names_sha256,
                "usdm_feature_names_sha256": self.usdm_feature_names_sha256,
                "source_dataset_sha256": self.source_dataset_sha256,
                "dataset_design_sha256": self.dataset_design_sha256,
                "target_accessed": False,
                "trading_authority": False,
            }
        )

    def validate(self) -> Round21InferencePanel:
        condition_ids = np.asarray(self.condition_ids)
        event_starts = np.asarray(self.event_start_ms, dtype=np.int64)
        decisions = np.asarray(self.decision_time_ms, dtype=np.int64)
        structural = np.asarray(self.structural_probability)
        market_prior = np.asarray(self.market_prior_probability)
        rows = len(condition_ids)
        spot_available = np.asarray(self.spot_available)
        usdm_available = np.asarray(self.usdm_available)
        hashes = (
            self.core_feature_names_sha256,
            self.spot_feature_names_sha256,
            self.usdm_feature_names_sha256,
            self.source_dataset_sha256,
            self.feature_batch_sha256,
            self.dataset_design_sha256,
        )
        arrays = (
            self.condition_ids,
            self.event_start_ms,
            self.decision_time_ms,
            self.structural_probability,
            self.market_prior_probability,
            self.core_features,
            self.spot_features,
            self.usdm_features,
            self.spot_available,
            self.usdm_available,
        )
        if (
            rows < 1
            or event_starts.shape != (rows,)
            or decisions.shape != (rows,)
            or structural.shape != (rows,)
            or structural.dtype.kind != "f"
            or market_prior.shape != (rows,)
            or market_prior.dtype.kind != "f"
            or spot_available.shape != (rows,)
            or spot_available.dtype.kind != "b"
            or usdm_available.shape != (rows,)
            or usdm_available.dtype.kind != "b"
            or any(
                _SHA256.fullmatch(str(value or "").strip()) is None
                or str(value).strip() == _EMPTY_SHA256
                for value in hashes
            )
            or self.dataset_design_sha256 != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
            or self.target_accessed
            or self.trading_authority
            or any(
                not isinstance(value, np.ndarray) or value.flags.writeable
                for value in arrays
            )
        ):
            raise ValueError("Round 21 inference panel is invalid")
        core = _float_matrix(self.core_features, rows=rows)
        spot = _float_matrix(self.spot_features, rows=rows)
        usdm = _float_matrix(self.usdm_features, rows=rows)
        structural64 = np.asarray(structural, dtype=np.float64)
        market_prior64 = np.asarray(market_prior, dtype=np.float64)
        if (
            not np.all(np.isfinite(structural64))
            or not np.all(np.isfinite(market_prior64))
            or np.any(structural64 < _PROBABILITY_FLOOR)
            or np.any(structural64 > 1.0 - _PROBABILITY_FLOOR)
            or np.any(market_prior64 < _PROBABILITY_FLOOR)
            or np.any(market_prior64 > 1.0 - _PROBABILITY_FLOOR)
            or np.any(event_starts <= 0)
            or np.any(event_starts % _CONDITION_DURATION_MS)
            or np.any(decisions < event_starts)
            or np.any(decisions >= event_starts + _CONDITION_DURATION_MS)
            or np.any((decisions - event_starts) % 250)
            or np.any(event_starts[1:] < event_starts[:-1])
            or np.any(
                (event_starts[1:] == event_starts[:-1])
                & (decisions[1:] < decisions[:-1])
            )
            or np.any(spot[~spot_available] != 0.0)
            or np.any(usdm[~usdm_available] != 0.0)
            or np.any(usdm_available & ~spot_available)
        ):
            raise ValueError("Round 21 inference panel is invalid")
        groups = _condition_groups(condition_ids)
        for condition, start, end in groups:
            if (
                _CONDITION_ID.fullmatch(condition) is None
                or event_starts[start] != event_starts[end - 1]
                or np.any(decisions[start + 1 : end] <= decisions[start : end - 1])
            ):
                raise ValueError("Round 21 inference condition identity differs")
        group_event_starts = np.asarray(
            [event_starts[start] for _condition, start, _end in groups],
            dtype=np.int64,
        )
        if np.any(group_event_starts[1:] <= group_event_starts[:-1]):
            raise ValueError("Round 21 inference condition identity differs")
        rebuilt = replace(
            self,
            condition_ids=condition_ids,
            event_start_ms=event_starts,
            decision_time_ms=decisions,
            structural_probability=structural64,
            market_prior_probability=market_prior64,
            core_features=core,
            spot_features=spot,
            usdm_features=usdm,
            spot_available=np.asarray(spot_available, dtype=np.bool_),
            usdm_available=np.asarray(usdm_available, dtype=np.bool_),
        )
        if rebuilt.feature_batch_sha256 != rebuilt._computed_feature_batch_sha256():
            raise ValueError("Round 21 inference panel identity differs")
        return rebuilt

    def row_sha256(self, panel_row_index: int) -> str:
        selected = self.validate()
        index = int(panel_row_index)
        if index < 0 or index >= len(selected.condition_ids):
            raise ValueError("Round 21 inference row is unavailable")
        return _canonical_sha256(
            {
                "schema_version": POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION,
                "feature_batch_sha256": selected.feature_batch_sha256,
                "row_index": index,
                "condition_id": str(selected.condition_ids[index]),
                "event_start_ms": int(selected.event_start_ms[index]),
                "decision_time_ms": int(selected.decision_time_ms[index]),
                "structural_probability": format(
                    float(selected.structural_probability[index]),
                    ".17g",
                ),
                "market_prior_probability": format(
                    float(selected.market_prior_probability[index]),
                    ".17g",
                ),
                "core_features_sha256": _array_sha256(
                    selected.core_features[index],
                    dtype="<f4",
                ),
                "spot_features_sha256": _array_sha256(
                    selected.spot_features[index],
                    dtype="<f4",
                ),
                "usdm_features_sha256": _array_sha256(
                    selected.usdm_features[index],
                    dtype="<f4",
                ),
                "spot_available": bool(selected.spot_available[index]),
                "usdm_available": bool(selected.usdm_available[index]),
                "core_feature_names_sha256": (
                    selected.core_feature_names_sha256
                ),
                "spot_feature_names_sha256": (
                    selected.spot_feature_names_sha256
                ),
                "usdm_feature_names_sha256": (
                    selected.usdm_feature_names_sha256
                ),
                "target_accessed": False,
                "trading_authority": False,
            }
        )


Round21FeaturePanel = Round21DevelopmentPanel | Round21InferencePanel


def _layer_mask(panel: Round21FeaturePanel, layer: str) -> np.ndarray:
    if layer == "core":
        return np.ones(len(panel.condition_ids), dtype=np.bool_)
    if layer == "core_spot":
        return panel.spot_available.copy()
    if layer == "core_spot_usdm":
        return panel.spot_available & panel.usdm_available
    raise ValueError("Round 21 feature layer is invalid")


def _layer_matrix(panel: Round21FeaturePanel, layer: str) -> np.ndarray:
    if layer == "core":
        values = panel.core_features
    elif layer == "core_spot":
        values = np.concatenate(
            (panel.core_features, panel.spot_features),
            axis=1,
        )
    elif layer == "core_spot_usdm":
        values = np.concatenate(
            (panel.core_features, panel.spot_features, panel.usdm_features),
            axis=1,
        )
    else:
        raise ValueError("Round 21 feature layer is invalid")
    return np.asarray(values, dtype=np.float32, order="C")


def _feature_layer_sha256(panel: Round21FeaturePanel, layer: str) -> str:
    payload: dict[str, object] = {
        "layer": layer,
        "core_feature_names_sha256": panel.core_feature_names_sha256,
    }
    if layer in {"core_spot", "core_spot_usdm"}:
        payload["spot_feature_names_sha256"] = panel.spot_feature_names_sha256
    if layer == "core_spot_usdm":
        payload["usdm_feature_names_sha256"] = panel.usdm_feature_names_sha256
    if layer not in _LAYERS:
        raise ValueError("Round 21 feature layer is invalid")
    return _canonical_sha256(payload)


def _selected_indices(
    panel: Round21FeaturePanel,
    layer: str,
) -> np.ndarray:
    return np.flatnonzero(_layer_mask(panel, layer))


def _condition_count(condition_ids: np.ndarray) -> int:
    return len(_condition_groups(np.asarray(condition_ids, dtype=object)))


def _condition_weights(condition_ids: np.ndarray) -> np.ndarray:
    groups = _condition_groups(np.asarray(condition_ids, dtype=object))
    weights = np.empty(len(condition_ids), dtype=np.float64)
    for _condition, start, end in groups:
        weights[start:end] = 1.0 / len(groups) / (end - start)
    return weights


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    threshold = float(probability) * float(cumulative[-1])
    index = min(
        len(ordered_values) - 1,
        int(np.searchsorted(cumulative, threshold, side="left")),
    )
    return float(ordered_values[index])


def _fit_transform(
    matrix: np.ndarray,
    weights: np.ndarray,
) -> dict[str, list[float]]:
    lower = np.empty(matrix.shape[1], dtype=np.float64)
    upper = np.empty(matrix.shape[1], dtype=np.float64)
    center = np.empty(matrix.shape[1], dtype=np.float64)
    scale = np.empty(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        column = np.asarray(matrix[:, index], dtype=np.float64)
        lower[index] = _weighted_quantile(column, weights, 0.001)
        upper[index] = _weighted_quantile(column, weights, 0.999)
        clipped = np.clip(column, lower[index], upper[index])
        center[index] = _weighted_quantile(clipped, weights, 0.5)
        first_quartile = _weighted_quantile(clipped, weights, 0.25)
        third_quartile = _weighted_quantile(clipped, weights, 0.75)
        robust_scale = (third_quartile - first_quartile) / 1.3489795003921634
        if robust_scale <= 1e-12:
            absolute_deviation = np.abs(clipped - center[index])
            robust_scale = (
                _weighted_quantile(absolute_deviation, weights, 0.5) * 1.482602218505602
            )
        scale[index] = max(robust_scale, 1.0 if lower[index] == upper[index] else 1e-6)
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
    }


def _transform_matrix(
    matrix: np.ndarray,
    transform: Mapping[str, object],
    *,
    normalize: bool,
) -> np.ndarray:
    lower = np.asarray(transform["lower"], dtype=np.float32)
    upper = np.asarray(transform["upper"], dtype=np.float32)
    if (
        lower.shape != (matrix.shape[1],)
        or upper.shape != (matrix.shape[1],)
        or np.any(lower > upper)
    ):
        raise ValueError("Round 21 feature transform is invalid")
    output = np.asarray(matrix, dtype=np.float32, order="C").copy()
    np.clip(output, lower, upper, out=output)
    if normalize:
        center = np.asarray(transform["center"], dtype=np.float32)
        scale = np.asarray(transform["scale"], dtype=np.float32)
        if (
            center.shape != lower.shape
            or scale.shape != lower.shape
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Round 21 feature transform is invalid")
        np.subtract(output, center, out=output)
        np.divide(output, scale, out=output)
    return output


def _fit_logistic_residual(
    matrix: np.ndarray,
    labels: np.ndarray,
    structural_probability: np.ndarray,
    weights: np.ndarray,
    transform: Mapping[str, object],
    *,
    l2: float,
    population_layer: str,
    feature_layer: str,
    feature_names_sha256: str,
    candidate_namespace: str,
) -> dict[str, object]:
    normalized = _transform_matrix(matrix, transform, normalize=True)
    offset = logit(_probability(structural_probability))
    target = np.asarray(labels, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficient = parameters[1:]
        linear = offset + intercept + normalized @ coefficient
        residual = expit(linear) - target
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
            + 0.5 * l2 * float(coefficient @ coefficient)
        )
        weighted_residual = weights * residual
        gradient = np.concatenate(
            (
                np.asarray([np.sum(weighted_residual)]),
                np.asarray(normalized.T @ weighted_residual) + l2 * coefficient,
            )
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 512, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Round 21 residual logistic fit failed: {str(result.message).strip()}"
        )
    return {
        "candidate_id": (f"{candidate_namespace}-logistic-l2-{format(l2, 'g')}"),
        "family": "logistic_residual",
        "layer": population_layer,
        "population_layer": population_layer,
        "feature_layer": feature_layer,
        "feature_names_sha256": feature_names_sha256,
        "l2": float(l2),
        "transform": dict(transform),
        "intercept": float(result.x[0]),
        "coefficient": result.x[1:].tolist(),
    }


def _fit_lightgbm_residual(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_structural: np.ndarray,
    train_weights: np.ndarray,
    stop_matrix: np.ndarray,
    stop_labels: np.ndarray,
    stop_structural: np.ndarray,
    stop_weights: np.ndarray,
    transform: Mapping[str, object],
    configuration: Mapping[str, int],
    backend_parameters: Mapping[str, object],
    *,
    backend_kind: str,
    backend_device: str,
    population_layer: str,
    feature_layer: str,
    feature_names_sha256: str,
    candidate_namespace: str,
) -> dict[str, object]:
    train_values = _transform_matrix(train_matrix, transform, normalize=False)
    stop_values = _transform_matrix(stop_matrix, transform, normalize=False)
    train_set = lgb.Dataset(
        train_values,
        label=train_labels,
        weight=train_weights,
        init_score=logit(_probability(train_structural)),
        free_raw_data=False,
    )
    stop_set = lgb.Dataset(
        stop_values,
        label=stop_labels,
        weight=stop_weights,
        init_score=logit(_probability(stop_structural)),
        reference=train_set,
        free_raw_data=False,
    )
    parameters: dict[str, object] = {
        **dict(backend_parameters),
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "max_depth": int(configuration["max_depth"]),
        "num_leaves": int(configuration["num_leaves"]),
        "min_data_in_leaf": int(configuration["min_data_in_leaf"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.1,
        "max_bin": 127,
        "seed": POLYMARKET_ROUND21_MODEL_SEED,
        "verbosity": -1,
    }
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=512,
        valid_sets=[stop_set],
        valid_names=["tune_early_stop"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = max(1, int(booster.best_iteration))
    model_string = booster.model_to_string(num_iteration=best_iteration)
    reloaded = lgb.Booster(model_str=model_string)
    original = np.asarray(
        booster.predict(stop_values, num_iteration=best_iteration, raw_score=True),
        dtype=np.float64,
    )
    restored = np.asarray(
        reloaded.predict(stop_values, raw_score=True),
        dtype=np.float64,
    )
    if float(np.max(np.abs(original - restored), initial=0.0)) > 1e-10:
        raise RuntimeError("Round 21 LightGBM serialization identity failed")
    label = (
        f"d{configuration['max_depth']}-l{configuration['num_leaves']}-"
        f"m{configuration['min_data_in_leaf']}"
    )
    return {
        "candidate_id": f"{candidate_namespace}-lightgbm-{label}",
        "family": "lightgbm_residual",
        "layer": population_layer,
        "population_layer": population_layer,
        "feature_layer": feature_layer,
        "feature_names_sha256": feature_names_sha256,
        "configuration": dict(configuration),
        "transform": dict(transform),
        "best_iteration": best_iteration,
        "model_string": model_string,
        "lightgbm_version": str(lgb.__version__),
        "backend_kind": backend_kind,
        "backend_device": backend_device,
    }


def _raw_prediction(
    model: Mapping[str, object],
    matrix: np.ndarray,
    structural_probability: np.ndarray,
) -> np.ndarray:
    family = str(model.get("family") or "")
    offset = logit(_probability(structural_probability))
    transform = model.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Round 21 model transform is unavailable")
    if family == "logistic_residual":
        values = _transform_matrix(matrix, transform, normalize=True)
        coefficient = np.asarray(model["coefficient"], dtype=np.float32)
        if coefficient.shape != (values.shape[1],):
            raise ValueError("Round 21 logistic coefficient shape differs")
        linear = (
            offset
            + float(model["intercept"])
            + np.asarray(values @ coefficient, dtype=np.float64)
        )
    elif family == "lightgbm_residual":
        values = _transform_matrix(matrix, transform, normalize=False)
        booster = lgb.Booster(model_str=str(model["model_string"]))
        linear = offset + np.asarray(
            booster.predict(values, raw_score=True),
            dtype=np.float64,
        )
    else:
        raise ValueError("Round 21 model family is invalid")
    return _probability(expit(linear))


def _fit_platt(
    labels: np.ndarray,
    predictions: np.ndarray,
    condition_ids: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.float64)
    raw = logit(_probability(predictions))
    weights = _condition_weights(condition_ids)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = float(parameters[0]) + float(parameters[1]) * raw
        residual = expit(linear) - target
        loss = float(np.sum(weights * (np.logaddexp(0.0, linear) - target * linear)))
        gradient = np.asarray(
            [
                np.sum(weights * residual),
                np.sum(weights * residual * raw),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((-20.0, 20.0), (0.0, 20.0)),
        options={"maxiter": 256, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Round 21 Platt calibration failed")
    return {"intercept": float(result.x[0]), "slope": float(result.x[1])}


def _apply_platt(
    predictions: np.ndarray,
    calibration: Mapping[str, object],
) -> np.ndarray:
    intercept = float(calibration["intercept"])
    slope = float(calibration["slope"])
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope < 0.0:
        raise ValueError("Round 21 calibration is invalid")
    return _probability(expit(intercept + slope * logit(_probability(predictions))))


def _predict_validated_round21_candidate(
    model: Mapping[str, object],
    selected: Round21FeaturePanel,
) -> tuple[np.ndarray, np.ndarray]:
    population_layer = str(model.get("population_layer") or "")
    feature_layer = str(model.get("feature_layer") or "")
    if (
        model.get("layer") != population_layer
        or population_layer not in _LAYERS
        or feature_layer not in _LAYERS
        or model.get("feature_names_sha256")
        != _feature_layer_sha256(selected, feature_layer)
    ):
        raise ValueError("Round 21 model layer identity differs")
    indices = _selected_indices(selected, population_layer)
    matrix = _layer_matrix(selected, feature_layer)[indices]
    raw = _raw_prediction(
        model,
        matrix,
        selected.structural_probability[indices],
    )
    calibration = model.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Round 21 model calibration is unavailable")
    return indices, _apply_platt(raw, calibration)


def predict_round21_candidate(
    model: Mapping[str, object],
    panel: Round21FeaturePanel,
) -> tuple[np.ndarray, np.ndarray]:
    return _predict_validated_round21_candidate(model, panel.validate())


@dataclass(frozen=True, slots=True)
class Round21ProbabilityBatch:
    population_layer: str
    selected_candidate_id: str
    contributing_candidate_ids: tuple[str, ...]
    indices: np.ndarray
    probability_up: np.ndarray
    lower_up: np.ndarray
    upper_up: np.ndarray
    source_model_artifact_sha256: str
    feature_batch_sha256: str
    prediction_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        population_layer: str,
        selected_candidate_id: str,
        contributing_candidate_ids: Sequence[str],
        indices: np.ndarray,
        probability_up: np.ndarray,
        lower_up: np.ndarray,
        upper_up: np.ndarray,
        source_model_artifact_sha256: str,
        feature_batch_sha256: str,
    ) -> Round21ProbabilityBatch:
        layer = str(population_layer or "").strip()
        selected_id = str(selected_candidate_id or "").strip()
        candidate_ids = tuple(
            str(value or "").strip() for value in contributing_candidate_ids
        )
        selected_indices = np.array(indices, dtype="<i8", order="C", copy=True)
        probability = np.array(probability_up, dtype="<f8", order="C", copy=True)
        lower = np.array(lower_up, dtype="<f8", order="C", copy=True)
        upper = np.array(upper_up, dtype="<f8", order="C", copy=True)
        model_sha = str(source_model_artifact_sha256 or "").strip().lower()
        batch_sha = str(feature_batch_sha256 or "").strip().lower()
        if (
            layer not in _LAYERS
            or not selected_id
            or not candidate_ids
            or selected_id not in candidate_ids
            or len(set(candidate_ids)) != len(candidate_ids)
            or any(not value or len(value) > 200 for value in candidate_ids)
            or selected_indices.ndim != 1
            or probability.shape != selected_indices.shape
            or lower.shape != selected_indices.shape
            or upper.shape != selected_indices.shape
            or not np.all(np.isfinite(probability))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(selected_indices < 0)
            or len(set(selected_indices.tolist())) != len(selected_indices)
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or np.any(lower < 0.0)
            or np.any(upper > 1.0)
            or np.any(lower > probability)
            or np.any(probability > upper)
            or _SHA256.fullmatch(model_sha) is None
            or model_sha == _EMPTY_SHA256
            or _SHA256.fullmatch(batch_sha) is None
            or batch_sha == _EMPTY_SHA256
        ):
            raise ValueError("Round 21 probability batch is invalid")
        for array in (selected_indices, probability, lower, upper):
            array.setflags(write=False)
        provisional = cls(
            population_layer=layer,
            selected_candidate_id=selected_id,
            contributing_candidate_ids=candidate_ids,
            indices=selected_indices,
            probability_up=probability,
            lower_up=lower,
            upper_up=upper,
            source_model_artifact_sha256=model_sha,
            feature_batch_sha256=batch_sha,
            prediction_sha256=_EMPTY_SHA256,
        )
        return replace(
            provisional,
            prediction_sha256=_canonical_sha256(provisional.identity_payload()),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_PROBABILITY_BATCH_SCHEMA_VERSION,
            "model_design_sha256": POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
            "probability_envelope_design_sha256": (
                POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256
            ),
            "population_layer": self.population_layer,
            "selected_candidate_id": self.selected_candidate_id,
            "contributing_candidate_ids": list(self.contributing_candidate_ids),
            "row_count": len(self.indices),
            "indices_sha256": _array_sha256(self.indices, dtype="<i8"),
            "probability_up_sha256": _array_sha256(
                self.probability_up,
                dtype="<f8",
            ),
            "lower_up_sha256": _array_sha256(self.lower_up, dtype="<f8"),
            "upper_up_sha256": _array_sha256(self.upper_up, dtype="<f8"),
            "source_model_artifact_sha256": self.source_model_artifact_sha256,
            "feature_batch_sha256": self.feature_batch_sha256,
            "bound_semantics": "calibrated_candidate_disagreement_hull",
            "formal_coverage_claim": False,
            "trading_authority": False,
        }

    def validated(self) -> Round21ProbabilityBatch:
        rebuilt = self.create(
            population_layer=self.population_layer,
            selected_candidate_id=self.selected_candidate_id,
            contributing_candidate_ids=self.contributing_candidate_ids,
            indices=self.indices,
            probability_up=self.probability_up,
            lower_up=self.lower_up,
            upper_up=self.upper_up,
            source_model_artifact_sha256=self.source_model_artifact_sha256,
            feature_batch_sha256=self.feature_batch_sha256,
        )
        if (
            any(
                array.flags.writeable
                for array in (
                    self.indices,
                    self.probability_up,
                    self.lower_up,
                    self.upper_up,
                )
            )
            or self.population_layer != rebuilt.population_layer
            or self.selected_candidate_id != rebuilt.selected_candidate_id
            or self.contributing_candidate_ids != rebuilt.contributing_candidate_ids
            or not np.array_equal(self.indices, rebuilt.indices)
            or not np.array_equal(self.probability_up, rebuilt.probability_up)
            or not np.array_equal(self.lower_up, rebuilt.lower_up)
            or not np.array_equal(self.upper_up, rebuilt.upper_up)
            or self.source_model_artifact_sha256 != rebuilt.source_model_artifact_sha256
            or self.feature_batch_sha256 != rebuilt.feature_batch_sha256
            or self.prediction_sha256 != rebuilt.prediction_sha256
            or self.trading_authority
        ):
            raise ValueError("Round 21 probability batch differs")
        return self

    def row(self, panel_row_index: int) -> tuple[float, float, float]:
        selected = self.validated()
        matches = np.flatnonzero(selected.indices == int(panel_row_index))
        if len(matches) != 1:
            raise ValueError("Round 21 probability row is unavailable")
        position = int(matches[0])
        return (
            float(selected.probability_up[position]),
            float(selected.lower_up[position]),
            float(selected.upper_up[position]),
        )


def _condition_losses(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    groups = _condition_groups(condition_ids)
    losses: list[float] = []
    conditions: list[str] = []
    probability = _probability(predictions)
    for condition, start, end in groups:
        target = labels[start:end]
        selected = probability[start:end]
        if metric == "log_loss":
            value = -np.mean(
                target * np.log(selected) + (1.0 - target) * np.log1p(-selected)
            )
        elif metric == "brier":
            value = np.mean(np.square(selected - target))
        else:
            raise ValueError("Round 21 metric is invalid")
        conditions.append(condition)
        losses.append(float(value))
    return np.asarray(losses, dtype=np.float64), tuple(conditions)


def _metrics(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    losses, conditions = _condition_losses(
        condition_ids,
        labels,
        predictions,
        metric="log_loss",
    )
    brier, _ = _condition_losses(
        condition_ids,
        labels,
        predictions,
        metric="brier",
    )
    return {
        "condition_count": len(conditions),
        "condition_equal_log_loss": float(np.mean(losses)),
        "condition_equal_brier_score": float(np.mean(brier)),
        "log_loss_standard_error": (
            0.0
            if len(losses) < 2
            else float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        ),
    }


def _paired_improvement(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    metric: str,
    seed_offset: int,
) -> dict[str, float | int]:
    control_loss, control_conditions = _condition_losses(
        condition_ids,
        labels,
        control,
        metric=metric,
    )
    candidate_loss, candidate_conditions = _condition_losses(
        condition_ids,
        labels,
        candidate,
        metric=metric,
    )
    if control_conditions != candidate_conditions:
        raise RuntimeError("Round 21 paired condition identities differ")
    difference = control_loss - candidate_loss
    generator = np.random.default_rng(POLYMARKET_ROUND21_MODEL_SEED + int(seed_offset))
    samples = np.empty(POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES):
        selected = generator.integers(0, len(difference), size=len(difference))
        samples[index] = float(np.mean(difference[selected]))
    return {
        "condition_count": len(difference),
        "mean": float(np.mean(difference)),
        "lower_95": float(np.quantile(samples, 0.025, method="linear")),
        "upper_95": float(np.quantile(samples, 0.975, method="linear")),
    }


def _split_calibration_indices(
    panel: Round21DevelopmentPanel,
    layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _selected_indices(panel, layer)
    selected_conditions = panel.condition_ids[indices]
    groups = _condition_groups(selected_conditions)
    if len(groups) < 2 * POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS:
        raise ValueError("Round 21 calibration layer has too few conditions")
    midpoint = len(groups) // 2
    split_row = groups[midpoint][1]
    return indices[:split_row], indices[split_row:]


def _fit_layer_candidates(
    train: Round21DevelopmentPanel,
    calibration: Round21DevelopmentPanel,
    selection: Round21DevelopmentPanel,
    population_layer: str,
    feature_layer: str,
    candidate_namespace: str,
    backend_parameters: Mapping[str, object],
    *,
    backend_kind: str,
    backend_device: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    train_indices = _selected_indices(train, population_layer)
    selection_indices = _selected_indices(selection, population_layer)
    stop_indices, platt_indices = _split_calibration_indices(
        calibration,
        population_layer,
    )
    for name, panel, indices in (
        ("train", train, train_indices),
        ("tune early-stop", calibration, stop_indices),
        ("tune calibration", calibration, platt_indices),
        ("tune selection", selection, selection_indices),
    ):
        labels = panel.labels[indices]
        if _condition_count(panel.condition_ids[indices]) < (
            POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
        ) or set(labels.tolist()) != {0.0, 1.0}:
            raise ValueError(
                f"Round 21 {population_layer} {name} has too few conditions"
            )
    train_matrix = _layer_matrix(train, feature_layer)[train_indices]
    stop_matrix = _layer_matrix(calibration, feature_layer)[stop_indices]
    platt_matrix = _layer_matrix(calibration, feature_layer)[platt_indices]
    selection_matrix = _layer_matrix(selection, feature_layer)[selection_indices]
    train_weights = _condition_weights(train.condition_ids[train_indices])
    stop_weights = _condition_weights(calibration.condition_ids[stop_indices])
    transform = _fit_transform(train_matrix, train_weights)
    feature_names_sha = _feature_layer_sha256(train, feature_layer)
    if any(
        _feature_layer_sha256(panel, feature_layer) != feature_names_sha
        for panel in (calibration, selection)
    ):
        raise ValueError("Round 21 feature layer identity differs")
    models: list[dict[str, object]] = []
    for l2 in _LOGISTIC_L2:
        models.append(
            _fit_logistic_residual(
                train_matrix,
                train.labels[train_indices],
                train.structural_probability[train_indices],
                train_weights,
                transform,
                l2=l2,
                population_layer=population_layer,
                feature_layer=feature_layer,
                feature_names_sha256=feature_names_sha,
                candidate_namespace=candidate_namespace,
            )
        )
    for configuration in _LIGHTGBM_GRID:
        models.append(
            _fit_lightgbm_residual(
                train_matrix,
                train.labels[train_indices],
                train.structural_probability[train_indices],
                train_weights,
                stop_matrix,
                calibration.labels[stop_indices],
                calibration.structural_probability[stop_indices],
                stop_weights,
                transform,
                configuration,
                backend_parameters,
                backend_kind=backend_kind,
                backend_device=backend_device,
                population_layer=population_layer,
                feature_layer=feature_layer,
                feature_names_sha256=feature_names_sha,
                candidate_namespace=candidate_namespace,
            )
        )
    records: list[dict[str, object]] = []
    for model in models:
        raw_calibration = _raw_prediction(
            model,
            platt_matrix,
            calibration.structural_probability[platt_indices],
        )
        model["calibration"] = _fit_platt(
            calibration.labels[platt_indices],
            raw_calibration,
            calibration.condition_ids[platt_indices],
        )
        prediction = _apply_platt(
            _raw_prediction(
                model,
                selection_matrix,
                selection.structural_probability[selection_indices],
            ),
            model["calibration"],  # type: ignore[arg-type]
        )
        records.append(
            {
                "candidate_id": model["candidate_id"],
                "family": model["family"],
                "population_layer": population_layer,
                "feature_layer": feature_layer,
                "model": model,
                "selection_metrics": _metrics(
                    selection.condition_ids[selection_indices],
                    selection.labels[selection_indices],
                    prediction,
                ),
            }
        )
    return records, selection_indices


def _select_candidate(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        raise ValueError("Round 21 candidate ledger is empty")
    best = min(
        records,
        key=lambda item: float(
            item["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        ),
    )
    threshold = float(
        best["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
    ) + float(
        best["selection_metrics"]["log_loss_standard_error"]  # type: ignore[index]
    )
    eligible = [
        item
        for item in records
        if float(
            item["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        )
        <= threshold
    ]
    return min(
        eligible,
        key=lambda item: (
            0 if item["family"] == "logistic_residual" else 1,
            -float(item["model"].get("l2", 0.0)),  # type: ignore[union-attr]
            float(
                item["selection_metrics"][  # type: ignore[index]
                    "condition_equal_log_loss"
                ]
            ),
            str(item["candidate_id"]),
        ),
    )


def _control_predictions(
    train: Round21DevelopmentPanel,
    calibration: Round21DevelopmentPanel,
    selection: Round21DevelopmentPanel,
) -> list[dict[str, object]]:
    _stop_indices, platt_indices = _split_calibration_indices(
        calibration,
        "core",
    )
    controls: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "structural_probability",
            calibration.structural_probability[platt_indices],
            selection.structural_probability,
        ),
        (
            "executable_market_prior",
            calibration.market_prior_probability[platt_indices],
            selection.market_prior_probability,
        ),
    ]
    records: list[dict[str, object]] = []
    for index, (control_id, calibration_raw, selection_raw) in enumerate(controls):
        calibration_parameters = _fit_platt(
            calibration.labels[platt_indices],
            calibration_raw,
            calibration.condition_ids[platt_indices],
        )
        calibrated = _apply_platt(selection_raw, calibration_parameters)
        records.extend(
            (
                {
                    "control_id": f"{control_id}_raw",
                    "prediction": selection_raw.tolist(),
                    "metrics": _metrics(
                        selection.condition_ids,
                        selection.labels,
                        selection_raw,
                    ),
                },
                {
                    "control_id": f"{control_id}_calibrated",
                    "calibration": calibration_parameters,
                    "prediction": calibrated.tolist(),
                    "metrics": _metrics(
                        selection.condition_ids,
                        selection.labels,
                        calibrated,
                    ),
                },
            )
        )
    prevalence = float(np.sum(_condition_weights(train.condition_ids) * train.labels))
    prevalence_prediction = np.full(
        len(selection.labels),
        prevalence,
        dtype=np.float64,
    )
    records.append(
        {
            "control_id": "training_prevalence",
            "probability_up": prevalence,
            "prediction": prevalence_prediction.tolist(),
            "metrics": _metrics(
                selection.condition_ids,
                selection.labels,
                prevalence_prediction,
            ),
        }
    )
    return records


def _dataset_identity(panel: Round21DevelopmentPanel) -> dict[str, object]:
    return {
        "role": panel.role,
        "row_count": len(panel.labels),
        "condition_count": _condition_count(panel.condition_ids),
        "first_event_start_ms": int(panel.event_start_ms[0]),
        "last_event_start_ms": int(panel.event_start_ms[-1]),
        "dataset_sha256": panel.dataset_sha256,
        "target_manifest_sha256": panel.target_manifest_sha256,
        "dataset_design_sha256": panel.dataset_design_sha256,
    }


def fit_round21_development(
    *,
    train: Round21DevelopmentPanel,
    tune_calibration: Round21DevelopmentPanel,
    tune_selection: Round21DevelopmentPanel,
    compute_backend: str = "auto",
) -> dict[str, object]:
    train = train.validate()
    tune_calibration = tune_calibration.validate()
    tune_selection = tune_selection.validate()
    if (
        train.role != "train"
        or tune_calibration.role != "tune_calibration"
        or tune_selection.role != "tune_selection"
        or tune_calibration.event_start_ms[0] - train.event_start_ms[-1]
        < _TRAIN_TO_TUNE_MINIMUM_START_GAP_MS
        or tune_selection.event_start_ms[0] - tune_calibration.event_start_ms[-1]
        < _CONDITION_DURATION_MS
        or not set(train.condition_ids).isdisjoint(tune_calibration.condition_ids)
        or not set(train.condition_ids).isdisjoint(tune_selection.condition_ids)
        or not set(tune_calibration.condition_ids).isdisjoint(
            tune_selection.condition_ids
        )
        or len(
            {
                train.core_feature_names_sha256,
                tune_calibration.core_feature_names_sha256,
                tune_selection.core_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.spot_feature_names_sha256,
                tune_calibration.spot_feature_names_sha256,
                tune_selection.spot_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.usdm_feature_names_sha256,
                tune_calibration.usdm_feature_names_sha256,
                tune_selection.usdm_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.core_features.shape[1],
                tune_calibration.core_features.shape[1],
                tune_selection.core_features.shape[1],
            }
        )
        != 1
        or len(
            {
                train.spot_features.shape[1],
                tune_calibration.spot_features.shape[1],
                tune_selection.spot_features.shape[1],
            }
        )
        != 1
        or len(
            {
                train.usdm_features.shape[1],
                tune_calibration.usdm_features.shape[1],
                tune_selection.usdm_features.shape[1],
            }
        )
        != 1
    ):
        raise ValueError("Round 21 development partition boundary differs")
    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        POLYMARKET_ROUND21_MODEL_SEED,
        reproducible=True,
        pin_opencl_device=True,
    )
    controls = _control_predictions(
        train,
        tune_calibration,
        tune_selection,
    )
    strongest_control = min(
        controls,
        key=lambda item: float(
            item["metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        ),
    )
    layer_results: dict[str, object] = {}
    for layer_index, layer in enumerate(_LAYERS):
        records, selection_indices = _fit_layer_candidates(
            train,
            tune_calibration,
            tune_selection,
            layer,
            layer,
            layer,
            backend_parameters,
            backend_kind=backend_kind,
            backend_device=backend_device,
        )
        selected_record = _select_candidate(records)
        selected_model = selected_record["model"]
        if not isinstance(selected_model, Mapping):
            raise RuntimeError("Round 21 selected model is unavailable")
        indices, prediction = _predict_validated_round21_candidate(
            selected_model,
            tune_selection,
        )
        if not np.array_equal(indices, selection_indices):
            raise RuntimeError("Round 21 selected prediction population differs")
        layer_payload: dict[str, object] = {
            "candidate_ledger": records,
            "selected_candidate_id": selected_model["candidate_id"],
            "selection_indices_sha256": hashlib.sha256(
                np.asarray(selection_indices, dtype="<i8").tobytes()
            ).hexdigest(),
        }
        if layer == "core":
            improvements = {
                str(control["control_id"]): {
                    metric: _paired_improvement(
                        tune_selection.condition_ids,
                        tune_selection.labels,
                        np.asarray(control["prediction"], dtype=np.float64),
                        prediction,
                        metric=metric,
                        seed_offset=layer_index * 100
                        + control_index * 10
                        + metric_index,
                    )
                    for metric_index, metric in enumerate(("log_loss", "brier"))
                }
                for control_index, control in enumerate(controls)
            }
            accepted = all(
                result[metric]["lower_95"] > 0.0
                for result in improvements.values()
                for metric in ("log_loss", "brier")
            )
            comparison = {
                "against_every_control": improvements,
                "predictive_development_accepted": accepted,
            }
        else:
            matched_records, matched_indices = _fit_layer_candidates(
                train,
                tune_calibration,
                tune_selection,
                layer,
                "core",
                f"{layer}-matched-core",
                backend_parameters,
                backend_kind=backend_kind,
                backend_device=backend_device,
            )
            matched_record = _select_candidate(matched_records)
            matched_model = matched_record["model"]
            if not isinstance(matched_model, Mapping):
                raise RuntimeError("Round 21 matched core model is unavailable")
            predicted_indices, matched_core = _predict_validated_round21_candidate(
                matched_model,
                tune_selection,
            )
            if not np.array_equal(
                matched_indices, selection_indices
            ) or not np.array_equal(
                predicted_indices,
                selection_indices,
            ):
                raise RuntimeError("Round 21 matched core population differs")
            improvements = {
                metric: _paired_improvement(
                    tune_selection.condition_ids[selection_indices],
                    tune_selection.labels[selection_indices],
                    matched_core,
                    prediction,
                    metric=metric,
                    seed_offset=layer_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("log_loss", "brier"))
            }
            accepted = all(
                improvements[metric]["lower_95"] > 0.0
                for metric in ("log_loss", "brier")
            )
            comparison = {
                "matched_core_candidate_id": matched_model["candidate_id"],
                "matched_decision_count": len(selection_indices),
                "matched_condition_count": _condition_count(
                    tune_selection.condition_ids[selection_indices]
                ),
                "incremental_improvement": improvements,
                "predictive_development_accepted": accepted,
            }
            layer_payload["matched_core_candidate_ledger"] = matched_records
            layer_payload["matched_core_selected_candidate_id"] = matched_model[
                "candidate_id"
            ]
        layer_payload["comparison"] = comparison
        layer_results[layer] = layer_payload
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
        "model_design_sha256": POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
        "dataset_and_partition": {
            "train": _dataset_identity(train),
            "tune_calibration": _dataset_identity(tune_calibration),
            "tune_selection": _dataset_identity(tune_selection),
            "core_feature_names_sha256": train.core_feature_names_sha256,
            "spot_feature_names_sha256": train.spot_feature_names_sha256,
            "usdm_feature_names_sha256": train.usdm_feature_names_sha256,
        },
        "controls": [
            {key: value for key, value in control.items() if key != "prediction"}
            for control in controls
        ],
        "strongest_control_id": strongest_control["control_id"],
        "layers": layer_results,
        "preprocessing": {
            "fit_population": "train_only_condition_equal_weighted",
            "winsor_quantiles": ["0.001", "0.999"],
            "center": "weighted_median",
            "scale": "weighted_iqr_over_1.3489795003921634",
            "constant_feature_scale": "1",
            "optional_missingness": "explicit_zero_with_availability_gate",
        },
        "selection_rule": {
            "primary_metric": "condition_equal_log_loss",
            "prefer_simplest_within_one_standard_error": True,
            "optional_layers_use_exact_matched_core_train_population": True,
            "optional_layers_use_exact_matched_core_early_stop_population": True,
            "optional_layers_use_exact_matched_core_calibration_population": True,
            "optional_layers_use_exact_matched_core_selection_population": True,
        },
        "compute": {
            "requested": str(compute_backend),
            "lightgbm_backend_kind": backend_kind,
            "lightgbm_backend_device": backend_device,
        },
        "economic_evaluation_completed": False,
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "model_selected": False,
        "ai_edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_round21_development_artifact(payload)


def _valid_transform(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "lower",
        "upper",
        "center",
        "scale",
    }:
        return False
    try:
        lower = np.asarray(value["lower"], dtype=np.float64)
        upper = np.asarray(value["upper"], dtype=np.float64)
        center = np.asarray(value["center"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        lower.ndim == 1
        and len(lower) > 0
        and upper.shape == lower.shape
        and center.shape == lower.shape
        and scale.shape == lower.shape
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(scale))
        and np.all(lower <= center)
        and np.all(center <= upper)
        and np.all(scale > 0)
    )


def _artifact_feature_layer_sha256(
    dataset_and_partition: Mapping[str, object],
    layer: str,
) -> str:
    payload: dict[str, object] = {
        "layer": layer,
        "core_feature_names_sha256": dataset_and_partition.get(
            "core_feature_names_sha256"
        ),
    }
    if layer in {"core_spot", "core_spot_usdm"}:
        payload["spot_feature_names_sha256"] = dataset_and_partition.get(
            "spot_feature_names_sha256"
        )
    if layer == "core_spot_usdm":
        payload["usdm_feature_names_sha256"] = dataset_and_partition.get(
            "usdm_feature_names_sha256"
        )
    return _canonical_sha256(payload)


def _valid_candidate_ledger(
    value: object,
    *,
    population_layer: str,
    feature_layer: str,
    feature_names_sha256: str,
    candidate_namespace: str,
) -> bool:
    if not isinstance(value, list) or len(value) != 5:
        return False
    candidate_ids: set[str] = set()
    family_counts = {"logistic_residual": 0, "lightgbm_residual": 0}
    for record in value:
        if not isinstance(record, Mapping):
            return False
        model = record.get("model")
        metrics = record.get("selection_metrics")
        if not isinstance(model, Mapping) or not isinstance(metrics, Mapping):
            return False
        candidate_id = str(record.get("candidate_id") or "")
        family = str(record.get("family") or "")
        if (
            not candidate_id.startswith(f"{candidate_namespace}-")
            or candidate_id in candidate_ids
            or family not in family_counts
            or record.get("population_layer") != population_layer
            or record.get("feature_layer") != feature_layer
            or model.get("candidate_id") != candidate_id
            or model.get("family") != family
            or model.get("layer") != population_layer
            or model.get("population_layer") != population_layer
            or model.get("feature_layer") != feature_layer
            or model.get("feature_names_sha256") != feature_names_sha256
            or not _valid_transform(model.get("transform"))
        ):
            return False
        try:
            metric_values = (
                float(metrics["condition_equal_log_loss"]),
                float(metrics["condition_equal_brier_score"]),
                float(metrics["log_loss_standard_error"]),
            )
            condition_count = int(metrics["condition_count"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            not all(math.isfinite(item) and item >= 0 for item in metric_values)
            or condition_count < POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
        ):
            return False
        transform = model["transform"]
        if not isinstance(transform, Mapping):
            return False
        width = len(transform["lower"])  # type: ignore[arg-type]
        if family == "logistic_residual":
            try:
                coefficient = np.asarray(model["coefficient"], dtype=np.float64)
                intercept = float(model["intercept"])
                l2 = float(model["l2"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if (
                coefficient.shape != (width,)
                or not np.all(np.isfinite(coefficient))
                or not math.isfinite(intercept)
                or l2 not in _LOGISTIC_L2
            ):
                return False
        else:
            try:
                best_iteration = int(model.get("best_iteration", 0))
            except (TypeError, ValueError, OverflowError):
                return False
            if (
                model.get("configuration") not in _LIGHTGBM_GRID
                or not 1 <= best_iteration <= 512
                or not str(model.get("model_string") or "")
                or not 1 <= len(str(model.get("lightgbm_version") or "")) <= 64
                or model.get("backend_kind") not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
                or not str(model.get("backend_device") or "")
            ):
                return False
        candidate_ids.add(candidate_id)
        family_counts[family] += 1
    return family_counts == {"logistic_residual": 3, "lightgbm_residual": 2}


def validate_round21_development_artifact(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    layers = payload.get("layers")
    controls = payload.get("controls")
    dataset_and_partition = payload.get("dataset_and_partition")
    preprocessing = payload.get("preprocessing")
    selection_rule = payload.get("selection_rule")
    false_fields = (
        "economic_evaluation_completed",
        "test_features_accessed",
        "test_targets_accessed",
        "model_selected",
        "ai_edge_claim",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version") != POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION
        or payload.get("contract_sha256") != POLYMARKET_ROUND21_CONTRACT_SHA256
        or payload.get("dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or payload.get("model_design_sha256") != POLYMARKET_ROUND21_MODEL_DESIGN_SHA256
        or not isinstance(controls, list)
        or len(controls) != 5
        or not isinstance(dataset_and_partition, Mapping)
        or not isinstance(layers, Mapping)
        or set(layers) != set(_LAYERS)
        or preprocessing
        != {
            "fit_population": "train_only_condition_equal_weighted",
            "winsor_quantiles": ["0.001", "0.999"],
            "center": "weighted_median",
            "scale": "weighted_iqr_over_1.3489795003921634",
            "constant_feature_scale": "1",
            "optional_missingness": "explicit_zero_with_availability_gate",
        }
        or selection_rule
        != {
            "primary_metric": "condition_equal_log_loss",
            "prefer_simplest_within_one_standard_error": True,
            "optional_layers_use_exact_matched_core_train_population": True,
            "optional_layers_use_exact_matched_core_early_stop_population": True,
            "optional_layers_use_exact_matched_core_calibration_population": True,
            "optional_layers_use_exact_matched_core_selection_population": True,
        }
        or any(
            not isinstance(layer, Mapping)
            or not _valid_candidate_ledger(
                layer.get("candidate_ledger"),
                population_layer=layer_name,
                feature_layer=layer_name,
                feature_names_sha256=_artifact_feature_layer_sha256(
                    dataset_and_partition,
                    layer_name,
                ),
                candidate_namespace=layer_name,
            )
            or layer.get("selected_candidate_id")
            not in {
                record.get("candidate_id")
                for record in layer["candidate_ledger"]
                if isinstance(record, Mapping)
            }
            or _SHA256.fullmatch(str(layer.get("selection_indices_sha256") or ""))
            is None
            or (
                layer_name == "core"
                and (
                    "matched_core_candidate_ledger" in layer
                    or "matched_core_selected_candidate_id" in layer
                )
            )
            or (
                layer_name != "core"
                and (
                    not _valid_candidate_ledger(
                        layer.get("matched_core_candidate_ledger"),
                        population_layer=layer_name,
                        feature_layer="core",
                        feature_names_sha256=_artifact_feature_layer_sha256(
                            dataset_and_partition,
                            "core",
                        ),
                        candidate_namespace=f"{layer_name}-matched-core",
                    )
                    or layer.get("matched_core_selected_candidate_id")
                    not in {
                        record.get("candidate_id")
                        for record in layer.get(
                            "matched_core_candidate_ledger",
                            [],
                        )
                        if isinstance(record, Mapping)
                    }
                    or not isinstance(layer.get("comparison"), Mapping)
                    or layer["comparison"].get("matched_core_candidate_id")
                    != layer.get("matched_core_selected_candidate_id")
                )
            )
            for layer_name, layer in layers.items()
        )
        or any(payload.get(field) is not False for field in false_fields)
    ):
        raise ValueError("Round 21 development artifact differs")
    return {**payload, "artifact_sha256": claimed}


def predict_round21_probability_batch(
    artifact: Mapping[str, object],
    *,
    population_layer: str,
    panel: Round21FeaturePanel,
) -> Round21ProbabilityBatch:
    """Predict a target-free conservative hull over frozen calibrated candidates."""

    validated_artifact = validate_round21_development_artifact(artifact)
    selected_panel = panel.validate()
    layer_name = str(population_layer or "").strip()
    layers = validated_artifact["layers"]
    if not isinstance(layers, Mapping) or layer_name not in _LAYERS:
        raise ValueError("Round 21 probability layer is invalid")
    layer = layers.get(layer_name)
    if not isinstance(layer, Mapping):
        raise ValueError("Round 21 probability layer is unavailable")
    selected_candidate_id = str(layer.get("selected_candidate_id") or "")
    ledgers = [layer.get("candidate_ledger")]
    if layer_name != "core":
        ledgers.append(layer.get("matched_core_candidate_ledger"))
    candidate_ids: list[str] = []
    selected_indices: np.ndarray | None = None
    point: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    for ledger in ledgers:
        if not isinstance(ledger, list):
            raise ValueError("Round 21 probability candidate ledger is unavailable")
        for record in ledger:
            if not isinstance(record, Mapping) or not isinstance(
                record.get("model"),
                Mapping,
            ):
                raise ValueError("Round 21 probability candidate is unavailable")
            candidate_id = str(record.get("candidate_id") or "")
            indices, prediction = _predict_validated_round21_candidate(
                record["model"],
                selected_panel,
            )
            if selected_indices is None:
                selected_indices = indices
                lower = prediction.copy()
                upper = prediction.copy()
            elif not np.array_equal(indices, selected_indices):
                raise RuntimeError("Round 21 probability populations differ")
            else:
                if lower is None or upper is None:
                    raise AssertionError("Round 21 probability bounds are unavailable")
                np.minimum(lower, prediction, out=lower)
                np.maximum(upper, prediction, out=upper)
            if candidate_id == selected_candidate_id:
                point = prediction.copy()
            candidate_ids.append(candidate_id)
    if selected_indices is None or point is None or lower is None or upper is None:
        raise RuntimeError("Round 21 probability batch is unavailable")
    feature_batch_sha = (
        selected_panel.dataset_sha256
        if isinstance(selected_panel, Round21DevelopmentPanel)
        else selected_panel.feature_batch_sha256
    )
    return Round21ProbabilityBatch.create(
        population_layer=layer_name,
        selected_candidate_id=selected_candidate_id,
        contributing_candidate_ids=candidate_ids,
        indices=selected_indices,
        probability_up=point,
        lower_up=lower,
        upper_up=upper,
        source_model_artifact_sha256=validated_artifact["artifact_sha256"],
        feature_batch_sha256=feature_batch_sha,
    ).validated()


__all__ = [
    "POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES",
    "POLYMARKET_ROUND21_DATASET_DESIGN_SHA256",
    "POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS",
    "POLYMARKET_ROUND21_MODEL_DESIGN_SHA256",
    "POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_PROBABILITY_BATCH_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256",
    "Round21DevelopmentPanel",
    "Round21InferencePanel",
    "Round21ProbabilityBatch",
    "fit_round21_development",
    "predict_round21_candidate",
    "predict_round21_probability_batch",
    "validate_round21_development_artifact",
]
