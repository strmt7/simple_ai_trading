"""Leak-resistant scaling for the Round 74 causal event representation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
)


ROUND74_EVENT_SCALER_SCHEMA_VERSION = "round-074-event-feature-scaler-v1"
ROUND74_EVENT_BINARY_FEATURE_COUNT = 8
ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS = 250_000
ROUND74_EVENT_SCALER_STANDARDIZED_CLIP = 12.0
ROUND74_EVENT_SCALER_MINIMUM_SCALE = 1e-6


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _finite_vector(value: object, *, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=np.float64)
    if selected.shape != (len(ROUND74_EVENT_FEATURE_NAMES),):
        raise ValueError(f"Round 74 scaler {label} dimensions differ")
    if not np.isfinite(selected).all():
        raise ValueError(f"Round 74 scaler {label} contains nonfinite values")
    return np.ascontiguousarray(selected)


@dataclass(frozen=True)
class Round74EventFeatureScaler:
    """Training-only robust statistics bound to the exact feature schema."""

    median: np.ndarray
    scale: np.ndarray
    lower_clip: np.ndarray
    upper_clip: np.ndarray
    constant_mask: np.ndarray
    fit_input_rows: int
    fit_sample_rows: int
    fit_sample_index_sha256: str
    feature_names_sha256: str = ROUND74_EVENT_FEATURE_NAMES_SHA256
    schema_version: str = ROUND74_EVENT_SCALER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        median = _finite_vector(self.median, label="median")
        scale = _finite_vector(self.scale, label="scale")
        lower = _finite_vector(self.lower_clip, label="lower clip")
        upper = _finite_vector(self.upper_clip, label="upper clip")
        constant = np.asarray(self.constant_mask, dtype=np.bool_)
        if constant.shape != median.shape:
            raise ValueError("Round 74 scaler constant-mask dimensions differ")
        if np.any(scale <= 0.0):
            raise ValueError("Round 74 scaler scale must be positive")
        if np.any(lower > upper):
            raise ValueError("Round 74 scaler clip interval is inverted")
        if self.feature_names_sha256 != ROUND74_EVENT_FEATURE_NAMES_SHA256:
            raise ValueError("Round 74 scaler feature schema differs")
        if self.schema_version != ROUND74_EVENT_SCALER_SCHEMA_VERSION:
            raise ValueError("Round 74 scaler schema differs")
        if (
            int(self.fit_input_rows) < 1
            or int(self.fit_sample_rows) < 1
            or int(self.fit_sample_rows) > int(self.fit_input_rows)
        ):
            raise ValueError("Round 74 scaler fit row counts are invalid")
        digest = str(self.fit_sample_index_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("Round 74 scaler sample-index digest is invalid")
        binary = slice(0, ROUND74_EVENT_BINARY_FEATURE_COUNT)
        if (
            np.any(median[binary] != 0.0)
            or np.any(scale[binary] != 1.0)
            or np.any(lower[binary] != 0.0)
            or np.any(upper[binary] != 1.0)
            or np.any(constant[binary])
        ):
            raise ValueError("Round 74 scaler binary-feature contract differs")
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "lower_clip", lower)
        object.__setattr__(self, "upper_clip", upper)
        object.__setattr__(self, "constant_mask", np.ascontiguousarray(constant))

    @property
    def scaler_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values)
        if selected.ndim < 2 or selected.shape[-1] != len(
            ROUND74_EVENT_FEATURE_NAMES
        ):
            raise ValueError("Round 74 scaler input dimensions differ")
        numeric = selected.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("Round 74 scaler input contains nonfinite values")
        binary = numeric[..., :ROUND74_EVENT_BINARY_FEATURE_COUNT]
        if np.any((binary != 0.0) & (binary != 1.0)):
            raise ValueError("Round 74 scaler binary input is not one-hot data")
        clipped = np.clip(numeric, self.lower_clip, self.upper_clip)
        transformed = (clipped - self.median) / self.scale
        transformed = np.clip(
            transformed,
            -ROUND74_EVENT_SCALER_STANDARDIZED_CLIP,
            ROUND74_EVENT_SCALER_STANDARDIZED_CLIP,
        )
        transformed[..., :ROUND74_EVENT_BINARY_FEATURE_COUNT] = binary
        transformed[..., self.constant_mask] = 0.0
        result = np.ascontiguousarray(transformed, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Round 74 scaled features contain nonfinite values")
        return result

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
            "binary_feature_count": ROUND74_EVENT_BINARY_FEATURE_COUNT,
            "fit_partition_role": "training",
            "fit_input_rows": int(self.fit_input_rows),
            "fit_sample_rows": int(self.fit_sample_rows),
            "fit_sample_index_sha256": self.fit_sample_index_sha256,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "lower_clip": self.lower_clip.tolist(),
            "upper_clip": self.upper_clip.tolist(),
            "constant_mask": self.constant_mask.tolist(),
            "validation_or_test_statistics_used": False,
        }
        if include_sha256:
            value["scaler_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74EventFeatureScaler:
        payload = dict(value)
        claimed = str(payload.pop("scaler_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 scaler payload digest differs")
        if payload.get("fit_partition_role") != "training":
            raise ValueError("Round 74 scaler fit role differs")
        if payload.get("validation_or_test_statistics_used") is not False:
            raise ValueError("Round 74 scaler payload reports leakage")
        return cls(
            schema_version=str(payload["schema_version"]),
            feature_names_sha256=str(payload["feature_names_sha256"]),
            fit_input_rows=int(payload["fit_input_rows"]),
            fit_sample_rows=int(payload["fit_sample_rows"]),
            fit_sample_index_sha256=str(payload["fit_sample_index_sha256"]),
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            lower_clip=np.asarray(payload["lower_clip"], dtype=np.float64),
            upper_clip=np.asarray(payload["upper_clip"], dtype=np.float64),
            constant_mask=np.asarray(payload["constant_mask"], dtype=np.bool_),
        )


def _uniform_sample_indices(rows: int, maximum_rows: int) -> np.ndarray:
    selected_rows = min(int(rows), int(maximum_rows))
    if selected_rows < 1:
        raise ValueError("Round 74 scaler sample size must be positive")
    if selected_rows == rows:
        return np.arange(rows, dtype=np.int64)
    indices = (
        np.arange(selected_rows, dtype=np.uint64) * np.uint64(rows)
        // np.uint64(selected_rows)
    ).astype(np.int64)
    if indices.size != np.unique(indices).size or indices[-1] >= rows:
        raise ArithmeticError("Round 74 scaler sample indices are invalid")
    return indices


def fit_round74_event_feature_scaler(
    training_event_features: np.ndarray,
    *,
    partition_role: str,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74EventFeatureScaler:
    """Fit bounded robust statistics from unique training events only."""

    if str(partition_role) != "training":
        raise ValueError("Round 74 scaler may only fit the training partition")
    values = np.asarray(training_event_features)
    if values.ndim != 2 or values.shape[1] != len(ROUND74_EVENT_FEATURE_NAMES):
        raise ValueError("Round 74 scaler training matrix dimensions differ")
    if values.shape[0] < 2:
        raise ValueError("Round 74 scaler requires at least two training events")
    if not math.isfinite(float(maximum_fit_rows)) or int(maximum_fit_rows) < 2:
        raise ValueError("Round 74 scaler maximum fit rows are invalid")
    if not np.isfinite(values).all():
        raise ValueError("Round 74 scaler training matrix contains nonfinite values")
    binary = values[:, :ROUND74_EVENT_BINARY_FEATURE_COUNT]
    if np.any((binary != 0.0) & (binary != 1.0)):
        raise ValueError("Round 74 scaler training binary features are invalid")
    if np.any(binary[:, :5].sum(axis=1) != 1.0) or np.any(
        binary[:, 5:8].sum(axis=1) != 1.0
    ):
        raise ValueError("Round 74 scaler training one-hot groups are invalid")

    indices = _uniform_sample_indices(values.shape[0], int(maximum_fit_rows))
    sample = np.asarray(values[indices], dtype=np.float64)
    quantiles = np.quantile(
        sample,
        (0.001, 0.25, 0.5, 0.75, 0.999),
        axis=0,
        method="linear",
    )
    lower, q25, median, q75, upper = quantiles
    standard_deviation = sample.std(axis=0, dtype=np.float64)
    observed_minimum = values.min(axis=0).astype(np.float64)
    observed_maximum = values.max(axis=0).astype(np.float64)
    constant = observed_minimum == observed_maximum
    scale = np.maximum.reduce(
        (
            q75 - q25,
            (upper - lower) / 4.0,
            standard_deviation,
            np.full_like(median, ROUND74_EVENT_SCALER_MINIMUM_SCALE),
        )
    )
    sparse_nonconstant = (upper == lower) & ~constant
    lower[sparse_nonconstant] = observed_minimum[sparse_nonconstant]
    upper[sparse_nonconstant] = observed_maximum[sparse_nonconstant]
    scale[sparse_nonconstant] = np.maximum(
        scale[sparse_nonconstant],
        (
            observed_maximum[sparse_nonconstant]
            - observed_minimum[sparse_nonconstant]
        )
        / 4.0,
    )
    binary_slice = slice(0, ROUND74_EVENT_BINARY_FEATURE_COUNT)
    median[binary_slice] = 0.0
    scale[binary_slice] = 1.0
    lower[binary_slice] = 0.0
    upper[binary_slice] = 1.0
    constant[binary_slice] = False
    sample_index_sha256 = hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()
    return Round74EventFeatureScaler(
        median=median,
        scale=scale,
        lower_clip=lower,
        upper_clip=upper,
        constant_mask=constant,
        fit_input_rows=int(values.shape[0]),
        fit_sample_rows=int(indices.size),
        fit_sample_index_sha256=sample_index_sha256,
    )


__all__ = [
    "ROUND74_EVENT_BINARY_FEATURE_COUNT",
    "ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS",
    "ROUND74_EVENT_SCALER_MINIMUM_SCALE",
    "ROUND74_EVENT_SCALER_SCHEMA_VERSION",
    "ROUND74_EVENT_SCALER_STANDARDIZED_CLIP",
    "Round74EventFeatureScaler",
    "fit_round74_event_feature_scaler",
]
