"""Leak-resistant scaling for the Round 74 causal event representation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np

from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
)


ROUND74_EVENT_SCALER_SCHEMA_VERSION = "round-074-event-feature-scaler-v6"
ROUND74_EVENT_BINARY_FEATURE_COUNT = 11
ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS = 250_000
ROUND74_EVENT_SCALER_STANDARDIZED_CLIP = 12.0
ROUND74_EVENT_SCALER_MINIMUM_SCALE = 1e-6
ROUND74_EVENT_SCALER_SAMPLING_ALGORITHM = "splitmix64-smallest-priority-v1"
ROUND74_EVENT_SCALER_SAMPLING_SEED = 7404
ROUND74_EVENT_SCALER_SOURCE_SCOPES = (
    "unbound_training_matrix",
    "training_partition_all_runs",
    "segmented_optimization_training_runs",
)


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


def _validated_source_provenance(
    *,
    scope: object,
    run_ids: object,
    partition_sha256: object,
    selection_sha256: object,
) -> tuple[str, tuple[str, ...], str, str]:
    selected_scope = str(scope)
    try:
        selected_run_ids = tuple(str(run_id) for run_id in run_ids)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Round 74 scaler source provenance differs") from exc
    selected_partition = str(partition_sha256)
    selected_selection = str(selection_sha256)
    if (
        selected_scope not in ROUND74_EVENT_SCALER_SOURCE_SCOPES
        or len(set(selected_run_ids)) != len(selected_run_ids)
        or any(
            len(run_id) != 32
            or any(character not in "0123456789abcdef" for character in run_id)
            for run_id in selected_run_ids
        )
    ):
        raise ValueError("Round 74 scaler source provenance differs")

    def is_sha256(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    if selected_scope == "unbound_training_matrix":
        valid = (
            not selected_run_ids
            and selected_partition == ""
            and selected_selection == ""
        )
    elif selected_scope == "training_partition_all_runs":
        valid = (
            bool(selected_run_ids)
            and is_sha256(selected_partition)
            and selected_selection == ""
        )
    else:
        valid = (
            bool(selected_run_ids)
            and is_sha256(selected_partition)
            and is_sha256(selected_selection)
        )
    if not valid:
        raise ValueError("Round 74 scaler source provenance differs")
    return (
        selected_scope,
        selected_run_ids,
        selected_partition,
        selected_selection,
    )


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
    fit_source_scope: str = "unbound_training_matrix"
    fit_source_run_ids: tuple[str, ...] = ()
    fit_source_partition_sha256: str = ""
    fit_source_selection_sha256: str = ""
    fit_sampling_algorithm: str = ROUND74_EVENT_SCALER_SAMPLING_ALGORITHM
    fit_sampling_seed: int = ROUND74_EVENT_SCALER_SAMPLING_SEED
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
            self.fit_sampling_algorithm != ROUND74_EVENT_SCALER_SAMPLING_ALGORITHM
            or int(self.fit_sampling_seed) != ROUND74_EVENT_SCALER_SAMPLING_SEED
        ):
            raise ValueError("Round 74 scaler sampling contract differs")
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
        (
            source_scope,
            source_run_ids,
            source_partition_sha256,
            source_selection_sha256,
        ) = _validated_source_provenance(
            scope=self.fit_source_scope,
            run_ids=self.fit_source_run_ids,
            partition_sha256=self.fit_source_partition_sha256,
            selection_sha256=self.fit_source_selection_sha256,
        )
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
        object.__setattr__(self, "fit_source_scope", source_scope)
        object.__setattr__(self, "fit_source_run_ids", source_run_ids)
        object.__setattr__(
            self,
            "fit_source_partition_sha256",
            source_partition_sha256,
        )
        object.__setattr__(
            self,
            "fit_source_selection_sha256",
            source_selection_sha256,
        )

    @property
    def scaler_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values)
        if selected.ndim < 2 or selected.shape[-1] != len(ROUND74_EVENT_FEATURE_NAMES):
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
            "fit_source_scope": self.fit_source_scope,
            "fit_source_run_ids": list(self.fit_source_run_ids),
            "fit_source_run_count": len(self.fit_source_run_ids),
            "fit_source_run_ids_sha256": _canonical_sha256(
                list(self.fit_source_run_ids)
            ),
            "fit_source_partition_sha256": self.fit_source_partition_sha256,
            "fit_source_selection_sha256": self.fit_source_selection_sha256,
            "fit_input_rows": int(self.fit_input_rows),
            "fit_sample_rows": int(self.fit_sample_rows),
            "fit_sample_index_sha256": self.fit_sample_index_sha256,
            "fit_sampling_algorithm": self.fit_sampling_algorithm,
            "fit_sampling_seed": int(self.fit_sampling_seed),
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
        selected = cls(
            schema_version=str(payload["schema_version"]),
            feature_names_sha256=str(payload["feature_names_sha256"]),
            fit_input_rows=int(payload["fit_input_rows"]),
            fit_sample_rows=int(payload["fit_sample_rows"]),
            fit_sample_index_sha256=str(payload["fit_sample_index_sha256"]),
            fit_source_scope=str(payload["fit_source_scope"]),
            fit_source_run_ids=tuple(
                str(run_id)
                for run_id in payload["fit_source_run_ids"]  # type: ignore[union-attr]
            ),
            fit_source_partition_sha256=str(payload["fit_source_partition_sha256"]),
            fit_source_selection_sha256=str(payload["fit_source_selection_sha256"]),
            fit_sampling_algorithm=str(payload["fit_sampling_algorithm"]),
            fit_sampling_seed=int(payload["fit_sampling_seed"]),
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            lower_clip=np.asarray(payload["lower_clip"], dtype=np.float64),
            upper_clip=np.asarray(payload["upper_clip"], dtype=np.float64),
            constant_mask=np.asarray(payload["constant_mask"], dtype=np.bool_),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 scaler static policy differs")
        return selected


def _splitmix64_priorities(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = (
            values
            + np.uint64(ROUND74_EVENT_SCALER_SAMPLING_SEED)
            + np.uint64(0x9E3779B97F4A7C15)
        )
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _smallest_priority_positions(
    priorities: np.ndarray,
    indices: np.ndarray,
    maximum_rows: int,
) -> np.ndarray:
    if priorities.size <= maximum_rows:
        return np.arange(priorities.size, dtype=np.int64)
    cutoff = np.partition(priorities, maximum_rows - 1)[maximum_rows - 1]
    lower = np.flatnonzero(priorities < cutoff)
    equal = np.flatnonzero(priorities == cutoff)
    remaining = maximum_rows - lower.size
    if remaining < 1 or equal.size < remaining:
        raise ArithmeticError("Round 74 scaler priority selection differs")
    selected_equal = equal[np.argsort(indices[equal], kind="stable")[:remaining]]
    selected = np.concatenate((lower, selected_equal))
    if selected.size != maximum_rows:
        raise ArithmeticError("Round 74 scaler reservoir size differs")
    return selected


def _validate_training_chunk(value: object) -> np.ndarray:
    selected = np.asarray(value)
    if selected.ndim != 2 or selected.shape[1] != len(ROUND74_EVENT_FEATURE_NAMES):
        raise ValueError("Round 74 scaler training matrix dimensions differ")
    if not np.isfinite(selected).all():
        raise ValueError("Round 74 scaler training matrix contains nonfinite values")
    binary = selected[:, :ROUND74_EVENT_BINARY_FEATURE_COUNT]
    if np.any((binary != 0.0) & (binary != 1.0)):
        raise ValueError("Round 74 scaler training binary features are invalid")
    if np.any(binary[:, :5].sum(axis=1) != 1.0) or np.any(
        binary[:, 5:8].sum(axis=1) != 1.0
    ):
        raise ValueError("Round 74 scaler training one-hot groups are invalid")
    return np.asarray(selected, dtype=np.float64)


def _scaler_from_sample(
    *,
    sample: np.ndarray,
    sample_indices: np.ndarray,
    input_rows: int,
    observed_minimum: np.ndarray,
    observed_maximum: np.ndarray,
    fit_source_scope: str,
    fit_source_run_ids: tuple[str, ...],
    fit_source_partition_sha256: str,
    fit_source_selection_sha256: str,
) -> Round74EventFeatureScaler:
    order = np.argsort(sample_indices, kind="stable")
    indices = np.ascontiguousarray(sample_indices[order], dtype=np.int64)
    selected = np.ascontiguousarray(sample[order], dtype=np.float64)
    if indices.size != np.unique(indices).size:
        raise ArithmeticError("Round 74 scaler sampled duplicate events")
    quantiles = np.quantile(
        selected,
        (0.001, 0.25, 0.5, 0.75, 0.999),
        axis=0,
        method="linear",
    )
    lower, q25, median, q75, upper = quantiles
    standard_deviation = selected.std(axis=0, dtype=np.float64)
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
        (observed_maximum[sparse_nonconstant] - observed_minimum[sparse_nonconstant])
        / 4.0,
    )
    binary_slice = slice(0, ROUND74_EVENT_BINARY_FEATURE_COUNT)
    median[binary_slice] = 0.0
    scale[binary_slice] = 1.0
    lower[binary_slice] = 0.0
    upper[binary_slice] = 1.0
    constant[binary_slice] = False
    return Round74EventFeatureScaler(
        median=median,
        scale=scale,
        lower_clip=lower,
        upper_clip=upper,
        constant_mask=constant,
        fit_input_rows=int(input_rows),
        fit_sample_rows=int(indices.size),
        fit_sample_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        fit_source_scope=fit_source_scope,
        fit_source_run_ids=fit_source_run_ids,
        fit_source_partition_sha256=fit_source_partition_sha256,
        fit_source_selection_sha256=fit_source_selection_sha256,
    )


def fit_round74_event_feature_scaler_stream(
    training_event_feature_chunks: Iterable[np.ndarray],
    *,
    partition_role: str,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    fit_source_scope: str = "unbound_training_matrix",
    fit_source_run_ids: Iterable[str] = (),
    fit_source_partition_sha256: str = "",
    fit_source_selection_sha256: str = "",
) -> Round74EventFeatureScaler:
    """Fit one bounded, chunk-invariant sample from unique training events."""

    if str(partition_role) != "training":
        raise ValueError("Round 74 scaler may only fit the training partition")
    (
        source_scope,
        source_run_ids,
        source_partition_sha256,
        source_selection_sha256,
    ) = _validated_source_provenance(
        scope=fit_source_scope,
        run_ids=fit_source_run_ids,
        partition_sha256=fit_source_partition_sha256,
        selection_sha256=fit_source_selection_sha256,
    )
    if (
        not math.isfinite(float(maximum_fit_rows))
        or float(maximum_fit_rows) != int(maximum_fit_rows)
        or int(maximum_fit_rows) < 2
    ):
        raise ValueError("Round 74 scaler maximum fit rows are invalid")
    maximum = int(maximum_fit_rows)
    sampled_values = np.empty(
        (0, len(ROUND74_EVENT_FEATURE_NAMES)),
        dtype=np.float64,
    )
    sampled_indices = np.empty(0, dtype=np.int64)
    sampled_priorities = np.empty(0, dtype=np.uint64)
    observed_minimum: np.ndarray | None = None
    observed_maximum: np.ndarray | None = None
    input_rows = 0
    for chunk_value in training_event_feature_chunks:
        chunk = _validate_training_chunk(chunk_value)
        if chunk.shape[0] == 0:
            continue
        chunk_minimum = chunk.min(axis=0)
        chunk_maximum = chunk.max(axis=0)
        observed_minimum = (
            chunk_minimum
            if observed_minimum is None
            else np.minimum(observed_minimum, chunk_minimum)
        )
        observed_maximum = (
            chunk_maximum
            if observed_maximum is None
            else np.maximum(observed_maximum, chunk_maximum)
        )
        chunk_indices = np.arange(
            input_rows,
            input_rows + chunk.shape[0],
            dtype=np.int64,
        )
        chunk_priorities = _splitmix64_priorities(chunk_indices)
        candidate_values = np.concatenate((sampled_values, chunk), axis=0)
        candidate_indices = np.concatenate(
            (sampled_indices, chunk_indices),
            axis=0,
        )
        candidate_priorities = np.concatenate(
            (sampled_priorities, chunk_priorities),
            axis=0,
        )
        positions = _smallest_priority_positions(
            candidate_priorities,
            candidate_indices,
            maximum,
        )
        sampled_values = np.ascontiguousarray(candidate_values[positions])
        sampled_indices = np.ascontiguousarray(candidate_indices[positions])
        sampled_priorities = np.ascontiguousarray(candidate_priorities[positions])
        input_rows += int(chunk.shape[0])
    if (
        input_rows < 2
        or observed_minimum is None
        or observed_maximum is None
        or sampled_values.shape[0] < 2
    ):
        raise ValueError("Round 74 scaler requires at least two training events")
    return _scaler_from_sample(
        sample=sampled_values,
        sample_indices=sampled_indices,
        input_rows=input_rows,
        observed_minimum=observed_minimum,
        observed_maximum=observed_maximum,
        fit_source_scope=source_scope,
        fit_source_run_ids=source_run_ids,
        fit_source_partition_sha256=source_partition_sha256,
        fit_source_selection_sha256=source_selection_sha256,
    )


def fit_round74_event_feature_scaler(
    training_event_features: np.ndarray,
    *,
    partition_role: str,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    fit_source_scope: str = "unbound_training_matrix",
    fit_source_run_ids: Iterable[str] = (),
    fit_source_partition_sha256: str = "",
    fit_source_selection_sha256: str = "",
) -> Round74EventFeatureScaler:
    """Fit bounded robust statistics from unique training events only."""

    values = np.asarray(training_event_features)
    return fit_round74_event_feature_scaler_stream(
        (values,),
        partition_role=partition_role,
        maximum_fit_rows=maximum_fit_rows,
        fit_source_scope=fit_source_scope,
        fit_source_run_ids=fit_source_run_ids,
        fit_source_partition_sha256=fit_source_partition_sha256,
        fit_source_selection_sha256=fit_source_selection_sha256,
    )


__all__ = [
    "ROUND74_EVENT_BINARY_FEATURE_COUNT",
    "ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS",
    "ROUND74_EVENT_SCALER_MINIMUM_SCALE",
    "ROUND74_EVENT_SCALER_SAMPLING_ALGORITHM",
    "ROUND74_EVENT_SCALER_SAMPLING_SEED",
    "ROUND74_EVENT_SCALER_SCHEMA_VERSION",
    "ROUND74_EVENT_SCALER_SOURCE_SCOPES",
    "ROUND74_EVENT_SCALER_STANDARDIZED_CLIP",
    "Round74EventFeatureScaler",
    "fit_round74_event_feature_scaler",
    "fit_round74_event_feature_scaler_stream",
]
