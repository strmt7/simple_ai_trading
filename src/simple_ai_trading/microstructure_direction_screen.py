"""Research-only mirror-equivariant side-superiority LightGBM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_action_features import mirror_microstructure_direction
from .microstructure_barriers import (
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_outcome_lightgbm import (
    _target_arrays,
    _validate_indexes,
    _validate_weights,
)
from .microstructure_shared_action_lightgbm import _train_booster
from .storage import write_json_atomic


DIRECTION_SCREEN_MODEL_SCHEMA_VERSION = "direction-screen-lightgbm-v1"
DIRECTION_SCREEN_MODEL_FAMILY = "mirror_equivariant_binary_side_superiority_lightgbm"
_WEIGHTING_METHODS = frozenset({"uniqueness", "utility_margin"})
_MINIMUM_ROLE_CLASS_ROWS = 256
_MINIMUM_MARGIN_MULTIPLIER = 0.5
_MAXIMUM_MARGIN_MULTIPLIER = 3.0
_MAXIMUM_ARTIFACT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DirectionScreenSpec:
    """Frozen LightGBM parameters shared by all screen variants."""

    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    max_bin: int
    num_boost_round: int
    early_stopping_rounds: int
    gpu_use_dp_required: bool

    def __post_init__(self) -> None:
        integers = (
            self.num_leaves,
            self.max_depth,
            self.min_data_in_leaf,
            self.bagging_freq,
            self.max_bin,
            self.num_boost_round,
            self.early_stopping_rounds,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integers
            )
            or not 0.0 < self.learning_rate <= 0.5
            or not 2 <= self.num_leaves <= 255
            or not 1 <= self.max_depth <= 16
            or self.min_data_in_leaf < 16
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or self.bagging_freq < 0
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 15 <= self.max_bin <= 255
            or self.num_boost_round < 10
            or not 5 <= self.early_stopping_rounds < self.num_boost_round
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("direction-screen LightGBM specification is invalid")


@dataclass(frozen=True)
class TrainedDirectionScreenModel:
    """Hash-bound research model with no execution or promotion authority."""

    schema_version: str
    model_family: str
    variant: str
    feature_set: str
    weighting: str
    spec: DirectionScreenSpec
    source_feature_version: str
    source_feature_names: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    seed: int
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    train_role_rows: int
    train_opportunity_rows: int
    early_stop_role_rows: int
    early_stop_opportunity_rows: int
    train_long_superior_rows: int
    train_short_superior_rows: int
    early_stop_long_superior_rows: int
    early_stop_short_superior_rows: int
    utility_margin_scale_bps: float
    train_weight_multiplier_mean: float
    early_stop_weight_multiplier_mean: float
    best_iteration: int
    model_string: str
    model_sha256: str
    promotion_permitted: bool = False
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class DirectionScreenPrediction:
    """Paired side-superiority estimates from one shared classifier."""

    endpoint_indexes: np.ndarray
    long_superiority_probability: np.ndarray
    short_superiority_probability: np.ndarray
    conditional_long_probability: np.ndarray
    direction_score: np.ndarray
    selected_side: np.ndarray
    promotion_permitted: bool = False
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def __post_init__(self) -> None:
        indexes = np.asarray(self.endpoint_indexes)
        rows = len(indexes)
        vectors = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.long_superiority_probability,
                self.short_superiority_probability,
                self.conditional_long_probability,
                self.direction_score,
            )
        )
        side = np.asarray(self.selected_side)
        authority = (
            self.promotion_permitted,
            self.trading_authority,
            self.execution_claim,
            self.profitability_claim,
            self.portfolio_claim,
            self.leverage_applied,
        )
        if (
            indexes.ndim != 1
            or rows == 0
            or indexes.dtype.kind not in {"i", "u"}
            or any(value.shape != (rows,) for value in vectors)
            or side.shape != (rows,)
            or any(not np.all(np.isfinite(value)) for value in vectors)
            or any(np.any(value < 0.0) or np.any(value > 1.0) for value in vectors[:3])
            or np.any(vectors[3] < -1.0)
            or np.any(vectors[3] > 1.0)
            or not set(np.unique(side)).issubset({-1, 0, 1})
            or not np.array_equal(
                side,
                np.sign(vectors[0] - vectors[1]).astype(np.int8),
            )
            or any(authority)
        ):
            raise ValueError("direction-screen prediction contract is invalid")

    @property
    def rows(self) -> int:
        return len(self.endpoint_indexes)


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


def _model_sha256(model: TrainedDirectionScreenModel) -> str:
    payload = asdict(model)
    payload.pop("model_sha256")
    return _canonical_sha256(payload)


def _selected_feature_indexes(names: Sequence[str]) -> np.ndarray:
    selected = tuple(str(value) for value in names)
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= set(MICROSTRUCTURE_FEATURE_NAMES)
    ):
        raise ValueError("direction-screen selected feature contract is invalid")
    return np.asarray(
        [MICROSTRUCTURE_FEATURE_NAMES.index(name) for name in selected],
        dtype=np.int64,
    )


def _opportunity_mask(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
    long_actual = np.asarray(long_values, dtype=np.float64)
    short_actual = np.asarray(short_values, dtype=np.float64)
    if (
        long_actual.ndim != 1
        or short_actual.shape != long_actual.shape
        or len(long_actual) == 0
        or not np.all(np.isfinite(long_actual))
        or not np.all(np.isfinite(short_actual))
    ):
        raise ValueError("direction-screen utility arrays are invalid")
    return (np.maximum(long_actual, short_actual) > 0.0) & (long_actual != short_actual)


def utility_margin_multiplier(
    long_values: np.ndarray,
    short_values: np.ndarray,
    scale_bps: float,
) -> np.ndarray:
    """Return the frozen bounded economic-separation multiplier."""

    if not np.isfinite(scale_bps) or scale_bps <= 0.0:
        raise ValueError("direction-screen utility-margin scale is invalid")
    margin = np.abs(
        np.asarray(long_values, dtype=np.float64)
        - np.asarray(short_values, dtype=np.float64)
    )
    if margin.ndim != 1 or not np.all(np.isfinite(margin)):
        raise ValueError("direction-screen utility margins are invalid")
    return np.clip(
        margin / float(scale_bps),
        _MINIMUM_MARGIN_MULTIPLIER,
        _MAXIMUM_MARGIN_MULTIPLIER,
    )


def _validate_model(model: TrainedDirectionScreenModel, *, reload: bool) -> None:
    _selected_feature_indexes(model.selected_feature_names)
    claims = (
        model.promotion_permitted,
        model.trading_authority,
        model.execution_claim,
        model.profitability_claim,
        model.portfolio_claim,
        model.leverage_applied,
    )
    if (
        model.schema_version != DIRECTION_SCREEN_MODEL_SCHEMA_VERSION
        or model.model_family != DIRECTION_SCREEN_MODEL_FAMILY
        or not model.variant.strip()
        or not model.feature_set.strip()
        or model.weighting not in _WEIGHTING_METHODS
        or model.source_feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or model.source_feature_names != MICROSTRUCTURE_FEATURE_NAMES
        or isinstance(model.seed, bool)
        or not isinstance(model.seed, int)
        or model.backend_kind not in {"cpu", "opencl"}
        or not model.backend_requested.strip()
        or not model.backend_device.strip()
        or not model.lightgbm_version.strip()
        or min(
            model.train_role_rows,
            model.train_opportunity_rows,
            model.early_stop_role_rows,
            model.early_stop_opportunity_rows,
            model.train_long_superior_rows,
            model.train_short_superior_rows,
            model.early_stop_long_superior_rows,
            model.early_stop_short_superior_rows,
            model.best_iteration,
        )
        <= 0
        or model.train_long_superior_rows + model.train_short_superior_rows
        != model.train_opportunity_rows
        or model.early_stop_long_superior_rows + model.early_stop_short_superior_rows
        != model.early_stop_opportunity_rows
        or not np.isfinite(model.utility_margin_scale_bps)
        or model.utility_margin_scale_bps <= 0.0
        or not _MINIMUM_MARGIN_MULTIPLIER
        <= model.train_weight_multiplier_mean
        <= _MAXIMUM_MARGIN_MULTIPLIER
        or not _MINIMUM_MARGIN_MULTIPLIER
        <= model.early_stop_weight_multiplier_mean
        <= _MAXIMUM_MARGIN_MULTIPLIER
        or not model.model_string.strip()
        or len(model.model_string.encode("utf-8")) > _MAXIMUM_ARTIFACT_BYTES
        or _model_sha256(model) != model.model_sha256
        or any(claims)
    ):
        raise ValueError("direction-screen model contract is invalid")
    if reload:
        try:
            booster = lgb.Booster(model_str=model.model_string)
        except lgb.basic.LightGBMError as exc:
            raise ValueError("direction-screen booster cannot be reloaded") from exc
        if booster.num_feature() != len(model.selected_feature_names):
            raise ValueError("direction-screen booster feature count drifted")


def train_direction_screen_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    early_stop_endpoints: np.ndarray,
    train_sample_weights: np.ndarray,
    early_stop_sample_weights: np.ndarray,
    selected_feature_names: Sequence[str],
    variant: str,
    feature_set: str,
    weighting: str,
    spec: DirectionScreenSpec,
    compute_backend: str,
    seed: int,
) -> TrainedDirectionScreenModel:
    """Fit one prespecified consumed-data direction-screen variant."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if (
        dataset.feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or dataset.feature_names != MICROSTRUCTURE_FEATURE_NAMES
        or weighting not in _WEIGHTING_METHODS
        or not isinstance(spec, DirectionScreenSpec)
        or not isinstance(variant, str)
        or not variant.strip()
        or not isinstance(feature_set, str)
        or not feature_set.strip()
        or not isinstance(compute_backend, str)
        or not compute_backend.strip()
        or isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("direction-screen training contract is invalid")
    feature_names = tuple(str(value) for value in selected_feature_names)
    feature_indexes = _selected_feature_indexes(feature_names)
    train = _validate_indexes(
        train_endpoints,
        rows=dataset.rows,
        label="direction-screen training",
        minimum_rows=1_024,
    )
    early = _validate_indexes(
        early_stop_endpoints,
        rows=dataset.rows,
        label="direction-screen early-stop",
        minimum_rows=512,
    )
    if train[-1] >= early[0]:
        raise ValueError("direction-screen training and early-stop roles overlap")
    train_weights = _validate_weights(
        train_sample_weights,
        rows=len(train),
        label="direction-screen training",
    )
    early_weights = _validate_weights(
        early_stop_sample_weights,
        rows=len(early),
        label="direction-screen early-stop",
    )
    targets, exits = _target_arrays(dataset, barrier_targets, scenario="stress")
    maximum_exit = np.maximum(exits["long"], exits["short"])
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[early[0]]) or any(
        not np.all(np.isfinite(targets[side][role]))
        or np.any(exits[side][role] <= dataset.decision_time_ms[role])
        for side in ("long", "short")
        for role in (train, early)
    ):
        raise ValueError("direction-screen role labels are invalid or overlapping")
    train_long = np.asarray(targets["long"][train], dtype=np.float64)
    train_short = np.asarray(targets["short"][train], dtype=np.float64)
    early_long = np.asarray(targets["long"][early], dtype=np.float64)
    early_short = np.asarray(targets["short"][early], dtype=np.float64)
    train_mask = _opportunity_mask(train_long, train_short)
    early_mask = _opportunity_mask(early_long, early_short)
    train_selected = train[train_mask]
    early_selected = early[early_mask]
    train_long = train_long[train_mask]
    train_short = train_short[train_mask]
    early_long = early_long[early_mask]
    early_short = early_short[early_mask]
    train_weights = np.asarray(train_weights[train_mask], dtype=np.float64)
    early_weights = np.asarray(early_weights[early_mask], dtype=np.float64)
    train_long_label = train_long > train_short
    early_long_label = early_long > early_short
    support = (
        int(np.sum(train_long_label)),
        int(np.sum(~train_long_label)),
        int(np.sum(early_long_label)),
        int(np.sum(~early_long_label)),
    )
    if min(support) < _MINIMUM_ROLE_CLASS_ROWS:
        raise ValueError("direction-screen role class support is insufficient")
    positive_margin = np.abs(train_long - train_short)
    positive_margin = positive_margin[positive_margin > 0.0]
    if len(positive_margin) == 0:
        raise ValueError("direction-screen training utility margin is empty")
    margin_scale = float(np.median(positive_margin))
    if weighting == "utility_margin":
        train_multiplier = utility_margin_multiplier(
            train_long,
            train_short,
            margin_scale,
        )
        early_multiplier = utility_margin_multiplier(
            early_long,
            early_short,
            margin_scale,
        )
    else:
        train_multiplier = np.ones(len(train_selected), dtype=np.float64)
        early_multiplier = np.ones(len(early_selected), dtype=np.float64)
    source = np.asarray(dataset.features, dtype=np.float32)
    train_source = source[train_selected]
    early_source = source[early_selected]
    train_mirror = mirror_microstructure_direction(train_source)
    early_mirror = mirror_microstructure_direction(early_source)
    paired_train = np.concatenate(
        (train_source[:, feature_indexes], train_mirror[:, feature_indexes]),
        axis=0,
    )
    paired_early = np.concatenate(
        (early_source[:, feature_indexes], early_mirror[:, feature_indexes]),
        axis=0,
    )
    paired_train_labels = np.concatenate(
        (train_long_label, ~train_long_label),
    ).astype(np.float32)
    paired_early_labels = np.concatenate(
        (early_long_label, ~early_long_label),
    ).astype(np.float32)
    weighted_train = train_weights * train_multiplier * 0.5
    weighted_early = early_weights * early_multiplier * 0.5
    paired_train_weights = np.concatenate((weighted_train, weighted_train)).astype(
        np.float32
    )
    paired_early_weights = np.concatenate((weighted_early, weighted_early)).astype(
        np.float32
    )
    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("direction-screen LightGBM OpenCL FP64 is required")
    parameters: dict[str, object] = {
        **backend_parameters,
        "learning_rate": float(spec.learning_rate),
        "num_leaves": int(spec.num_leaves),
        "max_depth": int(spec.max_depth),
        "min_data_in_leaf": int(spec.min_data_in_leaf),
        "feature_fraction": float(spec.feature_fraction),
        "bagging_fraction": float(spec.bagging_fraction),
        "bagging_freq": int(spec.bagging_freq),
        "lambda_l1": float(spec.lambda_l1),
        "lambda_l2": float(spec.lambda_l2),
        "max_bin": int(spec.max_bin),
    }
    booster, best_iteration = _train_booster(
        x_train=paired_train,
        y_train=paired_train_labels,
        train_weights=paired_train_weights,
        x_early_stop=paired_early,
        y_early_stop=paired_early_labels,
        early_stop_weights=paired_early_weights,
        parameters=parameters,
        objective="binary",
        metric="binary_logloss",
        num_boost_round=spec.num_boost_round,
        early_stopping_rounds=spec.early_stopping_rounds,
    )
    model = TrainedDirectionScreenModel(
        schema_version=DIRECTION_SCREEN_MODEL_SCHEMA_VERSION,
        model_family=DIRECTION_SCREEN_MODEL_FAMILY,
        variant=variant.strip(),
        feature_set=feature_set.strip(),
        weighting=weighting,
        spec=spec,
        source_feature_version=dataset.feature_version,
        source_feature_names=dataset.feature_names,
        selected_feature_names=feature_names,
        seed=int(seed),
        backend_requested=compute_backend.strip().lower(),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        train_role_rows=len(train),
        train_opportunity_rows=len(train_selected),
        early_stop_role_rows=len(early),
        early_stop_opportunity_rows=len(early_selected),
        train_long_superior_rows=support[0],
        train_short_superior_rows=support[1],
        early_stop_long_superior_rows=support[2],
        early_stop_short_superior_rows=support[3],
        utility_margin_scale_bps=margin_scale,
        train_weight_multiplier_mean=float(np.mean(train_multiplier)),
        early_stop_weight_multiplier_mean=float(np.mean(early_multiplier)),
        best_iteration=best_iteration,
        model_string=booster.model_to_string(num_iteration=best_iteration),
        model_sha256="PENDING",
    )
    model = replace(model, model_sha256=_model_sha256(model))
    _validate_model(model, reload=True)
    return model


def predict_direction_screen_model(
    model: TrainedDirectionScreenModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> DirectionScreenPrediction:
    """Score both action orientations with the same fitted classifier."""

    validate_microstructure_dataset(dataset)
    _validate_model(model, reload=False)
    if (
        dataset.feature_version != model.source_feature_version
        or dataset.feature_names != model.source_feature_names
    ):
        raise ValueError("direction-screen prediction feature contract drifted")
    selected = _validate_indexes(
        endpoints,
        rows=dataset.rows,
        label="direction-screen prediction",
        minimum_rows=1,
    )
    feature_indexes = _selected_feature_indexes(model.selected_feature_names)
    try:
        booster = lgb.Booster(model_str=model.model_string)
    except lgb.basic.LightGBMError as exc:
        raise ValueError("direction-screen booster cannot be reloaded") from exc
    source = np.asarray(dataset.features[selected], dtype=np.float32)
    mirrored = mirror_microstructure_direction(source)
    long_probability = np.asarray(
        booster.predict(
            source[:, feature_indexes],
            num_iteration=model.best_iteration,
        ),
        dtype=np.float64,
    )
    short_probability = np.asarray(
        booster.predict(
            mirrored[:, feature_indexes],
            num_iteration=model.best_iteration,
        ),
        dtype=np.float64,
    )
    denominator = long_probability + short_probability
    if (
        long_probability.shape != (len(selected),)
        or short_probability.shape != (len(selected),)
        or not np.all(np.isfinite(long_probability))
        or not np.all(np.isfinite(short_probability))
        or np.any(long_probability < 0.0)
        or np.any(long_probability > 1.0)
        or np.any(short_probability < 0.0)
        or np.any(short_probability > 1.0)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("direction-screen model produced invalid probabilities")
    conditional_long = long_probability / denominator
    direction_score = long_probability - short_probability
    return DirectionScreenPrediction(
        endpoint_indexes=selected,
        long_superiority_probability=long_probability,
        short_superiority_probability=short_probability,
        conditional_long_probability=conditional_long,
        direction_score=direction_score,
        selected_side=np.sign(direction_score).astype(np.int8),
    )


def save_direction_screen_model(
    path: str | Path,
    model: TrainedDirectionScreenModel,
) -> None:
    """Persist a validated research artifact atomically."""

    _validate_model(model, reload=True)
    write_json_atomic(Path(path), asdict(model), indent=2, sort_keys=True)


def load_direction_screen_model(path: str | Path) -> TrainedDirectionScreenModel:
    """Load and fully validate a research artifact."""

    artifact = Path(path)
    if not artifact.is_file() or artifact.stat().st_size > _MAXIMUM_ARTIFACT_BYTES:
        raise ValueError("direction-screen artifact is missing or oversized")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("direction-screen artifact is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("direction-screen artifact root is invalid")
    try:
        model = TrainedDirectionScreenModel(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            variant=str(payload["variant"]),
            feature_set=str(payload["feature_set"]),
            weighting=str(payload["weighting"]),
            spec=DirectionScreenSpec(**dict(payload["spec"])),
            source_feature_version=str(payload["source_feature_version"]),
            source_feature_names=tuple(payload["source_feature_names"]),
            selected_feature_names=tuple(payload["selected_feature_names"]),
            seed=int(payload["seed"]),
            backend_requested=str(payload["backend_requested"]),
            backend_kind=str(payload["backend_kind"]),
            backend_device=str(payload["backend_device"]),
            lightgbm_version=str(payload["lightgbm_version"]),
            train_role_rows=int(payload["train_role_rows"]),
            train_opportunity_rows=int(payload["train_opportunity_rows"]),
            early_stop_role_rows=int(payload["early_stop_role_rows"]),
            early_stop_opportunity_rows=int(payload["early_stop_opportunity_rows"]),
            train_long_superior_rows=int(payload["train_long_superior_rows"]),
            train_short_superior_rows=int(payload["train_short_superior_rows"]),
            early_stop_long_superior_rows=int(payload["early_stop_long_superior_rows"]),
            early_stop_short_superior_rows=int(
                payload["early_stop_short_superior_rows"]
            ),
            utility_margin_scale_bps=float(payload["utility_margin_scale_bps"]),
            train_weight_multiplier_mean=float(payload["train_weight_multiplier_mean"]),
            early_stop_weight_multiplier_mean=float(
                payload["early_stop_weight_multiplier_mean"]
            ),
            best_iteration=int(payload["best_iteration"]),
            model_string=str(payload["model_string"]),
            model_sha256=str(payload["model_sha256"]),
            promotion_permitted=bool(payload.get("promotion_permitted", False)),
            trading_authority=bool(payload.get("trading_authority", False)),
            execution_claim=bool(payload.get("execution_claim", False)),
            profitability_claim=bool(payload.get("profitability_claim", False)),
            portfolio_claim=bool(payload.get("portfolio_claim", False)),
            leverage_applied=bool(payload.get("leverage_applied", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("direction-screen artifact fields are invalid") from exc
    _validate_model(model, reload=True)
    return model


__all__ = [
    "DIRECTION_SCREEN_MODEL_FAMILY",
    "DIRECTION_SCREEN_MODEL_SCHEMA_VERSION",
    "DirectionScreenPrediction",
    "DirectionScreenSpec",
    "TrainedDirectionScreenModel",
    "load_direction_screen_model",
    "predict_direction_screen_model",
    "save_direction_screen_model",
    "train_direction_screen_model",
    "utility_margin_multiplier",
]
