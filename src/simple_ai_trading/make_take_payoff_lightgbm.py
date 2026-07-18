"""Shared conditional mean and lower-tail models for make/take actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np

from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .make_take_action_features import (
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MakeTakeActionFeatureBatch,
)
from .make_take_payoff_panel import (
    MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
    MAKE_TAKE_PAYOFF_SYMBOLS,
    MakeTakeConditionalPayoffPanel,
    validate_make_take_conditional_payoff_panel,
)
from .storage import write_json_atomic


MAKE_TAKE_PAYOFF_LIGHTGBM_SCHEMA_VERSION = "make-take-conditional-payoff-lightgbm-v1"
MAKE_TAKE_PAYOFF_ENSEMBLE_SCHEMA_VERSION = "make-take-payoff-ensemble-v1"
MAKE_TAKE_PAYOFF_MODEL_FAMILY = "shared_four_action_conditional_mean_q20"
MAKE_TAKE_PAYOFF_HEADS = ("conditional_mean", "conditional_q20")
MAKE_TAKE_PAYOFF_SEEDS = (5701, 5702, 5703)
_ACTION_CODES = np.arange(4, dtype=np.uint8)
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class MakeTakePayoffLightGBMSpec:
    candidate_id: str = "queue_censored_make_take_lightgbm"
    family: str = MAKE_TAKE_PAYOFF_MODEL_FAMILY
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 8
    min_data_in_leaf: int = 256
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127
    num_boost_round: int = 1_000
    early_stopping_rounds: int = 50
    lower_quantile: float = 0.20
    minimum_training_rows_per_action_symbol: int = 128
    minimum_early_stop_rows_per_action_symbol: int = 32
    minimum_calibration_rows_per_action_symbol: int = 32
    minimum_relative_quality_improvement: float = 0.0025
    gpu_use_dp_required: bool = True

    def __post_init__(self) -> None:
        integral = (
            self.num_leaves,
            self.max_depth,
            self.min_data_in_leaf,
            self.bagging_freq,
            self.max_bin,
            self.num_boost_round,
            self.early_stopping_rounds,
            self.minimum_training_rows_per_action_symbol,
            self.minimum_early_stop_rows_per_action_symbol,
            self.minimum_calibration_rows_per_action_symbol,
        )
        numeric = (
            self.learning_rate,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
            self.lower_quantile,
            self.minimum_relative_quality_improvement,
        )
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or self.family != MAKE_TAKE_PAYOFF_MODEL_FAMILY
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in integral
            )
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in numeric
            )
            or not all(math.isfinite(float(value)) for value in numeric)
            or not 0.0 < self.learning_rate <= 0.25
            or not 2 <= self.num_leaves <= 255
            or not 1 <= self.max_depth <= 16
            or not 2 <= self.min_data_in_leaf <= 65_536
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= self.bagging_freq <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= self.max_bin <= 255
            or not 10 <= self.num_boost_round <= 10_000
            or not 5 <= self.early_stopping_rounds < self.num_boost_round
            or not 0.05 <= self.lower_quantile <= 0.40
            or self.minimum_training_rows_per_action_symbol < 2
            or self.minimum_early_stop_rows_per_action_symbol < 2
            or self.minimum_calibration_rows_per_action_symbol < 2
            or not 0.0 <= self.minimum_relative_quality_improvement <= 0.10
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("make/take payoff LightGBM specification is invalid")


@dataclass(frozen=True)
class SymbolActionPayoffSupport:
    symbol: str
    training: tuple[int, int, int, int]
    early_stop: tuple[int, int, int, int]
    calibration: tuple[int, int, int, int]


@dataclass(frozen=True)
class PayoffQualityDiagnostics:
    early_stop_rows: int
    weighted_mean_rmse_bps: float
    baseline_mean_rmse_bps: float
    weighted_q20_pinball_bps: float
    baseline_q20_pinball_bps: float
    adverse_markout_5s_rows: int
    adverse_markout_5s_mean_rmse_bps: float
    adverse_markout_5s_baseline_rmse_bps: float
    adverse_markout_15s_rows: int
    adverse_markout_15s_mean_rmse_bps: float
    adverse_markout_15s_baseline_rmse_bps: float
    quality_gate_passed: bool


@dataclass(frozen=True)
class TrainedMakeTakePayoffLightGBMModel:
    schema_version: str
    model_family: str
    spec: MakeTakePayoffLightGBMSpec
    feature_names: tuple[str, ...]
    source_feature_spec_sha256: str
    source_dataset_sha256_by_symbol: tuple[tuple[str, str], ...]
    training_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    early_stop_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    calibration_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    training_end_ms: int
    early_stop_start_ms: int
    early_stop_end_ms: int
    calibration_start_ms: int
    action_support: tuple[SymbolActionPayoffSupport, ...]
    training_baseline_mean_bps: tuple[float, float, float, float]
    training_baseline_q20_bps: tuple[float, float, float, float]
    mean_calibration_offset_bps: tuple[float, float, float, float]
    q20_calibration_offset_bps: tuple[float, float, float, float]
    q20_calibration_coverage: tuple[float, float, float, float]
    early_quality: PayoffQualityDiagnostics
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    best_iterations: tuple[int, int]
    model_strings: tuple[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class TrainedMakeTakePayoffLightGBMEnsemble:
    schema_version: str
    members: tuple[TrainedMakeTakePayoffLightGBMModel, ...]
    early_quality: PayoffQualityDiagnostics
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def spec(self) -> MakeTakePayoffLightGBMSpec:
        return self.members[0].spec

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.members[0].feature_names

    @property
    def source_feature_spec_sha256(self) -> str:
        return self.members[0].source_feature_spec_sha256

    @property
    def source_dataset_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].source_dataset_sha256_by_symbol

    @property
    def training_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].training_panel_sha256_by_symbol

    @property
    def early_stop_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].early_stop_panel_sha256_by_symbol

    @property
    def calibration_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].calibration_panel_sha256_by_symbol


MakeTakePayoffLightGBMArtifact = (
    TrainedMakeTakePayoffLightGBMModel | TrainedMakeTakePayoffLightGBMEnsemble
)


@dataclass(frozen=True)
class MakeTakePayoffPredictionBatch:
    source_action_feature_sha256: str
    model_sha256: str
    symbol: str
    event_index: np.ndarray
    decision_time_ms: np.ndarray
    action_code: np.ndarray
    action_side: np.ndarray
    conditional_mean_bps: np.ndarray
    conditional_q20_bps: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.action_code.size)

    def __post_init__(self) -> None:
        rows = self.rows
        vectors = (
            self.event_index,
            self.decision_time_ms,
            self.action_code,
            self.action_side,
            self.conditional_mean_bps,
            self.conditional_q20_bps,
        )
        if (
            rows <= 0
            or self.symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
            or not _is_sha256(self.source_action_feature_sha256)
            or not _is_sha256(self.model_sha256)
            or any(np.asarray(value).shape != (rows,) for value in vectors)
            or np.any(np.diff(self.decision_time_ms) < 0)
            or np.any(self.event_index < 0)
            or not np.all(np.isin(self.action_code, _ACTION_CODES))
            or not np.all(np.isin(self.action_side, (-1, 1)))
            or not np.array_equal(
                self.action_side,
                np.where(np.isin(self.action_code, (0, 2)), 1, -1),
            )
            or not np.isfinite(self.conditional_mean_bps).all()
            or not np.isfinite(self.conditional_q20_bps).all()
            or np.any(self.conditional_q20_bps > self.conditional_mean_bps + 1e-12)
        ):
            raise ValueError("make/take payoff prediction batch is invalid")


@dataclass(frozen=True)
class MakeTakeConditionalPayoffPredictionBatch:
    source_panel_sha256: str
    model_sha256: str
    symbol: str
    action_code: np.ndarray
    action_side: np.ndarray
    conditional_mean_bps: np.ndarray
    conditional_q20_bps: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.action_code.size)

    def __post_init__(self) -> None:
        rows = self.rows
        vectors = (
            self.action_code,
            self.action_side,
            self.conditional_mean_bps,
            self.conditional_q20_bps,
        )
        if (
            rows <= 0
            or self.symbol not in MAKE_TAKE_PAYOFF_SYMBOLS
            or not _is_sha256(self.source_panel_sha256)
            or not _is_sha256(self.model_sha256)
            or any(np.asarray(value).shape != (rows,) for value in vectors)
            or not np.all(np.isin(self.action_code, _ACTION_CODES))
            or not np.array_equal(
                self.action_side,
                np.where(np.isin(self.action_code, (0, 2)), 1, -1),
            )
            or not np.isfinite(self.conditional_mean_bps).all()
            or not np.isfinite(self.conditional_q20_bps).all()
            or np.any(self.conditional_q20_bps > self.conditional_mean_bps + 1e-12)
        ):
            raise ValueError("make/take conditional payoff prediction batch is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _model_payload(model: TrainedMakeTakePayoffLightGBMModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedMakeTakePayoffLightGBMModel) -> str:
    return hashlib.sha256(_canonical_json(_model_payload(model)).encode("ascii")).hexdigest()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    importance = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 1
        or data.shape != importance.shape
        or data.size == 0
        or not np.isfinite(data).all()
        or not np.isfinite(importance).all()
        or np.any(importance <= 0.0)
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError("weighted quantile source is invalid")
    ordering = np.argsort(data, kind="stable")
    ordered = data[ordering]
    cumulative = np.cumsum(importance[ordering], dtype=np.float64)
    threshold = float(quantile) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), data.size - 1)
    return float(ordered[index])


def _weighted_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sqrt(np.average(np.square(truth - prediction), weights=weights)))


def _weighted_pinball(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    residual = truth - prediction
    loss = np.maximum(alpha * residual, (alpha - 1.0) * residual)
    return float(np.average(loss, weights=weights))


def _ordered_panels(
    panels: Sequence[MakeTakeConditionalPayoffPanel],
    *,
    role: str,
) -> tuple[MakeTakeConditionalPayoffPanel, ...]:
    values = tuple(panels)
    try:
        for panel in values:
            validate_make_take_conditional_payoff_panel(panel)
    except ValueError as exc:
        raise ValueError(f"make/take payoff {role} panel contract is invalid") from exc
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {panel.symbol for panel in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
        or any(panel.schema_version != MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION for panel in values)
    ):
        raise ValueError(f"make/take payoff {role} panel contract is invalid")
    ordered = tuple(sorted(values, key=lambda panel: panel.symbol))
    feature_names = ordered[0].feature_names
    feature_spec = ordered[0].source_feature_spec_sha256
    if any(
        panel.feature_names != feature_names
        or panel.source_feature_spec_sha256 != feature_spec
        for panel in ordered[1:]
    ):
        raise ValueError(f"make/take payoff {role} feature contracts drifted")
    return ordered


def _role_support(
    panels: tuple[MakeTakeConditionalPayoffPanel, ...],
) -> dict[str, tuple[int, int, int, int]]:
    return {
        panel.symbol: tuple(
            int(np.count_nonzero(panel.action_code == action)) for action in range(4)
        )
        for panel in panels
    }


def _validate_support(
    support: tuple[SymbolActionPayoffSupport, ...],
    spec: MakeTakePayoffLightGBMSpec,
) -> None:
    if tuple(item.symbol for item in support) != tuple(sorted(MAKE_TAKE_PAYOFF_SYMBOLS)):
        raise ValueError("make/take payoff symbol support contract is invalid")
    minimums = (
        ("training", spec.minimum_training_rows_per_action_symbol),
        ("early_stop", spec.minimum_early_stop_rows_per_action_symbol),
        ("calibration", spec.minimum_calibration_rows_per_action_symbol),
    )
    for item in support:
        for role, minimum in minimums:
            values = getattr(item, role)
            if (
                len(values) != 4
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral)
                    or value < minimum
                    for value in values
                )
            ):
                raise ValueError("make/take payoff per-action support is insufficient")


def _pooled_data(
    panels: tuple[MakeTakeConditionalPayoffPanel, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(sum(panel.rows for panel in panels))
    weights = tuple(
        np.full(panel.rows, total / (len(panels) * panel.rows), dtype=np.float32)
        for panel in panels
    )
    return (
        np.concatenate([panel.features for panel in panels], axis=0).astype(
            np.float32, copy=False
        ),
        np.concatenate([panel.net_bps for panel in panels]).astype(np.float32, copy=False),
        np.concatenate([panel.action_code for panel in panels]).astype(np.uint8, copy=False),
        np.concatenate([panel.markout_5s_bps for panel in panels]).astype(
            np.float32, copy=False
        ),
        np.concatenate([panel.markout_15s_bps for panel in panels]).astype(
            np.float32, copy=False
        ),
        np.concatenate(weights),
    )


def _action_baselines(
    targets: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    quantiles: list[float] = []
    for action in range(4):
        mask = actions == action
        means.append(_weighted_mean(targets[mask], weights[mask]))
        quantiles.append(_weighted_quantile(targets[mask], weights[mask], alpha))
    return tuple(means), tuple(quantiles)


def _quality_diagnostics(
    *,
    truth: np.ndarray,
    mean_prediction: np.ndarray,
    q20_prediction: np.ndarray,
    baseline_mean_prediction: np.ndarray,
    baseline_q20_prediction: np.ndarray,
    markout_5s_bps: np.ndarray,
    markout_15s_bps: np.ndarray,
    weights: np.ndarray,
    spec: MakeTakePayoffLightGBMSpec,
) -> PayoffQualityDiagnostics:
    mean_rmse = _weighted_rmse(truth, mean_prediction, weights)
    baseline_mean_rmse = _weighted_rmse(truth, baseline_mean_prediction, weights)
    q20_pinball = _weighted_pinball(truth, q20_prediction, weights, spec.lower_quantile)
    baseline_q20_pinball = _weighted_pinball(
        truth, baseline_q20_prediction, weights, spec.lower_quantile
    )

    def adverse(values: np.ndarray) -> tuple[int, float, float]:
        mask = values < 0.0
        rows = int(np.count_nonzero(mask))
        if rows == 0:
            return 0, 0.0, 0.0
        return (
            rows,
            _weighted_rmse(truth[mask], mean_prediction[mask], weights[mask]),
            _weighted_rmse(truth[mask], baseline_mean_prediction[mask], weights[mask]),
        )

    adverse_5 = adverse(markout_5s_bps)
    adverse_15 = adverse(markout_15s_bps)
    improvement = spec.minimum_relative_quality_improvement
    gate = bool(
        baseline_mean_rmse > 0.0
        and baseline_q20_pinball > 0.0
        and mean_rmse <= baseline_mean_rmse * (1.0 - improvement)
        and q20_pinball <= baseline_q20_pinball * (1.0 - improvement)
        and adverse_5[0] > 0
        and adverse_15[0] > 0
        and adverse_5[1] <= adverse_5[2]
        and adverse_15[1] <= adverse_15[2]
    )
    return PayoffQualityDiagnostics(
        early_stop_rows=int(truth.size),
        weighted_mean_rmse_bps=mean_rmse,
        baseline_mean_rmse_bps=baseline_mean_rmse,
        weighted_q20_pinball_bps=q20_pinball,
        baseline_q20_pinball_bps=baseline_q20_pinball,
        adverse_markout_5s_rows=adverse_5[0],
        adverse_markout_5s_mean_rmse_bps=adverse_5[1],
        adverse_markout_5s_baseline_rmse_bps=adverse_5[2],
        adverse_markout_15s_rows=adverse_15[0],
        adverse_markout_15s_mean_rmse_bps=adverse_15[1],
        adverse_markout_15s_baseline_rmse_bps=adverse_15[2],
        quality_gate_passed=gate,
    )


def _validate_quality(value: PayoffQualityDiagnostics) -> None:
    integral = (
        value.early_stop_rows,
        value.adverse_markout_5s_rows,
        value.adverse_markout_15s_rows,
    )
    numeric = tuple(
        field_value
        for field_name, field_value in value.__dict__.items()
        if field_name not in {
            "early_stop_rows",
            "adverse_markout_5s_rows",
            "adverse_markout_15s_rows",
            "quality_gate_passed",
        }
    )
    if (
        any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, Integral)
            or item < 0
            for item in integral
        )
        or not all(math.isfinite(float(item)) and float(item) >= 0.0 for item in numeric)
        or not isinstance(value.quality_gate_passed, bool)
    ):
        raise ValueError("make/take payoff quality diagnostics are invalid")


def _validate_model(
    model: TrainedMakeTakePayoffLightGBMModel,
    *,
    reload: bool,
) -> None:
    symbols = tuple(sorted(MAKE_TAKE_PAYOFF_SYMBOLS))
    source_symbols = tuple(symbol for symbol, _sha in model.source_dataset_sha256_by_symbol)
    tuple4 = (
        model.training_baseline_mean_bps,
        model.training_baseline_q20_bps,
        model.mean_calibration_offset_bps,
        model.q20_calibration_offset_bps,
        model.q20_calibration_coverage,
    )
    timeline = (
        model.training_end_ms,
        model.early_stop_start_ms,
        model.early_stop_end_ms,
        model.calibration_start_ms,
    )
    if (
        model.schema_version != MAKE_TAKE_PAYOFF_LIGHTGBM_SCHEMA_VERSION
        or model.model_family != MAKE_TAKE_PAYOFF_MODEL_FAMILY
        or model.spec.family != MAKE_TAKE_PAYOFF_MODEL_FAMILY
        or not model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or any(not isinstance(name, str) or not name.strip() for name in model.feature_names)
        or not _is_sha256(model.source_feature_spec_sha256)
        or source_symbols != symbols
        or any(not _is_sha256(sha) for _symbol, sha in model.source_dataset_sha256_by_symbol)
        or any(
            tuple(symbol for symbol, _sha in values) != symbols
            or any(not _is_sha256(sha) for _symbol, sha in values)
            for values in (
                model.training_panel_sha256_by_symbol,
                model.early_stop_panel_sha256_by_symbol,
                model.calibration_panel_sha256_by_symbol,
            )
        )
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in timeline
        )
        or not model.training_end_ms < model.early_stop_start_ms
        or not model.early_stop_end_ms < model.calibration_start_ms
        or any(len(values) != 4 for values in tuple4)
        or not all(math.isfinite(float(value)) for values in tuple4 for value in values)
        or any(not 0.0 <= value <= 1.0 for value in model.q20_calibration_coverage)
        or model.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
        or not all(
            isinstance(value, str) and value.strip()
            for value in (
                model.backend_requested,
                model.backend_device,
                model.lightgbm_version,
            )
        )
        or isinstance(model.seed, (bool, np.bool_))
        or not isinstance(model.seed, Integral)
        or model.seed not in MAKE_TAKE_PAYOFF_SEEDS
        or len(model.best_iterations) != len(MAKE_TAKE_PAYOFF_HEADS)
        or len(model.model_strings) != len(MAKE_TAKE_PAYOFF_HEADS)
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or not 1 <= value <= model.spec.num_boost_round
            for value in model.best_iterations
        )
        or any(not isinstance(value, str) or not value.strip() for value in model.model_strings)
        or model.trading_authority is not False
        or model.execution_claim is not False
        or model.profitability_claim is not False
        or model.portfolio_claim is not False
        or model.leverage_applied is not False
        or not _is_sha256(model.model_sha256)
        or model.model_sha256 != _model_sha256(model)
    ):
        raise ValueError("make/take payoff LightGBM model contract is invalid")
    _validate_support(model.action_support, model.spec)
    _validate_quality(model.early_quality)
    if reload:
        try:
            for model_string in model.model_strings:
                booster = lgb.Booster(model_str=model_string)
                if booster.num_feature() != len(model.feature_names):
                    raise ValueError("make/take payoff booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("make/take payoff booster payload cannot be reloaded") from exc


def _ensemble_payload(
    model: TrainedMakeTakePayoffLightGBMEnsemble,
) -> dict[str, object]:
    return {
        "schema_version": model.schema_version,
        "member_model_sha256": [member.model_sha256 for member in model.members],
        "member_seeds": [member.seed for member in model.members],
        "early_quality": asdict(model.early_quality),
        "trading_authority": model.trading_authority,
        "execution_claim": model.execution_claim,
        "profitability_claim": model.profitability_claim,
        "portfolio_claim": model.portfolio_claim,
        "leverage_applied": model.leverage_applied,
    }


def _ensemble_sha256(model: TrainedMakeTakePayoffLightGBMEnsemble) -> str:
    return hashlib.sha256(
        _canonical_json(_ensemble_payload(model)).encode("ascii")
    ).hexdigest()


def _validate_ensemble_members(
    members: tuple[TrainedMakeTakePayoffLightGBMModel, ...],
    *,
    reload: bool,
) -> None:
    for member in members:
        _validate_model(member, reload=reload)
    shared_fields = (
        "spec",
        "feature_names",
        "source_feature_spec_sha256",
        "source_dataset_sha256_by_symbol",
        "training_panel_sha256_by_symbol",
        "early_stop_panel_sha256_by_symbol",
        "calibration_panel_sha256_by_symbol",
        "training_end_ms",
        "early_stop_start_ms",
        "early_stop_end_ms",
        "calibration_start_ms",
        "action_support",
        "training_baseline_mean_bps",
        "training_baseline_q20_bps",
        "backend_requested",
        "backend_kind",
        "backend_device",
        "lightgbm_version",
    )
    if (
        len(members) != len(MAKE_TAKE_PAYOFF_SEEDS)
        or tuple(member.seed for member in members) != MAKE_TAKE_PAYOFF_SEEDS
        or any(
            getattr(member, field) != getattr(members[0], field)
            for member in members[1:]
            for field in shared_fields
        )
        or len({member.model_sha256 for member in members}) != len(members)
    ):
        raise ValueError("make/take payoff LightGBM ensemble members are invalid")


def _validate_ensemble(
    model: TrainedMakeTakePayoffLightGBMEnsemble,
    *,
    reload: bool,
) -> None:
    _validate_ensemble_members(tuple(model.members), reload=reload)
    if (
        model.schema_version != MAKE_TAKE_PAYOFF_ENSEMBLE_SCHEMA_VERSION
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or not _is_sha256(model.model_sha256)
        or _ensemble_sha256(model) != model.model_sha256
    ):
        raise ValueError("make/take payoff LightGBM ensemble contract is invalid")
    _validate_quality(model.early_quality)


def validate_make_take_payoff_lightgbm_model(
    model: MakeTakePayoffLightGBMArtifact,
    *,
    reload: bool = False,
) -> None:
    """Public artifact validator used by downstream evidence gates."""

    if isinstance(model, TrainedMakeTakePayoffLightGBMEnsemble):
        _validate_ensemble(model, reload=reload)
        return
    _validate_model(model, reload=reload)


def _train_head(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_weight: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    early_weight: np.ndarray,
    parameters: dict[str, object],
    feature_names: tuple[str, ...],
    spec: MakeTakePayoffLightGBMSpec,
    head: str,
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weight,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        x_early,
        label=y_early,
        weight=early_weight,
        feature_name=list(feature_names),
        reference=training,
        free_raw_data=False,
    )
    objective = {"objective": "regression", "metric": "l2"}
    if head == "conditional_q20":
        objective = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": float(spec.lower_quantile),
        }
    booster = lgb.train(
        {**parameters, **objective},
        training,
        num_boost_round=spec.num_boost_round,
        valid_sets=[early],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    iteration = int(booster.best_iteration or spec.num_boost_round)
    if not 1 <= iteration <= spec.num_boost_round:
        raise RuntimeError("make/take payoff booster selected an invalid iteration")
    return booster, iteration


def train_make_take_payoff_lightgbm_model(
    *,
    training_panels: Sequence[MakeTakeConditionalPayoffPanel],
    early_stop_panels: Sequence[MakeTakeConditionalPayoffPanel],
    calibration_panels: Sequence[MakeTakeConditionalPayoffPanel],
    spec: MakeTakePayoffLightGBMSpec = MakeTakePayoffLightGBMSpec(),
    compute_backend: str = "auto",
    seed: int = MAKE_TAKE_PAYOFF_SEEDS[0],
    progress: Callable[[int, int], None] | None = None,
) -> TrainedMakeTakePayoffLightGBMModel:
    """Train shared payoff heads on purged chronological BTC/ETH/SOL roles."""

    train = _ordered_panels(training_panels, role="training")
    early = _ordered_panels(early_stop_panels, role="early-stop")
    calibration = _ordered_panels(calibration_panels, role="calibration")
    feature_names = train[0].feature_names
    feature_spec = train[0].source_feature_spec_sha256
    if any(
        panel.feature_names != feature_names
        or panel.source_feature_spec_sha256 != feature_spec
        for panel in (*early, *calibration)
    ):
        raise ValueError("make/take payoff role feature contracts drifted")
    source_by_role = tuple(
        tuple((panel.symbol, panel.source_dataset_sha256) for panel in panels)
        for panels in (train, early, calibration)
    )
    if source_by_role[0] != source_by_role[1] or source_by_role[0] != source_by_role[2]:
        raise ValueError("make/take payoff source datasets drifted across roles")
    training_end = max(panel.source_label_end_ms for panel in train)
    early_start = min(panel.source_first_decision_time_ms for panel in early)
    early_end = max(panel.source_label_end_ms for panel in early)
    calibration_start = min(panel.source_first_decision_time_ms for panel in calibration)
    if not training_end < early_start or not early_end < calibration_start:
        raise ValueError("make/take payoff chronological roles overlap")
    support_by_role = tuple(_role_support(panels) for panels in (train, early, calibration))
    support = tuple(
        SymbolActionPayoffSupport(
            symbol=symbol,
            training=support_by_role[0][symbol],
            early_stop=support_by_role[1][symbol],
            calibration=support_by_role[2][symbol],
        )
        for symbol in sorted(MAKE_TAKE_PAYOFF_SYMBOLS)
    )
    _validate_support(support, spec)
    selected_seed = int(seed)
    if isinstance(seed, (bool, np.bool_)) or selected_seed not in MAKE_TAKE_PAYOFF_SEEDS:
        raise ValueError("make/take payoff seed is outside the frozen ensemble")
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend, selected_seed, reproducible=True
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True or backend.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("make/take payoff LightGBM OpenCL FP64 accumulation is required")
    common: dict[str, object] = {
        **backend,
        "learning_rate": float(spec.learning_rate),
        "num_leaves": spec.num_leaves,
        "max_depth": spec.max_depth,
        "min_data_in_leaf": spec.min_data_in_leaf,
        "feature_fraction": float(spec.feature_fraction),
        "bagging_fraction": float(spec.bagging_fraction),
        "bagging_freq": spec.bagging_freq,
        "lambda_l1": float(spec.lambda_l1),
        "lambda_l2": float(spec.lambda_l2),
        "max_bin": spec.max_bin,
    }
    x_train, y_train, action_train, _mark5_train, _mark15_train, weight_train = (
        _pooled_data(train)
    )
    x_early, y_early, action_early, mark5_early, mark15_early, weight_early = (
        _pooled_data(early)
    )
    x_cal, y_cal, action_cal, _mark5_cal, _mark15_cal, weight_cal = _pooled_data(
        calibration
    )
    baseline_mean, baseline_q20 = _action_baselines(
        y_train, action_train, weight_train, spec.lower_quantile
    )
    boosters: list[lgb.Booster] = []
    iterations: list[int] = []
    early_predictions: list[np.ndarray] = []
    calibration_predictions: list[np.ndarray] = []
    for index, head in enumerate(MAKE_TAKE_PAYOFF_HEADS, start=1):
        if progress is not None:
            progress(index, len(MAKE_TAKE_PAYOFF_HEADS))
        booster, iteration = _train_head(
            x_train=x_train,
            y_train=y_train,
            train_weight=weight_train,
            x_early=x_early,
            y_early=y_early,
            early_weight=weight_early,
            parameters=common,
            feature_names=feature_names,
            spec=spec,
            head=head,
        )
        early_prediction = np.asarray(
            booster.predict(x_early, num_iteration=iteration), dtype=np.float64
        )
        calibration_prediction = np.asarray(
            booster.predict(x_cal, num_iteration=iteration), dtype=np.float64
        )
        if (
            early_prediction.shape != y_early.shape
            or calibration_prediction.shape != y_cal.shape
            or not np.isfinite(early_prediction).all()
            or not np.isfinite(calibration_prediction).all()
        ):
            raise RuntimeError("make/take payoff booster prediction is invalid")
        boosters.append(booster)
        iterations.append(iteration)
        early_predictions.append(early_prediction)
        calibration_predictions.append(calibration_prediction)
    baseline_mean_early = np.asarray(baseline_mean, dtype=np.float64)[action_early]
    baseline_q20_early = np.asarray(baseline_q20, dtype=np.float64)[action_early]
    quality = _quality_diagnostics(
        truth=y_early,
        mean_prediction=early_predictions[0],
        q20_prediction=early_predictions[1],
        baseline_mean_prediction=baseline_mean_early,
        baseline_q20_prediction=baseline_q20_early,
        markout_5s_bps=mark5_early,
        markout_15s_bps=mark15_early,
        weights=weight_early,
        spec=spec,
    )
    mean_offsets: list[float] = []
    q20_offsets: list[float] = []
    q20_coverage: list[float] = []
    for action in range(4):
        mask = action_cal == action
        action_weights = weight_cal[mask]
        mean_offsets.append(
            _weighted_mean(y_cal[mask] - calibration_predictions[0][mask], action_weights)
        )
        q20_offset = _weighted_quantile(
            y_cal[mask] - calibration_predictions[1][mask],
            action_weights,
            spec.lower_quantile,
        )
        q20_offsets.append(q20_offset)
        q20_coverage.append(
            _weighted_mean(
                np.asarray(
                    y_cal[mask] <= calibration_predictions[1][mask] + q20_offset,
                    dtype=np.float64,
                ),
                action_weights,
            )
        )
    provisional = TrainedMakeTakePayoffLightGBMModel(
        schema_version=MAKE_TAKE_PAYOFF_LIGHTGBM_SCHEMA_VERSION,
        model_family=MAKE_TAKE_PAYOFF_MODEL_FAMILY,
        spec=spec,
        feature_names=feature_names,
        source_feature_spec_sha256=feature_spec,
        source_dataset_sha256_by_symbol=source_by_role[0],
        training_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in train
        ),
        early_stop_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in early
        ),
        calibration_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in calibration
        ),
        training_end_ms=training_end,
        early_stop_start_ms=early_start,
        early_stop_end_ms=early_end,
        calibration_start_ms=calibration_start,
        action_support=support,
        training_baseline_mean_bps=tuple(float(value) for value in baseline_mean),
        training_baseline_q20_bps=tuple(float(value) for value in baseline_q20),
        mean_calibration_offset_bps=tuple(mean_offsets),
        q20_calibration_offset_bps=tuple(q20_offsets),
        q20_calibration_coverage=tuple(q20_coverage),
        early_quality=quality,
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=selected_seed,
        best_iterations=tuple(iterations),
        model_strings=tuple(
            booster.model_to_string(num_iteration=iteration)
            for booster, iteration in zip(boosters, iterations, strict=True)
        ),
        model_sha256="",
    )
    model = TrainedMakeTakePayoffLightGBMModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload=False)
    return model


def _predict_raw_arrays(
    model: TrainedMakeTakePayoffLightGBMModel,
    features: np.ndarray,
    action_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_model(model, reload=False)
    values = np.asarray(features, dtype=np.float32)
    actions = np.asarray(action_code, dtype=np.uint8)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != len(model.feature_names)
        or actions.shape != (values.shape[0],)
        or not np.all(np.isin(actions, _ACTION_CODES))
        or not np.isfinite(values).all()
    ):
        raise ValueError("make/take payoff prediction matrix is invalid")
    predictions: list[np.ndarray] = []
    try:
        for model_string, iteration in zip(
            model.model_strings, model.best_iterations, strict=True
        ):
            booster = lgb.Booster(model_str=model_string)
            prediction = np.asarray(
                booster.predict(values, num_iteration=iteration),
                dtype=np.float64,
            )
            if prediction.shape != (values.shape[0],) or not np.isfinite(
                prediction
            ).all():
                raise ValueError("make/take payoff booster prediction is invalid")
            predictions.append(prediction)
    except lgb.basic.LightGBMError as exc:
        raise ValueError("make/take payoff booster payload cannot be reloaded") from exc
    return predictions[0], predictions[1]


def _predict_calibrated_arrays(
    model: TrainedMakeTakePayoffLightGBMModel,
    features: np.ndarray,
    action_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean, q20 = _predict_raw_arrays(model, features, action_code)
    actions = np.asarray(action_code, dtype=np.uint8)
    mean = mean + np.asarray(model.mean_calibration_offset_bps)[actions]
    q20 = q20 + np.asarray(model.q20_calibration_offset_bps)[actions]
    return mean, np.minimum(q20, mean)


def build_make_take_payoff_lightgbm_ensemble(
    members: Sequence[TrainedMakeTakePayoffLightGBMModel],
    *,
    early_stop_panels: Sequence[MakeTakeConditionalPayoffPanel],
) -> TrainedMakeTakePayoffLightGBMEnsemble:
    """Bind and score every frozen seed without outcome-based member selection."""

    ordered_members = tuple(sorted(tuple(members), key=lambda member: member.seed))
    _validate_ensemble_members(ordered_members, reload=True)
    early = _ordered_panels(early_stop_panels, role="ensemble early-stop")
    reference = ordered_members[0]
    if (
        tuple((panel.symbol, panel.panel_sha256) for panel in early)
        != reference.early_stop_panel_sha256_by_symbol
        or tuple((panel.symbol, panel.source_dataset_sha256) for panel in early)
        != reference.source_dataset_sha256_by_symbol
        or any(
            panel.feature_names != reference.feature_names
            or panel.source_feature_spec_sha256
            != reference.source_feature_spec_sha256
            for panel in early
        )
    ):
        raise ValueError("make/take payoff ensemble early-stop evidence drifted")
    _features, truth, actions, markout_5s, markout_15s, weights = _pooled_data(early)
    member_mean: list[np.ndarray] = []
    member_q20: list[np.ndarray] = []
    for member in ordered_members:
        per_symbol = tuple(
            _predict_raw_arrays(member, panel.features, panel.action_code)
            for panel in early
        )
        member_mean.append(np.concatenate([values[0] for values in per_symbol]))
        member_q20.append(np.concatenate([values[1] for values in per_symbol]))
    mean_prediction = np.mean(np.stack(member_mean, axis=0), axis=0)
    q20_prediction = np.mean(np.stack(member_q20, axis=0), axis=0)
    baseline_mean = np.asarray(
        reference.training_baseline_mean_bps, dtype=np.float64
    )[actions]
    baseline_q20 = np.asarray(
        reference.training_baseline_q20_bps, dtype=np.float64
    )[actions]
    quality = _quality_diagnostics(
        truth=truth,
        mean_prediction=mean_prediction,
        q20_prediction=q20_prediction,
        baseline_mean_prediction=baseline_mean,
        baseline_q20_prediction=baseline_q20,
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        weights=weights,
        spec=reference.spec,
    )
    provisional = TrainedMakeTakePayoffLightGBMEnsemble(
        schema_version=MAKE_TAKE_PAYOFF_ENSEMBLE_SCHEMA_VERSION,
        members=ordered_members,
        early_quality=quality,
        model_sha256="",
    )
    model = TrainedMakeTakePayoffLightGBMEnsemble(
        **{
            **provisional.__dict__,
            "model_sha256": _ensemble_sha256(provisional),
        }
    )
    _validate_ensemble(model, reload=True)
    return model


def predict_make_take_payoff_lightgbm_model(
    model: MakeTakePayoffLightGBMArtifact,
    *,
    symbol: str,
    action_features: MakeTakeActionFeatureBatch,
) -> MakeTakePayoffPredictionBatch:
    """Predict calibrated conditional values for all four candidate actions."""

    if isinstance(model, TrainedMakeTakePayoffLightGBMEnsemble):
        _validate_ensemble(model, reload=False)
        predictions = tuple(
            predict_make_take_payoff_lightgbm_model(
                member,
                symbol=symbol,
                action_features=action_features,
            )
            for member in model.members
        )
        reference = predictions[0]
        mean = np.mean(
            np.stack(
                [prediction.conditional_mean_bps for prediction in predictions],
                axis=0,
            ),
            axis=0,
        )
        q20 = np.minimum(
            np.mean(
                np.stack(
                    [prediction.conditional_q20_bps for prediction in predictions],
                    axis=0,
                ),
                axis=0,
            ),
            mean,
        )
        retained = tuple(
            np.array(value, copy=True)
            for value in (
                reference.event_index,
                reference.decision_time_ms,
                reference.action_code,
                reference.action_side,
                mean,
                q20,
            )
        )
        for array in retained:
            array.setflags(write=False)
        return MakeTakePayoffPredictionBatch(
            source_action_feature_sha256=reference.source_action_feature_sha256,
            model_sha256=model.model_sha256,
            symbol=reference.symbol,
            event_index=retained[0],
            decision_time_ms=retained[1],
            action_code=retained[2],
            action_side=retained[3],
            conditional_mean_bps=retained[4],
            conditional_q20_bps=retained[5],
        )
    _validate_model(model, reload=False)
    normalized_symbol = str(symbol).strip().upper()
    source = dict(model.source_dataset_sha256_by_symbol)
    action_features.spec.validate()
    expected_action_code = np.tile(_ACTION_CODES, action_features.event_rows)
    expected_action_side = np.tile(
        np.asarray([1, -1, 1, -1], dtype=np.int8), action_features.event_rows
    )
    if (
        normalized_symbol not in source
        or action_features.schema_version != MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION
        or action_features.source_dataset_sha256 != source[normalized_symbol]
        or action_features.spec_sha256 != model.source_feature_spec_sha256
        or action_features.spec_sha256 != action_features.spec.spec_sha256
        or not _is_sha256(action_features.batch_sha256)
        or not _is_sha256(action_features.source_flow_sha256)
        or action_features.feature_names != model.feature_names
        or action_features.event_rows <= 0
        or action_features.action_rows != action_features.event_rows * 4
        or action_features.event_indexes.shape != (action_features.event_rows,)
        or action_features.decision_time_ms.shape != (action_features.event_rows,)
        or action_features.action_code.shape != (action_features.action_rows,)
        or action_features.action_side.shape != (action_features.action_rows,)
        or action_features.eligible.shape != (action_features.action_rows,)
        or action_features.features.shape
        != (action_features.action_rows, len(model.feature_names))
        or action_features.event_indexes[0] < 0
        or np.any(np.diff(action_features.event_indexes) <= 0)
        or np.any(np.diff(action_features.decision_time_ms) <= 0)
        or not np.array_equal(action_features.action_code, expected_action_code)
        or not np.array_equal(action_features.action_side, expected_action_side)
        or not np.isfinite(action_features.features).all()
    ):
        raise ValueError("make/take payoff prediction feature contract drifted")
    action_code = np.array(action_features.action_code, dtype=np.uint8, copy=True)
    action_side = np.array(action_features.action_side, dtype=np.int8, copy=True)
    mean, q20 = _predict_calibrated_arrays(
        model,
        action_features.features,
        action_code,
    )
    event_index = np.repeat(action_features.event_indexes, 4).astype(np.int64, copy=False)
    decision_time = np.repeat(action_features.decision_time_ms, 4).astype(
        np.int64, copy=False
    )
    retained = (event_index, decision_time, action_code, action_side, mean, q20)
    for array in retained:
        array.setflags(write=False)
    return MakeTakePayoffPredictionBatch(
        source_action_feature_sha256=action_features.batch_sha256,
        model_sha256=model.model_sha256,
        symbol=normalized_symbol,
        event_index=event_index,
        decision_time_ms=decision_time,
        action_code=action_code,
        action_side=action_side,
        conditional_mean_bps=mean,
        conditional_q20_bps=q20,
    )


def predict_make_take_conditional_payoff_panel(
    model: MakeTakePayoffLightGBMArtifact,
    panel: MakeTakeConditionalPayoffPanel,
) -> MakeTakeConditionalPayoffPredictionBatch:
    """Predict a conditional-payoff panel without requiring unfilled action rows."""

    if isinstance(model, TrainedMakeTakePayoffLightGBMEnsemble):
        _validate_ensemble(model, reload=False)
        predictions = tuple(
            predict_make_take_conditional_payoff_panel(member, panel)
            for member in model.members
        )
        mean = np.mean(
            np.stack(
                [prediction.conditional_mean_bps for prediction in predictions],
                axis=0,
            ),
            axis=0,
        )
        q20 = np.minimum(
            np.mean(
                np.stack(
                    [prediction.conditional_q20_bps for prediction in predictions],
                    axis=0,
                ),
                axis=0,
            ),
            mean,
        )
        action_code = np.array(panel.action_code, dtype=np.uint8, copy=True)
        action_side = np.array(panel.action_side, dtype=np.int8, copy=True)
        for array in (action_code, action_side, mean, q20):
            array.setflags(write=False)
        return MakeTakeConditionalPayoffPredictionBatch(
            source_panel_sha256=panel.panel_sha256,
            model_sha256=model.model_sha256,
            symbol=panel.symbol,
            action_code=action_code,
            action_side=action_side,
            conditional_mean_bps=mean,
            conditional_q20_bps=q20,
        )
    _validate_model(model, reload=False)
    validate_make_take_conditional_payoff_panel(panel)
    source = dict(model.source_dataset_sha256_by_symbol)
    if (
        panel.symbol not in source
        or panel.source_dataset_sha256 != source[panel.symbol]
        or panel.source_feature_spec_sha256 != model.source_feature_spec_sha256
        or panel.feature_names != model.feature_names
    ):
        raise ValueError("make/take conditional payoff panel drifted")
    action_code = np.array(panel.action_code, dtype=np.uint8, copy=True)
    action_side = np.array(panel.action_side, dtype=np.int8, copy=True)
    mean, q20 = _predict_calibrated_arrays(model, panel.features, action_code)
    for array in (action_code, action_side, mean, q20):
        array.setflags(write=False)
    return MakeTakeConditionalPayoffPredictionBatch(
        source_panel_sha256=panel.panel_sha256,
        model_sha256=model.model_sha256,
        symbol=panel.symbol,
        action_code=action_code,
        action_side=action_side,
        conditional_mean_bps=mean,
        conditional_q20_bps=q20,
    )


def save_make_take_payoff_lightgbm_model(
    path: str | Path,
    model: TrainedMakeTakePayoffLightGBMModel,
) -> None:
    _validate_model(model, reload=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("make/take payoff model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def _support_from_payload(value: object) -> tuple[SymbolActionPayoffSupport, ...]:
    if not isinstance(value, list):
        raise ValueError("make/take payoff support payload is invalid")
    output: list[SymbolActionPayoffSupport] = []
    try:
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("symbol"), str):
                raise ValueError("make/take payoff support payload is invalid")
            roles: dict[str, tuple[int, int, int, int]] = {}
            for role in ("training", "early_stop", "calibration"):
                raw = item[role]
                if (
                    not isinstance(raw, list)
                    or len(raw) != 4
                    or any(isinstance(count, bool) or not isinstance(count, int) for count in raw)
                ):
                    raise ValueError("make/take payoff support payload is invalid")
                roles[role] = tuple(raw)
            output.append(
                SymbolActionPayoffSupport(
                    symbol=item["symbol"],
                    training=roles["training"],
                    early_stop=roles["early_stop"],
                    calibration=roles["calibration"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("make/take payoff support payload is invalid") from exc
    return tuple(output)


def load_make_take_payoff_lightgbm_model(
    path: str | Path,
) -> TrainedMakeTakePayoffLightGBMModel:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size <= 0 or size > _MAX_ARTIFACT_BYTES or not source.is_file():
            raise ValueError("make/take payoff artifact size is invalid")
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("make/take payoff model artifact is unreadable") from exc
    expected = {field.name for field in fields(TrainedMakeTakePayoffLightGBMModel)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("make/take payoff model artifact fields drifted")
    try:
        payload["spec"] = MakeTakePayoffLightGBMSpec(**payload["spec"])
        payload["feature_names"] = tuple(payload["feature_names"])
        for field_name in (
            "source_dataset_sha256_by_symbol",
            "training_panel_sha256_by_symbol",
            "early_stop_panel_sha256_by_symbol",
            "calibration_panel_sha256_by_symbol",
        ):
            payload[field_name] = tuple(tuple(value) for value in payload[field_name])
        payload["action_support"] = _support_from_payload(payload["action_support"])
        for field_name in (
            "training_baseline_mean_bps",
            "training_baseline_q20_bps",
            "mean_calibration_offset_bps",
            "q20_calibration_offset_bps",
            "q20_calibration_coverage",
        ):
            payload[field_name] = tuple(float(value) for value in payload[field_name])
        payload["early_quality"] = PayoffQualityDiagnostics(**payload["early_quality"])
        payload["best_iterations"] = tuple(payload["best_iterations"])
        payload["model_strings"] = tuple(payload["model_strings"])
        model = TrainedMakeTakePayoffLightGBMModel(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("make/take payoff model artifact payload is invalid") from exc
    _validate_model(model, reload=True)
    return model


__all__ = [
    "MAKE_TAKE_PAYOFF_HEADS",
    "MAKE_TAKE_PAYOFF_ENSEMBLE_SCHEMA_VERSION",
    "MAKE_TAKE_PAYOFF_LIGHTGBM_SCHEMA_VERSION",
    "MAKE_TAKE_PAYOFF_MODEL_FAMILY",
    "MAKE_TAKE_PAYOFF_SEEDS",
    "MakeTakeConditionalPayoffPredictionBatch",
    "MakeTakePayoffLightGBMArtifact",
    "MakeTakePayoffLightGBMSpec",
    "MakeTakePayoffPredictionBatch",
    "PayoffQualityDiagnostics",
    "SymbolActionPayoffSupport",
    "TrainedMakeTakePayoffLightGBMEnsemble",
    "TrainedMakeTakePayoffLightGBMModel",
    "build_make_take_payoff_lightgbm_ensemble",
    "load_make_take_payoff_lightgbm_model",
    "predict_make_take_conditional_payoff_panel",
    "predict_make_take_payoff_lightgbm_model",
    "save_make_take_payoff_lightgbm_model",
    "train_make_take_payoff_lightgbm_model",
    "validate_make_take_payoff_lightgbm_model",
]
