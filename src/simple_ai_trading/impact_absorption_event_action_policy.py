"""Target-free Round 74 action scoring and tuning-only policy selection.

Forecasts become research candidates here, not orders. Candidate derivation
cannot receive realized targets. A separate selector may replay candidates only
on the predeclared policy-selection runs and uses exact captured entry/exit
times with an explicit shared-capital allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import torch

from .impact_absorption_event_calibration import (
    ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS,
    ROUND74_TUNING_POLICY_SELECTION_RUNS,
    Round74ProbabilityCalibration,
    Round74TuningSubpartition,
    apply_round74_probability_calibration,
    apply_round74_risk_quantile_calibration,
)
from .impact_absorption_event_dataset import (
    ROUND74_EVENT_DATASET_SCHEMA_VERSION,
    ROUND74_EVENT_PARTITION_ROLES,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74EventTrainingBatch,
)
from .impact_absorption_event_financial_metrics import (
    round74_conservative_maximum_drawdown_bps,
    round74_maximum_concurrent_adverse_excursion_bps,
    round74_maximum_realized_drawdown_bps,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
    ROUND74_EVENT_SYMBOLS,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS,
    Round74EventTargetOutcome,
)


ROUND74_ACTION_CONTEXT_SCHEMA_VERSION = "round-074-action-context-v5"
ROUND74_ACTION_EXECUTION_PANEL_SCHEMA_VERSION = "round-074-action-execution-panel-v1"
ROUND74_ACTION_POLICY_SCHEMA_VERSION = "round-074-action-policy-v14"
ROUND74_ACTION_HORIZONS_SECONDS = (30, 300)
ROUND74_ACTION_PROFILES = ("conservative", "regular", "aggressive")
ROUND74_ACTION_DEFAULT_PROFILE = "conservative"
ROUND74_ACTION_POSITION_CAPITAL_FRACTION = 1.0 / len(ROUND74_EVENT_SYMBOLS)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 action {label} digest differs")
    return selected


def _module_sha256(filename: str) -> str:
    payload = (Path(__file__).parent / filename).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _update_array_digest(digest: object, value: np.ndarray) -> None:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(int(canonical.ndim).to_bytes(2, "little", signed=False))
    for size in canonical.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float32).numpy()
    )


def _model_output_sha256(output: Round74EventModelOutput) -> str:
    output.validate(int(output.payoff_quantiles_bps.shape[0]))
    digest = hashlib.sha256(b"round-074-model-output-v2")
    for value in (
        output.payoff_quantiles_bps,
        output.maximum_adverse_excursion_quantiles_bps,
        output.positive_payoff_logits,
        output.adverse_selection_logits,
        output.regime_unpredictability_logits,
    ):
        _update_array_digest(digest, _tensor_array(value))
    diagnostics = output.epistemic_diagnostics
    if diagnostics is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(int(diagnostics.peer_count).to_bytes(4, "little", signed=False))
        for value in (
            diagnostics.payoff_quantile_standard_deviation_bps,
            diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps,
            diagnostics.positive_payoff_probability_standard_deviation,
            diagnostics.adverse_selection_probability_standard_deviation,
            diagnostics.regime_unpredictability_probability_standard_deviation,
        ):
            _update_array_digest(digest, _tensor_array(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class Round74ActionProfileSpec:
    """Predeclared selectivity and tuning gates for one risk appetite."""

    profile: str
    downside_penalty: float
    adverse_excursion_penalty: float
    minimum_positive_probability: float
    maximum_adverse_probability: float
    maximum_unpredictability_probability: float
    lower_quartile_mae_tolerance: float
    maximum_mae_to_median_ratio: float
    threshold_quantiles: tuple[float, ...]
    minimum_trades: int
    minimum_active_runs: int
    minimum_profitable_run_ratio: float
    minimum_profit_factor: float
    maximum_drawdown_to_gross_profit: float
    maximum_adverse_selection_rate: float
    maximum_symbol_trade_share: float
    objective_drawdown_penalty: float
    objective_adverse_excursion_penalty: float

    def validate(self) -> None:
        probabilities = (
            self.minimum_positive_probability,
            self.maximum_adverse_probability,
            self.maximum_unpredictability_probability,
            self.minimum_profitable_run_ratio,
            self.maximum_adverse_selection_rate,
            self.maximum_symbol_trade_share,
        )
        nonnegative = (
            self.downside_penalty,
            self.adverse_excursion_penalty,
            self.lower_quartile_mae_tolerance,
            self.objective_drawdown_penalty,
            self.objective_adverse_excursion_penalty,
        )
        if (
            self.profile not in ROUND74_ACTION_PROFILES
            or any(
                not math.isfinite(float(value))
                for value in (*probabilities, *nonnegative)
            )
            or any(not 0.0 <= float(value) <= 1.0 for value in probabilities)
            or any(float(value) < 0.0 for value in nonnegative)
            or not math.isfinite(float(self.maximum_mae_to_median_ratio))
            or self.maximum_mae_to_median_ratio <= 0.0
            or not self.threshold_quantiles
            or tuple(sorted(set(self.threshold_quantiles))) != self.threshold_quantiles
            or any(
                not math.isfinite(float(value)) or not 0.0 < value < 1.0
                for value in self.threshold_quantiles
            )
            or isinstance(self.minimum_trades, bool)
            or int(self.minimum_trades) < 1
            or isinstance(self.minimum_active_runs, bool)
            or not 1 <= int(self.minimum_active_runs) <= 6
            or not math.isfinite(float(self.minimum_profit_factor))
            or self.minimum_profit_factor < 1.0
            or not math.isfinite(float(self.maximum_drawdown_to_gross_profit))
            or self.maximum_drawdown_to_gross_profit <= 0.0
        ):
            raise ValueError("Round 74 action profile differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "profile": self.profile,
            "downside_penalty": self.downside_penalty,
            "adverse_excursion_penalty": self.adverse_excursion_penalty,
            "minimum_positive_probability": self.minimum_positive_probability,
            "maximum_adverse_probability": self.maximum_adverse_probability,
            "maximum_unpredictability_probability": (
                self.maximum_unpredictability_probability
            ),
            "lower_quartile_mae_tolerance": (self.lower_quartile_mae_tolerance),
            "maximum_mae_to_median_ratio": (self.maximum_mae_to_median_ratio),
            "threshold_quantiles": list(self.threshold_quantiles),
            "minimum_trades": self.minimum_trades,
            "minimum_active_runs": self.minimum_active_runs,
            "minimum_profitable_run_ratio": (self.minimum_profitable_run_ratio),
            "minimum_profit_factor": self.minimum_profit_factor,
            "maximum_drawdown_to_gross_profit": (self.maximum_drawdown_to_gross_profit),
            "maximum_adverse_selection_rate": (self.maximum_adverse_selection_rate),
            "maximum_symbol_trade_share": self.maximum_symbol_trade_share,
            "objective_drawdown_penalty": self.objective_drawdown_penalty,
            "objective_adverse_excursion_penalty": (
                self.objective_adverse_excursion_penalty
            ),
        }


_PROFILE_SPECS = {
    "conservative": Round74ActionProfileSpec(
        profile="conservative",
        downside_penalty=0.80,
        adverse_excursion_penalty=0.65,
        minimum_positive_probability=0.62,
        maximum_adverse_probability=0.28,
        maximum_unpredictability_probability=0.30,
        lower_quartile_mae_tolerance=0.0,
        maximum_mae_to_median_ratio=0.75,
        threshold_quantiles=(0.50, 0.70, 0.85, 0.95),
        minimum_trades=12,
        minimum_active_runs=5,
        minimum_profitable_run_ratio=4.0 / 6.0,
        minimum_profit_factor=1.25,
        maximum_drawdown_to_gross_profit=0.45,
        maximum_adverse_selection_rate=0.30,
        maximum_symbol_trade_share=0.60,
        objective_drawdown_penalty=2.0,
        objective_adverse_excursion_penalty=0.50,
    ),
    "regular": Round74ActionProfileSpec(
        profile="regular",
        downside_penalty=0.55,
        adverse_excursion_penalty=0.45,
        minimum_positive_probability=0.57,
        maximum_adverse_probability=0.38,
        maximum_unpredictability_probability=0.40,
        lower_quartile_mae_tolerance=0.50,
        maximum_mae_to_median_ratio=1.25,
        threshold_quantiles=(0.35, 0.55, 0.75, 0.90),
        minimum_trades=9,
        minimum_active_runs=4,
        minimum_profitable_run_ratio=0.50,
        minimum_profit_factor=1.15,
        maximum_drawdown_to_gross_profit=0.65,
        maximum_adverse_selection_rate=0.40,
        maximum_symbol_trade_share=0.70,
        objective_drawdown_penalty=1.5,
        objective_adverse_excursion_penalty=0.35,
    ),
    "aggressive": Round74ActionProfileSpec(
        profile="aggressive",
        downside_penalty=0.35,
        adverse_excursion_penalty=0.30,
        minimum_positive_probability=0.53,
        maximum_adverse_probability=0.48,
        maximum_unpredictability_probability=0.50,
        lower_quartile_mae_tolerance=1.0,
        maximum_mae_to_median_ratio=2.0,
        threshold_quantiles=(0.20, 0.40, 0.60, 0.80),
        minimum_trades=6,
        minimum_active_runs=3,
        minimum_profitable_run_ratio=0.50,
        minimum_profit_factor=1.05,
        maximum_drawdown_to_gross_profit=0.85,
        maximum_adverse_selection_rate=0.50,
        maximum_symbol_trade_share=0.80,
        objective_drawdown_penalty=1.0,
        objective_adverse_excursion_penalty=0.20,
    ),
}
for _profile_spec in _PROFILE_SPECS.values():
    _profile_spec.validate()


def round74_action_profile(
    profile: str = ROUND74_ACTION_DEFAULT_PROFILE,
) -> Round74ActionProfileSpec:
    """Return one immutable risk profile; conservative is the default."""

    try:
        return _PROFILE_SPECS[str(profile)]
    except KeyError as exc:
        raise ValueError("Round 74 action profile is unsupported") from exc


@dataclass(frozen=True)
class Round74ActionInferenceContext:
    """Causal model input and row identity with no realized target fields."""

    role: str
    partition_sha256: str
    scaler_sha256: str
    window_representation: str
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    decision_monotonic_ns: np.ndarray
    decision_wall_ns: np.ndarray
    endpoint_frame_index: np.ndarray
    endpoint_message_index: np.ndarray
    anchor_index: np.ndarray
    sample_sha256: tuple[str, ...]
    feature_window_sha256: tuple[str, ...]
    feature_values: np.ndarray
    feature_row_sha256: tuple[str, ...]
    schema_version: str = ROUND74_ACTION_CONTEXT_SCHEMA_VERSION

    @property
    def rows(self) -> int:
        return len(self.run_id)

    def validate(self) -> None:
        identity_arrays = (
            self.decision_monotonic_ns,
            self.decision_wall_ns,
            self.endpoint_frame_index,
            self.endpoint_message_index,
            self.anchor_index,
        )
        if (
            self.schema_version != ROUND74_ACTION_CONTEXT_SCHEMA_VERSION
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or self.window_representation not in ROUND74_EVENT_WINDOW_REPRESENTATIONS
            or self.rows < 1
            or len(self.symbol) != self.rows
            or len(self.sample_sha256) != self.rows
            or len(self.feature_window_sha256) != self.rows
            or len(self.feature_row_sha256) != self.rows
            or any(_RUN_ID.fullmatch(value) is None for value in self.run_id)
            or any(value not in ROUND74_EVENT_SYMBOLS for value in self.symbol)
            or any(
                _SHA256.fullmatch(value) is None for value in self.feature_row_sha256
            )
            or any(_SHA256.fullmatch(value) is None for value in self.sample_sha256)
            or any(
                _SHA256.fullmatch(value) is None for value in self.feature_window_sha256
            )
            or _SHA256.fullmatch(self.partition_sha256) is None
            or _SHA256.fullmatch(self.scaler_sha256) is None
            or self.feature_values.shape
            != (
                self.rows,
                ROUND74_EVENT_SEQUENCE_LENGTH,
                len(ROUND74_EVENT_FEATURE_NAMES),
            )
            or self.feature_values.dtype != np.float32
            or self.feature_values.flags.writeable
            or not np.isfinite(self.feature_values).all()
            or any(value.shape != (self.rows,) for value in identity_arrays)
            or any(value.dtype != np.int64 for value in identity_arrays)
            or any(value.flags.writeable for value in identity_arrays)
            or any(np.any(value < 0) for value in identity_arrays)
        ):
            raise ValueError("Round 74 action inference context differs")
        keys = tuple(
            (
                int(self.decision_wall_ns[index]),
                self.run_id[index],
                int(self.decision_monotonic_ns[index]),
                int(self.endpoint_frame_index[index]),
                int(self.endpoint_message_index[index]),
                self.symbol[index],
                int(self.anchor_index[index]),
            )
            for index in range(self.rows)
        )
        if any(current <= prior for prior, current in zip(keys, keys[1:])):
            raise ValueError("Round 74 action inference order regressed")
        if self.feature_row_sha256 != tuple(
            _feature_row_sha256(self, index) for index in range(self.rows)
        ):
            raise ValueError("Round 74 action feature-row identity differs")

    @property
    def context_sha256(self) -> str:
        self.validate()
        identity = {
            "schema_version": self.schema_version,
            "source_dataset_schema_version": (ROUND74_EVENT_DATASET_SCHEMA_VERSION),
            "role": self.role,
            "partition_sha256": self.partition_sha256,
            "scaler_sha256": self.scaler_sha256,
            "window_representation": self.window_representation,
            "run_id": list(self.run_id),
            "symbol": list(self.symbol),
            "sample_sha256": list(self.sample_sha256),
            "feature_window_sha256": list(self.feature_window_sha256),
            "feature_row_sha256": list(self.feature_row_sha256),
            "contains_realized_targets": False,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
        for value in (
            self.decision_monotonic_ns,
            self.decision_wall_ns,
            self.endpoint_frame_index,
            self.endpoint_message_index,
            self.anchor_index,
            self.feature_values,
        ):
            _update_array_digest(digest, value)
        return digest.hexdigest()


def _feature_row_sha256(
    context: Round74ActionInferenceContext,
    index: int,
) -> str:
    identity = {
        "schema_version": ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
        "partition_sha256": context.partition_sha256,
        "scaler_sha256": context.scaler_sha256,
        "window_representation": context.window_representation,
        "role": context.role,
        "run_id": context.run_id[index],
        "symbol": context.symbol[index],
        "decision_monotonic_ns": int(context.decision_monotonic_ns[index]),
        "decision_wall_ns": int(context.decision_wall_ns[index]),
        "endpoint_frame_index": int(context.endpoint_frame_index[index]),
        "endpoint_message_index": int(context.endpoint_message_index[index]),
        "anchor_index": int(context.anchor_index[index]),
        "sample_sha256": context.sample_sha256[index],
        "feature_window_sha256": context.feature_window_sha256[index],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
    _update_array_digest(digest, context.feature_values[index])
    return digest.hexdigest()


def build_round74_action_inference_context(
    batch: Round74EventTrainingBatch,
) -> Round74ActionInferenceContext:
    """Expose immutable causal identity/features without duplicating the batch."""

    batch.validate()
    provisional = Round74ActionInferenceContext(
        role=batch.role,
        partition_sha256=batch.partition_sha256,
        scaler_sha256=batch.scaler_sha256,
        window_representation=batch.window_representation,
        run_id=tuple(batch.run_id),
        symbol=tuple(batch.symbol),
        decision_monotonic_ns=batch.decision_monotonic_ns,
        decision_wall_ns=batch.decision_wall_ns,
        endpoint_frame_index=batch.endpoint_frame_index,
        endpoint_message_index=batch.endpoint_message_index,
        anchor_index=batch.anchor_index,
        sample_sha256=batch.sample_sha256,
        feature_window_sha256=batch.feature_window_sha256,
        feature_values=batch.feature_values,
        feature_row_sha256=(),
    )
    context = Round74ActionInferenceContext(
        **{
            **provisional.__dict__,
            "feature_row_sha256": tuple(
                _feature_row_sha256(provisional, index)
                for index in range(provisional.rows)
            ),
        }
    )
    context.validate()
    return context


@dataclass(frozen=True)
class Round74ActionCandidateBatch:
    """One target-free, risk-profile candidate or abstention per row."""

    profile: str
    context_sha256: str
    model_output_sha256: str
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    tuning_subpartition_sha256: str
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    feature_row_sha256: tuple[str, ...]
    horizon_seconds: np.ndarray
    side: np.ndarray
    risk_adjusted_strength_bps: np.ndarray
    quality_score: np.ndarray
    positive_payoff_probability: np.ndarray
    adverse_selection_probability: np.ndarray
    regime_unpredictability_probability: np.ndarray
    payoff_quantiles_bps: np.ndarray
    maximum_adverse_excursion_quantiles_bps: np.ndarray
    eligible: np.ndarray
    schema_version: str = ROUND74_ACTION_POLICY_SCHEMA_VERSION
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def rows(self) -> int:
        return len(self.run_id)

    def validate(self) -> None:
        row_shape = (self.rows,)
        float_arrays = (
            self.risk_adjusted_strength_bps,
            self.quality_score,
            self.positive_payoff_probability,
            self.adverse_selection_probability,
            self.regime_unpredictability_probability,
        )
        quantile_shape = (self.rows, len(ROUND74_EVENT_PAYOFF_QUANTILES))
        if (
            self.schema_version != ROUND74_ACTION_POLICY_SCHEMA_VERSION
            or self.profile not in ROUND74_ACTION_PROFILES
            or self.rows < 1
            or len(self.symbol) != self.rows
            or len(self.feature_row_sha256) != self.rows
            or any(_RUN_ID.fullmatch(value) is None for value in self.run_id)
            or any(value not in ROUND74_EVENT_SYMBOLS for value in self.symbol)
            or any(
                _SHA256.fullmatch(value) is None for value in self.feature_row_sha256
            )
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.context_sha256,
                    self.model_output_sha256,
                    self.pretest_policy_sha256,
                    self.probability_calibration_sha256,
                    self.tuning_subpartition_sha256,
                )
            )
            or self.horizon_seconds.shape != row_shape
            or self.horizon_seconds.dtype != np.int64
            or self.horizon_seconds.flags.writeable
            or self.side.shape != row_shape
            or self.side.dtype != np.int8
            or self.side.flags.writeable
            or self.eligible.shape != row_shape
            or self.eligible.dtype != np.bool_
            or self.eligible.flags.writeable
            or any(value.shape != row_shape for value in float_arrays)
            or any(value.dtype != np.float64 for value in float_arrays)
            or any(value.flags.writeable for value in float_arrays)
            or any(not np.isfinite(value).all() for value in float_arrays)
            or self.payoff_quantiles_bps.shape != quantile_shape
            or self.maximum_adverse_excursion_quantiles_bps.shape != quantile_shape
            or self.payoff_quantiles_bps.dtype != np.float64
            or self.maximum_adverse_excursion_quantiles_bps.dtype != np.float64
            or self.payoff_quantiles_bps.flags.writeable
            or self.maximum_adverse_excursion_quantiles_bps.flags.writeable
            or not np.isfinite(self.payoff_quantiles_bps).all()
            or not np.isfinite(self.maximum_adverse_excursion_quantiles_bps).all()
            or any(
                (
                    self.trading_authority,
                    self.execution_claim,
                    self.profitability_claim,
                    self.portfolio_claim,
                    self.leverage_applied,
                )
            )
        ):
            raise ValueError("Round 74 action candidate contract differs")
        active = self.eligible
        inactive = ~active
        probabilities = (
            self.positive_payoff_probability,
            self.adverse_selection_probability,
            self.regime_unpredictability_probability,
        )
        if (
            np.any(
                ~np.isin(self.horizon_seconds, (0, *ROUND74_ACTION_HORIZONS_SECONDS))
            )
            or np.any(~np.isin(self.side, (-1, 0, 1)))
            or np.any((self.horizon_seconds > 0) != active)
            or np.any((self.side != 0) != active)
            or np.any(self.risk_adjusted_strength_bps < 0.0)
            or np.any(self.quality_score < 0.0)
            or np.any(self.risk_adjusted_strength_bps[inactive] != 0.0)
            or np.any(self.quality_score[inactive] != 0.0)
            or any(np.any((value < 0.0) | (value > 1.0)) for value in probabilities)
            or any(np.any(value[inactive] != 0.0) for value in probabilities)
            or np.any(self.payoff_quantiles_bps[inactive] != 0.0)
            or np.any(self.maximum_adverse_excursion_quantiles_bps[inactive] != 0.0)
            or np.any(np.diff(self.payoff_quantiles_bps[active], axis=1) < 0.0)
            or np.any(
                np.diff(
                    self.maximum_adverse_excursion_quantiles_bps[active],
                    axis=1,
                )
                < 0.0
            )
            or np.any(self.maximum_adverse_excursion_quantiles_bps[active] < 0.0)
        ):
            raise ValueError("Round 74 action candidate values differ")

    @property
    def candidate_sha256(self) -> str:
        self.validate()
        identity = {
            "schema_version": self.schema_version,
            "profile_spec": round74_action_profile(self.profile).as_dict(),
            "context_sha256": self.context_sha256,
            "model_output_sha256": self.model_output_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "run_id": list(self.run_id),
            "symbol": list(self.symbol),
            "feature_row_sha256": list(self.feature_row_sha256),
            "target_fields_consumed": False,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
        for value in (
            self.horizon_seconds,
            self.side,
            self.risk_adjusted_strength_bps,
            self.quality_score,
            self.positive_payoff_probability,
            self.adverse_selection_probability,
            self.regime_unpredictability_probability,
            self.payoff_quantiles_bps,
            self.maximum_adverse_excursion_quantiles_bps,
            self.eligible,
        ):
            _update_array_digest(digest, value)
        return digest.hexdigest()


def derive_round74_action_candidates(
    model_output: Round74EventModelOutput,
    context: Round74ActionInferenceContext,
    probability_calibration: Round74ProbabilityCalibration,
    *,
    pretest_policy_sha256: str,
    profile: str = ROUND74_ACTION_DEFAULT_PROFILE,
) -> Round74ActionCandidateBatch:
    """Derive at most one target-free 30/300-second candidate per row."""

    context.validate()
    model_output.validate(context.rows)
    probability_calibration.validate()
    policy_sha256 = _require_sha256(
        pretest_policy_sha256,
        "pretest policy",
    )
    if probability_calibration.pretest_policy_sha256 != policy_sha256:
        raise ValueError("Round 74 action calibration policy differs")
    spec = round74_action_profile(profile)
    positive, adverse, unpredictable = apply_round74_probability_calibration(
        probability_calibration,
        positive_payoff_logits=model_output.positive_payoff_logits,
        adverse_selection_logits=model_output.adverse_selection_logits,
        regime_unpredictability_logits=(model_output.regime_unpredictability_logits),
    )
    calibrated_payoff = model_output.payoff_quantiles_bps
    calibrated_mae = model_output.maximum_adverse_excursion_quantiles_bps
    if probability_calibration.risk_quantiles is not None:
        calibrated_payoff, calibrated_mae = apply_round74_risk_quantile_calibration(
            probability_calibration.risk_quantiles,
            payoff_quantiles_bps=calibrated_payoff,
            maximum_adverse_excursion_quantiles_bps=calibrated_mae,
        )
    payoff = _tensor_array(calibrated_payoff).astype(np.float64)
    mae = _tensor_array(calibrated_mae).astype(np.float64)
    positive_values = _tensor_array(positive).astype(np.float64)
    adverse_values = _tensor_array(adverse).astype(np.float64)
    unpredictable_values = _tensor_array(unpredictable).astype(np.float64)
    horizons = np.zeros(context.rows, dtype=np.int64)
    sides = np.zeros(context.rows, dtype=np.int8)
    strengths = np.zeros(context.rows, dtype=np.float64)
    quality = np.zeros(context.rows, dtype=np.float64)
    selected_positive = np.zeros(context.rows, dtype=np.float64)
    selected_adverse = np.zeros(context.rows, dtype=np.float64)
    selected_unpredictable = np.zeros(context.rows, dtype=np.float64)
    selected_payoff = np.zeros(
        (context.rows, len(ROUND74_EVENT_PAYOFF_QUANTILES)),
        dtype=np.float64,
    )
    selected_mae = np.zeros_like(selected_payoff)
    eligible = np.zeros(context.rows, dtype=np.bool_)
    for row_index in range(context.rows):
        choices: list[tuple[tuple[float, float, int, int], int, int, float, float]] = []
        for horizon in ROUND74_ACTION_HORIZONS_SECONDS:
            horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon)
            regime_probability = unpredictable_values[row_index, horizon_index]
            for side_index, _side_name in enumerate(ROUND74_EVENT_PAYOFF_SIDES):
                quantiles = payoff[row_index, horizon_index, side_index]
                adverse_quantiles = mae[row_index, horizon_index, side_index]
                q10, q25, q50 = (
                    float(quantiles[0]),
                    float(quantiles[1]),
                    float(quantiles[2]),
                )
                mae_q90 = float(adverse_quantiles[4])
                positive_probability = positive_values[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                adverse_probability = adverse_values[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                downside = max(0.0, -q10)
                strength = (
                    q50
                    - spec.downside_penalty * downside
                    - spec.adverse_excursion_penalty * mae_q90
                )
                lower_tail_gate = (
                    q25 + spec.lower_quartile_mae_tolerance * mae_q90
                ) >= 0.0
                mae_ratio = mae_q90 / max(q50, np.finfo(np.float64).eps)
                passes = (
                    q50 > 0.0
                    and strength > 0.0
                    and positive_probability >= spec.minimum_positive_probability
                    and adverse_probability <= spec.maximum_adverse_probability
                    and regime_probability <= spec.maximum_unpredictability_probability
                    and lower_tail_gate
                    and mae_ratio <= spec.maximum_mae_to_median_ratio
                )
                if not passes:
                    continue
                candidate_quality = (
                    strength
                    * positive_probability
                    * (1.0 - adverse_probability)
                    * (1.0 - regime_probability)
                    / max(mae_q90, 0.25)
                )
                choices.append(
                    (
                        (
                            float(candidate_quality),
                            -mae_q90,
                            -horizon,
                            -side_index,
                        ),
                        horizon_index,
                        side_index,
                        float(strength),
                        float(candidate_quality),
                    )
                )
        if not choices:
            continue
        _rank, horizon_index, side_index, strength, candidate_quality = max(
            choices,
            key=lambda value: value[0],
        )
        horizon = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS[horizon_index]
        eligible[row_index] = True
        horizons[row_index] = horizon
        sides[row_index] = 1 if side_index == 0 else -1
        strengths[row_index] = strength
        quality[row_index] = candidate_quality
        selected_positive[row_index] = positive_values[
            row_index,
            horizon_index,
            side_index,
        ]
        selected_adverse[row_index] = adverse_values[
            row_index,
            horizon_index,
            side_index,
        ]
        selected_unpredictable[row_index] = unpredictable_values[
            row_index,
            horizon_index,
        ]
        selected_payoff[row_index] = payoff[
            row_index,
            horizon_index,
            side_index,
        ]
        selected_mae[row_index] = mae[
            row_index,
            horizon_index,
            side_index,
        ]
    result = Round74ActionCandidateBatch(
        profile=spec.profile,
        context_sha256=context.context_sha256,
        model_output_sha256=_model_output_sha256(model_output),
        pretest_policy_sha256=policy_sha256,
        probability_calibration_sha256=(probability_calibration.calibration_sha256),
        tuning_subpartition_sha256=(probability_calibration.tuning_subpartition_sha256),
        run_id=context.run_id,
        symbol=context.symbol,
        feature_row_sha256=context.feature_row_sha256,
        horizon_seconds=_readonly(horizons),
        side=_readonly(sides),
        risk_adjusted_strength_bps=_readonly(strengths),
        quality_score=_readonly(quality),
        positive_payoff_probability=_readonly(selected_positive),
        adverse_selection_probability=_readonly(selected_adverse),
        regime_unpredictability_probability=_readonly(selected_unpredictable),
        payoff_quantiles_bps=_readonly(selected_payoff),
        maximum_adverse_excursion_quantiles_bps=_readonly(selected_mae),
        eligible=_readonly(eligible),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74ActionExecutionOutcomeRow:
    """One target-free feature row joined to its exact delayed L2 outcomes."""

    feature_row_sha256: str
    run_id: str
    symbol: str
    anchor_index: int
    feature_window_sha256: str
    outcomes: tuple[Round74EventTargetOutcome, ...]

    def validate(self) -> None:
        expected_keys = tuple(
            (horizon, side)
            for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
            for side in ROUND74_EVENT_PAYOFF_SIDES
        )
        if (
            _SHA256.fullmatch(self.feature_row_sha256) is None
            or _SHA256.fullmatch(self.feature_window_sha256) is None
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.symbol not in ROUND74_EVENT_SYMBOLS
            or isinstance(self.anchor_index, bool)
            or not isinstance(self.anchor_index, int)
            or self.anchor_index < 0
            or len(self.outcomes) != len(expected_keys)
        ):
            raise ValueError("Round 74 action execution outcome row differs")
        for outcome in self.outcomes:
            if not isinstance(outcome, Round74EventTargetOutcome):
                raise TypeError("Round 74 action execution outcome type differs")
            outcome.validate()
        if (
            tuple((outcome.horizon_seconds, outcome.side) for outcome in self.outcomes)
            != expected_keys
            or any(
                outcome.symbol != self.symbol
                or outcome.anchor_index != self.anchor_index
                or outcome.feature_window_sha256 != self.feature_window_sha256
                for outcome in self.outcomes
            )
            or len({outcome.target_context_sha256 for outcome in self.outcomes}) != 1
            or len({outcome.target_spec_sha256 for outcome in self.outcomes}) != 1
        ):
            raise ValueError("Round 74 action execution outcome identity differs")

    @property
    def row_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "feature_row_sha256": self.feature_row_sha256,
                "run_id": self.run_id,
                "symbol": self.symbol,
                "anchor_index": self.anchor_index,
                "feature_window_sha256": self.feature_window_sha256,
                "target_outcome_sha256": [
                    outcome.outcome_sha256 for outcome in self.outcomes
                ],
            }
        )

    def outcome(
        self,
        horizon_seconds: int,
        side: int,
    ) -> Round74EventTargetOutcome:
        if int(side) not in (-1, 1):
            raise ValueError("Round 74 action execution side differs")
        selected_side = "long" if int(side) == 1 else "short"
        for outcome in self.outcomes:
            if (
                outcome.horizon_seconds == int(horizon_seconds)
                and outcome.side == selected_side
            ):
                return outcome
        raise ValueError("Round 74 action execution outcome is missing")


@dataclass(frozen=True)
class Round74ActionExecutionPanel:
    """Profile-specific delayed L2 economics kept separate from model identity."""

    profile: str
    partition_sha256: str
    decision_latency_evidence_sha256: str
    additional_entry_latency_ns: int
    source_target_assembly_sha256: tuple[tuple[str, str], ...]
    source_capture_report_sha256: tuple[tuple[str, str], ...]
    execution_replay_module_sha256: str
    rows: tuple[Round74ActionExecutionOutcomeRow, ...]
    schema_version: str = ROUND74_ACTION_EXECUTION_PANEL_SCHEMA_VERSION
    target_fields_used_for_candidate_derivation: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        spec = round74_action_profile(self.profile)
        run_ids = tuple(run_id for run_id, _ in self.source_target_assembly_sha256)
        capture_run_ids = tuple(
            run_id for run_id, _ in self.source_capture_report_sha256
        )
        digests = (
            self.partition_sha256,
            self.decision_latency_evidence_sha256,
            self.execution_replay_module_sha256,
            *(digest for _, digest in self.source_target_assembly_sha256),
            *(digest for _, digest in self.source_capture_report_sha256),
        )
        if (
            self.schema_version != ROUND74_ACTION_EXECUTION_PANEL_SCHEMA_VERSION
            or spec.profile != self.profile
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or self.execution_replay_module_sha256
            != _module_sha256("round74_delayed_execution_panel.py")
            or isinstance(self.additional_entry_latency_ns, bool)
            or not isinstance(self.additional_entry_latency_ns, int)
            or not 0
            < self.additional_entry_latency_ns
            <= ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
            or not self.rows
            or not run_ids
            or run_ids != capture_run_ids
            or len(run_ids) != len(set(run_ids))
            or any(_RUN_ID.fullmatch(run_id) is None for run_id in run_ids)
            or any(
                (
                    self.target_fields_used_for_candidate_derivation,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 action execution panel differs")
        for row in self.rows:
            if not isinstance(row, Round74ActionExecutionOutcomeRow):
                raise TypeError("Round 74 action execution row type differs")
            row.validate()
        feature_rows = tuple(row.feature_row_sha256 for row in self.rows)
        observed_run_ids = tuple(dict.fromkeys(row.run_id for row in self.rows))
        if (
            len(feature_rows) != len(set(feature_rows))
            or observed_run_ids != run_ids
            or any(row.run_id not in run_ids for row in self.rows)
        ):
            raise ValueError("Round 74 action execution panel coverage differs")

    @property
    def panel_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "partition_sha256": self.partition_sha256,
            "decision_latency_evidence_sha256": (self.decision_latency_evidence_sha256),
            "additional_entry_latency_ns": self.additional_entry_latency_ns,
            "source_target_assembly_sha256": dict(self.source_target_assembly_sha256),
            "source_capture_report_sha256": dict(self.source_capture_report_sha256),
            "execution_replay_module_sha256": (self.execution_replay_module_sha256),
            "row_sha256": [row.row_sha256 for row in self.rows],
            "row_count": len(self.rows),
            "target_outcome_count": sum(len(row.outcomes) for row in self.rows),
            "economics_source": (
                "profile-specific exact delayed L2 replay on policy-selection runs"
            ),
            "target_fields_used_for_candidate_derivation": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["panel_sha256"] = _canonical_sha256(value)
        return value

    def row_mapping(self) -> dict[str, Round74ActionExecutionOutcomeRow]:
        self.validate()
        return {row.feature_row_sha256: row for row in self.rows}


@dataclass(frozen=True)
class Round74ActionTraceMetrics:
    trades: int
    active_runs: int
    distinct_symbols: int
    total_net_bps: float
    mean_run_net_bps: float
    mean_net_bps: float
    median_net_bps: float
    win_rate: float
    profit_factor: float | None
    maximum_drawdown_bps: float
    realized_maximum_drawdown_bps: float
    maximum_concurrent_adverse_excursion_bps: float
    gross_profit_bps: float
    gross_loss_bps: float
    worst_trade_bps: float
    mean_maximum_adverse_excursion_bps: float
    mean_run_maximum_adverse_excursion_bps: float
    adverse_selection_rate: float
    profitable_run_ratio: float
    maximum_symbol_trade_share: float

    def validate(self) -> None:
        finite = (
            self.total_net_bps,
            self.mean_run_net_bps,
            self.mean_net_bps,
            self.median_net_bps,
            self.win_rate,
            self.maximum_drawdown_bps,
            self.realized_maximum_drawdown_bps,
            self.maximum_concurrent_adverse_excursion_bps,
            self.gross_profit_bps,
            self.gross_loss_bps,
            self.worst_trade_bps,
            self.mean_maximum_adverse_excursion_bps,
            self.mean_run_maximum_adverse_excursion_bps,
            self.adverse_selection_rate,
            self.profitable_run_ratio,
            self.maximum_symbol_trade_share,
        )
        if (
            isinstance(self.trades, bool)
            or self.trades < 0
            or isinstance(self.active_runs, bool)
            or not 0 <= self.active_runs <= self.trades
            or isinstance(self.distinct_symbols, bool)
            or not 0
            <= self.distinct_symbols
            <= min(self.trades, len(ROUND74_EVENT_SYMBOLS))
            or any(not math.isfinite(float(value)) for value in finite)
            or any(
                not 0.0 <= float(value) <= 1.0
                for value in (
                    self.win_rate,
                    self.adverse_selection_rate,
                    self.profitable_run_ratio,
                    self.maximum_symbol_trade_share,
                )
            )
            or min(
                self.maximum_drawdown_bps,
                self.realized_maximum_drawdown_bps,
                self.maximum_concurrent_adverse_excursion_bps,
                self.gross_profit_bps,
                self.gross_loss_bps,
                self.mean_maximum_adverse_excursion_bps,
                self.mean_run_maximum_adverse_excursion_bps,
            )
            < 0.0
            or self.maximum_drawdown_bps + 1e-12 < self.realized_maximum_drawdown_bps
            or self.maximum_drawdown_bps + 1e-12
            < self.maximum_concurrent_adverse_excursion_bps
            or (
                self.profit_factor is not None
                and (
                    not math.isfinite(float(self.profit_factor))
                    or self.profit_factor < 0.0
                )
            )
        ):
            raise ValueError("Round 74 action trace metrics differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {key: value for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74ActionTraceMetrics:
        payload = dict(value)
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Round 74 action trace metrics payload differs")

        def integer(name: str) -> int:
            selected = payload[name]
            if isinstance(selected, bool) or not isinstance(selected, int):
                raise ValueError("Round 74 action trace metrics integer differs")
            return selected

        def number(name: str) -> float:
            selected = payload[name]
            if (
                isinstance(selected, bool)
                or not isinstance(selected, (int, float))
                or not math.isfinite(float(selected))
            ):
                raise ValueError("Round 74 action trace metrics number differs")
            return float(selected)

        profit_factor = payload["profit_factor"]
        if profit_factor is not None:
            profit_factor = number("profit_factor")
        selected = cls(
            trades=integer("trades"),
            active_runs=integer("active_runs"),
            distinct_symbols=integer("distinct_symbols"),
            total_net_bps=number("total_net_bps"),
            mean_run_net_bps=number("mean_run_net_bps"),
            mean_net_bps=number("mean_net_bps"),
            median_net_bps=number("median_net_bps"),
            win_rate=number("win_rate"),
            profit_factor=profit_factor,
            maximum_drawdown_bps=number("maximum_drawdown_bps"),
            realized_maximum_drawdown_bps=number("realized_maximum_drawdown_bps"),
            maximum_concurrent_adverse_excursion_bps=number(
                "maximum_concurrent_adverse_excursion_bps"
            ),
            gross_profit_bps=number("gross_profit_bps"),
            gross_loss_bps=number("gross_loss_bps"),
            worst_trade_bps=number("worst_trade_bps"),
            mean_maximum_adverse_excursion_bps=number(
                "mean_maximum_adverse_excursion_bps"
            ),
            mean_run_maximum_adverse_excursion_bps=number(
                "mean_run_maximum_adverse_excursion_bps"
            ),
            adverse_selection_rate=number("adverse_selection_rate"),
            profitable_run_ratio=number("profitable_run_ratio"),
            maximum_symbol_trade_share=number("maximum_symbol_trade_share"),
        )
        selected.validate()
        if _canonical_json(selected.as_dict()) != _canonical_json(payload):
            raise ValueError("Round 74 action trace metrics encoding differs")
        return selected


@dataclass(frozen=True)
class Round74ActionTrace:
    """Exact unlevered replay with one equal-capital sleeve per symbol."""

    threshold_score: float
    expected_run_ids: tuple[str, ...]
    row_index: tuple[int, ...]
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    feature_row_sha256: tuple[str, ...]
    horizon_seconds: tuple[int, ...]
    side: tuple[int, ...]
    entry_monotonic_ns: tuple[int, ...]
    exit_monotonic_ns: tuple[int, ...]
    net_payoff_bps: tuple[float, ...]
    maximum_adverse_excursion_bps: tuple[float, ...]
    adverse_selection: tuple[int, ...]
    skipped_target_ineligible: int
    skipped_same_symbol_overlap: int
    metrics: Round74ActionTraceMetrics
    position_capital_fraction: float = ROUND74_ACTION_POSITION_CAPITAL_FRACTION
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def validate(self) -> None:
        rows = len(self.row_index)
        vectors = (
            self.run_id,
            self.symbol,
            self.feature_row_sha256,
            self.horizon_seconds,
            self.side,
            self.entry_monotonic_ns,
            self.exit_monotonic_ns,
            self.net_payoff_bps,
            self.maximum_adverse_excursion_bps,
            self.adverse_selection,
        )
        if (
            not math.isfinite(float(self.threshold_score))
            or self.threshold_score < 0.0
            or not self.expected_run_ids
            or len(set(self.expected_run_ids)) != len(self.expected_run_ids)
            or any(_RUN_ID.fullmatch(value) is None for value in self.expected_run_ids)
            or any(len(value) != rows for value in vectors)
            or any(value < 0 for value in self.row_index)
            or any(
                current <= prior
                for prior, current in zip(self.row_index, self.row_index[1:])
            )
            or any(_RUN_ID.fullmatch(value) is None for value in self.run_id)
            or any(value not in self.expected_run_ids for value in self.run_id)
            or any(value not in ROUND74_EVENT_SYMBOLS for value in self.symbol)
            or any(
                _SHA256.fullmatch(value) is None for value in self.feature_row_sha256
            )
            or any(
                value not in ROUND74_ACTION_HORIZONS_SECONDS
                for value in self.horizon_seconds
            )
            or any(value not in (-1, 1) for value in self.side)
            or any(
                exit_value < entry_value
                for entry_value, exit_value in zip(
                    self.entry_monotonic_ns,
                    self.exit_monotonic_ns,
                )
            )
            or any(
                not math.isfinite(float(value))
                for value in (
                    *self.net_payoff_bps,
                    *self.maximum_adverse_excursion_bps,
                )
            )
            or any(value < 0.0 for value in self.maximum_adverse_excursion_bps)
            or any(value not in (0, 1) for value in self.adverse_selection)
            or self.skipped_target_ineligible < 0
            or self.skipped_same_symbol_overlap < 0
            or not math.isclose(
                float(self.position_capital_fraction),
                ROUND74_ACTION_POSITION_CAPITAL_FRACTION,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.position_capital_fraction * len(ROUND74_EVENT_SYMBOLS) > 1.0
            or any(
                (
                    self.trading_authority,
                    self.execution_claim,
                    self.profitability_claim,
                    self.portfolio_claim,
                    self.leverage_applied,
                )
            )
        ):
            raise ValueError("Round 74 action trace differs")
        open_until: dict[tuple[str, str], int] = {}
        for run_id, symbol, entry, exit_value in zip(
            self.run_id,
            self.symbol,
            self.entry_monotonic_ns,
            self.exit_monotonic_ns,
            strict=True,
        ):
            key = (run_id, symbol)
            if entry < open_until.get(key, -1):
                raise ValueError("Round 74 action trace overlaps")
            open_until[key] = exit_value
        self.metrics.validate()
        if self.metrics.trades != rows or self.metrics.active_runs > len(
            self.expected_run_ids
        ):
            raise ValueError("Round 74 action trace count differs")
        recalculated = _trace_metrics(
            run_ids=self.run_id,
            symbols=self.symbol,
            net_payoff_bps=self.net_payoff_bps,
            maximum_adverse_excursion_bps=self.maximum_adverse_excursion_bps,
            adverse_selection=self.adverse_selection,
            entry_monotonic_ns=self.entry_monotonic_ns,
            exit_monotonic_ns=self.exit_monotonic_ns,
            expected_run_ids=self.expected_run_ids,
        )
        if not _trace_metrics_reconcile(recalculated, self.metrics):
            raise ValueError("Round 74 action trace metrics reconciliation differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "threshold_score": self.threshold_score,
            "expected_run_ids": list(self.expected_run_ids),
            "row_index": list(self.row_index),
            "run_id": list(self.run_id),
            "symbol": list(self.symbol),
            "feature_row_sha256": list(self.feature_row_sha256),
            "horizon_seconds": list(self.horizon_seconds),
            "side": list(self.side),
            "entry_monotonic_ns": list(self.entry_monotonic_ns),
            "exit_monotonic_ns": list(self.exit_monotonic_ns),
            "net_payoff_bps": list(self.net_payoff_bps),
            "maximum_adverse_excursion_bps": list(self.maximum_adverse_excursion_bps),
            "adverse_selection": list(self.adverse_selection),
            "skipped_target_ineligible": self.skipped_target_ineligible,
            "skipped_same_symbol_overlap": (self.skipped_same_symbol_overlap),
            "metrics": self.metrics.as_dict(),
            "position_capital_fraction": self.position_capital_fraction,
            "maximum_concurrent_positions": len(ROUND74_EVENT_SYMBOLS),
            "maximum_concurrent_gross_capital_fraction": (
                self.position_capital_fraction * len(ROUND74_EVENT_SYMBOLS)
            ),
            "replay_semantics": ("one_equal_capital_sleeve_per_run_and_symbol"),
            "exact_target_entry_exit_times_used": True,
            "drawdown_order": (
                "expected_run_then_actual_exit_monotonic_ns_then_signal_order"
            ),
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74ActionTrace:
        payload = dict(value)
        expected_keys = {
            "threshold_score",
            "expected_run_ids",
            "row_index",
            "run_id",
            "symbol",
            "feature_row_sha256",
            "horizon_seconds",
            "side",
            "entry_monotonic_ns",
            "exit_monotonic_ns",
            "net_payoff_bps",
            "maximum_adverse_excursion_bps",
            "adverse_selection",
            "skipped_target_ineligible",
            "skipped_same_symbol_overlap",
            "metrics",
            "position_capital_fraction",
            "maximum_concurrent_positions",
            "maximum_concurrent_gross_capital_fraction",
            "replay_semantics",
            "exact_target_entry_exit_times_used",
            "drawdown_order",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 action trace payload differs")

        def sequence(name: str) -> list[object]:
            selected = payload[name]
            if not isinstance(selected, list):
                raise ValueError("Round 74 action trace sequence differs")
            return selected

        def strings(name: str) -> tuple[str, ...]:
            selected = sequence(name)
            if any(not isinstance(item, str) for item in selected):
                raise ValueError("Round 74 action trace string sequence differs")
            return tuple(selected)

        def integers(name: str) -> tuple[int, ...]:
            selected = sequence(name)
            if any(
                isinstance(item, bool) or not isinstance(item, int) for item in selected
            ):
                raise ValueError("Round 74 action trace integer sequence differs")
            return tuple(selected)

        def numbers(name: str) -> tuple[float, ...]:
            selected = sequence(name)
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in selected
            ):
                raise ValueError("Round 74 action trace number sequence differs")
            return tuple(float(item) for item in selected)

        def integer(name: str) -> int:
            selected = payload[name]
            if isinstance(selected, bool) or not isinstance(selected, int):
                raise ValueError("Round 74 action trace integer differs")
            return selected

        threshold = payload["threshold_score"]
        metrics = payload["metrics"]
        position_capital_fraction = payload["position_capital_fraction"]
        maximum_concurrent_positions = payload["maximum_concurrent_positions"]
        maximum_concurrent_gross_capital_fraction = payload[
            "maximum_concurrent_gross_capital_fraction"
        ]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not isinstance(metrics, Mapping)
            or isinstance(position_capital_fraction, bool)
            or not isinstance(position_capital_fraction, (int, float))
            or not math.isfinite(float(position_capital_fraction))
            or isinstance(maximum_concurrent_positions, bool)
            or not isinstance(maximum_concurrent_positions, int)
            or isinstance(maximum_concurrent_gross_capital_fraction, bool)
            or not isinstance(
                maximum_concurrent_gross_capital_fraction,
                (int, float),
            )
            or not math.isfinite(float(maximum_concurrent_gross_capital_fraction))
        ):
            raise ValueError("Round 74 action trace types differ")
        selected = cls(
            threshold_score=float(threshold),
            expected_run_ids=strings("expected_run_ids"),
            row_index=integers("row_index"),
            run_id=strings("run_id"),
            symbol=strings("symbol"),
            feature_row_sha256=strings("feature_row_sha256"),
            horizon_seconds=integers("horizon_seconds"),
            side=integers("side"),
            entry_monotonic_ns=integers("entry_monotonic_ns"),
            exit_monotonic_ns=integers("exit_monotonic_ns"),
            net_payoff_bps=numbers("net_payoff_bps"),
            maximum_adverse_excursion_bps=numbers("maximum_adverse_excursion_bps"),
            adverse_selection=integers("adverse_selection"),
            skipped_target_ineligible=integer("skipped_target_ineligible"),
            skipped_same_symbol_overlap=integer("skipped_same_symbol_overlap"),
            metrics=Round74ActionTraceMetrics.from_dict(metrics),
            position_capital_fraction=float(position_capital_fraction),
            trading_authority=payload["trading_authority"],
            execution_claim=payload["execution_claim"],
            profitability_claim=payload["profitability_claim"],
            portfolio_claim=payload["portfolio_claim"],
            leverage_applied=payload["leverage_applied"],
        )
        selected.validate()
        if (
            maximum_concurrent_positions != len(ROUND74_EVENT_SYMBOLS)
            or not math.isclose(
                float(maximum_concurrent_gross_capital_fraction),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or payload["replay_semantics"]
            != "one_equal_capital_sleeve_per_run_and_symbol"
            or _canonical_json(selected.as_dict()) != _canonical_json(payload)
        ):
            raise ValueError("Round 74 action trace policy differs")
        return selected


def _trace_metrics(
    *,
    run_ids: tuple[str, ...],
    symbols: tuple[str, ...],
    net_payoff_bps: tuple[float, ...],
    maximum_adverse_excursion_bps: tuple[float, ...],
    adverse_selection: tuple[int, ...],
    entry_monotonic_ns: tuple[int, ...],
    exit_monotonic_ns: tuple[int, ...],
    expected_run_ids: tuple[str, ...],
) -> Round74ActionTraceMetrics:
    values = np.asarray(net_payoff_bps, dtype=np.float64)
    adverse_excursion = np.asarray(
        maximum_adverse_excursion_bps,
        dtype=np.float64,
    )
    adverse = np.asarray(adverse_selection, dtype=np.float64)
    if values.size:
        gross_profit = float(values[values > 0.0].sum())
        gross_loss = float(-values[values < 0.0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None
        symbol_counts = {
            symbol: symbols.count(symbol) for symbol in ROUND74_EVENT_SYMBOLS
        }
        maximum_symbol_share = max(symbol_counts.values()) / len(symbols)
    else:
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = None
        maximum_symbol_share = 0.0
    run_pnl = {run_id: 0.0 for run_id in expected_run_ids}
    run_adverse_excursion = {run_id: 0.0 for run_id in expected_run_ids}
    run_trades = {run_id: 0 for run_id in expected_run_ids}
    for run_id, value, excursion in zip(
        run_ids,
        values,
        adverse_excursion,
        strict=True,
    ):
        run_pnl[run_id] += float(value)
        run_adverse_excursion[run_id] += float(excursion)
        run_trades[run_id] += 1
    realized_drawdown = round74_maximum_realized_drawdown_bps(
        values,
        run_ids=run_ids,
        exit_monotonic_ns=exit_monotonic_ns,
        expected_run_ids=expected_run_ids,
    )
    concurrent_adverse_excursion = round74_maximum_concurrent_adverse_excursion_bps(
        adverse_excursion,
        run_ids=run_ids,
        entry_monotonic_ns=entry_monotonic_ns,
        exit_monotonic_ns=exit_monotonic_ns,
        expected_run_ids=expected_run_ids,
    )
    conservative_drawdown = round74_conservative_maximum_drawdown_bps(
        values,
        adverse_excursion,
        run_ids=run_ids,
        entry_monotonic_ns=entry_monotonic_ns,
        exit_monotonic_ns=exit_monotonic_ns,
        expected_run_ids=expected_run_ids,
    )
    result = Round74ActionTraceMetrics(
        trades=int(values.size),
        active_runs=sum(count > 0 for count in run_trades.values()),
        distinct_symbols=len(set(symbols)),
        total_net_bps=float(values.sum()) if values.size else 0.0,
        mean_run_net_bps=float(np.mean(tuple(run_pnl.values()))),
        mean_net_bps=float(values.mean()) if values.size else 0.0,
        median_net_bps=float(np.median(values)) if values.size else 0.0,
        win_rate=float(np.mean(values > 0.0)) if values.size else 0.0,
        profit_factor=profit_factor,
        maximum_drawdown_bps=conservative_drawdown,
        realized_maximum_drawdown_bps=realized_drawdown,
        maximum_concurrent_adverse_excursion_bps=(concurrent_adverse_excursion),
        gross_profit_bps=gross_profit,
        gross_loss_bps=gross_loss,
        worst_trade_bps=float(values.min()) if values.size else 0.0,
        mean_maximum_adverse_excursion_bps=(
            float(adverse_excursion.mean()) if values.size else 0.0
        ),
        mean_run_maximum_adverse_excursion_bps=float(
            np.mean(tuple(run_adverse_excursion.values()))
        ),
        adverse_selection_rate=(float(adverse.mean()) if values.size else 0.0),
        profitable_run_ratio=float(
            np.mean(np.asarray(tuple(run_pnl.values()), dtype=np.float64) > 0.0)
        ),
        maximum_symbol_trade_share=float(maximum_symbol_share),
    )
    result.validate()
    return result


def _trace_metrics_reconcile(
    expected: Round74ActionTraceMetrics,
    observed: Round74ActionTraceMetrics,
) -> bool:
    """Compare recomputed metrics without rejecting harmless float summation order."""

    for name in expected.__dataclass_fields__:
        left = getattr(expected, name)
        right = getattr(observed, name)
        if left is None or right is None:
            if left is not right:
                return False
        elif isinstance(left, float) or isinstance(right, float):
            if not math.isclose(
                float(left),
                float(right),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
        elif left != right:
            return False
    return True


def _round74_batch_order_key(
    batch: Round74EventTrainingBatch,
    row_index: int,
) -> tuple[int, str, int, int, int, str, int]:
    return (
        int(batch.decision_wall_ns[row_index]),
        batch.run_id[row_index],
        int(batch.decision_monotonic_ns[row_index]),
        int(batch.endpoint_frame_index[row_index]),
        int(batch.endpoint_message_index[row_index]),
        batch.symbol[row_index],
        int(batch.anchor_index[row_index]),
    )


def _validate_round74_action_panel(
    batches: Sequence[Round74EventTrainingBatch],
    candidate_batches: Sequence[Round74ActionCandidateBatch],
    *,
    expected_run_ids: tuple[str, ...],
    required_role: str,
    expected_run_count: int,
) -> tuple[
    tuple[Round74EventTrainingBatch, ...],
    tuple[Round74ActionCandidateBatch, ...],
]:
    selected_batches = tuple(batches)
    selected_candidates = tuple(candidate_batches)
    expected = tuple(expected_run_ids)
    if (
        not selected_batches
        or len(selected_batches) != len(selected_candidates)
        or required_role not in ROUND74_EVENT_PARTITION_ROLES
        or isinstance(expected_run_count, bool)
        or expected_run_count < 1
        or len(expected) != expected_run_count
        or len(set(expected)) != len(expected)
        or any(_RUN_ID.fullmatch(value) is None for value in expected)
    ):
        raise ValueError("Round 74 action replay panel differs")
    first_batch = selected_batches[0]
    first_candidates = selected_candidates[0]
    seen_runs: set[str] = set()
    seen_samples: set[str] = set()
    seen_feature_rows: set[str] = set()
    prior_last_key: tuple[int, str, int, int, int, str, int] | None = None
    for batch, candidates in zip(
        selected_batches,
        selected_candidates,
        strict=True,
    ):
        batch.validate()
        candidates.validate()
        context = build_round74_action_inference_context(batch)
        first_key = _round74_batch_order_key(batch, 0)
        last_key = _round74_batch_order_key(batch, batch.rows - 1)
        sample_set = set(batch.sample_sha256)
        feature_set = set(candidates.feature_row_sha256)
        if (
            batch.role != required_role
            or batch.partition_sha256 != first_batch.partition_sha256
            or batch.scaler_sha256 != first_batch.scaler_sha256
            or candidates.profile != first_candidates.profile
            or candidates.pretest_policy_sha256
            != first_candidates.pretest_policy_sha256
            or candidates.probability_calibration_sha256
            != first_candidates.probability_calibration_sha256
            or candidates.tuning_subpartition_sha256
            != first_candidates.tuning_subpartition_sha256
            or candidates.context_sha256 != context.context_sha256
            or candidates.run_id != context.run_id
            or candidates.symbol != context.symbol
            or candidates.feature_row_sha256 != context.feature_row_sha256
            or any(run_id not in expected for run_id in batch.run_id)
            or len(sample_set) != batch.rows
            or len(feature_set) != batch.rows
            or seen_samples.intersection(sample_set)
            or seen_feature_rows.intersection(feature_set)
            or (prior_last_key is not None and first_key <= prior_last_key)
        ):
            raise ValueError("Round 74 action replay identity differs")
        seen_runs.update(batch.run_id)
        seen_samples.update(sample_set)
        seen_feature_rows.update(feature_set)
        prior_last_key = last_key
    if seen_runs != set(expected):
        raise ValueError("Round 74 action replay run coverage differs")
    return selected_batches, selected_candidates


def _validated_execution_rows(
    batches: Sequence[Round74EventTrainingBatch],
    candidate_batches: Sequence[Round74ActionCandidateBatch],
    execution_panel: Round74ActionExecutionPanel | None,
    *,
    expected_run_ids: Sequence[str],
) -> dict[str, Round74ActionExecutionOutcomeRow] | None:
    if execution_panel is None:
        return None
    execution_panel.validate()
    selected_batches = tuple(batches)
    selected_candidates = tuple(candidate_batches)
    expected = tuple(expected_run_ids)
    feature_rows = tuple(
        feature_row
        for candidates in selected_candidates
        for feature_row in candidates.feature_row_sha256
    )
    rows = {row.feature_row_sha256: row for row in execution_panel.rows}
    if (
        not selected_batches
        or len(selected_batches) != len(selected_candidates)
        or execution_panel.partition_sha256 != selected_batches[0].partition_sha256
        or execution_panel.profile != selected_candidates[0].profile
        or tuple(run_id for run_id, _ in execution_panel.source_target_assembly_sha256)
        != expected
        or tuple(rows) != feature_rows
    ):
        raise ValueError("Round 74 action execution panel binding differs")
    for batch, candidates in zip(
        selected_batches,
        selected_candidates,
        strict=True,
    ):
        for row_index, feature_row_sha256 in enumerate(candidates.feature_row_sha256):
            row = rows[feature_row_sha256]
            if (
                row.run_id != batch.run_id[row_index]
                or row.symbol != batch.symbol[row_index]
                or row.anchor_index != int(batch.anchor_index[row_index])
                or row.feature_window_sha256 != batch.feature_window_sha256[row_index]
            ):
                raise ValueError("Round 74 action execution row binding differs")
    return rows


def _simulate_round74_action_trace_batches(
    batches: Sequence[Round74EventTrainingBatch],
    candidate_batches: Sequence[Round74ActionCandidateBatch],
    *,
    threshold_score: float,
    expected_run_ids: tuple[str, ...],
    required_role: str,
    expected_run_count: int,
    execution_rows: Mapping[str, Round74ActionExecutionOutcomeRow] | None = None,
) -> Round74ActionTrace:
    """Replay a chronological batch panel without concatenating feature tensors."""

    threshold = float(threshold_score)
    expected = tuple(expected_run_ids)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Round 74 action replay identity differs")
    selected_batches, selected_candidate_batches = _validate_round74_action_panel(
        batches,
        candidate_batches,
        expected_run_ids=expected,
        required_role=required_role,
        expected_run_count=expected_run_count,
    )
    selected_rows: list[int] = []
    selected_runs: list[str] = []
    selected_symbols: list[str] = []
    selected_features: list[str] = []
    selected_horizons: list[int] = []
    selected_sides: list[int] = []
    entries: list[int] = []
    exits: list[int] = []
    payoffs: list[float] = []
    adverse_excursions: list[float] = []
    adverse_selections: list[int] = []
    open_until: dict[tuple[str, str], int] = {}
    skipped_target_ineligible = 0
    skipped_overlap = 0
    global_row_offset = 0
    for batch, candidates in zip(
        selected_batches,
        selected_candidate_batches,
        strict=True,
    ):
        for row_index in range(batch.rows):
            if (
                not candidates.eligible[row_index]
                or candidates.quality_score[row_index] < threshold
            ):
                continue
            horizon = int(candidates.horizon_seconds[row_index])
            side = int(candidates.side[row_index])
            if execution_rows is None:
                horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon)
                side_index = 0 if side == 1 else 1
                if (
                    batch.action_eligibility[row_index, horizon_index, side_index]
                    != 1.0
                ):
                    skipped_target_ineligible += 1
                    continue
                entry = int(
                    batch.actual_entry_monotonic_ns[
                        row_index,
                        horizon_index,
                        side_index,
                    ]
                )
                exit_value = int(
                    batch.actual_exit_monotonic_ns[
                        row_index,
                        horizon_index,
                        side_index,
                    ]
                )
                payoff = ROUND74_ACTION_POSITION_CAPITAL_FRACTION * float(
                    batch.net_payoff_bps[
                        row_index,
                        horizon_index,
                        side_index,
                    ]
                )
                adverse_excursion = ROUND74_ACTION_POSITION_CAPITAL_FRACTION * float(
                    batch.maximum_adverse_excursion_bps[
                        row_index,
                        horizon_index,
                        side_index,
                    ]
                )
                adverse_selection = int(
                    batch.adverse_selection[
                        row_index,
                        horizon_index,
                        side_index,
                    ]
                )
            else:
                feature_row = candidates.feature_row_sha256[row_index]
                outcome = execution_rows[feature_row].outcome(horizon, side)
                if not outcome.eligible:
                    skipped_target_ineligible += 1
                    continue
                assert outcome.actual_entry_monotonic_ns is not None
                assert outcome.actual_exit_monotonic_ns is not None
                assert outcome.capital_scaled_net_payoff_bps is not None
                assert outcome.capital_scaled_maximum_adverse_excursion_bps is not None
                assert outcome.adverse_selection is not None
                entry = int(outcome.actual_entry_monotonic_ns)
                exit_value = int(outcome.actual_exit_monotonic_ns)
                payoff = ROUND74_ACTION_POSITION_CAPITAL_FRACTION * float(
                    outcome.capital_scaled_net_payoff_bps
                )
                adverse_excursion = ROUND74_ACTION_POSITION_CAPITAL_FRACTION * float(
                    outcome.capital_scaled_maximum_adverse_excursion_bps
                )
                adverse_selection = int(outcome.adverse_selection)
            position_key = (batch.run_id[row_index], batch.symbol[row_index])
            if entry < open_until.get(position_key, -1):
                skipped_overlap += 1
                continue
            open_until[position_key] = exit_value
            selected_rows.append(global_row_offset + row_index)
            selected_runs.append(batch.run_id[row_index])
            selected_symbols.append(batch.symbol[row_index])
            selected_features.append(candidates.feature_row_sha256[row_index])
            selected_horizons.append(horizon)
            selected_sides.append(side)
            entries.append(entry)
            exits.append(exit_value)
            payoffs.append(payoff)
            adverse_excursions.append(adverse_excursion)
            adverse_selections.append(adverse_selection)
        global_row_offset += batch.rows
    run_tuple = tuple(selected_runs)
    symbol_tuple = tuple(selected_symbols)
    payoff_tuple = tuple(payoffs)
    adverse_excursion_tuple = tuple(adverse_excursions)
    adverse_selection_tuple = tuple(adverse_selections)
    result = Round74ActionTrace(
        threshold_score=threshold,
        expected_run_ids=expected,
        row_index=tuple(selected_rows),
        run_id=run_tuple,
        symbol=symbol_tuple,
        feature_row_sha256=tuple(selected_features),
        horizon_seconds=tuple(selected_horizons),
        side=tuple(selected_sides),
        entry_monotonic_ns=tuple(entries),
        exit_monotonic_ns=tuple(exits),
        net_payoff_bps=payoff_tuple,
        maximum_adverse_excursion_bps=adverse_excursion_tuple,
        adverse_selection=adverse_selection_tuple,
        skipped_target_ineligible=skipped_target_ineligible,
        skipped_same_symbol_overlap=skipped_overlap,
        metrics=_trace_metrics(
            run_ids=run_tuple,
            symbols=symbol_tuple,
            net_payoff_bps=payoff_tuple,
            maximum_adverse_excursion_bps=adverse_excursion_tuple,
            adverse_selection=adverse_selection_tuple,
            entry_monotonic_ns=tuple(entries),
            exit_monotonic_ns=tuple(exits),
            expected_run_ids=expected,
        ),
    )
    result.validate()
    return result


def simulate_round74_action_trace_batches(
    batches: Sequence[Round74EventTrainingBatch],
    candidate_batches: Sequence[Round74ActionCandidateBatch],
    *,
    threshold_score: float,
    expected_run_ids: tuple[str, ...],
    execution_panel: Round74ActionExecutionPanel | None = None,
) -> Round74ActionTrace:
    """Replay only the six policy-selection runs through the shared core."""

    execution_rows = _validated_execution_rows(
        batches,
        candidate_batches,
        execution_panel,
        expected_run_ids=expected_run_ids,
    )
    return _simulate_round74_action_trace_batches(
        batches,
        candidate_batches,
        threshold_score=threshold_score,
        expected_run_ids=expected_run_ids,
        required_role="tuning",
        expected_run_count=len(expected_run_ids),
        execution_rows=execution_rows,
    )


def simulate_round74_action_trace(
    batch: Round74EventTrainingBatch,
    candidates: Round74ActionCandidateBatch,
    *,
    threshold_score: float,
    expected_run_ids: tuple[str, ...],
    execution_panel: Round74ActionExecutionPanel | None = None,
) -> Round74ActionTrace:
    """Replay one compatibility batch through the bounded panel implementation."""

    return simulate_round74_action_trace_batches(
        (batch,),
        (candidates,),
        threshold_score=threshold_score,
        expected_run_ids=expected_run_ids,
        execution_panel=execution_panel,
    )


def _trace_gate_reasons(
    trace: Round74ActionTrace,
    spec: Round74ActionProfileSpec,
) -> tuple[str, ...]:
    metrics = trace.metrics
    run_scale = len(trace.expected_run_ids) / ROUND74_TUNING_POLICY_SELECTION_RUNS
    minimum_trades = math.ceil(spec.minimum_trades * run_scale)
    minimum_active_runs = min(
        len(trace.expected_run_ids),
        math.ceil(spec.minimum_active_runs * run_scale),
    )
    reasons: list[str] = []
    if trace.skipped_target_ineligible > 0:
        reasons.append("selected_action_target_coverage_incomplete")
    if metrics.trades < minimum_trades:
        reasons.append("minimum_trades_not_met")
    if metrics.active_runs < minimum_active_runs:
        reasons.append("minimum_active_runs_not_met")
    if metrics.distinct_symbols != len(ROUND74_EVENT_SYMBOLS):
        reasons.append("asset_diversification_not_met")
    if metrics.total_net_bps <= 0.0 or metrics.mean_net_bps <= 0.0:
        reasons.append("positive_after_cost_payoff_not_met")
    if metrics.profitable_run_ratio < spec.minimum_profitable_run_ratio:
        reasons.append("profitable_run_ratio_not_met")
    if metrics.gross_loss_bps > 0.0 and (
        metrics.profit_factor is None
        or metrics.profit_factor < spec.minimum_profit_factor
    ):
        reasons.append("profit_factor_not_met")
    drawdown_ratio = (
        metrics.maximum_drawdown_bps / metrics.gross_profit_bps
        if metrics.gross_profit_bps > 0.0
        else math.inf
    )
    if drawdown_ratio > spec.maximum_drawdown_to_gross_profit:
        reasons.append("drawdown_to_gross_profit_not_met")
    if metrics.adverse_selection_rate > spec.maximum_adverse_selection_rate:
        reasons.append("adverse_selection_rate_not_met")
    if metrics.maximum_symbol_trade_share > spec.maximum_symbol_trade_share:
        reasons.append("symbol_concentration_not_met")
    return tuple(reasons)


@dataclass(frozen=True)
class Round74ActionThresholdEvaluation:
    quantile: float
    threshold_score: float
    objective_bps: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    trace: Round74ActionTrace
    objective_semantics: str = (
        "mean_run_net_bps_minus_worst_drawdown_and_mean_run_mae_penalties"
    )

    def validate(self) -> None:
        self.trace.validate()
        if (
            not math.isfinite(float(self.quantile))
            or not 0.0 < self.quantile < 1.0
            or not math.isfinite(float(self.threshold_score))
            or self.threshold_score < 0.0
            or not math.isfinite(float(self.objective_bps))
            or self.trace.threshold_score != self.threshold_score
            or self.accepted == bool(self.rejection_reasons)
            or self.objective_semantics
            not in {
                "mean_run_net_bps_minus_worst_drawdown_and_mean_run_mae_penalties",
                "total_net_bps_minus_worst_drawdown_and_total_mae_penalties",
            }
        ):
            raise ValueError("Round 74 action threshold evaluation differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "quantile": self.quantile,
            "threshold_score": self.threshold_score,
            "objective_bps": self.objective_bps,
            "objective_semantics": self.objective_semantics,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "trace": self.trace.as_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74ActionThresholdEvaluation:
        payload = dict(value)
        expected_keys = {
            "quantile",
            "threshold_score",
            "objective_bps",
            "objective_semantics",
            "accepted",
            "rejection_reasons",
            "trace",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 action threshold payload differs")
        trace = payload["trace"]
        reasons = payload["rejection_reasons"]
        accepted = payload["accepted"]
        numbers = (
            payload["quantile"],
            payload["threshold_score"],
            payload["objective_bps"],
        )
        if (
            not isinstance(trace, Mapping)
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
            or not isinstance(accepted, bool)
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in numbers
            )
        ):
            raise ValueError("Round 74 action threshold types differ")
        selected = cls(
            quantile=float(numbers[0]),
            threshold_score=float(numbers[1]),
            objective_bps=float(numbers[2]),
            accepted=accepted,
            rejection_reasons=tuple(reasons),
            trace=Round74ActionTrace.from_dict(trace),
            objective_semantics=str(payload["objective_semantics"]),
        )
        selected.validate()
        if _canonical_json(selected.as_dict()) != _canonical_json(payload):
            raise ValueError("Round 74 action threshold policy differs")
        return selected


@dataclass(frozen=True)
class Round74ActionPolicySelection:
    """Hash-bound tuning result that may validly choose to abstain."""

    profile: str
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    tuning_subpartition_sha256: str
    target_batch_sha256: tuple[str, ...]
    candidate_sha256: tuple[str, ...]
    accepted: bool
    selected_quantile: float | None
    selected_threshold_score: float | None
    evaluations: tuple[Round74ActionThresholdEvaluation, ...]
    rejection_reasons: tuple[str, ...]
    execution_outcome_panel_sha256: str | None = None
    optimization_population: str = "capture_run"
    schema_version: str = ROUND74_ACTION_POLICY_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def validate(self) -> None:
        spec = round74_action_profile(self.profile)
        if (
            self.schema_version != ROUND74_ACTION_POLICY_SCHEMA_VERSION
            or self.optimization_population
            not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.pretest_policy_sha256,
                    self.probability_calibration_sha256,
                    self.tuning_subpartition_sha256,
                    *self.target_batch_sha256,
                    *self.candidate_sha256,
                )
            )
            or not self.target_batch_sha256
            or (
                self.execution_outcome_panel_sha256 is not None
                and _SHA256.fullmatch(self.execution_outcome_panel_sha256) is None
            )
            or len(self.target_batch_sha256) != len(self.candidate_sha256)
            or len(set(self.target_batch_sha256)) != len(self.target_batch_sha256)
            or len(set(self.candidate_sha256)) != len(self.candidate_sha256)
            or len(self.evaluations) != len(spec.threshold_quantiles)
            or tuple(value.quantile for value in self.evaluations)
            != spec.threshold_quantiles
            or any(
                value.trace.expected_run_ids
                != self.evaluations[0].trace.expected_run_ids
                for value in self.evaluations
            )
            or any(
                (
                    self.sealed_test_accessed,
                    self.trading_authority,
                    self.execution_claim,
                    self.profitability_claim,
                    self.portfolio_claim,
                    self.leverage_applied,
                )
            )
        ):
            raise ValueError("Round 74 action policy selection differs")
        for evaluation in self.evaluations:
            evaluation.validate()
        if self.accepted:
            if (
                self.selected_quantile is None
                or self.selected_threshold_score is None
                or self.rejection_reasons
                or not any(
                    value.accepted
                    and value.quantile == self.selected_quantile
                    and value.threshold_score == self.selected_threshold_score
                    for value in self.evaluations
                )
            ):
                raise ValueError("Round 74 accepted action policy differs")
        elif (
            self.selected_quantile is not None
            or self.selected_threshold_score is not None
            or not self.rejection_reasons
        ):
            raise ValueError("Round 74 rejected action policy differs")

    @property
    def selection_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "profile_spec": round74_action_profile(self.profile).as_dict(),
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "target_batch_sha256": list(self.target_batch_sha256),
            "candidate_sha256": list(self.candidate_sha256),
            "execution_outcome_panel_sha256": (self.execution_outcome_panel_sha256),
            "execution_economics_source": (
                "profile_specific_exact_delayed_l2_panel"
                if self.execution_outcome_panel_sha256 is not None
                else "baseline_training_batch"
            ),
            "accepted": self.accepted,
            "selected_quantile": self.selected_quantile,
            "selected_threshold_score": self.selected_threshold_score,
            "evaluations": [evaluation.as_dict() for evaluation in self.evaluations],
            "rejection_reasons": list(self.rejection_reasons),
            "selection_data_role": "policy_selection_tuning_runs_only",
            "sealed_test_accessed": False,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        value["optimization_population"] = self.optimization_population
        if include_sha256:
            value["selection_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74ActionPolicySelection:
        payload = dict(value)
        claimed = payload.pop("selection_sha256", None)
        if (
            not isinstance(claimed, str)
            or _SHA256.fullmatch(claimed) is None
            or claimed != _canonical_sha256(payload)
        ):
            raise ValueError("Round 74 action policy digest differs")
        expected_keys = {
            "schema_version",
            "profile",
            "profile_spec",
            "pretest_policy_sha256",
            "probability_calibration_sha256",
            "tuning_subpartition_sha256",
            "target_batch_sha256",
            "candidate_sha256",
            "execution_outcome_panel_sha256",
            "execution_economics_source",
            "accepted",
            "selected_quantile",
            "selected_threshold_score",
            "evaluations",
            "rejection_reasons",
            "selection_data_role",
            "sealed_test_accessed",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
            "optimization_population",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 action policy payload differs")

        def strings(name: str) -> tuple[str, ...]:
            values = payload[name]
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise ValueError("Round 74 action policy sequence differs")
            return tuple(values)

        evaluations = payload["evaluations"]
        reasons = payload["rejection_reasons"]
        accepted = payload["accepted"]
        if (
            not isinstance(evaluations, list)
            or any(not isinstance(item, Mapping) for item in evaluations)
            or not isinstance(reasons, list)
            or any(not isinstance(item, str) for item in reasons)
            or not isinstance(accepted, bool)
        ):
            raise ValueError("Round 74 action policy types differ")

        def optional_number(name: str) -> float | None:
            value = payload[name]
            if value is None:
                return None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("Round 74 action policy number differs")
            return float(value)

        selected = cls(
            profile=str(payload["profile"]),
            pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
            probability_calibration_sha256=str(
                payload["probability_calibration_sha256"]
            ),
            tuning_subpartition_sha256=str(payload["tuning_subpartition_sha256"]),
            target_batch_sha256=strings("target_batch_sha256"),
            candidate_sha256=strings("candidate_sha256"),
            accepted=accepted,
            selected_quantile=optional_number("selected_quantile"),
            selected_threshold_score=optional_number("selected_threshold_score"),
            evaluations=tuple(
                Round74ActionThresholdEvaluation.from_dict(item) for item in evaluations
            ),
            rejection_reasons=tuple(reasons),
            execution_outcome_panel_sha256=(
                str(payload["execution_outcome_panel_sha256"])
                if payload["execution_outcome_panel_sha256"] is not None
                else None
            ),
            optimization_population=(str(payload["optimization_population"])),
            schema_version=str(payload["schema_version"]),
            sealed_test_accessed=payload["sealed_test_accessed"],
            trading_authority=payload["trading_authority"],
            execution_claim=payload["execution_claim"],
            profitability_claim=payload["profitability_claim"],
            portfolio_claim=payload["portfolio_claim"],
            leverage_applied=payload["leverage_applied"],
        )
        selected.validate()
        expected_economics_source = (
            "profile_specific_exact_delayed_l2_panel"
            if selected.execution_outcome_panel_sha256 is not None
            else "baseline_training_batch"
        )
        if payload["execution_economics_source"] != expected_economics_source:
            raise ValueError("Round 74 action policy execution source differs")
        if _canonical_json(selected.as_dict(include_sha256=False)) != _canonical_json(
            payload
        ):
            raise ValueError("Round 74 action policy contract differs")
        if selected.selection_sha256 != claimed:
            raise ValueError("Round 74 action policy identity differs")
        return selected


def _equal_run_score_threshold(
    scores_by_run: Mapping[str, Sequence[float]],
    *,
    quantile: float,
    expected_run_ids: Sequence[str],
) -> float:
    """Give each capture run one vote in the threshold grid."""

    expected = tuple(expected_run_ids)
    selected_quantile = float(quantile)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not 0.0 < selected_quantile < 1.0
        or any(run_id not in expected for run_id in scores_by_run)
    ):
        raise ValueError("Round 74 equal-run threshold identity differs")
    run_quantiles: list[float] = []
    for run_id in expected:
        values = np.asarray(tuple(scores_by_run.get(run_id, ())), dtype=np.float64)
        if values.size == 0:
            continue
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Round 74 equal-run threshold score differs")
        run_quantiles.append(
            float(np.quantile(values, selected_quantile, method="linear"))
        )
    if not run_quantiles:
        raise ValueError("Round 74 equal-run threshold has no active run")
    return float(np.median(np.asarray(run_quantiles, dtype=np.float64)))


def _eligible_target_score_threshold(
    scores_by_run: Mapping[str, Sequence[float]],
    *,
    quantile: float,
    expected_run_ids: Sequence[str],
) -> float:
    """Weight every duration-normalized eligible target exactly once."""

    expected = tuple(expected_run_ids)
    selected_quantile = float(quantile)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not 0.0 < selected_quantile < 1.0
        or any(run_id not in expected for run_id in scores_by_run)
    ):
        raise ValueError("Round 74 eligible-target threshold identity differs")
    panels = tuple(
        np.asarray(tuple(scores_by_run.get(run_id, ())), dtype=np.float64)
        for run_id in expected
    )
    if any(
        values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0)
        for values in panels
    ):
        raise ValueError("Round 74 eligible-target threshold score differs")
    active = tuple(values for values in panels if values.size)
    if not active:
        raise ValueError("Round 74 eligible-target threshold has no active target")
    return float(
        np.quantile(
            np.concatenate(active),
            selected_quantile,
            method="linear",
        )
    )


def _action_selection_objective(
    metrics: Round74ActionTraceMetrics,
    spec: Round74ActionProfileSpec,
    *,
    optimization_population: str,
    expected_run_count: int,
) -> tuple[float, str]:
    if optimization_population == "capture_run":
        return (
            float(
                metrics.mean_run_net_bps
                - spec.objective_drawdown_penalty * metrics.maximum_drawdown_bps
                - spec.objective_adverse_excursion_penalty
                * metrics.mean_run_maximum_adverse_excursion_bps
            ),
            "mean_run_net_bps_minus_worst_drawdown_and_mean_run_mae_penalties",
        )
    if (
        optimization_population != "eligible_target"
        or isinstance(expected_run_count, bool)
        or expected_run_count < 1
    ):
        raise ValueError("Round 74 action optimization population differs")
    return (
        float(
            metrics.total_net_bps
            - spec.objective_drawdown_penalty * metrics.maximum_drawdown_bps
            - spec.objective_adverse_excursion_penalty
            * metrics.mean_run_maximum_adverse_excursion_bps
            * expected_run_count
        ),
        "total_net_bps_minus_worst_drawdown_and_total_mae_penalties",
    )


def select_round74_action_policy_batches(
    batches: Sequence[Round74EventTrainingBatch],
    candidate_batches: Sequence[Round74ActionCandidateBatch],
    tuning_subpartition: Round74TuningSubpartition,
    *,
    execution_panel: Round74ActionExecutionPanel | None = None,
    optimization_population: str = "capture_run",
) -> Round74ActionPolicySelection:
    """Select one threshold from the complete frozen policy-selection panel."""

    tuning_subpartition.validate()
    selected_population = str(optimization_population)
    if selected_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS:
        raise ValueError("Round 74 action optimization population differs")
    expected_runs = tuning_subpartition.policy_selection_run_ids
    requested_batches = tuple(batches)
    requested_candidate_batches = tuple(candidate_batches)
    provided_runs = {run_id for batch in requested_batches for run_id in batch.run_id}
    if requested_batches and (
        provided_runs != set(expected_runs)
        or any(
            candidates.tuning_subpartition_sha256
            != tuning_subpartition.subpartition_sha256
            for candidates in requested_candidate_batches
        )
    ):
        raise ValueError("Round 74 action policy data role differs")
    selected_batches, selected_candidate_batches = _validate_round74_action_panel(
        requested_batches,
        requested_candidate_batches,
        expected_run_ids=expected_runs,
        required_role="tuning",
        expected_run_count=len(expected_runs),
    )
    first_candidates = selected_candidate_batches[0]
    if (
        first_candidates.tuning_subpartition_sha256
        != tuning_subpartition.subpartition_sha256
    ):
        raise ValueError("Round 74 action policy data role differs")
    spec = round74_action_profile(first_candidates.profile)
    execution_rows = _validated_execution_rows(
        selected_batches,
        selected_candidate_batches,
        execution_panel,
        expected_run_ids=expected_runs,
    )
    scores_by_run: dict[str, list[float]] = {run_id: [] for run_id in expected_runs}
    for candidates in selected_candidate_batches:
        for run_id, score, eligible in zip(
            candidates.run_id,
            candidates.quality_score,
            candidates.eligible,
            strict=True,
        ):
            if eligible:
                scores_by_run[run_id].append(float(score))
    has_active_scores = any(scores_by_run.values())
    evaluations: list[Round74ActionThresholdEvaluation] = []
    if has_active_scores:
        for quantile in spec.threshold_quantiles:
            threshold_function = (
                _equal_run_score_threshold
                if selected_population == "capture_run"
                else _eligible_target_score_threshold
            )
            threshold = threshold_function(
                scores_by_run,
                quantile=quantile,
                expected_run_ids=expected_runs,
            )
            trace = _simulate_round74_action_trace_batches(
                selected_batches,
                selected_candidate_batches,
                threshold_score=threshold,
                expected_run_ids=expected_runs,
                required_role="tuning",
                expected_run_count=len(expected_runs),
                execution_rows=execution_rows,
            )
            reasons = _trace_gate_reasons(trace, spec)
            metrics = trace.metrics
            objective, objective_semantics = _action_selection_objective(
                metrics,
                spec,
                optimization_population=selected_population,
                expected_run_count=len(expected_runs),
            )
            evaluations.append(
                Round74ActionThresholdEvaluation(
                    quantile=quantile,
                    threshold_score=threshold,
                    objective_bps=objective,
                    accepted=not reasons,
                    rejection_reasons=reasons,
                    trace=trace,
                    objective_semantics=objective_semantics,
                )
            )
    else:
        for quantile in spec.threshold_quantiles:
            trace = _simulate_round74_action_trace_batches(
                selected_batches,
                selected_candidate_batches,
                threshold_score=np.finfo(np.float64).max,
                expected_run_ids=expected_runs,
                required_role="tuning",
                expected_run_count=len(expected_runs),
                execution_rows=execution_rows,
            )
            _, objective_semantics = _action_selection_objective(
                trace.metrics,
                spec,
                optimization_population=selected_population,
                expected_run_count=len(expected_runs),
            )
            evaluations.append(
                Round74ActionThresholdEvaluation(
                    quantile=quantile,
                    threshold_score=np.finfo(np.float64).max,
                    objective_bps=0.0,
                    accepted=False,
                    rejection_reasons=("no_target_free_candidates",),
                    trace=trace,
                    objective_semantics=objective_semantics,
                )
            )
    accepted = [value for value in evaluations if value.accepted]
    if accepted:
        selected = max(
            accepted,
            key=lambda value: (
                value.objective_bps,
                -value.trace.metrics.maximum_drawdown_bps,
                -value.trace.metrics.adverse_selection_rate,
                value.quantile,
            ),
        )
        rejection_reasons: tuple[str, ...] = ()
        selected_quantile: float | None = selected.quantile
        selected_threshold: float | None = selected.threshold_score
        did_accept = True
    else:
        rejection_reasons = ("no_policy_threshold_passed_risk_gates",)
        selected_quantile = None
        selected_threshold = None
        did_accept = False
    result = Round74ActionPolicySelection(
        profile=spec.profile,
        pretest_policy_sha256=first_candidates.pretest_policy_sha256,
        probability_calibration_sha256=(
            first_candidates.probability_calibration_sha256
        ),
        tuning_subpartition_sha256=tuning_subpartition.subpartition_sha256,
        target_batch_sha256=tuple(batch.batch_sha256 for batch in selected_batches),
        candidate_sha256=tuple(
            candidates.candidate_sha256 for candidates in selected_candidate_batches
        ),
        accepted=did_accept,
        selected_quantile=selected_quantile,
        selected_threshold_score=selected_threshold,
        evaluations=tuple(evaluations),
        rejection_reasons=rejection_reasons,
        execution_outcome_panel_sha256=(
            execution_panel.panel_sha256 if execution_panel is not None else None
        ),
        optimization_population=selected_population,
    )
    result.validate()
    return result


def select_round74_action_policy(
    batch: Round74EventTrainingBatch,
    candidates: Round74ActionCandidateBatch,
    tuning_subpartition: Round74TuningSubpartition,
    *,
    execution_panel: Round74ActionExecutionPanel | None = None,
    optimization_population: str = "capture_run",
) -> Round74ActionPolicySelection:
    """Select through the panel implementation for single-batch callers."""

    return select_round74_action_policy_batches(
        (batch,),
        (candidates,),
        tuning_subpartition,
        execution_panel=execution_panel,
        optimization_population=optimization_population,
    )


__all__ = [
    "ROUND74_ACTION_CONTEXT_SCHEMA_VERSION",
    "ROUND74_ACTION_DEFAULT_PROFILE",
    "ROUND74_ACTION_EXECUTION_PANEL_SCHEMA_VERSION",
    "ROUND74_ACTION_HORIZONS_SECONDS",
    "ROUND74_ACTION_POLICY_SCHEMA_VERSION",
    "ROUND74_ACTION_POSITION_CAPITAL_FRACTION",
    "ROUND74_ACTION_PROFILES",
    "Round74ActionCandidateBatch",
    "Round74ActionExecutionOutcomeRow",
    "Round74ActionExecutionPanel",
    "Round74ActionInferenceContext",
    "Round74ActionPolicySelection",
    "Round74ActionProfileSpec",
    "Round74ActionThresholdEvaluation",
    "Round74ActionTrace",
    "Round74ActionTraceMetrics",
    "build_round74_action_inference_context",
    "derive_round74_action_candidates",
    "round74_action_profile",
    "select_round74_action_policy",
    "select_round74_action_policy_batches",
    "simulate_round74_action_trace",
    "simulate_round74_action_trace_batches",
]
