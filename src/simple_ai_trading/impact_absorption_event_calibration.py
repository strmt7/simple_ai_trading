"""Chronological tuning split and probability calibration for Round 74."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
)


ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION = "round-074-tuning-subpartition-v1"
ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION = "round-074-temperature-calibration-v1"
ROUND74_TUNING_EXPECTED_RUNS = 24
ROUND74_TUNING_MODEL_SELECTION_RUNS = 12
ROUND74_TUNING_CALIBRATION_RUNS = 6
ROUND74_TUNING_POLICY_SELECTION_RUNS = 6
ROUND74_TEMPERATURE_MINIMUM = 0.05
ROUND74_TEMPERATURE_MAXIMUM = 20.0
ROUND74_TEMPERATURE_CANDIDATE_COUNT = 257
ROUND74_TEMPERATURE_ECE_BINS = 20


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Round 74 calibration {label} digest differs")
    return str(value)


@dataclass(frozen=True)
class Round74TuningSubpartition:
    """Disjoint chronological tuning roles fixed before target access."""

    parent_partition_sha256: str
    model_selection_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    policy_selection_run_ids: tuple[str, ...]
    schema_version: str = ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION

    def validate(self) -> None:
        groups = (
            self.model_selection_run_ids,
            self.calibration_run_ids,
            self.policy_selection_run_ids,
        )
        expected_counts = (
            ROUND74_TUNING_MODEL_SELECTION_RUNS,
            ROUND74_TUNING_CALIBRATION_RUNS,
            ROUND74_TUNING_POLICY_SELECTION_RUNS,
        )
        if (
            self.schema_version != ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION
            or tuple(len(group) for group in groups) != expected_counts
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for group in groups
                for value in group
            )
            or len({value for group in groups for value in group})
            != ROUND74_TUNING_EXPECTED_RUNS
        ):
            raise ValueError("Round 74 tuning subpartition differs")
        _require_sha256(self.parent_partition_sha256, "parent partition")

    @property
    def subpartition_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_partition_sha256": self.parent_partition_sha256,
            "split_unit": "whole_capture_run",
            "chronological": True,
            "random_row_split_permitted": False,
            "model_selection_run_ids": list(self.model_selection_run_ids),
            "calibration_run_ids": list(self.calibration_run_ids),
            "policy_selection_run_ids": list(self.policy_selection_run_ids),
            "sealed_test_run_accessed": False,
        }
        if include_sha256:
            payload["subpartition_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74TuningSubpartition:
        payload = dict(value)
        claimed = str(payload.pop("subpartition_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 tuning subpartition digest differs")
        try:
            selected = cls(
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                model_selection_run_ids=tuple(
                    str(item) for item in payload["model_selection_run_ids"]
                ),
                calibration_run_ids=tuple(
                    str(item) for item in payload["calibration_run_ids"]
                ),
                policy_selection_run_ids=tuple(
                    str(item) for item in payload["policy_selection_run_ids"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 tuning subpartition payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 tuning subpartition policy differs")
        selected.validate()
        return selected


def build_round74_tuning_subpartition(
    partition: Round74EventRunPartition,
) -> Round74TuningSubpartition:
    """Split exactly 24 chronological tuning runs without row reuse."""

    partition.validate()
    run_ids = tuple(
        entry.run_id for entry in partition.entries if entry.role == "tuning"
    )
    if len(run_ids) != ROUND74_TUNING_EXPECTED_RUNS:
        raise ValueError("Round 74 tuning run count differs")
    model_end = ROUND74_TUNING_MODEL_SELECTION_RUNS
    calibration_end = model_end + ROUND74_TUNING_CALIBRATION_RUNS
    selected = Round74TuningSubpartition(
        parent_partition_sha256=partition.partition_sha256,
        model_selection_run_ids=run_ids[:model_end],
        calibration_run_ids=run_ids[model_end:calibration_end],
        policy_selection_run_ids=run_ids[calibration_end:],
    )
    selected.validate()
    return selected


@dataclass(frozen=True)
class Round74TemperatureFit:
    """One bounded scalar temperature selected on calibration data."""

    temperature: float
    eligible_observations: int
    positive_observations: int
    uncalibrated_nll: float
    calibrated_nll: float
    uncalibrated_brier: float
    calibrated_brier: float
    uncalibrated_ece: float
    calibrated_ece: float

    def validate(self) -> None:
        metrics = (
            self.temperature,
            self.uncalibrated_nll,
            self.calibrated_nll,
            self.uncalibrated_brier,
            self.calibrated_brier,
            self.uncalibrated_ece,
            self.calibrated_ece,
        )
        if (
            any(not math.isfinite(float(value)) for value in metrics)
            or not ROUND74_TEMPERATURE_MINIMUM
            <= self.temperature
            <= ROUND74_TEMPERATURE_MAXIMUM
            or isinstance(self.eligible_observations, bool)
            or not isinstance(self.eligible_observations, int)
            or self.eligible_observations < 2
            or isinstance(self.positive_observations, bool)
            or not isinstance(self.positive_observations, int)
            or not 0 < self.positive_observations < self.eligible_observations
            or self.calibrated_nll > self.uncalibrated_nll + 1e-7
            or min(
                self.uncalibrated_nll,
                self.calibrated_nll,
                self.uncalibrated_brier,
                self.calibrated_brier,
                self.uncalibrated_ece,
                self.calibrated_ece,
            )
            < 0.0
        ):
            raise ValueError("Round 74 temperature fit differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "temperature": self.temperature,
            "eligible_observations": self.eligible_observations,
            "positive_observations": self.positive_observations,
            "uncalibrated_nll": self.uncalibrated_nll,
            "calibrated_nll": self.calibrated_nll,
            "uncalibrated_brier": self.uncalibrated_brier,
            "calibrated_brier": self.calibrated_brier,
            "uncalibrated_ece": self.uncalibrated_ece,
            "calibrated_ece": self.calibrated_ece,
        }


@dataclass(frozen=True)
class Round74ProbabilityCalibration:
    """Hash-bound temperatures for the three probability head families."""

    pretest_policy_sha256: str
    tuning_subpartition_sha256: str
    calibration_source_sha256: str
    calibration_data_sha256: str
    positive_payoff: Round74TemperatureFit
    adverse_selection: Round74TemperatureFit
    regime_unpredictability: Round74TemperatureFit
    backend_kind: str
    backend_device: str
    schema_version: str = ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
            or not self.backend_kind.strip()
            or not self.backend_device.strip()
        ):
            raise ValueError("Round 74 probability calibration differs")
        _require_sha256(self.pretest_policy_sha256, "pretest policy")
        _require_sha256(
            self.tuning_subpartition_sha256,
            "tuning subpartition",
        )
        _require_sha256(
            self.calibration_source_sha256,
            "calibration source",
        )
        _require_sha256(self.calibration_data_sha256, "calibration data")
        self.positive_payoff.validate()
        self.adverse_selection.validate()
        self.regime_unpredictability.validate()

    @property
    def calibration_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "tuning_subpartition_sha256": (self.tuning_subpartition_sha256),
            "calibration_source_sha256": (self.calibration_source_sha256),
            "calibration_data_sha256": self.calibration_data_sha256,
            "positive_payoff": self.positive_payoff.as_dict(),
            "adverse_selection": self.adverse_selection.as_dict(),
            "regime_unpredictability": (self.regime_unpredictability.as_dict()),
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "candidate_temperature_count": (ROUND74_TEMPERATURE_CANDIDATE_COUNT),
            "candidate_temperature_minimum": (ROUND74_TEMPERATURE_MINIMUM),
            "candidate_temperature_maximum": (ROUND74_TEMPERATURE_MAXIMUM),
            "selection_objective": ("binary_cross_entropy_on_calibration_runs_only"),
            "sealed_test_accessed": False,
            "calibration_implies_financial_edge": False,
        }
        if include_sha256:
            payload["calibration_sha256"] = _canonical_sha256(payload)
        return payload


def _validate_binary_panel(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eligibility: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logits.shape != labels.shape
        or logits.shape != eligibility.shape
        or not logits.is_floating_point()
        or not labels.is_floating_point()
        or not eligibility.is_floating_point()
        or logits.device != labels.device
        or logits.device != eligibility.device
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(labels).all())
        or not bool(torch.isfinite(eligibility).all())
        or bool(((eligibility != 0.0) & (eligibility != 1.0)).any())
        or bool(((labels < 0.0) | (labels > 1.0)).any())
    ):
        raise ValueError(f"Round 74 {label} calibration panel differs")
    mask = eligibility == 1.0
    selected_logits = logits.detach()[mask].to(dtype=torch.float32)
    selected_labels = labels.detach()[mask].to(dtype=torch.float32)
    if (
        selected_logits.numel() < 2
        or not bool((selected_labels == 0.0).any())
        or not bool((selected_labels == 1.0).any())
    ):
        raise ValueError(f"Round 74 {label} calibration class support differs")
    return selected_logits, selected_labels


def _expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    boundaries = torch.linspace(
        0.0,
        1.0,
        ROUND74_TEMPERATURE_ECE_BINS + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    total = float(probabilities.numel())
    ece = torch.zeros(
        (),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    for index in range(ROUND74_TEMPERATURE_ECE_BINS):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index + 1 == ROUND74_TEMPERATURE_ECE_BINS:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum().item())
        if count:
            confidence = probabilities[mask].mean()
            frequency = labels[mask].mean()
            ece = ece + abs(confidence - frequency) * (count / total)
    return float(ece.item())


def _fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Round74TemperatureFit:
    log_temperatures = torch.linspace(
        math.log(ROUND74_TEMPERATURE_MINIMUM),
        math.log(ROUND74_TEMPERATURE_MAXIMUM),
        ROUND74_TEMPERATURE_CANDIDATE_COUNT,
        dtype=logits.dtype,
        device=logits.device,
    )
    temperatures = torch.exp(log_temperatures)
    losses: list[torch.Tensor] = []
    for chunk in temperatures.split(16):
        scaled = logits.unsqueeze(0) / chunk.unsqueeze(1)
        target = labels.unsqueeze(0).expand_as(scaled)
        losses.append(
            (F.softplus(scaled) - target * scaled).mean(dim=1)
        )
    candidate_loss = torch.cat(losses)
    selected_index = int(torch.argmin(candidate_loss).item())
    temperature = float(temperatures[selected_index].item())
    uncalibrated_probability = torch.sigmoid(logits)
    calibrated_probability = torch.sigmoid(logits / temperature)
    uncalibrated_nll = float(
        (F.softplus(logits) - labels * logits).mean().item()
    )
    calibrated_nll = float(candidate_loss[selected_index].item())
    fit = Round74TemperatureFit(
        temperature=temperature,
        eligible_observations=int(labels.numel()),
        positive_observations=int((labels == 1.0).sum().item()),
        uncalibrated_nll=uncalibrated_nll,
        calibrated_nll=calibrated_nll,
        uncalibrated_brier=float(
            torch.mean((uncalibrated_probability - labels) ** 2).item()
        ),
        calibrated_brier=float(
            torch.mean((calibrated_probability - labels) ** 2).item()
        ),
        uncalibrated_ece=_expected_calibration_error(
            uncalibrated_probability,
            labels,
        ),
        calibrated_ece=_expected_calibration_error(
            calibrated_probability,
            labels,
        ),
    )
    fit.validate()
    return fit


def _update_tensor_digest(
    digest: object,
    value: torch.Tensor,
) -> None:
    array = np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(int(array.ndim).to_bytes(2, "little", signed=False))
    for size in array.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(array).cast("B"))


def fit_round74_probability_calibration(
    *,
    positive_payoff_logits: torch.Tensor,
    positive_payoff_labels: torch.Tensor,
    adverse_selection_logits: torch.Tensor,
    adverse_selection_labels: torch.Tensor,
    action_eligibility: torch.Tensor,
    regime_unpredictability_logits: torch.Tensor,
    regime_unpredictability_labels: torch.Tensor,
    regime_eligibility: torch.Tensor,
    pretest_policy_sha256: str,
    tuning_subpartition_sha256: str,
    calibration_source_sha256: str,
    backend_kind: str,
    backend_device: str,
) -> Round74ProbabilityCalibration:
    """Fit three scalar temperatures using only calibration-run labels."""

    _require_sha256(pretest_policy_sha256, "pretest policy")
    _require_sha256(
        tuning_subpartition_sha256,
        "tuning subpartition",
    )
    _require_sha256(calibration_source_sha256, "calibration source")
    positive_logits, positive_labels = _validate_binary_panel(
        positive_payoff_logits,
        positive_payoff_labels,
        action_eligibility,
        label="positive-payoff",
    )
    adverse_logits, adverse_labels = _validate_binary_panel(
        adverse_selection_logits,
        adverse_selection_labels,
        action_eligibility,
        label="adverse-selection",
    )
    regime_logits, regime_labels = _validate_binary_panel(
        regime_unpredictability_logits,
        regime_unpredictability_labels,
        regime_eligibility,
        label="regime-unpredictability",
    )
    devices = {
        positive_logits.device,
        adverse_logits.device,
        regime_logits.device,
    }
    if len(devices) != 1:
        raise ValueError("Round 74 calibration devices differ")
    identity = {
        "schema_version": ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
        "pretest_policy_sha256": pretest_policy_sha256,
        "tuning_subpartition_sha256": tuning_subpartition_sha256,
        "calibration_source_sha256": calibration_source_sha256,
        "backend_kind": str(backend_kind),
        "backend_device": str(backend_device),
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
    for value in (
        positive_payoff_logits,
        positive_payoff_labels,
        adverse_selection_logits,
        adverse_selection_labels,
        action_eligibility,
        regime_unpredictability_logits,
        regime_unpredictability_labels,
        regime_eligibility,
    ):
        _update_tensor_digest(digest, value)
    result = Round74ProbabilityCalibration(
        pretest_policy_sha256=pretest_policy_sha256,
        tuning_subpartition_sha256=tuning_subpartition_sha256,
        calibration_source_sha256=calibration_source_sha256,
        calibration_data_sha256=digest.hexdigest(),
        positive_payoff=_fit_temperature(
            positive_logits,
            positive_labels,
        ),
        adverse_selection=_fit_temperature(
            adverse_logits,
            adverse_labels,
        ),
        regime_unpredictability=_fit_temperature(
            regime_logits,
            regime_labels,
        ),
        backend_kind=str(backend_kind),
        backend_device=str(backend_device),
    )
    result.validate()
    return result


def apply_round74_probability_calibration(
    calibration: Round74ProbabilityCalibration,
    *,
    positive_payoff_logits: torch.Tensor,
    adverse_selection_logits: torch.Tensor,
    regime_unpredictability_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply frozen temperatures without fitting or target access."""

    calibration.validate()
    tensors = (
        positive_payoff_logits,
        adverse_selection_logits,
        regime_unpredictability_logits,
    )
    if (
        any(not value.is_floating_point() for value in tensors)
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
        or len({value.device for value in tensors}) != 1
    ):
        raise ValueError("Round 74 calibration inference input differs")
    return (
        torch.sigmoid(positive_payoff_logits / calibration.positive_payoff.temperature),
        torch.sigmoid(
            adverse_selection_logits / calibration.adverse_selection.temperature
        ),
        torch.sigmoid(
            regime_unpredictability_logits
            / calibration.regime_unpredictability.temperature
        ),
    )


__all__ = [
    "ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CANDIDATE_COUNT",
    "ROUND74_TEMPERATURE_MAXIMUM",
    "ROUND74_TEMPERATURE_MINIMUM",
    "ROUND74_TUNING_CALIBRATION_RUNS",
    "ROUND74_TUNING_EXPECTED_RUNS",
    "ROUND74_TUNING_MODEL_SELECTION_RUNS",
    "ROUND74_TUNING_POLICY_SELECTION_RUNS",
    "ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION",
    "Round74ProbabilityCalibration",
    "Round74TemperatureFit",
    "Round74TuningSubpartition",
    "apply_round74_probability_calibration",
    "build_round74_tuning_subpartition",
    "fit_round74_probability_calibration",
]
