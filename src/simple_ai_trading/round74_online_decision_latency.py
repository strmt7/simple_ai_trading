"""Target-free latency evidence for the Round 74 online decision component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns
import warnings

import numpy as np
import torch
from torch import nn

from .compute import BackendInfo
from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionInferenceContext,
    derive_round74_action_candidates,
)
from .impact_absorption_event_calibration import Round74ProbabilityCalibration
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
    Round74EventFeatureScaler,
)


ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION = (
    "round-074-online-decision-latency-v1"
)
ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE = 300
ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE = 16
ROUND74_ONLINE_DECISION_LATENCY_QUANTILE = 0.99
ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE = 0.95
ROUND74_ONLINE_DECISION_LATENCY_MAXIMUM_NS = 5_000_000_000

_SHA256 = set("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    selected = str(value)
    return len(selected) == 64 and set(selected) <= _SHA256


def _module_sha256(filename: str) -> str:
    payload = (Path(__file__).parent / filename).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    selected = np.ascontiguousarray(value)
    selected.setflags(write=False)
    return selected


def _upper_confidence_order_statistic(
    values: Sequence[int],
    *,
    quantile: float,
    confidence: float,
) -> tuple[int, int]:
    selected = tuple(values)
    if (
        not selected
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in selected)
        or not 0.0 < float(quantile) < 1.0
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("Round 74 decision-latency order statistic differs")
    ordered = sorted(selected)
    sample_count = len(ordered)
    cumulative = 0.0
    for below_count in range(sample_count):
        cumulative += (
            math.comb(sample_count, below_count)
            * quantile**below_count
            * (1.0 - quantile) ** (sample_count - below_count)
        )
        if cumulative >= confidence:
            return below_count, ordered[below_count]
    raise ValueError(
        "Round 74 decision-latency sample cannot bound the requested tail"
    )


def _higher_quantile(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(float(quantile) * len(ordered)) - 1)
    return int(ordered[index])


@dataclass(frozen=True)
class Round74ProfileDecisionLatency:
    """Raw samples and recomputable summary for one risk profile."""

    profile: str
    latency_ns: tuple[int, ...]
    p50_ns: int
    p95_ns: int
    p99_upper_confidence_order_index: int
    p99_upper_confidence_ns: int
    maximum_ns: int

    def validate(self) -> None:
        if (
            self.profile not in ROUND74_ACTION_PROFILES
            or len(self.latency_ns)
            != ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= ROUND74_ONLINE_DECISION_LATENCY_MAXIMUM_NS
                for value in self.latency_ns
            )
        ):
            raise ValueError("Round 74 profile decision-latency samples differ")
        order_index, upper = _upper_confidence_order_statistic(
            self.latency_ns,
            quantile=ROUND74_ONLINE_DECISION_LATENCY_QUANTILE,
            confidence=ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE,
        )
        if (
            self.p50_ns != _higher_quantile(self.latency_ns, 0.50)
            or self.p95_ns != _higher_quantile(self.latency_ns, 0.95)
            or self.p99_upper_confidence_order_index != order_index
            or self.p99_upper_confidence_ns != upper
            or self.maximum_ns != max(self.latency_ns)
            or not self.p50_ns
            <= self.p95_ns
            <= self.p99_upper_confidence_ns
            <= self.maximum_ns
        ):
            raise ValueError("Round 74 profile decision-latency summary differs")

    @classmethod
    def from_samples(
        cls,
        profile: str,
        samples: Sequence[int],
    ) -> Round74ProfileDecisionLatency:
        selected = tuple(int(value) for value in samples)
        order_index, upper = _upper_confidence_order_statistic(
            selected,
            quantile=ROUND74_ONLINE_DECISION_LATENCY_QUANTILE,
            confidence=ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE,
        )
        result = cls(
            profile=str(profile),
            latency_ns=selected,
            p50_ns=_higher_quantile(selected, 0.50),
            p95_ns=_higher_quantile(selected, 0.95),
            p99_upper_confidence_order_index=order_index,
            p99_upper_confidence_ns=upper,
            maximum_ns=max(selected),
        )
        result.validate()
        return result

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "profile": self.profile,
            "latency_ns": list(self.latency_ns),
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_upper_confidence_order_index": (
                self.p99_upper_confidence_order_index
            ),
            "p99_upper_confidence_ns": self.p99_upper_confidence_ns,
            "maximum_ns": self.maximum_ns,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74ProfileDecisionLatency:
        payload = dict(value)
        if set(payload) != {
            "profile",
            "latency_ns",
            "p50_ns",
            "p95_ns",
            "p99_upper_confidence_order_index",
            "p99_upper_confidence_ns",
            "maximum_ns",
        } or not isinstance(payload["latency_ns"], list):
            raise ValueError("Round 74 profile decision-latency payload differs")
        integer_fields = (
            "p50_ns",
            "p95_ns",
            "p99_upper_confidence_order_index",
            "p99_upper_confidence_ns",
            "maximum_ns",
        )
        if (
            any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in payload["latency_ns"]
            )
            or any(
                isinstance(payload[name], bool)
                or not isinstance(payload[name], int)
                for name in integer_fields
            )
        ):
            raise ValueError("Round 74 profile decision-latency types differ")
        try:
            result = cls(
                profile=str(payload["profile"]),
                latency_ns=tuple(int(item) for item in payload["latency_ns"]),
                p50_ns=int(payload["p50_ns"]),
                p95_ns=int(payload["p95_ns"]),
                p99_upper_confidence_order_index=int(
                    payload["p99_upper_confidence_order_index"]
                ),
                p99_upper_confidence_ns=int(payload["p99_upper_confidence_ns"]),
                maximum_ns=int(payload["maximum_ns"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 profile decision-latency payload differs"
            ) from exc
        result.validate()
        if result.as_dict() != payload:
            raise ValueError("Round 74 profile decision-latency encoding differs")
        return result


@dataclass(frozen=True)
class Round74OnlineDecisionLatencyEvidence:
    """Host-bound evidence for only the feature-window-to-candidate component."""

    pretest_policy_sha256: str
    pretest_model_sha256: str
    scaler_sha256: str
    probability_calibration_sha256: str
    tuning_subpartition_sha256: str
    source_context_sha256: tuple[str, ...]
    source_feature_row_sha256: tuple[str, ...]
    backend_requested: str
    backend_kind: str
    backend_device: str
    backend_vendor: str
    torch_version: str
    torch_directml_version: str
    warning_count: int
    profiles: tuple[Round74ProfileDecisionLatency, ...]
    decision_latency_module_sha256: str
    action_policy_module_sha256: str
    scaler_module_sha256: str
    schema_version: str = ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION

    def validate(self) -> None:
        digest_values = (
            self.pretest_policy_sha256,
            self.pretest_model_sha256,
            self.scaler_sha256,
            self.probability_calibration_sha256,
            self.tuning_subpartition_sha256,
            *self.source_context_sha256,
            *self.source_feature_row_sha256,
            self.decision_latency_module_sha256,
            self.action_policy_module_sha256,
            self.scaler_module_sha256,
        )
        if (
            self.schema_version != ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION
            or any(not _is_sha256(value) for value in digest_values)
            or not self.source_context_sha256
            or len(self.source_context_sha256) != len(set(self.source_context_sha256))
            or len(self.source_feature_row_sha256)
            != ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE
            or tuple(profile.profile for profile in self.profiles)
            != ROUND74_ACTION_PROFILES
            or any(not str(value).strip() for value in (
                self.backend_requested,
                self.backend_kind,
                self.backend_device,
                self.backend_vendor,
                self.torch_version,
                self.torch_directml_version,
            ))
            or self.backend_kind
            not in {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
            or self.backend_requested
            not in {"auto", "cpu", "cuda", "rocm", "xpu", "directml", "mps"}
            or (
                self.backend_requested != "auto"
                and self.backend_requested != self.backend_kind
            )
            or isinstance(self.warning_count, bool)
            or not isinstance(self.warning_count, int)
            or self.warning_count < 0
        ):
            raise ValueError("Round 74 online decision-latency evidence differs")
        for profile in self.profiles:
            profile.validate()
        if {
            "decision_latency_module_sha256": self.decision_latency_module_sha256,
            "action_policy_module_sha256": self.action_policy_module_sha256,
            "scaler_module_sha256": self.scaler_module_sha256,
        } != {
            "decision_latency_module_sha256": _module_sha256(
                "round74_online_decision_latency.py"
            ),
            "action_policy_module_sha256": _module_sha256(
                "impact_absorption_event_action_policy.py"
            ),
            "scaler_module_sha256": _module_sha256(
                "impact_absorption_event_scaling.py"
            ),
        }:
            raise ValueError("Round 74 decision-latency source identity differs")

    @property
    def evidence_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "identity": {
                "pretest_policy_sha256": self.pretest_policy_sha256,
                "pretest_model_sha256": self.pretest_model_sha256,
                "scaler_sha256": self.scaler_sha256,
                "probability_calibration_sha256": (
                    self.probability_calibration_sha256
                ),
                "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
                "source_context_sha256": list(self.source_context_sha256),
                "source_feature_row_sha256": list(
                    self.source_feature_row_sha256
                ),
            },
            "measurement_contract": {
                "clock": "time.perf_counter_ns",
                "input_boundary": "ready raw 128-event feature window",
                "output_boundary": (
                    "materialized validated profile-specific action candidate"
                ),
                "included": [
                    "tuple-to-numpy conversion",
                    "training-only scaler transform",
                    "single-row causal context construction and validation",
                    "host-to-device tensor transfer",
                    "ensemble forward pass",
                    "device-to-host output materialization",
                    "probability calibration",
                    "profile-specific candidate derivation and validation",
                ],
                "excluded_required_components": [
                    "socket receipt and frame parsing",
                    "event feature extraction before the ready window",
                    "risk coordinator and order construction",
                    "order submission to terminal execution receipt",
                ],
                "warmups_per_profile": (
                    ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE
                ),
                "measurements_per_profile": (
                    ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE
                ),
                "tail_estimator": (
                    "distribution-free one-sided 95 percent upper confidence "
                    "order statistic for p99"
                ),
                "tail_quantile": ROUND74_ONLINE_DECISION_LATENCY_QUANTILE,
                "tail_confidence": ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE,
                "accelerator_completion_barrier": (
                    "blocking device-to-host tensor materialization"
                ),
                "torch_non_blocking_transfer_requested": False,
                "runtime_must_warm_before_trading": True,
                "target_fields_or_outcomes_consumed": False,
            },
            "backend": {
                "requested": self.backend_requested,
                "kind": self.backend_kind,
                "device": self.backend_device,
                "vendor": self.backend_vendor,
                "torch_version": self.torch_version,
                "torch_directml_version": self.torch_directml_version,
                "warning_count": self.warning_count,
            },
            "profiles": [profile.as_dict() for profile in self.profiles],
            "source": {
                "decision_latency_module_sha256": (
                    self.decision_latency_module_sha256
                ),
                "action_policy_module_sha256": self.action_policy_module_sha256,
                "scaler_module_sha256": self.scaler_module_sha256,
            },
            "authority": {
                "component_latency_measured": True,
                "end_to_end_tick_to_trade_latency_measured": False,
                "mainnet_execution_equivalence_claim": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        if include_sha256:
            value["evidence_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74OnlineDecisionLatencyEvidence:
        payload = dict(value)
        claimed = payload.pop("evidence_sha256", None)
        if not _is_sha256(claimed) or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 decision-latency evidence digest differs")
        if set(payload) != {
            "schema_version",
            "identity",
            "measurement_contract",
            "backend",
            "profiles",
            "source",
            "authority",
        }:
            raise ValueError("Round 74 decision-latency payload differs")
        identity = payload["identity"]
        measurement = payload["measurement_contract"]
        backend = payload["backend"]
        profiles = payload["profiles"]
        source = payload["source"]
        authority = payload["authority"]
        if (
            not isinstance(identity, Mapping)
            or not isinstance(measurement, Mapping)
            or not isinstance(backend, Mapping)
            or not isinstance(profiles, list)
            or not isinstance(source, Mapping)
            or not isinstance(authority, Mapping)
        ):
            raise ValueError("Round 74 decision-latency payload types differ")
        expected_measurement = {
            "clock": "time.perf_counter_ns",
            "input_boundary": "ready raw 128-event feature window",
            "output_boundary": (
                "materialized validated profile-specific action candidate"
            ),
            "included": [
                "tuple-to-numpy conversion",
                "training-only scaler transform",
                "single-row causal context construction and validation",
                "host-to-device tensor transfer",
                "ensemble forward pass",
                "device-to-host output materialization",
                "probability calibration",
                "profile-specific candidate derivation and validation",
            ],
            "excluded_required_components": [
                "socket receipt and frame parsing",
                "event feature extraction before the ready window",
                "risk coordinator and order construction",
                "order submission to terminal execution receipt",
            ],
            "warmups_per_profile": (
                ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE
            ),
            "measurements_per_profile": (
                ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE
            ),
            "tail_estimator": (
                "distribution-free one-sided 95 percent upper confidence "
                "order statistic for p99"
            ),
            "tail_quantile": ROUND74_ONLINE_DECISION_LATENCY_QUANTILE,
            "tail_confidence": ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE,
            "accelerator_completion_barrier": (
                "blocking device-to-host tensor materialization"
            ),
            "torch_non_blocking_transfer_requested": False,
            "runtime_must_warm_before_trading": True,
            "target_fields_or_outcomes_consumed": False,
        }
        expected_authority = {
            "component_latency_measured": True,
            "end_to_end_tick_to_trade_latency_measured": False,
            "mainnet_execution_equivalence_claim": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        }
        if (
            dict(measurement) != expected_measurement
            or dict(authority) != expected_authority
            or set(identity)
            != {
                "pretest_policy_sha256",
                "pretest_model_sha256",
                "scaler_sha256",
                "probability_calibration_sha256",
                "tuning_subpartition_sha256",
                "source_context_sha256",
                "source_feature_row_sha256",
            }
            or set(backend)
            != {
                "requested",
                "kind",
                "device",
                "vendor",
                "torch_version",
                "torch_directml_version",
                "warning_count",
            }
            or set(source)
            != {
                "decision_latency_module_sha256",
                "action_policy_module_sha256",
                "scaler_module_sha256",
            }
            or not isinstance(identity["source_context_sha256"], list)
            or not isinstance(identity["source_feature_row_sha256"], list)
            or any(not isinstance(item, Mapping) for item in profiles)
            or isinstance(backend.get("warning_count"), bool)
            or not isinstance(backend.get("warning_count"), int)
        ):
            raise ValueError("Round 74 decision-latency static contract differs")
        try:
            result = cls(
                pretest_policy_sha256=str(identity["pretest_policy_sha256"]),
                pretest_model_sha256=str(identity["pretest_model_sha256"]),
                scaler_sha256=str(identity["scaler_sha256"]),
                probability_calibration_sha256=str(
                    identity["probability_calibration_sha256"]
                ),
                tuning_subpartition_sha256=str(
                    identity["tuning_subpartition_sha256"]
                ),
                source_context_sha256=tuple(
                    str(item) for item in identity["source_context_sha256"]
                ),
                source_feature_row_sha256=tuple(
                    str(item) for item in identity["source_feature_row_sha256"]
                ),
                backend_requested=str(backend["requested"]),
                backend_kind=str(backend["kind"]),
                backend_device=str(backend["device"]),
                backend_vendor=str(backend["vendor"]),
                torch_version=str(backend["torch_version"]),
                torch_directml_version=str(backend["torch_directml_version"]),
                warning_count=int(backend["warning_count"]),
                profiles=tuple(
                    Round74ProfileDecisionLatency.from_dict(item)
                    for item in profiles
                ),
                decision_latency_module_sha256=str(
                    source["decision_latency_module_sha256"]
                ),
                action_policy_module_sha256=str(
                    source["action_policy_module_sha256"]
                ),
                scaler_module_sha256=str(source["scaler_module_sha256"]),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 decision-latency payload differs") from exc
        result.validate()
        if result.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 decision-latency encoding differs")
        if result.evidence_sha256 != claimed:
            raise ValueError("Round 74 decision-latency identity differs")
        return result


def _raw_feature_window(
    scaled: np.ndarray,
    scaler: Round74EventFeatureScaler,
) -> tuple[tuple[float, ...], ...]:
    selected = np.asarray(scaled, dtype=np.float64)
    raw = selected * scaler.scale + scaler.median
    raw[:, :ROUND74_EVENT_BINARY_FEATURE_COUNT] = selected[
        :, :ROUND74_EVENT_BINARY_FEATURE_COUNT
    ]
    raw[:, scaler.constant_mask] = scaler.median[scaler.constant_mask]
    reconstructed = scaler.transform(raw[np.newaxis, ...])[0]
    if not np.array_equal(reconstructed, np.asarray(scaled, dtype=np.float32)):
        raise ValueError("Round 74 decision-latency scaler round trip differs")
    return tuple(tuple(float(value) for value in row) for row in raw)


def _single_row_context(
    source: Round74ActionInferenceContext,
    row_index: int,
    scaled: np.ndarray,
) -> Round74ActionInferenceContext:
    index = int(row_index)

    def integer(name: str) -> np.ndarray:
        value = np.asarray([getattr(source, name)[index]], dtype=np.int64)
        return _readonly(value)

    context = Round74ActionInferenceContext(
        role=source.role,
        partition_sha256=source.partition_sha256,
        scaler_sha256=source.scaler_sha256,
        window_representation=source.window_representation,
        run_id=(source.run_id[index],),
        symbol=(source.symbol[index],),
        decision_monotonic_ns=integer("decision_monotonic_ns"),
        decision_wall_ns=integer("decision_wall_ns"),
        endpoint_frame_index=integer("endpoint_frame_index"),
        endpoint_message_index=integer("endpoint_message_index"),
        anchor_index=integer("anchor_index"),
        sample_sha256=(source.sample_sha256[index],),
        feature_window_sha256=(source.feature_window_sha256[index],),
        feature_values=_readonly(np.asarray(scaled, dtype=np.float32)[np.newaxis, ...]),
        feature_row_sha256=(source.feature_row_sha256[index],),
    )
    context.validate()
    return context


def _timed_candidate(
    model: nn.Module,
    *,
    raw_feature_window: tuple[tuple[float, ...], ...],
    source_context: Round74ActionInferenceContext,
    row_index: int,
    scaler: Round74EventFeatureScaler,
    calibration: Round74ProbabilityCalibration,
    pretest_policy_sha256: str,
    profile: str,
    device: object,
) -> int:
    started = perf_counter_ns()
    raw = np.asarray(raw_feature_window, dtype=np.float64)
    scaled = scaler.transform(raw[np.newaxis, ...])
    context = _single_row_context(source_context, row_index, scaled[0])
    features = torch.from_numpy(np.array(scaled, dtype=np.float32, order="C", copy=True))
    output = model(features.to(device))
    candidate = derive_round74_action_candidates(
        output,
        context,
        calibration,
        pretest_policy_sha256=pretest_policy_sha256,
        profile=profile,
    )
    bool(candidate.eligible[0])
    elapsed = perf_counter_ns() - started
    if elapsed <= 0:
        raise RuntimeError("Round 74 online decision-latency clock did not advance")
    return elapsed


def benchmark_round74_online_decision_latency(
    model: nn.Module,
    *,
    scaler: Round74EventFeatureScaler,
    calibration: Round74ProbabilityCalibration,
    contexts: Sequence[Round74ActionInferenceContext],
    pretest_policy_sha256: str,
    pretest_model_sha256: str,
    backend: BackendInfo,
    device: object,
    torch_directml_version: str,
) -> Round74OnlineDecisionLatencyEvidence:
    """Measure the target-free online model component on the effective backend."""

    selected_contexts = tuple(contexts)
    if (
        not isinstance(model, nn.Module)
        or not isinstance(scaler, Round74EventFeatureScaler)
        or not isinstance(calibration, Round74ProbabilityCalibration)
        or not selected_contexts
        or any(
            not isinstance(context, Round74ActionInferenceContext)
            for context in selected_contexts
        )
        or not _is_sha256(pretest_policy_sha256)
        or not _is_sha256(pretest_model_sha256)
    ):
        raise ValueError("Round 74 online decision-latency inputs differ")
    for context in selected_contexts:
        context.validate()
        if context.scaler_sha256 != scaler.scaler_sha256:
            raise ValueError("Round 74 online decision-latency scaler differs")
    calibration.validate()
    if (
        calibration.pretest_policy_sha256 != pretest_policy_sha256
        or calibration.tuning_subpartition_sha256 == ""
        or not backend.request_satisfied
    ):
        raise ValueError("Round 74 online decision-latency calibration differs")
    rows = tuple(
        (context, row_index)
        for context in selected_contexts
        for row_index in range(context.rows)
    )
    source_rows = tuple(
        rows[index % len(rows)]
        for index in range(ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE)
    )
    raw_windows = tuple(
        _raw_feature_window(context.feature_values[row_index], scaler)
        for context, row_index in source_rows
    )
    source_feature_sha256 = tuple(
        context.feature_row_sha256[row_index] for context, row_index in source_rows
    )
    model.eval()
    warning_messages: list[str] = []
    profiles: list[Round74ProfileDecisionLatency] = []
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for profile in ROUND74_ACTION_PROFILES:
            for warmup_index in range(
                ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE
            ):
                context, row_index = source_rows[warmup_index % len(source_rows)]
                _timed_candidate(
                    model,
                    raw_feature_window=raw_windows[warmup_index % len(raw_windows)],
                    source_context=context,
                    row_index=row_index,
                    scaler=scaler,
                    calibration=calibration,
                    pretest_policy_sha256=pretest_policy_sha256,
                    profile=profile,
                    device=device,
                )
            samples = tuple(
                _timed_candidate(
                    model,
                    raw_feature_window=raw_window,
                    source_context=context,
                    row_index=row_index,
                    scaler=scaler,
                    calibration=calibration,
                    pretest_policy_sha256=pretest_policy_sha256,
                    profile=profile,
                    device=device,
                )
                for raw_window, (context, row_index) in zip(
                    raw_windows,
                    source_rows,
                    strict=True,
                )
            )
            profiles.append(Round74ProfileDecisionLatency.from_samples(profile, samples))
        warning_messages.extend(str(item.message) for item in caught)
    fallback = tuple(
        message
        for message in warning_messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    )
    if fallback:
        raise RuntimeError(f"Round 74 decision path used CPU fallback: {fallback}")
    result = Round74OnlineDecisionLatencyEvidence(
        pretest_policy_sha256=pretest_policy_sha256,
        pretest_model_sha256=pretest_model_sha256,
        scaler_sha256=scaler.scaler_sha256,
        probability_calibration_sha256=calibration.calibration_sha256,
        tuning_subpartition_sha256=calibration.tuning_subpartition_sha256,
        source_context_sha256=tuple(
            context.context_sha256 for context in selected_contexts
        ),
        source_feature_row_sha256=source_feature_sha256,
        backend_requested=backend.requested,
        backend_kind=backend.kind,
        backend_device=str(device),
        backend_vendor=backend.vendor,
        torch_version=str(torch.__version__),
        torch_directml_version=str(torch_directml_version),
        warning_count=len(warning_messages),
        profiles=tuple(profiles),
        decision_latency_module_sha256=_module_sha256(
            "round74_online_decision_latency.py"
        ),
        action_policy_module_sha256=_module_sha256(
            "impact_absorption_event_action_policy.py"
        ),
        scaler_module_sha256=_module_sha256("impact_absorption_event_scaling.py"),
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE",
    "ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE",
    "ROUND74_ONLINE_DECISION_LATENCY_QUANTILE",
    "ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION",
    "ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE",
    "Round74OnlineDecisionLatencyEvidence",
    "Round74ProfileDecisionLatency",
    "benchmark_round74_online_decision_latency",
]
