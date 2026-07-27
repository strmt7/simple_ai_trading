"""Coordinate post-cohort Round 74 training, calibration, and policy selection.

This module is development-only. It cannot load a sealed test batch, place an
order, or grant paper, testnet, or live execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionPolicySelection,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    select_round74_action_policy_batches,
)
from .impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    build_round74_tuning_subpartition,
    fit_round74_probability_calibration,
)
from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventTrainingBatch,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_scaling import Round74EventFeatureScaler
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
)
from .impact_absorption_event_training import (
    Round74EventTrainingConfig,
    Round74PretestPolicyArtifact,
    load_round74_pretest_policy,
    train_and_seal_round74_pretest_policy_from_prepared_roles,
)
from .round74_event_model_operator import (
    Round74PreparedDevelopmentData,
    Round74PreparedTuningRoles,
    prepare_round74_development_data,
    split_round74_prepared_tuning_roles,
)
from .round74_event_development_inputs import Round74DevelopmentInputs
from .round74_delayed_execution_panel import (
    build_round74_delayed_execution_panels,
)
from .round74_online_decision_latency import (
    Round74OnlineDecisionLatencyEvidence,
    benchmark_round74_online_decision_latency,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .storage import write_bytes_atomic


ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION = "round-074-development-policy-operator-v2"
ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION = (
    "round-074-development-policy-bundle-v3"
)

_SHA256 = "0123456789abcdef"


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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _module_sha256(filename: str) -> str:
    payload = (Path(__file__).parent / filename).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _current_source_sha256() -> dict[str, str]:
    return {
        "event_model_module_sha256": _module_sha256("impact_absorption_event_model.py"),
        "calibration_module_sha256": _module_sha256(
            "impact_absorption_event_calibration.py"
        ),
        "action_policy_module_sha256": _module_sha256(
            "impact_absorption_event_action_policy.py"
        ),
        "decision_latency_module_sha256": _module_sha256(
            "round74_online_decision_latency.py"
        ),
        "delayed_execution_module_sha256": _module_sha256(
            "round74_delayed_execution_panel.py"
        ),
        "development_operator_module_sha256": _module_sha256(
            "round74_event_development_operator.py"
        ),
    }


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key, value in pairs:
        if key in selected:
            raise ValueError(f"Round 74 development duplicate JSON key: {key}")
        selected[key] = value
    return selected


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"Round 74 development non-finite JSON value: {value}")


def _to_device_array(value: np.ndarray, device: object) -> torch.Tensor:
    copied = np.array(value, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(copied).to(device)


def _concatenate_outputs(
    outputs: Sequence[Round74EventModelOutput],
) -> Round74EventModelOutput:
    selected = tuple(outputs)
    if not selected:
        raise ValueError("Round 74 development model output is empty")

    def concatenate(name: str) -> torch.Tensor:
        return torch.cat(tuple(getattr(value, name) for value in selected), dim=0)

    result = Round74EventModelOutput(
        payoff_quantiles_bps=concatenate("payoff_quantiles_bps"),
        maximum_adverse_excursion_quantiles_bps=concatenate(
            "maximum_adverse_excursion_quantiles_bps"
        ),
        positive_payoff_logits=concatenate("positive_payoff_logits"),
        adverse_selection_logits=concatenate("adverse_selection_logits"),
        regime_unpredictability_logits=concatenate("regime_unpredictability_logits"),
    )
    result.validate(int(result.payoff_quantiles_bps.shape[0]))
    return result


def _infer_batch(
    model: nn.Module,
    batch: Round74EventTrainingBatch,
    *,
    minibatch_rows: int,
    device: object,
) -> Round74EventModelOutput:
    batch.validate()
    if (
        isinstance(minibatch_rows, bool)
        or not isinstance(minibatch_rows, int)
        or minibatch_rows < 1
    ):
        raise ValueError("Round 74 development inference minibatch differs")
    outputs: list[Round74EventModelOutput] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, batch.rows, minibatch_rows):
            stop = min(start + minibatch_rows, batch.rows)
            features = _to_device_array(batch.feature_values[start:stop], device)
            outputs.append(model(features))
    result = _concatenate_outputs(outputs)
    if int(result.payoff_quantiles_bps.shape[0]) != batch.rows:
        raise RuntimeError("Round 74 development inference row count differs")
    return result


def _calibration_source_sha256(
    *,
    policy: Mapping[str, object],
    roles: Round74PreparedTuningRoles,
) -> str:
    artifact = policy.get("model_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("Round 74 development model artifact differs")
    value = {
        "schema_version": ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION,
        "pretest_policy_sha256": policy.get("policy_sha256"),
        "model_sha256": artifact.get("sha256"),
        "tuning_subpartition_sha256": roles.subpartition.subpartition_sha256,
        "calibration_batch_sha256": [
            batch.batch_sha256 for batch in roles.calibration_batches
        ],
        "event_model_module_sha256": _module_sha256("impact_absorption_event_model.py"),
        "calibration_module_sha256": _module_sha256(
            "impact_absorption_event_calibration.py"
        ),
        "development_operator_module_sha256": _module_sha256(
            "round74_event_development_operator.py"
        ),
    }
    return _canonical_sha256(value)


@dataclass(frozen=True)
class Round74DevelopmentPolicyBundle:
    """Hash-bound development result for all three risk profiles."""

    pretest_policy_sha256: str
    pretest_model_sha256: str
    feature_scaler_sha256: str
    tuning_subpartition_sha256: str
    model_selection_batch_sha256: tuple[str, ...]
    calibration_batch_sha256: tuple[str, ...]
    policy_selection_batch_sha256: tuple[str, ...]
    probability_calibration: Round74ProbabilityCalibration
    online_decision_latency: Round74OnlineDecisionLatencyEvidence
    execution_outcome_panel_sha256: tuple[str, ...]
    execution_outcome_panel_rows: tuple[int, ...]
    action_policies: tuple[Round74ActionPolicySelection, ...]
    backend_kind: str
    backend_device: str
    backend_vendor: str
    warning_count: int
    event_model_module_sha256: str
    calibration_module_sha256: str
    action_policy_module_sha256: str
    decision_latency_module_sha256: str
    delayed_execution_module_sha256: str
    development_operator_module_sha256: str
    schema_version: str = ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION

    def validate(self) -> None:
        digest_values = (
            self.pretest_policy_sha256,
            self.pretest_model_sha256,
            self.feature_scaler_sha256,
            self.tuning_subpartition_sha256,
            *self.model_selection_batch_sha256,
            *self.calibration_batch_sha256,
            *self.policy_selection_batch_sha256,
            *self.execution_outcome_panel_sha256,
            self.event_model_module_sha256,
            self.calibration_module_sha256,
            self.action_policy_module_sha256,
            self.decision_latency_module_sha256,
            self.delayed_execution_module_sha256,
            self.development_operator_module_sha256,
        )
        if (
            self.schema_version != ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION
            or any(not _is_sha256(value) for value in digest_values)
            or len(self.model_selection_batch_sha256) != 12
            or len(self.calibration_batch_sha256) != 6
            or len(self.policy_selection_batch_sha256) != 6
            or len(self.execution_outcome_panel_sha256)
            != len(ROUND74_ACTION_PROFILES)
            or len(self.execution_outcome_panel_rows)
            != len(ROUND74_ACTION_PROFILES)
            or len(set(self.execution_outcome_panel_sha256))
            != len(ROUND74_ACTION_PROFILES)
            or len(set(self.execution_outcome_panel_rows)) != 1
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.execution_outcome_panel_rows
            )
            or len(
                set(
                    (
                        *self.model_selection_batch_sha256,
                        *self.calibration_batch_sha256,
                        *self.policy_selection_batch_sha256,
                    )
                )
            )
            != 24
            or not self.backend_kind.strip()
            or not self.backend_device.strip()
            or not self.backend_vendor.strip()
            or isinstance(self.warning_count, bool)
            or self.warning_count < 0
        ):
            raise ValueError("Round 74 development policy bundle differs")
        if {
            "event_model_module_sha256": self.event_model_module_sha256,
            "calibration_module_sha256": self.calibration_module_sha256,
            "action_policy_module_sha256": self.action_policy_module_sha256,
            "decision_latency_module_sha256": (
                self.decision_latency_module_sha256
            ),
            "delayed_execution_module_sha256": (
                self.delayed_execution_module_sha256
            ),
            "development_operator_module_sha256": (
                self.development_operator_module_sha256
            ),
        } != _current_source_sha256():
            raise ValueError("Round 74 development policy source identity differs")
        self.probability_calibration.validate()
        self.online_decision_latency.validate()
        if (
            self.probability_calibration.pretest_policy_sha256
            != self.pretest_policy_sha256
            or self.probability_calibration.tuning_subpartition_sha256
            != self.tuning_subpartition_sha256
            or tuple(policy.profile for policy in self.action_policies)
            != ROUND74_ACTION_PROFILES
            or self.online_decision_latency.pretest_policy_sha256
            != self.pretest_policy_sha256
            or self.online_decision_latency.pretest_model_sha256
            != self.pretest_model_sha256
            or self.online_decision_latency.scaler_sha256
            != self.feature_scaler_sha256
            or self.online_decision_latency.probability_calibration_sha256
            != self.probability_calibration.calibration_sha256
            or self.online_decision_latency.tuning_subpartition_sha256
            != self.tuning_subpartition_sha256
            or self.online_decision_latency.backend_kind != self.backend_kind
            or self.online_decision_latency.backend_device != self.backend_device
            or self.online_decision_latency.backend_vendor != self.backend_vendor
        ):
            raise ValueError("Round 74 development policy identity differs")
        for policy in self.action_policies:
            policy.validate()
            if (
                policy.pretest_policy_sha256 != self.pretest_policy_sha256
                or policy.probability_calibration_sha256
                != self.probability_calibration.calibration_sha256
                or policy.tuning_subpartition_sha256 != self.tuning_subpartition_sha256
                or policy.target_batch_sha256 != self.policy_selection_batch_sha256
                or policy.execution_outcome_panel_sha256
                != self.execution_outcome_panel_sha256[
                    ROUND74_ACTION_PROFILES.index(policy.profile)
                ]
            ):
                raise ValueError("Round 74 development action policy differs")

    @property
    def bundle_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "operator_schema_version": ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "pretest_model_sha256": self.pretest_model_sha256,
            "feature_scaler_sha256": self.feature_scaler_sha256,
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "model_selection_batch_sha256": list(self.model_selection_batch_sha256),
            "calibration_batch_sha256": list(self.calibration_batch_sha256),
            "policy_selection_batch_sha256": list(self.policy_selection_batch_sha256),
            "probability_calibration": self.probability_calibration.as_dict(),
            "online_decision_latency": self.online_decision_latency.as_dict(),
            "execution_outcome_panels": [
                {
                    "profile": profile,
                    "panel_sha256": panel_sha256,
                    "row_count": rows,
                    "target_outcome_count": (
                        rows
                        * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
                        * len(ROUND74_EVENT_PAYOFF_SIDES)
                    ),
                    "full_outcome_rows_persisted": False,
                    "reproduction_source": (
                        "read-only captured event store and source target assemblies"
                    ),
                }
                for profile, panel_sha256, rows in zip(
                    ROUND74_ACTION_PROFILES,
                    self.execution_outcome_panel_sha256,
                    self.execution_outcome_panel_rows,
                    strict=True,
                )
            ],
            "action_policies": [policy.as_dict() for policy in self.action_policies],
            "backend": {
                "kind": self.backend_kind,
                "device": self.backend_device,
                "vendor": self.backend_vendor,
                "warning_count": self.warning_count,
            },
            "source": {
                "event_model_module_sha256": self.event_model_module_sha256,
                "calibration_module_sha256": self.calibration_module_sha256,
                "action_policy_module_sha256": self.action_policy_module_sha256,
                "decision_latency_module_sha256": (
                    self.decision_latency_module_sha256
                ),
                "delayed_execution_module_sha256": (
                    self.delayed_execution_module_sha256
                ),
                "development_operator_module_sha256": (
                    self.development_operator_module_sha256
                ),
            },
            "authority": {
                "representative_market_training_completed": True,
                "sealed_test_accessed": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
                "ai_uplift_claim": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        if include_sha256:
            value["bundle_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74DevelopmentPolicyBundle:
        payload = dict(value)
        claimed = payload.pop("bundle_sha256", None)
        if not _is_sha256(claimed) or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 development policy digest differs")
        expected_keys = {
            "schema_version",
            "operator_schema_version",
            "pretest_policy_sha256",
            "pretest_model_sha256",
            "feature_scaler_sha256",
            "tuning_subpartition_sha256",
            "model_selection_batch_sha256",
            "calibration_batch_sha256",
            "policy_selection_batch_sha256",
            "probability_calibration",
            "online_decision_latency",
            "execution_outcome_panels",
            "action_policies",
            "backend",
            "source",
            "authority",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 development policy payload differs")

        def strings(name: str) -> tuple[str, ...]:
            values = payload[name]
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise ValueError("Round 74 development policy sequence differs")
            return tuple(values)

        calibration = payload["probability_calibration"]
        latency = payload["online_decision_latency"]
        execution_panels = payload["execution_outcome_panels"]
        policies = payload["action_policies"]
        backend = payload["backend"]
        source = payload["source"]
        if (
            not isinstance(calibration, Mapping)
            or not isinstance(latency, Mapping)
            or not isinstance(execution_panels, list)
            or len(execution_panels) != len(ROUND74_ACTION_PROFILES)
            or any(not isinstance(item, Mapping) for item in execution_panels)
            or not isinstance(policies, list)
            or any(not isinstance(item, Mapping) for item in policies)
            or not isinstance(backend, Mapping)
            or not isinstance(source, Mapping)
        ):
            raise ValueError("Round 74 development policy types differ")
        warning_count = backend.get("warning_count")
        if isinstance(warning_count, bool) or not isinstance(warning_count, int):
            raise ValueError("Round 74 development warning count differs")
        expected_panel_keys = {
            "profile",
            "panel_sha256",
            "row_count",
            "target_outcome_count",
            "full_outcome_rows_persisted",
            "reproduction_source",
        }
        parsed_panel_sha256: list[str] = []
        parsed_panel_rows: list[int] = []
        for profile, raw_panel in zip(
            ROUND74_ACTION_PROFILES,
            execution_panels,
            strict=True,
        ):
            assert isinstance(raw_panel, Mapping)
            panel = dict(raw_panel)
            rows = panel.get("row_count")
            if (
                set(panel) != expected_panel_keys
                or panel.get("profile") != profile
                or not _is_sha256(panel.get("panel_sha256"))
                or isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 1
                or panel.get("target_outcome_count")
                != rows
                * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
                * len(ROUND74_EVENT_PAYOFF_SIDES)
                or panel.get("full_outcome_rows_persisted") is not False
                or panel.get("reproduction_source")
                != "read-only captured event store and source target assemblies"
            ):
                raise ValueError("Round 74 execution outcome panel differs")
            parsed_panel_sha256.append(str(panel["panel_sha256"]))
            parsed_panel_rows.append(rows)
        try:
            selected = cls(
                pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
                pretest_model_sha256=str(payload["pretest_model_sha256"]),
                feature_scaler_sha256=str(payload["feature_scaler_sha256"]),
                tuning_subpartition_sha256=str(payload["tuning_subpartition_sha256"]),
                model_selection_batch_sha256=strings("model_selection_batch_sha256"),
                calibration_batch_sha256=strings("calibration_batch_sha256"),
                policy_selection_batch_sha256=strings("policy_selection_batch_sha256"),
                probability_calibration=Round74ProbabilityCalibration.from_dict(
                    calibration
                ),
                online_decision_latency=(
                    Round74OnlineDecisionLatencyEvidence.from_dict(latency)
                ),
                execution_outcome_panel_sha256=tuple(parsed_panel_sha256),
                execution_outcome_panel_rows=tuple(parsed_panel_rows),
                action_policies=tuple(
                    Round74ActionPolicySelection.from_dict(item) for item in policies
                ),
                backend_kind=str(backend["kind"]),
                backend_device=str(backend["device"]),
                backend_vendor=str(backend["vendor"]),
                warning_count=warning_count,
                event_model_module_sha256=str(source["event_model_module_sha256"]),
                calibration_module_sha256=str(source["calibration_module_sha256"]),
                action_policy_module_sha256=str(source["action_policy_module_sha256"]),
                decision_latency_module_sha256=str(
                    source["decision_latency_module_sha256"]
                ),
                delayed_execution_module_sha256=str(
                    source["delayed_execution_module_sha256"]
                ),
                development_operator_module_sha256=str(
                    source["development_operator_module_sha256"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 development policy payload differs") from exc
        selected.validate()
        if _canonical_bytes(selected.as_dict(include_sha256=False)) != _canonical_bytes(
            payload
        ):
            raise ValueError("Round 74 development policy contract differs")
        if selected.bundle_sha256 != claimed:
            raise ValueError("Round 74 development policy identity differs")
        return selected


@dataclass(frozen=True)
class Round74DevelopmentPolicyArtifact:
    bundle_sha256: str
    bundle_path: Path
    pretest_policy: Round74PretestPolicyArtifact
    bundle: Round74DevelopmentPolicyBundle


def _fit_calibration(
    outputs: Sequence[Round74EventModelOutput],
    batches: Sequence[Round74EventTrainingBatch],
    *,
    roles: Round74PreparedTuningRoles,
    policy: Mapping[str, object],
    backend_kind: str,
    backend_device: str,
    device: object,
) -> Round74ProbabilityCalibration:
    combined = _concatenate_outputs(outputs)
    selected_batches = tuple(batches)

    def concatenate(name: str) -> torch.Tensor:
        return _to_device_array(
            np.concatenate(
                tuple(np.asarray(getattr(batch, name)) for batch in selected_batches),
                axis=0,
            ),
            device,
        )

    payoff = concatenate("net_payoff_bps")
    return fit_round74_probability_calibration(
        positive_payoff_logits=combined.positive_payoff_logits,
        positive_payoff_labels=(payoff > 0.0).to(dtype=torch.float32),
        adverse_selection_logits=combined.adverse_selection_logits,
        adverse_selection_labels=concatenate("adverse_selection"),
        action_eligibility=concatenate("action_eligibility"),
        regime_unpredictability_logits=(combined.regime_unpredictability_logits),
        regime_unpredictability_labels=concatenate("regime_unpredictability"),
        regime_eligibility=concatenate("regime_unpredictability_eligibility"),
        row_run_ids=tuple(
            run_id for batch in selected_batches for run_id in batch.run_id
        ),
        tuning_subpartition=roles.subpartition,
        pretest_policy_sha256=str(policy["policy_sha256"]),
        calibration_source_sha256=_calibration_source_sha256(
            policy=policy,
            roles=roles,
        ),
        backend_kind=backend_kind,
        backend_device=backend_device,
    )


def calibrate_and_select_round74_development_policy(
    tuning_roles: Round74PreparedTuningRoles,
    *,
    pretest_policy_path: str | Path,
    feature_scaler: Round74EventFeatureScaler,
    execution_store: object,
    execution_partition: Round74EventRunPartition,
    execution_target_assembly_by_run_id: Mapping[
        str,
        Round74SourceTargetAssembly,
    ],
    compute_backend: str = "auto",
    minibatch_rows: int = 128,
) -> Round74DevelopmentPolicyBundle:
    """Calibrate and select all profiles without loading sealed test data."""

    tuning_roles.validate()
    execution_partition.validate()
    expected_execution_runs = tuning_roles.subpartition.policy_selection_run_ids
    execution_assemblies = dict(execution_target_assembly_by_run_id)
    if (
        not isinstance(feature_scaler, Round74EventFeatureScaler)
        or {
            batch.scaler_sha256
            for batch in (
                *tuning_roles.model_selection_batches,
                *tuning_roles.calibration_batches,
                *tuning_roles.policy_selection_batches,
            )
        }
        != {feature_scaler.scaler_sha256}
        or execution_partition.partition_sha256
        != tuning_roles.subpartition.parent_partition_sha256
        or set(execution_assemblies) != set(expected_execution_runs)
        or any(
            not isinstance(
                execution_assemblies[run_id],
                Round74SourceTargetAssembly,
            )
            for run_id in expected_execution_runs
        )
    ):
        raise ValueError("Round 74 development execution replay input differs")
    model, policy = load_round74_pretest_policy(pretest_policy_path)
    development = policy.get("development_data")
    artifact = policy.get("model_artifact")
    authority = policy.get("authority")
    if not all(
        isinstance(value, Mapping) for value in (development, artifact, authority)
    ):
        raise ValueError("Round 74 development pretest policy differs")
    assert isinstance(development, Mapping)
    assert isinstance(artifact, Mapping)
    assert isinstance(authority, Mapping)
    training_batch_sha256 = development.get("training_batch_sha256")
    tuning_batch_sha256 = development.get("tuning_batch_sha256")
    if (
        not isinstance(training_batch_sha256, list)
        or len(training_batch_sha256) != 120
        or not isinstance(tuning_batch_sha256, list)
        or tuning_batch_sha256
        != [batch.batch_sha256 for batch in tuning_roles.model_selection_batches]
        or {
            batch.window_representation
            for batch in (
                *tuning_roles.model_selection_batches,
                *tuning_roles.calibration_batches,
                *tuning_roles.policy_selection_batches,
            )
        }
        != {development.get("window_representation")}
        or development.get("representative_window_policy_applied") is not True
        or development.get("test_batches_consumed") != 0
        or authority.get("sealed_test_evaluated") is not False
    ):
        raise ValueError("Round 74 development tuning role binding differs")
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    model = model.to(device)
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    warning_messages: list[str] = []
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibration_outputs = tuple(
                _infer_batch(
                    model,
                    batch,
                    minibatch_rows=minibatch_rows,
                    device=device,
                )
                for batch in tuning_roles.calibration_batches
            )
            calibration = _fit_calibration(
                calibration_outputs,
                tuning_roles.calibration_batches,
                roles=tuning_roles,
                policy=policy,
                backend_kind=backend.kind,
                backend_device=str(device),
                device=device,
            )
            policy_outputs = tuple(
                _infer_batch(
                    model,
                    batch,
                    minibatch_rows=minibatch_rows,
                    device=device,
                )
                for batch in tuning_roles.policy_selection_batches
            )
            contexts = tuple(
                build_round74_action_inference_context(batch)
                for batch in tuning_roles.policy_selection_batches
            )
            candidates_by_profile = {
                profile: tuple(
                    derive_round74_action_candidates(
                        output,
                        context,
                        calibration,
                        pretest_policy_sha256=str(policy["policy_sha256"]),
                        profile=profile,
                    )
                    for output, context in zip(
                        policy_outputs,
                        contexts,
                        strict=True,
                    )
                )
                for profile in ROUND74_ACTION_PROFILES
            }
            decision_latency = benchmark_round74_online_decision_latency(
                model,
                scaler=feature_scaler,
                calibration=calibration,
                contexts=contexts,
                pretest_policy_sha256=str(policy["policy_sha256"]),
                pretest_model_sha256=str(artifact["sha256"]),
                backend=backend,
                device=device,
                torch_directml_version=_package_version("torch-directml"),
            )
            warning_messages.extend(str(item.message) for item in caught)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        model.to("cpu")
    del calibration_outputs, policy_outputs
    fallback = tuple(
        message
        for message in warning_messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    )
    if fallback:
        raise RuntimeError(f"Round 74 development used CPU fallback: {fallback}")
    execution_panels = build_round74_delayed_execution_panels(
        execution_store,
        partition=execution_partition,
        policy_selection_batches=tuning_roles.policy_selection_batches,
        target_assembly_by_run_id=execution_assemblies,
        latency_evidence=decision_latency,
    )
    if tuple(panel.profile for panel in execution_panels) != ROUND74_ACTION_PROFILES:
        raise ValueError("Round 74 development execution profile panel differs")
    action_policies = tuple(
        select_round74_action_policy_batches(
            tuning_roles.policy_selection_batches,
            candidates_by_profile[profile],
            tuning_roles.subpartition,
            execution_panel=panel,
        )
        for profile, panel in zip(
            ROUND74_ACTION_PROFILES,
            execution_panels,
            strict=True,
        )
    )
    source_sha256 = _current_source_sha256()
    result = Round74DevelopmentPolicyBundle(
        pretest_policy_sha256=str(policy["policy_sha256"]),
        pretest_model_sha256=str(artifact["sha256"]),
        feature_scaler_sha256=feature_scaler.scaler_sha256,
        tuning_subpartition_sha256=tuning_roles.subpartition.subpartition_sha256,
        model_selection_batch_sha256=tuple(
            batch.batch_sha256 for batch in tuning_roles.model_selection_batches
        ),
        calibration_batch_sha256=tuple(
            batch.batch_sha256 for batch in tuning_roles.calibration_batches
        ),
        policy_selection_batch_sha256=tuple(
            batch.batch_sha256 for batch in tuning_roles.policy_selection_batches
        ),
        probability_calibration=calibration,
        online_decision_latency=decision_latency,
        execution_outcome_panel_sha256=tuple(
            panel.panel_sha256 for panel in execution_panels
        ),
        execution_outcome_panel_rows=tuple(
            len(panel.rows) for panel in execution_panels
        ),
        action_policies=action_policies,
        backend_kind=backend.kind,
        backend_device=str(device),
        backend_vendor=backend.vendor,
        warning_count=len(warning_messages) + decision_latency.warning_count,
        **source_sha256,
    )
    result.validate()
    return result


def train_calibrate_and_select_round74_development_policy(
    prepared: Round74PreparedDevelopmentData,
    tuning_roles: Round74PreparedTuningRoles,
    *,
    output_directory: str | Path,
    execution_store: object,
    execution_partition: Round74EventRunPartition,
    execution_target_assembly_by_run_id: Mapping[
        str,
        Round74SourceTargetAssembly,
    ],
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
) -> Round74DevelopmentPolicyArtifact:
    """Run the complete development path and write one immutable bundle."""

    prepared.validate()
    tuning_roles.validate()
    execution_partition.validate()
    prepared_tuning_sha256 = tuple(
        batch.batch_sha256 for batch in prepared.tuning_batches
    )
    role_tuning_sha256 = tuple(
        batch.batch_sha256
        for batch in (
            *tuning_roles.model_selection_batches,
            *tuning_roles.calibration_batches,
            *tuning_roles.policy_selection_batches,
        )
    )
    if (
        prepared_tuning_sha256 != role_tuning_sha256
        or prepared.training_batches[0].partition_sha256
        != tuning_roles.subpartition.parent_partition_sha256
        or execution_partition.partition_sha256
        != tuning_roles.subpartition.parent_partition_sha256
    ):
        raise ValueError("Round 74 development prepared-role binding differs")
    output = Path(output_directory)
    pretest = train_and_seal_round74_pretest_policy_from_prepared_roles(
        prepared.training_batches,
        tuning_roles,
        output_directory=output,
        compute_backend=compute_backend,
        config=config,
        feature_scaler=prepared.scaler,
    )
    bundle = calibrate_and_select_round74_development_policy(
        tuning_roles,
        pretest_policy_path=pretest.policy_path,
        feature_scaler=prepared.scaler,
        execution_store=execution_store,
        execution_partition=execution_partition,
        execution_target_assembly_by_run_id=(
            execution_target_assembly_by_run_id
        ),
        compute_backend=compute_backend,
        minibatch_rows=inference_minibatch_rows,
    )
    payload = _canonical_bytes(bundle.as_dict()) + b"\n"
    path = output / f"round74-development-policy-{bundle.bundle_sha256}.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Round 74 immutable development bundle differs")
    else:
        write_bytes_atomic(path, payload)
    persisted = load_round74_development_policy_bundle(path)
    if persisted.as_dict() != bundle.as_dict():
        raise RuntimeError("Round 74 persisted development bundle differs")
    return Round74DevelopmentPolicyArtifact(
        bundle_sha256=bundle.bundle_sha256,
        bundle_path=path,
        pretest_policy=pretest,
        bundle=bundle,
    )


def train_round74_development_policy_from_inputs(
    store: object,
    inputs: Round74DevelopmentInputs,
    *,
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
    window_representation: str = "per_symbol",
) -> Round74DevelopmentPolicyArtifact:
    """Prepare, train, calibrate, and select from a sealed-input-safe panel."""

    inputs.validate()
    prepared = prepare_round74_development_data(
        store,
        partition=inputs.partition,
        target_assembly_by_run_id=inputs.target_assembly_by_run_id(),
        window_representation=window_representation,
    )
    subpartition = build_round74_tuning_subpartition(inputs.partition)
    roles = split_round74_prepared_tuning_roles(
        prepared,
        subpartition=subpartition,
    )
    assemblies = inputs.target_assembly_by_run_id()
    return train_calibrate_and_select_round74_development_policy(
        prepared,
        roles,
        output_directory=output_directory,
        execution_store=store,
        execution_partition=inputs.partition,
        execution_target_assembly_by_run_id={
            run_id: assemblies[run_id]
            for run_id in roles.subpartition.policy_selection_run_ids
        },
        compute_backend=compute_backend,
        config=config,
        inference_minibatch_rows=inference_minibatch_rows,
    )


def load_round74_development_policy_bundle(
    path: str | Path,
) -> Round74DevelopmentPolicyBundle:
    """Load one canonical, source-bound development bundle and fail closed."""

    selected_path = Path(path)
    try:
        raw = selected_path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 74 development bundle JSON differs") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 74 development bundle root differs")
    bundle = Round74DevelopmentPolicyBundle.from_dict(value)
    expected_name = f"round74-development-policy-{bundle.bundle_sha256}.json"
    if selected_path.name != expected_name:
        raise ValueError("Round 74 development bundle filename differs")
    if raw != _canonical_bytes(bundle.as_dict()) + b"\n":
        raise ValueError("Round 74 development bundle encoding differs")
    return bundle


__all__ = [
    "ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION",
    "ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION",
    "Round74DevelopmentPolicyArtifact",
    "Round74DevelopmentPolicyBundle",
    "calibrate_and_select_round74_development_policy",
    "load_round74_development_policy_bundle",
    "train_calibrate_and_select_round74_development_policy",
    "train_round74_development_policy_from_inputs",
]
