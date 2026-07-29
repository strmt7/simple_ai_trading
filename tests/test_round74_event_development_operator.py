from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simple_ai_trading.round74_event_development_operator as subject
from simple_ai_trading.impact_absorption_event_calibration import (
    Round74TuningSubpartition,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_model import (
    build_round74_event_model,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _real_batch(identity: int, start_wall_ns: int) -> Round74EventTrainingBatch:
    rows = 6
    features = np.zeros(
        (rows, ROUND74_EVENT_SEQUENCE_LENGTH, len(ROUND74_EVENT_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for row in range(rows):
        features[row, :, row % 5] = 1.0
        features[row, :, 5 + row % 3] = 1.0
        features[row, :, 10:] = float(row + 1) / 10.0
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (rows, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS))
    payoff = np.empty(action_shape, dtype=np.float32)
    adverse_selection = np.empty(action_shape, dtype=np.float32)
    regime = np.empty(regime_shape, dtype=np.float32)
    for row in range(rows):
        directional_move = 2.0 if row % 2 else -2.0
        payoff[row, :, 0] = directional_move - 0.25
        payoff[row, :, 1] = -directional_move - 0.25
        adverse_selection[row] = float(row % 2)
        regime[row] = float((row + 1) % 2)
    times = np.arange(rows, dtype=np.int64) * 1_000_000_000
    actual_entry = np.broadcast_to(
        (times + 10_000_000).reshape(rows, 1, 1),
        action_shape,
    ).copy()
    actual_exit = np.broadcast_to(
        (times + 20_000_000).reshape(rows, 1, 1),
        action_shape,
    ).copy()
    batch = Round74EventTrainingBatch(
        role="tuning",
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=tuple(f"{identity:032x}" for _ in range(rows)),
        symbol=tuple(("BTCUSDT", "ETHUSDT", "SOLUSDT")[row % 3] for row in range(rows)),
        decision_monotonic_ns=_readonly(times.copy()),
        decision_wall_ns=_readonly(times + start_wall_ns),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.zeros(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(_digest(identity * 100 + row) for row in range(rows)),
        feature_window_sha256=tuple(
            _digest(identity * 1000 + row) for row in range(rows)
        ),
        target_context_sha256=tuple("3" * 64 for _row in range(rows)),
        test_access_sha256=tuple("" for _row in range(rows)),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(
            np.full(action_shape, 1.0, dtype=np.float32)
        ),
        adverse_selection=_readonly(adverse_selection),
        regime_unpredictability=_readonly(regime),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float32)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    batch.validate()
    return batch


@dataclass(frozen=True)
class _Batch:
    batch_sha256: str
    run_id: tuple[str, ...]
    scaler_sha256: str = "2" * 64
    window_representation: str = "per_symbol"


class _Subpartition:
    parent_partition_sha256 = _digest(900)
    subpartition_sha256 = _digest(901)

    def validate(self) -> None:
        return None


class _Roles:
    def __init__(self) -> None:
        self.model_selection_batches = tuple(
            _Batch(_digest(index), (f"{index:032x}",)) for index in range(1, 13)
        )
        self.calibration_batches = tuple(
            _Batch(_digest(index), (f"{index:032x}",)) for index in range(13, 19)
        )
        self.policy_selection_batches = tuple(
            _Batch(_digest(index), (f"{index:032x}",)) for index in range(19, 25)
        )
        self.subpartition = _Subpartition()
        self.subpartition.policy_selection_run_ids = tuple(
            batch.run_id[0] for batch in self.policy_selection_batches
        )

    def validate(self) -> None:
        return None


class _Model:
    def to(self, _device: object) -> _Model:
        return self


class _Calibration:
    pretest_policy_sha256 = _digest(700)
    tuning_subpartition_sha256 = _Subpartition.subpartition_sha256
    calibration_sha256 = _digest(702)
    optimization_population = "capture_run"
    risk_quantiles = object()

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
        }


class _Policy:
    def __init__(
        self,
        profile: str,
        target_sha256: tuple[str, ...],
        execution_outcome_panel_sha256: str,
    ) -> None:
        self.profile = profile
        self.pretest_policy_sha256 = _Calibration.pretest_policy_sha256
        self.probability_calibration_sha256 = _Calibration.calibration_sha256
        self.tuning_subpartition_sha256 = _Subpartition.subpartition_sha256
        self.target_batch_sha256 = target_sha256
        self.execution_outcome_panel_sha256 = execution_outcome_panel_sha256
        self.optimization_population = "capture_run"

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "target_batch_sha256": list(self.target_batch_sha256),
            "execution_outcome_panel_sha256": (self.execution_outcome_panel_sha256),
        }


class _Scaler:
    scaler_sha256 = "2" * 64


class _Partition:
    def __init__(self, partition_sha256: str) -> None:
        self.partition_sha256 = partition_sha256

    def validate(self) -> None:
        return None


class _Assembly:
    pass


@dataclass(frozen=True)
class _Latency:
    pretest_policy_sha256: str = _Calibration.pretest_policy_sha256
    pretest_model_sha256: str = _digest(701)
    scaler_sha256: str = _Scaler.scaler_sha256
    probability_calibration_sha256: str = _Calibration.calibration_sha256
    tuning_subpartition_sha256: str = _Subpartition.subpartition_sha256
    backend_kind: str = "cpu"
    backend_device: str = "cpu"
    backend_vendor: str = "portable CPU reference"
    warning_count: int = 0

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "pretest_model_sha256": self.pretest_model_sha256,
            "scaler_sha256": self.scaler_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "backend_vendor": self.backend_vendor,
            "warning_count": self.warning_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> _Latency:
        if not isinstance(value, dict):
            raise ValueError("latency differs")
        return cls(**value)


@dataclass(frozen=True)
class _ExecutionPanel:
    profile: str
    panel_sha256: str
    rows: tuple[int, ...] = (1, 2, 3)


def _mock_execution_boundary(
    monkeypatch: pytest.MonkeyPatch,
    roles: object,
    *,
    parent_partition_sha256: str,
) -> tuple[_Scaler, _Partition, dict[str, _Assembly]]:
    monkeypatch.setattr(subject, "Round74EventFeatureScaler", _Scaler)
    monkeypatch.setattr(subject, "Round74SourceTargetAssembly", _Assembly)
    monkeypatch.setattr(subject, "Round74OnlineDecisionLatencyEvidence", _Latency)

    def benchmark(
        _model: object,
        *,
        scaler: object,
        calibration: object,
        pretest_policy_sha256: str,
        pretest_model_sha256: str,
        backend: object,
        device: object,
        **_kwargs: object,
    ) -> _Latency:
        return _Latency(
            pretest_policy_sha256=pretest_policy_sha256,
            pretest_model_sha256=pretest_model_sha256,
            scaler_sha256=str(scaler.scaler_sha256),
            probability_calibration_sha256=str(calibration.calibration_sha256),
            tuning_subpartition_sha256=(roles.subpartition.subpartition_sha256),
            backend_kind=str(backend.kind),
            backend_device=str(device),
            backend_vendor=str(backend.vendor),
        )

    monkeypatch.setattr(
        subject,
        "benchmark_round74_online_decision_latency",
        benchmark,
    )
    panels = tuple(
        _ExecutionPanel(profile, _digest(800 + index))
        for index, profile in enumerate(subject.ROUND74_ACTION_PROFILES)
    )
    monkeypatch.setattr(
        subject,
        "build_round74_delayed_execution_panels",
        lambda *_args, **_kwargs: panels,
    )
    partition = _Partition(parent_partition_sha256)
    assemblies = {
        run_id: _Assembly() for run_id in roles.subpartition.policy_selection_run_ids
    }
    return _Scaler(), partition, assemblies


def _policy(roles: _Roles, *, representative: bool = True) -> dict[str, object]:
    return {
        "policy_sha256": _Calibration.pretest_policy_sha256,
        "development_data": {
            "training_batch_sha256": [_digest(index) for index in range(100, 220)],
            "tuning_batch_sha256": [
                batch.batch_sha256 for batch in roles.model_selection_batches
            ],
            "window_representation": "per_symbol",
            "representative_window_policy_applied": representative,
            "test_batches_consumed": 0,
        },
        "model_artifact": {"sha256": _digest(701)},
        "authority": {"sealed_test_evaluated": False},
    }


def test_round74_development_coordinator_reuses_policy_outputs_for_all_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = _Roles()
    inference_calls: list[_Batch] = []
    derivations: list[tuple[object, str]] = []

    monkeypatch.setattr(
        subject,
        "load_round74_pretest_policy",
        lambda _path: (_Model(), _policy(roles)),
    )

    def infer(
        _model: object,
        batch: _Batch,
        *,
        minibatch_rows: int,
        device: object,
    ) -> object:
        assert minibatch_rows == 128
        assert str(device) == "cpu"
        inference_calls.append(batch)
        return SimpleNamespace(batch_sha256=batch.batch_sha256)

    monkeypatch.setattr(subject, "_infer_batch", infer)
    monkeypatch.setattr(
        subject,
        "_fit_calibration",
        lambda outputs, batches, **_kwargs: (
            _Calibration()
            if len(outputs) == len(batches) == 6
            else pytest.fail("calibration panel differs")
        ),
    )
    monkeypatch.setattr(
        subject,
        "build_round74_action_inference_context",
        lambda batch: SimpleNamespace(batch_sha256=batch.batch_sha256),
    )

    def derive(
        output: object,
        _context: object,
        _calibration: object,
        *,
        pretest_policy_sha256: str,
        profile: str,
    ) -> object:
        assert pretest_policy_sha256 == _Calibration.pretest_policy_sha256
        derivations.append((output, profile))
        return SimpleNamespace(profile=profile)

    monkeypatch.setattr(subject, "derive_round74_action_candidates", derive)
    scaler, partition, assemblies = _mock_execution_boundary(
        monkeypatch,
        roles,
        parent_partition_sha256=roles.subpartition.parent_partition_sha256,
    )

    def select(
        batches: tuple[_Batch, ...],
        candidates: tuple[object, ...],
        _subpartition: object,
        *,
        execution_panel: _ExecutionPanel,
        optimization_population: str,
    ) -> _Policy:
        assert len(batches) == len(candidates) == 6
        assert optimization_population == "capture_run"
        profile = str(candidates[0].profile)
        assert all(candidate.profile == profile for candidate in candidates)
        return _Policy(
            profile,
            tuple(batch.batch_sha256 for batch in batches),
            execution_panel.panel_sha256,
        )

    monkeypatch.setattr(subject, "select_round74_action_policy_batches", select)

    result = subject.calibrate_and_select_round74_development_policy(
        roles,  # type: ignore[arg-type]
        pretest_policy_path="unused.json",
        feature_scaler=scaler,  # type: ignore[arg-type]
        execution_store=object(),
        execution_partition=partition,  # type: ignore[arg-type]
        execution_target_assembly_by_run_id=assemblies,  # type: ignore[arg-type]
        compute_backend="cpu",
    )

    assert inference_calls == [
        *roles.calibration_batches,
        *roles.policy_selection_batches,
    ]
    assert len(derivations) == 18
    assert tuple(policy.profile for policy in result.action_policies) == (
        subject.ROUND74_ACTION_PROFILES
    )
    for index in range(6):
        reused = tuple(derivations[profile * 6 + index][0] for profile in range(3))
        assert reused[0] is reused[1] is reused[2]
    assert result.as_dict()["authority"] == {
        "representative_market_training_completed": True,
        "sealed_test_accessed": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
    }


def test_round74_development_coordinator_rejects_unrepresentative_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = _Roles()
    monkeypatch.setattr(
        subject,
        "load_round74_pretest_policy",
        lambda _path: (_Model(), _policy(roles, representative=False)),
    )
    scaler, partition, assemblies = _mock_execution_boundary(
        monkeypatch,
        roles,
        parent_partition_sha256=roles.subpartition.parent_partition_sha256,
    )

    with pytest.raises(ValueError, match="tuning role binding differs"):
        subject.calibrate_and_select_round74_development_policy(
            roles,  # type: ignore[arg-type]
            pretest_policy_path="unused.json",
            feature_scaler=scaler,  # type: ignore[arg-type]
            execution_store=object(),
            execution_partition=partition,  # type: ignore[arg-type]
            execution_target_assembly_by_run_id=assemblies,  # type: ignore[arg-type]
            compute_backend="cpu",
        )


def test_round74_development_coordinator_runs_real_calibration_and_policy_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_wall_ns = 1_800_000_000_000_000_000
    batches = tuple(
        _real_batch(index, base_wall_ns + index * 10_000_000_000)
        for index in range(1, 25)
    )
    subpartition = Round74TuningSubpartition(
        parent_partition_sha256="1" * 64,
        model_selection_run_ids=tuple(f"{index:032x}" for index in range(1, 13)),
        calibration_run_ids=tuple(f"{index:032x}" for index in range(13, 19)),
        policy_selection_run_ids=tuple(f"{index:032x}" for index in range(19, 25)),
    )
    subpartition.validate()
    roles = SimpleNamespace(
        model_selection_batches=batches[:12],
        calibration_batches=batches[12:18],
        policy_selection_batches=batches[18:],
        subpartition=subpartition,
        validate=lambda: None,
    )
    policy = {
        "policy_sha256": _Calibration.pretest_policy_sha256,
        "development_data": {
            "training_batch_sha256": [_digest(index) for index in range(100, 220)],
            "tuning_batch_sha256": [batch.batch_sha256 for batch in batches[:12]],
            "window_representation": "per_symbol",
            "representative_window_policy_applied": True,
            "test_batches_consumed": 0,
        },
        "model_artifact": {"sha256": _digest(701)},
        "authority": {"sealed_test_evaluated": False},
    }
    model = build_round74_event_model("event_pooling_linear")
    monkeypatch.setattr(
        subject,
        "load_round74_pretest_policy",
        lambda _path: (model, policy),
    )
    scaler, partition, assemblies = _mock_execution_boundary(
        monkeypatch,
        roles,
        parent_partition_sha256=subpartition.parent_partition_sha256,
    )
    baseline_select = subject.select_round74_action_policy_batches

    def delayed_select(*args, execution_panel, **kwargs):
        baseline = baseline_select(*args, **kwargs)
        return replace(
            baseline,
            execution_outcome_panel_sha256=execution_panel.panel_sha256,
        )

    monkeypatch.setattr(
        subject,
        "select_round74_action_policy_batches",
        delayed_select,
    )

    result = subject.calibrate_and_select_round74_development_policy(
        roles,  # type: ignore[arg-type]
        pretest_policy_path="unused.json",
        feature_scaler=scaler,  # type: ignore[arg-type]
        execution_store=object(),
        execution_partition=partition,  # type: ignore[arg-type]
        execution_target_assembly_by_run_id=assemblies,  # type: ignore[arg-type]
        compute_backend="cpu",
        minibatch_rows=3,
    )

    assert result.probability_calibration.calibration_run_ids == (
        subpartition.calibration_run_ids
    )
    assert tuple(policy.profile for policy in result.action_policies) == (
        "conservative",
        "regular",
        "aggressive",
    )
    assert all(
        policy.sealed_test_accessed is False for policy in result.action_policies
    )
    assert all(policy.profitability_claim is False for policy in result.action_policies)
    assert len(result.bundle_sha256) == 64

    output = tmp_path / (f"round74-development-policy-{result.bundle_sha256}.json")
    output.write_bytes(subject._canonical_bytes(result.as_dict()) + b"\n")
    restored = subject.load_round74_development_policy_bundle(output)
    assert restored == result
    assert restored.probability_calibration == result.probability_calibration
    assert restored.action_policies == result.action_policies

    duplicate = tmp_path / output.name
    duplicate.write_bytes(
        output.read_bytes().replace(
            b'{"action_policies":',
            b'{"action_policies":[],"action_policies":',
            1,
        )
    )
    with pytest.raises(ValueError, match="bundle JSON differs"):
        subject.load_round74_development_policy_bundle(duplicate)

    source_mismatch = result.as_dict()
    source_mismatch["source"]["action_policy_module_sha256"] = "0" * 64
    source_mismatch.pop("bundle_sha256")
    source_mismatch["bundle_sha256"] = subject._canonical_sha256(source_mismatch)
    mismatch_path = tmp_path / (
        f"round74-development-policy-{source_mismatch['bundle_sha256']}.json"
    )
    mismatch_path.write_text(
        json.dumps(
            source_mismatch,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="source identity differs"):
        subject.load_round74_development_policy_bundle(mismatch_path)


def test_round74_development_input_pipeline_prepares_roles_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    inputs = SimpleNamespace(
        partition=object(),
        validate=lambda: calls.append("inputs"),
        target_assembly_by_run_id=lambda: {"run": "assembly"},
    )
    prepared = SimpleNamespace()
    subpartition = SimpleNamespace(policy_selection_run_ids=("run",))
    roles = SimpleNamespace(subpartition=subpartition)
    artifact = SimpleNamespace()
    monkeypatch.setattr(
        subject,
        "prepare_round74_development_data",
        lambda store, **kwargs: calls.append(("prepare", store, kwargs)) or prepared,
    )
    monkeypatch.setattr(
        subject,
        "build_round74_tuning_subpartition",
        lambda partition: calls.append(("subpartition", partition)) or subpartition,
    )
    monkeypatch.setattr(
        subject,
        "split_round74_prepared_tuning_roles",
        lambda selected, **kwargs: calls.append(("roles", selected, kwargs)) or roles,
    )
    monkeypatch.setattr(
        subject,
        "train_calibrate_and_select_round74_development_policy",
        lambda selected, selected_roles, **kwargs: (
            calls.append(("train", selected, selected_roles, kwargs)) or artifact
        ),
    )

    result = subject.train_round74_development_policy_from_inputs(
        "read-only-store",
        inputs,
        output_directory="output",
        compute_backend="cpu",
        inference_minibatch_rows=64,
    )

    assert result is artifact
    assert calls == [
        "inputs",
        (
            "prepare",
            "read-only-store",
            {
                "partition": inputs.partition,
                "target_assembly_by_run_id": {"run": "assembly"},
                "window_representation": "per_symbol",
            },
        ),
        ("subpartition", inputs.partition),
        ("roles", prepared, {"subpartition": subpartition}),
        (
            "train",
            prepared,
            roles,
            {
                "output_directory": "output",
                "execution_store": "read-only-store",
                "execution_partition": inputs.partition,
                "execution_target_assembly_by_run_id": {
                    "run": "assembly",
                },
                "compute_backend": "cpu",
                "config": None,
                "inference_minibatch_rows": 64,
            },
        ),
    ]
