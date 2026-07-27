from __future__ import annotations

from dataclasses import dataclass
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
        payoff[row] = 2.0 if row % 2 else -2.0
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

    def validate(self) -> None:
        return None


class _Model:
    def to(self, _device: object) -> _Model:
        return self


class _Calibration:
    pretest_policy_sha256 = _digest(700)
    tuning_subpartition_sha256 = _Subpartition.subpartition_sha256
    calibration_sha256 = _digest(702)

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
        }


class _Policy:
    def __init__(self, profile: str, target_sha256: tuple[str, ...]) -> None:
        self.profile = profile
        self.pretest_policy_sha256 = _Calibration.pretest_policy_sha256
        self.probability_calibration_sha256 = _Calibration.calibration_sha256
        self.tuning_subpartition_sha256 = _Subpartition.subpartition_sha256
        self.target_batch_sha256 = target_sha256

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "target_batch_sha256": list(self.target_batch_sha256),
        }


def _policy(roles: _Roles, *, representative: bool = True) -> dict[str, object]:
    return {
        "policy_sha256": _Calibration.pretest_policy_sha256,
        "development_data": {
            "training_batch_sha256": [_digest(index) for index in range(100, 220)],
            "tuning_batch_sha256": [
                batch.batch_sha256 for batch in roles.model_selection_batches
            ],
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

    def select(
        batches: tuple[_Batch, ...],
        candidates: tuple[object, ...],
        _subpartition: object,
    ) -> _Policy:
        assert len(batches) == len(candidates) == 6
        profile = str(candidates[0].profile)
        assert all(candidate.profile == profile for candidate in candidates)
        return _Policy(
            profile,
            tuple(batch.batch_sha256 for batch in batches),
        )

    monkeypatch.setattr(subject, "select_round74_action_policy_batches", select)

    result = subject.calibrate_and_select_round74_development_policy(
        roles,  # type: ignore[arg-type]
        pretest_policy_path="unused.json",
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

    with pytest.raises(ValueError, match="tuning role binding differs"):
        subject.calibrate_and_select_round74_development_policy(
            roles,  # type: ignore[arg-type]
            pretest_policy_path="unused.json",
            compute_backend="cpu",
        )


def test_round74_development_coordinator_runs_real_calibration_and_policy_selection(
    monkeypatch: pytest.MonkeyPatch,
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

    result = subject.calibrate_and_select_round74_development_policy(
        roles,  # type: ignore[arg-type]
        pretest_policy_path="unused.json",
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
