from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    build_round74_action_inference_context,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    Round74TemperatureFit,
    Round74TuningSubpartition,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_model import (
    Round74EventModelOutput,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
    Round74EventFeatureScaler,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
import simple_ai_trading.round74_online_decision_latency as subject
from simple_ai_trading.round74_online_decision_latency import (
    ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE,
    Round74OnlineDecisionLatencyEvidence,
    Round74ProfileDecisionLatency,
)


SOURCE = Path(__file__).parents[1] / "src" / "simple_ai_trading"


def _sha(filename: str) -> str:
    payload = (SOURCE / filename).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _profile(name: str, offset: int) -> Round74ProfileDecisionLatency:
    samples = tuple(
        offset + index
        for index in range(
            1,
            ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE + 1,
        )
    )
    return Round74ProfileDecisionLatency.from_samples(name, samples)


def _evidence() -> Round74OnlineDecisionLatencyEvidence:
    result = Round74OnlineDecisionLatencyEvidence(
        pretest_policy_sha256="1" * 64,
        pretest_model_sha256="2" * 64,
        scaler_sha256="3" * 64,
        probability_calibration_sha256="4" * 64,
        tuning_subpartition_sha256="5" * 64,
        source_context_sha256=("6" * 64, "7" * 64),
        source_feature_row_sha256=tuple(
            f"{index + 1000:064x}"
            for index in range(
                ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE
            )
        ),
        backend_requested="auto",
        backend_kind="cpu",
        backend_device="cpu",
        backend_vendor="host",
        torch_version="test",
        torch_directml_version="not-installed",
        warning_count=0,
        profiles=tuple(
            _profile(profile, index * 1_000)
            for index, profile in enumerate(ROUND74_ACTION_PROFILES)
        ),
        decision_latency_module_sha256=_sha(
            "round74_online_decision_latency.py"
        ),
        action_policy_module_sha256=_sha(
            "impact_absorption_event_action_policy.py"
        ),
        scaler_module_sha256=_sha("impact_absorption_event_scaling.py"),
    )
    result.validate()
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _scaler() -> Round74EventFeatureScaler:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    lower = np.full(count, -10.0, dtype=np.float64)
    upper = np.full(count, 10.0, dtype=np.float64)
    lower[:ROUND74_EVENT_BINARY_FEATURE_COUNT] = 0.0
    upper[:ROUND74_EVENT_BINARY_FEATURE_COUNT] = 1.0
    return Round74EventFeatureScaler(
        median=np.zeros(count, dtype=np.float64),
        scale=np.ones(count, dtype=np.float64),
        lower_clip=lower,
        upper_clip=upper,
        constant_mask=np.zeros(count, dtype=np.bool_),
        fit_input_rows=10,
        fit_sample_rows=10,
        fit_sample_index_sha256="8" * 64,
    )


def _context(scaler: Round74EventFeatureScaler):
    action_shape = (
        1,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (1, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS))
    features = np.zeros(
        (
            1,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    features[:, :, 0] = 1.0
    batch = Round74EventTrainingBatch(
        role="tuning",
        partition_sha256="9" * 64,
        scaler_sha256=scaler.scaler_sha256,
        run_id=("a" * 32,),
        symbol=("BTCUSDT",),
        decision_monotonic_ns=_readonly(
            np.asarray([1_000_000_000], dtype=np.int64)
        ),
        decision_wall_ns=_readonly(
            np.asarray([1_800_000_000_000_000_000], dtype=np.int64)
        ),
        endpoint_frame_index=_readonly(np.asarray([1], dtype=np.int64)),
        endpoint_message_index=_readonly(np.asarray([1], dtype=np.int64)),
        anchor_index=_readonly(np.asarray([0], dtype=np.int64)),
        sample_sha256=("a" * 64,),
        feature_window_sha256=("b" * 64,),
        target_context_sha256=("c" * 64,),
        test_access_sha256=("",),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(
            np.full(action_shape, 1_100_000_000, dtype=np.int64)
        ),
        actual_exit_monotonic_ns=_readonly(
            np.full(action_shape, 31_100_000_000, dtype=np.int64)
        ),
        net_payoff_bps=_readonly(np.ones(action_shape, dtype=np.float32)),
        maximum_adverse_excursion_bps=_readonly(
            np.ones(action_shape, dtype=np.float32)
        ),
        adverse_selection=_readonly(np.zeros(action_shape, dtype=np.float32)),
        regime_unpredictability=_readonly(
            np.zeros(regime_shape, dtype=np.float32)
        ),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float32)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    batch.validate()
    return build_round74_action_inference_context(batch)


def _calibration() -> Round74ProbabilityCalibration:
    run_ids = tuple(f"{index:032x}" for index in range(1, 25))
    subpartition = Round74TuningSubpartition(
        parent_partition_sha256="9" * 64,
        model_selection_run_ids=run_ids[:12],
        calibration_run_ids=run_ids[12:18],
        policy_selection_run_ids=run_ids[18:],
    )
    fit = Round74TemperatureFit(
        temperature=1.0,
        eligible_observations=12,
        positive_observations=6,
        calibration_runs=6,
        minimum_run_observations=2,
        maximum_run_observations=2,
        uncalibrated_run_balanced_nll=0.5,
        calibrated_run_balanced_nll=0.5,
        uncalibrated_nll=0.5,
        calibrated_nll=0.5,
        uncalibrated_brier=0.2,
        calibrated_brier=0.2,
        uncalibrated_ece=0.1,
        calibrated_ece=0.1,
    )
    result = Round74ProbabilityCalibration(
        pretest_policy_sha256="1" * 64,
        tuning_subpartition_sha256=subpartition.subpartition_sha256,
        calibration_source_sha256="d" * 64,
        calibration_data_sha256="e" * 64,
        calibration_run_ids=subpartition.calibration_run_ids,
        calibration_row_run_ids_sha256="f" * 64,
        positive_payoff=fit,
        adverse_selection=fit,
        regime_unpredictability=fit,
        backend_kind="cpu",
        backend_device="cpu",
    )
    result.validate()
    return result


class _NoActionModel(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        rows = int(values.shape[0])
        action_shape = (
            rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        quantile_shape = (*action_shape, len(ROUND74_EVENT_PAYOFF_QUANTILES))
        return Round74EventModelOutput(
            payoff_quantiles_bps=torch.zeros(
                quantile_shape,
                dtype=torch.float32,
                device=values.device,
            ),
            maximum_adverse_excursion_quantiles_bps=torch.ones(
                quantile_shape,
                dtype=torch.float32,
                device=values.device,
            ),
            positive_payoff_logits=torch.full(
                action_shape,
                math.log(0.25),
                dtype=torch.float32,
                device=values.device,
            ),
            adverse_selection_logits=torch.full(
                action_shape,
                math.log(4.0),
                dtype=torch.float32,
                device=values.device,
            ),
            regime_unpredictability_logits=torch.full(
                (rows, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)),
                math.log(4.0),
                dtype=torch.float32,
                device=values.device,
            ),
        )


def test_round74_decision_latency_uses_recomputable_distribution_free_tail() -> None:
    profile = _profile(ROUND74_ACTION_PROFILES[0], 0)

    assert profile.p50_ns == 150
    assert profile.p95_ns == 285
    assert profile.p99_upper_confidence_order_index == 299
    assert profile.p99_upper_confidence_ns == 300
    assert profile.maximum_ns == 300
    assert Round74ProfileDecisionLatency.from_dict(profile.as_dict()) == profile

    changed = profile.as_dict()
    changed["p99_upper_confidence_ns"] = 299
    with pytest.raises(ValueError, match="summary differs"):
        Round74ProfileDecisionLatency.from_dict(changed)


def test_round74_decision_latency_evidence_round_trips_and_rejects_drift() -> None:
    evidence = _evidence()
    payload = evidence.as_dict()

    assert (
        Round74OnlineDecisionLatencyEvidence.from_dict(payload).as_dict()
        == payload
    )
    assert payload["authority"] == {
        "component_latency_measured": True,
        "end_to_end_tick_to_trade_latency_measured": False,
        "mainnet_execution_equivalence_claim": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
    }

    changed = deepcopy(payload)
    changed["measurement_contract"]["excluded_required_components"] = []
    with pytest.raises(ValueError, match="digest differs"):
        Round74OnlineDecisionLatencyEvidence.from_dict(changed)


def test_round74_decision_latency_rejects_short_or_boolean_samples() -> None:
    with pytest.raises(ValueError, match="samples differ"):
        Round74ProfileDecisionLatency.from_samples(
            ROUND74_ACTION_PROFILES[0],
            [1] * (ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE - 1),
        )
    payload = _profile(ROUND74_ACTION_PROFILES[0], 0).as_dict()
    payload["latency_ns"][0] = True
    with pytest.raises(ValueError, match="types differ"):
        Round74ProfileDecisionLatency.from_dict(payload)


def test_round74_decision_latency_benchmarks_real_target_free_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "ROUND74_ONLINE_DECISION_LATENCY_MEASUREMENTS_PER_PROFILE",
        3,
    )
    monkeypatch.setattr(
        subject,
        "ROUND74_ONLINE_DECISION_LATENCY_WARMUPS_PER_PROFILE",
        1,
    )
    monkeypatch.setattr(subject, "ROUND74_ONLINE_DECISION_LATENCY_QUANTILE", 0.5)
    monkeypatch.setattr(subject, "ROUND74_ONLINE_DECISION_LATENCY_CONFIDENCE", 0.75)
    scaler = _scaler()

    evidence = subject.benchmark_round74_online_decision_latency(
        _NoActionModel(),
        scaler=scaler,
        calibration=_calibration(),
        contexts=(_context(scaler),),
        pretest_policy_sha256="1" * 64,
        pretest_model_sha256="2" * 64,
        backend=BackendInfo(
            requested="cpu",
            kind="cpu",
            device="cpu",
            vendor="host",
            reason="test",
        ),
        device=torch.device("cpu"),
        torch_directml_version="not-installed",
    )

    assert tuple(profile.profile for profile in evidence.profiles) == (
        ROUND74_ACTION_PROFILES
    )
    assert all(len(profile.latency_ns) == 3 for profile in evidence.profiles)
    assert evidence.as_dict()["measurement_contract"][
        "target_fields_or_outcomes_consumed"
    ] is False
