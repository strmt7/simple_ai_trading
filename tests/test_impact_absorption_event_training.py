from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from simple_ai_trading.impact_absorption_event_dataset import (  # noqa: E402
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_sequence import (  # noqa: E402
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
from simple_ai_trading.impact_absorption_event_scaling import (  # noqa: E402
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
    Round74EventFeatureScaler,
)
from simple_ai_trading.impact_absorption_event_training import (  # noqa: E402
    ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT,
    ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS,
    ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION,
    Round74EventEnsemble,
    Round74EventTargetLossScale,
    Round74EventTrainingConfig,
    _complexity_gated_candidate_id,
    _eligible_target_minibatch_schedule,
    _eligible_target_weighted_group_loss,
    _empty_metric_sums,
    _loss_for_minibatch,
    _losses_for_minibatch_group,
    fit_round74_event_target_loss_scale,
    load_round74_pretest_policy,
    load_round74_pretest_scaler,
    train_and_seal_round74_pretest_policy,
    train_and_seal_round74_pretest_policy_from_prepared_roles,
)
from simple_ai_trading.impact_absorption_event_model import (  # noqa: E402
    ROUND74_EVENT_MODEL_CANDIDATES,
    Round74EventModelOutput,
    build_round74_event_model,
)
from simple_ai_trading.round74_event_model_operator import (  # noqa: E402
    split_round74_tuning_batch_roles,
)
from simple_ai_trading.round74_segmented_model_operator import (  # noqa: E402
    Round74SegmentedTuningSubpartition,
    round74_segmented_window_policy,
)


WALL_NS = 1_800_000_000_000_000_000
PURGE_NS = 310_500_000_000


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _batch(
    role: str,
    *,
    start_wall_ns: int,
    identity: int,
    rows: int = 2,
) -> Round74EventTrainingBatch:
    generator = np.random.default_rng(7400 + identity)
    features = generator.normal(
        size=(
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        )
    ).astype(np.float32)
    features[:, :, :8] = 0.0
    for row in range(rows):
        features[row, :, row % 5] = 1.0
        features[row, :, 5 + row % 3] = 1.0
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    payoff = generator.normal(size=action_shape).astype(np.float32)
    adverse_excursion = np.abs(generator.normal(size=action_shape)).astype(np.float32)
    adverse = generator.integers(0, 2, size=action_shape).astype(np.float32)
    unpredictable = generator.integers(
        0,
        2,
        size=regime_shape,
    ).astype(np.float32)
    eligibility = np.ones(action_shape, dtype=np.float32)
    eligibility[0, 0, 0] = 0.0
    actual_entry = np.full(action_shape, 10, dtype=np.int64)
    actual_exit = np.full(action_shape, 20, dtype=np.int64)
    actual_entry[0, 0, 0] = -1
    actual_exit[0, 0, 0] = -1
    payoff[0, 0, 0] = 0.0
    adverse_excursion[0, 0, 0] = 0.0
    adverse[0, 0, 0] = 0.0
    times = np.arange(rows, dtype=np.int64) * 1_000_000_000
    batch = Round74EventTrainingBatch(
        role=role,
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=tuple(f"{identity:032x}" for _ in range(rows)),
        symbol=tuple(("BTCUSDT", "ETHUSDT", "SOLUSDT")[row % 3] for row in range(rows)),
        decision_monotonic_ns=_readonly(times.copy()),
        decision_wall_ns=_readonly(times + start_wall_ns),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.zeros(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{identity * 100 + row:064x}" for row in range(rows)),
        feature_window_sha256=tuple(
            f"{identity * 1000 + row:064x}" for row in range(rows)
        ),
        target_context_sha256=tuple("3" * 64 for _ in range(rows)),
        test_access_sha256=tuple(
            "4" * 64 if role == "test" else "" for _ in range(rows)
        ),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(adverse),
        regime_unpredictability=_readonly(unpredictable),
        action_eligibility=_readonly(eligibility),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    batch.validate()
    return batch


def _config() -> Round74EventTrainingConfig:
    return Round74EventTrainingConfig(
        seeds=(7411,),
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=2,
        minimum_role_rows=2,
        execution_mode="preflight",
    )


def _scaler() -> Round74EventFeatureScaler:
    feature_count = len(ROUND74_EVENT_FEATURE_NAMES)
    lower = np.full(feature_count, -10.0, dtype=np.float64)
    upper = np.full(feature_count, 10.0, dtype=np.float64)
    lower[:ROUND74_EVENT_BINARY_FEATURE_COUNT] = 0.0
    upper[:ROUND74_EVENT_BINARY_FEATURE_COUNT] = 1.0
    return Round74EventFeatureScaler(
        median=np.zeros(feature_count, dtype=np.float64),
        scale=np.ones(feature_count, dtype=np.float64),
        lower_clip=lower,
        upper_clip=upper,
        constant_mask=np.zeros(feature_count, dtype=np.bool_),
        fit_input_rows=10,
        fit_sample_rows=10,
        fit_sample_index_sha256="5" * 64,
    )


def test_target_loss_scale_is_training_only_complete_and_reloadable() -> None:
    training = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=701,
        rows=9,
    )
    selected = fit_round74_event_target_loss_scale(
        (training,),
        require_complete_panel=True,
    )
    reloaded = Round74EventTargetLossScale.from_dict(selected.as_dict())

    assert reloaded.as_dict() == selected.as_dict()
    assert selected.training_batch_sha256 == (training.batch_sha256,)
    assert bool((selected.eligible_target_count > 0).all())
    assert bool((selected.payoff_scale_bps > 0.0).all())
    assert bool((selected.maximum_adverse_excursion_scale_bps > 0.0).all())
    assert selected.as_dict()["tuning_targets_used"] is False
    assert selected.as_dict()["test_targets_used"] is False
    with pytest.raises(ValueError, match="subgroup panel is incomplete"):
        fit_round74_event_target_loss_scale(
            (
                _batch(
                    "training",
                    start_wall_ns=WALL_NS,
                    identity=702,
                    rows=2,
                ),
            ),
            require_complete_panel=True,
        )
    with pytest.raises(ValueError, match="non-training role"):
        fit_round74_event_target_loss_scale(
            (
                _batch(
                    "tuning",
                    start_wall_ns=WALL_NS,
                    identity=703,
                    rows=9,
                ),
            ),
            require_complete_panel=True,
        )


def test_segmented_training_supports_complexity_gate_and_no_run_cycling() -> None:
    Round74EventTrainingConfig(execution_mode="segmented_cohort").validate()
    config = Round74EventTrainingConfig(
        candidate_ids=(ROUND74_EVENT_MODEL_CANDIDATES[0],),
        architecture_selection_mode="fixed",
        execution_mode="segmented_cohort",
    )
    config.validate()

    short = _batch("training", start_wall_ns=WALL_NS, identity=1, rows=2)
    long = _batch(
        "training",
        start_wall_ns=WALL_NS + 10_000_000_000,
        identity=2,
        rows=5,
    )
    totals = _empty_metric_sums()
    per_run = (_empty_metric_sums(), _empty_metric_sums())
    schedule = _eligible_target_minibatch_schedule(
        (short, long),
        2,
        totals=totals,
        per_run_totals=per_run,
    )

    assert sum(row[0] == 0 for row in schedule) == 1
    assert sum(row[0] == 1 for row in schedule) == 3


def test_segmented_complexity_gate_requires_the_complete_dynamic_run_panel() -> None:
    run_count = 45
    losses = {
        candidate_id: tuple(4.0 - candidate_index * 0.1 for _ in range(run_count))
        for candidate_index, candidate_id in enumerate(ROUND74_EVENT_MODEL_CANDIDATES)
    }
    parameter_counts = {
        candidate_id: 100 * (candidate_index + 1)
        for candidate_index, candidate_id in enumerate(ROUND74_EVENT_MODEL_CANDIDATES)
    }

    winner, reports = _complexity_gated_candidate_id(
        ROUND74_EVENT_MODEL_CANDIDATES,
        losses,
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=run_count,
    )
    blocked, incomplete = _complexity_gated_candidate_id(
        ROUND74_EVENT_MODEL_CANDIDATES,
        losses,
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=run_count + 1,
    )

    assert winner == ROUND74_EVENT_MODEL_CANDIDATES[-1]
    assert all(report["complete_tuning_panel"] for report in reports)
    assert all(
        report["required_paired_capture_run_count"] == run_count for report in reports
    )
    assert blocked == ROUND74_EVENT_MODEL_CANDIDATES[0]
    assert all(not report["complete_tuning_panel"] for report in incomplete)


def test_segmented_gradient_is_exactly_eligible_target_weighted() -> None:
    def components(value: float) -> dict[str, torch.Tensor]:
        return {
            name: torch.tensor(value, dtype=torch.float32)
            for name in (
                "payoff_pinball",
                "maximum_adverse_excursion_pinball",
                "positive_bce",
                "adverse_bce",
                "unpredictability_bce",
            )
        }

    grouped = (
        (torch.tensor(0.0), components(1.0), 2, 1),
        (torch.tensor(0.0), components(3.0), 6, 3),
    )
    objective = _eligible_target_weighted_group_loss(
        grouped,
        total_action_weight=8,
        total_regime_weight=4,
    )

    action_coefficient = 1.0 + 0.35 + 0.25 + 0.20
    regime_coefficient = 0.10
    expected = action_coefficient * (1.0 * 2 / 8 + 3.0 * 6 / 8) + regime_coefficient * (
        1.0 * 1 / 4 + 3.0 * 3 / 4
    )
    assert float(objective) == pytest.approx(expected)


def test_segmented_training_seals_and_reloads_pooled_target_policy(
    tmp_path: Path,
) -> None:
    from simple_ai_trading.round74_segmented_model_operator import (
        round74_segmented_window_policy,
    )

    scaler = _scaler()
    training = tuple(
        replace(batch, scaler_sha256=scaler.scaler_sha256)
        for batch in (
            _batch("training", start_wall_ns=WALL_NS, identity=1, rows=2),
            _batch(
                "training",
                start_wall_ns=WALL_NS + 10_000_000_000,
                identity=2,
                rows=5,
            ),
        )
    )
    tuning = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 20_000_000_000,
            identity=3,
            rows=3,
        ),
        scaler_sha256=scaler.scaler_sha256,
    )
    config = Round74EventTrainingConfig(
        candidate_ids=(ROUND74_EVENT_MODEL_CANDIDATES[0],),
        seeds=(7411,),
        maximum_epochs=2,
        early_stopping_patience=1,
        minibatch_rows=2,
        minimum_role_rows=2,
        execution_mode="segmented_cohort",
        architecture_selection_mode="fixed",
    )

    artifact = train_and_seal_round74_pretest_policy(
        training,
        (tuning,),
        output_directory=tmp_path,
        compute_backend="cpu",
        config=config,
        representative_window_policy_sha256=(
            round74_segmented_window_policy()["policy_sha256"]
        ),
        feature_scaler=scaler,
    )
    _model, policy = load_round74_pretest_policy(artifact.policy_path)

    assert policy["optimization_population"]["unit"] == "eligible_target"
    report = next(iter(policy["candidate_panel"].values()))["peer_reports"][0]
    first = report["history"][0]
    assert first["selection_loss_name"] == "loss"
    assert first["optimization_metrics"]["optimizer_steps"] == 1.0
    assert first["optimization_metrics"]["minimum_run_minibatch_contributions"] == 1.0
    assert first["optimization_metrics"]["maximum_run_minibatch_contributions"] == 3.0
    selected_metrics = policy["selection"]["selected_tuning_metrics"]
    assert artifact.tuning_loss == pytest.approx(selected_metrics["loss"])


def test_prepared_roles_forward_segmented_model_selection_without_discarding_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simple_ai_trading.impact_absorption_event_training as training_subject

    model_count, calibration_count, policy_count = 45, 23, 23
    total = model_count + calibration_count + policy_count
    batches = tuple(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + index * 2_000_000_000,
            identity=100 + index,
        )
        for index in range(total)
    )
    model_end = model_count
    calibration_end = model_end + calibration_count
    subpartition = Round74SegmentedTuningSubpartition(
        parent_partition_sha256="1" * 64,
        cohort_plan_sha256="a" * 64,
        model_selection_run_ids=tuple(batch.run_id[0] for batch in batches[:model_end]),
        calibration_run_ids=tuple(
            batch.run_id[0] for batch in batches[model_end:calibration_end]
        ),
        policy_selection_run_ids=tuple(
            batch.run_id[0] for batch in batches[calibration_end:]
        ),
        model_selection_slot_ordinals=tuple(range(514, 559)),
        calibration_slot_ordinals=tuple(range(566, 589)),
        policy_selection_slot_ordinals=tuple(range(592, 615)),
        model_selection_eligible_anchor_ns=(900_000_000_000,) * model_count,
        calibration_eligible_anchor_ns=(900_000_000_000,) * calibration_count,
        policy_selection_eligible_anchor_ns=(900_000_000_000,) * policy_count,
    )
    roles = split_round74_tuning_batch_roles(
        batches,
        subpartition=subpartition,
    )
    roles.validate()
    config = Round74EventTrainingConfig(
        seeds=(7411,),
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=2,
        minimum_role_rows=2,
        execution_mode="segmented_cohort",
    )
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_train(
        training_batches: object,
        tuning_batches: object,
        **kwargs: object,
    ) -> object:
        observed["training_batches"] = training_batches
        observed["tuning_batches"] = tuning_batches
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        training_subject,
        "train_and_seal_round74_pretest_policy",
        fake_train,
    )

    result = train_and_seal_round74_pretest_policy_from_prepared_roles(
        (_batch("training", start_wall_ns=WALL_NS - PURGE_NS, identity=1),),
        roles,
        output_directory=tmp_path,
        compute_backend="cpu",
        config=config,
        feature_scaler=_scaler(),
    )

    assert result is sentinel
    assert observed["tuning_batches"] == roles.model_selection_batches
    assert (
        observed["representative_window_policy_sha256"]
        == (round74_segmented_window_policy()["policy_sha256"])
    )
    assert observed["matched_preparation_sha256"] is None
    assert observed["config"] is config


def test_round74_cohort_mode_rejects_partial_or_unbound_population(
    tmp_path: Path,
) -> None:
    import simple_ai_trading.impact_absorption_event_training as training_subject
    from simple_ai_trading.round74_event_model_operator import (
        round74_matched_representative_window_policy,
        round74_representative_window_policy,
    )

    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )

    with pytest.raises(ValueError, match="exactly 120 training and 12"):
        train_and_seal_round74_pretest_policy(
            [training],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
        )
    with pytest.raises(ValueError, match="cannot claim representative"):
        train_and_seal_round74_pretest_policy(
            [training],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
            representative_window_policy_sha256="a" * 64,
        )
    assert training_subject._cohort_window_policy_identity(
        round74_representative_window_policy()["policy_sha256"],
        None,
    ) == ("single_representation", None)
    assert training_subject._cohort_window_policy_identity(
        round74_matched_representative_window_policy()["policy_sha256"],
        "f" * 64,
    ) == ("matched_representation", "f" * 64)
    with pytest.raises(ValueError, match="matched preparation identity differs"):
        training_subject._cohort_window_policy_identity(
            round74_matched_representative_window_policy()["policy_sha256"],
            None,
        )
    with pytest.raises(ValueError, match="single-representation preparation differs"):
        training_subject._cohort_window_policy_identity(
            round74_representative_window_policy()["policy_sha256"],
            "f" * 64,
        )


def test_round74_pretest_persists_and_tamper_checks_exact_scaler(
    tmp_path: Path,
) -> None:
    scaler = _scaler()
    training = replace(
        _batch("training", start_wall_ns=WALL_NS, identity=1),
        scaler_sha256=scaler.scaler_sha256,
    )
    tuning = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
            identity=2,
        ),
        scaler_sha256=scaler.scaler_sha256,
    )
    training.validate()
    tuning.validate()

    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=_config(),
        feature_scaler=scaler,
    )
    loaded = load_round74_pretest_scaler(artifact.policy_path)
    assert loaded.as_dict() == scaler.as_dict()
    _model, policy = load_round74_pretest_policy(artifact.policy_path)
    scaler_artifact = policy["scaler_artifact"]
    assert scaler_artifact["available"] is True
    assert scaler_artifact["scaler_sha256"] == scaler.scaler_sha256

    scaler_path = tmp_path / str(scaler_artifact["filename"])
    scaler_path.write_bytes(scaler_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="scaler artifact differs"):
        load_round74_pretest_policy(artifact.policy_path)


def _with_fully_censored_prefix(
    batch: Round74EventTrainingBatch,
    rows: int,
) -> Round74EventTrainingBatch:
    payoff = batch.net_payoff_bps.copy()
    adverse_excursion = batch.maximum_adverse_excursion_bps.copy()
    adverse = batch.adverse_selection.copy()
    unpredictable = batch.regime_unpredictability.copy()
    action_eligibility = batch.action_eligibility.copy()
    regime_eligibility = batch.regime_unpredictability_eligibility.copy()
    actual_entry = batch.actual_entry_monotonic_ns.copy()
    actual_exit = batch.actual_exit_monotonic_ns.copy()
    payoff[:rows] = 0.0
    adverse_excursion[:rows] = 0.0
    adverse[:rows] = 0.0
    unpredictable[:rows] = 0.0
    action_eligibility[:rows] = 0.0
    regime_eligibility[:rows] = 0.0
    actual_entry[:rows] = -1
    actual_exit[:rows] = -1
    selected = replace(
        batch,
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(adverse),
        regime_unpredictability=_readonly(unpredictable),
        action_eligibility=_readonly(action_eligibility),
        regime_unpredictability_eligibility=_readonly(regime_eligibility),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
    )
    selected.validate()
    return selected


class _FixedClassificationPeer(torch.nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        rows = int(values.shape[0])
        action_shape = (
            rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        regime_shape = (
            rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        )
        logit = torch.logit(torch.tensor(self.probability))
        quantiles = torch.arange(5, dtype=values.dtype).reshape(1, 1, 1, 5)
        quantiles = quantiles.expand(*action_shape, 5)
        return Round74EventModelOutput(
            payoff_quantiles_bps=quantiles,
            maximum_adverse_excursion_quantiles_bps=quantiles,
            positive_payoff_logits=logit.expand(action_shape),
            adverse_selection_logits=logit.expand(action_shape),
            regime_unpredictability_logits=logit.expand(regime_shape),
        )


def test_round74_ensemble_averages_probabilities_not_logits() -> None:
    ensemble = Round74EventEnsemble("event_pooling_mlp", 2)
    ensemble.peers = torch.nn.ModuleList(
        (_FixedClassificationPeer(0.5), _FixedClassificationPeer(0.9))
    )

    output = ensemble(torch.zeros(3, 2, len(ROUND74_EVENT_FEATURE_NAMES)))

    expected = torch.full_like(output.positive_payoff_logits, 0.7)
    torch.testing.assert_close(
        torch.sigmoid(output.positive_payoff_logits),
        expected,
    )
    torch.testing.assert_close(
        torch.sigmoid(output.adverse_selection_logits),
        expected,
    )
    torch.testing.assert_close(
        torch.sigmoid(output.regime_unpredictability_logits),
        torch.full_like(output.regime_unpredictability_logits, 0.7),
    )
    mean_logit = (torch.logit(torch.tensor(0.5)) + torch.logit(torch.tensor(0.9))) / 2.0
    assert not torch.allclose(torch.sigmoid(mean_logit), torch.tensor(0.7))


def test_round74_complexity_gate_requires_consistent_complete_panel() -> None:
    candidate_ids = (
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    )
    parameter_counts = {
        "event_pooling_linear": 13_000,
        "event_pooling_mlp": 31_396,
        "causal_event_tcn": 129_060,
        "causal_event_attention": 151_876,
    }
    linear = (1.0,) * 12
    mlp = (0.9,) * 12
    tcn = (0.8,) * 12
    attention = (0.7,) * 12

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        {
            "event_pooling_linear": linear,
            "event_pooling_mlp": mlp,
            "causal_event_tcn": tcn,
            "causal_event_attention": attention,
        },
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
    )

    assert winner == "causal_event_attention"
    assert len(reports) == ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT == 3
    assert reports[0]["incumbent_candidate_id"] == "event_pooling_linear"
    assert reports[0]["challenger_win_count"] == 12
    assert reports[0]["challenger_loss_count"] == 0
    assert reports[0]["complete_tuning_panel"] is True
    assert reports[0]["all_paired_runs_noninferior"] is True
    assert reports[0]["maximum_paired_run_loss_degradation"] < 0.0
    assert reports[0]["promoted"] is True
    assert reports[1]["incumbent_candidate_id"] == "event_pooling_mlp"
    assert reports[1]["challenger_win_count"] == 12
    assert reports[1]["promoted"] is True
    assert reports[2]["incumbent_candidate_id"] == "causal_event_tcn"
    assert reports[2]["challenger_win_count"] == 12
    assert reports[2]["promoted"] is True


def test_round74_complexity_gate_rejects_point_better_but_weak_challengers() -> None:
    candidate_ids = (
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    )
    parameter_counts = {
        "event_pooling_linear": 13_000,
        "event_pooling_mlp": 31_396,
        "causal_event_tcn": 129_060,
        "causal_event_attention": 151_876,
    }
    linear = (1.0,) * 12
    weak = (0.8,) * 8 + (1.2,) * 4

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        {
            "event_pooling_linear": linear,
            "event_pooling_mlp": weak,
            "causal_event_tcn": weak,
            "causal_event_attention": weak,
        },
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
    )

    assert winner == "event_pooling_linear"
    assert all(report["promoted"] is False for report in reports)
    assert reports[0]["mean_proper_loss_improvement"] > 0.0
    assert reports[0]["complete_tuning_panel"] is True
    assert reports[0]["all_paired_runs_noninferior"] is False
    assert reports[0]["maximum_paired_run_loss_degradation"] > 0.0


def test_round74_complexity_gate_rejects_hidden_symbol_horizon_degradation() -> None:
    candidate_ids = ROUND74_EVENT_MODEL_CANDIDATES
    parameter_counts = {
        "event_pooling_linear": 13_000,
        "event_pooling_mlp": 31_396,
        "causal_event_tcn": 129_060,
        "causal_event_attention": 151_876,
    }
    run_losses = {
        "event_pooling_linear": (1.0,) * 12,
        "event_pooling_mlp": (0.9,) * 12,
        "causal_event_tcn": (1.0,) * 12,
        "causal_event_attention": (1.0,) * 12,
    }
    group_keys = tuple(
        f"{run_id:032x}:{symbol}:{horizon}"
        for run_id in range(12)
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
    )
    baseline_groups = {key: 1.0 for key in group_keys}
    mlp_groups = {key: 0.9 for key in group_keys}
    hidden_failure = next(key for key in group_keys if ":SOLUSDT:" in key)
    mlp_groups[hidden_failure] = 1.2

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        run_losses,
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
        candidate_group_losses={
            "event_pooling_linear": baseline_groups,
            "event_pooling_mlp": mlp_groups,
            "causal_event_tcn": baseline_groups,
            "causal_event_attention": baseline_groups,
        },
    )

    assert winner == "event_pooling_linear"
    first = reports[0]
    assert first["mean_proper_loss_improvement"] > 0.0
    assert first["all_paired_runs_noninferior"] is True
    assert first["subgroup_gate_applied"] is True
    assert first["paired_run_symbol_horizon_group_count"] == len(group_keys)
    assert first["worst_run_symbol_horizon_group"] == hidden_failure
    assert first["maximum_paired_group_loss_degradation"] == pytest.approx(0.2)
    assert first["all_paired_run_symbol_horizon_groups_noninferior"] is False
    assert first["promoted"] is False


def test_round74_complexity_gate_excludes_exact_ties() -> None:
    winner, reports = _complexity_gated_candidate_id(
        (
            "event_pooling_linear",
            "event_pooling_mlp",
            "causal_event_tcn",
            "causal_event_attention",
        ),
        {
            "event_pooling_linear": (1.0,) * 12,
            "event_pooling_mlp": (0.9,) * 9 + (1.0,) * 3,
            "causal_event_tcn": (1.0,) * 12,
            "causal_event_attention": (1.0,) * 12,
        },
        {
            "event_pooling_linear": 13_000,
            "event_pooling_mlp": 31_396,
            "causal_event_tcn": 129_060,
            "causal_event_attention": 151_876,
        },
        minimum_mean_loss_improvement=1e-5,
    )

    assert winner == "event_pooling_mlp"
    assert reports[0]["exact_tie_count"] == 3
    assert reports[0]["paired_capture_run_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    assert reports[0]["complete_tuning_panel"] is True
    assert reports[0]["all_paired_runs_noninferior"] is True
    assert reports[0]["promoted"] is True
    assert reports[1]["incumbent_candidate_id"] == "event_pooling_mlp"
    assert reports[1]["promoted"] is False
    assert reports[2]["incumbent_candidate_id"] == "event_pooling_mlp"
    assert reports[2]["promoted"] is False


def _write_rehashed_policy(path: Path, policy: dict[str, object]) -> Path:
    policy.pop("policy_sha256", None)
    encoded = json.dumps(
        policy,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()
    policy["policy_sha256"] = digest
    target = path / f"round74-pretest-policy-{digest}.json"
    target.write_text(
        json.dumps(
            policy,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return target


def test_round74_trainer_rejects_test_data_before_backend_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    import simple_ai_trading.impact_absorption_event_training as training

    def unexpected_backend(_value: object) -> object:
        raise AssertionError("backend resolution must follow data validation")

    monkeypatch.setattr(training, "resolve_backend", unexpected_backend)
    test_batch = _batch(
        "test",
        start_wall_ns=WALL_NS,
        identity=9,
    )
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )

    with pytest.raises(ValueError, match="rejects 'test' data"):
        train_and_seal_round74_pretest_policy(
            [test_batch],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )


def test_round74_trainer_requires_order_purge_and_per_run_context(
    tmp_path: object,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + 2_000_000_000,
        identity=2,
    )
    with pytest.raises(ValueError, match="minimum purge"):
        train_and_seal_round74_pretest_policy(
            [training],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )

    mixed_context = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
            identity=2,
        ),
        target_context_sha256=("3" * 64, "4" * 64),
    )
    mixed_context.validate()
    with pytest.raises(ValueError, match="mixes target contexts"):
        train_and_seal_round74_pretest_policy(
            [training],
            [mixed_context],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )

    global_tuning = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
            identity=2,
        ),
        window_representation="global_cross_asset",
    )
    global_tuning.validate()
    with pytest.raises(ValueError, match="window representation differs"):
        train_and_seal_round74_pretest_policy(
            [training],
            [global_tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )


def test_round74_trainer_rejects_mixed_or_repeated_capture_runs(
    tmp_path: object,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )
    mixed = replace(
        training,
        run_id=(training.run_id[0], f"{9:032x}"),
    )
    mixed.validate()

    with pytest.raises(ValueError, match="mixes capture runs"):
        train_and_seal_round74_pretest_policy(
            [mixed],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )

    repeated = _batch(
        "training",
        start_wall_ns=WALL_NS + 10_000_000_000,
        identity=1,
    )
    with pytest.raises(ValueError, match="capture run is repeated"):
        train_and_seal_round74_pretest_policy(
            [training, repeated],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )


def test_round74_trainer_records_and_skips_fully_censored_minibatches(
    tmp_path: Path,
) -> None:
    training = _with_fully_censored_prefix(
        _batch("training", start_wall_ns=WALL_NS, identity=1, rows=3),
        2,
    )
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )

    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=_config(),
    )
    _model, policy = load_round74_pretest_policy(artifact.policy_path)

    for report in policy["candidate_panel"].values():
        optimization = report["peer_reports"][0]["history"][0]["optimization_metrics"]
        assert optimization["fully_censored_minibatches"] == 1.0
        assert optimization["fully_censored_rows"] == 2.0
        assert optimization["eligible_action_targets"] > 0.0


def test_round74_trainer_balances_optimizer_contributions_by_capture_run(
    tmp_path: Path,
) -> None:
    short_run = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=1,
        rows=2,
    )
    long_run = _batch(
        "training",
        start_wall_ns=WALL_NS + 10_000_000_000,
        identity=2,
        rows=5,
    )
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 20_000_000_000,
        identity=3,
    )

    artifact = train_and_seal_round74_pretest_policy(
        [short_run, long_run],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=_config(),
    )
    _model, policy = load_round74_pretest_policy(artifact.policy_path)

    assert policy["optimization_population"] == {
        "unit": "capture_run",
        "optimizer_step": (
            "one eligible minibatch per training capture run with gradient accumulation"
        ),
        "gradient_divisor": "training_capture_run_count",
        "shorter_run_policy": (
            "deterministic epoch-rotated cycling of eligible minibatches"
        ),
        "fully_censored_minibatches_contribute_gradients": False,
        "fully_censored_capture_run_policy": "reject",
        "row_pooled_optimizer_steps_permitted": False,
        "cohort_training_capture_runs": 120,
        "cohort_model_selection_capture_runs": 12,
        "calibration_or_policy_selection_runs_used_for_candidate_fit": False,
    }
    for report in policy["candidate_panel"].values():
        optimization = report["peer_reports"][0]["history"][0]["optimization_metrics"]
        assert optimization["run_count"] == 2.0
        assert optimization["optimizer_steps"] == 3.0
        assert optimization["run_contributions_per_optimizer_step"] == 2.0
        assert optimization["minimum_run_minibatch_contributions"] == 3.0
        assert optimization["maximum_run_minibatch_contributions"] == 3.0
        assert optimization["minimum_eligible_minibatches_per_run"] == 1.0
        assert optimization["maximum_eligible_minibatches_per_run"] == 3.0
        assert optimization["worst_run_loss"] >= (optimization["run_balanced_loss"])


def test_round74_device_run_group_preserves_equal_run_gradient() -> None:
    first_batch = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=1,
        rows=2,
    )
    second_batch = _batch(
        "training",
        start_wall_ns=WALL_NS + 10_000_000_000,
        identity=2,
        rows=3,
    )
    first_slice = slice(0, 2)
    second_slice = slice(0, 3)
    torch.manual_seed(7401)
    independent_model = build_round74_event_model("event_pooling_linear")
    grouped_model = build_round74_event_model("event_pooling_linear")
    grouped_model.load_state_dict(independent_model.state_dict(), strict=True)

    independent = (
        _loss_for_minibatch(independent_model, first_batch, first_slice, "cpu"),
        _loss_for_minibatch(independent_model, second_batch, second_slice, "cpu"),
    )
    (
        torch.stack(tuple(item[0] for item in independent)).sum() / len(independent)
    ).backward()
    grouped = _losses_for_minibatch_group(
        grouped_model,
        ((first_batch, first_slice), (second_batch, second_slice)),
        "cpu",
    )
    (torch.stack(tuple(item[0] for item in grouped)).sum() / len(grouped)).backward()

    for independent_item, grouped_item in zip(independent, grouped, strict=True):
        assert torch.allclose(
            independent_item[0], grouped_item[0], atol=1e-6, rtol=1e-6
        )
        assert independent_item[2:] == grouped_item[2:]
        for name in independent_item[1]:
            assert torch.allclose(
                independent_item[1][name],
                grouped_item[1][name],
                atol=1e-6,
                rtol=1e-6,
            )
    for independent_parameter, grouped_parameter in zip(
        independent_model.parameters(),
        grouped_model.parameters(),
        strict=True,
    ):
        assert independent_parameter.grad is not None
        assert grouped_parameter.grad is not None
        assert torch.allclose(
            independent_parameter.grad,
            grouped_parameter.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_round74_pretest_policy_is_safe_hash_bound_and_non_authoritative(
    tmp_path: Path,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )
    later_tuning = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 10_000_000_000,
            identity=3,
            rows=3,
        ),
        target_context_sha256=tuple("4" * 64 for _ in range(3)),
    )
    later_tuning.validate()
    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning, later_tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=_config(),
    )
    model, policy = load_round74_pretest_policy(artifact.policy_path)

    assert model.candidate_id == artifact.selected_candidate_id
    assert policy["policy_sha256"] == artifact.policy_sha256
    assert policy["model_artifact"]["sha256"] == artifact.model_sha256
    assert policy["model_artifact"]["pickle_permitted"] is False
    assert policy["development_data"]["test_batches_consumed"] == 0
    development = policy["development_data"]
    assert development["window_representation"] == "per_symbol"
    assert development["representative_window_policy_kind"] == "preflight"
    assert development["matched_preparation_sha256"] is None
    assert development["target_context_panel_schema_version"] == (
        ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
    )
    assert development["target_context_sha256"] == ["3" * 64, "4" * 64]
    assert development["target_context_count"] == 2
    assert len(development["target_context_panel_sha256"]) == 64
    target_loss_scale = policy["target_loss_scale"]
    assert target_loss_scale["training_batch_sha256"] == [training.batch_sha256]
    assert target_loss_scale["fit_partition_role"] == "training"
    assert target_loss_scale["tuning_targets_used"] is False
    assert target_loss_scale["test_targets_used"] is False
    assert target_loss_scale["forecast_units_changed"] is False
    assert policy["backend"]["kind"] == "cpu"
    assert set(policy["candidate_panel"]) == {
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    }
    assert policy["selection"]["backtest_metric_used_for_selection"] is False
    assert policy["selection"]["criterion"].startswith(
        "sequential parameter-count complexity promotion"
    )
    assert policy["selection"]["selected_candidate_id"] == "event_pooling_linear"
    assert policy["selection"]["ordered_candidate_ids"] == [
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    ]
    assert policy["selection"]["planned_comparison_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
    )
    assert policy["selection"]["required_paired_capture_run_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    assert (
        policy["selection"]["statistical_independence_or_significance_claim"] is False
    )
    assert len(policy["selection"]["promotion_reports"]) == 3
    assert all(
        report["complete_tuning_panel"] is False and report["promoted"] is False
        for report in policy["selection"]["promotion_reports"]
    )
    for report in policy["candidate_panel"].values():
        assert len(report["ensemble_tuning_run_losses"]) == 2
    selected_metrics = policy["selection"]["selected_tuning_metrics"]
    assert selected_metrics["run_count"] == 2.0
    assert selected_metrics["worst_run_loss"] >= (selected_metrics["run_balanced_loss"])
    assert selected_metrics["run_balanced_loss"] != selected_metrics["loss"]
    assert artifact.tuning_loss == selected_metrics["run_balanced_loss"]
    assert all(
        policy["authority"][name] is False
        for name in (
            "sealed_test_evaluated",
            "representative_market_evidence_claim",
            "financial_edge_tested",
            "profitability_claim",
            "ai_uplift_claim",
            "paper_trading_authority",
            "testnet_trading_authority",
            "live_trading_authority",
        )
    )
    tampered = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    tampered["authority"]["profitability_claim"] = True
    tampered_path = _write_rehashed_policy(tmp_path, tampered)
    with pytest.raises(ValueError, match="overstates authority"):
        load_round74_pretest_policy(tampered_path)

    changed_source = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_source["source_binding"]["model_module_sha256"] = "f" * 64
    changed_source_path = _write_rehashed_policy(tmp_path, changed_source)
    with pytest.raises(ValueError, match="source binding differs"):
        load_round74_pretest_policy(changed_source_path)

    changed_context_panel = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_context_panel["development_data"]["target_context_sha256"].append("5" * 64)
    changed_context_path = _write_rehashed_policy(
        tmp_path,
        changed_context_panel,
    )
    with pytest.raises(ValueError, match="target-context panel differs"):
        load_round74_pretest_policy(changed_context_path)

    changed_representation = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
    changed_representation["development_data"]["window_representation"] = "unbound"
    changed_representation_path = _write_rehashed_policy(
        tmp_path,
        changed_representation,
    )
    with pytest.raises(ValueError, match="data identity differs"):
        load_round74_pretest_policy(changed_representation_path)

    changed_promotion = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_promotion["selection"]["promotion_reports"][0]["promoted"] = True
    changed_promotion_path = _write_rehashed_policy(
        tmp_path,
        changed_promotion,
    )
    with pytest.raises(ValueError, match="selection or artifact differs"):
        load_round74_pretest_policy(changed_promotion_path)

    artifact.model_path.write_bytes(artifact.model_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="model artifact differs"):
        load_round74_pretest_policy(artifact.policy_path)


def test_round74_fixed_architecture_mode_has_no_second_model_search(
    tmp_path: Path,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )
    config = replace(
        _config(),
        candidate_ids=("event_pooling_linear",),
        architecture_selection_mode="fixed",
    )
    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=config,
    )
    _model, policy = load_round74_pretest_policy(artifact.policy_path)

    assert artifact.selected_candidate_id == "event_pooling_linear"
    assert list(policy["candidate_panel"]) == ["event_pooling_linear"]
    assert policy["training_policy"]["architecture_selection_mode"] == "fixed"
    assert policy["selection"]["architecture_selection_mode"] == "fixed"
    assert policy["selection"]["planned_comparison_count"] == 0
    assert policy["selection"]["required_paired_capture_run_count"] == 0
    assert policy["selection"]["promotion_reports"] == []
    assert policy["selection"]["criterion"].startswith(
        "fixed baseline-selected architecture"
    )

    with pytest.raises(ValueError, match="configuration differs"):
        replace(
            config,
            candidate_ids=("event_pooling_linear", "event_pooling_mlp"),
        ).validate()
