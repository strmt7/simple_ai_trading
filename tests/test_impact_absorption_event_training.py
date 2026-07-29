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
    ROUND74_EVENT_CLOCK_FEATURE_INDICES,
    ROUND74_EVENT_CLOCK_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_VIEWS,
    ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES,
    ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,
    ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES,
    ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION,
    Round74EventEnsemble,
    Round74EventTargetLossScale,
    Round74EventTrainingConfig,
    _CandidateFit,
    _complexity_gated_candidate_id,
    _eligible_target_minibatch_schedule,
    _eligible_target_weighted_group_loss,
    _empty_metric_sums,
    _feature_view_contains_order_flow,
    _feature_view_promotion_report,
    _fit_candidate,
    _loss_for_minibatch,
    _losses_for_minibatch_group,
    _order_flow_challenger_feature_view,
    _select_state_conditioned_flow_with_ablation_gate,
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
    round74_event_encoder_parameters,
)
from simple_ai_trading.impact_absorption_event_pretraining import (  # noqa: E402
    Round74EventPretrainingConfig,
    _Round74NextEventHead,
    _next_event_loss,
    _next_event_row_losses,
    _validate_pretraining_split,
    build_round74_event_pretraining_split,
    pretrain_round74_event_encoder,
)
from simple_ai_trading.round74_event_model_operator import (  # noqa: E402
    split_round74_tuning_batch_roles,
)
from simple_ai_trading.round74_segmented_model_operator import (  # noqa: E402
    Round74SegmentedTrainingSplit,
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
    directional_move = generator.normal(size=action_shape[:-1]).astype(np.float32)
    round_trip_cost = (
        np.abs(generator.normal(size=action_shape[:-1])).astype(np.float32) * 0.05
        + 0.01
    )
    payoff = np.stack(
        (
            directional_move - round_trip_cost,
            -directional_move - round_trip_cost,
        ),
        axis=2,
    ).astype(np.float32)
    adverse_excursion = np.maximum.accumulate(
        np.abs(generator.normal(size=action_shape)),
        axis=1,
    ).astype(np.float32)
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


def _assert_gradient_health(
    optimization: dict[str, float],
    *,
    optimizer_steps: int,
    gradient_clip_norm: float,
) -> None:
    minimum = optimization["preclip_gradient_norm_minimum"]
    mean = optimization["preclip_gradient_norm_mean"]
    maximum = optimization["preclip_gradient_norm_maximum"]
    zero_steps = optimization["zero_gradient_steps"]
    clipped_steps = optimization["clipped_gradient_steps"]

    assert 0.0 <= minimum <= mean <= maximum
    assert zero_steps in {float(step) for step in range(optimizer_steps + 1)}
    assert clipped_steps in {float(step) for step in range(optimizer_steps + 1)}
    assert optimization["zero_gradient_fraction"] == pytest.approx(
        zero_steps / optimizer_steps
    )
    assert optimization["gradient_clip_fraction"] == pytest.approx(
        clipped_steps / optimizer_steps
    )
    assert optimization["gradient_clip_norm_limit"] == pytest.approx(gradient_clip_norm)


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


def test_causal_next_event_pretraining_is_training_only_and_purged() -> None:
    training = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=91,
        rows=450,
    )
    feature_values = training.feature_values.copy()
    feature_values[:, :, 8:ROUND74_EVENT_BINARY_FEATURE_COUNT] = 0.0
    training = replace(training, feature_values=_readonly(feature_values))
    config = Round74EventPretrainingConfig(
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=128,
        minimum_partition_rows_per_symbol=2,
    )
    split = build_round74_event_pretraining_split((training,), config=config)
    changed_targets = training.net_payoff_bps.copy()
    changed_targets[training.action_eligibility == 1.0] *= 1000.0
    target_changed_training = replace(
        training,
        net_payoff_bps=_readonly(changed_targets),
    )
    target_changed_training.validate()
    target_changed_split = build_round74_event_pretraining_split(
        (target_changed_training,),
        config=config,
    )

    assert split.training_rows > 0
    assert split.validation_rows > 0
    assert target_changed_training.batch_sha256 != training.batch_sha256
    assert target_changed_split.split_sha256 == split.split_sha256
    _validate_pretraining_split(split, (target_changed_training,), config)
    copied_feature_training = replace(
        training,
        feature_values=_readonly(training.feature_values.copy()),
    )
    with pytest.raises(ValueError, match="split contract differs"):
        _validate_pretraining_split(split, (copied_feature_training,), config)
    assert set(split.training_indices[0]).isdisjoint(split.validation_indices[0])
    assert int(training.decision_wall_ns[split.training_indices[0][-1]]) < int(
        training.decision_wall_ns[split.validation_indices[0][0]]
    )
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        symbol_mask = np.asarray(training.symbol) == symbol
        train_rows = split.training_indices[0][symbol_mask[split.training_indices[0]]]
        validation_rows = split.validation_indices[0][
            symbol_mask[split.validation_indices[0]]
        ]
        assert (
            int(training.anchor_index[validation_rows[0]])
            - int(training.anchor_index[train_rows[-1]])
            > ROUND74_EVENT_SEQUENCE_LENGTH
        )

    torch.manual_seed(7411)
    model = build_round74_event_model("causal_event_tcn")
    downstream_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("readout.", "heads."))
    }
    report = pretrain_round74_event_encoder(
        model,
        (training,),
        device=torch.device("cpu"),
        config=config,
        split=split,
    )

    assert len(report["training_feature_batch_sha256"]) == 1
    assert len(report["training_feature_batch_sha256"][0]) == 64
    assert report["training_capture_run_count"] == 1
    assert report["supervised_targets_used"] is False
    assert report["tuning_features_used"] is False
    assert report["tuning_targets_used"] is False
    assert report["calibration_data_used"] is False
    assert report["test_data_used"] is False
    assert report["initial_encoder_sha256"] != report["final_encoder_sha256"]
    assert downstream_before
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in downstream_before.items()
    )
    assert report["encoder_state_restored"] is True
    assert report["temporary_prediction_head_persisted"] is False
    assert report["financial_edge_claim"] is False


def test_causal_pretraining_device_group_preserves_equal_run_gradient() -> None:
    first = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=93,
        rows=1,
    ).feature_values
    second = _batch(
        "training",
        start_wall_ns=WALL_NS + 1_000_000_000,
        identity=94,
        rows=2,
    ).feature_values
    torch.manual_seed(7411)
    independent_model = build_round74_event_model("causal_event_tcn")
    independent_head = _Round74NextEventHead(64)
    grouped_model = build_round74_event_model("causal_event_tcn")
    grouped_head = _Round74NextEventHead(64)
    grouped_model.load_state_dict(independent_model.state_dict(), strict=True)
    grouped_head.load_state_dict(independent_head.state_dict(), strict=True)
    independent_model.eval()
    independent_head.eval()
    grouped_model.eval()
    grouped_head.eval()

    first_loss, _event, _continuous = _next_event_loss(
        independent_model,
        independent_head,
        torch.from_numpy(first.copy()),
        masked_feature_indices=(),
    )
    second_loss, _event, _continuous = _next_event_loss(
        independent_model,
        independent_head,
        torch.from_numpy(second.copy()),
        masked_feature_indices=(),
    )
    ((first_loss + second_loss) / 2.0).backward()

    grouped_values = torch.from_numpy(np.concatenate((first, second), axis=0))
    grouped_rows, _event_rows, _continuous_rows = _next_event_row_losses(
        grouped_model,
        grouped_head,
        grouped_values,
        masked_feature_indices=(),
    )
    ((grouped_rows[:1].mean() + grouped_rows[1:].mean()) / 2.0).backward()

    independent_parameters = (
        *round74_event_encoder_parameters(independent_model),
        *tuple(independent_head.parameters()),
    )
    grouped_parameters = (
        *round74_event_encoder_parameters(grouped_model),
        *tuple(grouped_head.parameters()),
    )
    for independent, grouped in zip(
        independent_parameters,
        grouped_parameters,
        strict=True,
    ):
        assert independent.grad is not None
        assert grouped.grad is not None
        assert torch.allclose(
            independent.grad,
            grouped.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_market_state_pretraining_keeps_the_unmasked_next_event_target() -> None:
    values = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=94,
        rows=3,
    ).feature_values
    model = build_round74_event_model("causal_event_tcn")
    head = _Round74NextEventHead(64)
    masked_indices = tuple(
        sorted(
            {
                *ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,
                *ROUND74_EVENT_CLOCK_FEATURE_INDICES,
            }
        )
    )

    row_loss, event_loss, continuous_loss = _next_event_row_losses(
        model,
        head,
        torch.from_numpy(values.copy()),
        masked_feature_indices=masked_indices,
    )

    assert row_loss.shape == event_loss.shape == continuous_loss.shape == (3,)
    assert torch.isfinite(row_loss).all()


def test_segmented_pretraining_reuses_one_split_across_peers(
    tmp_path: Path,
) -> None:
    scaler = _scaler()
    training = replace(
        _batch(
            "training",
            start_wall_ns=WALL_NS,
            identity=95,
            rows=450,
        ),
        scaler_sha256=scaler.scaler_sha256,
    )
    tuning = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + 1_000_000_000_000,
            identity=105,
            rows=450,
        ),
        scaler_sha256=scaler.scaler_sha256,
    )
    config = Round74EventTrainingConfig(
        candidate_ids=("causal_event_tcn",),
        seeds=(7411, 7412),
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=450,
        minimum_role_rows=450,
        execution_mode="segmented_cohort",
        architecture_selection_mode="fixed",
        pretraining=Round74EventPretrainingConfig(
            maximum_epochs=1,
            early_stopping_patience=1,
            minibatch_rows=450,
            minimum_partition_rows_per_symbol=2,
            device_run_group_size=2,
        ),
    )
    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=config,
        representative_window_policy_sha256=round74_segmented_window_policy()[
            "policy_sha256"
        ],
        feature_scaler=scaler,
    )
    _model, policy = load_round74_pretest_policy(artifact.policy_path)
    peers = policy["initialization_panel"]["causal_next_event_pretrained"][
        "peer_reports"
    ]
    split_sha256 = {peer["causal_pretraining"]["split_sha256"] for peer in peers}
    feature_batches = {
        tuple(peer["causal_pretraining"]["training_feature_batch_sha256"])
        for peer in peers
    }
    partition_rows = {
        (
            peer["causal_pretraining"]["training_rows"],
            peer["causal_pretraining"]["validation_rows"],
        )
        for peer in peers
    }
    assert len(peers) == 2
    assert len(split_sha256) == 1
    assert len(feature_batches) == 1
    assert len(partition_rows) == 1
    assert (
        policy["feature_view_selection"]["selected_tuning_metrics"]
        == (policy["initialization_panel"]["random"]["ensemble_tuning_metrics"])
    )
    assert (
        policy["feature_view_selection"]["selected_feature_view"]
        == (policy["initialization_panel"]["random"]["feature_view"])
    )

    tampered = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    tampered["initialization_panel"]["causal_next_event_pretrained"]["peer_reports"][1][
        "causal_pretraining"
    ]["split_sha256"] = "f" * 64
    tampered_path = _write_rehashed_policy(tmp_path, tampered)
    with pytest.raises(ValueError, match="peer split differs"):
        load_round74_pretest_policy(tampered_path)


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


def test_checkpoint_selection_and_promotion_use_disjoint_batches() -> None:
    training = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=711,
        rows=3,
    )
    early_stopping = _batch(
        "training",
        start_wall_ns=WALL_NS + PURGE_NS + 1_000_000_000,
        identity=712,
        rows=3,
    )
    promotions = (
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + 2 * PURGE_NS + 2_000_000_000,
            identity=713,
            rows=3,
        ),
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + 2 * PURGE_NS + 6_000_000_000,
            identity=714,
            rows=3,
        ),
    )
    config = Round74EventTrainingConfig(
        candidate_ids=("event_pooling_linear",),
        seeds=(7411,),
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=3,
        minimum_role_rows=3,
        execution_mode="preflight",
        architecture_selection_mode="fixed",
    )
    target_loss_scale = fit_round74_event_target_loss_scale(
        (training,),
        require_complete_panel=False,
    )

    fit = _fit_candidate(
        "event_pooling_linear",
        "market_state_clock_neutral",
        (training,),
        (early_stopping,),
        promotions,
        config=config,
        device=torch.device("cpu"),
        target_loss_scale=target_loss_scale,
    )

    peer = fit.peer_reports[0]
    assert peer["best_early_stopping_metrics"]["run_count"] == 1.0
    assert peer["history"][0]["early_stopping_metrics"]["run_count"] == 1.0
    assert fit.ensemble_metrics["run_count"] == 2.0
    assert len(fit.ensemble_run_losses) == 2


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
                "positive_log_loss",
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
    _assert_gradient_health(
        first["optimization_metrics"],
        optimizer_steps=1,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    selected_metrics = policy["selection"]["selected_tuning_metrics"]
    assert artifact.tuning_loss == pytest.approx(selected_metrics["loss"])


def test_prepared_roles_forward_segmented_model_selection_without_discarding_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simple_ai_trading.impact_absorption_event_training as training_subject

    model_count, calibration_count, policy_count, ai_count = 43, 22, 21, 17
    total = model_count + calibration_count + policy_count + ai_count
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
    policy_end = calibration_end + policy_count
    subpartition = Round74SegmentedTuningSubpartition(
        parent_partition_sha256="1" * 64,
        cohort_plan_sha256="a" * 64,
        model_selection_run_ids=tuple(batch.run_id[0] for batch in batches[:model_end]),
        calibration_run_ids=tuple(
            batch.run_id[0] for batch in batches[model_end:calibration_end]
        ),
        policy_selection_run_ids=tuple(
            batch.run_id[0] for batch in batches[calibration_end:policy_end]
        ),
        ai_qualification_run_ids=tuple(
            batch.run_id[0] for batch in batches[policy_end:]
        ),
        model_selection_slot_ordinals=tuple(range(514, 557)),
        calibration_slot_ordinals=tuple(range(557, 579)),
        policy_selection_slot_ordinals=tuple(range(579, 600)),
        ai_qualification_slot_ordinals=tuple(range(600, 617)),
        model_selection_eligible_anchor_ns=(900_000_000_000,) * model_count,
        calibration_eligible_anchor_ns=(900_000_000_000,) * calibration_count,
        policy_selection_eligible_anchor_ns=(900_000_000_000,) * policy_count,
        ai_qualification_eligible_anchor_ns=(900_000_000_000,) * ai_count,
    )
    roles = split_round74_tuning_batch_roles(
        batches,
        subpartition=subpartition,
    )
    roles.validate()
    assert len(roles.ai_qualification_batches) == ai_count
    assert (
        tuple(batch.run_id[0] for batch in roles.ai_qualification_batches)
        == subpartition.ai_qualification_run_ids
    )
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

    training = tuple(
        _batch(
            "training",
            start_wall_ns=(WALL_NS - (160 - index) * 1_500_000_000_000),
            identity=1_000 + index,
        )
        for index in range(160)
    )
    split = Round74SegmentedTrainingSplit(
        parent_partition_sha256="1" * 64,
        cohort_plan_sha256="a" * 64,
        optimization_run_ids=tuple(batch.run_id[0] for batch in training[:128]),
        purged_run_ids=(),
        early_stopping_run_ids=tuple(batch.run_id[0] for batch in training[128:]),
        optimization_slot_ordinals=tuple(range(128)),
        purged_slot_ordinals=(),
        early_stopping_slot_ordinals=tuple(range(128, 160)),
        optimization_last_eligible_anchor_wall_ns=int(
            training[127].decision_wall_ns[-1]
        ),
        early_stopping_first_eligible_anchor_wall_ns=int(
            training[128].decision_wall_ns[0]
        ),
    )
    split.validate()
    with pytest.raises(TypeError, match="training split is required"):
        train_and_seal_round74_pretest_policy_from_prepared_roles(
            training,
            roles,
            output_directory=tmp_path,
            compute_backend="cpu",
            config=config,
            feature_scaler=_scaler(),
        )
    with pytest.raises(ValueError, match="scaler provenance differs"):
        train_and_seal_round74_pretest_policy_from_prepared_roles(
            training,
            roles,
            output_directory=tmp_path,
            compute_backend="cpu",
            config=config,
            feature_scaler=_scaler(),
            segmented_training_split=split,
        )
    segmented_scaler = replace(
        _scaler(),
        fit_source_scope="segmented_optimization_training_runs",
        fit_source_run_ids=tuple(batch.run_id[0] for batch in training[:128]),
        fit_source_partition_sha256="1" * 64,
        fit_source_selection_sha256=split.split_sha256,
    )
    result = train_and_seal_round74_pretest_policy_from_prepared_roles(
        training,
        roles,
        output_directory=tmp_path,
        compute_backend="cpu",
        config=config,
        feature_scaler=segmented_scaler,
        segmented_training_split=split,
    )

    assert result is sentinel
    assert observed["tuning_batches"] == roles.model_selection_batches
    protocol = observed["selection_protocol"]
    assert isinstance(protocol, training_subject.Round74EventSelectionProtocol)
    assert len(protocol.optimization_batches) == 128
    assert len(protocol.purged_training_batches) == 0
    assert len(protocol.early_stopping_batches) == 32
    assert [len(batches) for batches in protocol.promotion_stage_batches] == [
        9,
        9,
        8,
        9,
        8,
    ]
    assert (
        tuple(
            batch
            for stage_batches in protocol.promotion_stage_batches
            for batch in stage_batches
        )
        == roles.model_selection_batches
    )
    protocol_payload = protocol.as_dict()
    assert len(protocol.protocol_sha256) == 64
    assert protocol_payload["target_loss_scale_fit_scope"] == (
        "optimization_training_runs_only"
    )
    assert protocol_payload["feature_scaler_fit_scope"] == (
        "segmented_optimization_training_runs_only"
    )
    assert protocol_payload["feature_scaler_fit_selection_sha256"] == split.split_sha256
    assert protocol_payload["training_split_sha256"] == split.split_sha256
    assert protocol_payload["training_split"] == split.as_dict()
    assert protocol_payload["early_stopping_targets_used_for_gradient_updates"] is False
    assert protocol_payload["promotion_targets_used_for_checkpoint_selection"] is False
    assert protocol_payload["cross_stage_promotion_run_reuse_permitted"] is False
    assert protocol_payload["chronological_gap_ns"] >= PURGE_NS
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
        positive_logit = torch.logit(torch.tensor(self.probability / 2.0))
        quantiles = torch.arange(5, dtype=values.dtype).reshape(1, 1, 1, 5)
        quantiles = quantiles.expand(*action_shape, 5)
        return Round74EventModelOutput(
            payoff_quantiles_bps=quantiles,
            maximum_adverse_excursion_quantiles_bps=quantiles,
            positive_payoff_logits=positive_logit.expand(action_shape),
            adverse_selection_logits=logit.expand(action_shape),
            regime_unpredictability_logits=logit.expand(regime_shape),
        )


class _RecordingClassificationPeer(_FixedClassificationPeer):
    def __init__(self) -> None:
        super().__init__(0.5)
        self.last_values: torch.Tensor | None = None

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        self.last_values = values.detach().clone()
        return super().forward(values)


def test_round74_ensemble_averages_probabilities_not_logits() -> None:
    ensemble = Round74EventEnsemble("event_pooling_mlp", 2)
    ensemble.peers = torch.nn.ModuleList(
        (_FixedClassificationPeer(0.5), _FixedClassificationPeer(0.9))
    )

    output = ensemble(torch.zeros(3, 2, len(ROUND74_EVENT_FEATURE_NAMES)))

    expected_positive = torch.full_like(output.positive_payoff_logits, 0.35)
    torch.testing.assert_close(
        torch.sigmoid(output.positive_payoff_logits),
        expected_positive,
    )
    torch.testing.assert_close(
        torch.sigmoid(output.adverse_selection_logits),
        torch.full_like(output.adverse_selection_logits, 0.7),
    )
    torch.testing.assert_close(
        torch.sigmoid(output.regime_unpredictability_logits),
        torch.full_like(output.regime_unpredictability_logits, 0.7),
    )
    diagnostics = output.epistemic_diagnostics
    assert diagnostics is not None
    assert diagnostics.peer_count == 2
    torch.testing.assert_close(
        diagnostics.payoff_quantile_standard_deviation_bps,
        torch.zeros_like(diagnostics.payoff_quantile_standard_deviation_bps),
    )
    torch.testing.assert_close(
        diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps,
        torch.zeros_like(
            diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps
        ),
    )
    torch.testing.assert_close(
        diagnostics.positive_payoff_probability_standard_deviation,
        torch.full_like(
            diagnostics.positive_payoff_probability_standard_deviation,
            0.1,
        ),
    )
    torch.testing.assert_close(
        diagnostics.adverse_selection_probability_standard_deviation,
        torch.full_like(
            diagnostics.adverse_selection_probability_standard_deviation,
            0.2,
        ),
    )
    torch.testing.assert_close(
        diagnostics.regime_unpredictability_probability_standard_deviation,
        torch.full_like(
            diagnostics.regime_unpredictability_probability_standard_deviation,
            0.2,
        ),
    )
    mean_logit = (
        torch.logit(torch.tensor(0.5 / 2.0)) + torch.logit(torch.tensor(0.9 / 2.0))
    ) / 2.0
    assert not torch.allclose(torch.sigmoid(mean_logit), torch.tensor(0.35))


def test_round74_clock_neutral_view_masks_only_exchange_clock_features() -> None:
    ensemble = Round74EventEnsemble(
        "event_pooling_linear",
        1,
        feature_view="clock_neutral",
    )
    peer = _RecordingClassificationPeer()
    ensemble.peers = torch.nn.ModuleList((peer,))
    values = torch.arange(
        2 * 3 * len(ROUND74_EVENT_FEATURE_NAMES),
        dtype=torch.float32,
    ).reshape(2, 3, len(ROUND74_EVENT_FEATURE_NAMES))

    output = ensemble(values)

    assert peer.last_values is not None
    assert output.epistemic_diagnostics is None
    expected = values.clone()
    expected[:, :, list(ROUND74_EVENT_CLOCK_FEATURE_INDICES)] = 0.0
    torch.testing.assert_close(peer.last_values, expected)
    assert len(ROUND74_EVENT_CLOCK_FEATURE_NAMES) == 11
    assert (
        tuple(
            ROUND74_EVENT_FEATURE_NAMES[index]
            for index in ROUND74_EVENT_CLOCK_FEATURE_INDICES
        )
        == ROUND74_EVENT_CLOCK_FEATURE_NAMES
    )
    assert {
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    }.issubset(ROUND74_EVENT_CLOCK_FEATURE_NAMES)
    assert not {
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    } & set(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES)


def test_round74_state_views_mask_the_fixed_order_flow_panel() -> None:
    values = torch.arange(
        2 * 3 * len(ROUND74_EVENT_FEATURE_NAMES),
        dtype=torch.float32,
    ).reshape(2, 3, len(ROUND74_EVENT_FEATURE_NAMES))
    for feature_view, masked_indices in (
        (
            "market_state_clock_neutral",
            tuple(
                sorted(
                    {
                        *ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,
                        *ROUND74_EVENT_CLOCK_FEATURE_INDICES,
                    }
                )
            ),
        ),
        ("market_state_with_clock", ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES),
    ):
        ensemble = Round74EventEnsemble(
            "event_pooling_linear",
            1,
            feature_view=feature_view,
        )
        peer = _RecordingClassificationPeer()
        ensemble.peers = torch.nn.ModuleList((peer,))

        ensemble(values)

        assert peer.last_values is not None
        expected = values.clone()
        expected[:, :, list(masked_indices)] = 0.0
        torch.testing.assert_close(peer.last_values, expected)
    assert len(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES) == 31
    assert len(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES) == 33
    assert (
        tuple(
            ROUND74_EVENT_FEATURE_NAMES[index]
            for index in ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES
        )
        == ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES
    )
    assert not set(ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES) & set(
        ROUND74_EVENT_CLOCK_FEATURE_INDICES
    )
    assert (
        set(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES)
        | set(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES)
        | set(ROUND74_EVENT_CLOCK_FEATURE_NAMES)
    ) == set(ROUND74_EVENT_FEATURE_NAMES)


def _task_group_losses(
    groups: dict[str, float],
    *,
    task_name: str = "net_payoff_quantiles",
) -> dict[str, float]:
    return {f"{key}:{task_name}": value for key, value in groups.items()}


def test_round74_feature_view_gate_is_complete_and_subgroup_noninferior() -> None:
    neutral = (1.0,) * 12
    full = (0.9,) * 12
    groups = {
        "run:BTCUSDT:1": 1.0,
        "run:ETHUSDT:5": 1.0,
        "run:SOLUSDT:30": 1.0,
    }
    promoted = _feature_view_promotion_report(
        "clock_neutral",
        "full",
        neutral,
        full,
        incumbent_group_losses=groups,
        challenger_group_losses={key: 0.9 for key in groups},
        incumbent_task_group_losses=_task_group_losses(groups),
        challenger_task_group_losses=_task_group_losses({key: 0.9 for key in groups}),
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=12,
    )
    incomplete = _feature_view_promotion_report(
        "clock_neutral",
        "full",
        neutral[:-1],
        full[:-1],
        incumbent_group_losses=groups,
        challenger_group_losses={key: 0.9 for key in groups},
        incumbent_task_group_losses=_task_group_losses(groups),
        challenger_task_group_losses=_task_group_losses({key: 0.9 for key in groups}),
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=12,
    )
    subgroup_degradation = _feature_view_promotion_report(
        "clock_neutral",
        "full",
        neutral,
        full,
        incumbent_group_losses=groups,
        challenger_group_losses={
            "run:BTCUSDT:1": 0.9,
            "run:ETHUSDT:5": 0.9,
            "run:SOLUSDT:30": 1.1,
        },
        incumbent_task_group_losses=_task_group_losses(groups),
        challenger_task_group_losses=_task_group_losses(
            {
                "run:BTCUSDT:1": 0.9,
                "run:ETHUSDT:5": 0.9,
                "run:SOLUSDT:30": 1.1,
            }
        ),
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=12,
    )

    assert promoted["incumbent_feature_view"] == "clock_neutral"
    assert promoted["challenger_feature_view"] == "full"
    assert promoted["promoted"] is True
    assert incomplete["complete_tuning_panel"] is False
    assert incomplete["promoted"] is False
    assert (
        subgroup_degradation["all_paired_run_symbol_horizon_groups_noninferior"]
        is False
    )
    assert subgroup_degradation["promoted"] is False
    assert (
        _order_flow_challenger_feature_view("market_state_clock_neutral")
        == "clock_neutral"
    )
    assert _order_flow_challenger_feature_view("market_state_with_clock") == "full"
    with pytest.raises(ValueError, match="clock-control winner"):
        _order_flow_challenger_feature_view("full")


def _interaction_fit(
    *,
    state_conditioned_flow: bool,
    losses: tuple[float, ...],
    group_losses: dict[str, float],
    feature_view: str = "clock_neutral",
) -> _CandidateFit:
    return _CandidateFit(
        candidate_id="causal_event_tcn",
        feature_view=feature_view,
        state_conditioned_flow=state_conditioned_flow,
        peer_states=({},),
        peer_reports=({},),
        ensemble_metrics={"loss": sum(losses) / len(losses)},
        ensemble_run_losses=losses,
        ensemble_group_losses=group_losses,
        ensemble_task_group_losses=_task_group_losses(group_losses),
        ensemble_prediction_sha256="a" * 64,
        parameter_count_per_peer=(130_114 if state_conditioned_flow else 129_060),
    )


def test_round74_state_conditioned_flow_requires_admitted_flow_and_full_gate() -> None:
    groups = {"run-a:BTCUSDT:5": 1.0, "run-b:ETHUSDT:30": 1.0}
    incumbent = _interaction_fit(
        state_conditioned_flow=False,
        losses=(1.0, 1.0),
        group_losses=groups,
    )
    challenger = _interaction_fit(
        state_conditioned_flow=True,
        losses=(0.9, 0.9),
        group_losses={key: 0.9 for key in groups},
    )

    winner, report = _select_state_conditioned_flow_with_ablation_gate(
        incumbent,
        challenger,
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=2,
    )

    assert winner is challenger
    assert report["promoted"] is True
    assert _feature_view_contains_order_flow("clock_neutral") is True
    assert _feature_view_contains_order_flow("full") is True
    assert _feature_view_contains_order_flow("market_state_clock_neutral") is False
    with pytest.raises(ValueError, match="ablation panel differs"):
        _select_state_conditioned_flow_with_ablation_gate(
            replace(incumbent, feature_view="market_state_clock_neutral"),
            replace(challenger, feature_view="market_state_clock_neutral"),
            minimum_mean_loss_improvement=1e-5,
            required_paired_run_count=2,
        )


def test_round74_feature_view_gate_rejects_non_broad_or_one_run_driven_gain() -> None:
    incumbent = (1.0,) * 12
    groups = {"run-a:BTCUSDT:5": 1.0, "run-b:ETHUSDT:30": 1.0}
    minority_material_wins = _feature_view_promotion_report(
        "clock_neutral",
        "full",
        incumbent,
        (0.9,) * 5 + (0.999999,) * 7,
        incumbent_group_losses=groups,
        challenger_group_losses={key: 0.9 for key in groups},
        incumbent_task_group_losses=_task_group_losses(groups),
        challenger_task_group_losses=_task_group_losses({key: 0.9 for key in groups}),
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=12,
    )
    one_run_driven = _feature_view_promotion_report(
        "clock_neutral",
        "full",
        incumbent,
        (0.0,) + (0.999989,) * 7 + (1.0,) * 4,
        incumbent_group_losses=groups,
        challenger_group_losses={key: 0.9 for key in groups},
        incumbent_task_group_losses=_task_group_losses(groups),
        challenger_task_group_losses=_task_group_losses({key: 0.9 for key in groups}),
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=12,
    )

    assert minority_material_wins["mean_proper_loss_improvement"] > 1e-5
    assert minority_material_wins["all_paired_runs_noninferior"] is True
    assert minority_material_wins["material_challenger_win_count"] == 5
    assert minority_material_wins["minimum_required_material_win_count"] == 7
    assert minority_material_wins["material_win_majority"] is False
    assert (
        minority_material_wins[
            "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement"
        ]
        is True
    )
    assert minority_material_wins["promoted"] is False
    assert one_run_driven["material_win_majority"] is True
    assert (
        one_run_driven["minimum_leave_one_capture_run_out_mean_proper_loss_improvement"]
        < 1e-5
    )
    assert (
        one_run_driven[
            "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement"
        ]
        is False
    )
    assert one_run_driven["promoted"] is False
    assert one_run_driven["statistical_independence_or_significance_claim"] is False


def test_round74_state_conditioned_flow_rejects_hidden_subgroup_degradation() -> None:
    incumbent = _interaction_fit(
        state_conditioned_flow=False,
        losses=(1.0, 1.0),
        group_losses={"run-a:BTCUSDT:5": 1.0, "run-b:ETHUSDT:30": 1.0},
    )
    challenger = _interaction_fit(
        state_conditioned_flow=True,
        losses=(0.9, 0.9),
        group_losses={"run-a:BTCUSDT:5": 0.8, "run-b:ETHUSDT:30": 1.1},
    )

    winner, report = _select_state_conditioned_flow_with_ablation_gate(
        incumbent,
        challenger,
        minimum_mean_loss_improvement=1e-5,
        required_paired_run_count=2,
    )

    assert winner is incumbent
    assert report["promoted"] is False
    assert report["all_paired_run_symbol_horizon_groups_noninferior"] is False


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


def test_round74_complexity_gate_rejects_hidden_forecast_task_degradation() -> None:
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
    task_names = (
        "net_payoff_quantiles",
        "maximum_adverse_excursion_quantiles",
        "positive_payoff",
        "adverse_selection",
        "regime_unpredictability",
    )
    baseline_groups = {key: 1.0 for key in group_keys}
    improved_groups = {key: 0.9 for key in group_keys}
    baseline_tasks = {
        f"{key}:{task_name}": 1.0 for key in group_keys for task_name in task_names
    }
    improved_tasks = {
        f"{key}:{task_name}": 0.8 for key in group_keys for task_name in task_names
    }
    hidden_failure = next(
        key
        for key in improved_tasks
        if ":SOLUSDT:" in key and key.endswith(":net_payoff_quantiles")
    )
    improved_tasks[hidden_failure] = 1.1

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        run_losses,
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
        candidate_group_losses={
            "event_pooling_linear": baseline_groups,
            "event_pooling_mlp": improved_groups,
            "causal_event_tcn": baseline_groups,
            "causal_event_attention": baseline_groups,
        },
        candidate_task_group_losses={
            "event_pooling_linear": baseline_tasks,
            "event_pooling_mlp": improved_tasks,
            "causal_event_tcn": baseline_tasks,
            "causal_event_attention": baseline_tasks,
        },
    )

    assert winner == "event_pooling_linear"
    first = reports[0]
    assert first["all_paired_runs_noninferior"] is True
    assert first["all_paired_run_symbol_horizon_groups_noninferior"] is True
    assert first["task_subgroup_gate_applied"] is True
    assert first["paired_run_symbol_horizon_task_count"] == len(baseline_tasks)
    assert first["worst_run_symbol_horizon_task"] == hidden_failure
    assert first["maximum_paired_task_group_loss_degradation"] == pytest.approx(0.1)
    assert first["all_paired_run_symbol_horizon_tasks_noninferior"] is False
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
        _assert_gradient_health(
            optimization,
            optimizer_steps=3,
            gradient_clip_norm=_config().gradient_clip_norm,
        )


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
    assert model.feature_view == artifact.selected_feature_view
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
    assert policy["selection_protocol"]["mode"] == "legacy_shared_tuning_panel"
    assert policy["selection_protocol"]["training_only_early_stopping"] is False
    assert policy["selection_protocol"]["disjoint_promotion_stages"] is False
    assert policy["selection_protocol"]["eligible_for_segmented_cohort_policy"] is False
    assert policy["backend"]["kind"] == "cpu"
    assert set(policy["candidate_panel"]) == {
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    }
    assert set(policy["feature_view_panel"]) == {
        "clock_features",
        "order_flow_features",
    }
    clock_stage = policy["feature_view_panel"]["clock_features"]
    order_flow_stage = policy["feature_view_panel"]["order_flow_features"]
    assert clock_stage["incumbent"]["feature_view"] == ("market_state_clock_neutral")
    assert clock_stage["challenger"]["feature_view"] == ("market_state_with_clock")
    assert order_flow_stage["incumbent"]["feature_view"] == (
        "market_state_clock_neutral"
    )
    assert order_flow_stage["challenger"]["feature_view"] == "clock_neutral"
    assert policy["feature_view_selection"]["supported_feature_views"] == list(
        ROUND74_EVENT_FEATURE_VIEWS
    )
    assert policy["feature_view_selection"]["evaluated_feature_views"] == [
        "market_state_clock_neutral",
        "market_state_with_clock",
        "clock_neutral",
    ]
    assert all(
        report["feature_view"] == "market_state_clock_neutral"
        for report in policy["candidate_panel"].values()
    )
    assert (
        policy["feature_view_selection"]["selected_feature_view"]
        == artifact.selected_feature_view
        == "market_state_clock_neutral"
    )
    assert (
        policy["feature_view_selection"]["order_flow_promotion_report"][
            "complete_tuning_panel"
        ]
        is False
    )
    assert (
        policy["feature_view_selection"]["clock_promotion_report"][
            "complete_tuning_panel"
        ]
        is False
    )
    assert (
        policy["feature_view_selection"]["state_first_incumbent"]
        == "market_state_clock_neutral"
    )
    assert (
        policy["feature_view_selection"]["clock_default_on_gate_failure"]
        == "market_state_clock_neutral"
    )
    assert (
        policy["feature_view_selection"]["order_flow_default_on_gate_failure"]
        == "market_state_clock_neutral"
    )
    assert set(policy["state_conditioned_flow_panel"]) == {"unconditioned_order_flow"}
    assert (
        policy["state_conditioned_flow_panel"]["unconditioned_order_flow"]
        == order_flow_stage["incumbent"]
    )
    interaction = policy["state_conditioned_flow_selection"]
    assert interaction["reason"] == "order_flow_layer_not_selected"
    assert interaction["evaluated"] is False
    assert interaction["selected_state_conditioned_flow"] is False
    assert interaction["promotion_report"] is None
    assert interaction["required_paired_capture_run_count"] == 0
    assert interaction["feature_view_fixed_before_interaction_gate"] is True
    assert interaction["order_flow_required"] is True
    assert interaction["neutral_multiplier_at_initialization"] == 1.0
    assert interaction["test_targets_used"] is False
    assert set(policy["initialization_panel"]) == {"random"}
    assert (
        policy["initialization_panel"]["random"]
        == policy["state_conditioned_flow_panel"]["unconditioned_order_flow"]
    )
    assert (
        policy["initialization_selection"]["reason"]
        == "preflight_never_evaluates_pretraining"
    )
    assert policy["initialization_selection"]["pretraining_evaluated"] is False
    assert policy["initialization_selection"]["selected_initialization_id"] == "random"
    assert policy["initialization_selection"]["promotion_report"] is None
    assert (
        policy["initialization_selection"]["supervised_targets_used_by_pretraining"]
        is False
    )
    assert (
        policy["initialization_selection"]["tuning_features_used_by_pretraining"]
        is False
    )
    assert (
        policy["initialization_selection"]["tuning_targets_used_by_pretraining"]
        is False
    )
    assert policy["initialization_selection"]["test_data_used_by_pretraining"] is False
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

    changed_selection_protocol = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
    changed_selection_protocol["selection_protocol"]["training_only_early_stopping"] = (
        True
    )
    changed_selection_protocol_path = _write_rehashed_policy(
        tmp_path,
        changed_selection_protocol,
    )
    with pytest.raises(ValueError, match="legacy selection protocol differs"):
        load_round74_pretest_policy(changed_selection_protocol_path)

    changed_promotion = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_promotion["selection"]["promotion_reports"][0]["promoted"] = True
    changed_promotion_path = _write_rehashed_policy(
        tmp_path,
        changed_promotion,
    )
    with pytest.raises(ValueError, match="selection or artifact differs"):
        load_round74_pretest_policy(changed_promotion_path)

    changed_feature_view = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_feature_view["feature_view_selection"]["selected_feature_view"] = "full"
    changed_feature_view_path = _write_rehashed_policy(
        tmp_path,
        changed_feature_view,
    )
    with pytest.raises(ValueError, match="feature-view selection differs"):
        load_round74_pretest_policy(changed_feature_view_path)

    changed_feature_membership = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
    changed_feature_membership["feature_view_selection"]["order_flow_feature_names"][
        0
    ] = "spread_bps"
    changed_feature_membership_path = _write_rehashed_policy(
        tmp_path,
        changed_feature_membership,
    )
    with pytest.raises(ValueError, match="feature-view selection differs"):
        load_round74_pretest_policy(changed_feature_membership_path)

    changed_initialization = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
    changed_initialization["initialization_selection"]["selected_initialization_id"] = (
        "causal_next_event_pretrained"
    )
    changed_initialization_path = _write_rehashed_policy(
        tmp_path,
        changed_initialization,
    )
    with pytest.raises(ValueError, match="initialization selection differs"):
        load_round74_pretest_policy(changed_initialization_path)

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
