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
from simple_ai_trading.impact_absorption_event_training import (  # noqa: E402
    ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT,
    ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS,
    ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION,
    Round74EventEnsemble,
    Round74EventTrainingConfig,
    _complexity_gated_candidate_id,
    load_round74_pretest_policy,
    train_and_seal_round74_pretest_policy,
)
from simple_ai_trading.impact_absorption_event_model import (  # noqa: E402
    Round74EventModelOutput,
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
    )


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
    mean_logit = (
        torch.logit(torch.tensor(0.5)) + torch.logit(torch.tensor(0.9))
    ) / 2.0
    assert not torch.allclose(torch.sigmoid(mean_logit), torch.tensor(0.7))


def test_round74_complexity_gate_requires_consistent_complete_panel() -> None:
    candidate_ids = (
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
    )
    parameter_counts = {
        "event_pooling_linear": 13_000,
        "event_pooling_mlp": 31_396,
        "causal_event_tcn": 129_060,
    }
    linear = (1.0,) * 24
    mlp = (0.9,) * 24
    tcn = (0.8,) * 24

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        {
            "event_pooling_linear": linear,
            "event_pooling_mlp": mlp,
            "causal_event_tcn": tcn,
        },
        parameter_counts,
        minimum_mean_loss_improvement=1e-5,
    )

    assert winner == "causal_event_tcn"
    assert len(reports) == ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT == 2
    assert reports[0]["incumbent_candidate_id"] == "event_pooling_linear"
    assert reports[0]["challenger_win_count"] == 24
    assert reports[0]["challenger_loss_count"] == 0
    assert reports[0]["complete_tuning_panel"] is True
    assert reports[0]["all_paired_runs_noninferior"] is True
    assert reports[0]["maximum_paired_run_loss_degradation"] < 0.0
    assert reports[0]["promoted"] is True
    assert reports[1]["incumbent_candidate_id"] == "event_pooling_mlp"
    assert reports[1]["challenger_win_count"] == 24
    assert reports[1]["promoted"] is True


def test_round74_complexity_gate_rejects_point_better_but_weak_challengers() -> None:
    candidate_ids = (
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
    )
    parameter_counts = {
        "event_pooling_linear": 13_000,
        "event_pooling_mlp": 31_396,
        "causal_event_tcn": 129_060,
    }
    linear = (1.0,) * 24
    weak = (0.8,) * 17 + (1.2,) * 7

    winner, reports = _complexity_gated_candidate_id(
        candidate_ids,
        {
            "event_pooling_linear": linear,
            "event_pooling_mlp": weak,
            "causal_event_tcn": weak,
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


def test_round74_complexity_gate_excludes_exact_ties() -> None:
    winner, reports = _complexity_gated_candidate_id(
        (
            "event_pooling_linear",
            "event_pooling_mlp",
            "causal_event_tcn",
        ),
        {
            "event_pooling_linear": (1.0,) * 24,
            "event_pooling_mlp": (0.9,) * 18 + (1.0,) * 6,
            "causal_event_tcn": (1.0,) * 24,
        },
        {
            "event_pooling_linear": 13_000,
            "event_pooling_mlp": 31_396,
            "causal_event_tcn": 129_060,
        },
        minimum_mean_loss_improvement=1e-5,
    )

    assert winner == "event_pooling_mlp"
    assert reports[0]["exact_tie_count"] == 6
    assert reports[0]["paired_capture_run_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    assert reports[0]["complete_tuning_panel"] is True
    assert reports[0]["all_paired_runs_noninferior"] is True
    assert reports[0]["promoted"] is True
    assert reports[1]["incumbent_candidate_id"] == "event_pooling_mlp"
    assert reports[1]["promoted"] is False


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
        optimization = report["peer_reports"][0]["history"][0][
            "optimization_metrics"
        ]
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
            "one eligible minibatch per training capture run with "
            "gradient accumulation"
        ),
        "gradient_divisor": "training_capture_run_count",
        "shorter_run_policy": (
            "deterministic epoch-rotated cycling of eligible minibatches"
        ),
        "fully_censored_minibatches_contribute_gradients": False,
        "fully_censored_capture_run_policy": "reject",
        "row_pooled_optimizer_steps_permitted": False,
    }
    for report in policy["candidate_panel"].values():
        optimization = report["peer_reports"][0]["history"][0][
            "optimization_metrics"
        ]
        assert optimization["run_count"] == 2.0
        assert optimization["optimizer_steps"] == 3.0
        assert optimization["run_contributions_per_optimizer_step"] == 2.0
        assert optimization["minimum_run_minibatch_contributions"] == 3.0
        assert optimization["maximum_run_minibatch_contributions"] == 3.0
        assert optimization["minimum_eligible_minibatches_per_run"] == 1.0
        assert optimization["maximum_eligible_minibatches_per_run"] == 3.0
        assert optimization["worst_run_loss"] >= (
            optimization["run_balanced_loss"]
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
    assert development["target_context_panel_schema_version"] == (
        ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
    )
    assert development["target_context_sha256"] == ["3" * 64, "4" * 64]
    assert development["target_context_count"] == 2
    assert len(development["target_context_panel_sha256"]) == 64
    assert policy["backend"]["kind"] == "cpu"
    assert set(policy["candidate_panel"]) == {
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
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
    ]
    assert policy["selection"]["planned_comparison_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
    )
    assert policy["selection"]["required_paired_capture_run_count"] == (
        ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    assert policy["selection"]["statistical_independence_or_significance_claim"] is False
    assert len(policy["selection"]["promotion_reports"]) == 2
    assert all(
        report["complete_tuning_panel"] is False
        and report["promoted"] is False
        for report in policy["selection"]["promotion_reports"]
    )
    for report in policy["candidate_panel"].values():
        assert len(report["ensemble_tuning_run_losses"]) == 2
    selected_metrics = policy["selection"]["selected_tuning_metrics"]
    assert selected_metrics["run_count"] == 2.0
    assert selected_metrics["worst_run_loss"] >= (
        selected_metrics["run_balanced_loss"]
    )
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

    changed_context_panel = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
    changed_context_panel["development_data"]["target_context_sha256"].append(
        "5" * 64
    )
    changed_context_path = _write_rehashed_policy(
        tmp_path,
        changed_context_panel,
    )
    with pytest.raises(ValueError, match="target-context panel differs"):
        load_round74_pretest_policy(changed_context_path)

    changed_promotion = json.loads(
        artifact.policy_path.read_text(encoding="ascii")
    )
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
