from __future__ import annotations

from dataclasses import replace

import numpy as np

import simple_ai_trading.make_take_payoff_lightgbm as payoff_model_module
import simple_ai_trading.make_take_payoff_panel as payoff_panel_module
import simple_ai_trading.make_take_predictive_evaluation as evaluation_module
import simple_ai_trading.queue_fill_lightgbm as fill_model_module
import simple_ai_trading.queue_fill_survival as fill_panel_module
from simple_ai_trading.make_take_action_features import MakeTakeFeatureSpec
from simple_ai_trading.make_take_payoff_lightgbm import (
    MakeTakePayoffLightGBMSpec,
    MakeTakeConditionalPayoffPredictionBatch,
    train_make_take_payoff_lightgbm_model,
)
from simple_ai_trading.make_take_payoff_panel import (
    MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
    MAKE_TAKE_PAYOFF_SYMBOLS,
    MakeTakeConditionalPayoffPanel,
)
from simple_ai_trading.make_take_predictive_evaluation import (
    build_make_take_predictive_evaluation,
    validate_make_take_predictive_evaluation,
)
from simple_ai_trading.queue_fill_lightgbm import (
    QueueFillLightGBMSpec,
    QueueFillPredictionBatch,
    train_queue_fill_lightgbm_model,
)
from simple_ai_trading.queue_fill_survival import (
    PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
    PassiveFillSurvivalPanel,
    hazards_to_bucket_probabilities,
)


def _fill_panel(symbol: str, *, start_ms: int, source_sha: str):
    rows = 16
    event_index = np.repeat(np.arange(rows // 2, dtype=np.int64), 2)
    decisions = start_ms + event_index * 10_000
    sides = np.tile(np.asarray([1, -1], dtype=np.int8), rows // 2)
    buckets = np.tile(np.repeat(np.asarray([1, 2, 3, 0], dtype=np.uint8), 2), 2)
    provisional = PassiveFillSurvivalPanel(
        schema_version=PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
        symbol=symbol,
        feature_names=("feature_a",),
        source_action_feature_sha256=f"{start_ms + 200:064x}",
        source_entry_sha256=f"{start_ms + 201:064x}",
        source_dataset_sha256=source_sha,
        source_first_decision_time_ms=int(decisions[0]),
        source_last_decision_time_ms=int(decisions[-1]),
        event_index=event_index,
        decision_time_ms=decisions,
        action_side=sides,
        features=np.arange(rows, dtype=np.float32).reshape(rows, 1),
        fill_bucket=buckets,
        panel_sha256="",
    )
    return replace(
        provisional,
        panel_sha256=fill_panel_module._sha256(
            fill_panel_module._panel_payload(provisional)
        ),
    )


def _payoff_panel(symbol: str, *, start_ms: int, source_sha: str):
    events = 8
    rows = events * 4
    event_index = np.repeat(np.arange(events, dtype=np.int64), 4)
    decisions = start_ms + event_index * 10_000
    actions = np.tile(np.arange(4, dtype=np.uint8), events)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    state = np.repeat(np.linspace(-2.0, 2.0, events), 4)
    net = 3.0 * state + actions.astype(np.float64)
    provisional = MakeTakeConditionalPayoffPanel(
        schema_version=MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
        scenario="base",
        symbol=symbol,
        feature_names=("feature_a", "feature_b"),
        source_feature_spec_sha256=MakeTakeFeatureSpec().spec_sha256,
        source_action_feature_sha256=f"{start_ms + 300:064x}",
        source_entry_sha256=f"{start_ms + 301:064x}",
        source_target_sha256=f"{start_ms + 302:064x}",
        source_dataset_sha256=source_sha,
        source_first_decision_time_ms=int(decisions[0]),
        source_last_decision_time_ms=int(decisions[-1]),
        source_label_end_ms=int(decisions[-1]) + 315_750,
        event_index=event_index,
        decision_time_ms=decisions,
        action_code=actions,
        action_side=sides,
        features=np.column_stack((state, actions)).astype(np.float32),
        net_bps=net,
        markout_5s_bps=net / 2.0,
        markout_15s_bps=net / 3.0,
        terminal_time_ms=decisions + 300_000,
        stop_bps=np.full(rows, 40.0),
        take_bps=np.full(rows, 60.0),
        panel_sha256="",
    )
    return replace(
        provisional,
        panel_sha256=payoff_panel_module._sha256(
            payoff_panel_module._panel_payload(provisional)
        ),
    )


def _roles():
    source = {
        symbol: f"{index + 30:064x}"
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    }
    starts = (0, 500_000, 1_000_000, 1_500_000)
    fill_roles = tuple(
        tuple(
            _fill_panel(symbol, start_ms=start, source_sha=source[symbol])
            for symbol in MAKE_TAKE_PAYOFF_SYMBOLS
        )
        for start in starts
    )
    payoff_roles = tuple(
        tuple(
            _payoff_panel(symbol, start_ms=start, source_sha=source[symbol])
            for symbol in MAKE_TAKE_PAYOFF_SYMBOLS
        )
        for start in starts
    )
    return fill_roles, payoff_roles


class _FakeBooster:
    def __init__(
        self,
        model_str: str | None = None,
        *,
        columns: int = 1,
        value: float = 0.0,
    ) -> None:
        if model_str:
            _, raw_columns, raw_value = model_str.split(":")
            self.columns = int(raw_columns)
            self.value = float(raw_value)
        else:
            self.columns = columns
            self.value = value
        self.best_iteration = 3

    def predict(self, features, num_iteration=None):
        del num_iteration
        return np.full(len(features), self.value, dtype=np.float64)

    def model_to_string(self, num_iteration=None) -> str:
        del num_iteration
        return f"fake:{self.columns}:{self.value}"

    def num_feature(self) -> int:
        return self.columns


def _models(monkeypatch, fill_roles, payoff_roles):
    monkeypatch.setattr(
        fill_model_module,
        "lightgbm_backend_parameters",
        lambda *_args, **_kwargs: ({"device_type": "cpu"}, "cpu", "cpu"),
    )
    monkeypatch.setattr(
        payoff_model_module,
        "lightgbm_backend_parameters",
        lambda *_args, **_kwargs: ({"device_type": "cpu"}, "cpu", "cpu"),
    )
    monkeypatch.setattr(
        payoff_model_module.lgb,
        "train",
        lambda parameters, _dataset, **_kwargs: _FakeBooster(
            columns=1 if parameters["objective"] == "binary" else 2,
            value=(
                0.5
                if parameters["objective"] == "binary"
                else 0.0
                if parameters["objective"] == "regression"
                else -1.0
            ),
        ),
    )
    monkeypatch.setattr(fill_model_module.lgb, "Booster", _FakeBooster)
    fill_model = train_queue_fill_lightgbm_model(
        training_panels=fill_roles[0],
        early_stop_panels=fill_roles[1],
        calibration_panels=fill_roles[2],
        spec=QueueFillLightGBMSpec(
            min_data_in_leaf=2,
            num_boost_round=10,
            early_stopping_rounds=5,
            minimum_training_class_rows_per_symbol=2,
            minimum_early_stop_class_rows_per_symbol=2,
            minimum_calibration_class_rows_per_symbol=2,
        ),
        compute_backend="cpu",
    )
    payoff_model = train_make_take_payoff_lightgbm_model(
        training_panels=payoff_roles[0],
        early_stop_panels=payoff_roles[1],
        calibration_panels=payoff_roles[2],
        spec=MakeTakePayoffLightGBMSpec(
            min_data_in_leaf=2,
            num_boost_round=10,
            early_stopping_rounds=5,
            minimum_training_rows_per_action_symbol=2,
            minimum_early_stop_rows_per_action_symbol=2,
            minimum_calibration_rows_per_action_symbol=2,
        ),
        compute_backend="cpu",
    )
    quality = replace(payoff_model.early_quality, quality_gate_passed=True)
    provisional = replace(
        payoff_model,
        early_quality=quality,
        model_sha256="",
    )
    payoff_model = replace(
        provisional,
        model_sha256=payoff_model_module._model_sha256(provisional),
    )
    return fill_model, payoff_model


def _fill_prediction(
    panel: PassiveFillSurvivalPanel,
    *,
    model_sha256: str,
    perfect: bool,
):
    if perfect:
        hazards = np.zeros((panel.rows, 3), dtype=np.float64)
        for row, bucket in enumerate(panel.fill_bucket):
            if bucket > 0:
                hazards[row, int(bucket) - 1] = 1.0
    else:
        hazards = np.tile(np.asarray([0.25, 1.0 / 3.0, 0.5]), (panel.rows, 1))
    buckets = hazards_to_bucket_probabilities(hazards)
    return QueueFillPredictionBatch(
        source_action_feature_sha256=panel.source_action_feature_sha256,
        source_panel_sha256=panel.panel_sha256,
        model_sha256=model_sha256,
        symbol=panel.symbol,
        event_index=panel.event_index,
        decision_time_ms=panel.decision_time_ms,
        action_side=panel.action_side,
        hazard_probabilities=hazards,
        bucket_probabilities=buckets,
        fill_probability_15s=np.sum(buckets[:, :3], axis=1),
    )


def _payoff_prediction(
    panel: MakeTakeConditionalPayoffPanel,
    *,
    model_sha256: str,
):
    return MakeTakeConditionalPayoffPredictionBatch(
        source_panel_sha256=panel.panel_sha256,
        model_sha256=model_sha256,
        symbol=panel.symbol,
        action_code=panel.action_code,
        action_side=panel.action_side,
        conditional_mean_bps=panel.net_bps,
        conditional_q20_bps=panel.net_bps,
    )


def test_predictive_evaluation_passes_perfect_proper_scores_and_rank_skill(
    monkeypatch,
) -> None:
    fill_roles, payoff_roles = _roles()
    fill_model, payoff_model = _models(monkeypatch, fill_roles, payoff_roles)
    monkeypatch.setattr(
        evaluation_module,
        "predict_queue_fill_lightgbm_model",
        lambda model, panel: _fill_prediction(
            panel,
            model_sha256=model.model_sha256,
            perfect=True,
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "predict_make_take_conditional_payoff_panel",
        lambda model, panel: _payoff_prediction(
            panel,
            model_sha256=model.model_sha256,
        ),
    )
    report = build_make_take_predictive_evaluation(
        role="policy_calibration",
        fill_model=fill_model,
        payoff_model=payoff_model,
        training_fill_panels=fill_roles[0],
        evaluation_fill_panels=fill_roles[3],
        training_payoff_panels=payoff_roles[0],
        evaluation_payoff_panels=payoff_roles[3],
    )

    assert report.predictive_gate_passed is True
    assert all(metric.passed for metric in report.fill_metrics)
    assert all(metric.passed for metric in report.payoff_metrics)
    assert all(metric.spearman == 1.0 for metric in report.payoff_metrics)
    validate_make_take_predictive_evaluation(report)


def test_predictive_evaluation_rejects_zero_skill_fill_predictions(monkeypatch) -> None:
    fill_roles, payoff_roles = _roles()
    fill_model, payoff_model = _models(monkeypatch, fill_roles, payoff_roles)
    monkeypatch.setattr(
        evaluation_module,
        "predict_queue_fill_lightgbm_model",
        lambda model, panel: _fill_prediction(
            panel,
            model_sha256=model.model_sha256,
            perfect=False,
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "predict_make_take_conditional_payoff_panel",
        lambda model, panel: _payoff_prediction(
            panel,
            model_sha256=model.model_sha256,
        ),
    )
    report = build_make_take_predictive_evaluation(
        role="evaluation",
        fill_model=fill_model,
        payoff_model=payoff_model,
        training_fill_panels=fill_roles[0],
        evaluation_fill_panels=fill_roles[3],
        training_payoff_panels=payoff_roles[0],
        evaluation_payoff_panels=payoff_roles[3],
    )

    assert report.predictive_gate_passed is False
    assert all(abs(metric.log_loss_skill) < 1e-12 for metric in report.fill_metrics)
    assert all(metric.passed is False for metric in report.fill_metrics)
