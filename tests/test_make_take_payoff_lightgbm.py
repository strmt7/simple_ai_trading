from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import simple_ai_trading.make_take_payoff_lightgbm as module
import simple_ai_trading.make_take_payoff_panel as panel_module
from simple_ai_trading.make_take_action_features import (
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MakeTakeActionFeatureBatch,
    MakeTakeFeatureSpec,
)
from simple_ai_trading.make_take_payoff_lightgbm import (
    MAKE_TAKE_PAYOFF_SEEDS,
    MakeTakePayoffLightGBMSpec,
    load_make_take_payoff_lightgbm_model,
    predict_make_take_payoff_lightgbm_model,
    save_make_take_payoff_lightgbm_model,
    train_make_take_payoff_lightgbm_model,
)
from simple_ai_trading.make_take_payoff_panel import (
    MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
    MAKE_TAKE_PAYOFF_SYMBOLS,
    MakeTakeConditionalPayoffPanel,
)


class _FakeBooster:
    def __init__(
        self,
        model_str: str | None = None,
        *,
        columns: int = 2,
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


def _panel(
    symbol: str,
    *,
    start_ms: int,
    source_sha: str,
) -> MakeTakeConditionalPayoffPanel:
    events = 8
    rows = events * 4
    event_index = np.repeat(np.arange(events, dtype=np.int64), 4)
    decision_time = np.repeat(
        start_ms + np.arange(events, dtype=np.int64) * 10_000,
        4,
    )
    action_code = np.tile(np.arange(4, dtype=np.uint8), events)
    action_side = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    trend = np.repeat(np.linspace(-1.0, 1.0, events), 4)
    features = np.column_stack((trend, action_code)).astype(np.float32)
    net_bps = 4.0 * trend + action_code.astype(np.float64) - 1.5
    markout_5s = np.where(event_index % 2 == 0, -2.0, 2.0)
    markout_15s = np.where(event_index % 3 == 0, -3.0, 3.0)
    source_last = int(decision_time[-1])
    provisional = MakeTakeConditionalPayoffPanel(
        schema_version=MAKE_TAKE_PAYOFF_PANEL_SCHEMA_VERSION,
        scenario="base",
        symbol=symbol,
        feature_names=("feature_a", "feature_b"),
        source_feature_spec_sha256=MakeTakeFeatureSpec().spec_sha256,
        source_action_feature_sha256=f"{start_ms + 100:064x}",
        source_entry_sha256=f"{start_ms + 101:064x}",
        source_target_sha256=f"{start_ms + 102:064x}",
        source_dataset_sha256=source_sha,
        source_first_decision_time_ms=int(decision_time[0]),
        source_last_decision_time_ms=source_last,
        source_label_end_ms=source_last + 315_750,
        event_index=event_index,
        decision_time_ms=decision_time,
        action_code=action_code,
        action_side=action_side,
        features=features,
        net_bps=net_bps,
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        terminal_time_ms=decision_time + 300_000,
        stop_bps=np.full(rows, 40.0),
        take_bps=np.full(rows, 60.0),
        panel_sha256="",
    )
    return replace(
        provisional,
        panel_sha256=panel_module._sha256(panel_module._panel_payload(provisional)),
    )


def _roles(*, starts: tuple[int, int, int] = (0, 500_000, 1_000_000)):
    source = {
        symbol: f"{index + 20:064x}"
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    }
    return tuple(
        tuple(
            _panel(symbol, start_ms=start, source_sha=source[symbol])
            for symbol in MAKE_TAKE_PAYOFF_SYMBOLS
        )
        for start in starts
    )


def _action_features(panel: MakeTakeConditionalPayoffPanel) -> MakeTakeActionFeatureBatch:
    spec = MakeTakeFeatureSpec()
    events = panel.rows // 4
    return MakeTakeActionFeatureBatch(
        schema_version=MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
        spec=spec,
        spec_sha256=spec.spec_sha256,
        source_dataset_sha256=panel.source_dataset_sha256,
        source_flow_sha256="a" * 64,
        feature_names=panel.feature_names,
        event_indexes=np.arange(events, dtype=np.int64),
        decision_time_ms=panel.decision_time_ms[::4],
        action_code=panel.action_code,
        action_side=panel.action_side,
        eligible=np.ones(panel.rows, dtype=np.bool_),
        features=panel.features,
        batch_sha256="b" * 64,
    )


def _test_spec() -> MakeTakePayoffLightGBMSpec:
    return MakeTakePayoffLightGBMSpec(
        min_data_in_leaf=2,
        num_boost_round=10,
        early_stopping_rounds=5,
        minimum_training_rows_per_action_symbol=2,
        minimum_early_stop_rows_per_action_symbol=2,
        minimum_calibration_rows_per_action_symbol=2,
    )


def test_shared_payoff_training_prediction_and_exact_reload(monkeypatch, tmp_path) -> None:
    training, early, calibration = _roles()
    monkeypatch.setattr(
        module,
        "lightgbm_backend_parameters",
        lambda *_args, **_kwargs: ({"device_type": "cpu"}, "cpu", "cpu"),
    )
    monkeypatch.setattr(
        module.lgb,
        "train",
        lambda parameters, _dataset, **_kwargs: _FakeBooster(
            columns=2,
            value=0.0 if parameters["objective"] == "regression" else -1.0,
        ),
    )
    monkeypatch.setattr(module.lgb, "Booster", _FakeBooster)

    model = train_make_take_payoff_lightgbm_model(
        training_panels=training,
        early_stop_panels=early,
        calibration_panels=calibration,
        spec=_test_spec(),
        compute_backend="cpu",
        seed=MAKE_TAKE_PAYOFF_SEEDS[0],
    )
    prediction = predict_make_take_payoff_lightgbm_model(
        model,
        symbol=calibration[0].symbol,
        action_features=_action_features(calibration[0]),
    )
    artifact = tmp_path / "make-take-payoff.json"
    save_make_take_payoff_lightgbm_model(artifact, model)
    loaded = load_make_take_payoff_lightgbm_model(artifact)

    assert loaded.model_sha256 == model.model_sha256
    assert model.best_iterations == (3, 3)
    assert prediction.rows == calibration[0].rows
    assert np.all(prediction.conditional_q20_bps <= prediction.conditional_mean_bps)
    assert len(model.q20_calibration_coverage) == 4
    assert model.early_quality.quality_gate_passed is False

    drifted_side = prediction.action_side.copy()
    drifted_side[0] = -1
    drifted_features = replace(
        _action_features(calibration[0]),
        action_side=drifted_side,
    )
    with pytest.raises(ValueError, match="feature contract drifted"):
        predict_make_take_payoff_lightgbm_model(
            model,
            symbol=calibration[0].symbol,
            action_features=drifted_features,
        )


def test_shared_payoff_training_rejects_label_horizon_overlap(monkeypatch) -> None:
    training, early, calibration = _roles(starts=(0, 350_000, 1_000_000))
    monkeypatch.setattr(
        module,
        "lightgbm_backend_parameters",
        lambda *_args, **_kwargs: ({"device_type": "cpu"}, "cpu", "cpu"),
    )

    with pytest.raises(ValueError, match="chronological roles overlap"):
        train_make_take_payoff_lightgbm_model(
            training_panels=training,
            early_stop_panels=early,
            calibration_panels=calibration,
            spec=_test_spec(),
            compute_backend="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("num_leaves", 31.5), ("num_boost_round", True), ("lower_quantile", True)),
)
def test_payoff_spec_rejects_lossy_or_boolean_scalars(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="specification is invalid"):
        MakeTakePayoffLightGBMSpec(**{field: value})
