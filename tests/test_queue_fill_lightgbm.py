from __future__ import annotations

import numpy as np
import pytest

import simple_ai_trading.queue_fill_lightgbm as module
from simple_ai_trading.queue_fill_lightgbm import (
    QUEUE_FILL_SYMBOLS,
    QueueFillLightGBMSpec,
    load_queue_fill_lightgbm_model,
    predict_queue_fill_lightgbm_model,
    save_queue_fill_lightgbm_model,
    train_queue_fill_lightgbm_model,
)
from simple_ai_trading.queue_fill_survival import (
    PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
    PassiveFillSurvivalPanel,
)


class _FakeBooster:
    def __init__(self, model_str: str | None = None, *, columns: int = 3) -> None:
        self.columns = int(model_str.split(":")[1]) if model_str else columns
        self.best_iteration = 3

    def predict(self, features, num_iteration=None):
        del num_iteration
        return np.full(len(features), 0.5, dtype=np.float64)

    def model_to_string(self, num_iteration=None) -> str:
        del num_iteration
        return f"fake:{self.columns}"

    def num_feature(self) -> int:
        return self.columns


def _panel(symbol: str, *, start_ms: int, source_sha: str) -> PassiveFillSurvivalPanel:
    rows = 32
    fill_bucket = np.tile(np.asarray([1, 2, 3, 0], dtype=np.uint8), rows // 4)
    event_index = np.arange(rows, dtype=np.int64)
    decision_time = start_ms + np.repeat(np.arange(rows // 2), 2) * 10_000
    action_side = np.tile(np.asarray([1, -1], dtype=np.int8), rows // 2)
    features = np.column_stack(
        (
            np.linspace(-1.0, 1.0, rows),
            action_side,
            fill_bucket,
        )
    ).astype(np.float32)
    return PassiveFillSurvivalPanel(
        schema_version=PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
        symbol=symbol,
        feature_names=("feature_a", "feature_b", "feature_c"),
        source_action_feature_sha256="1" * 64,
        source_entry_sha256="2" * 64,
        source_dataset_sha256=source_sha,
        source_first_decision_time_ms=int(decision_time[0]),
        source_last_decision_time_ms=int(decision_time[-1]),
        event_index=event_index,
        decision_time_ms=decision_time,
        action_side=action_side,
        features=features,
        fill_bucket=fill_bucket,
        panel_sha256=("3" if start_ms == 0 else "4" if start_ms < 2_000_000 else "5")
        * 64,
    )


def _roles():
    source = {
        symbol: f"{index + 10:064x}" for index, symbol in enumerate(QUEUE_FILL_SYMBOLS)
    }
    return tuple(
        [
            _panel(symbol, start_ms=start, source_sha=source[symbol])
            for symbol in QUEUE_FILL_SYMBOLS
        ]
        for start in (0, 1_000_000, 2_000_000)
    )


def test_pooled_hazard_training_prediction_and_exact_reload(monkeypatch, tmp_path) -> None:
    training, early, calibration = _roles()
    monkeypatch.setattr(
        module,
        "lightgbm_backend_parameters",
        lambda *_args, **_kwargs: ({"device_type": "cpu"}, "cpu", "cpu"),
    )
    monkeypatch.setattr(
        module.lgb,
        "train",
        lambda _parameters, _dataset, **_kwargs: _FakeBooster(columns=3),
    )
    monkeypatch.setattr(module.lgb, "Booster", _FakeBooster)
    spec = QueueFillLightGBMSpec(
        min_data_in_leaf=2,
        num_boost_round=10,
        early_stopping_rounds=5,
        minimum_training_class_rows_per_symbol=2,
        minimum_early_stop_class_rows_per_symbol=2,
        minimum_calibration_class_rows_per_symbol=2,
    )

    model = train_queue_fill_lightgbm_model(
        training_panels=training,
        early_stop_panels=early,
        calibration_panels=calibration,
        spec=spec,
        compute_backend="cpu",
    )
    prediction = predict_queue_fill_lightgbm_model(model, calibration[0])
    artifact = tmp_path / "queue-fill.json"
    save_queue_fill_lightgbm_model(artifact, model)
    loaded = load_queue_fill_lightgbm_model(artifact)

    assert model.model_sha256 == loaded.model_sha256
    assert model.best_iterations == (3, 3, 3)
    assert prediction.hazard_probabilities.shape == (calibration[0].rows, 3)
    np.testing.assert_allclose(np.sum(prediction.bucket_probabilities, axis=1), 1.0)
    assert np.all((prediction.fill_probability_15s > 0.0) & (prediction.fill_probability_15s < 1.0))


@pytest.mark.parametrize(
    ("field", "value"),
    (("num_leaves", 31.5), ("num_boost_round", True), ("learning_rate", True)),
)
def test_spec_rejects_lossy_or_boolean_scalars(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="specification is invalid"):
        QueueFillLightGBMSpec(**{field: value})


def test_loader_rejects_malformed_json_with_controlled_error(tmp_path) -> None:
    artifact = tmp_path / "queue-fill.json"
    artifact.write_text('{"model_sha256": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact is unreadable"):
        load_queue_fill_lightgbm_model(artifact)
