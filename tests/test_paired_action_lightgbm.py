from __future__ import annotations

from pathlib import Path

import numpy as np

from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, SYMBOLS
from simple_ai_trading.paired_action_lightgbm import (
    ACTION_NAMES,
    CENTERED_SIGNED_FEATURES,
    SEMIVOLATILITY_FEATURES,
    SIGNED_FEATURES,
    OuterFold,
    PairedActionPanel,
    PairedActionSpec,
    action_conditioned_feature_names,
    apply_paired_action_calibration,
    build_monthly_outer_folds,
    build_paired_action_panel,
    embargoed_interval_mask,
    fit_paired_action_calibration,
    load_paired_action_calibration,
    load_paired_action_model,
    paired_action_decisions,
    predict_paired_action_model,
    save_paired_action_calibration,
    save_paired_action_model,
    train_paired_action_model,
)


HOUR_MS = 60 * MINUTE_MS
GATE_FEATURES = (
    "target_same_minute_of_week_liquidity_ratio",
    "target_quote_volume_vs_1440m_mean",
    "target_realized_volatility_60m_bps",
    "target_realized_volatility_1440m_bps",
    "invariant_signal",
)


def _source_feature_names() -> tuple[str, ...]:
    names = [*SIGNED_FEATURES, *CENTERED_SIGNED_FEATURES]
    names.extend(name for pair in SEMIVOLATILITY_FEATURES.values() for name in pair)
    names.extend(GATE_FEATURES)
    return tuple(dict.fromkeys(names))


def _source_panel(timestamps: int = 1_100) -> PairedActionPanel:
    rng = np.random.default_rng(56)
    names = _source_feature_names()
    positions = {name: index for index, name in enumerate(names)}
    values = rng.normal(size=(timestamps, len(SYMBOLS), len(names))).astype(np.float32)
    values[..., positions["target_close_location"]] = 0.75
    values[..., positions["target_same_minute_of_week_liquidity_ratio"]] = 1.0
    values[..., positions["target_quote_volume_vs_1440m_mean"]] = 1.0
    values[..., positions["target_realized_volatility_60m_bps"]] = 5.0
    values[..., positions["target_realized_volatility_1440m_bps"]] = 6.0
    for horizon, (downside, upside) in SEMIVOLATILITY_FEATURES.items():
        values[..., positions[downside]] = 4.0 + horizon / 1_000.0
        values[..., positions[upside]] = 7.0 + horizon / 1_000.0
    timestamps_ms = np.arange(timestamps, dtype=np.int64) * 2 * HOUR_MS
    stop = np.full((timestamps, len(SYMBOLS)), 100.0, dtype=np.float32)
    signal = values[..., positions["target_return_60m_bps"]].astype(np.float64)
    noise = rng.normal(scale=0.5, size=signal.shape)
    long_target = 15.0 * signal + noise
    short_target = -15.0 * signal + noise
    exits = np.broadcast_to(
        timestamps_ms[:, None] + 61 * MINUTE_MS,
        stop.shape,
    ).copy()
    return build_paired_action_panel(
        features=values,
        feature_names=names,
        timestamps_ms=timestamps_ms,
        stop_bps=stop,
        long_target_bps=long_target,
        short_target_bps=short_target,
        long_exit_time_ms=exits,
        short_exit_time_ms=exits,
    )


def test_paired_action_transform_is_exact_and_symmetric() -> None:
    panel = _source_panel(20)
    source_names = panel.source_feature_names
    output_names = action_conditioned_feature_names(source_names)
    source_positions = {name: index for index, name in enumerate(source_names)}
    output_positions = {name: index for index, name in enumerate(output_names)}

    raw_return = "target_return_60m_bps"
    aligned_return = f"action_aligned_{raw_return}"
    original = panel.features[..., output_positions[aligned_return]]
    np.testing.assert_allclose(original[..., 0], -original[..., 1])
    np.testing.assert_allclose(
        panel.features[..., output_positions["action_aligned_target_close_location"]],
        np.broadcast_to(np.asarray((0.25, -0.25)), panel.target_bps.shape),
    )
    favorable = panel.features[
        ..., output_positions["action_favorable_semivolatility_60m_bps"]
    ]
    adverse = panel.features[
        ..., output_positions["action_adverse_semivolatility_60m_bps"]
    ]
    assert np.all(favorable[..., 0] > adverse[..., 0])
    assert np.all(favorable[..., 1] < adverse[..., 1])
    invariant_position = output_positions["invariant_signal"]
    np.testing.assert_array_equal(
        panel.features[..., 0, invariant_position],
        panel.features[..., 1, invariant_position],
    )
    np.testing.assert_allclose(
        panel.features[..., -1],
        np.broadcast_to(np.asarray((1.0, -1.0)), panel.target_bps.shape),
    )
    assert len(output_names) == len(source_names) + 1
    assert source_positions[raw_return] != source_positions["target_close_location"]


def test_monthly_folds_and_interval_masks_embargo_label_paths() -> None:
    start = np.datetime64("2022-01-01T00:00")
    end = np.datetime64("2024-07-01T00:00")
    timestamps_ms = (
        np.arange(start, end, np.timedelta64(1, "h")).astype("datetime64[ms]").astype(np.int64)
    )
    shape = (timestamps_ms.size, len(SYMBOLS), len(ACTION_NAMES))
    panel = PairedActionPanel(
        timestamps_ms=timestamps_ms,
        source_feature_names=("source",),
        feature_names=("action_sign",),
        features=np.ones((*shape, 1), dtype=np.float32),
        stop_bps=np.full(shape, 100.0, dtype=np.float32),
        target_bps=np.zeros(shape, dtype=np.float32),
        exit_time_ms=np.broadcast_to(
            timestamps_ms[:, None, None] + 61 * MINUTE_MS, shape
        ).copy(),
    )
    folds = build_monthly_outer_folds(panel)
    assert len(folds) == 12
    assert folds[0].fold_id == "2023-07"
    assert folds[-1].fold_id == "2024-06"
    assert np.count_nonzero(folds[-1].training_mask) > np.count_nonzero(
        folds[0].training_mask
    )
    calibration_validation = embargoed_interval_mask(
        panel,
        start="2024-01-01",
        end="2024-07-01",
    )
    final_timestamp = np.max(panel.timestamps_ms[calibration_validation])
    assert final_timestamp == int(np.datetime64("2024-06-30T22:00", "ms").astype(np.int64))


def _fold(
    panel: PairedActionPanel,
    fold_id: str,
    train: slice,
    early: slice,
    outer: slice,
) -> OuterFold:
    masks = []
    for selected in (train, early, outer):
        mask = np.zeros(panel.timestamps, dtype=bool)
        mask[selected] = True
        masks.append(mask)
    return OuterFold(
        fold_id=fold_id,
        training_mask=masks[0],
        early_stopping_mask=masks[1],
        outer_mask=masks[2],
        training_start_ms=int(panel.timestamps_ms[train.start or 0]),
        early_stopping_start_ms=int(panel.timestamps_ms[early.start or 0]),
        outer_start_ms=int(panel.timestamps_ms[outer.start or 0]),
        outer_end_ms=int(panel.timestamps_ms[(outer.stop or panel.timestamps) - 1] + 2 * HOUR_MS),
    )


def test_paired_action_model_rolls_refits_and_reloads(tmp_path: Path) -> None:
    panel = _source_panel()
    folds = (
        _fold(panel, "fold-1", slice(0, 400), slice(400, 550), slice(550, 700)),
        _fold(panel, "fold-2", slice(0, 550), slice(550, 700), slice(700, 850)),
    )
    final = np.zeros(panel.timestamps, dtype=bool)
    final[:900] = True
    probe = np.zeros(panel.timestamps, dtype=bool)
    probe[900:] = True
    signal_index = panel.feature_names.index(
        "action_aligned_target_return_60m_bps"
    )
    factor_values = np.square(panel.features[..., signal_index : signal_index + 1])
    result = train_paired_action_model(
        treatment_id="ai_program_augmented",
        view_id="raw_stress_payoff",
        objective_id="conditional_mean",
        seed=5601,
        panel=panel,
        outer_folds=folds,
        final_refit_mask=final,
        prediction_probe_mask=probe,
        factor_values=factor_values,
        factor_feature_names=("ai_test_aligned_signal_square",),
        source_dataset_sha256="a" * 64,
        payoff_dataset_sha256="b" * 64,
        design_sha256="c" * 64,
        spec=PairedActionSpec(
            min_data_in_leaf=32,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            maximum_boosting_rounds=25,
            early_stopping_rounds=5,
        ),
        compute_backend="cpu",
    )
    assert np.count_nonzero(result.oof_mask) == 300 * len(SYMBOLS) * 2
    path = tmp_path / "paired-model.json"
    save_paired_action_model(path, result.model)
    loaded = load_paired_action_model(path)
    direct = predict_paired_action_model(result.model, panel, factor_values)
    restored = predict_paired_action_model(loaded, panel, factor_values)
    np.testing.assert_allclose(direct, restored, rtol=0.0, atol=1e-12)


def test_calibration_reloads_and_decisions_apply_all_gates(tmp_path: Path) -> None:
    rng = np.random.default_rng(5600)
    timestamps = 200
    truth = np.empty((timestamps, len(SYMBOLS), 2), dtype=np.float64)
    truth[..., 0] = 6.0 + rng.normal(scale=0.5, size=truth.shape[:2])
    truth[..., 1] = -6.0 + rng.normal(scale=0.5, size=truth.shape[:2])
    point = np.empty((2, 3, *truth.shape), dtype=np.float64)
    q20 = np.empty_like(point)
    for view in range(2):
        for seed in range(3):
            point[view, seed] = truth + rng.normal(scale=0.4, size=truth.shape)
            q20[view, seed] = truth - 1.0 + rng.normal(scale=0.2, size=truth.shape)
    timestamps_ms = np.arange(timestamps, dtype=np.int64) * HOUR_MS
    fit_mask = np.zeros(timestamps, dtype=bool)
    fit_mask[:120] = True
    calibration = fit_paired_action_calibration(
        treatment_id="baseline_72",
        point_predictions_bps=point,
        q20_predictions_bps=q20,
        truth_bps=truth,
        timestamp_mask=fit_mask,
        timestamps_ms=timestamps_ms,
    )
    path = tmp_path / "calibration.json"
    save_paired_action_calibration(path, calibration)
    loaded = load_paired_action_calibration(path)
    calibrated_point, calibrated_q20 = apply_paired_action_calibration(
        loaded, point, q20
    )
    assert calibrated_point.shape == (2, *truth.shape)
    assert calibrated_q20.shape == calibrated_point.shape

    source_names = GATE_FEATURES[:4]
    source = np.ones((timestamps, len(SYMBOLS), len(source_names)), dtype=np.float64)
    positions = {name: index for index, name in enumerate(source_names)}
    source[..., positions["target_realized_volatility_60m_bps"]] = 5.0
    source[..., positions["target_realized_volatility_1440m_bps"]] = 6.0
    source[0, 0, positions["target_same_minute_of_week_liquidity_ratio"]] = 0.1
    decisions = paired_action_decisions(
        point_predictions_bps=point,
        q20_predictions_bps=q20,
        calibration=loaded,
        source_features=source,
        source_feature_names=source_names,
        stop_bps=np.full(truth.shape[:2], 100.0),
    )
    assert decisions.actions[0, 0] == 0
    assert np.all(decisions.actions[1:] == 1)
    assert np.all((decisions.size_multiplier >= 0.0) & (decisions.size_multiplier <= 1.0))
