from __future__ import annotations

from simple_ai_trading.prequential_meta_label_model import (
    EVALUATION_MONTHS,
    PRIMARY_TARGET_MONTHS,
    frozen_meta_schedules,
    frozen_primary_schedules,
)


def test_primary_panel_schedules_are_strictly_prequential() -> None:
    schedules = frozen_primary_schedules()

    assert tuple(schedule.target_month for schedule in schedules) == (
        PRIMARY_TARGET_MONTHS
    )
    assert len(schedules) == 14
    first = schedules[0]
    assert first.training_start == "2022-07-01"
    assert first.training_end == "2023-09-30"
    assert first.early_stop_start == "2023-10-01"
    assert first.early_stop_end == "2023-10-31"
    assert first.prediction_start == "2023-11-01"
    assert first.prediction_end == "2023-11-30"
    last = schedules[-1]
    assert last.training_start == "2023-08-01"
    assert last.training_end == "2024-10-31"
    assert last.early_stop_start == "2024-11-01"
    assert last.early_stop_end == "2024-11-30"
    assert last.prediction_start == "2024-12-01"
    assert last.prediction_end == "2024-12-31"


def test_meta_schedules_use_six_oof_months_before_early_stop() -> None:
    schedules = frozen_meta_schedules()

    assert tuple(schedule.evaluation_month for schedule in schedules) == (
        EVALUATION_MONTHS
    )
    first = schedules[0]
    assert first.meta_fit_start == "2023-11-01"
    assert first.meta_fit_end == "2024-04-30"
    assert first.meta_early_stop_start == "2024-05-01"
    assert first.meta_early_stop_end == "2024-05-31"
    assert first.threshold_calibration_start == "2024-06-01"
    assert first.threshold_calibration_end == "2024-06-30"
    assert first.evaluation_start == "2024-07-01"
    assert first.evaluation_end == "2024-07-31"
    last = schedules[-1]
    assert last.meta_fit_start == "2024-04-01"
    assert last.meta_fit_end == "2024-09-30"
    assert last.meta_early_stop_start == "2024-10-01"
    assert last.meta_early_stop_end == "2024-10-31"
    assert last.threshold_calibration_start == "2024-11-01"
    assert last.threshold_calibration_end == "2024-11-30"
    assert last.evaluation_start == "2024-12-01"
    assert last.evaluation_end == "2024-12-31"
