"""Bounded probability calibration shared by research model families."""

from __future__ import annotations

import math

import numpy as np


def apply_platt_scaling(
    probabilities: np.ndarray,
    calibration: tuple[float, float],
) -> np.ndarray:
    """Apply a bounded affine transform in log-odds space."""

    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    slope, intercept = calibration
    if (
        not math.isfinite(float(slope))
        or not math.isfinite(float(intercept))
        or not 0.05 <= float(slope) <= 10.0
        or not -10.0 <= float(intercept) <= 10.0
    ):
        raise ValueError("Platt calibration parameters are invalid")
    scaled = np.clip(float(slope) * logits + float(intercept), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_platt_scaling(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Fit bounded Platt scaling with damped, loss-decreasing Newton steps."""

    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    outcomes = np.asarray(labels, dtype=np.float64)
    if (
        values.shape != outcomes.shape
        or values.ndim != 1
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(outcomes))
        or not np.all(np.isin(outcomes, (0.0, 1.0)))
    ):
        raise ValueError("probability calibration arrays are inconsistent")
    if min(int(np.sum(outcomes == 0.0)), int(np.sum(outcomes == 1.0))) < 2:
        raise ValueError("probability calibration requires both outcomes")
    logits = np.log(values / (1.0 - values))
    slope = 1.0
    base_rate = float(np.clip(np.mean(outcomes), 1e-6, 1.0 - 1e-6))
    intercept = float(
        np.clip(
            math.log(base_rate / (1.0 - base_rate)) - np.mean(logits),
            -10.0,
            10.0,
        )
    )
    regularization = 1e-3 * len(values)
    intercept_regularization = 0.01 * regularization

    def objective(candidate_slope: float, candidate_intercept: float) -> float:
        linear = candidate_slope * logits + candidate_intercept
        return float(
            np.sum(np.logaddexp(0.0, linear) - outcomes * linear)
            + 0.5 * regularization * (candidate_slope - 1.0) ** 2
            + 0.5 * intercept_regularization * candidate_intercept**2
        )

    for _ in range(100):
        linear = slope * logits + intercept
        fitted = np.exp(-np.logaddexp(0.0, -linear))
        residual = fitted - outcomes
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        gradient = np.asarray(
            [
                np.sum(residual * logits) + regularization * (slope - 1.0),
                np.sum(residual) + intercept_regularization * intercept,
            ],
            dtype=np.float64,
        )
        hessian = np.asarray(
            [
                [
                    np.sum(weights * logits * logits) + regularization,
                    np.sum(weights * logits),
                ],
                [
                    np.sum(weights * logits),
                    np.sum(weights) + intercept_regularization,
                ],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        current_objective = objective(slope, intercept)
        accepted_scale = 0.0
        for line_search_step in range(25):
            scale = 2.0 ** (-line_search_step)
            candidate_slope = float(np.clip(slope - scale * step[0], 0.05, 10.0))
            candidate_intercept = float(
                np.clip(intercept - scale * step[1], -10.0, 10.0)
            )
            if objective(candidate_slope, candidate_intercept) <= current_objective:
                slope = candidate_slope
                intercept = candidate_intercept
                accepted_scale = scale
                break
        if accepted_scale == 0.0 or float(
            np.max(np.abs(accepted_scale * step))
        ) < 1e-8:
            break
    return slope, intercept


__all__ = ["apply_platt_scaling", "fit_platt_scaling"]
