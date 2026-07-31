from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading.polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21DevelopmentPanel,
    fit_round21_development,
    predict_round21_candidate,
    validate_round21_development_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _panel(
    role: str,
    *,
    first_condition: int,
    condition_count: int,
) -> Round21DevelopmentPanel:
    condition_numbers = np.arange(
        first_condition,
        first_condition + condition_count,
        dtype=np.int64,
    )
    labels = (condition_numbers % 2).astype(np.float64)
    event_start = 1_800_000_000_000 + condition_numbers * 300_000
    condition_ids = np.asarray(
        ["0x" + format(int(value), "064x") for value in condition_numbers],
        dtype=object,
    )
    signed = labels * 2.0 - 1.0
    core = np.column_stack(
        (
            np.sin(condition_numbers * 0.13),
            np.cos(condition_numbers * 0.07),
            (condition_numbers % 3).astype(np.float64) - 1.0,
        )
    ).astype(np.float32)
    spot_available = condition_numbers % 7 != 0
    usdm_available = spot_available & (condition_numbers % 5 != 0)
    spot = np.column_stack(
        (
            signed * 1.6,
            signed * 0.9 + np.sin(condition_numbers * 0.03),
        )
    ).astype(np.float32)
    usdm = np.column_stack(
        (
            signed * 1.2,
            signed * 0.7 + np.cos(condition_numbers * 0.05),
        )
    ).astype(np.float32)
    spot[~spot_available] = 0.0
    usdm[~usdm_available] = 0.0
    return Round21DevelopmentPanel(
        role=role,
        condition_ids=condition_ids,
        event_start_ms=event_start,
        decision_time_ms=event_start + 150_000,
        labels=labels,
        structural_probability=(
            0.5 + 0.02 * np.sin(condition_numbers * 0.17)
        ).astype(np.float64),
        market_prior_probability=(
            0.5 + 0.015 * np.cos(condition_numbers * 0.11)
        ).astype(np.float64),
        core_features=core,
        spot_features=spot,
        usdm_features=usdm,
        spot_available=spot_available,
        usdm_available=usdm_available,
        core_feature_names_sha256=_sha("core-v1"),
        spot_feature_names_sha256=_sha("spot-v1"),
        usdm_feature_names_sha256=_sha("usdm-v1"),
        dataset_sha256=_sha(f"dataset-{role}"),
        target_manifest_sha256=_sha(f"targets-{role}"),
        dataset_design_sha256=POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round21_panel_requires_explicit_zero_missingness() -> None:
    panel = _panel("train", first_condition=0, condition_count=80)
    spot = panel.spot_features.copy()
    spot[~panel.spot_available, 0] = 1.0

    with pytest.raises(ValueError, match="panel is invalid"):
        replace(panel, spot_features=spot).validate()


def test_round21_panel_rejects_usdm_without_spot() -> None:
    panel = _panel("train", first_condition=0, condition_count=80)
    usdm_available = panel.usdm_available.copy()
    unavailable = int(np.flatnonzero(~panel.spot_available)[0])
    usdm_available[unavailable] = True

    with pytest.raises(ValueError, match="panel is invalid"):
        replace(panel, usdm_available=usdm_available).validate()


def test_round21_panel_rejects_probability_clipping_or_hash_case_drift() -> None:
    panel = _panel("train", first_condition=0, condition_count=80)
    structural = panel.structural_probability.copy()
    structural[0] = 1.01

    with pytest.raises(ValueError, match="panel is invalid"):
        replace(panel, structural_probability=structural).validate()
    with pytest.raises(ValueError, match="panel is invalid"):
        replace(
            panel,
            core_feature_names_sha256=panel.core_feature_names_sha256.upper(),
        ).validate()


def test_round21_fits_core_and_exact_matched_optional_challengers() -> None:
    train = _panel("train", first_condition=0, condition_count=100)
    calibration = _panel(
        "tune_calibration",
        first_condition=100,
        condition_count=120,
    )
    selection = _panel(
        "tune_selection",
        first_condition=220,
        condition_count=80,
    )

    artifact = fit_round21_development(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        compute_backend="cpu",
    )

    assert artifact["economic_evaluation_completed"] is False
    assert (
        artifact["dataset_design_sha256"]
        == POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
    )
    assert artifact["test_features_accessed"] is False
    assert artifact["test_targets_accessed"] is False
    assert artifact["model_selected"] is False
    assert artifact["profitability_claim"] is False
    assert artifact["live_trading_authority"] is False
    layers = artifact["layers"]
    assert set(layers) == {"core", "core_spot", "core_spot_usdm"}
    assert all(len(layer["candidate_ledger"]) == 5 for layer in layers.values())
    assert layers["core_spot"]["comparison"]["matched_decision_count"] == int(
        np.count_nonzero(selection.spot_available)
    )
    assert layers["core_spot_usdm"]["comparison"][
        "matched_decision_count"
    ] == int(np.count_nonzero(selection.usdm_available))
    assert layers["core_spot"]["comparison"]["predictive_development_accepted"]
    assert layers["core_spot_usdm"]["comparison"][
        "predictive_development_accepted"
    ]

    core_record = next(
        record
        for record in layers["core"]["candidate_ledger"]
        if record["candidate_id"] == layers["core"]["selected_candidate_id"]
    )
    indices, predictions = predict_round21_candidate(
        core_record["model"],
        selection,
    )
    assert np.array_equal(indices, np.arange(len(selection.labels)))
    assert predictions.shape == selection.labels.shape
    assert np.all((predictions > 0.0) & (predictions < 1.0))


def test_round21_artifact_rejects_rehashed_authority_drift() -> None:
    train = _panel("train", first_condition=0, condition_count=100)
    calibration = _panel(
        "tune_calibration",
        first_condition=100,
        condition_count=120,
    )
    selection = _panel(
        "tune_selection",
        first_condition=220,
        condition_count=80,
    )
    artifact = fit_round21_development(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        compute_backend="cpu",
    )
    artifact["live_trading_authority"] = True

    with pytest.raises(ValueError, match="artifact differs"):
        validate_round21_development_artifact(_rehash(artifact))
