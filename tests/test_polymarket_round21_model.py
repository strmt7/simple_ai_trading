from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import simple_ai_trading.polymarket_round21_model as model_module
from simple_ai_trading.polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
    POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256,
    Round21DevelopmentPanel,
    Round21InferencePanel,
    fit_round21_development,
    predict_round21_candidate,
    predict_round21_probability_batch,
    validate_round21_development_artifact,
)

MODEL_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-matched-model-design-v2.json"
)
PROBABILITY_ENVELOPE_DESIGN_PATH = (
    MODEL_DESIGN_PATH.parent / "round-021-probability-envelope-design-v1.json"
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
        structural_probability=(0.5 + 0.02 * np.sin(condition_numbers * 0.17)).astype(
            np.float64
        ),
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


def test_round21_model_design_is_canonical_and_target_blind() -> None:
    design = json.loads(MODEL_DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")
    actual = hashlib.sha256(
        json.dumps(
            design,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert claimed == actual == POLYMARKET_ROUND21_MODEL_DESIGN_SHA256
    assert design["optional_binance_comparison"][
        "matched_core_is_refit_on_each_optional_population"
    ]
    assert not any(design["authority"].values())


def test_round21_probability_envelope_design_makes_no_coverage_claim() -> None:
    design = json.loads(PROBABILITY_ENVELOPE_DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")
    actual = hashlib.sha256(
        json.dumps(
            design,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert claimed == actual == POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256
    assert design["bounds"]["semantics"] == "calibrated_candidate_disagreement_hull"
    assert design["bounds"]["formal_frequentist_coverage_claim"] is False


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


def test_round21_panel_rejects_duplicate_condition_decision_rows() -> None:
    panel = _panel("train", first_condition=0, condition_count=80)
    condition_ids = panel.condition_ids.copy()
    event_start_ms = panel.event_start_ms.copy()
    decision_time_ms = panel.decision_time_ms.copy()
    labels = panel.labels.copy()
    condition_ids[1] = condition_ids[0]
    event_start_ms[1] = event_start_ms[0]
    decision_time_ms[1] = decision_time_ms[0]
    labels[1] = labels[0]

    with pytest.raises(ValueError, match="condition target identity differs"):
        replace(
            panel,
            condition_ids=condition_ids,
            event_start_ms=event_start_ms,
            decision_time_ms=decision_time_ms,
            labels=labels,
        ).validate()


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


def test_round21_transform_uses_weighted_median_and_iqr_not_mean_std() -> None:
    matrix = np.asarray([[0], [1], [2], [3], [100_000]], dtype=np.float32)
    transform = model_module._fit_transform(
        matrix,
        np.full(5, 0.2, dtype=np.float64),
    )

    assert transform["center"] == [2.0]
    assert "mean" not in transform
    assert 1.4 < transform["scale"][0] < 1.5


def test_round21_development_requires_the_frozen_train_to_tune_purge() -> None:
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

    with pytest.raises(ValueError, match="partition boundary differs"):
        fit_round21_development(
            train=train,
            tune_calibration=calibration,
            tune_selection=selection,
            compute_backend="cpu",
        )


def test_round21_fits_core_and_exact_matched_optional_challengers() -> None:
    train = _panel("train", first_condition=0, condition_count=100)
    calibration = _panel(
        "tune_calibration",
        first_condition=106,
        condition_count=120,
    )
    selection = _panel(
        "tune_selection",
        first_condition=226,
        condition_count=80,
    )

    artifact = fit_round21_development(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        compute_backend="cpu",
    )

    assert artifact["economic_evaluation_completed"] is False
    assert artifact["dataset_design_sha256"] == POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
    assert artifact["test_features_accessed"] is False
    assert artifact["test_targets_accessed"] is False
    assert artifact["model_selected"] is False
    assert artifact["profitability_claim"] is False
    assert artifact["live_trading_authority"] is False
    layers = artifact["layers"]
    assert set(layers) == {"core", "core_spot", "core_spot_usdm"}
    assert all(len(layer["candidate_ledger"]) == 5 for layer in layers.values())
    assert all(
        len(layers[layer]["matched_core_candidate_ledger"]) == 5
        for layer in ("core_spot", "core_spot_usdm")
    )
    assert layers["core_spot"]["comparison"]["matched_decision_count"] == int(
        np.count_nonzero(selection.spot_available)
    )
    assert layers["core_spot_usdm"]["comparison"]["matched_decision_count"] == int(
        np.count_nonzero(selection.usdm_available)
    )
    assert layers["core_spot"]["comparison"]["predictive_development_accepted"]
    assert layers["core_spot_usdm"]["comparison"]["predictive_development_accepted"]
    for layer in ("core_spot", "core_spot_usdm"):
        matched_record = next(
            record
            for record in layers[layer]["matched_core_candidate_ledger"]
            if record["candidate_id"]
            == layers[layer]["matched_core_selected_candidate_id"]
        )
        assert matched_record["model"]["population_layer"] == layer
        assert matched_record["model"]["feature_layer"] == "core"
        assert (
            layers[layer]["comparison"]["matched_core_candidate_id"]
            == matched_record["candidate_id"]
        )

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

    inference = Round21InferencePanel.from_development(selection)
    assert not hasattr(inference, "labels")
    assert inference.target_accessed is False
    assert all(
        not value.flags.writeable
        for value in (
            inference.condition_ids,
            inference.event_start_ms,
            inference.decision_time_ms,
            inference.structural_probability,
            inference.market_prior_probability,
            inference.core_features,
            inference.spot_features,
            inference.usdm_features,
            inference.spot_available,
            inference.usdm_available,
        )
    )
    inference_indices, inference_predictions = predict_round21_candidate(
        core_record["model"],
        inference,
    )
    assert np.array_equal(inference_indices, indices)
    assert np.array_equal(inference_predictions, predictions)
    mutated_core = inference.core_features.copy()
    mutated_core[0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="inference panel is invalid"):
        replace(inference, core_features=mutated_core).validate()
    mutated_core.setflags(write=False)
    with pytest.raises(ValueError, match="inference panel identity differs"):
        replace(inference, core_features=mutated_core).validate()

    core_batch = predict_round21_probability_batch(
        artifact,
        population_layer="core",
        panel=inference,
    )
    optional_batch = predict_round21_probability_batch(
        artifact,
        population_layer="core_spot",
        panel=inference,
    )
    assert len(core_batch.contributing_candidate_ids) == 5
    assert len(optional_batch.contributing_candidate_ids) == 10
    assert np.all(core_batch.lower_up <= core_batch.probability_up)
    assert np.all(core_batch.probability_up <= core_batch.upper_up)
    assert not core_batch.probability_up.flags.writeable
    assert (
        core_batch.identity_payload()["probability_envelope_design_sha256"]
        == POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256
    )
    assert core_batch.row(int(core_batch.indices[0]))[0] == pytest.approx(
        core_batch.probability_up[0]
    )
    with pytest.raises(ValueError, match="probability batch is invalid"):
        replace(
            core_batch,
            lower_up=np.full_like(core_batch.lower_up, 1.0),
        ).validated()

    changed = json.loads(json.dumps(artifact))
    changed["layers"]["core_spot"]["matched_core_candidate_ledger"][0]["model"][
        "feature_layer"
    ] = "core_spot"
    with pytest.raises(ValueError, match="artifact differs"):
        validate_round21_development_artifact(_rehash(changed))


def test_round21_artifact_rejects_rehashed_authority_drift() -> None:
    train = _panel("train", first_condition=0, condition_count=100)
    calibration = _panel(
        "tune_calibration",
        first_condition=106,
        condition_count=120,
    )
    selection = _panel(
        "tune_selection",
        first_condition=226,
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
