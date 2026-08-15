from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import polymarket_round21_operator as operator_module
from simple_ai_trading.polymarket_round21_ablation import (
    POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256,
    evaluate_round21_probability_basis_ablation,
    load_round21_probability_basis_ablation_design,
    load_round21_probability_basis_ablation_result,
    validate_round21_probability_basis_ablation_result,
)
from simple_ai_trading.polymarket_round21_model import Round21DevelopmentPanel
from polymarket_round21_support import round21_panel


PUBLICATION_SHA = "1" * 64
TERMINAL_SHA = "2" * 64


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rehash(value: dict[str, object]) -> dict[str, object]:
    changed = deepcopy(value)
    changed.pop("result_sha256", None)
    changed["result_sha256"] = _canonical_sha256(changed)
    return changed


def _panel_with_prior(
    role: str,
    *,
    first_condition: int,
    condition_count: int,
    predictive: bool,
) -> Round21DevelopmentPanel:
    panel = round21_panel(
        role,
        first_condition=first_condition,
        condition_count=condition_count,
    )
    structural = np.full(len(panel.labels), 0.5, dtype=np.float64)
    market_prior = (
        np.where(panel.labels == 1.0, 0.9, 0.1) if predictive else structural.copy()
    )
    return replace(
        panel,
        structural_probability=structural,
        market_prior_probability=np.asarray(market_prior, dtype=np.float64),
        core_features=np.zeros_like(panel.core_features),
    ).validate()


def _panels(
    *, predictive: bool
) -> tuple[
    Round21DevelopmentPanel,
    Round21DevelopmentPanel,
    Round21DevelopmentPanel,
]:
    return (
        _panel_with_prior(
            "train",
            first_condition=0,
            condition_count=100,
            predictive=predictive,
        ),
        _panel_with_prior(
            "tune_calibration",
            first_condition=106,
            condition_count=120,
            predictive=predictive,
        ),
        _panel_with_prior(
            "tune_selection",
            first_condition=226,
            condition_count=80,
            predictive=predictive,
        ),
    )


def test_round21_basis_ablation_design_is_canonical() -> None:
    repository = Path(__file__).resolve().parents[1]

    design = load_round21_probability_basis_ablation_design(repository)

    assert design["design_sha256"] == POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256
    assert design["authority"]["live_trading_authority"] is False


def test_round21_basis_ablation_accepts_only_clear_paired_uplift(
    tmp_path: Path,
) -> None:
    train, calibration, selection = _panels(predictive=True)

    result = evaluate_round21_probability_basis_ablation(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        publication_manifest_sha256=PUBLICATION_SHA,
        terminal_transport_manifest_sha256=TERMINAL_SHA,
    )

    assert result["basis_accepted"] is True
    assert result["source_evidence"] == {
        "publication_manifest_sha256": PUBLICATION_SHA,
        "terminal_transport_manifest_sha256": TERMINAL_SHA,
    }
    assert result["arms"]["challenger"]["feature_count"] == (
        result["arms"]["baseline"]["feature_count"] + 1
    )
    assert all(
        result["paired_improvement"][metric]["lower_95"] > 0.0
        for metric in ("log_loss", "brier")
    )
    assert result["development_targets_accessed"] is True
    assert result["sealed_test_targets_accessed"] is False
    assert result["economic_evaluation_completed"] is False
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False

    output = tmp_path / "basis-ablation.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    assert load_round21_probability_basis_ablation_result(output) == result
    assembly = SimpleNamespace(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        publication_manifest_sha256=PUBLICATION_SHA,
        terminal_transport_manifest_sha256=TERMINAL_SHA,
    )
    assert (
        operator_module._require_accepted_round21_probability_basis(  # noqa: SLF001
            result,
            assembly=assembly,
            require_exact_dataset_identity=True,
        )
        == result
    )

    rejected_source = deepcopy(result)
    rejected_source["source_evidence"]["publication_manifest_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="gate is not accepted"):
        operator_module._require_accepted_round21_probability_basis(  # noqa: SLF001
            _rehash(rejected_source),
            assembly=assembly,
            require_exact_dataset_identity=True,
        )


def test_round21_basis_ablation_rejects_no_uplift() -> None:
    train, calibration, selection = _panels(predictive=False)

    result = evaluate_round21_probability_basis_ablation(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        publication_manifest_sha256=PUBLICATION_SHA,
        terminal_transport_manifest_sha256=TERMINAL_SHA,
    )

    assert result["basis_accepted"] is False
    assert result["next_action"].startswith("reject_basis")
    assert all(
        result["paired_improvement"][metric]["lower_95"] <= 0.0
        for metric in ("log_loss", "brier")
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["arms"]["baseline"]["selected_model"].update(
            {"coefficient": "not-an-array"}
        ),
        lambda value: value["arms"]["challenger"].update(
            {"feature_count": value["arms"]["baseline"]["feature_count"]}
        ),
        lambda value: value["arms"]["baseline"]["regularization_candidates"][0].update(
            {"candidate_id": "wrong"}
        ),
        lambda value: value["dataset_and_partition"]["train"].update(
            {"row_count": "100"}
        ),
        lambda value: value["paired_improvement"]["log_loss"].update(
            {"lower_95": "not-a-number"}
        ),
    ),
)
def test_round21_basis_ablation_validator_fails_closed(mutation: object) -> None:
    train, calibration, selection = _panels(predictive=True)
    result = evaluate_round21_probability_basis_ablation(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        publication_manifest_sha256=PUBLICATION_SHA,
        terminal_transport_manifest_sha256=TERMINAL_SHA,
    )
    changed = deepcopy(result)
    mutation(changed)  # type: ignore[operator]

    with pytest.raises(ValueError, match="result differs"):
        validate_round21_probability_basis_ablation_result(_rehash(changed))
