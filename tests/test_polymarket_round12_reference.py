from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round12_reference import (
    POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256,
    PolymarketRound12ReferenceModel,
    decide_round12_primary_action,
    load_round12_confirmation_contract,
    load_round12_reference_from_round11_artifact,
    load_round12_reference_from_round11_bytes,
    polymarket_round12_primary_policy,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-011-single-leg-directional-value-artifact.json"
)
CONTRACT = ARTIFACT.with_name(
    "round-012-fixed-calibration-confirmation-contract.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@pytest.fixture(scope="module")
def reference() -> PolymarketRound12ReferenceModel:
    return load_round12_reference_from_round11_artifact(ARTIFACT)


def test_round12_reference_validates_the_exact_predecessor(
    reference: PolymarketRound12ReferenceModel,
) -> None:
    assert (
        reference.predecessor_artifact_sha256
        == POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256
    )
    assert reference.external_feature_coefficients_applied is False
    assert reference.validated() is reference
    assert len(reference.model_sha256) == 64


def test_round12_confirmation_contract_binds_model_policy_and_implementation() -> None:
    contract = load_round12_confirmation_contract(CONTRACT)

    assert contract["status"] == "frozen_before_fresh_capture"
    assert contract["freshness"]["capture_started"] is False
    assert contract["authority"]["profitability_claim"] is False
    assert len(str(contract["contract_sha256"])) == 64


def test_round12_confirmation_contract_is_relocatable(tmp_path: Path) -> None:
    contract_path = tmp_path / CONTRACT.name
    artifact_path = tmp_path / ARTIFACT.name
    contract_path.write_bytes(CONTRACT.read_bytes())
    artifact_path.write_bytes(ARTIFACT.read_bytes())

    relocated = load_round12_confirmation_contract(contract_path)

    assert relocated["contract_sha256"] == load_round12_confirmation_contract(
        CONTRACT
    )["contract_sha256"]


def test_round12_confirmation_contract_rejects_rehashed_policy_mutation(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["primary_policy"]["minimum_direction_probability"] = 0.79
    payload.pop("contract_sha256")
    payload["contract_sha256"] = _canonical_sha256(payload)
    mutated = tmp_path / CONTRACT.name
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model or policy contract"):
        load_round12_confirmation_contract(
            mutated,
            predecessor_artifact_path=ARTIFACT,
        )


def test_round12_confirmation_contract_rejects_implementation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simple_ai_trading import polymarket_action_pipeline

    monkeypatch.setattr(
        polymarket_action_pipeline,
        "polymarket_action_pipeline_implementation_sha256",
        lambda: "f" * 64,
    )

    with pytest.raises(ValueError, match="implementation differs"):
        load_round12_confirmation_contract(CONTRACT)


def test_round12_reference_is_path_and_working_directory_independent(
    reference: PolymarketRound12ReferenceModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relocated = tmp_path / "arbitrary" / "installation" / "model.json"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(ARTIFACT.read_bytes())
    monkeypatch.chdir(tmp_path)

    loaded = load_round12_reference_from_round11_artifact(relocated)

    assert loaded == reference
    assert loaded.predict_pair(0.731) == reference.predict_pair(0.731)


def test_round12_reference_rejects_tamper_even_if_json_remains_valid() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["direction_calibration"]["slope"] = 1.5
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")

    with pytest.raises(ValueError, match="identity or provenance"):
        load_round12_reference_from_round11_bytes(raw)


def test_round12_reference_rejects_duplicate_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_round12_reference_from_round11_bytes(
            b'{"artifact_sha256":"a","artifact_sha256":"b"}'
        )


@pytest.mark.parametrize("prior", [0.01, 0.10, 0.25, 0.5, 0.75, 0.90, 0.99])
def test_round12_probabilities_are_finite_and_complementary(
    reference: PolymarketRound12ReferenceModel,
    prior: float,
) -> None:
    probability_up, probability_down = reference.predict_pair(prior)

    assert math.isfinite(probability_up)
    assert math.isfinite(probability_down)
    assert 0.0 < probability_up < 1.0
    assert 0.0 < probability_down < 1.0
    assert probability_up + probability_down == 1.0


def test_round12_calibration_is_strictly_monotone(
    reference: PolymarketRound12ReferenceModel,
) -> None:
    predictions = [reference.predict_up(index / 100.0) for index in range(1, 100)]
    assert all(left < right for left, right in zip(predictions, predictions[1:]))


@pytest.mark.parametrize("prior", [float("nan"), float("inf"), -0.1, 0.0, 1.0, 1.1])
def test_round12_reference_rejects_invalid_probability_evidence(
    reference: PolymarketRound12ReferenceModel,
    prior: float,
) -> None:
    with pytest.raises(ValueError):
        reference.predict_up(prior)


def test_round12_reference_matches_independent_torch_float64_formula(
    reference: PolymarketRound12ReferenceModel,
) -> None:
    torch = pytest.importorskip("torch")
    priors = torch.tensor([0.01, 0.2, 0.5, 0.8, 0.99], dtype=torch.float64)
    clipped = torch.clamp(
        priors,
        min=reference.probability_clip,
        max=1.0 - reference.probability_clip,
    )
    logits = torch.log(clipped) - torch.log1p(-clipped)
    challenger = torch.sigmoid(
        reference.calibration_intercept
        + reference.calibration_slope * (logits + reference.residual_intercept)
    )
    expected = challenger.detach().cpu().tolist()
    actual = [reference.predict_up(float(value)) for value in priors.tolist()]

    assert actual == pytest.approx(expected, rel=0.0, abs=2e-15)


def _eligible_decision(
    reference: PolymarketRound12ReferenceModel,
    **overrides: object,
):
    values: dict[str, object] = {
        "asset": "BTC",
        "market_prior_up": 0.80,
        "remaining_seconds": 180.0,
        "up_quantity": 5.0,
        "down_quantity": 5.0,
        "up_conservative_entry_cost_quote": 3.9,
        "down_conservative_entry_cost_quote": 3.9,
        "decision_evidence_admissible": True,
        "lifecycle_clear": True,
    }
    values.update(overrides)
    return decide_round12_primary_action(reference, **values)  # type: ignore[arg-type]


def test_round12_primary_policy_is_single_and_hash_bound() -> None:
    policy = polymarket_round12_primary_policy()
    assert policy.profile == "conservative"
    assert policy.minimum_direction_probability == 0.80
    assert policy.identity_payload()["forced_activity"] is False
    assert policy.validated() is policy


def test_round12_primary_action_selects_only_one_positive_side(
    reference: PolymarketRound12ReferenceModel,
) -> None:
    decision = _eligible_decision(reference)

    assert decision.action == "BUY_FOK_HOLD_TO_RESOLUTION"
    assert decision.outcome == "Up"
    assert decision.probability is not None and decision.probability > 0.80
    assert decision.expected_edge_quote is not None
    assert decision.expected_edge_quote > 0.02


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"asset": "DOGE"}, "unsupported_asset"),
        ({"decision_evidence_admissible": False}, "decision_evidence_not_admissible"),
        ({"lifecycle_clear": False}, "lifecycle_not_clear"),
        ({"market_prior_up": float("nan")}, "invalid_numeric_evidence"),
        ({"remaining_seconds": 119.999}, "inside_minimum_remaining_window"),
        ({"up_quantity": 0.0}, "invalid_execution_economics"),
        (
            {
                "market_prior_up": 0.50,
                "up_conservative_entry_cost_quote": 2.0,
                "down_conservative_entry_cost_quote": 2.0,
            },
            "no_positive_conservative_edge",
        ),
    ],
)
def test_round12_primary_action_fails_closed(
    reference: PolymarketRound12ReferenceModel,
    overrides: dict[str, object],
    reason: str,
) -> None:
    decision = _eligible_decision(reference, **overrides)

    assert decision.abstained
    assert decision.reason == reason
    assert decision.outcome is None
