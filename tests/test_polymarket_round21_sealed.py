from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round21_tcn as tcn_module

from polymarket_round21_support import (
    round21_accepted_basis_ablation,
    round21_panel,
    round21_replay_condition,
    sha,
)
import simple_ai_trading.polymarket_round21_sealed as sealed_module
from simple_ai_trading.polymarket_round21_model import fit_round21_development
from simple_ai_trading.polymarket_round21_replay import replay_round21_full_matrix
from simple_ai_trading.polymarket_round21_sealed import (
    POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
    build_round21_sealed_evaluation_result,
    build_round21_sealed_result_bundle,
    evaluate_round21_sealed_economics,
    evaluate_round21_sealed_predictions,
    load_round21_sealed_design,
    load_round21_sealed_result_bundle,
    load_verified_round21_sealed_result_bundle,
    validate_round21_sealed_design,
    validate_round21_sealed_result_bundle,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-terminal-sealed-evaluation-design-v7.json"
)


@pytest.fixture(scope="module", autouse=True)
def _bounded_tcn_test_epochs():
    patch = pytest.MonkeyPatch()
    patch.setattr(tcn_module, "ROUND21_TCN_MAXIMUM_EPOCHS", 1)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    train = round21_panel("train", first_condition=0, condition_count=100)
    calibration = round21_panel(
        "tune_calibration",
        first_condition=106,
        condition_count=120,
    )
    selection = round21_panel(
        "tune_selection",
        first_condition=226,
        condition_count=80,
    )
    return fit_round21_development(
        train=train,
        tune_calibration=calibration,
        tune_selection=selection,
        basis_ablation_result=round21_accepted_basis_ablation(
            train,
            calibration,
            selection,
        ),
        compute_backend="cpu",
    )


@pytest.fixture(scope="module")
def predictive(artifact: dict[str, object]):
    return evaluate_round21_sealed_predictions(
        artifact,
        population_layer="core",
        test_panel=round21_panel("test", first_condition=306, condition_count=80),
    )


@pytest.fixture(scope="module")
def economic(predictive):
    matrix = replay_round21_full_matrix((round21_replay_condition(),))
    result = evaluate_round21_sealed_economics(
        matrix,
        test_dataset_sha256=predictive.test_dataset_sha256,
        test_target_manifest_sha256=predictive.test_target_manifest_sha256,
    )
    result = replace(
        result,
        condition_count=predictive.resolved_condition_count,
        condition_population_sha256=predictive.condition_population_sha256,
        result_sha256=sha("placeholder"),
    )
    return replace(
        result,
        result_sha256=sealed_module._canonical_sha256(result.identity_payload()),
    ).validated()


def test_round21_sealed_design_is_canonical_and_has_no_authority() -> None:
    raw = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = raw.pop("design_sha256")
    actual = hashlib.sha256(
        json.dumps(
            raw,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert claimed == actual == POLYMARKET_ROUND21_SEALED_DESIGN_SHA256
    assert load_round21_sealed_design(REPOSITORY)["design_sha256"] == claimed
    assert raw["one_use_state_machine"]["result_binds_predeclared_test_population"]
    assert not any(raw["authority"].values())

    tampered = {**raw, "round": 22, "design_sha256": claimed}
    with pytest.raises(ValueError, match="design differs"):
        validate_round21_sealed_design(tampered)


def test_round21_sealed_predictive_gate_replays_every_control_without_refit(
    predictive,
) -> None:
    assert predictive.resolved_condition_count == 80
    assert predictive.calendar_day_count == 1
    assert set(predictive.control_metrics) == {
        "structural_probability_raw",
        "structural_probability_calibrated",
        "executable_market_prior_raw",
        "executable_market_prior_calibrated",
        "training_prevalence",
    }
    assert set(predictive.paired_improvements) == set(predictive.control_metrics)
    assert predictive.gate_passed is False
    assert "insufficient_resolved_conditions" in predictive.reasons
    assert "insufficient_calendar_days" in predictive.reasons
    assert predictive.profitability_claim is False
    assert predictive.live_trading_authority is False

    with pytest.raises(ValueError, match="predictive result differs"):
        replace(predictive, test_dataset_sha256=sha("tampered")).validated()


def test_round21_sealed_predictive_rejects_non_test_role(
    artifact: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="panel role differs"):
        evaluate_round21_sealed_predictions(
            artifact,
            population_layer="core",
            test_panel=round21_panel(
                "tune_selection",
                first_condition=306,
                condition_count=80,
            ),
        )


def test_round21_sealed_economic_gate_requires_all_81_ledgers(economic) -> None:
    assert economic.ledger_count == 81
    assert economic.qualified_ledger_count == 0
    assert len(set(economic.ledger_sha256)) == 81
    assert economic.gate_passed is False
    assert economic.reasons == ("not_all_81_ledgers_qualified",)
    assert economic.profitability_claim is False
    assert economic.paper_trading_authority is False

    with pytest.raises(ValueError, match="economic matrix differs"):
        evaluate_round21_sealed_economics(
            (),
            test_dataset_sha256=economic.test_dataset_sha256,
            test_target_manifest_sha256=economic.test_target_manifest_sha256,
        )


def test_round21_sealed_result_binds_population_and_never_promotes(
    predictive,
    economic,
) -> None:
    result = build_round21_sealed_evaluation_result(
        claim_sha256=sha("claim"),
        test_access_sha256=sha("access"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=sha("sealed-test-population"),
        predictive=predictive,
        economic=economic,
    )

    assert result.sealed_test_population_manifest_sha256 == sha(
        "sealed-test-population"
    )
    assert result.candidate_accepted is False
    assert result.ai_model is None
    assert result.automatic_promotion is False
    assert result.live_trading_authority is False
    with pytest.raises(ValueError, match="sealed evaluation result differs"):
        replace(result, candidate_accepted=True).validated()
    mismatched = replace(
        economic,
        test_dataset_sha256=sha("different-test-dataset"),
        result_sha256=sha("placeholder"),
    )
    mismatched = replace(
        mismatched,
        result_sha256=sealed_module._canonical_sha256(mismatched.identity_payload()),
    ).validated()
    with pytest.raises(ValueError, match="populations differ"):
        build_round21_sealed_evaluation_result(
            claim_sha256=sha("claim"),
            test_access_sha256=sha("access"),
            selected_population_layer="core",
            sealed_test_population_manifest_sha256=sha("sealed-test-population"),
            predictive=predictive,
            economic=mismatched,
        )


def test_round21_sealed_bundle_round_trips_nested_restart_state(
    predictive,
    economic,
    tmp_path: Path,
) -> None:
    result = build_round21_sealed_evaluation_result(
        claim_sha256=sha("bundle-claim"),
        test_access_sha256=sha("bundle-access"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=sha("bundle-population"),
        predictive=predictive,
        economic=economic,
    )
    bundle = build_round21_sealed_result_bundle(result)

    assert validate_round21_sealed_result_bundle(bundle) == result
    path = tmp_path / "sealed-result.json"
    path.write_text(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )
    assert load_round21_sealed_result_bundle(path) == result

    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["economic"]["qualified_ledger_count"] = 81
    body = dict(tampered)
    body.pop("bundle_sha256")
    tampered["bundle_sha256"] = sealed_module._canonical_sha256(body)
    with pytest.raises(ValueError, match="bundle differs"):
        validate_round21_sealed_result_bundle(tampered)

    path.write_text('{"a":1,"a":2}', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_round21_sealed_result_bundle(path)


def test_round21_verified_sealed_bundle_binds_exact_file(
    predictive,
    economic,
    tmp_path,
) -> None:
    result = build_round21_sealed_evaluation_result(
        claim_sha256=sha("claim"),
        test_access_sha256=sha("access"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=sha("test-population"),
        predictive=predictive,
        economic=economic,
    )
    payload = build_round21_sealed_result_bundle(result)
    path = tmp_path / "sealed-result.json"
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    verified = load_verified_round21_sealed_result_bundle(
        path,
        expected_file_sha256=expected,
    )
    assert verified.result == result
    assert verified.path == path.resolve()
    assert verified.file_sha256 == expected

    with pytest.raises(ValueError, match="evidence hash differs"):
        load_verified_round21_sealed_result_bundle(
            path,
            expected_file_sha256=sha("different-file"),
        )


def test_round21_sealed_design_loader_rejects_malformed_sources(tmp_path) -> None:
    with pytest.raises(ValueError, match="design is unavailable"):
        load_round21_sealed_design(tmp_path)

    path = (
        tmp_path
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-021-terminal-sealed-evaluation-design-v7.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text('{"a":1,"a":2}', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_round21_sealed_design(tmp_path)
    path.write_text("{x", encoding="ascii")
    with pytest.raises(ValueError, match="design is invalid"):
        load_round21_sealed_design(tmp_path)
    path.write_text("[]", encoding="ascii")
    with pytest.raises(ValueError, match="design is invalid"):
        load_round21_sealed_design(tmp_path)


def test_round21_sealed_results_reject_metric_and_ledger_drift(
    predictive,
    economic,
) -> None:
    assert predictive.asdict()["result_sha256"] == predictive.result_sha256
    with pytest.raises(ValueError, match="predictive result differs"):
        replace(predictive, candidate_metrics={}).validated()
    malformed = dict(predictive.candidate_metrics)
    malformed["calibration_slope"] = "bad"
    with pytest.raises(ValueError, match="predictive result differs"):
        replace(predictive, candidate_metrics=malformed).validated()
    malformed_improvements = dict(predictive.paired_improvements)
    malformed_improvements["training_prevalence"] = {}
    with pytest.raises(ValueError, match="predictive result differs"):
        replace(predictive, paired_improvements=malformed_improvements).validated()

    assert economic.asdict()["result_sha256"] == economic.result_sha256
    with pytest.raises(ValueError, match="economic result differs"):
        replace(economic, ledger_count=80).validated()


def test_round21_sealed_economics_accepts_only_fully_qualified_matrix(
    monkeypatch,
) -> None:
    class Replay:
        qualification_reasons = ("sealed_test_evidence_unavailable",)
        economic_gate_passed = True

        def __init__(self, profile: str, scenario: str, index: int) -> None:
            self.profile = profile
            self.scenario = scenario
            self.replay_sha256 = sha(f"qualified-ledger-{index}")
            self.conditions = (
                SimpleNamespace(
                    condition_id="0x" + "1" * 64,
                    event_start_ms=1_800_000_000_000,
                    outcome_sha256=sha("outcome"),
                ),
            )

        def validated(self):
            return self

    matrix = tuple(
        Replay(profile.name, scenario.name, index)
        for index, (profile, scenario) in enumerate(
            (profile, scenario)
            for profile in sealed_module.POLYMARKET_ROUND21_RISK_PROFILES
            for scenario in sealed_module.POLYMARKET_ROUND21_EXECUTION_SCENARIOS
        )
    )
    monkeypatch.setattr(
        sealed_module,
        "round21_replay_matrix_sha256",
        lambda _matrix: sha("qualified-matrix"),
    )

    result = evaluate_round21_sealed_economics(
        matrix,
        test_dataset_sha256=sha("test-dataset"),
        test_target_manifest_sha256=sha("test-targets"),
    )
    assert result.qualified_ledger_count == 81
    assert result.gate_passed is True
    assert result.reasons == ()
    matrix[1].conditions = (
        SimpleNamespace(
            condition_id="0x" + "2" * 64,
            event_start_ms=1_800_000_300_000,
            outcome_sha256=sha("different-outcome"),
        ),
    )
    with pytest.raises(ValueError, match="economic populations differ"):
        evaluate_round21_sealed_economics(
            matrix,
            test_dataset_sha256=sha("test-dataset"),
            test_target_manifest_sha256=sha("test-targets"),
        )


def test_round21_sealed_builder_checks_optional_and_ai_bindings(
    predictive,
    economic,
) -> None:
    optional_predictive = replace(
        predictive,
        population_layer="core_spot",
        result_sha256=sha("placeholder"),
    )
    optional_predictive = replace(
        optional_predictive,
        result_sha256=sealed_module._canonical_sha256(
            optional_predictive.identity_payload()
        ),
    ).validated()
    optional = SimpleNamespace(
        challenger_layer="core_spot",
        challenger_matrix_sha256=economic.matrix_sha256,
        all_replays_accepted=True,
        comparison_sha256=sha("optional-comparison"),
    )
    optional.validated = lambda: optional
    ai = SimpleNamespace(
        baseline_matrix_sha256=economic.matrix_sha256,
        development_qualified=True,
        comparison_sha256=sha("ai-comparison"),
        model="qwen3.5:9b",
        model_digest=sha("qwen3.5:9b"),
    )
    ai.validated = lambda: ai

    result = build_round21_sealed_evaluation_result(
        claim_sha256=sha("claim"),
        test_access_sha256=sha("access"),
        selected_population_layer="core_spot",
        sealed_test_population_manifest_sha256=sha("test-population"),
        predictive=optional_predictive,
        economic=economic,
        optional_comparison=optional,
        ai_comparison=ai,
    )
    assert result.optional_uplift_gate_passed is True
    assert result.ai_uplift_gate_passed is True
    assert result.ai_model == "qwen3.5:9b"
    assert result.ai_model_digest == sha("qwen3.5:9b")

    with pytest.raises(ValueError, match="selected layer differs"):
        build_round21_sealed_evaluation_result(
            claim_sha256=sha("claim"),
            test_access_sha256=sha("access"),
            selected_population_layer="bad",
            sealed_test_population_manifest_sha256=sha("test-population"),
            predictive=predictive,
            economic=economic,
        )
    with pytest.raises(ValueError, match="optional comparison differs"):
        build_round21_sealed_evaluation_result(
            claim_sha256=sha("claim"),
            test_access_sha256=sha("access"),
            selected_population_layer="core_spot",
            sealed_test_population_manifest_sha256=sha("test-population"),
            predictive=optional_predictive,
            economic=economic,
        )
    optional.challenger_matrix_sha256 = sha("wrong-matrix")
    with pytest.raises(ValueError, match="optional comparison differs"):
        build_round21_sealed_evaluation_result(
            claim_sha256=sha("claim"),
            test_access_sha256=sha("access"),
            selected_population_layer="core_spot",
            sealed_test_population_manifest_sha256=sha("test-population"),
            predictive=optional_predictive,
            economic=economic,
            optional_comparison=optional,
        )
    ai.baseline_matrix_sha256 = sha("wrong-matrix")
    with pytest.raises(ValueError, match="AI comparison differs"):
        build_round21_sealed_evaluation_result(
            claim_sha256=sha("claim"),
            test_access_sha256=sha("access"),
            selected_population_layer="core",
            sealed_test_population_manifest_sha256=sha("test-population"),
            predictive=predictive,
            economic=economic,
            ai_comparison=ai,
        )
