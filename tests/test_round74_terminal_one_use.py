from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from simple_ai_trading.round74_terminal_one_use import (
    Round74TerminalOneUseStore,
    Round74TerminalPreaccessIdentity,
    Round74TerminalReuseError,
    build_round74_terminal_result_bundle,
    validate_round74_terminal_result_bundle,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _preaccess() -> Round74TerminalPreaccessIdentity:
    result = Round74TerminalPreaccessIdentity(
        plan_sha256="1" * 64,
        coverage_sha256="2" * 64,
        partition_sha256="3" * 64,
        test_population_sha256="4" * 64,
        test_run_ids=tuple(f"{index:032x}" for index in range(1, 25)),
        database_route_sha256="f" * 64,
        optimization_population="eligible_target",
        development_bundle_sha256="5" * 64,
        pretest_policy_sha256="6" * 64,
        feature_scaler_sha256="0" * 64,
        probability_calibration_sha256="7" * 64,
        action_selection_sha256="8" * 64,
        final_action_configuration_sha256="9" * 64,
        ai_pretest_qualification_sha256="a" * 64,
        ai_manifest_sha256=("b" * 64, "c" * 64),
        profile="conservative",
        backend_preflight_sha256="d" * 64,
        model_provenance_sha256=("e" * 64, "f" * 64),
        terminal_observed_wall_ns=1_800_000_000_000_000_000,
    )
    result.validate()
    return result


def _result_bundle(claim) -> dict[str, object]:
    test_access_sha256 = _sha(
        {
            "pretest_model_policy_sha256": claim.preaccess.pretest_policy_sha256,
            "test_unlock_sha256": claim.test_unlock_sha256,
        }
    )
    dataset = {
        "schema_version": "round-074-sealed-dataset-identity-test",
        "test_access_sha256": test_access_sha256,
        "partition_sha256": claim.preaccess.partition_sha256,
        "scaler_sha256": claim.preaccess.feature_scaler_sha256,
        "optimization_population": claim.preaccess.optimization_population,
        "test_population_sha256": claim.preaccess.test_population_sha256,
        "test_run_ids": list(claim.preaccess.test_run_ids),
    }
    dataset_sha256 = _sha(dataset)
    report: dict[str, object] = {
        "schema_version": "round-074-sealed-evaluation-test",
        "reservation_id": "a" * 64,
        "dataset_sha256": dataset_sha256,
        "test_access_sha256": test_access_sha256,
        "pretest_policy_sha256": claim.preaccess.pretest_policy_sha256,
        "probability_calibration_sha256": (
            claim.preaccess.probability_calibration_sha256
        ),
        "action_selection_sha256": claim.preaccess.action_selection_sha256,
        "final_action_configuration_sha256": (
            claim.preaccess.final_action_configuration_sha256
        ),
        "ai_pretest_qualification_sha256": (
            claim.preaccess.ai_pretest_qualification_sha256
        ),
        "profile": claim.preaccess.profile,
        "optimization_population": claim.preaccess.optimization_population,
        "result_outcome": "candidate_failed_predeclared_gates",
    }
    report["report_sha256"] = _sha(report)
    sealed_claim: dict[str, object] = {
        "schema_version": "round-074-sealed-claim-test",
        "reservation_id": "a" * 64,
        "dataset_sha256": dataset_sha256,
        "test_access_sha256": test_access_sha256,
        "partition_sha256": claim.preaccess.partition_sha256,
        "scaler_sha256": claim.preaccess.feature_scaler_sha256,
        "pretest_policy_sha256": claim.preaccess.pretest_policy_sha256,
        "probability_calibration_sha256": (
            claim.preaccess.probability_calibration_sha256
        ),
        "action_selection_sha256": claim.preaccess.action_selection_sha256,
        "final_action_configuration_sha256": (
            claim.preaccess.final_action_configuration_sha256
        ),
        "ai_pretest_qualification_sha256": (
            claim.preaccess.ai_pretest_qualification_sha256
        ),
        "profile": claim.preaccess.profile,
        "optimization_population": claim.preaccess.optimization_population,
        "test_population_sha256": claim.preaccess.test_population_sha256,
        "test_run_ids": list(claim.preaccess.test_run_ids),
        "status": "complete",
        "result_sha256": report["report_sha256"],
    }
    sealed_claim["claim_sha256"] = _sha(sealed_claim)
    return build_round74_terminal_result_bundle(
        access_claim=claim,
        dataset_identity=dataset,
        sealed_report=report,
        finalized_sealed_claim=sealed_claim,
    )


def test_terminal_store_reserves_before_result_and_never_reuses(tmp_path) -> None:
    store = Round74TerminalOneUseStore(tmp_path / "terminal.sqlite3")
    claim = store.reserve(_preaccess())

    assert claim.status == "reserved"
    assert claim.preaccess.sealed_targets_read is False
    assert claim.as_dict()["access_consumed_before_target_loading"] is True
    with pytest.raises(Round74TerminalReuseError, match="already consumed"):
        store.reserve(_preaccess())


def test_terminal_store_persists_complete_recoverable_bundle(tmp_path) -> None:
    store = Round74TerminalOneUseStore(tmp_path / "terminal.sqlite3")
    claim = store.reserve(_preaccess())
    bundle = _result_bundle(claim)

    completed = store.finalize_success(claim, bundle)

    assert completed.status == "complete"
    assert completed.result_sha256 == bundle["bundle_sha256"]
    assert store.load_completed_bundle() == bundle
    with pytest.raises(Round74TerminalReuseError, match="already consumed"):
        store.reserve(_preaccess())


def test_terminal_store_failure_consumes_access_without_result_recovery(
    tmp_path,
) -> None:
    store = Round74TerminalOneUseStore(tmp_path / "terminal.sqlite3")
    claim = store.reserve(_preaccess())

    failed = store.finalize_failure(claim, RuntimeError("target loader stopped"))

    assert failed.status == "failed"
    assert failed.error == "RuntimeError: target loader stopped"
    with pytest.raises(ValueError, match="completed result is unavailable"):
        store.load_completed_bundle()
    with pytest.raises(Round74TerminalReuseError, match="already consumed"):
        store.reserve(_preaccess())


def test_terminal_bundle_rejects_dataset_or_report_tampering(tmp_path) -> None:
    store = Round74TerminalOneUseStore(tmp_path / "terminal.sqlite3")
    claim = store.reserve(_preaccess())
    bundle = _result_bundle(claim)
    tampered = dict(bundle)
    tampered_report = dict(tampered["sealed_report"])
    tampered_report["result_outcome"] = "candidate_passed_predeclared_gates"
    tampered["sealed_report"] = tampered_report

    with pytest.raises(ValueError, match="result bundle differs"):
        validate_round74_terminal_result_bundle(tampered)


def test_terminal_bundle_rejects_rehashed_cross_identity_tampering(tmp_path) -> None:
    store = Round74TerminalOneUseStore(tmp_path / "terminal.sqlite3")
    claim = store.reserve(_preaccess())
    tampered = _result_bundle(claim)
    report = dict(tampered["sealed_report"])
    report["profile"] = "aggressive"
    report.pop("report_sha256")
    report["report_sha256"] = _sha(report)
    sealed_claim = dict(tampered["finalized_sealed_claim"])
    sealed_claim["profile"] = "aggressive"
    sealed_claim["result_sha256"] = report["report_sha256"]
    sealed_claim.pop("claim_sha256")
    sealed_claim["claim_sha256"] = _sha(sealed_claim)
    tampered["sealed_report"] = report
    tampered["finalized_sealed_claim"] = sealed_claim
    tampered.pop("bundle_sha256")
    tampered["bundle_sha256"] = _sha(tampered)

    with pytest.raises(ValueError, match="cross-binding differs"):
        validate_round74_terminal_result_bundle(tampered)


def test_terminal_recovery_rejects_database_payload_tampering(tmp_path) -> None:
    path = tmp_path / "terminal.sqlite3"
    store = Round74TerminalOneUseStore(path)
    claim = store.reserve(_preaccess())
    store.finalize_success(claim, _result_bundle(claim))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE terminal_access SET result_json = replace(result_json, ?, ?)",
            (
                "candidate_failed_predeclared_gates",
                "candidate_passed_predeclared_gates",
            ),
        )

    with pytest.raises(ValueError, match="result bundle differs"):
        store.load_completed_bundle()
