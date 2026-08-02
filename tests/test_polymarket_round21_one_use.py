from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round21_one_use as one_use_module
from simple_ai_trading.polymarket_round21_one_use import (
    Round21OneUseStore,
    Round21PretestManifest,
    build_round21_pretest_manifest,
    create_round21_one_use_claim,
    execute_round21_one_use,
)
from simple_ai_trading.polymarket_round21_sealed import (
    Round21SealedEconomicResult,
    Round21SealedPredictiveResult,
    build_round21_sealed_evaluation_result,
)


CONTROL_IDS = (
    "executable_market_prior_calibrated",
    "executable_market_prior_raw",
    "structural_probability_calibrated",
    "structural_probability_raw",
    "training_prevalence",
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _core_publication() -> dict[str, object]:
    return {
        "manifest_sha256": _sha("core-publication"),
        "sealed_test_population_manifest_sha256": _sha("test-population"),
    }


def _pretest(*, ai_model: str | None = None) -> Round21PretestManifest:
    ai_digest = None if ai_model is None else _sha("ai-model")
    ai_comparison = None if ai_model is None else _sha("ai-development-comparison")
    provisional = Round21PretestManifest(
        created_at_ms=1_700_000_000_000,
        selected_population_layer="core",
        core_corpus_publication_manifest_sha256=_sha("core-publication"),
        optional_campaign_terminal_sha256=None,
        sealed_test_population_manifest_sha256=_sha("test-population"),
        development_model_artifact_sha256=_sha("model-artifact"),
        development_economic_matrix_sha256=_sha("economic-matrix"),
        development_optional_comparison_sha256=None,
        development_ai_selection_sha256=_sha("ai-selection"),
        nominated_ai_model=ai_model,
        nominated_ai_model_digest=ai_digest,
        nominated_ai_comparison_sha256=ai_comparison,
        repository_commit_oid="a" * 40,
        repository_tree_oid="b" * 40,
        repository_file_sha256={
            relative: _sha(relative) for relative in one_use_module._REQUIRED_FILES
        },
        manifest_sha256=_sha("placeholder"),
    )
    return replace(
        provisional,
        manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _predictive() -> Round21SealedPredictiveResult:
    diagnostics = {
        "condition_count": 1_800,
        "condition_equal_log_loss": 0.50,
        "condition_equal_brier_score": 0.20,
        "log_loss_standard_error": 0.01,
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "expected_calibration_error": 0.02,
        "balanced_accuracy": 0.60,
        "matthews_correlation_coefficient": 0.20,
    }
    improvement = {
        "condition_count": 1_800,
        "mean": 0.02,
        "lower_95": 0.01,
        "upper_95": 0.03,
    }
    provisional = Round21SealedPredictiveResult(
        population_layer="core",
        model_artifact_sha256=_sha("model-artifact"),
        test_dataset_sha256=_sha("test-dataset"),
        test_target_manifest_sha256=_sha("test-targets"),
        probability_batch_sha256=_sha("probabilities"),
        resolved_condition_count=1_800,
        calendar_day_count=7,
        candidate_metrics=diagnostics,
        control_metrics={name: diagnostics for name in CONTROL_IDS},
        paired_improvements={
            name: {"log_loss": improvement, "brier": improvement}
            for name in CONTROL_IDS
        },
        gate_passed=True,
        reasons=(),
        result_sha256=_sha("placeholder"),
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _economic() -> Round21SealedEconomicResult:
    provisional = Round21SealedEconomicResult(
        matrix_sha256=_sha("sealed-economic-matrix"),
        ledger_count=81,
        qualified_ledger_count=81,
        ledger_sha256=tuple(_sha(f"ledger-{index}") for index in range(81)),
        gate_passed=True,
        reasons=(),
        result_sha256=_sha("placeholder"),
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _sealed_result(claim, access: str, *, population_sha256: str | None = None):
    return build_round21_sealed_evaluation_result(
        claim_sha256=claim.claim_sha256,
        test_access_sha256=access,
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=(
            claim.sealed_test_population_manifest_sha256
            if population_sha256 is None
            else population_sha256
        ),
        predictive=_predictive(),
        economic=_economic(),
    )


def test_round21_one_use_claim_cannot_predate_pretest() -> None:
    pretest = _pretest()

    with pytest.raises(ValueError, match="predates"):
        create_round21_one_use_claim(
            pretest,
            opened_at_ms=pretest.created_at_ms - 1,
        )


def test_round21_one_use_success_is_durable_and_cannot_reopen(tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "sealed.sqlite3"
    callback_count = 0

    def evaluator(access: str):
        nonlocal callback_count
        callback_count += 1
        return _sealed_result(claim, access)

    result = execute_round21_one_use(
        store_path=store_path,
        claim=claim,
        evaluator=evaluator,
    )
    assert result.candidate_accepted is True
    assert callback_count == 1
    with Round21OneUseStore(store_path) as store:
        snapshot = store.snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["test_access_consumed"] is True
    assert snapshot["event_count"] == 3

    with pytest.raises(RuntimeError, match="cannot be reopened"):
        execute_round21_one_use(
            store_path=store_path,
            claim=claim,
            evaluator=evaluator,
        )
    assert callback_count == 1


def test_round21_one_use_failure_is_terminal(tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "failed.sqlite3"
    callback_count = 0

    def evaluator(_access: str):
        nonlocal callback_count
        callback_count += 1
        raise RuntimeError("sealed evaluator failed")

    with pytest.raises(RuntimeError, match="sealed evaluator failed"):
        execute_round21_one_use(
            store_path=store_path,
            claim=claim,
            evaluator=evaluator,
        )
    with Round21OneUseStore(store_path) as store:
        snapshot = store.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["test_access_consumed"] is True
    assert snapshot["event_count"] == 3

    with pytest.raises(RuntimeError, match="cannot be reopened"):
        execute_round21_one_use(
            store_path=store_path,
            claim=claim,
            evaluator=evaluator,
        )
    assert callback_count == 1


def test_round21_one_use_rejects_wrong_population_or_ai_identity(tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(ai_model="qwen3:8b"),
        opened_at_ms=1_700_000_000_001,
    )
    with Round21OneUseStore(tmp_path / "binding.sqlite3") as store:
        store.open_claim(claim)
        access = store.consume_test_access(
            claim,
            observed_at_ms=claim.opened_at_ms + 1,
        )
        wrong_population = _sealed_result(
            claim,
            access,
            population_sha256=_sha("other-test-population"),
        )
        with pytest.raises(ValueError, match="result claim differs"):
            store.complete(
                claim,
                wrong_population,
                observed_at_ms=claim.opened_at_ms + 2,
            )
        no_ai = _sealed_result(claim, access)
        with pytest.raises(ValueError, match="result claim differs"):
            store.complete(
                claim,
                no_ai,
                observed_at_ms=claim.opened_at_ms + 2,
            )


def test_round21_one_use_detects_event_tampering(tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "tampered.sqlite3"
    with Round21OneUseStore(store_path) as store:
        store.open_claim(claim)
    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE round21_one_use_event SET event_json = '{}' WHERE sequence = 1"
        )
    with Round21OneUseStore(store_path) as store:
        with pytest.raises(ValueError, match="event chain differs"):
            store.snapshot()


def test_round21_one_use_strict_json_and_git_attestation(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate keys"):
        one_use_module._strict_json('{"a":1,"a":2}', label="fixture")
    with pytest.raises(ValueError, match="contains NaN"):
        one_use_module._strict_json('{"a":NaN}', label="fixture")
    with pytest.raises(ValueError, match="is invalid"):
        one_use_module._strict_json("{x", label="fixture")
    with pytest.raises(ValueError, match="not an object"):
        one_use_module._strict_json("[]", label="fixture")
    with pytest.raises(ValueError, match="size differs"):
        one_use_module._strict_json("", label="fixture")
    with pytest.raises(ValueError, match="Git operation failed"):
        one_use_module._git(tmp_path, "not-a-real-git-subcommand")

    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in one_use_module._REQUIRED_FILES:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="ascii")
    for arguments in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-m", "fixture"),
    ):
        subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
        )
    commit, tree, files = one_use_module._repository_attestation(repository)
    assert len(commit) == 40
    assert len(tree) == 40
    assert set(files) == set(one_use_module._REQUIRED_FILES)

    first = repository / one_use_module._REQUIRED_FILES[0]
    first.write_text("dirty", encoding="ascii")
    with pytest.raises(ValueError, match="clean worktree"):
        one_use_module._repository_attestation(repository)


def test_round21_pretest_builder_binds_qualified_development(
    monkeypatch,
    tmp_path,
) -> None:
    profiles = ("conservative", "regular", "aggressive")
    scenarios = tuple(f"scenario-{index}" for index in range(27))

    class Replay:
        def __init__(self, profile: str, scenario: str) -> None:
            self.profile = profile
            self.scenario = scenario
            self.economic_gate_passed = True

        def validated(self):
            return self

    matrix = tuple(Replay(profile, scenario) for profile in profiles for scenario in scenarios)
    artifact = {
        "artifact_sha256": _sha("artifact"),
        "layers": {
            "core": {"comparison": {"predictive_development_accepted": True}}
        },
    }
    ai = SimpleNamespace(
        selection_sha256=_sha("ai-selection"),
        nominated_model=None,
        nominated_model_digest=None,
        nominated_comparison_sha256=None,
    )
    ai.validated = lambda: ai
    monkeypatch.setattr(one_use_module, "load_round21_sealed_design", lambda _root: {})
    monkeypatch.setattr(
        one_use_module,
        "validate_round21_core_publication_boundary",
        lambda _directory: _core_publication(),
    )
    monkeypatch.setattr(
        one_use_module,
        "validate_round21_development_artifact",
        lambda _artifact: artifact,
    )
    monkeypatch.setattr(
        one_use_module,
        "round21_replay_matrix_sha256",
        lambda _matrix: _sha("matrix"),
    )
    monkeypatch.setattr(
        one_use_module,
        "_repository_attestation",
        lambda _root: (
            "a" * 40,
            "b" * 40,
            {
                relative: _sha(relative)
                for relative in one_use_module._REQUIRED_FILES
            },
        ),
    )

    manifest = build_round21_pretest_manifest(
        tmp_path,
        selected_population_layer="core",
        core_corpus_publication_directory=tmp_path / "core-publication",
        optional_campaign_terminal_sha256=None,
        development_model_artifact={},
        development_economic_matrix=matrix,
        development_optional_comparison=None,
        development_ai_selection=ai,
        created_at_ms=1_700_000_000_000,
    )
    assert manifest.development_economic_matrix_sha256 == _sha("matrix")
    assert manifest.selected_population_layer == "core"
    assert manifest.asdict()["manifest_sha256"] == manifest.manifest_sha256

    with pytest.raises(RuntimeError, match="economic matrix did not qualify"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="core",
            core_corpus_publication_directory=tmp_path / "core-publication",
            optional_campaign_terminal_sha256=None,
            development_model_artifact={},
            development_economic_matrix=matrix[:-1],
            development_optional_comparison=None,
            development_ai_selection=ai,
        )


def test_round21_store_rejects_invalid_transitions(tmp_path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="store path differs"):
        Round21OneUseStore(directory)

    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    other_pretest = replace(
        _pretest(),
        sealed_test_population_manifest_sha256=_sha("other-population"),
        manifest_sha256=_sha("placeholder"),
    )
    other_pretest = replace(
        other_pretest,
        manifest_sha256=_canonical_sha256(other_pretest.identity_payload()),
    ).validated()
    other = create_round21_one_use_claim(
        other_pretest,
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "transitions.sqlite3"
    with Round21OneUseStore(store_path) as store:
        assert store.snapshot()["status"] == "empty"
        store.open_claim(claim)
        with pytest.raises(RuntimeError, match="already has a claim"):
            store.open_claim(other)
        with pytest.raises(ValueError, match="access time differs"):
            store.consume_test_access(
                claim,
                observed_at_ms=claim.opened_at_ms - 1,
            )
        access = store.consume_test_access(
            claim,
            observed_at_ms=claim.opened_at_ms + 1,
        )
        with pytest.raises(RuntimeError, match="already consumed"):
            store.consume_test_access(
                claim,
                observed_at_ms=claim.opened_at_ms + 2,
            )
        result = _sealed_result(claim, access)
        with pytest.raises(ValueError, match="result time differs"):
            store.complete(
                claim,
                result,
                observed_at_ms=claim.opened_at_ms,
            )
        store.complete(
            claim,
            result,
            observed_at_ms=claim.opened_at_ms + 2,
        )
        with pytest.raises(RuntimeError, match="already completed"):
            store.fail(
                claim,
                reason="late failure",
                observed_at_ms=claim.opened_at_ms + 3,
            )


def test_round21_execute_rejects_invalid_callback_result(tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "invalid-callback.sqlite3"

    with pytest.raises(TypeError, match="invalid result type"):
        execute_round21_one_use(
            store_path=store_path,
            claim=claim,
            evaluator=lambda _access: None,
        )
    with Round21OneUseStore(store_path) as store:
        assert store.snapshot()["status"] == "failed"


def test_round21_pretest_and_claim_validation_reject_drift() -> None:
    pretest = _pretest()
    with pytest.raises(ValueError, match="pretest manifest differs"):
        replace(pretest, created_at_ms=0).validated()
    with pytest.raises(ValueError, match="digest differs"):
        one_use_module._digest("bad", name="fixture")

    claim = create_round21_one_use_claim(
        pretest,
        opened_at_ms=pretest.created_at_ms + 1,
    )
    with pytest.raises(ValueError, match="claim differs"):
        replace(claim, opened_at_ms=0).validated()
    malformed = claim.asdict()
    malformed.pop("opened_at_ms")
    with pytest.raises(ValueError, match="claim schema differs"):
        one_use_module._claim_from_mapping(malformed)
    malformed = claim.asdict()
    malformed["opened_at_ms"] = None
    with pytest.raises(ValueError, match="claim schema differs"):
        one_use_module._claim_from_mapping(malformed)


def test_round21_pretest_builder_rejects_selection_drift(monkeypatch, tmp_path) -> None:
    profiles = ("conservative", "regular", "aggressive")
    scenarios = tuple(f"scenario-{index}" for index in range(27))

    class Replay:
        economic_gate_passed = True

        def __init__(self, profile: str, scenario: str) -> None:
            self.profile = profile
            self.scenario = scenario

        def validated(self):
            return self

    class Optional:
        challenger_layer = "core_spot"
        all_replays_accepted = False
        comparison_sha256 = _sha("optional-comparison")

        def validated(self):
            return self

    matrix = tuple(Replay(profile, scenario) for profile in profiles for scenario in scenarios)
    artifact = {
        "artifact_sha256": _sha("artifact"),
        "layers": {
            "core": {"comparison": {"predictive_development_accepted": True}},
            "core_spot": {
                "comparison": {"predictive_development_accepted": True}
            },
        },
    }
    ai = SimpleNamespace(
        selection_sha256=_sha("ai-selection"),
        nominated_model=None,
        nominated_model_digest=None,
        nominated_comparison_sha256=None,
    )
    ai.validated = lambda: ai
    monkeypatch.setattr(one_use_module, "load_round21_sealed_design", lambda _root: {})
    monkeypatch.setattr(
        one_use_module,
        "validate_round21_core_publication_boundary",
        lambda _directory: _core_publication(),
    )
    monkeypatch.setattr(
        one_use_module,
        "validate_round21_development_artifact",
        lambda _artifact: artifact,
    )
    monkeypatch.setattr(
        one_use_module,
        "_repository_attestation",
        lambda _root: (
            "a" * 40,
            "b" * 40,
            {
                relative: _sha(relative)
                for relative in one_use_module._REQUIRED_FILES
            },
        ),
    )

    common = {
        "core_corpus_publication_directory": tmp_path / "core-publication",
        "development_model_artifact": {},
        "development_economic_matrix": matrix,
        "development_ai_selection": ai,
    }
    with pytest.raises(ValueError, match="selected layer differs"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="bad",
            optional_campaign_terminal_sha256=None,
            development_optional_comparison=None,
            **common,
        )
    artifact["layers"]["core"]["comparison"][
        "predictive_development_accepted"
    ] = False
    with pytest.raises(RuntimeError, match="predictive layer did not qualify"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="core",
            optional_campaign_terminal_sha256=None,
            development_optional_comparison=None,
            **common,
        )
    artifact["layers"]["core"]["comparison"][
        "predictive_development_accepted"
    ] = True
    with pytest.raises(ValueError, match="optional comparison differs"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="core",
            optional_campaign_terminal_sha256=None,
            development_optional_comparison=Optional(),
            **common,
        )
    with pytest.raises(RuntimeError, match="optional layer did not qualify"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="core_spot",
            optional_campaign_terminal_sha256=_sha("optional-terminal"),
            development_optional_comparison=Optional(),
            **common,
        )
    with pytest.raises(ValueError, match="optional campaign binding differs"):
        build_round21_pretest_manifest(
            tmp_path,
            selected_population_layer="core",
            optional_campaign_terminal_sha256=_sha("optional-terminal"),
            development_optional_comparison=None,
            **common,
        )


def test_round21_repository_and_store_detect_persisted_drift(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(one_use_module, "_git", lambda _repository, *_args: "")
    with pytest.raises(ValueError, match="file is unavailable"):
        one_use_module._repository_attestation(tmp_path)

    metadata_path = tmp_path / "metadata.sqlite3"
    with Round21OneUseStore(metadata_path):
        pass
    with sqlite3.connect(metadata_path) as connection:
        connection.execute(
            "UPDATE round21_one_use_metadata SET schema_version = 'tampered'"
        )
    with pytest.raises(ValueError, match="store schema differs"):
        Round21OneUseStore(metadata_path)

    orphan_path = tmp_path / "orphan.sqlite3"
    with Round21OneUseStore(orphan_path):
        pass
    with sqlite3.connect(orphan_path) as connection:
        connection.execute(
            """
            INSERT INTO round21_one_use_event(
                event_sha256, previous_event_sha256, event_json
            ) VALUES (?, '', '{}')
            """,
            (_sha("orphan"),),
        )
    with Round21OneUseStore(orphan_path) as store:
        with pytest.raises(ValueError, match="events exist without a claim"):
            store.snapshot()


def test_round21_store_rejects_result_failure_and_status_drift(monkeypatch, tmp_path) -> None:
    claim = create_round21_one_use_claim(
        _pretest(),
        opened_at_ms=1_700_000_000_001,
    )
    store_path = tmp_path / "drift.sqlite3"
    with Round21OneUseStore(store_path) as store:
        store.open_claim(claim)
        with pytest.raises(ValueError, match="claim is unavailable"):
            store._row(
                create_round21_one_use_claim(
                    _pretest(),
                    opened_at_ms=1_700_000_000_002,
                )
            )
        with pytest.raises(ValueError, match="failure differs"):
            store.fail(claim, reason="", observed_at_ms=claim.opened_at_ms)
        access = store.consume_test_access(
            claim,
            observed_at_ms=claim.opened_at_ms + 2,
        )
        result = _sealed_result(claim, access)
        monkeypatch.setattr(one_use_module, "_MAXIMUM_JSON_BYTES", 1)
        with pytest.raises(ValueError, match="result is too large"):
            store.complete(
                claim,
                result,
                observed_at_ms=claim.opened_at_ms + 3,
            )
        monkeypatch.setattr(one_use_module, "_MAXIMUM_JSON_BYTES", 4 * 1024 * 1024)
        wrong_access = _sealed_result(claim, _sha("wrong-access"))
        with pytest.raises(ValueError, match="result access differs"):
            store.complete(
                claim,
                wrong_access,
                observed_at_ms=claim.opened_at_ms + 3,
            )
        with pytest.raises(ValueError, match="failure time differs"):
            store.fail(
                claim,
                reason="too early",
                observed_at_ms=claim.opened_at_ms + 1,
            )
        store.fail(
            claim,
            reason="terminal fixture",
            observed_at_ms=claim.opened_at_ms + 3,
        )

    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE round21_one_use_claim SET failure_json = '{}' WHERE singleton = 1"
        )
    with Round21OneUseStore(store_path) as store:
        with pytest.raises(ValueError, match="stored one-use failure differs"):
            store.snapshot()
