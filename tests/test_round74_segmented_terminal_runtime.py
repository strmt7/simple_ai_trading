from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.compute import BackendInfo
import simple_ai_trading.round74_segmented_terminal_runtime as subject
from simple_ai_trading.round74_terminal_one_use import Round74TerminalOneUseStore


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


def _paths(tmp_path: Path) -> dict[str, object]:
    databases = (tmp_path / "capture-a.duckdb", tmp_path / "capture-b.duckdb")
    result = {
        "repository": tmp_path,
        "database_paths": databases,
        "plan_path": tmp_path / "plan.json",
        "state_root": tmp_path / "state",
        "recovery_outcome_directory": tmp_path / "recovery",
        "test_target_assembly_directory": tmp_path / "targets",
        "source_artifact_root": tmp_path / "sources",
        "development_bundle_path": tmp_path / "development.json",
        "pretest_policy_path": tmp_path / "policy.json",
        "ai_qualification_path": tmp_path / "qualification.json",
        "one_use_store_path": tmp_path / "one-use.sqlite3",
        "sealed_ledger_path": tmp_path / "sealed.sqlite3",
        "output_path": tmp_path / "result.json",
    }
    for name in (
        "state_root",
        "recovery_outcome_directory",
        "test_target_assembly_directory",
        "source_artifact_root",
    ):
        path = result[name]
        assert isinstance(path, Path)
        path.mkdir()
    for database in databases:
        database.write_bytes(b"x")
    for name in (
        "plan_path",
        "development_bundle_path",
        "pretest_policy_path",
        "ai_qualification_path",
    ):
        path = result[name]
        assert isinstance(path, Path)
        path.write_bytes(b"x")
    return result


def _install_terminal_fakes(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, object],
    events: list[str],
) -> tuple[object, ...]:
    run_ids = tuple(f"{index:032x}" for index in range(1, 25))
    databases = paths["database_paths"]
    assert isinstance(databases, tuple)
    manifests = ("b" * 64, "c" * 64)
    for run_id in run_ids:
        target_directory = paths["test_target_assembly_directory"]
        assert isinstance(target_directory, Path)
        (target_directory / f"{run_id}.json").write_bytes(b"sealed")
    partition = SimpleNamespace(partition_sha256="3" * 64)
    coverage_value = SimpleNamespace(
        plan=SimpleNamespace(plan_sha256="1" * 64),
        coverage=SimpleNamespace(
            coverage_sha256="2" * 64,
            partition=partition,
        ),
        test_bindings_by_run_id=lambda: {
            run_id: SimpleNamespace(binding_sha256="4" * 64) for run_id in run_ids
        },
    )
    population = SimpleNamespace(
        test_run_ids=run_ids,
        population_sha256="4" * 64,
        optimization_population="eligible_target",
    )
    action_selection = SimpleNamespace(
        selection_sha256="8" * 64,
        profile="conservative",
    )
    final_configuration = SimpleNamespace(
        action_selection=action_selection,
        configuration_sha256="9" * 64,
    )
    calibration = SimpleNamespace(calibration_sha256="7" * 64)
    bundle = SimpleNamespace(
        bundle_sha256="5" * 64,
        pretest_policy_sha256="6" * 64,
        feature_scaler_sha256="a" * 64,
        probability_calibration=calibration,
    )
    qualification = SimpleNamespace(
        qualification_passed=True,
        profile="conservative",
        candidate_model_manifest_sha256=manifests,
        model_manifest_sha256=manifests,
        action_selection_sha256="8" * 64,
        final_action_configuration_sha256="9" * 64,
        pretest_policy_sha256="6" * 64,
        probability_calibration_sha256="7" * 64,
        qualification_sha256="d" * 64,
    )
    bindings = tuple(
        SimpleNamespace(
            manifest=SimpleNamespace(manifest_sha256=manifest),
            runtime=SimpleNamespace(endpoint="http://127.0.0.1:11434"),
            model_name=f"model-{index}",
        )
        for index, manifest in enumerate(manifests)
    )
    scaler = SimpleNamespace(scaler_sha256="a" * 64)

    def dataset_mapping() -> dict[str, object]:
        one_use_path = paths["one_use_store_path"]
        assert isinstance(one_use_path, Path)
        access = Round74TerminalOneUseStore(one_use_path).claim()
        assert access is not None
        test_access_sha256 = _sha(
            {
                "pretest_model_policy_sha256": access.preaccess.pretest_policy_sha256,
                "test_unlock_sha256": access.test_unlock_sha256,
            }
        )
        return {
            "schema_version": "round-074-sealed-dataset-identity-test",
            "test_access_sha256": test_access_sha256,
            "partition_sha256": "3" * 64,
            "scaler_sha256": "a" * 64,
            "optimization_population": "eligible_target",
            "test_population_sha256": "4" * 64,
            "test_run_ids": list(run_ids),
        }

    class DatasetIdentity:
        @property
        def dataset_sha256(self) -> str:
            return _sha(dataset_mapping())

        @staticmethod
        def as_dict() -> dict[str, object]:
            return dataset_mapping()

    dataset_identity = DatasetIdentity()

    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda databases, **_kwargs: tuple(1 for _ in databases),
    )
    database_by_run_id = {
        run_id: databases[index % len(databases)]
        for index, run_id in enumerate(run_ids)
    }
    monkeypatch.setattr(
        subject,
        "_route_test_run_databases",
        lambda *_args, **_kwargs: (
            database_by_run_id,
            "0" * 64,
            databases,
        ),
    )
    monkeypatch.setattr(
        subject,
        "load_round74_segmented_terminal_coverage",
        lambda **_kwargs: coverage_value,
    )
    monkeypatch.setattr(
        subject,
        "build_round74_segmented_test_population",
        lambda *_args, **_kwargs: population,
    )
    monkeypatch.setattr(
        subject,
        "load_round74_development_policy_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        subject,
        "select_round74_final_action_configuration",
        lambda *_args, **_kwargs: final_configuration,
    )
    monkeypatch.setattr(
        subject,
        "load_round74_ai_pretest_qualification",
        lambda *_args, **_kwargs: qualification,
    )
    monkeypatch.setattr(
        subject,
        "load_round74_pretest_scaler",
        lambda *_args, **_kwargs: scaler,
    )
    monkeypatch.setattr(
        subject,
        "_policy_window_representation",
        lambda *_args, **_kwargs: "per_symbol",
    )
    monkeypatch.setattr(
        subject,
        "_preflight_backend",
        lambda *_args, **_kwargs: (
            BackendInfo(
                requested="auto",
                kind="directml",
                device="privateuseone:0",
                vendor="AMD",
                reason="",
                selection="test",
            ),
            "e" * 64,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_preflight_ai_models",
        lambda *_args, **_kwargs: ("e" * 64, "f" * 64),
    )

    def load_targets(*_args, **_kwargs):
        one_use_path = paths["one_use_store_path"]
        assert isinstance(one_use_path, Path)
        claim = Round74TerminalOneUseStore(one_use_path).claim()
        assert claim is not None and claim.status == "reserved"
        events.append("targets_loaded_after_reservation")
        return {run_id: object() for run_id in run_ids}

    monkeypatch.setattr(subject, "_load_test_target_assemblies", load_targets)

    monkeypatch.setattr(
        subject,
        "_assemble_test_batches_from_databases",
        lambda **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        subject,
        "build_round74_segmented_sealed_dataset_identity",
        lambda *_args, **_kwargs: dataset_identity,
    )
    monkeypatch.setattr(
        subject,
        "Round74PreparedSealedAIReviewProvider",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject,
        "Round74ShardedExecutionReplayProvider",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject,
        "Round74SealedEvaluationLedger",
        lambda *_args, **_kwargs: object(),
    )

    def evaluate(_identity, *, test_batch_loader, **_kwargs):
        assert len(test_batch_loader(claim=object())) == 1
        events.append("sealed_evaluated")
        one_use_path = paths["one_use_store_path"]
        assert isinstance(one_use_path, Path)
        access = Round74TerminalOneUseStore(one_use_path).claim()
        assert access is not None
        dataset = dataset_mapping()
        dataset_sha256 = _sha(dataset)
        test_access_sha256 = str(dataset["test_access_sha256"])
        report_mapping: dict[str, object] = {
            "schema_version": "round-074-sealed-evaluation-test",
            "reservation_id": "1" * 64,
            "dataset_sha256": dataset_sha256,
            "test_access_sha256": test_access_sha256,
            "pretest_policy_sha256": "6" * 64,
            "probability_calibration_sha256": "7" * 64,
            "action_selection_sha256": "8" * 64,
            "final_action_configuration_sha256": "9" * 64,
            "ai_pretest_qualification_sha256": "d" * 64,
            "profile": "conservative",
            "optimization_population": "eligible_target",
            "result_outcome": "candidate_failed_predeclared_gates",
        }
        report_mapping["report_sha256"] = _sha(report_mapping)
        sealed_claim_mapping: dict[str, object] = {
            "schema_version": "round-074-sealed-claim-test",
            "reservation_id": "1" * 64,
            "dataset_sha256": dataset_sha256,
            "test_access_sha256": test_access_sha256,
            "partition_sha256": "3" * 64,
            "scaler_sha256": "a" * 64,
            "pretest_policy_sha256": "6" * 64,
            "probability_calibration_sha256": "7" * 64,
            "action_selection_sha256": "8" * 64,
            "final_action_configuration_sha256": "9" * 64,
            "ai_pretest_qualification_sha256": "d" * 64,
            "profile": "conservative",
            "optimization_population": "eligible_target",
            "test_population_sha256": "4" * 64,
            "test_run_ids": list(run_ids),
            "status": "complete",
            "result_sha256": report_mapping["report_sha256"],
        }
        sealed_claim_mapping["claim_sha256"] = _sha(sealed_claim_mapping)
        report = SimpleNamespace(
            report_sha256=report_mapping["report_sha256"],
            result_outcome="candidate_failed_predeclared_gates",
            qualified_configuration=(),
            as_dict=lambda: dict(report_mapping),
        )
        sealed_claim = SimpleNamespace(as_dict=lambda: dict(sealed_claim_mapping))
        return SimpleNamespace(report=report, finalized_claim=sealed_claim)

    monkeypatch.setattr(subject, "evaluate_round74_sealed_once", evaluate)
    return bindings


def test_terminal_ai_preflight_binds_compatible_runtime_patch() -> None:
    bindings = tuple(
        SimpleNamespace(
            validate=lambda: None,
            manifest=SimpleNamespace(
                manifest_sha256=manifest,
                model_artifact_sha256=artifact,
                runtime_version="0.32.4",
            ),
            runtime=SimpleNamespace(endpoint="http://127.0.0.1:11434"),
            model_name=f"model-{index}",
        )
        for index, (manifest, artifact) in enumerate(
            (("1" * 64, "3" * 64), ("2" * 64, "4" * 64))
        )
    )
    version_calls = 0

    def version(_endpoint: str, _timeout: float) -> str:
        nonlocal version_calls
        version_calls += 1
        return "0.32.5"

    digests = subject._preflight_ai_models(
        bindings,
        resolver=lambda _endpoint, model, _timeout: (
            "3" * 64 if model == "model-0" else "4" * 64,
            "5" * 64 if model == "model-0" else "6" * 64,
        ),
        runtime_version_resolver=version,
    )

    assert len(digests) == 2
    assert len(set(digests)) == 2
    assert version_calls == 1


def test_terminal_ai_preflight_accepts_one_qualified_model() -> None:
    binding = SimpleNamespace(
        validate=lambda: None,
        manifest=SimpleNamespace(
            manifest_sha256="1" * 64,
            model_artifact_sha256="2" * 64,
            runtime_version="0.32.4",
        ),
        runtime=SimpleNamespace(endpoint="http://127.0.0.1:11434"),
        model_name="model",
    )

    digests = subject._preflight_ai_models(
        (binding,),
        resolver=lambda *_args: ("2" * 64, "3" * 64),
        runtime_version_resolver=lambda *_args: "0.32.4",
    )

    assert len(digests) == 1
    assert len(digests[0]) == 64


@pytest.mark.parametrize("observed", ["0.32.3", "0.33.0", "invalid"])
def test_terminal_ai_preflight_rejects_incompatible_runtime(observed: str) -> None:
    binding = SimpleNamespace(
        validate=lambda: None,
        manifest=SimpleNamespace(
            manifest_sha256="1" * 64,
            model_artifact_sha256="2" * 64,
            runtime_version="0.32.4",
        ),
        runtime=SimpleNamespace(endpoint="http://127.0.0.1:11434"),
        model_name="model",
    )

    with pytest.raises(ValueError, match="runtime version differs"):
        subject._preflight_ai_models(
            (binding,),
            resolver=lambda *_args: ("2" * 64, "3" * 64),
            runtime_version_resolver=lambda *_args: observed,
        )


def test_terminal_database_router_covers_each_run_once_across_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    databases = (tmp_path / "shard-a.duckdb", tmp_path / "shard-b.duckdb")
    for database in databases:
        database.write_bytes(b"store")
    run_ids = tuple(f"{index:032x}" for index in range(1, 5))
    rows_by_database = {
        databases[0]: (run_ids[0], run_ids[2]),
        databases[1]: (run_ids[1], run_ids[3]),
    }

    class Result:
        def __init__(self, rows: tuple[str, ...]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[str]]:
            return [(run_id,) for run_id in self.rows]

    class Connection:
        def __init__(self, rows: tuple[str, ...]) -> None:
            self.rows = rows

        def execute(self, statement: str) -> Result:
            assert "impact_capture_run" in statement
            return Result(self.rows)

    class Store:
        def __init__(self, path: Path, **_kwargs: object) -> None:
            self.path = path

        def __enter__(self) -> Store:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def connect(self) -> Connection:
            return Connection(rows_by_database[self.path])

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda databases, **_kwargs: tuple(1 for _ in databases),
    )
    routed, route_sha256, used = subject._route_test_run_databases(
        databases,
        test_run_ids=run_ids,
        memory_limit="4GB",
        database_threads=2,
    )
    _reverse, reverse_sha256, _reverse_used = subject._route_test_run_databases(
        tuple(reversed(databases)),
        test_run_ids=run_ids,
        memory_limit="4GB",
        database_threads=2,
    )

    assert tuple(routed) == run_ids
    assert routed[run_ids[0]] == databases[0]
    assert routed[run_ids[1]] == databases[1]
    assert used == databases
    assert len(route_sha256) == 64
    assert reverse_sha256 == route_sha256

    rows_by_database[databases[1]] = (run_ids[0], run_ids[1], run_ids[3])
    with pytest.raises(ValueError, match="multiple stores"):
        subject._route_test_run_databases(
            databases,
            test_run_ids=run_ids,
            memory_limit="4GB",
            database_threads=2,
        )


def test_terminal_sharded_replay_opens_one_store_at_a_time_and_restores_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    databases = (tmp_path / "shard-a.duckdb", tmp_path / "shard-b.duckdb")
    for database in databases:
        database.write_bytes(b"store")
    run_ids = ("1" * 32, "2" * 32)
    manifests = ("a" * 64, "b" * 64)
    partition = SimpleNamespace(
        partition_sha256="c" * 64,
        entries=tuple(
            SimpleNamespace(run_id=run_id, role="test") for run_id in run_ids
        ),
        validate=lambda: None,
    )
    open_count = 0
    maximum_open_count = 0
    opened: list[Path] = []

    class Store:
        def __init__(self, path: Path, **_kwargs: object) -> None:
            self.path = path

        def __enter__(self) -> Store:
            nonlocal open_count, maximum_open_count
            open_count += 1
            maximum_open_count = max(maximum_open_count, open_count)
            opened.append(self.path)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal open_count
            open_count -= 1

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda databases, **_kwargs: tuple(1 for _ in databases),
    )

    def replay(_store: object, *, instructions, **_kwargs: object):
        return tuple(
            SimpleNamespace(row_index=row.row_index, validate=lambda: None)
            for row in instructions
        )

    monkeypatch.setattr(subject, "replay_round74_ai_execution_store_run", replay)
    provider = subject.Round74ShardedExecutionReplayProvider(
        database_by_run_id={run_ids[0]: databases[0], run_ids[1]: databases[1]},
        partition=partition,
        assembly_by_run_id={run_id: object() for run_id in run_ids},
        memory_limit="4GB",
        database_threads=2,
    )
    action_sha256 = "d" * 64

    def instruction(manifest: str, run_id: str, row_index: int):
        return SimpleNamespace(
            model_manifest_sha256=manifest,
            partition_sha256=partition.partition_sha256,
            action_selection_sha256=action_sha256,
            run_id=run_id,
            row_index=row_index,
            validate=lambda: None,
        )

    instructions = {
        manifest: (
            instruction(manifest, run_ids[1], index * 10 + 2),
            instruction(manifest, run_ids[0], index * 10 + 1),
        )
        for index, manifest in enumerate(manifests)
    }
    claim = SimpleNamespace(
        status="reserved",
        partition_sha256=partition.partition_sha256,
        test_run_ids=run_ids,
        ai_manifest_sha256=manifests,
        action_selection_sha256=action_sha256,
        validate=lambda: None,
    )

    result = provider(claim=claim, instructions_by_manifest=instructions)

    assert maximum_open_count == 1
    assert open_count == 0
    assert opened == list(databases)
    for manifest in manifests:
        assert tuple(row.row_index for row in result[manifest]) == tuple(
            row.row_index for row in instructions[manifest]
        )


def test_terminal_sharded_batch_assembly_opens_one_store_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    databases = (tmp_path / "shard-a.duckdb", tmp_path / "shard-b.duckdb")
    for database in databases:
        database.write_bytes(b"store")
    run_ids = ("1" * 32, "2" * 32)
    entries = {
        run_id: SimpleNamespace(run_id=run_id, role="test") for run_id in run_ids
    }
    partition = SimpleNamespace(
        entries=tuple(entries.values()),
        entry=lambda run_id: entries[run_id],
        validate=lambda: None,
    )
    open_count = 0
    maximum_open_count = 0
    opened: list[Path] = []

    class Store:
        def __init__(self, path: Path, **_kwargs: object) -> None:
            self.path = path

        def __enter__(self) -> Store:
            nonlocal open_count, maximum_open_count
            open_count += 1
            maximum_open_count = max(maximum_open_count, open_count)
            opened.append(self.path)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal open_count
            open_count -= 1

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda databases, **_kwargs: tuple(1 for _ in databases),
    )
    monkeypatch.setattr(
        subject,
        "iter_round74_segmented_labeled_event_windows",
        lambda _store, *, binding, **_kwargs: iter((binding.run_id,)),
    )
    monkeypatch.setattr(
        subject,
        "select_round74_segmented_event_windows",
        lambda samples, **_kwargs: tuple(samples),
    )
    monkeypatch.setattr(
        subject,
        "round74_segmented_windows_per_symbol",
        lambda _entry: 1,
    )

    def build(samples, **_kwargs):
        run_id = tuple(samples)[0]
        return SimpleNamespace(
            role="test",
            run_id=(run_id, run_id, run_id),
            rows=3,
            validate=lambda: None,
        )

    monkeypatch.setattr(subject, "build_round74_event_training_batch", build)
    batches = subject._assemble_test_batches_from_databases(
        database_by_run_id={run_ids[0]: databases[0], run_ids[1]: databases[1]},
        test_run_ids=run_ids,
        partition=partition,
        bindings_by_run_id={
            run_id: SimpleNamespace(run_id=run_id) for run_id in run_ids
        },
        scaler=object(),
        target_assembly_by_run_id={run_id: object() for run_id in run_ids},
        pretest_model_policy_sha256="a" * 64,
        test_unlock_sha256="b" * 64,
        window_representation="per_symbol",
        memory_limit="4GB",
        database_threads=2,
    )

    assert maximum_open_count == 1
    assert open_count == 0
    assert opened == list(databases)
    assert tuple(batch.run_id[0] for batch in batches) == run_ids


def test_terminal_runtime_reserves_before_target_loading_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    events: list[str] = []
    bindings = _install_terminal_fakes(monkeypatch, paths, events)

    result = subject.run_round74_segmented_terminal_evaluation(
        **paths,
        terminal_observed_wall_ns=1_800_000_000_000_000_000,
        progress=lambda *_args, **_kwargs: None,
        model_bindings=bindings,
        provenance_resolver=lambda *_args: ("0" * 64, "1" * 64),
    )

    assert events == ["targets_loaded_after_reservation", "sealed_evaluated"]
    assert result["result_outcome"] == "candidate_failed_predeclared_gates"
    assert result["profitability_claim"] is False
    assert paths["output_path"].is_file()
    recovery_output = tmp_path / "recovered.json"
    recovery = subject.recover_round74_segmented_terminal_result(
        one_use_store_path=paths["one_use_store_path"],
        output_path=recovery_output,
    )
    assert recovery["sealed_test_reopened"] is False
    assert recovery["model_rerun"] is False
    assert recovery_output.read_bytes() == paths["output_path"].read_bytes()


def test_terminal_runtime_target_failure_is_final_and_not_retriable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    events: list[str] = []
    bindings = _install_terminal_fakes(monkeypatch, paths, events)

    def fail_after_reservation(*_args, **_kwargs):
        claim = Round74TerminalOneUseStore(paths["one_use_store_path"]).claim()
        assert claim is not None and claim.status == "reserved"
        raise RuntimeError("sealed target read stopped")

    monkeypatch.setattr(subject, "_load_test_target_assemblies", fail_after_reservation)
    arguments = {
        **paths,
        "terminal_observed_wall_ns": 1_800_000_000_000_000_000,
        "progress": lambda *_args, **_kwargs: None,
        "model_bindings": bindings,
        "provenance_resolver": lambda *_args: ("0" * 64, "1" * 64),
    }

    with pytest.raises(RuntimeError, match="sealed target read stopped"):
        subject.run_round74_segmented_terminal_evaluation(**arguments)

    claim = Round74TerminalOneUseStore(paths["one_use_store_path"]).claim()
    assert claim is not None and claim.status == "failed"
    with pytest.raises(RuntimeError, match="already consumed"):
        subject.run_round74_segmented_terminal_evaluation(**arguments)


def test_terminal_runtime_existing_output_fails_before_access_or_target_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths["output_path"].write_text("occupied", encoding="utf-8")
    target_called = False

    def target_loader(*_args, **_kwargs):
        nonlocal target_called
        target_called = True

    monkeypatch.setattr(subject, "_load_test_target_assemblies", target_loader)

    with pytest.raises(ValueError, match="path panel differs"):
        subject.run_round74_segmented_terminal_evaluation(
            **paths,
            terminal_observed_wall_ns=1_800_000_000_000_000_000,
            progress=lambda *_args, **_kwargs: None,
        )

    assert target_called is False
    assert paths["one_use_store_path"].exists() is False
