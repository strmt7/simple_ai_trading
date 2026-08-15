from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.round74_segmented_development_runtime as subject


def test_segmented_runtime_binds_target_free_candidate_device_groups() -> None:
    candidates = subject.Round74EventTrainingConfig().candidate_ids
    preflight = {
        "preflight_sha256": "a" * 64,
        "selected_group_sizes": {
            candidate_id: index + 1 for index, candidate_id in enumerate(candidates)
        },
    }

    selected = subject._device_group_training_config(
        policy="auto",
        fixed_group_size=8,
        preflight=preflight,
    )

    assert selected.execution_mode == "segmented_cohort"
    assert selected.device_group_selection_mode == "target_free_host_benchmark"
    assert selected.device_group_preflight_sha256 == "a" * 64
    assert tuple(
        selected.device_run_group_size_for(candidate_id) for candidate_id in candidates
    ) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="preflight is missing"):
        subject._device_group_training_config(
            policy="auto",
            fixed_group_size=8,
            preflight=None,
        )


def test_segmented_runtime_releases_training_batches_before_accepted_ai_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    partition = object()
    assemblies = {
        "training": object(),
        "ai": object(),
    }
    inputs = SimpleNamespace(
        coverage=SimpleNamespace(partition=partition),
        target_assembly_by_run_id=lambda: dict(assemblies),
        development_bindings_by_run_id=lambda: {"training": object()},
    )
    action_policies = tuple(
        SimpleNamespace(
            profile=profile,
            selection_sha256=f"{index + 1:064x}",
            accepted=profile != "regular",
            rejection_reasons=(
                () if profile != "regular" else ("no_viable_threshold",)
            ),
        )
        for index, profile in enumerate(subject.ROUND74_ACTION_PROFILES)
    )
    bundle = SimpleNamespace(
        action_policies=action_policies,
        probability_calibration=object(),
        pretest_policy_sha256="8" * 64,
        validate=lambda: None,
    )
    policy = SimpleNamespace(
        bundle=bundle,
        bundle_sha256="7" * 64,
        pretest_policy=SimpleNamespace(
            policy_sha256="6" * 64,
            policy_path=tmp_path / "policy.json",
        ),
    )
    preparation = SimpleNamespace(
        preparation_sha256="5" * 64,
        prepared=SimpleNamespace(
            scaler=SimpleNamespace(scaler_sha256="4" * 64),
            training_batches=(object(), object()),
            tuning_batches=(object(), object()),
        ),
        tuning_subpartition=object(),
        tuning_roles=SimpleNamespace(
            ai_qualification_batches=(object(),),
        ),
    )
    population = SimpleNamespace(run_ids=("ai",))
    database = tmp_path / "cohort.duckdb"
    database.write_bytes(b"database")
    routes = {"training": database, "ai": database}
    provider_calls: list[dict[str, object]] = []
    builder_calls: list[dict[str, object]] = []
    qualification_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        subject,
        "_prepare_sharded_development",
        lambda **_kwargs: events.append("prepare") or preparation,
    )
    monkeypatch.setattr(
        subject,
        "train_round74_segmented_development_policy",
        lambda *_args, **_kwargs: events.append("train") or policy,
    )
    monkeypatch.setattr(
        subject,
        "build_round74_segmented_ai_qualification_population",
        lambda selected: (
            population
            if selected is preparation.tuning_subpartition
            else pytest.fail("qualification subpartition differs")
        ),
    )
    monkeypatch.setattr(
        subject,
        "select_round74_final_action_configuration",
        lambda selected_bundle, *, profile: SimpleNamespace(
            profile=profile,
            action_selection=next(
                policy
                for policy in selected_bundle.action_policies
                if policy.profile == profile
            ),
            configuration_sha256=(
                f"{subject.ROUND74_ACTION_PROFILES.index(profile) + 30:064x}"
            ),
            mode="baseline",
            development_bundle_sha256="7" * 64,
        ),
    )

    def provider(**kwargs: object) -> object:
        provider_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        subject,
        "Round74ShardedAIQualificationExecutionReplayProvider",
        provider,
    )
    monkeypatch.setattr(
        subject,
        "Round74ShardedDevelopmentExecutionPanelBuilder",
        lambda **kwargs: builder_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        subject.gc,
        "collect",
        lambda: events.append("gc") or 0,
    )

    def qualify(_batches: object, **kwargs: object) -> object:
        configuration = kwargs["final_action_configuration"]
        selected = configuration.action_selection
        events.append(f"qualify-{selected.profile}")
        qualification_calls.append(kwargs)
        return SimpleNamespace(
            final_action_configuration=configuration,
            inference=SimpleNamespace(
                action_selection_sha256=selected.selection_sha256,
            ),
            qualification=SimpleNamespace(
                profile=selected.profile,
                qualification_sha256=f"{len(qualification_calls) + 20:064x}",
                qualification_passed=True,
                final_action_configuration_sha256=(configuration.configuration_sha256),
            ),
            validate=lambda: None,
        )

    monkeypatch.setattr(
        subject,
        "run_round74_ai_pretest_qualification",
        qualify,
    )
    progress_rows: list[tuple[str, dict[str, object]]] = []
    trained, qualified, preparation_sha256 = subject._run_development(
        routes,
        inputs,  # type: ignore[arg-type]
        model_output_directory=tmp_path / "model",
        qualification_output_directory=tmp_path / "qualification",
        compute_backend="cpu",
        training_config=subject._device_group_training_config(
            policy="fixed",
            fixed_group_size=8,
            preflight=None,
        ),
        inference_minibatch_rows=32,
        enable_ai=True,
        memory_limit="1GB",
        database_threads=1,
        progress=lambda stage, **values: progress_rows.append((stage, values)),
    )

    assert trained is policy
    assert qualified is not None
    qualified.validate()
    assert preparation_sha256 == "5" * 64
    assert events == [
        "prepare",
        "train",
        "gc",
        "qualify-conservative",
        "qualify-aggressive",
    ]
    assert len(provider_calls) == 1
    assert builder_calls == [
        {
            "database_by_run_id": routes,
            "memory_limit": "1GB",
            "database_threads": 1,
        }
    ]
    assert provider_calls[0]["database_by_run_id"] == routes
    assert provider_calls[0]["assembly_by_run_id"] == {"ai": assemblies["ai"]}
    assert [call["qualification_output_path"].name for call in qualification_calls] == [
        "round74-ai-pretest-qualification-conservative.json",
        "round74-ai-pretest-qualification-aggressive.json",
    ]
    assert any(stage == "training_batches_released" for stage, _values in progress_rows)


@pytest.mark.parametrize("enable_ai", (None, 1, "yes"))
def test_segmented_runtime_rejects_invalid_ai_toggle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_ai: object,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    database = repository / "cohort.duckdb"
    database.write_bytes(b"database")
    paths = []
    for name in ("state", "recovery", "assemblies", "sources"):
        path = repository / name
        path.mkdir()
        paths.append(path)
    plan = repository / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(subject, "_guard_idle_databases", pytest.fail)

    with pytest.raises(ValueError, match="runtime policy differs"):
        subject.run_round74_segmented_development(
            repository=repository,
            database_paths=(database,),
            plan_path=plan,
            state_root=paths[0],
            recovery_outcome_directory=paths[1],
            target_assembly_directory=paths[2],
            source_artifact_root=paths[3],
            model_output_directory=repository / "model",
            qualification_output_directory=repository / "qualification",
            terminal_observed_wall_ns=1,
            progress=lambda *_args, **_kwargs: None,
            enable_ai=enable_ai,  # type: ignore[arg-type]
        )


def test_segmented_runtime_routes_before_loading_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    databases = (repository / "one.duckdb", repository / "two.duckdb")
    for database in databases:
        database.write_bytes(b"database")
    directories = {}
    for name in ("state", "recovery", "assemblies", "sources"):
        path = repository / name
        path.mkdir()
        directories[name] = path
    plan_path = repository / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    run_ids = ("training", "tuning")
    partition = SimpleNamespace(
        partition_sha256="3" * 64,
        entries=tuple(
            SimpleNamespace(run_id=run_id, role=role)
            for run_id, role in zip(run_ids, ("training", "tuning"), strict=True)
        ),
    )
    coverage = SimpleNamespace(
        coverage_sha256="2" * 64,
        partition=partition,
    )
    terminal = SimpleNamespace(
        plan=SimpleNamespace(plan_sha256="1" * 64),
        coverage=coverage,
        slot_evidence=((0, "result", "4" * 64),),
    )
    inputs = SimpleNamespace(
        inputs_sha256="5" * 64,
        plan=terminal.plan,
        coverage=coverage,
        slot_evidence=terminal.slot_evidence,
        target_assemblies=(("training", object()), ("tuning", object())),
        development_bindings_by_run_id=lambda: {
            "training": object(),
            "tuning": object(),
        },
    )
    action_policies = tuple(
        SimpleNamespace(
            profile=profile,
            accepted=False,
            rejection_reasons=("development_gate_failed",),
        )
        for profile in subject.ROUND74_ACTION_PROFILES
    )
    policy = SimpleNamespace(
        bundle=SimpleNamespace(action_policies=action_policies),
        bundle_sha256="6" * 64,
        bundle_path=repository / "bundle.json",
        pretest_policy=SimpleNamespace(
            policy_sha256="7" * 64,
            policy_path=repository / "policy.json",
            model_sha256="8" * 64,
            model_path=repository / "model.pt",
            selected_candidate_id="candidate",
            tuning_loss=1.0,
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda selected, **_kwargs: tuple(10 for _path in selected),
    )
    monkeypatch.setattr(
        subject,
        "load_round74_segmented_terminal_coverage",
        lambda **_kwargs: events.append("coverage") or terminal,
    )

    def route(*_args: object, **_kwargs: object):
        events.append("route")
        return (
            {"training": databases[0], "tuning": databases[1]},
            "9" * 64,
            databases,
        )

    monkeypatch.setattr(subject, "_route_development_run_databases", route)
    monkeypatch.setattr(
        subject,
        "load_round74_segmented_development_inputs",
        lambda **_kwargs: events.append("targets") or inputs,
    )
    monkeypatch.setattr(
        subject,
        "_preflight_backend",
        lambda _requested: SimpleNamespace(
            requested="cpu",
            kind="cpu",
            device="cpu",
            vendor="host",
            selection="explicit",
            accelerated=False,
        ),
    )
    preflight = {
        "preflight_sha256": "a" * 64,
        "selected_group_sizes": {
            candidate_id: index + 1
            for index, candidate_id in enumerate(
                subject.Round74EventTrainingConfig().candidate_ids
            )
        },
    }
    monkeypatch.setattr(
        subject,
        "run_round74_device_group_preflight_subprocess",
        lambda *_args, **_kwargs: events.append("device-preflight") or preflight,
    )
    monkeypatch.setattr(
        subject,
        "write_round74_device_group_preflight",
        lambda *_args, **_kwargs: (
            events.append("device-write")
            or repository / "model-output" / "preflight.json"
        ),
    )
    monkeypatch.setattr(
        subject,
        "_run_development",
        lambda *_args, **_kwargs: (policy, None, "a" * 64),
    )
    progress: list[str] = []
    result = subject.run_round74_segmented_development(
        repository=repository,
        database_paths=databases,
        plan_path=plan_path,
        state_root=directories["state"],
        recovery_outcome_directory=directories["recovery"],
        target_assembly_directory=directories["assemblies"],
        source_artifact_root=directories["sources"],
        model_output_directory=repository / "model-output",
        qualification_output_directory=repository / "qualification-output",
        terminal_observed_wall_ns=123,
        progress=lambda stage, **_values: progress.append(stage),
        compute_backend="cpu",
        enable_ai=False,
    )

    assert events == [
        "coverage",
        "route",
        "device-preflight",
        "device-write",
        "targets",
    ]
    assert result["database_count"] == 2
    assert result["used_database_count"] == 2
    assert result["database_bytes_total"] == 20
    assert result["database_route_sha256"] == "9" * 64
    assert result["database_access_policy"] == "federated_read_only_one_at_a_time"
    assert result["authority"]["profitability_claim"] is False
    assert progress.index("target_free_database_route_completed") < progress.index(
        "input_validation_started"
    )
    assert progress.index("device_group_preflight_completed") < progress.index(
        "input_validation_started"
    )


def test_segmented_runtime_routes_shards_target_blind_and_order_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.duckdb"
    second = tmp_path / "second.duckdb"
    unused = tmp_path / "unused.duckdb"
    for path in (first, second, unused):
        path.write_bytes(b"database")
    rows_by_path = {
        first: (("b",), ("foreign",)),
        second: (("a",), ("c",)),
        unused: (("test",),),
    }
    active = 0
    maximum_active = 0

    class Store:
        def __init__(self, path: Path, **kwargs: object) -> None:
            self.path = Path(path)
            assert kwargs["read_only"] is True

        def __enter__(self) -> "Store":
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active -= 1

        def connect(self) -> "Store":
            return self

        def execute(self, query: str) -> "Store":
            assert "SELECT run_id" in query
            return self

        def fetchall(self) -> tuple[tuple[str], ...]:
            return rows_by_path[self.path]

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )
    expected = ("a", "b", "c")
    routes, digest, used = subject._route_development_run_databases(
        (first, unused, second),
        development_run_ids=expected,
        memory_limit="1GB",
        database_threads=1,
    )
    reversed_routes, reversed_digest, reversed_used = (
        subject._route_development_run_databases(
            (second, unused, first),
            development_run_ids=expected,
            memory_limit="1GB",
            database_threads=1,
        )
    )

    assert tuple(routes) == expected
    assert routes == reversed_routes == {"a": second, "b": first, "c": second}
    assert digest == reversed_digest
    assert used == (second, first)
    assert reversed_used == (second, first)
    assert maximum_active == 1


def test_segmented_runtime_rejects_run_duplicated_across_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    databases = (tmp_path / "one.duckdb", tmp_path / "two.duckdb")
    for path in databases:
        path.write_bytes(b"database")

    class Store:
        def __init__(self, _path: Path, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Store":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def connect(self) -> "Store":
            return self

        def execute(self, _query: str) -> "Store":
            return self

        def fetchall(self) -> tuple[tuple[str], ...]:
            return (("duplicate",),)

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )

    with pytest.raises(ValueError, match="exists in multiple stores"):
        subject._route_development_run_databases(
            databases,
            development_run_ids=("duplicate",),
            memory_limit="1GB",
            database_threads=1,
        )


def test_segmented_runtime_rejects_missing_run_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "one.duckdb"
    database.write_bytes(b"database")

    class Store:
        def __init__(self, _path: Path, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Store":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def connect(self) -> "Store":
            return self

        def execute(self, _query: str) -> "Store":
            return self

        def fetchall(self) -> tuple[tuple[str], ...]:
            return (("present",),)

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )

    with pytest.raises(ValueError, match="database coverage differs"):
        subject._route_development_run_databases(
            (database,),
            development_run_ids=("present", "missing"),
            memory_limit="1GB",
            database_threads=1,
        )


def test_segmented_runtime_rejects_wal_on_any_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    databases = (tmp_path / "one.duckdb", tmp_path / "two.duckdb")
    for database in databases:
        database.write_bytes(b"database")
    results = iter(((8, 0), (8, 1)))
    monkeypatch.setattr(subject, "_active_segmented_capture_processes", lambda: ())
    monkeypatch.setattr(subject, "_database_and_wal_bytes", lambda _path: next(results))

    with pytest.raises(RuntimeError, match="database WAL exists"):
        subject._guard_idle_databases(databases, repeated=False)


def test_segmented_runtime_retains_single_store_source_builder_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "one.duckdb"
    observed: list[tuple[tuple[Path, ...], bool]] = []

    def guard(paths: tuple[Path, ...], *, repeated: bool) -> tuple[int, ...]:
        observed.append((paths, repeated))
        return (42,)

    monkeypatch.setattr(subject, "_guard_idle_databases", guard)

    assert subject._guard_idle_database(database, repeated=True) == 42
    assert observed == [((database,), True)]


def test_segmented_runtime_sharded_scaler_preserves_run_order_and_one_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.duckdb"
    second = tmp_path / "second.duckdb"
    first.write_bytes(b"database")
    second.write_bytes(b"database")
    active = 0
    maximum_active = 0
    opened: list[Path] = []

    class Store:
        def __init__(self, path: Path, **_kwargs: object) -> None:
            self.path = Path(path)

        def __enter__(self) -> "Store":
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            opened.append(self.path)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active -= 1

    run_ids = ("run-a", "run-b", "run-c")
    values = {run_id: float(index) for index, run_id in enumerate(run_ids, 1)}
    bindings = {run_id: SimpleNamespace(run_id=run_id) for run_id in run_ids}
    entries = {
        run_id: SimpleNamespace(
            capture_start_wall_ns=1,
            capture_end_wall_ns=10,
        )
        for run_id in run_ids
    }
    partition = SimpleNamespace(
        validate=lambda: None,
        entry=entries.__getitem__,
    )
    training_split = SimpleNamespace(
        validate=lambda: None,
        optimization_run_ids=run_ids,
        parent_partition_sha256="1" * 64,
        split_sha256="2" * 64,
    )
    observed: dict[str, object] = {}

    def observations(_store: Store, *, binding: object):
        yield SimpleNamespace(
            token=SimpleNamespace(
                received_wall_ns=5,
                feature_values=(values[binding.run_id],),
            )
        )

    def fit(chunks: object, **kwargs: object) -> object:
        observed["values"] = [
            float(value) for chunk in chunks for value in chunk[:, 0].tolist()
        ]
        observed["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(subject, "_validate_binding_entry", lambda *_a, **_k: None)
    monkeypatch.setattr(
        subject,
        "iter_round74_v10_segment_event_observations",
        observations,
    )
    monkeypatch.setattr(subject, "fit_round74_event_feature_scaler_stream", fit)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )

    result = subject._fit_sharded_optimization_feature_scaler(
        database_by_run_id={"run-a": first, "run-b": second, "run-c": first},
        partition=partition,  # type: ignore[arg-type]
        bindings_by_run_id=bindings,  # type: ignore[arg-type]
        training_split=training_split,
        memory_limit="1GB",
        database_threads=1,
        chunk_rows=2,
        maximum_fit_rows=10,
    )

    assert result is not None
    assert observed["values"] == [1.0, 2.0, 3.0]
    assert observed["kwargs"]["fit_source_run_ids"] == run_ids
    assert opened == [first, second, first]
    assert maximum_active == 1


def test_segmented_runtime_sharded_execution_builder_restores_run_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.duckdb"
    second = tmp_path / "second.duckdb"
    first.write_bytes(b"database")
    second.write_bytes(b"database")
    run_ids = ("run-a", "run-b", "run-c")
    active = 0
    maximum_active = 0
    opened: list[Path] = []

    class Store:
        def __init__(self, path: Path, **_kwargs: object) -> None:
            self.path = Path(path)

        def __enter__(self) -> "Store":
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            opened.append(self.path)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active -= 1

    class Panel:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

        def validate(self) -> None:
            return None

    batches = tuple(
        SimpleNamespace(run_id=(run_id,), validate=lambda: None) for run_id in run_ids
    )
    calls: list[tuple[str, ...]] = []

    def build(
        _store: Store,
        *,
        policy_selection_batches: tuple[object, ...],
        **_kwargs: object,
    ) -> tuple[Panel, ...]:
        grouped = tuple(batch.run_id[0] for batch in policy_selection_batches)
        calls.append(grouped)
        return tuple(
            Panel(
                profile=profile,
                partition_sha256="1" * 64,
                decision_latency_evidence_sha256="2" * 64,
                additional_entry_latency_ns=index + 1,
                source_target_assembly_sha256=tuple(
                    (run_id, "3" * 64) for run_id in grouped
                ),
                source_capture_report_sha256=tuple(
                    (run_id, "4" * 64) for run_id in grouped
                ),
                execution_replay_module_sha256="5" * 64,
                rows=tuple((profile, run_id) for run_id in grouped),
            )
            for index, profile in enumerate(subject.ROUND74_ACTION_PROFILES)
        )

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(subject, "Round74ActionExecutionPanel", Panel)
    monkeypatch.setattr(subject, "build_round74_delayed_execution_panels", build)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )
    builder = subject.Round74ShardedDevelopmentExecutionPanelBuilder(
        database_by_run_id={"run-a": first, "run-b": second, "run-c": first},
        memory_limit="1GB",
        database_threads=1,
    )
    panels = builder(
        partition=SimpleNamespace(validate=lambda: None),  # type: ignore[arg-type]
        policy_selection_batches=batches,  # type: ignore[arg-type]
        target_assembly_by_run_id={run_id: object() for run_id in run_ids},
        latency_evidence=SimpleNamespace(validate=lambda: None),
    )

    assert calls == [("run-a",), ("run-b",), ("run-c",)]
    assert opened == [first, second, first]
    assert maximum_active == 1
    for panel in panels:
        assert (
            tuple(run_id for run_id, _digest in panel.source_target_assembly_sha256)
            == run_ids
        )
        assert tuple(run_id for _profile, run_id in panel.rows) == run_ids


def test_segmented_runtime_sharded_ai_replay_restores_instruction_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.duckdb"
    second = tmp_path / "second.duckdb"
    first.write_bytes(b"database")
    second.write_bytes(b"database")
    run_ids = ("run-a", "run-b", "run-c")
    active = 0
    maximum_active = 0

    class Store:
        def __init__(self, _path: Path, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Store":
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active -= 1

    population = SimpleNamespace(
        run_ids=run_ids,
        population_sha256="1" * 64,
        parent_tuning_subpartition_sha256="2" * 64,
        validate=lambda: None,
    )
    partition = SimpleNamespace(
        partition_sha256="3" * 64,
        validate=lambda: None,
        entry=lambda _run_id: SimpleNamespace(role="tuning"),
    )
    action_selection = SimpleNamespace(
        tuning_subpartition_sha256="2" * 64,
        selection_sha256="4" * 64,
        validate=lambda: None,
    )

    class Instruction:
        def __init__(self, run_id: str, row_index: int, manifest: str) -> None:
            self.run_id = run_id
            self.row_index = row_index
            self.model_manifest_sha256 = manifest
            self.partition_sha256 = "3" * 64
            self.action_selection_sha256 = "4" * 64
            self.feature_row_sha256 = f"{row_index + 10:064x}"
            self.symbol = "BTCUSDT"
            self.side = "long"
            self.horizon_seconds = 30
            self.source_review_sha256 = "7" * 64

        def validate(self) -> None:
            return None

    class Evidence:
        def __init__(self, instruction: Instruction) -> None:
            self.row_index = instruction.row_index
            self.feature_row_sha256 = instruction.feature_row_sha256
            self.run_id = instruction.run_id
            self.symbol = instruction.symbol
            self.side = instruction.side
            self.horizon_seconds = instruction.horizon_seconds
            self.source_review_sha256 = instruction.source_review_sha256
            self.partition_sha256 = instruction.partition_sha256

        def validate(self) -> None:
            return None

    manifests = ("5" * 64, "6" * 64)
    rows_by_manifest = {
        manifest: tuple(
            Instruction(run_id, index, manifest)
            for index, run_id in enumerate(("run-c", "run-a", "run-b"))
        )
        for manifest in manifests
    }

    def replay(
        _store: Store,
        *,
        instructions: tuple[Instruction, ...],
        **_kwargs: object,
    ) -> tuple[Evidence, ...]:
        return tuple(Evidence(row) for row in instructions)

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(subject, "replay_round74_ai_execution_store_run", replay)
    monkeypatch.setattr(
        subject,
        "_guard_idle_databases",
        lambda *_args, **_kwargs: (8,),
    )
    provider = subject.Round74ShardedAIQualificationExecutionReplayProvider(
        database_by_run_id={"run-a": first, "run-b": second, "run-c": first},
        partition=partition,  # type: ignore[arg-type]
        qualification_population=population,  # type: ignore[arg-type]
        assembly_by_run_id={run_id: object() for run_id in run_ids},
        memory_limit="1GB",
        database_threads=1,
    )
    result = provider(
        qualification_population=population,  # type: ignore[arg-type]
        action_selection=action_selection,
        instructions_by_manifest=rows_by_manifest,  # type: ignore[arg-type]
    )

    assert maximum_active == 1
    for manifest in manifests:
        assert tuple(row.row_index for row in result[manifest]) == (0, 1, 2)
