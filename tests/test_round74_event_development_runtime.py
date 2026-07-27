from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.round74_event_development_runtime as subject


def _paths(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repository"
    plan = repository / "plan.json"
    bindings = repository / "bindings"
    assemblies = repository / "assemblies"
    sources = repository / "sources"
    output = repository / "output"
    database = repository / "microstructure.duckdb"
    bindings.mkdir(parents=True)
    assemblies.mkdir()
    sources.mkdir()
    plan.write_text("{}\n", encoding="ascii")
    database.write_bytes(b"database")
    return {
        "repository": repository,
        "plan": plan,
        "bindings": bindings,
        "assemblies": assemblies,
        "sources": sources,
        "output": output,
        "database": database,
    }


def _inputs() -> SimpleNamespace:
    return SimpleNamespace(
        inputs_sha256="1" * 64,
        plan=SimpleNamespace(plan_sha256="2" * 64),
        partition=SimpleNamespace(partition_sha256="3" * 64),
        target_assemblies=tuple((str(index), object()) for index in range(144)),
    )


def _artifact(output: Path) -> SimpleNamespace:
    policies = tuple(
        SimpleNamespace(
            profile=profile,
            accepted=profile != "aggressive",
            selected_quantile=0.75 if profile != "aggressive" else None,
            selected_threshold_score=1.25 if profile != "aggressive" else None,
            rejection_reasons=()
            if profile != "aggressive"
            else ("after_cost_gate_failed",),
        )
        for profile in ("conservative", "regular", "aggressive")
    )
    return SimpleNamespace(
        bundle_sha256="4" * 64,
        bundle_path=output / f"round74-development-policy-{'4' * 64}.json",
        pretest_policy=SimpleNamespace(
            policy_sha256="5" * 64,
            policy_path=output / f"round74-pretest-policy-{'5' * 64}.json",
            model_sha256="6" * 64,
            model_path=output / f"round74-model-{'6' * 64}.safetensors",
            selected_candidate_id="event_pooling_linear",
            tuning_loss=0.125,
        ),
        bundle=SimpleNamespace(action_policies=policies),
    )


def _run(
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    events: list[object],
) -> dict[str, object]:
    inputs = _inputs()

    def load_inputs(**kwargs: object) -> SimpleNamespace:
        events.append(("inputs", kwargs, inputs))
        return inputs

    class Store:
        def __init__(self, path: Path, **kwargs: object) -> None:
            events.append(("store_init", path, kwargs))

        def __enter__(self) -> Store:
            events.append("store_open")
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            events.append(("store_close", exc_type, exc, traceback))

    def train(store: object, selected: object, **kwargs: object) -> SimpleNamespace:
        events.append(("train", store, selected, kwargs))
        return _artifact(paths["output"])

    monkeypatch.setattr(subject, "load_round74_development_inputs", load_inputs)
    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(subject, "_train_development", train)
    monkeypatch.setattr(subject, "_active_capture_processes", lambda: [])
    monkeypatch.setattr(
        subject,
        "_database_and_wal_bytes",
        lambda _path: (len(b"database"), 0),
    )
    monkeypatch.setattr(
        subject,
        "_preflight_backend",
        lambda request: (
            events.append(("backend", request))
            or SimpleNamespace(
                requested=request,
                kind="directml",
                device="privateuseone:0",
                vendor="test AMD",
                selection="runtime_current_device",
                accelerated=True,
            )
        ),
    )
    return subject.run_round74_event_development(
        repository=paths["repository"],
        database_path=paths["database"],
        plan_path=paths["plan"],
        binding_directory=paths["bindings"],
        target_assembly_directory=paths["assemblies"],
        source_artifact_root=paths["sources"],
        output_directory=paths["output"],
        compute_backend="directml",
        memory_limit="4GB",
        database_threads=2,
        inference_minibatch_rows=64,
        progress_interval_seconds=60.0,
        progress=lambda stage, **values: events.append(("progress", stage, values)),
    )


def test_runner_validates_inputs_before_read_only_database_and_reports_no_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    events: list[object] = []

    result = _run(paths, monkeypatch, events)

    input_event = next(value for value in events if value[0] == "inputs")
    input_index = events.index(input_event)
    store_event = next(value for value in events if value[0] == "store_init")
    assert input_index < events.index(store_event)
    assert input_event[1]["source_artifact_root"] == paths["sources"].resolve()
    assert store_event[1] == paths["database"].resolve()
    assert store_event[2] == {
        "memory_limit": "4GB",
        "threads": 2,
        "read_only": True,
    }
    train_event = next(value for value in events if value[0] == "train")
    assert train_event[2] is input_event[2]
    assert train_event[3] == {
        "output_directory": paths["output"].resolve(),
        "compute_backend": "directml",
        "inference_minibatch_rows": 64,
    }
    assert result["schema_version"] == "round-074-development-run-v1"
    assert result["inputs_sha256"] == "1" * 64
    assert result["development_target_assembly_count"] == 144
    assert result["backend"]["kind"] == "directml"
    assert result["backend"]["accelerated"] is True
    assert [row["profile"] for row in result["profiles"]] == [
        "conservative",
        "regular",
        "aggressive",
    ]
    assert result["profiles"][2]["accepted"] is False
    assert set(result["authority"].values()) == {False}
    claimed = result.pop("result_sha256")
    assert claimed == subject._canonical_sha256(result)
    stages = [value[1] for value in events if value[0] == "progress"]
    assert stages == [
        "input_validation_started",
        "input_validation_completed",
        "backend_ready",
        "development_started",
        "development_completed",
    ]


def test_runner_rechecks_capture_and_wal_without_opening_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    store_opened = False

    class Store:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal store_opened
            store_opened = True

    monkeypatch.setattr(subject, "ImpactAbsorptionStore", Store)
    monkeypatch.setattr(
        subject,
        "load_round74_development_inputs",
        lambda **_kwargs: _inputs(),
    )
    monkeypatch.setattr(
        subject,
        "_preflight_backend",
        lambda _request: SimpleNamespace(
            requested="auto",
            kind="cpu",
            device="cpu",
            vendor="test CPU",
            selection="deterministic_cpu_reference",
            accelerated=False,
        ),
    )
    process_panels = iter(([], [{"process_id": 17}]))
    monkeypatch.setattr(
        subject,
        "_active_capture_processes",
        lambda: next(process_panels),
    )
    monkeypatch.setattr(
        subject,
        "_database_and_wal_bytes",
        lambda _path: (len(b"database"), 0),
    )

    with pytest.raises(RuntimeError, match="capture became active"):
        subject.run_round74_event_development(
            repository=paths["repository"],
            database_path=paths["database"],
            plan_path=paths["plan"],
            binding_directory=paths["bindings"],
            target_assembly_directory=paths["assemblies"],
            source_artifact_root=paths["sources"],
            output_directory=paths["output"],
            progress=lambda *_args, **_kwargs: None,
        )
    assert store_opened is False

    monkeypatch.setattr(subject, "_active_capture_processes", lambda: [])
    wal_panels = iter(((len(b"database"), 0), (len(b"database"), 19)))
    monkeypatch.setattr(
        subject,
        "_database_and_wal_bytes",
        lambda _path: next(wal_panels),
    )
    with pytest.raises(RuntimeError, match="WAL appeared"):
        subject.run_round74_event_development(
            repository=paths["repository"],
            database_path=paths["database"],
            plan_path=paths["plan"],
            binding_directory=paths["bindings"],
            target_assembly_directory=paths["assemblies"],
            source_artifact_root=paths["sources"],
            output_directory=paths["output"],
            progress=lambda *_args, **_kwargs: None,
        )
    assert store_opened is False


def test_runner_rejects_active_capture_before_panel_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    panel_loaded = False

    def load_inputs(**_kwargs: object) -> object:
        nonlocal panel_loaded
        panel_loaded = True
        return object()

    monkeypatch.setattr(subject, "load_round74_development_inputs", load_inputs)
    monkeypatch.setattr(
        subject,
        "_active_capture_processes",
        lambda: [{"process_id": 18}],
    )

    with pytest.raises(RuntimeError, match="capture is active"):
        subject.run_round74_event_development(
            repository=paths["repository"],
            database_path=paths["database"],
            plan_path=paths["plan"],
            binding_directory=paths["bindings"],
            target_assembly_directory=paths["assemblies"],
            source_artifact_root=paths["sources"],
            output_directory=paths["output"],
            progress=lambda *_args, **_kwargs: None,
        )
    assert panel_loaded is False
