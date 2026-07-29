from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.round74_segmented_development_runtime as subject


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
    provider_calls: list[dict[str, object]] = []
    qualification_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        subject,
        "prepare_round74_segmented_development",
        lambda *_args, **_kwargs: events.append("prepare") or preparation,
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
        "Round74AIQualificationStoreExecutionReplayProvider",
        provider,
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
                final_action_configuration_sha256=(
                    configuration.configuration_sha256
                ),
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
        object(),  # type: ignore[arg-type]
        inputs,  # type: ignore[arg-type]
        model_output_directory=tmp_path / "model",
        qualification_output_directory=tmp_path / "qualification",
        compute_backend="cpu",
        inference_minibatch_rows=32,
        enable_ai=True,
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
    monkeypatch.setattr(subject, "_guard_idle_database", pytest.fail)

    with pytest.raises(ValueError, match="runtime policy differs"):
        subject.run_round74_segmented_development(
            repository=repository,
            database_path=database,
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
