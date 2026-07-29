from __future__ import annotations

from types import SimpleNamespace

import pytest

from simple_ai_trading.impact_absorption_event_training import (
    Round74EventTrainingConfig,
)
import simple_ai_trading.round74_segmented_representation_comparison as subject


def test_segmented_representation_coordinator_fixes_challenger_and_binds_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    representation_calls: list[str] = []
    preparations = {
        "per_symbol": object(),
        "global_cross_asset": object(),
    }
    prepared = SimpleNamespace(
        preparation_sha256="a" * 64,
        representation=lambda value: (
            representation_calls.append(value) or preparations[value]
        ),
    )
    prepare_calls: list[dict[str, object]] = []

    def prepare(*_args: object, **kwargs: object) -> object:
        prepare_calls.append(kwargs)
        return prepared

    monkeypatch.setattr(
        subject,
        "prepare_round74_segmented_matched_development",
        prepare,
    )
    artifacts: list[object] = []
    training_calls: list[tuple[object, dict[str, object]]] = []

    def train(preparation: object, **kwargs: object) -> object:
        training_calls.append((preparation, kwargs))
        representation = (
            "per_symbol"
            if preparation is preparations["per_symbol"]
            else "global_cross_asset"
        )
        artifact = SimpleNamespace(
            pretest_policy=SimpleNamespace(
                selected_candidate_id="causal_event_tcn",
                policy_path=f"{representation}.json",
            ),
            marker=representation,
        )
        artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(
        subject,
        "train_round74_segmented_development_policy",
        train,
    )
    policies = {
        "per_symbol.json": {"policy_sha256": "b" * 64},
        "global_cross_asset.json": {"policy_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        subject,
        "load_round74_pretest_policy",
        lambda path: (object(), policies[path]),
    )
    comparison = SimpleNamespace(
        comparison_sha256="d" * 64,
        as_dict=lambda: {
            "schema_version": "round-074-representation-comparison-v3",
            "comparison_sha256": "d" * 64,
        },
    )
    build_calls: list[dict[str, object]] = []

    def build(selected: object, **kwargs: object) -> object:
        assert selected is prepared
        build_calls.append(kwargs)
        return comparison

    monkeypatch.setattr(subject, "build_round74_representation_comparison", build)
    monkeypatch.setattr(
        subject,
        "load_round74_representation_comparison",
        lambda _path: comparison,
    )

    result = subject.train_and_compare_round74_segmented_representations(
        object(),
        partition=object(),
        bindings_by_run_id={"run": object()},
        target_assembly_by_run_id={"run": object()},
        output_directory=tmp_path,
    )

    assert len(prepare_calls) == 1
    assert representation_calls == ["per_symbol", "global_cross_asset"]
    assert len(training_calls) == 2
    baseline_config = training_calls[0][1]["config"]
    challenger_config = training_calls[1][1]["config"]
    assert baseline_config.execution_mode == "segmented_cohort"
    assert baseline_config.architecture_selection_mode == "complexity_gate"
    assert challenger_config.execution_mode == "segmented_cohort"
    assert challenger_config.architecture_selection_mode == "fixed"
    assert challenger_config.candidate_ids == ("causal_event_tcn",)
    assert all(
        call[1]["matched_preparation_sha256"] == "a" * 64
        for call in training_calls
    )
    assert build_calls[0]["minimum_mean_improvement"] == (
        baseline_config.minimum_tuning_improvement
    )
    assert result.baseline is artifacts[0]
    assert result.challenger is artifacts[1]


def test_segmented_representation_coordinator_rejects_nonsegmented_mode() -> None:
    with pytest.raises(ValueError, match="requires complexity mode"):
        subject.train_and_compare_round74_segmented_representations(
            object(),
            partition=object(),
            bindings_by_run_id={},
            target_assembly_by_run_id={},
            output_directory="unused",
            config=Round74EventTrainingConfig(execution_mode="cohort"),
        )
