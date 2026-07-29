"""Efficient matched representation selection for segmented Round 74 data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path

from .impact_absorption_event_dataset import Round74EventRunPartition
from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
from .impact_absorption_event_training import (
    Round74EventTrainingConfig,
    load_round74_pretest_policy,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .round74_representation_comparison import (
    Round74RepresentationComparisonArtifact,
    build_round74_representation_comparison,
    load_round74_representation_comparison,
)
from .round74_segmented_development_operator import (
    prepare_round74_segmented_matched_development,
    train_round74_segmented_development_policy,
)
from .storage import write_bytes_atomic


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def train_and_compare_round74_segmented_representations(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
) -> Round74RepresentationComparisonArtifact:
    """Train matched views and promote cross-asset context only through both gates."""

    selected_config = config or replace(
        Round74EventTrainingConfig(),
        execution_mode="segmented_cohort",
    )
    selected_config.validate()
    if (
        selected_config.execution_mode != "segmented_cohort"
        or selected_config.architecture_selection_mode != "complexity_gate"
    ):
        raise ValueError(
            "Round 74 segmented representation baseline requires complexity mode"
        )
    prepared = prepare_round74_segmented_matched_development(
        store,
        partition=partition,
        bindings_by_run_id=bindings_by_run_id,
        target_assembly_by_run_id=target_assembly_by_run_id,
    )
    preparation_sha256 = prepared.preparation_sha256
    output = Path(output_directory)
    baseline_preparation = prepared.representation("per_symbol")
    baseline = train_round74_segmented_development_policy(
        baseline_preparation,
        store=store,
        partition=partition,
        target_assembly_by_run_id=target_assembly_by_run_id,
        output_directory=output / "per-symbol",
        compute_backend=compute_backend,
        config=selected_config,
        inference_minibatch_rows=inference_minibatch_rows,
        matched_preparation_sha256=preparation_sha256,
    )
    _baseline_model, baseline_policy = load_round74_pretest_policy(
        baseline.pretest_policy.policy_path
    )
    challenger_config = replace(
        selected_config,
        candidate_ids=(baseline.pretest_policy.selected_candidate_id,),
        architecture_selection_mode="fixed",
    )
    challenger_config.validate()
    challenger_preparation = prepared.representation("global_cross_asset")
    challenger = train_round74_segmented_development_policy(
        challenger_preparation,
        store=store,
        partition=partition,
        target_assembly_by_run_id=target_assembly_by_run_id,
        output_directory=output / "global-cross-asset",
        compute_backend=compute_backend,
        config=challenger_config,
        inference_minibatch_rows=inference_minibatch_rows,
        matched_preparation_sha256=preparation_sha256,
    )
    _challenger_model, challenger_policy = load_round74_pretest_policy(
        challenger.pretest_policy.policy_path
    )
    comparison = build_round74_representation_comparison(
        prepared,
        baseline=baseline,
        challenger=challenger,
        baseline_policy=baseline_policy,
        challenger_policy=challenger_policy,
        minimum_mean_improvement=selected_config.minimum_tuning_improvement,
    )
    payload = _canonical_bytes(comparison.as_dict()) + b"\n"
    path = (
        output
        / f"round74-representation-comparison-{comparison.comparison_sha256}.json"
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Round 74 immutable representation result differs")
    else:
        write_bytes_atomic(path, payload)
    persisted = load_round74_representation_comparison(path)
    if persisted.as_dict() != comparison.as_dict():
        raise RuntimeError("Round 74 persisted representation result differs")
    return Round74RepresentationComparisonArtifact(
        comparison=comparison,
        comparison_path=path,
        baseline=baseline,
        challenger=challenger,
    )


__all__ = ["train_and_compare_round74_segmented_representations"]
