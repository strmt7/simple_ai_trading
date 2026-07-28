"""Disjoint tuning-only execution of the Round 74 local-AI qualification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .impact_absorption_ai_execution_replay import (
    Round74AIExecutionReplayInstruction,
    build_round74_ai_execution_replay_instructions,
)
from .impact_absorption_ai_review_preparation import (
    Round74AIReviewModelBinding,
    Round74AIReviewPanel,
    prepare_round74_target_free_ai_reviews,
)
from .impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
    review_round74_ai_candidate,
)
from .impact_absorption_ai_uplift import (
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
    Round74AIPretestQualificationPanel,
    Round74AIQualificationPopulation,
    Round74AIUpliftDevelopmentReport,
    build_round74_ai_pretest_qualification,
    evaluate_round74_ai_overlay_development,
    write_round74_ai_pretest_qualification,
)
from .impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionTrace,
    build_round74_action_inference_context,
    simulate_round74_action_trace_batches,
)
from .impact_absorption_event_calibration import Round74ProbabilityCalibration
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_sealed_evaluation import (
    Round74TargetFreeCandidateInference,
    infer_round74_target_free_candidates,
)
from .round74_event_model_operator import Round74PreparedTuningRoles


ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION = (
    "round-074-ai-qualification-operator-v3"
)

ReviewRunner = Callable[..., Round74AIRuntimeOutcome]
ModelBatchHook = Callable[[Round74AIReviewModelBinding], object]
ProgressCallback = Callable[[Mapping[str, object]], None]


class Round74AIQualificationExecutionReplayProvider(Protocol):
    """Replay AI decisions only on the preassigned qualification population."""

    def __call__(
        self,
        *,
        qualification_population: Round74AIQualificationPopulation,
        action_selection: Round74ActionPolicySelection,
        instructions_by_manifest: Mapping[
            str,
            Sequence[Round74AIExecutionReplayInstruction],
        ],
    ) -> Mapping[str, Sequence[Round74AIExecutionReplayEvidence]]: ...


@dataclass(frozen=True)
class Round74AIQualificationOperatorResult:
    """In-memory, hash-bound evidence produced before sealed-test access."""

    qualification_population: Round74AIQualificationPopulation
    inference: Round74TargetFreeCandidateInference
    baseline_trace: Round74ActionTrace
    review_panel: Round74AIReviewPanel
    instructions_by_manifest: tuple[
        tuple[str, tuple[Round74AIExecutionReplayInstruction, ...]],
        ...,
    ]
    executions_by_manifest: tuple[
        tuple[str, tuple[Round74AIExecutionReplayEvidence, ...]],
        ...,
    ]
    development_reports: tuple[Round74AIUpliftDevelopmentReport, ...]
    qualification: Round74AIPretestQualificationPanel
    schema_version: str = ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    model_selection_performed: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        self.qualification_population.validate()
        self.inference.validate()
        self.baseline_trace.validate()
        self.review_panel.validate()
        for report in self.development_reports:
            report.validate()
        self.qualification.validate()
        instructions = dict(self.instructions_by_manifest)
        executions = dict(self.executions_by_manifest)
        panel_manifests = tuple(
            binding.manifest.manifest_sha256
            for binding in self.review_panel.model_bindings
        )
        report_by_manifest = {
            report.model_manifest_sha256: report for report in self.development_reports
        }
        observed_runs = tuple(dict.fromkeys(self.baseline_trace.run_id))
        observed_run_set = set(observed_runs)
        if (
            self.schema_version != ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION
            or self.inference.data_scope != "ai_qualification_tuning"
            or self.inference.expected_run_ids != self.qualification_population.run_ids
            or self.baseline_trace.expected_run_ids
            != self.qualification_population.run_ids
            or observed_runs
            != tuple(
                run_id
                for run_id in self.qualification_population.run_ids
                if run_id in observed_run_set
            )
            or self.review_panel.candidate_inference_sha256
            != self.inference.inference_sha256
            or tuple(instructions) != panel_manifests
            or tuple(executions) != panel_manifests
            or len(instructions) != 2
            or set(report_by_manifest) != set(panel_manifests)
            or tuple(
                report.report_sha256
                for report in self.qualification.development_reports
            )
            != tuple(
                report.report_sha256
                for report in sorted(
                    self.development_reports,
                    key=lambda value: value.model_manifest_sha256,
                )
            )
            or any(
                not isinstance(value, bool)
                for value in (
                    self.sealed_test_accessed,
                    self.model_selection_performed,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
            or any(
                (
                    self.sealed_test_accessed,
                    self.model_selection_performed,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 AI qualification operator result differs")
        expected_rows = self.baseline_trace.row_index
        for manifest in panel_manifests:
            rows = instructions[manifest]
            replay_rows = executions[manifest]
            report = report_by_manifest[manifest]
            for row in rows:
                row.validate()
            for row in replay_rows:
                row.validate()
            if (
                tuple(row.row_index for row in rows) != expected_rows
                or tuple(row.row_index for row in replay_rows) != expected_rows
                or tuple(row.source_review_sha256 for row in rows)
                != report.review_sha256
                or tuple(row.replay_sha256 for row in replay_rows)
                != report.execution_replay_sha256
                or any(
                    replay.source_review_sha256 != instruction.source_review_sha256
                    or replay.feature_row_sha256 != instruction.feature_row_sha256
                    or replay.run_id != instruction.run_id
                    or replay.symbol != instruction.symbol
                    or replay.side != instruction.side
                    or replay.horizon_seconds != instruction.horizon_seconds
                    for instruction, replay in zip(
                        rows,
                        replay_rows,
                        strict=True,
                    )
                )
                or tuple(
                    candidate.candidate_sha256
                    for candidate in self.inference.candidates
                )
                != report.candidate_sha256
            ):
                raise ValueError(
                    "Round 74 AI qualification instruction coverage differs"
                )


def _validate_qualification_inputs(
    batches: tuple[Round74EventTrainingBatch, ...],
    *,
    population: Round74AIQualificationPopulation,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
) -> None:
    population.validate()
    action_selection.validate()
    probability_calibration.validate()
    if not batches:
        raise ValueError("Round 74 AI qualification batches are missing")
    for batch in batches:
        batch.validate()
    batch_run_ids = tuple(
        next(iter(set(batch.run_id)))
        for batch in batches
        if len(set(batch.run_id)) == 1
    )
    policy_run_ids = {
        run_id
        for evaluation in action_selection.evaluations
        for run_id in evaluation.trace.expected_run_ids
    }
    if (
        len(batch_run_ids) != len(batches)
        or batch_run_ids != population.run_ids
        or any(batch.role != "tuning" for batch in batches)
        or len({batch.partition_sha256 for batch in batches}) != 1
        or len({batch.scaler_sha256 for batch in batches}) != 1
        or len({batch.window_representation for batch in batches}) != 1
        or action_selection.tuning_subpartition_sha256
        != population.parent_tuning_subpartition_sha256
        or action_selection.probability_calibration_sha256
        != probability_calibration.calibration_sha256
        or action_selection.pretest_policy_sha256
        != probability_calibration.pretest_policy_sha256
        or action_selection.optimization_population
        != population.optimization_population
        or probability_calibration.optimization_population
        != population.optimization_population
        or not action_selection.accepted
        or action_selection.selected_threshold_score is None
        or not policy_run_ids.issubset(set(population.prior_run_ids))
        or policy_run_ids.intersection(population.run_ids)
        or set(action_selection.target_batch_sha256).intersection(
            batch.batch_sha256 for batch in batches
        )
    ):
        raise ValueError("Round 74 AI qualification input identity differs")


def run_round74_ai_pretest_qualification(
    qualification_batches: Sequence[Round74EventTrainingBatch],
    *,
    qualification_population: Round74AIQualificationPopulation,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: str | Path,
    execution_replay_provider: Round74AIQualificationExecutionReplayProvider,
    qualification_output_path: str | Path,
    compute_backend: str = "auto",
    inference_minibatch_rows: int = 2_048,
    model_bindings: Sequence[Round74AIReviewModelBinding] | None = None,
    review_runner: ReviewRunner = review_round74_ai_candidate,
    model_batch_preparer: ModelBatchHook | None = None,
    model_batch_finalizer: ModelBatchHook | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Round74AIQualificationOperatorResult:
    """Run both pinned AI reviewers on the fourth tuning subpopulation."""

    batches = tuple(qualification_batches)
    _validate_qualification_inputs(
        batches,
        population=qualification_population,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
    )
    if not callable(execution_replay_provider):
        raise ValueError("Round 74 AI qualification execution input differs")
    contexts = tuple(build_round74_action_inference_context(batch) for batch in batches)
    inference = infer_round74_target_free_candidates(
        contexts,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
        pretest_policy_path=pretest_policy_path,
        compute_backend=compute_backend,
        minibatch_rows=inference_minibatch_rows,
        data_scope="ai_qualification_tuning",
        expected_run_ids=qualification_population.run_ids,
    )
    threshold = action_selection.selected_threshold_score
    assert threshold is not None
    baseline_trace = simulate_round74_action_trace_batches(
        batches,
        inference.candidates,
        threshold_score=threshold,
        expected_run_ids=qualification_population.run_ids,
    )
    review_panel = prepare_round74_target_free_ai_reviews(
        inference,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
        model_bindings=model_bindings,
        review_runner=review_runner,
        model_batch_preparer=model_batch_preparer,
        model_batch_finalizer=model_batch_finalizer,
        progress_callback=progress_callback,
    )
    selected_reviews_by_manifest: dict[
        str,
        tuple[Round74AIPairedReviewEvidence, ...],
    ] = {}
    instructions_by_manifest: dict[
        str,
        tuple[Round74AIExecutionReplayInstruction, ...],
    ] = {}
    for manifest, reviews in review_panel.reviews_by_manifest().items():
        by_row = {review.row_index: review for review in reviews}
        try:
            selected_reviews = tuple(
                by_row[row_index] for row_index in baseline_trace.row_index
            )
        except KeyError as exc:
            raise ValueError("Round 74 AI qualification review is missing") from exc
        selected_reviews_by_manifest[manifest] = selected_reviews
        instructions_by_manifest[manifest] = (
            build_round74_ai_execution_replay_instructions(
                action_selection,
                contexts=contexts,
                reviews=selected_reviews,
                trace=baseline_trace,
            )
        )
    replayed = {
        str(manifest): tuple(rows)
        for manifest, rows in execution_replay_provider(
            qualification_population=qualification_population,
            action_selection=action_selection,
            instructions_by_manifest=instructions_by_manifest,
        ).items()
    }
    if tuple(replayed) != tuple(instructions_by_manifest):
        raise ValueError("Round 74 AI qualification replay panel differs")
    reports = tuple(
        evaluate_round74_ai_overlay_development(
            action_selection,
            selected_reviews_by_manifest[manifest],
            replayed[manifest],
            qualification_population=qualification_population,
            qualification_trace=baseline_trace,
            qualification_candidate_sha256=tuple(
                candidate.candidate_sha256 for candidate in inference.candidates
            ),
        )
        for manifest in instructions_by_manifest
    )
    qualification = build_round74_ai_pretest_qualification(reports)
    write_round74_ai_pretest_qualification(
        qualification,
        qualification_output_path,
    )
    result = Round74AIQualificationOperatorResult(
        qualification_population=qualification_population,
        inference=inference,
        baseline_trace=baseline_trace,
        review_panel=review_panel,
        instructions_by_manifest=tuple(instructions_by_manifest.items()),
        executions_by_manifest=tuple(replayed.items()),
        development_reports=reports,
        qualification=qualification,
    )
    result.validate()
    return result


def run_round74_prepared_ai_pretest_qualification(
    prepared_tuning_roles: Round74PreparedTuningRoles,
    *,
    qualification_population: Round74AIQualificationPopulation,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: str | Path,
    execution_replay_provider: Round74AIQualificationExecutionReplayProvider,
    qualification_output_path: str | Path,
    compute_backend: str = "auto",
    inference_minibatch_rows: int = 2_048,
    model_bindings: Sequence[Round74AIReviewModelBinding] | None = None,
    review_runner: ReviewRunner = review_round74_ai_candidate,
    model_batch_preparer: ModelBatchHook | None = None,
    model_batch_finalizer: ModelBatchHook | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Round74AIQualificationOperatorResult:
    """Run qualification directly from the validated four-way tuning split."""

    if not isinstance(prepared_tuning_roles, Round74PreparedTuningRoles):
        raise TypeError("Round 74 prepared tuning roles are required")
    prepared_tuning_roles.validate()
    qualification_population.validate()
    if (
        not prepared_tuning_roles.ai_qualification_batches
        or tuple(
            next(iter(set(batch.run_id)))
            for batch in prepared_tuning_roles.ai_qualification_batches
        )
        != qualification_population.run_ids
    ):
        raise ValueError("Round 74 prepared AI qualification population differs")
    return run_round74_ai_pretest_qualification(
        prepared_tuning_roles.ai_qualification_batches,
        qualification_population=qualification_population,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
        pretest_policy_path=pretest_policy_path,
        execution_replay_provider=execution_replay_provider,
        qualification_output_path=qualification_output_path,
        compute_backend=compute_backend,
        inference_minibatch_rows=inference_minibatch_rows,
        model_bindings=model_bindings,
        review_runner=review_runner,
        model_batch_preparer=model_batch_preparer,
        model_batch_finalizer=model_batch_finalizer,
        progress_callback=progress_callback,
    )


__all__ = [
    "ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION",
    "Round74AIQualificationExecutionReplayProvider",
    "Round74AIQualificationOperatorResult",
    "run_round74_ai_pretest_qualification",
    "run_round74_prepared_ai_pretest_qualification",
]
