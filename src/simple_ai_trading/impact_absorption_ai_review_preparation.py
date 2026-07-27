"""Target-free two-model AI review preparation for Round 74 sealed evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
import time

import numpy as np
import torch

from .impact_absorption_ai_bridge import build_round74_ai_review_request
from .impact_absorption_ai_protocol import Round74AIModelManifest
from .impact_absorption_ai_runtime import (
    Round74AIRuntimeConfig,
    Round74AIRuntimeOutcome,
    Round74AIWorkerSession,
    review_round74_ai_candidate,
    unload_round74_ai_model,
)
from .impact_absorption_ai_uplift import Round74AIPairedReviewEvidence
from .impact_absorption_event_action_policy import Round74ActionPolicySelection
from .impact_absorption_event_calibration import Round74ProbabilityCalibration
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sealed_evaluation import (
    Round74TargetFreeCandidateInference,
)
from .impact_absorption_event_sealed_ledger import Round74SealedEvaluationClaim
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)


ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION = "round-074-ai-review-panel-v12"
ROUND74_AI_REVIEW_VALIDITY_NS = 30_000_000_000
ROUND74_AI_REVIEW_UNIT_RISK_BPS = 10_000

_SHA256 = re.compile(r"[0-9a-f]{64}")

ReviewRunner = Callable[..., Round74AIRuntimeOutcome]
ModelBatchFinalizer = Callable[["Round74AIReviewModelBinding"], object]
ProgressCallback = Callable[[Mapping[str, object]], None]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _finalize_round74_ai_model_batch(
    binding: "Round74AIReviewModelBinding",
) -> object:
    return unload_round74_ai_model(binding.runtime, binding.manifest)


@dataclass(frozen=True)
class Round74AIReviewModelBinding:
    """Pinned local model plus its fail-closed runtime limits."""

    model_name: str
    manifest: Round74AIModelManifest
    runtime: Round74AIRuntimeConfig
    role: str

    def validate(self) -> None:
        self.manifest.validate()
        self.runtime.validate()
        if (
            self.role not in ("finance_primary", "general_control")
            or self.runtime.model_name != self.model_name
            or self.manifest.runtime_version != "0.32.4"
        ):
            raise ValueError("Round 74 AI review model binding differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "role": self.role,
            "model_name": self.model_name,
            "manifest": self.manifest.as_dict(),
            "runtime": {
                "endpoint": self.runtime.endpoint,
                "timeout_seconds": self.runtime.timeout_seconds,
                "minimum_free_ram_gb": self.runtime.minimum_free_ram_gb,
                "minimum_free_vram_gb": self.runtime.minimum_free_vram_gb,
            },
        }


def round74_default_ai_review_model_panel() -> tuple[
    Round74AIReviewModelBinding,
    ...,
]:
    """Return the predeclared finance primary and general 8B control."""

    panel = (
        Round74AIReviewModelBinding(
            model_name="fino1:8b",
            manifest=Round74AIModelManifest(
                model_id="TheFinAI/Fino1-8B",
                model_revision="1044f6211d3c061002c985bd820c27eab291d7af",
                model_artifact_sha256=(
                    "083c6422a2dd90d62ec33638ab84271edddd2cf1fa6a9841898ea18a35e27b87"
                ),
                model_artifact_kind="ollama_manifest",
                parameter_count=8_030_000_000,
                quantization="q6_k",
                runtime_backend="llama.cpp-vulkan",
                runtime_version="0.32.4",
                license_id="llama3.1",
                model_card_url="https://huggingface.co/TheFinAI/Fino1-8B",
                minimum_vram_bytes=8 * 1024**3,
                finance_specialized=True,
            ),
            runtime=Round74AIRuntimeConfig(model_name="fino1:8b"),
            role="finance_primary",
        ),
        Round74AIReviewModelBinding(
            model_name="qwen3:8b",
            manifest=Round74AIModelManifest(
                model_id="Qwen/Qwen3-8B",
                model_revision="b968826d9c46dd6066d109eabc6255188de91218",
                model_artifact_sha256=(
                    "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
                ),
                model_artifact_kind="ollama_manifest",
                parameter_count=8_200_000_000,
                quantization="q4_k_m",
                runtime_backend="llama.cpp-vulkan",
                runtime_version="0.32.4",
                license_id="Apache-2.0",
                model_card_url="https://huggingface.co/Qwen/Qwen3-8B",
                minimum_vram_bytes=8 * 1024**3,
                finance_specialized=False,
            ),
            runtime=Round74AIRuntimeConfig(model_name="qwen3:8b"),
            role="general_control",
        ),
    )
    for value in panel:
        value.validate()
    if len({value.manifest.manifest_sha256 for value in panel}) != len(panel):
        raise ValueError("Round 74 AI review manifests are duplicated")
    return panel


@dataclass(frozen=True)
class Round74TargetFreeReviewRow:
    """One model candidate that must be reviewed before target scoring."""

    row_index: int
    panel_index: int
    local_row_index: int
    feature_row_sha256: str
    run_id: str
    symbol: str
    side: int
    horizon_seconds: int
    candidate_sha256: str
    deterministic_risk_state_sha256: str
    decision_wall_ns: int

    def validate(self) -> None:
        if (
            isinstance(self.row_index, bool)
            or self.row_index < 0
            or isinstance(self.panel_index, bool)
            or self.panel_index < 0
            or isinstance(self.local_row_index, bool)
            or self.local_row_index < 0
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.feature_row_sha256,
                    self.candidate_sha256,
                    self.deterministic_risk_state_sha256,
                )
            )
            or len(self.run_id) != 32
            or any(value not in "0123456789abcdef" for value in self.run_id)
            or self.symbol not in ROUND74_EVENT_SYMBOLS
            or self.side not in (-1, 1)
            or self.horizon_seconds not in (30, 300)
            or isinstance(self.decision_wall_ns, bool)
            or not isinstance(self.decision_wall_ns, int)
            or self.decision_wall_ns <= 0
        ):
            raise ValueError("Round 74 target-free review row differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


@dataclass(frozen=True)
class Round74AIReviewPanel:
    """Hash-bound target-free runtime evidence for both predeclared models."""

    candidate_inference_sha256: str
    action_selection_sha256: str
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    profile: str
    same_entry_latency_budget_ns: int
    rows: tuple[Round74TargetFreeReviewRow, ...]
    model_bindings: tuple[Round74AIReviewModelBinding, ...]
    reviews: tuple[tuple[Round74AIPairedReviewEvidence, ...], ...]
    model_batch_unload_enforced: bool
    schema_version: str = ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION
    target_fields_accessed: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.candidate_inference_sha256,
                    self.action_selection_sha256,
                    self.pretest_policy_sha256,
                    self.probability_calibration_sha256,
                )
            )
            or self.profile not in ("conservative", "regular", "aggressive")
            or isinstance(self.same_entry_latency_budget_ns, bool)
            or not isinstance(self.same_entry_latency_budget_ns, int)
            or self.same_entry_latency_budget_ns <= 0
            or not self.model_bindings
            or len(self.model_bindings) != len(self.reviews)
            or not isinstance(self.model_batch_unload_enforced, bool)
            or self.target_fields_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 74 AI review panel differs")
        for row in self.rows:
            row.validate()
        if tuple(row.row_index for row in self.rows) != tuple(
            sorted(row.row_index for row in self.rows)
        ) or len({row.row_index for row in self.rows}) != len(self.rows):
            raise ValueError("Round 74 AI review row order differs")
        if any(
            current.decision_wall_ns < prior.decision_wall_ns
            for prior, current in zip(self.rows, self.rows[1:])
        ):
            raise ValueError("Round 74 AI review wall-time order differs")
        manifest_sha256: set[str] = set()
        expected_rows = tuple(row.row_index for row in self.rows)
        for binding, reviews in zip(
            self.model_bindings,
            self.reviews,
            strict=True,
        ):
            binding.validate()
            manifest = binding.manifest.manifest_sha256
            if manifest in manifest_sha256:
                raise ValueError("Round 74 AI review model identity is duplicated")
            manifest_sha256.add(manifest)
            for review in reviews:
                review.validate()
            if (
                len(reviews) != len(self.rows)
                or tuple(review.row_index for review in reviews) != expected_rows
                or any(review.model_manifest_sha256 != manifest for review in reviews)
                or any(
                    review.same_entry_latency_budget_ns
                    != self.same_entry_latency_budget_ns
                    for review in reviews
                )
                or len({review.review_sha256 for review in reviews}) != len(reviews)
            ):
                raise ValueError("Round 74 AI review evidence coverage differs")
            for row, review in zip(self.rows, reviews, strict=True):
                if (
                    review.feature_row_sha256 != row.feature_row_sha256
                    or review.run_id != row.run_id
                    or review.symbol != row.symbol
                    or review.side != row.side
                    or review.horizon_seconds != row.horizon_seconds
                    or review.pretest_policy_sha256 != self.pretest_policy_sha256
                    or review.probability_calibration_sha256
                    != self.probability_calibration_sha256
                ):
                    raise ValueError("Round 74 AI review evidence identity differs")

    @property
    def panel_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "candidate_inference_sha256": self.candidate_inference_sha256,
            "action_selection_sha256": self.action_selection_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "profile": self.profile,
            "same_entry_latency_budget_ns": self.same_entry_latency_budget_ns,
            "rows": [row.as_dict() for row in self.rows],
            "model_bindings": [binding.as_dict() for binding in self.model_bindings],
            "reviews": [
                [review.as_dict() for review in model_reviews]
                for model_reviews in self.reviews
            ],
            "model_batch_unload_enforced": self.model_batch_unload_enforced,
            "model_batch_unload_policy": (
                "exact_declared_model_only_verified_absent_before_next_batch"
                if self.model_batch_unload_enforced
                else "not_applicable_to_injected_review_runner"
            ),
            "review_coverage_defined_before_target_scoring": True,
            "same_entry_latency_budget_policy": (
                "externally_measured_signal_to_entry_slack"
            ),
            "queue_model": (
                "single_server_per_candidate_model_in_historical_decision_order"
            ),
            "queue_delay_included_in_same_entry_eligibility": True,
            "latency_adjusted_replay_performed": False,
            "target_fields_accessed": False,
            "trading_authority": False,
        }
        if include_sha256:
            value["panel_sha256"] = _canonical_sha256(value)
        return value

    def reviews_by_manifest(
        self,
    ) -> dict[str, tuple[Round74AIPairedReviewEvidence, ...]]:
        self.validate()
        return {
            binding.manifest.manifest_sha256: reviews
            for binding, reviews in zip(
                self.model_bindings,
                self.reviews,
                strict=True,
            )
        }


def _row_output(
    output: Round74EventModelOutput,
    row_index: int,
) -> Round74EventModelOutput:
    result = Round74EventModelOutput(
        payoff_quantiles_bps=output.payoff_quantiles_bps[row_index : row_index + 1],
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps[row_index : row_index + 1]
        ),
        positive_payoff_logits=output.positive_payoff_logits[row_index : row_index + 1],
        adverse_selection_logits=output.adverse_selection_logits[
            row_index : row_index + 1
        ],
        regime_unpredictability_logits=output.regime_unpredictability_logits[
            row_index : row_index + 1
        ],
    )
    result.validate(1)
    return result


def _review_rows(
    inference: Round74TargetFreeCandidateInference,
    *,
    threshold_score: float,
) -> tuple[Round74TargetFreeReviewRow, ...]:
    rows: list[Round74TargetFreeReviewRow] = []
    offset = 0
    for panel_index, (context, candidates) in enumerate(
        zip(inference.contexts, inference.candidates, strict=True)
    ):
        for local_index in range(context.rows):
            if (
                not candidates.eligible[local_index]
                or candidates.quality_score[local_index] < threshold_score
            ):
                continue
            row_index = offset + local_index
            risk_state = _canonical_sha256(
                {
                    "schema_version": ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
                    "candidate_inference_sha256": inference.inference_sha256,
                    "action_selection_sha256": (inference.action_selection_sha256),
                    "profile": inference.profile,
                    "row_index": row_index,
                    "feature_row_sha256": (candidates.feature_row_sha256[local_index]),
                    "candidate_sha256": candidates.candidate_sha256,
                    "decision_wall_ns": int(context.decision_wall_ns[local_index]),
                    "paired_evaluation_unit_risk_bps": (
                        ROUND74_AI_REVIEW_UNIT_RISK_BPS
                    ),
                    "live_account_state_used": False,
                }
            )
            rows.append(
                Round74TargetFreeReviewRow(
                    row_index=row_index,
                    panel_index=panel_index,
                    local_row_index=local_index,
                    feature_row_sha256=(candidates.feature_row_sha256[local_index]),
                    run_id=context.run_id[local_index],
                    symbol=context.symbol[local_index],
                    side=int(candidates.side[local_index]),
                    horizon_seconds=int(candidates.horizon_seconds[local_index]),
                    candidate_sha256=candidates.candidate_sha256,
                    deterministic_risk_state_sha256=risk_state,
                    decision_wall_ns=int(context.decision_wall_ns[local_index]),
                )
            )
        offset += context.rows
    result = tuple(rows)
    for value in result:
        value.validate()
    return result


def prepare_round74_target_free_ai_reviews(
    inference: Round74TargetFreeCandidateInference,
    *,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    same_entry_latency_budget_ns: int,
    model_bindings: Sequence[Round74AIReviewModelBinding] | None = None,
    review_runner: ReviewRunner = review_round74_ai_candidate,
    model_batch_finalizer: ModelBatchFinalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> Round74AIReviewPanel:
    """Review every frozen target-free candidate for both pinned local models."""

    inference.validate()
    action_selection.validate()
    probability_calibration.validate()
    if (
        isinstance(same_entry_latency_budget_ns, bool)
        or not isinstance(same_entry_latency_budget_ns, int)
        or same_entry_latency_budget_ns <= 0
        or not action_selection.accepted
        or action_selection.selected_threshold_score is None
        or inference.action_selection_sha256 != action_selection.selection_sha256
        or inference.pretest_policy_sha256 != action_selection.pretest_policy_sha256
        or inference.probability_calibration_sha256
        != probability_calibration.calibration_sha256
        or inference.profile != action_selection.profile
    ):
        raise ValueError("Round 74 AI review preparation identity differs")
    bindings = tuple(
        model_bindings
        if model_bindings is not None
        else round74_default_ai_review_model_panel()
    )
    if len(bindings) != 2 or tuple(value.role for value in bindings) != (
        "finance_primary",
        "general_control",
    ):
        raise ValueError("Round 74 AI review requires the two-model panel")
    for value in bindings:
        value.validate()
    threshold = action_selection.selected_threshold_score
    assert threshold is not None
    rows = _review_rows(inference, threshold_score=threshold)
    all_reviews: list[tuple[Round74AIPairedReviewEvidence, ...]] = []
    selected_batch_finalizer = model_batch_finalizer
    if (
        selected_batch_finalizer is None
        and review_runner is review_round74_ai_candidate
    ):
        selected_batch_finalizer = _finalize_round74_ai_model_batch
    total = len(bindings) * len(rows)
    completed = 0
    for binding in bindings:
        reviews: list[Round74AIPairedReviewEvidence] = []
        model_available_wall_ns = 0
        batch_attempted = False
        worker_session = (
            Round74AIWorkerSession()
            if review_runner is review_round74_ai_candidate
            else None
        )
        try:
            for row in rows:
                context = inference.contexts[row.panel_index]
                output = inference.model_outputs[row.panel_index]
                feature_values = torch.from_numpy(
                    np.array(
                        context.feature_values[
                            row.local_row_index : row.local_row_index + 1
                        ],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                )
                requested_wall_ns = int(wall_time_ns())
                request = build_round74_ai_review_request(
                    model_output=_row_output(output, row.local_row_index),
                    scaled_feature_values=feature_values,
                    row_index=0,
                    asset_slot=ROUND74_EVENT_SYMBOLS.index(row.symbol),
                    side=(
                        ROUND74_EVENT_PAYOFF_SIDES[0]
                        if row.side == 1
                        else ROUND74_EVENT_PAYOFF_SIDES[1]
                    ),
                    horizon_seconds=row.horizon_seconds,
                    pretest_policy_sha256=inference.pretest_policy_sha256,
                    sample_sha256=row.feature_row_sha256,
                    deterministic_risk_state_sha256=(
                        row.deterministic_risk_state_sha256
                    ),
                    risk_profile=action_selection.profile,
                    probability_calibration=probability_calibration,
                    requested_wall_ns=requested_wall_ns,
                    expires_wall_ns=(requested_wall_ns + ROUND74_AI_REVIEW_VALIDITY_NS),
                    proposed_risk_size_bps=ROUND74_AI_REVIEW_UNIT_RISK_BPS,
                )
                batch_attempted = True
                if worker_session is None:
                    outcome = review_runner(
                        binding.runtime,
                        binding.manifest,
                        request,
                        deterministic_risk_gate_passed=True,
                        observed_wall_ns=requested_wall_ns,
                    )
                else:
                    outcome = review_round74_ai_candidate(
                        binding.runtime,
                        binding.manifest,
                        request,
                        deterministic_risk_gate_passed=True,
                        observed_wall_ns=requested_wall_ns,
                        worker_session=worker_session,
                    )
                outcome.validate()
                queue_delay_ns = max(
                    0,
                    model_available_wall_ns - row.decision_wall_ns,
                )
                model_available_wall_ns = (
                    max(model_available_wall_ns, row.decision_wall_ns)
                    + outcome.elapsed_ns
                )
                evidence = Round74AIPairedReviewEvidence.from_runtime(
                    row_index=row.row_index,
                    feature_row_sha256=row.feature_row_sha256,
                    run_id=row.run_id,
                    symbol=row.symbol,
                    side=row.side,
                    horizon_seconds=row.horizon_seconds,
                    request=request,
                    outcome=outcome,
                    same_entry_latency_budget_ns=same_entry_latency_budget_ns,
                    queue_delay_ns=queue_delay_ns,
                )
                reviews.append(evidence)
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "completed_reviews": completed,
                            "total_reviews": total,
                            "model_role": binding.role,
                            "model_manifest_sha256": (binding.manifest.manifest_sha256),
                            "row_index": row.row_index,
                            "runtime_status": outcome.status,
                            "runtime_elapsed_ns": outcome.elapsed_ns,
                            "queue_delay_ns": evidence.queue_delay_ns,
                            "effective_review_latency_ns": (
                                evidence.effective_review_latency_ns
                            ),
                            "same_entry_latency_eligible": (
                                evidence.same_entry_latency_eligible
                            ),
                        }
                    )
        finally:
            if worker_session is not None:
                worker_session.close()
            if batch_attempted and selected_batch_finalizer is not None:
                selected_batch_finalizer(binding)
        all_reviews.append(tuple(reviews))
    result = Round74AIReviewPanel(
        candidate_inference_sha256=inference.inference_sha256,
        action_selection_sha256=action_selection.selection_sha256,
        pretest_policy_sha256=action_selection.pretest_policy_sha256,
        probability_calibration_sha256=(probability_calibration.calibration_sha256),
        profile=action_selection.profile,
        same_entry_latency_budget_ns=same_entry_latency_budget_ns,
        rows=rows,
        model_bindings=bindings,
        reviews=tuple(all_reviews),
        model_batch_unload_enforced=(selected_batch_finalizer is not None),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74PreparedSealedAIReviewProvider:
    """Run the pinned target-free two-model panel for a reserved test claim."""

    probability_calibration: Round74ProbabilityCalibration
    same_entry_latency_budget_ns: int
    model_bindings: tuple[Round74AIReviewModelBinding, ...] = field(
        default_factory=round74_default_ai_review_model_panel
    )
    review_runner: ReviewRunner = review_round74_ai_candidate
    model_batch_finalizer: ModelBatchFinalizer | None = None
    progress_callback: ProgressCallback | None = None
    wall_time_ns: Callable[[], int] = time.time_ns

    def __post_init__(self) -> None:
        bindings = tuple(self.model_bindings)
        object.__setattr__(self, "model_bindings", bindings)
        self.probability_calibration.validate()
        if (
            isinstance(self.same_entry_latency_budget_ns, bool)
            or not isinstance(self.same_entry_latency_budget_ns, int)
            or self.same_entry_latency_budget_ns <= 0
            or len(bindings) != 2
            or tuple(value.role for value in bindings)
            != ("finance_primary", "general_control")
            or not callable(self.review_runner)
            or (
                self.model_batch_finalizer is not None
                and not callable(self.model_batch_finalizer)
            )
            or (
                self.progress_callback is not None
                and not callable(self.progress_callback)
            )
            or not callable(self.wall_time_ns)
        ):
            raise ValueError("Round 74 sealed AI review provider differs")
        for binding in bindings:
            binding.validate()

    def __call__(
        self,
        *,
        claim: Round74SealedEvaluationClaim,
        manifests: tuple[str, ...],
        inference: Round74TargetFreeCandidateInference,
        action_selection: Round74ActionPolicySelection,
    ) -> dict[str, tuple[Round74AIPairedReviewEvidence, ...]]:
        claim.validate()
        inference.validate()
        action_selection.validate()
        requested_manifests = tuple(manifests)
        bound_manifests = tuple(
            value.manifest.manifest_sha256 for value in self.model_bindings
        )
        context_run_ids = tuple(
            dict.fromkeys(
                run_id for context in inference.contexts for run_id in context.run_id
            )
        )
        context_partitions = {
            context.partition_sha256 for context in inference.contexts
        }
        if (
            claim.status != "reserved"
            or requested_manifests != claim.ai_manifest_sha256
            or set(bound_manifests) != set(requested_manifests)
            or action_selection.selection_sha256 != claim.action_selection_sha256
            or action_selection.pretest_policy_sha256 != claim.pretest_policy_sha256
            or action_selection.probability_calibration_sha256
            != claim.probability_calibration_sha256
            or action_selection.profile != claim.profile
            or inference.action_selection_sha256 != claim.action_selection_sha256
            or inference.pretest_policy_sha256 != claim.pretest_policy_sha256
            or inference.probability_calibration_sha256
            != claim.probability_calibration_sha256
            or inference.profile != claim.profile
            or self.probability_calibration.calibration_sha256
            != claim.probability_calibration_sha256
            or context_run_ids != claim.test_run_ids
            or context_partitions != {claim.partition_sha256}
        ):
            raise ValueError("Round 74 sealed AI review identity differs")
        panel = prepare_round74_target_free_ai_reviews(
            inference,
            action_selection=action_selection,
            probability_calibration=self.probability_calibration,
            same_entry_latency_budget_ns=self.same_entry_latency_budget_ns,
            model_bindings=self.model_bindings,
            review_runner=self.review_runner,
            model_batch_finalizer=self.model_batch_finalizer,
            progress_callback=self.progress_callback,
            wall_time_ns=self.wall_time_ns,
        )
        reviews = panel.reviews_by_manifest()
        if set(reviews) != set(requested_manifests):
            raise ValueError("Round 74 sealed AI review output panel differs")
        return {manifest: reviews[manifest] for manifest in requested_manifests}


__all__ = [
    "ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION",
    "ROUND74_AI_REVIEW_UNIT_RISK_BPS",
    "ROUND74_AI_REVIEW_VALIDITY_NS",
    "Round74AIReviewModelBinding",
    "Round74AIReviewPanel",
    "Round74PreparedSealedAIReviewProvider",
    "Round74TargetFreeReviewRow",
    "prepare_round74_target_free_ai_reviews",
    "round74_default_ai_review_model_panel",
]
