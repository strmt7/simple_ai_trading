"""Target-free two-model AI review preparation for Round 74 sealed evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    review_round74_ai_candidate,
)
from .impact_absorption_ai_uplift import Round74AIPairedReviewEvidence
from .impact_absorption_event_action_policy import Round74ActionPolicySelection
from .impact_absorption_event_calibration import Round74ProbabilityCalibration
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sealed_evaluation import (
    Round74TargetFreeCandidateInference,
)
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)


ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION = "round-074-ai-review-panel-v1"
ROUND74_AI_REVIEW_VALIDITY_NS = 30_000_000_000
ROUND74_AI_REVIEW_UNIT_RISK_BPS = 10_000

_SHA256 = re.compile(r"[0-9a-f]{64}")

ReviewRunner = Callable[..., Round74AIRuntimeOutcome]
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
    rows: tuple[Round74TargetFreeReviewRow, ...]
    model_bindings: tuple[Round74AIReviewModelBinding, ...]
    reviews: tuple[tuple[Round74AIPairedReviewEvidence, ...], ...]
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
            or not self.model_bindings
            or len(self.model_bindings) != len(self.reviews)
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
                or any(
                    review.model_manifest_sha256 != manifest
                    for review in reviews
                )
                or len({review.review_sha256 for review in reviews})
                != len(reviews)
            ):
                raise ValueError("Round 74 AI review evidence coverage differs")
            for row, review in zip(self.rows, reviews, strict=True):
                if (
                    review.feature_row_sha256 != row.feature_row_sha256
                    or review.run_id != row.run_id
                    or review.symbol != row.symbol
                    or review.side != row.side
                    or review.horizon_seconds != row.horizon_seconds
                    or review.pretest_policy_sha256
                    != self.pretest_policy_sha256
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
            "probability_calibration_sha256": (
                self.probability_calibration_sha256
            ),
            "profile": self.profile,
            "rows": [row.as_dict() for row in self.rows],
            "model_bindings": [
                binding.as_dict() for binding in self.model_bindings
            ],
            "reviews": [
                [review.as_dict() for review in model_reviews]
                for model_reviews in self.reviews
            ],
            "review_coverage_defined_before_target_scoring": True,
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
        payoff_quantiles_bps=output.payoff_quantiles_bps[
            row_index : row_index + 1
        ],
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps[
                row_index : row_index + 1
            ]
        ),
        positive_payoff_logits=output.positive_payoff_logits[
            row_index : row_index + 1
        ],
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
                    "action_selection_sha256": (
                        inference.action_selection_sha256
                    ),
                    "profile": inference.profile,
                    "row_index": row_index,
                    "feature_row_sha256": (
                        candidates.feature_row_sha256[local_index]
                    ),
                    "candidate_sha256": candidates.candidate_sha256,
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
                    feature_row_sha256=(
                        candidates.feature_row_sha256[local_index]
                    ),
                    run_id=context.run_id[local_index],
                    symbol=context.symbol[local_index],
                    side=int(candidates.side[local_index]),
                    horizon_seconds=int(
                        candidates.horizon_seconds[local_index]
                    ),
                    candidate_sha256=candidates.candidate_sha256,
                    deterministic_risk_state_sha256=risk_state,
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
    model_bindings: Sequence[Round74AIReviewModelBinding] | None = None,
    review_runner: ReviewRunner = review_round74_ai_candidate,
    progress_callback: ProgressCallback | None = None,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> Round74AIReviewPanel:
    """Review every frozen target-free candidate for both pinned local models."""

    inference.validate()
    action_selection.validate()
    probability_calibration.validate()
    if (
        not action_selection.accepted
        or action_selection.selected_threshold_score is None
        or inference.action_selection_sha256
        != action_selection.selection_sha256
        or inference.pretest_policy_sha256
        != action_selection.pretest_policy_sha256
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
    total = len(bindings) * len(rows)
    completed = 0
    for binding in bindings:
        reviews: list[Round74AIPairedReviewEvidence] = []
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
                probability_calibration=probability_calibration,
                requested_wall_ns=requested_wall_ns,
                expires_wall_ns=(
                    requested_wall_ns + ROUND74_AI_REVIEW_VALIDITY_NS
                ),
                proposed_risk_size_bps=ROUND74_AI_REVIEW_UNIT_RISK_BPS,
            )
            outcome = review_runner(
                binding.runtime,
                binding.manifest,
                request,
                deterministic_risk_gate_passed=True,
                observed_wall_ns=requested_wall_ns,
            )
            outcome.validate()
            evidence = Round74AIPairedReviewEvidence.from_runtime(
                row_index=row.row_index,
                feature_row_sha256=row.feature_row_sha256,
                run_id=row.run_id,
                symbol=row.symbol,
                side=row.side,
                horizon_seconds=row.horizon_seconds,
                request=request,
                outcome=outcome,
            )
            reviews.append(evidence)
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed_reviews": completed,
                        "total_reviews": total,
                        "model_role": binding.role,
                        "model_manifest_sha256": (
                            binding.manifest.manifest_sha256
                        ),
                        "row_index": row.row_index,
                        "runtime_status": outcome.status,
                    }
                )
        all_reviews.append(tuple(reviews))
    result = Round74AIReviewPanel(
        candidate_inference_sha256=inference.inference_sha256,
        action_selection_sha256=action_selection.selection_sha256,
        pretest_policy_sha256=action_selection.pretest_policy_sha256,
        probability_calibration_sha256=(
            probability_calibration.calibration_sha256
        ),
        profile=action_selection.profile,
        rows=rows,
        model_bindings=bindings,
        reviews=tuple(all_reviews),
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION",
    "ROUND74_AI_REVIEW_UNIT_RISK_BPS",
    "ROUND74_AI_REVIEW_VALIDITY_NS",
    "Round74AIReviewModelBinding",
    "Round74AIReviewPanel",
    "Round74TargetFreeReviewRow",
    "prepare_round74_target_free_ai_reviews",
    "round74_default_ai_review_model_panel",
]
