"""Single-audit pretest training and publication orchestration for Round 73."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import time

from .impact_absorption_holdout_store import (
    Round73PretestManifestReport,
    publish_round73_pretest_manifest,
)
from .impact_absorption_model_slice import (
    ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES,
    iter_round73_staged_symbol_slices,
)
from .impact_absorption_shallow_model import (
    ProgressCallback,
    Round73PreparedPretestArtifacts,
    prepare_round73_pretest_artifacts,
)
from .impact_absorption_target_store_v3 import (
    Round73DevelopmentTargetStudyReport,
    seal_round73_development_targets,
)


ROUND73_PRETEST_TRAINING_SCHEMA_VERSION = "round-073-pretest-training-v1"


@dataclass(frozen=True)
class Round73PretestTrainingReport:
    development: Round73DevelopmentTargetStudyReport
    prepared: Round73PreparedPretestArtifacts
    publication: Round73PretestManifestReport
    wall_seconds: float

    def as_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": ROUND73_PRETEST_TRAINING_SCHEMA_VERSION,
            "development": self.development.as_dict(),
            "training": self.prepared.as_dict(),
            "publication": self.publication.as_dict(),
            "wall_seconds": self.wall_seconds,
            "development_exact_wire_audit_count": 1,
            "development_seal_reused_for_loading_and_publication": True,
            "test_target_read": False,
            "model_evaluated": False,
            "profitability_claim": False,
            "trading_authority": False,
        }


def train_and_publish_round73_pretest(
    database: str | Path,
    *,
    study_id: str,
    repository_root: str | Path,
    compute_backend: str = "auto",
    memory_budget_bytes: int = ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES,
    memory_limit: str = "2GB",
    threads: int = 2,
    progress_callback: ProgressCallback | None = None,
) -> Round73PretestTrainingReport:
    """Audit development once, fit every symbol, and freeze every artifact."""

    started = time.perf_counter()
    if progress_callback is not None:
        progress_callback("development_seal_started", {"study_id": study_id})
    development = seal_round73_development_targets(
        database,
        study_id=study_id,
        memory_limit=memory_limit,
        threads=threads,
    )
    if progress_callback is not None:
        progress_callback(
            "development_seal_completed",
            {
                "study_id": study_id,
                "development_study_manifest_sha256": (
                    development.development_study_manifest_sha256
                ),
                "source_run_count": development.source_run_count,
            },
        )

    def reuse_development_seal(
        _database: str | Path,
        **kwargs: object,
    ) -> Round73DevelopmentTargetStudyReport:
        if (
            str(kwargs.get("study_id", "")).strip().lower()
            != str(study_id).strip().lower()
        ):
            raise ValueError("Round 73 reused development seal study differs")
        return development

    datasets = iter_round73_staged_symbol_slices(
        database,
        study_id=study_id,
        role_scope="development",
        memory_budget_bytes=memory_budget_bytes,
        memory_limit=memory_limit,
        threads=threads,
        development_seal_function=reuse_development_seal,
    )
    prepared = prepare_round73_pretest_artifacts(
        datasets,
        compute_backend=compute_backend,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            "pretest_publication_started",
            {
                "study_id": study_id,
                "artifact_count": len(prepared.artifacts),
            },
        )
    publication = publish_round73_pretest_manifest(
        database,
        study_id=study_id,
        model_manifest=prepared.model_manifest,
        artifacts=prepared.artifacts,
        repository_root=repository_root,
        memory_limit=memory_limit,
        threads=threads,
        development_seal_function=reuse_development_seal,
    )
    if (
        publication.development_study_manifest_sha256
        != development.development_study_manifest_sha256
    ):
        raise ValueError("Round 73 publication changed the development seal")
    if progress_callback is not None:
        progress_callback(
            "pretest_publication_completed",
            {
                "study_id": study_id,
                "pretest_manifest_sha256": publication.pretest_manifest_sha256,
                "artifact_count": publication.artifact_count,
                "artifact_bytes": publication.artifact_bytes,
            },
        )
    return Round73PretestTrainingReport(
        development=development,
        prepared=prepared,
        publication=publication,
        wall_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ROUND73_PRETEST_TRAINING_SCHEMA_VERSION",
    "Round73PretestTrainingReport",
    "train_and_publish_round73_pretest",
]
