"""Lazy, hash-bound TCN corpus sources backed by the Round 25 feature store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re

from .polymarket_round25_controls import Round25LogisticResidualArtifact
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    Round25DevelopmentDataset,
    Round25DevelopmentSample,
    require_round25_dataset_minimum,
)
from .polymarket_round25_joint_store import (
    load_round25_joint_condition_batch,
    load_round25_joint_store_manifest,
)
from .polymarket_round25_sequence import (
    Round25SequenceConditionBatch,
    build_round25_sequence_condition_batch,
    round25_feature_transform_sha256,
)
from .polymarket_round25_tcn import (
    Round25TCNCorpusSource,
    create_round25_tcn_corpus_source,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class Round25TCNStoreSourceBuild:
    source: Round25TCNCorpusSource
    feature_store_manifest_sha256: str
    condition_batch_count: int
    source_scan_count: int = 1
    target_accessed_beyond_dataset: bool = False
    trading_authority: bool = False

    def validated(self) -> Round25TCNStoreSourceBuild:
        self.source.validated()
        if (
            _SHA256.fullmatch(self.feature_store_manifest_sha256) is None
            or self.condition_batch_count != len(self.source.condition_ids)
            or self.condition_batch_count <= 0
            or self.source_scan_count != 1
            or self.target_accessed_beyond_dataset
            or self.trading_authority
        ):
            raise ValueError("Round 25 TCN store source build differs")
        return self


def _samples_by_condition(
    dataset: Round25DevelopmentDataset,
) -> dict[str, tuple[Round25DevelopmentSample, ...]]:
    grouped: defaultdict[str, list[Round25DevelopmentSample]] = defaultdict(list)
    for sample in dataset.samples:
        grouped[sample.condition_id].append(sample)
    output = {
        condition_id: tuple(
            sorted(samples, key=lambda item: item.decision_time_ms)
        )
        for condition_id, samples in grouped.items()
    }
    if (
        len(output) != dataset.condition_count
        or any(
            len(samples) != POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            or any(sample.role != dataset.role for sample in samples)
            or len({sample.decision_time_ms for sample in samples})
            != POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            for samples in output.values()
        )
    ):
        raise ValueError("Round 25 TCN dataset condition population differs")
    return output


def create_round25_store_tcn_source(
    *,
    feature_database: str | Path,
    feature_store_manifest_sha256: str,
    dataset: Round25DevelopmentDataset,
    logistic: Round25LogisticResidualArtifact,
    progress: ProgressCallback | None = None,
) -> Round25TCNStoreSourceBuild:
    """Hash each condition once, then rebuild only requested 16-condition batches."""

    if not isinstance(dataset, Round25DevelopmentDataset):
        raise TypeError("Round 25 TCN store dataset type differs")
    require_round25_dataset_minimum(dataset)
    if dataset.role not in {"train", "calibration"}:
        raise ValueError("Round 25 TCN store role differs")
    if not isinstance(logistic, Round25LogisticResidualArtifact):
        raise TypeError("Round 25 TCN store logistic artifact type differs")
    manifest = load_round25_joint_store_manifest(feature_database)
    if (
        _SHA256.fullmatch(feature_store_manifest_sha256) is None
        or manifest["manifest_sha256"] != feature_store_manifest_sha256
        or logistic.train_resolution_authority_sha256
        != dataset.resolution_authority_sha256
    ):
        raise ValueError("Round 25 TCN store source identity differs")
    transform_sha256 = round25_feature_transform_sha256(
        logistic.center,
        logistic.scale,
    )
    grouped = _samples_by_condition(dataset)
    identities = tuple(sorted(
        (
            (samples[0].event_start_ms, condition_id)
            for condition_id, samples in grouped.items()
        ),
        key=lambda item: (item[0], item[1]),
    ))

    def load_batches(
        condition_ids: tuple[str, ...],
    ) -> tuple[Round25SequenceConditionBatch, ...]:
        materializations = load_round25_joint_condition_batch(
            feature_database,
            condition_ids,
            expected_manifest_sha256=feature_store_manifest_sha256,
        )
        output: list[Round25SequenceConditionBatch] = []
        for materialization in materializations:
            samples = grouped.get(materialization.condition_id)
            if (
                samples is None
                or materialization.role != dataset.role
                or materialization.event_start_ms != samples[0].event_start_ms
            ):
                raise ValueError("Round 25 TCN store batch population differs")
            output.append(build_round25_sequence_condition_batch(
                snapshots=materialization.persisted_snapshots,
                endpoint_samples=samples,
                center=logistic.center,
                scale=logistic.scale,
                source_dataset_sha256=dataset.dataset_sha256,
                resolution_authority_sha256=(
                    dataset.resolution_authority_sha256
                ),
            ))
        return tuple(output)

    batch_hashes: list[str] = []
    for offset in range(0, len(identities), 16):
        selected = identities[offset : offset + 16]
        condition_ids = tuple(item[1] for item in selected)
        batches = load_batches(condition_ids)
        if tuple(batch.condition_id for batch in batches) != condition_ids:
            raise ValueError("Round 25 TCN store manifest order differs")
        batch_hashes.extend(batch.batch_sha256 for batch in batches)
        if progress is not None:
            progress(
                "tcn_source_hashing",
                {
                    "conditions_processed": min(offset + 16, len(identities)),
                    "role": dataset.role,
                    "total_conditions": len(identities),
                },
            )
    source = create_round25_tcn_corpus_source(
        role=dataset.role,
        condition_ids=tuple(item[1] for item in identities),
        event_start_ms=tuple(item[0] for item in identities),
        batch_sha256=tuple(batch_hashes),
        source_dataset_sha256=dataset.dataset_sha256,
        resolution_authority_sha256=dataset.resolution_authority_sha256,
        feature_transform_sha256=transform_sha256,
        loader=load_batches,
    )
    return Round25TCNStoreSourceBuild(
        source=source,
        feature_store_manifest_sha256=feature_store_manifest_sha256,
        condition_batch_count=len(identities),
    ).validated()


def create_round25_store_tcn_fit_sources(
    *,
    feature_database: str | Path,
    feature_store_manifest_sha256: str,
    train: Round25DevelopmentDataset,
    calibration: Round25DevelopmentDataset,
    logistic: Round25LogisticResidualArtifact,
    progress: ProgressCallback | None = None,
) -> tuple[Round25TCNCorpusSource, Round25TCNCorpusSource]:
    train_build = create_round25_store_tcn_source(
        feature_database=feature_database,
        feature_store_manifest_sha256=feature_store_manifest_sha256,
        dataset=train,
        logistic=logistic,
        progress=progress,
    )
    calibration_build = create_round25_store_tcn_source(
        feature_database=feature_database,
        feature_store_manifest_sha256=feature_store_manifest_sha256,
        dataset=calibration,
        logistic=logistic,
        progress=progress,
    )
    return train_build.source, calibration_build.source


__all__ = [
    "Round25TCNStoreSourceBuild",
    "create_round25_store_tcn_fit_sources",
    "create_round25_store_tcn_source",
]
