from __future__ import annotations

from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round25_tcn_store_source as store_source
from simple_ai_trading.polymarket_round25_tcn_store_source import (
    create_round25_store_tcn_source,
)


MANIFEST_SHA256 = "a" * 64
AUTHORITY_SHA256 = "b" * 64
DATASET_SHA256 = "c" * 64
TRANSFORM_SHA256 = "d" * 64


class _Dataset:
    def __init__(self, *, role: str = "train") -> None:
        self.role = role
        self.dataset_sha256 = DATASET_SHA256
        self.resolution_authority_sha256 = AUTHORITY_SHA256
        self.condition_ids = (
            "0x" + "1" * 64,
            "0x" + "2" * 64,
        )
        self.event_starts = (1_786_406_700_000, 1_786_407_000_000)
        self.samples = tuple(
            SimpleNamespace(
                condition_id=condition_id,
                decision_time_ms=event_start + index * 1_000,
                event_start_ms=event_start,
                role=role,
            )
            for condition_id, event_start in zip(
                self.condition_ids,
                self.event_starts,
                strict=True,
            )
            for index in range(16)
        )
        self.condition_count = 2


class _Logistic:
    center = (0.0,)
    scale = (1.0,)
    train_resolution_authority_sha256 = AUTHORITY_SHA256


def test_store_tcn_source_hashes_once_then_loads_only_requested_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset()
    calls: list[tuple[str, ...]] = []
    batch_by_id: dict[str, object] = {}

    monkeypatch.setattr(store_source, "Round25DevelopmentDataset", _Dataset)
    monkeypatch.setattr(store_source, "Round25LogisticResidualArtifact", _Logistic)
    monkeypatch.setattr(
        store_source,
        "require_round25_dataset_minimum",
        lambda value: value,
    )
    monkeypatch.setattr(
        store_source,
        "load_round25_joint_store_manifest",
        lambda _database: {"manifest_sha256": MANIFEST_SHA256},
    )
    monkeypatch.setattr(
        store_source,
        "round25_feature_transform_sha256",
        lambda *_args: TRANSFORM_SHA256,
    )

    def load_batch(
        _database: object,
        condition_ids: tuple[str, ...],
        *,
        expected_manifest_sha256: str,
    ) -> tuple[object, ...]:
        assert expected_manifest_sha256 == MANIFEST_SHA256
        calls.append(condition_ids)
        return tuple(
            SimpleNamespace(
                condition_id=condition_id,
                event_start_ms=dataset.event_starts[
                    dataset.condition_ids.index(condition_id)
                ],
                role="train",
                persisted_snapshots=(object(),),
            )
            for condition_id in condition_ids
        )

    def build_batch(**kwargs: object) -> object:
        samples = kwargs["endpoint_samples"]
        condition_id = samples[0].condition_id
        batch = SimpleNamespace(
            condition_id=condition_id,
            event_start_ms=samples[0].event_start_ms,
            role="train",
            batch_sha256=("e" if condition_id.endswith("1") else "f") * 64,
        )
        batch_by_id[condition_id] = batch
        return batch

    monkeypatch.setattr(
        store_source,
        "load_round25_joint_condition_batch",
        load_batch,
    )
    monkeypatch.setattr(
        store_source,
        "build_round25_sequence_condition_batch",
        build_batch,
    )

    built = create_round25_store_tcn_source(
        feature_database="feature.duckdb",
        feature_store_manifest_sha256=MANIFEST_SHA256,
        dataset=dataset,
        logistic=_Logistic(),
    )

    assert built.condition_batch_count == 2
    assert built.source.condition_ids == dataset.condition_ids
    assert built.source.batch_sha256 == ("e" * 64, "f" * 64)
    assert calls == [dataset.condition_ids]
    loaded = tuple(built.source.loader((dataset.condition_ids[1],)))
    assert loaded == (batch_by_id[dataset.condition_ids[1]],)
    assert calls == [dataset.condition_ids, (dataset.condition_ids[1],)]


def test_store_tcn_source_rejects_manifest_and_role_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_source, "Round25DevelopmentDataset", _Dataset)
    monkeypatch.setattr(store_source, "Round25LogisticResidualArtifact", _Logistic)
    monkeypatch.setattr(
        store_source,
        "require_round25_dataset_minimum",
        lambda value: value,
    )
    monkeypatch.setattr(
        store_source,
        "load_round25_joint_store_manifest",
        lambda _database: {"manifest_sha256": "0" * 64},
    )

    with pytest.raises(ValueError, match="source identity differs"):
        create_round25_store_tcn_source(
            feature_database="feature.duckdb",
            feature_store_manifest_sha256=MANIFEST_SHA256,
            dataset=_Dataset(),
            logistic=_Logistic(),
        )
    with pytest.raises(ValueError, match="role differs"):
        create_round25_store_tcn_source(
            feature_database="feature.duckdb",
            feature_store_manifest_sha256="0" * 64,
            dataset=_Dataset(role="selection"),
            logistic=_Logistic(),
        )
