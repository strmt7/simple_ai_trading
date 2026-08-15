from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_CORE_FEATURE_NAMES,
    Round21CoreFeatureSnapshot,
)
from simple_ai_trading.polymarket_round21_corpus import (
    POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
    Round21CoreConditionMaterialization,
)
from simple_ai_trading import polymarket_round21_corpus_store as store_module
from simple_ai_trading.polymarket_round21_corpus_store import (
    audit_round21_core_partition,
    audit_round21_sealed_core_partition,
    load_round21_core_development_publication,
    load_round21_core_partition_snapshots,
    load_round21_core_publication_manifest,
    load_round21_sealed_core_partition_snapshots,
    publish_round21_core_corpus,
    validate_round21_core_publication_boundary,
)
from simple_ai_trading.polymarket_round21_dataset import Round21PartitionPolicy
from simple_ai_trading.polymarket_round21_one_use import (
    Round21OneUseClaim,
    Round21OneUseStore,
)


CAMPAIGN_START_MS = 1_800_001_200_000
TEST_START_MS = CAMPAIGN_START_MS + 1_989_000_000
RUN_ID = "3" * 32


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _policy() -> Round21PartitionPolicy:
    return Round21PartitionPolicy.create(
        campaign_start_ms=CAMPAIGN_START_MS,
        campaign_end_ms=CAMPAIGN_START_MS + 30 * 86_400_000,
    )


def _materialization(
    *,
    role: str,
    event_start_ms: int,
    digit: str,
) -> Round21CoreConditionMaterialization:
    condition_id = "0x" + digit * 64
    rows = tuple(
        Round21CoreFeatureSnapshot(
            condition_id=condition_id,
            event_start_ms=event_start_ms,
            decision_time_ms=event_start_ms + 30_000 + index * 250,
            available=True,
            reasons=(),
            structural_probability=0.5000000000000001 + index * 0.01,
            market_prior_probability=0.49999999999999994 - index * 0.01,
            values=tuple(
                (feature + 1) * 1.0e-12 * (index + 1)
                for feature in range(len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES))
            ),
            source_chain_sha256=chr(ord("a") + index) * 64,
            maximum_receipt_ms=event_start_ms + 29_999 + index * 250,
        )
        for index in range(2)
    )
    body: dict[str, object] = {
        "schema_version": "polymarket-round21-condition-admission-v1",
        "core_corpus_design_sha256": POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
        "run_id": RUN_ID,
        "segment_index": 1,
        "snapshot_sha256": "d" * 64,
        "condition_id": condition_id,
        "event_start_ms": event_start_ms,
        "event_end_ms": event_start_ms + 300_000,
        "role": role,
        "union_event_count": 500,
        "union_event_chain_sha256": "e" * 64,
        "lane_event_counts": {"clob-a": 500, "clob-b": 500},
        "lane_coverage_fraction": {"clob-a": 1.0, "clob-b": 1.0},
        "shared_event_count": 500,
        "shared_fraction": 1.0,
        "lane_gap_counts": {"clob-a": 0, "clob-b": 0},
        "joint_unhealthy_ms": 0,
        "up_full_book_observed": True,
        "down_full_book_observed": True,
        "exact_chainlink_open_receipt_count": 1,
        "exact_chainlink_close_receipt_count": 1,
        "chainlink_connection_count": 1,
        "admitted": True,
        "rejection_reasons": [],
        "available_feature_row_count": len(rows),
        "unavailable_feature_row_count": 1,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "optional_binance_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    admission = {**body, "admission_sha256": _sha(body)}
    return Round21CoreConditionMaterialization(
        admission=admission,
        available_features=rows,
        unavailable_feature_row_count=1,
        unavailable_reason_counts={"chainlink_coverage_below_minimum": 1},
    ).validated()


def _claim(sealed_population_sha256: str) -> Round21OneUseClaim:
    provisional = Round21OneUseClaim(
        pretest_manifest_sha256="4" * 64,
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=sealed_population_sha256,
        repository_commit_oid="5" * 40,
        nominated_ai_model=None,
        nominated_ai_model_digest=None,
        opened_at_ms=1_900_000_000_000,
        claim_sha256="6" * 64,
    )
    return replace(
        provisional,
        claim_sha256=_sha(provisional.identity_payload()),
    ).validated()


def test_partition_store_round_trips_exact_binary64_and_detects_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "development.duckdb"
    writer = store_module._PartitionWriter(
        path,
        partition="development",
        partition_policy=_policy(),
    )
    writer.add(
        _materialization(
            role="train",
            event_start_ms=CAMPAIGN_START_MS,
            digit="1",
        )
    )
    manifest = writer.finalize(terminal_receipt_audit_sha256="f" * 64)

    audit = audit_round21_core_partition(path, manifest)

    assert audit["feature_chunk_count"] == 1
    assert audit["feature_row_count"] == 2
    loaded = load_round21_core_partition_snapshots(path, manifest)
    original = _materialization(
        role="train",
        event_start_ms=CAMPAIGN_START_MS,
        digit="1",
    ).available_features
    assert loaded == original
    with duckdb.connect(str(path)) as connection:
        chunk_json, compressed_bytes = connection.execute(
            "SELECT chunk_manifest_json, compressed_payload FROM feature_chunk"
        ).fetchone()
        chunk_metadata = json.loads(chunk_json)
        chunk_metadata.pop("chunk_sha256")
        decoded = store_module._decode_feature_chunk(
            bytes(compressed_bytes),
            chunk_metadata,
        )
        assert [value.hex() for value in decoded[0].values] == [
            value.hex() for value in original[0].values
        ]
        payload = bytearray(
            connection.execute(
                "SELECT compressed_payload FROM feature_chunk"
            ).fetchone()[0]
        )
        payload[-1] ^= 1
        connection.execute(
            "UPDATE feature_chunk SET compressed_payload = ?",
            [bytes(payload)],
        )
    with pytest.raises(ValueError, match="chunk metadata differs"):
        audit_round21_core_partition(path, manifest)


def test_sealed_partition_requires_explicit_access(tmp_path: Path) -> None:
    path = tmp_path / "sealed.duckdb"
    writer = store_module._PartitionWriter(
        path,
        partition="sealed_test",
        partition_policy=_policy(),
    )
    writer.add(_materialization(role="test", event_start_ms=TEST_START_MS, digit="2"))
    manifest = writer.finalize(terminal_receipt_audit_sha256="f" * 64)

    with pytest.raises(PermissionError, match="one-use access"):
        audit_round21_core_partition(path, manifest)

    claim = _claim(str(manifest["manifest_sha256"]))
    ledger = tmp_path / "one-use.sqlite3"
    with Round21OneUseStore(ledger) as store:
        store.open_claim(claim)
        access = store.consume_test_access(
            claim,
            observed_at_ms=claim.opened_at_ms + 1,
        )
    audit = audit_round21_sealed_core_partition(
        path,
        manifest,
        one_use_store_path=ledger,
        claim=claim,
        test_access_sha256=access,
    )
    assert audit["claim_sha256"] == claim.claim_sha256
    assert (
        load_round21_sealed_core_partition_snapshots(
            path,
            manifest,
            one_use_store_path=ledger,
            claim=claim,
            test_access_sha256=access,
        )
        == _materialization(
            role="test",
            event_start_ms=TEST_START_MS,
            digit="2",
        ).available_features
    )
    with pytest.raises(PermissionError, match="access is unavailable"):
        audit_round21_sealed_core_partition(
            path,
            manifest,
            one_use_store_path=ledger,
            claim=claim,
            test_access_sha256="7" * 64,
        )
    with Round21OneUseStore(ledger) as store:
        store.fail(
            claim,
            reason="fixture terminal failure",
            observed_at_ms=claim.opened_at_ms + 2,
        )
    with pytest.raises(PermissionError, match="access is unavailable"):
        audit_round21_sealed_core_partition(
            path,
            manifest,
            one_use_store_path=ledger,
            claim=claim,
            test_access_sha256=access,
        )


def test_atomic_publication_writes_both_partitions_or_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _materialization(
        role="train",
        event_start_ms=CAMPAIGN_START_MS,
        digit="1",
    )
    sealed = _materialization(role="test", event_start_ms=TEST_START_MS, digit="2")
    transport = {
        "campaign_start_ms": CAMPAIGN_START_MS,
        "campaign_end_ms": CAMPAIGN_START_MS + 30 * 86_400_000,
        "manifest_sha256": "9" * 64,
    }

    class FakeObserver:
        def __init__(self, *, conditions, partition_policy, sink) -> None:
            del partition_policy
            self.conditions = conditions
            self.sink = sink
            self.materialized_condition_count = 0

    def fake_audit(*, observer, observed_at_ms, **_kwargs):
        observer.sink(development)
        observer.sink(sealed)
        observer.materialized_condition_count = 2
        body = {
            "schema_version": "polymarket-round21-terminal-receipt-audit-v1",
            "created_at_ms": observed_at_ms,
            "terminal_transport_manifest_sha256": transport["manifest_sha256"],
            "receipt_replay_complete": True,
            "condition_admission_pending": True,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        return {**body, "audit_sha256": _sha(body)}

    monkeypatch.setattr(
        store_module, "load_round21_core_corpus_design", lambda _root: {}
    )
    monkeypatch.setattr(
        store_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        store_module,
        "load_round21_core_conditions",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(store_module, "Round21CoreCorpusObserver", FakeObserver)
    monkeypatch.setattr(store_module, "audit_round21_terminal_receipts", fake_audit)
    monkeypatch.setattr(
        store_module,
        "validate_round21_terminal_receipt_audit",
        lambda value, **_kwargs: value,
    )
    destination = tmp_path / "publication"

    manifest = publish_round21_core_corpus(
        repository=tmp_path,
        source_database=tmp_path / "source.duckdb",
        terminal_transport_manifest=transport,
        publication_directory=destination,
        observed_at_ms=1_900_000_000_000,
    )

    assert destination.is_dir()
    assert (destination / "development.duckdb").is_file()
    assert (destination / "sealed-test.duckdb").is_file()
    assert load_round21_core_publication_manifest(destination) == manifest
    assert validate_round21_core_publication_boundary(destination) == manifest
    loaded_manifest, snapshots = load_round21_core_development_publication(destination)
    assert loaded_manifest == manifest
    assert snapshots == development.available_features
    assert manifest["optional_binance_consulted"] is False

    failed_destination = tmp_path / "failed-publication"

    def failing_audit(*, observer, **_kwargs):
        observer.sink(development)
        raise RuntimeError("receipt audit failed")

    monkeypatch.setattr(
        store_module,
        "audit_round21_terminal_receipts",
        failing_audit,
    )
    with pytest.raises(RuntimeError, match="receipt audit failed"):
        publish_round21_core_corpus(
            repository=tmp_path,
            source_database=tmp_path / "source.duckdb",
            terminal_transport_manifest=transport,
            publication_directory=failed_destination,
            observed_at_ms=1_900_000_000_000,
        )
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed-publication.*.staging"))
