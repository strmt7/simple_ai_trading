"""Resumable bounded materialization for Round 9 Polymarket action values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib
import json

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_action_value import (
    POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
    PolymarketActionValueConfig,
    build_polymarket_action_value_dataset,
    materialize_polymarket_action_value_dataset,
)
from .polymarket_features import (
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
    load_polymarket_feature_source_context,
    materialize_polymarket_feature_dataset,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_repricing import PolymarketRepricingExecutionContext


POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION = (
    "polymarket-action-value-bounded-pipeline-v1"
)
POLYMARKET_ACTION_BATCH_SCHEMA_VERSION = "polymarket-action-value-batch-v1"
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


@dataclass(frozen=True)
class PolymarketActionPipelineConfig:
    """Bounded compute policy; model and execution semantics remain frozen elsewhere."""

    market_groups_per_batch: int = 1
    feature: PolymarketFeatureConfig = field(
        default_factory=lambda: PolymarketFeatureConfig(
            cadence_ms=250,
            warmup_ms=5_000,
            minimum_resolved_markets_per_asset=30,
            allow_segmented_gaps=False,
        )
    )
    action: PolymarketActionValueConfig = field(
        default_factory=PolymarketActionValueConfig
    )

    def validated(self) -> PolymarketActionPipelineConfig:
        if int(self.market_groups_per_batch) != 1:
            raise ValueError(
                "market_groups_per_batch must be 1 under the bounded-memory policy"
            )
        feature = self.feature.validated()
        action = self.action.validated()
        if int(feature.cadence_ms) != action.decision_cadence_ms:
            raise ValueError("Polymarket feature/action cadence differs")
        return self

    def asdict(self) -> dict[str, object]:
        self.validated()
        return {
            "market_groups_per_batch": int(self.market_groups_per_batch),
            "feature": self.feature.asdict(),
            "action": self.action.asdict(),
        }


@dataclass(frozen=True)
class PolymarketActionBatchResult:
    batch_id: str
    status: str
    group_starts_ms: tuple[int, ...]
    condition_ids: tuple[str, ...]
    feature_dataset_sha256: str
    action_dataset_sha256: str
    feature_row_count: int
    action_count: int
    classifier_eligible_count: int
    positive_complete_count: int
    category_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    batch_sha256: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("status")
        payload["group_starts_ms"] = list(self.group_starts_ms)
        payload["condition_ids"] = list(self.condition_ids)
        payload["category_counts"] = dict(sorted(self.category_counts.items()))
        payload["terminal_reason_counts"] = dict(
            sorted(self.terminal_reason_counts.items())
        )
        return payload

    def validated(self) -> PolymarketActionBatchResult:
        if (
            not _is_sha256(self.batch_id)
            or not _is_sha256(self.feature_dataset_sha256)
            or not _is_sha256(self.action_dataset_sha256)
            or not _is_sha256(self.batch_sha256)
            or self.status not in {"created", "existing"}
            or not self.group_starts_ms
            or len(self.condition_ids) != len(self.group_starts_ms) * len(_ASSETS)
            or self.feature_row_count < 0
            or self.action_count != self.feature_row_count * 2
            or not 0 <= self.positive_complete_count <= self.classifier_eligible_count
            or not 0 <= self.classifier_eligible_count <= self.action_count
            or sum(self.category_counts.values()) != self.action_count
            or sum(self.terminal_reason_counts.values()) != self.action_count
        ):
            raise ValueError("Polymarket action batch result is invalid")
        return self


@dataclass(frozen=True)
class PolymarketActionPipelineReport:
    run_id: str
    run_report_sha256: str
    config: PolymarketActionPipelineConfig
    eligibility_sha256: str
    batches: tuple[PolymarketActionBatchResult, ...]
    action_count: int
    classifier_eligible_count: int
    positive_complete_count: int
    category_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "run_id": self.run_id,
            "run_report_sha256": self.run_report_sha256,
            "config": self.config.asdict(),
            "eligibility_sha256": self.eligibility_sha256,
            "batches": [batch.asdict() for batch in self.batches],
            "action_count": self.action_count,
            "classifier_eligible_count": self.classifier_eligible_count,
            "positive_complete_count": self.positive_complete_count,
            "category_counts": dict(sorted(self.category_counts.items())),
            "terminal_reason_counts": dict(sorted(self.terminal_reason_counts.items())),
            "training_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> PolymarketActionPipelineReport:
        expected_categories: dict[str, int] = {}
        expected_reasons: dict[str, int] = {}
        for batch in self.batches:
            batch.validated()
            _add_counts(expected_categories, batch.category_counts)
            _add_counts(expected_reasons, batch.terminal_reason_counts)
        if (
            not self.run_id
            or not _is_sha256(self.run_report_sha256)
            or (self.eligibility_sha256 and not _is_sha256(self.eligibility_sha256))
            or not self.batches
            or self.action_count != sum(item.action_count for item in self.batches)
            or self.classifier_eligible_count
            != sum(item.classifier_eligible_count for item in self.batches)
            or self.positive_complete_count
            != sum(item.positive_complete_count for item in self.batches)
            or self.category_counts != expected_categories
            or self.terminal_reason_counts != expected_reasons
            or len({item.batch_id for item in self.batches}) != len(self.batches)
            or self.report_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket action pipeline report is invalid")
        return self


def _ensure_pipeline_tables(store: PolymarketEvidenceStore) -> None:
    store.connect().execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_action_value_batch (
            batch_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            run_report_sha256 VARCHAR NOT NULL,
            config_json VARCHAR NOT NULL,
            eligibility_sha256 VARCHAR NOT NULL,
            group_starts_json VARCHAR NOT NULL,
            condition_ids_json VARCHAR NOT NULL,
            feature_dataset_sha256 VARCHAR NOT NULL,
            action_dataset_sha256 VARCHAR NOT NULL,
            feature_row_count UBIGINT NOT NULL,
            action_count UBIGINT NOT NULL,
            classifier_eligible_count UBIGINT NOT NULL,
            positive_complete_count UBIGINT NOT NULL,
            category_counts_json VARCHAR NOT NULL,
            terminal_reason_counts_json VARCHAR NOT NULL,
            batch_sha256 VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS polymarket_action_value_pipeline (
            report_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            run_report_sha256 VARCHAR NOT NULL,
            config_json VARCHAR NOT NULL,
            eligibility_sha256 VARCHAR NOT NULL,
            batch_ids_json VARCHAR NOT NULL,
            action_dataset_sha256_json VARCHAR NOT NULL,
            action_count UBIGINT NOT NULL,
            classifier_eligible_count UBIGINT NOT NULL,
            positive_complete_count UBIGINT NOT NULL,
            category_counts_json VARCHAR NOT NULL,
            terminal_reason_counts_json VARCHAR NOT NULL,
            report_json VARCHAR NOT NULL
        );
        """
    )


def _batch_identity(
    *,
    run_id: str,
    run_report_sha256: str,
    config: PolymarketActionPipelineConfig,
    eligibility_sha256: str,
    group_starts_ms: Sequence[int],
    condition_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_ACTION_BATCH_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        "run_id": run_id,
        "run_report_sha256": run_report_sha256,
        "config": config.asdict(),
        "eligibility_sha256": eligibility_sha256,
        "group_starts_ms": [int(value) for value in group_starts_ms],
        "condition_ids": list(condition_ids),
    }


def _load_existing_batch(
    store: PolymarketEvidenceStore,
    *,
    batch_id: str,
    identity: Mapping[str, object],
) -> PolymarketActionBatchResult | None:
    row = (
        store.connect()
        .execute(
            """
        SELECT batch_id, schema_version, contract_sha256, run_id,
               run_report_sha256, config_json, eligibility_sha256,
               group_starts_json, condition_ids_json,
               feature_dataset_sha256, action_dataset_sha256,
               feature_row_count, action_count, classifier_eligible_count,
               positive_complete_count, category_counts_json,
               terminal_reason_counts_json, batch_sha256
        FROM polymarket_action_value_batch WHERE batch_id = ?
        """,
            [batch_id],
        )
        .fetchone()
    )
    if row is None:
        return None
    expected_prefix = (
        batch_id,
        POLYMARKET_ACTION_BATCH_SCHEMA_VERSION,
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        str(identity["run_id"]),
        str(identity["run_report_sha256"]),
        _canonical_json(identity["config"]),
        str(identity["eligibility_sha256"]),
        _canonical_json(identity["group_starts_ms"]),
        _canonical_json(identity["condition_ids"]),
    )
    if tuple(row[:9]) != expected_prefix:
        raise ValueError("stored Polymarket action batch identity is inconsistent")
    feature_manifest = (
        store.connect()
        .execute(
            """
        SELECT d.row_count, count(r.feature_id)
        FROM polymarket_feature_dataset AS d
        LEFT JOIN polymarket_feature_row AS r
          ON r.dataset_id = d.dataset_id
        WHERE d.dataset_sha256 = ? GROUP BY d.row_count
        """,
            [row[9]],
        )
        .fetchone()
    )
    action_manifest = (
        store.connect()
        .execute(
            """
        SELECT d.action_count, d.classifier_eligible_count,
               d.positive_complete_count, d.category_counts_json,
               d.terminal_reason_counts_json, count(r.action_index),
               coalesce(sum(CASE WHEN r.classifier_eligible THEN 1 ELSE 0 END), 0),
               coalesce(sum(CASE WHEN r.positive_complete THEN 1 ELSE 0 END), 0)
        FROM polymarket_action_value_dataset AS d
        LEFT JOIN polymarket_action_value_row AS r
          ON r.dataset_sha256 = d.dataset_sha256
        WHERE d.dataset_sha256 = ?
        GROUP BY d.action_count, d.classifier_eligible_count,
                 d.positive_complete_count, d.category_counts_json,
                 d.terminal_reason_counts_json
        """,
            [row[10]],
        )
        .fetchone()
    )
    if (
        feature_manifest != (row[11], row[11])
        or action_manifest is None
        or tuple(action_manifest[:3]) != tuple(row[12:15])
        or tuple(action_manifest[5:]) != tuple(row[12:15])
        or str(action_manifest[3]) != str(row[15])
        or str(action_manifest[4]) != str(row[16])
    ):
        raise ValueError("stored Polymarket action batch references invalid manifests")
    batch_payload = {
        **dict(identity),
        "feature_dataset_sha256": str(row[9]),
        "action_dataset_sha256": str(row[10]),
        "feature_row_count": int(row[11]),
        "action_count": int(row[12]),
        "classifier_eligible_count": int(row[13]),
        "positive_complete_count": int(row[14]),
        "category_counts": json.loads(str(row[15])),
        "terminal_reason_counts": json.loads(str(row[16])),
    }
    if _sha256(batch_payload) != str(row[17]):
        raise ValueError("stored Polymarket action batch digest is invalid")
    return PolymarketActionBatchResult(
        batch_id=batch_id,
        status="existing",
        group_starts_ms=tuple(int(value) for value in identity["group_starts_ms"]),
        condition_ids=tuple(str(value) for value in identity["condition_ids"]),
        feature_dataset_sha256=str(row[9]),
        action_dataset_sha256=str(row[10]),
        feature_row_count=int(row[11]),
        action_count=int(row[12]),
        classifier_eligible_count=int(row[13]),
        positive_complete_count=int(row[14]),
        category_counts={
            str(key): int(value)
            for key, value in batch_payload["category_counts"].items()
        },
        terminal_reason_counts={
            str(key): int(value)
            for key, value in batch_payload["terminal_reason_counts"].items()
        },
        batch_sha256=str(row[17]),
    ).validated()


def _persist_batch(
    store: PolymarketEvidenceStore,
    *,
    identity: Mapping[str, object],
    feature_dataset_sha256: str,
    action_dataset_sha256: str,
    feature_row_count: int,
    action_count: int,
    classifier_eligible_count: int,
    positive_complete_count: int,
    category_counts: Mapping[str, int],
    terminal_reason_counts: Mapping[str, int],
) -> PolymarketActionBatchResult:
    batch_id = _sha256(identity)
    payload = {
        **dict(identity),
        "feature_dataset_sha256": feature_dataset_sha256,
        "action_dataset_sha256": action_dataset_sha256,
        "feature_row_count": int(feature_row_count),
        "action_count": int(action_count),
        "classifier_eligible_count": int(classifier_eligible_count),
        "positive_complete_count": int(positive_complete_count),
        "category_counts": dict(sorted(category_counts.items())),
        "terminal_reason_counts": dict(sorted(terminal_reason_counts.items())),
    }
    batch_sha256 = _sha256(payload)
    values = (
        batch_id,
        POLYMARKET_ACTION_BATCH_SCHEMA_VERSION,
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        str(identity["run_id"]),
        str(identity["run_report_sha256"]),
        _canonical_json(identity["config"]),
        str(identity["eligibility_sha256"]),
        _canonical_json(identity["group_starts_ms"]),
        _canonical_json(identity["condition_ids"]),
        feature_dataset_sha256,
        action_dataset_sha256,
        int(feature_row_count),
        int(action_count),
        int(classifier_eligible_count),
        int(positive_complete_count),
        _canonical_json(dict(sorted(category_counts.items()))),
        _canonical_json(dict(sorted(terminal_reason_counts.items()))),
        batch_sha256,
    )
    store.connect().execute(
        """
        INSERT INTO polymarket_action_value_batch VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        values,
    )
    return PolymarketActionBatchResult(
        batch_id=batch_id,
        status="created",
        group_starts_ms=tuple(int(value) for value in identity["group_starts_ms"]),
        condition_ids=tuple(str(value) for value in identity["condition_ids"]),
        feature_dataset_sha256=feature_dataset_sha256,
        action_dataset_sha256=action_dataset_sha256,
        feature_row_count=int(feature_row_count),
        action_count=int(action_count),
        classifier_eligible_count=int(classifier_eligible_count),
        positive_complete_count=int(positive_complete_count),
        category_counts=dict(sorted(category_counts.items())),
        terminal_reason_counts=dict(sorted(terminal_reason_counts.items())),
        batch_sha256=batch_sha256,
    ).validated()


def _add_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def materialize_polymarket_action_value_batches(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    config: PolymarketActionPipelineConfig | None = None,
    eligible_condition_ids: Sequence[str] | None = None,
    eligibility_sha256: str = "",
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> PolymarketActionPipelineReport:
    """Build resumable action values without materializing a full-run CLOB replay."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Polymarket action pipeline requires an explicit run ID")
    cfg = (config or PolymarketActionPipelineConfig()).validated()
    eligibility_digest = str(eligibility_sha256 or "").strip()
    if eligibility_digest and not _is_sha256(eligibility_digest):
        raise ValueError("Polymarket eligibility digest is invalid")
    if cfg.feature.allow_segmented_gaps and (
        not eligibility_digest or eligible_condition_ids is None
    ):
        raise ValueError(
            "segmented action replay requires hash-bound eligible condition IDs"
        )
    run_row = (
        store.connect()
        .execute(
            """
        SELECT status, error, report_sha256
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
            [selected],
        )
        .fetchone()
    )
    if run_row is None:
        raise ValueError("unknown Polymarket action pipeline run")
    allowed_statuses = (
        {"complete", "degraded"} if cfg.feature.allow_segmented_gaps else {"complete"}
    )
    if str(run_row[0]) not in allowed_statuses or str(run_row[1] or "").strip():
        raise ValueError("Polymarket action pipeline run is not admissible")
    run_report_sha256 = str(run_row[2])
    if not _is_sha256(run_report_sha256):
        raise ValueError("Polymarket action pipeline run report is invalid")
    integrity = store.integrity_errors(selected, progress=progress)
    if integrity:
        raise ValueError(
            "Polymarket action pipeline integrity failed: " + "; ".join(integrity)
        )
    _ensure_pipeline_tables(store)
    markets = PolymarketEvidenceReplay.load_markets(store, run_id=selected)
    selected_conditions = (
        {market.condition_id for market in markets}
        if eligible_condition_ids is None
        else {str(value or "").strip().lower() for value in eligible_condition_ids}
    )
    if not selected_conditions or "" in selected_conditions:
        raise ValueError("Polymarket action pipeline condition selection is empty")
    market_by_condition = {market.condition_id: market for market in markets}
    if not selected_conditions.issubset(market_by_condition):
        raise ValueError("Polymarket action pipeline selected an unknown condition")
    groups: dict[int, list[PolymarketFiveMinuteMarket]] = {}
    for condition_id in selected_conditions:
        market = market_by_condition[condition_id]
        groups.setdefault(int(market.event_start_ms), []).append(market)
    ordered_groups: list[tuple[int, tuple[PolymarketFiveMinuteMarket, ...]]] = []
    for start_ms, values in sorted(groups.items()):
        ordered = tuple(sorted(values, key=lambda item: _ASSETS.index(item.asset)))
        if tuple(item.asset for item in ordered) != _ASSETS:
            raise ValueError(
                "Polymarket action pipeline requires synchronized BTC/ETH/SOL groups"
            )
        ordered_groups.append((start_ms, ordered))
    batch_specs: list[tuple[dict[str, object], str]] = []
    width = int(cfg.market_groups_per_batch)
    for offset in range(0, len(ordered_groups), width):
        chunk = ordered_groups[offset : offset + width]
        starts = tuple(item[0] for item in chunk)
        conditions = tuple(
            market.condition_id for _start, group in chunk for market in group
        )
        identity = _batch_identity(
            run_id=selected,
            run_report_sha256=run_report_sha256,
            config=cfg,
            eligibility_sha256=eligibility_digest,
            group_starts_ms=starts,
            condition_ids=conditions,
        )
        batch_specs.append((identity, _sha256(identity)))
    if not batch_specs:
        raise ValueError("Polymarket action pipeline has no synchronized batches")
    results: list[PolymarketActionBatchResult | None] = []
    missing_indexes: list[int] = []
    for index, (identity, batch_id) in enumerate(batch_specs):
        existing = _load_existing_batch(
            store,
            batch_id=batch_id,
            identity=identity,
        )
        results.append(existing)
        if existing is None:
            missing_indexes.append(index)
    source_context = None
    if missing_indexes:
        store.ensure_condition_message_cache(selected, progress=progress)
        if progress is not None:
            progress("source-context", {"remaining_batches": len(missing_indexes)})
        source_context = load_polymarket_feature_source_context(
            store,
            run_id=selected,
            config=cfg.feature,
            progress=progress,
        )
    for completed, index in enumerate(missing_indexes, start=1):
        identity, _batch_id = batch_specs[index]
        conditions = tuple(str(value) for value in identity["condition_ids"])
        if progress is not None:
            progress(
                "batch-started",
                {
                    "batch_index": index,
                    "completed_missing_batches": completed - 1,
                    "condition_count": len(conditions),
                },
            )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=selected,
            allow_segmented_gaps=cfg.feature.allow_segmented_gaps,
            book_sample_interval_ms=0,
            condition_ids=conditions,
        )
        feature_replay = replay.with_book_sample_interval(cfg.feature.cadence_ms)
        features = build_polymarket_feature_dataset(
            store,
            run_id=selected,
            config=cfg.feature,
            condition_ids=conditions,
            source_context=source_context,
            preloaded_replay=feature_replay,
        )
        materialize_polymarket_feature_dataset(store, features)
        actions = build_polymarket_action_value_dataset(
            features,
            PolymarketRepricingExecutionContext(replay),
            config=cfg.action,
        )
        action_materialization = materialize_polymarket_action_value_dataset(
            store,
            actions,
        )
        results[index] = _persist_batch(
            store,
            identity=identity,
            feature_dataset_sha256=features.dataset_sha256,
            action_dataset_sha256=actions.dataset_sha256,
            feature_row_count=len(features.rows),
            action_count=len(actions.features),
            classifier_eligible_count=(
                action_materialization.classifier_eligible_count
            ),
            positive_complete_count=action_materialization.positive_complete_count,
            category_counts=actions.category_counts,
            terminal_reason_counts=actions.terminal_reason_counts,
        )
        if progress is not None:
            progress(
                "batch-complete",
                {
                    "batch_index": index,
                    "feature_row_count": len(features.rows),
                    "action_count": len(actions.features),
                },
            )
        del actions, feature_replay, replay, features
        gc.collect()
    batches = tuple(item for item in results if item is not None)
    if len(batches) != len(batch_specs):
        raise RuntimeError("Polymarket action pipeline failed to finalize every batch")
    category_counts: dict[str, int] = {}
    terminal_counts: dict[str, int] = {}
    for batch in batches:
        _add_counts(category_counts, batch.category_counts)
        _add_counts(terminal_counts, batch.terminal_reason_counts)
    provisional = PolymarketActionPipelineReport(
        run_id=selected,
        run_report_sha256=run_report_sha256,
        config=cfg,
        eligibility_sha256=eligibility_digest,
        batches=batches,
        action_count=sum(item.action_count for item in batches),
        classifier_eligible_count=sum(
            item.classifier_eligible_count for item in batches
        ),
        positive_complete_count=sum(item.positive_complete_count for item in batches),
        category_counts=dict(sorted(category_counts.items())),
        terminal_reason_counts=dict(sorted(terminal_counts.items())),
        report_sha256="",
    )
    report = replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    ).validated()
    stored_values = (
        report.report_sha256,
        POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION,
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        report.run_id,
        report.run_report_sha256,
        _canonical_json(report.config.asdict()),
        report.eligibility_sha256,
        _canonical_json([item.batch_id for item in report.batches]),
        _canonical_json([item.action_dataset_sha256 for item in report.batches]),
        report.action_count,
        report.classifier_eligible_count,
        report.positive_complete_count,
        _canonical_json(report.category_counts),
        _canonical_json(report.terminal_reason_counts),
        _canonical_json(report.asdict()),
    )
    existing_report = (
        store.connect()
        .execute(
            """
        SELECT report_sha256, schema_version, contract_sha256, run_id,
               run_report_sha256, config_json, eligibility_sha256,
               batch_ids_json, action_dataset_sha256_json, action_count,
               classifier_eligible_count, positive_complete_count,
               category_counts_json, terminal_reason_counts_json, report_json
        FROM polymarket_action_value_pipeline WHERE report_sha256 = ?
        """,
            [report.report_sha256],
        )
        .fetchone()
    )
    if existing_report is None:
        store.connect().execute(
            """
            INSERT INTO polymarket_action_value_pipeline VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            stored_values,
        )
    elif tuple(existing_report) != stored_values:
        raise ValueError("stored Polymarket action pipeline report is inconsistent")
    return report


__all__ = [
    "POLYMARKET_ACTION_BATCH_SCHEMA_VERSION",
    "POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION",
    "PolymarketActionBatchResult",
    "PolymarketActionPipelineConfig",
    "PolymarketActionPipelineReport",
    "materialize_polymarket_action_value_batches",
]
