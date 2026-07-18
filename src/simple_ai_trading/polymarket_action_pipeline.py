"""Resumable bounded materialization for hash-bound Polymarket action evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib
import json
from pathlib import Path

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_action_value import (
    POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
    PolymarketActionValueConfig,
    build_polymarket_action_value_dataset,
    materialize_polymarket_action_value_dataset,
)
from .polymarket_continuity import polymarket_round9_evidence_window
from .polymarket_features import (
    POLYMARKET_DATASET_SCHEMA_VERSION,
    POLYMARKET_FEATURE_SCHEMA_VERSION,
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
    load_polymarket_feature_source_context,
    materialize_polymarket_feature_dataset,
    validate_polymarket_feature_source_scope,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_repricing import PolymarketRepricingExecutionContext
from .polymarket_round12_admission import (
    build_round12_action_evidence_datasets,
    load_round12_admission_dataset,
    materialize_round12_admission_dataset,
)
from .polymarket_round13 import (
    PolymarketRound13Program,
    build_round13_label_free_dataset,
    load_round13_label_free_dataset,
    materialize_round13_label_free_dataset,
)


POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION = (
    "polymarket-action-value-bounded-pipeline-v2"
)
POLYMARKET_ACTION_BATCH_SCHEMA_VERSION = "polymarket-action-value-batch-v2"
POLYMARKET_ACTION_IMPLEMENTATION_SCHEMA_VERSION = (
    "polymarket-action-value-implementation-v3"
)
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_CRITICAL_IMPLEMENTATION_FILES = (
    "cli.py",
    "duckdb_batch.py",
    "polymarket_action_pipeline.py",
    "polymarket_action_value.py",
    "polymarket_features.py",
    "polymarket_continuity.py",
    "polymarket_coverage.py",
    "polymarket_recorder.py",
    "polymarket_replay.py",
    "polymarket_repricing.py",
    "polymarket_resolution.py",
    "polymarket_round12_admission.py",
    "polymarket_round12_capture.py",
    "polymarket_round13.py",
    "polymarket_round13_capture.py",
    "polymarket_round13_evaluation.py",
    "polymarket_round13_publication.py",
)
_CONTINUITY_ADMISSION_MODES = frozenset({"group", "action_local"})


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(raw: object, *, name: str) -> object:
    try:
        return json.loads(
            str(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid JSON") from exc


def _nonnegative_count_mapping(raw: object, *, name: str) -> dict[str, int]:
    decoded = _strict_json(raw, name=name)
    if not isinstance(decoded, Mapping) or any(
        not isinstance(key, str) or not key or type(value) is not int or value < 0
        for key, value in decoded.items()
    ):
        raise ValueError(f"{name} is not a non-negative integer count object")
    return {key: value for key, value in decoded.items()}


def polymarket_action_pipeline_implementation_sha256() -> str:
    """Hash normalized UTF-8 source so identity is interpreter and OS neutral."""

    source_root = Path(__file__).resolve().parent
    module_digests: dict[str, str] = {}
    for filename in _CRITICAL_IMPLEMENTATION_FILES:
        path = source_root / filename
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"cannot attest Polymarket action implementation: {filename}"
            ) from exc
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        module_digests[filename] = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()
    return _sha256(
        {
            "schema_version": POLYMARKET_ACTION_IMPLEMENTATION_SCHEMA_VERSION,
            "critical_module_source_sha256": module_digests,
        }
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
    excluded_after_event_scope_count: int
    round13_scenario_dataset_sha256: str
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
        if not self.round13_scenario_dataset_sha256:
            payload.pop("round13_scenario_dataset_sha256")
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
            or any(value < 0 for value in self.category_counts.values())
            or any(value < 0 for value in self.terminal_reason_counts.values())
            or sum(self.category_counts.values()) != self.action_count
            or sum(self.terminal_reason_counts.values()) != self.action_count
            or self.excluded_after_event_scope_count < 0
            or (
                self.round13_scenario_dataset_sha256
                and not _is_sha256(self.round13_scenario_dataset_sha256)
            )
        ):
            raise ValueError("Polymarket action batch result is invalid")
        return self


@dataclass(frozen=True)
class PolymarketActionPipelineReport:
    run_id: str
    run_report_sha256: str
    config: PolymarketActionPipelineConfig
    eligibility_sha256: str
    implementation_sha256: str
    batches: tuple[PolymarketActionBatchResult, ...]
    action_count: int
    classifier_eligible_count: int
    positive_complete_count: int
    category_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    excluded_after_event_scope_count: int
    continuity_admission_mode: str
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "run_id": self.run_id,
            "run_report_sha256": self.run_report_sha256,
            "config": self.config.asdict(),
            "eligibility_sha256": self.eligibility_sha256,
            "implementation_sha256": self.implementation_sha256,
            "batches": [batch.asdict() for batch in self.batches],
            "action_count": self.action_count,
            "classifier_eligible_count": self.classifier_eligible_count,
            "positive_complete_count": self.positive_complete_count,
            "category_counts": dict(sorted(self.category_counts.items())),
            "terminal_reason_counts": dict(sorted(self.terminal_reason_counts.items())),
            "excluded_after_event_scope_count": (self.excluded_after_event_scope_count),
            "training_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if self.continuity_admission_mode != "group":
            payload["continuity_admission_mode"] = self.continuity_admission_mode
        return payload

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
            or self.continuity_admission_mode not in _CONTINUITY_ADMISSION_MODES
            or (
                self.continuity_admission_mode == "action_local"
                and (
                    not self.config.feature.allow_segmented_gaps
                    or not self.eligibility_sha256
                )
            )
            or not _is_sha256(self.implementation_sha256)
            or not self.batches
            or self.action_count != sum(item.action_count for item in self.batches)
            or self.classifier_eligible_count
            != sum(item.classifier_eligible_count for item in self.batches)
            or self.positive_complete_count
            != sum(item.positive_complete_count for item in self.batches)
            or any(value < 0 for value in self.category_counts.values())
            or any(value < 0 for value in self.terminal_reason_counts.values())
            or self.category_counts != expected_categories
            or self.terminal_reason_counts != expected_reasons
            or self.excluded_after_event_scope_count
            != sum(item.excluded_after_event_scope_count for item in self.batches)
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
            batch_sha256 VARCHAR NOT NULL,
            implementation_sha256 VARCHAR,
            excluded_after_event_scope_count UBIGINT,
            round13_scenario_dataset_sha256 VARCHAR
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
            report_json VARCHAR NOT NULL,
            implementation_sha256 VARCHAR,
            excluded_after_event_scope_count UBIGINT
        );

        ALTER TABLE polymarket_action_value_batch
        ADD COLUMN IF NOT EXISTS implementation_sha256 VARCHAR;
        ALTER TABLE polymarket_action_value_batch
        ADD COLUMN IF NOT EXISTS excluded_after_event_scope_count UBIGINT;
        ALTER TABLE polymarket_action_value_batch
        ADD COLUMN IF NOT EXISTS round13_scenario_dataset_sha256 VARCHAR;
        ALTER TABLE polymarket_action_value_pipeline
        ADD COLUMN IF NOT EXISTS implementation_sha256 VARCHAR;
        ALTER TABLE polymarket_action_value_pipeline
        ADD COLUMN IF NOT EXISTS excluded_after_event_scope_count UBIGINT;
        """
    )


def _batch_identity(
    *,
    run_id: str,
    run_report_sha256: str,
    config: PolymarketActionPipelineConfig,
    eligibility_sha256: str,
    implementation_sha256: str,
    continuity_admission_mode: str,
    round13_contract_sha256: str,
    group_starts_ms: Sequence[int],
    condition_ids: Sequence[str],
) -> dict[str, object]:
    identity = {
        "schema_version": POLYMARKET_ACTION_BATCH_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        "run_id": run_id,
        "run_report_sha256": run_report_sha256,
        "config": config.asdict(),
        "eligibility_sha256": eligibility_sha256,
        "implementation_sha256": implementation_sha256,
        "group_starts_ms": [int(value) for value in group_starts_ms],
        "condition_ids": list(condition_ids),
    }
    if continuity_admission_mode != "group":
        identity["continuity_admission_mode"] = continuity_admission_mode
    if round13_contract_sha256:
        identity["round13_contract_sha256"] = round13_contract_sha256
    return identity


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
               terminal_reason_counts_json, batch_sha256,
               implementation_sha256, excluded_after_event_scope_count,
               round13_scenario_dataset_sha256
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
        SELECT d.row_count, count(r.feature_id), d.replay_diagnostics_json,
               d.schema_version, d.feature_schema_version, d.source_scope_json
        FROM polymarket_feature_dataset AS d
        LEFT JOIN polymarket_feature_row AS r
          ON r.dataset_id = d.dataset_id
        WHERE d.dataset_sha256 = ?
        GROUP BY d.row_count, d.replay_diagnostics_json, d.schema_version,
                 d.feature_schema_version, d.source_scope_json
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
    if feature_manifest is None:
        raise ValueError("stored Polymarket action batch has no feature manifest")
    try:
        replay_diagnostics = _strict_json(
            feature_manifest[2], name="stored replay diagnostics"
        )
        if not isinstance(replay_diagnostics, Mapping):
            raise ValueError("stored replay diagnostics is not an object")
        source_scope = validate_polymarket_feature_source_scope(
            _strict_json(feature_manifest[5], name="stored feature source scope"),
            run_id=str(identity["run_id"]),
            condition_ids=tuple(str(value) for value in identity["condition_ids"]),
            require_bounded=True,
        )
        excluded_after_scope = int(
            replay_diagnostics["excluded_after_event_scope_count"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "stored Polymarket action batch replay diagnostics are invalid"
        ) from exc
    if (
        tuple(feature_manifest[:2]) != (row[11], row[11])
        or str(feature_manifest[3]) != POLYMARKET_DATASET_SCHEMA_VERSION
        or str(feature_manifest[4]) != POLYMARKET_FEATURE_SCHEMA_VERSION
        or tuple(source_scope["condition_ids"])
        != tuple(sorted(str(value) for value in identity["condition_ids"]))
        or action_manifest is None
        or tuple(action_manifest[:3]) != tuple(row[12:15])
        or tuple(action_manifest[5:]) != tuple(row[12:15])
        or str(action_manifest[3]) != str(row[15])
        or str(action_manifest[4]) != str(row[16])
        or str(row[18] or "") != str(identity["implementation_sha256"])
        or row[19] is None
        or int(row[19]) != excluded_after_scope
    ):
        raise ValueError("stored Polymarket action batch references invalid manifests")
    category_counts = _nonnegative_count_mapping(
        row[15], name="stored action category counts"
    )
    terminal_reason_counts = _nonnegative_count_mapping(
        row[16], name="stored action terminal reason counts"
    )
    batch_payload = {
        **dict(identity),
        "feature_dataset_sha256": str(row[9]),
        "action_dataset_sha256": str(row[10]),
        "feature_row_count": int(row[11]),
        "action_count": int(row[12]),
        "classifier_eligible_count": int(row[13]),
        "positive_complete_count": int(row[14]),
        "category_counts": category_counts,
        "terminal_reason_counts": terminal_reason_counts,
        "excluded_after_event_scope_count": excluded_after_scope,
    }
    round13_dataset_sha256 = str(row[20] or "")
    expected_round13 = str(identity.get("round13_contract_sha256") or "")
    if expected_round13:
        if not _is_sha256(round13_dataset_sha256):
            raise ValueError("stored Round 13 scenario dataset is missing")
        scenario_dataset = load_round13_label_free_dataset(
            store,
            source_action_dataset_sha256=str(row[10]),
        )
        if (
            scenario_dataset.dataset_sha256 != round13_dataset_sha256
            or scenario_dataset.contract_sha256 != expected_round13
        ):
            raise ValueError("stored Round 13 scenario dataset differs")
        batch_payload["round13_scenario_dataset_sha256"] = round13_dataset_sha256
    elif round13_dataset_sha256:
        raise ValueError("legacy action batch unexpectedly references Round 13")
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
        excluded_after_event_scope_count=excluded_after_scope,
        round13_scenario_dataset_sha256=round13_dataset_sha256,
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
    excluded_after_event_scope_count: int,
    round13_scenario_dataset_sha256: str = "",
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
        "excluded_after_event_scope_count": int(excluded_after_event_scope_count),
    }
    if round13_scenario_dataset_sha256:
        if not _is_sha256(round13_scenario_dataset_sha256):
            raise ValueError("Round 13 scenario dataset digest is invalid")
        payload["round13_scenario_dataset_sha256"] = round13_scenario_dataset_sha256
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
        str(identity["implementation_sha256"]),
        int(excluded_after_event_scope_count),
        round13_scenario_dataset_sha256 or None,
    )
    store.connect().execute(
        """
        INSERT INTO polymarket_action_value_batch (
            batch_id, schema_version, contract_sha256, run_id,
            run_report_sha256, config_json, eligibility_sha256,
            group_starts_json, condition_ids_json,
            feature_dataset_sha256, action_dataset_sha256,
            feature_row_count, action_count, classifier_eligible_count,
            positive_complete_count, category_counts_json,
            terminal_reason_counts_json, batch_sha256,
            implementation_sha256, excluded_after_event_scope_count,
            round13_scenario_dataset_sha256
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
        excluded_after_event_scope_count=int(excluded_after_event_scope_count),
        round13_scenario_dataset_sha256=round13_scenario_dataset_sha256,
        batch_sha256=batch_sha256,
    ).validated()


def _add_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _require_no_round13_resolution_evidence(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> None:
    count = int(
        store.connect()
        .execute(
            "SELECT count(*) FROM polymarket_resolution_evidence WHERE run_id = ?",
            [run_id],
        )
        .fetchone()[0]
    )
    if count:
        raise ValueError(
            "Round 13 label-free materialization requires no official resolution "
            "evidence"
        )


def materialize_polymarket_action_value_batches(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    config: PolymarketActionPipelineConfig | None = None,
    eligible_condition_ids: Sequence[str] | None = None,
    eligibility_sha256: str = "",
    continuity_admission_mode: str = "group",
    round13_program: PolymarketRound13Program | None = None,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> PolymarketActionPipelineReport:
    """Build resumable action values without materializing a full-run CLOB replay."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Polymarket action pipeline requires an explicit run ID")
    cfg = (config or PolymarketActionPipelineConfig()).validated()
    eligibility_digest = str(eligibility_sha256 or "").strip()
    admission_mode = str(continuity_admission_mode or "").strip().lower()
    frozen_round13 = None if round13_program is None else round13_program.validated()
    round13_contract_digest = (
        "" if frozen_round13 is None else frozen_round13.contract_sha256
    )
    if admission_mode not in _CONTINUITY_ADMISSION_MODES:
        raise ValueError("Polymarket continuity admission mode is invalid")
    if eligibility_digest and not _is_sha256(eligibility_digest):
        raise ValueError("Polymarket eligibility digest is invalid")
    if (
        admission_mode == "group"
        and cfg.feature.allow_segmented_gaps
        and (not eligibility_digest or eligible_condition_ids is None)
    ):
        raise ValueError(
            "segmented action replay requires hash-bound eligible condition IDs"
        )
    if admission_mode == "action_local" and (
        not cfg.feature.allow_segmented_gaps
        or not eligibility_digest
        or eligible_condition_ids is not None
    ):
        raise ValueError(
            "action-local replay requires segmented gaps, a frozen contract digest, "
            "and full condition scope"
        )
    if frozen_round13 is not None and (
        admission_mode != "action_local"
        or eligibility_digest != frozen_round13.contract_sha256
    ):
        raise ValueError(
            "Round 13 materialization requires its action-local contract digest"
        )
    implementation_sha256 = polymarket_action_pipeline_implementation_sha256()
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
    integrity = store.resume_integrity_errors(selected, progress=progress)
    if integrity:
        raise ValueError(
            "Polymarket action pipeline integrity failed: " + "; ".join(integrity)
        )
    if frozen_round13 is not None:
        _require_no_round13_resolution_evidence(store, selected)
    _ensure_pipeline_tables(store)
    markets = PolymarketEvidenceReplay.load_markets(store, run_id=selected)
    if admission_mode == "action_local":
        discovered_groups: dict[int, list[PolymarketFiveMinuteMarket]] = {}
        for market in markets:
            discovered_groups.setdefault(int(market.event_start_ms), []).append(market)
        selected_conditions = {
            market.condition_id
            for values in discovered_groups.values()
            if tuple(
                item.asset
                for item in sorted(values, key=lambda item: _ASSETS.index(item.asset))
            )
            == _ASSETS
            for market in values
        }
    else:
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
    replay_cutoff_by_condition: dict[str, int] = {}
    for condition_id in sorted(selected_conditions):
        market = market_by_condition[condition_id]
        _window_start, window_end = polymarket_round9_evidence_window(
            event_start_ms=market.event_start_ms,
            end_ms=market.end_ms,
        )
        replay_cutoff_by_condition[condition_id] = window_end
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
            implementation_sha256=implementation_sha256,
            continuity_admission_mode=admission_mode,
            round13_contract_sha256=round13_contract_digest,
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
        if existing is not None and admission_mode == "action_local":
            load_round12_admission_dataset(
                store,
                source_action_dataset_sha256=existing.action_dataset_sha256,
            )
        results.append(existing)
        if existing is None:
            missing_indexes.append(index)
    if missing_indexes:
        store.ensure_condition_message_cache(
            selected,
            condition_ids=tuple(sorted(selected_conditions)),
            progress=progress,
        )
        store.ensure_capture_chunk_receipt_index(selected, progress=progress)
    for completed, index in enumerate(missing_indexes, start=1):
        if frozen_round13 is not None:
            _require_no_round13_resolution_evidence(store, selected)
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
        source_context = load_polymarket_feature_source_context(
            store,
            run_id=selected,
            config=cfg.feature,
            condition_ids=tuple(sorted(selected_conditions)),
            source_window_condition_ids=conditions,
            continuity_report_sha256=(
                eligibility_digest if admission_mode == "group" else ""
            ),
            progress=progress,
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=selected,
            allow_segmented_gaps=cfg.feature.allow_segmented_gaps,
            book_sample_interval_ms=0,
            condition_ids=conditions,
            continuity_report_sha256=(
                eligibility_digest if admission_mode == "group" else ""
            ),
            maximum_received_wall_ms_by_condition={
                condition: replay_cutoff_by_condition[condition]
                for condition in conditions
            },
            materialized_minimum_depth_levels=3,
            cap_materialized_depth_to_minimum_order_size=(frozen_round13 is None),
        )
        feature_replay = replay.with_book_sample_interval(cfg.feature.cadence_ms)
        store.recycle_analytical_connections()
        if progress is not None:
            progress(
                "analytical-connections-recycled",
                {"batch_index": index},
            )
        features = build_polymarket_feature_dataset(
            store,
            run_id=selected,
            config=cfg.feature,
            condition_ids=conditions,
            source_context=source_context,
            preloaded_replay=feature_replay,
        )
        materialize_polymarket_feature_dataset(store, features)
        execution_context = PolymarketRepricingExecutionContext(replay)
        admissions = None
        if admission_mode == "action_local":
            actions, admissions = build_round12_action_evidence_datasets(
                features,
                execution_context,
                config=cfg.action,
            )
        else:
            actions = build_polymarket_action_value_dataset(
                features,
                execution_context,
                config=cfg.action,
            )
        action_materialization = materialize_polymarket_action_value_dataset(
            store,
            actions,
        )
        if admissions is not None:
            materialize_round12_admission_dataset(store, admissions)
        round13_dataset_sha256 = ""
        if frozen_round13 is not None:
            round13_dataset = build_round13_label_free_dataset(
                features,
                actions,
                execution_context,
                frozen_round13,
            )
            _require_no_round13_resolution_evidence(store, selected)
            materialize_round13_label_free_dataset(store, round13_dataset)
            round13_dataset_sha256 = round13_dataset.dataset_sha256
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
            excluded_after_event_scope_count=(
                replay.diagnostics.excluded_after_event_scope_count
            ),
            round13_scenario_dataset_sha256=round13_dataset_sha256,
        )
        if progress is not None:
            progress(
                "batch-complete",
                {
                    "batch_index": index,
                    "feature_row_count": len(features.rows),
                    "action_count": len(actions.features),
                    "excluded_after_event_scope_count": (
                        replay.diagnostics.excluded_after_event_scope_count
                    ),
                },
            )
        if frozen_round13 is not None:
            del round13_dataset
        del actions, admissions, execution_context
        del feature_replay, replay, features, source_context
        gc.collect()
    if frozen_round13 is not None:
        _require_no_round13_resolution_evidence(store, selected)
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
        implementation_sha256=implementation_sha256,
        batches=batches,
        action_count=sum(item.action_count for item in batches),
        classifier_eligible_count=sum(
            item.classifier_eligible_count for item in batches
        ),
        positive_complete_count=sum(item.positive_complete_count for item in batches),
        category_counts=dict(sorted(category_counts.items())),
        terminal_reason_counts=dict(sorted(terminal_counts.items())),
        excluded_after_event_scope_count=sum(
            item.excluded_after_event_scope_count for item in batches
        ),
        continuity_admission_mode=admission_mode,
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
        report.implementation_sha256,
        report.excluded_after_event_scope_count,
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
               , implementation_sha256, excluded_after_event_scope_count
        FROM polymarket_action_value_pipeline WHERE report_sha256 = ?
        """,
            [report.report_sha256],
        )
        .fetchone()
    )
    if existing_report is None:
        store.connect().execute(
            """
            INSERT INTO polymarket_action_value_pipeline (
                report_sha256, schema_version, contract_sha256, run_id,
                run_report_sha256, config_json, eligibility_sha256,
                batch_ids_json, action_dataset_sha256_json, action_count,
                classifier_eligible_count, positive_complete_count,
                category_counts_json, terminal_reason_counts_json, report_json,
                implementation_sha256, excluded_after_event_scope_count
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            stored_values,
        )
    elif tuple(existing_report) != stored_values:
        raise ValueError("stored Polymarket action pipeline report is inconsistent")
    return report


__all__ = [
    "POLYMARKET_ACTION_BATCH_SCHEMA_VERSION",
    "POLYMARKET_ACTION_IMPLEMENTATION_SCHEMA_VERSION",
    "POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION",
    "PolymarketActionBatchResult",
    "PolymarketActionPipelineConfig",
    "PolymarketActionPipelineReport",
    "materialize_polymarket_action_value_batches",
    "polymarket_action_pipeline_implementation_sha256",
]
