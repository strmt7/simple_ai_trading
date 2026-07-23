"""Bounded selected-anchor target publication for the Round 73 shock study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

import duckdb
import numpy as np

from .impact_absorption_cohort import (
    ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
    ROUND73_SHOCK_ANCHOR_TABLE,
    ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
    ROUND73_SHOCK_STUDY_TABLE,
    ROUND73_STUDY_NOT_BEFORE_WALL_NS,
    Round73ShockAnchor,
    audit_round73_shock_cohort,
)
from .impact_absorption_corpus import (
    ROUND73_CORPUS_RUN_TABLE,
    audit_round73_corpus_manifest,
)
from .impact_absorption_grid_store import (
    ROUND73_GRID_MANIFEST_TABLE,
    audit_round73_causal_grid,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
)
from .impact_absorption_target_store import (
    Round73TargetOption,
    Round73TargetReplayAnchor,
    parse_round73_target_quantity_rules,
    replay_round73_target_rows_v9,
    round73_target_option_from_row,
    round73_target_option_invariant_errors,
    round73_target_quantity_invariant_errors,
)
from .impact_absorption_targets import ROUND73_TARGET_SIDES


ROUND73_TARGET_V2_SCHEMA_VERSION = "round-073-selected-anchor-target-v2"
ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION = "round-073-target-study-v2"
ROUND73_TARGET_V2_OPTION_TABLE = "impact_target_option_v2"
ROUND73_TARGET_V2_MANIFEST_TABLE = "impact_target_run_manifest_v2"
ROUND73_TARGET_V2_STUDY_TABLE = "impact_target_study_manifest_v2"
ROUND73_TARGET_V1_MANIFEST_TABLE = "impact_target_run_manifest_v1"
ROUND73_TARGET_V2_ENTRY_DELAYS_MS = (500, 1_000)
ROUND73_TARGET_V2_HORIZONS_MS = (15_000, 60_000, 300_000)
ROUND73_TARGET_V2_REFERENCE_NOTIONALS = (100.0, 1_000.0, 5_000.0)
ROUND73_TARGET_V2_SIDES = ROUND73_TARGET_SIDES
ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR = (
    len(ROUND73_TARGET_V2_ENTRY_DELAYS_MS)
    * len(ROUND73_TARGET_V2_HORIZONS_MS)
    * len(ROUND73_TARGET_V2_REFERENCE_NOTIONALS)
    * len(ROUND73_TARGET_V2_SIDES)
)

_INELIGIBLE_REASONS = (
    "quantity_filter",
    "entry_state_late",
    "entry_capacity",
    "entry_minimum_notional",
    "funding_boundary",
    "path_capacity",
    "exit_state_late",
    "exit_capacity",
    "coverage_end",
)

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_STUDY_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FETCH_BATCH_SIZE = 8_192
_INSERT_BATCH_SIZE = 32_768
_BASE_OPTION_COLUMNS = tuple(Round73TargetOption.__dataclass_fields__)

AuditFunction = Callable[..., object]
ReplayFunction = Callable[..., list[Round73TargetOption]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _stream_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_json_object(raw_text: str, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key is forbidden in {label}: {key}")
            output[key] = value
        return output

    value = json.loads(
        raw_text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite JSON: {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _validated_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    selected = str(value).strip().lower()
    if pattern.fullmatch(selected) is None:
        raise ValueError(f"Round 73 {label} must be 32 lowercase hex characters")
    return selected


@dataclass(frozen=True)
class _StudySource:
    run_id: str
    corpus_manifest_sha256: str
    grid_manifest_sha256: str
    coverage_start_wall_ns: int
    coverage_end_wall_ns: int


@dataclass(frozen=True)
class _StudyContext:
    study_id: str
    study_manifest_sha256: str
    sources: tuple[_StudySource, ...]
    anchors: tuple[Round73ShockAnchor, ...]

    @property
    def source_by_run(self) -> dict[str, _StudySource]:
        return {source.run_id: source for source in self.sources}


@dataclass(frozen=True)
class Round73CohortTargetOption(Round73TargetOption):
    study_id: str
    selected_anchor_sha256: str
    cohort_option_sha256: str

    def values_without_cohort_hash(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())[:-1]

    def as_row(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


_V2_OPTION_COLUMNS = tuple(Round73CohortTargetOption.__dataclass_fields__)


@dataclass(frozen=True)
class Round73TargetV2BuildReport:
    study_id: str
    run_id: str
    selected_anchor_count: int
    option_count: int
    eligible_option_count: int
    positive_option_count: int
    target_manifest_sha256: str
    cohort_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_TARGET_V2_SCHEMA_VERSION,
                "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                "target_constructed": True,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73TargetV2Audit:
    study_id: str
    run_id: str
    passed: bool
    errors: tuple[str, ...]
    selected_anchor_count: int
    option_count: int
    eligible_option_count: int
    positive_option_count: int
    target_manifest_sha256: str
    deep_replay_performed: bool

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "round-073-selected-anchor-target-audit-v2",
                "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                "errors": list(self.errors),
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73TargetStudyReport:
    study_id: str
    source_run_count: int
    selected_anchor_count: int
    option_count: int
    eligible_option_count: int
    positive_option_count: int
    target_run_manifests_sha256: str
    target_study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION,
                "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                "target_study_sealed": True,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73TargetStudyAudit:
    study_id: str
    passed: bool
    errors: tuple[str, ...]
    source_run_count: int
    audited_target_run_count: int
    option_count: int
    target_study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "round-073-target-study-audit-v2",
                "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                "errors": list(self.errors),
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


def _shock_anchor_from_row(row: Sequence[object]) -> Round73ShockAnchor:
    columns = tuple(Round73ShockAnchor.__dataclass_fields__)
    if len(row) != len(columns):
        raise ValueError("Round 73 selected-anchor row width differs")
    values = dict(zip(columns, row, strict=True))
    for name in (
        "anchor_index",
        "anchor_monotonic_ns",
        "anchor_wall_ns",
        "source_max_received_monotonic_ns",
        "day_ordinal",
        "shock_direction",
    ):
        values[name] = int(values[name])
    for name in ("shock_ratio", "shock_direction_taker_share"):
        values[name] = float(values[name])
    return Round73ShockAnchor(**values)  # type: ignore[arg-type]


def _cohort_option_from_base(
    option: Round73TargetOption,
    *,
    study_id: str,
    selected_anchor_sha256: str,
) -> Round73CohortTargetOption:
    values = {
        **asdict(option),
        "study_id": study_id,
        "selected_anchor_sha256": selected_anchor_sha256,
    }
    row_hash = _sha256_text(_canonical_json(list(values.values())))
    return Round73CohortTargetOption(
        **values,  # type: ignore[arg-type]
        cohort_option_sha256=row_hash,
    )


def _cohort_option_from_row(row: Sequence[object]) -> Round73CohortTargetOption:
    if len(row) != len(_V2_OPTION_COLUMNS):
        raise ValueError("Round 73 v2 target row width differs")
    base = round73_target_option_from_row(row[: len(_BASE_OPTION_COLUMNS)])
    return Round73CohortTargetOption(
        **asdict(base),
        study_id=str(row[-3]),
        selected_anchor_sha256=str(row[-2]),
        cohort_option_sha256=str(row[-1]),
    )


def _load_study_context(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
) -> _StudyContext:
    required = (ROUND73_SHOCK_STUDY_TABLE, ROUND73_SHOCK_ANCHOR_TABLE)
    if not all(_table_exists(connection, table) for table in required):
        raise ValueError("Round 73 shock cohort tables are missing")
    row = connection.execute(
        f"SELECT schema_version, contract_sha256, manifest_json, manifest_sha256, "
        "source_runs_sha256, selected_anchor_rows_sha256, source_run_count, "
        f"selected_anchor_count FROM {ROUND73_SHOCK_STUDY_TABLE} WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 shock cohort study was not found")
    manifest_text = str(row[2])
    manifest_sha = str(row[3])
    identity = _strict_json_object(manifest_text, "Round 73 shock cohort manifest")
    if (
        str(row[0]) != ROUND73_SHOCK_STUDY_SCHEMA_VERSION
        or str(row[1]) != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
        or _SHA256.fullmatch(manifest_sha) is None
        or _sha256_text(manifest_text) != manifest_sha
        or identity.get("study_id") != study_id
        or identity.get("schema_version") != ROUND73_SHOCK_STUDY_SCHEMA_VERSION
        or identity.get("contract_sha256") != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
        or identity.get("target_observed") is not False
        or identity.get("model_evaluated") is not False
        or identity.get("trading_authority") is not False
    ):
        raise ValueError("Round 73 shock cohort identity differs")
    raw_sources = identity.get("source_runs")
    if not isinstance(raw_sources, Sequence) or isinstance(
        raw_sources, (str, bytes, bytearray)
    ):
        raise ValueError("Round 73 shock cohort sources are missing")
    sources: list[_StudySource] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise ValueError("Round 73 shock cohort source is invalid")
        source = _StudySource(
            run_id=str(raw["run_id"]),
            corpus_manifest_sha256=str(raw["corpus_manifest_sha256"]),
            grid_manifest_sha256=str(raw["grid_manifest_sha256"]),
            coverage_start_wall_ns=int(raw["coverage_start_wall_ns"]),
            coverage_end_wall_ns=int(raw["coverage_end_wall_ns"]),
        )
        if (
            _RUN_ID.fullmatch(source.run_id) is None
            or _SHA256.fullmatch(source.corpus_manifest_sha256) is None
            or _SHA256.fullmatch(source.grid_manifest_sha256) is None
            or source.coverage_end_wall_ns <= source.coverage_start_wall_ns
        ):
            raise ValueError("Round 73 shock cohort source identity is invalid")
        sources.append(source)
    if len(sources) != int(row[6]) or len({item.run_id for item in sources}) != len(
        sources
    ):
        raise ValueError("Round 73 shock cohort source count differs")
    expected_source_hash = _sha256_text(
        _canonical_json([asdict(item) for item in sources])
    )
    if expected_source_hash != str(row[4]) or identity.get("source_runs_sha256") != str(
        row[4]
    ):
        raise ValueError("Round 73 shock cohort source hash differs")
    columns = tuple(Round73ShockAnchor.__dataclass_fields__)
    anchor_rows = connection.execute(
        f"SELECT {', '.join(columns)} FROM {ROUND73_SHOCK_ANCHOR_TABLE} "
        "WHERE study_id = ? ORDER BY symbol, anchor_wall_ns, run_id, anchor_index",
        [study_id],
    ).fetchall()
    anchors = tuple(_shock_anchor_from_row(item) for item in anchor_rows)
    source_ids = {source.run_id for source in sources}
    for anchor in anchors:
        expected_hash = _sha256_text(
            _canonical_json(list(anchor.values_without_hash()))
        )
        if (
            anchor.study_id != study_id
            or anchor.run_id not in source_ids
            or anchor.symbol not in IMPACT_CAPTURE_SYMBOLS
            or anchor.selected_anchor_sha256 != expected_hash
            or _SHA256.fullmatch(anchor.feature_vector_sha256) is None
        ):
            raise ValueError("Round 73 shock cohort anchor identity differs")
    anchor_rows_sha = _stream_hash(
        [anchor.selected_anchor_sha256 for anchor in anchors]
    )
    if (
        len(anchors) != int(row[7])
        or anchor_rows_sha != str(row[5])
        or identity.get("selected_anchor_rows_sha256") != anchor_rows_sha
        or identity.get("selected_anchor_count") != len(anchors)
    ):
        raise ValueError("Round 73 shock cohort anchor aggregate differs")
    placeholders = ",".join("?" for _ in sources)
    if _table_exists(connection, ROUND73_TARGET_V1_MANIFEST_TABLE):
        contaminated = int(
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V1_MANIFEST_TABLE} "
                f"WHERE run_id IN ({placeholders})",
                [source.run_id for source in sources],
            ).fetchone()[0]
        )
        if contaminated:
            raise ValueError("Round 73 shock cohort was contaminated by v1 targets")
    return _StudyContext(
        study_id=study_id,
        study_manifest_sha256=manifest_sha,
        sources=tuple(sources),
        anchors=anchors,
    )


def _create_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V2_OPTION_TABLE} (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            entry_delay_ms USMALLINT NOT NULL,
            horizon_ms UINTEGER NOT NULL,
            reference_quote_notional UINTEGER NOT NULL,
            side VARCHAR NOT NULL,
            eligible BOOLEAN NOT NULL,
            ineligible_reason_mask UINTEGER NOT NULL,
            ineligible_reasons_json VARCHAR NOT NULL,
            decision_monotonic_ns UBIGINT NOT NULL,
            decision_book_received_monotonic_ns UBIGINT NOT NULL,
            requested_entry_monotonic_ns UBIGINT NOT NULL,
            actual_entry_monotonic_ns UBIGINT,
            entry_state_lateness_ms DOUBLE,
            requested_exit_monotonic_ns UBIGINT,
            actual_exit_monotonic_ns UBIGINT,
            exit_state_lateness_ms DOUBLE,
            base_quantity DOUBLE,
            decision_mid DOUBLE NOT NULL,
            entry_average_price DOUBLE,
            entry_quote_notional DOUBLE,
            exit_average_price DOUBLE,
            exit_quote_notional DOUBLE,
            gross_payoff_quote DOUBLE,
            charge_quote DOUBLE,
            net_payoff_quote DOUBLE,
            net_payoff_bps DOUBLE,
            positive_net_payoff BOOLEAN,
            maximum_adverse_excursion_bps DOUBLE,
            maximum_favorable_excursion_bps DOUBLE,
            maximum_spread_bps DOUBLE,
            minimum_exit_side_capacity_ratio DOUBLE,
            entry_update_id UBIGINT,
            exit_update_id UBIGINT,
            option_sha256 VARCHAR NOT NULL,
            study_id VARCHAR NOT NULL,
            selected_anchor_sha256 VARCHAR NOT NULL,
            cohort_option_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (
                study_id, run_id, symbol, anchor_index, entry_delay_ms,
                horizon_ms, reference_quote_notional, side
            ),
            CHECK (length(option_sha256) = 64),
            CHECK (length(study_id) = 32),
            CHECK (length(selected_anchor_sha256) = 64),
            CHECK (length(cohort_option_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V2_MANIFEST_TABLE} (
            study_id VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            cohort_manifest_sha256 VARCHAR NOT NULL,
            source_corpus_manifest_sha256 VARCHAR NOT NULL,
            source_grid_manifest_sha256 VARCHAR NOT NULL,
            target_manifest_json VARCHAR NOT NULL,
            target_manifest_sha256 VARCHAR NOT NULL,
            option_rows_sha256 VARCHAR NOT NULL,
            selected_anchor_count UINTEGER NOT NULL,
            option_count UBIGINT NOT NULL,
            eligible_option_count UBIGINT NOT NULL,
            positive_option_count UBIGINT NOT NULL,
            first_decision_wall_ns UBIGINT,
            last_decision_wall_ns UBIGINT,
            recorded_at_wall_ns UBIGINT NOT NULL,
            PRIMARY KEY (study_id, run_id),
            CHECK (length(study_id) = 32),
            CHECK (length(contract_sha256) = 64),
            CHECK (length(cohort_manifest_sha256) = 64),
            CHECK (length(source_corpus_manifest_sha256) = 64),
            CHECK (length(source_grid_manifest_sha256) = 64),
            CHECK (length(target_manifest_sha256) = 64),
            CHECK (length(option_rows_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V2_STUDY_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            cohort_manifest_sha256 VARCHAR NOT NULL,
            target_study_manifest_json VARCHAR NOT NULL,
            target_study_manifest_sha256 VARCHAR NOT NULL,
            target_run_manifests_sha256 VARCHAR NOT NULL,
            source_run_count USMALLINT NOT NULL,
            selected_anchor_count UINTEGER NOT NULL,
            option_count UBIGINT NOT NULL,
            eligible_option_count UBIGINT NOT NULL,
            positive_option_count UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(study_id) = 32),
            CHECK (length(contract_sha256) = 64),
            CHECK (length(cohort_manifest_sha256) = 64),
            CHECK (length(target_study_manifest_sha256) = 64),
            CHECK (length(target_run_manifests_sha256) = 64)
        )
        """
    )


def _assert_table_shapes(connection: duckdb.DuckDBPyConnection) -> None:
    expected = {
        ROUND73_TARGET_V2_OPTION_TABLE: _V2_OPTION_COLUMNS,
        ROUND73_TARGET_V2_MANIFEST_TABLE: (
            "study_id",
            "run_id",
            "schema_version",
            "contract_sha256",
            "cohort_manifest_sha256",
            "source_corpus_manifest_sha256",
            "source_grid_manifest_sha256",
            "target_manifest_json",
            "target_manifest_sha256",
            "option_rows_sha256",
            "selected_anchor_count",
            "option_count",
            "eligible_option_count",
            "positive_option_count",
            "first_decision_wall_ns",
            "last_decision_wall_ns",
            "recorded_at_wall_ns",
        ),
        ROUND73_TARGET_V2_STUDY_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "cohort_manifest_sha256",
            "target_study_manifest_json",
            "target_study_manifest_sha256",
            "target_run_manifests_sha256",
            "source_run_count",
            "selected_anchor_count",
            "option_count",
            "eligible_option_count",
            "positive_option_count",
            "recorded_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 v2 target table schema differs: {table}")


def _insert_option_batch(
    connection: duckdb.DuckDBPyConnection,
    options: Sequence[Round73CohortTargetOption],
    *,
    batch_index: int,
    table_name: str = ROUND73_TARGET_V2_OPTION_TABLE,
    view_namespace: str = "v2",
) -> None:
    if not options:
        return
    rows = [option.as_row() for option in options]
    views: list[str] = []
    projections: list[str] = []
    try:
        for index, column in enumerate(zip(*rows, strict=True)):
            values = tuple(column)
            view = f"_round73_target_{view_namespace}_{batch_index}_{index}"
            if all(isinstance(value, bool) for value in values):
                array = np.asarray(values, dtype=np.bool_)
                projection = f"{view}.column0"
            elif all(value is None or isinstance(value, bool) for value in values):
                array = np.asarray(
                    [-1 if value is None else int(value) for value in values],
                    dtype=np.int8,
                )
                projection = (
                    f"CASE WHEN {view}.column0 = -1 THEN NULL "
                    f"ELSE {view}.column0 != 0 END"
                )
            elif all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                array = np.asarray(values, dtype=np.int64)
                projection = f"{view}.column0"
            elif all(
                value is None
                or (isinstance(value, int) and not isinstance(value, bool))
                for value in values
            ):
                array = np.asarray(
                    [-1 if value is None else value for value in values],
                    dtype=np.int64,
                )
                projection = (
                    f"CASE WHEN {view}.column0 = -1 THEN NULL ELSE {view}.column0 END"
                )
            elif all(
                value is None or isinstance(value, (int, float)) for value in values
            ):
                array = np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype=np.float64,
                )
                projection = (
                    f"CASE WHEN isnan({view}.column0) THEN NULL ELSE {view}.column0 END"
                )
            elif all(isinstance(value, str) for value in values):
                array = np.asarray(values, dtype=np.str_)
                projection = f"{view}.column0"
            else:
                raise TypeError("Round 73 v2 target insert column type is unsupported")
            connection.register(view, array)
            views.append(view)
            projections.append(projection)
        connection.execute(
            f"INSERT INTO {table_name} SELECT "
            + ", ".join(projections)
            + " FROM "
            + " POSITIONAL JOIN ".join(views)
        )
    finally:
        for view in views:
            connection.unregister(view)


def _selected_anchors(
    context: _StudyContext,
    *,
    run_id: str,
) -> tuple[Round73ShockAnchor, ...]:
    return tuple(anchor for anchor in context.anchors if anchor.run_id == run_id)


def _replay_anchor_map(
    anchors: Sequence[Round73ShockAnchor],
) -> dict[str, list[Round73TargetReplayAnchor]]:
    output: dict[str, list[Round73TargetReplayAnchor]] = {
        symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    for anchor in anchors:
        output[anchor.symbol].append(
            Round73TargetReplayAnchor(
                symbol=anchor.symbol,
                anchor_index=anchor.anchor_index,
                decision_monotonic_ns=anchor.anchor_monotonic_ns,
                decision_wall_ns=anchor.anchor_wall_ns,
                source_max_received_monotonic_ns=(
                    anchor.source_max_received_monotonic_ns
                ),
            )
        )
    return output


def _summarize_options(
    options: Sequence[Round73CohortTargetOption],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    reason_counts = {reason: 0 for reason in _INELIGIBLE_REASONS}
    per_symbol = {
        symbol: {"options": 0, "eligible": 0, "ineligible": 0, "positive": 0}
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    for option in options:
        counts = per_symbol[option.symbol]
        counts["options"] += 1
        if option.eligible:
            counts["eligible"] += 1
            counts["positive"] += int(bool(option.positive_net_payoff))
            continue
        counts["ineligible"] += 1
        reasons = json.loads(option.ineligible_reasons_json)
        if not isinstance(reasons, list):
            raise ValueError("Round 73 v2 target reasons are invalid")
        for reason in reasons:
            selected = str(reason)
            if selected not in reason_counts:
                raise ValueError(f"Round 73 v2 target reason differs: {selected}")
            reason_counts[selected] += 1
    return reason_counts, per_symbol


def _report_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73TargetV2BuildReport:
    return Round73TargetV2BuildReport(
        study_id=str(identity["study_id"]),
        run_id=str(identity["run_id"]),
        selected_anchor_count=int(identity["selected_anchor_count"]),
        option_count=int(identity["option_count"]),
        eligible_option_count=int(identity["eligible_option_count"]),
        positive_option_count=int(identity["positive_option_count"]),
        target_manifest_sha256=manifest_sha256,
        cohort_manifest_sha256=str(identity["cohort_manifest_sha256"]),
    )


def _source_replay_inputs(
    connection: duckdb.DuckDBPyConnection,
    *,
    source: _StudySource,
    corpus_manifest_sha256: str,
    grid_manifest_sha256: str,
) -> tuple[int, int, tuple[tuple[object, ...], ...]]:
    row = connection.execute(
        f"""
        SELECT c.manifest_sha256, c.coverage_start_wall_ns,
               c.coverage_end_wall_ns, r.schema_version,
               r.capture_contract_sha256, r.started_wall_ns,
               r.started_monotonic_ns, g.build_manifest_sha256
        FROM {ROUND73_CORPUS_RUN_TABLE} c
        JOIN impact_capture_run r ON r.run_id = c.run_id
        JOIN {ROUND73_GRID_MANIFEST_TABLE} g ON g.run_id = c.run_id
        WHERE c.run_id = ?
        """,
        [source.run_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 v2 target source identity is missing")
    if (
        str(row[0]) != source.corpus_manifest_sha256
        or str(row[0]) != corpus_manifest_sha256
        or int(row[1]) != source.coverage_start_wall_ns
        or int(row[2]) != source.coverage_end_wall_ns
        or str(row[3]) != IMPACT_CAPTURE_V9_SCHEMA_VERSION
        or str(row[4]) != IMPACT_CAPTURE_V9_CONTRACT_SHA256
        or str(row[7]) != source.grid_manifest_sha256
        or str(row[7]) != grid_manifest_sha256
    ):
        raise ValueError("Round 73 v2 target source identity differs")
    segments = tuple(
        connection.execute(
            "SELECT symbol, status, tick_size FROM impact_capture_segment "
            "WHERE run_id = ? ORDER BY symbol",
            [source.run_id],
        ).fetchall()
    )
    if tuple(str(item[0]) for item in segments) != IMPACT_CAPTURE_SYMBOLS or any(
        str(item[1]) != "valid" for item in segments
    ):
        raise ValueError("Round 73 v2 target source segments are invalid")
    return int(row[5]), int(row[6]), segments


def _build_run_identity(
    *,
    context: _StudyContext,
    source: _StudySource,
    selected_anchors: Sequence[Round73ShockAnchor],
    options: Sequence[Round73CohortTargetOption],
) -> dict[str, object]:
    reason_counts, per_symbol = _summarize_options(options)
    option_rows_sha = _stream_hash([option.cohort_option_sha256 for option in options])
    roles = {
        role: sum(anchor.role == role for anchor in selected_anchors)
        for role in ("training", "tuning", "test")
    }
    first_wall = (
        None
        if not selected_anchors
        else min(item.anchor_wall_ns for item in selected_anchors)
    )
    last_wall = (
        None
        if not selected_anchors
        else max(item.anchor_wall_ns for item in selected_anchors)
    )
    eligible_count = sum(option.eligible for option in options)
    positive_count = sum(
        option.eligible and bool(option.positive_net_payoff) for option in options
    )
    return {
        "schema_version": ROUND73_TARGET_V2_SCHEMA_VERSION,
        "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
        "study_id": context.study_id,
        "run_id": source.run_id,
        "cohort_manifest_sha256": context.study_manifest_sha256,
        "source_corpus_manifest_sha256": source.corpus_manifest_sha256,
        "source_grid_manifest_sha256": source.grid_manifest_sha256,
        "dimensions": {
            "entry_delay_milliseconds": list(ROUND73_TARGET_V2_ENTRY_DELAYS_MS),
            "holding_horizon_milliseconds": list(ROUND73_TARGET_V2_HORIZONS_MS),
            "reference_quote_notional": list(ROUND73_TARGET_V2_REFERENCE_NOTIONALS),
            "side": list(ROUND73_TARGET_V2_SIDES),
        },
        "expected_options_per_selected_anchor": (
            ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
        ),
        "selected_anchor_count": len(selected_anchors),
        "selected_anchor_role_counts": roles,
        "option_count": len(options),
        "eligible_option_count": eligible_count,
        "ineligible_option_count": len(options) - eligible_count,
        "positive_option_count": positive_count,
        "reason_counts": reason_counts,
        "per_symbol": per_symbol,
        "option_rows_sha256": option_rows_sha,
        "first_decision_wall_ns": first_wall,
        "last_decision_wall_ns": last_wall,
        "crypto_formal_daily_close": False,
        "leverage_applied": False,
        "target_constructed": True,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }


def build_round73_selected_anchor_targets(
    database: str | Path,
    *,
    study_id: str,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
    verify_cohort: bool = True,
) -> Round73TargetV2BuildReport:
    """Replay and atomically publish v2 targets for one cohort source run."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_run = _validated_identifier(run_id, _RUN_ID, "target run ID")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        if any(
            anchor.anchor_wall_ns >= ROUND73_STUDY_NOT_BEFORE_WALL_NS
            for anchor in context.anchors
        ):
            raise ValueError(
                "Round 73 eligible targets require the staged v3 holdout store"
            )
        if _table_exists(connection, ROUND73_TARGET_V2_MANIFEST_TABLE):
            existing = connection.execute(
                f"SELECT target_manifest_json, target_manifest_sha256 "
                f"FROM {ROUND73_TARGET_V2_MANIFEST_TABLE} "
                "WHERE study_id = ? AND run_id = ?",
                [selected_study, selected_run],
            ).fetchone()
        else:
            existing = None
    if existing is not None:
        audit = audit_round73_selected_anchor_targets(
            database,
            study_id=selected_study,
            run_id=selected_run,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
            verify_cohort=verify_cohort,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing v2 target audit failed")
        identity = _strict_json_object(str(existing[0]), "Round 73 v2 target manifest")
        return _report_from_identity(identity, str(existing[1]))
    if verify_cohort:
        cohort_audit = cohort_audit_function(
            database,
            study_id=selected_study,
            deep_source_audit=False,
            memory_limit=memory_limit,
            threads=threads,
        )
        if getattr(cohort_audit, "passed", False) is not True:
            raise ValueError("Round 73 v2 target cohort audit failed")
    corpus_audit = corpus_audit_function(
        database,
        run_id=selected_run,
        memory_limit=memory_limit,
        threads=threads,
    )
    grid_audit = grid_audit_function(
        database,
        run_id=selected_run,
        memory_limit=memory_limit,
        threads=threads,
    )
    if (
        getattr(corpus_audit, "passed", False) is not True
        or getattr(grid_audit, "passed", False) is not True
    ):
        raise ValueError("Round 73 v2 target source audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        source = context.source_by_run.get(selected_run)
        if source is None:
            raise ValueError("Round 73 target run is not in the frozen cohort")
        if source.corpus_manifest_sha256 != getattr(
            corpus_audit, "manifest_sha256", ""
        ) or source.grid_manifest_sha256 != getattr(
            grid_audit, "build_manifest_sha256", ""
        ):
            raise ValueError("Round 73 v2 target audited source hash differs")
        started_wall_ns, started_monotonic_ns, segments = _source_replay_inputs(
            connection,
            source=source,
            corpus_manifest_sha256=source.corpus_manifest_sha256,
            grid_manifest_sha256=source.grid_manifest_sha256,
        )
        selected_anchors = _selected_anchors(context, run_id=selected_run)
        replay_anchors = _replay_anchor_map(selected_anchors)
        anchor_hashes: dict[tuple[str, int], str] = {}
        for anchor in selected_anchors:
            anchor_hashes[(anchor.symbol, anchor.anchor_index)] = (
                anchor.selected_anchor_sha256
            )
        quantity_rules = (
            {}
            if not selected_anchors
            else parse_round73_target_quantity_rules(connection, run_id=selected_run)
        )
        base_options = (
            []
            if not selected_anchors
            else replay_function(
                connection,
                run_id=selected_run,
                anchors=replay_anchors,
                quantity_rules=quantity_rules,
                run_started_wall_ns=started_wall_ns,
                run_started_monotonic_ns=started_monotonic_ns,
                coverage_start_wall_ns=source.coverage_start_wall_ns,
                coverage_end_wall_ns=source.coverage_end_wall_ns,
                segments=segments,
                entry_delays_ms=ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
                horizons_ms=ROUND73_TARGET_V2_HORIZONS_MS,
                reference_notionals=ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
                sides=ROUND73_TARGET_V2_SIDES,
            )
        )
    expected_options = (
        len(selected_anchors) * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
    )
    if len(base_options) != expected_options:
        raise ValueError("Round 73 v2 target option count differs")
    options: list[Round73CohortTargetOption] = []
    for option in base_options:
        anchor_hash = anchor_hashes.get((option.symbol, option.anchor_index))
        if anchor_hash is None or option.run_id != selected_run:
            raise ValueError("Round 73 v2 target option is not cohort-selected")
        errors = (
            *round73_target_option_invariant_errors(
                option,
                entry_delays_ms=ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
                horizons_ms=ROUND73_TARGET_V2_HORIZONS_MS,
                reference_notionals=ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
                sides=ROUND73_TARGET_V2_SIDES,
            ),
            *round73_target_quantity_invariant_errors(
                option, quantity_rules[option.symbol]
            ),
        )
        if errors:
            raise ValueError(
                "Round 73 v2 target financial invariant failed: "
                f"{option.symbol}:{option.anchor_index}:{','.join(errors)}"
            )
        options.append(
            _cohort_option_from_base(
                option,
                study_id=selected_study,
                selected_anchor_sha256=anchor_hash,
            )
        )
    options.sort(
        key=lambda item: (
            item.symbol,
            item.anchor_index,
            item.entry_delay_ms,
            item.horizon_ms,
            item.reference_quote_notional,
            item.side,
        )
    )
    identity = _build_run_identity(
        context=context,
        source=source,
        selected_anchors=selected_anchors,
        options=options,
    )
    manifest_text = _canonical_json(identity)
    manifest_sha = _sha256_text(manifest_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_tables(connection)
        _assert_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            current_context = _load_study_context(
                connection,
                study_id=selected_study,
            )
            if current_context != context:
                raise ValueError("Round 73 v2 cohort changed before publication")
            _source_replay_inputs(
                connection,
                source=source,
                corpus_manifest_sha256=source.corpus_manifest_sha256,
                grid_manifest_sha256=source.grid_manifest_sha256,
            )
            if connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V2_STUDY_TABLE} "
                "WHERE study_id = ?",
                [selected_study],
            ).fetchone()[0]:
                raise ValueError("Round 73 target study is already sealed")
            concurrent = connection.execute(
                f"SELECT target_manifest_sha256 FROM {ROUND73_TARGET_V2_MANIFEST_TABLE} "
                "WHERE study_id = ? AND run_id = ?",
                [selected_study, selected_run],
            ).fetchone()
            if concurrent is not None:
                if str(concurrent[0]) != manifest_sha:
                    raise ValueError("Round 73 concurrent v2 target build differs")
            else:
                for batch_index, start in enumerate(
                    range(0, len(options), _INSERT_BATCH_SIZE)
                ):
                    _insert_option_batch(
                        connection,
                        options[start : start + _INSERT_BATCH_SIZE],
                        batch_index=batch_index,
                    )
                connection.execute(
                    f"INSERT INTO {ROUND73_TARGET_V2_MANIFEST_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected_study,
                        selected_run,
                        ROUND73_TARGET_V2_SCHEMA_VERSION,
                        ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                        context.study_manifest_sha256,
                        source.corpus_manifest_sha256,
                        source.grid_manifest_sha256,
                        manifest_text,
                        manifest_sha,
                        str(identity["option_rows_sha256"]),
                        int(identity["selected_anchor_count"]),
                        int(identity["option_count"]),
                        int(identity["eligible_option_count"]),
                        int(identity["positive_option_count"]),
                        identity["first_decision_wall_ns"],
                        identity["last_decision_wall_ns"],
                        time.time_ns(),
                    ],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _report_from_identity(identity, manifest_sha)


def audit_round73_selected_anchor_targets(
    database: str | Path,
    *,
    study_id: str,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
    verify_cohort: bool = True,
    deep_source_audit: bool = True,
    deep_replay: bool = True,
) -> Round73TargetV2Audit:
    """Reconcile one selected-anchor target run and its source identities."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_run = _validated_identifier(run_id, _RUN_ID, "target run ID")
    errors: list[str] = []
    selected_anchor_count = 0
    option_count = 0
    eligible_count = 0
    positive_count = 0
    manifest_sha = ""
    deep_replay_performed = False
    if verify_cohort:
        try:
            cohort_audit = cohort_audit_function(
                database,
                study_id=selected_study,
                deep_source_audit=False,
                memory_limit=memory_limit,
                threads=threads,
            )
            if getattr(cohort_audit, "passed", False) is not True:
                errors.append("cohort_audit_failed")
        except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"cohort:{type(exc).__name__}:{exc}")
    try:
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            context = _load_study_context(connection, study_id=selected_study)
            source = context.source_by_run.get(selected_run)
            if source is None:
                raise ValueError("Round 73 target run is not in the frozen cohort")
    except (
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"study:{type(exc).__name__}:{exc}")
        return Round73TargetV2Audit(
            study_id=selected_study,
            run_id=selected_run,
            passed=False,
            errors=tuple(errors),
            selected_anchor_count=0,
            option_count=0,
            eligible_option_count=0,
            positive_option_count=0,
            target_manifest_sha256="",
            deep_replay_performed=False,
        )
    if deep_source_audit:
        try:
            corpus_audit = corpus_audit_function(
                database,
                run_id=selected_run,
                memory_limit=memory_limit,
                threads=threads,
            )
            grid_audit = grid_audit_function(
                database,
                run_id=selected_run,
                memory_limit=memory_limit,
                threads=threads,
            )
            if (
                getattr(corpus_audit, "passed", False) is not True
                or getattr(corpus_audit, "manifest_sha256", "")
                != source.corpus_manifest_sha256
                or getattr(grid_audit, "passed", False) is not True
                or getattr(grid_audit, "build_manifest_sha256", "")
                != source.grid_manifest_sha256
            ):
                errors.append("source_audit_failed")
        except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"source:{type(exc).__name__}:{exc}")
    try:
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            required = (
                ROUND73_TARGET_V2_OPTION_TABLE,
                ROUND73_TARGET_V2_MANIFEST_TABLE,
                ROUND73_TARGET_V2_STUDY_TABLE,
            )
            if not all(_table_exists(connection, table) for table in required):
                raise ValueError("Round 73 v2 target table set is incomplete")
            _assert_table_shapes(connection)
            started_wall_ns, started_monotonic_ns, segments = _source_replay_inputs(
                connection,
                source=source,
                corpus_manifest_sha256=source.corpus_manifest_sha256,
                grid_manifest_sha256=source.grid_manifest_sha256,
            )
            row = connection.execute(
                f"""
                SELECT schema_version, contract_sha256,
                       cohort_manifest_sha256,
                       source_corpus_manifest_sha256,
                       source_grid_manifest_sha256, target_manifest_json,
                       target_manifest_sha256, option_rows_sha256,
                       selected_anchor_count, option_count,
                       eligible_option_count, positive_option_count,
                       first_decision_wall_ns, last_decision_wall_ns
                FROM {ROUND73_TARGET_V2_MANIFEST_TABLE}
                WHERE study_id = ? AND run_id = ?
                """,
                [selected_study, selected_run],
            ).fetchone()
            if row is None:
                raise ValueError("Round 73 v2 target manifest was not found")
            manifest_text = str(row[5])
            manifest_sha = str(row[6])
            identity = _strict_json_object(manifest_text, "Round 73 v2 target manifest")
            if (
                str(row[0]) != ROUND73_TARGET_V2_SCHEMA_VERSION
                or str(row[1]) != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
                or str(row[2]) != context.study_manifest_sha256
                or str(row[3]) != source.corpus_manifest_sha256
                or str(row[4]) != source.grid_manifest_sha256
                or _SHA256.fullmatch(manifest_sha) is None
                or _sha256_text(manifest_text) != manifest_sha
                or identity.get("schema_version") != ROUND73_TARGET_V2_SCHEMA_VERSION
                or identity.get("contract_sha256")
                != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
                or identity.get("study_id") != selected_study
                or identity.get("run_id") != selected_run
                or identity.get("cohort_manifest_sha256")
                != context.study_manifest_sha256
                or identity.get("source_corpus_manifest_sha256")
                != source.corpus_manifest_sha256
                or identity.get("source_grid_manifest_sha256")
                != source.grid_manifest_sha256
                or identity.get("crypto_formal_daily_close") is not False
                or identity.get("leverage_applied") is not False
                or identity.get("target_constructed") is not True
                or identity.get("model_evaluated") is not False
                or identity.get("profitability_claim") is not False
                or identity.get("trading_authority") is not False
            ):
                raise ValueError("Round 73 v2 target manifest identity differs")
            expected_dimensions_identity = {
                "entry_delay_milliseconds": list(ROUND73_TARGET_V2_ENTRY_DELAYS_MS),
                "holding_horizon_milliseconds": list(ROUND73_TARGET_V2_HORIZONS_MS),
                "reference_quote_notional": list(ROUND73_TARGET_V2_REFERENCE_NOTIONALS),
                "side": list(ROUND73_TARGET_V2_SIDES),
            }
            if (
                identity.get("dimensions") != expected_dimensions_identity
                or identity.get("expected_options_per_selected_anchor")
                != ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
            ):
                raise ValueError("Round 73 v2 target dimensions differ")
            selected_anchors = _selected_anchors(context, run_id=selected_run)
            selected_anchor_count = len(selected_anchors)
            anchor_hashes = {
                (anchor.symbol, anchor.anchor_index): anchor.selected_anchor_sha256
                for anchor in selected_anchors
            }
            quantity_rules = (
                {}
                if not selected_anchors
                else parse_round73_target_quantity_rules(
                    connection, run_id=selected_run
                )
            )
            expected_dimensions = sorted(
                (delay, horizon, reference, side)
                for delay in ROUND73_TARGET_V2_ENTRY_DELAYS_MS
                for horizon in ROUND73_TARGET_V2_HORIZONS_MS
                for reference in ROUND73_TARGET_V2_REFERENCE_NOTIONALS
                for side in ROUND73_TARGET_V2_SIDES
            )
            query = (
                "SELECT "
                + ", ".join(_V2_OPTION_COLUMNS)
                + f" FROM {ROUND73_TARGET_V2_OPTION_TABLE} "
                "WHERE study_id = ? AND run_id = ? "
                "ORDER BY symbol, anchor_index, entry_delay_ms, horizon_ms, "
                "reference_quote_notional, side"
            )
            cursor = connection.cursor()
            cursor.execute(query, [selected_study, selected_run])
            options: list[Round73CohortTargetOption] = []
            base_options: list[Round73TargetOption] = []
            observed_dimensions: list[tuple[int, int, float, str]] = []
            prior_anchor: tuple[str, int] | None = None
            try:
                while batch := cursor.fetchmany(_FETCH_BATCH_SIZE):
                    for raw in batch:
                        option = _cohort_option_from_row(raw)
                        base = Round73TargetOption(
                            **{
                                name: getattr(option, name)
                                for name in _BASE_OPTION_COLUMNS
                            }
                        )
                        anchor_key = (option.symbol, option.anchor_index)
                        expected_anchor_hash = anchor_hashes.get(anchor_key)
                        expected_cohort_hash = _sha256_text(
                            _canonical_json(list(option.values_without_cohort_hash()))
                        )
                        invariant_errors = (
                            *round73_target_option_invariant_errors(
                                base,
                                entry_delays_ms=ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
                                horizons_ms=ROUND73_TARGET_V2_HORIZONS_MS,
                                reference_notionals=(
                                    ROUND73_TARGET_V2_REFERENCE_NOTIONALS
                                ),
                                sides=ROUND73_TARGET_V2_SIDES,
                            ),
                            *round73_target_quantity_invariant_errors(
                                base, quantity_rules[option.symbol]
                            ),
                        )
                        if (
                            option.study_id != selected_study
                            or option.run_id != selected_run
                            or expected_anchor_hash is None
                            or option.selected_anchor_sha256 != expected_anchor_hash
                            or option.cohort_option_sha256 != expected_cohort_hash
                            or invariant_errors
                        ):
                            raise ValueError(
                                "Round 73 v2 target row differs: "
                                f"{option.symbol}:{option.anchor_index}:"
                                + ",".join(invariant_errors)
                            )
                        current_anchor = (option.symbol, option.anchor_index)
                        if prior_anchor is not None and current_anchor != prior_anchor:
                            if observed_dimensions != expected_dimensions:
                                raise ValueError(
                                    "Round 73 v2 target anchor dimensions differ"
                                )
                            observed_dimensions = []
                        prior_anchor = current_anchor
                        observed_dimensions.append(
                            (
                                option.entry_delay_ms,
                                option.horizon_ms,
                                option.reference_quote_notional,
                                option.side,
                            )
                        )
                        options.append(option)
                        base_options.append(base)
                if (
                    prior_anchor is not None
                    and observed_dimensions != expected_dimensions
                ):
                    raise ValueError(
                        "Round 73 v2 target final anchor dimensions differ"
                    )
            finally:
                cursor.close()
            option_count = len(options)
            eligible_count = sum(item.eligible for item in options)
            positive_count = sum(
                item.eligible and bool(item.positive_net_payoff) for item in options
            )
            recomputed = _build_run_identity(
                context=context,
                source=source,
                selected_anchors=selected_anchors,
                options=options,
            )
            expected_option_count = (
                selected_anchor_count * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
            )
            if deep_replay:
                replayed = (
                    []
                    if not selected_anchors
                    else replay_function(
                        connection,
                        run_id=selected_run,
                        anchors=_replay_anchor_map(selected_anchors),
                        quantity_rules=quantity_rules,
                        run_started_wall_ns=started_wall_ns,
                        run_started_monotonic_ns=started_monotonic_ns,
                        coverage_start_wall_ns=source.coverage_start_wall_ns,
                        coverage_end_wall_ns=source.coverage_end_wall_ns,
                        segments=segments,
                        entry_delays_ms=ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
                        horizons_ms=ROUND73_TARGET_V2_HORIZONS_MS,
                        reference_notionals=ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
                        sides=ROUND73_TARGET_V2_SIDES,
                    )
                )
                if replayed != base_options:
                    raise ValueError("Round 73 v2 exact-wire replay differs")
                deep_replay_performed = True
            if (
                option_count != expected_option_count
                or recomputed != identity
                or str(row[7]) != identity.get("option_rows_sha256")
                or int(row[8]) != selected_anchor_count
                or int(row[9]) != option_count
                or int(row[10]) != eligible_count
                or int(row[11]) != positive_count
                or row[12] != identity.get("first_decision_wall_ns")
                or row[13] != identity.get("last_decision_wall_ns")
            ):
                raise ValueError("Round 73 v2 target aggregate differs")
    except (
        ArithmeticError,
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"target:{type(exc).__name__}:{exc}")
    return Round73TargetV2Audit(
        study_id=selected_study,
        run_id=selected_run,
        passed=not errors,
        errors=tuple(errors),
        selected_anchor_count=selected_anchor_count,
        option_count=option_count,
        eligible_option_count=eligible_count,
        positive_option_count=positive_count,
        target_manifest_sha256=manifest_sha,
        deep_replay_performed=deep_replay_performed,
    )


def _target_run_manifest_summaries(
    connection: duckdb.DuckDBPyConnection,
    *,
    context: _StudyContext,
) -> list[dict[str, object]]:
    rows = connection.execute(
        f"""
        SELECT run_id, target_manifest_sha256, selected_anchor_count,
               option_count, eligible_option_count, positive_option_count,
               option_rows_sha256
        FROM {ROUND73_TARGET_V2_MANIFEST_TABLE}
        WHERE study_id = ? ORDER BY run_id
        """,
        [context.study_id],
    ).fetchall()
    expected_run_ids = sorted(source.run_id for source in context.sources)
    if [str(row[0]) for row in rows] != expected_run_ids:
        raise ValueError("Round 73 target study run-manifest coverage differs")
    orphan_option_count = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM {ROUND73_TARGET_V2_OPTION_TABLE} o
            LEFT JOIN {ROUND73_TARGET_V2_MANIFEST_TABLE} m
              ON m.study_id = o.study_id AND m.run_id = o.run_id
            WHERE o.study_id = ? AND m.run_id IS NULL
            """,
            [context.study_id],
        ).fetchone()[0]
    )
    if orphan_option_count:
        raise ValueError("Round 73 target study contains orphan option rows")
    summaries: list[dict[str, object]] = []
    for row in rows:
        if (
            _SHA256.fullmatch(str(row[1])) is None
            or _SHA256.fullmatch(str(row[6])) is None
        ):
            raise ValueError("Round 73 target study run-manifest hash is invalid")
        summaries.append(
            {
                "run_id": str(row[0]),
                "target_manifest_sha256": str(row[1]),
                "selected_anchor_count": int(row[2]),
                "option_count": int(row[3]),
                "eligible_option_count": int(row[4]),
                "positive_option_count": int(row[5]),
                "option_rows_sha256": str(row[6]),
            }
        )
    return summaries


def _build_study_identity(
    *,
    context: _StudyContext,
    run_manifests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifests = [dict(item) for item in run_manifests]
    run_manifests_sha = _sha256_text(_canonical_json(manifests))
    selected_anchor_count = sum(
        int(item["selected_anchor_count"]) for item in manifests
    )
    option_count = sum(int(item["option_count"]) for item in manifests)
    eligible_count = sum(int(item["eligible_option_count"]) for item in manifests)
    positive_count = sum(int(item["positive_option_count"]) for item in manifests)
    if (
        len(manifests) != len(context.sources)
        or selected_anchor_count != len(context.anchors)
        or option_count
        != selected_anchor_count * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
    ):
        raise ValueError("Round 73 target study aggregate differs")
    return {
        "schema_version": ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION,
        "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
        "study_id": context.study_id,
        "cohort_manifest_sha256": context.study_manifest_sha256,
        "source_run_count": len(context.sources),
        "selected_anchor_count": selected_anchor_count,
        "option_count": option_count,
        "eligible_option_count": eligible_count,
        "ineligible_option_count": option_count - eligible_count,
        "positive_option_count": positive_count,
        "target_run_manifests": manifests,
        "target_run_manifests_sha256": run_manifests_sha,
        "all_source_runs_present": True,
        "fresh_cohort_and_source_audit_required": True,
        "crypto_formal_daily_close": False,
        "leverage_applied": False,
        "target_study_sealed": True,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }


def _study_report_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73TargetStudyReport:
    return Round73TargetStudyReport(
        study_id=str(identity["study_id"]),
        source_run_count=int(identity["source_run_count"]),
        selected_anchor_count=int(identity["selected_anchor_count"]),
        option_count=int(identity["option_count"]),
        eligible_option_count=int(identity["eligible_option_count"]),
        positive_option_count=int(identity["positive_option_count"]),
        target_run_manifests_sha256=str(identity["target_run_manifests_sha256"]),
        target_study_manifest_sha256=manifest_sha256,
    )


def seal_round73_target_study(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73TargetStudyReport:
    """Seal a complete target study only after fresh cohort and run audits."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        if any(
            anchor.anchor_wall_ns >= ROUND73_STUDY_NOT_BEFORE_WALL_NS
            for anchor in context.anchors
        ):
            raise ValueError(
                "Round 73 eligible targets require the staged v3 holdout store"
            )
        if _table_exists(connection, ROUND73_TARGET_V2_STUDY_TABLE):
            existing = connection.execute(
                f"SELECT target_study_manifest_json, target_study_manifest_sha256 "
                f"FROM {ROUND73_TARGET_V2_STUDY_TABLE} WHERE study_id = ?",
                [selected_study],
            ).fetchone()
        else:
            existing = None
    if existing is not None:
        audit = audit_round73_target_study(
            database,
            study_id=selected_study,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing target-study audit failed")
        identity = _strict_json_object(
            str(existing[0]), "Round 73 target-study manifest"
        )
        return _study_report_from_identity(identity, str(existing[1]))
    cohort_audit = cohort_audit_function(
        database,
        study_id=selected_study,
        deep_source_audit=True,
        memory_limit=memory_limit,
        threads=threads,
    )
    if getattr(cohort_audit, "passed", False) is not True:
        raise ValueError("Round 73 target study cohort/source audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        if not _table_exists(connection, ROUND73_TARGET_V2_MANIFEST_TABLE):
            raise ValueError("Round 73 target study run manifests are missing")
        source_run_ids = tuple(source.run_id for source in context.sources)
    for source_run_id in source_run_ids:
        audit = audit_round73_selected_anchor_targets(
            database,
            study_id=selected_study,
            run_id=source_run_id,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
            verify_cohort=False,
            deep_source_audit=False,
            deep_replay=True,
        )
        if not audit.passed:
            raise ValueError(
                f"Round 73 target run audit failed before seal: {source_run_id}"
            )
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        run_manifests = _target_run_manifest_summaries(
            connection,
            context=context,
        )
    identity = _build_study_identity(
        context=context,
        run_manifests=run_manifests,
    )
    manifest_text = _canonical_json(identity)
    manifest_sha = _sha256_text(manifest_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_tables(connection)
        _assert_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            current_context = _load_study_context(
                connection,
                study_id=selected_study,
            )
            current_run_manifests = _target_run_manifest_summaries(
                connection,
                context=current_context,
            )
            if current_context != context or current_run_manifests != run_manifests:
                raise ValueError("Round 73 target study changed before sealing")
            concurrent = connection.execute(
                f"SELECT target_study_manifest_sha256 "
                f"FROM {ROUND73_TARGET_V2_STUDY_TABLE} WHERE study_id = ?",
                [selected_study],
            ).fetchone()
            if concurrent is not None:
                if str(concurrent[0]) != manifest_sha:
                    raise ValueError("Round 73 concurrent target-study seal differs")
            else:
                connection.execute(
                    f"INSERT INTO {ROUND73_TARGET_V2_STUDY_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected_study,
                        ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION,
                        ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                        context.study_manifest_sha256,
                        manifest_text,
                        manifest_sha,
                        str(identity["target_run_manifests_sha256"]),
                        int(identity["source_run_count"]),
                        int(identity["selected_anchor_count"]),
                        int(identity["option_count"]),
                        int(identity["eligible_option_count"]),
                        int(identity["positive_option_count"]),
                        time.time_ns(),
                    ],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _study_report_from_identity(identity, manifest_sha)


def audit_round73_target_study(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73TargetStudyAudit:
    """Audit a sealed study and independently replay every target run."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    errors: list[str] = []
    source_run_count = 0
    audited_run_count = 0
    option_count = 0
    study_manifest_sha = ""
    try:
        cohort_audit = cohort_audit_function(
            database,
            study_id=selected_study,
            deep_source_audit=True,
            memory_limit=memory_limit,
            threads=threads,
        )
        if getattr(cohort_audit, "passed", False) is not True:
            raise ValueError("cohort/source audit failed")
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            context = _load_study_context(connection, study_id=selected_study)
            source_run_count = len(context.sources)
            source_run_ids = tuple(source.run_id for source in context.sources)
        for source_run_id in source_run_ids:
            run_audit = audit_round73_selected_anchor_targets(
                database,
                study_id=selected_study,
                run_id=source_run_id,
                memory_limit=memory_limit,
                threads=threads,
                cohort_audit_function=cohort_audit_function,
                corpus_audit_function=corpus_audit_function,
                grid_audit_function=grid_audit_function,
                replay_function=replay_function,
                verify_cohort=False,
                deep_source_audit=False,
                deep_replay=True,
            )
            if not run_audit.passed:
                raise ValueError(f"target run audit failed: {source_run_id}")
            audited_run_count += 1
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            context = _load_study_context(connection, study_id=selected_study)
            if not all(
                _table_exists(connection, table)
                for table in (
                    ROUND73_TARGET_V2_OPTION_TABLE,
                    ROUND73_TARGET_V2_MANIFEST_TABLE,
                    ROUND73_TARGET_V2_STUDY_TABLE,
                )
            ):
                raise ValueError("target study table set is incomplete")
            _assert_table_shapes(connection)
            run_manifests = _target_run_manifest_summaries(
                connection,
                context=context,
            )
            recomputed = _build_study_identity(
                context=context,
                run_manifests=run_manifests,
            )
            row = connection.execute(
                f"""
                SELECT schema_version, contract_sha256,
                       cohort_manifest_sha256,
                       target_study_manifest_json,
                       target_study_manifest_sha256,
                       target_run_manifests_sha256, source_run_count,
                       selected_anchor_count, option_count,
                       eligible_option_count, positive_option_count
                FROM {ROUND73_TARGET_V2_STUDY_TABLE} WHERE study_id = ?
                """,
                [selected_study],
            ).fetchone()
            if row is None:
                raise ValueError("target study manifest was not found")
            manifest_text = str(row[3])
            study_manifest_sha = str(row[4])
            identity = _strict_json_object(
                manifest_text, "Round 73 target-study manifest"
            )
            option_count = int(row[8])
            if (
                str(row[0]) != ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION
                or str(row[1]) != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
                or str(row[2]) != context.study_manifest_sha256
                or _SHA256.fullmatch(study_manifest_sha) is None
                or _sha256_text(manifest_text) != study_manifest_sha
                or identity != recomputed
                or str(row[5]) != identity.get("target_run_manifests_sha256")
                or int(row[6]) != identity.get("source_run_count")
                or int(row[7]) != identity.get("selected_anchor_count")
                or option_count != identity.get("option_count")
                or int(row[9]) != identity.get("eligible_option_count")
                or int(row[10]) != identity.get("positive_option_count")
            ):
                raise ValueError("target study manifest aggregate differs")
    except (
        ArithmeticError,
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"study:{type(exc).__name__}:{exc}")
    return Round73TargetStudyAudit(
        study_id=selected_study,
        passed=not errors,
        errors=tuple(errors),
        source_run_count=source_run_count,
        audited_target_run_count=audited_run_count,
        option_count=option_count,
        target_study_manifest_sha256=study_manifest_sha,
    )


__all__ = [
    "ROUND73_TARGET_STUDY_V2_SCHEMA_VERSION",
    "ROUND73_TARGET_V2_ENTRY_DELAYS_MS",
    "ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR",
    "ROUND73_TARGET_V2_HORIZONS_MS",
    "ROUND73_TARGET_V2_MANIFEST_TABLE",
    "ROUND73_TARGET_V2_OPTION_TABLE",
    "ROUND73_TARGET_V2_REFERENCE_NOTIONALS",
    "ROUND73_TARGET_V2_SCHEMA_VERSION",
    "ROUND73_TARGET_V2_STUDY_TABLE",
    "Round73CohortTargetOption",
    "Round73TargetStudyAudit",
    "Round73TargetStudyReport",
    "Round73TargetV2Audit",
    "Round73TargetV2BuildReport",
    "audit_round73_selected_anchor_targets",
    "audit_round73_target_study",
    "build_round73_selected_anchor_targets",
    "seal_round73_target_study",
]
