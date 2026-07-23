"""Outcome-blind prospective shock cohorts for Round 73."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

import duckdb
import numpy as np

from .impact_absorption_capture import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
)
from .impact_absorption_corpus import (
    ROUND73_CORPUS_CONTRACT_SHA256,
    ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
    ROUND73_CORPUS_RUN_TABLE,
    ROUND73_CORPUS_SCHEMA_VERSION,
    audit_round73_corpus_manifest,
)
from .impact_absorption_grid import (
    ROUND73_GRID_CONTRACT_SHA256,
    ROUND73_GRID_SCHEMA_VERSION,
)
from .impact_absorption_grid_store import (
    ROUND73_GRID_ANCHOR_TABLE,
    ROUND73_GRID_MANIFEST_TABLE,
    ROUND73_GRID_VECTOR_TABLE,
    audit_round73_causal_grid,
)
from .impact_absorption_store import ImpactAbsorptionStore


ROUND73_COMPACT_TARGET_CONTRACT_SHA256 = (
    "3a6e4ea172bc12c76cd5f86888bd04f1e407ee9e593aaf1f2009b10db5095c92"
)
ROUND73_SHOCK_STUDY_SCHEMA_VERSION = "round-073-shock-study-v1"
ROUND73_SHOCK_STUDY_TABLE = "impact_shock_study_v1"
ROUND73_SHOCK_ANCHOR_TABLE = "impact_shock_anchor_v1"
ROUND73_TARGET_V1_MANIFEST_TABLE = "impact_target_run_manifest_v1"
ROUND73_TARGET_V2_MANIFEST_TABLE = "impact_target_run_manifest_v2"
ROUND73_STUDY_SEARCH_START = date(2026, 7, 24)
ROUND73_STUDY_NOT_BEFORE_WALL_NS = 1_784_851_200_000_000_000
ROUND73_STUDY_DAY_NS = 86_400_000_000_000
ROUND73_STUDY_REQUIRED_DAYS = 7
ROUND73_STUDY_TRAINING_DAYS = 4
ROUND73_STUDY_TUNING_DAYS = 1
ROUND73_STUDY_TEST_DAYS = 2
ROUND73_SHOCK_MINIMUM_RATIO = 3.0
ROUND73_SHOCK_QUANTILE = 0.99
ROUND73_SHOCK_MINIMUM_TAKER_SHARE = 0.7
ROUND73_SHOCK_REFRACTORY_NS = 15_000_000_000
ROUND73_ROLE_BOUNDARY_EMBARGO_NS = 302_000_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STUDY_ID = re.compile(r"[0-9a-f]{32}")
_FETCH_BATCH_SIZE = 4_096

CorpusAuditFunction = Callable[..., object]
GridAuditFunction = Callable[..., object]


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


def _strict_json_object(raw_text: str, label: str) -> Mapping[str, object]:
    value = json.loads(
        raw_text,
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


def _day_start_ns(selected: date) -> int:
    value = datetime.combine(selected, datetime.min.time(), tzinfo=UTC)
    return int(value.timestamp()) * 1_000_000_000


def _role(day_ordinal: int) -> str:
    if 1 <= day_ordinal <= ROUND73_STUDY_TRAINING_DAYS:
        return "training"
    if day_ordinal == ROUND73_STUDY_TRAINING_DAYS + 1:
        return "tuning"
    if day_ordinal <= ROUND73_STUDY_REQUIRED_DAYS:
        return "test"
    raise ValueError("Round 73 study day ordinal is invalid")


def _role_end_wall_ns(study_start_wall_ns: int, role: str) -> int:
    day_count = {
        "training": ROUND73_STUDY_TRAINING_DAYS,
        "tuning": ROUND73_STUDY_TRAINING_DAYS + ROUND73_STUDY_TUNING_DAYS,
        "test": ROUND73_STUDY_REQUIRED_DAYS,
    }
    try:
        return study_start_wall_ns + day_count[role] * ROUND73_STUDY_DAY_NS
    except KeyError as exc:
        raise ValueError("Round 73 study role is invalid") from exc


def nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    """Return the exact nearest-rank empirical percentile."""

    selected = tuple(float(value) for value in values)
    quantile = float(probability)
    if not selected or not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("Round 73 nearest-rank input is invalid")
    if any(not math.isfinite(value) for value in selected):
        raise ValueError("Round 73 nearest-rank values must be finite")
    ordered = sorted(selected)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Round73StudyDay:
    utc_day: str
    finalized: bool
    eligible: bool
    coverage_ns: int
    required_coverage_ns: int
    interval_count: int
    contributing_run_ids: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["contributing_run_ids"] = list(self.contributing_run_ids)
        return payload


@dataclass(frozen=True)
class Round73ShockThreshold:
    symbol: str
    training_observation_count: int
    nearest_rank: int
    empirical_ratio: float
    effective_ratio: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Round73ShockAnchor:
    study_id: str
    run_id: str
    symbol: str
    anchor_index: int
    anchor_monotonic_ns: int
    anchor_wall_ns: int
    source_max_received_monotonic_ns: int
    utc_day: str
    day_ordinal: int
    role: str
    shock_ratio: float
    shock_direction: int
    shock_direction_taker_share: float
    feature_vector_sha256: str
    selected_anchor_sha256: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.symbol, self.anchor_wall_ns, self.run_id)

    def values_without_hash(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())[:-1]

    def as_row(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


_ANCHOR_COLUMNS = tuple(Round73ShockAnchor.__dataclass_fields__)


@dataclass(frozen=True)
class Round73ShockStudyReport:
    study_id: str
    selected_utc_days: tuple[str, ...]
    source_run_count: int
    selected_anchor_count: int
    selected_anchor_counts: Mapping[str, int]
    thresholds: tuple[Round73ShockThreshold, ...]
    selected_anchor_rows_sha256: str
    study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected_utc_days"] = list(self.selected_utc_days)
        payload["selected_anchor_counts"] = dict(self.selected_anchor_counts)
        payload["thresholds"] = [item.as_dict() for item in self.thresholds]
        payload["schema_version"] = ROUND73_SHOCK_STUDY_SCHEMA_VERSION
        payload["contract_sha256"] = ROUND73_COMPACT_TARGET_CONTRACT_SHA256
        payload["target_observed"] = False
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


@dataclass(frozen=True)
class Round73ShockStudyAudit:
    study_id: str
    passed: bool
    errors: tuple[str, ...]
    source_run_count: int
    selected_anchor_count: int
    deeply_audited_source_count: int
    selected_anchor_rows_sha256: str
    study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        payload["schema_version"] = "round-073-shock-study-audit-v1"
        payload["contract_sha256"] = ROUND73_COMPACT_TARGET_CONTRACT_SHA256
        payload["target_observed"] = False
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


class Round73ShockCohortNotReady(ValueError):
    """Raised when seven consecutive finalized integrity-complete days do not exist."""

    def __init__(self, examined_days: Sequence[Round73StudyDay]) -> None:
        self.examined_days = tuple(examined_days)
        super().__init__(
            "Round 73 compact cohort is not ready: seven consecutive "
            "integrity-complete UTC days are unavailable"
        )


@dataclass(frozen=True)
class _SourceRun:
    run_id: str
    corpus_manifest_sha256: str
    grid_manifest_sha256: str
    coverage_start_wall_ns: int
    coverage_end_wall_ns: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _SelectedAnchorIdentity:
    run_id: str
    symbol: str
    anchor_index: int
    anchor_monotonic_ns: int
    anchor_wall_ns: int
    source_max_received_monotonic_ns: int
    utc_day: str
    day_ordinal: int
    role: str
    shock_ratio: float
    shock_direction: int
    shock_direction_taker_share: float
    feature_vector_sha256: str

    def values(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


def _merge_coverage(
    intervals: Sequence[tuple[int, int]],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    merged: list[tuple[int, int]] = []
    for start_ns, end_ns in sorted(intervals):
        if end_ns <= start_ns:
            continue
        if not merged or start_ns > merged[-1][1]:
            merged.append((start_ns, end_ns))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ns))
    return (
        sum(end_ns - start_ns for start_ns, end_ns in merged),
        tuple(merged),
    )


def _candidate_source_runs(
    connection: duckdb.DuckDBPyConnection,
    *,
    selection_as_of_wall_ns: int,
) -> tuple[_SourceRun, ...]:
    selection_cutoff = int(selection_as_of_wall_ns)
    if selection_cutoff < ROUND73_STUDY_NOT_BEFORE_WALL_NS:
        raise ValueError("Round 73 compact cohort selection cutoff is invalid")
    required_tables = (
        "impact_capture_run",
        ROUND73_CORPUS_RUN_TABLE,
        ROUND73_GRID_MANIFEST_TABLE,
        ROUND73_GRID_ANCHOR_TABLE,
        ROUND73_GRID_VECTOR_TABLE,
    )
    missing = [
        table for table in required_tables if not _table_exists(connection, table)
    ]
    if missing:
        raise ValueError(
            "Round 73 compact cohort source tables are missing: " + ",".join(missing)
        )
    rows = connection.execute(
        f"""
        SELECT c.run_id, c.manifest_sha256, g.build_manifest_sha256,
               c.coverage_start_wall_ns, c.coverage_end_wall_ns,
               c.schema_version, c.contract_sha256,
               r.schema_version, r.capture_contract_sha256,
               g.schema_version, g.contract_sha256,
               g.source_corpus_manifest_sha256
        FROM {ROUND73_CORPUS_RUN_TABLE} c
        JOIN impact_capture_run r ON r.run_id = c.run_id
        JOIN {ROUND73_GRID_MANIFEST_TABLE} g ON g.run_id = c.run_id
        WHERE c.coverage_end_wall_ns > ?
          AND c.recorded_at_wall_ns <= ?
          AND g.recorded_at_wall_ns <= ?
        ORDER BY c.coverage_start_wall_ns, c.run_id
        """,
        [
            ROUND73_STUDY_NOT_BEFORE_WALL_NS,
            selection_cutoff,
            selection_cutoff,
        ],
    ).fetchall()
    output: list[_SourceRun] = []
    for row in rows:
        corpus_sha = str(row[1])
        grid_sha = str(row[2])
        start_ns = int(row[3])
        end_ns = int(row[4])
        if (
            str(row[5]) != ROUND73_CORPUS_SCHEMA_VERSION
            or str(row[6]) != ROUND73_CORPUS_CONTRACT_SHA256
            or str(row[7]) != IMPACT_CAPTURE_V9_SCHEMA_VERSION
            or str(row[8]) != IMPACT_CAPTURE_V9_CONTRACT_SHA256
            or str(row[9]) != ROUND73_GRID_SCHEMA_VERSION
            or str(row[10]) != ROUND73_GRID_CONTRACT_SHA256
            or str(row[11]) != corpus_sha
            or _SHA256.fullmatch(corpus_sha) is None
            or _SHA256.fullmatch(grid_sha) is None
            or end_ns <= start_ns
        ):
            raise ValueError("Round 73 compact cohort source identity differs")
        output.append(
            _SourceRun(
                run_id=str(row[0]),
                corpus_manifest_sha256=corpus_sha,
                grid_manifest_sha256=grid_sha,
                coverage_start_wall_ns=start_ns,
                coverage_end_wall_ns=end_ns,
            )
        )
    return tuple(output)


def _study_days(
    source_runs: Sequence[_SourceRun],
    *,
    now_wall_ns: int,
) -> tuple[tuple[Round73StudyDay, ...], tuple[Round73StudyDay, ...]]:
    now_ns = int(now_wall_ns)
    if now_ns < 0:
        raise ValueError("Round 73 compact cohort current time is invalid")
    current_day = datetime.fromtimestamp(now_ns / 1_000_000_000, tz=UTC).date()
    last_finalized = current_day - timedelta(days=1)
    examined: list[Round73StudyDay] = []
    streak: list[Round73StudyDay] = []
    selected = ROUND73_STUDY_SEARCH_START
    while selected <= last_finalized:
        start_ns = _day_start_ns(selected)
        end_ns = start_ns + ROUND73_STUDY_DAY_NS
        intervals: list[tuple[int, int]] = []
        contributing: list[str] = []
        for run in source_runs:
            start = max(start_ns, run.coverage_start_wall_ns)
            end = min(end_ns, run.coverage_end_wall_ns)
            if end > start:
                intervals.append((start, end))
                contributing.append(run.run_id)
        coverage_ns, merged = _merge_coverage(intervals)
        eligible = coverage_ns >= ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS
        reason = "eligible" if eligible else "insufficient_integrity_coverage"
        day = Round73StudyDay(
            utc_day=selected.isoformat(),
            finalized=True,
            eligible=eligible,
            coverage_ns=coverage_ns,
            required_coverage_ns=ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
            interval_count=len(merged),
            contributing_run_ids=tuple(sorted(set(contributing))),
            reason=reason,
        )
        examined.append(day)
        if eligible:
            streak.append(day)
        else:
            streak.clear()
        if len(streak) == ROUND73_STUDY_REQUIRED_DAYS:
            return tuple(examined), tuple(streak)
        selected += timedelta(days=1)
    raise Round73ShockCohortNotReady(examined)


def _selected_source_runs(
    source_runs: Sequence[_SourceRun],
    days: Sequence[Round73StudyDay],
) -> tuple[_SourceRun, ...]:
    run_ids = {run_id for day in days for run_id in day.contributing_run_ids}
    selected = tuple(run for run in source_runs if run.run_id in run_ids)
    if len(selected) != len(run_ids):
        raise ValueError("Round 73 compact cohort contributing run is missing")
    study_start_ns = _day_start_ns(date.fromisoformat(days[0].utc_day))
    study_end_ns = study_start_ns + ROUND73_STUDY_REQUIRED_DAYS * ROUND73_STUDY_DAY_NS
    clipped = sorted(
        (
            max(run.coverage_start_wall_ns, study_start_ns),
            min(run.coverage_end_wall_ns, study_end_ns),
            run.run_id,
        )
        for run in selected
    )
    if any(
        right[0] < left[1] for left, right in zip(clipped, clipped[1:], strict=False)
    ):
        raise ValueError("Round 73 compact cohort source intervals overlap")
    return tuple(
        sorted(selected, key=lambda run: (run.coverage_start_wall_ns, run.run_id))
    )


def _reject_prelabeled_sources(
    connection: duckdb.DuckDBPyConnection,
    source_runs: Sequence[_SourceRun],
) -> None:
    run_ids = tuple(run.run_id for run in source_runs)
    if not run_ids:
        raise ValueError("Round 73 compact cohort has no source runs")
    placeholders = ",".join("?" for _ in run_ids)
    for table in (ROUND73_TARGET_V1_MANIFEST_TABLE, ROUND73_TARGET_V2_MANIFEST_TABLE):
        if not _table_exists(connection, table):
            continue
        count = int(
            connection.execute(
                f"SELECT count(*) FROM {table} WHERE run_id IN ({placeholders})",
                list(run_ids),
            ).fetchone()[0]
        )
        if count:
            raise ValueError(
                "Round 73 compact cohort source already has target outcomes: " + table
            )


def _thresholds(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_runs: Sequence[_SourceRun],
    study_start_wall_ns: int,
) -> tuple[Round73ShockThreshold, ...]:
    run_ids = tuple(run.run_id for run in source_runs)
    placeholders = ",".join("?" for _ in run_ids)
    training_end = (
        study_start_wall_ns + ROUND73_STUDY_TRAINING_DAYS * ROUND73_STUDY_DAY_NS
    )
    output: list[Round73ShockThreshold] = []
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        common: list[object] = [
            *run_ids,
            symbol,
            study_start_wall_ns,
            training_end,
        ]
        count = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM {ROUND73_GRID_ANCHOR_TABLE} a
                JOIN {ROUND73_GRID_VECTOR_TABLE} v
                  USING (run_id, symbol, anchor_index)
                WHERE a.run_id IN ({placeholders}) AND a.symbol = ? AND a.valid
                  AND a.anchor_wall_ns >= ? AND a.anchor_wall_ns < ?
                  AND a.shock_ratio IS NOT NULL AND isfinite(a.shock_ratio)
                """,
                common,
            ).fetchone()[0]
        )
        if count < 1:
            raise ValueError(
                f"Round 73 compact cohort training shock support is empty: {symbol}"
            )
        rank = max(1, math.ceil(ROUND73_SHOCK_QUANTILE * count))
        row = connection.execute(
            f"""
            SELECT a.shock_ratio
            FROM {ROUND73_GRID_ANCHOR_TABLE} a
            JOIN {ROUND73_GRID_VECTOR_TABLE} v
              USING (run_id, symbol, anchor_index)
            WHERE a.run_id IN ({placeholders}) AND a.symbol = ? AND a.valid
              AND a.anchor_wall_ns >= ? AND a.anchor_wall_ns < ?
              AND a.shock_ratio IS NOT NULL AND isfinite(a.shock_ratio)
            ORDER BY a.shock_ratio, a.run_id, a.anchor_index
            LIMIT 1 OFFSET ?
            """,
            [*common, rank - 1],
        ).fetchone()
        if row is None or not math.isfinite(float(row[0])):
            raise ValueError("Round 73 compact cohort threshold is invalid")
        empirical = float(row[0])
        output.append(
            Round73ShockThreshold(
                symbol=symbol,
                training_observation_count=count,
                nearest_rank=rank,
                empirical_ratio=empirical,
                effective_ratio=max(ROUND73_SHOCK_MINIMUM_RATIO, empirical),
            )
        )
    return tuple(output)


def _selected_anchor_identities(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_runs: Sequence[_SourceRun],
    selected_days: Sequence[Round73StudyDay],
    thresholds: Sequence[Round73ShockThreshold],
) -> tuple[_SelectedAnchorIdentity, ...]:
    run_ids = tuple(run.run_id for run in source_runs)
    placeholders = ",".join("?" for _ in run_ids)
    threshold_map = {item.symbol: item.effective_ratio for item in thresholds}
    if tuple(sorted(threshold_map)) != tuple(sorted(IMPACT_CAPTURE_SYMBOLS)):
        raise ValueError("Round 73 compact cohort thresholds are incomplete")
    study_start = _day_start_ns(date.fromisoformat(selected_days[0].utc_day))
    study_end = study_start + ROUND73_STUDY_REQUIRED_DAYS * ROUND73_STUDY_DAY_NS
    threshold_predicate = " OR ".join(
        "(a.symbol = ? AND a.shock_ratio >= ?)" for _ in IMPACT_CAPTURE_SYMBOLS
    )
    threshold_parameters: list[object] = []
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        threshold_parameters.extend((symbol, threshold_map[symbol]))
    cursor = connection.cursor()
    cursor.execute(
        f"""
        SELECT a.run_id, a.symbol, a.anchor_index, a.anchor_monotonic_ns,
               a.anchor_wall_ns, a.source_max_received_monotonic_ns,
               a.shock_ratio, a.shock_direction,
               a.shock_direction_taker_share, v.vector_sha256
        FROM {ROUND73_GRID_ANCHOR_TABLE} a
        JOIN {ROUND73_GRID_VECTOR_TABLE} v
          USING (run_id, symbol, anchor_index)
        WHERE a.run_id IN ({placeholders}) AND a.valid
          AND a.anchor_wall_ns >= ? AND a.anchor_wall_ns < ?
          AND a.shock_ratio IS NOT NULL AND isfinite(a.shock_ratio)
          AND a.shock_direction IN (-1, 1)
          AND a.shock_direction_taker_share >= ?
          AND ({threshold_predicate})
        ORDER BY a.symbol, a.anchor_wall_ns, a.run_id, a.anchor_index
        """,
        [
            *run_ids,
            study_start,
            study_end,
            ROUND73_SHOCK_MINIMUM_TAKER_SHARE,
            *threshold_parameters,
        ],
    )
    output: list[_SelectedAnchorIdentity] = []
    last_selected_wall_ns: dict[str, int] = {}
    try:
        while rows := cursor.fetchmany(_FETCH_BATCH_SIZE):
            for row in rows:
                run_id = str(row[0])
                symbol = str(row[1])
                anchor_index = int(row[2])
                anchor_mono = int(row[3])
                anchor_wall = int(row[4])
                source_max = int(row[5])
                shock_ratio = float(row[6])
                direction = int(row[7])
                direction_share = float(row[8])
                vector_sha = str(row[9])
                if (
                    symbol not in threshold_map
                    or not math.isfinite(shock_ratio)
                    or shock_ratio < threshold_map[symbol]
                    or direction not in {-1, 1}
                    or not math.isfinite(direction_share)
                    or direction_share < ROUND73_SHOCK_MINIMUM_TAKER_SHARE
                    or source_max >= anchor_mono
                    or _SHA256.fullmatch(vector_sha) is None
                ):
                    raise ValueError("Round 73 compact cohort candidate differs")
                day_index = (anchor_wall - study_start) // ROUND73_STUDY_DAY_NS
                day_ordinal = int(day_index) + 1
                if not 1 <= day_ordinal <= ROUND73_STUDY_REQUIRED_DAYS:
                    raise ValueError("Round 73 compact cohort anchor day differs")
                role = _role(day_ordinal)
                if anchor_wall + ROUND73_ROLE_BOUNDARY_EMBARGO_NS >= _role_end_wall_ns(
                    study_start, role
                ):
                    continue
                previous = last_selected_wall_ns.get(symbol)
                if (
                    previous is not None
                    and anchor_wall - previous < ROUND73_SHOCK_REFRACTORY_NS
                ):
                    continue
                last_selected_wall_ns[symbol] = anchor_wall
                output.append(
                    _SelectedAnchorIdentity(
                        run_id=run_id,
                        symbol=symbol,
                        anchor_index=anchor_index,
                        anchor_monotonic_ns=anchor_mono,
                        anchor_wall_ns=anchor_wall,
                        source_max_received_monotonic_ns=source_max,
                        utc_day=selected_days[day_ordinal - 1].utc_day,
                        day_ordinal=day_ordinal,
                        role=role,
                        shock_ratio=shock_ratio,
                        shock_direction=direction,
                        shock_direction_taker_share=direction_share,
                        feature_vector_sha256=vector_sha,
                    )
                )
    finally:
        cursor.close()
    output.sort(
        key=lambda item: (
            item.symbol,
            item.anchor_wall_ns,
            item.run_id,
            item.anchor_index,
        )
    )
    keys = [(item.run_id, item.symbol, item.anchor_index) for item in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Round 73 compact cohort selected anchors are duplicated")
    return tuple(output)


def _stream_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _anchor_identity_sha256(anchor: _SelectedAnchorIdentity) -> str:
    return _sha256_text(_canonical_json(list(anchor.values())))


def _make_anchor(
    study_id: str, identity: _SelectedAnchorIdentity
) -> Round73ShockAnchor:
    values = (study_id, *identity.values())
    return Round73ShockAnchor(
        study_id=study_id,
        **asdict(identity),
        selected_anchor_sha256=_sha256_text(_canonical_json(list(values))),
    )


def _source_runs_sha256(source_runs: Sequence[_SourceRun]) -> str:
    return _sha256_text(_canonical_json([item.as_dict() for item in source_runs]))


def _selected_identity_rows_sha256(
    identities: Sequence[_SelectedAnchorIdentity],
) -> str:
    return _stream_hash([_anchor_identity_sha256(item) for item in identities])


def _selected_anchor_rows_sha256(anchors: Sequence[Round73ShockAnchor]) -> str:
    return _stream_hash([item.selected_anchor_sha256 for item in anchors])


def _selected_counts(
    anchors: Sequence[Round73ShockAnchor],
) -> dict[str, int]:
    return {
        symbol: sum(anchor.symbol == symbol for anchor in anchors)
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }


def _study_seed(
    *,
    selected_days: Sequence[Round73StudyDay],
    source_runs: Sequence[_SourceRun],
    thresholds: Sequence[Round73ShockThreshold],
    selected_identity_rows_sha256: str,
) -> dict[str, object]:
    return {
        "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
        "selected_utc_days": [day.utc_day for day in selected_days],
        "source_runs_sha256": _source_runs_sha256(source_runs),
        "thresholds": [item.as_dict() for item in thresholds],
        "selected_identity_rows_sha256": selected_identity_rows_sha256,
    }


def _manifest_identity(
    *,
    study_id: str,
    examined_days: Sequence[Round73StudyDay],
    selected_days: Sequence[Round73StudyDay],
    source_runs: Sequence[_SourceRun],
    thresholds: Sequence[Round73ShockThreshold],
    anchors: Sequence[Round73ShockAnchor],
    selected_identity_rows_sha256: str,
    selection_as_of_wall_ns: int,
) -> dict[str, object]:
    counts = _selected_counts(anchors)
    return {
        "schema_version": ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
        "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
        "capture_schema_version": IMPACT_CAPTURE_V9_SCHEMA_VERSION,
        "capture_contract_sha256": IMPACT_CAPTURE_V9_CONTRACT_SHA256,
        "corpus_contract_sha256": ROUND73_CORPUS_CONTRACT_SHA256,
        "grid_schema_version": ROUND73_GRID_SCHEMA_VERSION,
        "grid_contract_sha256": ROUND73_GRID_CONTRACT_SHA256,
        "study_id": study_id,
        "selection_as_of_wall_ns": int(selection_as_of_wall_ns),
        "eligible_anchor_wall_ns_not_before": ROUND73_STUDY_NOT_BEFORE_WALL_NS,
        "examined_days": [day.as_dict() for day in examined_days],
        "selected_utc_days": [day.utc_day for day in selected_days],
        "roles": {
            "training": [day.utc_day for day in selected_days[:4]],
            "tuning": [selected_days[4].utc_day],
            "test": [day.utc_day for day in selected_days[5:]],
        },
        "source_runs": [run.as_dict() for run in source_runs],
        "source_runs_sha256": _source_runs_sha256(source_runs),
        "source_run_count": len(source_runs),
        "thresholds": [item.as_dict() for item in thresholds],
        "threshold_source_role": "training",
        "selected_identity_rows_sha256": selected_identity_rows_sha256,
        "selected_anchor_rows_sha256": _selected_anchor_rows_sha256(anchors),
        "selected_anchor_count": len(anchors),
        "selected_anchor_counts": counts,
        "target_manifest_count_at_publish": 0,
        "target_observed": False,
        "model_evaluated": False,
        "predictive_edge_claim": False,
        "profitability_claim": False,
        "crypto_formal_daily_close": False,
        "trading_authority": False,
    }


def _report_from_manifest(
    identity: Mapping[str, object], manifest_sha256: str
) -> Round73ShockStudyReport:
    thresholds = tuple(
        Round73ShockThreshold(
            symbol=str(item["symbol"]),
            training_observation_count=int(item["training_observation_count"]),
            nearest_rank=int(item["nearest_rank"]),
            empirical_ratio=float(item["empirical_ratio"]),
            effective_ratio=float(item["effective_ratio"]),
        )
        for item in identity["thresholds"]  # type: ignore[union-attr]
    )
    return Round73ShockStudyReport(
        study_id=str(identity["study_id"]),
        selected_utc_days=tuple(
            str(value)
            for value in identity["selected_utc_days"]  # type: ignore[union-attr]
        ),
        source_run_count=int(identity["source_run_count"]),
        selected_anchor_count=int(identity["selected_anchor_count"]),
        selected_anchor_counts={
            str(key): int(value)
            for key, value in dict(identity["selected_anchor_counts"]).items()  # type: ignore[arg-type]
        },
        thresholds=thresholds,
        selected_anchor_rows_sha256=str(identity["selected_anchor_rows_sha256"]),
        study_manifest_sha256=manifest_sha256,
    )


def _create_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_SHOCK_ANCHOR_TABLE} (
            study_id VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            anchor_monotonic_ns UBIGINT NOT NULL,
            anchor_wall_ns UBIGINT NOT NULL,
            source_max_received_monotonic_ns UBIGINT NOT NULL,
            utc_day VARCHAR NOT NULL,
            day_ordinal UTINYINT NOT NULL,
            role VARCHAR NOT NULL,
            shock_ratio DOUBLE NOT NULL,
            shock_direction TINYINT NOT NULL,
            shock_direction_taker_share DOUBLE NOT NULL,
            feature_vector_sha256 VARCHAR NOT NULL,
            selected_anchor_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (study_id, run_id, symbol, anchor_index),
            CHECK (length(study_id) = 32),
            CHECK (length(feature_vector_sha256) = 64),
            CHECK (length(selected_anchor_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_SHOCK_STUDY_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            manifest_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            source_runs_sha256 VARCHAR NOT NULL,
            selected_identity_rows_sha256 VARCHAR NOT NULL,
            selected_anchor_rows_sha256 VARCHAR NOT NULL,
            source_run_count USMALLINT NOT NULL,
            selected_anchor_count UINTEGER NOT NULL,
            study_start_wall_ns UBIGINT NOT NULL,
            study_end_wall_ns UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(study_id) = 32),
            CHECK (length(contract_sha256) = 64),
            CHECK (length(manifest_sha256) = 64),
            CHECK (length(source_runs_sha256) = 64),
            CHECK (length(selected_identity_rows_sha256) = 64),
            CHECK (length(selected_anchor_rows_sha256) = 64)
        )
        """
    )


def _assert_table_shapes(connection: duckdb.DuckDBPyConnection) -> None:
    expected = {
        ROUND73_SHOCK_ANCHOR_TABLE: _ANCHOR_COLUMNS,
        ROUND73_SHOCK_STUDY_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "manifest_json",
            "manifest_sha256",
            "source_runs_sha256",
            "selected_identity_rows_sha256",
            "selected_anchor_rows_sha256",
            "source_run_count",
            "selected_anchor_count",
            "study_start_wall_ns",
            "study_end_wall_ns",
            "recorded_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 compact cohort table schema differs: {table}")


def _insert_anchors_columnar(
    connection: duckdb.DuckDBPyConnection,
    anchors: Sequence[Round73ShockAnchor],
) -> None:
    if not anchors:
        return
    rows = [anchor.as_row() for anchor in anchors]
    views: list[str] = []
    projections: list[str] = []
    try:
        for index, column in enumerate(zip(*rows, strict=True)):
            values = tuple(column)
            view = f"_round73_shock_anchor_{index}"
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                array = np.asarray(values, dtype=np.int64)
            elif all(isinstance(value, float) for value in values):
                array = np.asarray(values, dtype=np.float64)
            elif all(isinstance(value, str) for value in values):
                array = np.asarray(values, dtype=np.str_)
            else:
                raise TypeError("Round 73 compact cohort column type is unsupported")
            connection.register(view, array)
            views.append(view)
            projections.append(f"{view}.column0")
        connection.execute(
            f"INSERT INTO {ROUND73_SHOCK_ANCHOR_TABLE} SELECT "
            + ", ".join(projections)
            + " FROM "
            + " POSITIONAL JOIN ".join(views)
        )
    finally:
        for view in views:
            connection.unregister(view)


def _audit_sources(
    database: str | Path,
    *,
    source_runs: Sequence[_SourceRun],
    memory_limit: str,
    threads: int,
    corpus_audit_function: CorpusAuditFunction,
    grid_audit_function: GridAuditFunction,
) -> int:
    audited = 0
    for source in source_runs:
        corpus = corpus_audit_function(
            database,
            run_id=source.run_id,
            memory_limit=memory_limit,
            threads=threads,
        )
        grid = grid_audit_function(
            database,
            run_id=source.run_id,
            memory_limit=memory_limit,
            threads=threads,
        )
        if (
            getattr(corpus, "passed", False) is not True
            or getattr(corpus, "manifest_sha256", "") != source.corpus_manifest_sha256
            or getattr(grid, "passed", False) is not True
            or getattr(grid, "build_manifest_sha256", "") != source.grid_manifest_sha256
        ):
            raise ValueError(
                f"Round 73 compact cohort source audit failed: {source.run_id}"
            )
        audited += 1
    return audited


def build_round73_shock_cohort(
    database: str | Path,
    *,
    now_wall_ns: int | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    corpus_audit_function: CorpusAuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: GridAuditFunction = audit_round73_causal_grid,
) -> Round73ShockStudyReport:
    """Freeze the first eligible seven-day feature-only shock cohort."""

    current_ns = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        source_candidates = _candidate_source_runs(
            connection,
            selection_as_of_wall_ns=current_ns,
        )
        examined_days, selected_days = _study_days(
            source_candidates,
            now_wall_ns=current_ns,
        )
        source_runs = _selected_source_runs(source_candidates, selected_days)
        _reject_prelabeled_sources(connection, source_runs)
    _audit_sources(
        database,
        source_runs=source_runs,
        memory_limit=memory_limit,
        threads=threads,
        corpus_audit_function=corpus_audit_function,
        grid_audit_function=grid_audit_function,
    )
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _reject_prelabeled_sources(connection, source_runs)
        study_start = _day_start_ns(date.fromisoformat(selected_days[0].utc_day))
        thresholds = _thresholds(
            connection,
            source_runs=source_runs,
            study_start_wall_ns=study_start,
        )
        selected_identities = _selected_anchor_identities(
            connection,
            source_runs=source_runs,
            selected_days=selected_days,
            thresholds=thresholds,
        )
    selected_identity_sha = _selected_identity_rows_sha256(selected_identities)
    seed = _study_seed(
        selected_days=selected_days,
        source_runs=source_runs,
        thresholds=thresholds,
        selected_identity_rows_sha256=selected_identity_sha,
    )
    study_id = _sha256_text(_canonical_json(seed))[:32]
    anchors = tuple(_make_anchor(study_id, item) for item in selected_identities)
    identity = _manifest_identity(
        study_id=study_id,
        examined_days=examined_days,
        selected_days=selected_days,
        source_runs=source_runs,
        thresholds=thresholds,
        anchors=anchors,
        selected_identity_rows_sha256=selected_identity_sha,
        selection_as_of_wall_ns=current_ns,
    )
    manifest_text = _canonical_json(identity)
    manifest_sha = _sha256_text(manifest_text)
    study_end = study_start + ROUND73_STUDY_REQUIRED_DAYS * ROUND73_STUDY_DAY_NS
    existing_exact = False
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_tables(connection)
        _assert_table_shapes(connection)
        existing = connection.execute(
            f"SELECT study_id, manifest_sha256 FROM {ROUND73_SHOCK_STUDY_TABLE} "
            "WHERE contract_sha256 = ?",
            [ROUND73_COMPACT_TARGET_CONTRACT_SHA256],
        ).fetchall()
        if existing:
            if existing != [(study_id, manifest_sha)]:
                raise ValueError("Round 73 compact cohort already differs")
            existing_exact = True
        else:
            connection.execute("BEGIN TRANSACTION")
            try:
                _reject_prelabeled_sources(connection, source_runs)
                _insert_anchors_columnar(connection, anchors)
                connection.execute(
                    f"INSERT INTO {ROUND73_SHOCK_STUDY_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        study_id,
                        ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
                        ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                        manifest_text,
                        manifest_sha,
                        str(identity["source_runs_sha256"]),
                        selected_identity_sha,
                        str(identity["selected_anchor_rows_sha256"]),
                        len(source_runs),
                        len(anchors),
                        study_start,
                        study_end,
                        time.time_ns(),
                    ],
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
    if existing_exact:
        audit = audit_round73_shock_cohort(
            database,
            study_id=study_id,
            deep_source_audit=False,
            memory_limit=memory_limit,
            threads=threads,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing compact cohort audit failed")
    return _report_from_manifest(identity, manifest_sha)


def _source_runs_from_manifest(
    identity: Mapping[str, object],
) -> tuple[_SourceRun, ...]:
    raw = identity.get("source_runs")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("Round 73 compact cohort source runs are missing")
    output: list[_SourceRun] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Round 73 compact cohort source run is invalid")
        output.append(
            _SourceRun(
                run_id=str(item["run_id"]),
                corpus_manifest_sha256=str(item["corpus_manifest_sha256"]),
                grid_manifest_sha256=str(item["grid_manifest_sha256"]),
                coverage_start_wall_ns=int(item["coverage_start_wall_ns"]),
                coverage_end_wall_ns=int(item["coverage_end_wall_ns"]),
            )
        )
    return tuple(output)


def _days_from_manifest(identity: Mapping[str, object]) -> tuple[Round73StudyDay, ...]:
    selected = identity.get("selected_utc_days")
    if not isinstance(selected, Sequence) or isinstance(
        selected, (str, bytes, bytearray)
    ):
        raise ValueError("Round 73 compact cohort selected days are missing")
    if len(selected) != ROUND73_STUDY_REQUIRED_DAYS:
        raise ValueError("Round 73 compact cohort selected-day count differs")
    parsed = tuple(date.fromisoformat(str(value)) for value in selected)
    if parsed[0] < ROUND73_STUDY_SEARCH_START or any(
        following != previous + timedelta(days=1)
        for previous, following in zip(parsed, parsed[1:], strict=False)
    ):
        raise ValueError("Round 73 compact cohort selected days are not consecutive")
    return tuple(
        Round73StudyDay(
            utc_day=value.isoformat(),
            finalized=True,
            eligible=True,
            coverage_ns=ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
            required_coverage_ns=ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
            interval_count=0,
            contributing_run_ids=(),
            reason="eligible",
        )
        for value in parsed
    )


def _thresholds_from_manifest(
    identity: Mapping[str, object],
) -> tuple[Round73ShockThreshold, ...]:
    raw = identity.get("thresholds")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("Round 73 compact cohort thresholds are missing")
    return tuple(
        Round73ShockThreshold(
            symbol=str(item["symbol"]),
            training_observation_count=int(item["training_observation_count"]),
            nearest_rank=int(item["nearest_rank"]),
            empirical_ratio=float(item["empirical_ratio"]),
            effective_ratio=float(item["effective_ratio"]),
        )
        for item in raw
        if isinstance(item, Mapping)
    )


def _anchor_from_row(row: Sequence[object]) -> Round73ShockAnchor:
    if len(row) != len(_ANCHOR_COLUMNS):
        raise ValueError("Round 73 compact cohort anchor width differs")
    values = dict(zip(_ANCHOR_COLUMNS, row, strict=True))
    for field in (
        "anchor_index",
        "anchor_monotonic_ns",
        "anchor_wall_ns",
        "source_max_received_monotonic_ns",
        "day_ordinal",
        "shock_direction",
    ):
        values[field] = int(values[field])
    for field in ("shock_ratio", "shock_direction_taker_share"):
        values[field] = float(values[field])
    return Round73ShockAnchor(**values)  # type: ignore[arg-type]


def audit_round73_shock_cohort(
    database: str | Path,
    *,
    study_id: str,
    deep_source_audit: bool = True,
    memory_limit: str = "2GB",
    threads: int = 2,
    corpus_audit_function: CorpusAuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: GridAuditFunction = audit_round73_causal_grid,
) -> Round73ShockStudyAudit:
    """Recompute the frozen split, thresholds, selection, hashes, and source audits."""

    selected_study = str(study_id).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 compact cohort study ID is invalid")
    errors: list[str] = []
    source_runs: tuple[_SourceRun, ...] = ()
    selected_anchor_count = 0
    rows_sha = ""
    manifest_sha = ""
    identity: Mapping[str, object] = {}
    stored_anchors: tuple[Round73ShockAnchor, ...] = ()
    try:
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            if not _table_exists(
                connection, ROUND73_SHOCK_STUDY_TABLE
            ) or not _table_exists(connection, ROUND73_SHOCK_ANCHOR_TABLE):
                raise ValueError("Round 73 compact cohort tables are missing")
            row = connection.execute(
                f"""
                SELECT schema_version, contract_sha256, manifest_json,
                       manifest_sha256, source_runs_sha256,
                       selected_identity_rows_sha256,
                       selected_anchor_rows_sha256, source_run_count,
                       selected_anchor_count, study_start_wall_ns,
                       study_end_wall_ns
                FROM {ROUND73_SHOCK_STUDY_TABLE} WHERE study_id = ?
                """,
                [selected_study],
            ).fetchone()
            if row is None:
                raise ValueError("Round 73 compact cohort study was not found")
            manifest_text = str(row[2])
            manifest_sha = str(row[3])
            identity = _strict_json_object(manifest_text, "Round 73 cohort manifest")
            if (
                str(row[0]) != ROUND73_SHOCK_STUDY_SCHEMA_VERSION
                or str(row[1]) != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
                or _SHA256.fullmatch(manifest_sha) is None
                or _sha256_text(manifest_text) != manifest_sha
                or identity.get("schema_version") != ROUND73_SHOCK_STUDY_SCHEMA_VERSION
                or identity.get("contract_sha256")
                != ROUND73_COMPACT_TARGET_CONTRACT_SHA256
                or identity.get("study_id") != selected_study
                or identity.get("target_manifest_count_at_publish") != 0
                or identity.get("target_observed") is not False
                or identity.get("model_evaluated") is not False
                or identity.get("profitability_claim") is not False
                or identity.get("trading_authority") is not False
            ):
                raise ValueError("Round 73 compact cohort manifest identity differs")
            source_runs = _source_runs_from_manifest(identity)
            if (
                _source_runs_sha256(source_runs) != str(row[4])
                or identity.get("source_runs_sha256") != str(row[4])
                or len(source_runs) != int(row[7])
                or identity.get("source_run_count") != int(row[7])
            ):
                raise ValueError("Round 73 compact cohort source-run hash differs")
            current_sources = {
                item.run_id: item
                for item in _candidate_source_runs(
                    connection,
                    selection_as_of_wall_ns=int(identity["selection_as_of_wall_ns"]),
                )
            }
            if any(current_sources.get(item.run_id) != item for item in source_runs):
                raise ValueError("Round 73 compact cohort source manifest drifted")
            recomputed_examined, recomputed_selected = _study_days(
                tuple(current_sources.values()),
                now_wall_ns=int(identity["selection_as_of_wall_ns"]),
            )
            stored_examined = identity.get("examined_days")
            stored_selected_days = identity.get("selected_utc_days")
            if (
                not isinstance(stored_examined, Sequence)
                or isinstance(stored_examined, (str, bytes, bytearray))
                or [day.as_dict() for day in recomputed_examined]
                != list(stored_examined)
                or [day.utc_day for day in recomputed_selected] != stored_selected_days
            ):
                raise ValueError(
                    "Round 73 compact cohort earliest-day selection differs"
                )
            raw_anchor_rows = connection.execute(
                f"SELECT {', '.join(_ANCHOR_COLUMNS)} "
                f"FROM {ROUND73_SHOCK_ANCHOR_TABLE} WHERE study_id = ? "
                "ORDER BY symbol, anchor_wall_ns, run_id, anchor_index",
                [selected_study],
            ).fetchall()
            stored_anchors = tuple(_anchor_from_row(item) for item in raw_anchor_rows)
            selected_anchor_count = len(stored_anchors)
            for anchor in stored_anchors:
                expected_hash = _sha256_text(
                    _canonical_json(list(anchor.values_without_hash()))
                )
                if (
                    anchor.selected_anchor_sha256 != expected_hash
                    or anchor.study_id != selected_study
                    or anchor.symbol not in IMPACT_CAPTURE_SYMBOLS
                    or anchor.source_max_received_monotonic_ns
                    >= anchor.anchor_monotonic_ns
                    or not math.isfinite(anchor.shock_ratio)
                    or not math.isfinite(anchor.shock_direction_taker_share)
                    or _SHA256.fullmatch(anchor.feature_vector_sha256) is None
                ):
                    raise ValueError("Round 73 compact cohort anchor differs")
            rows_sha = _selected_anchor_rows_sha256(stored_anchors)
            if (
                rows_sha != str(row[6])
                or identity.get("selected_anchor_rows_sha256") != rows_sha
                or selected_anchor_count != int(row[8])
                or identity.get("selected_anchor_count") != selected_anchor_count
                or _selected_counts(stored_anchors)
                != identity.get("selected_anchor_counts")
            ):
                raise ValueError("Round 73 compact cohort anchor aggregate differs")
            selected_days = _days_from_manifest(identity)
            study_start = _day_start_ns(date.fromisoformat(selected_days[0].utc_day))
            study_end = study_start + ROUND73_STUDY_REQUIRED_DAYS * ROUND73_STUDY_DAY_NS
            if study_start != int(row[9]) or study_end != int(row[10]):
                raise ValueError("Round 73 compact cohort study bounds differ")
            recomputed_thresholds = _thresholds(
                connection,
                source_runs=source_runs,
                study_start_wall_ns=study_start,
            )
            stored_thresholds = _thresholds_from_manifest(identity)
            if recomputed_thresholds != stored_thresholds:
                raise ValueError("Round 73 compact cohort thresholds differ")
            recomputed_identities = _selected_anchor_identities(
                connection,
                source_runs=source_runs,
                selected_days=selected_days,
                thresholds=recomputed_thresholds,
            )
            identity_rows_sha = _selected_identity_rows_sha256(recomputed_identities)
            if (
                identity_rows_sha != str(row[5])
                or identity.get("selected_identity_rows_sha256") != identity_rows_sha
            ):
                raise ValueError("Round 73 compact cohort identity-row hash differs")
            recomputed_anchors = tuple(
                _make_anchor(selected_study, item) for item in recomputed_identities
            )
            if recomputed_anchors != stored_anchors:
                raise ValueError("Round 73 compact cohort selection differs")
    except (
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"cohort:{type(exc).__name__}:{exc}")
    deeply_audited = 0
    if deep_source_audit and not errors:
        try:
            deeply_audited = _audit_sources(
                database,
                source_runs=source_runs,
                memory_limit=memory_limit,
                threads=threads,
                corpus_audit_function=corpus_audit_function,
                grid_audit_function=grid_audit_function,
            )
        except (duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"source:{type(exc).__name__}:{exc}")
    return Round73ShockStudyAudit(
        study_id=selected_study,
        passed=not errors,
        errors=tuple(errors),
        source_run_count=len(source_runs),
        selected_anchor_count=selected_anchor_count,
        deeply_audited_source_count=deeply_audited,
        selected_anchor_rows_sha256=rows_sha,
        study_manifest_sha256=manifest_sha,
    )


__all__ = [
    "ROUND73_COMPACT_TARGET_CONTRACT_SHA256",
    "ROUND73_ROLE_BOUNDARY_EMBARGO_NS",
    "ROUND73_SHOCK_ANCHOR_TABLE",
    "ROUND73_SHOCK_MINIMUM_RATIO",
    "ROUND73_SHOCK_REFRACTORY_NS",
    "ROUND73_SHOCK_STUDY_SCHEMA_VERSION",
    "ROUND73_SHOCK_STUDY_TABLE",
    "ROUND73_STUDY_NOT_BEFORE_WALL_NS",
    "ROUND73_STUDY_REQUIRED_DAYS",
    "Round73ShockAnchor",
    "Round73ShockCohortNotReady",
    "Round73ShockStudyAudit",
    "Round73ShockStudyReport",
    "Round73ShockThreshold",
    "Round73StudyDay",
    "audit_round73_shock_cohort",
    "build_round73_shock_cohort",
    "nearest_rank_percentile",
]
