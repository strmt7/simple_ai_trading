from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from simple_ai_trading.impact_absorption_capture import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_cohort import (
    ROUND73_SHOCK_ANCHOR_TABLE,
    ROUND73_SHOCK_STUDY_TABLE,
    Round73ShockCohortNotReady,
    audit_round73_shock_cohort,
    build_round73_shock_cohort,
    nearest_rank_percentile,
)
from simple_ai_trading.impact_absorption_corpus import (
    ROUND73_CORPUS_CONTRACT_SHA256,
    ROUND73_CORPUS_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_grid import (
    ROUND73_GRID_CONTRACT_SHA256,
    ROUND73_GRID_SCHEMA_VERSION,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _wall_ns(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000


def _create_source_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE impact_capture_run (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            capture_contract_sha256 VARCHAR NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE impact_corpus_run_manifest_v3 (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            coverage_start_wall_ns UBIGINT NOT NULL,
            coverage_end_wall_ns UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE impact_feature_run_manifest_v4 (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            source_corpus_manifest_sha256 VARCHAR NOT NULL,
            build_manifest_sha256 VARCHAR NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE impact_feature_anchor_v4 (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            anchor_monotonic_ns UBIGINT NOT NULL,
            anchor_wall_ns UBIGINT NOT NULL,
            source_max_received_monotonic_ns UBIGINT NOT NULL,
            valid BOOLEAN NOT NULL,
            shock_ratio DOUBLE,
            shock_direction TINYINT NOT NULL,
            shock_direction_taker_share DOUBLE NOT NULL,
            PRIMARY KEY (run_id, symbol, anchor_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE impact_feature_vector_v4 (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            vector_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (run_id, symbol, anchor_index)
        )
        """
    )


def _seed_days(
    database: Path,
    *,
    day_count: int,
    coverage_hours: tuple[int, ...] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    corpus_hashes: dict[str, str] = {}
    grid_hashes: dict[str, str] = {}
    start = datetime(2026, 7, 24, tzinfo=UTC)
    hours = coverage_hours or tuple(23 for _ in range(day_count))
    with duckdb.connect(str(database)) as connection:
        _create_source_tables(connection)
        for day_index in range(day_count):
            run_id = f"{day_index + 1:032x}"
            corpus_sha = _sha(f"corpus-{day_index}")
            grid_sha = _sha(f"grid-{day_index}")
            corpus_hashes[run_id] = corpus_sha
            grid_hashes[run_id] = grid_sha
            day_start = start + timedelta(days=day_index)
            start_ns = _wall_ns(day_start)
            end_ns = _wall_ns(day_start + timedelta(hours=hours[day_index]))
            connection.execute(
                "INSERT INTO impact_capture_run VALUES (?, ?, ?)",
                [
                    run_id,
                    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
                    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
                ],
            )
            connection.execute(
                "INSERT INTO impact_corpus_run_manifest_v3 VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    ROUND73_CORPUS_SCHEMA_VERSION,
                    ROUND73_CORPUS_CONTRACT_SHA256,
                    corpus_sha,
                    start_ns,
                    end_ns,
                    end_ns + 1,
                ],
            )
            connection.execute(
                "INSERT INTO impact_feature_run_manifest_v4 VALUES (?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    ROUND73_GRID_SCHEMA_VERSION,
                    ROUND73_GRID_CONTRACT_SHA256,
                    corpus_sha,
                    grid_sha,
                    end_ns + 2,
                ],
            )
            for symbol_index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS):
                for anchor_index, ratio in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
                    selected_ratio = (
                        ratio
                        if day_index < 4 or ratio < 5.0
                        else 6.0 + day_index / 10.0
                    )
                    anchor_wall = start_ns + (100 + anchor_index * 20) * 1_000_000_000
                    anchor_mono = (
                        1_000_000_000_000
                        + day_index * 100_000_000_000
                        + symbol_index * 10_000_000_000
                        + anchor_index * 1_000_000_000
                    )
                    vector_sha = _sha(f"vector-{run_id}-{symbol}-{anchor_index}")
                    connection.execute(
                        "INSERT INTO impact_feature_anchor_v4 VALUES "
                        "(?, ?, ?, ?, ?, ?, true, ?, ?, ?)",
                        [
                            run_id,
                            symbol,
                            anchor_index,
                            anchor_mono,
                            anchor_wall,
                            anchor_mono - 1,
                            selected_ratio,
                            1 if anchor_index % 2 == 0 else -1,
                            0.8,
                        ],
                    )
                    connection.execute(
                        "INSERT INTO impact_feature_vector_v4 VALUES (?, ?, ?, ?)",
                        [run_id, symbol, anchor_index, vector_sha],
                    )
    return corpus_hashes, grid_hashes


def _audit_functions(
    corpus_hashes: dict[str, str],
    grid_hashes: dict[str, str],
):
    def corpus_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(
            passed=True,
            manifest_sha256=corpus_hashes[run_id],
        )

    def grid_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(
            passed=True,
            build_manifest_sha256=grid_hashes[run_id],
        )

    return corpus_audit, grid_audit


def test_nearest_rank_percentile_is_exact_and_rejects_nonfinite() -> None:
    assert nearest_rank_percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert nearest_rank_percentile([4.0, 1.0, 3.0, 2.0], 0.99) == 4.0
    with pytest.raises(ValueError, match="finite"):
        nearest_rank_percentile([1.0, float("nan")], 0.99)


def test_compact_cohort_is_deterministic_hash_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cohort.duckdb"
    corpus_hashes, grid_hashes = _seed_days(database, day_count=7)
    corpus_audit, grid_audit = _audit_functions(corpus_hashes, grid_hashes)
    now_ns = _wall_ns(datetime(2026, 7, 31, tzinfo=UTC))

    report = build_round73_shock_cohort(
        database,
        now_wall_ns=now_ns,
        corpus_audit_function=corpus_audit,
        grid_audit_function=grid_audit,
    )

    assert report.selected_utc_days == (
        "2026-07-24",
        "2026-07-25",
        "2026-07-26",
        "2026-07-27",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
    )
    assert report.selected_anchor_count == 21
    assert report.selected_anchor_counts == {
        symbol: 7 for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    assert all(item.training_observation_count == 20 for item in report.thresholds)
    assert all(item.nearest_rank == 20 for item in report.thresholds)
    assert all(item.empirical_ratio == 5.0 for item in report.thresholds)

    late_run_id = "f" * 32
    late_corpus_sha = _sha("late-corpus")
    late_grid_sha = _sha("late-grid")
    first_day_start = _wall_ns(datetime(2026, 7, 24, tzinfo=UTC))
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "INSERT INTO impact_capture_run VALUES (?, ?, ?)",
            [
                late_run_id,
                IMPACT_CAPTURE_V9_SCHEMA_VERSION,
                IMPACT_CAPTURE_V9_CONTRACT_SHA256,
            ],
        )
        connection.execute(
            "INSERT INTO impact_corpus_run_manifest_v3 VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                late_run_id,
                ROUND73_CORPUS_SCHEMA_VERSION,
                ROUND73_CORPUS_CONTRACT_SHA256,
                late_corpus_sha,
                first_day_start,
                first_day_start + 3_600_000_000_000,
                now_ns + 1,
            ],
        )
        connection.execute(
            "INSERT INTO impact_feature_run_manifest_v4 VALUES (?, ?, ?, ?, ?, ?)",
            [
                late_run_id,
                ROUND73_GRID_SCHEMA_VERSION,
                ROUND73_GRID_CONTRACT_SHA256,
                late_corpus_sha,
                late_grid_sha,
                now_ns + 2,
            ],
        )

    audit = audit_round73_shock_cohort(
        database,
        study_id=report.study_id,
        corpus_audit_function=corpus_audit,
        grid_audit_function=grid_audit,
    )
    assert audit.passed
    assert audit.errors == ()
    assert audit.deeply_audited_source_count == 7
    assert (
        build_round73_shock_cohort(
            database,
            now_wall_ns=now_ns,
            corpus_audit_function=corpus_audit,
            grid_audit_function=grid_audit,
        )
        == report
    )
    with duckdb.connect(str(database)) as connection:
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_SHOCK_STUDY_TABLE}"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_SHOCK_ANCHOR_TABLE}"
            ).fetchone()[0]
            == 21
        )
        connection.execute(
            f"UPDATE {ROUND73_SHOCK_ANCHOR_TABLE} SET shock_ratio = shock_ratio + 1 "
            "WHERE study_id = ? AND anchor_index = 4",
            [report.study_id],
        )
    tampered = audit_round73_shock_cohort(
        database,
        study_id=report.study_id,
        deep_source_audit=False,
    )
    assert not tampered.passed
    assert any("anchor differs" in error for error in tampered.errors)


def test_compact_cohort_requires_seven_consecutive_complete_days(
    tmp_path: Path,
) -> None:
    database = tmp_path / "not-ready.duckdb"
    corpus_hashes, grid_hashes = _seed_days(
        database,
        day_count=7,
        coverage_hours=(23, 23, 22, 23, 23, 23, 23),
    )
    corpus_audit, grid_audit = _audit_functions(corpus_hashes, grid_hashes)

    with pytest.raises(Round73ShockCohortNotReady) as failure:
        build_round73_shock_cohort(
            database,
            now_wall_ns=_wall_ns(datetime(2026, 7, 31, tzinfo=UTC)),
            corpus_audit_function=corpus_audit,
            grid_audit_function=grid_audit,
        )

    assert len(failure.value.examined_days) == 7
    assert failure.value.examined_days[2].reason == "insufficient_integrity_coverage"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert not connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            f"WHERE table_name = '{ROUND73_SHOCK_STUDY_TABLE}'"
        ).fetchone()[0]


def test_compact_cohort_restarts_after_an_incomplete_day(tmp_path: Path) -> None:
    database = tmp_path / "restarted-streak.duckdb"
    corpus_hashes, grid_hashes = _seed_days(
        database,
        day_count=8,
        coverage_hours=(22, 23, 23, 23, 23, 23, 23, 23),
    )
    corpus_audit, grid_audit = _audit_functions(corpus_hashes, grid_hashes)

    report = build_round73_shock_cohort(
        database,
        now_wall_ns=_wall_ns(datetime(2026, 8, 1, tzinfo=UTC)),
        corpus_audit_function=corpus_audit,
        grid_audit_function=grid_audit,
    )

    assert report.selected_utc_days[0] == "2026-07-25"
    assert report.selected_utc_days[-1] == "2026-07-31"
    with duckdb.connect(str(database), read_only=True) as connection:
        observed = connection.execute(
            "SELECT DISTINCT utc_day, day_ordinal, role "
            f"FROM {ROUND73_SHOCK_ANCHOR_TABLE} WHERE study_id = ?",
            [report.study_id],
        ).fetchall()
    expected_roles = {
        1: "training",
        2: "training",
        3: "training",
        4: "training",
        5: "tuning",
        6: "test",
        7: "test",
    }
    assert observed
    assert all(
        utc_day == report.selected_utc_days[ordinal - 1]
        and role == expected_roles[ordinal]
        for utc_day, ordinal, role in observed
    )


def test_compact_cohort_rejects_any_preexisting_target_manifest(
    tmp_path: Path,
) -> None:
    database = tmp_path / "prelabeled.duckdb"
    corpus_hashes, grid_hashes = _seed_days(database, day_count=7)
    corpus_audit, grid_audit = _audit_functions(corpus_hashes, grid_hashes)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE impact_target_run_manifest_v1 (run_id VARCHAR PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO impact_target_run_manifest_v1 VALUES (?)",
            [f"{1:032x}"],
        )

    with pytest.raises(ValueError, match="already has target outcomes"):
        build_round73_shock_cohort(
            database,
            now_wall_ns=_wall_ns(datetime(2026, 7, 31, tzinfo=UTC)),
            corpus_audit_function=corpus_audit,
            grid_audit_function=grid_audit,
        )
