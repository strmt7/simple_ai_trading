"""Role-staged target publication for the Round 73 prospective holdout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable

import duckdb

from .impact_absorption_cohort import (
    ROUND73_STUDY_NOT_BEFORE_WALL_NS,
    audit_round73_shock_cohort,
)
from .impact_absorption_corpus import audit_round73_corpus_manifest
from .impact_absorption_grid_store import audit_round73_causal_grid
from .impact_absorption_store import ImpactAbsorptionStore
from .impact_absorption_target_store import (
    Round73TargetOption,
    parse_round73_target_quantity_rules,
    replay_round73_target_rows_v9,
    round73_target_option_invariant_errors,
    round73_target_quantity_invariant_errors,
)
from .impact_absorption_target_store_v2 import (
    ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
    ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR,
    ROUND73_TARGET_V2_HORIZONS_MS,
    ROUND73_TARGET_V2_MANIFEST_TABLE,
    ROUND73_TARGET_V2_OPTION_TABLE,
    ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
    ROUND73_TARGET_V2_SIDES,
    ROUND73_TARGET_V2_STUDY_TABLE,
    Round73CohortTargetOption,
    _cohort_option_from_base,
    _cohort_option_from_row,
    _insert_option_batch,
    _load_study_context,
    _replay_anchor_map,
    _selected_anchors,
    _source_replay_inputs,
    _stream_hash,
    _strict_json_object,
    _table_exists,
    _validated_identifier,
    _V2_OPTION_COLUMNS,
    _RUN_ID,
    _SHA256,
    _STUDY_ID,
    _canonical_json,
    _sha256_text,
)
from .impact_absorption_pretest_contract import sha256_bytes as _sha256_bytes


ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256 = (
    "54ec74e2d24d5873d6cfef3d1d3265c22829c05c15001051c10a86ad99fd6217"
)
ROUND73_TARGET_V3_SCHEMA_VERSION = "round-073-role-staged-target-v3"
ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION = "round-073-development-target-study-v3"
ROUND73_TARGET_V3_OPTION_TABLE = "impact_target_option_v3"
ROUND73_TARGET_V3_ROLE_RUN_TABLE = "impact_target_role_run_manifest_v3"
ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE = (
    "impact_target_development_study_manifest_v3"
)
ROUND73_TARGET_V3_TEST_STUDY_TABLE = "impact_target_test_study_manifest_v3"
ROUND73_PRETEST_MODEL_SCHEMA_VERSION = "round-073-pretest-model-manifest-v1"
ROUND73_PRETEST_MODEL_MANIFEST_TABLE = "impact_pretest_model_manifest_v1"
ROUND73_PRETEST_MODEL_ARTIFACT_TABLE = "impact_pretest_model_artifact_v1"
ROUND73_TEST_UNLOCK_SCHEMA_VERSION = "round-073-test-unlock-v1"
ROUND73_TEST_UNLOCK_TABLE = "impact_test_unlock_v1"
ROUND73_TEST_STUDY_V3_SCHEMA_VERSION = "round-073-test-target-study-v3"
ROUND73_TARGET_V3_SCOPES = ("development", "test")
ROUND73_DEVELOPMENT_ROLES = ("training", "tuning")
ROUND73_TEST_ROLES = ("test",)

_INSERT_BATCH_SIZE = 32_768

AuditFunction = Callable[..., object]
ReplayFunction = Callable[..., list[Round73TargetOption]]


@dataclass(frozen=True)
class Round73RoleTargetBuildReport:
    study_id: str
    run_id: str
    role_scope: str
    selected_anchor_count: int
    option_count: int
    eligible_option_count: int | None
    positive_option_count: int | None
    target_manifest_sha256: str
    cohort_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_TARGET_V3_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "test_outcomes_redacted": self.role_scope == "test",
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73RoleTargetAudit:
    study_id: str
    run_id: str
    role_scope: str
    passed: bool
    errors: tuple[str, ...]
    selected_anchor_count: int
    option_count: int
    target_manifest_sha256: str
    deep_replay_performed: bool

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "round-073-role-target-audit-v3",
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "errors": list(self.errors),
                "outcome_summary_exposed": self.role_scope == "development",
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73DevelopmentTargetStudyReport:
    study_id: str
    source_run_count: int
    selected_anchor_count: int
    option_count: int
    eligible_option_count: int
    positive_option_count: int
    role_run_manifests_sha256: str
    development_study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "test_target_rows_exist": False,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73TestTargetStudyReport:
    study_id: str
    source_run_count: int
    selected_anchor_count: int
    option_count: int
    pretest_manifest_sha256: str
    role_run_manifests_sha256: str
    test_study_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_TEST_STUDY_V3_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "test_outcomes_redacted": True,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


def _scope_roles(role_scope: str) -> tuple[str, ...]:
    selected = str(role_scope).strip().lower()
    if selected == "development":
        return ROUND73_DEVELOPMENT_ROLES
    if selected == "test":
        return ROUND73_TEST_ROLES
    raise ValueError("Round 73 target role scope must be development or test")


def _create_v3_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V3_OPTION_TABLE} (
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
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V3_ROLE_RUN_TABLE} (
            study_id VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            role_scope VARCHAR NOT NULL,
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
            PRIMARY KEY (study_id, run_id, role_scope),
            CHECK (role_scope IN ('development', 'test')),
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
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            cohort_manifest_sha256 VARCHAR NOT NULL,
            manifest_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            role_run_manifests_sha256 VARCHAR NOT NULL,
            source_run_count USMALLINT NOT NULL,
            selected_anchor_count UINTEGER NOT NULL,
            option_count UBIGINT NOT NULL,
            eligible_option_count UBIGINT NOT NULL,
            positive_option_count UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(cohort_manifest_sha256) = 64),
            CHECK (length(manifest_sha256) = 64),
            CHECK (length(role_run_manifests_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_V3_TEST_STUDY_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            cohort_manifest_sha256 VARCHAR NOT NULL,
            development_study_manifest_sha256 VARCHAR NOT NULL,
            pretest_manifest_sha256 VARCHAR NOT NULL,
            manifest_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            role_run_manifests_sha256 VARCHAR NOT NULL,
            source_run_count USMALLINT NOT NULL,
            selected_anchor_count UINTEGER NOT NULL,
            option_count UBIGINT NOT NULL,
            eligible_option_count UBIGINT NOT NULL,
            positive_option_count UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(cohort_manifest_sha256) = 64),
            CHECK (length(development_study_manifest_sha256) = 64),
            CHECK (length(pretest_manifest_sha256) = 64),
            CHECK (length(manifest_sha256) = 64),
            CHECK (length(role_run_manifests_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_PRETEST_MODEL_MANIFEST_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            development_study_manifest_sha256 VARCHAR NOT NULL,
            pretest_manifest_json VARCHAR NOT NULL,
            pretest_manifest_sha256 VARCHAR NOT NULL,
            artifact_manifest_sha256 VARCHAR NOT NULL,
            artifact_count UINTEGER NOT NULL,
            artifact_bytes UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(development_study_manifest_sha256) = 64),
            CHECK (length(pretest_manifest_sha256) = 64),
            CHECK (length(artifact_manifest_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_PRETEST_MODEL_ARTIFACT_TABLE} (
            study_id VARCHAR NOT NULL,
            artifact_name VARCHAR NOT NULL,
            artifact_kind VARCHAR NOT NULL,
            symbol VARCHAR,
            media_type VARCHAR NOT NULL,
            artifact_sha256 VARCHAR NOT NULL,
            byte_count UBIGINT NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (study_id, artifact_name),
            CHECK (length(artifact_sha256) = 64),
            CHECK (byte_count > 0)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TEST_UNLOCK_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            pretest_manifest_sha256 VARCHAR NOT NULL,
            unlocked_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(pretest_manifest_sha256) = 64)
        )
        """
    )


def _assert_v3_table_shapes(connection: duckdb.DuckDBPyConnection) -> None:
    expected = {
        ROUND73_TARGET_V3_OPTION_TABLE: _V2_OPTION_COLUMNS,
        ROUND73_TARGET_V3_ROLE_RUN_TABLE: (
            "study_id",
            "run_id",
            "role_scope",
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
        ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "cohort_manifest_sha256",
            "manifest_json",
            "manifest_sha256",
            "role_run_manifests_sha256",
            "source_run_count",
            "selected_anchor_count",
            "option_count",
            "eligible_option_count",
            "positive_option_count",
            "recorded_at_wall_ns",
        ),
        ROUND73_TARGET_V3_TEST_STUDY_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "cohort_manifest_sha256",
            "development_study_manifest_sha256",
            "pretest_manifest_sha256",
            "manifest_json",
            "manifest_sha256",
            "role_run_manifests_sha256",
            "source_run_count",
            "selected_anchor_count",
            "option_count",
            "eligible_option_count",
            "positive_option_count",
            "recorded_at_wall_ns",
        ),
        ROUND73_PRETEST_MODEL_MANIFEST_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "development_study_manifest_sha256",
            "pretest_manifest_json",
            "pretest_manifest_sha256",
            "artifact_manifest_sha256",
            "artifact_count",
            "artifact_bytes",
            "recorded_at_wall_ns",
        ),
        ROUND73_PRETEST_MODEL_ARTIFACT_TABLE: (
            "study_id",
            "artifact_name",
            "artifact_kind",
            "symbol",
            "media_type",
            "artifact_sha256",
            "byte_count",
            "payload",
        ),
        ROUND73_TEST_UNLOCK_TABLE: (
            "study_id",
            "schema_version",
            "contract_sha256",
            "pretest_manifest_sha256",
            "unlocked_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 v3 target table schema differs: {table}")


def _reject_v2_contamination(
    connection: duckdb.DuckDBPyConnection,
    study_id: str,
) -> None:
    for table in (
        ROUND73_TARGET_V2_OPTION_TABLE,
        ROUND73_TARGET_V2_MANIFEST_TABLE,
        ROUND73_TARGET_V2_STUDY_TABLE,
    ):
        if _table_exists(connection, table):
            count = int(
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE study_id = ?", [study_id]
                ).fetchone()[0]
            )
            if count:
                raise ValueError(
                    "Round 73 eligible study is contaminated by v2 targets"
                )


def _require_prospective_context(context: object) -> None:
    starts = [int(source.coverage_start_wall_ns) for source in context.sources]
    if not starts or min(starts) < ROUND73_STUDY_NOT_BEFORE_WALL_NS:
        raise ValueError("Round 73 v3 targets require the prospective cohort")


def _count_role_option_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    roles: Sequence[str],
) -> int:
    return int(
        connection.execute(
            f"SELECT count(*) FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
            "JOIN impact_shock_anchor_v1 a "
            "ON a.study_id = o.study_id AND a.run_id = o.run_id "
            "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
            "WHERE o.study_id = ? AND a.role IN (" + ",".join("?" for _ in roles) + ")",
            [study_id, *roles],
        ).fetchone()[0]
    )


def _reject_orphan_v3_options(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
) -> None:
    orphan_count = int(
        connection.execute(
            f"SELECT count(*) FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
            "LEFT JOIN impact_shock_anchor_v1 a "
            "ON a.study_id = o.study_id AND a.run_id = o.run_id "
            "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
            "WHERE o.study_id = ? AND a.study_id IS NULL",
            [study_id],
        ).fetchone()[0]
    )
    if orphan_count:
        raise ValueError("Round 73 v3 target study contains orphan option rows")


def _scope_anchors(context: object, *, run_id: str, role_scope: str):
    roles = set(_scope_roles(role_scope))
    return tuple(
        anchor
        for anchor in _selected_anchors(context, run_id=run_id)
        if anchor.role in roles
    )


def _require_test_unlock(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
) -> None:
    if _SHA256.fullmatch(pretest_manifest_sha256) is None:
        raise ValueError("Round 73 pretest manifest hash is invalid")
    if not _table_exists(connection, ROUND73_TEST_UNLOCK_TABLE):
        raise ValueError("Round 73 test targets are locked")
    row = connection.execute(
        f"SELECT contract_sha256, pretest_manifest_sha256 "
        f"FROM {ROUND73_TEST_UNLOCK_TABLE} WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 test targets are locked")
    if row != (ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256, pretest_manifest_sha256):
        raise ValueError("Round 73 test target unlock identity differs")


def _role_run_identity(
    *,
    context: object,
    source: object,
    role_scope: str,
    anchors: Sequence[object],
    options: Sequence[Round73CohortTargetOption],
) -> dict[str, object]:
    roles = _scope_roles(role_scope)
    option_rows_sha = _stream_hash([item.cohort_option_sha256 for item in options])
    eligible_count = sum(item.eligible for item in options)
    positive_count = sum(
        item.eligible and bool(item.positive_net_payoff) for item in options
    )
    reason_counts: dict[str, int] = {}
    for option in options:
        for reason in json.loads(option.ineligible_reasons_json):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "schema_version": ROUND73_TARGET_V3_SCHEMA_VERSION,
        "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
        "study_id": context.study_id,
        "run_id": source.run_id,
        "role_scope": role_scope,
        "roles": list(roles),
        "cohort_manifest_sha256": context.study_manifest_sha256,
        "source_corpus_manifest_sha256": source.corpus_manifest_sha256,
        "source_grid_manifest_sha256": source.grid_manifest_sha256,
        "dimensions": {
            "entry_delay_milliseconds": list(ROUND73_TARGET_V2_ENTRY_DELAYS_MS),
            "holding_horizon_milliseconds": list(ROUND73_TARGET_V2_HORIZONS_MS),
            "reference_quote_notional": list(ROUND73_TARGET_V2_REFERENCE_NOTIONALS),
            "side": list(ROUND73_TARGET_V2_SIDES),
        },
        "selected_anchor_count": len(anchors),
        "selected_anchor_counts_by_role": {
            role: sum(anchor.role == role for anchor in anchors) for role in roles
        },
        "expected_options_per_selected_anchor": (
            ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
        ),
        "option_count": len(options),
        "eligible_option_count": eligible_count,
        "ineligible_option_count": len(options) - eligible_count,
        "positive_option_count": positive_count,
        "reason_counts": reason_counts,
        "option_rows_sha256": option_rows_sha,
        "first_decision_wall_ns": (
            None if not anchors else min(item.anchor_wall_ns for item in anchors)
        ),
        "last_decision_wall_ns": (
            None if not anchors else max(item.anchor_wall_ns for item in anchors)
        ),
        "test_outcomes_redacted_from_build_report": role_scope == "test",
        "target_constructed": True,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }


def _report_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73RoleTargetBuildReport:
    test_scope = str(identity["role_scope"]) == "test"
    return Round73RoleTargetBuildReport(
        study_id=str(identity["study_id"]),
        run_id=str(identity["run_id"]),
        role_scope=str(identity["role_scope"]),
        selected_anchor_count=int(identity["selected_anchor_count"]),
        option_count=int(identity["option_count"]),
        eligible_option_count=(
            None if test_scope else int(identity["eligible_option_count"])
        ),
        positive_option_count=(
            None if test_scope else int(identity["positive_option_count"])
        ),
        target_manifest_sha256=manifest_sha256,
        cohort_manifest_sha256=str(identity["cohort_manifest_sha256"]),
    )


def build_round73_role_targets(
    database: str | Path,
    *,
    study_id: str,
    run_id: str,
    role_scope: str,
    pretest_manifest_sha256: str | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73RoleTargetBuildReport:
    """Replay one source run for exactly one physically staged role scope."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_run = _validated_identifier(run_id, _RUN_ID, "target run ID")
    selected_scope = str(role_scope).strip().lower()
    _scope_roles(selected_scope)
    if selected_scope == "development" and pretest_manifest_sha256 is not None:
        raise ValueError("Round 73 development replay cannot receive a pretest hash")
    if selected_scope == "test" and pretest_manifest_sha256 is None:
        raise ValueError("Round 73 test replay requires a pretest manifest hash")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        _require_prospective_context(context)
        _reject_v2_contamination(connection, selected_study)
        source = context.source_by_run.get(selected_run)
        if source is None:
            raise ValueError("Round 73 v3 target run is outside the cohort")
        if selected_scope == "test":
            _require_test_unlock(
                connection,
                study_id=selected_study,
                pretest_manifest_sha256=str(pretest_manifest_sha256),
            )
        existing = (
            connection.execute(
                f"SELECT target_manifest_json, target_manifest_sha256 "
                f"FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                "WHERE study_id = ? AND run_id = ? AND role_scope = ?",
                [selected_study, selected_run, selected_scope],
            ).fetchone()
            if _table_exists(connection, ROUND73_TARGET_V3_ROLE_RUN_TABLE)
            else None
        )
        if (
            existing is None
            and selected_scope == "development"
            and any(
                _table_exists(connection, table)
                and int(
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE study_id = ?",
                        [selected_study],
                    ).fetchone()[0]
                )
                for table in (
                    ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE,
                    ROUND73_PRETEST_MODEL_MANIFEST_TABLE,
                    ROUND73_TEST_UNLOCK_TABLE,
                )
            )
        ):
            raise ValueError("Round 73 development targets are already frozen")
        if (
            existing is None
            and selected_scope == "test"
            and _table_exists(connection, ROUND73_TARGET_V3_TEST_STUDY_TABLE)
            and int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TARGET_V3_TEST_STUDY_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            )
        ):
            raise ValueError("Round 73 test targets are already sealed")
    if existing is not None:
        audit = audit_round73_role_targets(
            database,
            study_id=selected_study,
            run_id=selected_run,
            role_scope=selected_scope,
            pretest_manifest_sha256=pretest_manifest_sha256,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing v3 role-target audit failed")
        identity = _strict_json_object(str(existing[0]), "Round 73 v3 role manifest")
        return _report_from_identity(identity, str(existing[1]))
    cohort_audit = cohort_audit_function(
        database,
        study_id=selected_study,
        deep_source_audit=False,
        memory_limit=memory_limit,
        threads=threads,
    )
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
        getattr(cohort_audit, "passed", False) is not True
        or getattr(corpus_audit, "passed", False) is not True
        or getattr(grid_audit, "passed", False) is not True
    ):
        raise ValueError("Round 73 v3 role-target source audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        _require_prospective_context(context)
        _reject_v2_contamination(connection, selected_study)
        source = context.source_by_run[selected_run]
        if source.corpus_manifest_sha256 != getattr(
            corpus_audit, "manifest_sha256", ""
        ) or source.grid_manifest_sha256 != getattr(
            grid_audit, "build_manifest_sha256", ""
        ):
            raise ValueError("Round 73 v3 audited source hash differs")
        if selected_scope == "test":
            _require_test_unlock(
                connection,
                study_id=selected_study,
                pretest_manifest_sha256=str(pretest_manifest_sha256),
            )
        started_wall_ns, started_monotonic_ns, segments = _source_replay_inputs(
            connection,
            source=source,
            corpus_manifest_sha256=source.corpus_manifest_sha256,
            grid_manifest_sha256=source.grid_manifest_sha256,
        )
        anchors = _scope_anchors(
            context,
            run_id=selected_run,
            role_scope=selected_scope,
        )
        anchor_hashes = {
            (anchor.symbol, anchor.anchor_index): anchor.selected_anchor_sha256
            for anchor in anchors
        }
        quantity_rules = (
            {}
            if not anchors
            else parse_round73_target_quantity_rules(connection, run_id=selected_run)
        )
        base_options = (
            []
            if not anchors
            else replay_function(
                connection,
                run_id=selected_run,
                anchors=_replay_anchor_map(anchors),
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
    expected_count = len(anchors) * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
    if len(base_options) != expected_count:
        raise ValueError("Round 73 v3 role-target option count differs")
    options: list[Round73CohortTargetOption] = []
    for option in base_options:
        anchor_hash = anchor_hashes.get((option.symbol, option.anchor_index))
        if anchor_hash is None or option.run_id != selected_run:
            raise ValueError("Round 73 v3 option crossed its role scope")
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
            raise ValueError("Round 73 v3 target invariant failed: " + ",".join(errors))
        options.append(
            _cohort_option_from_base(
                option,
                study_id=selected_study,
                selected_anchor_sha256=anchor_hash,
            )
        )
    options.sort(
        key=lambda item: (
            item.run_id,
            item.symbol,
            item.anchor_index,
            item.entry_delay_ms,
            item.horizon_ms,
            item.reference_quote_notional,
            item.side,
        )
    )
    identity = _role_run_identity(
        context=context,
        source=source,
        role_scope=selected_scope,
        anchors=anchors,
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
        _create_v3_tables(connection)
        _assert_v3_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            _reject_v2_contamination(connection, selected_study)
            transaction_context = _load_study_context(
                connection, study_id=selected_study
            )
            _require_prospective_context(transaction_context)
            transaction_source = transaction_context.source_by_run.get(selected_run)
            if (
                transaction_context.study_manifest_sha256
                != context.study_manifest_sha256
                or transaction_source != source
                or tuple(
                    anchor.selected_anchor_sha256
                    for anchor in _scope_anchors(
                        transaction_context,
                        run_id=selected_run,
                        role_scope=selected_scope,
                    )
                )
                != tuple(anchor.selected_anchor_sha256 for anchor in anchors)
            ):
                raise ValueError("Round 73 v3 target source changed during replay")
            if selected_scope == "test":
                _require_test_unlock(
                    connection,
                    study_id=selected_study,
                    pretest_manifest_sha256=str(pretest_manifest_sha256),
                )
            elif any(
                int(
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE study_id = ?",
                        [selected_study],
                    ).fetchone()[0]
                )
                for table in (
                    ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE,
                    ROUND73_PRETEST_MODEL_MANIFEST_TABLE,
                    ROUND73_TEST_UNLOCK_TABLE,
                )
            ):
                raise ValueError("Round 73 development targets froze during replay")
            duplicate = connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                "WHERE study_id = ? AND run_id = ? AND role_scope = ?",
                [selected_study, selected_run, selected_scope],
            ).fetchone()[0]
            if int(duplicate):
                raise ValueError("Round 73 v3 role manifest appeared during build")
            for batch_index, start in enumerate(
                range(0, len(options), _INSERT_BATCH_SIZE)
            ):
                _insert_option_batch(
                    connection,
                    options[start : start + _INSERT_BATCH_SIZE],
                    batch_index=batch_index,
                    table_name=ROUND73_TARGET_V3_OPTION_TABLE,
                    view_namespace="v3",
                )
            connection.execute(
                f"INSERT INTO {ROUND73_TARGET_V3_ROLE_RUN_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    selected_study,
                    selected_run,
                    selected_scope,
                    ROUND73_TARGET_V3_SCHEMA_VERSION,
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
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


def audit_round73_role_targets(
    database: str | Path,
    *,
    study_id: str,
    run_id: str,
    role_scope: str,
    pretest_manifest_sha256: str | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
    deep_replay: bool = True,
) -> Round73RoleTargetAudit:
    """Reconcile one role-scoped manifest and optionally exact-replay it."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_run = _validated_identifier(run_id, _RUN_ID, "target run ID")
    selected_scope = str(role_scope).strip().lower()
    _scope_roles(selected_scope)
    errors: list[str] = []
    manifest_sha = ""
    selected_anchor_count = 0
    option_count = 0
    replay_performed = False
    try:
        cohort = cohort_audit_function(
            database,
            study_id=selected_study,
            deep_source_audit=False,
            memory_limit=memory_limit,
            threads=threads,
        )
        corpus = corpus_audit_function(
            database,
            run_id=selected_run,
            memory_limit=memory_limit,
            threads=threads,
        )
        grid = grid_audit_function(
            database,
            run_id=selected_run,
            memory_limit=memory_limit,
            threads=threads,
        )
        if any(
            getattr(value, "passed", False) is not True
            for value in (cohort, corpus, grid)
        ):
            raise ValueError("source audit failed")
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit=memory_limit,
            threads=threads,
        ) as store:
            connection = store.connect()
            _assert_v3_table_shapes(connection)
            context = _load_study_context(connection, study_id=selected_study)
            _require_prospective_context(context)
            _reject_v2_contamination(connection, selected_study)
            _reject_orphan_v3_options(connection, study_id=selected_study)
            source = context.source_by_run[selected_run]
            if source.corpus_manifest_sha256 != getattr(
                corpus, "manifest_sha256", ""
            ) or source.grid_manifest_sha256 != getattr(
                grid, "build_manifest_sha256", ""
            ):
                raise ValueError("source hash differs")
            if selected_scope == "test":
                if pretest_manifest_sha256 is None:
                    raise ValueError("test audit requires pretest hash")
                _require_test_unlock(
                    connection,
                    study_id=selected_study,
                    pretest_manifest_sha256=pretest_manifest_sha256,
                )
            anchors = _scope_anchors(
                context,
                run_id=selected_run,
                role_scope=selected_scope,
            )
            selected_anchor_count = len(anchors)
            manifest = connection.execute(
                f"SELECT target_manifest_json, target_manifest_sha256 "
                f"FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                "WHERE study_id = ? AND run_id = ? AND role_scope = ?",
                [selected_study, selected_run, selected_scope],
            ).fetchone()
            if manifest is None:
                raise ValueError("role manifest is missing")
            manifest_text = str(manifest[0])
            manifest_sha = str(manifest[1])
            identity = _strict_json_object(manifest_text, "Round 73 v3 role manifest")
            if (
                _SHA256.fullmatch(manifest_sha) is None
                or _sha256_text(manifest_text) != manifest_sha
                or identity.get("schema_version") != ROUND73_TARGET_V3_SCHEMA_VERSION
                or identity.get("contract_sha256")
                != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
                or identity.get("study_id") != selected_study
                or identity.get("run_id") != selected_run
                or identity.get("role_scope") != selected_scope
            ):
                raise ValueError("role manifest identity differs")
            raw_rows = connection.execute(
                "SELECT "
                + ", ".join(f"o.{column}" for column in _V2_OPTION_COLUMNS)
                + " "
                f"FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
                "JOIN impact_shock_anchor_v1 a "
                "ON a.study_id = o.study_id AND a.run_id = o.run_id "
                "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
                "WHERE o.study_id = ? AND o.run_id = ? "
                "AND a.role IN ("
                + ",".join("?" for _ in _scope_roles(selected_scope))
                + ") ORDER BY o.symbol, o.anchor_index, o.entry_delay_ms, "
                "o.horizon_ms, o.reference_quote_notional, o.side",
                [selected_study, selected_run, *_scope_roles(selected_scope)],
            ).fetchall()
            stored_options = tuple(_cohort_option_from_row(row) for row in raw_rows)
            option_count = len(stored_options)
            recomputed = _role_run_identity(
                context=context,
                source=source,
                role_scope=selected_scope,
                anchors=anchors,
                options=stored_options,
            )
            if identity != recomputed:
                raise ValueError("role manifest aggregate differs")
            if deep_replay and anchors:
                started_wall, started_mono, segments = _source_replay_inputs(
                    connection,
                    source=source,
                    corpus_manifest_sha256=source.corpus_manifest_sha256,
                    grid_manifest_sha256=source.grid_manifest_sha256,
                )
                rules = parse_round73_target_quantity_rules(
                    connection, run_id=selected_run
                )
                base_options = replay_function(
                    connection,
                    run_id=selected_run,
                    anchors=_replay_anchor_map(anchors),
                    quantity_rules=rules,
                    run_started_wall_ns=started_wall,
                    run_started_monotonic_ns=started_mono,
                    coverage_start_wall_ns=source.coverage_start_wall_ns,
                    coverage_end_wall_ns=source.coverage_end_wall_ns,
                    segments=segments,
                    entry_delays_ms=ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
                    horizons_ms=ROUND73_TARGET_V2_HORIZONS_MS,
                    reference_notionals=ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
                    sides=ROUND73_TARGET_V2_SIDES,
                )
                anchor_hashes = {
                    (anchor.symbol, anchor.anchor_index): (
                        anchor.selected_anchor_sha256
                    )
                    for anchor in anchors
                }
                expected = tuple(
                    sorted(
                        (
                            _cohort_option_from_base(
                                option,
                                study_id=selected_study,
                                selected_anchor_sha256=anchor_hashes[
                                    (option.symbol, option.anchor_index)
                                ],
                            )
                            for option in base_options
                        ),
                        key=lambda item: (
                            item.symbol,
                            item.anchor_index,
                            item.entry_delay_ms,
                            item.horizon_ms,
                            item.reference_quote_notional,
                            item.side,
                        ),
                    )
                )
                if expected != stored_options:
                    raise ValueError("exact-wire role replay differs")
                replay_performed = True
            elif deep_replay:
                replay_performed = True
    except (
        ArithmeticError,
        duckdb.Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"role:{type(exc).__name__}:{exc}")
    return Round73RoleTargetAudit(
        study_id=selected_study,
        run_id=selected_run,
        role_scope=selected_scope,
        passed=not errors,
        errors=tuple(errors),
        selected_anchor_count=selected_anchor_count,
        option_count=option_count,
        target_manifest_sha256=manifest_sha,
        deep_replay_performed=replay_performed,
    )


def seal_round73_development_targets(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73DevelopmentTargetStudyReport:
    """Seal every development scope while proving test targets do not exist."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    cohort = cohort_audit_function(
        database,
        study_id=selected_study,
        deep_source_audit=True,
        memory_limit=memory_limit,
        threads=threads,
    )
    if getattr(cohort, "passed", False) is not True:
        raise ValueError("Round 73 development seal cohort audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        context = _load_study_context(connection, study_id=selected_study)
        _require_prospective_context(context)
        source_run_ids = tuple(source.run_id for source in context.sources)
    for run_id in source_run_ids:
        audit = audit_round73_role_targets(
            database,
            study_id=selected_study,
            run_id=run_id,
            role_scope="development",
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
        )
        if not audit.passed:
            raise ValueError(
                f"Round 73 development run audit failed: {run_id}: "
                + "; ".join(audit.errors)
            )
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_v3_tables(connection)
        _assert_v3_table_shapes(connection)
        context = _load_study_context(connection, study_id=selected_study)
        _require_prospective_context(context)
        _reject_v2_contamination(connection, selected_study)
        _reject_orphan_v3_options(connection, study_id=selected_study)
        existing_seal = connection.execute(
            f"SELECT manifest_json, manifest_sha256 FROM "
            f"{ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE} WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        if existing_seal is None and (
            int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                    "WHERE study_id = ? AND role_scope = 'test'",
                    [selected_study],
                ).fetchone()[0]
            )
            or int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TEST_UNLOCK_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            )
        ):
            raise ValueError("Round 73 test state exists before development seal")
        test_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
                "JOIN impact_shock_anchor_v1 a "
                "ON a.study_id = o.study_id AND a.run_id = o.run_id "
                "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
                "WHERE o.study_id = ? AND a.role = 'test'",
                [selected_study],
            ).fetchone()[0]
        )
        if existing_seal is None and test_rows:
            raise ValueError("Round 73 test target rows exist before model freeze")
        rows = connection.execute(
            f"SELECT run_id, target_manifest_sha256, selected_anchor_count, "
            "option_count, eligible_option_count, positive_option_count "
            f"FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
            "WHERE study_id = ? AND role_scope = 'development' ORDER BY run_id",
            [selected_study],
        ).fetchall()
        if len(rows) != len(context.sources):
            raise ValueError("Round 73 development role manifests are incomplete")
        expected_runs = tuple(sorted(source.run_id for source in context.sources))
        if tuple(str(row[0]) for row in rows) != expected_runs:
            raise ValueError("Round 73 development role manifest runs differ")
        manifest_hashes = [str(row[1]) for row in rows]
        selected_anchor_count = sum(int(row[2]) for row in rows)
        option_count = sum(int(row[3]) for row in rows)
        eligible_count = sum(int(row[4]) for row in rows)
        positive_count = sum(int(row[5]) for row in rows)
        expected_anchors = sum(
            anchor.role in ROUND73_DEVELOPMENT_ROLES for anchor in context.anchors
        )
        if (
            selected_anchor_count != expected_anchors
            or option_count
            != expected_anchors * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
        ):
            raise ValueError("Round 73 development target aggregate differs")
        role_run_hash = _stream_hash(manifest_hashes)
        identity = {
            "schema_version": ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION,
            "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
            "study_id": selected_study,
            "cohort_manifest_sha256": context.study_manifest_sha256,
            "source_run_count": len(context.sources),
            "selected_anchor_count": selected_anchor_count,
            "option_count": option_count,
            "eligible_option_count": eligible_count,
            "positive_option_count": positive_count,
            "role_run_manifests_sha256": role_run_hash,
            "test_target_rows_exist": False,
            "pretest_model_manifest_exists": False,
            "model_evaluated": False,
            "profitability_claim": False,
            "trading_authority": False,
        }
        manifest_text = _canonical_json(identity)
        manifest_sha = _sha256_text(manifest_text)
        connection.execute("BEGIN TRANSACTION")
        try:
            existing = connection.execute(
                f"SELECT manifest_json, manifest_sha256 "
                f"FROM {ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE} "
                "WHERE study_id = ?",
                [selected_study],
            ).fetchone()
            if existing is None:
                connection.execute(
                    f"INSERT INTO {ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE} "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected_study,
                        ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION,
                        ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                        context.study_manifest_sha256,
                        manifest_text,
                        manifest_sha,
                        role_run_hash,
                        len(context.sources),
                        selected_anchor_count,
                        option_count,
                        eligible_count,
                        positive_count,
                        time.time_ns(),
                    ],
                )
            elif existing != (manifest_text, manifest_sha):
                raise ValueError("Round 73 development study seal differs")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73DevelopmentTargetStudyReport(
        study_id=selected_study,
        source_run_count=len(context.sources),
        selected_anchor_count=selected_anchor_count,
        option_count=option_count,
        eligible_option_count=eligible_count,
        positive_option_count=positive_count,
        role_run_manifests_sha256=role_run_hash,
        development_study_manifest_sha256=manifest_sha,
    )


def _development_seal_identity(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
) -> tuple[str, str]:
    row = connection.execute(
        f"SELECT cohort_manifest_sha256, manifest_sha256, manifest_json, "
        f"schema_version, contract_sha256 FROM "
        f"{ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE} WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 development target study is not sealed")
    cohort_hash = str(row[0])
    manifest_hash = str(row[1])
    manifest_text = str(row[2])
    if (
        str(row[3]) != ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION
        or str(row[4]) != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or _SHA256.fullmatch(cohort_hash) is None
        or _SHA256.fullmatch(manifest_hash) is None
        or _sha256_text(manifest_text) != manifest_hash
    ):
        raise ValueError("Round 73 development target seal identity differs")
    return cohort_hash, manifest_hash


def _pretest_storage_identity(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
) -> tuple[Mapping[str, object], str, str, int, int]:
    row = connection.execute(
        f"SELECT schema_version, contract_sha256, "
        "development_study_manifest_sha256, pretest_manifest_json, "
        "pretest_manifest_sha256, artifact_manifest_sha256, artifact_count, "
        f"artifact_bytes FROM {ROUND73_PRETEST_MODEL_MANIFEST_TABLE} "
        "WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 pretest model manifest is missing")
    manifest_text = str(row[3])
    manifest_hash = str(row[4])
    artifact_manifest_hash = str(row[5])
    artifact_count = int(row[6])
    artifact_bytes = int(row[7])
    identity = _strict_json_object(manifest_text, "Round 73 pretest manifest")
    if (
        str(row[0]) != ROUND73_PRETEST_MODEL_SCHEMA_VERSION
        or str(row[1]) != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or identity.get("schema_version") != ROUND73_PRETEST_MODEL_SCHEMA_VERSION
        or identity.get("contract_sha256") != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or identity.get("study_id") != study_id
        or identity.get("development_study_manifest_sha256") != str(row[2])
        or _sha256_text(manifest_text) != manifest_hash
        or identity.get("artifact_manifest_sha256") != artifact_manifest_hash
        or identity.get("artifact_count") != artifact_count
        or identity.get("artifact_bytes") != artifact_bytes
    ):
        raise ValueError("Round 73 pretest model manifest identity differs")
    rows = connection.execute(
        f"SELECT artifact_name, artifact_kind, symbol, media_type, "
        f"artifact_sha256, byte_count, payload FROM "
        f"{ROUND73_PRETEST_MODEL_ARTIFACT_TABLE} WHERE study_id = ? "
        "ORDER BY artifact_name",
        [study_id],
    ).fetchall()
    metadata: list[dict[str, object]] = []
    observed_bytes = 0
    for artifact in rows:
        payload = bytes(artifact[6])
        byte_count = int(artifact[5])
        artifact_hash = str(artifact[4])
        if (
            not payload
            or len(payload) != byte_count
            or _sha256_bytes(payload) != artifact_hash
        ):
            raise ValueError("Round 73 pretest model artifact bytes differ")
        observed_bytes += byte_count
        metadata.append(
            {
                "artifact_name": str(artifact[0]),
                "artifact_kind": str(artifact[1]),
                "symbol": None if artifact[2] is None else str(artifact[2]),
                "media_type": str(artifact[3]),
                "artifact_sha256": artifact_hash,
                "byte_count": byte_count,
            }
        )
    observed_artifact_hash = _sha256_text(_canonical_json(metadata))
    if (
        metadata != identity.get("artifacts")
        or len(metadata) != artifact_count
        or observed_bytes != artifact_bytes
        or observed_artifact_hash != artifact_manifest_hash
    ):
        raise ValueError("Round 73 pretest artifact manifest differs")
    return (
        identity,
        manifest_hash,
        artifact_manifest_hash,
        artifact_count,
        artifact_bytes,
    )


def seal_round73_test_targets(
    database: str | Path,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73TestTargetStudyReport:
    """Seal complete, deeply replayed test targets without exposing outcomes."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_pretest = str(pretest_manifest_sha256).strip().lower()
    if _SHA256.fullmatch(selected_pretest) is None:
        raise ValueError("Round 73 pretest manifest hash is invalid")
    development = seal_round73_development_targets(
        database,
        study_id=selected_study,
        memory_limit=memory_limit,
        threads=threads,
        cohort_audit_function=cohort_audit_function,
        corpus_audit_function=corpus_audit_function,
        grid_audit_function=grid_audit_function,
        replay_function=replay_function,
    )
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        context = _load_study_context(store.connect(), study_id=selected_study)
        source_run_ids = tuple(source.run_id for source in context.sources)
    for run_id in source_run_ids:
        audit = audit_round73_role_targets(
            database,
            study_id=selected_study,
            run_id=run_id,
            role_scope="test",
            pretest_manifest_sha256=selected_pretest,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=cohort_audit_function,
            corpus_audit_function=corpus_audit_function,
            grid_audit_function=grid_audit_function,
            replay_function=replay_function,
            deep_replay=True,
        )
        if not audit.passed or not audit.deep_replay_performed:
            raise ValueError(
                f"Round 73 test run audit failed: {run_id}: " + "; ".join(audit.errors)
            )
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_v3_tables(connection)
        _assert_v3_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            context = _load_study_context(connection, study_id=selected_study)
            _require_prospective_context(context)
            _reject_v2_contamination(connection, selected_study)
            _reject_orphan_v3_options(connection, study_id=selected_study)
            cohort_hash, development_hash = _development_seal_identity(
                connection, study_id=selected_study
            )
            if development_hash != development.development_study_manifest_sha256:
                raise ValueError("Round 73 development seal changed before test seal")
            pretest = _pretest_storage_identity(connection, study_id=selected_study)
            if pretest[1] != selected_pretest:
                raise ValueError("Round 73 test seal pretest identity differs")
            _require_test_unlock(
                connection,
                study_id=selected_study,
                pretest_manifest_sha256=selected_pretest,
            )
            rows = connection.execute(
                f"SELECT run_id, target_manifest_sha256, selected_anchor_count, "
                "option_count, eligible_option_count, positive_option_count "
                f"FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                "WHERE study_id = ? AND role_scope = 'test' ORDER BY run_id",
                [selected_study],
            ).fetchall()
            expected_runs = tuple(sorted(source.run_id for source in context.sources))
            if tuple(str(row[0]) for row in rows) != expected_runs:
                raise ValueError("Round 73 test role manifest runs differ")
            selected_anchor_count = sum(int(row[2]) for row in rows)
            option_count = sum(int(row[3]) for row in rows)
            eligible_count = sum(int(row[4]) for row in rows)
            positive_count = sum(int(row[5]) for row in rows)
            expected_anchors = sum(
                anchor.role in ROUND73_TEST_ROLES for anchor in context.anchors
            )
            if (
                selected_anchor_count != expected_anchors
                or option_count
                != expected_anchors * ROUND73_TARGET_V2_EXPECTED_OPTIONS_PER_ANCHOR
                or _count_role_option_rows(
                    connection,
                    study_id=selected_study,
                    roles=ROUND73_TEST_ROLES,
                )
                != option_count
            ):
                raise ValueError("Round 73 test target aggregate differs")
            role_run_hash = _stream_hash([str(row[1]) for row in rows])
            identity = {
                "schema_version": ROUND73_TEST_STUDY_V3_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "study_id": selected_study,
                "cohort_manifest_sha256": cohort_hash,
                "development_study_manifest_sha256": development_hash,
                "pretest_manifest_sha256": selected_pretest,
                "source_run_count": len(context.sources),
                "selected_anchor_count": selected_anchor_count,
                "option_count": option_count,
                "eligible_option_count": eligible_count,
                "positive_option_count": positive_count,
                "role_run_manifests_sha256": role_run_hash,
                "test_exact_wire_audit_passed": True,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
            manifest_text = _canonical_json(identity)
            manifest_hash = _sha256_text(manifest_text)
            existing = connection.execute(
                f"SELECT manifest_json, manifest_sha256 FROM "
                f"{ROUND73_TARGET_V3_TEST_STUDY_TABLE} WHERE study_id = ?",
                [selected_study],
            ).fetchone()
            if existing is None:
                connection.execute(
                    f"INSERT INTO {ROUND73_TARGET_V3_TEST_STUDY_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected_study,
                        ROUND73_TEST_STUDY_V3_SCHEMA_VERSION,
                        ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                        cohort_hash,
                        development_hash,
                        selected_pretest,
                        manifest_text,
                        manifest_hash,
                        role_run_hash,
                        len(context.sources),
                        selected_anchor_count,
                        option_count,
                        eligible_count,
                        positive_count,
                        time.time_ns(),
                    ],
                )
            elif existing != (manifest_text, manifest_hash):
                raise ValueError("Round 73 test target study seal differs")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73TestTargetStudyReport(
        study_id=selected_study,
        source_run_count=len(context.sources),
        selected_anchor_count=selected_anchor_count,
        option_count=option_count,
        pretest_manifest_sha256=selected_pretest,
        role_run_manifests_sha256=role_run_hash,
        test_study_manifest_sha256=manifest_hash,
    )


__all__ = [
    "ROUND73_DEVELOPMENT_STUDY_V3_SCHEMA_VERSION",
    "ROUND73_PRETEST_MODEL_ARTIFACT_TABLE",
    "ROUND73_PRETEST_MODEL_MANIFEST_TABLE",
    "ROUND73_PRETEST_MODEL_SCHEMA_VERSION",
    "ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256",
    "ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE",
    "ROUND73_TARGET_V3_OPTION_TABLE",
    "ROUND73_TARGET_V3_ROLE_RUN_TABLE",
    "ROUND73_TARGET_V3_SCHEMA_VERSION",
    "ROUND73_TARGET_V3_SCOPES",
    "ROUND73_TARGET_V3_TEST_STUDY_TABLE",
    "ROUND73_TEST_STUDY_V3_SCHEMA_VERSION",
    "ROUND73_TEST_UNLOCK_SCHEMA_VERSION",
    "ROUND73_TEST_UNLOCK_TABLE",
    "Round73DevelopmentTargetStudyReport",
    "Round73RoleTargetAudit",
    "Round73RoleTargetBuildReport",
    "Round73TestTargetStudyReport",
    "audit_round73_role_targets",
    "build_round73_role_targets",
    "seal_round73_development_targets",
    "seal_round73_test_targets",
]
