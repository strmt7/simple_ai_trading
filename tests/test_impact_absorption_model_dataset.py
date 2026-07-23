from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

from simple_ai_trading.impact_absorption_grid import ROUND73_GRID_FEATURE_NAMES
from simple_ai_trading.impact_absorption_model_dataset import (
    ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
    ROUND73_PRE_ENTRY_ABORT_STATUS,
    ROUND73_RIGHT_CENSORED_STATUS,
    build_round73_operational_dataset,
    build_round73_staged_operational_dataset,
    classify_round73_operational_outcome,
)


STUDY_ID = "a" * 32
RUN_ID = "b" * 32
COHORT_SHA = "c" * 64
TARGET_STUDY_SHA = "d" * 64
DEVELOPMENT_STUDY_SHA = "e" * 64
TEST_STUDY_SHA = "f" * 64


def _vector_sha(values: np.ndarray, anchor_index: int) -> str:
    identity = f"{RUN_ID}:BTCUSDT:{anchor_index}:".encode("ascii")
    payload = struct.pack(f"<{len(values)}d", *values)
    return hashlib.sha256(identity + payload).hexdigest()


def _raw_features(offset: float) -> np.ndarray:
    values = np.arange(1, len(ROUND73_GRID_FEATURE_NAMES) + 1, dtype=np.float64)
    values += offset
    for index, name in enumerate(ROUND73_GRID_FEATURE_NAMES):
        if name.endswith("buyer_taker_share"):
            values[index] = 0.8
    return values


def _seed(database: Path) -> None:
    base_wall = 1_800_000_000_000_000_000
    coverage_end = base_wall + 1_000_000_000_000
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE impact_shock_study_v1 "
            "(study_id VARCHAR, manifest_sha256 VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE impact_target_study_manifest_v2 "
            "(study_id VARCHAR, target_study_manifest_sha256 VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE impact_corpus_run_manifest_v3 "
            "(run_id VARCHAR, coverage_end_wall_ns UBIGINT)"
        )
        connection.execute(
            """
            CREATE TABLE impact_shock_anchor_v1 (
                study_id VARCHAR, run_id VARCHAR, symbol VARCHAR,
                anchor_index UINTEGER, anchor_monotonic_ns UBIGINT,
                anchor_wall_ns UBIGINT, utc_day VARCHAR, role VARCHAR,
                shock_ratio DOUBLE, shock_direction TINYINT,
                shock_direction_taker_share DOUBLE,
                feature_vector_sha256 VARCHAR,
                selected_anchor_sha256 VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE impact_feature_vector_v4 (
                run_id VARCHAR, symbol VARCHAR, anchor_index UINTEGER,
                feature_values DOUBLE[], vector_sha256 VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE impact_target_option_v2 (
                study_id VARCHAR, run_id VARCHAR, symbol VARCHAR,
                anchor_index UINTEGER, entry_delay_ms USMALLINT,
                horizon_ms UINTEGER, reference_quote_notional UINTEGER,
                side VARCHAR, eligible BOOLEAN,
                ineligible_reasons_json VARCHAR,
                positive_net_payoff BOOLEAN, net_payoff_bps DOUBLE,
                option_sha256 VARCHAR, selected_anchor_sha256 VARCHAR,
                cohort_option_sha256 VARCHAR
            )
            """
        )
        connection.execute(
            "INSERT INTO impact_shock_study_v1 VALUES (?, ?)",
            [STUDY_ID, COHORT_SHA],
        )
        connection.execute(
            "INSERT INTO impact_target_study_manifest_v2 VALUES (?, ?)",
            [STUDY_ID, TARGET_STUDY_SHA],
        )
        connection.execute(
            "INSERT INTO impact_corpus_run_manifest_v3 VALUES (?, ?)",
            [RUN_ID, coverage_end],
        )
        specifications = (
            (0, base_wall + 100_000_000_000, "training", True, "[]", True, 3.0),
            (
                1,
                base_wall + 200_000_000_000,
                "training",
                False,
                '["entry_capacity"]',
                None,
                None,
            ),
            (
                2,
                base_wall + 300_000_000_000,
                "tuning",
                False,
                '["path_capacity"]',
                None,
                None,
            ),
            (3, coverage_end - 60_000_000_000, "test", True, "[]", True, 2.0),
        )
        for anchor_index, wall, role, eligible, reason, positive, net in specifications:
            values = _raw_features(float(anchor_index))
            vector_sha = _vector_sha(values, anchor_index)
            anchor_sha = hashlib.sha256(f"anchor-{anchor_index}".encode()).hexdigest()
            connection.execute(
                "INSERT INTO impact_shock_anchor_v1 VALUES "
                "(?, ?, 'BTCUSDT', ?, ?, ?, '2027-01-15', ?, 5.0, 1, 0.8, ?, ?)",
                [
                    STUDY_ID,
                    RUN_ID,
                    anchor_index,
                    1_000_000_000_000 + anchor_index * 100_000_000_000,
                    wall,
                    role,
                    vector_sha,
                    anchor_sha,
                ],
            )
            connection.execute(
                "INSERT INTO impact_feature_vector_v4 VALUES (?, 'BTCUSDT', ?, ?, ?)",
                [RUN_ID, anchor_index, values.tolist(), vector_sha],
            )
            for side_index, side in enumerate(("long", "short")):
                option_sha = hashlib.sha256(
                    f"option-{anchor_index}-{side}".encode()
                ).hexdigest()
                cohort_option_sha = hashlib.sha256(
                    f"cohort-option-{anchor_index}-{side}".encode()
                ).hexdigest()
                connection.execute(
                    "INSERT INTO impact_target_option_v2 VALUES "
                    "(?, ?, 'BTCUSDT', ?, 500, 60000, 1000, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        STUDY_ID,
                        RUN_ID,
                        anchor_index,
                        side,
                        eligible,
                        reason,
                        positive,
                        net,
                        option_sha,
                        anchor_sha,
                        cohort_option_sha,
                    ],
                )
        connection.execute(
            "CREATE TABLE impact_target_option_v3 AS "
            "SELECT * FROM impact_target_option_v2"
        )
        connection.execute(
            "CREATE TABLE impact_target_development_study_manifest_v3 "
            "(study_id VARCHAR, cohort_manifest_sha256 VARCHAR, "
            "manifest_sha256 VARCHAR)"
        )
        connection.execute(
            "INSERT INTO impact_target_development_study_manifest_v3 VALUES (?, ?, ?)",
            [STUDY_ID, COHORT_SHA, DEVELOPMENT_STUDY_SHA],
        )
        connection.execute(
            "CREATE TABLE impact_target_test_study_manifest_v3 "
            "(study_id VARCHAR, cohort_manifest_sha256 VARCHAR, "
            "manifest_sha256 VARCHAR)"
        )
        connection.execute(
            "INSERT INTO impact_target_test_study_manifest_v3 VALUES (?, ?, ?)",
            [STUDY_ID, COHORT_SHA, TEST_STUDY_SHA],
        )


def _audit(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        passed=True,
        target_study_manifest_sha256=TARGET_STUDY_SHA,
    )


def test_operational_outcome_mapping_does_not_hide_execution_failure() -> None:
    pre_entry = classify_round73_operational_outcome(
        eligible=False,
        ineligible_reasons_json='["entry_state_late"]',
        positive_net_payoff=None,
        net_payoff_bps=None,
        deterministically_boundary_censored=False,
    )
    assert pre_entry.status == ROUND73_PRE_ENTRY_ABORT_STATUS
    assert pre_entry.binary_target == 0.0
    assert pre_entry.continuous_target_bps == 0.0
    unresolved = classify_round73_operational_outcome(
        eligible=False,
        ineligible_reasons_json='["exit_capacity"]',
        positive_net_payoff=None,
        net_payoff_bps=None,
        deterministically_boundary_censored=False,
    )
    assert unresolved.status == ROUND73_POST_ENTRY_UNRESOLVED_STATUS
    assert np.isnan(unresolved.binary_target)
    assert np.isnan(unresolved.continuous_target_bps)
    with pytest.raises(ValueError, match="outside deterministic censoring"):
        classify_round73_operational_outcome(
            eligible=False,
            ineligible_reasons_json='["coverage_end"]',
            positive_net_payoff=None,
            net_payoff_bps=None,
            deterministically_boundary_censored=False,
        )


def test_operational_dataset_is_audited_hash_bound_and_status_complete(
    tmp_path: Path,
) -> None:
    database = tmp_path / "round73-model-dataset.duckdb"
    _seed(database)
    first = build_round73_operational_dataset(
        database,
        study_id=STUDY_ID,
        target_study_audit_function=_audit,
    )
    second = build_round73_operational_dataset(
        database,
        study_id=STUDY_ID,
        target_study_audit_function=_audit,
    )

    assert first.dataset_sha256 == second.dataset_sha256
    assert first.rows == 8
    assert np.bincount(first.outcome_status, minlength=4).tolist() == [2, 2, 2, 2]
    assert np.count_nonzero(first.model_label_mask) == 4
    assert np.count_nonzero(first.completed_transaction) == 2
    assert first.outcome_status[-1] == ROUND73_RIGHT_CENSORED_STATUS
    assert np.isnan(first.binary_target[-1])
    assert not first.feature_values.flags.writeable
    first.validate()

    with duckdb.connect(str(database)) as connection:
        values = connection.execute(
            "SELECT feature_values FROM impact_feature_vector_v4 WHERE anchor_index = 0"
        ).fetchone()[0]
        values[0] += 1.0
        connection.execute(
            "UPDATE impact_feature_vector_v4 SET feature_values = ? "
            "WHERE anchor_index = 0",
            [values],
        )
    with pytest.raises(ValueError, match="source row identity"):
        build_round73_operational_dataset(
            database,
            study_id=STUDY_ID,
            target_study_audit_function=_audit,
        )


def test_staged_operational_dataset_never_crosses_role_scope(tmp_path: Path) -> None:
    database = tmp_path / "round73-staged-model-dataset.duckdb"
    _seed(database)

    development = build_round73_staged_operational_dataset(
        database,
        study_id=STUDY_ID,
        role_scope="development",
        development_seal_function=lambda *_args, **_kwargs: SimpleNamespace(
            development_study_manifest_sha256=DEVELOPMENT_STUDY_SHA
        ),
    )
    assert development.rows == 6
    assert set(development.role) == {"training", "tuning"}
    assert "test" not in development.role

    test = build_round73_staged_operational_dataset(
        database,
        study_id=STUDY_ID,
        role_scope="test",
        pretest_manifest_sha256="1" * 64,
        test_seal_function=lambda *_args, **_kwargs: SimpleNamespace(
            test_study_manifest_sha256=TEST_STUDY_SHA
        ),
    )
    assert test.rows == 2
    assert set(test.role) == {"test"}
    assert test.outcome_status.tolist() == [
        ROUND73_RIGHT_CENSORED_STATUS,
        ROUND73_RIGHT_CENSORED_STATUS,
    ]
    with pytest.raises(ValueError, match="requires a pretest hash"):
        build_round73_staged_operational_dataset(
            database,
            study_id=STUDY_ID,
            role_scope="test",
            test_seal_function=lambda *_args, **_kwargs: SimpleNamespace(
                test_study_manifest_sha256=TEST_STUDY_SHA
            ),
        )


def test_operational_dataset_requires_a_passing_complete_study_audit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "round73-model-audit.duckdb"
    _seed(database)

    def failed(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(passed=False, target_study_manifest_sha256="")

    with pytest.raises(ValueError, match="did not pass"):
        build_round73_operational_dataset(
            database,
            study_id=STUDY_ID,
            target_study_audit_function=failed,
        )
