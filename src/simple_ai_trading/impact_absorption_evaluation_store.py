"""Append-only access, prediction, and result governance for Round 73."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

import duckdb

from .impact_absorption_model_features import ROUND73_EVALUATION_CONTRACT_SHA256
from .impact_absorption_pretest_contract import (
    round73_repository_state,
    validated_repository_state,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_store_v2 import _STUDY_ID
from .impact_absorption_target_store_v3 import (
    ROUND73_PRETEST_MODEL_ARTIFACT_TABLE,
    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
    ROUND73_TARGET_V3_TEST_STUDY_TABLE,
    ROUND73_TEST_STUDY_V3_SCHEMA_VERSION,
    _assert_v3_table_shapes,
    _create_v3_tables,
    _pretest_storage_identity,
)


ROUND73_EVALUATION_ACCESS_SCHEMA_VERSION = "round-073-evaluation-access-v1"
ROUND73_EVALUATION_RESULT_SCHEMA_VERSION = "round-073-model-evaluation-v1"
ROUND73_EVALUATION_PREDICTION_SCHEMA_VERSION = "round-073-test-prediction-artifact-v1"
ROUND73_EVALUATION_ACCESS_TABLE = "impact_model_evaluation_access_v1"
ROUND73_EVALUATION_PREDICTION_TABLE = "impact_model_test_prediction_v1"
ROUND73_EVALUATION_RESULT_TABLE = "impact_model_evaluation_result_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40,64}")
_RESULT_STATUSES = frozenset({"passed", "failed", "interrupted"})

RepositoryStateFunction = Callable[[str | Path], Mapping[str, object]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("ascii"))


def _strict_json_object(value: str, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be an object")
    return parsed


@dataclass(frozen=True)
class Round73EvaluationAccessClaim:
    study_id: str
    pretest_manifest_sha256: str
    test_study_manifest_sha256: str
    repository_commit_sha: str
    repository_tree_sha: str
    claimed_at_wall_ns: int

    def as_dict(self) -> dict[str, object]:
        output = asdict(self)
        output.update(
            {
                "schema_version": ROUND73_EVALUATION_ACCESS_SCHEMA_VERSION,
                "staged_holdout_contract_sha256": (
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
                ),
                "evaluation_contract_sha256": (ROUND73_EVALUATION_CONTRACT_SHA256),
                "test_read_count": 1,
                "second_evaluation_permitted": False,
                "trading_authority": False,
            }
        )
        return output


@dataclass(frozen=True)
class Round73StoredEvaluationResult:
    study_id: str
    status: str
    result_sha256: str
    recorded_at_wall_ns: int

    def as_dict(self) -> dict[str, object]:
        output = asdict(self)
        output.update(
            {
                "schema_version": ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
                "staged_holdout_contract_sha256": (
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
                ),
                "evaluation_contract_sha256": (ROUND73_EVALUATION_CONTRACT_SHA256),
                "result_is_append_only": True,
                "trading_authority": False,
            }
        )
        return output


def _create_evaluation_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_EVALUATION_ACCESS_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            staged_holdout_contract_sha256 VARCHAR NOT NULL,
            evaluation_contract_sha256 VARCHAR NOT NULL,
            pretest_manifest_sha256 VARCHAR NOT NULL,
            test_study_manifest_sha256 VARCHAR NOT NULL,
            repository_commit_sha VARCHAR NOT NULL,
            repository_tree_sha VARCHAR NOT NULL,
            claimed_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(study_id) = 32),
            CHECK (length(staged_holdout_contract_sha256) = 64),
            CHECK (length(evaluation_contract_sha256) = 64),
            CHECK (length(pretest_manifest_sha256) = 64),
            CHECK (length(test_study_manifest_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_EVALUATION_PREDICTION_TABLE} (
            study_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            schema_version VARCHAR NOT NULL,
            source_rows_sha256 VARCHAR NOT NULL,
            artifact_sha256 VARCHAR NOT NULL,
            byte_count UBIGINT NOT NULL,
            payload BLOB NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            PRIMARY KEY (study_id, symbol),
            CHECK (length(study_id) = 32),
            CHECK (length(source_rows_sha256) = 64),
            CHECK (length(artifact_sha256) = 64),
            CHECK (byte_count > 0)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_EVALUATION_RESULT_TABLE} (
            study_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            staged_holdout_contract_sha256 VARCHAR NOT NULL,
            evaluation_contract_sha256 VARCHAR NOT NULL,
            pretest_manifest_sha256 VARCHAR NOT NULL,
            test_study_manifest_sha256 VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            result_sha256 VARCHAR NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(study_id) = 32),
            CHECK (length(staged_holdout_contract_sha256) = 64),
            CHECK (length(evaluation_contract_sha256) = 64),
            CHECK (length(pretest_manifest_sha256) = 64),
            CHECK (length(test_study_manifest_sha256) = 64),
            CHECK (length(result_sha256) = 64),
            CHECK (status IN ('passed', 'failed', 'interrupted'))
        )
        """
    )


def _assert_evaluation_table_shapes(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    expected = {
        ROUND73_EVALUATION_ACCESS_TABLE: (
            "study_id",
            "schema_version",
            "staged_holdout_contract_sha256",
            "evaluation_contract_sha256",
            "pretest_manifest_sha256",
            "test_study_manifest_sha256",
            "repository_commit_sha",
            "repository_tree_sha",
            "claimed_at_wall_ns",
        ),
        ROUND73_EVALUATION_PREDICTION_TABLE: (
            "study_id",
            "symbol",
            "schema_version",
            "source_rows_sha256",
            "artifact_sha256",
            "byte_count",
            "payload",
            "recorded_at_wall_ns",
        ),
        ROUND73_EVALUATION_RESULT_TABLE: (
            "study_id",
            "schema_version",
            "staged_holdout_contract_sha256",
            "evaluation_contract_sha256",
            "pretest_manifest_sha256",
            "test_study_manifest_sha256",
            "status",
            "result_json",
            "result_sha256",
            "recorded_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 evaluation table schema differs: {table}")


def _test_study_identity(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
) -> tuple[Mapping[str, object], str]:
    row = connection.execute(
        f"SELECT schema_version, contract_sha256, pretest_manifest_sha256, "
        f"manifest_json, manifest_sha256 FROM {ROUND73_TARGET_V3_TEST_STUDY_TABLE} "
        "WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 complete test target seal is missing")
    manifest_text = str(row[3])
    manifest_hash = str(row[4])
    identity = _strict_json_object(manifest_text, "Round 73 test target seal")
    if (
        str(row[0]) != ROUND73_TEST_STUDY_V3_SCHEMA_VERSION
        or str(row[1]) != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or str(row[2]) != pretest_manifest_sha256
        or identity.get("schema_version") != ROUND73_TEST_STUDY_V3_SCHEMA_VERSION
        or identity.get("contract_sha256") != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or identity.get("study_id") != study_id
        or identity.get("pretest_manifest_sha256") != pretest_manifest_sha256
        or _SHA256.fullmatch(manifest_hash) is None
        or _sha256_text(manifest_text) != manifest_hash
    ):
        raise ValueError("Round 73 test target seal identity differs")
    return identity, manifest_hash


def claim_round73_evaluation_access(
    database: str | Path,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
    repository_root: str | Path,
    memory_limit: str = "2GB",
    threads: int = 2,
    repository_state_function: RepositoryStateFunction = round73_repository_state,
) -> Round73EvaluationAccessClaim:
    """Consume the study's single test-evaluation access before target loading."""

    selected_study = str(study_id).strip().lower()
    selected_pretest = str(pretest_manifest_sha256).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 evaluation study ID is invalid")
    if _SHA256.fullmatch(selected_pretest) is None:
        raise ValueError("Round 73 evaluation pretest hash is invalid")
    repository = validated_repository_state(repository_state_function(repository_root))
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_v3_tables(connection)
        _assert_v3_table_shapes(connection)
        _create_evaluation_tables(connection)
        _assert_evaluation_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            pretest = _pretest_storage_identity(
                connection,
                study_id=selected_study,
            )
            pretest_identity = pretest[0]
            if (
                pretest[1] != selected_pretest
                or pretest_identity.get("repository") != repository
            ):
                raise ValueError("Round 73 pretest identity drifted before evaluation")
            _, test_manifest_hash = _test_study_identity(
                connection,
                study_id=selected_study,
                pretest_manifest_sha256=selected_pretest,
            )
            existing_access = int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_EVALUATION_ACCESS_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            )
            existing_result = int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_EVALUATION_RESULT_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            )
            existing_predictions = int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_EVALUATION_PREDICTION_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            )
            if existing_access or existing_result or existing_predictions:
                raise ValueError(
                    "Round 73 one-use evaluation access was already consumed"
                )
            claimed_at = time.time_ns()
            connection.execute(
                f"INSERT INTO {ROUND73_EVALUATION_ACCESS_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    selected_study,
                    ROUND73_EVALUATION_ACCESS_SCHEMA_VERSION,
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                    ROUND73_EVALUATION_CONTRACT_SHA256,
                    selected_pretest,
                    test_manifest_hash,
                    repository["commit_sha"],
                    repository["tree_sha"],
                    claimed_at,
                ],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73EvaluationAccessClaim(
        study_id=selected_study,
        pretest_manifest_sha256=selected_pretest,
        test_study_manifest_sha256=test_manifest_hash,
        repository_commit_sha=str(repository["commit_sha"]),
        repository_tree_sha=str(repository["tree_sha"]),
        claimed_at_wall_ns=claimed_at,
    )


def _validated_access_row(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
) -> Round73EvaluationAccessClaim:
    row = connection.execute(
        f"SELECT schema_version, staged_holdout_contract_sha256, "
        "evaluation_contract_sha256, pretest_manifest_sha256, "
        "test_study_manifest_sha256, repository_commit_sha, "
        f"repository_tree_sha, claimed_at_wall_ns FROM "
        f"{ROUND73_EVALUATION_ACCESS_TABLE} WHERE study_id = ?",
        [study_id],
    ).fetchone()
    if row is None:
        raise ValueError("Round 73 evaluation access claim is missing")
    if (
        str(row[0]) != ROUND73_EVALUATION_ACCESS_SCHEMA_VERSION
        or str(row[1]) != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or str(row[2]) != ROUND73_EVALUATION_CONTRACT_SHA256
        or _SHA256.fullmatch(str(row[3])) is None
        or _SHA256.fullmatch(str(row[4])) is None
        or _GIT_OBJECT.fullmatch(str(row[5])) is None
        or _GIT_OBJECT.fullmatch(str(row[6])) is None
    ):
        raise ValueError("Round 73 evaluation access identity differs")
    return Round73EvaluationAccessClaim(
        study_id=study_id,
        pretest_manifest_sha256=str(row[3]),
        test_study_manifest_sha256=str(row[4]),
        repository_commit_sha=str(row[5]),
        repository_tree_sha=str(row[6]),
        claimed_at_wall_ns=int(row[7]),
    )


def load_round73_claimed_pretest(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> tuple[Round73EvaluationAccessClaim, Mapping[str, object]]:
    """Revalidate the claim and every frozen artifact without target access."""

    selected_study = str(study_id).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 evaluation study ID is invalid")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _assert_v3_table_shapes(connection)
        _assert_evaluation_table_shapes(connection)
        claim = _validated_access_row(connection, study_id=selected_study)
        if connection.execute(
            f"SELECT count(*) FROM {ROUND73_EVALUATION_RESULT_TABLE} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()[0]:
            raise ValueError("Round 73 evaluation already has a terminal result")
        pretest = _pretest_storage_identity(connection, study_id=selected_study)
        if pretest[1] != claim.pretest_manifest_sha256:
            raise ValueError("Round 73 claimed pretest identity differs")
        _, test_hash = _test_study_identity(
            connection,
            study_id=selected_study,
            pretest_manifest_sha256=claim.pretest_manifest_sha256,
        )
        if test_hash != claim.test_study_manifest_sha256:
            raise ValueError("Round 73 claimed test seal identity differs")
        return claim, pretest[0]


def load_round73_claimed_symbol_artifacts(
    database: str | Path,
    *,
    study_id: str,
    symbol: str,
    artifact_names: Mapping[str, object],
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Mapping[str, bytes]:
    """Load a symbol's model artifacts only after the access claim exists."""

    selected_study = str(study_id).strip().lower()
    selected_symbol = str(symbol).strip().upper()
    if (
        _STUDY_ID.fullmatch(selected_study) is None
        or selected_symbol not in IMPACT_CAPTURE_SYMBOLS
        or not isinstance(artifact_names, Mapping)
    ):
        raise ValueError("Round 73 claimed artifact request is invalid")
    requested = {
        str(kind): str(name).strip().lower()
        for kind, name in artifact_names.items()
        if str(kind) in {"model", "preprocessor"}
    }
    if set(requested) != {"model", "preprocessor"}:
        raise ValueError("Round 73 model and preprocessor artifacts are required")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _assert_v3_table_shapes(connection)
        _assert_evaluation_table_shapes(connection)
        claim = _validated_access_row(connection, study_id=selected_study)
        if connection.execute(
            f"SELECT count(*) FROM {ROUND73_EVALUATION_RESULT_TABLE} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()[0]:
            raise ValueError("Round 73 evaluation already has a terminal result")
        pretest = _pretest_storage_identity(connection, study_id=selected_study)
        if pretest[1] != claim.pretest_manifest_sha256:
            raise ValueError("Round 73 claimed pretest identity differs")
        rows = connection.execute(
            f"SELECT artifact_name, artifact_sha256, byte_count, payload FROM "
            f"{ROUND73_PRETEST_MODEL_ARTIFACT_TABLE} WHERE study_id = ? "
            "AND symbol = ? AND artifact_name IN (?, ?) ORDER BY artifact_name",
            [
                selected_study,
                selected_symbol,
                requested["model"],
                requested["preprocessor"],
            ],
        ).fetchall()
    payload_by_name: dict[str, bytes] = {}
    for row in rows:
        name = str(row[0])
        payload = bytes(row[3])
        if (
            name not in set(requested.values())
            or len(payload) != int(row[2])
            or _sha256_bytes(payload) != str(row[1])
        ):
            raise ValueError("Round 73 claimed model artifact differs")
        payload_by_name[name] = payload
    if set(payload_by_name) != set(requested.values()):
        raise ValueError("Round 73 claimed model artifacts are incomplete")
    return {kind: payload_by_name[name] for kind, name in requested.items()}


def persist_round73_test_prediction(
    database: str | Path,
    *,
    study_id: str,
    symbol: str,
    source_rows_sha256: str,
    payload: bytes,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> str:
    """Persist one immutable, hash-bound test prediction artifact per symbol."""

    selected_study = str(study_id).strip().lower()
    selected_symbol = str(symbol).strip().upper()
    source_hash = str(source_rows_sha256).strip().lower()
    artifact = bytes(payload)
    if (
        _STUDY_ID.fullmatch(selected_study) is None
        or selected_symbol not in IMPACT_CAPTURE_SYMBOLS
        or _SHA256.fullmatch(source_hash) is None
        or not artifact
    ):
        raise ValueError("Round 73 test prediction artifact is invalid")
    artifact_hash = _sha256_bytes(artifact)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_evaluation_tables(connection)
        _assert_evaluation_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            _validated_access_row(connection, study_id=selected_study)
            if connection.execute(
                f"SELECT count(*) FROM {ROUND73_EVALUATION_RESULT_TABLE} "
                "WHERE study_id = ?",
                [selected_study],
            ).fetchone()[0]:
                raise ValueError("Round 73 evaluation already has a terminal result")
            if connection.execute(
                f"SELECT count(*) FROM {ROUND73_EVALUATION_PREDICTION_TABLE} "
                "WHERE study_id = ? AND symbol = ?",
                [selected_study, selected_symbol],
            ).fetchone()[0]:
                raise ValueError("Round 73 test prediction is immutable")
            connection.execute(
                f"INSERT INTO {ROUND73_EVALUATION_PREDICTION_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    selected_study,
                    selected_symbol,
                    ROUND73_EVALUATION_PREDICTION_SCHEMA_VERSION,
                    source_hash,
                    artifact_hash,
                    len(artifact),
                    artifact,
                    time.time_ns(),
                ],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return artifact_hash


def load_round73_test_prediction(
    database: str | Path,
    *,
    study_id: str,
    symbol: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> tuple[str, str, bytes]:
    selected_study = str(study_id).strip().lower()
    selected_symbol = str(symbol).strip().upper()
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _assert_evaluation_table_shapes(connection)
        _validated_access_row(connection, study_id=selected_study)
        row = connection.execute(
            f"SELECT schema_version, source_rows_sha256, artifact_sha256, "
            f"byte_count, payload FROM {ROUND73_EVALUATION_PREDICTION_TABLE} "
            "WHERE study_id = ? AND symbol = ?",
            [selected_study, selected_symbol],
        ).fetchone()
    if row is None:
        raise ValueError("Round 73 test prediction artifact is missing")
    payload = bytes(row[4])
    if (
        str(row[0]) != ROUND73_EVALUATION_PREDICTION_SCHEMA_VERSION
        or _SHA256.fullmatch(str(row[1])) is None
        or len(payload) != int(row[3])
        or _sha256_bytes(payload) != str(row[2])
    ):
        raise ValueError("Round 73 stored test prediction differs")
    return str(row[1]), str(row[2]), payload


def persist_round73_evaluation_result(
    database: str | Path,
    *,
    claim: Round73EvaluationAccessClaim,
    status: str,
    result: Mapping[str, object],
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73StoredEvaluationResult:
    """Append the study's sole terminal result, including failures."""

    selected_status = str(status).strip().lower()
    if selected_status not in _RESULT_STATUSES or not isinstance(result, Mapping):
        raise ValueError("Round 73 terminal evaluation result is invalid")
    if (
        result.get("schema_version") != ROUND73_EVALUATION_RESULT_SCHEMA_VERSION
        or result.get("study_id") != claim.study_id
        or result.get("status") != selected_status
        or result.get("staged_holdout_contract_sha256")
        != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or result.get("evaluation_contract_sha256")
        != ROUND73_EVALUATION_CONTRACT_SHA256
        or result.get("pretest_manifest_sha256") != claim.pretest_manifest_sha256
        or result.get("test_study_manifest_sha256") != claim.test_study_manifest_sha256
        or result.get("profitability_claim") is not False
        or result.get("trading_authority") is not False
    ):
        raise ValueError("Round 73 terminal result contract identity differs")
    result_text = _canonical_json(dict(result))
    result_hash = _sha256_text(result_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_evaluation_tables(connection)
        _assert_evaluation_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            observed = _validated_access_row(connection, study_id=claim.study_id)
            if observed != claim:
                raise ValueError("Round 73 terminal result claim identity differs")
            if connection.execute(
                f"SELECT count(*) FROM {ROUND73_EVALUATION_RESULT_TABLE} "
                "WHERE study_id = ?",
                [claim.study_id],
            ).fetchone()[0]:
                raise ValueError("Round 73 terminal evaluation result is immutable")
            recorded_at = time.time_ns()
            connection.execute(
                f"INSERT INTO {ROUND73_EVALUATION_RESULT_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    claim.study_id,
                    ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                    ROUND73_EVALUATION_CONTRACT_SHA256,
                    claim.pretest_manifest_sha256,
                    claim.test_study_manifest_sha256,
                    selected_status,
                    result_text,
                    result_hash,
                    recorded_at,
                ],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73StoredEvaluationResult(
        study_id=claim.study_id,
        status=selected_status,
        result_sha256=result_hash,
        recorded_at_wall_ns=recorded_at,
    )


def finalize_interrupted_round73_evaluation(
    database: str | Path,
    *,
    study_id: str,
    reason: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73StoredEvaluationResult:
    """Seal a consumed but interrupted evaluation without rereading test rows."""

    selected_study = str(study_id).strip().lower()
    detail = " ".join(str(reason).strip().split())[:500]
    if _STUDY_ID.fullmatch(selected_study) is None or not detail:
        raise ValueError("Round 73 interrupted evaluation reason is invalid")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _assert_evaluation_table_shapes(connection)
        claim = _validated_access_row(connection, study_id=selected_study)
    result = {
        "schema_version": ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
        "study_id": selected_study,
        "status": "interrupted",
        "staged_holdout_contract_sha256": (ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256),
        "evaluation_contract_sha256": ROUND73_EVALUATION_CONTRACT_SHA256,
        "pretest_manifest_sha256": claim.pretest_manifest_sha256,
        "test_study_manifest_sha256": claim.test_study_manifest_sha256,
        "reason": detail,
        "test_access_consumed": True,
        "evaluation_complete": False,
        "predictive_gate_passed": False,
        "economic_gate_passed": False,
        "operational_gate_passed": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    return persist_round73_evaluation_result(
        database,
        claim=claim,
        status="interrupted",
        result=result,
        memory_limit=memory_limit,
        threads=threads,
    )


__all__ = [
    "ROUND73_EVALUATION_ACCESS_SCHEMA_VERSION",
    "ROUND73_EVALUATION_ACCESS_TABLE",
    "ROUND73_EVALUATION_PREDICTION_SCHEMA_VERSION",
    "ROUND73_EVALUATION_PREDICTION_TABLE",
    "ROUND73_EVALUATION_RESULT_SCHEMA_VERSION",
    "ROUND73_EVALUATION_RESULT_TABLE",
    "Round73EvaluationAccessClaim",
    "Round73StoredEvaluationResult",
    "claim_round73_evaluation_access",
    "finalize_interrupted_round73_evaluation",
    "load_round73_claimed_pretest",
    "load_round73_claimed_symbol_artifacts",
    "load_round73_test_prediction",
    "persist_round73_evaluation_result",
    "persist_round73_test_prediction",
]
