"""Immutable model-freeze and one-time test-unlock governance for Round 73."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable

import duckdb

from .impact_absorption_cohort import audit_round73_shock_cohort
from .impact_absorption_corpus import audit_round73_corpus_manifest
from .impact_absorption_grid_store import audit_round73_causal_grid
from .impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
    ROUND73_EVALUATION_CONTRACT_SHA256,
    ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
)
from .impact_absorption_pretest_contract import (
    RepositoryStateFunction,
    round73_repository_state,
    validated_action_policy as _validated_action_policy,
    validated_compute_backend as _validated_compute_backend,
    validated_feature_schema as _validated_feature_schema,
    validated_repository_state as _validated_repository_state,
    validated_role_rows as _validated_role_rows,
    validated_symbol_models as _validated_symbol_models,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_store import (
    Round73TargetOption,
    replay_round73_target_rows_v9,
)
from .impact_absorption_target_store_v2 import (
    ROUND73_TARGET_V2_SIDES,
    _load_study_context,
    _SHA256,
    _STUDY_ID,
    _canonical_json,
    _sha256_text,
    _stream_hash,
    _validated_identifier,
)
from .impact_absorption_target_store_v3 import (
    ROUND73_DEVELOPMENT_ROLES,
    ROUND73_PRETEST_MODEL_ARTIFACT_TABLE,
    ROUND73_PRETEST_MODEL_MANIFEST_TABLE,
    ROUND73_PRETEST_MODEL_SCHEMA_VERSION,
    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
    ROUND73_TARGET_V3_OPTION_TABLE,
    ROUND73_TARGET_V3_ROLE_RUN_TABLE,
    ROUND73_TEST_ROLES,
    ROUND73_TEST_UNLOCK_SCHEMA_VERSION,
    ROUND73_TEST_UNLOCK_TABLE,
    _assert_v3_table_shapes,
    _count_role_option_rows,
    _create_v3_tables,
    _development_seal_identity,
    _pretest_storage_identity,
    _reject_orphan_v3_options,
    _reject_v2_contamination,
    _require_prospective_context,
    seal_round73_development_targets,
)
from .impact_absorption_targets import ROUND73_TARGET_MAX_STATE_LATENESS_NS


_PRIMARY_ENTRY_DELAY_MS = 500
_PRIMARY_HORIZON_MS = 60_000
_PRIMARY_REFERENCE_NOTIONAL = 1_000
_PRE_ENTRY_ABORT_REASONS = frozenset(
    {
        "quantity_filter",
        "entry_state_late",
        "entry_capacity",
        "entry_minimum_notional",
        "funding_boundary",
    }
)
_POST_ENTRY_UNRESOLVED_REASONS = frozenset(
    {"path_capacity", "exit_state_late", "exit_capacity"}
)

AuditFunction = Callable[..., object]
ReplayFunction = Callable[..., list[Round73TargetOption]]


@dataclass(frozen=True)
class Round73PretestManifestReport:
    study_id: str
    development_study_manifest_sha256: str
    pretest_manifest_sha256: str
    artifact_manifest_sha256: str
    artifact_count: int
    artifact_bytes: int

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_PRETEST_MODEL_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "test_target_rows_exist": False,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


@dataclass(frozen=True)
class Round73TestUnlockReport:
    study_id: str
    pretest_manifest_sha256: str
    unlocked_at_wall_ns: int

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ROUND73_TEST_UNLOCK_SCHEMA_VERSION,
                "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                "test_target_replay_permitted": True,
                "model_evaluated": False,
                "profitability_claim": False,
                "trading_authority": False,
            }
        )
        return payload


def _development_row_identities(
    connection: duckdb.DuckDBPyConnection,
    *,
    context: object,
) -> dict[str, dict[str, dict[str, object]]]:
    rows = connection.execute(
        f"SELECT a.role, a.symbol, a.anchor_wall_ns, o.run_id, "
        "o.anchor_index, o.side, o.eligible, o.ineligible_reasons_json, "
        f"o.cohort_option_sha256 FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
        "JOIN impact_shock_anchor_v1 a "
        "ON a.study_id = o.study_id AND a.run_id = o.run_id "
        "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
        "WHERE o.study_id = ? AND a.role IN ('training', 'tuning') "
        "AND o.entry_delay_ms = ? AND o.horizon_ms = ? "
        "AND o.reference_quote_notional = ? "
        "ORDER BY a.role, a.symbol, a.anchor_wall_ns, o.run_id, "
        "o.anchor_index, CASE o.side WHEN 'long' THEN 0 ELSE 1 END",
        [
            context.study_id,
            _PRIMARY_ENTRY_DELAY_MS,
            _PRIMARY_HORIZON_MS,
            _PRIMARY_REFERENCE_NOTIONAL,
        ],
    ).fetchall()
    expected_rows = sum(
        anchor.role in ROUND73_DEVELOPMENT_ROLES for anchor in context.anchors
    ) * len(ROUND73_TARGET_V2_SIDES)
    if len(rows) != expected_rows:
        raise ValueError("Round 73 development primary target rows are incomplete")
    hashes: dict[str, dict[str, list[str]]] = {
        role: {symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS}
        for role in ROUND73_DEVELOPMENT_ROLES
    }
    source_by_run = context.source_by_run
    known_reasons = (
        _PRE_ENTRY_ABORT_REASONS | _POST_ENTRY_UNRESOLVED_REASONS | {"coverage_end"}
    )
    for row in rows:
        role = str(row[0])
        symbol = str(row[1])
        anchor_wall_ns = int(row[2])
        run_id = str(row[3])
        eligible = bool(row[6])
        try:
            reasons = json.loads(str(row[7]))
        except json.JSONDecodeError as exc:
            raise ValueError("Round 73 development target reasons are invalid") from exc
        if (
            role not in ROUND73_DEVELOPMENT_ROLES
            or symbol not in IMPACT_CAPTURE_SYMBOLS
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
            or len(reasons) != len(set(reasons))
            or any(reason not in known_reasons for reason in reasons)
            or (eligible and reasons)
            or (not eligible and len(reasons) != 1)
            or _SHA256.fullmatch(str(row[8])) is None
        ):
            raise ValueError("Round 73 development target identity differs")
        coverage_end_wall_ns = int(source_by_run[run_id].coverage_end_wall_ns)
        required_complete_wall_ns = (
            anchor_wall_ns
            + _PRIMARY_ENTRY_DELAY_MS * 1_000_000
            + _PRIMARY_HORIZON_MS * 1_000_000
            + 2 * ROUND73_TARGET_MAX_STATE_LATENESS_NS
        )
        if required_complete_wall_ns >= coverage_end_wall_ns:
            continue
        if eligible or reasons[0] in _PRE_ENTRY_ABORT_REASONS:
            hashes[role][symbol].append(str(row[8]))
            continue
        if reasons[0] == "coverage_end":
            raise ValueError(
                "Round 73 coverage-end target is outside deterministic censoring"
            )
    return {
        role: {
            symbol: {
                "row_count": len(hashes[role][symbol]),
                "rows_sha256": _stream_hash(hashes[role][symbol]),
            }
            for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        for role in ROUND73_DEVELOPMENT_ROLES
    }


def round73_development_row_identities(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Mapping[str, object]:
    """Return exact fitting-row hashes without reading any test target."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
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
        _development_seal_identity(connection, study_id=selected_study)
        return _development_row_identities(connection, context=context)


def publish_round73_pretest_manifest(
    database: str | Path,
    *,
    study_id: str,
    model_manifest: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    repository_root: str | Path,
    memory_limit: str = "2GB",
    threads: int = 2,
    repository_state_function: RepositoryStateFunction = round73_repository_state,
    cohort_audit_function: AuditFunction = audit_round73_shock_cohort,
    corpus_audit_function: AuditFunction = audit_round73_corpus_manifest,
    grid_audit_function: AuditFunction = audit_round73_causal_grid,
    replay_function: ReplayFunction = replay_round73_target_rows_v9,
) -> Round73PretestManifestReport:
    """Freeze model, preprocessing, predictions, and policy before test replay."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
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
    repository = _validated_repository_state(repository_state_function(repository_root))
    feature_schema = _validated_feature_schema(model_manifest.get("feature_schema"))
    row_identities = model_manifest.get("row_identities")
    if not isinstance(row_identities, Mapping):
        raise ValueError("Round 73 pretest row identities are missing")
    normalized_rows = {
        role: _validated_role_rows(row_identities.get(role), role=role)
        for role in ROUND73_DEVELOPMENT_ROLES
    }
    compute_backend = _validated_compute_backend(model_manifest.get("compute_backend"))
    action_policy = _validated_action_policy(model_manifest.get("action_policy"))
    symbol_models, artifact_rows = _validated_symbol_models(
        model_manifest.get("symbol_models"), artifacts=artifacts
    )
    artifact_manifest_hash = _sha256_text(_canonical_json(artifact_rows))
    artifact_bytes = sum(int(item["byte_count"]) for item in artifact_rows)
    identity = {
        "schema_version": ROUND73_PRETEST_MODEL_SCHEMA_VERSION,
        "contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
        "study_id": selected_study,
        "cohort_manifest_sha256": None,
        "development_study_manifest_sha256": (
            development.development_study_manifest_sha256
        ),
        "contracts": {
            "staged_holdout_contract_sha256": (ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256),
            "evaluation_contract_sha256": ROUND73_EVALUATION_CONTRACT_SHA256,
            "feature_contract_sha256": ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
            "feature_names_sha256": ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
        },
        "repository": repository,
        "feature_schema": feature_schema,
        "row_identities": normalized_rows,
        "compute_backend": compute_backend,
        "symbol_models": symbol_models,
        "action_policy": action_policy,
        "artifacts": artifact_rows,
        "artifact_manifest_sha256": artifact_manifest_hash,
        "artifact_count": len(artifact_rows),
        "artifact_bytes": artifact_bytes,
        "test_feature_or_target_hash_included": False,
        "test_target_or_payoff_read_before_publish": False,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
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
                raise ValueError("Round 73 development seal changed before pretest")
            observed_rows = _development_row_identities(connection, context=context)
            if normalized_rows != observed_rows:
                raise ValueError("Round 73 pretest fitting-row identities differ")
            current_repository = _validated_repository_state(
                repository_state_function(repository_root)
            )
            if current_repository != repository:
                raise ValueError("Round 73 repository changed during model freeze")
            identity["cohort_manifest_sha256"] = cohort_hash
            if _count_role_option_rows(
                connection,
                study_id=selected_study,
                roles=ROUND73_TEST_ROLES,
            ) or int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                    "WHERE study_id = ? AND role_scope = 'test'",
                    [selected_study],
                ).fetchone()[0]
            ):
                raise ValueError("Round 73 test targets exist before model freeze")
            if int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TEST_UNLOCK_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            ):
                raise ValueError("Round 73 test was unlocked before model freeze")
            manifest_text = _canonical_json(identity)
            manifest_hash = _sha256_text(manifest_text)
            existing = connection.execute(
                f"SELECT pretest_manifest_sha256 FROM "
                f"{ROUND73_PRETEST_MODEL_MANIFEST_TABLE} WHERE study_id = ?",
                [selected_study],
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != manifest_hash:
                    raise ValueError("Round 73 pretest model manifest is immutable")
                stored = _pretest_storage_identity(connection, study_id=selected_study)
                if stored[1:] != (
                    manifest_hash,
                    artifact_manifest_hash,
                    len(artifact_rows),
                    artifact_bytes,
                ):
                    raise ValueError("Round 73 existing pretest storage differs")
            else:
                for artifact in artifact_rows:
                    name = str(artifact["artifact_name"])
                    connection.execute(
                        f"INSERT INTO {ROUND73_PRETEST_MODEL_ARTIFACT_TABLE} "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            selected_study,
                            name,
                            artifact["artifact_kind"],
                            artifact["symbol"],
                            artifact["media_type"],
                            artifact["artifact_sha256"],
                            artifact["byte_count"],
                            artifacts[name],
                        ],
                    )
                connection.execute(
                    f"INSERT INTO {ROUND73_PRETEST_MODEL_MANIFEST_TABLE} "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected_study,
                        ROUND73_PRETEST_MODEL_SCHEMA_VERSION,
                        ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                        development_hash,
                        manifest_text,
                        manifest_hash,
                        artifact_manifest_hash,
                        len(artifact_rows),
                        artifact_bytes,
                        time.time_ns(),
                    ],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73PretestManifestReport(
        study_id=selected_study,
        development_study_manifest_sha256=(
            development.development_study_manifest_sha256
        ),
        pretest_manifest_sha256=manifest_hash,
        artifact_manifest_sha256=artifact_manifest_hash,
        artifact_count=len(artifact_rows),
        artifact_bytes=artifact_bytes,
    )


def unlock_round73_test_targets(
    database: str | Path,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
    repository_root: str | Path,
    memory_limit: str = "2GB",
    threads: int = 2,
    repository_state_function: RepositoryStateFunction = round73_repository_state,
) -> Round73TestUnlockReport:
    """Append the single test unlock after revalidating every frozen byte."""

    selected_study = _validated_identifier(study_id, _STUDY_ID, "study ID")
    selected_pretest = str(pretest_manifest_sha256).strip().lower()
    if _SHA256.fullmatch(selected_pretest) is None:
        raise ValueError("Round 73 pretest manifest hash is invalid")
    repository = _validated_repository_state(repository_state_function(repository_root))
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
            pretest = _pretest_storage_identity(connection, study_id=selected_study)
            identity, observed_pretest = pretest[0], pretest[1]
            current_repository = _validated_repository_state(
                repository_state_function(repository_root)
            )
            if (
                observed_pretest != selected_pretest
                or current_repository != repository
                or identity.get("repository") != current_repository
                or identity.get("cohort_manifest_sha256") != cohort_hash
                or identity.get("development_study_manifest_sha256") != development_hash
                or identity.get("contracts")
                != {
                    "staged_holdout_contract_sha256": (
                        ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
                    ),
                    "evaluation_contract_sha256": (ROUND73_EVALUATION_CONTRACT_SHA256),
                    "feature_contract_sha256": (ROUND73_MODEL_FEATURE_CONTRACT_SHA256),
                    "feature_names_sha256": (
                        ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256
                    ),
                }
            ):
                raise ValueError("Round 73 pretest identity drifted before unlock")
            if int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TEST_UNLOCK_TABLE} "
                    "WHERE study_id = ?",
                    [selected_study],
                ).fetchone()[0]
            ):
                raise ValueError("Round 73 test unlock already exists")
            if _count_role_option_rows(
                connection,
                study_id=selected_study,
                roles=ROUND73_TEST_ROLES,
            ) or int(
                connection.execute(
                    f"SELECT count(*) FROM {ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
                    "WHERE study_id = ? AND role_scope = 'test'",
                    [selected_study],
                ).fetchone()[0]
            ):
                raise ValueError("Round 73 test targets exist before unlock")
            unlocked_at = time.time_ns()
            connection.execute(
                f"INSERT INTO {ROUND73_TEST_UNLOCK_TABLE} VALUES (?, ?, ?, ?, ?)",
                [
                    selected_study,
                    ROUND73_TEST_UNLOCK_SCHEMA_VERSION,
                    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
                    selected_pretest,
                    unlocked_at,
                ],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return Round73TestUnlockReport(
        study_id=selected_study,
        pretest_manifest_sha256=selected_pretest,
        unlocked_at_wall_ns=unlocked_at,
    )


__all__ = [
    "Round73PretestManifestReport",
    "Round73TestUnlockReport",
    "publish_round73_pretest_manifest",
    "round73_development_row_identities",
    "round73_repository_state",
    "unlock_round73_test_targets",
]
