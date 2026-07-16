"""Durable exactly-once claims for Polymarket evidence-model test access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import threading
import time
import weakref

from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_FIT_CLAIM_SCHEMA_VERSION = "polymarket-model-fit-claim-v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESERVATION_LOCK = threading.RLock()
_RESERVATIONS: weakref.WeakKeyDictionary[
    PolymarketEvidenceStore,
    dict[tuple[str, str, str, str, str, str], PolymarketFitClaim],
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class PolymarketFitClaim:
    experiment: str
    status: str
    parent_sha256: str
    dataset_sha256: str
    report_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validated_identity(
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
) -> tuple[str, str, str, str, str, str]:
    values = (
        str(experiment),
        str(parent_sha256),
        str(contract_sha256),
        str(dataset_sha256),
        str(report_table),
        str(report_parent_column),
    )
    if (
        not _IDENTIFIER.fullmatch(values[0])
        or not all(_SHA256.fullmatch(value) for value in values[1:4])
        or not all(_IDENTIFIER.fullmatch(value) for value in values[4:])
    ):
        raise ValueError("Polymarket fit-claim identity is invalid")
    return values


def _ensure_table(store: PolymarketEvidenceStore) -> None:
    store.connect().execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_model_fit_claim (
            experiment VARCHAR NOT NULL,
            parent_sha256 VARCHAR NOT NULL,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            dataset_sha256 VARCHAR NOT NULL,
            state VARCHAR NOT NULL CHECK(state IN ('started', 'failed', 'completed')),
            report_sha256 VARCHAR,
            failure_sha256 VARCHAR,
            started_at_ms UBIGINT NOT NULL,
            completed_at_ms UBIGINT,
            PRIMARY KEY(experiment, parent_sha256)
        )
        """
    )


def begin_polymarket_fit_claim(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
) -> PolymarketFitClaim:
    """Claim a parent before a model can read its untouched test partition."""

    identity = _validated_identity(
        experiment=experiment,
        parent_sha256=parent_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        report_table=report_table,
        report_parent_column=report_parent_column,
    )
    (
        experiment,
        parent_sha256,
        contract_sha256,
        dataset_sha256,
        report_table,
        report_parent_column,
    ) = identity
    _ensure_table(store)
    connection = store.connect()
    connection.execute("BEGIN TRANSACTION")
    try:
        claim = connection.execute(
            """
            SELECT schema_version, contract_sha256, dataset_sha256, state,
                   report_sha256
            FROM polymarket_model_fit_claim
            WHERE experiment = ? AND parent_sha256 = ?
            """,
            [experiment, parent_sha256],
        ).fetchone()
        report_table_exists = bool(
            connection.execute(
                """
                SELECT count(*) FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_name = ?
                """,
                [report_table],
            ).fetchone()[0]
        )
        persisted = []
        if report_table_exists:
            persisted = connection.execute(
                f"""
                SELECT report_sha256, contract_sha256, dataset_sha256
                FROM {report_table} WHERE {report_parent_column} = ?
                """,
                [parent_sha256],
            ).fetchall()
        if len(persisted) > 1:
            raise ValueError(f"multiple {experiment} reports exist for one parent")
        if claim is not None and (
            str(claim[0]) != POLYMARKET_FIT_CLAIM_SCHEMA_VERSION
            or str(claim[1]) != contract_sha256
            or str(claim[2]) != dataset_sha256
        ):
            raise ValueError(f"{experiment} fit claim identity is inconsistent")
        if persisted:
            report_sha256, stored_contract, stored_dataset = map(str, persisted[0])
            if (
                not _SHA256.fullmatch(report_sha256)
                or stored_contract != contract_sha256
                or stored_dataset != dataset_sha256
            ):
                raise ValueError(f"persisted {experiment} report identity is invalid")
            now_ms = time.time_ns() // 1_000_000
            if claim is None:
                connection.execute(
                    """
                    INSERT INTO polymarket_model_fit_claim VALUES (
                        ?, ?, ?, ?, ?, 'completed', ?, NULL, ?, ?
                    )
                    """,
                    [
                        experiment,
                        parent_sha256,
                        POLYMARKET_FIT_CLAIM_SCHEMA_VERSION,
                        contract_sha256,
                        dataset_sha256,
                        report_sha256,
                        now_ms,
                        now_ms,
                    ],
                )
            elif str(claim[3]) != "completed" or str(claim[4] or "") != report_sha256:
                connection.execute(
                    """
                    UPDATE polymarket_model_fit_claim
                    SET state = 'completed', report_sha256 = ?,
                        failure_sha256 = NULL, completed_at_ms = ?
                    WHERE experiment = ? AND parent_sha256 = ?
                    """,
                    [report_sha256, now_ms, experiment, parent_sha256],
                )
            connection.execute("COMMIT")
            return PolymarketFitClaim(
                experiment=experiment,
                status="existing",
                parent_sha256=parent_sha256,
                dataset_sha256=dataset_sha256,
                report_sha256=report_sha256,
            )
        if claim is not None:
            raise ValueError(f"{experiment} test is already claimed:state={claim[3]}")
        connection.execute(
            """
            INSERT INTO polymarket_model_fit_claim VALUES (
                ?, ?, ?, ?, ?, 'started', NULL, NULL, ?, NULL
            )
            """,
            [
                experiment,
                parent_sha256,
                POLYMARKET_FIT_CLAIM_SCHEMA_VERSION,
                contract_sha256,
                dataset_sha256,
                time.time_ns() // 1_000_000,
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return PolymarketFitClaim(
        experiment=experiment,
        status="claimed",
        parent_sha256=parent_sha256,
        dataset_sha256=dataset_sha256,
        report_sha256="",
    )


def reserve_polymarket_fit_claim(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
) -> PolymarketFitClaim:
    """Persist a claim before clear labels are loaded and reserve it once locally."""

    identity = _validated_identity(
        experiment=experiment,
        parent_sha256=parent_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        report_table=report_table,
        report_parent_column=report_parent_column,
    )
    key = identity
    with _RESERVATION_LOCK:
        if key in _RESERVATIONS.get(store, {}):
            raise ValueError(f"{identity[0]} fit claim is already reserved")
        claim = begin_polymarket_fit_claim(
            store,
            experiment=identity[0],
            parent_sha256=identity[1],
            contract_sha256=identity[2],
            dataset_sha256=identity[3],
            report_table=identity[4],
            report_parent_column=identity[5],
        )
        if claim.status == "claimed":
            reservations = _RESERVATIONS.setdefault(store, {})
            reservations[key] = claim
        return claim


def consume_polymarket_fit_claim(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
) -> PolymarketFitClaim:
    """Consume one matching in-process reservation or create a normal claim."""

    identity = _validated_identity(
        experiment=experiment,
        parent_sha256=parent_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        report_table=report_table,
        report_parent_column=report_parent_column,
    )
    key = identity
    with _RESERVATION_LOCK:
        reservations = _RESERVATIONS.get(store)
        claim = None if reservations is None else reservations.pop(key, None)
        if reservations is not None and not reservations:
            del _RESERVATIONS[store]
    if claim is not None:
        return claim
    return begin_polymarket_fit_claim(
        store,
        experiment=identity[0],
        parent_sha256=identity[1],
        contract_sha256=identity[2],
        dataset_sha256=identity[3],
        report_table=identity[4],
        report_parent_column=identity[5],
    )


def discard_polymarket_fit_reservation(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
) -> None:
    """Remove only the local token; the durable claim remains authoritative."""

    identity = _validated_identity(
        experiment=experiment,
        parent_sha256=parent_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        report_table=report_table,
        report_parent_column=report_parent_column,
    )
    key = identity
    with _RESERVATION_LOCK:
        reservations = _RESERVATIONS.get(store)
        if reservations is None:
            return
        reservations.pop(key, None)
        if not reservations:
            del _RESERVATIONS[store]


def complete_polymarket_fit_claim(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    contract_sha256: str,
    dataset_sha256: str,
    report_table: str,
    report_parent_column: str,
    report_sha256: str,
) -> None:
    """Bind a materialized report to its started claim."""

    identity = _validated_identity(
        experiment=experiment,
        parent_sha256=parent_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        report_table=report_table,
        report_parent_column=report_parent_column,
    )
    (
        experiment,
        parent_sha256,
        contract_sha256,
        dataset_sha256,
        report_table,
        report_parent_column,
    ) = identity
    if not _SHA256.fullmatch(str(report_sha256)):
        raise ValueError("Polymarket fit completion report digest is invalid")
    connection = store.connect()
    persisted = connection.execute(
        f"""
        SELECT {report_parent_column}, contract_sha256, dataset_sha256
        FROM {report_table} WHERE report_sha256 = ?
        """,
        [report_sha256],
    ).fetchone()
    claim = connection.execute(
        """
        SELECT contract_sha256, dataset_sha256, state, report_sha256
        FROM polymarket_model_fit_claim
        WHERE experiment = ? AND parent_sha256 = ?
        """,
        [experiment, parent_sha256],
    ).fetchone()
    if (
        persisted is None
        or str(persisted[0]) != parent_sha256
        or str(persisted[1]) != contract_sha256
        or str(persisted[2]) != dataset_sha256
        or claim is None
        or str(claim[0]) != contract_sha256
        or str(claim[1]) != dataset_sha256
        or str(claim[2]) not in {"started", "completed"}
        or str(claim[3] or report_sha256) != report_sha256
    ):
        raise ValueError(f"{experiment} fit completion claim is invalid")
    connection.execute(
        """
        UPDATE polymarket_model_fit_claim
        SET state = 'completed', report_sha256 = ?, failure_sha256 = NULL,
            completed_at_ms = ?
        WHERE experiment = ? AND parent_sha256 = ?
        """,
        [
            report_sha256,
            time.time_ns() // 1_000_000,
            experiment,
            parent_sha256,
        ],
    )


def fail_polymarket_fit_claim(
    store: PolymarketEvidenceStore,
    *,
    experiment: str,
    parent_sha256: str,
    error: BaseException,
) -> None:
    """Persist a failure so a retry cannot silently reopen test."""

    if not _IDENTIFIER.fullmatch(str(experiment)) or not _SHA256.fullmatch(
        str(parent_sha256)
    ):
        raise ValueError("Polymarket failed fit-claim identity is invalid")
    failure_sha256 = _sha256(
        {"error_type": type(error).__name__, "error_message": str(error)}
    )
    connection = store.connect()
    connection.execute(
        """
        UPDATE polymarket_model_fit_claim
        SET state = 'failed', failure_sha256 = ?, completed_at_ms = ?
        WHERE experiment = ? AND parent_sha256 = ? AND state = 'started'
        """,
        [
            failure_sha256,
            time.time_ns() // 1_000_000,
            experiment,
            parent_sha256,
        ],
    )


__all__ = [
    "POLYMARKET_FIT_CLAIM_SCHEMA_VERSION",
    "PolymarketFitClaim",
    "begin_polymarket_fit_claim",
    "complete_polymarket_fit_claim",
    "consume_polymarket_fit_claim",
    "discard_polymarket_fit_reservation",
    "fail_polymarket_fit_claim",
    "reserve_polymarket_fit_claim",
]
