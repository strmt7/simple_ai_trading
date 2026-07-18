"""Action-local continuity evidence for Polymarket Round 12 confirmation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .polymarket_action_value import PolymarketActionFeature
    from .polymarket_repricing import PolymarketRepricingDecisionExecution


POLYMARKET_ROUND12_ADMISSION_SCHEMA_VERSION = (
    "polymarket-round12-action-local-admission-v1"
)
POLYMARKET_ROUND12_CREATION_BOOK_MAXIMUM_AGE_MS = 500
POLYMARKET_ROUND12_EXECUTION_OBSERVATION_MAXIMUM_DELAY_MS = 500

_KNOWN_NO_FILL_TERMINALS = frozenset({"entry_not_filled"})
_PRE_SUBMIT_ABSTAIN_TERMINALS = frozenset(
    {
        "entry_enters_excluded_close_window",
        "entry_tick_drift",
        "missing_entry_execution_parameters",
        "unsupported_entry_minimum_order_age",
    }
)
_UNKNOWN_AFTER_SUBMIT_TERMINALS = frozenset(
    {
        "entry_confirmation_enters_excluded_close_window",
        "missing_entry_execution_book",
    }
)
_OBSERVATION_STATES = frozenset(
    {"not_submitted", "known_no_fill", "known_fill", "unknown_after_submit"}
)

_ADMISSION_ROW_COLUMNS = (
    "admission_dataset_sha256",
    "admission_index",
    "action_feature_sha256",
    "admission_sha256",
    "condition_id",
    "outcome",
    "terminal_reason",
    "decision_event_id",
    "decision_segment_id",
    "decision_monotonic_ns",
    "creation_book_event_id",
    "creation_book_segment_id",
    "creation_book_monotonic_ns",
    "entry_execution_target_monotonic_ns",
    "entry_book_event_id",
    "entry_book_segment_id",
    "entry_book_monotonic_ns",
    "decision_admissible",
    "submission_attempted",
    "observation_state",
    "condition_blocked",
    "reasons_json",
)
_ADMISSION_ROW_INSERT_SQL = (
    "INSERT INTO polymarket_round12_action_local_admission ("
    + ", ".join(_ADMISSION_ROW_COLUMNS)
    + ") SELECT "
    + ", ".join("unnest(?)" for _column in _ADMISSION_ROW_COLUMNS)
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class PolymarketRound12ActionLocalAdmission:
    """Hash-bound entry-horizon evidence; later feed state cannot alter admission."""

    action_feature_sha256: str
    condition_id: str
    outcome: str
    terminal_reason: str
    decision_event_id: str
    decision_segment_id: str
    decision_monotonic_ns: int | None
    creation_book_event_id: str
    creation_book_segment_id: str
    creation_book_monotonic_ns: int | None
    entry_execution_target_monotonic_ns: int | None
    entry_book_event_id: str
    entry_book_segment_id: str
    entry_book_monotonic_ns: int | None
    decision_admissible: bool
    submission_attempted: bool
    observation_state: str
    condition_blocked: bool
    reasons: tuple[str, ...]
    admission_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("admission_sha256")
        payload["reasons"] = list(self.reasons)
        return {
            "schema_version": POLYMARKET_ROUND12_ADMISSION_SCHEMA_VERSION,
            **payload,
            "admission_scope": "decision_through_entry_observation_only",
            "post_entry_feed_continuity_required_for_resolution_label": False,
            "unknown_after_submit_is_no_fill": False,
        }

    def validated(self) -> PolymarketRound12ActionLocalAdmission:
        event_fields = (
            self.decision_event_id,
            self.creation_book_event_id,
            self.entry_book_event_id,
        )
        segment_fields = (
            self.decision_segment_id,
            self.creation_book_segment_id,
            self.entry_book_segment_id,
        )
        timestamps = (
            self.decision_monotonic_ns,
            self.creation_book_monotonic_ns,
            self.entry_execution_target_monotonic_ns,
            self.entry_book_monotonic_ns,
        )
        if (
            not _is_sha256(self.action_feature_sha256)
            or not self.condition_id
            or self.outcome not in {"Up", "Down"}
            or not self.terminal_reason
            or self.observation_state not in _OBSERVATION_STATES
            or tuple(sorted(set(self.reasons))) != self.reasons
            or any(value is not None and value < 0 for value in timestamps)
            or any(value and not _is_sha256(value) for value in segment_fields)
            or not _is_sha256(self.admission_sha256)
            or self.admission_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 12 action-local admission is invalid")

        if self.decision_admissible:
            if (
                any(not value for value in event_fields[:2])
                or any(not value for value in segment_fields[:2])
                or self.decision_segment_id != self.creation_book_segment_id
                or self.decision_monotonic_ns is None
                or self.creation_book_monotonic_ns is None
                or not 0
                <= self.decision_monotonic_ns - self.creation_book_monotonic_ns
                <= POLYMARKET_ROUND12_CREATION_BOOK_MAXIMUM_AGE_MS * 1_000_000
            ):
                raise ValueError(
                    "Polymarket Round 12 decision continuity is invalid"
                )
        elif (
            self.submission_attempted
            or self.observation_state != "not_submitted"
            or self.condition_blocked
        ):
            raise ValueError("inadmissible Round 12 decision cannot submit")

        if self.submission_attempted:
            if self.entry_execution_target_monotonic_ns is None:
                raise ValueError("Round 12 submission has no execution target")
            if self.entry_execution_target_monotonic_ns <= int(
                self.decision_monotonic_ns or -1
            ):
                raise ValueError("Round 12 execution target is not causal")
        elif self.entry_execution_target_monotonic_ns is not None and (
            self.observation_state != "not_submitted"
        ):
            raise ValueError("Round 12 unsubmitted action has observation state")

        observed = self.observation_state in {"known_no_fill", "known_fill"}
        if observed:
            if (
                not self.submission_attempted
                or not self.entry_book_event_id
                or not self.entry_book_segment_id
                or self.entry_book_segment_id != self.decision_segment_id
                or self.entry_book_monotonic_ns is None
                or self.entry_execution_target_monotonic_ns is None
                or not 0
                <= self.entry_book_monotonic_ns
                - self.entry_execution_target_monotonic_ns
                <= POLYMARKET_ROUND12_EXECUTION_OBSERVATION_MAXIMUM_DELAY_MS
                * 1_000_000
            ):
                raise ValueError("Round 12 observed entry continuity is invalid")
        elif self.entry_book_event_id or self.entry_book_segment_id or (
            self.entry_book_monotonic_ns is not None
        ):
            raise ValueError("Round 12 unobserved entry contains book evidence")

        expected_blocked = self.observation_state in {
            "known_fill",
            "unknown_after_submit",
        }
        if self.condition_blocked != expected_blocked:
            raise ValueError("Round 12 condition blocking state is invalid")
        return self


def _finalize(
    provisional: PolymarketRound12ActionLocalAdmission,
) -> PolymarketRound12ActionLocalAdmission:
    return replace(
        provisional,
        admission_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def build_round12_action_local_admission(
    feature: PolymarketActionFeature,
    execution: PolymarketRepricingDecisionExecution | None,
    *,
    unavailable_reason: str = "missing_entry_creation_book",
) -> PolymarketRound12ActionLocalAdmission:
    """Derive causal submission and observation states from immutable replay evidence."""

    feature.validated()
    base = {
        "action_feature_sha256": feature.action_feature_sha256,
        "condition_id": feature.condition_id,
        "outcome": feature.outcome,
    }
    if execution is None:
        reason = str(unavailable_reason).strip()
        if not reason:
            raise ValueError("missing Round 12 action requires an explicit reason")
        return _finalize(
            PolymarketRound12ActionLocalAdmission(
                **base,
                terminal_reason=reason,
                decision_event_id=feature.decision_event_id,
                decision_segment_id="",
                decision_monotonic_ns=None,
                creation_book_event_id="",
                creation_book_segment_id="",
                creation_book_monotonic_ns=None,
                entry_execution_target_monotonic_ns=None,
                entry_book_event_id="",
                entry_book_segment_id="",
                entry_book_monotonic_ns=None,
                decision_admissible=False,
                submission_attempted=False,
                observation_state="not_submitted",
                condition_blocked=False,
                reasons=(reason,),
                admission_sha256="",
            )
        )

    decision = execution.decision.validated(
        maximum_creation_book_age_ms=(
            POLYMARKET_ROUND12_CREATION_BOOK_MAXIMUM_AGE_MS
        )
    )
    if (
        decision.condition_id != feature.condition_id
        or decision.token_id != feature.token_id
        or decision.outcome != feature.outcome
        or decision.event_id != feature.decision_event_id
        or decision.received_wall_ms != feature.decision_received_wall_ms
        or decision.received_monotonic_ns
        != feature.decision_received_monotonic_ns
    ):
        raise ValueError("Round 12 action execution and feature are misaligned")
    creation = decision.creation_book
    if creation.segment_id != decision.segment_id:
        raise ValueError("Round 12 creation book crossed continuity segments")

    reason = str(execution.terminal_reason).strip()
    target_ns = execution.entry_execution_target_monotonic_ns
    entry_book = execution.entry_book
    submission_attempted = reason not in _PRE_SUBMIT_ABSTAIN_TERMINALS
    reasons: list[str] = []
    if not submission_attempted:
        state = "not_submitted"
        target_ns = None
        entry_book = None
        reasons.append(reason)
    elif execution.entry_filled:
        state = "known_fill"
    elif reason in _KNOWN_NO_FILL_TERMINALS and execution.entry_result is not None:
        state = "known_no_fill"
    elif reason in _UNKNOWN_AFTER_SUBMIT_TERMINALS:
        state = "unknown_after_submit"
        reasons.append(reason)
        entry_book = None
    else:
        raise ValueError("Round 12 entry terminal state is not auditable")

    if submission_attempted and target_ns is None:
        raise ValueError("Round 12 submitted action has no execution target")
    if state in {"known_fill", "known_no_fill"}:
        if entry_book is None:
            raise ValueError("Round 12 known entry state has no observed book")
        if entry_book.segment_id != decision.segment_id:
            raise ValueError("Round 12 entry observation crossed continuity segments")
    else:
        entry_book = None

    return _finalize(
        PolymarketRound12ActionLocalAdmission(
            **base,
            terminal_reason=reason,
            decision_event_id=decision.event_id,
            decision_segment_id=decision.segment_id,
            decision_monotonic_ns=decision.received_monotonic_ns,
            creation_book_event_id=creation.event_id,
            creation_book_segment_id=creation.segment_id,
            creation_book_monotonic_ns=creation.received_monotonic_ns,
            entry_execution_target_monotonic_ns=target_ns,
            entry_book_event_id="" if entry_book is None else entry_book.event_id,
            entry_book_segment_id="" if entry_book is None else entry_book.segment_id,
            entry_book_monotonic_ns=(
                None if entry_book is None else entry_book.received_monotonic_ns
            ),
            decision_admissible=True,
            submission_attempted=submission_attempted,
            observation_state=state,
            condition_blocked=state in {"known_fill", "unknown_after_submit"},
            reasons=tuple(sorted(set(reasons))),
            admission_sha256="",
        )
    )


@dataclass(frozen=True)
class PolymarketRound12AdmissionDataset:
    source_action_dataset_sha256: str
    source_run_id: str
    admissions: tuple[PolymarketRound12ActionLocalAdmission, ...]
    observation_state_counts: Mapping[str, int]
    decision_admissible_count: int
    submission_attempted_count: int
    condition_blocked_count: int
    dataset_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round12-action-local-dataset-v1",
            "source_action_dataset_sha256": self.source_action_dataset_sha256,
            "source_run_id": self.source_run_id,
            "admission_sha256": [
                item.admission_sha256 for item in self.admissions
            ],
            "observation_state_counts": dict(
                sorted(self.observation_state_counts.items())
            ),
            "decision_admissible_count": self.decision_admissible_count,
            "submission_attempted_count": self.submission_attempted_count,
            "condition_blocked_count": self.condition_blocked_count,
        }

    def validated(self) -> PolymarketRound12AdmissionDataset:
        for item in self.admissions:
            item.validated()
        expected_counts = {
            state: sum(item.observation_state == state for item in self.admissions)
            for state in sorted(_OBSERVATION_STATES)
        }
        if (
            not _is_sha256(self.source_action_dataset_sha256)
            or not self.source_run_id
            or not self.admissions
            or len({item.action_feature_sha256 for item in self.admissions})
            != len(self.admissions)
            or dict(self.observation_state_counts) != expected_counts
            or self.decision_admissible_count
            != sum(item.decision_admissible for item in self.admissions)
            or self.submission_attempted_count
            != sum(item.submission_attempted for item in self.admissions)
            or self.condition_blocked_count
            != sum(item.condition_blocked for item in self.admissions)
            or not _is_sha256(self.dataset_sha256)
            or self.dataset_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 12 admission dataset is invalid")
        return self


def build_round12_action_evidence_datasets(
    source: object,
    execution_context: object,
    *,
    config: object | None = None,
) -> tuple[object, PolymarketRound12AdmissionDataset]:
    """Build legacy action labels and compact Round 12 admission rows in one replay."""

    from .polymarket_action_value import build_polymarket_action_value_dataset

    admissions: list[PolymarketRound12ActionLocalAdmission] = []

    def observe(feature: object, execution: object | None) -> None:
        admissions.append(  # type: ignore[arg-type]
            build_round12_action_local_admission(feature, execution)
        )

    action_dataset = build_polymarket_action_value_dataset(
        source,  # type: ignore[arg-type]
        execution_context,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        execution_observer=observe,  # type: ignore[arg-type]
    )
    if tuple(item.action_feature_sha256 for item in admissions) != tuple(
        item.action_feature_sha256 for item in action_dataset.features
    ):
        raise ValueError("Round 12 admission order differs from action evidence")
    counts = {
        state: sum(item.observation_state == state for item in admissions)
        for state in sorted(_OBSERVATION_STATES)
    }
    provisional = PolymarketRound12AdmissionDataset(
        source_action_dataset_sha256=action_dataset.dataset_sha256,
        source_run_id=action_dataset.source_run_id,
        admissions=tuple(admissions),
        observation_state_counts=counts,
        decision_admissible_count=sum(item.decision_admissible for item in admissions),
        submission_attempted_count=sum(item.submission_attempted for item in admissions),
        condition_blocked_count=sum(item.condition_blocked for item in admissions),
        dataset_sha256="",
    )
    return action_dataset, replace(
        provisional,
        dataset_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _admission_rows(
    dataset: PolymarketRound12AdmissionDataset,
) -> list[tuple[object, ...]]:
    return [
        (
            dataset.dataset_sha256,
            index,
            item.action_feature_sha256,
            item.admission_sha256,
            item.condition_id,
            item.outcome,
            item.terminal_reason,
            item.decision_event_id,
            item.decision_segment_id,
            item.decision_monotonic_ns,
            item.creation_book_event_id,
            item.creation_book_segment_id,
            item.creation_book_monotonic_ns,
            item.entry_execution_target_monotonic_ns,
            item.entry_book_event_id,
            item.entry_book_segment_id,
            item.entry_book_monotonic_ns,
            item.decision_admissible,
            item.submission_attempted,
            item.observation_state,
            item.condition_blocked,
            _canonical_json(list(item.reasons)),
        )
        for index, item in enumerate(dataset.admissions)
    ]


def materialize_round12_admission_dataset(
    store: object,
    dataset: PolymarketRound12AdmissionDataset,
) -> str:
    """Persist compact columnar admission evidence without duplicating raw books."""

    from .duckdb_batch import insert_rows_columnar

    selected = dataset.validated()
    connection = store.connect()  # type: ignore[attr-defined]
    source_manifest = connection.execute(
        """
        SELECT source_run_id, action_count
        FROM polymarket_action_value_dataset
        WHERE dataset_sha256 = ?
        """,
        [selected.source_action_dataset_sha256],
    ).fetchone()
    if source_manifest != (selected.source_run_id, len(selected.admissions)):
        raise ValueError("Round 12 admission source action dataset is inconsistent")
    source_features = tuple(
        str(row[0])
        for row in connection.execute(
            """
            SELECT action_feature_sha256
            FROM polymarket_action_value_row
            WHERE dataset_sha256 = ? ORDER BY action_index
            """,
            [selected.source_action_dataset_sha256],
        ).fetchall()
    )
    if source_features != tuple(
        item.action_feature_sha256 for item in selected.admissions
    ):
        raise ValueError("Round 12 admission source action rows are inconsistent")

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_round12_admission_dataset (
            dataset_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            source_action_dataset_sha256 VARCHAR NOT NULL,
            source_run_id VARCHAR NOT NULL,
            admission_count UBIGINT NOT NULL,
            observation_state_counts_json VARCHAR NOT NULL,
            decision_admissible_count UBIGINT NOT NULL,
            submission_attempted_count UBIGINT NOT NULL,
            condition_blocked_count UBIGINT NOT NULL,
            manifest_sha256 VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS polymarket_round12_action_local_admission (
            admission_dataset_sha256 VARCHAR NOT NULL,
            admission_index UBIGINT NOT NULL,
            action_feature_sha256 VARCHAR NOT NULL,
            admission_sha256 VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            outcome VARCHAR NOT NULL,
            terminal_reason VARCHAR NOT NULL,
            decision_event_id VARCHAR NOT NULL,
            decision_segment_id VARCHAR NOT NULL,
            decision_monotonic_ns UBIGINT,
            creation_book_event_id VARCHAR NOT NULL,
            creation_book_segment_id VARCHAR NOT NULL,
            creation_book_monotonic_ns UBIGINT,
            entry_execution_target_monotonic_ns UBIGINT,
            entry_book_event_id VARCHAR NOT NULL,
            entry_book_segment_id VARCHAR NOT NULL,
            entry_book_monotonic_ns UBIGINT,
            decision_admissible BOOLEAN NOT NULL,
            submission_attempted BOOLEAN NOT NULL,
            observation_state VARCHAR NOT NULL,
            condition_blocked BOOLEAN NOT NULL,
            reasons_json VARCHAR NOT NULL,
            PRIMARY KEY(admission_dataset_sha256, admission_index),
            UNIQUE(admission_dataset_sha256, action_feature_sha256)
        );
        """
    )
    manifest = (
        selected.dataset_sha256,
        "polymarket-round12-action-local-dataset-v1",
        selected.source_action_dataset_sha256,
        selected.source_run_id,
        len(selected.admissions),
        _canonical_json(dict(sorted(selected.observation_state_counts.items()))),
        selected.decision_admissible_count,
        selected.submission_attempted_count,
        selected.condition_blocked_count,
        selected.dataset_sha256,
    )
    expected_rows = _admission_rows(selected)
    existing = connection.execute(
        """
        SELECT dataset_sha256, schema_version, source_action_dataset_sha256,
               source_run_id, admission_count, observation_state_counts_json,
               decision_admissible_count, submission_attempted_count,
               condition_blocked_count, manifest_sha256
        FROM polymarket_round12_admission_dataset WHERE dataset_sha256 = ?
        """,
        [selected.dataset_sha256],
    ).fetchone()
    if existing is not None:
        if tuple(existing) != manifest:
            raise ValueError("stored Round 12 admission manifest is inconsistent")
        stored_rows = connection.execute(
            """
            SELECT admission_dataset_sha256, admission_index,
                   action_feature_sha256, admission_sha256, condition_id,
                   outcome, terminal_reason, decision_event_id,
                   decision_segment_id, decision_monotonic_ns,
                   creation_book_event_id, creation_book_segment_id,
                   creation_book_monotonic_ns,
                   entry_execution_target_monotonic_ns, entry_book_event_id,
                   entry_book_segment_id, entry_book_monotonic_ns,
                   decision_admissible, submission_attempted,
                   observation_state, condition_blocked, reasons_json
            FROM polymarket_round12_action_local_admission
            WHERE admission_dataset_sha256 = ? ORDER BY admission_index
            """,
            [selected.dataset_sha256],
        ).fetchall()
        if [tuple(row) for row in stored_rows] != expected_rows:
            raise ValueError("stored Round 12 admission rows are inconsistent")
        return "existing"

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT INTO polymarket_round12_admission_dataset
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            manifest,
        )
        insert_rows_columnar(
            connection,
            sql=_ADMISSION_ROW_INSERT_SQL,
            rows=expected_rows,
            width=len(_ADMISSION_ROW_COLUMNS),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return "created"


def load_round12_admission_dataset(
    store: object,
    *,
    source_action_dataset_sha256: str,
) -> PolymarketRound12AdmissionDataset:
    """Load and fully revalidate one compact admission dataset from DuckDB."""

    selected = str(source_action_dataset_sha256).strip().lower()
    if not _is_sha256(selected):
        raise ValueError("Round 12 source action dataset digest is invalid")
    connection = store.connect()  # type: ignore[attr-defined]
    manifests = connection.execute(
        """
        SELECT dataset_sha256, source_run_id, admission_count,
               observation_state_counts_json, decision_admissible_count,
               submission_attempted_count, condition_blocked_count,
               manifest_sha256
        FROM polymarket_round12_admission_dataset
        WHERE source_action_dataset_sha256 = ?
        """,
        [selected],
    ).fetchall()
    if len(manifests) != 1:
        raise ValueError("Round 12 admission manifest is missing or ambiguous")
    manifest = manifests[0]
    dataset_sha256 = str(manifest[0])
    rows = connection.execute(
        """
        SELECT action_feature_sha256, admission_sha256, condition_id,
               outcome, terminal_reason, decision_event_id,
               decision_segment_id, decision_monotonic_ns,
               creation_book_event_id, creation_book_segment_id,
               creation_book_monotonic_ns,
               entry_execution_target_monotonic_ns, entry_book_event_id,
               entry_book_segment_id, entry_book_monotonic_ns,
               decision_admissible, submission_attempted,
               observation_state, condition_blocked, reasons_json
        FROM polymarket_round12_action_local_admission
        WHERE admission_dataset_sha256 = ? ORDER BY admission_index
        """,
        [dataset_sha256],
    ).fetchall()
    admissions: list[PolymarketRound12ActionLocalAdmission] = []
    for row in rows:
        try:
            reasons = json.loads(str(row[19]))
        except json.JSONDecodeError as exc:
            raise ValueError("stored Round 12 admission reasons are invalid") from exc
        if not isinstance(reasons, list) or any(
            not isinstance(value, str) for value in reasons
        ):
            raise ValueError("stored Round 12 admission reasons are invalid")
        admissions.append(
            PolymarketRound12ActionLocalAdmission(
                action_feature_sha256=str(row[0]),
                admission_sha256=str(row[1]),
                condition_id=str(row[2]),
                outcome=str(row[3]),
                terminal_reason=str(row[4]),
                decision_event_id=str(row[5]),
                decision_segment_id=str(row[6]),
                decision_monotonic_ns=(None if row[7] is None else int(row[7])),
                creation_book_event_id=str(row[8]),
                creation_book_segment_id=str(row[9]),
                creation_book_monotonic_ns=(
                    None if row[10] is None else int(row[10])
                ),
                entry_execution_target_monotonic_ns=(
                    None if row[11] is None else int(row[11])
                ),
                entry_book_event_id=str(row[12]),
                entry_book_segment_id=str(row[13]),
                entry_book_monotonic_ns=(
                    None if row[14] is None else int(row[14])
                ),
                decision_admissible=bool(row[15]),
                submission_attempted=bool(row[16]),
                observation_state=str(row[17]),
                condition_blocked=bool(row[18]),
                reasons=tuple(reasons),
            ).validated()
        )
    try:
        state_counts = json.loads(str(manifest[3]))
    except json.JSONDecodeError as exc:
        raise ValueError("stored Round 12 state counts are invalid") from exc
    dataset = PolymarketRound12AdmissionDataset(
        source_action_dataset_sha256=selected,
        source_run_id=str(manifest[1]),
        admissions=tuple(admissions),
        observation_state_counts=dict(state_counts),
        decision_admissible_count=int(manifest[4]),
        submission_attempted_count=int(manifest[5]),
        condition_blocked_count=int(manifest[6]),
        dataset_sha256=dataset_sha256,
    ).validated()
    if int(manifest[2]) != len(admissions) or str(manifest[7]) != dataset_sha256:
        raise ValueError("stored Round 12 admission manifest is inconsistent")
    return dataset


__all__ = [
    "POLYMARKET_ROUND12_ADMISSION_SCHEMA_VERSION",
    "POLYMARKET_ROUND12_CREATION_BOOK_MAXIMUM_AGE_MS",
    "POLYMARKET_ROUND12_EXECUTION_OBSERVATION_MAXIMUM_DELAY_MS",
    "PolymarketRound12ActionLocalAdmission",
    "PolymarketRound12AdmissionDataset",
    "build_round12_action_local_admission",
    "build_round12_action_evidence_datasets",
    "load_round12_admission_dataset",
    "materialize_round12_admission_dataset",
]
