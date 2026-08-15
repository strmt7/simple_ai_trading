"""Resolve every result-less Round 74 slot after the campaign ends."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_event_segmented_cohort import (  # noqa: E402
    Round74SegmentedCohortPlan,
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.impact_absorption_store import (  # noqa: E402
    ImpactAbsorptionStore,
)
from simple_ai_trading.round74_segmented_development_inputs import (  # noqa: E402
    ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION,
    ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION,
    Round74SegmentedLateAdjudication,
    Round74SegmentedRecoveryOutcome,
    build_round74_segmented_late_adjudication,
    build_round74_segmented_recovery_outcome,
    write_round74_segmented_late_adjudication,
    write_round74_segmented_recovery_outcome,
)
from simple_ai_trading.round74_segmented_cohort_operator import (  # noqa: E402
    _strict_json_mapping,
    audit_and_adjudicate_round74_segmented_supervisor,
)


ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION = (
    "round-074-segmented-recovery-build-v2"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-segmented-event-cohort-plan-v3.json"
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("data/round74-segmented-event-cohort-v3-state"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/round74-segmented-event-cohort-v3-recovery"),
    )
    parser.add_argument(
        "--database",
        action="append",
        type=Path,
        default=[],
        help=(
            "repeat for every immutable campaign shard; required when a "
            "result-less slot contains a captured run"
        ),
    )
    parser.add_argument(
        "--audit-workers",
        type=int,
        default=1,
        help="independent database-shard audit workers (1-4)",
    )
    parser.add_argument(
        "--emit-progress",
        action="store_true",
        help="emit machine-readable progress records to stderr",
    )
    return parser


def _resolve(repository: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository / path


def _recovery_observed_wall_ns(
    output: Path,
    *,
    plan_sha256: str,
    total_slots: int,
    fallback_wall_ns: int,
) -> int:
    """Reuse the first immutable panel timestamp after an interrupted build."""

    if not output.exists():
        return fallback_wall_ns
    entries = tuple(output.iterdir())
    if not entries:
        return fallback_wall_ns
    observed: set[int] = set()
    for path in entries:
        if (
            path.is_symlink()
            or not path.is_file()
            or len(path.stem) != 3
            or not path.stem.isascii()
            or not path.stem.isdecimal()
            or path.suffix != ".json"
            or not 0 <= int(path.stem) < total_slots
        ):
            raise ValueError("Round 74 partial recovery output panel differs")
        payload = _strict_json_mapping(
            path.read_text(encoding="utf-8"),
            "partial segmented recovery outcome",
        )
        value = payload.get("observed_wall_ns")
        if (
            payload.get("plan_sha256") != plan_sha256
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError("Round 74 partial recovery identity differs")
        observed.add(value)
    if len(observed) != 1:
        raise ValueError("Round 74 partial recovery observation time differs")
    return observed.pop()


def _resultless_supervisor(slot_directory: Path | None) -> dict[str, object] | None:
    if slot_directory is None:
        return None
    required = {
        "reservation.json",
        "state.json",
        "capture.stdout.json",
        "capture.stderr.log",
    }
    allowed = required | {
        "recovery-adjudication.json",
        "recovery-audit-error.log",
    }
    entries = tuple(slot_directory.iterdir())
    observed = {entry.name for entry in entries}
    if (
        not required.issubset(observed)
        or not observed.issubset(allowed)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        return None
    try:
        return dict(
            _strict_json_mapping(
                (slot_directory / "capture.stdout.json").read_text(encoding="utf-8"),
                "result-less segmented supervisor",
            )
        )
    except (OSError, UnicodeError, ValueError):
        return None


def _route_captured_runs(
    database_paths: tuple[Path, ...],
    run_ids: tuple[str, ...],
    *,
    opened_database_paths: list[Path] | None = None,
) -> dict[str, Path]:
    if not run_ids:
        return {}
    if not database_paths:
        raise ValueError(
            "Round 74 captured result-less slots require --database shards"
        )
    expected = set(run_ids)
    routes: dict[str, Path] = {}
    placeholders = ",".join("?" for _run_id in run_ids)
    for path in database_paths:
        if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
            raise ValueError("Round 74 recovery database shard differs")
        with ImpactAbsorptionStore(
            path,
            read_only=True,
            memory_limit="2GB",
            threads=2,
        ) as store:
            connection = store.connect()
            if opened_database_paths is not None:
                opened_database_paths.append(path)
            rows = connection.execute(
                "SELECT run_id FROM impact_capture_run "
                f"WHERE run_id IN ({placeholders}) ORDER BY run_id",  # nosec B608
                list(run_ids),
            ).fetchall()
        for row in rows:
            run_id = str(row[0])
            if run_id in routes:
                raise ValueError(
                    "Round 74 recovery run appears in multiple database shards"
                )
            routes[run_id] = path
    if set(routes) != expected:
        raise ValueError("Round 74 recovery captured run routing differs")
    return routes


def _terminal_audit_preflight_reason(
    run_status: str,
    run_error: str,
    report: dict[str, object] | None,
) -> str | None:
    if report is None:
        return "stored_report_missing"
    completed = (
        run_status == "completed"
        and report.get("status") == "completed"
        and report.get("failure_class") == "none"
        and report.get("error") == ""
        and run_error == ""
        and report.get("capture_gate_passed") is True
        and report.get("data_qualification_passed") is True
    )
    transport_ended = (
        run_status == "failed"
        and report.get("status") == "failed"
        and report.get("failure_class") == "transport"
        and isinstance(report.get("error"), str)
        and bool(str(report["error"]))
        and run_error == report.get("error")
        and report.get("capture_gate_passed") is False
        and report.get("data_qualification_passed") is False
    )
    if not completed and not transport_ended:
        return "unsupported_terminal_class"
    if (
        report.get("resource_safety_passed") is not True
        or report.get("storage_efficiency_passed") is not True
        or report.get("audit_passed") is not True
        or report.get("audit_errors") != []
        or report.get("resource_safety_errors") != []
        or report.get("payload_cap_reached") is not False
        or report.get("database_size_cap_reached") is not False
    ):
        return "stored_safety_gate_failed"
    return None


def _preflight_captured_run_audits(
    routes: dict[str, Path],
) -> dict[str, str | None]:
    grouped: dict[Path, list[str]] = {}
    for run_id, database in routes.items():
        grouped.setdefault(database, []).append(run_id)
    result: dict[str, str | None] = {}
    for database, run_ids in grouped.items():
        placeholders = ",".join("?" for _run_id in run_ids)
        with ImpactAbsorptionStore(
            database,
            read_only=True,
            memory_limit="1GB",
            threads=2,
        ) as store:
            rows = (
                store.connect()
                .execute(
                    "SELECT r.run_id, r.status, r.error, p.report_json "
                    "FROM impact_capture_run AS r "
                    "LEFT JOIN impact_capture_report AS p USING (run_id) "
                    f"WHERE r.run_id IN ({placeholders}) ORDER BY r.run_id",  # nosec B608
                    run_ids,
                )
                .fetchall()
            )
        if {str(row[0]) for row in rows} != set(run_ids):
            raise ValueError("Round 74 recovery terminal preflight coverage differs")
        for run_id, run_status, run_error, report_json in rows:
            report: dict[str, object] | None = None
            if report_json is not None:
                try:
                    parsed = json.loads(str(report_json))
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    report = parsed
            result[str(run_id)] = _terminal_audit_preflight_reason(
                str(run_status),
                str(run_error),
                report,
            )
    if set(result) != set(routes):
        raise ValueError("Round 74 recovery terminal preflight identity differs")
    return result


def _emit_progress(enabled: bool, stage: str, **values: object) -> None:
    if not enabled:
        return
    print(
        json.dumps(
            {"stage": stage, **values},
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _load_existing_recovery_row(
    plan: Round74SegmentedCohortPlan,
    *,
    path: Path,
    slot_ordinal: int,
    slot_directory: Path | None,
    observed_wall_ns: int,
) -> dict[str, object]:
    payload = _strict_json_mapping(
        path.read_text(encoding="utf-8"),
        "existing segmented recovery outcome",
    )
    schema_version = payload.get("schema_version")
    if schema_version == ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION:
        recovery = Round74SegmentedRecoveryOutcome.from_dict(plan, payload)
        recovery.verify_slot_directory(slot_directory)
        digest = recovery.recovery_sha256
        outcome_status = recovery.outcome.status
        selected_ordinal = recovery.slot_ordinal
        selected_observed_wall_ns = recovery.observed_wall_ns
    elif schema_version == ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION:
        if slot_directory is None:
            raise ValueError("Round 74 existing late adjudication slot is missing")
        late = Round74SegmentedLateAdjudication.from_dict(plan, payload)
        late.verify_slot_directory(plan, slot_directory)
        digest = late.late_adjudication_sha256
        outcome_status = late.outcome.status
        selected_ordinal = late.slot_ordinal
        selected_observed_wall_ns = late.observed_wall_ns
    else:
        raise ValueError("Round 74 existing recovery schema differs")
    if (
        selected_ordinal != slot_ordinal
        or selected_observed_wall_ns != observed_wall_ns
    ):
        raise ValueError("Round 74 existing recovery binding differs")
    return {
        "slot_ordinal": slot_ordinal,
        "path": str(path),
        "schema_version": schema_version,
        "outcome_status": outcome_status,
        "recovery_sha256": digest,
    }


def _audit_captured_group(
    plan: Round74SegmentedCohortPlan,
    database_path: Path,
    rows: tuple[tuple[int, dict[str, object]], ...],
) -> tuple[tuple[int, object | None, str, str], ...]:
    adjudications: list[tuple[int, object | None, str, str]] = []
    with ImpactAbsorptionStore(
        database_path,
        read_only=True,
        memory_limit="2GB",
        threads=2,
    ) as store:
        for ordinal, supervisor in rows:
            try:
                adjudication = audit_and_adjudicate_round74_segmented_supervisor(
                    plan,
                    slot_ordinal=ordinal,
                    supervisor_payload=supervisor,
                    store=store,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                adjudications.append((ordinal, None, type(exc).__name__, str(exc)))
            else:
                adjudications.append((ordinal, adjudication, "", ""))
    return tuple(adjudications)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    plan_path = _resolve(repository, arguments.plan)
    state_root = _resolve(repository, arguments.state_root)
    output = _resolve(repository, arguments.output)
    database_paths = tuple(
        _resolve(repository, path).resolve() for path in arguments.database
    )
    opened_database_paths: list[Path] = []
    try:
        if (
            plan_path.is_symlink()
            or not plan_path.is_file()
            or state_root.is_symlink()
            or not state_root.is_dir()
            or output.is_symlink()
            or output.parent.is_symlink()
            or output.exists()
            and not output.is_dir()
            or len(database_paths) != len(set(database_paths))
            or isinstance(arguments.audit_workers, bool)
            or not 1 <= arguments.audit_workers <= 4
        ):
            raise ValueError("Round 74 segmented recovery build paths differ")
        plan = load_round74_segmented_cohort_plan(plan_path.read_text(encoding="utf-8"))
        observed_wall_ns = _recovery_observed_wall_ns(
            output,
            plan_sha256=plan.plan_sha256,
            total_slots=plan.total_slots,
            fallback_wall_ns=time.time_ns(),
        )
        output.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[int, Path | None, dict[str, object] | None]] = []
        captured_run_ids: list[str] = []
        completed_rows: dict[int, dict[str, object]] = {}
        for ordinal in range(plan.total_slots):
            slot_directory = state_root / f"slot-{ordinal:03d}"
            selected_directory = slot_directory if slot_directory.is_dir() else None
            if (
                selected_directory is not None
                and (selected_directory / "result.json").is_file()
            ):
                continue
            supervisor = _resultless_supervisor(selected_directory)
            recovery_path = output / f"{ordinal:03d}.json"
            if recovery_path.is_file():
                completed_rows[ordinal] = _load_existing_recovery_row(
                    plan,
                    path=recovery_path,
                    slot_ordinal=ordinal,
                    slot_directory=selected_directory,
                    observed_wall_ns=observed_wall_ns,
                )
                pending.append((ordinal, selected_directory, supervisor))
                continue
            if supervisor is not None:
                attempts = supervisor.get("attempts")
                if not isinstance(attempts, list):
                    raise ValueError("Round 74 result-less supervisor attempts differ")
                if attempts:
                    if len(attempts) != 1 or not isinstance(attempts[0], dict):
                        raise ValueError(
                            "Round 74 result-less supervisor attempts differ"
                        )
                    captured_run_ids.append(str(attempts[0].get("run_id", "")))
            pending.append((ordinal, selected_directory, supervisor))
        if len(captured_run_ids) != len(set(captured_run_ids)):
            raise ValueError("Round 74 result-less captured run identity is duplicated")
        _emit_progress(
            arguments.emit_progress,
            "recovery_scan_completed",
            resultless_slot_count=len(pending),
            resumed_receipt_count=len(completed_rows),
            unresolved_captured_run_count=len(captured_run_ids),
        )
        run_routes = _route_captured_runs(
            database_paths,
            tuple(captured_run_ids),
            opened_database_paths=opened_database_paths,
        )
        audit_preflight = _preflight_captured_run_audits(run_routes)
        _emit_progress(
            arguments.emit_progress,
            "recovery_run_routing_completed",
            supplied_database_count=len(database_paths),
            routed_run_count=len(run_routes),
            auditable_run_count=sum(
                reason is None for reason in audit_preflight.values()
            ),
            conservative_recovery_count=sum(
                reason is not None for reason in audit_preflight.values()
            ),
        )
        grouped: dict[Path, list[tuple[int, dict[str, object]]]] = {}
        conservative_recovery: dict[int, dict[str, object]] = {}
        for ordinal, _selected_directory, supervisor in pending:
            if ordinal in completed_rows or supervisor is None:
                continue
            attempts = supervisor["attempts"]
            if attempts:
                run_id = str(attempts[0]["run_id"])
                reason = audit_preflight[run_id]
                if reason is None:
                    grouped.setdefault(run_routes[run_id], []).append(
                        (ordinal, supervisor)
                    )
                else:
                    conservative_recovery[ordinal] = {
                        "slot_ordinal": ordinal,
                        "run_id": run_id,
                        "database": run_routes[run_id].name,
                        "reason": reason,
                    }
            else:
                conservative_recovery[ordinal] = {
                    "slot_ordinal": ordinal,
                    "run_id": None,
                    "database": None,
                    "reason": "capture_attempt_absent",
                }

        adjudications: dict[int, object] = {}
        audit_failures: dict[int, dict[str, object]] = {}
        if grouped:
            audit_groups = tuple(
                (database, (row,)) for database, rows in grouped.items() for row in rows
            )
            workers = min(arguments.audit_workers, len(audit_groups))
            completed_audit_group_count = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _audit_captured_group,
                        plan,
                        database,
                        rows,
                    ): (database, rows[0][0])
                    for database, rows in audit_groups
                }
                for future in as_completed(futures):
                    database, submitted_ordinal = futures[future]
                    for ordinal, adjudication, error_type, error in future.result():
                        if adjudication is None:
                            audit_failures[ordinal] = {
                                "slot_ordinal": ordinal,
                                "database": database.name,
                                "reason": "fresh_audit_failed",
                                "error_type": error_type,
                                "error": error,
                            }
                        else:
                            adjudications[ordinal] = adjudication
                    opened_database_paths.append(database)
                    completed_audit_group_count += 1
                    _emit_progress(
                        arguments.emit_progress,
                        "recovery_run_audit_completed",
                        database=database.name,
                        slot_ordinal=submitted_ordinal,
                        audit_succeeded=submitted_ordinal in adjudications,
                        completed_audit_group_count=completed_audit_group_count,
                        total_audit_group_count=len(audit_groups),
                    )
        recoveries = []
        admitted_count = 0
        late_adjudication_count = 0
        captured_run_recovery_count = 0
        for ordinal, selected_directory, supervisor in pending:
            if ordinal in completed_rows:
                row = completed_rows[ordinal]
                schema_version = row["schema_version"]
                outcome_status = row["outcome_status"]
                path = Path(str(row["path"]))
                digest = str(row["recovery_sha256"])
            elif (
                supervisor is None
                or ordinal in conservative_recovery
                or ordinal in audit_failures
            ):
                recovery = build_round74_segmented_recovery_outcome(
                    plan,
                    slot_ordinal=ordinal,
                    observed_wall_ns=observed_wall_ns,
                    slot_directory=selected_directory,
                )
                path = write_round74_segmented_recovery_outcome(
                    recovery,
                    plan=plan,
                    path=output / f"{ordinal:03d}.json",
                )
                recovery.verify_slot_directory(selected_directory)
                digest = recovery.recovery_sha256
                schema_version = ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION
                outcome_status = recovery.outcome.status
            else:
                if selected_directory is None:
                    raise RuntimeError(
                        "Round 74 result-less supervisor directory disappeared"
                    )
                adjudication = adjudications.get(ordinal)
                if adjudication is None:
                    raise RuntimeError(
                        "Round 74 result-less adjudication panel is incomplete"
                    )
                late = build_round74_segmented_late_adjudication(
                    plan,
                    slot_ordinal=ordinal,
                    observed_wall_ns=observed_wall_ns,
                    slot_directory=selected_directory,
                    adjudication=adjudication,
                )
                path = write_round74_segmented_late_adjudication(
                    late,
                    plan=plan,
                    path=output / f"{ordinal:03d}.json",
                )
                late.verify_slot_directory(plan, selected_directory)
                digest = late.late_adjudication_sha256
                schema_version = ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION
                outcome_status = late.outcome.status
            admitted_count += int(
                schema_version == ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION
                and outcome_status == "admitted"
            )
            late_adjudication_count += int(
                schema_version == ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION
            )
            captured_run_recovery_count += int(
                supervisor is not None
                and bool(supervisor.get("attempts"))
                and schema_version == ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION
            )
            recoveries.append(
                {
                    "slot_ordinal": ordinal,
                    "path": str(path),
                    "schema_version": schema_version,
                    "outcome_status": outcome_status,
                    "recovery_sha256": digest,
                }
            )
        expected_names = {f"{row['slot_ordinal']:03d}.json" for row in recoveries}
        output_entries = tuple(output.iterdir())
        if {path.name for path in output_entries} != expected_names or any(
            path.is_symlink() or not path.is_file() for path in output_entries
        ):
            raise ValueError("Round 74 segmented recovery output panel differs")
        result: dict[str, object] = {
            "schema_version": ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION,
            "supported_recovery_schema_versions": [
                ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION,
                ROUND74_SEGMENTED_LATE_ADJUDICATION_SCHEMA_VERSION,
            ],
            "plan_sha256": plan.plan_sha256,
            "observed_wall_ns": observed_wall_ns,
            "recovery_count": len(recoveries),
            "late_adjudication_count": late_adjudication_count,
            "late_admitted_count": admitted_count,
            "recoveries": recoveries,
            "admitted_data_created": admitted_count > 0,
            "database_opened": bool(opened_database_paths),
            "opened_database_shard_count": len(set(opened_database_paths)),
            "audit_worker_count": arguments.audit_workers,
            "captured_run_recovery_count": captured_run_recovery_count,
            "conservative_recoveries": sorted(
                [*conservative_recovery.values(), *audit_failures.values()],
                key=lambda row: int(row["slot_ordinal"]),
            ),
            "market_window_retried": False,
            "orders_submitted": False,
            "profitability_or_edge_claim": False,
            "trading_authority": False,
        }
        result["result_sha256"] = _canonical_sha256(result)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": (ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "admitted_data_created": False,
                    "database_opened": bool(opened_database_paths),
                    "opened_database_shard_count": len(opened_database_paths),
                    "trading_authority": False,
                },
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
