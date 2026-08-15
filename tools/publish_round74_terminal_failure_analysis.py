"""Publish source-bound Round 74 terminal slot and continuity diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
for import_root in (REPOSITORY, SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.impact_absorption_event_segmented_cohort import (  # noqa: E402
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.round74_segmented_cohort_operator import (  # noqa: E402
    load_round74_segmented_slot_adjudication,
)
from simple_ai_trading.round74_segmented_development_inputs import (  # noqa: E402
    _load_complete_campaign_outcomes,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_round74_terminal_campaign_outcome import (  # noqa: E402
    _build_unqualified_partition,
)


SCHEMA_VERSION = "round-074-terminal-failure-analysis-v1"
TERMINAL_SCHEMA_VERSION = "round-074-terminal-campaign-outcome-v2"
RECOVERY_BUILD_SCHEMA_VERSION = "round-074-segmented-recovery-build-v2"
MAJOR_GAP_MINIMUM_SLOTS = 10
CSV_FIELDS = (
    "slot_ordinal",
    "role",
    "scheduled_start_utc",
    "scheduled_end_utc",
    "status",
    "reason_code",
    "failure_classification",
    "evidence_kind",
    "slot_directory_present",
    "eligible_anchor_ns",
    "cumulative_training_eligible_anchor_ns",
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root differs")
    return value


def _resolve(repository: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository / value


def _relative(repository: Path, path: Path) -> str:
    return str(path.relative_to(repository)).replace("\\", "/")


def _utc(wall_ns: int) -> str:
    return datetime.fromtimestamp(
        wall_ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_hashed_artifact(
    path: Path,
    *,
    schema_version: str,
    hash_field: str,
    label: str,
) -> tuple[dict[str, object], str]:
    value = _load_json_object(path, label)
    canonical = dict(value)
    claimed = str(canonical.pop(hash_field, ""))
    if value.get("schema_version") != schema_version or claimed != _canonical_sha256(
        canonical
    ):
        raise ValueError(f"{label} identity differs")
    return value, claimed


def _apply_terminal_corrections(
    *,
    terminal: Mapping[str, object],
    plan: object,
    state_root: Path,
    outcomes: Sequence[object],
) -> tuple[object, ...]:
    corrections = terminal.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise ValueError("Round 74 terminal correction panel differs")
    corrected = list(outcomes)
    observed_ordinals: set[int] = set()
    for row in corrections:
        if not isinstance(row, dict):
            raise ValueError("Round 74 terminal correction row differs")
        ordinal = row.get("slot_ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < len(corrected)
            or ordinal in observed_ordinals
        ):
            raise ValueError("Round 74 terminal correction ordinal differs")
        prior = corrected[ordinal]
        adjudication_path = (
            state_root / f"slot-{ordinal:03d}" / ("recovery-adjudication.json")
        )
        if (
            adjudication_path.is_symlink()
            or not adjudication_path.is_file()
            or row.get("adjudication_file")
            != f"slot-{ordinal:03d}/recovery-adjudication.json"
            or row.get("adjudication_file_sha256") != _file_sha256(adjudication_path)
            or row.get("prior_status") != getattr(prior, "status")
            or row.get("prior_reason_code") != getattr(prior, "reason_code")
        ):
            raise ValueError("Round 74 terminal correction evidence differs")
        adjudication = load_round74_segmented_slot_adjudication(
            adjudication_path.read_text(encoding="utf-8"),
            plan=plan,
        )
        binding = adjudication.outcome.binding
        if (
            adjudication.outcome.status != row.get("corrected_status")
            or adjudication.outcome.reason_code != row.get("corrected_reason_code")
            or adjudication.adjudication_sha256 != row.get("adjudication_sha256")
            or binding is None
            or binding.binding_sha256 != row.get("binding_sha256")
        ):
            raise ValueError("Round 74 terminal correction adjudication differs")
        corrected[ordinal] = adjudication.outcome
        observed_ordinals.add(ordinal)
    return tuple(corrected)


def _contiguous_blocks(
    ordinals: Sequence[int], plan: object
) -> list[dict[str, object]]:
    if not ordinals:
        return []
    blocks: list[tuple[int, int]] = []
    start = previous = ordinals[0]
    for ordinal in ordinals[1:]:
        if ordinal != previous + 1:
            blocks.append((start, previous))
            start = ordinal
        previous = ordinal
    blocks.append((start, previous))
    result: list[dict[str, object]] = []
    for first, last in blocks:
        first_slot = plan.slot(first)
        last_slot = plan.slot(last)
        count = last - first + 1
        result.append(
            {
                "first_slot_ordinal": first,
                "last_slot_ordinal": last,
                "slot_count": count,
                "role": first_slot.role
                if first_slot.role == last_slot.role
                else "cross_role",
                "scheduled_window_start_utc": _utc(first_slot.scheduled_start_wall_ns),
                "scheduled_window_end_utc": _utc(last_slot.scheduled_end_wall_ns),
                "major_gap": count >= MAJOR_GAP_MINIMUM_SLOTS,
            }
        )
    return result


def _render_svg(
    *,
    rows: Sequence[Mapping[str, object]],
    terminal_artifact_sha256: str,
    plan_sha256: str,
    required_training_ns: int,
    observed_training_ns: int,
    major_blocks: Sequence[Mapping[str, object]],
) -> str:
    width = 1440
    height = 610
    left = 110.0
    right = 1390.0
    plot_width = right - left
    timeline_y = 128.0
    timeline_height = 88.0
    role_y = 226.0
    role_height = 15.0
    training_y_top = 335.0
    training_y_bottom = 535.0
    training_height = training_y_bottom - training_y_top
    status_colors = {
        "admitted": "#1f9d74",
        "transport_excluded": "#d89922",
        "missed": "#c84b53",
    }
    role_colors = {
        "training": "#4f79a7",
        "tuning": "#9b6ab4",
        "test": "#617160",
    }
    slot_width = plot_width / len(rows)
    bars: list[str] = []
    role_bars: list[str] = []
    cumulative_points: list[str] = []
    training_rows = [row for row in rows if row["role"] == "training"]
    for index, row in enumerate(rows):
        x = left + index * slot_width
        color = status_colors[str(row["status"])]
        bars.append(
            f'<rect x="{x:.3f}" y="{timeline_y:.1f}" width="{slot_width + 0.05:.3f}" '
            f'height="{timeline_height:.1f}" fill="{color}"/>'
        )
        role_bars.append(
            f'<rect x="{x:.3f}" y="{role_y:.1f}" width="{slot_width + 0.05:.3f}" '
            f'height="{role_height:.1f}" fill="{role_colors[str(row["role"])]}"/>'
        )
    for index, row in enumerate(training_rows):
        x = left + (index / max(1, len(training_rows) - 1)) * plot_width
        cumulative = int(row["cumulative_training_eligible_anchor_ns"])
        y = training_y_bottom - min(1.0, cumulative / required_training_ns) * (
            training_height
        )
        cumulative_points.append(f"{x:.3f},{y:.3f}")
    gap_labels: list[str] = []
    for block in major_blocks:
        first = int(block["first_slot_ordinal"])
        last = int(block["last_slot_ordinal"])
        x = left + first * slot_width
        gap_width = (last - first + 1) * slot_width
        label = f"slots {first}-{last} ({last - first + 1})"
        gap_labels.append(
            f'<rect x="{x:.3f}" y="{timeline_y - 6:.1f}" width="{gap_width:.3f}" '
            f'height="{timeline_height + 12:.1f}" fill="none" stroke="#701f28" '
            'stroke-width="2"/>'
            f'<text x="{x + gap_width / 2:.3f}" y="{timeline_y - 13:.1f}" '
            f'text-anchor="middle" class="gap">{html.escape(label)}</text>'
        )
    required_hours = required_training_ns / 3_600_000_000_000
    observed_hours = observed_training_ns / 3_600_000_000_000
    observed_y = training_y_bottom - (observed_training_ns / required_training_ns) * (
        training_height
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">Round 74 terminal slot coverage and training quota</title>
  <desc id="description">All 720 preregistered slots, their terminal classifications, and cumulative admitted training duration against the frozen requirement.</desc>
  <style>
    text {{ font-family: "Segoe UI", Arial, sans-serif; fill: #17212b; letter-spacing: 0; }}
    .title {{ font-size: 25px; font-weight: 700; }}
    .subtitle {{ font-size: 15px; fill: #43515f; }}
    .label {{ font-size: 14px; font-weight: 600; }}
    .tick {{ font-size: 12px; fill: #526171; }}
    .gap {{ font-size: 12px; font-weight: 700; fill: #701f28; }}
    .footer {{ font-size: 11px; fill: #607080; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left:.0f}" y="42" class="title">Round 74 terminal campaign diagnosis</text>
  <text x="{left:.0f}" y="70" class="subtitle">460 admitted | 22 transport-excluded | 238 missed | no model or profitability result</text>
  <text x="{left:.0f}" y="102" class="label">Terminal result by preregistered slot</text>
  {"".join(bars)}
  {"".join(role_bars)}
  {"".join(gap_labels)}
  <text x="{left:.0f}" y="263" class="tick">0</text>
  <text x="{right:.0f}" y="263" text-anchor="end" class="tick">719</text>
  <rect x="905" y="87" width="13" height="13" fill="#1f9d74"/><text x="925" y="99" class="tick">Admitted</text>
  <rect x="1000" y="87" width="13" height="13" fill="#d89922"/><text x="1020" y="99" class="tick">Transport-excluded</text>
  <rect x="1165" y="87" width="13" height="13" fill="#c84b53"/><text x="1185" y="99" class="tick">Missed</text>
  <text x="{left:.0f}" y="307" class="label">Cumulative admitted training duration</text>
  <line x1="{left:.0f}" y1="{training_y_top:.1f}" x2="{right:.0f}" y2="{training_y_top:.1f}" stroke="#8d2730" stroke-width="2" stroke-dasharray="7 5"/>
  <line x1="{left:.0f}" y1="{training_y_bottom:.1f}" x2="{right:.0f}" y2="{training_y_bottom:.1f}" stroke="#c7d0d8" stroke-width="1"/>
  <line x1="{left:.0f}" y1="{observed_y:.3f}" x2="{right:.0f}" y2="{observed_y:.3f}" stroke="#9eabb6" stroke-width="1" stroke-dasharray="3 5"/>
  <polyline points="{" ".join(cumulative_points)}" fill="none" stroke="#1f6f9e" stroke-width="3"/>
  <text x="{left - 12:.0f}" y="{training_y_top + 5:.1f}" text-anchor="end" class="tick">{required_hours:.2f} h required</text>
  <text x="{left - 12:.0f}" y="{observed_y + 5:.1f}" text-anchor="end" class="tick">{observed_hours:.2f} h observed</text>
  <text x="{left - 12:.0f}" y="{training_y_bottom + 5:.1f}" text-anchor="end" class="tick">0 h</text>
  <text x="{left:.0f}" y="563" class="subtitle">Result: training quota failed by {required_hours - observed_hours:.2f} h. Tuning and test quotas passed but remain unopened for model evaluation.</text>
  <text x="{left:.0f}" y="588" class="footer">Source-bound terminal outcome {html.escape(terminal_artifact_sha256[:16])}... | plan {html.escape(plan_sha256[:16])}... | exact host cause for missed scheduling gaps is unestablished</text>
</svg>
"""
    return svg


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
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--recovery-build", type=Path, required=True)
    parser.add_argument(
        "--terminal-outcome",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-campaign-outcome-v2-2026-08-10.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-failure-analysis-2026-08-10.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-slot-coverage-2026-08-10.csv"
        ),
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-slot-coverage-2026-08-10.svg"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    plan_path = _resolve(repository, arguments.plan)
    state_root = _resolve(repository, arguments.state_root)
    recovery_root = _resolve(repository, arguments.recovery)
    recovery_build_path = _resolve(repository, arguments.recovery_build)
    terminal_path = _resolve(repository, arguments.terminal_outcome)
    output_path = _resolve(repository, arguments.output)
    csv_path = _resolve(repository, arguments.csv_output)
    chart_path = _resolve(repository, arguments.chart_output)
    outputs = (output_path, csv_path, chart_path)
    created: list[Path] = []
    try:
        inputs = (
            plan_path,
            state_root,
            recovery_root,
            recovery_build_path,
            terminal_path,
        )
        if (
            any(path.is_symlink() for path in (*inputs, *outputs))
            or not plan_path.is_file()
            or not state_root.is_dir()
            or not recovery_root.is_dir()
            or not recovery_build_path.is_file()
            or not terminal_path.is_file()
            or any(path.exists() for path in outputs)
            or any(path.parent.is_symlink() for path in outputs)
        ):
            raise ValueError("Round 74 failure-analysis path panel differs")
        for output in outputs:
            output.relative_to(repository)

        plan = load_round74_segmented_cohort_plan(plan_path.read_text(encoding="utf-8"))
        terminal, terminal_sha256 = _validate_hashed_artifact(
            terminal_path,
            schema_version=TERMINAL_SCHEMA_VERSION,
            hash_field="artifact_sha256",
            label="Round 74 terminal outcome",
        )
        terminal_sources = terminal.get("source_bindings")
        terminal_coverage = terminal.get("terminal_coverage")
        terminal_decision = terminal.get("decision")
        terminal_scope = terminal.get("scope")
        if (
            not isinstance(terminal_sources, dict)
            or not isinstance(terminal_coverage, dict)
            or not isinstance(terminal_decision, dict)
            or not isinstance(terminal_scope, dict)
            or terminal_sources.get("plan_sha256") != plan.plan_sha256
            or terminal_sources.get("plan_file_sha256") != _file_sha256(plan_path)
            or terminal_decision.get("model_data_eligible") is not False
            or terminal_decision.get("representative_training_performed") is not False
            or terminal_decision.get("sealed_target_manifests_read") is not False
            or terminal_scope.get("source_database_access") != "not_opened"
            or terminal_scope.get("target_data_accessed") is not False
        ):
            raise ValueError("Round 74 terminal outcome scope differs")

        outcomes, slot_evidence, state_metadata = _load_complete_campaign_outcomes(
            plan,
            state_root=state_root,
            recovery_directory=recovery_root,
        )
        if terminal_sources.get("prior_outcome_panel_sha256") != _canonical_sha256(
            [outcome.as_dict() for outcome in outcomes]
        ) or terminal_sources.get("state_metadata_evidence") != dict(state_metadata):
            raise ValueError("Round 74 terminal source evidence differs")
        corrected = _apply_terminal_corrections(
            terminal=terminal,
            plan=plan,
            state_root=state_root,
            outcomes=outcomes,
        )
        if terminal_sources.get("corrected_outcome_panel_sha256") != _canonical_sha256(
            [outcome.as_dict() for outcome in corrected]
        ):
            raise ValueError("Round 74 corrected outcome panel differs")

        evidence_by_slot = {
            int(ordinal): str(kind) for ordinal, kind, _digest in slot_evidence
        }
        if len(evidence_by_slot) != plan.total_slots:
            raise ValueError("Round 74 slot evidence coverage differs")
        for correction in terminal["corrections"]:
            evidence_by_slot[int(correction["slot_ordinal"])] = (
                "supplemental_adjudication"
            )

        recovery_build, recovery_build_sha256 = _validate_hashed_artifact(
            recovery_build_path,
            schema_version=RECOVERY_BUILD_SCHEMA_VERSION,
            hash_field="result_sha256",
            label="Round 74 recovery build",
        )
        conservative_rows = recovery_build.get("conservative_recoveries")
        if recovery_build.get("plan_sha256") != plan.plan_sha256 or not isinstance(
            conservative_rows, list
        ):
            raise ValueError("Round 74 recovery build scope differs")
        conservative_by_slot: dict[int, str] = {}
        for row in conservative_rows:
            if not isinstance(row, dict):
                raise ValueError("Round 74 conservative recovery row differs")
            ordinal = row.get("slot_ordinal")
            reason = row.get("reason")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or not isinstance(reason, str)
                or not reason
                or ordinal in conservative_by_slot
            ):
                raise ValueError("Round 74 conservative recovery identity differs")
            conservative_by_slot[ordinal] = reason

        partition = _build_unqualified_partition(plan, corrected)
        eligible_by_run = {
            entry.run_id: (
                entry.eligible_anchor_end_wall_ns - entry.eligible_anchor_start_wall_ns
            )
            for entry in partition.entries
        }
        corrected_counts = Counter(outcome.status for outcome in corrected)
        if terminal_coverage.get("outcome_status_counts") != dict(
            sorted(corrected_counts.items())
        ):
            raise ValueError("Round 74 terminal status counts differ")

        rows: list[dict[str, object]] = []
        failure_counts: Counter[str] = Counter()
        no_directory_ordinals: list[int] = []
        cumulative_training_ns = 0
        for ordinal, outcome in enumerate(corrected):
            slot = plan.slot(ordinal)
            slot_directory = state_root / f"slot-{ordinal:03d}"
            directory_present = (
                slot_directory.is_dir() and not slot_directory.is_symlink()
            )
            eligible_ns = 0
            if outcome.status == "admitted":
                binding = outcome.binding
                if binding is None or binding.run_id not in eligible_by_run:
                    raise ValueError("Round 74 admitted slot partition differs")
                eligible_ns = eligible_by_run[binding.run_id]
                classification = "admitted"
            elif outcome.status == "transport_excluded":
                classification = "transport_excluded"
            elif outcome.status != "missed":
                raise ValueError("Round 74 terminal outcome status differs")
            elif not directory_present:
                classification = "never_started_no_slot_directory"
                no_directory_ordinals.append(ordinal)
            elif ordinal in conservative_by_slot:
                classification = f"captured_{conservative_by_slot[ordinal]}"
            else:
                audit_error = slot_directory / "recovery-audit-error.log"
                supervisor = slot_directory / "capture.stdout.json"
                if audit_error.is_file() and not audit_error.is_symlink():
                    error_text = audit_error.read_text(encoding="utf-8")
                    if "epoch audit terminal class differs" not in error_text:
                        raise ValueError("Round 74 recovery audit error class differs")
                    classification = "prior_audit_terminal_class_failed"
                elif supervisor.is_file() and not supervisor.is_symlink():
                    if supervisor.read_text(encoding="utf-8").strip():
                        raise ValueError(
                            "Round 74 unclassified supervisor output differs"
                        )
                    classification = "empty_supervisor_output"
                else:
                    raise ValueError("Round 74 attempted missed slot is unclassified")
            if slot.role == "training":
                cumulative_training_ns += eligible_ns
            failure_counts[classification] += 1
            rows.append(
                {
                    "slot_ordinal": ordinal,
                    "role": slot.role,
                    "scheduled_start_utc": _utc(slot.scheduled_start_wall_ns),
                    "scheduled_end_utc": _utc(slot.scheduled_end_wall_ns),
                    "status": outcome.status,
                    "reason_code": outcome.reason_code,
                    "failure_classification": classification,
                    "evidence_kind": evidence_by_slot[ordinal],
                    "slot_directory_present": str(directory_present).lower(),
                    "eligible_anchor_ns": eligible_ns,
                    "cumulative_training_eligible_anchor_ns": cumulative_training_ns,
                }
            )

        role_quotas = terminal_coverage.get("role_quotas")
        if not isinstance(role_quotas, dict) or not isinstance(
            role_quotas.get("training"), dict
        ):
            raise ValueError("Round 74 terminal role quota panel differs")
        training = role_quotas["training"]
        observed_training_ns = int(training["observed_eligible_anchor_ns"])
        required_training_ns = int(training["required_eligible_anchor_ns"])
        if cumulative_training_ns != observed_training_ns:
            raise ValueError("Round 74 per-slot training duration differs")

        blocks = _contiguous_blocks(no_directory_ordinals, plan)
        major_blocks = [block for block in blocks if block["major_gap"]]
        major_gap_slots = sum(int(block["slot_count"]) for block in major_blocks)
        started_slots = sum(row["slot_directory_present"] == "true" for row in rows)
        admitted_count = corrected_counts["admitted"]
        chart = _render_svg(
            rows=rows,
            terminal_artifact_sha256=terminal_sha256,
            plan_sha256=plan.plan_sha256,
            required_training_ns=required_training_ns,
            observed_training_ns=observed_training_ns,
            major_blocks=major_blocks,
        )

        _atomic_csv(csv_path, rows, CSV_FIELDS)
        created.append(csv_path)
        _atomic_text(chart_path, chart)
        created.append(chart_path)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "observed_at_utc": terminal["observed_at_utc"],
            "source_bindings": {
                "terminal_outcome_path": _relative(repository, terminal_path),
                "terminal_outcome_file_sha256": _file_sha256(terminal_path),
                "terminal_outcome_artifact_sha256": terminal_sha256,
                "plan_path": _relative(repository, plan_path),
                "plan_file_sha256": _file_sha256(plan_path),
                "plan_sha256": plan.plan_sha256,
                "recovery_build_location": "local_data_artifact_not_committed",
                "recovery_build_file_sha256": _file_sha256(recovery_build_path),
                "recovery_build_result_sha256": recovery_build_sha256,
                "state_and_recovery_location": "local_data_artifacts_not_committed",
                "state_metadata_evidence": dict(state_metadata),
                "slot_csv_path": _relative(repository, csv_path),
                "slot_csv_file_sha256": _file_sha256(csv_path),
                "chart_path": _relative(repository, chart_path),
                "chart_file_sha256": _file_sha256(chart_path),
            },
            "terminal_slot_analysis": {
                "total_slot_count": len(rows),
                "slot_directory_count": started_slots,
                "never_started_slot_count": len(no_directory_ordinals),
                "admitted_count": admitted_count,
                "transport_excluded_count": corrected_counts["transport_excluded"],
                "attempted_missed_count": corrected_counts["missed"]
                - len(no_directory_ordinals),
                "admitted_fraction_of_started_slots": admitted_count / started_slots,
                "failure_classification_counts": dict(sorted(failure_counts.items())),
                "never_started_contiguous_blocks": blocks,
                "major_gap_minimum_slots": MAJOR_GAP_MINIMUM_SLOTS,
                "major_gap_slot_count": major_gap_slots,
                "major_gap_fraction_of_never_started_slots": (
                    major_gap_slots / len(no_directory_ordinals)
                ),
            },
            "training_quota": {
                "observed_eligible_anchor_ns": observed_training_ns,
                "required_eligible_anchor_ns": required_training_ns,
                "deficit_eligible_anchor_ns": required_training_ns
                - observed_training_ns,
                "observed_hours": observed_training_ns / 3_600_000_000_000,
                "required_hours": required_training_ns / 3_600_000_000_000,
                "deficit_hours": (required_training_ns - observed_training_ns)
                / 3_600_000_000_000,
                "quota_passed": False,
            },
            "diagnosis": {
                "classification": "campaign_continuity_failure_indicated",
                "exact_host_cause_established": False,
                "market_feed_failure_established_as_primary_cause": False,
                "basis": (
                    "Most never-started slots are concentrated in two long "
                    "preregistered scheduling gaps; terminal evidence does not "
                    "identify the exact host-level cause."
                ),
                "model_architecture_evaluated": False,
                "predictive_edge_evaluated": False,
                "profitability_evaluated": False,
            },
            "scope": {
                "source_database_access": "not_opened",
                "target_data_accessed": False,
                "model_training_performed": False,
                "credentials_used": False,
                "orders_submitted": False,
                "paper_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        payload["artifact_sha256"] = _canonical_sha256(payload)
        write_json_atomic(output_path, payload, indent=2, sort_keys=True)
        created.append(output_path)
        restored, restored_sha256 = _validate_hashed_artifact(
            output_path,
            schema_version=SCHEMA_VERSION,
            hash_field="artifact_sha256",
            label="Round 74 terminal failure analysis",
        )
        if restored_sha256 != payload["artifact_sha256"] or restored != payload:
            raise RuntimeError("Round 74 failure-analysis reload differs")
    except (OSError, RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        print(
            f"Round 74 failure analysis failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
