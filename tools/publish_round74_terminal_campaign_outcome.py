"""Publish the terminal Round 74 campaign quota outcome without target access."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_event_dataset import (  # noqa: E402
    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
    ROUND74_EVENT_PARTITION_ROLES,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)
from simple_ai_trading.impact_absorption_event_segmented_cohort import (  # noqa: E402
    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS,
    Round74SegmentedCohortPlan,
    Round74SegmentedCohortSlotOutcome,
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.round74_segmented_development_inputs import (  # noqa: E402
    _load_complete_campaign_outcomes,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA_VERSION = "round-074-terminal-campaign-outcome-v1"


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


def _resolve(repository: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository / value


def _build_unqualified_partition(
    plan: Round74SegmentedCohortPlan,
    outcomes: tuple[Round74SegmentedCohortSlotOutcome, ...],
) -> Round74EventRunPartition:
    entries: list[Round74EventRunPartitionEntry] = []
    prior_role: str | None = None
    for outcome in outcomes:
        outcome.validate()
        binding = outcome.binding
        if binding is None:
            continue
        role_changed = prior_role is not None and binding.role != prior_role
        anchor_start = binding.feature_ready_wall_ns
        anchor_end = (
            binding.usable_end_wall_ns - ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
        )
        if role_changed:
            previous = entries[-1]
            entries[-1] = Round74EventRunPartitionEntry(
                run_id=previous.run_id,
                role=previous.role,
                capture_report_sha256=previous.capture_report_sha256,
                capture_start_wall_ns=previous.capture_start_wall_ns,
                capture_end_wall_ns=previous.capture_end_wall_ns,
                eligible_anchor_start_wall_ns=(previous.eligible_anchor_start_wall_ns),
                eligible_anchor_end_wall_ns=min(
                    previous.eligible_anchor_end_wall_ns,
                    previous.capture_end_wall_ns
                    - ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
                ),
            )
            anchor_start = max(
                anchor_start,
                previous.capture_end_wall_ns
                + ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
            )
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=binding.run_id,
                role=binding.role,
                capture_report_sha256=binding.report_sha256,
                capture_start_wall_ns=binding.feature_ready_wall_ns,
                capture_end_wall_ns=binding.usable_end_wall_ns,
                eligible_anchor_start_wall_ns=anchor_start,
                eligible_anchor_end_wall_ns=anchor_end,
            )
        )
        prior_role = binding.role
    partition = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256=plan.plan_sha256,
    )
    partition.validate()
    return partition


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
        "--recovery",
        type=Path,
        default=Path("data/round74-segmented-event-cohort-v3-recovery"),
    )
    parser.add_argument("--recovery-build", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-campaign-outcome-2026-08-10.json"
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
    output_path = _resolve(repository, arguments.output)
    try:
        paths = (plan_path, state_root, recovery_root, recovery_build_path, output_path)
        if (
            any(path.is_symlink() for path in paths)
            or not plan_path.is_file()
            or not state_root.is_dir()
            or not recovery_root.is_dir()
            or not recovery_build_path.is_file()
            or output_path.exists()
            or output_path.parent.is_symlink()
        ):
            raise ValueError("Round 74 terminal publication path panel differs")
        plan = load_round74_segmented_cohort_plan(plan_path.read_text(encoding="utf-8"))
        recovery_build = json.loads(recovery_build_path.read_text(encoding="utf-8"))
        if not isinstance(recovery_build, dict):
            raise ValueError("Round 74 recovery build root differs")
        claimed_recovery_build_sha256 = str(recovery_build.pop("result_sha256", ""))
        if claimed_recovery_build_sha256 != _canonical_sha256(recovery_build):
            raise ValueError("Round 74 recovery build hash differs")
        observed_wall_ns = recovery_build.get("observed_wall_ns")
        if isinstance(observed_wall_ns, bool) or not isinstance(observed_wall_ns, int):
            raise ValueError("Round 74 recovery observation time differs")
        outcomes, slot_evidence, state_metadata = _load_complete_campaign_outcomes(
            plan,
            state_root=state_root,
            recovery_directory=recovery_root,
        )
        if (
            recovery_build.get("recovery_count")
            != sum(kind != "result" for _ordinal, kind, _digest in slot_evidence)
            or len(outcomes) != plan.total_slots
        ):
            raise ValueError("Round 74 terminal recovery coverage differs")
        partition = _build_unqualified_partition(plan, outcomes)
        eligible_by_role = {
            role: sum(
                entry.eligible_anchor_end_wall_ns - entry.eligible_anchor_start_wall_ns
                for entry in partition.entries
                if entry.role == role
            )
            for role in ROUND74_EVENT_PARTITION_ROLES
        }
        role_quotas = {
            role: {
                "planned_slot_count": sum(
                    plan.slot(i).role == role for i in range(plan.total_slots)
                ),
                "admitted_count": sum(
                    outcome.role == role and outcome.status == "admitted"
                    for outcome in outcomes
                ),
                "transport_excluded_count": sum(
                    outcome.role == role and outcome.status == "transport_excluded"
                    for outcome in outcomes
                ),
                "missed_count": sum(
                    outcome.role == role and outcome.status == "missed"
                    for outcome in outcomes
                ),
                "observed_eligible_anchor_ns": eligible_by_role[role],
                "required_eligible_anchor_ns": (
                    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS[role]
                ),
                "deficit_eligible_anchor_ns": max(
                    0,
                    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS[role]
                    - eligible_by_role[role],
                ),
                "quota_passed": eligible_by_role[role]
                >= ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS[role],
            }
            for role in ROUND74_EVENT_PARTITION_ROLES
        }
        if all(row["quota_passed"] for row in role_quotas.values()):
            raise ValueError("Round 74 terminal publication requires quota failure")
        slot_kind_counts = Counter(kind for _ordinal, kind, _digest in slot_evidence)
        outcome_status_counts = Counter(outcome.status for outcome in outcomes)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "observed_at_utc": datetime.fromtimestamp(
                observed_wall_ns / 1_000_000_000,
                tz=timezone.utc,
            ).isoformat(),
            "source_bindings": {
                "plan_path": str(plan_path.relative_to(repository)).replace("\\", "/"),
                "plan_file_sha256": _file_sha256(plan_path),
                "plan_sha256": plan.plan_sha256,
                "recovery_build_path": str(recovery_build_path.name),
                "recovery_build_location": "local_data_artifact_not_committed",
                "recovery_build_file_sha256": _file_sha256(recovery_build_path),
                "recovery_build_result_sha256": claimed_recovery_build_sha256,
                "state_metadata_evidence": dict(state_metadata),
                "slot_evidence_panel_sha256": _canonical_sha256(slot_evidence),
                "outcome_panel_sha256": _canonical_sha256(
                    [outcome.as_dict() for outcome in outcomes]
                ),
            },
            "terminal_coverage": {
                "total_slots": len(outcomes),
                "slot_evidence_kind_counts": dict(sorted(slot_kind_counts.items())),
                "outcome_status_counts": dict(sorted(outcome_status_counts.items())),
                "partition_sha256": partition.partition_sha256,
                "partition_entry_count": len(partition.entries),
                "role_quotas": role_quotas,
            },
            "decision": {
                "status": "campaign_cannot_qualify_model",
                "reason": "training_eligible_anchor_quota_failed",
                "model_data_eligible": False,
                "representative_training_performed": False,
                "sealed_target_manifests_read": False,
                "sealed_test_access_reserved_or_consumed": False,
                "predictive_edge_established": False,
                "profitability_established": False,
            },
            "scope": {
                "source_database_access": "not_opened",
                "credentials_used": False,
                "orders_submitted": False,
                "paper_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        payload["artifact_sha256"] = _canonical_sha256(payload)
        write_json_atomic(output_path, payload, indent=2, sort_keys=True)
        restored = json.loads(output_path.read_text(encoding="utf-8"))
        claimed = restored.pop("artifact_sha256")
        if claimed != _canonical_sha256(restored):
            raise RuntimeError("Round 74 terminal publication reload differs")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Round 74 terminal publication failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
