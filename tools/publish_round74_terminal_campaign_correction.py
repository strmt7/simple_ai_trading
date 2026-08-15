"""Supersede conservative Round 74 terminal counts without rewriting evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
for import_root in (REPOSITORY, SOURCE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.impact_absorption_event_dataset import (  # noqa: E402
    ROUND74_EVENT_PARTITION_ROLES,
)
from simple_ai_trading.impact_absorption_event_segmented_cohort import (  # noqa: E402
    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS,
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.round74_segmented_cohort_operator import (  # noqa: E402
    load_round74_segmented_slot_adjudication,
)
from simple_ai_trading.round74_segmented_development_inputs import (  # noqa: E402
    ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION,
    Round74SegmentedRecoveryOutcome,
    _load_complete_campaign_outcomes,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_round74_terminal_campaign_outcome import (  # noqa: E402
    _build_unqualified_partition,
)


SCHEMA_VERSION = "round-074-terminal-campaign-outcome-v2"
PRIOR_SCHEMA_VERSION = "round-074-terminal-campaign-outcome-v1"


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


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root differs")
    return value


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
    parser.add_argument(
        "--prior-terminal-outcome",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-campaign-outcome-2026-08-10.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-terminal-campaign-outcome-v2-2026-08-10.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    plan_path = _resolve(repository, arguments.plan)
    state_root = _resolve(repository, arguments.state_root)
    recovery_root = _resolve(repository, arguments.recovery)
    prior_path = _resolve(repository, arguments.prior_terminal_outcome)
    output_path = _resolve(repository, arguments.output)
    try:
        paths = (plan_path, state_root, recovery_root, prior_path, output_path)
        if (
            any(path.is_symlink() for path in paths)
            or not plan_path.is_file()
            or not state_root.is_dir()
            or not recovery_root.is_dir()
            or not prior_path.is_file()
            or output_path.exists()
            or output_path.parent.is_symlink()
        ):
            raise ValueError("Round 74 terminal correction path panel differs")
        plan = load_round74_segmented_cohort_plan(plan_path.read_text(encoding="utf-8"))
        prior = _load_json_object(prior_path, "prior Round 74 terminal outcome")
        prior_canonical = dict(prior)
        prior_claimed_sha256 = str(prior_canonical.pop("artifact_sha256", ""))
        prior_decision = prior.get("decision")
        prior_scope = prior.get("scope")
        if (
            prior.get("schema_version") != PRIOR_SCHEMA_VERSION
            or prior_claimed_sha256 != _canonical_sha256(prior_canonical)
            or not isinstance(prior_decision, dict)
            or not isinstance(prior_scope, dict)
            or prior_decision.get("model_data_eligible") is not False
            or prior_scope.get("source_database_access") != "not_opened"
        ):
            raise ValueError("Prior Round 74 terminal outcome differs")
        outcomes, slot_evidence, state_metadata = _load_complete_campaign_outcomes(
            plan,
            state_root=state_root,
            recovery_directory=recovery_root,
        )
        prior_sources = prior.get("source_bindings")
        if (
            not isinstance(prior_sources, dict)
            or prior_sources.get("outcome_panel_sha256")
            != _canonical_sha256([outcome.as_dict() for outcome in outcomes])
            or prior_sources.get("slot_evidence_panel_sha256")
            != _canonical_sha256(slot_evidence)
            or prior_sources.get("state_metadata_evidence") != dict(state_metadata)
        ):
            raise ValueError("Prior Round 74 terminal source panel differs")

        corrected = list(outcomes)
        corrections: list[dict[str, object]] = []
        for ordinal, outcome in enumerate(outcomes):
            if outcome.status != "missed":
                continue
            receipt_path = recovery_root / f"{ordinal:03d}.json"
            slot_directory = state_root / f"slot-{ordinal:03d}"
            adjudication_path = slot_directory / "recovery-adjudication.json"
            if not receipt_path.is_file() or not adjudication_path.is_file():
                continue
            receipt_payload = _load_json_object(
                receipt_path,
                "Round 74 conservative recovery receipt",
            )
            if (
                receipt_payload.get("schema_version")
                != ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION
            ):
                continue
            receipt = Round74SegmentedRecoveryOutcome.from_dict(
                plan,
                receipt_payload,
            )
            receipt.verify_slot_directory(slot_directory)
            adjudication = load_round74_segmented_slot_adjudication(
                adjudication_path.read_text(encoding="utf-8"),
                plan=plan,
            )
            supervisor = _load_json_object(
                slot_directory / "capture.stdout.json",
                "Round 74 correction supervisor",
            )
            receipt_files = dict(receipt.slot_file_sha256)
            if (
                receipt.outcome.as_dict() != outcome.as_dict()
                or receipt_files.get("recovery-adjudication.json")
                != _file_sha256(adjudication_path)
                or supervisor != json.loads(adjudication.supervisor_json)
                or adjudication.outcome.status != "admitted"
                or adjudication.outcome.binding is None
            ):
                raise ValueError("Round 74 terminal correction evidence differs")
            corrected[ordinal] = adjudication.outcome
            corrections.append(
                {
                    "slot_ordinal": ordinal,
                    "role": outcome.role,
                    "prior_status": outcome.status,
                    "prior_reason_code": outcome.reason_code,
                    "corrected_status": adjudication.outcome.status,
                    "corrected_reason_code": adjudication.outcome.reason_code,
                    "recovery_receipt_file": receipt_path.name,
                    "recovery_receipt_file_sha256": _file_sha256(receipt_path),
                    "recovery_receipt_sha256": receipt.recovery_sha256,
                    "adjudication_file": (
                        f"slot-{ordinal:03d}/recovery-adjudication.json"
                    ),
                    "adjudication_file_sha256": _file_sha256(adjudication_path),
                    "adjudication_sha256": adjudication.adjudication_sha256,
                    "binding_sha256": adjudication.outcome.binding.binding_sha256,
                }
            )
        if not corrections:
            raise ValueError("Round 74 terminal correction panel is empty")

        partition = _build_unqualified_partition(plan, tuple(corrected))
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
                    plan.slot(index).role == role for index in range(plan.total_slots)
                ),
                "admitted_count": sum(
                    outcome.role == role and outcome.status == "admitted"
                    for outcome in corrected
                ),
                "transport_excluded_count": sum(
                    outcome.role == role and outcome.status == "transport_excluded"
                    for outcome in corrected
                ),
                "missed_count": sum(
                    outcome.role == role and outcome.status == "missed"
                    for outcome in corrected
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
            raise ValueError("Round 74 correction unexpectedly qualifies the cohort")
        outcome_counts = Counter(outcome.status for outcome in corrected)
        evidence_counts = Counter(kind for _ordinal, kind, _digest in slot_evidence)
        evidence_counts["recovery"] -= len(corrections)
        evidence_counts["supplemental_adjudication"] += len(corrections)
        if any(count < 0 for count in evidence_counts.values()):
            raise ValueError("Round 74 corrected evidence counts differ")

        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "observed_at_utc": prior["observed_at_utc"],
            "supersedes": {
                "artifact_path": str(prior_path.relative_to(repository)).replace(
                    "\\", "/"
                ),
                "artifact_file_sha256": _file_sha256(prior_path),
                "artifact_sha256": prior_claimed_sha256,
                "reason": "conservative_receipt_bound_valid_prior_adjudication",
            },
            "source_bindings": {
                "plan_path": str(plan_path.relative_to(repository)).replace("\\", "/"),
                "plan_file_sha256": _file_sha256(plan_path),
                "plan_sha256": plan.plan_sha256,
                "state_and_recovery_location": "local_data_artifacts_not_committed",
                "prior_outcome_panel_sha256": prior_sources["outcome_panel_sha256"],
                "prior_slot_evidence_panel_sha256": prior_sources[
                    "slot_evidence_panel_sha256"
                ],
                "corrected_outcome_panel_sha256": _canonical_sha256(
                    [outcome.as_dict() for outcome in corrected]
                ),
                "correction_panel_sha256": _canonical_sha256(corrections),
                "state_metadata_evidence": dict(state_metadata),
            },
            "corrections": corrections,
            "terminal_coverage": {
                "total_slots": len(corrected),
                "slot_evidence_kind_counts": dict(sorted(evidence_counts.items())),
                "outcome_status_counts": dict(sorted(outcome_counts.items())),
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
                "target_data_accessed": False,
                "credentials_used": False,
                "orders_submitted": False,
                "paper_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        payload["artifact_sha256"] = _canonical_sha256(payload)
        write_json_atomic(output_path, payload, indent=2, sort_keys=True)
        restored = _load_json_object(output_path, "Round 74 corrected terminal outcome")
        restored_claimed = restored.pop("artifact_sha256")
        if restored_claimed != _canonical_sha256(restored):
            raise RuntimeError("Round 74 terminal correction reload differs")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Round 74 terminal correction failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
