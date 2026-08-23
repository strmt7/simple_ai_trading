"""Audit frozen Round 75 terminal metadata without opening campaign databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round75_capture_supervisor import (  # noqa: E402
    Round75CaptureSupervisorConfig,
    inspect_round75_capture_supervisor,
)
from simple_ai_trading.round75_continuous_capture import (  # noqa: E402
    Round75ContinuousCaptureConfig,
)
from simple_ai_trading.round75_terminal_audit import (  # noqa: E402
    Round75TerminalAuditConfig,
    audit_round75_terminal_capture,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-repository", type=Path, required=True)
    parser.add_argument(
        "--observed-at-utc",
        required=True,
        help="Explicit immutable UTC observation timestamp.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs/model-research/action-value/"
        "round-075-terminal-campaign-audit-2026-08-23.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capture_repository = args.capture_repository.resolve()
    contract = REPOSITORY / (
        "docs/model-research/action-value/round-075-continuous-capture-contract-v4.json"
    )
    plan = REPOSITORY / (
        "docs/model-research/action-value/"
        "round-075-prospective-event-cohort-plan-v1.json"
    )
    capture = Round75ContinuousCaptureConfig(
        repository=capture_repository,
        contract_path=capture_repository / contract.relative_to(REPOSITORY),
        plan_path=capture_repository / plan.relative_to(REPOSITORY),
        prerequisite_path=capture_repository / "docs/model-research/action-value/"
        "round-074-segmented-prerequisite-attempt-003-success-2026-07-28.json",
        data_root=capture_repository / "data/round75-prospective-event-cohort",
        state_root=capture_repository / "data/round75-prospective-event-cohort-state",
        service_state_path=capture_repository
        / "data/round75-prospective-event-cohort-service-state.json",
        lease_path=capture_repository
        / "data/round75-prospective-event-cohort-service.lock",
        stop_request_path=capture_repository
        / "data/round75-prospective-event-cohort-stop.request",
    )
    supervisor = Round75CaptureSupervisorConfig(
        capture=capture,
        python_executable=Path(sys.executable).resolve(),
        service_tool_path=capture_repository
        / "tools/run_round75_continuous_capture.py",
        capture_tool_path=capture_repository / "tools/run_round74_segmented_capture.py",
        stdout_log_path=capture_repository
        / "data/round75-prospective-event-cohort-service.stdout.log",
        stderr_log_path=capture_repository
        / "data/round75-prospective-event-cohort-service.stderr.log",
    )
    inspection = inspect_round75_capture_supervisor(supervisor)
    audit = Round75TerminalAuditConfig(
        evidence_repository=REPOSITORY,
        capture_repository=capture_repository,
        contract_path=contract,
        activation_path=REPOSITORY / "docs/model-research/action-value/"
        "round-075-v4-host-activation-receipt-2026-08-10.json",
        plan_path=plan,
        data_root=capture.data_root,
        state_root=capture.state_root,
        service_state_path=capture.service_state_path,
        lease_path=capture.lease_path,
    )
    result = audit_round75_terminal_capture(
        audit,
        supervisor_inspection=inspection,
        observed_at_utc=args.observed_at_utc,
    )
    write_json_atomic(args.output, result, indent=2, sort_keys=True)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "rejected_incomplete_campaign" else 2


if __name__ == "__main__":
    raise SystemExit(main())
