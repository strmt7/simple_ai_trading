"""Run the leased Round 75 prospective capture scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round75_continuous_capture import (  # noqa: E402
    Round75ContinuousCaptureConfig,
    run_round75_continuous_capture,
)


def _resolve(repository: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository / value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-075-continuous-capture-contract-v4.json"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-075-prospective-event-cohort-plan-v1.json"
        ),
    )
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path(
            "docs/model-research/action-value/"
            "round-074-segmented-prerequisite-attempt-003-success-2026-07-28.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/round75-prospective-event-cohort"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("data/round75-prospective-event-cohort-state"),
    )
    parser.add_argument(
        "--service-state",
        type=Path,
        default=Path("data/round75-prospective-event-cohort-service-state.json"),
    )
    parser.add_argument(
        "--lease",
        type=Path,
        default=Path("data/round75-prospective-event-cohort-service.lock"),
    )
    parser.add_argument(
        "--stop-request",
        type=Path,
        default=Path("data/round75-prospective-event-cohort-stop.request"),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform at most one scheduler action and return.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    config = Round75ContinuousCaptureConfig(
        repository=repository,
        contract_path=_resolve(repository, arguments.contract),
        plan_path=_resolve(repository, arguments.plan),
        prerequisite_path=_resolve(repository, arguments.prerequisite),
        data_root=_resolve(repository, arguments.data_root),
        state_root=_resolve(repository, arguments.state_root),
        service_state_path=_resolve(repository, arguments.service_state),
        lease_path=_resolve(repository, arguments.lease),
        stop_request_path=_resolve(repository, arguments.stop_request),
    )
    try:
        state = run_round75_continuous_capture(
            config,
            once=bool(arguments.once),
        )
    except KeyboardInterrupt:
        print(
            "round75-continuous-capture interrupted; the external supervisor "
            "must reconcile any exact owned child before restart",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"round75-continuous-capture failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if arguments.json:
        print(json.dumps(state, allow_nan=False, ensure_ascii=True, sort_keys=True))
    else:
        print(
            "round75-continuous-capture: "
            f"phase={state['phase']} slot={state['slot_ordinal']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
