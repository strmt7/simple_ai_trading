"""Audit and adjudicate one captured Round 74 segmented slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.impact_absorption_store import ImpactAbsorptionStore
from simple_ai_trading.round74_segmented_cohort_operator import (
    audit_and_adjudicate_round74_segmented_supervisor,
)


def _strict_json(raw_text: str, label: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} has duplicate JSON keys")
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root differs")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently audit and adjudicate one segmented slot.",
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--slot-ordinal", required=True, type=int)
    parser.add_argument("--supervisor", required=True)
    parser.add_argument("--database")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--database-threads", default=2, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_round74_segmented_cohort_plan(
            Path(args.plan).read_text(encoding="utf-8")
        )
        supervisor = _strict_json(
            Path(args.supervisor).read_text(encoding="utf-8"),
            "supervisor",
        )
        attempts = supervisor.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("supervisor attempts differ")
        if attempts:
            if not args.database:
                raise ValueError("in-run adjudication requires --database")
            with ImpactAbsorptionStore(
                Path(args.database),
                read_only=True,
                memory_limit=str(args.memory_limit),
                threads=int(args.database_threads),
            ) as store:
                result = audit_and_adjudicate_round74_segmented_supervisor(
                    plan,
                    slot_ordinal=int(args.slot_ordinal),
                    supervisor_payload=supervisor,
                    store=store,
                )
        else:
            result = audit_and_adjudicate_round74_segmented_supervisor(
                plan,
                slot_ordinal=int(args.slot_ordinal),
                supervisor_payload=supervisor,
                store=None,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"round74-segmented-adjudication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
