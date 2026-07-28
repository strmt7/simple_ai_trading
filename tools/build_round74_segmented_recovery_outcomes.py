"""Seal result-less Round 74 slots as missed after the campaign ends."""

from __future__ import annotations

import argparse
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
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.round74_segmented_development_inputs import (  # noqa: E402
    ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION,
    build_round74_segmented_recovery_outcome,
    write_round74_segmented_recovery_outcome,
)


ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION = (
    "round-074-segmented-recovery-build-v1"
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
    return parser


def _resolve(repository: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository / path


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    plan_path = _resolve(repository, arguments.plan)
    state_root = _resolve(repository, arguments.state_root)
    output = _resolve(repository, arguments.output)
    observed_wall_ns = time.time_ns()
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
        ):
            raise ValueError("Round 74 segmented recovery build paths differ")
        plan = load_round74_segmented_cohort_plan(
            plan_path.read_text(encoding="utf-8")
        )
        output.mkdir(parents=True, exist_ok=True)
        recoveries = []
        for ordinal in range(plan.total_slots):
            slot_directory = state_root / f"slot-{ordinal:03d}"
            selected_directory = slot_directory if slot_directory.is_dir() else None
            if (
                selected_directory is not None
                and (selected_directory / "result.json").is_file()
            ):
                continue
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
            recoveries.append(
                {
                    "slot_ordinal": ordinal,
                    "path": str(path),
                    "recovery_sha256": recovery.recovery_sha256,
                }
            )
        expected_names = {
            f"{row['slot_ordinal']:03d}.json" for row in recoveries
        }
        output_entries = tuple(output.iterdir())
        if (
            {path.name for path in output_entries} != expected_names
            or any(path.is_symlink() or not path.is_file() for path in output_entries)
        ):
            raise ValueError(
                "Round 74 segmented recovery output panel differs"
            )
        result: dict[str, object] = {
            "schema_version": ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION,
            "recovery_schema_version": (
                ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION
            ),
            "plan_sha256": plan.plan_sha256,
            "observed_wall_ns": observed_wall_ns,
            "recovery_count": len(recoveries),
            "recoveries": recoveries,
            "admitted_data_created": False,
            "database_opened": False,
            "orders_submitted": False,
            "profitability_or_edge_claim": False,
            "trading_authority": False,
        }
        result["result_sha256"] = _canonical_sha256(result)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": (
                        ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION
                    ),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "admitted_data_created": False,
                    "database_opened": False,
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
