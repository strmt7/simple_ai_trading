"""CLI and Windows command surface for the Round 74 terminal evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .compute import SUPPORTED_COMPUTE_BACKENDS
from .round74_segmented_terminal_runtime import (
    recover_round74_segmented_terminal_result,
    run_round74_segmented_terminal_evaluation,
)


def _emit(payload: dict[str, object]) -> None:
    print(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


def _resolve(repository: Path, value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else repository / selected


def register_round74_segmented_terminal_commands(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001 - argparse has no public type
) -> None:
    evaluate = subparsers.add_parser(
        "binance-round74-sealed-evaluate",
        help="consume the Round 74 sealed test exactly once",
        description=(
            "Reserve the terminal test before reading target artifacts, run the "
            "frozen ML and two-model AI family, perform exact delayed L2 replay, "
            "and persist a recoverable result. This command grants no trading "
            "authority and cannot be retried after access is reserved."
        ),
    )
    evaluate.add_argument("--repository", default=".", help="repository root")
    evaluate.add_argument(
        "--database",
        action="append",
        required=True,
        help=(
            "completed Round 74 event database; repeat for every shard that may "
            "contain an admitted sealed-test run"
        ),
    )
    evaluate.add_argument(
        "--plan",
        default=(
            "docs/model-research/action-value/"
            "round-074-segmented-event-cohort-plan-v3.json"
        ),
        help="frozen segmented campaign plan",
    )
    evaluate.add_argument(
        "--state-root",
        default="data/round74-segmented-event-cohort-v3-state",
        help="terminal segmented campaign state",
    )
    evaluate.add_argument(
        "--recovery-outcomes",
        default="data/round74-segmented-event-cohort-v3-recovery",
        help="immutable missing-slot recovery evidence",
    )
    evaluate.add_argument(
        "--test-target-assemblies",
        required=True,
        help="directory containing only the admitted sealed-test manifests",
    )
    evaluate.add_argument(
        "--source-artifacts",
        required=True,
        help="root containing target-manifest source artifacts",
    )
    evaluate.add_argument(
        "--development-bundle",
        required=True,
        help="immutable Round 74 development-policy bundle",
    )
    evaluate.add_argument(
        "--pretest-policy",
        required=True,
        help="immutable pretest policy JSON beside its model and scaler",
    )
    evaluate.add_argument(
        "--ai-qualification",
        required=True,
        help="passing two-model pretest qualification JSON",
    )
    evaluate.add_argument(
        "--one-use-store",
        required=True,
        help="new durable pre-access and full-result SQLite store",
    )
    evaluate.add_argument(
        "--sealed-ledger",
        required=True,
        help="Round 74 sealed evaluation ledger",
    )
    evaluate.add_argument(
        "--output",
        required=True,
        help="new immutable terminal result JSON",
    )
    evaluate.add_argument(
        "--profile",
        choices=("conservative", "regular", "aggressive"),
        default="conservative",
        help="risk profile frozen by the development qualification",
    )
    evaluate.add_argument(
        "--compute-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
        help="required inference backend",
    )
    evaluate.add_argument(
        "--memory-limit",
        default="4GB",
        help="DuckDB memory ceiling",
    )
    evaluate.add_argument(
        "--database-threads",
        type=int,
        default=2,
        help="bounded DuckDB worker count",
    )
    evaluate.add_argument(
        "--inference-minibatch-rows",
        type=int,
        default=2048,
        help="bounded inference minibatch rows",
    )
    evaluate.add_argument(
        "--terminal-observed-wall-ns",
        type=int,
        default=None,
        help="optional fixed terminal observation time",
    )
    evaluate.add_argument(
        "--acknowledge-one-use-test-access",
        action="store_true",
        required=True,
        help="confirm that reservation permanently consumes sealed-test access",
    )
    evaluate.set_defaults(func=command_binance_round74_sealed_evaluate)

    recover = subparsers.add_parser(
        "binance-round74-recover-sealed",
        help="recover the completed Round 74 result without rerunning",
        description=(
            "Export the complete validated result from its one-use store. This "
            "command does not reopen the event database, targets, or AI models."
        ),
    )
    recover.add_argument(
        "--one-use-store",
        required=True,
        help="completed Round 74 terminal one-use store",
    )
    recover.add_argument(
        "--output",
        required=True,
        help="new immutable recovered result JSON",
    )
    recover.set_defaults(func=command_binance_round74_recover_sealed)


def command_binance_round74_sealed_evaluate(args: argparse.Namespace) -> int:
    repository = Path(args.repository).resolve()

    if getattr(args, "acknowledge_one_use_test_access", False) is not True:
        print(
            "binance-round74-sealed-evaluate failed: one-use acknowledgement "
            "is required",
            file=sys.stderr,
        )
        return 2

    def progress(stage: str, **values: object) -> None:
        _emit({"stage": stage, **values})

    try:
        result = run_round74_segmented_terminal_evaluation(
            repository=repository,
            database_paths=tuple(
                _resolve(repository, value) for value in args.database
            ),
            plan_path=_resolve(repository, args.plan),
            state_root=_resolve(repository, args.state_root),
            recovery_outcome_directory=_resolve(
                repository,
                args.recovery_outcomes,
            ),
            test_target_assembly_directory=_resolve(
                repository,
                args.test_target_assemblies,
            ),
            source_artifact_root=_resolve(repository, args.source_artifacts),
            development_bundle_path=_resolve(
                repository,
                args.development_bundle,
            ),
            pretest_policy_path=_resolve(repository, args.pretest_policy),
            ai_qualification_path=_resolve(repository, args.ai_qualification),
            one_use_store_path=_resolve(repository, args.one_use_store),
            sealed_ledger_path=_resolve(repository, args.sealed_ledger),
            output_path=_resolve(repository, args.output),
            terminal_observed_wall_ns=(
                time.time_ns()
                if args.terminal_observed_wall_ns is None
                else args.terminal_observed_wall_ns
            ),
            progress=progress,
            profile=args.profile,
            compute_backend=args.compute_backend,
            memory_limit=args.memory_limit,
            database_threads=args.database_threads,
            inference_minibatch_rows=args.inference_minibatch_rows,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"binance-round74-sealed-evaluate failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


def command_binance_round74_recover_sealed(args: argparse.Namespace) -> int:
    try:
        result = recover_round74_segmented_terminal_result(
            one_use_store_path=Path(args.one_use_store),
            output_path=Path(args.output),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"binance-round74-recover-sealed failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


__all__ = [
    "command_binance_round74_recover_sealed",
    "command_binance_round74_sealed_evaluate",
    "register_round74_segmented_terminal_commands",
]
