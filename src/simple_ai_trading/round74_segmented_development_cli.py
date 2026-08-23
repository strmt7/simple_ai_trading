"""Installed CLI and Windows workflow for Round 74 model development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from .compute import SUPPORTED_COMPUTE_BACKENDS


def run_round74_segmented_development(**kwargs: Any) -> dict[str, object]:
    """Load the optional Torch runtime only when development actually starts."""

    from .round74_segmented_development_runtime import (  # noqa: PLC0415
        run_round74_segmented_development as run,
    )

    return run(**kwargs)


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


def register_round74_segmented_development_command(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001 - argparse has no public type
) -> None:
    develop = subparsers.add_parser(
        "binance-round74-develop",
        help="train and qualify the complete Round 74 development population",
        description=(
            "Route every admitted training and tuning run across completed event "
            "database shards before reading target manifests, train the frozen ML "
            "family, and optionally qualify the local-AI veto. This development-only "
            "command never accesses the sealed test or grants trading authority."
        ),
    )
    develop.add_argument("--repository", default=".", help="repository root")
    develop.add_argument(
        "--database",
        action="append",
        required=True,
        help=(
            "completed Round 74 event database; repeat for every shard that may "
            "contain an admitted training or tuning run"
        ),
    )
    develop.add_argument(
        "--plan",
        default=(
            "docs/model-research/action-value/"
            "round-074-segmented-event-cohort-plan-v3.json"
        ),
        help="frozen segmented campaign plan",
    )
    develop.add_argument(
        "--state-root",
        default="data/round74-segmented-event-cohort-v3-state",
        help="terminal segmented campaign state",
    )
    develop.add_argument(
        "--recovery-outcomes",
        default="data/round74-segmented-event-cohort-v3-recovery",
        help="immutable missing-slot recovery evidence",
    )
    develop.add_argument(
        "--target-assemblies",
        required=True,
        help="directory containing only admitted development target manifests",
    )
    develop.add_argument(
        "--source-artifacts",
        required=True,
        help="root containing target-manifest source artifacts",
    )
    develop.add_argument(
        "--model-output",
        required=True,
        help="new directory for immutable model and development-policy artifacts",
    )
    develop.add_argument(
        "--qualification-output",
        required=True,
        help="new directory for profile-specific local-AI qualification artifacts",
    )
    develop.add_argument(
        "--compute-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
        help="required training and inference backend",
    )
    develop.add_argument(
        "--memory-limit",
        default="4GB",
        help="per-open DuckDB memory ceiling",
    )
    develop.add_argument(
        "--database-threads",
        type=int,
        default=2,
        help="bounded workers for the one currently open shard",
    )
    develop.add_argument(
        "--inference-minibatch-rows",
        type=int,
        default=128,
        help="bounded inference minibatch rows",
    )
    develop.add_argument(
        "--supervised-device-group-policy",
        choices=("auto", "fixed"),
        default="auto",
        help=(
            "select per-candidate supervised groups with a target-free host "
            "probe, or use one fixed group"
        ),
    )
    develop.add_argument(
        "--supervised-device-run-group-size",
        type=int,
        default=8,
        help="fixed supervised device group size from 1 through 32",
    )
    develop.add_argument(
        "--device-group-preflight-timeout-seconds",
        type=float,
        default=300.0,
        help="hard timeout for the isolated target-free host probe",
    )
    develop.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="heartbeat interval during long model work",
    )
    develop.add_argument(
        "--terminal-observed-wall-ns",
        type=int,
        default=None,
        help="optional fixed terminal observation time",
    )
    develop.add_argument(
        "--disable-ai",
        action="store_true",
        help="train ML only and skip local-AI qualification",
    )
    develop.set_defaults(func=command_binance_round74_develop)


def command_binance_round74_develop(args: argparse.Namespace) -> int:
    repository = Path(args.repository).resolve()

    def progress(stage: str, **values: object) -> None:
        _emit({"stage": stage, **values})

    try:
        result = run_round74_segmented_development(
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
            target_assembly_directory=_resolve(
                repository,
                args.target_assemblies,
            ),
            source_artifact_root=_resolve(repository, args.source_artifacts),
            model_output_directory=_resolve(repository, args.model_output),
            qualification_output_directory=_resolve(
                repository,
                args.qualification_output,
            ),
            terminal_observed_wall_ns=(
                time.time_ns()
                if args.terminal_observed_wall_ns is None
                else args.terminal_observed_wall_ns
            ),
            progress=progress,
            compute_backend=args.compute_backend,
            memory_limit=args.memory_limit,
            database_threads=args.database_threads,
            inference_minibatch_rows=args.inference_minibatch_rows,
            supervised_device_group_policy=(args.supervised_device_group_policy),
            supervised_device_run_group_size=(args.supervised_device_run_group_size),
            device_group_preflight_timeout_seconds=(
                args.device_group_preflight_timeout_seconds
            ),
            progress_interval_seconds=args.progress_interval_seconds,
            enable_ai=not args.disable_ai,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"binance-round74-develop failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


__all__ = [
    "command_binance_round74_develop",
    "register_round74_segmented_development_command",
]
