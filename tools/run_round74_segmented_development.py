"""Train the complete segmented Round 74 development panel once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.compute import (  # noqa: E402
    BackendUnavailableError,
    SUPPORTED_COMPUTE_BACKENDS,
)
from simple_ai_trading.round74_segmented_development_runtime import (  # noqa: E402
    ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_RELATIVE_PATH,
    ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION,
    run_round74_segmented_development,
)


def _emit(stage: str, **values: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--database", type=Path)
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
        "--recovery-outcomes",
        type=Path,
        default=Path("data/round74-segmented-event-cohort-v3-recovery"),
    )
    parser.add_argument("--target-assemblies", type=Path, required=True)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--qualification-output", type=Path, required=True)
    parser.add_argument(
        "--compute-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
    )
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--database-threads", type=int, default=2)
    parser.add_argument("--inference-minibatch-rows", type=int, default=128)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    parser.add_argument("--disable-ai", action="store_true")
    return parser


def _resolve(repository: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository / path


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    database = (
        repository / ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_RELATIVE_PATH
        if arguments.database is None
        else _resolve(repository, arguments.database)
    )
    try:
        result = run_round74_segmented_development(
            repository=repository,
            database_path=database,
            plan_path=_resolve(repository, arguments.plan),
            state_root=_resolve(repository, arguments.state_root),
            recovery_outcome_directory=_resolve(
                repository,
                arguments.recovery_outcomes,
            ),
            target_assembly_directory=_resolve(
                repository,
                arguments.target_assemblies,
            ),
            source_artifact_root=_resolve(
                repository,
                arguments.source_artifacts,
            ),
            model_output_directory=_resolve(
                repository,
                arguments.model_output,
            ),
            qualification_output_directory=_resolve(
                repository,
                arguments.qualification_output,
            ),
            terminal_observed_wall_ns=time.time_ns(),
            progress=_emit,
            compute_backend=arguments.compute_backend,
            memory_limit=arguments.memory_limit,
            database_threads=arguments.database_threads,
            inference_minibatch_rows=arguments.inference_minibatch_rows,
            progress_interval_seconds=arguments.progress_interval_seconds,
            enable_ai=not arguments.disable_ai,
        )
    except (
        BackendUnavailableError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": (
                        ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION
                    ),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "sealed_test_accessed": False,
                    "trading_authority": False,
                    "profitability_claim": False,
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
