"""Build one immutable no-order public-mainnet execution scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    load_round74_segmented_cohort_binding,
)
from simple_ai_trading.impact_absorption_execution_scenario import (
    build_round74_public_execution_scenario,
    collect_round74_public_transport_source,
    load_round74_execution_aggregate_source,
)
from simple_ai_trading.impact_absorption_store import (
    ImpactAbsorptionStore,
    validate_impact_store_resources,
)
from simple_ai_trading.round74_segmented_development_runtime import (
    _guard_idle_database,
)


_MAXIMUM_BINDING_BYTES = 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _load_binding(path: Path) -> object:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAXIMUM_BINDING_BYTES
    ):
        raise ValueError("Round 74 public execution binding file differs")
    selected = path.resolve()
    try:
        raw_text = selected.read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Round 74 public execution binding bytes differ") from exc
    return load_round74_segmented_cohort_binding(raw_text)


def _write_new_artifact(
    *,
    output_directory: Path,
    run_id: str,
    payload: Mapping[str, object],
) -> Path:
    if output_directory.is_symlink():
        raise ValueError("Round 74 public execution output directory differs")
    output_directory.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["artifact_sha256"] = _canonical_sha256(value)
    encoded = (_canonical_json(value) + "\n").encode("ascii")
    target = output_directory / f"{run_id}.json"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise FileExistsError("Round 74 public execution scenario already differs")
        return target
    temporary = target.with_suffix(".json.tmp")
    if temporary.exists():
        raise FileExistsError("Round 74 public execution temporary artifact exists")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    temporary.replace(target)
    return target


def build_round74_public_execution_scenario_artifact(
    *,
    database_path: Path,
    binding_path: Path,
    execution_aggregate_path: Path,
    output_directory: Path,
    memory_limit: str,
    threads: int,
) -> Path:
    """Audit sources, derive the scenario, and write immutable bytes."""

    database = database_path.resolve()
    if (
        database_path.is_symlink()
        or not database.is_file()
        or database.stat().st_size <= 0
    ):
        raise ValueError("Round 74 public execution database path differs")
    binding = _load_binding(binding_path)
    aggregate = load_round74_execution_aggregate_source(execution_aggregate_path)
    _guard_idle_database(database, repeated=False)
    resources = validate_impact_store_resources(
        memory_limit=memory_limit,
        threads=threads,
    )
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=resources.memory_limit,
        threads=resources.threads,
    ) as store:
        transport = collect_round74_public_transport_source(
            store,
            binding=binding,
        )
    _guard_idle_database(database, repeated=True)
    scenario = build_round74_public_execution_scenario(
        transport_source=transport,
        execution_aggregate=aggregate,
    )
    return _write_new_artifact(
        output_directory=output_directory,
        run_id=scenario.run_id,
        payload=scenario.as_dict(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one immutable Round 74 public-mainnet execution scenario "
            "without credentials or orders."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument(
        "--execution-aggregate",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        target = build_round74_public_execution_scenario_artifact(
            database_path=args.database,
            binding_path=args.binding,
            execution_aggregate_path=args.execution_aggregate,
            output_directory=args.output_directory,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = json.loads(target.read_text(encoding="ascii"))
    print(
        _canonical_json(
            {
                "artifact": str(target),
                "artifact_sha256": payload["artifact_sha256"],
                "environment": payload["environment"],
                "run_id": payload["run_id"],
                "orders_submitted": False,
                "mainnet_fill_evidence": False,
                "trading_authority": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
