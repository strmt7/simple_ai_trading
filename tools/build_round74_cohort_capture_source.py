"""Build one immutable admitted-cohort source wrapper without opening DuckDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_event_segmented_cohort import (  # noqa: E402
    load_round74_segmented_cohort_binding,
)
from simple_ai_trading.round74_public_target_sources import (  # noqa: E402
    build_round74_cohort_capture_source_payload,
)
from simple_ai_trading.round74_segmented_development_runtime import (  # noqa: E402
    _guard_idle_database,
)
from tools._round74_public_evidence_capture import (  # noqa: E402
    require_clean_tracked_worktree,
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


def _load_binding(path: Path) -> object:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= _MAXIMUM_BINDING_BYTES
    ):
        raise ValueError("Round 74 cohort source binding file differs")
    try:
        raw = path.read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Round 74 cohort source binding bytes differ") from exc
    return load_round74_segmented_cohort_binding(raw)


def _write_immutable(
    *,
    output_directory: Path,
    run_id: str,
    payload: Mapping[str, object],
) -> Path:
    if output_directory.is_symlink():
        raise ValueError("Round 74 cohort source output directory differs")
    output_directory.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("ascii")
    target = output_directory.resolve() / f"{run_id}.json"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise FileExistsError("Round 74 cohort source artifact already differs")
        return target
    temporary = target.with_suffix(".json.tmp")
    if temporary.exists():
        raise FileExistsError("Round 74 cohort source temporary artifact exists")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    temporary.replace(target)
    return target


def build_round74_cohort_capture_source_artifact(
    *,
    binding_path: Path,
    database_path: Path,
    output_directory: Path,
) -> Path:
    """Validate identity and write a no-authority cohort source artifact."""

    binding = _load_binding(binding_path.resolve())
    database = database_path.resolve()
    if (
        database_path.is_symlink()
        or not database.is_file()
        or database.stat().st_size <= 0
    ):
        raise ValueError("Round 74 cohort source database differs")
    try:
        relative_database = database.relative_to(REPOSITORY).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Round 74 cohort source database must remain in the repository"
        ) from exc
    _guard_idle_database(database, repeated=False)
    payload = build_round74_cohort_capture_source_payload(
        binding=binding,
        database_relative_path=relative_database,
    )
    _guard_idle_database(database, repeated=True)
    return _write_immutable(
        output_directory=output_directory,
        run_id=binding.run_id,
        payload=payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap one admitted Round 74 binding for read-only funding and "
            "target assembly without opening the campaign database."
        )
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_clean_tracked_worktree()
        target = build_round74_cohort_capture_source_artifact(
            binding_path=args.binding,
            database_path=args.database,
            output_directory=args.output_directory,
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
                "run_id": payload["run_id"],
                "database_opened": False,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
