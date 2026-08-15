#!/usr/bin/env python3
"""Advance the immutable Round 27 runtime source ledger deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.storage import write_json_atomic


_PREDECESSOR = Path(
    "docs/model-research/polymarket/round-027-effective-source-ledger-v6.json"
)
_ADDED_FILES = (
    "src/simple_ai_trading/polymarket_round27_campaign_admission.py",
    "tools/admit_polymarket_round27_campaign.py",
)
_SCHEMA_VERSION = "polymarket-round27-effective-source-ledger-v7"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-ms", type=int, required=True)
    return parser


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
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 source ledger contains duplicate keys")
        output[key] = value
    return output


def _load(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {raw}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("Round 27 source ledger must be an object")
    return value


def build_ledger(repository: Path, *, created_at_ms: int) -> dict[str, object]:
    predecessor = _load(repository / _PREDECESSOR)
    predecessor_sha256 = str(predecessor.pop("source_ledger_sha256", ""))
    if predecessor_sha256 != _canonical_sha256(predecessor):
        raise ValueError("Round 27 predecessor source ledger hash differs")
    raw_files = predecessor.get("files_sha256")
    raw_scope = predecessor.get("scope")
    if not isinstance(raw_files, Mapping) or not isinstance(raw_scope, Mapping):
        raise ValueError("Round 27 predecessor source ledger differs")
    relative_files = sorted({*(str(path) for path in raw_files), *_ADDED_FILES})
    files: dict[str, str] = {}
    for relative in relative_files:
        path = (repository / relative).resolve(strict=True)
        if repository not in path.parents or not path.is_file():
            raise ValueError("Round 27 source ledger path escapes the repository")
        files[relative] = _file_sha256(path)
    entrypoints = raw_scope.get("entrypoint_files")
    if not isinstance(entrypoints, list):
        raise ValueError("Round 27 predecessor entrypoints differ")
    scope = dict(raw_scope)
    scope["entrypoint_files"] = sorted({*(str(path) for path in entrypoints), *_ADDED_FILES})
    scope["locked_file_count"] = len(files)
    body = {
        **predecessor,
        "created_at_ms": created_at_ms,
        "files_sha256": files,
        "predecessor_source_ledger_sha256": predecessor_sha256,
        "schema_version": _SCHEMA_VERSION,
        "scope": scope,
    }
    body["source_ledger_sha256"] = _canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve(strict=True)
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else repository / arguments.output
    ).resolve()
    if repository not in output.parents or output.exists() or output.is_symlink():
        raise ValueError("Round 27 source ledger output must be new and repository-local")
    ledger = build_ledger(repository, created_at_ms=arguments.created_at_ms)
    write_json_atomic(output, ledger, indent=2, sort_keys=True)
    if _load(output) != ledger:
        raise ValueError("Round 27 persisted source ledger differs")
    print(ledger["source_ledger_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
