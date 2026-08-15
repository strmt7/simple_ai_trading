#!/usr/bin/env python3
"""Run the target-free Round 27 mechanics diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.polymarket_round27_mechanics import (
    analyze_round27_mechanics,
)
from simple_ai_trading.storage import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--condition-audit", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--capture-contract", type=Path, required=True)
    parser.add_argument("--capture-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.repository.resolve()
    progress_path = (
        None
        if arguments.progress is None
        else _resolve(root, arguments.progress)
    )

    def progress(phase: str, payload: Mapping[str, object]) -> None:
        message = {"phase": phase, **payload}
        if progress_path is not None:
            write_json_atomic(progress_path, message, indent=2, sort_keys=True)
        print(json.dumps(message, sort_keys=True), flush=True)

    result = analyze_round27_mechanics(
        root,
        database_path=_resolve(root, arguments.database),
        condition_audit_path=_resolve(root, arguments.condition_audit),
        preregistration_path=_resolve(root, arguments.preregistration),
        capture_contract_path=_resolve(root, arguments.capture_contract),
        capture_result_path=_resolve(root, arguments.capture_result),
        output_path=_resolve(root, arguments.output),
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
