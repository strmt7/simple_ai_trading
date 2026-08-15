#!/usr/bin/env python3
"""Qualify one Round 28 AI candidate on the actual 278-feature prompt shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    load_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_prompt_envelope import (
    evaluate_round28_ai_prompt_envelope,
    validate_round28_ai_prompt_envelope_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--host-report", type=Path, required=True)
    parser.add_argument("--prompt-envelope-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 AI prompt-envelope path must not be a symlink")
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    host_path = _resolve(repository, args.host_report, strict=True)
    output = _resolve(
        repository,
        args.prompt_envelope_report,
        strict=False,
    )
    if (
        not host_path.is_file()
        or host_path == output
        or (output.exists() and output.is_symlink())
    ):
        raise ValueError("Round 28 AI prompt-envelope paths differ")
    contract = load_round28_ai_contract(repository)
    host_report = load_mapping(host_path)
    if output.exists():
        report = validate_round28_ai_prompt_envelope_report(
            load_mapping(output),
            contract=contract,
            host_qualification_report=host_report,
        )
    else:
        report = evaluate_round28_ai_prompt_envelope(
            contract=contract,
            host_qualification_report=host_report,
        )
        artifact_writer(output, "report_sha256")(report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
