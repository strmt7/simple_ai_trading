#!/usr/bin/env python3
"""Run Round 28 AI inference only after an actual-shape prompt probe passes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_ai_trading.polymarket_round27_operator import load_mapping
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_prompt_envelope import (
    validate_round28_ai_prompt_envelope_report,
)
from tools.run_polymarket_round28_ai_inference import main as _base_main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--host-report", type=Path, required=True)
    parser.add_argument("--prompt-envelope-report", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 enveloped AI path must not be a symlink")
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    host_path = _resolve(repository, args.host_report, strict=True)
    envelope_path = _resolve(
        repository,
        args.prompt_envelope_report,
        strict=True,
    )
    if not host_path.is_file() or not envelope_path.is_file():
        raise ValueError("Round 28 enveloped AI evidence is missing")
    report = validate_round28_ai_prompt_envelope_report(
        load_mapping(envelope_path),
        contract=load_round28_ai_contract(repository),
        host_qualification_report=load_mapping(host_path),
    )
    if report["passed"] is not True:
        raise ValueError("Round 28 actual-shape prompt envelope did not pass")
    return _base_main(
        [
            "--repository",
            str(repository),
            "--case-panel",
            str(args.case_panel),
            "--host-report",
            str(args.host_report),
            "--inference-report",
            str(args.inference_report),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
