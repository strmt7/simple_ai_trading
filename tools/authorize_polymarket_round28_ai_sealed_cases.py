#!/usr/bin/env python3
"""Authorize sealed AI cases and emit a metrics-free access receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    load_mapping,
)
from simple_ai_trading.polymarket_round28_ai_sealed_access import (
    build_round28_ai_sealed_access_receipt,
    validate_round28_ai_sealed_access_receipt,
)


_PREREGISTRATION = Path(
    "docs/model-research/polymarket/"
    "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-economic-report", type=Path, required=True)
    parser.add_argument("--selection-ai-case-panel", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--access-receipt", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 sealed AI authorization path is a symlink")
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "manifest": _resolve(
            repository,
            args.selection_input_manifest,
            strict=True,
        ),
        "selection": _resolve(
            repository,
            args.selection_claim,
            strict=True,
        ),
        "economics": _resolve(
            repository,
            args.selection_economic_report,
            strict=True,
        ),
        "selection_ai_panel": _resolve(
            repository,
            args.selection_ai_case_panel,
            strict=True,
        ),
        "ai_selection": _resolve(
            repository,
            args.ai_selection_claim,
            strict=True,
        ),
    }
    output = _resolve(repository, args.access_receipt, strict=False)
    if (
        any(not path.is_file() for path in inputs.values())
        or output in inputs.values()
        or (output.exists() and output.is_symlink())
    ):
        raise ValueError("Round 28 sealed AI authorization paths differ")
    contract = load_round27_model_contract(repository)
    preregistration = load_mapping(repository / _PREREGISTRATION)
    manifest = load_mapping(inputs["manifest"])
    selection_claim = load_mapping(inputs["selection"])
    economics = load_mapping(inputs["economics"])
    selection_ai_panel = load_mapping(inputs["selection_ai_panel"])
    ai_selection = load_mapping(inputs["ai_selection"])
    validation = {
        "contract": contract,
        "preregistration": preregistration,
        "selection_input_manifest": manifest,
        "selection_claim": selection_claim,
        "selection_ai_case_panel": selection_ai_panel,
        "ai_selection_claim": ai_selection,
    }
    expected = build_round28_ai_sealed_access_receipt(
        selection_economic_report=economics,
        selection_resolution_evidence_sha256=str(
            economics.get("resolution_evidence_sha256", "")
        ),
        **validation,
    )
    if output.exists():
        receipt, _pair, _panel, _selection = (
            validate_round28_ai_sealed_access_receipt(
                load_mapping(output),
                **validation,
            )
        )
        if receipt != expected:
            raise ValueError("Round 28 persisted sealed AI access receipt differs")
    else:
        receipt = expected
        artifact_writer(output, "receipt_sha256")(receipt)
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
