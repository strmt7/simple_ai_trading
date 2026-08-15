from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_round21_sidecar_terminal import (  # noqa: E402
    load_round21_sidecar_terminal_manifest,
)
from simple_ai_trading.polymarket_round24_receipt_lead_lag import (  # noqa: E402
    assemble_round24_receipt_rows,
    run_round24_receipt_lead_lag,
)
from simple_ai_trading.polymarket_round25_terminal import (  # noqa: E402
    load_round25_terminal_transport_manifest,
)


ACKNOWLEDGEMENT = "consume-fresh-round24-development-selection-once"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _publish_once(
    path: Path,
    value: Mapping[str, object],
    *,
    label: str,
) -> str:
    target = path.resolve()
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"Round 24 {label} path must be new")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Round 24 temporary {label} path is not clean")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    if target.exists():
        temporary.unlink()
        raise FileExistsError(f"Round 24 {label} appeared concurrently")
    os.replace(temporary, target)
    return str(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consume the fresh Round 24 receipt-time selection once."
    )
    parser.add_argument("--core-database", type=Path, required=True)
    parser.add_argument("--terminal-transport-manifest", type=Path, required=True)
    parser.add_argument("--sidecar-database", type=Path, required=True)
    parser.add_argument("--sidecar-terminal-manifest", type=Path, required=True)
    parser.add_argument("--materialization-evidence", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledgement", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.acknowledgement != ACKNOWLEDGEMENT:
        raise ValueError("Round 24 one-use selection acknowledgement differs")
    one_use_paths = (
        arguments.materialization_evidence,
        arguments.selection_claim,
        arguments.output,
    )
    if any(path.is_symlink() or path.exists() for path in one_use_paths):
        raise FileExistsError("Round 24 one-use evidence paths must all be new")
    terminal = load_round25_terminal_transport_manifest(
        arguments.terminal_transport_manifest
    )
    sidecar_terminal = load_round21_sidecar_terminal_manifest(
        arguments.sidecar_terminal_manifest
    )
    print(
        _canonical_json(
            {
                "event": "round24_receipt_holdout_start",
                "mode": "terminal_read_only_one_use_selection",
            }
        ),
        flush=True,
    )
    spec, rows, source_evidence = assemble_round24_receipt_rows(
        repository=ROOT,
        core_database=arguments.core_database,
        terminal_transport_manifest=terminal,
        sidecar_database=arguments.sidecar_database,
        sidecar_terminal_manifest=sidecar_terminal,
    )
    materialization_evidence = _publish_once(
        arguments.materialization_evidence,
        source_evidence,
        label="materialization evidence",
    )

    def claim_selection(claim: Mapping[str, object]) -> str:
        _publish_once(
            arguments.selection_claim,
            claim,
            label="selection claim",
        )
        return str(claim["claim_sha256"])

    result = run_round24_receipt_lead_lag(
        spec=spec,
        rows=rows,
        claim_selection=claim_selection,
    )
    output = _publish_once(arguments.output, result, label="result")
    print(
        _canonical_json(
            {
                "conclusion": result["conclusion"],
                "event": "round24_receipt_holdout_complete",
                "mechanism_gate_passed": result["mechanism_gate_passed"],
                "materialization_evidence": materialization_evidence,
                "output": output,
                "result_sha256": result["result_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
