#!/usr/bin/env python3
"""Evaluate the nominated Round 28 AI veto on one sealed population."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round28_ai_cases import (
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_economics import (
    validate_round28_ai_economic_report,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    validate_round28_ai_host_report,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    round28_ai_inference_report_from_mapping,
    validate_round28_ai_inference_report,
)
from simple_ai_trading.polymarket_round28_ai_sealed import (
    build_round28_ai_sealed_terminal_result,
    evaluate_round28_ai_sealed_economics,
)
from simple_ai_trading.polymarket_round28_ai_selection import (
    round28_ai_candidate_selection_from_mapping,
)
from simple_ai_trading.polymarket_round28_sealed import (
    validate_round28_sealed_economic_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--round27-target-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--nominated-host-report", type=Path, required=True)
    parser.add_argument("--sealed-case-panel", type=Path, required=True)
    parser.add_argument("--sealed-inference-report", type=Path, required=True)
    parser.add_argument("--sealed-case-result", type=Path, required=True)
    parser.add_argument("--sealed-round28-economic-report", type=Path, required=True)
    parser.add_argument("--sealed-ai-economic-report", type=Path, required=True)
    parser.add_argument("--terminal-ai-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 sealed AI path must not be a symlink")
    return selected.resolve(strict=strict)


def _sealed_role(audit: Mapping[str, object]) -> Mapping[str, object]:
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Round 28 sealed AI target roles differ")
    matches = tuple(
        item
        for item in roles
        if isinstance(item, Mapping) and item.get("role") == "sealed"
    )
    if len(matches) != 1 or matches[0].get("finalized") is not True:
        raise ValueError("Round 28 sealed AI target evidence is not terminal")
    return matches[0]


def _validate_case_result(
    value: Mapping[str, object],
    *,
    selection_sha256: str,
    model_id: str,
    panel_sha256: str,
    inference_sha256: str,
) -> dict[str, object]:
    result = dict(value)
    claimed = str(result.pop("result_sha256", ""))
    if (
        claimed != canonical_sha256(result)
        or result.get("schema_version")
        != "polymarket-round28-ai-sealed-case-result-v1"
        or result.get("ai_selection_sha256") != selection_sha256
        or result.get("status") != "sealed_inference_frozen"
        or result.get("nominated_model_id") != model_id
        or result.get("sealed_case_panel_sha256") != panel_sha256
        or result.get("sealed_inference_report_sha256") != inference_sha256
        or any(
            result.get(field) is not False
            for field in (
                "target_accessed",
                "outcome_accessed",
                "future_books_accessed",
                "pnl_accessed",
                "credentials_used",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 sealed AI case result differs")
    return {**result, "result_sha256": claimed}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "target": _resolve(repository, args.round27_target_store, strict=True),
        "source": _resolve(repository, args.sealed_source_database, strict=True),
        "selection": _resolve(repository, args.ai_selection_claim, strict=True),
        "host": _resolve(repository, args.nominated_host_report, strict=True),
        "panel": _resolve(repository, args.sealed_case_panel, strict=True),
        "inference": _resolve(
            repository,
            args.sealed_inference_report,
            strict=True,
        ),
        "case_result": _resolve(repository, args.sealed_case_result, strict=True),
        "baseline": _resolve(
            repository,
            args.sealed_round28_economic_report,
            strict=True,
        ),
    }
    outputs = {
        "economics": _resolve(
            repository,
            args.sealed_ai_economic_report,
            strict=False,
        ),
        "terminal": _resolve(repository, args.terminal_ai_result, strict=False),
    }
    all_paths = (*inputs.values(), *outputs.values())
    if (
        any(not path.is_file() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("target", "source"))
        or len(set(all_paths)) != len(all_paths)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 28 sealed AI evaluation paths differ")

    ai_contract = load_round28_ai_contract(repository)
    selection = round28_ai_candidate_selection_from_mapping(
        load_mapping(inputs["selection"])
    )
    if (
        selection.nominated_model_id is None
        or selection.nominated_runtime_digest is None
    ):
        raise ValueError("Round 28 sealed AI evaluation has no nomination")
    host_report, candidate = validate_round28_ai_host_report(
        load_mapping(inputs["host"]),
        contract=ai_contract,
    )
    panel = round28_ai_case_panel_from_mapping(load_mapping(inputs["panel"]))
    inference = round28_ai_inference_report_from_mapping(
        load_mapping(inputs["inference"])
    )
    validate_round28_ai_inference_report(
        inference.asdict(),
        contract=ai_contract,
        host_qualification_report=host_report,
        panel=panel,
    )
    if (
        panel.partition_role != "sealed"
        or candidate.model_id != selection.nominated_model_id
        or candidate.runtime_digest != selection.nominated_runtime_digest
        or inference.candidate.get("model_id") != selection.nominated_model_id
        or inference.candidate.get("runtime_digest")
        != selection.nominated_runtime_digest
    ):
        raise ValueError("Round 28 sealed AI nominated artifacts differ")
    _validate_case_result(
        load_mapping(inputs["case_result"]),
        selection_sha256=selection.selection_sha256,
        model_id=candidate.model_id,
        panel_sha256=panel.panel_sha256,
        inference_sha256=inference.report_sha256,
    )
    baseline = validate_round28_sealed_economic_report(
        load_mapping(inputs["baseline"])
    )

    if outputs["terminal"].exists():
        if not outputs["economics"].is_file():
            raise ValueError("Round 28 sealed AI terminal economics are missing")
        ai_report = validate_round28_ai_economic_report(
            load_mapping(outputs["economics"])
        )
        terminal = build_round28_ai_sealed_terminal_result(
            ai_selection=selection,
            panel=panel,
            inference_report=inference.asdict(),
            sealed_round28_economic_report=baseline,
            sealed_ai_economic_report=ai_report,
        )
        if load_mapping(outputs["terminal"]) != terminal:
            raise ValueError("Round 28 persisted sealed AI terminal differs")
        print(json.dumps(terminal, allow_nan=False, indent=2, sort_keys=True))
        return 0

    with Round27TargetStore(inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        target_role = _sealed_role(target_audit)
        if (
            baseline["resolution_evidence_sha256"]
            != target_role["evidence_chain_sha256"]
        ):
            raise ValueError("Round 28 sealed AI resolution lineage differs")
        outcomes = target_store.outcomes_up(roles=("sealed",))
    condition_ids = tuple(sorted(outcomes))
    if (
        len(condition_ids) != panel.evaluated_condition_count
        or canonical_sha256(list(condition_ids))
        != panel.evaluated_condition_ids_sha256
    ):
        raise ValueError("Round 28 sealed AI outcome population differs")

    if outputs["economics"].exists():
        ai_report = validate_round28_ai_economic_report(
            load_mapping(outputs["economics"])
        )
        if (
            ai_report.get("case_panel_sha256") != panel.panel_sha256
            or ai_report.get("inference_report_sha256")
            != inference.report_sha256
            or ai_report.get("round28_economic_report_sha256")
            != baseline["report_sha256"]
            or ai_report.get("resolution_evidence_sha256")
            != target_role["evidence_chain_sha256"]
        ):
            raise ValueError("Round 28 persisted sealed AI economics differ")
    else:
        with PolymarketEvidenceStore(
            inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=panel.source_run_id,
                condition_ids=condition_ids,
            )
            ai_report = evaluate_round28_ai_sealed_economics(
                panel=panel,
                inference_report=inference.asdict(),
                contract=ai_contract,
                host_qualification_report=host_report,
                sealed_round28_economic_report=baseline,
                markets=markets,
                outcomes_up=outcomes,
                resolution_evidence_sha256=str(
                    target_role["evidence_chain_sha256"]
                ),
                book_batches=economic_book_batches(
                    source,
                    run_id=panel.source_run_id,
                    condition_ids=condition_ids,
                    maximum_conditions=int(
                        panel.economic_config[
                            "maximum_conditions_per_book_batch"
                        ]
                    ),
                ),
            )
        artifact_writer(outputs["economics"], "report_sha256")(ai_report)
    terminal = build_round28_ai_sealed_terminal_result(
        ai_selection=selection,
        panel=panel,
        inference_report=inference.asdict(),
        sealed_round28_economic_report=baseline,
        sealed_ai_economic_report=ai_report,
    )
    artifact_writer(outputs["terminal"], "result_sha256")(terminal)
    print(json.dumps(terminal, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
