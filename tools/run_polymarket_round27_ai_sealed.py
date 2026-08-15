#!/usr/bin/env python3
"""Evaluate the one-use Round 27 sealed AI candidate after receipts are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_ai_ablation_contract import (
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
    load_round27_ai_ablation_contract,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    round27_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round27_ai_economics import (
    evaluate_round27_ai_matched_economics,
    round27_ai_candidate_selection_from_mapping,
    validate_round27_ai_economic_report,
)
from simple_ai_trading.polymarket_round27_ai_inference import (
    round27_ai_inference_report_from_mapping,
)
from simple_ai_trading.polymarket_round27_economic_amendment import (
    load_round27_economic_amendment,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
    source_recomputed_artifact,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore


_TERMINAL_SCHEMA_VERSION = "polymarket-round27-ai-terminal-sealed-result-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--sealed-case-panel", type=Path, required=True)
    parser.add_argument("--sealed-inference-report", type=Path, required=True)
    parser.add_argument("--baseline-sealed-economic-report", type=Path, required=True)
    parser.add_argument("--sealed-ai-economic-report", type=Path, required=True)
    parser.add_argument("--terminal-ai-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _target_role(audit: Mapping[str, object]) -> Mapping[str, object]:
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Round 27 sealed AI target roles differ")
    matches = tuple(
        item
        for item in roles
        if isinstance(item, Mapping) and item.get("role") == "sealed"
    )
    if len(matches) != 1:
        raise ValueError("Round 27 sealed AI target role differs")
    return matches[0]


def _terminal_result(
    *,
    ai_selection_sha256: str,
    nominated_model_id: str,
    panel_sha256: str,
    inference_report_sha256: str,
    baseline_report_sha256: str,
    ai_report: Mapping[str, object],
) -> dict[str, object]:
    passed = ai_report.get("matched_after_cost_uplift_gate_passed") is True
    body: dict[str, object] = {
        "schema_version": _TERMINAL_SCHEMA_VERSION,
        "ablation_contract_sha256": (
            POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        ),
        "ai_selection_sha256": ai_selection_sha256,
        "nominated_model_id": nominated_model_id,
        "sealed_case_panel_sha256": panel_sha256,
        "sealed_inference_report_sha256": inference_report_sha256,
        "baseline_sealed_economic_report_sha256": baseline_report_sha256,
        "sealed_ai_economic_report_sha256": ai_report["report_sha256"],
        "sealed_matched_after_cost_uplift_gate_passed": passed,
        "model_prompt_or_threshold_changed_after_selection": False,
        "observed_after_cost_ai_uplift": passed,
        "edge_claim": False,
        "profitability_claim": False,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["result_sha256"] = canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    inputs = {
        "target": _resolve(repository, arguments.target_store).resolve(strict=True),
        "source": _resolve(
            repository,
            arguments.sealed_source_database,
        ).resolve(strict=True),
        "selection": _resolve(
            repository,
            arguments.ai_selection_claim,
        ).resolve(strict=True),
        "panel": _resolve(
            repository,
            arguments.sealed_case_panel,
        ).resolve(strict=True),
        "inference": _resolve(
            repository,
            arguments.sealed_inference_report,
        ).resolve(strict=True),
        "baseline": _resolve(
            repository,
            arguments.baseline_sealed_economic_report,
        ).resolve(strict=True),
    }
    outputs = {
        "economics": _resolve(
            repository,
            arguments.sealed_ai_economic_report,
        ).resolve(),
        "terminal": _resolve(repository, arguments.terminal_ai_result).resolve(),
    }
    if (
        any(path.is_symlink() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("target", "source"))
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 sealed AI evaluation paths differ")
    load_round27_model_contract(repository)
    load_round27_ai_ablation_contract(repository)
    load_round27_economic_amendment(repository)
    selection = round27_ai_candidate_selection_from_mapping(
        load_mapping(inputs["selection"])
    )
    if selection.nominated_model_id is None:
        raise ValueError("Round 27 sealed AI evaluation has no nominated candidate")
    panel = round27_ai_case_panel_from_mapping(load_mapping(inputs["panel"]))
    inference = round27_ai_inference_report_from_mapping(
        load_mapping(inputs["inference"])
    )
    if (
        panel.partition_role != "sealed"
        or inference.case_panel_sha256 != panel.panel_sha256
        or inference.candidate.get("model_id") != selection.nominated_model_id
        or inference.candidate.get("runtime_digest")
        != selection.nominated_runtime_digest
    ):
        raise ValueError("Round 27 sealed AI artifact binding differs")
    baseline = load_mapping(inputs["baseline"])
    with Round27TargetStore(inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        target_role = _target_role(target_audit)
        outcomes = target_store.outcomes_up(roles=("sealed",))
    condition_ids = tuple(sorted(outcomes))
    if (
        len(condition_ids) != panel.evaluated_condition_count
        or canonical_sha256(list(condition_ids))
        != panel.evaluated_condition_ids_sha256
    ):
        raise ValueError("Round 27 sealed AI outcome population differs")
    resolution_evidence_sha256 = str(target_role["evidence_chain_sha256"])
    with PolymarketEvidenceStore(
        inputs["source"],
        read_only=True,
        memory_limit=arguments.memory_limit,
        threads=2,
    ) as source:
        markets = PolymarketEvidenceReplay.load_markets(
            source,
            run_id=panel.source_run_id,
            condition_ids=condition_ids,
        )
        report = source_recomputed_artifact(
            outputs["economics"],
            "report_sha256",
            lambda: validate_round27_ai_economic_report(
                evaluate_round27_ai_matched_economics(
                    panel=panel,
                    inference_report=inference,
                    baseline_economic_report=baseline,
                    markets=markets,
                    outcomes_up=outcomes,
                    resolution_evidence_sha256=resolution_evidence_sha256,
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
                ),
            ),
        )
    terminal = _terminal_result(
        ai_selection_sha256=selection.selection_sha256,
        nominated_model_id=selection.nominated_model_id,
        panel_sha256=panel.panel_sha256,
        inference_report_sha256=inference.report_sha256,
        baseline_report_sha256=str(baseline["report_sha256"]),
        ai_report=report,
    )
    artifact_writer(outputs["terminal"], "result_sha256")(terminal)
    print(json.dumps(terminal, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
