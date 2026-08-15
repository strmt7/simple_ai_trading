#!/usr/bin/env python3
"""Join frozen Round 27 AI receipts to selection outcomes and nominate at most one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_ai_ablation_contract import (
    load_round27_ai_ablation_contract,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    round27_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round27_ai_economics import (
    evaluate_round27_ai_matched_economics,
    select_round27_ai_candidate,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--qwen-inference-report", type=Path, required=True)
    parser.add_argument("--oda-inference-report", type=Path, required=True)
    parser.add_argument("--baseline-economic-report", type=Path, required=True)
    parser.add_argument("--qwen-economic-report", type=Path, required=True)
    parser.add_argument("--oda-economic-report", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _target_role(audit: Mapping[str, object]) -> Mapping[str, object]:
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Round 27 AI selection target roles differ")
    matches = tuple(
        item
        for item in roles
        if isinstance(item, Mapping) and item.get("role") == "selection"
    )
    if len(matches) != 1:
        raise ValueError("Round 27 AI selection target role differs")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    inputs = {
        "target": _resolve(repository, arguments.target_store).resolve(strict=True),
        "source": _resolve(
            repository,
            arguments.selection_source_database,
        ).resolve(strict=True),
        "panel": _resolve(repository, arguments.case_panel).resolve(strict=True),
        "qwen_inference": _resolve(
            repository,
            arguments.qwen_inference_report,
        ).resolve(strict=True),
        "oda_inference": _resolve(
            repository,
            arguments.oda_inference_report,
        ).resolve(strict=True),
        "baseline": _resolve(
            repository,
            arguments.baseline_economic_report,
        ).resolve(strict=True),
    }
    outputs = {
        "qwen": _resolve(repository, arguments.qwen_economic_report).resolve(),
        "oda": _resolve(repository, arguments.oda_economic_report).resolve(),
        "selection": _resolve(repository, arguments.ai_selection_claim).resolve(),
    }
    if (
        any(path.is_symlink() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("target", "source"))
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 AI selection paths differ")
    load_round27_model_contract(repository)
    load_round27_ai_ablation_contract(repository)
    load_round27_economic_amendment(repository)
    panel = round27_ai_case_panel_from_mapping(load_mapping(inputs["panel"]))
    if panel.partition_role != "selection":
        raise ValueError("Round 27 AI selection panel role differs")
    inference_reports = {
        key: round27_ai_inference_report_from_mapping(load_mapping(inputs[key]))
        for key in ("qwen_inference", "oda_inference")
    }
    if any(
        report.case_panel_sha256 != panel.panel_sha256
        for report in inference_reports.values()
    ):
        raise ValueError("Round 27 AI selection inference binding differs")
    baseline = load_mapping(inputs["baseline"])
    with Round27TargetStore(inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        target_role = _target_role(target_audit)
        outcomes = target_store.outcomes_up(roles=("selection",))
    condition_ids = tuple(sorted(outcomes))
    if (
        len(condition_ids) != panel.evaluated_condition_count
        or canonical_sha256(list(condition_ids))
        != panel.evaluated_condition_ids_sha256
    ):
        raise ValueError("Round 27 AI selection outcome population differs")
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
        reports = {}
        for output_key, inference_key in (
            ("qwen", "qwen_inference"),
            ("oda", "oda_inference"),
        ):
            report = source_recomputed_artifact(
                outputs[output_key],
                "report_sha256",
                lambda inference_report=inference_reports[inference_key]: (
                    validate_round27_ai_economic_report(
                        evaluate_round27_ai_matched_economics(
                            panel=panel,
                            inference_report=inference_report,
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
                        )
                    ),
                ),
            )
            reports[output_key] = report
    selection = select_round27_ai_candidate((reports["qwen"], reports["oda"]))
    artifact_writer(outputs["selection"], "selection_sha256")(selection.asdict())
    print(
        json.dumps(
            {
                "qwen_matched_after_cost_uplift_gate_passed": reports["qwen"][
                    "matched_after_cost_uplift_gate_passed"
                ],
                "oda_matched_after_cost_uplift_gate_passed": reports["oda"][
                    "matched_after_cost_uplift_gate_passed"
                ],
                "nominated_model_id": selection.nominated_model_id,
                "ai_selection_sha256": selection.selection_sha256,
                "sealed_partition_accessed": False,
                "edge_claim": False,
                "profitability_claim": False,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
