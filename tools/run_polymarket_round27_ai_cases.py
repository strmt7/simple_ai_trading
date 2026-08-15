#!/usr/bin/env python3
"""Materialize target-free Round 27 AI cases and run both local candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_ai import (
    POLYMARKET_ROUND27_AI_HOST_CANDIDATES,
)
from simple_ai_trading.polymarket_round27_ai_ablation_contract import (
    load_round27_ai_ablation_contract,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    Round27AICasePanel,
    materialize_round27_ai_cases,
    round27_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round27_ai_inference import (
    Round27AIInferenceReport,
    round27_ai_inference_report_from_mapping,
    run_round27_ai_inference,
)
from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES,
    load_round27_economic_amendment,
)
from simple_ai_trading.polymarket_round27_economics import Round27EconomicConfig
from simple_ai_trading.polymarket_round27_experiment import (
    load_round27_selected_model,
)
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
    model_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--qwen-inference-report", type=Path, required=True)
    parser.add_argument("--oda-inference-report", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _selection_interval(contract: Mapping[str, object]) -> Mapping[str, object]:
    partitions = contract.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("Round 27 AI selection partitions differ")
    matches = tuple(
        item
        for item in partitions
        if isinstance(item, Mapping) and item.get("role") == "selection"
    )
    if len(matches) != 1 or matches[0].get("slot_id") != "stage1-b":
        raise ValueError("Round 27 AI selection interval differs")
    return matches[0]


def _load_existing_report(
    path: Path,
    *,
    panel: Round27AICasePanel,
    model_id: str,
) -> Round27AIInferenceReport:
    report = round27_ai_inference_report_from_mapping(load_mapping(path))
    if (
        report.case_panel_sha256 != panel.panel_sha256
        or report.candidate.get("model_id") != model_id
    ):
        raise ValueError("Round 27 persisted AI candidate report differs")
    return report


def _progress(model_id: str):
    def report(completed: int, total: int) -> None:
        if completed == total or completed % 5 == 0:
            print(
                json.dumps(
                    {
                        "phase": "target_free_ai_inference",
                        "model_id": model_id,
                        "completed_cases": completed,
                        "total_cases": total,
                        "targets_accessed": False,
                        "orders_submitted": False,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    feature_path = _resolve(repository, arguments.feature_store).resolve(strict=True)
    source_path = _resolve(
        repository,
        arguments.selection_source_database,
    ).resolve(strict=True)
    selection_path = _resolve(
        repository,
        arguments.selection_claim,
    ).resolve(strict=True)
    outputs = {
        "panel": _resolve(repository, arguments.case_panel).resolve(),
        "qwen": _resolve(repository, arguments.qwen_inference_report).resolve(),
        "oda": _resolve(repository, arguments.oda_inference_report).resolve(),
    }
    inputs = (feature_path, source_path, selection_path)
    if (
        any(path.is_symlink() for path in inputs)
        or any(Path(f"{path}.wal").exists() for path in inputs[:2])
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 target-free AI paths differ")
    contract = load_round27_model_contract(repository)
    load_round27_ai_ablation_contract(repository)
    load_round27_economic_amendment(repository)
    interval = _selection_interval(contract)
    selection_claim = load_mapping(selection_path)
    selected_model = load_round27_selected_model(
        selection_claim=selection_claim,
        contract=contract,
    )
    model_name, model_sha256 = model_identity(
        selected_model,
        contract_sha256=str(contract["contract_sha256"]),
    )
    config = Round27EconomicConfig(
        minimum_executed_trades=(
            POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES
        )
    ).validated()
    with Round27FeatureStore(feature_path, read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        rows = tuple(
            row
            for row in feature_store.load_rows(slot_id="stage1-b")
            if int(interval["start_ms"])
            <= row.event_start_ms
            < int(interval["end_ms"])
        )
    if not rows:
        raise ValueError("Round 27 target-free AI selection rows are empty")
    run_ids = {row.run_id for row in rows}
    condition_ids = tuple(sorted({row.condition_id for row in rows}))
    if len(run_ids) != 1:
        raise ValueError("Round 27 target-free AI source run differs")
    run_id = next(iter(run_ids))
    expected_condition_sha256 = canonical_sha256(list(condition_ids))
    if outputs["panel"].exists():
        panel = round27_ai_case_panel_from_mapping(load_mapping(outputs["panel"]))
        if (
            panel.model_name != model_name
            or panel.source_run_id != run_id
            or panel.model_sha256 != model_sha256
            or panel.source_audit_sha256 != feature_audit["audit_sha256"]
            or panel.economic_config != config.asdict()
            or panel.evaluated_condition_count != len(condition_ids)
            or panel.evaluated_condition_ids_sha256
            != expected_condition_sha256
        ):
            raise ValueError("Round 27 persisted AI case panel source differs")
    else:
        with PolymarketEvidenceStore(
            source_path,
            read_only=True,
            memory_limit=arguments.memory_limit,
            threads=2,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=run_id,
                condition_ids=condition_ids,
            )
            panel = materialize_round27_ai_cases(
                role="selection",
                rows=rows,
                selected_model=selected_model,
                model_name=model_name,
                model_sha256=model_sha256,
                markets=markets,
                source_audit_sha256=str(feature_audit["audit_sha256"]),
                config=config,
                book_batches=economic_book_batches(
                    source,
                    run_id=run_id,
                    condition_ids=condition_ids,
                    maximum_conditions=config.maximum_conditions_per_book_batch,
                ),
            )
    artifact_writer(outputs["panel"], "panel_sha256")(panel.asdict())
    reports: dict[str, Round27AIInferenceReport] = {}
    for key, candidate in zip(
        ("qwen", "oda"),
        POLYMARKET_ROUND27_AI_HOST_CANDIDATES,
        strict=True,
    ):
        report = (
            _load_existing_report(
                outputs[key],
                panel=panel,
                model_id=candidate.model_id,
            )
            if outputs[key].exists()
            else run_round27_ai_inference(
                panel=panel,
                candidate=candidate,
                progress=_progress(candidate.model_id),
            )
        )
        artifact_writer(outputs[key], "report_sha256")(report.asdict())
        reports[key] = report
    print(
        json.dumps(
            {
                "case_panel_sha256": panel.panel_sha256,
                "case_count": len(panel.cases),
                "qwen_inference_report_sha256": reports["qwen"].report_sha256,
                "qwen_matched_evaluation_eligible": reports[
                    "qwen"
                ].candidate_eligible_for_matched_evaluation,
                "oda_inference_report_sha256": reports["oda"].report_sha256,
                "oda_matched_evaluation_eligible": reports[
                    "oda"
                ].candidate_eligible_for_matched_evaluation,
                "target_accessed": False,
                "outcome_accessed": False,
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
