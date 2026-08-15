#!/usr/bin/env python3
"""Freeze target-free Round 27 sealed AI cases for the nominated candidate."""

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
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
    load_round27_ai_ablation_contract,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    materialize_round27_ai_cases,
    round27_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round27_ai_economics import (
    round27_ai_candidate_selection_from_mapping,
)
from simple_ai_trading.polymarket_round27_ai_inference import (
    round27_ai_inference_report_from_mapping,
    run_round27_ai_inference,
)
from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES,
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
from tools.run_polymarket_round27_ai_cases import _progress


_RESULT_SCHEMA_VERSION = "polymarket-round27-ai-sealed-case-result-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--sealed-case-panel", type=Path, required=True)
    parser.add_argument("--sealed-inference-report", type=Path, required=True)
    parser.add_argument("--sealed-case-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _sealed_interval(contract: Mapping[str, object]) -> Mapping[str, object]:
    partitions = contract.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("Round 27 sealed AI partitions differ")
    matches = tuple(
        item
        for item in partitions
        if isinstance(item, Mapping) and item.get("role") == "sealed"
    )
    if len(matches) != 1 or matches[0].get("slot_id") != "stage1-c":
        raise ValueError("Round 27 sealed AI interval differs")
    return matches[0]


def _result(
    *,
    selection_sha256: str,
    status: str,
    model_id: str | None,
    panel_sha256: str | None,
    inference_report_sha256: str | None,
) -> dict[str, object]:
    if status not in {"no_candidate_nominated", "sealed_inference_frozen"}:
        raise ValueError("Round 27 sealed AI case status differs")
    present = status == "sealed_inference_frozen"
    if present != all(
        value is not None
        for value in (model_id, panel_sha256, inference_report_sha256)
    ):
        raise ValueError("Round 27 sealed AI case result binding differs")
    body: dict[str, object] = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "ablation_contract_sha256": (
            POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
        ),
        "ai_selection_sha256": selection_sha256,
        "status": status,
        "nominated_model_id": model_id,
        "sealed_case_panel_sha256": panel_sha256,
        "sealed_inference_report_sha256": inference_report_sha256,
        "target_accessed": False,
        "outcome_accessed": False,
        "future_books_accessed": False,
        "pnl_accessed": False,
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
        "feature": _resolve(repository, arguments.feature_store).resolve(strict=True),
        "source": _resolve(
            repository,
            arguments.sealed_source_database,
        ).resolve(strict=True),
        "selection": _resolve(
            repository,
            arguments.selection_claim,
        ).resolve(strict=True),
        "ai_selection": _resolve(
            repository,
            arguments.ai_selection_claim,
        ).resolve(strict=True),
    }
    outputs = {
        "panel": _resolve(repository, arguments.sealed_case_panel).resolve(),
        "inference": _resolve(
            repository,
            arguments.sealed_inference_report,
        ).resolve(),
        "result": _resolve(repository, arguments.sealed_case_result).resolve(),
    }
    if (
        any(path.is_symlink() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("feature", "source"))
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 sealed target-free AI paths differ")
    contract = load_round27_model_contract(repository)
    load_round27_ai_ablation_contract(repository)
    load_round27_economic_amendment(repository)
    ai_selection = round27_ai_candidate_selection_from_mapping(
        load_mapping(inputs["ai_selection"])
    )
    if ai_selection.nominated_model_id is None:
        if outputs["panel"].exists() or outputs["inference"].exists():
            raise ValueError("Round 27 sealed AI artifacts exist without nomination")
        result = _result(
            selection_sha256=ai_selection.selection_sha256,
            status="no_candidate_nominated",
            model_id=None,
            panel_sha256=None,
            inference_report_sha256=None,
        )
        artifact_writer(outputs["result"], "result_sha256")(result)
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
        return 0
    candidate = next(
        item
        for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
        if item.model_id == ai_selection.nominated_model_id
    )
    interval = _sealed_interval(contract)
    selection_claim = load_mapping(inputs["selection"])
    selected_model = load_round27_selected_model(
        selection_claim=selection_claim,
        contract=contract,
    )
    model_name, model_sha256 = model_identity(
        selected_model,
        contract_sha256=str(contract["contract_sha256"]),
    )
    config = Round27EconomicConfig(
        minimum_executed_trades=POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES
    ).validated()
    with Round27FeatureStore(inputs["feature"], read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        rows = tuple(
            row
            for row in feature_store.load_rows(slot_id="stage1-c")
            if int(interval["start_ms"])
            <= row.event_start_ms
            < int(interval["end_ms"])
        )
    if not rows:
        raise ValueError("Round 27 target-free sealed AI rows are empty")
    run_ids = {row.run_id for row in rows}
    condition_ids = tuple(sorted({row.condition_id for row in rows}))
    if len(run_ids) != 1:
        raise ValueError("Round 27 target-free sealed AI source run differs")
    run_id = next(iter(run_ids))
    if outputs["panel"].exists():
        panel = round27_ai_case_panel_from_mapping(load_mapping(outputs["panel"]))
        if (
            panel.partition_role != "sealed"
            or panel.source_run_id != run_id
            or panel.model_name != model_name
            or panel.model_sha256 != model_sha256
            or panel.source_audit_sha256 != feature_audit["audit_sha256"]
            or panel.evaluated_condition_ids_sha256
            != canonical_sha256(list(condition_ids))
        ):
            raise ValueError("Round 27 persisted sealed AI panel differs")
    else:
        with PolymarketEvidenceStore(
            inputs["source"],
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
                role="sealed",
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
    if outputs["inference"].exists():
        inference = round27_ai_inference_report_from_mapping(
            load_mapping(outputs["inference"])
        )
        if (
            inference.case_panel_sha256 != panel.panel_sha256
            or inference.candidate.get("model_id") != candidate.model_id
        ):
            raise ValueError("Round 27 persisted sealed AI inference differs")
    else:
        inference = run_round27_ai_inference(
            panel=panel,
            candidate=candidate,
            progress=_progress(candidate.model_id),
        )
    artifact_writer(outputs["inference"], "report_sha256")(inference.asdict())
    result = _result(
        selection_sha256=ai_selection.selection_sha256,
        status="sealed_inference_frozen",
        model_id=candidate.model_id,
        panel_sha256=panel.panel_sha256,
        inference_report_sha256=inference.report_sha256,
    )
    artifact_writer(outputs["result"], "result_sha256")(result)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
