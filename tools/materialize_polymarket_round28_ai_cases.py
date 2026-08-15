#!/usr/bin/env python3
"""Materialize the target-free Round 28 AI selection case panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES,
)
from simple_ai_trading.polymarket_round27_economics import Round27EconomicConfig
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
)
from simple_ai_trading.polymarket_round28_ai_cases import (
    materialize_round28_ai_cases,
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES,
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    compose_round28_feature_rows,
    load_round28_book_ticker_overlay,
)
from simple_ai_trading.polymarket_round28_operator import (
    validate_round28_selection_input_manifest,
)
from simple_ai_trading.polymarket_round28_selection import (
    load_round28_selected_pair,
)


_PREREGISTRATION = Path(
    "docs/model-research/polymarket/"
    "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--round27-feature-store", type=Path, required=True)
    parser.add_argument("--round28-overlay-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 AI case path must not be a symlink")
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "feature": _resolve(repository, args.round27_feature_store, strict=True),
        "overlay": _resolve(repository, args.round28_overlay_store, strict=True),
        "source": _resolve(
            repository,
            args.selection_source_database,
            strict=True,
        ),
        "manifest": _resolve(
            repository,
            args.selection_input_manifest,
            strict=True,
        ),
        "selection": _resolve(repository, args.selection_claim, strict=True),
    }
    output = _resolve(repository, args.case_panel, strict=False)
    if (
        any(not path.is_file() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("feature", "overlay", "source"))
        or output in inputs.values()
        or (output.exists() and output.is_symlink())
    ):
        raise ValueError("Round 28 target-free AI case paths differ")

    contract = load_round27_model_contract(repository)
    load_round28_ai_contract(repository)
    preregistration = load_mapping(repository / _PREREGISTRATION)
    manifest = validate_round28_selection_input_manifest(
        load_mapping(inputs["manifest"])
    )
    selection_claim = load_mapping(inputs["selection"])
    selected_pair = load_round28_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
    )
    if selected_pair is None:
        raise ValueError("Round 28 AI cases require a selected augmented model")

    with Round27FeatureStore(inputs["feature"], read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        all_base_rows = tuple(
            sorted(feature_store.load_rows(), key=lambda row: row.decision_time_ms)
        )
        selection_base_rows = feature_store.load_rows(slot_id="stage1-b")
    overlay_rows, overlay_report = load_round28_book_ticker_overlay(inputs["overlay"])
    if (
        feature_audit["audit_sha256"]
        != manifest["round27_feature_store_audit_sha256"]
        or overlay_report["report_sha256"]
        != manifest["round28_overlay_report_sha256"]
        or contract["contract_sha256"]
        != manifest["round27_model_contract_sha256"]
        or preregistration["preregistration_sha256"]
        != manifest["round28_preregistration_sha256"]
    ):
        raise ValueError("Round 28 AI case source lineage differs")
    composed = compose_round28_feature_rows(
        base_rows=all_base_rows,
        overlay_rows=overlay_rows,
        report=overlay_report,
    )
    selection_decisions = {row.decision_time_ms for row in selection_base_rows}
    rows = tuple(row for row in composed if row.decision_time_ms in selection_decisions)
    if not rows:
        raise ValueError("Round 28 target-free AI selection rows are empty")
    run_ids = {row.run_id for row in rows}
    condition_ids = tuple(sorted({row.condition_id for row in rows}))
    selection_role = next(
        item for item in manifest["roles"] if item["role"] == "selection"
    )
    if (
        len(run_ids) != 1
        or len(condition_ids) != int(selection_role["condition_count"])
    ):
        raise ValueError("Round 28 AI selection population differs")
    run_id = next(iter(run_ids))
    config = Round27EconomicConfig(
        minimum_executed_trades=(
            POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES
        )
    ).validated()

    if output.exists():
        panel = round28_ai_case_panel_from_mapping(load_mapping(output))
        if (
            panel.partition_role != "selection"
            or panel.source_run_id != run_id
            or panel.selection_claim_sha256 != selection_claim["claim_sha256"]
            or panel.source_audit_sha256 != manifest["manifest_sha256"]
            or panel.economic_config != config.asdict()
            or panel.evaluated_condition_count != len(condition_ids)
            or panel.evaluated_condition_ids_sha256
            != canonical_sha256(list(condition_ids))
        ):
            raise ValueError("Round 28 persisted AI case panel source differs")
    else:
        with PolymarketEvidenceStore(
            inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=run_id,
                condition_ids=condition_ids,
            )
            panel = materialize_round28_ai_cases(
                role="selection",
                rows=rows,
                selected_model=selected_pair.augmented_model,
                selection_claim_sha256=str(selection_claim["claim_sha256"]),
                markets=markets,
                source_audit_sha256=str(manifest["manifest_sha256"]),
                config=config,
                book_batches=economic_book_batches(
                    source,
                    run_id=run_id,
                    condition_ids=condition_ids,
                    maximum_conditions=config.maximum_conditions_per_book_batch,
                ),
            )
    if len(panel.cases) < POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES:
        raise ValueError("Round 28 AI case population is below its frozen minimum")
    artifact_writer(output, "panel_sha256")(panel.asdict())
    print(
        json.dumps(
            {
                "case_panel_sha256": panel.panel_sha256,
                "case_count": len(panel.cases),
                "evaluated_condition_count": panel.evaluated_condition_count,
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
