#!/usr/bin/env python3
"""Run source-bound Round 29 selection and matched after-cost economics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time

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
    economic_book_batches,
    load_mapping,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round28_book_ticker import (
    compose_round28_feature_rows,
    load_round28_book_ticker_overlay,
)
from simple_ai_trading.polymarket_round29_economics import (
    evaluate_round29_matched_economics,
)
from simple_ai_trading.polymarket_round29_model import (
    Round29Partition,
    build_round29_model_samples,
)
from simple_ai_trading.polymarket_round29_operator import (
    build_round29_selection_input_manifest,
    validate_round29_economic_report,
    verify_round29_implementation_sources,
)
from simple_ai_trading.polymarket_round29_selection import (
    load_round29_selected_pair,
    run_round29_matched_selection,
)
from simple_ai_trading.polymarket_round29_source import (
    compose_round29_feature_pairs,
    validate_round29_source_report,
)


_PREREGISTRATION = Path(
    "docs/model-research/polymarket/"
    "round-029-settlement-state-matched-ablation-preregistration-v1.json"
)
_IMPLEMENTATION_AMENDMENT = Path(
    "docs/model-research/polymarket/"
    "round-029-model-economic-operator-implementation-amendment-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--round27-feature-store", type=Path, required=True)
    parser.add_argument("--round28-overlay-store", type=Path, required=True)
    parser.add_argument("--round27-target-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-economic-report", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    return selected.resolve(strict=strict)


def _validate_paths(
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> None:
    all_paths = tuple(inputs.values()) + tuple(outputs.values())
    existing_paths = tuple(path for path in all_paths if path.exists())
    aliases_existing_file = any(
        left.samefile(right)
        for index, left in enumerate(existing_paths)
        for right in existing_paths[index + 1 :]
    )
    if (
        any(path.is_symlink() or not path.is_file() for path in inputs.values())
        or any(Path(f"{path}.wal").exists() for path in inputs.values())
        or len(set(all_paths)) != len(all_paths)
        or aliases_existing_file
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 29 selection paths differ")


def _progress(phase: str, detail: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "observed_at_ms": time.time_ns() // 1_000_000,
                "phase": phase,
                **detail,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "feature": _resolve(repository, args.round27_feature_store, strict=True),
        "overlay": _resolve(repository, args.round28_overlay_store, strict=True),
        "target": _resolve(repository, args.round27_target_store, strict=True),
        "source": _resolve(
            repository,
            args.selection_source_database,
            strict=True,
        ),
    }
    outputs = {
        "manifest": _resolve(repository, args.selection_input_manifest, strict=False),
        "selection": _resolve(repository, args.selection_claim, strict=False),
        "economics": _resolve(
            repository,
            args.selection_economic_report,
            strict=False,
        ),
    }
    _validate_paths(inputs, outputs)

    contract = load_round27_model_contract(repository)
    preregistration = load_mapping(repository / _PREREGISTRATION)
    amendment = load_mapping(repository / _IMPLEMENTATION_AMENDMENT)
    verify_round29_implementation_sources(amendment, root=repository)
    partitions = contract.get("partitions")
    economics_contract = contract.get("economic_evaluation")
    if not isinstance(partitions, list) or not isinstance(
        economics_contract,
        Mapping,
    ):
        raise ValueError("Round 29 inherited model contract differs")

    with Round27FeatureStore(inputs["feature"], read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        all_base_rows = tuple(
            sorted(feature_store.load_rows(), key=lambda row: row.decision_time_ms)
        )
        base_rows_by_slot = {
            slot_id: feature_store.load_rows(slot_id=slot_id)
            for slot_id in ("stage1-a", "stage1-b")
        }
    overlay_rows, bbo_overlay_report = load_round28_book_ticker_overlay(
        inputs["overlay"]
    )
    round28_rows = compose_round28_feature_rows(
        base_rows=all_base_rows,
        overlay_rows=overlay_rows,
        report=bbo_overlay_report,
    )
    pairs, settlement_overlay_report = compose_round29_feature_pairs(
        base_rows=all_base_rows,
        round28_rows=round28_rows,
        round28_overlay_report=bbo_overlay_report,
    )
    validate_round29_source_report(settlement_overlay_report)
    pairs_by_decision = {pair[0].decision_time_ms: pair for pair in pairs}
    if len(pairs_by_decision) != len(pairs):
        raise ValueError("Round 29 composed decision population differs")
    rows_by_slot = {
        slot_id: tuple(
            pairs_by_decision[row.decision_time_ms]
            for row in base_rows
            if row.decision_time_ms in pairs_by_decision
        )
        for slot_id, base_rows in base_rows_by_slot.items()
    }

    with Round27TargetStore(inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        outcomes = target_store.outcomes_up(roles=("train", "calibration", "selection"))
    samples = build_round29_model_samples(
        rows_by_slot=rows_by_slot,
        outcomes_up=outcomes,
        role_intervals=[item for item in partitions if isinstance(item, Mapping)],
    )
    input_manifest = build_round29_selection_input_manifest(
        samples=samples,
        feature_store_audit=feature_audit,
        bbo_overlay_report=bbo_overlay_report,
        settlement_overlay_report=settlement_overlay_report,
        target_store_audit=target_audit,
        contract=contract,
        preregistration=preregistration,
        implementation_amendment=amendment,
    )
    artifact_writer(outputs["manifest"], "manifest_sha256")(input_manifest)
    manifest_sha256 = str(input_manifest["manifest_sha256"])
    _progress(
        "selection-input-frozen",
        {
            "manifest_sha256": manifest_sha256,
            "role_counts": input_manifest["roles"],
            "sealed_partition_accessed": False,
        },
    )

    if outputs["selection"].exists():
        selection_claim = load_mapping(outputs["selection"])
        selected_pair = load_round29_selected_pair(
            selection_claim,
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest_sha256=manifest_sha256,
        )
    else:
        selection_claim, selected_pair = run_round29_matched_selection(
            samples=samples,
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest_sha256=manifest_sha256,
            claim_writer=artifact_writer(outputs["selection"], "claim_sha256"),
            progress=_progress,
        )
    artifact_writer(outputs["selection"], "claim_sha256")(selection_claim)
    if selected_pair is None:
        if outputs["economics"].exists():
            raise ValueError("Round 29 rejected selection has an economic report")
        print(
            json.dumps(
                {
                    "status": selection_claim["status"],
                    "selection_input_manifest_sha256": manifest_sha256,
                    "selection_claim_sha256": selection_claim["claim_sha256"],
                    "economic_report_created": False,
                    "economic_uplift_gate_passed": False,
                    "sealed_partition_accessed": False,
                    "orders_submitted": False,
                    "trading_authority": False,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    selection = Round29Partition.from_samples(samples, role="selection")
    selection_conditions = tuple(
        condition_id
        for condition_id, _start in sorted(
            {
                sample.condition_id: sample.event_start_ms
                for sample in selection.samples
            }.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    selection_outcomes = {
        condition_id: outcomes[condition_id] for condition_id in selection_conditions
    }
    selection_run_ids = {pair[0].run_id for pair in rows_by_slot.get("stage1-b", ())}
    if len(selection_run_ids) != 1:
        raise ValueError("Round 29 selection source run differs")
    selection_run_id = next(iter(selection_run_ids))
    raw_role_audits = target_audit.get("roles")
    if not isinstance(raw_role_audits, list):
        raise ValueError("Round 29 target role audits differ")
    role_audits = {
        item["role"]: item
        for item in raw_role_audits
        if isinstance(item, Mapping) and isinstance(item.get("role"), str)
    }
    selection_target_audit = role_audits.get("selection")
    if (
        not isinstance(selection_target_audit, Mapping)
        or selection_target_audit.get("finalized") is not True
    ):
        raise ValueError("Round 29 selection target evidence differs")
    resolution_evidence_sha256 = str(selection_target_audit["evidence_chain_sha256"])
    economic_config = Round27EconomicConfig(
        minimum_executed_trades=(POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES)
    ).validated()
    if outputs["economics"].exists():
        economic_report = validate_round29_economic_report(
            load_mapping(outputs["economics"]),
            input_manifest=input_manifest,
            selection_claim=selection_claim,
            contract=contract,
            preregistration=preregistration,
            resolution_evidence_sha256=resolution_evidence_sha256,
        )
    else:
        with PolymarketEvidenceStore(
            inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source_store:
            markets = PolymarketEvidenceReplay.load_markets(
                source_store,
                run_id=selection_run_id,
                condition_ids=selection_conditions,
            )

            def batches():
                return economic_book_batches(
                    source_store,
                    run_id=selection_run_id,
                    condition_ids=selection_conditions,
                    maximum_conditions=int(
                        economics_contract["maximum_conditions_per_book_batch"]
                    ),
                )

            economic_report = evaluate_round29_matched_economics(
                samples=samples,
                selection_claim=selection_claim,
                contract=contract,
                preregistration=preregistration,
                implementation_amendment_sha256=str(amendment["amendment_sha256"]),
                markets=markets,
                outcomes_up=selection_outcomes,
                source_audit_sha256=manifest_sha256,
                resolution_evidence_sha256=resolution_evidence_sha256,
                book_batch_factory=batches,
                config=economic_config,
            )
        validate_round29_economic_report(
            economic_report,
            input_manifest=input_manifest,
            selection_claim=selection_claim,
            contract=contract,
            preregistration=preregistration,
            resolution_evidence_sha256=resolution_evidence_sha256,
        )
        artifact_writer(outputs["economics"], "report_sha256")(economic_report)
    print(
        json.dumps(
            {
                "selection_input_manifest_sha256": manifest_sha256,
                "selection_claim_sha256": selection_claim["claim_sha256"],
                "selected_model_family": selection_claim["selected_model_family"],
                "selection_economic_report_sha256": economic_report["report_sha256"],
                "economic_uplift_gate_passed": economic_report[
                    "economic_uplift_gate_passed"
                ],
                "sealed_partition_accessed": False,
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
