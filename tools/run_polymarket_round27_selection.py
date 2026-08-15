#!/usr/bin/env python3
"""Run Round 27 development selection and after-cost selection replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round27_experiment import (
    build_round27_selection_economic_claim,
    load_round27_selected_model,
    run_round27_development_selection,
)
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_model import (
    Round27Partition,
    build_round27_model_samples,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer as _writer,
    canonical_sha256 as _canonical_sha256,
    economic_book_batches as _batches,
    load_mapping as _load_mapping,
    model_identity as _model_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-economic-report", type=Path, required=True)
    parser.add_argument("--selection-economic-claim", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    feature_path = _resolve(repository, arguments.feature_store).resolve(strict=True)
    target_path = _resolve(repository, arguments.target_store).resolve(strict=True)
    source_path = _resolve(
        repository,
        arguments.selection_source_database,
    ).resolve(strict=True)
    outputs = {
        "selection": _resolve(repository, arguments.selection_claim).resolve(),
        "economic_report": _resolve(
            repository,
            arguments.selection_economic_report,
        ).resolve(),
        "economic_claim": _resolve(
            repository,
            arguments.selection_economic_claim,
        ).resolve(),
    }
    if (
        any(path.is_symlink() for path in (feature_path, target_path, source_path))
        or any(
            Path(f"{path}.wal").exists()
            for path in (feature_path, target_path, source_path)
        )
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 selection requires terminal unique inputs and outputs")
    contract = load_round27_model_contract(repository)
    partitions = contract.get("partitions")
    economics = contract.get("economic_evaluation")
    if not isinstance(partitions, list) or not isinstance(economics, Mapping):
        raise ValueError("Round 27 selection contract differs")
    with Round27FeatureStore(feature_path, read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        rows_by_slot = {
            slot_id: feature_store.load_rows(slot_id=slot_id)
            for slot_id in ("stage1-a", "stage1-b")
        }
    with Round27TargetStore(target_path, read_only=True) as target_store:
        target_audit = target_store.audit()
        outcomes = target_store.outcomes_up(
            roles=("train", "calibration", "selection")
        )
    samples = build_round27_model_samples(
        rows_by_slot=rows_by_slot,
        outcomes_up=outcomes,
        role_intervals=[item for item in partitions if isinstance(item, Mapping)],
    )
    if outputs["selection"].exists():
        selection_claim = _load_mapping(outputs["selection"])
        selected_model = load_round27_selected_model(
            selection_claim=selection_claim,
            contract=contract,
        )
    else:
        selection_claim, selected_model = run_round27_development_selection(
            samples=samples,
            contract=contract,
            claim_writer=_writer(outputs["selection"], "claim_sha256"),
            compute_backend=arguments.compute_backend,
        )
    selection = Round27Partition.from_samples(samples, role="selection")
    prior = 1.0 / (1.0 + np.exp(-selection.offsets))
    probabilities = (
        prior
        if selected_model is None
        else selected_model.predict(selection.features, selection.offsets)
    )
    selection_conditions = tuple(sorted(set(str(item) for item in selection.conditions)))
    selection_outcomes = {
        condition_id: outcomes[condition_id]
        for condition_id in selection_conditions
    }
    selection_run_ids = {row.run_id for row in rows_by_slot["stage1-b"]}
    if len(selection_run_ids) != 1:
        raise ValueError("Round 27 selection source run differs")
    selection_run_id = next(iter(selection_run_ids))
    target_roles = {
        item["role"]: item
        for item in target_audit["roles"]
        if isinstance(item, Mapping)
    }
    selection_target = target_roles.get("selection")
    if not isinstance(selection_target, Mapping):
        raise ValueError("Round 27 selection target evidence differs")
    model_name, model_sha256 = _model_identity(
        selected_model,
        contract_sha256=str(contract["contract_sha256"]),
    )
    resolution_evidence_sha256 = str(selection_target["evidence_chain_sha256"])
    probability_input_sha256 = _canonical_sha256(
        {
            "feature_row_sha256": [
                sample.feature_row_sha256 for sample in selection.samples
            ],
            "probabilities": [format(float(value), ".17g") for value in probabilities],
        }
    )
    economic_config = Round27EconomicConfig()
    if outputs["economic_report"].exists():
        economic_report = _load_mapping(outputs["economic_report"])
        if (
            economic_report.get("model_name") != model_name
            or economic_report.get("model_sha256") != model_sha256
            or economic_report.get("source_audit_sha256")
            != feature_audit["audit_sha256"]
            or economic_report.get("resolution_evidence_sha256")
            != resolution_evidence_sha256
            or economic_report.get("probability_input_sha256")
            != probability_input_sha256
            or economic_report.get("config") != economic_config.asdict()
            or any(
                economic_report.get(field) is not False
                for field in (
                    "edge_claim",
                    "profitability_claim",
                    "orders_submitted",
                    "trading_authority",
                )
            )
        ):
            raise ValueError("Round 27 persisted selection economics differ")
    else:
        with PolymarketEvidenceStore(
            source_path,
            read_only=True,
            memory_limit=arguments.memory_limit,
            threads=2,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=selection_run_id,
                condition_ids=selection_conditions,
            )
            economic_report = evaluate_round27_economic_scenarios(
                partition=selection,
                predictions=probabilities,
                markets=markets,
                outcomes_up=selection_outcomes,
                model_name=model_name,
                model_sha256=model_sha256,
                source_audit_sha256=str(feature_audit["audit_sha256"]),
                resolution_evidence_sha256=resolution_evidence_sha256,
                config=economic_config,
                book_batches=_batches(
                    source,
                    run_id=selection_run_id,
                    condition_ids=selection_conditions,
                    maximum_conditions=int(
                        economics["maximum_conditions_per_book_batch"]
                    ),
                ),
            )
    _writer(outputs["economic_report"], "report_sha256")(economic_report)
    economic_claim = build_round27_selection_economic_claim(
        contract=contract,
        selection_claim=selection_claim,
        selected_model=selected_model,
        economic_report=economic_report,
        claim_writer=_writer(outputs["economic_claim"], "claim_sha256"),
    )
    print(
        json.dumps(
            {
                "selection_claim_sha256": selection_claim["claim_sha256"],
                "selected_model_name": selection_claim["selected_model_name"],
                "selection_economic_report_sha256": economic_report["report_sha256"],
                "selection_economic_claim_sha256": economic_claim["claim_sha256"],
                "selection_economic_gate_passed": economic_claim[
                    "selection_economic_gate_passed"
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
