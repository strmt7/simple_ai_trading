#!/usr/bin/env python3
"""Run the one-use Round 27 sealed prediction and economic evaluation."""

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
    run_round27_sealed_evaluation,
    validate_round27_sealed_access_artifacts,
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
from tools.run_polymarket_round27_selection import (
    _batches,
    _canonical_sha256,
    _load_mapping,
    _model_identity,
    _writer,
)


_TERMINAL_SCHEMA_VERSION = "polymarket-round27-terminal-sealed-result-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-economic-report", type=Path, required=True)
    parser.add_argument("--selection-economic-claim", type=Path, required=True)
    parser.add_argument("--sealed-prediction-result", type=Path, required=True)
    parser.add_argument("--sealed-economic-report", type=Path, required=True)
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _terminal_result(
    *,
    contract_sha256: str,
    selection_claim: Mapping[str, object],
    selection_economic_claim: Mapping[str, object],
    selection_economic_report: Mapping[str, object],
    sealed_prediction: Mapping[str, object],
    sealed_economics: Mapping[str, object],
) -> dict[str, object]:
    prediction_passed = sealed_prediction.get("prediction_edge_gate_passed") is True
    economics_passed = sealed_economics.get("economic_edge_gate_passed") is True
    body: dict[str, object] = {
        "schema_version": _TERMINAL_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "selection_claim_sha256": selection_claim["claim_sha256"],
        "selection_economic_claim_sha256": selection_economic_claim["claim_sha256"],
        "selection_economic_report_sha256": selection_economic_report[
            "report_sha256"
        ],
        "sealed_prediction_result_sha256": sealed_prediction["result_sha256"],
        "sealed_economic_report_sha256": sealed_economics["report_sha256"],
        "prediction_edge_gate_passed": prediction_passed,
        "economic_edge_gate_passed": economics_passed,
        "observed_after_cost_edge_gate_passed": (
            prediction_passed and economics_passed
        ),
        "sealed_partition_accessed": True,
        "model_or_threshold_changed_after_selection": False,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    feature_path = _resolve(repository, arguments.feature_store).resolve(strict=True)
    target_path = _resolve(repository, arguments.target_store).resolve(strict=True)
    source_path = _resolve(
        repository,
        arguments.sealed_source_database,
    ).resolve(strict=True)
    input_artifacts = {
        "selection": _resolve(repository, arguments.selection_claim).resolve(
            strict=True
        ),
        "economic_claim": _resolve(
            repository,
            arguments.selection_economic_claim,
        ).resolve(strict=True),
        "economic_report": _resolve(
            repository,
            arguments.selection_economic_report,
        ).resolve(strict=True),
    }
    outputs = {
        "prediction": _resolve(
            repository,
            arguments.sealed_prediction_result,
        ).resolve(),
        "economics": _resolve(
            repository,
            arguments.sealed_economic_report,
        ).resolve(),
        "terminal": _resolve(repository, arguments.terminal_result).resolve(),
    }
    all_inputs = (feature_path, target_path, source_path, *input_artifacts.values())
    if (
        any(path.is_symlink() for path in all_inputs)
        or any(Path(f"{path}.wal").exists() for path in all_inputs[:3])
        or len(set(outputs.values())) != len(outputs)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 27 sealed evaluation paths differ")
    contract = load_round27_model_contract(repository)
    partitions = contract.get("partitions")
    economics = contract.get("economic_evaluation")
    if not isinstance(partitions, list) or not isinstance(economics, Mapping):
        raise ValueError("Round 27 sealed contract differs")
    selection_claim = _load_mapping(input_artifacts["selection"])
    selection_economic_claim = _load_mapping(input_artifacts["economic_claim"])
    selection_economic_report = _load_mapping(input_artifacts["economic_report"])
    selected_model = validate_round27_sealed_access_artifacts(
        contract=contract,
        selection_claim=selection_claim,
        selection_economic_claim=selection_economic_claim,
        selection_economic_report=selection_economic_report,
    )
    with Round27FeatureStore(feature_path, read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        rows_by_slot = {
            slot_id: feature_store.load_rows(slot_id=slot_id)
            for slot_id in ("stage1-a", "stage1-b", "stage1-c")
        }
    with Round27TargetStore(target_path, read_only=True) as target_store:
        target_audit = target_store.audit()
        outcomes = target_store.outcomes_up(
            roles=("train", "calibration", "selection", "sealed")
        )
    samples = build_round27_model_samples(
        rows_by_slot=rows_by_slot,
        outcomes_up=outcomes,
        role_intervals=[item for item in partitions if isinstance(item, Mapping)],
    )
    sealed_prediction = run_round27_sealed_evaluation(
        samples=samples,
        contract=contract,
        selection_claim=selection_claim,
        selection_economic_claim=selection_economic_claim,
        selection_economic_report=selection_economic_report,
        selected_model=selected_model,
    )
    _writer(outputs["prediction"], "result_sha256")(sealed_prediction)
    sealed = Round27Partition.from_samples(samples, role="sealed")
    prior = 1.0 / (1.0 + np.exp(-sealed.offsets))
    probabilities = (
        prior
        if selected_model is None
        else selected_model.predict(sealed.features, sealed.offsets)
    )
    sealed_conditions = tuple(sorted(set(str(item) for item in sealed.conditions)))
    sealed_outcomes = {
        condition_id: outcomes[condition_id] for condition_id in sealed_conditions
    }
    sealed_run_ids = {row.run_id for row in rows_by_slot["stage1-c"]}
    if len(sealed_run_ids) != 1:
        raise ValueError("Round 27 sealed source run differs")
    sealed_run_id = next(iter(sealed_run_ids))
    target_roles = {
        item["role"]: item
        for item in target_audit["roles"]
        if isinstance(item, Mapping)
    }
    sealed_target = target_roles.get("sealed")
    if not isinstance(sealed_target, Mapping):
        raise ValueError("Round 27 sealed target evidence differs")
    model_name, model_sha256 = _model_identity(
        selected_model,
        contract_sha256=str(contract["contract_sha256"]),
    )
    resolution_evidence_sha256 = str(sealed_target["evidence_chain_sha256"])
    probability_input_sha256 = _canonical_sha256(
        {
            "feature_row_sha256": [
                sample.feature_row_sha256 for sample in sealed.samples
            ],
            "probabilities": [format(float(value), ".17g") for value in probabilities],
        }
    )
    economic_config = Round27EconomicConfig()
    if outputs["economics"].exists():
        sealed_economics = _load_mapping(outputs["economics"])
        if (
            sealed_economics.get("partition_role") != "sealed"
            or sealed_economics.get("model_name") != model_name
            or sealed_economics.get("model_sha256") != model_sha256
            or sealed_economics.get("source_audit_sha256")
            != feature_audit["audit_sha256"]
            or sealed_economics.get("resolution_evidence_sha256")
            != resolution_evidence_sha256
            or sealed_economics.get("probability_input_sha256")
            != probability_input_sha256
            or sealed_economics.get("config") != economic_config.asdict()
            or any(
                sealed_economics.get(field) is not False
                for field in (
                    "edge_claim",
                    "profitability_claim",
                    "orders_submitted",
                    "trading_authority",
                )
            )
        ):
            raise ValueError("Round 27 persisted sealed economics differ")
    else:
        with PolymarketEvidenceStore(
            source_path,
            read_only=True,
            memory_limit=arguments.memory_limit,
            threads=2,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=sealed_run_id,
                condition_ids=sealed_conditions,
            )
            sealed_economics = evaluate_round27_economic_scenarios(
                partition=sealed,
                predictions=probabilities,
                markets=markets,
                outcomes_up=sealed_outcomes,
                model_name=model_name,
                model_sha256=model_sha256,
                source_audit_sha256=str(feature_audit["audit_sha256"]),
                resolution_evidence_sha256=resolution_evidence_sha256,
                config=economic_config,
                book_batches=_batches(
                    source,
                    run_id=sealed_run_id,
                    condition_ids=sealed_conditions,
                    maximum_conditions=int(
                        economics["maximum_conditions_per_book_batch"]
                    ),
                ),
            )
    _writer(outputs["economics"], "report_sha256")(sealed_economics)
    terminal = _terminal_result(
        contract_sha256=str(contract["contract_sha256"]),
        selection_claim=selection_claim,
        selection_economic_claim=selection_economic_claim,
        selection_economic_report=selection_economic_report,
        sealed_prediction=sealed_prediction,
        sealed_economics=sealed_economics,
    )
    _writer(outputs["terminal"], "result_sha256")(terminal)
    print(json.dumps(terminal, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
