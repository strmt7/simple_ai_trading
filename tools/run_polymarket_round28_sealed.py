#!/usr/bin/env python3
"""Run the frozen Round 28 pair once on the sealed Stage 1 population."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES,
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
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round28_book_ticker import (
    compose_round28_feature_rows,
    load_round28_book_ticker_overlay,
)
from simple_ai_trading.polymarket_round28_model import build_round28_model_samples
from simple_ai_trading.polymarket_round28_operator import (
    validate_round28_selection_input_manifest,
)
from simple_ai_trading.polymarket_round28_sealed import (
    build_round28_sealed_terminal_result,
    evaluate_round28_sealed_economics,
    evaluate_round28_sealed_prediction,
    validate_round28_sealed_access_artifacts,
    validate_round28_sealed_economic_report,
    validate_round28_sealed_prediction_result,
    validate_round28_sealed_terminal_result,
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
    parser.add_argument("--round27-target-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-economic-report", type=Path, required=True)
    parser.add_argument("--sealed-prediction-result", type=Path, required=True)
    parser.add_argument("--sealed-economic-report", type=Path, required=True)
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 sealed path must not be a symlink")
    return selected.resolve(strict=strict)


def _role(audit: Mapping[str, object], role: str) -> Mapping[str, object]:
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Round 28 sealed target roles differ")
    matches = tuple(
        item
        for item in roles
        if isinstance(item, Mapping) and item.get("role") == role
    )
    if len(matches) != 1 or matches[0].get("finalized") is not True:
        raise ValueError(f"Round 28 {role} target evidence is not terminal")
    return matches[0]


def _source_binding(
    *,
    feature_audit: Mapping[str, object],
    overlay_report: Mapping[str, object],
    target_audit: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    sealed_condition_ids: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "schema_version": "polymarket-round28-sealed-source-binding-v1",
            "round27_feature_store_audit_sha256": feature_audit["audit_sha256"],
            "round28_overlay_report_sha256": overlay_report["report_sha256"],
            "round27_target_store_audit_sha256": target_audit["audit_sha256"],
            "round27_campaign_admission_sha256": target_audit[
                "campaign_admission_sha256"
            ],
            "round27_model_contract_sha256": contract["contract_sha256"],
            "round28_preregistration_sha256": preregistration[
                "preregistration_sha256"
            ],
            "selection_input_manifest_sha256": selection_manifest[
                "manifest_sha256"
            ],
            "selection_claim_sha256": selection_claim["claim_sha256"],
            "sealed_condition_population_sha256": canonical_sha256(
                list(sealed_condition_ids)
            ),
        }
    )


def _print_result(terminal: Mapping[str, object]) -> None:
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "feature": _resolve(repository, args.round27_feature_store, strict=True),
        "overlay": _resolve(repository, args.round28_overlay_store, strict=True),
        "target": _resolve(repository, args.round27_target_store, strict=True),
        "source": _resolve(repository, args.sealed_source_database, strict=True),
        "manifest": _resolve(
            repository,
            args.selection_input_manifest,
            strict=True,
        ),
        "selection": _resolve(repository, args.selection_claim, strict=True),
        "selection_economics": _resolve(
            repository,
            args.selection_economic_report,
            strict=True,
        ),
    }
    outputs = {
        "prediction": _resolve(
            repository,
            args.sealed_prediction_result,
            strict=False,
        ),
        "economics": _resolve(
            repository,
            args.sealed_economic_report,
            strict=False,
        ),
        "terminal": _resolve(repository, args.terminal_result, strict=False),
    }
    all_paths = (*inputs.values(), *outputs.values())
    if (
        any(not path.is_file() for path in inputs.values())
        or any(Path(f"{inputs[key]}.wal").exists() for key in ("feature", "overlay", "target", "source"))
        or len(set(all_paths)) != len(all_paths)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 28 sealed paths differ")

    contract = load_round27_model_contract(repository)
    preregistration = load_mapping(repository / _PREREGISTRATION)
    manifest = validate_round28_selection_input_manifest(
        load_mapping(inputs["manifest"])
    )
    selection_claim = load_mapping(inputs["selection"])
    selection_economics = load_mapping(inputs["selection_economics"])
    partitions = contract.get("partitions")
    economics_contract = contract.get("economic_evaluation")
    if not isinstance(partitions, list) or not isinstance(
        economics_contract,
        Mapping,
    ):
        raise ValueError("Round 28 sealed inherited contract differs")

    with Round27FeatureStore(inputs["feature"], read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        all_base_rows = tuple(
            sorted(feature_store.load_rows(), key=lambda row: row.decision_time_ms)
        )
        sealed_base_rows = feature_store.load_rows(slot_id="stage1-c")
    overlay_rows, overlay_report = load_round28_book_ticker_overlay(
        inputs["overlay"]
    )
    if (
        feature_audit["audit_sha256"]
        != manifest["round27_feature_store_audit_sha256"]
        or overlay_report["report_sha256"]
        != manifest["round28_overlay_report_sha256"]
    ):
        raise ValueError("Round 28 sealed feature lineage differs")
    composed = compose_round28_feature_rows(
        base_rows=all_base_rows,
        overlay_rows=overlay_rows,
        report=overlay_report,
    )
    sealed_decisions = {row.decision_time_ms for row in sealed_base_rows}
    sealed_rows = tuple(
        row for row in composed if row.decision_time_ms in sealed_decisions
    )
    sealed_condition_ids = tuple(
        condition_id
        for condition_id, _start in sorted(
            {
                row.condition_id: row.event_start_ms for row in sealed_rows
            }.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    sealed_run_ids = {row.run_id for row in sealed_rows}
    if not sealed_condition_ids or len(sealed_run_ids) != 1:
        raise ValueError("Round 28 sealed feature population differs")
    sealed_run_id = next(iter(sealed_run_ids))

    with Round27TargetStore(inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        selection_target = _role(target_audit, "selection")
        sealed_target = _role(target_audit, "sealed")
        if (
            target_audit["audit_sha256"]
            != manifest["round27_target_store_audit_sha256"]
        ):
            raise ValueError("Round 28 sealed target-store lineage differs")
        pair = validate_round28_sealed_access_artifacts(
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest=manifest,
            selection_claim=selection_claim,
            selection_economic_report=selection_economics,
            selection_resolution_evidence_sha256=str(
                selection_target["evidence_chain_sha256"]
            ),
        )
        source_binding_sha256 = _source_binding(
            feature_audit=feature_audit,
            overlay_report=overlay_report,
            target_audit=target_audit,
            contract=contract,
            preregistration=preregistration,
            selection_manifest=manifest,
            selection_claim=selection_claim,
            sealed_condition_ids=sealed_condition_ids,
        )

        if outputs["terminal"].exists():
            if not outputs["prediction"].is_file() or not outputs["economics"].is_file():
                raise ValueError("Round 28 sealed terminal artifacts are incomplete")
            prediction = validate_round28_sealed_prediction_result(
                load_mapping(outputs["prediction"])
            )
            economic_report = validate_round28_sealed_economic_report(
                load_mapping(outputs["economics"])
            )
            terminal = validate_round28_sealed_terminal_result(
                load_mapping(outputs["terminal"])
            )
            rebuilt = build_round28_sealed_terminal_result(
                sealed_prediction_result=prediction,
                sealed_economic_report=economic_report,
            )
            if (
                terminal != rebuilt
                or prediction["source_binding_sha256"] != source_binding_sha256
                or economic_report["source_binding_sha256"]
                != source_binding_sha256
            ):
                raise ValueError("Round 28 persisted sealed result differs")
            _print_result(terminal)
            return 0

        outcomes = target_store.outcomes_up(roles=("sealed",))

    if set(outcomes) != set(sealed_condition_ids):
        raise ValueError("Round 28 sealed target population differs")
    samples = build_round28_model_samples(
        rows_by_slot={"stage1-c": sealed_rows},
        outcomes_up=outcomes,
        role_intervals=[item for item in partitions if isinstance(item, Mapping)],
    )
    expected_prediction = evaluate_round28_sealed_prediction(
        samples=samples,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=manifest,
        selection_claim=selection_claim,
        selection_economic_report=selection_economics,
        selection_resolution_evidence_sha256=str(
            selection_target["evidence_chain_sha256"]
        ),
        source_binding_sha256=source_binding_sha256,
    )
    if outputs["prediction"].exists():
        prediction = validate_round28_sealed_prediction_result(
            load_mapping(outputs["prediction"])
        )
        if prediction != expected_prediction:
            raise ValueError("Round 28 persisted sealed prediction differs")
    else:
        prediction = expected_prediction
        artifact_writer(outputs["prediction"], "result_sha256")(prediction)

    if outputs["economics"].exists():
        economic_report = validate_round28_sealed_economic_report(
            load_mapping(outputs["economics"])
        )
        if (
            economic_report["sealed_prediction_result_sha256"]
            != prediction["result_sha256"]
            or economic_report["source_binding_sha256"]
            != source_binding_sha256
            or economic_report["resolution_evidence_sha256"]
            != sealed_target["evidence_chain_sha256"]
        ):
            raise ValueError("Round 28 persisted sealed economics differ")
    else:
        config = Round27EconomicConfig(
            minimum_executed_trades=(
                POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES
            )
        ).validated()
        with PolymarketEvidenceStore(
            inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source_store:
            markets = PolymarketEvidenceReplay.load_markets(
                source_store,
                run_id=sealed_run_id,
                condition_ids=sealed_condition_ids,
            )

            def batches():
                return economic_book_batches(
                    source_store,
                    run_id=sealed_run_id,
                    condition_ids=sealed_condition_ids,
                    maximum_conditions=int(
                        economics_contract["maximum_conditions_per_book_batch"]
                    ),
                )

            economic_report = evaluate_round28_sealed_economics(
                samples=samples,
                pair=pair,
                selection_claim_sha256=str(selection_claim["claim_sha256"]),
                sealed_prediction_result=prediction,
                markets=markets,
                outcomes_up=outcomes,
                source_binding_sha256=source_binding_sha256,
                resolution_evidence_sha256=str(
                    sealed_target["evidence_chain_sha256"]
                ),
                config=config,
                book_batch_factory=batches,
            )
        validate_round28_sealed_economic_report(economic_report)
        artifact_writer(outputs["economics"], "report_sha256")(economic_report)

    terminal = build_round28_sealed_terminal_result(
        sealed_prediction_result=prediction,
        sealed_economic_report=economic_report,
    )
    artifact_writer(outputs["terminal"], "result_sha256")(terminal)
    _print_result(terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
