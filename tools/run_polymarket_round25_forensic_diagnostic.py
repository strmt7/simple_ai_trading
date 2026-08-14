"""Run one leakage-safe stage of the Round 25 forensic diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time

from simple_ai_trading.polymarket_round25_forensic_model import (
    evaluate_round25_forensic_selection,
    fit_and_freeze_round25_forensic_models,
    validate_round25_forensic_model_fit,
    validate_round25_forensic_prediction_artifact,
    write_round25_forensic_model_artifacts,
    write_round25_forensic_result,
)
from simple_ai_trading.polymarket_round25_forensic_resolution import (
    collect_round25_forensic_resolutions_once,
    initialize_round25_forensic_resolution_collection,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25ResolutionPublicClient,
)


def _mapping(path: Path, *, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 25 forensic {label} is not an object")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance exactly one target-isolated Round 25 diagnostic stage."
    )
    parser.add_argument("stage", choices=("fit", "selection"))
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--fit-resolutions", type=Path, required=True)
    parser.add_argument("--model-fit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--selection-resolutions", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--maximum-conditions", type=int, default=128)
    parser.add_argument("--minimum-request-interval", type=float, default=0.2)
    return parser


def _client(args: argparse.Namespace) -> Round25ResolutionPublicClient:
    return Round25ResolutionPublicClient(
        minimum_request_interval_seconds=args.minimum_request_interval
    )


def _fit(args: argparse.Namespace, partition: Mapping[str, object]) -> dict[str, object]:
    now = time.time_ns() // 1_000_000
    collection, claim = initialize_round25_forensic_resolution_collection(
        feature_database=args.feature_store,
        partition_manifest=partition,
        destination_database=args.fit_resolutions,
        stage="fit",
        created_at_ms=now,
    )
    report = collect_round25_forensic_resolutions_once(
        collection_database=collection,
        client=_client(args),
        maximum_conditions=args.maximum_conditions,
    )
    if not report["complete"]:
        return {
            "claim_sha256": claim["claim_sha256"],
            "collection": report,
            "next_stage": "rerun_fit_collection",
            "stage": "fit",
        }
    model_fit, prediction = fit_and_freeze_round25_forensic_models(
        feature_database=args.feature_store,
        partition_manifest=partition,
        fit_resolution_database=collection,
        created_at_ms=int(claim["created_at_ms"]) + 1,
    )
    write_round25_forensic_model_artifacts(
        model_fit_path=args.model_fit,
        prediction_path=args.predictions,
        model_fit=model_fit,
        prediction=prediction,
    )
    return {
        "claim_sha256": claim["claim_sha256"],
        "collection": report,
        "model_fit_sha256": model_fit["model_fit_sha256"],
        "next_stage": "selection",
        "prediction_artifact_sha256": prediction["prediction_artifact_sha256"],
        "selected_candidate_id": model_fit["selected_candidate_id"],
        "stage": "fit",
    }


def _selection(
    args: argparse.Namespace,
    partition: Mapping[str, object],
) -> dict[str, object]:
    if args.selection_resolutions is None or args.result is None:
        raise ValueError("selection requires --selection-resolutions and --result")
    prediction = validate_round25_forensic_prediction_artifact(
        _mapping(args.predictions, label="prediction artifact")
    )
    model_fit = validate_round25_forensic_model_fit(
        _mapping(args.model_fit, label="model-fit artifact")
    )
    if model_fit["model_fit_sha256"] != prediction["model_fit_sha256"]:
        raise ValueError("Round 25 forensic fitted model and prediction differ")
    access = prediction["access_freeze"]
    if not isinstance(access, Mapping):
        raise ValueError("Round 25 forensic selection freeze is unavailable")
    collection, claim = initialize_round25_forensic_resolution_collection(
        feature_database=args.feature_store,
        partition_manifest=partition,
        destination_database=args.selection_resolutions,
        stage="selection",
        created_at_ms=time.time_ns() // 1_000_000,
        selection_freeze=access,
    )
    report = collect_round25_forensic_resolutions_once(
        collection_database=collection,
        client=_client(args),
        maximum_conditions=args.maximum_conditions,
    )
    if not report["complete"]:
        return {
            "claim_sha256": claim["claim_sha256"],
            "collection": report,
            "next_stage": "rerun_selection_collection",
            "stage": "selection",
        }
    result = evaluate_round25_forensic_selection(
        prediction=prediction,
        selection_resolution_database=collection,
        created_at_ms=int(claim["created_at_ms"]) + 1,
    )
    write_round25_forensic_result(args.result, result)
    return {
        "claim_sha256": claim["claim_sha256"],
        "collection": report,
        "diagnostic_economic_gate_passed": result[
            "diagnostic_economic_gate_passed"
        ],
        "diagnostic_predictive_gate_passed": result[
            "diagnostic_predictive_gate_passed"
        ],
        "next_stage": "publish_truthful_diagnostic",
        "result_sha256": result["result_sha256"],
        "stage": "selection",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    partition = _mapping(args.partition, label="partition manifest")
    output = _fit(args, partition) if args.stage == "fit" else _selection(args, partition)
    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
    return 0 if not str(output["next_stage"]).startswith("rerun_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
