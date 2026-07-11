from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_VERSION,
    apply_path_aware_lifecycle_targets,
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_model import (
    MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
    save_microstructure_model_artifact,
    train_microstructure_action_value_model,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.storage import write_json_atomic


DESIGN_SCHEMA_VERSION = "action-value-v15-discovery-design-v1"
_RISK_LEVELS = ("conservative", "regular", "aggressive")
_RISK_PENALTY = {"conservative": 2.0, "regular": 1.5, "aggressive": 1.0}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_discovery_design(path: str | Path) -> dict[str, object]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("action-value discovery design is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("action-value discovery design must be a JSON object")
    expected_sha256 = str(payload.get("design_sha256") or "")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if expected_sha256 != _canonical_sha256(canonical):
        raise ValueError("action-value discovery design digest does not match its payload")
    if payload.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError("action-value discovery design schema is unsupported")
    if (
        payload.get("status") != "precommitted"
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("terminal_holdout") is not False
    ):
        raise ValueError("action-value discovery design authority contract is invalid")

    data = payload.get("data")
    execution = payload.get("execution")
    training = payload.get("training")
    risk_profiles = payload.get("risk_profiles")
    horizons = payload.get("horizon_seconds")
    if not all(
        isinstance(value, Mapping)
        for value in (data, execution, training, risk_profiles)
    ) or not isinstance(horizons, list):
        raise ValueError("action-value discovery design sections are incomplete")
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(risk_profiles, Mapping)
    if (
        data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or data.get("symbol") != "BTCUSDT"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not True
        or data.get("checksum_verified_partitions_required") is not True
        or data.get("selection_contaminated_after_run") is not True
    ):
        raise ValueError("action-value discovery data contract is invalid")
    try:
        start = datetime.strptime(str(data["start_date"]), "%Y-%m-%d").date()
        end = datetime.strptime(str(data["end_date"]), "%Y-%m-%d").date()
    except (KeyError, ValueError) as exc:
        raise ValueError("action-value discovery date window is invalid") from exc
    if start > end or (end - start).days + 1 < 5:
        raise ValueError("action-value discovery window must contain at least five UTC days")
    consumed = {str(value) for value in data.get("excluded_consumed_dates") or ()}
    cursor = start
    while cursor <= end:
        if cursor.isoformat() in consumed:
            raise ValueError("action-value discovery window intersects consumed evidence")
        cursor += timedelta(days=1)

    if (
        training.get("model_schema_version") != MICROSTRUCTURE_MODEL_SCHEMA_VERSION
        or training.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION
        or training.get("model_family") != "side_specific_hurdle_expected_value"
        or training.get("reproducible_backend") is not True
        or training.get("evaluate_terminal") is not False
    ):
        raise ValueError("action-value discovery training contract is invalid")
    if tuple(risk_profiles) != _RISK_LEVELS:
        raise ValueError("action-value discovery risk profiles are missing or out of order")
    for risk_level in _RISK_LEVELS:
        profile = risk_profiles[risk_level]
        if not isinstance(profile, Mapping):
            raise ValueError("action-value discovery risk profile is invalid")
        values = tuple(
            float(profile[name])
            for name in ("stop_loss_bps", "take_profit_bps", "max_l1_participation")
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("action-value discovery risk values must be positive")
        if values[1] <= values[0] or values[2] > 1.0:
            raise ValueError("action-value discovery risk profile is inconsistent")
    normalized_horizons = [int(value) for value in horizons]
    if (
        normalized_horizons != sorted(set(normalized_horizons))
        or any(value <= 0 for value in normalized_horizons)
        or int(payload.get("candidate_count") or 0)
        != len(normalized_horizons) * len(_RISK_LEVELS)
    ):
        raise ValueError("action-value discovery horizon matrix is invalid")
    numeric_execution = tuple(
        float(execution[name])
        for name in (
            "total_latency_ms",
            "taker_fee_bps_per_side",
            "additional_slippage_bps_per_side",
            "trigger_execution_slippage_bps",
            "max_quote_age_ms",
            "reference_order_notional_quote",
            "decision_cadence_seconds",
            "leverage",
        )
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric_execution):
        raise ValueError("action-value discovery execution values are invalid")
    if (
        float(execution["reference_order_notional_quote"]) <= 0.0
        or int(execution["decision_cadence_seconds"]) <= 0
        or float(execution["leverage"]) != 1.0
        or execution.get("maker_fill_claim") is not False
        or execution.get("suppress_overlapping_positions") is not True
    ):
        raise ValueError("action-value discovery execution contract is inconsistent")
    return payload


def discovery_candidates(design: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    profiles = design["risk_profiles"]
    horizons = design["horizon_seconds"]
    assert isinstance(profiles, Mapping)
    assert isinstance(horizons, list)
    candidates: list[dict[str, object]] = []
    for horizon in horizons:
        for risk_level in _RISK_LEVELS:
            profile = profiles[risk_level]
            assert isinstance(profile, Mapping)
            candidates.append(
                {
                    "candidate_id": f"{risk_level}-h{int(horizon)}",
                    "horizon_seconds": int(horizon),
                    "risk_level": risk_level,
                    "stop_loss_bps": float(profile["stop_loss_bps"]),
                    "take_profit_bps": float(profile["take_profit_bps"]),
                    "max_l1_participation": float(profile["max_l1_participation"]),
                }
            )
    return tuple(candidates)


def _date_bounds(design: Mapping[str, object]) -> tuple[int, int]:
    data = design["data"]
    assert isinstance(data, Mapping)
    start = datetime.strptime(str(data["start_date"]), "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(str(data["end_date"]), "%Y-%m-%d").replace(tzinfo=UTC)
    return int(start.timestamp() * 1_000), int((end + timedelta(days=1)).timestamp() * 1_000) - 1


def _result_from_artifact(
    candidate: Mapping[str, object],
    artifact,
    *,
    artifact_path: Path,
    artifact_sha256: str,
    path_evidence,
) -> dict[str, object]:
    metrics = artifact.selection_metrics
    risk_level = str(candidate["risk_level"])
    research_score = metrics.total_net_bps - _RISK_PENALTY[risk_level] * metrics.max_drawdown_bps
    source = artifact.dataset_summary.get("source_evidence")
    certificate = source.get("corpus_certificate") if isinstance(source, Mapping) else None
    return {
        **dict(candidate),
        "status": artifact.status,
        "rejection_reasons": list(artifact.rejection_reasons),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "model_schema_version": artifact.schema_version,
        "feature_version": artifact.feature_version,
        "backend_kind": artifact.training_backend_kind,
        "backend_device": artifact.training_backend_device,
        "rows": int(artifact.dataset_summary["rows"]),
        "unique_utc_days": artifact.unique_utc_days,
        "corpus_certificate_sha256": (
            certificate.get("certificate_sha256")
            if isinstance(certificate, Mapping)
            else None
        ),
        "policy": asdict(artifact.threshold_policy),
        "policy_metrics": asdict(artifact.policy_metrics),
        "selection_metrics": asdict(metrics),
        "selection_confidence": asdict(artifact.selection_confidence),
        "selection_auc": dict(artifact.selection_auc),
        "selection_brier": dict(artifact.selection_brier),
        "path_target_evidence": asdict(path_evidence),
        "research_score_bps": float(research_score),
        "terminal_evaluated": False,
        "trading_authority": False,
    }


def run_discovery(
    design_path: str | Path,
    *,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str = "8GB",
    threads: int = 8,
    compute_backend: str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    design = load_discovery_design(design_path)
    design_sha256 = str(design["design_sha256"])
    candidates = discovery_candidates(design)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    report_path = output / "report.json"
    execution = design["execution"]
    training = design["training"]
    data = design["data"]
    assert isinstance(execution, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(data, Mapping)
    backend = str(compute_backend or training["compute_backend"])
    start_ms, end_ms = _date_bounds(design)
    completed: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    def persist_status(phase: str, current: str | None = None) -> None:
        write_json_atomic(
            status_path,
            {
                "schema_version": DESIGN_SCHEMA_VERSION,
                "design_sha256": design_sha256,
                "phase": phase,
                "current_candidate": current,
                "completed_candidates": len(completed),
                "failed_candidates": len(errors),
                "total_candidates": len(candidates),
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )

    def progress(phase: str, complete: int, total: int | None) -> None:
        print(f"action-value-discovery {phase}: {complete}/{total if total is not None else '?'}", flush=True)

    persist_status("initializing")
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        warehouse.backfill_book_ticker_paths(progress=progress)
        warehouse.rebuild_causal_feature_bars(str(data["symbol"]), progress=progress)
        warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=True,
        )
        by_horizon: dict[int, list[dict[str, object]]] = {}
        for candidate in candidates:
            by_horizon.setdefault(int(candidate["horizon_seconds"]), []).append(candidate)
        for horizon, horizon_candidates in by_horizon.items():
            persist_status(f"build-h{horizon}")
            base_dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=str(data["symbol"]),
                horizon_seconds=horizon,
                total_latency_ms=int(execution["total_latency_ms"]),
                taker_fee_bps=float(execution["taker_fee_bps_per_side"]),
                additional_slippage_bps_per_side=float(
                    execution["additional_slippage_bps_per_side"]
                ),
                max_quote_age_ms=int(execution["max_quote_age_ms"]),
                reference_order_notional_quote=float(
                    execution["reference_order_notional_quote"]
                ),
                max_l1_participation=1.0,
                decision_cadence_seconds=int(execution["decision_cadence_seconds"]),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            for candidate in horizon_candidates:
                candidate_id = str(candidate["candidate_id"])
                artifact_path = output / f"{candidate_id}.json"
                result_path = output / f"{candidate_id}.result.json"
                if resume and artifact_path.exists() and result_path.exists():
                    try:
                        prior = json.loads(result_path.read_text(encoding="utf-8"))
                        if (
                            isinstance(prior, dict)
                            and prior.get("design_sha256") == design_sha256
                            and prior.get("candidate_id") == candidate_id
                            and prior.get("artifact_sha256") == _file_sha256(artifact_path)
                            and isinstance(prior.get("result"), Mapping)
                        ):
                            completed.append(dict(prior["result"]))
                            persist_status("resumed", candidate_id)
                            continue
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
                        pass
                persist_status("training", candidate_id)
                try:
                    limit = float(candidate["max_l1_participation"])
                    candidate_dataset = replace(
                        base_dataset,
                        max_l1_participation=limit,
                        long_liquidity_eligible=(
                            np.asarray(base_dataset.long_l1_participation) <= limit
                        ),
                        short_liquidity_eligible=(
                            np.asarray(base_dataset.short_l1_participation) <= limit
                        ),
                    )
                    candidate_dataset, path_evidence = apply_path_aware_lifecycle_targets(
                        warehouse,
                        candidate_dataset,
                        stop_loss_bps=float(candidate["stop_loss_bps"]),
                        take_profit_bps=float(candidate["take_profit_bps"]),
                        trigger_execution_slippage_bps=float(
                            execution["trigger_execution_slippage_bps"]
                        ),
                    )
                    artifact = train_microstructure_action_value_model(
                        candidate_dataset,
                        risk_level=str(candidate["risk_level"]),
                        compute_backend=backend,
                        seed=int(training["seed"]),
                        minimum_promotion_days=int(training["minimum_discovery_days"]),
                        deployment_calibration_days=2,
                        maximum_model_age_seconds=86_400,
                        evaluate_terminal=False,
                        progress=progress,
                    )
                    artifact_sha256 = save_microstructure_model_artifact(
                        artifact,
                        artifact_path,
                    )
                    result = _result_from_artifact(
                        candidate,
                        artifact,
                        artifact_path=artifact_path,
                        artifact_sha256=artifact_sha256,
                        path_evidence=path_evidence,
                    )
                    write_json_atomic(
                        result_path,
                        {
                            "design_sha256": design_sha256,
                            "candidate_id": candidate_id,
                            "artifact_sha256": artifact_sha256,
                            "result": result,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    completed.append(result)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "candidate_id": candidate_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                persist_status("candidate-complete", candidate_id)

    ranked = sorted(
        completed,
        key=lambda value: (
            float(value["research_score_bps"]),
            float(value["selection_metrics"]["total_net_bps"]),
        ),
        reverse=True,
    )
    report = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_sha256": design_sha256,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "completed"
            if len(completed) == len(candidates) and not errors
            else "failed"
        ),
        "artifact_class": "exchange_sourced_discovery_evidence",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "terminal_holdout_accessed": False,
        "selection_window_is_consumed": True,
        "warehouse": str(warehouse_path),
        "candidate_count": len(candidates),
        "completed_candidate_count": len(completed),
        "failed_candidate_count": len(errors),
        "unrejected_candidate_count": sum(
            value["status"] == "candidate" for value in completed
        ),
        "best_discovery_candidate": ranked[0] if ranked else None,
        "ranked_results": ranked,
        "errors": errors,
    }
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    persist_status("complete" if report["status"] == "completed" else "failed")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the precommitted v15 action-value discovery screen")
    parser.add_argument(
        "--design",
        default="docs/model-research/action-value/round-009-design.json",
    )
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compute-backend", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_discovery(
            args.design,
            warehouse_path=args.warehouse,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            memory_limit=args.memory_limit,
            threads=args.threads,
            compute_backend=args.compute_backend,
            resume=args.resume,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"action-value-discovery failed: {type(exc).__name__}: {exc}")
        return 2
    print(
        "action-value-discovery: "
        f"status={report['status']} completed={report['completed_candidate_count']}/"
        f"{report['candidate_count']} unrejected={report['unrejected_candidate_count']}"
    )
    best = report.get("best_discovery_candidate")
    if isinstance(best, Mapping):
        metrics = best["selection_metrics"]
        print(
            f"best={best['candidate_id']} status={best['status']} "
            f"trades={metrics['trades']} net_bps={float(metrics['total_net_bps']):+.4f} "
            f"max_dd_bps={float(metrics['max_drawdown_bps']):.4f}"
        )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
