from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np

from simple_ai_trading.microstructure_features import (
    apply_path_aware_lifecycle_targets,
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_model import (
    _apply_platt_scaling,
    _purged_split,
    _purged_tuning_subsplit,
    load_microstructure_model_artifact,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.storage import write_json_atomic
try:
    from tools.run_action_value_discovery import (
        _date_bounds,
        discovery_candidates,
        load_discovery_design,
    )
except ModuleNotFoundError:
    from run_action_value_discovery import (
        _date_bounds,
        discovery_candidates,
        load_discovery_design,
    )


_TOP_COUNTS = (20, 50, 100, 250, 500, 1_000)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct non-terminal action-value discovery diagnostics"
    )
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _rank_ic(scores: np.ndarray, targets: np.ndarray) -> float | None:
    if len(scores) < 2:
        return None
    score_ranks = _average_ranks(scores)
    target_ranks = _average_ranks(targets)
    if float(np.std(score_ranks)) <= 0.0 or float(np.std(target_ranks)) <= 0.0:
        return None
    value = float(np.corrcoef(score_ranks, target_ranks)[0, 1])
    return value if math.isfinite(value) else None


def _top_score_diagnostic(
    predicted_edge: np.ndarray,
    actual_net_bps: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, object]:
    edge = np.asarray(predicted_edge, dtype=np.float64)
    actual = np.asarray(actual_net_bps, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if edge.shape != actual.shape or edge.shape != mask.shape:
        raise ValueError("action-value diagnostic arrays are inconsistent")
    usable = np.flatnonzero(mask)
    order = usable[np.argsort(edge[usable], kind="stable")[::-1]]
    top: list[dict[str, object]] = []
    for requested in _TOP_COUNTS:
        count = min(requested, len(order))
        if count <= 0:
            continue
        indexes = order[:count]
        top.append(
            {
                "requested_rows": requested,
                "rows": count,
                "mean_predicted_edge_bps": float(np.mean(edge[indexes])),
                "mean_actual_net_bps": float(np.mean(actual[indexes])),
                "actual_profitable_ratio": float(np.mean(actual[indexes] > 0.0)),
                "actual_total_net_bps": float(np.sum(actual[indexes])),
            }
        )
    return {
        "eligible_rows": int(len(usable)),
        "positive_predicted_edge_rows": int(np.sum(edge[usable] > 0.0)),
        "predicted_edge_actual_spearman_ic": _rank_ic(edge[usable], actual[usable]),
        "top_score_rows": top,
    }


def _class_support(dataset, splits: Mapping[str, np.ndarray], split_evidence) -> dict[str, object]:
    early_stop, calibration = _purged_tuning_subsplit(
        dataset.decision_time_ms,
        splits["tuning"],
        purge_ms=split_evidence.purge_ms,
    )
    segments = {
        "train": splits["train"],
        "early_stop": early_stop,
        "calibration": calibration,
        "policy": splits["policy"],
        "selection": splits["selection"],
    }
    output: dict[str, object] = {}
    for side, targets, eligible in (
        ("long", dataset.long_net_bps, dataset.long_liquidity_eligible),
        ("short", dataset.short_net_bps, dataset.short_liquidity_eligible),
    ):
        side_values: dict[str, object] = {}
        for name, indexes in segments.items():
            side_indexes = indexes[np.asarray(eligible[indexes], dtype=bool)]
            positive = int(np.sum(targets[side_indexes] > 0.0))
            side_values[name] = {
                "eligible_rows": int(len(side_indexes)),
                "profitable_rows": positive,
                "non_profitable_rows": int(len(side_indexes) - positive),
                "profitable_ratio": float(positive / max(1, len(side_indexes))),
            }
        output[side] = side_values
    return output


def _artifact_diagnostics(artifact, dataset, splits: Mapping[str, np.ndarray]) -> dict[str, object]:
    if artifact.terminal_evaluated_at is not None or artifact.terminal_metrics is not None:
        raise ValueError("discovery diagnostic refuses terminal-evaluated artifacts")
    models = {
        name: lgb.Booster(model_str=model_string)
        for name, model_string in artifact.model_strings.items()
    }
    segment_values: dict[str, object] = {}
    for segment in ("policy", "selection"):
        indexes = splits[segment]
        predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        side_values: dict[str, object] = {}
        for side, actual, eligible in (
            ("long", dataset.long_net_bps[indexes], dataset.long_liquidity_eligible[indexes]),
            ("short", dataset.short_net_bps[indexes], dataset.short_liquidity_eligible[indexes]),
        ):
            probability_name = f"{side}_probability"
            raw_probability = models[probability_name].predict(
                dataset.features[indexes],
                num_iteration=artifact.best_iterations[probability_name],
            )
            probability = _apply_platt_scaling(
                raw_probability,
                artifact.probability_calibration[side],
            )
            win_name = f"{side}_win_magnitude"
            loss_name = f"{side}_loss_magnitude"
            win = np.maximum(
                0.0,
                models[win_name].predict(
                    dataset.features[indexes],
                    num_iteration=artifact.best_iterations[win_name],
                ),
            )
            loss = np.maximum(
                0.0,
                models[loss_name].predict(
                    dataset.features[indexes],
                    num_iteration=artifact.best_iterations[loss_name],
                ),
            )
            edge = probability * win - (1.0 - probability) * loss
            predictions[side] = (
                np.asarray(edge, dtype=np.float64),
                np.asarray(actual, dtype=np.float64),
                np.asarray(eligible, dtype=bool),
            )
            side_values[side] = {
                **_top_score_diagnostic(edge, actual, eligible),
                "mean_profitable_probability": float(np.mean(probability)),
            }

        long_edge, long_actual, long_eligible = predictions["long"]
        short_edge, short_actual, short_eligible = predictions["short"]
        valid = long_eligible | short_eligible
        choose_long = np.where(
            long_eligible & short_eligible,
            long_edge >= short_edge,
            long_eligible,
        )
        best_edge = np.where(choose_long, long_edge, short_edge)
        best_actual = np.where(choose_long, long_actual, short_actual)
        segment_values[segment] = {
            "sides": side_values,
            "best_predicted_side": _top_score_diagnostic(
                best_edge,
                best_actual,
                valid,
            ),
            "overlap_suppression_applied": False,
            "trade_claim": False,
        }
    return {
        "artifact_status": artifact.status,
        "threshold_policy": asdict(artifact.threshold_policy),
        "policy_metrics": asdict(artifact.policy_metrics),
        "selection_metrics": asdict(artifact.selection_metrics),
        "selection_auc": dict(artifact.selection_auc),
        "selection_brier": dict(artifact.selection_brier),
        "probability_calibration": {
            key: list(value) for key, value in artifact.probability_calibration.items()
        },
        "segments": segment_values,
    }


def diagnose(
    design_path: Path,
    evidence_root: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
    *,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    design = load_discovery_design(design_path)
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    if (
        report.get("design_sha256") != design["design_sha256"]
        or report.get("terminal_holdout_accessed") is not False
        or report.get("trading_authority") is not False
    ):
        raise ValueError("action-value discovery report binding is invalid")
    ranked = report.get("ranked_results")
    errors = report.get("errors")
    if not isinstance(ranked, list) or not isinstance(errors, list):
        raise ValueError("action-value discovery outcomes are invalid")
    trained = {
        str(value["candidate_id"]): value
        for value in ranked
        if isinstance(value, Mapping)
    }
    failed = {
        str(value["candidate_id"]): value
        for value in errors
        if isinstance(value, Mapping)
    }
    candidates = discovery_candidates(design)
    start_ms, end_ms = _date_bounds(design)
    data = design["data"]
    execution = design["execution"]
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    results: list[dict[str, object]] = []
    certificate_sha256: str | None = None
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=True,
        )
        certificate_sha256 = str(certificate["certificate_sha256"])
        by_horizon: dict[int, list[Mapping[str, object]]] = {}
        for candidate in candidates:
            by_horizon.setdefault(int(candidate["horizon_seconds"]), []).append(candidate)
        for horizon, horizon_candidates in by_horizon.items():
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
                limit = float(candidate["max_l1_participation"])
                dataset = replace(
                    base_dataset,
                    max_l1_participation=limit,
                    long_liquidity_eligible=(
                        np.asarray(base_dataset.long_l1_participation) <= limit
                    ),
                    short_liquidity_eligible=(
                        np.asarray(base_dataset.short_l1_participation) <= limit
                    ),
                )
                dataset, path_evidence = apply_path_aware_lifecycle_targets(
                    warehouse,
                    dataset,
                    stop_loss_bps=float(candidate["stop_loss_bps"]),
                    take_profit_bps=float(candidate["take_profit_bps"]),
                    trigger_execution_slippage_bps=float(
                        execution["trigger_execution_slippage_bps"]
                    ),
                )
                splits, split_evidence = _purged_split(dataset)
                value: dict[str, object] = {
                    **dict(candidate),
                    "dataset_rows": dataset.rows,
                    "split": asdict(split_evidence),
                    "class_support": _class_support(dataset, splits, split_evidence),
                    "path_target_evidence": asdict(path_evidence),
                    "terminal_accessed": False,
                }
                trained_result = trained.get(candidate_id)
                failed_result = failed.get(candidate_id)
                if (trained_result is None) == (failed_result is None):
                    raise ValueError(f"candidate outcome is ambiguous: {candidate_id}")
                if failed_result is not None:
                    value.update(
                        {
                            "fit_status": "fit_error",
                            "error": str(failed_result.get("error") or ""),
                        }
                    )
                else:
                    assert trained_result is not None
                    artifact_path = evidence_root / f"{candidate_id}.json"
                    artifact_sha256 = _file_sha256(artifact_path)
                    if artifact_sha256 != trained_result.get("artifact_sha256"):
                        raise ValueError(f"artifact hash mismatch: {candidate_id}")
                    artifact = load_microstructure_model_artifact(artifact_path)
                    artifact_certificate = artifact.dataset_summary.get(
                        "source_evidence"
                    )
                    if not isinstance(artifact_certificate, Mapping):
                        raise ValueError(f"artifact source evidence is invalid: {candidate_id}")
                    bound_certificate = artifact_certificate.get("corpus_certificate")
                    if (
                        not isinstance(bound_certificate, Mapping)
                        or bound_certificate.get("certificate_sha256")
                        != certificate_sha256
                    ):
                        raise ValueError(f"artifact corpus binding failed: {candidate_id}")
                    value.update(
                        {
                            "fit_status": "trained",
                            "artifact_sha256": artifact_sha256,
                            "model_diagnostics": _artifact_diagnostics(
                                artifact,
                                dataset,
                                splits,
                            ),
                        }
                    )
                results.append(value)
    payload: dict[str, object] = {
        "schema_version": "action-value-discovery-diagnostic-v1",
        "round": int(design["round"]),
        "artifact_class": "consumed_selection_diagnostic_no_trading_authority",
        "design_sha256": design["design_sha256"],
        "source_report_sha256": _file_sha256(report_path),
        "corpus_certificate_sha256": certificate_sha256,
        "terminal_holdout_accessed": False,
        "trading_authority": False,
        "profitability_claim": False,
        "top_score_rows_are_overlap_unsuppressed": True,
        "candidates": results,
    }
    payload["diagnostic_sha256"] = _canonical_sha256(payload)
    write_json_atomic(output_path, payload, indent=2, sort_keys=True)
    return payload


def main() -> int:
    args = _arguments()
    try:
        payload = diagnose(
            args.design,
            args.evidence_root,
            args.warehouse,
            args.cache_root,
            args.output,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"diagnose-action-value-discovery failed: {type(exc).__name__}: {exc}")
        return 2
    print(
        "diagnose-action-value-discovery: "
        f"round={payload['round']} candidates={len(payload['candidates'])} "
        f"sha256={payload['diagnostic_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
