"""Resumable exact-BBO confirmation for a frozen tape/depth candidate."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .microstructure_warehouse import MicrostructureWarehouse
from .storage import write_json_atomic
from .tape_depth_execution import (
    TapeDepthExecutionAssumptions,
    evaluate_tape_depth_taker_execution,
    load_tape_depth_execution_confirmation_design,
)
from .tape_depth_features import (
    build_tape_depth_forecast_dataset,
    tape_depth_dataset_fingerprint,
)
from .tape_depth_model import (
    TAPE_DEPTH_MODEL_SCHEMA_VERSION,
    save_tape_depth_model_artifact,
    score_tape_depth_evaluation,
    train_tape_depth_forecaster,
)
from .tape_depth_prequential import write_tape_depth_predictions


TAPE_DEPTH_EXECUTION_CONFIRMATION_SCHEMA_VERSION = (
    "tape-depth-execution-confirmation-v1"
)
TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION = "tape-depth-execution-period-v1"
TAPE_DEPTH_EXECUTION_PLAN_SCHEMA_VERSION = "tape-depth-execution-plan-v1"
_DAY_MS = 86_400_000
_Progress = Callable[[str, int, int], None]


def _canonical_sha256(payload: Mapping[str, object], *, omit: str | None = None) -> str:
    canonical = dict(payload)
    if omit is not None:
        canonical.pop(omit, None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _with_fingerprint(payload: Mapping[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = _canonical_sha256(result, omit=field)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"confirmation evidence is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _utc_period_bounds(period: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(str(period), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"invalid confirmation period: {period}") from exc
    start_ms = int(parsed.timestamp() * 1_000)
    return start_ms, start_ms + _DAY_MS


def _dataset_bounds_for_period(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    period: str,
    horizon_seconds: int,
    total_latency_ms: int,
) -> tuple[int, int, dict[str, int]]:
    day_start_ms, day_end_ms = _utc_period_bounds(period)
    row = warehouse.connect().execute(
        """
        SELECT min(second_ms), max(second_ms), count(*)
        FROM current_trade_1s
        WHERE symbol = ? AND second_ms >= ? AND second_ms < ?
        """,
        [symbol, day_start_ms, day_end_ms],
    ).fetchone()
    if row is None or row[0] is None or row[1] is None or int(row[2]) <= 0:
        raise ValueError(f"confirmation period {period} has no target trade data")
    first_second_ms = int(row[0])
    last_second_ms = int(row[1])
    entry_delay_seconds = max(1, int(math.ceil(total_latency_ms / 1_000.0)))
    target_offset_seconds = entry_delay_seconds + int(horizon_seconds)
    decision_start_ms = first_second_ms + 901_000
    decision_end_ms = last_second_ms - target_offset_seconds * 1_000 + 1_000
    if (
        not day_start_ms <= first_second_ms < day_end_ms
        or not day_start_ms <= last_second_ms < day_end_ms
        or decision_start_ms > decision_end_ms
        or decision_end_ms >= day_end_ms
    ):
        raise ValueError(f"confirmation period {period} lacks a complete causal window")
    return decision_start_ms, decision_end_ms, {
        "archive_day_start_ms": day_start_ms,
        "archive_day_end_exclusive_ms": day_end_ms,
        "first_trade_second_ms": first_second_ms,
        "last_trade_second_ms": last_second_ms,
        "trade_second_rows": int(row[2]),
        "decision_start_ms": decision_start_ms,
        "decision_end_ms": decision_end_ms,
    }


def _validate_design_runtime(design: Mapping[str, object]) -> None:
    candidate = design.get("candidate")
    execution = design.get("execution")
    if not isinstance(candidate, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("execution confirmation design sections are invalid")
    if (
        candidate.get("model_schema_version") != TAPE_DEPTH_MODEL_SCHEMA_VERSION
        or candidate.get("symbol") != "BTCUSDT"
        or candidate.get("peer_symbols") != ["ETHUSDT", "SOLUSDT"]
        or candidate.get("reproducible_backend") is not True
        or float(execution.get("leverage", 0.0)) != 1.0
    ):
        raise ValueError("execution confirmation runtime differs from the frozen design")


def _validate_checkpoint(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    period: str,
    design_sha256: str,
) -> dict[str, object]:
    payload = _read_json_object(checkpoint_path, f"confirmation checkpoint {period}")
    expected_fingerprint = str(payload.get("period_fingerprint") or "")
    if (
        payload.get("schema_version") != TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION
        or payload.get("period") != period
        or payload.get("design_sha256") != design_sha256
        or payload.get("status") != "complete"
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or expected_fingerprint
        != _canonical_sha256(payload, omit="period_fingerprint")
    ):
        raise ValueError(f"confirmation checkpoint {period} failed integrity validation")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"confirmation checkpoint {period} lacks evidence bindings")
    expected_paths = {
        "model": output_dir / "periods" / period / "model.json",
        "predictions": output_dir / "periods" / period / "predictions.csv.gz",
    }
    for label, path in expected_paths.items():
        if evidence.get(f"{label}_path") != str(path.relative_to(output_dir)).replace("\\", "/"):
            raise ValueError(f"confirmation checkpoint {period} {label} path differs")
        if evidence.get(f"{label}_sha256") != _file_sha256(path):
            raise ValueError(f"confirmation checkpoint {period} {label} hash differs")
    return payload


def _run_period_confirmation(
    warehouse: MicrostructureWarehouse,
    *,
    design: Mapping[str, object],
    design_sha256: str,
    period: str,
    output_dir: Path,
    progress: _Progress | None,
    period_index: int,
    period_count: int,
) -> dict[str, object]:
    candidate = dict(design["candidate"])  # type: ignore[arg-type]
    execution = dict(design["execution"])  # type: ignore[arg-type]
    symbol = str(candidate["symbol"])
    horizon_seconds = int(candidate["horizon_seconds"])
    total_latency_ms = int(candidate["total_latency_ms"])
    start_ms, end_ms, coverage = _dataset_bounds_for_period(
        warehouse,
        symbol=symbol,
        period=period,
        horizon_seconds=horizon_seconds,
        total_latency_ms=total_latency_ms,
    )
    if progress:
        progress(f"{period}:dataset", period_index, period_count)
    dataset = build_tape_depth_forecast_dataset(
        warehouse,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        horizon_seconds=horizon_seconds,
        total_latency_ms=total_latency_ms,
        decision_cadence_seconds=int(candidate["decision_cadence_seconds"]),
        maximum_depth_age_ms=int(candidate["maximum_depth_age_ms"]),
        maximum_rows=100_000,
    )
    if (
        dataset.rows <= 0
        or np.any(dataset.decision_time_ms < start_ms)
        or np.any(dataset.decision_time_ms > end_ms)
    ):
        raise RuntimeError(f"confirmation period {period} emitted invalid dataset rows")

    def training_progress(phase: str, _completed: int, _total: int) -> None:
        if progress:
            progress(f"{period}:train:{phase}", period_index, period_count)

    artifact = train_tape_depth_forecaster(
        dataset,
        risk_level=str(candidate["risk_level"]),
        model_profile=str(candidate["model_profile"]),
        feature_set=str(candidate["feature_set"]),
        compute_backend=str(candidate["compute_backend"]),
        seed=int(candidate["seed"]),
        minimum_segment_rows=int(candidate["minimum_segment_rows"]),
        progress=training_progress,
    )
    predictions = score_tape_depth_evaluation(artifact, dataset)
    assumptions = TapeDepthExecutionAssumptions(
        taker_fee_bps_per_side=float(execution["taker_fee_bps_per_side"]),
        additional_slippage_bps_per_side=float(
            execution["additional_slippage_bps_per_side"]
        ),
        max_quote_age_ms=int(execution["max_quote_age_ms"]),
        reference_order_notional_quote=float(
            execution["reference_order_notional_quote"]
        ),
        max_l1_participation=float(execution["max_l1_participation"]),
        suppress_overlapping_positions=execution["suppress_overlapping_positions"],
    )
    execution_report, execution_rows = evaluate_tape_depth_taker_execution(
        warehouse,
        symbol=symbol,
        predictions=predictions,
        assumptions=assumptions,
    )

    period_dir = output_dir / "periods" / period
    model_path = period_dir / "model.json"
    predictions_path = period_dir / "predictions.csv.gz"
    save_tape_depth_model_artifact(artifact, model_path)
    predictions_sha256 = write_tape_depth_predictions(predictions, predictions_path)
    model_sha256 = _file_sha256(model_path)
    source_evidence_sha256 = _canonical_sha256(dict(dataset.source_evidence))
    payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION,
        "status": "complete",
        "period": period,
        "design_sha256": design_sha256,
        "selection_contaminated": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "liquidation_events": 0,
        "coverage": coverage,
        "dataset": {
            "rows": dataset.rows,
            "fingerprint": tape_depth_dataset_fingerprint(dataset),
            "source_evidence_sha256": source_evidence_sha256,
        },
        "forecast": {
            "status": artifact.status,
            "rejection_reasons": list(artifact.rejection_reasons),
            "backend_kind": artifact.backend_kind,
            "backend_device": artifact.backend_device,
            "metrics": asdict(artifact.evaluation_metrics),
            "signal_policy": asdict(artifact.signal_policy),
        },
        "execution": execution_report.asdict(),
        "execution_rows": [asdict(row) for row in execution_rows],
        "evidence": {
            "model_path": str(model_path.relative_to(output_dir)).replace("\\", "/"),
            "model_sha256": model_sha256,
            "predictions_path": str(predictions_path.relative_to(output_dir)).replace(
                "\\", "/"
            ),
            "predictions_sha256": predictions_sha256,
            "prediction_fingerprint": predictions.fingerprint(),
        },
        "completed_at": datetime.now(tz=UTC).isoformat(),
    }
    checkpoint = _with_fingerprint(payload, "period_fingerprint")
    write_json_atomic(period_dir / "report.json", checkpoint, indent=2, sort_keys=True)
    return checkpoint


def aggregate_tape_depth_execution_confirmation(
    design: Mapping[str, object],
    *,
    design_sha256: str,
    period_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply only the precommitted acceptance gates to complete period reports."""

    periods = [str(value) for value in design["confirmation_periods"]]  # type: ignore[index]
    acceptance = dict(design["acceptance"])  # type: ignore[arg-type]
    if [str(report.get("period")) for report in period_reports] != periods:
        raise ValueError("execution confirmation reports differ from the frozen periods")
    for report in period_reports:
        if (
            report.get("schema_version") != TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION
            or report.get("status") != "complete"
            or report.get("design_sha256") != design_sha256
            or report.get("selection_contaminated") is not False
            or report.get("trading_authority") is not False
            or report.get("execution_claim") is not False
            or report.get("profitability_claim") is not False
            or report.get("period_fingerprint")
            != _canonical_sha256(report, omit="period_fingerprint")
        ):
            raise ValueError("execution confirmation period report is invalid")

    completed_periods = len(period_reports)
    forecast_candidate_periods = 0
    positive_mean_net_periods = 0
    combined_executable_rows = 0
    positive_net_rows = 0.0
    net_return_sum_bps = 0.0
    quote_rejection_rows = 0
    liquidation_events = 0
    period_summaries: list[dict[str, object]] = []
    for report in period_reports:
        forecast = report.get("forecast")
        execution = report.get("execution")
        execution_rows = report.get("execution_rows")
        if (
            not isinstance(forecast, Mapping)
            or not isinstance(execution, Mapping)
            or not isinstance(execution_rows, list)
        ):
            raise ValueError("execution confirmation period sections are invalid")
        metrics = execution.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("execution confirmation period metrics are invalid")
        executable_net: list[float] = []
        derived_quote_rejections = 0
        derived_participation_rejections = 0
        for row in execution_rows:
            if not isinstance(row, Mapping):
                raise ValueError("execution confirmation row is invalid")
            status = row.get("status")
            reason = str(row.get("rejection_reason") or "")
            if status == "executable":
                value = float(row.get("net_return_bps"))
                if reason or not math.isfinite(value):
                    raise ValueError("execution confirmation executable row is invalid")
                executable_net.append(value)
            elif status == "rejected":
                if reason == "l1_participation_exceeded":
                    derived_participation_rejections += 1
                elif reason:
                    derived_quote_rejections += 1
                else:
                    raise ValueError("execution confirmation rejected row lacks a reason")
            else:
                raise ValueError("execution confirmation row status is invalid")
        executable_rows = len(executable_net)
        positive_rows = sum(value > 0.0 for value in executable_net)
        mean_net_return_bps = (
            float(np.mean(np.asarray(executable_net, dtype=np.float64)))
            if executable_net
            else 0.0
        )
        positive_net_rate = positive_rows / executable_rows if executable_rows else 0.0
        scheduled_rows = int(metrics["scheduled_signal_rows"])
        selected_rows = int(metrics["selected_signal_rows"])
        summary_counts = (
            int(metrics["executable_rows"]),
            int(metrics["rejected_quote_rows"]),
            int(metrics["rejected_participation_rows"]),
        )
        if (
            scheduled_rows != len(execution_rows)
            or selected_rows < scheduled_rows
            or int(metrics["overlap_suppressed_rows"])
            != selected_rows - scheduled_rows
            or summary_counts
            != (
                executable_rows,
                derived_quote_rejections,
                derived_participation_rejections,
            )
            or not math.isclose(
                float(metrics["mean_net_return_bps"]),
                mean_net_return_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(metrics["positive_net_rate"]),
                positive_net_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isfinite(mean_net_return_bps)
            or not math.isfinite(positive_net_rate)
            or not 0.0 <= positive_net_rate <= 1.0
        ):
            raise ValueError("execution confirmation row evidence differs from its metrics")
        period_liquidations = int(report.get("liquidation_events", -1))
        if period_liquidations < 0:
            raise ValueError("execution confirmation liquidation evidence is invalid")
        forecast_candidate_periods += forecast.get("status") == "research_candidate"
        positive_mean_net_periods += executable_rows > 0 and mean_net_return_bps > 0.0
        combined_executable_rows += executable_rows
        net_return_sum_bps += executable_rows * mean_net_return_bps
        positive_net_rows += positive_rows
        quote_rejection_rows += derived_quote_rejections
        liquidation_events += period_liquidations
        period_summaries.append(
            {
                "period": report["period"],
                "period_fingerprint": report["period_fingerprint"],
                "forecast_status": forecast["status"],
                "execution_status": execution["status"],
                "selected_signal_rows": selected_rows,
                "executable_rows": executable_rows,
                "rejected_quote_rows": int(metrics["rejected_quote_rows"]),
                "rejected_participation_rows": int(
                    metrics["rejected_participation_rows"]
                ),
                "mean_net_return_bps": mean_net_return_bps,
                "positive_net_rate": positive_net_rate,
            }
        )
    combined_mean_net_return_bps = (
        net_return_sum_bps / combined_executable_rows
        if combined_executable_rows
        else 0.0
    )
    combined_positive_net_rate = (
        positive_net_rows / combined_executable_rows
        if combined_executable_rows
        else 0.0
    )
    actual = {
        "completed_periods": completed_periods,
        "forecast_candidate_periods": forecast_candidate_periods,
        "positive_mean_net_periods": positive_mean_net_periods,
        "combined_executable_rows": combined_executable_rows,
        "combined_mean_net_return_bps": combined_mean_net_return_bps,
        "combined_positive_net_rate": combined_positive_net_rate,
        "quote_rejection_rows": quote_rejection_rows,
        "liquidation_events": liquidation_events,
    }
    gates = {
        "completed_periods": completed_periods
        == int(acceptance["required_completed_periods"]),
        "forecast_candidate_periods": forecast_candidate_periods
        >= int(acceptance["minimum_forecast_candidate_periods"]),
        "positive_mean_net_periods": positive_mean_net_periods
        >= int(acceptance["minimum_positive_mean_net_periods"]),
        "combined_executable_rows": combined_executable_rows
        >= int(acceptance["minimum_combined_executable_rows"]),
        "combined_mean_net_return_bps": combined_mean_net_return_bps
        > float(acceptance["minimum_combined_mean_net_return_bps_exclusive"]),
        "combined_positive_net_rate": combined_positive_net_rate
        > float(acceptance["minimum_combined_positive_net_rate_exclusive"]),
        "quote_rejection_rows": quote_rejection_rows
        <= int(acceptance["maximum_quote_rejection_rows"]),
        "liquidation_events": liquidation_events
        <= int(acceptance["maximum_liquidations"]),
    }
    rejection_reasons = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_EXECUTION_CONFIRMATION_SCHEMA_VERSION,
        "status": (
            "confirmed_after_cost_candidate" if not rejection_reasons else "rejected"
        ),
        "rejection_reasons": rejection_reasons,
        "design_sha256": design_sha256,
        "availability_sha256": design["availability_sha256"],
        "periods": period_summaries,
        "acceptance": acceptance,
        "actual": actual,
        "gates": gates,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "terminal_holdout": False,
        "truth_basis": "official_checksums_plus_causal_fp64_model_plus_exact_100ms_bbo",
        "completed_at": datetime.now(tz=UTC).isoformat(),
    }
    return _with_fingerprint(payload, "confirmation_fingerprint")


def run_tape_depth_execution_confirmation(
    warehouse: MicrostructureWarehouse,
    *,
    design_path: str | Path,
    availability_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    progress: _Progress | None = None,
) -> dict[str, object]:
    """Run or resume every frozen period without overwriting observed outcomes."""

    design_file = Path(design_path).resolve()
    availability_file = Path(availability_path).resolve()
    destination = Path(output_dir).resolve()
    design, design_sha256 = load_tape_depth_execution_confirmation_design(
        design_file,
        availability_path=availability_file,
    )
    _validate_design_runtime(design)
    periods = [str(value) for value in design["confirmation_periods"]]
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "plan.json"
    plan: dict[str, object] = {
        "schema_version": TAPE_DEPTH_EXECUTION_PLAN_SCHEMA_VERSION,
        "design_sha256": design_sha256,
        "design_file_sha256": _file_sha256(design_file),
        "availability_sha256": _file_sha256(availability_file),
        "periods": periods,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
    }
    expected_plan = _with_fingerprint(plan, "plan_fingerprint")
    if plan_path.exists():
        existing_plan = _read_json_object(plan_path, "execution confirmation plan")
        if existing_plan != expected_plan:
            raise ValueError("execution confirmation plan differs from the frozen design")
    else:
        write_json_atomic(plan_path, expected_plan, indent=2, sort_keys=True)

    report_path = destination / "report.json"
    period_reports: list[dict[str, object]] = []
    for index, period in enumerate(periods, start=1):
        checkpoint_path = destination / "periods" / period / "report.json"
        if checkpoint_path.exists():
            if not resume:
                raise ValueError(
                    f"confirmation checkpoint {period} exists; use resume to preserve it"
                )
            checkpoint = _validate_checkpoint(
                checkpoint_path,
                output_dir=destination,
                period=period,
                design_sha256=design_sha256,
            )
            if progress:
                progress(f"{period}:resumed", index, len(periods))
        else:
            if progress:
                progress(f"{period}:started", index, len(periods))
            checkpoint = _run_period_confirmation(
                warehouse,
                design=design,
                design_sha256=design_sha256,
                period=period,
                output_dir=destination,
                progress=progress,
                period_index=index,
                period_count=len(periods),
            )
            if progress:
                progress(f"{period}:complete", index, len(periods))
        period_reports.append(checkpoint)

    report = aggregate_tape_depth_execution_confirmation(
        design,
        design_sha256=design_sha256,
        period_reports=period_reports,
    )
    if report_path.exists():
        if not resume:
            raise ValueError("execution confirmation report exists; use resume to preserve it")
        existing_report = _read_json_object(report_path, "execution confirmation report")
        if (
            existing_report.get("confirmation_fingerprint")
            != _canonical_sha256(existing_report, omit="confirmation_fingerprint")
            or existing_report.get("design_sha256") != design_sha256
            or [
                item.get("period_fingerprint")
                for item in existing_report.get("periods", [])
                if isinstance(item, Mapping)
            ]
            != [item["period_fingerprint"] for item in report["periods"]]  # type: ignore[index]
        ):
            raise ValueError("execution confirmation report failed integrity validation")
        return existing_report
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    return report


__all__ = [
    "TAPE_DEPTH_EXECUTION_CONFIRMATION_SCHEMA_VERSION",
    "TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION",
    "TAPE_DEPTH_EXECUTION_PLAN_SCHEMA_VERSION",
    "aggregate_tape_depth_execution_confirmation",
    "run_tape_depth_execution_confirmation",
]
