"""Checkpointed orchestration for precommitted tape/depth screening designs."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .model_experiment import validate_experiment_design_payload
from .storage import write_json_atomic
from .tape_depth_comparison import load_and_select_tape_depth_reports
from .tape_depth_prequential import (
    run_tape_depth_prequential,
    verify_tape_depth_prequential_report,
)


TAPE_DEPTH_STUDY_SCHEMA_VERSION = "tape-depth-screening-study-v1"
_MAX_DESIGN_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
StudyProgress = Callable[[str, int, int], None]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_object(
    path: Path,
    *,
    description: str,
    maximum_bytes: int,
) -> tuple[dict[str, object], bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            raise ValueError(f"{description} exceeds its size limit: {path}")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload, raw


def _candidate_options(candidate: Mapping[str, object]) -> dict[str, object]:
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("experiment candidate parameters are invalid")
    return {
        "risk_level": str(parameters["risk_level"]),
        "horizon_seconds": int(parameters["horizon_seconds"]),
        "decision_cadence_seconds": int(parameters["decision_cadence_seconds"]),
        "maximum_depth_age_ms": int(parameters["maximum_depth_age_ms"]),
        "model_profile": str(parameters["model_profile"]),
        "feature_set": str(parameters["feature_set"]),
    }


def _expected_report_config(
    *,
    symbols: Sequence[str],
    candidate_options: Mapping[str, object],
    training_window_days: int,
    tuning_window_days: int,
    calibration_window_days: int,
    evaluation_window_days: int,
    total_latency_ms: int,
    maximum_rows: int,
    maximum_cached_rows: int,
    dataset_cache: bool,
    max_folds: int,
    compute_backend: str,
    minimum_segment_rows: int,
) -> dict[str, object]:
    return {
        "symbols": list(symbols),
        "training_window_days": int(training_window_days),
        "tuning_window_days": int(tuning_window_days),
        "calibration_window_days": int(calibration_window_days),
        "evaluation_window_days": int(evaluation_window_days),
        "horizon_seconds": int(candidate_options["horizon_seconds"]),
        "total_latency_ms": int(total_latency_ms),
        "decision_cadence_seconds": int(
            candidate_options["decision_cadence_seconds"]
        ),
        "maximum_depth_age_ms": int(candidate_options["maximum_depth_age_ms"]),
        "maximum_rows": int(maximum_rows),
        "maximum_cached_rows": int(maximum_cached_rows),
        "dataset_cache": bool(dataset_cache),
        "study_stage": "screening",
        "fold_start": 0,
        "max_folds": int(max_folds),
        "risk_level": str(candidate_options["risk_level"]),
        "model_profile": str(candidate_options["model_profile"]),
        "feature_set": str(candidate_options["feature_set"]),
        "compute_backend": str(compute_backend),
        "minimum_segment_rows": int(minimum_segment_rows),
        "selection_lock_sha256": None,
    }


def _load_verified_resume_report(
    path: Path,
    *,
    expected_config: Mapping[str, object],
) -> dict[str, object]:
    report, _raw = _read_json_object(
        path,
        description="screening resume report",
        maximum_bytes=_MAX_REPORT_BYTES,
    )
    verify_tape_depth_prequential_report(path, report)
    if report.get("config") != dict(expected_config):
        raise ValueError(f"screening resume report configuration changed: {path}")
    if (
        report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
        or int(report.get("completed_folds", -1)) != int(report.get("total_folds", -2))
    ):
        raise ValueError(f"screening resume report is incomplete or unsafe: {path}")
    return report


def run_tape_depth_screening_study(
    warehouse: object,
    *,
    symbols: Sequence[str],
    design_path: str | Path,
    output_dir: str | Path,
    training_window_days: int = 730,
    tuning_window_days: int = 30,
    calibration_window_days: int = 30,
    evaluation_window_days: int = 90,
    total_latency_ms: int = 750,
    maximum_rows: int = 5_000_000,
    maximum_cached_rows: int = 15_000_000,
    dataset_cache: bool = True,
    max_folds: int = 4,
    compute_backend: str = "auto",
    minimum_segment_rows: int = 10_000,
    resume: bool = False,
    plan_only: bool = False,
    progress: StudyProgress | None = None,
) -> dict[str, object]:
    """Run every design candidate without accessing the confirmation suffix."""

    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not normalized_symbols or len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("screening study symbols are empty or duplicated")
    folds = int(max_folds)
    if folds not in {4, 6, 8, 10}:
        raise ValueError("screening study max_folds must be 4, 6, 8, or 10")
    resolved_design = Path(design_path).resolve()
    design_payload, design_raw = _read_json_object(
        resolved_design,
        description="experiment design",
        maximum_bytes=_MAX_DESIGN_BYTES,
    )
    design = validate_experiment_design_payload(design_payload)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict(candidate) for candidate in design["candidates"])  # type: ignore[arg-type]
    candidate_plan: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_plan.append(
            {
                "candidate_id": candidate_id,
                "source": candidate["source"],
                "design_index": candidate["design_index"],
                "parameters": candidate["parameters"],
                "output_dir": str(destination / "candidates" / candidate_id),
            }
        )
    base_payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_STUDY_SCHEMA_VERSION,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "terminal_holdout_consumed": False,
        "symbols": list(normalized_symbols),
        "design": {
            "path": str(resolved_design),
            "file_sha256": _sha256_bytes(design_raw),
            "design_sha256": design["design_sha256"],
            "trial_burden": design["trial_burden"],
        },
        "max_folds": folds,
        "candidate_plan": candidate_plan,
    }
    if plan_only:
        payload = {
            **base_payload,
            "status": "planned_research_only",
            "completed_candidates": 0,
            "selection": None,
        }
        write_json_atomic(
            destination / "study-plan.json",
            payload,
            indent=2,
            sort_keys=True,
        )
        return payload

    status_path = destination / "study-status.json"
    started_at = datetime.now(tz=UTC).isoformat()
    report_paths: list[Path] = []
    candidate_results: list[dict[str, object]] = []

    def persist(state: str, **extra: object) -> None:
        write_json_atomic(
            status_path,
            {
                **base_payload,
                "state": state,
                "started_at": started_at,
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "completed_candidates": len(candidate_results),
                "candidate_results": candidate_results,
                **extra,
            },
            indent=2,
            sort_keys=True,
        )

    persist("running")
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        options = _candidate_options(candidate)
        candidate_dir = destination / "candidates" / candidate_id
        report_path = candidate_dir / "report.json"
        expected_config = _expected_report_config(
            symbols=normalized_symbols,
            candidate_options=options,
            training_window_days=training_window_days,
            tuning_window_days=tuning_window_days,
            calibration_window_days=calibration_window_days,
            evaluation_window_days=evaluation_window_days,
            total_latency_ms=total_latency_ms,
            maximum_rows=maximum_rows,
            maximum_cached_rows=maximum_cached_rows,
            dataset_cache=dataset_cache,
            max_folds=folds,
            compute_backend=compute_backend,
            minimum_segment_rows=minimum_segment_rows,
        )
        if progress is not None:
            progress("candidate_started", candidate_index - 1, len(candidates))
        reused = False
        if resume and report_path.is_file():
            report = _load_verified_resume_report(
                report_path,
                expected_config=expected_config,
            )
            reused = True
        else:
            report = run_tape_depth_prequential(
                warehouse,  # type: ignore[arg-type]
                symbols=normalized_symbols,
                output_dir=candidate_dir,
                training_window_days=training_window_days,
                tuning_window_days=tuning_window_days,
                calibration_window_days=calibration_window_days,
                evaluation_window_days=evaluation_window_days,
                horizon_seconds=int(options["horizon_seconds"]),
                total_latency_ms=total_latency_ms,
                decision_cadence_seconds=int(options["decision_cadence_seconds"]),
                maximum_depth_age_ms=int(options["maximum_depth_age_ms"]),
                maximum_rows=maximum_rows,
                maximum_cached_rows=maximum_cached_rows,
                dataset_cache=dataset_cache,
                study_stage="screening",
                max_folds=folds,
                risk_level=str(options["risk_level"]),
                model_profile=str(options["model_profile"]),
                feature_set=str(options["feature_set"]),
                compute_backend=compute_backend,
                minimum_segment_rows=minimum_segment_rows,
                resume=resume,
                progress=(
                    None
                    if progress is None
                    else lambda phase, completed, total: progress(
                        f"candidate:{candidate_id[:12]}:{phase}",
                        completed,
                        total,
                    )
                ),
            )
            verify_tape_depth_prequential_report(report_path, report)
            if report.get("config") != expected_config:
                raise ValueError("completed screening report differs from its design candidate")
        report_paths.append(report_path)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "report_path": str(report_path),
                "status": report.get("status"),
                "reused": reused,
            }
        )
        persist("running", current_candidate_id=candidate_id)
        if progress is not None:
            progress("candidate_complete", candidate_index, len(candidates))

    selection_path = destination / "selection.json"
    selection = load_and_select_tape_depth_reports(
        report_paths,
        output=selection_path,
        design_path=resolved_design,
    )
    payload = {
        **base_payload,
        "status": selection["status"],
        "completed_candidates": len(candidate_results),
        "candidate_results": candidate_results,
        "selection_path": str(selection_path),
        "selection": selection,
    }
    write_json_atomic(
        destination / "study-report.json",
        payload,
        indent=2,
        sort_keys=True,
    )
    persist("complete", selection_path=str(selection_path), status=selection["status"])
    return payload


__all__ = [
    "TAPE_DEPTH_STUDY_SCHEMA_VERSION",
    "run_tape_depth_screening_study",
]
