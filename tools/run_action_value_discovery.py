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
    MicrostructureClassSupportError,
    _MINIMUM_CALIBRATION_CLASS_ROWS,
    _MINIMUM_EARLY_STOP_CLASS_ROWS,
    _MINIMUM_TRAIN_CLASS_ROWS,
    _purged_split,
    save_microstructure_model_artifact,
    train_microstructure_action_value_model,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.storage import write_json_atomic


DESIGN_SCHEMA_VERSION = "action-value-discovery-design-v2"
_SUPPORTED_DESIGN_SCHEMA_VERSIONS = frozenset(
    {"action-value-v15-discovery-design-v1", DESIGN_SCHEMA_VERSION}
)
_SUPPORTED_MODEL_SCHEMA_VERSIONS = frozenset(
    {"microstructure-action-value-v15", MICROSTRUCTURE_MODEL_SCHEMA_VERSION}
)
_SUPPORTED_FEATURE_VERSIONS = frozenset(
    {"l1-tape-causal-v6", MICROSTRUCTURE_FEATURE_VERSION}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/microstructure_features.py",
    "src/simple_ai_trading/microstructure_model.py",
    "src/simple_ai_trading/microstructure_runtime.py",
    "src/simple_ai_trading/microstructure_warehouse.py",
    "tools/run_action_value_discovery.py",
)
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


def _parse_utc_date(value: object, *, label: str):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must be a YYYY-MM-DD UTC date") from exc


def _utc_date_strings(start, end) -> tuple[str, ...]:
    values: list[str] = []
    cursor = start
    while cursor <= end:
        values.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(values)


def _load_consumed_registry(path: Path, expected_sha256: str) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("action-value consumed-period registry is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("action-value consumed-period registry must be an object")
    claimed_sha256 = str(payload.get("registry_sha256") or "")
    canonical = dict(payload)
    canonical.pop("registry_sha256", None)
    if (
        payload.get("schema_version") != "action-value-consumed-periods-v1"
        or claimed_sha256 != _canonical_sha256(canonical)
        or claimed_sha256 != expected_sha256
    ):
        raise ValueError("action-value consumed-period registry binding is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("action-value consumed-period registry has no records")
    consumed: set[str] = set()
    prior_round = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("action-value consumed-period record is invalid")
        round_number = int(record.get("round") or 0)
        windows = record.get("windows")
        if (
            round_number <= prior_round
            or record.get("status") != "consumed"
            or not isinstance(windows, list)
            or not windows
        ):
            raise ValueError("action-value consumed-period record is inconsistent")
        prior_round = round_number
        for window in windows:
            if not isinstance(window, Mapping):
                raise ValueError("action-value consumed-period window is invalid")
            first = _parse_utc_date(
                window.get("start_date"),
                label="consumed start_date",
            )
            last = _parse_utc_date(
                window.get("end_date"),
                label="consumed end_date",
            )
            if first > last:
                raise ValueError("action-value consumed-period window is reversed")
            consumed.update(_utc_date_strings(first, last))
    return consumed


def _validate_expected_split_days(
    data: Mapping[str, object],
    *,
    start,
    end,
) -> dict[str, tuple[str, ...]]:
    raw = data.get("expected_split_days")
    names = ("train", "tuning", "policy", "selection", "terminal")
    if not isinstance(raw, Mapping) or set(raw) != set(names):
        raise ValueError("action-value v2 design split calendar is incomplete")
    output: dict[str, tuple[str, ...]] = {}
    combined: list[str] = []
    for name in names:
        value = raw[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"action-value {name} split calendar is invalid")
        first = _parse_utc_date(value.get("start_date"), label=f"{name} start_date")
        last = _parse_utc_date(value.get("end_date"), label=f"{name} end_date")
        periods = _utc_date_strings(first, last) if first <= last else ()
        if not periods or int(value.get("day_count") or 0) != len(periods):
            raise ValueError(f"action-value {name} split day count is invalid")
        output[name] = periods
        combined.extend(periods)
    if tuple(combined) != _utc_date_strings(start, end):
        raise ValueError("action-value split calendar does not exactly partition the window")
    return output


def _validate_current_implementation(change_control: object) -> None:
    if not isinstance(change_control, Mapping):
        raise ValueError("action-value v2 change control is missing")
    commit = str(change_control.get("implementation_commit") or "")
    expected = change_control.get("implementation_files_sha256")
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(expected, Mapping)
        or set(expected) != set(_IMPLEMENTATION_FILES)
    ):
        raise ValueError("action-value implementation binding is incomplete")
    for relative_path in _IMPLEMENTATION_FILES:
        claimed = str(expected[relative_path])
        path = _REPOSITORY_ROOT / relative_path
        if len(claimed) != 64 or claimed != _file_sha256(path):
            raise ValueError(
                f"action-value implementation file drifted from design: {relative_path}"
            )


def load_discovery_design(
    path: str | Path,
    *,
    require_current: bool = False,
) -> dict[str, object]:
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
    if payload.get("schema_version") not in _SUPPORTED_DESIGN_SCHEMA_VERSIONS:
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
    if payload.get("schema_version") == DESIGN_SCHEMA_VERSION:
        registry_name = str(data.get("consumed_registry") or "")
        if not registry_name or Path(registry_name).name != registry_name:
            raise ValueError("action-value v2 consumed registry must be an adjacent file")
        consumed = _load_consumed_registry(
            target.parent / registry_name,
            str(data.get("consumed_registry_sha256") or ""),
        )
        expected_days = _validate_expected_split_days(data, start=start, end=end)
        preselection = set(
            expected_days["train"]
            + expected_days["tuning"]
            + expected_days["policy"]
        )
        window_consumed = consumed.intersection(_utc_date_strings(start, end))
        if not window_consumed.issubset(preselection):
            raise ValueError(
                "action-value selection or terminal calendar intersects consumed evidence"
            )
        if (
            data.get("selection_dates_previously_untouched") is not True
            or data.get("terminal_dates_previously_untouched") is not True
            or data.get("all_window_dates_consumed_after_run") is not True
        ):
            raise ValueError("action-value v2 date-consumption declarations are invalid")
    else:
        consumed = {str(value) for value in data.get("excluded_consumed_dates") or ()}
        cursor = start
        while cursor <= end:
            if cursor.isoformat() in consumed:
                raise ValueError("action-value discovery window intersects consumed evidence")
            cursor += timedelta(days=1)

    if (
        training.get("model_schema_version") not in _SUPPORTED_MODEL_SCHEMA_VERSIONS
        or training.get("feature_version") not in _SUPPORTED_FEATURE_VERSIONS
        or training.get("model_family") != "side_specific_hurdle_expected_value"
        or training.get("reproducible_backend") is not True
        or training.get("evaluate_terminal") is not False
    ):
        raise ValueError("action-value discovery training contract is invalid")
    if require_current and (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or training.get("model_schema_version") != MICROSTRUCTURE_MODEL_SCHEMA_VERSION
        or training.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION
    ):
        raise ValueError("action-value execution requires the current design and model schemas")
    if require_current:
        _validate_current_implementation(payload.get("change_control"))
    if payload.get("schema_version") == DESIGN_SCHEMA_VERSION:
        support_minimums = training.get("class_support_minimums")
        expected_minimums = {
            "train_each_class": _MINIMUM_TRAIN_CLASS_ROWS,
            "early_stop_each_class": _MINIMUM_EARLY_STOP_CLASS_ROWS,
            "calibration_each_class": _MINIMUM_CALIBRATION_CLASS_ROWS,
        }
        if support_minimums != expected_minimums:
            raise ValueError("action-value class-support design drifted from the learner")
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


def _split_calendar_evidence(
    dataset,
    design: Mapping[str, object],
) -> dict[str, object] | None:
    data = design["data"]
    assert isinstance(data, Mapping)
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        return None
    start = _parse_utc_date(data["start_date"], label="start_date")
    end = _parse_utc_date(data["end_date"], label="end_date")
    expected = _validate_expected_split_days(data, start=start, end=end)
    splits, split = _purged_split(dataset)
    actual: dict[str, tuple[str, ...]] = {}
    for name in ("train", "tuning", "policy", "selection", "terminal"):
        day_ids = np.unique(dataset.decision_time_ms[splits[name]] // 86_400_000)
        actual[name] = tuple(
            datetime.fromtimestamp(int(day_id) * 86_400, tz=UTC).date().isoformat()
            for day_id in day_ids
        )
        if actual[name] != expected[name]:
            raise ValueError(
                f"action-value actual {name} calendar drifted from the precommit"
            )
    return {
        "status": "pass",
        "target_labels_accessed": False,
        "purge_ms": split.purge_ms,
        "segments": {
            name: {
                "start_date": dates[0],
                "end_date": dates[-1],
                "day_count": len(dates),
                "rows": int(len(splits[name])),
            }
            for name, dates in actual.items()
        },
    }


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
    design = load_discovery_design(design_path, require_current=True)
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
    errors: list[dict[str, object]] = []
    split_calendars: dict[str, object] = {}

    def persist_status(phase: str, current: str | None = None) -> None:
        write_json_atomic(
            status_path,
            {
                "schema_version": design["schema_version"],
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
            split_calendar = _split_calendar_evidence(base_dataset, design)
            if split_calendar is not None:
                split_calendars[str(horizon)] = split_calendar
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
                except MicrostructureClassSupportError as exc:
                    errors.append(
                        {
                            "candidate_id": candidate_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "class_support_failure": dict(exc.evidence),
                        }
                    )
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
        "schema_version": design["schema_version"],
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
        "all_window_dates_consumed_after_run": bool(
            data.get("all_window_dates_consumed_after_run", True)
        ),
        "split_calendar_evidence": split_calendars,
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
    parser = argparse.ArgumentParser(
        description="Run the current precommitted action-value discovery screen"
    )
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
