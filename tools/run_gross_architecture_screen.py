"""Run a precommitted consumed-data gross-return architecture screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
from safetensors.numpy import save_file as save_safetensors


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    GROSS_ARCHITECTURE_SCHEMA_VERSION,
    GROSS_TARGET_MODE,
    GrossArchitectureSpec,
    average_label_uniqueness,
    causal_cusum_event_mask,
    evaluate_gross_forecast,
    gross_midpoint_log_returns_bps,
    predict_lightgbm_gross_model,
    predict_torch_gross_model,
    train_lightgbm_gross_baseline,
    train_torch_gross_model,
    valid_sequence_endpoints,
)
from simple_ai_trading.microstructure_cache import (  # noqa: E402
    load_microstructure_dataset_cache,
    microstructure_dataset_cache_key,
    save_microstructure_dataset_cache,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DESIGN_SCHEMA_VERSION = "gross-architecture-screen-design-v1"
REPORT_SCHEMA_VERSION = "gross-architecture-screen-report-v1"
_ROLE_NAMES = (
    "train",
    "early_stop",
    "calibration",
    "policy",
    "development_evaluation",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _is_git_oid(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _parse_date(value: object, *, label: str) -> date:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"gross architecture {label} must be YYYY-MM-DD") from exc


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_implementation_binding(binding: Mapping[str, object]) -> None:
    commit = str(binding.get("commit") or "").lower()
    files = binding.get("files")
    if not _is_git_oid(commit) or not isinstance(files, list) or not files:
        raise ValueError("gross architecture implementation binding is incomplete")
    try:
        _git_output("merge-base", "--is-ancestor", commit, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise ValueError("gross architecture implementation commit is not an ancestor") from exc
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("gross architecture implementation file is invalid")
        relative = Path(str(item.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("gross architecture implementation path is unsafe")
        normalized = relative.as_posix()
        if normalized in seen or not normalized:
            raise ValueError("gross architecture implementation paths are duplicated")
        seen.add(normalized)
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != item.get("sha256"):
            raise ValueError(f"gross architecture implementation changed: {normalized}")


def load_gross_architecture_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("gross architecture design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("gross architecture design must be an object")
    expected_sha = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if not _is_sha256(expected_sha) or expected_sha != _canonical_sha256(canonical):
        raise ValueError("gross architecture design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != 13
        or payload.get("purpose") != "consumed_data_architecture_development"
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("target_mode") != GROSS_TARGET_MODE
    ):
        raise ValueError("gross architecture design contract is invalid")
    data = payload.get("data")
    resources = payload.get("runtime_resources")
    sampler = payload.get("event_sampler")
    stages = payload.get("stages")
    candidates = payload.get("neural_candidates")
    gates = payload.get("development_gates")
    terminal = payload.get("reserved_terminal")
    execution = payload.get("execution")
    ranking = payload.get("ranking")
    if not all(
        isinstance(value, Mapping)
        for value in (
            data,
            resources,
            sampler,
            stages,
            gates,
            terminal,
            execution,
            ranking,
        )
    ) or not isinstance(candidates, list):
        raise ValueError("gross architecture design sections are incomplete")
    assert isinstance(data, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(gates, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(ranking, Mapping)
    roles = data.get("roles")
    if (
        data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not False
        or not isinstance(roles, Mapping)
        or set(roles) != set(_ROLE_NAMES)
    ):
        raise ValueError("gross architecture data contract is invalid")
    previous_end: date | None = None
    for role in _ROLE_NAMES:
        value = roles[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"gross architecture {role} role is invalid")
        first = _parse_date(value.get("start"), label=f"{role} start")
        last = _parse_date(value.get("end"), label=f"{role} end")
        if first > last or (previous_end is not None and first != previous_end + timedelta(days=1)):
            raise ValueError("gross architecture roles must be contiguous")
        previous_end = last
    start = _parse_date(data.get("start_date"), label="data start")
    end = _parse_date(data.get("end_date"), label="data end")
    first_role = roles[_ROLE_NAMES[0]]
    last_role = roles[_ROLE_NAMES[-1]]
    if start != _parse_date(first_role["start"], label="first role") or end != _parse_date(
        last_role["end"],
        label="last role",
    ):
        raise ValueError("gross architecture roles do not partition the data window")
    terminal_date = _parse_date(terminal.get("date"), label="reserved terminal")
    if (
        terminal_date != end + timedelta(days=1)
        or terminal.get("included_in_dataset") is not False
        or terminal.get("access_permitted") is not False
    ):
        raise ValueError("gross architecture reserved terminal contract is invalid")
    if (
        resources.get("duckdb_memory_limit") != "4GB"
        or int(resources.get("warehouse_threads") or 0) != 8
        or resources.get("compute_backend") != "directml"
        or resources.get("spill_directory_policy") != "warehouse_adjacent"
    ):
        raise ValueError("gross architecture resource contract is invalid")
    if (
        int(execution.get("horizon_seconds") or 0) != 300
        or int(execution.get("total_latency_ms") or -1) != 750
        or float(execution.get("taker_fee_bps_per_side") or -1.0) != 5.0
        or float(execution.get("additional_slippage_bps_per_side") or -1.0)
        != 1.0
        or int(execution.get("decision_cadence_seconds") or 0) != 5
        or int(execution.get("max_quote_age_ms") or 0) != 1_000
        or float(execution.get("reference_order_notional_quote") or 0.0)
        != 1_000.0
        or float(execution.get("max_l1_participation") or 0.0) != 1.0
    ):
        raise ValueError("gross architecture execution diagnostic contract is invalid")
    if (
        float(sampler.get("volatility_multiplier") or 0.0) <= 0.0
        or float(sampler.get("minimum_threshold_bps") or 0.0) <= 0.0
    ):
        raise ValueError("gross architecture event sampler is invalid")
    stage_one = stages.get("stage_one")
    stage_two = stages.get("stage_two")
    if not isinstance(stage_one, Mapping) or not isinstance(stage_two, Mapping):
        raise ValueError("gross architecture stage budgets are invalid")
    if (
        int(stage_one.get("training_stride") or 0) < 2
        or int(stage_one.get("max_epochs") or 0) < 1
        or int(stage_one.get("keep_candidates") or 0) != 2
        or int(stage_two.get("training_stride") or 0) != 1
        or int(stage_two.get("max_epochs") or 0)
        <= int(stage_one.get("max_epochs") or 0)
    ):
        raise ValueError("gross architecture successive-halving contract is invalid")
    parsed_specs = [GrossArchitectureSpec(**dict(value)) for value in candidates]
    if len(parsed_specs) < 3 or len({spec.candidate_id for spec in parsed_specs}) != len(
        parsed_specs
    ):
        raise ValueError("gross architecture candidates are incomplete or duplicated")
    required_gates = {
        "minimum_direction_auc",
        "minimum_spearman_ic",
        "require_mae_better_than_zero",
        "minimum_top_500_exact_after_cost_bps",
    }
    if set(gates) != required_gates:
        raise ValueError("gross architecture development gates are incomplete")
    if (
        ranking.get("stage_one_role") != "calibration"
        or ranking.get("final_ranking_role") != "policy"
        or tuple(ranking.get("lexicographic_descending") or ())
        != (
            "top_500_mean_exact_after_cost_bps",
            "spearman_information_coefficient",
            "direction_auc",
        )
        or tuple(ranking.get("diagnostic_top_rows") or ()) != (100, 500, 1_000)
        or ranking.get("development_evaluation_used_for_selection") is not False
    ):
        raise ValueError("gross architecture ranking contract is invalid")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("gross architecture implementation binding is missing")
    if require_current:
        _validate_implementation_binding(implementation)
    return payload, str(expected_sha)


def _utc_day_bounds(first: date, last: date) -> tuple[int, int]:
    start_ms = int(
        datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1_000
    )
    end_ms = int(
        datetime.combine(
            last + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
        * 1_000
    ) - 1
    return start_ms, end_ms


def _role_indexes(
    dataset,
    roles: Mapping[str, object],
    event_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    output: dict[str, np.ndarray] = {}
    evidence: dict[str, object] = {}
    for position, role in enumerate(_ROLE_NAMES):
        raw = roles[role]
        assert isinstance(raw, Mapping)
        first = _parse_date(raw["start"], label=f"{role} start")
        last = _parse_date(raw["end"], label=f"{role} end")
        first_ms, last_ms = _utc_day_bounds(first, last)
        indexes = np.flatnonzero(
            (dataset.decision_time_ms >= first_ms)
            & (dataset.decision_time_ms <= last_ms)
            & event_mask
        ).astype(np.int64)
        raw_rows = len(indexes)
        if position + 1 < len(_ROLE_NAMES):
            next_role = roles[_ROLE_NAMES[position + 1]]
            assert isinstance(next_role, Mapping)
            next_start = _parse_date(
                next_role["start"],
                label=f"{_ROLE_NAMES[position + 1]} start",
            )
            next_start_ms, _unused = _utc_day_bounds(next_start, next_start)
            indexes = indexes[
                (dataset.long_exit_time_ms[indexes] < next_start_ms)
                & (dataset.short_exit_time_ms[indexes] < next_start_ms)
            ]
        if len(indexes) < 256:
            raise ValueError(f"gross architecture {role} has insufficient event rows")
        output[role] = indexes
        evidence[role] = {
            "start": first.isoformat(),
            "end": last.isoformat(),
            "raw_event_rows": raw_rows,
            "purged_event_rows": len(indexes),
            "first_decision_time_ms": int(dataset.decision_time_ms[indexes[0]]),
            "last_decision_time_ms": int(dataset.decision_time_ms[indexes[-1]]),
        }
    return output, evidence


def _top_row(metrics: Mapping[str, object], requested: int = 500) -> Mapping[str, object]:
    rows = metrics.get("top_rows")
    if not isinstance(rows, Sequence):
        raise ValueError("gross architecture metrics have no top-row diagnostics")
    for value in rows:
        if isinstance(value, Mapping) and int(value.get("requested_rows") or 0) == requested:
            return value
    raise ValueError(f"gross architecture metrics lack top-{requested} diagnostics")


def _screen_rank(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    top = _top_row(metrics)
    return (
        float(top["mean_exact_after_cost_bps"]),
        float(metrics["spearman_information_coefficient"]),
        float(metrics["direction_auc"]),
    )


def _gate_reasons(
    metrics: Mapping[str, object],
    gates: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if float(metrics["direction_auc"]) <= float(gates["minimum_direction_auc"]):
        reasons.append("direction_auc_gate_failed")
    if float(metrics["spearman_information_coefficient"]) <= float(
        gates["minimum_spearman_ic"]
    ):
        reasons.append("spearman_ic_gate_failed")
    if bool(gates["require_mae_better_than_zero"]) and float(
        metrics["mean_absolute_error_bps"]
    ) >= float(metrics["zero_baseline_mae_bps"]):
        reasons.append("mae_not_better_than_zero")
    if float(_top_row(metrics)["mean_exact_after_cost_bps"]) <= float(
        gates["minimum_top_500_exact_after_cost_bps"]
    ):
        reasons.append("top_500_exact_after_cost_gate_failed")
    return reasons


def _artifact_summary(model) -> dict[str, object]:
    model_family = (
        str(model.model_family)
        if hasattr(model, "model_family")
        else str(model.spec.family)
    )
    output = {
        "schema_version": model.schema_version,
        "model_family": model_family,
        "model_sha256": model.model_sha256,
        "backend_requested": model.backend_requested,
        "backend_kind": model.backend_kind,
        "backend_device": model.backend_device,
        "target_mode": model.target_mode,
        "trading_authority": model.trading_authority,
        "execution_claim": model.execution_claim,
        "profitability_claim": model.profitability_claim,
    }
    if hasattr(model, "best_epoch"):
        output.update(
            {
                "spec": asdict(model.spec),
                "best_epoch": model.best_epoch,
                "training_loss": model.training_loss,
                "tuning_loss": model.tuning_loss,
            }
        )
    else:
        output.update(
            {
                "mean_iteration": model.mean_iteration,
                "direction_iteration": model.direction_iteration,
            }
        )
    return output


def _save_neural_artifact(path: Path, model) -> dict[str, object]:
    arrays = {
        "scaler_center": np.ascontiguousarray(model.scaler_center, dtype=np.float32),
        "scaler_scale": np.ascontiguousarray(model.scaler_scale, dtype=np.float32),
        **{
            f"state.{name}": np.ascontiguousarray(value, dtype=np.float32)
            for name, value in model.state.items()
        },
    }
    metadata = {
        "schema_version": model.schema_version,
        "candidate_id": model.spec.candidate_id,
        "model_sha256": model.model_sha256,
        "target_mode": model.target_mode,
        "trading_authority": "false",
        "execution_claim": "false",
        "profitability_claim": "false",
    }
    save_safetensors(arrays, str(path), metadata=metadata)
    return {"path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def run_gross_architecture_screen(
    *,
    design_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_gross_architecture_design(design_path)
    data = design["data"]
    resources = design["runtime_resources"]
    sampler = design["event_sampler"]
    stages = design["stages"]
    gates = design["development_gates"]
    ranking = design["ranking"]
    assert isinstance(data, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(gates, Mapping)
    assert isinstance(ranking, Mapping)
    requested_top_rows = tuple(int(value) for value in ranking["diagnostic_top_rows"])
    effective_memory = str(memory_limit or resources["duckdb_memory_limit"]).upper()
    effective_threads = int(threads or resources["warehouse_threads"])
    effective_backend = str(compute_backend or resources["compute_backend"]).lower()
    if (
        effective_memory != resources["duckdb_memory_limit"]
        or effective_threads != int(resources["warehouse_threads"])
        or effective_backend != resources["compute_backend"]
    ):
        raise ValueError("runtime overrides differ from the precommitted resource contract")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    runtime = {
        "duckdb_memory_limit": effective_memory,
        "warehouse_threads": effective_threads,
        "compute_backend_requested": effective_backend,
        "spill_directory_policy": "warehouse_adjacent",
    }

    def progress(phase: str, **extra: object) -> None:
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "design_sha256": design_sha256,
            "phase": phase,
            "runtime_resources": runtime,
            **extra,
        }
        print(
            "gross-architecture-screen "
            + " ".join(f"{name}={value}" for name, value in payload.items() if name != "runtime_resources"),
            flush=True,
        )
        write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    first = _parse_date(data["start_date"], label="data start")
    last = _parse_date(data["end_date"], label="data end")
    start_ms, end_ms = _utc_day_bounds(first, last)
    execution = design["execution"]
    assert isinstance(execution, Mapping)
    cache_parameters: dict[str, object]
    progress("initialize")
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
    ) as warehouse:
        progress("verify-source")
        source_evidence = dict(warehouse.require_causal_feature_bars(str(data["symbol"])))
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=bool(data["full_history_inventory_required"]),
        )
        source_evidence["corpus_certificate"] = certificate
        cache_parameters = {
            "symbol": str(data["symbol"]),
            "requested_start_ms": start_ms,
            "requested_end_ms": end_ms,
            "horizon_seconds": int(execution["horizon_seconds"]),
            "total_latency_ms": int(execution["total_latency_ms"]),
            "taker_fee_bps": float(execution["taker_fee_bps_per_side"]),
            "additional_slippage_bps_per_side": float(
                execution["additional_slippage_bps_per_side"]
            ),
            "reference_order_notional_quote": float(
                execution["reference_order_notional_quote"]
            ),
            "max_l1_participation": float(execution["max_l1_participation"]),
            "max_quote_age_ms": int(execution["max_quote_age_ms"]),
            "decision_cadence_seconds": int(execution["decision_cadence_seconds"]),
            "require_full_history_inventory": bool(
                data["full_history_inventory_required"]
            ),
            "source_evidence": source_evidence,
        }
        cache_key = microstructure_dataset_cache_key(**cache_parameters)
        progress("cache-lookup")
        dataset = load_microstructure_dataset_cache(warehouse, **cache_parameters)
        cache_state = "hit"
        if dataset is None:
            cache_state = "build"
            progress("exact-bbo-dataset-build")
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=str(data["symbol"]),
                horizon_seconds=int(execution["horizon_seconds"]),
                total_latency_ms=int(execution["total_latency_ms"]),
                taker_fee_bps=float(execution["taker_fee_bps_per_side"]),
                additional_slippage_bps_per_side=float(
                    execution["additional_slippage_bps_per_side"]
                ),
                max_quote_age_ms=int(execution["max_quote_age_ms"]),
                reference_order_notional_quote=float(
                    execution["reference_order_notional_quote"]
                ),
                max_l1_participation=float(execution["max_l1_participation"]),
                decision_cadence_seconds=int(execution["decision_cadence_seconds"]),
                start_ms=start_ms,
                end_ms=end_ms,
                require_full_history_inventory=bool(
                    data["full_history_inventory_required"]
                ),
            )
            progress("cache-write", dataset_rows=dataset.rows)
            cache_key = save_microstructure_dataset_cache(
                warehouse,
                dataset,
                requested_start_ms=start_ms,
                requested_end_ms=end_ms,
                require_full_history_inventory=bool(
                    data["full_history_inventory_required"]
                ),
            )
            cache_state = "written"
        gross_target = gross_midpoint_log_returns_bps(dataset)
        event_mask = causal_cusum_event_mask(
            dataset,
            volatility_multiplier=float(sampler["volatility_multiplier"]),
            minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
        )
        roles, role_evidence = _role_indexes(dataset, data["roles"], event_mask)
        progress(
            "dataset-ready",
            dataset_rows=dataset.rows,
            event_rows=int(np.sum(event_mask)),
            cache_state=cache_state,
        )

        neural_specs = [
            GrossArchitectureSpec(**dict(value))
            for value in design["neural_candidates"]
        ]
        stage_one = stages["stage_one"]
        stage_two = stages["stage_two"]
        assert isinstance(stage_one, Mapping)
        assert isinstance(stage_two, Mapping)
        stage_one_results: list[dict[str, object]] = []
        for spec in neural_specs:
            train = valid_sequence_endpoints(
                dataset.decision_time_ms,
                roles["train"],
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )[:: int(stage_one["training_stride"])]
            tuning = valid_sequence_endpoints(
                dataset.decision_time_ms,
                roles["early_stop"],
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )[:: int(stage_one["training_stride"])]
            train_weights = average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                train,
            )
            tuning_weights = average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                tuning,
            )
            progress("stage-one-train", candidate=spec.candidate_id)
            model = train_torch_gross_model(
                dataset,
                gross_target,
                train_endpoints=train,
                tuning_endpoints=tuning,
                spec=spec,
                compute_backend=effective_backend,
                seed=int(design["seed"]),
                batch_size=int(stage_one["batch_size"]),
                max_epochs=int(stage_one["max_epochs"]),
                patience=int(stage_one["patience"]),
                train_sample_weights=train_weights,
                tuning_sample_weights=tuning_weights,
                progress=lambda epoch, total, training_loss, tuning_loss, candidate=spec.candidate_id: progress(
                    "stage-one-epoch",
                    candidate=candidate,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                ),
            )
            calibration_endpoints = valid_sequence_endpoints(
                dataset.decision_time_ms,
                roles["calibration"],
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )
            prediction = predict_torch_gross_model(
                model,
                dataset,
                calibration_endpoints,
                compute_backend=effective_backend,
                batch_size=int(stage_one["batch_size"]),
            )
            metrics = evaluate_gross_forecast(
                dataset,
                gross_target,
                prediction,
                requested_top_rows=requested_top_rows,
            ).asdict()
            stage_one_results.append(
                {
                    "candidate_id": spec.candidate_id,
                    "artifact": _artifact_summary(model),
                    "calibration_metrics": metrics,
                    "rank": list(_screen_rank(metrics)),
                }
            )
            del model, prediction
        stage_one_results.sort(key=lambda value: tuple(value["rank"]), reverse=True)
        selected_ids = [
            str(value["candidate_id"])
            for value in stage_one_results[: int(stage_one["keep_candidates"])]
        ]
        progress("stage-one-complete", selected=",".join(selected_ids))

        train_full = roles["train"]
        tuning_full = roles["early_stop"]
        train_uniqueness = average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            train_full,
        )
        tuning_uniqueness = average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            tuning_full,
        )
        progress("stage-two-lightgbm")
        baseline = train_lightgbm_gross_baseline(
            dataset,
            gross_target,
            train_endpoints=train_full,
            tuning_endpoints=tuning_full,
            train_uniqueness=train_uniqueness,
            tuning_uniqueness=tuning_uniqueness,
            compute_backend=effective_backend,
            seed=int(design["seed"]),
        )
        baseline_policy = evaluate_gross_forecast(
            dataset,
            gross_target,
            predict_lightgbm_gross_model(baseline, dataset, roles["policy"]),
            requested_top_rows=requested_top_rows,
        ).asdict()
        baseline_development = evaluate_gross_forecast(
            dataset,
            gross_target,
            predict_lightgbm_gross_model(
                baseline,
                dataset,
                roles["development_evaluation"],
            ),
            requested_top_rows=requested_top_rows,
        ).asdict()
        baseline_artifact_path = destination / "lightgbm-baseline.json"
        write_json_atomic(
            baseline_artifact_path,
            {
                **_artifact_summary(baseline),
                "mean_model": baseline.mean_model,
                "direction_model": baseline.direction_model,
            },
            indent=2,
            sort_keys=True,
        )
        final_results: list[dict[str, object]] = [
            {
                "candidate_id": "lightgbm-gross-baseline",
                "artifact": _artifact_summary(baseline),
                "artifact_file": {
                    "path": baseline_artifact_path.name,
                    "sha256": _sha256_file(baseline_artifact_path),
                    "bytes": baseline_artifact_path.stat().st_size,
                },
                "policy_metrics": baseline_policy,
                "development_metrics": baseline_development,
                "rejection_reasons": _gate_reasons(baseline_development, gates),
            }
        ]
        for spec in neural_specs:
            if spec.candidate_id not in selected_ids:
                continue
            train = valid_sequence_endpoints(
                dataset.decision_time_ms,
                train_full,
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )
            tuning = valid_sequence_endpoints(
                dataset.decision_time_ms,
                tuning_full,
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )
            train_weights = average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                train,
            )
            tuning_weights = average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                tuning,
            )
            progress("stage-two-train", candidate=spec.candidate_id)
            model = train_torch_gross_model(
                dataset,
                gross_target,
                train_endpoints=train,
                tuning_endpoints=tuning,
                spec=spec,
                compute_backend=effective_backend,
                seed=int(design["seed"]),
                batch_size=int(stage_two["batch_size"]),
                max_epochs=int(stage_two["max_epochs"]),
                patience=int(stage_two["patience"]),
                train_sample_weights=train_weights,
                tuning_sample_weights=tuning_weights,
                progress=lambda epoch, total, training_loss, tuning_loss, candidate=spec.candidate_id: progress(
                    "stage-two-epoch",
                    candidate=candidate,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                ),
            )
            policy_endpoints = valid_sequence_endpoints(
                dataset.decision_time_ms,
                roles["policy"],
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )
            development_endpoints = valid_sequence_endpoints(
                dataset.decision_time_ms,
                roles["development_evaluation"],
                sequence_length=spec.sequence_length,
                cadence_seconds=dataset.decision_cadence_seconds,
            )
            policy_metrics = evaluate_gross_forecast(
                dataset,
                gross_target,
                predict_torch_gross_model(
                    model,
                    dataset,
                    policy_endpoints,
                    compute_backend=effective_backend,
                    batch_size=int(stage_two["batch_size"]),
                ),
                requested_top_rows=requested_top_rows,
            ).asdict()
            development_metrics = evaluate_gross_forecast(
                dataset,
                gross_target,
                predict_torch_gross_model(
                    model,
                    dataset,
                    development_endpoints,
                    compute_backend=effective_backend,
                    batch_size=int(stage_two["batch_size"]),
                ),
                requested_top_rows=requested_top_rows,
            ).asdict()
            artifact_file = _save_neural_artifact(
                destination / f"{spec.candidate_id}.safetensors",
                model,
            )
            baseline_top = _top_row(baseline_development)
            candidate_top = _top_row(development_metrics)
            final_results.append(
                {
                    "candidate_id": spec.candidate_id,
                    "artifact": _artifact_summary(model),
                    "artifact_file": artifact_file,
                    "policy_metrics": policy_metrics,
                    "development_metrics": development_metrics,
                    "uplift_vs_lightgbm": {
                        "direction_auc": float(development_metrics["direction_auc"])
                        - float(baseline_development["direction_auc"]),
                        "spearman_information_coefficient": float(
                            development_metrics["spearman_information_coefficient"]
                        )
                        - float(
                            baseline_development[
                                "spearman_information_coefficient"
                            ]
                        ),
                        "top_500_mean_exact_after_cost_bps": float(
                            candidate_top["mean_exact_after_cost_bps"]
                        )
                        - float(baseline_top["mean_exact_after_cost_bps"]),
                    },
                    "rejection_reasons": _gate_reasons(
                        development_metrics,
                        gates,
                    ),
                }
            )
            del model

    final_results.sort(
        key=lambda value: _screen_rank(value["policy_metrics"]),
        reverse=True,
    )
    for value in final_results:
        value["status"] = (
            "research_candidate" if not value["rejection_reasons"] else "rejected"
        )
        value["trading_authority"] = False
        value["execution_claim"] = False
        value["profitability_claim"] = False
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "consumed_data_gross_architecture_development_evidence",
        "status": (
            "research_candidate"
            if any(value["status"] == "research_candidate" for value in final_results)
            else "rejected"
        ),
        "round": 13,
        "design_sha256": design_sha256,
        "gross_model_schema_version": GROSS_ARCHITECTURE_SCHEMA_VERSION,
        "target_mode": GROSS_TARGET_MODE,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
        "portfolio_claim": False,
        "leverage_applied": False,
        "runtime_resources": runtime,
        "corpus_certificate_sha256": certificate["certificate_sha256"],
        "dataset": {
            "rows": dataset.rows,
            "cache_state": cache_state,
            "cache_key": cache_key,
            "source_manifest_fingerprint": source_evidence["manifest_fingerprint"],
            "gross_target_mean_bps": float(np.mean(gross_target)),
            "gross_target_std_bps": float(np.std(gross_target)),
            "event_rows": int(np.sum(event_mask)),
            "roles": role_evidence,
        },
        "successive_halving": {
            "stage_one_results": stage_one_results,
            "selected_candidate_ids": selected_ids,
        },
        "final_results": final_results,
        "limitations": [
            "top-row diagnostics contain overlapping forecasts and are not a portfolio return",
            "development dates were consumed before this architecture screen",
            "positive diagnostics cannot establish profitability or authorize trading",
            "the reserved terminal date was neither loaded nor labeled",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    progress("complete", report_sha256=report["report_sha256"], status=report["status"])
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the precommitted consumed-data gross architecture screen"
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--compute-backend")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_gross_architecture_screen(
        design_path=args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(
        f"gross-architecture-screen: status={report['status']} "
        f"sha256={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
