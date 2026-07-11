"""Run the precommitted adaptive-barrier action-value ensemble screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.compute import resolve_backend  # noqa: E402
from simple_ai_trading.microstructure_action_architecture import (  # noqa: E402
    ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION,
    ActionValueArchitectureSpec,
    ActionValueEnsembleBatch,
    TrainedActionValueModel,
    ensemble_action_value_predictions,
    predict_action_value_model,
    train_action_value_model,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ACTION_POLICY_SCHEMA_VERSION,
    ActionPolicySpec,
    ActionScoreBatch,
    barrier_trace_gate_reasons,
    derive_action_scores,
    select_barrier_threshold,
    simulate_barrier_action_trace,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    _auc,
    _correlation,
    average_label_uniqueness,
    causal_cusum_event_mask,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
    build_adaptive_barrier_targets,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402

try:  # noqa: E402
    from tools.run_gross_architecture_screen import (
        _artifact_summary,
        _canonical_sha256,
        _is_sha256,
        _parse_date,
        _save_neural_artifact,
        _validate_implementation_binding,
    )
    from tools.run_head_coherence_screen import _load_corpus
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from run_gross_architecture_screen import (
        _artifact_summary,
        _canonical_sha256,
        _is_sha256,
        _parse_date,
        _save_neural_artifact,
        _validate_implementation_binding,
    )
    from run_head_coherence_screen import _load_corpus


DESIGN_SCHEMA_VERSION = "adaptive-action-screen-design-v1"
REPORT_SCHEMA_VERSION = "adaptive-action-screen-report-v1"
_DAY_MS = 86_400_000
_ROLE_NAMES = (
    "train",
    "early_stop",
    "calibration",
    "policy",
    "development_evaluation",
)
_PROFILE_NAMES = ("conservative", "regular", "aggressive")
_GATE_FIELDS = {
    "minimum_trades",
    "minimum_total_net_bps",
    "maximum_drawdown_bps",
    "minimum_positive_day_ratio",
    "minimum_worst_trade_bps",
    "minimum_profit_factor",
}


def _day_id(value: object, *, label: str) -> int:
    parsed = _parse_date(value, label=label)
    return int(
        datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1_000
        // _DAY_MS
    )


def _validate_gates(gates: object, *, label: str) -> None:
    if not isinstance(gates, Mapping) or set(gates) != _GATE_FIELDS:
        raise ValueError(f"adaptive action {label} gates are incomplete")
    values = tuple(float(gates[name]) for name in _GATE_FIELDS)
    if (
        not all(math.isfinite(value) for value in values)
        or int(gates["minimum_trades"]) < 1
        or float(gates["minimum_total_net_bps"]) < 0.0
        or float(gates["maximum_drawdown_bps"]) <= 0.0
        or not 0.0 <= float(gates["minimum_positive_day_ratio"]) <= 1.0
        or float(gates["minimum_worst_trade_bps"]) >= 0.0
        or float(gates["minimum_profit_factor"]) < 1.0
    ):
        raise ValueError(f"adaptive action {label} gates are invalid")


def load_adaptive_action_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("adaptive action design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("adaptive action design must be an object")
    claimed = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("adaptive action design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != 16
        or payload.get("purpose")
        != "consumed_data_adaptive_barrier_action_value_screen"
        or payload.get("target_mode") != ADAPTIVE_BARRIER_TARGET_MODE
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("portfolio_claim") is not False
        or payload.get("leverage_applied") is not False
    ):
        raise ValueError("adaptive action design contract is invalid")
    section_names = (
        "implementation",
        "predecessor_evidence",
        "data",
        "execution",
        "barrier_targets",
        "runtime_resources",
        "event_sampler",
        "model",
        "training",
        "threshold_policy",
        "evaluation",
        "reserved_terminal",
    )
    sections = {name: payload.get(name) for name in section_names}
    if not all(isinstance(value, Mapping) for value in sections.values()):
        raise ValueError("adaptive action design sections are incomplete")
    implementation = sections["implementation"]
    predecessor = sections["predecessor_evidence"]
    data = sections["data"]
    execution = sections["execution"]
    barriers = sections["barrier_targets"]
    resources = sections["runtime_resources"]
    sampler = sections["event_sampler"]
    model = sections["model"]
    training = sections["training"]
    threshold = sections["threshold_policy"]
    evaluation = sections["evaluation"]
    terminal = sections["reserved_terminal"]
    assert isinstance(implementation, Mapping)
    assert isinstance(predecessor, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(barriers, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(threshold, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(terminal, Mapping)
    if require_current:
        _validate_implementation_binding(implementation)
    if (
        predecessor.get("round") != 15
        or predecessor.get("design_sha256")
        != "4ff50e579bb036d3146a8ea01e8f502efea0b2a0445df8f187f612199ebbe43c"
        or predecessor.get("source_report_canonical_sha256")
        != "3518173cd9d8ba25e9bcb8ff6b421254092901d981001cd77b19e8c10c15fc12"
        or predecessor.get("publication_sha256")
        != "ee18cdff45db44346960033a0bd55adf45359953106af1033738fb18099ef4a4"
    ):
        raise ValueError("adaptive action predecessor evidence is invalid")
    roles = data.get("roles")
    if (
        data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not False
        or data.get("start_date") != "2023-05-16"
        or data.get("end_date") != "2023-07-06"
        or not isinstance(roles, Mapping)
        or tuple(roles) != _ROLE_NAMES
    ):
        raise ValueError("adaptive action data contract is invalid")
    expected_roles = {
        "train": {"start": "2023-05-16", "end": "2023-06-15"},
        "early_stop": {"start": "2023-06-16", "end": "2023-06-20"},
        "calibration": {"start": "2023-06-21", "end": "2023-06-25"},
        "policy": {"start": "2023-06-26", "end": "2023-06-30"},
        "development_evaluation": {"start": "2023-07-01", "end": "2023-07-06"},
    }
    if roles != expected_roles:
        raise ValueError("adaptive action source roles drifted")
    if (
        int(execution.get("horizon_seconds") or 0) != 900
        or int(execution.get("total_latency_ms") or -1) != 750
        or float(execution.get("taker_fee_bps_per_side") or -1.0) != 5.0
        or float(execution.get("additional_slippage_bps_per_side") or -1.0) != 1.0
        or int(execution.get("decision_cadence_seconds") or 0) != 5
        or int(execution.get("max_quote_age_ms") or 0) != 1_000
        or float(execution.get("reference_order_notional_quote") or 0.0) != 1_000.0
        or float(execution.get("max_l1_participation") or 0.0) != 1.0
    ):
        raise ValueError("adaptive action execution contract is invalid")
    barrier_spec = AdaptiveBarrierSpec(**dict(barriers))
    if barrier_spec.horizon_seconds != int(execution["horizon_seconds"]):
        raise ValueError("adaptive action barrier horizon differs from execution")
    if (
        resources.get("duckdb_memory_limit") != "4GB"
        or int(resources.get("warehouse_threads") or 0) != 8
        or resources.get("compute_backend") != "directml"
        or resources.get("spill_directory_policy") != "warehouse_adjacent"
        or resources.get("training_worker_isolation") != "clean_process"
        or resources.get("cpu_fallback_permitted") is not False
    ):
        raise ValueError("adaptive action resource contract is invalid")
    if (
        sampler.get("method") != "daily_reset_causal_cusum"
        or float(sampler.get("volatility_multiplier") or 0.0) <= 0.0
        or float(sampler.get("minimum_threshold_bps") or 0.0) <= 0.0
        or sampler.get("minimum_activity_quota") is not None
    ):
        raise ValueError("adaptive action event sampler is invalid")
    model_spec = ActionValueArchitectureSpec(**dict(model))
    seeds = tuple(int(value) for value in training.get("ensemble_seeds") or ())
    if (
        model_spec.family != "shared_residual_mlp"
        or seeds != (29, 43, 71)
        or int(training.get("batch_size") or 0) < 1_024
        or int(training.get("max_epochs") or 0) < 2
        or int(training.get("patience") or 0) < 1
        or training.get("target_scenario") != "base"
        or training.get("ensemble_method") != "independent_seed_mean_and_dispersion"
    ):
        raise ValueError("adaptive action training contract is invalid")
    quantiles = threshold.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or len(quantiles) < 3
        or len(set(float(value) for value in quantiles)) != len(quantiles)
        or any(not 0.0 < float(value) < 1.0 for value in quantiles)
        or not math.isfinite(float(threshold.get("drawdown_penalty") or -1.0))
        or float(threshold.get("drawdown_penalty") or -1.0) < 0.0
        or threshold.get("selection_scenario") != "stress"
        or threshold.get("no_passing_threshold_action") != "abstain"
    ):
        raise ValueError("adaptive action threshold contract is invalid")
    profiles = payload.get("risk_profiles")
    if (
        not isinstance(profiles, list)
        or tuple(
            value.get("profile") if isinstance(value, Mapping) else None
            for value in profiles
        )
        != _PROFILE_NAMES
    ):
        raise ValueError("adaptive action risk profiles are incomplete")
    for raw in profiles:
        assert isinstance(raw, Mapping)
        ActionPolicySpec(
            **{
                name: raw[name]
                for name in (
                    "profile",
                    "epistemic_penalty",
                    "minimum_profitable_probability",
                    "minimum_member_agreement",
                    "maximum_epistemic_std_bps",
                    "minimum_lower_bound_bps",
                )
            }
        )
        _validate_gates(raw.get("calibration_gates"), label="calibration")
        _validate_gates(raw.get("policy_gates"), label="policy")
        _validate_gates(raw.get("development_gates"), label="development")
        if raw.get("leverage_applied") is not False:
            raise ValueError("adaptive action screen must remain unleveraged")
    if (
        evaluation.get("calibration_used_for_threshold_selection") is not True
        or evaluation.get("policy_used_for_profile_acceptance") is not True
        or evaluation.get("development_used_for_selection") is not False
        or evaluation.get("positions_must_exit_within_utc_day") is not True
        or int(evaluation.get("maximum_concurrent_positions") or 0) != 1
        or evaluation.get("overlap_permitted") is not False
    ):
        raise ValueError("adaptive action evaluation contract is invalid")
    data_end = _day_id(data["end_date"], label="data end")
    terminal_day = _day_id(terminal.get("date"), label="terminal")
    if (
        terminal_day != data_end + 1
        or terminal.get("included_in_dataset") is not False
        or terminal.get("access_permitted") is not False
    ):
        raise ValueError("adaptive action terminal contract is invalid")
    return payload, str(claimed)


def _role_indexes(
    dataset,
    targets: AdaptiveBarrierTargets,
    event_mask: np.ndarray,
    roles: Mapping[str, object],
    terminal_day: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    valid = np.zeros(dataset.rows, dtype=bool)
    valid[targets.source_indexes[targets.valid]] = True
    mask = np.asarray(event_mask, dtype=bool)
    if mask.shape != (dataset.rows,):
        raise ValueError("adaptive action event mask shape is invalid")
    max_exit = np.maximum.reduce(
        (
            targets.base_long_exit_time_ms,
            targets.base_short_exit_time_ms,
            targets.stress_long_exit_time_ms,
            targets.stress_short_exit_time_ms,
        )
    )
    max_exit_full = np.full(dataset.rows, -1, dtype=np.int64)
    max_exit_full[targets.source_indexes] = max_exit
    output: dict[str, np.ndarray] = {}
    evidence: dict[str, object] = {}
    for position, role in enumerate(_ROLE_NAMES):
        raw = roles[role]
        if not isinstance(raw, Mapping):
            raise ValueError(f"adaptive action {role} role is invalid")
        first_day = _day_id(raw["start"], label=f"{role} start")
        last_day = _day_id(raw["end"], label=f"{role} end")
        next_day = (
            _day_id(
                roles[_ROLE_NAMES[position + 1]]["start"],
                label=f"{role} next role",
            )
            if position + 1 < len(_ROLE_NAMES)
            else int(terminal_day)
        )
        indexes = np.flatnonzero(
            mask
            & valid
            & (dataset.decision_time_ms >= first_day * _DAY_MS)
            & (dataset.decision_time_ms < (last_day + 1) * _DAY_MS)
            & (max_exit_full < next_day * _DAY_MS)
        ).astype(np.int64)
        if len(indexes) < 256:
            raise ValueError(f"adaptive action {role} role support is insufficient")
        output[role] = indexes
        evidence[role] = {
            "start": raw["start"],
            "end": raw["end"],
            "rows": len(indexes),
            "first_decision_time_ms": int(dataset.decision_time_ms[indexes[0]]),
            "last_decision_time_ms": int(dataset.decision_time_ms[indexes[-1]]),
            "last_exit_time_ms": int(np.max(max_exit_full[indexes])),
            "next_role_start_ms": int(next_day * _DAY_MS),
            "purged": bool(np.max(max_exit_full[indexes]) < next_day * _DAY_MS),
        }
    return output, evidence


def _targets_sha256(targets: AdaptiveBarrierTargets) -> str:
    contract = {
        "schema_version": targets.schema_version,
        "target_mode": targets.target_mode,
        "spec": asdict(targets.spec),
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    )
    fields = (
        "source_indexes",
        "valid",
        "stop_barrier_bps",
        "take_barrier_bps",
        "base_long_net_bps",
        "base_short_net_bps",
        "base_long_exit_time_ms",
        "base_short_exit_time_ms",
        "base_long_outcome",
        "base_short_outcome",
        "stress_long_net_bps",
        "stress_short_net_bps",
        "stress_long_exit_time_ms",
        "stress_short_exit_time_ms",
        "stress_long_outcome",
        "stress_short_outcome",
    )
    for name in fields:
        values = np.ascontiguousarray(getattr(targets, name))
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(values.dtype.str.encode("ascii") + b"\x00")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _ensemble_for_role(
    models: Sequence[TrainedActionValueModel],
    dataset,
    endpoints: np.ndarray,
    *,
    compute_backend: str,
    batch_size: int,
) -> ActionValueEnsembleBatch:
    return ensemble_action_value_predictions(
        [
            predict_action_value_model(
                model,
                dataset,
                endpoints,
                compute_backend=compute_backend,
                batch_size=batch_size,
            )
            for model in models
        ]
    )


def _target_positions(
    targets: AdaptiveBarrierTargets, endpoints: np.ndarray
) -> np.ndarray:
    positions = np.searchsorted(targets.source_indexes, endpoints)
    if (
        np.any(positions >= targets.rows)
        or not np.array_equal(targets.source_indexes[positions], endpoints)
        or not np.all(targets.valid[positions])
    ):
        raise ValueError("adaptive action diagnostics differ from barrier targets")
    return positions


def _forecast_diagnostics(
    targets: AdaptiveBarrierTargets,
    prediction: ActionValueEnsembleBatch,
    *,
    scenario: str,
) -> dict[str, object]:
    endpoints = np.asarray(prediction.endpoint_indexes, dtype=np.int64)
    positions = _target_positions(targets, endpoints)
    if scenario == "base":
        actuals = (
            targets.base_long_net_bps[positions],
            targets.base_short_net_bps[positions],
        )
    elif scenario == "stress":
        actuals = (
            targets.stress_long_net_bps[positions],
            targets.stress_short_net_bps[positions],
        )
    else:
        raise ValueError("adaptive action diagnostic scenario is unsupported")
    fields = {
        "long": (
            prediction.long_mean_bps,
            prediction.long_profitable_probability,
            prediction.long_lower_bps,
            prediction.long_upper_bps,
            prediction.long_epistemic_std_bps,
        ),
        "short": (
            prediction.short_mean_bps,
            prediction.short_profitable_probability,
            prediction.short_lower_bps,
            prediction.short_upper_bps,
            prediction.short_epistemic_std_bps,
        ),
    }
    output: dict[str, object] = {}
    for side, actual, values in zip(
        ("long", "short"), actuals, fields.values(), strict=True
    ):
        predicted, probability, lower, upper, epistemic = (
            np.asarray(value, dtype=np.float64) for value in values
        )
        actual_values = np.asarray(actual, dtype=np.float64)
        labels = np.asarray(actual_values > 0.0, dtype=np.int8)
        prevalence = float(np.mean(labels))
        order = np.argsort(-(predicted - epistemic), kind="stable")
        top_rows = []
        for requested in (100, 500, 1_000):
            count = min(requested, len(order))
            selected = order[:count]
            top_rows.append(
                {
                    "requested_rows": requested,
                    "actual_rows": count,
                    "mean_actual_net_bps": float(np.mean(actual_values[selected])),
                    "positive_rate": float(np.mean(actual_values[selected] > 0.0)),
                }
            )
        output[side] = {
            "rows": len(actual_values),
            "actual_positive_ratio": prevalence,
            "mean_actual_net_bps": float(np.mean(actual_values)),
            "mean_prediction_bps": float(np.mean(predicted)),
            "mean_absolute_error_bps": float(
                np.mean(np.abs(predicted - actual_values))
            ),
            "zero_baseline_mae_bps": float(np.mean(np.abs(actual_values))),
            "root_mean_squared_error_bps": float(
                np.sqrt(np.mean((predicted - actual_values) ** 2))
            ),
            "zero_baseline_rmse_bps": float(np.sqrt(np.mean(actual_values**2))),
            "pearson_information_coefficient": _correlation(actual_values, predicted),
            "profitable_auc": _auc(labels, probability),
            "profitable_brier": float(np.mean((probability - labels) ** 2)),
            "prevalence_brier": float(np.mean((prevalence - labels) ** 2)),
            "interval_80_coverage": float(
                np.mean((actual_values >= lower) & (actual_values <= upper))
            ),
            "interval_crossing_rate": float(np.mean(lower > upper)),
            "mean_epistemic_std_bps": float(np.mean(epistemic)),
            "top_rows": top_rows,
        }
    return {
        "scenario": scenario,
        "sides": output,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _empty_profile_trace(
    dataset,
    targets: AdaptiveBarrierTargets,
    score: ActionScoreBatch,
    *,
    scenario: str,
):
    return simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario=scenario,
        strength_threshold_bps=np.finfo(float).max,
    )


def _profile_spec(raw: Mapping[str, object]) -> ActionPolicySpec:
    return ActionPolicySpec(
        **{
            name: raw[name]
            for name in (
                "profile",
                "epistemic_penalty",
                "minimum_profitable_probability",
                "minimum_member_agreement",
                "maximum_epistemic_std_bps",
                "minimum_lower_bound_bps",
            )
        }
    )


def _iso_days(role: Mapping[str, object]) -> tuple[int, ...]:
    first = _day_id(role["start"], label="role start")
    last = _day_id(role["end"], label="role end")
    return tuple(range(first, last + 1))


def run_adaptive_action_screen(
    *,
    design_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_adaptive_action_design(design_path)
    resources = design["runtime_resources"]
    data = design["data"]
    execution = design["execution"]
    sampler = design["event_sampler"]
    training = design["training"]
    threshold_policy = design["threshold_policy"]
    terminal = design["reserved_terminal"]
    assert isinstance(resources, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(threshold_policy, Mapping)
    assert isinstance(terminal, Mapping)
    effective_memory = str(memory_limit or resources["duckdb_memory_limit"]).upper()
    effective_threads = int(threads or resources["warehouse_threads"])
    effective_backend = str(compute_backend or resources["compute_backend"]).lower()
    if (
        effective_memory != resources["duckdb_memory_limit"]
        or effective_threads != int(resources["warehouse_threads"])
        or effective_backend != resources["compute_backend"]
    ):
        raise ValueError("runtime overrides differ from the precommitted contract")
    backend = resolve_backend(effective_backend)
    if backend.kind != effective_backend:
        raise RuntimeError(
            "precommitted accelerator is unavailable; CPU fallback is forbidden"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    runtime = {
        "duckdb_memory_limit": effective_memory,
        "warehouse_threads": effective_threads,
        "compute_backend_requested": effective_backend,
        "compute_backend_kind": backend.kind,
        "compute_backend_device": backend.device,
        "compute_backend_vendor": backend.vendor,
        "training_worker_isolation": "clean_process",
        "cpu_fallback_permitted": False,
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
            "adaptive-action "
            + " ".join(
                f"{name}={value}"
                for name, value in payload.items()
                if name != "runtime_resources"
            ),
            flush=True,
        )
        write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    progress("initialize")
    corpus = _load_corpus(
        design=design,
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
        progress=progress,
    )
    dataset = corpus["dataset"]
    event_mask = causal_cusum_event_mask(
        dataset,
        volatility_multiplier=float(sampler["volatility_multiplier"]),
        minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
    )
    event_indexes = np.flatnonzero(event_mask).astype(np.int64)
    barrier_spec = AdaptiveBarrierSpec(**dict(design["barrier_targets"]))
    progress("barrier-target-build-start", event_rows=len(event_indexes))
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
    ) as warehouse:
        targets = build_adaptive_barrier_targets(
            warehouse,
            dataset,
            event_indexes,
            spec=barrier_spec,
            progress=lambda day, total, valid: progress(
                "barrier-target-day",
                day=day,
                days=total,
                valid_rows=valid,
            ),
        )
    targets_sha256 = _targets_sha256(targets)
    roles, role_evidence = _role_indexes(
        dataset,
        targets,
        event_mask,
        data["roles"],
        _day_id(terminal["date"], label="terminal"),
    )
    progress(
        "dataset-ready",
        dataset_rows=dataset.rows,
        event_rows=len(event_indexes),
        valid_target_rows=targets.valid_rows,
        barrier_targets_sha256=targets_sha256,
        cache_state=corpus["cache_state"],
    )
    valid_positions = np.flatnonzero(targets.valid)
    valid_source = targets.source_indexes[valid_positions]
    max_base_exit = np.maximum(
        targets.base_long_exit_time_ms[valid_positions],
        targets.base_short_exit_time_ms[valid_positions],
    )
    exit_full = np.full(dataset.rows, -1, dtype=np.int64)
    exit_full[valid_source] = max_base_exit
    train_weights = average_label_uniqueness(
        dataset.decision_time_ms, exit_full, roles["train"]
    )
    tuning_weights = average_label_uniqueness(
        dataset.decision_time_ms, exit_full, roles["early_stop"]
    )
    model_spec = ActionValueArchitectureSpec(**dict(design["model"]))
    models: list[TrainedActionValueModel] = []
    artifacts: list[dict[str, object]] = []
    seeds = tuple(int(value) for value in training["ensemble_seeds"])
    for member, seed in enumerate(seeds, start=1):
        progress("model-start", member=member, members=len(seeds), seed=seed)
        member_spec = replace(
            model_spec, candidate_id=f"{model_spec.candidate_id}-seed-{seed}"
        )
        model = train_action_value_model(
            dataset,
            targets,
            train_endpoints=roles["train"],
            tuning_endpoints=roles["early_stop"],
            spec=member_spec,
            target_scenario=str(training["target_scenario"]),
            compute_backend=effective_backend,
            seed=seed,
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            train_sample_weights=train_weights,
            tuning_sample_weights=tuning_weights,
            progress=lambda epoch, total, training_loss, tuning_loss, index=member, model_seed=seed: (
                progress(
                    "model-epoch",
                    member=index,
                    seed=model_seed,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                )
            ),
        )
        if model.backend_kind != effective_backend:
            raise RuntimeError(
                "model training did not remain on the precommitted backend"
            )
        artifact_path = destination / "models" / f"seed-{seed}.safetensors"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = _save_neural_artifact(artifact_path, model)
        artifact["path"] = artifact_path.relative_to(destination).as_posix()
        summary = _artifact_summary(model)
        summary.update(
            {
                "target_schema_version": model.target_schema_version,
                "target_scenario": model.target_scenario,
                "target_contract_sha256": model.target_contract_sha256,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        artifacts.append({"seed": seed, "model": summary, "artifact": artifact})
        models.append(model)
        progress(
            "model-complete",
            member=member,
            seed=seed,
            best_epoch=model.best_epoch,
            tuning_loss=round(model.tuning_loss, 8),
            model_sha256=model.model_sha256,
        )
    batch_size = int(training["batch_size"])
    progress("calibration-predict")
    calibration_prediction = _ensemble_for_role(
        models,
        dataset,
        roles["calibration"],
        compute_backend=effective_backend,
        batch_size=batch_size,
    )
    progress("policy-predict")
    policy_prediction = _ensemble_for_role(
        models,
        dataset,
        roles["policy"],
        compute_backend=effective_backend,
        batch_size=batch_size,
    )
    roles_raw = data["roles"]
    assert isinstance(roles_raw, Mapping)
    calibration_days = _iso_days(roles_raw["calibration"])
    policy_days = _iso_days(roles_raw["policy"])
    development_days = _iso_days(roles_raw["development_evaluation"])
    profile_results: list[dict[str, object]] = []
    policy_survivors: list[str] = []
    for raw_profile in design["risk_profiles"]:
        assert isinstance(raw_profile, Mapping)
        spec = _profile_spec(raw_profile)
        calibration_score = derive_action_scores(calibration_prediction, spec)
        policy_score = derive_action_scores(policy_prediction, spec)
        selection = select_barrier_threshold(
            dataset,
            targets,
            calibration_score,
            quantiles=tuple(float(value) for value in threshold_policy["quantiles"]),
            expected_days=calibration_days,
            gates=raw_profile["calibration_gates"],
            drawdown_penalty=float(threshold_policy["drawdown_penalty"]),
        )
        policy_reasons: list[str] = []
        if selection.accepted:
            assert selection.threshold_bps is not None
            policy_base = simulate_barrier_action_trace(
                dataset,
                targets,
                policy_score,
                scenario="base",
                strength_threshold_bps=selection.threshold_bps,
            )
            policy_stress = simulate_barrier_action_trace(
                dataset,
                targets,
                policy_score,
                scenario="stress",
                strength_threshold_bps=selection.threshold_bps,
            )
            policy_reasons = barrier_trace_gate_reasons(
                policy_stress,
                expected_days=policy_days,
                gates=raw_profile["policy_gates"],
            )
        else:
            policy_base = _empty_profile_trace(
                dataset, targets, policy_score, scenario="base"
            )
            policy_stress = _empty_profile_trace(
                dataset, targets, policy_score, scenario="stress"
            )
            policy_reasons = ["calibration_threshold_rejected"]
        policy_passed = not policy_reasons
        if policy_passed:
            policy_survivors.append(spec.profile)
        profile_results.append(
            {
                "profile": spec.profile,
                "policy_spec": asdict(spec),
                "calibration_eligible_rows": int(np.sum(calibration_score.eligible)),
                "threshold_selection": selection.asdict(),
                "policy_eligible_rows": int(np.sum(policy_score.eligible)),
                "policy_base_trace": policy_base.asdict(),
                "policy_stress_trace": policy_stress.asdict(),
                "policy_status": "research_candidate" if policy_passed else "rejected",
                "policy_rejection_reasons": policy_reasons,
                "development_evaluated": False,
                "development_result": None,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        progress(
            "profile-policy-complete",
            profile=spec.profile,
            threshold_accepted=selection.accepted,
            policy_passed=policy_passed,
            stress_trades=policy_stress.metrics.trades,
            stress_total_net_bps=round(policy_stress.metrics.total_net_bps, 6),
        )
    development_prediction = None
    if policy_survivors:
        progress("development-predict", profiles=",".join(policy_survivors))
        development_prediction = _ensemble_for_role(
            models,
            dataset,
            roles["development_evaluation"],
            compute_backend=effective_backend,
            batch_size=batch_size,
        )
        by_profile = {str(value["profile"]): value for value in profile_results}
        profiles_by_name = {
            str(value["profile"]): value for value in design["risk_profiles"]
        }
        for profile in policy_survivors:
            raw_profile = profiles_by_name[profile]
            assert isinstance(raw_profile, Mapping)
            result = by_profile[profile]
            selection = result["threshold_selection"]
            assert isinstance(selection, Mapping)
            threshold = float(selection["threshold_bps"])
            development_score = derive_action_scores(
                development_prediction, _profile_spec(raw_profile)
            )
            development_base = simulate_barrier_action_trace(
                dataset,
                targets,
                development_score,
                scenario="base",
                strength_threshold_bps=threshold,
            )
            development_stress = simulate_barrier_action_trace(
                dataset,
                targets,
                development_score,
                scenario="stress",
                strength_threshold_bps=threshold,
            )
            reasons = barrier_trace_gate_reasons(
                development_stress,
                expected_days=development_days,
                gates=raw_profile["development_gates"],
            )
            result["development_evaluated"] = True
            result["development_result"] = {
                "eligible_rows": int(np.sum(development_score.eligible)),
                "base_trace": development_base.asdict(),
                "stress_trace": development_stress.asdict(),
                "status": "research_candidate" if not reasons else "rejected",
                "rejection_reasons": reasons,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
            progress(
                "profile-development-complete",
                profile=profile,
                passed=not reasons,
                stress_trades=development_stress.metrics.trades,
                stress_total_net_bps=round(development_stress.metrics.total_net_bps, 6),
            )
    final_profiles = [
        str(value["profile"])
        for value in profile_results
        if isinstance(value.get("development_result"), Mapping)
        and value["development_result"].get("status") == "research_candidate"
    ]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "consumed_data_adaptive_barrier_action_value_evidence",
        "status": "research_candidate" if final_profiles else "rejected",
        "round": 16,
        "design_sha256": design_sha256,
        "action_value_model_schema_version": ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION,
        "action_policy_schema_version": ACTION_POLICY_SCHEMA_VERSION,
        "barrier_schema_version": ADAPTIVE_BARRIER_SCHEMA_VERSION,
        "target_mode": ADAPTIVE_BARRIER_TARGET_MODE,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": development_prediction is not None,
        "runtime_resources": runtime,
        "corpus_certificate_sha256": corpus["certificate"]["certificate_sha256"],
        "dataset": {
            "rows": dataset.rows,
            "event_rows": len(event_indexes),
            "valid_barrier_rows": targets.valid_rows,
            "cache_key": corpus["cache_key"],
            "cache_state": corpus["cache_state"],
            "source_manifest_fingerprint": corpus["source_evidence"][
                "manifest_fingerprint"
            ],
            "barrier_targets_sha256": targets_sha256,
            "barrier_summary": targets.summary(),
            "roles": role_evidence,
        },
        "ensemble_models": artifacts,
        "forecast_diagnostics": {
            "calibration_base": _forecast_diagnostics(
                targets, calibration_prediction, scenario="base"
            ),
            "calibration_stress": _forecast_diagnostics(
                targets, calibration_prediction, scenario="stress"
            ),
            "policy_base": _forecast_diagnostics(
                targets, policy_prediction, scenario="base"
            ),
            "policy_stress": _forecast_diagnostics(
                targets, policy_prediction, scenario="stress"
            ),
            "development_base": (
                _forecast_diagnostics(targets, development_prediction, scenario="base")
                if development_prediction is not None
                else None
            ),
            "development_stress": (
                _forecast_diagnostics(
                    targets, development_prediction, scenario="stress"
                )
                if development_prediction is not None
                else None
            ),
        },
        "profile_results": profile_results,
        "policy_survivors": policy_survivors,
        "final_profiles": final_profiles,
        "limitations": [
            "the certified exact-BBO corpus spans weeks rather than the multi-year target",
            "the 100 ms BBO path cannot resolve queue position or hidden depth",
            "base and adverse scenarios are research replays, not fill guarantees",
            "all returns are unleveraged and no profile may apply leverage before edge validation",
            "the local neural ensemble is machine learning and is not the optional LLM risk-assessment overlay",
            "the reserved terminal date was neither loaded nor labeled",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    progress(
        "complete",
        status=report["status"],
        report_sha256=report["report_sha256"],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the precommitted adaptive action-value screen"
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--compute-backend")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_adaptive_action_screen(
        design_path=args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(
        "adaptive-action-screen: "
        f"status={report['status']} sha256={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
