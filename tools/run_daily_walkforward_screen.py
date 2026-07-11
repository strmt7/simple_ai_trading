"""Run a precommitted causal daily-refit gross-model screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
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

from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    GROSS_ARCHITECTURE_SCHEMA_VERSION,
    GROSS_TARGET_MODE,
    GrossArchitectureSpec,
    average_label_uniqueness,
    causal_cusum_event_mask,
    derive_gross_action_scores,
    evaluate_gross_action_scores,
    evaluate_gross_forecast,
    predict_torch_gross_model,
    train_torch_gross_model,
)
from simple_ai_trading.microstructure_model import _trading_metrics  # noqa: E402
from simple_ai_trading.microstructure_walkforward import (  # noqa: E402
    ActionTrace,
    WalkForwardFitSpec,
    plan_walk_forward_day,
    recency_weighted_uniqueness,
    select_calibrated_threshold,
    simulate_action_trace,
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


DESIGN_SCHEMA_VERSION = "daily-walk-forward-screen-design-v1"
REPORT_SCHEMA_VERSION = "daily-walk-forward-screen-report-v1"
_DAY_MS = 86_400_000
_REQUIRED_GATES = {
    "minimum_trades",
    "minimum_total_net_bps",
    "maximum_drawdown_bps",
    "minimum_positive_day_ratio",
    "minimum_worst_trade_bps",
}


def _day_id(value: object, *, label: str) -> int:
    parsed = _parse_date(value, label=label)
    return int(
        datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1_000
        // _DAY_MS
    )


def load_daily_walkforward_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("daily walk-forward design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("daily walk-forward design must be an object")
    claimed = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("daily walk-forward design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != 15
        or payload.get("purpose") != "consumed_data_daily_walk_forward_development"
        or payload.get("target_mode") != GROSS_TARGET_MODE
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("daily walk-forward design contract is invalid")
    required_sections = (
        "implementation",
        "predecessor_evidence",
        "data",
        "execution",
        "runtime_resources",
        "event_sampler",
        "model",
        "training",
        "threshold_policy",
        "policy_gates",
        "development_gates",
        "evaluation",
        "reserved_terminal",
    )
    sections = {name: payload.get(name) for name in required_sections}
    if not all(isinstance(value, Mapping) for value in sections.values()):
        raise ValueError("daily walk-forward design sections are incomplete")
    fit_candidates = payload.get("fit_candidates")
    if not isinstance(fit_candidates, list):
        raise ValueError("daily walk-forward fit candidates are missing")
    implementation = sections["implementation"]
    predecessor = sections["predecessor_evidence"]
    data = sections["data"]
    execution = sections["execution"]
    resources = sections["runtime_resources"]
    sampler = sections["event_sampler"]
    model = sections["model"]
    training = sections["training"]
    threshold = sections["threshold_policy"]
    policy_gates = sections["policy_gates"]
    development_gates = sections["development_gates"]
    evaluation = sections["evaluation"]
    terminal = sections["reserved_terminal"]
    assert isinstance(implementation, Mapping)
    assert isinstance(predecessor, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(threshold, Mapping)
    assert isinstance(policy_gates, Mapping)
    assert isinstance(development_gates, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(terminal, Mapping)
    if require_current:
        _validate_implementation_binding(implementation)
    if (
        predecessor.get("round") != 14
        or predecessor.get("design_sha256")
        != "b043639d7242bd599371fd4ee44283f46dbef5663129ebd04538792b8f582a0d"
        or predecessor.get("report_sha256")
        != "b0b0323c167c90dec1353409aadb631479f14315f42f7e2f9ca3aba642a32fa7"
    ):
        raise ValueError("daily walk-forward predecessor evidence is invalid")
    roles = data.get("roles")
    if (
        data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not False
        or not isinstance(roles, Mapping)
        or data.get("start_date") != "2023-05-16"
        or data.get("end_date") != "2023-07-06"
        or tuple(roles)
        != (
            "train",
            "early_stop",
            "calibration",
            "policy",
            "development_evaluation",
        )
    ):
        raise ValueError("daily walk-forward data contract is invalid")
    expected_roles = {
        "train": {"start": "2023-05-16", "end": "2023-06-15"},
        "early_stop": {"start": "2023-06-16", "end": "2023-06-20"},
        "calibration": {"start": "2023-06-21", "end": "2023-06-25"},
        "policy": {"start": "2023-06-26", "end": "2023-06-30"},
        "development_evaluation": {
            "start": "2023-07-01",
            "end": "2023-07-06",
        },
    }
    if roles != expected_roles:
        raise ValueError("daily walk-forward source roles drifted")
    if (
        int(execution.get("horizon_seconds") or 0) != 300
        or int(execution.get("total_latency_ms") or -1) != 750
        or float(execution.get("taker_fee_bps_per_side") or -1.0) != 5.0
        or float(execution.get("additional_slippage_bps_per_side") or -1.0) != 1.0
        or int(execution.get("decision_cadence_seconds") or 0) != 5
        or int(execution.get("max_quote_age_ms") or 0) != 1_000
        or float(execution.get("reference_order_notional_quote") or 0.0) != 1_000.0
        or float(execution.get("max_l1_participation") or 0.0) != 1.0
    ):
        raise ValueError("daily walk-forward execution contract is invalid")
    if (
        resources.get("duckdb_memory_limit") != "4GB"
        or int(resources.get("warehouse_threads") or 0) != 8
        or resources.get("compute_backend") != "directml"
        or resources.get("spill_directory_policy") != "warehouse_adjacent"
    ):
        raise ValueError("daily walk-forward resource contract is invalid")
    if (
        sampler.get("method") != "daily_reset_causal_cusum"
        or float(sampler.get("volatility_multiplier") or 0.0) <= 0.0
        or float(sampler.get("minimum_threshold_bps") or 0.0) <= 0.0
        or sampler.get("minimum_activity_quota") is not None
    ):
        raise ValueError("daily walk-forward event sampler is invalid")
    base_model = GrossArchitectureSpec(**dict(model))
    if (
        base_model.family != "tabular_mlp"
        or base_model.gmadl_weight != 0.0
        or base_model.head_coherence_weight != 0.0
    ):
        raise ValueError("daily walk-forward base model is invalid")
    parsed_candidates = [WalkForwardFitSpec(**dict(value)) for value in fit_candidates]
    if len(parsed_candidates) != 3 or len(
        {value.candidate_id for value in parsed_candidates}
    ) != len(parsed_candidates):
        raise ValueError("daily walk-forward candidate contract is invalid")
    if (
        int(training.get("batch_size") or 0) < 1_024
        or int(training.get("max_epochs") or 0) < 2
        or int(training.get("patience") or 0) < 1
        or training.get("action_score_method") != "direction_magnitude"
        or int(training.get("minimum_role_rows") or 0) < 256
    ):
        raise ValueError("daily walk-forward training contract is invalid")
    quantiles = threshold.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or len(quantiles) < 3
        or len(set(float(value) for value in quantiles)) != len(quantiles)
        or any(not 0.0 < float(value) < 1.0 for value in quantiles)
        or int(threshold.get("minimum_calibration_trades") or 0) < 1
        or float(threshold.get("maximum_calibration_drawdown_bps") or 0.0) <= 0.0
        or not 0.0 <= float(threshold.get("minimum_positive_day_ratio") or -1.0) <= 1.0
        or float(threshold.get("drawdown_penalty") or -1.0) < 0.0
    ):
        raise ValueError("daily walk-forward threshold contract is invalid")
    if (
        set(policy_gates) != _REQUIRED_GATES
        or set(development_gates) != _REQUIRED_GATES
    ):
        raise ValueError("daily walk-forward aggregate gates are incomplete")
    for label, gates in (
        ("policy", policy_gates),
        ("development", development_gates),
    ):
        numeric = tuple(float(gates[name]) for name in _REQUIRED_GATES)
        if (
            not all(math.isfinite(value) for value in numeric)
            or int(gates["minimum_trades"]) < 1
            or float(gates["minimum_total_net_bps"]) < 0.0
            or float(gates["maximum_drawdown_bps"]) <= 0.0
            or not 0.0 <= float(gates["minimum_positive_day_ratio"]) <= 1.0
            or float(gates["minimum_worst_trade_bps"]) >= 0.0
        ):
            raise ValueError(f"daily walk-forward {label} gates are invalid")
    policy_start = _day_id(evaluation.get("policy_start"), label="policy start")
    policy_end = _day_id(evaluation.get("policy_end"), label="policy end")
    development_start = _day_id(
        evaluation.get("development_start"),
        label="development start",
    )
    development_end = _day_id(
        evaluation.get("development_end"),
        label="development end",
    )
    if (
        policy_start > policy_end
        or development_start != policy_end + 1
        or development_start > development_end
        or evaluation.get("policy_start") != "2023-06-26"
        or evaluation.get("policy_end") != "2023-06-30"
        or evaluation.get("development_start") != "2023-07-01"
        or evaluation.get("development_end") != "2023-07-06"
        or evaluation.get("development_used_for_candidate_selection") is not False
    ):
        raise ValueError("daily walk-forward evaluation roles are invalid")
    data_end = _day_id(data.get("end_date"), label="data end")
    terminal_day = _day_id(terminal.get("date"), label="terminal")
    if (
        development_end != data_end
        or terminal_day != development_end + 1
        or terminal.get("included_in_dataset") is not False
        or terminal.get("access_permitted") is not False
    ):
        raise ValueError("daily walk-forward terminal contract is invalid")
    return payload, str(claimed)


def _positive_day_ratio(
    traces: Sequence[ActionTrace], expected_days: Sequence[int]
) -> float:
    daily = {int(day): 0.0 for day in expected_days}
    for trace in traces:
        for timestamp, value in zip(
            trace.timestamps_ms,
            trace.net_bps,
            strict=True,
        ):
            daily[int(timestamp) // _DAY_MS] += float(value)
    return float(np.mean(np.asarray(list(daily.values()), dtype=np.float64) > 0.0))


def _aggregate_traces(
    traces: Sequence[ActionTrace],
    *,
    expected_days: Sequence[int],
) -> dict[str, object]:
    pnls = tuple(value for trace in traces for value in trace.net_bps)
    gross = tuple(value for trace in traces for value in trace.gross_bps)
    sides = tuple(value for trace in traces for value in trace.sides)
    timestamps = tuple(value for trace in traces for value in trace.timestamps_ms)
    metrics = _trading_metrics(pnls, sides, timestamps)
    if len(gross) != metrics.trades:
        raise ValueError("daily walk-forward gross/net trace count drifted")
    expected = tuple(int(day) for day in expected_days)
    active_days = {int(value) // _DAY_MS for value in timestamps}
    return {
        "metrics": asdict(metrics),
        "total_gross_bps": float(np.sum(gross)) if gross else 0.0,
        "mean_gross_bps": float(np.mean(gross)) if gross else 0.0,
        "positive_day_ratio": _positive_day_ratio(traces, expected),
        "calendar_days": len(expected),
        "abstention_days": len(set(expected) - active_days),
        "portfolio_claim": False,
        "leverage_applied": False,
        "trading_authority": False,
    }


def _aggregate_gate_reasons(
    aggregate: Mapping[str, object],
    gates: Mapping[str, object],
) -> list[str]:
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("daily walk-forward aggregate metrics are missing")
    reasons: list[str] = []
    if int(metrics["trades"]) < int(gates["minimum_trades"]):
        reasons.append("minimum_trades_not_met")
    if float(metrics["total_net_bps"]) <= float(gates["minimum_total_net_bps"]):
        reasons.append("total_net_gate_failed")
    if float(metrics["max_drawdown_bps"]) > float(gates["maximum_drawdown_bps"]):
        reasons.append("drawdown_gate_failed")
    if float(aggregate["positive_day_ratio"]) < float(
        gates["minimum_positive_day_ratio"]
    ):
        reasons.append("positive_day_ratio_gate_failed")
    if int(metrics["trades"]) > 0 and float(metrics["worst_trade_bps"]) < float(
        gates["minimum_worst_trade_bps"]
    ):
        reasons.append("worst_trade_gate_failed")
    return reasons


def _selection_rank(aggregate: Mapping[str, object], drawdown_penalty: float):
    metrics = aggregate["metrics"]
    assert isinstance(metrics, Mapping)
    utility = float(metrics["total_net_bps"]) - float(drawdown_penalty) * float(
        metrics["max_drawdown_bps"]
    )
    return (
        utility,
        float(metrics["mean_net_bps"]),
        float(aggregate["positive_day_ratio"]),
        int(metrics["trades"]),
    )


def _iso_day(day_id: int) -> str:
    return (
        datetime.fromtimestamp(
            int(day_id) * _DAY_MS / 1_000.0,
            tz=timezone.utc,
        )
        .date()
        .isoformat()
    )


def _run_candidate_days(
    *,
    dataset,
    target: np.ndarray,
    event_mask: np.ndarray,
    corpus_start_day: int,
    day_ids: Sequence[int],
    fit_spec: WalkForwardFitSpec,
    model_spec: GrossArchitectureSpec,
    design: Mapping[str, object],
    compute_backend: str,
    destination: Path,
    phase: str,
    progress,
) -> tuple[list[dict[str, object]], list[ActionTrace]]:
    training = design["training"]
    threshold_policy = design["threshold_policy"]
    assert isinstance(training, Mapping)
    assert isinstance(threshold_policy, Mapping)
    rows: list[dict[str, object]] = []
    traces: list[ActionTrace] = []
    for position, day_id in enumerate(day_ids, start=1):
        day = _iso_day(day_id)
        progress(
            f"{phase}-day-start",
            candidate=fit_spec.candidate_id,
            day=day,
            day_index=position,
            day_count=len(day_ids),
        )
        plan = plan_walk_forward_day(
            dataset,
            event_mask,
            evaluation_day_id=day_id,
            corpus_start_day_id=corpus_start_day,
            spec=fit_spec,
            minimum_rows=int(training["minimum_role_rows"]),
        )
        train_uniqueness = average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            plan.train_indexes,
        )
        stop_uniqueness = average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            plan.early_stop_indexes,
        )
        train_weights = recency_weighted_uniqueness(
            dataset,
            plan.train_indexes,
            train_uniqueness,
            half_life_days=fit_spec.recency_half_life_days,
        )
        stop_weights = recency_weighted_uniqueness(
            dataset,
            plan.early_stop_indexes,
            stop_uniqueness,
            half_life_days=fit_spec.recency_half_life_days,
        )
        daily_spec = replace(
            model_spec,
            candidate_id=f"{fit_spec.candidate_id}-{day}",
        )
        model = train_torch_gross_model(
            dataset,
            target,
            train_endpoints=plan.train_indexes,
            tuning_endpoints=plan.early_stop_indexes,
            spec=daily_spec,
            compute_backend=compute_backend,
            seed=int(design["seed"]) + position,
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            train_sample_weights=train_weights,
            tuning_sample_weights=stop_weights,
            progress=lambda epoch, total, training_loss, tuning_loss, candidate=fit_spec.candidate_id, fit_day=day: (
                progress(
                    f"{phase}-epoch",
                    candidate=candidate,
                    day=fit_day,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                )
            ),
        )
        calibration_prediction = predict_torch_gross_model(
            model,
            dataset,
            plan.calibration_indexes,
            compute_backend=compute_backend,
            batch_size=int(training["batch_size"]),
        )
        calibration_score = derive_gross_action_scores(
            calibration_prediction,
            method=str(training["action_score_method"]),
        )
        threshold = select_calibrated_threshold(
            dataset,
            target,
            calibration_score,
            quantiles=tuple(float(value) for value in threshold_policy["quantiles"]),
            minimum_trades=int(threshold_policy["minimum_calibration_trades"]),
            maximum_drawdown_bps=float(
                threshold_policy["maximum_calibration_drawdown_bps"]
            ),
            minimum_positive_day_ratio=float(
                threshold_policy["minimum_positive_day_ratio"]
            ),
            drawdown_penalty=float(threshold_policy["drawdown_penalty"]),
        )
        evaluation_prediction = predict_torch_gross_model(
            model,
            dataset,
            plan.evaluation_indexes,
            compute_backend=compute_backend,
            batch_size=int(training["batch_size"]),
        )
        evaluation_score = derive_gross_action_scores(
            evaluation_prediction,
            method=str(training["action_score_method"]),
        )
        evaluation_forecast = evaluate_gross_forecast(
            dataset,
            target,
            evaluation_prediction,
        ).asdict()
        evaluation_action = evaluate_gross_action_scores(
            dataset,
            target,
            evaluation_prediction,
            evaluation_score,
        ).asdict()
        if threshold.accepted:
            assert threshold.threshold is not None
            trace = simulate_action_trace(
                dataset,
                target,
                evaluation_score,
                strength_threshold=threshold.threshold,
            )
        else:
            trace = threshold.selected_trace
        artifact_path = (
            destination
            / "models"
            / f"{phase}-{fit_spec.candidate_id}-{day}.safetensors"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_file = _save_neural_artifact(artifact_path, model)
        artifact_file["path"] = artifact_path.relative_to(destination).as_posix()
        rows.append(
            {
                "candidate_id": fit_spec.candidate_id,
                "evaluation_day": day,
                "plan": dict(plan.evidence),
                "model": _artifact_summary(model),
                "model_artifact": artifact_file,
                "threshold_selection": threshold.asdict(),
                "evaluation_forecast_metrics": evaluation_forecast,
                "evaluation_action_diagnostics": evaluation_action,
                "evaluation_trace": trace.asdict(),
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        traces.append(trace)
        progress(
            f"{phase}-day-complete",
            candidate=fit_spec.candidate_id,
            day=day,
            threshold_accepted=threshold.accepted,
            trades=trace.metrics.trades,
            total_net_bps=round(trace.metrics.total_net_bps, 6),
        )
        del model, calibration_prediction, evaluation_prediction
    return rows, traces


def run_daily_walkforward_screen(
    *,
    design_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_daily_walkforward_design(design_path)
    resources = design["runtime_resources"]
    evaluation = design["evaluation"]
    policy_gates = design["policy_gates"]
    development_gates = design["development_gates"]
    threshold_policy = design["threshold_policy"]
    assert isinstance(resources, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(policy_gates, Mapping)
    assert isinstance(development_gates, Mapping)
    assert isinstance(threshold_policy, Mapping)
    effective_memory = str(memory_limit or resources["duckdb_memory_limit"]).upper()
    effective_threads = int(threads or resources["warehouse_threads"])
    effective_backend = str(compute_backend or resources["compute_backend"]).lower()
    if (
        effective_memory != resources["duckdb_memory_limit"]
        or effective_threads != int(resources["warehouse_threads"])
        or effective_backend != resources["compute_backend"]
    ):
        raise ValueError("runtime overrides differ from the precommitted contract")
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
            "daily-walk-forward "
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
    target = corpus["target"]
    assert isinstance(target, np.ndarray)
    sampler = design["event_sampler"]
    assert isinstance(sampler, Mapping)
    event_mask = causal_cusum_event_mask(
        dataset,
        volatility_multiplier=float(sampler["volatility_multiplier"]),
        minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
    )
    data = design["data"]
    assert isinstance(data, Mapping)
    corpus_start_day = _day_id(data["start_date"], label="corpus start")
    policy_days = tuple(
        range(
            _day_id(evaluation["policy_start"], label="policy start"),
            _day_id(evaluation["policy_end"], label="policy end") + 1,
        )
    )
    development_days = tuple(
        range(
            _day_id(evaluation["development_start"], label="development start"),
            _day_id(evaluation["development_end"], label="development end") + 1,
        )
    )
    progress(
        "dataset-ready",
        dataset_rows=dataset.rows,
        event_rows=int(np.sum(event_mask)),
        cache_state=corpus["cache_state"],
        policy_days=len(policy_days),
        development_days=len(development_days),
    )
    fit_specs = [
        WalkForwardFitSpec(**dict(value)) for value in design["fit_candidates"]
    ]
    model_spec = GrossArchitectureSpec(**dict(design["model"]))
    policy_results: list[dict[str, object]] = []
    for fit_spec in fit_specs:
        daily_rows, traces = _run_candidate_days(
            dataset=dataset,
            target=target,
            event_mask=event_mask,
            corpus_start_day=corpus_start_day,
            day_ids=policy_days,
            fit_spec=fit_spec,
            model_spec=model_spec,
            design=design,
            compute_backend=effective_backend,
            destination=destination,
            phase="policy",
            progress=progress,
        )
        aggregate = _aggregate_traces(traces, expected_days=policy_days)
        reasons = _aggregate_gate_reasons(aggregate, policy_gates)
        policy_results.append(
            {
                "candidate": asdict(fit_spec),
                "daily_results": daily_rows,
                "aggregate": aggregate,
                "selection_rank": list(
                    _selection_rank(
                        aggregate,
                        float(threshold_policy["drawdown_penalty"]),
                    )
                ),
                "status": "research_candidate" if not reasons else "rejected",
                "rejection_reasons": reasons,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
    policy_results.sort(
        key=lambda value: tuple(value["selection_rank"]),
        reverse=True,
    )
    selected_policy = policy_results[0]
    selected_id = str(selected_policy["candidate"]["candidate_id"])
    selected_spec = next(
        value for value in fit_specs if value.candidate_id == selected_id
    )
    progress(
        "policy-selection-complete",
        selected=selected_id,
        selected_status=selected_policy["status"],
    )
    development_rows, development_traces = _run_candidate_days(
        dataset=dataset,
        target=target,
        event_mask=event_mask,
        corpus_start_day=corpus_start_day,
        day_ids=development_days,
        fit_spec=selected_spec,
        model_spec=model_spec,
        design=design,
        compute_backend=effective_backend,
        destination=destination,
        phase="development",
        progress=progress,
    )
    development_aggregate = _aggregate_traces(
        development_traces,
        expected_days=development_days,
    )
    development_reasons = _aggregate_gate_reasons(
        development_aggregate,
        development_gates,
    )
    selected_policy_passed = selected_policy["status"] == "research_candidate"
    if not selected_policy_passed:
        development_reasons = [
            "selected_policy_candidate_was_rejected",
            *development_reasons,
        ]
    final_status = "research_candidate" if not development_reasons else "rejected"
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "consumed_data_daily_walk_forward_development_evidence",
        "status": final_status,
        "round": 15,
        "design_sha256": design_sha256,
        "gross_model_schema_version": GROSS_ARCHITECTURE_SCHEMA_VERSION,
        "target_mode": GROSS_TARGET_MODE,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
        "runtime_resources": runtime,
        "corpus_certificate_sha256": corpus["certificate"]["certificate_sha256"],
        "dataset": {
            "rows": dataset.rows,
            "event_rows": int(np.sum(event_mask)),
            "cache_key": corpus["cache_key"],
            "cache_state": corpus["cache_state"],
            "source_manifest_fingerprint": corpus["source_evidence"][
                "manifest_fingerprint"
            ],
            "gross_target_mean_bps": float(np.mean(target)),
            "gross_target_std_bps": float(np.std(target)),
        },
        "policy_results": policy_results,
        "selected_policy_candidate_id": selected_id,
        "selected_policy_candidate_status": selected_policy["status"],
        "development_result": {
            "candidate": asdict(selected_spec),
            "daily_results": development_rows,
            "aggregate": development_aggregate,
            "status": final_status,
            "rejection_reasons": development_reasons,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
        "limitations": [
            "the exact-BBO corpus spans weeks rather than the multi-year terminal standard",
            "fixed-horizon traces do not yet include an intrahorizon stop-loss path",
            "all policy and development dates were consumed before Round 15",
            "daily results are research traces and cannot establish profitability or authorize trading",
            "the reserved terminal date was neither loaded nor labeled",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    progress("complete", report_sha256=report["report_sha256"], status=report["status"])
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the precommitted daily walk-forward model screen",
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
    report = run_daily_walkforward_screen(
        design_path=args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(
        "daily-walk-forward-screen: "
        f"status={report['status']} sha256={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
