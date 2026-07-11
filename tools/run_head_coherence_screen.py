"""Run the precommitted gross-head coherence and action-score screen."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    GROSS_ACTION_SCORE_METHODS,
    GROSS_ARCHITECTURE_SCHEMA_VERSION,
    GROSS_TARGET_MODE,
    GrossArchitectureSpec,
    average_label_uniqueness,
    causal_cusum_event_mask,
    derive_gross_action_scores,
    evaluate_gross_action_scores,
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
    MICROSTRUCTURE_FEATURE_VERSION,
    build_executable_microstructure_dataset,
    microstructure_feature_source_contract,
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
        _role_indexes,
        _save_neural_artifact,
        _sha256_file,
        _utc_day_bounds,
        _validate_implementation_binding,
    )
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from run_gross_architecture_screen import (
        _artifact_summary,
        _canonical_sha256,
        _is_sha256,
        _parse_date,
        _role_indexes,
        _save_neural_artifact,
        _sha256_file,
        _utc_day_bounds,
        _validate_implementation_binding,
    )


DESIGN_SCHEMA_VERSION = "head-coherence-screen-design-v1"
REPORT_SCHEMA_VERSION = "head-coherence-screen-report-v1"
_ROLE_NAMES = (
    "train",
    "early_stop",
    "calibration",
    "policy",
    "development_evaluation",
)
_REQUIRED_SCORE_METHODS = {
    "mean",
    "direction_confidence",
    "direction_magnitude",
    "head_consensus",
    "conservative_quantile",
}
_REQUIRED_GATES = {
    "minimum_development_direction_auc",
    "minimum_development_spearman_ic",
    "require_development_mae_better_than_zero",
    "minimum_policy_active_rows",
    "minimum_development_active_rows",
    "minimum_policy_top_500_signed_gross_bps",
    "minimum_development_top_500_signed_gross_bps",
    "minimum_policy_top_500_exact_after_cost_bps",
    "minimum_development_top_500_exact_after_cost_bps",
}


def load_head_coherence_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("head-coherence design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("head-coherence design must be an object")
    claimed = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("head-coherence design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != 14
        or payload.get("purpose") != "consumed_data_head_coherence_development"
        or payload.get("target_mode") != GROSS_TARGET_MODE
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("head-coherence design contract is invalid")
    sections = {
        name: payload.get(name)
        for name in (
            "implementation",
            "predecessor_evidence",
            "data",
            "execution",
            "runtime_resources",
            "event_sampler",
            "stages",
            "ranking",
            "development_gates",
            "reserved_terminal",
        )
    }
    if not all(isinstance(value, Mapping) for value in sections.values()):
        raise ValueError("head-coherence design sections are incomplete")
    candidates = payload.get("neural_candidates")
    score_methods = payload.get("action_score_methods")
    if not isinstance(candidates, list) or not isinstance(score_methods, list):
        raise ValueError("head-coherence candidates are incomplete")
    implementation = sections["implementation"]
    predecessor = sections["predecessor_evidence"]
    data = sections["data"]
    execution = sections["execution"]
    resources = sections["runtime_resources"]
    sampler = sections["event_sampler"]
    stages = sections["stages"]
    ranking = sections["ranking"]
    gates = sections["development_gates"]
    terminal = sections["reserved_terminal"]
    assert isinstance(implementation, Mapping)
    assert isinstance(predecessor, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(ranking, Mapping)
    assert isinstance(gates, Mapping)
    assert isinstance(terminal, Mapping)
    if require_current:
        _validate_implementation_binding(implementation)
    if (
        predecessor.get("round") != 13
        or predecessor.get("design_sha256")
        != "57fcf6d940810d251917961d281f96e0c3b9ac88e3bde06faa8c59cdeebcb6f7"
        or predecessor.get("report_sha256")
        != "0f7bb314a74a849a2fae8510793c118378f157d69d1c84d2161190f0bc573e33"
    ):
        raise ValueError("head-coherence predecessor evidence is invalid")
    roles = data.get("roles")
    if (
        data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not False
        or not isinstance(roles, Mapping)
        or tuple(roles) != _ROLE_NAMES
    ):
        raise ValueError("head-coherence data contract is invalid")
    previous_end = None
    for role in _ROLE_NAMES:
        raw = roles[role]
        if not isinstance(raw, Mapping):
            raise ValueError(f"head-coherence {role} role is invalid")
        first = _parse_date(raw.get("start"), label=f"{role} start")
        last = _parse_date(raw.get("end"), label=f"{role} end")
        if first > last or (
            previous_end is not None and first != previous_end + timedelta(days=1)
        ):
            raise ValueError("head-coherence roles must be contiguous")
        previous_end = last
    first_date = _parse_date(data.get("start_date"), label="data start")
    last_date = _parse_date(data.get("end_date"), label="data end")
    if first_date != _parse_date(
        roles["train"]["start"], label="first role"
    ) or last_date != _parse_date(
        roles["development_evaluation"]["end"],
        label="last role",
    ):
        raise ValueError("head-coherence roles do not partition the data window")
    terminal_date = _parse_date(terminal.get("date"), label="reserved terminal")
    if (
        terminal_date != last_date + timedelta(days=1)
        or terminal.get("included_in_dataset") is not False
        or terminal.get("access_permitted") is not False
    ):
        raise ValueError("head-coherence reserved terminal contract is invalid")
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
        raise ValueError("head-coherence execution contract is invalid")
    if (
        resources.get("duckdb_memory_limit") != "4GB"
        or int(resources.get("warehouse_threads") or 0) != 8
        or resources.get("compute_backend") != "directml"
        or resources.get("spill_directory_policy") != "warehouse_adjacent"
    ):
        raise ValueError("head-coherence resource contract is invalid")
    if (
        sampler.get("method") != "daily_reset_causal_cusum"
        or float(sampler.get("volatility_multiplier") or 0.0) <= 0.0
        or float(sampler.get("minimum_threshold_bps") or 0.0) <= 0.0
        or sampler.get("minimum_activity_quota") is not None
    ):
        raise ValueError("head-coherence event sampler is invalid")
    parsed_specs = [GrossArchitectureSpec(**dict(value)) for value in candidates]
    if (
        len(parsed_specs) != 4
        or len({spec.candidate_id for spec in parsed_specs}) != len(parsed_specs)
        or any(spec.family != "tabular_mlp" for spec in parsed_specs)
    ):
        raise ValueError("head-coherence neural candidate contract is invalid")
    if set(score_methods) != _REQUIRED_SCORE_METHODS or any(
        value not in GROSS_ACTION_SCORE_METHODS for value in score_methods
    ):
        raise ValueError("head-coherence action score methods are invalid")
    stage_one = stages.get("stage_one")
    stage_two = stages.get("stage_two")
    if not isinstance(stage_one, Mapping) or not isinstance(stage_two, Mapping):
        raise ValueError("head-coherence stage budgets are invalid")
    if (
        int(stage_one.get("training_stride") or 0) < 2
        or int(stage_one.get("batch_size") or 0) < 1_024
        or int(stage_one.get("max_epochs") or 0) < 1
        or int(stage_one.get("keep_candidates") or 0) != 2
        or int(stage_two.get("training_stride") or 0) != 1
        or int(stage_two.get("batch_size") or 0) < int(stage_one.get("batch_size") or 0)
        or int(stage_two.get("max_epochs") or 0)
        <= int(stage_one.get("max_epochs") or 0)
    ):
        raise ValueError("head-coherence successive-halving contract is invalid")
    if (
        ranking.get("stage_one_role") != "calibration"
        or ranking.get("final_ranking_role") != "policy"
        or tuple(ranking.get("lexicographic_descending") or ())
        != (
            "top_500_mean_exact_after_cost_bps",
            "top_500_mean_signed_gross_bps",
            "top_500_exact_after_cost_positive_rate",
        )
        or tuple(ranking.get("diagnostic_top_rows") or ()) != (100, 500, 1_000)
        or ranking.get("development_evaluation_used_for_selection") is not False
    ):
        raise ValueError("head-coherence ranking contract is invalid")
    if set(gates) != _REQUIRED_GATES:
        raise ValueError("head-coherence development gates are incomplete")
    return payload, str(claimed)


def _top_row(
    metrics: Mapping[str, object], requested: int = 500
) -> Mapping[str, object]:
    rows = metrics.get("top_rows")
    if not isinstance(rows, Sequence):
        raise ValueError("head-coherence metrics have no top-row diagnostics")
    matched = [
        value
        for value in rows
        if isinstance(value, Mapping)
        and int(value.get("requested_rows") or 0) == requested
    ]
    if len(matched) != 1:
        raise ValueError(f"head-coherence top-{requested} evidence is ambiguous")
    return matched[0]


def _action_rank(metrics: Mapping[str, object] | None) -> tuple[float, float, float]:
    if metrics is None:
        return (-1.0e100, -1.0e100, -1.0e100)
    top = _top_row(metrics)
    return (
        float(top["mean_exact_after_cost_bps"]),
        float(top["mean_signed_gross_bps"]),
        float(top["exact_after_cost_positive_rate"]),
    )


def _action_diagnostics(
    dataset,
    target: np.ndarray,
    prediction,
    methods: Sequence[str],
    requested_top_rows: Sequence[int],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in methods:
        score = derive_gross_action_scores(prediction, method=method)
        try:
            metrics = evaluate_gross_action_scores(
                dataset,
                target,
                prediction,
                score,
                requested_top_rows=requested_top_rows,
            ).asdict()
            error = None
        except ValueError as exc:
            if "no eligible active rows" not in str(exc):
                raise
            metrics = None
            error = str(exc)
        output.append(
            {
                "score_method": method,
                "metrics": metrics,
                "evaluation_error": error,
            }
        )
    return output


def _action_gate_reasons(
    *,
    policy: Mapping[str, object] | None,
    development: Mapping[str, object] | None,
    development_forecast: Mapping[str, object],
    gates: Mapping[str, object],
) -> list[str]:
    if policy is None or development is None:
        return ["action_score_has_no_eligible_active_rows"]
    reasons: list[str] = []
    if float(development_forecast["direction_auc"]) <= float(
        gates["minimum_development_direction_auc"]
    ):
        reasons.append("development_direction_auc_gate_failed")
    if float(development_forecast["spearman_information_coefficient"]) <= float(
        gates["minimum_development_spearman_ic"]
    ):
        reasons.append("development_spearman_ic_gate_failed")
    if bool(gates["require_development_mae_better_than_zero"]) and float(
        development_forecast["mean_absolute_error_bps"]
    ) >= float(development_forecast["zero_baseline_mae_bps"]):
        reasons.append("development_mae_not_better_than_zero")
    if int(policy["active_rows"]) < int(gates["minimum_policy_active_rows"]):
        reasons.append("policy_active_rows_gate_failed")
    if int(development["active_rows"]) < int(gates["minimum_development_active_rows"]):
        reasons.append("development_active_rows_gate_failed")
    policy_top = _top_row(policy)
    development_top = _top_row(development)
    comparisons = (
        (
            "policy_top_500_signed_gross_gate_failed",
            policy_top["mean_signed_gross_bps"],
            gates["minimum_policy_top_500_signed_gross_bps"],
        ),
        (
            "development_top_500_signed_gross_gate_failed",
            development_top["mean_signed_gross_bps"],
            gates["minimum_development_top_500_signed_gross_bps"],
        ),
        (
            "policy_top_500_exact_after_cost_gate_failed",
            policy_top["mean_exact_after_cost_bps"],
            gates["minimum_policy_top_500_exact_after_cost_bps"],
        ),
        (
            "development_top_500_exact_after_cost_gate_failed",
            development_top["mean_exact_after_cost_bps"],
            gates["minimum_development_top_500_exact_after_cost_bps"],
        ),
    )
    for reason, actual, minimum in comparisons:
        if float(actual) <= float(minimum):
            reasons.append(reason)
    return reasons


def _load_corpus(
    *,
    design: Mapping[str, object],
    warehouse_path: str | Path,
    cache_root: str | Path,
    memory_limit: str,
    threads: int,
    progress,
    feature_version: str = MICROSTRUCTURE_FEATURE_VERSION,
):
    data = design["data"]
    execution = design["execution"]
    sampler = design["event_sampler"]
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(sampler, Mapping)
    first = _parse_date(data["start_date"], label="data start")
    last = _parse_date(data["end_date"], label="data end")
    start_ms, end_ms = _utc_day_bounds(first, last)
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        progress("verify-source")
        source_evidence = dict(
            warehouse.require_causal_feature_bars(str(data["symbol"]))
        )
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=bool(
                data["full_history_inventory_required"]
            ),
        )
        source_evidence["corpus_certificate"] = certificate
        feature_source_contract = microstructure_feature_source_contract(
            feature_version
        )
        if feature_source_contract is not None:
            source_evidence["feature_source_contract"] = feature_source_contract
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
            "feature_version": feature_version,
        }
        cache_key = microstructure_dataset_cache_key(**cache_parameters)
        progress("cache-lookup")
        dataset = load_microstructure_dataset_cache(warehouse, **cache_parameters)
        cache_state = "hit"
        if dataset is None:
            cache_state = "build"
            progress("microstructure-dataset-build", feature_version=feature_version)
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
                feature_version=feature_version,
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
    target = gross_midpoint_log_returns_bps(dataset)
    event_mask = causal_cusum_event_mask(
        dataset,
        volatility_multiplier=float(sampler["volatility_multiplier"]),
        minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
    )
    roles, role_evidence = _role_indexes(dataset, data["roles"], event_mask)
    return {
        "dataset": dataset,
        "target": target,
        "roles": roles,
        "role_evidence": role_evidence,
        "event_rows": int(np.sum(event_mask)),
        "cache_key": cache_key,
        "cache_state": cache_state,
        "source_evidence": source_evidence,
        "certificate": certificate,
    }


def _final_action_results(
    *,
    dataset,
    target: np.ndarray,
    policy_prediction,
    development_prediction,
    development_forecast: Mapping[str, object],
    methods: Sequence[str],
    requested_top_rows: Sequence[int],
    gates: Mapping[str, object],
) -> list[dict[str, object]]:
    policy_rows = _action_diagnostics(
        dataset,
        target,
        policy_prediction,
        methods,
        requested_top_rows,
    )
    development_rows = _action_diagnostics(
        dataset,
        target,
        development_prediction,
        methods,
        requested_top_rows,
    )
    development_by_method = {
        str(value["score_method"]): value for value in development_rows
    }
    output: list[dict[str, object]] = []
    for policy_row in policy_rows:
        method = str(policy_row["score_method"])
        development_row = development_by_method[method]
        policy_metrics = policy_row["metrics"]
        development_metrics = development_row["metrics"]
        assert policy_metrics is None or isinstance(policy_metrics, Mapping)
        assert development_metrics is None or isinstance(
            development_metrics,
            Mapping,
        )
        reasons = _action_gate_reasons(
            policy=policy_metrics,
            development=development_metrics,
            development_forecast=development_forecast,
            gates=gates,
        )
        output.append(
            {
                "score_method": method,
                "policy_metrics": policy_metrics,
                "policy_evaluation_error": policy_row["evaluation_error"],
                "development_metrics": development_metrics,
                "development_evaluation_error": development_row["evaluation_error"],
                "selection_rank": list(_action_rank(policy_metrics)),
                "rejection_reasons": reasons,
                "status": "research_candidate" if not reasons else "rejected",
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
            }
        )
    output.sort(key=lambda value: tuple(value["selection_rank"]), reverse=True)
    return output


def run_head_coherence_screen(
    *,
    design_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_head_coherence_design(design_path)
    resources = design["runtime_resources"]
    stages = design["stages"]
    ranking = design["ranking"]
    gates = design["development_gates"]
    assert isinstance(resources, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(ranking, Mapping)
    assert isinstance(gates, Mapping)
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
            "head-coherence-screen "
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
    roles = corpus["roles"]
    assert isinstance(target, np.ndarray)
    assert isinstance(roles, Mapping)
    progress(
        "dataset-ready",
        dataset_rows=dataset.rows,
        event_rows=corpus["event_rows"],
        cache_state=corpus["cache_state"],
    )
    methods = tuple(str(value) for value in design["action_score_methods"])
    requested_top_rows = tuple(int(value) for value in ranking["diagnostic_top_rows"])
    stage_one = stages["stage_one"]
    stage_two = stages["stage_two"]
    assert isinstance(stage_one, Mapping)
    assert isinstance(stage_two, Mapping)
    specs = [
        GrossArchitectureSpec(**dict(value)) for value in design["neural_candidates"]
    ]
    stage_one_results: list[dict[str, object]] = []
    for spec in specs:
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
        progress("stage-one-train", candidate=spec.candidate_id)
        model = train_torch_gross_model(
            dataset,
            target,
            train_endpoints=train,
            tuning_endpoints=tuning,
            spec=spec,
            compute_backend=effective_backend,
            seed=int(design["seed"]),
            batch_size=int(stage_one["batch_size"]),
            max_epochs=int(stage_one["max_epochs"]),
            patience=int(stage_one["patience"]),
            train_sample_weights=average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                train,
            ),
            tuning_sample_weights=average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                tuning,
            ),
            progress=lambda epoch, total, training_loss, tuning_loss, candidate=spec.candidate_id: (
                progress(
                    "stage-one-epoch",
                    candidate=candidate,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                )
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
        forecast = evaluate_gross_forecast(
            dataset,
            target,
            prediction,
            requested_top_rows=requested_top_rows,
        ).asdict()
        actions = _action_diagnostics(
            dataset,
            target,
            prediction,
            methods,
            requested_top_rows,
        )
        best_action = max(
            actions,
            key=lambda value: _action_rank(value["metrics"]),
        )
        stage_one_results.append(
            {
                "candidate_id": spec.candidate_id,
                "artifact": _artifact_summary(model),
                "calibration_forecast_metrics": forecast,
                "calibration_action_results": actions,
                "best_calibration_score_method": best_action["score_method"],
                "selection_rank": list(_action_rank(best_action["metrics"])),
            }
        )
        del model, prediction
    stage_one_results.sort(
        key=lambda value: tuple(value["selection_rank"]),
        reverse=True,
    )
    selected_ids = [
        str(value["candidate_id"])
        for value in stage_one_results[: int(stage_one["keep_candidates"])]
    ]
    progress("stage-one-complete", selected=",".join(selected_ids))

    train_full = roles["train"]
    tuning_full = roles["early_stop"]
    progress("stage-two-lightgbm")
    baseline = train_lightgbm_gross_baseline(
        dataset,
        target,
        train_endpoints=train_full,
        tuning_endpoints=tuning_full,
        train_uniqueness=average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            train_full,
        ),
        tuning_uniqueness=average_label_uniqueness(
            dataset.decision_time_ms,
            dataset.long_exit_time_ms,
            tuning_full,
        ),
        compute_backend=effective_backend,
        seed=int(design["seed"]),
    )
    baseline_policy_prediction = predict_lightgbm_gross_model(
        baseline,
        dataset,
        roles["policy"],
    )
    baseline_development_prediction = predict_lightgbm_gross_model(
        baseline,
        dataset,
        roles["development_evaluation"],
    )
    baseline_policy_forecast = evaluate_gross_forecast(
        dataset,
        target,
        baseline_policy_prediction,
        requested_top_rows=requested_top_rows,
    ).asdict()
    baseline_development_forecast = evaluate_gross_forecast(
        dataset,
        target,
        baseline_development_prediction,
        requested_top_rows=requested_top_rows,
    ).asdict()
    baseline_path = destination / "lightgbm-baseline.json"
    write_json_atomic(
        baseline_path,
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
                "path": baseline_path.name,
                "sha256": _sha256_file(baseline_path),
                "bytes": baseline_path.stat().st_size,
            },
            "policy_forecast_metrics": baseline_policy_forecast,
            "development_forecast_metrics": baseline_development_forecast,
            "action_results": _final_action_results(
                dataset=dataset,
                target=target,
                policy_prediction=baseline_policy_prediction,
                development_prediction=baseline_development_prediction,
                development_forecast=baseline_development_forecast,
                methods=methods,
                requested_top_rows=requested_top_rows,
                gates=gates,
            ),
        }
    ]
    del baseline_policy_prediction, baseline_development_prediction

    for spec in specs:
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
        progress("stage-two-train", candidate=spec.candidate_id)
        model = train_torch_gross_model(
            dataset,
            target,
            train_endpoints=train,
            tuning_endpoints=tuning,
            spec=spec,
            compute_backend=effective_backend,
            seed=int(design["seed"]),
            batch_size=int(stage_two["batch_size"]),
            max_epochs=int(stage_two["max_epochs"]),
            patience=int(stage_two["patience"]),
            train_sample_weights=average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                train,
            ),
            tuning_sample_weights=average_label_uniqueness(
                dataset.decision_time_ms,
                dataset.long_exit_time_ms,
                tuning,
            ),
            progress=lambda epoch, total, training_loss, tuning_loss, candidate=spec.candidate_id: (
                progress(
                    "stage-two-epoch",
                    candidate=candidate,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                )
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
        policy_prediction = predict_torch_gross_model(
            model,
            dataset,
            policy_endpoints,
            compute_backend=effective_backend,
            batch_size=int(stage_two["batch_size"]),
        )
        development_prediction = predict_torch_gross_model(
            model,
            dataset,
            development_endpoints,
            compute_backend=effective_backend,
            batch_size=int(stage_two["batch_size"]),
        )
        policy_forecast = evaluate_gross_forecast(
            dataset,
            target,
            policy_prediction,
            requested_top_rows=requested_top_rows,
        ).asdict()
        development_forecast = evaluate_gross_forecast(
            dataset,
            target,
            development_prediction,
            requested_top_rows=requested_top_rows,
        ).asdict()
        artifact_file = _save_neural_artifact(
            destination / f"{spec.candidate_id}.safetensors",
            model,
        )
        final_results.append(
            {
                "candidate_id": spec.candidate_id,
                "artifact": _artifact_summary(model),
                "artifact_file": artifact_file,
                "policy_forecast_metrics": policy_forecast,
                "development_forecast_metrics": development_forecast,
                "action_results": _final_action_results(
                    dataset=dataset,
                    target=target,
                    policy_prediction=policy_prediction,
                    development_prediction=development_prediction,
                    development_forecast=development_forecast,
                    methods=methods,
                    requested_top_rows=requested_top_rows,
                    gates=gates,
                ),
            }
        )
        del model, policy_prediction, development_prediction

    for result in final_results:
        actions = result["action_results"]
        assert isinstance(actions, list)
        best_action = actions[0]
        result["best_policy_score_method"] = best_action["score_method"]
        result["selection_rank"] = best_action["selection_rank"]
        result["status"] = (
            "research_candidate"
            if any(value["status"] == "research_candidate" for value in actions)
            else "rejected"
        )
        result["trading_authority"] = False
        result["execution_claim"] = False
        result["profitability_claim"] = False
        result["portfolio_claim"] = False
    final_results.sort(
        key=lambda value: tuple(value["selection_rank"]),
        reverse=True,
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "consumed_data_head_coherence_development_evidence",
        "status": (
            "research_candidate"
            if any(value["status"] == "research_candidate" for value in final_results)
            else "rejected"
        ),
        "round": 14,
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
            "event_rows": corpus["event_rows"],
            "cache_key": corpus["cache_key"],
            "cache_state": corpus["cache_state"],
            "source_manifest_fingerprint": corpus["source_evidence"][
                "manifest_fingerprint"
            ],
            "gross_target_mean_bps": float(np.mean(target)),
            "gross_target_std_bps": float(np.std(target)),
            "roles": corpus["role_evidence"],
        },
        "successive_halving": {
            "stage_one_results": stage_one_results,
            "selected_candidate_ids": selected_ids,
        },
        "action_score_methods": list(methods),
        "final_results": final_results,
        "limitations": [
            "top-row diagnostics contain overlapping forecasts and are not trades or portfolio returns",
            "all evaluation dates in this screen were already consumed before Round 14",
            "candidate comparisons are multiple research hypotheses and require untouched confirmation",
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
        description="Run the precommitted gross-head coherence screen",
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
    report = run_head_coherence_screen(
        design_path=args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(
        "head-coherence-screen: "
        f"status={report['status']} sha256={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
