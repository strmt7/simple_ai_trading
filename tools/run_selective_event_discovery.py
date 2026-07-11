from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from simple_ai_trading.microstructure_features import (
    apply_path_aware_lifecycle_targets,
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_model import (
    _baseline_metrics,
    _minimum_evaluation_trades,
    _performance_confidence,
    _risk_utility,
    _SimulationTrace,
    _trading_metrics,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.storage import write_json_atomic

try:
    from tools.ablate_action_value_scores import (
        _causal_cusum_events,
        _event_model_scores,
        _select_score_threshold,
        _simulate_score_threshold,
        _top_score_diagnostic,
        _train_event_models,
    )
    from tools.run_action_value_discovery import (
        _canonical_sha256,
        _load_consumed_registry,
    )
except ModuleNotFoundError:
    from ablate_action_value_scores import (
        _causal_cusum_events,
        _event_model_scores,
        _select_score_threshold,
        _simulate_score_threshold,
        _top_score_diagnostic,
        _train_event_models,
    )
    from run_action_value_discovery import (
        _canonical_sha256,
        _load_consumed_registry,
    )


_ROLE_NAMES = ("train", "early_stop", "calibration", "policy", "selection")
_SCORE_METHODS = (
    "event_direct_mean",
    "event_upper_quantile",
    "event_distributional_value",
)
_BOUNDED_VIABILITY_PURPOSE = "bounded_exact_bbo_model_viability_screen"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date(value: object, *, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selective event {label} must be YYYY-MM-DD") from exc


def _date_strings(first: date, last: date) -> tuple[str, ...]:
    output: list[str] = []
    current = first
    while current <= last:
        output.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(output)


def _implementation_is_current(change_control: Mapping[str, object]) -> None:
    files = change_control.get("implementation_files_sha256")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("selective event implementation file binding is missing")
    for relative, expected in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("selective event implementation path is not repository-relative")
        path = (_PROJECT_ROOT / relative_path).resolve()
        if not path.is_relative_to(_PROJECT_ROOT):
            raise ValueError("selective event implementation path escapes the repository")
        if not path.is_file() or _file_sha256(path) != str(expected):
            raise ValueError(f"selective event implementation changed: {relative_path}")


def _ensure_causal_feature_bars(
    warehouse: MicrostructureWarehouse,
    symbol: str,
    *,
    progress: Callable[[str, int, int | None], None],
) -> dict[str, object]:
    try:
        evidence = warehouse.require_causal_feature_bars(symbol)
    except ValueError:
        return warehouse.rebuild_causal_feature_bars(symbol, progress=progress)
    rows = int(evidence.get("feature_rows") or 0)
    progress("causal-feature-reuse", rows, rows)
    return evidence


def load_selective_event_design(
    path: str | Path,
    *,
    require_current: bool = False,
) -> dict[str, object]:
    design_path = Path(path)
    try:
        payload = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selective event design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("selective event design must be an object")
    claimed = str(payload.get("design_sha256") or "")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if (
        payload.get("schema_version") != "selective-event-discovery-design-v1"
        or claimed != _canonical_sha256(canonical)
        or int(payload.get("round") or 0) < 12
        or payload.get("status") != "precommitted"
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("selective event design contract is invalid")
    change_control = payload.get("change_control")
    data = payload.get("data")
    execution = payload.get("execution")
    training = payload.get("training")
    risk_profiles = payload.get("risk_profiles")
    selection = payload.get("selection")
    reserved_terminal = payload.get("reserved_terminal")
    if not all(
        isinstance(value, Mapping)
        for value in (
            change_control,
            data,
            execution,
            training,
            risk_profiles,
            selection,
            reserved_terminal,
        )
    ):
        raise ValueError("selective event design sections are incomplete")
    assert isinstance(change_control, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(risk_profiles, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(reserved_terminal, Mapping)
    bounded_viability = payload.get("purpose") == _BOUNDED_VIABILITY_PURPOSE
    runtime_resources = payload.get("runtime_resources")
    implementation_commit = str(change_control.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("selective event implementation commit is invalid")
    if require_current:
        _implementation_is_current(change_control)
    if (
        data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or data.get("symbol") != "BTCUSDT"
        or data.get("required_data_types") != ["bookTicker", "trades"]
        or data.get("checksum_verified_partitions_required") is not True
        or data.get("selection_dates_previously_untouched") is not True
    ):
        raise ValueError("selective event data contract is invalid")
    if bounded_viability:
        if (
            data.get("full_history_inventory_required") is not False
            or data.get("inventory_scope") != "bounded_verified"
        ):
            raise ValueError("bounded viability inventory contract is invalid")
        resource_contract_required = int(payload.get("design_revision") or 6) >= 6
        if resource_contract_required and (
            not isinstance(runtime_resources, Mapping)
            or runtime_resources.get("duckdb_memory_limit") != "4GB"
            or runtime_resources.get("warehouse_threads") != 8
            or runtime_resources.get("spill_directory_policy")
            != "warehouse_adjacent"
            or runtime_resources.get("feature_build_chunk_clock") != "utc_event_day"
        ):
            raise ValueError("bounded viability runtime resource contract is invalid")
    elif data.get("full_history_inventory_required") is not True:
        raise ValueError("selective event full-history inventory is required")
    first = _parse_date(data.get("start_date"), label="start_date")
    last = _parse_date(data.get("end_date"), label="end_date")
    roles = data.get("roles")
    if first > last or not isinstance(roles, Mapping) or tuple(roles) != _ROLE_NAMES:
        raise ValueError("selective event role calendar is invalid")
    expected_start = first
    role_dates: dict[str, set[str]] = {}
    for role in _ROLE_NAMES:
        value = roles[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"selective event {role} role is invalid")
        role_start = _parse_date(value.get("start_date"), label=f"{role} start")
        role_end = _parse_date(value.get("end_date"), label=f"{role} end")
        dates = _date_strings(role_start, role_end)
        if (
            role_start != expected_start
            or int(value.get("day_count") or 0) != len(dates)
        ):
            raise ValueError("selective event role calendar is not contiguous")
        role_dates[role] = set(dates)
        expected_start = role_end + timedelta(days=1)
    if expected_start - timedelta(days=1) != last:
        raise ValueError("selective event roles do not partition the data window")
    registry_name = str(data.get("consumed_registry") or "")
    registry_hash = str(data.get("consumed_registry_sha256") or "")
    consumed = _load_consumed_registry(design_path.with_name(registry_name), registry_hash)
    if consumed & role_dates["selection"]:
        raise ValueError("selective event selection calendar was already consumed")
    terminal_start = _parse_date(
        reserved_terminal.get("start_date"),
        label="reserved terminal start",
    )
    terminal_end = _parse_date(
        reserved_terminal.get("end_date"),
        label="reserved terminal end",
    )
    terminal_dates = set(_date_strings(terminal_start, terminal_end))
    if (
        terminal_start != last + timedelta(days=1)
        or int(reserved_terminal.get("day_count") or 0) != len(terminal_dates)
        or reserved_terminal.get("included_in_dataset") is not False
        or reserved_terminal.get("labels_constructed") is not False
        or reserved_terminal.get("access_allowed_in_round_12") is not False
        or consumed & terminal_dates
    ):
        raise ValueError("selective event reserved terminal contract is invalid")
    if (
        int(execution.get("total_latency_ms") or 0) != 750
        or float(execution.get("taker_fee_bps_per_side") or 0.0) != 5.0
        or float(execution.get("additional_slippage_bps_per_side") or 0.0) != 1.0
        or int(execution.get("decision_cadence_seconds") or 0) != 5
        or execution.get("suppress_overlapping_positions") is not True
        or execution.get("maker_fill_claim") is not False
        or float(execution.get("leverage") or 0.0) != 1.0
    ):
        raise ValueError("selective event execution contract is invalid")
    if (
        training.get("model_family") != "causal_cusum_uniqueness_weighted_distributional_lgbm"
        or training.get("feature_version") != "l1-tape-causal-v7"
        or float(training.get("cusum_volatility_multiplier") or 0.0) != 1.0
        or float(training.get("cusum_minimum_threshold_bps") or 0.0) != 1.0
        or tuple(training.get("score_methods") or ()) != _SCORE_METHODS
        or training.get("compute_backend") != "directml"
        or training.get("evaluate_terminal") is not False
    ):
        raise ValueError("selective event training contract is invalid")
    parameter_profile = str(
        training.get("predictor_parameter_profile") or "risk_specific"
    )
    if bounded_viability:
        if parameter_profile != "shared_regularized":
            raise ValueError("bounded viability predictor profile is invalid")
    elif parameter_profile != "risk_specific":
        raise ValueError("selective event predictor profile is invalid")
    horizons = tuple(int(value) for value in payload.get("horizon_seconds") or ())
    expected_horizons = (300, 900) if bounded_viability else (300, 900, 1800)
    if horizons != expected_horizons:
        raise ValueError("selective event horizons are invalid")
    if tuple(risk_profiles) != ("conservative", "regular", "aggressive"):
        raise ValueError("selective event risk profiles are invalid")
    for name, profile in risk_profiles.items():
        if not isinstance(profile, Mapping) or any(
            not math.isfinite(float(profile.get(key) or 0.0))
            or float(profile.get(key) or 0.0) <= 0.0
            for key in (
                "stop_loss_bps",
                "take_profit_bps",
                "max_l1_participation",
                "max_selection_drawdown_bps",
            )
        ):
            raise ValueError(f"selective event {name} risk profile is invalid")
    expected_candidates = len(horizons) * len(risk_profiles) * len(_SCORE_METHODS)
    if (
        int(payload.get("model_fit_count") or 0)
        != len(horizons) * len(risk_profiles)
        or int(payload.get("candidate_count") or 0) != expected_candidates
    ):
        raise ValueError("selective event candidate count is invalid")
    selection_gate_is_valid = (
        selection.get("promotion_allowed") is False
        and selection.get("positive_daily_bootstrap_lower_bound_required") is True
    )
    if bounded_viability:
        selection_gate_is_valid = selection_gate_is_valid and (
            int(selection.get("minimum_policy_trades") or 0) >= 20
            and int(selection.get("minimum_selection_trades") or 0) >= 20
            and selection.get("activity_is_not_a_trade_quota") is True
        )
    else:
        selection_gate_is_valid = selection_gate_is_valid and (
            selection.get("minimum_trades_per_selection_day") == 5
        )
    if not selection_gate_is_valid:
        raise ValueError("selective event selection gates are invalid")
    return payload


def _resolved_runtime_settings(
    design: Mapping[str, object],
    *,
    memory_limit: str | None,
    threads: int | None,
    compute_backend: str | None,
) -> tuple[str, int, str]:
    training = design.get("training")
    resources = design.get("runtime_resources")
    if not isinstance(training, Mapping):
        raise ValueError("selective event training settings are missing")
    expected_backend = str(training.get("compute_backend") or "")
    expected_memory = (
        str(resources.get("duckdb_memory_limit"))
        if isinstance(resources, Mapping)
        else "8GB"
    )
    expected_threads = (
        int(resources.get("warehouse_threads") or 0)
        if isinstance(resources, Mapping)
        else 8
    )
    requested_memory = expected_memory if memory_limit is None else str(memory_limit).upper()
    requested_threads = expected_threads if threads is None else int(threads)
    requested_backend = (
        expected_backend if compute_backend is None else str(compute_backend).lower()
    )
    if requested_memory != expected_memory:
        raise ValueError("memory limit override violates the precommitted design")
    if requested_threads != expected_threads:
        raise ValueError("thread override violates the precommitted design")
    if requested_backend != expected_backend:
        raise ValueError("compute backend override violates the precommitted design")
    return expected_memory, expected_threads, expected_backend


def _role_calendar_split(
    dataset,
    roles: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    decision_days = dataset.decision_time_ms // 86_400_000
    available_days = set(int(value) for value in np.unique(decision_days))
    split: dict[str, np.ndarray] = {}
    evidence: dict[str, object] = {}
    prior_last = -1
    for index, role in enumerate(_ROLE_NAMES):
        value = roles[role]
        assert isinstance(value, Mapping)
        first = _parse_date(value["start_date"], label=f"{role} start")
        last = _parse_date(value["end_date"], label=f"{role} end")
        first_day = int(
            datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc).timestamp()
            // 86_400
        )
        last_day = int(
            datetime.combine(last, datetime.min.time(), tzinfo=timezone.utc).timestamp()
            // 86_400
        )
        expected_days = set(range(first_day, last_day + 1))
        if not expected_days <= available_days:
            raise ValueError(f"selective event {role} data days are incomplete")
        raw = np.flatnonzero(
            (decision_days >= first_day) & (decision_days <= last_day)
        ).astype(np.int64)
        kept = raw
        if index + 1 < len(_ROLE_NAMES):
            next_role = roles[_ROLE_NAMES[index + 1]]
            assert isinstance(next_role, Mapping)
            next_date = _parse_date(
                next_role["start_date"],
                label=f"{_ROLE_NAMES[index + 1]} start",
            )
            next_start_ms = int(
                datetime.combine(
                    next_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ).timestamp()
                * 1_000
            )
            kept = raw[
                (dataset.long_exit_time_ms[raw] < next_start_ms)
                & (dataset.short_exit_time_ms[raw] < next_start_ms)
            ]
        if len(kept) < 256 or int(kept[0]) <= prior_last:
            raise ValueError(f"selective event {role} split is invalid")
        prior_last = int(kept[-1])
        split[role] = kept
        evidence[role] = {
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "day_count": len(expected_days),
            "raw_rows": int(len(raw)),
            "rows": int(len(kept)),
            "purged_rows": int(len(raw) - len(kept)),
            "first_decision_time_ms": int(dataset.decision_time_ms[kept[0]]),
            "last_decision_time_ms": int(dataset.decision_time_ms[kept[-1]]),
        }
    return split, evidence


def _model_fit_id(risk_level: str, horizon_seconds: int) -> str:
    return f"{risk_level}-h{horizon_seconds}"


def _empty_trace() -> _SimulationTrace:
    return _SimulationTrace(
        metrics=_trading_metrics([], [], []),
        pnls=(),
        sides=(),
        timestamps=(),
    )


def _artifact_is_reusable(
    path: Path,
    *,
    design_sha256: str,
    corpus_sha256: str,
    model_fit_id: str,
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    claimed = str(payload.get("artifact_sha256") or "")
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    if (
        claimed != _canonical_sha256(canonical)
        or payload.get("design_sha256") != design_sha256
        or payload.get("corpus_certificate_sha256") != corpus_sha256
        or payload.get("model_fit_id") != model_fit_id
        or payload.get("terminal_holdout_accessed") is not False
        or not isinstance(payload.get("outcomes"), list)
    ):
        return None
    return payload


def run_selective_event_discovery(
    design_path: str | Path,
    *,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    design = load_selective_event_design(design_path, require_current=True)
    bounded_viability = design.get("purpose") == _BOUNDED_VIABILITY_PURPOSE
    design_sha256 = str(design["design_sha256"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    report_path = output / "report.json"
    data = design["data"]
    execution = design["execution"]
    training = design["training"]
    risk_profiles = design["risk_profiles"]
    selection_contract = design["selection"]
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(risk_profiles, Mapping)
    assert isinstance(selection_contract, Mapping)
    effective_memory_limit, effective_threads, backend = _resolved_runtime_settings(
        design,
        memory_limit=memory_limit,
        threads=threads,
        compute_backend=compute_backend,
    )
    runtime_resources = {
        "duckdb_memory_limit": effective_memory_limit,
        "warehouse_threads": effective_threads,
        "spill_directory_policy": "warehouse_adjacent",
        "compute_backend_requested": backend,
    }
    start = _parse_date(data["start_date"], label="start_date")
    end = _parse_date(data["end_date"], label="end_date")
    start_ms = int(
        datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1_000
    )
    end_ms = int(
        datetime.combine(
            end + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
        * 1_000
    ) - 1
    full_history_inventory_required = bool(
        data["full_history_inventory_required"]
    )
    completed: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    total_fits = len(design["horizon_seconds"]) * len(risk_profiles)

    def progress(phase: str, current: str | None = None) -> None:
        print(
            f"selective-event-discovery {phase} "
            f"fit={current or '-'} complete={len(completed)} errors={len(errors)}",
            flush=True,
        )
        write_json_atomic(
            status_path,
            {
                "schema_version": design["schema_version"],
                "design_sha256": design_sha256,
                "phase": phase,
                "current_model_fit": current,
                "completed_outcomes": len(completed),
                "failed_model_fits": len(errors),
                "total_model_fits": total_fits,
                "total_outcomes": int(design["candidate_count"]),
                "runtime_resources": runtime_resources,
            },
            indent=2,
            sort_keys=True,
        )

    progress("initialize")
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory_limit,
        threads=effective_threads,
    ) as warehouse:
        warehouse.backfill_book_ticker_paths(
            progress=lambda phase, done, total: progress(
                f"{phase}:{done}/{total if total is not None else '?'}"
            )
        )
        _ensure_causal_feature_bars(
            warehouse,
            str(data["symbol"]),
            progress=lambda phase, done, total: progress(
                f"{phase}:{done}/{total if total is not None else '?'}"
            ),
        )
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=full_history_inventory_required,
        )
        corpus_sha256 = str(certificate["certificate_sha256"])
        for horizon in design["horizon_seconds"]:
            horizon_value = int(horizon)
            progress(f"build-h{horizon_value}")
            base = build_executable_microstructure_dataset(
                warehouse,
                symbol=str(data["symbol"]),
                horizon_seconds=horizon_value,
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
                require_full_history_inventory=full_history_inventory_required,
            )
            event_mask = _causal_cusum_events(
                base,
                volatility_multiplier=float(training["cusum_volatility_multiplier"]),
                minimum_threshold_bps=float(training["cusum_minimum_threshold_bps"]),
            )
            for risk_level, raw_profile in risk_profiles.items():
                assert isinstance(raw_profile, Mapping)
                fit_id = _model_fit_id(str(risk_level), horizon_value)
                artifact_path = output / f"{fit_id}.json"
                if resume and artifact_path.exists():
                    prior = _artifact_is_reusable(
                        artifact_path,
                        design_sha256=design_sha256,
                        corpus_sha256=corpus_sha256,
                        model_fit_id=fit_id,
                    )
                    if prior is not None:
                        completed.extend(dict(value) for value in prior["outcomes"])
                        progress("resumed", fit_id)
                        continue
                progress("lifecycle", fit_id)
                limit = float(raw_profile["max_l1_participation"])
                dataset = replace(
                    base,
                    max_l1_participation=limit,
                    long_liquidity_eligible=(
                        np.asarray(base.long_l1_participation) <= limit
                    ),
                    short_liquidity_eligible=(
                        np.asarray(base.short_l1_participation) <= limit
                    ),
                )
                try:
                    dataset, path_evidence = apply_path_aware_lifecycle_targets(
                        warehouse,
                        dataset,
                        stop_loss_bps=float(raw_profile["stop_loss_bps"]),
                        take_profit_bps=float(raw_profile["take_profit_bps"]),
                        trigger_execution_slippage_bps=float(
                            execution["trigger_execution_slippage_bps"]
                        ),
                    )
                    roles, role_evidence = _role_calendar_split(
                        dataset,
                        data["roles"],
                    )
                    event_dataset = replace(
                        dataset,
                        long_liquidity_eligible=(
                            np.asarray(dataset.long_liquidity_eligible, dtype=bool)
                            & event_mask
                        ),
                        short_liquidity_eligible=(
                            np.asarray(dataset.short_liquidity_eligible, dtype=bool)
                            & event_mask
                        ),
                    )
                    progress("training", fit_id)
                    models, iterations, calibrations, model_evidence = _train_event_models(
                        dataset,
                        roles,
                        event_mask=event_mask,
                        risk_level=str(risk_level),
                        compute_backend=backend,
                        seed=int(training["seed"]),
                        explicit_role_indexes={
                            name: roles[name]
                            for name in ("train", "early_stop", "calibration")
                        },
                        parameter_profile=str(
                            training.get("predictor_parameter_profile")
                            or "risk_specific"
                        ),
                    )
                    segment_scores = {
                        role: _event_model_scores(
                            features=np.asarray(
                                dataset.features[roles[role]],
                                dtype=np.float32,
                            ),
                            models=models,
                            iterations=iterations,
                            calibrations=calibrations,
                            risk_level=str(risk_level),
                        )
                        for role in ("policy", "selection")
                    }
                    fit_outcomes: list[dict[str, object]] = []
                    for method in training["score_methods"]:
                        policy_scores = segment_scores["policy"][method]
                        policy = _select_score_threshold(
                            dataset=event_dataset,
                            indexes=roles["policy"],
                            long_scores=policy_scores["long"],
                            short_scores=policy_scores["short"],
                            risk_level=str(risk_level),
                            minimum_trades=(
                                int(selection_contract["minimum_policy_trades"])
                                if bounded_viability
                                else None
                            ),
                        )
                        selection_scores = segment_scores["selection"][method]
                        trace = (
                            _simulate_score_threshold(
                                dataset=event_dataset,
                                indexes=roles["selection"],
                                long_scores=selection_scores["long"],
                                short_scores=selection_scores["short"],
                                threshold=float(policy["threshold"]),
                            )
                            if policy["accepted"]
                            else _empty_trace()
                        )
                        confidence = _performance_confidence(
                            trace,
                            dataset.decision_time_ms[roles["selection"]],
                        )
                        baselines = _baseline_metrics(event_dataset, roles["selection"])
                        minimum_trades = (
                            int(selection_contract["minimum_selection_trades"])
                            if bounded_viability
                            else max(
                                _minimum_evaluation_trades(
                                    dataset.decision_time_ms[roles["selection"]]
                                ),
                                int(role_evidence["selection"]["day_count"])
                                * int(
                                    selection_contract[
                                        "minimum_trades_per_selection_day"
                                    ]
                                ),
                            )
                        )
                        reasons: list[str] = []
                        if not policy["accepted"]:
                            reasons.append("policy_has_no_positive_risk_utility")
                        if trace.metrics.trades < minimum_trades:
                            reasons.append("selection_trade_count_below_minimum")
                        if (
                            trace.metrics.total_net_bps <= 0.0
                            or _risk_utility(trace.metrics, str(risk_level)) <= 0.0
                        ):
                            reasons.append("selection_risk_utility_not_positive")
                        if (
                            trace.metrics.profit_factor is None
                            or trace.metrics.profit_factor <= 1.0
                        ):
                            reasons.append("selection_profit_factor_not_above_one")
                        if confidence.mean_daily_net_bps_ci_lower <= 0.0:
                            reasons.append("selection_daily_lower_bound_not_positive")
                        if trace.metrics.max_drawdown_bps > float(
                            raw_profile["max_selection_drawdown_bps"]
                        ):
                            reasons.append("selection_drawdown_limit_exceeded")
                        strongest_baseline = max(
                            value.total_net_bps for value in baselines.values()
                        )
                        if trace.metrics.total_net_bps <= strongest_baseline:
                            reasons.append("selection_not_above_directional_baseline")
                        outcome = {
                            "candidate_id": f"{fit_id}-{method}",
                            "model_fit_id": fit_id,
                            "risk_level": str(risk_level),
                            "horizon_seconds": horizon_value,
                            "score_method": str(method),
                            "status": "candidate" if not reasons else "rejected",
                            "rejection_reasons": reasons,
                            "policy": {
                                key: value
                                for key, value in policy.items()
                                if key != "evaluations"
                            },
                            "selection_metrics": asdict(trace.metrics),
                            "selection_confidence": asdict(confidence),
                            "selection_baselines": {
                                name: asdict(value) for name, value in baselines.items()
                            },
                            "selection_top_score_rows": _top_score_diagnostic(
                                dataset=event_dataset,
                                indexes=roles["selection"],
                                long_scores=selection_scores["long"],
                                short_scores=selection_scores["short"],
                            ),
                            "minimum_selection_trades": minimum_trades,
                            "research_score_bps": _risk_utility(
                                trace.metrics,
                                str(risk_level),
                            ),
                            "trading_authority": False,
                            "profitability_claim": False,
                        }
                        fit_outcomes.append(outcome)
                    artifact: dict[str, object] = {
                        "schema_version": "selective-event-model-fit-v1",
                        "design_sha256": design_sha256,
                        "corpus_certificate_sha256": corpus_sha256,
                        "model_fit_id": fit_id,
                        "feature_version": training["feature_version"],
                        "model_family": training["model_family"],
                        "runtime_resources": runtime_resources,
                        "source_evidence": dict(dataset.source_evidence or {}),
                        "path_target_evidence": asdict(path_evidence),
                        "role_evidence": role_evidence,
                        "event_rows": {
                            role: int(np.sum(event_mask[indexes]))
                            for role, indexes in roles.items()
                        },
                        "model_evidence": model_evidence,
                        "best_iterations": iterations,
                        "probability_calibration": {
                            side: list(values)
                            for side, values in calibrations.items()
                        },
                        "model_strings": {
                            name: model.model_to_string(num_iteration=iterations[name])
                            for name, model in models.items()
                        },
                        "outcomes": fit_outcomes,
                        "terminal_holdout_accessed": False,
                        "trading_authority": False,
                        "profitability_claim": False,
                    }
                    artifact["artifact_sha256"] = _canonical_sha256(artifact)
                    write_json_atomic(artifact_path, artifact, indent=2, sort_keys=True)
                    completed.extend(fit_outcomes)
                    progress("fit-complete", fit_id)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "model_fit_id": fit_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    progress("fit-failed", fit_id)
                finally:
                    gc.collect()
            del base, event_mask
            gc.collect()
    ranked = sorted(
        completed,
        key=lambda value: (
            float(value["research_score_bps"]),
            float(value["selection_metrics"]["total_net_bps"]),
        ),
        reverse=True,
    )
    report: dict[str, object] = {
        "schema_version": design["schema_version"],
        "design_sha256": design_sha256,
        "status": "completed" if len(completed) == int(design["candidate_count"]) and not errors else "failed",
        "artifact_class": "exchange_sourced_selective_event_discovery",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "terminal_holdout_accessed": False,
        "selection_window_is_consumed": True,
        "corpus_certificate_sha256": corpus_sha256,
        "runtime_resources": runtime_resources,
        "candidate_count": int(design["candidate_count"]),
        "completed_candidate_count": len(completed),
        "failed_model_fit_count": len(errors),
        "unrejected_candidate_count": sum(
            value["status"] == "candidate" for value in completed
        ),
        "best_discovery_candidate": ranked[0] if ranked else None,
        "ranked_results": ranked,
        "errors": errors,
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    progress("complete" if report["status"] == "completed" else "failed")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the precommitted long-span selective event discovery",
    )
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--compute-backend", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    report = run_selective_event_discovery(
        args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
        resume=args.resume,
    )
    print(
        "selective-event-discovery: "
        f"status={report['status']} "
        f"completed={report['completed_candidate_count']}/{report['candidate_count']} "
        f"unrejected={report['unrejected_candidate_count']}"
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
