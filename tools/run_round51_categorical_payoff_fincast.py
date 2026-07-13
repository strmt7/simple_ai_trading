"""Run the frozen Round 51 real-tick payoff-distribution screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, date, datetime, time as datetime_time
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.categorical_payoff_lightgbm import (  # noqa: E402
    CategoricalPayoffSpec,
    build_categorical_payoff_dataset,
    load_categorical_payoff_model,
    predict_categorical_payoff_model,
    save_categorical_payoff_model,
    train_categorical_payoff_model,
)
from simple_ai_trading.direct_payoff_lightgbm import (  # noqa: E402
    DirectPayoffSpec,
    load_direct_payoff_model,
    predict_direct_payoff_model,
    save_direct_payoff_model,
    train_direct_payoff_model,
)
from simple_ai_trading.fincast_runtime import (  # noqa: E402
    FINCAST_CHECKPOINT_SHA256,
    FINCAST_CONTEXT_SECONDS,
    FINCAST_PARAMETER_COUNT,
    FINCAST_SOURCE_COMMIT,
    FinCastRuntime,
    extract_fincast_feature_matrix,
    fincast_feature_names,
    fincast_runtime_contract_sha256,
)
from simple_ai_trading.lightgbm_backend import (  # noqa: E402
    lightgbm_backend_parameters,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    AdaptiveBarrierSpec,
    build_adaptive_barrier_targets,
)
from simple_ai_trading.microstructure_cache import (  # noqa: E402
    load_microstructure_dataset_cache,
    microstructure_dataset_fingerprint,
    save_microstructure_dataset_cache,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    AGGREGATE_DEPTH_FEATURE_VERSION,
    build_executable_microstructure_dataset,
    microstructure_feature_source_contract,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.payoff_distribution_analysis import (  # noqa: E402
    base_and_paired_stress_traces,
    categorical_forecast_metrics,
    direct_forecast_metrics,
    ensemble_action_score,
    pairwise_seed_spearman,
    portfolio_trace_metrics,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 51
DESIGN_SCHEMA = "categorical-payoff-fincast-screen-design-v1"
BINDING_SCHEMA = "round-051-categorical-payoff-fincast-execution-binding-v1"
REPORT_SCHEMA = "categorical-payoff-fincast-screen-report-v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (5101, 5102, 5103)
CANDIDATES = (
    "direct_mean_lightgbm",
    "categorical_payoff_lightgbm",
    "categorical_payoff_lightgbm_fincast",
)
SIDES = ("long", "short")
_DAY_MS = 86_400_000


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 51 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 51 design identity is invalid")
    data = design.get("data_contract")
    execution = design.get("execution_target")
    model = design.get("model_contract")
    ai = design.get("ai_contract")
    claims = design.get("claims")
    if not all(
        isinstance(value, Mapping) for value in (data, execution, model, ai, claims)
    ):
        raise ValueError("Round 51 design sections are incomplete")
    fincast = ai.get("fincast")
    lightgbm = model.get("lightgbm")
    categorical = model.get("categorical_target")
    if (
        tuple(data.get("symbols") or ()) != SYMBOLS
        or tuple(data.get("archive_products") or ())
        != ("bookTicker", "trades", "bookDepth")
        or data.get("decision_cadence_seconds") != 10
        or data.get("target_path_resolution_ms") != 100
        or data.get("synthetic_fabricated_or_interpolated_market_rows_permitted")
        is not False
        or not isinstance(fincast, Mapping)
        or fincast.get("source_commit") != FINCAST_SOURCE_COMMIT
        or fincast.get("checkpoint_sha256") != FINCAST_CHECKPOINT_SHA256
        or fincast.get("parameter_count") != FINCAST_PARAMETER_COUNT
        or fincast.get("context_seconds") != FINCAST_CONTEXT_SECONDS
        or not isinstance(lightgbm, Mapping)
        or lightgbm.get("opencl_fp64_accumulation_required") is not True
        or not isinstance(categorical, Mapping)
        or tuple(model.get("seeds") or ()) != SEEDS
        or claims.get("selection_contaminated") is not True
        or claims.get("profitability_claim_permitted") is not False
        or claims.get("testnet_or_live_authority_permitted") is not False
    ):
        raise ValueError("Round 51 frozen model or claims contract drifted")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 51 binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    commit = str(binding.get("implementation_commit") or "")
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
        or len(commit) != 40
        or _git("status", "--porcelain")
    ):
        raise ValueError("Round 51 binding identity or worktree state is invalid")
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 51 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 51 bound blob entry is invalid")
        relative = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{commit}:{relative}") != expected
            or _git("rev-parse", f"HEAD:{relative}") != expected
        ):
            raise ValueError(f"Round 51 bound blob drifted: {relative}")
    return binding, claimed, commit


def _utc_bounds(first: str, last: str) -> tuple[int, int]:
    start = datetime.combine(date.fromisoformat(first), datetime_time(), tzinfo=UTC)
    end = datetime.combine(date.fromisoformat(last), datetime_time(), tzinfo=UTC)
    return int(start.timestamp() * 1_000), int(end.timestamp() * 1_000) + _DAY_MS - 1


def _role_indexes(
    decision_time_ms: np.ndarray,
    roles: Mapping[str, object],
) -> dict[str, np.ndarray]:
    boundaries = {
        "train": ("2023-05-16", "2023-05-31"),
        "early_stop": ("2023-06-01", "2023-06-04"),
        "calibration": ("2023-06-05", "2023-06-08"),
        "evaluation": ("2023-06-09", "2023-06-14"),
    }
    expected = {
        "training": "2023-05-16 through 2023-05-31 UTC",
        "early_stop": "2023-06-01 through 2023-06-04 UTC",
        "calibration": "2023-06-05 through 2023-06-08 UTC",
        "evaluation": "2023-06-09 through 2023-06-14 UTC",
    }
    if any(roles.get(name) != value for name, value in expected.items()):
        raise ValueError("Round 51 chronological role text drifted")
    output: dict[str, np.ndarray] = {}
    for name, (first, last) in boundaries.items():
        lower, upper = _utc_bounds(first, last)
        selected = np.flatnonzero(
            (decision_time_ms >= lower) & (decision_time_ms <= upper)
        ).astype(np.int64)
        if len(selected) < 512:
            raise ValueError(f"Round 51 {name} role has insufficient rows")
        output[name] = selected
    if not (
        output["train"][-1]
        < output["early_stop"][0]
        < output["calibration"][0]
        < output["evaluation"][0]
    ):
        raise ValueError("Round 51 role indexes overlap")
    return output


def _progress(stage: str, **values: object) -> None:
    detail = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"round51 {stage}{(' ' + detail) if detail else ''}", flush=True)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _dataset_parameters(
    *,
    symbol: str,
    source_evidence: Mapping[str, object],
    start_ms: int,
    end_ms: int,
    execution: Mapping[str, object],
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "horizon_seconds": int(execution["horizon_seconds"]),
        "total_latency_ms": int(execution["total_entry_latency_ms"]),
        "taker_fee_bps": float(execution["taker_fee_bps_per_side"]),
        "additional_slippage_bps_per_side": float(
            execution["additional_slippage_bps_per_side"]
        ),
        "reference_order_notional_quote": float(
            execution["reference_order_notional_quote"]
        ),
        "max_l1_participation": float(execution["max_l1_participation"]),
        "max_quote_age_ms": int(execution["max_quote_age_ms"]),
        "decision_cadence_seconds": 10,
        "require_full_history_inventory": False,
        "source_evidence": source_evidence,
        "feature_version": AGGREGATE_DEPTH_FEATURE_VERSION,
    }


def _load_real_symbol_data(
    *,
    symbol: str,
    warehouse_path: Path,
    cache_root: Path,
    memory_limit: str,
    threads: int,
    data_contract: Mapping[str, object],
    execution: Mapping[str, object],
):
    start_ms, end_ms = _utc_bounds(
        str(data_contract["start_date"]), str(data_contract["end_date"])
    )
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        _progress("source-audit", symbol=symbol)
        source_evidence = dict(warehouse.require_causal_feature_bars(symbol))
        certificate = warehouse.require_corpus_certificate(
            symbol,
            required_data_types=("bookTicker", "trades", "bookDepth"),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=False,
        )
        source_evidence["corpus_certificate"] = certificate
        source_contract = microstructure_feature_source_contract(
            AGGREGATE_DEPTH_FEATURE_VERSION
        )
        if source_contract is None:
            raise ValueError("Round 51 aggregate-depth source contract is missing")
        source_evidence["feature_source_contract"] = source_contract
        parameters = _dataset_parameters(
            symbol=symbol,
            source_evidence=source_evidence,
            start_ms=start_ms,
            end_ms=end_ms,
            execution=execution,
        )
        dataset = load_microstructure_dataset_cache(warehouse, **parameters)
        cache_state = "hit"
        if dataset is None:
            cache_state = "build"
            _progress("dataset-build", symbol=symbol)
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=symbol,
                horizon_seconds=int(execution["horizon_seconds"]),
                total_latency_ms=int(execution["total_entry_latency_ms"]),
                taker_fee_bps=float(execution["taker_fee_bps_per_side"]),
                additional_slippage_bps_per_side=float(
                    execution["additional_slippage_bps_per_side"]
                ),
                max_quote_age_ms=int(execution["max_quote_age_ms"]),
                reference_order_notional_quote=float(
                    execution["reference_order_notional_quote"]
                ),
                max_l1_participation=float(execution["max_l1_participation"]),
                decision_cadence_seconds=10,
                start_ms=start_ms,
                end_ms=end_ms,
                require_full_history_inventory=False,
                feature_version=AGGREGATE_DEPTH_FEATURE_VERSION,
            )
            save_microstructure_dataset_cache(
                warehouse,
                dataset,
                requested_start_ms=start_ms,
                requested_end_ms=end_ms,
                require_full_history_inventory=False,
            )
            cache_state = "written"
        _progress("barrier-targets", symbol=symbol, rows=dataset.rows)
        barrier_spec = AdaptiveBarrierSpec(
            horizon_seconds=int(execution["horizon_seconds"]),
            volatility_feature_name=str(execution["volatility_feature"]),
            stop_volatility_multiple=float(execution["stop_volatility_multiple"]),
            take_volatility_multiple=float(execution["take_volatility_multiple"]),
            minimum_stop_bps=float(execution["minimum_stop_bps"]),
            maximum_stop_bps=float(execution["maximum_stop_bps"]),
            minimum_take_bps=float(execution["minimum_take_bps"]),
            maximum_take_bps=float(execution["maximum_take_bps"]),
            base_protection_delay_ms=int(execution["base_protection_delay_ms"]),
            stress_protection_delay_ms=int(execution["stress_protection_delay_ms"]),
            trigger_execution_slippage_bps=float(
                execution["trigger_execution_slippage_bps"]
            ),
            path_resolution_ms=100,
            same_utc_day_exit=True,
        )
        targets = build_adaptive_barrier_targets(
            warehouse,
            dataset,
            np.arange(dataset.rows, dtype=np.int64),
            barrier_spec,
            progress=lambda day, total, valid: _progress(
                "barrier-day",
                symbol=symbol,
                day=day,
                days=total,
                valid=valid,
            ),
        )
        query_start = (
            int(dataset.decision_time_ms[0]) - (FINCAST_CONTEXT_SECONDS + 1) * 1_000
        )
        source = (
            warehouse.connect()
            .execute(
                "SELECT second_ms, close_mid FROM current_book_ticker_1s "
                "WHERE symbol = ? AND second_ms BETWEEN ? AND ? ORDER BY second_ms",
                [symbol, query_start, int(dataset.decision_time_ms[-1]) - 1_000],
            )
            .fetchnumpy()
        )
        second_ms = np.asarray(source["second_ms"], dtype=np.int64)
        close_mid = np.asarray(source["close_mid"], dtype=np.float32)
    return {
        "dataset": dataset,
        "targets": targets,
        "second_ms": second_ms,
        "close_mid": close_mid,
        "cache_state": cache_state,
        "dataset_sha256": microstructure_dataset_fingerprint(dataset),
        "source_evidence": source_evidence,
    }


def _load_or_extract_fincast(
    *,
    symbol: str,
    dataset,
    second_ms: np.ndarray,
    close_mid: np.ndarray,
    cache_directory: Path,
    source: Path,
    checkpoint: Path,
    backend: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    source_sha = _array_sha256(second_ms, close_mid)
    decision_sha = _array_sha256(np.asarray(dataset.decision_time_ms, dtype=np.int64))
    identity = _canonical_sha256(
        {
            "symbol": symbol,
            "source_series_sha256": source_sha,
            "decision_times_sha256": decision_sha,
            "runtime_contract_sha256": fincast_runtime_contract_sha256(),
            "backend": backend,
        }
    )
    matrix_path = cache_directory / f"{symbol.lower()}-{identity}.npy"
    evidence_path = cache_directory / f"{symbol.lower()}-{identity}.json"
    if matrix_path.is_file() and evidence_path.is_file():
        evidence = _read_object(evidence_path, f"{symbol} FinCast cache evidence")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if (
            evidence.get("cache_identity") != identity
            or evidence.get("source_series_sha256") != source_sha
            or evidence.get("decision_times_sha256") != decision_sha
            or tuple(evidence.get("feature_names") or ()) != fincast_feature_names()
            or matrix.shape != (dataset.rows, len(fincast_feature_names()))
            or matrix.dtype != np.float32
            or _array_sha256(matrix) != evidence.get("features_sha256")
            or _file_sha256(matrix_path) != evidence.get("matrix_file_sha256")
        ):
            raise ValueError(f"{symbol} FinCast cache drifted")
        return matrix, evidence, {"state": "hit", **_artifact(matrix_path)}
    _progress("fincast-extract", symbol=symbol, rows=dataset.rows, backend=backend)
    with FinCastRuntime(
        source=source,
        checkpoint=checkpoint,
        backend=backend,
    ) as runtime:
        batch, extraction = extract_fincast_feature_matrix(
            runtime,
            second_ms=second_ms,
            close_mid=close_mid,
            decision_time_ms=dataset.decision_time_ms,
            batch_size=64,
            progress=lambda current, total: (
                _progress(
                    "fincast-batch",
                    symbol=symbol,
                    batch=current,
                    batches=total,
                )
                if current == 1 or current == total or current % 100 == 0
                else None
            ),
        )
        runtime_evidence = asdict(runtime.evidence)
    if (
        extraction.source_series_sha256 != source_sha
        or extraction.decision_times_sha256 != decision_sha
    ):
        raise ValueError(f"{symbol} FinCast extraction source identity drifted")
    temporary = matrix_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(
            stream, np.asarray(batch.features, dtype=np.float32), allow_pickle=False
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(matrix_path)
    evidence = {
        **asdict(extraction),
        "cache_identity": identity,
        "feature_names": list(batch.feature_names),
        "runtime_contract_sha256": fincast_runtime_contract_sha256(),
        "runtime": runtime_evidence,
        "matrix_file_sha256": _file_sha256(matrix_path),
        "matrix_bytes": matrix_path.stat().st_size,
    }
    write_json_atomic(evidence_path, evidence, indent=2, sort_keys=True)
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    return matrix, evidence, {"state": "written", **_artifact(matrix_path)}


def _specifications(model_contract: Mapping[str, object]):
    lightgbm = model_contract["lightgbm"]
    categorical = model_contract["categorical_target"]
    assert isinstance(lightgbm, Mapping)
    assert isinstance(categorical, Mapping)
    common = {
        "learning_rate": float(lightgbm["learning_rate"]),
        "num_leaves": int(lightgbm["num_leaves"]),
        "max_depth": int(lightgbm["max_depth"]),
        "min_data_in_leaf": int(lightgbm["min_data_in_leaf"]),
        "feature_fraction": float(lightgbm["feature_fraction"]),
        "bagging_fraction": float(lightgbm["bagging_fraction"]),
        "bagging_freq": int(lightgbm["bagging_freq"]),
        "lambda_l1": float(lightgbm["lambda_l1"]),
        "lambda_l2": float(lightgbm["lambda_l2"]),
        "max_bin": int(lightgbm["max_bin"]),
        "num_boost_round": int(lightgbm["num_boost_round"]),
        "early_stopping_rounds": int(lightgbm["early_stopping_rounds"]),
        "gpu_use_dp_required": True,
    }
    direct = DirectPayoffSpec(
        candidate_id="direct_mean_lightgbm",
        family="side_specific_direct_exact_payoff_mean",
        **common,
    )
    base = CategoricalPayoffSpec(
        candidate_id="categorical_payoff_lightgbm",
        family="side_specific_categorical_exact_payoff",
        bin_edge_quantiles=tuple(
            float(value) for value in categorical["bin_edge_training_quantiles"]
        ),
        minimum_unique_bins=int(categorical["minimum_unique_bins"]),
        **common,
    )
    ai = CategoricalPayoffSpec(
        **{**base.__dict__, "candidate_id": "categorical_payoff_lightgbm_fincast"}
    )
    return direct, base, ai


def _save_prediction(path: Path, **arrays: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return _artifact(path)


def _train_symbol_models(
    *,
    symbol: str,
    deterministic_dataset,
    ai_dataset,
    roles: Mapping[str, np.ndarray],
    model_contract: Mapping[str, object],
    compute_backend: str,
    artifact_root: Path,
) -> dict[str, list[dict[str, object]]]:
    direct_spec, categorical_spec, ai_spec = _specifications(model_contract)
    output: dict[str, list[dict[str, object]]] = {
        candidate: [] for candidate in CANDIDATES
    }
    for seed in SEEDS:
        _progress("train", symbol=symbol, candidate=direct_spec.candidate_id, seed=seed)
        started = time.perf_counter()
        model = train_direct_payoff_model(
            deterministic_dataset,
            train_indexes=roles["train"],
            early_stop_indexes=roles["early_stop"],
            spec=direct_spec,
            target_scenario="base",
            compute_backend=compute_backend,
            seed=seed,
        )
        model_path = (
            artifact_root / symbol / direct_spec.candidate_id / f"seed-{seed}.json"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_direct_payoff_model(model_path, model)
        output[direct_spec.candidate_id].append(
            {
                "seed": seed,
                "model_sha256": model.model_sha256,
                "backend_kind": model.backend_kind,
                "backend_device": model.backend_device,
                "best_iterations": dict(model.best_iterations),
                "training_seconds": time.perf_counter() - started,
                "model": _artifact(model_path),
            }
        )
        del model
        for specification, dataset in (
            (categorical_spec, deterministic_dataset),
            (ai_spec, ai_dataset),
        ):
            _progress(
                "train",
                symbol=symbol,
                candidate=specification.candidate_id,
                seed=seed,
            )
            started = time.perf_counter()
            model = train_categorical_payoff_model(
                dataset,
                train_indexes=roles["train"],
                early_stop_indexes=roles["early_stop"],
                calibration_indexes=roles["calibration"],
                spec=specification,
                target_scenario="base",
                compute_backend=compute_backend,
                seed=seed,
            )
            model_path = (
                artifact_root
                / symbol
                / specification.candidate_id
                / f"seed-{seed}.json"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            save_categorical_payoff_model(model_path, model)
            output[specification.candidate_id].append(
                {
                    "seed": seed,
                    "model_sha256": model.model_sha256,
                    "backend_kind": model.backend_kind,
                    "backend_device": model.backend_device,
                    "best_iterations": dict(model.best_iterations),
                    "temperature": dict(model.temperature),
                    "calibration_log_loss_before": dict(
                        model.calibration_log_loss_before
                    ),
                    "calibration_log_loss_after": dict(
                        model.calibration_log_loss_after
                    ),
                    "class_support": {
                        side: {
                            role: list(values)
                            for role, values in model.class_support[side].items()
                        }
                        for side in SIDES
                    },
                    "training_seconds": time.perf_counter() - started,
                    "model": _artifact(model_path),
                }
            )
            del model
    return output


def _evaluate_symbol_models(
    *,
    symbol: str,
    dataset,
    targets,
    deterministic_dataset,
    ai_dataset,
    roles: Mapping[str, np.ndarray],
    artifacts: dict[str, list[dict[str, object]]],
    artifact_root: Path,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    evaluation = roles["evaluation"]
    candidate_reports: dict[str, dict[str, object]] = {}
    base_traces: dict[str, object] = {}
    stress_traces: dict[str, object] = {}
    for candidate in CANDIDATES:
        categorical = candidate != "direct_mean_lightgbm"
        candidate_dataset = (
            ai_dataset
            if candidate == "categorical_payoff_lightgbm_fincast"
            else deterministic_dataset
        )
        predictions: list[object] = []
        expected_by_side: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        probability_by_side: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        for record in artifacts[candidate]:
            seed = int(record["seed"])
            model_path = Path(str(record["model"]["path"]))
            _progress("evaluate", symbol=symbol, candidate=candidate, seed=seed)
            if categorical:
                model = load_categorical_payoff_model(model_path)
                prediction = predict_categorical_payoff_model(
                    model,
                    candidate_dataset,
                    evaluation,
                )
                metrics = categorical_forecast_metrics(
                    model,
                    candidate_dataset,
                    evaluation,
                    prediction,
                )
                action = prediction.action_values
                expected_by_side["long"].append(
                    np.asarray(action.long_mean_bps, dtype=np.float64)
                )
                expected_by_side["short"].append(
                    np.asarray(action.short_mean_bps, dtype=np.float64)
                )
                probability_by_side["long"].append(
                    np.asarray(action.long_profitable_probability, dtype=np.float64)
                )
                probability_by_side["short"].append(
                    np.asarray(action.short_profitable_probability, dtype=np.float64)
                )
                prediction_artifact = _save_prediction(
                    artifact_root / symbol / candidate / f"seed-{seed}-evaluation.npz",
                    endpoint_indexes=action.endpoint_indexes,
                    long_mean_bps=action.long_mean_bps,
                    short_mean_bps=action.short_mean_bps,
                    long_profitable_probability=action.long_profitable_probability,
                    short_profitable_probability=action.short_profitable_probability,
                    long_lower_bps=action.long_lower_bps,
                    short_lower_bps=action.short_lower_bps,
                    long_upper_bps=action.long_upper_bps,
                    short_upper_bps=action.short_upper_bps,
                    long_cvar10_bps=prediction.long_cvar10_bps,
                    short_cvar10_bps=prediction.short_cvar10_bps,
                    long_probabilities=prediction.long_probabilities.astype(np.float32),
                    short_probabilities=prediction.short_probabilities.astype(
                        np.float32
                    ),
                )
            else:
                model = load_direct_payoff_model(model_path)
                prediction = predict_direct_payoff_model(
                    model,
                    candidate_dataset,
                    evaluation,
                )
                metrics = direct_forecast_metrics(
                    model,
                    candidate_dataset,
                    evaluation,
                    prediction,
                )
                action = prediction
                expected_by_side["long"].append(
                    np.asarray(action.long_mean_bps, dtype=np.float64)
                )
                expected_by_side["short"].append(
                    np.asarray(action.short_mean_bps, dtype=np.float64)
                )
                prediction_artifact = _save_prediction(
                    artifact_root / symbol / candidate / f"seed-{seed}-evaluation.npz",
                    endpoint_indexes=action.endpoint_indexes,
                    long_mean_bps=action.long_mean_bps,
                    short_mean_bps=action.short_mean_bps,
                )
            record["forecast_metrics"] = metrics
            record["prediction"] = prediction_artifact
            predictions.append(prediction)
            del model
        endpoint_indexes = np.asarray(
            (
                predictions[0].action_values.endpoint_indexes
                if categorical
                else predictions[0].endpoint_indexes
            ),
            dtype=np.int64,
        )
        for prediction in predictions[1:]:
            comparison = np.asarray(
                (
                    prediction.action_values.endpoint_indexes
                    if categorical
                    else prediction.endpoint_indexes
                ),
                dtype=np.int64,
            )
            if not np.array_equal(endpoint_indexes, comparison):
                raise ValueError(f"{symbol} {candidate} seed endpoints differ")
        score = ensemble_action_score(
            endpoint_indexes,
            long_means=expected_by_side["long"],
            short_means=expected_by_side["short"],
            long_probabilities=(probability_by_side["long"] if categorical else None),
            short_probabilities=(probability_by_side["short"] if categorical else None),
        )
        base, stress, overlap_violations = base_and_paired_stress_traces(
            dataset,
            targets,
            score,
            extra_stress_slippage_bps_per_side=2.0,
        )
        base_traces[candidate] = base
        stress_traces[candidate] = stress
        candidate_reports[candidate] = {
            "models": artifacts[candidate],
            "selection": {
                "rows": score.rows,
                "eligible_rows": int(np.sum(score.eligible)),
                "long_eligible_rows": int(np.sum(score.side == 1)),
                "short_eligible_rows": int(np.sum(score.side == -1)),
                "probability_gate_applied": categorical,
                "minimum_seed_mean_gate_bps": 0.0,
                "mean_profitable_probability_gate": 0.50 if categorical else None,
            },
            "seed_agreement": {
                side: pairwise_seed_spearman(expected_by_side[side]) for side in SIDES
            },
            "base_trace": base.asdict(),
            "stress_trace": stress.asdict(),
            "stress_same_ledger": True,
            "stress_extra_slippage_bps_per_side": 2.0,
            "stress_overlap_violations": overlap_violations,
        }
        del predictions
    return candidate_reports, base_traces, stress_traces


def _distribution_gate(
    symbol_reports: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    candidate: str,
) -> dict[str, object]:
    reasons: list[str] = []
    observations: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        report = symbol_reports[symbol][candidate]
        for side in SIDES:
            agreement = float(report["seed_agreement"][side]["minimum_spearman"])
            if agreement < 0.50:
                reasons.append(f"{symbol}_{side}_seed_spearman_below_0.50")
        for model in report["models"]:
            seed = int(model["seed"])
            for side in SIDES:
                metrics = model["forecast_metrics"][side]
                row = {"symbol": symbol, "seed": seed, "side": side, **metrics}
                observations.append(row)
                comparisons = (
                    (
                        "multinomial_log_loss_skill_nonpositive",
                        float(metrics["multinomial_log_loss_skill"]) <= 0.0,
                    ),
                    (
                        "ranked_probability_skill_nonpositive",
                        float(metrics["ranked_probability_skill"]) <= 0.0,
                    ),
                    (
                        "expected_payoff_mse_skill_nonpositive",
                        float(metrics["expected_payoff_mse_skill"]) <= 0.0,
                    ),
                    (
                        "expected_payoff_spearman_below_0.03",
                        float(metrics["expected_payoff_spearman"]) < 0.03,
                    ),
                    (
                        "calibration_error_above_0.05",
                        float(metrics["maximum_10_bin_calibration_error"]) > 0.05,
                    ),
                    (
                        "positive_daily_brier_skill_below_4",
                        int(metrics["days_with_positive_brier_skill"]) < 4,
                    ),
                )
                reasons.extend(
                    f"{symbol}_seed{seed}_{side}_{label}"
                    for label, failed in comparisons
                    if failed
                )
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "observations": observations,
    }


def _economic_gate(
    *,
    candidate: str,
    symbol_reports: Mapping[str, Mapping[str, Mapping[str, object]]],
    base_portfolio: Mapping[str, object],
    stress_portfolio: Mapping[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    base_metrics = base_portfolio["metrics"]
    stress_metrics = stress_portfolio["metrics"]
    assert isinstance(base_metrics, Mapping)
    assert isinstance(stress_metrics, Mapping)
    for scenario, metrics in (("base", base_metrics), ("stress", stress_metrics)):
        if int(metrics["trades"]) < 30:
            reasons.append(f"{scenario}_closed_trades_below_30")
        if float(metrics["total_net_bps"]) <= 0.0:
            reasons.append(f"{scenario}_portfolio_return_nonpositive")
        profit_factor = metrics["profit_factor"]
        if profit_factor is None or float(profit_factor) <= 1.0:
            reasons.append(f"{scenario}_profit_factor_not_above_one")
        if float(metrics["max_drawdown_bps"]) / 10_000.0 > 0.04:
            reasons.append(f"{scenario}_drawdown_above_4pct")
    positive_symbols = sum(
        float(
            symbol_reports[symbol][candidate]["base_trace"]["metrics"]["total_net_bps"]
        )
        > 0.0
        for symbol in SYMBOLS
    )
    if positive_symbols < 2:
        reasons.append("positive_symbols_below_two")
    if float(base_portfolio["maximum_single_symbol_positive_pnl_share"]) > 0.70:
        reasons.append("single_symbol_positive_pnl_share_above_0.70")
    overlaps = sum(
        int(symbol_reports[symbol][candidate]["stress_overlap_violations"])
        for symbol in SYMBOLS
    )
    if overlaps:
        reasons.append("paired_stress_ledger_overlap_detected")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "positive_symbols": positive_symbols,
        "stress_overlap_violations": overlaps,
    }


def _average_metric(
    symbol_reports: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    candidate: str,
    metric: str,
) -> float:
    values = [
        float(model["forecast_metrics"][side][metric])
        for symbol in SYMBOLS
        for model in symbol_reports[symbol][candidate]["models"]
        for side in SIDES
    ]
    return float(np.mean(values))


def _daily_uplift(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> tuple[float, list[dict[str, float | int]]]:
    control_values = {
        int(row["utc_day_id"]): float(row["net_bps"])
        for row in control["daily_net_bps"]
    }
    treatment_values = {
        int(row["utc_day_id"]): float(row["net_bps"])
        for row in treatment["daily_net_bps"]
    }
    first, last = _utc_bounds("2023-06-09", "2023-06-14")
    days = range(first // _DAY_MS, last // _DAY_MS + 1)
    rows = [
        {
            "utc_day_id": day,
            "control_net_bps": control_values.get(day, 0.0),
            "treatment_net_bps": treatment_values.get(day, 0.0),
            "uplift_bps": treatment_values.get(day, 0.0) - control_values.get(day, 0.0),
        }
        for day in days
    ]
    return float(np.mean([float(row["uplift_bps"]) for row in rows])), rows


def _ai_uplift_gate(
    *,
    symbol_reports: Mapping[str, Mapping[str, Mapping[str, object]]],
    portfolios: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    control = "categorical_payoff_lightgbm"
    treatment = "categorical_payoff_lightgbm_fincast"
    control_rps = _average_metric(
        symbol_reports,
        candidate=control,
        metric="ranked_probability_skill",
    )
    treatment_rps = _average_metric(
        symbol_reports,
        candidate=treatment,
        metric="ranked_probability_skill",
    )
    control_spearman = _average_metric(
        symbol_reports,
        candidate=control,
        metric="expected_payoff_spearman",
    )
    treatment_spearman = _average_metric(
        symbol_reports,
        candidate=treatment,
        metric="expected_payoff_spearman",
    )
    control_mse = _average_metric(
        symbol_reports,
        candidate=control,
        metric="expected_payoff_mse_bps2",
    )
    treatment_mse = _average_metric(
        symbol_reports,
        candidate=treatment,
        metric="expected_payoff_mse_bps2",
    )
    scenario_economics: dict[str, object] = {}
    reasons: list[str] = []
    for scenario in ("base", "stress"):
        control_portfolio = portfolios[control][scenario]
        treatment_portfolio = portfolios[treatment][scenario]
        mean_uplift, daily = _daily_uplift(control_portfolio, treatment_portfolio)
        control_metrics = control_portfolio["metrics"]
        treatment_metrics = treatment_portfolio["metrics"]
        control_pf = control_metrics["profit_factor"]
        treatment_pf = treatment_metrics["profit_factor"]
        scenario_economics[scenario] = {
            "mean_daily_uplift_bps": mean_uplift,
            "daily": daily,
            "control_max_drawdown_bps": control_metrics["max_drawdown_bps"],
            "treatment_max_drawdown_bps": treatment_metrics["max_drawdown_bps"],
            "control_profit_factor": control_pf,
            "treatment_profit_factor": treatment_pf,
        }
        if mean_uplift <= 0.0:
            reasons.append(f"{scenario}_mean_daily_uplift_nonpositive")
        if float(treatment_metrics["max_drawdown_bps"]) > float(
            control_metrics["max_drawdown_bps"]
        ):
            reasons.append(f"{scenario}_drawdown_worse_than_control")
        if (
            control_pf is None
            or treatment_pf is None
            or float(treatment_pf) < float(control_pf)
        ):
            reasons.append(f"{scenario}_profit_factor_worse_than_control")
    if treatment_rps - control_rps < 0.005:
        reasons.append("average_ranked_probability_skill_improvement_below_0.005")
    if treatment_spearman - control_spearman < 0.005:
        reasons.append("average_expected_payoff_spearman_improvement_below_0.005")
    if treatment_mse > control_mse * 1.01:
        reasons.append("average_expected_payoff_mse_degraded_above_1pct")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "control_average_ranked_probability_skill": control_rps,
        "treatment_average_ranked_probability_skill": treatment_rps,
        "ranked_probability_skill_improvement": treatment_rps - control_rps,
        "control_average_expected_payoff_spearman": control_spearman,
        "treatment_average_expected_payoff_spearman": treatment_spearman,
        "expected_payoff_spearman_improvement": treatment_spearman - control_spearman,
        "control_average_expected_payoff_mse_bps2": control_mse,
        "treatment_average_expected_payoff_mse_bps2": treatment_mse,
        "expected_payoff_mse_ratio": treatment_mse / max(control_mse, 1e-15),
        "economics": scenario_economics,
    }


def run_round51(
    *,
    design_path: Path,
    binding_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    evidence_root: Path,
    fincast_source: Path,
    fincast_checkpoint: Path,
    compute_backend: str,
    fincast_backend: str,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    started = time.perf_counter()
    design, design_sha = _validate_design(design_path)
    _binding, binding_sha, implementation_commit = _validate_binding(
        binding_path,
        design_sha256=design_sha,
    )
    lightgbm_parameters, lightgbm_backend_kind, lightgbm_backend_device = (
        lightgbm_backend_parameters(
            compute_backend,
            SEEDS[0],
            reproducible=True,
        )
    )
    if (
        lightgbm_backend_kind != "opencl"
        or lightgbm_parameters.get("device_type") != "gpu"
        or lightgbm_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError(
            "Round 51 requires LightGBM OpenCL with FP64 accumulation; "
            f"resolved {lightgbm_backend_kind}:{lightgbm_backend_device}"
        )
    data_contract = design["data_contract"]
    execution = design["execution_target"]
    model_contract = design["model_contract"]
    roles_contract = design["chronological_roles"]
    assert isinstance(data_contract, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(model_contract, Mapping)
    assert isinstance(roles_contract, Mapping)
    evidence_root.mkdir(parents=True, exist_ok=True)
    artifact_root = evidence_root / "artifacts"
    feature_cache = evidence_root / "fincast-feature-cache"
    prepared: dict[str, dict[str, object]] = {}
    data_evidence: dict[str, object] = {}

    for symbol in SYMBOLS:
        real = _load_real_symbol_data(
            symbol=symbol,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            memory_limit=memory_limit,
            threads=threads,
            data_contract=data_contract,
            execution=execution,
        )
        dataset = real["dataset"]
        targets = real["targets"]
        fincast_matrix, fincast_evidence, fincast_artifact = _load_or_extract_fincast(
            symbol=symbol,
            dataset=dataset,
            second_ms=real["second_ms"],
            close_mid=real["close_mid"],
            cache_directory=feature_cache,
            source=fincast_source,
            checkpoint=fincast_checkpoint,
            backend=fincast_backend,
        )
        deterministic_dataset = build_categorical_payoff_dataset(
            dataset,
            targets,
            target_scenario="base",
        )
        ai_dataset = build_categorical_payoff_dataset(
            dataset,
            targets,
            target_scenario="base",
            extra_feature_names=fincast_feature_names(),
            extra_features=fincast_matrix,
        )
        if (
            deterministic_dataset.rows != ai_dataset.rows
            or not np.array_equal(
                deterministic_dataset.decision_time_ms,
                ai_dataset.decision_time_ms,
            )
            or not np.array_equal(
                deterministic_dataset.source_row_indexes,
                ai_dataset.source_row_indexes,
            )
            or deterministic_dataset.long_net_bps.tobytes()
            != ai_dataset.long_net_bps.tobytes()
            or deterministic_dataset.short_net_bps.tobytes()
            != ai_dataset.short_net_bps.tobytes()
        ):
            raise ValueError(f"{symbol} matched FinCast candidate rows drifted")
        roles = _role_indexes(
            deterministic_dataset.decision_time_ms,
            roles_contract,
        )
        prepared[symbol] = {
            "dataset": dataset,
            "targets": targets,
            "deterministic_dataset": deterministic_dataset,
            "ai_dataset": ai_dataset,
            "roles": roles,
        }
        data_evidence[symbol] = {
            "microstructure_rows": dataset.rows,
            "valid_barrier_rows": targets.valid_rows,
            "categorical_rows": deterministic_dataset.rows,
            "microstructure_dataset_sha256": real["dataset_sha256"],
            "deterministic_dataset_sha256": deterministic_dataset.dataset_sha256,
            "ai_dataset_sha256": ai_dataset.dataset_sha256,
            "microstructure_feature_version": dataset.feature_version,
            "microstructure_feature_count": len(dataset.feature_names),
            "ai_feature_count": len(ai_dataset.feature_names),
            "dataset_cache_state": real["cache_state"],
            "source_evidence": real["source_evidence"],
            "barrier_summary": targets.summary(),
            "roles": {
                name: {
                    "rows": len(indexes),
                    "first_decision_time_ms": int(
                        deterministic_dataset.decision_time_ms[indexes[0]]
                    ),
                    "last_decision_time_ms": int(
                        deterministic_dataset.decision_time_ms[indexes[-1]]
                    ),
                }
                for name, indexes in roles.items()
            },
            "fincast": fincast_evidence,
            "fincast_feature_artifact": fincast_artifact,
            "synthetic_rows": 0,
        }
        del real["second_ms"], real["close_mid"], fincast_matrix
        gc.collect()

    model_artifacts: dict[str, dict[str, list[dict[str, object]]]] = {}
    for symbol in SYMBOLS:
        state = prepared[symbol]
        model_artifacts[symbol] = _train_symbol_models(
            symbol=symbol,
            deterministic_dataset=state["deterministic_dataset"],
            ai_dataset=state["ai_dataset"],
            roles=state["roles"],
            model_contract=model_contract,
            compute_backend=compute_backend,
            artifact_root=artifact_root,
        )
    _progress("all-models-trained", models=len(SYMBOLS) * len(CANDIDATES) * len(SEEDS))

    symbol_reports: dict[str, dict[str, dict[str, object]]] = {}
    base_traces: dict[str, dict[str, object]] = {
        candidate: {} for candidate in CANDIDATES
    }
    stress_traces: dict[str, dict[str, object]] = {
        candidate: {} for candidate in CANDIDATES
    }
    for symbol in SYMBOLS:
        state = prepared[symbol]
        report, symbol_base, symbol_stress = _evaluate_symbol_models(
            symbol=symbol,
            dataset=state["dataset"],
            targets=state["targets"],
            deterministic_dataset=state["deterministic_dataset"],
            ai_dataset=state["ai_dataset"],
            roles=state["roles"],
            artifacts=model_artifacts[symbol],
            artifact_root=artifact_root,
        )
        symbol_reports[symbol] = report
        for candidate in CANDIDATES:
            base_traces[candidate][symbol] = symbol_base[candidate]
            stress_traces[candidate][symbol] = symbol_stress[candidate]

    portfolios: dict[str, dict[str, dict[str, object]]] = {}
    economic_gates: dict[str, dict[str, object]] = {}
    for candidate in CANDIDATES:
        base_portfolio = portfolio_trace_metrics(
            base_traces[candidate],
            symbol_weight=1.0 / len(SYMBOLS),
        )
        stress_portfolio = portfolio_trace_metrics(
            stress_traces[candidate],
            symbol_weight=1.0 / len(SYMBOLS),
        )
        portfolios[candidate] = {
            "base": base_portfolio,
            "stress": stress_portfolio,
        }
        economic_gates[candidate] = _economic_gate(
            candidate=candidate,
            symbol_reports=symbol_reports,
            base_portfolio=base_portfolio,
            stress_portfolio=stress_portfolio,
        )
    distribution_gates = {
        candidate: _distribution_gate(symbol_reports, candidate=candidate)
        for candidate in (
            "categorical_payoff_lightgbm",
            "categorical_payoff_lightgbm_fincast",
        )
    }
    ai_gate = _ai_uplift_gate(
        symbol_reports=symbol_reports,
        portfolios=portfolios,
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "runtime_seconds": time.perf_counter() - started,
        "runtime_resources": {
            "compute_backend_requested": compute_backend,
            "lightgbm_backend_kind": lightgbm_backend_kind,
            "lightgbm_backend_device": lightgbm_backend_device,
            "lightgbm_gpu_use_dp": lightgbm_parameters["gpu_use_dp"],
            "fincast_backend_requested": fincast_backend,
            "duckdb_memory_limit": memory_limit,
            "duckdb_threads": threads,
            "warehouse": str(warehouse_path.resolve()),
            "cache_root": str(cache_root.resolve()),
        },
        "data": data_evidence,
        "symbol_results": symbol_reports,
        "portfolio_results": portfolios,
        "distribution_gates": distribution_gates,
        "economic_gates": economic_gates,
        "ai_uplift_gate": ai_gate,
        "round_gate": {
            "passed": bool(
                distribution_gates["categorical_payoff_lightgbm"]["passed"]
                and distribution_gates["categorical_payoff_lightgbm_fincast"]["passed"]
                and economic_gates["categorical_payoff_lightgbm"]["passed"]
                and economic_gates["categorical_payoff_lightgbm_fincast"]["passed"]
                and ai_gate["passed"]
            ),
            "promotion_permitted": False,
        },
        "claims": {
            "selection_contaminated": True,
            "beta_research_only": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "source_market_rows_synthetic": 0,
        },
    }
    report["report_canonical_sha256"] = _canonical_sha256(report)
    report_path = evidence_root / "report.json"
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    _progress(
        "complete",
        report=report_path,
        canonical_sha256=report["report_canonical_sha256"],
        round_gate=report["round_gate"]["passed"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs"
        / "model-research"
        / "action-value"
        / "round-051-categorical-payoff-fincast-design.json",
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--fincast-source", type=Path, required=True)
    parser.add_argument("--fincast-checkpoint", type=Path, required=True)
    parser.add_argument("--compute-backend", default="directml")
    parser.add_argument(
        "--fincast-backend", choices=("directml", "cpu"), default="directml"
    )
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1 or args.threads > 64:
        raise ValueError("Round 51 DuckDB threads must lie in [1, 64]")
    report = run_round51(
        design_path=args.design,
        binding_path=args.binding,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        fincast_source=args.fincast_source,
        fincast_checkpoint=args.fincast_checkpoint,
        compute_backend=args.compute_backend,
        fincast_backend=args.fincast_backend,
        memory_limit=str(args.memory_limit).upper(),
        threads=args.threads,
    )
    if args.json:
        print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
