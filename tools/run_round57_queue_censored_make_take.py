"""Run the frozen Round 57 queue-censored make/take mechanism screen."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.make_take_action_features import (  # noqa: E402
    MakeTakeActionFeatureBatch,
    MakeTakeFeatureSpec,
    build_make_take_action_features,
)
from simple_ai_trading.make_take_action_values import (  # noqa: E402
    MakeTakeActionValueBatch,
    build_make_take_action_values,
)
from simple_ai_trading.make_take_evaluation import (  # noqa: E402
    MakeTakeEconomicGateSpec,
    evaluate_make_take_policy,
)
from simple_ai_trading.make_take_historical_source import (  # noqa: E402
    load_historical_day_path,
    load_historical_placement_quotes,
    load_historical_trade_chunk,
    select_role_decision_indexes,
    utc_day_chunks,
)
from simple_ai_trading.make_take_payoff_lightgbm import (  # noqa: E402
    MAKE_TAKE_PAYOFF_SEEDS,
    MakeTakePayoffLightGBMSpec,
    build_make_take_payoff_lightgbm_ensemble,
    load_make_take_payoff_lightgbm_model,
    predict_make_take_payoff_lightgbm_model,
    save_make_take_payoff_lightgbm_model,
    train_make_take_payoff_lightgbm_model,
)
from simple_ai_trading.make_take_payoff_panel import (  # noqa: E402
    MakeTakeConditionalPayoffPanel,
    build_make_take_conditional_payoff_panel,
)
from simple_ai_trading.make_take_policy import (  # noqa: E402
    MakeTakePolicySpec,
    calibrate_make_take_policy,
)
from simple_ai_trading.make_take_predictive_evaluation import (  # noqa: E402
    MakeTakePredictiveEvaluation,
    build_make_take_predictive_evaluation,
)
from simple_ai_trading.make_take_scenario_entries import (  # noqa: E402
    build_make_take_scenario_entries,
)
from simple_ai_trading.make_take_targets import (  # noqa: E402
    MakeTakeTargetBatch,
    build_make_take_targets,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    AdaptiveBarrierSpec,
    volatility_scaled_barriers,
)
from simple_ai_trading.microstructure_cache import (  # noqa: E402
    load_microstructure_dataset_cache,
    microstructure_dataset_cache_key,
    microstructure_dataset_fingerprint,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    verify_executable_microstructure_source,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat  # noqa: E402
from simple_ai_trading.queue_censored_actions import (  # noqa: E402
    PassiveFillRequest,
    build_chunked_queue_censored_inputs,
)
from simple_ai_trading.queue_fill_lightgbm import (  # noqa: E402
    QUEUE_FILL_SEEDS,
    QueueFillLightGBMSpec,
    build_queue_fill_lightgbm_ensemble,
    load_queue_fill_lightgbm_model,
    predict_queue_fill_lightgbm_model,
    save_queue_fill_lightgbm_model,
    train_queue_fill_lightgbm_model,
)
from simple_ai_trading.queue_fill_survival import (  # noqa: E402
    PassiveFillSurvivalPanel,
    build_passive_fill_survival_panel,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 57
DESIGN_SCHEMA = "round-057-queue-censored-make-take-design-v1"
CONTRACT_SCHEMA = "round-057-queue-censored-make-take-execution-contract-v1"
BINDING_SCHEMA = "round-057-queue-censored-make-take-execution-binding-v1"
REPORT_SCHEMA = "round-057-queue-censored-make-take-report-v1"
ROLE_ORDER = (
    "training",
    "early_stop",
    "probability_calibration",
    "policy_calibration",
    "evaluation",
)
POLICY_ROLES = ("policy_calibration", "evaluation")
DAY_MS = 86_400_000


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
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 57 Git evidence command failed") from exc


def _progress(phase: str, **details: object) -> None:
    print(
        _canonical_json(
            {
                "observed_at_ms": int(time.time() * 1_000),
                "phase": phase,
                **details,
            }
        ),
        flush=True,
    )


def _json_dataclass(value: object) -> object:
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def load_round57_contract(
    design_path: Path,
    contract_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design = _read_object(design_path, "Round 57 design")
    canonical_design = dict(design)
    design_sha = str(canonical_design.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or design_sha != _canonical_sha256(canonical_design)
    ):
        raise ValueError("Round 57 design identity is invalid")
    contract = _read_object(contract_path, "Round 57 execution contract")
    canonical_contract = dict(contract)
    contract_sha = str(canonical_contract.pop("contract_sha256", ""))
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("round") != ROUND
        or contract.get("status") != "frozen"
        or contract.get("design_sha256") != design_sha
        or contract_sha != _canonical_sha256(canonical_contract)
        or any(value is not False for value in contract["authority"].values())
        or set(contract["roles"]) != set(ROLE_ORDER)
        or tuple(contract["source"]["symbols"])
        != ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        or tuple(contract["model_aggregation"]["seeds"]) != QUEUE_FILL_SEEDS
        or QUEUE_FILL_SEEDS != MAKE_TAKE_PAYOFF_SEEDS
    ):
        raise ValueError("Round 57 execution contract identity is invalid")
    if contract["predictive_gates"] != {
        "fill_integrated_brier_skill_strictly_above": 0.0,
        "fill_log_loss_skill_strictly_above": 0.0,
        "payoff_mean_mse_skill_strictly_above": 0.0,
        "payoff_minimum_spearman": 0.02,
        "payoff_q20_pinball_skill_strictly_above": 0.0,
    }:
        raise ValueError("Round 57 predictive gate contract drifted")
    expected_specs = (
        (MakeTakeFeatureSpec(), "feature_spec"),
        (QueueFillLightGBMSpec(), "queue_fill_model_spec"),
        (MakeTakePayoffLightGBMSpec(), "payoff_model_spec"),
        (MakeTakePolicySpec(), "policy_spec"),
        (MakeTakeEconomicGateSpec(), "economic_gate_spec"),
    )
    if any(_json_dataclass(value) != contract[name] for value, name in expected_specs):
        raise ValueError("Round 57 implementation specification drifted")
    AdaptiveBarrierSpec(**contract["barrier_spec"])
    roles = contract["roles"]
    previous_end = None
    for name in ROLE_ORDER:
        role = roles[name]
        start = int(role["start_ms"])
        end = int(role["end_ms_exclusive"])
        if (
            start < 0
            or end <= start
            or start % DAY_MS
            or end % DAY_MS
            or (previous_end is not None and start != previous_end)
        ):
            raise ValueError("Round 57 chronological role contract drifted")
        previous_end = end
    return design, contract, design_sha, contract_sha


def validate_round57_binding(
    path: Path,
    *,
    design_sha256: str,
    contract_sha256: str,
) -> tuple[dict[str, object], str]:
    binding = _read_object(path, "Round 57 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    implementation_commit = str(binding.get("implementation_commit", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or binding.get("contract_sha256") != contract_sha256
        or claimed != _canonical_sha256(canonical)
        or len(implementation_commit) != 40
    ):
        raise ValueError("Round 57 execution binding identity is invalid")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                "HEAD",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 57 implementation commit is not an ancestor") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 57 execution binding has no implementation blobs")
    for row in blobs:
        if not isinstance(row, dict) or set(row) != {"path", "git_blob_oid"}:
            raise ValueError("Round 57 implementation blob entry is invalid")
        relative = str(row["path"])
        expected = str(row["git_blob_oid"])
        if (
            _git("rev-parse", f"{implementation_commit}:{relative}") != expected
            or _git("rev-parse", f"HEAD:{relative}") != expected
        ):
            raise ValueError(f"Round 57 bound implementation drifted: {relative}")
    return binding, claimed


@dataclass(frozen=True)
class RoleArtifacts:
    action_features: MakeTakeActionFeatureBatch | None
    fill_panel: PassiveFillSurvivalPanel
    payoff_panel: MakeTakeConditionalPayoffPanel
    base_target: MakeTakeTargetBatch | None
    stress_target: MakeTakeTargetBatch | None


class _DayPathCache:
    def __init__(
        self,
        loader: Callable[[int], Mapping[str, np.ndarray]],
        *,
        capacity: int,
    ) -> None:
        self._loader = loader
        self._capacity = max(1, int(capacity))
        self._values: OrderedDict[int, Mapping[str, np.ndarray]] = OrderedDict()

    def __call__(self, day_start_ms: int) -> Mapping[str, np.ndarray]:
        key = int(day_start_ms)
        existing = self._values.pop(key, None)
        if existing is not None:
            self._values[key] = existing
            return existing
        value = self._loader(key)
        self._values[key] = value
        while len(self._values) > self._capacity:
            self._values.popitem(last=False)
        return value


def _source_cache_parameters(
    contract: Mapping[str, object],
    *,
    symbol: str,
    source_evidence: Mapping[str, object],
) -> dict[str, object]:
    source = contract["source"]
    assert isinstance(source, Mapping)
    return {
        "symbol": symbol,
        "requested_start_ms": int(source["requested_start_ms"]),
        "requested_end_ms": int(source["requested_end_ms"]),
        "horizon_seconds": int(source["horizon_seconds"]),
        "total_latency_ms": int(source["total_latency_ms"]),
        "taker_fee_bps": float(source["taker_fee_bps"]),
        "additional_slippage_bps_per_side": float(
            source["additional_slippage_bps_per_side"]
        ),
        "reference_order_notional_quote": float(
            source["reference_order_notional_quote"]
        ),
        "max_l1_participation": float(source["max_l1_participation"]),
        "max_quote_age_ms": int(source["max_quote_age_ms"]),
        "decision_cadence_seconds": int(source["decision_cadence_seconds"]),
        "require_full_history_inventory": bool(
            source["require_full_history_inventory"]
        ),
        "source_evidence": dict(source_evidence),
        "feature_version": str(source["feature_version"]),
    }


def _bound_source_row(
    binding: Mapping[str, object],
    symbol: str,
) -> Mapping[str, object]:
    values = binding.get("source_cache")
    if not isinstance(values, list):
        raise ValueError("Round 57 binding has no source-cache evidence")
    matches = [
        value
        for value in values
        if isinstance(value, dict) and value.get("symbol") == symbol
    ]
    if len(matches) != 1:
        raise ValueError(f"Round 57 {symbol} source-cache binding is ambiguous")
    return matches[0]


def _verify_bound_dataset(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    contract: Mapping[str, object],
    binding: Mapping[str, object],
):
    source = contract["source"]
    assert isinstance(source, Mapping)
    verified = verify_executable_microstructure_source(
        warehouse,
        symbol=symbol,
        start_ms=int(source["requested_start_ms"]),
        end_ms=int(source["requested_end_ms"]),
        require_full_history_inventory=bool(source["require_full_history_inventory"]),
        feature_version=str(source["feature_version"]),
    )
    evidence = dict(verified.evidence)
    parameters = _source_cache_parameters(
        contract,
        symbol=symbol,
        source_evidence=evidence,
    )
    cache_key = microstructure_dataset_cache_key(**parameters)
    expected = _bound_source_row(binding, symbol)
    certificate = evidence.get("corpus_certificate")
    if not isinstance(certificate, Mapping):
        raise ValueError(f"Round 57 {symbol} source certificate is missing")
    expected_identity = {
        "symbol": symbol,
        "cache_key": cache_key,
        "source_evidence_sha256": _canonical_sha256(evidence),
        "corpus_certificate_sha256": str(certificate.get("certificate_sha256", "")),
    }
    if any(expected.get(name) != value for name, value in expected_identity.items()):
        raise ValueError(f"Round 57 {symbol} source evidence differs from binding")
    dataset = load_microstructure_dataset_cache(warehouse, **parameters)
    if dataset is None:
        raise ValueError(f"Round 57 {symbol} exact frozen cache is absent")
    fingerprint = microstructure_dataset_fingerprint(dataset)
    if (
        expected.get("dataset_fingerprint") != fingerprint
        or int(expected.get("rows", -1)) != dataset.rows
        or int(expected.get("first_decision_time_ms", -1))
        != int(dataset.decision_time_ms[0])
        or int(expected.get("last_decision_time_ms", -1))
        != int(dataset.decision_time_ms[-1])
    ):
        raise ValueError(f"Round 57 {symbol} cache rows differ from binding")
    return dataset, fingerprint, expected_identity


def _role_window(
    contract: Mapping[str, object],
    role_name: str,
) -> tuple[int, int]:
    roles = contract["roles"]
    assert isinstance(roles, Mapping)
    value = roles[role_name]
    assert isinstance(value, Mapping)
    return int(value["start_ms"]), int(value["end_ms_exclusive"])


def _build_role_artifacts(
    *,
    warehouse: MicrostructureWarehouse,
    dataset,
    dataset_sha256: str,
    symbol: str,
    role_name: str,
    contract: Mapping[str, object],
    path_cache_days: int,
    heartbeat_seconds: float,
) -> tuple[RoleArtifacts, dict[str, object]]:
    lifecycle = contract["lifecycle"]
    source = contract["source"]
    assert isinstance(lifecycle, Mapping)
    assert isinstance(source, Mapping)
    feature_contract = contract["feature_spec"]
    barrier_contract = contract["barrier_spec"]
    assert isinstance(feature_contract, Mapping)
    assert isinstance(barrier_contract, Mapping)
    base_latency_ms = int(feature_contract["placement_latency_ms"])
    stress_latency_ms = int(barrier_contract["stress_protection_delay_ms"])
    role_start, role_end = _role_window(contract, role_name)
    indexes = select_role_decision_indexes(
        dataset,
        role_start_ms=role_start,
        role_end_ms_exclusive=role_end,
        feature_warmup_ms=int(lifecycle["feature_warmup_ms"]),
        maximum_lifecycle_ms=int(lifecycle["maximum_stress_lifecycle_ms"]),
    )
    decisions = np.asarray(dataset.decision_time_ms[indexes], dtype=np.int64)
    stress_all = load_historical_placement_quotes(
        warehouse.connect(),
        symbol=symbol,
        decision_time_ms=decisions,
        placement_latency_ms=stress_latency_ms,
        max_quote_age_ms=int(source["max_quote_age_ms"]),
    )
    valid_local = np.flatnonzero(stress_all.valid).astype(np.int64, copy=False)
    if valid_local.size == 0:
        raise ValueError(f"Round 57 {symbol} {role_name} has no valid stress quotes")
    indexes = indexes[valid_local]
    decisions = decisions[valid_local]
    stress = stress_all.select_rows(valid_local)
    base_bid = np.asarray(dataset.entry_bid_price[indexes], dtype=np.float64)
    base_ask = np.asarray(dataset.entry_ask_price[indexes], dtype=np.float64)
    base_bid_qty = np.asarray(dataset.entry_bid_qty[indexes], dtype=np.float64)
    base_ask_qty = np.asarray(dataset.entry_ask_qty[indexes], dtype=np.float64)
    requests = (
        PassiveFillRequest(
            name="base_long",
            buyer_is_maker=True,
            arrival_time_ms=decisions + base_latency_ms,
            placement_price=base_bid,
            queue_ahead_quantity=base_bid_qty,
        ),
        PassiveFillRequest(
            name="base_short",
            buyer_is_maker=False,
            arrival_time_ms=decisions + base_latency_ms,
            placement_price=base_ask,
            queue_ahead_quantity=base_ask_qty,
        ),
        PassiveFillRequest(
            name="stress_long",
            buyer_is_maker=True,
            arrival_time_ms=decisions + stress_latency_ms,
            placement_price=stress.bid_price,
            queue_ahead_quantity=stress.bid_quantity,
        ),
        PassiveFillRequest(
            name="stress_short",
            buyer_is_maker=False,
            arrival_time_ms=decisions + stress_latency_ms,
            placement_price=stress.ask_price,
            queue_ahead_quantity=stress.ask_quantity,
        ),
    )
    chunks = utc_day_chunks(role_start, role_end)
    loaded_chunks = 0

    def trade_loader(start_ms: int, end_ms: int) -> Mapping[str, np.ndarray]:
        nonlocal loaded_chunks
        _progress(
            "round57-trade-chunk-start",
            symbol=symbol,
            role=role_name,
            chunk=loaded_chunks + 1,
            chunks=len(chunks),
            start_ms=start_ms,
        )
        values = load_historical_trade_chunk(
            warehouse.connect(),
            symbol=symbol,
            start_ms=start_ms,
            end_ms_exclusive=end_ms,
        )
        loaded_chunks += 1
        _progress(
            "round57-trade-chunk-complete",
            symbol=symbol,
            role=role_name,
            chunk=loaded_chunks,
            chunks=len(chunks),
            rows=int(values["trade_id"].size),
        )
        return values

    with progress_heartbeat(
        _progress,
        phase="round57-queue-input-heartbeat",
        interval_seconds=heartbeat_seconds,
        details={"symbol": symbol, "role": role_name},
    ):
        queue_inputs = build_chunked_queue_censored_inputs(
            decision_time_ms=decisions,
            fill_requests=requests,
            source_chunks=chunks,
            load_trade_chunk=trade_loader,
            order_notional_quote=float(source["reference_order_notional_quote"]),
        )
    fills = dict(queue_inputs.fills)
    action_features = build_make_take_action_features(
        source_features=dataset.features[indexes],
        source_feature_names=dataset.feature_names,
        decision_time_ms=decisions,
        bid_price=base_bid,
        ask_price=base_ask,
        bid_quantity=base_bid_qty,
        ask_quantity=base_ask_qty,
        flow=queue_inputs.flow,
        source_dataset_sha256=dataset_sha256,
        spec=MakeTakeFeatureSpec(**contract["feature_spec"]),
    )
    base_entries = build_make_take_scenario_entries(
        scenario="base",
        decision_time_ms=decisions,
        bid_price=base_bid,
        ask_price=base_ask,
        bid_quantity=base_bid_qty,
        ask_quantity=base_ask_qty,
        long_fill=fills["base_long"],
        short_fill=fills["base_short"],
    )
    stress_entries = build_make_take_scenario_entries(
        scenario="stress",
        decision_time_ms=decisions,
        bid_price=stress.bid_price,
        ask_price=stress.ask_price,
        bid_quantity=stress.bid_quantity,
        ask_quantity=stress.ask_quantity,
        long_fill=fills["stress_long"],
        short_fill=fills["stress_short"],
    )
    barrier_spec = AdaptiveBarrierSpec(**contract["barrier_spec"])
    stop_bps, take_bps = volatility_scaled_barriers(dataset, indexes, barrier_spec)

    def path_loader(day_start_ms: int) -> Mapping[str, np.ndarray]:
        _progress(
            "round57-path-load",
            symbol=symbol,
            role=role_name,
            day_start_ms=day_start_ms,
        )
        return load_historical_day_path(
            warehouse.connect(),
            symbol=symbol,
            day_start_ms=day_start_ms,
        )

    cache = _DayPathCache(path_loader, capacity=path_cache_days)

    def target_progress(
        scenario: str,
    ) -> Callable[[int, int, int], None]:
        return lambda complete, total, rows: _progress(
            "round57-target-progress",
            symbol=symbol,
            role=role_name,
            scenario=scenario,
            complete=complete,
            total=total,
            realized_rows=rows,
        )

    with progress_heartbeat(
        _progress,
        phase="round57-target-heartbeat",
        interval_seconds=heartbeat_seconds,
        details={"symbol": symbol, "role": role_name, "scenario": "base"},
    ):
        base_target = build_make_take_targets(
            symbol=symbol,
            source_dataset_sha256=dataset_sha256,
            entries=base_entries,
            event_stop_bps=stop_bps,
            event_take_bps=take_bps,
            load_day_path=cache,
            progress=target_progress("base"),
        )
    with progress_heartbeat(
        _progress,
        phase="round57-target-heartbeat",
        interval_seconds=heartbeat_seconds,
        details={"symbol": symbol, "role": role_name, "scenario": "stress"},
    ):
        stress_target = build_make_take_targets(
            symbol=symbol,
            source_dataset_sha256=dataset_sha256,
            entries=stress_entries,
            event_stop_bps=stop_bps,
            event_take_bps=take_bps,
            load_day_path=cache,
            progress=target_progress("stress"),
        )
    fill_panel = build_passive_fill_survival_panel(
        action_features,
        base_entries,
        symbol=symbol,
    )
    payoff_panel = build_make_take_conditional_payoff_panel(
        symbol=symbol,
        action_features=action_features,
        entries=base_entries,
        targets=base_target,
    )
    keep_policy_objects = role_name in POLICY_ROLES
    artifacts = RoleArtifacts(
        action_features=action_features if keep_policy_objects else None,
        fill_panel=fill_panel,
        payoff_panel=payoff_panel,
        base_target=base_target if keep_policy_objects else None,
        stress_target=stress_target if keep_policy_objects else None,
    )
    summary = {
        "symbol": symbol,
        "role": role_name,
        "candidate_rows_before_stress_quote_gate": int(stress_all.rows),
        "stress_quote_valid_rows": int(valid_local.size),
        "stress_quote_invalid_rows": int(stress_all.rows - valid_local.size),
        "stress_quote_source_sha256": stress_all.batch_sha256,
        "selected_stress_quote_sha256": stress.batch_sha256,
        "trade_rows": queue_inputs.source_trade_rows,
        "trade_source_sha256": queue_inputs.source_trade_sha256,
        "trade_chunks": [asdict(value) for value in queue_inputs.source_chunks],
        "fills": {name: value.summary() for name, value in queue_inputs.fills},
        "action_features": action_features.summary(),
        "base_entries": base_entries.summary(),
        "stress_entries": stress_entries.summary(),
        "base_target": base_target.summary(),
        "stress_target": stress_target.summary(),
        "fill_panel": fill_panel.summary(),
        "payoff_panel": payoff_panel.summary(),
    }
    return artifacts, summary


def _model_file_manifest(path: Path, model_sha256: str, seed: int) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "model_sha256": model_sha256,
        "seed": seed,
    }


def _train_models(
    *,
    panels: Mapping[str, Sequence[object]],
    contract: Mapping[str, object],
    compute_backend: str,
    model_root: Path,
    heartbeat_seconds: float,
):
    queue_members = []
    payoff_members = []
    manifests: dict[str, list[dict[str, object]]] = {"queue_fill": [], "payoff": []}
    queue_spec = QueueFillLightGBMSpec(**contract["queue_fill_model_spec"])
    payoff_spec = MakeTakePayoffLightGBMSpec(**contract["payoff_model_spec"])
    for seed in QUEUE_FILL_SEEDS:
        _progress("round57-queue-model-start", seed=seed, backend=compute_backend)
        with progress_heartbeat(
            _progress,
            phase="round57-queue-model-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"seed": seed, "backend": compute_backend},
        ):
            model = train_queue_fill_lightgbm_model(
                training_panels=panels["training_fill"],
                early_stop_panels=panels["early_stop_fill"],
                calibration_panels=panels["probability_calibration_fill"],
                spec=queue_spec,
                compute_backend=compute_backend,
                seed=seed,
                progress=lambda head, total: _progress(
                    "round57-queue-model-head",
                    seed=seed,
                    head=head,
                    total=total,
                ),
            )
        path = model_root / f"queue-fill-seed-{seed}.json"
        save_queue_fill_lightgbm_model(path, model)
        loaded = load_queue_fill_lightgbm_model(path)
        if loaded.model_sha256 != model.model_sha256:
            raise ValueError(f"Round 57 queue model reload drifted for seed {seed}")
        queue_members.append(loaded)
        manifests["queue_fill"].append(
            _model_file_manifest(path, loaded.model_sha256, seed)
        )
        _progress(
            "round57-queue-model-complete",
            seed=seed,
            model_sha256=loaded.model_sha256,
        )
    queue_ensemble = build_queue_fill_lightgbm_ensemble(queue_members)
    for seed in MAKE_TAKE_PAYOFF_SEEDS:
        _progress("round57-payoff-model-start", seed=seed, backend=compute_backend)
        with progress_heartbeat(
            _progress,
            phase="round57-payoff-model-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"seed": seed, "backend": compute_backend},
        ):
            model = train_make_take_payoff_lightgbm_model(
                training_panels=panels["training_payoff"],
                early_stop_panels=panels["early_stop_payoff"],
                calibration_panels=panels["probability_calibration_payoff"],
                spec=payoff_spec,
                compute_backend=compute_backend,
                seed=seed,
                progress=lambda head, total: _progress(
                    "round57-payoff-model-head",
                    seed=seed,
                    head=head,
                    total=total,
                ),
            )
        path = model_root / f"payoff-seed-{seed}.json"
        save_make_take_payoff_lightgbm_model(path, model)
        loaded = load_make_take_payoff_lightgbm_model(path)
        if loaded.model_sha256 != model.model_sha256:
            raise ValueError(f"Round 57 payoff model reload drifted for seed {seed}")
        payoff_members.append(loaded)
        manifests["payoff"].append(
            _model_file_manifest(path, loaded.model_sha256, seed)
        )
        _progress(
            "round57-payoff-model-complete",
            seed=seed,
            model_sha256=loaded.model_sha256,
        )
    payoff_ensemble = build_make_take_payoff_lightgbm_ensemble(
        payoff_members,
        early_stop_panels=panels["early_stop_payoff"],
    )
    return queue_ensemble, payoff_ensemble, manifests


def _role_values(
    *,
    role: str,
    artifacts: Mapping[str, Mapping[str, RoleArtifacts]],
    fill_model,
    payoff_model,
) -> tuple[MakeTakeActionValueBatch, ...]:
    output = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        value = artifacts[symbol][role]
        if value.action_features is None:
            raise ValueError(f"Round 57 {role} action features were not retained")
        fill_prediction = predict_queue_fill_lightgbm_model(
            fill_model,
            value.fill_panel,
        )
        payoff_prediction = predict_make_take_payoff_lightgbm_model(
            payoff_model,
            symbol=symbol,
            action_features=value.action_features,
        )
        output.append(
            build_make_take_action_values(
                symbol=symbol,
                action_features=value.action_features,
                fill_predictions=fill_prediction,
                payoff_predictions=payoff_prediction,
            )
        )
    return tuple(output)


def _role_targets(
    artifacts: Mapping[str, Mapping[str, RoleArtifacts]],
    *,
    role: str,
    scenario: str,
) -> tuple[MakeTakeTargetBatch, ...]:
    values = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        source = artifacts[symbol][role]
        target = source.base_target if scenario == "base" else source.stress_target
        if target is None:
            raise ValueError(f"Round 57 {role} {scenario} target was not retained")
        values.append(target)
    return tuple(values)


def _expected_days(
    contract: Mapping[str, object],
    role: str,
) -> tuple[int, ...]:
    start, end = _role_window(contract, role)
    return tuple(range(start // DAY_MS, end // DAY_MS))


def _predictive_report(
    *,
    role: str,
    artifacts: Mapping[str, Mapping[str, RoleArtifacts]],
    fill_model,
    payoff_model,
) -> MakeTakePredictiveEvaluation:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    return build_make_take_predictive_evaluation(
        role=role,
        fill_model=fill_model,
        payoff_model=payoff_model,
        training_fill_panels=tuple(
            artifacts[symbol]["training"].fill_panel for symbol in symbols
        ),
        evaluation_fill_panels=tuple(
            artifacts[symbol][role].fill_panel for symbol in symbols
        ),
        training_payoff_panels=tuple(
            artifacts[symbol]["training"].payoff_panel for symbol in symbols
        ),
        evaluation_payoff_panels=tuple(
            artifacts[symbol][role].payoff_panel for symbol in symbols
        ),
    )


def _status(
    policy_predictive: MakeTakePredictiveEvaluation,
    evaluation_predictive: MakeTakePredictiveEvaluation,
    policy_selection,
    economic_evaluation,
) -> str:
    if not policy_predictive.predictive_gate_passed:
        return "rejected_policy_predictive"
    if policy_selection is None or not policy_selection.accepted:
        return "rejected_policy_calibration"
    if not evaluation_predictive.predictive_gate_passed:
        return "rejected_evaluation_predictive"
    if economic_evaluation is None or not economic_evaluation.economic_gate_passed:
        return "rejected_economic"
    return "passed_selection_contaminated_development_screen"


def _prepare_evidence_root(path: Path) -> tuple[Path, Path]:
    root = path.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Round 57 evidence root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    models = root / "models"
    models.mkdir()
    _write_run_state(root, status="initialized")
    return root, models


def _write_run_state(root: Path, *, status: str, **details: object) -> None:
    payload: dict[str, object] = {
        "schema_version": "round-057-run-state-v1",
        "round": ROUND,
        "status": status,
        "observed_at_ms": int(time.time() * 1_000),
        **details,
    }
    payload["state_sha256"] = _canonical_sha256(payload)
    write_json_atomic(root / "run-state.json", payload)


def run(arguments: argparse.Namespace) -> int:
    design, contract, design_sha, contract_sha = load_round57_contract(
        arguments.design.resolve(),
        arguments.contract.resolve(),
    )
    binding, binding_sha = validate_round57_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        contract_sha256=contract_sha,
    )
    evidence_root, model_root = _prepare_evidence_root(arguments.evidence_root)
    _write_run_state(
        evidence_root,
        status="running",
        design_sha256=design_sha,
        contract_sha256=contract_sha,
        binding_sha256=binding_sha,
    )
    _progress(
        "round57-start",
        design_sha256=design_sha,
        contract_sha256=contract_sha,
        binding_sha256=binding_sha,
        compute_backend=arguments.compute_backend,
    )
    symbols = tuple(contract["source"]["symbols"])
    artifacts: dict[str, dict[str, RoleArtifacts]] = {}
    source_reports: list[dict[str, object]] = []
    role_reports: list[dict[str, object]] = []
    panels: dict[str, list[object]] = {
        f"{role}_{kind}": []
        for role in ("training", "early_stop", "probability_calibration")
        for kind in ("fill", "payoff")
    }
    with MicrostructureWarehouse(
        arguments.warehouse.resolve(),
        cache_root=arguments.archive_cache.resolve(),
        memory_limit=arguments.memory_limit,
        threads=arguments.threads,
    ) as warehouse:
        for symbol in symbols:
            _progress("round57-source-start", symbol=symbol)
            with progress_heartbeat(
                _progress,
                phase="round57-source-heartbeat",
                interval_seconds=arguments.heartbeat_seconds,
                details={"symbol": symbol},
            ):
                dataset, dataset_sha, source_identity = _verify_bound_dataset(
                    warehouse,
                    symbol=symbol,
                    contract=contract,
                    binding=binding,
                )
            source_reports.append(
                {
                    **source_identity,
                    "dataset_fingerprint": dataset_sha,
                    "rows": dataset.rows,
                    "first_decision_time_ms": int(dataset.decision_time_ms[0]),
                    "last_decision_time_ms": int(dataset.decision_time_ms[-1]),
                    "cache_state": "exact_hit",
                }
            )
            artifacts[symbol] = {}
            for role in ROLE_ORDER:
                _progress(
                    "round57-role-start",
                    symbol=symbol,
                    role=role,
                    dataset_rows=dataset.rows,
                )
                values, summary = _build_role_artifacts(
                    warehouse=warehouse,
                    dataset=dataset,
                    dataset_sha256=dataset_sha,
                    symbol=symbol,
                    role_name=role,
                    contract=contract,
                    path_cache_days=arguments.path_cache_days,
                    heartbeat_seconds=arguments.heartbeat_seconds,
                )
                artifacts[symbol][role] = values
                role_reports.append(summary)
                if role in {
                    "training",
                    "early_stop",
                    "probability_calibration",
                }:
                    panels[f"{role}_fill"].append(values.fill_panel)
                    panels[f"{role}_payoff"].append(values.payoff_panel)
                _progress(
                    "round57-role-complete",
                    symbol=symbol,
                    role=role,
                    fill_panel_rows=values.fill_panel.rows,
                    payoff_panel_rows=values.payoff_panel.rows,
                )
    fill_model, payoff_model, model_manifests = _train_models(
        panels=panels,
        contract=contract,
        compute_backend=arguments.compute_backend,
        model_root=model_root,
        heartbeat_seconds=arguments.heartbeat_seconds,
    )
    policy_predictive = _predictive_report(
        role="policy_calibration",
        artifacts=artifacts,
        fill_model=fill_model,
        payoff_model=payoff_model,
    )
    evaluation_predictive = _predictive_report(
        role="evaluation",
        artifacts=artifacts,
        fill_model=fill_model,
        payoff_model=payoff_model,
    )
    policy_values = _role_values(
        role="policy_calibration",
        artifacts=artifacts,
        fill_model=fill_model,
        payoff_model=payoff_model,
    )
    evaluation_values = _role_values(
        role="evaluation",
        artifacts=artifacts,
        fill_model=fill_model,
        payoff_model=payoff_model,
    )
    policy_selection = None
    economic_evaluation = None
    if policy_predictive.predictive_gate_passed:
        policy_selection = calibrate_make_take_policy(
            predictive_evaluation=policy_predictive,
            action_values=policy_values,
            base_targets=_role_targets(
                artifacts, role="policy_calibration", scenario="base"
            ),
            stress_targets=_role_targets(
                artifacts, role="policy_calibration", scenario="stress"
            ),
            expected_days=_expected_days(contract, "policy_calibration"),
            spec=MakeTakePolicySpec(**contract["policy_spec"]),
        )
    if (
        policy_selection is not None
        and policy_selection.accepted
        and evaluation_predictive.predictive_gate_passed
    ):
        economic_evaluation = evaluate_make_take_policy(
            policy_selection=policy_selection,
            predictive_evaluation=evaluation_predictive,
            action_values=evaluation_values,
            base_targets=_role_targets(artifacts, role="evaluation", scenario="base"),
            stress_targets=_role_targets(
                artifacts, role="evaluation", scenario="stress"
            ),
            expected_days=_expected_days(contract, "evaluation"),
            gate_spec=MakeTakeEconomicGateSpec(**contract["economic_gate_spec"]),
        )
    status = _status(
        policy_predictive,
        evaluation_predictive,
        policy_selection,
        economic_evaluation,
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": status,
        "design_sha256": design_sha,
        "contract_sha256": contract_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": binding["implementation_commit"],
        "source": source_reports,
        "role_evidence": role_reports,
        "models": {
            "compute_backend_requested": arguments.compute_backend,
            "queue_fill_ensemble_sha256": fill_model.model_sha256,
            "queue_fill_member_sha256": [
                member.model_sha256 for member in fill_model.members
            ],
            "payoff_ensemble_sha256": payoff_model.model_sha256,
            "payoff_member_sha256": [
                member.model_sha256 for member in payoff_model.members
            ],
            "payoff_early_quality": asdict(payoff_model.early_quality),
            "artifacts": model_manifests,
        },
        "predictive": {
            "policy_calibration": policy_predictive.evidence(),
            "evaluation": evaluation_predictive.evidence(),
        },
        "action_values": {
            "policy_calibration": [value.summary() for value in policy_values],
            "evaluation": [value.summary() for value in evaluation_values],
        },
        "policy_selection": (
            policy_selection.evidence() if policy_selection is not None else None
        ),
        "economic_evaluation": (
            economic_evaluation.evidence()
            if economic_evaluation is not None
            else None
        ),
        "selection_contaminated": True,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "ai_uplift_claim": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    report_path = evidence_root / "round57-report.json"
    write_json_atomic(report_path, report)
    _write_run_state(
        evidence_root,
        status="complete",
        report_sha256=report["report_sha256"],
        result_status=status,
    )
    _progress(
        "round57-complete",
        status=status,
        report_sha256=report["report_sha256"],
        report_path=str(report_path),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-057-queue-censored-make-take-design.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            research
            / "round-057-queue-censored-make-take-execution-contract.json"
        ),
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=(
            research
            / "round-057-queue-censored-make-take-execution-binding.json"
        ),
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument(
        "--archive-cache",
        type=Path,
        default=ROOT / "data" / "archive-cache",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--path-cache-days", type=int, default=16)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if (
        not 1 <= arguments.threads <= 32
        or not 1 <= arguments.path_cache_days <= 16
        or not 1.0 <= arguments.heartbeat_seconds <= 300.0
    ):
        raise ValueError("Round 57 resource or heartbeat arguments are invalid")
    try:
        return run(arguments)
    except Exception as exc:
        root = arguments.evidence_root.resolve()
        state_path = root / "run-state.json"
        if state_path.is_file():
            failure: dict[str, object] = {
                "schema_version": "round-057-run-failure-v1",
                "round": ROUND,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
            failure["failure_sha256"] = _canonical_sha256(failure)
            write_json_atomic(root / "round57-failure.json", failure)
            _write_run_state(
                root,
                status="failed",
                failure_sha256=failure["failure_sha256"],
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
