"""Run a nested chronological confirmation of frozen action-value artifacts."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.compute import resolve_backend  # noqa: E402
from simple_ai_trading.frozen_action_confirmation import (  # noqa: E402
    FROZEN_CONFIRMATION_SCHEMA_VERSION,
    evaluate_frozen_profile_candidates,
    evaluate_frozen_profile_stage,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    causal_cusum_event_mask,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    build_adaptive_barrier_targets,
)
from simple_ai_trading.microstructure_cache import (  # noqa: E402
    load_microstructure_dataset_cache,
    microstructure_dataset_cache_key,
    save_microstructure_dataset_cache,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    build_executable_microstructure_dataset,
    microstructure_feature_source_contract,
)
from simple_ai_trading.microstructure_outcome_lightgbm import (  # noqa: E402
    LIGHTGBM_HURDLE_SCHEMA_VERSION,
    TrainedLightGBMHurdleModel,
    load_lightgbm_hurdle_model,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402

try:  # noqa: E402
    from tools.run_adaptive_action_screen import (
        ACTION_POLICY_SCHEMA_VERSION,
        _forecast_diagnostics,
        _iso_days,
        _targets_sha256,
    )
    from tools.run_gross_architecture_screen import (
        _canonical_sha256,
        _is_sha256,
        _parse_date,
        _sha256_file,
        _utc_day_bounds,
    )
    from tools.run_outcome_mixture_screen import (
        _ensemble_for_role,
        _router_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from run_adaptive_action_screen import (
        ACTION_POLICY_SCHEMA_VERSION,
        _forecast_diagnostics,
        _iso_days,
        _targets_sha256,
    )
    from run_gross_architecture_screen import (
        _canonical_sha256,
        _is_sha256,
        _parse_date,
        _sha256_file,
        _utc_day_bounds,
    )
    from run_outcome_mixture_screen import _ensemble_for_role, _router_diagnostics


DESIGN_SCHEMA_VERSION = "frozen-action-confirmation-design-v1"
REPORT_SCHEMA_VERSION = "frozen-action-confirmation-report-v1"
_ROUND30_DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-030-lightgbm-hurdle-architecture-design-v2.json"
)
_ROUND30_DESIGN_SHA256 = (
    "2adfba2a377666e204b22d926d3d757a6b3c1a20d58c67a3455dabfedbad9623"
)
_SHARED_SECTIONS = (
    "execution",
    "barrier_targets",
    "event_sampler",
    "threshold_policy",
    "risk_profiles",
)
_STAGE_NAMES = ("confirmation", "policy", "development")
_DAY_MS = 86_400_000


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("frozen confirmation Git binding failed") from exc


def _validate_implementation_binding(binding: Mapping[str, object]) -> None:
    commit = str(binding.get("commit") or "").lower()
    files = binding.get("files")
    if (
        binding.get("hash_mode") != "git_blob_sha256_v1"
        or len(commit) != 40
        or any(value not in "0123456789abcdef" for value in commit)
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("frozen confirmation implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("frozen confirmation implementation file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in seen
            or not (ROOT / relative).is_file()
        ):
            raise ValueError("frozen confirmation implementation path is unsafe")
        seen.add(normalized)
        expected = str(item["sha256"])
        if (
            hashlib.sha256(_git_bytes("show", f"{commit}:{normalized}")).hexdigest()
            != expected
            or hashlib.sha256(_git_bytes("show", f"HEAD:{normalized}")).hexdigest()
            != expected
        ):
            raise ValueError(
                f"frozen confirmation implementation changed: {normalized}"
            )
        try:
            subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--quiet", "--", normalized],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    normalized,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"frozen confirmation implementation worktree changed: {normalized}"
            ) from exc


def _read_json(path: str | Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _validate_stage_contracts(data: Mapping[str, object]) -> None:
    stages = data.get("stages")
    if not isinstance(stages, Mapping) or tuple(stages) != _STAGE_NAMES:
        raise ValueError("frozen confirmation stage set is invalid")
    previous_end = None
    previous_next = None
    for name in _STAGE_NAMES:
        stage = stages[name]
        if not isinstance(stage, Mapping) or set(stage) != {
            "context_start",
            "evaluation_start",
            "evaluation_end",
            "next_unopened_date",
        }:
            raise ValueError(f"frozen confirmation {name} stage is invalid")
        context = _parse_date(stage["context_start"], label=f"{name} context")
        first = _parse_date(stage["evaluation_start"], label=f"{name} start")
        last = _parse_date(stage["evaluation_end"], label=f"{name} end")
        next_day = _parse_date(
            stage["next_unopened_date"], label=f"{name} next unopened"
        )
        if context > first or first > last or last >= next_day:
            raise ValueError(f"frozen confirmation {name} chronology is invalid")
        if previous_end is not None and context < previous_end:
            raise ValueError("frozen confirmation stage contexts overlap future data")
        if previous_next is not None and first < previous_next:
            raise ValueError("frozen confirmation evaluation stages overlap")
        previous_end = last
        previous_next = next_day
    first_context = stages["confirmation"]["context_start"]
    last_evaluation = stages["development"]["evaluation_end"]
    if (
        str(data.get("start_date")) != str(first_context)
        or str(data.get("end_date")) != str(last_evaluation)
    ):
        raise ValueError("frozen confirmation data bounds differ from stages")


def load_frozen_confirmation_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    payload = _read_json(path, label="frozen confirmation design")
    claimed = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    calculated = _canonical_sha256(canonical)
    if not _is_sha256(claimed) or str(claimed) != calculated:
        raise ValueError("frozen confirmation design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or int(payload.get("round") or 0) != 31
        or int(payload.get("design_revision") or 0) != 1
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("portfolio_claim") is not False
        or payload.get("leverage_applied") is not False
    ):
        raise ValueError("frozen confirmation design identity is invalid")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("frozen confirmation implementation is missing")
    if require_current:
        _validate_implementation_binding(implementation)
    sealed = _read_json(_ROUND30_DESIGN, label="sealed Round 30 design")
    if sealed.get("design_sha256") != _ROUND30_DESIGN_SHA256:
        raise ValueError("sealed Round 30 design identity drifted")
    for section in _SHARED_SECTIONS:
        if payload.get(section) != sealed.get(section):
            raise ValueError(
                f"frozen confirmation {section} differs from sealed Round 30"
            )
    data = payload.get("data")
    terminal = payload.get("reserved_terminal")
    if (
        not isinstance(data, Mapping)
        or data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or data.get("required_data_types") != ["bookTicker", "trades"]
        or data.get("full_history_inventory_required") is not True
        or not isinstance(terminal, Mapping)
    ):
        raise ValueError("frozen confirmation data contract is invalid")
    _validate_stage_contracts(data)
    development = data["stages"]["development"]
    terminal_date = _parse_date(terminal.get("date"), label="terminal")
    if (
        terminal_date
        != _parse_date(development["evaluation_end"], label="development end")
        + timedelta(days=1)
        or terminal.get("included_in_dataset") is not False
        or terminal.get("access_permitted") is not False
    ):
        raise ValueError("frozen confirmation terminal contract is invalid")
    source_model = payload.get("source_model")
    availability = payload.get("availability")
    if (
        not isinstance(source_model, Mapping)
        or source_model.get("round") != 30
        or source_model.get("model_schema_version")
        != LIGHTGBM_HURDLE_SCHEMA_VERSION
        or not _is_sha256(source_model.get("report_file_sha256"))
        or not _is_sha256(source_model.get("report_canonical_sha256"))
        or not _is_sha256(source_model.get("design_sha256"))
        or not _is_sha256(source_model.get("barrier_targets_sha256"))
        or not _is_sha256(source_model.get("source_manifest_fingerprint"))
        or not _is_sha256(source_model.get("corpus_certificate_sha256"))
        or not _is_sha256(source_model.get("target_contract_sha256"))
        or not isinstance(source_model.get("models"), list)
        or len(source_model["models"]) != 3
        or not isinstance(availability, Mapping)
        or not _is_sha256(availability.get("plan_file_sha256"))
        or availability.get("truth_basis")
        != "official_binance_data_vision_s3_listing"
    ):
        raise ValueError("frozen confirmation predecessor evidence is invalid")
    for item in source_model["models"]:
        if (
            not isinstance(item, Mapping)
            or int(item.get("seed") or 0) <= 0
            or not _is_sha256(item.get("artifact_sha256"))
            or not _is_sha256(item.get("model_sha256"))
            or not str(item.get("path") or "").startswith("models/")
        ):
            raise ValueError("frozen confirmation source model entry is invalid")
    thresholds = payload.get("frozen_thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or tuple(thresholds) != ("conservative", "regular", "aggressive")
        or any(
            not isinstance(values, list) or len(values) != 4
            for values in thresholds.values()
        )
    ):
        raise ValueError("frozen confirmation thresholds are invalid")
    resources = payload.get("runtime_resources")
    if (
        not isinstance(resources, Mapping)
        or resources.get("compute_backend") != "directml"
        or resources.get("cpu_fallback_permitted") is not False
        or int(resources.get("warehouse_threads") or 0) <= 0
        or not str(resources.get("duckdb_memory_limit") or "").upper().endswith(
            "GB"
        )
    ):
        raise ValueError("frozen confirmation runtime contract is invalid")
    research = payload.get("research_basis")
    if not isinstance(research, list) or len(research) < 3:
        raise ValueError("frozen confirmation research basis is incomplete")
    return payload, calculated


def _validate_availability_plan(
    path: str | Path,
    availability: Mapping[str, object],
) -> dict[str, object]:
    source = Path(path)
    if _sha256_file(source) != str(availability["plan_file_sha256"]):
        raise ValueError("frozen confirmation availability plan hash differs")
    plan = _read_json(source, label="frozen confirmation availability plan")
    coverage = plan.get("coverage")
    snapshots = plan.get("inventory_snapshots")
    if (
        plan.get("status") != "ok"
        or plan.get("plan_only") is not True
        or plan.get("full_history") is not True
        or plan.get("inventory_identity_verified") is not True
        or plan.get("truth_basis") != availability["truth_basis"]
        or plan.get("missing") != []
        or plan.get("inventory_errors") != []
        or not isinstance(coverage, list)
        or not isinstance(snapshots, list)
    ):
        raise ValueError("frozen confirmation availability plan is invalid")
    expected_coverage = availability.get("coverage")
    if not isinstance(expected_coverage, list) or coverage != expected_coverage:
        raise ValueError("frozen confirmation availability coverage drifted")
    identities = availability.get("inventory_identities")
    if not isinstance(identities, list):
        raise ValueError("frozen confirmation inventory identities are missing")
    observed = [
        {
            "symbol": item.get("symbol"),
            "data_type": item.get("data_type"),
            "snapshot_id": item.get("snapshot_id"),
            "listing_sha256": item.get("listing_sha256"),
            "verification_phase": item.get("verification_phase"),
        }
        for item in snapshots
    ]
    if observed != identities:
        raise ValueError("frozen confirmation inventory identity drifted")
    return plan


def _safe_source_path(root: Path, relative_value: object) -> Path:
    relative = Path(str(relative_value or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("frozen confirmation source artifact path is unsafe")
    path = root / relative
    if not path.is_file():
        raise ValueError("frozen confirmation source artifact is missing")
    return path


def _load_source_models(
    root: str | Path,
    contract: Mapping[str, object],
) -> tuple[
    list[TrainedLightGBMHurdleModel],
    dict[str, object],
    dict[str, object],
]:
    source_root = Path(root)
    report_path = source_root / "report.json"
    if _sha256_file(report_path) != str(contract["report_file_sha256"]):
        raise ValueError("frozen confirmation source report file hash differs")
    report = _read_json(report_path, label="frozen confirmation source report")
    source_dataset = report.get("dataset")
    canonical = dict(report)
    claimed = canonical.pop("report_sha256", None)
    if (
        claimed != contract["report_canonical_sha256"]
        or _canonical_sha256(canonical) != claimed
        or report.get("round") != 30
        or report.get("design_sha256") != contract["design_sha256"]
        or report.get("trading_authority") is not False
        or report.get("profitability_claim") is not False
        or report.get("leverage_applied") is not False
        or not isinstance(source_dataset, Mapping)
        or report.get("corpus_certificate_sha256")
        != contract["corpus_certificate_sha256"]
        or source_dataset.get("barrier_targets_sha256")
        != contract["barrier_targets_sha256"]
        or source_dataset.get("source_manifest_fingerprint")
        != contract["source_manifest_fingerprint"]
    ):
        raise ValueError("frozen confirmation source report identity differs")
    report_models = report.get("ensemble_models")
    if not isinstance(report_models, list) or len(report_models) != 3:
        raise ValueError("frozen confirmation source ensemble is invalid")
    models: list[TrainedLightGBMHurdleModel] = []
    evidence: list[dict[str, object]] = []
    by_seed = {int(item["seed"]): item for item in report_models}
    for expected in contract["models"]:
        seed = int(expected["seed"])
        reported = by_seed.get(seed)
        if not isinstance(reported, Mapping):
            raise ValueError("frozen confirmation source seed is missing")
        path = _safe_source_path(source_root, expected["path"])
        reported_artifact = reported.get("artifact")
        reported_model = reported.get("model")
        if (
            not isinstance(reported_artifact, Mapping)
            or not isinstance(reported_model, Mapping)
            or _sha256_file(path) != expected["artifact_sha256"]
            or reported_artifact.get("sha256") != expected["artifact_sha256"]
        ):
            raise ValueError("frozen confirmation model artifact hash differs")
        model = load_lightgbm_hurdle_model(path)
        if (
            model.model_sha256 != expected["model_sha256"]
            or reported_model.get("model_sha256") != expected["model_sha256"]
            or model.backend_kind != "opencl"
            or len(model.model_strings) != 10
            or model.target_contract_sha256 != contract["target_contract_sha256"]
            or model.trading_authority
            or model.profitability_claim
            or model.leverage_applied
        ):
            raise ValueError("frozen confirmation model identity differs")
        models.append(model)
        evidence.append(
            {
                "seed": seed,
                "path": str(expected["path"]),
                "artifact_sha256": str(expected["artifact_sha256"]),
                "model_sha256": model.model_sha256,
                "backend_kind": model.backend_kind,
                "booster_count": len(model.model_strings),
                "reload_verified": True,
            }
        )
    return (
        models,
        {
            "report": {
                "file_sha256": str(contract["report_file_sha256"]),
                "canonical_sha256": str(contract["report_canonical_sha256"]),
                "design_sha256": str(contract["design_sha256"]),
                "barrier_targets_sha256": str(contract["barrier_targets_sha256"]),
                "source_manifest_fingerprint": str(
                    contract["source_manifest_fingerprint"]
                ),
                "corpus_certificate_sha256": str(
                    contract["corpus_certificate_sha256"]
                ),
                "target_contract_sha256": str(
                    contract["target_contract_sha256"]
                ),
            },
            "models": evidence,
        },
        report,
    )


def _validate_frozen_thresholds_against_source(
    design: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    profiles = report.get("profile_results")
    if not isinstance(profiles, list) or len(profiles) != 3:
        raise ValueError("frozen confirmation source profile evidence is invalid")
    observed: dict[str, list[dict[str, float]]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("frozen confirmation source profile is invalid")
        selection = profile.get("threshold_selection")
        if not isinstance(selection, Mapping) or not isinstance(
            selection.get("candidates"), list
        ):
            raise ValueError("frozen confirmation source thresholds are missing")
        observed[str(profile["profile"])] = [
            {
                "quantile": float(candidate["quantile"]),
                "threshold_bps": float(candidate["threshold_bps"]),
            }
            for candidate in selection["candidates"]
        ]
    if observed != design["frozen_thresholds"]:
        raise ValueError("frozen confirmation thresholds differ from Round 30")


def _ensure_causal_feature_bars(
    *,
    symbol: str,
    warehouse_path: str | Path,
    cache_root: str | Path,
    memory_limit: str,
    threads: int,
    progress: Callable[..., None],
) -> dict[str, object]:
    """Materialize the fixed causal transform without opening any target stage."""

    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        try:
            evidence = warehouse.require_causal_feature_bars(symbol)
            progress("causal-feature-bars", state="reused")
            return {**evidence, "materialization_state": "reused"}
        except ValueError:
            progress("causal-feature-bars", state="rebuild")
            evidence = warehouse.rebuild_causal_feature_bars(
                symbol,
                progress=lambda phase, completed, total: progress(
                    "causal-feature-build",
                    build_phase=phase,
                    completed=completed,
                    total=total,
                ),
            )
            return {**evidence, "materialization_state": "rebuilt"}


def _load_stage_dataset(
    *,
    design: Mapping[str, object],
    stage_name: str,
    warehouse_path: str | Path,
    cache_root: str | Path,
    memory_limit: str,
    threads: int,
    progress: Callable[..., None],
) -> dict[str, object]:
    data = design["data"]
    execution = design["execution"]
    sampler = design["event_sampler"]
    stage = data["stages"][stage_name]
    context = _parse_date(stage["context_start"], label=f"{stage_name} context")
    first = _parse_date(stage["evaluation_start"], label=f"{stage_name} start")
    last = _parse_date(stage["evaluation_end"], label=f"{stage_name} end")
    next_day = _parse_date(
        stage["next_unopened_date"], label=f"{stage_name} next unopened"
    )
    requested_start_ms, requested_end_ms = _utc_day_bounds(context, last)
    evaluation_start_ms, evaluation_end_ms = _utc_day_bounds(first, last)
    feature_version = str(design["feature_version"])
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        progress("source-verify", stage=stage_name)
        source_evidence = dict(
            warehouse.require_causal_feature_bars(str(data["symbol"]))
        )
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=requested_start_ms,
            required_end_ms=requested_end_ms,
            require_full_history_inventory=True,
        )
        source_evidence["corpus_certificate"] = certificate
        source_contract = microstructure_feature_source_contract(feature_version)
        if source_contract is not None:
            source_evidence["feature_source_contract"] = source_contract
        cache_parameters = {
            "symbol": str(data["symbol"]),
            "requested_start_ms": requested_start_ms,
            "requested_end_ms": requested_end_ms,
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
            "decision_cadence_seconds": int(
                execution["decision_cadence_seconds"]
            ),
            "require_full_history_inventory": True,
            "source_evidence": source_evidence,
            "feature_version": feature_version,
        }
        cache_key = microstructure_dataset_cache_key(**cache_parameters)
        progress("cache-lookup", stage=stage_name, cache_key=cache_key)
        dataset = load_microstructure_dataset_cache(warehouse, **cache_parameters)
        cache_state = "hit"
        if dataset is None:
            cache_state = "build"
            progress("dataset-build", stage=stage_name)
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
                max_l1_participation=float(
                    execution["max_l1_participation"]
                ),
                decision_cadence_seconds=int(
                    execution["decision_cadence_seconds"]
                ),
                start_ms=requested_start_ms,
                end_ms=requested_end_ms,
                require_full_history_inventory=True,
                feature_version=feature_version,
            )
            cache_key = save_microstructure_dataset_cache(
                warehouse,
                dataset,
                requested_start_ms=requested_start_ms,
                requested_end_ms=requested_end_ms,
                require_full_history_inventory=True,
            )
            cache_state = "written"
        event_mask = causal_cusum_event_mask(
            dataset,
            volatility_multiplier=float(sampler["volatility_multiplier"]),
            minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
        )
        evaluation_mask = (
            event_mask
            & (dataset.decision_time_ms >= evaluation_start_ms)
            & (dataset.decision_time_ms < evaluation_end_ms)
        )
        event_indexes = np.flatnonzero(evaluation_mask).astype(np.int64)
        if len(event_indexes) < 256:
            raise ValueError(f"frozen confirmation {stage_name} event support is low")
        progress(
            "barrier-target-build-start",
            stage=stage_name,
            event_rows=len(event_indexes),
        )
        targets = build_adaptive_barrier_targets(
            warehouse,
            dataset,
            event_indexes,
            spec=AdaptiveBarrierSpec(**dict(design["barrier_targets"])),
            progress=lambda day, total, valid: progress(
                "barrier-target-day",
                stage=stage_name,
                day=day,
                days=total,
                valid_rows=valid,
            ),
        )
    positions = np.flatnonzero(targets.valid)
    endpoints = targets.source_indexes[positions]
    max_exit = np.maximum.reduce(
        (
            targets.base_long_exit_time_ms[positions],
            targets.base_short_exit_time_ms[positions],
            targets.stress_long_exit_time_ms[positions],
            targets.stress_short_exit_time_ms[positions],
        )
    )
    next_start_ms = _utc_day_bounds(next_day, next_day)[0]
    if (
        len(endpoints) < 256
        or np.any(dataset.decision_time_ms[endpoints] < evaluation_start_ms)
        or np.any(dataset.decision_time_ms[endpoints] >= evaluation_end_ms)
        or np.any(max_exit >= next_start_ms)
    ):
        raise ValueError(f"frozen confirmation {stage_name} role is not purged")
    return {
        "dataset": dataset,
        "targets": targets,
        "endpoints": endpoints,
        "expected_days": _iso_days(
            {"start": first.isoformat(), "end": last.isoformat()}
        ),
        "evidence": {
            "stage": stage_name,
            "context_start": context.isoformat(),
            "evaluation_start": first.isoformat(),
            "evaluation_end": last.isoformat(),
            "next_unopened_date": next_day.isoformat(),
            "dataset_rows": dataset.rows,
            "event_rows": len(event_indexes),
            "valid_target_rows": targets.valid_rows,
            "first_decision_time_ms": int(dataset.decision_time_ms[endpoints[0]]),
            "last_decision_time_ms": int(dataset.decision_time_ms[endpoints[-1]]),
            "last_exit_time_ms": int(np.max(max_exit)),
            "purged": True,
            "cache_key": cache_key,
            "cache_state": cache_state,
            "source_manifest_fingerprint": source_evidence["manifest_fingerprint"],
            "corpus_certificate_sha256": certificate["certificate_sha256"],
            "barrier_targets_sha256": _targets_sha256(targets),
        },
    }


def _profile_by_name(design: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(item["profile"]): item for item in design["risk_profiles"]}


def _stage_report(
    *,
    design: Mapping[str, object],
    stage_name: str,
    stage_data: Mapping[str, object],
    models: list[TrainedLightGBMHurdleModel],
    active: Mapping[str, tuple[float, float]] | None,
    compute_backend: str,
    progress: Callable[..., None],
) -> tuple[dict[str, object], dict[str, tuple[float, float]]]:
    dataset = stage_data["dataset"]
    targets = stage_data["targets"]
    endpoints = stage_data["endpoints"]
    progress("predict", stage=stage_name, rows=len(endpoints))
    prediction = _ensemble_for_role(
        models,
        dataset,
        endpoints,
        compute_backend=compute_backend,
        batch_size=int(design["runtime_resources"]["prediction_batch_size"]),
    )
    profiles = _profile_by_name(design)
    results: list[dict[str, object]] = []
    survivors: dict[str, tuple[float, float]] = {}
    for name in ("conservative", "regular", "aggressive"):
        if active is not None and name not in active:
            continue
        profile = profiles[name]
        gate_name = {
            "confirmation": "calibration_gates",
            "policy": "policy_gates",
            "development": "development_gates",
        }[stage_name]
        if stage_name == "confirmation":
            result = evaluate_frozen_profile_candidates(
                dataset,
                targets,
                prediction,
                profile=profile,
                candidates=design["frozen_thresholds"][name],
                gates=profile[gate_name],
                expected_days=stage_data["expected_days"],
                drawdown_penalty=float(
                    design["threshold_policy"]["drawdown_penalty"]
                ),
                stage=stage_name,
            )
        else:
            assert active is not None
            quantile, threshold = active[name]
            result = evaluate_frozen_profile_stage(
                dataset,
                targets,
                prediction,
                profile=profile,
                quantile=quantile,
                threshold_bps=threshold,
                gates=profile[gate_name],
                expected_days=stage_data["expected_days"],
                drawdown_penalty=float(
                    design["threshold_policy"]["drawdown_penalty"]
                ),
                stage=stage_name,
            )
        results.append(result)
        if result["passed"]:
            survivors[name] = (
                float(result["selected_quantile"]),
                float(result["selected_threshold_bps"]),
            )
        progress(
            "profile-complete",
            stage=stage_name,
            profile=name,
            passed=bool(result["passed"]),
            eligible_rows=int(result["eligible_rows"]),
        )
    return (
        {
            "stage": stage_name,
            "data": stage_data["evidence"],
            "forecast_diagnostics": {
                "base": _forecast_diagnostics(targets, prediction, scenario="base"),
                "stress": _forecast_diagnostics(
                    targets, prediction, scenario="stress"
                ),
            },
            "representation_routing_diagnostics": _router_diagnostics(prediction),
            "profile_results": results,
            "surviving_profiles": list(survivors),
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
        survivors,
    )


def run_frozen_action_confirmation(
    *,
    design_path: str | Path,
    availability_plan_path: str | Path,
    source_model_root: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_frozen_confirmation_design(design_path)
    resources = design["runtime_resources"]
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
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("frozen confirmation output directory is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    runtime = {
        "duckdb_memory_limit": effective_memory,
        "warehouse_threads": effective_threads,
        "compute_backend_requested": effective_backend,
        "compute_backend_kind": backend.kind,
        "compute_backend_device": backend.device,
        "compute_backend_vendor": backend.vendor,
        "model_training_performed": False,
        "model_prediction_backend": "lightgbm_cpu_predict",
        "cpu_fallback_permitted": False,
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
            "frozen-action-confirmation "
            + " ".join(
                f"{name}={value}"
                for name, value in payload.items()
                if name != "runtime_resources"
            ),
            flush=True,
        )
        write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    progress("initialize")
    availability = _validate_availability_plan(
        availability_plan_path, design["availability"]
    )
    models, source_evidence, source_report = _load_source_models(
        source_model_root, design["source_model"]
    )
    _validate_frozen_thresholds_against_source(design, source_report)
    causal_feature_evidence = _ensure_causal_feature_bars(
        symbol=str(design["data"]["symbol"]),
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
        progress=progress,
    )
    stage_reports: dict[str, object] = {}
    stage_artifacts: dict[str, object] = {}
    access = {name: False for name in _STAGE_NAMES}
    active: dict[str, tuple[float, float]] | None = None
    for stage_name in _STAGE_NAMES:
        if stage_name != "confirmation" and not active:
            progress("stage-withheld", stage=stage_name, reason="prior_stage_rejected")
            break
        access[stage_name] = True
        progress("stage-open", stage=stage_name)
        stage_data = _load_stage_dataset(
            design=design,
            stage_name=stage_name,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            memory_limit=effective_memory,
            threads=effective_threads,
            progress=progress,
        )
        stage_report, active = _stage_report(
            design=design,
            stage_name=stage_name,
            stage_data=stage_data,
            models=models,
            active=active,
            compute_backend=effective_backend,
            progress=progress,
        )
        stage_reports[stage_name] = stage_report
        stage_path = destination / f"stage-{stage_name}.json"
        write_json_atomic(
            stage_path,
            stage_report,
            indent=2,
            sort_keys=True,
        )
        stage_artifacts[stage_name] = {
            "path": stage_path.name,
            "sha256": _sha256_file(stage_path),
            "bytes": stage_path.stat().st_size,
        }
    final_profiles = list(active) if access["development"] and active else []
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "frozen_round30_chronological_confirmation_evidence",
        "status": "research_candidate" if final_profiles else "rejected",
        "round": 31,
        "design_sha256": design_sha256,
        "model_schema_version": LIGHTGBM_HURDLE_SCHEMA_VERSION,
        "confirmation_schema_version": FROZEN_CONFIRMATION_SCHEMA_VERSION,
        "action_policy_schema_version": ACTION_POLICY_SCHEMA_VERSION,
        "barrier_schema_version": ADAPTIVE_BARRIER_SCHEMA_VERSION,
        "target_mode": ADAPTIVE_BARRIER_TARGET_MODE,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "policy_window_is_consumed": access["policy"],
        "development_window_is_consumed": access["development"],
        "stage_access": access,
        "runtime_resources": runtime,
        "availability_plan": {
            "file_sha256": _sha256_file(Path(availability_plan_path)),
            "truth_basis": availability["truth_basis"],
            "coverage": availability["coverage"],
            "inventory_identity_verified": True,
        },
        "source_model_evidence": source_evidence,
        "causal_feature_evidence": causal_feature_evidence,
        "stages": stage_reports,
        "stage_artifacts": stage_artifacts,
        "final_profiles": final_profiles,
        "limitations": [
            "the exact public BTCUSDT book-ticker archive spans 320 days rather than multiple years",
            "the frozen predictor was trained on an earlier 52-day research window",
            "the 100 ms BBO path cannot resolve queue position or hidden depth",
            "LightGBM prediction is CPU-bound even though the source models were trained on OpenCL",
            "this single-symbol confirmation cannot satisfy portfolio-diversification requirements",
            "all returns are unleveraged and no stage grants trading authority",
            "the reserved terminal date was neither queried nor labeled",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    progress("complete", status=report["status"], report_sha256=report["report_sha256"])
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a precommitted frozen action-value confirmation"
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--availability-plan", required=True)
    parser.add_argument("--source-model-root", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--compute-backend")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_frozen_action_confirmation(
        design_path=args.design,
        availability_plan_path=args.availability_plan,
        source_model_root=args.source_model_root,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(
        f"frozen-action-confirmation: status={report['status']} "
        f"sha256={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
