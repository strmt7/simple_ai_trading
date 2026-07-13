"""Run the hash-bound Round 49 cost-aware action-hurdle TCN experiment."""

from __future__ import annotations

import argparse
import ctypes
from datetime import UTC, datetime
import hashlib
from importlib import metadata
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

from simple_ai_trading.action_hurdle_analysis import (  # noqa: E402
    STRESS_EXECUTION_CHARGE_BPS,
    candidate_diagnostics,
    economic_gate,
    mechanism_ablation_gate,
    replay_fixed_trades,
    select_fixed_policy_trades,
)
from simple_ai_trading.action_hurdle_tcn_model import (  # noqa: E402
    CANDIDATES,
    HORIZONS_MINUTES,
    RECEPTIVE_FIELD_STEPS,
    SEEDS,
    ActionHurdleForecastBundle,
    build_action_hurdle_temporal_dataset,
    side_net_targets,
    train_action_hurdle_candidates,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 49
DESIGN_SCHEMA = "cost-aware-action-hurdle-tcn-design-v1"
BINDING_SCHEMA = "round-049-cost-aware-action-hurdle-execution-binding-v1"
REPORT_SCHEMA = "cost-aware-action-hurdle-tcn-report-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 49 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 49 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 49 design identity is invalid")
    model = design.get("model_contract")
    data = design.get("data_contract")
    runtime = design.get("runtime_and_governance")
    claims = design.get("claims")
    if not all(isinstance(item, Mapping) for item in (model, data, runtime, claims)):
        raise ValueError("Round 49 design sections are incomplete")
    candidates = model.get("candidates")
    shared_encoder = model.get("shared_encoder")
    if (
        not isinstance(candidates, list)
        or [item.get("id") for item in candidates if isinstance(item, Mapping)]
        != list(CANDIDATES)
        or data.get("symbols") != list(SYMBOLS)
        or data.get("primary_executable_horizon_minutes") != HORIZONS_MINUTES[0]
        or data.get("auxiliary_nonexecuting_horizon_minutes") != HORIZONS_MINUTES[1]
        or model.get("seeds") != list(SEEDS)
        or not isinstance(shared_encoder, Mapping)
        or shared_encoder.get("receptive_field_decision_steps") != RECEPTIVE_FIELD_STEPS
    ):
        raise ValueError("Round 49 implementation and design contracts differ")
    for field in (
        "selection_confirmation_2025_h2_access_permitted",
        "terminal_2026_access_permitted",
        "testnet_or_live_execution_permitted",
        "leverage_permitted",
        "fee_risk_or_gate_relaxation_permitted",
        "post_outcome_parameter_seed_threshold_or_candidate_selection_permitted",
        "manual_graph_or_result_editing_permitted",
    ):
        if runtime.get(field) is not False:
            raise ValueError(f"Round 49 governance must deny {field}")
    if any(value is not False for value in claims.values()):
        raise ValueError("Round 49 frozen claims must all be false")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 49 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 49 binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 49 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 49 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 49 execution requires a clean worktree")
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 49 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 49 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 49 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 49 bound blob changed: {relative_path}")
    return binding, claimed, implementation_commit


def _memory_evidence() -> dict[str, object]:
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            return {
                "working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "pagefile_bytes": int(counters.PagefileUsage),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
            }
    return {
        "working_set_bytes": None,
        "peak_working_set_bytes": None,
        "pagefile_bytes": None,
        "peak_pagefile_bytes": None,
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.sequence = 0

    def __call__(self, event: str, details: Mapping[str, object]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "round-049-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "event": event,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "details": dict(details),
        }
        write_json_atomic(self.path, payload, indent=2, sort_keys=True)
        print(_canonical_json(payload), flush=True)


def _write_failure_status(path: Path, error: Exception, elapsed_seconds: float) -> None:
    sequence = 1
    if path.is_file():
        try:
            sequence = (
                int(_read_object(path, "Round 49 progress").get("sequence", 0)) + 1
            )
        except (TypeError, ValueError):
            sequence = 1
    payload = {
        "schema_version": "round-049-progress-v1",
        "round": ROUND,
        "sequence": sequence,
        "event": "round49_failed",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "memory": _memory_evidence(),
        "details": {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "report_published": False,
        },
    }
    write_json_atomic(path, payload, indent=2, sort_keys=True)
    print(_canonical_json(payload), flush=True)


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "torch-directml"):
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def _predecessor_dataset_sha256(dataset: object) -> str:
    return _canonical_sha256(
        {
            "schema": "round-048-minute-temporal-dataset-v1",
            "symbols": list(SYMBOLS),
            "horizons_minutes": [15, 30, 60, 120],
            "timestamps": int(dataset.timestamps),
            "first_timestamp_ms": int(dataset.timestamps_ms[0]),
            "last_timestamp_ms": int(dataset.timestamps_ms[-1]),
            "feature_names": list(dataset.feature_names),
            "feature_stream_sha256": dataset.feature_stream_sha256,
            "target_stream_sha256": dataset.target_stream_sha256,
            "source_certificate_sha256": dataset.source_evidence[
                "source_certificate_sha256"
            ],
        }
    )


def _target_geometry(dataset: object) -> list[dict[str, object]]:
    targets = side_net_targets(dataset).astype(np.float64)
    rows: list[dict[str, object]] = []
    for role, role_mask in dataset.role_masks.items():
        for symbol_index, symbol in enumerate(SYMBOLS):
            for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
                for side_index, side in enumerate(("short", "long")):
                    values = targets[role_mask, symbol_index, horizon_index, side_index]
                    profitable = values > 0.0
                    if values.size == 0 or not np.any(profitable) or np.all(profitable):
                        raise RuntimeError("Round 49 target geometry is degenerate")
                    prevalence = float(np.mean(profitable))
                    gain_mean = float(np.mean(values[profitable]))
                    loss_mean = float(np.mean(-values[~profitable]))
                    direct_mean = float(np.mean(values))
                    identity_mean = (
                        prevalence * gain_mean - (1.0 - prevalence) * loss_mean
                    )
                    rows.append(
                        {
                            "role": role,
                            "symbol": symbol,
                            "horizon_minutes": horizon,
                            "side": side,
                            "rows": int(values.size),
                            "profit_prevalence": prevalence,
                            "conditional_gain_mean_bps": gain_mean,
                            "conditional_loss_mean_bps": loss_mean,
                            "direct_mean_net_bps": direct_mean,
                            "hurdle_identity_mean_net_bps": identity_mean,
                            "identity_absolute_error_bps": abs(
                                direct_mean - identity_mean
                            ),
                        }
                    )
    if max(float(row["identity_absolute_error_bps"]) for row in rows) > 1e-10:
        raise RuntimeError("Round 49 target hurdle identity failed")
    return rows


def _prediction_artifact_errors(
    bundle: ActionHurdleForecastBundle,
    seed_index: int,
    path: Path,
) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["global_indices"], bundle.global_indices):
            raise RuntimeError("Round 49 prediction artifact indices differ")
        errors = {
            "probability": float(
                np.max(
                    np.abs(
                        archive["probabilities"] - bundle.seed_probabilities[seed_index]
                    )
                )
            ),
            "expected_net": float(
                np.max(
                    np.abs(
                        archive["action_values_bps"]
                        - bundle.seed_action_values_bps[seed_index]
                    )
                )
            ),
            "auxiliary": float(
                np.max(
                    np.abs(
                        archive["auxiliary_mean_bps"]
                        - bundle.seed_auxiliary_mean_bps[seed_index]
                    )
                )
            ),
        }
        artifact_gain = archive["gain_means_bps"]
        artifact_loss = archive["loss_means_bps"]
        if bundle.seed_gain_means_bps is None or bundle.seed_loss_means_bps is None:
            if artifact_gain.size or artifact_loss.size:
                raise RuntimeError("Round 49 direct artifact has severity arrays")
            errors["gain"] = 0.0
            errors["loss"] = 0.0
        else:
            errors["gain"] = float(
                np.max(np.abs(artifact_gain - bundle.seed_gain_means_bps[seed_index]))
            )
            errors["loss"] = float(
                np.max(np.abs(artifact_loss - bundle.seed_loss_means_bps[seed_index]))
            )
    return errors


def _numerical_gate(
    bundle: ActionHurdleForecastBundle,
    diagnostics: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    checks: list[dict[str, object]] = []
    if [artifact.seed for artifact in bundle.artifacts] != list(SEEDS):
        reasons.append("checkpoint_seed_set_differs")
    if int(preflight.get("cpu_fallback_warnings", -1)) != 0:
        reasons.append("preflight_cpu_fallback_warning_count_nonzero")
    prediction_gate = diagnostics.get("numerical_prediction_gate")
    if not isinstance(prediction_gate, Mapping) or not bool(
        prediction_gate.get("passed")
    ):
        reasons.append("numerical_prediction_gate_failed")
    for seed_index, artifact in enumerate(bundle.artifacts):
        model_path = Path(artifact.model_path)
        prediction_path = Path(artifact.prediction_path)
        if not model_path.is_file() or not prediction_path.is_file():
            reasons.append(f"seed_{artifact.seed}_artifact_missing")
            continue
        model_digest = _file_sha256(model_path)
        prediction_digest = _file_sha256(prediction_path)
        errors = _prediction_artifact_errors(bundle, seed_index, prediction_path)
        maximum_reload_error = max(
            artifact.reload_max_abs_logit_error,
            artifact.reload_max_abs_primary_error,
            artifact.reload_max_abs_secondary_error,
            artifact.reload_max_abs_auxiliary_error,
        )
        if model_digest != artifact.model_sha256:
            reasons.append(f"seed_{artifact.seed}_model_hash_mismatch")
        if prediction_digest != artifact.prediction_sha256:
            reasons.append(f"seed_{artifact.seed}_prediction_hash_mismatch")
        if maximum_reload_error > 1e-6:
            reasons.append(f"seed_{artifact.seed}_checkpoint_reload_error")
        if max(errors.values()) > 0.0:
            reasons.append(f"seed_{artifact.seed}_prediction_reload_error")
        if (
            artifact.calibration_binary_log_loss_after
            > artifact.calibration_binary_log_loss_before + 1e-12
        ):
            reasons.append(f"seed_{artifact.seed}_probability_calibration_worsened")
        for label, before, after in (
            (
                "gain",
                artifact.calibration_gain_score_before,
                artifact.calibration_gain_score_after,
            ),
            (
                "loss",
                artifact.calibration_loss_score_before,
                artifact.calibration_loss_score_after,
            ),
        ):
            if before is not None and after is not None and after > before + 1e-12:
                reasons.append(
                    f"seed_{artifact.seed}_{label}_severity_calibration_worsened"
                )
        checks.append(
            {
                "candidate_id": artifact.candidate_id,
                "seed": artifact.seed,
                "model_file_sha256_verified": model_digest == artifact.model_sha256,
                "prediction_file_sha256_verified": (
                    prediction_digest == artifact.prediction_sha256
                ),
                "maximum_checkpoint_reload_error": maximum_reload_error,
                "prediction_array_reload_errors": errors,
                "backend_kind": artifact.backend_kind,
                "backend_device": artifact.backend_device,
            }
        )
    return {"passed": not reasons, "reasons": reasons, "artifact_checks": checks}


def _candidate_result(
    *,
    dataset: object,
    bundle: ActionHurdleForecastBundle,
    diagnostics: Mapping[str, object],
    numerical: Mapping[str, object],
    mechanism: Mapping[str, object] | None,
) -> dict[str, object]:
    trades = select_fixed_policy_trades(dataset, bundle)
    base = replay_fixed_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="base",
        execution_charge_bps=12.0,
    )
    stress = replay_fixed_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="stress",
        execution_charge_bps=STRESS_EXECUTION_CHARGE_BPS,
    )
    action_gate = diagnostics.get("action_quality_gate")
    if not isinstance(action_gate, Mapping):
        raise ValueError("Round 49 action gate is invalid")
    quality_passed = bool(numerical.get("passed")) and bool(action_gate.get("passed"))
    if mechanism is not None:
        quality_passed = quality_passed and bool(mechanism.get("passed"))
    economics = economic_gate(base, stress, quality_passed=quality_passed)
    return {
        "candidate_id": bundle.candidate_id,
        "artifacts": [item.asdict() for item in bundle.artifacts],
        "feature_scaler": bundle.feature_scaler.asdict(),
        "target_scaler": bundle.target_scaler.asdict(),
        "training_history": list(bundle.training_history),
        "diagnostics": dict(diagnostics),
        "numerical_quality_gate": dict(numerical),
        "mechanism_ablation_gate": dict(mechanism) if mechanism is not None else None,
        "combined_quality_gate_passed": quality_passed,
        "base": {
            "metrics": dict(base.metrics),
            "monthly": list(base.monthly),
            "daily_equity": list(base.daily_equity),
            "trades": [item.asdict() for item in base.trades],
            "trade_outcomes": list(base.trade_outcomes),
        },
        "stress": {
            "metrics": dict(stress.metrics),
            "monthly": list(stress.monthly),
            "daily_equity": list(stress.daily_equity),
            "trade_outcomes": list(stress.trade_outcomes),
        },
        "economic_gate": economics,
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    started = time.perf_counter()
    design, design_sha = _validate_design(arguments.design.resolve())
    _, binding_sha, implementation_commit = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        source_certificate_path=arguments.source_certificate.resolve(),
    )
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
        },
    )
    panel, price_source = load_verified_minute_panel(
        arguments.database.resolve(), progress=progress
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=progress,
    )
    hurdle_dataset = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=progress,
    )
    dataset = build_action_hurdle_temporal_dataset(hurdle_dataset)
    del panel, premium, funding, hurdle_dataset
    data_contract = design["data_contract"]
    predecessor = design["predecessor_evidence"]
    if not isinstance(data_contract, Mapping) or not isinstance(predecessor, Mapping):
        raise ValueError("Round 49 data identity contract is missing")
    predecessor_sha = _predecessor_dataset_sha256(dataset)
    if (
        dataset.timestamps != int(data_contract["expected_timestamps"])
        or dataset.rows != int(data_contract["expected_rows"])
        or len(dataset.feature_names) != int(data_contract["causal_predictor_count"])
        or predecessor_sha != predecessor.get("dataset_sha256")
    ):
        raise ValueError("Round 49 dataset differs from the frozen design")
    gap_count = int(
        (dataset.timestamps_ms[-1] - dataset.timestamps_ms[0]) // (5 * 60_000)
        - (dataset.timestamps_ms.size - 1)
    )
    target_geometry = _target_geometry(dataset)
    progress(
        "round49_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "feature_bytes_view": int(dataset.features.nbytes),
            "target_bytes": int(dataset.signed_target_bps.nbytes),
            "dataset_sha256": dataset.dataset_sha256,
            "predecessor_dataset_sha256": predecessor_sha,
            "source_gap_net_count": gap_count,
            "role_timestamp_counts": {
                role: int(mask.sum()) for role, mask in dataset.role_masks.items()
            },
            "target_geometry_rows": len(target_geometry),
            "maximum_target_identity_absolute_error_bps": max(
                float(row["identity_absolute_error_bps"]) for row in target_geometry
            ),
        },
    )
    bundles, preflight = train_action_hurdle_candidates(
        dataset,
        model_dir=evidence_root / "models",
        prediction_dir=evidence_root / "predictions",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    diagnostics: dict[str, Mapping[str, object]] = {}
    numerical: dict[str, Mapping[str, object]] = {}
    for candidate_id, bundle in bundles.items():
        diagnostics[candidate_id] = candidate_diagnostics(dataset, bundle)
        numerical[candidate_id] = _numerical_gate(
            bundle, diagnostics[candidate_id], preflight
        )
        progress(
            "round49_candidate_diagnostics",
            {
                "status": "complete",
                "candidate_id": candidate_id,
                "numerical_gate_passed": numerical[candidate_id]["passed"],
                "action_quality_gate_passed": diagnostics[candidate_id][
                    "action_quality_gate"
                ]["passed"],
            },
        )
    mechanism = mechanism_ablation_gate(
        diagnostics["direct_action_mean_tcn"],
        diagnostics["hurdle_action_value_tcn"],
    )
    candidate_results: list[dict[str, object]] = []
    for candidate_id, bundle in bundles.items():
        result = _candidate_result(
            dataset=dataset,
            bundle=bundle,
            diagnostics=diagnostics[candidate_id],
            numerical=numerical[candidate_id],
            mechanism=(
                mechanism if candidate_id == "hurdle_action_value_tcn" else None
            ),
        )
        candidate_results.append(result)
        progress(
            "round49_candidate_replay",
            {
                "status": "complete",
                "candidate_id": candidate_id,
                "trades": result["base"]["metrics"]["trades"],
                "base_total_net_return_fraction": result["base"]["metrics"][
                    "total_net_return_fraction"
                ],
                "base_maximum_drawdown_fraction": result["base"]["metrics"][
                    "maximum_drawdown_fraction"
                ],
                "economic_gate_passed": result["economic_gate"]["passed"],
            },
        )
    ai_eligible = any(
        bool(result["combined_quality_gate_passed"])
        and bool(result["economic_gate"]["passed"])
        for result in candidate_results
    )
    external_artifacts: list[dict[str, object]] = []
    for bundle in bundles.values():
        for artifact in bundle.artifacts:
            for kind, path_string, digest, size in (
                (
                    "model",
                    artifact.model_path,
                    artifact.model_sha256,
                    artifact.model_bytes,
                ),
                (
                    "predictions",
                    artifact.prediction_path,
                    artifact.prediction_sha256,
                    artifact.prediction_bytes,
                ),
            ):
                external_artifacts.append(
                    {
                        "kind": kind,
                        "candidate_id": artifact.candidate_id,
                        "seed": artifact.seed,
                        "path": path_string,
                        "bytes": size,
                        "sha256": digest,
                    }
                )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": (
            "deterministic_candidate_mechanical_gates_passed_ai_ablation_required"
            if ai_eligible
            else "quality_or_economic_gate_rejected"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "report_canonical_sha256": "PENDING",
        "dataset": {
            "dataset_sha256": dataset.dataset_sha256,
            "predecessor_dataset_sha256": predecessor_sha,
            "feature_stream_sha256": dataset.feature_stream_sha256,
            "target_stream_sha256": dataset.target_stream_sha256,
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "symbols": list(SYMBOLS),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "first_timestamp_ms": int(dataset.timestamps_ms[0]),
            "last_timestamp_ms": int(dataset.timestamps_ms[-1]),
            "role_timestamp_counts": {
                role: int(mask.sum()) for role, mask in dataset.role_masks.items()
            },
            "target_geometry": target_geometry,
            "source_evidence": dict(dataset.source_evidence),
            "persistent_feature_copy_created": False,
        },
        "backend": preflight,
        "candidate_results": candidate_results,
        "mechanism_ablation_gate": mechanism,
        "ai_decision": {
            "paired_veto_only_ablation_eligible": ai_eligible,
            "executed": False,
            "candidate_models": ["qwen3:8b", "fino1:8b"],
            "reason": (
                "A deterministic candidate cleared every frozen gate; a separately hash-bound paired AI veto ablation is required before any AI uplift statement."
                if ai_eligible
                else "No deterministic candidate cleared every numerical, action-quality, mechanism-applicable, and economic gate. AI cannot repair or obscure a failed numerical model."
            ),
            "ai_uplift_claim": False,
            "language_model_order_authority": False,
        },
        "runtime_evidence": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
            "package_versions": _package_versions(),
        },
        "external_artifacts": external_artifacts,
        "selection_confirmation_accessed": False,
        "terminal_2026_accessed": False,
        "leverage_applied": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "promotion_permitted": False,
    }
    canonical = dict(report)
    canonical.pop("report_canonical_sha256")
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    progress(
        "round49_complete",
        {
            "status": report["status"],
            "report_canonical_sha256": report["report_canonical_sha256"],
            "report_file_sha256": _file_sha256(evidence_root / "report.json"),
            "ai_ablation_eligible": ai_eligible,
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=("directml", "cpu"), default="directml"
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    evidence_root = parsed.evidence_root.resolve()
    root_preexisted = evidence_root.exists()
    started = time.perf_counter()
    try:
        run(parsed)
    except Exception as exc:
        if not root_preexisted and evidence_root.is_dir():
            _write_failure_status(
                evidence_root / "status.json",
                exc,
                time.perf_counter() - started,
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
