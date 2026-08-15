"""Target-free host tuning for Round 74 supervised device groups.

The probe runs in an isolated child process so an accelerator allocation failure
cannot poison the later training runtime. It uses synthetic tensors with the
exact production geometry and loss path; market data, targets, databases, and
model-selection outcomes are never available to it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

from .compute import BackendInfo, resolve_backend, torch_device_for_backend
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    build_round74_event_model,
)
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
from .impact_absorption_event_training import (
    _eligible_target_weighted_group_loss,
    _losses_for_minibatch_group,
)
from .storage import write_bytes_atomic


ROUND74_DEVICE_GROUP_PREFLIGHT_SCHEMA_VERSION = "round-074-device-group-preflight-v1"
ROUND74_DEVICE_GROUP_CANDIDATES = (1, 2, 4, 8, 16, 32)
ROUND74_DEVICE_GROUP_WARMUP_STEPS = 2
ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS = 3
ROUND74_DEVICE_GROUP_SELECTION_FRACTION = 0.975
ROUND74_DEVICE_GROUP_EARLY_STOP_FRACTION = 0.70
ROUND74_DEVICE_GROUP_SEED = 74_135

ProgressCallback = Callable[..., None]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    selected = str(value)
    return len(selected) == 64 and all(
        character in "0123456789abcdef" for character in selected
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _source_binding() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: _canonical_file_sha256(path)
        for path in (
            root / "impact_absorption_event_model.py",
            root / "impact_absorption_event_training.py",
            root / "round74_device_group_preflight.py",
        )
    }


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _synthetic_batch(rows: int) -> Round74EventTrainingBatch:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("Round 74 device preflight row count differs")
    generator = np.random.default_rng(ROUND74_DEVICE_GROUP_SEED)
    features = generator.normal(
        size=(
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        )
    ).astype(np.float32)
    features[:, :, :8] = 0.0
    for row in range(rows):
        features[row, :, row % 5] = 1.0
        features[row, :, 5 + row % 3] = 1.0
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = action_shape[:2]
    directional = generator.normal(size=action_shape[:-1]).astype(np.float32)
    costs = (
        np.abs(generator.normal(size=action_shape[:-1])).astype(np.float32) * 0.05
        + 0.01
    )
    payoff = np.stack(
        (directional - costs, -directional - costs),
        axis=2,
    ).astype(np.float32)
    adverse_excursion = np.maximum.accumulate(
        np.abs(generator.normal(size=action_shape)),
        axis=1,
    ).astype(np.float32)
    decision_monotonic = np.arange(rows, dtype=np.int64) * 1_000
    actual_entry = np.broadcast_to(
        decision_monotonic.reshape(-1, 1, 1) + 10,
        action_shape,
    ).copy()
    actual_exit = actual_entry.copy()
    for horizon_index in range(action_shape[1]):
        actual_exit[:, horizon_index, :] += 10 + horizon_index
    batch = Round74EventTrainingBatch(
        role="training",
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=tuple("1" * 32 for _ in range(rows)),
        symbol=tuple(("BTCUSDT", "ETHUSDT", "SOLUSDT")[row % 3] for row in range(rows)),
        decision_monotonic_ns=_readonly(decision_monotonic),
        decision_wall_ns=_readonly(decision_monotonic + 1_800_000_000_000_000_000),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.zeros(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{row + 1:064x}" for row in range(rows)),
        feature_window_sha256=tuple(f"{rows + row + 1:064x}" for row in range(rows)),
        target_context_sha256=tuple("3" * 64 for _ in range(rows)),
        test_access_sha256=tuple("" for _ in range(rows)),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(
            generator.integers(0, 2, size=action_shape).astype(np.float32)
        ),
        regime_unpredictability=_readonly(
            generator.integers(0, 2, size=regime_shape).astype(np.float32)
        ),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float32)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    batch.validate()
    return batch


def _is_fallback_warning(message: str) -> bool:
    return (
        "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    )


def _selected_group_size(measurements: Sequence[Mapping[str, object]]) -> int:
    if not measurements:
        raise ValueError("Round 74 device preflight candidate has no measurements")
    throughputs = tuple(float(item["rows_per_second"]) for item in measurements)
    if any(not math.isfinite(value) or value <= 0.0 for value in throughputs):
        raise ValueError("Round 74 device preflight throughput differs")
    threshold = max(throughputs) * ROUND74_DEVICE_GROUP_SELECTION_FRACTION
    return min(
        int(item["group_size"])
        for item in measurements
        if float(item["rows_per_second"]) >= threshold
    )


def _benchmark_candidate(
    candidate_id: str,
    batch: Round74EventTrainingBatch,
    *,
    minibatch_rows: int,
    device: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    measurements: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    best_throughput = 0.0
    best_group_size = 0
    for group_size in ROUND74_DEVICE_GROUP_CANDIDATES:
        torch.manual_seed(ROUND74_DEVICE_GROUP_SEED)
        model = build_round74_event_model(candidate_id).to(device).train()
        selections = tuple(
            (
                batch,
                slice(
                    index * minibatch_rows,
                    (index + 1) * minibatch_rows,
                ),
            )
            for index in range(group_size)
        )
        total_action_weight = (
            group_size
            * minibatch_rows
            * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES)
        )
        total_regime_weight = (
            group_size * minibatch_rows * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
        )

        def step(selected_model: torch.nn.Module) -> float:
            selected_model.zero_grad(set_to_none=True)
            grouped = _losses_for_minibatch_group(
                selected_model,
                selections,
                device,
            )
            loss = _eligible_target_weighted_group_loss(
                grouped,
                total_action_weight=total_action_weight,
                total_regime_weight=total_regime_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                selected_model.parameters(),
                max_norm=1.0,
                foreach=False,
            )
            return float(loss.detach().cpu())

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(ROUND74_DEVICE_GROUP_WARMUP_STEPS):
                    step(model)
                timings = []
                loss = math.nan
                for _ in range(ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS):
                    started = time.perf_counter_ns()
                    loss = step(model)
                    timings.append((time.perf_counter_ns() - started) / 1e9)
            fallback_messages = tuple(
                str(item.message)
                for item in caught
                if _is_fallback_warning(str(item.message))
            )
            median_seconds = statistics.median(timings)
            rows = group_size * minibatch_rows
            rows_per_second = rows / median_seconds
            if (
                fallback_messages
                or not math.isfinite(loss)
                or not math.isfinite(median_seconds)
                or median_seconds <= 0.0
                or not math.isfinite(rows_per_second)
                or rows_per_second <= 0.0
            ):
                raise RuntimeError("device benchmark produced invalid evidence")
            measurement = {
                "candidate_id": candidate_id,
                "group_size": group_size,
                "rows": rows,
                "warmup_steps": ROUND74_DEVICE_GROUP_WARMUP_STEPS,
                "measurement_seconds": timings,
                "median_seconds": median_seconds,
                "rows_per_second": rows_per_second,
                "terminal_loss": loss,
                "warning_count": len(caught),
                "cpu_fallback_warning_count": 0,
            }
            measurements.append(measurement)
            if rows_per_second > best_throughput:
                best_throughput = rows_per_second
                best_group_size = group_size
            if (
                group_size >= 2 * best_group_size
                and rows_per_second
                < best_throughput * ROUND74_DEVICE_GROUP_EARLY_STOP_FRACTION
            ):
                break
        except (MemoryError, RuntimeError) as exc:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "group_size": group_size,
                    "error_class": exc.__class__.__name__,
                    "reason": "allocation_or_runtime_failure_in_isolated_probe",
                }
            )
            break
        finally:
            del model
            gc.collect()
    if not measurements:
        raise RuntimeError(
            f"Round 74 device preflight has no viable group for {candidate_id}"
        )
    return measurements, failures


def benchmark_round74_device_groups(
    compute_backend: str,
    *,
    minibatch_rows: int = 128,
) -> dict[str, object]:
    """Benchmark all frozen candidates without reading financial evidence."""

    if (
        isinstance(minibatch_rows, bool)
        or not isinstance(minibatch_rows, int)
        or not 1 <= minibatch_rows <= 4_096
    ):
        raise ValueError("Round 74 device preflight minibatch policy differs")
    backend = resolve_backend(compute_backend, require=True)
    device = torch_device_for_backend(backend)
    batch = _synthetic_batch(max(ROUND74_DEVICE_GROUP_CANDIDATES) * minibatch_rows)
    measurements: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    selected: list[tuple[str, int]] = []
    started = time.perf_counter_ns()
    for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES:
        candidate_measurements, candidate_failures = _benchmark_candidate(
            candidate_id,
            batch,
            minibatch_rows=minibatch_rows,
            device=device,
        )
        measurements.extend(candidate_measurements)
        failures.extend(candidate_failures)
        selected.append((candidate_id, _selected_group_size(candidate_measurements)))
    payload: dict[str, object] = {
        "schema_version": ROUND74_DEVICE_GROUP_PREFLIGHT_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": str(device),
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
            "torch_version": str(torch.__version__),
            "torch_directml_version": _package_version("torch-directml"),
        },
        "source_binding": _source_binding(),
        "protocol": {
            "seed": ROUND74_DEVICE_GROUP_SEED,
            "candidate_ids": list(ROUND74_EVENT_MODEL_CANDIDATES),
            "group_sizes": list(ROUND74_DEVICE_GROUP_CANDIDATES),
            "minibatch_rows": minibatch_rows,
            "warmup_steps": ROUND74_DEVICE_GROUP_WARMUP_STEPS,
            "measurement_steps": ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS,
            "selection_fraction_of_best": (ROUND74_DEVICE_GROUP_SELECTION_FRACTION),
            "early_stop_fraction_of_best": (ROUND74_DEVICE_GROUP_EARLY_STOP_FRACTION),
            "selection_rule": "smallest_group_within_fraction_of_candidate_best",
            "probe_process": "isolated_child",
            "loss_path": "production_supervised_loss_and_backward",
            "pretraining_group_changed": False,
        },
        "measurements": measurements,
        "failures": failures,
        "selected_group_sizes": dict(selected),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
        "evidence_boundary": {
            "market_data_used": False,
            "database_opened": False,
            "realized_financial_target_used": False,
            "model_selection_output_used": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
    }
    payload["preflight_sha256"] = _canonical_sha256(payload)
    return validate_round74_device_group_preflight(payload)


def validate_round74_device_group_preflight(
    payload: Mapping[str, object],
    *,
    expected_backend: BackendInfo | None = None,
) -> dict[str, object]:
    """Validate source binding, measurements, selection, and evidence limits."""

    selected = dict(payload)
    claimed_sha256 = selected.pop("preflight_sha256", None)
    backend = selected.get("backend")
    protocol = selected.get("protocol")
    measurements = selected.get("measurements")
    failures = selected.get("failures")
    selected_groups = selected.get("selected_group_sizes")
    boundary = selected.get("evidence_boundary")
    expected_boundary = {
        "market_data_used": False,
        "database_opened": False,
        "realized_financial_target_used": False,
        "model_selection_output_used": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    try:
        created_utc = datetime.fromisoformat(str(selected.get("created_utc")))
    except ValueError as exc:
        raise ValueError("Round 74 device group preflight time differs") from exc
    elapsed_seconds = selected.get("elapsed_seconds")
    if (
        set(payload)
        != {
            "schema_version",
            "created_utc",
            "backend",
            "source_binding",
            "protocol",
            "measurements",
            "failures",
            "selected_group_sizes",
            "elapsed_seconds",
            "evidence_boundary",
            "preflight_sha256",
        }
        or selected.get("schema_version")
        != ROUND74_DEVICE_GROUP_PREFLIGHT_SCHEMA_VERSION
        or not _is_sha256(claimed_sha256)
        or claimed_sha256 != _canonical_sha256(selected)
        or selected.get("source_binding") != _source_binding()
        or not isinstance(backend, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(measurements, list)
        or not isinstance(failures, list)
        or not isinstance(selected_groups, Mapping)
        or not isinstance(boundary, Mapping)
        or set(selected_groups) != set(ROUND74_EVENT_MODEL_CANDIDATES)
        or dict(boundary) != expected_boundary
        or created_utc.utcoffset() != UTC.utcoffset(created_utc)
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) <= 0.0
    ):
        raise ValueError("Round 74 device group preflight differs")
    expected_protocol = {
        "seed": ROUND74_DEVICE_GROUP_SEED,
        "candidate_ids": list(ROUND74_EVENT_MODEL_CANDIDATES),
        "group_sizes": list(ROUND74_DEVICE_GROUP_CANDIDATES),
        "minibatch_rows": protocol.get("minibatch_rows"),
        "warmup_steps": ROUND74_DEVICE_GROUP_WARMUP_STEPS,
        "measurement_steps": ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS,
        "selection_fraction_of_best": ROUND74_DEVICE_GROUP_SELECTION_FRACTION,
        "early_stop_fraction_of_best": ROUND74_DEVICE_GROUP_EARLY_STOP_FRACTION,
        "selection_rule": "smallest_group_within_fraction_of_candidate_best",
        "probe_process": "isolated_child",
        "loss_path": "production_supervised_loss_and_backward",
        "pretraining_group_changed": False,
    }
    minibatch_rows = protocol.get("minibatch_rows")
    if (
        dict(protocol) != expected_protocol
        or isinstance(minibatch_rows, bool)
        or not isinstance(minibatch_rows, int)
        or not 1 <= minibatch_rows <= 4_096
        or set(backend)
        != {
            "requested",
            "kind",
            "device",
            "vendor",
            "selection",
            "accelerated",
            "torch_version",
            "torch_directml_version",
        }
        or not all(
            isinstance(backend.get(key), str) and bool(str(backend.get(key)))
            for key in (
                "requested",
                "kind",
                "device",
                "vendor",
                "selection",
                "torch_version",
                "torch_directml_version",
            )
        )
        or not isinstance(backend.get("accelerated"), bool)
    ):
        raise ValueError("Round 74 device group preflight policy differs")
    by_candidate: dict[str, list[Mapping[str, object]]] = {
        candidate_id: [] for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES
    }
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError("Round 74 device group measurement differs")
        candidate_id = measurement.get("candidate_id")
        group_size = measurement.get("group_size")
        timings = measurement.get("measurement_seconds")
        numeric = (
            measurement.get("median_seconds"),
            measurement.get("rows_per_second"),
            measurement.get("terminal_loss"),
        )
        if (
            set(measurement)
            != {
                "candidate_id",
                "group_size",
                "rows",
                "warmup_steps",
                "measurement_seconds",
                "median_seconds",
                "rows_per_second",
                "terminal_loss",
                "warning_count",
                "cpu_fallback_warning_count",
            }
            or candidate_id not in by_candidate
            or group_size not in ROUND74_DEVICE_GROUP_CANDIDATES
            or measurement.get("rows") != int(group_size) * minibatch_rows
            or measurement.get("warmup_steps") != ROUND74_DEVICE_GROUP_WARMUP_STEPS
            or not isinstance(timings, list)
            or len(timings) != ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in timings
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric
            )
            or float(numeric[0]) <= 0.0
            or float(numeric[1]) <= 0.0
            or not math.isclose(
                float(measurement["median_seconds"]),
                statistics.median(float(value) for value in timings),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(measurement["rows_per_second"]),
                int(measurement["rows"]) / float(measurement["median_seconds"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or measurement.get("cpu_fallback_warning_count") != 0
            or isinstance(measurement.get("warning_count"), bool)
            or not isinstance(measurement.get("warning_count"), int)
            or int(measurement["warning_count"]) < 0
        ):
            raise ValueError("Round 74 device group measurement differs")
        by_candidate[str(candidate_id)].append(measurement)
    for candidate_id, candidate_measurements in by_candidate.items():
        groups = tuple(int(item["group_size"]) for item in candidate_measurements)
        if (
            not groups
            or groups != tuple(sorted(set(groups)))
            or int(selected_groups[candidate_id])
            != _selected_group_size(candidate_measurements)
        ):
            raise ValueError("Round 74 device group selection differs")
    for failure in failures:
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"candidate_id", "group_size", "error_class", "reason"}
            or failure.get("candidate_id") not in by_candidate
            or failure.get("group_size") not in ROUND74_DEVICE_GROUP_CANDIDATES
            or failure.get("reason")
            != "allocation_or_runtime_failure_in_isolated_probe"
            or not str(failure.get("error_class", ""))
        ):
            raise ValueError("Round 74 device group failure evidence differs")
    if expected_backend is not None and (
        backend.get("kind") != expected_backend.kind
        or backend.get("vendor") != expected_backend.vendor
        or backend.get("accelerated") is not expected_backend.accelerated
    ):
        raise ValueError("Round 74 device group backend drifted")
    result = dict(payload)
    return result


def run_round74_device_group_preflight_subprocess(
    backend: BackendInfo,
    *,
    minibatch_rows: int,
    timeout_seconds: float = 300.0,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the probe out of process with bounded liveness and exact parsing."""

    if (
        not isinstance(backend, BackendInfo)
        or isinstance(timeout_seconds, bool)
        or not 30.0 <= float(timeout_seconds) <= 900.0
        or progress is not None
        and not callable(progress)
    ):
        raise ValueError("Round 74 device preflight subprocess policy differs")
    command = (
        sys.executable,
        "-m",
        "simple_ai_trading.round74_device_group_preflight",
        "--compute-backend",
        backend.kind,
        "--minibatch-rows",
        str(minibatch_rows),
    )
    process = subprocess.Popen(  # noqa: S603 - fixed interpreter and module
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    started = time.monotonic()
    next_progress = 10.0
    while True:
        elapsed = time.monotonic() - started
        remaining = float(timeout_seconds) - elapsed
        if remaining <= 0.0:
            process.kill()
            process.communicate()
            raise TimeoutError("Round 74 device preflight exceeded its hard timeout")
        try:
            stdout, stderr = process.communicate(timeout=min(10.0, remaining))
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            if progress is not None and elapsed >= next_progress:
                progress("device_group_preflight_heartbeat", elapsed_seconds=elapsed)
                next_progress += 10.0
    if process.returncode != 0:
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no detail"
        raise RuntimeError("Round 74 isolated device preflight failed: " + detail[:500])
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Round 74 isolated device preflight output differs") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Round 74 isolated device preflight root differs")
    return validate_round74_device_group_preflight(
        payload,
        expected_backend=backend,
    )


def write_round74_device_group_preflight(
    payload: Mapping[str, object],
    output_directory: Path,
) -> Path:
    selected = validate_round74_device_group_preflight(payload)
    digest = str(selected["preflight_sha256"])
    path = Path(output_directory) / f"round74-device-group-preflight-{digest}.json"
    serialized = _canonical_json_bytes(selected) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == serialized:
            return path
        raise FileExistsError(f"immutable Round 74 artifact already exists: {path}")
    write_bytes_atomic(path, serialized)
    return path


def _parse_arguments(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the target-free Round 74 device grouping preflight",
    )
    parser.add_argument("--compute-backend", required=True)
    parser.add_argument("--minibatch-rows", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    try:
        payload = benchmark_round74_device_groups(
            args.compute_backend,
            minibatch_rows=args.minibatch_rows,
        )
    except (ImportError, MemoryError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Round 74 device preflight failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(_canonical_json_bytes(payload).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROUND74_DEVICE_GROUP_PREFLIGHT_SCHEMA_VERSION",
    "benchmark_round74_device_groups",
    "run_round74_device_group_preflight_subprocess",
    "validate_round74_device_group_preflight",
    "write_round74_device_group_preflight",
]
