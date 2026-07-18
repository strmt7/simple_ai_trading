"""Capability-tested LightGBM CPU, OpenCL, and CUDA resolution."""

from __future__ import annotations

from functools import lru_cache
import os

from .compute import (
    BackendInfo,
    BackendUnavailableError,
    SUPPORTED_COMPUTE_BACKENDS,
    require_backend,
    resolve_backend,
)


OPENCL_PLATFORM_ENV = "SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID"
OPENCL_DEVICE_ENV = "SIMPLE_AI_TRADING_OPENCL_DEVICE_ID"


def _opencl_device_override() -> tuple[int | None, int | None, str]:
    platform_raw = (os.getenv(OPENCL_PLATFORM_ENV) or "").strip()
    device_raw = (os.getenv(OPENCL_DEVICE_ENV) or "").strip()
    if bool(platform_raw) != bool(device_raw):
        raise ValueError(
            f"{OPENCL_PLATFORM_ENV} and {OPENCL_DEVICE_ENV} must be set together"
        )
    if not platform_raw:
        return None, None, "opencl:auto"
    try:
        platform_id = int(platform_raw)
        device_id = int(device_raw)
    except ValueError as exc:
        raise ValueError("invalid OpenCL platform or device id") from exc
    if platform_id < 0 or device_id < 0:
        raise ValueError("OpenCL platform and device ids must be non-negative")
    return platform_id, device_id, f"opencl:{platform_id}:{device_id}"


@lru_cache(maxsize=16)
def _probe_lightgbm_target(
    target: str,
    platform_id: int | None,
    device_id: int | None,
) -> tuple[bool, str]:
    """Run one real tree update; package and driver presence are insufficient."""

    try:
        import lightgbm as lgb
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional runtime
        return False, f"LightGBM probe imports failed ({exc.__class__.__name__})"

    features = (
        np.arange(512 * 8, dtype=np.float32).reshape(512, 8) % np.float32(97.0)
    )
    labels = (np.arange(512, dtype=np.int32) % 2).astype(np.float32)
    parameters: dict[str, object] = {
        "objective": "binary",
        "device_type": target,
        "verbosity": -1,
        "num_threads": 1,
        "seed": 17,
        "max_bin": 63,
    }
    if target == "gpu" and platform_id is not None and device_id is not None:
        parameters.update(
            {
                "gpu_platform_id": platform_id,
                "gpu_device_id": device_id,
            }
        )
    try:
        dataset = lgb.Dataset(features, label=labels, free_raw_data=True)
        lgb.train(parameters, dataset, num_boost_round=1)
    except Exception as exc:  # pragma: no cover - host/build dependent
        detail = " ".join(str(exc).split()) or exc.__class__.__name__
        return False, detail[:500]
    return True, "one real tree update completed"


def _base_parameters(seed: int, *, reproducible: bool) -> dict[str, object]:
    parameters: dict[str, object] = {
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed) + 1,
        "bagging_seed": int(seed) + 2,
        "data_random_seed": int(seed) + 3,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
    }
    if reproducible:
        parameters.update({"deterministic": True, "force_col_wise": True})
    return parameters


def _accelerator_targets(requested: str, resolved: BackendInfo) -> tuple[str, ...]:
    if requested == "cuda":
        return ("cuda",)
    if requested == "auto" and resolved.kind == "cuda":
        return ("cuda", "gpu")
    return ("gpu",)


def lightgbm_backend_parameters(
    compute_backend: str,
    seed: int,
    *,
    resolved_backend: BackendInfo | None = None,
    reproducible: bool = False,
) -> tuple[dict[str, object], str, str]:
    """Resolve LightGBM by executing the selected library backend.

    PyTorch, DirectML, ROCm, and OpenCL capabilities are independent. ``auto``
    therefore probes LightGBM itself and may use the deterministic CPU path.
    Explicit accelerator requests fail closed when either the requested tensor
    runtime or the compatible LightGBM target cannot execute a tree update.
    """

    if not isinstance(reproducible, bool):
        raise ValueError("reproducible must be a boolean")
    requested = str(compute_backend or "auto").strip().lower()
    if requested not in SUPPORTED_COMPUTE_BACKENDS:
        raise ValueError(f"unsupported LightGBM compute backend: {compute_backend!r}")

    resolved = resolved_backend or resolve_backend(requested)
    if requested != "auto":
        resolved = require_backend(resolved)
    parameters = _base_parameters(seed, reproducible=reproducible)
    if requested == "cpu":
        parameters["device_type"] = "cpu"
        return parameters, "cpu", "cpu"

    platform_id, device_id, opencl_label = _opencl_device_override()
    failures: list[str] = []
    for target in _accelerator_targets(requested, resolved):
        probe_platform = platform_id if target == "gpu" else None
        probe_device = device_id if target == "gpu" else None
        available, reason = _probe_lightgbm_target(
            target,
            probe_platform,
            probe_device,
        )
        if not available:
            failures.append(f"{target}: {reason}")
            continue
        parameters.pop("deterministic", None)
        parameters.pop("force_col_wise", None)
        parameters["device_type"] = target
        if target == "cuda":
            device = resolved.device if resolved.kind == "cuda" else "cuda:auto"
            return parameters, "cuda", device
        if platform_id is not None and device_id is not None:
            parameters.update(
                {
                    "gpu_platform_id": platform_id,
                    "gpu_device_id": device_id,
                }
            )
        # LightGBM's deterministic switch is CPU-only. FP64 accumulation is
        # its documented mitigation for OpenCL run-to-run variance.
        parameters["gpu_use_dp"] = reproducible
        return parameters, "opencl", opencl_label

    if requested != "auto":
        detail = "; ".join(failures) or "no compatible target was probed"
        raise BackendUnavailableError(
            f"LightGBM cannot honor compute backend {requested!r}: {detail}"
        )
    parameters["device_type"] = "cpu"
    return parameters, "cpu", "cpu"


__all__ = [
    "OPENCL_DEVICE_ENV",
    "OPENCL_PLATFORM_ENV",
    "lightgbm_backend_parameters",
]
