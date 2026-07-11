"""Host-agnostic LightGBM CPU/OpenCL parameter resolution."""

from __future__ import annotations

import os

from .compute import BackendInfo, resolve_backend


OPENCL_PLATFORM_ENV = "SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID"
OPENCL_DEVICE_ENV = "SIMPLE_AI_TRADING_OPENCL_DEVICE_ID"


def lightgbm_backend_parameters(
    compute_backend: str,
    seed: int,
    *,
    resolved_backend: BackendInfo | None = None,
    reproducible: bool = False,
) -> tuple[dict[str, object], str, str]:
    """Resolve seeded LightGBM parameters without assuming device 0:0."""

    if not isinstance(reproducible, bool):
        raise ValueError("reproducible must be a boolean")

    backend = resolved_backend or resolve_backend(compute_backend)
    use_gpu = backend.kind != "cpu"
    parameters: dict[str, object] = {
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed) + 1,
        "bagging_seed": int(seed) + 2,
        "data_random_seed": int(seed) + 3,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "device_type": "gpu" if use_gpu else "cpu",
    }
    if not use_gpu:
        if reproducible:
            parameters.update({"deterministic": True, "force_col_wise": True})
        return parameters, "cpu", "cpu"

    platform_raw = (os.getenv(OPENCL_PLATFORM_ENV) or "").strip()
    device_raw = (os.getenv(OPENCL_DEVICE_ENV) or "").strip()
    if bool(platform_raw) != bool(device_raw):
        raise ValueError(
            f"{OPENCL_PLATFORM_ENV} and {OPENCL_DEVICE_ENV} must be set together"
        )
    device_label = "opencl:auto"
    if platform_raw:
        try:
            platform_id = int(platform_raw)
            device_id = int(device_raw)
        except ValueError as exc:
            raise ValueError("invalid OpenCL platform or device id") from exc
        if platform_id < 0 or device_id < 0:
            raise ValueError("OpenCL platform and device ids must be non-negative")
        parameters.update(
            {
                "gpu_platform_id": platform_id,
                "gpu_device_id": device_id,
            }
        )
        device_label = f"opencl:{platform_id}:{device_id}"
    # LightGBM's deterministic switch is CPU-only. FP64 accumulation is its
    # documented mitigation for OpenCL run-to-run variance.
    parameters["gpu_use_dp"] = reproducible
    return parameters, "opencl", device_label


__all__ = [
    "OPENCL_DEVICE_ENV",
    "OPENCL_PLATFORM_ENV",
    "lightgbm_backend_parameters",
]
