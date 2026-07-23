"""Capability-tested LightGBM CPU, OpenCL, and CUDA resolution."""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
import platform

from .compute import (
    BackendInfo,
    BackendUnavailableError,
    SUPPORTED_COMPUTE_BACKENDS,
    require_backend,
    resolve_backend,
)


OPENCL_PLATFORM_ENV = "SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID"
OPENCL_DEVICE_ENV = "SIMPLE_AI_TRADING_OPENCL_DEVICE_ID"
SUPPORTED_LIGHTGBM_BACKEND_KINDS = frozenset({"cpu", "opencl", "cuda"})

_CL_DEVICE_TYPE_GPU = 1 << 2
_CL_PLATFORM_VERSION = 0x0901
_CL_PLATFORM_NAME = 0x0902
_CL_PLATFORM_VENDOR = 0x0903
_CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
_CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
_CL_DEVICE_NAME = 0x102B
_CL_DEVICE_VENDOR = 0x102C
_CL_DRIVER_VERSION = 0x102D
_CL_DEVICE_VERSION = 0x102F
_CL_DEVICE_EXTENSIONS = 0x1030
_CL_DEVICE_BOARD_NAME_AMD = 0x4038


@dataclass(frozen=True)
class OpenCLDeviceIdentity:
    platform_id: int
    device_id: int
    platform_name: str
    platform_vendor: str
    platform_version: str
    device_name: str
    board_name: str
    device_vendor: str
    device_version: str
    driver_version: str
    global_memory_bytes: int
    maximum_compute_units: int

    def as_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["display_name"] = self.board_name or self.device_name
        return output

    @property
    def display_name(self) -> str:
        return self.board_name or self.device_name


def _opencl_library() -> ctypes.CDLL:
    system = platform.system().lower()
    if system == "windows":
        return ctypes.WinDLL("OpenCL.dll")
    if system == "darwin":
        return ctypes.CDLL("/System/Library/Frameworks/OpenCL.framework/OpenCL")
    name = ctypes.util.find_library("OpenCL")
    if not name:
        raise RuntimeError("OpenCL loader library is unavailable")
    return ctypes.CDLL(name)


def _opencl_text(
    function: object,
    handle: ctypes.c_void_p,
    parameter: int,
) -> str:
    size = ctypes.c_size_t()
    status = function(handle, parameter, 0, None, ctypes.byref(size))
    if status != 0 or size.value <= 1:
        raise RuntimeError(f"OpenCL text query failed with status {status}")
    buffer = ctypes.create_string_buffer(size.value)
    status = function(handle, parameter, size.value, buffer, None)
    if status != 0:
        raise RuntimeError(f"OpenCL text query failed with status {status}")
    return buffer.value.decode("utf-8", errors="replace").strip()


def _opencl_number(
    function: object,
    handle: ctypes.c_void_p,
    parameter: int,
    number_type: type[ctypes._SimpleCData],
) -> int:
    value = number_type()
    status = function(
        handle,
        parameter,
        ctypes.sizeof(value),
        ctypes.byref(value),
        None,
    )
    if status != 0:
        raise RuntimeError(f"OpenCL numeric query failed with status {status}")
    return int(value.value)


def _opencl_optional_text(
    function: object,
    handle: ctypes.c_void_p,
    parameter: int,
) -> str:
    try:
        return _opencl_text(function, handle, parameter)
    except RuntimeError:
        return ""


@lru_cache(maxsize=1)
def discover_opencl_gpu_devices() -> tuple[OpenCLDeviceIdentity, ...]:
    """Enumerate actual OpenCL GPU identities without an optional dependency."""

    library = _opencl_library()
    get_platform_ids = library.clGetPlatformIDs
    get_platform_ids.argtypes = (
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint),
    )
    get_platform_ids.restype = ctypes.c_int
    get_platform_info = library.clGetPlatformInfo
    get_platform_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    )
    get_platform_info.restype = ctypes.c_int
    get_device_ids = library.clGetDeviceIDs
    get_device_ids.argtypes = (
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint),
    )
    get_device_ids.restype = ctypes.c_int
    get_device_info = library.clGetDeviceInfo
    get_device_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    )
    get_device_info.restype = ctypes.c_int

    platform_count = ctypes.c_uint()
    status = get_platform_ids(0, None, ctypes.byref(platform_count))
    if status != 0 or platform_count.value == 0:
        raise RuntimeError(f"OpenCL platform enumeration failed with status {status}")
    platforms = (ctypes.c_void_p * platform_count.value)()
    status = get_platform_ids(platform_count.value, platforms, None)
    if status != 0:
        raise RuntimeError(f"OpenCL platform enumeration failed with status {status}")
    output: list[OpenCLDeviceIdentity] = []
    for platform_id, platform_handle in enumerate(platforms):
        device_count = ctypes.c_uint()
        status = get_device_ids(
            platform_handle,
            _CL_DEVICE_TYPE_GPU,
            0,
            None,
            ctypes.byref(device_count),
        )
        if status == -1 or device_count.value == 0:
            continue
        if status != 0:
            raise RuntimeError(f"OpenCL GPU enumeration failed with status {status}")
        devices = (ctypes.c_void_p * device_count.value)()
        status = get_device_ids(
            platform_handle,
            _CL_DEVICE_TYPE_GPU,
            device_count.value,
            devices,
            None,
        )
        if status != 0:
            raise RuntimeError(f"OpenCL GPU enumeration failed with status {status}")
        platform_name = _opencl_text(
            get_platform_info,
            platform_handle,
            _CL_PLATFORM_NAME,
        )
        platform_vendor = _opencl_text(
            get_platform_info,
            platform_handle,
            _CL_PLATFORM_VENDOR,
        )
        platform_version = _opencl_text(
            get_platform_info,
            platform_handle,
            _CL_PLATFORM_VERSION,
        )
        for device_id, device_handle in enumerate(devices):
            extensions = _opencl_text(
                get_device_info,
                device_handle,
                _CL_DEVICE_EXTENSIONS,
            )
            output.append(
                OpenCLDeviceIdentity(
                    platform_id=platform_id,
                    device_id=device_id,
                    platform_name=platform_name,
                    platform_vendor=platform_vendor,
                    platform_version=platform_version,
                    device_name=_opencl_text(
                        get_device_info,
                        device_handle,
                        _CL_DEVICE_NAME,
                    ),
                    board_name=(
                        _opencl_optional_text(
                            get_device_info,
                            device_handle,
                            _CL_DEVICE_BOARD_NAME_AMD,
                        )
                        if "cl_amd_device_attribute_query" in extensions.split()
                        else ""
                    ),
                    device_vendor=_opencl_text(
                        get_device_info,
                        device_handle,
                        _CL_DEVICE_VENDOR,
                    ),
                    device_version=_opencl_text(
                        get_device_info,
                        device_handle,
                        _CL_DEVICE_VERSION,
                    ),
                    driver_version=_opencl_text(
                        get_device_info,
                        device_handle,
                        _CL_DRIVER_VERSION,
                    ),
                    global_memory_bytes=_opencl_number(
                        get_device_info,
                        device_handle,
                        _CL_DEVICE_GLOBAL_MEM_SIZE,
                        ctypes.c_ulonglong,
                    ),
                    maximum_compute_units=_opencl_number(
                        get_device_info,
                        device_handle,
                        _CL_DEVICE_MAX_COMPUTE_UNITS,
                        ctypes.c_uint,
                    ),
                )
            )
    if not output:
        raise RuntimeError("OpenCL exposes no GPU devices")
    return tuple(output)


def selected_opencl_gpu_device(
    platform_id: int,
    device_id: int,
) -> OpenCLDeviceIdentity:
    matching = [
        device
        for device in discover_opencl_gpu_devices()
        if device.platform_id == platform_id and device.device_id == device_id
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"OpenCL GPU identity is unavailable: {platform_id}:{device_id}"
        )
    return matching[0]


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

    features = np.arange(512 * 8, dtype=np.float32).reshape(512, 8) % np.float32(97.0)
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
    pin_opencl_device: bool = False,
) -> tuple[dict[str, object], str, str]:
    """Resolve LightGBM by executing the selected library backend.

    PyTorch, DirectML, ROCm, and OpenCL capabilities are independent. ``auto``
    therefore probes LightGBM itself and may use the deterministic CPU path.
    Explicit accelerator requests fail closed when either the requested tensor
    runtime or the compatible LightGBM target cannot execute a tree update.
    """

    if not isinstance(reproducible, bool):
        raise ValueError("reproducible must be a boolean")
    if not isinstance(pin_opencl_device, bool):
        raise ValueError("pin_opencl_device must be a boolean")
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
        if target == "gpu" and pin_opencl_device:
            try:
                if platform_id is None:
                    selected = discover_opencl_gpu_devices()[0]
                    platform_id = selected.platform_id
                    device_id = selected.device_id
                else:
                    selected = selected_opencl_gpu_device(
                        platform_id,
                        int(device_id),
                    )
            except RuntimeError as exc:
                failures.append(f"gpu identity: {exc}")
                continue
            opencl_label = f"opencl:{platform_id}:{device_id}:{selected.display_name}"
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
    "SUPPORTED_LIGHTGBM_BACKEND_KINDS",
    "OpenCLDeviceIdentity",
    "discover_opencl_gpu_devices",
    "lightgbm_backend_parameters",
    "selected_opencl_gpu_device",
]
