"""Host-neutral compute backend discovery for training and inference.

The model contract is independent of the runtime used to evaluate it.  ``auto``
discovers an available accelerator at runtime and may fall back to the exact CPU
reference path.  An explicitly requested accelerator is *not* considered
satisfied when resolution falls back to CPU; callers can enforce that contract
with :func:`require_backend` or ``resolve_backend(..., require=True)``.

DirectML remains an optional compatibility adapter for supported Windows/WSL
installations.  It is not inferred from the operating system or GPU vendor.
Modern PyTorch installations are queried through ``torch.accelerator`` first,
with CUDA, ROCm, Intel XPU, Apple MPS, and legacy DirectML probes retained for
older runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any, Literal

BackendKind = Literal["cpu", "cuda", "rocm", "xpu", "directml", "mps"]

SUPPORTED_COMPUTE_BACKENDS = (
    "auto",
    "cpu",
    "cuda",
    "rocm",
    "xpu",
    "mps",
    "directml",
)
ACCELERATOR_COMPUTE_BACKENDS = frozenset(SUPPORTED_COMPUTE_BACKENDS) - {
    "auto",
    "cpu",
}
DEVICE_INDEX_ENV = "SIMPLE_AI_TRADING_DEVICE_INDEX"


class BackendUnavailableError(RuntimeError):
    """Raised when a pinned compute backend cannot be honored exactly."""


@dataclass(frozen=True)
class BackendInfo:
    """Resolved compute backend and auditable device-selection evidence."""

    requested: str
    kind: BackendKind
    device: str
    vendor: str
    reason: str
    selection: str = ""

    @property
    def accelerated(self) -> bool:
        return self.kind != "cpu"

    @property
    def request_satisfied(self) -> bool:
        return self.requested == "auto" or self.requested == self.kind

    @property
    def fell_back(self) -> bool:
        return not self.request_satisfied


def _probe_torch() -> tuple[Any | None, str]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - environmental
        return None, f"torch not importable ({exc.__class__.__name__})"
    return torch, ""


def _probe_torch_directml() -> tuple[Any | None, str]:
    try:
        return importlib.import_module("torch_directml"), ""
    except Exception as exc:  # pragma: no cover - environmental
        return None, f"torch-directml not importable ({exc.__class__.__name__})"


def _clean_device_name(value: object, fallback: str) -> str:
    candidate = str(value or "").replace("\x00", "").strip()
    return candidate or fallback


def _configured_device_index() -> int | None:
    raw = os.getenv(DEVICE_INDEX_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        index = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{DEVICE_INDEX_ENV} must be a non-negative integer") from exc
    if index < 0:
        raise ValueError(f"{DEVICE_INDEX_ENV} must be a non-negative integer")
    return index


def _select_device_index(
    count: int,
    *,
    current: int | None,
    free_memory: dict[int, int] | None = None,
) -> tuple[int, str]:
    if count <= 0:
        raise ValueError("accelerator reports no devices")
    configured = _configured_device_index()
    if configured is not None:
        if configured >= count:
            raise ValueError(
                f"{DEVICE_INDEX_ENV}={configured} is outside the discovered device range 0..{count - 1}"
            )
        return configured, f"operator_override:{DEVICE_INDEX_ENV}={configured}"
    usable_memory = {
        int(index): int(value)
        for index, value in (free_memory or {}).items()
        if 0 <= int(index) < count and int(value) >= 0
    }
    if usable_memory:
        selected = min(
            usable_memory,
            key=lambda index: (-usable_memory[index], index),
        )
        return selected, f"maximum_reported_free_memory:{usable_memory[selected]}"
    if current is not None and 0 <= int(current) < count:
        return int(current), "runtime_current_device"
    return 0, "first_verified_device"


def _cuda_free_memory(torch: Any, count: int) -> dict[int, int]:
    memory: dict[int, int] = {}
    getter = getattr(getattr(torch, "cuda", None), "mem_get_info", None)
    if not callable(getter):
        return memory
    for index in range(count):
        try:
            free, total = getter(index)
            free_value = int(free)
            total_value = int(total)
        except Exception:
            continue
        if free_value >= 0 and total_value > 0 and free_value <= total_value:
            memory[index] = free_value
    return memory


def _runtime_current_index(runtime: object) -> int | None:
    getter = getattr(runtime, "current_device", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def _try_cuda() -> BackendInfo | None:
    torch, _err = _probe_torch()
    if torch is None:
        return None
    try:
        if getattr(torch.version, "hip", None):
            return None
        runtime = torch.cuda
        if not runtime.is_available():
            return None
        count = int(runtime.device_count())
        index, selection = _select_device_index(
            count,
            current=_runtime_current_index(runtime),
            free_memory=_cuda_free_memory(torch, count),
        )
        name = runtime.get_device_name(index)
    except Exception:  # pragma: no cover - driver/configuration boundary
        return None
    return BackendInfo(
        requested="cuda",
        kind="cuda",
        device=f"cuda:{index}",
        vendor=_clean_device_name(name, "CUDA accelerator"),
        reason="",
        selection=selection,
    )


def _try_rocm() -> BackendInfo | None:
    torch, _err = _probe_torch()
    if torch is None:
        return None
    try:
        if not getattr(torch.version, "hip", None):
            return None
        runtime = torch.cuda  # ROCm intentionally uses PyTorch's cuda namespace.
        if not runtime.is_available():
            return None
        count = int(runtime.device_count())
        index, selection = _select_device_index(
            count,
            current=_runtime_current_index(runtime),
            free_memory=_cuda_free_memory(torch, count),
        )
        name = runtime.get_device_name(index)
    except Exception:  # pragma: no cover - driver/configuration boundary
        return None
    return BackendInfo(
        requested="rocm",
        kind="rocm",
        device=f"cuda:{index}",
        vendor=_clean_device_name(name, "ROCm accelerator"),
        reason="",
        selection=selection,
    )


def _try_xpu() -> BackendInfo | None:
    torch, _err = _probe_torch()
    if torch is None:
        return None
    try:
        runtime = getattr(torch, "xpu", None)
        if runtime is None or not runtime.is_available():
            return None
        count = int(runtime.device_count())
        index, selection = _select_device_index(
            count,
            current=_runtime_current_index(runtime),
        )
        name_getter = getattr(runtime, "get_device_name", None)
        name = name_getter(index) if callable(name_getter) else "Intel XPU"
    except Exception:  # pragma: no cover - driver/configuration boundary
        return None
    return BackendInfo(
        requested="xpu",
        kind="xpu",
        device=f"xpu:{index}",
        vendor=_clean_device_name(name, "Intel XPU"),
        reason="",
        selection=selection,
    )


def _try_mps() -> BackendInfo | None:
    torch, _err = _probe_torch()
    if torch is None:
        return None
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return None
    except Exception:  # pragma: no cover - driver boundary
        return None
    return BackendInfo(
        requested="mps",
        kind="mps",
        device="mps",
        vendor="Apple MPS",
        reason="",
        selection="runtime_single_device",
    )


def _try_torch_accelerator() -> BackendInfo | None:
    """Use PyTorch's device-agnostic API when the installed version provides it."""

    torch, _err = _probe_torch()
    if torch is None:
        return None
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return None
    try:
        if not bool(accelerator.is_available()):
            return None
        device = accelerator.current_accelerator(check_available=True)
        if device is None:
            return None
        device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
    except Exception:  # pragma: no cover - optional modern runtime boundary
        return None
    if device_type == "cuda":
        return _try_rocm() if getattr(torch.version, "hip", None) else _try_cuda()
    if device_type == "xpu":
        return _try_xpu()
    if device_type == "mps":
        return _try_mps()
    return None


def _try_directml() -> BackendInfo | None:
    torch, _torch_err = _probe_torch()
    directml, _directml_err = _probe_torch_directml()
    if torch is None or directml is None:
        return None
    try:
        is_available = getattr(directml, "is_available", None)
        if callable(is_available) and not bool(is_available()):
            return None
        count_getter = getattr(directml, "device_count", None)
        count = int(count_getter()) if callable(count_getter) else 1
        default_getter = getattr(directml, "default_device", None)
        current = int(default_getter()) if callable(default_getter) else None
        index, selection = _select_device_index(count, current=current)
        try:
            device = directml.device(index)
        except TypeError:
            if index != 0:
                raise
            device = directml.device()
        name_getter = getattr(directml, "device_name", None)
        name = name_getter(index) if callable(name_getter) else "DirectML"
    except Exception:  # pragma: no cover - driver/configuration boundary
        return None
    return BackendInfo(
        requested="directml",
        kind="directml",
        device=str(device),
        vendor=_clean_device_name(name, "DirectML"),
        reason="",
        selection=selection,
    )


def _cpu(requested: str, reason: str = "") -> BackendInfo:
    return BackendInfo(
        requested=requested,
        kind="cpu",
        device="cpu",
        vendor="portable CPU reference",
        reason=reason,
        selection="deterministic_cpu_reference",
    )


def default_compute_backend() -> str:
    """Return the platform-independent default backend request."""

    return "auto"


def require_backend(info: BackendInfo) -> BackendInfo:
    """Reject a CPU fallback when the operator pinned a different backend."""

    if not info.request_satisfied:
        detail = info.reason or f"resolved to {info.kind}"
        raise BackendUnavailableError(
            f"requested compute backend {info.requested!r} is unavailable: {detail}"
        )
    return info


def resolve_backend(requested: str | None, *, require: bool = False) -> BackendInfo:
    """Resolve a backend request without inferring capability from host identity.

    ``auto`` is allowed to resolve to CPU.  A pinned accelerator returns a CPU
    diagnostic when unavailable unless ``require=True`` requests fail-closed
    behavior.
    """

    name = (requested or default_compute_backend()).strip().lower()
    if name == "cpu":
        info = _cpu("cpu")
    elif name == "cuda":
        info = _try_cuda() or _cpu(
            "cuda", reason="CUDA unavailable or its selected device is invalid"
        )
    elif name == "rocm":
        info = _try_rocm() or _cpu(
            "rocm", reason="ROCm unavailable or its selected device is invalid"
        )
    elif name == "xpu":
        info = _try_xpu() or _cpu(
            "xpu", reason="Intel XPU unavailable or its selected device is invalid"
        )
    elif name == "directml":
        info = _try_directml() or _cpu(
            "directml",
            reason="DirectML unavailable or its selected device is invalid",
        )
    elif name == "mps":
        info = _try_mps() or _cpu(
            "mps", reason="MPS unavailable or its selected device is invalid"
        )
    elif name == "auto":
        probes = (
            _try_torch_accelerator,
            _try_rocm,
            _try_cuda,
            _try_xpu,
            _try_mps,
            _try_directml,
        )
        selected = next((candidate for probe in probes if (candidate := probe()) is not None), None)
        if selected is None:
            info = _cpu(
                "auto",
                reason="No supported accelerator passed runtime discovery; using CPU reference",
            )
        else:
            info = BackendInfo(
                requested="auto",
                kind=selected.kind,
                device=selected.device,
                vendor=selected.vendor,
                reason="",
                selection=selected.selection,
            )
    else:
        info = _cpu(name, reason=f"Unknown backend {requested!r}")
    return require_backend(info) if require else info


def backend_fallback_allowed(info: BackendInfo) -> bool:
    """Return whether an operation-level accelerator failure may use CPU."""

    return info.requested == "auto"


def torch_device_for_backend(info: BackendInfo) -> object:
    """Return the exact torch device selected by backend discovery."""

    if info.kind != "directml":
        return info.device
    directml, reason = _probe_torch_directml()
    if directml is None:
        raise BackendUnavailableError(reason or "torch-directml is unavailable")
    try:
        index = int(info.device.rsplit(":", 1)[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise BackendUnavailableError(
            f"invalid resolved DirectML device {info.device!r}"
        ) from exc
    try:
        return directml.device(index)
    except TypeError:
        if index != 0:
            raise BackendUnavailableError(
                "installed torch-directml cannot select the resolved non-default device"
            )
        return directml.device()


def describe_backend(info: BackendInfo) -> str:
    """Return a compact one-line description of the resolved backend."""

    fields = [
        f"requested={info.requested}",
        f"compute={info.kind}",
        f"device={info.device}",
        f"vendor={info.vendor}",
    ]
    if info.selection:
        fields.append(f"selection={info.selection}")
    if info.reason:
        fields.append(f"reason={info.reason}")
    return " ".join(fields)


__all__ = [
    "ACCELERATOR_COMPUTE_BACKENDS",
    "BackendInfo",
    "BackendKind",
    "BackendUnavailableError",
    "DEVICE_INDEX_ENV",
    "SUPPORTED_COMPUTE_BACKENDS",
    "backend_fallback_allowed",
    "default_compute_backend",
    "describe_backend",
    "require_backend",
    "resolve_backend",
    "torch_device_for_backend",
]
