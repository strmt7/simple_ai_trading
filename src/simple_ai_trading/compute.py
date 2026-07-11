"""Compute backend selection for training and inference.

Windows defaults to DirectML because it gives one GPU path across AMD, NVIDIA,
and Intel devices. CPU-only mode is still supported for wider installability,
but callers should warn the operator because model training, retraining, and
backtest scoring will be much slower and AI features are not allowed there.

``RuntimeConfig.compute_backend`` may be set to one of:

    * ``"cpu"``      - stdlib Python math (default, always available).
    * ``"cuda"``     - NVIDIA GPU via PyTorch (requires a CUDA PyTorch build).
    * ``"rocm"``     - AMD GPU via PyTorch (requires a ROCm PyTorch build).
    * ``"directml"`` - Windows GPU via ``torch-directml``; preferred on Windows.
    * ``"mps"``      - Apple Silicon via PyTorch MPS.
    * ``"auto"``     - probe the platform-preferred GPU stack, else CPU.

The selection never silently installs anything; if the requested backend is not
usable on the current host, :func:`resolve_backend` returns a ``BackendInfo``
whose ``kind`` is ``"cpu"`` and whose ``reason`` explains why, so the caller can
surface that to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import platform
from typing import Any, Literal

BackendKind = Literal["cpu", "cuda", "rocm", "directml", "mps"]


@dataclass(frozen=True)
class BackendInfo:
    """Resolved backend.

    Attributes:
        requested: The value supplied by the operator.
        kind: What was actually selected and is safe to use.
        device: A device identifier usable with torch (e.g. ``"cuda:0"``).
        vendor: Best-effort vendor label, for display.
        reason: Human-readable explanation of fallbacks, blank on success.
    """

    requested: str
    kind: BackendKind
    device: str
    vendor: str
    reason: str


def _probe_torch() -> tuple[Any | None, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environmental
        return None, f"torch not importable ({exc.__class__.__name__})"
    return torch, ""


def _probe_torch_directml() -> tuple[Any | None, str]:
    try:
        return importlib.import_module("torch_directml"), ""
    except Exception as exc:  # pragma: no cover - environmental
        return None, f"torch-directml not importable ({exc.__class__.__name__})"


def _try_cuda() -> BackendInfo | None:
    torch, err = _probe_torch()
    _ = err
    if torch is None:
        return None
    try:
        if getattr(torch.version, "hip", None):
            return None
        if not torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return None
        name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - driver corner case
        return None
    return BackendInfo(
        requested="cuda",
        kind="cuda",
        device="cuda:0",
        vendor=str(name),
        reason="",
    )


def _try_rocm() -> BackendInfo | None:
    torch, err = _probe_torch()
    if torch is None:
        return None
    try:
        # ROCm builds of PyTorch still expose their devices under the "cuda" namespace.
        version = getattr(torch.version, "hip", None)
        if not version:
            return None
        if not torch.cuda.is_available():
            return None
        if torch.cuda.device_count() <= 0:
            return None
        name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        return None
    return BackendInfo(
        requested="rocm",
        kind="rocm",
        device="cuda:0",
        vendor=str(name),
        reason="",
    )


def _try_mps() -> BackendInfo | None:
    torch, err = _probe_torch()
    if torch is None:
        return None
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return None
    try:
        if not mps.is_available():
            return None
    except Exception:  # pragma: no cover
        return None
    return BackendInfo(
        requested="mps",
        kind="mps",
        device="mps",
        vendor="Apple MPS",
        reason="",
    )


def _try_directml() -> BackendInfo | None:
    torch, _torch_err = _probe_torch()
    directml, _directml_err = _probe_torch_directml()
    if torch is None or directml is None:
        return None
    try:
        is_available = getattr(directml, "is_available", None)
        if callable(is_available) and not bool(is_available()):
            return None
        device = directml.device()
    except Exception:  # pragma: no cover - driver corner case
        return None
    return BackendInfo(
        requested="directml",
        kind="directml",
        device=str(device),
        vendor="DirectML",
        reason="",
    )


def _cpu(requested: str, reason: str = "") -> BackendInfo:
    return BackendInfo(
        requested=requested,
        kind="cpu",
        device="cpu",
        vendor="Python stdlib",
        reason=reason,
    )


def default_compute_backend() -> str:
    """Return the operator-friendly default backend for the current platform."""

    if platform.system().lower() == "windows":
        return "directml"
    return "auto"


def resolve_backend(requested: str | None) -> BackendInfo:
    """Resolve a requested backend name to a usable ``BackendInfo``.

    The function never raises on unsupported input; it falls back to CPU and
    includes a reason in the return value.
    """

    name = (requested or default_compute_backend()).strip().lower()
    if name == "cpu":
        return _cpu("cpu")

    if name == "cuda":
        info = _try_cuda()
        if info is not None:
            return info
        return _cpu("cuda", reason="CUDA unavailable (torch missing or no CUDA device)")

    if name == "rocm":
        info = _try_rocm()
        if info is not None:
            return info
        return _cpu(
            "rocm", reason="ROCm unavailable (torch missing or not a ROCm build)"
        )

    if name == "directml":
        info = _try_directml()
        if info is not None:
            return info
        return _cpu(
            "directml",
            reason="DirectML unavailable (torch-directml missing or no device)",
        )

    if name == "mps":
        info = _try_mps()
        if info is not None:
            return info
        return _cpu(
            "mps", reason="MPS unavailable (torch missing or not Apple Silicon)"
        )

    if name == "auto":
        probes = (_try_rocm, _try_cuda, _try_directml, _try_mps)
        if platform.system().lower() == "windows":
            probes = (_try_directml, _try_cuda, _try_rocm, _try_mps)
        for probe in probes:
            info = probe()
            if info is not None:
                return BackendInfo(
                    requested="auto",
                    kind=info.kind,
                    device=info.device,
                    vendor=info.vendor,
                    reason="",
                )
        return _cpu("auto", reason="No GPU backend available; running on CPU-only mode")

    return _cpu(name, reason=f"Unknown backend {requested!r}; defaulting to CPU")


def describe_backend(info: BackendInfo) -> str:
    """Return a compact one-line description of the resolved backend."""
    suffix = f" — {info.reason}" if info.reason else ""
    return f"compute={info.kind} device={info.device} vendor={info.vendor}{suffix}"
