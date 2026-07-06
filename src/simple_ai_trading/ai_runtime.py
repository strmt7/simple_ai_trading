"""AI runtime configuration and host capability preflight."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess  # nosec B404
from dataclasses import asdict, dataclass

from .compute import BackendInfo, default_compute_backend, resolve_backend


@dataclass(frozen=True)
class AIRuntimeConfig:
    enabled: bool = True
    provider: str = "auto"
    model: str = "qwen3:8b"
    require_gpu: bool = True
    compute_backend: str = ""
    min_free_vram_gb: float = 8.0
    min_free_ram_gb: float = 16.0
    min_model_parameters_b: float = 2.0
    allow_paper_fallback: bool = True

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AICapabilityReport:
    ok: bool
    provider: str
    model: str
    gpu_vendor: str
    compute_backend_requested: str
    compute_backend_kind: str
    compute_backend_device: str
    compute_backend_reason: str
    free_vram_gb: float | None
    free_ram_gb: float | None
    model_parameters_b: float | None
    messages: tuple[str, ...]
    warnings: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed == parsed else None


def estimate_model_parameters_b(model: str) -> float | None:
    """Extract an approximate parameter count from common local model names."""

    text = str(model or "").lower()
    if not text or text in {"auto", "operator-selected-local-llm"}:
        return None
    matches = list(re.finditer(r"(?<![a-z0-9])e?(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", text))
    values: list[float] = []
    for match in matches:
        number = _safe_float(match.group(1))
        if number is None:
            continue
        unit = match.group(2)
        values.append(number if unit == "b" else number / 1000.0)
    if values:
        return max(values)
    return None


def _memory_status_gb() -> float | None:
    if platform.system().lower() == "windows":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024**3)
        except Exception:
            return None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def _run_capture(command: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _nvidia_free_vram_gb() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    output = _run_capture([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    values = [_safe_float(line.strip()) for line in output.splitlines() if line.strip()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values) / 1024.0


def _amd_free_vram_gb() -> float | None:
    exe = shutil.which("rocm-smi") or shutil.which("rocm-smi.exe")
    if not exe:
        return None
    output = _run_capture([exe, "--showmeminfo", "vram"])
    numbers = []
    for token in output.replace(",", " ").split():
        value = _safe_float(token)
        if value is not None:
            numbers.append(value)
    if not numbers:
        return None
    # rocm-smi commonly reports MiB. Use the largest visible number as a
    # conservative free-VRAM proxy when detailed parsing is unavailable.
    return max(numbers) / 1024.0


def _windows_gpu_names() -> tuple[str, ...]:
    if platform.system().lower() != "windows":
        return ()
    output = _run_capture(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ],
        timeout=5.0,
    )
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _gpu_vendor_from_backend(backend: BackendInfo, names: tuple[str, ...]) -> str:
    joined = " ".join(names).lower()
    if "amd" in joined or "radeon" in joined:
        return "amd"
    if "nvidia" in joined or "geforce" in joined or "rtx" in joined or "gtx" in joined:
        return "nvidia"
    if "intel" in joined or "arc" in joined or "iris" in joined:
        return "intel"
    if backend.kind == "directml":
        return "directml"
    if backend.kind in {"cuda", "rocm", "mps"}:
        return backend.kind
    return "unknown"


def detect_ai_capabilities(config: AIRuntimeConfig | None = None) -> AICapabilityReport:
    cfg = config or AIRuntimeConfig()
    messages: list[str] = []
    warnings: list[str] = []
    backend = resolve_backend(cfg.compute_backend or default_compute_backend())
    free_ram = _memory_status_gb()
    nvidia_vram = _nvidia_free_vram_gb()
    amd_vram = _amd_free_vram_gb()
    windows_gpu_names = _windows_gpu_names()
    if nvidia_vram is not None:
        gpu_vendor = "nvidia"
        free_vram = nvidia_vram
    elif amd_vram is not None:
        gpu_vendor = "amd"
        free_vram = amd_vram
    else:
        gpu_vendor = _gpu_vendor_from_backend(backend, windows_gpu_names)
        free_vram = None

    if free_ram is None:
        messages.append("system RAM headroom could not be measured")
    elif free_ram < cfg.min_free_ram_gb:
        messages.append(f"free system RAM {free_ram:.1f} GiB is below required {cfg.min_free_ram_gb:.1f} GiB")

    if cfg.require_gpu:
        if backend.kind == "cpu":
            reason = f": {backend.reason}" if backend.reason else ""
            messages.append(f"AI requires a GPU compute backend; {backend.requested} resolved to CPU{reason}")
        elif free_vram is None:
            warnings.append(
                "free VRAM could not be measured through vendor tools; GPU backend functional check passed"
            )
        elif free_vram < cfg.min_free_vram_gb:
            messages.append(f"free VRAM {free_vram:.1f} GiB is below required {cfg.min_free_vram_gb:.1f} GiB")

    provider = cfg.provider
    if provider == "auto":
        provider = "local-gpu" if backend.kind != "cpu" else "cpu-only"
    model = cfg.model
    if model == "auto":
        model = "operator-selected-local-llm"
    model_parameters_b = estimate_model_parameters_b(model)

    if cfg.enabled:
        minimum_parameters_b = max(0.0, float(cfg.min_model_parameters_b))
        if model_parameters_b is None:
            warnings.append(
                "AI model parameter count could not be inferred from the model name; "
                "use a name like qwen3:8b for an enforceable multibillion check"
            )
        elif model_parameters_b < minimum_parameters_b:
            messages.append(
                f"AI model {model} is {model_parameters_b:.2f}B parameters, below required "
                f"{minimum_parameters_b:.2f}B"
            )

    if not cfg.enabled:
        messages.append("AI features are disabled")
    ok = cfg.enabled and not messages
    return AICapabilityReport(
        ok=ok,
        provider=provider,
        model=model,
        gpu_vendor=gpu_vendor,
        compute_backend_requested=backend.requested,
        compute_backend_kind=backend.kind,
        compute_backend_device=backend.device,
        compute_backend_reason=backend.reason,
        free_vram_gb=free_vram,
        free_ram_gb=free_ram,
        model_parameters_b=model_parameters_b,
        messages=tuple(messages),
        warnings=tuple(warnings),
    )


def render_ai_capability_report(report: AICapabilityReport) -> str:
    lines = [
        "AI capability report",
        f"ok={report.ok} provider={report.provider} model={report.model} gpu={report.gpu_vendor}",
        f"compute={report.compute_backend_kind} device={report.compute_backend_device}",
        f"free_vram_gb={report.free_vram_gb if report.free_vram_gb is not None else 'unknown'}",
        f"free_ram_gb={report.free_ram_gb if report.free_ram_gb is not None else 'unknown'}",
        f"model_parameters_b={report.model_parameters_b if report.model_parameters_b is not None else 'unknown'}",
    ]
    for message in report.messages:
        lines.append(f"blocked: {message}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
