"""AI runtime configuration and host capability preflight."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess  # nosec B404
from dataclasses import asdict, dataclass
import json
import math
from typing import Callable, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

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
    provider_available: bool = False
    model_available: bool = False
    model_local: bool = False

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OllamaResidencyReport:
    """Exact runtime residency reported by Ollama after model inference."""

    requested_model: str
    status: str
    loaded_model: str | None
    digest: str | None
    size_bytes: int | None
    size_vram_bytes: int | None
    vram_to_model_ratio: float | None

    @property
    def loaded(self) -> bool:
        return self.status in {"gpu_resident", "cpu_only"}

    @property
    def gpu_resident(self) -> bool:
        return self.status == "gpu_resident"

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "loaded": self.loaded,
            "gpu_resident": self.gpu_resident,
        }

    def validated(self) -> "OllamaResidencyReport":
        loaded_fields = (
            self.loaded_model,
            self.digest,
            self.size_bytes,
            self.size_vram_bytes,
            self.vram_to_model_ratio,
        )
        if (
            not isinstance(self.requested_model, str)
            or not self.requested_model
            or self.status not in {"unloaded", "gpu_resident", "cpu_only"}
        ):
            raise ValueError("Ollama residency report is invalid")
        if self.status == "unloaded":
            if any(value is not None for value in loaded_fields):
                raise ValueError(
                    "Ollama unloaded residency report contains runtime data"
                )
            return self
        if (
            not isinstance(self.loaded_model, str)
            or not self.loaded_model
            or not _is_sha256(self.digest)
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or isinstance(self.size_vram_bytes, bool)
            or not isinstance(self.size_vram_bytes, int)
            or self.size_vram_bytes < 0
            or isinstance(self.vram_to_model_ratio, bool)
            or not isinstance(self.vram_to_model_ratio, (int, float))
            or not math.isfinite(float(self.vram_to_model_ratio))
            or self.vram_to_model_ratio < 0.0
            or not math.isclose(
                float(self.vram_to_model_ratio),
                self.size_vram_bytes / self.size_bytes,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or self.gpu_resident != (self.size_vram_bytes > 0)
        ):
            raise ValueError("Ollama loaded residency report is invalid")
        return self


JsonGetter = Callable[[str, float], object]
_OLLAMA_RESIDENCY_PAYLOAD_FIELDS = {
    "requested_model",
    "status",
    "loaded_model",
    "digest",
    "size_bytes",
    "size_vram_bytes",
    "vram_to_model_ratio",
    "loaded",
    "gpu_resident",
}


def ollama_residency_from_mapping(value: object) -> OllamaResidencyReport:
    """Reconstruct strict residency evidence from a persisted JSON mapping."""

    if not isinstance(value, Mapping) or set(value) != _OLLAMA_RESIDENCY_PAYLOAD_FIELDS:
        raise ValueError("Ollama residency payload fields are invalid")
    report = OllamaResidencyReport(
        requested_model=value["requested_model"],
        status=value["status"],
        loaded_model=value["loaded_model"],
        digest=value["digest"],
        size_bytes=value["size_bytes"],
        size_vram_bytes=value["size_vram_bytes"],
        vram_to_model_ratio=value["vram_to_model_ratio"],
    ).validated()
    if (
        value["loaded"] is not report.loaded
        or value["gpu_resident"] is not report.gpu_resident
    ):
        raise ValueError("Ollama residency payload flags are invalid")
    return report


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_value(text: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant: {value}")
        ),
    )


def _get_json(url: str, timeout: float) -> object:
    request = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:  # nosec B310 - caller supplies local provider URL
            body = response.read(2_000_001)
    except (OSError, urllib_error.URLError) as exc:
        raise ValueError("Ollama runtime inventory is unavailable") from exc
    if len(body) > 2_000_000:
        raise ValueError("Ollama runtime inventory exceeds the response limit")
    try:
        return _strict_json_value(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Ollama runtime inventory is invalid JSON") from exc


def _normalized_model_name(value: object) -> str:
    name = str(value or "").strip().lower()
    return name if ":" in name else f"{name}:latest"


def inspect_ollama_model_residency(
    base_url: str,
    model: str,
    timeout_seconds: float = 2.0,
    *,
    expected_digest: str | None = None,
    get_json: JsonGetter = _get_json,
) -> OllamaResidencyReport:
    """Read `/api/ps` and bind runtime residency to one exact local model."""

    requested_model = str(model or "").strip()
    if not requested_model:
        raise ValueError("Ollama residency model is missing")
    if expected_digest is not None and not _is_sha256(expected_digest):
        raise ValueError("Ollama residency expected digest is invalid")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Ollama residency timeout is invalid")
    endpoint = f"{str(base_url or 'http://127.0.0.1:11434').rstrip('/')}/api/ps"
    payload = get_json(endpoint, timeout)
    if not isinstance(payload, Mapping) or set(payload) != {"models"}:
        raise ValueError("Ollama runtime inventory fields are invalid")
    models = payload["models"]
    if not isinstance(models, list):
        raise ValueError("Ollama runtime model inventory is invalid")
    requested_normalized = _normalized_model_name(requested_model)
    matches: list[Mapping[str, object]] = []
    for raw in models:
        if not isinstance(raw, Mapping):
            raise ValueError("Ollama runtime model entry is invalid")
        digest = raw.get("digest")
        names = {
            _normalized_model_name(raw.get("name")),
            _normalized_model_name(raw.get("model")),
        }
        if expected_digest is not None:
            matched = digest == expected_digest
        else:
            matched = requested_normalized in names
        if matched:
            matches.append(raw)
    if not matches:
        return OllamaResidencyReport(
            requested_model=requested_model,
            status="unloaded",
            loaded_model=None,
            digest=None,
            size_bytes=None,
            size_vram_bytes=None,
            vram_to_model_ratio=None,
        ).validated()
    if len(matches) != 1:
        raise ValueError("Ollama runtime inventory contains ambiguous model entries")
    raw = matches[0]
    loaded_model = raw.get("model") or raw.get("name")
    digest = raw.get("digest")
    size = raw.get("size")
    size_vram = raw.get("size_vram")
    if (
        not isinstance(loaded_model, str)
        or not loaded_model
        or not _is_sha256(digest)
        or (expected_digest is not None and digest != expected_digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or isinstance(size_vram, bool)
        or not isinstance(size_vram, int)
        or size_vram < 0
    ):
        raise ValueError("Ollama runtime model residency fields are invalid")
    return OllamaResidencyReport(
        requested_model=requested_model,
        status="gpu_resident" if size_vram > 0 else "cpu_only",
        loaded_model=loaded_model,
        digest=digest,
        size_bytes=size,
        size_vram_bytes=size_vram,
        vram_to_model_ratio=size_vram / size,
    ).validated()


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
    matches = list(
        re.finditer(r"(?<![a-z0-9])e?(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", text)
    )
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


def _ollama_inventory() -> dict[str, bool] | None:
    """Return model-name -> local-weights availability, or None when Ollama is unavailable."""

    exe = shutil.which("ollama")
    if not exe:
        return None
    output = _run_capture([exe, "list"], timeout=10.0)
    if not output:
        return None
    inventory: dict[str, bool] = {}
    for line in output.splitlines()[1:]:
        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) < 3:
            continue
        name = columns[0].strip()
        size = columns[2].strip()
        if name:
            inventory[name] = size not in {"", "-"} and not name.lower().endswith(
                ":cloud"
            )
    return inventory


def _nvidia_free_vram_gb() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    output = _run_capture(
        [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
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
        messages.append(
            f"free system RAM {free_ram:.1f} GiB is below required {cfg.min_free_ram_gb:.1f} GiB"
        )

    if cfg.require_gpu:
        if backend.kind == "cpu":
            reason = f": {backend.reason}" if backend.reason else ""
            messages.append(
                f"AI requires a GPU compute backend; {backend.requested} resolved to CPU{reason}"
            )
        elif free_vram is None:
            warnings.append(
                "free VRAM could not be measured through vendor tools; GPU backend functional check passed"
            )
        elif free_vram < cfg.min_free_vram_gb:
            messages.append(
                f"free VRAM {free_vram:.1f} GiB is below required {cfg.min_free_vram_gb:.1f} GiB"
            )

    provider = cfg.provider
    if provider in {"auto", "local-gpu"}:
        provider = "ollama"
    model = cfg.model
    if model == "auto":
        model = "operator-selected-local-llm"
    model_parameters_b = estimate_model_parameters_b(model)
    provider_available = False
    model_available = False
    model_local = False

    if cfg.enabled:
        if provider == "ollama":
            inventory = _ollama_inventory()
            provider_available = inventory is not None
            if inventory is None:
                messages.append(
                    "Ollama is not installed, not running, or returned no model inventory"
                )
            else:
                candidates = {model, f"{model}:latest"} if ":" not in model else {model}
                selected = next(
                    (name for name in candidates if name in inventory), None
                )
                model_available = selected is not None
                model_local = bool(selected is not None and inventory[selected])
                if not model_available:
                    messages.append(f"AI model {model} is not installed in Ollama")
                elif not model_local:
                    messages.append(
                        f"AI model {model} has no local weights and cannot satisfy local GPU AI"
                    )
        else:
            messages.append(
                f"AI provider {provider} cannot be verified as a supported local provider"
            )
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
        provider_available=provider_available,
        model_available=model_available,
        model_local=model_local,
    )


def render_ai_capability_report(report: AICapabilityReport) -> str:
    lines = [
        "AI capability report",
        f"ok={report.ok} provider={report.provider} model={report.model} gpu={report.gpu_vendor}",
        f"compute={report.compute_backend_kind} device={report.compute_backend_device}",
        f"free_vram_gb={report.free_vram_gb if report.free_vram_gb is not None else 'unknown'}",
        f"free_ram_gb={report.free_ram_gb if report.free_ram_gb is not None else 'unknown'}",
        f"model_parameters_b={report.model_parameters_b if report.model_parameters_b is not None else 'unknown'}",
        f"provider_available={report.provider_available} model_available={report.model_available} "
        f"model_local={report.model_local}",
    ]
    for message in report.messages:
        lines.append(f"blocked: {message}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
