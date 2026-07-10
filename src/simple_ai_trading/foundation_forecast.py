"""Optional, integrity-gated financial foundation forecasting runtime."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from .compute import BackendInfo, resolve_backend
from .foundation_model_source import (
    FoundationSourceReport,
    provision_kronos_source,
    verify_kronos_source,
)


@dataclass(frozen=True)
class PinnedHuggingFaceArtifact:
    repository: str
    revision: str
    config_size: int
    config_sha256: str
    weights_size: int
    weights_sha256: str
    expected_parameters: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


KRONOS_TOKENIZER_ARTIFACT = PinnedHuggingFaceArtifact(
    repository="NeoQuasar/Kronos-Tokenizer-base",
    revision="0e0117387f39004a9016484a186a908917e22426",
    config_size=301,
    config_sha256="2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
    weights_size=15_842_368,
    weights_sha256="59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
    expected_parameters=3_958_042,
)

KRONOS_MODEL_ARTIFACTS = {
    "small": PinnedHuggingFaceArtifact(
        repository="NeoQuasar/Kronos-small",
        revision="901c26c1332695a2a8f243eb2f37243a37bea320",
        config_size=228,
        config_sha256="5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
        weights_size=98_980_656,
        weights_sha256="b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
        expected_parameters=24_741_376,
    ),
    "base": PinnedHuggingFaceArtifact(
        repository="NeoQuasar/Kronos-base",
        revision="2b554741eca47781b64468546e77fef3e85130e6",
        config_size=228,
        config_sha256="77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2",
        weights_size=409_264_008,
        weights_sha256="abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
        expected_parameters=102_310_592,
    ),
}

KRONOS_MAX_CONTEXT = 512
KRONOS_PRETRAINING_CUTOFF = "2024-06-30T23:59:59Z"


@dataclass(frozen=True)
class FoundationEngineReport:
    provider: str
    model_size: str
    model_artifact: dict[str, object]
    tokenizer_artifact: dict[str, object]
    source: dict[str, object]
    backend: dict[str, object]
    max_context: int
    model_parameters: int
    tokenizer_parameters: int
    package_versions: dict[str, str]
    rng_seed_control: str
    load_seconds: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_huggingface_artifact_file(
    path: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> Path:
    """Verify an immutable model file before any deserialization occurs."""

    target = Path(path)
    if not target.is_file():
        raise RuntimeError(f"{label} is missing: {target}")
    actual_size = target.stat().st_size
    if actual_size != int(expected_size):
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, received {actual_size}"
        )
    actual_sha256 = _file_sha256(target)
    if actual_sha256 != str(expected_sha256).lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, received {actual_sha256}"
        )
    return target


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _backend_payload(backend: BackendInfo) -> dict[str, object]:
    return {
        "requested": backend.requested,
        "kind": backend.kind,
        "device": backend.device,
        "vendor": backend.vendor,
        "reason": backend.reason,
    }


def _torch_device(backend: BackendInfo) -> Any:
    import torch

    if backend.kind == "directml":
        import torch_directml

        return torch_directml.device()
    return torch.device(backend.device)


def _verified_huggingface_snapshot(spec: PinnedHuggingFaceArtifact) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "foundation AI dependencies are missing; install simple-ai-trading[foundation-ai,gpu]"
        ) from exc

    config = Path(
        hf_hub_download(spec.repository, "config.json", revision=spec.revision)
    )
    weights = Path(
        hf_hub_download(spec.repository, "model.safetensors", revision=spec.revision)
    )
    verify_huggingface_artifact_file(
        config,
        expected_size=spec.config_size,
        expected_sha256=spec.config_sha256,
        label=f"{spec.repository} config",
    )
    verify_huggingface_artifact_file(
        weights,
        expected_size=spec.weights_size,
        expected_sha256=spec.weights_sha256,
        label=f"{spec.repository} weights",
    )
    if config.parent != weights.parent:
        raise RuntimeError(f"{spec.repository} snapshot files resolved to different revisions")
    return config.parent


def _import_kronos_classes(source_root: Path) -> tuple[Any, Any, Any]:
    expected_root = source_root.resolve()
    existing = sys.modules.get("model")
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if expected_root not in existing_file.parents:
            raise RuntimeError(
                "top-level Python module 'model' is already loaded from an unverified source"
            )
    source_text = str(expected_root)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("model")
    finally:
        if inserted:
            sys.path.remove(source_text)
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    if expected_root not in module_file.parents:
        raise RuntimeError("Kronos module resolved outside the verified source root")
    return module.Kronos, module.KronosPredictor, module.KronosTokenizer


class KronosForecastEngine:
    """Pinned Kronos inference engine for research feature generation only."""

    def __init__(
        self,
        predictor: Any,
        report: FoundationEngineReport,
        torch_module: Any,
        device_generator: Any | None = None,
    ) -> None:
        self._predictor = predictor
        self.report = report
        self._torch = torch_module
        self._device_generator = device_generator

    @classmethod
    def load(
        cls,
        *,
        model_size: str = "base",
        backend: str = "directml",
        source_cache_root: str | Path | None = None,
        bootstrap_source: bool = False,
        repair_source: bool = False,
        require_accelerator: bool = False,
    ) -> "KronosForecastEngine":
        size = str(model_size).strip().lower()
        if size not in KRONOS_MODEL_ARTIFACTS:
            raise ValueError(f"unsupported Kronos model size: {model_size!r}")
        started = time.perf_counter()
        source_report: FoundationSourceReport
        if bootstrap_source:
            source_report = provision_kronos_source(
                source_cache_root,
                repair=bool(repair_source),
            )
        else:
            source_report = verify_kronos_source(source_cache_root)

        resolved = resolve_backend(backend)
        if require_accelerator and resolved.kind == "cpu":
            raise RuntimeError(
                f"foundation-model accelerator is required but {backend!r} resolved to CPU: "
                f"{resolved.reason or 'no accelerator available'}"
            )
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "foundation AI dependencies are missing; install simple-ai-trading[foundation-ai,gpu]"
            ) from exc

        model_spec = KRONOS_MODEL_ARTIFACTS[size]
        tokenizer_snapshot = _verified_huggingface_snapshot(KRONOS_TOKENIZER_ARTIFACT)
        model_snapshot = _verified_huggingface_snapshot(model_spec)
        Kronos, KronosPredictor, KronosTokenizer = _import_kronos_classes(
            Path(source_report.source_root)
        )
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_snapshot)
        model = Kronos.from_pretrained(model_snapshot)
        tokenizer_parameters = sum(parameter.numel() for parameter in tokenizer.parameters())
        model_parameters = sum(parameter.numel() for parameter in model.parameters())
        if tokenizer_parameters != KRONOS_TOKENIZER_ARTIFACT.expected_parameters:
            raise RuntimeError(
                "Kronos tokenizer parameter-count mismatch: "
                f"expected {KRONOS_TOKENIZER_ARTIFACT.expected_parameters}, "
                f"received {tokenizer_parameters}"
            )
        if model_parameters != model_spec.expected_parameters:
            raise RuntimeError(
                f"Kronos-{size} parameter-count mismatch: expected "
                f"{model_spec.expected_parameters}, received {model_parameters}"
            )
        tokenizer.eval()
        model.eval()
        device = _torch_device(resolved)
        predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=KRONOS_MAX_CONTEXT,
        )
        device_generator = None
        rng_seed_control = "torch.manual_seed"
        if resolved.kind == "directml":
            import torch_directml

            device_generator = torch_directml.default_generator
            rng_seed_control = "torch.manual_seed + torch_directml.default_generator.manual_seed"
        report = FoundationEngineReport(
            provider="kronos",
            model_size=size,
            model_artifact=model_spec.asdict(),
            tokenizer_artifact=KRONOS_TOKENIZER_ARTIFACT.asdict(),
            source=source_report.asdict(),
            backend=_backend_payload(resolved),
            max_context=KRONOS_MAX_CONTEXT,
            model_parameters=model_parameters,
            tokenizer_parameters=tokenizer_parameters,
            package_versions={
                "einops": _package_version("einops"),
                "huggingface_hub": _package_version("huggingface_hub"),
                "pandas": _package_version("pandas"),
                "safetensors": _package_version("safetensors"),
                "torch": str(torch.__version__),
                "torch-directml": _package_version("torch-directml"),
            },
            rng_seed_control=rng_seed_control,
            load_seconds=float(time.perf_counter() - started),
        )
        return cls(predictor, report, torch, device_generator)

    def predict_batch(
        self,
        frames: Sequence[Any],
        history_timestamps: Sequence[Any],
        future_timestamps: Sequence[Any],
        *,
        prediction_length: int,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 10,
        seed: int = 17,
    ) -> tuple[Any, ...]:
        """Generate raw forecasts without converting them into order authority."""

        horizon = int(prediction_length)
        samples = int(sample_count)
        if not frames or len(frames) != len(history_timestamps) or len(frames) != len(future_timestamps):
            raise ValueError("forecast batch inputs must be non-empty and have equal lengths")
        if horizon < 1 or samples < 1:
            raise ValueError("prediction_length and sample_count must be positive")
        if horizon >= KRONOS_MAX_CONTEXT:
            raise ValueError("prediction_length exceeds the pinned Kronos context contract")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if int(top_k) < 0:
            raise ValueError("top_k must be non-negative")
        if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self._torch.manual_seed(max(0, int(seed)))
        if self._device_generator is not None:
            self._device_generator.manual_seed(max(0, int(seed)))
        predictions = self._predictor.predict_batch(
            list(frames),
            list(history_timestamps),
            list(future_timestamps),
            pred_len=horizon,
            T=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            sample_count=samples,
            verbose=False,
        )
        if len(predictions) != len(frames):
            raise RuntimeError("Kronos returned the wrong forecast batch length")
        for prediction in predictions:
            values = prediction.to_numpy(dtype="float64")
            if values.shape != (horizon, 6):
                raise RuntimeError(f"Kronos returned invalid forecast shape {values.shape}")
            if not all(math.isfinite(float(value)) for value in values.flat):
                raise RuntimeError("Kronos returned a non-finite forecast")
            if bool((prediction["high"] < prediction["low"]).any()):
                raise RuntimeError("Kronos returned a forecast with high below low")
        return tuple(predictions)


__all__ = [
    "FoundationEngineReport",
    "KRONOS_MAX_CONTEXT",
    "KRONOS_MODEL_ARTIFACTS",
    "KRONOS_PRETRAINING_CUTOFF",
    "KRONOS_TOKENIZER_ARTIFACT",
    "KronosForecastEngine",
    "PinnedHuggingFaceArtifact",
    "verify_huggingface_artifact_file",
]
