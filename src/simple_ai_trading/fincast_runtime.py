"""Pinned FinCast inference and causal numeric feature extraction."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import torch
from torch.nn import functional as torch_functional


FINCAST_SOURCE_COMMIT = "488b19d1d85fa2b3d4b93469530cefdcf1cc97a4"
FINCAST_CHECKPOINT_SHA256 = (
    "d5ca999b02c944effa60d2b94174dc4d5a0cd2c0543ae289b2e36f37431492a8"
)
FINCAST_CHECKPOINT_BYTES = 3_966_703_063
FINCAST_PARAMETER_COUNT = 991_437_160
FINCAST_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FINCAST_FEATURE_HORIZONS_SECONDS = (5, 15, 30, 60, 120)
FINCAST_CONTEXT_SECONDS = 512
FINCAST_RUNTIME_SCHEMA_VERSION = "pinned-fincast-runtime-v1"
_FALLBACK_MARKERS = (
    "not currently supported on the dml backend",
    "fall back to run on the cpu",
    "falling back to cpu",
)


def fincast_feature_names(
    horizons: Sequence[int] = FINCAST_FEATURE_HORIZONS_SECONDS,
) -> tuple[str, ...]:
    names: list[str] = []
    for horizon in horizons:
        prefix = f"fincast_{int(horizon)}s"
        names.extend(
            (
                f"{prefix}_point_return_bps",
                f"{prefix}_median_return_bps",
                f"{prefix}_q10_return_bps",
                f"{prefix}_q90_return_bps",
                f"{prefix}_central80_width_bps",
                f"{prefix}_quantile_asymmetry_bps",
            )
        )
    return tuple(names)


@dataclass(frozen=True)
class FinCastForecastBatch:
    point_forecast: np.ndarray
    quantile_forecast: np.ndarray
    raw_quantile_crossings: int
    inference_seconds: float
    warning_count: int
    cpu_fallback_warning_count: int

    def __post_init__(self) -> None:
        point = np.asarray(self.point_forecast, dtype=np.float32)
        quantiles = np.asarray(self.quantile_forecast, dtype=np.float32)
        if (
            point.ndim != 2
            or quantiles.shape != (*point.shape, len(FINCAST_QUANTILES))
            or point.shape[1] != 128
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(quantiles))
            or np.any(np.diff(quantiles, axis=2) < 0.0)
            or self.raw_quantile_crossings < 0
            or not math.isfinite(self.inference_seconds)
            or self.inference_seconds <= 0.0
            or self.warning_count < 0
            or self.cpu_fallback_warning_count != 0
        ):
            raise ValueError("FinCast forecast batch is invalid")


@dataclass(frozen=True)
class FinCastFeatureBatch:
    feature_names: tuple[str, ...]
    features: np.ndarray
    raw_quantile_crossings: int

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float32)
        if (
            self.feature_names != fincast_feature_names()
            or values.ndim != 2
            or values.shape[1] != len(self.feature_names)
            or not np.all(np.isfinite(values))
            or self.raw_quantile_crossings < 0
        ):
            raise ValueError("FinCast feature batch is invalid")


@dataclass(frozen=True)
class FinCastFeatureExtractionEvidence:
    rows: int
    context_seconds: int
    batches: int
    batch_size: int
    source_series_sha256: str
    decision_times_sha256: str
    features_sha256: str
    inference_seconds: float
    raw_quantile_crossings: int
    warning_count: int
    cpu_fallback_warning_count: int

    def __post_init__(self) -> None:
        hashes = (
            self.source_series_sha256,
            self.decision_times_sha256,
            self.features_sha256,
        )
        if (
            self.rows <= 0
            or self.context_seconds != FINCAST_CONTEXT_SECONDS
            or self.batches <= 0
            or not 1 <= self.batch_size <= 64
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or not math.isfinite(self.inference_seconds)
            or self.inference_seconds <= 0.0
            or self.raw_quantile_crossings < 0
            or self.warning_count < 0
            or self.cpu_fallback_warning_count != 0
        ):
            raise ValueError("FinCast feature extraction evidence is invalid")


@dataclass(frozen=True)
class FinCastRuntimeEvidence:
    schema_version: str
    source_path: str
    source_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    parameter_count: int
    backend_kind: str
    backend_device: str
    model_load_seconds: float
    warning_count: int
    cpu_fallback_warning_count: int


class _FunctionalProxy:
    def __init__(self, *, directml: bool) -> None:
        self._directml = bool(directml)

    def __getattr__(self, name: str) -> object:
        return getattr(torch_functional, name)

    def one_hot(
        self,
        tensor: torch.Tensor,
        num_classes: int = -1,
    ) -> torch.Tensor:
        if not self._directml or tensor.device.type != "privateuseone":
            return torch_functional.one_hot(tensor, num_classes=num_classes)
        if num_classes <= 0:
            raise ValueError("FinCast DirectML one-hot requires explicit classes")
        classes = torch.arange(
            num_classes,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return (tensor.unsqueeze(-1) == classes).to(torch.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_head(source: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip().lower()


def _install_topk_shim() -> None:
    topk_result = namedtuple(
        "FinCastTopKResult", ("values", "indices", "coor_descent_values")
    )

    def topk(
        values: torch.Tensor,
        *,
        k: int,
        non_differentiable: bool,
        fused: bool,
    ) -> object:
        del fused
        if not non_differentiable:
            raise RuntimeError(
                "FinCast runtime forbids training-only differentiable top-k"
            )
        selected = torch.topk(values, k=k, dim=-1)
        return topk_result(selected.values, selected.indices, selected.values)

    shim = ModuleType("colt5_attention")
    shim.topk = topk  # type: ignore[attr-defined]
    sys.modules["colt5_attention"] = shim


def _exclusive_cumsum(tensor: torch.Tensor, dim: int = -3) -> torch.Tensor:
    if dim >= 0:
        raise ValueError("FinCast exclusive cumulative-sum dimension must be negative")
    length = tensor.shape[dim]
    if length <= 0:
        return tensor
    if length == 1:
        return torch.zeros_like(tensor)
    prefix = torch.zeros_like(tensor.narrow(dim, 0, 1))
    shifted = torch.cat((prefix, tensor.narrow(dim, 0, length - 1)), dim=dim)
    return shifted.cumsum(dim=dim)


def _load_decoder(source: Path, *, directml: bool) -> ModuleType:
    source_root = source / "src"
    decoder_path = source_root / "ffm" / "pytorch_patched_decoder_MOE.py"
    required = (
        decoder_path,
        source_root / "st_moe_pytorch" / "st_moe_pytorch.py",
        source_root / "st_moe_pytorch" / "distributed.py",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("pinned FinCast source tree is incomplete")
    _install_topk_shim()
    for name in tuple(sys.modules):
        if name == "st_moe_pytorch" or name.startswith("st_moe_pytorch."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root))
    module_name = f"_simple_ai_trading_fincast_{FINCAST_SOURCE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, decoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the pinned FinCast decoder")
    decoder = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = decoder
    spec.loader.exec_module(decoder)
    moe_module = sys.modules.get("st_moe_pytorch.st_moe_pytorch")
    if moe_module is None:
        raise RuntimeError("FinCast MoE module did not load")
    moe_module.cumsum_exclusive = _exclusive_cumsum  # type: ignore[attr-defined]
    moe_module.F = _FunctionalProxy(directml=directml)  # type: ignore[attr-defined]
    return decoder


def _normalized_state_dict(
    values: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    prefixes = ("_orig_mod.module.", "_orig_mod.", "module.")
    for name, value in values.items():
        normalized = name
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        if normalized in output:
            raise ValueError(f"duplicate FinCast checkpoint key: {normalized}")
        output[normalized] = value
    return output


def _materialize_meta_buffers(model: torch.nn.Module) -> None:
    if any(value.is_meta for value in model.parameters()):
        raise ValueError("FinCast checkpoint left meta parameters")
    for _module_name, module in model.named_modules():
        for name, value in tuple(module._buffers.items()):
            if value is None or not value.is_meta:
                continue
            if name == "zero" and tuple(value.shape) == (1,):
                module._buffers[name] = torch.zeros(value.shape, dtype=value.dtype)
            elif name == "dummy" and tuple(value.shape) == (1,):
                module._buffers[name] = torch.ones(value.shape, dtype=value.dtype)
            else:
                raise ValueError(f"unexpected FinCast meta buffer: {name}")
    if any(value.is_meta for value in model.buffers()):
        raise ValueError("FinCast runtime left meta buffers")


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    return sorted(
        {
            message
            for message in messages
            if any(marker in message.lower() for marker in _FALLBACK_MARKERS)
        }
    )


class FinCastRuntime:
    """One lazily loaded, lock-serialized FinCast inference runtime."""

    def __init__(
        self,
        *,
        source: str | Path,
        checkpoint: str | Path,
        backend: str = "directml",
    ) -> None:
        self.source = Path(source).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.backend = str(backend).strip().lower()
        if self.backend not in {"directml", "cpu"}:
            raise ValueError("FinCast backend must be 'directml' or 'cpu'")
        self._model: torch.nn.Module | None = None
        self._device: object | None = None
        self._lock = threading.RLock()
        self._evidence: FinCastRuntimeEvidence | None = None

    @property
    def evidence(self) -> FinCastRuntimeEvidence:
        self.load()
        if self._evidence is None:
            raise RuntimeError("FinCast runtime evidence is unavailable")
        return self._evidence

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.source.is_dir() or not self.checkpoint.is_file():
                raise FileNotFoundError("FinCast source or checkpoint is missing")
            source_commit = _source_head(self.source)
            if source_commit != FINCAST_SOURCE_COMMIT:
                raise ValueError(
                    "FinCast source commit does not match the pinned contract"
                )
            if (
                self.checkpoint.stat().st_size != FINCAST_CHECKPOINT_BYTES
                or _sha256(self.checkpoint) != FINCAST_CHECKPOINT_SHA256
            ):
                raise ValueError("FinCast checkpoint identity does not match")
            directml = self.backend == "directml"
            decoder = _load_decoder(self.source, directml=directml)
            state = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(state, Mapping):
                raise ValueError("FinCast checkpoint is not a tensor mapping")
            state = _normalized_state_dict(state)
            parameter_count = sum(int(value.numel()) for value in state.values())
            if parameter_count != FINCAST_PARAMETER_COUNT:
                raise ValueError("FinCast checkpoint parameter count drifted")
            config = decoder.FFMConfig(
                num_layers=50,
                num_heads=16,
                num_kv_heads=16,
                hidden_size=1280,
                intermediate_size=1280,
                head_dim=80,
                patch_len=32,
                horizon_len=128,
                num_experts=4,
                gating_top_n=2,
                threshold_train=0.2,
                threshold_eval=0.2,
                use_positional_embedding=False,
            )
            with torch.device("meta"):
                model = decoder.PatchedTimeSeriesDecoder_MOE(config)
            incompatible = model.load_state_dict(state, strict=True, assign=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    "FinCast checkpoint keys do not match the architecture"
                )
            _materialize_meta_buffers(model)
            if directml:
                try:
                    import torch_directml  # type: ignore
                except ImportError as exc:
                    raise RuntimeError("torch-directml is unavailable") from exc
                device = torch_directml.device()
                directml_version = str(
                    getattr(torch_directml, "__version__", "unknown")
                )
            else:
                device = torch.device("cpu")
                directml_version = "not_applicable"
            started = time.perf_counter()
            messages: list[str] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = model.to(device)
            messages.extend(str(item.message) for item in caught)
            load_seconds = time.perf_counter() - started
            fallback = _fallback_messages(messages)
            if fallback:
                raise RuntimeError(f"FinCast backend fallback detected: {fallback}")
            del state
            gc.collect()
            model.eval()
            self._model = model
            self._device = device
            self._evidence = FinCastRuntimeEvidence(
                schema_version=FINCAST_RUNTIME_SCHEMA_VERSION,
                source_path=str(self.source),
                source_commit=source_commit,
                checkpoint_path=str(self.checkpoint),
                checkpoint_sha256=FINCAST_CHECKPOINT_SHA256,
                parameter_count=parameter_count,
                backend_kind=self.backend,
                backend_device=str(device),
                model_load_seconds=load_seconds,
                warning_count=len(messages),
                cpu_fallback_warning_count=0,
            )
            if directml_version == "":
                raise RuntimeError("FinCast DirectML version evidence is empty")

    def forecast(self, contexts: np.ndarray) -> FinCastForecastBatch:
        values = np.asarray(contexts, dtype=np.float32)
        if (
            values.ndim != 2
            or not 1 <= values.shape[0] <= 64
            or values.shape[1] < 32
            or values.shape[1] % 32
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("FinCast contexts are invalid")
        with self._lock:
            self.load()
            if self._model is None or self._device is None:
                raise RuntimeError("FinCast runtime did not load")
            input_tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
                self._device
            )
            padding = torch.zeros(values.shape, dtype=torch.float32).to(self._device)
            frequency = torch.zeros((values.shape[0], 1), dtype=torch.int64).to(
                self._device
            )
            messages: list[str] = []
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.no_grad():
                    output, auxiliary = self._model(
                        input_tensor,
                        padding,
                        frequency,
                    )
                    selected = output[:, -1, :, :].detach().cpu().numpy()
                    auxiliary_cpu = auxiliary.detach().cpu().numpy()
            elapsed = time.perf_counter() - started
            messages.extend(str(item.message) for item in caught)
        fallback = _fallback_messages(messages)
        if fallback:
            raise RuntimeError(f"FinCast inference fallback detected: {fallback}")
        if not np.all(np.isfinite(selected)) or not np.all(np.isfinite(auxiliary_cpu)):
            raise RuntimeError("FinCast inference produced nonfinite values")
        raw_quantiles = np.asarray(selected[:, :, 1:], dtype=np.float32)
        crossings = int(np.sum(np.diff(raw_quantiles, axis=2) < 0.0))
        rearranged = np.sort(raw_quantiles, axis=2)
        return FinCastForecastBatch(
            point_forecast=np.asarray(selected[:, :, 0], dtype=np.float32),
            quantile_forecast=rearranged,
            raw_quantile_crossings=crossings,
            inference_seconds=elapsed,
            warning_count=len(messages),
            cpu_fallback_warning_count=0,
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._device = None
            self._evidence = None
            gc.collect()

    def __enter__(self) -> FinCastRuntime:
        self.load()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()


def derive_fincast_features(
    contexts: np.ndarray,
    forecast: FinCastForecastBatch,
) -> FinCastFeatureBatch:
    values = np.asarray(contexts, dtype=np.float64)
    point = np.asarray(forecast.point_forecast, dtype=np.float64)
    quantiles = np.asarray(forecast.quantile_forecast, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != point.shape[0]
        or point.shape[1] != 128
        or quantiles.shape != (*point.shape, len(FINCAST_QUANTILES))
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("FinCast feature inputs are invalid")
    anchor = values[:, -1]
    columns: list[np.ndarray] = []
    for horizon in FINCAST_FEATURE_HORIZONS_SECONDS:
        index = horizon - 1
        point_return = 10_000.0 * (point[:, index] / anchor - 1.0)
        q10 = 10_000.0 * (quantiles[:, index, 0] / anchor - 1.0)
        q50 = 10_000.0 * (quantiles[:, index, 4] / anchor - 1.0)
        q90 = 10_000.0 * (quantiles[:, index, 8] / anchor - 1.0)
        width = q90 - q10
        asymmetry = q90 + q10 - 2.0 * q50
        columns.extend((point_return, q50, q10, q90, width, asymmetry))
    features = np.column_stack(columns).astype(np.float32)
    return FinCastFeatureBatch(
        feature_names=fincast_feature_names(),
        features=features,
        raw_quantile_crossings=forecast.raw_quantile_crossings,
    )


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def extract_fincast_feature_matrix(
    runtime: FinCastRuntime,
    *,
    second_ms: np.ndarray,
    close_mid: np.ndarray,
    decision_time_ms: np.ndarray,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[FinCastFeatureBatch, FinCastFeatureExtractionEvidence]:
    """Extract causal features without materializing overlapping contexts."""

    seconds = np.asarray(second_ms, dtype=np.int64)
    prices = np.asarray(close_mid, dtype=np.float32)
    decisions = np.asarray(decision_time_ms, dtype=np.int64)
    selected_batch_size = int(batch_size)
    if (
        seconds.ndim != 1
        or prices.shape != seconds.shape
        or len(seconds) < FINCAST_CONTEXT_SECONDS
        or decisions.ndim != 1
        or len(decisions) == 0
        or np.any(np.diff(seconds) <= 0)
        or np.any(seconds % 1_000 != 0)
        or np.any(np.diff(decisions) <= 0)
        or not np.all(np.isfinite(prices))
        or np.any(prices <= 0.0)
        or not 1 <= selected_batch_size <= 64
    ):
        raise ValueError("FinCast causal feature source is invalid")
    anchors = decisions - 1_000
    anchor_indexes = np.searchsorted(seconds, anchors)
    if np.any(anchor_indexes >= len(seconds)) or not np.array_equal(
        seconds[anchor_indexes], anchors
    ):
        raise ValueError("FinCast decisions lack an exact preceding one-second close")
    start_indexes = anchor_indexes - (FINCAST_CONTEXT_SECONDS - 1)
    if np.any(start_indexes < 0):
        raise ValueError("FinCast decisions lack the complete context length")
    expected_span_ms = (FINCAST_CONTEXT_SECONDS - 1) * 1_000
    if np.any(seconds[anchor_indexes] - seconds[start_indexes] != expected_span_ms):
        raise ValueError("FinCast contexts contain one or more missing seconds")

    columns = len(fincast_feature_names())
    features = np.empty((len(decisions), columns), dtype=np.float32)
    offsets = np.arange(FINCAST_CONTEXT_SECONDS, dtype=np.int64)
    batches = math.ceil(len(decisions) / selected_batch_size)
    inference_seconds = 0.0
    raw_crossings = 0
    warning_count = 0
    for batch_index, first in enumerate(
        range(0, len(decisions), selected_batch_size), start=1
    ):
        last = min(first + selected_batch_size, len(decisions))
        context_indexes = start_indexes[first:last, None] + offsets[None, :]
        contexts = np.ascontiguousarray(prices[context_indexes], dtype=np.float32)
        forecast = runtime.forecast(contexts)
        derived = derive_fincast_features(contexts, forecast)
        features[first:last] = derived.features
        inference_seconds += forecast.inference_seconds
        raw_crossings += forecast.raw_quantile_crossings
        warning_count += forecast.warning_count
        if progress is not None:
            progress(batch_index, batches)
    output = FinCastFeatureBatch(
        feature_names=fincast_feature_names(),
        features=features,
        raw_quantile_crossings=raw_crossings,
    )
    evidence = FinCastFeatureExtractionEvidence(
        rows=len(decisions),
        context_seconds=FINCAST_CONTEXT_SECONDS,
        batches=batches,
        batch_size=selected_batch_size,
        source_series_sha256=_array_sha256(seconds, prices),
        decision_times_sha256=_array_sha256(decisions),
        features_sha256=_array_sha256(features),
        inference_seconds=inference_seconds,
        raw_quantile_crossings=raw_crossings,
        warning_count=warning_count,
        cpu_fallback_warning_count=0,
    )
    return output, evidence


def fincast_runtime_contract_sha256() -> str:
    payload = {
        "schema_version": FINCAST_RUNTIME_SCHEMA_VERSION,
        "source_commit": FINCAST_SOURCE_COMMIT,
        "checkpoint_sha256": FINCAST_CHECKPOINT_SHA256,
        "checkpoint_bytes": FINCAST_CHECKPOINT_BYTES,
        "parameter_count": FINCAST_PARAMETER_COUNT,
        "quantiles": FINCAST_QUANTILES,
        "feature_horizons_seconds": FINCAST_FEATURE_HORIZONS_SECONDS,
        "context_seconds": FINCAST_CONTEXT_SECONDS,
        "feature_names": fincast_feature_names(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
