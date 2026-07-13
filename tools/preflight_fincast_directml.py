"""Run the pinned FinCast checkpoint through an evidence-producing backend preflight."""

from __future__ import annotations

import argparse
from collections import OrderedDict, namedtuple
import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import struct
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Mapping
import warnings

import torch
from torch.nn import functional as torch_functional


EXPECTED_SOURCE_COMMIT = "488b19d1d85fa2b3d4b93469530cefdcf1cc97a4"
EXPECTED_CHECKPOINT_SHA256 = (
    "d5ca999b02c944effa60d2b94174dc4d5a0cd2c0543ae289b2e36f37431492a8"
)
EXPECTED_CHECKPOINT_BYTES = 3_966_703_063
EXPECTED_PARAMETER_COUNT = 991_437_160
SCHEMA_VERSION = "fincast-directml-preflight-v1"
_FALLBACK_MARKERS = (
    "not currently supported on the dml backend",
    "fall back to run on the cpu",
    "falling back to cpu",
)


def _progress(phase: str, **details: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"fincast-preflight phase={phase}{' ' if suffix else ''}{suffix}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(source: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip().lower()


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_real_price_contexts(
    database: Path,
    *,
    symbols: tuple[str, ...],
    end_ms: int,
    context_length: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    if not database.is_file():
        raise FileNotFoundError(f"market database does not exist: {database}")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("FinCast input symbols must be unique and nonempty")
    query = """
        SELECT open_time, close, source
        FROM candles
        WHERE symbol = ? AND market_type = 'futures' AND interval = '1s'
          AND open_time <= ?
        ORDER BY open_time DESC
        LIMIT ?
    """
    series: list[list[float]] = []
    input_symbols: list[dict[str, object]] = []
    digest = hashlib.sha256()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0)
    try:
        for symbol in symbols:
            rows = list(
                reversed(
                    connection.execute(
                        query,
                        (symbol, int(end_ms), int(context_length)),
                    ).fetchall()
                )
            )
            if len(rows) != context_length:
                raise ValueError(
                    f"insufficient one-second FinCast context for {symbol}: {len(rows)}"
                )
            timestamps = [int(row[0]) for row in rows]
            prices = [float(row[1]) for row in rows]
            sources = sorted({str(row[2]) for row in rows})
            if (
                timestamps[-1] != int(end_ms)
                or any(
                    right - left != 1_000
                    for left, right in zip(timestamps, timestamps[1:])
                )
                or not all(math.isfinite(value) and value > 0.0 for value in prices)
                or sources != ["binance_public_archive_aggTrades"]
            ):
                raise ValueError(f"invalid real one-second FinCast context for {symbol}")
            digest.update(symbol.encode("ascii"))
            digest.update(b"\0")
            for timestamp, price in zip(timestamps, prices, strict=True):
                digest.update(struct.pack("<q", timestamp))
                digest.update(struct.pack("<d", price))
            series.append(prices)
            input_symbols.append(
                {
                    "symbol": symbol,
                    "rows": len(rows),
                    "first_timestamp_ms": timestamps[0],
                    "last_timestamp_ms": timestamps[-1],
                    "sources": sources,
                }
            )
    finally:
        connection.close()
    database_stat = database.stat()
    return torch.tensor(series, dtype=torch.float32), {
        "truth_basis": "checksummed Binance USD-M aggregate-trade one-second bars",
        "database": str(database.resolve()),
        "database_bytes": database_stat.st_size,
        "database_modified_ns": database_stat.st_mtime_ns,
        "query": "futures 1s closes ending at the explicit common timestamp",
        "symbols": input_symbols,
        "selected_rows_sha256": digest.hexdigest(),
    }


def _install_nondifferentiable_topk_shim() -> None:
    """Avoid FinCast's training-only CoLT dependency for deterministic inference."""

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
            raise RuntimeError("FinCast preflight forbids the training-only differentiable top-k")
        selected = torch.topk(values, k=k, dim=-1)
        return topk_result(selected.values, selected.indices, selected.values)

    shim = ModuleType("colt5_attention")
    shim.topk = topk  # type: ignore[attr-defined]
    sys.modules["colt5_attention"] = shim


def _load_decoder_module(source: Path) -> ModuleType:
    source_root = source / "src"
    decoder_path = source_root / "ffm" / "pytorch_patched_decoder_MOE.py"
    moe_path = source_root / "st_moe_pytorch" / "st_moe_pytorch.py"
    distributed_path = source_root / "st_moe_pytorch" / "distributed.py"
    for required in (decoder_path, moe_path, distributed_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing pinned FinCast source file: {required}")
    _install_nondifferentiable_topk_shim()
    sys.path.insert(0, str(source_root))
    module_name = "_simple_ai_trading_fincast_decoder"
    spec = importlib.util.spec_from_file_location(module_name, decoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create the FinCast decoder import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_state_dict(
    values: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    prefixes = ("_orig_mod.module.", "_orig_mod.", "module.")
    for name, value in values.items():
        normalized_name = name
        for prefix in prefixes:
            if normalized_name.startswith(prefix):
                normalized_name = normalized_name[len(prefix) :]
                break
        if normalized_name in normalized:
            raise ValueError(f"duplicate normalized checkpoint key: {normalized_name}")
        normalized[normalized_name] = value
    return normalized


def _materialize_nonpersistent_meta_buffers(model: torch.nn.Module) -> list[str]:
    """Restore the two inference-only MoE buffers omitted from state dictionaries."""

    materialized: list[str] = []
    meta_parameters = [
        name for name, value in model.named_parameters() if bool(value.is_meta)
    ]
    if meta_parameters:
        raise ValueError(f"checkpoint left meta parameters: {meta_parameters[:10]}")
    for module_name, module in model.named_modules():
        for buffer_name, value in tuple(module._buffers.items()):
            if value is None or not bool(value.is_meta):
                continue
            qualified_name = (
                f"{module_name}.{buffer_name}" if module_name else buffer_name
            )
            if buffer_name == "zero" and tuple(value.shape) == (1,):
                replacement = torch.zeros(value.shape, dtype=value.dtype)
            elif buffer_name == "dummy" and tuple(value.shape) == (1,):
                replacement = torch.ones(value.shape, dtype=value.dtype)
            else:
                raise ValueError(
                    f"unexpected non-persistent FinCast meta buffer: {qualified_name}"
                )
            module._buffers[buffer_name] = replacement
            materialized.append(qualified_name)
    remaining = [
        name for name, value in model.named_buffers() if bool(value.is_meta)
    ]
    if remaining:
        raise ValueError(f"FinCast left meta buffers: {remaining[:10]}")
    return materialized


def _working_set_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        if os.name != "nt":
            return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.WorkingSetSize) if succeeded else None


def _backend(name: str) -> tuple[object, dict[str, object]]:
    if name == "cpu":
        return torch.device("cpu"), {
            "kind": "cpu",
            "device": "cpu",
            "torch_directml_version": None,
        }
    if name != "directml":
        raise ValueError(f"unsupported backend: {name}")
    try:
        import torch_directml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torch-directml is unavailable") from exc
    device = torch_directml.device()
    return device, {
        "kind": "directml",
        "device": str(device),
        "torch_directml_version": str(
            getattr(torch_directml, "__version__", "unknown")
        ),
    }


def _install_directml_compatibility(backend: str) -> list[str]:
    """Replace only operators with mathematically identical DML-safe forms."""

    if backend != "directml":
        return []
    original_one_hot = torch_functional.one_hot

    def comparison_one_hot(
        tensor: torch.Tensor,
        num_classes: int = -1,
    ) -> torch.Tensor:
        if tensor.device.type != "privateuseone":
            return original_one_hot(tensor, num_classes=num_classes)
        if num_classes <= 0:
            raise ValueError("DirectML FinCast one-hot requires explicit classes")
        classes = torch.arange(
            num_classes,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return (tensor.unsqueeze(-1) == classes).to(torch.int64)

    torch_functional.one_hot = comparison_one_hot
    moe_module = sys.modules.get("st_moe_pytorch.st_moe_pytorch")
    if moe_module is None:
        raise RuntimeError("FinCast MoE module is unavailable for DirectML adaptation")

    def concatenated_exclusive_cumsum(
        tensor: torch.Tensor,
        dim: int = -3,
    ) -> torch.Tensor:
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

    moe_module.cumsum_exclusive = concatenated_exclusive_cumsum  # type: ignore[attr-defined]
    return [
        "one_hot_scatter_to_exact_index_comparison",
        "mixed_pad_exclusive_cumsum_to_exact_concat_cumsum",
    ]


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not source.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("FinCast source or checkpoint does not exist")

    _progress("verify_provenance")
    source_commit = _git_head(source)
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            f"FinCast source commit mismatch: {source_commit} != {EXPECTED_SOURCE_COMMIT}"
        )
    checkpoint_identity = _file_identity(checkpoint)
    if (
        checkpoint_identity["sha256"] != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_identity["bytes"] != EXPECTED_CHECKPOINT_BYTES
    ):
        raise ValueError("FinCast checkpoint identity mismatch")
    source_files = {
        name: _file_identity(source / relative)
        for name, relative in {
            "decoder": "src/ffm/pytorch_patched_decoder_MOE.py",
            "mixture_of_experts": "src/st_moe_pytorch/st_moe_pytorch.py",
            "distributed_helpers": "src/st_moe_pytorch/distributed.py",
            "license": "LICENSE",
        }.items()
    }

    _progress("inspect_checkpoint")
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("FinCast checkpoint is not a nonempty tensor mapping")
    state = _normalize_state_dict(state)
    parameter_count = sum(int(value.numel()) for value in state.values())
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in state.values()
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"FinCast parameter count mismatch: {parameter_count} != "
            f"{EXPECTED_PARAMETER_COUNT}"
        )

    _progress("import_pinned_architecture")
    decoder = _load_decoder_module(source)
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
        raise ValueError(f"FinCast state mismatch: {incompatible}")
    materialized_buffers = _materialize_nonpersistent_meta_buffers(model)
    materialized_buffer_evidence = {
        "count": len(materialized_buffers),
        "zero_buffers": sum(name.endswith(".zero") for name in materialized_buffers),
        "dummy_buffers": sum(
            name.endswith(".dummy") for name in materialized_buffers
        ),
        "names_sha256": hashlib.sha256(
            "\n".join(materialized_buffers).encode("ascii")
        ).hexdigest(),
    }

    device, backend = _backend(args.backend)
    compatibility_rewrites = _install_directml_compatibility(args.backend)
    rss_before_move = _working_set_bytes()
    _progress("move_model", backend=backend["kind"], device=backend["device"])
    move_started = time.perf_counter()
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = model.to(device)
    messages.extend(str(item.message) for item in caught)
    move_seconds = time.perf_counter() - move_started
    del state
    gc.collect()
    model.eval()
    rss_after_move = _working_set_bytes()

    context_length = int(args.context_length)
    if context_length < 32 or context_length % 32:
        raise ValueError("context length must be a positive multiple of 32")
    symbols = tuple(
        value.strip().upper() for value in str(args.symbols).split(",") if value.strip()
    )
    prices, input_evidence = _load_real_price_contexts(
        Path(args.database).resolve(),
        symbols=symbols,
        end_ms=int(args.input_end_ms),
        context_length=context_length,
    )
    batch_size = len(symbols)
    input_ts = prices.to(device)
    padding = torch.zeros(
        (batch_size, context_length), dtype=torch.float32
    ).to(device)
    frequency = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)

    timings: list[float] = []
    output_cpu: torch.Tensor | None = None
    for run_index in range(int(args.runs)):
        _progress("inference", run=run_index + 1, total=args.runs)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # DirectML tensors do not support inference-mode version counters.
            # no_grad preserves deterministic inference without that optimization.
            with torch.no_grad():
                output, auxiliary = model(input_ts, padding, frequency)
                output_cpu = output.detach().cpu()
                auxiliary_cpu = auxiliary.detach().cpu()
        messages.extend(str(item.message) for item in caught)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        if not torch.isfinite(output_cpu).all() or not torch.isfinite(
            auxiliary_cpu
        ).all():
            raise RuntimeError("FinCast inference produced nonfinite output")
        _progress("inference_complete", run=run_index + 1, seconds=f"{elapsed:.6f}")

    if output_cpu is None:
        raise RuntimeError("FinCast preflight did not execute inference")
    fallback_messages = sorted(
        {
            message
            for message in messages
            if any(marker in message.lower() for marker in _FALLBACK_MARKERS)
        }
    )
    if fallback_messages:
        raise RuntimeError(f"FinCast DirectML CPU fallback detected: {fallback_messages}")
    values = output_cpu.numpy()
    output_summary = {
        "shape": list(values.shape),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in output_summary.items()
        if key != "shape"
    ):
        raise RuntimeError("FinCast output summary is nonfinite")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trading_authority": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "source": {
            "repository": "https://github.com/vincent05r/FinCast-fts",
            "commit": source_commit,
            "files": source_files,
        },
        "checkpoint": checkpoint_identity,
        "input": input_evidence,
        "architecture": {
            "parameter_count": parameter_count,
            "tensor_bytes": tensor_bytes,
            "layers": 50,
            "hidden_size": 1280,
            "attention_heads": 16,
            "experts": 4,
            "experts_per_token": 2,
            "input_patch_length": 32,
            "output_patch_length": 128,
            "quantiles": list(config.quantiles),
            "materialized_nonpersistent_buffers": materialized_buffer_evidence,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "backend": backend,
            "context_length": context_length,
            "batch_size": batch_size,
            "runs": len(timings),
            "model_move_seconds": move_seconds,
            "inference_seconds": timings,
            "steady_state_seconds": timings[-1],
            "rss_before_model_move_bytes": rss_before_move,
            "rss_after_model_move_bytes": rss_after_move,
            "warning_count": len(messages),
            "cpu_fallback_warning_count": 0,
            "exact_backend_compatibility_rewrites": compatibility_rewrites,
        },
        "output": output_summary,
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("directml", "cpu"), default="directml")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--database", required=True)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--input-end-ms", type=int, required=True)
    parser.add_argument("--runs", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    try:
        report = run_preflight(args)
    except Exception as exc:  # noqa: BLE001 - failures are persisted as evidence
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "trading_authority": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["canonical_sha256"] = _canonical_sha256(report)
        _write_report(output, report)
        _progress("failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    _write_report(output, report)
    _progress("passed", report=output, canonical_sha256=report["canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
