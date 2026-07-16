"""Build hash-verified local Ollama provenance for an AI benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from simple_ai_trading.ai_model_benchmark import (
    AI_MODEL_BENCHMARK_CONTRACT,
    rescore_finance_ai_benchmark_payload,
)
from simple_ai_trading.storage import write_json_atomic


_SCHEMA_VERSION = "ollama-local-model-provenance-v1"
_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_SHOW_BYTES = 4 * 1024 * 1024
ShowModel = Callable[[str], Mapping[str, object]]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    if not path.is_file() or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"JSON object has an invalid size: {path.name}")
    payload = path.read_bytes()
    if not 1 <= len(payload) <= _MAX_JSON_BYTES:
        raise ValueError(f"JSON object has an invalid size: {path.name}")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path.name}")
    return value, payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


def _sha256_file(
    path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    label: str = "blob",
) -> str:
    digest = hashlib.sha256()
    processed = 0
    next_progress = 1024 * 1024 * 1024
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            processed += len(chunk)
            if progress is not None and processed >= next_progress:
                progress(f"verified {label}: {processed} bytes")
                next_progress += 1024 * 1024 * 1024
    return digest.hexdigest()


def _artifact_path(path: Path, repository_root: Path) -> str:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "AI provenance artifacts must remain inside the repository"
        ) from exc


def _model_manifest_path(model: str, model_root: Path) -> Path:
    reference = str(model or "").strip()
    if not reference or "@" in reference:
        raise ValueError("Ollama model reference is invalid")
    name, separator, tag = reference.rpartition(":")
    if not separator or "/" in tag:
        name, tag = reference, "latest"
    parts = name.split("/")
    if any(not part or _SEGMENT.fullmatch(part) is None for part in (*parts, tag)):
        raise ValueError("Ollama model reference contains an invalid path segment")
    if len(parts) == 1:
        manifest_parts = ("registry.ollama.ai", "library", parts[0], tag)
    elif "." in parts[0] or parts[0] == "localhost":
        manifest_parts = (*parts, tag)
    else:
        manifest_parts = ("registry.ollama.ai", *parts, tag)
    root = (model_root / "manifests").resolve()
    path = root.joinpath(*manifest_parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Ollama manifest path escaped the model root") from exc
    return path


def _digest_and_size(row: Mapping[str, object], name: str) -> tuple[str, int]:
    digest_value = str(row.get("digest") or "")
    if not digest_value.startswith("sha256:"):
        raise ValueError(f"{name} has a non-SHA256 digest")
    digest = digest_value.removeprefix("sha256:").lower()
    raw_size = row.get("size")
    if (
        _SHA256.fullmatch(digest) is None
        or not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
    ):
        raise ValueError(f"{name} digest or size is invalid")
    size = raw_size
    if size < 1:
        raise ValueError(f"{name} size is invalid")
    return digest, size


def _verified_model_row(
    model: str,
    *,
    model_root: Path,
    show_model: ShowModel,
    progress: Callable[[str], None] | None,
    minimum_model_bytes: int,
) -> dict[str, object]:
    manifest_path = _model_manifest_path(model, model_root)
    manifest, manifest_bytes = _read_json_object(manifest_path)
    config = _mapping(manifest.get("config"), "Ollama manifest config")
    layers_raw = manifest.get("layers")
    if (
        manifest.get("schemaVersion") != 2
        or not isinstance(layers_raw, Sequence)
        or isinstance(layers_raw, (str, bytes))
        or not layers_raw
    ):
        raise ValueError("Ollama manifest schema is invalid")
    layers = [_mapping(item, "Ollama manifest layer") for item in layers_raw]
    rows = [
        ("config", config),
        *[(f"layer[{index}]", row) for index, row in enumerate(layers)],
    ]
    verified: list[tuple[str, int, str]] = []
    blob_root = (model_root / "blobs").resolve()
    for name, row in rows:
        digest, expected_size = _digest_and_size(row, name)
        blob = (blob_root / f"sha256-{digest}").resolve()
        try:
            blob.relative_to(blob_root)
        except ValueError as exc:
            raise ValueError("Ollama blob path escaped the model root") from exc
        if progress is not None:
            progress(f"verifying {model} {name} ({expected_size} bytes)")
        if not blob.is_file() or blob.stat().st_size != expected_size:
            raise ValueError(f"Ollama {name} blob size is invalid")
        if _sha256_file(blob, progress=progress, label=f"{model} {name}") != digest:
            raise ValueError(f"Ollama {name} blob hash mismatch")
        verified.append((digest, expected_size, str(row.get("mediaType") or "")))
    model_layers = [
        item for item in verified[1:] if item[2] == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1:
        raise ValueError("Ollama manifest must contain exactly one model layer")
    base_blob_sha256 = model_layers[0][0]
    show = _mapping(show_model(model), "Ollama show response")
    details = _mapping(show.get("details"), "Ollama show details")
    modelfile_value = show.get("modelfile")
    if not isinstance(modelfile_value, str):
        raise ValueError("Ollama show Modelfile is invalid")
    modelfile = modelfile_value
    from_digests = re.findall(r"(?im)^FROM\s+.*sha256-([0-9a-f]{64})\s*$", modelfile)
    if from_digests != [base_blob_sha256]:
        raise ValueError("Ollama show response does not bind the model blob")
    capabilities_raw = show.get("capabilities")
    if not isinstance(capabilities_raw, Sequence) or isinstance(
        capabilities_raw, (str, bytes)
    ):
        raise ValueError("Ollama model capabilities are invalid")
    capabilities = [str(item) for item in capabilities_raw]
    if (
        not capabilities
        or len(capabilities) > 16
        or len(set(capabilities)) != len(capabilities)
        or any(
            not isinstance(item, str) or not item or len(item) > 64
            for item in capabilities_raw
        )
    ):
        raise ValueError("Ollama model capabilities are invalid")
    total_size = sum(item[1] for item in verified)
    family_value = details.get("family")
    parameter_size_value = details.get("parameter_size")
    quantization_value = details.get("quantization_level")
    if not all(
        isinstance(item, str)
        for item in (family_value, parameter_size_value, quantization_value)
    ):
        raise ValueError("Ollama model details are invalid")
    family = str(family_value)
    parameter_size = str(parameter_size_value)
    quantization = str(quantization_value)
    if (
        total_size <= minimum_model_bytes
        or not family
        or not parameter_size
        or not quantization
        or max(len(family), len(parameter_size), len(quantization)) > 64
        or "completion" not in capabilities
    ):
        raise ValueError("Ollama model does not satisfy local AI provenance policy")
    return {
        "model": model,
        "source_reference": f"ollama:{model}",
        "ollama_manifest_digest": hashlib.sha256(manifest_bytes).hexdigest(),
        "base_blob_sha256": base_blob_sha256,
        "size_bytes": total_size,
        "family": family,
        "parameter_size": parameter_size,
        "quantization": quantization,
        "capabilities": capabilities,
        "verified_blob_count": len(verified),
        "locally_verified": True,
    }


def _report_models(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    report = rescore_finance_ai_benchmark_payload(payload)
    if report.benchmark_contract != AI_MODEL_BENCHMARK_CONTRACT:
        raise ValueError(f"{name} uses an obsolete AI benchmark contract")
    if any(not result.installed for result in report.results):
        raise ValueError(f"{name} contains a model not installed during inference")
    if (
        payload.get("selected_model") != report.selected_model
        or payload.get("passed") is not report.passed
    ):
        raise ValueError(f"{name} selection fields are inconsistent")
    return tuple(result.model for result in report.results)


def build_provenance(
    *,
    benchmark_path: Path,
    source_report_paths: Sequence[Path],
    output_path: Path,
    repository_root: Path,
    model_root: Path,
    show_model: ShowModel,
    observed_at: datetime | None = None,
    progress: Callable[[str], None] | None = None,
    minimum_model_bytes: int = 2_000_000_000,
) -> dict[str, object]:
    """Verify benchmark reports and every local Ollama blob, then write provenance."""

    if (
        not isinstance(minimum_model_bytes, int)
        or isinstance(minimum_model_bytes, bool)
        or minimum_model_bytes < 0
    ):
        raise ValueError("minimum AI model size is invalid")
    if observed_at is not None and (
        observed_at.tzinfo is None or observed_at.utcoffset() is None
    ):
        raise ValueError("AI provenance timestamp must include a timezone")
    benchmark, benchmark_bytes = _read_json_object(benchmark_path)
    benchmark_models = _report_models(benchmark, "AI benchmark")
    selected_model = str(benchmark.get("selected_model") or "") or None
    if (
        not isinstance(source_report_paths, Sequence)
        or isinstance(source_report_paths, (str, bytes))
        or not source_report_paths
    ):
        raise ValueError("at least one AI source report is required")
    source_rows: list[dict[str, object]] = []
    source_models: list[str] = []
    for path in source_report_paths:
        source, source_bytes = _read_json_object(path)
        models = _report_models(source, f"AI source report {path.name}")
        if len(models) != 1:
            raise ValueError("each AI source report must contain exactly one model")
        source_models.extend(models)
        source_rows.append(
            {
                "model": models[0],
                "path": _artifact_path(path, repository_root),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    if (
        len(set(source_models)) != len(source_models)
        or set(source_models) != set(benchmark_models)
        or (selected_model is not None and selected_model not in source_models)
    ):
        raise ValueError("AI source reports do not exactly cover benchmark models")
    model_rows = [
        _verified_model_row(
            model,
            model_root=model_root,
            show_model=show_model,
            progress=progress,
            minimum_model_bytes=minimum_model_bytes,
        )
        for model in source_models
    ]
    timestamp = (observed_at or datetime.now(UTC)).astimezone(UTC)
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
        "benchmark": {
            "contract": AI_MODEL_BENCHMARK_CONTRACT,
            "path": _artifact_path(benchmark_path, repository_root),
            "sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        },
        "source_reports": source_rows,
        "models": model_rows,
        "limitations": [
            "Local manifest and blob identity do not prove financial edge.",
            "Governance selection does not grant order or trading authority.",
            "A selected model still requires paired same-period after-cost uplift evidence.",
        ],
    }
    write_json_atomic(output_path, payload, sort_keys=False)
    return payload


def _ollama_show(base_url: str, timeout_seconds: float) -> ShowModel:
    parsed = urllib.parse.urlparse(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not math.isfinite(timeout_seconds)
        or not 0.1 <= timeout_seconds <= 60.0
    ):
        raise ValueError("Ollama provenance accepts only a local HTTP endpoint")
    endpoint = f"{base_url.rstrip('/')}/api/show"

    def show(model: str) -> Mapping[str, object]:
        body = json.dumps({"model": model, "verbose": False}).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
                response_bytes = response.read(_MAX_SHOW_BYTES + 1)
                if len(response_bytes) > _MAX_SHOW_BYTES:
                    raise ValueError("Ollama show response exceeded its size limit")
                value = json.loads(
                    response_bytes.decode("utf-8"),
                    object_pairs_hook=_strict_object,
                )
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
            raise ValueError(f"Ollama show failed for {model}") from exc
        return _mapping(value, "Ollama show response")

    return show


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama/models")),
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    output = args.output or args.benchmark.with_name("model-provenance.json")
    try:
        build_provenance(
            benchmark_path=args.benchmark,
            source_report_paths=args.source_report,
            output_path=output,
            repository_root=args.repository_root,
            model_root=args.model_root,
            show_model=_ollama_show(args.ollama_url, args.timeout),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"AI model provenance failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
