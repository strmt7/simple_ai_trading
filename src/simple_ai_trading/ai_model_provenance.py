"""Hash-bound local Ollama model provenance for AI governance evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .ai_benchmark_claim import (
    requires_preregistered_ai_benchmark,
    validate_preregistered_ai_runtime_evidence,
)

_SCHEMA_VERSION = "ollama-local-model-provenance-v2"
_LEGACY_SCHEMA_VERSION = "ollama-local-model-provenance-v1"


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


@dataclass(frozen=True)
class LocalAIModelProvenance:
    path: str
    provenance_sha256: str
    benchmark_sha256: str
    benchmark_contract: str
    model: str
    ollama_manifest_digest: str
    base_blob_sha256: str
    size_bytes: int
    inference_runtime_evidence_sha256: str | None = None
    inference_model_metadata_sha256: str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def load_local_ai_model_provenance(
    benchmark_path: str | Path,
    benchmark_bytes: bytes,
    *,
    model: str,
) -> LocalAIModelProvenance:
    """Bind a benchmark file to one locally verified manifest and weight blob."""

    path = Path(benchmark_path)
    provenance_path = path.with_name("model-provenance.json")
    provenance_bytes = provenance_path.read_bytes()
    try:
        benchmark_payload = json.loads(benchmark_bytes.decode("utf-8"))
        payload = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("local AI model provenance JSON is invalid") from exc
    benchmark = _mapping(benchmark_payload, "AI benchmark")
    provenance = _mapping(payload, "AI model provenance")
    binding = _mapping(provenance.get("benchmark"), "AI benchmark binding")
    expected_benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    models_raw = provenance.get("models")
    if (
        provenance.get("schema_version")
        not in {_SCHEMA_VERSION, _LEGACY_SCHEMA_VERSION}
        or binding.get("sha256") != expected_benchmark_sha256
        or binding.get("contract") != benchmark.get("benchmark_contract")
        or not isinstance(models_raw, Sequence)
        or isinstance(models_raw, (str, bytes))
    ):
        raise ValueError("local AI model provenance does not bind the benchmark")
    models = [_mapping(item, "AI model provenance row") for item in models_raw]
    names = [str(item.get("model") or "") for item in models]
    if not names or "" in names or len(set(names)) != len(names):
        raise ValueError("local AI model provenance models are invalid")
    selected = next((item for item in models if item.get("model") == model), None)
    if selected is None:
        raise ValueError(f"local AI model provenance is missing: {model}")
    manifest_digest = str(selected.get("ollama_manifest_digest") or "").lower()
    base_blob_sha256 = str(selected.get("base_blob_sha256") or "").lower()
    raw_size = selected.get("size_bytes")
    if isinstance(raw_size, bool):
        raise ValueError("local AI model provenance size is invalid")
    try:
        size_bytes = int(raw_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("local AI model provenance size is invalid") from exc
    if (
        selected.get("locally_verified") is not True
        or not _is_sha256(manifest_digest)
        or not _is_sha256(base_blob_sha256)
        or size_bytes <= 2_000_000_000
    ):
        raise ValueError("local AI model provenance weight evidence is invalid")
    schema_version = str(provenance["schema_version"])
    runtime_evidence_sha256 = None
    inference_model_metadata_sha256 = None
    runtime_evidence = selected.get("inference_runtime_evidence")
    if schema_version == _LEGACY_SCHEMA_VERSION:
        if requires_preregistered_ai_benchmark(model):
            raise ValueError(
                "legacy local AI provenance cannot authorize a preregistered model"
            )
    elif runtime_evidence is None:
        if "inference_runtime_evidence" not in selected:
            raise ValueError("local AI model provenance lacks runtime evidence status")
        if requires_preregistered_ai_benchmark(model):
            raise ValueError(
                "local AI model provenance lacks required inference-time evidence"
            )
    else:
        tests = benchmark.get("tests")
        if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)):
            raise ValueError("AI benchmark test inventory is invalid")
        validated_runtime = validate_preregistered_ai_runtime_evidence(
            runtime_evidence,
            model=model,
            case_count=len(tests),
            base_url=str(benchmark.get("base_url") or ""),
        )
        pre_inference = _mapping(
            validated_runtime.get("pre_inference"),
            "AI benchmark pre-inference provenance",
        )
        if pre_inference.get("model_digest") != manifest_digest:
            raise ValueError(
                "local AI model manifest differs from inference-time evidence"
            )
        inference_model_metadata_sha256 = str(pre_inference["model_metadata_sha256"])
        runtime_evidence_sha256 = hashlib.sha256(
            json.dumps(
                validated_runtime,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    return LocalAIModelProvenance(
        path=provenance_path.as_posix(),
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        benchmark_sha256=expected_benchmark_sha256,
        benchmark_contract=str(binding["contract"]),
        model=model,
        ollama_manifest_digest=manifest_digest,
        base_blob_sha256=base_blob_sha256,
        size_bytes=size_bytes,
        inference_runtime_evidence_sha256=runtime_evidence_sha256,
        inference_model_metadata_sha256=inference_model_metadata_sha256,
    )


__all__ = ["LocalAIModelProvenance", "load_local_ai_model_provenance"]
