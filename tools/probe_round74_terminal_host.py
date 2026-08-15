"""Publish read-only Round 74 compute and local-model preflight evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.ai_review import (  # noqa: E402
    resolve_ollama_model_provenance,
)
from simple_ai_trading.ai_runtime import (  # noqa: E402
    inspect_ollama_model_residency,
)
from simple_ai_trading.compute import (  # noqa: E402
    resolve_backend,
    torch_device_for_backend,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (  # noqa: E402
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.round74_segmented_terminal_runtime import (  # noqa: E402
    _preflight_ai_models,
    _resolve_ollama_runtime_version,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA_VERSION = "round-074-terminal-host-preflight-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def build_probe() -> dict[str, object]:
    backend = resolve_backend("auto", require=True)
    torch_device_for_backend(backend)
    bindings = round74_default_ai_review_model_panel()
    preflight_receipts = _preflight_ai_models(
        bindings,
        resolver=resolve_ollama_model_provenance,
        runtime_version_resolver=_resolve_ollama_runtime_version,
    )
    endpoints = {binding.runtime.endpoint.rstrip("/") for binding in bindings}
    if len(endpoints) != 1:
        raise ValueError("Round 74 terminal host endpoint panel differs")
    endpoint = next(iter(endpoints))
    runtime_version = _resolve_ollama_runtime_version(endpoint, 3.0)
    models = []
    for binding in bindings:
        artifact_sha256, metadata_sha256 = resolve_ollama_model_provenance(
            endpoint,
            binding.model_name,
            3.0,
        )
        if artifact_sha256 != binding.manifest.model_artifact_sha256:
            raise ValueError("Round 74 terminal host model digest differs")
        residency = inspect_ollama_model_residency(
            endpoint,
            binding.model_name,
            2.0,
            expected_digest=artifact_sha256,
        ).validated()
        model = {
            "role": binding.role,
            "model_name": binding.model_name,
            "model_manifest_sha256": binding.manifest.manifest_sha256,
            "model_artifact_sha256": artifact_sha256,
            "provider_metadata_sha256": metadata_sha256,
            "audited_runtime_floor": binding.manifest.runtime_version,
            "observed_runtime_version": runtime_version,
            "residency_status": residency.status,
            "loaded": residency.loaded,
            "gpu_resident": residency.gpu_resident,
        }
        model["model_probe_sha256"] = _sha256(model)
        models.append(model)
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": backend.device,
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
            "request_satisfied": backend.request_satisfied,
        },
        "ollama": {
            "endpoint": endpoint,
            "observed_runtime_version": runtime_version,
            "models": models,
            "terminal_preflight_receipt_sha256": list(preflight_receipts),
        },
        "capture_database_opened": False,
        "sealed_targets_read": False,
        "credentials_used": False,
        "orders_submitted": False,
        "model_inference_performed": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    value["artifact_sha256"] = _sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        print("Round 74 terminal host output already exists", file=sys.stderr)
        return 2
    try:
        value = build_probe()
        write_json_atomic(output, value, indent=2, sort_keys=True)
        restored = json.loads(output.read_text(encoding="utf-8"))
        claimed = restored.pop("artifact_sha256", "")
        if claimed != _sha256(restored):
            raise RuntimeError("Round 74 terminal host output digest differs")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json({"output": str(output), **value}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
