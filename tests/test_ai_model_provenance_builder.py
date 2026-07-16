from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.ai_model_benchmark import (
    _CONCEPT_ALIASES,
    benchmark_finance_ai_models,
    default_finance_ai_test_cases,
)
from tools import build_ai_model_provenance as builder


def _benchmark_payload() -> dict[str, object]:
    test_cases = default_finance_ai_test_cases()
    cases = iter(test_cases)
    rationale = " ".join(
        sorted(
            {term for case in test_cases for term in case.must_mention}
            | {aliases[0] for aliases in _CONCEPT_ALIASES.values()}
        )
    )

    def post(_url: str, _payload: object, _timeout: float) -> object:
        case = next(cases)
        response = {
            "action": case.expected_action,
            "risk_score": (case.min_risk_score + case.max_risk_score) / 2.0,
            "confidence": 0.99,
            "rationale": rationale,
            "concerns": [],
            "required_actions": [],
        }
        return {
            "message": {
                "content": json.dumps(response, separators=(",", ":")),
            }
        }

    report = benchmark_finance_ai_models(
        models=("qwen3:8b",),
        installed_models=("qwen3:8b",),
        post_json=post,
    )
    assert report.selected_model == "qwen3:8b"
    return report.asdict()


def _write_model_root(root: Path) -> tuple[dict[str, object], dict[Path, str]]:
    config = b"config"
    model = b"model-weights"
    config_sha = hashlib.sha256(config).hexdigest()
    model_sha = hashlib.sha256(model).hexdigest()
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    config_path = blobs / f"sha256-{config_sha}"
    model_path = blobs / f"sha256-{model_sha}"
    config_path.write_bytes(config)
    model_path.write_bytes(model)
    manifest = {
        "schemaVersion": 2,
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{config_sha}",
            "size": len(config),
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_sha}",
                "size": len(model),
            }
        ],
    }
    manifest_path = root / "manifests/registry.ollama.ai/library/qwen3/8b"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    show = {
        "modelfile": f"FROM {model_path}\n",
        "details": {
            "family": "qwen3",
            "parameter_size": "8.2B",
            "quantization_level": "Q4_K_M",
        },
        "capabilities": ["completion", "tools", "thinking"],
    }
    return show, {config_path: config_sha, model_path: model_sha}


def test_builder_hash_binds_reports_manifest_and_every_blob(tmp_path: Path) -> None:
    payload = _benchmark_payload()
    benchmark = tmp_path / "comparison.json"
    source = tmp_path / "qwen3-8b-source-v8.json"
    output = tmp_path / "model-provenance.json"
    serialized = json.dumps(payload, indent=2) + "\n"
    benchmark.write_text(serialized, encoding="utf-8")
    source.write_text(serialized, encoding="utf-8")
    model_root = tmp_path / "ollama-models"
    show, _blobs = _write_model_root(model_root)

    result = builder.build_provenance(
        benchmark_path=benchmark,
        source_report_paths=(source,),
        output_path=output,
        repository_root=tmp_path,
        model_root=model_root,
        show_model=lambda _model: show,
        observed_at=datetime(2026, 7, 16, tzinfo=UTC),
        minimum_model_bytes=0,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert (
        result["benchmark"]["sha256"]
        == hashlib.sha256(  # type: ignore[index]
            benchmark.read_bytes()
        ).hexdigest()
    )
    model = result["models"][0]  # type: ignore[index]
    assert model["base_blob_sha256"] == hashlib.sha256(b"model-weights").hexdigest()
    assert model["verified_blob_count"] == 2
    assert model["locally_verified"] is True


def test_builder_rejects_any_tampered_ollama_blob(tmp_path: Path) -> None:
    payload = _benchmark_payload()
    benchmark = tmp_path / "comparison.json"
    source = tmp_path / "source.json"
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    source.write_text(json.dumps(payload), encoding="utf-8")
    model_root = tmp_path / "ollama-models"
    show, blobs = _write_model_root(model_root)
    model_blob = next(path for path in blobs if path.read_bytes() == b"model-weights")
    model_blob.write_bytes(b"tampered-data")

    with pytest.raises(ValueError, match="size is invalid|hash mismatch"):
        builder.build_provenance(
            benchmark_path=benchmark,
            source_report_paths=(source,),
            output_path=tmp_path / "model-provenance.json",
            repository_root=tmp_path,
            model_root=model_root,
            show_model=lambda _model: show,
            minimum_model_bytes=0,
        )


@pytest.mark.parametrize(
    ("url", "timeout"),
    (
        ("https://127.0.0.1:11434", 10.0),
        ("http://example.com:11434", 10.0),
        ("http://localhost:11434/untrusted", 10.0),
        ("http://localhost:11434", float("inf")),
    ),
)
def test_builder_accepts_only_bounded_local_ollama_endpoints(
    url: str, timeout: float
) -> None:
    with pytest.raises(ValueError, match="only a local HTTP endpoint"):
        builder._ollama_show(url, timeout)


def test_builder_rejects_a_naive_evidence_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timestamp must include a timezone"):
        builder.build_provenance(
            benchmark_path=tmp_path / "missing.json",
            source_report_paths=(),
            output_path=tmp_path / "output.json",
            repository_root=tmp_path,
            model_root=tmp_path,
            show_model=lambda _model: {},
            observed_at=datetime(2026, 7, 16),
        )


def test_builder_rejects_tampered_selection_fields() -> None:
    payload = _benchmark_payload()
    payload["selected_model"] = None

    with pytest.raises(ValueError, match="selection fields are inconsistent"):
        builder._report_models(payload, "fixture")
