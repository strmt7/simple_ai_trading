from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.ai_model_benchmark import (
    AI_MODEL_BENCHMARK_CONTRACT,
    _result_from_case_results,
    merge_finance_ai_benchmark_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "ai" / "risk-review" / "latest"
REPORT_PATH = LATEST / "comparison.json"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_ai_benchmark_rebuilds_from_hash_bound_source_responses() -> None:
    source_paths = (
        LATEST / "qwen3-8b-source-v4.json",
        LATEST / "fino1-8b-source-v4.json",
    )
    sources = [_json(path) for path in source_paths]
    tracked = _json(REPORT_PATH)
    rebuilt = json.loads(
        json.dumps(merge_finance_ai_benchmark_payloads(sources).asdict())
    )
    rebuilt["generated_at_ms"] = tracked["generated_at_ms"]

    assert rebuilt == tracked
    assert tracked["benchmark_contract"] == AI_MODEL_BENCHMARK_CONTRACT
    assert tracked["financial_edge_tested"] is False
    assert tracked["trading_authority"] is False
    assert tracked["selected_model"] == "qwen3:8b"
    results = {item["model"]: item for item in tracked["results"]}  # type: ignore[index]
    assert results["qwen3:8b"]["passed"] is True
    assert results["fino1:8b"]["passed"] is False
    assert results["fino1:8b"]["failures"] == [
        "veto_liquidation_hidden_by_roi: missing_terms=leverage"
    ]


def test_aggregate_score_is_stable_across_python_float_sum_changes() -> None:
    scores = (
        0.9974999999999999,
        0.9824999999999999,
        0.9974999999999999,
        0.955,
        0.9974999999999999,
        0.9974999999999999,
        0.955,
        0.985,
        0.9974999999999999,
        0.955,
        0.9974999999999999,
    )
    case_results = tuple(
        {
            "name": f"case-{index}",
            "score": score,
            "valid_json": True,
            "action_match": True,
            "latency_seconds": 1.0,
            "failure": "",
        }
        for index, score in enumerate(scores)
    )

    result = _result_from_case_results(
        model="qwen3:8b",
        installed=True,
        case_results=case_results,
        minimum_score=0.78,
    )

    assert result.score == 0.983409090909091


def test_tracked_ai_model_provenance_binds_reports_and_weight_blobs() -> None:
    provenance = _json(LATEST / "model-provenance.json")
    assert provenance["benchmark"]["sha256"] == _sha256(REPORT_PATH)  # type: ignore[index]
    for source in provenance["source_reports"]:  # type: ignore[assignment]
        path = ROOT / source["path"]
        assert source["sha256"] == _sha256(path)
    models = {item["model"]: item for item in provenance["models"]}  # type: ignore[index]
    assert set(models) == {"qwen3:8b", "fino1:8b"}
    for model in models.values():
        assert model["locally_verified"] is True
        assert int(model["size_bytes"]) > 2_000_000_000
        for field in ("ollama_manifest_digest", "base_blob_sha256"):
            value = str(model[field])
            assert len(value) == 64
            assert all(character in "0123456789abcdef" for character in value)
    assert models["fino1:8b"]["conversion_status"] == "third_party_gguf_quantization"
