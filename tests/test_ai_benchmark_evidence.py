from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from simple_ai_trading.ai_model_benchmark import (
    AI_MODEL_BENCHMARK_CONTRACT,
    _result_from_case_results,
    default_finance_ai_test_cases,
    rescore_finance_ai_benchmark_payload,
)
from simple_ai_trading.ai_model_provenance import load_local_ai_model_provenance


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "ai" / "risk-review" / "latest"
REPORT_PATH = LATEST / "comparison.json"
QWEN3_14B_PREREGISTRATION = (
    ROOT / "docs" / "ai" / "risk-review" / "qwen3-14b-v8-preregistration.json"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_qwen3_14b_preregistration_binds_source_and_case_suite() -> None:
    preregistration = _json(QWEN3_14B_PREREGISTRATION)
    source = ROOT / "src" / "simple_ai_trading" / "ai_model_benchmark.py"
    suite = [asdict(case) for case in default_finance_ai_test_cases()]

    assert preregistration["benchmark_contract"] == AI_MODEL_BENCHMARK_CONTRACT
    assert preregistration["benchmark_source_sha256"] == _sha256(source)
    assert preregistration["test_suite_sha256"] == _canonical_sha256(suite)
    assert preregistration["prior_comparison_sha256"] == _sha256(REPORT_PATH)
    predecessor = ROOT / preregistration["revoked_predecessor"]["path"]  # type: ignore[index]
    assert preregistration["revoked_predecessor"]["sha256"] == (  # type: ignore[index]
        _sha256(predecessor)
    )
    assert preregistration["frozen_run"]["run_count"] == 1  # type: ignore[index]
    assert preregistration["frozen_run"]["prompt_or_case_changes_allowed"] is False  # type: ignore[index]


def test_tracked_v7_ai_benchmark_is_historical_rejected_evidence() -> None:
    source_paths = (
        LATEST / "qwen3-8b-source-v7.json",
        LATEST / "fin-r1-8b-source-v7.json",
        LATEST / "qwen35-9b-source-v7.json",
        LATEST / "fino1-8b-source-v7.json",
    )
    tracked = _json(REPORT_PATH)

    assert tracked["benchmark_contract"] == "finance-risk-review-adversarial-v7"
    assert tracked["benchmark_contract"] != AI_MODEL_BENCHMARK_CONTRACT
    assert tracked["financial_edge_tested"] is False
    assert tracked["trading_authority"] is False
    assert tracked["selected_model"] is None
    results = {item["model"]: item for item in tracked["results"]}  # type: ignore[index]
    assert set(results) == {"qwen3:8b", "fin-r1:8b", "qwen3.5:9b", "fino1:8b"}
    assert all(item["passed"] is False for item in results.values())
    assert results["qwen3:8b"]["action_match_cases"] == 9
    assert all(
        results[model]["action_match_cases"] == 8
        for model in ("fin-r1:8b", "qwen3.5:9b", "fino1:8b")
    )
    assert all(
        len(case["model_input_sha256"]) == 64
        for result in results.values()
        for case in result["case_results"]
    )
    for path in source_paths:
        source = _json(path)
        assert source["benchmark_contract"] == "finance-risk-review-adversarial-v7"
        with pytest.raises(ValueError, match="fresh inference"):
            rescore_finance_ai_benchmark_payload(source)


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
    assert set(models) == {"qwen3:8b", "fin-r1:8b", "qwen3.5:9b", "fino1:8b"}
    for model in models.values():
        assert model["locally_verified"] is True
        assert int(model["size_bytes"]) > 2_000_000_000
        for field in ("ollama_manifest_digest", "base_blob_sha256"):
            value = str(model[field])
            assert len(value) == 64
            assert all(character in "0123456789abcdef" for character in value)
    for model in ("fin-r1:8b", "fino1:8b"):
        assert models[model]["conversion_status"] == "third_party_gguf_quantization"

    selected = load_local_ai_model_provenance(
        REPORT_PATH,
        REPORT_PATH.read_bytes(),
        model="qwen3:8b",
    )
    assert selected.benchmark_sha256 == _sha256(REPORT_PATH)
    assert selected.ollama_manifest_digest == models["qwen3:8b"][
        "ollama_manifest_digest"
    ]


def test_local_ai_model_provenance_rejects_unbound_benchmark_bytes() -> None:
    with pytest.raises(ValueError, match="does not bind"):
        load_local_ai_model_provenance(
            REPORT_PATH,
            REPORT_PATH.read_bytes() + b" ",
            model="qwen3:8b",
        )
