"""Probe one frozen Round 21 AI candidate without market targets or authority."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_ai_trading.polymarket_ai_veto import (  # noqa: E402
    PolymarketAIVetoCase,
    PolymarketAIVetoConfig,
    benchmark_polymarket_ai_veto,
    unload_polymarket_ai_model,
)
from simple_ai_trading.polymarket_round21_ai_selection import (  # noqa: E402
    POLYMARKET_ROUND21_AI_CANDIDATES,
    POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
)


SCHEMA_VERSION = "polymarket-round21-target-free-ai-candidate-host-probe-v2"
DEFAULT_SOURCE = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-qwen3.5-9b-target-free-host-probe-2026-08-03.json"
)
_SEMANTICS = {
    "capture_data_accessed": False,
    "live_trading_authority": False,
    "market_target_or_outcome_accessed": False,
    "model_selected": False,
    "paper_trading_authority": False,
    "predictive_edge_claim": False,
    "profitability_claim": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _load_source(path: Path) -> tuple[PolymarketAIVetoCase, str, str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise ValueError("Round 21 AI host-probe source is invalid")
    claimed = str(payload.get("artifact_sha256") or "")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if claimed != _canonical_sha256(body):
        raise ValueError("Round 21 AI host-probe source hash differs")
    if body.get("passed") is not True or body.get("semantics") != _SEMANTICS:
        raise ValueError("Round 21 AI host-probe source is not admissible")
    raw = body.get("case")
    if not isinstance(raw, Mapping):
        raise ValueError("Round 21 AI host-probe source case is missing")
    expected_keys = {
        "asset",
        "case_id",
        "case_sha256",
        "condition_id",
        "decision_received_monotonic_ns",
        "decision_received_wall_ms",
        "event_start_ms",
        "prompt_payload",
        "sample_id",
        "schema_version",
    }
    if set(raw) != expected_keys:
        raise ValueError("Round 21 AI host-probe source case fields differ")
    case = PolymarketAIVetoCase(
        case_id=str(raw["case_id"]),
        condition_id=str(raw["condition_id"]),
        sample_id=str(raw["sample_id"]),
        asset=str(raw["asset"]),
        event_start_ms=int(raw["event_start_ms"]),
        decision_received_wall_ms=int(raw["decision_received_wall_ms"]),
        decision_received_monotonic_ns=int(raw["decision_received_monotonic_ns"]),
        prompt_payload=dict(raw["prompt_payload"]),
        case_sha256=str(raw["case_sha256"]),
    )
    if case.case_sha256 != _canonical_sha256(case.identity_payload()):
        raise ValueError("Round 21 AI host-probe source case hash differs")
    risk = body.get("risk_benchmark_evidence")
    if not isinstance(risk, Mapping):
        raise ValueError("Round 21 AI host-probe risk evidence is missing")
    risk_sha256 = str(risk.get("sha256") or "")
    if len(risk_sha256) != 64:
        raise ValueError("Round 21 AI host-probe risk evidence hash is invalid")
    return case, claimed, risk_sha256


def _output_path(model: str) -> Path:
    safe_model = model.replace(":", "-").replace("/", "-")
    return (
        ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / f"round-021-{safe_model}-target-free-host-probe-2026-08-03.json"
    )


def run_probe(*, model: str, source_path: Path) -> dict[str, object]:
    if model not in POLYMARKET_ROUND21_AI_CANDIDATES:
        raise ValueError("AI model is outside the frozen Round 21 candidate set")
    case, source_sha256, risk_sha256 = _load_source(source_path)
    config = PolymarketAIVetoConfig(
        model=model,
        timeout_seconds=60.0,
        maximum_advisory_latency_seconds=30.0,
        minimum_approval_confidence=0.65,
        seed=4701,
    ).validated()
    try:
        report = benchmark_polymarket_ai_veto(
            cases=(case,),
            all_condition_ids=(case.condition_id,),
            selection_sha256=POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
            risk_benchmark_evidence_sha256=risk_sha256,
            config=config,
        )
    finally:
        unload_polymarket_ai_model(config)
    result = report.results[0] if len(report.results) == 1 else None
    runtime = None if result is None else result.provider_runtime
    checks = {
        "case_constraint_semantics_passed": bool(
            result is not None and result.decision.valid
        ),
        "exact_model_digest_present": len(report.model_digest) == 64,
        "gpu_residency_passed": bool(
            isinstance(runtime, Mapping)
            and runtime.get("status") == "gpu_resident"
            and runtime.get("vram_to_model_ratio") == 1.0
        ),
        "parameter_floor_passed": report.model_parameters_b >= 2.0,
        "provider_failure_count_passed": report.provider_failure_count == 0,
        "target_free_source_preserved": True,
        "valid_typed_response_count_passed": report.valid_response_count == 1,
    }
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "provider_schema_latency_and_gpu_compatibility_only",
        "model": model,
        "candidate_selection_design_sha256": (
            POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256
        ),
        "source_target_free_probe_sha256": source_sha256,
        "source_case_sha256": case.case_sha256,
        "checks": checks,
        "passed": all(checks.values()),
        "report": report.asdict(),
        "semantics": dict(_SEMANTICS),
    }
    return {**body, "artifact_sha256": _canonical_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=POLYMARKET_ROUND21_AI_CANDIDATES
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or _output_path(args.model)).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing probe: {output}")
    artifact = run_probe(model=args.model, source_path=args.source.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(_canonical_json({"output": str(output), **artifact["checks"]}))
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
