"""Publish a truthful Round 27 AI host-qualification receipt."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
from typing import Any
from urllib import request as urllib_request

from simple_ai_trading.polymarket_round27_ai import (
    POLYMARKET_ROUND27_AI_HOST_PROBE_SCHEMA_VERSION,
    POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
    probe_round27_oda_host,
    probe_round27_qwen_host,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-027-ai-host-qualification-v1-2026-08-15.json"
)


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


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_json(url: str, timeout_seconds: float = 3.0) -> object:
    with urllib_request.urlopen(url, timeout=timeout_seconds) as response:  # nosec B310 - fixed loopback provider URL
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Round 27 AI inventory response exceeds its limit")
    return json.loads(raw.decode("utf-8"))


def _ollama_version() -> str:
    value = _get_json("http://127.0.0.1:11434/api/version")
    if not isinstance(value, dict) or set(value) != {"version"}:
        raise ValueError("Round 27 Ollama version response differs")
    version = value["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Round 27 Ollama version is invalid")
    return version


def _model_contract() -> dict[str, Any]:
    path = (
        ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-027-stage1-model-contract-v1.json"
    )
    value = json.loads(path.read_text(encoding="ascii"))
    if value.get("contract_sha256") != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256:
        raise ValueError("Round 27 model contract identity differs")
    return value


def build_report() -> dict[str, object]:
    contract = _model_contract()
    qwen = probe_round27_qwen_host()
    oda = probe_round27_oda_host()
    all_candidates_qualified = bool(qwen["passed"] and oda["passed"])
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_AI_HOST_PROBE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model_contract_sha256": POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "ollama": _ollama_version(),
        },
        "candidate_results": [qwen, oda],
        "qualification": {
            "qwen_host_runtime_qualified": qwen["passed"] is True,
            "oda_host_runtime_qualified": oda["passed"] is True,
            "all_preregistered_candidates_host_qualified": (
                all_candidates_qualified
            ),
            "matched_after_cost_ai_ablation_complete": False,
            "ai_promoted": False,
            "reason": (
                "Both candidates are eligible only for the later target-free "
                "matched ablation. No Stage 1 outcomes were accessed."
            ),
        },
        "data_authority": {
            "market_data_rows": 0,
            "targets": 0,
            "outcomes": 0,
            "resolutions": 0,
            "credentials_used": False,
            "orders_submitted": False,
            "trading_authority": False,
        },
        "limitations": [
            (
                "Host qualification measures artifact identity, structured-output "
                "conformance, latency, residency, and cleanup only; it does not "
                "measure intelligence, prediction, economic uplift, or edge."
            ),
            (
                "The ODA Q6_K artifact is a third-party quantization pinned by "
                "repository revision and file SHA-256. Its conversion from the "
                "official upstream weights was not independently reproduced."
            ),
            (
                "Candidates are qualified one at a time; concurrent residency "
                "and inference are outside this receipt."
            ),
        ],
        "model_contract_ai_assist": contract["ai_assist"],
        "sources": [
            {
                "title": "Ollama generate API",
                "url": "https://docs.ollama.com/api/generate",
            },
            {
                "title": "Ollama Windows GPU support",
                "url": "https://github.com/ollama/ollama/blob/main/docs/windows.mdx",
            },
            {
                "title": "Ollama GPU support",
                "url": "https://github.com/ollama/ollama/blob/main/docs/gpu.mdx",
            },
            {
                "title": "Qwen3.5-9B model card",
                "url": "https://huggingface.co/Qwen/Qwen3.5-9B",
            },
            {
                "title": "Qwen3.5 Ollama artifact",
                "url": "https://ollama.com/library/qwen3.5:9b",
            },
            {
                "title": "ODA-Fin-SFT-8B model card",
                "url": "https://huggingface.co/OpenDataArena/ODA-Fin-SFT-8B",
            },
            {
                "title": "ODA-Fin-SFT-8B GGUF quantization",
                "url": "https://huggingface.co/mradermacher/ODA-Fin-SFT-8B-GGUF",
            },
        ],
        "source_sha256": {
            "src/simple_ai_trading/polymarket_round27_ai.py": _file_sha256(
                ROOT / "src" / "simple_ai_trading" / "polymarket_round27_ai.py"
            ),
            "tools/probe_polymarket_round27_ai_host.py": _file_sha256(
                Path(__file__).resolve()
            ),
        },
    }
    report["evidence_sha256"] = _canonical_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(_canonical_json({
        "output": str(output),
        "evidence_sha256": report["evidence_sha256"],
        "qwen_host_runtime_qualified": report["qualification"][
            "qwen_host_runtime_qualified"
        ],
        "all_preregistered_candidates_host_qualified": report["qualification"][
            "all_preregistered_candidates_host_qualified"
        ],
        "ai_promoted": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
