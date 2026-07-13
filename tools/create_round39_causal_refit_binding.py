"""Create the immutable Round 39 rolling-refit and AI execution binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
DESIGN = RESEARCH / "round-039-causal-refit-utility-ai-ablation-design.json"
BINDING = RESEARCH / "round-039-causal-refit-utility-ai-execution-binding.json"
SOURCE_CERTIFICATE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
BOUND_PATHS = (
    "docs/model-research/action-value/consumed-periods-through-round-038.json",
    "docs/model-research/action-value/round-038-failure-analysis.json",
    "docs/model-research/action-value/round-039-causal-refit-utility-ai-ablation-design.json",
    "pyproject.toml",
    "src/simple_ai_trading/ai_trade_veto.py",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/derivatives_hurdle_data.py",
    "src/simple_ai_trading/derivatives_hurdle_model.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/market_store.py",
    "src/simple_ai_trading/rolling_refit_ai_veto.py",
    "src/simple_ai_trading/rolling_refit_model.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_rolling_refit_ai_veto.py",
    "tests/test_rolling_refit_model.py",
    "tests/test_round39_causal_refit_design.py",
    "tests/test_round39_execution_binding.py",
    "tools/create_round39_causal_refit_binding.py",
    "tools/run_causal_refit_utility_ai_ablation.py",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 39 binding Git command failed") from exc


def create_binding(source_certificate: Path) -> dict[str, object]:
    """Bind the frozen design, implementation blobs, and source certificate."""

    if _git("status", "--porcelain"):
        raise ValueError("Round 39 binding creation requires a clean worktree")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("Round 39 design root is not an object")
    canonical_design = dict(design)
    design_sha = str(canonical_design.pop("design_sha256", ""))
    if (
        design.get("schema_version")
        != "causal-refit-utility-ai-ablation-design-v3"
        or design.get("round") != 39
        or design_sha != _canonical_sha256(canonical_design)
    ):
        raise ValueError("Round 39 design identity is invalid")
    certificate = json.loads(source_certificate.read_text(encoding="utf-8"))
    if not isinstance(certificate, dict):
        raise ValueError("Round 39 source certificate root is not an object")
    canonical_certificate = dict(certificate)
    certificate_sha = str(
        canonical_certificate.pop("source_certificate_sha256", "")
    )
    if (
        certificate_sha != SOURCE_CERTIFICATE_SHA256
        or certificate_sha != _canonical_sha256(canonical_certificate)
        or certificate.get("round") != 38
    ):
        raise ValueError("Round 39 reused source certificate identity is invalid")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in BOUND_PATHS
    ]
    payload: dict[str, object] = {
        "schema_version": "round-039-causal-refit-utility-ai-execution-binding-v1",
        "round": 39,
        "design_path": DESIGN.relative_to(ROOT).as_posix(),
        "design_sha256": design_sha,
        "design_file_sha256": _file_sha256(DESIGN),
        "source_certificate": {
            "path": "external://round38-derivatives-source-20260712-v2/certificate.json",
            "canonical_sha256": certificate_sha,
            "file_sha256": _file_sha256(source_certificate),
            "source_round": 38,
            "ingestion_implementation_commit": certificate[
                "implementation_commit"
            ],
        },
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "execution": {
            "command": (
                ".venv311\\Scripts\\python.exe "
                "tools\\run_causal_refit_utility_ai_ablation.py "
                "--source-certificate <external-certificate.json> "
                "--evidence-root <new-external-evidence-root>"
            ),
            "database": "data/market_data.sqlite",
            "compute_backend": "auto_gpu_first_opencl",
            "candidate_count": 4,
            "monthly_refits": 6,
            "model_artifact_count": 60,
            "ai_models": list(("qwen3:8b", "fino1:8b")),
            "ai_batch_size": 12,
            "source_period_end": "2025-06-30",
        },
        "governance": {
            "clean_worktree_required": True,
            "implementation_must_be_ancestor_of_head": True,
            "bound_blob_identity_required": True,
            "selection_confirmation_access_permitted": False,
            "terminal_2026_access_permitted": False,
            "promotion_permitted": False,
            "trading_authority_permitted": False,
            "risk_gate_relaxation_permitted": False,
            "leverage_permitted": False,
        },
        "binding_sha256": "PENDING",
    }
    canonical = dict(payload)
    canonical.pop("binding_sha256")
    payload["binding_sha256"] = _canonical_sha256(canonical)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BINDING)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = create_binding(arguments.source_certificate.resolve())
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
