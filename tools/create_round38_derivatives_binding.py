"""Create the immutable Round 38 derivatives hurdle execution binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
DESIGN = RESEARCH / "round-038-derivatives-hurdle-ai-ablation-design.json"
BINDING = RESEARCH / "round-038-derivatives-hurdle-ai-execution-binding.json"
SOURCE_CERTIFICATE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
BOUND_PATHS = (
    "docs/model-research/action-value/consumed-periods-through-round-037.json",
    "docs/model-research/action-value/round-037-failure-analysis.json",
    "docs/model-research/action-value/round-038-derivatives-hurdle-ai-ablation-design.json",
    "pyproject.toml",
    "src/simple_ai_trading/ai_trade_veto.py",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/derivatives_ai_veto.py",
    "src/simple_ai_trading/derivatives_archive.py",
    "src/simple_ai_trading/derivatives_hurdle_data.py",
    "src/simple_ai_trading/derivatives_hurdle_model.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/market_store.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_derivatives_archive.py",
    "tests/test_derivatives_hurdle_model.py",
    "tests/test_round38_derivatives_hurdle_design.py",
    "tools/create_round38_derivatives_binding.py",
    "tools/ingest_round38_derivatives_archives.py",
    "tools/run_derivatives_hurdle_ai_ablation.py",
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
        raise ValueError("Round 38 binding Git command failed") from exc


def create_binding(source_certificate: Path) -> dict[str, object]:
    """Bind the frozen design, implementation blobs, and real source certificate."""

    if _git("status", "--porcelain"):
        raise ValueError("Round 38 binding creation requires a clean worktree")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("Round 38 design root is not an object")
    design_canonical = dict(design)
    design_sha = str(design_canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version")
        != "derivatives-hurdle-ai-ablation-design-v2"
        or design.get("round") != 38
        or design_sha != _canonical_sha256(design_canonical)
    ):
        raise ValueError("Round 38 design identity is invalid")
    certificate = json.loads(source_certificate.read_text(encoding="utf-8"))
    if not isinstance(certificate, dict):
        raise ValueError("Round 38 source certificate root is not an object")
    certificate_canonical = dict(certificate)
    certificate_sha = str(
        certificate_canonical.pop("source_certificate_sha256", "")
    )
    if (
        certificate_sha != SOURCE_CERTIFICATE_SHA256
        or certificate_sha != _canonical_sha256(certificate_canonical)
        or certificate.get("design_sha256") != design_sha
    ):
        raise ValueError("Round 38 source certificate identity is invalid")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in BOUND_PATHS
    ]
    payload: dict[str, object] = {
        "schema_version": "round-038-derivatives-hurdle-ai-execution-binding-v1",
        "round": 38,
        "design_path": DESIGN.relative_to(ROOT).as_posix(),
        "design_sha256": design_sha,
        "design_file_sha256": _file_sha256(DESIGN),
        "source_certificate": {
            "path": "external://round38-derivatives-source-20260712-v2/certificate.json",
            "canonical_sha256": certificate_sha,
            "file_sha256": _file_sha256(source_certificate),
            "ingestion_implementation_commit": certificate[
                "implementation_commit"
            ],
        },
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "execution": {
            "command": (
                ".venv311\\Scripts\\python.exe "
                "tools\\run_derivatives_hurdle_ai_ablation.py "
                "--source-certificate <external-certificate.json> "
                "--evidence-root <new-external-evidence-root>"
            ),
            "database": "data/market_data.sqlite",
            "compute_backend": "auto_gpu_first_opencl",
            "candidate_count": 32,
            "model_artifact_count": 96,
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
