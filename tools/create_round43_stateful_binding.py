"""Bind the frozen Round 43 design, AI audit, source, and implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 43
BINDING_SCHEMA = "round-043-stateful-turnover-ai-factor-execution-binding-v2"
BOUND_PATHS = (
    "docs/model-research/action-value/round-043-ai-factor-audit.json",
    "docs/model-research/action-value/round-043-stateful-turnover-ai-factor-design.json",
    "src/simple_ai_trading/stateful_turnover_model.py",
    "tools/run_stateful_turnover_ai_factor_ablation.py",
    "tests/test_stateful_turnover_model.py",
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


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _validate_canonical(
    value: dict[str, object],
    *,
    field: str,
    expected_schema: str,
) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if value.get("schema_version") != expected_schema or claimed != _canonical_sha256(
        canonical
    ):
        raise ValueError(f"invalid canonical identity for {expected_schema}")
    return claimed


def create(arguments: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise ValueError("create the Round 43 binding only from a clean worktree")
    design_path = arguments.design.resolve()
    audit_path = arguments.ai_factor_audit.resolve()
    source_path = arguments.source_certificate.resolve()
    design = _read_object(design_path)
    audit = _read_object(audit_path)
    source = _read_object(source_path)
    design_sha = _validate_canonical(
        design,
        field="design_sha256",
        expected_schema="stateful-turnover-ai-factor-ablation-design-v2",
    )
    audit_sha = _validate_canonical(
        audit,
        field="audit_sha256",
        expected_schema="round-043-ai-factor-audit-v1",
    )
    source_sha = _validate_canonical(
        source,
        field="source_certificate_sha256",
        expected_schema="round-038-derivatives-source-certificate-v1",
    )
    if source_sha != "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39":
        raise ValueError("Round 43 source certificate differs from the frozen source")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in BOUND_PATHS
    ]
    binding: dict[str, object] = {
        "schema_version": BINDING_SCHEMA,
        "round": ROUND,
        "design_sha256": design_sha,
        "ai_factor_audit_sha256": audit_sha,
        "implementation_commit": implementation_commit,
        "source_certificate": {
            "path": str(source_path),
            "canonical_sha256": source_sha,
            "file_sha256": _file_sha256(source_path),
        },
        "blobs": blobs,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    output = arguments.output.resolve()
    write_json_atomic(output, binding, indent=2, sort_keys=True)
    return binding


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-043-stateful-turnover-ai-factor-design.json",
    )
    parser.add_argument(
        "--ai-factor-audit",
        type=Path,
        default=research / "round-043-ai-factor-audit.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-043-stateful-turnover-ai-factor-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(create(_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
