"""Bind the frozen Round 48 design, source certificate, and implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 48
DESIGN_SCHEMA = "minute-logistic-mixture-tcn-design-v1"
BINDING_SCHEMA = "round-048-minute-logistic-mixture-execution-binding-v1"
SOURCE_SCHEMA = "round-038-derivatives-source-certificate-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
BOUND_PATHS = (
    "docs/model-research/action-value/round-048-minute-logistic-mixture-tcn-design.json",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/derivatives_hurdle_data.py",
    "src/simple_ai_trading/distributional_tcn_model.py",
    "src/simple_ai_trading/minute_logistic_mixture_tcn_model.py",
    "src/simple_ai_trading/minute_logistic_mixture_analysis.py",
    "src/simple_ai_trading/storage.py",
    "tools/run_minute_logistic_mixture_tcn_viability.py",
    "tools/create_round48_minute_logistic_mixture_binding.py",
    "tests/test_minute_logistic_mixture_tcn_model.py",
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
        raise ValueError("create the Round 48 binding only from a clean worktree")
    design_path = arguments.design.resolve()
    source_path = arguments.source_certificate.resolve()
    design = _read_object(design_path)
    source = _read_object(source_path)
    design_sha = _validate_canonical(
        design,
        field="design_sha256",
        expected_schema=DESIGN_SCHEMA,
    )
    source_sha = _validate_canonical(
        source,
        field="source_certificate_sha256",
        expected_schema=SOURCE_SCHEMA,
    )
    if source_sha != SOURCE_CANONICAL_SHA256:
        raise ValueError("Round 48 source certificate differs from the frozen source")
    data_contract = design.get("data_contract")
    if not isinstance(data_contract, Mapping) or data_contract.get(
        "source_certificate_canonical_sha256"
    ) != source_sha:
        raise ValueError("Round 48 design and source certificate differ")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git(
                "rev-parse", f"{implementation_commit}:{path}"
            ),
        }
        for path in BOUND_PATHS
    ]
    binding: dict[str, object] = {
        "schema_version": BINDING_SCHEMA,
        "round": ROUND,
        "design_sha256": design_sha,
        "implementation_commit": implementation_commit,
        "source_certificate": {
            "path": str(source_path),
            "canonical_sha256": source_sha,
            "file_sha256": _file_sha256(source_path),
        },
        "blobs": blobs,
        "command": ".venv311\\Scripts\\python.exe tools\\run_minute_logistic_mixture_tcn_viability.py --database <market_data.sqlite> --source-certificate <external-certificate.json> --design docs\\model-research\\action-value\\round-048-minute-logistic-mixture-tcn-design.json --binding docs\\model-research\\action-value\\round-048-minute-logistic-mixture-tcn-binding.json --evidence-root <new-external-evidence-root> --compute-backend directml",
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
        default=research / "round-048-minute-logistic-mixture-tcn-design.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-048-minute-logistic-mixture-tcn-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(create(_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
