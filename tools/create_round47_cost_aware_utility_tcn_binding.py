"""Bind the frozen Round 47 design, verified cache, and implementation."""

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


ROUND = 47
DESIGN_SCHEMA = "cost-aware-utility-distributional-tcn-design-v1"
BINDING_SCHEMA = "round-047-cost-aware-utility-tcn-execution-binding-v1"
SOURCE_SCHEMA = "round-038-derivatives-source-certificate-v1"
PREDECESSOR_REPORT_SCHEMA = "stability-regularized-distributional-tcn-report-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
PREDECESSOR_REPORT_CANONICAL_SHA256 = (
    "7cd0bce1e797a77a89a389670677e1a3bce785d2018549ac168c4d753122076b"
)
CACHE_FILES = (
    "timestamps_ms.npy",
    "features.npy",
    "hourly_return_bps.npy",
    "forward_return_bps.npy",
    "metadata.json",
)
BOUND_PATHS = (
    "docs/model-research/action-value/round-047-cost-aware-utility-tcn-design.json",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/distributional_tcn_model.py",
    "src/simple_ai_trading/joint_distributional_tcn_model.py",
    "src/simple_ai_trading/stable_distributional_tcn_model.py",
    "src/simple_ai_trading/cost_aware_utility_tcn_model.py",
    "src/simple_ai_trading/storage.py",
    "tools/run_stability_regularized_tcn_viability.py",
    "tools/run_cost_aware_utility_tcn_viability.py",
    "tools/publish_cost_aware_utility_tcn_viability.py",
    "tools/create_round47_cost_aware_utility_tcn_binding.py",
    "tests/test_distributional_tcn_model.py",
    "tests/test_joint_distributional_tcn_model.py",
    "tests/test_stable_distributional_tcn_model.py",
    "tests/test_cost_aware_utility_tcn_model.py",
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


def _cache_manifest(
    cache_root: Path,
    predecessor_report: Mapping[str, object],
) -> list[dict[str, object]]:
    dataset = predecessor_report.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Round 46 predecessor report has no dataset section")
    reported = dataset.get("derived_cache_inputs")
    if not isinstance(reported, list):
        raise ValueError("Round 46 predecessor report has no cache manifest")
    by_name = {
        str(item.get("name")): item for item in reported if isinstance(item, Mapping)
    }
    manifest: list[dict[str, object]] = []
    for name in CACHE_FILES:
        path = (cache_root / name).resolve()
        item = by_name.get(name)
        if not path.is_file() or item is None:
            raise ValueError(f"Round 47 cache input is missing: {name}")
        sha256 = _file_sha256(path)
        size = path.stat().st_size
        if item.get("sha256") != sha256 or item.get("bytes") != size:
            raise ValueError(f"Round 47 cache differs from Round 46: {name}")
        manifest.append(
            {"name": name, "path": str(path), "bytes": size, "sha256": sha256}
        )
    return manifest


def create(arguments: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise ValueError("create the Round 47 binding only from a clean worktree")
    design_path = arguments.design.resolve()
    source_path = arguments.source_certificate.resolve()
    predecessor_path = arguments.predecessor_report.resolve()
    cache_root = arguments.derived_cache.resolve()
    design = _read_object(design_path)
    source = _read_object(source_path)
    predecessor = _read_object(predecessor_path)
    design_sha = _validate_canonical(
        design, field="design_sha256", expected_schema=DESIGN_SCHEMA
    )
    source_sha = _validate_canonical(
        source,
        field="source_certificate_sha256",
        expected_schema=SOURCE_SCHEMA,
    )
    predecessor_sha = _validate_canonical(
        predecessor,
        field="report_canonical_sha256",
        expected_schema=PREDECESSOR_REPORT_SCHEMA,
    )
    if source_sha != SOURCE_CANONICAL_SHA256:
        raise ValueError("Round 47 source certificate differs from the frozen source")
    if predecessor_sha != PREDECESSOR_REPORT_CANONICAL_SHA256:
        raise ValueError("Round 47 predecessor report differs from the frozen report")
    cache = _cache_manifest(cache_root, predecessor)
    data_contract = design.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("Round 47 design has no data contract")
    metadata = next(item for item in cache if item["name"] == "metadata.json")
    if (
        data_contract.get("source_certificate_canonical_sha256") != source_sha
        or data_contract.get("round_45_derived_cache_metadata_sha256")
        != metadata["sha256"]
    ):
        raise ValueError("Round 47 design and external lineage differ")
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
        "implementation_commit": implementation_commit,
        "source_certificate": {
            "path": str(source_path),
            "canonical_sha256": source_sha,
            "file_sha256": _file_sha256(source_path),
        },
        "predecessor_report": {
            "path": str(predecessor_path),
            "canonical_sha256": predecessor_sha,
            "file_sha256": _file_sha256(predecessor_path),
        },
        "derived_cache": cache,
        "blobs": blobs,
        "command": ".venv311\\Scripts\\python.exe tools\\run_cost_aware_utility_tcn_viability.py --source-certificate <external-certificate.json> --predecessor-report <round46-report.json> --derived-cache <round45-derived-cache> --evidence-root <new-external-evidence-root> --compute-backend directml",
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
        default=research / "round-047-cost-aware-utility-tcn-design.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--predecessor-report", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(create(_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
