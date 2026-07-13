"""Bind the frozen Round 52 design to exact committed implementation blobs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "round-052-executable-support-hurdle-execution-binding-v1"
ROUND = 52
DESIGN_SHA256 = "af95d80a3adc21b72d6809d43afb3f2446213fe0a4e089b10366691465a0c669"
BOUND_PATHS = (
    "pyproject.toml",
    "docs/model-research/action-value/round-052-executable-support-hurdle-fincast-design.json",
    "src/simple_ai_trading/categorical_payoff_lightgbm.py",
    "src/simple_ai_trading/executable_payoff_lightgbm.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/microstructure_action_policy.py",
    "src/simple_ai_trading/microstructure_barriers.py",
    "src/simple_ai_trading/microstructure_cache.py",
    "src/simple_ai_trading/microstructure_features.py",
    "src/simple_ai_trading/microstructure_model.py",
    "src/simple_ai_trading/microstructure_warehouse.py",
    "src/simple_ai_trading/payoff_distribution_analysis.py",
    "src/simple_ai_trading/probability_calibration.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_executable_payoff_lightgbm.py",
    "tools/create_round52_execution_binding.py",
    "tools/run_round51_categorical_payoff_fincast.py",
    "tools/run_round52_executable_support_hurdle.py",
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


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _design_identity(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 52 design is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 52 design must be an object")
    canonical = dict(value)
    claimed = canonical.pop("design_sha256", None)
    if (
        value.get("round") != ROUND
        or value.get("status") != "frozen"
        or claimed != DESIGN_SHA256
        or _canonical_sha256(canonical) != claimed
    ):
        raise ValueError("Round 52 design identity is invalid")
    return str(claimed)


def create_binding(*, design_path: Path, output_path: Path) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise ValueError("Round 52 binding requires a clean committed worktree")
    commit = _git("rev-parse", "HEAD")
    design_sha = _design_identity(design_path)
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{commit}:{path}"),
        }
        for path in BOUND_PATHS
    ]
    binding: dict[str, object] = {
        "schema_version": SCHEMA,
        "round": ROUND,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
        "design_sha256": design_sha,
        "implementation_commit": commit,
        "blobs": blobs,
        "execution_contract": {
            "clean_worktree_required": True,
            "all_models_trained_before_evaluation": True,
            "support_alignment_required": True,
            "source_market_rows_synthetic": 0,
            "fincast_feature_rerun_required": False,
            "fincast_cached_artifact_hash_validation_required": True,
            "lightgbm_opencl_fp64_required": True,
            "lightgbm_compute_backend": "directml",
            "selection_contaminated": True,
            "profitability_claim_permitted": False,
            "trading_authority_permitted": False,
            "leverage_applied": False,
        },
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(binding, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-052-executable-support-hurdle-fincast-design.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-052-execution-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding = create_binding(
        design_path=args.design.resolve(),
        output_path=args.output.resolve(),
    )
    print(binding["binding_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
