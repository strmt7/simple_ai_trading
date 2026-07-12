"""Create the immutable execution binding for the Round 37 experiment."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA = "round-037-cross-asset-ai-execution-binding-v1"
ROUND = 37
BOUND_PATHS = (
    "docs/model-research/action-value/consumed-periods-through-round-036.json",
    "docs/model-research/action-value/round-036-failure-analysis.json",
    "docs/model-research/action-value/round-037-cross-asset-cost-aware-ai-ablation-design.json",
    "src/simple_ai_trading/ai_runtime.py",
    "src/simple_ai_trading/ai_trade_veto.py",
    "src/simple_ai_trading/compute.py",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/cross_asset_cost_model.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/storage.py",
    "tools/create_round37_cross_asset_binding.py",
    "tools/run_cross_asset_cost_aware_ai_ablation.py",
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError("Round 37 binding Git command failed") from exc


def create_binding(design_path: Path, output_path: Path) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise ValueError("Round 37 binding requires a clean worktree")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("Round 37 design root is invalid")
    canonical_design = dict(design)
    design_sha = str(canonical_design.pop("design_sha256", ""))
    if design_sha != _canonical_sha256(canonical_design):
        raise ValueError("Round 37 design canonical hash is invalid")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = []
    for relative_path in BOUND_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        blobs.append(
            {
                "path": relative_path,
                "git_blob_oid": _git(
                    "rev-parse",
                    f"{implementation_commit}:{relative_path}",
                ),
                "working_file_sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    binding: dict[str, object] = {
        "schema_version": SCHEMA,
        "round": ROUND,
        "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
        "design_sha256": design_sha,
        "design_file_sha256": _file_sha256(design_path),
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "claims": {
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    write_json_atomic(output_path, binding, indent=2, sort_keys=True)
    return binding


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-037-cross-asset-cost-aware-ai-ablation-design.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-037-cross-asset-ai-execution-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = create_binding(args.design.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "binding_sha256": binding["binding_sha256"],
                "implementation_commit": binding["implementation_commit"],
                "blobs": len(binding["blobs"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
