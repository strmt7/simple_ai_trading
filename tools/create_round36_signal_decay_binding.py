"""Create the one-time Git-blob binding for the Round 36 diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_multi_horizon_signal_decay import (  # noqa: E402
    BINDING_SCHEMA_VERSION,
    _REQUIRED_BOUND_PATHS,
    _canonical_sha256,
    _git_bytes,
    load_signal_decay_design,
)


def create_binding(*, design_path: Path, output_path: Path) -> dict[str, object]:
    """Bind committed implementation blobs and reject a dirty worktree."""

    design_path = design_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise ValueError("Round 36 execution binding already exists")
    if _git_bytes("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Round 36 binding creation requires a clean worktree")
    design, design_sha = load_signal_decay_design(design_path)
    commit = _git_bytes("rev-parse", "HEAD").decode("ascii").strip().lower()
    files: list[dict[str, str]] = []
    for relative in sorted(_REQUIRED_BOUND_PATHS):
        content = _git_bytes("show", f"{commit}:{relative}")
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    relative_design = design_path.relative_to(ROOT).as_posix()
    file_hashes = {item["path"]: item["sha256"] for item in files}
    payload: dict[str, object] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "round": 36,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding_sha256": "PENDING",
        "worktree_policy": "clean_including_untracked",
        "design": {
            "path": relative_design,
            "design_sha256": design_sha,
            "file_sha256": file_hashes[relative_design],
            "design_revision": design["design_revision"],
        },
        "implementation": {
            "commit": commit,
            "hash_mode": "git_blob_sha256_v1",
            "files": files,
        },
        "authority": {
            "post_hoc_consumed_data_diagnostic_only": True,
            "model_training_permitted": False,
            "model_candidate_permitted": False,
            "promotion_permitted": False,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
    }
    canonical = dict(payload)
    canonical.pop("binding_sha256")
    payload["binding_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(output_path, payload, indent=2, sort_keys=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(
        description="Create the immutable Round 36 signal-decay binding.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-036-multi-horizon-signal-decay-design.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-036-signal-decay-execution-binding.json",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    binding = create_binding(
        design_path=arguments.design,
        output_path=arguments.output,
    )
    print(binding["binding_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
