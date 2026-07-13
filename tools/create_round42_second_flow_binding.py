"""Create the immutable Round 42 second-flow execution binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_causal_meta_label_capacity_ai import (  # noqa: E402
    SOURCE_CERTIFICATE_CANONICAL_SHA256,
    _canonical_sha256,
    _file_sha256,
    _git,
)
from tools.run_second_flow_execution_overlay import (  # noqa: E402
    ROUND41_REPORT_CANONICAL_SHA256,
)


RESEARCH = ROOT / "docs/model-research/action-value"
DESIGN = RESEARCH / "round-042-second-flow-execution-overlay-design.json"
BINDING = RESEARCH / "round-042-second-flow-execution-binding.json"
BOUND_PATHS = (
    "docs/model-research/action-value/round-042-second-flow-execution-overlay-design.json",
    "pyproject.toml",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/derivatives_hurdle_data.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/market_store.py",
    "src/simple_ai_trading/microstructure_architecture.py",
    "src/simple_ai_trading/second_flow_data.py",
    "src/simple_ai_trading/second_flow_execution_model.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_round42_second_flow_execution_overlay_design.py",
    "tests/test_second_flow_execution_model.py",
    "tools/certify_round42_second_flow_source.py",
    "tools/create_round42_second_flow_binding.py",
    "tools/run_causal_meta_label_capacity_ai.py",
    "tools/run_second_flow_execution_overlay.py",
)


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def create_binding(
    second_flow_certificate: Path,
    minute_source_certificate: Path,
    round41_report: Path,
) -> dict[str, object]:
    """Bind the frozen design, verified sources, predecessor, and code blobs."""

    if _git("status", "--porcelain"):
        raise ValueError("Round 42 binding creation requires a clean worktree")
    design = _read_object(DESIGN, "Round 42 design")
    canonical_design = dict(design)
    design_sha = str(canonical_design.pop("design_sha256", ""))
    if (
        design.get("schema_version") != "second-flow-execution-overlay-design-v1"
        or design.get("round") != 42
        or design_sha != _canonical_sha256(canonical_design)
    ):
        raise ValueError("Round 42 design identity is invalid")
    second = _read_object(
        second_flow_certificate, "Round 42 one-second source certificate"
    )
    canonical_second = dict(second)
    second_sha = str(canonical_second.pop("certificate_sha256", ""))
    if (
        second.get("schema_version") != "round-042-second-flow-source-certificate-v1"
        or second_sha != _canonical_sha256(canonical_second)
        or second.get("rows_total") != 1_814_400
        or second.get("raw_aggregate_trade_rows_total") != 15_475_296
    ):
        raise ValueError("Round 42 one-second source identity is invalid")
    minute = _read_object(
        minute_source_certificate, "Round 38 minute source certificate"
    )
    canonical_minute = dict(minute)
    minute_sha = str(canonical_minute.pop("source_certificate_sha256", ""))
    if (
        minute.get("round") != 38
        or minute_sha != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or minute_sha != _canonical_sha256(canonical_minute)
    ):
        raise ValueError("Round 42 minute source identity is invalid")
    predecessor = _read_object(round41_report, "Round 41 predecessor report")
    canonical_predecessor = dict(predecessor)
    predecessor_sha = str(canonical_predecessor.pop("report_canonical_sha256", ""))
    if (
        predecessor.get("schema_version") != "prequential-meta-label-ai-report-v1"
        or predecessor.get("round") != 41
        or predecessor_sha != ROUND41_REPORT_CANONICAL_SHA256
        or predecessor_sha != _canonical_sha256(canonical_predecessor)
    ):
        raise ValueError("Round 41 predecessor identity is invalid")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in BOUND_PATHS
    ]
    payload: dict[str, object] = {
        "schema_version": "round-042-second-flow-execution-binding-v1",
        "round": 42,
        "design_path": DESIGN.relative_to(ROOT).as_posix(),
        "design_sha256": design_sha,
        "design_file_sha256": _file_sha256(DESIGN),
        "source_certificates": {
            "second_flow": {
                "path": "external://round42-second-flow-source-20260713-v1/certificate.json",
                "canonical_sha256": second_sha,
                "file_sha256": _file_sha256(second_flow_certificate),
                "rows": second["rows_total"],
                "raw_aggregate_trade_rows": second["raw_aggregate_trade_rows_total"],
            },
            "minute_derivatives": {
                "path": "external://round38-derivatives-source-20260712-v2/certificate.json",
                "canonical_sha256": minute_sha,
                "file_sha256": _file_sha256(minute_source_certificate),
                "source_round": 38,
                "ingestion_implementation_commit": minute["implementation_commit"],
            },
        },
        "round41_predecessor": {
            "path": "external://round41-prequential-meta-label-20260713-v2/report.json",
            "canonical_sha256": predecessor_sha,
            "file_sha256": _file_sha256(round41_report),
            "status": predecessor["status"],
            "implementation_commit": predecessor["implementation_commit"],
        },
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "execution": {
            "command": (
                ".venv311\\Scripts\\python.exe "
                "tools\\run_second_flow_execution_overlay.py "
                "--second-flow-certificate <external-second-certificate.json> "
                "--minute-source-certificate <external-minute-certificate.json> "
                "--round41-report <external-round41-report.json> "
                "--round41-evidence-root <external-round41-evidence-root> "
                "--evidence-root <new-external-evidence-root>"
            ),
            "database": "data/market_data.sqlite",
            "database_read_only": True,
            "compute_backend": "auto_gpu_first_opencl",
            "models": 6,
            "threshold_cells": 54,
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "source_days": 7,
            "evaluation_days": 2,
            "entry_delays_seconds": [0, 5, 15, 30],
            "base_round_trip_charge_bps": 12.0,
            "stress_round_trip_charge_bps": 16.0,
            "ai_cases": 0,
        },
        "governance": {
            "clean_worktree_required": True,
            "implementation_must_be_ancestor_of_head": True,
            "bound_blob_identity_required": True,
            "selection_contaminated": True,
            "development_only": True,
            "data_expansion_only_if_pilot_gate_passes": True,
            "historical_data_expansion_before_gate_permitted": False,
            "promotion_permitted": False,
            "trading_authority_permitted": False,
            "risk_gate_relaxation_permitted": False,
            "leverage_permitted": False,
            "ai_inference_permitted": False,
        },
        "binding_sha256": "PENDING",
    }
    canonical = dict(payload)
    canonical.pop("binding_sha256")
    payload["binding_sha256"] = _canonical_sha256(canonical)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--second-flow-certificate", type=Path, required=True)
    parser.add_argument("--minute-source-certificate", type=Path, required=True)
    parser.add_argument("--round41-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BINDING)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = create_binding(
        arguments.second_flow_certificate.resolve(),
        arguments.minute_source_certificate.resolve(),
        arguments.round41_report.resolve(),
    )
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
