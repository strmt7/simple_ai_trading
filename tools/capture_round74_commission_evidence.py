"""Capture account-specific Round 74 commission evidence without persisting keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import requests


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round74_commission_capture import (  # noqa: E402
    capture_round74_mainnet_commission,
)
from tools._round74_public_evidence_capture import (  # noqa: E402
    canonical_sha256,
    git_commit,
    require_clean_tracked_worktree,
    write_artifact,
)


ROUND74_COMMISSION_ARTIFACT_SCHEMA_VERSION = "round-074-commission-artifact-v1"
ROUND74_COMMISSION_API_KEY_ENVIRONMENT_VARIABLE = (
    "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_KEY"
)
ROUND74_COMMISSION_API_SECRET_ENVIRONMENT_VARIABLE = (
    "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_SECRET"
)


def _normalized_source_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture signed mainnet commission GET responses for BTC, ETH, and "
            "SOL without persisting credential material."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    require_clean_tracked_worktree()
    api_key = os.environ.get(
        ROUND74_COMMISSION_API_KEY_ENVIRONMENT_VARIABLE,
        "",
    )
    api_secret = os.environ.get(
        ROUND74_COMMISSION_API_SECRET_ENVIRONMENT_VARIABLE,
        "",
    )
    if not api_key or not api_secret:
        raise RuntimeError(
            "Round 74 commission credentials are absent from the required "
            "ephemeral environment variables"
        )
    with requests.Session() as session:
        result = capture_round74_mainnet_commission(
            api_key=api_key,
            api_secret=api_secret,
            timeout_seconds=arguments.timeout_seconds,
            request=session.request,
        )
    artifact: dict[str, object] = {
        "schema_version": ROUND74_COMMISSION_ARTIFACT_SCHEMA_VERSION,
        "capture": result.as_dict(),
        "execution_git_commit": git_commit(),
    }
    artifact["source"] = {
        "capture_module": ("src/simple_ai_trading/round74_commission_capture.py"),
        "capture_module_sha256": _normalized_source_sha256(
            SOURCE / "simple_ai_trading" / "round74_commission_capture.py"
        ),
        "capture_tool": "tools/capture_round74_commission_evidence.py",
        "capture_tool_sha256": _normalized_source_sha256(Path(__file__)),
    }
    artifact["credential_transport"] = {
        "source": "process_environment_only",
        "api_key_variable": ROUND74_COMMISSION_API_KEY_ENVIRONMENT_VARIABLE,
        "api_secret_variable": (ROUND74_COMMISSION_API_SECRET_ENVIRONMENT_VARIABLE),
        "credential_values_persisted": False,
    }
    artifact["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )
    write_artifact(arguments.output, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "capture_sha256": result.capture_sha256,
                "symbols": list(result.bundle.as_mapping()),
                "total_signed_request_weight": 60,
                "orders_submitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
