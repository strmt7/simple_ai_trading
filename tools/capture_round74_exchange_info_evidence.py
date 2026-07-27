"""Capture one bounded public Binance exchange-info evidence artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_exchange_info_evidence import (  # noqa: E402
    build_round74_quantity_rules_evidence,
)
from simple_ai_trading.impact_absorption_event_targets import (  # noqa: E402
    round74_quantity_rules_evidence_claims,
)
from tools._round74_public_evidence_capture import (  # noqa: E402
    bounded_json_get,
    canonical_sha256 as _canonical_sha256,
    git_commit as _git_commit,
    require_clean_tracked_worktree as _require_clean_tracked_worktree,
    write_artifact as _write_artifact,
)


ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION = (
    "round-074-exchange-info-capture-v1"
)
ROUND74_EXCHANGE_INFO_URL = (
    "https://fapi.binance.com/fapi/v1/exchangeInfo"
)
ROUND74_EXCHANGE_INFO_MAXIMUM_RESPONSE_BYTES = 8 * 1024 * 1024


def capture_round74_exchange_info(
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    """Perform one request, validate it, and return a compact artifact."""

    timeout = float(timeout_seconds)
    if not 1.0 <= timeout <= 60.0:
        raise ValueError("Round 74 exchange-info timeout differs")
    _require_clean_tracked_worktree()
    execution_commit = _git_commit()
    response = bounded_json_get(
        url=ROUND74_EXCHANGE_INFO_URL,
        timeout_seconds=timeout,
        maximum_response_bytes=(
            ROUND74_EXCHANGE_INFO_MAXIMUM_RESPONSE_BYTES
        ),
        user_agent="simple-ai-trading-round74-research/1",
    )
    payload = response.payload
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 exchange-info response root differs")
    bundle = build_round74_quantity_rules_evidence(
        payload=payload,
        environment="binance_usdm_mainnet",
        observed_wall_ns=response.received_wall_ns,
    )
    rules = bundle.as_mapping()
    artifact: dict[str, object] = {
        "schema_version": ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION,
        "artifact_sha256": "",
        "captured_at_utc": datetime.fromtimestamp(
            response.received_wall_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "execution_git_commit": execution_commit,
        "request": {
            "method": "GET",
            "url": ROUND74_EXCHANGE_INFO_URL,
            "security_type": "NONE",
            "request_weight": 1,
            "timeout_seconds": timeout,
            "retry_count": 0,
            "credential_material_sent": False,
            "request_started_wall_ns": response.request_started_wall_ns,
            "request_started_monotonic_ns": (
                response.request_started_monotonic_ns
            ),
        },
        "response": {
            "status": 200,
            "received_wall_ns": response.received_wall_ns,
            "received_monotonic_ns": response.received_monotonic_ns,
            "elapsed_monotonic_ns": response.elapsed_monotonic_ns,
            "body_bytes": len(response.body),
            "headers": response.header_mapping(),
            "raw_payload_persisted": False,
        },
        "source_binding": {
            "parser_path": (
                "src/simple_ai_trading/"
                "impact_absorption_exchange_info_evidence.py"
            ),
            "parser_sha256": hashlib.sha256(
                (
                    REPOSITORY
                    / "src"
                    / "simple_ai_trading"
                    / "impact_absorption_exchange_info_evidence.py"
                ).read_bytes()
            ).hexdigest(),
            "capture_tool_path": (
                "tools/capture_round74_exchange_info_evidence.py"
            ),
            "capture_tool_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "quantity_rules": round74_quantity_rules_evidence_claims(rules),
        "target_evidence": bundle.evidence.as_dict(),
        "authority": {
            "model_training": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_sha256"
        }
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    artifact = capture_round74_exchange_info(
        timeout_seconds=arguments.timeout_seconds,
    )
    _write_artifact(arguments.output, artifact)
    rules = artifact["quantity_rules"]
    assert isinstance(rules, Mapping)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "symbols": sorted(
                    rules["market_quantity_rules_by_symbol"]
                ),
                "response_body_bytes": artifact["response"]["body_bytes"],
                "used_weight_1m": artifact["response"]["headers"][
                    "x_mbx_used_weight_1m"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
