"""Capture one bounded public Binance exchange-info evidence artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_exchange_info_evidence import (  # noqa: E402
    build_round74_quantity_rules_evidence,
)
from simple_ai_trading.impact_absorption_event_targets import (  # noqa: E402
    round74_quantity_rules_evidence_claims,
)


ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION = (
    "round-074-exchange-info-capture-v1"
)
ROUND74_EXCHANGE_INFO_URL = (
    "https://fapi.binance.com/fapi/v1/exchangeInfo"
)
ROUND74_EXCHANGE_INFO_MAXIMUM_RESPONSE_BYTES = 8 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                "Round 74 exchange-info response contains duplicate JSON keys"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(
        f"Round 74 exchange-info response contains {value}"
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("Round 74 exchange-info git identity differs")
    return commit


def _require_clean_tracked_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError(
            "Round 74 exchange-info capture requires a clean tracked worktree"
        )


def _safe_header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    selected = str(value).strip()
    if not selected or len(selected) > 256:
        raise ValueError("Round 74 exchange-info response header differs")
    return selected


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
    request_started_wall_ns = time.time_ns()
    request_started_monotonic_ns = time.monotonic_ns()
    request = Request(
        ROUND74_EXCHANGE_INFO_URL,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-round74-research/1",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if (
                response.status != 200
                or response.geturl() != ROUND74_EXCHANGE_INFO_URL
            ):
                raise ValueError(
                    "Round 74 exchange-info response identity differs"
                )
            content_type = response.headers.get_content_type()
            if content_type != "application/json":
                raise ValueError(
                    "Round 74 exchange-info content type differs"
                )
            body = response.read(
                ROUND74_EXCHANGE_INFO_MAXIMUM_RESPONSE_BYTES + 1
            )
            if (
                not body
                or len(body)
                > ROUND74_EXCHANGE_INFO_MAXIMUM_RESPONSE_BYTES
            ):
                raise ValueError(
                    "Round 74 exchange-info response size differs"
                )
            response_headers = {
                "content_type": content_type,
                "date": _safe_header(response.headers, "Date"),
                "x_mbx_used_weight_1m": _safe_header(
                    response.headers,
                    "X-MBX-USED-WEIGHT-1M",
                ),
            }
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Round 74 exchange-info request failed without retry"
        ) from exc
    received_monotonic_ns = time.monotonic_ns()
    received_wall_ns = time.time_ns()
    try:
        payload = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Round 74 exchange-info response is not strict JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 exchange-info response root differs")
    bundle = build_round74_quantity_rules_evidence(
        payload=payload,
        environment="binance_usdm_mainnet",
        observed_wall_ns=received_wall_ns,
    )
    rules = bundle.as_mapping()
    artifact: dict[str, object] = {
        "schema_version": ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION,
        "artifact_sha256": "",
        "captured_at_utc": datetime.fromtimestamp(
            received_wall_ns / 1_000_000_000,
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
            "request_started_wall_ns": request_started_wall_ns,
            "request_started_monotonic_ns": request_started_monotonic_ns,
        },
        "response": {
            "status": 200,
            "received_wall_ns": received_wall_ns,
            "received_monotonic_ns": received_monotonic_ns,
            "elapsed_monotonic_ns": (
                received_monotonic_ns - request_started_monotonic_ns
            ),
            "body_bytes": len(body),
            "headers": response_headers,
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


def _write_artifact(path: Path, artifact: Mapping[str, object]) -> None:
    selected = path.resolve()
    try:
        selected.relative_to(REPOSITORY)
    except ValueError as exc:
        raise ValueError(
            "Round 74 exchange-info output must remain inside the repository"
        ) from exc
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("ascii")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)


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
