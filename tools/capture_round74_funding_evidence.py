"""Capture bounded public funding evidence for an audited Round 74 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence
from urllib.parse import urlencode


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_event_evidence import (  # noqa: E402
    build_round74_funding_evidence,
    load_round74_binance_clock_probes,
)
from simple_ai_trading.impact_absorption_event_targets import (  # noqa: E402
    ROUND74_EVENT_TARGET_SYMBOLS,
)
from simple_ai_trading.impact_absorption_store import (  # noqa: E402
    ImpactAbsorptionStore,
)
from tools._round74_public_evidence_capture import (  # noqa: E402
    bounded_json_get,
    canonical_sha256,
    git_commit,
    require_clean_tracked_worktree,
    strict_json_loads,
    write_artifact,
)


ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION = (
    "round-074-funding-capture-v1"
)
ROUND74_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
ROUND74_FUNDING_LIMIT = 1_000
ROUND74_FUNDING_MAXIMUM_RESPONSE_BYTES = 2 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_capture_artifact(
    path: Path,
    *,
    run_id: str,
    database_path: Path,
) -> dict[str, object]:
    payload = strict_json_loads(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 funding capture artifact root differs")
    selected = dict(payload)
    claimed = str(selected.pop("artifact_sha256", ""))
    if claimed != canonical_sha256(selected):
        raise ValueError("Round 74 funding capture artifact hash differs")
    selected["artifact_sha256"] = claimed
    capture = selected.get("capture")
    if (
        not isinstance(capture, Mapping)
        or capture.get("run_id") != run_id
        or capture.get("status") != "completed"
        or selected.get("database")
        != database_path.resolve().relative_to(REPOSITORY).as_posix()
    ):
        raise ValueError("Round 74 funding capture identity differs")
    return selected


def _progress(stage: str, **values: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _funding_url(
    *,
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
) -> str:
    query = urlencode(
        {
            "symbol": symbol,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": ROUND74_FUNDING_LIMIT,
        }
    )
    return f"{ROUND74_FUNDING_URL}?{query}"


def capture_round74_funding(
    *,
    database_path: Path,
    run_id: str,
    capture_artifact_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    """Reaudit one run, fetch three bounded responses, and bind evidence."""

    timeout = float(timeout_seconds)
    selected_run = str(run_id).strip()
    selected_database = database_path.resolve()
    selected_capture_artifact = capture_artifact_path.resolve()
    if (
        len(selected_run) != 32
        or not selected_database.is_file()
        or not selected_capture_artifact.is_file()
        or not 1.0 <= timeout <= 60.0
    ):
        raise ValueError("Round 74 funding capture arguments differ")
    require_clean_tracked_worktree()
    execution_commit = git_commit()
    capture_artifact = _load_capture_artifact(
        selected_capture_artifact,
        run_id=selected_run,
        database_path=selected_database,
    )
    _progress(
        "audit_started",
        run_id=selected_run,
        database_bytes=selected_database.stat().st_size,
    )
    store = ImpactAbsorptionStore(
        selected_database,
        memory_limit="2GB",
        threads=2,
        read_only=True,
    )
    try:
        audit = store.audit_run(selected_run)
        if not audit.passed or audit.errors:
            raise ValueError("Round 74 funding capture audit failed")
        _progress(
            "audit_completed",
            frame_count=audit.frame_count,
            message_count=audit.message_count,
        )
        probes = load_round74_binance_clock_probes(
            store.connect(),
            run_id=selected_run,
            capture_audit=audit,
        )
    finally:
        store.close()
    start_time_ms = probes[0].exchange_time_ms
    end_time_ms = probes[-1].exchange_time_ms
    if end_time_ms <= start_time_ms:
        raise ValueError("Round 74 funding capture clock range differs")

    payloads: dict[str, Sequence[Mapping[str, object]]] = {}
    request_records: list[dict[str, object]] = []
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        url = _funding_url(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
        response = bounded_json_get(
            url=url,
            timeout_seconds=timeout,
            maximum_response_bytes=(
                ROUND74_FUNDING_MAXIMUM_RESPONSE_BYTES
            ),
            user_agent="simple-ai-trading-round74-research/1",
        )
        if not isinstance(response.payload, Sequence) or isinstance(
            response.payload,
            (str, bytes, bytearray),
        ) or any(not isinstance(row, Mapping) for row in response.payload):
            raise ValueError("Round 74 funding response root differs")
        rows = tuple(response.payload)
        payloads[symbol] = rows
        request_records.append(
            {
                "symbol": symbol,
                "method": "GET",
                "url": url,
                "security_type": "NONE",
                "shared_rate_limit": "500 requests per 5 minutes per IP",
                "limit": ROUND74_FUNDING_LIMIT,
                "retry_count": 0,
                "credential_material_sent": False,
                "request_started_wall_ns": (
                    response.request_started_wall_ns
                ),
                "request_started_monotonic_ns": (
                    response.request_started_monotonic_ns
                ),
                "received_wall_ns": response.received_wall_ns,
                "received_monotonic_ns": response.received_monotonic_ns,
                "elapsed_monotonic_ns": response.elapsed_monotonic_ns,
                "response_body_bytes": len(response.body),
                "response_row_count": len(rows),
                "response_headers": response.header_mapping(),
                "raw_payload_persisted": False,
            }
        )
        _progress(
            "symbol_completed",
            symbol=symbol,
            row_count=len(rows),
            used_weight_1m=(
                response.header_mapping().get("x_mbx_used_weight_1m")
            ),
        )
    observed_wall_ns = max(
        int(record["received_wall_ns"]) for record in request_records
    )
    bundle = build_round74_funding_evidence(
        payload_by_symbol=payloads,
        environment="binance_usdm_mainnet",
        observed_wall_ns=observed_wall_ns,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        limit=ROUND74_FUNDING_LIMIT,
        clock_probes=probes,
    )
    clock_source_records = [
        probe.as_source_record() for probe in probes
    ]
    artifact: dict[str, object] = {
        "schema_version": ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION,
        "artifact_sha256": "",
        "captured_at_utc": datetime.fromtimestamp(
            observed_wall_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "execution_git_commit": execution_commit,
        "database": selected_database.relative_to(REPOSITORY).as_posix(),
        "database_open_mode": "read_only",
        "database_bytes": selected_database.stat().st_size,
        "capture_binding": {
            "capture_artifact_path": (
                selected_capture_artifact.relative_to(REPOSITORY).as_posix()
            ),
            "capture_artifact_file_sha256": _sha256_file(
                selected_capture_artifact
            ),
            "capture_artifact_sha256": (
                capture_artifact["artifact_sha256"]
            ),
            "run_id": selected_run,
            "fresh_full_run_audit_passed": audit.passed,
            "fresh_full_run_audit_errors": list(audit.errors),
            "frame_count": audit.frame_count,
            "message_count": audit.message_count,
            "compressed_payload_bytes": audit.compressed_payload_bytes,
            "last_frame_sha256": audit.last_frame_sha256,
            "capture_contract_sha256": audit.capture_contract_sha256,
        },
        "clock_binding": {
            "probe_count": len(probes),
            "probe_panel_sha256": canonical_sha256(clock_source_records),
            "first_exchange_time_ms": start_time_ms,
            "last_exchange_time_ms": end_time_ms,
            "interpolation_permitted": False,
        },
        "requests": request_records,
        "funding_row_count_by_symbol": {
            symbol: len(payloads[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        },
        "funding_boundary_intervals_monotonic_ns": {
            symbol: [list(interval) for interval in intervals]
            for symbol, intervals in bundle.boundary_mapping().items()
        },
        "funding_schedule_coverage_monotonic_ns": {
            symbol: list(coverage)
            for symbol, coverage in bundle.coverage_mapping().items()
        },
        "target_evidence": bundle.evidence.as_dict(),
        "source_binding": {
            "parser_path": (
                "src/simple_ai_trading/"
                "impact_absorption_event_evidence.py"
            ),
            "parser_sha256": _sha256_file(
                REPOSITORY
                / "src"
                / "simple_ai_trading"
                / "impact_absorption_event_evidence.py"
            ),
            "capture_tool_path": (
                "tools/capture_round74_funding_evidence.py"
            ),
            "capture_tool_sha256": _sha256_file(Path(__file__)),
            "transport_helper_path": (
                "tools/_round74_public_evidence_capture.py"
            ),
            "transport_helper_sha256": _sha256_file(
                REPOSITORY
                / "tools"
                / "_round74_public_evidence_capture.py"
            ),
        },
        "scope": {
            "capture_run_is_design_consumed": True,
            "capture_run_may_be_used_for_financial_evaluation": False,
            "funding_parser_and_source_validation_only": True,
            "synthetic_market_data_used": False,
            "credentials_used": False,
            "orders_submitted": False,
        },
        "authority": {
            "model_training": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_sha256"
        }
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    artifact = capture_round74_funding(
        database_path=arguments.database,
        run_id=arguments.run_id,
        capture_artifact_path=arguments.capture_artifact,
        timeout_seconds=arguments.timeout_seconds,
    )
    write_artifact(arguments.output, artifact)
    _progress(
        "artifact_written",
        artifact_sha256=artifact["artifact_sha256"],
        funding_row_count_by_symbol=(
            artifact["funding_row_count_by_symbol"]
        ),
        target_evidence_sha256=artifact["target_evidence"][
            "evidence_sha256"
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
