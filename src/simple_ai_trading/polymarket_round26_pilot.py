"""Preregister and run a bounded TWAP-60 receipt-time development pilot."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import BinaryIO, Callable, Mapping

from .polymarket import (
    POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
    PolymarketPublicClient,
)
from .polymarket_recorder import (
    POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
    PolymarketPublicRecorder,
    RecorderReport,
)
from .storage import write_bytes_atomic


ROUND26_PILOT_SCHEMA_VERSION = "polymarket-round26-twap60-pilot-contract-v2"
ROUND26_MANIFEST_SCHEMA_VERSION = "polymarket-round26-twap60-pilot-manifest-v2"
ROUND26_RESULT_SCHEMA_VERSION = "polymarket-round26-twap60-pilot-result-v2"
ROUND26_CAPTURE_DURATION_SECONDS = 3_600
ROUND26_DATABASE_CAP_BYTES = 4 * 1024**3
ROUND26_MINIMUM_FREE_BYTES = 32 * 1024**3
ROUND26_SOURCE_QUALIFICATION_RELATIVE = (
    "docs/model-research/polymarket/"
    "btc-5m-twap-60-wire-source-qualification-v1-2026-08-14.json"
)
ROUND26_REQUIRED_SOURCE_FILES = (
    "src/simple_ai_trading/polymarket.py",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_round26_pilot.py",
    "tools/run_polymarket_round26_pilot.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


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


def _text_sha256(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"source file is not UTF-8 text: {path}") from exc
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise ValueError(f"source file contains unsupported carriage returns: {path}")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class Round26PilotContract:
    created_at_ms: int
    effective_start_ms: int
    capture_duration_seconds: int
    repository_commit: str
    source_qualification_claim_sha256: str
    source_text_sha256: Mapping[str, str]
    contract_sha256: str


def create_round26_pilot_contract(
    repository: str | Path,
    *,
    created_at_ms: int,
    effective_start_ms: int,
) -> dict[str, object]:
    root = Path(repository).resolve()
    created = int(created_at_ms)
    effective = int(effective_start_ms)
    source_path = root / ROUND26_SOURCE_QUALIFICATION_RELATIVE
    source = _read_json(source_path, label="Round 26 source qualification")
    source_without_claim = dict(source)
    source_claim = str(source_without_claim.pop("qualification_sha256", "")).lower()
    if (
        _SHA256.fullmatch(source_claim) is None
        or source_claim != _canonical_sha256(source_without_claim)
        or source.get("status") != "passed"
        or source.get("resolution_source")
        != POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
        or source.get("required_rtds_topic")
        != POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC
        or source.get("twap_window_seconds") != 60
        or source.get("model_data_eligible") is not False
        or source.get("edge_claim") is not False
        or source.get("profitability_claim") is not False
    ):
        raise ValueError("Round 26 source qualification claim differs")
    payload: dict[str, object] = {
        "schema_version": ROUND26_PILOT_SCHEMA_VERSION,
        "created_at_ms": created,
        "effective_start_ms": effective,
        "capture_duration_seconds": ROUND26_CAPTURE_DURATION_SECONDS,
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "source_qualification_path": ROUND26_SOURCE_QUALIFICATION_RELATIVE,
        "source_qualification_sha256": source_claim,
        "source_text_sha256": {
            relative: _text_sha256(root / relative)
            for relative in ROUND26_REQUIRED_SOURCE_FILES
        },
        "predecessor_lineage": {
            "round25_plan_sha256": (
                "a0b5525697c3c1e1b175bd0f0ac724fdb62845638d2040e9964221031d3e7b20"
            ),
            "round25_terminal_state_sha256": (
                "827655168339cfb7fae34fa8a2b06770e58134667c0d1a845d32fa72559a27b7"
            ),
            "round25_forensic_report_sha256": (
                "45a613588f15ef45f57c51931d1b01e19f7cb7b0d6b08e7be0b0dd5b8d49631d"
            ),
            "predecessor_data_reused": False,
            "reason": "settlement source changed from a 30-second to 60-second TWAP",
        },
        "capture_scope": {
            "assets": ["BTC"],
            "required_streams": [
                "binance_futures",
                "binance_spot",
                "clob_market",
                "polymarket_rtds",
            ],
            "required_clob_lanes": ["clob"],
            "required_rtds_topics": [
                "crypto_prices",
                POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
            ],
            "resolution_source": (
                POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
            ),
            "crypto_market_config_id": "btc-5m-twap-60",
            "twap_window_seconds": 60,
            "one_process_monotonic_receipt_clock": True,
        },
        "experiment": {
            "role": "development_only",
            "sealed_selection_eligible": False,
            "primary_hypothesis": (
                "receipt-time Binance spot and futures innovations predict an "
                "executable Polymarket repricing after measured transport delay"
            ),
            "candidate_actions": [
                "cross_up_ask",
                "cross_down_ask",
                "post_only_up_bid",
                "post_only_down_bid",
                "abstain",
            ],
            "forward_receipt_horizons_ms": [100, 250, 500, 1000, 2000],
            "taker_accounting": (
                "actual displayed depth plus exact per-market taker fee; no rebate"
            ),
            "maker_accounting": (
                "price-time queue-ahead lower bound, adverse-selection markout, "
                "zero rebate in the primary result"
            ),
            "latency_policy": (
                "use empirical source-to-receipt and cross-feed receipt distributions; "
                "do not assume zero latency"
            ),
            "minimum_executable_actions_for_followup": 20,
            "pilot_pass_conditions": {
                "positive_after_cost_pnl": True,
                "positive_mean_markout": True,
                "no_integrity_errors": True,
                "minimum_action_count_met": True,
            },
            "pilot_pass_is_edge_claim": False,
            "pilot_pass_is_profitability_claim": False,
        },
        "resource_limits": {
            "database_cap_bytes": ROUND26_DATABASE_CAP_BYTES,
            "minimum_free_bytes": ROUND26_MINIMUM_FREE_BYTES,
            "duckdb_memory_limit": "1GB",
            "duckdb_threads": 2,
            "queue_capacity": 100_000,
        },
        "authority": {
            "credentials_used": False,
            "execution_connected": False,
            "orders_submitted": False,
            "model_data_eligible": False,
            "edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    validate_round26_pilot_contract(payload, repository=root)
    return payload


def validate_round26_pilot_contract(
    value: Mapping[str, object],
    *,
    repository: str | Path,
) -> Round26PilotContract:
    root = Path(repository).resolve()
    payload = dict(value)
    claimed = str(payload.pop("contract_sha256", "")).lower()
    source_files = payload.get("source_text_sha256")
    lineage = payload.get("predecessor_lineage")
    scope = payload.get("capture_scope")
    experiment = payload.get("experiment")
    resources = payload.get("resource_limits")
    authority = payload.get("authority")
    created = payload.get("created_at_ms")
    effective = payload.get("effective_start_ms")
    repository_commit = str(payload.get("repository_commit", "")).lower()
    source_claim_sha = str(
        payload.get("source_qualification_sha256", "")
    ).lower()
    if (
        not root.is_dir()
        or set(payload)
        != {
            "schema_version",
            "created_at_ms",
            "effective_start_ms",
            "capture_duration_seconds",
            "repository_commit",
            "source_qualification_path",
            "source_qualification_sha256",
            "source_text_sha256",
            "predecessor_lineage",
            "capture_scope",
            "experiment",
            "resource_limits",
            "authority",
        }
        or payload.get("schema_version") != ROUND26_PILOT_SCHEMA_VERSION
        or type(created) is not int
        or type(effective) is not int
        or created <= 0
        or effective <= created
        or effective % 300_000 != 0
        or effective - created > 900_000
        or payload.get("capture_duration_seconds")
        != ROUND26_CAPTURE_DURATION_SECONDS
        or _COMMIT.fullmatch(repository_commit) is None
        or payload.get("source_qualification_path")
        != ROUND26_SOURCE_QUALIFICATION_RELATIVE
        or _SHA256.fullmatch(source_claim_sha) is None
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or not isinstance(source_files, Mapping)
        or set(source_files) != set(ROUND26_REQUIRED_SOURCE_FILES)
        or any(_SHA256.fullmatch(str(item)) is None for item in source_files.values())
        or not isinstance(lineage, Mapping)
        or set(lineage)
        != {
            "round25_plan_sha256",
            "round25_terminal_state_sha256",
            "round25_forensic_report_sha256",
            "predecessor_data_reused",
            "reason",
        }
        or lineage.get("round25_plan_sha256")
        != "a0b5525697c3c1e1b175bd0f0ac724fdb62845638d2040e9964221031d3e7b20"
        or lineage.get("round25_terminal_state_sha256")
        != "827655168339cfb7fae34fa8a2b06770e58134667c0d1a845d32fa72559a27b7"
        or lineage.get("round25_forensic_report_sha256")
        != "45a613588f15ef45f57c51931d1b01e19f7cb7b0d6b08e7be0b0dd5b8d49631d"
        or lineage.get("predecessor_data_reused") is not False
        or lineage.get("reason")
        != "settlement source changed from a 30-second to 60-second TWAP"
        or not isinstance(scope, Mapping)
        or set(scope)
        != {
            "assets",
            "required_streams",
            "required_clob_lanes",
            "required_rtds_topics",
            "resolution_source",
            "crypto_market_config_id",
            "twap_window_seconds",
            "one_process_monotonic_receipt_clock",
        }
        or scope.get("assets") != ["BTC"]
        or scope.get("required_streams")
        != ["binance_futures", "binance_spot", "clob_market", "polymarket_rtds"]
        or scope.get("required_clob_lanes") != ["clob"]
        or scope.get("required_rtds_topics")
        != ["crypto_prices", POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC]
        or scope.get("resolution_source")
        != POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
        or scope.get("crypto_market_config_id") != "btc-5m-twap-60"
        or scope.get("twap_window_seconds") != 60
        or scope.get("one_process_monotonic_receipt_clock") is not True
        or not isinstance(experiment, Mapping)
        or set(experiment)
        != {
            "role",
            "sealed_selection_eligible",
            "primary_hypothesis",
            "candidate_actions",
            "forward_receipt_horizons_ms",
            "taker_accounting",
            "maker_accounting",
            "latency_policy",
            "minimum_executable_actions_for_followup",
            "pilot_pass_conditions",
            "pilot_pass_is_edge_claim",
            "pilot_pass_is_profitability_claim",
        }
        or experiment.get("role") != "development_only"
        or experiment.get("sealed_selection_eligible") is not False
        or experiment.get("primary_hypothesis")
        != (
            "receipt-time Binance spot and futures innovations predict an "
            "executable Polymarket repricing after measured transport delay"
        )
        or experiment.get("candidate_actions")
        != [
            "cross_up_ask",
            "cross_down_ask",
            "post_only_up_bid",
            "post_only_down_bid",
            "abstain",
        ]
        or experiment.get("forward_receipt_horizons_ms")
        != [100, 250, 500, 1000, 2000]
        or experiment.get("taker_accounting")
        != "actual displayed depth plus exact per-market taker fee; no rebate"
        or experiment.get("maker_accounting")
        != (
            "price-time queue-ahead lower bound, adverse-selection markout, "
            "zero rebate in the primary result"
        )
        or experiment.get("latency_policy")
        != (
            "use empirical source-to-receipt and cross-feed receipt distributions; "
            "do not assume zero latency"
        )
        or experiment.get("minimum_executable_actions_for_followup") != 20
        or experiment.get("pilot_pass_conditions")
        != {
            "positive_after_cost_pnl": True,
            "positive_mean_markout": True,
            "no_integrity_errors": True,
            "minimum_action_count_met": True,
        }
        or experiment.get("pilot_pass_is_edge_claim") is not False
        or experiment.get("pilot_pass_is_profitability_claim") is not False
        or not isinstance(resources, Mapping)
        or dict(resources)
        != {
            "database_cap_bytes": ROUND26_DATABASE_CAP_BYTES,
            "minimum_free_bytes": ROUND26_MINIMUM_FREE_BYTES,
            "duckdb_memory_limit": "1GB",
            "duckdb_threads": 2,
            "queue_capacity": 100_000,
        }
        or not isinstance(authority, Mapping)
        or dict(authority)
        != {
            "credentials_used": False,
            "execution_connected": False,
            "orders_submitted": False,
            "model_data_eligible": False,
            "edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 26 pilot contract differs")
    return Round26PilotContract(
        created_at_ms=created,
        effective_start_ms=effective,
        capture_duration_seconds=ROUND26_CAPTURE_DURATION_SECONDS,
        repository_commit=repository_commit,
        source_qualification_claim_sha256=source_claim_sha,
        source_text_sha256={str(key): str(item) for key, item in source_files.items()},
        contract_sha256=claimed,
    )


def load_round26_pilot_contract(
    path: str | Path,
    *,
    repository: str | Path,
) -> Round26PilotContract:
    return validate_round26_pilot_contract(
        _read_json(Path(path), label="Round 26 pilot contract"),
        repository=repository,
    )


def write_round26_pilot_contract(path: str | Path, value: Mapping[str, object]) -> None:
    payload = dict(value)
    write_bytes_atomic(
        Path(path),
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


@dataclass(frozen=True, slots=True)
class Round26PilotConfig:
    repository: Path
    contract_path: Path
    database_path: Path
    result_path: Path
    lock_path: Path

    def validated(self) -> Round26PilotConfig:
        root = self.repository.resolve()
        paths = tuple(
            path.resolve()
            for path in (
                self.contract_path,
                self.database_path,
                self.result_path,
                self.lock_path,
            )
        )
        if (
            not root.is_dir()
            or not paths[0].is_file()
            or any(root not in path.parents for path in paths)
            or len(set(paths)) != len(paths)
        ):
            raise ValueError("Round 26 pilot configuration differs")
        return Round26PilotConfig(root, *paths)


class _PilotFileLock(AbstractContextManager["_PilotFileLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> _PilotFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Round 26 pilot is already running") from exc
        self.handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _verify_sources(repository: Path, contract: Round26PilotContract) -> None:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract.repository_commit, "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Round 26 source commit is not an ancestor of HEAD")
    for relative, expected in contract.source_text_sha256.items():
        if _text_sha256(repository / relative) != expected:
            raise ValueError(f"Round 26 source file differs: {relative}")
    qualification = repository / ROUND26_SOURCE_QUALIFICATION_RELATIVE
    qualification_payload = _read_json(
        qualification, label="Round 26 source qualification"
    )
    qualification_claim = str(
        qualification_payload.pop("qualification_sha256", "")
    ).lower()
    if (
        qualification_claim != contract.source_qualification_claim_sha256
        or qualification_claim != _canonical_sha256(qualification_payload)
    ):
        raise ValueError("Round 26 source qualification claim differs")


def _resource_block(config: Round26PilotConfig) -> str | None:
    database_bytes = (
        config.database_path.stat().st_size if config.database_path.exists() else 0
    )
    if database_bytes >= ROUND26_DATABASE_CAP_BYTES:
        return "database_cap_reached"
    if shutil.disk_usage(config.database_path.parent).free < ROUND26_MINIMUM_FREE_BYTES:
        return "minimum_free_space_breached"
    return None


def _create_recorder(database: Path) -> PolymarketPublicRecorder:
    return PolymarketPublicRecorder(
        database,
        client=PolymarketPublicClient(
            required_five_minute_resolution_sources={
                "BTC": POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
            }
        ),
        queue_capacity=100_000,
        discovery_interval_seconds=30,
        memory_limit="1GB",
        database_threads=2,
        assets=("BTC",),
        include_binance_futures=True,
        include_binance_spot=True,
        include_rtds_binance=True,
        chainlink_price_mode="twap_60s",
        clob_lane_ids=("clob",),
    )


def _manifest(
    contract: Round26PilotContract,
    *,
    run_id: str,
    started_at_ms: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND26_MANIFEST_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "run_id": run_id,
        "created_at_ms": started_at_ms,
        "capture_duration_seconds": contract.capture_duration_seconds,
        "required_assets": ["BTC"],
        "required_streams": [
            "binance_futures",
            "binance_spot",
            "clob_market",
            "polymarket_rtds",
        ],
        "required_clob_lanes": ["clob"],
        "required_rtds_topics": [
            "crypto_prices",
            POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
        ],
        "source_qualification_sha256": contract.source_qualification_claim_sha256,
        "development_only": True,
        "sealed_selection_eligible": False,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _write_result(
    path: Path,
    contract: Round26PilotContract,
    report: RecorderReport,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND26_RESULT_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "status": report.status,
        "run_id": report.run_id,
        "report_sha256": report.report_sha256,
        "started_at_ms": report.started_at_ms,
        "ended_at_ms": report.ended_at_ms,
        "duration_seconds": report.duration_seconds,
        "raw_message_count": report.raw_message_count,
        "market_snapshot_count": report.market_snapshot_count,
        "condition_count": len(report.conditions),
        "stream_counts": dict(report.stream_counts),
        "stream_gap_count": report.stream_gap_count,
        "integrity_errors": list(report.integrity_errors),
        "errors": list(report.errors),
        "development_only": True,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return payload


async def run_round26_pilot(
    config: Round26PilotConfig,
    *,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
    recorder_factory: Callable[[Path], PolymarketPublicRecorder] = _create_recorder,
) -> dict[str, object]:
    selected = config.validated()
    contract = load_round26_pilot_contract(
        selected.contract_path, repository=selected.repository
    )
    _verify_sources(selected.repository, contract)
    selected.database_path.parent.mkdir(parents=True, exist_ok=True)
    selected.result_path.parent.mkdir(parents=True, exist_ok=True)
    with _PilotFileLock(selected.lock_path):
        while True:
            now = time.time_ns() // 1_000_000
            if now >= contract.effective_start_ms:
                break
            if progress is not None:
                progress(
                    "waiting",
                    {
                        "observed_at_ms": now,
                        "effective_start_ms": contract.effective_start_ms,
                        "remaining_seconds": (contract.effective_start_ms - now) / 1000,
                    },
                )
            await asyncio.sleep(min(15.0, (contract.effective_start_ms - now) / 1000))
        started = time.time_ns() // 1_000_000
        if started - contract.effective_start_ms > 60_000:
            raise RuntimeError("Round 26 pilot missed its preregistered start window")
        resource = _resource_block(selected)
        if resource is not None:
            raise RuntimeError(f"Round 26 pilot resource block: {resource}")

        def manifest_factory(run_id: str, started_at_ms: int) -> Mapping[str, object]:
            return _manifest(contract, run_id=run_id, started_at_ms=started_at_ms)

        report = await recorder_factory(selected.database_path).run(
            duration_seconds=contract.capture_duration_seconds,
            progress=progress,
            progress_interval_seconds=30,
            stop_requested=lambda: _resource_block(selected),
            preregistration_manifest_factory=manifest_factory,
        )
        _verify_sources(selected.repository, contract)
        return _write_result(selected.result_path, contract, report)


__all__ = [
    "ROUND26_CAPTURE_DURATION_SECONDS",
    "ROUND26_PILOT_SCHEMA_VERSION",
    "Round26PilotConfig",
    "Round26PilotContract",
    "create_round26_pilot_contract",
    "load_round26_pilot_contract",
    "run_round26_pilot",
    "validate_round26_pilot_contract",
    "write_round26_pilot_contract",
]
