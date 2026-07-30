"""Strict contract boundary for the independent Polymarket Round 20 corpus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


POLYMARKET_ROUND20_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round20-independent-redundant-corpus-contract-v1"
)
POLYMARKET_ROUND20_CONTRACT_SHA256 = (
    "90e837edc9bb8071f966f9d27335983e24c6060c4ba0d36fc3d2060913c421ad"
)
POLYMARKET_ROUND20_PARENT_RESULT_SHA256 = (
    "61a7a6fe2cebd3ddc8ba6d4f59c52d6c19b91fe895353fda1bb066e86ecbc5be"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTRACT_BYTES = 128 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 20 contract contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 20 contract contains {value}")


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


def _section(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    selected = value.get(name)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Round 20 {name} section is unavailable")
    return selected


@dataclass(frozen=True, slots=True)
class PolymarketRound20Program:
    contract_sha256: str
    parent_result_sha256: str
    capture_unit_seconds: int
    total_capture_units: int
    pairing_window_ms: int
    maximum_joint_unhealthy_ms: int


def validate_round20_contract(
    value: Mapping[str, object],
) -> PolymarketRound20Program:
    """Reject rehashed semantic drift as well as ordinary hash tampering."""

    contract = dict(value)
    claimed = str(contract.pop("contract_sha256", "")).strip().lower()
    parent = _section(contract, "parent")
    independence = _section(contract, "venue_independence")
    scope = _section(contract, "scope")
    capture = _section(contract, "capture")
    transport = _section(contract, "redundant_clob_transport")
    storage = _section(contract, "storage")
    admission = _section(contract, "condition_admission")
    labels = _section(contract, "labels_and_roles")
    authority = _section(contract, "authority")
    expected_top_level = {
        "schema_version",
        "round",
        "status",
        "parent",
        "venue_independence",
        "scope",
        "capture",
        "redundant_clob_transport",
        "storage",
        "condition_admission",
        "labels_and_roles",
        "authority",
    }
    if (
        set(contract) != expected_top_level
        or claimed != POLYMARKET_ROUND20_CONTRACT_SHA256
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != POLYMARKET_ROUND20_CONTRACT_SCHEMA_VERSION
        or contract.get("round") != 20
        or contract.get("status")
        != "preregistered_before_successor_capture"
        or parent
        != {
            "round19_contract_sha256": (
                "f412e449ba6716459444f07b9d18b98195e106d3322511b21ee78c7fe808ed5b"
            ),
            "round19_result_sha256": POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
            "round19_qualified": True,
        }
        or independence
        != {
            "execution_venue": "polymarket",
            "required_public_sources": [
                "polymarket_gamma",
                "polymarket_clob_a",
                "polymarket_clob_b",
                "polymarket_rtds_chainlink",
            ],
            "optional_predictor_sources": [
                "binance_spot_public",
                "binance_usdm_public",
            ],
            "binance_role": "optional_read_only_predictor",
            "binance_credentials_allowed": False,
            "binance_order_endpoints_allowed": False,
            "binance_unavailability_blocks_polymarket_core": False,
            "missing_optional_features": "explicitly_missing_never_imputed",
        }
        or scope
        != {
            "asset": "BTC",
            "market_horizon_seconds": 300,
            "decision_cadence_ms": 250,
            "current_and_next_market_only": True,
        }
        or capture
        != {
            "calendar_days": 30,
            "capture_unit_seconds": 1200,
            "total_capture_units": 2160,
            "maximum_concurrent_units": 1,
            "discovery_interval_seconds": 30,
            "progress_interval_seconds": 30,
            "outcomes_access_during_capture": False,
            "model_access_during_capture": False,
        }
        or transport
        != {
            "lane_ids": ["clob-a", "clob-b"],
            "heartbeat_seconds": 10,
            "fresh_event_seconds": 10,
            "subscribe_additions_before_removals": True,
            "preserve_exact_receipts_from_both_lanes": True,
            "semantic_digest": "sha256_canonical_json_event",
            "duplicate_pairing": "fifo_by_digest_within_window",
            "pairing_window_ms": 2000,
            "unmatched_receipts_preserved": True,
            "synthetic_source_sequence_allowed": False,
        }
        or storage
        != {
            "database": "duckdb",
            "schema": "polymarket-evidence-storage-v4",
            "compression": "zstd-level-1-exact-frames",
            "writer_count": 1,
            "database_threads": 2,
            "memory_limit": "1GB",
            "queue_capacity": 100000,
            "raw_receipts_are_authority": True,
            "derived_union_is_rebuildable": True,
            "terminal_manifest_required": True,
        }
        or admission
        != {
            "maximum_joint_unhealthy_ms": 2000,
            "minimum_lane_coverage_fraction": 0.9,
            "minimum_shared_fraction": 0.9,
            "maximum_json_parse_errors": 0,
            "maximum_writer_errors": 0,
            "exact_frame_audit_required": True,
            "both_token_books_required": True,
            "chainlink_open_and_close_required": True,
            "joint_gap_rejects_affected_condition_only": True,
            "single_lane_gap_never_silently_reclassified": True,
            "optional_predictor_coverage_required": False,
        }
        or labels
        != {
            "official_resolution_only": True,
            "whole_condition_role_assignment": True,
            "role_assignment_frozen_before_resolution_access": True,
            "train_calendar_days": 18,
            "tune_calendar_days": 5,
            "sealed_test_calendar_days": 7,
            "purge_seconds_between_roles": 1800,
            "future_books_in_inference": False,
            "future_reference_prices_in_inference": False,
        }
        or authority
        != {
            "model_data_eligible": False,
            "model_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        or _SHA256.fullmatch(claimed) is None
    ):
        raise ValueError("Round 20 contract differs")
    return PolymarketRound20Program(
        contract_sha256=claimed,
        parent_result_sha256=POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
        capture_unit_seconds=int(capture["capture_unit_seconds"]),
        total_capture_units=int(capture["total_capture_units"]),
        pairing_window_ms=int(transport["pairing_window_ms"]),
        maximum_joint_unhealthy_ms=int(
            admission["maximum_joint_unhealthy_ms"]
        ),
    )


def load_round20_contract(path: str | Path) -> PolymarketRound20Program:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_CONTRACT_BYTES
    ):
        raise ValueError("Round 20 contract is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 20 contract is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 20 contract is not an object")
    return validate_round20_contract(value)


__all__ = [
    "POLYMARKET_ROUND20_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND20_CONTRACT_SHA256",
    "POLYMARKET_ROUND20_PARENT_RESULT_SHA256",
    "PolymarketRound20Program",
    "load_round20_contract",
    "validate_round20_contract",
]
