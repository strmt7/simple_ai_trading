"""Frozen target-blind venue parameters for the Round 25 diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re


POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_SCHEMA_VERSION = (
    "polymarket-round25-v2-forensic-venue-parameter-audit-v1"
)
POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256 = (
    "244ad6bc32149315ca0a47c9c1d49473c3c54a068f23d3c345c4a1d630bbe3c2"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def validate_round25_forensic_venue_parameter_audit(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Round 25 forensic venue-parameter audit type differs")
    payload = dict(value)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    expected = {
        "created_at_utc",
        "economic_replay_binding",
        "observations",
        "query_boundary",
        "round",
        "schema_version",
        "source_bindings",
        "status",
        "truth_state",
    }
    source = payload.get("source_bindings")
    boundary = payload.get("query_boundary")
    observations = payload.get("observations")
    economics = payload.get("economic_replay_binding")
    truth = payload.get("truth_state")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or claimed != POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_SCHEMA_VERSION
        or payload.get("round") != 25
        or payload.get("status")
        != "target_blind_venue_parameters_match_frozen_economic_replay"
        or not isinstance(source, Mapping)
        or source.get("forensic_audit_sha256")
        != "8ee546844fada87ab4a542f6620bc5e83654b635b6a72145a338c02431c41276"
        or source.get("recorder_report_sha256")
        != "45a613588f15ef45f57c51931d1b01e19f7cb7b0d6b08e7be0b0dd5b8d49631d"
        or source.get("evidence_manifest_sha256")
        != "9d45a88606239646302af3d0c8b3c1ffdbf824ea3836b73418bdaaad35f045d9"
        or source.get("salvage_contract_sha256")
        != "f46c9c629427ab5e2ce5582bdec9be7f6e67bc8f69831fc27ebde1b1f13eafcb"
        or source.get("run_id") != "f96a24bdaa2d4f5f8cdad3f06193a0ce"
        or not isinstance(boundary, Mapping)
        or boundary.get("database_mode") != "read_only_inactive_wal_free"
        or boundary.get("table") != "polymarket_market_snapshot"
        or boundary.get("snapshot_row_count") != 111
        or boundary.get("distinct_condition_count") != 111
        or boundary.get("run_filter_applied") is not True
        or any(
            boundary.get(field) is not False
            for field in (
                "outcomes_or_resolutions_read",
                "model_scores_read",
                "database_mutated",
            )
        )
        or not isinstance(observations, Mapping)
        or observations.get("tick_size_groups")
        != [
            {
                "tick_size": "0.01",
                "snapshot_count": 111,
                "distinct_condition_count": 111,
            }
        ]
        or observations.get("fee_parameter_groups")
        != [
            {
                "fees_enabled": True,
                "fee_rate": "0.07",
                "snapshot_count": 111,
                "distinct_condition_count": 111,
            }
        ]
        or not isinstance(economics, Mapping)
        or economics.get("all_captured_conditions_match") is not True
        or economics.get("stressed_tick_size") != "0.01"
        or economics.get("taker_fee_rate") != "0.07"
        or economics.get("fee_curve_exponent") != 1
        or economics.get("fee_formula")
        != "ceil_to_0.00001(quantity*rate*(price*(1-price))^exponent)"
        or economics.get("per_market_parameter_assumption_allowed") is not False
        or economics.get("abort_if_admitted_population_differs") is not True
        or not isinstance(truth, Mapping)
        or any(value is not False for value in truth.values())
    ):
        raise ValueError("Round 25 forensic venue-parameter audit differs")
    return {**payload, "artifact_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256",
    "POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_SCHEMA_VERSION",
    "validate_round25_forensic_venue_parameter_audit",
]
