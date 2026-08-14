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
    "be5cb9626d8fff531cb0d4e3d9feac520e6e82e1019476d4486e3c8950b5fa67"
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
        or payload.get("created_at_utc") != "2026-08-14T18:47:19Z"
        or payload.get("status")
        != "target_blind_venue_parameters_match_frozen_economic_replay"
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "evidence_manifest_sha256",
            "forensic_audit_sha256",
            "recorder_report_sha256",
            "run_id",
            "salvage_contract_sha256",
        }
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
        or set(boundary)
        != {
            "database_mode",
            "database_mutated",
            "distinct_condition_count",
            "model_scores_read",
            "outcomes_or_resolutions_read",
            "run_filter_applied",
            "snapshot_row_count",
            "table",
        }
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
        or set(observations) != {"market_parameter_groups", "tick_size_groups"}
        or observations.get("tick_size_groups")
        != [
            {
                "tick_size": "0.01",
                "snapshot_count": 111,
                "distinct_condition_count": 111,
            }
        ]
        or observations.get("market_parameter_groups")
        != [
            {
                "tick_size": "0.01",
                "minimum_order_size": "5",
                "fees_enabled": True,
                "fee_rate": "0.07",
                "fee_exponent": 1,
                "fee_taker_only": True,
                "fee_rebate_rate": "0.2",
                "maker_base_fee": 1000,
                "taker_base_fee": 1000,
                "taker_order_delay_enabled": True,
                "minimum_order_age_seconds": 0,
                "snapshot_count": 111,
                "distinct_condition_count": 111,
            }
        ]
        or not isinstance(economics, Mapping)
        or set(economics)
        != {
            "abort_if_admitted_population_differs",
            "all_captured_conditions_match",
            "fee_curve_exponent",
            "fee_formula",
            "formal_after_cost_claim_allowed",
            "measured_order_delay_distribution_available",
            "minimum_order_size_shares",
            "per_market_parameter_assumption_allowed",
            "stressed_tick_size",
            "taker_order_delay_observed",
            "taker_fee_rate",
        }
        or economics.get("all_captured_conditions_match") is not True
        or economics.get("stressed_tick_size") != "0.01"
        or economics.get("taker_fee_rate") != "0.07"
        or economics.get("fee_curve_exponent") != 1
        or economics.get("fee_formula")
        != "ceil_to_0.00001(quantity*rate*(price*(1-price))^exponent)"
        or economics.get("minimum_order_size_shares") != "5"
        or economics.get("taker_order_delay_observed") is not True
        or economics.get("measured_order_delay_distribution_available") is not False
        or economics.get("formal_after_cost_claim_allowed") is not False
        or economics.get("per_market_parameter_assumption_allowed") is not False
        or economics.get("abort_if_admitted_population_differs") is not True
        or not isinstance(truth, Mapping)
        or set(truth)
        != {
            "feature_store_published",
            "live_trading_authority",
            "paper_trading_authority",
            "partition_manifest_published",
            "profitability_claim",
            "selection_predictions_frozen",
            "targets_accessed",
        }
        or any(value is not False for value in truth.values())
    ):
        raise ValueError("Round 25 forensic venue-parameter audit differs")
    return {**payload, "artifact_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256",
    "POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_SCHEMA_VERSION",
    "validate_round25_forensic_venue_parameter_audit",
]
