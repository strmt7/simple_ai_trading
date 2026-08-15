from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_historical_l2 import (
    POLYMARKET_HISTORICAL_L2_CODEC,
    POLYMARKET_ORDERBOOK_HISTORY_URL,
)


REPOSITORY = Path(__file__).resolve().parents[1]
QUALIFICATION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-022-historical-l2-source-qualification-2026-08-03.json"
)
PILOT_DESIGN = QUALIFICATION.with_name("round-022-historical-l2-pilot-design-v1.json")


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_round22_historical_l2_qualification_is_hash_bound_and_non_authoritative() -> (
    None
):
    payload = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")

    assert claimed == _canonical_sha256(payload)
    assert payload["schema_version"] == (
        "polymarket-round22-historical-l2-source-qualification-v1"
    )
    assert payload["status"] == "development_feature_source_qualified_with_limitations"
    assert not any(payload["authority"].values())
    assert not any(payload["binance_boundary"].values())
    assert payload["implementation"] == {
        "codec": POLYMARKET_HISTORICAL_L2_CODEC,
        "module": "src/simple_ai_trading/polymarket_historical_l2.py",
        "network_tests": False,
        "raw_response_persisted": False,
        "test": "tests/test_polymarket_historical_l2.py",
    }
    for relative in (
        payload["implementation"]["module"],
        payload["implementation"]["test"],
    ):
        path = REPOSITORY / relative
        assert path.is_file()
        assert not path.is_symlink()


def test_round22_qualification_records_observed_fidelity_and_source_limits() -> None:
    payload = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    validated = payload["validated_closed_window"]
    limitations = payload["source_limitations"]

    assert (
        payload["official_source_audit"]["endpoint"] == POLYMARKET_ORDERBOOK_HISTORY_URL
    )
    assert validated["record_count"] == 1398
    assert validated["first_timestamp_ms"] >= validated["event_start_ms"]
    assert validated["last_timestamp_ms"] < validated["event_end_ms"]
    assert validated["strictly_monotonic"] is True
    assert validated["compressed_size_bytes"] < validated["raw_size_bytes"]
    assert limitations == {
        "authenticated_order_events_present": False,
        "client_arrival_timestamps_present": False,
        "full_book_state_present": True,
        "individual_order_queue_identity_present": False,
        "market_data_gap_evidence_present": False,
        "server_timestamps_present": True,
        "transport_latency_present": False,
    }
    assert validated["slug"] in payload["qualification_exclusions"]["slugs"]
    assert payload["retention_point_probes"]["interpretation"].startswith(
        "point samples only"
    )


def test_round22_pilot_is_frozen_hash_bound_and_excludes_all_source_probes() -> None:
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    design = json.loads(PILOT_DESIGN.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert design["status"] == "frozen_before_bulk_ingestion_or_target_access"
    assert not any(design["authority"].values())
    assert (
        design["parents"]["historical_l2_source_qualification_sha256"]
        == (qualification["artifact_sha256"])
    )

    populations: set[str] = set()
    total = 0
    excluded = set(qualification["qualification_exclusions"]["slugs"])
    for partition in design["pilot_population"]["partitions"].values():
        generated: set[str] = set()
        for day in partition["days_utc"]:
            for hour in design["pilot_population"]["hours_utc_per_day"]:
                start = int(
                    datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")
                    .astimezone(timezone.utc)
                    .timestamp()
                )
                generated.update(
                    f"btc-updown-5m-{start + (offset * 300)}" for offset in range(12)
                )
        assert len(generated) == partition["market_count_expected"]
        assert not generated & populations
        assert not generated & excluded
        populations.update(generated)
        total += len(generated)
    assert total == design["pilot_population"]["market_count_expected"] == 576


def test_round22_pilot_requires_market_control_cost_stress_and_core_independence() -> (
    None
):
    design = json.loads(PILOT_DESIGN.read_text(encoding="utf-8"))

    assert design["models"]["market_prior_is_mandatory_control"] is True
    assert design["models"]["row_as_independent_sample_allowed"] is False
    assert design["feature_policy"]["inferred_trade_direction_allowed"] is False
    assert design["feature_policy"]["future_books_allowed"] is False
    assert design["feature_policy"]["target_or_resolution_features_allowed"] is False
    predictor = design["feature_policy"]["optional_binance_predictor"]
    assert predictor["absence_blocks_core"] is False
    assert predictor["credentials_allowed"] is False
    assert predictor["account_or_execution_data_allowed"] is False
    assert design["economics"]["midpoint_fill_allowed"] is False
    assert design["economics"]["maker_fill_or_queue_claim"] is False
    assert design["expansion_gate"]["full_backfill_before_pilot_pass"] is False
    assert (
        design["expansion_gate"]["sealed_test_access_before_development_pass"] is False
    )
    assert (
        design["expansion_gate"][
            "round20_transport_calibration_required_before_promotion"
        ]
        is True
    )
