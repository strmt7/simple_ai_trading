from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "btc-5m-twap-60-wire-source-qualification-v1-2026-08-14.json"
)
SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"
TOPIC = "crypto_prices_twap_sixty"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_live_twap_60_source_qualification_is_hash_bound_and_non_authorizing() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="ascii"))
    claimed = payload.pop("qualification_sha256")

    assert claimed == _sha256(_canonical_json(payload))
    assert payload["status"] == "passed"
    assert payload["resolution_source"] == SOURCE
    assert payload["crypto_market_config_id"] == "btc-5m-twap-60"
    assert payload["required_rtds_topic"] == TOPIC
    assert payload["twap_window_seconds"] == 60
    assert payload["exact_e18_value_observed"] is True
    assert payload["credentials_used"] is False
    assert payload["execution_connected"] is False
    assert payload["orders_submitted"] == 0
    assert payload["model_data_eligible"] is False
    assert payload["edge_claim"] is False
    assert payload["profitability_claim"] is False
    assert payload["paper_trading_authority"] is False
    assert payload["live_trading_authority"] is False


def test_live_twap_60_qualification_binds_market_and_wire_receipts() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="ascii"))
    market_qualification = payload["market_qualification"]
    markets = market_qualification["markets"]

    assert market_qualification["clob_protocol_version"] == 2
    assert market_qualification["market_count"] == 2
    assert len(markets) == 2
    assert markets[1]["event_start_ms"] - markets[0]["event_start_ms"] == 300_000
    for market in markets:
        assert market["resolution_source"] == SOURCE
        assert market["end_ms"] - market["event_start_ms"] == 300_000
        assert _sha256(market["gamma_payload_json"]) == market["gamma_payload_sha256"]
        assert _sha256(market["clob_info_json"]) == market["clob_info_sha256"]

    wire = payload["wire_probe"]
    observations = wire["observations"]
    assert len(observations) == 2
    assert observations[0]["source_timestamp_ms"] < observations[1]["source_timestamp_ms"]
    for observation in observations:
        assert observation["topic"] == TOPIC
        assert observation["type"] == "update"
        assert observation["symbol"] == "btc/usd"
        assert observation["window_s"] == 60
        assert observation["full_accuracy_value"].isdigit()
        assert int(observation["full_accuracy_value"]) > 0
