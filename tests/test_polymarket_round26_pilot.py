from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round26_pilot import (
    ROUND26_CAPTURE_DURATION_SECONDS,
    _create_recorder,
    _manifest,
    _text_sha256,
    create_round26_pilot_contract,
    validate_round26_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
V1_FAILURE = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-026-twap60-development-pilot-v1-infrastructure-failure.json"
)
CREATED_MS = 1_786_736_100_000
EFFECTIVE_START_MS = 1_786_736_400_000


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round26_pilot_contract_freezes_development_only_economic_probe() -> None:
    payload = create_round26_pilot_contract(
        ROOT,
        created_at_ms=CREATED_MS,
        effective_start_ms=EFFECTIVE_START_MS,
    )
    contract = validate_round26_pilot_contract(payload, repository=ROOT)

    assert contract.capture_duration_seconds == ROUND26_CAPTURE_DURATION_SECONDS
    assert contract.effective_start_ms == EFFECTIVE_START_MS
    assert payload["capture_scope"]["required_rtds_topics"] == [
        "crypto_prices",
        "crypto_prices_twap_sixty",
    ]
    assert payload["experiment"]["role"] == "development_only"
    assert payload["experiment"]["sealed_selection_eligible"] is False
    assert payload["experiment"]["minimum_executable_actions_for_followup"] == 20
    assert payload["authority"] == {
        "credentials_used": False,
        "execution_connected": False,
        "orders_submitted": False,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


def test_round26_pilot_contract_rejects_rehashed_semantic_drift() -> None:
    payload = create_round26_pilot_contract(
        ROOT,
        created_at_ms=CREATED_MS,
        effective_start_ms=EFFECTIVE_START_MS,
    )
    payload["experiment"]["minimum_executable_actions_for_followup"] = 1
    payload_without_claim = dict(payload)
    payload_without_claim.pop("contract_sha256")
    payload["contract_sha256"] = _canonical_sha256(payload_without_claim)

    with pytest.raises(ValueError, match="contract differs"):
        validate_round26_pilot_contract(payload, repository=ROOT)


def test_round26_pilot_recorder_has_exact_one_clock_source_scope() -> None:
    recorder = _create_recorder(ROOT / "data" / "unused-round26-test.duckdb")

    assert recorder.assets == ("BTC",)
    assert recorder.include_binance_spot is True
    assert recorder.include_binance_futures is True
    assert recorder.include_rtds_binance is True
    assert recorder.chainlink_price_mode == "twap_60s"
    assert recorder.required_streams == (
        "binance_futures",
        "binance_spot",
        "clob_market",
        "polymarket_rtds",
    )
    assert recorder.rtds_topics == (
        "crypto_prices",
        "crypto_prices_twap_sixty",
    )


def test_round26_pilot_manifest_is_hash_bound_and_non_authorizing() -> None:
    payload = create_round26_pilot_contract(
        ROOT,
        created_at_ms=CREATED_MS,
        effective_start_ms=EFFECTIVE_START_MS,
    )
    contract = validate_round26_pilot_contract(payload, repository=ROOT)
    manifest = _manifest(
        contract,
        run_id="a" * 32,
        started_at_ms=EFFECTIVE_START_MS,
    )
    claimed = manifest.pop("manifest_sha256")

    assert claimed == _canonical_sha256(manifest)
    assert manifest["development_only"] is True
    assert manifest["sealed_selection_eligible"] is False
    assert manifest["model_data_eligible"] is False
    assert manifest["edge_claim"] is False
    assert manifest["profitability_claim"] is False


def test_round26_source_hash_is_stable_across_git_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")

    assert _text_sha256(lf) == _text_sha256(crlf)


def test_round26_v1_portability_failure_is_hash_bound_and_non_data() -> None:
    payload = json.loads(V1_FAILURE.read_text(encoding="ascii"))
    claimed = payload.pop("artifact_sha256")

    assert claimed == _canonical_sha256(payload)
    assert payload["status"] == "failed_before_capture"
    assert payload["database_created"] is False
    assert payload["raw_messages_captured"] == 0
    assert payload["authority"]["edge_claim"] is False
    assert payload["authority"]["profitability_claim"] is False
