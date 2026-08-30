from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import screen_binance_options_clob_box_prefilter_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-contract-v2.json"
)
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-v2-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "806b99257dd081fddef2fcaa5657776e9dfecee65dd52da1bba351052a062e81"
RESULT_HASH = "a9b0e7a2aba9bda7f83b9515be587a17e6da69fa0bc987191a21f9d37e912d3b"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_v2_contract_and_implementation_lineage_reconstruct() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["erratum"]["missing_closeTime_ticker_count"] == 397
    assert contract["erratum"]["new_market_access"] is False
    for implementation in (
        contract["implementation"],
        contract["implementation"]["dependency"],
    ):
        payload = (ROOT / implementation["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == implementation["sha256"]


def test_missing_close_time_fails_synchronization_proxy() -> None:
    quote = v2._quote({"bidPrice": "1", "askPrice": "2"}, capture_completed_at_ms=1000)

    assert quote["close_time_ms"] is None
    assert quote["age_ms"] == -1
    assert not (0 <= int(quote["age_ms"]) <= 60_000)


def test_retained_prefilter_is_terminal_without_current_access() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["population"] == {
        "eligible_option_symbol_count": 1410,
        "underlying_expiry_group_count": 22,
        "evaluated_box_direction_count": 10382,
        "gross_positive_count": 0,
        "synchronized_count": 0,
        "synchronized_gross_positive_count": 0,
    }
    assert result["top_optimistic_gross_positive_rows"] == []
    assert result["adjudication"]["next_action"] == (
        "stop_without_current_market_requests"
    )
    assert result["authority"]["new_public_requests"] == 0
    assert result["authority"]["credentials_used"] is False


def test_existing_terminal_family_is_updated_without_duplicate_rank() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert len(registry["prioritized_hypotheses"]) == 44
    terminal = [
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "binance_option_box_parity"
    ]
    assert len(terminal) == 1
    assert terminal[0]["canonical_result_sha256"] == RESULT_HASH
    assert "10382" in terminal[0]["reason"]
