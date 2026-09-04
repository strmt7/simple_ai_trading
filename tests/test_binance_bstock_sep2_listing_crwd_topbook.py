from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
INVENTORY = (
    ACTION_VALUE / "binance-bstock-sep2-listing-inventory-result-v2-2026-09-04.json"
)
TOPBOOK = (
    ACTION_VALUE
    / "binance-crwdb-bstock-perpetual-topbook-result-v1-2026-09-04.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION_VALUE / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_listing_delta_selects_only_the_first_exact_match() -> None:
    result = _load(INVENTORY)
    body = dict(result)
    claimed = body.pop("result_sha256")

    assert hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == claimed
    assert result["baseline_row_count"] == 68
    assert result["current_row_count"] == 72
    assert result["new_tickers"] == ["CRWD", "MRNA", "SQQQ", "STX"]
    assert [row["ticker"] for row in result["matching_unscreened_pairs"]] == [
        "CRWD",
        "MRNA",
        "SQQQ",
    ]
    assert result["next_selected_ticker"] == "CRWD"
    assert result["request_count"] == 2


def test_crwd_zero_gross_basis_stops_before_depth_and_funding() -> None:
    result = _load(TOPBOOK)

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["capture"]["capture_skew_ms"] == 262
    economics = result["economics"]
    assert economics["spot_ask_USDT_per_share"] == "213.07000000"
    assert economics["futures_bid_USDT_per_share"] == "213.07000"
    assert economics["gross_entry_headroom_USDT_per_share"] == "0.00000000"
    assert economics["after_fixed_stress_USDT_per_share"] == "-1.06535000"
    assert economics["after_fixed_stress_bps"] == "-50.000"
    assert economics["passes_fixed_rejection_gate"] is False
    assert result["adjudication"]["depth_requests"] == 0
    assert result["adjudication"]["funding_requests"] == 0


def test_ledgers_preserve_unscreened_matches_and_future_trigger() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert registry["result_sha256"] == (
        "c6f6338f7d3b42baa2f6f61bbcf0b1433aff46cec541d9b3dbac5d6387ae0e59"
    )
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 65
    assert len(registry["terminal_do_not_repeat"]) == 184
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "bstock_reference_conversion_and_delta_neutral_perpetual_funding"
    )
    assert hypothesis["priority_rank"] == 12
    assert {
        "path": TOPBOOK.relative_to(ROOT).as_posix(),
        "result_sha256": _load(TOPBOOK)["result_sha256"],
    } in hypothesis["canonical_artifacts"]
    assert "2026_09_08T13_35_00Z" in hypothesis["next_action"]
    assert "MRNABUSDT_MRNAUSDT" in hypothesis["next_action"]
    assert audit["source_binding"]["registry_result_sha256"] == registry[
        "result_sha256"
    ]
    assert audit["result_sha256"] == (
        "ed029fb924466e9f937c6fcfb499387618209062f7cdce8645eedcf065545922"
    )
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
