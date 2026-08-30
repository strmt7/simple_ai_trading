from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_binance_all_symbol_triangular_cycles_retained import (
    build_cycles,
    evaluate_cycle,
)
from tools.screen_binance_indirect_internal_conversions import build_edges, parse_books


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-all-symbol-triangular-cycle-retained-v1-2026-08-29.json"
)
ADJUDICATION = ROOT / (
    "docs/model-research/action-value/"
    "binance-all-symbol-triangular-cycle-retained-adjudication-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RESULT_HASH = "30c5e00aa955ea3777f9b096b1fa1ae44d51318665561e4b6922f797f45706cc"
ADJUDICATION_HASH = "2fffd2044e72d1712ecdaa0c4e24cb829057ea2005c07e12129c443478b07902"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _embedded_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    payload = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _symbol(symbol: str, base: str, quote: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
        "orderTypes": ["MARKET"],
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.000001"}],
    }


def test_cycle_is_positive_only_in_the_zero_fee_upper_bound() -> None:
    exchange_info = {
        "symbols": [
            _symbol("AUSDT", "A", "USDT"),
            _symbol("BUSDT", "B", "USDT"),
            _symbol("CUSDT", "C", "USDT"),
            _symbol("AB", "A", "B"),
            _symbol("BC", "B", "C"),
            _symbol("CA", "C", "A"),
        ]
    }
    books = parse_books(
        [
            {
                "symbol": "AUSDT",
                "bidPrice": "1",
                "bidQty": "10000",
                "askPrice": "1.001",
                "askQty": "10000",
            },
            {
                "symbol": "BUSDT",
                "bidPrice": "0.5",
                "bidQty": "10000",
                "askPrice": "0.501",
                "askQty": "10000",
            },
            {
                "symbol": "CUSDT",
                "bidPrice": "0.25",
                "bidQty": "10000",
                "askPrice": "0.251",
                "askQty": "10000",
            },
            {
                "symbol": "AB",
                "bidPrice": "2",
                "bidQty": "10000",
                "askPrice": "2.001",
                "askQty": "10000",
            },
            {
                "symbol": "BC",
                "bidPrice": "2",
                "bidQty": "10000",
                "askPrice": "2.001",
                "askQty": "10000",
            },
            {
                "symbol": "CA",
                "bidPrice": "0.2507",
                "bidQty": "10000",
                "askPrice": "0.2508",
                "askQty": "10000",
            },
        ]
    )
    edges = build_edges(exchange_info)
    cycle = next(
        row
        for row in build_cycles(edges)
        if [edge.source for edge in row] == ["A", "B", "C"]
    )
    zero_fee = evaluate_cycle(
        cycle,
        start_usdt=Decimal("1000"),
        edges=edges,
        books=books,
        fee_rate=Decimal(0),
        stress_bips=Decimal(3),
    )
    vip0 = evaluate_cycle(
        cycle,
        start_usdt=Decimal("1000"),
        edges=edges,
        books=books,
        fee_rate=Decimal("0.001"),
        stress_bips=Decimal(3),
    )
    assert zero_fee is not None and vip0 is not None
    assert zero_fee["capacity_ok"] is True
    assert zero_fee["positive_after_gates"] is True
    assert zero_fee["stressed_profit_bips"] > 0
    assert vip0["positive_after_gates"] is False
    assert vip0["stressed_profit_bips"] < 0


def test_retained_population_is_terminal_before_fees() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["result_sha256"] == RESULT_HASH
    assert _embedded_hash(result, "result_sha256") == RESULT_HASH
    assert result["unique_directed_cycles"] == 3480
    assert result["activity_qualified_cycles"] == 1442
    assert result["zero_fee_upper_bound_candidate_count"] == 0
    assert result["vip0_all_taker_candidate_count"] == 0
    assert result["accepted_edge"] is False


def test_adjudication_discloses_ranking_defect_and_updates_existing_family() -> None:
    adjudication = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    assert adjudication["result_sha256"] == ADJUDICATION_HASH
    assert _embedded_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    assert adjudication["decision"]["accepted_edge"] is False
    assert adjudication["post_consumption_defect_disclosure"]["impact"].startswith(
        "diagnostic ranking only"
    )

    registry = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _embedded_hash(registry, "result_sha256") == REGISTRY_HASH
    families = registry["prioritized_hypotheses"]
    triangle = next(row for row in families if row["priority_rank"] == 16)
    assert triangle["mechanism"] == "three_leg_spot_conversion"
    assert triangle["canonical_artifacts"][-1]["result_sha256"] == (ADJUDICATION_HASH)
    assert registry["terminal_do_not_repeat"][0]["canonical_result_sha256"] == (
        ADJUDICATION_HASH
    )
