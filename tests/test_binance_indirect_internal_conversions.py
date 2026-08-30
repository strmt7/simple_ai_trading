from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.screen_binance_indirect_internal_conversions import (
    build_edges,
    build_routes,
    evaluate_route,
    parse_books,
)


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION = ROOT / (
    "docs/model-research/action-value/"
    "binance-indirect-internal-conversion-activity-adjudication-v1-2026-08-29.json"
)
SPREAD_TRIAGE = ROOT / (
    "docs/model-research/action-value/"
    "binance-direct-indirect-spread-replication-triage-v1-2026-08-30.json"
)
SPREAD_TRIAGE_CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-direct-indirect-spread-replication-triage-contract-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_ADJUDICATION_HASH = (
    "0307a9dbfb26ca62e94ae01e5b5d40316340b686a60829e85f258c07e565678c"
)
EXPECTED_SPREAD_TRIAGE_HASH = (
    "ba9d063f78b027f6aab5e45723f5dc4ea2e9df1303de4f493e780ee07d4425b7"
)
EXPECTED_SPREAD_TRIAGE_CONTRACT_HASH = (
    "8844cbab660ac00fc119089bc81c0d56348ac2086285ee9296b0c4d80ff10151"
)
EXPECTED_REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _embedded_hash(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _symbol(
    symbol: str, base: str, quote: str, step: str = "0.001"
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
        "orderTypes": ["LIMIT", "MARKET"],
        "filters": [
            {"filterType": "MARKET_LOT_SIZE", "stepSize": step},
            {"filterType": "LOT_SIZE", "stepSize": step},
        ],
    }


def test_indirect_route_applies_extra_fee_rounding_capacity_and_stress() -> None:
    exchange_info = {
        "symbols": [
            _symbol("AUSDT", "A", "USDT"),
            _symbol("BUSDT", "B", "USDT"),
            _symbol("XUSDT", "X", "USDT"),
            _symbol("AB", "A", "B"),
            _symbol("AX", "A", "X"),
            _symbol("XB", "X", "B"),
        ]
    }
    books = parse_books(
        [
            {
                "symbol": "AUSDT",
                "bidPrice": "9.99",
                "bidQty": "1000",
                "askPrice": "10.01",
                "askQty": "1000",
            },
            {
                "symbol": "BUSDT",
                "bidPrice": "0.999",
                "bidQty": "10000",
                "askPrice": "1.001",
                "askQty": "10000",
            },
            {
                "symbol": "XUSDT",
                "bidPrice": "1.999",
                "bidQty": "10000",
                "askPrice": "2.001",
                "askQty": "10000",
            },
            {
                "symbol": "AB",
                "bidPrice": "9.90",
                "bidQty": "1000",
                "askPrice": "9.92",
                "askQty": "1000",
            },
            {
                "symbol": "AX",
                "bidPrice": "5.00",
                "bidQty": "1000",
                "askPrice": "5.01",
                "askQty": "1000",
            },
            {
                "symbol": "XB",
                "bidPrice": "2.00",
                "bidQty": "1000",
                "askPrice": "2.01",
                "askQty": "1000",
            },
        ]
    )
    edges = build_edges(exchange_info)
    route = next(
        route
        for route in build_routes(edges)
        if route[0].source == "A" and route[0].target == "B" and route[1].target == "X"
    )
    result = evaluate_route(
        route,
        start_usdt=Decimal("100"),
        edges=edges,
        books=books,
        fee_rate=Decimal("0.001"),
        stress_bips=Decimal("3"),
    )
    assert result is not None
    assert result["route_id"] == "A->X->B_vs_A->B"
    assert result["capacity_ok"] is True
    assert result["savings_bips"] > Decimal("0")
    assert result["stressed_savings_bips"] > Decimal("0")
    assert result["indirect_residual_bips"] <= Decimal("1")
    assert result["positive_after_gates"] is True


def test_activity_adjudication_accepts_only_the_fail_closed_mechanism() -> None:
    artifact = _load(ADJUDICATION)
    assert artifact["result_sha256"] == EXPECTED_ADJUDICATION_HASH
    assert _embedded_hash(artifact, "result_sha256") == EXPECTED_ADJUDICATION_HASH
    assert artifact["population"]["initial_empirical_route_size_candidates"] == 20
    assert artifact["population"]["activity_survivors"] == 3
    assert artifact["capture_level_gates"]["all_passed"] is True
    assert artifact["decision"]["market_direction_forecast_required"] is False
    assert artifact["decision"]["accepted_scoped_mechanism"] is True
    assert artifact["accepted_edge"] is True
    assert artifact["deployment_ready"] is False
    assert all(
        survivor["static_route_accepted"] is False
        for survivor in artifact["activity_survivors"]
    )


def test_registry_adds_the_indirect_organic_conversion_overlay_once() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry, "result_sha256") == EXPECTED_REGISTRY_HASH
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 45)
    )
    family = registry["prioritized_hypotheses"][-1]
    assert family["mechanism"] == (
        "binance_indirect_internal_conversion_route_savings_for_organic_flow"
    )
    assert family["market_direction_forecast_required"] is False
    assert EXPECTED_ADJUDICATION_HASH in {
        artifact["result_sha256"] for artifact in family["canonical_artifacts"]
    }


def test_standalone_route_spread_replication_stays_fail_closed() -> None:
    contract = _load(SPREAD_TRIAGE_CONTRACT)
    assert contract["contract_sha256"] == EXPECTED_SPREAD_TRIAGE_CONTRACT_HASH
    assert (
        _embedded_hash(contract, "contract_sha256")
        == EXPECTED_SPREAD_TRIAGE_CONTRACT_HASH
    )
    triage = _load(SPREAD_TRIAGE)
    assert triage["result_sha256"] == EXPECTED_SPREAD_TRIAGE_HASH
    assert _embedded_hash(triage, "result_sha256") == EXPECTED_SPREAD_TRIAGE_HASH
    assert triage["contract"]["sha256"] == EXPECTED_SPREAD_TRIAGE_CONTRACT_HASH
    assert triage["adjudication"]["accepted_edge"] is False
    assert triage["adjudication"]["book_or_account_request_justified"] is False
    assert triage["adjudication"]["public_after_all_cost_profit_floor_usdt"] == 0

    execution = triage["replication_evidence"]["execution_and_capacity"]
    assert execution["all_2025_profitable_paths_max_usd_per_trade"] == 25
    assert (
        execution["target_trade_minutes_with_zero_executable_thinnest_leg_percent"]
        == 78
    )
    assert execution["profitable_routes_checked_for_complete_perpetual_legs"] == 86
    assert execution["profitable_routes_with_perpetual_on_every_leg"] == 0
    assert all(
        row["profitable"] is False for row in execution["scalable_route_examples"]
    )

    registry = _load(REGISTRY)
    assert _embedded_hash(registry, "result_sha256") == registry["result_sha256"]
    family = registry["prioritized_hypotheses"][-1]
    assert EXPECTED_SPREAD_TRIAGE_HASH in {
        artifact["result_sha256"] for artifact in family["canonical_artifacts"]
    }
    terminals = {
        row["family"]: row["canonical_result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    }
    terminal_family = (
        "binance_direct_versus_indirect_route_standalone_statistical_spread_"
        "secondary_replication_2026_08_30"
    )
    assert terminals[terminal_family] == EXPECTED_SPREAD_TRIAGE_HASH
