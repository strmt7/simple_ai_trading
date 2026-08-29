from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
WRAPPER = ACTION_VALUE / (
    "binance-ondo-bstock-stock-perpetual-wrapper-parity-candidate-v1-2026-08-27.json"
)
NOK = ACTION_VALUE / (
    "binance-nok-bstock-dividend-perpetual-underdebit-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
WRAPPER_HASH = "8bcf6f7bfa0cca6dab1fd6fd854a331d5ee41366ac6f9c0244b62a8f3545f475"
NOK_HASH = "79118e0e9a32a17d0d79040746068b94e6ec545179958a29dc45f3b8771434bb"
REGISTRY_HASH = "f9bb0f6582fca306d3083a8ad3aadeaa020936949702176b585ea6f926e87e08"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_wrapper_candidate_is_hash_bound_public_only_and_unaccepted() -> None:
    artifact = _load(WRAPPER)

    assert artifact["result_sha256"] == WRAPPER_HASH
    assert _canonical_hash(artifact) == WRAPPER_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "public_after_cost_profit_floor_usdt": "0",
        "status": (
            "materially_new_direction_independent_three_wrapper_parity_candidate_"
            "with_exact_multiplier_normalization_but_no_executable_Ondo_quote_"
            "after_cost_recurrence_or_trading_authority"
        ),
        "trading_authority": False,
    }
    authority = artifact["authority"]
    assert authority["authenticated_requests"] == 0
    assert authority["quote_requests_sent"] == 0
    assert authority["orders_or_transfers_submitted"] == 0
    assert authority["quote_client_installed"] is False


def test_wrapper_screen_normalizes_multiplier_and_labels_point_gaps() -> None:
    artifact = _load(WRAPPER)
    capture = artifact["capture"]
    rows = artifact["exact_point_screen_top_ten"]

    assert capture["exact_three_way_overlap_count"] == 60
    assert capture["positive_point_gap_count"] == 41
    assert capture["point_gap_at_least_10_bps_count"] == len(rows) == 10
    assert artifact["sources"]["dynamic_response_manifest"]["record_count"] == 60
    for row in rows:
        token_price = Decimal(row["ondo_token_price"])
        multiplier = Decimal(row["ondo_shares_multiplier"])
        reference = Decimal(row["ondo_reference_price_per_share"])
        best_bid = max(Decimal(row["bstock_bid"]), Decimal(row["perpetual_bid"]))
        gap = Decimal(row["gross_point_gap_usdt_per_share"])
        bps = Decimal(row["gross_point_gap_bps"])
        assert abs(token_price / multiplier - reference) <= Decimal("0.000001")
        assert abs(best_bid - reference - gap) <= Decimal("0.000001")
        assert abs((gap / reference * Decimal("10000")) - bps) <= Decimal("0.0001")
    assert "not_an_executable_buy_ask" in artifact["economic_identity"][
        "why_point_gap_is_not_profit"
    ]


def test_nok_candidate_proves_underdebit_only_as_a_gross_upper_bound() -> None:
    artifact = _load(NOK)

    assert artifact["result_sha256"] == NOK_HASH
    assert _canonical_hash(artifact) == NOK_HASH
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["authority"]["retained_public_http_responses"] == 10
    rows = artifact["historical_funding"]["snapshot_rows"]
    short_cashflow = sum(
        Decimal(row["funding_rate"]) * Decimal(row["mark_price"]) for row in rows
    )
    assert short_cashflow == Decimal("-0.008865103000")
    gross = Decimal(artifact["economic_contract"]["gross_dividend_upper_bound_usd_per_share"])
    upper_headroom = Decimal(
        artifact["economic_contract"][
            "snapshot_funding_gross_upper_headroom_before_all_other_costs"
        ]
    )
    assert gross + short_cashflow == upper_headroom
    assert artifact["economic_contract"]["public_net_distribution_floor_usdt"] == "0"
    multiplier = artifact["current_multiplier_evidence"]
    assert Decimal(multiplier["bsc_eth_call_uiMultiplier_raw_uint256"]) / Decimal(
        "1000000000000000000"
    ) == Decimal(multiplier["binance_dynamic_shares_multiplier"])


def test_registry_adds_both_candidates_without_promoting_an_edge() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    by_mechanism = {row["mechanism"]: row for row in hypotheses}
    assert by_mechanism[
        "binance_Ondo_bStock_stock_perpetual_exact_multiplier_wrapper_parity"
    ]["priority_rank"] == 35
    assert by_mechanism[
        "binance_NOK_bStock_dividend_perpetual_special_funding_underdebit"
    ]["priority_rank"] == 36
    assert "Binance Square" in registry["accepted_edge_scope"]
    assert "Binance Referral Pro" in registry["accepted_edge_scope"]
