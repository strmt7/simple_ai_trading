from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
ARTIFACT = ACTION_VALUE / ("polymarket-negrisk-taker-rebate-overlay-v1-2026-08-26.json")
NEGRISK = ACTION_VALUE / "polymarket-negrisk-maker-input-gate-v1-2026-08-26.json"
REBATE = ACTION_VALUE / ("polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json")
EXPECTED_HASH = "fbbaf4ff7a7d93f8cf5d306a829ff00518d82c9802be674fdace864cea907a60"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    assert isinstance(payload, dict)
    return payload


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _scenario_map(section: dict[str, object]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["tier"], row["quantity_shares"]): row for row in section["tier_scenarios"]
    }


def test_overlay_and_both_predecessors_are_exactly_hash_bound() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH

    predecessors = {row["path"]: row for row in artifact["predecessors"]}
    for path, expected in (
        (NEGRISK, predecessors[NEGRISK.relative_to(ROOT).as_posix()]),
        (REBATE, predecessors[REBATE.relative_to(ROOT).as_posix()]),
    ):
        payload = _load(path)
        assert payload["result_sha256"] == expected["result_sha256"]
        assert _embedded_hash(payload) == expected["result_sha256"]


def test_maximum_rebate_cannot_rescue_the_queue_free_all_taker_path() -> None:
    artifact = _load(ARTIFACT)
    negrisk = _load(NEGRISK)
    books = {
        row["asset"]: row for row in negrisk["sole_positive_live_state"]["top_books"]
    }
    rate = Decimal(
        negrisk["current_fee_and_conversion_contract"]["clob_fee_schedule"]["rate"]
    )
    prices = [
        Decimal(books["Bitcoin NO"]["ask_price"]),
        Decimal(books["Gold YES"]["bid_price"]),
        Decimal(books["S&P 500 YES"]["bid_price"]),
    ]

    with localcontext() as context:
        context.prec = 50
        gross_per_share = prices[1] + prices[2] - prices[0]
        fee_per_share = sum(rate * price * (Decimal(1) - price) for price in prices)
        break_even = Decimal(1) - gross_per_share / fee_per_share
        section = artifact["economics"]["all_taker"]
        assert gross_per_share == Decimal(section["gross_before_fee_per_share_pusd"])
        assert fee_per_share == Decimal(section["taker_fee_per_share_pusd"])
        assert break_even == Decimal(
            section["break_even_rebate_fraction_before_external_costs"]
        )
        assert break_even > Decimal(section["maximum_documented_rebate_fraction"])

        scenarios = _scenario_map(section)
        for tier, rebate in (("Gold", Decimal("0.18")), ("Obsidian", Decimal("0.50"))):
            for quantity in (Decimal(5), Decimal(20)):
                expected = quantity * (
                    gross_per_share - fee_per_share * (Decimal(1) - rebate)
                )
                row = scenarios[(tier, str(quantity))]
                assert expected == Decimal(row["net_pusd_before_external_costs"])
                assert expected < 0


def test_maker_input_rebate_improves_margin_without_proving_after_cost_profit() -> None:
    artifact = _load(ARTIFACT)
    negrisk = _load(NEGRISK)
    books = {
        row["asset"]: row for row in negrisk["sole_positive_live_state"]["top_books"]
    }
    rate = Decimal(
        negrisk["current_fee_and_conversion_contract"]["clob_fee_schedule"]["rate"]
    )
    input_price = Decimal(books["Bitcoin NO"]["bid_price"])
    output_prices = [
        Decimal(books["Gold YES"]["bid_price"]),
        Decimal(books["S&P 500 YES"]["bid_price"]),
    ]
    gross_per_share = sum(output_prices) - input_price
    fee_per_share = sum(rate * price * (Decimal(1) - price) for price in output_prices)
    section = artifact["economics"]["maker_input_then_taker_outputs"]
    assert gross_per_share == Decimal(section["gross_before_fee_per_share_pusd"])
    assert fee_per_share == Decimal(section["taker_output_fee_per_share_pusd"])

    base_sensitivity = {
        "5": Decimal(
            negrisk["conversion_access_and_latency"][
                "current_whole_transaction_gas_sensitivity"
            ]["margin_5_pusd_minus_usdt_sensitivity"]
        ),
        "20": Decimal(
            negrisk["conversion_access_and_latency"][
                "current_whole_transaction_gas_sensitivity"
            ]["margin_20_pusd_minus_usdt_sensitivity"]
        ),
    }
    scenarios = _scenario_map(section)
    for tier, rebate in (("Gold", Decimal("0.18")), ("Obsidian", Decimal("0.50"))):
        for quantity in (Decimal(5), Decimal(20)):
            row = scenarios[(tier, str(quantity))]
            improvement = quantity * fee_per_share * rebate
            net = quantity * (gross_per_share - fee_per_share * (Decimal(1) - rebate))
            assert improvement == Decimal(row["rebate_improvement_pusd"])
            assert net == Decimal(row["net_pusd_before_external_costs"])
            assert base_sensitivity[str(quantity)] + improvement == Decimal(
                row["predecessor_direct_gas_sensitivity_after_rebate_pusd_minus_usdt"]
            )

    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["adjudication"]["trading_authority"] is False
    assert artifact["research_decision"]["accepted_edge_count_change"] == 0


def test_protected_capture_contract_is_unchanged_and_not_consumed() -> None:
    artifact = _load(ARTIFACT)
    negrisk = _load(NEGRISK)
    protected = artifact["protected_capture"]
    original = negrisk["prospective_fill_and_unwind_capture"]
    assert protected["contract_internal_sha256"] == original["contract_internal_sha256"]
    assert protected["result_status"] == original["result_status"]
    assert "Do not restart, duplicate, or consume" in protected["rule"]
    assert artifact["verification"]["new_market_or_account_requests"] == 0
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_conversions_submitted"] == 0
