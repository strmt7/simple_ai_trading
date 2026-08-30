from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-pusd-to-binance-usdt-opportunity-cost-contract-v1-2026-08-30.json"
)
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-pusd-to-binance-usdt-opportunity-cost-gate-v1-2026-08-30.json"
)
JOURNAL = (
    ROOT / "data/polymarket-pusd-to-binance-usdt-opportunity-cost-v1/journal.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_opportunity_cost_quote_reconstructs_and_rejects_before_account_access() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    implementation = contract["implementation"]
    assert _sha256((ROOT / implementation["path"]).read_bytes()) == implementation[
        "sha256"
    ]
    for source in contract["retained_sources"]:
        assert _sha256((ROOT / source["path"]).read_bytes()) == source["sha256"]

    assert journal["state"] == "completed"
    assert journal["contract_sha256"] == contract["contract_sha256"]
    assert journal["result_sha256"] == result["result_sha256"]
    assert len(journal["requests"]) == contract["authority"]["max_public_requests"] == 4
    assert [row["method"] for row in journal["requests"]] == [
        "GET",
        "GET",
        "GET",
        "POST",
    ]
    for row in journal["requests"]:
        raw = (ROOT / row["raw_path"]).read_bytes()
        assert row["state"] == "received"
        assert row["status_code"] == 200
        assert len(raw) == row["response_bytes"]
        assert _sha256(raw) == row["response_sha256"]

    assets = json.loads(
        (
            ROOT
            / "data/polymarket-pusd-to-binance-usdt-opportunity-cost-v1/raw/"
            "supported-assets.raw.json"
        ).read_text(encoding="utf-8")
    )
    matches = [
        row
        for row in assets["supportedAssets"]
        if str(row["chainId"]) == "137" and row["token"]["symbol"] == "USDT"
    ]
    assert len(matches) == 1
    quote_body = {
        **contract["quote_request"],
        "toTokenAddress": matches[0]["token"]["address"],
    }
    assert journal["requests"][-1]["request_body_sha256"] == _sha256(
        _canonical(quote_body)
    )

    quote = json.loads(
        (
            ROOT
            / "data/polymarket-pusd-to-binance-usdt-opportunity-cost-v1/raw/"
            "quote.raw.json"
        ).read_text(encoding="utf-8")
    )
    amount = Decimal(contract["amount_pusd"])
    output = Decimal(quote["estToTokenBaseUnit"]) / Decimal(10**6)
    loss = amount - output
    assert Decimal(result["route"]["estimated_output_usdt"]) == output
    assert Decimal(result["route"]["estimated_conversion_loss_usdt"]) == loss
    assert Decimal(result["route"]["estimated_conversion_loss_bips"]) == (
        loss / amount * Decimal(10_000)
    )
    assert loss > Decimal(result["economics"]["optimistic_full_fixed_bonus_reward_usdt"])
    assert Decimal(
        result["economics"]["optimistic_full_fixed_bonus_reward_usdt"]
    ) > Decimal(
        result["economics"]["optimistic_incremental_reward_before_external_cost"]
    )
    assert result["adjudication"] == {
        "accepted_edge": False,
        "accepted_edge_count_change": 0,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "next_trigger": (
            "material_bridge_quote_or_bonus_term_change;_otherwise_do_not_"
            "move_pUSD_for_this_bonus_and_do_not_repeat_the_quote"
        ),
        "pUSD_to_USDT_one_to_one_assumed": False,
        "profitable_switch_proved": False,
        "reason": (
            "the_optimistic_one_way_executable_conversion_estimate_alone_"
            "exceeds_the_entire_remaining_fixed_bonus_before_charging_"
            "holding_yield_opportunity_cost_deposit_return_conversion_"
            "eligibility_capacity_custody_tax_or_operating_cost"
        ),
        "rejected_before_account_access": True,
        "status": "rejected_before_account_access_by_one_way_conversion_cost",
    }
    assert _self_hash(result, "result_sha256") == result["result_sha256"]


def test_opportunity_cost_result_is_routed_into_rank_one_registry_lineage() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_one = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 1
    )
    assert {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in rank_one["canonical_artifacts"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_pUSD_to_Binance_USDT_fixed_bonus_opportunity_cost_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
