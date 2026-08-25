from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_maker_rebates import (
    filled_maker_rebate_economics,
    paired_filled_maker_rebate_economics,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "crypto-maker-rebate-economics-v1.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def test_crypto_maker_rebate_matches_documented_unrounded_algebra() -> None:
    result = filled_maker_rebate_economics(
        quantity=Decimal("50"),
        price=Decimal("0.49"),
        taker_fee_rate=Decimal("0.07"),
        rebate_fraction=Decimal("0.20"),
    )

    assert result.fee_equivalent == Decimal("0.874650")
    assert result.nominal_maker_rebate == Decimal("0.1749300")


def test_paired_fill_combines_complete_set_spread_and_nominal_rebates() -> None:
    result = paired_filled_maker_rebate_economics(
        up_price=Decimal("0.49"),
        down_price=Decimal("0.49"),
        quantity=Decimal("50"),
        taker_fee_rate=Decimal("0.07"),
        rebate_fraction=Decimal("0.20"),
    )

    assert result.settlement.both_fill_gross_profit == Decimal("1.00")
    assert result.nominal_total_maker_rebate == Decimal("0.3498600")
    assert result.nominal_both_fill_profit_including_rebates == Decimal("1.3498600")
    assert result.settlement.maximum_orphan_loss == Decimal("24.50")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantity": Decimal("0")}, "positive"),
        ({"price": Decimal("1")}, "inside"),
        ({"taker_fee_rate": Decimal("1.1")}, "exceeds"),
        ({"rebate_fraction": Decimal("1.1")}, "exceeds"),
        ({"quantity": True}, "finite"),
        ({"quantity": object()}, "finite"),
        ({"quantity": Decimal("NaN")}, "finite"),
    ],
)
def test_filled_maker_rebate_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    inputs: dict[str, object] = {
        "quantity": Decimal("50"),
        "price": Decimal("0.49"),
        "taker_fee_rate": Decimal("0.07"),
        "rebate_fraction": Decimal("0.20"),
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        filled_maker_rebate_economics(**inputs)  # type: ignore[arg-type]


def test_paired_rebate_requires_non_crossing_physical_bids() -> None:
    with pytest.raises(ValueError, match="below one"):
        paired_filled_maker_rebate_economics(
            up_price=Decimal("0.50"),
            down_price=Decimal("0.50"),
            quantity=Decimal("50"),
            taker_fee_rate=Decimal("0.07"),
            rebate_fraction=Decimal("0.20"),
        )


def test_published_economics_reconstructs_hash_implementation_and_example() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    claimed_hash = artifact.pop("result_sha256")
    implementation = artifact["implementation"]
    module_path = ROOT / implementation["module_path"]

    assert hashlib.sha256(_canonical_json(artifact).encode("ascii")).hexdigest() == (
        claimed_hash
    )
    assert (
        hashlib.sha256(module_path.read_bytes()).hexdigest()
        == implementation["module_sha256"]
    )
    assert artifact["authority"] == {
        "accepted_edge": False,
        "credentials_used": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "publicly_proven_rebate_payout_lower_bound": "0",
    }

    documented = artifact["documented_contract"]
    example = artifact["example"]
    reconstructed = paired_filled_maker_rebate_economics(
        up_price=Decimal(example["up_bid_price"]),
        down_price=Decimal(example["down_bid_price"]),
        quantity=Decimal(example["paired_quantity"]),
        taker_fee_rate=Decimal(documented["crypto_taker_fee_rate"]),
        rebate_fraction=Decimal(documented["crypto_maker_rebate_fraction"]),
    )

    assert reconstructed.up_fill.fee_equivalent == Decimal(
        example["conditional_up_fill_fee_equivalent"]
    )
    assert reconstructed.up_fill.nominal_maker_rebate == Decimal(
        example["conditional_up_fill_nominal_rebate"]
    )
    assert reconstructed.down_fill.fee_equivalent == Decimal(
        example["conditional_down_fill_fee_equivalent"]
    )
    assert reconstructed.down_fill.nominal_maker_rebate == Decimal(
        example["conditional_down_fill_nominal_rebate"]
    )
    assert reconstructed.nominal_total_maker_rebate == Decimal(
        example["conditional_nominal_total_maker_rebate"]
    )
    assert reconstructed.nominal_both_fill_profit_including_rebates == Decimal(
        example["conditional_nominal_both_fill_profit_including_rebates"]
    )
    assert reconstructed.settlement.both_fill_gross_profit == Decimal(
        example["settlement_both_fill_gross_profit"]
    )
    assert reconstructed.settlement.maximum_orphan_loss == Decimal(
        example["maximum_orphan_settlement_loss_without_rebate_credit"]
    )
    assert len(artifact["sources"]) == 2
    assert all(
        len(source["retrieved_content_sha256"]) == 64 for source in artifact["sources"]
    )
