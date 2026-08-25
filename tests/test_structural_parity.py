from decimal import Decimal

import pytest

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.structural_parity import (
    NegativeRiskOutcome,
    SpotPairQuote,
    screen_spot_triangles,
    screen_negative_risk_parity,
    walk_structural_parity_depth,
)


ZERO_FEE = PolymarketFeeModel(
    enabled=False,
    rate=Decimal("0"),
    exponent=1,
    taker_only=True,
)


def _levels(*rows: tuple[str, str]) -> tuple[BookLevel, ...]:
    return tuple(BookLevel(Decimal(price), Decimal(size)) for price, size in rows)


def _outcome(
    label: str,
    *,
    yes_bid: str,
    yes_ask: str,
    no_ask: str,
    fee_model: PolymarketFeeModel = ZERO_FEE,
) -> NegativeRiskOutcome:
    return NegativeRiskOutcome(
        label=label,
        yes_bids=_levels((yes_bid, "100")),
        yes_asks=_levels((yes_ask, "100")),
        no_asks=_levels((no_ask, "100")),
        fee_model=fee_model,
    )


def test_depth_walk_does_not_invent_missing_quantity() -> None:
    assert (
        walk_structural_parity_depth(
            _levels(("0.20", "2"), ("0.30", "2")),
            quantity=Decimal("5"),
            fee_model=ZERO_FEE,
        )
        is None
    )


def test_all_yes_identity_breaks_even_before_fees() -> None:
    screen = screen_negative_risk_parity(
        (
            _outcome("A", yes_bid="0.19", yes_ask="0.20", no_ask="0.81"),
            _outcome("B", yes_bid="0.29", yes_ask="0.30", no_ask="0.71"),
            _outcome("C", yes_bid="0.49", yes_ask="0.50", no_ask="0.51"),
        ),
        quantity=Decimal("5"),
        conversion_fee_bips=0,
    )

    assert screen.buy_all_yes_hold is not None
    assert screen.buy_all_yes_hold.net_quote == 0
    assert screen.mint_all_yes_sell is not None
    assert screen.mint_all_yes_sell.net_quote == Decimal("-0.15")
    assert screen.profitable_path_count == 0


def test_dynamic_taker_fees_turn_gross_parity_negative() -> None:
    fee = PolymarketFeeModel(
        enabled=True,
        rate=Decimal("0.04"),
        exponent=1,
        taker_only=True,
    )
    screen = screen_negative_risk_parity(
        (
            _outcome(
                "Bitcoin",
                yes_bid="0.16",
                yes_ask="0.17",
                no_ask="0.84",
                fee_model=fee,
            ),
            _outcome(
                "Gold",
                yes_bid="0.27",
                yes_ask="0.28",
                no_ask="0.73",
                fee_model=fee,
            ),
            _outcome(
                "S&P 500",
                yes_bid="0.53",
                yes_ask="0.55",
                no_ask="0.47",
                fee_model=fee,
            ),
        ),
        quantity=Decimal("5"),
        conversion_fee_bips=0,
    )

    assert screen.buy_all_yes_hold is not None
    assert screen.buy_all_yes_hold.net_quote == Decimal("-0.11804")
    assert screen.buy_all_yes_hold.taker_fees_quote == Decimal("0.11804")


def test_no_conversion_checks_every_subset_and_finds_best_path() -> None:
    screen = screen_negative_risk_parity(
        (
            _outcome("A", yes_bid="0.09", yes_ask="0.11", no_ask="0.10"),
            _outcome("B", yes_bid="0.39", yes_ask="0.41", no_ask="0.61"),
            _outcome("C", yes_bid="0.39", yes_ask="0.41", no_ask="0.61"),
        ),
        quantity=Decimal("5"),
        conversion_fee_bips=0,
    )

    assert screen.evaluated_path_count == 9
    assert screen.executable_path_count == 9
    assert screen.best_no_conversion is not None
    assert screen.best_no_conversion.selected_no_outcomes == ("A",)
    assert screen.best_no_conversion.net_quote == Decimal("3.40")


def test_nonzero_conversion_fee_fails_closed() -> None:
    outcomes = (
        _outcome("A", yes_bid="0.19", yes_ask="0.20", no_ask="0.81"),
        _outcome("B", yes_bid="0.29", yes_ask="0.30", no_ask="0.71"),
        _outcome("C", yes_bid="0.49", yes_ask="0.50", no_ask="0.51"),
    )

    with pytest.raises(ValueError, match="separately verified model"):
        screen_negative_risk_parity(
            outcomes,
            quantity=Decimal("5"),
            conversion_fee_bips=1,
        )


def test_outcome_rejects_unsorted_depth() -> None:
    outcome = NegativeRiskOutcome(
        label="A",
        yes_bids=_levels(("0.20", "5"), ("0.30", "5")),
        yes_asks=_levels(("0.40", "5")),
        no_asks=_levels(("0.60", "5")),
        fee_model=ZERO_FEE,
    )

    with pytest.raises(ValueError, match="not price-sorted"):
        outcome.validated()


@pytest.mark.parametrize("quantity", [True, "bad", "Infinity", "0"])
def test_depth_walk_rejects_invalid_quantity(quantity: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        walk_structural_parity_depth(
            _levels(("0.20", "5")),
            quantity=quantity,  # type: ignore[arg-type]
            fee_model=ZERO_FEE,
        )


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (
            NegativeRiskOutcome("", (), (), (), ZERO_FEE),
            "label is invalid",
        ),
        (
            NegativeRiskOutcome(
                "A",
                _levels(("0.20", "5"), ("0.20", "6")),
                _levels(("0.40", "5")),
                _levels(("0.60", "5")),
                ZERO_FEE,
            ),
            "prices are duplicated",
        ),
        (
            NegativeRiskOutcome(
                "A",
                (),
                _levels(("1", "5")),
                _levels(("0.60", "5")),
                ZERO_FEE,
            ),
            "price must lie below one",
        ),
        (
            NegativeRiskOutcome(
                "A",
                _levels(("0.40", "5")),
                _levels(("0.40", "5")),
                _levels(("0.60", "5")),
                ZERO_FEE,
            ),
            "crossed or locked",
        ),
    ],
)
def test_outcome_validation_rejects_ambiguous_books(
    outcome: NegativeRiskOutcome, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        outcome.validated()


def _spot_quote(
    symbol: str,
    base: str,
    quote: str,
    *,
    bid: str,
    ask: str,
) -> SpotPairQuote:
    return SpotPairQuote(
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        bid_price=Decimal(bid),
        bid_quantity=Decimal("100"),
        ask_price=Decimal(ask),
        ask_quantity=Decimal("100"),
    )


def test_triangle_screen_separates_gross_dislocation_from_account_fee() -> None:
    screen = screen_spot_triangles(
        (
            _spot_quote("AB", "A", "B", bid="1", ask="1.0001"),
            _spot_quote("BC", "B", "C", bid="1", ask="1.0001"),
            _spot_quote("AC", "A", "C", bid="0.9993", ask="0.9994"),
        ),
        start_assets=("A",),
        taker_fee_bips_per_leg=Decimal("10"),
    )

    assert screen.evaluated_path_count == 2
    assert screen.gross_positive_path_count == 1
    assert screen.after_fee_positive_path_count == 0
    assert screen.best_gross_path is not None
    assert screen.best_gross_path.assets == ("A", "B", "C", "A")
    assert screen.best_gross_path.gross_net_bips > Decimal("6")
    assert screen.best_gross_path.break_even_fee_bips_per_leg < Decimal("3")


def test_triangle_screen_rejects_duplicate_pair_orientation() -> None:
    quotes = (
        _spot_quote("AB", "A", "B", bid="1", ask="1.1"),
        _spot_quote("BA", "B", "A", bid="0.8", ask="0.9"),
    )

    with pytest.raises(ValueError, match="duplicated in another orientation"):
        screen_spot_triangles(
            quotes,
            start_assets=("A",),
            taker_fee_bips_per_leg=Decimal("0"),
        )


def test_triangle_input_gates_and_disconnected_graph() -> None:
    valid = _spot_quote("AB", "A", "B", bid="1", ask="1.1")
    invalid_identity = _spot_quote("A-B", "A", "B", bid="1", ask="1.1")
    crossed = _spot_quote("AB", "A", "B", bid="1.1", ask="1.1")

    with pytest.raises(ValueError, match="identity is invalid"):
        invalid_identity.validated()
    with pytest.raises(ValueError, match="crossed or locked"):
        crossed.validated()
    with pytest.raises(ValueError, match="outside"):
        screen_spot_triangles(
            (valid,), start_assets=("A",), taker_fee_bips_per_leg=Decimal("-1")
        )
    with pytest.raises(ValueError, match="nonempty and unique"):
        screen_spot_triangles(
            (), start_assets=("A",), taker_fee_bips_per_leg=Decimal("0")
        )
    with pytest.raises(ValueError, match="start assets are invalid"):
        screen_spot_triangles(
            (valid,), start_assets=(), taker_fee_bips_per_leg=Decimal("0")
        )

    disconnected = screen_spot_triangles(
        (
            valid,
            _spot_quote("BC", "B", "C", bid="1", ask="1.1"),
        ),
        start_assets=("MISSING", "A"),
        taker_fee_bips_per_leg=Decimal("0"),
    )
    assert disconnected.evaluated_path_count == 0
    assert disconnected.best_gross_path is None


def test_negative_risk_input_gates_and_missing_paths() -> None:
    valid = (
        _outcome("A", yes_bid="0.19", yes_ask="0.20", no_ask="0.81"),
        _outcome("B", yes_bid="0.29", yes_ask="0.30", no_ask="0.71"),
        _outcome("C", yes_bid="0.49", yes_ask="0.50", no_ask="0.51"),
    )
    with pytest.raises(ValueError, match="must be an integer"):
        screen_negative_risk_parity(
            valid,
            quantity=Decimal("5"),
            conversion_fee_bips=True,
        )
    with pytest.raises(ValueError, match="unique exhaustive outcomes"):
        screen_negative_risk_parity(
            (valid[0], valid[0], valid[2]),
            quantity=Decimal("5"),
            conversion_fee_bips=0,
        )

    missing = tuple(
        NegativeRiskOutcome(
            label=item.label,
            yes_bids=(),
            yes_asks=(),
            no_asks=(),
            fee_model=ZERO_FEE,
        )
        for item in valid
    )
    screen = screen_negative_risk_parity(
        missing,
        quantity=Decimal("5"),
        conversion_fee_bips=0,
    )
    assert screen.executable_path_count == 0
    assert screen.best_path is None
