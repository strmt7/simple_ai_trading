from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
    OptionParityCandidate,
    confirm_option_candidate,
    discover_option_parity,
    minimum_ratio_quantities,
    primitive_convexity_weights,
)


def _contract(
    symbol: str,
    strike: str,
    *,
    side: str = "PUT",
    bid: str | None = "0.6",
    ask: str | None = "0.8",
    step: str = "0.01",
    minimum: str = "0.01",
    maximum: str = "100",
) -> OptionContractQuote:
    return OptionContractQuote(
        symbol=symbol,
        underlying="ETHUSDT",
        expiry_date_ms=1_800_000_000_000,
        side=side,
        strike=Decimal(strike),
        unit=Decimal("1"),
        minimum_quantity=Decimal(minimum),
        maximum_quantity=Decimal(maximum),
        step_size=Decimal(step),
        bid_price=None if bid is None else Decimal(bid),
        ask_price=None if ask is None else Decimal(ask),
    )


def _book(
    symbol: str,
    *,
    bids: tuple[tuple[str, str], ...] = (("0.8", "10"),),
    asks: tuple[tuple[str, str], ...] = (("1.0", "10"),),
    event_time_ms: int = 1_800_000_000_001,
) -> OptionDepthQuote:
    return OptionDepthQuote(
        symbol=symbol,
        event_time_ms=event_time_ms,
        bids=tuple(
            OptionBookLevel(Decimal(price), Decimal(quantity))
            for price, quantity in bids
        ),
        asks=tuple(
            OptionBookLevel(Decimal(price), Decimal(quantity))
            for price, quantity in asks
        ),
    )


def test_unequal_strike_convexity_uses_exact_primitive_ratio() -> None:
    weights = primitive_convexity_weights(
        Decimal("1300"), Decimal("1950"), Decimal("2000")
    )
    contracts = (
        _contract("ETH-1-1300-P", "1300"),
        _contract("ETH-1-1950-P", "1950"),
        _contract("ETH-1-2000-P", "2000"),
    )

    assert weights == (1, 14, 13)
    assert minimum_ratio_quantities(weights, contracts) == (
        Decimal("0.01"),
        Decimal("0.14"),
        Decimal("0.13"),
    )


@pytest.mark.parametrize("spot", ["0", "1300", "1800", "1950", "1975", "2500"])
@pytest.mark.parametrize("side", ["CALL", "PUT"])
def test_convexity_ratio_has_nonnegative_expiry_payoff(spot: str, side: str) -> None:
    strikes = (Decimal("1300"), Decimal("1950"), Decimal("2000"))
    weights = primitive_convexity_weights(*strikes)
    value = Decimal("0")
    for role, weight, strike in zip(
        ("buy", "sell", "buy"), weights, strikes, strict=True
    ):
        intrinsic = (
            max(Decimal(spot) - strike, Decimal("0"))
            if side == "CALL"
            else max(strike - Decimal(spot), Decimal("0"))
        )
        value += intrinsic * weight * (1 if role == "buy" else -1)
    assert value >= 0


def test_discovery_finds_call_and_put_vertical_dominance() -> None:
    call_screen = discover_option_parity(
        (
            _contract("ETH-1-1000-C", "1000", side="CALL", bid="0.8", ask="1.0"),
            _contract("ETH-1-1100-C", "1100", side="CALL", bid="1.2", ask="1.4"),
        )
    )
    put_screen = discover_option_parity(
        (
            _contract("ETH-1-1000-P", "1000", bid="1.2", ask="1.4"),
            _contract("ETH-1-1100-P", "1100", bid="0.8", ask="1.0"),
        )
    )

    assert call_screen.evaluated_vertical_count == 1
    assert call_screen.executable_vertical_count == 1
    assert call_screen.gross_positive_candidates[0].symbols == (
        "ETH-1-1000-C",
        "ETH-1-1100-C",
    )
    assert put_screen.gross_positive_candidates[0].symbols == (
        "ETH-1-1100-P",
        "ETH-1-1000-P",
    )


def test_discovery_finds_arbitrary_strike_convexity_credit() -> None:
    screen = discover_option_parity(
        (
            _contract("ETH-1-1300-P", "1300", bid=None, ask="0.2"),
            _contract("ETH-1-1950-P", "1950", bid="0.8", ask="1.0"),
            _contract("ETH-1-2000-P", "2000", bid="0.6", ask="0.8"),
        )
    )

    assert screen.evaluated_vertical_count == 3
    assert screen.evaluated_convexity_count == 1
    assert screen.executable_convexity_count == 1
    candidate = next(
        item
        for item in screen.gross_positive_candidates
        if item.mechanism == "strike_convexity"
    )
    assert candidate.integer_weights == (1, 14, 13)
    assert candidate.gross_credit_quote == Decimal("0.006")


def test_missing_ticker_side_is_counted_but_not_executable() -> None:
    screen = discover_option_parity(
        (
            _contract("ETH-1-1000-P", "1000", bid=None, ask=None),
            _contract("ETH-1-1100-P", "1100", bid="1.2", ask=None),
            _contract("ETH-1-1200-P", "1200", bid="1.4", ask="1.6"),
        )
    )

    assert screen.evaluated_vertical_count == 3
    assert screen.executable_vertical_count == 1
    assert screen.evaluated_convexity_count == 1
    assert screen.executable_convexity_count == 0


def test_executable_nonpositive_convexity_is_not_a_candidate() -> None:
    screen = discover_option_parity(
        (
            _contract("ETH-1-1000-P", "1000"),
            _contract("ETH-1-1100-P", "1100"),
            _contract("ETH-1-1200-P", "1200"),
        )
    )

    assert screen.executable_convexity_count == 1
    assert all(
        item.mechanism != "strike_convexity"
        for item in screen.gross_positive_candidates
    )


def test_depth_confirmation_walks_all_levels_and_can_reject_discovery() -> None:
    candidate = OptionParityCandidate(
        mechanism="strike_convexity",
        symbols=("LOW", "MID", "HIGH"),
        roles=("buy", "sell", "buy"),
        strikes=(Decimal("1"), Decimal("2"), Decimal("3")),
        integer_weights=(1, 2, 1),
        quantities=(Decimal("2"), Decimal("4"), Decimal("2")),
        gross_credit_quote=Decimal("1"),
    )
    result = confirm_option_candidate(
        candidate,
        {
            "LOW": _book("LOW", asks=(("1", "1"), ("2", "1"))),
            "MID": _book(
                "MID",
                bids=(("1.1", "3"), ("1", "1")),
                asks=(("2", "10"),),
            ),
            "HIGH": _book("HIGH", asks=(("2", "2"),)),
        },
    )

    assert result.executable is True
    assert result.gross_credit_quote == Decimal("-2.7")
    assert result.book_event_times_ms == (1_800_000_000_001,) * 3


def test_depth_confirmation_does_not_invent_missing_quantity() -> None:
    candidate = OptionParityCandidate(
        mechanism="vertical_dominance",
        symbols=("BUY", "SELL"),
        roles=("buy", "sell"),
        strikes=(Decimal("1"), Decimal("2")),
        integer_weights=(1, 1),
        quantities=(Decimal("2"), Decimal("2")),
        gross_credit_quote=Decimal("1"),
    )
    result = confirm_option_candidate(
        candidate,
        {
            "BUY": _book("BUY", asks=(("1", "1"),)),
            "SELL": _book("SELL"),
        },
    )

    assert result.executable is False
    assert result.gross_credit_quote is None
    assert result.book_event_times_ms == (1_800_000_000_001,)


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (_contract("bad symbol!", "1"), "identity"),
        (
            replace(_contract("GOOD", "1"), expiry_date_ms=True),
            "expiry",
        ),
    ],
)
def test_contract_rejects_invalid_identity_and_expiry(
    contract: OptionContractQuote, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contract.validated()


def test_contract_and_ratio_fail_closed_on_ambiguous_terms() -> None:
    base = _contract("GOOD", "1")
    with pytest.raises(ValueError, match="crossed"):
        _contract("CROSSED", "1", bid="2", ask="1").validated()
    with pytest.raises(ValueError, match="exceeds"):
        _contract("LIMIT", "1", minimum="2", maximum="1").validated()
    with pytest.raises(ValueError, match="finite decimal"):
        replace(base, strike=True).validated()
    with pytest.raises(ValueError, match="finite decimal"):
        replace(base, strike="bad").validated()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive finite"):
        replace(base, strike=Decimal("Infinity")).validated()
    with pytest.raises(ValueError, match="positive finite"):
        replace(base, step_size=Decimal("0")).validated()
    with pytest.raises(ValueError, match="weights"):
        minimum_ratio_quantities((), ())
    with pytest.raises(ValueError, match="maximum"):
        minimum_ratio_quantities(
            (1, 2),
            (
                _contract("MAX-ONE", "1", maximum="0.01"),
                _contract("MAX-TWO", "2", maximum="0.01"),
            ),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        primitive_convexity_weights(Decimal("1"), Decimal("1"), Decimal("2"))
    with pytest.raises(ValueError, match="nonempty and unique"):
        discover_option_parity(())
    with pytest.raises(ValueError, match="nonempty and unique"):
        discover_option_parity((base, base))


def test_discovery_rejects_duplicate_strikes_and_mixed_units() -> None:
    duplicate = (
        _contract("ONE", "1"),
        _contract("TWO", "1"),
    )
    mixed = (
        _contract("ONE", "1"),
        OptionContractQuote(
            symbol="TWO",
            underlying="ETHUSDT",
            expiry_date_ms=1_800_000_000_000,
            side="PUT",
            strike=Decimal("2"),
            unit=Decimal("2"),
            minimum_quantity=Decimal("0.01"),
            maximum_quantity=Decimal("100"),
            step_size=Decimal("0.01"),
            bid_price=Decimal("0.6"),
            ask_price=Decimal("0.8"),
        ),
    )
    with pytest.raises(ValueError, match="strike is duplicated"):
        discover_option_parity(duplicate)
    with pytest.raises(ValueError, match="units differ"):
        discover_option_parity(mixed)


@pytest.mark.parametrize(
    ("book", "message"),
    [
        (_book("bad symbol!"), "symbol"),
        (_book("GOOD", event_time_ms=0), "event time"),
        (_book("GOOD", bids=(("1", "1"), ("2", "1"))), "price-sorted"),
        (_book("GOOD", asks=(("1", "1"), ("1", "2"))), "duplicated"),
        (_book("GOOD", bids=(("2", "1"),), asks=(("1", "1"),)), "crossed"),
    ],
)
def test_depth_rejects_invalid_or_ambiguous_books(
    book: OptionDepthQuote, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        book.validated()


def test_confirmation_rejects_bad_candidate_or_depth_binding() -> None:
    empty = OptionParityCandidate("x", (), (), (), (), (), Decimal("0"))
    with pytest.raises(ValueError, match="shape"):
        confirm_option_candidate(empty, {})

    candidate = OptionParityCandidate(
        "x",
        ("GOOD",),
        ("hold",),
        (Decimal("1"),),
        (1,),
        (Decimal("1"),),
        Decimal("0"),
    )
    with pytest.raises(ValueError, match="role"):
        confirm_option_candidate(candidate, {"GOOD": _book("GOOD")})

    candidate = OptionParityCandidate(
        "x",
        ("GOOD",),
        ("buy",),
        (Decimal("1"),),
        (1,),
        (Decimal("1"),),
        Decimal("0"),
    )
    with pytest.raises(ValueError, match="absent"):
        confirm_option_candidate(candidate, {})
    with pytest.raises(ValueError, match="does not match"):
        confirm_option_candidate(candidate, {"GOOD": _book("OTHER")})
