from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.option_box_parity import (
    OptionBoxCandidate,
    confirm_option_box,
    discover_option_boxes,
)
from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
)


AS_OF_MS = 1_700_000_000_000
EXPIRY_MS = AS_OF_MS + 31_536_000_000


def _contract(
    symbol: str,
    strike: str,
    side: str,
    *,
    bid: str | None,
    ask: str | None,
) -> OptionContractQuote:
    return OptionContractQuote(
        symbol=symbol,
        underlying="ETHUSDT",
        expiry_date_ms=EXPIRY_MS,
        side=side,
        strike=Decimal(strike),
        unit=Decimal("1"),
        minimum_quantity=Decimal("0.01"),
        maximum_quantity=Decimal("100"),
        step_size=Decimal("0.01"),
        bid_price=None if bid is None else Decimal(bid),
        ask_price=None if ask is None else Decimal(ask),
    )


def _chain(
    *, short_profit: bool = False, long_profit: bool = False
) -> tuple[OptionContractQuote, ...]:
    if short_profit:
        return (
            _contract("C1", "100", "CALL", bid="60", ask="61"),
            _contract("P1", "100", "PUT", bid="9", ask="10"),
            _contract("C2", "150", "CALL", bid="9", ask="10"),
            _contract("P2", "150", "PUT", bid="51", ask="52"),
        )
    if long_profit:
        return (
            _contract("C1", "100", "CALL", bid="49", ask="50"),
            _contract("P1", "100", "PUT", bid="20", ask="21"),
            _contract("C2", "150", "CALL", bid="10", ask="11"),
            _contract("P2", "150", "PUT", bid="9", ask="10"),
        )
    return (
        _contract("C1", "100", "CALL", bid="49", ask="51"),
        _contract("P1", "100", "PUT", bid="9", ask="11"),
        _contract("C2", "150", "CALL", bid="9", ask="11"),
        _contract("P2", "150", "PUT", bid="19", ask="21"),
    )


def _book(
    symbol: str,
    *,
    bid: tuple[str, str] | None = ("1", "1"),
    ask: tuple[str, str] | None = ("2", "1"),
) -> OptionDepthQuote:
    return OptionDepthQuote(
        symbol=symbol,
        event_time_ms=AS_OF_MS,
        bids=()
        if bid is None
        else (OptionBookLevel(Decimal(bid[0]), Decimal(bid[1])),),
        asks=()
        if ask is None
        else (OptionBookLevel(Decimal(ask[0]), Decimal(ask[1])),),
    )


def test_short_box_ticker_credit_exceeds_fixed_liability() -> None:
    screen = discover_option_boxes(_chain(short_profit=True), as_of_ms=AS_OF_MS)

    assert screen.chain_count == 1
    assert screen.evaluated_strike_pair_count == 1
    assert screen.executable_short_box_count == 1
    assert len(screen.strict_positive_short_boxes) == 1
    candidate = screen.strict_positive_short_boxes[0]
    assert candidate.roles == ("sell", "buy", "sell", "buy")
    assert candidate.quantity == Decimal("0.01")
    assert candidate.fixed_expiry_cashflow_quote == Decimal("0.50")
    assert candidate.initial_credit_quote == Decimal("0.91")
    assert candidate.gross_expiry_profit_quote == Decimal("0.41")
    assert candidate.annualized_simple_return is None


def test_long_box_reports_nominal_return_and_annualized_carry() -> None:
    screen = discover_option_boxes(_chain(long_profit=True), as_of_ms=AS_OF_MS)

    candidate = screen.nominal_positive_long_boxes[0]
    assert candidate.roles == ("buy", "sell", "buy", "sell")
    assert candidate.initial_credit_quote == Decimal("-0.30")
    assert candidate.gross_expiry_profit_quote == Decimal("0.20")
    assert candidate.annualized_simple_return == Decimal(
        "0.6666666666666666666666666667"
    )
    assert screen.candidates == (candidate,)


@pytest.mark.parametrize("spot", ["0", "100", "125", "150", "1000"])
def test_long_and_short_box_expiry_cashflows_are_fixed(spot: str) -> None:
    value = Decimal(spot)
    call_1 = max(value - Decimal("100"), Decimal("0"))
    call_2 = max(value - Decimal("150"), Decimal("0"))
    put_1 = max(Decimal("100") - value, Decimal("0"))
    put_2 = max(Decimal("150") - value, Decimal("0"))

    assert call_1 - call_2 + put_2 - put_1 == Decimal("50")
    assert -call_1 + call_2 - put_2 + put_1 == Decimal("-50")


def test_nonpositive_and_missing_ticker_boxes_are_not_candidates() -> None:
    screen = discover_option_boxes(_chain(), as_of_ms=AS_OF_MS)
    missing = discover_option_boxes(
        tuple(
            replace(contract, bid_price=None, ask_price=None) for contract in _chain()
        ),
        as_of_ms=AS_OF_MS,
    )

    assert screen.candidates == ()
    assert screen.executable_long_box_count == 1
    assert screen.executable_short_box_count == 1
    assert missing.executable_long_box_count == 0
    assert missing.executable_short_box_count == 0


def test_expired_chain_is_counted_but_not_evaluated() -> None:
    expired = tuple(replace(contract, expiry_date_ms=AS_OF_MS) for contract in _chain())
    screen = discover_option_boxes(expired, as_of_ms=AS_OF_MS)

    assert screen.chain_count == 1
    assert screen.evaluated_strike_pair_count == 0


def test_depth_confirmation_reprices_box_and_missing_side_fails_closed() -> None:
    candidate = discover_option_boxes(
        _chain(short_profit=True), as_of_ms=AS_OF_MS
    ).strict_positive_short_boxes[0]
    depths = {
        "C1": _book("C1", bid=("60", "1"), ask=("61", "1")),
        "C2": _book("C2", bid=("9", "1"), ask=("10", "1")),
        "P2": _book("P2", bid=("51", "1"), ask=("52", "1")),
        "P1": _book("P1", bid=("9", "1"), ask=("10", "1")),
    }
    confirmed = confirm_option_box(candidate, depths)
    missing = confirm_option_box(candidate, {**depths, "C1": _book("C1", bid=None)})

    assert confirmed.executable is True
    assert confirmed.initial_credit_quote == Decimal("0.91")
    assert confirmed.gross_expiry_profit_quote == Decimal("0.41")
    assert confirmed.book_event_times_ms == (AS_OF_MS,) * 4
    assert missing.executable is False
    assert missing.initial_credit_quote is None
    assert missing.gross_expiry_profit_quote is None


def test_long_depth_confirmation_uses_fixed_receivable() -> None:
    candidate = discover_option_boxes(
        _chain(long_profit=True), as_of_ms=AS_OF_MS
    ).nominal_positive_long_boxes[0]
    depths = {
        "C1": _book("C1", bid=("49", "1"), ask=("50", "1")),
        "C2": _book("C2", bid=("10", "1"), ask=("11", "1")),
        "P2": _book("P2", bid=("9", "1"), ask=("10", "1")),
        "P1": _book("P1", bid=("20", "1"), ask=("21", "1")),
    }

    confirmed = confirm_option_box(candidate, depths)
    assert confirmed.initial_credit_quote == Decimal("-0.30")
    assert confirmed.gross_expiry_profit_quote == Decimal("0.20")


@pytest.mark.parametrize("as_of", [True, 0, 1.5])
def test_discovery_rejects_invalid_as_of(as_of: object) -> None:
    with pytest.raises(ValueError, match="as-of"):
        discover_option_boxes(_chain(), as_of_ms=as_of)  # type: ignore[arg-type]


def test_discovery_rejects_ambiguous_contract_sets() -> None:
    chain = _chain()
    with pytest.raises(ValueError, match="nonempty and unique"):
        discover_option_boxes((), as_of_ms=AS_OF_MS)
    with pytest.raises(ValueError, match="nonempty and unique"):
        discover_option_boxes((*chain, chain[0]), as_of_ms=AS_OF_MS)
    with pytest.raises(ValueError, match="unit-one"):
        discover_option_boxes(
            (replace(chain[0], unit=Decimal("2")), *chain[1:]),
            as_of_ms=AS_OF_MS,
        )
    with pytest.raises(ValueError, match="side is duplicated"):
        discover_option_boxes(
            (*chain, replace(chain[0], symbol="OTHER")),
            as_of_ms=AS_OF_MS,
        )


def test_candidate_rejects_unknown_kind_and_invalid_long_carry() -> None:
    from simple_ai_trading import option_box_parity

    chain = _chain(long_profit=True)
    by_symbol = {contract.symbol: contract for contract in chain}
    kwargs = {
        "lower_call": by_symbol["C1"],
        "upper_call": by_symbol["C2"],
        "upper_put": by_symbol["P2"],
        "lower_put": by_symbol["P1"],
        "as_of_ms": AS_OF_MS,
    }
    with pytest.raises(ValueError, match="kind"):
        option_box_parity._candidate(kind="unknown", **kwargs)
    zero_cost = {
        **kwargs,
        "lower_call": replace(
            by_symbol["C1"], bid_price=Decimal("9"), ask_price=Decimal("10")
        ),
        "lower_put": replace(
            by_symbol["P1"], bid_price=Decimal("10"), ask_price=Decimal("11")
        ),
    }
    with pytest.raises(ValueError, match="cost or expiry"):
        option_box_parity._candidate(kind="long", **zero_cost)


def test_confirmation_accepts_constructed_short_candidate() -> None:
    candidate = OptionBoxCandidate(
        kind="short",
        underlying="ETHUSDT",
        expiry_date_ms=EXPIRY_MS,
        lower_strike=Decimal("100"),
        upper_strike=Decimal("150"),
        symbols=("C1", "C2", "P2", "P1"),
        roles=("sell", "buy", "sell", "buy"),
        quantity=Decimal("0.01"),
        fixed_expiry_cashflow_quote=Decimal("0.5"),
        initial_credit_quote=Decimal("0.91"),
        gross_expiry_profit_quote=Decimal("0.41"),
        annualized_simple_return=None,
    )
    depths = {
        "C1": _book("C1", bid=("60", "1"), ask=("61", "1")),
        "C2": _book("C2", bid=("9", "1"), ask=("10", "1")),
        "P2": _book("P2", bid=("51", "1"), ask=("52", "1")),
        "P1": _book("P1", bid=("9", "1"), ask=("10", "1")),
    }

    assert confirm_option_box(candidate, depths).gross_expiry_profit_quote == Decimal(
        "0.41"
    )
