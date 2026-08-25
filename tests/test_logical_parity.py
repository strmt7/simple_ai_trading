from decimal import Decimal

import pytest

from simple_ai_trading.logical_parity import (
    CryptoThresholdQuestion,
    LogicalBinaryOutcome,
    parse_crypto_threshold_question,
    screen_logical_implication_bundle,
)
from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel


ZERO_FEE = PolymarketFeeModel(False, Decimal("0"), 1, True)
DYNAMIC_FEE = PolymarketFeeModel(True, Decimal("0.07"), 1, True)


@pytest.mark.parametrize(
    ("question", "kind", "asset", "threshold", "window"),
    [
        (
            "Will the price of Bitcoin be above $72,000 on August 25?",
            "above",
            "BTC",
            Decimal("72000"),
            "on august 25",
        ),
        (
            "Ethereum above 2,560 on August 24, 10PM ET?",
            "above",
            "ETH",
            Decimal("2560"),
            "on august 24, 10pm et",
        ),
        (
            "Will Solana reach $320 by December 31, 2026?",
            "reach",
            "SOL",
            Decimal("320"),
            "by december 31, 2026",
        ),
        (
            "Will Bitcoin hit $150k by December 31, 2027?",
            "reach",
            "BTC",
            Decimal("150000"),
            "by december 31, 2027",
        ),
        (
            "Will Ethereum dip to $1,400 in August?",
            "dip",
            "ETH",
            Decimal("1400"),
            "in august",
        ),
    ],
)
def test_parse_crypto_threshold_question(
    question: str,
    kind: str,
    asset: str,
    threshold: Decimal,
    window: str,
) -> None:
    parsed = parse_crypto_threshold_question(question)
    assert parsed is not None
    assert (parsed.kind, parsed.asset, parsed.threshold, parsed.window) == (
        kind,
        asset,
        threshold,
        window,
    )


@pytest.mark.parametrize(
    "question",
    [
        "Will Ethereum hit $1,000 or $3,000 first?",
        "Will Dogecoin reach $1 by December 31, 2026?",
        "Will Bitcoin maybe be above $80,000 on August 25?",
        "",
    ],
)
def test_parse_crypto_threshold_question_rejects_ambiguous_or_unsupported_forms(
    question: str,
) -> None:
    assert parse_crypto_threshold_question(question) is None


def test_threshold_implication_order_handles_upper_and_lower_barriers() -> None:
    reach_100 = CryptoThresholdQuestion("reach", "BTC", Decimal("100"), "in 2026")
    reach_200 = CryptoThresholdQuestion("reach", "BTC", Decimal("200"), "in 2026")
    dip_100 = CryptoThresholdQuestion("dip", "BTC", Decimal("100"), "in 2026")
    dip_200 = CryptoThresholdQuestion("dip", "BTC", Decimal("200"), "in 2026")
    assert reach_200.stronger_than(reach_100) is True
    assert reach_100.stronger_than(reach_200) is False
    assert dip_100.stronger_than(dip_200) is True
    assert dip_200.stronger_than(dip_100) is False
    assert reach_200.stronger_than(dip_100) is False


@pytest.mark.parametrize(
    "question",
    [
        CryptoThresholdQuestion("other", "BTC", Decimal("1"), "in 2026"),
        CryptoThresholdQuestion("reach", "DOGE", Decimal("1"), "in 2026"),
        CryptoThresholdQuestion("reach", "BTC", Decimal("1"), ""),
        CryptoThresholdQuestion("reach", "BTC", Decimal("0"), "in 2026"),
        CryptoThresholdQuestion("reach", "BTC", Decimal("NaN"), "in 2026"),
        CryptoThresholdQuestion("reach", "BTC", True, "in 2026"),
        CryptoThresholdQuestion("reach", "BTC", None, "in 2026"),  # type: ignore[arg-type]
    ],
)
def test_threshold_identity_validation_fails_closed(
    question: CryptoThresholdQuestion,
) -> None:
    with pytest.raises(ValueError):
        question.validated()


def _outcome(
    label: str,
    *,
    yes: tuple[tuple[str, str], ...],
    no: tuple[tuple[str, str], ...],
    fee_model: PolymarketFeeModel = ZERO_FEE,
) -> LogicalBinaryOutcome:
    def levels(values: tuple[tuple[str, str], ...]) -> tuple[BookLevel, ...]:
        return tuple(BookLevel(Decimal(price), Decimal(size)) for price, size in values)

    return LogicalBinaryOutcome(label, levels(yes), levels(no), fee_model)


def test_screen_logical_implication_bundle_walks_depth_and_terminal_floor() -> None:
    weaker = _outcome(
        "weaker",
        yes=(("0.30", "2"), ("0.35", "4")),
        no=(("0.70", "5"),),
    )
    stronger = _outcome(
        "stronger",
        yes=(("0.20", "5"),),
        no=(("0.60", "3"), ("0.65", "4")),
    )
    result = screen_logical_implication_bundle(
        weaker,
        stronger,
        quantity=Decimal("5"),
    )
    assert result is not None
    assert result.gross_cost_quote == Decimal("4.75")
    assert result.taker_fees_quote == 0
    assert result.terminal_payout_floor_quote == Decimal("5")
    assert result.net_quote == Decimal("0.25")


def test_screen_logical_implication_bundle_applies_each_taker_fee_curve() -> None:
    weaker = _outcome(
        "weaker",
        yes=(("0.02", "5"),),
        no=(("0.98", "5"),),
        fee_model=DYNAMIC_FEE,
    )
    stronger = _outcome(
        "stronger",
        yes=(("0.02", "5"),),
        no=(("0.98", "5"),),
        fee_model=DYNAMIC_FEE,
    )
    result = screen_logical_implication_bundle(
        weaker,
        stronger,
        quantity=Decimal("5"),
    )
    assert result is not None
    assert result.gross_cost_quote == Decimal("5.00")
    assert result.taker_fees_quote > 0
    assert result.net_quote < 0


def test_screen_logical_implication_bundle_requires_both_displayed_legs() -> None:
    weaker = _outcome("weaker", yes=(("0.2", "4"),), no=(("0.8", "5"),))
    stronger = _outcome("stronger", yes=(("0.2", "5"),), no=(("0.8", "5"),))
    assert (
        screen_logical_implication_bundle(
            weaker,
            stronger,
            quantity=Decimal("5"),
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        LogicalBinaryOutcome("", (), (), ZERO_FEE),
        LogicalBinaryOutcome(
            "bad order",
            (
                BookLevel(Decimal("0.4"), Decimal("5")),
                BookLevel(Decimal("0.3"), Decimal("5")),
            ),
            (),
            ZERO_FEE,
        ),
        LogicalBinaryOutcome(
            "duplicate",
            (
                BookLevel(Decimal("0.3"), Decimal("5")),
                BookLevel(Decimal("0.3"), Decimal("6")),
            ),
            (),
            ZERO_FEE,
        ),
        LogicalBinaryOutcome(
            "unit price",
            (BookLevel(Decimal("1"), Decimal("5")),),
            (),
            ZERO_FEE,
        ),
    ],
)
def test_logical_outcome_validation_fails_closed(
    mutation: LogicalBinaryOutcome,
) -> None:
    with pytest.raises(ValueError):
        mutation.validated()


def test_screen_logical_implication_bundle_rejects_invalid_quantity_and_identity() -> (
    None
):
    outcome = _outcome("same", yes=(("0.2", "5"),), no=(("0.8", "5"),))
    with pytest.raises(ValueError):
        screen_logical_implication_bundle(outcome, outcome, quantity=Decimal("5"))
    with pytest.raises(ValueError):
        screen_logical_implication_bundle(outcome, outcome, quantity=Decimal("0"))
