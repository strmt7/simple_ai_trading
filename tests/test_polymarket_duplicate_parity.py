from dataclasses import replace

import pytest

from simple_ai_trading.polymarket_duplicate_parity import (
    DuplicateContractTerms,
    discover_duplicate_contracts,
)


def _contract(
    market_id: str,
    *,
    question: str = "Will X happen?",
    description: str = "Resolves Yes exactly when X happens.",
) -> DuplicateContractTerms:
    number = int(market_id)
    return DuplicateContractTerms(
        event_id=str(1000 + number),
        market_id=market_id,
        condition_id="0x" + format(number, "064x"),
        question=question,
        description=description,
        end_date="2027-01-01T00:00:00Z",
        resolution_source="https://example.test/rules",
        group_item_title="X",
        outcomes=("Yes", "No"),
        token_ids=(str(10_000 + number * 2), str(10_001 + number * 2)),
    )


def test_exact_question_is_not_enough_when_rule_text_differs() -> None:
    screen = discover_duplicate_contracts(
        (
            _contract("1"),
            _contract("2", description="A materially different rule."),
        )
    )

    assert screen.contract_count == 2
    assert len(screen.exact_question_groups) == 1
    assert screen.exact_question_groups[0].distinct_payout_rule_count == 2
    assert screen.exact_payout_rule_groups == ()


def test_exact_payout_rule_group_requires_every_rule_field() -> None:
    first = _contract("1")
    second = _contract("2")
    third = replace(
        _contract("3"),
        question="Another question?",
        description="Another exact rule.",
    )
    screen = discover_duplicate_contracts((third, second, first))

    assert len(screen.exact_question_groups) == 1
    group = screen.exact_payout_rule_groups[0]
    assert tuple(contract.market_id for contract in group.contracts) == ("1", "2")
    assert group.payout_rule_sha256 == first.validated().payout_rule_sha256
    assert first.payout_rule_payload["outcomes"] == ["Yes", "No"]


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (replace(_contract("1"), event_id="bad"), "numeric identity"),
        (replace(_contract("1"), condition_id="0x1"), "condition identity"),
        (replace(_contract("1"), question=""), "payout rule is incomplete"),
        (replace(_contract("1"), outcomes=("No", "Yes")), "outcomes differ"),
        (replace(_contract("1"), token_ids=("1", "1")), "token identity"),
    ],
)
def test_contract_validation_fails_closed(
    contract: DuplicateContractTerms, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contract.validated()


def test_discovery_rejects_empty_or_repeated_identities() -> None:
    first = _contract("1")
    second = _contract("2")
    with pytest.raises(ValueError, match="requires contracts"):
        discover_duplicate_contracts(())
    with pytest.raises(ValueError, match="market identity"):
        discover_duplicate_contracts((first, replace(second, market_id="1")))
    with pytest.raises(ValueError, match="condition identity"):
        discover_duplicate_contracts(
            (first, replace(second, condition_id=first.condition_id))
        )
    with pytest.raises(ValueError, match="token identity"):
        discover_duplicate_contracts(
            (first, replace(second, token_ids=(first.token_ids[0], "999")))
        )


def test_nonduplicate_questions_are_retained_only_in_contract_count() -> None:
    screen = discover_duplicate_contracts(
        (
            _contract("1", question="Question one?"),
            _contract("2", question="Question two?"),
        )
    )

    assert screen.contract_count == 2
    assert screen.exact_question_groups == ()
    assert screen.exact_payout_rule_groups == ()
