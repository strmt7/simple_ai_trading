"""Strict payout-rule identity discovery across Polymarket conditions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class DuplicateContractTerms:
    """Canonical binary payout terms for one active condition."""

    event_id: str
    market_id: str
    condition_id: str
    question: str
    description: str
    end_date: str
    resolution_source: str
    group_item_title: str
    outcomes: tuple[str, str]
    token_ids: tuple[str, str]

    def validated(self) -> "DuplicateContractTerms":
        event_id = str(self.event_id or "")
        market_id = str(self.market_id or "")
        condition_id = str(self.condition_id or "").lower()
        if not event_id.isdigit() or not market_id.isdigit():
            raise ValueError("duplicate contract numeric identity is invalid")
        if (
            len(condition_id) != 66
            or not condition_id.startswith("0x")
            or any(
                character not in "0123456789abcdef" for character in condition_id[2:]
            )
        ):
            raise ValueError("duplicate contract condition identity is invalid")
        question = str(self.question or "")
        description = str(self.description or "")
        end_date = str(self.end_date or "")
        resolution_source = str(self.resolution_source or "")
        group_item_title = str(self.group_item_title or "")
        if not question or not description or not end_date:
            raise ValueError("duplicate contract payout rule is incomplete")
        outcomes = tuple(str(value) for value in self.outcomes)
        tokens = tuple(str(value) for value in self.token_ids)
        if outcomes != ("Yes", "No"):
            raise ValueError("duplicate contract outcomes differ from Yes/No")
        if (
            len(tokens) != 2
            or len(set(tokens)) != 2
            or any(not token.isdigit() for token in tokens)
        ):
            raise ValueError("duplicate contract token identity is invalid")
        return DuplicateContractTerms(
            event_id=event_id,
            market_id=market_id,
            condition_id=condition_id,
            question=question,
            description=description,
            end_date=end_date,
            resolution_source=resolution_source,
            group_item_title=group_item_title,
            outcomes=(outcomes[0], outcomes[1]),
            token_ids=(tokens[0], tokens[1]),
        )

    @property
    def payout_rule_payload(self) -> dict[str, object]:
        return {
            "question": self.question,
            "description": self.description,
            "end_date": self.end_date,
            "resolution_source": self.resolution_source,
            "group_item_title": self.group_item_title,
            "outcomes": list(self.outcomes),
        }

    @property
    def payout_rule_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.payout_rule_payload).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ExactQuestionGroup:
    """Different conditions sharing byte-identical question text."""

    question: str
    contracts: tuple[DuplicateContractTerms, ...]
    distinct_payout_rule_count: int


@dataclass(frozen=True, slots=True)
class ExactPayoutRuleGroup:
    """Different conditions sharing every frozen payout-rule field."""

    payout_rule_sha256: str
    contracts: tuple[DuplicateContractTerms, ...]


@dataclass(frozen=True, slots=True)
class DuplicateContractDiscovery:
    """Strict cross-condition duplicate discovery with no semantic inference."""

    contract_count: int
    exact_question_groups: tuple[ExactQuestionGroup, ...]
    exact_payout_rule_groups: tuple[ExactPayoutRuleGroup, ...]


def discover_duplicate_contracts(
    contracts: Sequence[DuplicateContractTerms],
) -> DuplicateContractDiscovery:
    """Group exact questions, then require exact payout-rule equality."""

    normalized = tuple(contract.validated() for contract in contracts)
    if not normalized:
        raise ValueError("duplicate contract discovery requires contracts")
    if len({contract.market_id for contract in normalized}) != len(normalized):
        raise ValueError("duplicate contract market identity is repeated")
    if len({contract.condition_id for contract in normalized}) != len(normalized):
        raise ValueError("duplicate contract condition identity is repeated")
    all_tokens = [token for contract in normalized for token in contract.token_ids]
    if len(set(all_tokens)) != len(all_tokens):
        raise ValueError("duplicate contract token identity is repeated")

    by_question: dict[str, list[DuplicateContractTerms]] = {}
    for contract in normalized:
        by_question.setdefault(contract.question, []).append(contract)
    question_groups: list[ExactQuestionGroup] = []
    payout_groups: list[ExactPayoutRuleGroup] = []
    for question, rows in by_question.items():
        if len(rows) < 2:
            continue
        ordered = tuple(sorted(rows, key=lambda contract: int(contract.market_id)))
        by_rule: dict[str, list[DuplicateContractTerms]] = {}
        for contract in ordered:
            by_rule.setdefault(contract.payout_rule_sha256, []).append(contract)
        question_groups.append(
            ExactQuestionGroup(
                question=question,
                contracts=ordered,
                distinct_payout_rule_count=len(by_rule),
            )
        )
        for fingerprint, matching in by_rule.items():
            if len(matching) >= 2:
                payout_groups.append(
                    ExactPayoutRuleGroup(
                        payout_rule_sha256=fingerprint,
                        contracts=tuple(matching),
                    )
                )
    question_groups.sort(key=lambda group: group.question)
    payout_groups.sort(key=lambda group: group.payout_rule_sha256)
    return DuplicateContractDiscovery(
        contract_count=len(normalized),
        exact_question_groups=tuple(question_groups),
        exact_payout_rule_groups=tuple(payout_groups),
    )


__all__ = [
    "DuplicateContractDiscovery",
    "DuplicateContractTerms",
    "ExactPayoutRuleGroup",
    "ExactQuestionGroup",
    "discover_duplicate_contracts",
]
