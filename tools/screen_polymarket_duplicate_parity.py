"""Discover exact cross-condition payout-rule duplicates on Polymarket."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.polymarket_duplicate_parity as duplicate_module
from simple_ai_trading.polymarket_duplicate_parity import (
    DuplicateContractTerms,
    discover_duplicate_contracts,
)
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "polymarket-duplicate-contract-parity-screen-v1"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
ASSET_TAGS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
PAGE_SIZE = 100
MAX_OFFSET = 900


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _json_list(value: object, *, name: str) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not JSON") from exc
    return _list(value, name=name)


def _get(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, params=params, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RuntimeError(
            "Polymarket rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    evidence: dict[str, object] = {
        "url": response.url,
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
    }
    return decoded, evidence


def _eligible_embedded_market(
    raw: object, *, event_id: str
) -> dict[str, object] | None:
    market = _mapping(raw, name=f"Gamma event {event_id} market")
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
        and market.get("negRisk") is False
    ):
        return None
    outcomes = _json_list(market.get("outcomes"), name="duplicate market outcomes")
    if outcomes != ["Yes", "No"]:
        return None
    market_id = str(market.get("id") or "")
    condition_id = str(market.get("conditionId") or "").lower()
    question = str(market.get("question") or "")
    if (
        not market_id.isdigit()
        or len(condition_id) != 66
        or not condition_id.startswith("0x")
        or not question
    ):
        raise ValueError(f"Gamma event {event_id} duplicate identity is invalid")
    return {
        "event_id": event_id,
        "market_id": market_id,
        "condition_id": condition_id,
        "question": question,
    }


def _canonical_terms(
    raw_event: object,
    *,
    selected_market_ids: set[str],
) -> tuple[DuplicateContractTerms, ...]:
    event = _mapping(raw_event, name="canonical duplicate event")
    event_id = str(event.get("id") or "")
    if (
        not event_id.isdigit()
        or event.get("active") is not True
        or event.get("closed") is not False
    ):
        raise ValueError("canonical duplicate event identity is invalid")
    result: list[DuplicateContractTerms] = []
    for raw_market in _list(event.get("markets"), name="canonical duplicate markets"):
        market = _mapping(raw_market, name="canonical duplicate market")
        market_id = str(market.get("id") or "")
        if market_id not in selected_market_ids:
            continue
        if not (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
            and market.get("enableOrderBook") is True
            and market.get("negRisk") is False
        ):
            raise ValueError("canonical duplicate market is no longer eligible")
        outcomes = tuple(
            str(value)
            for value in _json_list(
                market.get("outcomes"), name="canonical duplicate outcomes"
            )
        )
        tokens = tuple(
            str(value)
            for value in _json_list(
                market.get("clobTokenIds"), name="canonical duplicate tokens"
            )
        )
        result.append(
            DuplicateContractTerms(
                event_id=event_id,
                market_id=market_id,
                condition_id=str(market.get("conditionId") or ""),
                question=str(market.get("question") or ""),
                description=str(market.get("description") or ""),
                end_date=str(market.get("endDate") or ""),
                resolution_source=str(market.get("resolutionSource") or ""),
                group_item_title=str(market.get("groupItemTitle") or ""),
                outcomes=outcomes,  # type: ignore[arg-type]
                token_ids=tokens,  # type: ignore[arg-type]
            ).validated()
        )
    if {contract.market_id for contract in result} != selected_market_ids:
        raise ValueError("canonical duplicate event omitted a selected market")
    return tuple(result)


def _contract_payload(contract: DuplicateContractTerms) -> dict[str, object]:
    return {
        "event_id": contract.event_id,
        "market_id": contract.market_id,
        "condition_id": contract.condition_id,
        "question": contract.question,
        "description": contract.description,
        "end_date": contract.end_date,
        "resolution_source": contract.resolution_source,
        "group_item_title": contract.group_item_title,
        "outcomes": list(contract.outcomes),
        "token_ids": list(contract.token_ids),
        "payout_rule_sha256": contract.payout_rule_sha256,
    }


def run() -> dict[str, object]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-duplicate-parity-research/1.0",
        }
    )
    started_ms = time.time_ns() // 1_000_000
    events: dict[str, dict[str, object]] = {}
    tag_evidence: list[dict[str, object]] = []
    for asset, slug in ASSET_TAGS.items():
        tag_url = f"{GAMMA_BASE_URL}/tags/slug/{slug}"
        raw_tag, tag_source = _get(session, tag_url)
        tag = _mapping(raw_tag, name=f"Gamma {asset} tag")
        if str(tag.get("slug") or "").lower() != slug:
            raise ValueError(f"Gamma {asset} tag identity differs")
        tag_id = str(tag.get("id") or "")
        if not tag_id.isdigit():
            raise ValueError(f"Gamma {asset} tag ID is invalid")
        pages: list[dict[str, object]] = []
        for offset in range(0, MAX_OFFSET + 1, PAGE_SIZE):
            raw_page, page_source = _get(
                session,
                f"{GAMMA_BASE_URL}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "tag_id": tag_id,
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            page = _list(raw_page, name=f"Gamma {asset} event page")
            pages.append({**page_source, "offset": offset, "event_count": len(page)})
            for raw_event in page:
                event = _mapping(raw_event, name=f"Gamma {asset} event")
                event_id = str(event.get("id") or "")
                if (
                    not event_id.isdigit()
                    or event.get("active") is not True
                    or event.get("closed") is not False
                ):
                    raise ValueError(f"Gamma {asset} event filter drifted")
                events.setdefault(event_id, event)
            if len(page) < PAGE_SIZE:
                break
        else:
            raise ValueError(f"Gamma {asset} tag exceeded the bounded offset")
        tag_evidence.append(
            {
                "asset": asset,
                "slug": slug,
                "tag_id": tag_id,
                "tag_source": tag_source,
                "pages": pages,
            }
        )

    eligible: list[dict[str, object]] = []
    for event_id, event in events.items():
        for raw_market in _list(
            event.get("markets"), name=f"Gamma event {event_id} markets"
        ):
            parsed = _eligible_embedded_market(raw_market, event_id=event_id)
            if parsed is not None:
                eligible.append(parsed)
    by_question: dict[str, list[dict[str, object]]] = {}
    for market in eligible:
        by_question.setdefault(str(market["question"]), []).append(market)
    question_candidates = {
        question: rows
        for question, rows in by_question.items()
        if len({str(row["condition_id"]) for row in rows}) >= 2
    }

    event_market_ids: dict[str, set[str]] = {}
    for rows in question_candidates.values():
        for row in rows:
            event_market_ids.setdefault(str(row["event_id"]), set()).add(
                str(row["market_id"])
            )
    canonical_sources: list[dict[str, object]] = []
    contracts: list[DuplicateContractTerms] = []
    for event_id, market_ids in sorted(
        event_market_ids.items(), key=lambda item: int(item[0])
    ):
        raw_event, source = _get(session, f"{GAMMA_BASE_URL}/events/{event_id}")
        canonical_sources.append(source)
        contracts.extend(_canonical_terms(raw_event, selected_market_ids=market_ids))
    if not contracts:
        result_discovery = None
        exact_question_groups: list[dict[str, object]] = []
        exact_rule_groups: list[dict[str, object]] = []
    else:
        result_discovery = discover_duplicate_contracts(contracts)
        exact_question_groups = []
        rule_fields = tuple(contracts[0].payout_rule_payload)
        for group in result_discovery.exact_question_groups:
            differing = [
                field
                for field in rule_fields
                if len(
                    {
                        _canonical_json(contract.payout_rule_payload[field])
                        for contract in group.contracts
                    }
                )
                > 1
            ]
            exact_question_groups.append(
                {
                    "question": group.question,
                    "distinct_payout_rule_count": group.distinct_payout_rule_count,
                    "differing_payout_rule_fields": differing,
                    "contracts": [
                        _contract_payload(contract) for contract in group.contracts
                    ],
                }
            )
        exact_rule_groups = [
            {
                "payout_rule_sha256": group.payout_rule_sha256,
                "contracts": [
                    _contract_payload(contract) for contract in group.contracts
                ],
            }
            for group in result_discovery.exact_payout_rule_groups
        ]

    exact_rule_count = len(exact_rule_groups)
    status = (
        "unqualified_exact_rule_duplicates_require_depth_and_fee_screen"
        if exact_rule_count
        else "rejected_no_exact_payout_rule_duplicate"
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "target_free_cross_condition_exact_payout_rule_duplicate_discovery",
        "started_at_ms": started_ms,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "scope": {
            "assets": sorted(ASSET_TAGS),
            "active": True,
            "closed": False,
            "accepting_orders": True,
            "order_book_enabled": True,
            "negative_risk": False,
            "outcomes": ["Yes", "No"],
            "page_size": PAGE_SIZE,
            "maximum_offset": MAX_OFFSET,
        },
        "source_contract": {
            "gamma_base_url": GAMMA_BASE_URL,
            "tag_sources": tag_evidence,
            "canonical_candidate_event_sources": canonical_sources,
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(duplicate_module.__file__).name,
                "module_sha256": _sha256(Path(duplicate_module.__file__).read_bytes()),
            },
        },
        "identity_contract": {
            "question_match": "byte_exact",
            "payout_rule_fields": (
                list(contracts[0].payout_rule_payload) if contracts else []
            ),
            "semantic_substitution_allowed": False,
            "different_rule_text_is_equivalent": False,
            "different_conditions_require_all_payout_rule_fields_equal": True,
        },
        "universe": {
            "unique_scoped_active_event_count": len(events),
            "eligible_binary_market_count": len(eligible),
            "exact_question_candidate_group_count": len(question_candidates),
            "canonical_candidate_contract_count": len(contracts),
        },
        "exact_question_groups": exact_question_groups,
        "exact_payout_rule_groups": exact_rule_groups,
        "verdict": {
            "status": status,
            "exact_question_group_count": len(exact_question_groups),
            "exact_payout_rule_group_count": exact_rule_count,
            "depth_or_fee_screen_performed": False,
            "accepted_edge": False,
            "trading_authority": False,
        },
        "safety": {
            "public_market_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "title_only_equivalence_allowed": False,
            "non_equivalent_pairs_priced_as_guaranteed_bundles": False,
        },
        "limitations": [
            "Exact question text is only a discovery key and does not prove identical payout rules.",
            "Description, deadline, resolution source, group item, and outcomes must all match byte-for-byte before any guaranteed-bundle price screen.",
            "Separate conditions can still face independent oracle and resolution-process risk even when prose matches.",
            "No exact payout-rule group means depth, fees, persistence, fills, and promotion are out of scope.",
        ],
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
