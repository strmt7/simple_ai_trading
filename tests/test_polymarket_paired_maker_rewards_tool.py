from __future__ import annotations

import importlib.util
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_polymarket_paired_maker_rewards",
    ROOT / "tools" / "screen_polymarket_paired_maker_rewards.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
SNAPSHOT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "paired-maker-reward-snapshot-v1-2026-08-25.json"
)


def _clob_market() -> dict[str, object]:
    return {
        "condition_id": TOOL.CONDITION_ID,
        "question": "Frozen question",
        "enable_order_book": True,
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "neg_risk": True,
        "minimum_order_size": 5,
        "minimum_tick_size": 0.001,
        "tokens": [
            {"outcome": "Yes", "token_id": TOOL.YES_TOKEN_ID},
            {"outcome": "No", "token_id": TOOL.NO_TOKEN_ID},
        ],
        "rewards": {
            "rates": [{"rewards_daily_rate": 6}],
            "min_size": 20,
            "max_spread": 4.5,
        },
    }


def _reward_market() -> dict[str, object]:
    return {
        "data": [
            {
                "condition_id": TOOL.CONDITION_ID,
                "rewards_min_size": 20,
                "rewards_max_spread": 4.5,
                "market_competitiveness": 0.8,
                "rewards_config": [{"rate_per_day": 6}],
            }
        ]
    }


def _gamma_event() -> dict[str, object]:
    return {
        "id": TOOL.EVENT_ID,
        "negRisk": True,
        "negRiskAugmented": True,
        "markets": [
            {
                "conditionId": TOOL.CONDITION_ID,
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
                "outcomes": '["Yes","No"]',
                "clobTokenIds": json.dumps([TOOL.YES_TOKEN_ID, TOOL.NO_TOKEN_ID]),
                "feesEnabled": True,
                "feeSchedule": {
                    "rate": 0.04,
                    "takerOnly": True,
                    "rebateRate": 0.25,
                },
                "rewardsMinSize": 20,
                "rewardsMaxSpread": 4.5,
                "orderPriceMinTickSize": 0.001,
            }
        ],
    }


def _book(token_id: str, *, bid: str, ask: str) -> dict[str, object]:
    return {
        "market": TOOL.CONDITION_ID,
        "asset_id": token_id,
        "timestamp": "2000000000000",
        "bids": [
            {"price": "0.400", "size": "100"},
            {"price": bid, "size": "100"},
        ],
        "asks": [
            {"price": "0.900", "size": "100"},
            {"price": ask, "size": "100"},
        ],
        "min_order_size": "5",
        "tick_size": "0.001",
        "neg_risk": True,
    }


class _Response:
    def __init__(
        self,
        payload: object,
        *,
        url: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.url = url
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.content = json.dumps(payload, sort_keys=True).encode("ascii")

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(
        self, *, rate_limited: bool = False, stale_books: bool = False
    ) -> None:
        self.calls: list[tuple[str, str, object | None]] = []
        self.rate_limited = rate_limited
        self.stale_books = stale_books

    def request(
        self,
        method: str,
        url: str,
        *,
        json: object | None,
        timeout: int,
    ) -> _Response:
        assert timeout == 30
        self.calls.append((method, url, json))
        if self.rate_limited:
            return _Response(
                {}, url=url, status_code=429, headers={"Retry-After": "13"}
            )
        if "/rewards/markets/" in url:
            payload = _reward_market()
        elif "gamma-api" in url:
            payload = _gamma_event()
        elif url.endswith("/books"):
            payload = [
                _book(TOOL.NO_TOKEN_ID, bid="0.465", ask="0.532"),
                _book(TOOL.YES_TOKEN_ID, bid="0.468", ask="0.535"),
            ]
            if self.stale_books:
                for book in payload:
                    book["timestamp"] = "1999999990000"
        else:
            payload = _clob_market()
        return _Response(payload, url=url)


def test_run_uses_four_public_requests_and_never_accepts_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 2_000_000_000_000_000_000)
    session = _Session()

    report = TOOL.run(session=session)

    assert len(session.calls) == 4
    assert session.calls[-1] == (
        "POST",
        f"{TOOL.CLOB_BASE_URL}/books",
        [{"token_id": TOOL.YES_TOKEN_ID}, {"token_id": TOOL.NO_TOKEN_ID}],
    )
    assert report["capture"]["freshness_passed"] is True
    assert report["conditional_quote_diagnostic"]["combined_bid_price"] == "0.935"
    assert report["conditional_quote_diagnostic"]["both_fill_gross_profit"] == "1.300"
    assert report["verdict"]["displayed_both_fill_gross_positive"] is True
    assert report["verdict"]["freshness_passed"] is True
    assert report["verdict"]["publicly_proven_reward_payout_lower_bound"] == "0"
    assert report["verdict"]["accepted_edge"] is False


def test_source_snapshot_hash_implementation_and_diagnostic_reconstruct() -> None:
    report = json.loads(SNAPSHOT.read_text(encoding="ascii"))
    expected_hash = report.pop("result_sha256")

    assert (
        hashlib.sha256(TOOL._canonical_json(report).encode("ascii")).hexdigest()
        == expected_hash
    )
    implementation = report["source_contract"]["implementation"]
    assert (
        implementation["tool_sha256"]
        == hashlib.sha256(
            (ROOT / "tools" / "screen_polymarket_paired_maker_rewards.py").read_bytes()
        ).hexdigest()
    )
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(
            (
                ROOT / "src" / "simple_ai_trading" / "polymarket_liquidity_rewards.py"
            ).read_bytes()
        ).hexdigest()
    )
    books = {book["asset_id"]: book for book in report["source_contract"]["books"]}
    candidate = report["candidate"]
    reconstructed = TOOL._quote_payload(
        yes_book=books[TOOL.YES_TOKEN_ID],
        no_book=books[TOOL.NO_TOKEN_ID],
        tick_size=Decimal(candidate["tick_size"]),
        reward_size=Decimal(candidate["reward_minimum_size"]),
        maximum_spread=Decimal(candidate["reward_maximum_spread_cents"]) / 100,
        daily_reward_rate=Decimal(candidate["reward_daily_rate"]),
    )
    assert reconstructed == report["conditional_quote_diagnostic"]


def test_rate_limit_stops_without_retry() -> None:
    session = _Session(rate_limited=True)
    with pytest.raises(RuntimeError, match="stopped without retry; Retry-After=13"):
        TOOL.run(session=session)
    assert len(session.calls) == 1


def test_stale_books_are_explicitly_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 2_000_000_000_000_000_000)
    report = TOOL.run(session=_Session(stale_books=True))

    assert report["capture"]["freshness_passed"] is False
    assert report["verdict"]["status"] == "rejected_stale_book_snapshot"


def test_augmented_event_contract_fails_closed() -> None:
    event = _gamma_event()
    event["negRiskAugmented"] = False
    with pytest.raises(ValueError, match="augmented"):
        TOOL._gamma_market(event)


def test_duplicate_book_fails_closed() -> None:
    book = _book(TOOL.YES_TOKEN_ID, bid="0.468", ask="0.535")
    with pytest.raises(ValueError, match="duplicated"):
        TOOL._book_rows([book, book])
