from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_polymarket_duplicate_parity",
    ROOT / "tools" / "screen_polymarket_duplicate_parity.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "19"}


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, _url: str, *, params: object, timeout: int) -> _RateLimitedResponse:
        assert params is None
        assert timeout == 30
        self.calls += 1
        return _RateLimitedResponse()


def _market() -> dict[str, object]:
    return {
        "id": "1",
        "conditionId": "0x" + "1" * 64,
        "question": "Will X happen?",
        "description": "Resolves Yes exactly when X happens.",
        "endDate": "2027-01-01T00:00:00Z",
        "resolutionSource": "",
        "groupItemTitle": "X",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["10", "11"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "negRisk": False,
    }


def test_rate_limit_stops_without_retry() -> None:
    session = _Session()
    with pytest.raises(RuntimeError, match="stopped without retry; Retry-After=19"):
        TOOL._get(session, "https://gamma-api.polymarket.com/events")
    assert session.calls == 1


def test_embedded_and_canonical_market_identity_reconstructs() -> None:
    market = _market()
    embedded = TOOL._eligible_embedded_market(market, event_id="100")
    terms = TOOL._canonical_terms(
        {
            "id": "100",
            "active": True,
            "closed": False,
            "markets": [market],
        },
        selected_market_ids={"1"},
    )

    assert embedded == {
        "event_id": "100",
        "market_id": "1",
        "condition_id": "0x" + "1" * 64,
        "question": "Will X happen?",
    }
    assert terms[0].market_id == "1"
    assert terms[0].token_ids == ("10", "11")


def test_nonbinary_or_ineligible_embedded_market_is_excluded() -> None:
    assert (
        TOOL._eligible_embedded_market(
            {**_market(), "acceptingOrders": False}, event_id="100"
        )
        is None
    )
    assert (
        TOOL._eligible_embedded_market(
            {**_market(), "outcomes": '["Up", "Down"]'}, event_id="100"
        )
        is None
    )


def test_canonical_market_omission_fails_closed() -> None:
    with pytest.raises(ValueError, match="omitted"):
        TOOL._canonical_terms(
            {
                "id": "100",
                "active": True,
                "closed": False,
                "markets": [_market()],
            },
            selected_market_ids={"2"},
        )


def test_invalid_json_list_fails_closed() -> None:
    with pytest.raises(ValueError, match="is not JSON"):
        TOOL._json_list("not-json", name="field")
