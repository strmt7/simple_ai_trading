from __future__ import annotations

from dataclasses import replace
import json

import pytest

from simple_ai_trading.polymarket_historical_l2 import (
    POLYMARKET_ORDERBOOK_HISTORY_URL,
    PolymarketHistoricalL2Client,
    decode_historical_l2_chunk,
    decode_historical_l2_window,
    encode_historical_l2_window,
    parse_historical_orderbook_page,
)


CONDITION_ID = "0x" + ("a" * 64)
ASSET_ID = "1" * 40


def _snapshot(
    timestamp_ms: int, *, bid: str = "0.4", ask: str = "0.6"
) -> dict[str, object]:
    return {
        "market": CONDITION_ID,
        "asset_id": ASSET_ID,
        "timestamp": str(timestamp_ms),
        "hash": "b" * 40,
        "bids": [
            {"price": "0.1", "size": "5"},
            {"price": bid, "size": "7.5"},
        ],
        "asks": [
            {"price": "0.9", "size": "6"},
            {"price": ask, "size": "8.5"},
        ],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
        "last_trade_price": "0.5",
    }


def _page(*snapshots: dict[str, object], count: int | None = None) -> bytes:
    return json.dumps(
        {"count": len(snapshots) if count is None else count, "data": snapshots},
        separators=(",", ":"),
    ).encode()


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        url: str = POLYMARKET_ORDERBOOK_HISTORY_URL,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self.cookies: list[object] = []
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class _ResponseCookieSession(_Session):
    def get(self, url: str, **kwargs: object) -> _Response:
        response = super().get(url, **kwargs)
        self.cookies.append("public-edge-cookie")
        return response


def test_page_parser_is_strict_and_canonicalizes_book_order() -> None:
    parsed = parse_historical_orderbook_page(
        _page(_snapshot(1_100)),
        expected_condition_id=CONDITION_ID,
        expected_asset_id=ASSET_ID,
        start_ms=1_000,
        end_ms=2_000,
    )

    assert parsed.remaining_record_count == 1
    assert [level.price for level in parsed.snapshots[0].bids] == ["0.4", "0.1"]
    assert [level.price for level in parsed.snapshots[0].asks] == ["0.6", "0.9"]
    assert len(parsed.source_payload_sha256) == 64
    assert len(parsed.snapshots[0].source_payload_sha256) == 64


@pytest.mark.parametrize(
    "raw,error",
    [
        (b'{"count":0,"count":0,"data":[]}', "duplicate keys"),
        (_page(_snapshot(999)), "outside the requested window"),
        (_page(_snapshot(1_100, bid="0.6", ask="0.6")), "crossed"),
    ],
)
def test_page_parser_rejects_ambiguous_or_invalid_evidence(
    raw: bytes, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        parse_historical_orderbook_page(
            raw,
            expected_condition_id=CONDITION_ID,
            expected_asset_id=ASSET_ID,
            start_ms=1_000,
            end_ms=2_000,
        )


def test_client_rejects_authority_headers_and_cookies() -> None:
    session = _Session([_Response(_page())])
    session.headers["POLY_API_KEY"] = "redacted"
    client = PolymarketHistoricalL2Client(
        session=session,
        minimum_request_interval_seconds=0,
    )
    with pytest.raises(ValueError, match="authority headers"):
        client.fetch_page(
            condition_id=CONDITION_ID,
            asset_id=ASSET_ID,
            start_ms=1_000,
            end_ms=2_000,
        )

    session.headers.clear()
    session.cookies.append(object())
    with pytest.raises(ValueError, match="contains cookies"):
        client.fetch_page(
            condition_id=CONDITION_ID,
            asset_id=ASSET_ID,
            start_ms=1_000,
            end_ms=2_000,
        )


def test_client_retries_bounded_transient_failure_without_credentials() -> None:
    session = _Session(
        [
            _Response(
                b'{"error":"busy"}', status_code=503, headers={"Retry-After": "1"}
            ),
            _Response(_page(_snapshot(1_100))),
        ]
    )
    sleeps: list[float] = []
    client = PolymarketHistoricalL2Client(
        session=session,
        minimum_request_interval_seconds=0,
        sleeper=sleeps.append,
    )

    page = client.fetch_page(
        condition_id=CONDITION_ID,
        asset_id=ASSET_ID,
        start_ms=1_000,
        end_ms=2_000,
    )

    assert len(page.snapshots) == 1
    assert sleeps == [1.0]
    assert len(session.calls) == 2
    for call in session.calls:
        headers = {str(key).lower() for key in call["headers"]}
        assert not headers & {
            "authorization",
            "poly_api_key",
            "poly_signature",
        }
        assert call["allow_redirects"] is False


def test_client_discards_public_response_cookies_before_next_request() -> None:
    session = _ResponseCookieSession([_Response(_page(_snapshot(1_100)))])
    client = PolymarketHistoricalL2Client(
        session=session,
        minimum_request_interval_seconds=0,
    )

    page = client.fetch_page(
        condition_id=CONDITION_ID,
        asset_id=ASSET_ID,
        start_ms=1_000,
        end_ms=2_000,
    )

    assert len(page.snapshots) == 1
    assert session.cookies == []


def test_closed_window_paginates_without_overlap_and_compresses_exactly() -> None:
    session = _Session(
        [
            _Response(_page(_snapshot(1_100), _snapshot(1_200), count=3)),
            _Response(_page(_snapshot(1_500), count=1)),
        ]
    )
    client = PolymarketHistoricalL2Client(
        session=session,
        minimum_request_interval_seconds=0,
        clock_ms=lambda: 1_000_000,
    )

    window = client.fetch_closed_window(
        condition_id=CONDITION_ID,
        asset_id=ASSET_ID,
        event_start_ms=1_000,
        event_end_ms=2_000,
        limit=2,
    )
    chunk = encode_historical_l2_window(window)
    decoded = decode_historical_l2_chunk(chunk)
    reconstructed = decode_historical_l2_window(chunk)

    assert [snapshot.timestamp_ms for snapshot in window.snapshots] == [
        1_100,
        1_200,
        1_500,
    ]
    assert session.calls[1]["params"]["startTs"] == 1_201
    assert decoded["source"] == {
        "authentication_used": False,
        "binance_used": False,
        "endpoint": POLYMARKET_ORDERBOOK_HISTORY_URL,
        "source_chain_sha256": window.source_chain_sha256,
    }
    assert not any(decoded["authority"].values())
    assert chunk.compressed_size_bytes < chunk.raw_size_bytes
    assert reconstructed == window

    tampered = bytearray(chunk.payload)
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="envelope differs"):
        decode_historical_l2_chunk(replace(chunk, payload=bytes(tampered)))


def test_window_rejects_active_or_empty_history() -> None:
    active = PolymarketHistoricalL2Client(
        session=_Session([]),
        minimum_request_interval_seconds=0,
        clock_ms=lambda: 2_100,
    )
    with pytest.raises(ValueError, match="not durably closed"):
        active.fetch_closed_window(
            condition_id=CONDITION_ID,
            asset_id=ASSET_ID,
            event_start_ms=1_000,
            event_end_ms=2_000,
        )

    empty = PolymarketHistoricalL2Client(
        session=_Session([_Response(_page())]),
        minimum_request_interval_seconds=0,
        clock_ms=lambda: 1_000_000,
    )
    with pytest.raises(ValueError, match="contains no snapshots"):
        empty.fetch_closed_window(
            condition_id=CONDITION_ID,
            asset_id=ASSET_ID,
            event_start_ms=1_000,
            event_end_ms=2_000,
        )
