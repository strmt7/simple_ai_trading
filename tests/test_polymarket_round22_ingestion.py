from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_historical_l2 import (
    HistoricalBookLevel,
    HistoricalBookSnapshot,
    HistoricalL2Window,
)
from simple_ai_trading.polymarket_round22_ingestion import (
    POLYMARKET_GAMMA_EVENT_BY_SLUG_URL,
    Round22GammaIdentityClient,
    ingest_round22_development_conditions,
    sanitize_round22_identity_event,
)
from simple_ai_trading.polymarket_round22_pilot import (
    Round22PilotStore,
    development_conditions,
    load_round22_pilot_contract,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONDITION_ID = "0x" + ("a" * 64)
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40


def _iso(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _event(slug: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "id": "70001",
        "ticker": slug,
        "slug": slug,
        "closed": True,
        "resolution": "Up",
        "series": [{"id": "10684", "title": "Bitcoin five minute"}],
        "markets": [
            {
                "id": "80001",
                "conditionId": CONDITION_ID,
                "slug": slug,
                "question": "Will Bitcoin be up or down?",
                "eventStartTime": _iso(start_ms),
                "endDate": _iso(end_ms),
                "active": False,
                "closed": True,
                "enableOrderBook": True,
                "acceptingOrders": False,
                "outcomes": '["Up","Down"]',
                "outcomePrices": '["1","0"]',
                "clobTokenIds": json.dumps([UP_TOKEN_ID, DOWN_TOKEN_ID]),
                "resolutionSource": "https://data.chain.link/streams/btc-usd",
                "orderPriceMinTickSize": 0.01,
                "orderMinSize": 5,
                "feesEnabled": True,
                "feeSchedule": {
                    "rate": 0.25,
                    "exponent": 2,
                    "takerOnly": True,
                    "rebateRate": 0,
                },
                "winner": "Up",
            }
        ],
    }


class _Response:
    def __init__(
        self,
        value: object,
        *,
        slug: str,
        status_code: int = 200,
    ) -> None:
        self.content = json.dumps(value, separators=(",", ":")).encode()
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.url = f"{POLYMARKET_GAMMA_EVENT_BY_SLUG_URL}/{slug}"


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self.cookies: list[object] = []
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class _L2Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_closed_window(
        self,
        *,
        condition_id: str,
        asset_id: str,
        event_start_ms: int,
        event_end_ms: int,
        limit: int = 1_000,
    ) -> HistoricalL2Window:
        del limit
        self.calls.append(asset_id)
        snapshot = HistoricalBookSnapshot(
            condition_id=condition_id,
            asset_id=asset_id,
            timestamp_ms=event_start_ms + 100,
            book_hash="b" * 40,
            bids=(HistoricalBookLevel(price="0.4", size="10"),),
            asks=(HistoricalBookLevel(price="0.6", size="10"),),
            minimum_order_size="5",
            tick_size="0.01",
            negative_risk=False,
            last_trade_price="0.5",
            source_payload_sha256="c" * 64,
        )
        return HistoricalL2Window(
            condition_id=condition_id,
            asset_id=asset_id,
            event_start_ms=event_start_ms,
            event_end_ms=event_end_ms,
            snapshots=(snapshot,),
            source_chain_sha256="d" * 64,
        )


def test_round22_identity_sanitizer_drops_targets_before_parser_boundary() -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    expected = development_conditions(contract)[0]

    sanitized = sanitize_round22_identity_event(
        _event(expected.slug, expected.event_start_ms, expected.event_end_ms)
    )
    encoded = json.dumps(sanitized, sort_keys=True)

    assert "outcomePrices" not in encoded
    assert "winner" not in encoded
    assert 'resolution"' not in encoded
    assert sanitized["series"] == [{"id": "10684"}]


def test_round22_identity_client_is_public_only_retry_bounded_and_target_blind() -> (
    None
):
    contract = load_round22_pilot_contract(REPOSITORY)
    expected = development_conditions(contract)[0]
    event = _event(expected.slug, expected.event_start_ms, expected.event_end_ms)
    session = _Session(
        [
            _Response({"error": "busy"}, slug=expected.slug, status_code=503),
            _Response(event, slug=expected.slug),
        ]
    )
    sleeps: list[float] = []
    client = Round22GammaIdentityClient(
        session=session,
        minimum_request_interval_seconds=0,
        clock_ms=lambda: expected.event_end_ms + 600_000,
        sleeper=sleeps.append,
    )

    market = client.fetch_identity(expected, contract=contract)

    assert market.slug == expected.slug
    assert "outcomePrices" not in market.identity_payload_json
    assert "winner" not in market.identity_payload_json
    assert sleeps == [0.5]
    assert len(session.calls) == 2
    assert all(call["allow_redirects"] is False for call in session.calls)
    assert all("params" not in call for call in session.calls)

    session.headers["POLY_API_KEY"] = "redacted"
    with pytest.raises(ValueError, match="authority headers"):
        client.fetch_identity(expected, contract=contract)


def test_round22_operator_commits_one_condition_then_resumes_without_network(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    expected = development_conditions(contract)[0]
    contract = replace(contract, conditions=(expected,))
    event = _event(expected.slug, expected.event_start_ms, expected.event_end_ms)
    session = _Session([_Response(event, slug=expected.slug)])
    identity = Round22GammaIdentityClient(
        session=session,
        minimum_request_interval_seconds=0,
        clock_ms=lambda: expected.event_end_ms + 600_000,
    )
    l2 = _L2Client()
    progress: list[tuple[str, dict[str, object]]] = []

    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as store:
        result = ingest_round22_development_conditions(
            store=store,
            contract=contract,
            identity_client=identity,
            l2_client=l2,
            progress=lambda phase, payload: progress.append((phase, dict(payload))),
        )
        resumed = ingest_round22_development_conditions(
            store=store,
            contract=contract,
            identity_client=identity,
            l2_client=l2,
            maximum_conditions=1,
        )

        assert result.committed_slugs == (expected.slug,)
        assert result.selection_role == "all_development"
        assert result.committed_count == 1
        assert result.remaining_development_count == 0
        assert not any(
            (
                result.target_accessed,
                result.binance_used,
                result.authentication_used,
                result.paper_trading_authority,
                result.live_trading_authority,
            )
        )
        assert resumed.committed_count == 0
        assert resumed.already_complete_count == 1
        assert len(session.calls) == 1
        assert l2.calls == [UP_TOKEN_ID, DOWN_TOKEN_ID]
        assert [phase for phase, _ in progress] == [
            "identity_fetch",
            "up_book_fetch",
            "down_book_fetch",
            "condition_committed",
        ]
        assert store.target_row_count() == 0


@pytest.mark.parametrize("maximum", [0, 49, True])
def test_round22_operator_rejects_unbounded_runs(
    tmp_path: Path, maximum: object
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as store:
        with pytest.raises(ValueError, match="condition limit"):
            ingest_round22_development_conditions(
                store=store,
                contract=contract,
                identity_client=Round22GammaIdentityClient(),
                l2_client=_L2Client(),
                maximum_conditions=maximum,  # type: ignore[arg-type]
            )


def test_round22_operator_can_schedule_one_frozen_role_without_crossing_partitions(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    selected = next(
        item for item in contract.conditions if item.role == "tune_selection"
    )
    contract = replace(contract, conditions=(selected,))
    event = _event(selected.slug, selected.event_start_ms, selected.event_end_ms)
    identity = Round22GammaIdentityClient(
        session=_Session([_Response(event, slug=selected.slug)]),
        minimum_request_interval_seconds=0,
        clock_ms=lambda: selected.event_end_ms + 600_000,
    )
    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as store:
        result = ingest_round22_development_conditions(
            store=store,
            contract=contract,
            identity_client=identity,
            l2_client=_L2Client(),
            role="tune_selection",
        )

    assert result.selection_role == "tune_selection"
    assert result.committed_slugs == (selected.slug,)
