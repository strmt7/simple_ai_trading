from __future__ import annotations

import ast
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_historical_screen import (
    HistoricalScreenStore,
    PublicPayload,
    load_historical_screen_contract,
    parse_historical_btc_event,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-screen-v2.json"
)
START_MS = int(datetime(2026, 3, 20, 12, 0, tzinfo=UTC).timestamp() * 1_000)
END_MS = START_MS + 300_000
SLUG = f"btc-updown-5m-{START_MS // 1_000}"
CONDITION = "0x" + "1" * 64
UP_TOKEN = "2" * 40
DOWN_TOKEN = "3" * 40


def _event() -> dict[str, object]:
    market = {
        "id": "1234567",
        "conditionId": CONDITION,
        "slug": SLUG,
        "question": "Bitcoin Up or Down - test",
        "eventStartTime": "2026-03-20T12:00:00Z",
        "endDate": "2026-03-20T12:05:00Z",
        "active": True,
        "closed": True,
        "enableOrderBook": True,
        "acceptingOrders": False,
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["1", "0"]',
        "clobTokenIds": f'["{UP_TOKEN}", "{DOWN_TOKEN}"]',
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "rate": 0.07,
            "exponent": 1,
            "takerOnly": True,
            "rebateRate": 0,
        },
    }
    return {
        "id": "7654321",
        "ticker": SLUG,
        "slug": SLUG,
        "closed": True,
        "series": [{"id": "10684"}],
        "markets": [market],
    }


def _public(value: object, *, observed_at_ms: int = END_MS + 1_000) -> PublicPayload:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return PublicPayload(
        value=value,
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        observed_at_ms=observed_at_ms,
    )


def _terminal_payloads(*, winner: str = "Up") -> tuple[PublicPayload, PublicPayload]:
    winner_index = 0 if winner == "Up" else 1
    prices = ["0", "0"]
    prices[winner_index] = "1"
    gamma = deepcopy(_event()["markets"][0])
    assert isinstance(gamma, dict)
    gamma.update(
        {
            "active": False,
            "closed": True,
            "acceptingOrders": False,
            "outcomePrices": json.dumps(prices),
        }
    )
    clob = {
        "condition_id": CONDITION,
        "market_slug": SLUG,
        "closed": True,
        "active": False,
        "accepting_orders": False,
        "tokens": [
            {
                "token_id": token,
                "outcome": outcome,
                "winner": index == winner_index,
                "price": 1 if index == winner_index else 0,
            }
            for index, (token, outcome) in enumerate(
                ((UP_TOKEN, "Up"), (DOWN_TOKEN, "Down"))
            )
        ],
    }
    return _public(gamma), _public(clob)


def test_contract_hash_and_authority_boundary_are_frozen() -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)

    assert contract.contract_sha256 == (
        "086365c0488ac404124ac1a9ff14f7a96784576b53baeeed1b725e9b8999fb79"
    )
    assert contract.eligible_days == (
        "2026-03-20",
        "2026-04-20",
        "2026-05-01",
        "2026-06-22",
    )
    assert contract.role_for_day("2026-06-22") == "test"


def test_target_free_identity_parser_omits_terminal_outcome() -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    market = parse_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1_000,
    )

    assert market.role == "train"
    assert market.condition_id == CONDITION
    assert market.token_ids == (UP_TOKEN, DOWN_TOKEN)
    assert "outcomePrices" not in market.identity_payload_json
    assert '"winner"' not in market.identity_payload_json
    assert market.excluded is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda event: event.update(closed=False), "event identity"),
        (
            lambda event: event.update(series=[{"id": "999"}]),
            "event series",
        ),
        (
            lambda event: event["markets"][0].update(acceptingOrders=True),
            "closed BTC CLOB",
        ),
        (
            lambda event: event["markets"][0].update(
                resolutionSource="https://example.invalid"
            ),
            "closed BTC CLOB",
        ),
        (
            lambda event: event["markets"][0].update(
                eventStartTime="2026-03-20T12:00:01Z"
            ),
            "window differs",
        ),
    ],
)
def test_identity_parser_fails_closed_on_source_drift(change, message: str) -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    event = _event()
    change(event)

    with pytest.raises(ValueError, match=message):
        parse_historical_btc_event(
            event,
            contract=contract,
            observed_at_ms=END_MS + 1_000,
        )


def test_development_and_test_targets_are_phase_separated(tmp_path: Path) -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    market = parse_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1_000,
    )
    gamma, clob = _terminal_payloads()
    with HistoricalScreenStore(tmp_path / "screen.duckdb", contract=contract) as store:
        store.upsert_market(market)
        store.transition("initialized", "identities_complete")
        with pytest.raises(ValueError, match="not authorized"):
            store.record_resolution(market=market, gamma=gamma, clob=clob)

        store.transition("identities_complete", "features_complete")
        assert store.record_resolution(market=market, gamma=gamma, clob=clob) == "Up"
        assert (
            store.connect()
            .execute(
                "SELECT count(*) FROM target.official_resolution WHERE role = 'train'"
            )
            .fetchone()[0]
            == 1
        )
        store.transition("features_complete", "development_targets_complete")
        store.transition("development_targets_complete", "pretest_complete")
        with pytest.raises(ValueError, match="not authorized"):
            store.record_resolution(market=market, gamma=gamma, clob=clob)


def test_test_target_cannot_open_before_pretest_manifest(tmp_path: Path) -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    event = _event()
    event_start = int(datetime(2026, 6, 22, 12, 0, tzinfo=UTC).timestamp())
    event["slug"] = f"btc-updown-5m-{event_start}"
    event["ticker"] = event["slug"]
    market_payload = event["markets"][0]
    assert isinstance(market_payload, dict)
    market_payload["slug"] = event["slug"]
    market_payload["eventStartTime"] = "2026-06-22T12:00:00Z"
    market_payload["endDate"] = "2026-06-22T12:05:00Z"
    market_payload["conditionId"] = "0x" + "4" * 64
    market_payload["clobTokenIds"] = json.dumps(["5" * 40, "6" * 40])
    market = parse_historical_btc_event(
        event,
        contract=contract,
        observed_at_ms=(event_start + 301) * 1_000,
    )
    gamma_value = deepcopy(market_payload)
    gamma_value["outcomePrices"] = '["1", "0"]'
    clob_value = {
        "condition_id": market.condition_id,
        "market_slug": market.slug,
        "closed": True,
        "active": False,
        "accepting_orders": False,
        "tokens": [
            {
                "token_id": market.up_token_id,
                "outcome": "Up",
                "winner": True,
                "price": 1,
            },
            {
                "token_id": market.down_token_id,
                "outcome": "Down",
                "winner": False,
                "price": 0,
            },
        ],
    }
    gamma = _public(gamma_value, observed_at_ms=market.end_ms + 1_000)
    clob = _public(clob_value, observed_at_ms=market.end_ms + 1_000)

    with HistoricalScreenStore(tmp_path / "test.duckdb", contract=contract) as store:
        store.upsert_market(market)
        store.transition("initialized", "identities_complete")
        store.transition("identities_complete", "features_complete")
        with pytest.raises(ValueError, match="not authorized"):
            store.record_resolution(market=market, gamma=gamma, clob=clob)
        store.transition("features_complete", "development_targets_complete")
        with pytest.raises(ValueError, match="not authorized"):
            store.record_resolution(market=market, gamma=gamma, clob=clob)
        store.transition("development_targets_complete", "pretest_complete")
        assert store.record_resolution(market=market, gamma=gamma, clob=clob) == "Up"


def test_historical_source_has_no_account_or_execution_boundary() -> None:
    module_path = ROOT / "src" / "simple_ai_trading" / "polymarket_historical_screen.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any("binance" in name.lower() for name in imports)
    for forbidden in (
        "private_key",
        "api_secret",
        "create_order",
        "place_order",
        "cancel_order",
        "account_balance",
        "position_risk",
    ):
        assert forbidden not in source.lower()
