from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
)
from simple_ai_trading.polymarket_resolution import (
    PolymarketResolutionFinalizer,
    load_official_resolutions,
    validate_official_resolution,
)
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay


EPOCH = 1_784_058_600


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market_payload(asset: str) -> dict[str, object]:
    digit = {"BTC": "7", "ETH": "8", "SOL": "9"}[asset]
    market_id = {"BTC": "1001", "ETH": "1002", "SOL": "1003"}[asset]
    return {
        "id": market_id,
        "question": f"{asset} Up or Down",
        "conditionId": "0x" + digit * 64,
        "slug": f"{asset.lower()}-updown-5m-{EPOCH}",
        "eventStartTime": "2026-07-14T19:50:00Z",
        "endDate": "2026-07-14T19:55:00Z",
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([digit * 40, digit * 39 + "1"]),
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["0.5", "0.5"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000.5,
        "volumeNum": 50_000.25,
        "resolutionSource": f"https://data.chain.link/streams/{asset.lower()}-usd",
    }


def _evidence(asset: str) -> MarketEvidence:
    market = parse_polymarket_five_minute_market(_market_payload(asset))
    clob = _canonical({"c": market.condition_id, "t": list(market.token_ids)})
    fee = _canonical({"base_fee": 1000})
    return MarketEvidence(
        market=market,
        observed_wall_ms=EPOCH * 1_000 + 100,
        observed_monotonic_ns=100_000_000,
        clob_info_json=clob,
        clob_info_sha256=_sha(clob),
        up_fee_rate_json=fee,
        up_fee_rate_sha256=_sha(fee),
        down_fee_rate_json=fee,
        down_fee_rate_sha256=_sha(fee),
        maker_base_fee=1000,
        taker_base_fee=1000,
        taker_order_delay_enabled=False,
        minimum_order_age_seconds=0,
    )


def _official_payloads(asset: str, *, winner: str = "Up") -> tuple[dict[str, object], dict[str, object]]:
    gamma = deepcopy(_market_payload(asset))
    market = parse_polymarket_five_minute_market(_market_payload(asset))
    winner_index = 0 if winner == "Up" else 1
    prices = ["0", "0"]
    prices[winner_index] = "1"
    gamma.update(
        {
            "closed": True,
            "active": False,
            "acceptingOrders": False,
            "outcomePrices": json.dumps(prices),
        }
    )
    clob_tokens = []
    for index, (token_id, outcome) in enumerate(
        zip(market.token_ids, ("Up", "Down"), strict=True)
    ):
        clob_tokens.append(
            {
                "token_id": token_id,
                "outcome": outcome,
                "price": 1 if index == winner_index else 0,
                "winner": index == winner_index,
            }
        )
    clob = {
        "condition_id": market.condition_id,
        "market_slug": market.slug,
        "closed": True,
        "active": False,
        "accepting_orders": False,
        "tokens": clob_tokens,
    }
    return clob, gamma


def _complete_store(path: Path, run_id: str = "resolution-run") -> PolymarketEvidenceStore:
    store = PolymarketEvidenceStore(path)
    store.connect()
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    for sequence, stream in enumerate(
        ("clob_market", "polymarket_rtds", "binance_spot"), start=1
    ):
        if stream == "clob_market":
            btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
            payload: object = {
                "event_type": "book",
                "market": btc.condition_id,
                "asset_id": btc.up_token_id,
                "timestamp": str(EPOCH * 1_000 + 1_000),
                "hash": "resolution-fixture-book",
                "bids": [{"price": "0.49", "size": "10"}],
                "asks": [{"price": "0.51", "size": "10"}],
            }
        else:
            payload = {"event_type": "fixture"}
        store.append_messages(
            run_id,
            [
                RawStreamMessage(
                    stream=stream,
                    connection_id=f"{stream}-connection",
                    sequence_number=sequence,
                    received_wall_ms=EPOCH * 1_000 + 1_000 + sequence,
                    received_monotonic_ns=1_000_000_000 + sequence,
                    raw_text=_canonical(payload),
                )
            ],
        )
    report = store.finish_run(
        run_id,
        started_at_ms=EPOCH * 1_000,
        ended_at_ms=EPOCH * 1_000 + 400_000,
        database=str(path),
        errors=(),
    )
    assert report.status == "complete"
    return store


class _OfficialClient:
    def __init__(self) -> None:
        self.clob_calls: list[str] = []
        self.gamma_calls: list[str] = []
        self.by_condition = {}
        self.by_market = {}
        for asset in ("BTC", "ETH", "SOL"):
            clob, gamma = _official_payloads(asset)
            self.by_condition[str(clob["condition_id"])] = clob
            self.by_market[str(gamma["id"])] = gamma

    def clob_market(self, condition_id: str) -> dict[str, object]:
        self.clob_calls.append(condition_id)
        return deepcopy(self.by_condition[condition_id])

    def gamma_market(self, market_id: str) -> dict[str, object]:
        self.gamma_calls.append(market_id)
        return deepcopy(self.by_market[market_id])


def test_resolution_waits_for_both_sources_and_rejects_disagreement() -> None:
    market = parse_polymarket_five_minute_market(_market_payload("BTC"))
    clob, gamma = _official_payloads("BTC", winner="Down")
    gamma["closed"] = False
    assert validate_official_resolution(
        market,
        clob,
        gamma,
        observed_wall_ms=market.end_ms + 1,
    ) is None

    gamma["closed"] = True
    assert validate_official_resolution(
        market,
        clob,
        gamma,
        observed_wall_ms=market.end_ms + 1,
    ) == (market.down_token_id, "Down")

    gamma["outcomePrices"] = '["1", "0"]'
    with pytest.raises(ValueError, match="disagree"):
        validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=market.end_ms + 1,
        )


def test_resolution_requires_terminal_prices_and_non_accepting_market() -> None:
    market = parse_polymarket_five_minute_market(_market_payload("BTC"))
    clob, gamma = _official_payloads("BTC")
    clob["accepting_orders"] = True
    with pytest.raises(ValueError, match="accepts orders"):
        validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=market.end_ms + 1,
        )

    clob["accepting_orders"] = False
    gamma["outcomePrices"] = '["0.9995", "0.0005"]'
    with pytest.raises(ValueError, match="terminal outcome prices"):
        validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=market.end_ms + 1,
        )


def test_finalizer_is_immutable_idempotent_and_detects_tampering(tmp_path: Path) -> None:
    store = _complete_store(tmp_path / "resolution.duckdb")
    client = _OfficialClient()
    finalizer = PolymarketResolutionFinalizer(
        store,
        client=client,  # type: ignore[arg-type]
        wall_clock_ms=lambda: EPOCH * 1_000 + 400_001,
        monotonic_clock_ns=lambda: 9_000_000_000,
    )
    try:
        first = finalizer.finalize(run_id="resolution-run")
        second = finalizer.finalize(run_id="resolution-run")
        assert first.status == "complete"
        assert first.finalized_count == first.newly_finalized_count == 3
        assert second.status == "complete"
        assert second.newly_finalized_count == 0
        assert len(client.clob_calls) == len(client.gamma_calls) == 3
        resolutions = load_official_resolutions(store, run_id="resolution-run")
        assert {item.asset for item in resolutions} == {"BTC", "ETH", "SOL"}
        assert all(item.winning_outcome == "Up" for item in resolutions)
        replay = PolymarketEvidenceReplay.load(store, run_id="resolution-run")
        assert len(replay.resolutions) == 3
        assert {item.source for item in replay.resolutions} == {
            "clob_gamma_crosscheck"
        }

        store.connect().execute(
            """
            UPDATE polymarket_resolution_evidence
            SET winning_outcome = 'Down'
            WHERE asset = 'BTC'
            """
        )
        with pytest.raises(ValueError, match="winner drifted"):
            load_official_resolutions(store, run_id="resolution-run")
    finally:
        store.close()
