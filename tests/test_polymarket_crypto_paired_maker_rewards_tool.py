from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_polymarket_crypto_paired_maker_rewards",
    ROOT / "tools" / "screen_polymarket_crypto_paired_maker_rewards.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
EPOCH = 1_787_640_000
NOW_MS = EPOCH * 1_000 + 60_000
ASSETS = ("BTC", "ETH", "SOL")


def _iso(epoch_seconds: int) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _identifiers(asset: str) -> tuple[str, str, str]:
    digit = {"BTC": "7", "ETH": "8", "SOL": "9"}[asset]
    return "0x" + digit * 64, digit * 40, digit * 39 + "1"


def _market(asset: str) -> dict[str, object]:
    condition, up_token, down_token = _identifiers(asset)
    lower = asset.lower()
    return {
        "id": f"market-{asset}-{EPOCH}",
        "question": f"{asset} Up or Down",
        "conditionId": condition,
        "slug": f"{lower}-updown-5m-{EPOCH}",
        "eventStartTime": _iso(EPOCH),
        "endDate": _iso(EPOCH + 300),
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([up_token, down_token]),
        "outcomes": '["Up","Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 1000,
        "volumeNum": 2000,
        "resolutionSource": (
            f"https://data.chain.link/streams/{lower}-usd-twap-60s-streams"
        ),
        "cryptoMarketConfigId": f"{lower}-5m-twap-60",
        "cryptoMarketConfig": {
            "asset": lower,
            "duration": "5m",
            "id": f"{lower}-5m-twap-60",
            "twapEnabled": True,
            "twapLookbackSeconds": 60,
        },
        "rewardsMinSize": 50,
        "rewardsMaxSpread": 1.5,
    }


def _reward(asset: str) -> dict[str, object]:
    condition, _up_token, _down_token = _identifiers(asset)
    rate = {"BTC": 10000, "ETH": 1666.666667, "SOL": 1666.666667}[asset]
    return {
        "data": [
            {
                "condition_id": condition,
                "rewards_min_size": 50,
                "rewards_max_spread": 1.5,
                "market_competitiveness": 0.01,
                "rewards_config": [
                    {
                        "asset_address": "0x" + "a" * 40,
                        "start_date": "2026-08-25",
                        "end_date": "2500-12-31",
                        "rate_per_day": rate,
                        "total_rewards": 0,
                        "id": 0,
                    }
                ],
            }
        ],
        "next_cursor": "LTE=",
    }


def _book(
    asset: str,
    token: str,
    *,
    best_bid: str = "0.48",
    timestamp_ms: int = NOW_MS,
) -> dict[str, object]:
    condition, _up_token, _down_token = _identifiers(asset)
    return {
        "market": condition,
        "asset_id": token,
        "timestamp": str(timestamp_ms),
        "bids": [
            {"price": "0.40", "size": "100"},
            {"price": best_bid, "size": "100"},
        ],
        "asks": [
            {"price": "0.90", "size": "100"},
            {"price": "0.52", "size": "100"},
        ],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
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
        self.content = TOOL._canonical_json(payload).encode("ascii")

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(
        self,
        *,
        rate_limited: bool = False,
        stale: bool = False,
        crossing_assets: frozenset[str] = frozenset(),
    ) -> None:
        self.rate_limited = rate_limited
        self.stale = stale
        self.crossing_assets = crossing_assets
        self.calls: list[tuple[str, str, object, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: object,
        json: object,
        timeout: int,
    ) -> _Response:
        assert timeout == 30
        self.calls.append((method, url, params, json))
        if self.rate_limited:
            return _Response(
                {},
                url=url,
                status_code=429,
                headers={"Retry-After": "17"},
            )
        if "gamma-api" in url:
            payload: object = [_market(asset) for asset in reversed(ASSETS)]
        elif "/rewards/markets/" in url:
            asset = next(asset for asset in ASSETS if _identifiers(asset)[0] in url)
            payload = _reward(asset)
        else:
            books = []
            for asset in ASSETS:
                _condition, up_token, down_token = _identifiers(asset)
                bid = "0.49" if asset in self.crossing_assets else "0.48"
                timestamp = NOW_MS - 10_000 if self.stale else NOW_MS
                books.extend(
                    [
                        _book(asset, down_token, best_bid=bid, timestamp_ms=timestamp),
                        _book(asset, up_token, best_bid=bid, timestamp_ms=timestamp),
                    ]
                )
            payload = list(reversed(books))
        return _Response(payload, url=url)


def test_run_uses_five_requests_and_only_advances_to_capture_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: NOW_MS * 1_000_000)
    monkeypatch.setattr(TOOL.time, "monotonic_ns", lambda: 123)
    session = _Session(crossing_assets=frozenset({"ETH"}))

    report = TOOL.run(session=session)

    assert len(session.calls) == 5
    assert [row["asset"] for row in report["asset_results"]] == list(ASSETS)
    diagnostics = {row["asset"]: row["diagnostic"] for row in report["asset_results"]}
    assert diagnostics["BTC"]["economics"]["combined_bid_price"] == "0.98"
    assert diagnostics["BTC"]["economics"]["both_fill_gross_profit"] == "1.00"
    assert diagnostics["BTC"]["status"] == "prospective_capture_design_candidate"
    assert diagnostics["ETH"]["status"] == "rejected_combined_bid_not_below_one"
    assert report["verdict"]["candidate_assets"] == ["BTC", "SOL"]
    assert report["verdict"]["prospective_capture_design_eligible"] is True
    assert report["verdict"]["publicly_proven_reward_payout_lower_bound"] == "0"
    assert report["verdict"]["accepted_edge"] is False
    assert report["safety"]["orders_placed"] is False


def test_stale_books_reject_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: NOW_MS * 1_000_000)
    report = TOOL.run(session=_Session(stale=True))

    assert report["capture"]["freshness_passed"] is False
    assert report["verdict"]["status"] == "rejected_without_resampling"
    assert report["verdict"]["prospective_capture_design_eligible"] is False


def test_rate_limit_stops_without_retry() -> None:
    session = _Session(rate_limited=True)
    with pytest.raises(RuntimeError, match="without retry; Retry-After=17"):
        TOOL.run(session=session)
    assert len(session.calls) == 1


def test_missing_gamma_asset_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: NOW_MS * 1_000_000)
    session = _Session()
    original = session.request

    def request_without_sol(*args: object, **kwargs: object) -> _Response:
        response = original(*args, **kwargs)
        if "gamma-api" in str(args[1]):
            return _Response([_market("BTC"), _market("ETH")], url=str(args[1]))
        return response

    session.request = request_without_sol  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="frozen three slugs"):
        TOOL.run(session=session)
    assert len(session.calls) == 1
