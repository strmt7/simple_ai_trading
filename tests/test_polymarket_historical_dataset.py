from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import duckdb

from simple_ai_trading.polymarket_historical_dataset import (
    FEATURE_NAMES,
    HistoricalFeatureRow,
    _insert_feature_rows,
    build_historical_feature_row,
)
from simple_ai_trading.polymarket_historical_screen import (
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
DAY_START_MS = int(datetime(2026, 3, 20, tzinfo=UTC).timestamp() * 1_000)
EVENT_START_MS = DAY_START_MS + 60_000
SLUG = f"btc-updown-5m-{EVENT_START_MS // 1_000}"
CONDITION = "0x" + "1" * 64
UP_TOKEN = "2" * 40
DOWN_TOKEN = "3" * 40


def _event() -> dict[str, object]:
    return {
        "id": "7654321",
        "ticker": SLUG,
        "slug": SLUG,
        "closed": True,
        "series": [{"id": "10684"}],
        "markets": [
            {
                "id": "1234567",
                "conditionId": CONDITION,
                "slug": SLUG,
                "question": "Bitcoin Up or Down - causal test",
                "eventStartTime": "2026-03-20T00:01:00Z",
                "endDate": "2026-03-20T00:06:00Z",
                "active": True,
                "closed": True,
                "enableOrderBook": True,
                "acceptingOrders": False,
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["1", "0"]',
                "clobTokenIds": json.dumps([UP_TOKEN, DOWN_TOKEN]),
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
        ],
    }


def _flow() -> dict[str, np.ndarray]:
    index = np.arange(86_400, dtype=np.float64)
    output: dict[str, np.ndarray] = {
        "second_ms": DAY_START_MS + np.arange(86_400, dtype=np.int64) * 1_000,
    }
    for prefix, base in (("spot", 100_000.0), ("perpetual", 100_010.0)):
        quote = np.full(86_400, 1_000.0, dtype=np.float64)
        output[f"{prefix}_close"] = base * np.exp(index * 1e-7)
        output[f"{prefix}_quote_volume"] = quote
        output[f"{prefix}_aggressive_buy_quote"] = quote * 0.55
        output[f"{prefix}_aggressive_sell_quote"] = quote * 0.45
        output[f"{prefix}_aggregate_count"] = np.full(86_400, 10.0)
        output[f"{prefix}_constituent_trade_count"] = np.full(86_400, 12.0)
        output[f"{prefix}_maximum_aggregate_quote"] = np.full(86_400, 200.0)
        output[f"{prefix}_squared_aggregate_quote_sum"] = np.full(86_400, 100_000.0)
        output[f"{prefix}_last_trade_age_seconds"] = np.zeros(86_400)
    return output


def test_feature_row_is_finite_and_ignores_future_flow() -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    market = parse_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=EVENT_START_MS + 301_000,
    )
    flow = _flow()
    first = build_historical_feature_row(
        market,
        day_start_ms=DAY_START_MS,
        decision_offset_seconds=30,
        flow=flow,
    )
    mutated = deepcopy(flow)
    first_unavailable_index = (EVENT_START_MS - DAY_START_MS) // 1_000 + 30
    mutated["spot_close"][first_unavailable_index:] *= 2
    mutated["spot_quote_volume"][first_unavailable_index:] *= 10
    second = build_historical_feature_row(
        market,
        day_start_ms=DAY_START_MS,
        decision_offset_seconds=30,
        flow=mutated,
    )

    assert first.decision_time_ms == EVENT_START_MS + 30_000
    assert first.feature_values.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(first.feature_values))
    np.testing.assert_array_equal(first.feature_values, second.feature_values)
    assert first.row_sha256 == second.row_sha256


def test_last_completed_second_changes_the_row() -> None:
    contract = load_historical_screen_contract(CONTRACT_PATH)
    market = parse_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=EVENT_START_MS + 301_000,
    )
    flow = _flow()
    first = build_historical_feature_row(
        market,
        day_start_ms=DAY_START_MS,
        decision_offset_seconds=30,
        flow=flow,
    )
    mutated = deepcopy(flow)
    last_completed_index = (EVENT_START_MS - DAY_START_MS) // 1_000 + 29
    mutated["spot_close"][last_completed_index] *= 1.01
    second = build_historical_feature_row(
        market,
        day_start_ms=DAY_START_MS,
        decision_offset_seconds=30,
        flow=mutated,
    )

    assert first.row_sha256 != second.row_sha256


def test_dataset_module_cannot_read_polymarket_prices_or_targets() -> None:
    source = (
        ROOT / "src" / "simple_ai_trading" / "polymarket_historical_dataset.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "target.",
        "official_resolution",
        "winning_outcome",
        "outcomePrices",
        "price_history",
        "up_price",
        "polymarket_prior",
    ):
        assert forbidden not in source


def test_feature_rows_use_vectorized_numpy_insert() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE SCHEMA feature")
    connection.execute(
        """
        CREATE TABLE feature.causal_row (
            condition_id VARCHAR,
            role VARCHAR,
            event_start_ms BIGINT,
            decision_time_ms BIGINT,
            decision_offset_seconds INTEGER,
            feature_values FLOAT[],
            feature_vector_sha256 VARCHAR,
            row_sha256 VARCHAR
        )
        """
    )
    rows = tuple(
        HistoricalFeatureRow(
            condition_id=f"condition-{index}",
            role="train",
            event_start_ms=index * 1_000,
            decision_time_ms=index * 1_000 + 30_000,
            decision_offset_seconds=30,
            feature_values=np.full(
                len(FEATURE_NAMES),
                float(index),
                dtype=np.float32,
            ),
            feature_vector_sha256=f"{index:064x}",
            row_sha256=f"{index + 1:064x}",
        )
        for index in range(513)
    )
    events: list[tuple[str, dict[str, object]]] = []

    _insert_feature_rows(
        connection,
        rows,
        progress=lambda event, values: events.append((event, dict(values))),
    )

    assert connection.execute(
        "SELECT count(*), count(DISTINCT condition_id) FROM feature.causal_row"
    ).fetchone() == (513, 513)
    assert [values["rows_written"] for _, values in events] == [0, 513]
    connection.close()
