from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import simple_ai_trading.polymarket_round16_targets as round16_targets
from simple_ai_trading.polymarket_round16_dataset import (
    ROUND16_FEATURE_NAMES,
    _repair_round16_leading_close_gaps,
    build_round16_feature_row,
)
from simple_ai_trading.polymarket_round16 import (
    ROUND16_DECISION_OFFSETS_SECONDS,
    ROUND16_MARKETS_PER_DAY,
    Round16HistoricalPublicClient,
    collect_round16_market_identities,
    load_round16_historical_contract,
    parse_round16_historical_btc_event,
)
from simple_ai_trading.polymarket_historical_screen import (
    HistoricalScreenStore,
    PublicPayload,
)
from simple_ai_trading.polymarket_round16_targets import (
    _record_resolution,
    collect_round16_development_targets,
    record_round16_target_implementation,
    verify_round16_target_implementation,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v2.json"
)
START_MS = int(datetime(2026, 3, 20, 12, 0, tzinfo=UTC).timestamp() * 1_000)
END_MS = START_MS + 900_000
SLUG = f"btc-updown-15m-{START_MS // 1_000}"
CONDITION = "0x" + "1" * 64
UP_TOKEN = "2" * 40
DOWN_TOKEN = "3" * 40


def _event() -> dict[str, object]:
    market = {
        "id": "1234567",
        "conditionId": CONDITION,
        "slug": SLUG,
        "question": "Bitcoin Up or Down - 15 minute test",
        "eventStartTime": "2026-03-20T12:00:00Z",
        "endDate": "2026-03-20T12:15:00Z",
        "active": False,
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
        "winner": "Up",
    }
    return {
        "id": "7654321",
        "ticker": SLUG,
        "slug": SLUG,
        "closed": True,
        "series": [{"id": "10192", "ticker": "btc-up-or-down-15m"}],
        "markets": [market],
        "resolution": "Up",
    }


def _event_at(start_ms: int, index: int) -> dict[str, object]:
    event = deepcopy(_event())
    slug = f"btc-updown-15m-{start_ms // 1_000}"
    end_ms = start_ms + 900_000
    market = event["markets"][0]
    assert isinstance(market, dict)
    event.update(
        {
            "id": str(7_000_000 + index),
            "ticker": slug,
            "slug": slug,
        }
    )
    market.update(
        {
            "id": str(2_000_000 + index),
            "conditionId": f"0x{index + 1:064x}",
            "slug": slug,
            "eventStartTime": (
                datetime.fromtimestamp(start_ms / 1_000, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "endDate": (
                datetime.fromtimestamp(end_ms / 1_000, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "clobTokenIds": json.dumps(
                [
                    str(10**39 + index * 2),
                    str(10**39 + index * 2 + 1),
                ]
            ),
        }
    )
    return event


def _flow() -> dict[str, np.ndarray]:
    day_start_ms = START_MS // 86_400_000 * 86_400_000
    length = 45_000
    index = np.arange(length, dtype=np.float64)
    output: dict[str, np.ndarray] = {
        "second_ms": day_start_ms + np.arange(length, dtype=np.int64) * 1_000,
    }
    for prefix, base in (("spot", 100_000.0), ("perpetual", 100_010.0)):
        quote = np.full(length, 1_000.0, dtype=np.float64)
        output[f"{prefix}_close"] = base * np.exp(index * 1e-7)
        output[f"{prefix}_quote_volume"] = quote
        output[f"{prefix}_aggressive_buy_quote"] = quote * 0.55
        output[f"{prefix}_aggressive_sell_quote"] = quote * 0.45
        output[f"{prefix}_aggregate_count"] = np.full(length, 10.0)
        output[f"{prefix}_constituent_trade_count"] = np.full(length, 12.0)
        output[f"{prefix}_maximum_aggregate_quote"] = np.full(length, 200.0)
        output[f"{prefix}_squared_aggregate_quote_sum"] = np.full(
            length,
            100_000.0,
        )
        output[f"{prefix}_last_trade_age_seconds"] = np.zeros(length)
    return output


class _Response:
    def __init__(self, payload: object) -> None:
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, object, float]] = []

    def get(self, url: str, *, params: object, timeout: float) -> _Response:
        self.calls.append((url, params, timeout))
        return _Response(self.payload)


def _public(value: object, *, observed_at_ms: int = END_MS + 1) -> PublicPayload:
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


def test_round16_contract_loader_is_exact_and_has_no_authority() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)

    assert contract.contract_sha256 == (
        "6037c9ef473bcc736dbc7c3e98db76b75170e69e23de9574373bad7ae3fcdb67"
    )
    assert contract.duration_ms == 900_000
    assert contract.historical.series_id == "10192"
    assert len(contract.historical.eligible_days) == 154
    assert contract.historical.required_market_count_per_day == ROUND16_MARKETS_PER_DAY
    assert (
        contract.historical.decision_offsets_seconds == ROUND16_DECISION_OFFSETS_SECONDS
    )
    assert contract.historical.excluded_slugs == frozenset()


def test_round16_identity_parser_never_serializes_terminal_targets() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)

    market = parse_round16_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1,
    )

    assert market.end_ms - market.event_start_ms == 900_000
    assert market.role == "train"
    assert market.token_ids == (UP_TOKEN, DOWN_TOKEN)
    assert "outcomePrices" not in market.identity_payload_json
    assert '"winner"' not in market.identity_payload_json
    assert '"resolution"' not in market.identity_payload_json


def test_round16_causal_boundary_carry_uses_only_prior_observations() -> None:
    sentinel = float(np.iinfo(np.uint32).max)
    values = {
        "spot_close": np.asarray([99.0, 100.0, 101.0]),
        "spot_last_trade_age_seconds": np.asarray([0.0, 0.0, 0.0]),
        "perpetual_close": np.asarray([0.0, 0.0, 102.0]),
        "perpetual_last_trade_age_seconds": np.asarray([sentinel, sentinel, 0.0]),
    }
    prior = {
        "spot_close": np.asarray([98.0]),
        "spot_last_trade_age_seconds": np.asarray([0.0]),
        "perpetual_close": np.asarray([101.5]),
        "perpetual_last_trade_age_seconds": np.asarray([3.0]),
    }

    carry = _repair_round16_leading_close_gaps(
        values,
        close_masks={
            "spot": np.asarray([False, False, False]),
            "perpetual": np.asarray([True, True, False]),
        },
        prior=prior,
    )

    assert carry == {"spot": 0, "perpetual": 2}
    assert values["perpetual_close"].tolist() == [101.5, 101.5, 102.0]
    assert values["perpetual_last_trade_age_seconds"].tolist() == [
        4.0,
        5.0,
        0.0,
    ]


@pytest.mark.parametrize(
    ("mask", "prior"),
    (
        (np.asarray([True, False, True]), {"provided": True}),
        (np.asarray([True, True, True]), {"provided": True}),
        (np.asarray([True, False, False]), None),
    ),
)
def test_round16_causal_boundary_carry_rejects_unbounded_or_noncausal_gaps(
    mask: np.ndarray,
    prior: Mapping[str, object] | None,
) -> None:
    sentinel = float(np.iinfo(np.uint32).max)
    values = {
        "spot_close": np.asarray([99.0, 100.0, 101.0]),
        "spot_last_trade_age_seconds": np.asarray([0.0, 0.0, 0.0]),
        "perpetual_close": np.asarray([0.0, 102.0, 103.0]),
        "perpetual_last_trade_age_seconds": np.asarray([sentinel, 0.0, 0.0]),
    }
    prior_values = (
        {
            "spot_close": np.asarray([98.0]),
            "spot_last_trade_age_seconds": np.asarray([0.0]),
            "perpetual_close": np.asarray([101.5]),
            "perpetual_last_trade_age_seconds": np.asarray([0.0]),
        }
        if prior is not None
        else None
    )

    with pytest.raises(ValueError, match="causal carry policy"):
        _repair_round16_leading_close_gaps(
            values,
            close_masks={
                "spot": np.asarray([False, False, False]),
                "perpetual": mask,
            },
            prior=prior_values,
        )


def test_round16_client_strips_targets_before_identity_storage() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    response = {
        "events": [_event()],
        "next_cursor": "",
        "$schema": "https://example.invalid/schema",
    }
    session = _Session(response)
    client = Round16HistoricalPublicClient(
        session=session,
        clock_ms=lambda: END_MS + 1,
    )

    page = client.events_page(
        contract=contract.historical,
        day="2026-03-20",
    )
    wire = page.canonical_json

    assert "outcomePrices" not in wire
    assert '"winner"' not in wire
    assert '"resolution"' not in wire
    assert '"ticker":"btc-up-or-down-15m"' not in wire
    assert page.sha256
    _url, params, _timeout = session.calls[0]
    assert ("series_id", "10192") in params
    assert ("end_date_max", "2026-03-21T00:15:00Z") in params


def test_round16_exact_slug_client_strips_terminal_targets() -> None:
    session = _Session(_event())
    client = Round16HistoricalPublicClient(
        session=session,
        clock_ms=lambda: END_MS + 1,
    )

    event = client.event_identity_by_slug(SLUG)

    assert "outcomePrices" not in event.canonical_json
    assert '"winner"' not in event.canonical_json
    assert '"resolution"' not in event.canonical_json
    url, params, _timeout = session.calls[0]
    assert url.endswith(f"/events/slug/{SLUG}")
    assert params is None


def test_round16_identity_collection_repairs_bounded_keyset_omission(
    tmp_path: Path,
) -> None:
    loaded = load_round16_historical_contract(CONTRACT_PATH)
    day = "2026-03-20"
    day_start_ms = int(datetime(2026, 3, 20, tzinfo=UTC).timestamp() * 1_000)
    historical = replace(
        loaded.historical,
        eligible_days=(day,),
        roles={day: "train"},
    )
    contract = replace(loaded, historical=historical)
    events = tuple(
        _event_at(day_start_ms + index * 900_000, index)
        for index in range(ROUND16_MARKETS_PER_DAY)
    )
    missing_index = 81

    class IdentityClient:
        def __init__(self) -> None:
            self.exact_slugs: list[str] = []

        def events_page(
            self,
            *,
            contract: object,
            day: str,
            after_cursor: str = "",
        ) -> PublicPayload:
            assert not after_cursor
            return _public(
                {
                    "events": [
                        event
                        for index, event in enumerate(events)
                        if index != missing_index
                    ],
                    "next_cursor": "",
                },
                observed_at_ms=day_start_ms + 86_400_000 + 1,
            )

        def event_identity_by_slug(self, slug: str) -> PublicPayload:
            self.exact_slugs.append(slug)
            return _public(
                events[missing_index],
                observed_at_ms=day_start_ms + 86_400_000 + 1,
            )

    client = IdentityClient()
    with HistoricalScreenStore(
        tmp_path / "identity-repair.duckdb",
        contract=historical,
    ) as store:
        counts = collect_round16_market_identities(
            store,
            contract,
            client,  # type: ignore[arg-type]
        )
        identities = store.markets(include_excluded=True)

        assert counts == {day: ROUND16_MARKETS_PER_DAY}
        assert store.state == "identities_complete"
        assert len(identities) == ROUND16_MARKETS_PER_DAY
        assert client.exact_slugs == [
            f"btc-updown-15m-{(day_start_ms + missing_index * 900_000) // 1_000}"
        ]
        assert all(
            "outcomePrices" not in row.identity_payload_json for row in identities
        )
        assert all('"winner"' not in row.identity_payload_json for row in identities)


def test_round16_gamma_target_batch_repairs_exact_coverage() -> None:
    market = _event()["markets"][0]
    assert isinstance(market, dict)
    session = _Session({"markets": [market], "next_cursor": ""})
    client = Round16HistoricalPublicClient(
        session=session,
        clock_ms=lambda: END_MS + 1,
    )

    batch = client.gamma_markets(("1234567",))

    assert tuple(batch) == ("1234567",)
    assert batch["1234567"].value == market
    url, params, _timeout = session.calls[0]
    assert url.endswith("/markets/keyset")
    assert ("id", "1234567") in params
    assert ("closed", "true") in params
    assert ("limit", "1") in params

    class MissingThenExactSession:
        def __init__(self, exact: object) -> None:
            self.exact = exact
            self.calls: list[tuple[str, object, float]] = []

        def get(self, url: str, *, params: object, timeout: float) -> _Response:
            self.calls.append((url, params, timeout))
            if url.endswith("/markets/keyset"):
                return _Response({"markets": [], "next_cursor": ""})
            return _Response(self.exact)

    fallback_session = MissingThenExactSession(market)
    missing = Round16HistoricalPublicClient(
        session=fallback_session,  # type: ignore[arg-type]
    )
    repaired = missing.gamma_markets(("1234567",))

    assert repaired["1234567"].value == market
    assert len(fallback_session.calls) == 2
    assert fallback_session.calls[1][0].endswith("/markets/1234567")
    assert fallback_session.calls[1][1] is None

    mismatch = MissingThenExactSession({**market, "id": "7654321"})
    invalid = Round16HistoricalPublicClient(
        session=mismatch,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="exact Gamma market identity differs"):
        invalid.gamma_markets(("1234567",))


def test_round16_target_implementation_manifest_fails_closed_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    with HistoricalScreenStore(
        tmp_path / "target-manifest.duckdb",
        contract=contract.historical,
    ) as store:
        with pytest.raises(ValueError, match="manifest is missing"):
            verify_round16_target_implementation(store)
        assert (
            store.connect()
            .execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'target'
                  AND table_name = 'round16_resolution_manifest'
                """
            )
            .fetchone()[0]
            == 0
        )
        recorded = record_round16_target_implementation(store)
        assert verify_round16_target_implementation(store) == recorded
        changed = dict(recorded)
        changed["manifest_sha256"] = "f" * 64
        monkeypatch.setattr(
            round16_targets,
            "round16_target_implementation_manifest",
            lambda: changed,
        )
        with pytest.raises(ValueError, match="changed after label access"):
            verify_round16_target_implementation(store)


def test_round16_target_collection_batches_gamma_and_crosschecks_clob(
    tmp_path: Path,
) -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    market = parse_round16_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1,
    )
    gamma_value = _event()["markets"][0]
    assert isinstance(gamma_value, dict)
    clob_value = {
        "condition_id": CONDITION,
        "market_slug": SLUG,
        "closed": True,
        "active": False,
        "accepting_orders": False,
        "tokens": [
            {
                "token_id": UP_TOKEN,
                "outcome": "Up",
                "winner": True,
                "price": 1,
            },
            {
                "token_id": DOWN_TOKEN,
                "outcome": "Down",
                "winner": False,
                "price": 0,
            },
        ],
    }

    class TargetClient:
        def __init__(self) -> None:
            self.gamma_calls: list[tuple[str, ...]] = []
            self.clob_calls: list[str] = []

        def gamma_markets(
            self,
            market_ids: tuple[str, ...],
        ) -> Mapping[str, PublicPayload]:
            self.gamma_calls.append(market_ids)
            return {market.market_id: _public(gamma_value)}

        def clob_market(self, condition_id: str) -> PublicPayload:
            self.clob_calls.append(condition_id)
            return _public(clob_value)

    client = TargetClient()
    with HistoricalScreenStore(
        tmp_path / "target-collection.duckdb",
        contract=contract.historical,
    ) as store:
        store.upsert_market(market)
        store.transition("initialized", "identities_complete")
        store.transition("identities_complete", "features_complete")

        counts = collect_round16_development_targets(
            store,
            contract,
            client,  # type: ignore[arg-type]
        )

        assert counts == {"Up": 1, "Down": 0}
        assert store.state == "development_targets_complete"
        assert client.gamma_calls == [("1234567",)]
        assert client.clob_calls == [CONDITION]
        assert (
            store.connect()
            .execute("SELECT count(*) FROM target.round16_resolution_manifest")
            .fetchone()[0]
            == 1
        )


def test_round16_parser_rejects_horizon_drift() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    event = deepcopy(_event())
    market = event["markets"][0]
    assert isinstance(market, dict)
    market["endDate"] = "2026-03-20T12:05:00Z"

    with pytest.raises(ValueError, match="window differs"):
        parse_round16_historical_btc_event(
            event,
            contract=contract,
            observed_at_ms=END_MS + 1,
        )


def test_round16_feature_row_is_causal_and_finite() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    market = parse_round16_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1,
    )
    flow = _flow()
    day_start_ms = START_MS // 86_400_000 * 86_400_000
    first = build_round16_feature_row(
        market,
        flow_start_ms=day_start_ms,
        decision_offset_seconds=60,
        flow=flow,
    )
    mutated = deepcopy(flow)
    first_unavailable = (START_MS - day_start_ms) // 1_000 + 60
    mutated["spot_close"][first_unavailable:] *= 2
    mutated["spot_quote_volume"][first_unavailable:] *= 10
    second = build_round16_feature_row(
        market,
        flow_start_ms=day_start_ms,
        decision_offset_seconds=60,
        flow=mutated,
    )

    assert first.feature_values.shape == (len(ROUND16_FEATURE_NAMES),)
    assert np.all(np.isfinite(first.feature_values))
    np.testing.assert_array_equal(first.feature_values, second.feature_values)
    assert first.row_sha256 == second.row_sha256
    spot_moneyness = ROUND16_FEATURE_NAMES.index("spot_event_to_date_log_moneyness")
    spot_scaled_moneyness = ROUND16_FEATURE_NAMES.index(
        "spot_volatility_scaled_digital_moneyness"
    )
    assert first.feature_values[spot_moneyness] == pytest.approx(60e-7)
    assert first.feature_values[spot_scaled_moneyness] > 0


def test_round16_terminal_controls_use_only_observed_flow() -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    market = parse_round16_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1,
    )
    flow = _flow()
    day_start_ms = START_MS // 86_400_000 * 86_400_000
    baseline = build_round16_feature_row(
        market,
        flow_start_ms=day_start_ms,
        decision_offset_seconds=180,
        flow=flow,
    )
    changed = deepcopy(flow)
    last_completed = (START_MS - day_start_ms) // 1_000 + 179
    changed["spot_quote_volume"][last_completed - 29 : last_completed + 1] *= 4
    changed["spot_aggressive_buy_quote"][last_completed - 29 : last_completed + 1] *= 4
    changed["spot_aggressive_sell_quote"][last_completed - 29 : last_completed + 1] *= 4
    shifted = build_round16_feature_row(
        market,
        flow_start_ms=day_start_ms,
        decision_offset_seconds=180,
        flow=changed,
    )

    assert baseline.row_sha256 != shifted.row_sha256
    terminal_index = ROUND16_FEATURE_NAMES.index(
        "terminal_spot_log_quote_rate_ratio_30s_to_prior_120s"
    )
    assert (
        shifted.feature_values[terminal_index] > baseline.feature_values[terminal_index]
    )


def test_round16_dataset_source_has_no_polymarket_target_inputs() -> None:
    source = (
        ROOT / "src" / "simple_ai_trading" / "polymarket_round16_dataset.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "outcomePrices",
        "winning_outcome",
        "official_resolution",
        "up_price",
        "polymarket_prior",
    ):
        assert forbidden not in source


def test_round16_resolution_is_unavailable_before_feature_freeze(
    tmp_path: Path,
) -> None:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    market = parse_round16_historical_btc_event(
        _event(),
        contract=contract,
        observed_at_ms=END_MS + 1,
    )
    gamma_value = _event()["markets"][0]
    assert isinstance(gamma_value, dict)
    gamma = _public(gamma_value)
    clob = _public(
        {
            "condition_id": CONDITION,
            "market_slug": SLUG,
            "closed": True,
            "active": False,
            "accepting_orders": False,
            "tokens": [
                {
                    "token_id": UP_TOKEN,
                    "outcome": "Up",
                    "winner": True,
                    "price": 1,
                },
                {
                    "token_id": DOWN_TOKEN,
                    "outcome": "Down",
                    "winner": False,
                    "price": 0,
                },
            ],
        }
    )
    with HistoricalScreenStore(
        tmp_path / "round16.duckdb",
        contract=contract.historical,
    ) as store:
        with pytest.raises(ValueError, match="not authorized"):
            _record_resolution(
                store,
                contract,
                market,
                gamma=gamma,
                clob=clob,
            )
        store.transition("initialized", "identities_complete")
        store.transition("identities_complete", "features_complete")

        assert (
            _record_resolution(
                store,
                contract,
                market,
                gamma=gamma,
                clob=clob,
            )
            == "Up"
        )
        with pytest.raises(ValueError, match="not authorized"):
            _record_resolution(
                store,
                contract,
                replace(market, role="test"),
                gamma=gamma,
                clob=clob,
            )
