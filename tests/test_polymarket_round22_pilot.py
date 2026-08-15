from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import PolymarketFeeSchedule
from simple_ai_trading.polymarket_historical_l2 import (
    HistoricalBookLevel,
    HistoricalBookSnapshot,
    HistoricalL2Window,
)
from simple_ai_trading.polymarket_historical_screen import HistoricalBtcMarket
from simple_ai_trading.polymarket_round22_pilot import (
    Round22PilotContract,
    Round22PilotStore,
    development_conditions,
    load_round22_pilot_contract,
    validate_round22_market_identity,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONDITION_ID = "0x" + ("a" * 64)
SECOND_CONDITION_ID = "0x" + ("b" * 64)
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40


def _market(
    contract: Round22PilotContract,
    *,
    condition_index: int = 0,
    condition_id: str = CONDITION_ID,
    up_token_id: str = UP_TOKEN_ID,
    down_token_id: str = DOWN_TOKEN_ID,
) -> HistoricalBtcMarket:
    expected = development_conditions(contract)[condition_index]
    event_id = str(10_000 + condition_index)
    market_id = str(20_000 + condition_index)
    identity_json = json.dumps(
        {
            "event_id": event_id,
            "market": {
                "conditionId": condition_id,
                "id": market_id,
                "question": "Will Bitcoin be up or down?",
                "slug": expected.slug,
            },
            "role": expected.role,
            "schema_version": "test-target-free-identity-v1",
            "series_id": "10684",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return HistoricalBtcMarket(
        event_id=event_id,
        market_id=market_id,
        condition_id=condition_id,
        slug=expected.slug,
        question="Will Bitcoin be up or down?",
        event_start_ms=expected.event_start_ms,
        end_ms=expected.event_end_ms,
        role=expected.role,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.25"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        excluded=False,
        exclusion_reason="",
        identity_payload_json=identity_json,
        identity_payload_sha256=hashlib.sha256(
            identity_json.encode("ascii")
        ).hexdigest(),
        source_payload_sha256="c" * 64,
        observed_at_ms=expected.event_end_ms + 600_000,
    )


def _window(market: HistoricalBtcMarket, *, outcome: str) -> HistoricalL2Window:
    token = market.up_token_id if outcome == "Up" else market.down_token_id
    snapshots = tuple(
        HistoricalBookSnapshot(
            condition_id=market.condition_id,
            asset_id=token,
            timestamp_ms=market.event_start_ms + offset,
            book_hash="d" * 40,
            bids=(HistoricalBookLevel(price="0.4", size="10"),),
            asks=(HistoricalBookLevel(price="0.6", size="10"),),
            minimum_order_size="5",
            tick_size="0.01",
            negative_risk=False,
            last_trade_price="0.5",
            source_payload_sha256="e" * 64,
        )
        for offset in (100, 300)
    )
    return HistoricalL2Window(
        condition_id=market.condition_id,
        asset_id=token,
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        snapshots=snapshots,
        source_chain_sha256="f" * 64,
    )


def test_round22_contract_has_exact_partition_and_target_free_parser_scope() -> None:
    contract = load_round22_pilot_contract(REPOSITORY)

    assert len(contract.conditions) == 576
    assert len(development_conditions(contract)) == 480
    assert sum(item.role == "sealed_test" for item in contract.conditions) == 96
    assert not {item.slug for item in contract.conditions} & contract.excluded_slugs
    parser_contract = contract.identity_parser_contract()
    assert parser_contract.series_id == "10684"
    assert parser_contract.required_flow_rows_per_day == 0
    assert parser_contract.required_source_symbol_count == 0


def test_round22_store_commits_atomically_audits_and_is_idempotent(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    market = _market(contract)
    database = tmp_path / "round22.duckdb"

    with Round22PilotStore(database, contract=contract) as store:
        assert store.put_condition(
            market=market,
            up_window=_window(market, outcome="Up"),
            down_window=_window(market, outcome="Down"),
        )
        assert not store.put_condition(
            market=market,
            up_window=_window(market, outcome="Up"),
            down_window=_window(market, outcome="Down"),
        )
        assert store.completed_slugs() == {market.slug}
        assert store.feature_counts() == {"train": 1}
        assert store.target_row_count() == 0
        assert store.market(market.condition_id) == market
        up_window, down_window = store.condition_windows(market.condition_id)
        assert up_window == _window(market, outcome="Up")
        assert down_window == _window(market, outcome="Down")
        assert store.audit_condition(market.condition_id) == {
            "condition_id": market.condition_id,
            "down_record_count": 2,
            "manifest_sha256": store.connection.execute(
                "SELECT manifest_sha256 FROM feature.condition_manifest"
            ).fetchone()[0],
            "role": "train",
            "slug": market.slug,
            "up_record_count": 2,
        }

    with Round22PilotStore(database, contract=contract, read_only=True) as reopened:
        assert reopened.completed_slugs() == {market.slug}
        assert reopened.target_row_count() == 0
        assert reopened.market(market.condition_id) == market


def test_round22_store_rolls_back_cross_condition_token_collision(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    first = _market(contract)
    collision = _market(
        contract,
        condition_index=1,
        condition_id=SECOND_CONDITION_ID,
    )

    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as store:
        assert store.put_condition(
            market=first,
            up_window=_window(first, outcome="Up"),
            down_window=_window(first, outcome="Down"),
        )
        with pytest.raises(
            Exception, match="constraint|duplicate", check=lambda _: True
        ):
            store.put_condition(
                market=collision,
                up_window=_window(collision, outcome="Up"),
                down_window=_window(collision, outcome="Down"),
            )
        assert store.completed_slugs() == {first.slug}
        assert store.feature_counts() == {"train": 1}


def test_round22_store_rejects_tamper_and_changed_idempotent_evidence(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    market = _market(contract)
    database = tmp_path / "round22.duckdb"

    with Round22PilotStore(database, contract=contract) as store:
        up = _window(market, outcome="Up")
        down = _window(market, outcome="Down")
        assert store.put_condition(market=market, up_window=up, down_window=down)
        with pytest.raises(ValueError, match="existing condition manifest differs"):
            store.put_condition(
                market=market,
                up_window=replace(up, source_chain_sha256="0" * 64),
                down_window=down,
            )
        payload = bytearray(
            store.connection.execute(
                "SELECT payload FROM feature.book_chunk WHERE outcome = 'Up'"
            ).fetchone()[0]
        )
        payload[-1] ^= 1
        store.connection.execute(
            "UPDATE feature.book_chunk SET payload = ? WHERE outcome = 'Up'",
            [bytes(payload)],
        )
        with pytest.raises(ValueError, match="envelope differs"):
            store.audit_condition(market.condition_id)


def test_round22_store_blocks_sealed_or_malformed_feature_evidence(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    sealed_expected = next(
        item for item in contract.conditions if item.role == "sealed_test"
    )
    base = _market(contract)
    sealed = replace(
        base,
        slug=sealed_expected.slug,
        role=sealed_expected.role,
        event_start_ms=sealed_expected.event_start_ms,
        end_ms=sealed_expected.event_end_ms,
    )
    with pytest.raises(ValueError, match="sealed-test identity access is blocked"):
        validate_round22_market_identity(sealed, contract=contract)

    malformed = replace(base, identity_payload_sha256="0" * 64)
    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as store:
        with pytest.raises(ValueError, match="identity evidence differs"):
            store.put_condition(
                market=malformed,
                up_window=_window(malformed, outcome="Up"),
                down_window=_window(malformed, outcome="Down"),
            )
