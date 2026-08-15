from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import PolymarketFeeSchedule
from simple_ai_trading.polymarket_historical_l2 import (
    HistoricalBookLevel,
    HistoricalBookSnapshot,
    HistoricalL2Window,
)
from simple_ai_trading.polymarket_historical_screen import HistoricalBtcMarket
from simple_ai_trading.polymarket_round22_feature_store import Round22FeatureStore
from simple_ai_trading.polymarket_round22_feature_operator import (
    materialize_round22_development_features,
)
from simple_ai_trading.polymarket_round22_features import (
    build_round22_condition_features,
)
from simple_ai_trading.polymarket_round22_pilot import (
    Round22PilotContract,
    Round22PilotStore,
    development_conditions,
    load_round22_pilot_contract,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONDITION_ID = "0x" + ("a" * 64)
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40


def _market(contract: Round22PilotContract) -> HistoricalBtcMarket:
    expected = development_conditions(contract)[0]
    identity_json = '{"schema_version":"feature-store-test-identity-v1"}'
    return HistoricalBtcMarket(
        event_id="10001",
        market_id="20001",
        condition_id=CONDITION_ID,
        slug=expected.slug,
        question="Will Bitcoin be up or down?",
        event_start_ms=expected.event_start_ms,
        end_ms=expected.event_end_ms,
        role=expected.role,
        up_token_id=UP_TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
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
            bids=(HistoricalBookLevel(price="0.4", size="100"),),
            asks=(HistoricalBookLevel(price="0.6", size="100"),),
            minimum_order_size="5",
            tick_size="0.01",
            negative_risk=False,
            last_trade_price="0.5",
            source_payload_sha256=hashlib.sha256(
                f"{outcome}:{offset}".encode("ascii")
            ).hexdigest(),
        )
        for offset in (100, 250)
    )
    return HistoricalL2Window(
        condition_id=market.condition_id,
        asset_id=token,
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        snapshots=snapshots,
        source_chain_sha256=hashlib.sha256(outcome.encode("ascii")).hexdigest(),
    )


def test_round22_feature_store_commits_audits_resumes_and_detects_tamper(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    market = _market(contract)
    up = _window(market, outcome="Up")
    down = _window(market, outcome="Down")
    result = build_round22_condition_features(
        repository=REPOSITORY,
        up_window=up,
        down_window=down,
    )
    database = tmp_path / "round22.duckdb"

    with Round22PilotStore(database, contract=contract) as pilot:
        assert pilot.put_condition(market=market, up_window=up, down_window=down)
        features = Round22FeatureStore(pilot)
        assert features.put_condition(result)
        assert not features.put_condition(result)
        assert features.completed_condition_ids() == {market.condition_id}
        audit = features.audit_condition(market.condition_id)
        assert audit["row_count"] == 1_199
        assert audit["available_count"] == 5
        assert audit["sequence_complete_count"] == 0
        assert audit["tabular_history_complete_count"] == 0
        assert audit["target_row_count"] == 0
        assert audit["compressed_size_bytes"] < audit["raw_size_bytes"]

    with Round22PilotStore(database, contract=contract, read_only=True) as pilot:
        features = Round22FeatureStore(pilot)
        assert features.audit_condition(market.condition_id) == audit

    with Round22PilotStore(database, contract=contract) as pilot:
        features = Round22FeatureStore(pilot)
        pilot.connection.execute(
            """
            UPDATE feature.condition_feature_chunk SET compressed_sha256 = ?
            WHERE condition_id = ?
            """,
            ["0" * 64, market.condition_id],
        )
        with pytest.raises(ValueError, match="chunk envelope differs"):
            features.audit_condition(market.condition_id)


def test_round22_feature_store_refuses_schema_drift(tmp_path: Path) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    database = tmp_path / "round22.duckdb"
    with Round22PilotStore(database, contract=contract) as pilot:
        Round22FeatureStore(pilot)
        pilot.connection.execute(
            "UPDATE feature.causal_feature_schema SET feature_names_sha256 = ?",
            ["0" * 64],
        )
        with pytest.raises(ValueError, match="schema differs"):
            Round22FeatureStore(pilot)


def test_round22_feature_operator_materializes_once_with_progress(
    tmp_path: Path,
) -> None:
    contract = load_round22_pilot_contract(REPOSITORY)
    market = _market(contract)
    up = _window(market, outcome="Up")
    down = _window(market, outcome="Down")
    progress: list[str] = []
    with Round22PilotStore(tmp_path / "round22.duckdb", contract=contract) as pilot:
        assert pilot.put_condition(market=market, up_window=up, down_window=down)
        features = Round22FeatureStore(pilot)
        first = materialize_round22_development_features(
            pilot_store=pilot,
            feature_store=features,
            progress=lambda phase, _: progress.append(phase),
        )
        second = materialize_round22_development_features(
            pilot_store=pilot,
            feature_store=features,
        )

        assert first.committed_condition_ids == (market.condition_id,)
        assert first.selection_role == "all_development"
        assert first.source_condition_count == 1
        assert first.committed_count == 1
        assert first.remaining_materializable_condition_count == 0
        assert second.committed_count == 0
        assert second.already_complete_count == 1
        assert progress == [
            "feature_source_audit",
            "feature_grid_built",
            "feature_condition_committed",
        ]
        assert not any(
            (first.target_accessed, first.binance_used, first.trading_authority)
        )
