from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)


_START_MS = 1_786_784_400_000
_OPENED_MS = _START_MS + 400_000
_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    gamma = {
        "id": "12345",
        "conditionId": "0x" + "7" * 64,
        "slug": "btc-updown-5m-1786784400",
        "outcomes": ["Up", "Down"],
        "clobTokenIds": ["7" * 40, "8" * 40],
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
    }
    canonical = _canonical(gamma)
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id="0x" + "7" * 64,
        slug="btc-updown-5m-1786784400",
        question="Bitcoin Up or Down",
        event_start_ms=_START_MS,
        end_ms=_START_MS + 300_000,
        up_token_id="7" * 40,
        down_token_id="8" * 40,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.2"),
        ),
        liquidity_quote=Decimal("10000"),
        volume_quote=Decimal("20000"),
        resolution_source="https://data.chain.link/streams/btc-usd",
        gamma_payload_sha256=_sha(canonical),
        gamma_payload_json=canonical,
    )


def _row() -> Round27FeatureRow:
    return Round27FeatureRow.create(
        run_id="stage1-a-run",
        condition_id=_market().condition_id,
        event_start_ms=_START_MS,
        decision_time_ms=_START_MS + 30_000,
        market_prior_probability=0.5,
        values=[0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES),
        maximum_receipt_wall_ms=_START_MS + 30_000,
        source_chain_sha256="d" * 64,
    )


def _envelope(
    value: dict[str, object],
    observed_ms: int,
) -> Round25OfficialPublicPayload:
    canonical = _canonical(value)
    return Round25OfficialPublicPayload(
        value=value,
        canonical_json=canonical,
        sha256=_sha(canonical),
        observed_wall_ms=observed_ms,
        observed_monotonic_ns=observed_ms * 1_000_000,
    )


def _official() -> tuple[dict[str, object], dict[str, object]]:
    market = _market()
    gamma = {
        "id": market.market_id,
        "conditionId": market.condition_id,
        "slug": market.slug,
        "outcomes": ["Up", "Down"],
        "clobTokenIds": list(market.token_ids),
        "resolutionSource": market.resolution_source,
        "outcomePrices": ["1", "0"],
        "closed": True,
        "acceptingOrders": False,
    }
    clob = {
        "condition_id": market.condition_id,
        "market_slug": market.slug,
        "closed": True,
        "accepting_orders": False,
        "tokens": [
            {
                "outcome": outcome,
                "price": "1" if outcome == "Up" else "0",
                "token_id": token_id,
                "winner": outcome == "Up",
            }
            for token_id, outcome in zip(
                market.token_ids,
                ("Up", "Down"),
                strict=True,
            )
        ],
    }
    return clob, gamma


class _Client:
    def clob_market(self, _condition_id: str) -> Round25OfficialPublicPayload:
        clob, _gamma = _official()
        return _envelope(clob, _OPENED_MS + 1)

    def gamma_market(self, _market_id: str) -> Round25OfficialPublicPayload:
        _clob, gamma = _official()
        return _envelope(gamma, _OPENED_MS + 2)


def _open(store: Round27TargetStore, *, role: str = "train") -> bool:
    return store.open_role(
        role=role,
        slot_id="stage1-a",
        run_id="stage1-a-run",
        contract=load_round27_model_contract(_ROOT),
        feature_store_audit_sha256="b" * 64,
        role_intervals=(
            {
                "role": role,
                "slot_id": "stage1-a",
                "start_ms": _START_MS,
                "end_ms": _START_MS + 300_000,
            },
        ),
        feature_rows=(_row(),),
        markets=((_market(), "c" * 64),),
        opened_at_ms=_OPENED_MS,
    )


def test_target_store_is_role_gated_separate_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "targets.duckdb"
    with Round27TargetStore(path) as store:
        assert _open(store) is True
        assert _open(store) is False
        initial = store.audit()
        assert initial["feature_targets_co_located"] is False
        assert initial["resolved_condition_count"] == 0

        collected = store.collect_once(role="train", client=_Client())
        assert collected["newly_resolved_condition_count"] == 1
        final = store.finalize_role("train")
        assert final["finalized"] is True
        assert store.outcomes_up(roles=("train",)) == {_market().condition_id: 1}

    with Round27TargetStore(path, read_only=True) as reopened:
        audit = reopened.audit()
        assert audit["condition_count"] == 1
        assert audit["resolved_condition_count"] == 1
        assert reopened.outcomes_up(roles=("train",)) == {
            _market().condition_id: 1
        }


def test_target_store_rejects_sealed_access_without_exact_artifacts(
    tmp_path: Path,
) -> None:
    with Round27TargetStore(tmp_path / "targets.duckdb") as store:
        with pytest.raises(ValueError, match="sealed target artifacts are required"):
            _open(store, role="sealed")
