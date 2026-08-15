from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from simple_ai_trading import polymarket_round27_resolution as resolution
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
)


RUN_ID = "round27-resolution-test"
CLAIMED_AT_MS = 1_700_000_400_000


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    gamma = {
        "id": "12345",
        "conditionId": "0x" + "7" * 64,
        "slug": "btc-updown-5m-1700000000",
        "outcomes": ["Up", "Down"],
        "clobTokenIds": ["7" * 40, "8" * 40],
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
    }
    canonical = _canonical(gamma)
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id="0x" + "7" * 64,
        slug="btc-updown-5m-1700000000",
        question="Bitcoin Up or Down",
        event_start_ms=1_700_000_000_000,
        end_ms=1_700_000_300_000,
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


def _stage0_inputs():
    market = _market()
    lineage = {
        "condition_audit_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "capture_contract_sha256": "3" * 64,
        "capture_result_sha256": "4" * 64,
        "run_id": RUN_ID,
        "cohort_role": "preregistered_stage0_mechanics",
        "preregistered_stage_0": True,
    }
    mechanics = {"mechanics_sha256": "5" * 64}
    return lineage, mechanics, ((market, "6" * 64),)


def _envelope(
    value: dict[str, object], observed_ms: int
) -> Round25OfficialPublicPayload:
    canonical = _canonical(value)
    return Round25OfficialPublicPayload(
        value=value,
        canonical_json=canonical,
        sha256=_sha(canonical),
        observed_wall_ms=observed_ms,
        observed_monotonic_ns=observed_ms * 1_000_000,
    )


def _official(*, winner: str = "Up") -> tuple[dict[str, object], dict[str, object]]:
    market = _market()
    index = 0 if winner == "Up" else 1
    prices = ["0", "0"]
    prices[index] = "1"
    gamma = {
        "id": market.market_id,
        "conditionId": market.condition_id,
        "slug": market.slug,
        "outcomes": ["Up", "Down"],
        "clobTokenIds": list(market.token_ids),
        "resolutionSource": market.resolution_source,
        "outcomePrices": prices,
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
                "price": "1" if position == index else "0",
                "token_id": token_id,
                "winner": position == index,
            }
            for position, (token_id, outcome) in enumerate(
                zip(market.token_ids, ("Up", "Down"), strict=True)
            )
        ],
    }
    return clob, gamma


class _Client:
    def __init__(self, *, disagree: bool = False) -> None:
        self.disagree = disagree

    def clob_market(self, _condition_id: str) -> Round25OfficialPublicPayload:
        clob, _ = _official(winner="Up")
        return _envelope(clob, CLAIMED_AT_MS + 1)

    def gamma_market(self, _market_id: str) -> Round25OfficialPublicPayload:
        _, gamma = _official(winner="Down" if self.disagree else "Up")
        return _envelope(gamma, CLAIMED_AT_MS + 2)


def _initialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(
        resolution, "_load_stage0_inputs", lambda **_kwargs: _stage0_inputs()
    )
    destination = tmp_path / "resolutions.duckdb"
    collection, claim = resolution.initialize_round27_resolution_collection(
        source_database=tmp_path / "source.duckdb",
        condition_audit_path=tmp_path / "audit.json",
        preregistration_path=tmp_path / "prereg.json",
        capture_contract_path=tmp_path / "contract.json",
        capture_result_path=tmp_path / "result.json",
        mechanics_path=tmp_path / "mechanics.json",
        destination_database=destination,
        created_at_ms=CLAIMED_AT_MS,
    )
    assert claim["target_access_opened"] is True
    assert claim["condition_count"] == 1
    assert claim["profitability_claim"] is False
    return collection, destination


def test_access_claim_is_persisted_before_any_target_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, _ = _initialize(tmp_path, monkeypatch)

    with duckdb.connect(str(collection), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM round27_resolution_access_claim"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM round27_resolution_evidence"
            ).fetchone()[0]
            == 0
        )


def test_dual_source_resolution_round_trips_through_deep_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, destination = _initialize(tmp_path, monkeypatch)
    monkeypatch.setattr(
        resolution,
        "_load_condition_markets",
        lambda *_args, **_kwargs: {_market().condition_id: _market()},
    )

    collection_report = resolution.collect_round27_resolutions_once(
        source_database=tmp_path / "source.duckdb",
        collection_database=collection,
        client=_Client(),
    )
    report = resolution.finalize_round27_resolution_collection(
        collection_database=collection,
        destination_database=destination,
        source_database=tmp_path / "source.duckdb",
    )

    assert collection_report["status"] == "complete"
    assert report["resolution_count"] == 1
    assert report["dual_source_agreement_count"] == 1
    assert report["winner_counts"] == {"Up": 1}
    assert report["edge_claim"] is False
    assert report == resolution.audit_round27_resolution_collection(
        destination,
        source_database=tmp_path / "source.duckdb",
    )


def test_source_disagreement_aborts_without_persisting_a_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, _ = _initialize(tmp_path, monkeypatch)
    monkeypatch.setattr(
        resolution,
        "_load_condition_markets",
        lambda *_args, **_kwargs: {_market().condition_id: _market()},
    )

    with pytest.raises(ValueError, match="disagree"):
        resolution.collect_round27_resolutions_once(
            source_database=tmp_path / "source.duckdb",
            collection_database=collection,
            client=_Client(disagree=True),
        )
    with duckdb.connect(str(collection), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM round27_resolution_evidence"
            ).fetchone()[0]
            == 0
        )


def test_tampered_compressed_payload_fails_the_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, _ = _initialize(tmp_path, monkeypatch)
    monkeypatch.setattr(
        resolution,
        "_load_condition_markets",
        lambda *_args, **_kwargs: {_market().condition_id: _market()},
    )
    resolution.collect_round27_resolutions_once(
        source_database=tmp_path / "source.duckdb",
        collection_database=collection,
        client=_Client(),
    )
    with duckdb.connect(str(collection)) as connection:
        payload = bytearray(
            connection.execute(
                "SELECT clob_payload FROM round27_resolution_evidence"
            ).fetchone()[0]
        )
        payload[-1] ^= 1
        connection.execute(
            "UPDATE round27_resolution_evidence SET clob_payload = ?",
            [bytes(payload)],
        )

    with pytest.raises(ValueError, match="CLOB resolution payload"):
        resolution.audit_round27_resolution_collection(
            collection,
            source_database=tmp_path / "source.duckdb",
        )
