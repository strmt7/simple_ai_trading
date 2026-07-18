from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round13_capture as round13_capture
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


def _official_payloads(
    asset: str, *, winner: str = "Up"
) -> tuple[dict[str, object], dict[str, object]]:
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


def _round13_manifest(run_id: str, contract_sha256: str) -> dict[str, object]:
    contract_path = "docs/model-research/polymarket/round-013-contract.json"
    predecessor_path = "docs/model-research/polymarket/round-011-artifact.json"
    required = {path: "e" * 64 for path in round13_capture._REQUIRED_REPOSITORY_FILES}
    required[contract_path] = "e" * 64
    required[predecessor_path] = "e" * 64
    return round13_capture.create_round13_capture_manifest(
        run_id=run_id,
        started_at_ms=EPOCH * 1_000,
        capture_duration_seconds=(
            round13_capture.POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS
        ),
        repository_commit="1" * 40,
        repository_tree="2" * 40,
        contract_repository_path=contract_path,
        predecessor_repository_path=predecessor_path,
        contract_sha256=contract_sha256,
        model_sha256="3" * 64,
        policy_sha256="4" * 64,
        reference_implementation_sha256="5" * 64,
        action_pipeline_implementation_sha256="6" * 64,
        round13_program_implementation_sha256="7" * 64,
        required_file_sha256=required,
    )


def _complete_store(
    path: Path,
    run_id: str = "resolution-run",
    *,
    round13_contract_sha256: str = "",
) -> PolymarketEvidenceStore:
    store = PolymarketEvidenceStore(path)
    store.connect()
    manifest = (
        None
        if not round13_contract_sha256
        else _round13_manifest(run_id, round13_contract_sha256)
    )
    store.start_run(
        run_id,
        EPOCH * 1_000,
        preregistration_manifest=manifest,
    )
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
                "tick_size": "0.01",
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


def _insert_round13_open_claim(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    contract_sha256: str,
) -> None:
    store.connect().execute(
        """
        CREATE TABLE polymarket_round13_evaluation_claim (
            contract_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            claim_sha256 VARCHAR NOT NULL UNIQUE,
            run_id VARCHAR NOT NULL,
            pipeline_report_sha256 VARCHAR NOT NULL,
            scenario_dataset_sha256_json VARCHAR NOT NULL,
            opened_at_ms BIGINT NOT NULL,
            status VARCHAR NOT NULL,
            report_sha256 VARCHAR NOT NULL,
            error VARCHAR NOT NULL
        )
        """
    )
    pipeline_sha256 = "8" * 64
    scenario_ids = ["9" * 64]
    opened_at_ms = EPOCH * 1_000 + 400_000
    identity = {
        "schema_version": "polymarket-round13-one-use-claim-v1",
        "contract_sha256": contract_sha256,
        "run_id": run_id,
        "pipeline_report_sha256": pipeline_sha256,
        "scenario_dataset_sha256": scenario_ids,
        "opened_at_ms": opened_at_ms,
        "state": "opened_before_resolution_query",
        "preexisting_resolution_count": 0,
    }
    store.connect().execute(
        "INSERT INTO polymarket_round13_evaluation_claim VALUES "
        "(?, ?, ?, ?, ?, ?, ?, 'opened', '', '')",
        [
            contract_sha256,
            identity["schema_version"],
            _sha(_canonical(identity)),
            run_id,
            pipeline_sha256,
            _canonical(scenario_ids),
            opened_at_ms,
        ],
    )


def test_resolution_waits_for_both_sources_and_rejects_disagreement() -> None:
    market = parse_polymarket_five_minute_market(_market_payload("BTC"))
    clob, gamma = _official_payloads("BTC", winner="Down")
    gamma["closed"] = False
    assert (
        validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=market.end_ms + 1,
        )
        is None
    )

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


def test_round13_finalizer_requires_committed_open_evaluation_claim(
    tmp_path: Path,
) -> None:
    contract_sha256 = "a" * 64
    run_id = "round13-resolution-run"
    store = _complete_store(
        tmp_path / "round13-resolution.duckdb",
        run_id,
        round13_contract_sha256=contract_sha256,
    )
    client = _OfficialClient()
    finalizer = PolymarketResolutionFinalizer(
        store,
        client=client,  # type: ignore[arg-type]
        wall_clock_ms=lambda: EPOCH * 1_000 + 400_001,
        monotonic_clock_ns=lambda: 9_000_000_000,
    )
    try:
        with pytest.raises(ValueError, match="one-use evaluation claim"):
            finalizer.finalize(run_id=run_id)
        with pytest.raises(ValueError, match="one-use evaluation claim"):
            finalizer.finalize(
                run_id=run_id,
                round13_contract_sha256="b" * 64,
            )
        assert client.clob_calls == client.gamma_calls == []

        _insert_round13_open_claim(
            store,
            run_id=run_id,
            contract_sha256=contract_sha256,
        )
        report = finalizer.finalize(
            run_id=run_id,
            round13_contract_sha256=contract_sha256,
        )

        assert report.status == "complete"
        assert report.finalized_count == 3
    finally:
        store.close()


def test_finalizer_is_immutable_idempotent_and_detects_tampering(
    tmp_path: Path,
) -> None:
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
        assert {item.source for item in replay.resolutions} == {"clob_gamma_crosscheck"}

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
