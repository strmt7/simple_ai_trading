from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round25_campaign as round25
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_DESIGN_SHA256,
    POLYMARKET_ROUND25_END_MS,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    POLYMARKET_ROUND25_START_MS,
    PolymarketRound25CampaignConfig,
    create_round25_campaign_plan,
    load_round25_design,
    qualify_round25_source,
    run_round25_campaign,
    validate_round25_campaign_plan,
    write_round25_campaign_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict[str, object]:
    return create_round25_campaign_plan(
        created_at_ms=POLYMARKET_ROUND25_START_MS - 60_000,
        repository_commit_oid="a" * 40,
        repository_tree_oid="b" * 40,
        repository_file_sha256={name: "c" * 64 for name in round25._REQUIRED_FILES},
        source_qualification_sha256="d" * 64,
    )


def _market(epoch_ms: int, *, source: str = POLYMARKET_ROUND25_RESOLUTION_SOURCE):
    epoch = epoch_ms // 1_000
    start = datetime.fromtimestamp(epoch, tz=timezone.utc)
    end = datetime.fromtimestamp(epoch + 300, tz=timezone.utc)
    payload = {
        "id": str(epoch),
        "question": "Bitcoin Up or Down",
        "conditionId": "0x" + f"{epoch % 16:x}" * 64,
        "slug": f"btc-updown-5m-{epoch}",
        "eventStartTime": start.isoformat().replace("+00:00", "Z"),
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps(["7" * 40, "7" * 39 + "1"]),
        "outcomes": '["Up", "Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 2,
            "rate": 0.25,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000,
        "volumeNum": 50_000,
        "resolutionSource": source,
    }
    if source == POLYMARKET_ROUND25_RESOLUTION_SOURCE:
        payload["cryptoMarketConfigId"] = "btc-5m-twap-30"
        payload["cryptoMarketConfig"] = {
            "asset": "btc",
            "duration": "5m",
            "id": "btc-5m-twap-30",
            "twapEnabled": True,
            "twapLookbackSeconds": 30,
        }
    return parse_polymarket_five_minute_market(payload)


class _QualificationClient:
    def __init__(self, markets: tuple[object, ...]) -> None:
        self.markets = markets

    def discover_five_minute_markets(self, **_kwargs: object) -> tuple[object, ...]:
        return self.markets

    def protocol_version(self) -> int:
        return 2

    def clob_market_info(self, condition_id: str) -> dict[str, object]:
        market = next(item for item in self.markets if item.condition_id == condition_id)
        return {
            "c": market.condition_id,
            "t": [
                {"t": market.up_token_id, "o": "Up"},
                {"t": market.down_token_id, "o": "Down"},
            ],
            "mos": 5,
            "mts": 0.01,
            "mbf": 1000,
            "tbf": 1000,
            "itode": True,
            "fd": {"r": 0.25, "e": 2, "to": True},
        }


def test_design_and_plan_are_hash_bound_and_non_authoritative() -> None:
    design = load_round25_design(ROOT)
    plan = _plan()

    assert design["design_sha256"] == POLYMARKET_ROUND25_DESIGN_SHA256
    assert design["source_regime"]["legacy_and_twap_conditions_may_be_pooled"] is False
    assert design["source_regime"]["rtds_topic"] == "crypto_prices_twap_thirty"
    assert design["source_regime"]["exact_e18_value_required"] is True
    assert validate_round25_campaign_plan(plan).scheduled_end_ms == (
        POLYMARKET_ROUND25_END_MS
    )
    assert plan["model_data_eligible"] is False
    assert plan["live_trading_authority"] is False

    tampered = dict(plan)
    tampered["resolution_source"] = "https://data.chain.link/streams/btc-usd"
    with pytest.raises(ValueError, match="plan differs"):
        validate_round25_campaign_plan(tampered)


def test_campaign_recorder_uses_exact_twap_wire_mode(tmp_path) -> None:
    recorder = round25._create_recorder(tmp_path / "capture.duckdb")

    assert recorder.assets == ("BTC",)
    assert recorder.include_binance_spot is False
    assert recorder.include_rtds_binance is False
    assert recorder.chainlink_price_mode == "twap_30s"
    assert recorder.rtds_topics == ("crypto_prices_twap_thirty",)


def test_twap_wire_source_qualification_is_hash_bound() -> None:
    path = (
        ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-twap-wire-source-qualification-v2-2026-08-10.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("qualification_sha256")

    assert round25._canonical_sha256(payload) == claimed
    assert payload["status"] == "passed"
    assert payload["required_rtds_topic"] == "crypto_prices_twap_thirty"
    assert payload["twap_window_seconds"] == 30
    assert payload["exact_e18_value_observed"] is True
    assert payload["wire_probe"]["empty_control_frames"] == 1
    assert len(payload["wire_probe"]["observations"]) == 2
    assert all(
        observation["window_s"] == 30
        and observation["symbol"] == "btc/usd"
        and observation["topic"] == "crypto_prices_twap_thirty"
        for observation in payload["wire_probe"]["observations"]
    )
    assert payload["credentials_used"] is False
    assert payload["execution_connected"] is False
    assert payload["orders_submitted"] == 0
    assert payload["model_data_eligible"] is False
    assert payload["live_trading_authority"] is False


def test_source_qualification_binds_two_consecutive_twap_markets() -> None:
    observed = POLYMARKET_ROUND25_START_MS + 31_000
    epoch = observed // 300_000 * 300_000
    client = _QualificationClient((_market(epoch), _market(epoch + 300_000)))

    result = qualify_round25_source(client, observed_at_ms=observed)

    assert result["status"] == "passed"
    assert result["market_count"] == 2
    assert result["resolution_source"] == POLYMARKET_ROUND25_RESOLUTION_SOURCE
    assert result["credentials_used"] is False
    assert result["execution_connected"] is False

    drifted = _QualificationClient(
        (
            _market(epoch),
            _market(
                epoch + 300_000,
                source="https://data.chain.link/streams/btc-usd",
            ),
        )
    )
    with pytest.raises(ValueError, match="consecutive source markets"):
        qualify_round25_source(drifted, observed_at_ms=observed)


class _Recorder:
    def __init__(self, clock: list[int], *, source_failure: bool = False) -> None:
        self.clock = clock
        self.source_failure = source_failure
        self.calls = 0

    async def run(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        manifest = kwargs["preregistration_manifest_factory"](
            "run-25", self.clock[0]
        )
        kwargs["progress"](
            "capturing",
            {
                "run_id": "run-25",
                "written_message_count": 5,
                "queue_size": 0,
            },
        )
        if self.source_failure:
            return SimpleNamespace(
                status="failed",
                run_id="run-25",
                report_sha256="e" * 64,
                started_at_ms=self.clock[0],
                ended_at_ms=self.clock[0] + 1_000,
                duration_seconds=1.0,
                raw_message_count=0,
                stream_gap_count=0,
                stream_counts={},
                conditions=(),
                integrity_errors=(),
                errors=(
                    "discovery:market resolution source differs from the configured exact source",
                ),
            )
        self.clock[0] = POLYMARKET_ROUND25_END_MS
        return SimpleNamespace(
            status="complete",
            run_id="run-25",
            report_sha256="f" * 64,
            started_at_ms=int(manifest["created_at_ms"]),
            ended_at_ms=POLYMARKET_ROUND25_END_MS,
            duration_seconds=1.0,
            raw_message_count=5,
            stream_gap_count=0,
            stream_counts={"clob_market": 3, "polymarket_rtds": 2},
            conditions=("0x" + "1" * 64,),
            integrity_errors=(),
            errors=(),
        )


def _config(tmp_path: Path) -> PolymarketRound25CampaignConfig:
    plan_path = tmp_path / "data" / "plan.json"
    write_round25_campaign_plan(plan_path, _plan())
    return PolymarketRound25CampaignConfig(
        repository=tmp_path,
        plan_path=plan_path,
        database_path=tmp_path / "data" / "capture.duckdb",
        state_root=tmp_path / "data" / "state",
    )


def test_campaign_persists_manifest_progress_and_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    clock = [POLYMARKET_ROUND25_START_MS + 1_000]
    recorder = _Recorder(clock)
    monkeypatch.setattr(round25, "load_round25_design", lambda _repository: {})
    monkeypatch.setattr(round25, "_verify_repository", lambda _repository, _plan: None)
    monkeypatch.setattr(round25.time, "time_ns", lambda: clock[0] * 1_000_000)

    result = asyncio.run(
        run_round25_campaign(
            config,
            recorder_factory=lambda _database: recorder,
        )
    )

    assert result["status"] == "campaign_window_ended"
    assert result["status_counts"] == {"complete": 1}
    assert recorder.calls == 1
    assert (config.state_root / "segment-0000-manifest.json").is_file()
    assert (config.state_root / "segment-0000-result.json").is_file()


def test_campaign_does_not_retry_a_source_regime_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    clock = [POLYMARKET_ROUND25_START_MS + 1_000]
    recorder = _Recorder(clock, source_failure=True)
    monkeypatch.setattr(round25, "load_round25_design", lambda _repository: {})
    monkeypatch.setattr(round25, "_verify_repository", lambda _repository, _plan: None)
    monkeypatch.setattr(round25.time, "time_ns", lambda: clock[0] * 1_000_000)

    result = asyncio.run(
        run_round25_campaign(
            config,
            recorder_factory=lambda _database: recorder,
        )
    )

    assert result["status"] == "source_regime_changed"
    assert result["status_counts"] == {"failed": 1}
    assert recorder.calls == 1
