from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFifteenMinuteMarket,
    PolymarketPublicClient,
)
from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
)
from simple_ai_trading.polymarket_round16_decision import (
    PolymarketRound16PromotedDecisionProvider,
)
from simple_ai_trading.polymarket_round16_shadow import (
    PolymarketRound16ShadowDecision,
    PolymarketRound16ShadowScorer,
)


EVENT_START_MS = 1_800_000_000_000
NOW_MS = EVENT_START_MS + 120_000
MARKET_ID = "0x" + "1" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
GATES = {
    "prospective_untouched_test": True,
    "source_integrity": True,
    "causal_feature_replay": True,
    "proper_scoring_uplift": True,
    "after_cost_edge": True,
    "uncertainty_lower_bound": True,
    "drawdown_limit": True,
    "latency_stress": True,
    "displayed_depth_stress": True,
    "authenticated_order_lifecycle": True,
    "settlement_and_redemption": True,
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promotion(root: Path) -> VerifiedPolymarketLivePromotion:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    evidence: dict[str, dict[str, str]] = {}
    for name in ("model", "evaluation", "implementation"):
        path = root / f"{name}.json"
        path.write_text(_canonical({name: "frozen"}), encoding="ascii")
        paths[name] = path
        evidence[name] = {"path": path.name, "sha256": _file_sha(path)}
    body: dict[str, object] = {
        "schema_version": "polymarket-live-promotion-v1",
        "promotion_id": "a" * 64,
        "created_at_ms": NOW_MS - 60_000,
        "expires_at_ms": NOW_MS + 60_000,
        "source_commit": "b" * 40,
        "venue": "polymarket",
        "protocol_version": 2,
        "asset": "BTC",
        "market_variant": "fifteenminute",
        "environment": "live",
        "bot_id": "round16-decision-test",
        "model_artifact": evidence["model"],
        "evaluation_report": evidence["evaluation"],
        "implementation_manifest": evidence["implementation"],
        "gates": GATES,
        "policy": {
            "minimum_expected_edge_quote_per_share": "0.02",
            "maximum_prediction_age_ms": 1_000,
            "minimum_remaining_seconds": 30,
        },
        "authority": {"paper": True, "live": True},
    }
    payload = {**body, "promotion_sha256": _sha(body)}
    path = root / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")
    return load_polymarket_live_promotion(
        path,
        evidence_root=root,
        require_live_authority=True,
        observed_at_ms=NOW_MS,
    )


def _market() -> PolymarketFifteenMinuteMarket:
    return PolymarketFifteenMinuteMarket(
        asset="BTC",
        market_id="123",
        condition_id=MARKET_ID,
        slug=f"btc-updown-15m-{EVENT_START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=EVENT_START_MS,
        end_ms=EVENT_START_MS + 900_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.25"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("10000"),
        volume_quote=Decimal("100000"),
        resolution_source="https://data.chain.link/streams/btc-usd",
        gamma_payload_sha256="3" * 64,
        gamma_payload_json="{}",
    )


class _PublicClient(PolymarketPublicClient):
    def __init__(
        self,
        *,
        ask_price: str = "0.40",
        ask_size: str = "100",
        source_time_ms: int = NOW_MS,
    ) -> None:
        self.ask_price = ask_price
        self.ask_size = ask_size
        self.source_time_ms = source_time_ms
        self.calls = 0

    def order_book(self, token_id: str) -> dict[str, object]:
        self.calls += 1
        return {
            "market": MARKET_ID,
            "asset_id": token_id,
            "timestamp": str(self.source_time_ms),
            "bids": [{"price": "0.39", "size": "100"}],
            "asks": [{"price": self.ask_price, "size": self.ask_size}],
        }


class _Scorer(PolymarketRound16ShadowScorer):
    def __init__(
        self,
        *,
        promotion: VerifiedPolymarketLivePromotion,
        probability_up: float,
    ) -> None:
        self.predictor = type(
            "Predictor",
            (),
            {
                "pretest_file_sha256": (
                    promotion.promotion.model_artifact.sha256
                ),
                "evaluation_file_sha256": (
                    promotion.promotion.evaluation_report.sha256
                ),
            },
        )()
        self.probability_up = probability_up

    def evaluate(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> PolymarketRound16ShadowDecision:
        return PolymarketRound16ShadowDecision(
            status="observed",
            reason="",
            event_start_ms=event_start_ms,
            decision_time_ms=decision_time_ms,
            observed_at_ms=observed_at_ms,
            probability_up=self.probability_up,
            candidate_id="fixture",
            pretest_envelope_sha256="4" * 64,
            evaluation_envelope_sha256="5" * 64,
            input_sha256="6" * 64,
        )


def test_promoted_round16_provider_builds_one_polymarket_only_proposal(
    tmp_path: Path,
) -> None:
    promotion = _promotion(tmp_path)
    client = _PublicClient()
    provider = PolymarketRound16PromotedDecisionProvider(
        public_client=client,
        scorer=_Scorer(promotion=promotion, probability_up=0.7),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    )

    first = provider.decide(markets=(_market(),), observed_at_ms=NOW_MS)
    second = provider.decide(markets=(_market(),), observed_at_ms=NOW_MS)

    assert len(first.proposals) == 1
    proposal = first.proposals[0]
    assert proposal.market_id == MARKET_ID
    assert proposal.token_id == UP_TOKEN
    assert proposal.outcome == "Up"
    assert proposal.market_variant == "fifteenminute"
    assert proposal.model_artifact_sha256 == (
        promotion.promotion.model_artifact.sha256
    )
    assert len(proposal.input_sha256) == 64
    assert second.proposals == ()
    assert second.reasons == ("model_timestamp_already_consumed",)
    assert client.calls == 1


def test_round16_provider_rejects_edge_and_evidence_drift(
    tmp_path: Path,
) -> None:
    promotion = _promotion(tmp_path)
    low_edge = PolymarketRound16PromotedDecisionProvider(
        public_client=_PublicClient(ask_price="0.60"),
        scorer=_Scorer(promotion=promotion, probability_up=0.51),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    ).decide(markets=(_market(),), observed_at_ms=NOW_MS)

    assert low_edge.proposals == ()
    assert low_edge.reasons == ("insufficient_after_cost_edge",)

    scorer = _Scorer(promotion=promotion, probability_up=0.7)
    scorer.predictor.pretest_file_sha256 = "0" * 64
    with pytest.raises(PolymarketLiveBlocked, match="promotion evidence"):
        PolymarketRound16PromotedDecisionProvider(
            public_client=_PublicClient(),
            scorer=scorer,
            promotion=promotion,
            requested_quantity=Decimal("5"),
        )


def test_round16_provider_selects_down_and_rejects_unsafe_books(
    tmp_path: Path,
) -> None:
    promotion = _promotion(tmp_path)
    down = PolymarketRound16PromotedDecisionProvider(
        public_client=_PublicClient(),
        scorer=_Scorer(promotion=promotion, probability_up=0.3),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    ).decide(markets=(_market(),), observed_at_ms=NOW_MS)
    stale = PolymarketRound16PromotedDecisionProvider(
        public_client=_PublicClient(source_time_ms=NOW_MS - 1_001),
        scorer=_Scorer(promotion=promotion, probability_up=0.7),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    ).decide(markets=(_market(),), observed_at_ms=NOW_MS)
    shallow = PolymarketRound16PromotedDecisionProvider(
        public_client=_PublicClient(ask_size="4.999999"),
        scorer=_Scorer(promotion=promotion, probability_up=0.7),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    ).decide(markets=(_market(),), observed_at_ms=NOW_MS)

    assert down.proposals[0].outcome == "Down"
    assert down.proposals[0].token_id == DOWN_TOKEN
    assert stale.proposals == ()
    assert stale.reasons == ("polymarket_book_stale",)
    assert shallow.proposals == ()
    assert shallow.reasons == ("insufficient_displayed_ask_depth",)


def test_round16_provider_counts_book_fetch_time_against_prediction_ttl(
    tmp_path: Path,
) -> None:
    promotion = _promotion(tmp_path)
    times = iter((NOW_MS, NOW_MS + 1_001))
    expired = PolymarketRound16PromotedDecisionProvider(
        public_client=_PublicClient(source_time_ms=NOW_MS + 1_001),
        scorer=_Scorer(promotion=promotion, probability_up=0.7),
        promotion=promotion,
        requested_quantity=Decimal("5"),
        clock_ms=lambda: next(times),
        monotonic_ns=lambda: 1,
    ).decide(markets=(_market(),), observed_at_ms=NOW_MS)

    assert expired.proposals == ()
    assert expired.reasons == ("proposal_expired_during_book_fetch",)
