from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_autonomous import (
    PolymarketAutonomousOpenProposal,
    submit_promoted_open,
)
from simple_ai_trading.polymarket_external_signal import (
    PolymarketBtcReferenceFeatures,
    PolymarketExternalSignalDecision,
)
from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderRecord,
    PolymarketOpenQuote,
)
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
NOW_MS = EVENT_START_MS + 120_000
MARKET_ID = "0x" + "1" * 64
TOKEN_ID = "1" * 40
MODEL_SHA = "2" * 64
BOOK_SHA = "3" * 64
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


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_promotion(
    root: Path,
    *,
    live: bool = True,
    expires_at_ms: int = NOW_MS + 86_400_000,
    require_live_authority: bool = True,
) -> VerifiedPolymarketLivePromotion:
    root.mkdir(exist_ok=True)
    model_path = root / "model.json"
    report_path = root / "evaluation.json"
    manifest_path = root / "implementation.json"
    model_path.write_bytes(b'{"model":"frozen"}\n')
    report_path.write_bytes(b'{"evaluation":"passed"}\n')
    manifest_path.write_bytes(b'{"implementation":"frozen"}\n')
    body: dict[str, object] = {
        "schema_version": "polymarket-live-promotion-v1",
        "promotion_id": "a" * 64,
        "created_at_ms": NOW_MS - 86_400_000,
        "expires_at_ms": expires_at_ms,
        "source_commit": "b" * 40,
        "venue": "polymarket",
        "protocol_version": 2,
        "asset": "BTC",
        "market_variant": "fiveminute",
        "environment": "live",
        "bot_id": "simple-ai-trading-polymarket-btc",
        "model_artifact": {
            "path": model_path.name,
            "sha256": _file_sha(model_path),
        },
        "evaluation_report": {
            "path": report_path.name,
            "sha256": _file_sha(report_path),
        },
        "implementation_manifest": {
            "path": manifest_path.name,
            "sha256": _file_sha(manifest_path),
        },
        "gates": dict(GATES),
        "policy": {
            "minimum_expected_edge_quote_per_share": "0.02",
            "maximum_prediction_age_ms": 1_000,
            "minimum_remaining_seconds": 30,
        },
        "authority": {"paper": True, "live": live},
    }
    payload = {**body, "promotion_sha256": _canonical_sha(body)}
    path = root / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")
    return load_polymarket_live_promotion(
        path,
        evidence_root=root,
        require_live_authority=require_live_authority,
        observed_at_ms=NOW_MS,
    )


def _proposal(
    promotion: VerifiedPolymarketLivePromotion,
    *,
    probability: str = "0.60",
    quantity: str = "5",
    decision_time_ms: int = NOW_MS - 200,
    expires_at_ms: int = EVENT_END_MS,
    model_sha: str | None = None,
) -> PolymarketAutonomousOpenProposal:
    policy = promotion.promotion
    return PolymarketAutonomousOpenProposal(
        proposal_id="round14-btc-5m-proposal",
        input_sha256="4" * 64,
        model_artifact_sha256=model_sha or policy.model_artifact.sha256,
        promotion_sha256=policy.promotion_sha256,
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        market_variant="fiveminute",
        outcome="Up",
        selected_outcome_probability=Decimal(probability),
        requested_quantity=Decimal(quantity),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=decision_time_ms,
        expires_at_ms=expires_at_ms,
    )


class _Venue:
    def __init__(self, *, observed_at_ms: int = NOW_MS) -> None:
        self.observed_at_ms = observed_at_ms
        self.minimum_order_size = Decimal("5")
        self.calls: list[dict[str, object]] = []

    def open_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        outcome: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketOpenQuote:
        self.calls.append(
            {
                "market_id": market_id,
                "token_id": token_id,
                "outcome": outcome,
                "quantity": quantity,
                "maximum_book_age_ms": maximum_book_age_ms,
            }
        )
        if quantity < self.minimum_order_size:
            raise PolymarketLiveBlocked(
                "proposed Polymarket quantity is below the venue minimum"
            )
        fee = (quantity * Decimal("0.07") * Decimal("0.25")).quantize(
            Decimal("0.00001")
        )
        return PolymarketOpenQuote(
            market_id=market_id,
            token_id=token_id,
            outcome=outcome,
            quantity=quantity,
            limit_price=Decimal("0.50"),
            average_price=Decimal("0.50"),
            fee_quote=fee,
            total_quote=Decimal("0.50") * quantity + fee,
            fee_rate=Decimal("0.07"),
            fee_exponent=1,
            tick_size=Decimal("0.01"),
            minimum_order_size=self.minimum_order_size,
            neg_risk=False,
            source_time_ms=self.observed_at_ms,
            observed_at_ms=self.observed_at_ms,
            book_payload_sha256=BOOK_SHA,
        )


class _Coordinator:
    def __init__(self, venue: _Venue) -> None:
        self.venue = venue
        self.submissions: list[
            tuple[PolymarketLiveOrderIntent, Decimal, bool]
        ] = []

    def submit(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketLiveOrderRecord:
        self.submissions.append((intent, tick_size, neg_risk))
        return PolymarketLiveOrderRecord(
            intent=intent,
            expected_order_id="0x" + "9" * 64,
            state="filled",
            remote_status="MATCHED",
            matched_quantity=intent.quantity,
            failure_code="",
            updated_at_ms=NOW_MS,
        )


def _features(*, observed_at_ms: int = NOW_MS) -> PolymarketBtcReferenceFeatures:
    return PolymarketBtcReferenceFeatures(
        observed_at_ms=observed_at_ms,
        spot_mid=Decimal("64000"),
        futures_mid=Decimal("64001"),
        spot_spread_bps=Decimal("0.5"),
        futures_spread_bps=Decimal("0.6"),
        futures_basis_bps=Decimal("0.15625"),
        spot_log_return=0.0,
        futures_log_return=0.0,
        event_time_skew_ms=10,
        receive_time_skew_ms=10,
    )


def test_authority_absence_or_expiry_blocks_before_quote(tmp_path: Path) -> None:
    venue = _Venue()
    coordinator = _Coordinator(venue)
    no_live = _verified_promotion(
        tmp_path / "no-live",
        live=False,
        require_live_authority=False,
    )
    with pytest.raises(PolymarketLiveBlocked, match="no live authority"):
        submit_promoted_open(
            _proposal(no_live),
            no_live,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    expired = _verified_promotion(
        tmp_path / "expired",
        expires_at_ms=NOW_MS - 1,
        require_live_authority=False,
    )
    with pytest.raises(PolymarketLiveBlocked, match="expired"):
        submit_promoted_open(
            _proposal(expired),
            expired,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    assert venue.calls == []


def test_stale_or_late_proposal_blocks_before_quote(tmp_path: Path) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)
    with pytest.raises(PolymarketLiveBlocked, match="prediction is stale"):
        submit_promoted_open(
            _proposal(promotion, decision_time_ms=NOW_MS - 1_001),
            promotion,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    late_now = EVENT_END_MS - 29_999
    with pytest.raises(PolymarketLiveBlocked, match="too little event time"):
        submit_promoted_open(
            _proposal(promotion, decision_time_ms=late_now - 1),
            promotion,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: late_now,
        )
    assert venue.calls == []


def test_external_signal_can_only_veto_or_reduce(tmp_path: Path) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)
    abstain = PolymarketExternalSignalDecision(
        action="abstain",
        maximum_size_multiplier=Decimal("0"),
        reasons=("stale",),
        features=None,
    )
    with pytest.raises(PolymarketLiveBlocked, match="vetoed"):
        submit_promoted_open(
            _proposal(promotion),
            promotion,
            coordinator,  # type: ignore[arg-type]
            external_signal=abstain,
            clock_ms=lambda: NOW_MS,
        )
    reduce_below_minimum = PolymarketExternalSignalDecision(
        action="reduce",
        maximum_size_multiplier=Decimal("0.5"),
        reasons=("wide_reference_spread",),
        features=_features(),
    )
    with pytest.raises(PolymarketLiveBlocked, match="below the venue minimum"):
        submit_promoted_open(
            _proposal(promotion),
            promotion,
            coordinator,  # type: ignore[arg-type]
            external_signal=reduce_below_minimum,
            clock_ms=lambda: NOW_MS,
        )
    assert len(venue.calls) == 1
    assert coordinator.submissions == []


def test_after_cost_edge_gate_counts_worst_fee(tmp_path: Path) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)

    with pytest.raises(PolymarketLiveBlocked, match="after-cost edge"):
        submit_promoted_open(
            _proposal(promotion, probability="0.51"),
            promotion,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )

    assert len(venue.calls) == 1
    assert coordinator.submissions == []


def test_valid_promoted_proposal_submits_exact_conservative_fok(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)
    reduce = PolymarketExternalSignalDecision(
        action="reduce",
        maximum_size_multiplier=Decimal("0.5"),
        reasons=("wide_reference_spread",),
        features=_features(),
    )

    result = submit_promoted_open(
        _proposal(promotion, quantity="10"),
        promotion,
        coordinator,  # type: ignore[arg-type]
        external_signal=reduce,
        clock_ms=lambda: NOW_MS,
    )

    intent, tick_size, neg_risk = coordinator.submissions[0]
    assert intent.order_type == "FOK"
    assert intent.side == "BUY"
    assert intent.outcome == "Up"
    assert intent.quantity == Decimal("5.000000")
    assert intent.limit_price == Decimal("0.50")
    assert intent.fee_reserve_quote == Decimal("0.08750")
    assert result.worst_notional_quote == Decimal("2.50000000")
    assert result.net_edge_quote_per_share == Decimal("0.0825")
    assert tick_size == Decimal("0.01")
    assert neg_risk is False


def test_model_hash_mismatch_and_unverified_object_block_before_quote(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)
    with pytest.raises(PolymarketLiveBlocked, match="model hash differs"):
        submit_promoted_open(
            _proposal(promotion, model_sha="f" * 64),
            promotion,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    with pytest.raises(PolymarketLiveBlocked, match="verified promotion evidence"):
        submit_promoted_open(
            _proposal(promotion),
            promotion.promotion,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    assert venue.calls == []


def test_proposal_hash_tampering_is_rejected(tmp_path: Path) -> None:
    promotion = _verified_promotion(tmp_path)
    proposal = _proposal(promotion)
    values = proposal.asdict()
    values["requested_quantity"] = "6"

    with pytest.raises(ValueError, match="proposal hash differs"):
        PolymarketAutonomousOpenProposal(
            proposal_id=str(values["proposal_id"]),
            input_sha256=str(values["input_sha256"]),
            model_artifact_sha256=str(values["model_artifact_sha256"]),
            promotion_sha256=str(values["promotion_sha256"]),
            market_id=str(values["market_id"]),
            token_id=str(values["token_id"]),
            symbol=str(values["symbol"]),
            market_variant=str(values["market_variant"]),
            outcome=str(values["outcome"]),
            selected_outcome_probability=Decimal(
                str(values["selected_outcome_probability"])
            ),
            requested_quantity=Decimal(str(values["requested_quantity"])),
            event_start_time_ms=int(values["event_start_time_ms"]),
            event_end_time_ms=int(values["event_end_time_ms"]),
            decision_time_ms=int(values["decision_time_ms"]),
            expires_at_ms=int(values["expires_at_ms"]),
            proposal_sha256=str(values["proposal_sha256"]),
        )
