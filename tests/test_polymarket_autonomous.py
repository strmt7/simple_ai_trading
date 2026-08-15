from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from polymarket_live_support import write_polymarket_live_implementation_manifest
from simple_ai_trading.polymarket_autonomous import (
    PolymarketAutonomousLockProposal,
    PolymarketAutonomousOpenProposal,
    PolymarketAutonomousReduceProposal,
    submit_promoted_lock,
    submit_promoted_open,
    submit_promoted_reduction,
)
from simple_ai_trading.polymarket_external_signal import (
    PolymarketBtcReferenceFeatures,
    PolymarketExternalSignalDecision,
)
from simple_ai_trading.polymarket_live import (
    PolymarketCloseQuote,
    PolymarketConditionAccounting,
    PolymarketLiveBlocked,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderRecord,
    PolymarketOpenQuote,
)
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
)
from simple_ai_trading.polymarket_live_qualification import (
    VerifiedPolymarketLifecycleQualification,
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
    write_polymarket_live_implementation_manifest(
        manifest_path,
        source_commit="b" * 40,
    )
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


def _qualification() -> VerifiedPolymarketLifecycleQualification:
    return Mock(spec=VerifiedPolymarketLifecycleQualification)


def _proposal(
    promotion: VerifiedPolymarketLivePromotion,
    *,
    probability: str = "0.60",
    quantity: str = "5",
    decision_time_ms: int = NOW_MS - 200,
    expires_at_ms: int = EVENT_END_MS,
    model_sha: str | None = None,
    maximum_downside: str = "10",
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
        risk_state_sha256="9" * 64,
        maximum_projected_inventory_downside_quote=Decimal(maximum_downside),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=decision_time_ms,
        expires_at_ms=expires_at_ms,
    )


def _reduction(
    promotion: VerifiedPolymarketLivePromotion,
    *,
    maximum_probability: str = "0.60",
) -> PolymarketAutonomousReduceProposal:
    policy = promotion.promotion
    return PolymarketAutonomousReduceProposal(
        proposal_id="round21-btc-5m-reduction",
        input_sha256="5" * 64,
        model_artifact_sha256=policy.model_artifact.sha256,
        promotion_sha256=policy.promotion_sha256,
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        market_variant="fiveminute",
        outcome="Up",
        parent_intent_id="poly-open-parent-0001",
        maximum_outcome_probability=Decimal(maximum_probability),
        requested_quantity=Decimal("5"),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=NOW_MS - 200,
        expires_at_ms=EVENT_END_MS,
    )


def _lock(
    promotion: VerifiedPolymarketLivePromotion,
    *,
    quantity: str = "5",
) -> PolymarketAutonomousLockProposal:
    policy = promotion.promotion
    return PolymarketAutonomousLockProposal(
        proposal_id="round21-btc-5m-lock",
        input_sha256="6" * 64,
        model_artifact_sha256=policy.model_artifact.sha256,
        promotion_sha256=policy.promotion_sha256,
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        market_variant="fiveminute",
        outcome="Down",
        owned_outcome="Up",
        requested_quantity=Decimal(quantity),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=NOW_MS - 200,
        expires_at_ms=EVENT_END_MS,
    )


def test_fifteen_minute_proposal_has_horizon_bound_chronology() -> None:
    values = {
        "proposal_id": "fifteen-minute-proposal",
        "input_sha256": "1" * 64,
        "model_artifact_sha256": "2" * 64,
        "promotion_sha256": "3" * 64,
        "market_id": "0x" + "4" * 64,
        "token_id": "5" * 40,
        "symbol": "BTC",
        "market_variant": "fifteenminute",
        "outcome": "Up",
        "selected_outcome_probability": Decimal("0.60"),
        "requested_quantity": Decimal("5.000000"),
        "risk_state_sha256": "6" * 64,
        "maximum_projected_inventory_downside_quote": Decimal("10"),
        "event_start_time_ms": 1_800_000_000_000,
        "event_end_time_ms": 1_800_000_900_000,
        "decision_time_ms": 1_800_000_120_000,
        "expires_at_ms": 1_800_000_121_000,
    }
    proposal = PolymarketAutonomousOpenProposal(**values)

    assert proposal.market_variant == "fifteenminute"
    with pytest.raises(ValueError, match="chronology"):
        PolymarketAutonomousOpenProposal(
            **{
                **values,
                "event_end_time_ms": 1_800_000_300_000,
            }
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
        self.submissions: list[tuple[PolymarketLiveOrderIntent, Decimal, bool]] = []
        self.targeted_closes: list[dict[str, object]] = []
        self.ledger = Mock()
        self.ledger.condition_accounting.return_value = PolymarketConditionAccounting(
            condition_id=MARKET_ID,
            gross_buy_cost_quote=Decimal("2.50"),
            gross_sell_proceeds_quote=Decimal("0"),
            confirmed_redemption_payout_quote=Decimal("0"),
            up_quantity=Decimal("5"),
            down_quantity=Decimal("0"),
            up_cost_basis_quote=Decimal("2.00"),
            down_cost_basis_quote=Decimal("0"),
            confirmed_fill_count=1,
        )

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

    def submit_owned_close_order(
        self,
        *,
        parent_intent_id: str,
        quantity: Decimal,
        action_binding_sha256: str,
        minimum_net_quote: Decimal,
        maximum_book_age_ms: int,
    ) -> tuple[PolymarketLiveOrderRecord, PolymarketCloseQuote]:
        self.targeted_closes.append(
            {
                "parent_intent_id": parent_intent_id,
                "quantity": quantity,
                "action_binding_sha256": action_binding_sha256,
                "minimum_net_quote": minimum_net_quote,
                "maximum_book_age_ms": maximum_book_age_ms,
            }
        )
        net_quote = Decimal("0.70") * quantity
        if net_quote < minimum_net_quote:
            raise PolymarketLiveBlocked(
                "targeted close no longer meets its after-cost proceeds floor"
            )
        quote = PolymarketCloseQuote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=quantity,
            limit_price=Decimal("0.70"),
            average_price=Decimal("0.70"),
            fee_quote=Decimal("0"),
            net_quote=net_quote,
            fee_rate=Decimal("0"),
            fee_exponent=1,
            tick_size=Decimal("0.01"),
            minimum_order_size=Decimal("5"),
            neg_risk=False,
            source_time_ms=NOW_MS,
            observed_at_ms=NOW_MS,
            book_payload_sha256=BOOK_SHA,
        )
        intent = PolymarketLiveOrderIntent(
            intent_id="policy-close-result-0001",
            bot_id="simple-ai-trading-polymarket-btc",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            symbol="BTC",
            outcome="Up",
            side="SELL",
            order_type="FAK",
            limit_price=quote.limit_price,
            quantity=quantity,
            fee_reserve_quote=quote.fee_quote,
            created_at_ms=NOW_MS,
            expires_at_ms=NOW_MS + 1_000,
            parent_intent_id=parent_intent_id,
            closing_only=True,
        )
        return (
            PolymarketLiveOrderRecord(
                intent=intent,
                expected_order_id="0x" + "8" * 64,
                state="filled",
                remote_status="MATCHED",
                matched_quantity=quantity,
                failure_code="",
                updated_at_ms=NOW_MS,
            ),
            quote,
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
            lifecycle_qualification=_qualification(),
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
            lifecycle_qualification=_qualification(),
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
            lifecycle_qualification=_qualification(),
            clock_ms=lambda: NOW_MS,
        )
    late_now = EVENT_END_MS - 29_999
    with pytest.raises(PolymarketLiveBlocked, match="too little event time"):
        submit_promoted_open(
            _proposal(promotion, decision_time_ms=late_now - 1),
            promotion,
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
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
            lifecycle_qualification=_qualification(),
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
            lifecycle_qualification=_qualification(),
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
            lifecycle_qualification=_qualification(),
            clock_ms=lambda: NOW_MS,
        )

    assert len(venue.calls) == 1
    assert coordinator.submissions == []


def test_execution_quote_cannot_exceed_bound_live_risk_headroom(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)

    with pytest.raises(PolymarketLiveBlocked, match="live risk headroom"):
        submit_promoted_open(
            _proposal(promotion, maximum_downside="4.5"),
            promotion,
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
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
        lifecycle_qualification=_qualification(),
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
    assert result.projected_inventory_downside_quote == Decimal("4.58750000")
    assert result.net_edge_quote_per_share == Decimal("0.0825")
    assert tick_size == Decimal("0.01")
    assert neg_risk is False


def test_valid_promoted_reduction_submits_one_parent_bound_fak(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    coordinator = _Coordinator(_Venue())
    proposal = _reduction(promotion)

    result = submit_promoted_reduction(
        proposal,
        promotion,
        coordinator,  # type: ignore[arg-type]
        lifecycle_qualification=_qualification(),
        clock_ms=lambda: NOW_MS,
    )

    assert len(coordinator.targeted_closes) == 1
    request = coordinator.targeted_closes[0]
    assert request["parent_intent_id"] == proposal.parent_intent_id
    assert request["quantity"] == Decimal("5")
    assert request["action_binding_sha256"] == proposal.proposal_sha256
    assert request["minimum_net_quote"] == Decimal("3.10")
    assert result.record.intent.closing_only is True
    assert result.record.intent.side == "SELL"
    assert result.net_edge_quote_per_share == Decimal("0.10")


def test_promoted_reduction_rechecks_after_cost_floor_before_submission(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    coordinator = _Coordinator(_Venue())

    with pytest.raises(PolymarketLiveBlocked, match="proceeds floor"):
        submit_promoted_reduction(
            _reduction(promotion, maximum_probability="0.69"),
            promotion,
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
            clock_ms=lambda: NOW_MS,
        )

    assert len(coordinator.targeted_closes) == 1
    assert coordinator.submissions == []


def test_promoted_lock_is_fak_and_cannot_increase_event_downside(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    coordinator = _Coordinator(_Venue())

    result = submit_promoted_lock(
        _lock(promotion),
        promotion,
        coordinator,  # type: ignore[arg-type]
        lifecycle_qualification=_qualification(),
        clock_ms=lambda: NOW_MS,
    )

    intent, tick_size, neg_risk = coordinator.submissions[0]
    assert intent.intent_id.startswith("poly-lock-")
    assert intent.order_type == "FAK"
    assert intent.side == "BUY"
    assert intent.outcome == "Down"
    assert intent.quantity == Decimal("5")
    assert result.projected_maximum_loss_quote <= (
        result.prior_accounting.maximum_loss_quote
    )
    assert result.locked_edge_quote_per_share == Decimal("0.0825")
    assert tick_size == Decimal("0.01")
    assert neg_risk is False


def test_promoted_lock_cannot_exceed_unpaired_owned_inventory(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)

    with pytest.raises(PolymarketLiveBlocked, match="unpaired"):
        submit_promoted_lock(
            _lock(promotion, quantity="5.000001"),
            promotion,
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
            clock_ms=lambda: NOW_MS,
        )

    assert venue.calls == []
    assert coordinator.submissions == []


def test_model_hash_mismatch_and_unverified_object_block_before_quote(
    tmp_path: Path,
) -> None:
    promotion = _verified_promotion(tmp_path)
    venue = _Venue()
    coordinator = _Coordinator(venue)
    with pytest.raises(PolymarketLiveBlocked, match="authenticated lifecycle"):
        submit_promoted_open(
            _proposal(promotion),
            promotion,
            coordinator,  # type: ignore[arg-type]
            clock_ms=lambda: NOW_MS,
        )
    with pytest.raises(PolymarketLiveBlocked, match="model hash differs"):
        submit_promoted_open(
            _proposal(promotion, model_sha="f" * 64),
            promotion,
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
            clock_ms=lambda: NOW_MS,
        )
    with pytest.raises(PolymarketLiveBlocked, match="verified promotion evidence"):
        submit_promoted_open(
            _proposal(promotion),
            promotion.promotion,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
            lifecycle_qualification=_qualification(),
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
            risk_state_sha256=str(values["risk_state_sha256"]),
            maximum_projected_inventory_downside_quote=Decimal(
                str(values["maximum_projected_inventory_downside_quote"])
            ),
            event_start_time_ms=int(values["event_start_time_ms"]),
            event_end_time_ms=int(values["event_end_time_ms"]),
            decision_time_ms=int(values["decision_time_ms"]),
            expires_at_ms=int(values["expires_at_ms"]),
            proposal_sha256=str(values["proposal_sha256"]),
        )
