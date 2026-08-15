from __future__ import annotations

import asyncio
import ast
from collections.abc import Callable
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

import simple_ai_trading.polymarket_live_settlement as settlement_module
from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveError,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderLedger,
    PolymarketLiveUnknownState,
    PolymarketOwnedInventory,
    PolymarketPreparedOrder,
    PolymarketRedemptionRecord,
    PolymarketRemoteFill,
    PolymarketRemoteOrder,
    PolymarketRemotePosition,
    PolymarketStateConflict,
)
from simple_ai_trading.polymarket_live_settlement import (
    OfficialPolymarketUnifiedRedemptionVenue,
    PolymarketGaslessCredentials,
    PolymarketRedemptionCoordinator,
    PolymarketRedemptionFailed,
    PolymarketRedemptionOutcome,
    PolymarketRedemptionPreflight,
    PolymarketRedemptionRecovery,
    PolymarketRedemptionRejected,
    PolymarketRedemptionSubmission,
    PolymarketSettlementService,
)
from simple_ai_trading.polymarket_live_v2 import PolymarketLiveCredentials


NOW_MS = 1_800_000_000_000
CONDITION_ID = "0x" + "1" * 64
OTHER_CONDITION_ID = "0x" + "2" * 64
TOKEN_ID = "1" * 40
OTHER_TOKEN_ID = "2" * 40
ORDER_ID = "0x" + "3" * 64
TRANSACTION_HASH = "0x" + "4" * 64
OTHER_TRANSACTION_HASH = "0x" + "5" * 64
WALLET = "0x" + "a" * 40
ADAPTER = "0x" + "b" * 40
CONDITIONAL_TOKENS = "0x" + "c" * 40
COLLATERAL_TOKEN = "0x" + "d" * 40
USDCE = "0x" + "e" * 40
NEG_RISK_ADAPTER = "0x" + "f" * 40
BLOCK_HASH = "0x" + "6" * 64
PAYOUT_PROOF = "f" * 64


def _topic_address(address: str) -> str:
    return "0x" + address[2:].lower().rjust(64, "0")


def _words(*values: int) -> str:
    return "0x" + "".join(f"{value:064x}" for value in values)


def _standard_redemption_receipt(
    *,
    transaction_hash: str = TRANSACTION_HASH,
    payout_base_units: int = 2_000_000,
    status: str = "0x1",
) -> dict[str, object]:
    return {
        "transactionHash": transaction_hash,
        "blockHash": BLOCK_HASH,
        "blockNumber": "0x10",
        "status": status,
        "logs": [
            {
                "address": CONDITIONAL_TOKENS,
                "topics": [
                    settlement_module._CTF_PAYOUT_EVENT_TOPIC,
                    _topic_address(ADAPTER),
                    _topic_address(USDCE),
                    "0x" + "0" * 64,
                ],
                "data": _words(
                    int(CONDITION_ID, 16),
                    96,
                    payout_base_units,
                    2,
                    1,
                    2,
                ),
                "removed": False,
            },
            {
                "address": COLLATERAL_TOKEN,
                "topics": [
                    settlement_module._WRAPPED_EVENT_TOPIC,
                    _topic_address(ADAPTER),
                    _topic_address(USDCE),
                    _topic_address(WALLET),
                ],
                "data": _words(payout_base_units),
                "removed": False,
            },
            {
                "address": COLLATERAL_TOKEN,
                "topics": [
                    settlement_module._TRANSFER_EVENT_TOPIC,
                    "0x" + "0" * 64,
                    _topic_address(WALLET),
                ],
                "data": _words(payout_base_units),
                "removed": False,
            },
        ],
    }


def _negative_risk_redemption_receipt(
    *,
    payout_base_units: int = 2_000_000,
) -> dict[str, object]:
    receipt = _standard_redemption_receipt(
        payout_base_units=payout_base_units,
    )
    receipt["logs"] = [
        {
            "address": NEG_RISK_ADAPTER,
            "topics": [
                settlement_module._NEG_RISK_PAYOUT_EVENT_TOPIC,
                _topic_address(ADAPTER),
                CONDITION_ID,
            ],
            "data": _words(64, payout_base_units, 2, payout_base_units, 0),
            "removed": False,
        },
        *receipt["logs"][1:],  # type: ignore[index]
    ]
    return receipt


def _preflight_payload() -> dict[str, object]:
    return {
        "condition_id": CONDITION_ID,
        "adapter_address": ADAPTER,
        "neg_risk": False,
        "gas_estimate": 100_000,
        "gas_price_wei": 1_000,
        "native_balance_wei": 1_000_000_000,
        "required_native_balance_wei": 200_000_000,
        "gasless": False,
    }


def _intent() -> PolymarketLiveOrderIntent:
    return PolymarketLiveOrderIntent(
        intent_id="redemption-intent-0001",
        bot_id="polymarket-live-bot",
        market_id=CONDITION_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        outcome="Up",
        side="BUY",
        order_type="FAK",
        limit_price=Decimal("0.51"),
        quantity=Decimal("2"),
        fee_reserve_quote=Decimal("0.10"),
        created_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 120_000,
    )


def _seed_confirmed_inventory(
    ledger: PolymarketLiveOrderLedger,
    *,
    fill_status: str = "CONFIRMED",
) -> None:
    intent = _intent()
    prepared = PolymarketPreparedOrder(
        intent=intent,
        expected_order_id=ORDER_ID,
        metadata=intent.metadata,
        opaque_signed_order={"signature": "never-persist"},
    )
    ledger.reserve(prepared, observed_at_ms=NOW_MS)
    ledger.record_fill(
        PolymarketRemoteFill(
            trade_id="redemption-trade-0001",
            order_id=ORDER_ID,
            market_id=CONDITION_ID,
            token_id=TOKEN_ID,
            side="BUY",
            quantity=Decimal("2"),
            price=Decimal("0.51"),
            status=fill_status,
            observed_at_ms=NOW_MS + 1,
            role="TAKER",
            reported_fee_rate_bps=0,
            fee_rate=Decimal("0"),
            fee_exponent=1,
            fee_quote=Decimal("0"),
            fee_schedule_sha256="a" * 64,
            transaction_hash="0x" + "d" * 64,
        )
    )
    ledger.transition(
        intent.intent_id,
        expected_states=("prepared",),
        state="filled",
        observed_at_ms=NOW_MS + 2,
        remote_status="CONFIRMED",
        matched_quantity=Decimal("2"),
    )


class FakeAccount:
    def __init__(self) -> None:
        self.open: tuple[PolymarketRemoteOrder, ...] = ()
        self.remote_positions: tuple[PolymarketRemotePosition, ...] = (
            PolymarketRemotePosition(
                market_id=CONDITION_ID,
                token_id=TOKEN_ID,
                quantity=Decimal("2"),
                redeemable=True,
            ),
        )

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]:
        return self.open

    def positions(self) -> tuple[PolymarketRemotePosition, ...]:
        return self.remote_positions


class FakeRedemptionVenue:
    def __init__(self) -> None:
        self.submit_calls: list[str] = []
        self.observed_states: list[str] = []
        self.ledger: PolymarketLiveOrderLedger | None = None
        self.submit_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.recover_error: Exception | None = None
        self.preflight_condition = CONDITION_ID
        self.transaction_hash = TRANSACTION_HASH
        self.recovery = PolymarketRedemptionRecovery(
            state="pending",
            transaction_id="",
            transaction_hash=TRANSACTION_HASH,
        )

    def assert_redemption_ready(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionPreflight:
        return PolymarketRedemptionPreflight(
            condition_id=self.preflight_condition,
            adapter_address=ADAPTER,
            neg_risk=False,
            gas_estimate=100_000,
            gas_price_wei=1_000,
            native_balance_wei=1_000_000_000,
            required_native_balance_wei=200_000_000,
        )

    def submit_redemption(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionSubmission:
        self.submit_calls.append(condition_id)
        if self.ledger is not None:
            self.observed_states.append(self.ledger.redemption_records()[-1].state)
        if self.submit_error is not None:
            raise self.submit_error
        return PolymarketRedemptionSubmission(
            transaction_id="",
            transaction_hash=self.transaction_hash,
            condition_id=condition_id,
            adapter_address=ADAPTER,
            neg_risk=False,
            _handle=object(),
        )

    def wait_redemption(
        self,
        submission: PolymarketRedemptionSubmission,
    ) -> PolymarketRedemptionOutcome:
        if self.wait_error is not None:
            raise self.wait_error
        return PolymarketRedemptionOutcome(
            transaction_id=submission.transaction_id,
            transaction_hash=submission.transaction_hash,
            payout_quote=Decimal("2"),
            payout_proof_sha256=PAYOUT_PROOF,
        )

    def recover_redemption(
        self,
        *,
        transaction_id: str,
        transaction_hash: str,
        condition_id: str,
        adapter_address: str,
        neg_risk: bool,
    ) -> PolymarketRedemptionRecovery:
        del transaction_id, transaction_hash, condition_id, adapter_address, neg_risk
        if self.recover_error is not None:
            raise self.recover_error
        return self.recovery


def _coordinator(
    tmp_path: Path,
) -> tuple[
    PolymarketLiveOrderLedger,
    FakeAccount,
    FakeRedemptionVenue,
    PolymarketRedemptionCoordinator,
]:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    _seed_confirmed_inventory(ledger)
    account = FakeAccount()
    venue = FakeRedemptionVenue()
    venue.ledger = ledger
    coordinator = PolymarketRedemptionCoordinator(
        account,
        venue,
        ledger,
        clock_ms=lambda: NOW_MS + 10,
    )
    return ledger, account, venue, coordinator


def _owned_inventory(
    *,
    market_id: str = CONDITION_ID,
    token_id: str = TOKEN_ID,
    quantity: str = "2",
    provisional: bool = False,
) -> PolymarketOwnedInventory:
    return PolymarketOwnedInventory(
        market_id=market_id,
        token_id=token_id,
        quantity=Decimal(quantity),
        provisional=provisional,
    )


def test_redemption_is_reserved_before_one_broadcast_and_removes_inventory(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)

    record = coordinator.redeem_next_ready()

    assert record is not None
    assert record.state == "confirmed"
    assert record.attempt == 1
    assert record.transaction_hash == TRANSACTION_HASH
    assert json.loads(record.preflight_json)["gas_estimate"] == 100_000
    assert venue.submit_calls == [CONDITION_ID]
    assert venue.observed_states == ["submitting"]
    assert ledger.owned_inventory() == ()


@pytest.mark.parametrize("foreign_kind", ["position", "open_order"])
def test_foreign_account_state_blocks_redemption_without_submission(
    tmp_path: Path,
    foreign_kind: str,
) -> None:
    _, account, venue, coordinator = _coordinator(tmp_path)
    if foreign_kind == "position":
        account.remote_positions += (
            PolymarketRemotePosition(
                market_id=OTHER_CONDITION_ID,
                token_id=OTHER_TOKEN_ID,
                quantity=Decimal("1"),
                redeemable=True,
            ),
        )
    else:
        intent = _intent()
        account.open = (
            PolymarketRemoteOrder(
                order_id="0x" + "6" * 64,
                market_id=intent.market_id,
                token_id=intent.token_id,
                maker_address=WALLET,
                side="BUY",
                order_type=intent.order_type,
                price=intent.limit_price,
                status="LIVE",
                original_quantity=Decimal("1"),
                matched_quantity=Decimal("0"),
            ),
        )

    with pytest.raises(PolymarketLiveBlocked):
        coordinator.redeem_next_ready()

    assert venue.submit_calls == []


def test_redemption_requires_redeemable_exact_position_economics(
    tmp_path: Path,
) -> None:
    _, account, venue, coordinator = _coordinator(tmp_path)
    account.remote_positions = (
        PolymarketRemotePosition(
            market_id=CONDITION_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("2.1"),
            redeemable=True,
        ),
    )
    with pytest.raises(PolymarketLiveBlocked, match="economics differ"):
        coordinator.redeem_next_ready()

    account.remote_positions = (
        PolymarketRemotePosition(
            market_id=CONDITION_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("2"),
            redeemable=False,
        ),
    )
    assert coordinator.redeem_next_ready() is None
    with pytest.raises(PolymarketLiveBlocked, match="not fully redeemable"):
        coordinator.redeem_next_ready(condition_id=CONDITION_ID)
    with pytest.raises(PolymarketLiveBlocked, match="no bot-owned inventory"):
        coordinator.redeem_next_ready(condition_id=OTHER_CONDITION_ID)

    assert venue.submit_calls == []


def test_duplicate_remote_position_or_wrong_preflight_condition_blocks(
    tmp_path: Path,
) -> None:
    _, account, venue, coordinator = _coordinator(tmp_path)
    account.remote_positions += account.remote_positions
    with pytest.raises(PolymarketLiveBlocked, match="duplicate"):
        coordinator.redeem_next_ready()

    account.remote_positions = account.remote_positions[:1]
    venue.preflight_condition = OTHER_CONDITION_ID
    with pytest.raises(PolymarketLiveBlocked, match="preflight condition"):
        coordinator.redeem_next_ready()

    assert venue.submit_calls == []


def test_open_owned_order_or_provisional_fill_blocks_redemption(
    tmp_path: Path,
) -> None:
    open_ledger = PolymarketLiveOrderLedger(tmp_path / "open-owned.sqlite3")
    intent = _intent()
    prepared = PolymarketPreparedOrder(
        intent=intent,
        expected_order_id=ORDER_ID,
        metadata=intent.metadata,
        opaque_signed_order={},
    )
    open_ledger.reserve(prepared, observed_at_ms=NOW_MS)
    open_ledger.transition(
        intent.intent_id,
        expected_states=("prepared",),
        state="live",
        observed_at_ms=NOW_MS + 1,
        remote_status="LIVE",
    )
    open_venue = FakeRedemptionVenue()
    open_coordinator = PolymarketRedemptionCoordinator(
        FakeAccount(),
        open_venue,
        open_ledger,
        clock_ms=lambda: NOW_MS + 10,
    )
    with pytest.raises(PolymarketLiveBlocked, match="open bot-owned"):
        open_coordinator.redeem_next_ready()

    provisional_ledger = PolymarketLiveOrderLedger(tmp_path / "provisional.sqlite3")
    _seed_confirmed_inventory(provisional_ledger, fill_status="MATCHED")
    provisional_coordinator = PolymarketRedemptionCoordinator(
        FakeAccount(),
        FakeRedemptionVenue(),
        provisional_ledger,
        clock_ms=lambda: NOW_MS + 10,
    )
    with pytest.raises(PolymarketLiveBlocked, match="provisional"):
        provisional_coordinator.redeem_next_ready()


def test_proven_prebroadcast_rejection_is_terminal_without_retry(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    venue.submit_error = PolymarketRedemptionRejected("market not resolved")

    record = coordinator.redeem_next_ready()

    assert record is not None and record.state == "failed"
    assert venue.submit_calls == [CONDITION_ID]
    assert ledger.redemption_records()[0].failure_code == (
        "PolymarketRedemptionRejected"
    )


def test_ambiguous_submission_is_never_retried_and_blocks_new_attempt(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    venue.submit_error = TimeoutError("response lost")

    with pytest.raises(PolymarketLiveUnknownState, match="submission outcome"):
        coordinator.redeem_next_ready()
    with pytest.raises(PolymarketLiveBlocked, match="unresolved"):
        coordinator.redeem_next_ready()

    assert venue.submit_calls == [CONDITION_ID]
    assert ledger.redemption_records()[0].state == "unknown"


def test_proven_failure_allows_only_a_new_numbered_attempt(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    venue.wait_error = PolymarketRedemptionFailed("receipt status zero")

    first = coordinator.redeem_next_ready()
    venue.wait_error = None
    venue.transaction_hash = OTHER_TRANSACTION_HASH
    venue.recovery = PolymarketRedemptionRecovery(
        state="confirmed",
        transaction_id="",
        transaction_hash=OTHER_TRANSACTION_HASH,
        payout_quote=Decimal("2"),
        payout_proof_sha256=PAYOUT_PROOF,
    )
    second = coordinator.redeem_next_ready()

    assert first is not None and first.state == "failed"
    assert second is not None and second.state == "confirmed"
    assert [record.attempt for record in ledger.redemption_records()] == [1, 2]
    assert venue.submit_calls == [CONDITION_ID, CONDITION_ID]


def test_unknown_wait_recovers_only_from_matching_receipt_proof(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    venue.wait_error = TimeoutError("receipt wait timed out")

    with pytest.raises(PolymarketLiveUnknownState, match="confirmation"):
        coordinator.redeem_next_ready()

    venue.recovery = PolymarketRedemptionRecovery(
        state="confirmed",
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        payout_quote=Decimal("2"),
        payout_proof_sha256=PAYOUT_PROOF,
    )
    recovered = coordinator.recover_incomplete()

    assert recovered[0].state == "confirmed"
    assert ledger.owned_inventory() == ()
    assert venue.submit_calls == [CONDITION_ID]


def test_restart_during_submission_stays_unknown_without_transaction_proof(
    tmp_path: Path,
) -> None:
    ledger, account, venue, coordinator = _coordinator(tmp_path)
    inventory = ledger.owned_inventory()
    prepared = ledger.reserve_redemption(
        CONDITION_ID,
        inventory,
        observed_at_ms=NOW_MS + 3,
    )
    ledger.transition_redemption(
        prepared.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 4,
    )

    recovered = coordinator.recover_incomplete()

    assert recovered[0].state == "unknown"
    assert account.remote_positions[0].redeemable is True
    assert venue.submit_calls == []


def test_restart_before_submission_is_proven_safe_failure(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    prepared = ledger.reserve_redemption(
        CONDITION_ID,
        ledger.owned_inventory(),
        observed_at_ms=NOW_MS + 3,
        preflight=_preflight_payload(),
    )

    recovered = coordinator.recover_incomplete()

    assert recovered[0].redemption_id == prepared.redemption_id
    assert recovered[0].state == "failed"
    assert coordinator.recover_incomplete() == ()
    assert venue.submit_calls == []


def test_recovery_query_failure_and_pending_state_never_trigger_retry(
    tmp_path: Path,
) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    prepared = ledger.reserve_redemption(
        CONDITION_ID,
        ledger.owned_inventory(),
        observed_at_ms=NOW_MS + 3,
        preflight=_preflight_payload(),
    )
    submitting = ledger.transition_redemption(
        prepared.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 4,
    )
    ledger.transition_redemption(
        submitting.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 5,
        transaction_hash=TRANSACTION_HASH,
    )
    venue.recover_error = TimeoutError("RPC unavailable")

    first = coordinator.recover_incomplete()
    second = coordinator.recover_incomplete()

    assert first[0].state == "unknown"
    assert second[0].state == "unknown"
    venue.recover_error = None
    venue.recovery = PolymarketRedemptionRecovery(
        state="pending",
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
    )
    pending = coordinator.recover_incomplete()
    assert pending[0].state == "unknown"
    venue.recovery = PolymarketRedemptionRecovery(
        state="failed",
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
    )
    failed = coordinator.recover_incomplete()
    assert failed[0].state == "failed"
    assert venue.submit_calls == []


def test_redemption_snapshot_tampering_is_detected(tmp_path: Path) -> None:
    ledger, _, venue, coordinator = _coordinator(tmp_path)
    venue.submit_error = TimeoutError("response lost")
    with pytest.raises(PolymarketLiveUnknownState):
        coordinator.redeem_next_ready()
    connection = sqlite3.connect(ledger.path)
    try:
        connection.execute(
            """
            UPDATE polymarket_live_redemptions
            SET transaction_hash = ?
            """,
            [OTHER_TRANSACTION_HASH],
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PolymarketLiveError, match="snapshot hash differs"):
        ledger.redemption_records()


def test_v2_redemption_ledger_migrates_without_inventing_payout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-redemption.sqlite3"
    ledger = PolymarketLiveOrderLedger(path)
    _seed_confirmed_inventory(ledger)
    prepared = ledger.reserve_redemption(
        CONDITION_ID,
        ledger.owned_inventory(),
        observed_at_ms=NOW_MS + 3,
        preflight=_preflight_payload(),
    )
    submitting = ledger.transition_redemption(
        prepared.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 4,
    )
    submitted = ledger.transition_redemption(
        submitting.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 5,
        transaction_hash=TRANSACTION_HASH,
    )
    ledger.transition_redemption(
        submitted.redemption_id,
        expected_states=("submitted",),
        state="confirmed",
        observed_at_ms=NOW_MS + 6,
        payout_quote=Decimal("2"),
        payout_proof_sha256=PAYOUT_PROOF,
    )

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM polymarket_live_redemptions").fetchone()
        assert row is not None
        payload = PolymarketLiveOrderLedger._redemption_row_payload_v2(row)
        payload_json = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            """
            CREATE TABLE polymarket_live_redemptions_v2 (
                redemption_id TEXT PRIMARY KEY,
                condition_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                inventory_json TEXT NOT NULL,
                preflight_json TEXT NOT NULL,
                state TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                transaction_hash TEXT NOT NULL,
                failure_code TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL,
                UNIQUE (condition_id, attempt),
                CHECK (attempt > 0)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO polymarket_live_redemptions_v2
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                *payload.values(),
                hashlib.sha256(payload_json.encode("ascii")).hexdigest(),
            ],
        )
        connection.execute("DROP TABLE polymarket_live_redemptions")
        connection.execute(
            "ALTER TABLE polymarket_live_redemptions_v2 "
            "RENAME TO polymarket_live_redemptions"
        )
        connection.execute(
            """
            UPDATE polymarket_live_metadata SET value = 'polymarket-live-ledger-v2'
            WHERE key = 'schema_version'
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = PolymarketLiveOrderLedger(path)
    record = migrated.redemption_records()[0]
    assert record.state == "confirmed"
    assert record.payout_quote == 0
    assert record.payout_proof_sha256 == ""
    assert record.payout_accounting_state == "UNKNOWN"
    assert migrated.unverified_redemption_accounting_count() == 1
    venue = FakeRedemptionVenue()
    venue.recovery = PolymarketRedemptionRecovery(
        state="failed",
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
    )
    recovered = PolymarketRedemptionCoordinator(
        FakeAccount(),
        venue,
        migrated,
        clock_ms=lambda: NOW_MS + 7,
    ).recover_incomplete()
    assert recovered[0].state == "confirmed"
    assert recovered[0].payout_accounting_state == "UNKNOWN"

    upgraded = migrated.transition_redemption(
        record.redemption_id,
        expected_states=("confirmed",),
        state="confirmed",
        observed_at_ms=NOW_MS + 8,
        payout_quote=Decimal("2"),
        payout_proof_sha256=PAYOUT_PROOF,
    )
    assert upgraded.payout_accounting_state == "VERIFIED"
    assert migrated.unverified_redemption_accounting_count() == 0


@pytest.mark.parametrize(
    "inventory",
    [
        (),
        (_owned_inventory(provisional=True),),
        (_owned_inventory(market_id=OTHER_CONDITION_ID),),
        (_owned_inventory(), _owned_inventory()),
    ],
)
def test_redemption_reservation_rejects_invalid_inventory(
    tmp_path: Path,
    inventory: tuple[PolymarketOwnedInventory, ...],
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "invalid-inventory.sqlite3")

    with pytest.raises(ValueError):
        ledger.reserve_redemption(
            CONDITION_ID,
            inventory,
            observed_at_ms=NOW_MS,
        )


def test_redemption_reservation_is_idempotent_but_economics_are_immutable(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "idempotent.sqlite3")
    first = ledger.reserve_redemption(
        CONDITION_ID,
        (_owned_inventory(),),
        observed_at_ms=NOW_MS,
    )

    second = ledger.reserve_redemption(
        CONDITION_ID,
        (_owned_inventory(),),
        observed_at_ms=NOW_MS + 1,
    )
    with pytest.raises(PolymarketLiveBlocked, match="different"):
        ledger.reserve_redemption(
            CONDITION_ID,
            (_owned_inventory(quantity="3"),),
            observed_at_ms=NOW_MS + 2,
        )

    assert second == first


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "JSON is invalid"),
        ("[ ]", "not canonical"),
        ("[1]", "row is invalid"),
        ("[]", "token set is invalid"),
        (
            json.dumps(
                [
                    {
                        "market_id": OTHER_CONDITION_ID,
                        "token_id": TOKEN_ID,
                        "quantity": "2",
                    }
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "condition differs",
        ),
        (
            json.dumps(
                [
                    {
                        "market_id": CONDITION_ID,
                        "token_id": TOKEN_ID,
                        "quantity": "2",
                    },
                    {
                        "market_id": CONDITION_ID,
                        "token_id": TOKEN_ID,
                        "quantity": "1",
                    },
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "token set is invalid",
        ),
    ],
)
def test_redemption_inventory_parser_fails_closed(
    payload: str,
    message: str,
) -> None:
    with pytest.raises(PolymarketLiveError, match=message):
        PolymarketLiveOrderLedger._parse_redemption_inventory(
            CONDITION_ID,
            payload,
        )


def _reserved_redemption(
    tmp_path: Path,
    name: str,
) -> tuple[PolymarketLiveOrderLedger, PolymarketRedemptionRecord]:
    ledger = PolymarketLiveOrderLedger(tmp_path / f"{name}.sqlite3")
    record = ledger.reserve_redemption(
        CONDITION_ID,
        (_owned_inventory(),),
        observed_at_ms=NOW_MS,
    )
    return ledger, record


def test_redemption_transition_validation_is_monotonic(tmp_path: Path) -> None:
    ledger, record = _reserved_redemption(tmp_path, "transition-validation")
    redemption_id = record.redemption_id

    with pytest.raises(ValueError, match="expected"):
        ledger.transition_redemption(
            redemption_id,
            expected_states=(),
            state="submitting",
            observed_at_ms=NOW_MS,
        )
    with pytest.raises(ValueError, match="state"):
        ledger.transition_redemption(
            redemption_id,
            expected_states=("prepared",),
            state="invalid",
            observed_at_ms=NOW_MS,
        )
    with pytest.raises(ValueError, match="positive"):
        ledger.transition_redemption(
            redemption_id,
            expected_states=("prepared",),
            state="submitting",
            observed_at_ms=0,
        )
    with pytest.raises(PolymarketStateConflict, match="does not permit"):
        ledger.transition_redemption(
            redemption_id,
            expected_states=("submitted",),
            state="confirmed",
            observed_at_ms=NOW_MS,
            transaction_hash=TRANSACTION_HASH,
        )
    with pytest.raises(PolymarketStateConflict, match="invalid"):
        ledger.transition_redemption(
            redemption_id,
            expected_states=("prepared",),
            state="confirmed",
            observed_at_ms=NOW_MS,
            transaction_hash=TRANSACTION_HASH,
        )
    with pytest.raises(KeyError):
        ledger.transition_redemption(
            "redemption:" + "9" * 64 + ":000001",
            expected_states=("prepared",),
            state="submitting",
            observed_at_ms=NOW_MS,
        )
    with pytest.raises(ValueError, match="positive"):
        ledger.reserve_redemption(
            CONDITION_ID,
            (_owned_inventory(),),
            observed_at_ms=0,
        )

    submitting = ledger.transition_redemption(
        redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 10,
    )
    with pytest.raises(ValueError, match="backwards"):
        ledger.transition_redemption(
            submitting.redemption_id,
            expected_states=("submitting",),
            state="unknown",
            observed_at_ms=NOW_MS + 9,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("attempt", 2, "attempt identity"),
        ("state", "invalid", "state is invalid"),
        ("updated_at_ms", 0, "chronology"),
        ("preflight_json", "{", "preflight JSON is invalid"),
        ("preflight_json", "[]", "preflight JSON is not canonical"),
    ],
)
def test_rehashed_redemption_corruption_still_fails_semantic_validation(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    ledger, record = _reserved_redemption(tmp_path, f"semantic-{column}")
    connection = ledger._connect()
    try:
        connection.execute(
            f"""
            UPDATE polymarket_live_redemptions
            SET {column} = ?
            WHERE redemption_id = ?
            """,
            [value, record.redemption_id],
        )
        ledger._write_redemption_row_hash(connection, record.redemption_id)
    finally:
        connection.close()

    with pytest.raises(PolymarketLiveError, match=message):
        ledger.redemption_records()


def test_redemption_hash_writer_rejects_missing_attempt(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "missing-hash-row.sqlite3")
    connection = ledger._connect()
    try:
        with pytest.raises(KeyError):
            ledger._write_redemption_row_hash(
                connection,
                "redemption:" + "9" * 64 + ":000001",
            )
    finally:
        connection.close()


def test_redemption_transaction_identity_is_immutable(tmp_path: Path) -> None:
    ledger, record = _reserved_redemption(tmp_path, "identity")
    submitting = ledger.transition_redemption(
        record.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 1,
    )
    submitted = ledger.transition_redemption(
        submitting.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 2,
        transaction_id="transaction-0001",
        transaction_hash=TRANSACTION_HASH,
    )

    with pytest.raises(PolymarketLiveBlocked, match="ID cannot change"):
        ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="unknown",
            observed_at_ms=NOW_MS + 3,
            transaction_id="transaction-0002",
        )
    with pytest.raises(PolymarketLiveBlocked, match="hash cannot change"):
        ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="unknown",
            observed_at_ms=NOW_MS + 3,
            transaction_hash=OTHER_TRANSACTION_HASH,
        )


def test_redemption_transaction_identity_cannot_be_reused(
    tmp_path: Path,
) -> None:
    ledger, first = _reserved_redemption(tmp_path, "duplicate-transaction")
    first_submitting = ledger.transition_redemption(
        first.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 1,
    )
    first_submitted = ledger.transition_redemption(
        first_submitting.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 2,
        transaction_hash=TRANSACTION_HASH,
    )
    ledger.transition_redemption(
        first_submitted.redemption_id,
        expected_states=("submitted",),
        state="failed",
        observed_at_ms=NOW_MS + 3,
    )
    second = ledger.reserve_redemption(
        CONDITION_ID,
        (_owned_inventory(),),
        observed_at_ms=NOW_MS + 4,
    )
    second_submitting = ledger.transition_redemption(
        second.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 5,
    )

    with pytest.raises(PolymarketLiveBlocked, match="already used"):
        ledger.transition_redemption(
            second_submitting.redemption_id,
            expected_states=("submitting",),
            state="submitted",
            observed_at_ms=NOW_MS + 6,
            transaction_hash=TRANSACTION_HASH,
        )


def test_redemption_transition_rejects_missing_identity_and_long_failure(
    tmp_path: Path,
) -> None:
    ledger, record = _reserved_redemption(tmp_path, "missing-identity")
    submitting = ledger.transition_redemption(
        record.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 1,
    )

    with pytest.raises(ValueError, match="lacks transaction identity"):
        ledger.transition_redemption(
            submitting.redemption_id,
            expected_states=("submitting",),
            state="submitted",
            observed_at_ms=NOW_MS + 2,
        )
    with pytest.raises(ValueError, match="too long"):
        ledger.transition_redemption(
            submitting.redemption_id,
            expected_states=("submitting",),
            state="failed",
            observed_at_ms=NOW_MS + 2,
            failure_code="x" * 257,
        )


def test_confirmed_redemption_requires_hash_and_cannot_exceed_inventory(
    tmp_path: Path,
) -> None:
    ledger, record = _reserved_redemption(tmp_path, "confirmed-validation")
    submitting = ledger.transition_redemption(
        record.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 1,
    )
    submitted = ledger.transition_redemption(
        submitting.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 2,
        transaction_id="transaction-0001",
    )
    with pytest.raises(ValueError, match="lacks a transaction hash"):
        ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="confirmed",
            observed_at_ms=NOW_MS + 3,
        )
    with pytest.raises(ValueError, match="verified payout accounting"):
        ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="confirmed",
            observed_at_ms=NOW_MS + 3,
            transaction_hash=TRANSACTION_HASH,
        )
    with pytest.raises(PolymarketLiveBlocked, match="exceeds reserved inventory"):
        ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="confirmed",
            observed_at_ms=NOW_MS + 3,
            transaction_hash=TRANSACTION_HASH,
            payout_quote=Decimal("3"),
            payout_proof_sha256=PAYOUT_PROOF,
        )

    inventory_ledger = PolymarketLiveOrderLedger(tmp_path / "excess-inventory.sqlite3")
    _seed_confirmed_inventory(inventory_ledger)
    excess = inventory_ledger.reserve_redemption(
        CONDITION_ID,
        (_owned_inventory(quantity="3"),),
        observed_at_ms=NOW_MS + 3,
    )
    submitting_excess = inventory_ledger.transition_redemption(
        excess.redemption_id,
        expected_states=("prepared",),
        state="submitting",
        observed_at_ms=NOW_MS + 4,
    )
    submitted_excess = inventory_ledger.transition_redemption(
        submitting_excess.redemption_id,
        expected_states=("submitting",),
        state="submitted",
        observed_at_ms=NOW_MS + 5,
        transaction_hash=TRANSACTION_HASH,
    )
    inventory_ledger.transition_redemption(
        submitted_excess.redemption_id,
        expected_states=("submitted",),
        state="confirmed",
        observed_at_ms=NOW_MS + 6,
        payout_quote=Decimal("0"),
        payout_proof_sha256=PAYOUT_PROOF,
    )

    with pytest.raises(PolymarketLiveError, match="exceeds owned inventory"):
        inventory_ledger.owned_inventory()


class FakeHandle:
    def __init__(
        self,
        *,
        transaction_id: str | None = None,
        transaction_hash: str = TRANSACTION_HASH,
        wait_error: Exception | None = None,
        outcome_transaction_id: str | None = None,
        outcome_transaction_hash: str | None = None,
    ) -> None:
        self.transaction_id = transaction_id
        self.transaction_hash = transaction_hash
        self.wait_error = wait_error
        self.outcome_transaction_id = outcome_transaction_id
        self.outcome_transaction_hash = outcome_transaction_hash

    def wait(self) -> object:
        if self.wait_error is not None:
            raise self.wait_error
        return type(
            "Outcome",
            (),
            {
                "transaction_id": self.outcome_transaction_id,
                "transaction_hash": (
                    self.outcome_transaction_hash
                    if self.outcome_transaction_hash is not None
                    else self.transaction_hash
                ),
            },
        )()


class FakeUnifiedClient:
    wallet = WALLET

    def __init__(
        self,
        *,
        wallet_type: str = "EOA",
        rpc: object | None = None,
    ) -> None:
        self.wallet_type = wallet_type
        self.conditions: list[str] = []
        self.closed = False
        self.redeem_error: Exception | None = None
        self.handle = FakeHandle()
        self.context_condition_id = CONDITION_ID
        self.context_adapter = ADAPTER
        self.context_tokens = CONDITIONAL_TOKENS
        self._ctx = type(
            "Context",
            (),
            {
                "rpc": rpc or FakeOnchainRpc(),
                "environment": type(
                    "Environment",
                    (),
                    {
                        "collateral_token": COLLATERAL_TOKEN,
                        "conditional_tokens": CONDITIONAL_TOKENS,
                        "neg_risk_adapter": NEG_RISK_ADAPTER,
                        "relayer_poll_frequency_ms": 2_000,
                        "relayer_max_polls": 100,
                    },
                )(),
                "relayer": object(),
            },
        )()

    def _resolve_market_position_context(
        self,
        *,
        condition_id: str,
        closed: bool,
    ) -> object:
        assert closed is True
        return type(
            "PositionContext",
            (),
            {
                "condition_id": self.context_condition_id,
                "adapter_address": self.context_adapter,
                "position_erc1155_address": self.context_tokens,
                "neg_risk": False,
            },
        )()

    def redeem_positions(self, *, condition_id: str, metadata: str) -> FakeHandle:
        assert metadata
        self.conditions.append(condition_id)
        if self.redeem_error is not None:
            raise self.redeem_error
        return self.handle

    def close(self) -> None:
        self.closed = True


class FakeRpcResponse:
    def __init__(self, payload: object) -> None:
        self.content = json.dumps(payload).encode("utf-8")
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class OversizedRpcResponse(FakeRpcResponse):
    def __init__(self) -> None:
        super().__init__({})
        self.content = b"x" * (1024 * 1024 + 1)


class FakeRpcSession:
    def __init__(
        self,
        status: str,
        *,
        balance_result: object = hex(1_000_000_000),
        receipt_result: object | None = None,
        finalized_block: int = 0x20,
        canonical_block_hash: str = BLOCK_HASH,
    ) -> None:
        self.status = status
        self.balance_result = balance_result
        self.receipt_result = receipt_result
        self.finalized_block = finalized_block
        self.canonical_block_hash = canonical_block_hash
        self.calls: list[object] = []
        self.closed = False

    def post(
        self,
        url: str,
        *,
        json: object,
        timeout: float,
    ) -> FakeRpcResponse:
        del url, timeout
        self.calls.append(json)
        assert isinstance(json, dict)
        method = json["method"]
        if method == "eth_getBalance":
            result: object = self.balance_result
        elif method == "eth_getBlockByNumber":
            assert isinstance(json["params"], list)
            result = (
                {"number": hex(self.finalized_block)}
                if json["params"][0] == "finalized"
                else {
                    "number": "0x10",
                    "hash": self.canonical_block_hash,
                }
            )
        elif self.receipt_result is not None:
            result = self.receipt_result
        else:
            result = _standard_redemption_receipt(status=self.status)
        return FakeRpcResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": result,
            }
        )

    def close(self) -> None:
        self.closed = True


class StaticRpcSession(FakeRpcSession):
    def __init__(self, result: object) -> None:
        super().__init__("0x1")
        self.result = result

    def post(
        self,
        url: str,
        *,
        json: object,
        timeout: float,
    ) -> FakeRpcResponse:
        del url, timeout
        assert isinstance(json, dict)
        if json.get("method") == "eth_getBlockByNumber":
            assert isinstance(json["params"], list)
            result = (
                {"number": "0x20"}
                if json["params"][0] == "finalized"
                else {"number": "0x10", "hash": BLOCK_HASH}
            )
        else:
            result = self.result
        return FakeRpcResponse({"jsonrpc": "2.0", "id": 1, "result": result})


class FakeOnchainRpc:
    def __init__(
        self,
        *,
        approval: str = "0x" + "0" * 63 + "1",
        gas_estimate: int = 100_000,
        gas_price: int = 1_000,
    ) -> None:
        self.approval = approval
        self.gas_estimate = gas_estimate
        self.gas_price = gas_price

    def eth_call(self, *, to: str, data: str) -> str:
        assert to == CONDITIONAL_TOKENS
        assert data.startswith("0xe985e9c5")
        return self.approval

    def eth_estimate_gas(self, payload: object) -> int:
        assert payload
        return self.gas_estimate

    def eth_gas_price(self) -> int:
        return self.gas_price


def _credentials(*, signature_type: int = 0) -> PolymarketLiveCredentials:
    return PolymarketLiveCredentials(
        private_key="0x" + "7" * 64,
        api_key="offline-api-key",
        api_secret="offline-api-secret",
        api_passphrase="offline-passphrase",
        funder_address=WALLET,
        signature_type=signature_type,
    )


@pytest.mark.parametrize(
    "response",
    [
        OversizedRpcResponse(),
        FakeRpcResponse([]),
        FakeRpcResponse({"error": {"code": -1}}),
    ],
)
def test_polygon_rpc_response_parser_fails_closed(response: object) -> None:
    with pytest.raises(PolymarketLiveUnknownState):
        settlement_module._response_json(response)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("neg_risk", "receipt"),
    [
        (False, _standard_redemption_receipt()),
        (True, _negative_risk_redemption_receipt()),
    ],
)
def test_redemption_receipt_proof_binds_condition_wallet_and_payout(
    neg_risk: bool,
    receipt: dict[str, object],
) -> None:
    payout, proof = settlement_module._redemption_payout_proof(
        receipt,
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        wallet_address=WALLET,
        collateral_token=COLLATERAL_TOKEN,
        conditional_tokens=CONDITIONAL_TOKENS,
        neg_risk_adapter=NEG_RISK_ADAPTER,
        neg_risk=neg_risk,
    )

    assert payout == Decimal("2")
    assert len(proof) == 64
    assert (
        proof
        == settlement_module._redemption_payout_proof(
            receipt,
            transaction_hash=TRANSACTION_HASH,
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            wallet_address=WALLET,
            collateral_token=COLLATERAL_TOKEN,
            conditional_tokens=CONDITIONAL_TOKENS,
            neg_risk_adapter=NEG_RISK_ADAPTER,
            neg_risk=neg_risk,
        )[1]
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda receipt: receipt["logs"].pop(0),
        lambda receipt: receipt["logs"].append(dict(receipt["logs"][0])),
        lambda receipt: receipt["logs"][1].update({"removed": True}),
        lambda receipt: receipt["logs"][1].update({"data": _words(1)}),
        lambda receipt: receipt["logs"][2].update(
            {
                "topics": [
                    settlement_module._TRANSFER_EVENT_TOPIC,
                    "0x" + "0" * 64,
                    _topic_address("0x" + "9" * 40),
                ]
            }
        ),
    ],
)
def test_redemption_receipt_proof_rejects_missing_or_contradictory_events(
    mutator: Callable[[dict[str, object]], object],
) -> None:
    receipt = _standard_redemption_receipt()
    mutator(receipt)

    with pytest.raises(PolymarketLiveUnknownState):
        settlement_module._redemption_payout_proof(
            receipt,
            transaction_hash=TRANSACTION_HASH,
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            wallet_address=WALLET,
            collateral_token=COLLATERAL_TOKEN,
            conditional_tokens=CONDITIONAL_TOKENS,
            neg_risk_adapter=NEG_RISK_ADAPTER,
            neg_risk=False,
        )


def test_official_adapter_validates_transport_and_wallet_identity() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(),
            client=FakeUnifiedClient(),
            rpc_url="http://polygon.invalid",
        )
    with pytest.raises(ValueError, match="multiplier"):
        OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(),
            client=FakeUnifiedClient(),
            gas_reserve_multiplier=0,
        )
    with pytest.raises(ValueError, match="TTL"):
        OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(),
            client=FakeUnifiedClient(),
            preflight_ttl_ms=100,
        )
    wrong_wallet = FakeUnifiedClient()
    wrong_wallet.wallet = "0x" + "e" * 40
    with pytest.raises(PolymarketLiveBlocked, match="wallet identity"):
        OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(),
            client=wrong_wallet,
        )


def _uninitialized_official_venue(
    credentials: PolymarketLiveCredentials,
    gasless_credentials: PolymarketGaslessCredentials | None = None,
) -> OfficialPolymarketUnifiedRedemptionVenue:
    venue = object.__new__(OfficialPolymarketUnifiedRedemptionVenue)
    venue.credentials = credentials
    venue.gasless_credentials = gasless_credentials
    return venue


def test_official_client_builder_pins_version_and_skips_wallet_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    from polymarket import SecureClient

    venue = _uninitialized_official_venue(_credentials())
    monkeypatch.setattr(
        settlement_module,
        "package_version",
        lambda name: "unexpected",
    )
    with pytest.raises(RuntimeError, match="audited pin"):
        venue._build_client()

    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(settlement_module, "package_version", missing)
    with pytest.raises(RuntimeError, match="extra"):
        venue._build_client()

    monkeypatch.setattr(
        settlement_module,
        "package_version",
        lambda name: settlement_module.POLYMARKET_UNIFIED_SDK_VERSION,
    )
    monkeypatch.setattr(SecureClient, "_create", None)
    with pytest.raises(RuntimeError, match="construction path"):
        venue._build_client()

    client = FakeUnifiedClient()
    captured: list[object] = []

    def create(**kwargs: object) -> FakeUnifiedClient:
        captured.append(kwargs)
        return client

    monkeypatch.setattr(SecureClient, "_create", create)
    assert venue._build_client() is client
    assert len(captured) == 1


@pytest.mark.parametrize("deployed", [False, True])
def test_official_smart_wallet_builder_checks_deployment_without_creating_it(
    monkeypatch: pytest.MonkeyPatch,
    deployed: bool,
) -> None:
    from polymarket import SecureClient
    from polymarket._internal.actions.relayer import deployed as deployed_module

    client = FakeUnifiedClient(wallet_type="DEPOSIT_WALLET")
    gasless = PolymarketGaslessCredentials(
        kind="relayer",
        key="relayer-api-key",
        address=WALLET,
    )
    venue = _uninitialized_official_venue(
        _credentials(signature_type=3),
        gasless,
    )
    monkeypatch.setattr(
        settlement_module,
        "package_version",
        lambda name: settlement_module.POLYMARKET_UNIFIED_SDK_VERSION,
    )
    monkeypatch.setattr(SecureClient, "_create", lambda **kwargs: client)
    observed: list[object] = []

    def fetch(
        relayer: object,
        *,
        address: str,
        type: object,
    ) -> bool:
        observed.append((relayer, address, type))
        return deployed

    monkeypatch.setattr(deployed_module, "fetch_deployed_sync", fetch)
    if deployed:
        assert venue._build_client() is client
    else:
        with pytest.raises(PolymarketLiveBlocked, match="not deployed"):
            venue._build_client()
        assert client.closed is True
    assert len(observed) == 1


def test_official_smart_wallet_builder_closes_after_deployment_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polymarket import SecureClient
    from polymarket._internal.actions.relayer import deployed as deployed_module

    client = FakeUnifiedClient(wallet_type="DEPOSIT_WALLET")
    venue = _uninitialized_official_venue(
        _credentials(signature_type=3),
        PolymarketGaslessCredentials(
            kind="relayer",
            key="relayer-api-key",
            address=WALLET,
        ),
    )
    monkeypatch.setattr(
        settlement_module,
        "package_version",
        lambda name: settlement_module.POLYMARKET_UNIFIED_SDK_VERSION,
    )
    monkeypatch.setattr(SecureClient, "_create", lambda **kwargs: client)

    def fail(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise TimeoutError("deployment query failed")

    monkeypatch.setattr(deployed_module, "fetch_deployed_sync", fail)

    with pytest.raises(TimeoutError):
        venue._build_client()

    assert client.closed is True


def test_official_preflight_rejects_unproved_market_context() -> None:
    incomplete = type(
        "IncompleteClient",
        (),
        {"wallet": WALLET, "wallet_type": "EOA"},
    )()
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=incomplete,
    )
    with pytest.raises(RuntimeError, match="internals"):
        venue.assert_redemption_ready(CONDITION_ID)

    wrong_condition = FakeUnifiedClient()
    wrong_condition.context_condition_id = OTHER_CONDITION_ID
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=wrong_condition,
    )
    with pytest.raises(PolymarketLiveBlocked, match="condition differs"):
        venue.assert_redemption_ready(CONDITION_ID)

    invalid_context = FakeUnifiedClient()
    invalid_context.context_adapter = "invalid"
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=invalid_context,
    )
    with pytest.raises(RuntimeError, match="market context"):
        venue.assert_redemption_ready(CONDITION_ID)


def test_official_preflight_requires_approval_and_gas_reserve() -> None:
    no_approval = FakeUnifiedClient(rpc=FakeOnchainRpc(approval="0x" + "0" * 64))
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=no_approval,
    )
    with pytest.raises(PolymarketLiveBlocked, match="approval"):
        venue.assert_redemption_ready(CONDITION_ID)

    low_balance = FakeRpcSession("0x1", balance_result=hex(1))
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=low_balance,  # type: ignore[arg-type]
    )
    with pytest.raises(PolymarketLiveBlocked, match="gas reserve"):
        venue.assert_redemption_ready(CONDITION_ID)

    invalid_balance = FakeRpcSession("0x1", balance_result="invalid")
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=invalid_balance,  # type: ignore[arg-type]
    )
    with pytest.raises(PolymarketLiveBlocked, match="native balance"):
        venue.assert_redemption_ready(CONDITION_ID)


def test_official_eoa_adapter_preserves_identity_and_verifies_receipt() -> None:
    client = FakeUnifiedClient()
    session = FakeRpcSession("0x1")
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=client,
        session=session,  # type: ignore[arg-type]
    )

    preflight = venue.assert_redemption_ready(CONDITION_ID)
    submission = venue.submit_redemption(CONDITION_ID)
    outcome = venue.wait_redemption(submission)
    recovered = venue.recover_redemption(
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
    )

    assert client.conditions == [CONDITION_ID]
    assert preflight.required_native_balance_wei == 200_000_000
    assert outcome.transaction_hash == TRANSACTION_HASH
    assert recovered.state == "confirmed"
    assert len(session.calls) == 7


def test_redemption_recovery_waits_for_finality_and_rejects_orphan_receipt() -> None:
    pending = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=FakeRpcSession(
            "0x1",
            finalized_block=0x0F,
        ),  # type: ignore[arg-type]
    ).recover_redemption(
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
    )
    assert pending.state == "pending"

    orphaned = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=FakeRpcSession(
            "0x1",
            canonical_block_hash=OTHER_TRANSACTION_HASH,
        ),  # type: ignore[arg-type]
    )
    with pytest.raises(PolymarketLiveUnknownState, match="canonical finalized"):
        orphaned.recover_redemption(
            transaction_id="",
            transaction_hash=TRANSACTION_HASH,
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
        )


def test_official_submit_and_wait_classify_only_proven_failures() -> None:
    from polymarket.errors import TransactionFailedError, UserInputError

    client = FakeUnifiedClient()
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=client,
    )
    client.redeem_error = UserInputError("market is not resolved")
    venue.assert_redemption_ready(CONDITION_ID)
    with pytest.raises(PolymarketRedemptionRejected):
        venue.submit_redemption(CONDITION_ID)

    client.redeem_error = TimeoutError("response lost")
    venue.assert_redemption_ready(CONDITION_ID)
    with pytest.raises(TimeoutError):
        venue.submit_redemption(CONDITION_ID)

    client.redeem_error = None
    failed_handle = FakeHandle(
        wait_error=TransactionFailedError("transaction reverted")
    )
    failed = PolymarketRedemptionSubmission(
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
        _handle=failed_handle,
    )
    with pytest.raises(PolymarketRedemptionFailed):
        venue.wait_redemption(failed)

    ambiguous = PolymarketRedemptionSubmission(
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
        _handle=FakeHandle(wait_error=TimeoutError("receipt unavailable")),
    )
    with pytest.raises(TimeoutError):
        venue.wait_redemption(ambiguous)


def test_official_submit_requires_fresh_single_use_preflight() -> None:
    clock = [0]
    client = FakeUnifiedClient()
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=client,
        preflight_ttl_ms=500,
        monotonic_ns=lambda: clock[0],
    )

    with pytest.raises(PolymarketRedemptionRejected, match="preflight"):
        venue.submit_redemption(CONDITION_ID)

    venue.assert_redemption_ready(CONDITION_ID)
    clock[0] = 501_000_000
    with pytest.raises(PolymarketRedemptionRejected, match="preflight"):
        venue.submit_redemption(CONDITION_ID)

    venue.assert_redemption_ready(CONDITION_ID)
    venue.submit_redemption(CONDITION_ID)
    with pytest.raises(PolymarketRedemptionRejected, match="preflight"):
        venue.submit_redemption(CONDITION_ID)

    assert client.conditions == [CONDITION_ID]


@pytest.mark.parametrize("mismatch", ["id", "hash"])
def test_official_wait_rejects_transaction_identity_drift(
    mismatch: str,
) -> None:
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
    )
    handle = FakeHandle(
        transaction_id="transaction-0001",
        outcome_transaction_id=(
            "transaction-0002" if mismatch == "id" else "transaction-0001"
        ),
        outcome_transaction_hash=(
            OTHER_TRANSACTION_HASH if mismatch == "hash" else TRANSACTION_HASH
        ),
    )
    submission = PolymarketRedemptionSubmission(
        transaction_id="transaction-0001",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
        _handle=handle,
    )

    with pytest.raises(PolymarketLiveUnknownState, match="differs"):
        venue.wait_redemption(submission)


def test_official_adapter_requires_explicit_smart_wallet_credentials() -> None:
    with pytest.raises(PolymarketLiveBlocked, match="relayer credentials"):
        OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(signature_type=2),
            client=FakeUnifiedClient(),
            session=FakeRpcSession("0x1"),  # type: ignore[arg-type]
        )


def test_smart_wallet_submission_bypasses_sdk_retry_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeUnifiedClient(wallet_type="DEPOSIT_WALLET")
    submitted_calls: list[tuple[object, ...]] = []

    def observe_submit(
        context: object,
        *,
        calls: list[object],
        metadata: str,
    ) -> object:
        assert context is client._ctx
        assert len(calls) == 1
        assert metadata
        submitted_calls.append(tuple(calls))
        return type(
            "Response",
            (),
            {
                "transaction_id": "transaction-0001",
                "transaction_hash": None,
            },
        )()

    from polymarket._internal.actions.relayer import gasless

    monkeypatch.setattr(gasless, "_submit_for_wallet_type_sync", observe_submit)
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(signature_type=3),
        gasless_credentials=PolymarketGaslessCredentials(
            kind="relayer",
            key="relayer-api-key",
            address=WALLET,
        ),
        client=client,
        session=FakeRpcSession("0x1"),  # type: ignore[arg-type]
    )

    preflight = venue.assert_redemption_ready(CONDITION_ID)
    submission = venue.submit_redemption(CONDITION_ID)

    assert preflight.gasless is True
    assert submission.transaction_id == "transaction-0001"
    assert client.conditions == []
    assert len(submitted_calls) == 1


def test_gasless_credentials_are_environment_only_and_redacted() -> None:
    credentials = PolymarketGaslessCredentials.from_environment(
        {
            "SIMPLE_AI_TRADING_POLYMARKET_RELAYER_API_KEY": "relayer-secret-key",
            "SIMPLE_AI_TRADING_POLYMARKET_RELAYER_API_KEY_ADDRESS": WALLET,
        }
    )

    assert credentials.kind == "relayer"
    assert "relayer-secret-key" not in repr(credentials)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "invalid", "key": "valid-api-key"}, "kind"),
        ({"kind": "relayer", "key": "short", "address": WALLET}, "key"),
        (
            {
                "kind": "builder",
                "key": "builder-api-key",
                "secret": "short",
                "passphrase": "builder-passphrase",
            },
            "builder",
        ),
        (
            {
                "kind": "builder",
                "key": "builder-api-key",
                "secret": "builder-secret",
                "passphrase": "builder-passphrase",
                "address": WALLET,
            },
            "address",
        ),
        (
            {
                "kind": "relayer",
                "key": "relayer-api-key",
                "secret": "unexpected-secret",
                "address": WALLET,
            },
            "shared secrets",
        ),
        (
            {
                "kind": "relayer",
                "key": "relayer-api-key",
                "address": "invalid",
            },
            "address",
        ),
    ],
)
def test_gasless_credentials_validate_complete_shapes(
    kwargs: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PolymarketGaslessCredentials(**kwargs)


def test_gasless_environment_selects_one_complete_credential_type() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PolymarketGaslessCredentials.from_environment({})
    with pytest.raises(ValueError, match="exactly one"):
        PolymarketGaslessCredentials.from_environment(
            {
                "SIMPLE_AI_TRADING_POLYMARKET_RELAYER_API_KEY": "relayer-key",
                "SIMPLE_AI_TRADING_POLYMARKET_BUILDER_API_KEY": "builder-key",
            }
        )
    builder = PolymarketGaslessCredentials.from_environment(
        {
            "SIMPLE_AI_TRADING_POLYMARKET_BUILDER_API_KEY": "builder-api-key",
            "SIMPLE_AI_TRADING_POLYMARKET_BUILDER_SECRET": "builder-secret",
            "SIMPLE_AI_TRADING_POLYMARKET_BUILDER_PASSPHRASE": ("builder-passphrase"),
        }
    )

    assert builder.kind == "builder"
    assert builder.to_sdk_api_key().__class__.__name__ == "BuilderApiKey"
    relayer = PolymarketGaslessCredentials(
        kind="relayer",
        key="relayer-api-key",
        address=WALLET,
    )
    assert relayer.to_sdk_api_key().__class__.__name__ == "RelayerApiKey"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: PolymarketRedemptionPreflight(
                condition_id="invalid",
                adapter_address=ADAPTER,
                neg_risk=False,
                gas_estimate=1,
                gas_price_wei=1,
                native_balance_wei=1,
                required_native_balance_wei=1,
            ),
            "condition",
        ),
        (
            lambda: PolymarketRedemptionPreflight(
                condition_id=CONDITION_ID,
                adapter_address="invalid",
                neg_risk=False,
                gas_estimate=1,
                gas_price_wei=1,
                native_balance_wei=1,
                required_native_balance_wei=1,
            ),
            "adapter",
        ),
        (
            lambda: PolymarketRedemptionPreflight(
                condition_id=CONDITION_ID,
                adapter_address=ADAPTER,
                neg_risk=False,
                gas_estimate=1,
                gas_price_wei=1,
                native_balance_wei=1,
                required_native_balance_wei=1,
                gasless=True,
            ),
            "gasless",
        ),
        (
            lambda: PolymarketRedemptionPreflight(
                condition_id=CONDITION_ID,
                adapter_address=ADAPTER,
                neg_risk=False,
                gas_estimate=0,
                gas_price_wei=1,
                native_balance_wei=1,
                required_native_balance_wei=1,
            ),
            "gas economics",
        ),
        (
            lambda: PolymarketRedemptionPreflight(
                condition_id=CONDITION_ID,
                adapter_address=ADAPTER,
                neg_risk=False,
                gas_estimate=2,
                gas_price_wei=2,
                native_balance_wei=1,
                required_native_balance_wei=3,
            ),
            "native balance",
        ),
    ],
)
def test_redemption_preflight_value_contracts(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PolymarketRedemptionSubmission(
            transaction_id="bad",
            transaction_hash="",
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
            _handle=object(),
        ),
        lambda: PolymarketRedemptionSubmission(
            transaction_id="",
            transaction_hash="bad",
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
            _handle=object(),
        ),
        lambda: PolymarketRedemptionSubmission(
            transaction_id="",
            transaction_hash="",
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
            _handle=object(),
        ),
        lambda: PolymarketRedemptionOutcome(
            transaction_id="bad",
            transaction_hash=TRANSACTION_HASH,
            payout_quote=Decimal("1"),
            payout_proof_sha256=PAYOUT_PROOF,
        ),
        lambda: PolymarketRedemptionOutcome(
            transaction_id="",
            transaction_hash="bad",
            payout_quote=Decimal("1"),
            payout_proof_sha256=PAYOUT_PROOF,
        ),
        lambda: PolymarketRedemptionOutcome(
            transaction_id="",
            transaction_hash=TRANSACTION_HASH,
            payout_quote=Decimal("NaN"),
            payout_proof_sha256=PAYOUT_PROOF,
        ),
        lambda: PolymarketRedemptionOutcome(
            transaction_id="",
            transaction_hash=TRANSACTION_HASH,
            payout_quote=Decimal("1"),
            payout_proof_sha256="",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="invalid",
            transaction_id="",
            transaction_hash="",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="pending",
            transaction_id="bad",
            transaction_hash="",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="pending",
            transaction_id="",
            transaction_hash="bad",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="confirmed",
            transaction_id="transaction-0001",
            transaction_hash="",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="confirmed",
            transaction_id="transaction-0001",
            transaction_hash=TRANSACTION_HASH,
            payout_quote=Decimal("1"),
            payout_proof_sha256="",
        ),
        lambda: PolymarketRedemptionRecovery(
            state="failed",
            transaction_id="",
            transaction_hash="",
        ),
    ],
)
def test_redemption_transaction_value_contracts_fail_closed(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_smart_wallet_recovery_uses_terminal_relayer_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polymarket._internal.actions.relayer import poll
    from polymarket.models.clob.relayer import (
        GaslessTransaction,
        RelayerTransactionState,
    )

    def fetch(
        relayer: object,
        *,
        transaction_id: str,
    ) -> GaslessTransaction:
        assert relayer is client._ctx.relayer
        return GaslessTransaction(
            state=RelayerTransactionState.CONFIRMED,
            transaction_hash=TRANSACTION_HASH,
            transaction_id=transaction_id,
        )

    client = FakeUnifiedClient(wallet_type="DEPOSIT_WALLET")
    monkeypatch.setattr(poll, "fetch_gasless_transaction_sync", fetch)
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(signature_type=3),
        gasless_credentials=PolymarketGaslessCredentials(
            kind="relayer",
            key="relayer-api-key",
            address=WALLET,
        ),
        client=client,
        session=FakeRpcSession("0x1"),  # type: ignore[arg-type]
    )

    recovered = venue.recover_redemption(
        transaction_id="transaction-0001",
        transaction_hash="",
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
    )

    assert recovered.state == "confirmed"
    assert recovered.transaction_hash == TRANSACTION_HASH


@pytest.mark.parametrize(
    ("state_name", "expected"),
    [
        ("NEW", "pending"),
        ("FAILED", "failed"),
    ],
)
def test_smart_wallet_recovery_preserves_pending_and_failure_proof(
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
    expected: str,
) -> None:
    from polymarket._internal.actions.relayer import poll
    from polymarket.models.clob.relayer import (
        GaslessTransaction,
        RelayerTransactionState,
    )

    client = FakeUnifiedClient(wallet_type="DEPOSIT_WALLET")

    def fetch(
        relayer: object,
        *,
        transaction_id: str,
    ) -> GaslessTransaction:
        del relayer
        return GaslessTransaction(
            state=getattr(RelayerTransactionState, state_name),
            transaction_hash=(TRANSACTION_HASH if state_name == "FAILED" else None),
            transaction_id=transaction_id,
            error_msg="proven failure" if state_name == "FAILED" else None,
        )

    monkeypatch.setattr(poll, "fetch_gasless_transaction_sync", fetch)
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(signature_type=3),
        gasless_credentials=PolymarketGaslessCredentials(
            kind="relayer",
            key="relayer-api-key",
            address=WALLET,
        ),
        client=client,
    )

    recovered = venue.recover_redemption(
        transaction_id="transaction-0001",
        transaction_hash=(TRANSACTION_HASH if state_name == "FAILED" else ""),
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
    )

    assert recovered.state == expected


def test_eoa_recovery_rejects_missing_or_contradictory_receipts() -> None:
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=StaticRpcSession(None),  # type: ignore[arg-type]
    )
    pending = venue.recover_redemption(
        transaction_id="",
        transaction_hash=TRANSACTION_HASH,
        condition_id=CONDITION_ID,
        adapter_address=ADAPTER,
        neg_risk=False,
    )
    assert pending.state == "pending"

    with pytest.raises(PolymarketLiveUnknownState, match="requires"):
        venue.recover_redemption(
            transaction_id="",
            transaction_hash="",
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
        )

    for result, message in (
        ([], "receipt is invalid"),
        (
            {
                "transactionHash": OTHER_TRANSACTION_HASH,
                "status": "0x1",
            },
            "hash differs",
        ),
        (
            {
                "transactionHash": TRANSACTION_HASH,
                "status": "0x2",
            },
            "unrecognized",
        ),
    ):
        invalid = OfficialPolymarketUnifiedRedemptionVenue(
            _credentials(),
            client=FakeUnifiedClient(),
            session=StaticRpcSession(result),  # type: ignore[arg-type]
        )
        with pytest.raises(PolymarketLiveUnknownState, match=message):
            invalid.recover_redemption(
                transaction_id="",
                transaction_hash=TRANSACTION_HASH,
                condition_id=CONDITION_ID,
                adapter_address=ADAPTER,
                neg_risk=False,
            )

    failed = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=FakeUnifiedClient(),
        session=StaticRpcSession(
            {
                "transactionHash": TRANSACTION_HASH,
                "blockHash": BLOCK_HASH,
                "blockNumber": "0x10",
                "status": "0x0",
                "logs": [],
            }
        ),  # type: ignore[arg-type]
    )
    assert (
        failed.recover_redemption(
            transaction_id="",
            transaction_hash=TRANSACTION_HASH,
            condition_id=CONDITION_ID,
            adapter_address=ADAPTER,
            neg_risk=False,
        ).state
        == "failed"
    )


def test_official_adapter_closes_client_and_transport() -> None:
    client = FakeUnifiedClient()
    session = FakeRpcSession("0x1")
    venue = OfficialPolymarketUnifiedRedemptionVenue(
        _credentials(),
        client=client,
        session=session,  # type: ignore[arg-type]
    )

    venue.close()

    assert client.closed is True
    assert session.closed is True


class FakeRuntimeAuthority:
    def __init__(self) -> None:
        self.assertions: list[bool] = []
        self.failures: list[str] = []

    def reconciliation_checkpoint(self) -> int:
        return 0

    def note_reconciliation(
        self,
        result: object,
        *,
        checkpoint: int | None = None,
    ) -> None:
        del result
        del checkpoint

    def note_reconciliation_failure(
        self,
        failure_code: str,
        *,
        checkpoint: int | None = None,
    ) -> None:
        del checkpoint
        self.failures.append(failure_code)

    def assert_submission_allowed(self, *, closing_only: bool) -> None:
        self.assertions.append(closing_only)


def test_settlement_service_requires_explicit_enablement(
    tmp_path: Path,
) -> None:
    _, _, venue, coordinator = _coordinator(tmp_path)
    authority = FakeRuntimeAuthority()
    disabled = PolymarketSettlementService(
        coordinator,
        authority,  # type: ignore[arg-type]
    )

    assert asyncio.run(disabled.run_once()) is None
    assert venue.submit_calls == []

    enabled = PolymarketSettlementService(
        coordinator,
        authority,  # type: ignore[arg-type]
        automatic_redemption_enabled=True,
    )
    record = asyncio.run(enabled.run_once())

    assert record is not None and record.state == "confirmed"
    assert authority.assertions == [True]
    assert venue.submit_calls == [CONDITION_ID]


def test_settlement_service_latches_unknown_redemption_outcomes(
    tmp_path: Path,
) -> None:
    _, _, venue, coordinator = _coordinator(tmp_path)
    venue.submit_error = TimeoutError("response lost")
    authority = FakeRuntimeAuthority()
    enabled = PolymarketSettlementService(
        coordinator,
        authority,  # type: ignore[arg-type]
        automatic_redemption_enabled=True,
    )

    with pytest.raises(PolymarketLiveUnknownState):
        asyncio.run(enabled.run_once())
    assert authority.failures == ["unknown_redemption_state"]

    disabled = PolymarketSettlementService(
        coordinator,
        authority,  # type: ignore[arg-type]
    )
    assert asyncio.run(disabled.run_once()) is None
    assert authority.failures == [
        "unknown_redemption_state",
        "unknown_redemption_state",
    ]


def test_settlement_service_supervisor_survives_iteration_failures(
    tmp_path: Path,
) -> None:
    _, _, _, coordinator = _coordinator(tmp_path)
    authority = FakeRuntimeAuthority()
    service = PolymarketSettlementService(
        coordinator,
        authority,  # type: ignore[arg-type]
    )
    service.interval_seconds = 0.01

    async def run_normal() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await asyncio.sleep(0.02)
        stop.set()
        await task

    asyncio.run(run_normal())

    class ExplodingCoordinator:
        def recover_incomplete(self) -> object:
            raise RuntimeError("isolated settlement failure")

    failing = PolymarketSettlementService(
        ExplodingCoordinator(),  # type: ignore[arg-type]
        authority,  # type: ignore[arg-type]
    )
    failing.interval_seconds = 0.01

    async def run_failing() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(failing.run(stop))
        await asyncio.sleep(0.02)
        stop.set()
        await task

    asyncio.run(run_failing())
    assert authority.failures
    assert set(authority.failures) == {"settlement_iteration_failure:RuntimeError"}


@pytest.mark.parametrize(
    "module_name",
    [
        "polymarket_live.py",
        "polymarket_live_runtime.py",
        "polymarket_live_v2.py",
        "polymarket_live_settlement.py",
    ],
)
def test_live_polymarket_boundary_never_imports_binance_execution(
    module_name: str,
) -> None:
    source_path = Path(__file__).parents[1] / "src" / "simple_ai_trading" / module_name
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        part.startswith("binance")
        for module in imported_modules
        for part in module.lower().split(".")
    )
