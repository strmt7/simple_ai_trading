"""Durable, independently owned Polymarket position redemption."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version as package_version
import os
import re
from threading import RLock
import time
from typing import Protocol

import requests

from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
    PolymarketLiveUnknownState,
    PolymarketOwnedInventory,
    PolymarketRedemptionRecord,
    PolymarketRemoteOrder,
    PolymarketRemotePosition,
    PolymarketRuntimeAuthority,
)
from .polymarket_live_v2 import PolymarketLiveCredentials


POLYMARKET_UNIFIED_SDK_VERSION = "0.2.0"
POLYMARKET_POLYGON_RPC_URL = "https://polygon.drpc.org"
_TRANSACTION_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_TRANSACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_POSITION_TOLERANCE = Decimal("0.000001")


class PolymarketRedemptionRejected(PolymarketLiveBlocked):
    """A local or venue proof establishes that no transaction was broadcast."""


class PolymarketRedemptionFailed(PolymarketLiveBlocked):
    """A transaction receipt or relayer result proves terminal failure."""


@dataclass(frozen=True, slots=True)
class PolymarketGaslessCredentials:
    kind: str
    key: str = field(repr=False)
    secret: str = field(default="", repr=False)
    passphrase: str = field(default="", repr=False)
    address: str = ""

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip().lower()
        if kind not in {"builder", "relayer"}:
            raise ValueError("gasless credential kind is invalid")
        key = str(self.key or "").strip()
        secret = str(self.secret or "").strip()
        passphrase = str(self.passphrase or "").strip()
        address = str(self.address or "").strip().lower()
        if not 8 <= len(key) <= 512:
            raise ValueError("gasless API key format is invalid")
        if kind == "builder":
            if not 8 <= len(secret) <= 512 or not 8 <= len(passphrase) <= 512:
                raise ValueError("builder credential format is invalid")
            if address:
                raise ValueError("builder credentials cannot include an address")
        else:
            if secret or passphrase:
                raise ValueError("relayer credentials cannot include shared secrets")
            if re.fullmatch(r"0x[0-9a-f]{40}", address) is None:
                raise ValueError("relayer API key address is invalid")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "secret", secret)
        object.__setattr__(self, "passphrase", passphrase)
        object.__setattr__(self, "address", address)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "PolymarketGaslessCredentials":
        source = os.environ if environment is None else environment
        relayer_key = str(
            source.get("SIMPLE_AI_TRADING_POLYMARKET_RELAYER_API_KEY", "")
        ).strip()
        relayer_address = str(
            source.get(
                "SIMPLE_AI_TRADING_POLYMARKET_RELAYER_API_KEY_ADDRESS",
                "",
            )
        ).strip()
        builder_key = str(
            source.get("SIMPLE_AI_TRADING_POLYMARKET_BUILDER_API_KEY", "")
        ).strip()
        builder_secret = str(
            source.get("SIMPLE_AI_TRADING_POLYMARKET_BUILDER_SECRET", "")
        ).strip()
        builder_passphrase = str(
            source.get(
                "SIMPLE_AI_TRADING_POLYMARKET_BUILDER_PASSPHRASE",
                "",
            )
        ).strip()
        has_relayer = bool(relayer_key or relayer_address)
        has_builder = bool(builder_key or builder_secret or builder_passphrase)
        if has_relayer == has_builder:
            raise ValueError(
                "configure exactly one complete Polymarket gasless credential set"
            )
        if has_relayer:
            return cls(
                kind="relayer",
                key=relayer_key,
                address=relayer_address,
            )
        return cls(
            kind="builder",
            key=builder_key,
            secret=builder_secret,
            passphrase=builder_passphrase,
        )

    def to_sdk_api_key(self) -> object:
        try:
            from polymarket.auth import BuilderApiKey, RelayerApiKey
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket settlement requires the 'polymarket-live' extra"
            ) from exc
        if self.kind == "builder":
            return BuilderApiKey(
                self.key,
                self.secret,
                self.passphrase,
            )
        return RelayerApiKey(key=self.key, address=self.address)


@dataclass(frozen=True, slots=True)
class PolymarketRedemptionPreflight:
    condition_id: str
    adapter_address: str
    neg_risk: bool
    gas_estimate: int
    gas_price_wei: int
    native_balance_wei: int
    required_native_balance_wei: int
    gasless: bool = False

    def __post_init__(self) -> None:
        condition_id = str(self.condition_id or "").strip().lower()
        adapter = str(self.adapter_address or "").strip().lower()
        if re.fullmatch(r"^0x[0-9a-f]{64}$", condition_id) is None:
            raise ValueError("redemption preflight condition ID is invalid")
        if re.fullmatch(r"^0x[0-9a-f]{40}$", adapter) is None:
            raise ValueError("redemption preflight adapter is invalid")
        gas_estimate = int(self.gas_estimate)
        gas_price = int(self.gas_price_wei)
        native_balance = int(self.native_balance_wei)
        required_balance = int(self.required_native_balance_wei)
        gasless = bool(self.gasless)
        if gasless:
            if any(
                value != 0
                for value in (
                    gas_estimate,
                    gas_price,
                    native_balance,
                    required_balance,
                )
            ):
                raise ValueError("gasless redemption cannot report EOA gas")
        elif gas_estimate <= 0 or gas_price <= 0:
            raise ValueError("redemption preflight gas economics are invalid")
        if native_balance < 0 or (
            not gasless and required_balance < gas_estimate * gas_price
        ):
            raise ValueError("redemption preflight native balance is invalid")
        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "adapter_address", adapter)
        object.__setattr__(self, "gas_estimate", gas_estimate)
        object.__setattr__(self, "gas_price_wei", gas_price)
        object.__setattr__(self, "native_balance_wei", native_balance)
        object.__setattr__(
            self,
            "required_native_balance_wei",
            required_balance,
        )
        object.__setattr__(self, "gasless", gasless)


@dataclass(frozen=True, slots=True)
class PolymarketRedemptionSubmission:
    transaction_id: str
    transaction_hash: str
    _handle: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        transaction_id = str(self.transaction_id or "").strip()
        transaction_hash = str(self.transaction_hash or "").strip().lower()
        if transaction_id and _TRANSACTION_ID.fullmatch(transaction_id) is None:
            raise ValueError("redemption transaction ID is invalid")
        if (
            transaction_hash
            and _TRANSACTION_HASH.fullmatch(transaction_hash) is None
        ):
            raise ValueError("redemption transaction hash is invalid")
        if not transaction_id and not transaction_hash:
            raise ValueError("redemption submission lacks transaction identity")
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "transaction_hash", transaction_hash)


@dataclass(frozen=True, slots=True)
class PolymarketRedemptionOutcome:
    transaction_id: str
    transaction_hash: str

    def __post_init__(self) -> None:
        transaction_id = str(self.transaction_id or "").strip()
        transaction_hash = str(self.transaction_hash or "").strip().lower()
        if transaction_id and _TRANSACTION_ID.fullmatch(transaction_id) is None:
            raise ValueError("redemption outcome transaction ID is invalid")
        if _TRANSACTION_HASH.fullmatch(transaction_hash) is None:
            raise ValueError("redemption outcome transaction hash is invalid")
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "transaction_hash", transaction_hash)


@dataclass(frozen=True, slots=True)
class PolymarketRedemptionRecovery:
    state: str
    transaction_id: str
    transaction_hash: str

    def __post_init__(self) -> None:
        state = str(self.state or "").strip().lower()
        if state not in {"pending", "confirmed", "failed"}:
            raise ValueError("redemption recovery state is invalid")
        transaction_id = str(self.transaction_id or "").strip()
        transaction_hash = str(self.transaction_hash or "").strip().lower()
        if transaction_id and _TRANSACTION_ID.fullmatch(transaction_id) is None:
            raise ValueError("redemption recovery transaction ID is invalid")
        if (
            transaction_hash
            and _TRANSACTION_HASH.fullmatch(transaction_hash) is None
        ):
            raise ValueError("redemption recovery transaction hash is invalid")
        if state == "confirmed" and not transaction_hash:
            raise ValueError("terminal redemption recovery lacks a transaction hash")
        if state == "failed" and not (transaction_id or transaction_hash):
            raise ValueError("failed redemption recovery lacks transaction identity")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "transaction_hash", transaction_hash)


class PolymarketSettlementAccount(Protocol):
    """Read-only account snapshot used to prove dedicated-wallet ownership."""

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]: ...

    def positions(self) -> tuple[PolymarketRemotePosition, ...]: ...


class PolymarketRedemptionVenue(Protocol):
    """Transaction-only boundary; it never submits CLOB orders."""

    def assert_redemption_ready(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionPreflight: ...

    def submit_redemption(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionSubmission: ...

    def wait_redemption(
        self,
        submission: PolymarketRedemptionSubmission,
    ) -> PolymarketRedemptionOutcome: ...

    def recover_redemption(
        self,
        *,
        transaction_id: str,
        transaction_hash: str,
    ) -> PolymarketRedemptionRecovery: ...


def _response_json(response: requests.Response) -> object:
    if len(response.content) > 1024 * 1024:
        raise PolymarketLiveUnknownState("Polygon RPC response exceeded its bound")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise PolymarketLiveUnknownState("Polygon RPC response is invalid")
    if payload.get("error") is not None:
        raise PolymarketLiveUnknownState("Polygon RPC returned an error")
    return payload.get("result")


class OfficialPolymarketUnifiedRedemptionVenue:
    """Pinned official-SDK redemption with exactly one transaction submit."""

    def __init__(
        self,
        credentials: PolymarketLiveCredentials,
        *,
        client: object | None = None,
        gasless_credentials: PolymarketGaslessCredentials | None = None,
        session: requests.Session | None = None,
        rpc_url: str = POLYMARKET_POLYGON_RPC_URL,
        timeout_seconds: float = 10.0,
        gas_reserve_multiplier: int = 2,
        preflight_ttl_ms: int = 5_000,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if credentials.signature_type != 0 and gasless_credentials is None:
            raise PolymarketLiveBlocked(
                "smart-wallet redemption requires explicit relayer credentials"
            )
        self.credentials = credentials
        self.gasless_credentials = gasless_credentials
        self.session = session or requests.Session()
        self.rpc_url = str(rpc_url or "").strip()
        if not self.rpc_url.startswith("https://"):
            raise ValueError("Polygon RPC URL must use HTTPS")
        self.timeout_seconds = max(1.0, min(30.0, float(timeout_seconds)))
        self.gas_reserve_multiplier = int(gas_reserve_multiplier)
        if not 1 <= self.gas_reserve_multiplier <= 10:
            raise ValueError("gas reserve multiplier must lie in [1, 10]")
        self.preflight_ttl_ms = int(preflight_ttl_ms)
        if not 500 <= self.preflight_ttl_ms <= 30_000:
            raise ValueError("preflight TTL must lie in [500, 30000]")
        self._monotonic_ns = monotonic_ns
        self._preflight_lock = RLock()
        self._fresh_preflights: dict[str, int] = {}
        self._client = client or self._build_client()
        wallet = str(getattr(self._client, "wallet", "")).lower()
        wallet_type = str(getattr(self._client, "wallet_type", ""))
        expected_wallet_type = {
            0: "EOA",
            1: "POLY_PROXY",
            2: "GNOSIS_SAFE",
            3: "DEPOSIT_WALLET",
        }[credentials.signature_type]
        if (
            wallet != credentials.funder_address
            or wallet_type != expected_wallet_type
        ):
            raise PolymarketLiveBlocked(
                "unified SDK wallet identity differs from configured credentials"
            )

    def _build_client(self) -> object:
        try:
            from polymarket import SecureClient
            from polymarket.models.clob.api_key import ApiKeyCreds
        except ImportError as exc:
            raise RuntimeError(
                "Polymarket settlement requires the 'polymarket-live' extra"
            ) from exc
        try:
            installed = package_version("polymarket-client")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Polymarket settlement requires the 'polymarket-live' extra"
            ) from exc
        if installed != POLYMARKET_UNIFIED_SDK_VERSION:
            raise RuntimeError(
                "Polymarket unified SDK version differs from the audited pin"
            )
        credentials = ApiKeyCreds(
            apiKey=self.credentials.api_key,
            secret=self.credentials.api_secret,
            passphrase=self.credentials.api_passphrase,
        )
        api_key = (
            None
            if self.gasless_credentials is None
            else self.gasless_credentials.to_sdk_api_key()
        )
        creator = getattr(SecureClient, "_create", None)
        if not callable(creator):
            raise RuntimeError("audited unified SDK construction path differs")
        client = creator(
            private_key=self.credentials.private_key,
            wallet=self.credentials.funder_address,
            credentials=credentials,
            api_key=api_key,
            validate_credentials=True,
        )
        wallet_type = str(getattr(client, "wallet_type", ""))
        if wallet_type != "EOA":
            try:
                from polymarket._internal.actions.relayer.deployed import (
                    fetch_deployed_sync,
                )
                from polymarket.clients.secure import (
                    _relayer_transaction_type_for_wallet,
                )
            except ImportError as exc:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                raise RuntimeError(
                    "audited unified SDK deployment check differs"
                ) from exc
            context = getattr(client, "_ctx", None)
            try:
                deployed = fetch_deployed_sync(
                    context.relayer,
                    address=str(client.wallet),
                    type=_relayer_transaction_type_for_wallet(wallet_type),
                )
            except BaseException:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                raise
            if not deployed:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                raise PolymarketLiveBlocked(
                    "configured Polymarket smart wallet is not deployed"
                )
        return client

    def _native_balance_wei(self) -> int:
        response = self.session.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBalance",
                "params": [self.credentials.funder_address, "latest"],
            },
            timeout=self.timeout_seconds,
        )
        result = _response_json(response)
        if not isinstance(result, str) or re.fullmatch(r"0x[0-9a-fA-F]+", result) is None:
            raise PolymarketLiveBlocked("Polygon native balance is invalid")
        return int(result, 16)

    def assert_redemption_ready(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionPreflight:
        resolver = getattr(
            self._client,
            "_resolve_market_position_context",
            None,
        )
        context_owner = getattr(self._client, "_ctx", None)
        rpc = getattr(context_owner, "rpc", None)
        environment = getattr(context_owner, "environment", None)
        if not callable(resolver) or rpc is None or environment is None:
            raise RuntimeError("audited unified SDK internals differ")
        context = resolver(condition_id=condition_id, closed=True)
        resolved_condition = str(getattr(context, "condition_id", "")).lower()
        if resolved_condition != str(condition_id).lower():
            raise PolymarketLiveBlocked(
                "resolved market condition differs from redemption request"
            )
        adapter = str(getattr(context, "adapter_address", "")).lower()
        conditional_tokens = str(
            getattr(context, "position_erc1155_address", "")
        ).lower()
        if (
            re.fullmatch(r"0x[0-9a-f]{40}", adapter) is None
            or re.fullmatch(r"0x[0-9a-f]{40}", conditional_tokens) is None
        ):
            raise RuntimeError("audited unified SDK market context differs")
        approval_data = (
            "0xe985e9c5"
            + self.credentials.funder_address[2:].rjust(64, "0")
            + adapter[2:].rjust(64, "0")
        )
        approval = str(
            rpc.eth_call(to=conditional_tokens, data=approval_data)
        ).lower()
        if approval != "0x" + "0" * 63 + "1":
            raise PolymarketLiveBlocked(
                "redemption adapter approval is absent"
            )
        try:
            from polymarket._internal.actions.relayer.calls import (
                ctf_redeem_positions_call,
            )
        except ImportError as exc:
            raise RuntimeError("audited unified SDK internals differ") from exc
        call = ctf_redeem_positions_call(
            ctf=adapter,
            collateral=str(environment.collateral_token),
            condition_id=condition_id,
        )
        estimate_payload = {
            "from": self.credentials.funder_address,
            "to": str(call.to),
            "value": hex(call.value),
            "data": call.data,
        }
        gasless = self.credentials.signature_type != 0
        if gasless:
            gas_estimate = 0
            gas_price = 0
            required_balance = 0
            native_balance = 0
        else:
            gas_estimate = int(rpc.eth_estimate_gas(estimate_payload))
            gas_price = int(rpc.eth_gas_price())
            required_balance = (
                gas_estimate * gas_price * self.gas_reserve_multiplier
            )
            native_balance = self._native_balance_wei()
            if native_balance < required_balance:
                raise PolymarketLiveBlocked(
                    "dedicated EOA lacks the configured Polygon gas reserve"
                )
        result = PolymarketRedemptionPreflight(
            condition_id=condition_id,
            adapter_address=adapter,
            neg_risk=bool(getattr(context, "neg_risk")),
            gas_estimate=gas_estimate,
            gas_price_wei=gas_price,
            native_balance_wei=native_balance,
            required_native_balance_wei=required_balance,
            gasless=gasless,
        )
        with self._preflight_lock:
            self._fresh_preflights[result.condition_id] = (
                self._monotonic_ns() + self.preflight_ttl_ms * 1_000_000
            )
        return result

    @staticmethod
    def _deterministic_submit_error(exc: Exception) -> bool:
        try:
            from polymarket.errors import SigningError, UserInputError
        except ImportError:
            return False
        return isinstance(exc, (SigningError, UserInputError))

    def submit_redemption(
        self,
        condition_id: str,
    ) -> PolymarketRedemptionSubmission:
        normalized_condition = str(condition_id or "").strip().lower()
        with self._preflight_lock:
            deadline = self._fresh_preflights.pop(normalized_condition, None)
        if deadline is None or self._monotonic_ns() > deadline:
            raise PolymarketRedemptionRejected(
                "fresh single-use redemption preflight is required"
            )
        try:
            if self.credentials.signature_type == 0:
                handle = self._client.redeem_positions(
                    condition_id=normalized_condition,
                    metadata=f"Simple AI Trading redemption {normalized_condition}",
                )
            else:
                handle = self._submit_gasless_once(normalized_condition)
        except Exception as exc:
            if self._deterministic_submit_error(exc):
                raise PolymarketRedemptionRejected(
                    exc.__class__.__name__
                ) from exc
            raise
        return PolymarketRedemptionSubmission(
            transaction_id=str(getattr(handle, "transaction_id", "") or ""),
            transaction_hash=str(getattr(handle, "transaction_hash", "") or ""),
            _handle=handle,
        )

    def _submit_gasless_once(self, condition_id: str) -> object:
        try:
            from polymarket._internal.actions.relayer.calls import (
                ctf_redeem_positions_call,
            )
            from polymarket._internal.actions.relayer.gasless import (
                _submit_for_wallet_type_sync,
            )
            from polymarket.transactions import SyncGaslessTransactionHandle
        except ImportError as exc:
            raise RuntimeError("audited unified SDK submit path differs") from exc
        context = self._client._resolve_market_position_context(
            condition_id=condition_id,
            closed=True,
        )
        client_context = self._client._ctx
        call = ctf_redeem_positions_call(
            ctf=context.adapter_address,
            collateral=client_context.environment.collateral_token,
            condition_id=condition_id,
        )
        response = _submit_for_wallet_type_sync(
            client_context,
            calls=[call],
            metadata=f"Simple AI Trading redemption {condition_id}",
        )
        poll_delay = (
            client_context.environment.relayer_poll_frequency_ms / 1_000
        )
        return SyncGaslessTransactionHandle(
            transaction_id=response.transaction_id,
            transaction_hash=response.transaction_hash,
            _relayer=client_context.relayer,
            _max_polls=client_context.environment.relayer_max_polls,
            _poll_delay_s=poll_delay,
        )

    def wait_redemption(
        self,
        submission: PolymarketRedemptionSubmission,
    ) -> PolymarketRedemptionOutcome:
        try:
            outcome = submission._handle.wait()
        except Exception as exc:
            try:
                from polymarket.errors import TransactionFailedError
            except ImportError:
                TransactionFailedError = ()  # type: ignore[assignment,misc]
            if isinstance(exc, TransactionFailedError):
                raise PolymarketRedemptionFailed(
                    exc.__class__.__name__
                ) from exc
            raise
        transaction_id = str(getattr(outcome, "transaction_id", "") or "")
        transaction_hash = str(getattr(outcome, "transaction_hash", "") or "")
        if (
            submission.transaction_id
            and transaction_id
            and transaction_id != submission.transaction_id
        ):
            raise PolymarketLiveUnknownState(
                "redemption outcome transaction ID differs"
            )
        if (
            submission.transaction_hash
            and transaction_hash.lower() != submission.transaction_hash
        ):
            raise PolymarketLiveUnknownState(
                "redemption outcome transaction hash differs"
            )
        return PolymarketRedemptionOutcome(
            transaction_id=transaction_id or submission.transaction_id,
            transaction_hash=transaction_hash,
        )

    def recover_redemption(
        self,
        *,
        transaction_id: str,
        transaction_hash: str,
    ) -> PolymarketRedemptionRecovery:
        normalized_id = str(transaction_id or "").strip()
        normalized_hash = str(transaction_hash or "").strip().lower()
        if normalized_id:
            try:
                from polymarket._internal.actions.relayer.poll import (
                    _terminal_outcome,
                    fetch_gasless_transaction_sync,
                )
                from polymarket.errors import TransactionFailedError
            except ImportError as exc:
                raise RuntimeError(
                    "audited unified SDK recovery path differs"
                ) from exc
            try:
                transaction = fetch_gasless_transaction_sync(
                    self._client._ctx.relayer,
                    transaction_id=normalized_id,
                )
                outcome = _terminal_outcome(
                    transaction,
                    transaction_id=normalized_id,
                    fallback_hash=normalized_hash or None,
                )
            except TransactionFailedError:
                return PolymarketRedemptionRecovery(
                    state="failed",
                    transaction_id=normalized_id,
                    transaction_hash=normalized_hash,
                )
            if outcome is None:
                return PolymarketRedemptionRecovery(
                    state="pending",
                    transaction_id=normalized_id,
                    transaction_hash=normalized_hash,
                )
            return PolymarketRedemptionRecovery(
                state="confirmed",
                transaction_id=str(outcome.transaction_id or normalized_id),
                transaction_hash=str(outcome.transaction_hash),
            )
        if _TRANSACTION_HASH.fullmatch(normalized_hash) is None:
            raise PolymarketLiveUnknownState(
                "EOA redemption recovery requires a transaction hash"
            )
        response = self.session.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getTransactionReceipt",
                "params": [normalized_hash],
            },
            timeout=self.timeout_seconds,
        )
        result = _response_json(response)
        if result is None:
            return PolymarketRedemptionRecovery(
                state="pending",
                transaction_id="",
                transaction_hash=normalized_hash,
            )
        if not isinstance(result, Mapping):
            raise PolymarketLiveUnknownState("Polygon receipt is invalid")
        receipt_hash = str(result.get("transactionHash") or "").lower()
        if receipt_hash != normalized_hash:
            raise PolymarketLiveUnknownState(
                "Polygon receipt transaction hash differs"
            )
        status = result.get("status")
        if status in {"0x1", 1}:
            state = "confirmed"
        elif status in {"0x0", 0}:
            state = "failed"
        else:
            raise PolymarketLiveUnknownState(
                "Polygon receipt has an unrecognized status"
            )
        return PolymarketRedemptionRecovery(
            state=state,
            transaction_id="",
            transaction_hash=normalized_hash,
        )

    def close(self) -> None:
        with self._preflight_lock:
            self._fresh_preflights.clear()
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self.session.close()


class PolymarketRedemptionCoordinator:
    """Redeem only exact, confirmed bot inventory from a dedicated wallet."""

    def __init__(
        self,
        account: PolymarketSettlementAccount,
        venue: PolymarketRedemptionVenue,
        ledger: PolymarketLiveOrderLedger,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.account = account
        self.venue = venue
        self.ledger = ledger
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))

    @staticmethod
    def _position_map(
        positions: Sequence[PolymarketRemotePosition],
    ) -> dict[str, PolymarketRemotePosition]:
        output: dict[str, PolymarketRemotePosition] = {}
        for position in positions:
            if position.token_id in output:
                raise PolymarketLiveBlocked(
                    "remote position snapshot contains duplicate tokens"
                )
            output[position.token_id] = position
        return output

    def _owned_ready_inventory(
        self,
    ) -> tuple[
        tuple[PolymarketOwnedInventory, ...],
        tuple[PolymarketRemotePosition, ...],
    ]:
        local = self.ledger.owned_inventory()
        if any(item.provisional for item in local):
            raise PolymarketLiveBlocked(
                "provisional fills block Polymarket redemption"
            )
        remote = self.account.positions()
        local_by_token = {item.token_id: item for item in local}
        if len(local_by_token) != len(local):
            raise PolymarketLiveBlocked(
                "bot-owned inventory contains duplicate tokens"
            )
        remote_by_token = self._position_map(remote)
        if set(local_by_token) != set(remote_by_token):
            raise PolymarketLiveBlocked(
                "remote positions differ from bot-owned inventory"
            )
        for token_id, owned in local_by_token.items():
            observed = remote_by_token[token_id]
            if (
                observed.market_id != owned.market_id
                or abs(observed.quantity - owned.quantity)
                > _POSITION_TOLERANCE
            ):
                raise PolymarketLiveBlocked(
                    "remote position economics differ from bot-owned inventory"
                )
        return local, remote

    def _assert_quiescent(self) -> None:
        if self.ledger.open_owned_order_ids():
            raise PolymarketLiveBlocked(
                "open bot-owned orders block Polymarket redemption"
            )
        if self.account.open_orders():
            raise PolymarketLiveBlocked(
                "remote open orders block Polymarket redemption"
            )
        unresolved = tuple(
            record
            for record in self.ledger.redemption_records()
            if record.state in {"prepared", "submitting", "submitted", "unknown"}
        )
        if unresolved:
            raise PolymarketLiveBlocked(
                "an unresolved Polymarket redemption blocks another transaction"
            )

    def redeem_next_ready(
        self,
        *,
        condition_id: str | None = None,
    ) -> PolymarketRedemptionRecord | None:
        self._assert_quiescent()
        local, remote = self._owned_ready_inventory()
        remote_by_token = self._position_map(remote)
        by_condition: dict[str, list[PolymarketOwnedInventory]] = {}
        for item in local:
            by_condition.setdefault(item.market_id, []).append(item)
        ready_conditions = {
            market_id
            for market_id, inventory in by_condition.items()
            if all(remote_by_token[item.token_id].redeemable for item in inventory)
        }
        condition = (
            str(condition_id or "").strip().lower()
            if condition_id is not None
            else (min(ready_conditions) if ready_conditions else "")
        )
        if not condition:
            return None
        inventory = tuple(item for item in local if item.market_id == condition)
        if not inventory:
            raise PolymarketLiveBlocked(
                "requested condition has no bot-owned inventory"
            )
        if condition not in ready_conditions or any(
            not remote_by_token[item.token_id].redeemable for item in inventory
        ):
            raise PolymarketLiveBlocked(
                "requested condition is not fully redeemable"
            )
        preflight = self.venue.assert_redemption_ready(condition)
        if preflight.condition_id != condition:
            raise PolymarketLiveBlocked(
                "redemption preflight condition differs"
            )
        now = self._clock_ms()
        prepared = self.ledger.reserve_redemption(
            condition,
            inventory,
            observed_at_ms=now,
            preflight=asdict(preflight),
        )
        prepared = self.ledger.transition_redemption(
            prepared.redemption_id,
            expected_states=("prepared",),
            state="submitting",
            observed_at_ms=self._clock_ms(),
        )
        try:
            submission = self.venue.submit_redemption(condition)
        except PolymarketRedemptionRejected as exc:
            return self.ledger.transition_redemption(
                prepared.redemption_id,
                expected_states=("submitting",),
                state="failed",
                observed_at_ms=self._clock_ms(),
                failure_code=exc.__class__.__name__,
            )
        except Exception as exc:
            self.ledger.transition_redemption(
                prepared.redemption_id,
                expected_states=("submitting",),
                state="unknown",
                observed_at_ms=self._clock_ms(),
                failure_code=exc.__class__.__name__,
            )
            raise PolymarketLiveUnknownState(
                "Polymarket redemption submission outcome is unknown"
            ) from exc
        submitted = self.ledger.transition_redemption(
            prepared.redemption_id,
            expected_states=("submitting",),
            state="submitted",
            observed_at_ms=self._clock_ms(),
            transaction_id=submission.transaction_id,
            transaction_hash=submission.transaction_hash,
        )
        try:
            outcome = self.venue.wait_redemption(submission)
        except PolymarketRedemptionFailed as exc:
            return self.ledger.transition_redemption(
                submitted.redemption_id,
                expected_states=("submitted",),
                state="failed",
                observed_at_ms=self._clock_ms(),
                failure_code=exc.__class__.__name__,
            )
        except Exception as exc:
            self.ledger.transition_redemption(
                submitted.redemption_id,
                expected_states=("submitted",),
                state="unknown",
                observed_at_ms=self._clock_ms(),
                failure_code=exc.__class__.__name__,
            )
            raise PolymarketLiveUnknownState(
                "Polymarket redemption confirmation is unknown"
            ) from exc
        return self.ledger.transition_redemption(
            submitted.redemption_id,
            expected_states=("submitted",),
            state="confirmed",
            observed_at_ms=self._clock_ms(),
            transaction_id=outcome.transaction_id or submitted.transaction_id,
            transaction_hash=outcome.transaction_hash,
        )

    def recover_incomplete(self) -> tuple[PolymarketRedemptionRecord, ...]:
        output: list[PolymarketRedemptionRecord] = []
        for record in self.ledger.redemption_records():
            if record.state == "prepared":
                output.append(
                    self.ledger.transition_redemption(
                        record.redemption_id,
                        expected_states=("prepared",),
                        state="failed",
                        observed_at_ms=self._clock_ms(),
                        failure_code="restart_before_submission",
                    )
                )
                continue
            if record.state == "submitting":
                output.append(
                    self.ledger.transition_redemption(
                        record.redemption_id,
                        expected_states=("submitting",),
                        state="unknown",
                        observed_at_ms=self._clock_ms(),
                        failure_code="restart_during_submission",
                    )
                )
                continue
            if record.state not in {"submitted", "unknown"}:
                continue
            try:
                recovered = self.venue.recover_redemption(
                    transaction_id=record.transaction_id,
                    transaction_hash=record.transaction_hash,
                )
            except Exception:
                if record.state == "submitted":
                    output.append(
                        self.ledger.transition_redemption(
                            record.redemption_id,
                            expected_states=("submitted",),
                            state="unknown",
                            observed_at_ms=self._clock_ms(),
                            failure_code="recovery_query_failed",
                        )
                    )
                else:
                    output.append(record)
                continue
            if recovered.state == "pending":
                output.append(record)
                continue
            output.append(
                self.ledger.transition_redemption(
                    record.redemption_id,
                    expected_states=(record.state,),
                    state=recovered.state,
                    observed_at_ms=self._clock_ms(),
                    transaction_id=(
                        recovered.transaction_id or record.transaction_id
                    ),
                    transaction_hash=(
                        recovered.transaction_hash or record.transaction_hash
                    ),
                    failure_code=(
                        "proven_transaction_failure"
                        if recovered.state == "failed"
                        else ""
                    ),
                )
            )
        return tuple(output)


class PolymarketSettlementService:
    """Recover continuously and optionally redeem one proven condition at a time."""

    def __init__(
        self,
        coordinator: PolymarketRedemptionCoordinator,
        runtime_authority: PolymarketRuntimeAuthority,
        *,
        automatic_redemption_enabled: bool = False,
        interval_seconds: float = 30.0,
    ) -> None:
        self.coordinator = coordinator
        self.runtime_authority = runtime_authority
        self.automatic_redemption_enabled = bool(automatic_redemption_enabled)
        self.interval_seconds = max(5.0, min(300.0, float(interval_seconds)))

    async def run_once(self) -> PolymarketRedemptionRecord | None:
        recovered = await asyncio.to_thread(
            self.coordinator.recover_incomplete
        )
        if any(record.state == "unknown" for record in recovered):
            self.runtime_authority.note_reconciliation_failure(
                "unknown_redemption_state"
            )
            return None
        if not self.automatic_redemption_enabled:
            return None
        self.runtime_authority.assert_submission_allowed(closing_only=True)
        try:
            return await asyncio.to_thread(
                self.coordinator.redeem_next_ready
            )
        except PolymarketLiveUnknownState:
            self.runtime_authority.note_reconciliation_failure(
                "unknown_redemption_state"
            )
            raise

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The runtime authority is already fail-closed. Keep recovery
                # alive instead of terminating the independent supervisor.
                pass
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self.interval_seconds,
                )
            except TimeoutError:
                continue


__all__ = [
    "POLYMARKET_POLYGON_RPC_URL",
    "POLYMARKET_UNIFIED_SDK_VERSION",
    "OfficialPolymarketUnifiedRedemptionVenue",
    "PolymarketGaslessCredentials",
    "PolymarketRedemptionCoordinator",
    "PolymarketRedemptionFailed",
    "PolymarketRedemptionOutcome",
    "PolymarketRedemptionPreflight",
    "PolymarketRedemptionRecovery",
    "PolymarketRedemptionRejected",
    "PolymarketRedemptionSubmission",
    "PolymarketSettlementService",
]
