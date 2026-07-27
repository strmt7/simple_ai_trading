"""Deterministic testnet execution-calibration campaign planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Mapping

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL,
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE,
)


ROUND74_EXECUTION_CAMPAIGN_SCHEMA_VERSION = (
    "round-074-execution-calibration-campaign-v1"
)
ROUND74_EXECUTION_CAMPAIGN_MAXIMUM_PAIRS_PER_SYMBOL = 10_000
_CAMPAIGN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _campaign_id(value: object) -> str:
    selected = str(value).strip()
    if not _CAMPAIGN_ID_PATTERN.fullmatch(selected):
        raise ValueError("Round 74 execution campaign ID differs")
    return selected


def _target_notional(value: object) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Round 74 execution campaign target notional differs") from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError("Round 74 execution campaign target notional differs")
    return selected


def _pairs_per_symbol(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Round 74 execution campaign pair count differs")
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 execution campaign pair count differs") from exc
    if (
        selected < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
        or selected > ROUND74_EXECUTION_CAMPAIGN_MAXIMUM_PAIRS_PER_SYMBOL
    ):
        raise ValueError("Round 74 execution campaign pair count differs")
    return selected


@dataclass(frozen=True)
class Round74ExecutionCampaignSlot:
    ordinal: int
    symbol_pair_index: int
    symbol: str
    entry_side: str
    round_trip_id: str

    def as_dict(self) -> dict[str, object]:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or isinstance(self.symbol_pair_index, bool)
            or not isinstance(self.symbol_pair_index, int)
            or self.symbol_pair_index < 0
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or self.entry_side not in {"BUY", "SELL"}
            or not str(self.round_trip_id).strip()
        ):
            raise ValueError("Round 74 execution campaign slot differs")
        return {
            "ordinal": self.ordinal,
            "symbol_pair_index": self.symbol_pair_index,
            "symbol": self.symbol,
            "entry_side": self.entry_side,
            "round_trip_id": self.round_trip_id,
        }


@dataclass(frozen=True)
class Round74ExecutionCampaignPlan:
    campaign_id: str
    target_quote_notional: Decimal
    pairs_per_symbol: int
    slots: tuple[Round74ExecutionCampaignSlot, ...]
    schema_version: str = ROUND74_EXECUTION_CAMPAIGN_SCHEMA_VERSION

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        selected_campaign_id = _campaign_id(self.campaign_id)
        target = _target_notional(self.target_quote_notional)
        pair_count = _pairs_per_symbol(self.pairs_per_symbol)
        expected_count = pair_count * len(ROUND74_EVENT_TARGET_SYMBOLS)
        slot_payloads = [slot.as_dict() for slot in self.slots]
        if (
            self.schema_version != ROUND74_EXECUTION_CAMPAIGN_SCHEMA_VERSION
            or selected_campaign_id != self.campaign_id
            or len(slot_payloads) != expected_count
            or [item["ordinal"] for item in slot_payloads]
            != list(range(expected_count))
            or len({item["round_trip_id"] for item in slot_payloads}) != expected_count
        ):
            raise ValueError("Round 74 execution campaign plan differs")
        symbol_counts = Counter(item["symbol"] for item in slot_payloads)
        side_counts = Counter(
            (item["symbol"], item["entry_side"]) for item in slot_payloads
        )
        if any(
            symbol_counts[symbol] != pair_count
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ) or any(
            side_counts[(symbol, side)]
            < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            for side in ("BUY", "SELL")
        ):
            raise ValueError("Round 74 execution campaign coverage differs")
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "environment": "binance_usdm_testnet",
            "target_quote_notional": format(target, "f"),
            "pairs_per_symbol": pair_count,
            "minimum_pairs_per_symbol": (
                ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
            ),
            "minimum_pairs_per_symbol_entry_side": (
                ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
            ),
            "slots": slot_payloads,
            "authority": {
                "testnet_execution_runtime_calibration_only": True,
                "mainnet_execution_evidence": False,
                "mainnet_trading_authority": False,
                "model_training": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
            },
        }
        if include_sha256:
            value["plan_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74ExecutionCampaignPlan:
        payload = dict(value)
        claimed_sha256 = str(payload.pop("plan_sha256", ""))
        if claimed_sha256 != _canonical_sha256(payload):
            raise ValueError("Round 74 execution campaign digest differs")
        raw_slots = payload.get("slots")
        if not isinstance(raw_slots, list):
            raise ValueError("Round 74 execution campaign slots differ")
        slots = tuple(
            Round74ExecutionCampaignSlot(
                ordinal=int(item["ordinal"]),
                symbol_pair_index=int(item["symbol_pair_index"]),
                symbol=str(item["symbol"]),
                entry_side=str(item["entry_side"]),
                round_trip_id=str(item["round_trip_id"]),
            )
            for item in raw_slots
            if isinstance(item, Mapping)
        )
        selected = cls(
            campaign_id=str(payload.get("campaign_id", "")),
            target_quote_notional=_target_notional(
                payload.get("target_quote_notional")
            ),
            pairs_per_symbol=_pairs_per_symbol(payload.get("pairs_per_symbol")),
            slots=slots,
            schema_version=str(payload.get("schema_version", "")),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 execution campaign payload differs")
        return selected

    def next_slot(
        self,
        *,
        completed_round_trip_ids: tuple[str, ...],
    ) -> Round74ExecutionCampaignSlot | None:
        completed = tuple(str(value) for value in completed_round_trip_ids)
        if len(set(completed)) != len(completed) or not set(completed).issubset(
            {slot.round_trip_id for slot in self.slots}
        ):
            raise ValueError("Round 74 execution campaign completion set differs")
        completed_set = set(completed)
        return next(
            (slot for slot in self.slots if slot.round_trip_id not in completed_set),
            None,
        )


def build_round74_execution_campaign_plan(
    *,
    campaign_id: str,
    target_quote_notional: Decimal,
    pairs_per_symbol: int = (ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL),
) -> Round74ExecutionCampaignPlan:
    selected_id = _campaign_id(campaign_id)
    selected_target = _target_notional(target_quote_notional)
    selected_count = _pairs_per_symbol(pairs_per_symbol)
    slots: list[Round74ExecutionCampaignSlot] = []
    for symbol_pair_index in range(selected_count):
        entry_side = "BUY" if symbol_pair_index % 2 == 0 else "SELL"
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
            ordinal = len(slots)
            round_trip_id = (
                f"{selected_id}-{symbol.lower()}-"
                f"{symbol_pair_index:05d}-{entry_side.lower()}"
            )
            slots.append(
                Round74ExecutionCampaignSlot(
                    ordinal=ordinal,
                    symbol_pair_index=symbol_pair_index,
                    symbol=symbol,
                    entry_side=entry_side,
                    round_trip_id=round_trip_id,
                )
            )
    plan = Round74ExecutionCampaignPlan(
        campaign_id=selected_id,
        target_quote_notional=selected_target,
        pairs_per_symbol=selected_count,
        slots=tuple(slots),
    )
    plan.as_dict()
    return plan


__all__ = [
    "ROUND74_EXECUTION_CAMPAIGN_MAXIMUM_PAIRS_PER_SYMBOL",
    "ROUND74_EXECUTION_CAMPAIGN_SCHEMA_VERSION",
    "Round74ExecutionCampaignPlan",
    "Round74ExecutionCampaignSlot",
    "build_round74_execution_campaign_plan",
]
