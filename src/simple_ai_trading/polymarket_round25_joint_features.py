"""Hash-bound composition of Round 25 TWAP and CLOB causal features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from .polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    Round25ClobFeatureSnapshot,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
    Round25TwapFeatureSnapshot,
)


POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION = (
    "polymarket-round25-joint-twap-clob-features-v1"
)
POLYMARKET_ROUND25_JOINT_FEATURE_NAMES = (
    *POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    *POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Round25JointFeatureSnapshot:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    available: bool
    reasons: tuple[str, ...]
    market_prior_probability: float
    values: tuple[float, ...]
    source_chain_sha256: str
    twap_source_chain_sha256: str
    clob_source_chain_sha256: str
    maximum_receipt_ms: int
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or type(self.decision_time_ms) is not int
            or type(self.available) is not bool
            or not isinstance(self.reasons, tuple)
            or any(not isinstance(reason, str) or not reason for reason in self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
            or not isinstance(self.values, tuple)
            or len(self.values) != len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in self.values
            )
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in (
                    self.source_chain_sha256,
                    self.twap_source_chain_sha256,
                    self.clob_source_chain_sha256,
                )
            )
            or type(self.maximum_receipt_ms) is not int
            or self.model_design_sha256 != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.trading_authority is not False
        ):
            raise ValueError("Round 25 joint feature snapshot is invalid")
        if self.available:
            expected_chain = _canonical_sha256(
                {
                    "clob_source_chain_sha256": self.clob_source_chain_sha256,
                    "condition_id": self.condition_id,
                    "decision_time_ms": self.decision_time_ms,
                    "feature_schema_version": (
                        POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
                    ),
                    "model_design_sha256": self.model_design_sha256,
                    "twap_source_chain_sha256": self.twap_source_chain_sha256,
                }
            )
            if (
                self.reasons
                or not 0.0 < self.market_prior_probability < 1.0
                or self.source_chain_sha256 != expected_chain
                or _EMPTY_SHA256
                in (
                    self.source_chain_sha256,
                    self.twap_source_chain_sha256,
                    self.clob_source_chain_sha256,
                )
                or not 0 < self.maximum_receipt_ms <= self.decision_time_ms
            ):
                raise ValueError("Round 25 available joint snapshot differs")
        elif (
            not self.reasons
            or self.market_prior_probability != 0.5
            or any(self.values)
            or any(
                value != _EMPTY_SHA256
                for value in (
                    self.source_chain_sha256,
                    self.twap_source_chain_sha256,
                    self.clob_source_chain_sha256,
                )
            )
            or self.maximum_receipt_ms != 0
        ):
            raise ValueError("Round 25 unavailable joint snapshot differs")


def combine_round25_features(
    twap: Round25TwapFeatureSnapshot,
    clob: Round25ClobFeatureSnapshot,
) -> Round25JointFeatureSnapshot:
    """Combine exact same-decision sources or return a zeroed blocked row."""

    if not isinstance(twap, Round25TwapFeatureSnapshot) or not isinstance(
        clob, Round25ClobFeatureSnapshot
    ):
        raise TypeError("Round 25 joint feature source type differs")
    if (
        twap.condition_id != clob.condition_id
        or twap.event_start_ms != clob.event_start_ms
        or twap.decision_time_ms != clob.decision_time_ms
        or twap.model_design_sha256 != clob.model_design_sha256
        or twap.trading_authority
        or clob.trading_authority
    ):
        raise ValueError("Round 25 joint feature source identity differs")
    if not twap.available or not clob.available:
        reasons = tuple(
            dict.fromkeys(
                (
                    *(f"twap:{reason}" for reason in twap.reasons),
                    *(f"clob:{reason}" for reason in clob.reasons),
                )
            )
        )
        if not reasons:
            raise RuntimeError("Round 25 unavailable source omitted its reason")
        return Round25JointFeatureSnapshot(
            condition_id=twap.condition_id,
            event_start_ms=twap.event_start_ms,
            decision_time_ms=twap.decision_time_ms,
            available=False,
            reasons=reasons,
            market_prior_probability=0.5,
            values=(0.0,) * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES),
            source_chain_sha256=_EMPTY_SHA256,
            twap_source_chain_sha256=_EMPTY_SHA256,
            clob_source_chain_sha256=_EMPTY_SHA256,
            maximum_receipt_ms=0,
        )
    source_chain = _canonical_sha256(
        {
            "clob_source_chain_sha256": clob.source_chain_sha256,
            "condition_id": twap.condition_id,
            "decision_time_ms": twap.decision_time_ms,
            "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
            "model_design_sha256": twap.model_design_sha256,
            "twap_source_chain_sha256": twap.source_chain_sha256,
        }
    )
    return Round25JointFeatureSnapshot(
        condition_id=twap.condition_id,
        event_start_ms=twap.event_start_ms,
        decision_time_ms=twap.decision_time_ms,
        available=True,
        reasons=(),
        market_prior_probability=clob.market_prior_probability,
        values=(*twap.values, *clob.values),
        source_chain_sha256=source_chain,
        twap_source_chain_sha256=twap.source_chain_sha256,
        clob_source_chain_sha256=clob.source_chain_sha256,
        maximum_receipt_ms=max(twap.maximum_receipt_ms, clob.maximum_receipt_ms),
    )


__all__ = [
    "POLYMARKET_ROUND25_JOINT_FEATURE_NAMES",
    "POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION",
    "Round25JointFeatureSnapshot",
    "combine_round25_features",
]
