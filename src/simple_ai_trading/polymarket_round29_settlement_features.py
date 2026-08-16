"""Target-blind settlement-state interactions for Polymarket Round 29.

The Chainlink TWAP calculation is intentionally not reconstructed.  These
features transform only exact causal fields already present in a validated
Round 27 row and can be matched with either the Round 27 or Round 28 view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from .polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
    Round28FeatureRow,
)


POLYMARKET_ROUND29_SETTLEMENT_OVERLAY_SCHEMA_VERSION = (
    "polymarket-round29-settlement-state-overlay-v1"
)
POLYMARKET_ROUND29_FEATURE_SCHEMA_VERSION = (
    "polymarket-round29-settlement-state-features-v1"
)
POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES = (
    "settlement.twap_current_side",
    "settlement.twap_margin_per_sqrt_remaining_second",
    "settlement.twap_margin_per_remaining_second",
    "settlement.twap_diffusion_state_asinh",
    "settlement.twap_signed_path_efficiency",
    "settlement.twap_margin_x_elapsed_fraction",
)
POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES).encode("ascii")
).hexdigest()
POLYMARKET_ROUND29_BASE_FEATURE_NAMES = (
    *POLYMARKET_ROUND27_FEATURE_NAMES,
    *POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
)
POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND29_BASE_FEATURE_NAMES).encode("ascii")
).hexdigest()
POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES = (
    *POLYMARKET_ROUND28_FEATURE_NAMES,
    *POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
)
POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES).encode("ascii")
).hexdigest()

Round29FeatureView = Literal[
    "round29_settlement_augmented",
    "round29_bbo_settlement_augmented",
]
_FEATURE_CONTRACTS: dict[Round29FeatureView, tuple[tuple[str, ...], str]] = {
    "round29_settlement_augmented": (
        POLYMARKET_ROUND29_BASE_FEATURE_NAMES,
        POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256,
    ),
    "round29_bbo_settlement_augmented": (
        POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES,
        POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256,
    ),
}
_ROUND27_INDEX = {
    name: index for index, name in enumerate(POLYMARKET_ROUND27_FEATURE_NAMES)
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_NUMERICAL_SCALE_FLOOR = math.ulp(1.0)


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


def _side(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _settlement_values(base: Round27FeatureRow) -> tuple[float, ...]:
    row = base.validated()
    margin = float(row.values[_ROUND27_INDEX["twap.log_distance_from_open"]])
    variance_rate = float(row.values[_ROUND27_INDEX["twap.variance_rate_per_second"]])
    path_efficiency = float(row.values[_ROUND27_INDEX["twap.path_efficiency"]])
    remaining = float(row.values[_ROUND27_INDEX["phase.remaining_seconds"]])
    elapsed_fraction = float(row.values[_ROUND27_INDEX["phase.elapsed_fraction"]])
    if (
        variance_rate < 0.0
        or not 0.0 <= path_efficiency <= 1.0
        or not 0.0 <= elapsed_fraction <= 1.0
        or remaining <= 0.0
    ):
        raise ValueError("Round 29 settlement source fields differ")
    signed_side = _side(margin)
    diffusion_scale = math.sqrt(variance_rate * remaining)
    values = (
        signed_side,
        margin / math.sqrt(remaining),
        margin / remaining,
        math.asinh(margin / max(diffusion_scale, _NUMERICAL_SCALE_FLOOR)),
        signed_side * path_efficiency,
        margin * elapsed_fraction,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Round 29 settlement feature is non-finite")
    return values


@dataclass(frozen=True, slots=True)
class Round29SettlementOverlayRow:
    schema_version: str
    decision_time_ms: int
    base_row_sha256: str
    values: tuple[float, ...]
    feature_names_sha256: str
    source_chain_sha256: str
    row_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def create(cls, base: Round27FeatureRow) -> "Round29SettlementOverlayRow":
        selected = base.validated()
        source_identity = {
            "decision_time_ms": selected.decision_time_ms,
            "base_row_sha256": selected.row_sha256,
        }
        payload = {
            "schema_version": POLYMARKET_ROUND29_SETTLEMENT_OVERLAY_SCHEMA_VERSION,
            "decision_time_ms": selected.decision_time_ms,
            "base_row_sha256": selected.row_sha256,
            "values": _settlement_values(selected),
            "feature_names_sha256": (
                POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES_SHA256
            ),
            "source_chain_sha256": _canonical_sha256(source_identity),
            "target_accessed": False,
            "trading_authority": False,
        }
        return cls(
            schema_version=POLYMARKET_ROUND29_SETTLEMENT_OVERLAY_SCHEMA_VERSION,
            decision_time_ms=selected.decision_time_ms,
            base_row_sha256=selected.row_sha256,
            values=_settlement_values(selected),
            feature_names_sha256=(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES_SHA256),
            source_chain_sha256=_canonical_sha256(source_identity),
            row_sha256=_canonical_sha256(payload),
            target_accessed=False,
            trading_authority=False,
        ).validated()

    def validated(self) -> "Round29SettlementOverlayRow":
        payload = {
            "schema_version": self.schema_version,
            "decision_time_ms": self.decision_time_ms,
            "base_row_sha256": self.base_row_sha256,
            "values": self.values,
            "feature_names_sha256": self.feature_names_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }
        source_identity = {
            "decision_time_ms": self.decision_time_ms,
            "base_row_sha256": self.base_row_sha256,
        }
        if (
            self.schema_version != POLYMARKET_ROUND29_SETTLEMENT_OVERLAY_SCHEMA_VERSION
            or type(self.decision_time_ms) is not int
            or self.decision_time_ms <= 0
            or self.decision_time_ms % 1_000
            or _SHA256.fullmatch(self.base_row_sha256) is None
            or len(self.values) != len(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.feature_names_sha256
            != POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES_SHA256
            or self.source_chain_sha256 != _canonical_sha256(source_identity)
            or self.row_sha256 != _canonical_sha256(payload)
            or self.target_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 29 settlement overlay row differs")
        return self


@dataclass(frozen=True, slots=True)
class Round29FeatureRow:
    schema_version: str
    feature_view: Round29FeatureView
    run_id: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    values: tuple[float, ...]
    feature_names_sha256: str
    maximum_receipt_wall_ms: int
    base_row_sha256: str
    settlement_overlay_row_sha256: str
    bbo_row_sha256: str | None
    source_chain_sha256: str
    row_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def from_round27(
        cls,
        base: Round27FeatureRow,
        overlay: Round29SettlementOverlayRow,
    ) -> "Round29FeatureRow":
        selected = base.validated()
        return cls._create(
            feature_view="round29_settlement_augmented",
            run_id=selected.run_id,
            condition_id=selected.condition_id,
            event_start_ms=selected.event_start_ms,
            decision_time_ms=selected.decision_time_ms,
            market_prior_probability=selected.market_prior_probability,
            values=(*selected.values, *overlay.values),
            maximum_receipt_wall_ms=selected.maximum_receipt_wall_ms,
            base_row_sha256=selected.row_sha256,
            settlement_overlay=overlay,
            bbo_row_sha256=None,
        )

    @classmethod
    def from_round28(
        cls,
        base: Round28FeatureRow,
        overlay: Round29SettlementOverlayRow,
    ) -> "Round29FeatureRow":
        selected = base.validated()
        return cls._create(
            feature_view="round29_bbo_settlement_augmented",
            run_id=selected.run_id,
            condition_id=selected.condition_id,
            event_start_ms=selected.event_start_ms,
            decision_time_ms=selected.decision_time_ms,
            market_prior_probability=selected.market_prior_probability,
            values=(*selected.values, *overlay.values),
            maximum_receipt_wall_ms=selected.maximum_receipt_wall_ms,
            base_row_sha256=selected.base_row_sha256,
            settlement_overlay=overlay,
            bbo_row_sha256=selected.row_sha256,
        )

    @classmethod
    def _create(
        cls,
        *,
        feature_view: Round29FeatureView,
        run_id: str,
        condition_id: str,
        event_start_ms: int,
        decision_time_ms: int,
        market_prior_probability: float,
        values: tuple[float, ...],
        maximum_receipt_wall_ms: int,
        base_row_sha256: str,
        settlement_overlay: Round29SettlementOverlayRow,
        bbo_row_sha256: str | None,
    ) -> "Round29FeatureRow":
        selected_overlay = settlement_overlay.validated()
        if (
            selected_overlay.base_row_sha256 != base_row_sha256
            or selected_overlay.decision_time_ms != decision_time_ms
        ):
            raise ValueError("Round 29 base and settlement identities differ")
        feature_names, feature_hash = _FEATURE_CONTRACTS[feature_view]
        if len(values) != len(feature_names):
            raise ValueError("Round 29 composed feature width differs")
        source_identity = {
            "base_row_sha256": base_row_sha256,
            "settlement_overlay_row_sha256": selected_overlay.row_sha256,
            "bbo_row_sha256": bbo_row_sha256,
        }
        payload = {
            "schema_version": POLYMARKET_ROUND29_FEATURE_SCHEMA_VERSION,
            "feature_view": feature_view,
            "run_id": run_id,
            "condition_id": condition_id,
            "event_start_ms": event_start_ms,
            "decision_time_ms": decision_time_ms,
            "market_prior_probability": market_prior_probability,
            "values": values,
            "feature_names_sha256": feature_hash,
            "maximum_receipt_wall_ms": maximum_receipt_wall_ms,
            "base_row_sha256": base_row_sha256,
            "settlement_overlay_row_sha256": selected_overlay.row_sha256,
            "bbo_row_sha256": bbo_row_sha256,
            "source_chain_sha256": _canonical_sha256(source_identity),
            "target_accessed": False,
            "trading_authority": False,
        }
        return cls(
            schema_version=POLYMARKET_ROUND29_FEATURE_SCHEMA_VERSION,
            feature_view=feature_view,
            run_id=run_id,
            condition_id=condition_id,
            event_start_ms=event_start_ms,
            decision_time_ms=decision_time_ms,
            market_prior_probability=market_prior_probability,
            values=values,
            feature_names_sha256=feature_hash,
            maximum_receipt_wall_ms=maximum_receipt_wall_ms,
            base_row_sha256=base_row_sha256,
            settlement_overlay_row_sha256=selected_overlay.row_sha256,
            bbo_row_sha256=bbo_row_sha256,
            source_chain_sha256=_canonical_sha256(source_identity),
            row_sha256=_canonical_sha256(payload),
            target_accessed=False,
            trading_authority=False,
        ).validated()

    def validated(self) -> "Round29FeatureRow":
        try:
            feature_names, feature_hash = _FEATURE_CONTRACTS[self.feature_view]
        except (KeyError, TypeError) as exc:
            raise ValueError("Round 29 feature view differs") from exc
        source_identity = {
            "base_row_sha256": self.base_row_sha256,
            "settlement_overlay_row_sha256": self.settlement_overlay_row_sha256,
            "bbo_row_sha256": self.bbo_row_sha256,
        }
        payload = {
            "schema_version": self.schema_version,
            "feature_view": self.feature_view,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "market_prior_probability": self.market_prior_probability,
            "values": self.values,
            "feature_names_sha256": self.feature_names_sha256,
            "maximum_receipt_wall_ms": self.maximum_receipt_wall_ms,
            "base_row_sha256": self.base_row_sha256,
            "settlement_overlay_row_sha256": self.settlement_overlay_row_sha256,
            "bbo_row_sha256": self.bbo_row_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }
        if _round29_core_fields_differ(self, feature_names, feature_hash):
            raise ValueError("Round 29 feature row differs")
        if _round29_view_binding_differs(self):
            raise ValueError("Round 29 feature row differs")
        if (
            self.source_chain_sha256 != _canonical_sha256(source_identity)
            or self.row_sha256 != _canonical_sha256(payload)
            or self.target_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 29 feature row differs")
        return self


def _round29_core_fields_differ(
    row: Round29FeatureRow,
    feature_names: tuple[str, ...],
    feature_hash: str,
) -> bool:
    return (
        row.schema_version != POLYMARKET_ROUND29_FEATURE_SCHEMA_VERSION
        or not row.run_id
        or _CONDITION_ID.fullmatch(row.condition_id) is None
        or not row.event_start_ms <= row.decision_time_ms < row.event_start_ms + 300_000
        or bool(row.decision_time_ms % 1_000)
        or not 0.0 < row.market_prior_probability < 1.0
        or len(row.values) != len(feature_names)
        or any(not math.isfinite(value) for value in row.values)
        or row.feature_names_sha256 != feature_hash
        or not 0 < row.maximum_receipt_wall_ms < row.decision_time_ms
        or _SHA256.fullmatch(row.base_row_sha256) is None
        or _SHA256.fullmatch(row.settlement_overlay_row_sha256) is None
    )


def _round29_view_binding_differs(row: Round29FeatureRow) -> bool:
    if row.feature_view == "round29_settlement_augmented":
        return row.bbo_row_sha256 is not None
    return row.bbo_row_sha256 is None or _SHA256.fullmatch(row.bbo_row_sha256) is None


__all__ = [
    "POLYMARKET_ROUND29_BASE_FEATURE_NAMES",
    "POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES",
    "POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND29_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES",
    "POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND29_SETTLEMENT_OVERLAY_SCHEMA_VERSION",
    "Round29FeatureRow",
    "Round29FeatureView",
    "Round29SettlementOverlayRow",
]
