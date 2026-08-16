"""Target-blind Binance BBO overlay for the Round 27 BTC feature corpus.

The independent Round 21 sidecar records Binance spot and USD-M trades plus
best-bid/offer updates.  This module retains only the BBO-derived fields and
joins them to Round 27 by receipt-time decision identity.  It never reads a
market resolution, target, fill, account, credential, or order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import duckdb

from .duckdb_batch import insert_rows_columnar
from .polymarket_round21_binance_features import (
    POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    Round21OptionalBinanceFeatures,
)
from .polymarket_round21_sidecar_replay import Round21SidecarReplay
from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    Round27FeatureRow,
)


POLYMARKET_ROUND28_BOOK_TICKER_SCHEMA_VERSION = (
    "polymarket-round28-binance-bbo-overlay-v1"
)
POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION = (
    "polymarket-round28-receipt-time-features-v1"
)
POLYMARKET_ROUND28_BOOK_TICKER_STORE_SCHEMA_VERSION = (
    "polymarket-round28-binance-bbo-overlay-store-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_AUTHORITY = {
    "credentials_used": False,
    "execution_connected": False,
    "orders_submitted": False,
    "target_accessed": False,
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
}


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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 28 {name} SHA-256 differs")
    return selected


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"Round 28 {name} is not an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Round 28 {name} is not an integer") from exc


def _is_source_bbo_feature(name: str) -> bool:
    return (
        ".book_" in name
        or name.endswith(".book_receipt_age_ms")
        or name
        in {
            "usdm.log_mid_basis",
            "usdm.log_microprice_basis",
            "usdm.spot_minus_usdm_book_receipt_skew_ms",
        }
        or name.startswith("usdm.spot_minus_usdm_mid_log_return_")
    )


def _professional_bbo_name(name: str) -> str:
    if name.startswith("spot.book_"):
        suffix = name.removeprefix("spot.book_").replace(
            "book_update_count", "update_count"
        )
        return f"binance_spot.bbo_{suffix}"
    if name == "spot.book_receipt_age_ms":
        return "binance_spot.bbo_receipt_age_ms"
    if name.startswith("usdm.book_"):
        suffix = name.removeprefix("usdm.book_").replace(
            "book_update_count", "update_count"
        )
        return f"binance_usdm.bbo_{suffix}"
    if name == "usdm.book_receipt_age_ms":
        return "binance_usdm.bbo_receipt_age_ms"
    replacements = {
        "usdm.log_mid_basis": "binance_cross.usdm_minus_spot_log_mid_basis",
        "usdm.log_microprice_basis": (
            "binance_cross.usdm_minus_spot_log_microprice_basis"
        ),
        "usdm.spot_minus_usdm_book_receipt_skew_ms": (
            "binance_cross.spot_minus_usdm_bbo_receipt_skew_ms"
        ),
    }
    if name in replacements:
        return replacements[name]
    if name.startswith("usdm.spot_minus_usdm_mid_log_return_"):
        return "binance_cross." + name.removeprefix("usdm.")
    raise ValueError(f"Round 28 source feature is not BBO-derived: {name}")


_SOURCE_NAMES = (
    *POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    *POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
)
_SOURCE_BBO_INDEXES = tuple(
    index for index, name in enumerate(_SOURCE_NAMES) if _is_source_bbo_feature(name)
)
POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES = tuple(
    _professional_bbo_name(_SOURCE_NAMES[index]) for index in _SOURCE_BBO_INDEXES
)
POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES).encode("ascii")
).hexdigest()
POLYMARKET_ROUND28_FEATURE_NAMES = (
    *POLYMARKET_ROUND27_FEATURE_NAMES,
    *POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
)
POLYMARKET_ROUND28_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND28_FEATURE_NAMES).encode("ascii")
).hexdigest()

if (
    len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES) != 96
    or len(set(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)) != 96
    or set(POLYMARKET_ROUND27_FEATURE_NAMES)
    & set(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
):
    raise RuntimeError("Round 28 BBO feature contract differs")


def _selected_bbo_values(
    features: Round21OptionalBinanceFeatures,
) -> tuple[float, ...]:
    source_values = (*features.spot_values, *features.usdm_values)
    selected = tuple(float(source_values[index]) for index in _SOURCE_BBO_INDEXES)
    if (
        len(source_values) != len(_SOURCE_NAMES)
        or len(selected) != len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
        or any(not math.isfinite(value) for value in selected)
    ):
        raise ValueError("Round 28 BBO feature values differ")
    return selected


@dataclass(frozen=True, slots=True)
class Round28BookTickerOverlayRow:
    decision_time_ms: int
    values: tuple[float, ...]
    feature_names_sha256: str
    spot_source_chain_sha256: str
    usdm_source_chain_sha256: str
    maximum_receipt_wall_ms: int
    source_chain_sha256: str
    row_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        features: Round21OptionalBinanceFeatures,
    ) -> "Round28BookTickerOverlayRow":
        if not features.spot_available or not features.usdm_available:
            raise ValueError("Round 28 BBO source is unavailable at the decision")
        maximum_receipt = max(
            int(features.spot_maximum_receipt_ms),
            int(features.usdm_maximum_receipt_ms),
        )
        source_identity = {
            "decision_time_ms": int(features.decision_time_ms),
            "spot_source_chain_sha256": features.spot_source_chain_sha256,
            "usdm_source_chain_sha256": features.usdm_source_chain_sha256,
            "maximum_receipt_wall_ms": maximum_receipt,
        }
        payload = {
            "decision_time_ms": int(features.decision_time_ms),
            "values": _selected_bbo_values(features),
            "feature_names_sha256": (
                POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256
            ),
            "spot_source_chain_sha256": features.spot_source_chain_sha256,
            "usdm_source_chain_sha256": features.usdm_source_chain_sha256,
            "maximum_receipt_wall_ms": maximum_receipt,
            "source_chain_sha256": _canonical_sha256(source_identity),
            "target_accessed": False,
            "trading_authority": False,
        }
        return cls(
            decision_time_ms=int(features.decision_time_ms),
            values=_selected_bbo_values(features),
            feature_names_sha256=(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256),
            spot_source_chain_sha256=features.spot_source_chain_sha256,
            usdm_source_chain_sha256=features.usdm_source_chain_sha256,
            maximum_receipt_wall_ms=maximum_receipt,
            source_chain_sha256=_canonical_sha256(source_identity),
            row_sha256=_canonical_sha256(payload),
            target_accessed=False,
            trading_authority=False,
        ).validated()

    def validated(self) -> "Round28BookTickerOverlayRow":
        payload = {
            "decision_time_ms": self.decision_time_ms,
            "values": self.values,
            "feature_names_sha256": self.feature_names_sha256,
            "spot_source_chain_sha256": self.spot_source_chain_sha256,
            "usdm_source_chain_sha256": self.usdm_source_chain_sha256,
            "maximum_receipt_wall_ms": self.maximum_receipt_wall_ms,
            "source_chain_sha256": self.source_chain_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }
        source_identity = {
            "decision_time_ms": self.decision_time_ms,
            "spot_source_chain_sha256": self.spot_source_chain_sha256,
            "usdm_source_chain_sha256": self.usdm_source_chain_sha256,
            "maximum_receipt_wall_ms": self.maximum_receipt_wall_ms,
        }
        if (
            type(self.decision_time_ms) is not int
            or self.decision_time_ms <= 0
            or self.decision_time_ms % 1_000
            or len(self.values) != len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.feature_names_sha256
            != POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256
            or _SHA256.fullmatch(self.spot_source_chain_sha256) is None
            or _SHA256.fullmatch(self.usdm_source_chain_sha256) is None
            or self.spot_source_chain_sha256 == _EMPTY_SHA256
            or self.usdm_source_chain_sha256 == _EMPTY_SHA256
            or not 0 < self.maximum_receipt_wall_ms < self.decision_time_ms
            or self.source_chain_sha256 != _canonical_sha256(source_identity)
            or self.row_sha256 != _canonical_sha256(payload)
            or self.target_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 28 BBO overlay row differs")
        return self


@dataclass(frozen=True, slots=True)
class Round28FeatureRow:
    schema_version: str
    run_id: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    values: tuple[float, ...]
    feature_names_sha256: str
    maximum_receipt_wall_ms: int
    base_row_sha256: str
    overlay_row_sha256: str
    source_chain_sha256: str
    row_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        base: Round27FeatureRow,
        overlay: Round28BookTickerOverlayRow,
    ) -> "Round28FeatureRow":
        selected_base = base.validated()
        selected_overlay = overlay.validated()
        if selected_base.decision_time_ms != selected_overlay.decision_time_ms:
            raise ValueError("Round 28 feature decision identity differs")
        source_identity = {
            "base_row_sha256": selected_base.row_sha256,
            "overlay_row_sha256": selected_overlay.row_sha256,
        }
        payload = {
            "schema_version": POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION,
            "run_id": selected_base.run_id,
            "condition_id": selected_base.condition_id,
            "event_start_ms": selected_base.event_start_ms,
            "decision_time_ms": selected_base.decision_time_ms,
            "market_prior_probability": selected_base.market_prior_probability,
            "values": (*selected_base.values, *selected_overlay.values),
            "feature_names_sha256": POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
            "maximum_receipt_wall_ms": max(
                selected_base.maximum_receipt_wall_ms,
                selected_overlay.maximum_receipt_wall_ms,
            ),
            "base_row_sha256": selected_base.row_sha256,
            "overlay_row_sha256": selected_overlay.row_sha256,
            "source_chain_sha256": _canonical_sha256(source_identity),
            "target_accessed": False,
            "trading_authority": False,
        }
        return cls(
            schema_version=POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION,
            run_id=selected_base.run_id,
            condition_id=selected_base.condition_id,
            event_start_ms=selected_base.event_start_ms,
            decision_time_ms=selected_base.decision_time_ms,
            market_prior_probability=selected_base.market_prior_probability,
            values=(*selected_base.values, *selected_overlay.values),
            feature_names_sha256=POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
            maximum_receipt_wall_ms=max(
                selected_base.maximum_receipt_wall_ms,
                selected_overlay.maximum_receipt_wall_ms,
            ),
            base_row_sha256=selected_base.row_sha256,
            overlay_row_sha256=selected_overlay.row_sha256,
            source_chain_sha256=_canonical_sha256(source_identity),
            row_sha256=_canonical_sha256(payload),
            target_accessed=False,
            trading_authority=False,
        ).validated()

    def validated(self) -> "Round28FeatureRow":
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "market_prior_probability": self.market_prior_probability,
            "values": self.values,
            "feature_names_sha256": self.feature_names_sha256,
            "maximum_receipt_wall_ms": self.maximum_receipt_wall_ms,
            "base_row_sha256": self.base_row_sha256,
            "overlay_row_sha256": self.overlay_row_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }
        source_identity = {
            "base_row_sha256": self.base_row_sha256,
            "overlay_row_sha256": self.overlay_row_sha256,
        }
        if (
            self.schema_version != POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION
            or not self.run_id
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or self.decision_time_ms % 1_000
            or not 0.0 < self.market_prior_probability < 1.0
            or len(self.values) != len(POLYMARKET_ROUND28_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.feature_names_sha256 != POLYMARKET_ROUND28_FEATURE_NAMES_SHA256
            or not 0 < self.maximum_receipt_wall_ms < self.decision_time_ms
            or _SHA256.fullmatch(self.base_row_sha256) is None
            or _SHA256.fullmatch(self.overlay_row_sha256) is None
            or self.source_chain_sha256 != _canonical_sha256(source_identity)
            or self.row_sha256 != _canonical_sha256(payload)
            or self.target_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 28 feature row differs")
        return self


def _hash_chain(values: Sequence[str]) -> str:
    chain = _EMPTY_SHA256
    for value in values:
        chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(value)).hexdigest()
    return chain


def materialize_round28_book_ticker_overlay(
    *,
    base_rows: Sequence[Round27FeatureRow],
    sidecar_replay: Round21SidecarReplay,
) -> tuple[tuple[Round28BookTickerOverlayRow, ...], dict[str, object]]:
    """Build the incremental BBO overlay on a strictly matched decision grid."""

    selected_base = tuple(row.validated() for row in base_rows)
    replay = sidecar_replay.validated()
    decisions = tuple(row.decision_time_ms for row in selected_base)
    if (
        not selected_base
        or tuple(sorted(set(decisions))) != decisions
        or replay.decision_times_ms != decisions
    ):
        raise ValueError("Round 28 base and sidecar decision grids differ")
    overlay: list[Round28BookTickerOverlayRow] = []
    rejection_counts: dict[str, int] = {}
    for feature in replay.features:
        if not feature.spot_available:
            reason = "spot_bbo_unavailable_or_stale"
        elif not feature.usdm_available:
            reason = "usdm_bbo_unavailable_or_stale"
        else:
            overlay.append(Round28BookTickerOverlayRow.create(feature))
            continue
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    base_chain = _hash_chain(tuple(row.row_sha256 for row in selected_base))
    overlay_chain = _hash_chain(tuple(row.row_sha256 for row in overlay))
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_BOOK_TICKER_SCHEMA_VERSION,
        "round27_feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "book_ticker_feature_names_sha256": (
            POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256
        ),
        "book_ticker_feature_count": len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES),
        "round28_feature_names_sha256": POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
        "round28_feature_count": len(POLYMARKET_ROUND28_FEATURE_NAMES),
        "terminal_manifest_sha256": replay.terminal_manifest_sha256,
        "sidecar_receipt_chain_sha256": replay.receipt_chain_sha256,
        "sidecar_raw_message_count": replay.raw_message_count,
        "sidecar_stream_counts": dict(sorted(replay.stream_counts.items())),
        "sidecar_stream_gap_count": replay.stream_gap_count,
        "base_decision_count": len(selected_base),
        "accepted_decision_count": len(overlay),
        "rejected_decision_count": len(selected_base) - len(overlay),
        "accepted_fraction": len(overlay) / len(selected_base),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "base_row_chain_sha256": base_chain,
        "overlay_row_chain_sha256": overlay_chain,
        "receipt_time_causal": True,
        "bbo_only_incremental_features": True,
        "matched_base_and_augmented_population_required": True,
        "unavailable_or_stale_bbo_rows_excluded_from_both_candidates": True,
        "official_outcomes_accessed": False,
        "model_data_eligible": False,
        **_AUTHORITY,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return tuple(overlay), report


def compose_round28_feature_rows(
    *,
    base_rows: Sequence[Round27FeatureRow],
    overlay_rows: Sequence[Round28BookTickerOverlayRow],
    report: Mapping[str, object],
) -> tuple[Round28FeatureRow, ...]:
    """Create augmented rows while preserving a matched base-row population."""

    selected_base = tuple(row.validated() for row in base_rows)
    selected_overlay = tuple(row.validated() for row in overlay_rows)
    validated_report = validate_round28_book_ticker_report(report)
    base_by_decision = {row.decision_time_ms: row for row in selected_base}
    overlay_decisions = tuple(row.decision_time_ms for row in selected_overlay)
    if (
        len(base_by_decision) != len(selected_base)
        or tuple(sorted(set(overlay_decisions))) != overlay_decisions
        or any(decision not in base_by_decision for decision in overlay_decisions)
        or validated_report["base_row_chain_sha256"]
        != _hash_chain(tuple(row.row_sha256 for row in selected_base))
        or validated_report["overlay_row_chain_sha256"]
        != _hash_chain(tuple(row.row_sha256 for row in selected_overlay))
    ):
        raise ValueError("Round 28 feature composition population differs")
    return tuple(
        Round28FeatureRow.create(base_by_decision[row.decision_time_ms], row)
        for row in selected_overlay
    )


def validate_round28_book_ticker_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop("report_sha256", ""), name="report")
    expected_fields = {
        "schema_version",
        "round27_feature_names_sha256",
        "book_ticker_feature_names_sha256",
        "book_ticker_feature_count",
        "round28_feature_names_sha256",
        "round28_feature_count",
        "terminal_manifest_sha256",
        "sidecar_receipt_chain_sha256",
        "sidecar_raw_message_count",
        "sidecar_stream_counts",
        "sidecar_stream_gap_count",
        "base_decision_count",
        "accepted_decision_count",
        "rejected_decision_count",
        "accepted_fraction",
        "rejection_counts",
        "base_row_chain_sha256",
        "overlay_row_chain_sha256",
        "receipt_time_causal",
        "bbo_only_incremental_features",
        "matched_base_and_augmented_population_required",
        "unavailable_or_stale_bbo_rows_excluded_from_both_candidates",
        "official_outcomes_accessed",
        "model_data_eligible",
        *_AUTHORITY,
    }
    counts = payload.get("sidecar_stream_counts")
    rejected = payload.get("rejection_counts")
    try:
        base_count = _integer(payload["base_decision_count"], name="base count")
        accepted_count = _integer(
            payload["accepted_decision_count"], name="accepted count"
        )
        rejected_count = _integer(
            payload["rejected_decision_count"], name="rejected count"
        )
        accepted_fraction_value = payload["accepted_fraction"]
        if isinstance(accepted_fraction_value, bool) or not isinstance(
            accepted_fraction_value, (str, int, float)
        ):
            raise TypeError("accepted fraction is not numeric")
        accepted_fraction = float(accepted_fraction_value)
        raw_message_count = _integer(
            payload["sidecar_raw_message_count"], name="raw message count"
        )
        gap_count = _integer(
            payload["sidecar_stream_gap_count"], name="stream gap count"
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 28 BBO report counts differ") from exc
    if (
        set(payload) != expected_fields
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_BOOK_TICKER_SCHEMA_VERSION
        or payload.get("round27_feature_names_sha256")
        != POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
        or payload.get("book_ticker_feature_names_sha256")
        != POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256
        or payload.get("book_ticker_feature_count")
        != len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
        or payload.get("round28_feature_names_sha256")
        != POLYMARKET_ROUND28_FEATURE_NAMES_SHA256
        or payload.get("round28_feature_count") != len(POLYMARKET_ROUND28_FEATURE_NAMES)
        or _SHA256.fullmatch(str(payload.get("terminal_manifest_sha256", ""))) is None
        or _SHA256.fullmatch(str(payload.get("sidecar_receipt_chain_sha256", "")))
        is None
        or not isinstance(counts, Mapping)
        or set(counts) != {"binance_spot", "binance_futures"}
        or any(type(count) is not int or count <= 0 for count in counts.values())
        or sum(int(count) for count in counts.values()) != raw_message_count
        or raw_message_count <= 0
        or gap_count < 0
        or base_count <= 0
        or accepted_count < 0
        or rejected_count < 0
        or accepted_count + rejected_count != base_count
        or not math.isclose(
            accepted_fraction,
            accepted_count / base_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not isinstance(rejected, Mapping)
        or any(
            key
            not in {
                "spot_bbo_unavailable_or_stale",
                "usdm_bbo_unavailable_or_stale",
            }
            or type(count) is not int
            or count <= 0
            for key, count in rejected.items()
        )
        or sum(int(count) for count in rejected.values()) != rejected_count
        or _SHA256.fullmatch(str(payload.get("base_row_chain_sha256", ""))) is None
        or _SHA256.fullmatch(str(payload.get("overlay_row_chain_sha256", ""))) is None
        or payload.get("receipt_time_causal") is not True
        or payload.get("bbo_only_incremental_features") is not True
        or payload.get("matched_base_and_augmented_population_required") is not True
        or payload.get("unavailable_or_stale_bbo_rows_excluded_from_both_candidates")
        is not True
        or payload.get("official_outcomes_accessed") is not False
        or payload.get("model_data_eligible") is not False
        or any(payload.get(key) is not expected for key, expected in _AUTHORITY.items())
    ):
        raise ValueError("Round 28 BBO report differs")
    return {**payload, "report_sha256": claimed}


def _overlay_payload(row: Round28BookTickerOverlayRow) -> tuple[object, ...]:
    selected = row.validated()
    return (
        selected.decision_time_ms,
        list(selected.values),
        selected.feature_names_sha256,
        selected.spot_source_chain_sha256,
        selected.usdm_source_chain_sha256,
        selected.maximum_receipt_wall_ms,
        selected.source_chain_sha256,
        selected.row_sha256,
        False,
        False,
    )


def write_round28_book_ticker_overlay(
    path: str | Path,
    *,
    rows: Sequence[Round28BookTickerOverlayRow],
    report: Mapping[str, object],
) -> None:
    """Atomically persist only the incremental overlay in one DuckDB file."""

    destination = Path(path).resolve()
    selected = tuple(row.validated() for row in rows)
    validated_report = validate_round28_book_ticker_report(report)
    decisions = tuple(row.decision_time_ms for row in selected)
    if (
        not selected
        or tuple(sorted(set(decisions))) != decisions
        or validated_report["accepted_decision_count"] != len(selected)
        or validated_report["overlay_row_chain_sha256"]
        != _hash_chain(tuple(row.row_sha256 for row in selected))
        or destination.exists()
        or Path(f"{destination}.wal").exists()
    ):
        raise ValueError("Round 28 BBO overlay store inputs differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary_wal = Path(f"{temporary}.wal")
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(temporary))
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            """
            CREATE TABLE round28_book_ticker_overlay (
                decision_time_ms BIGINT PRIMARY KEY,
                feature_values DOUBLE[],
                feature_names_sha256 VARCHAR,
                spot_source_chain_sha256 VARCHAR,
                usdm_source_chain_sha256 VARCHAR,
                maximum_receipt_wall_ms BIGINT,
                source_chain_sha256 VARCHAR,
                row_sha256 VARCHAR,
                target_accessed BOOLEAN,
                trading_authority BOOLEAN
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE round28_book_ticker_manifest (
                schema_version VARCHAR PRIMARY KEY,
                report_json VARCHAR,
                report_sha256 VARCHAR,
                row_count BIGINT,
                row_chain_sha256 VARCHAR,
                target_accessed BOOLEAN,
                trading_authority BOOLEAN
            )
            """
        )
        connection.execute("BEGIN TRANSACTION")
        insert_rows_columnar(
            connection,
            sql="""
                INSERT INTO round28_book_ticker_overlay
                SELECT unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
                       unnest(?), unnest(?), unnest(?), unnest(?), unnest(?)
            """,
            rows=tuple(_overlay_payload(row) for row in selected),
            width=10,
            batch_size=4_096,
        )
        connection.execute(
            "INSERT INTO round28_book_ticker_manifest VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                POLYMARKET_ROUND28_BOOK_TICKER_STORE_SCHEMA_VERSION,
                _canonical_json(validated_report),
                validated_report["report_sha256"],
                len(selected),
                validated_report["overlay_row_chain_sha256"],
                False,
                False,
            ],
        )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        if temporary_wal.exists():
            raise RuntimeError("Round 28 BBO temporary WAL remains after checkpoint")
        os.replace(temporary, destination)
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        temporary_wal.unlink(missing_ok=True)
        raise


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 28 BBO store JSON has duplicate keys")
        output[key] = value
    return output


def load_round28_book_ticker_overlay(
    path: str | Path,
) -> tuple[tuple[Round28BookTickerOverlayRow, ...], dict[str, object]]:
    source = Path(path).resolve()
    if not source.is_file() or Path(f"{source}.wal").exists():
        raise ValueError("Round 28 BBO overlay store is unavailable or non-terminal")
    with duckdb.connect(str(source), read_only=True) as connection:
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        if tables != {
            "round28_book_ticker_manifest",
            "round28_book_ticker_overlay",
        }:
            raise ValueError("Round 28 BBO overlay store schema differs")
        manifest_rows = connection.execute(
            "SELECT * FROM round28_book_ticker_manifest"
        ).fetchall()
        stored_rows = connection.execute(
            """
            SELECT decision_time_ms, feature_values, feature_names_sha256,
                   spot_source_chain_sha256, usdm_source_chain_sha256,
                   maximum_receipt_wall_ms, source_chain_sha256, row_sha256,
                   target_accessed, trading_authority
            FROM round28_book_ticker_overlay ORDER BY decision_time_ms
            """
        ).fetchall()
    if len(manifest_rows) != 1:
        raise ValueError("Round 28 BBO overlay manifest population differs")
    manifest = manifest_rows[0]
    try:
        raw_report = json.loads(
            str(manifest[1]),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Round 28 BBO overlay report does not decode") from exc
    if not isinstance(raw_report, Mapping):
        raise ValueError("Round 28 BBO overlay report differs")
    report = validate_round28_book_ticker_report(raw_report)
    rows: list[Round28BookTickerOverlayRow] = []
    for stored in stored_rows:
        try:
            row = Round28BookTickerOverlayRow(
                decision_time_ms=int(stored[0]),
                values=tuple(float(value) for value in stored[1]),
                feature_names_sha256=str(stored[2]),
                spot_source_chain_sha256=str(stored[3]),
                usdm_source_chain_sha256=str(stored[4]),
                maximum_receipt_wall_ms=int(stored[5]),
                source_chain_sha256=str(stored[6]),
                row_sha256=str(stored[7]),
                target_accessed=bool(stored[8]),
                trading_authority=bool(stored[9]),
            ).validated()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 28 stored BBO row differs") from exc
        rows.append(row)
    selected = tuple(rows)
    if (
        manifest[0] != POLYMARKET_ROUND28_BOOK_TICKER_STORE_SCHEMA_VERSION
        or str(manifest[1]) != _canonical_json(report)
        or manifest[2] != report["report_sha256"]
        or int(manifest[3]) != len(selected)
        or int(manifest[3]) != report["accepted_decision_count"]
        or manifest[4] != _hash_chain(tuple(row.row_sha256 for row in selected))
        or manifest[4] != report["overlay_row_chain_sha256"]
        or manifest[5] is not False
        or manifest[6] is not False
    ):
        raise ValueError("Round 28 BBO overlay store audit differs")
    return selected, report


__all__ = [
    "POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES",
    "POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND28_BOOK_TICKER_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_BOOK_TICKER_STORE_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_FEATURE_NAMES",
    "POLYMARKET_ROUND28_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND28_FEATURE_SCHEMA_VERSION",
    "Round28BookTickerOverlayRow",
    "Round28FeatureRow",
    "compose_round28_feature_rows",
    "load_round28_book_ticker_overlay",
    "materialize_round28_book_ticker_overlay",
    "validate_round28_book_ticker_report",
    "write_round28_book_ticker_overlay",
]
