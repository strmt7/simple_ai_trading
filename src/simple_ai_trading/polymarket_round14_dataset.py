"""Outcome-blind admission and causal features for Polymarket Round 14."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_btc_reference import (
    PolymarketBtcEndpointEstimator,
    PolymarketBtcReferenceWindow,
    PolymarketChainlinkBtcTick,
    parse_polymarket_chainlink_btc_tick,
)
from .polymarket_external_signal import PolymarketBtcReferenceFeatures
from .polymarket_recorder import (
    DecodedPublicEvent,
    PolymarketEvidenceStore,
)
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
)
from .polymarket_round14_capture import validate_round14_capture_manifest
from .polymarket_round14_features import (
    POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
    PolymarketRound14FeatureRow,
    PolymarketRound14Snapshot,
    build_round14_snapshot_features,
)


POLYMARKET_ROUND14_ADMISSION_SPEC_SCHEMA_VERSION = (
    "polymarket-round14-admission-spec-v1"
)
POLYMARKET_ROUND14_CONDITION_ADMISSION_SCHEMA_VERSION = (
    "polymarket-round14-condition-admission-v1"
)
POLYMARKET_ROUND14_LABEL_FREE_DATASET_SCHEMA_VERSION = (
    "polymarket-round14-label-free-dataset-v1"
)

_EXPECTED_CONTRACT_SHA256 = (
    "60cde01112a749a9971447368b3a5d73b203d095e62a974327004c16cb021f1b"
)
_EXPECTED_CAMPAIGN_PLAN_SHA256 = (
    "c19ef7733efb86202742f045a45b3d92e8e17bb922c3c5f780240243889609b5"
)
_SHA256_LENGTH = 64
_BINANCE_STREAMS = ("binance_futures", "binance_spot")
_FEATURE_SOURCE_STREAMS = (
    "binance_futures",
    "binance_spot",
    "polymarket_rtds",
)
_TERMINAL_RUN_STATUSES = frozenset({"complete", "degraded"})


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _source_sha256(filename: str) -> str:
    try:
        payload = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(f"Round 14 source file is unavailable: {filename}") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _fraction(value: object, *, name: str) -> float:
    parsed = float(_positive_decimal(value, name=name))
    if not 0 < parsed <= 1:
        raise ValueError(f"{name} must lie in (0, 1]")
    return parsed


@dataclass(frozen=True, slots=True)
class PolymarketRound14AdmissionSpec:
    spec_sha256: str
    contract_sha256: str
    campaign_plan_sha256: str
    decision_cadence_ms: int
    decision_start_offset_ms: int
    decision_end_offset_ms: int
    exact_chainlink_open_maximum_receipt_delay_ms: int
    maximum_chainlink_source_age_ms: int
    maximum_receipt_age_ms: int
    maximum_source_future_ms: int
    minimum_row_coverage_fraction: float
    maximum_consecutive_missing_ms: int
    binance_minimum_row_coverage_fraction: float
    binance_maximum_consecutive_missing_ms: int
    binance_maximum_event_skew_ms: int
    binance_maximum_receive_skew_ms: int
    binance_maximum_source_age_ms: int
    binance_maximum_receipt_age_ms: int
    binance_return_horizon_ms: int
    payload: Mapping[str, object]

    def validated(self) -> "PolymarketRound14AdmissionSpec":
        if (
            not _is_sha256(self.spec_sha256)
            or self.contract_sha256 != _EXPECTED_CONTRACT_SHA256
            or self.campaign_plan_sha256 != _EXPECTED_CAMPAIGN_PLAN_SHA256
            or self.decision_cadence_ms != 250
            or self.decision_start_offset_ms != 30_000
            or self.decision_end_offset_ms != 2_000
            or self.exact_chainlink_open_maximum_receipt_delay_ms != 5_000
            or self.maximum_chainlink_source_age_ms != 3_500
            or self.maximum_receipt_age_ms != 1_500
            or self.maximum_source_future_ms != 250
            or self.minimum_row_coverage_fraction != 0.90
            or self.maximum_consecutive_missing_ms != 3_000
            or self.binance_minimum_row_coverage_fraction != 0.90
            or self.binance_maximum_consecutive_missing_ms != 3_000
            or self.binance_maximum_event_skew_ms != 1_000
            or self.binance_maximum_receive_skew_ms != 1_000
            or self.binance_maximum_source_age_ms != 1_500
            or self.binance_maximum_receipt_age_ms != 1_500
            or self.binance_return_horizon_ms != 1_000
        ):
            raise ValueError("Polymarket Round 14 admission spec drifted")
        payload = dict(self.payload)
        claimed = str(payload.pop("spec_sha256", "")).strip().lower()
        if claimed != self.spec_sha256 or _canonical_sha256(payload) != claimed:
            raise ValueError("Polymarket Round 14 admission spec hash differs")
        return self


def validate_round14_admission_spec(
    value: Mapping[str, object],
) -> PolymarketRound14AdmissionSpec:
    payload = dict(value)
    expected_keys = {
        "binance_layer",
        "campaign_plan_sha256",
        "condition_admission",
        "contract_sha256",
        "created_at_utc",
        "eligibility",
        "future_data_policy",
        "outcomes_consulted",
        "labels_consulted",
        "model_scores_consulted",
        "paper_trading_authority",
        "live_trading_authority",
        "profitability_claim",
        "schema_version",
        "spec_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Polymarket Round 14 admission spec schema differs")
    condition = payload["condition_admission"]
    binance = payload["binance_layer"]
    eligibility = payload["eligibility"]
    future = payload["future_data_policy"]
    if (
        payload["schema_version"] != POLYMARKET_ROUND14_ADMISSION_SPEC_SCHEMA_VERSION
        or not isinstance(condition, Mapping)
        or set(condition)
        != {
            "decision_cadence_ms",
            "decision_end_offset_ms",
            "decision_start_offset_ms",
            "exact_chainlink_open_maximum_receipt_delay_ms",
            "maximum_chainlink_source_age_ms",
            "maximum_consecutive_missing_ms",
            "maximum_receipt_age_ms",
            "maximum_source_future_ms",
            "minimum_row_coverage_fraction",
            "required_core_streams",
        }
        or condition["required_core_streams"] != ["clob_market", "polymarket_rtds"]
        or not isinstance(binance, Mapping)
        or set(binance)
        != {
            "maximum_consecutive_missing_ms",
            "maximum_event_skew_ms",
            "maximum_receipt_age_ms",
            "maximum_receive_skew_ms",
            "maximum_source_age_ms",
            "minimum_row_coverage_fraction",
            "required_streams",
            "return_horizon_ms",
            "zero_trade_sentinel_policy",
        }
        or binance["required_streams"] != list(_BINANCE_STREAMS)
        or binance["zero_trade_sentinel_policy"]
        != "discard_and_count_only_price_0_quantity_0_order_type_NA_status_1"
        or not isinstance(eligibility, Mapping)
        or eligibility
        != {
            "allow_condition_local_recovery_from_degraded_runs": True,
            "allow_qualification_capture_for_diagnostics_only": True,
            "purpose": "prospective",
            "required_asset": "BTC",
            "required_event_duration_ms": 300_000,
            "required_manifest_model_data_eligible": True,
            "required_terminal_run_statuses": ["complete", "degraded"],
        }
        or not isinstance(future, Mapping)
        or future
        != {
            "feature_replay_include_resolutions": False,
            "feature_rows_may_read_close_price": False,
            "feature_rows_may_read_events_after_decision": False,
            "feature_rows_may_read_labels": False,
            "label_acquisition_is_separate": True,
            "resolution_tables_may_be_read": False,
        }
        or any(
            payload[name] is not False
            for name in (
                "outcomes_consulted",
                "labels_consulted",
                "model_scores_consulted",
                "paper_trading_authority",
                "live_trading_authority",
                "profitability_claim",
            )
        )
    ):
        raise ValueError("Polymarket Round 14 admission semantics drifted")
    return PolymarketRound14AdmissionSpec(
        spec_sha256=str(payload["spec_sha256"]).strip().lower(),
        contract_sha256=str(payload["contract_sha256"]).strip().lower(),
        campaign_plan_sha256=str(payload["campaign_plan_sha256"]).strip().lower(),
        decision_cadence_ms=int(condition["decision_cadence_ms"]),
        decision_start_offset_ms=int(condition["decision_start_offset_ms"]),
        decision_end_offset_ms=int(condition["decision_end_offset_ms"]),
        exact_chainlink_open_maximum_receipt_delay_ms=int(
            condition["exact_chainlink_open_maximum_receipt_delay_ms"]
        ),
        maximum_chainlink_source_age_ms=int(
            condition["maximum_chainlink_source_age_ms"]
        ),
        maximum_receipt_age_ms=int(condition["maximum_receipt_age_ms"]),
        maximum_source_future_ms=int(condition["maximum_source_future_ms"]),
        minimum_row_coverage_fraction=_fraction(
            condition["minimum_row_coverage_fraction"],
            name="minimum row coverage",
        ),
        maximum_consecutive_missing_ms=int(condition["maximum_consecutive_missing_ms"]),
        binance_minimum_row_coverage_fraction=_fraction(
            binance["minimum_row_coverage_fraction"],
            name="Binance minimum row coverage",
        ),
        binance_maximum_consecutive_missing_ms=int(
            binance["maximum_consecutive_missing_ms"]
        ),
        binance_maximum_event_skew_ms=int(binance["maximum_event_skew_ms"]),
        binance_maximum_receive_skew_ms=int(binance["maximum_receive_skew_ms"]),
        binance_maximum_source_age_ms=int(binance["maximum_source_age_ms"]),
        binance_maximum_receipt_age_ms=int(binance["maximum_receipt_age_ms"]),
        binance_return_horizon_ms=int(binance["return_horizon_ms"]),
        payload=payload,
    ).validated()


def load_round14_admission_spec(
    path: str | Path,
) -> PolymarketRound14AdmissionSpec:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket Round 14 admission spec is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Polymarket Round 14 admission spec is not an object")
    return validate_round14_admission_spec(payload)


@dataclass(frozen=True, slots=True)
class PolymarketRound14ConditionAdmission:
    run_id: str
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    candidate_row_count: int
    materialized_row_count: int
    external_row_count: int
    row_coverage_fraction: float
    external_coverage_fraction: float
    maximum_consecutive_missing_ms: int
    external_maximum_consecutive_missing_ms: int
    chainlink_tick_count: int
    spot_bbo_count: int
    futures_bbo_count: int
    spot_trade_count: int
    futures_trade_count: int
    ignored_futures_zero_trade_count: int
    exact_chainlink_open_event_sha256: str
    row_manifest_sha256: str
    core_eligible: bool
    binance_layer_eligible: bool
    reasons: tuple[str, ...]
    binance_reasons: tuple[str, ...]
    admission_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("admission_sha256")
        payload["reasons"] = list(self.reasons)
        payload["binance_reasons"] = list(self.binance_reasons)
        return {
            "schema_version": (POLYMARKET_ROUND14_CONDITION_ADMISSION_SCHEMA_VERSION),
            **payload,
            "labels_consulted": False,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "training_authority": False,
            "trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "admission_sha256": self.admission_sha256}

    def validated(self) -> "PolymarketRound14ConditionAdmission":
        fractions = (self.row_coverage_fraction, self.external_coverage_fraction)
        if (
            not self.run_id
            or not self.condition_id
            or self.event_start_ms <= 0
            or self.event_end_ms - self.event_start_ms != 300_000
            or min(
                self.candidate_row_count,
                self.materialized_row_count,
                self.external_row_count,
                self.maximum_consecutive_missing_ms,
                self.external_maximum_consecutive_missing_ms,
                self.chainlink_tick_count,
                self.spot_bbo_count,
                self.futures_bbo_count,
                self.spot_trade_count,
                self.futures_trade_count,
                self.ignored_futures_zero_trade_count,
            )
            < 0
            or self.materialized_row_count > self.candidate_row_count
            or self.external_row_count > self.materialized_row_count
            or any(
                not math.isfinite(value) or not 0 <= value <= 1 for value in fractions
            )
            or tuple(sorted(set(self.reasons))) != self.reasons
            or tuple(sorted(set(self.binance_reasons))) != self.binance_reasons
            or self.core_eligible == bool(self.reasons)
            or self.binance_layer_eligible
            != (self.core_eligible and not self.binance_reasons)
            or (
                self.exact_chainlink_open_event_sha256
                and not _is_sha256(self.exact_chainlink_open_event_sha256)
            )
            or not _is_sha256(self.row_manifest_sha256)
            or not _is_sha256(self.admission_sha256)
            or self.admission_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 14 condition admission is invalid")
        return self


@dataclass(frozen=True, slots=True)
class PolymarketRound14LabelFreeDataset:
    run_id: str
    run_report_sha256: str
    capture_manifest_sha256: str
    admission_spec_sha256: str
    feature_names_sha256: str
    materializer_source_sha256: str
    feature_builder_source_sha256: str
    replay_source_sha256: str
    diagnostic_only: bool
    admissions: tuple[PolymarketRound14ConditionAdmission, ...]
    rows: tuple[PolymarketRound14FeatureRow, ...]
    dataset_sha256: str

    @property
    def admitted_condition_count(self) -> int:
        return sum(item.core_eligible for item in self.admissions)

    @property
    def binance_layer_condition_count(self) -> int:
        return sum(item.binance_layer_eligible for item in self.admissions)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND14_LABEL_FREE_DATASET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_report_sha256": self.run_report_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "admission_spec_sha256": self.admission_spec_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "materializer_source_sha256": self.materializer_source_sha256,
            "feature_builder_source_sha256": self.feature_builder_source_sha256,
            "replay_source_sha256": self.replay_source_sha256,
            "diagnostic_only": self.diagnostic_only,
            "admissions": [item.asdict() for item in self.admissions],
            "row_input_sha256s": [row.input_sha256 for row in self.rows],
            "admitted_condition_count": self.admitted_condition_count,
            "binance_layer_condition_count": self.binance_layer_condition_count,
            "row_count": len(self.rows),
            "labels_consulted": False,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "training_authority": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "profitability_claim": False,
        }

    def asdict(self, *, include_rows: bool = False) -> dict[str, object]:
        payload = {
            **self.identity_payload(),
            "dataset_sha256": self.dataset_sha256,
        }
        if include_rows:
            payload["rows"] = [row.asdict() for row in self.rows]
        return payload

    def validated(self) -> "PolymarketRound14LabelFreeDataset":
        for admission in self.admissions:
            admission.validated()
        if (
            not self.run_id
            or not _is_sha256(self.run_report_sha256)
            or not _is_sha256(self.capture_manifest_sha256)
            or not _is_sha256(self.admission_spec_sha256)
            or self.feature_names_sha256 != POLYMARKET_ROUND14_FEATURE_NAMES_SHA256
            or not _is_sha256(self.materializer_source_sha256)
            or not _is_sha256(self.feature_builder_source_sha256)
            or not _is_sha256(self.replay_source_sha256)
            or tuple(
                sorted(
                    self.admissions,
                    key=lambda item: (item.event_start_ms, item.condition_id),
                )
            )
            != self.admissions
            or any(
                row.condition_id not in {item.condition_id for item in self.admissions}
                for row in self.rows
            )
            or not _is_sha256(self.dataset_sha256)
            or self.dataset_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 14 label-free dataset is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _Bbo:
    source: str
    event_time_ms: int
    received_at_ms: int
    bid: Decimal
    ask: Decimal


@dataclass(frozen=True, slots=True)
class _Trade:
    source: str
    event_time_ms: int
    received_at_ms: int
    price: Decimal


@dataclass(frozen=True, slots=True)
class _ParsedConditionSources:
    chainlink_ticks: tuple[PolymarketChainlinkBtcTick, ...]
    chainlink_receipts: tuple[int, ...]
    spot_bbos: tuple[_Bbo, ...]
    spot_bbo_receipts: tuple[int, ...]
    futures_bbos: tuple[_Bbo, ...]
    futures_bbo_receipts: tuple[int, ...]
    spot_trades: tuple[_Trade, ...]
    spot_trade_receipts: tuple[int, ...]
    futures_trades: tuple[_Trade, ...]
    futures_trade_receipts: tuple[int, ...]
    ignored_futures_zero_trade_count: int
    core_errors: tuple[str, ...]
    binance_errors: tuple[str, ...]


def _latest_at_or_before(
    values: Sequence[object],
    receipts: Sequence[int],
    observed_at_ms: int,
) -> object | None:
    index = bisect_right(receipts, observed_at_ms) - 1
    return None if index < 0 else values[index]


def _maximum_missing_ms(flags: Sequence[bool], cadence_ms: int) -> int:
    maximum = 0
    current = 0
    for available in flags:
        if available:
            current = 0
        else:
            current += cadence_ms
            maximum = max(maximum, current)
    return maximum


def _validate_chainlink_bootstrap(event: DecodedPublicEvent) -> None:
    envelope = event.event
    if set(envelope) != {"payload", "timestamp", "topic", "type"}:
        raise ValueError("Chainlink bootstrap envelope schema drifted")
    if (
        envelope["type"] != "subscribe"
        or envelope["topic"] != "crypto_prices"
        or int(envelope["timestamp"]) > event.received_wall_ms + 5_000
    ):
        raise ValueError("Chainlink bootstrap envelope identity drifted")
    body = envelope["payload"]
    if not isinstance(body, Mapping) or set(body) != {"data", "symbol"}:
        raise ValueError("Chainlink bootstrap payload schema drifted")
    if str(body["symbol"]).strip().lower() != "btc/usd":
        raise ValueError("Chainlink bootstrap is not BTC/USD")
    points = body["data"]
    if not isinstance(points, list) or not points:
        raise ValueError("Chainlink bootstrap contains no observations")
    previous = -1
    publisher_time_ms = int(envelope["timestamp"])
    for point in points:
        if not isinstance(point, Mapping) or set(point) != {"timestamp", "value"}:
            raise ValueError("Chainlink bootstrap observation schema drifted")
        source_time_ms = int(point["timestamp"])
        _positive_decimal(point["value"], name="Chainlink bootstrap value")
        if source_time_ms <= previous or source_time_ms > publisher_time_ms:
            raise ValueError("Chainlink bootstrap chronology drifted")
        previous = source_time_ms


def _binance_payload(
    event: DecodedPublicEvent,
    *,
    expected_stream: str,
    expected_suffix: str,
) -> Mapping[str, object]:
    envelope = event.event
    if set(envelope) != {"data", "stream"}:
        raise ValueError("Binance combined-stream envelope schema drifted")
    stream_name = str(envelope["stream"]).lower()
    if stream_name != f"btcusdt@{expected_suffix}":
        raise ValueError("Binance BTC stream identity drifted")
    body = envelope["data"]
    if not isinstance(body, Mapping) or str(body.get("s") or "") != "BTCUSDT":
        raise ValueError("Binance payload is not BTCUSDT")
    if event.stream != expected_stream:
        raise ValueError("Binance recorder stream identity drifted")
    return body


def _parse_bbo(event: DecodedPublicEvent) -> _Bbo:
    if event.stream == "binance_spot":
        body = _binance_payload(
            event,
            expected_stream="binance_spot",
            expected_suffix="ticker",
        )
        if body.get("e") != "24hrTicker":
            raise ValueError("Binance spot ticker event type drifted")
        bid = _positive_decimal(body.get("b"), name="Binance spot bid")
        ask = _positive_decimal(body.get("a"), name="Binance spot ask")
        source = "spot"
    elif event.stream == "binance_futures":
        body = _binance_payload(
            event,
            expected_stream="binance_futures",
            expected_suffix="depth5@100ms",
        )
        if body.get("e") != "depthUpdate":
            raise ValueError("Binance futures depth event type drifted")
        bids = body.get("b")
        asks = body.get("a")
        if (
            not isinstance(bids, list)
            or not bids
            or not isinstance(asks, list)
            or not asks
            or not isinstance(bids[0], list)
            or len(bids[0]) != 2
            or not isinstance(asks[0], list)
            or len(asks[0]) != 2
        ):
            raise ValueError("Binance futures depth ladder drifted")
        bid = _positive_decimal(bids[0][0], name="Binance futures bid")
        ask = _positive_decimal(asks[0][0], name="Binance futures ask")
        source = "futures"
    else:
        raise ValueError("unsupported Binance BBO stream")
    if ask <= bid:
        raise ValueError("Binance BBO is crossed or locked")
    source_time = int(event.source_time_ms or -1)
    if source_time <= 0:
        raise ValueError("Binance BBO has no source timestamp")
    return _Bbo(
        source=source,
        event_time_ms=source_time,
        received_at_ms=event.received_wall_ms,
        bid=bid,
        ask=ask,
    )


def _parse_trade(event: DecodedPublicEvent) -> _Trade | None:
    expected_stream = event.stream
    body = _binance_payload(
        event,
        expected_stream=expected_stream,
        expected_suffix="trade",
    )
    if body.get("e") != "trade":
        raise ValueError("Binance trade event type drifted")
    source_time = int(event.source_time_ms or -1)
    if source_time <= 0:
        raise ValueError("Binance trade has no source timestamp")
    raw_price = Decimal(str(body.get("p")))
    raw_quantity = Decimal(str(body.get("q")))
    if (
        expected_stream == "binance_futures"
        and raw_price == 0
        and raw_quantity == 0
        and body.get("X") == "NA"
        and body.get("st") == 1
    ):
        return None
    price = _positive_decimal(raw_price, name="Binance trade price")
    _positive_decimal(raw_quantity, name="Binance trade quantity")
    return _Trade(
        source="spot" if expected_stream == "binance_spot" else "futures",
        event_time_ms=source_time,
        received_at_ms=event.received_wall_ms,
        price=price,
    )


def _parse_condition_sources(
    events: Sequence[DecodedPublicEvent],
) -> _ParsedConditionSources:
    chainlink: list[PolymarketChainlinkBtcTick] = []
    spot_bbos: list[_Bbo] = []
    futures_bbos: list[_Bbo] = []
    spot_trades: list[_Trade] = []
    futures_trades: list[_Trade] = []
    core_errors: list[str] = []
    binance_errors: list[str] = []
    ignored_futures_zero_trade_count = 0
    for event in events:
        try:
            if event.stream == "polymarket_rtds":
                if event.event_type == "crypto_prices_chainlink:subscribe":
                    _validate_chainlink_bootstrap(event)
                elif event.event_type == "crypto_prices_chainlink:update":
                    body = event.event.get("payload")
                    if (
                        not isinstance(body, Mapping)
                        or "full_accuracy_value" not in body
                    ):
                        raise ValueError("Chainlink live update is not exact precision")
                    chainlink.append(
                        parse_polymarket_chainlink_btc_tick(
                            event.event,
                            received_at_ms=event.received_wall_ms,
                        )
                    )
            elif event.stream == "binance_spot":
                if event.event_type == "24hrTicker":
                    spot_bbos.append(_parse_bbo(event))
                elif event.event_type == "trade":
                    trade = _parse_trade(event)
                    if trade is None:
                        raise ValueError("spot stream emitted a futures sentinel")
                    spot_trades.append(trade)
            elif event.stream == "binance_futures":
                if event.event_type == "depthUpdate":
                    futures_bbos.append(_parse_bbo(event))
                elif event.event_type == "trade":
                    trade = _parse_trade(event)
                    if trade is None:
                        ignored_futures_zero_trade_count += 1
                    else:
                        futures_trades.append(trade)
        except (ArithmeticError, TypeError, ValueError) as exc:
            reason = f"{event.stream}_schema_or_chronology:{exc.__class__.__name__}"
            if event.stream == "polymarket_rtds":
                core_errors.append(reason)
            else:
                binance_errors.append(reason)

    chainlink.sort(key=lambda item: (item.received_at_ms, item.source_time_ms))
    spot_bbos.sort(key=lambda item: (item.received_at_ms, item.event_time_ms))
    futures_bbos.sort(key=lambda item: (item.received_at_ms, item.event_time_ms))
    spot_trades.sort(key=lambda item: (item.received_at_ms, item.event_time_ms))
    futures_trades.sort(key=lambda item: (item.received_at_ms, item.event_time_ms))
    for current, following in zip(chainlink, chainlink[1:], strict=False):
        if following.source_time_ms < current.source_time_ms or (
            following.source_time_ms == current.source_time_ms
            and following.price != current.price
        ):
            core_errors.append("chainlink_source_chronology_or_value_conflict")
            break
    return _ParsedConditionSources(
        chainlink_ticks=tuple(chainlink),
        chainlink_receipts=tuple(item.received_at_ms for item in chainlink),
        spot_bbos=tuple(spot_bbos),
        spot_bbo_receipts=tuple(item.received_at_ms for item in spot_bbos),
        futures_bbos=tuple(futures_bbos),
        futures_bbo_receipts=tuple(item.received_at_ms for item in futures_bbos),
        spot_trades=tuple(spot_trades),
        spot_trade_receipts=tuple(item.received_at_ms for item in spot_trades),
        futures_trades=tuple(futures_trades),
        futures_trade_receipts=tuple(item.received_at_ms for item in futures_trades),
        ignored_futures_zero_trade_count=ignored_futures_zero_trade_count,
        core_errors=tuple(sorted(set(core_errors))),
        binance_errors=tuple(sorted(set(binance_errors))),
    )


def _external_features(
    sources: _ParsedConditionSources,
    *,
    decision_time_ms: int,
    spec: PolymarketRound14AdmissionSpec,
) -> PolymarketBtcReferenceFeatures | None:
    spot_bbo = _latest_at_or_before(
        sources.spot_bbos,
        sources.spot_bbo_receipts,
        decision_time_ms,
    )
    futures_bbo = _latest_at_or_before(
        sources.futures_bbos,
        sources.futures_bbo_receipts,
        decision_time_ms,
    )
    spot_trade = _latest_at_or_before(
        sources.spot_trades,
        sources.spot_trade_receipts,
        decision_time_ms,
    )
    futures_trade = _latest_at_or_before(
        sources.futures_trades,
        sources.futures_trade_receipts,
        decision_time_ms,
    )
    prior_time = decision_time_ms - spec.binance_return_horizon_ms
    prior_spot = _latest_at_or_before(
        sources.spot_trades,
        sources.spot_trade_receipts,
        prior_time,
    )
    prior_futures = _latest_at_or_before(
        sources.futures_trades,
        sources.futures_trade_receipts,
        prior_time,
    )
    if any(
        value is None
        for value in (
            spot_bbo,
            futures_bbo,
            spot_trade,
            futures_trade,
            prior_spot,
            prior_futures,
        )
    ):
        return None
    assert isinstance(spot_bbo, _Bbo)
    assert isinstance(futures_bbo, _Bbo)
    assert isinstance(spot_trade, _Trade)
    assert isinstance(futures_trade, _Trade)
    assert isinstance(prior_spot, _Trade)
    assert isinstance(prior_futures, _Trade)
    used = (spot_bbo, futures_bbo, spot_trade, futures_trade)
    if any(
        decision_time_ms - item.received_at_ms > spec.binance_maximum_receipt_age_ms
        or decision_time_ms - item.event_time_ms > spec.binance_maximum_source_age_ms
        or item.event_time_ms - decision_time_ms > spec.maximum_source_future_ms
        for item in used
    ) or any(
        prior_time - item.received_at_ms > spec.binance_maximum_receipt_age_ms
        for item in (prior_spot, prior_futures)
    ):
        return None
    event_skew = abs(spot_bbo.event_time_ms - futures_bbo.event_time_ms)
    receive_skew = abs(spot_bbo.received_at_ms - futures_bbo.received_at_ms)
    if (
        event_skew > spec.binance_maximum_event_skew_ms
        or receive_skew > spec.binance_maximum_receive_skew_ms
    ):
        return None
    spot_mid = (spot_bbo.bid + spot_bbo.ask) / Decimal("2")
    futures_mid = (futures_bbo.bid + futures_bbo.ask) / Decimal("2")
    return PolymarketBtcReferenceFeatures(
        observed_at_ms=max(item.received_at_ms for item in used),
        spot_mid=spot_mid,
        futures_mid=futures_mid,
        spot_spread_bps=(spot_bbo.ask - spot_bbo.bid) / spot_mid * Decimal("10000"),
        futures_spread_bps=(futures_bbo.ask - futures_bbo.bid)
        / futures_mid
        * Decimal("10000"),
        futures_basis_bps=(futures_mid / spot_mid - 1) * Decimal("10000"),
        spot_log_return=math.log(float(spot_trade.price / prior_spot.price)),
        futures_log_return=math.log(float(futures_trade.price / prior_futures.price)),
        event_time_skew_ms=event_skew,
        receive_time_skew_ms=receive_skew,
    )


def _condition_gaps(
    gap_rows: Sequence[Sequence[object]],
    *,
    streams: frozenset[str],
    start_ms: int,
    end_ms: int,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"stream_gap:{str(row[0])}"
                for row in gap_rows
                if str(row[0]) in streams and start_ms <= int(str(row[1])) <= end_ms
            }
        )
    )


def _finalize_admission(
    provisional: PolymarketRound14ConditionAdmission,
) -> PolymarketRound14ConditionAdmission:
    return replace(
        provisional,
        admission_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _build_condition(
    *,
    store: PolymarketEvidenceStore,
    run_id: str,
    market: PolymarketFiveMinuteMarket,
    replay_books: Mapping[str, tuple[PolymarketRecordedBook, ...]],
    gap_rows: Sequence[Sequence[object]],
    global_reasons: Sequence[str],
    spec: PolymarketRound14AdmissionSpec,
) -> tuple[
    PolymarketRound14ConditionAdmission,
    tuple[PolymarketRound14FeatureRow, ...],
]:
    decision_start = market.event_start_ms + spec.decision_start_offset_ms
    decision_end = market.end_ms - spec.decision_end_offset_ms
    decisions = tuple(range(decision_start, decision_end + 1, spec.decision_cadence_ms))
    reasons = list(global_reasons)
    binance_reasons: list[str] = []
    if (
        market.asset != "BTC"
        or market.event_start_ms % 300_000
        or market.end_ms - market.event_start_ms != 300_000
    ):
        reasons.append("market_identity_or_interval_invalid")
    reasons.extend(
        _condition_gaps(
            gap_rows,
            streams=frozenset({"clob_market", "polymarket_rtds"}),
            start_ms=market.event_start_ms,
            end_ms=decision_end,
        )
    )
    binance_reasons.extend(
        _condition_gaps(
            gap_rows,
            streams=frozenset(_BINANCE_STREAMS),
            start_ms=market.event_start_ms,
            end_ms=decision_end,
        )
    )
    events = tuple(
        store.iter_public_events(
            run_id,
            streams=_FEATURE_SOURCE_STREAMS,
            ordered=True,
            verified_source=True,
            minimum_received_wall_ms=max(
                0,
                market.event_start_ms - 180_000,
            ),
            maximum_received_wall_ms=decision_end,
        )
    )
    sources = _parse_condition_sources(events)
    reasons.extend(sources.core_errors)
    binance_reasons.extend(sources.binance_errors)
    opening_candidates = tuple(
        tick
        for tick in sources.chainlink_ticks
        if tick.source_time_ms == market.event_start_ms
        and tick.received_at_ms
        <= market.event_start_ms + spec.exact_chainlink_open_maximum_receipt_delay_ms
    )
    opening = (
        min(opening_candidates, key=lambda tick: tick.received_at_ms)
        if opening_candidates
        else None
    )
    if opening is None:
        reasons.append("missing_exact_chainlink_open")
    elif any(tick.price != opening.price for tick in opening_candidates):
        reasons.append("contradictory_exact_chainlink_open")
    up_books = replay_books.get(market.up_token_id, ())
    down_books = replay_books.get(market.down_token_id, ())
    if not up_books:
        reasons.append("missing_up_book_replay")
    if not down_books:
        reasons.append("missing_down_book_replay")

    rows: list[PolymarketRound14FeatureRow] = []
    core_flags: list[bool] = []
    external_flags: list[bool] = []
    estimator = PolymarketBtcEndpointEstimator(
        maximum_staleness_ms=spec.maximum_chainlink_source_age_ms
    )
    tick_index = 0
    latest_tick: PolymarketChainlinkBtcTick | None = None
    chronology_failed = False
    up_receipts = [book.received_wall_ms for book in up_books]
    down_receipts = [book.received_wall_ms for book in down_books]
    reference = (
        None
        if opening is None
        else PolymarketBtcReferenceWindow(
            event_start_ms=market.event_start_ms,
            end_ms=market.end_ms,
            open_price=opening.price,
            close_price=None,
            observed_at_ms=opening.received_at_ms,
            completed=False,
            incomplete=True,
            cached=False,
            source_payload_sha256=opening.source_payload_sha256,
        )
    )
    for decision in decisions:
        while (
            tick_index < len(sources.chainlink_ticks)
            and sources.chainlink_ticks[tick_index].received_at_ms <= decision
        ):
            tick = sources.chainlink_ticks[tick_index]
            tick_index += 1
            try:
                if estimator.observe(tick):
                    latest_tick = tick
            except ValueError:
                chronology_failed = True
                break
        if chronology_failed:
            core_flags.extend([False] * (len(decisions) - len(core_flags)))
            external_flags.extend([False] * (len(decisions) - len(external_flags)))
            break
        up_recorded = _latest_at_or_before(up_books, up_receipts, decision)
        down_recorded = _latest_at_or_before(down_books, down_receipts, decision)
        if (
            reference is None
            or latest_tick is None
            or not isinstance(up_recorded, PolymarketRecordedBook)
            or not isinstance(down_recorded, PolymarketRecordedBook)
        ):
            core_flags.append(False)
            external_flags.append(False)
            continue
        estimate = estimator.estimate(reference, observed_at_ms=decision)
        external = _external_features(
            sources,
            decision_time_ms=decision,
            spec=spec,
        )
        try:
            row = build_round14_snapshot_features(
                PolymarketRound14Snapshot(
                    condition_id=market.condition_id,
                    up_token_id=market.up_token_id,
                    down_token_id=market.down_token_id,
                    event_start_ms=market.event_start_ms,
                    event_end_ms=market.end_ms,
                    decision_time_ms=decision,
                    reference=reference,
                    chainlink_tick=latest_tick,
                    structural_estimate=estimate,
                    up_book=up_recorded.snapshot,
                    down_book=down_recorded.snapshot,
                    external_features=external,
                    maximum_source_age_ms=spec.maximum_receipt_age_ms,
                )
            )
        except (ArithmeticError, TypeError, ValueError):
            core_flags.append(False)
            external_flags.append(False)
            continue
        rows.append(row)
        core_flags.append(True)
        external_flags.append(external is not None)

    if chronology_failed:
        reasons.append("chainlink_source_chronology_regressed")
    row_count = len(rows)
    candidate_count = len(decisions)
    external_count = sum(external_flags)
    row_coverage = 0.0 if not candidate_count else row_count / candidate_count
    external_coverage = 0.0 if not row_count else external_count / row_count
    maximum_missing = _maximum_missing_ms(
        core_flags,
        spec.decision_cadence_ms,
    )
    external_maximum_missing = _maximum_missing_ms(
        external_flags,
        spec.decision_cadence_ms,
    )
    if row_coverage < spec.minimum_row_coverage_fraction:
        reasons.append("insufficient_causal_feature_coverage")
    if maximum_missing > spec.maximum_consecutive_missing_ms:
        reasons.append("causal_feature_gap_exceeds_limit")
    if not sources.spot_bbos:
        binance_reasons.append("missing_spot_bbo")
    if not sources.futures_bbos:
        binance_reasons.append("missing_futures_bbo")
    if not sources.spot_trades:
        binance_reasons.append("missing_spot_trades")
    if not sources.futures_trades:
        binance_reasons.append("missing_futures_trades")
    if external_coverage < spec.binance_minimum_row_coverage_fraction:
        binance_reasons.append("insufficient_binance_feature_coverage")
    if external_maximum_missing > spec.binance_maximum_consecutive_missing_ms:
        binance_reasons.append("binance_feature_gap_exceeds_limit")

    normalized_reasons = tuple(sorted(set(reasons)))
    normalized_binance_reasons = tuple(sorted(set(binance_reasons)))
    row_manifest = _canonical_sha256(
        {
            "condition_id": market.condition_id,
            "feature_names_sha256": POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
            "row_input_sha256s": [row.input_sha256 for row in rows],
        }
    )
    provisional = PolymarketRound14ConditionAdmission(
        run_id=run_id,
        condition_id=market.condition_id,
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        candidate_row_count=candidate_count,
        materialized_row_count=row_count,
        external_row_count=external_count,
        row_coverage_fraction=row_coverage,
        external_coverage_fraction=external_coverage,
        maximum_consecutive_missing_ms=maximum_missing,
        external_maximum_consecutive_missing_ms=external_maximum_missing,
        chainlink_tick_count=len(sources.chainlink_ticks),
        spot_bbo_count=len(sources.spot_bbos),
        futures_bbo_count=len(sources.futures_bbos),
        spot_trade_count=len(sources.spot_trades),
        futures_trade_count=len(sources.futures_trades),
        ignored_futures_zero_trade_count=(sources.ignored_futures_zero_trade_count),
        exact_chainlink_open_event_sha256=(
            "" if opening is None else opening.source_payload_sha256
        ),
        row_manifest_sha256=row_manifest,
        core_eligible=not normalized_reasons,
        binance_layer_eligible=(
            not normalized_reasons and not normalized_binance_reasons
        ),
        reasons=normalized_reasons,
        binance_reasons=normalized_binance_reasons,
        admission_sha256="0" * 64,
    )
    return _finalize_admission(provisional), tuple(rows)


def materialize_round14_label_free_run(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    spec: PolymarketRound14AdmissionSpec,
    diagnostic_only: bool = False,
) -> PolymarketRound14LabelFreeDataset:
    """Build target-free rows from one immutable terminal capture unit."""

    spec = spec.validated()
    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Round 14 label-free materialization requires a run ID")
    if not store.read_only:
        raise ValueError(
            "Round 14 label-free materialization requires a read-only store"
        )
    if not isinstance(diagnostic_only, bool):
        raise ValueError("diagnostic_only must be a boolean")
    connection = store.connect()
    run_row = connection.execute(
        """
        SELECT status, started_at_ms, ended_at_ms, report_json, report_sha256, error
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
        [selected],
    ).fetchone()
    if run_row is None:
        raise ValueError("unknown Polymarket Round 14 capture run")
    manifest_row = connection.execute(
        """
        SELECT manifest_json, manifest_sha256
        FROM polymarket_preregistration_manifest WHERE run_id = ?
        """,
        [selected],
    ).fetchone()
    if manifest_row is None:
        raise ValueError("Round 14 capture has no preregistration manifest")
    try:
        report = json.loads(
            str(run_row[3]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
        manifest_value = json.loads(
            str(manifest_row[0]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 14 capture metadata is invalid JSON") from exc
    if not isinstance(report, Mapping) or not isinstance(manifest_value, Mapping):
        raise ValueError("Round 14 capture metadata is not an object")
    manifest = validate_round14_capture_manifest(
        manifest_value,
        expected_run_id=selected,
    )
    capture_manifest_sha256 = str(manifest_row[1]).strip().lower()
    run_report_sha256 = str(run_row[4]).strip().lower()
    if (
        manifest["manifest_sha256"] != capture_manifest_sha256
        or report.get("report_sha256") != run_report_sha256
        or not _is_sha256(run_report_sha256)
    ):
        raise ValueError("Round 14 stored capture metadata hash differs")

    global_reasons: list[str] = []
    status = str(run_row[0])
    if status not in _TERMINAL_RUN_STATUSES:
        global_reasons.append("capture_run_not_terminal")
    if str(run_row[5] or "").strip():
        global_reasons.append("capture_run_error_present")
    if manifest["purpose"] != "prospective":
        global_reasons.append("capture_not_prospective")
    if manifest["model_data_eligible"] is not True:
        global_reasons.append("capture_manifest_not_model_eligible")
    if manifest["contract_sha256"] != spec.contract_sha256:
        global_reasons.append("capture_contract_hash_differs")
    if manifest["campaign_plan_sha256"] != spec.campaign_plan_sha256:
        global_reasons.append("capture_campaign_plan_hash_differs")
    if manifest["labels_consulted"] is not False:
        global_reasons.append("capture_labels_were_consulted")
    if manifest["outcome_endpoints_queried"] is not False:
        global_reasons.append("capture_outcome_endpoint_was_queried")
    if manifest["model_scores_consulted"] is not False:
        global_reasons.append("capture_model_scores_were_consulted")
    report_stream_counts = report.get("stream_counts")
    if not isinstance(report_stream_counts, Mapping) or any(
        int(report_stream_counts.get(stream, 0)) <= 0
        for stream in (*_BINANCE_STREAMS, "clob_market", "polymarket_rtds")
    ):
        global_reasons.append("capture_required_stream_missing")
    integrity_errors = store.resume_integrity_errors(selected)
    if integrity_errors:
        global_reasons.append("capture_integrity_failed")
    if global_reasons and not diagnostic_only:
        raise ValueError(
            "Round 14 capture is not eligible for target-free materialization: "
            + ",".join(sorted(set(global_reasons)))
        )

    markets = PolymarketEvidenceReplay.load_markets(store, run_id=selected)
    cutoffs = {
        market.condition_id: market.end_ms - spec.decision_end_offset_ms
        for market in markets
    }
    replay = PolymarketEvidenceReplay.load(
        store,
        run_id=selected,
        allow_segmented_gaps=status == "degraded",
        include_resolutions=False,
        condition_ids=tuple(market.condition_id for market in markets),
        maximum_received_wall_ms_by_condition=cutoffs,
        materialized_minimum_depth_levels=1,
    )
    if replay.resolutions:
        raise ValueError("Round 14 target-free replay exposed resolutions")
    replay_books: dict[str, tuple[PolymarketRecordedBook, ...]] = {}
    for token_id in {token_id for market in markets for token_id in market.token_ids}:
        replay_books[token_id] = tuple(
            sorted(
                (book for book in replay.books if book.token_id == token_id),
                key=lambda book: (
                    book.received_wall_ms,
                    book.received_monotonic_ns,
                    book.event_id,
                ),
            )
        )
    gap_rows = connection.execute(
        """
        SELECT stream, opened_at_ms
        FROM polymarket_stream_gap
        WHERE run_id = ? ORDER BY opened_at_ms, gap_id
        """,
        [selected],
    ).fetchall()

    admissions: list[PolymarketRound14ConditionAdmission] = []
    rows: list[PolymarketRound14FeatureRow] = []
    normalized_global_reasons = tuple(sorted(set(global_reasons)))
    for market in sorted(
        markets, key=lambda item: (item.event_start_ms, item.condition_id)
    ):
        admission, condition_rows = _build_condition(
            store=store,
            run_id=selected,
            market=market,
            replay_books=replay_books,
            gap_rows=gap_rows,
            global_reasons=normalized_global_reasons,
            spec=spec,
        )
        admissions.append(admission)
        rows.extend(condition_rows)
    ordered_rows = tuple(
        sorted(rows, key=lambda row: (row.decision_time_ms, row.condition_id))
    )
    provisional = PolymarketRound14LabelFreeDataset(
        run_id=selected,
        run_report_sha256=run_report_sha256,
        capture_manifest_sha256=capture_manifest_sha256,
        admission_spec_sha256=spec.spec_sha256,
        feature_names_sha256=POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
        materializer_source_sha256=_source_sha256("polymarket_round14_dataset.py"),
        feature_builder_source_sha256=_source_sha256("polymarket_round14_features.py"),
        replay_source_sha256=_source_sha256("polymarket_replay.py"),
        diagnostic_only=diagnostic_only,
        admissions=tuple(admissions),
        rows=ordered_rows,
        dataset_sha256="0" * 64,
    )
    return replace(
        provisional,
        dataset_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


__all__ = [
    "POLYMARKET_ROUND14_ADMISSION_SPEC_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_CONDITION_ADMISSION_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_LABEL_FREE_DATASET_SCHEMA_VERSION",
    "PolymarketRound14AdmissionSpec",
    "PolymarketRound14ConditionAdmission",
    "PolymarketRound14LabelFreeDataset",
    "load_round14_admission_spec",
    "materialize_round14_label_free_run",
    "validate_round14_admission_spec",
]
