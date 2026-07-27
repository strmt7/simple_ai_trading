"""Source-derived Binance evidence for Round 74 executable targets.

The builders in this module are pure. Authentication and transport stay in the
existing API layer; only credential-free response bodies and exact capture clock
probes enter the research artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Mapping, Sequence

import zstandard

from .impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_ENVIRONMENTS,
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_commission_evidence_claims,
    round74_funding_schedule_evidence_claims,
    round74_quantity_rules_evidence_claims,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_FRAME_TABLE,
    IMPACT_CAPTURE_V10_REST_CONTEXT_TABLE,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
    ImpactCaptureAudit,
)
from .impact_absorption_targets import Round73MarketQuantityRules
from .impact_capture_frame import decode_impact_capture_frame


ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION = "round-074-binance-source-evidence-v2"
ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION = "round-074-binance-clock-probe-v1"
ROUND74_BINANCE_CLOCK_MAXIMUM_RTT_NS = 5_000_000_000
ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS = 90_000_000_000
ROUND74_BINANCE_FUNDING_MAXIMUM_LIMIT = 1_000
ROUND74_BINANCE_COMMISSION_MAXIMUM_RATE = Decimal("0.01")
_MILLISECONDS_TO_NANOSECONDS = 1_000_000
_MILLISECOND_QUANTIZATION_MAX_NS = _MILLISECONDS_TO_NANOSECONDS - 1
_SENSITIVE_KEYS = frozenset(
    {"apikey", "secret", "secretkey", "signature", "xmbxapikey"}
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 Binance evidence is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    selected = str(value)
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 74 Binance {label} digest is invalid")
    return selected


def _environment(value: object) -> str:
    selected = str(value).strip()
    if selected not in ROUND74_EVENT_TARGET_ENVIRONMENTS:
        raise ValueError("Round 74 Binance evidence environment differs")
    return selected


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 Binance {label} is invalid")
    if value <= 0:
        raise ValueError(f"Round 74 Binance {label} is invalid")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 Binance {label} is invalid")
    if value < 0:
        raise ValueError(f"Round 74 Binance {label} is invalid")
    return value


def _strict_decimal_string(
    value: object,
    label: str,
    *,
    minimum: Decimal,
    maximum: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Round 74 Binance {label} must be a decimal string")
    try:
        selected = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Round 74 Binance {label} is invalid") from exc
    if (
        not selected.is_finite()
        or selected < minimum
        or (maximum is not None and selected > maximum)
    ):
        raise ValueError(f"Round 74 Binance {label} is invalid")
    return selected


def _reject_sensitive_keys(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = (
                str(key).strip().lower().replace("-", "").replace("_", "")
            )
            if normalized in _SENSITIVE_KEYS:
                raise ValueError(
                    f"Round 74 Binance {path} contains credential material"
                )
            _reject_sensitive_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_sensitive_keys(nested, f"{path}[{index}]")


def _normalized_payload(value: object) -> object:
    _reject_sensitive_keys(value)
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


@dataclass(frozen=True)
class Round74BinanceClockProbe:
    """One exact `/fapi/v1/time` receipt and its local request interval."""

    capture_run_id: str
    capture_contract_sha256: str
    capture_audit_sha256: str
    frame_index: int
    message_index: int
    request_started_wall_ns: int
    received_wall_ns: int
    request_started_monotonic_ns: int
    received_monotonic_ns: int
    exchange_time_ms: int
    source_payload_sha256: str
    schema_version: str = ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.capture_run_id).strip():
            raise ValueError("Round 74 Binance clock capture run is empty")
        _sha256_digest(self.capture_contract_sha256, "capture contract")
        _sha256_digest(self.capture_audit_sha256, "capture audit")
        _sha256_digest(self.source_payload_sha256, "clock payload")
        frame = _nonnegative_integer(self.frame_index, "clock frame index")
        message = _nonnegative_integer(self.message_index, "clock message index")
        started_wall = _positive_integer(
            self.request_started_wall_ns,
            "clock request wall time",
        )
        received_wall = _positive_integer(
            self.received_wall_ns,
            "clock receipt wall time",
        )
        started = _positive_integer(
            self.request_started_monotonic_ns,
            "clock request monotonic time",
        )
        received = _positive_integer(
            self.received_monotonic_ns,
            "clock receipt monotonic time",
        )
        _positive_integer(self.exchange_time_ms, "clock exchange time")
        if (
            frame != self.frame_index
            or message != self.message_index
            or received_wall < started_wall
            or received <= started
            or received - started > ROUND74_BINANCE_CLOCK_MAXIMUM_RTT_NS
        ):
            raise ValueError("Round 74 Binance clock probe interval is invalid")
        if self.schema_version != ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION:
            raise ValueError("Round 74 Binance clock probe schema differs")

    @property
    def exchange_lower_ns(self) -> int:
        return int(self.exchange_time_ms) * _MILLISECONDS_TO_NANOSECONDS

    @property
    def exchange_upper_ns(self) -> int:
        return self.exchange_lower_ns + _MILLISECOND_QUANTIZATION_MAX_NS

    def as_source_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capture_run_id": self.capture_run_id,
            "capture_contract_sha256": self.capture_contract_sha256,
            "capture_audit_sha256": self.capture_audit_sha256,
            "frame_index": self.frame_index,
            "message_index": self.message_index,
            "request_started_wall_ns": self.request_started_wall_ns,
            "received_wall_ns": self.received_wall_ns,
            "request_started_monotonic_ns": (
                self.request_started_monotonic_ns
            ),
            "received_monotonic_ns": self.received_monotonic_ns,
            "exchange_time_ms": self.exchange_time_ms,
            "exchange_timestamp_quantization_ns": [
                0,
                _MILLISECOND_QUANTIZATION_MAX_NS,
            ],
            "source_payload_sha256": self.source_payload_sha256,
        }


@dataclass(frozen=True)
class Round74CommissionEvidenceBundle:
    """Validated account-specific taker fees and their immutable evidence."""

    taker_fee_bps_by_symbol: tuple[tuple[str, float], ...]
    evidence: Round74EventTargetEvidence

    def as_mapping(self) -> dict[str, float]:
        return dict(self.taker_fee_bps_by_symbol)


@dataclass(frozen=True)
class Round74QuantityRulesEvidenceBundle:
    """Validated market-order filters and their immutable source evidence."""

    quantity_rules_by_symbol: tuple[
        tuple[str, Round73MarketQuantityRules],
        ...,
    ]
    evidence: Round74EventTargetEvidence

    def as_mapping(self) -> dict[str, Round73MarketQuantityRules]:
        return dict(self.quantity_rules_by_symbol)


@dataclass(frozen=True)
class Round74FundingEvidenceBundle:
    """Complete queried funding panel mapped into conservative local intervals."""

    funding_boundary_intervals_monotonic_ns: tuple[
        tuple[str, tuple[tuple[int, int], ...]],
        ...,
    ]
    funding_schedule_coverage_monotonic_ns: tuple[
        tuple[str, tuple[int, int]],
        ...,
    ]
    evidence: Round74EventTargetEvidence

    def boundary_mapping(self) -> dict[str, tuple[tuple[int, int], ...]]:
        return dict(self.funding_boundary_intervals_monotonic_ns)

    def coverage_mapping(self) -> dict[str, tuple[int, int]]:
        return dict(self.funding_schedule_coverage_monotonic_ns)


def build_round74_quantity_rules_evidence(
    *,
    payload: Mapping[str, object],
    environment: str,
    observed_wall_ns: int,
) -> Round74QuantityRulesEvidenceBundle:
    """Derive executable market-order constraints from exchange metadata."""

    selected_environment = _environment(environment)
    observed = _positive_integer(
        observed_wall_ns,
        "quantity-rules observation time",
    )
    normalized = _normalized_payload(payload)
    if not isinstance(normalized, Mapping):
        raise ValueError("Round 74 Binance exchange-info payload differs")
    symbol_rows = normalized.get("symbols")
    if not isinstance(symbol_rows, Sequence) or isinstance(
        symbol_rows,
        (str, bytes, bytearray),
    ):
        raise ValueError("Round 74 Binance exchange-info symbols differ")

    selected_rows: dict[str, Mapping[str, object]] = {}
    for row in symbol_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 74 Binance exchange-info symbol row differs")
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol not in ROUND74_EVENT_TARGET_SYMBOLS:
            continue
        if symbol in selected_rows:
            raise ValueError(
                "Round 74 Binance exchange-info target symbol is duplicated"
            )
        selected_rows[symbol] = row
    if tuple(sorted(selected_rows)) != ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 Binance exchange-info symbol panel differs")

    rules: dict[str, Round73MarketQuantityRules] = {}
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        row = selected_rows[symbol]
        order_types = row.get("orderTypes")
        if (
            row.get("status") != "TRADING"
            or row.get("contractType") != "PERPETUAL"
            or str(row.get("pair", "")).strip().upper() != symbol
            or row.get("quoteAsset") != "USDT"
            or row.get("marginAsset") != "USDT"
            or not isinstance(order_types, Sequence)
            or isinstance(order_types, (str, bytes, bytearray))
            or "MARKET" not in order_types
        ):
            raise ValueError(
                "Round 74 Binance exchange-info trading contract differs"
            )
        filters = row.get("filters")
        if not isinstance(filters, Sequence) or isinstance(
            filters,
            (str, bytes, bytearray),
        ):
            raise ValueError("Round 74 Binance exchange-info filters differ")
        filter_by_type: dict[str, Mapping[str, object]] = {}
        for item in filters:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Round 74 Binance exchange-info filter row differs"
                )
            filter_type = str(item.get("filterType", "")).strip()
            if not filter_type or filter_type in filter_by_type:
                raise ValueError(
                    "Round 74 Binance exchange-info filter type differs"
                )
            filter_by_type[filter_type] = item
        try:
            lot = filter_by_type["MARKET_LOT_SIZE"]
            notional = filter_by_type["MIN_NOTIONAL"]
        except KeyError as exc:
            raise ValueError(
                "Round 74 Binance exchange-info executable filters differ"
            ) from exc
        rules[symbol] = Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size=_strict_decimal_string(
                lot.get("stepSize"),
                "MARKET_LOT_SIZE.stepSize",
                minimum=Decimal("0"),
            ),
            minimum_quantity=_strict_decimal_string(
                lot.get("minQty"),
                "MARKET_LOT_SIZE.minQty",
                minimum=Decimal("0"),
            ),
            maximum_quantity=_strict_decimal_string(
                lot.get("maxQty"),
                "MARKET_LOT_SIZE.maxQty",
                minimum=Decimal("0"),
            ),
            minimum_notional=_strict_decimal_string(
                notional.get("notional"),
                "MIN_NOTIONAL.notional",
                minimum=Decimal("0"),
            ),
        )

    claims = round74_quantity_rules_evidence_claims(rules)
    query_contract = {
        "schema_version": ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION,
        "environment": selected_environment,
        "method": "GET",
        "path": "/fapi/v1/exchangeInfo",
        "security_type": "NONE",
        "symbols": list(ROUND74_EVENT_TARGET_SYMBOLS),
        "contract_requirements": {
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "orderType": "MARKET",
        },
        "filters_used": ["MARKET_LOT_SIZE", "MIN_NOTIONAL"],
        "precision_fields_are_not_quantity_rules": True,
        "credential_material_persisted": False,
    }
    evidence = Round74EventTargetEvidence.create(
        kind="quantity_rules",
        environment=selected_environment,
        observed_wall_ns=observed,
        record_count=len(rules),
        source_query_or_protocol_sha256=_canonical_sha256(query_contract),
        source_payload_sha256=_canonical_sha256(normalized),
        claims=claims,
    )
    return Round74QuantityRulesEvidenceBundle(
        quantity_rules_by_symbol=tuple(
            (symbol, rules[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        evidence=evidence,
    )


def build_round74_commission_evidence(
    *,
    payload_by_symbol: Mapping[str, Mapping[str, object]],
    environment: str,
    observed_wall_ns: int,
) -> Round74CommissionEvidenceBundle:
    """Validate exact signed endpoint responses without retaining credentials."""

    selected_environment = _environment(environment)
    observed = _positive_integer(observed_wall_ns, "commission observation time")
    normalized_input = {
        str(symbol).strip().upper(): payload
        for symbol, payload in payload_by_symbol.items()
    }
    if tuple(sorted(normalized_input)) != ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 Binance commission symbol panel differs")
    normalized_payloads: dict[str, object] = {}
    fees: dict[str, float] = {}
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        payload = normalized_input[symbol]
        normalized = _normalized_payload(payload)
        if not isinstance(normalized, Mapping):
            raise ValueError("Round 74 Binance commission payload differs")
        if str(normalized.get("symbol", "")).strip().upper() != symbol:
            raise ValueError("Round 74 Binance commission response symbol differs")
        for field in (
            "makerCommissionRate",
            "takerCommissionRate",
            "rpiCommissionRate",
        ):
            _strict_decimal_string(
                normalized.get(field),
                field,
                minimum=Decimal("0"),
                maximum=ROUND74_BINANCE_COMMISSION_MAXIMUM_RATE,
            )
        taker_rate = _strict_decimal_string(
            normalized["takerCommissionRate"],
            "takerCommissionRate",
            minimum=Decimal("0"),
            maximum=ROUND74_BINANCE_COMMISSION_MAXIMUM_RATE,
        )
        fees[symbol] = float(taker_rate * Decimal("10000"))
        normalized_payloads[symbol] = normalized
    claims = round74_commission_evidence_claims(fees)
    query_contract = {
        "schema_version": ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION,
        "environment": selected_environment,
        "method": "GET",
        "path": "/fapi/v1/commissionRate",
        "security_type": "USER_DATA_SIGNED",
        "symbols": list(ROUND74_EVENT_TARGET_SYMBOLS),
        "response_fields_used": [
            "symbol",
            "makerCommissionRate",
            "takerCommissionRate",
            "rpiCommissionRate",
        ],
        "credential_material_persisted": False,
    }
    evidence = Round74EventTargetEvidence.create(
        kind="commission",
        environment=selected_environment,
        observed_wall_ns=observed,
        record_count=len(normalized_payloads),
        source_query_or_protocol_sha256=_canonical_sha256(query_contract),
        source_payload_sha256=_canonical_sha256(normalized_payloads),
        claims=claims,
    )
    return Round74CommissionEvidenceBundle(
        taker_fee_bps_by_symbol=tuple(sorted(fees.items())),
        evidence=evidence,
    )


def _validated_clock_probes(
    probes: Sequence[Round74BinanceClockProbe],
) -> tuple[Round74BinanceClockProbe, ...]:
    selected = tuple(probes)
    if len(selected) < 4:
        raise ValueError("Round 74 Binance clock probe panel is incomplete")
    if any(not isinstance(probe, Round74BinanceClockProbe) for probe in selected):
        raise ValueError("Round 74 Binance clock probe type differs")
    if len({(probe.frame_index, probe.message_index) for probe in selected}) != len(
        selected
    ):
        raise ValueError("Round 74 Binance clock probe location is duplicated")
    if len({probe.capture_run_id for probe in selected}) != 1 or len(
        {probe.capture_contract_sha256 for probe in selected}
    ) != 1 or len({probe.capture_audit_sha256 for probe in selected}) != 1:
        raise ValueError("Round 74 Binance clock probe capture identity differs")
    for previous, current in zip(selected, selected[1:], strict=False):
        if (
            current.request_started_monotonic_ns
            <= previous.received_monotonic_ns
            or current.exchange_lower_ns <= previous.exchange_upper_ns
            or current.request_started_monotonic_ns
            - previous.request_started_monotonic_ns
            > ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS
        ):
            raise ValueError("Round 74 Binance clock probe order differs")
    return selected


def load_round74_binance_clock_probes(
    connection: object,
    *,
    run_id: str,
    capture_audit: ImpactCaptureAudit,
) -> tuple[Round74BinanceClockProbe, ...]:
    """Extract audited `/time` probes without rescanning every capture frame."""

    selected_run = str(run_id).strip()
    if (
        not selected_run
        or not isinstance(capture_audit, ImpactCaptureAudit)
        or capture_audit.run_id != selected_run
        or not capture_audit.passed
        or capture_audit.errors
        or capture_audit.capture_contract_sha256
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
    ):
        raise ValueError("Round 74 Binance capture audit differs")
    audit_sha256 = _canonical_sha256(capture_audit.as_dict())
    run = connection.execute(
        """
        SELECT schema_version, design_sha256, capture_contract_sha256,
               status, last_frame_sha256
        FROM impact_capture_run WHERE run_id = ?
        """,
        [selected_run],
    ).fetchone()
    if (
        run is None
        or str(run[0]) != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or str(run[1]) != ROUND74_CAPTURE_DESIGN_SHA256
        or str(run[2]) != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or str(run[3]) != "completed"
        or str(run[4]) != capture_audit.last_frame_sha256
    ):
        raise ValueError("Round 74 Binance capture run identity differs")
    context_rows = connection.execute(
        f"""
        SELECT frame_index, message_index, request_path,
               request_parameters_json, response_status,
               request_started_wall_ns, request_started_monotonic_ns,
               exchange_time_ms
        FROM {IMPACT_CAPTURE_V10_REST_CONTEXT_TABLE}
        WHERE run_id = ? AND event_type = 'serverTime'
        ORDER BY frame_index, message_index
        """,
        [selected_run],
    ).fetchall()
    if len(context_rows) < 4:
        raise ValueError("Round 74 Binance capture clock context is incomplete")
    context_by_frame: dict[int, list[tuple[object, ...]]] = {}
    for row in context_rows:
        frame_index = _nonnegative_integer(row[0], "clock frame index")
        message_index = _nonnegative_integer(row[1], "clock message index")
        if (
            str(row[2]) != "/fapi/v1/time"
            or json.loads(str(row[3])) != {}
            or int(row[4]) != 200
            or row[7] is None
        ):
            raise ValueError("Round 74 Binance capture clock context differs")
        context_by_frame.setdefault(frame_index, []).append(
            (
                message_index,
                _positive_integer(row[5], "clock request wall time"),
                _positive_integer(row[6], "clock request monotonic time"),
                _positive_integer(row[7], "clock exchange time"),
            )
        )
    frame_rows = connection.execute(
        f"""
        SELECT DISTINCT f.frame_index, f.message_count,
               f.uncompressed_bytes, f.uncompressed_sha256,
               f.compressed_bytes, f.compressed_sha256,
               f.compressed_payload
        FROM {IMPACT_CAPTURE_V10_FRAME_TABLE} AS f
        INNER JOIN {IMPACT_CAPTURE_V10_REST_CONTEXT_TABLE} AS r
          ON r.run_id = f.run_id AND r.frame_index = f.frame_index
        WHERE f.run_id = ? AND r.event_type = 'serverTime'
        ORDER BY f.frame_index
        """,
        [selected_run],
    ).fetchall()
    if tuple(int(row[0]) for row in frame_rows) != tuple(context_by_frame):
        raise ValueError("Round 74 Binance capture clock frames differ")
    decompressor = zstandard.ZstdDecompressor()
    probes: list[Round74BinanceClockProbe] = []
    for row in frame_rows:
        frame_index = int(row[0])
        compressed = bytes(row[6])
        if (
            len(compressed) != int(row[4])
            or hashlib.sha256(compressed).hexdigest() != str(row[5])
        ):
            raise ValueError("Round 74 Binance capture compressed frame differs")
        try:
            uncompressed = decompressor.decompress(
                compressed,
                max_output_size=int(row[2]),
            )
        except zstandard.ZstdError as exc:
            raise ValueError(
                "Round 74 Binance capture clock frame cannot decompress"
            ) from exc
        if (
            len(uncompressed) != int(row[2])
            or hashlib.sha256(uncompressed).hexdigest() != str(row[3])
        ):
            raise ValueError("Round 74 Binance capture uncompressed frame differs")
        records = decode_impact_capture_frame(
            uncompressed,
            expected_message_count=int(row[1]),
        )
        for (
            message_index,
            request_started_wall_ns,
            request_started_monotonic_ns,
            exchange_time_ms,
        ) in context_by_frame[frame_index]:
            try:
                record = records[message_index].record
            except IndexError as exc:
                raise ValueError(
                    "Round 74 Binance capture clock location differs"
                ) from exc
            try:
                body = json.loads(record.raw_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Round 74 Binance capture clock payload differs"
                ) from exc
            if (
                record.stream != "binance_futures_rest"
                or not isinstance(body, Mapping)
                or set(body) != {"serverTime"}
                or body["serverTime"] != exchange_time_ms
            ):
                raise ValueError("Round 74 Binance capture clock payload differs")
            probes.append(
                Round74BinanceClockProbe(
                    capture_run_id=selected_run,
                    capture_contract_sha256=IMPACT_CAPTURE_V10_CONTRACT_SHA256,
                    capture_audit_sha256=audit_sha256,
                    frame_index=frame_index,
                    message_index=message_index,
                    request_started_wall_ns=request_started_wall_ns,
                    received_wall_ns=record.received_wall_ns,
                    request_started_monotonic_ns=(
                        request_started_monotonic_ns
                    ),
                    received_monotonic_ns=record.received_monotonic_ns,
                    exchange_time_ms=exchange_time_ms,
                    source_payload_sha256=hashlib.sha256(
                        record.raw_text.encode("utf-8", errors="strict")
                    ).hexdigest(),
                )
            )
    return _validated_clock_probes(probes)


def _funding_interval_for_exchange_time(
    funding_time_ms: int,
    probes: tuple[Round74BinanceClockProbe, ...],
) -> tuple[int, int] | None:
    previous = next(
        (
            probe
            for probe in reversed(probes)
            if probe.exchange_time_ms < funding_time_ms
        ),
        None,
    )
    following = next(
        (
            probe
            for probe in probes
            if probe.exchange_time_ms > funding_time_ms
        ),
        None,
    )
    if previous is None or following is None:
        return None
    return (
        previous.request_started_monotonic_ns,
        following.received_monotonic_ns,
    )


def _trim_coverage_around_intervals(
    *,
    initial_coverage: tuple[int, int],
    intervals: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    lower, upper = initial_coverage
    changed = True
    while changed:
        changed = False
        for start, end in intervals:
            if start < lower <= end:
                lower = end + 1
                changed = True
            if start <= upper < end:
                upper = start - 1
                changed = True
        if lower >= upper:
            raise ValueError("Round 74 Binance funding coverage was exhausted")
    return lower, upper


def build_round74_funding_evidence(
    *,
    payload_by_symbol: Mapping[str, Sequence[Mapping[str, object]]],
    environment: str,
    observed_wall_ns: int,
    start_time_ms: int,
    end_time_ms: int,
    limit: int,
    clock_probes: Sequence[Round74BinanceClockProbe],
) -> Round74FundingEvidenceBundle:
    """Bind complete funding responses to non-interpolated monotonic intervals."""

    selected_environment = _environment(environment)
    observed = _positive_integer(observed_wall_ns, "funding observation time")
    start = _positive_integer(start_time_ms, "funding query start")
    end = _positive_integer(end_time_ms, "funding query end")
    selected_limit = _positive_integer(limit, "funding query limit")
    if end <= start or selected_limit > ROUND74_BINANCE_FUNDING_MAXIMUM_LIMIT:
        raise ValueError("Round 74 Binance funding query differs")
    probes = _validated_clock_probes(clock_probes)
    if start > probes[0].exchange_time_ms or end < probes[-1].exchange_time_ms:
        raise ValueError("Round 74 Binance funding query does not cover capture clocks")
    normalized_input = {
        str(symbol).strip().upper(): rows
        for symbol, rows in payload_by_symbol.items()
    }
    if tuple(sorted(normalized_input)) != ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 Binance funding symbol panel differs")

    normalized_payloads: dict[str, object] = {}
    funding_times: dict[str, tuple[int, ...]] = {}
    record_count = 0
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        rows = tuple(normalized_input[symbol])
        if not rows or len(rows) >= selected_limit:
            raise ValueError(
                "Round 74 Binance funding response is empty or may be truncated"
            )
        normalized_rows: list[object] = []
        times: list[int] = []
        for row in rows:
            normalized = _normalized_payload(row)
            if not isinstance(normalized, Mapping):
                raise ValueError("Round 74 Binance funding payload differs")
            if str(normalized.get("symbol", "")).strip().upper() != symbol:
                raise ValueError("Round 74 Binance funding response symbol differs")
            funding_time = _positive_integer(
                normalized.get("fundingTime"),
                "funding time",
            )
            if not start <= funding_time <= end:
                raise ValueError("Round 74 Binance funding row is outside query")
            _strict_decimal_string(
                normalized.get("fundingRate"),
                "fundingRate",
                minimum=Decimal("-1"),
                maximum=Decimal("1"),
            )
            _strict_decimal_string(
                normalized.get("markPrice"),
                "funding markPrice",
                minimum=Decimal("0"),
            )
            rate_type = normalized.get("rateType")
            if rate_type is not None and rate_type not in {"Regular", "Special"}:
                raise ValueError("Round 74 Binance funding rate type differs")
            times.append(funding_time)
            normalized_rows.append(normalized)
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Round 74 Binance funding response order differs")
        normalized_payloads[symbol] = normalized_rows
        funding_times[symbol] = tuple(times)
        record_count += len(times)

    initial_coverage = (
        probes[1].request_started_monotonic_ns,
        probes[-2].received_monotonic_ns,
    )
    boundaries: dict[str, tuple[tuple[int, int], ...]] = {}
    coverage: dict[str, tuple[int, int]] = {}
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        mapped = tuple(
            interval
            for funding_time in funding_times[symbol]
            if (
                interval := _funding_interval_for_exchange_time(
                    funding_time,
                    probes,
                )
            )
            is not None
        )
        selected_coverage = _trim_coverage_around_intervals(
            initial_coverage=initial_coverage,
            intervals=mapped,
        )
        selected_boundaries = tuple(
            interval
            for interval in mapped
            if interval[0] >= selected_coverage[0]
            and interval[1] <= selected_coverage[1]
        )
        if any(
            interval[0] <= selected_coverage[1]
            and interval[1] >= selected_coverage[0]
            and interval not in selected_boundaries
            for interval in mapped
        ):
            raise ValueError("Round 74 Binance funding interval coverage differs")
        boundaries[symbol] = selected_boundaries
        coverage[symbol] = selected_coverage

    claims = round74_funding_schedule_evidence_claims(
        funding_boundary_intervals_monotonic_ns=boundaries,
        funding_schedule_coverage_monotonic_ns=coverage,
    )
    query_contract = {
        "schema_version": ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION,
        "environment": selected_environment,
        "method": "GET",
        "path": "/fapi/v1/fundingRate",
        "security_type": "NONE",
        "symbols": list(ROUND74_EVENT_TARGET_SYMBOLS),
        "startTime": start,
        "endTime": end,
        "inclusive_time_bounds": True,
        "limit": selected_limit,
        "response_order": "ascending",
        "full_page_is_rejected_as_potentially_truncated": True,
        "clock_path": "/fapi/v1/time",
        "clock_mapping": (
            "funding timestamp is bounded by preceding request start and "
            "following response receipt; no interpolation"
        ),
        "capture_run_id": probes[0].capture_run_id,
        "capture_contract_sha256": probes[0].capture_contract_sha256,
        "capture_audit_sha256": probes[0].capture_audit_sha256,
    }
    source_payload = {
        "funding_responses": normalized_payloads,
        "clock_probes": [probe.as_source_record() for probe in probes],
    }
    evidence = Round74EventTargetEvidence.create(
        kind="funding_schedule",
        environment=selected_environment,
        observed_wall_ns=observed,
        record_count=record_count,
        source_query_or_protocol_sha256=_canonical_sha256(query_contract),
        source_payload_sha256=_canonical_sha256(source_payload),
        claims=claims,
    )
    return Round74FundingEvidenceBundle(
        funding_boundary_intervals_monotonic_ns=tuple(
            (symbol, boundaries[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        funding_schedule_coverage_monotonic_ns=tuple(
            (symbol, coverage[symbol])
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        evidence=evidence,
    )


__all__ = [
    "ROUND74_BINANCE_CLOCK_MAXIMUM_PROBE_GAP_NS",
    "ROUND74_BINANCE_CLOCK_MAXIMUM_RTT_NS",
    "ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION",
    "ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION",
    "ROUND74_BINANCE_FUNDING_MAXIMUM_LIMIT",
    "Round74BinanceClockProbe",
    "Round74CommissionEvidenceBundle",
    "Round74FundingEvidenceBundle",
    "Round74QuantityRulesEvidenceBundle",
    "build_round74_commission_evidence",
    "build_round74_funding_evidence",
    "build_round74_quantity_rules_evidence",
    "load_round74_binance_clock_probes",
]
