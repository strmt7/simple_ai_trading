"""Strict source loaders for Round 74 public-mainnet paper targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from .impact_absorption_event_evidence import (
    build_round74_commission_evidence,
)
from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_funding_schedule_evidence_claims,
    round74_quantity_rules_evidence_claims,
)
from .impact_absorption_targets import Round73MarketQuantityRules


ROUND74_COHORT_CAPTURE_SOURCE_SCHEMA_VERSION = (
    "round-074-segmented-cohort-capture-source-v1"
)
ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION = "round-074-exchange-info-capture-v1"
ROUND74_COMMISSION_ARTIFACT_SCHEMA_VERSION = "round-074-commission-artifact-v1"
ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION = "round-074-commission-capture-v1"
ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION = "round-074-funding-capture-v1"
ROUND74_PUBLIC_TARGET_ENVIRONMENT = "binance_usdm_mainnet"
ROUND74_TESTNET_EXECUTION_ENVIRONMENT = "binance_usdm_testnet"
ROUND74_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
ROUND74_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
ROUND74_FUNDING_LIMIT = 1_000
ROUND74_COMMISSION_BASE_URL = "https://fapi.binance.com"
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FALSE_AUTHORITY_KEYS = frozenset(
    {
        "model_training",
        "financial_edge_tested",
        "profitability_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    }
)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 public target source is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        selected: dict[str, object] = {}
        for key, value in pairs:
            if key in selected:
                raise ValueError(f"duplicate key: {key}")
            selected[key] = value
        return selected

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite value: {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Round 74 {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 74 {label} root differs")
    return value


def _false_authority(
    value: object,
    *,
    required_keys: frozenset[str] = _FALSE_AUTHORITY_KEYS,
) -> bool:
    return (
        isinstance(value, Mapping)
        and required_keys.issubset(value)
        and all(value[key] is False for key in required_keys)
    )


def _symbol_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or tuple(sorted(str(symbol) for symbol in value))
        != ROUND74_EVENT_TARGET_SYMBOLS
    ):
        raise ValueError(f"Round 74 {label} symbol panel differs")
    return value


@dataclass(frozen=True)
class Round74CanonicalSourceArtifact:
    """Canonical payload plus its logical and physical identities."""

    payload: Mapping[str, object]
    artifact_sha256: str
    artifact_file_sha256: str

    def validate(self) -> None:
        if (
            not _is_sha256(self.artifact_sha256)
            or not _is_sha256(self.artifact_file_sha256)
            or _canonical_sha256(self.payload) != self.artifact_sha256
        ):
            raise ValueError("Round 74 source artifact identity differs")


@dataclass(frozen=True)
class Round74ExchangeInfoTargetSource:
    rules_by_symbol: tuple[tuple[str, Round73MarketQuantityRules], ...]
    evidence: Round74EventTargetEvidence

    def mapping(self) -> dict[str, Round73MarketQuantityRules]:
        return dict(self.rules_by_symbol)


@dataclass(frozen=True)
class Round74CommissionTargetSource:
    taker_fee_bps_by_symbol: tuple[tuple[str, float], ...]
    evidence: Round74EventTargetEvidence

    def mapping(self) -> dict[str, float]:
        return dict(self.taker_fee_bps_by_symbol)


@dataclass(frozen=True)
class Round74FundingTargetSource:
    run_id: str
    boundary_intervals_by_symbol: tuple[
        tuple[str, tuple[tuple[int, int], ...]],
        ...,
    ]
    coverage_by_symbol: tuple[tuple[str, tuple[int, int]], ...]
    evidence: Round74EventTargetEvidence

    def boundary_mapping(self) -> dict[str, tuple[tuple[int, int], ...]]:
        return dict(self.boundary_intervals_by_symbol)

    def coverage_mapping(self) -> dict[str, tuple[int, int]]:
        return dict(self.coverage_by_symbol)


def load_round74_canonical_source_artifact(
    path: str | Path,
    *,
    label: str,
) -> Round74CanonicalSourceArtifact:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 0 < selected.stat().st_size <= _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ValueError(f"Round 74 {label} artifact file differs")
    raw = selected.read_bytes()
    payload = _strict_json_object(raw, label=f"{label} artifact")
    accepted_encodings = {
        (_canonical_json(payload) + "\n").encode("ascii"),
        (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=False,
            )
            + "\n"
        ).encode("ascii"),
    }
    if raw not in accepted_encodings:
        raise ValueError(f"Round 74 {label} artifact encoding differs")
    claimed = str(payload.pop("artifact_sha256", ""))
    source = Round74CanonicalSourceArtifact(
        payload=payload,
        artifact_sha256=claimed,
        artifact_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    source.validate()
    return source


def build_round74_cohort_capture_source_payload(
    *,
    binding: Round74SegmentedCohortRunBinding,
    database_relative_path: str,
) -> dict[str, object]:
    """Wrap one exact admitted binding for downstream read-only evidence."""

    binding.validate()
    relative = PurePosixPath(str(database_relative_path))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".duckdb"
        or "\\" in str(database_relative_path)
    ):
        raise ValueError("Round 74 cohort source database path differs")
    payload: dict[str, object] = {
        "schema_version": ROUND74_COHORT_CAPTURE_SOURCE_SCHEMA_VERSION,
        "environment": ROUND74_PUBLIC_TARGET_ENVIRONMENT,
        "database": relative.as_posix(),
        "run_id": binding.run_id,
        "cohort_binding_sha256": binding.binding_sha256,
        "capture": {
            "run_id": binding.run_id,
            "status": "completed",
            "report_sha256": binding.report_sha256,
            "terminal_status": binding.terminal_status,
            "terminal_error": binding.terminal_error,
        },
        "cohort_binding": binding.as_dict(),
        "authority": {
            "model_training": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    audit_round74_cohort_capture_source_payload(
        payload,
        run_id=binding.run_id,
        cohort_binding_sha256=binding.binding_sha256,
    )
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def audit_round74_cohort_capture_source_payload(
    value: Mapping[str, object],
    *,
    run_id: str,
    cohort_binding_sha256: str,
) -> Round74SegmentedCohortRunBinding:
    payload = dict(value)
    capture = payload.get("capture")
    raw_binding = payload.get("cohort_binding")
    database = PurePosixPath(str(payload.get("database", "")))
    if (
        set(payload)
        != {
            "schema_version",
            "environment",
            "database",
            "run_id",
            "cohort_binding_sha256",
            "capture",
            "cohort_binding",
            "authority",
        }
        or payload.get("schema_version") != ROUND74_COHORT_CAPTURE_SOURCE_SCHEMA_VERSION
        or payload.get("environment") != ROUND74_PUBLIC_TARGET_ENVIRONMENT
        or payload.get("run_id") != run_id
        or payload.get("cohort_binding_sha256") != cohort_binding_sha256
        or database.is_absolute()
        or not database.parts
        or any(part in {"", ".", ".."} for part in database.parts)
        or database.suffix != ".duckdb"
        or "\\" in str(payload.get("database", ""))
        or not isinstance(capture, Mapping)
        or set(capture)
        != {
            "run_id",
            "status",
            "report_sha256",
            "terminal_status",
            "terminal_error",
        }
        or capture.get("run_id") != run_id
        or capture.get("status") != "completed"
        or not isinstance(raw_binding, Mapping)
        or not _false_authority(payload.get("authority"))
    ):
        raise ValueError("Round 74 cohort capture source differs")
    binding = Round74SegmentedCohortRunBinding.from_dict(raw_binding)
    if (
        binding.run_id != run_id
        or binding.binding_sha256 != cohort_binding_sha256
        or binding.report_sha256 != capture.get("report_sha256")
        or binding.terminal_status != capture.get("terminal_status")
        or binding.terminal_error != capture.get("terminal_error")
    ):
        raise ValueError("Round 74 cohort capture binding differs")
    return binding


def parse_round74_exchange_info_source(
    value: Mapping[str, object],
) -> Round74ExchangeInfoTargetSource:
    payload = dict(value)
    request = payload.get("request")
    response = payload.get("response")
    source_binding = payload.get("source_binding")
    claims = payload.get("quantity_rules")
    raw_evidence = payload.get("target_evidence")
    if (
        set(payload)
        != {
            "schema_version",
            "captured_at_utc",
            "execution_git_commit",
            "request",
            "response",
            "source_binding",
            "quantity_rules",
            "target_evidence",
            "authority",
        }
        or payload.get("schema_version") != ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION
        or not _is_git_commit(payload.get("execution_git_commit"))
        or not isinstance(request, Mapping)
        or request.get("method") != "GET"
        or request.get("url") != ROUND74_EXCHANGE_INFO_URL
        or request.get("security_type") != "NONE"
        or request.get("request_weight") != 1
        or request.get("retry_count") != 0
        or request.get("credential_material_sent") is not False
        or not isinstance(response, Mapping)
        or response.get("status") != 200
        or response.get("raw_payload_persisted") is not False
        or not isinstance(source_binding, Mapping)
        or source_binding.get("parser_path")
        != "src/simple_ai_trading/impact_absorption_exchange_info_evidence.py"
        or source_binding.get("capture_tool_path")
        != "tools/capture_round74_exchange_info_evidence.py"
        or not _is_sha256(source_binding.get("parser_sha256"))
        or not _is_sha256(source_binding.get("capture_tool_sha256"))
        or not isinstance(claims, Mapping)
        or not isinstance(raw_evidence, Mapping)
        or not _false_authority(payload.get("authority"))
    ):
        raise ValueError("Round 74 exchange-info target source differs")
    raw_rules = claims.get("market_quantity_rules_by_symbol")
    rules_panel = _symbol_mapping(
        raw_rules,
        label="exchange-info quantity rules",
    )
    rules: list[tuple[str, Round73MarketQuantityRules]] = []
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        row = rules_panel[symbol]
        if not isinstance(row, Mapping) or set(row) != {
            "step_size",
            "minimum_quantity",
            "maximum_quantity",
            "minimum_notional",
        }:
            raise ValueError("Round 74 exchange-info quantity rule differs")
        rules.append(
            (
                symbol,
                Round73MarketQuantityRules.create(
                    symbol=symbol,
                    step_size=row["step_size"],
                    minimum_quantity=row["minimum_quantity"],
                    maximum_quantity=row["maximum_quantity"],
                    minimum_notional=row["minimum_notional"],
                ),
            )
        )
    selected = dict(rules)
    evidence = Round74EventTargetEvidence.from_dict(raw_evidence)
    received_wall_ns = response.get("received_wall_ns")
    if (
        claims != round74_quantity_rules_evidence_claims(selected)
        or evidence.kind != "quantity_rules"
        or evidence.environment != ROUND74_PUBLIC_TARGET_ENVIRONMENT
        or evidence.observed_wall_ns != received_wall_ns
        or not evidence.binds(claims)
    ):
        raise ValueError("Round 74 exchange-info evidence differs")
    return Round74ExchangeInfoTargetSource(tuple(rules), evidence)


def parse_round74_commission_source(
    value: Mapping[str, object],
) -> Round74CommissionTargetSource:
    payload = dict(value)
    capture = payload.get("capture")
    source = payload.get("source")
    transport = payload.get("credential_transport")
    if (
        set(payload)
        != {
            "schema_version",
            "capture",
            "execution_git_commit",
            "source",
            "credential_transport",
        }
        or payload.get("schema_version") != ROUND74_COMMISSION_ARTIFACT_SCHEMA_VERSION
        or not _is_git_commit(payload.get("execution_git_commit"))
        or not isinstance(capture, Mapping)
        or not isinstance(source, Mapping)
        or source.get("capture_module")
        != "src/simple_ai_trading/round74_commission_capture.py"
        or source.get("capture_tool") != "tools/capture_round74_commission_evidence.py"
        or not _is_sha256(source.get("capture_module_sha256"))
        or not _is_sha256(source.get("capture_tool_sha256"))
        or not isinstance(transport, Mapping)
        or transport.get("source") != "process_environment_only"
        or transport.get("credential_values_persisted") is not False
    ):
        raise ValueError("Round 74 commission target source differs")
    capture_payload = dict(capture)
    capture_sha256 = str(capture_payload.pop("capture_sha256", ""))
    payload_by_symbol = capture_payload.get("payload_by_symbol")
    request_evidence = capture_payload.get("request_evidence")
    raw_fees = capture_payload.get("taker_fee_bps_by_symbol")
    raw_evidence = capture_payload.get("target_evidence")
    if (
        capture_sha256 != _canonical_sha256(capture_payload)
        or capture_payload.get("schema_version")
        != ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION
        or capture_payload.get("environment") != ROUND74_PUBLIC_TARGET_ENVIRONMENT
        or capture_payload.get("base_url") != ROUND74_COMMISSION_BASE_URL
        or not isinstance(payload_by_symbol, Mapping)
        or not isinstance(request_evidence, list)
        or not isinstance(raw_fees, Mapping)
        or not isinstance(raw_evidence, Mapping)
        or not _false_authority(
            capture_payload.get("authority"),
            required_keys=(_FALSE_AUTHORITY_KEYS | frozenset({"orders_submitted"})),
        )
    ):
        raise ValueError("Round 74 commission capture differs")
    _symbol_mapping(payload_by_symbol, label="commission payload")
    fees_panel = _symbol_mapping(raw_fees, label="commission fee")
    if len(request_evidence) != len(ROUND74_EVENT_TARGET_SYMBOLS) or any(
        not isinstance(row, Mapping) for row in request_evidence
    ):
        raise ValueError("Round 74 commission request evidence differs")
    observed_wall_ns = 0
    for symbol, row in zip(
        ROUND74_EVENT_TARGET_SYMBOLS,
        request_evidence,
        strict=True,
    ):
        assert isinstance(row, Mapping)
        clock = row.get("clock")
        commission = row.get("commission")
        if (
            row.get("symbol") != symbol
            or not isinstance(clock, Mapping)
            or clock.get("path") != "/fapi/v1/time"
            or not isinstance(commission, Mapping)
            or commission.get("method") != "GET"
            or commission.get("path") != "/fapi/v1/commissionRate"
            or commission.get("security_type") != "USER_DATA_SIGNED"
            or commission.get("request_weight") != 20
            or commission.get("retry_count") != 0
            or commission.get("signed_query_persisted") is not False
            or commission.get("credential_material_persisted") is not False
        ):
            raise ValueError("Round 74 commission request evidence differs")
        received = commission.get("received_wall_ns")
        if isinstance(received, bool) or not isinstance(received, int):
            raise ValueError("Round 74 commission request time differs")
        observed_wall_ns = max(observed_wall_ns, received)
    rebuilt = build_round74_commission_evidence(
        payload_by_symbol={
            symbol: payload_by_symbol[symbol] for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        },
        environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
        observed_wall_ns=observed_wall_ns,
    )
    evidence = Round74EventTargetEvidence.from_dict(raw_evidence)
    rebuilt_fees = rebuilt.as_mapping()
    if {
        symbol: float(fees_panel[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    } != rebuilt_fees or evidence.as_dict() != rebuilt.evidence.as_dict():
        raise ValueError("Round 74 commission evidence differs")
    return Round74CommissionTargetSource(
        tuple(
            (symbol, rebuilt_fees[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        evidence,
    )


def parse_round74_funding_source(
    value: Mapping[str, object],
    *,
    run_id: str,
) -> Round74FundingTargetSource:
    payload = dict(value)
    capture = payload.get("capture_binding")
    clock = payload.get("clock_binding")
    requests = payload.get("requests")
    raw_boundaries = payload.get("funding_boundary_intervals_monotonic_ns")
    raw_coverage = payload.get("funding_schedule_coverage_monotonic_ns")
    raw_evidence = payload.get("target_evidence")
    source = payload.get("source_binding")
    scope = payload.get("scope")
    if (
        payload.get("schema_version") != ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION
        or not _is_git_commit(payload.get("execution_git_commit"))
        or payload.get("database_open_mode") != "read_only"
        or not isinstance(capture, Mapping)
        or capture.get("run_id") != run_id
        or capture.get("fresh_full_run_audit_passed") is not True
        or capture.get("fresh_full_run_audit_errors") != []
        or not isinstance(clock, Mapping)
        or clock.get("interpolation_permitted") is not False
        or not isinstance(requests, list)
        or not isinstance(raw_boundaries, Mapping)
        or not isinstance(raw_coverage, Mapping)
        or not isinstance(raw_evidence, Mapping)
        or not isinstance(source, Mapping)
        or source.get("parser_path")
        != "src/simple_ai_trading/impact_absorption_event_evidence.py"
        or source.get("capture_tool_path")
        != "tools/capture_round74_funding_evidence.py"
        or source.get("transport_helper_path")
        != "tools/_round74_public_evidence_capture.py"
        or not all(
            _is_sha256(source.get(f"{label}_sha256"))
            for label in ("parser", "capture_tool", "transport_helper")
        )
        or not isinstance(scope, Mapping)
        or scope.get("synthetic_market_data_used") is not False
        or scope.get("credentials_used") is not False
        or scope.get("orders_submitted") is not False
        or not _false_authority(payload.get("authority"))
    ):
        raise ValueError("Round 74 funding target source differs")
    boundaries_panel = _symbol_mapping(
        raw_boundaries,
        label="funding boundary",
    )
    coverage_panel = _symbol_mapping(
        raw_coverage,
        label="funding coverage",
    )
    boundaries: dict[str, tuple[tuple[int, int], ...]] = {}
    coverage: dict[str, tuple[int, int]] = {}
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        raw_symbol_boundaries = boundaries_panel[symbol]
        raw_symbol_coverage = coverage_panel[symbol]
        if (
            not isinstance(raw_symbol_boundaries, Sequence)
            or isinstance(raw_symbol_boundaries, (str, bytes, bytearray))
            or any(
                not isinstance(interval, Sequence)
                or isinstance(interval, (str, bytes, bytearray))
                or len(interval) != 2
                for interval in raw_symbol_boundaries
            )
            or not isinstance(raw_symbol_coverage, Sequence)
            or isinstance(raw_symbol_coverage, (str, bytes, bytearray))
            or len(raw_symbol_coverage) != 2
        ):
            raise ValueError("Round 74 funding interval differs")
        boundaries[symbol] = tuple(
            (int(interval[0]), int(interval[1])) for interval in raw_symbol_boundaries
        )
        coverage[symbol] = (
            int(raw_symbol_coverage[0]),
            int(raw_symbol_coverage[1]),
        )
    if len(requests) != len(ROUND74_EVENT_TARGET_SYMBOLS) or any(
        not isinstance(row, Mapping) for row in requests
    ):
        raise ValueError("Round 74 funding request panel differs")
    observed_wall_ns = 0
    for symbol, row in zip(ROUND74_EVENT_TARGET_SYMBOLS, requests, strict=True):
        assert isinstance(row, Mapping)
        received = row.get("received_wall_ns")
        if (
            row.get("symbol") != symbol
            or row.get("method") != "GET"
            or not str(row.get("url", "")).startswith(f"{ROUND74_FUNDING_URL}?")
            or row.get("security_type") != "NONE"
            or row.get("limit") != ROUND74_FUNDING_LIMIT
            or row.get("retry_count") != 0
            or row.get("credential_material_sent") is not False
            or row.get("raw_payload_persisted") is not False
            or isinstance(received, bool)
            or not isinstance(received, int)
        ):
            raise ValueError("Round 74 funding request differs")
        observed_wall_ns = max(observed_wall_ns, received)
    evidence = Round74EventTargetEvidence.from_dict(raw_evidence)
    claims = round74_funding_schedule_evidence_claims(
        funding_boundary_intervals_monotonic_ns=boundaries,
        funding_schedule_coverage_monotonic_ns=coverage,
    )
    if (
        evidence.kind != "funding_schedule"
        or evidence.environment != ROUND74_PUBLIC_TARGET_ENVIRONMENT
        or evidence.observed_wall_ns != observed_wall_ns
        or not evidence.binds(claims)
    ):
        raise ValueError("Round 74 funding evidence differs")
    return Round74FundingTargetSource(
        run_id=run_id,
        boundary_intervals_by_symbol=tuple(
            (symbol, boundaries[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        coverage_by_symbol=tuple(
            (symbol, coverage[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        evidence=evidence,
    )


def audit_round74_public_target_source_payload(
    *,
    label: str,
    value: Mapping[str, object],
    run_id: str,
    cohort_binding_sha256: str,
) -> None:
    """Apply the exact source-specific contract during manifest reload."""

    if label == "cohort_capture":
        audit_round74_cohort_capture_source_payload(
            value,
            run_id=run_id,
            cohort_binding_sha256=cohort_binding_sha256,
        )
    elif label == "exchange_info":
        parse_round74_exchange_info_source(value)
    elif label == "commission":
        parse_round74_commission_source(value)
    elif label == "funding":
        parse_round74_funding_source(value, run_id=run_id)


__all__ = [
    "ROUND74_COHORT_CAPTURE_SOURCE_SCHEMA_VERSION",
    "ROUND74_COMMISSION_ARTIFACT_SCHEMA_VERSION",
    "ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION",
    "ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION",
    "ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION",
    "ROUND74_PUBLIC_TARGET_ENVIRONMENT",
    "ROUND74_TESTNET_EXECUTION_ENVIRONMENT",
    "Round74CanonicalSourceArtifact",
    "Round74CommissionTargetSource",
    "Round74ExchangeInfoTargetSource",
    "Round74FundingTargetSource",
    "audit_round74_cohort_capture_source_payload",
    "audit_round74_public_target_source_payload",
    "build_round74_cohort_capture_source_payload",
    "load_round74_canonical_source_artifact",
    "parse_round74_commission_source",
    "parse_round74_exchange_info_source",
    "parse_round74_funding_source",
]
