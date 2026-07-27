"""Source-derived Binance exchange-info evidence for Round 74 targets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Mapping, Sequence

from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_ENVIRONMENTS,
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_quantity_rules_evidence_claims,
)
from .impact_absorption_targets import Round73MarketQuantityRules


ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION = (
    "round-074-exchange-info-source-evidence-v1"
)
_SENSITIVE_KEYS = frozenset(
    {"apikey", "secret", "secretkey", "signature", "xmbxapikey"}
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Round 74 exchange-info evidence is not canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _environment(value: object) -> str:
    selected = str(value).strip()
    if selected not in ROUND74_EVENT_TARGET_ENVIRONMENTS:
        raise ValueError("Round 74 exchange-info environment differs")
    return selected


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Round 74 exchange-info {label} is invalid")
    return value


def _strict_positive_decimal_string(
    value: object,
    label: str,
) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Round 74 exchange-info {label} must be a decimal string"
        )
    try:
        selected = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"Round 74 exchange-info {label} is invalid"
        ) from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError(f"Round 74 exchange-info {label} is invalid")
    return selected


def _reject_sensitive_keys(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = (
                str(key).strip().lower().replace("-", "").replace("_", "")
            )
            if normalized in _SENSITIVE_KEYS:
                raise ValueError(
                    f"Round 74 exchange-info {path} contains credential material"
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
class Round74QuantityRulesEvidenceBundle:
    """Validated market-order filters and their immutable source evidence."""

    quantity_rules_by_symbol: tuple[
        tuple[str, Round73MarketQuantityRules],
        ...,
    ]
    evidence: Round74EventTargetEvidence

    def as_mapping(self) -> dict[str, Round73MarketQuantityRules]:
        return dict(self.quantity_rules_by_symbol)


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
        raise ValueError("Round 74 exchange-info payload differs")
    symbol_rows = normalized.get("symbols")
    if not isinstance(symbol_rows, Sequence) or isinstance(
        symbol_rows,
        (str, bytes, bytearray),
    ):
        raise ValueError("Round 74 exchange-info symbols differ")

    selected_rows: dict[str, Mapping[str, object]] = {}
    for row in symbol_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 74 exchange-info symbol row differs")
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol not in ROUND74_EVENT_TARGET_SYMBOLS:
            continue
        if symbol in selected_rows:
            raise ValueError(
                "Round 74 exchange-info target symbol is duplicated"
            )
        selected_rows[symbol] = row
    if tuple(sorted(selected_rows)) != ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 exchange-info symbol panel differs")

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
                "Round 74 exchange-info trading contract differs"
            )
        filters = row.get("filters")
        if not isinstance(filters, Sequence) or isinstance(
            filters,
            (str, bytes, bytearray),
        ):
            raise ValueError("Round 74 exchange-info filters differ")
        filter_by_type: dict[str, Mapping[str, object]] = {}
        for item in filters:
            if not isinstance(item, Mapping):
                raise ValueError("Round 74 exchange-info filter row differs")
            filter_type = str(item.get("filterType", "")).strip()
            if not filter_type or filter_type in filter_by_type:
                raise ValueError("Round 74 exchange-info filter type differs")
            filter_by_type[filter_type] = item
        try:
            lot = filter_by_type["MARKET_LOT_SIZE"]
            notional = filter_by_type["MIN_NOTIONAL"]
        except KeyError as exc:
            raise ValueError(
                "Round 74 exchange-info executable filters differ"
            ) from exc
        rules[symbol] = Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size=_strict_positive_decimal_string(
                lot.get("stepSize"),
                "MARKET_LOT_SIZE.stepSize",
            ),
            minimum_quantity=_strict_positive_decimal_string(
                lot.get("minQty"),
                "MARKET_LOT_SIZE.minQty",
            ),
            maximum_quantity=_strict_positive_decimal_string(
                lot.get("maxQty"),
                "MARKET_LOT_SIZE.maxQty",
            ),
            minimum_notional=_strict_positive_decimal_string(
                notional.get("notional"),
                "MIN_NOTIONAL.notional",
            ),
        )

    claims = round74_quantity_rules_evidence_claims(rules)
    query_contract = {
        "schema_version": ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION,
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


__all__ = [
    "ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION",
    "Round74QuantityRulesEvidenceBundle",
    "build_round74_quantity_rules_evidence",
]
