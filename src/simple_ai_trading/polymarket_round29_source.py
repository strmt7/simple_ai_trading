"""Target-blind source composition for Polymarket Round 29."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

from .polymarket_round27_features import Round27FeatureRow
from .polymarket_round28_book_ticker import Round28FeatureRow
from .polymarket_round29_settlement_features import (
    Round29FeatureRow,
    Round29SettlementOverlayRow,
)


POLYMARKET_ROUND29_SOURCE_REPORT_SCHEMA_VERSION = (
    "polymarket-round29-target-blind-source-report-v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 29 {name} SHA-256 differs")
    return selected


def _validated_claim(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> dict[str, object]:
    body = dict(value)
    claimed = _sha256(body.pop(hash_field, None), name=name)
    if claimed != _canonical_sha256(body):
        raise ValueError(f"Round 29 {name} hash differs")
    return {**body, hash_field: claimed}


def compose_round29_feature_pairs(
    *,
    base_rows: Sequence[Round27FeatureRow],
    round28_rows: Sequence[Round28FeatureRow],
    round28_overlay_report: Mapping[str, object],
) -> tuple[
    tuple[tuple[Round29FeatureRow, Round29FeatureRow], ...],
    dict[str, object],
]:
    """Create both matched views without reading targets or duplicating base data."""

    overlay_report = _validated_claim(
        round28_overlay_report,
        hash_field="report_sha256",
        name="Round 28 overlay report",
    )
    selected_base = tuple(row.validated() for row in base_rows)
    selected_round28 = tuple(row.validated() for row in round28_rows)
    if not selected_base or not selected_round28:
        raise ValueError("Round 29 source population is empty")
    base_by_decision = {row.decision_time_ms: row for row in selected_base}
    round28_by_decision = {row.decision_time_ms: row for row in selected_round28}
    if (
        len(base_by_decision) != len(selected_base)
        or len(round28_by_decision) != len(selected_round28)
        or not set(round28_by_decision) <= set(base_by_decision)
    ):
        raise ValueError("Round 29 source decision population differs")

    pairs: list[tuple[Round29FeatureRow, Round29FeatureRow]] = []
    settlement_hashes: list[str] = []
    for decision_time_ms in sorted(round28_by_decision):
        base = base_by_decision[decision_time_ms]
        round28 = round28_by_decision[decision_time_ms]
        if (
            round28.base_row_sha256 != base.row_sha256
            or round28.condition_id != base.condition_id
            or round28.event_start_ms != base.event_start_ms
            or round28.market_prior_probability != base.market_prior_probability
        ):
            raise ValueError("Round 29 base and BBO source identities differ")
        settlement = Round29SettlementOverlayRow.create(base)
        diagnostic = Round29FeatureRow.from_round27(base, settlement)
        primary = Round29FeatureRow.from_round28(round28, settlement)
        pairs.append((diagnostic, primary))
        settlement_hashes.append(settlement.row_sha256)

    diagnostic_hashes = [diagnostic.row_sha256 for diagnostic, _primary in pairs]
    primary_hashes = [primary.row_sha256 for _diagnostic, primary in pairs]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_SOURCE_REPORT_SCHEMA_VERSION,
        "round28_overlay_report_sha256": overlay_report["report_sha256"],
        "source_base_row_count": len(selected_base),
        "matched_row_count": len(pairs),
        "excluded_without_bbo_count": len(selected_base) - len(pairs),
        "source_base_population_sha256": _canonical_sha256(
            [row.row_sha256 for row in selected_base]
        ),
        "matched_base_population_sha256": _canonical_sha256(
            [
                base_by_decision[decision_time_ms].row_sha256
                for decision_time_ms in sorted(round28_by_decision)
            ]
        ),
        "round28_population_sha256": _canonical_sha256(
            [round28_by_decision[key].row_sha256 for key in sorted(round28_by_decision)]
        ),
        "settlement_overlay_population_sha256": _canonical_sha256(settlement_hashes),
        "diagnostic_population_sha256": _canonical_sha256(diagnostic_hashes),
        "primary_population_sha256": _canonical_sha256(primary_hashes),
        "first_decision_time_ms": min(round28_by_decision),
        "last_decision_time_ms": max(round28_by_decision),
        "diagnostic_and_primary_rows_matched": True,
        "official_outcomes_accessed": False,
        "target_accessed": False,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return tuple(pairs), body


def validate_round29_source_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="source report",
    )
    hash_fields = (
        "round28_overlay_report_sha256",
        "source_base_population_sha256",
        "matched_base_population_sha256",
        "round28_population_sha256",
        "settlement_overlay_population_sha256",
        "diagnostic_population_sha256",
        "primary_population_sha256",
    )
    source_count = report.get("source_base_row_count")
    matched_count = report.get("matched_row_count")
    excluded_count = report.get("excluded_without_bbo_count")
    first_decision_time_ms = report.get("first_decision_time_ms")
    last_decision_time_ms = report.get("last_decision_time_ms")
    if (
        report.get("schema_version") != POLYMARKET_ROUND29_SOURCE_REPORT_SCHEMA_VERSION
        or type(source_count) is not int
        or type(matched_count) is not int
        or type(excluded_count) is not int
        or int(source_count) <= 0
        or int(matched_count) <= 0
        or int(excluded_count) < 0
        or int(matched_count) + int(excluded_count) != int(source_count)
        or any(
            _sha256(report.get(field), name=field) != report.get(field)
            for field in hash_fields
        )
        or type(first_decision_time_ms) is not int
        or type(last_decision_time_ms) is not int
        or int(first_decision_time_ms) > int(last_decision_time_ms)
        or report.get("diagnostic_and_primary_rows_matched") is not True
        or any(
            report.get(field) is not False
            for field in (
                "official_outcomes_accessed",
                "target_accessed",
                "edge_claim",
                "profitability_claim",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 29 source report differs")
    return report


__all__ = [
    "POLYMARKET_ROUND29_SOURCE_REPORT_SCHEMA_VERSION",
    "compose_round29_feature_pairs",
    "validate_round29_source_report",
]
