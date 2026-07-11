"""Audit sampled aggregate-depth coverage in a cached action-value dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Mapping

import duckdb


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_features import (  # noqa: E402
    AGGREGATE_DEPTH_FEATURE_VERSION,
    AGGREGATE_DEPTH_MAX_AGE_MS,
    microstructure_feature_source_contract,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA_VERSION = "sampled-aggregate-depth-coverage-v1"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_DEPTH_FEATURES = (
    "aggregate_depth_available",
    "aggregate_depth_age_seconds",
    "log_bid_notional_within_1pct",
    "log_ask_notional_within_1pct",
    "log_bid_notional_within_5pct",
    "log_ask_notional_within_5pct",
    "aggregate_depth_notional_imbalance_1pct",
    "aggregate_depth_notional_imbalance_5pct",
    "bid_depth_concentration_1pct_to_5pct",
    "ask_depth_concentration_1pct_to_5pct",
    "aggregate_depth_concentration_skew",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"sampled-depth audit cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"sampled-depth audit {path.name} must be an object")
    return value


def audit_depth_coverage(
    report_path: Path,
    warehouse_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Bind depth availability statistics to one report and cache fingerprint."""

    report = _read_json(report_path)
    dataset = report.get("dataset")
    if (
        report.get("round") != 28
        or report.get("status") != "rejected"
        or not isinstance(dataset, Mapping)
        or not isinstance(dataset.get("cache_key"), str)
        or int(dataset.get("rows") or 0) <= 0
    ):
        raise ValueError("sampled-depth audit source report is invalid")
    cache_key = str(dataset["cache_key"])
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        manifest = connection.execute(
            "SELECT feature_version, row_count, source_evidence_json, "
            "dataset_fingerprint, rows_table "
            "FROM microstructure_dataset_cache_manifest WHERE cache_key = ?",
            [cache_key],
        ).fetchone()
        if manifest is None:
            raise ValueError("sampled-depth cache manifest is missing")
        feature_version, row_count, source_json, fingerprint, rows_table = manifest
        table = str(rows_table)
        if (
            feature_version != AGGREGATE_DEPTH_FEATURE_VERSION
            or int(row_count) != int(dataset["rows"])
            or _SAFE_IDENTIFIER.fullmatch(table) is None
            or len(str(fingerprint)) != 64
        ):
            raise ValueError("sampled-depth cache manifest is inconsistent")
        try:
            source_evidence = json.loads(str(source_json))
        except json.JSONDecodeError as exc:
            raise ValueError("sampled-depth source evidence is unreadable") from exc
        expected_contract = microstructure_feature_source_contract(
            AGGREGATE_DEPTH_FEATURE_VERSION
        )
        certificate = (
            source_evidence.get("corpus_certificate")
            if isinstance(source_evidence, Mapping)
            else None
        )
        if (
            not isinstance(source_evidence, Mapping)
            or source_evidence.get("feature_source_contract") != expected_contract
            or not isinstance(certificate, Mapping)
            or certificate.get("certificate_sha256")
            != report.get("corpus_certificate_sha256")
            or tuple(certificate.get("required_data_types") or ())
            != ("bookTicker", "trades", "bookDepth")
        ):
            raise ValueError("sampled-depth source contract is inconsistent")
        nonfinite = " OR ".join(f"NOT isfinite({name})" for name in _DEPTH_FEATURES)
        unavailable_nonzero = " OR ".join(
            f"{name} <> 0.0" for name in _DEPTH_FEATURES[1:]
        )
        row = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                count(*) FILTER (WHERE aggregate_depth_available = 1.0)::BIGINT,
                count(*) FILTER (WHERE aggregate_depth_available = 0.0)::BIGINT,
                min(aggregate_depth_age_seconds)
                    FILTER (WHERE aggregate_depth_available = 1.0),
                quantile_cont(aggregate_depth_age_seconds, 0.50)
                    FILTER (WHERE aggregate_depth_available = 1.0),
                quantile_cont(aggregate_depth_age_seconds, 0.99)
                    FILTER (WHERE aggregate_depth_available = 1.0),
                max(aggregate_depth_age_seconds)
                    FILTER (WHERE aggregate_depth_available = 1.0),
                count(*) FILTER (
                    WHERE aggregate_depth_available NOT IN (0.0, 1.0)
                       OR ({nonfinite})
                       OR (aggregate_depth_available = 0.0 AND ({unavailable_nonzero}))
                       OR (aggregate_depth_available = 1.0 AND (
                            aggregate_depth_age_seconds < 1.0
                            OR aggregate_depth_age_seconds > {AGGREGATE_DEPTH_MAX_AGE_MS / 1000.0}
                            OR abs(aggregate_depth_notional_imbalance_1pct) > 1.0
                            OR abs(aggregate_depth_notional_imbalance_5pct) > 1.0
                            OR bid_depth_concentration_1pct_to_5pct NOT BETWEEN 0.0 AND 1.0
                            OR ask_depth_concentration_1pct_to_5pct NOT BETWEEN 0.0 AND 1.0
                            OR abs(aggregate_depth_concentration_skew) > 1.0
                       ))
                )::BIGINT
            FROM {table}
            WHERE cache_key = ?
            """,
            [cache_key],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("sampled-depth coverage query returned no result")
    (
        rows,
        available_rows,
        unavailable_rows,
        age_min,
        age_p50,
        age_p99,
        age_max,
        invalid_rows,
    ) = row
    if (
        int(rows) != int(dataset["rows"])
        or int(available_rows) + int(unavailable_rows) != int(rows)
        or int(available_rows) <= 0
        or int(invalid_rows) != 0
    ):
        raise ValueError("sampled-depth cached feature coverage is invalid")
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "round": 28,
        "feature_version": AGGREGATE_DEPTH_FEATURE_VERSION,
        "data_product": "Binance bookDepth sampled aggregate percentage bands",
        "bands_used_percent": [1.0, 5.0],
        "full_l2_order_book": False,
        "queue_position_evidence": False,
        "maker_fill_evidence": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "source_report_sha256": _sha256(report_path),
        "source_report_canonical_sha256": report["report_sha256"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "cache_key": cache_key,
        "dataset_fingerprint": str(fingerprint),
        "maximum_age_ms": AGGREGATE_DEPTH_MAX_AGE_MS,
        "rows": int(rows),
        "available_rows": int(available_rows),
        "unavailable_rows": int(unavailable_rows),
        "available_ratio": int(available_rows) / int(rows),
        "age_min_seconds": float(age_min),
        "age_p50_seconds": float(age_p50),
        "age_p99_seconds": float(age_p99),
        "age_max_seconds": float(age_max),
        "invalid_rows": int(invalid_rows),
    }
    result["audit_sha256"] = _canonical_sha256(result)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit sampled aggregate-depth coverage for one action-value report"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    result = audit_depth_coverage(args.report, args.warehouse, args.output)
    print(
        "sampled-depth-coverage: "
        f"rows={result['rows']} available={result['available_rows']} "
        f"sha256={result['audit_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
