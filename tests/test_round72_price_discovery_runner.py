from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_round72_price_discovery_screen import (
    CORPUS_REPORT_DEFAULT,
    METRICS_DEFAULT,
    OUTPUT_DEFAULT,
    _canonical_sha256,
    _load_corpus_report,
    _metrics_rows,
    build_parser,
)


INVENTORY_SHA256 = "e8c505132716c68ad753cbdd93b23094b778d9067c8a6c9381fad0e20cdd662c"


def _certificate() -> dict[str, object]:
    return {
        "schema_version": "spot-perpetual-corpus-certificate-v1",
        "research_round": 72,
        "inventory_sha256": INVENTORY_SHA256,
        "status": "complete",
        "day_count": 69,
        "source_count": 414,
        "symbol_count": 3,
        "flow_rows": 17_884_800,
        "compressed_bytes": 5_964_131_852,
        "uncompressed_bytes": 10_000_000_000,
        "first_period": "2020-10-19",
        "last_period": "2026-06-01",
        "manifest_fingerprint": "a" * 64,
        "source_fingerprint": "b" * 64,
    }


def _corpus_report(warehouse: Path) -> dict[str, object]:
    without_hash: dict[str, object] = {
        "schema_version": "round-072-spot-perpetual-corpus-ingestion-v1",
        "round": 72,
        "status": "complete",
        "warehouse_path": str(warehouse),
        "inventory_sha256": INVENTORY_SHA256,
        "completed_days": 69,
        "completed_files": 414,
        "completed_compressed_bytes": 5_964_131_852,
        "raw_aggregate_trades_retained": False,
        "selected_archives_retained": False,
        "profitability_claim": False,
        "execution_or_fill_claim": False,
        "trading_authority": False,
        "corpus_certificate": _certificate(),
    }
    return {**without_hash, "report_sha256": _canonical_sha256(without_hash)}


def test_round72_runner_requires_hash_bound_complete_corpus_report(
    tmp_path: Path,
) -> None:
    warehouse = (tmp_path / "microstructure.duckdb").resolve()
    path = tmp_path / "corpus.json"
    report = _corpus_report(warehouse)
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded, certificate = _load_corpus_report(
        path,
        inventory_sha256=INVENTORY_SHA256,
        warehouse_path=warehouse,
    )

    assert loaded["report_sha256"] == report["report_sha256"]
    assert certificate == _certificate()

    report["completed_files"] = 413
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="completion report"):
        _load_corpus_report(
            path,
            inventory_sha256=INVENTORY_SHA256,
            warehouse_path=warehouse,
        )


def test_round72_runner_metrics_rows_keep_report_provenance() -> None:
    report = {
        "report_sha256": "c" * 64,
        "layer_reports": [
            {
                "symbol": "BTCUSDT",
                "horizon_seconds": 30,
                "feature_layer": "spot_perpetual",
                "head": "binary_direction",
                "prevalence_comparison": {
                    "log_loss": {
                        "model": 0.60,
                        "training_prevalence_baseline": 0.69,
                        "relative_improvement": 0.13,
                    },
                    "brier_score": {
                        "model": 0.20,
                        "training_prevalence_baseline": 0.25,
                        "relative_improvement": 0.20,
                    },
                },
            }
        ],
        "feature_comparisons": [
            {
                "symbol": "BTCUSDT",
                "horizon_seconds": 30,
                "head": "binary_direction",
                "metric": "log_loss",
                "challenger_mean_loss": 0.60,
                "baseline_mean_loss": 0.61,
                "relative_improvement": 0.016,
                "q_value": 0.01,
                "passed": True,
            }
        ],
    }

    rows = _metrics_rows(report)

    assert len(rows) == 3
    assert all(row["evaluation_report_sha256"] == "c" * 64 for row in rows)
    assert rows[-1]["record_type"] == "spot_perpetual_vs_perpetual_only"


def test_round72_runner_defaults_are_non_overlapping() -> None:
    arguments = build_parser().parse_args([])

    assert Path(arguments.corpus_report) == CORPUS_REPORT_DEFAULT
    assert Path(arguments.output) == OUTPUT_DEFAULT
    assert Path(arguments.metrics) == METRICS_DEFAULT
    assert len(
        {
            arguments.design,
            arguments.inventory,
            arguments.implementation,
            arguments.warehouse,
            arguments.corpus_report,
            arguments.output,
            arguments.metrics,
            arguments.progress,
        }
    ) == 8
