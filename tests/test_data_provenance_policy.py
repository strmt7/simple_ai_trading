from __future__ import annotations

import csv
import gzip
import hashlib
import json

from tools import audit_data_provenance
from tools.audit_data_provenance import audit
from simple_ai_trading.optimization_progress import build_optimization_progress_artifacts


def _artifact_entry(root, relative: str) -> dict[str, object]:
    path = root / relative
    payload = path.read_bytes()
    entry: dict[str, object] = {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if relative.endswith(".csv") or relative.endswith(".csv.gz"):
        opener = gzip.open if relative.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        entry["columns"] = rows[0] if rows else []
        entry["row_count"] = max(0, len(rows) - 1)
    return entry


def test_tracked_repo_does_not_publish_synthetic_financial_evidence() -> None:
    assert audit() == []


def test_audit_allows_declared_exchange_sourced_optimization_artifacts(tmp_path, monkeypatch) -> None:
    report = tmp_path / "docs" / "optimization" / "round-001" / "data" / "report.json"
    metrics = tmp_path / "docs" / "optimization" / "round-001" / "data" / "backtest-metrics.csv"
    timeline = tmp_path / "docs" / "optimization" / "round-001" / "data" / "BTCUSDT-timeline.csv.gz"
    chart = tmp_path / "docs" / "optimization" / "round-001" / "charts" / "BTCUSDT.svg"
    report.parent.mkdir(parents=True)
    chart.parent.mkdir(parents=True)
    metrics.write_text("round_id,symbol\nround-001,BTCUSDT\n", encoding="utf-8")
    with gzip.open(timeline, "wt", encoding="utf-8", newline="") as handle:
        handle.write("timestamp_ms,strategy_equity\n1,1000.0\n")
    chart.write_text("<svg></svg>\n", encoding="utf-8")
    payload = {
        "artifact_class": "exchange_sourced_backtest_graph_data",
        "tracked_repo_artifact": True,
        "tracked_artifacts": [
            "docs/optimization/round-001/data/report.json",
            "docs/optimization/round-001/data/backtest-metrics.csv",
            "docs/optimization/round-001/data/BTCUSDT-timeline.csv.gz",
            "docs/optimization/round-001/charts/BTCUSDT.svg",
        ],
    }
    payload["artifact_integrity"] = [
        _artifact_entry(tmp_path, "docs/optimization/round-001/data/backtest-metrics.csv"),
        _artifact_entry(tmp_path, "docs/optimization/round-001/data/BTCUSDT-timeline.csv.gz"),
        _artifact_entry(tmp_path, "docs/optimization/round-001/charts/BTCUSDT.svg"),
    ]
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(audit_data_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        audit_data_provenance,
        "_tracked_files",
        lambda: [
            "docs/optimization/round-001/data/report.json",
            "docs/optimization/round-001/data/backtest-metrics.csv",
            "docs/optimization/round-001/data/BTCUSDT-timeline.csv.gz",
            "docs/optimization/round-001/charts/BTCUSDT.svg",
        ],
    )

    assert audit_data_provenance.audit() == []

    chart.write_text("<svg><text>tampered</text></svg>\n", encoding="utf-8")
    failures = audit_data_provenance.audit()
    assert any("requires real-data provenance" in item for item in failures)
    chart.write_text("<svg></svg>\n", encoding="utf-8")

    report.write_text('{"artifact_class":"exchange_sourced_backtest_graph_data"}', encoding="utf-8")
    failures = audit_data_provenance.audit()
    assert any("requires real-data provenance" in item for item in failures)


def test_audit_rejects_multiple_historical_optimization_graph_sets(tmp_path, monkeypatch) -> None:
    tracked: list[str] = []
    for round_id in ("round-001", "round-002"):
        report = tmp_path / "docs" / "optimization" / round_id / "data" / "report.json"
        metrics = tmp_path / "docs" / "optimization" / round_id / "data" / "backtest-metrics.csv"
        chart = tmp_path / "docs" / "optimization" / round_id / "charts" / "BTCUSDT.svg"
        report.parent.mkdir(parents=True)
        chart.parent.mkdir(parents=True)
        metrics.write_text("round_id,symbol\n%s,BTCUSDT\n" % round_id, encoding="utf-8")
        chart.write_text("<svg></svg>\n", encoding="utf-8")
        report_rel = f"docs/optimization/{round_id}/data/report.json"
        metrics_rel = f"docs/optimization/{round_id}/data/backtest-metrics.csv"
        chart_rel = f"docs/optimization/{round_id}/charts/BTCUSDT.svg"
        payload = {
            "artifact_class": "exchange_sourced_backtest_graph_data",
            "tracked_repo_artifact": True,
            "tracked_artifacts": [report_rel, metrics_rel, chart_rel],
            "artifact_integrity": [
                _artifact_entry(tmp_path, metrics_rel),
                _artifact_entry(tmp_path, chart_rel),
            ],
        }
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tracked.extend([report_rel, metrics_rel, chart_rel])

    monkeypatch.setattr(audit_data_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_data_provenance, "_tracked_files", lambda: tracked)

    failures = audit_data_provenance.audit()

    assert any("latest-only" in item and "round-001" in item and "round-002" in item for item in failures)


def test_iteration_progress_artifact_is_manifested_and_auditable(tmp_path, monkeypatch) -> None:
    source_report = tmp_path / "docs" / "optimization" / "round-005" / "data" / "report.json"
    source_report.parent.mkdir(parents=True)
    source_report.write_text(
        json.dumps(
            {
                "round_id": "round-005",
                "generated_at_utc": "2026-07-06T00:00:00Z",
                "artifact_class": "exchange_sourced_backtest_graph_data",
                "tracked_repo_artifact": True,
                "market_type": "futures",
                "interval": "1s",
                "objective": "conservative",
                "symbol_count_completed": 3,
                "critical_analysis": {"verdict": "fail"},
                "promotion_grade": False,
                "promotion_grade_contract": {"status": "not_requested"},
                "progress": {
                    "accepted_symbol_count": 0,
                    "mean_roi_pct": -0.5,
                    "median_roi_pct": -0.4,
                    "mean_baseline_roi_pct": 0.2,
                    "worst_max_drawdown_pct": 1.7,
                    "total_closed_trades": 12,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_optimization_progress_artifacts(tmp_path / "docs" / "optimization")
    tracked = [str(item).replace("\\", "/") for item in report["tracked_artifacts"]]

    monkeypatch.setattr(audit_data_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_data_provenance, "_tracked_files", lambda: tracked)

    assert report["latest_round_id"] == "round-005"
    generated_root = tmp_path / "docs" / "optimization" / "iteration-progress"
    assert (generated_root / "data" / "progress.csv").exists()
    assert (generated_root / "charts" / "progress.svg").exists()
    for generated in (
        generated_root / "data" / "progress.csv",
        generated_root / "data" / "report.json",
        generated_root / "charts" / "progress.svg",
    ):
        assert b"\r\n" not in generated.read_bytes()
    assert audit_data_provenance.audit() == []
