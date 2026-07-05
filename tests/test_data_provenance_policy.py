from __future__ import annotations

import csv
import gzip
import hashlib
import json

from tools import audit_data_provenance
from tools.audit_data_provenance import audit


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
