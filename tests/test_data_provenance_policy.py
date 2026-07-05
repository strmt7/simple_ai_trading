from __future__ import annotations

from tools import audit_data_provenance
from tools.audit_data_provenance import audit


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
    timeline.write_bytes(b"\x1f\x8b")
    chart.write_text("<svg></svg>\n", encoding="utf-8")
    report.write_text(
        """
        {
          "artifact_class": "exchange_sourced_backtest_graph_data",
          "tracked_repo_artifact": true,
          "tracked_artifacts": [
            "docs/optimization/round-001/data/report.json",
            "docs/optimization/round-001/data/backtest-metrics.csv",
            "docs/optimization/round-001/data/BTCUSDT-timeline.csv.gz",
            "docs/optimization/round-001/charts/BTCUSDT.svg"
          ]
        }
        """,
        encoding="utf-8",
    )

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

    report.write_text('{"artifact_class":"exchange_sourced_backtest_graph_data"}', encoding="utf-8")
    failures = audit_data_provenance.audit()
    assert any("requires real-data provenance" in item for item in failures)
