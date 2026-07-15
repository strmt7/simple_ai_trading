from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"


def test_round_002_publication_is_internally_consistent() -> None:
    report = json.loads(
        (RESEARCH / "round-002-prospective-pipeline-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    with (RESEARCH / "round-002-market-rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        markets = list(csv.DictReader(handle))

    assert report["round"] == 2
    assert report["status"] == "pipeline_verified_model_evaluation_blocked"
    assert len(report["recorder"]["report_sha256"]) == 64
    assert len(report["dataset"]["dataset_sha256"]) == 64
    assert len(markets) == report["recorder"]["market_snapshot_count"] == 12
    assert sum(int(row["feature_rows"]) for row in markets) == report["dataset"][
        "row_count"
    ]
    for asset, evidence in report["per_asset"].items():
        asset_rows = [row for row in markets if row["asset"] == asset]
        assert len(asset_rows) == evidence["official_resolutions"] == 4
        assert sum(int(row["feature_rows"]) for row in asset_rows) == evidence[
            "feature_rows"
        ]
        assert sum(int(row["feature_rows"]) > 0 for row in asset_rows) == report[
            "dataset"
        ]["labeled_market_counts"][asset]

    chart = RESEARCH / "latest" / "charts" / "causal-feature-coverage.svg"
    root = ET.fromstring(chart.read_text(encoding="utf-8"))
    chart_text = " ".join(root.itertext())
    for value in ("1,278", "1,158", "1,022", "30"):
        assert value in chart_text
    assert "not a profitability chart" in chart_text.lower()
    manifest = {entry["path"]: entry for entry in report["artifact_integrity"]}
    assert set(report["tracked_artifacts"]) - {
        "docs/model-research/polymarket/round-002-prospective-pipeline-evidence.json"
    } == set(manifest)
    for relative_path, expected in manifest.items():
        artifact = ROOT / relative_path
        payload = artifact.read_bytes()
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    market_manifest = manifest[
        "docs/model-research/polymarket/round-002-market-rows.csv"
    ]
    assert market_manifest["row_count"] == len(markets)
    assert market_manifest["columns"] == list(markets[0])
