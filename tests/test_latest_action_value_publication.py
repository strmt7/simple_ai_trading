from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "model-research" / "action-value" / "latest"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_latest_action_value_publication_is_round33_hash_verified_and_accessible() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert report["schema_version"] == "selective-action-viability-publication-v1"
    assert report["round"] == 33
    assert report["status"] == "rejected"
    assert report["stage_access"] == {
        "calibration_prediction": True,
        "calibration_threshold_selection": False,
        "policy": False,
        "development": False,
        "distant_confirmation": False,
    }
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert report[field] is False

    for artifact in report["artifact_integrity"]:
        path = LATEST / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                assert sum(1 for _row in csv.DictReader(handle)) == artifact["row_count"]

    with (LATEST / "progress.csv").open("r", encoding="utf-8", newline="") as handle:
        progress = list(csv.DictReader(handle))
    assert [int(row["round"]) for row in progress] == list(range(1, 34))
    assert progress[-1]["status"] == "rejected"
    assert progress[-1]["calibration_eligible_rows"] == "0"
    assert progress[-1]["calibration_threshold_traces"] == "0"

    assert not (LATEST / "candidates.csv").exists()
    assert not (LATEST / "thresholds.csv").exists()
    assert not (LATEST / "charts" / "candidate-economics.svg").exists()
    assert not (LATEST / "charts" / "threshold-economics.svg").exists()
    assert (LATEST / "architecture.csv").is_file()
    assert (LATEST / "charts" / "architecture-gates.svg").is_file()
    for chart in (LATEST / "charts").glob("*.svg"):
        document = ET.parse(chart).getroot()
        namespace = "{http://www.w3.org/2000/svg}"
        assert document.attrib["role"] == "img"
        assert document.find(f"{namespace}title") is not None
        assert document.find(f"{namespace}desc") is not None
        text = chart.read_text(encoding="utf-8").casefold()
        assert ">nan<" not in text
        assert '="nan"' not in text

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert "Round 33" in readme
    assert "Round 32" not in readme
    assert "not a reason to loosen risk controls" in readme
