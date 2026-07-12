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


def test_latest_action_value_publication_is_round34_hash_verified_and_accessible() -> (
    None
):
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert report["schema_version"] == ("three-action-utility-viability-publication-v1")
    assert report["round"] == 34
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "18cdd663e5fa154c9c6a2521bea35a77eacbba02"
    )
    assert report["source_corpus_certificate_sha256"] == (
        "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
    )
    assert report["source_barrier_targets_sha256"] == (
        "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
    )
    assert report["source_cache_key"] == (
        "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
    )
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
                assert (
                    sum(1 for _row in csv.DictReader(handle)) == artifact["row_count"]
                )

    with (LATEST / "progress.csv").open("r", encoding="utf-8", newline="") as handle:
        progress = list(csv.DictReader(handle))
    assert [int(row["round"]) for row in progress] == list(range(1, 35))
    assert progress[-1]["status"] == "rejected"
    assert progress[-1]["calibration_eligible_rows"] == "39"
    assert progress[-1]["selected_signals"] == "39"
    assert progress[-1]["calibration_threshold_traces"] == "0"
    assert progress[-1]["opportunity_auc"] == "0.6636738763493204"
    assert progress[-1]["side_profit_auc"] == "0.6065548206367726"
    assert progress[-1]["architecture_gates_passed"] == "4"
    assert progress[-1]["architecture_gate_count"] == "7"

    assert not (LATEST / "candidates.csv").exists()
    assert not (LATEST / "thresholds.csv").exists()
    assert not (LATEST / "charts" / "candidate-economics.svg").exists()
    assert not (LATEST / "charts" / "threshold-economics.svg").exists()
    assert (LATEST / "architecture.csv").is_file()
    assert (LATEST / "charts" / "architecture-gates.svg").is_file()
    with (LATEST / "architecture.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        architecture = list(csv.DictReader(handle))
    assert len(architecture) == 7
    assert sum(row["passed"] == "True" for row in architecture) == 4
    assert {row["metric"] for row in architecture} >= {
        "side_profit_roc_auc",
        "side_profit_brier_to_base_rate_ratio",
        "multiclass_log_loss_to_class_prior_ratio",
    }

    with (LATEST / "profiles.csv").open("r", encoding="utf-8", newline="") as handle:
        profiles = list(csv.DictReader(handle))
    assert {row["profile"]: int(row["eligible_rows"]) for row in profiles} == {
        "conservative": 0,
        "regular": 10,
        "aggressive": 39,
    }

    with (LATEST / "forecast.csv").open("r", encoding="utf-8", newline="") as handle:
        forecast = list(csv.DictReader(handle))
    assert {row["diagnostic"] for row in forecast} >= {
        "action_opportunity",
        "conditional_action_direction",
        "side_profit",
        "selected_action",
    }
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
    assert "# Round 34:" in readme
    assert "Round 32" not in readme
    assert "separate from side-profit probabilities" in readme
    assert "not a reason to loosen risk controls" in readme
