from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "model-research" / "action-value" / "latest"
PUBLISHER = ROOT / "tools" / "publish_round61_carry_economic_replay.py"
PUBLICATION_SHA256 = "4e3f697c83ca10df0accf8a9aaa8e37579272ce8148ea4967febeb9542d5c8d4"
REPORT_FILE_SHA256 = "dc9fa604257db59a1b2d1766c70fa8131aafc5cb132238a2f536f13ad2b08908"
REPORT_CANONICAL_SHA256 = (
    "e2f6275232b7f6b7b511211b26a697536a401e14a118393502bcfda96ae4d6e4"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _csv(name: str) -> list[dict[str, str]]:
    with (LATEST / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(name: str) -> dict[str, object]:
    value = json.loads((LATEST / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_latest_action_value_publication_is_round61_hash_verified() -> None:
    publication = _json("report.json")
    canonical = dict(publication)
    claimed = canonical.pop("publication_canonical_sha256")

    assert claimed == PUBLICATION_SHA256
    assert claimed == _canonical_sha256(canonical)
    assert publication["schema_version"] == (
        "round-061-carry-economic-replay-publication-v1"
    )
    assert publication["round"] == 61
    assert publication["publisher_path"] == PUBLISHER.relative_to(ROOT).as_posix()
    assert PUBLISHER.is_file()
    assert publication["claims"] == {
        "status": "rejected_elevated_funding_carry",
        "selection_contaminated": True,
        "tick_execution_replay_authorized": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "model_training_authorized": False,
        "trading_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "leverage_applied": False,
    }
    assert publication["result"] == {
        "passed_symbols": [],
        "required_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "all_symbols_passed": False,
        "authorized_next_step": ("none; reject this elevated-funding carry family"),
    }
    declared = {item["path"] for item in publication["artifacts"]}
    actual = {
        path.relative_to(LATEST).as_posix()
        for path in LATEST.rglob("*")
        if path.is_file()
    }
    assert actual == declared | {"report.json"}
    for artifact in publication["artifacts"]:
        path = LATEST / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_latest_action_value_source_report_is_exact_and_fail_closed() -> None:
    report = _json("screen.json")
    canonical = dict(report)
    claimed = canonical.pop("report_sha256")

    assert claimed == REPORT_CANONICAL_SHA256
    assert claimed == _canonical_sha256(canonical)
    assert hashlib.sha256((LATEST / "screen.json").read_bytes()).hexdigest() == (
        REPORT_FILE_SHA256
    )
    assert report["schema_version"] == "round-061-carry-economic-replay-report-v1"
    assert report["round"] == 61
    assert report["status"] == "rejected_elevated_funding_carry"
    assert report["tick_execution_replay_authorized"] is False
    assert report["synthetic_or_filled_source_rows"] is False
    for field in (
        "model_training_authorized",
        "ai_evaluation_authorized",
        "trading_authority",
        "testnet_or_live_authority",
        "profitability_claim",
        "leverage_applied",
    ):
        assert report[field] is False
    expected = {
        "BTCUSDT": (72, 72, 30),
        "ETHUSDT": (76, 76, 20),
        "SOLUSDT": (62, 61, 0),
    }
    assert [row["symbol"] for row in report["symbol_results"]] == list(expected)
    for result in report["symbol_results"]:
        canonical_result = dict(result)
        result_sha = canonical_result.pop("result_sha256")
        summary = result["summary"]
        assert result_sha == _canonical_sha256(canonical_result)
        assert (
            summary["manifest_episodes"],
            summary["source_eligible_episodes"],
            summary["capacity_eligible_episodes"],
        ) == expected[result["symbol"]]
        assert result["gate"]["passed"] is False


def test_latest_action_value_tables_reconcile_to_round61_report() -> None:
    expected_counts = {
        "summary.csv": 3,
        "episodes.csv": 210,
        "capacity.csv": 836,
        "yearly.csv": 7,
        "gates.csv": 45,
        "cumulative.csv": 50,
        "progress.csv": 61,
    }
    for name, count in expected_counts.items():
        assert len(_csv(name)) == count
    summary = {row["symbol"]: row for row in _csv("summary.csv")}
    assert math.isclose(
        float(summary["BTCUSDT"]["capacity_eligible_fraction"]),
        30 / 72,
        abs_tol=1e-15,
    )
    assert math.isclose(
        float(summary["ETHUSDT"]["median_stress_net_committed_capital_bps"]),
        -5.559926832949844,
        abs_tol=1e-15,
    )
    assert summary["SOLUSDT"]["economically_scored_episodes"] == "0"
    assert summary["SOLUSDT"]["mean_stress_net_committed_capital_bps"] == ""
    episodes = _csv("episodes.csv")
    assert sum(row["source_eligible"] == "false" for row in episodes) == 1
    assert sum(row["economically_scored"] == "true" for row in episodes) == 50
    capacity = _csv("capacity.csv")
    pass_counts = {
        (symbol, fill): sum(
            row["symbol"] == symbol and row["fill"] == fill and row["passed"] == "true"
            for row in capacity
        )
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        for fill in (
            "spot_entry",
            "spot_exit",
            "perpetual_entry",
            "perpetual_exit",
        )
    }
    assert pass_counts[("BTCUSDT", "spot_entry")] == 42
    assert pass_counts[("ETHUSDT", "spot_exit")] == 23
    assert pass_counts[("SOLUSDT", "perpetual_exit")] == 18
    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 62))
    assert progress[-1]["status"] == "rejected"
    assert progress[-1]["best_model_id"] == "elevated_funding_carry_rejected"


def test_latest_action_value_charts_are_accessible_and_round61_only() -> None:
    expected = {
        "cumulative-stress-net.svg",
        "pnl-decomposition.svg",
        "research-progress.svg",
        "source-capacity-eligibility.svg",
        "stress-net-economics.svg",
    }
    charts = {path.name for path in (LATEST / "charts").glob("*.svg")}
    assert charts == expected
    namespace = "{http://www.w3.org/2000/svg}"
    for chart in (LATEST / "charts").glob("*.svg"):
        document = ET.parse(chart).getroot()
        assert document.attrib["role"] == "img"
        assert document.find(f"{namespace}title") is not None
        assert document.find(f"{namespace}desc") is not None
        text = chart.read_text(encoding="utf-8").casefold()
        assert ">nan<" not in text
        assert '="nan"' not in text
        assert ">inf<" not in text
        assert '="inf"' not in text
        assert 'height="-' not in text
    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 61: Matched Spot-Perpetual Economic Replay")
    assert "Rejected. No profitability or trading claim." in readme
    assert "No missing price was interpolated or filled." in readme
    assert "# Round 60:" not in readme
