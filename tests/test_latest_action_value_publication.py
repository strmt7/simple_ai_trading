from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "model-research" / "action-value" / "latest"
PUBLICATION_SHA256 = (
    "6060c119a248d8f5ecb121a5ebd3905741216413dacfb2b9253cc25f5ad4ce7e"
)
EVALUATION_SHA256 = (
    "65900fa58299d56fffa04206dcff83a343e9b005ca71f654efe4e939508d3e3d"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS = (30, 60, 300)
LAYERS = ("perpetual_only", "spot_perpetual")
HEADS = ("binary_direction", "continuous_return_bps")
FOLDS = tuple(range(1, 7))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _csv(name: str) -> list[dict[str, str]]:
    with (LATEST / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(name: str) -> dict[str, object]:
    value = json.loads((LATEST / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_embedded_hash(name: str, key: str, expected: str) -> dict[str, object]:
    value = _json(name)
    canonical = dict(value)
    claimed = canonical.pop(key)
    assert claimed == expected
    assert claimed == _canonical_sha256(canonical)
    return value


def test_latest_action_value_publication_is_exact_round72_evidence() -> None:
    publication = _json("report.json")
    canonical = dict(publication)
    claimed = canonical.pop("publication_canonical_sha256")

    assert claimed == PUBLICATION_SHA256
    assert claimed == _canonical_sha256(canonical)
    assert publication["schema_version"] == (
        "round-072-price-discovery-publication-v1"
    )
    assert publication["round"] == 72
    assert publication["claims"] == {
        "status": "rejected",
        "terminal_holdout_read": False,
        "profitability_claim": False,
        "execution_or_fill_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "leverage_authority": False,
    }
    assert publication["result"] == {
        "decision": "reject_round_072_price_discovery",
        "primary_gate_passed": False,
        "feature_comparison_passes": 0,
        "feature_comparisons": 36,
        "component_passes": 0,
        "components": 9,
        "models": 216,
        "backend": {
            "device": "opencl:auto",
            "kind": "opencl",
            "lightgbm_version": "4.6.0",
            "requested": "auto",
        },
    }
    for path_key, oid_key in (
        ("publisher_path", "publisher_git_blob_oid"),
        ("chart_renderer_path", "chart_renderer_git_blob_oid"),
    ):
        path = ROOT / str(publication[path_key])
        assert path.is_file()
        assert _git_blob_oid(path) == publication[oid_key]

    declared = {str(item["path"]) for item in publication["artifacts"]}
    actual = {
        path.relative_to(LATEST).as_posix()
        for path in LATEST.rglob("*")
        if path.is_file()
    }
    assert len(declared) == 17
    assert actual == declared | {"report.json"}
    for artifact in publication["artifacts"]:
        path = LATEST / str(artifact["path"])
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_round72_evaluation_is_complete_reproducible_and_fail_closed() -> None:
    report = _verify_embedded_hash(
        "evaluation.json", "report_sha256", EVALUATION_SHA256
    )
    source = _json("report.json")["source"]
    assert hashlib.sha256((LATEST / "evaluation.json").read_bytes()).hexdigest() == (
        source["evaluation_file_sha256"]
    )
    assert report["decision"] == "reject_round_072_price_discovery"
    assert report["primary_gate_passed"] is False
    assert report["feature_increment_gate_passed"] is False
    for claim in (
        "profitability_claim",
        "execution_or_fill_claim",
        "trading_authority",
        "leverage_authority",
    ):
        assert report[claim] is False
    assert report["scope"] == {
        "development_last_month": "2026-03",
        "feature_layers": list(LAYERS),
        "heads": list(HEADS),
        "horizons_seconds": list(HORIZONS),
        "profit_or_execution_target": False,
        "symbols": list(SYMBOLS),
        "terminal_holdout_read": False,
        "terminal_months_excluded": ["2026-04", "2026-05", "2026-06"],
    }

    expected_models = set(itertools.product(SYMBOLS, HORIZONS, LAYERS, HEADS, FOLDS))
    models = report["models"]
    assert len(models) == 216
    assert {
        (
            row["symbol"],
            row["horizon_seconds"],
            row["feature_layer"],
            row["head"],
            row["fold"],
        )
        for row in models
    } == expected_models
    assert all(
        row["reload_max_absolute_prediction_difference"] == 0.0
        and row["training_rows"] > 0
        and row["tuning_rows"] > 0
        and row["test_rows"] > 0
        for row in models
    )
    assert len(report["layer_reports"]) == 36
    assert len(report["feature_comparisons"]) == 36
    assert not any(row["passed"] for row in report["feature_comparisons"])
    assert len(report["symbol_horizon_components"]) == 9
    assert not any(row["passed"] for row in report["symbol_horizon_components"])
    assert report["corpus_certificate"] == {
        "compressed_bytes": 5_964_131_852,
        "day_count": 69,
        "first_period": "2020-10-19",
        "flow_rows": 17_884_800,
        "inventory_sha256": (
            "e8c505132716c68ad753cbdd93b23094b778d9067c8a6c9381fad0e20cdd662c"
        ),
        "last_period": "2026-06-22",
        "manifest_fingerprint": (
            "98630d694488ef492963cbbbb68d78d0569b91f34610d150fb8c967967e30925"
        ),
        "research_round": 72,
        "schema_version": "spot-perpetual-corpus-certificate-v1",
        "source_count": 414,
        "source_fingerprint": (
            "1f7e6c1190529991c205de3fe258c900520e4fcf06415391ceafe560f0c5b653"
        ),
        "status": "complete",
        "symbol_count": 3,
        "uncompressed_bytes": 31_726_079_288,
    }


def test_round72_tables_and_source_contracts_reconcile() -> None:
    components = _csv("components.csv")
    comparisons = _csv("feature-comparisons.csv")
    metrics = _csv("metrics.csv")
    models = _csv("models.csv")
    progress = _csv("progress.csv")
    assert (len(components), len(comparisons), len(metrics), len(models)) == (
        9,
        36,
        108,
        216,
    )
    assert all(row["component_passed"] == "false" for row in components)
    assert all(row["passed"] == "false" for row in comparisons)
    assert sum(float(row["relative_improvement"]) > 0.0 for row in comparisons) == 15
    q_values = [float(row["q_value"]) for row in comparisons]
    assert math.isclose(min(q_values), 0.9695030496950304, abs_tol=1e-15)
    assert math.isclose(max(q_values), 0.9814018598140186, abs_tol=1e-15)
    assert all(row["evaluation_report_sha256"] == EVALUATION_SHA256 for row in metrics)
    assert {
        (
            row["symbol"],
            int(row["horizon_seconds"]),
            row["feature_layer"],
            row["head"],
            int(row["fold"]),
        )
        for row in models
    } == set(itertools.product(SYMBOLS, HORIZONS, LAYERS, HEADS, FOLDS))
    assert all(float(row["reload_max_absolute_prediction_difference"]) == 0.0 for row in models)

    assert [int(row["round"]) for row in progress] == list(range(1, 73))
    assert progress[61]["status"] == "passed_predictive_gate"
    assert {
        row["status"] for row in progress[62:70]
    } == {"implementation_only_no_backtest"}
    assert progress[70]["status"] == "research_only_no_model"
    assert progress[71]["status"] == "rejected"
    assert progress[71]["best_model_id"] == "spot_perpetual_increment_rejected"

    source_hashes = {
        "corpus-ingestion.json": (
            "report_sha256",
            "1d7791db923f1d1a7eddc8189934424795246ea01250f6dbef26a59483605adb",
        ),
        "design.json": (
            "design_sha256",
            "505818f74cdd9484f66b1a504de821d4d366ec35c7d7bf978ffa15d613104812",
        ),
        "inventory.json": (
            "inventory_sha256",
            "e8c505132716c68ad753cbdd93b23094b778d9067c8a6c9381fad0e20cdd662c",
        ),
        "implementation.json": (
            "implementation_sha256",
            "d8679606e75ec7fa2bf00032b34489218085f7c7f5159419e192f3ee351dfad9",
        ),
        "decision-analysis.json": (
            "analysis_sha256",
            "65747d7d75c8eb5dcb16eacbf2a78e62fccaa2b3b20ee50db4e7f7d0ddbe7b97",
        ),
    }
    for name, (key, expected) in source_hashes.items():
        _verify_embedded_hash(name, key, expected)
    corpus = _json("corpus-ingestion.json")
    assert corpus["status"] == "complete"
    assert corpus["completed_days"] == 69
    assert corpus["completed_files"] == 414
    assert corpus["raw_aggregate_trades_retained"] is False
    assert corpus["selected_archives_retained"] is False


def test_round72_charts_and_readme_are_accessible_and_current() -> None:
    expected = {
        "day-block-confidence.svg",
        "primary-binary-skill.svg",
        "primary-continuous-skill.svg",
        "research-progress.svg",
        "spot-perpetual-increment.svg",
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
        for invalid in (">nan<", '="nan"', ">inf<", '="inf"', 'height="-'):
            assert invalid not in text

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 72: Spot-Perpetual Price Discovery")
    assert "Rejected. No profitability or trading claim." in readme
    assert "17,884,800 one-second rows" in readme
    assert "crypto trade continuously" in readme
    assert "Listed ETFs and futures follow their own venue calendars" in readme
    assert "not continuous tick coverage of every day" in readme
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        assert (LATEST / target).is_file()
    stale = {
        "cumulative-stress-net.svg",
        "pnl-decomposition.svg",
        "source-capacity-eligibility.svg",
        "stress-net-economics.svg",
    }
    assert not any((LATEST / "charts" / name).exists() for name in stale)
