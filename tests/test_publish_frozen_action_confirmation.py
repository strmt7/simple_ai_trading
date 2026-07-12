from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import tools.publish_frozen_action_confirmation as publisher
from tools.run_gross_architecture_screen import (
    _canonical_sha256,
    _sha256_file,
)


DESIGN = (
    publisher.ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-031-frozen-chronological-confirmation-design.json"
)
PRIOR_PROGRESS = (
    publisher.ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "latest"
    / "progress.csv"
)


def _forecast_side(*, auc: float) -> dict[str, object]:
    return {
        "rows": 5_000,
        "profitable_auc": auc,
        "profitable_brier": 0.22,
        "prevalence_brier": 0.24,
        "pearson_information_coefficient": 0.08,
        "mean_actual_net_bps": -0.5,
        "top_rows": [
            {
                "requested_rows": requested,
                "actual_rows": requested,
                "mean_actual_net_bps": value,
            }
            for requested, value in ((100, 1.2), (500, 0.4), (1_000, -0.1))
        ],
    }


def _candidate(quantile: float, threshold: float) -> dict[str, object]:
    def trace(net_bps: float) -> dict[str, object]:
        return {
            "metrics": {
                "trades": 7,
                "total_net_bps": net_bps,
                "mean_net_bps": net_bps / 7.0,
                "max_drawdown_bps": 3.5,
                "profit_factor": 1.1,
                "win_rate": 0.57,
                "worst_trade_bps": -2.0,
            }
        }

    return {
        "quantile": quantile,
        "threshold_bps": threshold,
        "passed": False,
        "gate_reasons": ["minimum_trades"],
        "drawdown_adjusted_utility_bps": -2.0,
        "base_trace": trace(2.0),
        "stress_trace": trace(1.0),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _write_evidence(root: Path) -> Path:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    candidates = design["frozen_thresholds"]
    stage = {
        "stage": "confirmation",
        "data": {
            "evaluation_start": "2024-01-01",
            "evaluation_end": "2024-02-04",
            "valid_target_rows": 5_000,
        },
        "forecast_diagnostics": {
            scenario: {
                "sides": {
                    "long": _forecast_side(auc=0.61),
                    "short": _forecast_side(auc=0.58),
                }
            }
            for scenario in ("base", "stress")
        },
        "profile_results": [
            {
                "profile": profile,
                "eligible_rows": 900,
                "passed": False,
                "selected_threshold_bps": None,
                "candidates": [
                    _candidate(float(item["quantile"]), float(item["threshold_bps"]))
                    for item in candidates[profile]
                ],
            }
            for profile in ("conservative", "regular", "aggressive")
        ],
        "surviving_profiles": [],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    root.mkdir(parents=True)
    stage_path = root / "stage-confirmation.json"
    stage_path.write_text(
        json.dumps(stage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "schema_version": publisher.REPORT_SCHEMA_VERSION,
        "round": 31,
        "design_sha256": design["design_sha256"],
        "status": "rejected",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "policy_window_is_consumed": False,
        "development_window_is_consumed": False,
        "stage_access": {
            "confirmation": True,
            "policy": False,
            "development": False,
        },
        "consumed_period_governance": {
            "excluded_target_dates": design["data"]["excluded_target_dates"],
            "all_consumed_dates_excluded_from_targets": True,
        },
        "stages": {"confirmation": stage},
        "stage_artifacts": {
            "confirmation": {
                "path": stage_path.name,
                "sha256": _sha256_file(stage_path),
                "bytes": stage_path.stat().st_size,
            }
        },
        "final_profiles": [],
    }
    report["report_sha256"] = _canonical_sha256(report)
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stage_path


def test_round31_publication_is_hash_verified_parseable_and_truthful(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    stage_path = _write_evidence(evidence)
    output = tmp_path / "publication"

    report = publisher.publish(
        evidence_root=evidence,
        design_path=DESIGN,
        prior_progress_path=PRIOR_PROGRESS,
        output_dir=output,
    )

    assert report["status"] == "rejected"
    assert report["trading_authority"] is False
    assert report["deepest_opened_stage"] == "confirmation"
    assert report["deepest_stage_candidate_count"] == 12
    assert report["deepest_stage_passing_count"] == 0
    with (output / "progress.csv").open(encoding="utf-8", newline="") as handle:
        progress = list(csv.DictReader(handle))
    assert progress[-1]["round"] == "31"
    assert progress[-1]["status"] == "rejected"
    assert progress[-1]["development_consumed"] == "False"
    assert (output / "README.md").read_text(encoding="utf-8").count(
        "without trading authority"
    ) >= 1
    for chart in (output / "charts").glob("*.svg"):
        root = ET.parse(chart).getroot()
        namespace = "{http://www.w3.org/2000/svg}"
        assert root.attrib["role"] == "img"
        assert root.find(f"{namespace}title") is not None
        assert root.find(f"{namespace}desc") is not None
        chart_text = chart.read_text(encoding="utf-8").casefold()
        assert ">nan<" not in chart_text
        assert '="nan"' not in chart_text

    stage_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage artifact differs"):
        publisher.publish(
            evidence_root=evidence,
            design_path=DESIGN,
            prior_progress_path=PRIOR_PROGRESS,
            output_dir=tmp_path / "tampered",
        )
