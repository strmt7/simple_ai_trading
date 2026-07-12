from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
FAILURE = RESEARCH / "round-034-failure-analysis.json"
REGISTRY33 = RESEARCH / "consumed-periods-through-round-033.json"
REGISTRY34 = RESEARCH / "consumed-periods-through-round-034.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical_sha256(payload: dict[str, object], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round34_failure_analysis_is_hash_bound_and_fail_closed() -> None:
    failure = _read(FAILURE)

    assert failure["analysis_sha256"] == _canonical_sha256(
        failure,
        "analysis_sha256",
    )
    assert failure["status"] == "rejected"
    assert failure["trading_authority"] is False
    assert failure["execution_claim"] is False
    assert failure["profitability_claim"] is False
    assert failure["portfolio_claim"] is False
    assert failure["leverage_applied"] is False
    experiment = failure["next_experiment"]
    assert experiment["post_hoc_consumed_data_discovery_only"] is True
    assert experiment["promotion_permitted"] is False
    assert experiment["hyperparameter_search_permitted"] is False
    assert experiment["risk_gate_relaxation_permitted"] is False
    assert experiment["leverage_permitted"] is False
    assert experiment["oracle_feature_or_label_use_permitted"] is False


def test_round34_failure_analysis_identifies_direction_without_oracle_promotion() -> (
    None
):
    failure = _read(FAILURE)
    architecture = failure["calibration_architecture"]

    assert architecture["opportunity_auc"] >= architecture["minimum_opportunity_auc"]
    assert architecture["side_profit_auc"] >= architecture["minimum_side_profit_auc"]
    assert (
        architecture["conditional_direction_auc"]
        < architecture["minimum_conditional_direction_auc"]
    )
    assert architecture["selected_top_100_mean_stress_net_bps"] < 0.0
    assert architecture["selected_top_500_mean_stress_net_bps"] < 0.0
    assert all(
        route["top_500_mean_stress_net_bps"] < 0.0
        for route in failure["learned_route_diagnostics"]
    )
    assert all(
        control["top_500_mean_stress_net_bps"] > 0.0
        for control in failure["oracle_side_decomposition_controls"]
    )
    assert "future realized outcomes" in failure["oracle_control_warning"]
    assert "prohibited" in failure["oracle_control_warning"]


def test_round34_consumed_registry_extends_round33_without_mutation() -> None:
    previous = _read(REGISTRY33)
    current = _read(REGISTRY34)

    assert current["registry_sha256"] == _canonical_sha256(
        current,
        "registry_sha256",
    )
    assert current["records"][:-1] == previous["records"]
    assert current["records"][-1] == {
        "round": 34,
        "status": "consumed",
        "outcome": "rejected",
        "design_sha256": (
            "61aea61eeec1ddc87aecb8b8fe17b915b8411760501e46eecf9c2f4e92ac5e03"
        ),
        "report_sha256": (
            "60f03a07e996230e8e278ef668fc7b9a1de35e1299fc7921e16a99fa51a53303"
        ),
        "diagnostic_sha256": (
            "05756cff0b99eaf62dc8253cfb4726b16438df2da8528a37b6a779a0f05455c3"
        ),
        "windows": [{"start_date": "2023-05-16", "end_date": "2023-07-06"}],
    }
