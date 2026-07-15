from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from simple_ai_trading.make_take_action_features import MakeTakeFeatureSpec
from simple_ai_trading.make_take_evaluation import MakeTakeEconomicGateSpec
from simple_ai_trading.make_take_payoff_lightgbm import (
    MAKE_TAKE_PAYOFF_SEEDS,
    MakeTakePayoffLightGBMSpec,
)
from simple_ai_trading.make_take_policy import MakeTakePolicySpec
from simple_ai_trading.microstructure_barriers import AdaptiveBarrierSpec
from simple_ai_trading.queue_fill_lightgbm import (
    QUEUE_FILL_SEEDS,
    QueueFillLightGBMSpec,
)
from tools.run_round57_queue_censored_make_take import load_round57_contract


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-057-queue-censored-make-take-design.json"
)
CONTRACT_PATH = DESIGN_PATH.with_name(
    "round-057-queue-censored-make-take-execution-contract.json"
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


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def test_round57_model_seeds_match_frozen_design() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    frozen_seeds = tuple(design["model_contract"]["lightgbm"]["seeds"])

    assert design["status"] == "frozen"
    assert QUEUE_FILL_SEEDS == frozen_seeds
    assert MAKE_TAKE_PAYOFF_SEEDS == frozen_seeds


def test_round57_execution_contract_matches_implementation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    canonical = dict(contract)
    claimed = canonical.pop("contract_sha256")
    barrier = AdaptiveBarrierSpec(**contract["barrier_spec"])

    assert contract["status"] == "frozen"
    assert contract["design_sha256"] == json.loads(
        DESIGN_PATH.read_text(encoding="utf-8")
    )["design_sha256"]
    assert claimed == _canonical_sha256(canonical)
    assert tuple(contract["model_aggregation"]["seeds"]) == QUEUE_FILL_SEEDS
    assert _json_value(asdict(MakeTakeFeatureSpec())) == contract["feature_spec"]
    assert _json_value(asdict(QueueFillLightGBMSpec())) == contract[
        "queue_fill_model_spec"
    ]
    assert _json_value(asdict(MakeTakePayoffLightGBMSpec())) == contract[
        "payoff_model_spec"
    ]
    assert _json_value(asdict(barrier)) == contract["barrier_spec"]
    assert _json_value(asdict(MakeTakePolicySpec())) == contract["policy_spec"]
    assert _json_value(asdict(MakeTakeEconomicGateSpec())) == contract[
        "economic_gate_spec"
    ]


def test_round57_runner_accepts_only_the_frozen_contract() -> None:
    design, contract, design_sha, contract_sha = load_round57_contract(
        DESIGN_PATH,
        CONTRACT_PATH,
    )

    assert design_sha == design["design_sha256"]
    assert contract_sha == contract["contract_sha256"]
    assert set(contract["roles"]) == {
        "training",
        "early_stop",
        "probability_calibration",
        "policy_calibration",
        "evaluation",
    }
