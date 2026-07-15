from __future__ import annotations

import json
from pathlib import Path

from simple_ai_trading.make_take_payoff_lightgbm import MAKE_TAKE_PAYOFF_SEEDS
from simple_ai_trading.queue_fill_lightgbm import QUEUE_FILL_SEEDS


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-057-queue-censored-make-take-design.json"
)


def test_round57_model_seeds_match_frozen_design() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    frozen_seeds = tuple(design["model_contract"]["lightgbm"]["seeds"])

    assert design["status"] == "frozen"
    assert QUEUE_FILL_SEEDS == frozen_seeds
    assert MAKE_TAKE_PAYOFF_SEEDS == frozen_seeds
