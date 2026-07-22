from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-impact-absorption-design.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_round73_design_is_sealed_and_fail_closed() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert design["round"] == 73
    assert design["source_contract"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert design["source_contract"]["market_calendar"].startswith("continuous")
    assert design["source_contract"]["listed_etf_or_equity_close_feature"] is False
    assert design["order_book_integrity_contract"]["sequence_gap_policy"].startswith(
        "invalidate"
    )
    assert design["order_book_integrity_contract"]["queue_overflow_policy"].startswith(
        "invalidate"
    )
    assert design["model_contract"]["temporal_neural_challenger_permitted"] is False
    assert design["model_contract"]["reinforcement_learning_permitted"] is False
    assert design["model_contract"]["ai_veto_permitted"] is False
    assert design["economic_gate_after_predictive_pass"]["unlevered_only"] is True
    assert design["evaluation_contract"]["minimum_symbols_for_portfolio_research"] == 2
    assert design["governance"]["profitability_claim_permitted"] is False
    assert design["governance"]["trading_authority_permitted"] is False
