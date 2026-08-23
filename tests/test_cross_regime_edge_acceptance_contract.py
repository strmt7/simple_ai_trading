from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/model-research/cross-regime-edge-acceptance-contract-v1.json"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def test_cross_regime_edge_contract_fails_closed_without_promising_no_loss() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    claimed = contract.pop("contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["schema_version"] == "cross-regime-edge-acceptance-contract-v1"
    assert contract["claim_boundary"] == {
        "cannot_lose_claim_permitted": False,
        "every_regime_must_generate_trades": False,
        "missing_or_inadequate_slice_may_be_extrapolated": False,
        "portfolio_survival_is_independent_of_model_output": True,
        "profitable_in_every_future_market_is_guaranteed": False,
        "unsupported_regime_action": "abstain_from_new_exposure",
    }
    assert set(contract["regime_definition"]["direction_axis"]) == {
        "bullish",
        "bearish",
        "sideways",
    }
    assert "choppy" in contract["regime_definition"]["path_axis"]
    assert contract["risk_independence"]["ai_or_ml_may_override_a_safety_gate"] is False
    assert contract["risk_independence"]["leverage_can_create_edge"] is False
    assert contract["authority"] == {
        "edge_claim": False,
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }
