from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (  # noqa: E402
    _canonical_hash,
    _sha256,
)
from tools.screen_polymarket_wallet_opposite_lock import (  # noqa: E402
    _all_in_cost,
    _screen,
)


CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-wallet-opposite-lock-validation-contract-v1-2026-08-31.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-wallet-opposite-lock-validation-result-v1-2026-08-31.json"
)
CANDIDATE = (
    ROOT
    / "docs/model-research/action-value/polymarket-existing-inventory-opposite-lock-candidate-v1-2026-08-31.json"
)
ROUNDING_CORRECTION = (
    ROOT
    / "docs/model-research/action-value/polymarket-wallet-opposite-lock-rounding-correction-v1-2026-08-31.json"
)
ROBUSTNESS_CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-wallet-opposite-lock-robustness-contract-v1-2026-08-31.json"
)
ROBUSTNESS_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-wallet-opposite-lock-robustness-result-v1-2026-08-31.json"
)
FEES = (
    ROOT
    / "docs/model-research/action-value/raw/polymarket-organic-taker-rebate-v1/07-fees.html"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def test_published_usdc_fee_is_added_to_share_price() -> None:
    assert _all_in_cost(Decimal("0.50"), Decimal("0.07")) == Decimal("0.5175")
    text = FEES.read_text(encoding="utf-8")
    assert "fee = C" in text and "feeRate" in text and "p × (1 - p)" in text
    assert "number of shares traded" in text
    assert "price of the shares" in text
    assert "Taker fees are calculated in USDC" in text


def test_synthetic_opposite_fill_locks_only_after_causal_lag() -> None:
    wallet = "0x1111111111111111111111111111111111111111"
    contract = {
        "wallet": wallet,
        "request": {"limit": 100},
        "validation_window": {
            "start_epoch_inclusive": 1000,
            "end_epoch_exclusive": 2000,
        },
        "economics": {
            "assets": ["btc", "eth", "sol"],
            "durations": {"5m": 300, "15m": 900, "4h": 14400},
            "taker_fee_rate": "0.07",
            "tick_size": "0.01",
            "hedge_adverse_ticks": 1,
            "minimum_lag_seconds": 1,
            "maximum_lag_seconds": 60,
            "minimum_seconds_before_close": 10,
        },
        "validation_gates": {
            "minimum_locks": 1,
            "minimum_conditions": 1,
            "minimum_positive_assets": 1,
            "minimum_conditions_per_half": 0,
            "maximum_single_condition_pnl_share": "1",
        },
    }
    base = {
        "conditionId": "condition",
        "eventSlug": "btc-updown-5m-1000",
        "proxyWallet": wallet,
        "side": "BUY",
        "size": "5",
    }
    analysis = _screen(
        contract,
        [
            {**base, "timestamp": 1100, "outcomeIndex": 0, "price": "0.40"},
            {**base, "timestamp": 1105, "outcomeIndex": 1, "price": "0.50"},
        ],
    )
    assert analysis["analysis"]["lock_count"] == 1
    assert analysis["analysis"]["matched_shares"] == "5"
    assert analysis["analysis"]["stress_locked_pnl_pusd"] == "0.278535"


def test_frozen_result_and_candidate_are_source_bound_and_fail_closed() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    candidate = _load(CANDIDATE)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _canonical_hash(candidate, "result_sha256") == candidate["result_sha256"]
    assert (
        _sha256(FEES.read_bytes())
        == candidate["fee_semantics_audit"]["source_file_sha256"]
    )  # type: ignore[index]

    raw = ROOT / result["capture"]["receipt"]["raw_path"]  # type: ignore[index]
    assert _sha256(raw.read_bytes()) == result["capture"]["receipt"]["response_sha256"]  # type: ignore[index]
    assert result["analysis"]["lock_count"] == 66  # type: ignore[index]
    assert (
        result["analysis"]["stress_locked_pnl_pusd"]
        == "217.6302921908223351705204892211"
    )  # type: ignore[index]
    assert candidate["adjudication"]["accepted_edge"] is False  # type: ignore[index]
    assert candidate["adjudication"]["profitability_claim"] is False  # type: ignore[index]
    assert candidate["adjudication"]["public_forward_profit_floor_pusd"] == "0"  # type: ignore[index]


def test_fixed_lock_set_survives_conservative_fee_rounding_ceiling() -> None:
    result = _load(RESULT)
    correction = _load(ROUNDING_CORRECTION)
    candidate = _load(CANDIDATE)

    assert _canonical_hash(correction, "result_sha256") == correction["result_sha256"]
    assert correction["source_binding"]["base_result_sha256"] == result["result_sha256"]  # type: ignore[index]
    assert correction["source_binding"]["fees_source_file_sha256"] == _sha256(
        FEES.read_bytes()
    )  # type: ignore[index]
    assert correction["method"]["base_lock_set_reselected"] is False  # type: ignore[index]
    assert correction["analysis"]["base_lock_count"] == 66  # type: ignore[index]
    assert correction["analysis"]["nonpositive_lock_count_after_correction"] == 0  # type: ignore[index]
    drag = Decimal(correction["analysis"]["conservative_fee_rounding_drag_pusd"])  # type: ignore[index]
    corrected = Decimal(correction["analysis"]["corrected_locked_pnl_pusd"])  # type: ignore[index]
    minimum = Decimal(correction["analysis"]["minimum_corrected_lock_pnl_pusd"])  # type: ignore[index]
    assert Decimal("0") < drag < Decimal(66 * 2) * Decimal("0.00001")
    assert corrected == Decimal("217.62970125422710510000000000000000000000000000000")
    assert minimum > 0
    assert correction["adjudication"]["accepted_edge"] is False  # type: ignore[index]
    assert correction["adjudication"]["public_forward_profit_floor_pusd"] == "0"  # type: ignore[index]
    assert correction["authority"]["network_requests"] == 0  # type: ignore[index]
    assert (
        candidate["fee_rounding_correction"]["result_sha256"]
        == correction["result_sha256"]
    )  # type: ignore[index]


def test_fixed_lock_set_fails_predeclared_cross_lock_robustness_gate() -> None:
    contract = _load(ROBUSTNESS_CONTRACT)
    result = _load(ROBUSTNESS_RESULT)
    candidate = _load(CANDIDATE)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]  # type: ignore[index]
    assert result["method"]["base_lock_set_reselected"] is False  # type: ignore[index]
    assert result["authority"]["network_requests"] == 0  # type: ignore[index]
    assert result["analysis"]["base_lock_count"] == 66  # type: ignore[index]
    assert result["analysis"]["predeclared_gate_passed"] is False  # type: ignore[index]

    gate = result["analysis"]["predeclared_gate_scenario"]  # type: ignore[index]
    assert gate["total_hedge_adverse_ticks"] == 5
    assert gate["fixed_operating_cost_per_lock_pusd"] == "0.05"
    assert gate["positive_lock_count"] == 43
    assert Decimal(gate["positive_lock_fraction"]) < Decimal("0.80")
    assert Decimal(gate["total_locked_pnl_pusd"]) > 0
    assert gate["all_asset_aggregate_pnl_positive"] is True

    robustness = candidate["fixed_lock_robustness_audit"]  # type: ignore[index]
    assert robustness["result_sha256"] == result["result_sha256"]
    assert robustness["predeclared_gate_passed"] is False
    assert candidate["adjudication"]["accepted_edge"] is False  # type: ignore[index]
    assert candidate["adjudication"]["public_forward_profit_floor_pusd"] == "0"  # type: ignore[index]
