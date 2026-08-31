from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION_VALUE
    / "binance-crypto-option-perpetual-lower-bound-retained-contract-v1-2026-08-31.json"
)
RESULT = (
    ACTION_VALUE
    / "binance-crypto-option-perpetual-lower-bound-retained-result-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
TOOL = ROOT / "tools/screen_binance_crypto_option_perpetual_lower_bound_retained.py"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_binds_complete_retained_population_and_frozen_costs() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == (
        "b31c691c728d7d2c7a5d7e13151c57139c6aad1c5fc22fb2be913edb6e5b9a60"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert (
        contract["implementation"]["sha256"]
        == hashlib.sha256(TOOL.read_bytes()).hexdigest()
    )
    assert contract["population"]["expected_eligible_option_count"] == 1410
    assert contract["population"]["allowed_underlyings"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert contract["observation"]["option_closeTime_used_as_quote_timestamp"] is False
    assert sum(
        Decimal(contract["stress"][key])
        for key in (
            "one_option_taker_fee_bps",
            "one_option_settlement_fee_bps",
            "futures_round_trip_fee_bps",
            "perpetual_expiry_basis_bps",
        )
    ) == Decimal("33.5")
    for key, source in contract["retained_sources"].items():
        if key == "funding_histories":
            continue
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    for source in contract["retained_sources"]["funding_histories"].values():
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_long_option_plus_opposite_perpetual_has_the_frozen_terminal_floor() -> None:
    entry = Decimal("101.73")
    strike = Decimal("90")
    call_ask = Decimal("11.56")
    put_ask = Decimal("3.25")
    call_floor = entry - strike - call_ask
    put_floor = strike - entry - put_ask

    for terminal in map(Decimal, ("0", "89", "90", "101.73", "150", "1000")):
        call_package = (
            max(terminal - strike, Decimal("0")) - call_ask + entry - terminal
        )
        put_package = max(strike - terminal, Decimal("0")) - put_ask + terminal - entry
        assert call_package >= call_floor
        assert put_package >= put_floor


def test_all_retained_options_are_exhausted_and_fixed_cost_rejects_best_rows() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == (
        "90c05ed35db00da7e5b4a2d8ec6ac0a51367a1a768dc58a39ef479510d5aa745"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["authority"]["new_public_requests"] == 0
    assert result["authority"]["authenticated_requests"] == 0
    assert result["population"] == {
        "after_all_retained_stress_positive_count": 0,
        "eligible_option_count": 1410,
        "gross_positive_count": 2,
        "positive_entry_side_count": 1115,
    }
    gross_rows = [
        row for row in result["all_rows"] if row["gross_terminal_floor_positive"]
    ]
    assert [row["symbol"] for row in gross_rows] == [
        "SOL-260828-90-C",
        "SOL-260828-92-C",
    ]
    expected = {
        "SOL-260828-90-C": (
            Decimal("0.1700"),
            Decimal("-41.09843077122238240639387453"),
        ),
        "SOL-260828-92-C": (
            Decimal("0.1000"),
            Decimal("-47.97939017356190860318931344"),
        ),
    }
    for row in gross_rows:
        gross, after_bps = expected[row["symbol"]]
        assert Decimal(row["gross_terminal_floor_per_unit_USDT"]) == gross
        assert Decimal(row["fixed_fee_and_basis_bps"]) == Decimal("33.5")
        assert Decimal(row["fixed_fee_and_basis_cost_per_unit_USDT"]) > gross
        assert Decimal(row["after_all_retained_stress_bps"]) == after_bps
        assert row["futures_top_level_capacity_passes"] is True
        assert row["option_depth_quantity_verified"] is False
        assert row["eligible_for_one_exact_option_depth_request"] is False
    assert result["depth_request_candidates"] == []
    assert (
        result["adjudication"]["next_action"] == "stop_without_current_market_requests"
    )


def test_registry_terminalizes_only_the_retained_single_option_population() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_crypto_option_opposite_USDT_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["priority_rank"] == 47
    assert hypothesis["market_direction_forecast_required"] is False
    assert hypothesis["canonical_artifacts"][-1] == {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": _load(RESULT)["result_sha256"],
    }
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_retained_BTC_ETH_SOL_single_long_option_opposite_perpetual_terminal_lower_bound"
    )
    assert terminal["canonical_result_sha256"] == _load(RESULT)["result_sha256"]
