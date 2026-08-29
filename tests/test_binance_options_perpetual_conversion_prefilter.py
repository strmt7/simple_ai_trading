from __future__ import annotations

import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import screen_binance_options_perpetual_conversion_prefilter as screen
from tools import screen_binance_options_perpetual_conversion_prefilter_v2 as v2
from tools import stress_binance_options_perpetual_conversion_retained as stress


ROOT = Path(__file__).resolve().parents[1]
V1_CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-contract-v1.json"
)
V1_RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-v1-2026-08-29.json"
)
V2_CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-contract-v2.json"
)
V2_RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-v2-2026-08-29.json"
)
STRESS_CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-stress-contract-v1.json"
)
STRESS_RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-stress-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
V1_CONTRACT_HASH = "493c558dc191dafb9e6789db9279a70f4d82eb5e89cef45f833515f806c09ac8"
V1_RESULT_HASH = "c7dfb805da6d55bb3fcccb48cb45baa7bb3044f0d304f26dfc1dd67b3bc7529f"
V2_CONTRACT_HASH = "5cbcc2f302b78da934d63801a58c56e640670e4b31935a6e2542536360502ee5"
V2_RESULT_HASH = "1adf3c51cd008d40744c8ae91e1e4865e9e19e7dbb1ae9a0995e2b2676b3bd58"
STRESS_CONTRACT_HASH = (
    "5ac091035b9eeadda23292fa28631dcc7c8bb0b64e001faa34c94ffad5b6ecc5"
)
STRESS_RESULT_HASH = "c09d62e98cd0df88622d4b98d9d8f01247121ccd786fffb580bc72429ef6bf30"
REGISTRY_HASH = "afa26a57c9ca4525021ef1d728993ecc52a427ac03e8ee3f48bd15ab0203bf71"


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


def _option(symbol: str, side: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "side": side,
        "underlying": "BTCUSDT",
        "expiryDate": 2_000,
        "strikePrice": "100",
        "minQty": "0.01",
        "filters": [{"filterType": "LOT_SIZE", "minQty": "0.01"}],
    }


def _future() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "minQty": "0.001"}],
    }


def _evaluate(
    direction: str,
    *,
    call_ticker: dict[str, object],
    put_ticker: dict[str, object],
    futures_book: dict[str, object],
) -> dict[str, object] | None:
    return screen._evaluate(
        direction=direction,
        option_call=_option("BTC-TEST-100-C", "CALL"),
        option_put=_option("BTC-TEST-100-P", "PUT"),
        call_ticker=call_ticker,
        put_ticker=put_ticker,
        futures_symbol=_future(),
        futures_book=futures_book,
        options_completed_at_ms=1_000,
        futures_completed_at_ms=1_000,
        maximum_option_quote_age_ms=60_000,
        maximum_cross_source_skew_ms=5_000,
    )


def test_v1_frozen_contract_sources_and_implementation_reconstruct() -> None:
    contract = json.loads(V1_CONTRACT.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == V1_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == V1_CONTRACT_HASH
    assert contract["identity"]["market_direction_forecast_required"] is False
    assert contract["stopping_rules"]["current_depth_before_retained_stress"] is False
    parent = contract["retained_sources"]["parent_result"]
    parent_payload = json.loads((ROOT / parent["path"]).read_text(encoding="ascii"))
    assert parent_payload["result_sha256"] == parent["sha256"]
    assert _canonical_hash(parent_payload, "result_sha256") == parent["sha256"]
    for name, source in contract["retained_sources"].items():
        if name == "parent_result":
            continue
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == (source["sha256"])
        )
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == (implementation["sha256"])
    )


def test_conversion_and_reversal_payoff_algebra() -> None:
    conversion = _evaluate(
        "conversion_short_perpetual",
        call_ticker={"bidPrice": "1", "askPrice": "2", "closeTime": 990},
        put_ticker={"bidPrice": "2", "askPrice": "3", "closeTime": 990},
        futures_book={"bidPrice": "101", "askPrice": "102", "time": 995},
    )
    reversal = _evaluate(
        "reversal_long_perpetual",
        call_ticker={"bidPrice": "3", "askPrice": "4", "closeTime": 990},
        put_ticker={"bidPrice": "1", "askPrice": "2", "closeTime": 990},
        futures_book={"bidPrice": "98", "askPrice": "99", "time": 995},
    )

    assert conversion is not None
    assert Decimal(str(conversion["gross_profit_per_unit_USDT"])) == Decimal("1")
    assert conversion["passes_frozen_synchronization_gate"] is True
    assert reversal is not None
    assert Decimal(str(reversal["gross_profit_per_unit_USDT"])) == Decimal("2")
    assert reversal["passes_frozen_synchronization_gate"] is True


def test_v1_missing_option_timestamp_is_unconditionally_unsynchronized() -> None:
    row = _evaluate(
        "conversion_short_perpetual",
        call_ticker={"bidPrice": "1", "askPrice": "2"},
        put_ticker={"bidPrice": "2", "askPrice": "3", "closeTime": 990},
        futures_book={"bidPrice": "101", "askPrice": "102", "time": 995},
    )

    assert row is not None
    assert row["call_close_time_ms"] is None
    assert row["maximum_cross_source_skew_ms"] is None
    assert row["passes_frozen_synchronization_gate"] is False


def test_v1_consumed_result_is_preserved_without_reinterpretation() -> None:
    result = json.loads(V1_RESULT.read_text(encoding="ascii"))

    assert result["result_sha256"] == V1_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == V1_RESULT_HASH
    assert result["population"] == {
        "complete_call_put_strike_count": 705,
        "eligible_option_symbol_count": 1410,
        "evaluated_direction_count": 1410,
        "executable_side_direction_count": 1077,
        "gross_positive_count": 71,
        "synchronized_count": 0,
        "synchronized_gross_positive_count": 0,
    }
    assert len(result["top_optimistic_gross_positive_rows"]) == 20
    assert all(
        not row["passes_frozen_synchronization_gate"]
        for row in result["top_optimistic_gross_positive_rows"]
    )
    assert result["adjudication"] == {
        "accepted_edge": False,
        "current_market_requests": 0,
        "deployment_ready": False,
        "next_action": "stop_without_current_market_requests",
        "profitability_claim": False,
        "status": "no_retained_synchronized_gross_positive_candidate",
    }
    assert (
        result["gates"][
            "fees_quantity_depth_margin_funding_expiry_basis_leg_risk_settlement_and_capital_costs_applied_in_prefilter"
        ]
        is False
    )
    assert result["authority"]["new_public_requests"] == 0
    assert result["authority"]["credentials_used"] is False


def test_v2_repairs_only_timestamp_semantics_and_reconstructs() -> None:
    contract = json.loads(V2_CONTRACT.read_text(encoding="ascii"))
    result = json.loads(V2_RESULT.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == V2_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == V2_CONTRACT_HASH
    assert contract["erratum"]["v1_result_sha256"] == V1_RESULT_HASH
    assert contract["erratum"]["scope_change"] == "synchronization_provenance_only"
    assert contract["gates"]["option_closeTime_used_as_quote_timestamp"] is False
    for source in contract["retained_sources"].values():
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == (source["sha256"])
        )
    for implementation in (
        contract["implementation"],
        contract["implementation"]["dependency"],
    ):
        assert (
            hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
            == implementation["sha256"]
        )
    assert result["result_sha256"] == V2_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == V2_RESULT_HASH
    assert result["population"] == {
        "complete_call_put_strike_count": 705,
        "eligible_option_symbol_count": 1410,
        "evaluated_direction_count": 1410,
        "gross_positive_count": 71,
        "positive_entry_side_direction_count": 1077,
        "synchronized_count": 1077,
        "synchronized_gross_positive_count": 71,
    }
    assert len(result["synchronized_gross_positive_rows"]) == 71
    assert {
        row["maximum_observation_window_skew_ms"]
        for row in result["synchronized_gross_positive_rows"]
    } == {1639}
    assert all(
        row["option_close_time_semantics"] == "transaction_time_diagnostic_only"
        for row in result["synchronized_gross_positive_rows"]
    )


def test_v2_observation_window_gate_does_not_relabel_transaction_time() -> None:
    row = v2._evaluate(
        direction="conversion_short_perpetual",
        option_call=_option("BTC-TEST-100-C", "CALL"),
        option_put=_option("BTC-TEST-100-P", "PUT"),
        call_ticker={"bidPrice": "1", "askPrice": "2"},
        put_ticker={"bidPrice": "2", "askPrice": "3", "closeTime": 100},
        futures_symbol=_future(),
        futures_book={"bidPrice": "101", "askPrice": "102", "time": 2_700},
        options_requested_at_ms=1_000,
        options_completed_at_ms=2_000,
        futures_requested_at_ms=2_500,
        futures_completed_at_ms=3_000,
        maximum_observation_window_skew_ms=2_000,
        maximum_futures_book_age_ms=60_000,
    )

    assert row is not None
    assert row["call_close_time_ms"] is None
    assert row["option_close_time_semantics"] == "transaction_time_diagnostic_only"
    assert row["maximum_observation_window_skew_ms"] == 2_000
    assert row["passes_frozen_synchronization_gate"] is True


def test_complete_retained_stress_rejects_all_rows_before_current_depth() -> None:
    contract = json.loads(STRESS_CONTRACT.read_text(encoding="ascii"))
    result = json.loads(STRESS_RESULT.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == STRESS_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == STRESS_CONTRACT_HASH
    assert result["result_sha256"] == STRESS_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == STRESS_RESULT_HASH
    assert result["population"] == {
        "after_all_retained_stress_positive_count": 0,
        "exact_quantity_grid_count": 71,
        "synchronized_gross_positive_input_count": 71,
    }
    assert len(result["all_rows"]) == 71
    assert result["top_current_depth_eligible_rows"] == []
    assert all(
        Decimal(row["after_all_retained_stress_per_unit_USDT"]) < 0
        and row["option_depth_quantity_verified"] is False
        and row["eligible_for_separate_current_depth_confirmation"] is False
        for row in result["all_rows"]
    )
    assert result["adjudication"]["next_action"] == (
        "stop_without_current_market_requests"
    )
    assert result["authority"]["new_public_requests"] == 0
    assert result["authority"]["credentials_used"] is False


def test_exact_common_quantity_uses_the_lattice_not_only_maximum_minimum() -> None:
    assert stress._common_quantity(
        [Decimal("0.02"), Decimal("0.03"), Decimal("0.01")],
        [Decimal("0.01"), Decimal("0.01"), Decimal("0.01")],
    ) == Decimal("0.06")


def test_terminal_registry_entry_is_unique_and_accepted_count_is_unchanged() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert len(registry["prioritized_hypotheses"]) == 42
    assert len(registry["terminal_do_not_repeat"]) == 37
    assert registry["accepted_edge_count"] == 19
    terminal = [
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_BTC_ETH_SOL_option_perpetual_conversion_reversal_retained_prefilter"
    ]
    assert len(terminal) == 1
    assert terminal[0]["canonical_result_sha256"] == STRESS_RESULT_HASH
    assert "wrongly_used_option_transaction_closeTime" in terminal[0]["reason"]
    assert "zero_survived_exact_quantity" in terminal[0]["reason"]


def test_tools_have_no_network_or_credential_client() -> None:
    sources = [
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (screen, v2, stress)
    ]

    for source in sources:
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            str(node.module).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert imported_roots.isdisjoint({"requests", "httpx", "urllib", "aiohttp"})
        assert "API_KEY" not in source
        assert "SECRET" not in source
