import hashlib
import json
from decimal import Decimal
from pathlib import Path

from tools.screen_polymarket_binance_tradfi_perps_funding_history import (
    EIGHT_HOURS_MS,
    HOUR_MS,
    FundingObservation,
    align_funding,
    evaluate_history,
    parse_binance_funding,
    parse_polymarket_funding,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load(name: str) -> dict[str, object]:
    return json.loads((ACTION / name).read_text(encoding="ascii"))


def test_parse_polymarket_funding_requires_complete_page() -> None:
    rows = parse_polymarket_funding(
        {
            "data": [
                {"funding_rate": "0.0002", "timestamp": 2 * HOUR_MS},
                {"funding_rate": "0.0001", "timestamp": HOUR_MS},
            ],
            "more": False,
        }
    )
    assert rows == [
        (HOUR_MS, Decimal("0.0001")),
        (2 * HOUR_MS, Decimal("0.0002")),
    ]


def test_parse_binance_funding_sums_regular_and_special_rows() -> None:
    rows, rate_types = parse_binance_funding(
        [
            {
                "fundingRate": "0.0003",
                "fundingTime": EIGHT_HOURS_MS,
                "rateType": "Regular",
            },
            {
                "fundingRate": "0.0002",
                "fundingTime": EIGHT_HOURS_MS,
                "rateType": "Special",
            },
        ]
    )
    assert rows == [(EIGHT_HOURS_MS, Decimal("0.0005"))]
    assert rate_types == {"Regular": 1, "Special": 1}


def test_align_funding_uses_fixed_orientation_and_one_hour_price_path() -> None:
    settlement = 16 * HOUR_MS
    polymarket = [
        (settlement - offset * HOUR_MS, Decimal("0.0001")) for offset in range(8)
    ]
    binance = [(settlement + 17, Decimal("0.0002"))]
    klines = [
        (
            settlement - EIGHT_HOURS_MS + offset * HOUR_MS,
            Decimal("100") + Decimal(offset),
            Decimal("101") + Decimal(offset),
        )
        for offset in range(8)
    ]

    aligned, diagnostics = align_funding(
        polymarket,
        binance,
        klines,
        orientation="short_polymarket_long_binance",
        maximum_timestamp_phase_skew_ms=60_000,
    )

    assert len(aligned) == 1
    assert aligned[0].timestamp_ms == settlement
    assert aligned[0].carry == Decimal("0.0006")
    assert aligned[0].return_8h == Decimal("108") / Decimal("100") - 1
    assert diagnostics["maximum_observed_timestamp_phase_skew_ms"] == 17


def _regime_rows(carry: str) -> tuple[FundingObservation, ...]:
    returns = (
        "-0.02",
        "-0.01",
        "0",
        "0.001",
        "0.01",
        "0.02",
        "-0.02",
        "-0.01",
        "0",
        "0.001",
        "0.01",
        "0.02",
    )
    return tuple(
        FundingObservation(
            timestamp_ms=index * EIGHT_HOURS_MS,
            carry=Decimal(carry),
            return_8h=Decimal(return_value),
        )
        for index, return_value in enumerate(returns, start=1)
    )


def test_evaluate_history_requires_each_role_and_each_regime() -> None:
    result = evaluate_history(
        _regime_rows("0.003"),
        execution_hurdle_bips=Decimal("20"),
        annual_opportunity_hurdle_bips_per_leg=Decimal("500"),
        minimum_aligned_rows=12,
        minimum_regime_rows=2,
    )
    assert result["economic_role_pass"] is True
    assert result["cross_regime_pass"] is True
    assert result["historical_persistence_candidate"] is True
    assert [
        result["roles"][role]["count"] for role in ("training", "validation", "test")
    ] == [6, 3, 3]


def test_evaluate_history_rejects_repeated_small_gross_carry() -> None:
    result = evaluate_history(
        _regime_rows("0.0001"),
        execution_hurdle_bips=Decimal("20"),
        annual_opportunity_hurdle_bips_per_leg=Decimal("500"),
        minimum_aligned_rows=12,
        minimum_regime_rows=2,
    )
    assert result["economic_role_pass"] is False
    assert result["historical_persistence_candidate"] is False


def test_consumed_history_and_offline_shortfall_adjudication_are_source_bound() -> None:
    history_contract = _load(
        "polymarket-binance-tradfi-perps-funding-history-contract-v1.json"
    )
    history = _load(
        "polymarket-binance-tradfi-perps-funding-history-v1-2026-08-29.json"
    )
    shortfall_contract = _load(
        "polymarket-binance-tradfi-perps-funding-history-shortfall-adjudication-contract-v1.json"
    )
    shortfall = _load(
        "polymarket-binance-tradfi-perps-funding-history-shortfall-adjudication-v1-2026-08-29.json"
    )

    assert history_contract["result_sha256"] == (
        "6059d02dbda9e6a314f3fc79793f8d21ba3c3040803da033bd81ead3d2b872de"
    )
    assert history["result_sha256"] == (
        "ad896a698edd65b42b039f84d1b037cf67302c7b2bb7ae59e9008f45328939bb"
    )
    assert shortfall_contract["result_sha256"] == (
        "f06981e97236a0c577a626b882cc72f69383752112f8c0ffab8e04e1b7b9e480"
    )
    assert shortfall["result_sha256"] == (
        "5e67277ad30b9f0164a3987804162ed2d1cdabb820e7e258d6a0b79748cf7d06"
    )
    for payload in (history_contract, history, shortfall_contract, shortfall):
        assert _canonical_hash(payload) == payload["result_sha256"]

    assert (
        hashlib.sha256(
            (
                ROOT / "tools/screen_polymarket_binance_tradfi_perps_funding_history.py"
            ).read_bytes()
        ).hexdigest()
        == history_contract["implementation_sha256"]
    )
    assert (
        hashlib.sha256(
            (
                ROOT
                / "tools/adjudicate_polymarket_binance_tradfi_perps_funding_history_shortfall.py"
            ).read_bytes()
        ).hexdigest()
        == shortfall_contract["implementation_sha256"]
    )
    assert history["authority"]["http_get_requests"] == 15
    assert history["historical_persistence_candidates"] == []
    assert history["role_only_survivors"] == []
    assert history["adjudication"]["accepted_edge"] is False
    assert history["symbol_results"]["SKHYNIX"]["binance_rate_type_counts"] == {
        "Regular": 26,
        "Special": 1,
    }
    assert {
        symbol: value["result"]["aligned_count"]
        for symbol, value in history["symbol_results"].items()
    } == {"SKHYNIX": 26, "CRWV": 11, "ARM": 11, "HOOD": 11, "MSTR": 11}

    assert shortfall["authority"]["network_requests"] == 0
    assert shortfall["adjudication"] == {
        "accepted_edge": False,
        "candidate_for_books": False,
        "deployment_ready": False,
        "all_symbols_economically_rejected_on_actual_retained_rows": True,
        "status": "retained_actual_rows_reject_every_symbol",
    }
    arm = shortfall["diagnostics"]["ARM"]["retained_diagnostic"]
    assert Decimal(arm["total_net_after_execution_and_capital_bips"]) == Decimal(
        "-24.62966210045662100456621005"
    )
    assert all(value["passes"] is False for value in arm["roles"].values())


def test_current_cxmt_trigger_reopens_only_a_future_history_test() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert len(registry["prioritized_hypotheses"]) == 65
    hypothesis = next(
        value
        for value in registry["prioritized_hypotheses"]
        if value["mechanism"]
        == "polymarket_binance_TradFi_perpetual_fixed_orientation_cross_venue_funding_spread"
    )
    assert hypothesis["priority_rank"] == 43
    assert hypothesis["market_direction_forecast_required"] is False
    assert "CXMT_post_change_history_test" in hypothesis["current_status"]
    assert "not_before_2026_09_06T08_10_00Z" in hypothesis["retry_trigger"]
    assert hypothesis["canonical_artifacts"][0] == {
        "path": (
            "docs/model-research/action-value/"
            "binance-cxmt-four-hour-funding-trigger-v1-2026-09-04.json"
        ),
        "result_sha256": (
            "99d0fded6f7378d0b398b33cd5221f515704cd8c31c3b360e9773eac784f6402"
        ),
    }

    trigger_path = (
        ACTION / "binance-cxmt-four-hour-funding-trigger-v1-2026-09-04.json"
    )
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert _canonical_hash(trigger) == trigger["result_sha256"]
    assert trigger["exact_cross_venue_match"]["exact_match_passed"] is True
    assert trigger["official_change"]["daily_cap_increased"] is False
    assert trigger["economic_interpretation"]["history_requests_made_now"] == 0
    assert trigger["adjudication"]["candidate_for_post_change_history"] is True
    assert trigger["adjudication"]["accepted_edge"] is False

    source = trigger["official_change"]
    raw = ROOT / source["raw_path"]
    journal = ROOT / source["journal_path"]
    assert len(raw.read_bytes()) == source["raw_bytes"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == source["raw_sha256"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == source["journal_sha256"]

    terminal = {value["family"]: value for value in registry["terminal_do_not_repeat"]}
    assert terminal[
        "polymarket_binance_TradFi_perpetual_fixed_orientation_cross_venue_funding_spread_top_five_snapshot"
    ]["canonical_result_sha256"] == (
        "5e67277ad30b9f0164a3987804162ed2d1cdabb820e7e258d6a0b79748cf7d06"
    )
