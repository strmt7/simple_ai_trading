from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from simple_ai_trading.option_box_parity import (
    confirm_option_box,
    discover_option_boxes,
)
from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
SOURCE = RESEARCH / "binance-option-parity-snapshot-v1-2026-08-25.json"
ARTIFACT = RESEARCH / "binance-option-box-parity-snapshot-v1-2026-08-25.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _contract(row: dict[str, object]) -> OptionContractQuote:
    return OptionContractQuote(
        symbol=str(row["symbol"]),
        underlying=str(row["underlying"]),
        expiry_date_ms=int(row["expiry_date_ms"]),
        side=str(row["side"]),
        strike=Decimal(str(row["strike"])),
        unit=Decimal(str(row["unit"])),
        minimum_quantity=Decimal(str(row["minimum_quantity"])),
        maximum_quantity=Decimal(str(row["maximum_quantity"])),
        step_size=Decimal(str(row["step_size"])),
        bid_price=(
            None if row["bid_price"] is None else Decimal(str(row["bid_price"]))
        ),
        ask_price=(
            None if row["ask_price"] is None else Decimal(str(row["ask_price"]))
        ),
    )


def test_box_artifact_reconstructs_hashes_and_source_lineage() -> None:
    report = _report(ARTIFACT)
    claimed = report.pop("result_sha256")
    canonical = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed

    source_contract = report["source_contract"]
    assert source_contract["source_snapshot_file_sha256"] == _sha256(SOURCE)
    assert (
        source_contract["source_snapshot_result_sha256"]
        == _report(SOURCE)["result_sha256"]
    )
    implementation = source_contract["implementation"]
    assert (
        _sha256(ROOT / "tools" / implementation["tool_path"])
        == implementation["tool_sha256"]
    )
    assert (
        _sha256(ROOT / "src" / "simple_ai_trading" / implementation["module_path"])
        == implementation["module_sha256"]
    )


def test_box_artifact_reconstructs_discovery_and_candidates() -> None:
    source = _report(SOURCE)
    report = _report(ARTIFACT)
    contracts = tuple(_contract(row) for row in source["contracts"])
    screen = discover_option_boxes(
        contracts,
        as_of_ms=report["source_contract"]["source_snapshot_as_of_ms"],
    )
    discovery = report["discovery"]

    assert discovery["chain_count"] == screen.chain_count
    assert discovery["evaluated_strike_pair_count"] == (
        screen.evaluated_strike_pair_count
    )
    assert discovery["executable_long_box_count"] == screen.executable_long_box_count
    assert discovery["executable_short_box_count"] == (
        screen.executable_short_box_count
    )
    assert discovery["nominal_positive_long_box_count"] == len(
        screen.nominal_positive_long_boxes
    )
    assert discovery["strict_positive_short_box_count"] == len(
        screen.strict_positive_short_boxes
    )
    assert discovery["ticker_candidate_count"] == len(screen.candidates)
    for candidate, row in zip(
        screen.candidates, discovery["ticker_candidates"], strict=True
    ):
        assert row["kind"] == candidate.kind
        assert row["symbols"] == list(candidate.symbols)
        assert row["roles"] == list(candidate.roles)
        assert row["minimum_quantity"] == format(candidate.quantity, "f")
        assert row["fixed_expiry_cashflow_quote"] == format(
            candidate.fixed_expiry_cashflow_quote, "f"
        )
        assert row["ticker_gross_expiry_profit_quote"] == format(
            candidate.gross_expiry_profit_quote, "f"
        )


def test_box_artifact_reconstructs_depth_rejection() -> None:
    source = _report(SOURCE)
    report = _report(ARTIFACT)
    contracts = tuple(_contract(row) for row in source["contracts"])
    candidates = {
        candidate.symbols: candidate
        for candidate in discover_option_boxes(
            contracts,
            as_of_ms=report["source_contract"]["source_snapshot_as_of_ms"],
        ).candidates
    }
    for sweep in report["depth_confirmation_sweeps"]:
        books = {
            row["symbol"]: OptionDepthQuote(
                symbol=row["symbol"],
                event_time_ms=row["event_time_ms"],
                bids=tuple(
                    OptionBookLevel(Decimal(price), Decimal(quantity))
                    for price, quantity in row["bids"]
                ),
                asks=tuple(
                    OptionBookLevel(Decimal(price), Decimal(quantity))
                    for price, quantity in row["asks"]
                ),
            )
            for row in sweep["books"]
        }
        for row in sweep["candidate_results"]:
            confirmation = confirm_option_box(candidates[tuple(row["symbols"])], books)
            assert row["executable"] is confirmation.executable
            assert row["initial_credit_quote"] == (
                None
                if confirmation.initial_credit_quote is None
                else format(confirmation.initial_credit_quote, "f")
            )
            assert row["gross_expiry_profit_quote"] == (
                None
                if confirmation.gross_expiry_profit_quote is None
                else format(confirmation.gross_expiry_profit_quote, "f")
            )

    assert report["verdict"] == {
        "accepted_edge": False,
        "account_margin_and_short_inventory_verified": False,
        "atomic_multi_leg_execution_verified": False,
        "exact_account_commission_verified": False,
        "financing_and_opportunity_cost_verified": False,
        "initial_fresh_executable_positive_count": 0,
        "persistent_positive_count": 0,
        "status": "rejected_no_fresh_executable_depth_positive_box",
        "trading_authority": False,
    }
