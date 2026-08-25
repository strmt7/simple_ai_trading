from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
    confirm_option_candidate,
    discover_option_parity,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-option-parity-snapshot-v1-2026-08-25.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="ascii"))


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


def test_option_parity_artifact_reconstructs_hashes_and_implementation() -> None:
    report = _report()
    claimed_hash = report.pop("result_sha256")
    canonical = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed_hash

    implementation = report["source_contract"]["implementation"]
    assert (
        _sha256(ROOT / "tools" / implementation["tool_path"])
        == implementation["tool_sha256"]
    )
    assert (
        _sha256(ROOT / "src" / "simple_ai_trading" / implementation["module_path"])
        == implementation["module_sha256"]
    )


def test_option_parity_artifact_reconstructs_every_discovery_count() -> None:
    report = _report()
    contracts = tuple(_contract(row) for row in report["contracts"])
    screen = discover_option_parity(contracts)
    discovery = report["discovery"]

    assert discovery["scoped_contract_count"] == len(contracts)
    assert discovery["chain_count"] == len(
        {
            (contract.underlying, contract.expiry_date_ms, contract.side)
            for contract in contracts
        }
    )
    assert discovery["evaluated_vertical_count"] == screen.evaluated_vertical_count
    assert discovery["executable_vertical_count"] == screen.executable_vertical_count
    assert discovery["evaluated_convexity_count"] == screen.evaluated_convexity_count
    assert discovery["executable_convexity_count"] == screen.executable_convexity_count
    assert discovery["ticker_gross_positive_candidate_count"] == len(
        screen.gross_positive_candidates
    )
    recorded = discovery["ticker_gross_positive_candidates"]
    for candidate, row in zip(screen.gross_positive_candidates, recorded, strict=True):
        assert row["mechanism"] == candidate.mechanism
        assert row["symbols"] == list(candidate.symbols)
        assert row["roles"] == list(candidate.roles)
        assert row["integer_weights"] == list(candidate.integer_weights)
        assert row["minimum_quantities"] == [
            format(value, "f") for value in candidate.quantities
        ]
        assert row["ticker_gross_credit_quote"] == format(
            candidate.gross_credit_quote, "f"
        )


def test_option_parity_artifact_reconstructs_depth_rejection() -> None:
    report = _report()
    contracts = tuple(_contract(row) for row in report["contracts"])
    candidates = {
        candidate.symbols: candidate
        for candidate in discover_option_parity(contracts).gross_positive_candidates
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
            candidate = candidates[tuple(row["symbols"])]
            confirmation = confirm_option_candidate(candidate, books)
            assert row["executable"] is confirmation.executable
            assert row["gross_credit_quote"] == (
                None
                if confirmation.gross_credit_quote is None
                else format(confirmation.gross_credit_quote, "f")
            )
            assert row["book_event_times_ms"] == list(confirmation.book_event_times_ms)

    assert report["verdict"] == {
        "accepted_edge": False,
        "account_margin_and_short_inventory_verified": False,
        "atomic_multi_leg_execution_verified": False,
        "exact_account_commission_verified": False,
        "initial_fresh_gross_positive_candidate_count": 0,
        "persistent_gross_positive_candidate_count": 0,
        "status": "rejected_no_fresh_gross_positive_depth_candidate",
        "trading_authority": False,
    }
    assert report["safety"] == {
        "credentials_used": False,
        "orders_placed": False,
        "public_books_prove_fills": False,
        "public_market_data_only": True,
        "ticker_is_discovery_not_execution_evidence": True,
    }
