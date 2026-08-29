"""Repair the retained Binance option/perpetual synchronization provenance."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping

from tools import screen_binance_options_perpetual_conversion_prefilter as v1


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-contract-v2.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-v2-2026-08-29.json"
)


def _observation_window_skew_ms(
    *,
    options_requested_at_ms: int,
    options_completed_at_ms: int,
    futures_requested_at_ms: int,
    futures_completed_at_ms: int,
) -> int:
    """Return the largest possible skew between two HTTP observation windows."""
    return max(
        abs(options_requested_at_ms - futures_completed_at_ms),
        abs(options_completed_at_ms - futures_requested_at_ms),
    )


def _evaluate(
    *,
    direction: str,
    option_call: Mapping[str, object],
    option_put: Mapping[str, object],
    call_ticker: Mapping[str, object],
    put_ticker: Mapping[str, object],
    futures_symbol: Mapping[str, object],
    futures_book: Mapping[str, object],
    options_requested_at_ms: int,
    options_completed_at_ms: int,
    futures_requested_at_ms: int,
    futures_completed_at_ms: int,
    maximum_observation_window_skew_ms: int,
    maximum_futures_book_age_ms: int,
) -> dict[str, object] | None:
    row = v1._evaluate(
        direction=direction,
        option_call=option_call,
        option_put=option_put,
        call_ticker=call_ticker,
        put_ticker=put_ticker,
        futures_symbol=futures_symbol,
        futures_book=futures_book,
        options_completed_at_ms=options_completed_at_ms,
        futures_completed_at_ms=futures_completed_at_ms,
        maximum_option_quote_age_ms=2**63 - 1,
        maximum_cross_source_skew_ms=2**63 - 1,
    )
    if row is None:
        return None
    observation_skew_ms = _observation_window_skew_ms(
        options_requested_at_ms=options_requested_at_ms,
        options_completed_at_ms=options_completed_at_ms,
        futures_requested_at_ms=futures_requested_at_ms,
        futures_completed_at_ms=futures_completed_at_ms,
    )
    futures_age_ms = int(row["futures_book_age_ms"])
    row.update(
        {
            "option_close_time_semantics": "transaction_time_diagnostic_only",
            "option_ticker_observation_window_ms": [
                options_requested_at_ms,
                options_completed_at_ms,
            ],
            "futures_book_observation_window_ms": [
                futures_requested_at_ms,
                futures_completed_at_ms,
            ],
            "maximum_observation_window_skew_ms": observation_skew_ms,
            "passes_frozen_synchronization_gate": (
                observation_skew_ms <= maximum_observation_window_skew_ms
                and 0 <= futures_age_ms <= maximum_futures_book_age_ms
            ),
        }
    )
    return row


def main() -> int:
    if RESULT_PATH.exists():
        raise FileExistsError("one-use retained v2 prefilter result already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if v1._canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise ValueError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if v1._sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    dependency = ROOT / contract["implementation"]["dependency"]["path"]
    if (
        v1._sha256(dependency.read_bytes())
        != contract["implementation"]["dependency"]["sha256"]
    ):
        raise ValueError("dependency hash mismatch")

    sources = contract["retained_sources"]
    option_exchange = v1._load_source(sources["options_exchange_info"])
    option_ticker_rows = v1._load_source(sources["options_all_tickers"])
    futures_exchange = v1._load_source(sources["futures_exchange_info"])
    futures_book_rows = v1._load_source(sources["futures_all_book_tickers"])
    option_tickers = {str(row["symbol"]): row for row in option_ticker_rows}
    futures_books = {str(row["symbol"]): row for row in futures_book_rows}

    allowed = set(contract["population"]["allowed_underlyings"])
    option_symbols = [
        row
        for row in option_exchange["optionSymbols"]
        if row.get("status") == "TRADING"
        and row.get("contractType") == "CRYPTO_OPTIONS"
        and row.get("underlyingType") == "CRYPTO"
        and row.get("underlying") in allowed
        and v1._decimal(row.get("unit")) == Decimal("1")
    ]
    futures_symbols = {
        str(row["symbol"]): row
        for row in futures_exchange["symbols"]
        if row.get("status") == "TRADING"
        and row.get("contractType") == "PERPETUAL"
        and row.get("symbol") in allowed
    }
    pairs: dict[tuple[str, int, Decimal], dict[str, Mapping[str, object]]] = {}
    for symbol in option_symbols:
        key = (
            str(symbol["underlying"]),
            int(symbol["expiryDate"]),
            v1._decimal(symbol["strikePrice"]),
        )
        pairs.setdefault(key, {})[str(symbol["side"])] = symbol

    rows: list[dict[str, object]] = []
    evaluated_direction_count = 0
    for (underlying, _expiry, _strike), legs in sorted(pairs.items()):
        if set(legs) != {"CALL", "PUT"}:
            continue
        if underlying not in futures_symbols or underlying not in futures_books:
            continue
        call_symbol = legs["CALL"]
        put_symbol = legs["PUT"]
        call_ticker = option_tickers.get(str(call_symbol["symbol"]))
        put_ticker = option_tickers.get(str(put_symbol["symbol"]))
        if call_ticker is None or put_ticker is None:
            continue
        for direction in (
            "conversion_short_perpetual",
            "reversal_long_perpetual",
        ):
            evaluated_direction_count += 1
            row = _evaluate(
                direction=direction,
                option_call=call_symbol,
                option_put=put_symbol,
                call_ticker=call_ticker,
                put_ticker=put_ticker,
                futures_symbol=futures_symbols[underlying],
                futures_book=futures_books[underlying],
                options_requested_at_ms=sources["options_all_tickers"][
                    "requested_at_ms"
                ],
                options_completed_at_ms=sources["options_all_tickers"][
                    "completed_at_ms"
                ],
                futures_requested_at_ms=sources["futures_all_book_tickers"][
                    "requested_at_ms"
                ],
                futures_completed_at_ms=sources["futures_all_book_tickers"][
                    "completed_at_ms"
                ],
                maximum_observation_window_skew_ms=contract["gates"][
                    "maximum_observation_window_skew_ms"
                ],
                maximum_futures_book_age_ms=contract["gates"][
                    "maximum_futures_book_age_ms"
                ],
            )
            if row is not None:
                rows.append(row)

    gross_positive = [
        row
        for row in rows
        if v1._decimal(row["gross_profit_per_unit_USDT"]) > Decimal("0")
    ]
    synchronized = [row for row in rows if row["passes_frozen_synchronization_gate"]]
    synchronized_positive = [
        row for row in gross_positive if row["passes_frozen_synchronization_gate"]
    ]
    retained = sorted(
        synchronized_positive,
        key=lambda row: v1._decimal(row["gross_profit_per_unit_USDT"]),
        reverse=True,
    )[: contract["reporting"]["maximum_retained_rows"]]
    result: dict[str, object] = {
        "schema_version": "binance-options-perpetual-conversion-retained-prefilter-v2",
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "erratum": contract["erratum"],
        "retained_sources": sources,
        "population": {
            "eligible_option_symbol_count": len(option_symbols),
            "complete_call_put_strike_count": sum(
                1 for legs in pairs.values() if set(legs) == {"CALL", "PUT"}
            ),
            "evaluated_direction_count": evaluated_direction_count,
            "positive_entry_side_direction_count": len(rows),
            "gross_positive_count": len(gross_positive),
            "synchronized_count": len(synchronized),
            "synchronized_gross_positive_count": len(synchronized_positive),
        },
        "identity": contract["identity"],
        "gates": contract["gates"],
        "synchronized_gross_positive_rows": retained,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "current_market_requests": 0,
            "status": (
                "retained_synchronized_gross_positive_candidate_only"
                if synchronized_positive
                else "no_retained_synchronized_gross_positive_candidate"
            ),
            "next_action": (
                "freeze_complete_population_retained_funding_expiry_basis_fee_margin_leg_risk_and_capital_stress"
                if synchronized_positive
                else "stop_without_current_market_requests"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = v1._canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        v1._canonical_json(
            {
                "gross_positive_count": len(gross_positive),
                "maximum_observation_window_skew_ms": _observation_window_skew_ms(
                    options_requested_at_ms=sources["options_all_tickers"][
                        "requested_at_ms"
                    ],
                    options_completed_at_ms=sources["options_all_tickers"][
                        "completed_at_ms"
                    ],
                    futures_requested_at_ms=sources["futures_all_book_tickers"][
                        "requested_at_ms"
                    ],
                    futures_completed_at_ms=sources["futures_all_book_tickers"][
                        "completed_at_ms"
                    ],
                ),
                "synchronized_gross_positive_count": len(synchronized_positive),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
