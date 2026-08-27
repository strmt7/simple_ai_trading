"""Run a bounded public broad-crypto spot/perpetual funding-carry preflight."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
SCHEMA_VERSION = "binance-broad-crypto-funding-carry-preflight-v1"
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-broad-crypto-funding-carry-preflight-contract-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json"
)
DEFAULT_RAW_DIR = Path("data/binance-broad-crypto-funding-carry-preflight-v1/raw")
IMPLEMENTATION_PATH = Path("tools/screen_binance_broad_crypto_funding_carry.py")
DAY_MS = Decimal(86_400_000)
YEAR_DAYS = Decimal(365)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text()), name="contract")
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"contract hash differs: declared={declared} actual={actual}")
    contract["contract_sha256"] = declared
    if contract.get("status") != "frozen_before_public_market_outcome_access":
        raise ValueError("contract is not frozen")
    implementation = _mapping(contract["implementation"], name="implementation")
    if implementation.get("sha256") != _sha256(IMPLEMENTATION_PATH.read_bytes()):
        raise ValueError("implementation hash differs from frozen contract")
    return contract, actual


class _Client:
    def __init__(self, raw_dir: Path, *, request_ceiling: int) -> None:
        self.raw_dir = raw_dir
        self.request_ceiling = request_ceiling
        self.receipts: list[dict[str, object]] = []
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )

    def _persist_journal(self, *, complete: bool = False) -> None:
        journal = {
            "schema_version": f"{SCHEMA_VERSION}-request-journal",
            "complete": complete,
            "request_ceiling": self.request_ceiling,
            "request_count": len(self.receipts),
            "responses": self.receipts,
        }
        write_bytes_atomic(
            self.raw_dir / "request-journal.json",
            (json.dumps(journal, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> object:
        if len(self.receipts) >= self.request_ceiling:
            raise ValueError("frozen request ceiling exceeded")
        requested_before_ms = time.time_ns() // 1_000_000
        response = self.session.get(url, params=params, timeout=30)
        received_after_ms = time.time_ns() // 1_000_000
        payload = response.content
        raw_path = self.raw_dir / f"{len(self.receipts) + 1:02d}-{name}.raw"
        write_bytes_atomic(raw_path, payload)
        receipt = {
            "name": name,
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "requested_before_ms": requested_before_ms,
            "received_after_ms": received_after_ms,
            "elapsed_ms": received_after_ms - requested_before_ms,
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": raw_path.as_posix(),
            "x_mbx_used_weight_1m": response.headers.get("x-mbx-used-weight-1m"),
        }
        self.receipts.append(receipt)
        self._persist_journal()
        response.raise_for_status()
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def _nearest_rank(values: Sequence[Decimal], quantile: Decimal) -> Decimal:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty population")
    ordered = sorted(values)
    rank = max(1, math.ceil(float(quantile * Decimal(len(ordered)))))
    return ordered[rank - 1]


def _sign(value: Decimal) -> int:
    return (value > 0) - (value < 0)


def _observations(raw_history: object, *, symbol: str) -> list[dict[str, object]]:
    rows = [
        _mapping(value, name=f"{symbol} funding row")
        for value in _list(raw_history, name=f"{symbol} funding history")
    ]
    if len(rows) < 240 or len(rows) > 1000:
        raise ValueError(f"{symbol} funding history count is outside 240..1000")
    rows.sort(key=lambda row: int(row["fundingTime"]))
    times = [int(row["fundingTime"]) for row in rows]
    if len(set(times)) != len(times):
        raise ValueError(f"{symbol} funding history contains duplicate times")
    observations: list[dict[str, object]] = []
    for index in range(3, len(rows)):
        row = rows[index]
        if row.get("symbol") != symbol:
            raise ValueError(f"{symbol} funding history identity differs")
        prior_mark = Decimal(str(rows[index - 1]["markPrice"]))
        two_back_mark = Decimal(str(rows[index - 2]["markPrice"]))
        three_back_mark = Decimal(str(rows[index - 3]["markPrice"]))
        if min(prior_mark, two_back_mark, three_back_mark) <= 0:
            raise ValueError(f"{symbol} funding history contains nonpositive mark")
        observations.append(
            {
                "funding_time_ms": int(row["fundingTime"]),
                "short_funding_received_bips": _decimal_text(
                    Decimal(str(row["fundingRate"])) * Decimal(10_000)
                ),
                "lagged_mark_return": _decimal_text(
                    prior_mark / two_back_mark - Decimal(1)
                ),
                "previous_lagged_mark_return": _decimal_text(
                    two_back_mark / three_back_mark - Decimal(1)
                ),
            }
        )
    return observations


def _split_roles(
    observations: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    training_end = len(observations) * 3 // 5
    validation_end = len(observations) * 4 // 5
    return {
        "training": observations[:training_end],
        "validation": observations[training_end:validation_end],
        "test": observations[validation_end:],
    }


def _regime_thresholds(training: Sequence[Mapping[str, object]]) -> dict[str, Decimal]:
    absolute_returns = [
        abs(Decimal(str(row["lagged_mark_return"]))) for row in training
    ]
    return {
        "normal_high_boundary": _nearest_rank(absolute_returns, Decimal("0.50")),
        "high_stress_boundary": _nearest_rank(absolute_returns, Decimal("0.80")),
    }


def _regimes(
    row: Mapping[str, object], thresholds: Mapping[str, Decimal]
) -> tuple[str, str, str]:
    lagged_return = Decimal(str(row["lagged_mark_return"]))
    previous = Decimal(str(row["previous_lagged_mark_return"]))
    if lagged_return > Decimal("0.0025"):
        direction = "bullish"
    elif lagged_return < Decimal("-0.0025"):
        direction = "bearish"
    else:
        direction = "sideways"
    absolute_return = abs(lagged_return)
    if absolute_return <= thresholds["normal_high_boundary"]:
        volatility = "low_or_normal_volatility"
    elif absolute_return <= thresholds["high_stress_boundary"]:
        volatility = "high_volatility"
    else:
        volatility = "stress_volatility"
    path = (
        "choppy"
        if _sign(lagged_return) != 0
        and _sign(previous) != 0
        and _sign(lagged_return) != _sign(previous)
        else "directional"
    )
    return direction, volatility, path


def _bootstrap_lower_bound(
    values: Sequence[Decimal],
    *,
    hurdle_bips: Decimal,
    symbol: str,
    role: str,
    repetitions: int,
    block_length: int,
    family_size: int,
) -> Decimal:
    seed_material = f"{SCHEMA_VERSION}:{symbol}:{role}".encode("ascii")
    generator = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8]))
    block = min(block_length, len(values))
    start_count = len(values) - block + 1
    samples: list[Decimal] = []
    for _ in range(repetitions):
        selected: list[Decimal] = []
        while len(selected) < len(values):
            start = generator.randrange(start_count)
            selected.extend(values[start : start + block])
        samples.append(sum(selected[: len(values)], Decimal(0)) - hurdle_bips)
    samples.sort()
    alpha = Decimal("0.05") / Decimal(family_size)
    return samples[max(0, math.ceil(float(alpha * Decimal(repetitions))) - 1)]


def _role_metrics(
    rows: list[dict[str, object]],
    *,
    thresholds: Mapping[str, Decimal],
    symbol: str,
    role: str,
    contract: Mapping[str, object],
    family_size: int,
) -> dict[str, object]:
    gates = _mapping(contract["economic_and_stability_gates"], name="gates")
    times = [int(row["funding_time_ms"]) for row in rows]
    intervals = [right - left for left, right in zip(times, times[1:], strict=False)]
    typical_interval_ms = int(statistics.median(intervals)) if intervals else 28_800_000
    duration_days = Decimal(times[-1] - times[0] + typical_interval_ms) / DAY_MS
    round_trip = Decimal(str(gates["round_trip_execution_stress_bips"]))
    annual = Decimal(str(gates["annual_opportunity_hurdle_bips_per_capital_leg"]))
    legs = Decimal(str(gates["gross_capital_legs"]))
    capital_hurdle = annual * legs * duration_days / YEAR_DAYS
    total_hurdle = round_trip + capital_hurdle
    rates = [Decimal(str(row["short_funding_received_bips"])) for row in rows]
    gross = sum(rates, Decimal(0))
    net = gross - total_hurdle
    per_event_hurdle = total_hurdle / Decimal(len(rows))
    cumulative = Decimal(0)
    peak = Decimal(0)
    maximum_drawdown = Decimal(0)
    for rate in rates:
        cumulative += rate - per_event_hurdle
        peak = max(peak, cumulative)
        maximum_drawdown = max(maximum_drawdown, peak - cumulative)
    weekly: dict[int, Decimal] = defaultdict(Decimal)
    for row in rows:
        week = int(row["funding_time_ms"]) // (7 * 86_400_000)
        weekly[week] += max(
            Decimal(str(row["short_funding_received_bips"])), Decimal(0)
        )
    positive_total = sum(weekly.values(), Decimal(0))
    concentration = (
        max(weekly.values(), default=Decimal(0)) / positive_total
        if positive_total > 0
        else Decimal(1)
    )
    slice_values: dict[str, list[Decimal]] = defaultdict(list)
    for row in rows:
        rate = Decimal(str(row["short_funding_received_bips"]))
        for label in _regimes(row, thresholds):
            slice_values[label].append(rate)
    required_slices = (
        "bullish",
        "bearish",
        "sideways",
        "low_or_normal_volatility",
        "high_volatility",
        "stress_volatility",
        "directional",
        "choppy",
    )
    minimum_slice = int(gates["minimum_observations_per_slice"])
    slices: dict[str, object] = {}
    for label in required_slices:
        values = slice_values[label]
        allocated_hurdle = total_hurdle * Decimal(len(values)) / Decimal(len(rows))
        slice_net = sum(values, Decimal(0)) - allocated_hurdle
        slices[label] = {
            "observation_count": len(values),
            "net_after_allocated_hurdle_bips": _decimal_text(slice_net),
            "passes": len(values) >= minimum_slice and slice_net > 0,
        }
    bootstrap_lower = _bootstrap_lower_bound(
        rates,
        hurdle_bips=total_hurdle,
        symbol=symbol,
        role=role,
        repetitions=int(gates["moving_block_bootstrap_repetitions"]),
        block_length=int(gates["moving_block_length_settlements"]),
        family_size=family_size,
    )
    reasons: list[str] = []
    if len(rows) < int(gates["minimum_observations_per_role"]):
        reasons.append("insufficient_role_population")
    if net <= 0:
        reasons.append("nonpositive_net_after_frozen_hurdles")
    if bootstrap_lower <= 0:
        reasons.append("family_adjusted_bootstrap_lower_bound_not_positive")
    if any(not bool(_mapping(value, name="slice")["passes"]) for value in slices.values()):
        reasons.append("one_or_more_required_slices_failed")
    if maximum_drawdown > Decimal(str(gates["maximum_net_drawdown_bips"])):
        reasons.append("maximum_drawdown_exceeded")
    if concentration > Decimal(str(gates["maximum_positive_week_concentration"])):
        reasons.append("positive_week_concentration_exceeded")
    return {
        "observation_count": len(rows),
        "first_funding_time_ms": times[0],
        "last_funding_time_ms": times[-1],
        "duration_days": _decimal_text(duration_days),
        "gross_funding_bips": _decimal_text(gross),
        "round_trip_execution_stress_bips": _decimal_text(round_trip),
        "capital_opportunity_hurdle_bips": _decimal_text(capital_hurdle),
        "net_after_frozen_hurdles_bips": _decimal_text(net),
        "family_adjusted_bootstrap_net_lower_bound_bips": _decimal_text(
            bootstrap_lower
        ),
        "maximum_net_drawdown_bips": _decimal_text(maximum_drawdown),
        "maximum_positive_week_concentration": _decimal_text(concentration),
        "slices": slices,
        "passes": not reasons,
        "rejection_reasons": reasons,
    }


def _ticker_map(raw: object, *, name: str) -> dict[str, dict[str, object]]:
    return {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name=f"{name} row") for value in _list(raw, name=name)
        )
    }


def _select_universe(
    *,
    spot_exchange: object,
    futures_exchange: object,
    spot_tickers: object,
    futures_tickers: object,
    spot_books: object,
    futures_books: object,
    contract: Mapping[str, object],
) -> tuple[list[dict[str, object]], int]:
    scope = _mapping(contract["population_and_selection"], name="selection")
    spot_rows = [
        _mapping(value, name="spot symbol")
        for value in _list(_mapping(spot_exchange, name="spot exchange")["symbols"], name="spot symbols")
    ]
    futures_rows = [
        _mapping(value, name="future symbol")
        for value in _list(
            _mapping(futures_exchange, name="futures exchange")["symbols"],
            name="future symbols",
        )
    ]
    spot_by_base = {
        str(row["baseAsset"]): row
        for row in spot_rows
        if row.get("status") == "TRADING"
        and row.get("quoteAsset") == "USDT"
        and row.get("isSpotTradingAllowed") is True
    }
    spot_ticker_by_symbol = _ticker_map(spot_tickers, name="spot tickers")
    future_ticker_by_symbol = _ticker_map(futures_tickers, name="future tickers")
    spot_book_by_symbol = _ticker_map(spot_books, name="spot books")
    future_book_by_symbol = _ticker_map(futures_books, name="future books")
    eligible: list[dict[str, object]] = []
    for future in futures_rows:
        if not (
            future.get("status") == "TRADING"
            and future.get("contractType") == "PERPETUAL"
            and future.get("quoteAsset") == "USDT"
            and future.get("marginAsset") == "USDT"
            and future.get("underlyingType") == "COIN"
        ):
            continue
        base = str(future["baseAsset"])
        spot = spot_by_base.get(base)
        if spot is None:
            continue
        spot_symbol = str(spot["symbol"])
        future_symbol = str(future["symbol"])
        required = (
            spot_ticker_by_symbol.get(spot_symbol),
            future_ticker_by_symbol.get(future_symbol),
            spot_book_by_symbol.get(spot_symbol),
            future_book_by_symbol.get(future_symbol),
        )
        if any(value is None for value in required):
            continue
        spot_ticker, future_ticker, spot_book, future_book = required
        assert spot_ticker is not None
        assert future_ticker is not None
        assert spot_book is not None
        assert future_book is not None
        values = [
            Decimal(str(spot_ticker["quoteVolume"])),
            Decimal(str(future_ticker["quoteVolume"])),
            Decimal(str(spot_book["bidPrice"])),
            Decimal(str(spot_book["askPrice"])),
            Decimal(str(future_book["bidPrice"])),
            Decimal(str(future_book["askPrice"])),
        ]
        if min(values) <= 0:
            continue
        minimum_volume = min(values[0], values[1])
        if minimum_volume < Decimal(str(scope["minimum_each_leg_quote_volume_usdt_24h"])):
            continue
        eligible.append(
            {
                "base_asset": base,
                "spot_symbol": spot_symbol,
                "future_symbol": future_symbol,
                "minimum_leg_quote_volume_usdt_24h": minimum_volume,
                "spot_bid": values[2],
                "spot_ask": values[3],
                "future_bid": values[4],
                "future_ask": values[5],
            }
        )
    eligible.sort(
        key=lambda row: (-Decimal(row["minimum_leg_quote_volume_usdt_24h"]), str(row["future_symbol"]))
    )
    return eligible[: int(scope["maximum_selected_symbols"])], len(eligible)


def _public_source_summary(receipts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "request_count": len(receipts),
        "total_response_bytes": sum(int(row["response_bytes"]) for row in receipts),
        "responses": [dict(row) for row in receipts],
    }


def run(*, raw_dir: Path) -> dict[str, object]:
    contract, contract_hash = _contract()
    source = _mapping(contract["source_contract"], name="source contract")
    client = _Client(raw_dir, request_ceiling=int(source["maximum_public_get_requests"]))
    started_ms = time.time_ns() // 1_000_000
    spot_exchange = client.get(
        f"{SPOT_BASE_URL}/api/v3/exchangeInfo", name="spot-exchange-info"
    )
    futures_exchange = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo", name="futures-exchange-info"
    )
    spot_tickers = client.get(
        f"{SPOT_BASE_URL}/api/v3/ticker/24hr",
        name="spot-24hr-tickers",
        params={"type": "MINI"},
    )
    futures_tickers = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/ticker/24hr", name="futures-24hr-tickers"
    )
    spot_books = client.get(
        f"{SPOT_BASE_URL}/api/v3/ticker/bookTicker", name="spot-book-tickers"
    )
    futures_books = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/ticker/bookTicker",
        name="futures-book-tickers",
    )
    selected, eligible_count = _select_universe(
        spot_exchange=spot_exchange,
        futures_exchange=futures_exchange,
        spot_tickers=spot_tickers,
        futures_tickers=futures_tickers,
        spot_books=spot_books,
        futures_books=futures_books,
        contract=contract,
    )
    selection = _mapping(contract["population_and_selection"], name="selection")
    if len(selected) < int(selection["minimum_selected_symbols"]):
        raise ValueError("liquidity-qualified broad universe is too small")
    rows: list[dict[str, object]] = []
    family_size = len(selected)
    for candidate in selected:
        symbol = str(candidate["future_symbol"])
        raw_history = client.get(
            f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
            name=f"funding-{symbol.lower()}",
            params={"symbol": symbol, "limit": 1000},
        )
        observations = _observations(raw_history, symbol=symbol)
        roles = _split_roles(observations)
        thresholds = _regime_thresholds(roles["training"])
        metrics = {
            role: _role_metrics(
                role_rows,
                thresholds=thresholds,
                symbol=symbol,
                role=role,
                contract=contract,
                family_size=family_size,
            )
            for role, role_rows in roles.items()
        }
        all_roles_pass = all(bool(value["passes"]) for value in metrics.values())
        spot_bid = Decimal(candidate["spot_bid"])
        spot_ask = Decimal(candidate["spot_ask"])
        future_bid = Decimal(candidate["future_bid"])
        future_ask = Decimal(candidate["future_ask"])
        rows.append(
            {
                "base_asset": str(candidate["base_asset"]),
                "spot_symbol": str(candidate["spot_symbol"]),
                "future_symbol": symbol,
                "minimum_leg_quote_volume_usdt_24h": _decimal_text(
                    Decimal(candidate["minimum_leg_quote_volume_usdt_24h"])
                ),
                "current_top_book": {
                    "spot_bid": _decimal_text(spot_bid),
                    "spot_ask": _decimal_text(spot_ask),
                    "future_bid": _decimal_text(future_bid),
                    "future_ask": _decimal_text(future_ask),
                    "long_spot_short_future_entry_basis_bips": _decimal_text(
                        Decimal(10_000) * (future_bid / spot_ask - Decimal(1))
                    ),
                    "combined_round_trip_top_spread_bips": _decimal_text(
                        Decimal(10_000)
                        * ((spot_ask / spot_bid - Decimal(1)) + (future_ask / future_bid - Decimal(1)))
                    ),
                    "top_book_has_no_quantity_or_atomicity_claim": True,
                },
                "training_regime_thresholds": {
                    key: _decimal_text(value) for key, value in thresholds.items()
                },
                "roles": metrics,
                "all_roles_pass_funding_only_gate": all_roles_pass,
            }
        )
    passing = [row for row in rows if row["all_roles_pass_funding_only_gate"]]
    finished_ms = time.time_ns() // 1_000_000
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": finished_ms,
        "capture_started_at_ms": started_ms,
        "capture_finished_at_ms": finished_ms,
        "purpose": "test whether a deterministic volume-ranked broad crypto universe contains stable market-direction-neutral spot-perpetual funding carry with enough conservative funding-only headroom to justify basis and account-cost work",
        "contract": {
            "path": CONTRACT_PATH.as_posix(),
            "sha256": contract_hash,
            "status": str(contract["status"]),
        },
        "implementation": {
            "path": IMPLEMENTATION_PATH.as_posix(),
            "sha256": _sha256(IMPLEMENTATION_PATH.read_bytes()),
        },
        "authority": {
            "public_unauthenticated_gets_only": True,
            "credentials_used": False,
            "account_state_accessed": False,
            "orders_or_transfers_submitted": 0,
            "trading_authority": False,
            "one_use_session_scope_exception_exhausted": True,
        },
        "scope": {
            "eligible_exact_spot_perpetual_pair_count": eligible_count,
            "selected_symbol_count": len(selected),
            "selected_by_current_minimum_leg_24h_quote_volume_only": True,
            "current_liquidity_selection_is_not_historical_point_in_time_universe_evidence": True,
        },
        "symbol_results": rows,
        "funding_only_gate_pass_count": len(passing),
        "funding_only_gate_pass_symbols": [str(row["future_symbol"]) for row in passing],
        "sources": _public_source_summary(client.receipts),
        "limitations": [
            "Current liquidity ranking applied to a trailing history is selection-biased and cannot promote an edge.",
            "Funding history contains mark prices but not synchronized executable spot/perpetual basis, depth, fills, or exit basis.",
            "The frozen fee and opportunity-cost values are conservative labeled stresses, not same-account commission or capital evidence.",
            "Top-book prices contain no atomicity, queue, latency, or fill guarantee.",
            "A passing funding-only row can justify only a separate frozen point-in-time-universe basis and execution study; it cannot justify trading.",
        ],
        "verdict": {
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
            "status": (
                "funding_only_candidates_require_separate_unbiased_basis_execution_and_account_cost_study"
                if passing
                else "no_broad_universe_symbol_passed_the_frozen_funding_only_cross_regime_gate"
            ),
        },
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    client._persist_journal(complete=True)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(raw_dir=args.raw_dir)
    write_bytes_atomic(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"output={args.output}")
    print(f"selected={result['scope']['selected_symbol_count']}")
    print(f"passing={result['funding_only_gate_pass_count']}")
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
