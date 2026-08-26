"""Run the frozen one-shot public Binance bStock funding-carry screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
BSTOCK_LIST_URL = (
    "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/"
    "market/token/rwa/stock/detail/list/ai"
)
SCHEMA_VERSION = "binance-bstock-funding-full-universe-v1"
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-bstock-funding-full-universe-contract-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-bstock-funding-full-universe-v1-2026-08-26.json"
)
DEFAULT_RAW_DIR = Path("data/binance-bstock-funding-full-universe-v1/raw")
IMPLEMENTATION_PATH = Path("tools/screen_binance_bstock_funding_full_universe.py")
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
    if contract.get("status") != (
        "frozen_before_unobserved_full_universe_funding_outcome_access"
    ):
        raise ValueError("contract is not frozen")
    return contract, actual


class _Client:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        final_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                payload = response.content
                raw_path = self.raw_dir / f"{name}.raw"
                write_bytes_atomic(raw_path, payload)
                receipt = {
                    "method": "GET",
                    "url": response.url,
                    "status_code": response.status_code,
                    "response_bytes": len(payload),
                    "response_sha256": _sha256(payload),
                    "raw_path": str(raw_path.as_posix()),
                    "attempt_count": attempt + 1,
                }
                try:
                    return response.json(), receipt
                except requests.JSONDecodeError as exc:
                    raise ValueError(f"{name} did not return JSON") from exc
            except (requests.RequestException, ValueError) as exc:
                final_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        raise ValueError(f"{name} failed after three attempts") from final_error


def _nearest_rank(values: Sequence[Decimal], quantile: Decimal) -> Decimal:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty population")
    ordered = sorted(values)
    rank = max(1, math.ceil(float(quantile * Decimal(len(ordered)))))
    return ordered[rank - 1]


def _sign(value: Decimal) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _observations(raw_history: object, *, symbol: str) -> list[dict[str, object]]:
    rows = [
        _mapping(value, name=f"{symbol} funding row")
        for value in _list(raw_history, name=f"{symbol} funding history")
    ]
    if not rows or len(rows) >= 1000:
        raise ValueError(f"{symbol} funding history is empty or limit-saturated")
    rows.sort(key=lambda row: int(row["fundingTime"]))
    times = [int(row["fundingTime"]) for row in rows]
    if len(set(times)) != len(times):
        raise ValueError(f"{symbol} funding history contains duplicate times")
    for row in rows:
        if row.get("symbol") != symbol:
            raise ValueError(f"{symbol} funding history identity differs")
        if Decimal(str(row["markPrice"])) <= 0:
            raise ValueError(f"{symbol} funding history contains nonpositive mark")
    observations: list[dict[str, object]] = []
    for index in range(3, len(rows)):
        current = rows[index]
        prior = rows[index - 1]
        two_back = rows[index - 2]
        three_back = rows[index - 3]
        prior_mark = Decimal(str(prior["markPrice"]))
        two_back_mark = Decimal(str(two_back["markPrice"]))
        three_back_mark = Decimal(str(three_back["markPrice"]))
        lagged_return = prior_mark / two_back_mark - Decimal(1)
        previous_lagged_return = two_back_mark / three_back_mark - Decimal(1)
        observations.append(
            {
                "funding_time_ms": int(current["fundingTime"]),
                "short_funding_received_bips": (
                    Decimal(str(current["fundingRate"])) * Decimal(10_000)
                ),
                "lagged_mark_return": lagged_return,
                "previous_lagged_mark_return": previous_lagged_return,
            }
        )
    return observations


def _split_roles(
    observations: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    training_end = len(observations) // 2
    validation_end = training_end + (len(observations) - training_end) // 2
    return {
        "training": observations[:training_end],
        "validation": observations[training_end:validation_end],
        "test": observations[validation_end:],
    }


def _regime_thresholds(training: list[dict[str, object]]) -> dict[str, Decimal]:
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
    rates: Sequence[Decimal],
    *,
    hurdle_bips: Decimal,
    symbol: str,
    role: str,
    repetitions: int,
    block_length: int,
    family_size: int,
) -> Decimal:
    if not rates:
        raise ValueError("bootstrap population is empty")
    seed_material = f"{SCHEMA_VERSION}:{symbol}:{role}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = random.Random(seed)
    length = len(rates)
    block = min(block_length, length)
    start_count = length - block + 1
    samples: list[Decimal] = []
    for _ in range(repetitions):
        selected: list[Decimal] = []
        while len(selected) < length:
            start = generator.randrange(start_count)
            selected.extend(rates[start : start + block])
        samples.append(sum(selected[:length], Decimal(0)) - hurdle_bips)
    samples.sort()
    alpha = Decimal("0.05") / Decimal(family_size)
    rank = max(0, math.ceil(float(alpha * Decimal(repetitions))) - 1)
    return samples[rank]


def _role_metrics(
    rows: list[dict[str, object]],
    *,
    thresholds: Mapping[str, Decimal],
    contract: Mapping[str, object],
    symbol: str,
    role: str,
    family_size: int,
) -> dict[str, object]:
    causal = _mapping(contract["causal_roles"], name="causal roles")
    costs = _mapping(contract["cost_and_capital_hurdles"], name="cost hurdles")
    regimes = _mapping(contract["regimes"], name="regimes")
    stats = _mapping(contract["statistical_and_risk_gates"], name="risk gates")
    minimum_role = int(causal["minimum_observations_per_role"])
    minimum_slice = int(regimes["minimum_observations_per_slice"])
    if not rows:
        return {
            "observation_count": 0,
            "passes": False,
            "rejection_reasons": ["empty_role"],
        }
    times = [int(row["funding_time_ms"]) for row in rows]
    intervals = [right - left for left, right in zip(times, times[1:], strict=False)]
    typical_interval_ms = (
        sorted(intervals)[len(intervals) // 2] if intervals else 28_800_000
    )
    duration_days = Decimal(times[-1] - times[0] + typical_interval_ms) / DAY_MS
    round_trip = Decimal(str(costs["round_trip_execution_stress_bips"]))
    annual = Decimal(str(costs["annual_opportunity_hurdle_bips_per_capital_leg"]))
    legs = Decimal(str(costs["gross_capital_legs"]))
    capital_hurdle = annual * legs * duration_days / YEAR_DAYS
    total_hurdle = round_trip + capital_hurdle
    rates = [Decimal(str(row["short_funding_received_bips"])) for row in rows]
    gross = sum(rates, Decimal(0))
    net = gross - total_hurdle
    per_event_hurdle = total_hurdle / Decimal(len(rows))
    event_nets = [rate - per_event_hurdle for rate in rates]
    cumulative = Decimal(0)
    peak = Decimal(0)
    maximum_drawdown = Decimal(0)
    for value in event_nets:
        cumulative += value
        peak = max(peak, cumulative)
        maximum_drawdown = max(maximum_drawdown, peak - cumulative)
    tail_count = max(1, math.ceil(len(event_nets) * 0.025))
    expected_shortfall = sum(sorted(event_nets)[:tail_count], Decimal(0)) / Decimal(
        tail_count
    )
    weekly: dict[int, Decimal] = defaultdict(Decimal)
    for row in rows:
        week = int(row["funding_time_ms"]) // (7 * 86_400_000)
        weekly[week] += max(
            Decimal(str(row["short_funding_received_bips"])), Decimal(0)
        )
    positive_total = sum(weekly.values(), Decimal(0))
    positive_week_concentration = (
        max(weekly.values(), default=Decimal(0)) / positive_total
        if positive_total > 0
        else Decimal(1)
    )
    slice_values: dict[str, list[Decimal]] = defaultdict(list)
    for row in rows:
        rate = Decimal(str(row["short_funding_received_bips"]))
        for label in _regimes(row, thresholds):
            slice_values[label].append(rate)
    required_slices = [
        "bullish",
        "bearish",
        "sideways",
        "low_or_normal_volatility",
        "high_volatility",
        "stress_volatility",
        "directional",
        "choppy",
    ]
    slice_rows: dict[str, object] = {}
    for label in required_slices:
        values = slice_values[label]
        allocated_hurdle = total_hurdle * Decimal(len(values)) / Decimal(len(rows))
        slice_net = sum(values, Decimal(0)) - allocated_hurdle
        slice_rows[label] = {
            "observation_count": len(values),
            "gross_funding_bips": _decimal_text(sum(values, Decimal(0))),
            "allocated_hurdle_bips": _decimal_text(allocated_hurdle),
            "net_after_allocated_hurdle_bips": _decimal_text(slice_net),
            "passes": len(values) >= minimum_slice and slice_net >= 0,
        }
    bootstrap_lower = _bootstrap_lower_bound(
        rates,
        hurdle_bips=total_hurdle,
        symbol=symbol,
        role=role,
        repetitions=int(stats["moving_block_bootstrap_repetitions"]),
        block_length=int(stats["moving_block_length_settlements"]),
        family_size=family_size,
    )
    reasons: list[str] = []
    if len(rows) < minimum_role:
        reasons.append("insufficient_role_population")
    if net <= 0:
        reasons.append("nonpositive_net_after_frozen_hurdles")
    if bootstrap_lower <= Decimal(
        str(stats["required_bootstrap_net_lower_bound_bips"])
    ):
        reasons.append("family_adjusted_bootstrap_lower_bound_not_positive")
    if any(
        not bool(_mapping(value, name="slice")["passes"])
        for value in slice_rows.values()
    ):
        reasons.append("one_or_more_required_slices_failed")
    if maximum_drawdown > Decimal(str(stats["maximum_net_drawdown_bips"])):
        reasons.append("maximum_drawdown_exceeded")
    if expected_shortfall < Decimal(
        str(stats["minimum_net_expected_shortfall_97_5_bips"])
    ):
        reasons.append("expected_shortfall_failed")
    if positive_week_concentration > Decimal(
        str(stats["maximum_positive_week_concentration"])
    ):
        reasons.append("positive_week_concentration_failed")
    return {
        "observation_count": len(rows),
        "first_funding_time_ms": times[0],
        "last_funding_time_ms": times[-1],
        "duration_days": _decimal_text(duration_days),
        "gross_funding_bips": _decimal_text(gross),
        "round_trip_execution_stress_bips": _decimal_text(round_trip),
        "capital_opportunity_hurdle_bips": _decimal_text(capital_hurdle),
        "total_hurdle_bips": _decimal_text(total_hurdle),
        "net_after_frozen_hurdles_bips": _decimal_text(net),
        "family_adjusted_bootstrap_net_lower_bound_bips": _decimal_text(
            bootstrap_lower
        ),
        "maximum_net_drawdown_bips": _decimal_text(maximum_drawdown),
        "net_expected_shortfall_97_5_bips": _decimal_text(expected_shortfall),
        "maximum_positive_week_concentration": _decimal_text(
            positive_week_concentration
        ),
        "slices": slice_rows,
        "passes": not reasons,
        "rejection_reasons": reasons,
    }


def _fill_asks(raw_asks: object, *, target_usdt: Decimal) -> dict[str, object]:
    remaining = target_usdt
    quantity = Decimal(0)
    cost = Decimal(0)
    levels = 0
    for raw_level in _list(raw_asks, name="spot asks"):
        level = _list(raw_level, name="spot ask")
        price = Decimal(str(level[0]))
        available = Decimal(str(level[1]))
        take = min(available, remaining / price)
        quantity += take
        cost += take * price
        remaining -= take * price
        levels += 1
        if remaining <= Decimal("0.00000001"):
            remaining = Decimal(0)
            break
    if remaining > 0 or quantity <= 0:
        raise ValueError(f"spot depth cannot fill {target_usdt} USDT")
    return {
        "quantity": quantity,
        "spot_cost_usdt": cost,
        "spot_vwap": cost / quantity,
        "spot_levels_used": levels,
    }


def _fill_bids(raw_bids: object, *, target_quantity: Decimal) -> dict[str, object]:
    remaining = target_quantity
    proceeds = Decimal(0)
    levels = 0
    for raw_level in _list(raw_bids, name="future bids"):
        level = _list(raw_level, name="future bid")
        price = Decimal(str(level[0]))
        available = Decimal(str(level[1]))
        take = min(available, remaining)
        proceeds += take * price
        remaining -= take
        levels += 1
        if remaining <= Decimal("0.00000001"):
            remaining = Decimal(0)
            break
    if remaining > 0:
        raise ValueError(f"future depth cannot fill {target_quantity} units")
    return {
        "future_short_proceeds_usdt": proceeds,
        "future_vwap": proceeds / target_quantity,
        "future_levels_used": levels,
    }


def _current_depth(
    client: _Client,
    *,
    ticker: str,
    spot_symbol: str,
    future_symbol: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    started_ms = time.time_ns() // 1_000_000
    spot_raw, spot_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/depth",
        name=f"selected-spot-depth-{spot_symbol.lower()}",
        params={"symbol": spot_symbol, "limit": 1000},
    )
    future_raw, future_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/depth",
        name=f"selected-future-depth-{future_symbol.lower()}",
        params={"symbol": future_symbol, "limit": 1000},
    )
    finished_ms = time.time_ns() // 1_000_000
    spot = _mapping(spot_raw, name=f"{ticker} spot depth")
    future = _mapping(future_raw, name=f"{ticker} future depth")
    sizes: list[dict[str, object]] = []
    for target in (Decimal("1000"), Decimal("5000")):
        spot_fill = _fill_asks(spot["asks"], target_usdt=target)
        future_fill = _fill_bids(
            future["bids"], target_quantity=Decimal(spot_fill["quantity"])
        )
        spot_cost = Decimal(spot_fill["spot_cost_usdt"])
        future_proceeds = Decimal(future_fill["future_short_proceeds_usdt"])
        sizes.append(
            {
                "target_usdt": _decimal_text(target),
                "quantity": _decimal_text(Decimal(spot_fill["quantity"])),
                "spot_cost_usdt": _decimal_text(spot_cost),
                "spot_vwap": _decimal_text(Decimal(spot_fill["spot_vwap"])),
                "spot_levels_used": int(spot_fill["spot_levels_used"]),
                "future_short_proceeds_usdt": _decimal_text(future_proceeds),
                "future_vwap": _decimal_text(Decimal(future_fill["future_vwap"])),
                "future_levels_used": int(future_fill["future_levels_used"]),
                "entry_basis_bips": _decimal_text(
                    Decimal(10_000) * (future_proceeds / spot_cost - Decimal(1))
                ),
            }
        )
    return (
        {
            "capture_window_ms": finished_ms - started_ms,
            "sizes": sizes,
            "both_sizes_fillable": len(sizes) == 2,
        },
        [spot_source, future_source],
    )


def run(*, raw_dir: Path) -> dict[str, object]:
    contract, contract_hash = _contract()
    client = _Client(raw_dir)
    started_ms = time.time_ns() // 1_000_000
    spot_exchange_raw, spot_exchange_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/exchangeInfo", name="spot-exchange-info"
    )
    futures_exchange_raw, futures_exchange_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo",
        name="futures-exchange-info",
    )
    bstocks_raw, bstocks_source = client.get(
        BSTOCK_LIST_URL, name="bstock-token-list", params={"type": 3}
    )
    spot_exchange = _mapping(spot_exchange_raw, name="spot exchange info")
    futures_exchange = _mapping(futures_exchange_raw, name="futures exchange info")
    trading_spot = {
        str(row["symbol"])
        for row in (
            _mapping(value, name="spot symbol")
            for value in _list(spot_exchange["symbols"], name="spot symbols")
        )
        if row.get("status") == "TRADING"
    }
    trading_futures = {
        str(row["symbol"])
        for row in (
            _mapping(value, name="future symbol")
            for value in _list(futures_exchange["symbols"], name="future symbols")
        )
        if row.get("status") == "TRADING"
        and row.get("contractType") == "TRADIFI_PERPETUAL"
    }
    envelope = _mapping(bstocks_raw, name="bStock envelope")
    tokens = [
        _mapping(value, name="bStock token")
        for value in _list(envelope["data"], name="bStock tokens")
        if _mapping(value, name="bStock token").get("type") == 3
    ]
    matched = [
        token
        for token in tokens
        if str(token.get("cs")) in trading_spot
        and f"{token.get('ticker')}USDT" in trading_futures
    ]
    exact = [
        token
        for token in matched
        if Decimal(str(token.get("multiplier"))) == Decimal(1)
    ]
    non_exact = [
        {
            "ticker": str(token["ticker"]),
            "spot_symbol": str(token["cs"]),
            "future_symbol": f"{token['ticker']}USDT",
            "multiplier": str(token["multiplier"]),
        }
        for token in matched
        if Decimal(str(token.get("multiplier"))) != Decimal(1)
    ]
    exposure = _mapping(
        contract["outcome_exposure_control"], name="outcome exposure control"
    )
    exposed = {
        str(value)
        for value in _list(
            exposure[
                "previously_observed_exact_multiplier_tickers_excluded_from_confirmation"
            ],
            name="exposed exact tickers",
        )
    }
    eligible_count = sum(str(token["ticker"]) not in exposed for token in exact)
    if eligible_count <= 0:
        raise ValueError("confirmation-eligible universe is empty")
    funding_sources: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    sealed_roles: dict[str, dict[str, list[dict[str, object]]]] = {}
    thresholds_by_symbol: dict[str, dict[str, Decimal]] = {}
    for token in sorted(exact, key=lambda value: str(value["ticker"])):
        ticker = str(token["ticker"])
        future_symbol = f"{ticker}USDT"
        raw_history, source = client.get(
            f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
            name=f"funding-{future_symbol.lower()}",
            params={"symbol": future_symbol, "limit": 1000},
        )
        funding_sources.append(source)
        observations = _observations(raw_history, symbol=future_symbol)
        roles = _split_roles(observations)
        thresholds = _regime_thresholds(roles["training"])
        thresholds_by_symbol[ticker] = thresholds
        training = _role_metrics(
            roles["training"],
            thresholds=thresholds,
            contract=contract,
            symbol=future_symbol,
            role="training",
            family_size=eligible_count,
        )
        eligible = ticker not in exposed
        training_pass = eligible and bool(training["passes"])
        training_rows.append(
            {
                "ticker": ticker,
                "spot_symbol": str(token["cs"]),
                "future_symbol": future_symbol,
                "multiplier": str(token["multiplier"]),
                "previously_observed": not eligible,
                "confirmation_eligible": eligible,
                "regime_thresholds": {
                    key: _decimal_text(value) for key, value in thresholds.items()
                },
                "training": training,
                "selected_by_training_only": training_pass,
            }
        )
        if training_pass:
            sealed_roles[ticker] = roles
    selected_rows: list[dict[str, object]] = []
    depth_sources: list[dict[str, object]] = []
    for training_row in training_rows:
        if not training_row["selected_by_training_only"]:
            continue
        ticker = str(training_row["ticker"])
        roles = sealed_roles[ticker]
        thresholds = thresholds_by_symbol[ticker]
        validation = _role_metrics(
            roles["validation"],
            thresholds=thresholds,
            contract=contract,
            symbol=str(training_row["future_symbol"]),
            role="validation",
            family_size=eligible_count,
        )
        test = _role_metrics(
            roles["test"],
            thresholds=thresholds,
            contract=contract,
            symbol=str(training_row["future_symbol"]),
            role="test",
            family_size=eligible_count,
        )
        historical_pass = bool(validation["passes"]) and bool(test["passes"])
        current_depth: dict[str, object] | None = None
        if historical_pass:
            current_depth, receipts = _current_depth(
                client,
                ticker=ticker,
                spot_symbol=str(training_row["spot_symbol"]),
                future_symbol=str(training_row["future_symbol"]),
            )
            depth_sources.extend(receipts)
        selected_rows.append(
            {
                **training_row,
                "validation": validation,
                "test": test,
                "all_historical_roles_pass": historical_pass,
                "current_depth": current_depth,
                "empirical_gate_pass": historical_pass
                and current_depth is not None
                and bool(current_depth["both_sizes_fillable"]),
            }
        )
    empirical_passes = [row for row in selected_rows if row["empirical_gate_pass"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": time.time_ns() // 1_000_000,
        "capture_started_at_ms": started_ms,
        "purpose": contract["purpose"],
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_placed": False,
            "conversions_requested": False,
            "trading_authority": False,
        },
        "contract": {
            "path": str(CONTRACT_PATH.as_posix()),
            "sha256": contract_hash,
            "status": contract["status"],
        },
        "scope": {
            "research_only_outside_current_btc_eth_sol_execution_scope": True,
            "matched_structural_count": len(matched),
            "exact_multiplier_count": len(exact),
            "non_exact_multiplier_count": len(non_exact),
            "confirmation_eligible_count": eligible_count,
            "previously_observed_exact_count": len(
                exposed & {str(row["ticker"]) for row in exact}
            ),
        },
        "non_exact_multiplier_exclusions": non_exact,
        "training_screen": training_rows,
        "training_selected_count": len(selected_rows),
        "causal_confirmation": selected_rows,
        "empirical_gate_pass_count": len(empirical_passes),
        "empirical_gate_pass_tickers": [str(row["ticker"]) for row in empirical_passes],
        "sources": {
            "spot_exchange_info": spot_exchange_source,
            "futures_exchange_info": futures_exchange_source,
            "bstock_token_list": bstocks_source,
            "funding_histories": funding_sources,
            "selected_current_depth": depth_sources,
        },
        "blocking_evidence": [
            "same-account bStock and TradFi perpetual eligibility",
            "exact account spot and futures fees taxes funding adjustments and rounding",
            "verified bStock conversion availability and inventory reconciliation",
            "historical executable two-leg entry and exit basis rather than funding marks",
            "prospective fill persistence and operational capacity",
            "explicit scope expansion beyond BTC ETH and SOL before execution work",
        ],
        "verdict": {
            "accepted_edge": False,
            "deployment_ready": False,
            "empirical_gate_pass": bool(empirical_passes),
            "status": (
                "research_only_empirical_candidate_account_execution_and_scope_gated"
                if empirical_passes
                else "full_universe_causal_cross_regime_gate_rejected"
            ),
            "trading_authority": False,
        },
        "implementation": {
            "path": str(IMPLEMENTATION_PATH.as_posix()),
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(raw_dir=args.raw_dir)
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(f"output={args.output}")
    print(f"result_sha256={result['result_sha256']}")
    print(f"training_selected_count={result['training_selected_count']}")
    print(f"empirical_gate_pass_count={result['empirical_gate_pass_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
