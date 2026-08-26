"""Adjudicate the frozen bStock ranked-basket hypothesis on sealed roles."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping

from screen_binance_bstock_funding_full_universe import (
    _canonical_json,
    _contract as _full_universe_contract,
    _mapping,
    _observations,
    _regime_thresholds,
    _role_metrics,
    _sha256,
    _split_roles,
)
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "binance-bstock-ranked-basket-v1"
CONTRACT_PATH = Path(
    "docs/model-research/action-value/binance-bstock-ranked-basket-contract-v1.json"
)
SOURCE_PATH = Path(
    "docs/model-research/action-value/"
    "binance-bstock-funding-full-universe-v1-2026-08-26.json"
)
IMPLEMENTATION_PATH = Path("tools/adjudicate_binance_bstock_ranked_basket.py")
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/binance-bstock-ranked-basket-v1-2026-08-26.json"
)


def _load_contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text()), name="basket contract")
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"basket contract hash differs: {declared=} {actual=}")
    contract["contract_sha256"] = declared
    if contract.get("status") != (
        "frozen_after_training_summary_and_before_validation_or_test_outcome_calculation"
    ):
        raise ValueError("basket contract is not frozen")
    return contract, actual


def _load_source(contract: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(json.loads(SOURCE_PATH.read_text()), name="development source")
    declared = str(source["result_sha256"])
    body = dict(source)
    body.pop("result_sha256")
    actual = _sha256(_canonical_json(body).encode("ascii"))
    expected = str(
        _mapping(contract["development_source"], name="development source contract")[
            "result_sha256"
        ]
    )
    if declared != actual or actual != expected:
        raise ValueError(
            f"development source hash differs: {declared=} {actual=} {expected=}"
        )
    if source.get("causal_confirmation") != []:
        raise ValueError("development source already calculated sealed roles")
    return source


def _raw_history(source: Mapping[str, object], *, symbol: str) -> object:
    receipts = _mapping(source["sources"], name="sources")["funding_histories"]
    for raw_receipt in receipts:
        receipt = _mapping(raw_receipt, name="funding receipt")
        if f"symbol={symbol}&" not in str(receipt["url"]):
            continue
        path = Path(str(receipt["raw_path"]))
        payload = path.read_bytes()
        if _sha256(payload) != receipt["response_sha256"]:
            raise ValueError(f"{symbol} raw funding history hash differs")
        return json.loads(payload)
    raise ValueError(f"{symbol} raw funding history receipt is absent")


def _bootstrap_mean_lower(
    values: list[Decimal], *, role: str, repetitions: int
) -> Decimal:
    seed = int.from_bytes(
        hashlib.sha256(f"{SCHEMA_VERSION}:{role}".encode("ascii")).digest()[:8],
        "big",
    )
    generator = random.Random(seed)
    samples = [
        sum((generator.choice(values) for _ in values), Decimal(0))
        / Decimal(len(values))
        for _ in range(repetitions)
    ]
    samples.sort()
    rank = max(0, math.ceil(Decimal("0.05") * repetitions) - 1)
    return samples[rank]


def _portfolio_role(
    symbol_rows: list[dict[str, object]],
    *,
    role: str,
    contract: Mapping[str, object],
) -> dict[str, object]:
    evaluation = _mapping(contract["evaluation"], name="basket evaluation")
    nets = [
        Decimal(
            str(
                _mapping(row[role], name=f"{role} metrics")[
                    "net_after_frozen_hurdles_bips"
                ]
            )
        )
        for row in symbol_rows
    ]
    mean_net = sum(nets, Decimal(0)) / Decimal(len(nets))
    positive_fraction = Decimal(sum(value > 0 for value in nets)) / Decimal(len(nets))
    leave_one_out = [
        sum(
            (value for index, value in enumerate(nets) if index != excluded), Decimal(0)
        )
        / Decimal(len(nets) - 1)
        for excluded in range(len(nets))
    ]
    bootstrap_lower = _bootstrap_mean_lower(
        nets,
        role=role,
        repetitions=int(evaluation["cross_sectional_bootstrap_repetitions"]),
    )
    slice_labels = [
        "bullish",
        "bearish",
        "sideways",
        "low_or_normal_volatility",
        "high_volatility",
        "stress_volatility",
        "directional",
        "choppy",
    ]
    slices: dict[str, object] = {}
    for label in slice_labels:
        contributions: list[Decimal] = []
        for row in symbol_rows:
            metrics = _mapping(row[role], name=f"{role} metrics")
            slice_row = _mapping(
                _mapping(metrics["slices"], name="slices")[label], name=label
            )
            if int(slice_row["observation_count"]) > 0:
                contributions.append(
                    Decimal(str(slice_row["net_after_allocated_hurdle_bips"]))
                )
        aggregate = (
            sum(contributions, Decimal(0)) / Decimal(len(contributions))
            if contributions
            else Decimal(0)
        )
        passes = len(contributions) >= int(
            evaluation["required_contributing_symbols_per_slice"]
        ) and aggregate >= Decimal(
            str(
                evaluation[
                    "required_aggregate_direction_volatility_and_path_slice_net_bips"
                ]
            )
        )
        slices[label] = {
            "contributing_symbol_count": len(contributions),
            "equal_weight_mean_net_after_allocated_hurdle_bips": format(aggregate, "f"),
            "passes": passes,
        }
    reasons: list[str] = []
    if mean_net <= 0:
        reasons.append("portfolio_mean_net_not_positive")
    if positive_fraction < Decimal(
        str(evaluation["required_positive_symbol_fraction"])
    ):
        reasons.append("positive_symbol_fraction_failed")
    if min(leave_one_out) <= Decimal(
        str(evaluation["required_worst_leave_one_symbol_out_net_bips"])
    ):
        reasons.append("worst_leave_one_symbol_out_net_failed")
    if bootstrap_lower <= Decimal(
        str(evaluation["required_bootstrap_net_lower_bound_bips"])
    ):
        reasons.append("cross_sectional_bootstrap_lower_bound_not_positive")
    if any(
        not bool(_mapping(value, name="portfolio slice")["passes"])
        for value in slices.values()
    ):
        reasons.append("one_or_more_aggregate_regime_slices_failed")
    return {
        "selected_symbol_count": len(symbol_rows),
        "equal_weight_mean_net_after_frozen_hurdles_bips": format(mean_net, "f"),
        "positive_symbol_count": sum(value > 0 for value in nets),
        "positive_symbol_fraction": format(positive_fraction, "f"),
        "worst_leave_one_symbol_out_net_bips": format(min(leave_one_out), "f"),
        "cross_sectional_bootstrap_net_lower_bound_bips": format(bootstrap_lower, "f"),
        "slices": slices,
        "passes": not reasons,
        "rejection_reasons": reasons,
    }


def run() -> dict[str, object]:
    contract, contract_hash = _load_contract()
    source = _load_source(contract)
    selector = _mapping(contract["selector"], name="selector")
    eligible = [
        _mapping(row, name="training row")
        for row in source["training_screen"]
        if _mapping(row, name="training row")["confirmation_eligible"] is True
    ]
    eligible.sort(
        key=lambda row: (
            -Decimal(
                str(
                    _mapping(row["training"], name="training metrics")[
                        "net_after_frozen_hurdles_bips"
                    ]
                )
            ),
            str(row["ticker"]),
        )
    )
    selected_count = math.ceil(len(eligible) * Decimal(str(selector["top_fraction"])))
    selected = eligible[:selected_count]
    if len(selected) != 12:
        raise ValueError(f"frozen selector count differs: {len(selected)}")
    if bool(selector["require_positive_training_net"]) and any(
        Decimal(
            str(
                _mapping(row["training"], name="training metrics")[
                    "net_after_frozen_hurdles_bips"
                ]
            )
        )
        <= 0
        for row in selected
    ):
        raise ValueError("frozen selector includes a nonpositive training symbol")
    full_contract, _ = _full_universe_contract()
    evaluated: list[dict[str, object]] = []
    for row in selected:
        ticker = str(row["ticker"])
        future_symbol = str(row["future_symbol"])
        roles = _split_roles(
            _observations(
                _raw_history(source, symbol=future_symbol), symbol=future_symbol
            )
        )
        thresholds = _regime_thresholds(roles["training"])
        recorded_thresholds = _mapping(row["regime_thresholds"], name="thresholds")
        if any(
            format(value, "f") != str(recorded_thresholds[key])
            for key, value in thresholds.items()
        ):
            raise ValueError(f"{ticker} training threshold reconstruction differs")
        role_metrics = {
            role: _role_metrics(
                roles[role],
                thresholds=thresholds,
                contract=full_contract,
                symbol=future_symbol,
                role=role,
                family_size=int(source["scope"]["confirmation_eligible_count"]),
            )
            for role in ("validation", "test")
        }
        evaluated.append(
            {
                "ticker": ticker,
                "spot_symbol": row["spot_symbol"],
                "future_symbol": future_symbol,
                "training_net_after_frozen_hurdles_bips": _mapping(
                    row["training"], name="training metrics"
                )["net_after_frozen_hurdles_bips"],
                **role_metrics,
            }
        )
    portfolio = {
        role: _portfolio_role(evaluated, role=role, contract=contract)
        for role in ("validation", "test")
    }
    both_pass = all(bool(portfolio[role]["passes"]) for role in portfolio)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": contract["hypothesis"],
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_placed": False,
            "trading_authority": False,
        },
        "contract": {
            "path": CONTRACT_PATH.as_posix(),
            "sha256": contract_hash,
            "status": contract["status"],
        },
        "development_source": {
            "path": SOURCE_PATH.as_posix(),
            "result_sha256": source["result_sha256"],
        },
        "selection": {
            "eligible_count": len(eligible),
            "selected_count": len(selected),
            "selected_tickers": [str(row["ticker"]) for row in selected],
            "top_fraction": selector["top_fraction"],
            "weights": selector["weights"],
        },
        "selected_symbol_results": evaluated,
        "portfolio_roles": portfolio,
        "verdict": {
            "validation_pass": bool(portfolio["validation"]["passes"]),
            "test_pass": bool(portfolio["test"]["passes"]),
            "persistent_after_frozen_sensitivity_research_candidate": both_pass,
            "accepted_edge": False,
            "deployment_ready": False,
            "status": (
                "persistent_research_candidate_account_execution_and_scope_gated"
                if both_pass
                else "ranked_basket_rejected_without_parameter_retry"
            ),
            "trading_authority": False,
        },
        "blocking_evidence": [
            "same-account bStock and TradFi perpetual eligibility",
            "exact account fees taxes and rounding",
            "historical executable two-leg entry and exit basis",
            "calendar-synchronous portfolio replay instead of lifecycle-role aggregation",
            "prospective capacity and operational reconciliation",
            "explicit scope expansion beyond BTC ETH and SOL",
        ],
        "implementation": {
            "path": IMPLEMENTATION_PATH.as_posix(),
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(f"output={args.output}")
    print(f"result_sha256={result['result_sha256']}")
    print(f"selected={','.join(result['selection']['selected_tickers'])}")
    print(f"validation_pass={result['verdict']['validation_pass']}")
    print(f"test_pass={result['verdict']['test_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
