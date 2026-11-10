"""Apply the frozen Seoul temperature cover screen to retained Gamma bytes."""

from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from simple_ai_trading.partition_cover import separate_partition_cover
from tools.capture_public_source_contract import (
    _load_object,
    _root_path,
    _validate_contract,
)
from tools.screen_polymarket_exact_negrisk_long_only_frontier import _markets
from tools.adjudicate_polymarket_retained_negrisk_pairwise_no_v2 import _fee_model_v2


_LABEL = re.compile(r"(-?\d{1,2})\s*°C(?: or (below|higher))?")
_CONDITION = re.compile(r"0x[0-9a-f]{64}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON key")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _partition_markets(event: dict, contract: dict) -> tuple[list[dict], str, str]:
    if (
        event.get("slug") != contract["event_slug"]
        or not str(event.get("id", "")).isdigit()
    ):
        raise ValueError("Gamma event identity differs")
    if str(event.get("title", "")) != contract["title"]:
        raise ValueError("Gamma event title differs")
    values = event.get("markets")
    if not isinstance(values, list) or not 3 <= len(values) <= 32:
        raise ValueError("event is outside the frozen partition size envelope")
    rows = _markets(event, len(values))
    source = event.get("resolutionSource")
    if not isinstance(source, str) or not source.startswith("https://"):
        raise ValueError("common resolution source is absent")
    description = rows[0].get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("settlement description is absent")
    normalized = description.casefold()
    if not (
        any(term in normalized for term in contract["date_markers"])
        and any(term in normalized for term in contract["station_markers"])
        and "highest temperature" in normalized
        and any(term in normalized for term in contract["integer_celsius_markers"])
    ):
        raise ValueError("settlement date, station or whole-degree Celsius is unbound")
    if any(
        row.get("description") != description or row.get("resolutionSource") != source
        for row in rows
    ):
        raise ValueError("siblings disagree on settlement description or source")
    conditions: set[str] = set()
    labels: list[tuple[int, str, dict]] = []
    for row in rows:
        label = _LABEL.fullmatch(str(row.get("groupItemTitle", "")))
        condition = str(row.get("conditionId", ""))
        if (
            label is None
            or not _CONDITION.fullmatch(condition)
            or condition in conditions
        ):
            raise ValueError("invalid or duplicate condition or integer Celsius label")
        conditions.add(condition)
        labels.append((int(label[1]), label[2] or "", row))
    labels.sort(key=lambda item: item[0])
    for index, (temperature, tail, row) in enumerate(labels):
        wanted_tail = (
            "below" if index == 0 else "higher" if index == len(labels) - 1 else ""
        )
        if (
            tail != wanted_tail
            or temperature != labels[0][0] + index
            or str(row.get("groupItemThreshold")) != str(index)
        ):
            raise ValueError(
                "temperature bins or ordinal identities do not form a complete partition"
            )
    return [item[2] for item in labels], source, description


def _quote(value: object) -> Fraction | None:
    if value is None:
        return None
    if (
        not isinstance(value, (str, int, Decimal))
        or isinstance(value, bool)
        or len(str(value)) > 32
    ):
        raise ValueError("invalid quote representation")
    number = Decimal(value)
    if (
        not number.is_finite()
        or not 0 <= number <= 1
        or number.as_tuple().exponent < -12
    ):
        raise ValueError("quote outside unit interval")
    return Fraction(number) if 0 < number < 1 else None


def _pack(separation) -> dict:
    if separation.state_weights is not None:
        return {"state_weights": [str(weight) for weight in separation.state_weights]}
    witness = separation.strict_cover
    if witness is None:
        raise RuntimeError("partition separator omitted the witness")
    return {
        "yes_quantities": [str(quantity) for quantity in witness.yes_quantities],
        "no_quantities": [str(quantity) for quantity in witness.no_quantities],
        "cost": str(witness.acquisition_cost),
        "payout_floor": str(witness.minimum_payout),
        "surplus": str(witness.floor_surplus),
    }


def _screen(event: dict, contract: dict) -> dict:
    rows, source, description = _partition_markets(event, contract)
    yes = []
    no = []
    for row in rows:
        ask, bid = _quote(row.get("bestAsk")), _quote(row.get("bestBid"))
        if ask is not None and bid is not None and ask < bid:
            raise ValueError("Gamma returned a crossed proxy quote")
        yes.append(ask)
        no.append(1 - bid if bid is not None else None)
    gross = separate_partition_cover(yes, no)
    result = {
        "event_id": str(event["id"]),
        "market_ids": [str(row["id"]) for row in rows],
        "labels": [str(row["groupItemTitle"]) for row in rows],
        "resolution_source_url": source,
        "common_description_sha256": sha256(description.encode()).hexdigest(),
        "yes_price_complete": sum(value is not None for value in yes),
        "no_price_complete": sum(value is not None for value in no),
        "gross_yes_prices_pusd": [None if x is None else str(x) for x in yes],
        "gross_no_proxies_pusd": [None if x is None else str(x) for x in no],
        "gross": _pack(gross),
    }
    if gross.state_weights is not None:
        result["decision"] = "rejected_gross_linear_cover"
        return result
    quantity = Decimal(contract["stress_quantity_shares_each_leg"])
    increments = []
    for row in rows:
        tick = _quote(row.get("orderPriceMinTickSize"))
        minimum = Decimal(str(row.get("orderMinSize")))
        if tick is None or not minimum.is_finite() or minimum <= 0:
            raise ValueError("invalid venue tick or minimum lot")
        peak_fee = _fee_model_v2(row)(Decimal("0.5"), quantity, "taker")
        increments.append(
            (tick + Fraction(peak_fee) / Fraction(quantity), tick, minimum)
        )

    def stressed(costs):
        return [
            None
            if cost is None or cost + tick >= 1 or minimum > quantity
            else cost + increment
            for cost, (increment, tick, minimum) in zip(costs, increments, strict=True)
        ]

    stress = separate_partition_cover(stressed(yes), stressed(no))
    result["stress_at_frozen_quantity"] = _pack(stress)
    result["decision"] = (
        "rejected_configured_fee_and_tick"
        if stress.state_weights is not None
        else "conditional_proxy_candidate_requires_separate_rule_and_depth_qualification"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    path = args.contract.resolve()
    contract = _load_object(path)
    _validate_contract(contract, path)
    output = _root_path(contract["adjudication_path"])
    if output.exists():
        raise FileExistsError("one-use adjudication already exists")
    capture = _load_object(_root_path(contract["outputs"]["result_path"]))
    raw = _root_path(contract["outputs"]["raw_path"]).read_bytes()
    receipt = capture["capture"]["receipt"]
    if (
        capture["contract"]["sha256"] != contract["contract_sha256"]
        or sha256(raw).hexdigest() != receipt["response_sha256"]
    ):
        raise ValueError("native receipt identity differs")
    result = {"contract_sha256": contract["contract_sha256"], "capture": receipt}
    if not capture["source_gate"]["passed"]:
        result["decision"] = "terminal_transport_or_representation_failure"
    else:
        try:
            event = json.loads(
                raw,
                parse_float=Decimal,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
            if not isinstance(event, dict):
                raise ValueError("native event JSON is not an object")
            result.update(_screen(event, contract))
        except (KeyError, ValueError, RuntimeError, ArithmeticError) as failure:
            result.update(
                decision="terminal_rule_or_price_admission_failure",
                failure=str(failure),
            )
    result.update(accepted_edge=False, book_requests=0, account_requests=0)
    result["evidence_sha256"] = sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "market_count": len(result.get("labels", [])),
                "failure": result.get("failure"),
            }
        )
    )


if __name__ == "__main__":
    main()
