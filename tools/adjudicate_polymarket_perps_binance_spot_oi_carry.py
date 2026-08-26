"""Adjudicate retained BTC Polymarket Perps funding hedged by Binance spot."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.storage import write_bytes_atomic


HOUR_MS = 3_600_000
EIGHT_HOURS_MS = 8 * HOUR_MS
YEAR_HOURS = Decimal("8760")


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _decimal(value: object, *, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _verify_self_hash(payload: Mapping[str, object]) -> str:
    body = dict(payload)
    claimed = str(body.pop("result_sha256", ""))
    observed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != observed:
        raise ValueError(
            f"contract result_sha256 mismatch: expected {claimed}, observed {observed}"
        )
    return claimed


def _normalize_hour(timestamp_ms: int, tolerance_ms: int) -> tuple[int, int]:
    rounded = ((timestamp_ms + HOUR_MS // 2) // HOUR_MS) * HOUR_MS
    residual = timestamp_ms - rounded
    if abs(residual) > tolerance_ms:
        raise ValueError(
            f"funding timestamp residual {residual} exceeds {tolerance_ms} ms"
        )
    return rounded, residual


def role_economics(
    funding: Mapping[int, Decimal],
    *,
    start_ms: int,
    end_ms: int,
    annual_reward_bips: Decimal,
    annual_opportunity_bips_per_leg: Decimal,
) -> dict[str, object]:
    if end_ms <= start_ms or (end_ms - start_ms) % HOUR_MS:
        raise ValueError("role window must contain a positive whole number of hours")
    hours = (end_ms - start_ms) // HOUR_MS
    expected = tuple(range(start_ms + HOUR_MS, end_ms + 1, HOUR_MS))
    observed_rates = [
        funding[timestamp] for timestamp in expected if timestamp in funding
    ]
    funding_sum = sum(observed_rates, Decimal("0"))
    reward = annual_reward_bips / Decimal("10000") * Decimal(hours) / YEAR_HOURS
    opportunity = (
        annual_opportunity_bips_per_leg
        / Decimal("10000")
        * Decimal("2")
        * Decimal(hours)
        / YEAR_HOURS
    )
    excess = funding_sum + reward - opportunity
    return {
        "calendar_hours": hours,
        "excess_before_round_trip_and_basis_bips": str(excess * Decimal("10000")),
        "expected_funding_hours": len(expected),
        "funding_bips": str(funding_sum * Decimal("10000")),
        "missing_funding_hours_valued_at_zero": len(expected) - len(observed_rates),
        "observed_funding_hours": len(observed_rates),
        "opportunity_hurdle_bips": str(opportunity * Decimal("10000")),
        "reward_bips": str(reward * Decimal("10000")),
    }


def _stats(values: Sequence[Decimal]) -> dict[str, object]:
    return {
        "count": len(values),
        "positive_count": sum(value > 0 for value in values),
        "sum_bips": str(sum(values, Decimal("0")) * Decimal("10000")),
    }


def _load_bound_response(
    raw_root: Path,
    entry: Mapping[str, object],
) -> object:
    path = Path(str(entry["raw_path"]))
    if not path.is_absolute():
        path = raw_root.parent.parent / path
    raw = path.read_bytes()
    observed = _sha256(raw)
    expected = str(entry["payload_sha256"])
    if observed != expected:
        raise ValueError(f"raw response hash mismatch for {path}")
    return json.loads(raw)


def run(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _verify_self_hash(contract)
    if _sha256(Path(__file__).read_bytes()) != str(contract["implementation_sha256"]):
        raise ValueError("implementation SHA-256 does not match frozen contract")
    raw_root = Path(str(contract["raw_directory"]))
    journal_path = raw_root / "journal.json"
    journal_raw = journal_path.read_bytes()
    if _sha256(journal_raw) != str(contract["journal_sha256"]):
        raise ValueError("journal SHA-256 does not match frozen contract")
    journal = _mapping(json.loads(journal_raw), name="journal")
    responses = [
        _mapping(value, name="journal response")
        for value in _list(journal.get("responses"), name="journal responses")
    ]
    labels = {str(value["label"]): value for value in responses}
    funding: dict[int, Decimal] = {}
    residuals: list[int] = []
    for label in sorted(
        value for value in labels if value.startswith("polymarket-btc-funding-")
    ):
        page = _mapping(
            _load_bound_response(raw_root, labels[label]), name="funding page"
        )
        for raw_row in _list(page.get("data"), name="funding rows"):
            row = _mapping(raw_row, name="funding row")
            raw_timestamp = row.get("timestamp")
            if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, int):
                raise ValueError("funding timestamp is invalid")
            timestamp, residual = _normalize_hour(
                raw_timestamp, int(contract["timestamp_tolerance_ms"])
            )
            if timestamp in funding:
                raise ValueError("duplicate normalized funding timestamp")
            funding[timestamp] = _decimal(row.get("funding_rate"), name="funding rate")
            residuals.append(residual)
    kline_payload = _load_bound_response(raw_root, labels["binance-btc-klines"])
    klines: list[tuple[int, Decimal, Decimal]] = []
    for raw_row in _list(kline_payload, name="Binance klines"):
        row = _list(raw_row, name="Binance kline")
        if len(row) < 5 or isinstance(row[0], bool) or not isinstance(row[0], int):
            raise ValueError("Binance kline is invalid")
        klines.append(
            (
                row[0],
                _decimal(row[1], name="kline open"),
                _decimal(row[4], name="kline close"),
            )
        )
    start_ms = int(contract["start_timestamp_ms"])
    end_ms = int(contract["end_timestamp_ms"])
    role_width = (end_ms - start_ms) // 3
    role_bounds = {
        "training": (start_ms, start_ms + role_width),
        "validation": (start_ms + role_width, start_ms + 2 * role_width),
        "test": (start_ms + 2 * role_width, end_ms),
    }
    roles = {
        name: role_economics(
            funding,
            start_ms=role_start,
            end_ms=role_end,
            annual_reward_bips=Decimal(str(contract["annual_oi_reward_bips"])),
            annual_opportunity_bips_per_leg=Decimal(
                str(contract["annual_opportunity_hurdle_bips_per_leg"])
            ),
        )
        for name, (role_start, role_end) in role_bounds.items()
    }
    full = role_economics(
        funding,
        start_ms=start_ms,
        end_ms=end_ms,
        annual_reward_bips=Decimal(str(contract["annual_oi_reward_bips"])),
        annual_opportunity_bips_per_leg=Decimal(
            str(contract["annual_opportunity_hurdle_bips_per_leg"])
        ),
    )
    friction = Decimal(str(contract["round_trip_and_basis_hurdle_bips"]))
    full_net = Decimal(str(full["excess_before_round_trip_and_basis_bips"])) - friction
    interval_reward = (
        Decimal(str(contract["annual_oi_reward_bips"]))
        / Decimal("10000")
        * Decimal("8")
        / YEAR_HOURS
    )
    interval_opportunity = (
        Decimal(str(contract["annual_opportunity_hurdle_bips_per_leg"]))
        / Decimal("10000")
        * Decimal("2")
        * Decimal("8")
        / YEAR_HOURS
    )
    regime_rows: list[tuple[Decimal, Decimal]] = []
    for open_time, open_price, close_price in klines:
        close_time = open_time + EIGHT_HOURS_MS
        if open_time < start_ms or close_time > end_ms or open_price <= 0:
            continue
        hourly = tuple(range(open_time + HOUR_MS, close_time + 1, HOUR_MS))
        carry = (
            sum(
                (funding.get(timestamp, Decimal("0")) for timestamp in hourly),
                Decimal("0"),
            )
            + interval_reward
            - interval_opportunity
        )
        regime_rows.append((close_price / open_price - Decimal("1"), carry))
    absolute_returns = sorted(abs(value[0]) for value in regime_rows)
    median_abs = absolute_returns[len(absolute_returns) // 2]
    regimes = {
        "down": [carry for ret, carry in regime_rows if ret < Decimal("-0.0025")],
        "high_volatility": [
            carry for ret, carry in regime_rows if abs(ret) >= median_abs
        ],
        "low_volatility": [
            carry for ret, carry in regime_rows if abs(ret) < median_abs
        ],
        "sideways": [
            carry for ret, carry in regime_rows if abs(ret) <= Decimal("0.0025")
        ],
        "up": [carry for ret, carry in regime_rows if ret > Decimal("0.0025")],
    }
    regime_stats = {name: _stats(values) for name, values in regimes.items()}
    role_pass = all(
        Decimal(str(value["excess_before_round_trip_and_basis_bips"])) > 0
        for value in roles.values()
    )
    regime_pass = all(
        int(value["count"]) >= 5 and Decimal(str(value["sum_bips"])) > 0
        for value in regime_stats.values()
    )
    coverage = Decimal(int(full["observed_funding_hours"])) / Decimal(
        int(full["expected_funding_hours"])
    )
    public_candidate = (
        full_net > 0
        and role_pass
        and regime_pass
        and coverage >= Decimal(str(contract["minimum_funding_coverage_fraction"]))
    )
    result: dict[str, object] = {
        "contract_path": contract_path.as_posix(),
        "contract_result_sha256": contract_hash,
        "decision": {
            "accepted_edge": False,
            "public_persistence_candidate": public_candidate,
            "requires_authenticated_and_execution_evidence": public_candidate,
        },
        "full_window": {
            **full,
            "funding_coverage_fraction": str(coverage),
            "net_after_round_trip_and_basis_hurdle_bips": str(full_net),
            "round_trip_and_basis_hurdle_bips": str(friction),
        },
        "funding_rate_statistics": {
            "maximum_hourly_bips": str(max(funding.values()) * Decimal("10000")),
            "minimum_hourly_bips": str(min(funding.values()) * Decimal("10000")),
            "negative_count": sum(value < 0 for value in funding.values()),
            "positive_count": sum(value > 0 for value in funding.values()),
            "zero_count": sum(value == 0 for value in funding.values()),
        },
        "regimes": regime_stats,
        "result_sha256": "",
        "roles": roles,
        "schema_version": "polymarket-perps-binance-spot-oi-carry-result-v1",
        "timestamp_normalization": {
            "maximum_residual_ms": max(residuals),
            "minimum_residual_ms": min(residuals),
            "rule": "nearest_UTC_hour",
            "tolerance_ms": int(contract["timestamp_tolerance_ms"]),
        },
    }
    body = dict(result)
    body.pop("result_sha256")
    result["result_sha256"] = _sha256(_canonical_json(body).encode("ascii"))
    write_bytes_atomic(output_path, (_canonical_json(result) + "\n").encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.contract, args.output)
    decision = _mapping(result["decision"], name="decision")
    print(f"public_persistence_candidate={decision['public_persistence_candidate']}")
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
