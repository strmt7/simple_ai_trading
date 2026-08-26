"""Screen Polymarket Perps OI rewards plus Binance-neutral funding carry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


POLYMARKET_BASE_URL = "https://api.perpetuals.polymarket.com"
BINANCE_BASE_URL = "https://fapi.binance.com"
ASSETS = ("BTC", "ETH", "SOL")
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


def _canonical_payload_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    observed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != observed:
        raise ValueError(f"{field} mismatch: expected {claimed}, observed {observed}")
    return claimed


@dataclass(frozen=True)
class AlignedFunding:
    timestamp_ms: int
    polymarket_rate_8h: Decimal
    binance_rate_8h: Decimal
    return_8h: Decimal

    @property
    def difference(self) -> Decimal:
        return self.polymarket_rate_8h - self.binance_rate_8h


class RawJournal:
    def __init__(self, root: Path, contract_hash: str) -> None:
        self.root = root
        self.path = root / "journal.json"
        if self.path.exists() or any(root.glob("response-*.json")):
            raise RuntimeError("raw journal already exists; frozen screen is one-use")
        self.payload: dict[str, object] = {
            "contract_result_sha256": contract_hash,
            "created_at_ms": time.time_ns() // 1_000_000,
            "responses": [],
            "status": "running",
        }
        root.mkdir(parents=True, exist_ok=True)
        self._persist()

    def _persist(self) -> None:
        write_bytes_atomic(
            self.path,
            (_canonical_json(self.payload) + "\n").encode("ascii"),
        )

    def record(
        self,
        *,
        label: str,
        response: requests.Response,
        requested_before_ms: int,
        received_after_ms: int,
    ) -> bytes:
        responses = _list(self.payload["responses"], name="journal responses")
        index = len(responses) + 1
        raw_path = self.root / f"response-{index:02d}-{label}.json"
        write_bytes_atomic(raw_path, response.content)
        responses.append(
            {
                "http_status": response.status_code,
                "label": label,
                "payload_sha256": _sha256(response.content),
                "raw_path": raw_path.as_posix(),
                "received_after_ms": received_after_ms,
                "request_elapsed_ms": received_after_ms - requested_before_ms,
                "requested_before_ms": requested_before_ms,
                "url": response.url,
            }
        )
        self.payload["responses"] = responses
        self._persist()
        return response.content

    def complete(self, result_hash: str) -> None:
        self.payload["completed_at_ms"] = time.time_ns() // 1_000_000
        self.payload["result_sha256"] = result_hash
        self.payload["status"] = "complete"
        self._persist()


def _get_json(
    session: requests.Session,
    journal: RawJournal,
    *,
    label: str,
    url: str,
    params: Mapping[str, object] | None = None,
) -> object:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, params=params, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    raw = journal.record(
        label=label,
        response=response,
        requested_before_ms=before_ms,
        received_after_ms=after_ms,
    )
    if response.status_code == 429:
        raise RuntimeError(
            "rate limited; stopped without retry after retaining response"
        )
    response.raise_for_status()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} did not return JSON; raw response retained") from exc


def _parse_polymarket_funding_page(
    payload: object,
) -> tuple[list[tuple[int, Decimal]], bool]:
    page = _mapping(payload, name="Polymarket funding page")
    rows: list[tuple[int, Decimal]] = []
    for raw in _list(page.get("data"), name="Polymarket funding data"):
        row = _mapping(raw, name="Polymarket funding row")
        timestamp = row.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("Polymarket funding timestamp must be an integer")
        rows.append(
            (
                timestamp,
                _decimal(row.get("funding_rate"), name="Polymarket funding rate"),
            )
        )
    if rows != sorted(rows, reverse=True):
        raise ValueError("Polymarket funding page is not newest-first")
    more = page.get("more")
    if not isinstance(more, bool):
        raise ValueError("Polymarket funding more flag must be boolean")
    if more and not rows:
        raise ValueError("Polymarket funding page cannot be empty when more is true")
    return rows, more


def aggregate_aligned_funding(
    polymarket_rows: Sequence[tuple[int, Decimal]],
    binance_rows: Sequence[tuple[int, Decimal]],
    kline_rows: Sequence[tuple[int, Decimal, Decimal]],
) -> tuple[AlignedFunding, ...]:
    polymarket = dict(polymarket_rows)
    if len(polymarket) != len(polymarket_rows):
        raise ValueError("duplicate Polymarket funding timestamp")
    klines = {
        open_time: (open_price, close_price)
        for open_time, open_price, close_price in kline_rows
    }
    if len(klines) != len(kline_rows):
        raise ValueError("duplicate Binance kline timestamp")
    result: list[AlignedFunding] = []
    for timestamp, binance_rate in sorted(binance_rows):
        hourly_timestamps = tuple(timestamp - offset * HOUR_MS for offset in range(8))
        if any(value not in polymarket for value in hourly_timestamps):
            continue
        kline = klines.get(timestamp - EIGHT_HOURS_MS)
        if kline is None or kline[0] <= 0:
            continue
        result.append(
            AlignedFunding(
                timestamp_ms=timestamp,
                polymarket_rate_8h=sum(
                    (polymarket[value] for value in hourly_timestamps), Decimal("0")
                ),
                binance_rate_8h=binance_rate,
                return_8h=kline[1] / kline[0] - Decimal("1"),
            )
        )
    return tuple(result)


def _stats(values: Sequence[Decimal]) -> dict[str, object]:
    if not values:
        return {"count": 0, "sum_bips": "0"}
    total = sum(values, Decimal("0"))
    return {
        "count": len(values),
        "maximum_bips": str(max(values) * Decimal("10000")),
        "mean_bips": str(total * Decimal("10000") / len(values)),
        "minimum_bips": str(min(values) * Decimal("10000")),
        "positive_count": sum(value > 0 for value in values),
        "sum_bips": str(total * Decimal("10000")),
    }


def evaluate_aligned_funding(
    rows: Sequence[AlignedFunding],
    *,
    execution_hurdle_bips: Decimal,
    annual_oi_reward_bips: Decimal,
    annual_opportunity_hurdle_bips_per_leg: Decimal,
) -> dict[str, object]:
    if len(rows) < 12:
        raise ValueError("at least 12 aligned rows are required")
    first_end = len(rows) // 2
    second_end = first_end + (len(rows) - first_end) // 2
    slices = {
        "training": rows[:first_end],
        "validation": rows[first_end:second_end],
        "test": rows[second_end:],
    }
    training_mean = sum(
        (row.difference for row in slices["training"]), Decimal("0")
    ) / len(slices["training"])
    orientation = Decimal("1") if training_mean >= 0 else Decimal("-1")
    reward_per_row = (
        annual_oi_reward_bips / Decimal("10000") * Decimal("8") / YEAR_HOURS
    )
    opportunity_per_row = (
        annual_opportunity_hurdle_bips_per_leg
        / Decimal("10000")
        * Decimal("2")
        * Decimal("8")
        / YEAR_HOURS
    )
    execution = execution_hurdle_bips / Decimal("10000")
    role_results: dict[str, object] = {}
    for role, role_rows in slices.items():
        funding = [orientation * row.difference for row in role_rows]
        reward = reward_per_row * len(role_rows)
        opportunity = opportunity_per_row * len(role_rows)
        net = sum(funding, Decimal("0")) + reward - opportunity - execution
        role_results[role] = {
            "funding": _stats(funding),
            "net_after_reward_execution_and_capital_bips": str(net * Decimal("10000")),
            "opportunity_hurdle_bips": str(opportunity * Decimal("10000")),
            "reward_bips": str(reward * Decimal("10000")),
        }
    returns = [row.return_8h for row in rows]
    absolute_returns = sorted(abs(value) for value in returns)
    median_abs = absolute_returns[len(absolute_returns) // 2]
    regimes = {
        "down": [row for row in rows if row.return_8h < Decimal("-0.0025")],
        "high_volatility": [row for row in rows if abs(row.return_8h) >= median_abs],
        "low_volatility": [row for row in rows if abs(row.return_8h) < median_abs],
        "sideways": [row for row in rows if abs(row.return_8h) <= Decimal("0.0025")],
        "up": [row for row in rows if row.return_8h > Decimal("0.0025")],
    }
    regime_results = {
        name: _stats(
            [orientation * row.difference + reward_per_row for row in selected]
        )
        for name, selected in regimes.items()
    }
    role_pass = all(
        Decimal(str(result["net_after_reward_execution_and_capital_bips"])) > 0
        for result in role_results.values()
        if isinstance(result, Mapping)
    )
    regime_pass = all(
        int(result["count"]) >= 5 and Decimal(str(result["sum_bips"])) > 0
        for result in regime_results.values()
    )
    return {
        "orientation": (
            "short_polymarket_long_binance"
            if orientation > 0
            else "long_polymarket_short_binance"
        ),
        "public_persistence_candidate": role_pass and regime_pass,
        "regimes": regime_results,
        "roles": role_results,
        "training_mean_raw_difference_bips": str(training_mean * Decimal("10000")),
    }


def _instrument_map(payload: object) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for raw in _list(payload, name="Polymarket instruments"):
        instrument = _mapping(raw, name="Polymarket instrument")
        asset = str(instrument.get("base_asset") or "").upper()
        if asset not in ASSETS or str(instrument.get("category") or "") != "crypto":
            continue
        if str(instrument.get("funding_interval") or "") != "1h":
            raise ValueError(f"{asset} funding interval is not 1h")
        if asset in selected:
            raise ValueError(f"multiple Polymarket instruments for {asset}")
        instrument_id = instrument.get("instrument_id")
        if isinstance(instrument_id, bool) or not isinstance(instrument_id, int):
            raise ValueError(f"{asset} instrument ID is invalid")
        selected[asset] = instrument
    return selected


def _parse_binance_funding(payload: object) -> list[tuple[int, Decimal]]:
    rows: list[tuple[int, Decimal]] = []
    for raw in _list(payload, name="Binance funding rows"):
        row = _mapping(raw, name="Binance funding row")
        timestamp = row.get("fundingTime")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("Binance funding timestamp is invalid")
        rows.append(
            (timestamp, _decimal(row.get("fundingRate"), name="Binance funding rate"))
        )
    return rows


def _parse_binance_klines(payload: object) -> list[tuple[int, Decimal, Decimal]]:
    rows: list[tuple[int, Decimal, Decimal]] = []
    for raw in _list(payload, name="Binance klines"):
        row = _list(raw, name="Binance kline")
        if len(row) < 5 or isinstance(row[0], bool) or not isinstance(row[0], int):
            raise ValueError("Binance kline is invalid")
        rows.append(
            (
                row[0],
                _decimal(row[1], name="Binance kline open"),
                _decimal(row[4], name="Binance kline close"),
            )
        )
    return rows


def run(contract_path: Path, output_path: Path, raw_root: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_payload_hash(contract, "result_sha256")
    expected_implementation = str(contract.get("implementation_sha256") or "")
    if expected_implementation != _sha256(Path(__file__).read_bytes()):
        raise ValueError("implementation SHA-256 does not match frozen contract")
    request_contract = _mapping(
        contract.get("request_contract"), name="request contract"
    )
    start_ms = int(request_contract["start_timestamp_ms"])
    end_ms = int(request_contract["end_timestamp_ms"])
    max_pages = int(request_contract["maximum_polymarket_funding_pages_per_asset"])
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-perps-oi-carry-research/1.0",
        }
    )
    journal = RawJournal(raw_root, contract_hash)
    instruments_payload = _get_json(
        session,
        journal,
        label="polymarket-instruments",
        url=f"{POLYMARKET_BASE_URL}/v1/info/instruments",
    )
    fees_payload = _get_json(
        session,
        journal,
        label="polymarket-fees",
        url=f"{POLYMARKET_BASE_URL}/v1/info/fees",
    )
    instruments = _instrument_map(instruments_payload)
    asset_results: dict[str, object] = {}
    for asset in ASSETS:
        instrument = instruments.get(asset)
        if instrument is None:
            asset_results[asset] = {"status": "missing_polymarket_instrument"}
            continue
        instrument_id = int(instrument["instrument_id"])
        funding_rows: list[tuple[int, Decimal]] = []
        page_end = end_ms
        source_complete = False
        for page_number in range(1, max_pages + 1):
            payload = _get_json(
                session,
                journal,
                label=f"polymarket-{asset.lower()}-funding-{page_number:02d}",
                url=f"{POLYMARKET_BASE_URL}/v1/info/funding",
                params={
                    "instrument_id": instrument_id,
                    "start_timestamp": start_ms,
                    "end_timestamp": page_end,
                },
            )
            page_rows, more = _parse_polymarket_funding_page(payload)
            funding_rows.extend(page_rows)
            if not more:
                source_complete = True
                break
            page_end = page_rows[-1][0] - 1
            if page_end < start_ms:
                source_complete = True
                break
        unique_funding = sorted(set(funding_rows))
        if len(unique_funding) != len(funding_rows):
            raise ValueError(f"duplicate Polymarket {asset} funding row across pages")
        if not source_complete or len(unique_funding) < 480:
            asset_results[asset] = {
                "instrument": instrument,
                "polymarket_funding_count": len(unique_funding),
                "source_complete": source_complete,
                "status": "insufficient_polymarket_history",
            }
            continue
        binance_funding_payload = _get_json(
            session,
            journal,
            label=f"binance-{asset.lower()}-funding",
            url=f"{BINANCE_BASE_URL}/fapi/v1/fundingRate",
            params={
                "symbol": f"{asset}USDT",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        binance_klines_payload = _get_json(
            session,
            journal,
            label=f"binance-{asset.lower()}-klines",
            url=f"{BINANCE_BASE_URL}/fapi/v1/klines",
            params={
                "symbol": f"{asset}USDT",
                "interval": "8h",
                "startTime": start_ms - EIGHT_HOURS_MS,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        aligned = aggregate_aligned_funding(
            unique_funding,
            _parse_binance_funding(binance_funding_payload),
            _parse_binance_klines(binance_klines_payload),
        )
        if len(aligned) < int(contract["minimum_aligned_rows"]):
            asset_results[asset] = {
                "aligned_count": len(aligned),
                "instrument": instrument,
                "polymarket_funding_count": len(unique_funding),
                "source_complete": source_complete,
                "status": "insufficient_aligned_history",
            }
            continue
        asset_results[asset] = {
            "aligned_count": len(aligned),
            "first_aligned_timestamp_ms": aligned[0].timestamp_ms,
            "funding_only": evaluate_aligned_funding(
                aligned,
                execution_hurdle_bips=Decimal(str(contract["execution_hurdle_bips"])),
                annual_oi_reward_bips=Decimal("0"),
                annual_opportunity_hurdle_bips_per_leg=Decimal(
                    str(contract["annual_opportunity_hurdle_bips_per_leg"])
                ),
            ),
            "instrument": instrument,
            "last_aligned_timestamp_ms": aligned[-1].timestamp_ms,
            "polymarket_funding_count": len(unique_funding),
            "source_complete": source_complete,
            "status": "evaluated",
            "with_conditional_oi_reward": evaluate_aligned_funding(
                aligned,
                execution_hurdle_bips=Decimal(str(contract["execution_hurdle_bips"])),
                annual_oi_reward_bips=Decimal(str(contract["annual_oi_reward_bips"])),
                annual_opportunity_hurdle_bips_per_leg=Decimal(
                    str(contract["annual_opportunity_hurdle_bips_per_leg"])
                ),
            ),
        }
    candidates = [
        asset
        for asset, raw in asset_results.items()
        if isinstance(raw, Mapping)
        and isinstance(raw.get("with_conditional_oi_reward"), Mapping)
        and raw["with_conditional_oi_reward"].get("public_persistence_candidate")
        is True
    ]
    result: dict[str, object] = {
        "asset_results": asset_results,
        "contract_path": contract_path.as_posix(),
        "contract_result_sha256": contract_hash,
        "fee_schedule": fees_payload,
        "finished_at_ms": time.time_ns() // 1_000_000,
        "public_persistence_candidates": candidates,
        "result_sha256": "",
        "schema_version": "polymarket-binance-perps-oi-carry-result-v1",
        "status": (
            "candidate_requires_authenticated_account_and_execution_evidence"
            if candidates
            else "rejected_public_screen"
        ),
    }
    body = dict(result)
    body.pop("result_sha256")
    result["result_sha256"] = _sha256(_canonical_json(body).encode("ascii"))
    write_bytes_atomic(output_path, (_canonical_json(result) + "\n").encode("ascii"))
    journal.complete(str(result["result_sha256"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output, args.raw_root)
    print(f"status={result['status']}")
    print(f"candidates={','.join(result['public_persistence_candidates']) or 'none'}")
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
