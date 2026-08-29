"""Prefilter matched Polymarket and Binance TradFi perpetual funding carry."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


POLYMARKET_INSTRUMENTS_URL = "https://api.perpetuals.polymarket.com/v1/info/instruments"
POLYMARKET_TICKERS_URL = "https://api.perpetuals.polymarket.com/v1/info/tickers"
BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
EMPTY_BODY_SHA256 = hashlib.sha256(b"").hexdigest()
TRADFI_CATEGORIES = frozenset({"commodity", "equity", "index"})


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


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_payload_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    observed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != observed:
        raise ValueError(f"{field} mismatch: expected {claimed}, observed {observed}")
    return claimed


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an explicit UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be parseable") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be UTC")
    return parsed


class RawJournal:
    """Persist the exact request plan before access and every response before parse."""

    def __init__(
        self,
        root: Path,
        *,
        contract_hash: str,
        request_plan: Sequence[Mapping[str, object]],
    ) -> None:
        self.root = root
        self.path = root / "journal.json"
        if self.path.exists() or any(root.glob("response-*.json")):
            raise RuntimeError(
                "raw journal already exists; frozen prefilter is one-use"
            )
        planned: list[dict[str, object]] = []
        for index, raw in enumerate(request_plan, start=1):
            row = _mapping(raw, name=f"request plan row {index}")
            planned.append(
                {
                    "body_sha256": str(row["body_sha256"]),
                    "label": str(row["label"]),
                    "method": str(row["method"]),
                    "status": "planned",
                    "url": str(row["url"]),
                }
            )
        self.payload: dict[str, object] = {
            "contract_result_sha256": contract_hash,
            "created_at_ms": time.time_ns() // 1_000_000,
            "requests": planned,
            "status": "running",
        }
        root.mkdir(parents=True, exist_ok=True)
        self._persist()

    def _persist(self) -> None:
        write_bytes_atomic(
            self.path,
            (_canonical_json(self.payload) + "\n").encode("ascii"),
        )

    def start(self, *, label: str, url: str) -> tuple[int, dict[str, object]]:
        rows = _list(self.payload["requests"], name="journal requests")
        row = next(
            (
                _mapping(raw, name="journal request")
                for raw in rows
                if isinstance(raw, Mapping) and raw.get("label") == label
            ),
            None,
        )
        if row is None or row.get("status") != "planned":
            raise RuntimeError(f"request {label} is absent or already consumed")
        if row.get("method") != "GET" or row.get("url") != url:
            raise RuntimeError(f"request {label} differs from frozen plan")
        if row.get("body_sha256") != EMPTY_BODY_SHA256:
            raise RuntimeError(f"request {label} has a nonempty body hash")
        before_ms = time.time_ns() // 1_000_000
        row["requested_before_ms"] = before_ms
        row["status"] = "requesting"
        for index, raw in enumerate(rows):
            if isinstance(raw, Mapping) and raw.get("label") == label:
                rows[index] = row
                break
        self.payload["requests"] = rows
        self._persist()
        return before_ms, row

    def record(
        self,
        *,
        label: str,
        row: dict[str, object],
        response: requests.Response,
        requested_before_ms: int,
        received_after_ms: int,
    ) -> bytes:
        rows = _list(self.payload["requests"], name="journal requests")
        response_index = next(
            index
            for index, raw in enumerate(rows, start=1)
            if isinstance(raw, Mapping) and raw.get("label") == label
        )
        raw_path = self.root / f"response-{response_index:02d}-{label}.json"
        write_bytes_atomic(raw_path, response.content)
        row.update(
            {
                "http_status": response.status_code,
                "payload_sha256": _sha256(response.content),
                "raw_path": raw_path.as_posix(),
                "received_after_ms": received_after_ms,
                "request_elapsed_ms": received_after_ms - requested_before_ms,
                "response_url": response.url,
                "status": "received",
            }
        )
        rows[response_index - 1] = row
        self.payload["requests"] = rows
        self._persist()
        return response.content

    def fail_transport(self, *, label: str, row: dict[str, object], error: str) -> None:
        rows = _list(self.payload["requests"], name="journal requests")
        row.update(
            {
                "error": error,
                "failed_at_ms": time.time_ns() // 1_000_000,
                "status": "transport_failed",
            }
        )
        for index, raw in enumerate(rows):
            if isinstance(raw, Mapping) and raw.get("label") == label:
                rows[index] = row
                break
        self.payload["requests"] = rows
        self.payload["status"] = "failed"
        self._persist()

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
) -> object:
    before_ms, row = journal.start(label=label, url=url)
    try:
        response = session.get(url, timeout=30)
    except requests.RequestException as exc:
        journal.fail_transport(label=label, row=row, error=type(exc).__name__)
        raise
    after_ms = time.time_ns() // 1_000_000
    raw = journal.record(
        label=label,
        row=row,
        response=response,
        requested_before_ms=before_ms,
        received_after_ms=after_ms,
    )
    response.raise_for_status()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} did not return JSON; raw response retained") from exc


def analyze_snapshot(
    polymarket_instruments: object,
    polymarket_tickers: object,
    binance_exchange_info: object,
    binance_premium_index: object,
    *,
    prefilter_threshold_bips_per_8h: Decimal,
    maximum_followup_symbols: int,
) -> dict[str, object]:
    if maximum_followup_symbols < 1:
        raise ValueError("maximum follow-up symbols must be positive")

    polymarket_by_id: dict[int, dict[str, object]] = {}
    for raw in _list(polymarket_instruments, name="Polymarket instruments"):
        row = _mapping(raw, name="Polymarket instrument")
        if str(row.get("category") or "") not in TRADFI_CATEGORIES:
            continue
        instrument_id = _integer(
            row.get("instrument_id"), name="Polymarket instrument ID"
        )
        if instrument_id in polymarket_by_id:
            raise ValueError("duplicate Polymarket instrument ID")
        if str(row.get("funding_interval") or "") != "1h":
            raise ValueError("matched Polymarket TradFi funding interval is not 1h")
        polymarket_by_id[instrument_id] = row

    polymarket_ticker_by_id: dict[int, dict[str, object]] = {}
    for raw in _list(polymarket_tickers, name="Polymarket tickers"):
        row = _mapping(raw, name="Polymarket ticker")
        instrument_id = _integer(
            row.get("instrument_id"), name="Polymarket ticker instrument ID"
        )
        if instrument_id in polymarket_ticker_by_id:
            raise ValueError("duplicate Polymarket ticker instrument ID")
        polymarket_ticker_by_id[instrument_id] = row

    exchange_info = _mapping(binance_exchange_info, name="Binance exchange info")
    binance_symbols: dict[str, dict[str, object]] = {}
    for raw in _list(exchange_info.get("symbols"), name="Binance symbols"):
        row = _mapping(raw, name="Binance symbol")
        if row.get("status") != "TRADING":
            continue
        if row.get("contractType") != "TRADIFI_PERPETUAL":
            continue
        symbol = str(row.get("symbol") or "")
        if symbol in binance_symbols:
            raise ValueError("duplicate Binance TradFi symbol")
        binance_symbols[symbol] = row

    premium_by_symbol: dict[str, dict[str, object]] = {}
    for raw in _list(binance_premium_index, name="Binance premium index"):
        row = _mapping(raw, name="Binance premium row")
        symbol = str(row.get("symbol") or "")
        if symbol in premium_by_symbol:
            raise ValueError("duplicate Binance premium-index symbol")
        premium_by_symbol[symbol] = row

    rows: list[dict[str, object]] = []
    for instrument_id, instrument in polymarket_by_id.items():
        base_asset = str(instrument.get("base_asset") or "").upper()
        binance_symbol = f"{base_asset}USDT"
        binance_instrument = binance_symbols.get(binance_symbol)
        polymarket_ticker = polymarket_ticker_by_id.get(instrument_id)
        binance_ticker = premium_by_symbol.get(binance_symbol)
        if (
            binance_instrument is None
            or polymarket_ticker is None
            or binance_ticker is None
        ):
            continue

        polymarket_hourly = _decimal(
            polymarket_ticker.get("funding_rate"),
            name="Polymarket hourly funding rate",
        )
        binance_eight_hour = _decimal(
            binance_ticker.get("lastFundingRate"),
            name="Binance eight-hour funding rate",
        )
        polymarket_eight_hour = polymarket_hourly * Decimal("8")
        difference = polymarket_eight_hour - binance_eight_hour
        spread_bips = abs(difference) * Decimal("10000")
        rows.append(
            {
                "base_asset": base_asset,
                "binance_contract_type": binance_instrument.get("contractType"),
                "binance_funding_rate_8h": str(binance_eight_hour),
                "binance_next_funding_time_ms": _integer(
                    binance_ticker.get("nextFundingTime"),
                    name="Binance next funding time",
                ),
                "binance_symbol": binance_symbol,
                "binance_time_ms": _integer(
                    binance_ticker.get("time"), name="Binance premium time"
                ),
                "current_orientation": (
                    "short_polymarket_long_binance"
                    if difference > 0
                    else "long_polymarket_short_binance"
                ),
                "funding_spread_bips_per_8h": str(spread_bips),
                "polymarket_category": instrument.get("category"),
                "polymarket_funding_rate_1h": str(polymarket_hourly),
                "polymarket_funding_rate_8h_equivalent": str(polymarket_eight_hour),
                "polymarket_instrument_id": instrument_id,
                "polymarket_symbol": instrument.get("symbol"),
                "polymarket_timestamp_ms": _integer(
                    polymarket_ticker.get("timestamp"),
                    name="Polymarket ticker timestamp",
                ),
                "prefilter_pass": spread_bips >= prefilter_threshold_bips_per_8h,
            }
        )

    rows.sort(
        key=lambda row: (
            -Decimal(str(row["funding_spread_bips_per_8h"])),
            str(row["base_asset"]),
        )
    )
    passing = [row for row in rows if row["prefilter_pass"] is True]
    selected = [str(row["base_asset"]) for row in passing[:maximum_followup_symbols]]
    return {
        "exact_current_match_count": len(rows),
        "maximum_funding_spread_bips_per_8h": (
            rows[0]["funding_spread_bips_per_8h"] if rows else "0"
        ),
        "passing_count": len(passing),
        "prefilter_threshold_bips_per_8h": str(prefilter_threshold_bips_per_8h),
        "ranked_rows": rows,
        "selected_for_separate_history_contract": selected,
    }


def run(contract_path: Path, output_path: Path, raw_root: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_payload_hash(contract, "result_sha256")
    frozen_at = _parse_utc(contract.get("frozen_at_utc"), name="frozen_at_utc")
    if frozen_at > datetime.now(UTC):
        raise ValueError("frozen_at_utc is in the future")
    expected_implementation = str(contract.get("implementation_sha256") or "")
    if expected_implementation != _sha256(Path(__file__).read_bytes()):
        raise ValueError("implementation SHA-256 does not match frozen contract")

    request_contract = _mapping(
        contract.get("request_contract"), name="request contract"
    )
    expected_raw_root = Path(str(request_contract.get("raw_directory") or ""))
    if raw_root != expected_raw_root:
        raise ValueError("raw directory differs from frozen contract")
    plan = [
        _mapping(raw, name="request plan row")
        for raw in _list(request_contract.get("requests"), name="request plan")
    ]
    exact_plan = [
        {
            "body_sha256": EMPTY_BODY_SHA256,
            "label": "polymarket-instruments",
            "method": "GET",
            "url": POLYMARKET_INSTRUMENTS_URL,
        },
        {
            "body_sha256": EMPTY_BODY_SHA256,
            "label": "polymarket-tickers",
            "method": "GET",
            "url": POLYMARKET_TICKERS_URL,
        },
        {
            "body_sha256": EMPTY_BODY_SHA256,
            "label": "binance-exchange-info",
            "method": "GET",
            "url": BINANCE_EXCHANGE_INFO_URL,
        },
        {
            "body_sha256": EMPTY_BODY_SHA256,
            "label": "binance-premium-index",
            "method": "GET",
            "url": BINANCE_PREMIUM_INDEX_URL,
        },
    ]
    if plan != exact_plan:
        raise ValueError("request plan differs from the four frozen public GETs")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-tradfi-perps-funding-prefilter/1.0",
        }
    )
    journal = RawJournal(
        raw_root,
        contract_hash=contract_hash,
        request_plan=plan,
    )
    payloads = {
        row["label"]: _get_json(
            session,
            journal,
            label=str(row["label"]),
            url=str(row["url"]),
        )
        for row in plan
    }
    economics = _mapping(contract.get("economics"), name="economics")
    analysis = analyze_snapshot(
        payloads["polymarket-instruments"],
        payloads["polymarket-tickers"],
        payloads["binance-exchange-info"],
        payloads["binance-premium-index"],
        prefilter_threshold_bips_per_8h=Decimal(
            str(economics["prefilter_threshold_bips_per_8h"])
        ),
        maximum_followup_symbols=int(request_contract["maximum_followup_symbols"]),
    )

    result: dict[str, object] = {
        "schema_version": ("polymarket-binance-tradfi-perps-funding-prefilter-v1"),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "result_sha256": contract_hash,
        },
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "http_get_requests": 4,
            "orders_transfers_or_account_mutations": 0,
            "paper_or_live_trading_authority": False,
        },
        "economics": economics,
        "analysis": analysis,
        "adjudication": {
            "accepted_edge": False,
            "candidate_for_books": False,
            "candidate_for_separate_history_contract": bool(
                analysis["selected_for_separate_history_contract"]
            ),
            "deployment_ready": False,
            "status": (
                "current_funding_prefilter_has_history_candidates"
                if analysis["selected_for_separate_history_contract"]
                else "current_funding_prefilter_rejected_before_history"
            ),
        },
        "next_action": (
            "freeze_one_bounded_funding_history_contract_for_only_the_selected_"
            "symbols_without_books"
            if analysis["selected_for_separate_history_contract"]
            else "do_not_request_funding_history_or_books_without_a_material_"
            "funding_fee_session_or_instrument_change"
        ),
    }
    result_hash = _sha256(_canonical_json(result).encode("ascii"))
    result["result_sha256"] = result_hash
    write_bytes_atomic(
        output_path,
        (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
    )
    journal.complete(result_hash)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args.contract, args.output, args.raw_root)
    print(
        _canonical_json(
            {
                "accepted_edge": result["adjudication"]["accepted_edge"],
                "candidate_for_history": result["adjudication"][
                    "candidate_for_separate_history_contract"
                ],
                "exact_current_match_count": result["analysis"][
                    "exact_current_match_count"
                ],
                "passing_count": result["analysis"]["passing_count"],
                "result_sha256": result["result_sha256"],
                "selected": result["analysis"][
                    "selected_for_separate_history_contract"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
