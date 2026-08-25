"""Audit whether Binance options and quarterly futures share settlement values."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-option-future-settlement-equivalence-contract-v1.json"
)
EXERCISE_URL = "https://eapi.binance.com/eapi/v1/exerciseHistory"
MAX_RESPONSE_BYTES = 1_048_576
SCHEMA_VERSION = "binance-option-future-settlement-equivalence-audit-v1"
JOURNAL_SCHEMA_VERSION = "binance-option-future-settlement-equivalence-journal-v1"


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


def _normalized_sha256(path: Path) -> str:
    body = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256(body)


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


def _verified_json(
    path: Path,
    *,
    expected_result_hash: str | None = None,
    expected_file_hash: str | None = None,
) -> dict[str, object]:
    raw = path.read_bytes()
    if expected_file_hash is not None and _sha256(raw) != expected_file_hash:
        raise ValueError(f"source file hash differs: {path.name}")
    payload = _mapping(json.loads(raw), name=path.name)
    body = dict(payload)
    claimed = body.pop("result_sha256", None)
    computed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != computed or (
        expected_result_hash is not None and claimed != expected_result_hash
    ):
        raise ValueError(f"source result hash differs: {path.name}")
    return payload


def _load_contract() -> dict[str, object]:
    contract = _verified_json(CONTRACT_PATH)
    if contract.get("status") != "frozen_before_new_public_request":
        raise ValueError("settlement-equivalence contract is not frozen")
    implementation = _mapping(
        contract.get("implementation"), name="contract implementation"
    )
    if implementation.get("tool_sha256") != _normalized_sha256(Path(__file__)):
        raise ValueError("settlement-equivalence implementation hash differs")
    return contract


def _bound_source(
    contract: Mapping[str, object], source_name: str
) -> dict[str, object]:
    sources = _mapping(contract.get("source_binding"), name="source binding")
    source = _mapping(sources.get(source_name), name=f"{source_name} source")
    return _verified_json(
        ROOT / str(source["path"]),
        expected_result_hash=str(source["result_sha256"]),
        expected_file_hash=str(source["file_sha256"]),
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("execution git identity differs")
    return commit


def _require_clean_tracked_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("audit requires a clean tracked worktree")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    body = (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode("ascii")
    write_bytes_atomic(path, body)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("audit receipt paths must stay inside the repository") from exc


def _require_receipt_paths(
    contract: Mapping[str, object], *, output: Path, journal: Path
) -> None:
    receipt = _mapping(contract.get("receipt"), name="contract receipt")
    expected_output = (ROOT / str(receipt["result_path"])).resolve()
    expected_journal = (ROOT / str(receipt["journal_path"])).resolve()
    if output.resolve() != expected_output or journal.resolve() != expected_journal:
        raise ValueError("audit receipt paths differ from the frozen contract")


class _Journal:
    def __init__(
        self, *, path: Path, contract: Mapping[str, object], request_count: int
    ) -> None:
        if path.exists():
            raise RuntimeError("audit journal already exists; no rerun is permitted")
        self.path = path
        self.payload: dict[str, object] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "status": "active",
            "contract_result_sha256": contract["result_sha256"],
            "planned_request_count": request_count,
            "completed_request_count": 0,
            "events": [],
            "orders_submitted": False,
            "credentials_used": False,
            "adaptive_requests_used": False,
        }
        self._write()

    def _write(self) -> None:
        body = dict(self.payload)
        body.pop("journal_sha256", None)
        self.payload["journal_sha256"] = _sha256(_canonical_json(body).encode("ascii"))
        _write_json(self.path, self.payload)

    def event(self, phase: str, **fields: object) -> None:
        events = list(self.payload["events"])
        events.append({"phase": phase, **fields})
        self.payload["events"] = events
        self._write()

    def request_complete(self) -> None:
        self.payload["completed_request_count"] = (
            int(self.payload["completed_request_count"]) + 1
        )
        self._write()

    def complete(self, *, result_sha256: str) -> None:
        self.payload["status"] = "complete"
        self.payload["result_sha256"] = result_sha256
        self._write()

    def fail(self, error: Exception) -> None:
        self.payload["status"] = "terminal_failure_without_retry"
        self.payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write()


def _response_body(response: requests.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=65_536):
        if not chunk:
            continue
        size += len(chunk)
        chunks.append(chunk)
        if size > MAX_RESPONSE_BYTES:
            break
    return b"".join(chunks)


def _get(
    session: requests.Session,
    *,
    request: Mapping[str, object],
    journal: _Journal,
    timeout_seconds: float,
) -> list[object]:
    request_id = str(request["request_id"])
    params = _mapping(request.get("params"), name=f"{request_id} params")
    journal.event(
        "reserved_before_request",
        request_id=request_id,
        method="GET",
        url=EXERCISE_URL,
        params=params,
    )
    before_ms = time.time_ns() // 1_000_000
    try:
        response = session.get(
            EXERCISE_URL,
            params=params,
            timeout=timeout_seconds,
            stream=True,
            allow_redirects=False,
        )
        body = _response_body(response)
    except requests.RequestException as exc:
        raise RuntimeError("exercise-history request failed without retry") from exc
    after_ms = time.time_ns() // 1_000_000
    journal.event(
        "raw_response_persisted_before_validation",
        request_id=request_id,
        status_code=response.status_code,
        requested_before_ms=before_ms,
        received_after_ms=after_ms,
        elapsed_ms=after_ms - before_ms,
        response_url=response.url,
        content_type=response.headers.get("Content-Type"),
        retry_after=response.headers.get("Retry-After"),
        raw_response_size=len(body),
        raw_response_sha256=_sha256(body),
        raw_response_base64=base64.b64encode(body).decode("ascii"),
    )
    if response.is_redirect or 300 <= response.status_code < 400:
        raise RuntimeError("exercise-history redirect rejected without retry")
    if response.status_code == 429:
        raise RuntimeError("exercise-history rate limit reached without retry")
    if response.status_code != 200:
        raise RuntimeError(
            f"exercise-history HTTP {response.status_code} rejected without retry"
        )
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("exercise-history response exceeded the bounded size")
    try:
        payload = _list(json.loads(body), name=f"{request_id} response")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("exercise-history response was not valid JSON") from exc
    journal.event(
        "response_validated",
        request_id=request_id,
        row_count=len(payload),
        canonical_payload_sha256=_sha256(_canonical_json(payload).encode("ascii")),
    )
    journal.request_complete()
    return payload


def _delivery_prices(
    delivery_source: Mapping[str, object], *, dates: set[int]
) -> dict[tuple[str, int], Decimal]:
    source = _mapping(
        delivery_source.get("source_contract"), name="delivery source contract"
    )
    ledger = _list(source.get("request_ledger"), name="delivery request ledger")
    selected: dict[tuple[str, int], Decimal] = {}
    for raw_entry in ledger:
        entry = _mapping(raw_entry, name="delivery request")
        url = str(entry.get("url") or "")
        if "delivery-price?pair=" not in url:
            continue
        underlying = url.rsplit("=", 1)[-1]
        for raw_row in _list(entry.get("decoded_payload"), name="delivery payload"):
            row = _mapping(raw_row, name="delivery row")
            day_start_ms = int(row["deliveryTime"])
            if day_start_ms in dates:
                selected[(underlying, day_start_ms)] = _decimal(
                    row["deliveryPrice"], name="delivery price", positive=True
                )
    return selected


def _exercise_price(
    payload: list[object], *, underlying: str, day_start_ms: int
) -> tuple[Decimal, int, int]:
    if not 0 < len(payload) <= 100:
        raise ValueError("exercise-history response row count is invalid")
    prices: set[Decimal] = set()
    sides: set[str] = set()
    expected_expiry_ms = day_start_ms + 8 * 60 * 60 * 1000
    for raw_row in payload:
        row = _mapping(raw_row, name="exercise-history row")
        symbol = str(row.get("symbol") or "")
        if not symbol.startswith(underlying.removesuffix("USDT") + "-"):
            raise ValueError("exercise-history underlying differs")
        if int(row.get("expiryDate") or 0) != expected_expiry_ms:
            raise ValueError("exercise-history expiry is not the frozen 08:00 UTC")
        strike_result = str(row.get("strikeResult") or "")
        if strike_result not in {
            "REALISTIC_VALUE_STRICKEN",
            "EXTRINSIC_VALUE_EXPIRED",
        }:
            raise ValueError("exercise-history strike result differs")
        if symbol.endswith("-C"):
            sides.add("CALL")
        elif symbol.endswith("-P"):
            sides.add("PUT")
        else:
            raise ValueError("exercise-history option side differs")
        prices.add(
            _decimal(row.get("realStrikePrice"), name="exercise price", positive=True)
        )
    if len(prices) != 1 or sides != {"CALL", "PUT"}:
        raise ValueError(
            "exercise-history settlement value is not unique and two-sided"
        )
    return next(iter(prices)), len(payload), expected_expiry_ms


def _stale_discovery(
    option_snapshot: Mapping[str, object],
    quarterly_snapshot: Mapping[str, object],
) -> dict[str, object]:
    book_sources = {
        str(_mapping(raw, name="book source")["symbol"]): _mapping(
            raw, name="book source"
        )
        for raw in _list(
            _mapping(
                quarterly_snapshot.get("source_contract"),
                name="quarterly source contract",
            ).get("book_sources"),
            name="quarterly book sources",
        )
    }
    futures: dict[tuple[str, int], tuple[str, Decimal, Decimal]] = {}
    for raw_screen in _list(quarterly_snapshot.get("screens"), name="screens"):
        screen = _mapping(raw_screen, name="quarterly screen")
        source = book_sources[str(screen["symbol"])]
        book = _mapping(source.get("future_book"), name="future book")
        bids = _list(book.get("bids"), name="future bids")
        asks = _list(book.get("asks"), name="future asks")
        futures[(str(screen["pair"]), int(screen["delivery_time_ms"]))] = (
            str(screen["symbol"]),
            _decimal(_list(bids[0], name="future bid")[0], name="future bid"),
            _decimal(_list(asks[0], name="future ask")[0], name="future ask"),
        )
    chains: dict[tuple[str, int, Decimal], dict[str, dict[str, object]]] = defaultdict(
        dict
    )
    for raw_contract in _list(option_snapshot.get("contracts"), name="contracts"):
        contract = _mapping(raw_contract, name="option contract")
        key = (
            str(contract["underlying"]),
            int(contract["expiry_date_ms"]),
            _decimal(contract["strike"], name="strike", positive=True),
        )
        chains[key][str(contract["side"])] = contract
    candidates: list[dict[str, object]] = []
    pair_count = 0
    ticker_path_available_pair_count = 0
    for (underlying, expiry_ms, strike), sides in chains.items():
        future = futures.get((underlying, expiry_ms))
        if future is None or set(sides) != {"CALL", "PUT"}:
            continue
        pair_count += 1
        future_symbol, future_bid, future_ask = future
        call = sides["CALL"]
        put = sides["PUT"]
        raw_prices = {
            "call_bid": call.get("bid_price"),
            "call_ask": call.get("ask_price"),
            "put_bid": put.get("bid_price"),
            "put_ask": put.get("ask_price"),
        }
        prices = {
            name: None if value is None else _decimal(value, name=name)
            for name, value in raw_prices.items()
        }
        paths = {
            "synthetic_long_short_future": (
                None
                if prices["call_ask"] is None or prices["put_bid"] is None
                else future_bid + prices["put_bid"] - prices["call_ask"] - strike
            ),
            "synthetic_short_long_future": (
                None
                if prices["call_bid"] is None or prices["put_ask"] is None
                else strike - future_ask + prices["call_bid"] - prices["put_ask"]
            ),
        }
        if any(gross is not None for gross in paths.values()):
            ticker_path_available_pair_count += 1
        for mechanism, gross in paths.items():
            if gross is not None and gross > 0:
                candidates.append(
                    {
                        "mechanism": mechanism,
                        "underlying": underlying,
                        "expiry_ms": expiry_ms,
                        "strike": str(strike),
                        "call_symbol": call["symbol"],
                        "put_symbol": put["symbol"],
                        "future_symbol": future_symbol,
                        "gross_quote_per_base": str(gross),
                    }
                )
    candidates.sort(
        key=lambda candidate: (
            _decimal(candidate["gross_quote_per_base"], name="gross"),
            str(candidate["call_symbol"]),
        ),
        reverse=True,
    )
    return {
        "common_expiry_strike_pair_count": pair_count,
        "ticker_path_available_pair_count": ticker_path_available_pair_count,
        "gross_positive_ticker_combination_count": len(candidates),
        "best_candidates": candidates[:20],
        "synchronous": False,
        "execution_evidence": False,
    }


def _requests(contract: Mapping[str, object]) -> list[dict[str, object]]:
    requests_contract = _list(contract.get("requests"), name="contract requests")
    requests_list = [
        _mapping(raw, name="contract request") for raw in requests_contract
    ]
    if len(requests_list) != 8:
        raise ValueError("settlement-equivalence request count differs")
    if len({str(item.get("request_id")) for item in requests_list}) != 8:
        raise ValueError("settlement-equivalence request ids differ")
    return requests_list


def run(
    *,
    session: requests.Session,
    contract: Mapping[str, object],
    journal: _Journal,
    timeout_seconds: float,
) -> dict[str, object]:
    option_snapshot = _bound_source(contract, "option_snapshot")
    quarterly_snapshot = _bound_source(contract, "quarterly_snapshot")
    delivery_source = _bound_source(contract, "delivery_source")
    stale_discovery = _stale_discovery(option_snapshot, quarterly_snapshot)
    if stale_discovery["gross_positive_ticker_combination_count"] != 20:
        raise ValueError("frozen non-synchronous discovery no longer reconstructs")
    requests_list = _requests(contract)
    dates = {
        int(_mapping(item["params"], name="params")["startTime"])
        for item in requests_list
    }
    delivery_prices = _delivery_prices(delivery_source, dates=dates)
    observations: list[dict[str, object]] = []
    for request in requests_list:
        params = _mapping(request["params"], name="request params")
        underlying = str(params["underlying"])
        day_start_ms = int(params["startTime"])
        payload = _get(
            session,
            request=request,
            journal=journal,
            timeout_seconds=timeout_seconds,
        )
        option_price, row_count, option_expiry_ms = _exercise_price(
            payload,
            underlying=underlying,
            day_start_ms=day_start_ms,
        )
        future_price = delivery_prices.get((underlying, day_start_ms))
        if future_price is None:
            raise ValueError("frozen futures delivery price is absent")
        difference = option_price - future_price
        observations.append(
            {
                "request_id": request["request_id"],
                "underlying": underlying,
                "calendar_day_start_ms": day_start_ms,
                "option_expiry_ms": option_expiry_ms,
                "returned_option_row_count": row_count,
                "option_real_strike_price": str(option_price),
                "futures_delivery_price": str(future_price),
                "option_minus_futures": str(difference),
                "exactly_equal": difference == 0,
            }
        )
    exact_count = sum(bool(item["exactly_equal"]) for item in observations)
    all_equal = exact_count == len(observations)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "test_option_future_settlement_benchmark_equivalence_before_book_refresh",
        "source_contract": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_result_sha256": contract["result_sha256"],
            "execution_git_commit": _git_commit(),
            "journal_path": _repo_relative(journal.path),
            "journal_sha256_before_completion": journal.payload["journal_sha256"],
        },
        "mechanism": {
            "synthetic_long_short_future_profit": "F_bid+P_bid-C_ask-K",
            "synthetic_short_long_future_profit": "K-F_ask+C_bid-P_ask",
            "fixed_only_if_option_and_future_settlement_values_are_identical": True,
        },
        "non_synchronous_discovery": stale_discovery,
        "settlement_observations": observations,
        "verdict": {
            "observation_count": len(observations),
            "exact_equality_count": exact_count,
            "historical_exact_equality_passed": all_equal,
            "accepted_edge": False,
            "trading_authority": False,
            "status": (
                "historical_settlement_equivalence_supports_synchronized_depth_screen"
                if all_equal
                else "settlement_benchmark_mismatch_rejects_fixed_payoff_claim"
            ),
        },
        "limitations": [
            "historical calendar-date equality is not current executable depth",
            "the option ticker and quarterly book snapshots are non-synchronous",
            "ticker prices have no displayed quantity",
            "fees margin liquidation atomicity capacity and operational costs are unresolved",
            "the historical futures deliveryTime field is used only as a calendar-date marker",
        ],
        "safety": {
            "new_public_get_count": len(requests_list),
            "credentials_used": False,
            "orders_submitted": False,
            "retries": 0,
            "adaptive_requests": 0,
            "profitability_claim": False,
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Binance option/future settlement audit once."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _repo_relative(arguments.output)
    _repo_relative(arguments.journal)
    if arguments.output.resolve() == arguments.journal.resolve():
        raise ValueError("audit output and journal paths must be distinct")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        raise ValueError("audit timeout must be a positive finite number")
    contract = _load_contract()
    _require_receipt_paths(
        contract,
        output=arguments.output,
        journal=arguments.journal,
    )
    if arguments.output.exists():
        raise RuntimeError("audit output already exists; no rerun is permitted")
    _require_clean_tracked_worktree()
    requests_list = _requests(contract)
    journal = _Journal(
        path=arguments.journal,
        contract=contract,
        request_count=len(requests_list),
    )
    try:
        with requests.Session() as session:
            result = run(
                session=session,
                contract=contract,
                journal=journal,
                timeout_seconds=arguments.timeout_seconds,
            )
        _write_json(arguments.output, result)
        journal.complete(result_sha256=str(result["result_sha256"]))
    except Exception as exc:
        journal.fail(exc)
        raise
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "exact_equality_count": result["verdict"]["exact_equality_count"],
                "observation_count": result["verdict"]["observation_count"],
                "orders_submitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
