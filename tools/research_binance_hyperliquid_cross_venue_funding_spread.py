"""Extend the archived Binance-Hyperliquid static funding-spread study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping
import zipfile

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/action-value/binance-hyperliquid-cross-venue-funding-spread-extension-contract-v1-2026-08-27.json"
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


class JournalSession:
    def __init__(
        self, *, journal_dir: Path, maximum_requests: int, session: requests.Session | None = None
    ) -> None:
        self.journal_dir = journal_dir
        self.maximum_requests = maximum_requests
        self.session = session or requests.Session()
        self.sources: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        source_name: str,
        params: Mapping[str, object] | None = None,
        json_body: object | None = None,
        expect_json: bool = True,
    ) -> object:
        if len(self.sources) >= self.maximum_requests:
            raise ValueError("frozen request ceiling reached")
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        before_ms = datetime.now(timezone.utc).timestamp() * 1000
        intent = {
            "method": method,
            "url": url,
            "params": params,
            "json_body_sha256": (
                None
                if json_body is None
                else _sha256(_canonical(json_body).encode("ascii"))
            ),
            "requested_before_ms": int(before_ms),
        }
        write_bytes_atomic(
            self.journal_dir / f"{source_name}.intent.json",
            (_canonical(intent) + "\n").encode("ascii"),
        )
        response = self.session.request(
            method, url, params=params, json=json_body, timeout=30
        )
        after_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        write_bytes_atomic(
            self.journal_dir / f"{source_name}.raw", response.content
        )
        metadata = {
            "role": source_name,
            "method": method,
            "status_code": response.status_code,
            "final_url": response.url,
            "payload_bytes": len(response.content),
            "payload_sha256": _sha256(response.content),
            "requested_before_ms": int(before_ms),
            "received_after_ms": after_ms,
            "elapsed_ms": after_ms - int(before_ms),
        }
        write_bytes_atomic(
            self.journal_dir / f"{source_name}.response.json",
            (_canonical(metadata) + "\n").encode("ascii"),
        )
        self.sources.append(metadata)
        if response.status_code == 429:
            raise RuntimeError("public source rate limited; stopped without retry")
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise ValueError("public response exceeded frozen byte ceiling")
        return response.json() if expect_json else response.content


def _load_contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body).encode("ascii")):
        raise ValueError("contract embedded hash does not reconstruct")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _zip_csv_rows(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, str]]:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive path identity failed for {suffix}")
    raw = archive.read(matches[0]).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def _archive_year_aprs(archive: zipfile.ZipFile, asset: str) -> dict[str, str]:
    hyper_rows = _zip_csv_rows(archive, f"/data/{asset.lower()}_funding.csv")
    binance_rows = _zip_csv_rows(
        archive, f"/data/{asset.lower()}_binance_funding.csv"
    )
    hyper_daily: defaultdict[date, Decimal] = defaultdict(Decimal)
    for row in hyper_rows:
        observed = datetime.fromtimestamp(
            int(row["time_ms"]) / 1000, tz=timezone.utc
        ).date()
        hyper_daily[observed] += _decimal(row["funding_rate"], name="archive HL rate")
    binance_daily = {
        date.fromisoformat(row["date"]): _decimal(row["rate"], name="archive Binance rate")
        for row in binance_rows
    }
    result: dict[str, str] = {}
    for year in (2024, 2025, 2026):
        values = [
            hyper_daily[day] - binance_daily[day]
            for day in sorted(set(hyper_daily) & set(binance_daily))
            if day.year == year
        ]
        if not values:
            raise ValueError(f"archive has no overlap for {asset} {year}")
        result[str(year)] = str(sum(values, Decimal(0)) / len(values) * Decimal(36500))
    return result


def _fetch_hyperliquid(
    journal: JournalSession, asset: str, start_ms: int, end_ms: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = start_ms
    page = 1
    while cursor < end_ms:
        raw = journal.request(
            "POST",
            "https://api.hyperliquid.xyz/info",
            source_name=f"hl-{asset.lower()}-funding-{page:02d}",
            json_body={
                "type": "fundingHistory",
                "coin": asset,
                "startTime": cursor,
                "endTime": end_ms - 1,
            },
        )
        batch = [
            _mapping(value, name="Hyperliquid funding row")
            for value in _list(raw, name="Hyperliquid funding response")
        ]
        if not batch:
            break
        rows.extend(batch)
        last = max(int(row["time"]) for row in batch)
        if last < cursor:
            raise ValueError("Hyperliquid funding pagination did not advance")
        cursor = last + 1
        page += 1
        if len(batch) < 500:
            break
    return rows


def _fetch_binance_funding(
    journal: JournalSession, asset: str, start_ms: int, end_ms: int
) -> list[dict[str, object]]:
    raw = journal.request(
        "GET",
        "https://fapi.binance.com/fapi/v1/fundingRate",
        source_name=f"binance-{asset.lower()}-funding",
        params={
            "symbol": f"{asset}USDT",
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": 1000,
        },
    )
    return [
        _mapping(value, name="Binance funding row")
        for value in _list(raw, name="Binance funding response")
    ]


def _fetch_binance_premiums(
    journal: JournalSession, asset: str, start_ms: int, end_ms: int
) -> list[list[object]]:
    rows: list[list[object]] = []
    cursor = start_ms
    page = 1
    while cursor < end_ms:
        raw = journal.request(
            "GET",
            "https://fapi.binance.com/fapi/v1/premiumIndexKlines",
            source_name=f"binance-{asset.lower()}-premium-{page:02d}",
            params={
                "symbol": f"{asset}USDT",
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1500,
            },
        )
        batch = [
            _list(value, name="Binance premium kline")
            for value in _list(raw, name="Binance premium response")
        ]
        if not batch:
            break
        rows.extend(batch)
        last = max(int(row[0]) for row in batch)
        if last < cursor:
            raise ValueError("Binance premium pagination did not advance")
        cursor = last + 3_600_000
        page += 1
        if len(batch) < 1500:
            break
    return rows


def _asset_extension(
    *,
    asset: str,
    hyper_rows: list[dict[str, object]],
    binance_funding_rows: list[dict[str, object]],
    binance_premium_rows: list[list[object]],
    start_ms: int,
    end_ms: int,
    elapsed_days: int,
    cost: Decimal,
) -> dict[str, object]:
    hyper_by_time: dict[int, dict[str, object]] = {}
    hyper_daily: defaultdict[date, Decimal] = defaultdict(Decimal)
    for row in hyper_rows:
        if str(row.get("coin")) != asset:
            raise ValueError(f"Hyperliquid returned wrong asset for {asset}")
        timestamp = int(row["time"])
        if not start_ms <= timestamp < end_ms or timestamp in hyper_by_time:
            raise ValueError(f"Hyperliquid timestamp identity failed for {asset}")
        hyper_by_time[timestamp] = row
        day = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date()
        hyper_daily[day] += _decimal(row.get("fundingRate"), name="HL funding rate")
    binance_daily: defaultdict[date, Decimal] = defaultdict(Decimal)
    funding_times: set[int] = set()
    for row in binance_funding_rows:
        if str(row.get("symbol")) != f"{asset}USDT":
            raise ValueError(f"Binance returned wrong funding asset for {asset}")
        timestamp = int(row["fundingTime"])
        if not start_ms <= timestamp < end_ms or timestamp in funding_times:
            raise ValueError(f"Binance funding timestamp identity failed for {asset}")
        funding_times.add(timestamp)
        day = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date()
        binance_daily[day] += _decimal(
            row.get("fundingRate"), name="Binance funding rate"
        )
    binance_premium_by_hour: dict[int, Decimal] = {}
    for row in binance_premium_rows:
        if len(row) < 5:
            raise ValueError("Binance premium kline is truncated")
        timestamp = int(row[0])
        if not start_ms <= timestamp < end_ms or timestamp in binance_premium_by_hour:
            raise ValueError(f"Binance premium timestamp identity failed for {asset}")
        binance_premium_by_hour[timestamp // 3_600_000] = _decimal(
            row[4], name="Binance premium close"
        )
    hyper_premium_by_hour = {
        timestamp // 3_600_000: _decimal(row.get("premium"), name="HL premium")
        for timestamp, row in hyper_by_time.items()
    }
    premium_hours = sorted(set(hyper_premium_by_hour) & set(binance_premium_by_hour))
    if not premium_hours:
        raise ValueError(f"no synchronized premium observations for {asset}")
    entry_hour, exit_hour = premium_hours[0], premium_hours[-1]
    entry_basis = hyper_premium_by_hour[entry_hour] - binance_premium_by_hour[entry_hour]
    exit_basis = hyper_premium_by_hour[exit_hour] - binance_premium_by_hour[exit_hour]
    basis_pnl = entry_basis - exit_basis
    days = sorted(set(hyper_daily) & set(binance_daily))
    daily_spreads = [hyper_daily[day] - binance_daily[day] for day in days]
    gross_funding = sum(daily_spreads, Decimal(0))
    basis_inclusive = gross_funding + basis_pnl
    after_cost = basis_inclusive - cost
    annualizer = Decimal(365) / Decimal(elapsed_days)
    expected_hours = elapsed_days * 24
    return {
        "asset": asset,
        "coverage": {
            "elapsed_days": elapsed_days,
            "expected_Hyperliquid_hourly_rows": expected_hours,
            "Hyperliquid_rows": len(hyper_by_time),
            "Hyperliquid_hourly_coverage_fraction": str(
                Decimal(len(hyper_by_time)) / Decimal(expected_hours)
            ),
            "Binance_funding_events": len(funding_times),
            "Binance_premium_hourly_rows": len(binance_premium_by_hour),
            "synchronized_premium_hours": len(premium_hours),
            "overlap_days": len(days),
        },
        "extension_economics": {
            "gross_funding_spread_return_fraction": str(gross_funding),
            "gross_funding_spread_APR_percent": str(
                gross_funding * annualizer * Decimal(100)
            ),
            "entry_basis_fraction": str(entry_basis),
            "exit_basis_fraction": str(exit_basis),
            "basis_pnl_fraction": str(basis_pnl),
            "basis_inclusive_return_fraction": str(basis_inclusive),
            "frozen_round_trip_cost_fraction": str(cost),
            "after_cost_return_fraction": str(after_cost),
            "after_cost_APR_percent": str(after_cost * annualizer * Decimal(100)),
            "positive_daily_funding_spread_fraction": str(
                Decimal(sum(value > 0 for value in daily_spreads))
                / Decimal(len(daily_spreads))
            ),
        },
    }


def _latest_dgs3mo(raw: bytes, *, end_date: date) -> tuple[date, Decimal]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    candidates: list[tuple[date, Decimal]] = []
    for row in rows:
        raw_date = row.get("observation_date") or row.get("DATE")
        raw_value = row.get("DGS3MO")
        if not raw_date or raw_value in (None, "", "."):
            continue
        observed = date.fromisoformat(raw_date)
        if observed <= end_date:
            candidates.append((observed, _decimal(raw_value, name="DGS3MO")))
    if not candidates:
        raise ValueError("no DGS3MO observation before frozen end")
    return max(candidates, key=lambda item: item[0])


def run(*, archive_path: Path, journal_dir: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _load_contract()
    expected_md5 = str(_mapping(contract["prior_evidence"], name="prior evidence")["archive_md5"])
    actual_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()  # noqa: S324 - source identity, not security
    if actual_md5 != expected_md5:
        raise ValueError("replication archive MD5 does not match Zenodo record")
    extension = _mapping(contract["extension"], name="extension")
    start = datetime.fromisoformat(
        str(extension["start_utc_inclusive"]).replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        str(extension["end_utc_exclusive"]).replace("Z", "+00:00")
    )
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    elapsed_days = (end - start).days
    assets = [str(value) for value in _list(extension["assets"], name="assets")]
    request_contract = _mapping(contract["request_contract"], name="request contract")
    journal = JournalSession(
        journal_dir=journal_dir,
        maximum_requests=int(request_contract["maximum_public_requests"]),
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive_aprs = {asset: _archive_year_aprs(archive, asset) for asset in assets}
    cost = _decimal(
        _mapping(contract["cost_and_hurdle"], name="cost and hurdle")[
            "frozen_total_round_trip_cost_bps"
        ],
        name="round trip bps",
    ) / Decimal(10000)
    results: list[dict[str, object]] = []
    for asset in assets:
        hyper_rows = _fetch_hyperliquid(journal, asset, start_ms, end_ms)
        binance_funding = _fetch_binance_funding(journal, asset, start_ms, end_ms)
        binance_premiums = _fetch_binance_premiums(journal, asset, start_ms, end_ms)
        result = _asset_extension(
            asset=asset,
            hyper_rows=hyper_rows,
            binance_funding_rows=binance_funding,
            binance_premium_rows=binance_premiums,
            start_ms=start_ms,
            end_ms=end_ms,
            elapsed_days=elapsed_days,
            cost=cost,
        )
        result["archive_gross_funding_spread_APR_percent"] = archive_aprs[asset]
        results.append(result)
    fred_raw = journal.request(
        "GET",
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        source_name="fred-DGS3MO",
        params={"id": "DGS3MO"},
        expect_json=False,
    )
    if not isinstance(fred_raw, bytes):
        raise ValueError("FRED response was not bytes")
    hurdle_date, hurdle_percent = _latest_dgs3mo(
        fred_raw, end_date=end.date()
    )
    by_asset = {str(row["asset"]): row for row in results}
    primary = [str(value) for value in _list(extension["primary_assets"], name="primary assets")]
    primary_after_cost_aprs = [
        _decimal(
            _mapping(by_asset[asset]["extension_economics"], name="economics")[
                "after_cost_APR_percent"
            ],
            name="after cost APR",
        )
        for asset in primary
    ]
    basket_apr = sum(primary_after_cost_aprs, Decimal(0)) / Decimal(len(primary))
    archive_positive = all(
        _decimal(value, name="archive APR") > 0
        for asset in primary
        for value in _mapping(
            by_asset[asset]["archive_gross_funding_spread_APR_percent"],
            name="archive APRs",
        ).values()
    )
    coverage_passed = all(
        _decimal(
            _mapping(by_asset[asset]["coverage"], name="coverage")[
                "Hyperliquid_hourly_coverage_fraction"
            ],
            name="coverage fraction",
        )
        >= Decimal("0.95")
        for asset in primary
    )
    after_cost_positive = all(value > 0 for value in primary_after_cost_aprs)
    hurdle_passed = all(value > hurdle_percent for value in primary_after_cost_aprs) and basket_apr > hurdle_percent
    candidate = archive_positive and coverage_passed and after_cost_positive and hurdle_passed
    artifact: dict[str, object] = {
        "schema_version": "binance-hyperliquid-cross-venue-funding-spread-extension-v1",
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "archive": {
            "path_name": archive_path.name,
            "md5": actual_md5,
            "doi": _mapping(contract["prior_evidence"], name="prior evidence")["doi"],
        },
        "extension": {
            "start_utc_inclusive": extension["start_utc_inclusive"],
            "end_utc_exclusive": extension["end_utc_exclusive"],
            "elapsed_days": elapsed_days,
            "asset_results": results,
            "primary_equal_weight_after_cost_APR_percent": str(basket_apr),
        },
        "hurdle": {
            "series": "DGS3MO",
            "observation_date": hurdle_date.isoformat(),
            "percent": str(hurdle_percent),
        },
        "gates": {
            "archive_primary_each_year_positive": archive_positive,
            "extension_primary_coverage_passed": coverage_passed,
            "extension_primary_after_cost_positive": after_cost_positive,
            "extension_primary_and_basket_exceed_DGS3MO": hurdle_passed,
        },
        "verdict": {
            "status": (
                "accepted_scoped_public_candidate_not_deployment_ready"
                if candidate
                else "rejected_without_refitting_or_resampling"
            ),
            "accepted_scoped_public_candidate": candidate,
            "deployment_ready": False,
            "account_realized_profit_claim": False,
            "public_profit_floor_USD": "0",
        },
        "authority": {
            "public_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "public_request_count": len(journal.sources),
            "public_requests": journal.sources,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "Public rates and premiums do not prove account fees margin eligibility or collateral economics.",
            "Equal-notional opposite perps retain cross-venue basis liquidation oracle operational and venue-credit risk.",
            "The extension is one frozen out-of-sample window and future funding is not guaranteed.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(archive_path=args.archive, journal_dir=args.journal_dir)
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "raw_responses_retained_before_validation": True,
        }
        write_bytes_atomic(
            args.journal_dir / "terminal-failure.json",
            (_canonical(failure) + "\n").encode("ascii"),
        )
        raise
    write_bytes_atomic(args.output, (_canonical(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
