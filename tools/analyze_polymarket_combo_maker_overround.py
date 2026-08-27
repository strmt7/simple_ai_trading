#!/usr/bin/env python3
"""Collect and adjudicate the frozen Polymarket Combo maker-overround study."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import requests

from simple_ai_trading.storage import write_json_atomic


USER_AGENT = "simple-ai-trading-public-combo-research/1.0"
RESOLVED_STATUSES = "RESOLVED_WIN,RESOLVED_PARTIAL,RESOLVED_LOSS"
PROJECTION_FIELDS = (
    "cohort",
    "user_address",
    "combo_condition_id",
    "combo_position_id",
    "side",
    "status",
    "first_entry_at",
    "resolved_at",
    "entry_avg_price_usdc",
    "gross_entry_cost_usdc",
    "entry_fees_usdc",
    "realized_payout_usdc",
    "total_cost_usdc",
    "shares_balance",
    "legs_total",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    return parser


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.pop(hash_field)).lower()
    actual = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    if claimed != actual:
        raise ValueError(f"{path.name} {hash_field} mismatch: {claimed} != {actual}")
    payload[hash_field] = claimed
    return payload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal {field}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite decimal {field}: {value!r}")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def _prepare_url(base_url: str, params: dict[str, object]) -> str:
    request = requests.Request("GET", base_url, params=params)
    return request.prepare().url or base_url


def _get_raw(url: str) -> tuple[bytes, dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(3):
        requested_at = _utc_now()
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain"},
                timeout=30,
            )
            body = response.content
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
                    continue
            response.raise_for_status()
            record = {
                "url": url,
                "requested_at_utc": requested_at,
                "received_at_utc": _utc_now(),
                "status_code": response.status_code,
                "http_date": response.headers.get("Date"),
                "body_bytes": len(body),
                "body_sha256": _sha256_bytes(body),
                "body_base64": base64.b64encode(body).decode("ascii"),
            }
            return body, record
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
    raise RuntimeError(f"GET failed after bounded retries: {url}: {last_error}")


def _get(url: str) -> tuple[dict[str, Any] | list[Any], dict[str, object]]:
    body, record = _get_raw(url)
    try:
        return json.loads(body), record
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"response was not JSON: {url}") from exc


def _leaderboard_pages(
    base_url: str,
    *,
    offsets: Iterable[int],
    order_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, object]] = []
    for offset in offsets:
        url = _prepare_url(
            base_url,
            {
                "category": "SPORTS",
                "timePeriod": "ALL",
                "orderBy": order_by,
                "limit": 50,
                "offset": offset,
            },
        )
        payload, record = _get(url)
        if not isinstance(payload, list):
            raise ValueError(f"leaderboard response was not a list: {url}")
        records.append(record)
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"leaderboard row was not an object: {url}")
            wallet = str(row.get("proxyWallet", "")).lower()
            if len(wallet) != 42 or not wallet.startswith("0x"):
                raise ValueError(f"invalid leaderboard wallet: {wallet!r}")
            rows.append({**row, "proxyWallet": wallet, "source_order_by": order_by})
    return rows, records


def _fetch_wallet_positions(
    wallet: str,
    cohort: str,
    base_url: str,
    maximum_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, object]] = []
    for offset in range(0, maximum_offset + 1, 1000):
        url = _prepare_url(
            base_url,
            {
                "user": wallet,
                "status": RESOLVED_STATUSES,
                "limit": 1000,
                "offset": offset,
            },
        )
        payload, record = _get(url)
        records.append(record)
        if not isinstance(payload, dict) or not isinstance(payload.get("combos"), list):
            raise ValueError(f"invalid Combo position response: {url}")
        for row in payload["combos"]:
            if not isinstance(row, dict):
                raise ValueError(f"invalid Combo position row: {url}")
            rows.append({**row, "cohort": cohort})
        pagination = payload.get("pagination")
        if not isinstance(pagination, dict):
            raise ValueError(f"missing Combo pagination object: {url}")
        if not bool(pagination.get("has_more")):
            return rows, records
    raise RuntimeError(f"Combo pagination exceeded frozen maximum_offset for {wallet}")


def _write_raw_archive(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
            for record in records:
                compressed.write(_canonical_json(record).encode("ascii") + b"\n")


def _project_row(row: dict[str, Any]) -> dict[str, object]:
    return {field: row.get(field) for field in PROJECTION_FIELDS}


def _write_projection(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=PROJECTION_FIELDS)
                writer.writeheader()
                writer.writerows(_project_row(row) for row in rows)


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = (
            str(row.get("user_address", "")).lower(),
            str(row.get("combo_condition_id", "")),
            str(row.get("combo_position_id", "")),
        )
        if not all(key):
            raise ValueError(f"missing Combo deduplication key: {key}")
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = row
            continue
        duplicate_count += 1
        if _project_row(prior) != _project_row(row):
            raise ValueError(f"conflicting duplicate Combo position: {key}")
    return list(by_key.values()), duplicate_count


def _economics(row: dict[str, Any]) -> dict[str, Any]:
    gross = _decimal(row.get("gross_entry_cost_usdc"), field="gross_entry_cost_usdc")
    fees = _decimal(row.get("entry_fees_usdc"), field="entry_fees_usdc")
    payout = _decimal(row.get("realized_payout_usdc"), field="realized_payout_usdc")
    if gross <= 0 or fees < 0 or fees > gross or payout < 0:
        raise ValueError("invalid Combo economics row")
    net_basis = gross - fees
    return {
        **row,
        "gross": gross,
        "fees": fees,
        "payout": payout,
        "buyer_net": payout - gross,
        "seller_proxy": net_basis - payout,
        "entry_date": str(row.get("first_entry_at", ""))[:10],
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, object]:
    gross = sum((row["gross"] for row in rows), Decimal(0))
    fees = sum((row["fees"] for row in rows), Decimal(0))
    payout = sum((row["payout"] for row in rows), Decimal(0))
    buyer_net = payout - gross
    seller_proxy = gross - fees - payout
    return {
        "position_count": len(rows),
        "wallet_count": len({str(row["user_address"]).lower() for row in rows}),
        "first_entry_utc_date_count": len({str(row["entry_date"]) for row in rows}),
        "gross_entry_cost_pUSD": _decimal_text(gross),
        "attributed_buyer_fees_pUSD": _decimal_text(fees),
        "realized_payout_pUSD": _decimal_text(payout),
        "buyer_net_pnl_pUSD": _decimal_text(buyer_net),
        "buyer_return_on_gross_cost": (
            _decimal_text(buyer_net / gross) if gross else None
        ),
        "opposite_side_gross_spread_proxy_pUSD": _decimal_text(seller_proxy),
        "opposite_side_proxy_on_net_basis": (
            _decimal_text(seller_proxy / (gross - fees)) if gross > fees else None
        ),
    }


def _cluster_lower_bound(
    rows: list[dict[str, Any]],
    *,
    key: str,
    seed: int,
    repetitions: int,
) -> Decimal | None:
    totals: defaultdict[str, Decimal] = defaultdict(Decimal)
    for row in rows:
        cluster = str(row[key]).lower() if key == "user_address" else str(row[key])
        totals[cluster] += row["seller_proxy"]
    values = list(totals.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        samples.append(sum((rng.choice(values) for _ in values), Decimal(0)))
    samples.sort()
    index = max(0, math.ceil(repetitions * 0.025) - 1)
    return samples[index]


def _date_roles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dates = sorted({str(row["entry_date"]) for row in rows})
    train_end = math.floor(len(dates) * 0.60)
    validation_end = math.floor(len(dates) * 0.80)
    role_dates = {
        "training": set(dates[:train_end]),
        "validation": set(dates[train_end:validation_end]),
        "test": set(dates[validation_end:]),
    }
    return {
        role: [row for row in rows if str(row["entry_date"]) in selected]
        for role, selected in role_dates.items()
    }


def _positive_wallet_dominance(rows: list[dict[str, Any]]) -> Decimal | None:
    totals: defaultdict[str, Decimal] = defaultdict(Decimal)
    for row in rows:
        totals[str(row["user_address"]).lower()] += row["seller_proxy"]
    positive = [value for value in totals.values() if value > 0]
    if not positive:
        return None
    return max(positive) / sum(positive, Decimal(0))


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    args = _parser().parse_args()
    if args.threads < 1 or args.threads > 24:
        raise ValueError("--threads must be between 1 and 24")
    contract = _load_hashed_json(args.contract, "contract_sha256")
    validation = contract["unseen_validation_cohort"]
    leaderboard_url = str(validation["leaderboard_url"])
    positions_url = str(validation["positions_url"])
    max_offset = int(validation["position_parameters"]["maximum_offset"])

    discovery_offsets = contract["discovery_boundary_already_observed"]["leaderboard_query"][
        "offsets"
    ]
    validation_offsets = validation["leaderboard_parameters"]["offsets"]
    all_records: list[dict[str, object]] = []
    discovery_wallets: set[str] = set()
    validation_rows: dict[str, list[dict[str, Any]]] = {}
    for order_by in ("PNL", "VOL"):
        discovery, records = _leaderboard_pages(
            leaderboard_url, offsets=discovery_offsets, order_by=order_by
        )
        all_records.extend(records)
        discovery_wallets.update(str(row["proxyWallet"]) for row in discovery)
        cohort, records = _leaderboard_pages(
            leaderboard_url, offsets=validation_offsets, order_by=order_by
        )
        all_records.extend(records)
        validation_rows[order_by] = cohort

    pnl_wallets = [
        str(row["proxyWallet"])
        for row in validation_rows["PNL"]
        if str(row["proxyWallet"]) not in discovery_wallets
    ]
    pnl_wallets = list(dict.fromkeys(pnl_wallets))
    pnl_set = set(pnl_wallets)
    vol_wallets = [
        str(row["proxyWallet"])
        for row in validation_rows["VOL"]
        if str(row["proxyWallet"]) not in discovery_wallets
        and str(row["proxyWallet"]) not in pnl_set
    ]
    vol_wallets = list(dict.fromkeys(vol_wallets))
    assignments = [(wallet, "PNL") for wallet in pnl_wallets] + [
        (wallet, "VOL_only") for wallet in vol_wallets
    ]

    def collect(assignment: tuple[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
        return _fetch_wallet_positions(
            assignment[0], assignment[1], positions_url, max_offset
        )

    collected_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for rows, records in pool.map(collect, assignments):
            collected_rows.extend(rows)
            all_records.extend(records)

    deduplicated, duplicate_count = _deduplicate(collected_rows)
    yes_rows = []
    for row in deduplicated:
        if str(row.get("status")) not in RESOLVED_STATUSES.split(","):
            raise ValueError(f"unexpected position status: {row.get('status')!r}")
        if str(row.get("side")) == "YES":
            yes_rows.append(_economics(row))
    yes_rows.sort(
        key=lambda row: (
            str(row["first_entry_at"]),
            str(row["user_address"]).lower(),
            str(row["combo_condition_id"]),
            str(row["combo_position_id"]),
        )
    )

    _write_raw_archive(args.raw_archive, all_records)
    _write_projection(args.projection, deduplicated)

    overall = _metrics(yes_rows)
    cohorts = {
        cohort: _metrics([row for row in yes_rows if row["cohort"] == cohort])
        for cohort in ("PNL", "VOL_only")
    }
    roles = {role: _metrics(rows) for role, rows in _date_roles(yes_rows).items()}
    repetitions = int(contract["robustness"]["cluster_bootstrap_repetitions"])
    seed = int(contract["robustness"]["cluster_bootstrap_seed"])
    wallet_lower = _cluster_lower_bound(
        yes_rows,
        key="user_address",
        seed=seed,
        repetitions=repetitions,
    )
    date_lower = _cluster_lower_bound(
        yes_rows,
        key="entry_date",
        seed=seed + 1,
        repetitions=repetitions,
    )
    dominance = _positive_wallet_dominance(yes_rows)
    minimums = contract["robustness"]
    counts_pass = (
        int(overall["wallet_count"])
        >= int(minimums["minimum_unique_validation_wallets_with_resolved_yes"])
        and int(overall["position_count"])
        >= int(minimums["minimum_resolved_yes_positions"])
        and int(overall["first_entry_utc_date_count"])
        >= int(minimums["minimum_unique_first_entry_utc_dates"])
    )
    slices_positive = all(
        Decimal(str(metrics["opposite_side_gross_spread_proxy_pUSD"])) > 0
        for metrics in [*cohorts.values(), *roles.values()]
    )
    bootstrap_pass = (
        wallet_lower is not None
        and wallet_lower > 0
        and date_lower is not None
        and date_lower > 0
    )
    dominance_pass = (
        dominance is not None
        and dominance <= Decimal(str(minimums["maximum_single_wallet_share_of_positive_proxy"]))
    )
    overall_positive = Decimal(str(overall["opposite_side_gross_spread_proxy_pUSD"])) > 0
    passes = counts_pass and slices_positive and bootstrap_pass and dominance_pass and overall_positive

    source_documents = []
    for url in contract["source_contracts"]:
        _payload, record = _get_raw(str(url))
        all_records.append(record)
        source_documents.append(
            {
                "url": url,
                "payload_bytes": record["body_bytes"],
                "payload_sha256": record["body_sha256"],
                "http_date": record["http_date"],
            }
        )
    # Rewrite once so the source documents are included in the retained archive.
    _write_raw_archive(args.raw_archive, all_records)

    repository_root = Path(__file__).resolve().parents[1]
    result: dict[str, object] = {
        "schema_version": "polymarket-combo-maker-overround-validation-v1",
        "created_at_utc": _utc_now(),
        "contract": {
            "path": _portable_path(args.contract, repository_root),
            "contract_sha256": contract["contract_sha256"],
            "frozen_before_validation_access": True,
        },
        "authority": {
            "public_unauthenticated_requests_only": True,
            "credentials_present_or_used": False,
            "authenticated_requests": 0,
            "quote_requests_or_acceptances": 0,
            "orders_transfers_or_account_reads": 0,
        },
        "collection": {
            "discovery_wallet_count_excluded": len(discovery_wallets),
            "validation_pnl_wallet_count": len(pnl_wallets),
            "validation_vol_only_wallet_count": len(vol_wallets),
            "validation_wallet_count": len(assignments),
            "wallets_with_any_resolved_combo_position": len(
                {str(row["user_address"]).lower() for row in deduplicated}
            ),
            "resolved_combo_position_count_all_sides": len(deduplicated),
            "resolved_yes_position_count": len(yes_rows),
            "duplicate_rows_removed": duplicate_count,
            "explicit_status_filter": RESOLVED_STATUSES,
            "http_request_count": len(all_records),
            "http_status_counts": dict(
                sorted(Counter(str(record["status_code"]) for record in all_records).items())
            ),
            "raw_archive": {
                "path": _portable_path(args.raw_archive, repository_root),
                "bytes": args.raw_archive.stat().st_size,
                "sha256": _sha256(args.raw_archive),
            },
            "repository_projection": {
                "path": _portable_path(args.projection, repository_root),
                "bytes": args.projection.stat().st_size,
                "sha256": _sha256(args.projection),
            },
        },
        "economics": {
            "overall": overall,
            "cohorts": cohorts,
            "chronological_roles": roles,
            "wallet_cluster_bootstrap_lower_2_5pct_pUSD": (
                _decimal_text(wallet_lower) if wallet_lower is not None else None
            ),
            "date_cluster_bootstrap_lower_2_5pct_pUSD": (
                _decimal_text(date_lower) if date_lower is not None else None
            ),
            "maximum_single_wallet_share_of_positive_proxy": (
                _decimal_text(dominance) if dominance is not None else None
            ),
            "gross_proxy_only": True,
            "maker_counterparty_or_after_cost_profit_proved": False,
        },
        "gate": {
            "minimum_counts_pass": counts_pass,
            "all_required_cohorts_and_roles_positive": slices_positive,
            "cluster_bootstrap_pass": bootstrap_pass,
            "wallet_dominance_pass": dominance_pass,
            "overall_proxy_positive": overall_positive,
            "passes_unaccepted_candidate_gate": passes,
        },
        "source_documents": source_documents,
        "limitations": [
            "The leaderboard cohort is selected on all-time sports PnL or volume and is not a random sample of Combo requesters.",
            "A requester position does not identify the maker counterparty or prove that one maker captured the opposite economics.",
            "The opposite-side proxy omits seller fees, quote competition, hedges, collateral, funding, capital return latency, Last Look declines, inventory transforms, and operational costs.",
            "Historical resolved positions do not prove a forward executable quote or recurring after-cost profit.",
        ],
        "verdict": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "market_direction_forecast_required": False,
            "status": (
                "high_priority_unaccepted_combo_market_maker_overround_candidate"
                if passes
                else "broad_public_combo_maker_overround_validation_failed"
            ),
            "next_action": (
                "obtain approved maker access and a direct quote ledger under separate authority then measure exact net receive hedges collateral occupancy and forward paper recurrence"
                if passes
                else "do_not_repeat_leaderboard_mining_without_a_less_selected_population_or_direct_maker_quote_ledger"
            ),
        },
        "source_code_sha256": _sha256(Path(__file__)),
    }
    result["result_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("ascii")
    ).hexdigest()
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
