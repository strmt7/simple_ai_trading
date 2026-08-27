"""Run the frozen current seven-asset Polymarket 5-minute TWAP reward screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import screen_polymarket_crypto_twap_liquidity_rewards as prior
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/crypto-twap-5m-liquidity-reward-screen-contract-v1-2026-08-27.json"
)


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_contract(now: datetime) -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body).encode("ascii")):
        raise ValueError("contract embedded hash does not reconstruct")
    capture = prior._mapping(contract.get("capture"), name="capture")
    lower = datetime.fromisoformat(
        str(capture["activation_not_before_utc"]).replace("Z", "+00:00")
    )
    upper = datetime.fromisoformat(
        str(capture["activation_not_after_utc"]).replace("Z", "+00:00")
    )
    if not lower <= now <= upper:
        raise ValueError(f"outside frozen activation window: {now.isoformat()}")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _request_document(
    session: requests.Session, *, journal_dir: Path
) -> dict[str, object]:
    source_name = "01-official-liquidity-rewards-documentation"
    url = "https://docs.polymarket.com/programs/liquidity-rewards"
    before_ms = time.time_ns() // 1_000_000
    journal_dir.mkdir(parents=True, exist_ok=True)
    intent = {
        "method": "GET",
        "url": url,
        "requested_before_ms": before_ms,
    }
    write_bytes_atomic(
        journal_dir / f"{source_name}.intent.json",
        (_canonical(intent) + "\n").encode("ascii"),
    )
    response = session.get(url, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    write_bytes_atomic(journal_dir / f"{source_name}.raw", response.content)
    metadata = {
        "status_code": response.status_code,
        "final_url": response.url,
        "payload_bytes": len(response.content),
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "elapsed_ms": after_ms - before_ms,
    }
    write_bytes_atomic(
        journal_dir / f"{source_name}.response.json",
        (_canonical(metadata) + "\n").encode("ascii"),
    )
    if response.status_code == 429:
        raise RuntimeError("Polymarket rate limit reached; stopped without retry")
    response.raise_for_status()
    if len(response.content) > prior.MAX_RESPONSE_BYTES:
        raise ValueError("documentation response exceeded the frozen byte ceiling")
    text = response.content.decode(response.encoding or "utf-8", errors="replace")
    required_markers = (
        "5-Minute Markets",
        "550k",
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "HYPE",
        "BNB",
        "DOGE",
    )
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise ValueError(f"official five-minute allocation markers missing: {missing}")
    return metadata


def _active_gamma_reward(
    market: Mapping[str, object], *, capture_date: str
) -> dict[str, object]:
    gamma = prior._mapping(market["gamma_payload"], name="Gamma payload")
    rows = [
        prior._mapping(value, name="Gamma clob reward")
        for value in prior._json_list(gamma.get("clobRewards"), name="clobRewards")
    ]
    active: list[dict[str, object]] = []
    observed_date = datetime.fromisoformat(capture_date).date()
    condition_id = str(market["condition_id"]).lower()
    for row in rows:
        if str(row.get("conditionId") or "").lower() != condition_id:
            raise ValueError("Gamma embedded reward condition identity changed")
        start_date = datetime.fromisoformat(str(row.get("startDate"))).date()
        raw_end = row.get("endDate")
        end_date = (
            None if raw_end in (None, "") else datetime.fromisoformat(str(raw_end)).date()
        )
        if start_date <= observed_date and (end_date is None or observed_date <= end_date):
            active.append(row)
    daily_rate = sum(
        (
            prior._decimal(
                row.get("rewardsDailyRate"), name="Gamma rewards daily rate", positive=True
            )
            for row in active
        ),
        Decimal("0"),
    )
    if not active or daily_rate <= 0:
        raise ValueError("no positive exact active Gamma reward allocation")
    minimum_size = prior._decimal(
        gamma.get("rewardsMinSize"), name="reward minimum size", positive=True
    )
    maximum_spread = prior._decimal(
        gamma.get("rewardsMaxSpread"), name="reward maximum spread", positive=True
    )
    if minimum_size < prior._decimal(
        market["minimum_order_size"], name="minimum order size"
    ):
        raise ValueError("reward size is below the executable order minimum")
    return {
        "minimum_size": minimum_size,
        "maximum_spread": maximum_spread,
        "daily_rate": daily_rate,
        "active": active,
    }


def run(*, journal_dir: Path, session: requests.Session | None = None) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _load_contract(started)
    capture = prior._mapping(contract["capture"], name="capture")
    expected: dict[str, tuple[str, int, int]] = {}
    for value in prior._list(capture["markets"], name="markets"):
        row = prior._mapping(value, name="market contract")
        expected[str(row["slug"])] = (
            str(row["asset"]),
            int(row["start_epoch_seconds"]) * 1000,
            int(capture["duration_ms"]),
        )
    http = session or requests.Session()
    documentation_source = _request_document(http, journal_dir=journal_dir)
    gamma_raw, gamma_source = prior._request(
        http,
        "GET",
        "https://gamma-api.polymarket.com/markets",
        journal_dir=journal_dir,
        source_name="02-gamma-seven-markets",
        params=[*(('slug', slug) for slug in expected), ("closed", "false")],
    )
    markets = prior._market_rows(gamma_raw, expected)
    rewards = {
        slug: _active_gamma_reward(market, capture_date=started.date().isoformat())
        for slug, market in markets.items()
    }
    tokens = [
        str(token)
        for market in markets.values()
        for token in prior._list(market["tokens"], name="tokens")
    ]
    books_raw, books_source = prior._request(
        http,
        "POST",
        "https://clob.polymarket.com/books",
        journal_dir=journal_dir,
        source_name="03-fourteen-token-books",
        json_body=[{"token_id": token} for token in tokens],
    )
    received_ms = int(books_source["received_after_ms"])
    books = {
        str(prior._mapping(value, name="book")["asset_id"]): prior._mapping(
            value, name="book"
        )
        for value in prior._list(books_raw, name="books")
    }
    if set(books) != set(tokens) or len(books) != len(tokens):
        raise ValueError("batch books did not return the fourteen exact tokens")
    timestamps: list[int] = []
    for token, book in books.items():
        market = next(value for value in markets.values() if token in value["tokens"])
        if prior._decimal(
            book.get("tick_size"), name="book tick", positive=True
        ) != market["tick_size"] or prior._decimal(
            book.get("min_order_size"), name="book min", positive=True
        ) != market["minimum_order_size"]:
            raise ValueError("book execution parameters disagree with Gamma")
        prior._levels(book, "bids")
        prior._levels(book, "asks")
        timestamps.append(int(str(book["timestamp"])))
    receipt_span = received_ms - int(documentation_source["requested_before_ms"])
    age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    remaining = min(int(market["end_ms"]) for market in markets.values()) - received_ms
    fresh = (
        0 <= age <= int(capture["book_max_event_age_ms"])
        and skew <= int(capture["book_max_timestamp_skew_ms"])
        and receipt_span <= int(capture["receipt_max_span_ms"])
        and remaining >= int(capture["minimum_remaining_market_ms"])
    )
    results: list[dict[str, object]] = []
    for slug, market in markets.items():
        reward = rewards[slug]
        results.append(
            {
                "slug": slug,
                "asset": market["asset"],
                "condition_id": market["condition_id"],
                "tokens": market["tokens"],
                "fee_schedule": market["fee_schedule"],
                "reward": {
                    "minimum_size": str(reward["minimum_size"]),
                    "maximum_spread_cents": str(reward["maximum_spread"]),
                    "daily_rate_pUSD": str(reward["daily_rate"]),
                    "active_configurations": reward["active"],
                },
                "diagnostic": prior._diagnostic(market, reward, books),
            }
        )
    candidates = (
        [
            str(row["slug"])
            for row in results
            if prior._mapping(row["diagnostic"], name="diagnostic")["status"]
            == "prospective_capture_candidate_not_an_edge"
        ]
        if fresh
        else []
    )
    artifact: dict[str, object] = {
        "schema_version": "polymarket-crypto-twap-5m-liquidity-reward-screen-v1",
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "capture": {
            "receipt_span_ms": receipt_span,
            "oldest_book_age_ms": age,
            "book_timestamp_skew_ms": skew,
            "remaining_market_ms": remaining,
            "freshness_passed": fresh,
        },
        "market_results": results,
        "verdict": {
            "status": (
                "prospective_capture_design_eligible_not_an_edge"
                if candidates
                else "rejected_without_resampling"
            ),
            "candidate_slugs": candidates,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "publicly_proven_reward_payout_floor_pUSD": "0",
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
            "documentation_request": documentation_source,
            "gamma_request": gamma_source,
            "books_request": books_source,
            "gamma_payload": gamma_raw,
            "books_payload": books_raw,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "A public top-midpoint score is conditional because maker identities queue order persistence random sampling and final epoch normalization are unavailable.",
            "A one-sided fill is directional orphan exposure.",
            "A one-time public snapshot cannot prove realized reward after-cost recurrence.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(journal_dir=args.journal_dir)
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
