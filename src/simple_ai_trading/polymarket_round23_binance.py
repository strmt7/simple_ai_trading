"""Credential-free, in-memory Binance archive ingestion for Round 23."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import hashlib
from io import BytesIO, StringIO, TextIOWrapper
import json
from pathlib import Path
import re
import time
from typing import Protocol
from urllib.parse import urlparse
from zipfile import BadZipFile, ZipFile

import requests

from .polymarket_round22_pilot import Round22PilotStore


POLYMARKET_ROUND23_BINANCE_SOURCE_RELATIVE = (
    "docs/model-research/polymarket/round-023-binance-lead-lag-source-v2.json"
)
POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256 = (
    "9275cfb5cb95427ee9565e3196c87b1c91ce114f713eae2561ad8f6144ddc6f0"
)
_POLYMARKET_ROUND23_BINANCE_SOURCE_V1_RELATIVE = (
    "docs/model-research/polymarket/round-023-binance-lead-lag-source-v1.json"
)
_POLYMARKET_ROUND23_BINANCE_SOURCE_V1_SHA256 = (
    "3424105cc6374a623f9a2944c38a777d19ecc5f50444b2444827dce85e6a454a"
)
_POLYMARKET_ROUND23_QUALIFICATION_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-023-source-qualification-attempt1-2026-08-03.json"
)
_POLYMARKET_ROUND23_QUALIFICATION_SHA256 = (
    "c5790ee5580fe9d9b43fbb9319d32c729a99ce25d10f39f10c7477790cedef56"
)
POLYMARKET_ROUND23_BINANCE_SCHEMA_VERSION = "polymarket-round23-binance-second-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_CONTRACT_BYTES = 1 * 1024 * 1024
_RETRYABLE_STATUS_CODES = frozenset((425, 429, 500, 502, 503, 504))
_PUBLIC_SESSION_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "connection",
        "user-agent",
    }
)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str


class _Session(Protocol):
    headers: Mapping[str, str]
    cookies: object

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> _Response: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 23 source JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 23 source JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _decimal(value: str, *, name: str, positive: bool = False) -> Decimal:
    try:
        selected = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Round 23 {name} is invalid") from exc
    if not selected.is_finite() or (selected <= 0 if positive else selected < 0):
        raise ValueError(f"Round 23 {name} is outside the bound")
    return selected


def _signed_decimal(value: str, *, name: str) -> Decimal:
    try:
        selected = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Round 23 {name} is invalid") from exc
    if not selected.is_finite():
        raise ValueError(f"Round 23 {name} is outside the bound")
    return selected


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validated_bar(
    *,
    source: str,
    date: str,
    timestamp_ms: int,
    open_price: str,
    high_price: str,
    low_price: str,
    close_price: str,
    base_volume: str,
    quote_volume: str,
    signed_quote_volume: str,
    trade_count: int,
) -> Round23SecondBar:
    if source not in {"spot_1s", "futures_aggTrades"}:
        raise ValueError("Round 23 Binance bar source differs")
    try:
        expected_day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("Round 23 Binance bar date differs") from exc
    day_start_ms = int(expected_day.timestamp() * 1_000)
    if (
        timestamp_ms % 1_000 != 0
        or not day_start_ms <= timestamp_ms < day_start_ms + 86_400_000
        or type(trade_count) is not int
        or trade_count < 0
    ):
        raise ValueError("Round 23 Binance bar identity differs")
    open_value = _decimal(open_price, name="bar open", positive=True)
    high_value = _decimal(high_price, name="bar high", positive=True)
    low_value = _decimal(low_price, name="bar low", positive=True)
    close_value = _decimal(close_price, name="bar close", positive=True)
    base_value = _decimal(base_volume, name="bar base volume")
    quote_value = _decimal(quote_volume, name="bar quote volume")
    signed_value = _signed_decimal(
        signed_quote_volume,
        name="bar signed quote volume",
    )
    if (
        low_value > min(open_value, close_value)
        or high_value < max(open_value, close_value)
        or high_value < low_value
        or abs(signed_value) > quote_value
        or (trade_count == 0 and (base_value != 0 or quote_value != 0))
        or (trade_count > 0 and (base_value <= 0 or quote_value <= 0))
    ):
        raise ValueError("Round 23 Binance bar economics differ")
    return Round23SecondBar(
        source=source,
        date=date,
        timestamp_ms=timestamp_ms,
        open_price=_decimal_text(open_value),
        high_price=_decimal_text(high_value),
        low_price=_decimal_text(low_value),
        close_price=_decimal_text(close_value),
        base_volume=_decimal_text(base_value),
        quote_volume=_decimal_text(quote_value),
        signed_quote_volume=_decimal_text(signed_value),
        trade_count=trade_count,
    )


def _load_contract_object(path: Path, *, name: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_CONTRACT_BYTES
    ):
        raise ValueError(f"Round 23 {name} is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 23 {name} is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Round 23 {name} is not an object")
    return dict(decoded)


def load_round23_binance_source_contract(repository: str | Path) -> dict[str, object]:
    root = Path(repository).resolve()
    contract = _load_contract_object(
        root / _POLYMARKET_ROUND23_BINANCE_SOURCE_V1_RELATIVE,
        name="Binance source v1 contract",
    )
    claimed = str(contract.pop("source_contract_sha256", "")).strip().lower()
    archives = contract.get("archives")
    authority = contract.get("authority")
    if (
        claimed != _POLYMARKET_ROUND23_BINANCE_SOURCE_V1_SHA256
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != "polymarket-round23-binance-lead-lag-source-v1"
        or contract.get("status")
        != "frozen_before_archive_download_or_condition_level_label_join"
        or not isinstance(archives, list)
        or len(archives) != 6
        or not isinstance(authority, Mapping)
        or set(authority)
        != {
            "binance_account_api",
            "binance_execution",
            "live_trading",
            "polymarket_authentication",
            "polymarket_execution",
            "profitability_claim",
        }
        or any(value is not False for value in authority.values())
    ):
        raise ValueError("Round 23 Binance source contract differs")
    identities: set[tuple[str, str]] = set()
    for archive in archives:
        if not isinstance(archive, Mapping):
            raise ValueError("Round 23 Binance archive contract is malformed")
        identity = (str(archive.get("source")), str(archive.get("date")))
        if (
            identity in identities
            or identity[0] not in {"spot_1s", "futures_aggTrades"}
            or _SHA256.fullmatch(str(archive.get("sha256"))) is None
            or int(archive.get("compressed_bytes", 0)) <= 0
        ):
            raise ValueError("Round 23 Binance archive contract differs")
        identities.add(identity)
    if identities != {
        (source, date)
        for source in ("spot_1s", "futures_aggTrades")
        for date in ("2026-04-27", "2026-06-15", "2026-07-06")
    }:
        raise ValueError("Round 23 Binance archive population differs")
    qualification = _load_contract_object(
        root / _POLYMARKET_ROUND23_QUALIFICATION_RELATIVE,
        name="source qualification evidence",
    )
    if (
        _canonical_sha256(qualification) != _POLYMARKET_ROUND23_QUALIFICATION_SHA256
        or qualification.get("status") != "immutable_failed_assumption_evidence"
        or qualification.get("parent_source_contract_sha256") != claimed
    ):
        raise ValueError("Round 23 source qualification evidence differs")
    revision = _load_contract_object(
        root / POLYMARKET_ROUND23_BINANCE_SOURCE_RELATIVE,
        name="Binance source v2 contract",
    )
    revision_claim = str(revision.pop("source_contract_sha256", "")).strip().lower()
    revision_authority = revision.get("authority")
    policy = revision.get("futures_no_trade_second_policy")
    parents = revision.get("parents")
    if (
        revision_claim != POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256
        or revision_claim != _canonical_sha256(revision)
        or revision.get("schema_version")
        != "polymarket-round23-binance-lead-lag-source-v2"
        or revision.get("status")
        != "frozen_after_first_source_qualification_before_remaining_archive_download_feature_construction_or_label_join"
        or revision_authority != authority
        or not isinstance(parents, Mapping)
        or parents.get("source_contract_v1_sha256") != claimed
        or parents.get("source_qualification_attempt1_sha256")
        != _POLYMARKET_ROUND23_QUALIFICATION_SHA256
        or not isinstance(policy, Mapping)
        or policy.get("trigger")
        != "no_aggregate_trade_timestamp_in_a_checksum_verified_source_second"
        or policy.get("leading_no_trade_second")
        != "fail_closed_without_a_prior_same_archive_trade"
        or policy.get("trade_count") != 0
        or policy.get("quote_or_book_claim") is not False
    ):
        raise ValueError("Round 23 Binance source v2 contract differs")
    return {
        **contract,
        "futures_no_trade_second_policy": dict(policy),
        "schema_version": revision["schema_version"],
        "source_contract_sha256": revision_claim,
        "source_contract_v1_sha256": claimed,
        "status": revision["status"],
    }


@dataclass(frozen=True, slots=True)
class Round23SecondBar:
    source: str
    date: str
    timestamp_ms: int
    open_price: str
    high_price: str
    low_price: str
    close_price: str
    base_volume: str
    quote_volume: str
    signed_quote_volume: str
    trade_count: int

    def body(self) -> dict[str, object]:
        return {
            "base_volume": self.base_volume,
            "close_price": self.close_price,
            "date": self.date,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "open_price": self.open_price,
            "quote_volume": self.quote_volume,
            "schema_version": POLYMARKET_ROUND23_BINANCE_SCHEMA_VERSION,
            "signed_quote_volume": self.signed_quote_volume,
            "source": self.source,
            "timestamp_ms": self.timestamp_ms,
            "trade_count": self.trade_count,
        }


@dataclass(frozen=True, slots=True)
class Round23ArchiveResult:
    source: str
    date: str
    row_count: int
    first_timestamp_ms: int
    last_timestamp_ms: int
    archive_sha256: str
    row_chain_sha256: str
    downloaded: bool
    archive_persisted: bool = False
    authentication_used: bool = False
    binance_execution_authority: bool = False
    polymarket_execution_authority: bool = False


class Round23BinanceArchiveClient:
    """Public exact-URL client that holds each bounded archive only in memory."""

    def __init__(
        self,
        *,
        session: _Session | None = None,
        timeout_seconds: float = 60.0,
        maximum_attempts: int = 4,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = max(5.0, min(120.0, float(timeout_seconds)))
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 8:
            raise ValueError("Round 23 Binance retry bound differs")
        self.maximum_attempts = maximum_attempts
        self.sleeper = sleeper or time.sleep

    def _assert_public_session(self) -> None:
        headers = {str(name).strip().lower() for name in self.session.headers}
        cookies = getattr(self.session, "cookies", None)
        if not headers <= _PUBLIC_SESSION_HEADERS or (
            cookies is not None and len(cookies)
        ):
            raise ValueError("Round 23 Binance session contains authority")

    def _request(self, url: str, *, maximum_bytes: int) -> bytes:
        self._assert_public_session()
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc.lower() != "data.binance.vision"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Round 23 Binance URL is outside the source contract")
        for attempt in range(self.maximum_attempts):
            try:
                response = self.session.get(
                    url,
                    headers={
                        "Accept": "application/zip, text/plain",
                        "User-Agent": "simple-ai-trading-round23-public-data/0.1",
                    },
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt + 1 == self.maximum_attempts:
                    raise ValueError(
                        "Round 23 Binance transport retries exhausted"
                    ) from exc
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
                continue
            status = int(response.status_code)
            if status == 200:
                response_url = urlparse(str(response.url))
                content = bytes(response.content)
                if (
                    response_url != parsed
                    or not 1 <= len(content) <= maximum_bytes
                    or getattr(self.session, "cookies", None)
                    and len(self.session.cookies)
                ):
                    raise ValueError("Round 23 Binance response differs")
                return content
            if (
                status not in _RETRYABLE_STATUS_CODES
                or attempt + 1 == self.maximum_attempts
            ):
                raise ValueError(f"Round 23 Binance request failed with HTTP {status}")
            self.sleeper(min(30.0, 0.5 * (2**attempt)))
        raise AssertionError("unreachable Round 23 Binance retry state")

    def archive(self, archive: Mapping[str, object], *, maximum_bytes: int) -> bytes:
        url = str(archive["url"])
        checksum = (
            self._request(url + ".CHECKSUM", maximum_bytes=512).decode("ascii").strip()
        )
        expected_checksum = f"{archive['sha256']}  {archive['filename']}"
        if checksum != expected_checksum:
            raise ValueError("Round 23 official Binance checksum changed")
        content = self._request(url, maximum_bytes=maximum_bytes)
        if (
            len(content) != int(archive["compressed_bytes"])
            or hashlib.sha256(content).hexdigest() != archive["sha256"]
        ):
            raise ValueError("Round 23 Binance archive bytes changed")
        return content


def _row_chain(previous: str, bar: Round23SecondBar) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(bar.body()))
    ).hexdigest()


def _spot_bar(row: list[str], *, date: str) -> Round23SecondBar:
    if len(row) != 12:
        raise ValueError("Round 23 spot row width differs")
    raw_timestamp = int(row[0])
    if raw_timestamp < 1_000_000_000_000_000 or raw_timestamp % 1_000 != 0:
        raise ValueError("Round 23 spot timestamp is not exact microseconds")
    open_price = _decimal(row[1], name="spot open", positive=True)
    high_price = _decimal(row[2], name="spot high", positive=True)
    low_price = _decimal(row[3], name="spot low", positive=True)
    close_price = _decimal(row[4], name="spot close", positive=True)
    base_volume = _decimal(row[5], name="spot base volume")
    quote_volume = _decimal(row[7], name="spot quote volume")
    trade_count = int(row[8])
    taker_buy_quote = _decimal(row[10], name="spot taker buy quote")
    if (
        low_price > min(open_price, close_price)
        or high_price < max(open_price, close_price)
        or taker_buy_quote > quote_volume
        or trade_count < 0
    ):
        raise ValueError("Round 23 spot bar economics differ")
    return _validated_bar(
        source="spot_1s",
        date=date,
        timestamp_ms=raw_timestamp // 1_000,
        open_price=_decimal_text(open_price),
        high_price=_decimal_text(high_price),
        low_price=_decimal_text(low_price),
        close_price=_decimal_text(close_price),
        base_volume=_decimal_text(base_volume),
        quote_volume=_decimal_text(quote_volume),
        signed_quote_volume=_decimal_text(2 * taker_buy_quote - quote_volume),
        trade_count=trade_count,
    )


def _parse_archive(
    content: bytes,
    archive: Mapping[str, object],
    *,
    start_ms: int,
    end_ms: int,
    maximum_uncompressed_bytes: int,
) -> tuple[Round23SecondBar, ...]:
    expected_member = str(archive["filename"]).removesuffix(".zip") + ".csv"
    try:
        zip_file = ZipFile(BytesIO(content))
    except BadZipFile as exc:
        raise ValueError("Round 23 Binance archive is not ZIP") from exc
    with zip_file:
        members = zip_file.infolist()
        if (
            len(members) != 1
            or members[0].filename != expected_member
            or members[0].is_dir()
            or not 1 <= members[0].file_size <= maximum_uncompressed_bytes
        ):
            raise ValueError("Round 23 Binance ZIP member differs")
        with zip_file.open(members[0], "r") as raw:
            reader = csv.reader(TextIOWrapper(raw, encoding="ascii", newline=""))
            if archive["source"] == "spot_1s":
                bars: list[Round23SecondBar] = []
                for row in reader:
                    if row and row[0] == "open_time":
                        continue
                    bar = _spot_bar(row, date=str(archive["date"]))
                    if bar.timestamp_ms >= end_ms:
                        break
                    if bar.timestamp_ms >= start_ms:
                        bars.append(bar)
            else:
                bars = _parse_futures_rows(
                    reader,
                    date=str(archive["date"]),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
    if (
        not bars
        or bars[0].timestamp_ms != start_ms
        or bars[-1].timestamp_ms != end_ms - 1_000
        or any(
            current.timestamp_ms != prior.timestamp_ms + 1_000
            for prior, current in zip(bars, bars[1:], strict=False)
        )
    ):
        raise ValueError("Round 23 Binance retained second coverage differs")
    return tuple(bars)


def _parse_futures_rows(
    reader: Iterable[list[str]],
    *,
    date: str,
    start_ms: int,
    end_ms: int,
) -> list[Round23SecondBar]:
    aggregates: dict[int, dict[str, object]] = {}
    previous_trade_id = -1
    previous_timestamp = -1
    prior_trade_price: Decimal | None = None
    for raw_row in reader:
        row = list(raw_row)
        if row and row[0] == "agg_trade_id":
            continue
        if len(row) != 7:
            raise ValueError("Round 23 futures row width differs")
        trade_id = int(row[0])
        timestamp = int(row[5])
        if trade_id <= previous_trade_id or timestamp < previous_timestamp:
            raise ValueError("Round 23 futures archive ordering differs")
        previous_trade_id = trade_id
        previous_timestamp = timestamp
        price = _decimal(row[1], name="futures price", positive=True)
        if timestamp >= end_ms:
            break
        if timestamp < start_ms:
            prior_trade_price = price
            continue
        quantity = _decimal(row[2], name="futures quantity")
        first_trade = int(row[3])
        last_trade = int(row[4])
        if (
            first_trade < 0
            or last_trade < first_trade
            or row[6] not in {"true", "false"}
        ):
            raise ValueError("Round 23 futures aggregate trade differs")
        second = timestamp - (timestamp % 1_000)
        quote = price * quantity
        aggregate = aggregates.setdefault(
            second,
            {
                "base": Decimal("0"),
                "close": price,
                "high": price,
                "low": price,
                "open": price,
                "quote": Decimal("0"),
                "signed": Decimal("0"),
                "trades": 0,
            },
        )
        aggregate["close"] = price
        aggregate["high"] = max(aggregate["high"], price)
        aggregate["low"] = min(aggregate["low"], price)
        aggregate["base"] += quantity
        aggregate["quote"] += quote
        aggregate["signed"] += -quote if row[6] == "true" else quote
        aggregate["trades"] += last_trade - first_trade + 1
    output: list[Round23SecondBar] = []
    for timestamp in range(start_ms, end_ms, 1_000):
        aggregate = aggregates.get(timestamp)
        if aggregate is None:
            if prior_trade_price is None:
                raise ValueError("Round 23 futures window has no leading trade price")
            output.append(
                _validated_bar(
                    source="futures_aggTrades",
                    date=date,
                    timestamp_ms=timestamp,
                    open_price=_decimal_text(prior_trade_price),
                    high_price=_decimal_text(prior_trade_price),
                    low_price=_decimal_text(prior_trade_price),
                    close_price=_decimal_text(prior_trade_price),
                    base_volume="0",
                    quote_volume="0",
                    signed_quote_volume="0",
                    trade_count=0,
                )
            )
            continue
        prior_trade_price = aggregate["close"]
        output.append(
            _validated_bar(
                source="futures_aggTrades",
                date=date,
                timestamp_ms=timestamp,
                open_price=_decimal_text(aggregate["open"]),
                high_price=_decimal_text(aggregate["high"]),
                low_price=_decimal_text(aggregate["low"]),
                close_price=_decimal_text(aggregate["close"]),
                base_volume=_decimal_text(aggregate["base"]),
                quote_volume=_decimal_text(aggregate["quote"]),
                signed_quote_volume=_decimal_text(aggregate["signed"]),
                trade_count=int(aggregate["trades"]),
            )
        )
    return output


def _window(store: Round22PilotStore, date: str) -> tuple[int, int]:
    day_start = int(
        datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1_000
    )
    rows = store.connection.execute(
        """
        SELECT event_start_ms, event_end_ms FROM feature.market_identity
        WHERE event_start_ms >= ? AND event_start_ms < ?
        ORDER BY event_start_ms
        """,
        [day_start, day_start + 86_400_000],
    ).fetchall()
    if (
        len(rows) != 12
        or int(rows[0][0]) != day_start
        or int(rows[-1][1]) != day_start + 3_600_000
        or any(
            int(current[0]) != int(prior[1])
            for prior, current in zip(rows, rows[1:], strict=False)
        )
    ):
        raise ValueError("Round 23 Polymarket date window differs")
    return day_start, day_start + 3_600_000


def _archive_evidence(
    *,
    archive_sha256: str,
    first_timestamp_ms: int,
    last_timestamp_ms: int,
    row_chain_sha256: str,
    row_count: int,
    source: str,
    source_contract_sha256: str,
) -> dict[str, object]:
    return {
        "archive_sha256": archive_sha256,
        "first_timestamp_ms": first_timestamp_ms,
        "last_timestamp_ms": last_timestamp_ms,
        "row_chain_sha256": row_chain_sha256,
        "row_count": row_count,
        "schema_version": POLYMARKET_ROUND23_BINANCE_SCHEMA_VERSION,
        "source": source,
        "source_contract_sha256": source_contract_sha256,
    }


def _initialize(store: Round22PilotStore) -> None:
    store.connection.execute(
        """
        CREATE SCHEMA IF NOT EXISTS exploratory;
        CREATE TABLE IF NOT EXISTS exploratory.round23_binance_second (
            source VARCHAR NOT NULL,
            archive_date VARCHAR NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            open_price VARCHAR NOT NULL,
            high_price VARCHAR NOT NULL,
            low_price VARCHAR NOT NULL,
            close_price VARCHAR NOT NULL,
            base_volume VARCHAR NOT NULL,
            quote_volume VARCHAR NOT NULL,
            signed_quote_volume VARCHAR NOT NULL,
            trade_count BIGINT NOT NULL,
            row_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (source, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS exploratory.round23_archive_manifest (
            source VARCHAR NOT NULL,
            archive_date VARCHAR NOT NULL,
            source_contract_sha256 VARCHAR NOT NULL,
            archive_sha256 VARCHAR NOT NULL,
            compressed_bytes BIGINT NOT NULL,
            first_timestamp_ms BIGINT NOT NULL,
            last_timestamp_ms BIGINT NOT NULL,
            row_count BIGINT NOT NULL,
            row_chain_sha256 VARCHAR NOT NULL,
            evidence_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (source, archive_date)
        );
        """
    )


def _insert_second_rows(
    store: Round22PilotStore,
    rows: list[list[object]],
) -> None:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    buffer.seek(0)
    relation = store.connection.read_csv(
        buffer,
        columns={
            "source": "VARCHAR",
            "archive_date": "VARCHAR",
            "timestamp_ms": "BIGINT",
            "open_price": "VARCHAR",
            "high_price": "VARCHAR",
            "low_price": "VARCHAR",
            "close_price": "VARCHAR",
            "base_volume": "VARCHAR",
            "quote_volume": "VARCHAR",
            "signed_quote_volume": "VARCHAR",
            "trade_count": "BIGINT",
            "row_sha256": "VARCHAR",
        },
        header=False,
    )
    relation.insert_into("exploratory.round23_binance_second")


def _audit_archive(
    store: Round22PilotStore,
    *,
    archive: Mapping[str, object],
    source_contract_sha256: str,
    start_ms: int,
    end_ms: int,
) -> Round23ArchiveResult | None:
    source = str(archive["source"])
    date = str(archive["date"])
    manifest = store.connection.execute(
        """
        SELECT source_contract_sha256, archive_sha256, compressed_bytes,
               first_timestamp_ms, last_timestamp_ms, row_count,
               row_chain_sha256, evidence_sha256
        FROM exploratory.round23_archive_manifest
        WHERE source = ? AND archive_date = ?
        """,
        [source, date],
    ).fetchone()
    rows = store.connection.execute(
        """
        SELECT source, archive_date, timestamp_ms, open_price, high_price,
               low_price, close_price, base_volume, quote_volume,
               signed_quote_volume, trade_count, row_sha256
        FROM exploratory.round23_binance_second
        WHERE source = ? AND archive_date = ?
        ORDER BY timestamp_ms
        """,
        [source, date],
    ).fetchall()
    if manifest is None:
        if rows:
            raise ValueError("Round 23 orphaned Binance rows exist")
        return None
    if len(rows) != 3_600:
        raise ValueError("Round 23 stored Binance row count differs")
    chain = "0" * 64
    for index, row in enumerate(rows):
        bar = _validated_bar(
            source=str(row[0]),
            date=str(row[1]),
            timestamp_ms=int(row[2]),
            open_price=str(row[3]),
            high_price=str(row[4]),
            low_price=str(row[5]),
            close_price=str(row[6]),
            base_volume=str(row[7]),
            quote_volume=str(row[8]),
            signed_quote_volume=str(row[9]),
            trade_count=int(row[10]),
        )
        if (
            bar.source != source
            or bar.date != date
            or bar.timestamp_ms != start_ms + index * 1_000
            or str(row[11]) != _canonical_sha256(bar.body())
        ):
            raise ValueError("Round 23 stored Binance row differs")
        chain = _row_chain(chain, bar)
    evidence = _archive_evidence(
        archive_sha256=str(archive["sha256"]),
        first_timestamp_ms=start_ms,
        last_timestamp_ms=end_ms - 1_000,
        row_chain_sha256=chain,
        row_count=len(rows),
        source=source,
        source_contract_sha256=source_contract_sha256,
    )
    if (
        str(manifest[0]) != source_contract_sha256
        or str(manifest[1]) != archive["sha256"]
        or int(manifest[2]) != int(archive["compressed_bytes"])
        or int(manifest[3]) != start_ms
        or int(manifest[4]) != end_ms - 1_000
        or int(manifest[5]) != len(rows)
        or str(manifest[6]) != chain
        or str(manifest[7]) != _canonical_sha256(evidence)
    ):
        raise ValueError("Round 23 stored Binance manifest differs")
    return Round23ArchiveResult(
        source=source,
        date=date,
        row_count=len(rows),
        first_timestamp_ms=start_ms,
        last_timestamp_ms=end_ms - 1_000,
        archive_sha256=str(archive["sha256"]),
        row_chain_sha256=chain,
        downloaded=False,
    )


def qualify_round23_binance_archive(
    *,
    repository: str | Path,
    source: str,
    date: str,
    start_ms: int,
    end_ms: int,
    client: Round23BinanceArchiveClient | None = None,
) -> Round23ArchiveResult:
    """Download, verify, and parse one frozen archive without writing it."""
    contract = load_round23_binance_source_contract(repository)
    data_contract = contract["data_contract"]
    archives = contract["archives"]
    assert isinstance(data_contract, Mapping)
    assert isinstance(archives, list)
    archive = next(
        (
            item
            for item in archives
            if isinstance(item, Mapping)
            and item.get("source") == source
            and item.get("date") == date
        ),
        None,
    )
    if archive is None or end_ms - start_ms != 3_600_000:
        raise ValueError("Round 23 Binance qualification selection differs")
    content = (client or Round23BinanceArchiveClient()).archive(
        archive,
        maximum_bytes=int(data_contract["maximum_compressed_archive_bytes"]),
    )
    bars = _parse_archive(
        content,
        archive,
        start_ms=start_ms,
        end_ms=end_ms,
        maximum_uncompressed_bytes=int(
            data_contract["maximum_uncompressed_member_bytes"]
        ),
    )
    chain = "0" * 64
    for bar in bars:
        chain = _row_chain(chain, bar)
    return Round23ArchiveResult(
        source=source,
        date=date,
        row_count=len(bars),
        first_timestamp_ms=bars[0].timestamp_ms,
        last_timestamp_ms=bars[-1].timestamp_ms,
        archive_sha256=str(archive["sha256"]),
        row_chain_sha256=chain,
        downloaded=True,
    )


def audit_round23_binance_archives(
    store: Round22PilotStore,
) -> tuple[Round23ArchiveResult, ...]:
    """Replay every retained row against the frozen public-source contracts."""
    contract = load_round23_binance_source_contract(store.contract.repository)
    table_count = int(
        store.connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'exploratory'
              AND table_name IN (
                  'round23_archive_manifest',
                  'round23_binance_second'
              )
            """
        ).fetchone()[0]
    )
    if table_count != 2:
        raise ValueError("Round 23 Binance archive tables are unavailable")
    archives = contract["archives"]
    assert isinstance(archives, list)
    results: list[Round23ArchiveResult] = []
    for archive in archives:
        assert isinstance(archive, Mapping)
        start_ms, end_ms = _window(store, str(archive["date"]))
        result = _audit_archive(
            store,
            archive=archive,
            source_contract_sha256=str(contract["source_contract_sha256"]),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if result is None:
            raise ValueError("Round 23 Binance archive is incomplete")
        results.append(result)
    return tuple(results)


def ingest_round23_binance_archives(
    store: Round22PilotStore,
    *,
    client: Round23BinanceArchiveClient | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Round23ArchiveResult, ...]:
    if store.read_only:
        raise ValueError("Round 23 Binance ingestion requires a writable store")
    contract = load_round23_binance_source_contract(store.contract.repository)
    data_contract = contract["data_contract"]
    assert isinstance(data_contract, Mapping)
    _initialize(store)
    archive_client = client or Round23BinanceArchiveClient()
    output: list[Round23ArchiveResult] = []
    archives = contract["archives"]
    assert isinstance(archives, list)
    for index, archive in enumerate(archives, start=1):
        assert isinstance(archive, Mapping)
        source = str(archive["source"])
        date = str(archive["date"])
        start_ms, end_ms = _window(store, date)
        existing = _audit_archive(
            store,
            archive=archive,
            source_contract_sha256=str(contract["source_contract_sha256"]),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if existing is not None:
            output.append(existing)
            if progress is not None:
                progress(
                    "round23_archive_reused",
                    {
                        "archive_count": len(archives),
                        "archive_index": index,
                        "date": date,
                        "row_count": existing.row_count,
                        "source": source,
                    },
                )
            continue
        if progress is not None:
            progress(
                "round23_archive_download",
                {
                    "archive_index": index,
                    "archive_count": len(archives),
                    "date": date,
                    "source": source,
                },
            )
        content = archive_client.archive(
            archive,
            maximum_bytes=int(data_contract["maximum_compressed_archive_bytes"]),
        )
        bars = _parse_archive(
            content,
            archive,
            start_ms=start_ms,
            end_ms=end_ms,
            maximum_uncompressed_bytes=int(
                data_contract["maximum_uncompressed_member_bytes"]
            ),
        )
        del content
        chain = "0" * 64
        rows: list[list[object]] = []
        for bar in bars:
            row_sha = _canonical_sha256(bar.body())
            chain = _row_chain(chain, bar)
            rows.append(
                [
                    bar.source,
                    bar.date,
                    bar.timestamp_ms,
                    bar.open_price,
                    bar.high_price,
                    bar.low_price,
                    bar.close_price,
                    bar.base_volume,
                    bar.quote_volume,
                    bar.signed_quote_volume,
                    bar.trade_count,
                    row_sha,
                ]
            )
        evidence = _archive_evidence(
            archive_sha256=str(archive["sha256"]),
            first_timestamp_ms=bars[0].timestamp_ms,
            last_timestamp_ms=bars[-1].timestamp_ms,
            row_chain_sha256=chain,
            row_count=len(bars),
            source=source,
            source_contract_sha256=str(contract["source_contract_sha256"]),
        )
        transaction_started = False
        try:
            store.connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            _insert_second_rows(store, rows)
            store.connection.execute(
                "INSERT INTO exploratory.round23_archive_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    source,
                    date,
                    contract["source_contract_sha256"],
                    archive["sha256"],
                    archive["compressed_bytes"],
                    bars[0].timestamp_ms,
                    bars[-1].timestamp_ms,
                    len(bars),
                    chain,
                    _canonical_sha256(evidence),
                ],
            )
            store.connection.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                store.connection.execute("ROLLBACK")
            raise
        committed = _audit_archive(
            store,
            archive=archive,
            source_contract_sha256=str(contract["source_contract_sha256"]),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if committed is None:
            raise AssertionError("Round 23 committed archive is unavailable")
        output.append(
            Round23ArchiveResult(
                source=committed.source,
                date=committed.date,
                row_count=committed.row_count,
                first_timestamp_ms=committed.first_timestamp_ms,
                last_timestamp_ms=committed.last_timestamp_ms,
                archive_sha256=committed.archive_sha256,
                row_chain_sha256=committed.row_chain_sha256,
                downloaded=True,
            )
        )
        if progress is not None:
            progress(
                "round23_archive_committed",
                {
                    "archive_index": index,
                    "archive_count": len(archives),
                    "date": date,
                    "row_count": len(bars),
                    "source": source,
                },
            )
    return tuple(output)


__all__ = [
    "POLYMARKET_ROUND23_BINANCE_SCHEMA_VERSION",
    "POLYMARKET_ROUND23_BINANCE_SOURCE_RELATIVE",
    "POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256",
    "Round23ArchiveResult",
    "Round23BinanceArchiveClient",
    "Round23SecondBar",
    "audit_round23_binance_archives",
    "ingest_round23_binance_archives",
    "load_round23_binance_source_contract",
    "qualify_round23_binance_archive",
]
