"""Frozen, checksummed spot/perpetual one-second research corpus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np
import requests

from .binance_archive import archive_file_url
from .microstructure_warehouse import (
    MicrostructureWarehouse,
    _download_verified_archive,
    _exclusive_operation_lock,
    create_archive_http_session,
)
from .spot_perpetual_flow import (
    FLOW_MARKET_TYPES,
    FLOW_SYMBOLS,
    SECONDS_PER_DAY,
    FlowDay,
    aggregate_trade_zip,
)


ROUND72_INVENTORY_SCHEMA = "round-072-spot-perpetual-inventory-v1"
ROUND72_DESIGN_SCHEMA = "round-072-spot-perpetual-price-discovery-design-v1"
SPOT_PERPETUAL_CORPUS_SCHEMA = "spot-perpetual-flow-corpus-v1"
SPOT_PERPETUAL_RESEARCH_ROUND = 72
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_LINE_RE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?([^\r\n]+)$")
_ProgressCallback = Callable[[str, int, int | None], None]


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


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_utc(value: object, label: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} does not include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _normalize_etag(value: object, label: str) -> str:
    text = str(value or "").strip().strip('"').lower()
    if not text or len(text) > 256 or any(ord(character) < 0x20 for character in text):
        raise ValueError(f"{label} is invalid")
    return text


def _require_sha256(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _month_sequence(first: str, last: str) -> tuple[str, ...]:
    try:
        start = date.fromisoformat(f"{first}-01")
        end = date.fromisoformat(f"{last}-01")
    except ValueError as exc:
        raise ValueError("Round 72 month bounds must use YYYY-MM") from exc
    if start > end:
        raise ValueError("Round 72 start month follows its end month")
    output: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        output.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return tuple(output)


@dataclass(frozen=True)
class FrozenFlowArchive:
    market_type: str
    symbol: str
    period: str
    url: str
    expected_bytes: int
    last_modified: str
    etag: str
    checksum_expected_bytes: int
    checksum_last_modified: str
    checksum_etag: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenFlowArchive":
        expected_fields = {
            "market_type",
            "symbol",
            "period",
            "url",
            "expected_bytes",
            "last_modified",
            "etag",
            "checksum_expected_bytes",
            "checksum_last_modified",
            "checksum_etag",
        }
        if set(value) != expected_fields:
            raise ValueError("Round 72 archive item fields differ from schema v1")
        market_type = str(value["market_type"])
        symbol = str(value["symbol"])
        period = str(value["period"])
        if market_type not in FLOW_MARKET_TYPES or symbol not in FLOW_SYMBOLS:
            raise ValueError("Round 72 archive has an unsupported stream")
        try:
            parsed_period = date.fromisoformat(period)
        except ValueError as exc:
            raise ValueError("Round 72 archive period is invalid") from exc
        if parsed_period.isoformat() != period:
            raise ValueError("Round 72 archive period is not canonical")
        expected_url = archive_file_url(
            symbol=symbol,
            interval="1s",
            period=period,
            market_type=market_type,
            cadence="daily",
            data_type="aggTrades",
        )
        if str(value["url"]) != expected_url:
            raise ValueError("Round 72 archive URL is not the exact official object")
        expected_bytes = int(value["expected_bytes"])
        checksum_expected_bytes = int(value["checksum_expected_bytes"])
        if expected_bytes <= 0 or checksum_expected_bytes <= 0:
            raise ValueError("Round 72 archive object size is not positive")
        return cls(
            market_type=market_type,
            symbol=symbol,
            period=period,
            url=expected_url,
            expected_bytes=expected_bytes,
            last_modified=_canonical_utc(value["last_modified"], "archive last-modified"),
            etag=_normalize_etag(value["etag"], "archive ETag"),
            checksum_expected_bytes=checksum_expected_bytes,
            checksum_last_modified=_canonical_utc(
                value["checksum_last_modified"], "checksum last-modified"
            ),
            checksum_etag=_normalize_etag(value["checksum_etag"], "checksum ETag"),
        )

    @property
    def source_key(self) -> str:
        return f"{self.market_type}:{self.symbol}"

    @property
    def contract_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FrozenFlowDay:
    month: str
    period: str
    selection_digest: str
    compressed_bytes: int
    archives: tuple[FrozenFlowArchive, ...]

    def contract_sha256(self, inventory_sha256: str) -> str:
        return _canonical_sha256(
            {
                "contract": SPOT_PERPETUAL_CORPUS_SCHEMA,
                "inventory_sha256": inventory_sha256,
                "month": self.month,
                "period": self.period,
                "sources": [asdict(value) for value in self.archives],
            }
        )


@dataclass(frozen=True)
class FrozenFlowContract:
    design_sha256: str
    inventory_sha256: str
    inventory_file_sha256: str
    selected_compressed_bytes: int
    days: tuple[FrozenFlowDay, ...]

    @property
    def expected_files(self) -> int:
        return sum(len(day.archives) for day in self.days)

    @property
    def expected_rows(self) -> int:
        return len(self.days) * len(FLOW_SYMBOLS) * SECONDS_PER_DAY


@dataclass(frozen=True)
class VerifiedFlowSource:
    archive: FrozenFlowArchive
    expected_sha256: str
    source_sha256: str
    compressed_bytes: int
    flow: FlowDay


@dataclass(frozen=True)
class FlowDayIngestResult:
    day_id: str
    period: str
    status: str
    source_count: int
    flow_rows: int
    compressed_bytes: int
    uncompressed_bytes: int
    combined_flow_sha256: str


def load_frozen_round72_contract(
    design_path: str | Path,
    inventory_path: str | Path,
) -> FrozenFlowContract:
    """Load and cryptographically bind the frozen Round 72 source selection."""

    design_file = Path(design_path)
    inventory_file = Path(inventory_path)
    design = _read_object(design_file, "Round 72 design")
    inventory_bytes = inventory_file.read_bytes()
    inventory = _read_object(inventory_file, "Round 72 inventory")

    canonical_design = dict(design)
    design_sha256 = _require_sha256(
        canonical_design.pop("design_sha256", ""), "Round 72 design hash"
    )
    canonical_inventory = dict(inventory)
    inventory_sha256 = _require_sha256(
        canonical_inventory.pop("inventory_sha256", ""), "Round 72 inventory hash"
    )
    inventory_file_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    if design_sha256 != _canonical_sha256(canonical_design):
        raise ValueError("Round 72 design canonical hash differs")
    if inventory_sha256 != _canonical_sha256(canonical_inventory):
        raise ValueError("Round 72 inventory canonical hash differs")
    source = design.get("source_contract")
    governance = design.get("governance")
    if (
        design.get("round") != SPOT_PERPETUAL_RESEARCH_ROUND
        or design.get("schema_version") != ROUND72_DESIGN_SCHEMA
        or inventory.get("round") != SPOT_PERPETUAL_RESEARCH_ROUND
        or inventory.get("schema_version") != ROUND72_INVENTORY_SCHEMA
        or inventory.get("status") != "complete"
        or not isinstance(source, Mapping)
        or not isinstance(governance, Mapping)
        or source.get("inventory_canonical_sha256") != inventory_sha256
        or source.get("inventory_file_sha256") != inventory_file_sha256
        or source.get("data_type") != "aggTrades"
        or tuple(source.get("market_types", ())) != FLOW_MARKET_TYPES
        or tuple(source.get("symbols", ())) != FLOW_SYMBOLS
        or source.get("raw_aggregate_trades_retained") is not False
        or source.get("derived_one_second_rows_retained") is not True
        or governance.get("profitability_claim_permitted") is not False
        or governance.get("trading_authority_permitted") is not False
    ):
        raise ValueError("Round 72 design and inventory contract is inconsistent")
    if (
        inventory.get("price_or_return_data_used_for_selection") is not False
        or inventory.get("profitability_claim") is not False
        or inventory.get("trading_authority") is not False
        or tuple(inventory.get("market_types", ())) != FLOW_MARKET_TYPES
        or tuple(inventory.get("symbols", ())) != FLOW_SYMBOLS
        or inventory.get("data_type") != "aggTrades"
    ):
        raise ValueError("Round 72 inventory governance fields are invalid")

    months = _month_sequence(str(inventory["start_month"]), str(inventory["end_month"]))
    selected = inventory.get("selected_months")
    if not isinstance(selected, list) or len(selected) != len(months):
        raise ValueError("Round 72 selected-month cardinality differs")
    seed = str(inventory.get("selection_seed") or "")
    expected_streams = {
        (market_type, symbol)
        for market_type in FLOW_MARKET_TYPES
        for symbol in FLOW_SYMBOLS
    }
    days: list[FrozenFlowDay] = []
    seen_urls: set[str] = set()
    for expected_month, raw_day in zip(months, selected, strict=True):
        if not isinstance(raw_day, Mapping):
            raise ValueError("Round 72 selected-month entry is not an object")
        if set(raw_day) != {
            "month",
            "selected_day",
            "selection_digest",
            "compressed_bytes",
            "files",
        }:
            raise ValueError("Round 72 selected-month fields differ from schema v1")
        month = str(raw_day["month"])
        period = str(raw_day["selected_day"])
        selection_digest = str(raw_day["selection_digest"])
        raw_archives = raw_day["files"]
        if month != expected_month or not period.startswith(f"{month}-"):
            raise ValueError("Round 72 selected day is outside its ordered month")
        expected_digest = hashlib.sha256(
            f"{seed}\x00{month}\x00{period}".encode("ascii")
        ).hexdigest()
        if selection_digest != expected_digest or not isinstance(raw_archives, list):
            raise ValueError("Round 72 hash-based day selection differs")
        if any(not isinstance(value, Mapping) for value in raw_archives):
            raise ValueError("Round 72 archive item is not an object")
        archives = tuple(
            sorted(
                (FrozenFlowArchive.from_mapping(value) for value in raw_archives),
                key=lambda value: (FLOW_MARKET_TYPES.index(value.market_type), value.symbol),
            )
        )
        if (
            len(archives) != len(expected_streams)
            or {(value.market_type, value.symbol) for value in archives} != expected_streams
            or {value.period for value in archives} != {period}
            or any(value.url in seen_urls for value in archives)
        ):
            raise ValueError("Round 72 selected day does not contain six unique streams")
        seen_urls.update(value.url for value in archives)
        compressed_bytes = int(raw_day["compressed_bytes"])
        if compressed_bytes != sum(value.expected_bytes for value in archives):
            raise ValueError("Round 72 selected-day compressed-byte total differs")
        days.append(
            FrozenFlowDay(
                month=month,
                period=period,
                selection_digest=selection_digest,
                compressed_bytes=compressed_bytes,
                archives=archives,
            )
        )

    selected_bytes = sum(value.compressed_bytes for value in days)
    selected_files = sum(len(value.archives) for value in days)
    if (
        int(inventory["complete_months"]) != len(days)
        or int(inventory["selected_files"]) != selected_files
        or int(inventory["selected_compressed_bytes"]) != selected_bytes
        or int(source["complete_months"]) != len(days)
        or int(source["selected_files"]) != selected_files
        or int(source["selected_compressed_bytes"]) != selected_bytes
        or int(inventory.get("excluded_month_count", -1)) != 0
        or inventory.get("excluded_months") != []
    ):
        raise ValueError("Round 72 frozen source totals differ")
    return FrozenFlowContract(
        design_sha256=design_sha256,
        inventory_sha256=inventory_sha256,
        inventory_file_sha256=inventory_file_sha256,
        selected_compressed_bytes=selected_bytes,
        days=tuple(days),
    )


def _http_last_modified(value: str) -> str:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("archive response Last-Modified header is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("archive response Last-Modified header lacks a timezone")
    return parsed.astimezone(UTC).isoformat()


def _fetch_verified_checksum(
    session: requests.Session,
    archive: FrozenFlowArchive,
    *,
    timeout_seconds: float,
) -> str:
    response = session.get(
        f"{archive.url}.CHECKSUM",
        timeout=max(1.0, float(timeout_seconds)),
    )
    response.raise_for_status()
    content = bytes(response.content)
    if len(content) != archive.checksum_expected_bytes or len(content) > 4_096:
        raise ValueError("checksum object size differs from frozen inventory")
    if _normalize_etag(response.headers.get("ETag"), "checksum response ETag") != archive.checksum_etag:
        raise ValueError("checksum object ETag differs from frozen inventory")
    if _http_last_modified(response.headers.get("Last-Modified", "")) != archive.checksum_last_modified:
        raise ValueError("checksum object Last-Modified differs from frozen inventory")
    try:
        line = content.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("checksum object is not ASCII") from exc
    match = _CHECKSUM_LINE_RE.fullmatch(line)
    if match is None or match.group(2) != Path(archive.url).name:
        raise ValueError("checksum object does not bind the requested archive filename")
    return match.group(1).lower()


class SpotPerpetualCorpusStore:
    """Atomic one-writer storage for verified spot/perpetual flow days."""

    def __init__(
        self,
        path: str | Path = "data/microstructure.duckdb",
        *,
        cache_root: str | Path = "data/archive-cache",
        memory_limit: str = "8GB",
        threads: int = 8,
        read_only: bool = False,
    ) -> None:
        self._warehouse = MicrostructureWarehouse(
            path,
            cache_root=cache_root,
            memory_limit=memory_limit,
            threads=threads,
            read_only=read_only,
        )
        self.read_only = bool(read_only)
        self._schema_ready = False

    @property
    def path(self) -> Path:
        return self._warehouse.path

    @property
    def cache_root(self) -> Path:
        return self._warehouse.cache_root

    def connect(self):
        conn = self._warehouse.connect()
        if not self._schema_ready:
            if self.read_only:
                required = {
                    "spot_perpetual_flow_day_manifest",
                    "spot_perpetual_flow_source_manifest",
                    "spot_perpetual_flow_1s",
                }
                observed = {
                    str(row[0])
                    for row in conn.execute("SHOW TABLES").fetchall()
                }
                if not required.issubset(observed):
                    raise ValueError("read-only spot/perpetual corpus schema is missing")
            else:
                self._init_schema()
            self._schema_ready = True
        return conn

    def close(self) -> None:
        self._warehouse.close()
        self._schema_ready = False

    def __enter__(self) -> "SpotPerpetualCorpusStore":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        conn = self._warehouse.connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spot_perpetual_flow_day_manifest (
                day_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                research_round UINTEGER NOT NULL,
                period VARCHAR NOT NULL,
                selected_month VARCHAR NOT NULL,
                inventory_sha256 VARCHAR NOT NULL,
                source_contract_sha256 VARCHAR NOT NULL,
                combined_flow_sha256 VARCHAR NOT NULL,
                source_count UINTEGER NOT NULL,
                symbol_count UINTEGER NOT NULL,
                seconds_per_symbol UINTEGER NOT NULL,
                flow_rows UBIGINT NOT NULL,
                expected_compressed_bytes UBIGINT NOT NULL,
                compressed_bytes UBIGINT NOT NULL,
                uncompressed_bytes UBIGINT NOT NULL,
                source_rows UBIGINT NOT NULL,
                aggregate_trade_count UBIGINT NOT NULL,
                constituent_trade_count UBIGINT NOT NULL,
                first_second_ms BIGINT NOT NULL,
                last_second_ms BIGINT NOT NULL,
                status VARCHAR NOT NULL,
                is_current BOOLEAN NOT NULL,
                ingested_at_ms BIGINT NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS spot_perpetual_flow_source_manifest (
                day_id VARCHAR NOT NULL,
                source_contract_sha256 VARCHAR NOT NULL,
                market_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                period VARCHAR NOT NULL,
                url VARCHAR NOT NULL,
                expected_bytes UBIGINT NOT NULL,
                compressed_bytes UBIGINT NOT NULL,
                uncompressed_bytes UBIGINT NOT NULL,
                last_modified VARCHAR NOT NULL,
                etag VARCHAR NOT NULL,
                checksum_expected_bytes UBIGINT NOT NULL,
                checksum_last_modified VARCHAR NOT NULL,
                checksum_etag VARCHAR NOT NULL,
                expected_sha256 VARCHAR NOT NULL,
                source_sha256 VARCHAR NOT NULL,
                flow_sha256 VARCHAR NOT NULL,
                source_rows UBIGINT NOT NULL,
                aggregate_trade_count UBIGINT NOT NULL,
                constituent_trade_count UBIGINT NOT NULL,
                first_aggregate_trade_id UBIGINT NOT NULL,
                last_aggregate_trade_id UBIGINT NOT NULL,
                aggregate_trade_id_gaps UBIGINT NOT NULL,
                constituent_trade_id_gaps UBIGINT NOT NULL,
                first_trade_time_ms BIGINT NOT NULL,
                last_trade_time_ms BIGINT NOT NULL,
                best_match_false_count UBIGINT NOT NULL,
                header_present BOOLEAN NOT NULL,
                member_name VARCHAR NOT NULL,
                raw_archive_retained BOOLEAN NOT NULL,
                PRIMARY KEY (day_id, market_type, symbol)
            );

            CREATE TABLE IF NOT EXISTS spot_perpetual_flow_1s (
                day_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                second_ms BIGINT NOT NULL,
                spot_open DOUBLE,
                spot_high DOUBLE,
                spot_low DOUBLE,
                spot_close DOUBLE,
                spot_base_volume DOUBLE NOT NULL,
                spot_quote_volume DOUBLE NOT NULL,
                spot_aggressive_buy_quote DOUBLE NOT NULL,
                spot_aggressive_sell_quote DOUBLE NOT NULL,
                spot_aggregate_count UINTEGER NOT NULL,
                spot_constituent_trade_count UINTEGER NOT NULL,
                spot_maximum_aggregate_quote DOUBLE NOT NULL,
                spot_squared_aggregate_quote_sum DOUBLE NOT NULL,
                spot_last_trade_age_seconds UINTEGER NOT NULL,
                perpetual_open DOUBLE,
                perpetual_high DOUBLE,
                perpetual_low DOUBLE,
                perpetual_close DOUBLE,
                perpetual_base_volume DOUBLE NOT NULL,
                perpetual_quote_volume DOUBLE NOT NULL,
                perpetual_aggressive_buy_quote DOUBLE NOT NULL,
                perpetual_aggressive_sell_quote DOUBLE NOT NULL,
                perpetual_aggregate_count UINTEGER NOT NULL,
                perpetual_constituent_trade_count UINTEGER NOT NULL,
                perpetual_maximum_aggregate_quote DOUBLE NOT NULL,
                perpetual_squared_aggregate_quote_sum DOUBLE NOT NULL,
                perpetual_last_trade_age_seconds UINTEGER NOT NULL
            );

            CREATE OR REPLACE VIEW current_spot_perpetual_flow_1s AS
                SELECT
                    f.*,
                    f.second_ms + 1000 AS available_time_ms,
                    f.spot_quote_volume / nullif(f.spot_base_volume, 0) AS spot_vwap,
                    f.perpetual_quote_volume
                        / nullif(f.perpetual_base_volume, 0) AS perpetual_vwap,
                    (f.spot_aggressive_buy_quote - f.spot_aggressive_sell_quote)
                        / nullif(f.spot_quote_volume, 0) AS spot_taker_imbalance,
                    (f.perpetual_aggressive_buy_quote
                        - f.perpetual_aggressive_sell_quote)
                        / nullif(f.perpetual_quote_volume, 0) AS perpetual_taker_imbalance
                FROM spot_perpetual_flow_1s f
                JOIN spot_perpetual_flow_day_manifest m USING (day_id)
                WHERE m.status = 'complete' AND m.is_current;
            """
        )

    @staticmethod
    def _combined_flow_sha256(sources: Sequence[VerifiedFlowSource]) -> str:
        return _canonical_sha256(
            [
                {
                    "expected_sha256": value.expected_sha256,
                    "flow_sha256": value.flow.flow_sha256,
                    "source_contract_sha256": value.archive.contract_sha256,
                    "source_key": value.archive.source_key,
                    "source_sha256": value.source_sha256,
                }
                for value in sorted(sources, key=lambda item: item.archive.source_key)
            ]
        )

    @staticmethod
    def _validate_sources(
        day: FrozenFlowDay,
        sources: Sequence[VerifiedFlowSource],
    ) -> tuple[VerifiedFlowSource, ...]:
        ordered = tuple(sorted(sources, key=lambda value: value.archive.source_key))
        expected = {value.source_key: value for value in day.archives}
        if len(ordered) != len(expected) or {
            value.archive.source_key for value in ordered
        } != set(expected):
            raise ValueError("verified flow sources differ from the frozen six-stream day")
        for source in ordered:
            archive = expected[source.archive.source_key]
            if source.archive != archive:
                raise ValueError("verified flow source metadata differs from frozen inventory")
            expected_sha256 = _require_sha256(
                source.expected_sha256, "source sidecar SHA-256"
            )
            source_sha256 = _require_sha256(source.source_sha256, "source SHA-256")
            if expected_sha256 != source_sha256:
                raise ValueError("verified flow source differs from sidecar SHA-256")
            if int(source.compressed_bytes) != archive.expected_bytes:
                raise ValueError("verified flow source size differs from frozen inventory")
            flow = source.flow
            flow.validate(require_full_day=True)
            if (
                flow.symbol != archive.symbol
                or flow.market_type != archive.market_type
                or flow.period != archive.period
                or flow.rows != SECONDS_PER_DAY
                or flow.audit.member_uncompressed_bytes <= 0
            ):
                raise ValueError("verified flow aggregation identity differs")
        return ordered

    @staticmethod
    def _batch(spot: FlowDay, perpetual: FlowDay) -> dict[str, np.ndarray]:
        if not np.array_equal(spot.second_ms, perpetual.second_ms):
            raise ValueError("spot and perpetual seconds are not identical")
        output: dict[str, np.ndarray] = {"second_ms": spot.second_ms}
        fields = (
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume",
            "aggressive_buy_quote",
            "aggressive_sell_quote",
            "aggregate_count",
            "constituent_trade_count",
            "maximum_aggregate_quote",
            "squared_aggregate_quote_sum",
            "last_trade_age_seconds",
        )
        for prefix, flow in (("spot", spot), ("perpetual", perpetual)):
            for field in fields:
                output[f"{prefix}_{field}"] = getattr(flow, field)
        return output

    def _commit_unlocked(
        self,
        day: FrozenFlowDay,
        *,
        inventory_sha256: str,
        sources: Sequence[VerifiedFlowSource],
    ) -> FlowDayIngestResult:
        inventory_sha256 = _require_sha256(inventory_sha256, "inventory SHA-256")
        ordered = self._validate_sources(day, sources)
        day_id = day.contract_sha256(inventory_sha256)
        combined_flow_sha256 = self._combined_flow_sha256(ordered)
        by_stream = {
            (value.archive.market_type, value.archive.symbol): value.flow
            for value in ordered
        }
        compressed_bytes = sum(value.compressed_bytes for value in ordered)
        uncompressed_bytes = sum(
            value.flow.audit.member_uncompressed_bytes for value in ordered
        )
        source_rows = sum(value.flow.audit.source_rows for value in ordered)
        aggregate_count = sum(
            value.flow.audit.aggregate_trade_count for value in ordered
        )
        constituent_count = sum(
            value.flow.audit.constituent_trade_count for value in ordered
        )
        flow_rows = len(FLOW_SYMBOLS) * SECONDS_PER_DAY
        first_second_ms = min(value.flow.second_ms[0] for value in ordered)
        last_second_ms = max(value.flow.second_ms[-1] for value in ordered)
        now_ms = int(time.time() * 1_000)
        conn = self.connect()
        conn.execute("BEGIN TRANSACTION")
        registered = False
        try:
            stale_day_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT day_id FROM spot_perpetual_flow_day_manifest "
                    "WHERE period = ? AND day_id != ?",
                    [day.period, day_id],
                ).fetchall()
            ]
            for stale_day_id in stale_day_ids:
                conn.execute(
                    "DELETE FROM spot_perpetual_flow_1s WHERE day_id = ?",
                    [stale_day_id],
                )
                conn.execute(
                    "DELETE FROM spot_perpetual_flow_source_manifest WHERE day_id = ?",
                    [stale_day_id],
                )
                conn.execute(
                    "DELETE FROM spot_perpetual_flow_day_manifest WHERE day_id = ?",
                    [stale_day_id],
                )
            conn.execute("DELETE FROM spot_perpetual_flow_1s WHERE day_id = ?", [day_id])
            conn.execute(
                "DELETE FROM spot_perpetual_flow_source_manifest WHERE day_id = ?",
                [day_id],
            )
            conn.execute(
                "DELETE FROM spot_perpetual_flow_day_manifest WHERE day_id = ?",
                [day_id],
            )
            insert_columns = (
                "second_ms,spot_open,spot_high,spot_low,spot_close,spot_base_volume,"
                "spot_quote_volume,spot_aggressive_buy_quote,spot_aggressive_sell_quote,"
                "spot_aggregate_count,spot_constituent_trade_count,"
                "spot_maximum_aggregate_quote,spot_squared_aggregate_quote_sum,"
                "spot_last_trade_age_seconds,perpetual_open,perpetual_high,perpetual_low,"
                "perpetual_close,perpetual_base_volume,perpetual_quote_volume,"
                "perpetual_aggressive_buy_quote,perpetual_aggressive_sell_quote,"
                "perpetual_aggregate_count,perpetual_constituent_trade_count,"
                "perpetual_maximum_aggregate_quote,perpetual_squared_aggregate_quote_sum,"
                "perpetual_last_trade_age_seconds"
            )
            for symbol in FLOW_SYMBOLS:
                batch = self._batch(
                    by_stream[("spot", symbol)], by_stream[("futures", symbol)]
                )
                conn.register("_spot_perpetual_flow_batch", batch)
                registered = True
                conn.execute(
                    "INSERT INTO spot_perpetual_flow_1s "
                    f"SELECT ?, ?, {insert_columns} FROM _spot_perpetual_flow_batch",
                    [day_id, symbol],
                )
                conn.unregister("_spot_perpetual_flow_batch")
                registered = False
            source_rows_to_insert = []
            for value in ordered:
                archive = value.archive
                audit = value.flow.audit
                source_rows_to_insert.append(
                    (
                        day_id,
                        archive.contract_sha256,
                        archive.market_type,
                        archive.symbol,
                        archive.period,
                        archive.url,
                        archive.expected_bytes,
                        value.compressed_bytes,
                        audit.member_uncompressed_bytes,
                        archive.last_modified,
                        archive.etag,
                        archive.checksum_expected_bytes,
                        archive.checksum_last_modified,
                        archive.checksum_etag,
                        value.expected_sha256,
                        value.source_sha256,
                        value.flow.flow_sha256,
                        audit.source_rows,
                        audit.aggregate_trade_count,
                        audit.constituent_trade_count,
                        audit.first_aggregate_trade_id,
                        audit.last_aggregate_trade_id,
                        audit.aggregate_trade_id_gaps,
                        audit.constituent_trade_id_gaps,
                        audit.first_trade_time_ms,
                        audit.last_trade_time_ms,
                        audit.best_match_false_count,
                        audit.header_present,
                        audit.member_name,
                        False,
                    )
                )
            conn.executemany(
                """
                INSERT INTO spot_perpetual_flow_source_manifest VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                source_rows_to_insert,
            )
            conn.execute(
                """
                INSERT INTO spot_perpetual_flow_day_manifest VALUES (
                    ?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?,?
                )
                """,
                [
                    day_id,
                    SPOT_PERPETUAL_CORPUS_SCHEMA,
                    SPOT_PERPETUAL_RESEARCH_ROUND,
                    day.period,
                    day.month,
                    inventory_sha256,
                    day_id,
                    combined_flow_sha256,
                    len(ordered),
                    len(FLOW_SYMBOLS),
                    SECONDS_PER_DAY,
                    flow_rows,
                    day.compressed_bytes,
                    compressed_bytes,
                    uncompressed_bytes,
                    source_rows,
                    aggregate_count,
                    constituent_count,
                    int(first_second_ms),
                    int(last_second_ms),
                    "complete",
                    True,
                    now_ms,
                    "",
                ],
            )
            conn.execute("COMMIT")
        except Exception:
            if registered:
                conn.unregister("_spot_perpetual_flow_batch")
            conn.execute("ROLLBACK")
            raise
        return FlowDayIngestResult(
            day_id=day_id,
            period=day.period,
            status="complete",
            source_count=len(ordered),
            flow_rows=flow_rows,
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=uncompressed_bytes,
            combined_flow_sha256=combined_flow_sha256,
        )

    def commit_verified_day(
        self,
        day: FrozenFlowDay,
        *,
        inventory_sha256: str,
        sources: Sequence[VerifiedFlowSource],
    ) -> FlowDayIngestResult:
        """Commit six already verified sources as one indivisible UTC day."""

        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        with _exclusive_operation_lock(lock_path):
            return self._commit_unlocked(
                day,
                inventory_sha256=inventory_sha256,
                sources=sources,
            )

    def _cache_path(self, archive: FrozenFlowArchive) -> Path:
        market = "spot" if archive.market_type == "spot" else "usdm"
        return (
            self.cache_root
            / "binance"
            / market
            / "aggTrades"
            / archive.symbol
            / Path(archive.url).name
        )

    def ingest_day(
        self,
        day: FrozenFlowDay,
        *,
        inventory_sha256: str,
        timeout_seconds: float = 120.0,
        maximum_uncompressed_bytes: int = 8 * 1024**3,
        session: requests.Session | None = None,
        progress: _ProgressCallback | None = None,
    ) -> FlowDayIngestResult:
        """Download, verify, stream, and atomically commit one frozen day."""

        if not 1.0 <= float(timeout_seconds) <= 900.0:
            raise ValueError("Round 72 archive timeout is outside 1..900 seconds")
        if int(maximum_uncompressed_bytes) < 64 * 1024**2:
            raise ValueError("Round 72 uncompressed archive bound is too small")
        inventory_sha256 = _require_sha256(inventory_sha256, "inventory SHA-256")
        own_session = session is None
        active_session = session or create_archive_http_session()
        cache_paths: set[Path] = set()
        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        try:
            with _exclusive_operation_lock(lock_path):
                reusable = self._certify_day_or_none(day, inventory_sha256=inventory_sha256)
                if reusable is not None:
                    return FlowDayIngestResult(
                        day_id=str(reusable["day_id"]),
                        period=day.period,
                        status="skipped",
                        source_count=int(reusable["source_count"]),
                        flow_rows=int(reusable["flow_rows"]),
                        compressed_bytes=int(reusable["compressed_bytes"]),
                        uncompressed_bytes=int(reusable["uncompressed_bytes"]),
                        combined_flow_sha256=str(reusable["combined_flow_sha256"]),
                    )
                verified: list[VerifiedFlowSource] = []
                for archive in day.archives:
                    if progress:
                        progress(f"checksum:{archive.source_key}", 0, archive.checksum_expected_bytes)
                    expected_sha256 = _fetch_verified_checksum(
                        active_session,
                        archive,
                        timeout_seconds=timeout_seconds,
                    )
                    if progress:
                        progress(
                            f"checksum:{archive.source_key}",
                            archive.checksum_expected_bytes,
                            archive.checksum_expected_bytes,
                        )
                    path = self._cache_path(archive)
                    cache_paths.add(path)

                    def source_progress(
                        phase: str,
                        current: int,
                        total: int | None,
                        *,
                        source_key: str = archive.source_key,
                    ) -> None:
                        if progress:
                            progress(f"{phase}:{source_key}", current, total)

                    compressed_bytes, source_sha256 = _download_verified_archive(
                        active_session,
                        archive.url,
                        path,
                        expected_sha256=expected_sha256,
                        timeout_seconds=timeout_seconds,
                        expected_bytes=archive.expected_bytes,
                        max_download_bytes=max(
                            archive.expected_bytes + 8 * 1024**2,
                            int(archive.expected_bytes * 1.1),
                        ),
                        progress=source_progress,
                    )
                    if compressed_bytes != archive.expected_bytes:
                        raise ValueError("archive object size differs from frozen inventory")
                    flow = aggregate_trade_zip(
                        path,
                        symbol=archive.symbol,
                        market_type=archive.market_type,
                        period=archive.period,
                        maximum_uncompressed_bytes=min(
                            int(maximum_uncompressed_bytes),
                            max(64 * 1024**2, archive.expected_bytes * 100),
                        ),
                    )
                    verified.append(
                        VerifiedFlowSource(
                            archive=archive,
                            expected_sha256=expected_sha256,
                            source_sha256=source_sha256,
                            compressed_bytes=compressed_bytes,
                            flow=flow,
                        )
                    )
                    if progress:
                        progress(
                            f"aggregated:{archive.source_key}",
                            flow.audit.source_rows,
                            flow.audit.source_rows,
                        )
                return self._commit_unlocked(
                    day,
                    inventory_sha256=inventory_sha256,
                    sources=verified,
                )
        finally:
            for path in cache_paths:
                for candidate in (
                    path,
                    path.with_suffix(path.suffix + ".part"),
                    *path.parent.glob(f"{path.name}.checksum-mismatch-*"),
                ):
                    try:
                        candidate.unlink()
                    except OSError:
                        pass
            if own_session:
                active_session.close()

    def _certify_day_or_none(
        self,
        day: FrozenFlowDay,
        *,
        inventory_sha256: str,
    ) -> dict[str, object] | None:
        try:
            return self.certify_day(day, inventory_sha256=inventory_sha256)
        except ValueError:
            return None

    def certify_day(
        self,
        day: FrozenFlowDay,
        *,
        inventory_sha256: str,
    ) -> dict[str, object]:
        """Revalidate one persisted day against metadata and physical rows."""

        inventory_sha256 = _require_sha256(inventory_sha256, "inventory SHA-256")
        day_id = day.contract_sha256(inventory_sha256)
        conn = self.connect()
        manifest = conn.execute(
            """
            SELECT schema_version, research_round, period, selected_month,
                   inventory_sha256, source_contract_sha256, combined_flow_sha256,
                   source_count, symbol_count, seconds_per_symbol, flow_rows,
                   expected_compressed_bytes, compressed_bytes, uncompressed_bytes,
                   first_second_ms, last_second_ms, status, is_current
            FROM spot_perpetual_flow_day_manifest WHERE day_id = ?
            """,
            [day_id],
        ).fetchone()
        day_start_ms = int(
            datetime.fromisoformat(day.period).replace(tzinfo=UTC).timestamp() * 1_000
        )
        expected_rows = len(FLOW_SYMBOLS) * SECONDS_PER_DAY
        if manifest is None or (
            str(manifest[0]) != SPOT_PERPETUAL_CORPUS_SCHEMA
            or int(manifest[1]) != SPOT_PERPETUAL_RESEARCH_ROUND
            or str(manifest[2]) != day.period
            or str(manifest[3]) != day.month
            or str(manifest[4]) != inventory_sha256
            or str(manifest[5]) != day_id
            or not _SHA256_RE.fullmatch(str(manifest[6]))
            or int(manifest[7]) != len(day.archives)
            or int(manifest[8]) != len(FLOW_SYMBOLS)
            or int(manifest[9]) != SECONDS_PER_DAY
            or int(manifest[10]) != expected_rows
            or int(manifest[11]) != day.compressed_bytes
            or int(manifest[12]) != day.compressed_bytes
            or int(manifest[14]) != day_start_ms
            or int(manifest[15]) != day_start_ms + (SECONDS_PER_DAY - 1) * 1_000
            or str(manifest[16]) != "complete"
            or bool(manifest[17]) is not True
        ):
            raise ValueError(f"{day.period} spot/perpetual day manifest is not reusable")
        source_rows = conn.execute(
            """
            SELECT source_contract_sha256, market_type, symbol, period, url,
                   expected_bytes, compressed_bytes, last_modified, etag,
                   checksum_expected_bytes, checksum_last_modified, checksum_etag,
                   expected_sha256, source_sha256, flow_sha256, raw_archive_retained
            FROM spot_perpetual_flow_source_manifest
            WHERE day_id = ? ORDER BY market_type, symbol
            """,
            [day_id],
        ).fetchall()
        expected_by_key = {value.source_key: value for value in day.archives}
        if len(source_rows) != len(expected_by_key):
            raise ValueError(f"{day.period} source-manifest cardinality differs")
        for row in source_rows:
            key = f"{row[1]}:{row[2]}"
            archive = expected_by_key.get(key)
            if archive is None or (
                str(row[0]) != archive.contract_sha256
                or str(row[3]) != archive.period
                or str(row[4]) != archive.url
                or int(row[5]) != archive.expected_bytes
                or int(row[6]) != archive.expected_bytes
                or str(row[7]) != archive.last_modified
                or str(row[8]) != archive.etag
                or int(row[9]) != archive.checksum_expected_bytes
                or str(row[10]) != archive.checksum_last_modified
                or str(row[11]) != archive.checksum_etag
                or str(row[12]) != str(row[13])
                or not _SHA256_RE.fullmatch(str(row[12]))
                or not _SHA256_RE.fullmatch(str(row[14]))
                or bool(row[15]) is not False
            ):
                raise ValueError(f"{day.period} source-manifest identity differs")
        physical = conn.execute(
            """
            SELECT count(*)::UBIGINT,
                   count(DISTINCT symbol)::UBIGINT,
                   count(*) - count(DISTINCT (symbol, second_ms)),
                   min(second_ms), max(second_ms),
                   count(*) FILTER (WHERE second_ms % 1000 != 0)::UBIGINT,
                   count(*) FILTER (
                       WHERE spot_base_volume IS NULL OR spot_base_volume < 0
                          OR spot_quote_volume IS NULL OR spot_quote_volume < 0
                          OR spot_aggressive_buy_quote IS NULL
                          OR spot_aggressive_sell_quote IS NULL
                          OR abs(spot_aggressive_buy_quote + spot_aggressive_sell_quote
                                 - spot_quote_volume)
                             > greatest(1e-8, spot_quote_volume * 1e-10)
                          OR perpetual_base_volume IS NULL OR perpetual_base_volume < 0
                          OR perpetual_quote_volume IS NULL OR perpetual_quote_volume < 0
                          OR perpetual_aggressive_buy_quote IS NULL
                          OR perpetual_aggressive_sell_quote IS NULL
                          OR abs(perpetual_aggressive_buy_quote
                                 + perpetual_aggressive_sell_quote
                                 - perpetual_quote_volume)
                             > greatest(1e-8, perpetual_quote_volume * 1e-10)
                          OR (spot_aggregate_count > 0 AND (
                              spot_open IS NULL OR spot_high IS NULL OR spot_low IS NULL
                              OR spot_close IS NULL OR spot_low <= 0
                              OR spot_high < greatest(spot_open, spot_close)
                              OR spot_low > least(spot_open, spot_close)
                              OR spot_last_trade_age_seconds != 0))
                          OR (perpetual_aggregate_count > 0 AND (
                              perpetual_open IS NULL OR perpetual_high IS NULL
                              OR perpetual_low IS NULL OR perpetual_close IS NULL
                              OR perpetual_low <= 0
                              OR perpetual_high < greatest(perpetual_open, perpetual_close)
                              OR perpetual_low > least(perpetual_open, perpetual_close)
                              OR perpetual_last_trade_age_seconds != 0))
                   )::UBIGINT
            FROM spot_perpetual_flow_1s WHERE day_id = ?
            """,
            [day_id],
        ).fetchone()
        by_symbol = conn.execute(
            """
            SELECT symbol, count(*)::UBIGINT, min(second_ms), max(second_ms)
            FROM spot_perpetual_flow_1s WHERE day_id = ?
            GROUP BY symbol ORDER BY symbol
            """,
            [day_id],
        ).fetchall()
        if physical is None or (
            int(physical[0]) != expected_rows
            or int(physical[1]) != len(FLOW_SYMBOLS)
            or int(physical[2]) != 0
            or int(physical[3]) != day_start_ms
            or int(physical[4]) != day_start_ms + (SECONDS_PER_DAY - 1) * 1_000
            or int(physical[5]) != 0
            or int(physical[6]) != 0
            or len(by_symbol) != len(FLOW_SYMBOLS)
            or {str(row[0]) for row in by_symbol} != set(FLOW_SYMBOLS)
            or any(
                int(row[1]) != SECONDS_PER_DAY
                or int(row[2]) != day_start_ms
                or int(row[3]) != day_start_ms + (SECONDS_PER_DAY - 1) * 1_000
                for row in by_symbol
            )
        ):
            raise ValueError(f"{day.period} physical spot/perpetual rows are invalid")
        return {
            "day_id": day_id,
            "period": day.period,
            "source_count": int(manifest[7]),
            "flow_rows": int(manifest[10]),
            "compressed_bytes": int(manifest[12]),
            "uncompressed_bytes": int(manifest[13]),
            "combined_flow_sha256": str(manifest[6]),
        }

    def certify_corpus(self, contract: FrozenFlowContract) -> dict[str, object]:
        """Certify the complete frozen corpus with one fact-table scan."""

        inventory_sha256 = _require_sha256(
            contract.inventory_sha256, "inventory SHA-256"
        )
        expected_days = {
            day.contract_sha256(inventory_sha256): day for day in contract.days
        }
        conn = self.connect()
        manifests = conn.execute(
            """
            SELECT day_id, period, selected_month, source_contract_sha256,
                   combined_flow_sha256, source_count, symbol_count,
                   seconds_per_symbol, flow_rows, expected_compressed_bytes,
                   compressed_bytes, uncompressed_bytes, status, is_current,
                   schema_version, research_round
            FROM spot_perpetual_flow_day_manifest
            WHERE inventory_sha256 = ? ORDER BY period
            """,
            [inventory_sha256],
        ).fetchall()
        if len(manifests) != len(expected_days) or {
            str(row[0]) for row in manifests
        } != set(expected_days):
            raise ValueError("Round 72 corpus day-manifest set differs")
        manifest_by_id = {str(row[0]): row for row in manifests}
        for day_id, day in expected_days.items():
            row = manifest_by_id[day_id]
            if (
                str(row[1]) != day.period
                or str(row[2]) != day.month
                or str(row[3]) != day_id
                or not _SHA256_RE.fullmatch(str(row[4]))
                or int(row[5]) != len(day.archives)
                or int(row[6]) != len(FLOW_SYMBOLS)
                or int(row[7]) != SECONDS_PER_DAY
                or int(row[8]) != len(FLOW_SYMBOLS) * SECONDS_PER_DAY
                or int(row[9]) != day.compressed_bytes
                or int(row[10]) != day.compressed_bytes
                or int(row[11]) <= 0
                or str(row[12]) != "complete"
                or bool(row[13]) is not True
                or str(row[14]) != SPOT_PERPETUAL_CORPUS_SCHEMA
                or int(row[15]) != SPOT_PERPETUAL_RESEARCH_ROUND
            ):
                raise ValueError(f"{day.period} corpus manifest differs")

        sources = conn.execute(
            """
            SELECT s.day_id, s.market_type, s.symbol, s.source_contract_sha256,
                   s.expected_bytes, s.compressed_bytes, s.expected_sha256,
                   s.source_sha256, s.flow_sha256, s.raw_archive_retained
            FROM spot_perpetual_flow_source_manifest s
            JOIN spot_perpetual_flow_day_manifest d USING (day_id)
            WHERE d.inventory_sha256 = ?
            ORDER BY s.day_id, s.market_type, s.symbol
            """,
            [inventory_sha256],
        ).fetchall()
        expected_sources = {
            (day_id, archive.market_type, archive.symbol): archive
            for day_id, day in expected_days.items()
            for archive in day.archives
        }
        if len(sources) != contract.expected_files or {
            (str(row[0]), str(row[1]), str(row[2])) for row in sources
        } != set(expected_sources):
            raise ValueError("Round 72 corpus source-manifest set differs")
        for row in sources:
            key = (str(row[0]), str(row[1]), str(row[2]))
            archive = expected_sources[key]
            if (
                str(row[3]) != archive.contract_sha256
                or int(row[4]) != archive.expected_bytes
                or int(row[5]) != archive.expected_bytes
                or str(row[6]) != str(row[7])
                or not _SHA256_RE.fullmatch(str(row[6]))
                or not _SHA256_RE.fullmatch(str(row[8]))
                or bool(row[9]) is not False
            ):
                raise ValueError("Round 72 corpus source identity differs")

        facts = conn.execute(
            """
            SELECT f.day_id, f.symbol, count(*)::UBIGINT,
                   count(DISTINCT f.second_ms)::UBIGINT,
                   min(f.second_ms), max(f.second_ms),
                   count(*) FILTER (WHERE f.second_ms % 1000 != 0)::UBIGINT,
                   count(*) FILTER (
                       WHERE f.spot_base_volume IS NULL OR f.spot_base_volume < 0
                          OR f.spot_quote_volume IS NULL OR f.spot_quote_volume < 0
                          OR f.spot_aggressive_buy_quote IS NULL
                          OR f.spot_aggressive_sell_quote IS NULL
                          OR abs(f.spot_aggressive_buy_quote
                                 + f.spot_aggressive_sell_quote
                                 - f.spot_quote_volume)
                             > greatest(1e-8, f.spot_quote_volume * 1e-10)
                          OR f.perpetual_base_volume IS NULL
                          OR f.perpetual_base_volume < 0
                          OR f.perpetual_quote_volume IS NULL
                          OR f.perpetual_quote_volume < 0
                          OR f.perpetual_aggressive_buy_quote IS NULL
                          OR f.perpetual_aggressive_sell_quote IS NULL
                          OR abs(f.perpetual_aggressive_buy_quote
                                 + f.perpetual_aggressive_sell_quote
                                 - f.perpetual_quote_volume)
                             > greatest(1e-8, f.perpetual_quote_volume * 1e-10)
                          OR (f.spot_aggregate_count > 0 AND (
                              f.spot_open IS NULL OR f.spot_high IS NULL
                              OR f.spot_low IS NULL OR f.spot_close IS NULL
                              OR f.spot_low <= 0
                              OR f.spot_high < greatest(f.spot_open, f.spot_close)
                              OR f.spot_low > least(f.spot_open, f.spot_close)
                              OR f.spot_last_trade_age_seconds != 0))
                          OR (f.perpetual_aggregate_count > 0 AND (
                              f.perpetual_open IS NULL OR f.perpetual_high IS NULL
                              OR f.perpetual_low IS NULL OR f.perpetual_close IS NULL
                              OR f.perpetual_low <= 0
                              OR f.perpetual_high
                                 < greatest(f.perpetual_open, f.perpetual_close)
                              OR f.perpetual_low
                                 > least(f.perpetual_open, f.perpetual_close)
                              OR f.perpetual_last_trade_age_seconds != 0))
                   )::UBIGINT
            FROM spot_perpetual_flow_1s f
            JOIN spot_perpetual_flow_day_manifest d USING (day_id)
            WHERE d.inventory_sha256 = ?
            GROUP BY f.day_id, f.symbol
            ORDER BY f.day_id, f.symbol
            """,
            [inventory_sha256],
        ).fetchall()
        expected_fact_keys = {
            (day_id, symbol)
            for day_id in expected_days
            for symbol in FLOW_SYMBOLS
        }
        if len(facts) != len(expected_fact_keys) or {
            (str(row[0]), str(row[1])) for row in facts
        } != expected_fact_keys:
            raise ValueError("Round 72 corpus fact partition set differs")
        for row in facts:
            day = expected_days[str(row[0])]
            start_ms = int(
                datetime.fromisoformat(day.period).replace(tzinfo=UTC).timestamp()
                * 1_000
            )
            if (
                int(row[2]) != SECONDS_PER_DAY
                or int(row[3]) != SECONDS_PER_DAY
                or int(row[4]) != start_ms
                or int(row[5]) != start_ms + (SECONDS_PER_DAY - 1) * 1_000
                or int(row[6]) != 0
                or int(row[7]) != 0
            ):
                raise ValueError(f"{day.period} {row[1]} fact partition is invalid")

        manifest_fingerprint = _canonical_sha256(
            [
                {
                    "combined_flow_sha256": str(row[4]),
                    "day_id": str(row[0]),
                    "period": str(row[1]),
                }
                for row in manifests
            ]
        )
        source_fingerprint = _canonical_sha256(
            [
                {
                    "day_id": str(row[0]),
                    "flow_sha256": str(row[8]),
                    "market_type": str(row[1]),
                    "source_sha256": str(row[7]),
                    "symbol": str(row[2]),
                }
                for row in sources
            ]
        )
        return {
            "schema_version": "spot-perpetual-corpus-certificate-v1",
            "research_round": SPOT_PERPETUAL_RESEARCH_ROUND,
            "inventory_sha256": inventory_sha256,
            "status": "complete",
            "day_count": len(manifests),
            "source_count": len(sources),
            "symbol_count": len(FLOW_SYMBOLS),
            "flow_rows": sum(int(row[2]) for row in facts),
            "compressed_bytes": sum(int(row[5]) for row in sources),
            "uncompressed_bytes": sum(int(row[11]) for row in manifests),
            "first_period": contract.days[0].period,
            "last_period": contract.days[-1].period,
            "manifest_fingerprint": manifest_fingerprint,
            "source_fingerprint": source_fingerprint,
        }


__all__ = [
    "ROUND72_DESIGN_SCHEMA",
    "ROUND72_INVENTORY_SCHEMA",
    "SPOT_PERPETUAL_CORPUS_SCHEMA",
    "FlowDayIngestResult",
    "FrozenFlowArchive",
    "FrozenFlowContract",
    "FrozenFlowDay",
    "SpotPerpetualCorpusStore",
    "VerifiedFlowSource",
    "load_frozen_round72_contract",
]
