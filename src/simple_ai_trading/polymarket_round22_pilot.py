"""Frozen Round 22 historical-L2 pilot contract and compact feature store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import duckdb

from .polymarket import PolymarketFeeSchedule
from .polymarket_historical_l2 import (
    HistoricalL2Chunk,
    HistoricalL2Window,
    decode_historical_l2_chunk,
    decode_historical_l2_window,
    encode_historical_l2_window,
)
from .polymarket_historical_screen import (
    HistoricalBtcMarket,
    HistoricalScreenContract,
    HistoricalScreenTestGates,
)


POLYMARKET_ROUND22_PILOT_DESIGN_SHA256 = (
    "b37a9475d8b6ad0b4d2e9c53de3756baf44341f6e6d5097579052db7b99d1d2a"
)
POLYMARKET_ROUND22_SOURCE_QUALIFICATION_SHA256 = (
    "3564d59d97899f9defae77c19655e942062342ac96f0f2365b6f9a75f5d7f050"
)
POLYMARKET_ROUND22_STORE_SCHEMA_VERSION = "polymarket-round22-pilot-store-v1"

_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/round-022-historical-l2-pilot-design-v1.json"
)
_QUALIFICATION_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-022-historical-l2-source-qualification-2026-08-03.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SLUG = re.compile(r"^btc-updown-5m-([0-9]{10})$")
_BOOK_HASH = re.compile(r"^[0-9a-f]{40}$")
_ROLES = ("train", "tune_calibration", "tune_selection", "sealed_test")
_DEVELOPMENT_ROLES = frozenset(_ROLES[:3])
_MAXIMUM_ARTIFACT_BYTES = 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 22 JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 JSON contains {value}")


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


def _load_artifact(path: Path, *, hash_field: str) -> tuple[dict[str, object], str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ValueError("Round 22 artifact is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 22 artifact is not strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 22 artifact is not an object")
    payload = dict(decoded)
    claimed = str(payload.pop(hash_field, "")).strip().lower()
    if _SHA256.fullmatch(claimed) is None or claimed != _canonical_sha256(payload):
        raise ValueError("Round 22 artifact SHA-256 differs")
    return payload, claimed


@dataclass(frozen=True, slots=True)
class Round22ExpectedCondition:
    slug: str
    role: str
    event_start_ms: int
    event_end_ms: int


@dataclass(frozen=True, slots=True)
class Round22PilotContract:
    repository: Path
    design_sha256: str
    qualification_sha256: str
    conditions: tuple[Round22ExpectedCondition, ...]
    excluded_slugs: frozenset[str]
    series_id: str

    @property
    def by_slug(self) -> dict[str, Round22ExpectedCondition]:
        return {condition.slug: condition for condition in self.conditions}

    def identity_parser_contract(self) -> HistoricalScreenContract:
        roles = {
            datetime.fromtimestamp(
                condition.event_start_ms / 1_000,
                tz=UTC,
            )
            .date()
            .isoformat(): condition.role
            for condition in self.conditions
        }
        return HistoricalScreenContract(
            path=self.repository / _DESIGN_RELATIVE,
            contract_sha256=self.design_sha256,
            series_id=self.series_id,
            eligible_days=tuple(roles),
            roles=roles,
            excluded_slugs=self.excluded_slugs,
            requested_page_limit=500,
            decision_offsets_seconds=tuple(range(1, 300)),
            return_horizons_seconds=(),
            flow_windows_seconds=(),
            source_inventory_sha256=None,
            source_research_round=22,
            required_source_symbol_count=0,
            required_flow_rows_per_day=0,
            required_market_count_per_day=48,
            test_gates=HistoricalScreenTestGates(
                minimum_terminal_conditions=0,
                minimum_outcomes_per_class=0,
                minimum_decision_rows=0,
                bootstrap_repetitions=0,
                calibration_slope_minimum=0,
                calibration_slope_maximum=0,
                expected_calibration_error_maximum=0,
            ),
        )


def load_round22_pilot_contract(repository: str | Path) -> Round22PilotContract:
    root = Path(repository).resolve()
    qualification, qualification_sha = _load_artifact(
        root / _QUALIFICATION_RELATIVE,
        hash_field="artifact_sha256",
    )
    design, design_sha = _load_artifact(
        root / _DESIGN_RELATIVE,
        hash_field="design_sha256",
    )
    parents = design.get("parents")
    population = design.get("pilot_population")
    data_contract = design.get("data_contract")
    authority = design.get("authority")
    if (
        qualification_sha != POLYMARKET_ROUND22_SOURCE_QUALIFICATION_SHA256
        or design_sha != POLYMARKET_ROUND22_PILOT_DESIGN_SHA256
        or design.get("schema_version")
        != "polymarket-round22-historical-l2-pilot-design-v1"
        or design.get("status") != "frozen_before_bulk_ingestion_or_target_access"
        or not isinstance(parents, Mapping)
        or parents.get("historical_l2_source_qualification_sha256") != qualification_sha
        or not isinstance(population, Mapping)
        or not isinstance(data_contract, Mapping)
        or not isinstance(authority, Mapping)
        or any(authority.values())
    ):
        raise ValueError("Round 22 pilot design differs")
    exclusion = qualification.get("qualification_exclusions")
    if not isinstance(exclusion, Mapping) or not isinstance(
        exclusion.get("slugs"), list
    ):
        raise ValueError("Round 22 qualification exclusions differ")
    excluded = frozenset(str(value) for value in exclusion["slugs"])
    if not excluded or any(_SLUG.fullmatch(slug) is None for slug in excluded):
        raise ValueError("Round 22 qualification exclusion slug differs")
    hours = population.get("hours_utc_per_day")
    partitions = population.get("partitions")
    if hours != [0, 6, 12, 18] or not isinstance(partitions, Mapping):
        raise ValueError("Round 22 pilot clock stratification differs")
    conditions: list[Round22ExpectedCondition] = []
    for role in _ROLES:
        partition = partitions.get(role)
        if not isinstance(partition, Mapping) or not isinstance(
            partition.get("days_utc"), list
        ):
            raise ValueError("Round 22 pilot partition differs")
        role_conditions: list[Round22ExpectedCondition] = []
        for day in partition["days_utc"]:
            if not isinstance(day, str):
                raise ValueError("Round 22 pilot day differs")
            for hour in hours:
                start = int(
                    datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00").timestamp()
                    * 1_000
                )
                for offset in range(12):
                    event_start = start + (offset * 300_000)
                    role_conditions.append(
                        Round22ExpectedCondition(
                            slug=f"btc-updown-5m-{event_start // 1_000}",
                            role=role,
                            event_start_ms=event_start,
                            event_end_ms=event_start + 300_000,
                        )
                    )
        if len(role_conditions) != int(partition.get("market_count_expected", -1)):
            raise ValueError("Round 22 pilot partition count differs")
        conditions.extend(role_conditions)
    slugs = [condition.slug for condition in conditions]
    if (
        len(conditions) != int(population.get("market_count_expected", -1))
        or len(set(slugs)) != len(slugs)
        or set(slugs) & excluded
        or data_contract.get("asset") != "BTC"
        or data_contract.get("polymarket_series_id") != "10684"
        or data_contract.get("target_separation")
        != "separate_target_schema_unavailable_to_feature_materialization"
    ):
        raise ValueError("Round 22 pilot population differs")
    return Round22PilotContract(
        repository=root,
        design_sha256=design_sha,
        qualification_sha256=qualification_sha,
        conditions=tuple(sorted(conditions, key=lambda item: item.event_start_ms)),
        excluded_slugs=excluded,
        series_id="10684",
    )


def validate_round22_market_identity(
    market: HistoricalBtcMarket,
    *,
    contract: Round22PilotContract,
    allow_sealed_test: bool = False,
) -> Round22ExpectedCondition:
    expected = contract.by_slug.get(market.slug)
    if (
        expected is None
        or market.excluded
        or market.role != expected.role
        or market.event_start_ms != expected.event_start_ms
        or market.end_ms != expected.event_end_ms
        or _CONDITION_ID.fullmatch(market.condition_id) is None
        or _TOKEN_ID.fullmatch(market.up_token_id) is None
        or _TOKEN_ID.fullmatch(market.down_token_id) is None
        or market.up_token_id == market.down_token_id
    ):
        raise ValueError("Round 22 market identity differs")
    if expected.role == "sealed_test" and not allow_sealed_test:
        raise ValueError("Round 22 sealed-test identity access is blocked")
    return expected


def _chunk_metadata(chunk: HistoricalL2Chunk) -> dict[str, object]:
    return {
        "codec": chunk.codec,
        "compressed_sha256": chunk.compressed_sha256,
        "compressed_size_bytes": chunk.compressed_size_bytes,
        "raw_sha256": chunk.raw_sha256,
        "raw_size_bytes": chunk.raw_size_bytes,
        "record_count": chunk.record_count,
    }


class Round22PilotStore:
    """One transactional DuckDB containing compressed features and quarantined targets."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract: Round22PilotContract,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.contract = contract
        self.read_only = bool(read_only)
        if self.path.is_symlink():
            raise ValueError("Round 22 pilot database must not be a symlink")
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path), read_only=self.read_only)
        if not self.read_only:
            self._initialize()
        self._verify_manifest()

    def __enter__(self) -> Round22PilotStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE SCHEMA IF NOT EXISTS feature;
            CREATE SCHEMA IF NOT EXISTS target;
            CREATE TABLE IF NOT EXISTS feature.pilot_manifest (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                design_sha256 VARCHAR NOT NULL,
                qualification_sha256 VARCHAR NOT NULL,
                state VARCHAR NOT NULL,
                completed_condition_count BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.market_identity (
                condition_id VARCHAR PRIMARY KEY,
                slug VARCHAR NOT NULL UNIQUE,
                role VARCHAR NOT NULL,
                event_start_ms BIGINT NOT NULL,
                event_end_ms BIGINT NOT NULL,
                up_token_id VARCHAR NOT NULL UNIQUE,
                down_token_id VARCHAR NOT NULL UNIQUE,
                tick_size VARCHAR NOT NULL,
                minimum_order_size VARCHAR NOT NULL,
                fee_rate VARCHAR NOT NULL,
                fee_exponent INTEGER NOT NULL,
                fee_rebate_rate VARCHAR NOT NULL,
                identity_payload_json VARCHAR NOT NULL,
                identity_payload_sha256 VARCHAR NOT NULL,
                source_payload_sha256 VARCHAR NOT NULL,
                observed_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.book_chunk (
                condition_id VARCHAR NOT NULL,
                outcome VARCHAR NOT NULL,
                asset_id VARCHAR NOT NULL UNIQUE,
                record_count BIGINT NOT NULL,
                first_timestamp_ms BIGINT NOT NULL,
                last_timestamp_ms BIGINT NOT NULL,
                source_chain_sha256 VARCHAR NOT NULL,
                codec VARCHAR NOT NULL,
                raw_size_bytes BIGINT NOT NULL,
                compressed_size_bytes BIGINT NOT NULL,
                raw_sha256 VARCHAR NOT NULL,
                compressed_sha256 VARCHAR NOT NULL,
                payload BLOB NOT NULL,
                PRIMARY KEY (condition_id, outcome)
            );
            CREATE TABLE IF NOT EXISTS feature.condition_manifest (
                condition_id VARCHAR PRIMARY KEY,
                slug VARCHAR NOT NULL UNIQUE,
                role VARCHAR NOT NULL,
                up_chunk_sha256 VARCHAR NOT NULL,
                down_chunk_sha256 VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS target.official_resolution (
                condition_id VARCHAR PRIMARY KEY,
                access_claim_sha256 VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                winning_outcome VARCHAR NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            INSERT INTO feature.pilot_manifest
            SELECT true, ?, ?, ?, 'feature_ingestion', 0
            WHERE NOT EXISTS (SELECT 1 FROM feature.pilot_manifest)
            """,
            [
                POLYMARKET_ROUND22_STORE_SCHEMA_VERSION,
                self.contract.design_sha256,
                self.contract.qualification_sha256,
            ],
        )

    def _verify_manifest(self) -> None:
        row = self.connection.execute(
            """
            SELECT schema_version, design_sha256, qualification_sha256, state,
                   completed_condition_count
            FROM feature.pilot_manifest WHERE singleton
            """
        ).fetchone()
        committed = self.connection.execute(
            "SELECT COUNT(*) FROM feature.condition_manifest"
        ).fetchone()
        target_rows = self.connection.execute(
            "SELECT COUNT(*) FROM target.official_resolution"
        ).fetchone()
        if (
            row is None
            or row[0] != POLYMARKET_ROUND22_STORE_SCHEMA_VERSION
            or row[1] != self.contract.design_sha256
            or row[2] != self.contract.qualification_sha256
            or row[3]
            not in {
                "feature_ingestion",
                "diagnostic_targets_open",
                "feature_complete",
                "sealed",
            }
            or type(row[4]) is not int
            or row[4] < 0
            or committed is None
            or int(committed[0]) != row[4]
            or target_rows is None
            or (row[3] == "feature_ingestion" and int(target_rows[0]) != 0)
        ):
            raise ValueError("Round 22 pilot store manifest differs")
        if row[3] == "diagnostic_targets_open":
            claim_table = int(
                self.connection.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'target'
                      AND table_name = 'round22_access_claim'
                    """
                ).fetchone()[0]
            )
            claim_count = (
                0
                if not claim_table
                else int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM target.round22_access_claim"
                    ).fetchone()[0]
                )
            )
            if claim_table != 1 or claim_count != 1 or int(target_rows[0]) > 36:
                raise ValueError("Round 22 diagnostic target claim differs")

    def close(self) -> None:
        if not self.read_only:
            self.connection.execute("CHECKPOINT")
        self.connection.close()

    def completed_slugs(self) -> frozenset[str]:
        rows = self.connection.execute(
            "SELECT slug FROM feature.condition_manifest ORDER BY slug"
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def put_condition(
        self,
        *,
        market: HistoricalBtcMarket,
        up_window: HistoricalL2Window,
        down_window: HistoricalL2Window,
    ) -> bool:
        if self.read_only:
            raise ValueError("Round 22 pilot store is read-only")
        expected = validate_round22_market_identity(market, contract=self.contract)
        try:
            identity_payload = json.loads(
                market.identity_payload_json,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 22 market identity payload is invalid") from exc
        if (
            not isinstance(identity_payload, Mapping)
            or hashlib.sha256(market.identity_payload_json.encode("ascii")).hexdigest()
            != market.identity_payload_sha256
            or _SHA256.fullmatch(market.source_payload_sha256) is None
        ):
            raise ValueError("Round 22 market identity evidence differs")
        for outcome, token, window in (
            ("Up", market.up_token_id, up_window),
            ("Down", market.down_token_id, down_window),
        ):
            if (
                window.condition_id != market.condition_id
                or window.asset_id != token
                or window.event_start_ms != expected.event_start_ms
                or window.event_end_ms != expected.event_end_ms
                or not window.snapshots
                or _SHA256.fullmatch(window.source_chain_sha256) is None
            ):
                raise ValueError(f"Round 22 {outcome} book window differs")
            previous_timestamp = expected.event_start_ms - 1
            for snapshot in window.snapshots:
                if (
                    snapshot.condition_id != market.condition_id
                    or snapshot.asset_id != token
                    or not expected.event_start_ms
                    <= snapshot.timestamp_ms
                    < expected.event_end_ms
                    or snapshot.timestamp_ms <= previous_timestamp
                    or _BOOK_HASH.fullmatch(snapshot.book_hash) is None
                    or _SHA256.fullmatch(snapshot.source_payload_sha256) is None
                ):
                    raise ValueError(f"Round 22 {outcome} book snapshot differs")
                previous_timestamp = snapshot.timestamp_ms
        up_chunk = encode_historical_l2_window(up_window)
        down_chunk = encode_historical_l2_window(down_window)
        manifest_body = {
            "condition_id": market.condition_id,
            "design_sha256": self.contract.design_sha256,
            "down": _chunk_metadata(down_chunk),
            "identity_payload_sha256": market.identity_payload_sha256,
            "role": market.role,
            "slug": market.slug,
            "up": _chunk_metadata(up_chunk),
        }
        manifest_sha = _canonical_sha256(manifest_body)
        existing = self.connection.execute(
            "SELECT manifest_sha256 FROM feature.condition_manifest WHERE condition_id = ?",
            [market.condition_id],
        ).fetchone()
        if existing is not None:
            if existing[0] != manifest_sha:
                raise ValueError("Round 22 existing condition manifest differs")
            return False
        transaction_started = False
        try:
            self.connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            self.connection.execute(
                """
                INSERT INTO feature.market_identity VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    market.condition_id,
                    market.slug,
                    market.role,
                    market.event_start_ms,
                    market.end_ms,
                    market.up_token_id,
                    market.down_token_id,
                    str(market.tick_size),
                    str(market.minimum_order_size),
                    str(market.fee_schedule.rate),
                    market.fee_schedule.exponent,
                    str(market.fee_schedule.rebate_rate),
                    market.identity_payload_json,
                    market.identity_payload_sha256,
                    market.source_payload_sha256,
                    market.observed_at_ms,
                ],
            )
            for outcome, window, chunk in (
                ("Up", up_window, up_chunk),
                ("Down", down_window, down_chunk),
            ):
                self.connection.execute(
                    """
                    INSERT INTO feature.book_chunk VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        market.condition_id,
                        outcome,
                        window.asset_id,
                        chunk.record_count,
                        window.snapshots[0].timestamp_ms,
                        window.snapshots[-1].timestamp_ms,
                        window.source_chain_sha256,
                        chunk.codec,
                        chunk.raw_size_bytes,
                        chunk.compressed_size_bytes,
                        chunk.raw_sha256,
                        chunk.compressed_sha256,
                        chunk.payload,
                    ],
                )
            self.connection.execute(
                "INSERT INTO feature.condition_manifest VALUES (?, ?, ?, ?, ?, ?)",
                [
                    market.condition_id,
                    market.slug,
                    market.role,
                    up_chunk.compressed_sha256,
                    down_chunk.compressed_sha256,
                    manifest_sha,
                ],
            )
            self.connection.execute(
                """
                UPDATE feature.pilot_manifest
                SET completed_condition_count = completed_condition_count + 1
                WHERE singleton
                """
            )
            self.connection.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                self.connection.execute("ROLLBACK")
            raise
        return True

    def audit_condition(self, condition_id: str) -> dict[str, object]:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 22 audit condition ID is invalid")
        identity = self.connection.execute(
            """
            SELECT slug, role, identity_payload_sha256, identity_payload_json
            FROM feature.market_identity WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        chunks = self.connection.execute(
            """
            SELECT outcome, asset_id, codec, record_count, raw_size_bytes,
                   compressed_size_bytes, raw_sha256, compressed_sha256, payload,
                   source_chain_sha256
            FROM feature.book_chunk WHERE condition_id = ? ORDER BY outcome
            """,
            [selected],
        ).fetchall()
        manifest = self.connection.execute(
            """
            SELECT slug, role, up_chunk_sha256, down_chunk_sha256, manifest_sha256
            FROM feature.condition_manifest WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        if identity is None or manifest is None or len(chunks) != 2:
            raise ValueError("Round 22 condition is incomplete")
        decoded: dict[str, dict[str, object]] = {}
        compressed_by_outcome: dict[str, str] = {}
        chunk_metadata_by_outcome: dict[str, dict[str, object]] = {}
        for row in chunks:
            outcome = str(row[0])
            chunk = HistoricalL2Chunk(
                codec=str(row[2]),
                record_count=int(row[3]),
                raw_size_bytes=int(row[4]),
                compressed_size_bytes=int(row[5]),
                raw_sha256=str(row[6]),
                compressed_sha256=str(row[7]),
                payload=bytes(row[8]),
            )
            decoded[outcome] = decode_historical_l2_chunk(chunk)
            compressed_by_outcome[outcome] = chunk.compressed_sha256
            chunk_metadata_by_outcome[outcome] = _chunk_metadata(chunk)
            source = decoded[outcome].get("source")
            if (
                decoded[outcome].get("condition_id") != selected
                or decoded[outcome].get("asset_id") != str(row[1])
                or not isinstance(source, Mapping)
                or source.get("source_chain_sha256") != str(row[9])
            ):
                raise ValueError("Round 22 decoded book linkage differs")
        try:
            identity_payload_sha = hashlib.sha256(
                str(identity[3]).encode("ascii")
            ).hexdigest()
        except UnicodeError as exc:
            raise ValueError("Round 22 identity payload is not ASCII") from exc
        manifest_body = {
            "condition_id": selected,
            "design_sha256": self.contract.design_sha256,
            "down": chunk_metadata_by_outcome.get("Down"),
            "identity_payload_sha256": str(identity[2]),
            "role": str(identity[1]),
            "slug": str(identity[0]),
            "up": chunk_metadata_by_outcome.get("Up"),
        }
        if (
            identity_payload_sha != identity[2]
            or manifest[0] != identity[0]
            or manifest[1] != identity[1]
            or manifest[2] != compressed_by_outcome.get("Up")
            or manifest[3] != compressed_by_outcome.get("Down")
            or manifest[4] != _canonical_sha256(manifest_body)
        ):
            raise ValueError("Round 22 condition manifest linkage differs")
        return {
            "condition_id": selected,
            "down_record_count": len(decoded["Down"]["snapshots"]),
            "manifest_sha256": str(manifest[4]),
            "role": str(identity[1]),
            "slug": str(identity[0]),
            "up_record_count": len(decoded["Up"]["snapshots"]),
        }

    def condition_windows(
        self,
        condition_id: str,
    ) -> tuple[HistoricalL2Window, HistoricalL2Window]:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 22 condition-window ID is invalid")
        identity = self.connection.execute(
            """
            SELECT up_token_id, down_token_id, event_start_ms, event_end_ms
            FROM feature.market_identity WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        rows = self.connection.execute(
            """
            SELECT outcome, asset_id, record_count, first_timestamp_ms,
                   last_timestamp_ms, source_chain_sha256, codec,
                   raw_size_bytes, compressed_size_bytes, raw_sha256,
                   compressed_sha256, payload
            FROM feature.book_chunk WHERE condition_id = ? ORDER BY outcome
            """,
            [selected],
        ).fetchall()
        if identity is None or len(rows) != 2:
            raise ValueError("Round 22 condition windows are incomplete")
        windows: dict[str, HistoricalL2Window] = {}
        for row in rows:
            outcome = str(row[0])
            if outcome not in {"Up", "Down"} or outcome in windows:
                raise ValueError("Round 22 condition-window outcome differs")
            chunk = HistoricalL2Chunk(
                codec=str(row[6]),
                record_count=int(row[2]),
                raw_size_bytes=int(row[7]),
                compressed_size_bytes=int(row[8]),
                raw_sha256=str(row[9]),
                compressed_sha256=str(row[10]),
                payload=bytes(row[11]),
            )
            window = decode_historical_l2_window(chunk)
            expected_token = str(identity[0] if outcome == "Up" else identity[1])
            if (
                window.condition_id != selected
                or window.asset_id != expected_token
                or window.asset_id != str(row[1])
                or window.event_start_ms != int(identity[2])
                or window.event_end_ms != int(identity[3])
                or len(window.snapshots) != int(row[2])
                or window.snapshots[0].timestamp_ms != int(row[3])
                or window.snapshots[-1].timestamp_ms != int(row[4])
                or window.source_chain_sha256 != str(row[5])
            ):
                raise ValueError("Round 22 condition-window linkage differs")
            windows[outcome] = window
        return windows["Up"], windows["Down"]

    def market(self, condition_id: str) -> HistoricalBtcMarket:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 22 market condition ID is invalid")
        row = self.connection.execute(
            """
            SELECT condition_id, slug, role, event_start_ms, event_end_ms,
                   up_token_id, down_token_id,
                   tick_size, minimum_order_size, fee_rate, fee_exponent,
                   fee_rebate_rate, identity_payload_json,
                   identity_payload_sha256, source_payload_sha256,
                   observed_at_ms
            FROM feature.market_identity WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise ValueError("Round 22 market is unavailable")
        try:
            identity = json.loads(
                str(row[12]),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 22 stored market identity is invalid") from exc
        identity_market = (
            identity.get("market") if isinstance(identity, Mapping) else None
        )
        event_id = (
            str(identity.get("event_id") or "") if isinstance(identity, Mapping) else ""
        )
        market_id = (
            str(identity_market.get("id") or "")
            if isinstance(identity_market, Mapping)
            else ""
        )
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity_market, Mapping)
            or not event_id.isdigit()
            or not market_id.isdigit()
            or str(identity.get("role") or "") != str(row[2])
            or str(identity_market.get("conditionId") or "").strip().lower()
            != str(row[0])
            or str(identity_market.get("slug") or "") != str(row[1])
            or hashlib.sha256(str(row[12]).encode("ascii")).hexdigest() != str(row[13])
            or _SHA256.fullmatch(str(row[14])) is None
        ):
            raise ValueError("Round 22 stored market identity differs")
        market = HistoricalBtcMarket(
            event_id=event_id,
            market_id=market_id,
            condition_id=str(row[0]),
            slug=str(row[1]),
            question=str(identity_market.get("question") or ""),
            event_start_ms=int(row[3]),
            end_ms=int(row[4]),
            role=str(row[2]),
            up_token_id=str(row[5]),
            down_token_id=str(row[6]),
            tick_size=Decimal(str(row[7])),
            minimum_order_size=Decimal(str(row[8])),
            fee_schedule=PolymarketFeeSchedule(
                enabled=True,
                rate=Decimal(str(row[9])),
                exponent=int(row[10]),
                taker_only=True,
                rebate_rate=Decimal(str(row[11])),
            ),
            excluded=False,
            exclusion_reason="",
            identity_payload_json=str(row[12]),
            identity_payload_sha256=str(row[13]),
            source_payload_sha256=str(row[14]),
            observed_at_ms=int(row[15]),
        )
        validate_round22_market_identity(market, contract=self.contract)
        return market

    def feature_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT role, COUNT(*) FROM feature.condition_manifest
            GROUP BY role ORDER BY role
            """
        ).fetchall()
        return {str(role): int(count) for role, count in rows}

    def target_row_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM target.official_resolution"
            ).fetchone()[0]
        )


def development_conditions(
    contract: Round22PilotContract,
) -> tuple[Round22ExpectedCondition, ...]:
    return tuple(
        condition
        for condition in contract.conditions
        if condition.role in _DEVELOPMENT_ROLES
    )


Round22Role = Literal[
    "train",
    "tune_calibration",
    "tune_selection",
    "sealed_test",
]


__all__ = [
    "POLYMARKET_ROUND22_PILOT_DESIGN_SHA256",
    "POLYMARKET_ROUND22_SOURCE_QUALIFICATION_SHA256",
    "POLYMARKET_ROUND22_STORE_SCHEMA_VERSION",
    "Round22ExpectedCondition",
    "Round22PilotContract",
    "Round22PilotStore",
    "Round22Role",
    "development_conditions",
    "load_round22_pilot_contract",
    "validate_round22_market_identity",
]
