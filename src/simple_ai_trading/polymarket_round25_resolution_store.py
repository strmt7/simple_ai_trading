"""Separate, resumable, dual-source official targets for Round 25."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Protocol
from urllib.parse import quote, urlparse

import duckdb
import requests
import zstandard

from .polymarket_resolution import validate_official_resolution
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
    POLYMARKET_ROUND25_SELECTION_END_MS,
    Round25OfficialResolution,
    Round25ResolutionAuthority,
)
from .polymarket_round25_evaluation import (
    Round25PredictionPanel,
    Round25SelectionAccessStore,
)
from .polymarket_round25_joint_materialization import Round25JointReceiptCondition
from .polymarket_round25_joint_store import (
    POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION,
    audit_round25_joint_store,
    load_round25_joint_condition_identities,
)


POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256 = (
    "bc2e4f462ac45b99340872a0db1ad3078e3a53626f07ab1aedce0e65a0023a36"
)
POLYMARKET_ROUND25_RESOLUTION_ACCESS_CLAIM_SCHEMA_VERSION = (
    "polymarket-round25-resolution-access-claim-v1"
)
POLYMARKET_ROUND25_RESOLUTION_EVIDENCE_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-evidence-v1"
)
POLYMARKET_ROUND25_RESOLUTION_AUDIT_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-audit-v1"
)
POLYMARKET_ROUND25_RESOLUTION_STORE_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-store-v1"
)
POLYMARKET_ROUND25_RESOLUTION_STORE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-store-manifest-v1"
)
POLYMARKET_ROUND25_RESOLUTION_COLLECTION_REPORT_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-collection-report-v1"
)
POLYMARKET_ROUND25_RESOLUTION_CODEC = "canonical-json-zstd-3"
POLYMARKET_ROUND25_MAXIMUM_OFFICIAL_PAYLOAD_BYTES = 2 * 1024 * 1024
POLYMARKET_ROUND25_GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_ROUND25_CLOB_MARKET_URL = "https://clob.polymarket.com/markets"

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_MARKET_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_ROLES = ("train", "calibration", "selection")
_COLLECTION_TABLES = {
    "round25_resolution_access_claim",
    "round25_resolution_evidence",
    "round25_resolution_source_condition",
}
_FINAL_TABLES = _COLLECTION_TABLES | {
    "round25_official_resolution",
    "round25_resolution_authority",
    "round25_resolution_store_manifest",
}
_RETRYABLE_STATUS_CODES = frozenset((425, 429, 500, 502, 503, 504))
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "poly_address",
        "poly_signature",
        "poly_timestamp",
        "poly_nonce",
        "poly_api_key",
        "poly_passphrase",
    }
)
_REPLACE_RETRY_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str
    request: object


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


class Round25ResolutionTransportError(RuntimeError):
    """A bounded public transport failure that may be retried later."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 resolution JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 resolution JSON contains {value}")


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


def _hash_chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _strict_json(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 {label} is not strict JSON") from exc
    if not isinstance(decoded, Mapping) or _canonical_json(decoded) != value:
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    return decoded


def _table_names(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    if any(str(row[1]) != "BASE TABLE" for row in rows):
        raise ValueError("Round 25 resolution store contains a non-table object")
    return {str(row[0]) for row in rows}


def _replace_with_retries(source: Path, destination: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_SECONDS):
                raise
            time.sleep(_REPLACE_RETRY_SECONDS[attempt])


def _condition_identity_payload(
    condition: Round25JointReceiptCondition,
) -> dict[str, object]:
    selected = condition.validated()
    return {
        "condition_id": selected.condition_id,
        "down_token_id": selected.down_token_id,
        "event_end_ms": selected.event_end_ms,
        "event_start_ms": selected.event_start_ms,
        "market_id": selected.market_id,
        "resolution_source": selected.resolution_source,
        "role": selected.role,
        "run_id": selected.run_id,
        "segment_index": selected.segment_index,
        "slug": selected.slug,
        "source_snapshot_observed_wall_ms": selected.snapshot_observed_wall_ms,
        "source_snapshot_sha256": selected.snapshot_sha256,
        "up_token_id": selected.up_token_id,
    }


def _condition_from_payload(value: Mapping[str, object]) -> Round25JointReceiptCondition:
    expected = {
        "condition_id",
        "down_token_id",
        "event_end_ms",
        "event_start_ms",
        "market_id",
        "resolution_source",
        "role",
        "run_id",
        "segment_index",
        "slug",
        "source_snapshot_observed_wall_ms",
        "source_snapshot_sha256",
        "up_token_id",
    }
    if (
        set(value) != expected
        or not isinstance(value.get("condition_id"), str)
        or not isinstance(value.get("down_token_id"), str)
        or type(value.get("event_end_ms")) is not int
        or type(value.get("event_start_ms")) is not int
        or not isinstance(value.get("market_id"), str)
        or not isinstance(value.get("resolution_source"), str)
        or not isinstance(value.get("role"), str)
        or not isinstance(value.get("run_id"), str)
        or type(value.get("segment_index")) is not int
        or not isinstance(value.get("slug"), str)
        or type(value.get("source_snapshot_observed_wall_ms")) is not int
        or not isinstance(value.get("source_snapshot_sha256"), str)
        or not isinstance(value.get("up_token_id"), str)
    ):
        raise ValueError("Round 25 resolution source condition differs")
    return Round25JointReceiptCondition(
        run_id=value["run_id"],
        segment_index=value["segment_index"],
        snapshot_sha256=value["source_snapshot_sha256"],
        snapshot_observed_wall_ms=value["source_snapshot_observed_wall_ms"],
        market_id=value["market_id"],
        condition_id=value["condition_id"],
        slug=value["slug"],
        event_start_ms=value["event_start_ms"],
        event_end_ms=value["event_end_ms"],
        up_token_id=value["up_token_id"],
        down_token_id=value["down_token_id"],
        resolution_source=value["resolution_source"],
        role=value["role"],
    ).validated()


def _condition_population_sha256(
    conditions: Sequence[Round25JointReceiptCondition],
) -> str:
    chain = _EMPTY_SHA256
    previous: tuple[int, str] | None = None
    for condition in conditions:
        selected = condition.validated()
        identity = (selected.event_start_ms, selected.condition_id)
        if previous is not None and identity <= previous:
            raise ValueError("Round 25 resolution condition chronology differs")
        previous = identity
        chain = _hash_chain(chain, _condition_identity_payload(selected))
    if not conditions or chain == _EMPTY_SHA256:
        raise ValueError("Round 25 resolution condition population is empty")
    return chain


@dataclass(frozen=True, slots=True)
class Round25OfficialPublicPayload:
    value: Mapping[str, object]
    canonical_json: str
    sha256: str
    observed_wall_ms: int
    observed_monotonic_ns: int


class Round25ResolutionPublicClient:
    """Credential-free, paced, bounded client for the two official endpoints."""

    def __init__(
        self,
        *,
        session: _Session | None = None,
        timeout_seconds: float = 20.0,
        minimum_request_interval_seconds: float = 0.2,
        maximum_attempts: int = 4,
        clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = max(2.0, min(60.0, float(timeout_seconds)))
        self.minimum_request_interval_seconds = max(
            0.0,
            min(5.0, float(minimum_request_interval_seconds)),
        )
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 8:
            raise ValueError("Round 25 resolution attempts are outside the bound")
        self.maximum_attempts = maximum_attempts
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic_ns = monotonic_ns or time.monotonic_ns
        self.monotonic = monotonic or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._last_request_by_origin: dict[str, float] = {}

    def _assert_public_session(self) -> None:
        names = {str(name).strip().lower() for name in self.session.headers}
        if names & _SENSITIVE_HEADERS or any(name.startswith("poly_") for name in names):
            raise ValueError("Round 25 resolution session contains authority headers")
        cookies = getattr(self.session, "cookies", None)
        if cookies is not None and len(cookies):
            raise ValueError("Round 25 resolution session contains cookies")

    def _discard_response_cookies(self, response: _Response) -> None:
        request = getattr(response, "request", None)
        request_headers = getattr(request, "headers", None)
        if not isinstance(request_headers, Mapping):
            raise ValueError("Round 25 resolution request metadata is unavailable")
        sent_cookie = any(
            str(name).strip().lower() == "cookie" for name in request_headers
        )
        cookies = getattr(self.session, "cookies", None)
        if cookies is not None:
            clear = getattr(cookies, "clear", None)
            if not callable(clear):
                raise ValueError("Round 25 resolution cookie jar cannot be cleared")
            clear()
            if len(cookies):
                raise ValueError("Round 25 resolution response cookies persisted")
        if sent_cookie:
            raise ValueError("Round 25 resolution request sent cookies")

    def _wait(self, origin: str) -> None:
        now = float(self.monotonic())
        previous = self._last_request_by_origin.get(origin)
        if previous is not None:
            delay = self.minimum_request_interval_seconds - (now - previous)
            if delay > 0:
                self.sleeper(delay)
                now = float(self.monotonic())
        self._last_request_by_origin[origin] = now

    @staticmethod
    def _retry_after(response: _Response, attempt: int) -> float:
        fallback = min(8.0, 0.5 * (2**attempt))
        value = str(response.headers.get("Retry-After") or "").strip()
        if not value.isdigit():
            return fallback
        return max(fallback, min(30.0, float(value)))

    def _request(self, url: str, *, origin: str, path: str) -> Round25OfficialPublicPayload:
        for attempt in range(self.maximum_attempts):
            self._assert_public_session()
            self._wait(origin)
            try:
                response = self.session.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "simple-ai-trading-round25-resolution/0.1",
                    },
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                cookies = getattr(self.session, "cookies", None)
                clear = getattr(cookies, "clear", None)
                if callable(clear):
                    clear()
                if attempt + 1 == self.maximum_attempts:
                    raise Round25ResolutionTransportError(
                        "Round 25 resolution transport retries were exhausted"
                    ) from exc
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
                continue
            self._discard_response_cookies(response)
            status = int(response.status_code)
            if status == 200:
                parsed_url = urlparse(str(response.url))
                content = bytes(response.content)
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if (
                    parsed_url.scheme != "https"
                    or parsed_url.netloc.lower() != origin
                    or parsed_url.path != path
                    or not content_type.startswith("application/json")
                    or not 2 <= len(content) <= POLYMARKET_ROUND25_MAXIMUM_OFFICIAL_PAYLOAD_BYTES
                ):
                    raise ValueError("Round 25 resolution response boundary differs")
                try:
                    decoded = json.loads(
                        content.decode("utf-8"),
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_nonfinite,
                    )
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Round 25 resolution response is not strict JSON") from exc
                if not isinstance(decoded, Mapping):
                    raise ValueError("Round 25 resolution response is not an object")
                canonical = _canonical_json(decoded)
                return Round25OfficialPublicPayload(
                    value=dict(decoded),
                    canonical_json=canonical,
                    sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
                    observed_wall_ms=int(self.clock_ms()),
                    observed_monotonic_ns=int(self.monotonic_ns()),
                )
            if status not in _RETRYABLE_STATUS_CODES:
                raise ValueError(
                    f"Round 25 resolution request failed with HTTP {status}"
                )
            if attempt + 1 == self.maximum_attempts:
                raise Round25ResolutionTransportError(
                    "Round 25 resolution HTTP retries were exhausted"
                )
            self.sleeper(self._retry_after(response, attempt))
        raise AssertionError("unreachable Round 25 resolution retry state")

    def gamma_market(self, market_id: str) -> Round25OfficialPublicPayload:
        selected = str(market_id or "").strip()
        if _MARKET_ID.fullmatch(selected) is None:
            raise ValueError("Round 25 resolution Gamma market ID is invalid")
        encoded = quote(selected, safe="")
        path = f"/markets/{encoded}"
        return self._request(
            f"{POLYMARKET_ROUND25_GAMMA_MARKET_URL}/{encoded}",
            origin="gamma-api.polymarket.com",
            path=path,
        )

    def clob_market(self, condition_id: str) -> Round25OfficialPublicPayload:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 25 resolution CLOB condition ID is invalid")
        path = f"/markets/{selected}"
        return self._request(
            f"{POLYMARKET_ROUND25_CLOB_MARKET_URL}/{selected}",
            origin="clob.polymarket.com",
            path=path,
        )


def round25_resolution_collection_database(destination: str | Path) -> Path:
    selected = Path(destination)
    return selected.with_name(f".{selected.name}.collecting")


def validate_round25_resolution_access_claim(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("claim_sha256", "")).strip().lower()
    expected = {
        "condition_count",
        "condition_population_sha256",
        "contract_sha256",
        "created_at_ms",
        "feature_store_manifest_sha256",
        "feature_store_schema_version",
        "live_trading_authority",
        "model_data_eligible",
        "paper_trading_authority",
        "profitability_claim",
        "schema_version",
        "target_access_opened",
        "terminal_receipt_audit_sha256",
        "terminal_transport_manifest_sha256",
    }
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_RESOLUTION_ACCESS_CLAIM_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256
        or payload.get("feature_store_schema_version")
        != POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= POLYMARKET_ROUND25_SELECTION_END_MS
        or type(payload.get("condition_count")) is not int
        or payload["condition_count"] <= 0
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in (
                "condition_population_sha256",
                "feature_store_manifest_sha256",
                "terminal_receipt_audit_sha256",
                "terminal_transport_manifest_sha256",
            )
        )
        or payload.get("target_access_opened") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 25 resolution access claim differs")
    return {**payload, "claim_sha256": claimed}


def _load_claim(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    rows = connection.execute(
        "SELECT claim_json, claim_sha256 FROM round25_resolution_access_claim"
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("Round 25 resolution access claim row differs")
    claim = validate_round25_resolution_access_claim(
        _strict_json(rows[0][0], label="resolution access claim")
    )
    if rows[0][1] != claim["claim_sha256"]:
        raise ValueError("Round 25 resolution access claim hash differs")
    return claim


def _load_source_conditions(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[Round25JointReceiptCondition, ...]:
    rows = connection.execute(
        """
        SELECT condition_id, event_start_ms, role, identity_json, identity_sha256
        FROM round25_resolution_source_condition
        ORDER BY event_start_ms, condition_id
        """
    ).fetchall()
    output: list[Round25JointReceiptCondition] = []
    for row in rows:
        payload = _strict_json(row[3], label="resolution source condition")
        condition = _condition_from_payload(payload)
        if (
            row[0] != condition.condition_id
            or row[1] != condition.event_start_ms
            or row[2] != condition.role
            or row[4] != _canonical_sha256(payload)
        ):
            raise ValueError("Round 25 resolution source condition row differs")
        output.append(condition)
    return tuple(output)


def initialize_round25_resolution_collection(
    *,
    feature_database: str | Path,
    destination_database: str | Path,
    created_at_ms: int,
) -> tuple[Path, dict[str, object]]:
    """Persist the target-access claim before any public target request."""

    feature = Path(feature_database)
    destination = Path(destination_database)
    collection = round25_resolution_collection_database(destination)
    initializing = collection.with_name(f".{collection.name}.initializing")
    resolved_paths = {
        feature.resolve(strict=False),
        destination.resolve(strict=False),
        collection.resolve(strict=False),
    }
    if (
        len(resolved_paths) != 3
        or destination.is_symlink()
        or destination.exists()
        or collection.is_symlink()
        or initializing.is_symlink()
        or initializing.exists()
        or Path(f"{initializing}.wal").exists()
    ):
        raise ValueError("Round 25 resolution database boundary differs")
    feature_before = feature.stat()
    feature_manifest = audit_round25_joint_store(feature)
    conditions = load_round25_joint_condition_identities(feature)
    population_sha256 = _condition_population_sha256(conditions)
    feature_after = feature.stat()
    if (
        feature_before.st_size != feature_after.st_size
        or feature_before.st_mtime_ns != feature_after.st_mtime_ns
        or population_sha256 != feature_manifest["condition_population_sha256"]
        or Path(f"{feature}.wal").exists()
    ):
        raise RuntimeError("Round 25 feature store changed during target opening")
    if collection.exists():
        with duckdb.connect(str(collection), read_only=True) as connection:
            if _table_names(connection) not in (_COLLECTION_TABLES, _FINAL_TABLES):
                raise ValueError("Round 25 resolution collection schema differs")
            claim = _load_claim(connection)
            stored = _load_source_conditions(connection)
        if (
            claim["feature_store_manifest_sha256"]
            != feature_manifest["manifest_sha256"]
            or claim["condition_population_sha256"] != population_sha256
            or _condition_population_sha256(stored) != population_sha256
        ):
            raise ValueError("Round 25 resolution collection source drifted")
        return collection, claim
    body: dict[str, object] = {
        "condition_count": len(conditions),
        "condition_population_sha256": population_sha256,
        "contract_sha256": POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256,
        "created_at_ms": created_at_ms,
        "feature_store_manifest_sha256": feature_manifest["manifest_sha256"],
        "feature_store_schema_version": POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION,
        "live_trading_authority": False,
        "model_data_eligible": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "schema_version": POLYMARKET_ROUND25_RESOLUTION_ACCESS_CLAIM_SCHEMA_VERSION,
        "target_access_opened": True,
        "terminal_receipt_audit_sha256": feature_manifest[
            "terminal_receipt_audit_sha256"
        ],
        "terminal_transport_manifest_sha256": feature_manifest[
            "terminal_transport_manifest_sha256"
        ],
    }
    claim = validate_round25_resolution_access_claim(
        {**body, "claim_sha256": _canonical_sha256(body)}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(initializing))
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        connection.execute("PRAGMA enable_checkpoint_on_shutdown")
        connection.execute(
            """
            CREATE TABLE round25_resolution_access_claim (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                claim_json VARCHAR NOT NULL,
                claim_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_resolution_source_condition (
                condition_id VARCHAR PRIMARY KEY,
                event_start_ms BIGINT NOT NULL,
                role VARCHAR NOT NULL,
                identity_json VARCHAR NOT NULL,
                identity_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_resolution_evidence (
                condition_id VARCHAR PRIMARY KEY,
                evidence_json VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                gamma_payload BLOB NOT NULL,
                clob_payload BLOB NOT NULL
            );
            """
        )
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "INSERT INTO round25_resolution_access_claim VALUES (TRUE, ?, ?)",
            [_canonical_json(claim), claim["claim_sha256"]],
        )
        for condition in conditions:
            payload = _condition_identity_payload(condition)
            connection.execute(
                "INSERT INTO round25_resolution_source_condition VALUES (?, ?, ?, ?, ?)",
                [
                    condition.condition_id,
                    condition.event_start_ms,
                    condition.role,
                    _canonical_json(payload),
                    _canonical_sha256(payload),
                ],
            )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        if Path(f"{initializing}.wal").exists() or collection.exists():
            raise RuntimeError("Round 25 resolution claim file state differs")
        _replace_with_retries(initializing, collection)
    except Exception:
        if connection is not None:
            try:
                connection.execute("ROLLBACK")
            except duckdb.Error:
                pass
            connection.close()
        for path in (initializing, Path(f"{initializing}.wal")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return collection, claim


def load_round25_resolution_access_claim(
    collection_database: str | Path,
) -> dict[str, object]:
    path = Path(collection_database)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Round 25 resolution collection is unavailable")
    with duckdb.connect(str(path), read_only=True) as connection:
        if _table_names(connection) not in (_COLLECTION_TABLES, _FINAL_TABLES):
            raise ValueError("Round 25 resolution collection schema differs")
        return _load_claim(connection)


def _compress_payload(canonical_json: str) -> bytes:
    raw = canonical_json.encode("ascii")
    if not 2 <= len(raw) <= POLYMARKET_ROUND25_MAXIMUM_OFFICIAL_PAYLOAD_BYTES:
        raise ValueError("Round 25 official payload size is outside the bound")
    return zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)


def _decompress_payload(
    compressed: bytes,
    *,
    raw_size: int,
    compressed_sha256: str,
    payload_sha256: str,
    label: str,
) -> tuple[str, Mapping[str, object]]:
    if (
        type(raw_size) is not int
        or not 2 <= raw_size <= POLYMARKET_ROUND25_MAXIMUM_OFFICIAL_PAYLOAD_BYTES
        or _SHA256.fullmatch(compressed_sha256) is None
        or hashlib.sha256(compressed).hexdigest() != compressed_sha256
        or _SHA256.fullmatch(payload_sha256) is None
    ):
        raise ValueError(f"Round 25 {label} envelope differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=raw_size,
        )
    except zstandard.ZstdError as exc:
        raise ValueError(f"Round 25 {label} is not valid zstd") from exc
    if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != payload_sha256:
        raise ValueError(f"Round 25 {label} content differs")
    try:
        canonical = raw.decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"Round 25 {label} is not canonical ASCII") from exc
    return canonical, _strict_json(canonical, label=label)


def _evidence_payload(
    *,
    condition: Round25JointReceiptCondition,
    gamma: Round25OfficialPublicPayload,
    clob: Round25OfficialPublicPayload,
    winning_token_id: str,
    winning_outcome: str,
    gamma_compressed: bytes,
    clob_compressed: bytes,
) -> dict[str, object]:
    resolved_at_ms = max(gamma.observed_wall_ms, clob.observed_wall_ms)
    official_payload_sha256 = _canonical_sha256(
        {
            "clob_payload_sha256": clob.sha256,
            "gamma_payload_sha256": gamma.sha256,
        }
    )
    body = {
        "clob_compressed_sha256": hashlib.sha256(clob_compressed).hexdigest(),
        "clob_observed_monotonic_ns": clob.observed_monotonic_ns,
        "clob_observed_wall_ms": clob.observed_wall_ms,
        "clob_payload_sha256": clob.sha256,
        "clob_raw_size_bytes": len(clob.canonical_json.encode("ascii")),
        "condition_id": condition.condition_id,
        "contract_sha256": POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256,
        "gamma_compressed_sha256": hashlib.sha256(gamma_compressed).hexdigest(),
        "gamma_observed_monotonic_ns": gamma.observed_monotonic_ns,
        "gamma_observed_wall_ms": gamma.observed_wall_ms,
        "gamma_payload_sha256": gamma.sha256,
        "gamma_raw_size_bytes": len(gamma.canonical_json.encode("ascii")),
        "live_trading_authority": False,
        "official_payload_sha256": official_payload_sha256,
        "paper_trading_authority": False,
        "resolved_at_ms": resolved_at_ms,
        "schema_version": POLYMARKET_ROUND25_RESOLUTION_EVIDENCE_SCHEMA_VERSION,
        "source_condition_sha256": _canonical_sha256(
            _condition_identity_payload(condition)
        ),
        "winning_outcome": winning_outcome,
        "winning_token_id": winning_token_id,
    }
    return {**body, "evidence_sha256": _canonical_sha256(body)}


def _decode_evidence(
    *,
    condition: Round25JointReceiptCondition,
    evidence_json: object,
    evidence_sha256: object,
    gamma_compressed: object,
    clob_compressed: object,
) -> tuple[dict[str, object], Mapping[str, object], Mapping[str, object]]:
    payload = dict(_strict_json(evidence_json, label="resolution evidence"))
    claimed = str(payload.pop("evidence_sha256", "")).strip().lower()
    expected = {
        "clob_compressed_sha256",
        "clob_observed_monotonic_ns",
        "clob_observed_wall_ms",
        "clob_payload_sha256",
        "clob_raw_size_bytes",
        "condition_id",
        "contract_sha256",
        "gamma_compressed_sha256",
        "gamma_observed_monotonic_ns",
        "gamma_observed_wall_ms",
        "gamma_payload_sha256",
        "gamma_raw_size_bytes",
        "live_trading_authority",
        "official_payload_sha256",
        "paper_trading_authority",
        "resolved_at_ms",
        "schema_version",
        "source_condition_sha256",
        "winning_outcome",
        "winning_token_id",
    }
    integer_fields = (
        "clob_observed_monotonic_ns",
        "clob_observed_wall_ms",
        "clob_raw_size_bytes",
        "gamma_observed_monotonic_ns",
        "gamma_observed_wall_ms",
        "gamma_raw_size_bytes",
        "resolved_at_ms",
    )
    if (
        set(payload) != expected
        or evidence_sha256 != claimed
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_RESOLUTION_EVIDENCE_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256
        or payload.get("condition_id") != condition.condition_id
        or payload.get("source_condition_sha256")
        != _canonical_sha256(_condition_identity_payload(condition))
        or any(type(payload.get(field)) is not int for field in integer_fields)
        or any(payload[field] <= 0 for field in integer_fields)
        or payload.get("resolved_at_ms")
        != max(payload["gamma_observed_wall_ms"], payload["clob_observed_wall_ms"])
        or payload.get("winning_outcome") not in {"Up", "Down"}
        or payload.get("winning_token_id")
        not in {condition.up_token_id, condition.down_token_id}
        or payload.get("paper_trading_authority") is not False
        or payload.get("live_trading_authority") is not False
        or not isinstance(gamma_compressed, bytes)
        or not isinstance(clob_compressed, bytes)
    ):
        raise ValueError("Round 25 official resolution evidence differs")
    _, gamma = _decompress_payload(
        gamma_compressed,
        raw_size=payload["gamma_raw_size_bytes"],
        compressed_sha256=str(payload["gamma_compressed_sha256"]),
        payload_sha256=str(payload["gamma_payload_sha256"]),
        label="Gamma resolution payload",
    )
    _, clob = _decompress_payload(
        clob_compressed,
        raw_size=payload["clob_raw_size_bytes"],
        compressed_sha256=str(payload["clob_compressed_sha256"]),
        payload_sha256=str(payload["clob_payload_sha256"]),
        label="CLOB resolution payload",
    )
    resolved = validate_official_resolution(
        condition,
        clob,
        gamma,
        observed_wall_ms=payload["resolved_at_ms"],
    )
    if (
        resolved != (payload["winning_token_id"], payload["winning_outcome"])
        or payload.get("official_payload_sha256")
        != _canonical_sha256(
            {
                "clob_payload_sha256": payload["clob_payload_sha256"],
                "gamma_payload_sha256": payload["gamma_payload_sha256"],
            }
        )
    ):
        raise ValueError("Round 25 official sources no longer verify the winner")
    return {**payload, "evidence_sha256": claimed}, gamma, clob


def collect_round25_resolutions_once(
    *,
    collection_database: str | Path,
    client: Round25ResolutionPublicClient,
    maximum_conditions: int = 128,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Attempt one bounded collection batch; unresolved markets stay pending."""

    if type(maximum_conditions) is not int or not 1 <= maximum_conditions <= 512:
        raise ValueError("Round 25 resolution batch size is outside the bound")
    path = Path(collection_database)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Round 25 resolution collection is unavailable")
    connection = duckdb.connect(str(path))
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET threads = 2")
    if _table_names(connection) != _COLLECTION_TABLES:
        connection.close()
        raise ValueError("Round 25 resolution collection is not open")
    claim = _load_claim(connection)
    conditions = _load_source_conditions(connection)
    if (
        len(conditions) != claim["condition_count"]
        or _condition_population_sha256(conditions)
        != claim["condition_population_sha256"]
    ):
        connection.close()
        raise ValueError("Round 25 resolution collection population differs")
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT condition_id FROM round25_resolution_evidence"
        ).fetchall()
    }
    pending = [item for item in conditions if item.condition_id not in existing]
    selected = pending[:maximum_conditions]
    attempted = 0
    inserted = 0
    unresolved = 0
    transport_failures = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        for index, condition in enumerate(selected, start=1):
            attempted += 1
            if progress is not None:
                progress(
                    "resolution_condition",
                    {
                        "batch_index": index,
                        "batch_size": len(selected),
                        "condition_id": condition.condition_id,
                    },
                )
            try:
                gamma = client.gamma_market(condition.market_id)
                clob = client.clob_market(condition.condition_id)
            except Round25ResolutionTransportError:
                transport_failures += 1
                continue
            result = validate_official_resolution(
                condition,
                clob.value,
                gamma.value,
                observed_wall_ms=max(
                    gamma.observed_wall_ms,
                    clob.observed_wall_ms,
                ),
            )
            if result is None:
                unresolved += 1
                continue
            gamma_compressed = _compress_payload(gamma.canonical_json)
            clob_compressed = _compress_payload(clob.canonical_json)
            evidence = _evidence_payload(
                condition=condition,
                gamma=gamma,
                clob=clob,
                winning_token_id=result[0],
                winning_outcome=result[1],
                gamma_compressed=gamma_compressed,
                clob_compressed=clob_compressed,
            )
            _decode_evidence(
                condition=condition,
                evidence_json=_canonical_json(evidence),
                evidence_sha256=evidence["evidence_sha256"],
                gamma_compressed=gamma_compressed,
                clob_compressed=clob_compressed,
            )
            connection.execute(
                "INSERT INTO round25_resolution_evidence VALUES (?, ?, ?, ?, ?)",
                [
                    condition.condition_id,
                    _canonical_json(evidence),
                    evidence["evidence_sha256"],
                    gamma_compressed,
                    clob_compressed,
                ],
            )
            inserted += 1
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except duckdb.Error:
            pass
        connection.close()
        raise
    resolved_count = int(
        connection.execute("SELECT COUNT(*) FROM round25_resolution_evidence").fetchone()[0]
    )
    connection.close()
    if Path(f"{path}.wal").exists():
        raise RuntimeError("Round 25 resolution collection retained a WAL")
    body: dict[str, object] = {
        "attempted_condition_count": attempted,
        "finalization_ready": resolved_count == len(conditions),
        "live_trading_authority": False,
        "newly_resolved_condition_count": inserted,
        "pending_condition_count": len(conditions) - resolved_count,
        "resolved_condition_count": resolved_count,
        "schema_version": POLYMARKET_ROUND25_RESOLUTION_COLLECTION_REPORT_SCHEMA_VERSION,
        "transport_failure_count": transport_failures,
        "unresolved_condition_count": unresolved,
    }
    return {**body, "report_sha256": _canonical_sha256(body)}


def audit_round25_resolution_collection(
    collection_database: str | Path,
) -> dict[str, object]:
    """Deep-audit every source identity, compressed payload, and winner."""

    path = Path(collection_database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 resolution collection is unavailable")
    with duckdb.connect(str(path), read_only=True) as connection:
        if _table_names(connection) not in (_COLLECTION_TABLES, _FINAL_TABLES):
            raise ValueError("Round 25 resolution collection schema differs")
        claim = _load_claim(connection)
        conditions = _load_source_conditions(connection)
        evidence_rows = connection.execute(
            """
            SELECT condition_id, evidence_json, evidence_sha256,
                   gamma_payload, clob_payload
            FROM round25_resolution_evidence
            ORDER BY condition_id
            """
        ).fetchall()
    population_sha256 = _condition_population_sha256(conditions)
    if (
        len(conditions) != claim["condition_count"]
        or population_sha256 != claim["condition_population_sha256"]
    ):
        raise ValueError("Round 25 resolution collection population differs")
    by_id = {item.condition_id: item for item in conditions}
    evidence_by_id: dict[str, dict[str, object]] = {}
    for row in evidence_rows:
        condition_id = str(row[0])
        condition = by_id.get(condition_id)
        if condition is None or condition_id in evidence_by_id:
            raise ValueError("Round 25 resolution evidence population differs")
        evidence, _, _ = _decode_evidence(
            condition=condition,
            evidence_json=row[1],
            evidence_sha256=row[2],
            gamma_compressed=bytes(row[3]),
            clob_compressed=bytes(row[4]),
        )
        evidence_by_id[condition_id] = evidence
    chain = _EMPTY_SHA256
    resolved_role_counts: Counter[str] = Counter()
    pending_population = _EMPTY_SHA256
    for condition in conditions:
        evidence = evidence_by_id.get(condition.condition_id)
        if evidence is None:
            pending_population = _hash_chain(
                pending_population,
                _condition_identity_payload(condition),
            )
            continue
        chain = _hash_chain(
            chain,
            {
                "condition_id": condition.condition_id,
                "evidence_sha256": evidence["evidence_sha256"],
            },
        )
        resolved_role_counts[condition.role] += 1
    body: dict[str, object] = {
        "claim_sha256": claim["claim_sha256"],
        "condition_count": len(conditions),
        "condition_population_sha256": population_sha256,
        "contract_sha256": POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256,
        "evidence_chain_sha256": chain,
        "feature_store_manifest_sha256": claim["feature_store_manifest_sha256"],
        "live_trading_authority": False,
        "pending_condition_count": len(conditions) - len(evidence_by_id),
        "pending_population_sha256": pending_population,
        "resolved_condition_count": len(evidence_by_id),
        "resolved_role_counts": {
            role: resolved_role_counts[role] for role in _ROLES
        },
        "schema_version": POLYMARKET_ROUND25_RESOLUTION_AUDIT_SCHEMA_VERSION,
        "terminal_receipt_audit_sha256": claim[
            "terminal_receipt_audit_sha256"
        ],
        "terminal_transport_manifest_sha256": claim[
            "terminal_transport_manifest_sha256"
        ],
    }
    return {**body, "audit_sha256": _canonical_sha256(body)}


def _load_authority(value: Mapping[str, object]) -> Round25ResolutionAuthority:
    expected = {
        "authority_sha256",
        "candidate_amendment_sha256",
        "candidate_design_sha256",
        "created_at_ms",
        "official_resolution_audit_sha256",
        "official_resolution_semantics_verified",
        "schema_version",
        "source_campaign_plan_sha256",
        "terminal_transport_sha256",
        "trading_authority",
    }
    if set(value) != expected:
        raise ValueError("Round 25 stored resolution authority differs")
    return Round25ResolutionAuthority(**dict(value)).validated()


def _load_resolution(
    value: Mapping[str, object],
    *,
    authority: Round25ResolutionAuthority,
) -> Round25OfficialResolution:
    expected = {
        "condition_id",
        "down_token_id",
        "event_start_ms",
        "official_payload_sha256",
        "resolution_authority_sha256",
        "resolution_sha256",
        "resolved_at_ms",
        "target_origin",
        "up_token_id",
        "winning_token_id",
    }
    if set(value) != expected:
        raise ValueError("Round 25 stored official resolution differs")
    return Round25OfficialResolution(**dict(value)).validated(authority)


def _validate_final_manifest(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    expected = {
        "atomic_file_publication",
        "authority_sha256",
        "condition_count",
        "condition_population_sha256",
        "contract_sha256",
        "created_at_ms",
        "evidence_chain_sha256",
        "feature_store_manifest_sha256",
        "live_trading_authority",
        "model_data_eligible",
        "paper_trading_authority",
        "profitability_claim",
        "resolution_audit_sha256",
        "resolution_chain_sha256",
        "resolution_count",
        "role_resolution_counts",
        "schema_version",
        "store_schema_version",
        "terminal_receipt_audit_sha256",
        "terminal_transport_manifest_sha256",
    }
    roles = payload.get("role_resolution_counts")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_RESOLUTION_STORE_MANIFEST_SCHEMA_VERSION
        or payload.get("store_schema_version")
        != POLYMARKET_ROUND25_RESOLUTION_STORE_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= POLYMARKET_ROUND25_SELECTION_END_MS
        or type(payload.get("condition_count")) is not int
        or type(payload.get("resolution_count")) is not int
        or payload["condition_count"] <= 0
        or payload["resolution_count"] != payload["condition_count"]
        or not isinstance(roles, Mapping)
        or set(roles) != set(_ROLES)
        or any(
            type(roles[role]) is not int
            or roles[role] < POLYMARKET_ROUND25_MINIMUM_CONDITIONS[role]
            for role in _ROLES
        )
        or sum(roles.values()) != payload["resolution_count"]
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in (
                "authority_sha256",
                "condition_population_sha256",
                "evidence_chain_sha256",
                "feature_store_manifest_sha256",
                "resolution_audit_sha256",
                "resolution_chain_sha256",
                "terminal_receipt_audit_sha256",
                "terminal_transport_manifest_sha256",
            )
        )
        or payload.get("atomic_file_publication") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 25 resolution store manifest differs")
    return {**payload, "manifest_sha256": claimed}


def audit_round25_resolution_store(database: str | Path) -> dict[str, object]:
    """Deep-audit the final authority and every official target row."""

    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 resolution store is unavailable")
    collection_audit = audit_round25_resolution_collection(path)
    with duckdb.connect(str(path), read_only=True) as connection:
        if _table_names(connection) != _FINAL_TABLES:
            raise ValueError("Round 25 final resolution store schema differs")
        authority_rows = connection.execute(
            "SELECT authority_json, authority_sha256 FROM round25_resolution_authority"
        ).fetchall()
        manifest_rows = connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM round25_resolution_store_manifest"
        ).fetchall()
        resolution_rows = connection.execute(
            """
            SELECT condition_id, resolution_json, resolution_sha256,
                   evidence_sha256
            FROM round25_official_resolution
            ORDER BY event_start_ms, condition_id
            """
        ).fetchall()
    if len(authority_rows) != 1 or len(manifest_rows) != 1:
        raise ValueError("Round 25 final resolution singleton rows differ")
    authority_payload = _strict_json(
        authority_rows[0][0], label="resolution authority"
    )
    authority = _load_authority(authority_payload)
    if authority_rows[0][1] != authority.authority_sha256:
        raise ValueError("Round 25 resolution authority hash differs")
    manifest = _validate_final_manifest(
        _strict_json(manifest_rows[0][0], label="resolution store manifest")
    )
    if manifest_rows[0][1] != manifest["manifest_sha256"]:
        raise ValueError("Round 25 resolution store manifest hash differs")
    evidence_by_id: dict[str, str] = {}
    with duckdb.connect(str(path), read_only=True) as connection:
        for row in connection.execute(
            "SELECT condition_id, evidence_sha256 FROM round25_resolution_evidence"
        ).fetchall():
            evidence_by_id[str(row[0])] = str(row[1])
    chain = _EMPTY_SHA256
    role_counts: Counter[str] = Counter()
    conditions = {
        item.condition_id: item
        for item in load_round25_resolution_source_conditions(path)
    }
    for row in resolution_rows:
        payload = _strict_json(row[1], label="official resolution")
        resolution = _load_resolution(payload, authority=authority)
        condition = conditions.get(resolution.condition_id)
        if (
            row[0] != resolution.condition_id
            or row[2] != resolution.resolution_sha256
            or row[3] != evidence_by_id.get(resolution.condition_id)
            or condition is None
            or resolution.event_start_ms != condition.event_start_ms
            or resolution.up_token_id != condition.up_token_id
            or resolution.down_token_id != condition.down_token_id
        ):
            raise ValueError("Round 25 official resolution row differs")
        chain = _hash_chain(
            chain,
            {
                "condition_id": resolution.condition_id,
                "resolution_sha256": resolution.resolution_sha256,
            },
        )
        role_counts[condition.role] += 1
    if (
        manifest["authority_sha256"] != authority.authority_sha256
        or manifest["resolution_audit_sha256"] != collection_audit["audit_sha256"]
        or manifest["condition_population_sha256"]
        != collection_audit["condition_population_sha256"]
        or manifest["evidence_chain_sha256"]
        != collection_audit["evidence_chain_sha256"]
        or manifest["resolution_count"] != len(resolution_rows)
        or manifest["resolution_chain_sha256"] != chain
        or manifest["role_resolution_counts"]
        != {role: role_counts[role] for role in _ROLES}
    ):
        raise ValueError("Round 25 final resolution deep audit differs")
    return manifest


def _load_final_singletons(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, object], Round25ResolutionAuthority]:
    if _table_names(connection) != _FINAL_TABLES:
        raise ValueError("Round 25 final resolution store schema differs")
    authority_rows = connection.execute(
        "SELECT authority_json, authority_sha256 FROM round25_resolution_authority"
    ).fetchall()
    manifest_rows = connection.execute(
        "SELECT manifest_json, manifest_sha256 FROM round25_resolution_store_manifest"
    ).fetchall()
    if len(authority_rows) != 1 or len(manifest_rows) != 1:
        raise ValueError("Round 25 final resolution singleton rows differ")
    authority = _load_authority(
        _strict_json(authority_rows[0][0], label="resolution authority")
    )
    manifest = _validate_final_manifest(
        _strict_json(manifest_rows[0][0], label="resolution store manifest")
    )
    if (
        authority_rows[0][1] != authority.authority_sha256
        or manifest_rows[0][1] != manifest["manifest_sha256"]
        or manifest["authority_sha256"] != authority.authority_sha256
        or manifest["resolution_audit_sha256"]
        != authority.official_resolution_audit_sha256
    ):
        raise ValueError("Round 25 final resolution singleton hashes differ")
    return manifest, authority


def _load_verified_resolution_role(
    connection: duckdb.DuckDBPyConnection,
    *,
    role: str,
    authority: Round25ResolutionAuthority,
    expected_count: int,
) -> tuple[Round25OfficialResolution, ...]:
    rows = connection.execute(
        """
        SELECT r.condition_id, r.event_start_ms, r.role, r.resolution_json,
               r.resolution_sha256, r.evidence_sha256,
               e.evidence_json, e.evidence_sha256,
               e.gamma_payload, e.clob_payload, s.identity_json,
               s.identity_sha256
        FROM round25_official_resolution AS r
        JOIN round25_resolution_evidence AS e USING (condition_id)
        JOIN round25_resolution_source_condition AS s USING (condition_id)
        WHERE r.role = ?
        ORDER BY r.event_start_ms, r.condition_id
        """,
        [role],
    ).fetchall()
    if len(rows) != expected_count:
        raise ValueError("Round 25 resolution role population differs")
    output: list[Round25OfficialResolution] = []
    for row in rows:
        condition_payload = _strict_json(
            row[10], label="resolution source condition"
        )
        condition = _condition_from_payload(condition_payload)
        resolution = _load_resolution(
            _strict_json(row[3], label="official resolution"),
            authority=authority,
        )
        evidence, _, _ = _decode_evidence(
            condition=condition,
            evidence_json=row[6],
            evidence_sha256=row[7],
            gamma_compressed=bytes(row[8]),
            clob_compressed=bytes(row[9]),
        )
        if (
            row[0] != condition.condition_id
            or row[0] != resolution.condition_id
            or row[1] != condition.event_start_ms
            or row[1] != resolution.event_start_ms
            or row[2] != role
            or condition.role != role
            or row[4] != resolution.resolution_sha256
            or row[5] != evidence["evidence_sha256"]
            or row[5] != row[7]
            or row[11] != _canonical_sha256(condition_payload)
            or resolution.official_payload_sha256
            != evidence["official_payload_sha256"]
            or resolution.winning_token_id != evidence["winning_token_id"]
            or resolution.resolved_at_ms != evidence["resolved_at_ms"]
        ):
            raise ValueError("Round 25 verified resolution role row differs")
        output.append(resolution)
    return tuple(output)


def load_round25_fit_resolution_inputs(
    database: str | Path,
) -> tuple[
    dict[str, object],
    Round25ResolutionAuthority,
    dict[str, tuple[Round25OfficialResolution, ...]],
]:
    """Read and verify train/calibration targets without selecting selection rows."""

    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 fit resolution store is unavailable")
    before = path.stat()
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        manifest, authority = _load_final_singletons(connection)
        resolved = {
            role: _load_verified_resolution_role(
                connection,
                role=role,
                authority=authority,
                expected_count=int(manifest["role_resolution_counts"][role]),
            )
            for role in ("train", "calibration")
        }
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or Path(f"{path}.wal").exists()
    ):
        raise RuntimeError("Round 25 resolution store changed during fit access")
    return manifest, authority, resolved


def load_round25_selection_resolution_inputs(
    database: str | Path,
    *,
    panel: Round25PredictionPanel,
    access_store: Round25SelectionAccessStore,
    allow_consumed_recovery: bool = False,
) -> tuple[
    dict[str, object],
    Round25ResolutionAuthority,
    tuple[Round25OfficialResolution, ...],
    str,
]:
    """Open selection targets only across a durable prediction-panel lock."""

    if not isinstance(access_store, Round25SelectionAccessStore):
        raise TypeError("Round 25 selection access store type differs")
    if type(allow_consumed_recovery) is not bool:
        raise TypeError("Round 25 selection recovery flag differs")
    status, claim_sha256 = access_store.validate_prediction_binding(panel=panel)
    if status != "prediction_panel_frozen" and not allow_consumed_recovery:
        raise RuntimeError("Round 25 selection target access is already consumed")
    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 selection resolution store is unavailable")
    before = path.stat()
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        manifest, authority = _load_final_singletons(connection)
        resolutions = _load_verified_resolution_role(
            connection,
            role="selection",
            authority=authority,
            expected_count=int(manifest["role_resolution_counts"]["selection"]),
        )
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or Path(f"{path}.wal").exists()
        or access_store.validate_prediction_binding(panel=panel)
        != (status, claim_sha256)
    ):
        raise RuntimeError("Round 25 selection target boundary changed during access")
    return manifest, authority, resolutions, claim_sha256


def load_round25_resolution_source_conditions(
    database: str | Path,
) -> tuple[Round25JointReceiptCondition, ...]:
    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 resolution source population is unavailable")
    with duckdb.connect(str(path), read_only=True) as connection:
        if _table_names(connection) not in (_COLLECTION_TABLES, _FINAL_TABLES):
            raise ValueError("Round 25 resolution source schema differs")
        return _load_source_conditions(connection)


def finalize_round25_resolution_store(
    *,
    feature_database: str | Path,
    destination_database: str | Path,
    created_at_ms: int,
) -> dict[str, object]:
    """Create authority and targets only after a complete dual-source audit."""

    feature_manifest = audit_round25_joint_store(feature_database)
    destination = Path(destination_database)
    collection = round25_resolution_collection_database(destination)
    if destination.is_symlink() or destination.exists() or not collection.is_file():
        raise ValueError("Round 25 final resolution destination differs")
    collection_audit = audit_round25_resolution_collection(collection)
    claim = load_round25_resolution_access_claim(collection)
    if (
        claim["feature_store_manifest_sha256"]
        != feature_manifest["manifest_sha256"]
        or claim["condition_population_sha256"]
        != feature_manifest["condition_population_sha256"]
        or collection_audit["pending_condition_count"] != 0
        or collection_audit["resolved_condition_count"]
        != collection_audit["condition_count"]
        or any(
            collection_audit["resolved_role_counts"][role]
            < POLYMARKET_ROUND25_MINIMUM_CONDITIONS[role]
            for role in _ROLES
        )
    ):
        raise ValueError("Round 25 resolution finalization gate is closed")
    with duckdb.connect(str(collection), read_only=True) as connection:
        table_names = _table_names(connection)
    if table_names == _FINAL_TABLES:
        manifest = audit_round25_resolution_store(collection)
        _replace_with_retries(collection, destination)
        return audit_round25_resolution_store(destination)
    if table_names != _COLLECTION_TABLES:
        raise ValueError("Round 25 resolution collection schema differs")
    authority = Round25ResolutionAuthority.create(
        terminal_transport_sha256=claim["terminal_transport_manifest_sha256"],
        official_resolution_audit_sha256=collection_audit["audit_sha256"],
        created_at_ms=created_at_ms,
        official_resolution_semantics_verified=True,
    )
    connection = duckdb.connect(str(collection))
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET threads = 2")
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            CREATE TABLE round25_resolution_authority (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                authority_json VARCHAR NOT NULL,
                authority_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_official_resolution (
                condition_id VARCHAR PRIMARY KEY,
                event_start_ms BIGINT NOT NULL,
                role VARCHAR NOT NULL,
                resolution_json VARCHAR NOT NULL,
                resolution_sha256 VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_resolution_store_manifest (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                manifest_json VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO round25_resolution_authority VALUES (TRUE, ?, ?)",
            [_canonical_json(asdict(authority)), authority.authority_sha256],
        )
        conditions = _load_source_conditions(connection)
        evidence_rows = {
            str(row[0]): (str(row[1]), str(row[2]))
            for row in connection.execute(
                """
                SELECT condition_id, evidence_json, evidence_sha256
                FROM round25_resolution_evidence
                """
            ).fetchall()
        }
        resolution_chain = _EMPTY_SHA256
        role_counts: Counter[str] = Counter()
        for condition in conditions:
            stored = evidence_rows.get(condition.condition_id)
            if stored is None:
                raise ValueError("Round 25 final resolution evidence is missing")
            evidence = _strict_json(stored[0], label="resolution evidence")
            resolution = Round25OfficialResolution.create(
                condition_id=condition.condition_id,
                event_start_ms=condition.event_start_ms,
                up_token_id=condition.up_token_id,
                down_token_id=condition.down_token_id,
                winning_token_id=str(evidence["winning_token_id"]),
                resolved_at_ms=int(evidence["resolved_at_ms"]),
                official_payload_sha256=str(evidence["official_payload_sha256"]),
                authority=authority,
            )
            connection.execute(
                "INSERT INTO round25_official_resolution VALUES (?, ?, ?, ?, ?, ?)",
                [
                    condition.condition_id,
                    condition.event_start_ms,
                    condition.role,
                    _canonical_json(asdict(resolution)),
                    resolution.resolution_sha256,
                    stored[1],
                ],
            )
            resolution_chain = _hash_chain(
                resolution_chain,
                {
                    "condition_id": resolution.condition_id,
                    "resolution_sha256": resolution.resolution_sha256,
                },
            )
            role_counts[condition.role] += 1
        body: dict[str, object] = {
            "atomic_file_publication": True,
            "authority_sha256": authority.authority_sha256,
            "condition_count": len(conditions),
            "condition_population_sha256": collection_audit[
                "condition_population_sha256"
            ],
            "contract_sha256": POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256,
            "created_at_ms": created_at_ms,
            "evidence_chain_sha256": collection_audit["evidence_chain_sha256"],
            "feature_store_manifest_sha256": claim[
                "feature_store_manifest_sha256"
            ],
            "live_trading_authority": False,
            "model_data_eligible": False,
            "paper_trading_authority": False,
            "profitability_claim": False,
            "resolution_audit_sha256": collection_audit["audit_sha256"],
            "resolution_chain_sha256": resolution_chain,
            "resolution_count": len(conditions),
            "role_resolution_counts": {
                role: role_counts[role] for role in _ROLES
            },
            "schema_version": (
                POLYMARKET_ROUND25_RESOLUTION_STORE_MANIFEST_SCHEMA_VERSION
            ),
            "store_schema_version": POLYMARKET_ROUND25_RESOLUTION_STORE_SCHEMA_VERSION,
            "terminal_receipt_audit_sha256": claim[
                "terminal_receipt_audit_sha256"
            ],
            "terminal_transport_manifest_sha256": claim[
                "terminal_transport_manifest_sha256"
            ],
        }
        manifest = _validate_final_manifest(
            {**body, "manifest_sha256": _canonical_sha256(body)}
        )
        connection.execute(
            "INSERT INTO round25_resolution_store_manifest VALUES (TRUE, ?, ?)",
            [_canonical_json(manifest), manifest["manifest_sha256"]],
        )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        connection.close()
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except duckdb.Error:
            pass
        connection.close()
        raise
    if Path(f"{collection}.wal").exists():
        raise RuntimeError("Round 25 finalized resolution collection retained a WAL")
    audit_round25_resolution_store(collection)
    _replace_with_retries(collection, destination)
    return audit_round25_resolution_store(destination)


__all__ = [
    "POLYMARKET_ROUND25_RESOLUTION_ACCESS_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_RESOLUTION_AUDIT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_RESOLUTION_CODEC",
    "POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_RESOLUTION_EVIDENCE_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_RESOLUTION_STORE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_RESOLUTION_STORE_SCHEMA_VERSION",
    "Round25OfficialPublicPayload",
    "Round25ResolutionPublicClient",
    "Round25ResolutionTransportError",
    "audit_round25_resolution_collection",
    "audit_round25_resolution_store",
    "collect_round25_resolutions_once",
    "finalize_round25_resolution_store",
    "initialize_round25_resolution_collection",
    "load_round25_resolution_access_claim",
    "load_round25_fit_resolution_inputs",
    "load_round25_selection_resolution_inputs",
    "load_round25_resolution_source_conditions",
    "round25_resolution_collection_database",
    "validate_round25_resolution_access_claim",
]
