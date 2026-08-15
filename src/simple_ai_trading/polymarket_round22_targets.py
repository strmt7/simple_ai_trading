"""Claim-gated dual-source targets for the Round 22 diagnostic cohort."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Protocol
from urllib.parse import urlparse

import requests

from .polymarket_resolution import validate_official_resolution
from .polymarket_round22_feature_store import Round22FeatureStore
from .polymarket_round22_features import (
    POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
)
from .polymarket_round22_pilot import Round22PilotStore


POLYMARKET_ROUND22_DIAGNOSTIC_RELATIVE = (
    "docs/model-research/polymarket/round-022-diagnostic-mini-cohort-v1.json"
)
POLYMARKET_ROUND22_TARGET_SCHEMA_VERSION = "polymarket-round22-diagnostic-target-v1"
POLYMARKET_ROUND22_TARGET_CLAIM_SCHEMA_VERSION = (
    "polymarket-round22-target-access-claim-v1"
)
POLYMARKET_ROUND22_DIAGNOSTIC_SHA256 = (
    "1014fd3ca79aa043acd4a79652a9503b1713eda22e64f04ecc95f0b374dddd75"
)
POLYMARKET_ROUND22_GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_ROUND22_CLOB_MARKET_URL = "https://clob.polymarket.com/markets"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_MAXIMUM_ARTIFACT_BYTES = 2 * 1024 * 1024
_MAXIMUM_RESPONSE_BYTES = 8 * 1024 * 1024
_RETRYABLE_STATUS_CODES = frozenset((425, 429, 500, 502, 503, 504))
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "poly_address",
        "poly_signature",
        "poly_timestamp",
        "poly_nonce",
        "poly_api_key",
        "poly_passphrase",
    }
)
_IMPLEMENTATION_PATHS = (
    "src/simple_ai_trading/polymarket_resolution.py",
    "src/simple_ai_trading/polymarket_round22_pilot.py",
    "src/simple_ai_trading/polymarket_round22_targets.py",
    "tools/run_polymarket_round22_diagnostic_targets.py",
)
_AUTHORITY_KEYS = frozenset(
    {
        "binance_private_api",
        "live_trading",
        "paper_trading",
        "polymarket_authentication",
        "polymarket_order_submission",
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
            raise ValueError("Round 22 target JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 target JSON contains {value}")


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


@dataclass(frozen=True, slots=True)
class Round22PublicPayload:
    value: Mapping[str, object]
    canonical_json: str
    sha256: str
    observed_at_ms: int


class Round22PublicTargetClient:
    """Credential-free dual-origin client with no order or account methods."""

    def __init__(
        self,
        *,
        session: _Session | None = None,
        timeout_seconds: float = 20.0,
        minimum_request_interval_seconds: float = 0.2,
        maximum_attempts: int = 4,
        clock_ms: Callable[[], int] | None = None,
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
            raise ValueError("Round 22 target attempts are outside the bound")
        self.maximum_attempts = maximum_attempts
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic = monotonic or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._last_request_by_origin: dict[str, float] = {}

    def _assert_public_session(self) -> None:
        headers = {str(name).strip().lower() for name in self.session.headers}
        if headers & _SENSITIVE_HEADERS:
            raise ValueError("Round 22 target session contains authority headers")
        cookies = getattr(self.session, "cookies", None)
        if cookies is not None and len(cookies):
            raise ValueError("Round 22 target session contains cookies")

    def _discard_response_cookies(self) -> None:
        cookies = getattr(self.session, "cookies", None)
        if cookies is None or not len(cookies):
            return
        clear = getattr(cookies, "clear", None)
        if not callable(clear):
            raise ValueError("Round 22 target session retained cookies")
        clear()
        if len(cookies):
            raise ValueError("Round 22 target session retained cookies")

    def _wait(self, origin: str) -> None:
        now = float(self.monotonic())
        prior = self._last_request_by_origin.get(origin)
        if prior is not None:
            delay = self.minimum_request_interval_seconds - (now - prior)
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

    def _request(self, url: str, *, origin: str, path: str) -> Round22PublicPayload:
        self._assert_public_session()
        for attempt in range(self.maximum_attempts):
            self._wait(origin)
            try:
                response = self.session.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "simple-ai-trading-round22-target/0.1",
                    },
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt + 1 == self.maximum_attempts:
                    raise ValueError(
                        "Round 22 target transport retries were exhausted"
                    ) from exc
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
                continue
            self._discard_response_cookies()
            status = int(response.status_code)
            if status == 200:
                parsed_url = urlparse(str(response.url))
                if (
                    parsed_url.scheme != "https"
                    or parsed_url.netloc.lower() != origin
                    or parsed_url.path != path
                ):
                    raise ValueError("Round 22 target response changed origin")
                content_type = str(response.headers.get("Content-Type") or "").lower()
                content = bytes(response.content)
                if (
                    not content_type.startswith("application/json")
                    or not 2 <= len(content) <= _MAXIMUM_RESPONSE_BYTES
                ):
                    raise ValueError("Round 22 target response is not bounded JSON")
                try:
                    decoded = json.loads(
                        content.decode("utf-8"),
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_nonfinite,
                    )
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Round 22 target response is not strict JSON"
                    ) from exc
                if not isinstance(decoded, Mapping):
                    raise ValueError("Round 22 target response is not an object")
                canonical = _canonical_json(decoded)
                return Round22PublicPayload(
                    value=dict(decoded),
                    canonical_json=canonical,
                    sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
                    observed_at_ms=int(self.clock_ms()),
                )
            if (
                status not in _RETRYABLE_STATUS_CODES
                or attempt + 1 == self.maximum_attempts
            ):
                raise ValueError(f"Round 22 target request failed with HTTP {status}")
            self.sleeper(self._retry_after(response, attempt))
        raise AssertionError("unreachable Round 22 target retry state")

    def gamma_market(self, market_id: str) -> Round22PublicPayload:
        selected = str(market_id or "").strip()
        if not selected.isdigit() or len(selected) > 20:
            raise ValueError("Round 22 Gamma target market ID is invalid")
        path = f"/markets/{selected}"
        return self._request(
            f"{POLYMARKET_ROUND22_GAMMA_MARKET_URL}/{selected}",
            origin="gamma-api.polymarket.com",
            path=path,
        )

    def clob_market(self, condition_id: str) -> Round22PublicPayload:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 22 CLOB target condition ID is invalid")
        path = f"/markets/{selected}"
        return self._request(
            f"{POLYMARKET_ROUND22_CLOB_MARKET_URL}/{selected}",
            origin="clob.polymarket.com",
            path=path,
        )


def load_round22_diagnostic_preregistration(
    repository: str | Path,
) -> dict[str, object]:
    path = Path(repository).resolve() / POLYMARKET_ROUND22_DIAGNOSTIC_RELATIVE
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ValueError("Round 22 diagnostic preregistration is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 22 diagnostic preregistration is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 22 diagnostic preregistration is not an object")
    artifact = dict(decoded)
    claimed = str(artifact.pop("preregistration_sha256", "")).strip().lower()
    population = artifact.get("population")
    authority = artifact.get("authority")
    conditions = (
        population.get("conditions") if isinstance(population, Mapping) else None
    )
    if (
        claimed != _canonical_sha256(artifact)
        or claimed != POLYMARKET_ROUND22_DIAGNOSTIC_SHA256
        or _SHA256.fullmatch(claimed) is None
        or artifact.get("schema_version")
        != "polymarket-round22-diagnostic-mini-cohort-v1"
        or artifact.get("status") != "frozen_after_features_before_any_target_access"
        or not isinstance(authority, Mapping)
        or set(authority) != _AUTHORITY_KEYS
        or any(value is not False for value in authority.values())
        or not isinstance(conditions, list)
        or len(conditions) != 36
    ):
        raise ValueError("Round 22 diagnostic preregistration differs")
    roles: dict[str, int] = {}
    identities: set[str] = set()
    for item in conditions:
        if not isinstance(item, Mapping):
            raise ValueError("Round 22 diagnostic population is malformed")
        condition_id = str(item.get("condition_id") or "").strip().lower()
        role = str(item.get("role") or "")
        if (
            set(item)
            != {
                "condition_id",
                "feature_manifest_sha256",
                "role",
                "slug",
                "source_manifest_sha256",
            }
            or _CONDITION_ID.fullmatch(condition_id) is None
            or condition_id in identities
            or role not in {"train", "tune_calibration", "tune_selection"}
            or _SHA256.fullmatch(str(item.get("feature_manifest_sha256"))) is None
            or _SHA256.fullmatch(str(item.get("source_manifest_sha256"))) is None
        ):
            raise ValueError("Round 22 diagnostic population differs")
        identities.add(condition_id)
        roles[role] = roles.get(role, 0) + 1
    if roles != {"train": 12, "tune_calibration": 12, "tune_selection": 12}:
        raise ValueError("Round 22 diagnostic role counts differ")
    return {**artifact, "preregistration_sha256": claimed}


def round22_target_implementation_manifest(
    repository: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve()
    hashes: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Round 22 target implementation is unavailable")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema_version": "polymarket-round22-target-implementation-v1",
        "files_sha256": hashes,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _population(artifact: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    population = artifact["population"]
    assert isinstance(population, Mapping)
    conditions = population["conditions"]
    assert isinstance(conditions, list)
    return tuple(item for item in conditions if isinstance(item, Mapping))


def validate_round22_diagnostic_target_opening(
    pilot_store: Round22PilotStore,
) -> dict[str, object]:
    artifact = load_round22_diagnostic_preregistration(pilot_store.contract.repository)
    conditions = _population(artifact)
    state = str(
        pilot_store.connection.execute(
            "SELECT state FROM feature.pilot_manifest WHERE singleton"
        ).fetchone()[0]
    )
    claim_table_count = int(
        pilot_store.connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'target' AND table_name = 'round22_access_claim'
            """
        ).fetchone()[0]
    )
    if (
        state != "feature_ingestion"
        or claim_table_count != 0
        or pilot_store.target_row_count() != 0
    ):
        raise ValueError("Round 22 target opening preflight phase differs")
    feature_store = Round22FeatureStore(pilot_store)
    role_counts: dict[str, int] = {}
    for item in conditions:
        condition_id = str(item["condition_id"])
        source = pilot_store.connection.execute(
            "SELECT manifest_sha256 FROM feature.condition_manifest WHERE condition_id = ?",
            [condition_id],
        ).fetchone()
        feature = feature_store.audit_condition(condition_id)
        if (
            source is None
            or str(source[0]) != item["source_manifest_sha256"]
            or feature["manifest_sha256"] != item["feature_manifest_sha256"]
            or feature["target_row_count"] != 0
        ):
            raise ValueError("Round 22 target opening feature population differs")
        role = str(item["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    implementation = round22_target_implementation_manifest(
        pilot_store.contract.repository
    )
    return {
        "authentication_used": False,
        "condition_count": len(conditions),
        "implementation_manifest_sha256": implementation["manifest_sha256"],
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "polymarket_order_submission": False,
        "population_sha256": _canonical_sha256(list(conditions)),
        "preregistration_sha256": artifact["preregistration_sha256"],
        "role_counts": role_counts,
        "target_row_count": 0,
    }


def open_round22_diagnostic_target_claim(
    pilot_store: Round22PilotStore,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> str:
    if pilot_store.read_only:
        raise ValueError("Round 22 target claim requires a writable store")
    state = str(
        pilot_store.connection.execute(
            "SELECT state FROM feature.pilot_manifest WHERE singleton"
        ).fetchone()[0]
    )
    table_exists = bool(
        pilot_store.connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'target' AND table_name = 'round22_access_claim'
            """
        ).fetchone()[0]
    )
    if table_exists:
        if state != "diagnostic_targets_open":
            raise ValueError("Round 22 existing target claim differs")
        return str(_claim(pilot_store)[9])
    preflight = validate_round22_diagnostic_target_opening(pilot_store)
    artifact = load_round22_diagnostic_preregistration(pilot_store.contract.repository)
    implementation = round22_target_implementation_manifest(
        pilot_store.contract.repository
    )
    opened_at_ms = int((clock_ms or (lambda: time.time_ns() // 1_000_000))())
    population_sha = str(preflight["population_sha256"])
    claim_body = {
        "feature_names_sha256": POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
        "feature_policy_sha256": POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
        "implementation_manifest_sha256": implementation["manifest_sha256"],
        "opened_at_ms": opened_at_ms,
        "population_sha256": population_sha,
        "preregistration_sha256": artifact["preregistration_sha256"],
        "preexisting_target_count": 0,
        "schema_version": POLYMARKET_ROUND22_TARGET_CLAIM_SCHEMA_VERSION,
        "state": "opened_before_any_target_request",
    }
    claim_sha = _canonical_sha256(claim_body)
    transaction_started = False
    try:
        pilot_store.connection.execute("BEGIN TRANSACTION")
        transaction_started = True
        phase = pilot_store.connection.execute(
            """
            SELECT state,
                   (SELECT COUNT(*) FROM target.official_resolution),
                   (SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'target'
                      AND table_name = 'round22_access_claim')
            FROM feature.pilot_manifest WHERE singleton
            """
        ).fetchone()
        if phase != ("feature_ingestion", 0, 0):
            raise ValueError("Round 22 target claim phase changed during opening")
        pilot_store.connection.execute(
            """
            CREATE TABLE target.round22_access_claim (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                preregistration_sha256 VARCHAR NOT NULL,
                population_sha256 VARCHAR NOT NULL,
                feature_policy_sha256 VARCHAR NOT NULL,
                feature_names_sha256 VARCHAR NOT NULL,
                implementation_manifest_json VARCHAR NOT NULL,
                implementation_manifest_sha256 VARCHAR NOT NULL,
                opened_at_ms BIGINT NOT NULL,
                status VARCHAR NOT NULL,
                claim_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE target.round22_resolution_evidence (
                condition_id VARCHAR PRIMARY KEY,
                role VARCHAR NOT NULL,
                winning_token_id VARCHAR NOT NULL,
                winning_outcome VARCHAR NOT NULL,
                gamma_payload_json VARCHAR NOT NULL,
                gamma_payload_sha256 VARCHAR NOT NULL,
                clob_payload_json VARCHAR NOT NULL,
                clob_payload_sha256 VARCHAR NOT NULL,
                observed_at_ms BIGINT NOT NULL,
                evidence_json VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL
            );
            """
        )
        pilot_store.connection.execute(
            "INSERT INTO target.round22_access_claim VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                True,
                POLYMARKET_ROUND22_TARGET_CLAIM_SCHEMA_VERSION,
                artifact["preregistration_sha256"],
                population_sha,
                POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
                POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
                _canonical_json(implementation),
                implementation["manifest_sha256"],
                opened_at_ms,
                "opened",
                claim_sha,
            ],
        )
        pilot_store.connection.execute(
            "UPDATE feature.pilot_manifest SET state = 'diagnostic_targets_open' WHERE singleton"
        )
        pilot_store.connection.execute("COMMIT")
        transaction_started = False
    except Exception:
        if transaction_started:
            pilot_store.connection.execute("ROLLBACK")
        raise
    return claim_sha


def _claim(pilot_store: Round22PilotStore) -> tuple[object, ...]:
    row = pilot_store.connection.execute(
        """
        SELECT schema_version, preregistration_sha256, population_sha256,
               feature_policy_sha256, feature_names_sha256,
               implementation_manifest_json, implementation_manifest_sha256,
               opened_at_ms, status, claim_sha256
        FROM target.round22_access_claim WHERE singleton
        """
    ).fetchone()
    if row is None:
        raise ValueError("Round 22 target access claim is unavailable")
    artifact = load_round22_diagnostic_preregistration(pilot_store.contract.repository)
    implementation = round22_target_implementation_manifest(
        pilot_store.contract.repository
    )
    conditions = _population(artifact)
    state = str(
        pilot_store.connection.execute(
            "SELECT state FROM feature.pilot_manifest WHERE singleton"
        ).fetchone()[0]
    )
    evidence_count = int(
        pilot_store.connection.execute(
            "SELECT COUNT(*) FROM target.round22_resolution_evidence"
        ).fetchone()[0]
    )
    summary_count = pilot_store.target_row_count()
    claim_body = {
        "feature_names_sha256": str(row[4]),
        "feature_policy_sha256": str(row[3]),
        "implementation_manifest_sha256": str(row[6]),
        "opened_at_ms": int(row[7]),
        "population_sha256": str(row[2]),
        "preregistration_sha256": str(row[1]),
        "preexisting_target_count": 0,
        "schema_version": str(row[0]),
        "state": "opened_before_any_target_request",
    }
    if (
        row[0] != POLYMARKET_ROUND22_TARGET_CLAIM_SCHEMA_VERSION
        or row[1] != artifact["preregistration_sha256"]
        or row[2] != _canonical_sha256(list(conditions))
        or row[3] != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
        or row[4] != POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
        or row[5] != _canonical_json(implementation)
        or row[6] != implementation["manifest_sha256"]
        or row[8] not in {"opened", "complete"}
        or state != "diagnostic_targets_open"
        or evidence_count != summary_count
        or not 0 <= evidence_count <= len(conditions)
        or (row[8] == "complete") != (evidence_count == len(conditions))
        or row[9] != _canonical_sha256(claim_body)
    ):
        raise ValueError("Round 22 target access claim differs")
    return row


def _audit_target(
    pilot_store: Round22PilotStore,
    *,
    condition_id: str,
    claim_sha256: str,
) -> str:
    row = pilot_store.connection.execute(
        """
        SELECT role, winning_token_id, winning_outcome, gamma_payload_json,
               gamma_payload_sha256, clob_payload_json, clob_payload_sha256,
               observed_at_ms, evidence_json, evidence_sha256
        FROM target.round22_resolution_evidence WHERE condition_id = ?
        """,
        [condition_id],
    ).fetchone()
    summary = pilot_store.connection.execute(
        "SELECT access_claim_sha256, evidence_sha256, winning_outcome FROM target.official_resolution WHERE condition_id = ?",
        [condition_id],
    ).fetchone()
    if row is None or summary is None:
        raise ValueError("Round 22 target evidence is incomplete")
    try:
        gamma = json.loads(
            str(row[3]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
        clob = json.loads(
            str(row[5]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
        evidence = json.loads(
            str(row[8]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 22 target evidence is invalid") from exc
    if not all(isinstance(value, Mapping) for value in (gamma, clob, evidence)):
        raise ValueError("Round 22 target evidence is malformed")
    market = pilot_store.market(condition_id)
    winner = validate_official_resolution(
        market.official_market(gamma),
        clob,
        gamma,
        observed_wall_ms=int(row[7]),
    )
    expected = {
        "access_claim_sha256": claim_sha256,
        "clob_payload_sha256": str(row[6]),
        "condition_id": condition_id,
        "gamma_payload_sha256": str(row[4]),
        "observed_at_ms": int(row[7]),
        "role": str(row[0]),
        "schema_version": POLYMARKET_ROUND22_TARGET_SCHEMA_VERSION,
        "winning_outcome": str(row[2]),
        "winning_token_id": str(row[1]),
    }
    if (
        winner != (str(row[1]), str(row[2]))
        or str(row[3]) != _canonical_json(gamma)
        or str(row[4]) != _canonical_sha256(gamma)
        or str(row[5]) != _canonical_json(clob)
        or str(row[6]) != _canonical_sha256(clob)
        or str(row[8]) != _canonical_json(evidence)
        or evidence != expected
        or str(row[9]) != _canonical_sha256(expected)
        or summary != (claim_sha256, str(row[9]), str(row[2]))
    ):
        raise ValueError("Round 22 target evidence differs")
    return str(row[2])


@dataclass(frozen=True, slots=True)
class Round22TargetCollectionResult:
    population_count: int
    existing_count: int
    collected_count: int
    remaining_count: int
    up_count: int
    down_count: int
    claim_sha256: str
    trading_authority: bool = False


def collect_round22_diagnostic_targets(
    pilot_store: Round22PilotStore,
    *,
    client: Round22PublicTargetClient | None = None,
    maximum_conditions: int = 36,
    progress: ProgressCallback | None = None,
) -> Round22TargetCollectionResult:
    if pilot_store.read_only:
        raise ValueError("Round 22 target collection requires a writable store")
    if type(maximum_conditions) is not int or not 1 <= maximum_conditions <= 36:
        raise ValueError("Round 22 target collection limit differs")
    claim = _claim(pilot_store)
    claim_sha = str(claim[9])
    artifact = load_round22_diagnostic_preregistration(pilot_store.contract.repository)
    conditions = _population(artifact)
    existing = {
        str(row[0])
        for row in pilot_store.connection.execute(
            "SELECT condition_id FROM target.round22_resolution_evidence"
        ).fetchall()
    }
    for condition_id in existing:
        _audit_target(
            pilot_store,
            condition_id=condition_id,
            claim_sha256=claim_sha,
        )
    pending = tuple(
        item for item in conditions if str(item["condition_id"]) not in existing
    )
    selected = pending[:maximum_conditions]
    target_client = client or Round22PublicTargetClient()
    collected = 0
    for index, item in enumerate(selected, start=1):
        condition_id = str(item["condition_id"])
        market = pilot_store.market(condition_id)
        if progress is not None:
            progress(
                "target_dual_source_fetch",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "role": market.role,
                    "slug": market.slug,
                },
            )
        gamma = target_client.gamma_market(market.market_id)
        clob = target_client.clob_market(market.condition_id)
        observed = max(gamma.observed_at_ms, clob.observed_at_ms)
        winner = validate_official_resolution(
            market.official_market(gamma.value),
            clob.value,
            gamma.value,
            observed_wall_ms=observed,
        )
        if winner is None:
            raise ValueError("Round 22 diagnostic target is not terminal")
        evidence = {
            "access_claim_sha256": claim_sha,
            "clob_payload_sha256": clob.sha256,
            "condition_id": condition_id,
            "gamma_payload_sha256": gamma.sha256,
            "observed_at_ms": observed,
            "role": market.role,
            "schema_version": POLYMARKET_ROUND22_TARGET_SCHEMA_VERSION,
            "winning_outcome": winner[1],
            "winning_token_id": winner[0],
        }
        evidence_sha = _canonical_sha256(evidence)
        transaction_started = False
        try:
            pilot_store.connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            pilot_store.connection.execute(
                "INSERT INTO target.round22_resolution_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    condition_id,
                    market.role,
                    winner[0],
                    winner[1],
                    gamma.canonical_json,
                    gamma.sha256,
                    clob.canonical_json,
                    clob.sha256,
                    observed,
                    _canonical_json(evidence),
                    evidence_sha,
                ],
            )
            pilot_store.connection.execute(
                "INSERT INTO target.official_resolution VALUES (?, ?, ?, ?)",
                [condition_id, claim_sha, evidence_sha, winner[1]],
            )
            pilot_store.connection.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                pilot_store.connection.execute("ROLLBACK")
            raise
        collected += 1
        if progress is not None:
            progress(
                "target_condition_committed",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "condition_id": condition_id,
                },
            )
    all_rows = pilot_store.connection.execute(
        "SELECT condition_id, winning_outcome FROM target.round22_resolution_evidence"
    ).fetchall()
    outcomes = {str(condition_id): str(outcome) for condition_id, outcome in all_rows}
    if set(outcomes) - {str(item["condition_id"]) for item in conditions}:
        raise ValueError("Round 22 diagnostic target population expanded")
    for condition_id in outcomes:
        _audit_target(
            pilot_store,
            condition_id=condition_id,
            claim_sha256=claim_sha,
        )
    remaining = len(conditions) - len(outcomes)
    pilot_store.connection.execute(
        "UPDATE target.round22_access_claim SET status = ? WHERE singleton",
        ["complete" if remaining == 0 else "opened"],
    )
    return Round22TargetCollectionResult(
        population_count=len(conditions),
        existing_count=len(existing),
        collected_count=collected,
        remaining_count=remaining,
        up_count=sum(value == "Up" for value in outcomes.values()),
        down_count=sum(value == "Down" for value in outcomes.values()),
        claim_sha256=claim_sha,
    )


__all__ = [
    "POLYMARKET_ROUND22_CLOB_MARKET_URL",
    "POLYMARKET_ROUND22_DIAGNOSTIC_RELATIVE",
    "POLYMARKET_ROUND22_DIAGNOSTIC_SHA256",
    "POLYMARKET_ROUND22_GAMMA_MARKET_URL",
    "POLYMARKET_ROUND22_TARGET_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND22_TARGET_SCHEMA_VERSION",
    "Round22PublicPayload",
    "Round22PublicTargetClient",
    "Round22TargetCollectionResult",
    "collect_round22_diagnostic_targets",
    "load_round22_diagnostic_preregistration",
    "open_round22_diagnostic_target_claim",
    "round22_target_implementation_manifest",
    "validate_round22_diagnostic_target_opening",
]
