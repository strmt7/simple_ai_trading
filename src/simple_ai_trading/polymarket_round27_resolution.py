"""Target-isolated official settlement evidence for Polymarket Round 27."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Protocol

import duckdb
import zstandard

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_resolution import validate_official_resolution
from .polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
    Round25ResolutionPublicClient,
)
from .polymarket_round27_mechanics import _load_claim, _validate_stage0_lineage


POLYMARKET_ROUND27_RESOLUTION_CLAIM_SCHEMA_VERSION = (
    "polymarket-round27-stage0-resolution-access-claim-v1"
)
POLYMARKET_ROUND27_RESOLUTION_EVIDENCE_SCHEMA_VERSION = (
    "polymarket-round27-stage0-official-resolution-evidence-v1"
)
POLYMARKET_ROUND27_RESOLUTION_AUDIT_SCHEMA_VERSION = (
    "polymarket-round27-stage0-resolution-mechanics-audit-v1"
)
POLYMARKET_ROUND27_RESOLUTION_CODEC = "canonical-json-zstd-3"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TABLES = {
    "round27_resolution_access_claim",
    "round27_resolution_condition",
    "round27_resolution_evidence",
}
_FINAL_TABLES = _TABLES | {"round27_resolution_audit"}
_REPLACE_RETRY_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class _ResolutionClient(Protocol):
    def clob_market(self, condition_id: str) -> Round25OfficialPublicPayload: ...

    def gamma_market(self, market_id: str) -> Round25OfficialPublicPayload: ...


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 resolution JSON has duplicate keys")
        output[key] = value
    return output


def _strict_json(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 27 {label} is not canonical JSON")
    try:
        decoded = json.loads(value, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 27 {label} is not strict JSON") from exc
    if not isinstance(decoded, Mapping) or _canonical_json(decoded) != value:
        raise ValueError(f"Round 27 {label} is not canonical JSON")
    return decoded


def _hash_chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _replace_with_retries(source: Path, destination: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_SECONDS):
                raise
            time.sleep(_REPLACE_RETRY_SECONDS[attempt])


def _tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name, table_type FROM information_schema.tables
        WHERE table_schema = 'main' ORDER BY table_name
        """
    ).fetchall()
    if any(str(row[1]) != "BASE TABLE" for row in rows):
        raise ValueError("Round 27 resolution database contains a non-table object")
    return {str(row[0]) for row in rows}


def _condition_payload(
    market: PolymarketFiveMinuteMarket,
    *,
    snapshot_sha256: str,
) -> dict[str, object]:
    if _SHA256.fullmatch(snapshot_sha256) is None:
        raise ValueError("Round 27 market snapshot hash differs")
    return {
        "asset": market.asset,
        "condition_id": market.condition_id,
        "down_token_id": market.down_token_id,
        "end_ms": market.end_ms,
        "event_start_ms": market.event_start_ms,
        "gamma_payload_sha256": market.gamma_payload_sha256,
        "market_id": market.market_id,
        "resolution_source": market.resolution_source,
        "slug": market.slug,
        "source_snapshot_sha256": snapshot_sha256,
        "up_token_id": market.up_token_id,
    }


def _condition_population_sha256(
    conditions: Sequence[tuple[PolymarketFiveMinuteMarket, str]],
) -> str:
    chain = _EMPTY_SHA256
    previous: tuple[int, str] | None = None
    for market, snapshot_sha256 in conditions:
        identity = (market.event_start_ms, market.condition_id)
        if previous is not None and identity <= previous:
            raise ValueError("Round 27 resolution condition chronology differs")
        previous = identity
        chain = _hash_chain(
            chain,
            _condition_payload(market, snapshot_sha256=snapshot_sha256),
        )
    if not conditions:
        raise ValueError("Round 27 resolution condition population is empty")
    return chain


def _validate_mechanics(
    mechanics: Mapping[str, object],
    *,
    lineage: Mapping[str, object],
) -> None:
    authority = mechanics.get("authority")
    interpretation = mechanics.get("interpretation")
    mechanics_lineage = mechanics.get("lineage")
    if (
        mechanics.get("schema_version") != "polymarket-round27-mechanics-diagnostic-v2"
        or not isinstance(authority, Mapping)
        or not isinstance(interpretation, Mapping)
        or not isinstance(mechanics_lineage, Mapping)
        or any(mechanics_lineage.get(key) != value for key, value in lineage.items())
        or any(
            authority.get(key) is not False
            for key in (
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or interpretation.get("mechanics_only") is not True
        or interpretation.get("edge_claim") is not False
        or interpretation.get("profitability_claim") is not False
        or interpretation.get("promotion_eligible") is not False
    ):
        raise ValueError("Round 27 mechanics freeze differs")


def _load_stage0_inputs(
    *,
    source_database: Path,
    condition_audit_path: Path,
    preregistration_path: Path,
    capture_contract_path: Path,
    capture_result_path: Path,
    mechanics_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    tuple[tuple[PolymarketFiveMinuteMarket, str], ...],
]:
    audit = _load_claim(
        condition_audit_path,
        claim="audit_sha256",
        label="Round 27 Stage 0 condition audit",
    )
    preregistration = _load_claim(
        preregistration_path,
        claim="preregistration_sha256",
        label="Round 27 preregistration",
    )
    capture_contract = _load_claim(
        capture_contract_path,
        claim="contract_sha256",
        label="Round 27 Stage 0 capture contract",
    )
    capture_result = _load_claim(
        capture_result_path,
        claim="result_sha256",
        label="Round 27 Stage 0 capture result",
    )
    mechanics = _load_claim(
        mechanics_path,
        claim="mechanics_sha256",
        label="Round 27 Stage 0 mechanics freeze",
    )
    lineage = _validate_stage0_lineage(
        audit=audit,
        preregistration=preregistration,
        capture_contract=capture_contract,
        capture_result=capture_result,
    )
    _validate_mechanics(mechanics, lineage=lineage)
    raw_ids = audit.get("eligible_condition_ids")
    raw_conditions = audit.get("conditions")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) != int(audit.get("eligible_condition_count", 0))
        or len(set(raw_ids)) != len(raw_ids)
        or not isinstance(raw_conditions, list)
    ):
        raise ValueError("Round 27 eligible condition audit differs")
    condition_rows = {
        str(item.get("condition_id")): item
        for item in raw_conditions
        if isinstance(item, Mapping) and item.get("eligible") is True
    }
    eligible_ids = tuple(str(value).lower() for value in raw_ids)
    if set(condition_rows) != set(eligible_ids):
        raise ValueError("Round 27 eligible condition identities differ")
    with PolymarketEvidenceStore(
        source_database,
        memory_limit="512MB",
        threads=2,
        read_only=True,
    ) as source:
        markets = PolymarketEvidenceReplay.load_markets(
            source,
            run_id=str(audit["run_id"]),
            condition_ids=eligible_ids,
        )
        rows = (
            source.connect()
            .execute(
                """
            SELECT condition_id, snapshot_sha256
            FROM polymarket_market_snapshot
            WHERE run_id = ? AND condition_id IN (
                SELECT unnest(?::VARCHAR[])
            )
            ORDER BY event_start_ms, condition_id
            """,
                [str(audit["run_id"]), list(eligible_ids)],
            )
            .fetchall()
        )
        run_row = (
            source.connect()
            .execute(
                """
            SELECT status, report_sha256 FROM polymarket_recorder_run
            WHERE run_id = ?
            """,
                [str(audit["run_id"])],
            )
            .fetchone()
        )
    snapshot_by_condition = {str(row[0]): str(row[1]) for row in rows}
    if (
        len(markets) != len(eligible_ids)
        or set(snapshot_by_condition) != set(eligible_ids)
        or run_row is None
        or str(run_row[0]) != audit.get("run_status")
        or str(run_row[1]) != audit.get("run_report_sha256")
    ):
        raise ValueError("Round 27 source capture identity differs")
    ordered: list[tuple[PolymarketFiveMinuteMarket, str]] = []
    for market in markets:
        audited = condition_rows[market.condition_id]
        if (
            audited.get("slug") != market.slug
            or audited.get("event_start_ms") != market.event_start_ms
            or audited.get("end_ms") != market.end_ms
            or market.asset != "BTC"
        ):
            raise ValueError("Round 27 audited market identity differs")
        ordered.append((market, snapshot_by_condition[market.condition_id]))
    ordered.sort(key=lambda item: (item[0].event_start_ms, item[0].condition_id))
    return (
        lineage,
        mechanics,
        tuple(ordered),
    )


def round27_resolution_collection_database(destination: str | Path) -> Path:
    selected = Path(destination)
    return selected.with_name(f".{selected.name}.collecting")


def _validate_claim(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("claim_sha256", "")).lower()
    expected = {
        "access_scope",
        "capture_contract_sha256",
        "capture_result_sha256",
        "condition_audit_sha256",
        "condition_count",
        "condition_population_sha256",
        "created_at_ms",
        "edge_claim",
        "live_trading_authority",
        "mechanics_sha256",
        "paper_trading_authority",
        "preregistration_sha256",
        "profitability_claim",
        "run_id",
        "schema_version",
        "source_endpoints",
        "target_access_opened",
        "target_use",
    }
    sources = payload.get("source_endpoints")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_RESOLUTION_CLAIM_SCHEMA_VERSION
        or payload.get("access_scope") != "stage0_settlement_label_mechanics"
        or payload.get("target_use") != "mechanics_validation_only"
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= 0
        or type(payload.get("condition_count")) is not int
        or payload["condition_count"] <= 0
        or not isinstance(payload.get("run_id"), str)
        or not payload["run_id"]
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in (
                "capture_contract_sha256",
                "capture_result_sha256",
                "condition_audit_sha256",
                "condition_population_sha256",
                "mechanics_sha256",
                "preregistration_sha256",
            )
        )
        or sources
        != [
            "https://clob.polymarket.com/markets/{condition_id}",
            "https://gamma-api.polymarket.com/markets/{market_id}",
        ]
        or payload.get("target_access_opened") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "edge_claim",
                "live_trading_authority",
                "paper_trading_authority",
                "profitability_claim",
            )
        )
    ):
        raise ValueError("Round 27 resolution access claim differs")
    return {**payload, "claim_sha256": claimed}


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE round27_resolution_access_claim (
            singleton BOOLEAN PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            claim_json VARCHAR NOT NULL,
            claim_sha256 VARCHAR NOT NULL UNIQUE
        );
        CREATE TABLE round27_resolution_condition (
            condition_id VARCHAR PRIMARY KEY,
            event_start_ms BIGINT NOT NULL UNIQUE,
            identity_json VARCHAR NOT NULL,
            identity_sha256 VARCHAR NOT NULL UNIQUE
        );
        CREATE TABLE round27_resolution_evidence (
            condition_id VARCHAR PRIMARY KEY,
            observed_wall_ms BIGINT NOT NULL,
            observed_monotonic_ns UBIGINT NOT NULL,
            winning_asset_id VARCHAR NOT NULL,
            winning_outcome VARCHAR NOT NULL,
            clob_payload_sha256 VARCHAR NOT NULL,
            gamma_payload_sha256 VARCHAR NOT NULL,
            clob_payload BLOB NOT NULL,
            gamma_payload BLOB NOT NULL,
            evidence_json VARCHAR NOT NULL,
            evidence_sha256 VARCHAR NOT NULL UNIQUE
        );
        """
    )


def _load_stored_claim(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT schema_version, claim_json, claim_sha256
        FROM round27_resolution_access_claim WHERE singleton
        """
    ).fetchone()
    if row is None:
        raise ValueError("Round 27 resolution access claim is unavailable")
    claim = _validate_claim(_strict_json(row[1], label="resolution access claim"))
    if claim["schema_version"] != str(row[0]) or claim["claim_sha256"] != str(row[2]):
        raise ValueError("Round 27 stored resolution access claim differs")
    return claim


def initialize_round27_resolution_collection(
    *,
    source_database: str | Path,
    condition_audit_path: str | Path,
    preregistration_path: str | Path,
    capture_contract_path: str | Path,
    capture_result_path: str | Path,
    mechanics_path: str | Path,
    destination_database: str | Path,
    created_at_ms: int,
) -> tuple[Path, dict[str, object]]:
    """Persist the exact target-access claim before any resolution request."""

    destination = Path(destination_database)
    collecting = round27_resolution_collection_database(destination)
    if destination.exists() and collecting.exists():
        raise ValueError("Round 27 resolution final and collecting databases coexist")
    lineage, mechanics, conditions = _load_stage0_inputs(
        source_database=Path(source_database),
        condition_audit_path=Path(condition_audit_path),
        preregistration_path=Path(preregistration_path),
        capture_contract_path=Path(capture_contract_path),
        capture_result_path=Path(capture_result_path),
        mechanics_path=Path(mechanics_path),
    )
    population_sha256 = _condition_population_sha256(conditions)
    if destination.exists():
        with duckdb.connect(str(destination), read_only=True) as connection:
            claim = _load_stored_claim(connection)
        return destination, claim
    if collecting.exists():
        with duckdb.connect(str(collecting)) as connection:
            if _tables(connection) != _TABLES:
                raise ValueError("Round 27 collecting resolution schema differs")
            claim = _load_stored_claim(connection)
            condition_count = int(
                connection.execute(
                    "SELECT count(*) FROM round27_resolution_condition"
                ).fetchone()[0]
            )
        if (
            claim["condition_population_sha256"] != population_sha256
            or claim["condition_count"] != len(conditions)
            or condition_count != len(conditions)
            or any(
                claim[key] != lineage[key] for key in lineage if key.endswith("sha256")
            )
            or claim["mechanics_sha256"] != mechanics["mechanics_sha256"]
            or claim["run_id"] != lineage["run_id"]
        ):
            raise ValueError("Round 27 resumed resolution lineage differs")
        return collecting, claim
    if type(created_at_ms) is not int or created_at_ms <= max(
        market.end_ms for market, _ in conditions
    ):
        raise ValueError("Round 27 resolution access must open after every market ends")
    body: dict[str, object] = {
        "access_scope": "stage0_settlement_label_mechanics",
        "capture_contract_sha256": lineage["capture_contract_sha256"],
        "capture_result_sha256": lineage["capture_result_sha256"],
        "condition_audit_sha256": lineage["condition_audit_sha256"],
        "condition_count": len(conditions),
        "condition_population_sha256": population_sha256,
        "created_at_ms": created_at_ms,
        "edge_claim": False,
        "live_trading_authority": False,
        "mechanics_sha256": mechanics["mechanics_sha256"],
        "paper_trading_authority": False,
        "preregistration_sha256": lineage["preregistration_sha256"],
        "profitability_claim": False,
        "run_id": lineage["run_id"],
        "schema_version": POLYMARKET_ROUND27_RESOLUTION_CLAIM_SCHEMA_VERSION,
        "source_endpoints": [
            "https://clob.polymarket.com/markets/{condition_id}",
            "https://gamma-api.polymarket.com/markets/{market_id}",
        ],
        "target_access_opened": True,
        "target_use": "mechanics_validation_only",
    }
    claim = _validate_claim({**body, "claim_sha256": _canonical_sha256(body)})
    collecting.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(collecting)) as connection:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        _create_schema(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "INSERT INTO round27_resolution_access_claim VALUES (?, ?, ?, ?)",
                [
                    True,
                    claim["schema_version"],
                    _canonical_json(claim),
                    claim["claim_sha256"],
                ],
            )
            for market, snapshot_sha256 in conditions:
                identity = _condition_payload(
                    market,
                    snapshot_sha256=snapshot_sha256,
                )
                connection.execute(
                    "INSERT INTO round27_resolution_condition VALUES (?, ?, ?, ?)",
                    [
                        market.condition_id,
                        market.event_start_ms,
                        _canonical_json(identity),
                        _canonical_sha256(identity),
                    ],
                )
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return collecting, claim


def _payload(value: object, *, label: str) -> Round25OfficialPublicPayload:
    if not isinstance(value, Round25OfficialPublicPayload):
        raise ValueError(f"Round 27 {label} client omitted the public receipt envelope")
    if (
        _canonical_json(value.value) != value.canonical_json
        or hashlib.sha256(value.canonical_json.encode("ascii")).hexdigest()
        != value.sha256
        or min(value.observed_wall_ms, value.observed_monotonic_ns) <= 0
    ):
        raise ValueError(f"Round 27 {label} public receipt differs")
    return value


def _load_condition_markets(
    source_database: Path,
    *,
    run_id: str,
    condition_ids: Sequence[str],
) -> dict[str, PolymarketFiveMinuteMarket]:
    with PolymarketEvidenceStore(
        source_database,
        memory_limit="256MB",
        threads=1,
        read_only=True,
    ) as source:
        markets = PolymarketEvidenceReplay.load_markets(
            source,
            run_id=run_id,
            condition_ids=condition_ids,
        )
    return {market.condition_id: market for market in markets}


def collect_round27_resolutions_once(
    *,
    source_database: str | Path,
    collection_database: str | Path,
    client: _ResolutionClient | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Collect each unresolved official label once under the persisted claim."""

    path = Path(collection_database)
    if not path.is_file():
        raise ValueError("Round 27 resolution collection database is unavailable")
    selected_client = client or Round25ResolutionPublicClient()
    compressor = zstandard.ZstdCompressor(level=3)
    with duckdb.connect(str(path)) as connection:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        if _tables(connection) != _TABLES:
            raise ValueError("Round 27 collecting resolution schema differs")
        claim = _load_stored_claim(connection)
        rows = connection.execute(
            """
            SELECT condition_id FROM round27_resolution_condition
            WHERE condition_id NOT IN (
                SELECT condition_id FROM round27_resolution_evidence
            ) ORDER BY event_start_ms, condition_id
            """
        ).fetchall()
        unresolved = tuple(str(row[0]) for row in rows)
        markets = (
            _load_condition_markets(
                Path(source_database),
                run_id=str(claim["run_id"]),
                condition_ids=unresolved,
            )
            if unresolved
            else {}
        )
        if set(markets) != set(unresolved):
            raise ValueError("Round 27 resolution source market population differs")
        newly_resolved = 0
        pending: list[str] = []
        for index, condition_id in enumerate(unresolved, start=1):
            market = markets[condition_id]
            clob = _payload(
                selected_client.clob_market(condition_id),
                label="CLOB resolution",
            )
            gamma = _payload(
                selected_client.gamma_market(market.market_id),
                label="Gamma resolution",
            )
            if min(clob.observed_wall_ms, gamma.observed_wall_ms) < int(
                claim["created_at_ms"]
            ):
                raise ValueError("Round 27 target receipt predates its access claim")
            observed_wall_ms = max(clob.observed_wall_ms, gamma.observed_wall_ms)
            observed_monotonic_ns = max(
                clob.observed_monotonic_ns,
                gamma.observed_monotonic_ns,
            )
            winner = validate_official_resolution(
                market,
                clob.value,
                gamma.value,
                observed_wall_ms=observed_wall_ms,
            )
            if winner is None:
                pending.append(condition_id)
                continue
            evidence = {
                "claim_sha256": claim["claim_sha256"],
                "clob_payload_sha256": clob.sha256,
                "condition_id": condition_id,
                "gamma_payload_sha256": gamma.sha256,
                "observed_monotonic_ns": observed_monotonic_ns,
                "observed_wall_ms": observed_wall_ms,
                "schema_version": POLYMARKET_ROUND27_RESOLUTION_EVIDENCE_SCHEMA_VERSION,
                "winning_asset_id": winner[0],
                "winning_outcome": winner[1],
            }
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO round27_resolution_evidence VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        condition_id,
                        observed_wall_ms,
                        observed_monotonic_ns,
                        winner[0],
                        winner[1],
                        clob.sha256,
                        gamma.sha256,
                        compressor.compress(clob.canonical_json.encode("ascii")),
                        compressor.compress(gamma.canonical_json.encode("ascii")),
                        _canonical_json(evidence),
                        _canonical_sha256(evidence),
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            newly_resolved += 1
            if progress is not None:
                progress(
                    "condition_resolved",
                    {
                        "completed": index,
                        "condition_id": condition_id,
                        "total": len(unresolved),
                        "winning_outcome": winner[1],
                    },
                )
        total = int(
            connection.execute(
                "SELECT count(*) FROM round27_resolution_evidence"
            ).fetchone()[0]
        )
    return {
        "claim_sha256": claim["claim_sha256"],
        "condition_count": claim["condition_count"],
        "newly_resolved_condition_count": newly_resolved,
        "pending_condition_count": len(pending),
        "pending_condition_ids": pending,
        "resolved_condition_count": total,
        "status": "complete" if total == claim["condition_count"] else "pending",
    }


def _decode_payload(
    compressed: object,
    expected_sha256: str,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(compressed, bytes):
        raise ValueError(f"Round 27 {label} payload is not binary")
    try:
        raw = zstandard.ZstdDecompressor(max_window_size=2_048).decompress(
            compressed,
            max_output_size=2 * 1024 * 1024,
        )
        canonical = raw.decode("ascii")
    except (UnicodeError, zstandard.ZstdError) as exc:
        raise ValueError(f"Round 27 {label} payload is invalid") from exc
    value = _strict_json(canonical, label=label)
    if hashlib.sha256(canonical.encode("ascii")).hexdigest() != expected_sha256:
        raise ValueError(f"Round 27 {label} payload hash differs")
    return value


def audit_round27_resolution_collection(
    database: str | Path,
    *,
    source_database: str | Path,
) -> dict[str, object]:
    path = Path(database)
    with duckdb.connect(str(path), read_only=True) as connection:
        present_tables = _tables(connection)
        if present_tables != _TABLES and present_tables != _FINAL_TABLES:
            raise ValueError("Round 27 resolution schema differs")
        claim = _load_stored_claim(connection)
        condition_rows = connection.execute(
            """
            SELECT condition_id, event_start_ms, identity_json, identity_sha256
            FROM round27_resolution_condition ORDER BY event_start_ms, condition_id
            """
        ).fetchall()
        evidence_rows = connection.execute(
            """
            SELECT condition_id, observed_wall_ms, observed_monotonic_ns,
                   winning_asset_id, winning_outcome, clob_payload_sha256,
                   gamma_payload_sha256, clob_payload, gamma_payload,
                   evidence_json, evidence_sha256
            FROM round27_resolution_evidence ORDER BY condition_id
            """
        ).fetchall()
        stored_audit = None
        if present_tables == _FINAL_TABLES:
            stored_audit = connection.execute(
                """
                SELECT schema_version, audit_json, audit_sha256
                FROM round27_resolution_audit WHERE singleton
                """
            ).fetchone()
    chain = _EMPTY_SHA256
    condition_ids: list[str] = []
    for condition_id, _start, identity_json, identity_sha256 in condition_rows:
        identity = _strict_json(identity_json, label="condition identity")
        if identity.get("condition_id") != str(condition_id) or _canonical_sha256(
            identity
        ) != str(identity_sha256):
            raise ValueError("Round 27 stored condition identity differs")
        chain = _hash_chain(chain, identity)
        condition_ids.append(str(condition_id))
    if (
        chain != claim["condition_population_sha256"]
        or len(condition_ids) != claim["condition_count"]
    ):
        raise ValueError("Round 27 stored condition population differs")
    markets = _load_condition_markets(
        Path(source_database),
        run_id=str(claim["run_id"]),
        condition_ids=condition_ids,
    )
    winners: Counter[str] = Counter()
    evidence_chain = _EMPTY_SHA256
    for row in evidence_rows:
        condition_id = str(row[0])
        market = markets.get(condition_id)
        if market is None:
            raise ValueError("Round 27 resolution references an unknown condition")
        clob = _decode_payload(row[7], str(row[5]), label="CLOB resolution")
        gamma = _decode_payload(row[8], str(row[6]), label="Gamma resolution")
        winner = validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=int(row[1]),
        )
        evidence = _strict_json(row[9], label="resolution evidence")
        expected = {
            "claim_sha256": claim["claim_sha256"],
            "clob_payload_sha256": str(row[5]),
            "condition_id": condition_id,
            "gamma_payload_sha256": str(row[6]),
            "observed_monotonic_ns": int(row[2]),
            "observed_wall_ms": int(row[1]),
            "schema_version": POLYMARKET_ROUND27_RESOLUTION_EVIDENCE_SCHEMA_VERSION,
            "winning_asset_id": str(row[3]),
            "winning_outcome": str(row[4]),
        }
        if (
            winner != (str(row[3]), str(row[4]))
            or evidence != expected
            or _canonical_sha256(evidence) != str(row[10])
            or int(row[1]) < claim["created_at_ms"]
        ):
            raise ValueError("Round 27 official resolution evidence differs")
        evidence_chain = _hash_chain(evidence_chain, evidence)
        winners[str(row[4])] += 1
    resolved = len(evidence_rows)
    if resolved > len(condition_ids) or any(
        str(row[0]) not in set(condition_ids) for row in evidence_rows
    ):
        raise ValueError("Round 27 resolution evidence population differs")
    report_body: dict[str, object] = {
        "access_claim_sha256": claim["claim_sha256"],
        "condition_count": claim["condition_count"],
        "dual_source_agreement_count": resolved,
        "edge_claim": False,
        "evidence_chain_sha256": evidence_chain,
        "live_trading_authority": False,
        "mechanics_validation_complete": resolved == claim["condition_count"],
        "paper_trading_authority": False,
        "pending_condition_count": claim["condition_count"] - resolved,
        "profitability_claim": False,
        "resolution_count": resolved,
        "run_id": claim["run_id"],
        "schema_version": POLYMARKET_ROUND27_RESOLUTION_AUDIT_SCHEMA_VERSION,
        "source_disagreement_count": 0,
        "target_use": "mechanics_validation_only",
        "winner_counts": {key: winners[key] for key in sorted(winners)},
    }
    report = {**report_body, "audit_sha256": _canonical_sha256(report_body)}
    if stored_audit is not None:
        stored = _strict_json(stored_audit[1], label="stored resolution audit")
        if (
            str(stored_audit[0]) != report["schema_version"]
            or str(stored_audit[2]) != report["audit_sha256"]
            or stored != report
        ):
            raise ValueError("Round 27 stored resolution audit differs")
    return report


def finalize_round27_resolution_collection(
    *,
    collection_database: str | Path,
    destination_database: str | Path,
    source_database: str | Path,
) -> dict[str, object]:
    collection = Path(collection_database)
    destination = Path(destination_database)
    if destination.exists():
        return audit_round27_resolution_collection(
            destination,
            source_database=source_database,
        )
    report = audit_round27_resolution_collection(
        collection,
        source_database=source_database,
    )
    if report["mechanics_validation_complete"] is not True:
        raise ValueError("Round 27 resolution collection remains incomplete")
    with duckdb.connect(str(collection)) as connection:
        if _tables(connection) != _TABLES:
            raise ValueError("Round 27 collecting resolution schema differs")
        connection.execute(
            """
            CREATE TABLE round27_resolution_audit (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                audit_json VARCHAR NOT NULL,
                audit_sha256 VARCHAR NOT NULL UNIQUE
            )
            """
        )
        connection.execute(
            "INSERT INTO round27_resolution_audit VALUES (?, ?, ?, ?)",
            [
                True,
                report["schema_version"],
                _canonical_json(report),
                report["audit_sha256"],
            ],
        )
        connection.execute("CHECKPOINT")
    _replace_with_retries(collection, destination)
    verified = audit_round27_resolution_collection(
        destination,
        source_database=source_database,
    )
    if verified != report:
        raise ValueError("Round 27 finalized resolution audit differs")
    return verified


def write_round27_resolution_audit(
    path: str | Path,
    report: Mapping[str, object],
) -> Path:
    destination = Path(path)
    validated = dict(report)
    claimed = str(validated.pop("audit_sha256", "")).lower()
    if claimed != _canonical_sha256(validated):
        raise ValueError("Round 27 resolution audit hash differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(_canonical_json(report) + "\n", encoding="ascii")
    _replace_with_retries(temporary, destination)
    return destination


__all__ = [
    "POLYMARKET_ROUND27_RESOLUTION_AUDIT_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_RESOLUTION_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_RESOLUTION_CODEC",
    "POLYMARKET_ROUND27_RESOLUTION_EVIDENCE_SCHEMA_VERSION",
    "audit_round27_resolution_collection",
    "collect_round27_resolutions_once",
    "finalize_round27_resolution_collection",
    "initialize_round27_resolution_collection",
    "round27_resolution_collection_database",
    "write_round27_resolution_audit",
]
