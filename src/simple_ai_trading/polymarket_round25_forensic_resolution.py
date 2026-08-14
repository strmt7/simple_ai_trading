"""Two-stage official targets for the Round 25 forensic diagnostic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import duckdb
import zstandard

from .polymarket_resolution import validate_official_resolution
from .polymarket_round25_forensic_partition import (
    validate_round25_forensic_partition_manifest,
)
from .polymarket_round25_joint_materialization import Round25JointReceiptCondition
from .polymarket_round25_joint_store import (
    audit_round25_joint_store,
    load_round25_joint_condition_identities,
)
from .polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
    Round25ResolutionPublicClient,
    Round25ResolutionTransportError,
)


POLYMARKET_ROUND25_FORENSIC_RESOLUTION_SCHEMA_VERSION = (
    "polymarket-round25-forensic-resolution-collection-v1"
)
POLYMARKET_ROUND25_FORENSIC_RESOLUTION_EVIDENCE_SCHEMA_VERSION = (
    "polymarket-round25-forensic-resolution-evidence-v1"
)
POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION = (
    "polymarket-round25-v2-forensic-selection-freeze-v1"
)
POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256 = (
    "f80d0396d7b51afdc63868a1e259099c2621ef45af7a3a31ab28b64967534896"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_TABLES = {
    "round25_forensic_resolution_claim",
    "round25_forensic_resolution_condition",
    "round25_forensic_resolution_evidence",
}
_MAXIMUM_PAYLOAD_BYTES = 2 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 forensic resolution JSON has duplicate keys")
        output[key] = value
    return output


def _strict_json(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 25 forensic {label} is not canonical JSON")
    decoded = json.loads(value, object_pairs_hook=_strict_object)
    if not isinstance(decoded, Mapping) or _canonical_json(decoded) != value:
        raise ValueError(f"Round 25 forensic {label} differs")
    return decoded


def _hash_chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name, table_type FROM information_schema.tables
        WHERE table_schema = 'main' ORDER BY table_name
        """
    ).fetchall()
    if any(str(row[1]) != "BASE TABLE" for row in rows):
        raise ValueError("Round 25 forensic resolution contains a non-table object")
    return {str(row[0]) for row in rows}


def _condition_payload(
    condition: Round25JointReceiptCondition,
    *,
    role: str,
) -> dict[str, object]:
    selected = condition.validated()
    return {
        "condition_id": selected.condition_id,
        "down_token_id": selected.down_token_id,
        "event_end_ms": selected.event_end_ms,
        "event_start_ms": selected.event_start_ms,
        "market_id": selected.market_id,
        "resolution_source": selected.resolution_source,
        "role": role,
        "slug": selected.slug,
        "up_token_id": selected.up_token_id,
    }


def _condition_population_sha256(
    conditions: Sequence[tuple[Round25JointReceiptCondition, str]],
) -> str:
    chain = _EMPTY_SHA256
    for condition, role in conditions:
        chain = _hash_chain(chain, _condition_payload(condition, role=role))
    if not conditions or chain == _EMPTY_SHA256:
        raise ValueError("Round 25 forensic resolution population is empty")
    return chain


def validate_round25_forensic_selection_freeze(
    value: Mapping[str, object],
    *,
    partition_manifest: Mapping[str, object],
) -> dict[str, object]:
    partition = validate_round25_forensic_partition_manifest(partition_manifest)
    payload = dict(value)
    claimed = str(payload.pop("freeze_sha256", "")).strip().lower()
    expected = {
        "condition_count",
        "created_at_ms",
        "evaluation_contract_sha256",
        "feature_store_manifest_sha256",
        "partition_sha256",
        "prediction_population_sha256",
        "profitability_claim",
        "schema_version",
        "selected_candidate_id",
        "selection_predictions_frozen",
        "trade_policy_sha256",
    }
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION
        or payload.get("evaluation_contract_sha256")
        != POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        or payload.get("feature_store_manifest_sha256")
        != partition["feature_store_manifest_sha256"]
        or payload.get("partition_sha256") != partition["partition_sha256"]
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= partition["created_at_ms"]
        or type(payload.get("condition_count")) is not int
        or payload["condition_count"] != partition["role_counts"]["selection"]
        or not isinstance(payload.get("selected_candidate_id"), str)
        or not payload["selected_candidate_id"]
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in ("prediction_population_sha256", "trade_policy_sha256")
        )
        or payload.get("selection_predictions_frozen") is not True
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("Round 25 forensic selection freeze differs")
    return {**payload, "freeze_sha256": claimed}


def _validate_claim(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("claim_sha256", "")).strip().lower()
    expected = {
        "condition_count",
        "condition_population_sha256",
        "created_at_ms",
        "evaluation_contract_sha256",
        "feature_store_manifest_sha256",
        "live_trading_authority",
        "paper_trading_authority",
        "partition_sha256",
        "profitability_claim",
        "role_counts",
        "schema_version",
        "selection_freeze_sha256",
        "stage",
        "target_access_opened",
    }
    stage = payload.get("stage")
    roles = payload.get("role_counts")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_RESOLUTION_SCHEMA_VERSION
        or payload.get("evaluation_contract_sha256")
        != POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        or stage not in {"fit", "selection"}
        or type(payload.get("condition_count")) is not int
        or payload["condition_count"] <= 0
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= 0
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in (
                "condition_population_sha256",
                "feature_store_manifest_sha256",
                "partition_sha256",
            )
        )
        or not isinstance(roles, Mapping)
        or any(type(value) is not int or value <= 0 for value in roles.values())
        or sum(roles.values()) != payload["condition_count"]
        or set(roles) != ({"train", "calibration"} if stage == "fit" else {"selection"})
        or (stage == "fit" and payload.get("selection_freeze_sha256") is not None)
        or (
            stage == "selection"
            and _SHA256.fullmatch(str(payload.get("selection_freeze_sha256") or ""))
            is None
        )
        or payload.get("target_access_opened") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "live_trading_authority",
                "paper_trading_authority",
                "profitability_claim",
            )
        )
    ):
        raise ValueError("Round 25 forensic resolution claim differs")
    return {**payload, "claim_sha256": claimed}


def initialize_round25_forensic_resolution_collection(
    *,
    feature_database: str | Path,
    partition_manifest: Mapping[str, object],
    destination_database: str | Path,
    stage: str,
    created_at_ms: int,
    selection_freeze: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Open either fit targets or sealed selection targets, never both."""

    feature = Path(feature_database)
    destination = Path(destination_database)
    selected_stage = str(stage or "").strip().lower()
    if selected_stage not in {"fit", "selection"}:
        raise ValueError("Round 25 forensic resolution stage differs")
    feature_manifest = audit_round25_joint_store(feature)
    partition = validate_round25_forensic_partition_manifest(
        partition_manifest,
        expected_feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
    )
    freeze = None
    if selected_stage == "selection":
        if selection_freeze is None:
            raise ValueError("Round 25 forensic selection predictions are not frozen")
        freeze = validate_round25_forensic_selection_freeze(
            selection_freeze,
            partition_manifest=partition,
        )
    elif selection_freeze is not None:
        raise ValueError("Round 25 forensic fit access cannot bind selection predictions")
    roles = {"train", "calibration"} if selected_stage == "fit" else {"selection"}
    partition_rows = {
        str(row["condition_id"]): (int(row["event_start_ms"]), str(row["role"]))
        for row in partition["conditions"]
        if row["role"] in roles
    }
    source = load_round25_joint_condition_identities(feature)
    source_by_id = {condition.condition_id: condition for condition in source}
    if set(partition_rows) - set(source_by_id):
        raise ValueError("Round 25 forensic resolution source population differs")
    conditions = tuple(
        (source_by_id[condition_id], role)
        for condition_id, (event_start_ms, role) in sorted(
            partition_rows.items(), key=lambda item: (item[1][0], item[0])
        )
        if source_by_id[condition_id].event_start_ms == event_start_ms
    )
    if len(conditions) != len(partition_rows):
        raise ValueError("Round 25 forensic resolution chronology differs")
    population_sha256 = _condition_population_sha256(conditions)
    role_counts = Counter(role for _, role in conditions)
    body: dict[str, object] = {
        "condition_count": len(conditions),
        "condition_population_sha256": population_sha256,
        "created_at_ms": created_at_ms,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        ),
        "feature_store_manifest_sha256": feature_manifest["manifest_sha256"],
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "partition_sha256": partition["partition_sha256"],
        "profitability_claim": False,
        "role_counts": {role: role_counts[role] for role in sorted(role_counts)},
        "schema_version": POLYMARKET_ROUND25_FORENSIC_RESOLUTION_SCHEMA_VERSION,
        "selection_freeze_sha256": None if freeze is None else freeze["freeze_sha256"],
        "stage": selected_stage,
        "target_access_opened": True,
    }
    claim = _validate_claim({**body, "claim_sha256": _canonical_sha256(body)})
    if created_at_ms <= max(condition.event_end_ms for condition, _ in conditions):
        raise ValueError("Round 25 forensic resolution access predates a market end")
    if destination.is_symlink() or Path(f"{destination}.wal").exists():
        raise ValueError("Round 25 forensic resolution destination differs")
    if destination.exists():
        with duckdb.connect(str(destination), read_only=True) as connection:
            existing, existing_conditions = _load_collection(connection)
            expected_existing_body = {
                **body,
                "created_at_ms": existing["created_at_ms"],
            }
            expected_existing = _validate_claim(
                {
                    **expected_existing_body,
                    "claim_sha256": _canonical_sha256(expected_existing_body),
                }
            )
            if (
                existing != expected_existing
                or _condition_population_sha256(existing_conditions)
                != population_sha256
            ):
                raise ValueError("Round 25 forensic resolution claim drifted")
        return destination, existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    initializing = destination.with_name(f".{destination.name}.initializing")
    if initializing.exists() or Path(f"{initializing}.wal").exists():
        raise ValueError("Round 25 forensic resolution initialization differs")
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(initializing))
        connection.execute("SET memory_limit = '512MB'")
        connection.execute("SET threads = 2")
        connection.execute(
            """
            CREATE TABLE round25_forensic_resolution_claim (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                claim_json VARCHAR NOT NULL,
                claim_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_forensic_resolution_condition (
                condition_id VARCHAR PRIMARY KEY,
                event_start_ms BIGINT NOT NULL,
                role VARCHAR NOT NULL,
                identity_json VARCHAR NOT NULL,
                identity_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE round25_forensic_resolution_evidence (
                condition_id VARCHAR PRIMARY KEY,
                evidence_json VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                gamma_payload BLOB NOT NULL,
                clob_payload BLOB NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO round25_forensic_resolution_claim VALUES (TRUE, ?, ?)",
            [_canonical_json(claim), claim["claim_sha256"]],
        )
        for condition, role in conditions:
            identity = _condition_payload(condition, role=role)
            connection.execute(
                "INSERT INTO round25_forensic_resolution_condition VALUES (?, ?, ?, ?, ?)",
                [
                    condition.condition_id,
                    condition.event_start_ms,
                    role,
                    _canonical_json(identity),
                    _canonical_sha256(identity),
                ],
            )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        os.replace(initializing, destination)
    finally:
        if connection is not None:
            connection.close()
        for path in (initializing, Path(f"{initializing}.wal")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return destination, claim


def _compress(payload: Round25OfficialPublicPayload) -> bytes:
    raw = payload.canonical_json.encode("ascii")
    if (
        not 2 <= len(raw) <= _MAXIMUM_PAYLOAD_BYTES
        or hashlib.sha256(raw).hexdigest() != payload.sha256
        or _canonical_json(payload.value) != payload.canonical_json
    ):
        raise ValueError("Round 25 forensic official payload differs")
    return zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)


def _condition_from_identity(value: Mapping[str, object]) -> tuple[Round25JointReceiptCondition, str]:
    expected = {
        "condition_id",
        "down_token_id",
        "event_end_ms",
        "event_start_ms",
        "market_id",
        "resolution_source",
        "role",
        "slug",
        "up_token_id",
    }
    if set(value) != expected or value.get("role") not in {"train", "calibration", "selection"}:
        raise ValueError("Round 25 forensic resolution identity differs")
    start = int(value["event_start_ms"])
    condition = Round25JointReceiptCondition(
        run_id="0" * 32,
        segment_index=0,
        snapshot_sha256="0" * 64,
        snapshot_observed_wall_ms=start - 1,
        market_id=str(value["market_id"]),
        condition_id=str(value["condition_id"]),
        slug=str(value["slug"]),
        event_start_ms=start,
        event_end_ms=int(value["event_end_ms"]),
        up_token_id=str(value["up_token_id"]),
        down_token_id=str(value["down_token_id"]),
        resolution_source=str(value["resolution_source"]),
        role="train",
    ).validated()
    return condition, str(value["role"])


def _load_collection(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, object], tuple[tuple[Round25JointReceiptCondition, str], ...]]:
    if _tables(connection) != _TABLES:
        raise ValueError("Round 25 forensic resolution schema differs")
    claim_row = connection.execute(
        "SELECT claim_json, claim_sha256 FROM round25_forensic_resolution_claim"
    ).fetchone()
    if claim_row is None:
        raise ValueError("Round 25 forensic resolution claim is unavailable")
    claim = _validate_claim(_strict_json(claim_row[0], label="resolution claim"))
    if claim_row[1] != claim["claim_sha256"]:
        raise ValueError("Round 25 forensic resolution claim hash differs")
    conditions: list[tuple[Round25JointReceiptCondition, str]] = []
    for row in connection.execute(
        """
        SELECT condition_id, event_start_ms, role, identity_json, identity_sha256
        FROM round25_forensic_resolution_condition
        ORDER BY event_start_ms, condition_id
        """
    ).fetchall():
        identity = _strict_json(row[3], label="resolution condition")
        condition, role = _condition_from_identity(identity)
        if (
            row[0] != condition.condition_id
            or row[1] != condition.event_start_ms
            or row[2] != role
            or row[4] != _canonical_sha256(identity)
        ):
            raise ValueError("Round 25 forensic resolution condition row differs")
        conditions.append((condition, role))
    selected = tuple(conditions)
    if (
        len(selected) != claim["condition_count"]
        or _condition_population_sha256(selected)
        != claim["condition_population_sha256"]
    ):
        raise ValueError("Round 25 forensic resolution population differs")
    return claim, selected


def collect_round25_forensic_resolutions_once(
    *,
    collection_database: str | Path,
    client: Round25ResolutionPublicClient,
    maximum_conditions: int = 128,
) -> dict[str, object]:
    """Collect one bounded public batch; unresolved markets remain pending."""

    if type(maximum_conditions) is not int or not 1 <= maximum_conditions <= 512:
        raise ValueError("Round 25 forensic resolution batch size differs")
    path = Path(collection_database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 forensic resolution collection differs")
    connection = duckdb.connect(str(path))
    connection.execute("SET memory_limit = '512MB'")
    connection.execute("SET threads = 2")
    claim, conditions = _load_collection(connection)
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT condition_id FROM round25_forensic_resolution_evidence"
        ).fetchall()
    }
    selected = [item for item in conditions if item[0].condition_id not in existing][
        :maximum_conditions
    ]
    inserted = 0
    unresolved = 0
    transport_failures = 0
    try:
        for condition, role in selected:
            try:
                gamma = client.gamma_market(condition.market_id)
                clob = client.clob_market(condition.condition_id)
            except Round25ResolutionTransportError:
                transport_failures += 1
                continue
            observed = max(gamma.observed_wall_ms, clob.observed_wall_ms)
            winner = validate_official_resolution(
                condition,
                clob.value,
                gamma.value,
                observed_wall_ms=observed,
            )
            if winner is None:
                unresolved += 1
                continue
            gamma_compressed = _compress(gamma)
            clob_compressed = _compress(clob)
            body = {
                "clob_compressed_sha256": hashlib.sha256(clob_compressed).hexdigest(),
                "clob_payload_sha256": clob.sha256,
                "condition_id": condition.condition_id,
                "gamma_compressed_sha256": hashlib.sha256(gamma_compressed).hexdigest(),
                "gamma_payload_sha256": gamma.sha256,
                "observed_wall_ms": observed,
                "official_payload_sha256": _canonical_sha256(
                    {"clob": clob.sha256, "gamma": gamma.sha256}
                ),
                "role": role,
                "schema_version": (
                    POLYMARKET_ROUND25_FORENSIC_RESOLUTION_EVIDENCE_SCHEMA_VERSION
                ),
                "winning_outcome": winner[1],
                "winning_token_id": winner[0],
            }
            evidence = {**body, "evidence_sha256": _canonical_sha256(body)}
            connection.execute(
                "INSERT INTO round25_forensic_resolution_evidence VALUES (?, ?, ?, ?, ?)",
                [
                    condition.condition_id,
                    _canonical_json(evidence),
                    evidence["evidence_sha256"],
                    gamma_compressed,
                    clob_compressed,
                ],
            )
            inserted += 1
        connection.execute("CHECKPOINT")
        completed = len(existing) + inserted
        return {
            "attempted_count": len(selected),
            "claim_sha256": claim["claim_sha256"],
            "complete": completed == len(conditions),
            "inserted_count": inserted,
            "pending_count": len(conditions) - completed,
            "stage": claim["stage"],
            "transport_failure_count": transport_failures,
            "unresolved_count": unresolved,
        }
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class Round25ForensicResolutionTarget:
    condition_id: str
    event_start_ms: int
    role: str
    target_up: bool
    resolved_at_ms: int
    official_payload_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.role not in {"train", "calibration", "selection"}
            or type(self.target_up) is not bool
            or type(self.resolved_at_ms) is not int
            or self.resolved_at_ms < self.event_start_ms + 300_000
            or _SHA256.fullmatch(self.official_payload_sha256) is None
            or _SHA256.fullmatch(self.evidence_sha256) is None
        ):
            raise ValueError("Round 25 forensic resolution target differs")


def _decompress(value: bytes, expected_sha256: str) -> Mapping[str, object]:
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise ValueError("Round 25 forensic compressed payload differs")
    raw = zstandard.ZstdDecompressor().decompress(
        value,
        max_output_size=_MAXIMUM_PAYLOAD_BYTES,
    )
    return _strict_json(raw.decode("ascii"), label="official payload")


def load_round25_forensic_resolution_targets(
    collection_database: str | Path,
) -> tuple[dict[str, object], tuple[Round25ForensicResolutionTarget, ...]]:
    """Deep-verify dual-source payloads and return targets only when complete."""

    path = Path(collection_database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 forensic resolution collection differs")
    with duckdb.connect(str(path), read_only=True) as connection:
        claim, conditions = _load_collection(connection)
        rows = connection.execute(
            """
            SELECT condition_id, evidence_json, evidence_sha256,
                   gamma_payload, clob_payload
            FROM round25_forensic_resolution_evidence ORDER BY condition_id
            """
        ).fetchall()
    if len(rows) != len(conditions):
        raise RuntimeError("Round 25 forensic resolution collection is incomplete")
    evidence_by_id = {str(row[0]): row for row in rows}
    targets: list[Round25ForensicResolutionTarget] = []
    for condition, role in conditions:
        row = evidence_by_id.get(condition.condition_id)
        if row is None:
            raise ValueError("Round 25 forensic resolution evidence is missing")
        evidence = dict(_strict_json(row[1], label="resolution evidence"))
        claimed = str(evidence.pop("evidence_sha256", ""))
        expected = {
            "clob_compressed_sha256",
            "clob_payload_sha256",
            "condition_id",
            "gamma_compressed_sha256",
            "gamma_payload_sha256",
            "observed_wall_ms",
            "official_payload_sha256",
            "role",
            "schema_version",
            "winning_outcome",
            "winning_token_id",
        }
        if (
            set(evidence) != expected
            or row[2] != claimed
            or claimed != _canonical_sha256(evidence)
            or _SHA256.fullmatch(claimed) is None
            or evidence.get("condition_id") != condition.condition_id
            or evidence.get("role") != role
            or evidence.get("schema_version")
            != POLYMARKET_ROUND25_FORENSIC_RESOLUTION_EVIDENCE_SCHEMA_VERSION
            or type(evidence.get("observed_wall_ms")) is not int
            or evidence["observed_wall_ms"] < condition.event_end_ms
            or evidence.get("winning_outcome") not in {"Up", "Down"}
            or evidence.get("winning_token_id")
            not in {condition.up_token_id, condition.down_token_id}
            or any(
                _SHA256.fullmatch(str(evidence.get(field) or "")) is None
                for field in (
                    "clob_compressed_sha256",
                    "clob_payload_sha256",
                    "gamma_compressed_sha256",
                    "gamma_payload_sha256",
                    "official_payload_sha256",
                )
            )
        ):
            raise ValueError("Round 25 forensic resolution evidence differs")
        gamma = _decompress(bytes(row[3]), str(evidence["gamma_compressed_sha256"]))
        clob = _decompress(bytes(row[4]), str(evidence["clob_compressed_sha256"]))
        if (
            _canonical_sha256(gamma) != evidence["gamma_payload_sha256"]
            or _canonical_sha256(clob) != evidence["clob_payload_sha256"]
            or evidence["official_payload_sha256"]
            != _canonical_sha256(
                {
                    "clob": evidence["clob_payload_sha256"],
                    "gamma": evidence["gamma_payload_sha256"],
                }
            )
        ):
            raise ValueError("Round 25 forensic official payload hash differs")
        winner = validate_official_resolution(
            condition,
            clob,
            gamma,
            observed_wall_ms=int(evidence["observed_wall_ms"]),
        )
        if winner != (evidence["winning_token_id"], evidence["winning_outcome"]):
            raise ValueError("Round 25 forensic official sources disagree")
        targets.append(
            Round25ForensicResolutionTarget(
                condition_id=condition.condition_id,
                event_start_ms=condition.event_start_ms,
                role=role,
                target_up=winner[1] == "Up",
                resolved_at_ms=int(evidence["observed_wall_ms"]),
                official_payload_sha256=str(evidence["official_payload_sha256"]),
                evidence_sha256=claimed,
            )
        )
    return claim, tuple(targets)


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_FORENSIC_RESOLUTION_EVIDENCE_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_FORENSIC_RESOLUTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION",
    "Round25ForensicResolutionTarget",
    "collect_round25_forensic_resolutions_once",
    "initialize_round25_forensic_resolution_collection",
    "load_round25_forensic_resolution_targets",
    "validate_round25_forensic_selection_freeze",
]
