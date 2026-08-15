"""Role-gated official target evidence for Polymarket Round 27."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import duckdb
import zstandard

from .polymarket import PolymarketFeeSchedule, PolymarketFiveMinuteMarket
from .polymarket_resolution import validate_official_resolution
from .polymarket_round25_resolution_store import Round25OfficialPublicPayload
from .polymarket_round27_experiment import (
    validate_round27_sealed_access_artifacts,
)
from .polymarket_round27_features import Round27FeatureRow
from .polymarket_round27_model import Round27RoleInterval
from .polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD,
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
)
from .polymarket_round27_model_contract import (
    POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION,
    POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
)


POLYMARKET_ROUND27_TARGET_STORE_SCHEMA_VERSION = (
    "polymarket-round27-role-gated-target-store-v1"
)
POLYMARKET_ROUND27_TARGET_ACCESS_SCHEMA_VERSION = (
    "polymarket-round27-role-target-access-v1"
)
POLYMARKET_ROUND27_TARGET_EVIDENCE_SCHEMA_VERSION = (
    "polymarket-round27-official-target-evidence-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TARGET_ROLES = frozenset({"train", "calibration", "selection", "sealed"})
_MAXIMUM_PUBLIC_PAYLOAD_BYTES = 2 * 1024 * 1024


class _ResolutionClient(Protocol):
    def clob_market(self, condition_id: str) -> Round25OfficialPublicPayload: ...

    def gamma_market(self, market_id: str) -> Round25OfficialPublicPayload: ...


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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 27 target {name} SHA-256 differs")
    return selected


def _hash_chain(previous: str, current: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(current)).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 target JSON contains duplicate keys")
        output[key] = value
    return output


def _strict_json(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 27 target {name} is not JSON")
    try:
        decoded = json.loads(value, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Round 27 target {name} is not strict JSON") from exc
    if not isinstance(decoded, dict) or value != _canonical_json(decoded):
        raise ValueError(f"Round 27 target {name} is not canonical JSON")
    return decoded


def _market_payload(
    market: PolymarketFiveMinuteMarket,
    *,
    source_snapshot_sha256: str,
    feature_row_chain_sha256: str,
) -> dict[str, object]:
    if (
        market.asset != "BTC"
        or market.end_ms - market.event_start_ms != 300_000
        or not market.condition_id.startswith("0x")
        or hashlib.sha256(market.gamma_payload_json.encode("ascii")).hexdigest()
        != market.gamma_payload_sha256
    ):
        raise ValueError("Round 27 target market identity differs")
    fee = market.fee_schedule
    return {
        "asset": market.asset,
        "condition_id": market.condition_id,
        "down_token_id": market.down_token_id,
        "end_ms": market.end_ms,
        "event_start_ms": market.event_start_ms,
        "fee_enabled": fee.enabled,
        "fee_exponent": fee.exponent,
        "fee_rate": format(fee.rate, "f"),
        "fee_rebate_rate": format(fee.rebate_rate, "f"),
        "fee_taker_only": fee.taker_only,
        "feature_row_chain_sha256": _sha256(
            feature_row_chain_sha256,
            name="feature row chain",
        ),
        "gamma_payload_json": market.gamma_payload_json,
        "gamma_payload_sha256": market.gamma_payload_sha256,
        "liquidity_quote": format(market.liquidity_quote, "f"),
        "market_id": market.market_id,
        "minimum_order_size": format(market.minimum_order_size, "f"),
        "question": market.question,
        "resolution_source": market.resolution_source,
        "slug": market.slug,
        "source_snapshot_sha256": _sha256(
            source_snapshot_sha256,
            name="source snapshot",
        ),
        "tick_size": format(market.tick_size, "f"),
        "up_token_id": market.up_token_id,
        "volume_quote": format(market.volume_quote, "f"),
    }


def _market_from_payload(value: Mapping[str, object]) -> PolymarketFiveMinuteMarket:
    try:
        market = PolymarketFiveMinuteMarket(
            asset=str(value["asset"]),
            market_id=str(value["market_id"]),
            condition_id=str(value["condition_id"]),
            slug=str(value["slug"]),
            question=str(value["question"]),
            event_start_ms=int(value["event_start_ms"]),
            end_ms=int(value["end_ms"]),
            up_token_id=str(value["up_token_id"]),
            down_token_id=str(value["down_token_id"]),
            tick_size=Decimal(str(value["tick_size"])),
            minimum_order_size=Decimal(str(value["minimum_order_size"])),
            fee_schedule=PolymarketFeeSchedule(
                enabled=value["fee_enabled"] is True,
                rate=Decimal(str(value["fee_rate"])),
                exponent=int(value["fee_exponent"]),
                taker_only=value["fee_taker_only"] is True,
                rebate_rate=Decimal(str(value["fee_rebate_rate"])),
            ),
            liquidity_quote=Decimal(str(value["liquidity_quote"])),
            volume_quote=Decimal(str(value["volume_quote"])),
            resolution_source=str(value["resolution_source"]),
            gamma_payload_sha256=_sha256(
                value["gamma_payload_sha256"],
                name="Gamma payload",
            ),
            gamma_payload_json=str(value["gamma_payload_json"]),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError("Round 27 stored target market differs") from exc
    expected = _market_payload(
        market,
        source_snapshot_sha256=str(value.get("source_snapshot_sha256") or ""),
        feature_row_chain_sha256=str(
            value.get("feature_row_chain_sha256") or ""
        ),
    )
    if dict(value) != expected:
        raise ValueError("Round 27 stored target market differs")
    return market


def _public_payload(value: object, *, name: str) -> Round25OfficialPublicPayload:
    if not isinstance(value, Round25OfficialPublicPayload):
        raise ValueError(f"Round 27 {name} omitted its receipt envelope")
    if (
        value.canonical_json != _canonical_json(value.value)
        or hashlib.sha256(value.canonical_json.encode("ascii")).hexdigest()
        != value.sha256
        or min(value.observed_wall_ms, value.observed_monotonic_ns) <= 0
        or len(value.canonical_json.encode("ascii")) > _MAXIMUM_PUBLIC_PAYLOAD_BYTES
    ):
        raise ValueError(f"Round 27 {name} receipt differs")
    return value


def _compress(canonical: str) -> bytes:
    return zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(canonical.encode("ascii"))


def _decompress(value: object, *, expected_sha256: str, name: str) -> dict[str, object]:
    if not isinstance(value, bytes):
        raise ValueError(f"Round 27 stored {name} payload differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            value,
            max_output_size=_MAXIMUM_PUBLIC_PAYLOAD_BYTES,
        )
        canonical = raw.decode("ascii")
    except (UnicodeError, zstandard.ZstdError) as exc:
        raise ValueError(f"Round 27 stored {name} payload differs") from exc
    payload = _strict_json(canonical, name=name)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Round 27 stored {name} hash differs")
    return payload


def _row_chains(
    rows: Sequence[Round27FeatureRow],
) -> dict[str, tuple[int, str]]:
    selected = tuple(row.validated() for row in rows)
    grouped: dict[str, list[Round27FeatureRow]] = {}
    for row in selected:
        grouped.setdefault(row.condition_id, []).append(row)
    output: dict[str, tuple[int, str]] = {}
    for condition_id, condition_rows in grouped.items():
        ordered = sorted(condition_rows, key=lambda row: row.decision_time_ms)
        if (
            len({row.decision_time_ms for row in ordered}) != len(ordered)
            or len({row.event_start_ms for row in ordered}) != 1
        ):
            raise ValueError("Round 27 target feature rows are duplicated")
        chain = _EMPTY_SHA256
        for row in ordered:
            chain = _hash_chain(chain, row.row_sha256)
        output[condition_id] = (ordered[0].event_start_ms, chain)
    if not output:
        raise ValueError("Round 27 target feature population is empty")
    return output


def _artifact_sha256(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    claimed = _sha256(payload.pop(field, ""), name=field)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"Round 27 target {field} differs")
    return claimed


class Round27TargetStore:
    """A separate, append-only target ledger with a sealed-role gate."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        if read_only and not self.path.is_file():
            raise ValueError("Round 27 target store is unavailable")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = bool(read_only)
        self.connection = duckdb.connect(str(self.path), read_only=self.read_only)
        self.connection.execute("SET threads=1")
        self.connection.execute("SET memory_limit='512MB'")
        self.connection.execute("SET preserve_insertion_order=false")
        if not self.read_only:
            self._initialize()
        self._verify_schema()

    def __enter__(self) -> "Round27TargetStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS round27_target_store_schema (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                feature_targets_co_located BOOLEAN NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round27_target_access (
                role VARCHAR PRIMARY KEY,
                slot_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                feature_store_audit_sha256 VARCHAR NOT NULL,
                opened_at_ms BIGINT NOT NULL,
                condition_count INTEGER NOT NULL,
                condition_population_sha256 VARCHAR NOT NULL,
                selection_claim_sha256 VARCHAR,
                selection_economic_claim_sha256 VARCHAR,
                selection_economic_report_sha256 VARCHAR,
                claim_json VARCHAR NOT NULL,
                claim_sha256 VARCHAR NOT NULL UNIQUE,
                finalized BOOLEAN NOT NULL,
                evidence_chain_sha256 VARCHAR
            );
            CREATE TABLE IF NOT EXISTS round27_target_condition (
                condition_id VARCHAR PRIMARY KEY,
                role VARCHAR NOT NULL,
                slot_id VARCHAR NOT NULL,
                event_start_ms BIGINT NOT NULL UNIQUE,
                market_json VARCHAR NOT NULL,
                market_sha256 VARCHAR NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS round27_target_evidence (
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
        self.connection.execute(
            """
            INSERT INTO round27_target_store_schema
            SELECT true, ?, false
            WHERE NOT EXISTS (SELECT 1 FROM round27_target_store_schema)
            """,
            [POLYMARKET_ROUND27_TARGET_STORE_SCHEMA_VERSION],
        )

    def _verify_schema(self) -> None:
        row = self.connection.execute(
            """
            SELECT schema_version, feature_targets_co_located
            FROM round27_target_store_schema WHERE singleton
            """
        ).fetchone()
        if row != (POLYMARKET_ROUND27_TARGET_STORE_SCHEMA_VERSION, False):
            raise ValueError("Round 27 target-store schema differs")

    def open_role(
        self,
        *,
        role: str,
        slot_id: str,
        run_id: str,
        contract: Mapping[str, object],
        feature_store_audit_sha256: str,
        role_intervals: Sequence[Mapping[str, object]],
        feature_rows: Sequence[Round27FeatureRow],
        markets: Sequence[tuple[PolymarketFiveMinuteMarket, str]],
        opened_at_ms: int,
        selection_claim: Mapping[str, object] | None = None,
        selection_economic_claim: Mapping[str, object] | None = None,
        selection_economic_report: Mapping[str, object] | None = None,
    ) -> bool:
        if self.read_only:
            raise ValueError("Round 27 target store is read-only")
        selected_role = str(role or "").lower()
        selected_slot = str(slot_id or "").lower()
        selected_run = str(run_id or "")
        contract_sha256 = _sha256(
            contract.get("contract_sha256"),
            name="model contract",
        )
        contract_body = dict(contract)
        contract_body.pop("contract_sha256", None)
        model_amendment_sha256 = contract_body.pop(
            POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD,
            None,
        )
        feature_audit_sha256 = _sha256(
            feature_store_audit_sha256,
            name="feature-store audit",
        )
        intervals = tuple(
            Round27RoleInterval.from_mapping(item) for item in role_intervals
        )
        row_chains = _row_chains(feature_rows)
        market_by_id = {market.condition_id: (market, snapshot) for market, snapshot in markets}
        if (
            selected_role not in _TARGET_ROLES
            or not selected_slot.startswith("stage1-")
            or not selected_run
            or contract_sha256 != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
            or contract.get("schema_version")
            != POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION
            or model_amendment_sha256
            != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
            or contract_sha256 != _canonical_sha256(contract_body)
            or type(opened_at_ms) is not int
            or opened_at_ms <= 0
            or len(market_by_id) != len(markets)
            or set(market_by_id) != set(row_chains)
            or any(row.validated().run_id != selected_run for row in feature_rows)
        ):
            raise ValueError("Round 27 target role population differs")
        artifacts = (
            selection_claim,
            selection_economic_claim,
            selection_economic_report,
        )
        selection_sha256: str | None = None
        economic_claim_sha256: str | None = None
        economic_report_sha256: str | None = None
        if selected_role == "sealed":
            if any(item is None for item in artifacts):
                raise ValueError("Round 27 sealed target artifacts are required")
            assert selection_claim is not None
            assert selection_economic_claim is not None
            assert selection_economic_report is not None
            validate_round27_sealed_access_artifacts(
                contract=contract,
                selection_claim=selection_claim,
                selection_economic_claim=selection_economic_claim,
                selection_economic_report=selection_economic_report,
            )
            selection_sha256 = _artifact_sha256(selection_claim, "claim_sha256")
            economic_claim_sha256 = _artifact_sha256(
                selection_economic_claim,
                "claim_sha256",
            )
            economic_report_sha256 = _artifact_sha256(
                selection_economic_report,
                "report_sha256",
            )
        elif any(item is not None for item in artifacts):
            raise ValueError("Round 27 nonsealed target artifacts are forbidden")
        condition_payloads: list[dict[str, object]] = []
        for condition_id, (event_start_ms, row_chain) in sorted(
            row_chains.items(),
            key=lambda item: (item[1][0], item[0]),
        ):
            market, snapshot = market_by_id[condition_id]
            matches = [
                item
                for item in intervals
                if item.slot_id == selected_slot
                and item.start_ms <= event_start_ms < item.end_ms
            ]
            if (
                len(matches) != 1
                or matches[0].role != selected_role
                or market.event_start_ms != event_start_ms
                or opened_at_ms <= market.end_ms
            ):
                raise ValueError("Round 27 target role assignment differs")
            condition_payloads.append(
                _market_payload(
                    market,
                    source_snapshot_sha256=snapshot,
                    feature_row_chain_sha256=row_chain,
                )
            )
        population_chain = _EMPTY_SHA256
        for item in condition_payloads:
            population_chain = _hash_chain(
                population_chain,
                _canonical_sha256(item),
            )
        claim_body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND27_TARGET_ACCESS_SCHEMA_VERSION,
            "role": selected_role,
            "slot_id": selected_slot,
            "run_id": selected_run,
            "contract_sha256": contract_sha256,
            "feature_store_audit_sha256": feature_audit_sha256,
            "opened_at_ms": opened_at_ms,
            "condition_count": len(condition_payloads),
            "condition_population_sha256": population_chain,
            "selection_claim_sha256": selection_sha256,
            "selection_economic_claim_sha256": economic_claim_sha256,
            "selection_economic_report_sha256": economic_report_sha256,
            "target_access_opened": True,
            "edge_claim": False,
            "profitability_claim": False,
            "trading_authority": False,
            "orders_submitted": False,
        }
        claim_sha256 = _canonical_sha256(claim_body)
        existing = self.connection.execute(
            "SELECT claim_sha256 FROM round27_target_access WHERE role=?",
            [selected_role],
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != claim_sha256:
                raise ValueError("Round 27 target role was opened differently")
            self.audit()
            return False
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for item in condition_payloads:
                self.connection.execute(
                    "INSERT INTO round27_target_condition VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        item["condition_id"],
                        selected_role,
                        selected_slot,
                        item["event_start_ms"],
                        _canonical_json(item),
                        _canonical_sha256(item),
                    ],
                )
            self.connection.execute(
                """
                INSERT INTO round27_target_access VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, false, NULL
                )
                """,
                [
                    selected_role,
                    selected_slot,
                    selected_run,
                    contract_sha256,
                    feature_audit_sha256,
                    opened_at_ms,
                    len(condition_payloads),
                    population_chain,
                    selection_sha256,
                    economic_claim_sha256,
                    economic_report_sha256,
                    _canonical_json({**claim_body, "claim_sha256": claim_sha256}),
                    claim_sha256,
                ],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return True

    def collect_once(
        self,
        *,
        role: str,
        client: _ResolutionClient,
    ) -> dict[str, object]:
        if self.read_only:
            raise ValueError("Round 27 target store is read-only")
        selected_role = str(role or "").lower()
        access = self.connection.execute(
            """
            SELECT opened_at_ms, claim_sha256, finalized
            FROM round27_target_access WHERE role=?
            """,
            [selected_role],
        ).fetchone()
        if access is None or access[2] is True:
            raise ValueError("Round 27 target role is not open for collection")
        rows = self.connection.execute(
            """
            SELECT c.condition_id, c.market_json
            FROM round27_target_condition c
            LEFT JOIN round27_target_evidence e USING (condition_id)
            WHERE c.role=? AND e.condition_id IS NULL
            ORDER BY c.event_start_ms, c.condition_id
            """,
            [selected_role],
        ).fetchall()
        pending: list[str] = []
        newly_resolved = 0
        for condition_id, market_json in rows:
            market_payload = _strict_json(market_json, name="market")
            market = _market_from_payload(market_payload)
            clob = _public_payload(
                client.clob_market(str(condition_id)),
                name="CLOB target",
            )
            gamma = _public_payload(
                client.gamma_market(market.market_id),
                name="Gamma target",
            )
            observed_wall_ms = max(clob.observed_wall_ms, gamma.observed_wall_ms)
            observed_monotonic_ns = max(
                clob.observed_monotonic_ns,
                gamma.observed_monotonic_ns,
            )
            if min(clob.observed_wall_ms, gamma.observed_wall_ms) < int(access[0]):
                raise ValueError("Round 27 target receipt predates target access")
            winner = validate_official_resolution(
                market,
                clob.value,
                gamma.value,
                observed_wall_ms=observed_wall_ms,
            )
            if winner is None:
                pending.append(str(condition_id))
                continue
            evidence = {
                "schema_version": POLYMARKET_ROUND27_TARGET_EVIDENCE_SCHEMA_VERSION,
                "access_claim_sha256": str(access[1]),
                "condition_id": str(condition_id),
                "observed_wall_ms": observed_wall_ms,
                "observed_monotonic_ns": observed_monotonic_ns,
                "winning_asset_id": winner[0],
                "winning_outcome": winner[1],
                "clob_payload_sha256": clob.sha256,
                "gamma_payload_sha256": gamma.sha256,
            }
            evidence_sha256 = _canonical_sha256(evidence)
            self.connection.execute("BEGIN TRANSACTION")
            try:
                self.connection.execute(
                    """
                    INSERT INTO round27_target_evidence VALUES (
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
                        _compress(clob.canonical_json),
                        _compress(gamma.canonical_json),
                        _canonical_json(evidence),
                        evidence_sha256,
                    ],
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            newly_resolved += 1
        resolved = int(
            self.connection.execute(
                """
                SELECT count(*) FROM round27_target_evidence e
                JOIN round27_target_condition c USING (condition_id)
                WHERE c.role=?
                """,
                [selected_role],
            ).fetchone()[0]
        )
        return {
            "role": selected_role,
            "newly_resolved_condition_count": newly_resolved,
            "resolved_condition_count": resolved,
            "pending_condition_count": len(pending),
            "pending_condition_ids": pending,
            "orders_submitted": False,
            "trading_authority": False,
        }

    def finalize_role(self, role: str) -> dict[str, object]:
        if self.read_only:
            raise ValueError("Round 27 target store is read-only")
        selected_role = str(role or "").lower()
        row = self.connection.execute(
            """
            SELECT condition_count, finalized, evidence_chain_sha256
            FROM round27_target_access WHERE role=?
            """,
            [selected_role],
        ).fetchone()
        if row is None:
            raise ValueError("Round 27 target role is unavailable")
        evidence_rows = self.connection.execute(
            """
            SELECT e.evidence_sha256 FROM round27_target_evidence e
            JOIN round27_target_condition c USING (condition_id)
            WHERE c.role=? ORDER BY c.event_start_ms, c.condition_id
            """,
            [selected_role],
        ).fetchall()
        chain = _EMPTY_SHA256
        for evidence in evidence_rows:
            chain = _hash_chain(chain, _sha256(evidence[0], name="evidence"))
        if len(evidence_rows) != int(row[0]):
            raise ValueError("Round 27 target role remains unresolved")
        if row[1] is True:
            if row[2] != chain:
                raise ValueError("Round 27 finalized target role differs")
        else:
            self.connection.execute(
                """
                UPDATE round27_target_access
                SET finalized=true, evidence_chain_sha256=? WHERE role=?
                """,
                [chain, selected_role],
            )
            self.connection.execute("CHECKPOINT")
        report = self.audit()
        return next(
            item for item in report["roles"] if item["role"] == selected_role
        )

    def outcomes_up(self, *, roles: Sequence[str]) -> dict[str, int]:
        selected = tuple(str(role or "").lower() for role in roles)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Round 27 target roles differ")
        access = self.connection.execute(
            """
            SELECT role, finalized FROM round27_target_access
            WHERE role IN (SELECT unnest(?::VARCHAR[])) ORDER BY role
            """,
            [list(selected)],
        ).fetchall()
        if len(access) != len(selected) or any(row[1] is not True for row in access):
            raise ValueError("Round 27 target role is not finalized")
        rows = self.connection.execute(
            """
            SELECT e.condition_id, e.winning_outcome
            FROM round27_target_evidence e
            JOIN round27_target_condition c USING (condition_id)
            WHERE c.role IN (SELECT unnest(?::VARCHAR[]))
            ORDER BY c.event_start_ms, c.condition_id
            """,
            [list(selected)],
        ).fetchall()
        return {str(row[0]): 1 if row[1] == "Up" else 0 for row in rows}

    def audit(self) -> dict[str, object]:
        accesses = self.connection.execute(
            """
            SELECT role, slot_id, run_id, contract_sha256,
                   feature_store_audit_sha256, opened_at_ms, condition_count,
                   condition_population_sha256, selection_claim_sha256,
                   selection_economic_claim_sha256,
                   selection_economic_report_sha256, claim_json, claim_sha256,
                   finalized, evidence_chain_sha256
            FROM round27_target_access ORDER BY role
            """
        ).fetchall()
        role_reports: list[dict[str, object]] = []
        total_conditions = 0
        total_resolved = 0
        for access in accesses:
            role = str(access[0])
            claim = _strict_json(access[11], name="access claim")
            claimed = _sha256(claim.pop("claim_sha256", ""), name="access claim")
            if (
                claimed != access[12]
                or claimed != _canonical_sha256(claim)
                or claim.get("schema_version")
                != POLYMARKET_ROUND27_TARGET_ACCESS_SCHEMA_VERSION
                or claim.get("role") != access[0]
                or claim.get("slot_id") != access[1]
                or claim.get("run_id") != access[2]
                or claim.get("contract_sha256") != access[3]
                or claim.get("feature_store_audit_sha256") != access[4]
                or claim.get("opened_at_ms") != int(access[5])
                or claim.get("condition_count") != int(access[6])
                or claim.get("selection_claim_sha256") != access[8]
                or claim.get("selection_economic_claim_sha256") != access[9]
                or claim.get("selection_economic_report_sha256") != access[10]
                or claim.get("target_access_opened") is not True
                or any(
                    claim.get(field) is not False
                    for field in (
                        "edge_claim",
                        "profitability_claim",
                        "trading_authority",
                        "orders_submitted",
                    )
                )
            ):
                raise ValueError("Round 27 target access claim differs")
            conditions = self.connection.execute(
                """
                SELECT condition_id, event_start_ms, market_json, market_sha256
                FROM round27_target_condition
                WHERE role=? ORDER BY event_start_ms, condition_id
                """,
                [role],
            ).fetchall()
            population_chain = _EMPTY_SHA256
            evidence_chain = _EMPTY_SHA256
            resolved = 0
            for condition_id, event_start, market_json, market_sha in conditions:
                market_payload = _strict_json(market_json, name="market")
                market = _market_from_payload(market_payload)
                if (
                    market.condition_id != condition_id
                    or market.event_start_ms != int(event_start)
                    or _canonical_sha256(market_payload) != market_sha
                ):
                    raise ValueError("Round 27 target condition differs")
                population_chain = _hash_chain(population_chain, str(market_sha))
                evidence = self.connection.execute(
                    """
                    SELECT observed_wall_ms, observed_monotonic_ns,
                           winning_asset_id, winning_outcome,
                           clob_payload_sha256, gamma_payload_sha256,
                           clob_payload, gamma_payload, evidence_json,
                           evidence_sha256
                    FROM round27_target_evidence WHERE condition_id=?
                    """,
                    [condition_id],
                ).fetchone()
                if evidence is None:
                    continue
                clob = _decompress(
                    evidence[6],
                    expected_sha256=str(evidence[4]),
                    name="CLOB target",
                )
                gamma = _decompress(
                    evidence[7],
                    expected_sha256=str(evidence[5]),
                    name="Gamma target",
                )
                winner = validate_official_resolution(
                    market,
                    clob,
                    gamma,
                    observed_wall_ms=int(evidence[0]),
                )
                evidence_payload = _strict_json(evidence[8], name="evidence")
                expected_evidence = {
                    "schema_version": (
                        POLYMARKET_ROUND27_TARGET_EVIDENCE_SCHEMA_VERSION
                    ),
                    "access_claim_sha256": str(access[12]),
                    "condition_id": str(condition_id),
                    "observed_wall_ms": int(evidence[0]),
                    "observed_monotonic_ns": int(evidence[1]),
                    "winning_asset_id": str(evidence[2]),
                    "winning_outcome": str(evidence[3]),
                    "clob_payload_sha256": str(evidence[4]),
                    "gamma_payload_sha256": str(evidence[5]),
                }
                if (
                    winner != (str(evidence[2]), str(evidence[3]))
                    or evidence_payload != expected_evidence
                    or _canonical_sha256(evidence_payload) != evidence[9]
                    or int(evidence[0]) < int(access[5])
                    or not math.isfinite(float(evidence[0]))
                ):
                    raise ValueError("Round 27 target evidence differs")
                evidence_chain = _hash_chain(evidence_chain, str(evidence[9]))
                resolved += 1
            if (
                len(conditions) != int(access[6])
                or population_chain != access[7]
                or claim.get("condition_population_sha256") != access[7]
                or (access[13] is True and resolved != len(conditions))
                or (access[13] is False and access[14] is not None)
                or (access[13] is True and evidence_chain != access[14])
            ):
                raise ValueError("Round 27 target role manifest differs")
            role_reports.append(
                {
                    "role": role,
                    "slot_id": access[1],
                    "condition_count": len(conditions),
                    "resolved_condition_count": resolved,
                    "finalized": bool(access[13]),
                    "claim_sha256": access[12],
                    "condition_population_sha256": population_chain,
                    "evidence_chain_sha256": (
                        evidence_chain if access[13] is True else None
                    ),
                }
            )
            total_conditions += len(conditions)
            total_resolved += resolved
        body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND27_TARGET_STORE_SCHEMA_VERSION,
            "role_count": len(role_reports),
            "condition_count": total_conditions,
            "resolved_condition_count": total_resolved,
            "roles": role_reports,
            "feature_targets_co_located": False,
            "edge_claim": False,
            "profitability_claim": False,
            "trading_authority": False,
            "orders_submitted": False,
        }
        body["audit_sha256"] = _canonical_sha256(body)
        return body


__all__ = [
    "POLYMARKET_ROUND27_TARGET_ACCESS_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_TARGET_EVIDENCE_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_TARGET_STORE_SCHEMA_VERSION",
    "Round27TargetStore",
]
