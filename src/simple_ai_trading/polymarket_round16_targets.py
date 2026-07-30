"""One-way, phase-gated terminal labels for the Round 16 BTC screen."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

from .polymarket import PolymarketFifteenMinuteMarket
from .polymarket_historical_screen import (
    HistoricalBtcMarket,
    HistoricalScreenStore,
    ProgressCallback,
    PublicPayload,
)
from .polymarket_resolution import validate_official_resolution
from .polymarket_round16 import (
    ROUND16_DURATION_MS,
    ROUND16_RESOLUTION_SOURCE,
    Round16HistoricalContract,
    Round16HistoricalPublicClient,
)


ROUND16_RESOLUTION_SCHEMA_VERSION = "polymarket-round16-btc-15m-resolution-v1"
ROUND16_TARGET_IMPLEMENTATION_SCHEMA_VERSION = (
    "polymarket-round16-target-implementation-v1"
)
ROUND16_GAMMA_TARGET_BATCH_SIZE = 100
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def round16_target_implementation_manifest() -> Mapping[str, object]:
    source_root = Path(__file__).parent
    paths = {
        "round16_targets": Path(__file__),
        "round16_public_client": source_root / "polymarket_round16.py",
        "resolution_validator": source_root / "polymarket_resolution.py",
        "shared_public_client": source_root / "polymarket_historical_screen.py",
    }
    body: dict[str, object] = {
        "schema_version": ROUND16_TARGET_IMPLEMENTATION_SCHEMA_VERSION,
        "implementation_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _record_or_verify_target_implementation(
    store: HistoricalScreenStore,
    *,
    create: bool,
    expected_sha256: str = "",
) -> Mapping[str, object]:
    manifest = round16_target_implementation_manifest()
    manifest_sha = str(manifest["manifest_sha256"])
    expected = str(expected_sha256 or "").strip().lower()
    if expected and (_SHA256.fullmatch(expected) is None or expected != manifest_sha):
        raise ValueError("Round 16 pretest target implementation differs")
    connection = store.connect()
    if create:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS target.round16_resolution_manifest (
                singleton BOOLEAN PRIMARY KEY CHECK(singleton),
                manifest_json VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            )
            """
        )
    elif (
        connection.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = 'target'
              AND table_name = 'round16_resolution_manifest'
            """
        ).fetchone()[0]
        != 1
    ):
        raise ValueError("Round 16 target implementation manifest is missing")
    row = connection.execute(
        """
        SELECT manifest_json, manifest_sha256
        FROM target.round16_resolution_manifest
        WHERE singleton
        """
    ).fetchone()
    if row is None:
        if not create:
            raise ValueError("Round 16 target implementation manifest is missing")
        connection.execute(
            """
            INSERT INTO target.round16_resolution_manifest
            VALUES (true, ?, ?, ?)
            """,
            [
                _canonical_json(manifest),
                manifest_sha,
                time.time_ns() // 1_000_000,
            ],
        )
        return manifest
    try:
        stored = json.loads(str(row[0]))
    except json.JSONDecodeError as exc:
        raise ValueError("Round 16 target implementation manifest is not JSON") from exc
    if (
        not isinstance(stored, Mapping)
        or str(row[1]) != manifest_sha
        or dict(stored) != dict(manifest)
    ):
        raise ValueError("Round 16 target implementation changed after label access")
    return manifest


def verify_round16_target_implementation(
    store: HistoricalScreenStore,
    *,
    expected_sha256: str = "",
) -> Mapping[str, object]:
    return _record_or_verify_target_implementation(
        store,
        create=False,
        expected_sha256=expected_sha256,
    )


def record_round16_target_implementation(
    store: HistoricalScreenStore,
) -> Mapping[str, object]:
    return _record_or_verify_target_implementation(store, create=True)


def _official_market(
    market: HistoricalBtcMarket,
    gamma_payload: Mapping[str, object],
) -> PolymarketFifteenMinuteMarket:
    if market.end_ms - market.event_start_ms != ROUND16_DURATION_MS:
        raise ValueError("Round 16 resolution market duration differs")
    canonical = _canonical_json(dict(gamma_payload))
    return PolymarketFifteenMinuteMarket(
        asset="BTC",
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        question=market.question,
        event_start_ms=market.event_start_ms,
        end_ms=market.end_ms,
        up_token_id=market.up_token_id,
        down_token_id=market.down_token_id,
        tick_size=market.tick_size,
        minimum_order_size=market.minimum_order_size,
        fee_schedule=market.fee_schedule,
        liquidity_quote=Decimal("0"),
        volume_quote=Decimal("0"),
        resolution_source=ROUND16_RESOLUTION_SOURCE,
        gamma_payload_sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        gamma_payload_json=canonical,
    )


def _record_resolution(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    market: HistoricalBtcMarket,
    *,
    gamma: PublicPayload,
    clob: PublicPayload,
    implementation_sha256: str | None = None,
) -> str:
    allowed = (
        market.role in {"train", "tune"} and store.state == "features_complete"
    ) or (market.role == "test" and store.state == "pretest_complete")
    if store.contract != contract.historical or not allowed:
        raise ValueError("Round 16 target role or phase is not authorized")
    if not isinstance(gamma.value, Mapping) or not isinstance(clob.value, Mapping):
        raise ValueError("Round 16 terminal payload is malformed")
    implementation_sha = str(
        implementation_sha256
        or round16_target_implementation_manifest()["manifest_sha256"]
    )
    if _SHA256.fullmatch(implementation_sha) is None:
        raise ValueError("Round 16 target implementation identity is invalid")
    winner = validate_official_resolution(
        _official_market(market, gamma.value),
        clob.value,
        gamma.value,
        observed_wall_ms=max(gamma.observed_at_ms, clob.observed_at_ms),
    )
    if winner is None:
        raise ValueError("Round 16 official resolution is not terminal")
    evidence = {
        "schema_version": ROUND16_RESOLUTION_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "condition_id": market.condition_id,
        "role": market.role,
        "winning_token_id": winner[0],
        "winning_outcome": winner[1],
        "target_implementation_sha256": implementation_sha,
        "gamma_payload_sha256": gamma.sha256,
        "clob_payload_sha256": clob.sha256,
    }
    evidence_sha = _canonical_sha256(evidence)
    store.connect().execute(
        """
        INSERT INTO target.official_resolution VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT (condition_id) DO UPDATE SET
            role = excluded.role,
            winning_token_id = excluded.winning_token_id,
            winning_outcome = excluded.winning_outcome,
            gamma_payload_json = excluded.gamma_payload_json,
            gamma_payload_sha256 = excluded.gamma_payload_sha256,
            clob_payload_json = excluded.clob_payload_json,
            clob_payload_sha256 = excluded.clob_payload_sha256,
            evidence_sha256 = excluded.evidence_sha256,
            observed_at_ms = excluded.observed_at_ms
        """,
        [
            market.condition_id,
            market.role,
            winner[0],
            winner[1],
            gamma.canonical_json,
            gamma.sha256,
            clob.canonical_json,
            clob.sha256,
            evidence_sha,
            max(gamma.observed_at_ms, clob.observed_at_ms),
        ],
    )
    return winner[1]


def _collect_targets(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    client: Round16HistoricalPublicClient,
    *,
    roles: Sequence[str],
    expected_state: str,
    next_state: str,
    progress: ProgressCallback | None,
) -> Mapping[str, int]:
    selected_roles = tuple(str(role) for role in roles)
    if (
        store.contract != contract.historical
        or store.state != expected_state
        or not selected_roles
        or len(selected_roles) != len(set(selected_roles))
        or any(role not in {"train", "tune", "test"} for role in selected_roles)
    ):
        raise ValueError("Round 16 target collection phase differs")
    if "test" in selected_roles and selected_roles != ("test",):
        raise ValueError("Round 16 test labels require an isolated one-use phase")
    expected_implementation_sha = ""
    if selected_roles == ("test",):
        pretest, _pretest_envelope_sha = store.pretest_artifact()
        implementation = pretest.get("implementation_sha256")
        if not isinstance(implementation, Mapping):
            raise ValueError("Round 16 pretest implementation manifest is missing")
        expected_implementation_sha = str(
            implementation.get("round16_target_manifest") or ""
        )
    target_implementation = _record_or_verify_target_implementation(
        store,
        create=selected_roles != ("test",),
        expected_sha256=expected_implementation_sha,
    )
    target_implementation_sha = str(target_implementation["manifest_sha256"])
    markets = store.markets(roles=selected_roles)
    existing = store.resolved_conditions(roles=selected_roles)
    counts = {"Up": 0, "Down": 0}
    if existing:
        placeholders = ",".join("?" for _ in selected_roles)
        for winner, count in (
            store.connect()
            .execute(
                f"""
            SELECT winning_outcome, count(*)
            FROM target.official_resolution
            WHERE role IN ({placeholders})
            GROUP BY winning_outcome
            """,
                list(selected_roles),
            )
            .fetchall()
        ):
            if str(winner) not in counts:
                raise ValueError("Round 16 stored target outcome differs")
            counts[str(winner)] = int(count)
    pending = tuple(market for market in markets if market.condition_id not in existing)
    index_by_condition = {
        market.condition_id: index for index, market in enumerate(markets, start=1)
    }
    for first in range(0, len(pending), ROUND16_GAMMA_TARGET_BATCH_SIZE):
        batch = pending[first : first + ROUND16_GAMMA_TARGET_BATCH_SIZE]
        gamma_by_market = client.gamma_markets(
            tuple(market.market_id for market in batch)
        )
        if progress:
            progress(
                "round16_target_gamma_batch",
                {
                    "batch_index": first // ROUND16_GAMMA_TARGET_BATCH_SIZE + 1,
                    "batch_count": (len(pending) + ROUND16_GAMMA_TARGET_BATCH_SIZE - 1)
                    // ROUND16_GAMMA_TARGET_BATCH_SIZE,
                    "market_count": len(batch),
                },
            )
        for market in batch:
            gamma = gamma_by_market.get(market.market_id)
            if gamma is None:
                raise ValueError("Round 16 Gamma target batch coverage differs")
            clob = client.clob_market(market.condition_id)
            winner = _record_resolution(
                store,
                contract,
                market,
                gamma=gamma,
                clob=clob,
                implementation_sha256=target_implementation_sha,
            )
            counts[winner] += 1
            if progress:
                progress(
                    "round16_target_condition",
                    {
                        "role": market.role,
                        "condition_index": index_by_condition[market.condition_id],
                        "condition_count": len(markets),
                        "up_count": counts["Up"],
                        "down_count": counts["Down"],
                    },
                )
    resolved = store.resolved_conditions(roles=selected_roles)
    expected = {market.condition_id for market in markets}
    if resolved != expected or sum(counts.values()) != len(markets):
        raise ValueError("Round 16 target coverage is incomplete")
    store.transition(expected_state, next_state)
    return counts


def collect_round16_development_targets(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    client: Round16HistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    return _collect_targets(
        store,
        contract,
        client,
        roles=("train", "tune"),
        expected_state="features_complete",
        next_state="development_targets_complete",
        progress=progress,
    )


def collect_round16_test_targets_once(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    client: Round16HistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    return _collect_targets(
        store,
        contract,
        client,
        roles=("test",),
        expected_state="pretest_complete",
        next_state="targets_complete",
        progress=progress,
    )


__all__ = [
    "ROUND16_GAMMA_TARGET_BATCH_SIZE",
    "ROUND16_RESOLUTION_SCHEMA_VERSION",
    "ROUND16_TARGET_IMPLEMENTATION_SCHEMA_VERSION",
    "collect_round16_development_targets",
    "collect_round16_test_targets_once",
    "record_round16_target_implementation",
    "round16_target_implementation_manifest",
    "verify_round16_target_implementation",
]
