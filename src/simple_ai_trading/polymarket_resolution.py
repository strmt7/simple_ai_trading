"""Immutable dual-source resolution evidence for Polymarket paper trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import time
from typing import Callable, Mapping, Sequence

from .polymarket import (
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    parse_polymarket_five_minute_market,
)
from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_RESOLUTION_SCHEMA_VERSION = "polymarket-official-resolution-v1"
_ROUND13_CLAIM_SCHEMA_VERSION = "polymarket-round13-one-use-claim-v1"


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(raw: object, *, name: str) -> object:
    try:
        return json.loads(
            str(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc


def _round13_capture_contract(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> str | None:
    from .polymarket_round13_capture import (
        POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION,
        validate_round13_capture_manifest_payload,
    )

    row = (
        store.connect()
        .execute(
            """
        SELECT schema_version, manifest_json, manifest_sha256
        FROM polymarket_preregistration_manifest WHERE run_id = ?
        """,
            [run_id],
        )
        .fetchone()
    )
    if row is None or str(row[0]) != POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION:
        return None
    manifest = _strict_json(row[1], name="Round 13 capture manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Round 13 capture manifest is invalid")
    validated = validate_round13_capture_manifest_payload(
        manifest,
        expected_run_id=run_id,
    )
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    contract = str(validated.get("contract_sha256") or "").strip().lower()
    if (
        body.get("schema_version") != POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION
        or body.get("run_id") != run_id
        or not _is_sha256(contract)
        or claimed != str(row[2])
        or claimed != _canonical_sha256(body)
        or _canonical_json(manifest) != str(row[1])
    ):
        raise ValueError("Round 13 capture manifest does not revalidate")
    return contract


def _require_round13_resolution_authority(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    supplied_contract_sha256: str,
) -> None:
    capture_contract = _round13_capture_contract(store, run_id)
    supplied = str(supplied_contract_sha256 or "").strip().lower()
    if capture_contract is None:
        if supplied:
            raise ValueError("Round 13 resolution authority targets a non-Round 13 run")
        return
    if supplied != capture_contract:
        raise ValueError(
            "Round 13 resolution requires its committed one-use evaluation claim"
        )
    table_exists = bool(
        store.connect()
        .execute(
            """
            SELECT count(*) FROM duckdb_tables()
            WHERE table_name = 'polymarket_round13_evaluation_claim'
            """
        )
        .fetchone()[0]
    )
    if not table_exists:
        raise ValueError(
            "Round 13 resolution requires its committed one-use evaluation claim"
        )
    row = (
        store.connect()
        .execute(
            """
        SELECT schema_version, claim_sha256, pipeline_report_sha256,
               scenario_dataset_sha256_json, opened_at_ms, status,
               report_sha256, error
        FROM polymarket_round13_evaluation_claim
        WHERE contract_sha256 = ? AND run_id = ?
        """,
            [capture_contract, run_id],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError(
            "Round 13 resolution requires its committed one-use evaluation claim"
        )
    scenarios = _strict_json(row[3], name="Round 13 claim scenario identity")
    identity = {
        "schema_version": _ROUND13_CLAIM_SCHEMA_VERSION,
        "contract_sha256": capture_contract,
        "run_id": run_id,
        "pipeline_report_sha256": str(row[2]),
        "scenario_dataset_sha256": scenarios,
        "opened_at_ms": int(row[4]),
        "state": "opened_before_resolution_query",
        "preexisting_resolution_count": 0,
    }
    maximum_end_ms = int(
        store.connect()
        .execute(
            "SELECT max(end_ms) FROM polymarket_market_snapshot WHERE run_id = ?",
            [run_id],
        )
        .fetchone()[0]
    )
    if (
        str(row[0]) != _ROUND13_CLAIM_SCHEMA_VERSION
        or not _is_sha256(row[1])
        or not _is_sha256(row[2])
        or not isinstance(scenarios, list)
        or not scenarios
        or not all(_is_sha256(value) for value in scenarios)
        or int(row[4]) < maximum_end_ms
        or str(row[5]) != "opened"
        or str(row[6] or "")
        or str(row[7] or "")
        or _canonical_sha256(identity) != str(row[1])
    ):
        raise ValueError("Round 13 evaluation claim does not authorize resolution")


def _canonical_mapping(value: Mapping[str, object], *, name: str) -> tuple[str, str]:
    try:
        payload = _canonical_json(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not canonical JSON") from exc
    return payload, hashlib.sha256(payload.encode("ascii")).hexdigest()


def _json_array(value: object, *, name: str) -> list[object]:
    parsed = value
    if isinstance(value, str):
        parsed = _strict_json(value, name=name)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be an array")
    return parsed


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


@dataclass(frozen=True)
class PolymarketOfficialResolutionEvidence:
    resolution_id: str
    run_id: str
    schema_version: str
    condition_id: str
    market_id: str
    asset: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    winning_asset_id: str
    winning_outcome: str
    clob_payload_sha256: str
    gamma_payload_sha256: str
    evidence_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolymarketResolutionFinalizationReport:
    run_id: str
    status: str
    market_count: int
    finalized_count: int
    newly_finalized_count: int
    pending_condition_ids: tuple[str, ...]
    resolution_ids: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _validate_market_identity(
    market: PolymarketFiveMinuteMarket,
    clob_payload: Mapping[str, object],
    gamma_payload: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[Decimal]]:
    if (
        str(clob_payload.get("condition_id") or "").strip().lower()
        != market.condition_id
    ):
        raise ValueError("CLOB resolution condition does not match recorded evidence")
    if str(clob_payload.get("market_slug") or "").strip() != market.slug:
        raise ValueError("CLOB resolution slug does not match recorded evidence")
    raw_tokens = clob_payload.get("tokens")
    if not isinstance(raw_tokens, list) or len(raw_tokens) != 2:
        raise ValueError("CLOB resolution must contain exactly two tokens")
    tokens: list[Mapping[str, object]] = []
    for item in raw_tokens:
        if not isinstance(item, Mapping):
            raise ValueError("CLOB resolution token is malformed")
        tokens.append(item)
    clob_identity = [
        (str(item.get("token_id") or ""), str(item.get("outcome") or ""))
        for item in tokens
    ]
    if clob_identity != [
        (market.up_token_id, "Up"),
        (market.down_token_id, "Down"),
    ]:
        raise ValueError("CLOB resolution token mapping drifted")

    if str(gamma_payload.get("id") or "").strip() != market.market_id:
        raise ValueError("Gamma resolution market ID does not match recorded evidence")
    if (
        str(gamma_payload.get("conditionId") or "").strip().lower()
        != market.condition_id
    ):
        raise ValueError("Gamma resolution condition does not match recorded evidence")
    if str(gamma_payload.get("slug") or "").strip() != market.slug:
        raise ValueError("Gamma resolution slug does not match recorded evidence")
    gamma_outcomes = [
        str(item)
        for item in _json_array(gamma_payload.get("outcomes"), name="Gamma outcomes")
    ]
    gamma_tokens = [
        str(item)
        for item in _json_array(
            gamma_payload.get("clobTokenIds"), name="Gamma token IDs"
        )
    ]
    if gamma_outcomes != ["Up", "Down"] or gamma_tokens != list(market.token_ids):
        raise ValueError("Gamma resolution token mapping drifted")
    resolution_source = str(gamma_payload.get("resolutionSource") or "").strip().lower()
    if resolution_source.rstrip("/") != market.resolution_source.rstrip("/"):
        raise ValueError("Gamma resolution source drifted")
    prices = [
        _decimal(item, name="Gamma outcome price")
        for item in _json_array(
            gamma_payload.get("outcomePrices"), name="Gamma outcome prices"
        )
    ]
    if len(prices) != 2:
        raise ValueError("Gamma resolution must contain exactly two prices")
    return tokens, prices


def validate_official_resolution(
    market: PolymarketFiveMinuteMarket,
    clob_payload: Mapping[str, object],
    gamma_payload: Mapping[str, object],
    *,
    observed_wall_ms: int,
) -> tuple[str, str] | None:
    """Return the winner only when CLOB and Gamma are jointly terminal."""

    tokens, gamma_prices = _validate_market_identity(
        market, clob_payload, gamma_payload
    )
    clob_closed = clob_payload.get("closed")
    gamma_closed = gamma_payload.get("closed")
    if not isinstance(clob_closed, bool) or not isinstance(gamma_closed, bool):
        raise ValueError("official resolution closed flags are malformed")
    if not (clob_closed and gamma_closed):
        return None
    observed = int(observed_wall_ms)
    if observed < market.end_ms:
        raise ValueError("official resolution was observed before the market ended")
    if clob_payload.get("accepting_orders") is True:
        raise ValueError("closed CLOB market still accepts orders")
    if gamma_payload.get("acceptingOrders") is True:
        raise ValueError("closed Gamma market still accepts orders")

    winners = [item for item in tokens if item.get("winner") is True]
    if len(winners) != 1 or any(
        not isinstance(item.get("winner"), bool) for item in tokens
    ):
        raise ValueError("closed CLOB market must identify exactly one winner")
    clob_prices = [
        _decimal(item.get("price"), name="CLOB terminal token price") for item in tokens
    ]
    if sorted(clob_prices) != [Decimal("0"), Decimal("1")]:
        raise ValueError("closed CLOB market does not have terminal token prices")
    if sorted(gamma_prices) != [Decimal("0"), Decimal("1")]:
        raise ValueError("closed Gamma market does not have terminal outcome prices")

    winner = winners[0]
    winning_asset_id = str(winner.get("token_id") or "")
    winning_outcome = str(winner.get("outcome") or "")
    winner_index = (
        0 if winning_outcome == "Up" else 1 if winning_outcome == "Down" else -1
    )
    if winner_index < 0 or gamma_prices[winner_index] != Decimal("1"):
        raise ValueError("CLOB and Gamma disagree on the winning outcome")
    if clob_prices[winner_index] != Decimal("1"):
        raise ValueError("CLOB winner flag disagrees with its terminal price")
    return winning_asset_id, winning_outcome


def _evidence_payload(
    *,
    resolution_id: str,
    run_id: str,
    market: PolymarketFiveMinuteMarket,
    observed_wall_ms: int,
    observed_monotonic_ns: int,
    winning_asset_id: str,
    winning_outcome: str,
    clob_payload_sha256: str,
    gamma_payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_RESOLUTION_SCHEMA_VERSION,
        "resolution_id": resolution_id,
        "run_id": run_id,
        "condition_id": market.condition_id,
        "market_id": market.market_id,
        "asset": market.asset,
        "observed_wall_ms": int(observed_wall_ms),
        "observed_monotonic_ns": int(observed_monotonic_ns),
        "winning_asset_id": winning_asset_id,
        "winning_outcome": winning_outcome,
        "clob_payload_sha256": clob_payload_sha256,
        "gamma_payload_sha256": gamma_payload_sha256,
    }


def _load_markets(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> tuple[PolymarketFiveMinuteMarket, ...]:
    rows = (
        store.connect()
        .execute(
            """
        SELECT condition_id, gamma_payload_json
        FROM polymarket_market_snapshot
        WHERE run_id = ? ORDER BY event_start_ms, asset
        """,
            [run_id],
        )
        .fetchall()
    )
    markets: list[PolymarketFiveMinuteMarket] = []
    for condition_id, payload_json in rows:
        payload = _strict_json(payload_json, name="stored Gamma market evidence")
        if not isinstance(payload, Mapping):
            raise ValueError("stored Gamma market evidence must be an object")
        market = parse_polymarket_five_minute_market(payload)
        if market.condition_id != str(condition_id):
            raise ValueError("stored Gamma market identity drifted")
        markets.append(market)
    if not markets:
        raise ValueError("Polymarket resolution run has no market snapshots")
    return tuple(markets)


def _resolution_from_row(
    row: Sequence[object],
    market: PolymarketFiveMinuteMarket,
) -> PolymarketOfficialResolutionEvidence:
    (
        resolution_id,
        run_id,
        schema_version,
        condition_id,
        market_id,
        asset,
        observed_wall_ms,
        observed_monotonic_ns,
        winning_asset_id,
        winning_outcome,
        clob_json,
        clob_sha,
        gamma_json,
        gamma_sha,
        payload_json,
        evidence_sha,
    ) = row
    clob_payload = _strict_json(clob_json, name="stored CLOB resolution")
    gamma_payload = _strict_json(gamma_json, name="stored Gamma resolution")
    stored_payload = _strict_json(payload_json, name="stored resolution evidence")
    if not all(
        isinstance(item, Mapping)
        for item in (clob_payload, gamma_payload, stored_payload)
    ):
        raise ValueError("stored official resolution JSON must contain objects")
    canonical_clob, actual_clob_sha = _canonical_mapping(
        clob_payload, name="stored CLOB resolution"
    )
    canonical_gamma, actual_gamma_sha = _canonical_mapping(
        gamma_payload, name="stored Gamma resolution"
    )
    if canonical_clob != str(clob_json) or canonical_gamma != str(gamma_json):
        raise ValueError("stored official resolution source JSON is not canonical")
    if not hmac.compare_digest(actual_clob_sha, str(clob_sha)):
        raise ValueError("stored CLOB resolution hash is invalid")
    if not hmac.compare_digest(actual_gamma_sha, str(gamma_sha)):
        raise ValueError("stored Gamma resolution hash is invalid")
    if str(schema_version) != POLYMARKET_RESOLUTION_SCHEMA_VERSION:
        raise ValueError("stored official resolution schema is unsupported")
    if str(condition_id) != market.condition_id or str(market_id) != market.market_id:
        raise ValueError("stored official resolution market identity drifted")
    winner = validate_official_resolution(
        market,
        clob_payload,
        gamma_payload,
        observed_wall_ms=int(observed_wall_ms),
    )
    if winner != (str(winning_asset_id), str(winning_outcome)):
        raise ValueError("stored official resolution winner drifted")
    identity = {
        "schema_version": POLYMARKET_RESOLUTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "condition_id": market.condition_id,
        "observed_wall_ms": int(observed_wall_ms),
        "observed_monotonic_ns": int(observed_monotonic_ns),
        "clob_payload_sha256": str(clob_sha),
        "gamma_payload_sha256": str(gamma_sha),
    }
    if not hmac.compare_digest(_canonical_sha256(identity), str(resolution_id)):
        raise ValueError("stored official resolution ID is invalid")
    expected_payload = _evidence_payload(
        resolution_id=str(resolution_id),
        run_id=str(run_id),
        market=market,
        observed_wall_ms=int(observed_wall_ms),
        observed_monotonic_ns=int(observed_monotonic_ns),
        winning_asset_id=str(winning_asset_id),
        winning_outcome=str(winning_outcome),
        clob_payload_sha256=str(clob_sha),
        gamma_payload_sha256=str(gamma_sha),
    )
    if (
        _canonical_json(stored_payload) != str(payload_json)
        or stored_payload != expected_payload
    ):
        raise ValueError("stored official resolution payload is inconsistent")
    if not hmac.compare_digest(_canonical_sha256(expected_payload), str(evidence_sha)):
        raise ValueError("stored official resolution evidence hash is invalid")
    if str(asset) != market.asset:
        raise ValueError("stored official resolution asset drifted")
    return PolymarketOfficialResolutionEvidence(
        resolution_id=str(resolution_id),
        run_id=str(run_id),
        schema_version=str(schema_version),
        condition_id=market.condition_id,
        market_id=market.market_id,
        asset=market.asset,
        observed_wall_ms=int(observed_wall_ms),
        observed_monotonic_ns=int(observed_monotonic_ns),
        winning_asset_id=str(winning_asset_id),
        winning_outcome=str(winning_outcome),
        clob_payload_sha256=str(clob_sha),
        gamma_payload_sha256=str(gamma_sha),
        evidence_sha256=str(evidence_sha),
    )


def load_official_resolutions(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
) -> tuple[PolymarketOfficialResolutionEvidence, ...]:
    markets = {market.condition_id: market for market in _load_markets(store, run_id)}
    rows = (
        store.connect()
        .execute(
            """
        SELECT resolution_id, run_id, schema_version, condition_id, market_id,
               asset, observed_wall_ms, observed_monotonic_ns, winning_asset_id,
               winning_outcome, clob_payload_json, clob_payload_sha256,
               gamma_payload_json, gamma_payload_sha256, evidence_payload_json,
               evidence_sha256
        FROM polymarket_resolution_evidence
        WHERE run_id = ? ORDER BY condition_id
        """,
            [run_id],
        )
        .fetchall()
    )
    output: list[PolymarketOfficialResolutionEvidence] = []
    for row in rows:
        market = markets.get(str(row[3]))
        if market is None:
            raise ValueError("official resolution references an unknown market")
        output.append(_resolution_from_row(row, market))
    return tuple(output)


class PolymarketResolutionFinalizer:
    """Fetch and persist only jointly terminal public resolution states."""

    def __init__(
        self,
        store: PolymarketEvidenceStore,
        *,
        client: PolymarketPublicClient | None = None,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self.store = store
        self.client = client or PolymarketPublicClient()
        self.wall_clock_ms = wall_clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic_clock_ns = monotonic_clock_ns or time.monotonic_ns

    def finalize(
        self,
        *,
        run_id: str,
        integrity_prevalidated: bool = False,
        round13_contract_sha256: str = "",
    ) -> PolymarketResolutionFinalizationReport:
        selected = str(run_id or "").strip()
        if not selected:
            raise ValueError("run_id is required for Polymarket resolution")
        run = (
            self.store.connect()
            .execute(
                "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
                [selected],
            )
            .fetchone()
        )
        if run is None:
            raise ValueError(f"unknown Polymarket recorder run: {selected}")
        if str(run[0]) not in {"complete", "degraded"}:
            raise ValueError("Polymarket resolution requires a finished valid run")
        _require_round13_resolution_authority(
            self.store,
            run_id=selected,
            supplied_contract_sha256=round13_contract_sha256,
        )
        if not integrity_prevalidated:
            integrity = self.store.resume_integrity_errors(selected)
            if integrity:
                raise ValueError(
                    "Polymarket resolution evidence failed integrity: "
                    + "; ".join(integrity)
                )
        markets = _load_markets(self.store, selected)
        existing = {
            item.condition_id: item
            for item in load_official_resolutions(self.store, run_id=selected)
        }
        newly_finalized = 0
        pending: list[str] = []
        for market in markets:
            if market.condition_id in existing:
                continue
            if int(self.wall_clock_ms()) < market.end_ms:
                pending.append(market.condition_id)
                continue
            clob_payload = self.client.clob_market(market.condition_id)
            gamma_payload = self.client.gamma_market(market.market_id)
            observed_wall_ms = int(self.wall_clock_ms())
            observed_monotonic_ns = int(self.monotonic_clock_ns())
            winner = validate_official_resolution(
                market,
                clob_payload,
                gamma_payload,
                observed_wall_ms=observed_wall_ms,
            )
            if winner is None:
                pending.append(market.condition_id)
                continue
            clob_json, clob_sha = _canonical_mapping(
                clob_payload, name="CLOB resolution"
            )
            gamma_json, gamma_sha = _canonical_mapping(
                gamma_payload, name="Gamma resolution"
            )
            identity = {
                "schema_version": POLYMARKET_RESOLUTION_SCHEMA_VERSION,
                "run_id": selected,
                "condition_id": market.condition_id,
                "observed_wall_ms": observed_wall_ms,
                "observed_monotonic_ns": observed_monotonic_ns,
                "clob_payload_sha256": clob_sha,
                "gamma_payload_sha256": gamma_sha,
            }
            resolution_id = _canonical_sha256(identity)
            evidence_payload = _evidence_payload(
                resolution_id=resolution_id,
                run_id=selected,
                market=market,
                observed_wall_ms=observed_wall_ms,
                observed_monotonic_ns=observed_monotonic_ns,
                winning_asset_id=winner[0],
                winning_outcome=winner[1],
                clob_payload_sha256=clob_sha,
                gamma_payload_sha256=gamma_sha,
            )
            self.store.connect().execute(
                """
                INSERT INTO polymarket_resolution_evidence VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    resolution_id,
                    selected,
                    POLYMARKET_RESOLUTION_SCHEMA_VERSION,
                    market.condition_id,
                    market.market_id,
                    market.asset,
                    observed_wall_ms,
                    observed_monotonic_ns,
                    winner[0],
                    winner[1],
                    clob_json,
                    clob_sha,
                    gamma_json,
                    gamma_sha,
                    _canonical_json(evidence_payload),
                    _canonical_sha256(evidence_payload),
                ],
            )
            newly_finalized += 1
        resolutions = load_official_resolutions(self.store, run_id=selected)
        finalized_conditions = {item.condition_id for item in resolutions}
        pending = sorted(
            set(pending)
            | {
                market.condition_id
                for market in markets
                if market.condition_id not in finalized_conditions
            }
        )
        status = "complete" if not pending else "pending"
        return PolymarketResolutionFinalizationReport(
            run_id=selected,
            status=status,
            market_count=len(markets),
            finalized_count=len(resolutions),
            newly_finalized_count=newly_finalized,
            pending_condition_ids=tuple(pending),
            resolution_ids=tuple(item.resolution_id for item in resolutions),
        )


__all__ = [
    "POLYMARKET_RESOLUTION_SCHEMA_VERSION",
    "PolymarketOfficialResolutionEvidence",
    "PolymarketResolutionFinalizationReport",
    "PolymarketResolutionFinalizer",
    "load_official_resolutions",
    "validate_official_resolution",
]
