"""Label-free synchronized continuity eligibility for Polymarket Round 9."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Mapping

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket_action_value import POLYMARKET_ACTION_VALUE_CONTRACT_SHA256
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay


POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION = (
    "polymarket-round9-continuity-eligibility-v1"
)
POLYMARKET_ACTION_CONTRACT_COMMIT = "3dee7424ebb8740767f05c3ea9e64b10d1c59851"
POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS = 1_784_143_695_000
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


@dataclass(frozen=True)
class PolymarketContinuityConfig:
    chainlink_anchor_allowance_ms: int = 2_000
    feature_warmup_ms: int = 5_000
    minimum_remaining_market_time_ms: int = 30_000
    maximum_execution_confirmation_delay_ms: int = 500
    minimum_eligible_groups: int = 30

    def validated(self) -> PolymarketContinuityConfig:
        if asdict(self) != {
            "chainlink_anchor_allowance_ms": 2_000,
            "feature_warmup_ms": 5_000,
            "minimum_remaining_market_time_ms": 30_000,
            "maximum_execution_confirmation_delay_ms": 500,
            "minimum_eligible_groups": 30,
        }:
            raise ValueError("Polymarket continuity config drifted from Round 9")
        return self

    def asdict(self) -> dict[str, int]:
        self.validated()
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class PolymarketContinuityGroup:
    event_start_ms: int
    window_start_ms: int
    window_end_ms: int
    condition_ids: tuple[str, ...]
    eligible: bool
    reasons: tuple[str, ...]
    evidence: dict[str, object]
    group_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round9-continuity-group-v1",
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "event_start_ms": self.event_start_ms,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "condition_ids": list(self.condition_ids),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "group_sha256": self.group_sha256}

    def validated(self) -> PolymarketContinuityGroup:
        if (
            self.event_start_ms <= 0
            or self.window_start_ms >= self.event_start_ms
            or self.window_end_ms <= self.event_start_ms
            or len(self.condition_ids) != len(_ASSETS)
            or len(set(self.condition_ids)) != len(_ASSETS)
            or self.eligible == bool(self.reasons)
            or tuple(sorted(set(self.reasons))) != self.reasons
            or not _is_sha256(self.group_sha256)
            or self.group_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket continuity group is invalid")
        return self


@dataclass(frozen=True)
class PolymarketContinuityReport:
    run_id: str
    run_report_sha256: str
    run_started_at_ms: int
    config: PolymarketContinuityConfig
    groups: tuple[PolymarketContinuityGroup, ...]
    eligible_group_count: int
    confirmation_eligible: bool
    confirmation_reasons: tuple[str, ...]
    report_sha256: str

    @property
    def eligible_condition_ids(self) -> tuple[str, ...]:
        return tuple(
            condition
            for group in self.groups
            if group.eligible
            for condition in group.condition_ids
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "contract_commit": POLYMARKET_ACTION_CONTRACT_COMMIT,
            "contract_committed_at_ms": POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS,
            "run_id": self.run_id,
            "run_report_sha256": self.run_report_sha256,
            "run_started_at_ms": self.run_started_at_ms,
            "config": self.config.asdict(),
            "groups": [group.asdict() for group in self.groups],
            "eligible_group_count": self.eligible_group_count,
            "eligible_condition_ids": list(self.eligible_condition_ids),
            "confirmation_eligible": self.confirmation_eligible,
            "confirmation_reasons": list(self.confirmation_reasons),
            "outcomes_consulted": False,
            "labels_consulted": False,
            "model_scores_consulted": False,
            "training_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> PolymarketContinuityReport:
        for group in self.groups:
            group.validated()
        if (
            not self.run_id
            or not _is_sha256(self.run_report_sha256)
            or self.run_started_at_ms <= 0
            or not self.groups
            or self.eligible_group_count != sum(group.eligible for group in self.groups)
            or self.confirmation_eligible == bool(self.confirmation_reasons)
            or tuple(sorted(set(self.confirmation_reasons)))
            != self.confirmation_reasons
            or self.report_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket continuity report is invalid")
        return self


def _create_window_tables(
    store: PolymarketEvidenceStore,
    groups: Mapping[int, tuple[object, ...]],
    config: PolymarketContinuityConfig,
) -> tuple[dict[str, int], list[tuple[object, ...]], list[tuple[object, ...]]]:
    connection = store.connect()
    connection.execute(
        """
        DROP TABLE IF EXISTS continuity_market_window;
        DROP TABLE IF EXISTS continuity_token_window;
        DROP TABLE IF EXISTS continuity_asset_window;
        CREATE TEMP TABLE continuity_market_window (
            event_start_ms BIGINT, asset VARCHAR, condition_id VARCHAR,
            window_start_ms BIGINT, window_end_ms BIGINT,
            baseline_deadline_ms BIGINT
        );
        CREATE TEMP TABLE continuity_token_window (
            event_start_ms BIGINT, asset VARCHAR, condition_id VARCHAR,
            token_id VARCHAR, window_start_ms BIGINT, window_end_ms BIGINT,
            baseline_deadline_ms BIGINT
        );
        CREATE TEMP TABLE continuity_asset_window (
            event_start_ms BIGINT, asset VARCHAR,
            window_start_ms BIGINT, window_end_ms BIGINT
        );
        """
    )
    condition_start: dict[str, int] = {}
    market_rows: list[tuple[object, ...]] = []
    token_rows: list[tuple[object, ...]] = []
    asset_rows: list[tuple[object, ...]] = []
    for event_start_ms, markets in sorted(groups.items()):
        window_start = event_start_ms - config.chainlink_anchor_allowance_ms
        window_end = (
            markets[0].end_ms
            - config.minimum_remaining_market_time_ms
            + config.maximum_execution_confirmation_delay_ms
        )
        baseline_deadline = event_start_ms + config.feature_warmup_ms
        for market in markets:
            condition_start[market.condition_id] = event_start_ms
            market_rows.append(
                (
                    event_start_ms,
                    market.asset,
                    market.condition_id,
                    window_start,
                    window_end,
                    baseline_deadline,
                )
            )
            for token_id in market.token_ids:
                token_rows.append(
                    (
                        event_start_ms,
                        market.asset,
                        market.condition_id,
                        token_id,
                        window_start,
                        window_end,
                        baseline_deadline,
                    )
                )
            asset_rows.append((event_start_ms, market.asset, window_start, window_end))
    connection.executemany(
        "INSERT INTO continuity_market_window VALUES (?, ?, ?, ?, ?, ?)",
        market_rows,
    )
    connection.executemany(
        "INSERT INTO continuity_token_window VALUES (?, ?, ?, ?, ?, ?, ?)",
        token_rows,
    )
    connection.executemany(
        "INSERT INTO continuity_asset_window VALUES (?, ?, ?, ?)",
        asset_rows,
    )
    return condition_start, market_rows, token_rows


def _continuity_evidence(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> tuple[
    dict[tuple[int, str, str], tuple[object, ...]],
    dict[tuple[int, str], tuple[object, ...]],
    dict[tuple[int, str], tuple[object, ...]],
    dict[int, tuple[str, ...]],
    dict[str, int],
    dict[tuple[str, str], int],
]:
    connection = store.connect()
    token_windows = connection.execute(
        """
        SELECT event_start_ms, asset, condition_id, token_id,
               window_start_ms, window_end_ms, baseline_deadline_ms
        FROM continuity_token_window
        ORDER BY event_start_ms, asset, token_id
        """
    ).fetchall()
    asset_windows = connection.execute(
        """
        SELECT event_start_ms, asset, window_start_ms, window_end_ms
        FROM continuity_asset_window
        ORDER BY asset, window_start_ms, event_start_ms
        """
    ).fetchall()
    lane_summaries = store.raw_message_lane_summaries(
        run_id,
        streams=("binance_spot", "clob_market", "polymarket_rtds"),
    )
    connection_starts = {
        (summary.stream, summary.connection_id): summary.first_received_wall_ms
        for summary in lane_summaries
    }

    indexed_asset_windows: dict[str, list[tuple[int, int, int]]] = {}
    for event_start_ms, asset, window_start_ms, window_end_ms in asset_windows:
        indexed_asset_windows.setdefault(str(asset), []).append(
            (int(window_start_ms), int(window_end_ms), int(event_start_ms))
        )
    asset_window_starts: dict[str, tuple[int, ...]] = {}
    for asset, windows in indexed_asset_windows.items():
        previous_end: int | None = None
        for window_start, window_end, _event_start in windows:
            if window_start > window_end:
                raise ValueError("Polymarket continuity window is inverted")
            if previous_end is not None and window_start <= previous_end:
                raise ValueError("Polymarket continuity windows overlap")
            previous_end = window_end
        asset_window_starts[asset] = tuple(item[0] for item in windows)

    def event_window(asset: str, received_wall_ms: int) -> int | None:
        windows = indexed_asset_windows.get(asset)
        if not windows:
            return None
        position = bisect_right(asset_window_starts[asset], received_wall_ms) - 1
        if position < 0:
            return None
        window_start, window_end, event_start = windows[position]
        if window_start <= received_wall_ms <= window_end:
            return event_start
        return None

    token_window_by_identity = {
        (str(condition_id).lower(), str(token_id)): (
            int(event_start_ms),
            str(asset),
            int(window_start_ms),
            int(window_end_ms),
            int(baseline_deadline_ms),
        )
        for (
            event_start_ms,
            asset,
            condition_id,
            token_id,
            window_start_ms,
            window_end_ms,
            baseline_deadline_ms,
        ) in token_windows
    }
    clob_accumulators: dict[tuple[int, str, str], dict[str, object]] = {}
    binance_accumulators: dict[tuple[int, str], dict[str, object]] = {}
    rtds_accumulators: dict[tuple[int, str], dict[str, object]] = {}
    for decoded in store.iter_public_events(
        run_id,
        streams=("binance_spot", "clob_market", "polymarket_rtds"),
        verified_source=True,
    ):
        received_wall_ms = decoded.received_wall_ms
        event_type = decoded.event_type.lower()
        if decoded.stream == "clob_market":
            window = token_window_by_identity.get(
                (decoded.condition_id.lower(), decoded.asset_id)
            )
            if window is None or received_wall_ms > window[3]:
                continue
            key = (window[0], window[1], decoded.asset_id)
            accumulator = clob_accumulators.setdefault(
                key,
                {
                    "window_events": 0,
                    "window_connections": set(),
                    "baseline": None,
                },
            )
            if window[2] <= received_wall_ms <= window[3]:
                accumulator["window_events"] = int(accumulator["window_events"]) + 1
                connections = accumulator["window_connections"]
                assert isinstance(connections, set)
                connections.add(decoded.connection_id)
            if event_type == "book" and received_wall_ms <= window[4]:
                candidate = (
                    received_wall_ms,
                    decoded.received_monotonic_ns,
                    decoded.event_id,
                    decoded.connection_id,
                )
                baseline = accumulator["baseline"]
                if baseline is None or candidate > baseline:
                    accumulator["baseline"] = candidate
            continue
        asset = decoded.symbol.upper()
        event_start = event_window(asset, received_wall_ms)
        if event_start is None:
            continue
        key = (event_start, asset)
        if decoded.stream == "binance_spot":
            accumulator = binance_accumulators.setdefault(
                key,
                {"bookticker": 0, "trade": 0, "connections": set()},
            )
            if event_type in {"bookticker", "trade"}:
                accumulator[event_type] = int(accumulator[event_type]) + 1
            connections = accumulator["connections"]
            assert isinstance(connections, set)
            connections.add(decoded.connection_id)
        elif event_type.startswith("crypto_prices_chainlink:"):
            accumulator = rtds_accumulators.setdefault(
                key,
                {"events": 0, "connections": set()},
            )
            accumulator["events"] = int(accumulator["events"]) + 1
            connections = accumulator["connections"]
            assert isinstance(connections, set)
            connections.add(decoded.connection_id)

    clob: dict[tuple[int, str, str], tuple[object, ...]] = {}
    for event_start_ms, asset, _condition_id, token_id, *_bounds in token_windows:
        key = (int(event_start_ms), str(asset), str(token_id))
        accumulator = clob_accumulators.get(key)
        if accumulator is None:
            clob[key] = (0, 0, None, None, None)
            continue
        connections = accumulator["window_connections"]
        baseline = accumulator["baseline"]
        assert isinstance(connections, set)
        clob[key] = (
            int(accumulator["window_events"]),
            len(connections),
            min(connections) if connections else None,
            None if baseline is None else baseline[3],
            None if baseline is None else baseline[0],
        )
    binance: dict[tuple[int, str], tuple[object, ...]] = {}
    rtds: dict[tuple[int, str], tuple[object, ...]] = {}
    for event_start_ms, asset, _window_start, _window_end in asset_windows:
        key = (int(event_start_ms), str(asset))
        binance_accumulator = binance_accumulators.get(key)
        if binance_accumulator is None:
            binance[key] = (0, 0, 0, None)
        else:
            connections = binance_accumulator["connections"]
            assert isinstance(connections, set)
            binance[key] = (
                int(binance_accumulator["bookticker"]),
                int(binance_accumulator["trade"]),
                len(connections),
                min(connections) if connections else None,
            )
        rtds_accumulator = rtds_accumulators.get(key)
        if rtds_accumulator is None:
            rtds[key] = (0, 0, None)
        else:
            connections = rtds_accumulator["connections"]
            assert isinstance(connections, set)
            rtds[key] = (
                int(rtds_accumulator["events"]),
                len(connections),
                min(connections) if connections else None,
            )

    raw_gaps = connection.execute(
        """
        SELECT stream, connection_id, opened_at_ms
        FROM polymarket_stream_gap
        WHERE run_id = ?
          AND stream IN ('clob_market', 'binance_spot', 'polymarket_rtds')
        ORDER BY opened_at_ms, gap_id
        """,
        [run_id],
    ).fetchall()
    gap_counts: dict[tuple[int, str], int] = {}
    group_windows = sorted(
        {
            (int(event_start), int(window_start), int(window_end))
            for event_start, _asset, window_start, window_end in asset_windows
        }
    )
    for stream, connection_id, opened_at_ms in raw_gaps:
        normalized_stream = str(stream)
        normalized_connection = str(connection_id)
        opened = int(opened_at_ms)
        resumed = min(
            (
                started_at
                for (
                    candidate_stream,
                    candidate_connection,
                ), started_at in connection_starts.items()
                if candidate_stream == normalized_stream
                and candidate_connection != normalized_connection
                and started_at > opened
            ),
            default=None,
        )
        for event_start, window_start, window_end in group_windows:
            if opened <= window_end and (resumed is None or resumed > window_start):
                key = (event_start, normalized_stream)
                gap_counts[key] = gap_counts.get(key, 0) + 1
    snapshot_rows = connection.execute(
        """
        SELECT condition_id, observed_wall_ms
        FROM polymarket_market_snapshot WHERE run_id = ?
        ORDER BY condition_id
        """,
        [run_id],
    ).fetchall()
    return (
        clob,
        binance,
        rtds,
        {
            int(start): tuple(
                sorted(
                    f"stream_gap:{stream}:{count}"
                    for (candidate, stream), count in gap_counts.items()
                    if candidate == start
                )
            )
            for start in {candidate for candidate, _stream in gap_counts}
        },
        {str(condition): int(wall_ms) for condition, wall_ms in snapshot_rows},
        connection_starts,
    )


def _persist_report(
    store: PolymarketEvidenceStore,
    report: PolymarketContinuityReport,
) -> None:
    connection = store.connect()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_continuity_eligibility_report (
            report_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            run_report_sha256 VARCHAR NOT NULL,
            report_json VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS polymarket_continuity_eligibility_group (
            report_sha256 VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            eligible BOOLEAN NOT NULL,
            condition_ids_json VARCHAR NOT NULL,
            reasons_json VARCHAR NOT NULL,
            evidence_json VARCHAR NOT NULL,
            group_sha256 VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, event_start_ms)
        );
        """
    )
    report_row = (
        report.report_sha256,
        POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION,
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        report.run_id,
        report.run_report_sha256,
        _canonical_json(report.asdict()),
    )
    group_rows = [
        (
            report.report_sha256,
            group.event_start_ms,
            group.eligible,
            _canonical_json(list(group.condition_ids)),
            _canonical_json(list(group.reasons)),
            _canonical_json(group.evidence),
            group.group_sha256,
        )
        for group in report.groups
    ]
    existing = connection.execute(
        """
        SELECT report_sha256, schema_version, contract_sha256, run_id,
               run_report_sha256, report_json
        FROM polymarket_continuity_eligibility_report
        WHERE report_sha256 = ?
        """,
        [report.report_sha256],
    ).fetchone()
    if existing is not None:
        stored_groups = connection.execute(
            """
            SELECT report_sha256, event_start_ms, eligible,
                   condition_ids_json, reasons_json, evidence_json, group_sha256
            FROM polymarket_continuity_eligibility_group
            WHERE report_sha256 = ? ORDER BY event_start_ms
            """,
            [report.report_sha256],
        ).fetchall()
        if (
            tuple(existing) != report_row
            or [tuple(row) for row in stored_groups] != group_rows
        ):
            raise ValueError("stored Polymarket continuity report is inconsistent")
        return
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "INSERT INTO polymarket_continuity_eligibility_report VALUES (?, ?, ?, ?, ?, ?)",
            report_row,
        )
        connection.executemany(
            "INSERT INTO polymarket_continuity_eligibility_group VALUES (?, ?, ?, ?, ?, ?, ?)",
            group_rows,
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def evaluate_polymarket_continuity_eligibility(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    config: PolymarketContinuityConfig | None = None,
) -> PolymarketContinuityReport:
    """Persist deterministic gap/segment eligibility without consulting outcomes."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Polymarket continuity eligibility requires a run ID")
    cfg = (config or PolymarketContinuityConfig()).validated()
    run = (
        store.connect()
        .execute(
            """
        SELECT status, error, report_sha256, started_at_ms, ended_at_ms
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
            [selected],
        )
        .fetchone()
    )
    if run is None:
        raise ValueError("unknown Polymarket continuity run")
    if str(run[0]) not in {"complete", "degraded"} or str(run[1] or "").strip():
        raise ValueError("Polymarket continuity requires a finished error-free run")
    report_sha256 = str(run[2])
    started_at_ms = int(run[3])
    ended_at_ms = None if run[4] is None else int(run[4])
    if not _is_sha256(report_sha256) or ended_at_ms is None:
        raise ValueError("Polymarket continuity run report is invalid")
    integrity = store.integrity_errors(selected)
    if integrity:
        raise ValueError(
            "Polymarket continuity integrity failed: " + "; ".join(integrity)
        )
    markets = PolymarketEvidenceReplay.load_markets(store, run_id=selected)
    groups: dict[int, list[object]] = {}
    for market in markets:
        groups.setdefault(int(market.event_start_ms), []).append(market)
    synchronized: dict[int, tuple[object, ...]] = {}
    for event_start_ms, values in sorted(groups.items()):
        ordered = tuple(sorted(values, key=lambda item: _ASSETS.index(item.asset)))
        if tuple(item.asset for item in ordered) != _ASSETS:
            raise ValueError(
                "Polymarket continuity requires complete synchronized groups"
            )
        synchronized[event_start_ms] = ordered
    _create_window_tables(store, synchronized, cfg)
    clob, binance, rtds, gaps, snapshots, connection_starts = _continuity_evidence(
        store, selected
    )
    evaluated: list[PolymarketContinuityGroup] = []
    for event_start_ms, market_group in synchronized.items():
        reasons: list[str] = list(gaps.get(event_start_ms, ()))
        window_start = event_start_ms - cfg.chainlink_anchor_allowance_ms
        window_end = (
            market_group[0].end_ms
            - cfg.minimum_remaining_market_time_ms
            + cfg.maximum_execution_confirmation_delay_ms
        )
        evidence: dict[str, object] = {
            "run_bounds": {
                "started_at_ms": started_at_ms,
                "ended_at_ms": ended_at_ms,
            },
            "assets": {},
        }
        if started_at_ms > window_start:
            reasons.append("run_started_after_window_start")
        if ended_at_ms < window_end:
            reasons.append("run_ended_before_window_end")
        for market in market_group:
            asset_evidence: dict[str, object] = {}
            observed_wall_ms = snapshots.get(market.condition_id)
            asset_evidence["market_snapshot_received_wall_ms"] = observed_wall_ms
            if observed_wall_ms is None or observed_wall_ms > (
                event_start_ms + cfg.feature_warmup_ms
            ):
                reasons.append(f"late_or_missing_market_snapshot:{market.asset}")
            token_evidence: dict[str, object] = {}
            for outcome, token_id in zip(("Up", "Down"), market.token_ids, strict=True):
                values = clob.get((event_start_ms, market.asset, token_id))
                if values is None:
                    values = (0, 0, None, None, None)
                window_events = int(values[0] or 0)
                window_connections = int(values[1] or 0)
                window_connection = str(values[2] or "")
                baseline_connection = str(values[3] or "")
                baseline_wall_ms = None if values[4] is None else int(values[4])
                token_evidence[outcome] = {
                    "window_event_count": window_events,
                    "window_connection_count": window_connections,
                    "window_connection_id": window_connection,
                    "window_connection_started_at_ms": connection_starts.get(
                        ("clob_market", window_connection)
                    ),
                    "baseline_connection_id": baseline_connection,
                    "baseline_received_wall_ms": baseline_wall_ms,
                }
                if not baseline_connection:
                    reasons.append(f"missing_clob_baseline:{market.asset}:{outcome}")
                if window_events == 0:
                    reasons.append(f"missing_clob_window:{market.asset}:{outcome}")
                if window_connections != 1:
                    reasons.append(
                        f"clob_segment_count:{market.asset}:{outcome}:{window_connections}"
                    )
                connection_start = connection_starts.get(
                    ("clob_market", window_connection)
                )
                if connection_start is None or connection_start > window_start:
                    reasons.append(
                        f"clob_segment_started_after_window:{market.asset}:{outcome}"
                    )
                if baseline_connection and window_connection != baseline_connection:
                    reasons.append(
                        f"clob_baseline_segment_mismatch:{market.asset}:{outcome}"
                    )
            asset_evidence["clob"] = token_evidence
            binance_values = binance.get(
                (event_start_ms, market.asset), (0, 0, 0, None)
            )
            asset_evidence["binance"] = {
                "book_event_count": int(binance_values[0] or 0),
                "trade_event_count": int(binance_values[1] or 0),
                "connection_count": int(binance_values[2] or 0),
                "connection_id": str(binance_values[3] or ""),
                "connection_started_at_ms": connection_starts.get(
                    ("binance_spot", str(binance_values[3] or ""))
                ),
            }
            if int(binance_values[0] or 0) == 0:
                reasons.append(f"missing_binance_book:{market.asset}")
            if int(binance_values[1] or 0) == 0:
                reasons.append(f"missing_binance_trade:{market.asset}")
            if int(binance_values[2] or 0) != 1:
                reasons.append(
                    f"binance_segment_count:{market.asset}:{int(binance_values[2] or 0)}"
                )
            binance_connection_start = connection_starts.get(
                ("binance_spot", str(binance_values[3] or ""))
            )
            if (
                binance_connection_start is None
                or binance_connection_start > window_start
            ):
                reasons.append(f"binance_segment_started_after_window:{market.asset}")
            rtds_values = rtds.get((event_start_ms, market.asset), (0, 0, None))
            asset_evidence["rtds_chainlink"] = {
                "event_count": int(rtds_values[0] or 0),
                "connection_count": int(rtds_values[1] or 0),
                "connection_id": str(rtds_values[2] or ""),
                "connection_started_at_ms": connection_starts.get(
                    ("polymarket_rtds", str(rtds_values[2] or ""))
                ),
            }
            if int(rtds_values[0] or 0) == 0:
                reasons.append(f"missing_rtds_chainlink:{market.asset}")
            if int(rtds_values[1] or 0) != 1:
                reasons.append(
                    f"rtds_segment_count:{market.asset}:{int(rtds_values[1] or 0)}"
                )
            rtds_connection_start = connection_starts.get(
                ("polymarket_rtds", str(rtds_values[2] or ""))
            )
            if rtds_connection_start is None or rtds_connection_start > window_start:
                reasons.append(f"rtds_segment_started_after_window:{market.asset}")
            evidence["assets"][market.asset] = asset_evidence
        frozen_reasons = tuple(sorted(set(reasons)))
        provisional_group = PolymarketContinuityGroup(
            event_start_ms=event_start_ms,
            window_start_ms=window_start,
            window_end_ms=window_end,
            condition_ids=tuple(market.condition_id for market in market_group),
            eligible=not frozen_reasons,
            reasons=frozen_reasons,
            evidence=evidence,
            group_sha256="",
        )
        evaluated.append(
            replace(
                provisional_group,
                group_sha256=_sha256(provisional_group.identity_payload()),
            ).validated()
        )
    eligible_count = sum(group.eligible for group in evaluated)
    confirmation_reasons: list[str] = []
    if started_at_ms <= POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS:
        confirmation_reasons.append("run_started_before_round9_contract_commit")
    if eligible_count < cfg.minimum_eligible_groups:
        confirmation_reasons.append(
            f"eligible_group_count:{eligible_count}/{cfg.minimum_eligible_groups}"
        )
    frozen_confirmation_reasons = tuple(sorted(confirmation_reasons))
    provisional = PolymarketContinuityReport(
        run_id=selected,
        run_report_sha256=report_sha256,
        run_started_at_ms=started_at_ms,
        config=cfg,
        groups=tuple(evaluated),
        eligible_group_count=eligible_count,
        confirmation_eligible=not frozen_confirmation_reasons,
        confirmation_reasons=frozen_confirmation_reasons,
        report_sha256="",
    )
    report = replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    ).validated()
    _persist_report(store, report)
    return report


__all__ = [
    "POLYMARKET_ACTION_CONTRACT_COMMIT",
    "POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS",
    "POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION",
    "PolymarketContinuityConfig",
    "PolymarketContinuityGroup",
    "PolymarketContinuityReport",
    "evaluate_polymarket_continuity_eligibility",
]
