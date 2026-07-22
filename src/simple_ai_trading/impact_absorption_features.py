"""Causal feature-source replay for qualified Round 73 captures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import zstandard

from .impact_absorption import (
    DepthLevelChange,
    L2BookState,
    ROUND73_LEVEL_BANDS,
    SynchronizedDepthBook,
    pre_event_level_band,
    validate_combined_stream_name,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_CONTRACT_SHA256,
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_DEPTH_BAND_FLOW_TABLE,
    IMPACT_DEPTH_UPDATE_TABLE,
    IMPACT_EVENT_LINK_TABLE,
    IMPACT_L2_STATE_TABLE,
    ImpactAbsorptionStore,
)
from .impact_capture_frame import decode_impact_capture_frame


ROUND73_FEATURE_SOURCE_SCHEMA_VERSION = "round-073-feature-source-diagnostic-v2"
_FRAME_BATCH_SIZE = 64
_BAND_METRICS = ("added_quote", "removed_quote", "change_count")
_DEPTH_BAND_VALUE_COLUMNS = tuple(
    f"{side}_{metric}_{band}"
    for side in ("bid", "ask")
    for band in ROUND73_LEVEL_BANDS
    for metric in _BAND_METRICS
)
_V4_SCHEMA_VERSION = "round-073-prospective-evidence-v4"
_V4_CAPTURE_CONTRACT_SHA256 = (
    "c34687c5dff9a4eda98b2e50d6444a12ee1a4f5594806c2410e15cb0242d7529"
)
_V4_REPORT_SCHEMA_VERSION = "round-073-capture-report-v4"
_V5_SCHEMA_VERSION = "round-073-prospective-evidence-v5"
_V5_CAPTURE_CONTRACT_SHA256 = (
    "63a440f1fb875db8ee78bab1631033f24850a65cc7ed80d4fd37078dd6ee9a1b"
)
_V5_REPORT_SCHEMA_VERSION = "round-073-capture-report-v5"
_DEPTH_BAND_SCHEMAS = frozenset(
    {IMPACT_CAPTURE_SCHEMA_VERSION, _V5_SCHEMA_VERSION}
)


def _strict_json_object(raw_text: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, Mapping):
        raise ValueError("feature-source wire payload must be a JSON object")
    return parsed


def _finite_close(observed: object, expected: float, label: str) -> None:
    value = float(observed)
    if not math.isfinite(value) or not math.isclose(
        value,
        float(expected),
        rel_tol=1e-11,
        abs_tol=1e-9,
    ):
        raise ValueError(f"Round 73 feature-source mismatch: {label}")


def _compare_state(row: tuple[object, ...], state: L2BookState) -> None:
    if int(row[2]) != state.update_id:
        raise ValueError("Round 73 feature-source mismatch: L2 update ID")
    scalar_values = (
        (row[3], state.best_bid, "L2 best bid"),
        (row[4], state.best_ask, "L2 best ask"),
        (row[5], state.spread_bps, "L2 spread"),
        (row[6], state.mid, "L2 mid"),
        (row[11], state.bid_depth_quote_5, "L2 bid depth 5"),
        (row[12], state.ask_depth_quote_5, "L2 ask depth 5"),
        (row[13], state.bid_depth_quote_10, "L2 bid depth 10"),
        (row[14], state.ask_depth_quote_10, "L2 ask depth 10"),
        (row[15], state.bid_depth_quote_20, "L2 bid depth 20"),
        (row[16], state.ask_depth_quote_20, "L2 ask depth 20"),
        (row[17], state.imbalance_5, "L2 imbalance 5"),
        (row[18], state.imbalance_10, "L2 imbalance 10"),
        (row[19], state.imbalance_20, "L2 imbalance 20"),
    )
    for observed, expected, label in scalar_values:
        _finite_close(observed, expected, label)
    expected_arrays = (
        tuple(price for price, _quantity in state.bid_levels),
        tuple(quantity for _price, quantity in state.bid_levels),
        tuple(price for price, _quantity in state.ask_levels),
        tuple(quantity for _price, quantity in state.ask_levels),
    )
    for observed, expected, label in zip(
        row[7:11],
        expected_arrays,
        ("bid prices", "bid quantities", "ask prices", "ask quantities"),
        strict=True,
    ):
        values = tuple(float(value) for value in observed)
        if len(values) != len(expected):
            raise ValueError(f"Round 73 feature-source mismatch: L2 {label} length")
        for index, (value, expected_value) in enumerate(
            zip(values, expected, strict=True)
        ):
            _finite_close(value, expected_value, f"L2 {label} {index}")


@dataclass(frozen=True)
class Round73FeatureSourceDiagnostic:
    run_id: str
    capture_contract_sha256: str
    stored_report_sha256: str
    capture_audit_passed: bool
    frame_count: int
    message_count: int
    depth_snapshot_count: int
    depth_update_count: int
    synchronized_depth_update_count: int
    stale_depth_update_count: int
    level_change_count: int
    stored_depth_band_row_count: int
    stored_depth_band_rows_reconciled: bool
    symbols: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = ROUND73_FEATURE_SOURCE_SCHEMA_VERSION
        payload["feature_semantics"] = {
            "level_band_clock": "pre-event synchronized local book",
            "level_band_definition": {
                "levels_1_5": "prospective price rank 1 through 5",
                "levels_6_10": "prospective price rank 6 through 10",
                "levels_11_20": "prospective price rank 11 through 20",
                "outside_20": "prospective price rank greater than 20",
            },
            "availability_clock": "received_monotonic_ns",
            "future_or_target_data_used": False,
            "identity_whale_or_manipulation_inference": False,
        }
        payload["authority"] = {
            "depth_band_primitives_reconstructed": True,
            "stored_depth_band_rows_reconciled": (
                self.stored_depth_band_rows_reconciled
            ),
            "all_grid_anchor_features_constructed": False,
            "shock_threshold_selected": False,
            "target_constructed": False,
            "model_evaluated": False,
            "profitability_claim": False,
            "trading_authority": False,
        }
        return payload


def _empty_symbol_state() -> dict[str, object]:
    return {
        "depth_snapshots": 0,
        "depth_updates": 0,
        "synchronized_depth_updates": 0,
        "stale_depth_updates": 0,
        "level_changes": 0,
        "quote_flow": {
            side: {
                band: {"added_quote": 0.0, "removed_quote": 0.0, "changes": 0}
                for band in ROUND73_LEVEL_BANDS
            }
            for side in ("bid", "ask")
        },
    }


def _depth_band_flow(
    state: L2BookState,
    changes: Iterable[DepthLevelChange],
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {
        side: {
            band: {"added_quote": 0.0, "removed_quote": 0.0, "change_count": 0}
            for band in ROUND73_LEVEL_BANDS
        }
        for side in ("bid", "ask")
    }
    for change in changes:
        band = pre_event_level_band(state, change)
        bucket = output[change.side][band]
        bucket["added_quote"] = float(bucket["added_quote"]) + change.added_quote
        bucket["removed_quote"] = (
            float(bucket["removed_quote"]) + change.removed_quote
        )
        bucket["change_count"] = int(bucket["change_count"]) + 1
    return output


def _compare_depth_band_row(
    row: tuple[object, ...],
    flow: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    expected = tuple(
        flow[side][band][metric]
        for side in ("bid", "ask")
        for band in ROUND73_LEVEL_BANDS
        for metric in _BAND_METRICS
    )
    if len(row) != len(expected):
        raise ValueError("Round 73 stored depth-band row width differs")
    for column, observed, expected_value in zip(
        _DEPTH_BAND_VALUE_COLUMNS,
        row,
        expected,
        strict=True,
    ):
        if column.endswith("change_count"):
            if int(observed) != int(expected_value):
                raise ValueError(
                    f"Round 73 stored depth-band mismatch: {column}"
                )
        else:
            _finite_close(
                observed,
                float(expected_value),
                f"stored depth band {column}",
            )


def diagnose_round73_feature_source(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73FeatureSourceDiagnostic:
    """Replay one gated v4-v6 run and reconcile causal depth-band primitives."""

    selected = str(run_id).strip().lower()
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        run = connection.execute(
            """
            SELECT schema_version, capture_contract_sha256, status
            FROM impact_capture_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None:
            raise ValueError("Round 73 feature-source run was not found")
        run_schema = str(run[0])
        capture_contracts = {
            IMPACT_CAPTURE_SCHEMA_VERSION: IMPACT_CAPTURE_CONTRACT_SHA256,
            _V5_SCHEMA_VERSION: _V5_CAPTURE_CONTRACT_SHA256,
            _V4_SCHEMA_VERSION: _V4_CAPTURE_CONTRACT_SHA256,
        }
        try:
            expected_contract = capture_contracts[run_schema]
        except KeyError as exc:
            raise ValueError(
                "Round 73 feature-source replay requires capture v4, v5, or v6"
            ) from exc
        if str(run[1]) != expected_contract:
            raise ValueError("Round 73 feature-source capture contract differs")
        if str(run[2]) != "completed":
            raise ValueError("Round 73 feature-source replay requires a completed run")
        report_row = connection.execute(
            """
            SELECT schema_version, report_json, report_sha256
            FROM impact_capture_report WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if report_row is None:
            raise ValueError("Round 73 feature-source replay requires a terminal report")
        expected_report_schema = {
            IMPACT_CAPTURE_SCHEMA_VERSION: IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
            _V5_SCHEMA_VERSION: _V5_REPORT_SCHEMA_VERSION,
            _V4_SCHEMA_VERSION: _V4_REPORT_SCHEMA_VERSION,
        }[run_schema]
        if str(report_row[0]) != expected_report_schema:
            raise ValueError("Round 73 feature-source report schema differs")
        report_text = str(report_row[1])
        if hashlib.sha256(report_text.encode("ascii")).hexdigest() != str(report_row[2]):
            raise ValueError("Round 73 feature-source report hash differs")
        report = _strict_json_object(report_text)
        if not (
            report.get("qualification_passed") is True
            or report.get("capture_gate_passed") is True
        ):
            raise ValueError(
                "Round 73 feature-source replay requires a passed capture gate"
            )
        segment_rows = connection.execute(
            """
            SELECT symbol, status, tick_size FROM impact_capture_segment
            WHERE run_id = ? ORDER BY symbol
            """,
            [selected],
        ).fetchall()
        if tuple(str(row[0]) for row in segment_rows) != IMPACT_CAPTURE_SYMBOLS:
            raise ValueError("Round 73 feature-source segments are incomplete")
        if any(str(row[1]) != "valid" for row in segment_rows):
            raise ValueError("Round 73 feature-source segment is invalid")
        audit = store.audit_run(selected)
        if not audit.passed:
            raise ValueError("Round 73 feature-source capture audit failed")

        books = {
            str(symbol): SynchronizedDepthBook(str(symbol), float(tick_size))
            for symbol, _status, tick_size in segment_rows
        }
        symbol_state = {
            symbol: _empty_symbol_state() for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        event_link_table = (
            IMPACT_EVENT_LINK_TABLE
            if run_schema in _DEPTH_BAND_SCHEMAS
            else "impact_event_link_v4"
        )
        frame_cursor = connection.cursor()
        lookup = connection.cursor()
        frame_cursor.execute(
            """
            SELECT frame_index, message_count, uncompressed_bytes,
                   compressed_payload
            FROM impact_capture_frame WHERE run_id = ? ORDER BY frame_index
            """,
            [selected],
        )
        decompressor = zstandard.ZstdDecompressor()
        replayed_messages = 0
        replayed_frames = 0
        reconciled_band_rows = 0

        while frame_rows := frame_cursor.fetchmany(_FRAME_BATCH_SIZE):
            first_frame = int(frame_rows[0][0])
            last_frame = int(frame_rows[-1][0])
            links = {
                (int(row[0]), int(row[1])): (str(row[2]), str(row[3]))
                for row in lookup.execute(
                    f"""
                    SELECT frame_index, message_index, event_type, symbol
                    FROM {event_link_table}
                    WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                    ORDER BY frame_index, message_index
                    """,
                    [selected, first_frame, last_frame],
                ).fetchall()
            }
            depth_rows = {
                (int(row[0]), int(row[1])): tuple(row)
                for row in lookup.execute(
                    f"""
                    SELECT frame_index, message_index, first_update_id,
                           final_update_id, previous_update_id, stale,
                           best_bid, best_ask, bid_added_quote,
                           bid_removed_quote, ask_added_quote, ask_removed_quote
                    FROM {IMPACT_DEPTH_UPDATE_TABLE}
                    WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                    """,
                    [selected, first_frame, last_frame],
                ).fetchall()
            }
            l2_rows = {
                (int(row[0]), int(row[1])): tuple(row)
                for row in lookup.execute(
                    f"""
                    SELECT frame_index, message_index, update_id, best_bid,
                           best_ask, spread_bps, mid, bid_prices,
                           bid_quantities, ask_prices, ask_quantities,
                           bid_depth_quote_5, ask_depth_quote_5,
                           bid_depth_quote_10, ask_depth_quote_10,
                           bid_depth_quote_20, ask_depth_quote_20,
                           imbalance_5, imbalance_10, imbalance_20
                    FROM {IMPACT_L2_STATE_TABLE}
                    WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                    """,
                    [selected, first_frame, last_frame],
                ).fetchall()
            }
            stored_band_rows = (
                {
                    (int(row[0]), int(row[1])): tuple(row[2:])
                    for row in lookup.execute(
                        f"SELECT frame_index, message_index, "
                        f"{', '.join(_DEPTH_BAND_VALUE_COLUMNS)} "
                        f"FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} "
                        "WHERE run_id = ? AND frame_index BETWEEN ? AND ?",
                        [selected, first_frame, last_frame],
                    ).fetchall()
                }
                if run_schema in _DEPTH_BAND_SCHEMAS
                else {}
            )
            for frame_index_value, message_count, uncompressed_bytes, blob in frame_rows:
                frame_index = int(frame_index_value)
                uncompressed = decompressor.decompress(
                    bytes(blob),
                    max_output_size=int(uncompressed_bytes),
                )
                decoded = decode_impact_capture_frame(
                    uncompressed,
                    expected_message_count=int(message_count),
                )
                replayed_frames += 1
                replayed_messages += len(decoded)
                for item in decoded:
                    key = (frame_index, item.message_index)
                    try:
                        event_type, symbol = links[key]
                    except KeyError as exc:
                        raise ValueError(
                            "Round 73 feature-source event link is missing"
                        ) from exc
                    if event_type == "depthSnapshot":
                        body = _strict_json_object(item.record.raw_text)
                        books[symbol].initialize(body)
                        symbol_state[symbol]["depth_snapshots"] = (
                            int(symbol_state[symbol]["depth_snapshots"]) + 1
                        )
                        continue
                    if event_type != "depthUpdate":
                        continue
                    root = _strict_json_object(item.record.raw_text)
                    stream_name = str(root.get("stream", ""))
                    payload = root.get("data")
                    if not isinstance(payload, Mapping):
                        raise ValueError("Round 73 depth wrapper has no object data")
                    validate_combined_stream_name(
                        stream_name,
                        event_type="depthUpdate",
                        symbol=symbol,
                    )
                    book = books[symbol]
                    pre_state = book.state(20)
                    event = book.apply(
                        payload,
                        receive_time_ns=item.record.received_monotonic_ns,
                    )
                    typed = depth_rows.get(key)
                    if typed is None:
                        raise ValueError("Round 73 typed depth row is missing")
                    if (
                        int(typed[2]) != event.first_update_id
                        or int(typed[3]) != event.final_update_id
                        or int(typed[4]) != event.previous_update_id
                        or bool(typed[5]) != event.stale
                    ):
                        raise ValueError("Round 73 typed depth identity differs")
                    _finite_close(typed[6], event.best_bid, "depth best bid")
                    _finite_close(typed[7], event.best_ask, "depth best ask")
                    sums = {
                        "bid_added": math.fsum(
                            change.added_quote
                            for change in event.changes
                            if change.side == "bid"
                        ),
                        "bid_removed": math.fsum(
                            change.removed_quote
                            for change in event.changes
                            if change.side == "bid"
                        ),
                        "ask_added": math.fsum(
                            change.added_quote
                            for change in event.changes
                            if change.side == "ask"
                        ),
                        "ask_removed": math.fsum(
                            change.removed_quote
                            for change in event.changes
                            if change.side == "ask"
                        ),
                    }
                    for observed, name in zip(
                        typed[8:12],
                        ("bid_added", "bid_removed", "ask_added", "ask_removed"),
                        strict=True,
                    ):
                        _finite_close(observed, sums[name], f"depth {name} quote")
                    event_band_flow = _depth_band_flow(pre_state, event.changes)
                    if run_schema in _DEPTH_BAND_SCHEMAS:
                        stored_band_row = stored_band_rows.get(key)
                        if stored_band_row is None:
                            raise ValueError(
                                "Round 73 stored depth-band row is missing"
                            )
                        _compare_depth_band_row(stored_band_row, event_band_flow)
                        reconciled_band_rows += 1
                    state = symbol_state[symbol]
                    state["depth_updates"] = int(state["depth_updates"]) + 1
                    if event.stale:
                        state["stale_depth_updates"] = (
                            int(state["stale_depth_updates"]) + 1
                        )
                        if key in l2_rows:
                            raise ValueError("stale Round 73 depth has an L2 state")
                        continue
                    state["synchronized_depth_updates"] = (
                        int(state["synchronized_depth_updates"]) + 1
                    )
                    state["level_changes"] = int(state["level_changes"]) + len(
                        event.changes
                    )
                    quote_flow = state["quote_flow"]
                    if not isinstance(quote_flow, dict):
                        raise RuntimeError("Round 73 quote-flow state is invalid")
                    for side in ("bid", "ask"):
                        for band in ROUND73_LEVEL_BANDS:
                            event_bucket = event_band_flow[side][band]
                            bucket = quote_flow[side][band]
                            bucket["added_quote"] += event_bucket["added_quote"]
                            bucket["removed_quote"] += event_bucket["removed_quote"]
                            bucket["changes"] += event_bucket["change_count"]
                    stored_l2 = l2_rows.get(key)
                    if stored_l2 is None:
                        raise ValueError("accepted Round 73 depth has no L2 state")
                    _compare_state(stored_l2, book.state(20))

        for symbol, state in symbol_state.items():
            if int(state["depth_snapshots"]) != 1:
                raise ValueError(
                    f"Round 73 feature-source {symbol} snapshot count differs"
                )
            if int(state["synchronized_depth_updates"]) < 1:
                raise ValueError(
                    f"Round 73 feature-source {symbol} has no synchronized depth"
                )
        depth_updates = sum(
            int(state["depth_updates"]) for state in symbol_state.values()
        )
        synchronized = sum(
            int(state["synchronized_depth_updates"])
            for state in symbol_state.values()
        )
        stale = sum(
            int(state["stale_depth_updates"]) for state in symbol_state.values()
        )
        level_changes = sum(
            int(state["level_changes"]) for state in symbol_state.values()
        )
        stored_depth_updates = int(
            connection.execute(
                f"SELECT count(*) FROM {IMPACT_DEPTH_UPDATE_TABLE} WHERE run_id = ?",
                [selected],
            ).fetchone()[0]
        )
        if depth_updates != stored_depth_updates:
            raise ValueError("Round 73 replayed depth count differs")
        if (
            run_schema in _DEPTH_BAND_SCHEMAS
            and reconciled_band_rows != depth_updates
        ):
            raise ValueError("Round 73 reconciled depth-band count differs")
        if replayed_frames != audit.frame_count or replayed_messages != audit.message_count:
            raise ValueError("Round 73 feature-source frame totals differ")
        for state in symbol_state.values():
            quote_flow = state["quote_flow"]
            if not isinstance(quote_flow, dict):
                raise RuntimeError("Round 73 quote-flow state is invalid")
            for side in ("bid", "ask"):
                for band in ROUND73_LEVEL_BANDS:
                    bucket = quote_flow[side][band]
                    if not math.isfinite(bucket["added_quote"]) or not math.isfinite(
                        bucket["removed_quote"]
                    ):
                        raise ValueError("Round 73 quote-flow total is nonfinite")

    return Round73FeatureSourceDiagnostic(
        run_id=selected,
        capture_contract_sha256=expected_contract,
        stored_report_sha256=str(report_row[2]),
        capture_audit_passed=True,
        frame_count=replayed_frames,
        message_count=replayed_messages,
        depth_snapshot_count=sum(
            int(state["depth_snapshots"]) for state in symbol_state.values()
        ),
        depth_update_count=depth_updates,
        synchronized_depth_update_count=synchronized,
        stale_depth_update_count=stale,
        level_change_count=level_changes,
        stored_depth_band_row_count=reconciled_band_rows,
        stored_depth_band_rows_reconciled=(
            run_schema in _DEPTH_BAND_SCHEMAS
            and reconciled_band_rows == depth_updates
        ),
        symbols=symbol_state,
    )


__all__ = [
    "ROUND73_FEATURE_SOURCE_SCHEMA_VERSION",
    "ROUND73_LEVEL_BANDS",
    "Round73FeatureSourceDiagnostic",
    "diagnose_round73_feature_source",
    "pre_event_level_band",
]
