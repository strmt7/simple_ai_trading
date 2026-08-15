from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import duckdb
import pytest

from simple_ai_trading.polymarket_round21_binance_features import (
    POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    Round21OptionalBinanceFeatures,
)
from simple_ai_trading.polymarket_round21_sidecar_replay import Round21SidecarReplay
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
    POLYMARKET_ROUND28_FEATURE_NAMES,
    Round28BookTickerOverlayRow,
    compose_round28_feature_rows,
    load_round28_book_ticker_overlay,
    materialize_round28_book_ticker_overlay,
    validate_round28_book_ticker_report,
    write_round28_book_ticker_overlay,
)


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _base_row(decision_time_ms: int, *, marker: float = 0.0) -> Round27FeatureRow:
    return Round27FeatureRow.create(
        run_id="round28-test-run",
        condition_id="0x" + "1" * 64,
        event_start_ms=1_800_000,
        decision_time_ms=decision_time_ms,
        market_prior_probability=0.55,
        values=(marker,) + (0.0,) * (len(POLYMARKET_ROUND27_FEATURE_NAMES) - 1),
        maximum_receipt_wall_ms=decision_time_ms - 1,
        source_chain_sha256="2" * 64,
    )


def _sidecar_features(
    decision_time_ms: int,
    *,
    spot_available: bool = True,
    usdm_available: bool = True,
) -> Round21OptionalBinanceFeatures:
    if not spot_available:
        spot_values = (0.0,) * len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES)
        spot_hash = _EMPTY_SHA256
        spot_receipt = 0
    else:
        spot_values = tuple(
            (index + 1) / 1_000.0
            for index in range(len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES))
        )
        spot_hash = "3" * 64
        spot_receipt = decision_time_ms - 3
    if not usdm_available:
        usdm_values = (0.0,) * len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES)
        usdm_hash = _EMPTY_SHA256
        usdm_receipt = 0
    else:
        usdm_values = tuple(
            (index + 1) / 2_000.0
            for index in range(len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES))
        )
        usdm_hash = "4" * 64
        usdm_receipt = decision_time_ms - 2
    return Round21OptionalBinanceFeatures(
        decision_time_ms=decision_time_ms,
        spot_values=spot_values,
        usdm_values=usdm_values,
        spot_available=spot_available,
        usdm_available=usdm_available,
        spot_source_chain_sha256=spot_hash,
        usdm_source_chain_sha256=usdm_hash,
        spot_maximum_receipt_ms=spot_receipt,
        usdm_maximum_receipt_ms=usdm_receipt,
    )


def _replay(
    features: tuple[Round21OptionalBinanceFeatures, ...],
) -> Round21SidecarReplay:
    return Round21SidecarReplay(
        terminal_manifest_sha256="5" * 64,
        eligible_run_ids=("sidecar-run",),
        decision_times_ms=tuple(item.decision_time_ms for item in features),
        features=features,
        raw_message_count=5,
        stream_counts={"binance_spot": 2, "binance_futures": 3},
        stream_gap_count=2,
        receipt_chain_sha256="6" * 64,
    ).validated()


def test_round28_feature_contract_is_incremental_and_professionally_named() -> None:
    assert len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES) == 96
    assert len(POLYMARKET_ROUND28_FEATURE_NAMES) == 278
    assert len(set(POLYMARKET_ROUND28_FEATURE_NAMES)) == 278
    assert not set(POLYMARKET_ROUND27_FEATURE_NAMES) & set(
        POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES
    )
    assert all(
        name.startswith(("binance_spot.", "binance_usdm.", "binance_cross."))
        for name in POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES
    )
    assert not any("book_book" in name for name in POLYMARKET_ROUND28_FEATURE_NAMES)


def test_round28_overlay_is_causal_matched_and_gap_explicit() -> None:
    decisions = (1_830_000, 1_831_000, 1_832_000)
    base = tuple(_base_row(decision) for decision in decisions)
    replay = _replay(
        (
            _sidecar_features(decisions[0]),
            _sidecar_features(decisions[1], spot_available=False, usdm_available=False),
            _sidecar_features(decisions[2], usdm_available=False),
        )
    )

    overlay, report = materialize_round28_book_ticker_overlay(
        base_rows=base,
        sidecar_replay=replay,
    )

    assert tuple(row.decision_time_ms for row in overlay) == (decisions[0],)
    assert report["accepted_decision_count"] == 1
    assert report["rejected_decision_count"] == 2
    assert report["sidecar_stream_gap_count"] == 2
    assert report["rejection_counts"] == {
        "spot_bbo_unavailable_or_stale": 1,
        "usdm_bbo_unavailable_or_stale": 1,
    }
    assert report["official_outcomes_accessed"] is False
    assert report["edge_claim"] is False
    assert validate_round28_book_ticker_report(report) == report

    augmented = compose_round28_feature_rows(
        base_rows=base,
        overlay_rows=overlay,
        report=report,
    )
    assert len(augmented) == 1
    assert augmented[0].decision_time_ms == decisions[0]
    assert (
        augmented[0].values[: len(POLYMARKET_ROUND27_FEATURE_NAMES)] == base[0].values
    )
    assert (
        augmented[0].values[len(POLYMARKET_ROUND27_FEATURE_NAMES) :]
        == overlay[0].values
    )

    payload = asdict(augmented[0])
    payload.pop("row_sha256")
    payload["source_chain_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="feature row differs"):
        replace(
            augmented[0],
            source_chain_sha256="0" * 64,
            row_sha256=_canonical_sha256(payload),
        ).validated()


def test_round28_rejects_decision_or_base_population_drift() -> None:
    base = (_base_row(1_830_000),)
    replay = _replay((_sidecar_features(1_831_000),))
    with pytest.raises(ValueError, match="decision grids"):
        materialize_round28_book_ticker_overlay(
            base_rows=base,
            sidecar_replay=replay,
        )

    overlay, report = materialize_round28_book_ticker_overlay(
        base_rows=base,
        sidecar_replay=_replay((_sidecar_features(1_830_000),)),
    )
    with pytest.raises(ValueError, match="population differs"):
        compose_round28_feature_rows(
            base_rows=(_base_row(1_830_000, marker=1.0),),
            overlay_rows=overlay,
            report=report,
        )


def test_round28_overlay_store_round_trip_and_tamper_detection(tmp_path) -> None:
    base = (_base_row(1_830_000),)
    overlay, report = materialize_round28_book_ticker_overlay(
        base_rows=base,
        sidecar_replay=_replay((_sidecar_features(1_830_000),)),
    )
    store = tmp_path / "round28-overlay.duckdb"

    write_round28_book_ticker_overlay(store, rows=overlay, report=report)
    loaded_rows, loaded_report = load_round28_book_ticker_overlay(store)

    assert loaded_rows == overlay
    assert loaded_report == report
    with pytest.raises(ValueError, match="store inputs differ"):
        write_round28_book_ticker_overlay(store, rows=overlay, report=report)

    with duckdb.connect(str(store)) as connection:
        connection.execute(
            "UPDATE round28_book_ticker_manifest SET report_sha256 = ?",
            ["0" * 64],
        )
        connection.execute("CHECKPOINT")
    with pytest.raises(ValueError, match="store audit differs"):
        load_round28_book_ticker_overlay(store)


def test_round28_overlay_row_rejects_future_receipt() -> None:
    feature = _sidecar_features(1_830_000)
    row = Round28BookTickerOverlayRow.create(feature)
    with pytest.raises(ValueError, match="overlay row differs"):
        Round28BookTickerOverlayRow(
            decision_time_ms=row.decision_time_ms,
            values=row.values,
            feature_names_sha256=row.feature_names_sha256,
            spot_source_chain_sha256=row.spot_source_chain_sha256,
            usdm_source_chain_sha256=row.usdm_source_chain_sha256,
            maximum_receipt_wall_ms=row.decision_time_ms,
            source_chain_sha256=row.source_chain_sha256,
            row_sha256=row.row_sha256,
        ).validated()
