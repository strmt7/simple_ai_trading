from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_cohort import (
    ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
    ROUND73_SHOCK_ANCHOR_TABLE,
    ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
    ROUND73_SHOCK_STUDY_TABLE,
    ROUND73_STUDY_NOT_BEFORE_WALL_NS,
    Round73ShockAnchor,
    _create_tables as _create_cohort_tables,
)
from simple_ai_trading.impact_absorption_corpus import (
    ROUND73_CORPUS_CONTRACT_SHA256,
    ROUND73_CORPUS_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_grid import (
    ROUND73_GRID_CONTRACT_SHA256,
    ROUND73_GRID_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_target_store import _TargetReplay
from simple_ai_trading.impact_absorption_target_store_v2 import (
    ROUND73_TARGET_V2_MANIFEST_TABLE,
    ROUND73_TARGET_V2_OPTION_TABLE,
    audit_round73_selected_anchor_targets,
    audit_round73_target_study,
    build_round73_selected_anchor_targets,
    seal_round73_target_study,
)
from simple_ai_trading.impact_absorption_holdout_store import (
    publish_round73_pretest_manifest,
    round73_development_row_identities,
    unlock_round73_test_targets,
)
from simple_ai_trading.impact_absorption_target_store_v3 import (
    ROUND73_PRETEST_MODEL_ARTIFACT_TABLE,
    ROUND73_TARGET_V3_OPTION_TABLE,
    ROUND73_TARGET_V3_TEST_STUDY_TABLE,
    audit_round73_role_targets,
    build_round73_role_targets,
    seal_round73_development_targets,
    seal_round73_test_targets,
    stage_round73_role_targets,
)
from simple_ai_trading.impact_absorption_targets import Round73MarketQuantityRules


STUDY_ID = "b" * 32
RUN_IDS = ("1" * 32, "2" * 32)
START_WALL_NS = int(datetime(2026, 7, 23, tzinfo=UTC).timestamp()) * 1_000_000_000
START_MONOTONIC_NS = 1_000_000_000_000


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _stream_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _state(symbol: str, update_id: int) -> L2BookState:
    bids = tuple((99.9 - index * 0.1, 100.0) for index in range(20))
    asks = tuple((100.1 + index * 0.1, 100.0) for index in range(20))

    def depth(levels: tuple[tuple[float, float], ...], count: int) -> float:
        return sum(price * quantity for price, quantity in levels[:count])

    bid5, ask5 = depth(bids, 5), depth(asks, 5)
    bid10, ask10 = depth(bids, 10), depth(asks, 10)
    bid20, ask20 = depth(bids, 20), depth(asks, 20)
    return L2BookState(
        symbol=symbol,
        update_id=update_id,
        best_bid=99.9,
        best_ask=100.1,
        spread_bps=20.0,
        mid=100.0,
        bid_levels=bids,
        ask_levels=asks,
        bid_depth_quote_5=bid5,
        ask_depth_quote_5=ask5,
        bid_depth_quote_10=bid10,
        ask_depth_quote_10=ask10,
        bid_depth_quote_20=bid20,
        ask_depth_quote_20=ask20,
        imbalance_5=(bid5 - ask5) / (bid5 + ask5),
        imbalance_10=(bid10 - ask10) / (bid10 + ask10),
        imbalance_20=(bid20 - ask20) / (bid20 + ask20),
    )


def _rules() -> dict[str, Round73MarketQuantityRules]:
    return {
        symbol: Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size="0.001",
            minimum_quantity="0.001",
            maximum_quantity="100000",
            minimum_notional="5",
        )
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }


def _replay(_connection: object, **kwargs: object):
    replay = _TargetReplay(
        run_id=str(kwargs["run_id"]),
        anchors=kwargs["anchors"],
        quantity_rules=kwargs["quantity_rules"],
        run_started_wall_ns=int(kwargs["run_started_wall_ns"]),
        run_started_monotonic_ns=int(kwargs["run_started_monotonic_ns"]),
        coverage_end_monotonic_ns=START_MONOTONIC_NS + 400_000_000_000,
        entry_delays_ms=kwargs["entry_delays_ms"],
        horizons_ms=kwargs["horizons_ms"],
        reference_notionals=kwargs["reference_notionals"],
        sides=kwargs["sides"],
    )
    replay.observe_mark(
        symbol="BTCUSDT",
        next_funding_time_ms=2_000_000_000_000,
    )
    replay.observe_depth(
        symbol="BTCUSDT",
        received_monotonic_ns=START_MONOTONIC_NS + 100_000_000,
        state=_state("BTCUSDT", 1),
    )
    for index in range(2, 608):
        timestamp = START_MONOTONIC_NS + index * 500_000_000
        replay.before_record(timestamp)
        replay.observe_depth(
            symbol="BTCUSDT",
            received_monotonic_ns=timestamp,
            state=_state("BTCUSDT", index),
        )
    return replay.finish()


def _seed(
    database: Path,
    *,
    start_wall_ns: int = START_WALL_NS,
    utc_day: str = "2026-07-23",
    include_test_anchor: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    corpus_hashes = {run_id: _sha(f"corpus-{run_id}") for run_id in RUN_IDS}
    grid_hashes = {run_id: _sha(f"grid-{run_id}") for run_id in RUN_IDS}
    sources: list[dict[str, object]] = []
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE impact_capture_run (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                capture_contract_sha256 VARCHAR NOT NULL,
                started_wall_ns UBIGINT NOT NULL,
                started_monotonic_ns UBIGINT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE impact_capture_segment (
                run_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                tick_size DOUBLE NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE impact_corpus_run_manifest_v3 (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL,
                coverage_start_wall_ns UBIGINT NOT NULL,
                coverage_end_wall_ns UBIGINT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE impact_feature_run_manifest_v4 (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                source_corpus_manifest_sha256 VARCHAR NOT NULL,
                build_manifest_sha256 VARCHAR NOT NULL
            )
            """
        )
        for index, run_id in enumerate(RUN_IDS):
            started_wall = start_wall_ns + index * 500_000_000_000
            coverage_start = started_wall
            coverage_end = started_wall + 400_000_000_000
            connection.execute(
                "INSERT INTO impact_capture_run VALUES (?, ?, ?, ?, ?)",
                [
                    run_id,
                    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
                    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
                    started_wall,
                    START_MONOTONIC_NS,
                ],
            )
            for symbol in IMPACT_CAPTURE_SYMBOLS:
                connection.execute(
                    "INSERT INTO impact_capture_segment VALUES (?, ?, 'valid', 0.01)",
                    [run_id, symbol],
                )
            connection.execute(
                "INSERT INTO impact_corpus_run_manifest_v3 VALUES (?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    ROUND73_CORPUS_SCHEMA_VERSION,
                    ROUND73_CORPUS_CONTRACT_SHA256,
                    corpus_hashes[run_id],
                    coverage_start,
                    coverage_end,
                ],
            )
            connection.execute(
                "INSERT INTO impact_feature_run_manifest_v4 VALUES (?, ?, ?, ?, ?)",
                [
                    run_id,
                    ROUND73_GRID_SCHEMA_VERSION,
                    ROUND73_GRID_CONTRACT_SHA256,
                    corpus_hashes[run_id],
                    grid_hashes[run_id],
                ],
            )
            sources.append(
                {
                    "run_id": run_id,
                    "corpus_manifest_sha256": corpus_hashes[run_id],
                    "grid_manifest_sha256": grid_hashes[run_id],
                    "coverage_start_wall_ns": coverage_start,
                    "coverage_end_wall_ns": coverage_end,
                }
            )
        _create_cohort_tables(connection)
        anchor_inputs = [
            {
                "study_id": STUDY_ID,
                "run_id": RUN_IDS[0],
                "symbol": "BTCUSDT",
                "anchor_index": 0,
                "anchor_monotonic_ns": START_MONOTONIC_NS + 1_000_000_000,
                "anchor_wall_ns": start_wall_ns + 1_000_000_000,
                "source_max_received_monotonic_ns": (START_MONOTONIC_NS + 900_000_000),
                "utc_day": utc_day,
                "day_ordinal": 1,
                "role": "training",
                "shock_ratio": 5.0,
                "shock_direction": 1,
                "shock_direction_taker_share": 0.8,
                "feature_vector_sha256": _sha("vector-training"),
            }
        ]
        if include_test_anchor:
            anchor_inputs.append(
                {
                    "study_id": STUDY_ID,
                    "run_id": RUN_IDS[1],
                    "symbol": "BTCUSDT",
                    "anchor_index": 0,
                    "anchor_monotonic_ns": START_MONOTONIC_NS + 1_000_000_000,
                    "anchor_wall_ns": start_wall_ns + 501_000_000_000,
                    "source_max_received_monotonic_ns": (
                        START_MONOTONIC_NS + 900_000_000
                    ),
                    "utc_day": "2026-07-29",
                    "day_ordinal": 6,
                    "role": "test",
                    "shock_ratio": 5.0,
                    "shock_direction": -1,
                    "shock_direction_taker_share": 0.8,
                    "feature_vector_sha256": _sha("vector-test"),
                }
            )
        anchor_hashes: list[str] = []
        for anchor_values in anchor_inputs:
            anchor_hash = _sha(_canonical_json(list(anchor_values.values())))
            anchor_hashes.append(anchor_hash)
            anchor = Round73ShockAnchor(
                **anchor_values,  # type: ignore[arg-type]
                selected_anchor_sha256=anchor_hash,
            )
            connection.execute(
                f"INSERT INTO {ROUND73_SHOCK_ANCHOR_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                list(anchor.as_row()),
            )
        source_hash = _sha(_canonical_json(sources))
        anchor_rows_hash = _stream_hash(anchor_hashes)
        identity = {
            "schema_version": ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
            "contract_sha256": ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
            "study_id": STUDY_ID,
            "source_runs": sources,
            "source_runs_sha256": source_hash,
            "source_run_count": len(sources),
            "selected_anchor_rows_sha256": anchor_rows_hash,
            "selected_anchor_count": len(anchor_inputs),
            "target_observed": False,
            "model_evaluated": False,
            "trading_authority": False,
        }
        manifest_text = _canonical_json(identity)
        connection.execute(
            f"INSERT INTO {ROUND73_SHOCK_STUDY_TABLE} VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                STUDY_ID,
                ROUND73_SHOCK_STUDY_SCHEMA_VERSION,
                ROUND73_COMPACT_TARGET_CONTRACT_SHA256,
                manifest_text,
                _sha(manifest_text),
                source_hash,
                _sha("identity-rows"),
                anchor_rows_hash,
                len(sources),
                len(anchor_inputs),
                start_wall_ns,
                start_wall_ns + 7 * 86_400_000_000_000,
                start_wall_ns + 8 * 86_400_000_000_000,
            ],
        )
    return corpus_hashes, grid_hashes


def test_v2_target_lifecycle_is_bounded_complete_and_tamper_evident(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "targets-v2.duckdb"
    corpus_hashes, grid_hashes = _seed(database)

    def cohort_audit(*_args: object, **_kwargs: object):
        return SimpleNamespace(passed=True)

    def corpus_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(
            passed=True,
            manifest_sha256=corpus_hashes[run_id],
        )

    def grid_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(
            passed=True,
            build_manifest_sha256=grid_hashes[run_id],
        )

    monkeypatch.setattr(
        "simple_ai_trading.impact_absorption_target_store_v2."
        "parse_round73_target_quantity_rules",
        lambda *_args, **_kwargs: _rules(),
    )
    common = {
        "study_id": STUDY_ID,
        "cohort_audit_function": cohort_audit,
        "corpus_audit_function": corpus_audit,
        "grid_audit_function": grid_audit,
        "replay_function": _replay,
    }
    first = build_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[0],
        **common,
    )
    assert first.selected_anchor_count == 1
    assert first.option_count == 36
    assert first.eligible_option_count == 36
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            f"SELECT DISTINCT horizon_ms FROM {ROUND73_TARGET_V2_OPTION_TABLE} "
            "ORDER BY horizon_ms"
        ).fetchall() == [(15_000,), (60_000,), (300_000,)]
    first_audit = audit_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[0],
        **common,
    )
    assert first_audit.passed
    assert first_audit.deep_replay_performed
    divergent_replay = audit_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[0],
        **{**common, "replay_function": lambda *_args, **_kwargs: []},
    )
    assert not divergent_replay.passed
    assert "exact-wire replay differs" in " ".join(divergent_replay.errors)

    try:
        seal_round73_target_study(database, **common)
    except ValueError as exc:
        assert "failed before seal" in str(exc)
    else:
        raise AssertionError("partial target study must not seal")

    second = build_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[1],
        **common,
    )
    assert second.selected_anchor_count == 0
    assert second.option_count == 0
    second_audit = audit_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[1],
        **common,
    )
    assert second_audit.passed

    orphan_run = "f" * 32
    with duckdb.connect(str(database)) as connection:
        row = list(
            connection.execute(
                f"SELECT * FROM {ROUND73_TARGET_V2_OPTION_TABLE} LIMIT 1"
            ).fetchone()
        )
        row[0] = orphan_run
        connection.execute(
            f"INSERT INTO {ROUND73_TARGET_V2_OPTION_TABLE} VALUES ("
            + ",".join("?" for _ in row)
            + ")",
            row,
        )
    try:
        seal_round73_target_study(database, **common)
    except ValueError as exc:
        assert "orphan option rows" in str(exc)
    else:
        raise AssertionError("orphan v2 options must prevent the study seal")
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            f"DELETE FROM {ROUND73_TARGET_V2_OPTION_TABLE} WHERE run_id = ?",
            [orphan_run],
        )

    sealed = seal_round73_target_study(database, **common)
    assert sealed.source_run_count == 2
    assert sealed.selected_anchor_count == 1
    assert sealed.option_count == 36
    assert audit_round73_target_study(database, **common).passed
    assert (
        build_round73_selected_anchor_targets(
            database,
            run_id=RUN_IDS[0],
            **common,
        )
        == first
    )

    with duckdb.connect(str(database)) as connection:
        connection.execute(
            f"UPDATE {ROUND73_TARGET_V2_OPTION_TABLE} "
            "SET cohort_option_sha256 = ? WHERE study_id = ? AND run_id = ? "
            "AND entry_delay_ms = 500 AND horizon_ms = 15000 "
            "AND reference_quote_notional = 100 AND side = 'long'",
            ["0" * 64, STUDY_ID, RUN_IDS[0]],
        )
    tampered = audit_round73_selected_anchor_targets(
        database,
        run_id=RUN_IDS[0],
        **common,
    )
    assert not tampered.passed
    assert "target row differs" in " ".join(tampered.errors)
    assert not audit_round73_target_study(database, **common).passed


def test_v2_manifest_table_is_per_run(tmp_path: Path) -> None:
    database = tmp_path / "manifest-key.duckdb"
    _seed(database)
    with duckdb.connect(str(database)) as connection:
        from simple_ai_trading.impact_absorption_target_store_v2 import (
            _assert_table_shapes,
            _create_tables,
        )

        _create_tables(connection)
        _assert_table_shapes(connection)
        primary_key_columns = [
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info('{ROUND73_TARGET_V2_MANIFEST_TABLE}')"
            ).fetchall()
            if bool(row[5])
        ]
    assert primary_key_columns == ["study_id", "run_id"]


def test_v2_builder_rejects_the_prospective_eligible_holdout(tmp_path: Path) -> None:
    database = tmp_path / "v2-holdout-block.duckdb"
    _seed(
        database,
        start_wall_ns=ROUND73_STUDY_NOT_BEFORE_WALL_NS,
        utc_day="2026-07-24",
    )

    with pytest.raises(ValueError, match="staged v3 holdout store"):
        build_round73_selected_anchor_targets(
            database,
            study_id=STUDY_ID,
            run_id=RUN_IDS[0],
            verify_cohort=False,
        )


def test_v3_physically_stages_development_pretest_and_one_time_test(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "targets-v3.duckdb"
    corpus_hashes, grid_hashes = _seed(
        database,
        start_wall_ns=ROUND73_STUDY_NOT_BEFORE_WALL_NS,
        utc_day="2026-07-24",
        include_test_anchor=True,
    )

    def cohort_audit(*_args: object, **_kwargs: object):
        return SimpleNamespace(passed=True)

    def corpus_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(passed=True, manifest_sha256=corpus_hashes[run_id])

    def grid_audit(_database: object, *, run_id: str, **_kwargs: object):
        return SimpleNamespace(
            passed=True,
            build_manifest_sha256=grid_hashes[run_id],
        )

    repository_state = {
        "commit_sha": "1" * 40,
        "tree_sha": "2" * 40,
        "clean": True,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }
    monkeypatch.setattr(
        "simple_ai_trading.impact_absorption_target_store_v3."
        "parse_round73_target_quantity_rules",
        lambda *_args, **_kwargs: _rules(),
    )
    common = {
        "study_id": STUDY_ID,
        "cohort_audit_function": cohort_audit,
        "corpus_audit_function": corpus_audit,
        "grid_audit_function": grid_audit,
        "replay_function": _replay,
    }
    first = build_round73_role_targets(
        database,
        run_id=RUN_IDS[0],
        role_scope="development",
        **common,
    )
    second = build_round73_role_targets(
        database,
        run_id=RUN_IDS[1],
        role_scope="development",
        **common,
    )
    assert first.option_count == 36
    assert second.option_count == 0
    progress: list[tuple[str, object]] = []
    staged_development = stage_round73_role_targets(
        database,
        role_scope="development",
        progress_callback=lambda event, details: progress.append((event, details)),
        **common,
    )
    assert staged_development.source_run_count == 2
    assert staged_development.option_count == 36
    assert staged_development.eligible_option_count == first.eligible_option_count
    assert staged_development.positive_option_count == first.positive_option_count
    assert progress[-1][0] == "role_target_stage_completed"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
                "JOIN impact_shock_anchor_v1 a "
                "ON a.study_id = o.study_id AND a.run_id = o.run_id "
                "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
                "WHERE a.role = 'test'"
            ).fetchone()[0]
            == 0
        )
    with pytest.raises(ValueError, match="locked"):
        build_round73_role_targets(
            database,
            run_id=RUN_IDS[1],
            role_scope="test",
            pretest_manifest_sha256="3" * 64,
            **common,
        )

    development = seal_round73_development_targets(database, **common)
    assert development.selected_anchor_count == 1
    development_rows = round73_development_row_identities(
        database,
        study_id=STUDY_ID,
    )
    artifacts = {
        "btcusdt-model.bin": b"frozen model bytes",
        "btcusdt-preprocessor.bin": b"frozen preprocessor bytes",
        "btcusdt-training-predictions.bin": b"frozen training predictions",
        "btcusdt-tuning-predictions.bin": b"frozen tuning predictions",
    }
    model_manifest = {
        "feature_schema": {
            "feature_names": list(ROUND73_ACTION_ALIGNED_FEATURE_NAMES),
            "feature_names_sha256": ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
            "transforms": {"action_alignment": "frozen-v1"},
            "dropped_zero_iqr_columns": {
                symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS
            },
        },
        "row_identities": development_rows,
        "compute_backend": {
            "resolved_backend": "cpu",
            "device_name": "unit-test CPU",
            "platform_name": "unit-test",
            "device_type": "cpu",
            "gpu_accelerated": False,
            "library_versions": {"lightgbm": "test"},
        },
        "symbol_models": {
            "BTCUSDT": {
                "status": "enabled",
                "model_family": "lightgbm",
                "selected_feature_layer": "impact_absorption",
                "best_boosting_iteration": 7,
                "probability_threshold": 0.7,
                "artifact_names": {
                    "model": "btcusdt-model.bin",
                    "preprocessor": "btcusdt-preprocessor.bin",
                    "training_predictions": ("btcusdt-training-predictions.bin"),
                    "tuning_predictions": "btcusdt-tuning-predictions.bin",
                },
            },
            "ETHUSDT": {"status": "disabled", "reason": "unit-test fixture"},
            "SOLUSDT": {"status": "disabled", "reason": "unit-test fixture"},
        },
        "action_policy": {
            "candidate_probability_thresholds": [
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
            ],
            "one_active_position_per_symbol": True,
            "pre_entry_revalidation": True,
            "exact_side_score_tie_policy": "no_trade",
            "profit_reinvestment": False,
            "leverage": 1.0,
        },
    }
    pretest = publish_round73_pretest_manifest(
        database,
        model_manifest=model_manifest,
        artifacts=artifacts,
        repository_root=tmp_path,
        repository_state_function=lambda _root: repository_state,
        **common,
    )
    assert pretest.artifact_count == 4
    with duckdb.connect(str(database), read_only=True) as connection:
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_PRETEST_MODEL_ARTIFACT_TABLE}"
            ).fetchone()[0]
            == 4
        )
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V3_OPTION_TABLE} o "
                "JOIN impact_shock_anchor_v1 a "
                "ON a.study_id = o.study_id AND a.run_id = o.run_id "
                "AND a.symbol = o.symbol AND a.anchor_index = o.anchor_index "
                "WHERE a.role = 'test'"
            ).fetchone()[0]
            == 0
        )

    unlock = unlock_round73_test_targets(
        database,
        study_id=STUDY_ID,
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        repository_root=tmp_path,
        repository_state_function=lambda _root: repository_state,
    )
    assert unlock.pretest_manifest_sha256 == pretest.pretest_manifest_sha256
    with pytest.raises(ValueError, match="already exists"):
        unlock_round73_test_targets(
            database,
            study_id=STUDY_ID,
            pretest_manifest_sha256=pretest.pretest_manifest_sha256,
            repository_root=tmp_path,
            repository_state_function=lambda _root: repository_state,
        )

    test_zero = build_round73_role_targets(
        database,
        run_id=RUN_IDS[0],
        role_scope="test",
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        **common,
    )
    test_rows = build_round73_role_targets(
        database,
        run_id=RUN_IDS[1],
        role_scope="test",
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        **common,
    )
    assert test_zero.option_count == 0
    assert test_rows.option_count == 36
    assert test_rows.eligible_option_count is None
    staged_test = stage_round73_role_targets(
        database,
        role_scope="test",
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        **common,
    )
    assert staged_test.source_run_count == 2
    assert staged_test.option_count == 36
    assert staged_test.eligible_option_count is None
    assert staged_test.positive_option_count is None
    test_study = seal_round73_test_targets(
        database,
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        **common,
    )
    assert test_study.option_count == 36
    assert "eligible_option_count" not in test_study.as_dict()
    with duckdb.connect(str(database), read_only=True) as connection:
        assert (
            connection.execute(
                f"SELECT count(*) FROM {ROUND73_TARGET_V3_TEST_STUDY_TABLE}"
            ).fetchone()[0]
            == 1
        )

    with duckdb.connect(str(database)) as connection:
        connection.execute(
            f"UPDATE {ROUND73_TARGET_V3_OPTION_TABLE} "
            "SET cohort_option_sha256 = ? WHERE study_id = ? AND run_id = ? "
            "AND entry_delay_ms = 500 AND horizon_ms = 15000 "
            "AND reference_quote_notional = 100 AND side = 'long'",
            ["0" * 64, STUDY_ID, RUN_IDS[1]],
        )
    tampered = audit_round73_role_targets(
        database,
        run_id=RUN_IDS[1],
        role_scope="test",
        pretest_manifest_sha256=pretest.pretest_manifest_sha256,
        **common,
    )
    assert not tampered.passed
    assert "role manifest aggregate differs" in " ".join(tampered.errors)
