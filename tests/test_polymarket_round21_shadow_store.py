from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sqlite3
import zlib

import pytest

import simple_ai_trading.polymarket_round21_shadow_store as shadow_module
from simple_ai_trading.polymarket_round21_shadow_store import (
    Round21ProspectiveShadowStore,
)

from polymarket_round21_support import (
    SHADOW_CONDITION_ID,
    SHADOW_MODEL_SHA,
    SHADOW_SEALED_SHA,
    START_MS,
    round21_shadow_prediction,
    sha,
)


RUN_ID = "1" * 32
CONDITION = SHADOW_CONDITION_ID
MODEL_SHA = SHADOW_MODEL_SHA
SEALED_SHA = SHADOW_SEALED_SHA
_prediction = round21_shadow_prediction


def _start(store: Round21ProspectiveShadowStore) -> None:
    store.start_run(
        run_id=RUN_ID,
        source_model_artifact_sha256=MODEL_SHA,
        sealed_result_sha256=SEALED_SHA,
        population_layer="core",
        started_at_ms=START_MS,
    )


def test_shadow_store_roundtrips_hash_chain_and_restart_audit(tmp_path) -> None:
    path = tmp_path / "round21-shadow.sqlite3"
    with Round21ProspectiveShadowStore(path) as store:
        run = store.start_run(
            run_id=RUN_ID,
            source_model_artifact_sha256=MODEL_SHA,
            sealed_result_sha256=SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS,
        )
        first = store.append_prediction(
            RUN_ID,
            _prediction(),
            recorded_at_ms=START_MS + 1_010,
        )
        second = store.append_prediction(
            RUN_ID,
            _prediction(1_250, observed=False),
            recorded_at_ms=START_MS + 1_260,
        )
        assert first.previous_record_sha256 == run.run_sha256
        assert second.previous_record_sha256 == first.record_sha256
        store.terminate_run(
            RUN_ID,
            status="complete",
            finished_at_ms=START_MS + 2_000,
        )

    with Round21ProspectiveShadowStore(path) as restarted:
        audit = restarted.audit_run(RUN_ID)
        assert restarted.run_ids() == (RUN_ID,)
        predictions = restarted.predictions(RUN_ID)
        assert audit.integrity_passed is True
        assert audit.prediction_count == 2
        assert audit.observed_count == 1
        assert audit.abstention_count == 1
        assert audit.terminal is not None
        assert audit.terminal.status == "complete"
        assert [item.prediction for item in predictions] == [
            _prediction(),
            _prediction(1_250, observed=False),
        ]
        assert not any(
            (
                audit.target_accessed,
                audit.credentials_used,
                audit.account_connected,
                audit.binance_execution_connected,
                audit.grants_execution_authority,
                audit.profitability_claim,
                audit.paper_trading_authority,
                audit.live_trading_authority,
            )
        )


def test_shadow_store_is_idempotent_and_rejects_divergent_duplicate(tmp_path) -> None:
    with Round21ProspectiveShadowStore(tmp_path / "shadow.sqlite3") as store:
        _start(store)
        prediction = _prediction()
        first = store.append_prediction(
            RUN_ID,
            prediction,
            recorded_at_ms=START_MS + 1_010,
        )
        duplicate = store.append_prediction(
            RUN_ID,
            prediction,
            recorded_at_ms=START_MS + 1_020,
        )
        assert duplicate == first
        with pytest.raises(ValueError, match="key is immutable"):
            store.append_prediction(
                RUN_ID,
                _prediction(latency_ns=41),
                recorded_at_ms=START_MS + 1_020,
            )
        with pytest.raises(ValueError, match="run ID is immutable"):
            store.start_run(
                run_id=RUN_ID,
                source_model_artifact_sha256=sha("other"),
                sealed_result_sha256=SEALED_SHA,
                population_layer="core",
                started_at_ms=START_MS,
            )


@pytest.mark.parametrize("status,reason", [("complete", ""), ("interrupted", "stop")])
def test_shadow_terminal_is_immutable_and_blocks_new_records(
    tmp_path,
    status: str,
    reason: str,
) -> None:
    with Round21ProspectiveShadowStore(tmp_path / f"{status}.sqlite3") as store:
        _start(store)
        store.append_prediction(
            RUN_ID,
            _prediction(),
            recorded_at_ms=START_MS + 1_010,
        )
        terminal = store.terminate_run(
            RUN_ID,
            status=status,
            reason=reason,
            finished_at_ms=START_MS + 2_000,
        )
        assert terminal.live_trading_authority is False
        assert (
            store.terminate_run(
                RUN_ID,
                status=status,
                reason=reason,
                finished_at_ms=START_MS + 2_000,
            )
            == terminal
        )
        with pytest.raises(ValueError, match="run is terminal"):
            store.append_prediction(
                RUN_ID,
                _prediction(1_250),
                recorded_at_ms=START_MS + 2_010,
            )
        with pytest.raises(ValueError, match="terminal is immutable"):
            store.terminate_run(
                RUN_ID,
                status="failed",
                reason="different",
                finished_at_ms=START_MS + 2_001,
            )
        assert store.audit_run(RUN_ID).terminal == terminal


def test_shadow_tables_reject_update_delete_and_audit_detects_byte_tamper(
    tmp_path,
) -> None:
    path = tmp_path / "tamper.sqlite3"
    with Round21ProspectiveShadowStore(path) as store:
        _start(store)
        store.append_prediction(
            RUN_ID,
            _prediction(),
            recorded_at_ms=START_MS + 1_010,
        )
        store.terminate_run(
            RUN_ID,
            status="complete",
            finished_at_ms=START_MS + 2_000,
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(  # noqa: SLF001 - verifies database guard
                "UPDATE round21_shadow_prediction SET recorded_at_ms = recorded_at_ms + 1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(  # noqa: SLF001 - verifies database guard
                "DELETE FROM round21_shadow_prediction"
            )

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER round21_shadow_prediction_no_update")
        connection.execute(
            """
            UPDATE round21_shadow_prediction
            SET payload = zeroblob(compressed_byte_count)
            """
        )
    with Round21ProspectiveShadowStore(path) as restarted:
        with pytest.raises(ValueError, match="compressed prediction differs"):
            restarted.audit_run(RUN_ID)


def test_shadow_identity_and_run_validation_fail_closed(tmp_path) -> None:
    symlink_path = tmp_path / "symlink.sqlite3"
    symlink_path.touch()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(symlink_path), "is_symlink", lambda _self: True)
        with pytest.raises(ValueError, match="may not be a symlink"):
            Round21ProspectiveShadowStore(symlink_path)
    with pytest.raises(TypeError, match="wall clock"):
        Round21ProspectiveShadowStore(tmp_path / "clock.sqlite3", wall_time_ms=0)
    with Round21ProspectiveShadowStore(tmp_path / "identity.sqlite3") as store:
        with pytest.raises(ValueError, match="run ID"):
            store.start_run(
                run_id="bad",
                source_model_artifact_sha256=MODEL_SHA,
                sealed_result_sha256=SEALED_SHA,
                population_layer="core",
                started_at_ms=START_MS,
            )
        with pytest.raises(ValueError, match="model artifact digest"):
            store.start_run(
                run_id=RUN_ID,
                source_model_artifact_sha256=hashlib.sha256(b"").hexdigest(),
                sealed_result_sha256=SEALED_SHA,
                population_layer="core",
                started_at_ms=START_MS,
            )
        with pytest.raises(ValueError, match="population layer"):
            store.start_run(
                run_id=RUN_ID,
                source_model_artifact_sha256=MODEL_SHA,
                sealed_result_sha256=SEALED_SHA,
                population_layer="unknown",
                started_at_ms=START_MS,
            )
        with pytest.raises(ValueError, match="start timestamp"):
            store.start_run(
                run_id=RUN_ID,
                source_model_artifact_sha256=MODEL_SHA,
                sealed_result_sha256=SEALED_SHA,
                population_layer="core",
                started_at_ms=True,
            )
        run = store.start_run(
            run_id=RUN_ID,
            source_model_artifact_sha256=MODEL_SHA,
            sealed_result_sha256=SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS,
        )
        assert (
            store.start_run(
                run_id=RUN_ID,
                source_model_artifact_sha256=MODEL_SHA,
                sealed_result_sha256=SEALED_SHA,
                population_layer="core",
                started_at_ms=START_MS,
            )
            == run
        )
        with pytest.raises(ValueError, match="run identity differs"):
            replace(run, live_trading_authority=True).validated()
        with pytest.raises(KeyError, match="unknown Round 21 shadow run"):
            store.predictions("f" * 32)


def test_shadow_append_rejects_wrong_identity_time_and_chronology(tmp_path) -> None:
    path = tmp_path / "append-guards.sqlite3"
    with Round21ProspectiveShadowStore(path) as store:
        _start(store)
        with pytest.raises(TypeError, match="prediction type"):
            store.append_prediction(RUN_ID, object())
        prediction = _prediction()
        with pytest.raises(ValueError, match="precedes observation"):
            store.append_prediction(
                RUN_ID,
                prediction,
                recorded_at_ms=prediction.observed_at_ms - 1,
            )
        store.append_prediction(
            RUN_ID,
            prediction,
            recorded_at_ms=START_MS + 2_000,
        )
        with pytest.raises(ValueError, match="chronology differs"):
            store.append_prediction(
                RUN_ID,
                _prediction(1_250),
                recorded_at_ms=START_MS + 1_500,
            )
        with pytest.raises(ValueError, match="chronology differs"):
            store.append_prediction(
                RUN_ID,
                _prediction(750),
                recorded_at_ms=START_MS + 2_100,
            )
        with pytest.raises(ValueError, match="not terminal"):
            store.audit_run(RUN_ID)

    with Round21ProspectiveShadowStore(tmp_path / "binding.sqlite3") as store:
        store.start_run(
            run_id=RUN_ID,
            source_model_artifact_sha256=sha("different-model"),
            sealed_result_sha256=SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS,
        )
        with pytest.raises(ValueError, match="prediction identity differs"):
            store.append_prediction(
                RUN_ID,
                _prediction(),
                recorded_at_ms=START_MS + 1_010,
            )

    with Round21ProspectiveShadowStore(tmp_path / "pre-run.sqlite3") as store:
        store.start_run(
            run_id=RUN_ID,
            source_model_artifact_sha256=MODEL_SHA,
            sealed_result_sha256=SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS + 2_000,
        )
        with pytest.raises(ValueError, match="prediction precedes run"):
            store.append_prediction(
                RUN_ID,
                _prediction(),
                recorded_at_ms=START_MS + 2_010,
            )


def test_shadow_terminal_validation_rejects_invalid_or_regressed_time(tmp_path) -> None:
    with Round21ProspectiveShadowStore(tmp_path / "terminal-guards.sqlite3") as store:
        _start(store)
        for status, reason in (
            ("passed", ""),
            ("complete", "unexpected"),
            ("failed", ""),
            ("failed", "x" * 513),
        ):
            with pytest.raises(ValueError, match="terminal state"):
                store.terminate_run(
                    RUN_ID,
                    status=status,
                    reason=reason,
                    finished_at_ms=START_MS + 1,
                )
        with pytest.raises(ValueError, match="precedes run"):
            store.terminate_run(
                RUN_ID,
                status="failed",
                reason="clock",
                finished_at_ms=START_MS - 1,
            )
        store.append_prediction(
            RUN_ID,
            _prediction(),
            recorded_at_ms=START_MS + 2_000,
        )
        with pytest.raises(ValueError, match="precedes last record"):
            store.terminate_run(
                RUN_ID,
                status="failed",
                reason="clock",
                finished_at_ms=START_MS + 1_500,
            )


def _decode_raw(raw: bytes):
    payload = zlib.compress(raw)
    return shadow_module._decode_prediction(
        payload,
        raw_byte_count=len(raw),
        payload_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_shadow_decoder_rejects_corrupt_bounded_or_noncanonical_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = _prediction()
    canonical = json.dumps(
        prediction.asdict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    compressed = zlib.compress(canonical)
    with pytest.raises(ValueError, match="compressed prediction differs"):
        shadow_module._decode_prediction(
            compressed,
            raw_byte_count=len(canonical),
            payload_sha256=hashlib.sha256(canonical).hexdigest(),
            compressed_payload_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="compression is invalid"):
        shadow_module._decode_prediction(
            b"not-zlib",
            raw_byte_count=8,
            payload_sha256=sha("raw"),
            compressed_payload_sha256=hashlib.sha256(b"not-zlib").hexdigest(),
        )
    with pytest.raises(ValueError, match="payload differs"):
        shadow_module._decode_prediction(
            compressed + b"trailing",
            raw_byte_count=len(canonical),
            payload_sha256=hashlib.sha256(canonical).hexdigest(),
            compressed_payload_sha256=hashlib.sha256(
                compressed + b"trailing"
            ).hexdigest(),
        )
    with pytest.raises(ValueError, match="JSON is not an object"):
        _decode_raw(b"[]")
    with pytest.raises(ValueError, match="JSON is invalid"):
        _decode_raw(b'{"duplicate":1,"duplicate":2}')
    with pytest.raises(ValueError, match="JSON is invalid"):
        _decode_raw(b'{"value":NaN}')
    pretty = json.dumps(prediction.asdict(), indent=2, sort_keys=True).encode("ascii")
    with pytest.raises(ValueError, match="serialization differs"):
        _decode_raw(pretty)
    oversized = zlib.compress(b"x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="payload differs"):
        shadow_module._decode_prediction(
            oversized,
            raw_byte_count=256 * 1024,
            payload_sha256=sha("oversized"),
            compressed_payload_sha256=hashlib.sha256(oversized).hexdigest(),
        )

    monkeypatch.setattr(shadow_module, "_MAX_RAW_PREDICTION_BYTES", 1)
    with pytest.raises(ValueError, match="payload is too large"):
        shadow_module._compress_prediction(prediction)
    monkeypatch.setattr(shadow_module, "_MAX_RAW_PREDICTION_BYTES", 256 * 1024)
    monkeypatch.setattr(shadow_module, "_MAX_COMPRESSED_PREDICTION_BYTES", 1)
    with pytest.raises(ValueError, match="compressed prediction is too large"):
        shadow_module._compress_prediction(prediction)


def test_shadow_open_rejects_stale_schema_or_table_shape(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.sqlite3"
    with sqlite3.connect(metadata_path) as connection:
        connection.execute(
            "CREATE TABLE round21_shadow_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO round21_shadow_metadata VALUES ('schema_version', 'stale')"
        )
    with pytest.raises(RuntimeError, match="store schema differs"):
        Round21ProspectiveShadowStore(metadata_path)

    shape_path = tmp_path / "shape.sqlite3"
    with sqlite3.connect(shape_path) as connection:
        connection.execute(
            "CREATE TABLE round21_shadow_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute("CREATE TABLE round21_shadow_run (run_id TEXT PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="table schema differs"):
        Round21ProspectiveShadowStore(shape_path)


def _terminal_shadow_database(path) -> None:
    with Round21ProspectiveShadowStore(path) as store:
        _start(store)
        store.append_prediction(
            RUN_ID,
            _prediction(),
            recorded_at_ms=START_MS + 1_010,
        )
        store.terminate_run(
            RUN_ID,
            status="complete",
            finished_at_ms=START_MS + 2_000,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "DROP TRIGGER round21_shadow_run_no_update; "
            "UPDATE round21_shadow_run SET schema_version = 'stale'",
            "run schema differs",
        ),
        (
            "DROP TRIGGER round21_shadow_run_no_update; "
            f"UPDATE round21_shadow_run SET run_sha256 = '{'a' * 64}'",
            "run hash differs",
        ),
        (
            "DROP TRIGGER round21_shadow_prediction_no_update; "
            f"UPDATE round21_shadow_prediction SET condition_id = '0x{'3' * 64}'",
            "prediction record differs",
        ),
        (
            "DROP TRIGGER round21_shadow_terminal_no_update; "
            "UPDATE round21_shadow_terminal SET schema_version = 'stale'",
            "terminal schema differs",
        ),
        (
            "DROP TRIGGER round21_shadow_terminal_no_update; "
            f"UPDATE round21_shadow_terminal SET terminal_sha256 = '{'b' * 64}'",
            "terminal hash differs",
        ),
    ),
)
def test_shadow_audit_rejects_semantic_table_tamper(
    tmp_path, mutation, message
) -> None:
    path = tmp_path / f"tamper-{hashlib.sha256(mutation.encode()).hexdigest()}.sqlite3"
    _terminal_shadow_database(path)
    with sqlite3.connect(path) as connection:
        connection.executescript(mutation)
    with Round21ProspectiveShadowStore(path) as store:
        with pytest.raises(ValueError, match=message):
            store.audit_run(RUN_ID)


def test_shadow_audit_rejects_rehashed_prediction_and_terminal_chain_forks(
    tmp_path,
) -> None:
    prediction_path = tmp_path / "prediction-chain.sqlite3"
    _terminal_shadow_database(prediction_path)
    with Round21ProspectiveShadowStore(prediction_path) as store:
        stored = store.predictions(RUN_ID)[0]
        row = store._connection.execute(  # noqa: SLF001 - builds tamper fixture
            "SELECT * FROM round21_shadow_prediction"
        ).fetchone()
        fork_previous = "c" * 64
        payload = store._prediction_record_payload(  # noqa: SLF001
            run_id=RUN_ID,
            sequence_number=stored.sequence_number,
            prediction=stored.prediction,
            previous_record_sha256=fork_previous,
            payload_sha256=str(row["payload_sha256"]),
            compressed_payload_sha256=str(row["compressed_payload_sha256"]),
            raw_byte_count=int(row["raw_byte_count"]),
            compressed_byte_count=int(row["compressed_byte_count"]),
            recorded_at_ms=stored.recorded_at_ms,
        )
        fork_sha = shadow_module._canonical_sha256(payload)
    with sqlite3.connect(prediction_path) as connection:
        connection.execute("DROP TRIGGER round21_shadow_prediction_no_update")
        connection.execute(
            """
            UPDATE round21_shadow_prediction
            SET previous_record_sha256 = ?, record_sha256 = ?
            """,
            (fork_previous, fork_sha),
        )
    with Round21ProspectiveShadowStore(prediction_path) as store:
        with pytest.raises(ValueError, match="prediction chain differs"):
            store.audit_run(RUN_ID)

    terminal_path = tmp_path / "terminal-chain.sqlite3"
    _terminal_shadow_database(terminal_path)
    fork_previous = "d" * 64
    payload = Round21ProspectiveShadowStore._terminal_payload(
        run_id=RUN_ID,
        sequence_number=2,
        status="complete",
        reason="",
        previous_record_sha256=fork_previous,
        finished_at_ms=START_MS + 2_000,
    )
    fork_sha = shadow_module._canonical_sha256(payload)
    with sqlite3.connect(terminal_path) as connection:
        connection.execute("DROP TRIGGER round21_shadow_terminal_no_update")
        connection.execute(
            """
            UPDATE round21_shadow_terminal
            SET previous_record_sha256 = ?, terminal_sha256 = ?
            """,
            (fork_previous, fork_sha),
        )
    with Round21ProspectiveShadowStore(terminal_path) as store:
        with pytest.raises(ValueError, match="terminal chain differs"):
            store.audit_run(RUN_ID)


def test_shadow_store_has_no_credential_account_or_order_surface(tmp_path) -> None:
    with Round21ProspectiveShadowStore(tmp_path / "surface.sqlite3") as store:
        public_names = {name for name in dir(store) if not name.startswith("_")}
        assert public_names == {
            "account_connected",
            "append_prediction",
            "audit_run",
            "binance_execution_connected",
            "close",
            "credentials_used",
            "grants_execution_authority",
            "path",
            "paper_trading_authority",
            "predictions",
            "run_ids",
            "start_run",
            "terminate_run",
            "trading_authority",
            "live_trading_authority",
        }
        assert not any(
            (
                shadow_module.credentials_used,
                shadow_module.account_connected,
                shadow_module.binance_execution_connected,
                shadow_module.grants_execution_authority,
                shadow_module.profitability_claim,
                shadow_module.paper_trading_authority,
                shadow_module.live_trading_authority,
            )
        )
