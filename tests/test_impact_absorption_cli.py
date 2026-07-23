from __future__ import annotations

import argparse
import asyncio
import json

from simple_ai_trading import cli
from simple_ai_trading.command_contract import command_specs, workflow_commands
from simple_ai_trading.impact_absorption_capture import (
    ImpactCaptureReport,
    ImpactCaptureSupervisorReport,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_SCHEMA_VERSION,
    ImpactAbsorptionStore,
)


def _attempt(
    run_id: str = "a" * 32,
    *,
    qualification_passed: bool = False,
) -> ImpactCaptureReport:
    return ImpactCaptureReport(
        run_id=run_id,
        capture_schema_version=IMPACT_CAPTURE_SCHEMA_VERSION,
        mode="probe",
        status="completed",
        capture_gate_passed=False,
        qualification_passed=qualification_passed,
        started_wall_ns=1,
        ended_wall_ns=2,
        elapsed_seconds=1.0,
        queue_high_water_messages=1,
        queue_capacity_messages=65_536,
        queue_maximum_utilization=1 / 65_536,
        writer_frame_count=1,
        writer_message_count=1,
        writer_compressed_payload_bytes=1,
        payload_cap_reached=False,
        database_physical_start_bytes=0,
        database_physical_bytes=1,
        database_physical_growth_bytes=1,
        database_physical_growth_bytes_per_message=1.0,
        database_size_cap_bytes=8 * 1024 * 1024 * 1024,
        database_size_cap_reached=False,
        process_io_scope="capture phase through writer connection close",
        process_io_provider="test",
        process_io_semantics="test counter",
        process_io_start_write_bytes=0,
        process_io_end_write_bytes=1,
        process_io_delta_write_bytes=1,
        process_io_write_bytes_per_message=1.0,
        frames_per_stream_minute=1.0,
        storage_efficiency_passed=False,
        terminal_process_io_provider="test",
        terminal_process_io_semantics="test counter",
        terminal_process_io_start_write_bytes=1,
        terminal_process_io_end_write_bytes=1,
        terminal_process_io_delta_write_bytes=0,
        event_counts={"serverTime": 1},
        symbol_event_counts={},
        negative_corrected_latency_fraction=0.0,
        audit_passed=True,
        audit_errors=(),
        error="",
        failure_class="none",
    )


def _supervisor(*, qualification_passed: bool = False) -> ImpactCaptureSupervisorReport:
    attempt = _attempt(qualification_passed=qualification_passed)
    return ImpactCaptureSupervisorReport(
        status="completed",
        capture_schema_version=IMPACT_CAPTURE_SCHEMA_VERSION,
        qualification_passed=qualification_passed,
        selected_run_id=attempt.run_id,
        attempt_count=1,
        reconnect_count=0,
        reconnect_delays_seconds=(),
        attempts=(attempt,),
        startup_errors=(),
        terminal_error="",
    )


def _capture_args(**overrides) -> argparse.Namespace:
    values = {
        "database": "data/microstructure.duckdb",
        "mode": "probe",
        "duration_seconds": None,
        "compressed_payload_cap_bytes": 2_147_483_648,
        "database_size_cap_bytes": 8 * 1024 * 1024 * 1024,
        "memory_limit": "2GB",
        "database_threads": 2,
        "maximum_reconnects": 6,
        "progress_interval_seconds": 30.0,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_impact_commands_have_parser_and_windows_taxonomy_parity() -> None:
    capture = cli._parse_args(
        [
            "impact-capture",
            "--mode",
            "qualification",
            "--duration-seconds",
            "3660",
            "--maximum-reconnects",
            "2",
        ]
    )
    audit = cli._parse_args(["impact-audit", "--run-id", "a" * 32])
    features = cli._parse_args(["impact-feature-source", "--run-id", "a" * 32])
    corpus_index = cli._parse_args(["impact-corpus-index", "--run-id", "b" * 32])
    grid_build = cli._parse_args(["impact-grid-build", "--run-id", "b" * 32])
    corpus_audit = cli._parse_args(["impact-corpus-audit", "--run-id", "b" * 32])
    grid_audit = cli._parse_args(["impact-grid-audit", "--run-id", "b" * 32])
    corpus_day = cli._parse_args(["impact-corpus-day", "--utc-day", "2026-07-22"])
    corpus_collect = cli._parse_args(["impact-corpus-collect", "--segments", "0"])
    corpus_batch_audit = cli._parse_args(
        ["impact-corpus-batch-audit", "--batch-id", "c" * 32, "--deep"]
    )

    assert capture.duration_seconds == 3660.0
    assert capture.maximum_reconnects == 2
    assert capture.database_size_cap_bytes == 8 * 1024 * 1024 * 1024
    assert audit.run_id == "a" * 32
    assert features.run_id == "a" * 32
    assert corpus_index.run_id == "b" * 32
    assert grid_build.run_id == "b" * 32
    assert corpus_audit.run_id == "b" * 32
    assert grid_audit.run_id == "b" * 32
    assert corpus_day.utc_day == "2026-07-22"
    assert corpus_collect.segments == 0
    assert corpus_batch_audit.batch_id == "c" * 32
    assert corpus_batch_audit.deep is True
    specs = {item.name: item for item in command_specs()}
    assert {
        "impact-capture",
        "impact-audit",
        "impact-feature-source",
        "impact-corpus-index",
        "impact-grid-build",
        "impact-corpus-audit",
        "impact-grid-audit",
        "impact-corpus-day",
        "impact-corpus-collect",
        "impact-corpus-batch-audit",
    } <= set(specs)
    workflow = {item.name: (item.page, item.group) for item in workflow_commands()}
    assert workflow["impact-capture"] == ("Data", "Market data")
    assert workflow["impact-audit"] == ("Data", "Integrity and outcomes")
    assert workflow["impact-feature-source"] == ("Research", "Microstructure models")
    assert workflow["impact-corpus-index"] == ("Research", "Microstructure models")
    assert workflow["impact-grid-build"] == ("Research", "Microstructure models")
    assert workflow["impact-corpus-audit"] == ("Data", "Integrity and outcomes")
    assert workflow["impact-grid-audit"] == ("Data", "Integrity and outcomes")
    assert workflow["impact-corpus-day"] == ("Data", "Integrity and outcomes")
    assert workflow["impact-corpus-collect"] == ("Data", "Market data")
    assert workflow["impact-corpus-batch-audit"] == (
        "Data",
        "Integrity and outcomes",
    )


def test_impact_corpus_handlers_emit_machine_reports(monkeypatch, capsys) -> None:
    class Manifest:
        run_id = "b" * 32
        frame_count = 12
        message_count = 345
        coverage_duration_ns = 3_600_000_000_000

        def as_dict(self):
            return {
                "schema_version": "round-073-segmented-corpus-v2",
                "run_id": self.run_id,
            }

    class Audit:
        run_id = "b" * 32
        passed = True
        errors = ()
        frame_count = 12
        message_count = 345

        def as_dict(self):
            return {
                "schema_version": "round-073-corpus-manifest-audit-v1",
                "passed": True,
            }

    class Day:
        utc_day = "2026-07-22"
        finalized = True
        eligible = False
        coverage_ns = 3_600_000_000_000

        def as_dict(self):
            return {
                "schema_version": "round-073-corpus-day-coverage-v1",
                "crypto_formal_daily_close": False,
            }

    observed = []

    def fake_index(database, **kwargs):
        observed.append(("index", str(database), kwargs))
        return Manifest()

    def fake_audit(database, **kwargs):
        observed.append(("audit", str(database), kwargs))
        return Audit()

    def fake_day(database, **kwargs):
        observed.append(("day", str(database), kwargs))
        return Day()

    monkeypatch.setattr(cli, "index_round73_corpus_run", fake_index)
    monkeypatch.setattr(cli, "audit_round73_corpus_manifest", fake_audit)
    monkeypatch.setattr(cli, "round73_corpus_day_coverage", fake_day)
    common = {
        "database": "corpus.duckdb",
        "memory_limit": "1GB",
        "database_threads": 1,
        "json": True,
    }

    assert (
        cli.command_impact_corpus_index(argparse.Namespace(**common, run_id="b" * 32))
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run_id"] == "b" * 32
    assert (
        cli.command_impact_corpus_audit(argparse.Namespace(**common, run_id="b" * 32))
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert (
        cli.command_impact_corpus_day(
            argparse.Namespace(**common, utc_day="2026-07-22")
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["crypto_formal_daily_close"] is False
    assert observed == [
        (
            "index",
            "corpus.duckdb",
            {"run_id": "b" * 32, "memory_limit": "1GB", "threads": 1},
        ),
        (
            "audit",
            "corpus.duckdb",
            {"run_id": "b" * 32, "memory_limit": "1GB", "threads": 1},
        ),
        (
            "day",
            "corpus.duckdb",
            {"utc_day": "2026-07-22", "memory_limit": "1GB", "threads": 1},
        ),
    ]


def test_impact_grid_handlers_emit_machine_reports(monkeypatch, capsys) -> None:
    class Report:
        run_id = "b" * 32
        anchor_count = 10_620
        valid_anchor_count = 10_000
        vector_count = 10_000

        def as_dict(self):
            return {
                "schema_version": "round-073-causal-grid-v2",
                "run_id": self.run_id,
                "target_constructed": False,
            }

    class Audit:
        run_id = "b" * 32
        passed = True
        errors = ()
        anchor_count = 10_620
        valid_anchor_count = 10_000
        vector_count = 10_000

        def as_dict(self):
            return {
                "schema_version": "round-073-grid-build-audit-v1",
                "passed": True,
            }

    observed = []

    def fake_build(database, **kwargs):
        observed.append(("build", str(database), kwargs))
        return Report()

    def fake_audit(database, **kwargs):
        observed.append(("audit", str(database), kwargs))
        return Audit()

    monkeypatch.setattr(cli, "build_round73_causal_grid", fake_build)
    monkeypatch.setattr(cli, "audit_round73_causal_grid", fake_audit)
    common = {
        "database": "corpus.duckdb",
        "run_id": "b" * 32,
        "memory_limit": "1GB",
        "database_threads": 1,
        "json": True,
    }

    assert cli.command_impact_grid_build(argparse.Namespace(**common)) == 0
    assert json.loads(capsys.readouterr().out)["target_constructed"] is False
    assert cli.command_impact_grid_audit(argparse.Namespace(**common)) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert observed == [
        (
            "build",
            "corpus.duckdb",
            {"run_id": "b" * 32, "memory_limit": "1GB", "threads": 1},
        ),
        (
            "audit",
            "corpus.duckdb",
            {"run_id": "b" * 32, "memory_limit": "1GB", "threads": 1},
        ),
    ]


def test_impact_corpus_collect_handler_uses_bounded_rotation(
    monkeypatch, capsys
) -> None:
    class Report:
        status = "completed"
        batch_id = "d" * 32
        qualified_capture_segment_count = 0
        requested_capture_segments = 0
        recovered_segment_count = 1
        indexed_segment_count = 1
        error = ""

        def as_dict(self):
            return {
                "schema_version": "round-073-rotation-runner-v1",
                "batch_id": self.batch_id,
                "status": self.status,
            }

    observed = {}

    async def fake_run(config, *, progress_interval_seconds):
        observed["config"] = config
        observed["progress_interval_seconds"] = progress_interval_seconds
        return Report()

    monkeypatch.setattr(cli, "_run_impact_corpus_rotation_with_progress", fake_run)
    args = argparse.Namespace(
        database="corpus.duckdb",
        segments=0,
        compressed_payload_cap_bytes=2_147_483_648,
        database_size_cap_bytes=8 * 1024 * 1024 * 1024,
        memory_limit="1GB",
        database_threads=1,
        progress_interval_seconds=30.0,
        json=True,
    )

    assert cli.command_impact_corpus_collect(args) == 0
    assert json.loads(capsys.readouterr().out)["batch_id"] == "d" * 32
    assert observed["config"].segment_count == 0
    assert observed["config"].capture_config().duration_seconds == 3_600
    assert observed["config"].capture_config().maximum_reconnects == 0
    assert observed["progress_interval_seconds"] == 30.0


def test_impact_corpus_collect_progress_monitor_is_nonblocking(
    monkeypatch,
    capsys,
) -> None:
    class Report:
        status = "completed"

    async def delayed_rotation(_config):
        await asyncio.sleep(0.02)
        return Report()

    monkeypatch.setattr(cli, "run_round73_corpus_rotation", delayed_rotation)
    result = asyncio.run(
        cli._run_impact_corpus_rotation_with_progress(
            cli.Round73CorpusRotationConfig(segment_count=0),
            progress_interval_seconds=0.001,
        )
    )

    assert result.status == "completed"
    assert "impact-corpus-collect-progress:" in capsys.readouterr().err


def test_impact_corpus_collect_rejects_silent_progress_interval(capsys) -> None:
    args = argparse.Namespace(progress_interval_seconds=4.9)

    assert cli.command_impact_corpus_collect(args) == 2
    assert "between 5 and 120" in capsys.readouterr().err


def test_impact_corpus_batch_audit_handler_forwards_deep_mode(
    monkeypatch,
    capsys,
) -> None:
    class Audit:
        batch_id = "e" * 32
        passed = True
        status = "completed"
        segment_count = 2
        deeply_audited_manifest_count = 2
        errors = ()

        def as_dict(self):
            return {
                "schema_version": "round-073-rotation-batch-audit-v1",
                "passed": True,
            }

    observed = {}

    def fake_audit(database, **kwargs):
        observed["database"] = str(database)
        observed.update(kwargs)
        return Audit()

    monkeypatch.setattr(cli, "audit_round73_rotation_batch", fake_audit)
    args = argparse.Namespace(
        database="corpus.duckdb",
        batch_id="e" * 32,
        deep=True,
        memory_limit="1GB",
        database_threads=1,
        json=True,
    )

    assert cli.command_impact_corpus_batch_audit(args) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert observed == {
        "database": "corpus.duckdb",
        "batch_id": "e" * 32,
        "deep_manifest_audit": True,
        "memory_limit": "1GB",
        "threads": 1,
    }


def test_impact_feature_source_handler_emits_machine_report(
    monkeypatch, capsys
) -> None:
    class Diagnostic:
        run_id = "a" * 32
        frame_count = 11
        message_count = 101
        depth_update_count = 7
        level_change_count = 23

        def as_dict(self):
            return {
                "schema_version": "round-073-feature-source-diagnostic-v2",
                "run_id": self.run_id,
            }

    observed = {}

    def fake_diagnostic(database, **kwargs):
        observed["database"] = str(database)
        observed.update(kwargs)
        return Diagnostic()

    monkeypatch.setattr(cli, "diagnose_round73_feature_source", fake_diagnostic)
    args = argparse.Namespace(
        database="feature.duckdb",
        run_id="a" * 32,
        memory_limit="1GB",
        database_threads=1,
        json=True,
    )

    assert cli.command_impact_feature_source(args) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "a" * 32
    assert observed == {
        "database": "feature.duckdb",
        "run_id": "a" * 32,
        "memory_limit": "1GB",
        "threads": 1,
    }


def test_impact_capture_handler_uses_mode_default_and_machine_report(
    monkeypatch, capsys
) -> None:
    observed = {}

    async def fake_run(config, *, progress_interval_seconds):
        observed["config"] = config
        observed["progress_interval_seconds"] = progress_interval_seconds
        return _supervisor()

    monkeypatch.setattr(cli, "_run_impact_capture_with_progress", fake_run)

    assert cli.command_impact_capture(_capture_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "round-073-capture-supervisor-report-v1"
    assert payload["attempt_evidence_combined"] is False
    assert observed["config"].duration_seconds == 180.0
    assert observed["config"].mode == "probe"
    assert observed["progress_interval_seconds"] == 30.0


def test_impact_capture_qualification_fails_exit_without_qualification(
    monkeypatch, capsys
) -> None:
    async def fake_run(config, *, progress_interval_seconds):
        assert config.duration_seconds == 3_600.0
        assert progress_interval_seconds == 30.0
        return _supervisor(qualification_passed=False)

    monkeypatch.setattr(cli, "_run_impact_capture_with_progress", fake_run)

    assert cli.command_impact_capture(_capture_args(mode="qualification")) == 2
    assert json.loads(capsys.readouterr().out)["qualification_passed"] is False


def test_impact_capture_rejects_progress_interval_before_start(capsys) -> None:
    assert cli.command_impact_capture(_capture_args(progress_interval_seconds=4.9)) == 2
    assert "between 5 and 120" in capsys.readouterr().err


def test_impact_capture_progress_monitor_reports_without_blocking(
    monkeypatch, capsys
) -> None:
    async def delayed_capture(_config):
        await asyncio.sleep(0.02)
        return _supervisor()

    monkeypatch.setattr(cli, "capture_round73_supervised", delayed_capture)
    result = asyncio.run(
        cli._run_impact_capture_with_progress(
            cli.ImpactCaptureConfig(duration_seconds=1),
            progress_interval_seconds=0.001,
        )
    )

    assert result.status == "completed"
    assert "impact-capture-progress:" in capsys.readouterr().err


def test_impact_audit_selects_latest_terminal_run(tmp_path, capsys) -> None:
    database = tmp_path / "impact.duckdb"
    with ImpactAbsorptionStore(database) as store:
        for index, run_id in enumerate(("b" * 32, "c" * 32), start=1):
            store.start_run(
                run_id=run_id,
                started_wall_ns=index,
                started_monotonic_ns=index,
                config={"mode": "probe"},
            )
            store.finish_run(
                run_id=run_id,
                status="completed",
                ended_wall_ns=index + 10,
            )

    args = argparse.Namespace(
        database=str(database),
        run_id=None,
        memory_limit="1GB",
        database_threads=1,
        json=True,
    )
    assert cli.command_impact_audit(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "c" * 32
    assert payload["passed"] is True
    assert payload["stored_report_sha256"] == ""


def test_impact_audit_missing_run_fails_closed(tmp_path, capsys) -> None:
    database = tmp_path / "impact.duckdb"
    with ImpactAbsorptionStore(database):
        pass
    args = argparse.Namespace(
        database=str(database),
        run_id="d" * 32,
        memory_limit="1GB",
        database_threads=1,
        json=True,
    )

    assert cli.command_impact_audit(args) == 2
    assert "no matching terminal" in capsys.readouterr().err
