"""Real operator flow with synthetic native responses and no account access."""

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import cli
from simple_ai_trading import binance_close_application_cli as operator
from simple_ai_trading import binance_recovery_reads as reads
from simple_ai_trading.binance_open_intents import OpenIntentError
from test_binance_close_application import CLIENT_ID, stage
from test_binance_recovery_reads import native_reads as native_reads


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for name in (
        "BINANCE_BASE_URL",
        "BINANCE_SPOT_BASE_URL",
        "BINANCE_FUTURES_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("implicit credentials or ordinary client construction")

    monkeypatch.setattr(cli, "load_runtime", forbidden)
    monkeypatch.setattr(cli, "_build_client", forbidden)


def pending(tmp_path, enroll, **options):
    store, control, opened = stage(tmp_path, enroll, **options)
    with sqlite3.connect(store.opening_intents.path) as connection:
        row = connection.execute(
            "SELECT order_json, trades_json FROM closing_recovery WHERE client_id=?",
            (CLIENT_ID,),
        ).fetchone()
        connection.execute(
            "DELETE FROM closing_recovery WHERE client_id=?", (CLIENT_ID,)
        )
    return store, control, opened, json.loads(row[0]), json.loads(row[1])


def args(store, **changes):
    result = dict(data_dir=store.root, client_order_id=CLIENT_ID, action="inspect")
    result.update(changes)
    return argparse.Namespace(**result)


def selection(store, **changes):
    code, inspected = operator.run_close_recovery(args(store))
    assert code == 0
    return args(
        store,
        action="collect",
        expected_scope_sha256=inspected["execution_scope_sha256"],
        expected_request_sha256=inspected["request_sha256"],
        **changes,
    )


def runtime(**changes):
    values = dict(
        market_type="spot",
        testnet=True,
        demo=False,
        api_key="offline-application",
        api_secret="offline-secret",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_pending_inspection_needs_no_terminal_record_or_credentials(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    monkeypatch.setattr(
        operator,
        "_configured_credentials",
        lambda *_: pytest.fail("loaded credentials"),
    )
    status, result = operator.run_close_recovery(args(store))
    assert status == 0 and result["status"] == "inspected"
    assert result["local_intent_state"] == "UNKNOWN"
    assert result["recovery_sha256"] is None and result["retained_evidence"] is False
    assert result["application_recorded"] is False
    assert result["account_queried"] is False
    assert len(result["request_sha256"]) == 64 and result["symbol"] == "BTCUSDT"
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize("quantity", ["0", "0.6", "1"])
def test_real_cli_inspect_collect_cache_and_apply(
    tmp_path, enrolled_binance_runtime, native_reads, monkeypatch, capsys, quantity
):
    store, control, _, order, trades = pending(
        tmp_path, enrolled_binance_runtime, quantity=quantity
    )
    marker = native_reads[0](order, trades, api_key="offline-application")
    argv = [
        "recover-close",
        "--data-dir",
        str(store.root),
        "--client-order-id",
        CLIENT_ID,
        "--json",
    ]
    assert cli.main(argv) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["retained_evidence"] is False
    calls = []

    def load():
        calls.append("configured")
        assert control.snapshot().state == "stop_requested"
        return runtime()

    monkeypatch.setattr(cli, "load_runtime", load)
    collect = argv + [
        "--action",
        "collect",
        "--expected-scope-sha256",
        inspected["execution_scope_sha256"],
        "--expected-request-sha256",
        inspected["request_sha256"],
    ]
    before, stopped = store.load_snapshot(strict=True), control.snapshot()
    assert cli.main(collect) == 0
    collected = json.loads(capsys.readouterr().out)
    assert collected["collection_outcome"] == "retained"
    assert collected["account_queried"] is True
    assert collected["order_read_state"] == "response_received"
    assert collected["trades_read_state"] == (
        "not_started" if quantity == "0" else "response_received"
    )
    assert collected["application_outcome"] == "not_requested"
    assert store.load_snapshot(strict=True) == before
    assert control.snapshot() == stopped
    assert len(marker.read_text().splitlines()) == (1 if quantity == "0" else 2)

    monkeypatch.setattr(
        cli, "load_runtime", lambda: pytest.fail("cached command loaded credentials")
    )
    assert cli.main(collect) == 0
    cached = json.loads(capsys.readouterr().out)
    assert (
        cached["collection_outcome"] == "cached" and cached["account_queried"] is False
    )
    assert cached["order_read_state"] == cached["trades_read_state"] == "not_started"
    assert calls == ["configured"] and len(native_reads[1]) == 1

    apply = argv + [
        "--action",
        "apply",
        "--expected-scope-sha256",
        collected["execution_scope_sha256"],
        "--expected-recovery-sha256",
        collected["recovery_sha256"],
    ]
    assert cli.main(apply) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["application_outcome"] == "applied"
    assert (
        applied["account_queried"]
        is applied["order_submitted"]
        is applied["rearmed"]
        is False
    )
    assert applied["result"]["executed_quantity"] == quantity
    assert control.snapshot().state == "stop_requested"
    assert cli.main(apply) == 0
    assert json.loads(capsys.readouterr().out) == applied


@pytest.mark.parametrize(
    "failure", ["digest", "scope", "owner", "enrollment", "timeout"]
)
def test_local_admission_precedes_private_configuration(
    tmp_path, enrolled_binance_runtime, native_reads, failure
):
    store, control, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, api_key="offline-application")
    selected = selection(store)
    if failure == "digest":
        selected.expected_request_sha256 = "0" * 64
    elif failure == "scope":
        selected.expected_scope_sha256 = "0" * 64
    elif failure == "owner":
        assert control.complete_stale_stop(exposure_closed=True)
        control.acquire()
        control.request_stop(reason="synthetic owner remains unresolved")
    elif failure == "enrollment":
        store.opening_intents.path.with_name(
            store.opening_intents.path.name + ".binance_execution.enrollment.json"
        ).unlink()
    else:
        selected.read_timeout_seconds = float("nan")
    status, result = operator.run_close_recovery(
        selected,
        load_runtime_fn=lambda: pytest.fail(
            "loaded credentials before local admission"
        ),
    )
    assert status == 2 and result["account_queried"] is False
    assert not native_reads[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key": "rotated"},
        {"market_type": "futures"},
        {"testnet": False},
        {"demo": True},
        {"api_secret": ""},
    ],
)
def test_changed_current_configuration_cannot_query_original_scope(
    tmp_path, enrolled_binance_runtime, native_reads, changes
):
    store, _, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, api_key="offline-application")
    status, result = operator.run_close_recovery(
        selection(store), load_runtime_fn=lambda: runtime(**changes)
    )
    assert status == 2 and result["account_queried"] is False
    assert not native_reads[1]


@pytest.mark.parametrize("mode", ["stall-order", "stall-trades", "exit", "reject"])
def test_cli_faults_preserve_read_and_retention_uncertainty(
    tmp_path, enrolled_binance_runtime, native_reads, mode
):
    store, control, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, mode, api_key="offline-application")
    status, result = operator.run_close_recovery(
        selection(store, read_timeout_seconds=3), load_runtime_fn=runtime
    )
    assert status == 2 and result["collection_outcome"] == "unknown"
    assert result["account_queried"] is (True if mode == "stall-trades" else None)
    assert result["retained_evidence"] is None
    assert "PRIVATE-SENTINEL" not in json.dumps(result)
    assert control.snapshot().state == "stop_requested"
    assert len(native_reads[1]) == 1


def test_post_retention_failure_does_not_claim_absence_or_apply(
    tmp_path, enrolled_binance_runtime, native_reads, monkeypatch
):
    store, control, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, api_key="offline-application")
    real = operator.collect_stopped_closing_recovery

    def interrupted(*a, **kw):
        real(*a, **kw)
        raise OpenIntentError("collection completion is uncertain")

    monkeypatch.setattr(operator, "collect_stopped_closing_recovery", interrupted)
    status, result = operator.run_close_recovery(
        selection(store), load_runtime_fn=runtime
    )
    assert status == 2 and result["account_queried"] is True
    assert (
        result["collection_outcome"] == "unknown"
        and result["retained_evidence"] is None
    )
    assert result["application_outcome"] == "not_requested"
    assert operator.run_close_recovery(args(store))[1]["retained_evidence"] is True
    assert control.snapshot().state == "stop_requested"


@pytest.mark.parametrize("mutation", ["request", "enrollment"])
def test_private_configuration_step_cannot_bypass_fresh_local_binding(
    tmp_path, enrolled_binance_runtime, native_reads, mutation
):
    store, _, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, api_key="offline-application")

    def changed():
        if mutation == "request":
            with sqlite3.connect(store.opening_intents.path) as connection:
                connection.execute(
                    "UPDATE close_intent SET request_json=request_json || ' '"
                )
        else:
            store.opening_intents.path.with_name(
                store.opening_intents.path.name + ".binance_execution.enrollment.json"
            ).unlink()
        return runtime()

    status, result = operator.run_close_recovery(
        selection(store), load_runtime_fn=changed
    )
    assert status == 2 and result["account_queried"] is False
    assert not native_reads[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_scope_sha256": None},
        {"expected_request_sha256": None},
        {"expected_recovery_sha256": "0" * 64},
        {"read_timeout_seconds": 0},
        {"read_timeout_seconds": 61},
        {"read_timeout_seconds": True},
    ],
)
def test_collect_rejects_incomplete_or_incompatible_options(
    tmp_path, enrolled_binance_runtime, changes
):
    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    selected = selection(store)
    for key, value in changes.items():
        setattr(selected, key, value)
    status, result = operator.run_close_recovery(
        selected,
        load_runtime_fn=lambda: pytest.fail("invalid options loaded credentials"),
    )
    assert status == 2 and result["collection_outcome"] == "not_started"
    assert result["account_queried"] is False


@pytest.mark.parametrize("failure", ["exception", "override"])
def test_private_configuration_failures_never_echo_sensitive_values(
    tmp_path, enrolled_binance_runtime, native_reads, monkeypatch, failure
):
    store, _, _, order, trades = pending(tmp_path, enrolled_binance_runtime)
    native_reads[0](order, trades, api_key="offline-application")

    def load():
        if failure == "exception":
            raise RuntimeError("PRIVATE-SENTINEL")
        return runtime()

    if failure == "override":
        monkeypatch.setenv(
            "BINANCE_SPOT_BASE_URL", "https://private.invalid/PRIVATE-SENTINEL"
        )
    status, result = operator.run_close_recovery(selection(store), load_runtime_fn=load)
    assert status == 2 and result["account_queried"] is False
    assert "PRIVATE-SENTINEL" not in json.dumps(result)
    assert not native_reads[1]


@pytest.mark.parametrize(
    "progress", [object(), reads.RecoveryReadProgress(order_state="attempted")]
)
def test_api_rejects_invalid_or_reused_progress_before_loading_credentials(
    tmp_path, enrolled_binance_runtime, progress
):
    store, control, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    selected = selection(store)
    scope, _ = operator._original_scope(store)
    with pytest.raises(OpenIntentError, match="progress"):
        reads.collect_stopped_closing_recovery(
            store,
            scope=scope,
            client_id=CLIENT_ID,
            expected_request_sha256=selected.expected_request_sha256,
            control=control,
            progress=progress,
            credential_provider=lambda: pytest.fail(
                "invalid progress loaded credentials"
            ),
        )


def test_provider_must_return_one_explicit_credential_pair(
    tmp_path, enrolled_binance_runtime
):
    store, control, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    scope, _ = operator._original_scope(store)
    with pytest.raises(OpenIntentError, match="credentials are unavailable"):
        reads.collect_stopped_closing_recovery(
            store,
            scope=scope,
            client_id=CLIENT_ID,
            expected_request_sha256=selection(store).expected_request_sha256,
            control=control,
            credential_provider=lambda: ["PRIVATE-SENTINEL"],
        )
    with pytest.raises(OpenIntentError, match="one explicit"):
        reads.collect_stopped_closing_recovery(
            store,
            scope=scope,
            client_id=CLIENT_ID,
            expected_request_sha256=selection(store).expected_request_sha256,
            control=control,
            credential_provider=runtime,
            api_key="synthetic",
        )


@pytest.mark.parametrize("digest", ["0" * 64, "", 1, "A" * 64])
def test_core_collection_binds_request_before_any_order_read(
    tmp_path, enrolled_binance_runtime, digest
):
    from simple_ai_trading.binance_close_recovery import collect_closing_recovery

    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    scope, _ = operator._original_scope(store)
    client = reads._RetainedScope(scope)
    with pytest.raises(OpenIntentError, match="request"):
        collect_closing_recovery(
            client, store, client_id=CLIENT_ID, expected_request_sha256=digest
        )


def test_cached_record_disappearing_cannot_load_credentials_or_start_a_reader(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    real = reads.collect_closing_recovery

    def lost(client, *a, **kw):
        with sqlite3.connect(store.opening_intents.path) as connection:
            connection.execute("DELETE FROM closing_recovery")
        return real(client, *a, **kw)

    monkeypatch.setattr(reads, "collect_closing_recovery", lost)
    code, result = operator.run_close_recovery(
        selection(store),
        load_runtime_fn=lambda: pytest.fail("cached record accessed credentials"),
    )
    assert code == 2 and result["account_queried"] is False
    assert result["collection_outcome"] == "unknown"


def test_generated_native_options_bind_the_real_parser():
    from simple_ai_trading.command_contract import (
        command_contract_digest,
        command_specs,
    )

    spec = next(item for item in command_specs() if item.name == "recover-close")
    action = next(item for item in spec.options if "--action" in item.flags)
    assert set(action.choices) == {"inspect", "collect", "apply"}
    assert {"--expected-request-sha256", "--read-timeout-seconds"} <= {
        flag for item in spec.options for flag in item.flags
    }
    header = (
        Path(__file__).resolve().parents[1]
        / "native/windows/generated/command_contract.hpp"
    ).read_text(encoding="utf-8")
    assert command_contract_digest() in header
    assert 'L"inspect, collect, apply"' in header


def test_plain_cli_read_failure_reports_uncertainty_without_error_payload(
    tmp_path, enrolled_binance_runtime, monkeypatch, capsys
):
    from simple_ai_trading.api import BinanceAPIError

    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    selected = selection(store)

    def failed(*a, **kw):
        kw["progress"].order_state = "attempted"
        raise BinanceAPIError("PRIVATE-SENTINEL")

    monkeypatch.setattr(operator, "collect_stopped_closing_recovery", failed)
    assert (
        cli.main(
            [
                "recover-close",
                "--data-dir",
                str(store.root),
                "--client-order-id",
                CLIENT_ID,
                "--action",
                "collect",
                "--expected-scope-sha256",
                selected.expected_scope_sha256,
                "--expected-request-sha256",
                selected.expected_request_sha256,
            ]
        )
        == 2
    )
    text = capsys.readouterr()
    assert not text.out and "order_read=attempted" in text.err
    assert "PRIVATE-SENTINEL" not in text.err


@pytest.mark.parametrize("action", ["collect", "apply"])
def test_control_errors_are_operator_blocks_with_private_diagnostics_hidden(
    tmp_path, enrolled_binance_runtime, monkeypatch, action
):
    from simple_ai_trading.binance_runtime_control import RuntimeControlError

    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    _, inspected = operator.run_close_recovery(args(store))

    def denied(*a, **kw):
        raise RuntimeControlError("PRIVATE-SENTINEL")

    if action == "collect":
        selected = selection(store)
        monkeypatch.setattr(operator, "collect_stopped_closing_recovery", denied)
    else:
        selected = args(
            store,
            action="apply",
            expected_scope_sha256=inspected["execution_scope_sha256"],
            expected_recovery_sha256=inspected["recovery_sha256"],
        )
        monkeypatch.setattr(operator, "apply_spot_closing_recovery", denied)
    code, result = operator.run_close_recovery(selected)
    assert code == 2 and result["status"] == "blocked"
    assert result["account_queried"] is False
    assert "PRIVATE-SENTINEL" not in json.dumps(result)


def test_application_receipt_without_terminal_source_is_not_pending_admission(
    tmp_path, enrolled_binance_runtime
):
    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    _, inspected = operator.run_close_recovery(args(store))
    code, _ = operator.run_close_recovery(
        args(
            store,
            action="apply",
            expected_scope_sha256=inspected["execution_scope_sha256"],
            expected_recovery_sha256=inspected["recovery_sha256"],
        )
    )
    assert code == 0
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE close_intent SET state='UNKNOWN'")
        connection.execute("DELETE FROM closing_recovery")
    code, result = operator.run_close_recovery(args(store))
    assert code == 2 and "lacks terminal evidence" in result["message"]


def test_no_read_adapter_cannot_query_a_changed_cached_record(
    tmp_path, enrolled_binance_runtime
):
    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    scope, _ = operator._original_scope(store)
    client = reads._RetainedScope(scope)
    with pytest.raises(OpenIntentError, match="cached evidence changed"):
        client.get_order_trades("BTCUSDT", order_id="2", expected_scope=scope)


def test_collection_without_retained_evidence_is_not_reported_successful(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, _, _, _, _ = pending(tmp_path, enrolled_binance_runtime)
    monkeypatch.setattr(
        operator, "collect_stopped_closing_recovery", lambda *a, **kw: None
    )
    code, result = operator.run_close_recovery(selection(store))
    assert code == 2 and result["collection_outcome"] == "unknown"
    assert result["retained_evidence"] is None and result["account_queried"] is False
