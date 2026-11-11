"""Local recovery commands must not construct a venue client or reinterpret scope."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from simple_ai_trading import cli
from simple_ai_trading import binance_close_application_cli as module
from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from test_binance_close_application import CLIENT_ID, stage


def arguments(store, **changes):
    values = dict(data_dir=store.root, client_order_id=CLIENT_ID, action="inspect")
    values.update(changes)
    return argparse.Namespace(**values)


def selected_apply(store, inspection, **changes):
    return arguments(
        store,
        action="apply",
        expected_scope_sha256=inspection["execution_scope_sha256"],
        expected_recovery_sha256=inspection["recovery_sha256"],
        **changes,
    )


@pytest.fixture(autouse=True)
def forbid_credentials_and_client_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local recovery accessed runtime credentials or a client")

    monkeypatch.setattr(cli, "load_runtime", forbidden)
    monkeypatch.setattr(cli, "_build_client", forbidden)


def test_default_inspection_is_read_only_and_application_checks_are_not_claimed(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    monkeypatch.setattr(
        module,
        "apply_spot_closing_recovery",
        lambda *a, **kw: pytest.fail("inspection applied inventory"),
    )
    status, result = module.run_close_recovery(arguments(store))
    assert status == 0 and result["status"] == "inspected"
    assert result["application_admission_checked"] is False
    assert result["application_outcome"] == "not_requested"
    assert (
        result["account_queried"]
        is result["order_submitted"]
        is result["rearmed"]
        is False
    )
    assert len(result["execution_scope_sha256"]) == len(result["recovery_sha256"]) == 64
    assert "credential_fingerprint" not in json.dumps(result)
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == before


def test_inspect_then_apply_and_repeat_uses_original_scope(
    tmp_path, enrolled_binance_runtime
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    _, inspected = module.run_close_recovery(arguments(store))
    stopped = control.snapshot()
    status, result = module.run_close_recovery(selected_apply(store, inspected))
    assert status == 0 and result["application_outcome"] == "applied"
    assert result["result"]["remaining_quantity"] == "0.4"
    assert store.load_open(strict=True)[0].qty == 0.4
    assert len(store.load_ledger(strict=True)) == 1
    assert control.snapshot() == stopped
    assert module.run_close_recovery(selected_apply(store, inspected))[1] == result


@pytest.mark.parametrize(
    "which,value",
    [
        ("expected_scope_sha256", None),
        ("expected_recovery_sha256", None),
        ("expected_scope_sha256", "0" * 64),
        ("expected_recovery_sha256", "0" * 64),
    ],
)
def test_missing_or_changed_digests_cannot_reach_application(
    tmp_path, enrolled_binance_runtime, monkeypatch, which, value
):
    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    _, inspected = module.run_close_recovery(arguments(store))
    args = selected_apply(store, inspected)
    setattr(args, which, value)
    before = store.load_snapshot(strict=True)
    monkeypatch.setattr(
        module,
        "apply_spot_closing_recovery",
        lambda *a, **kw: pytest.fail("digest gate bypassed"),
    )
    status, result = module.run_close_recovery(args)
    assert status == 2 and result["application_outcome"] == "not_started"
    assert store.load_snapshot(strict=True) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"action": "unknown"},
        {"client_order_id": "foreign"},
        {"data_dir": None},
        {"expected_scope_sha256": "0" * 64},
    ],
)
def test_invalid_local_selection_rejects_without_creating_storage(tmp_path, changes):
    from simple_ai_trading.positions import PositionsStore

    root = tmp_path / "not-created"
    status, result = module.run_close_recovery(
        arguments(PositionsStore(root), **changes)
    )
    assert status == 2 and result["status"] == "blocked"
    assert not root.exists()


@pytest.mark.parametrize(
    "encoded", ["not-json", "[]", "{}", '{"private-path":"PRIVATE-SENTINEL"}']
)
def test_malformed_scope_values_are_not_reflected_into_output(
    tmp_path, enrolled_binance_runtime, encoded
):
    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE execution_scope SET scope_json=?", (encoded,))
    status, result = module.run_close_recovery(arguments(store))
    assert status == 2
    assert "PRIVATE-SENTINEL" not in json.dumps(result)


def test_os_error_text_and_paths_are_not_reflected(tmp_path, monkeypatch):
    def inaccessible(*args):
        raise OSError("PRIVATE-SENTINEL")

    monkeypatch.setattr(module, "_original_scope", inaccessible)
    status, result = module.run_close_recovery(
        argparse.Namespace(
            action="inspect", client_order_id=CLIENT_ID, data_dir=tmp_path
        )
    )
    assert status == 2 and "PRIVATE-SENTINEL" not in json.dumps(result)


def test_custom_enrollments_and_directory_with_spaces_are_respected(tmp_path):
    from types import SimpleNamespace

    root = tmp_path / "owned evidence"
    root.mkdir()

    def enroll(store, scope, **options):
        control = BinanceRuntimeControl(
            root / "selected-control.sqlite3",
            enrollment_path=root / "selected-control.json",
        )
        control.provision_storage()
        BinanceExecutionStorage(
            store.opening_intents, control, root / "selected-execution.json"
        ).provision(scope=scope)
        return SimpleNamespace(control=control)

    store, control, _ = stage(root, enroll, prefix=True)
    _, inspected = module.run_close_recovery(arguments(store))
    status, result = module.run_close_recovery(
        selected_apply(
            store,
            inspected,
            runtime_control=control.ledger_path,
            runtime_enrollment=control.enrollment_path,
            execution_enrollment=root / "selected-execution.json",
        )
    )
    assert status == 0 and result["status"] == "applied"
    assert not (root / "binance_runtime_control.sqlite3").exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_real_parser_and_handler_inspect_and_apply(
    tmp_path, enrolled_binance_runtime, capsys, json_output
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    argv = [
        "recover-close",
        "--data-dir",
        str(store.root),
        "--client-order-id",
        CLIENT_ID,
    ]
    if json_output:
        argv.append("--json")
    assert cli.main(argv) == 0
    observed = capsys.readouterr()
    assert not observed.err
    _, inspected = module.run_close_recovery(arguments(store))
    if json_output:
        assert json.loads(observed.out) == inspected
    else:
        assert "application checks remain pending" in observed.out
        assert "local_intent=UNKNOWN application_receipt=absent" in observed.out
    assert (
        cli.main(
            argv
            + [
                "--action",
                "apply",
                "--expected-scope-sha256",
                inspected["execution_scope_sha256"],
                "--expected-recovery-sha256",
                inspected["recovery_sha256"],
            ]
        )
        == 0
    )
    observed = capsys.readouterr()
    if json_output:
        assert json.loads(observed.out)["result"]["remaining_quantity"] == "0.4"
    else:
        assert "terminal=CANCELED executed=0.6 remaining=0.4" in observed.out
    assert control.snapshot().state == "stop_requested"


def test_cli_application_error_preserves_postcommit_uncertainty(
    tmp_path, enrolled_binance_runtime, monkeypatch, capsys
):
    store, _, _ = stage(tmp_path, enrolled_binance_runtime)
    _, inspected = module.run_close_recovery(arguments(store))

    def uncertain(*args, **kwargs):
        raise module.OpenIntentError(
            "closing application could not complete; inspect retained state"
        )

    monkeypatch.setattr(module, "apply_spot_closing_recovery", uncertain)
    assert (
        cli.main(
            [
                "recover-close",
                "--action",
                "apply",
                "--data-dir",
                str(store.root),
                "--client-order-id",
                CLIENT_ID,
                "--expected-scope-sha256",
                inspected["execution_scope_sha256"],
                "--expected-recovery-sha256",
                inspected["recovery_sha256"],
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert not output.out and "may have committed" in output.err
    status, result = module.run_close_recovery(selected_apply(store, inspected))
    assert status == 2 and result["application_outcome"] == "unknown"


def test_workflow_exposes_recovery_under_risk():
    from simple_ai_trading.command_contract import workflow_commands

    assert [
        (item.page, item.group)
        for item in workflow_commands()
        if item.name == "recover-close"
    ] == [("Risk", "Owned close recovery")]


def test_inspection_reports_committed_application_without_publishing_pending_redo(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    from simple_ai_trading import position_transactions

    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    _, inspected = module.run_close_recovery(arguments(store))
    original = position_transactions._recover

    def interrupt(connection, root):
        row = connection.execute(
            "SELECT state FROM position_replacement WHERE id=1"
        ).fetchone()
        if row == ("PREPARED",):
            raise OSError("synthetic interruption after durable commit")
        return original(connection, root)

    monkeypatch.setattr(position_transactions, "_recover", interrupt)
    status, uncertain = module.run_close_recovery(selected_apply(store, inspected))
    assert status == 2 and uncertain["application_outcome"] == "unknown"
    before = (store.open_path.read_bytes(), store.ledger_path.read_bytes())
    status, committed = module.run_close_recovery(arguments(store))
    assert status == 0 and committed["application_recorded"] is True
    assert committed["local_intent_state"] == "RECORDED"
    assert "projections were not inspected" in committed["message"]
    assert (store.open_path.read_bytes(), store.ledger_path.read_bytes()) == before
    monkeypatch.setattr(position_transactions, "_recover", original)
    assert module.run_close_recovery(selected_apply(store, inspected))[0] == 0
    assert store.load_open(strict=True)[0].qty == 0.4


def test_actual_cli_subprocess_is_offline_without_credential_configuration(
    tmp_path, enrolled_binance_runtime
):
    from tools.cocoindex_agent_search import run_captured_process

    store, _, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    script = tmp_path / "offline_operator_cli.py"
    script.write_text(
        """import socket
import sys
def forbidden(*args, **kwargs):
    raise AssertionError("local operator attempted credentials, client construction or a network connection")
socket.socket.connect = forbidden
from simple_ai_trading import cli
cli.load_runtime = forbidden
cli._build_client = forbidden
raise SystemExit(cli.main(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(
        PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = run_captured_process(
        [
            sys.executable,
            "-B",
            str(script),
            "recover-close",
            "--data-dir",
            str(store.root),
            "--client-order-id",
            CLIENT_ID,
            "--json",
        ],
        cwd=root,
        env=env,
        input_text="",
        timeout=25,
        inherit_environment=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert (
        payload["status"] == "inspected"
        and payload["application_outcome"] == "not_requested"
    )
    assert payload["account_queried"] is payload["order_submitted"] is False
    apply_argv = [
        sys.executable,
        "-B",
        str(script),
        "recover-close",
        "--data-dir",
        str(store.root),
        "--client-order-id",
        CLIENT_ID,
        "--json",
        "--action",
        "apply",
        "--expected-scope-sha256",
        payload["execution_scope_sha256"],
        "--expected-recovery-sha256",
        payload["recovery_sha256"],
    ]
    applied = run_captured_process(
        apply_argv,
        cwd=root,
        env=env,
        input_text="",
        timeout=25,
        inherit_environment=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert json.loads(applied.stdout)["result"]["remaining_quantity"] == "0.4"
    repeated = run_captured_process(
        apply_argv,
        cwd=root,
        env=env,
        input_text="",
        timeout=25,
        inherit_environment=False,
    )
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout) == json.loads(applied.stdout)
    assert len(store.load_ledger(strict=True)) == 1


def test_required_parser_arguments_reject_before_any_local_recovery(
    capsys, monkeypatch
):
    monkeypatch.setattr(
        module,
        "run_close_recovery",
        lambda *_: pytest.fail("parser bypassed required selection"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["recover-close"])
    assert error.value.code == 2
    assert "required" in capsys.readouterr().err
