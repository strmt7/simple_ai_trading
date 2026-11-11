"""Real enrolled SQLite policy state with no credentials or venue operations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import sqlite3
import os
from pathlib import Path
import sys

import pytest

from simple_ai_trading import cli, entrypoint
from simple_ai_trading.autonomous import AutonomousConfig, _loss_budget_guard
from simple_ai_trading.binance_capital_policy import (
    BinanceCapitalPolicy,
    BinanceCapitalPolicyStore,
)
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.positions import ClosedTrade, PositionsStore
from simple_ai_trading.runtime_control_storage import RuntimeStorageEnrollment
from simple_ai_trading.types import StrategyConfig

SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-policy"
)


@pytest.fixture
def policy_case(tmp_path, enrolled_binance_runtime, monkeypatch):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE, start=False)
    policies = BinanceCapitalPolicyStore(store)
    observed = policies.inspect()
    document = dict(
        schema_version="binance-capital-policy-v1",
        execution_scope_sha256=observed["execution_scope_sha256"],
        execution_store_id=observed["execution_store_id"],
        quote_asset="USDC",
        reference_cash_quote="1000",
        maximum_daily_loss_fraction="0.006",
        maximum_lifetime_loss_fraction="0.012",
        maximum_settled_peak_drawdown_fraction="0.1",
        maximum_consecutive_losses=2,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("capital policy accessed configuration or a venue client")

    monkeypatch.setattr(cli, "load_runtime", forbidden)
    monkeypatch.setattr(cli, "_build_client", forbidden)
    return (
        store,
        authority.control,
        policies,
        BinanceCapitalPolicy(**document),
        observed,
    )


def guard(store, strategy=None, reference=1000):
    return _loss_budget_guard(
        store,
        100,
        strategy or StrategyConfig(),
        AutonomousConfig(dry_run=False, starting_reference_cash=reference),
        now_ms_value=3000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )


def activate(case):
    return case[2].activate(case[3], expected_head_sha256=case[4]["head_sha256"])


def test_inspection_never_initializes_a_policy_and_signed_admission_blocks(policy_case):
    store, control, policies, _, observed = policy_case
    before = store.opening_intents.path.read_bytes()
    assert observed["revision"] == 0 and observed["policy"] is None
    assert not policies.enrollment_path.exists()
    result = guard(store)
    assert not result.allowed and result.reason == "risk-policy-activation-unqualified"
    assert store.opening_intents.path.read_bytes() == before
    assert control.snapshot().state == "stop_requested"


def test_activation_binding_survives_restart_without_rearm(policy_case):
    store, control, policies, policy, _ = policy_case
    stopped = control.snapshot()
    active = activate(policy_case)
    assert active["revision"] == 1 and active["enrollment_state"] == "ready"
    assert active["policy"] == asdict(policy)
    assert BinanceCapitalPolicyStore(PositionsStore(store.root)).inspect() == active
    decision = guard(store)
    assert decision.allowed and decision.capital_policy_sha256 == active["head_sha256"]
    assert control.snapshot() == stopped
    assert (
        policies.activate(policy, expected_head_sha256=active["head_sha256"]) == active
    )


@pytest.mark.parametrize(
    "change", ["reference", "daily", "session", "drawdown", "streak", "unit", "product"]
)
def test_effective_loss_configuration_cannot_silently_change(policy_case, change):
    store, _, policies, _, _ = policy_case
    activate(policy_case)
    strategy, reference = StrategyConfig(), 1000
    if change == "reference":
        reference = 2000
    elif change == "daily":
        strategy.max_daily_loss_pct = 0.005
    elif change == "session":
        strategy.max_session_loss_pct = 0.013
    elif change == "drawdown":
        strategy.max_drawdown_limit = 0.09
    elif change == "streak":
        strategy.max_consecutive_losses = 3
    if change in {"unit", "product"}:
        with pytest.raises(ValueError):
            policies.assert_effective(
                strategy,
                reference,
                symbol="BTCUSDT" if change == "unit" else "BTCUSDC",
                market_type="spot" if change == "unit" else "futures",
            )
    else:
        result = guard(store, strategy, reference)
        assert not result.allowed and result.capital_policy_sha256 is None


def test_policy_revision_does_not_clear_loss_history_or_change_capital_units(
    policy_case,
):
    store, control, policies, policy, _ = policy_case
    active = activate(policy_case)
    for index, pnl in enumerate((200, -150)):
        store.record_close(
            ClosedTrade(
                id=str(index),
                symbol="BTCUSDC",
                market_type="spot",
                side="LONG",
                qty=1,
                entry_price=100,
                exit_price=100,
                leverage=1,
                opened_at_ms=0,
                closed_at_ms=1000 + index,
                realized_pnl=pnl,
                realized_pnl_pct=pnl / 1000,
            )
        )
    before = store.load_snapshot(strict=True)
    new = replace(policy, maximum_daily_loss_fraction="0.007")
    updated = policies.activate(new, expected_head_sha256=active["head_sha256"])
    assert updated["revision"] == 2
    decision = guard(store, replace(StrategyConfig(), max_daily_loss_pct=0.007))
    assert not decision.allowed and decision.reason.startswith("drawdown-lockout")
    assert decision.realized_peak_drawdown == pytest.approx(0.15)
    assert store.load_snapshot(strict=True) == before
    for changed in [
        replace(new, reference_cash_quote="2000"),
        replace(new, quote_asset="USDT"),
    ]:
        with pytest.raises(ValueError, match="migration"):
            policies.activate(changed, expected_head_sha256=updated["head_sha256"])
    assert control.snapshot().state == "stop_requested"


def test_activation_compare_and_swap_and_ownerless_stop(policy_case):
    _, control, policies, policy, observed = policy_case
    with pytest.raises(ValueError, match="binding"):
        policies.activate(policy, expected_head_sha256="0" * 64)
    assert control.complete_stale_stop(exposure_closed=True)
    control.acquire()
    control.request_stop(reason="synthetic unresolved owner")
    with pytest.raises(RuntimeError, match="ownerless"):
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])
    assert not policies.enrollment_path.exists()


@pytest.mark.parametrize(
    "damage", ["identity", "head", "history", "all", "marker", "hash"]
)
def test_policy_damage_never_bootstraps_or_rearms(policy_case, damage):
    store, control, policies, policy, observed = policy_case
    activate(policy_case)
    if damage == "marker":
        policies.enrollment_path.unlink()
    else:
        with sqlite3.connect(store.opening_intents.path) as connection:
            if damage == "all":
                for table in [
                    "capital_policy_identity",
                    "capital_policy_head",
                    "capital_policy_history",
                ]:
                    connection.execute(f"DROP TABLE {table}")
            elif damage == "hash":
                connection.execute(
                    "UPDATE capital_policy_history SET entry_sha256=?", ("0" * 64,)
                )
            else:
                connection.execute(f"DELETE FROM capital_policy_{damage}")
    assert not guard(store).allowed
    with pytest.raises((ValueError, OSError)):
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])
    assert control.snapshot().state == "stop_requested"


def test_first_commit_interruption_has_explicit_exact_recovery(
    policy_case, monkeypatch
):
    _, control, policies, policy, observed = policy_case
    original = RuntimeStorageEnrollment.finish

    def interrupted(*args):
        raise OSError("PRIVATE-SENTINEL")

    monkeypatch.setattr(RuntimeStorageEnrollment, "finish", interrupted)
    with pytest.raises(OSError):
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])
    with pytest.raises(ValueError):
        policies.inspect()
    pending = policies.inspect(allow_prepared=True)
    assert pending["enrollment_state"] == "prepared"
    monkeypatch.setattr(RuntimeStorageEnrollment, "finish", original)
    with pytest.raises(ValueError):
        policies.recover(
            expected_store_id=pending["policy_store_id"],
            expected_record_sha256="0" * 64,
        )
    active = policies.recover(
        expected_store_id=pending["policy_store_id"],
        expected_record_sha256=pending["enrollment_record_sha256"],
    )
    assert active["enrollment_state"] == "ready" and active["revision"] == 1
    assert control.snapshot().state == "stop_requested"


def test_rolled_back_first_transaction_cannot_be_promoted(policy_case, monkeypatch):
    from simple_ai_trading import binance_capital_policy as module

    _, _, policies, policy, observed = policy_case
    original = module._sha

    def failed(value):
        if isinstance(value, list) and len(value) == 3 and value[0] == 1:
            raise OSError("synthetic failure before SQLite commit")
        return original(value)

    monkeypatch.setattr(module, "_sha", failed)
    with pytest.raises(OSError):
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])
    monkeypatch.setattr(module, "_sha", original)
    assert policies.enrollment_path.exists()
    with pytest.raises(ValueError):
        policies.inspect(allow_prepared=True)
    with pytest.raises(ValueError):
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])


def test_concurrent_first_activations_have_one_winner(policy_case):
    _, _, policies, policy, observed = policy_case

    def attempt():
        try:
            return policies.activate(
                policy, expected_head_sha256=observed["head_sha256"]
            )["revision"]
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: attempt(), range(2)), key=str) == [
            1,
            "rejected",
        ]
    assert policies.inspect()["revision"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unknown"),
        ("maximum_consecutive_losses", True),
        ("maximum_consecutive_losses", 1.5),
        ("reference_cash_quote", "0"),
        ("reference_cash_quote", "NaN"),
        ("reference_cash_quote", "1e999999"),
        ("maximum_daily_loss_fraction", 0.01),
        ("maximum_daily_loss_fraction", "0"),
        ("maximum_daily_loss_fraction", "2"),
        ("quote_asset", "private/path"),
        ("execution_scope_sha256", "invalid"),
    ],
)
def test_policy_schema_is_strict(policy_case, field, value):
    payload = asdict(policy_case[3])
    payload[field] = value
    with pytest.raises(ValueError):
        BinanceCapitalPolicy.parse(json.dumps(payload).encode())


def test_document_unknown_duplicate_and_oversized_fields_reject(policy_case):
    payload = asdict(policy_case[3])
    payload["unknown"] = True
    for raw in [
        json.dumps(payload).encode(),
        b'{"schema_version":"a","schema_version":"b"}',
        b" " * 8193,
        b"[]",
    ]:
        with pytest.raises(ValueError):
            BinanceCapitalPolicy.parse(raw)


def test_real_installed_parser_operator_flow_and_native_registration(
    policy_case, tmp_path, capsys
):
    from simple_ai_trading.command_contract import (
        command_contract_digest,
        workflow_commands,
    )
    from pathlib import Path

    store, _, policies, policy, _ = policy_case
    argv = ["capital-policy", "--data-dir", str(store.root), "--json"]
    assert entrypoint.main(argv) == 0
    observed = json.loads(capsys.readouterr().out)["result"]
    document = tmp_path / "private policy.json"
    document.write_text(json.dumps(asdict(policy)), encoding="utf-8")
    assert (
        entrypoint.main(
            argv
            + [
                "--action",
                "activate",
                "--policy",
                str(document),
                "--expected-head-sha256",
                observed["head_sha256"],
            ]
        )
        == 0
    )
    active = json.loads(capsys.readouterr().out)
    assert active["activation_outcome"] == "active"
    assert (
        active["account_queried"]
        is active["order_submitted"]
        is active["rearmed"]
        is False
    )
    assert [
        (x.page, x.group) for x in workflow_commands() if x.name == "capital-policy"
    ] == [("Risk", "Owned close recovery")]
    header = (
        Path(__file__).resolve().parents[1]
        / "native/windows/generated/command_contract.hpp"
    ).read_text(encoding="utf-8")
    assert command_contract_digest() in header
    assert policies.inspect()["revision"] == 1


def test_fresh_contained_process_reads_policy_without_credentials(policy_case):
    from tools.cocoindex_agent_search import run_captured_process

    store = policy_case[0]
    active = activate(policy_case)
    root = Path(__file__).resolve().parents[1]
    script = """import json,socket,sys
def forbidden(*a,**kw): raise AssertionError('unexpected network or credentials')
socket.socket.connect=forbidden
from pathlib import Path
from simple_ai_trading import cli
cli.load_runtime=cli._build_client=forbidden
from simple_ai_trading.binance_capital_policy import BinanceCapitalPolicyStore
from simple_ai_trading.positions import PositionsStore
print(json.dumps(BinanceCapitalPolicyStore(PositionsStore(Path(sys.argv[1]))).inspect()))
"""
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
        [sys.executable, "-B", "-c", script, str(store.root)],
        cwd=root,
        env=env,
        input_text="",
        timeout=30,
        inherit_environment=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == active


def test_update_transaction_failure_preserves_the_approved_head(
    policy_case, monkeypatch
):
    from simple_ai_trading import binance_capital_policy as module

    policies = policy_case[2]
    active = activate(policy_case)
    changed = replace(policy_case[3], maximum_daily_loss_fraction="0.007")
    original = module._sha

    def unavailable(value):
        if isinstance(value, list) and len(value) == 3 and value[0] == 2:
            raise OSError("synthetic update interruption")
        return original(value)

    monkeypatch.setattr(module, "_sha", unavailable)
    with pytest.raises(OSError):
        policies.activate(changed, expected_head_sha256=active["head_sha256"])
    monkeypatch.setattr(module, "_sha", original)
    assert policies.inspect() == active


def test_operator_postcommit_error_remains_uncertain(
    policy_case, tmp_path, capsys, monkeypatch
):
    store, _, policies, policy, observed = policy_case
    original = BinanceCapitalPolicyStore.activate

    def interrupted(self, *a, **kw):
        original(self, *a, **kw)
        raise OSError("PRIVATE-SENTINEL")

    monkeypatch.setattr(BinanceCapitalPolicyStore, "activate", interrupted)
    document = tmp_path / "private-policy.json"
    document.write_text(json.dumps(asdict(policy)), encoding="utf-8")
    assert (
        entrypoint.main(
            [
                "capital-policy",
                "--data-dir",
                str(store.root),
                "--action",
                "activate",
                "--policy",
                str(document),
                "--expected-head-sha256",
                observed["head_sha256"],
                "--json",
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert not result.out and "PRIVATE-SENTINEL" not in result.err
    assert json.loads(result.err)["activation_outcome"] == "unknown"
    assert policies.inspect()["revision"] == 1


@pytest.mark.parametrize("value", [0, "false", None])
def test_execution_mode_must_be_an_explicit_boolean(tmp_path, value):
    result = _loss_budget_guard(
        PositionsStore(tmp_path),
        None,
        StrategyConfig(),
        AutonomousConfig(dry_run=value),
        now_ms_value=3000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert (
        not result.allowed and result.reason == "risk-policy-execution-mode-unqualified"
    )


def test_missing_policy_does_not_mask_known_cash_loss_reduction(policy_case):
    store = policy_case[0]
    for index, pnl in enumerate((200, -150)):
        store.record_close(
            ClosedTrade(
                id=str(index),
                symbol="BTCUSDC",
                market_type="spot",
                side="LONG",
                qty=1,
                entry_price=100,
                exit_price=100,
                leverage=1,
                opened_at_ms=0,
                closed_at_ms=1000 + index,
                realized_pnl=pnl,
                realized_pnl_pct=pnl / 1000,
            )
        )
    result = guard(store)
    assert not result.allowed and result.force_close
    assert (
        result.reason.startswith("drawdown-lockout")
        and result.capital_policy_sha256 is None
    )


def test_operator_missing_storage_is_noncreating(tmp_path, capsys):
    root = tmp_path / "missing owned storage"
    assert entrypoint.main(["capital-policy", "--data-dir", str(root), "--json"]) == 2
    output = capsys.readouterr()
    assert not output.out and json.loads(output.err)["status"] == "blocked"
    assert not root.exists()


def test_oversized_history_rejects_before_document_parsing(policy_case, monkeypatch):
    store, _, policies, _, _ = policy_case
    activate(policy_case)
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute(
            "UPDATE capital_policy_history SET policy_json=?", ("x" * 8193,)
        )
    monkeypatch.setattr(
        BinanceCapitalPolicy,
        "parse",
        lambda *_: pytest.fail("oversized document was parsed"),
    )
    with pytest.raises(ValueError, match="budget"):
        policies.inspect()
