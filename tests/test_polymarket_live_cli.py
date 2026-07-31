from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib
from decimal import Decimal

import pytest

from simple_ai_trading import entrypoint
from simple_ai_trading import polymarket_live_cli as live_cli
from simple_ai_trading.polymarket_autonomous_runtime import (
    PolymarketAutonomousRuntimeSnapshot,
)
from simple_ai_trading.polymarket_historical_shadow_feed import (
    PolymarketShadowFeedHealth,
)
from simple_ai_trading.polymarket_live import (
    PolymarketCancelResult,
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
    PolymarketRedemptionRecord,
    PolymarketReconciliation,
)
from simple_ai_trading.polymarket_live_runtime import PolymarketRuntimeSnapshot
from simple_ai_trading.polymarket_live_v2 import OfficialPolymarketV2Venue
from simple_ai_trading.polymarket_runtime_control import PolymarketRuntimeControl


def _args(*values: str) -> argparse.Namespace:
    return entrypoint._parse_args(["polymarket-live", *values])


def _reconciliation(
    *,
    ok: bool = True,
    can_open: bool = True,
    can_close: bool = True,
    errors: tuple[str, ...] = (),
) -> PolymarketReconciliation:
    return PolymarketReconciliation(
        ok=ok,
        can_open=can_open,
        can_close=can_close,
        foreign_order_ids=(),
        foreign_position_token_ids=(),
        missing_position_token_ids=(),
        blocking_intent_ids=(),
        errors=errors,
    )


def test_entrypoint_registers_independent_polymarket_live_command() -> None:
    args = _args()

    assert args.action == "status"
    assert args.risk_level == "conservative"
    assert args.automatic_redemption is False
    assert args.stop_timeout_seconds == 30
    assert args.promotion is None
    assert args.evidence_root is None
    assert args.requested_quantity == Decimal("5")
    assert args.disable_binance_bbo_safeguard is False
    assert args.func is live_cli.command_polymarket_live


def test_installed_cli_native_app_and_contract_share_entrypoint() -> None:
    root = Path(entrypoint.__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    native = (root / "native" / "windows" / "src" / "main.cpp").read_text(
        encoding="utf-8"
    )
    contract = (root / "src" / "simple_ai_trading" / "command_contract.py").read_text(
        encoding="utf-8"
    )

    assert (
        project["project"]["scripts"]["simple-ai-trading"]
        == "simple_ai_trading.entrypoint:main"
    )
    assert "-m simple_ai_trading.entrypoint" in native
    assert 'L"polymarket-live --action stop"' in native
    assert "from .entrypoint import _build_parser" in contract


def test_entrypoint_fails_closed_without_command_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entrypoint.cli, "_build_parser", argparse.ArgumentParser)

    with pytest.raises(RuntimeError, match="no command registry"):
        entrypoint._build_parser()


def test_entrypoint_main_preserves_menu_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        entrypoint.cli,
        "command_menu",
        lambda _args: 4,
    )
    assert entrypoint.main([]) == 4

    monkeypatch.setattr(
        entrypoint,
        "_parse_args",
        lambda _argv: argparse.Namespace(func=lambda _args: 7),
    )
    assert entrypoint.main(["anything"]) == 7

    monkeypatch.setattr(entrypoint.sys, "argv", ["simple-ai-trading", "status"])
    assert entrypoint.main(None) == 7


def test_local_status_does_not_create_missing_ledger(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "missing.sqlite3"

    assert (
        live_cli.command_polymarket_live(
            _args("--action", "status", "--ledger", str(ledger), "--json")
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "polymarket"
    assert payload["symbol"] == "BTC"
    assert payload["can_open"] is False
    assert payload["ledger_exists"] is False
    assert not ledger.exists()


def test_local_status_audits_existing_empty_ledger(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "live.sqlite3"
    PolymarketLiveOrderLedger(path).records()

    assert (
        live_cli.command_polymarket_live(
            _args("--action", "status", "--ledger", str(path))
        )
        == 0
    )

    rendered = capsys.readouterr().out
    assert "ledger_exists=True" in rendered
    assert "owned_position_quantity=0" in rendered
    assert "can_open=False" in rendered


def test_authenticated_action_fails_without_credentials_and_never_echoes_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "never-print-this-secret"

    def fail_credentials():
        raise ValueError("missing Polymarket live environment variables")

    monkeypatch.setattr(live_cli, "_credentials_and_venue", fail_credentials)

    assert live_cli.command_polymarket_live(_args("--action", "preflight")) == 2

    rendered = capsys.readouterr().err
    assert "missing Polymarket live environment variables" in rendered
    assert secret not in rendered


def test_preflight_closes_venue_and_returns_gate_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(closed=False)
    venue.close = lambda: setattr(venue, "closed", True)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )

    class Coordinator:
        def __init__(self, *_args, **_kwargs):
            pass

        def preflight(self):
            return _reconciliation()

    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "preflight",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    assert result == 0
    assert venue.closed is True
    assert '"can_open": true' in capsys.readouterr().out


def test_reconcile_propagates_failed_gate_and_text_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )

    class Coordinator:
        def __init__(self, *_args, **_kwargs):
            pass

        def reconcile(self):
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=True,
                errors=("unknown_order_state",),
            )

    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "reconcile",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
        )
    )

    rendered = capsys.readouterr().out
    assert result == 2
    assert "reconciliation: ok=False can_open=False can_close=True" in rendered
    assert "unknown_order_state" in rendered


def test_cancel_owned_refuses_foreign_state_before_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    calls: list[str] = []

    class Coordinator:
        def __init__(self, *_args, **_kwargs):
            pass

        def preflight(self):
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=False,
                errors=("foreign_positions",),
            )

        def cancel_owned_open_orders(self):
            calls.append("cancel")
            return PolymarketCancelResult((), ())

    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "cancel-owned",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
        )
    )

    assert result == 2
    assert calls == []
    assert "foreign_positions" in capsys.readouterr().err


def test_cancel_owned_reports_exact_result_and_final_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    order_id = "0x" + "1" * 64

    class Coordinator:
        def __init__(self, *_args, **_kwargs):
            pass

        def preflight(self):
            return _reconciliation()

        def cancel_owned_open_orders(self):
            return PolymarketCancelResult((order_id,), ())

        def reconcile(self):
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=True,
                errors=("unknown_order_state",),
            )

    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "cancel-owned",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload["cancelled_order_ids"] == [order_id]
    assert payload["reconciliation"]["errors"] == ["unknown_order_state"]


def test_stop_owned_exposure_closes_only_local_inventory() -> None:
    order_id = "0x" + "1" * 64
    inventory = [
        SimpleNamespace(
            market_id="0x" + "2" * 64,
            token_id="2" * 40,
            quantity=Decimal("5"),
            provisional=False,
        )
    ]

    class Ledger:
        def owned_inventory(self):
            return tuple(inventory)

        def open_owned_order_ids(self):
            return ()

    class Coordinator:
        def cancel_owned_open_orders(self):
            return PolymarketCancelResult((order_id,), ())

        def reconcile(self):
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=True,
                errors=("foreign_positions",),
            )

        def submit_owned_close_orders(self, *, maximum_book_age_ms: int):
            assert maximum_book_age_ms == 1_500
            inventory.clear()
            return (
                SimpleNamespace(
                    intent=SimpleNamespace(intent_id="stop-close-test-intent")
                ),
            )

    payload, status = live_cli._stop_owned_exposure(
        coordinator=Coordinator(),  # type: ignore[arg-type]
        ledger=Ledger(),  # type: ignore[arg-type]
        timeout_seconds=2,
    )

    assert status == 0
    assert payload["completed"] is True
    assert payload["initial_owned_quantity"] == "5"
    assert payload["remaining_owned_quantity"] == "0"
    assert payload["cancelled_order_ids"] == [order_id]
    assert payload["submitted_close_intent_ids"] == ["stop-close-test-intent"]
    assert payload["foreign_state_untouched"] is True


def test_stop_owned_exposure_reports_blocked_inventory_without_claiming_close() -> None:
    inventory = (
        SimpleNamespace(
            market_id="0x" + "2" * 64,
            token_id="2" * 40,
            quantity=Decimal("4.9"),
            provisional=False,
        ),
    )
    cancelled = False

    class Ledger:
        def owned_inventory(self):
            return inventory

        def open_owned_order_ids(self):
            return ()

    class Coordinator:
        def cancel_owned_open_orders(self):
            nonlocal cancelled
            cancelled = True
            return PolymarketCancelResult((), ())

        def reconcile(self):
            return _reconciliation()

        def submit_owned_close_orders(self, *, maximum_book_age_ms: int):
            del maximum_book_age_ms
            raise PolymarketLiveBlocked(
                "bot-owned close quantity is below the venue minimum"
            )

    payload, status = live_cli._stop_owned_exposure(
        coordinator=Coordinator(),  # type: ignore[arg-type]
        ledger=Ledger(),  # type: ignore[arg-type]
        timeout_seconds=2,
    )

    assert cancelled is True
    assert status == 2
    assert payload["completed"] is False
    assert payload["remaining_owned_quantity"] == "4.9"
    assert "below the venue minimum" in str(payload["reason"])


def test_stop_owned_exposure_validates_timeout_and_reports_no_close() -> None:
    empty_ledger = SimpleNamespace(
        owned_inventory=lambda: (),
        open_owned_order_ids=lambda: (),
    )
    with pytest.raises(ValueError, match="stop-timeout-seconds"):
        live_cli._stop_owned_exposure(
            coordinator=SimpleNamespace(),  # type: ignore[arg-type]
            ledger=empty_ledger,  # type: ignore[arg-type]
            timeout_seconds=0,
        )

    inventory = (
        SimpleNamespace(
            market_id="0x" + "2" * 64,
            token_id="2" * 40,
            quantity=Decimal("5"),
            provisional=False,
        ),
    )
    ledger = SimpleNamespace(
        owned_inventory=lambda: inventory,
        open_owned_order_ids=lambda: (),
    )
    coordinator = SimpleNamespace(
        cancel_owned_open_orders=lambda: PolymarketCancelResult((), ()),
        reconcile=lambda: _reconciliation(),
        submit_owned_close_orders=lambda **_kwargs: (),
    )

    payload, status = live_cli._stop_owned_exposure(
        coordinator=coordinator,  # type: ignore[arg-type]
        ledger=ledger,  # type: ignore[arg-type]
        timeout_seconds=2,
    )

    assert status == 2
    assert payload["reason"] == "owned_inventory_has_no_executable_close"


def test_stop_owned_exposure_polls_open_order_then_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = (
        SimpleNamespace(
            market_id="0x" + "2" * 64,
            token_id="2" * 40,
            quantity=Decimal("5"),
            provisional=False,
        ),
    )
    ledger = SimpleNamespace(
        owned_inventory=lambda: inventory,
        open_owned_order_ids=lambda: ("0x" + "3" * 64,),
    )
    coordinator = SimpleNamespace(
        cancel_owned_open_orders=lambda: PolymarketCancelResult((), ()),
        reconcile=lambda: _reconciliation(),
    )
    clock = iter((0.0, 0.0, 0.0, 2.0, 2.0, 2.0))
    sleeps: list[float] = []
    monkeypatch.setattr(live_cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(live_cli.time, "sleep", sleeps.append)

    payload, status = live_cli._stop_owned_exposure(
        coordinator=coordinator,  # type: ignore[arg-type]
        ledger=ledger,  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    assert status == 2
    assert payload["reason"] == "stop_timeout"
    assert sleeps == [0.25]


def test_stop_owned_exposure_detects_concurrent_inventory_after_empty_snapshot() -> (
    None
):
    item = SimpleNamespace(
        market_id="0x" + "2" * 64,
        token_id="2" * 40,
        quantity=Decimal("5"),
        provisional=False,
    )
    snapshots = iter(((), (), (item,), (item,)))

    class Ledger:
        def owned_inventory(self):
            return next(snapshots)

        def open_owned_order_ids(self):
            return ()

    coordinator = SimpleNamespace(
        cancel_owned_open_orders=lambda: PolymarketCancelResult((), ()),
        reconcile=lambda: _reconciliation(),
    )

    payload, status = live_cli._stop_owned_exposure(
        coordinator=coordinator,  # type: ignore[arg-type]
        ledger=Ledger(),  # type: ignore[arg-type]
        timeout_seconds=2,
    )

    assert status == 2
    assert payload["reason"] == "owned_exposure_remains"


def test_stop_command_dispatches_bounded_owned_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    coordinator = SimpleNamespace()
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: coordinator,
    )
    called: dict[str, object] = {}

    def stop_owned_exposure(**kwargs):
        called.update(kwargs)
        return (
            {
                "schema_version": "polymarket-live-stop-result-v1",
                "action": "stop",
                "completed": True,
            },
            0,
        )

    monkeypatch.setattr(live_cli, "_stop_owned_exposure", stop_owned_exposure)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "stop",
            "--stop-timeout-seconds",
            "12",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    assert result == 0
    assert called["coordinator"] is coordinator
    assert called["timeout_seconds"] == 12
    assert json.loads(capsys.readouterr().out)["completed"] is True


def test_stop_latch_survives_missing_live_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "live.sqlite3"
    control = PolymarketRuntimeControl(path)
    control.acquire(owner_process_id=101)

    def fail_credentials():
        raise ValueError("missing Polymarket live environment variables")

    monkeypatch.setattr(live_cli, "_credentials_and_venue", fail_credentials)

    result = live_cli.command_polymarket_live(
        _args("--action", "stop", "--ledger", str(path), "--json")
    )

    assert result == 2
    assert control.snapshot().state == "stop_requested"


def test_redeem_requires_confirmation_before_settlement_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    monkeypatch.setattr(
        live_cli,
        "_settlement_coordinator",
        lambda **_kwargs: pytest.fail("settlement client must not be created"),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "redeem",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
        )
    )

    assert result == 2
    assert "--confirm-redemption" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (("confirmed", "failed"), 0),
        (("submitted",), 2),
        (("unknown",), 2),
    ],
)
def test_recover_redemptions_reports_unresolved_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    states: tuple[str, ...],
    expected: int,
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    settlement = SimpleNamespace(closed=False)
    settlement.close = lambda: setattr(settlement, "closed", True)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    records = tuple(
        PolymarketRedemptionRecord(
            redemption_id=f"redemption:{index}",
            condition_id="0x" + "1" * 64,
            attempt=index + 1,
            inventory=(),
            preflight_json="{}",
            state=state,
            transaction_id=f"transaction-{index}",
            transaction_hash="",
            failure_code="",
            created_at_ms=1,
            updated_at_ms=2,
        )
        for index, state in enumerate(states)
    )
    redemption = SimpleNamespace(recover_incomplete=lambda: records)
    monkeypatch.setattr(
        live_cli,
        "_settlement_coordinator",
        lambda **_kwargs: (redemption, settlement),
    )

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "recover-redemptions",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == expected
    assert [item["state"] for item in payload["redemptions"]] == list(states)
    assert settlement.closed is True


@pytest.mark.parametrize("has_record", [False, True])
def test_confirmed_redeem_passes_exact_condition_and_closes_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    has_record: bool,
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    settlement = SimpleNamespace(closed=False)
    settlement.close = lambda: setattr(settlement, "closed", True)
    condition = "0x" + "2" * 64
    observed: list[str | None] = []
    record = (
        PolymarketRedemptionRecord(
            redemption_id="redemption:confirmed",
            condition_id=condition,
            attempt=1,
            inventory=(),
            preflight_json="{}",
            state="confirmed",
            transaction_id="transaction-1",
            transaction_hash="0x" + "3" * 64,
            failure_code="",
            created_at_ms=1,
            updated_at_ms=2,
        )
        if has_record
        else None
    )
    redemption = SimpleNamespace(
        redeem_next_ready=lambda *, condition_id: (
            observed.append(condition_id) or record
        )
    )
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        live_cli,
        "_settlement_coordinator",
        lambda **_kwargs: (redemption, settlement),
    )

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "redeem",
            "--condition-id",
            condition,
            "--confirm-redemption",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert observed == [condition]
    assert (payload["redemption"] is not None) is has_record
    assert settlement.closed is True


def test_credential_and_settlement_factories_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = SimpleNamespace(signature_type=0)
    venue = SimpleNamespace()
    monkeypatch.setattr(
        live_cli.PolymarketLiveCredentials,
        "from_environment",
        lambda: credentials,
    )
    monkeypatch.setattr(
        live_cli,
        "OfficialPolymarketV2Venue",
        lambda observed: venue if observed is credentials else pytest.fail(),
    )

    assert live_cli._credentials_and_venue() == (credentials, venue)
    assert live_cli._gasless_credentials(credentials) is None

    smart = SimpleNamespace(signature_type=3)
    gasless = SimpleNamespace()
    monkeypatch.setattr(
        live_cli.PolymarketGaslessCredentials,
        "from_environment",
        lambda: gasless,
    )
    assert live_cli._gasless_credentials(smart) is gasless

    settlement_venue = SimpleNamespace()
    redemption = SimpleNamespace()
    monkeypatch.setattr(
        live_cli,
        "OfficialPolymarketUnifiedRedemptionVenue",
        lambda observed, *, gasless_credentials: (
            settlement_venue
            if observed is smart and gasless_credentials is gasless
            else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketRedemptionCoordinator",
        lambda account, observed_venue, ledger: (
            redemption
            if account is venue
            and observed_venue is settlement_venue
            and ledger == "ledger"
            else pytest.fail()
        ),
    )

    assert live_cli._settlement_coordinator(
        credentials=smart,
        account=venue,
        ledger="ledger",  # type: ignore[arg-type]
    ) == (redemption, settlement_venue)


def test_v2_venue_close_releases_sdk_and_http_session() -> None:
    events: list[str] = []
    venue = object.__new__(OfficialPolymarketV2Venue)
    venue._client = SimpleNamespace(close=lambda: events.append("client"))
    venue.session = SimpleNamespace(close=lambda: events.append("session"))

    venue.close()

    assert events == ["client", "session"]


def test_polymarket_operator_module_has_no_binance_execution_dependency() -> None:
    source = Path(live_cli.__file__).read_text(encoding="utf-8")

    assert "BinanceClient" not in source
    assert "from .api import" not in source
    assert "from .autonomous import" not in source
    assert "Binance orders, balances, positions, or risk state" in source
    assert "PolymarketLiveCoordinator" in source
    assert "PolymarketAuthenticatedUserStream" in source
    assert "PolymarketSettlementService" in source


def test_supervision_argument_bounds_fail_before_network_tasks() -> None:
    with pytest.raises(ValueError, match="duration-seconds"):
        asyncio.run(
            live_cli._supervise(
                credentials=SimpleNamespace(),
                venue=SimpleNamespace(),
                ledger=SimpleNamespace(),
                risk_limits=live_cli._risk_limits("conservative"),
                duration_seconds=-1,
                reconciliation_seconds=5,
                automatic_redemption=False,
            )
        )


def test_supervision_rejects_interval_and_failed_initial_gate() -> None:
    with pytest.raises(ValueError, match="reconciliation-seconds"):
        asyncio.run(
            live_cli._supervise(
                credentials=SimpleNamespace(),
                venue=SimpleNamespace(),
                ledger=SimpleNamespace(),
                risk_limits=live_cli._risk_limits("regular"),
                duration_seconds=1,
                reconciliation_seconds=0,
                automatic_redemption=False,
            )
        )


def test_supervision_runs_independent_services_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clean = _reconciliation()

    class Guard:
        def __init__(self, **kwargs):
            events.append(f"guard:{kwargs['maximum_reconciliation_age_ms']}")
            self.stopped = False

        def mark_stopped(self):
            self.stopped = True
            events.append("guard-stopped")

        def snapshot(self):
            return PolymarketRuntimeSnapshot(
                stream_connected=False,
                stream_age_ms=None,
                reconciliation_age_ms=1,
                reconciliation_can_open=False,
                reconciliation_can_close=True,
                soft_fault="runtime_stopped",
                hard_faults=(),
                stopped=self.stopped,
            )

    class Coordinator:
        def __init__(self, _venue, _ledger, **kwargs):
            assert isinstance(kwargs["runtime_authority"], Guard)

        def preflight(self):
            events.append("preflight")
            return clean

        def reconcile(self):
            events.append("reconcile")
            return clean

    class WaitService:
        def __init__(self, name: str):
            self.name = name

        async def run(self, stop):
            events.append(f"{self.name}-start")
            await stop.wait()
            events.append(f"{self.name}-stop")

    settlement_venue = SimpleNamespace(close=lambda: events.append("settlement-close"))
    ledger = SimpleNamespace(
        records=lambda: (
            SimpleNamespace(intent=SimpleNamespace(market_id="0x" + "1" * 64)),
        )
    )
    monkeypatch.setattr(live_cli, "PolymarketLiveRuntimeGuard", Guard)
    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)
    monkeypatch.setattr(
        live_cli,
        "PolymarketUserStreamConsumer",
        lambda _ledger, _guard: SimpleNamespace(),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketAuthenticatedUserStream",
        lambda _credentials, _consumer, *, markets: (
            WaitService("stream") if markets == ("0x" + "1" * 64,) else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketReconciliationService",
        lambda _coordinator, _guard, *, interval_seconds: (
            WaitService("reconciliation") if interval_seconds == 2 else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "_settlement_coordinator",
        lambda **_kwargs: (SimpleNamespace(), settlement_venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketSettlementService",
        lambda _coordinator, _guard, **kwargs: (
            WaitService("settlement")
            if kwargs["automatic_redemption_enabled"] is True
            and kwargs["interval_seconds"] == 5
            else pytest.fail()
        ),
    )

    payload = asyncio.run(
        live_cli._supervise(
            credentials=SimpleNamespace(),
            venue=SimpleNamespace(),
            ledger=ledger,
            risk_limits=live_cli._risk_limits("aggressive"),
            duration_seconds=0.001,
            reconciliation_seconds=2,
            automatic_redemption=True,
        )
    )

    assert payload["opened_exposure"] is False
    assert payload["automatic_redemption"] is True
    assert payload["runtime"]["stopped"] is True
    assert events[-3:] == ["guard-stopped", "settlement-close", "reconcile"]


def test_supervision_fails_closed_before_starting_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _reconciliation(
        ok=False,
        can_open=False,
        can_close=False,
        errors=("foreign_positions",),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveRuntimeGuard",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(preflight=lambda: blocked),
    )

    with pytest.raises(PolymarketLiveBlocked, match="foreign_positions"):
        asyncio.run(
            live_cli._supervise(
                credentials=SimpleNamespace(),
                venue=SimpleNamespace(),
                ledger=SimpleNamespace(),
                risk_limits=live_cli._risk_limits("conservative"),
                duration_seconds=1,
                reconciliation_seconds=5,
                automatic_redemption=False,
            )
        )


def test_supervise_command_dispatches_and_preserves_zero_exposure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    async def supervise(**kwargs):
        assert kwargs["duration_seconds"] == 3
        assert kwargs["reconciliation_seconds"] == 7
        return {
            "schema_version": "polymarket-live-supervision-v1",
            "opened_exposure": False,
        }

    monkeypatch.setattr(live_cli, "_supervise", supervise)

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "supervise",
            "--duration-seconds",
            "3",
            "--reconciliation-seconds",
            "7",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["action"] == "supervise"
    assert payload["opened_exposure"] is False


def test_autonomous_requires_explicit_hash_bound_evidence() -> None:
    with pytest.raises(
        PolymarketLiveBlocked,
        match=r"requires --promotion",
    ):
        asyncio.run(
            live_cli._autonomous(
                credentials=SimpleNamespace(),
                venue=SimpleNamespace(),
                ledger=SimpleNamespace(),
                risk_limits=live_cli._risk_limits("conservative"),
                duration_seconds=1,
                reconciliation_seconds=5,
                stop_timeout_seconds=30,
                automatic_redemption=False,
                promotion_path=None,
                evidence_root=None,
                round16_contract="contract.json",
                pretest_envelope_sha256=None,
                evaluation_envelope_sha256=None,
                requested_quantity=Decimal("5"),
                binance_bbo_safeguard=True,
            )
        )


def test_autonomous_assembles_independent_promoted_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    model_path = tmp_path / "pretest.json"
    evaluation_path = tmp_path / "evaluation.json"
    promotion_policy = SimpleNamespace(
        promotion_sha256="1" * 64,
        model_artifact=SimpleNamespace(sha256="2" * 64),
        market_variant="fifteenminute",
    )
    promotion = SimpleNamespace(
        promotion=promotion_policy,
        model_artifact_path=model_path,
        evaluation_report_path=evaluation_path,
    )
    predictor = object()
    flow = object()

    class PublicClient:
        def __init__(self) -> None:
            self.session = SimpleNamespace(close=lambda: events.append("public-close"))

    class Feed:
        trading_authority = False

        def __init__(self, *, flow: object) -> None:
            assert flow is globals_flow

        def health(self) -> PolymarketShadowFeedHealth:
            return PolymarketShadowFeedHealth(
                running=False,
                queue_size=0,
                queue_capacity=32_768,
                queue_high_watermark=7,
                received_counts={"spot": 10, "perpetual": 11},
                ingested_counts={"spot": 10, "perpetual": 11},
                reconnect_counts={"spot": 0, "perpetual": 0},
                stale_epoch_discard_counts={"spot": 0, "perpetual": 0},
                last_received_at_ms={"spot": 100, "perpetual": 101},
                last_event_time_ms={"spot": 99, "perpetual": 100},
                current_epochs={"spot": 1, "perpetual": 1},
                last_errors={"spot": "", "perpetual": ""},
            )

    globals_flow = flow
    guard = object()
    ledger = SimpleNamespace(
        path=tmp_path / "live.sqlite3",
        records=lambda: (),
        owned_inventory=lambda: (),
    )
    clean = _reconciliation()

    class Coordinator:
        def __init__(
            self,
            venue: object,
            selected_ledger: object,
            *,
            risk_limits: object,
            runtime_authority: object,
        ) -> None:
            assert venue is venue_object
            assert selected_ledger is ledger
            assert risk_limits is limits
            assert runtime_authority is guard
            self.ledger = selected_ledger
            self.runtime_authority = runtime_authority

        def reconcile(self) -> PolymarketReconciliation:
            events.append("reconcile")
            return clean

    class Stream:
        def __init__(
            self, credentials: object, consumer: object, *, markets: tuple[str, ...]
        ):
            assert credentials is credentials_object
            assert consumer.ledger is ledger
            assert consumer.runtime_guard is guard
            assert markets == ()
            self.consumer = consumer
            self.markets = ()

    class Reconciliation:
        def __init__(
            self,
            coordinator: object,
            runtime_guard: object,
            *,
            interval_seconds: float,
        ) -> None:
            assert interval_seconds == 7
            self.coordinator = coordinator
            self.runtime_guard = runtime_guard

    settlement_venue = SimpleNamespace(close=lambda: events.append("settlement-close"))
    snapshot = PolymarketAutonomousRuntimeSnapshot(
        venue="polymarket",
        symbol="BTC",
        market_variant="fifteenminute",
        horizon_minutes=15,
        paused=True,
        stop_requested=True,
        stop_completed=True,
        discovered_market_ids=("0x" + "1" * 64,),
        subscribed_market_ids=("0x" + "1" * 64,),
        decisions=2,
        submitted_opens=1,
        blocked_opens=0,
        requested_closes=1,
        completed_closes=1,
        last_fault="",
        external_signal_enabled=False,
        binance_execution_connected=False,
    )
    captured: dict[str, object] = {}

    class Supervisor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, *, duration_seconds: float) -> None:
            assert duration_seconds == 3
            events.append("run")

        def snapshot(self) -> PolymarketAutonomousRuntimeSnapshot:
            return snapshot

    components = SimpleNamespace(
        load_polymarket_live_promotion=lambda path, **kwargs: (
            promotion
            if path == "promotion.json"
            and kwargs["evidence_root"] == str(tmp_path)
            and kwargs["require_live_authority"] is True
            else pytest.fail()
        ),
        load_verified_round16_shadow_predictor=lambda **kwargs: (
            predictor
            if kwargs["pretest_path"] == model_path
            and kwargs["evaluation_path"] == evaluation_path
            and kwargs["expected_pretest_envelope_sha256"] == "3" * 64
            and kwargs["expected_evaluation_envelope_sha256"] == "4" * 64
            else pytest.fail()
        ),
        PolymarketPublicClient=PublicClient,
        PolymarketBtcFlowBuffer=lambda *, retention_seconds: (
            flow if retention_seconds == 1_200 else pytest.fail()
        ),
        PolymarketHistoricalShadowFeed=Feed,
        PolymarketRound16LiveFeatureBuilder=lambda selected_flow: (
            "builder" if selected_flow is flow else pytest.fail()
        ),
        PolymarketRound16ShadowScorer=lambda **kwargs: (
            "scorer"
            if kwargs == {"predictor": predictor, "feature_builder": "builder"}
            else pytest.fail()
        ),
        PolymarketRound16PromotedDecisionProvider=(
            lambda **kwargs: "decision-provider"
        ),
        BinanceBtcPublicSignalProvider=lambda: pytest.fail(),
        PolymarketAutonomousSupervisor=Supervisor,
    )
    monkeypatch.setattr(live_cli, "_load_autonomous_components", lambda: components)
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveRuntimeGuard",
        lambda **kwargs: (
            guard
            if kwargs["maximum_reconciliation_age_ms"] == 21_000
            and kwargs["opening_interlock"].lease_id
            else pytest.fail()
        ),
    )
    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)
    monkeypatch.setattr(
        live_cli,
        "PolymarketUserStreamConsumer",
        lambda selected_ledger, selected_guard: SimpleNamespace(
            ledger=selected_ledger,
            runtime_guard=selected_guard,
        ),
    )
    monkeypatch.setattr(live_cli, "PolymarketAuthenticatedUserStream", Stream)
    monkeypatch.setattr(
        live_cli,
        "PolymarketReconciliationService",
        Reconciliation,
    )
    monkeypatch.setattr(
        live_cli,
        "_settlement_coordinator",
        lambda **_kwargs: ("redemption", settlement_venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketSettlementService",
        lambda *args, **kwargs: (
            "settlement"
            if args == ("redemption", guard)
            and kwargs["automatic_redemption_enabled"] is True
            and kwargs["interval_seconds"] == 7
            else pytest.fail()
        ),
    )
    credentials_object = SimpleNamespace()
    venue_object = SimpleNamespace()
    limits = live_cli._risk_limits("conservative")
    payload = asyncio.run(
        live_cli._autonomous(
            credentials=credentials_object,
            venue=venue_object,
            ledger=ledger,
            risk_limits=limits,
            duration_seconds=3,
            reconciliation_seconds=7,
            stop_timeout_seconds=19,
            automatic_redemption=True,
            promotion_path="promotion.json",
            evidence_root=str(tmp_path),
            round16_contract="contract.json",
            pretest_envelope_sha256="3" * 64,
            evaluation_envelope_sha256="4" * 64,
            requested_quantity=Decimal("5"),
            binance_bbo_safeguard=False,
        )
    )

    assert captured["decision_data_service"].trading_authority is False
    assert captured["durable_control_service"].trading_authority is False
    assert captured["external_signal_provider"] is None
    assert captured["stop_timeout_seconds"] == 19
    assert payload["opened_exposure"] is True
    assert payload["binance_credentials_used"] is False
    assert payload["binance_execution_connected"] is False
    assert events == [
        "run",
        "reconcile",
        "settlement-close",
        "public-close",
    ]


def test_autonomous_command_dispatches_explicit_operator_pins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venue = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: (SimpleNamespace(signature_type=0), venue),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCoordinator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    async def autonomous(**kwargs: object) -> dict[str, object]:
        assert kwargs["promotion_path"] == "promotion.json"
        assert kwargs["evidence_root"] == "evidence"
        assert kwargs["requested_quantity"] == Decimal("6.5")
        assert kwargs["binance_bbo_safeguard"] is False
        return {
            "schema_version": "polymarket-live-autonomous-v1",
            "opened_exposure": False,
        }

    monkeypatch.setattr(live_cli, "_autonomous", autonomous)
    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "autonomous",
            "--ledger",
            str(tmp_path / "live.sqlite3"),
            "--promotion",
            "promotion.json",
            "--evidence-root",
            "evidence",
            "--pretest-envelope-sha256",
            "3" * 64,
            "--evaluation-envelope-sha256",
            "4" * 64,
            "--requested-quantity",
            "6.5",
            "--disable-binance-bbo-safeguard",
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["action"] == "autonomous"
    assert payload["opened_exposure"] is False


def test_cancel_result_fixture_remains_strict() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PolymarketCancelResult(("0x" + "1" * 64,), ("0x" + "1" * 64,))


def test_explicit_live_boundary_error_is_stable() -> None:
    error = PolymarketLiveBlocked("blocked")
    assert str(error) == "blocked"
