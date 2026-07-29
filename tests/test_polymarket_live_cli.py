from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from simple_ai_trading import entrypoint
from simple_ai_trading import polymarket_live_cli as live_cli
from simple_ai_trading.polymarket_live import (
    PolymarketCancelResult,
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
    PolymarketRedemptionRecord,
    PolymarketReconciliation,
)
from simple_ai_trading.polymarket_live_runtime import PolymarketRuntimeSnapshot
from simple_ai_trading.polymarket_live_v2 import OfficialPolymarketV2Venue


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
    assert args.func is live_cli.command_polymarket_live


def test_installed_cli_native_app_and_contract_share_entrypoint() -> None:
    root = Path(entrypoint.__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    native = (root / "native" / "windows" / "src" / "main.cpp").read_text(
        encoding="utf-8"
    )
    contract = (
        root / "src" / "simple_ai_trading" / "command_contract.py"
    ).read_text(encoding="utf-8")

    assert (
        project["project"]["scripts"]["simple-ai-trading"]
        == "simple_ai_trading.entrypoint:main"
    )
    assert "-m simple_ai_trading.entrypoint" in native
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

    assert live_cli.command_polymarket_live(
        _args("--action", "status", "--ledger", str(ledger), "--json")
    ) == 0

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

    assert live_cli.command_polymarket_live(
        _args("--action", "status", "--ledger", str(path))
    ) == 0

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

    settlement_venue = SimpleNamespace(
        close=lambda: events.append("settlement-close")
    )
    ledger = SimpleNamespace(
        records=lambda: (
            SimpleNamespace(
                intent=SimpleNamespace(market_id="0x" + "1" * 64)
            ),
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
            WaitService("stream")
            if markets == ("0x" + "1" * 64,)
            else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "PolymarketReconciliationService",
        lambda _coordinator, _guard, *, interval_seconds: (
            WaitService("reconciliation")
            if interval_seconds == 2
            else pytest.fail()
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


def test_cancel_result_fixture_remains_strict() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PolymarketCancelResult(("0x" + "1" * 64,), ("0x" + "1" * 64,))


def test_explicit_live_boundary_error_is_stable() -> None:
    error = PolymarketLiveBlocked("blocked")
    assert str(error) == "blocked"
