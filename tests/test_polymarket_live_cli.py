from __future__ import annotations

import argparse
import ast
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
    assert args.activation is None
    assert args.ledger == live_cli.DEFAULT_POLYMARKET_LIVE_LEDGER.as_posix()
    assert (
        args.activation_output == live_cli.DEFAULT_POLYMARKET_LIVE_ACTIVATION.as_posix()
    )
    assert args.replace_activation is False
    assert args.promotion is None
    assert args.evidence_root is None
    assert args.lifecycle_qualification is None
    assert args.lifecycle_qualification_sha256 is None
    assert args.requested_quantity == Decimal("5")
    assert args.risk_capital_quote is None
    assert not hasattr(args, "disable_binance_bbo_safeguard")
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
    assert 'L"polymarket-live --action pause"' in native
    assert 'L"polymarket-live --action autonomous --activation "' in native
    assert 'L"data/polymarket/live-activation.json"' in native
    assert 'L"Start Polymarket"' in native
    assert 'L"Pause Polymarket"' in native
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
    assert payload["schema_version"] == "polymarket-live-operator-status-v2"
    assert payload["venue"] == "polymarket"
    assert payload["execution_mode"] == "authenticated_live_clob_v2"
    assert payload["execution_venue"] == "polymarket_clob_v2"
    assert payload["settlement_venue"] == "polymarket_polygon"
    assert payload["symbol"] == "BTC"
    assert payload["paper_execution"] is False
    assert payload["paper_fallback_allowed"] is False
    assert payload["binance_required_for_core_model"] is False
    assert payload["binance_public_predictor_optional"] is True
    assert payload["binance_public_predictor_data_allowed"] is True
    assert payload["binance_public_predictor_trading_authority"] is False
    assert payload["binance_credentials_allowed"] is False
    assert payload["binance_account_authority"] is False
    assert payload["binance_execution_authority"] is False
    assert payload["binance_position_authority"] is False
    assert payload["binance_risk_or_stop_authority"] is False
    assert payload["can_open"] is False
    assert payload["ledger_exists"] is False
    assert payload["runtime_state"] == "unconfigured"
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
    assert "runtime_state=" in rendered
    assert "can_open=False" in rendered


def test_pause_and_resume_are_durable_and_do_not_require_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "live.sqlite3"
    control = PolymarketRuntimeControl(path)
    control.acquire(owner_process_id=101)
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: pytest.fail("pause and resume must not create an execution client"),
    )

    assert (
        live_cli.command_polymarket_live(
            _args("--action", "pause", "--ledger", str(path), "--json")
        )
        == 0
    )
    paused = json.loads(capsys.readouterr().out)
    assert paused["runtime_control"]["paused"] is True
    assert control.snapshot().paused is True

    assert (
        live_cli.command_polymarket_live(
            _args("--action", "resume", "--ledger", str(path), "--json")
        )
        == 0
    )
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["runtime_control"]["paused"] is False
    assert control.snapshot().paused is False


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
            self.args = _args
            self.kwargs = _kwargs

        @staticmethod
        def preflight():
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
            self.args = _args
            self.kwargs = _kwargs

        @staticmethod
        def reconcile():
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


def test_cancel_owned_cancels_exact_owned_orders_despite_foreign_state(
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
            self.args = _args
            self.kwargs = _kwargs

        def cancel_owned_open_orders(self):
            calls.append("cancel")
            return PolymarketCancelResult((), ())

        @staticmethod
        def reconcile():
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=False,
                errors=("foreign_positions",),
            )

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
    assert calls == ["cancel"]
    assert "foreign_positions" in capsys.readouterr().out


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
            self.args = _args
            self.kwargs = _kwargs

        @staticmethod
        def preflight():
            return _reconciliation()

        @staticmethod
        def cancel_owned_open_orders():
            return PolymarketCancelResult((order_id,), ())

        @staticmethod
        def reconcile():
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
        @staticmethod
        def owned_inventory():
            return tuple(inventory)

        def open_owned_order_ids(self):
            return ()

    class Coordinator:
        @staticmethod
        def cancel_owned_open_orders():
            return PolymarketCancelResult((order_id,), ())

        @staticmethod
        def reconcile():
            return _reconciliation(
                ok=False,
                can_open=False,
                can_close=True,
                errors=("foreign_positions",),
            )

        @staticmethod
        def submit_owned_close_orders(*, maximum_book_age_ms: int):
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
        @staticmethod
        def owned_inventory():
            return inventory

        def open_owned_order_ids(self):
            return ()

    class Coordinator:
        @staticmethod
        def cancel_owned_open_orders():
            nonlocal cancelled
            cancelled = True
            return PolymarketCancelResult((), ())

        @staticmethod
        def reconcile():
            return _reconciliation()

        @staticmethod
        def submit_owned_close_orders(*, maximum_book_age_ms: int):
            assert maximum_book_age_ms == 1_500
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
        reconcile=_reconciliation,
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
        reconcile=_reconciliation,
    )
    clock = iter((0.0, 0.0, 0.0, 2.0, 2.0, 2.0))
    sleeps: list[float] = []

    def next_clock_value() -> float:
        try:
            return next(clock)
        except StopIteration:
            pytest.fail("monotonic test clock exhausted")

    monkeypatch.setattr(live_cli.time, "monotonic", next_clock_value)
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
        @staticmethod
        def owned_inventory():
            try:
                return next(snapshots)
            except StopIteration:
                pytest.fail("inventory snapshots exhausted")

        @staticmethod
        def open_owned_order_ids():
            return ()

    coordinator = SimpleNamespace(
        cancel_owned_open_orders=lambda: PolymarketCancelResult((), ()),
        reconcile=_reconciliation,
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


def test_polymarket_authority_modules_import_only_polymarket_local_boundaries() -> None:
    authority_modules = (
        "polymarket_live.py",
        "polymarket_live_v2.py",
        "polymarket_live_runtime.py",
        "polymarket_live_settlement.py",
        "polymarket_live_stop.py",
        "polymarket_runtime_control.py",
        "polymarket_autonomous.py",
        "polymarket_autonomous_runtime.py",
    )
    package_root = Path(live_cli.__file__).parent

    for filename in authority_modules:
        path = package_root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.module is not None
                assert node.module.startswith("polymarket"), (
                    f"{filename} imports non-Polymarket local module "
                    f"{node.module!r}; keep venue authority independent"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("simple_ai_trading."):
                        assert alias.name.startswith("simple_ai_trading.polymarket"), (
                            f"{filename} imports non-Polymarket package module "
                            f"{alias.name!r}; keep venue authority independent"
                        )


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

        @staticmethod
        def reconcile():
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
                lifecycle_qualification_path=None,
                lifecycle_qualification_sha256=None,
                round16_contract="contract.json",
                pretest_envelope_sha256=None,
                evaluation_envelope_sha256=None,
                requested_quantity=Decimal("5"),
                risk_level="conservative",
                risk_capital_quote=Decimal("10000"),
            )
        )


def test_autonomous_requires_explicit_positive_risk_capital() -> None:
    with pytest.raises(
        PolymarketLiveBlocked,
        match=r"requires positive --risk-capital-quote",
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
                promotion_path="promotion.json",
                evidence_root="evidence",
                lifecycle_qualification_path="lifecycle.json",
                lifecycle_qualification_sha256="7" * 64,
                round16_contract="contract.json",
                pretest_envelope_sha256=None,
                evaluation_envelope_sha256=None,
                requested_quantity=Decimal("5"),
                risk_level="conservative",
                risk_capital_quote=None,
            )
        )


def test_five_minute_autonomous_stack_does_not_require_round16_pins() -> None:
    public_client = object()
    promotion = SimpleNamespace(promotion=SimpleNamespace(market_variant="fiveminute"))
    data_service = object()
    decision_provider = object()
    components = SimpleNamespace(
        build_polymarket_round21_runtime_stack=lambda **kwargs: (
            SimpleNamespace(
                data_service=data_service,
                decision_provider=decision_provider,
            )
            if kwargs
            == {
                "public_client": public_client,
                "promotion": promotion,
                "requested_quantity": Decimal("2.5"),
                "risk_level": "aggressive",
            }
            else pytest.fail("unexpected Round 21 stack arguments")
        ),
        load_verified_round16_shadow_predictor=lambda **_kwargs: pytest.fail(
            "Round 16 evidence must remain unopened"
        ),
    )

    actual = live_cli._build_autonomous_decision_stack(
        components=components,
        public_client=public_client,
        promotion=promotion,
        round16_contract=None,
        pretest_envelope_sha256=None,
        evaluation_envelope_sha256=None,
        requested_quantity=Decimal("2.5"),
        risk_level="aggressive",
    )

    assert actual == (data_service, decision_provider)


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
    lifecycle_qualification = SimpleNamespace(
        qualification=SimpleNamespace(qualification_sha256="6" * 64),
        assert_runtime_binding=lambda **_kwargs: events.append("lifecycle-bound"),
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

        @staticmethod
        def health() -> PolymarketShadowFeedHealth:
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

        @staticmethod
        def reconcile() -> PolymarketReconciliation:
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
        binance_execution_connected=False,
    )
    captured: dict[str, object] = {}

    class Supervisor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, *, duration_seconds: float) -> None:
            assert duration_seconds == 3
            events.append("run")

        @staticmethod
        def snapshot() -> PolymarketAutonomousRuntimeSnapshot:
            return snapshot

    def load_live_promotion(path, **kwargs):
        assert path == "promotion.json"
        assert kwargs["evidence_root"] == str(tmp_path)
        assert kwargs["require_live_authority"] is True
        assert kwargs["expected_file_sha256"] == "8" * 64
        return promotion

    def load_lifecycle_qualification(path, **kwargs):
        assert path == "lifecycle.json"
        assert kwargs["expected_file_sha256"] == "7" * 64
        return lifecycle_qualification

    def load_shadow_predictor(**kwargs):
        assert kwargs["pretest_path"] == model_path
        assert kwargs["evaluation_path"] == evaluation_path
        assert kwargs["expected_pretest_envelope_sha256"] == "3" * 64
        assert kwargs["expected_evaluation_envelope_sha256"] == "4" * 64
        assert kwargs["expected_contract_file_sha256"] == "9" * 64
        return predictor

    def build_flow(*, retention_seconds):
        assert retention_seconds == 1_200
        return flow

    def build_feature(selected_flow):
        assert selected_flow is flow
        return "builder"

    def build_scorer(**kwargs):
        assert kwargs == {"predictor": predictor, "feature_builder": "builder"}
        return "scorer"

    def build_decision_provider(**_kwargs):
        return "decision-provider"

    def build_runtime_guard(**kwargs):
        assert kwargs["maximum_reconciliation_age_ms"] == 21_000
        assert kwargs["opening_interlock"].lease_id
        return guard

    def build_user_stream_consumer(selected_ledger, selected_guard):
        return SimpleNamespace(
            ledger=selected_ledger,
            runtime_guard=selected_guard,
        )

    def build_settlement_service(*args, **kwargs):
        assert args == ("redemption", guard)
        assert kwargs["automatic_redemption_enabled"] is True
        assert kwargs["interval_seconds"] == 7
        return "settlement"

    components = SimpleNamespace(
        load_polymarket_live_promotion=load_live_promotion,
        load_polymarket_lifecycle_qualification=load_lifecycle_qualification,
        load_verified_round16_shadow_predictor=load_shadow_predictor,
        PolymarketPublicClient=PublicClient,
        PolymarketBtcFlowBuffer=build_flow,
        PolymarketHistoricalShadowFeed=Feed,
        PolymarketRound16LiveFeatureBuilder=build_feature,
        PolymarketRound16ShadowScorer=build_scorer,
        PolymarketRound16PromotedDecisionProvider=build_decision_provider,
        PolymarketAutonomousSupervisor=Supervisor,
    )
    monkeypatch.setattr(live_cli, "_load_autonomous_components", lambda: components)
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveRuntimeGuard",
        build_runtime_guard,
    )
    monkeypatch.setattr(live_cli, "PolymarketLiveCoordinator", Coordinator)
    monkeypatch.setattr(
        live_cli,
        "PolymarketUserStreamConsumer",
        build_user_stream_consumer,
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
        build_settlement_service,
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
            lifecycle_qualification_path="lifecycle.json",
            lifecycle_qualification_sha256="7" * 64,
            round16_contract="contract.json",
            pretest_envelope_sha256="3" * 64,
            evaluation_envelope_sha256="4" * 64,
            requested_quantity=Decimal("5"),
            risk_level="conservative",
            risk_capital_quote=Decimal("10000"),
            promotion_file_sha256="8" * 64,
            round16_contract_file_sha256="9" * 64,
        )
    )

    assert captured["decision_data_service"].trading_authority is False
    assert captured["lifecycle_qualification"] is lifecycle_qualification
    assert captured["durable_control_service"].trading_authority is False
    assert "external_signal_provider" not in captured
    assert captured["stop_timeout_seconds"] == 19
    assert captured["risk_capital_quote"] == Decimal("10000")
    assert captured["risk_level"] == "conservative"
    assert payload["opened_exposure"] is True
    assert payload["binance_credentials_used"] is False
    assert payload["binance_execution_connected"] is False
    assert events == [
        "lifecycle-bound",
        "run",
        "reconcile",
        "settlement-close",
        "public-close",
    ]


def test_autonomous_cleanup_surfaces_failed_stop_latch_after_closing_resources() -> (
    None
):
    events: list[str] = []

    class Control:
        def request_stop(self, *, reason: str) -> None:
            events.append(f"stop:{reason}")
            raise RuntimeError("durable stop latch failed")

        @staticmethod
        def release(*_args: object, **_kwargs: object) -> None:
            pytest.fail("owned exposure must never release the runtime lease")

    ledger = SimpleNamespace(owned_inventory=lambda: (object(),))
    settlement = SimpleNamespace(close=lambda: events.append("settlement-close"))
    public = SimpleNamespace(
        session=SimpleNamespace(close=lambda: events.append("public-close"))
    )

    with pytest.raises(RuntimeError, match="durable stop latch failed"):
        live_cli._finalize_autonomous_resources(
            runtime_control=Control(),
            runtime_lease="lease",
            runtime_released=False,
            ledger=ledger,
            settlement_venue=settlement,
            public_client=public,
        )

    assert events == [
        "stop:autonomous_exit_with_owned_exposure",
        "settlement-close",
        "public-close",
    ]


def test_autonomous_cleanup_latches_stop_when_inventory_is_unknown() -> None:
    events: list[str] = []

    class Control:
        @staticmethod
        def request_stop(*, reason: str) -> None:
            events.append(f"stop:{reason}")

        @staticmethod
        def release(*_args: object, **_kwargs: object) -> None:
            pytest.fail("unknown inventory must never release the runtime lease")

    def unreadable_inventory() -> tuple[object, ...]:
        raise OSError("ownership ledger unavailable")

    ledger = SimpleNamespace(owned_inventory=unreadable_inventory)
    settlement = SimpleNamespace(close=lambda: events.append("settlement-close"))
    public = SimpleNamespace(
        session=SimpleNamespace(close=lambda: events.append("public-close"))
    )

    with pytest.raises(OSError, match="ownership ledger unavailable"):
        live_cli._finalize_autonomous_resources(
            runtime_control=Control(),
            runtime_lease="lease",
            runtime_released=False,
            ledger=ledger,
            settlement_venue=settlement,
            public_client=public,
        )

    assert events == [
        "stop:autonomous_exit_inventory_unknown",
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
        assert kwargs["lifecycle_qualification_path"] == "lifecycle.json"
        assert kwargs["lifecycle_qualification_sha256"] == "7" * 64
        assert kwargs["requested_quantity"] == Decimal("6.5")
        assert kwargs["risk_level"] == "conservative"
        assert kwargs["risk_capital_quote"] == Decimal("10000")
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
            "--lifecycle-qualification",
            "lifecycle.json",
            "--lifecycle-qualification-sha256",
            "7" * 64,
            "--pretest-envelope-sha256",
            "3" * 64,
            "--evaluation-envelope-sha256",
            "4" * 64,
            "--requested-quantity",
            "6.5",
            "--risk-capital-quote",
            "10000",
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["action"] == "autonomous"
    assert payload["opened_exposure"] is False


def test_prepare_autonomous_writes_verified_non_secret_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    promotion_path = evidence / "promotion.json"
    lifecycle_path = evidence / "lifecycle.json"
    promotion_path.write_text('{"promotion":true}\n', encoding="utf-8")
    lifecycle_path.write_text('{"qualification":true}\n', encoding="utf-8")
    binding_calls: list[dict[str, object]] = []

    class Lifecycle:
        @staticmethod
        def assert_runtime_binding(**kwargs: object) -> None:
            binding_calls.append(dict(kwargs))

    promotion = SimpleNamespace(promotion=SimpleNamespace(market_variant="fiveminute"))
    components = SimpleNamespace(
        load_polymarket_live_promotion=lambda *_args, **_kwargs: promotion,
        load_polymarket_lifecycle_qualification=(lambda *_args, **_kwargs: Lifecycle()),
    )
    credentials = SimpleNamespace(funder_address="0x" + "1" * 40)
    monkeypatch.setattr(live_cli, "_load_autonomous_components", lambda: components)
    monkeypatch.setattr(
        live_cli,
        "PolymarketLiveCredentials",
        SimpleNamespace(from_environment=lambda: credentials),
    )
    output = tmp_path / "activation" / "live.json"
    args = _args(
        "--action",
        "prepare-autonomous",
        "--activation-output",
        str(output),
        "--promotion",
        str(promotion_path),
        "--evidence-root",
        str(evidence),
        "--lifecycle-qualification",
        str(lifecycle_path),
        "--risk-level",
        "regular",
        "--risk-capital-quote",
        "500",
        "--requested-quantity",
        "7.5",
    )

    result = live_cli._prepare_autonomous_activation(args)

    assert result["activation_path"] == str(output.resolve())
    assert result["risk_level"] == "regular"
    assert result["risk_capital_quote"] == "500"
    assert result["credentials_stored"] is False
    assert result["orders_submitted"] is False
    assert binding_calls and binding_calls[0]["credentials"] is credentials
    serialized = output.read_text(encoding="utf-8").lower()
    assert "funder_address" not in serialized
    assert "private_key" not in serialized


def test_activation_bundle_overrides_default_risk_inputs_and_rejects_mixing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    promotion = evidence / "promotion.json"
    lifecycle = evidence / "lifecycle.json"
    promotion.write_text("promotion", encoding="utf-8")
    lifecycle.write_text("lifecycle", encoding="utf-8")
    activation = live_cli.write_polymarket_live_activation(
        tmp_path / "activation.json",
        market_variant="fiveminute",
        risk_level="aggressive",
        risk_capital_quote=Decimal("900"),
        requested_quantity=Decimal("8"),
        promotion_path=promotion,
        evidence_root=evidence,
        lifecycle_qualification_path=lifecycle,
        round16_contract_path=None,
        pretest_envelope_sha256="",
        evaluation_envelope_sha256="",
    )

    resolved = live_cli._activation_autonomous_arguments(
        _args("--action", "autonomous", "--activation", str(activation.path))
    )

    assert resolved["risk_level"] == "aggressive"
    assert resolved["risk_capital_quote"] == Decimal("900")
    assert resolved["requested_quantity"] == Decimal("8")
    assert resolved["activation_sha256"] == activation.activation.activation_sha256
    assert resolved["promotion_path"] == str(promotion.resolve())
    assert (
        resolved["promotion_file_sha256"] == activation.activation.promotion.file_sha256
    )
    assert resolved["round16_contract_file_sha256"] == ""

    with pytest.raises(PolymarketLiveBlocked, match="cannot be mixed"):
        live_cli._activation_autonomous_arguments(
            _args(
                "--action",
                "autonomous",
                "--activation",
                str(activation.path),
                "--risk-capital-quote",
                "1",
            )
        )


def test_prepare_autonomous_never_constructs_live_venue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "schema_version": "polymarket-live-operator-result-v1",
        "action": "prepare-autonomous",
        "venue": "polymarket",
        "orders_submitted": False,
    }
    monkeypatch.setattr(
        live_cli,
        "_prepare_autonomous_activation",
        lambda _args: expected,
    )
    monkeypatch.setattr(
        live_cli,
        "_credentials_and_venue",
        lambda: pytest.fail("prepare-autonomous constructed a live venue"),
    )

    result = live_cli.command_polymarket_live(
        _args(
            "--action",
            "prepare-autonomous",
            "--activation-output",
            str(tmp_path / "activation.json"),
            "--json",
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_cancel_result_fixture_remains_strict() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PolymarketCancelResult(("0x" + "1" * 64,), ("0x" + "1" * 64,))


def test_explicit_live_boundary_error_is_stable() -> None:
    error = PolymarketLiveBlocked("blocked")
    assert str(error) == "blocked"
