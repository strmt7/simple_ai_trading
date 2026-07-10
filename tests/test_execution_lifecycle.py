from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from simple_ai_trading.api_budget import build_api_budget_report
from simple_ai_trading.execution_lifecycle import (
    build_execution_lifecycle_plan,
    render_execution_lifecycle_plan,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore, bot_client_order_id
from simple_ai_trading.reconciliation import ReconciliationMismatch, ReconciliationReport
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


def _runtime(**overrides) -> RuntimeConfig:
    payload = {
        "symbol": "BTCUSDC",
        "symbols": ("BTCUSDC", "ETHUSDC", "SOLUSDC"),
        "market_type": "futures",
        "testnet": True,
        "dry_run": False,
        "api_key": "k",
        "api_secret": "s",
        "managed_usdc": 1000.0,
    }
    payload.update(overrides)
    return RuntimeConfig(**payload)


def _strategy(**overrides) -> StrategyConfig:
    payload = {"max_open_positions": 3}
    payload.update(overrides)
    return StrategyConfig(**payload)


def _ok_reconciliation(runtime: RuntimeConfig, *, local_live: int = 0, exchange: int = 0) -> ReconciliationReport:
    return ReconciliationReport(
        ok=True,
        market_type=runtime.market_type,
        symbols_checked=list(runtime.symbols),
        local_open_count=local_live,
        local_live_open_count=local_live,
        local_paper_open_count=0,
        exchange_exposure_count=exchange,
    )


def _position(*, verified: bool = True, dry_run: bool = False) -> OpenPosition:
    position_id = "pos-btc-long"
    return OpenPosition(
        id=position_id,
        symbol="BTCUSDC",
        market_type="futures",
        side="LONG",
        qty=0.01,
        entry_price=50_000.0,
        leverage=5.0,
        opened_at_ms=1,
        notional=500.0,
        dry_run=dry_run,
        open_client_order_id=bot_client_order_id(position_id, "open") if verified and not dry_run else "",
        open_exchange_order_id="123" if verified and not dry_run else "",
        exchange_status="FILLED" if verified and not dry_run else "local",
    )


def test_dry_run_lifecycle_does_not_require_signed_reconciliation(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    runtime = _runtime(dry_run=True, api_key="", api_secret="")

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="open",
        effective_dry_run=True,
    )

    assert plan.can_open is True
    assert plan.can_close is True
    assert plan.effective_dry_run is True
    assert "paper/dry-run" in render_execution_lifecycle_plan(plan)


def test_live_lifecycle_blocks_without_reconciliation(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    runtime = _runtime()

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="start",
        effective_dry_run=False,
        reconciliation=None,
    )

    assert plan.can_open is False
    assert plan.can_close is False
    assert "reconciliation:missing signed account reconciliation" in plan.open_block_reasons
    assert "reconciliation:missing signed account reconciliation" in plan.close_block_reasons


def test_live_lifecycle_blocks_external_exchange_exposure(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    runtime = _runtime()
    reconciliation = ReconciliationReport(
        ok=False,
        market_type="futures",
        symbols_checked=["BTCUSDC"],
        local_open_count=0,
        local_live_open_count=0,
        local_paper_open_count=0,
        exchange_exposure_count=1,
        external_exchange_exposure_count=1,
        mismatches=[
            ReconciliationMismatch(
                symbol="BTCUSDC",
                side="LONG",
                local_qty=0.0,
                exchange_qty=0.25,
                difference=0.25,
                reason="exchange_exposure_without_local_position",
            )
        ],
    )

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="open",
        effective_dry_run=False,
        reconciliation=reconciliation,
    )

    assert plan.can_open is False
    assert plan.can_close is False
    assert plan.external_exchange_exposure_count == 1
    assert any("exchange_exposure_without_local_position" in reason for reason in plan.open_block_reasons)


def test_live_lifecycle_blocks_unverified_local_position_even_when_quantities_match(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(verified=False))
    runtime = _runtime()

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="close",
        effective_dry_run=False,
        reconciliation=_ok_reconciliation(runtime, local_live=1, exchange=1),
    )

    assert plan.can_open is False
    assert plan.can_close is False
    assert plan.unverified_live_open_count == 1
    assert any("bot ownership" in reason for reason in plan.close_block_reasons)


def test_live_lifecycle_allows_verified_bot_owned_close_even_when_risk_blocks_new_entries(
    tmp_path: Path,
) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(verified=True))
    runtime = _runtime(managed_usdc=0.0)

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="close",
        effective_dry_run=False,
        reconciliation=_ok_reconciliation(runtime, local_live=1, exchange=1),
    )

    assert plan.can_open is False
    assert plan.can_close is True
    assert any(reason.startswith("risk policy:") for reason in plan.open_block_reasons)
    assert plan.close_block_reasons == ()


def test_api_budget_exhaustion_blocks_open_but_not_verified_close(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(verified=True))
    runtime = _runtime()
    budget = build_api_budget_report(
        market_type="futures",
        exchange_info={
            "rateLimits": [
                {
                    "rateLimitType": "REQUEST_WEIGHT",
                    "interval": "MINUTE",
                    "intervalNum": 1,
                    "limit": 1200,
                }
            ]
        },
        request_info={"rate_limit_headers": {"X-MBX-USED-WEIGHT-1M": "960"}},
    )

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="open",
        effective_dry_run=False,
        reconciliation=_ok_reconciliation(runtime, local_live=1, exchange=1),
        api_budget_report=budget,
    )

    assert plan.can_open is False
    assert plan.can_close is True
    assert any(reason.startswith("api budget:") for reason in plan.open_block_reasons)


def test_corrupt_open_ledger_blocks_all_signed_lifecycle_actions(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.open_path.parent.mkdir(parents=True, exist_ok=True)
    store.open_path.write_text("{", encoding="utf-8")
    runtime = _runtime()

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="start",
        effective_dry_run=False,
        reconciliation=_ok_reconciliation(runtime),
    )

    assert plan.can_open is False
    assert plan.can_close is False
    assert any(reason.startswith("ledger integrity:") for reason in plan.open_block_reasons)


def test_unknown_open_ledger_fields_block_lifecycle(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.open_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(_position(verified=True))
    payload["manual_override"] = True
    store.open_path.write_text(json.dumps([payload]), encoding="utf-8")
    runtime = _runtime()

    plan = build_execution_lifecycle_plan(
        runtime,
        _strategy(),
        store,
        action="start",
        effective_dry_run=False,
        reconciliation=_ok_reconciliation(runtime),
    )

    assert plan.can_open is False
    assert plan.can_close is False
    assert "unknown_fields=manual_override" in "; ".join(plan.open_block_reasons)
