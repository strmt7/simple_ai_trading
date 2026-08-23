from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_cross_regime_evaluation import (
    CROSS_REGIME_EDGE_ACCEPTANCE_CONTRACT_SHA256,
    POLYMARKET_REQUIRED_REGIME_SLICES,
    polymarket_cross_regime_evaluation_sha256,
)
from simple_ai_trading.polymarket_live_manifest import (
    build_polymarket_live_implementation_manifest,
)


POLYMARKET_LIVE_PROMOTION_GATES = {
    "prospective_untouched_test": True,
    "source_integrity": True,
    "causal_feature_replay": True,
    "proper_scoring_uplift": True,
    "after_cost_edge": True,
    "uncertainty_lower_bound": True,
    "drawdown_limit": True,
    "latency_stress": True,
    "displayed_depth_stress": True,
    "authenticated_order_lifecycle": True,
    "settlement_and_redemption": True,
    "cross_regime_after_cost": True,
}
_MANDATORY_ABSTENTION_SLICES = {
    "liquidity:stale_or_missing_book",
    "execution:no_fill_or_unknown_state",
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def _implementation_manifest_bytes(source_commit: str) -> bytes:
    payload = build_polymarket_live_implementation_manifest(
        source_commit=source_commit,
    )
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def write_polymarket_live_implementation_manifest(
    path: Path,
    *,
    source_commit: str,
) -> None:
    path.write_bytes(_implementation_manifest_bytes(source_commit))


def build_cross_regime_evaluation_fixture(
    *,
    model_artifact_sha256: str,
    source_commit: str,
    created_at_ms: int,
    market_variant: str,
    risk_profile: str = "conservative",
) -> dict[str, object]:
    """Build synthetic test evidence; it makes no financial or edge claim."""

    slices: list[dict[str, object]] = []
    for slice_id in POLYMARKET_REQUIRED_REGIME_SLICES:
        abstain = slice_id in _MANDATORY_ABSTENTION_SLICES
        slices.append(
            {
                "slice_id": slice_id,
                "classification_count": 30,
                "decision_count": 30,
                "policy_action": "abstain" if abstain else "trade",
                "opened_position_count": 0 if abstain else 30,
                "closed_trade_count": 0 if abstain else 30,
                "after_cost_net_pnl_quote": "0" if abstain else "1",
                "paired_delta_lower_bound_quote": "0" if abstain else "0.1",
                "maximum_drawdown_capital_fraction": ("0" if abstain else "0.001"),
                "expected_shortfall_capital_fraction": ("0" if abstain else "0.0005"),
                "unknown_execution_state_count": 0,
                "untracked_inventory_count": 0,
                "abstention_replay_passed": abstain,
                "passed": True,
            }
        )
    traded_slice_count = len(slices) - len(_MANDATORY_ABSTENTION_SLICES)
    concentration = Decimal("1") / Decimal(traded_slice_count)
    body: dict[str, object] = {
        "schema_version": "polymarket-cross-regime-evaluation-v1",
        "evaluation_id": "a" * 64,
        "created_at_ms": created_at_ms,
        "source_commit": source_commit,
        "venue": "polymarket",
        "asset": "BTC",
        "market_variant": market_variant,
        "risk_profile": risk_profile,
        "model_artifact_sha256": model_artifact_sha256,
        "cross_regime_contract_sha256": (CROSS_REGIME_EDGE_ACCEPTANCE_CONTRACT_SHA256),
        "data_manifest_sha256": "4" * 64,
        "cost_model_sha256": "5" * 64,
        "regime_definition_sha256": "6" * 64,
        "selection_policy_sha256": "7" * 64,
        "role_evidence": {
            "train_sha256": "8" * 64,
            "tune_sha256": "9" * 64,
            "test_sha256": "c" * 64,
            "roles_nonoverlapping": True,
            "test_sealed_before_selection": True,
            "selection_frozen_before_test": True,
        },
        "thresholds": {
            "minimum_classifications_per_slice": 30,
            "minimum_closed_trades_per_traded_slice": 30,
            "bootstrap_confidence": "0.95",
            "maximum_profit_concentration_fraction": "0.60",
        },
        "slices": slices,
        "aggregate": {
            "classification_count": 30 * len(slices),
            "decision_count": 30 * len(slices),
            "opened_position_count": 30 * traded_slice_count,
            "closed_trade_count": 30 * traded_slice_count,
            "after_cost_net_pnl_quote": str(traded_slice_count),
            "block_bootstrap_lower_bound_quote": "1",
            "maximum_drawdown_capital_fraction": "0.01",
            "expected_shortfall_capital_fraction": "0.001",
            "maximum_profit_concentration_fraction": str(concentration),
            "unknown_execution_state_count": 0,
            "untracked_inventory_count": 0,
            "passed": True,
        },
        "authority": {
            "edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    return {
        **body,
        "evaluation_sha256": polymarket_cross_regime_evaluation_sha256(body),
    }


def build_polymarket_live_promotion_fixture(
    root: Path,
    *,
    now_ms: int,
    source_commit: str = "b" * 40,
    market_variant: str = "fiveminute",
    risk_profile: str = "conservative",
    bot_id: str = "simple-ai-trading-polymarket-btc",
    live: bool = True,
    created_at_ms: int | None = None,
    expires_at_ms: int | None = None,
) -> dict[str, object]:
    """Write complete synthetic promotion fixtures with semantic evidence."""

    root.mkdir(parents=True, exist_ok=True)
    created = now_ms - 1_000 if created_at_ms is None else created_at_ms
    expires = now_ms + 86_400_000 if expires_at_ms is None else expires_at_ms
    model_path = root / "model.json"
    model_path.write_bytes(b'{"model":"synthetic-test-fixture"}\n')
    model_sha = _file_sha(model_path)
    evaluation_path = root / "evaluation.json"
    evaluation = build_cross_regime_evaluation_fixture(
        model_artifact_sha256=model_sha,
        source_commit=source_commit,
        created_at_ms=created - 1,
        market_variant=market_variant,
        risk_profile=risk_profile,
    )
    evaluation_path.write_text(_canonical(evaluation), encoding="ascii")
    manifest_path = root / "implementation.json"
    write_polymarket_live_implementation_manifest(
        manifest_path,
        source_commit=source_commit,
    )
    body: dict[str, object] = {
        "schema_version": "polymarket-live-promotion-v2",
        "promotion_id": "d" * 64,
        "created_at_ms": created,
        "expires_at_ms": expires,
        "source_commit": source_commit,
        "venue": "polymarket",
        "protocol_version": 2,
        "asset": "BTC",
        "market_variant": market_variant,
        "risk_profile": risk_profile,
        "environment": "live",
        "bot_id": bot_id,
        "model_artifact": {
            "path": model_path.name,
            "sha256": model_sha,
        },
        "evaluation_report": {
            "path": evaluation_path.name,
            "sha256": _file_sha(evaluation_path),
        },
        "implementation_manifest": {
            "path": manifest_path.name,
            "sha256": _file_sha(manifest_path),
        },
        "gates": dict(POLYMARKET_LIVE_PROMOTION_GATES),
        "policy": {
            "minimum_expected_edge_quote_per_share": "0.02",
            "maximum_prediction_age_ms": 1_000,
            "minimum_remaining_seconds": 30,
        },
        "authority": {"paper": True, "live": live},
    }
    return {**body, "promotion_sha256": _sha(body)}


__all__ = [
    "POLYMARKET_LIVE_PROMOTION_GATES",
    "build_cross_regime_evaluation_fixture",
    "build_polymarket_live_promotion_fixture",
    "write_polymarket_live_implementation_manifest",
]
