from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round27_campaign_admission import (
    round27_campaign_role_population_sha256,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)


_START_MS = 1_786_784_400_000
_OPENED_MS = _START_MS + 400_000
_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _admission(contract: dict[str, object], *, role: str) -> dict[str, object]:
    reports = [
        {
            "slot_id": slot_id,
            "run_id": f"{slot_id}-run",
            "condition_audit_sha256": _sha(f"{slot_id}-audit"),
            "feature_report_sha256": _sha(f"{slot_id}-report"),
            "condition_count": count,
            "row_count": count,
            "row_chain_sha256": _sha(f"{slot_id}-rows"),
        }
        for slot_id, count in (("stage1-a", 110), ("stage1-b", 95), ("stage1-c", 95))
    ]
    body: dict[str, object] = {
        "schema_version": "polymarket-round27-campaign-admission-v1",
        "admitted_at_ms": _OPENED_MS - 1,
        "campaign_contract_sha256": contract["campaign_contract_sha256"],
        "round27_model_contract_sha256": contract["contract_sha256"],
        "round27_model_amendment_sha256": contract[
            "model_implementation_amendment_sha256"
        ],
        "feature_store_audit_sha256": "b" * 64,
        "audited_slot_ids": ["stage1-a", "stage1-b", "stage1-c"],
        "primary_slot_ids": ["stage1-a", "stage1-b", "stage1-c"],
        "slot_reports": reports,
        "primary_eligible_condition_count": 300,
        "contingency_condition_count": 0,
        "eligible_condition_count": 300,
        "role_condition_counts": {
            "train": 80,
            "calibration": 30,
            "selection": 95,
            "sealed": 95,
            "purged": 0,
        },
        "role_population_sha256": {
            selected_role: (
                round27_campaign_role_population_sha256((_row(),))
                if selected_role == role
                else _sha(f"{selected_role}-population")
            )
            for selected_role in ("train", "calibration", "selection", "sealed")
        },
        "minimum_population": {
            name: contract["minimum_population"][name]
            for name in sorted(contract["minimum_population"])
        },
        "contingency_required": False,
        "contingency_used": False,
        "all_primary_target_free_audits_present": True,
        "target_free": True,
        "target_accessed": False,
        "target_access_authorized": True,
        "authority": {
            "credentials_used": False,
            "edge_claim": False,
            "execution_connected": False,
            "live_trading_authority": False,
            "orders_submitted": False,
            "paper_trading_authority": False,
            "profitability_claim": False,
        },
    }
    body["admission_sha256"] = _sha(_canonical(body))
    return body


def _market() -> PolymarketFiveMinuteMarket:
    gamma = {
        "id": "12345",
        "conditionId": "0x" + "7" * 64,
        "slug": "btc-updown-5m-1786784400",
        "outcomes": ["Up", "Down"],
        "clobTokenIds": ["7" * 40, "8" * 40],
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
    }
    canonical = _canonical(gamma)
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id="0x" + "7" * 64,
        slug="btc-updown-5m-1786784400",
        question="Bitcoin Up or Down",
        event_start_ms=_START_MS,
        end_ms=_START_MS + 300_000,
        up_token_id="7" * 40,
        down_token_id="8" * 40,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.2"),
        ),
        liquidity_quote=Decimal("10000"),
        volume_quote=Decimal("20000"),
        resolution_source="https://data.chain.link/streams/btc-usd",
        gamma_payload_sha256=_sha(canonical),
        gamma_payload_json=canonical,
    )


def _row() -> Round27FeatureRow:
    return Round27FeatureRow.create(
        run_id="stage1-a-run",
        condition_id=_market().condition_id,
        event_start_ms=_START_MS,
        decision_time_ms=_START_MS + 30_000,
        market_prior_probability=0.5,
        values=[0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES),
        maximum_receipt_wall_ms=_START_MS + 29_999,
        source_chain_sha256="d" * 64,
    )


def _envelope(
    value: dict[str, object],
    observed_ms: int,
) -> Round25OfficialPublicPayload:
    canonical = _canonical(value)
    return Round25OfficialPublicPayload(
        value=value,
        canonical_json=canonical,
        sha256=_sha(canonical),
        observed_wall_ms=observed_ms,
        observed_monotonic_ns=observed_ms * 1_000_000,
    )


def _official() -> tuple[dict[str, object], dict[str, object]]:
    market = _market()
    gamma = {
        "id": market.market_id,
        "conditionId": market.condition_id,
        "slug": market.slug,
        "outcomes": ["Up", "Down"],
        "clobTokenIds": list(market.token_ids),
        "resolutionSource": market.resolution_source,
        "outcomePrices": ["1", "0"],
        "closed": True,
        "acceptingOrders": False,
    }
    clob = {
        "condition_id": market.condition_id,
        "market_slug": market.slug,
        "closed": True,
        "accepting_orders": False,
        "tokens": [
            {
                "outcome": outcome,
                "price": "1" if outcome == "Up" else "0",
                "token_id": token_id,
                "winner": outcome == "Up",
            }
            for token_id, outcome in zip(
                market.token_ids,
                ("Up", "Down"),
                strict=True,
            )
        ],
    }
    return clob, gamma


class _Client:
    def clob_market(self, _condition_id: str) -> Round25OfficialPublicPayload:
        clob, _gamma = _official()
        return _envelope(clob, _OPENED_MS + 1)

    def gamma_market(self, _market_id: str) -> Round25OfficialPublicPayload:
        _clob, gamma = _official()
        return _envelope(gamma, _OPENED_MS + 2)


def _open(
    store: Round27TargetStore,
    *,
    role: str = "train",
    contract: dict[str, object] | None = None,
    admission: dict[str, object] | None = None,
) -> bool:
    selected_contract = contract or load_round27_model_contract(_ROOT)
    return store.open_role(
        role=role,
        slot_id="stage1-a",
        run_id="stage1-a-run",
        contract=selected_contract,
        feature_store_audit_sha256="b" * 64,
        campaign_admission=admission or _admission(selected_contract, role=role),
        role_intervals=(
            {
                "role": role,
                "slot_id": "stage1-a",
                "start_ms": _START_MS,
                "end_ms": _START_MS + 300_000,
            },
        ),
        feature_rows=(_row(),),
        markets=((_market(), "c" * 64),),
        opened_at_ms=_OPENED_MS,
    )


def test_target_store_is_role_gated_separate_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "targets.duckdb"
    with Round27TargetStore(path) as store:
        assert _open(store) is True
        assert _open(store) is False
        initial = store.audit()
        assert initial["feature_targets_co_located"] is False
        assert initial["resolved_condition_count"] == 0

        collected = store.collect_once(role="train", client=_Client())
        assert collected["newly_resolved_condition_count"] == 1
        final = store.finalize_role("train")
        assert final["finalized"] is True
        assert store.outcomes_up(roles=("train",)) == {_market().condition_id: 1}

    with Round27TargetStore(path, read_only=True) as reopened:
        audit = reopened.audit()
        assert audit["condition_count"] == 1
        assert audit["resolved_condition_count"] == 1
        assert reopened.outcomes_up(roles=("train",)) == {
            _market().condition_id: 1
        }


def test_target_store_rejects_sealed_access_without_exact_artifacts(
    tmp_path: Path,
) -> None:
    with Round27TargetStore(tmp_path / "targets.duckdb") as store:
        with pytest.raises(ValueError, match="sealed target artifacts are required"):
            _open(store, role="sealed")


def test_target_store_rejects_a_changed_model_amendment(tmp_path: Path) -> None:
    contract = load_round27_model_contract(_ROOT)
    contract["model_implementation_amendment_sha256"] = "0" * 64

    with Round27TargetStore(tmp_path / "targets.duckdb") as store:
        with pytest.raises(ValueError, match="campaign model contract differs"):
            _open(store, contract=contract)


def test_target_store_rejects_missing_campaign_admission(tmp_path: Path) -> None:
    contract = load_round27_model_contract(_ROOT)
    with Round27TargetStore(tmp_path / "targets.duckdb") as store:
        with pytest.raises(ValueError, match="campaign admission SHA-256 differs"):
            store.open_role(
                role="train",
                slot_id="stage1-a",
                run_id="stage1-a-run",
                contract=contract,
                feature_store_audit_sha256="b" * 64,
                campaign_admission={},
                role_intervals=(
                    {
                        "role": "train",
                        "slot_id": "stage1-a",
                        "start_ms": _START_MS,
                        "end_ms": _START_MS + 300_000,
                    },
                ),
                feature_rows=(_row(),),
                markets=((_market(), "c" * 64),),
                opened_at_ms=_OPENED_MS,
            )


def test_target_store_rejects_wrong_admitted_role_population(tmp_path: Path) -> None:
    contract = load_round27_model_contract(_ROOT)
    admission = _admission(contract, role="train")
    admission["role_population_sha256"]["train"] = "0" * 64
    body = dict(admission)
    body.pop("admission_sha256")
    admission["admission_sha256"] = _sha(_canonical(body))

    with Round27TargetStore(tmp_path / "targets.duckdb") as store:
        with pytest.raises(ValueError, match="target role population differs"):
            _open(store, contract=contract, admission=admission)
