from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
)
from simple_ai_trading.polymarket_round21_dataset import Round21CausalFeatureRow
import simple_ai_trading.polymarket_round24_receipt_lead_lag as round24


REPOSITORY = Path(__file__).resolve().parents[1]
CHAIN = hashlib.sha256(b"round24-test").hexdigest()


def _values(names: tuple[str, ...], updates: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(updates.get(name, 0.0)) for name in names)


def _row(
    *,
    condition_id: str,
    event_start_ms: int,
    decision_time_ms: int,
    market_prior: float,
    signal: float,
) -> Round21CausalFeatureRow:
    core = _values(
        POLYMARKET_ROUND21_FEATURE_SCHEMA.core_names,
        {
            "core.elapsed_fraction": 0.5,
            "core.up_bid_depth_5": 10.0,
            "core.up_ask_depth_5": 8.0,
            "core.down_bid_depth_5": 8.0,
            "core.down_ask_depth_5": 10.0,
            "core.up_microprice": market_prior,
            "core.down_microprice": 1.0 - market_prior,
            "core.normalized_market_prior_up": market_prior,
        },
    )
    spot = _values(
        POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names,
        {
            "spot.trade_log_return_1000ms": signal / 10_000.0,
            "spot.trade_log_return_5000ms": signal / 20_000.0,
            "spot.trade_log_return_15000ms": signal / 30_000.0,
            "spot.book_receipt_age_ms": 10.0,
        },
    )
    usdm = _values(
        POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names,
        {
            "usdm.trade_log_return_1000ms": signal / 10_000.0,
            "usdm.trade_log_return_5000ms": signal / 20_000.0,
            "usdm.trade_log_return_15000ms": signal / 30_000.0,
            "usdm.book_receipt_age_ms": 12.0,
            "usdm.spot_minus_usdm_book_receipt_skew_ms": -2.0,
        },
    )
    return Round21CausalFeatureRow.create(
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        decision_time_ms=decision_time_ms,
        structural_probability=0.5,
        market_prior_probability=market_prior,
        core_values=core,
        spot_values=spot,
        usdm_values=usdm,
        spot_available=True,
        usdm_available=True,
        feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
        core_source_chain_sha256=CHAIN,
        spot_source_chain_sha256=CHAIN,
        usdm_source_chain_sha256=CHAIN,
        core_maximum_receipt_ms=decision_time_ms - 5,
        spot_maximum_receipt_ms=decision_time_ms - 4,
        usdm_maximum_receipt_ms=decision_time_ms - 3,
    )


def _condition_for_fold(fold: int, *, offset: int) -> str:
    for value in range(offset, offset + 100_000):
        condition = "0x" + f"{value:064x}"
        if round24._fold(condition, 8) == fold:  # noqa: SLF001
            return condition
    raise AssertionError("unable to construct Round 24 test fold")


def test_round24_spec_is_future_only_target_free_and_compute_bounded() -> None:
    spec = round24.load_round24_receipt_lead_lag_spec(REPOSITORY)

    assert spec["specification_sha256"] == round24.POLYMARKET_ROUND24_SPEC_SHA256
    assert spec["data"]["fresh_condition_start_minimum_ms"] == 1786406400000
    assert spec["data"]["resolution_source"].endswith("btc-usd-twap-30s-streams")
    assert spec["data"]["legacy_core_data_reused"] is False
    assert set(spec["authority"].values()) == {False}
    assert spec["knowledge_at_freeze"]["new_core_capture_started"] is False
    assert spec["knowledge_at_freeze"]["round24_targets_constructed"] is False
    assert spec["compute_strategy"]["condition_fold_count"] == 8
    assert spec["compute_strategy"]["bootstrap_batch_draws_maximum"] == 5_000


def test_round24_vectors_map_receipt_time_features_with_correct_units() -> None:
    row = _row(
        condition_id="0x" + ("1" * 64),
        event_start_ms=1785801600000,
        decision_time_ms=1785801721000,
        market_prior=0.6,
        signal=2.0,
    )

    baseline, candidate = round24._vectors(row)  # noqa: SLF001

    assert len(baseline) == 16
    assert len(candidate) == 33
    assert math.isclose(baseline[0], math.log(1.5))
    assert math.isclose(baseline[3], 1.0 / 9.0)
    assert math.isclose(baseline[4], -1.0 / 9.0)
    assert candidate[16:18] == (2.0, 1.0)
    assert math.isclose(candidate[18], 2.0 / 3.0)
    assert candidate[-3:] == (10.0, 12.0, -2.0)


def test_round24_synthetic_prospective_gate_recovers_incremental_signal() -> None:
    spec = deepcopy(round24.load_round24_receipt_lead_lag_spec(REPOSITORY))
    spec["minimum_population"] = {
        "train_conditions": 8,
        "tune_calibration_conditions": 8,
        "tune_selection_conditions": 8,
    }
    spec["evaluation"]["bootstrap_draws"] = 2_000
    spec["compute_strategy"]["bootstrap_batch_draws_maximum"] = 200
    rows: list[Round21CausalFeatureRow] = []
    intervals = {item["role"]: item for item in spec["partitions"]}
    for role_index, role in enumerate(("train", "tune_calibration", "tune_selection")):
        start = int(intervals[role]["start_ms"])
        for fold in range(8):
            condition = _condition_for_fold(
                fold,
                offset=1 + role_index * 1_000_000,
            )
            event_start = start + fold * 300_000
            decision = event_start + 121_000
            signal = -2.0 if fold % 2 else 2.0
            future_logit = signal * 0.08
            future_prior = 1.0 / (1.0 + math.exp(-future_logit))
            rows.extend(
                (
                    _row(
                        condition_id=condition,
                        event_start_ms=event_start,
                        decision_time_ms=decision,
                        market_prior=0.5,
                        signal=signal,
                    ),
                    _row(
                        condition_id=condition,
                        event_start_ms=event_start,
                        decision_time_ms=decision + 1_000,
                        market_prior=future_prior,
                        signal=signal,
                    ),
                )
            )

    claims: list[dict[str, object]] = []

    def claim(value: Mapping[str, object]) -> str:
        claims.append(dict(value))
        return str(value["claim_sha256"])

    result = round24.run_round24_receipt_lead_lag(
        spec=spec,
        rows=rows,
        claim_selection=claim,
    )

    assert result["mechanism_gate_passed"] is True
    assert result["selection"]["mse_relative_improvement"] > 0.9
    assert len(claims) == 1
    assert claims[0]["status"] == (
        "models_and_calibration_frozen_before_selection_target_construction"
    )
    assert result["dataset"]["development"]["official_resolution_accessed"] is False
    assert result["dataset"]["selection"]["official_resolution_accessed"] is False


def test_round24_source_has_no_outcome_or_execution_dependency() -> None:
    source = Path(round24.__file__).read_text(encoding="utf-8")

    assert "load_round21_official_outcomes" not in source
    assert "polymarket_round21_operator" not in source
    assert "Round21PartitionPolicy" not in source
    assert "load_round21_core_development_publication" not in source
    assert "polymarket_live" not in source
    assert "create_order" not in source
    assert "place_order" not in source


def test_round24_core_assembly_operates_without_optional_binance_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = round24.load_round24_receipt_lead_lag_spec(REPOSITORY)
    data = spec["data"]
    row = _row(
        condition_id="0x" + "9" * 64,
        event_start_ms=int(data["campaign_start_ms"]),
        decision_time_ms=int(data["campaign_start_ms"]) + 30_000,
        market_prior=0.5,
        signal=0.0,
    )
    transport = {
        "manifest_sha256": "a" * 64,
        "source_plan_sha256": data["core_campaign_plan_sha256"],
        "source_capture_design_sha256": data["core_campaign_design_sha256"],
        "resolution_source": data["resolution_source"],
        "campaign_start_ms": data["campaign_start_ms"],
        "campaign_end_ms": data["campaign_end_ms"],
    }
    monkeypatch.setattr(
        round24,
        "validate_round25_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        round24,
        "materialize_round25_round24_core",
        lambda **_kwargs: ((object(),), {"materialization": True}, {"audit": True}),
    )
    monkeypatch.setattr(round24, "_core_rows", lambda _snapshots: (row,))

    loaded_spec, rows, evidence = round24.assemble_round24_receipt_rows(
        repository=REPOSITORY,
        core_database="unused.duckdb",
        terminal_transport_manifest=transport,
        sidecar_database=None,
        sidecar_terminal_manifest=None,
    )

    assert loaded_spec["specification_sha256"] == spec["specification_sha256"]
    assert rows == (row,)
    assert evidence["optional_sidecar_joined"] is False
    assert evidence["core_materialization"] == {"materialization": True}

    with pytest.raises(ValueError, match="must be supplied together"):
        round24.assemble_round24_receipt_rows(
            repository=REPOSITORY,
            core_database="unused.duckdb",
            terminal_transport_manifest=transport,
            sidecar_database="sidecar.duckdb",
            sidecar_terminal_manifest=None,
        )
