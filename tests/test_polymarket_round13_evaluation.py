from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
)
from simple_ai_trading.polymarket_round13 import (
    PolymarketRound13Program,
    polymarket_round13_evaluation_gates,
    polymarket_round13_scenarios,
)
from simple_ai_trading import polymarket_round13_evaluation as evaluation


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-artifact.json"
)


def _program() -> PolymarketRound13Program:
    digest = "a" * 64
    return PolymarketRound13Program(
        contract={"contract_sha256": digest},
        contract_sha256=digest,
        model=load_round12_reference_from_round11_artifact(ARTIFACT),
        policy=polymarket_round12_primary_policy(),
        scenarios=polymarket_round13_scenarios(),
        evaluation_gates=polymarket_round13_evaluation_gates(),
        confirmation_capital_quote=Decimal("1000"),
    ).validated()


def _insert_resolution_row(store: PolymarketEvidenceStore, run_id: str) -> None:
    store.connect().execute(
        """
        INSERT INTO polymarket_resolution_evidence VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            "resolution",
            run_id,
            "schema",
            "condition",
            "market",
            "BTC",
            1,
            1,
            "token",
            "Up",
            "{}",
            "b" * 64,
            "{}",
            "c" * 64,
            "{}",
            "d" * 64,
        ],
    )


def test_round13_bootstrap_is_deterministic_and_platform_serialized() -> None:
    values = [float((index % 7) - 2) / 10 for index in range(160)]
    gates = polymarket_round13_evaluation_gates()

    first = evaluation._bootstrap_group_mean(
        values,
        series_name="determinism",
        gates=gates,
    )
    second = evaluation._bootstrap_group_mean(
        values,
        series_name="determinism",
        gates=gates,
    )

    assert first == second
    assert first["samples"] == 10_000
    assert first["block_length_groups"] == 12
    assert len(str(first["bootstrap_samples_sha256"])) == 64


@pytest.mark.parametrize("raw", ["NaN", '{"value":1,"value":2}'])
def test_round13_evaluation_rejects_noncanonical_json(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        evaluation._strict_json(raw, name="test evidence")


def test_round13_claim_consumes_preexisting_resolution_evidence(
    tmp_path: Path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "claim.duckdb") as store:
        _insert_resolution_row(store, "run")

        with pytest.raises(ValueError, match="consumed by resolution evidence"):
            evaluation._open_claim(
                store,
                run_id="run",
                pipeline_report_sha256="b" * 64,
                scenario_dataset_ids=("c" * 64,),
                program=_program(),
            )
        claim = (
            store.connect()
            .execute(
                """
            SELECT status, report_sha256, error
            FROM polymarket_round13_evaluation_claim
            """
            )
            .fetchone()
        )

        with pytest.raises(ValueError, match="already consumed"):
            evaluation._open_claim(
                store,
                run_id="run",
                pipeline_report_sha256="b" * 64,
                scenario_dataset_ids=("c" * 64,),
                program=_program(),
            )

    assert claim == (
        "failed",
        "",
        "preexisting_resolution_evidence_before_one_use_claim",
    )


def test_round13_claim_opens_before_any_resolution_row(tmp_path: Path) -> None:
    with PolymarketEvidenceStore(tmp_path / "open.duckdb") as store:
        opened_at_ms, report = evaluation._open_claim(
            store,
            run_id="run",
            pipeline_report_sha256="b" * 64,
            scenario_dataset_ids=("c" * 64,),
            program=_program(),
        )
        claim = (
            store.connect()
            .execute("SELECT status, error FROM polymarket_round13_evaluation_claim")
            .fetchone()
        )
        resumed_at_ms, resumed_report = evaluation._open_claim(
            store,
            run_id="run",
            pipeline_report_sha256="b" * 64,
            scenario_dataset_ids=("c" * 64,),
            program=_program(),
        )

    assert opened_at_ms > 0
    assert report is None
    assert resumed_at_ms == opened_at_ms
    assert resumed_report is None
    assert claim == ("opened", "")


def test_round13_control_gate_requires_non_tied_conditions() -> None:
    groups = list(range(160))
    equity = [
        {
            "event_start_ms": group,
            "group_utility_quote": 0.1,
            "cumulative_utility_quote": (group + 1) * 0.1,
            "drawdown_quote": 0.0,
        }
        for group in groups
    ]
    conditions = [
        {
            "condition_id": f"condition-{group}",
            "asset": ("BTC", "ETH", "SOL")[group % 3],
            "event_start_ms": group,
            "utility_quote": 0.1,
        }
        for group in groups
    ]
    treatment: dict[str, object] = {
        "scenario": "primary",
        "total_utility_quote": 16.0,
        "equity": equity,
        "per_condition_utility": conditions,
        "gate_reasons_without_control": [],
    }
    control = {
        "total_utility_quote": 16.0,
        "equity": equity,
        "per_condition_utility": conditions,
    }

    evaluation._paired_control_gate(
        treatment,
        control,
        polymarket_round13_evaluation_gates(),
    )

    assert treatment["gate_passed"] is False
    assert (
        "insufficient_non_tied_treatment_control_conditions"
        in treatment["gate_reasons"]
    )
    assert treatment["control_comparison"]["non_tied_condition_count"] == 0


def test_round13_policy_metrics_separate_group_exposure_from_cumulative_turnover() -> (
    None
):
    batches = []
    winners: dict[str, str] = {}
    for group in range(160):
        snapshots = []
        attempts = []
        for asset in ("BTC", "ETH", "SOL"):
            condition = f"{group}-{asset}"
            winners[condition] = "Up"
            snapshots.append(
                SimpleNamespace(
                    condition_id=condition,
                    asset=asset,
                    event_start_ms=group,
                )
            )
        if group % 5 == 0 and group // 5 < 30:
            asset = ("BTC", "ETH", "SOL")[(group // 5) % 3]
            condition = f"{group}-{asset}"
            attempts.append(
                SimpleNamespace(
                    scenario="primary",
                    policy="calibrated",
                    condition_id=condition,
                    event_start_ms=group,
                    asset=asset,
                    outcome="Up",
                    remaining_seconds=180.0,
                    minimum_signed_quantity="5",
                    observation=SimpleNamespace(
                        observation_state="simulated_fill",
                        entry_modeled_quantity="5",
                        entry_cost_quote="4",
                        maximum_entry_loss_quote="5.1",
                    ),
                )
            )
        batches.append(
            SimpleNamespace(
                calibration_snapshots=tuple(snapshots),
                attempts=tuple(attempts),
                abstention_counts={},
            )
        )

    treatment = evaluation._policy_metrics(
        batches,  # type: ignore[arg-type]
        winners,
        scenario="primary",
        policy="calibrated",
        program=_program(),
    )
    with localcontext() as caller_context:
        caller_context.prec = 6
        caller_context.rounding = ROUND_CEILING
        changed_context = evaluation._policy_metrics(
            batches,  # type: ignore[arg-type]
            winners,
            scenario="primary",
            policy="calibrated",
            program=_program(),
        )
    assert changed_context == treatment
    control = evaluation._policy_metrics(
        batches,  # type: ignore[arg-type]
        winners,
        scenario="primary",
        policy="raw_market_prior",
        program=_program(),
    )
    evaluation._paired_control_gate(
        treatment,
        control,
        polymarket_round13_evaluation_gates(),
    )

    assert treatment["maximum_group_entry_exposure_quote"] == 4.0
    assert treatment["capital_deployed_quote"] == 120.0
    assert treatment["turnover_quote"] == 120.0
    assert treatment["simulated_filled_conditions"] == 30
    assert treatment["median_condition_utility_quote"] == 0.0
    assert treatment["median_simulated_filled_condition_utility_quote"] == 1.0
    assert treatment["control_comparison"]["non_tied_condition_count"] == 30
    assert treatment["gate_passed"] is True
