from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import (
    cli,
    polymarket_mlp as polymarket_mlp_module,
    polymarket_ridge as polymarket_ridge_module,
)
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.polymarket_action_value import (
    POLYMARKET_ACTION_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_mlp import (
    POLYMARKET_MLP_CONTRACT_SHA256,
    POLYMARKET_MLP_SEEDS,
    fit_and_evaluate_polymarket_mlp,
    materialize_polymarket_mlp_report,
)
from simple_ai_trading.polymarket_fit_claim import (
    PolymarketFitClaim,
    consume_polymarket_fit_claim,
    reserve_polymarket_fit_claim,
)
from simple_ai_trading.polymarket_ridge import (
    POLYMARKET_RIDGE_CONTRACT_SHA256,
    POLYMARKET_RIDGE_L2_GRID,
    POLYMARKET_RIDGE_THRESHOLD_GRID,
    PolymarketRidgeObservation,
    build_polymarket_ridge_dataset,
    fit_and_evaluate_polymarket_ridge,
    load_polymarket_ridge_report,
    materialize_polymarket_ridge_report,
    split_polymarket_ridge_dataset,
)
from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore


_ASSETS = ("BTC", "ETH", "SOL")


def _digest(*values: object) -> str:
    return sha256("|".join(map(str, values)).encode("ascii")).hexdigest()


def _synthetic_observations(
    group_count: int = 30,
) -> tuple[PolymarketRidgeObservation, ...]:
    observations: list[PolymarketRidgeObservation] = []
    start = 1_800_000_000_000
    for group in range(group_count):
        event_start_ms = start + group * 300_000
        for asset_index, asset in enumerate(_ASSETS):
            condition_id = _digest("condition", group, asset)
            official_up = (group + asset_index) % 2 == 0
            source_sha = _digest("source", group, asset)
            for decision in range(2):
                decision_ns = (
                    group * 10_000 + asset_index * 2_000 + decision * 800
                ) * 1_000_000
                positive_outcome = (
                    "Up" if (group + asset_index + decision) % 2 == 0 else "Down"
                )
                for outcome in ("Up", "Down"):
                    positive = outcome == positive_outcome
                    signal = 3.0 if positive else -3.0
                    values = [0.0] * len(POLYMARKET_ACTION_FEATURE_NAMES)
                    values[0] = signal
                    values[1] = float(asset_index)
                    values[2] = float(decision)
                    observations.append(
                        PolymarketRidgeObservation(
                            action_feature_sha256=_digest(
                                "feature", group, asset, decision, outcome
                            ),
                            action_label_sha256=_digest(
                                "label", group, asset, decision, outcome
                            ),
                            source_feature_row_sha256=source_sha,
                            condition_id=condition_id,
                            asset=asset,
                            outcome=outcome,
                            event_start_ms=event_start_ms,
                            decision_received_monotonic_ns=decision_ns,
                            release_monotonic_ns=decision_ns + 400_000_000,
                            feature_values=tuple(values),
                            official_up=official_up,
                            classifier_eligible=True,
                            positive_complete=positive,
                            category="successful_round_trip",
                            condition_blocked=False,
                            stress_utility_quote=0.05 if positive else -0.05,
                        )
                    )
    return tuple(observations)


def _dataset(group_count: int = 30):
    return build_polymarket_ridge_dataset(
        pipeline_report_sha256="a" * 64,
        eligibility_sha256="b" * 64,
        observations=_synthetic_observations(group_count),
    )


def test_round9_ridge_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-009-ridge-implementation-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_RIDGE_CONTRACT_SHA256
    assert sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert tuple(contract["candidate_grid"]["l2"]) == POLYMARKET_RIDGE_L2_GRID
    assert (
        tuple(contract["candidate_grid"]["thresholds"])
        == POLYMARKET_RIDGE_THRESHOLD_GRID
    )
    assert contract["truth_constraints"]["profitability_claim"] is False


def test_round9_ridge_split_is_purged_grouped_and_broad() -> None:
    dataset = _dataset()
    split = split_polymarket_ridge_dataset(dataset)

    assert dataset.dataset_sha256 == polymarket_ridge_module._sha256(
        dataset.identity_payload()
    )
    assert len(split.train_groups) == 16
    assert len(split.validation_groups) == 6
    assert len(split.test_groups) == 6
    assert len(split.purged_groups) == 2
    assert max(split.train_groups) < split.purged_groups[0]
    assert split.purged_groups[0] < min(split.validation_groups)
    assert max(split.validation_groups) < split.purged_groups[1]
    assert split.purged_groups[1] < min(split.test_groups)


def test_round9_ridge_selects_only_from_validation_and_is_deterministic() -> None:
    dataset = _dataset()

    first = fit_and_evaluate_polymarket_ridge(dataset)
    second = fit_and_evaluate_polymarket_ridge(dataset)

    assert first.report_sha256 == second.report_sha256
    assert first.selected_policy == "ridge_logit"
    assert first.selected_threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
    assert not first.neural_challenger_authorized
    assert first.development_passed
    assert first.test_metrics.gate_passed
    assert first.test_metrics.completed_trade_count >= 30
    assert first.test_metrics.failed_exit_count == 0
    assert first.test_metrics.aggregate_stress_utility_quote > 0.0
    assert all(value > 0.0 for value in first.test_metrics.pnl_by_asset.values())
    assert not first.asdict()["profitability_claim"]
    assert not first.asdict()["trading_authority"]


def test_round9_ridge_report_rejects_rehashed_financial_inconsistency() -> None:
    report = fit_and_evaluate_polymarket_ridge(_dataset())
    inconsistent_metrics = replace(
        report.test_metrics,
        aggregate_stress_utility_quote=(
            report.test_metrics.aggregate_stress_utility_quote + 1.0
        ),
    )
    provisional = replace(
        report,
        test_metrics=inconsistent_metrics,
        report_sha256="",
    )
    tampered = replace(
        provisional,
        report_sha256=polymarket_ridge_module._sha256(provisional.identity_payload()),
    )

    with pytest.raises(ValueError, match="policy metrics are invalid"):
        tampered.validated()


def test_round9_ridge_metric_parser_rejects_boolean_counts() -> None:
    payload = fit_and_evaluate_polymarket_ridge(_dataset()).test_metrics.asdict()
    payload["attempt_count"] = True

    with pytest.raises(ValueError, match="stored Polymarket ridge metrics"):
        polymarket_ridge_module._ridge_metrics_from_payload(payload)


def test_round9_ridge_refuses_fewer_than_30_synchronized_groups() -> None:
    with pytest.raises(ValueError, match="insufficient synchronized groups:29/30"):
        split_polymarket_ridge_dataset(_dataset(29))


def test_round9_ridge_blocks_unproven_post_submission_entry_state() -> None:
    assert (
        polymarket_ridge_module._training_blocking_entry_terminal_counts(
            {
                "entry_not_filled": 4,
                "entry_confirmation_enters_excluded_close_window": 0,
                "missing_entry_execution_book": 0,
            }
        )
        == {}
    )
    assert polymarket_ridge_module._training_blocking_entry_terminal_counts(
        {
            "entry_confirmation_enters_excluded_close_window": 2,
            "missing_entry_execution_book": 1,
        }
    ) == {
        "entry_confirmation_enters_excluded_close_window": 2,
        "missing_entry_execution_book": 1,
    }
    for malformed in (
        None,
        {"missing_entry_execution_book": True},
        {"missing_entry_execution_book": "1"},
        {"missing_entry_execution_book": -1},
        {1: 0},
    ):
        with pytest.raises(ValueError, match="terminal counts are invalid"):
            polymarket_ridge_module._training_blocking_entry_terminal_counts(malformed)


def test_round9_ridge_materialization_is_idempotent_and_tamper_evident(
    tmp_path,
) -> None:
    dataset = _dataset()
    report = fit_and_evaluate_polymarket_ridge(dataset)
    with PolymarketEvidenceStore(tmp_path / "ridge.duckdb") as store:
        created = materialize_polymarket_ridge_report(store, dataset, report)
        existing = materialize_polymarket_ridge_report(store, dataset, report)
        loaded = load_polymarket_ridge_report(
            store,
            report_sha256=report.report_sha256,
        )
        loaded_materialization = (
            polymarket_ridge_module.load_polymarket_ridge_materialization(
                store,
                loaded,
            )
        )
        store.connect().execute(
            """
            UPDATE polymarket_ridge_selected_action
            SET stress_utility_quote = '999'
            WHERE report_sha256 = ? AND partition = 'test' AND sequence = 0
            """,
            [report.report_sha256],
        )
        with pytest.raises(ValueError, match="selected_action rows are inconsistent"):
            materialize_polymarket_ridge_report(store, dataset, report)

    assert created.status == "created"
    assert existing.status == "existing"
    assert loaded_materialization == existing
    assert loaded == report
    assert created.selected_test_action_count >= 30


def test_round9_ridge_cli_and_native_contract_share_the_frozen_input(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    dataset = _dataset()
    expected_report = fit_and_evaluate_polymarket_ridge(dataset)
    load_count = 0

    def load_dataset(*_args, **_kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            return dataset
        raise polymarket_ridge_module.PolymarketRidgeFitAlreadyComplete(
            expected_report.report_sha256
        )

    monkeypatch.setattr(cli, "load_polymarket_ridge_dataset", load_dataset)
    arguments = [
        "polymarket-ridge",
        "--database",
        str(tmp_path / "ridge-cli.duckdb"),
        "--pipeline-report-sha256",
        dataset.pipeline_report_sha256,
        "--memory-limit",
        "512MB",
        "--database-threads",
        "1",
        "--json",
    ]

    status = cli.main(arguments)
    payload = json.loads(capsys.readouterr().out)
    spec = next(item for item in command_specs() if item.name == "polymarket-ridge")

    assert status == 0
    assert payload["development_passed"] is True
    assert payload["profitability_claim"] is False
    assert payload["trading_authority"] is False
    assert payload["materialization"]["status"] == "created"
    assert {option.dest for option in spec.options} == {
        "database",
        "pipeline_report_sha256",
        "memory_limit",
        "database_threads",
        "json",
    }

    def unexpected_refit(*_args, **_kwargs):
        raise AssertionError("a completed ridge claim must never reopen test")

    monkeypatch.setattr(cli, "fit_and_evaluate_polymarket_ridge", unexpected_refit)
    repeated_status = cli.main(arguments)
    repeated_payload = json.loads(capsys.readouterr().out)

    assert repeated_status == 0
    assert repeated_payload["report_sha256"] == payload["report_sha256"]
    assert repeated_payload["materialization"]["status"] == "existing"


def test_round9_ridge_failed_claim_cannot_silently_retry(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    dataset = _dataset()
    database = tmp_path / "ridge-failed-claim.duckdb"
    monkeypatch.setattr(cli, "load_polymarket_ridge_dataset", lambda *_a, **_k: dataset)
    calls = 0

    def fail_after_claim(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("forced fit failure")

    monkeypatch.setattr(cli, "fit_and_evaluate_polymarket_ridge", fail_after_claim)
    arguments = [
        "polymarket-ridge",
        "--database",
        str(database),
        "--pipeline-report-sha256",
        dataset.pipeline_report_sha256,
        "--json",
    ]

    assert cli.main(arguments) == 2
    first_error = capsys.readouterr().err
    assert "forced fit failure" in first_error
    assert cli.main(arguments) == 2
    second_error = capsys.readouterr().err

    assert calls == 1
    assert "already claimed:state=failed" in second_error
    with PolymarketEvidenceStore(database) as store:
        state, failure_sha256 = (
            store.connect()
            .execute(
                """
            SELECT state, failure_sha256 FROM polymarket_model_fit_claim
            WHERE experiment = 'round9_ridge' AND parent_sha256 = ?
            """,
                [dataset.pipeline_report_sha256],
            )
            .fetchone()
        )
    assert state == "failed"
    assert len(failure_sha256) == 64


def test_round9_fit_claim_reservation_is_exact_and_single_use(tmp_path) -> None:
    identity = {
        "experiment": "round9_ridge",
        "parent_sha256": "a" * 64,
        "contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
        "dataset_sha256": "b" * 64,
        "report_table": "polymarket_ridge_report",
        "report_parent_column": "pipeline_report_sha256",
    }
    with PolymarketEvidenceStore(tmp_path / "reserved-claim.duckdb") as store:
        reserved = reserve_polymarket_fit_claim(store, **identity)
        consumed = consume_polymarket_fit_claim(store, **identity)

        assert reserved.status == "claimed"
        assert consumed == reserved
        with pytest.raises(ValueError, match="already claimed:state=started"):
            consume_polymarket_fit_claim(store, **identity)


def test_round9_opaque_identity_matches_clear_dataset_digest() -> None:
    continuity_payload = {
        "confirmation_eligible": True,
        "outcomes_consulted": False,
        "labels_consulted": False,
        "model_scores_consulted": False,
    }
    eligibility_sha256 = polymarket_ridge_module._sha256(continuity_payload)
    dataset = build_polymarket_ridge_dataset(
        pipeline_report_sha256="a" * 64,
        eligibility_sha256=eligibility_sha256,
        observations=_synthetic_observations(),
    )
    action_dataset_sha256 = "c" * 64
    run_report_sha256 = "d" * 64
    implementation_sha256 = (
        polymarket_ridge_module.polymarket_action_pipeline_implementation_sha256()
    )
    pipeline_payload = {
        "run_id": "opaque-run",
        "run_report_sha256": run_report_sha256,
        "eligibility_sha256": dataset.eligibility_sha256,
        "implementation_sha256": implementation_sha256,
        "batches": [{"action_dataset_sha256": action_dataset_sha256}],
    }
    pipeline_report_sha256 = polymarket_ridge_module._sha256(pipeline_payload)
    pipeline_json = json.dumps(
        {**pipeline_payload, "report_sha256": pipeline_report_sha256}
    )
    continuity_json = json.dumps(
        {
            **continuity_payload,
            "report_sha256": eligibility_sha256,
        }
    )

    class Cursor:
        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchmany(self, size):
            batch, self.rows = self.rows[:size], self.rows[size:]
            return batch

    class Connection:
        def execute(self, statement, _parameters=None):
            if "FROM polymarket_action_value_pipeline" in statement:
                return Cursor(
                    [
                        (
                            pipeline_json,
                            polymarket_ridge_module.POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION,
                            polymarket_ridge_module.POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
                            "opaque-run",
                            run_report_sha256,
                            dataset.eligibility_sha256,
                            json.dumps([action_dataset_sha256]),
                            implementation_sha256,
                        )
                    ]
                )
            if "FROM polymarket_continuity_eligibility_report" in statement:
                return Cursor([(continuity_json, "opaque-run", run_report_sha256)])
            if "SELECT a.action_feature_sha256" in statement:
                return Cursor(
                    [
                        (
                            item.action_feature_sha256,
                            item.action_label_sha256,
                            item.source_feature_row_sha256,
                        )
                        for item in dataset.observations
                    ]
                )
            return Cursor()

        def executemany(self, _statement, _parameters):
            return None

    store = SimpleNamespace(connect=lambda: Connection())
    identity = polymarket_ridge_module.load_polymarket_ridge_dataset_identity(
        store,
        pipeline_report_sha256=pipeline_report_sha256,
    )
    expected_payload = dataset.identity_payload()
    expected_payload["pipeline_report_sha256"] = pipeline_report_sha256

    assert identity.observation_count == len(dataset.observations)
    assert identity.dataset_sha256 == polymarket_ridge_module._sha256(expected_payload)


def test_round9_parent_loader_reserves_mlp_before_clear_labels(monkeypatch) -> None:
    dataset = _dataset()
    parent = fit_and_evaluate_polymarket_ridge(dataset)
    events: list[str] = []

    class Connection:
        def execute(self, _statement, _parameters=None):
            return SimpleNamespace(
                fetchone=lambda: (
                    dataset.pipeline_report_sha256,
                    dataset.eligibility_sha256,
                    dataset.dataset_sha256,
                    POLYMARKET_RIDGE_CONTRACT_SHA256,
                )
            )

    store = SimpleNamespace(connect=lambda: Connection())

    def reserve(*_args, **_kwargs):
        events.append("reserve")
        return PolymarketFitClaim(
            experiment="round9_mlp",
            status="claimed",
            parent_sha256=parent.report_sha256,
            dataset_sha256=dataset.dataset_sha256,
            report_sha256="",
        )

    def load_dataset(*_args, **_kwargs):
        events.append("clear-labels")
        return dataset

    monkeypatch.setattr(
        polymarket_ridge_module, "reserve_polymarket_fit_claim", reserve
    )
    monkeypatch.setattr(
        polymarket_ridge_module,
        "_load_polymarket_ridge_dataset_after_claim",
        load_dataset,
    )
    monkeypatch.setattr(
        polymarket_ridge_module,
        "load_polymarket_ridge_report",
        lambda *_args, **_kwargs: parent,
    )
    monkeypatch.setattr(
        polymarket_ridge_module,
        "materialize_polymarket_ridge_report",
        lambda *_args, **_kwargs: SimpleNamespace(status="existing"),
    )

    loaded = polymarket_ridge_module.load_polymarket_ridge_evidence(
        store,
        report_sha256=parent.report_sha256,
    )

    assert loaded == (dataset, parent)
    assert events == ["reserve", "clear-labels"]


def test_round9_completed_claims_do_not_reopen_clear_labels(monkeypatch) -> None:
    dataset = _dataset()
    parent = fit_and_evaluate_polymarket_ridge(dataset)
    identity = polymarket_ridge_module.PolymarketRidgeDatasetIdentity(
        pipeline_report_sha256=dataset.pipeline_report_sha256,
        eligibility_sha256=dataset.eligibility_sha256,
        observation_count=len(dataset.observations),
        dataset_sha256=dataset.dataset_sha256,
    )

    class Connection:
        def execute(self, _statement, _parameters=None):
            return SimpleNamespace(
                fetchone=lambda: (
                    dataset.pipeline_report_sha256,
                    dataset.eligibility_sha256,
                    dataset.dataset_sha256,
                    POLYMARKET_RIDGE_CONTRACT_SHA256,
                )
            )

    store = SimpleNamespace(connect=lambda: Connection())

    def unexpected_clear_load(*_args, **_kwargs):
        raise AssertionError("a completed fit must not reconstruct clear labels")

    monkeypatch.setattr(
        polymarket_ridge_module,
        "load_polymarket_ridge_dataset_identity",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        polymarket_ridge_module,
        "reserve_polymarket_fit_claim",
        lambda *_args, **_kwargs: PolymarketFitClaim(
            experiment="round9_ridge",
            status="existing",
            parent_sha256=dataset.pipeline_report_sha256,
            dataset_sha256=dataset.dataset_sha256,
            report_sha256=parent.report_sha256,
        ),
    )
    monkeypatch.setattr(
        polymarket_ridge_module,
        "_load_polymarket_ridge_dataset_after_claim",
        unexpected_clear_load,
    )
    with pytest.raises(
        polymarket_ridge_module.PolymarketRidgeFitAlreadyComplete,
        match=parent.report_sha256,
    ):
        polymarket_ridge_module.load_polymarket_ridge_dataset(
            store,
            pipeline_report_sha256=dataset.pipeline_report_sha256,
        )

    monkeypatch.setattr(
        polymarket_ridge_module,
        "reserve_polymarket_fit_claim",
        lambda *_args, **_kwargs: PolymarketFitClaim(
            experiment="round9_mlp",
            status="existing",
            parent_sha256=parent.report_sha256,
            dataset_sha256=dataset.dataset_sha256,
            report_sha256="f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="Polymarket MLP is already complete"):
        polymarket_ridge_module.load_polymarket_ridge_evidence(
            store,
            report_sha256=parent.report_sha256,
        )


def test_round9_mlp_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-009-causal-mlp-challenger-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_MLP_CONTRACT_SHA256
    assert sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert contract["fit"]["architecture"].startswith("One fixed 39 -> 64")
    assert contract["status"].startswith("frozen_before_any_round_9_neural")


def test_round9_mlp_refuses_insufficient_group_breadth_before_training() -> None:
    dataset = _dataset(59)
    parent = fit_and_evaluate_polymarket_ridge(dataset)

    with pytest.raises(ValueError, match="insufficient synchronized groups:59/60"):
        fit_and_evaluate_polymarket_mlp(dataset, parent, compute_backend="cpu")


def test_round9_mlp_refuses_short_test_partition_before_runtime(monkeypatch) -> None:
    dataset = _dataset(90)
    parent = fit_and_evaluate_polymarket_ridge(dataset)

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("MLP runtime must stay closed for an undersized test")

    monkeypatch.setattr(polymarket_mlp_module, "_torch_runtime", unexpected_runtime)
    with pytest.raises(ValueError, match="insufficient untouched test groups:18/30"):
        fit_and_evaluate_polymarket_mlp(dataset, parent, compute_backend="cpu")


def test_round9_mlp_refuses_silent_explicit_backend_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        polymarket_mlp_module,
        "resolve_backend",
        lambda _requested: BackendInfo(
            requested="directml",
            kind="cpu",
            device="cpu",
            vendor="Python stdlib",
            reason="DirectML unavailable",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="requested compute backend 'directml' is unavailable",
    ):
        polymarket_mlp_module._torch_runtime("directml")


def test_round9_mlp_is_reproducible_and_keeps_test_closed_without_admission(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    dataset = _dataset(150)
    parent = fit_and_evaluate_polymarket_ridge(dataset)
    progress: list[tuple[str, dict[str, object]]] = []

    report = fit_and_evaluate_polymarket_mlp(
        dataset,
        parent,
        compute_backend="cpu",
        progress=lambda phase, payload: progress.append((phase, dict(payload))),
    )

    assert tuple(item.seed for item in report.ensemble.members) == POLYMARKET_MLP_SEEDS
    assert report.ensemble.backend.kind == "cpu"
    assert report.ensemble.backend.canonical_replay_max_probability_drift <= 0.00001
    assert "preflight_seconds" in report.ensemble.backend.asdict()
    assert "training_seconds" in report.ensemble.backend.asdict()
    assert "preflight_seconds" not in report.ensemble.identity_payload()["backend"]
    assert "training_seconds" not in report.ensemble.identity_payload()["backend"]
    assert report.ensemble.reproducibility_max_probability_drift <= 0.00001
    assert report.validation_log_loss < report.ridge_validation_log_loss
    assert not report.test_evaluated
    assert not report.development_passed
    assert report.validation_stress_utility_uplift_quote == pytest.approx(0.0)
    assert "validation_stress_utility_not_above_ridge" in (
        report.validation_admission_reasons
    )
    assert not any(
        reason.startswith("untouched_test_group_count:")
        for reason in report.validation_admission_reasons
    )
    assert report.test_gate_reasons == ()
    assert report.asdict()["foundation_ai_authorized"] is False
    assert report.asdict()["profitability_claim"] is False
    assert report.asdict()["trading_authority"] is False
    inconsistent_bootstrap = replace(
        report.validation_log_loss_uplift,
        sample_count=report.validation_log_loss_uplift.sample_count + 1,
    )
    provisional = replace(
        report,
        validation_log_loss_uplift=inconsistent_bootstrap,
        report_sha256="",
    )
    inconsistent_report = replace(
        provisional,
        report_sha256=polymarket_mlp_module._sha256(provisional.identity_payload()),
    )
    with pytest.raises(ValueError, match="bootstrap evidence is invalid"):
        inconsistent_report.validated()
    assert {phase for phase, _payload in progress} >= {
        "polymarket_mlp_preflight",
        "polymarket_mlp_seed",
        "polymarket_mlp_epoch",
        "polymarket_mlp_reproducibility",
        "polymarket_mlp_validation",
    }
    assert "polymarket_mlp_test" not in {phase for phase, _payload in progress}
    with PolymarketEvidenceStore(tmp_path / "mlp.duckdb") as store:
        created = materialize_polymarket_mlp_report(store, dataset, parent, report)
        existing = materialize_polymarket_mlp_report(store, dataset, parent, report)
        runtime_count = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_mlp_runtime_evidence")
            .fetchone()[0]
        )
        store.connect().execute(
            """
            UPDATE polymarket_mlp_prediction SET probability = 0.123
            WHERE report_sha256 = ? AND partition = 'validation' AND sequence = 0
            """,
            [report.report_sha256],
        )
        with pytest.raises(ValueError, match="prediction rows are inconsistent"):
            materialize_polymarket_mlp_report(store, dataset, parent, report)

    assert created.status == "created"
    assert existing.status == "existing"
    assert runtime_count == 1
    assert created.validation_prediction_count > 0
    assert created.test_prediction_count == 0
    monkeypatch.setattr(
        cli,
        "load_polymarket_ridge_evidence",
        lambda *_args, **_kwargs: (dataset, parent),
    )
    monkeypatch.setattr(
        cli,
        "fit_and_evaluate_polymarket_mlp",
        lambda *_args, **_kwargs: report,
    )
    arguments = [
        "polymarket-mlp",
        "--database",
        str(tmp_path / "mlp-cli.duckdb"),
        "--ridge-report-sha256",
        parent.report_sha256,
        "--compute-backend",
        "cpu",
        "--memory-limit",
        "512MB",
        "--database-threads",
        "1",
        "--json",
    ]
    status = cli.main(arguments)
    payload = json.loads(capsys.readouterr().out)
    spec = next(item for item in command_specs() if item.name == "polymarket-mlp")

    assert status == 2
    assert payload["development_passed"] is False
    assert payload["materialization"]["status"] == "created"
    assert {option.dest for option in spec.options} == {
        "database",
        "ridge_report_sha256",
        "compute_backend",
        "memory_limit",
        "database_threads",
        "json",
    }

    def unexpected_refit(*_args, **_kwargs):
        raise AssertionError("a completed MLP claim must never reopen test")

    monkeypatch.setattr(cli, "fit_and_evaluate_polymarket_mlp", unexpected_refit)
    assert cli.main(arguments) == 2
    repeated = capsys.readouterr()

    assert repeated.out == ""
    assert "Polymarket MLP is already complete" in repeated.err
