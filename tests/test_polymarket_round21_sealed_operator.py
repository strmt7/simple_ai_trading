from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.polymarket_ai_veto import (
    POLYMARKET_AI_REPORT_SCHEMA_VERSION,
    PolymarketAIVetoCase,
    PolymarketAIVetoConfig,
)
from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
)
from simple_ai_trading.polymarket_round21_dataset import (
    Round21CausalFeatureRow,
    Round21OfficialOutcome,
    Round21PartitionPolicy,
    build_round21_development_panel,
    build_round21_sealed_test_panel,
)
import simple_ai_trading.polymarket_round21_economic_operator as economic_module
import simple_ai_trading.polymarket_round21_sealed_operator as operator_module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _sealed_inputs():
    campaign_start = 1_800_000_000_000
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=campaign_start,
        campaign_end_ms=campaign_start + 2_592_000_000,
    )
    event_starts = (
        campaign_start + 1_989_000_000,
        campaign_start + 1_989_300_000,
    )
    rows = []
    outcomes = []
    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA
    for index, event_start in enumerate(event_starts):
        condition_id = "0x" + format(index + 1, "064x")
        decision_time = event_start + 120_000
        rows.append(
            Round21CausalFeatureRow.create(
                condition_id=condition_id,
                event_start_ms=event_start,
                decision_time_ms=decision_time,
                structural_probability=0.51,
                market_prior_probability=0.49,
                core_values=(0.0,) * len(schema.core_names),
                spot_values=(0.0,) * len(schema.spot_names),
                usdm_values=(0.0,) * len(schema.usdm_names),
                spot_available=False,
                usdm_available=False,
                feature_schema=schema,
                core_source_chain_sha256=_sha(f"core-{index}"),
                spot_source_chain_sha256=hashlib.sha256(b"").hexdigest(),
                usdm_source_chain_sha256=hashlib.sha256(b"").hexdigest(),
                core_maximum_receipt_ms=decision_time,
            )
        )
        outcomes.append(
            Round21OfficialOutcome.create(
                condition_id=condition_id,
                event_start_ms=event_start,
                resolved_up=bool(index),
                observed_at_ms=event_start + 300_100,
                source="polymarket_clob_gamma_consensus",
                source_payload_sha256=_sha(f"outcome-{index}"),
            )
        )
    return policy, tuple(rows), tuple(outcomes)


def _sealed_panel(*, access: str = "access"):
    policy, rows, outcomes = _sealed_inputs()
    return build_round21_sealed_test_panel(
        feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
        partition_policy=policy,
        feature_rows=rows,
        outcomes=outcomes,
        claim_sha256=_sha("claim"),
        test_access_sha256=_sha(access),
        sealed_test_population_manifest_sha256=_sha("population"),
    )


def test_round21_sealed_panel_requires_and_binds_consumed_access() -> None:
    first = _sealed_panel(access="first")
    second = _sealed_panel(access="second")

    assert first.role == "test"
    assert first.dataset_sha256 != second.dataset_sha256
    assert first.target_manifest_sha256 != second.target_manifest_sha256
    policy, rows, outcomes = _sealed_inputs()
    with pytest.raises(ValueError, match="sealed-test access identity"):
        build_round21_sealed_test_panel(
            feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
            partition_policy=policy,
            feature_rows=rows,
            outcomes=outcomes,
            claim_sha256="bad",
            test_access_sha256=_sha("access"),
            sealed_test_population_manifest_sha256=_sha("population"),
        )
    with pytest.raises(ValueError, match="invalid or sealed"):
        build_round21_development_panel(
            role="test",
            feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
            partition_policy=policy,
            feature_rows=rows,
            outcomes=outcomes,
        )


def test_round21_sealed_replay_disables_development_identity_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _sealed_panel()
    observed: dict[str, object] = {}
    replay = SimpleNamespace(
        result_sha256=_sha("replay"),
        source_condition_count=1,
        validated=lambda: None,
    )
    replay.validated = lambda: replay
    case = PolymarketAIVetoCase(
        case_id="case-1",
        condition_id="0x" + "1" * 64,
        sample_id="sample-1",
        asset="BTC",
        event_start_ms=1_800_000_000_000,
        decision_received_wall_ms=1_800_000_000_100,
        decision_received_monotonic_ns=1,
        prompt_payload={"schema_version": "fixture"},
        case_sha256=_sha("case"),
    )

    def fake_replay(**kwargs):
        observed.update(kwargs)
        return replay, (case,)

    monkeypatch.setattr(
        economic_module,
        "_replay_round21_development_economics",
        fake_replay,
    )
    evidence = economic_module.replay_round21_sealed_economics_with_ai_cases(
        source_database="closed.duckdb",
        terminal_transport_manifest={},
        partition_policy=_sealed_inputs()[0],
        test_panel=panel,
        development_model_artifact={},
        core_publication_manifest_sha256=_sha("publication"),
        claim_sha256=_sha("claim"),
        test_access_sha256=_sha("access"),
        sealed_test_population_manifest_sha256=_sha("population"),
    )

    assert observed["expected_roles"] == ("test",)
    assert observed["verify_artifact_population"] is False
    assert evidence.test_dataset_sha256 == panel.dataset_sha256
    assert evidence.test_target_manifest_sha256 == panel.target_manifest_sha256
    with pytest.raises(ValueError, match="sealed economic replay evidence differs"):
        replace(
            evidence,
            test_target_manifest_sha256=_sha("different-target-manifest"),
        ).validated()


def test_round21_sealed_operator_preflights_files_before_consuming_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claim = SimpleNamespace(
        pretest_manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        nominated_ai_model=None,
        validated=lambda: None,
    )
    claim.validated = lambda: claim
    pretest = SimpleNamespace(
        manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        development_model_artifact_sha256=_sha("model"),
        core_corpus_publication_manifest_sha256=_sha("publication"),
        optional_campaign_terminal_sha256=None,
        nominated_ai_comparison_sha256=None,
        nominated_ai_model=None,
        nominated_ai_model_digest=None,
        validated=lambda: None,
    )
    pretest.validated = lambda: pretest
    monkeypatch.setattr(
        operator_module,
        "validate_round21_development_artifact",
        lambda _value: {"artifact_sha256": _sha("model")},
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_core_publication_boundary",
        lambda _value: {
            "manifest_sha256": _sha("publication"),
            "sealed_test_population_manifest_sha256": _sha("population"),
            "terminal_transport_manifest_sha256": _sha("transport"),
        },
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda _value: {"manifest_sha256": _sha("transport")},
    )
    consumed = False

    def fail_if_consumed(**_kwargs):
        nonlocal consumed
        consumed = True
        raise AssertionError("test access was consumed")

    monkeypatch.setattr(operator_module, "execute_round21_one_use", fail_if_consumed)
    with pytest.raises(ValueError, match="source database is unavailable"):
        operator_module.evaluate_round21_terminal_sealed_once(
            store_path=tmp_path / "one-use.sqlite3",
            claim=claim,
            pretest=pretest,
            publication_directory=tmp_path / "publication",
            source_database=tmp_path / "missing.duckdb",
            terminal_transport_manifest={},
            development_model_artifact={},
        )
    assert consumed is False


def test_round21_sealed_ai_report_identity_rejects_tampering() -> None:
    config = PolymarketAIVetoConfig(model="qwen3.5:9b").validated()
    payload = {
        "schema_version": POLYMARKET_AI_REPORT_SCHEMA_VERSION,
        "config": config.asdict(),
        "model_digest": _sha("model"),
    }
    report_sha256 = operator_module._canonical_sha256(payload)
    report = SimpleNamespace(
        schema_version=POLYMARKET_AI_REPORT_SCHEMA_VERSION,
        config=config,
        report_sha256=report_sha256,
        advisory_only=True,
        trading_authority=False,
        profitability_claim=False,
        asdict=lambda: {**payload, "report_sha256": report_sha256},
    )

    assert operator_module._validated_ai_report_identity(report) is report
    report.asdict = lambda: {
        **payload,
        "model_digest": _sha("tampered-model"),
        "report_sha256": report_sha256,
    }
    with pytest.raises(ValueError, match="AI report identity differs"):
        operator_module._validated_ai_report_identity(report)


def test_round21_sealed_ai_model_is_preflighted_before_consuming_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "closed.duckdb"
    source.touch()
    model = "qwen3.5:9b"
    digest = _sha("expected-model")
    report = SimpleNamespace(
        report_sha256=_sha("development-report"),
        config=SimpleNamespace(model=model),
    )
    comparison = SimpleNamespace(
        comparison_sha256=_sha("development-comparison"),
        model=model,
        model_digest=digest,
        ai_report_sha256=report.report_sha256,
        validated=lambda: None,
    )
    comparison.validated = lambda: comparison
    claim = SimpleNamespace(
        pretest_manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        nominated_ai_model=model,
        nominated_ai_model_digest=digest,
        validated=lambda: None,
    )
    claim.validated = lambda: claim
    pretest = SimpleNamespace(
        manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        development_model_artifact_sha256=_sha("model"),
        core_corpus_publication_manifest_sha256=_sha("publication"),
        optional_campaign_terminal_sha256=None,
        nominated_ai_comparison_sha256=comparison.comparison_sha256,
        nominated_ai_model=model,
        nominated_ai_model_digest=digest,
        validated=lambda: None,
    )
    pretest.validated = lambda: pretest
    monkeypatch.setattr(
        operator_module,
        "validate_round21_development_artifact",
        lambda _value: {"artifact_sha256": _sha("model")},
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_core_publication_boundary",
        lambda _value: {
            "manifest_sha256": _sha("publication"),
            "sealed_test_population_manifest_sha256": _sha("population"),
            "terminal_transport_manifest_sha256": _sha("transport"),
        },
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda _value: {"manifest_sha256": _sha("transport")},
    )
    monkeypatch.setattr(
        operator_module,
        "_validated_ai_report_identity",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module,
        "polymarket_ai_model_evidence",
        lambda _config: (_sha("different-model"), _sha("metadata")),
    )
    consumed = False

    def fail_if_consumed(**_kwargs):
        nonlocal consumed
        consumed = True
        raise AssertionError("test access was consumed")

    monkeypatch.setattr(operator_module, "execute_round21_one_use", fail_if_consumed)
    with pytest.raises(ValueError, match="digest differs before test access"):
        operator_module.evaluate_round21_terminal_sealed_once(
            store_path=tmp_path / "one-use.sqlite3",
            claim=claim,
            pretest=pretest,
            publication_directory=tmp_path / "publication",
            source_database=source,
            terminal_transport_manifest={},
            development_model_artifact={},
            development_ai_report=report,
            development_ai_comparison=comparison,
        )
    assert consumed is False


def test_round21_sealed_operator_consumes_once_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "closed.duckdb"
    source.touch()
    claim = SimpleNamespace(
        claim_sha256=_sha("claim"),
        pretest_manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        nominated_ai_model=None,
        validated=lambda: None,
    )
    claim.validated = lambda: claim
    pretest = SimpleNamespace(
        manifest_sha256=_sha("pretest"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        development_model_artifact_sha256=_sha("model"),
        core_corpus_publication_manifest_sha256=_sha("publication"),
        optional_campaign_terminal_sha256=None,
        nominated_ai_comparison_sha256=None,
        nominated_ai_model=None,
        nominated_ai_model_digest=None,
        validated=lambda: None,
    )
    pretest.validated = lambda: pretest
    panel = _sealed_panel()
    assembly = operator_module.Round21SealedTestAssembly(
        publication_manifest_sha256=_sha("publication"),
        sealed_test_population_manifest_sha256=_sha("population"),
        terminal_transport_manifest_sha256=_sha("transport"),
        sidecar_terminal_manifest_sha256=None,
        partition_policy=_sealed_inputs()[0],
        test_panel=panel,
    )
    replay_result = SimpleNamespace(
        selected_matrix=("matrix",), optional_comparison=None
    )
    replay = economic_module.Round21SealedEconomicReplayEvidence(
        claim_sha256=_sha("claim"),
        test_access_sha256=_sha("access"),
        sealed_test_population_manifest_sha256=_sha("population"),
        test_dataset_sha256=panel.dataset_sha256,
        test_target_manifest_sha256=panel.target_manifest_sha256,
        replay_result=replay_result,
        historical_ai_cases=(),
        evidence_sha256=_sha("evidence"),
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_development_artifact",
        lambda _value: {"artifact_sha256": _sha("model")},
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_core_publication_boundary",
        lambda _value: {
            "manifest_sha256": _sha("publication"),
            "sealed_test_population_manifest_sha256": _sha("population"),
            "terminal_transport_manifest_sha256": _sha("transport"),
        },
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda _value: {"manifest_sha256": _sha("transport")},
    )
    monkeypatch.setattr(
        operator_module,
        "assemble_round21_sealed_test",
        lambda **_kwargs: assembly,
    )
    monkeypatch.setattr(
        operator_module,
        "evaluate_round21_sealed_predictions",
        lambda *_args, **_kwargs: "predictive",
    )
    monkeypatch.setattr(
        operator_module,
        "replay_round21_sealed_economics_with_ai_cases",
        lambda **_kwargs: replay,
    )
    monkeypatch.setattr(
        operator_module,
        "evaluate_round21_sealed_economics",
        lambda _matrix, **_kwargs: "economic",
    )
    monkeypatch.setattr(
        operator_module,
        "build_round21_sealed_evaluation_result",
        lambda **_kwargs: "sealed-result",
    )
    consumed: list[str] = []

    def execute_once(*, evaluator, **_kwargs):
        consumed.append("before")
        result = evaluator(_sha("access"))
        consumed.append("after")
        return result

    monkeypatch.setattr(operator_module, "execute_round21_one_use", execute_once)
    outcome = operator_module.evaluate_round21_terminal_sealed_once(
        store_path=tmp_path / "one-use.sqlite3",
        claim=claim,
        pretest=pretest,
        publication_directory=tmp_path / "publication",
        source_database=source,
        terminal_transport_manifest={},
        development_model_artifact={},
    )

    assert consumed == ["before", "after"]
    assert outcome.result == "sealed-result"
    assert outcome.replay_evidence is replay


def test_round21_sealed_ai_uses_same_sealed_cases_and_receipt_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _sealed_panel()
    case = PolymarketAIVetoCase(
        case_id="sealed-case-1",
        condition_id="0x" + "1" * 64,
        sample_id="sealed-sample-1",
        asset="BTC",
        event_start_ms=1_800_000_000_000,
        decision_received_wall_ms=1_800_000_000_100,
        decision_received_monotonic_ns=1,
        prompt_payload={"schema_version": "sealed-fixture"},
        case_sha256=_sha("sealed-case"),
    )
    replay_result = SimpleNamespace(
        selected_matrix="baseline-matrix",
        optional_comparison=None,
    )
    baseline = economic_module.Round21SealedEconomicReplayEvidence(
        claim_sha256=_sha("claim"),
        test_access_sha256=_sha("access"),
        sealed_test_population_manifest_sha256=_sha("population"),
        test_dataset_sha256=panel.dataset_sha256,
        test_target_manifest_sha256=panel.target_manifest_sha256,
        replay_result=replay_result,
        historical_ai_cases=(case,),
        evidence_sha256=_sha("evidence"),
    )
    model = "qwen3.5:9b"
    digest = _sha("model-digest")
    config = SimpleNamespace(model=model)
    development_report = SimpleNamespace(
        report_sha256=_sha("development-report"),
        config=config,
        model_digest=digest,
        risk_benchmark_evidence_sha256=_sha("risk-benchmark"),
    )
    development_comparison = SimpleNamespace(
        model=model,
        model_digest=digest,
        ai_report_sha256=development_report.report_sha256,
        comparison_sha256=_sha("development-comparison"),
        validated=lambda: None,
    )
    development_comparison.validated = lambda: development_comparison
    claim = SimpleNamespace(
        claim_sha256=_sha("claim"),
        selected_population_layer="core",
        sealed_test_population_manifest_sha256=_sha("population"),
        nominated_ai_model=model,
        nominated_ai_model_digest=digest,
        validated=lambda: None,
    )
    claim.validated = lambda: claim
    assembly = operator_module.Round21SealedTestAssembly(
        publication_manifest_sha256=_sha("publication"),
        sealed_test_population_manifest_sha256=_sha("population"),
        terminal_transport_manifest_sha256=_sha("transport"),
        sidecar_terminal_manifest_sha256=None,
        partition_policy=_sealed_inputs()[0],
        test_panel=panel,
    )
    sealed_report = SimpleNamespace(report_sha256=_sha("sealed-report"))
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        operator_module,
        "_validated_ai_report_identity",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module,
        "polymarket_ai_model_evidence",
        lambda value: (
            (digest, {}) if value is config else pytest.fail("config differs")
        ),
    )

    def benchmark(cases, **kwargs):
        observed["benchmark_cases"] = cases
        observed["benchmark_kwargs"] = kwargs
        return sealed_report

    monkeypatch.setattr(operator_module, "benchmark_polymarket_ai_veto", benchmark)
    unloaded: list[object] = []
    monkeypatch.setattr(
        operator_module,
        "unload_polymarket_ai_model",
        unloaded.append,
    )

    def permissions(*, cases, report):
        observed["permission_cases"] = cases
        observed["permission_report"] = report
        return "sealed-permissions"

    monkeypatch.setattr(
        operator_module,
        "round21_permissions_from_ai_report",
        permissions,
    )

    class FakeAccumulator:
        def __init__(self, **kwargs):
            observed["accumulator_kwargs"] = kwargs

        def observe(self, condition):
            observed["observed_condition"] = condition

        def finish(self):
            return "ai-matrix"

    monkeypatch.setattr(
        operator_module,
        "Round21EconomicMatrixAccumulator",
        FakeAccumulator,
    )

    condition = SimpleNamespace(
        matched_population_sha256=lambda: _sha("matched-condition")
    )

    def replay(**kwargs):
        observed["replay_kwargs"] = kwargs
        kwargs["selected_condition_sinks"][0](condition)
        return baseline

    monkeypatch.setattr(
        operator_module,
        "replay_round21_sealed_economics_with_ai_cases",
        replay,
    )
    sealed_comparison = SimpleNamespace(comparison_sha256=_sha("sealed-comparison"))

    def compare(**kwargs):
        observed["compare_kwargs"] = kwargs
        return sealed_comparison

    monkeypatch.setattr(
        operator_module,
        "compare_round21_ai_replay_matrices",
        compare,
    )

    report, comparison = operator_module._sealed_ai_comparison(
        baseline=baseline,
        development_report=development_report,
        development_comparison=development_comparison,
        source_database="closed.duckdb",
        terminal_transport_manifest={"manifest_sha256": _sha("transport")},
        assembly=assembly,
        model_artifact={"artifact_sha256": _sha("model")},
        claim=claim,
        test_access_sha256=_sha("access"),
        initial_capital_quote=Decimal("10000"),
        minimum_edge_per_share=Decimal("0.02"),
        builder_taker_fee_bps=Decimal("0"),
        cache_store=None,
        progress=None,
    )

    assert report is sealed_report
    assert comparison is sealed_comparison
    assert observed["benchmark_cases"] == (case,)
    assert observed["benchmark_kwargs"]["all_condition_ids"] == (case.condition_id,)
    assert observed["benchmark_kwargs"]["expected_model_digest"] == digest
    assert observed["permission_cases"] == (case,)
    assert observed["permission_report"] is sealed_report
    assert observed["accumulator_kwargs"]["directional_permissions"] == (
        "sealed-permissions"
    )
    assert observed["replay_kwargs"]["test_panel"] is panel
    assert observed["replay_kwargs"]["claim_sha256"] == _sha("claim")
    assert observed["replay_kwargs"]["test_access_sha256"] == _sha("access")
    assert observed["compare_kwargs"]["baseline_matrix"] == "baseline-matrix"
    assert observed["compare_kwargs"]["ai_matrix"] == "ai-matrix"
    assert observed["compare_kwargs"]["cases"] == (case,)
    assert unloaded == [config]
