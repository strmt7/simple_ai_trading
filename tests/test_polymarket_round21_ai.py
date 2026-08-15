from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading import polymarket_round21_economic_operator as economic_operator
from simple_ai_trading import polymarket_round21_ai_operator as ai_operator
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_ai_veto import (
    PolymarketAIVetoConfig,
    PolymarketAIVetoReport,
    benchmark_polymarket_ai_veto,
    unload_polymarket_ai_model,
)
from simple_ai_trading.polymarket_round21_ai import (
    POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
    POLYMARKET_ROUND21_AI_HISTORICAL_CASE_SCHEMA_VERSION,
    POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256,
    Round21AICaseReceipt,
    Round21AIHistoricalSchedule,
    Round21HistoricalAICaseCollector,
    Round21HistoricalAICaseFactory,
    build_round21_ai_veto_cases,
    build_round21_historical_ai_veto_cases,
    round21_ai_case_source_evidence_sha256,
    round21_permissions_from_ai_report,
)
from simple_ai_trading.polymarket_round21_ai_comparison import (
    POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
    compare_round21_ai_full_matrix,
    compare_round21_ai_replay_matrices,
)
from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
)
from simple_ai_trading.polymarket_round21_dataset import (
    Round21OfficialOutcome,
)
from simple_ai_trading.polymarket_round21_execution import (
    Round21MarketExecutionEvidence,
)
from simple_ai_trading.polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21InferencePanel,
    Round21ProbabilityBatch,
)
from simple_ai_trading.polymarket_round21_policy import (
    Round21ProbabilityEnvelope,
)
from simple_ai_trading.polymarket_round21_replay import (
    Round21PairedEconomicMatrixAccumulator,
    Round21ReplayCondition,
    replay_round21_economics,
    replay_round21_full_matrix,
)


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000
CONDITION_ID = "0x" + "8" * 64
UP_TOKEN = "3" * 40
DOWN_TOKEN = "4" * 40
DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-ai-veto-design-v7.json"
)
HISTORICAL_DESIGN_PATH = (
    DESIGN_PATH.parent / "round-021-ai-historical-schedule-design-v7.json"
)
QWEN35_PROBE_PATH = (
    DESIGN_PATH.parent / "round-021-qwen3.5-9b-target-free-host-probe-2026-08-03.json"
)
QWEN35_FAILED_PROBE_PATHS = (
    DESIGN_PATH.parent
    / "round-021-qwen3.5-9b-target-free-host-probe-attempt1-failed-2026-08-03.json",
    DESIGN_PATH.parent
    / "round-021-qwen3.5-9b-target-free-host-probe-attempt2-failed-2026-08-03.json",
)
AI_CANDIDATE_PROBE_PATHS = {
    "qwen3.5:9b": (
        DESIGN_PATH.parent
        / "round-021-qwen3.5-9b-target-free-candidate-host-probe-2026-08-03.json"
    ),
    "fin-r1:8b": (
        DESIGN_PATH.parent
        / "round-021-fin-r1-8b-target-free-host-probe-2026-08-03.json"
    ),
    "fino1:8b": (
        DESIGN_PATH.parent / "round-021-fino1-8b-target-free-host-probe-2026-08-03.json"
    ),
}
FINO1_SCHEMA_ONLY_ATTEMPT_PATH = (
    DESIGN_PATH.parent
    / "round-021-fino1-8b-target-free-host-probe-attempt1-invalid-approval-2026-08-03.json"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="67890",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.20"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=_sha("market"),
        gamma_payload_json="{}",
    )


def _evidence() -> Round21MarketExecutionEvidence:
    return Round21MarketExecutionEvidence.create(
        condition_id=CONDITION_ID,
        observed_wall_ms=DECISION_MS - 1_000,
        observed_monotonic_ns=(DECISION_MS - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=0,
        minimum_order_age_seconds=0,
        clob_info_sha256=_sha("clob"),
        up_fee_rate_sha256=_sha("up-fee"),
        down_fee_rate_sha256=_sha("down-fee"),
        snapshot_sha256=_sha("snapshot"),
    )


def _book(
    outcome: str,
    received_wall_ms: int,
    *,
    ask: str = "0.50",
) -> PaperBookSnapshot:
    token = UP_TOKEN if outcome == "Up" else DOWN_TOKEN
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token,
        bids=(BookLevel(Decimal("0.49"), Decimal("1000")),),
        asks=(BookLevel(Decimal(ask), Decimal("1000")),),
        source_time_ms=received_wall_ms - 10,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=received_wall_ms * 1_000_000,
        source_payload_sha256=_sha(f"{outcome}-{received_wall_ms}-{ask}"),
        connected=True,
        gap_free=True,
    ).validated()


def _panel() -> Round21InferencePanel:
    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA.validated()
    core = np.zeros((1, len(schema.core_names)), dtype=np.float32)
    core[0, schema.core_names.index("core.elapsed_fraction")] = 0.4
    core[0, schema.core_names.index("core.remaining_seconds")] = 180.0
    core[0, schema.core_names.index("core.structural_probability_up")] = 0.70
    core[0, schema.core_names.index("core.normalized_market_prior_up")] = 0.50
    return Round21InferencePanel.create(
        condition_ids=np.asarray([CONDITION_ID], dtype=object),
        event_start_ms=np.asarray([START_MS], dtype=np.int64),
        decision_time_ms=np.asarray([DECISION_MS], dtype=np.int64),
        structural_probability=np.asarray([0.70], dtype=np.float64),
        market_prior_probability=np.asarray([0.50], dtype=np.float64),
        core_features=core,
        spot_features=np.zeros((1, len(schema.spot_names)), dtype=np.float32),
        usdm_features=np.zeros((1, len(schema.usdm_names)), dtype=np.float32),
        spot_available=np.asarray([False], dtype=np.bool_),
        usdm_available=np.asarray([False], dtype=np.bool_),
        core_feature_names_sha256=schema.core_names_sha256,
        spot_feature_names_sha256=schema.spot_names_sha256,
        usdm_feature_names_sha256=schema.usdm_names_sha256,
        source_dataset_sha256=_sha("target-free-source-dataset"),
        dataset_design_sha256=POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    )


def _batch(
    panel: Round21InferencePanel,
    *,
    model_sha256: str | None = None,
) -> Round21ProbabilityBatch:
    return Round21ProbabilityBatch.create(
        population_layer="core",
        selected_candidate_id="ridge_l2_1",
        contributing_candidate_ids=(
            "ridge_l2_0_1",
            "ridge_l2_1",
            "ridge_l2_10",
            "elasticnet_0_5",
            "elasticnet_0_8",
        ),
        indices=np.asarray([0], dtype=np.int64),
        probability_up=np.asarray([0.80], dtype=np.float64),
        lower_up=np.asarray([0.75], dtype=np.float64),
        upper_up=np.asarray([0.85], dtype=np.float64),
        feature_support_eligible=np.asarray([True], dtype=np.bool_),
        source_model_artifact_sha256=model_sha256 or _sha("core-model"),
        feature_batch_sha256=panel.feature_batch_sha256,
    )


def _envelope(
    panel: Round21InferencePanel,
    batch: Round21ProbabilityBatch,
) -> Round21ProbabilityEnvelope:
    return Round21ProbabilityEnvelope.from_probability_batch(
        batch=batch,
        panel=panel,
        panel_row_index=0,
    )


def _condition(
    envelope: Round21ProbabilityEnvelope,
    *,
    resolved_up: bool = True,
    future_ask: str = "0.50",
) -> Round21ReplayCondition:
    books = [
        _book("Up", DECISION_MS - 50),
        _book("Down", DECISION_MS - 50),
    ]
    for offset in (500, 750, 1_250):
        books.extend(
            (
                _book("Up", DECISION_MS + offset, ask=future_ask),
                _book("Down", DECISION_MS + offset),
            )
        )
    return Round21ReplayCondition.create(
        market=_market(),
        market_evidence=_evidence(),
        envelopes=(envelope,),
        books=books,
        outcome=Round21OfficialOutcome.create(
            condition_id=CONDITION_ID,
            event_start_ms=START_MS,
            resolved_up=resolved_up,
            observed_at_ms=START_MS + 300_100,
            source="official-polymarket-resolution",
            source_payload_sha256=_sha(f"outcome-{resolved_up}"),
        ),
        source_manifest_sha256=_sha("source-manifest"),
        reconciliation_sha256=_sha("reconciliation"),
    )


def _receipt(
    panel: Round21InferencePanel,
    batch: Round21ProbabilityBatch,
    condition: Round21ReplayCondition,
) -> Round21AICaseReceipt:
    source_evidence_sha256 = round21_ai_case_source_evidence_sha256(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        feature_batch_sha256=panel.feature_batch_sha256,
        feature_row_sha256=panel.row_sha256(0),
        probability_batch_sha256=batch.prediction_sha256,
        model_artifact_sha256=batch.source_model_artifact_sha256,
        causal_market_path_sha256=condition.causal_market_path_sha256(
            decision_time_ms=DECISION_MS
        ),
    )
    return Round21AICaseReceipt.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        received_wall_ms=DECISION_MS,
        received_monotonic_ns=DECISION_MS * 1_000_000,
        source_evidence_sha256=source_evidence_sha256,
    )


def _cases(
    *,
    condition: Round21ReplayCondition | None = None,
    panel: Round21InferencePanel | None = None,
    batch: Round21ProbabilityBatch | None = None,
):
    selected_panel = panel or _panel()
    selected_batch = batch or _batch(selected_panel)
    selected_condition = condition or _condition(
        _envelope(selected_panel, selected_batch)
    )
    return build_round21_ai_veto_cases(
        conditions=(selected_condition,),
        panel=selected_panel,
        probability_batch=selected_batch,
        case_receipts=(_receipt(selected_panel, selected_batch, selected_condition),),
    )


def _gpu_residency(
    _base_url: str,
    model: str,
    _timeout: float,
    *,
    expected_digest: str | None = None,
) -> OllamaResidencyReport:
    return OllamaResidencyReport(
        requested_model=model,
        status="gpu_resident",
        loaded_model=model,
        digest=expected_digest or "f" * 64,
        size_bytes=6_000_000_000,
        size_vram_bytes=6_000_000_000,
        vram_to_model_ratio=1.0,
    ).validated()


def _provider_response(
    action: str,
    *,
    model: str = "qwen3:8b",
) -> dict[str, object]:
    reason_codes = {
        "approve": ["edge_after_fees"],
        "veto": ["liquidity_stress"],
        "cooldown": ["cooldown_required"],
    }[action]
    return {
        "model": model,
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": action,
                    "confidence": 0.90,
                    "reason_codes": reason_codes,
                    "summary": "Causal evidence supports this risk-only decision.",
                }
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1,
        "load_duration": 0,
        "prompt_eval_count": 320,
        "prompt_eval_duration": 1,
        "eval_count": 24,
        "eval_duration": 1,
    }


def _post_json(
    action: str,
    *,
    fail_chat: bool = False,
    model: str = "qwen3:8b",
    digest: str = "f" * 64,
) -> Callable[[str, dict[str, object], float, str], object]:
    def post(
        url: str,
        _payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        if url.endswith("/api/tags"):
            return {"models": [{"name": model, "digest": digest}]}
        if url.endswith("/api/show"):
            return {"model": model, "parameters": "8B"}
        if fail_chat:
            raise RuntimeError("provider unavailable")
        return _provider_response(action, model=model)

    return post


def _report(
    monkeypatch: pytest.MonkeyPatch,
    cases,
    *,
    action: str,
    fail_chat: bool = False,
    latency_seconds: float = 0.125,
    model: str = "qwen3:8b",
    digest: str = "f" * 64,
    selection_sha256: str | None = None,
    risk_benchmark_evidence_sha256: str | None = None,
) -> PolymarketAIVetoReport:
    clock = iter((10.0, 10.0 + latency_seconds))
    monkeypatch.setattr(
        "simple_ai_trading.polymarket_ai_veto.time.perf_counter",
        lambda: next(clock),
    )
    return benchmark_polymarket_ai_veto(
        cases,
        all_condition_ids=(CONDITION_ID,),
        selection_sha256=selection_sha256 or _sha("ai-selection"),
        risk_benchmark_evidence_sha256=(
            risk_benchmark_evidence_sha256 or _sha("risk-benchmark")
        ),
        config=PolymarketAIVetoConfig(model=model),
        post_json=_post_json(  # type: ignore[arg-type]
            action,
            fail_chat=fail_chat,
            model=model,
            digest=digest,
        ),
        expected_model_digest=digest,
        residency_inspector=_gpu_residency,
    )


def test_polymarket_ai_model_unload_requires_exact_provider_evidence() -> None:
    calls: list[tuple[str, dict[str, object], float, str]] = []

    def post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        method: str,
    ) -> object:
        calls.append((url, payload, timeout, method))
        return {
            "model": "qwen3:8b",
            "response": "",
            "done": True,
            "done_reason": "unload",
        }

    unload_polymarket_ai_model(
        PolymarketAIVetoConfig(model="qwen3:8b", timeout_seconds=30),
        post_json=post,
    )

    assert calls == [
        (
            "http://127.0.0.1:11434/api/generate",
            {"model": "qwen3:8b", "keep_alive": 0, "stream": False},
            10.0,
            "POST",
        )
    ]
    with pytest.raises(ValueError, match="unload evidence is invalid"):
        unload_polymarket_ai_model(
            PolymarketAIVetoConfig(model="qwen3:8b"),
            post_json=lambda *_args, **_kwargs: {
                "model": "qwen3:8b",
                "response": "generated text",
                "done": True,
                "done_reason": "stop",
            },
        )


def test_round21_ai_model_preflight_precedes_source_replay() -> None:
    models = ("qwen3.5:9b", "fin-r1:8b", "fino1:8b")
    digests = {model: str(index) * 64 for index, model in enumerate(models, start=1)}
    calls: list[str] = []

    def evidence(config: PolymarketAIVetoConfig) -> tuple[str, str]:
        calls.append(config.model)
        return digests[config.model], _sha(f"{config.model}-metadata")

    observed = ai_operator.preflight_round21_ai_candidate_models(
        configs=tuple(PolymarketAIVetoConfig(model=model) for model in models),
        expected_model_digests=digests,
        model_evidence_reader=evidence,
    )

    assert calls == list(models)
    assert tuple(row[0] for row in observed) == models
    with pytest.raises(ValueError, match="differs from benchmark provenance"):
        ai_operator.preflight_round21_ai_candidate_models(
            configs=tuple(PolymarketAIVetoConfig(model=model) for model in models),
            expected_model_digests={**digests, "fino1:8b": "9" * 64},
            model_evidence_reader=evidence,
        )


def _rehash_report(report: PolymarketAIVetoReport) -> PolymarketAIVetoReport:
    provisional = replace(report, report_sha256="")
    payload = provisional.asdict()
    payload.pop("report_sha256")
    return replace(provisional, report_sha256=_canonical_sha256(payload))


def _economic_result(
    condition: Round21ReplayCondition,
) -> economic_operator.Round21DevelopmentEconomicResult:
    matrix = replay_round21_full_matrix((condition,))
    provisional = economic_operator.Round21DevelopmentEconomicResult(
        selected_population_layer="core",
        terminal_transport_manifest_sha256=_sha("terminal"),
        core_publication_manifest_sha256=_sha("publication"),
        model_artifact_sha256=_sha("model"),
        terminal_receipt_audit_sha256=_sha("audit"),
        source_condition_set_sha256=_sha("condition-set"),
        source_condition_count=1,
        selected_matrix=matrix,
        optional_comparison=None,
        development_gate_passed=False,
        result_sha256=hashlib.sha256(b"").hexdigest(),
    )
    return replace(
        provisional,
        result_sha256=economic_operator._canonical_sha256(  # noqa: SLF001
            provisional.identity_payload()
        ),
    ).validated()


def test_round21_ai_design_is_canonical_target_free_and_non_authoritative() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert claimed == POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256
    assert design["role"]["per_tick_direction_or_order_generation"] is False
    assert design["role"]["may_create_reverse_or_increase_risk"] is False
    assert design["cases"]["outcome_or_resolution"] is False
    assert not any(design["authority"].values())
    assert design["models"]["candidates"][0] == "qwen3.5:9b"
    assert (
        design["target_free_host_preflight"]["compatibility_pass_is_predictive_edge"]
        is False
    )
    assert PolymarketAIVetoConfig(model="fino1:8b").validated().model == "fino1:8b"


def test_qwen35_target_free_host_probe_preserves_failures_and_passes_v4() -> None:
    artifacts = tuple(
        json.loads(path.read_text(encoding="ascii"))
        for path in (*QWEN35_FAILED_PROBE_PATHS, QWEN35_PROBE_PATH)
    )
    for artifact in artifacts:
        claimed = artifact.pop("artifact_sha256")
        assert claimed == _canonical_sha256(artifact)
        assert artifact["semantics"] == {
            "capture_data_accessed": False,
            "live_trading_authority": False,
            "market_target_or_outcome_accessed": False,
            "model_selected": False,
            "paper_trading_authority": False,
            "predictive_edge_claim": False,
            "profitability_claim": False,
        }
    assert [artifact["passed"] for artifact in artifacts] == [False, False, True]
    final = artifacts[-1]
    assert final["selection_design_sha256"] == (
        "abc56f7ffcbbf4832c9725dcb3a028201395a5a9537717c3e0e79b120a5ac4b0"
    )
    assert final["report"]["config"]["model"] == "qwen3.5:9b"
    assert final["report"]["valid_response_count"] == 1
    assert final["report"]["provider_failure_count"] == 0
    assert final["report"]["results"][0]["provider_runtime"]["status"] == (
        "gpu_resident"
    )
    assert final["report"]["results"][0]["decision"]["valid"] is True


def test_round21_ai_candidate_host_probes_enforce_packet_constraints() -> None:
    artifacts: dict[str, dict[str, object]] = {}
    for model, path in AI_CANDIDATE_PROBE_PATHS.items():
        artifact = json.loads(path.read_text(encoding="ascii"))
        claimed = artifact.pop("artifact_sha256")
        assert claimed == _canonical_sha256(artifact)
        assert artifact["model"] == model
        assert artifact["schema_version"].endswith("-v2")
        assert artifact["semantics"] == {
            "capture_data_accessed": False,
            "live_trading_authority": False,
            "market_target_or_outcome_accessed": False,
            "model_selected": False,
            "paper_trading_authority": False,
            "predictive_edge_claim": False,
            "profitability_claim": False,
        }
        artifacts[model] = artifact

    for model in ("qwen3.5:9b", "fin-r1:8b"):
        artifact = artifacts[model]
        assert artifact["passed"] is True
        assert artifact["checks"]["case_constraint_semantics_passed"] is True
        assert artifact["report"]["valid_response_count"] == 1
        assert artifact["report"]["provider_failure_count"] == 0
        assert artifact["report"]["results"][0]["provider_runtime"]["status"] == (
            "gpu_resident"
        )

    rejected = artifacts["fino1:8b"]
    assert rejected["passed"] is False
    assert rejected["checks"]["case_constraint_semantics_passed"] is False
    assert rejected["report"]["valid_response_count"] == 0
    assert rejected["report"]["provider_failure_count"] == 1
    result = rejected["report"]["results"][0]
    assert result["decision"]["action"] == "veto"
    assert result["decision"]["valid"] is False
    assert result["response_payload"]["error_type"] == "ValueError"
    assert result["response_payload"]["provider_response_received"] is True

    attempt = json.loads(FINO1_SCHEMA_ONLY_ATTEMPT_PATH.read_text(encoding="ascii"))
    attempt_claimed = attempt.pop("artifact_sha256")
    assert attempt_claimed == _canonical_sha256(attempt)
    assert attempt["report"]["results"][0]["decision"]["action"] == "approve"
    source = json.loads(QWEN35_PROBE_PATH.read_text(encoding="ascii"))
    assert attempt["source_target_free_probe_sha256"] == source["artifact_sha256"]
    packet = source["case"]["prompt_payload"]
    assert Decimal(packet["expected_edge_per_contract_after_fee"]) < Decimal(
        packet["minimum_required_edge_per_contract"]
    )


def test_round21_historical_ai_schedule_is_canonical_and_explicitly_virtual() -> None:
    design = json.loads(HISTORICAL_DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert claimed == POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256
    assert design["parents"]["round21_ai_veto_design_v7_sha256"] == (
        POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256
    )
    assert design["supersession"] == {
        "change": "bind_unchanged_virtual_timing_to_round21_ai_veto_design_v7",
        "clock_or_latency_semantics_changed": False,
        "candidate_count_changed": False,
        "capture_data_used_for_change": False,
        "targets_used_for_change": False,
        "market_outcomes_used_for_change": False,
    }
    assert design["clock"]["historical_host_receipt_claim"] is False
    assert design["clock"]["market_data_receipts"] == (
        "exact_captured_host_receipts_only"
    )
    assert not any(
        design["semantics"][key]
        for key in (
            "profitability_claim",
            "paper_trading_authority",
            "live_trading_authority",
        )
    )

    schedule = Round21AIHistoricalSchedule.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        source_evidence_sha256=_sha("historical-source"),
    )
    assert schedule.virtual_dispatch_wall_ms == DECISION_MS
    assert schedule.virtual_dispatch_monotonic_ns == DECISION_MS * 1_000_000
    assert Round21AIHistoricalSchedule.from_mapping(schedule.asdict()) == schedule
    with pytest.raises(ValueError, match="schedule differs"):
        Round21AIHistoricalSchedule.from_mapping(
            {
                **schedule.asdict(),
                "historical_host_receipt_observed": True,
            }
        )


def test_round21_ai_case_is_unchanged_by_outcome_or_future_books() -> None:
    panel = _panel()
    batch = _batch(panel)
    envelope = _envelope(panel, batch)
    first = _condition(envelope, resolved_up=True, future_ask="0.50")
    changed_future = _condition(
        envelope,
        resolved_up=False,
        future_ask="0.90",
    )

    first_case = _cases(condition=first, panel=panel, batch=batch)[0]
    changed_case = _cases(
        condition=changed_future,
        panel=panel,
        batch=batch,
    )[0]

    assert first_case == changed_case
    assert first.causal_market_path_sha256(
        decision_time_ms=DECISION_MS
    ) == changed_future.causal_market_path_sha256(decision_time_ms=DECISION_MS)
    assert (
        first.matched_population_sha256() != changed_future.matched_population_sha256()
    )
    assert "matched_population_sha256" not in first_case.prompt_payload["identity"]
    assert "outcome_sha256" not in first_case.prompt_payload["identity"]
    assert first_case.prompt_payload["hard_constraints"][
        "no_outcome_resolution_future_book_or_future_pnl"
    ]


def test_round21_historical_ai_case_discloses_virtual_timing_and_stays_causal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    envelope = _envelope(panel, batch)
    first = _condition(envelope)
    changed_future = _condition(
        envelope,
        resolved_up=False,
        future_ask="0.99",
    )

    first_cases = build_round21_historical_ai_veto_cases(
        conditions=(first,),
        panel=panel,
        probability_batch=batch,
    )
    changed_cases = build_round21_historical_ai_veto_cases(
        conditions=(changed_future,),
        panel=panel,
        probability_batch=batch,
    )

    assert first_cases == changed_cases
    factory = Round21HistoricalAICaseFactory(
        panel=panel,
        probability_batch=batch,
    )
    monkeypatch.setattr(
        Round21InferencePanel,
        "validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compiled AI factory revalidated the full panel")
        ),
    )
    assert factory.build(first) == first_cases[0]
    collector = Round21HistoricalAICaseCollector((factory,))
    with pytest.raises(ValueError, match="population is incomplete"):
        collector.finish()
    collector(first)
    assert collector.finish() == first_cases
    with pytest.raises(ValueError, match="population differs"):
        collector(first)
    case = first_cases[0]
    assert case.prompt_payload["schema_version"] == (
        POLYMARKET_ROUND21_AI_HISTORICAL_CASE_SCHEMA_VERSION
    )
    assert "case_receipt_sha256" not in case.prompt_payload["identity"]
    schedule = case.prompt_payload["historical_schedule"]
    assert schedule["historical_host_receipt_observed"] is False
    assert schedule["historical_provider_load_observed"] is False
    assert case.decision_received_wall_ms == DECISION_MS
    assert case.decision_received_monotonic_ns == DECISION_MS * 1_000_000

    report = _report(
        monkeypatch,
        first_cases,
        action="veto",
        latency_seconds=0.125,
    )
    permissions = round21_permissions_from_ai_report(
        cases=first_cases,
        report=report,
    )
    assert permissions[0].effective_at_ms == DECISION_MS + 125
    assert permissions[0].directional_entry_allowed is False


def test_round21_ai_case_requires_exact_receipt_and_probability_provenance() -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    receipt = _receipt(panel, batch, condition)
    late_receipt = replace(
        receipt,
        received_wall_ms=DECISION_MS + 251,
    )
    tampered_receipt = replace(
        receipt,
        source_evidence_sha256=_sha("different-source"),
    )
    wrong_source_receipt = Round21AICaseReceipt.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        received_wall_ms=DECISION_MS,
        received_monotonic_ns=DECISION_MS * 1_000_000,
        source_evidence_sha256=_sha("different-source"),
    )

    with pytest.raises(ValueError, match="receipt is invalid"):
        build_round21_ai_veto_cases(
            conditions=(condition,),
            panel=panel,
            probability_batch=batch,
            case_receipts=(late_receipt,),
        )
    with pytest.raises(ValueError, match="receipt differs"):
        build_round21_ai_veto_cases(
            conditions=(condition,),
            panel=panel,
            probability_batch=batch,
            case_receipts=(tampered_receipt,),
        )
    with pytest.raises(ValueError, match="probability evidence differs"):
        build_round21_ai_veto_cases(
            conditions=(condition,),
            panel=panel,
            probability_batch=batch,
            case_receipts=(wrong_source_receipt,),
        )
    with pytest.raises(ValueError, match="population differs"):
        build_round21_ai_veto_cases(
            conditions=(condition,),
            panel=panel,
            probability_batch=batch,
            case_receipts=(),
        )

    foreign_batch = _batch(panel, model_sha256=_sha("foreign-model"))
    with pytest.raises(ValueError, match="probability evidence differs"):
        build_round21_ai_veto_cases(
            conditions=(condition,),
            panel=panel,
            probability_batch=foreign_batch,
            case_receipts=(receipt,),
        )

    wrong_envelope = Round21ProbabilityEnvelope.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        probability_up=Decimal("0.80"),
        lower_up=Decimal("0.75"),
        upper_up=Decimal("0.85"),
        model_layer="core",
        source_model_artifact_sha256=batch.source_model_artifact_sha256,
        source_probability_batch_sha256=batch.prediction_sha256,
        feature_row_sha256=_sha("wrong-feature-row"),
    )
    with pytest.raises(ValueError, match="probability evidence differs"):
        _cases(
            condition=_condition(wrong_envelope),
            panel=panel,
            batch=batch,
        )


def test_round21_ai_case_builder_uses_bulk_indexed_probability_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    receipt = _receipt(panel, batch, condition)

    def reject_scalar_panel_row(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("scalar panel row hash was used")

    def reject_scalar_probability_row(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, float, float]:
        raise AssertionError("scalar probability row lookup was used")

    monkeypatch.setattr(
        Round21InferencePanel,
        "row_sha256",
        reject_scalar_panel_row,
    )
    monkeypatch.setattr(
        Round21ProbabilityBatch,
        "row",
        reject_scalar_probability_row,
    )

    cases = build_round21_ai_veto_cases(
        conditions=(condition,),
        panel=panel,
        probability_batch=batch,
        case_receipts=(receipt,),
    )

    assert len(cases) == 1
    assert cases[0].condition_id == CONDITION_ID


@pytest.mark.parametrize(
    ("action", "fail_chat", "expected_allowed"),
    (
        ("approve", False, True),
        ("veto", False, False),
        ("cooldown", False, False),
        ("approve", True, True),
    ),
)
def test_round21_ai_report_maps_only_valid_vetoes_to_delayed_blocks(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    fail_chat: bool,
    expected_allowed: bool,
) -> None:
    cases = _cases()
    report = _report(
        monkeypatch,
        cases,
        action=action,
        fail_chat=fail_chat,
    )

    permissions = round21_permissions_from_ai_report(cases=cases, report=report)

    assert len(permissions) == 1
    assert permissions[0].directional_entry_allowed is expected_allowed
    assert permissions[0].effective_at_ms == DECISION_MS + 125
    if fail_chat:
        assert report.provider_failure_count == 1
        assert not report.results[0].decision.valid


def test_round21_ai_latency_cannot_change_an_earlier_replay_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = _cases(condition=condition, panel=panel, batch=batch)
    veto = _report(monkeypatch, cases, action="veto")
    permissions = round21_permissions_from_ai_report(cases=cases, report=veto)

    replay = replay_round21_economics(
        (condition,),
        scenario_name="primary",
        directional_permissions=permissions,
    )

    assert replay.conditions[0].steps[0].decision_time_ms == DECISION_MS
    assert replay.conditions[0].steps[0].action == "buy_up"
    assert permissions[0].effective_at_ms > DECISION_MS


def test_round21_ai_bridge_rejects_rehashed_semantic_and_response_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _cases()
    report = _report(monkeypatch, cases, action="approve")
    wrong_parameters = _rehash_report(replace(report, model_parameters_b=7.0))
    tampered_result = replace(
        report.results[0],
        response_payload={"tampered": True},
    )
    wrong_response = _rehash_report(replace(report, results=(tampered_result,)))
    prompt_payload = dict(cases[0].prompt_payload)
    prompt_payload["outcome"] = "Up"
    tampered_case = replace(
        cases[0],
        prompt_payload=prompt_payload,
        case_id=_canonical_sha256(prompt_payload),
        case_sha256="",
    )
    tampered_case = replace(
        tampered_case,
        case_sha256=_canonical_sha256(tampered_case.identity_payload()),
    )

    with pytest.raises(ValueError, match="report differs"):
        round21_permissions_from_ai_report(
            cases=cases,
            report=wrong_parameters,
        )
    with pytest.raises(ValueError, match="result differs"):
        round21_permissions_from_ai_report(
            cases=cases,
            report=wrong_response,
        )
    with pytest.raises(ValueError, match="case differs"):
        round21_permissions_from_ai_report(
            cases=(tampered_case,),
            report=report,
        )


def test_round21_ai_full_matrix_never_selects_from_insufficient_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = _cases(condition=condition, panel=panel, batch=batch)
    veto = _report(
        monkeypatch,
        cases,
        action="veto",
        latency_seconds=0.0,
    )

    comparison = compare_round21_ai_full_matrix(
        conditions=(condition,),
        cases=cases,
        report=veto,
    )

    assert len(comparison.deltas) == 81
    assert comparison.matched_decision_count == 1
    assert comparison.non_tied_primary_action_count == 1
    assert comparison.development_qualified is False
    assert comparison.ai_model_selected is False
    assert comparison.profitability_claim is False
    assert comparison.paper_trading_authority is False
    assert comparison.live_trading_authority is False


def test_round21_ai_comparison_streams_once_and_reuses_precomputed_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = _cases(condition=condition, panel=panel, batch=batch)
    report = _report(
        monkeypatch,
        cases,
        action="veto",
        latency_seconds=0.0,
        model="qwen3.5:9b",
    )
    iterations = 0

    def conditions():
        nonlocal iterations
        iterations += 1
        if iterations > 1:
            raise AssertionError("AI source was replayed more than once")
        yield condition

    streamed = compare_round21_ai_full_matrix(
        conditions=conditions(),
        cases=cases,
        report=report,
    )
    permissions = round21_permissions_from_ai_report(cases=cases, report=report)
    pair = Round21PairedEconomicMatrixAccumulator(
        challenger_directional_permissions=permissions,
    )
    pair.observe(condition)
    baseline, ai = pair.finish()
    matched_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
            "condition_sha256": list(pair.matched_condition_sha256),
        }
    )
    precomputed = compare_round21_ai_replay_matrices(
        baseline_matrix=baseline,
        ai_matrix=ai,
        cases=cases,
        report=report,
        matched_population_sha256=matched_sha256,
    )

    assert iterations == 1
    assert streamed == precomputed
    with pytest.raises(ValueError, match="matched population differs"):
        compare_round21_ai_replay_matrices(
            baseline_matrix=ai,
            ai_matrix=baseline,
            cases=cases,
            report=report,
            matched_population_sha256=matched_sha256,
        )


def test_round21_development_ai_operator_replays_candidates_in_one_source_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = _cases(condition=condition, panel=panel, batch=batch)
    report = _report(
        monkeypatch,
        cases,
        action="veto",
        latency_seconds=0.0,
        model="qwen3.5:9b",
    )
    economic = _economic_result(condition)
    source_passes = 0

    def fake_economic_replay(**kwargs):
        nonlocal source_passes
        source_passes += 1
        for sink in kwargs["selected_condition_sinks"]:
            sink(condition)
        return economic

    monkeypatch.setattr(
        ai_operator,
        "replay_round21_development_economics",
        fake_economic_replay,
    )
    result = ai_operator.replay_round21_development_ai_ablation(
        source_database="source.duckdb",
        terminal_transport_manifest={},
        partition_policy=object(),  # type: ignore[arg-type]
        development_panels=(),
        development_model_artifact={},
        core_publication_manifest_sha256=_sha("publication"),
        selected_population_layer="core",
        expected_development_economic_result=economic,
        cases=cases,
        reports=(report,),
    )

    assert source_passes == 1
    assert len(result.comparisons) == 1
    assert result.comparisons[0].model == "qwen3.5:9b"
    assert result.comparisons[0].baseline_matrix_sha256 == (
        result.baseline_matrix_sha256
    )
    assert result.ai_model_selected is False
    assert result.profitability_claim is False
    assert result.live_trading_authority is False
    assert result.asdict()["candidate_count"] == 1
    with pytest.raises(ValueError, match="AI replay result differs"):
        replace(result, live_trading_authority=True).validated()


def test_round21_development_ai_benchmark_binds_all_models_and_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = build_round21_historical_ai_veto_cases(
        conditions=(condition,),
        panel=panel,
        probability_batch=batch,
    )
    economic = _economic_result(condition)
    models = ("qwen3.5:9b", "fin-r1:8b", "fino1:8b")
    digests = {model: str(index) * 64 for index, model in enumerate(models, start=1)}
    calls = []
    unloaded: list[str] = []

    def fake_benchmark(received_cases, **kwargs):
        config = kwargs["config"]
        calls.append((received_cases, kwargs))
        return _report(
            monkeypatch,
            received_cases,
            action="veto",
            latency_seconds=0.0,
            model=config.model,
            digest=digests[config.model],
            selection_sha256=kwargs["selection_sha256"],
            risk_benchmark_evidence_sha256=(kwargs["risk_benchmark_evidence_sha256"]),
        )

    monkeypatch.setattr(
        ai_operator,
        "benchmark_polymarket_ai_veto",
        fake_benchmark,
    )
    result = ai_operator.benchmark_round21_development_ai_candidates(
        economic_result=economic,
        development_model_artifact_sha256=_sha("model"),
        selected_population_layer="core",
        risk_benchmark_evidence_sha256=_sha("risk-benchmark"),
        cases=cases,
        configs=tuple(PolymarketAIVetoConfig(model=model) for model in models),
        expected_model_digests=digests,
        model_unloader=lambda config: unloaded.append(config.model),
    )

    assert len(calls) == 3
    assert tuple(unloaded) == models
    assert tuple(value.config.model for value in result.reports) == models
    assert result.cases == cases
    assert result.historical_case_set_sha256 != hashlib.sha256(b"").hexdigest()
    assert result.ai_model_selected is False
    assert result.profitability_claim is False
    assert result.asdict()["model"] == list(models)
    with pytest.raises(ValueError, match="benchmark result differs"):
        replace(result, cases=()).validated()


def test_round21_development_ai_program_binds_benchmark_replay_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    batch = _batch(panel)
    condition = _condition(_envelope(panel, batch))
    cases = build_round21_historical_ai_veto_cases(
        conditions=(condition,),
        panel=panel,
        probability_batch=batch,
    )
    economic = _economic_result(condition)
    models = ("qwen3.5:9b", "fin-r1:8b", "fino1:8b")
    digests = {model: str(index) * 64 for index, model in enumerate(models, start=1)}

    def fake_benchmark(received_cases, **kwargs):
        config = kwargs["config"]
        return _report(
            monkeypatch,
            received_cases,
            action="veto",
            latency_seconds=0.0,
            model=config.model,
            digest=digests[config.model],
            selection_sha256=kwargs["selection_sha256"],
            risk_benchmark_evidence_sha256=(kwargs["risk_benchmark_evidence_sha256"]),
        )

    monkeypatch.setattr(
        ai_operator,
        "benchmark_polymarket_ai_veto",
        fake_benchmark,
    )
    benchmark = ai_operator.benchmark_round21_development_ai_candidates(
        economic_result=economic,
        development_model_artifact_sha256=_sha("model"),
        selected_population_layer="core",
        risk_benchmark_evidence_sha256=_sha("risk-benchmark"),
        cases=cases,
        configs=tuple(PolymarketAIVetoConfig(model=model) for model in models),
        expected_model_digests=digests,
        model_unloader=lambda _config: None,
    )
    source_passes = 0

    def fake_economic_replay(**kwargs):
        nonlocal source_passes
        source_passes += 1
        for sink in kwargs["selected_condition_sinks"]:
            sink(condition)
        return economic

    monkeypatch.setattr(
        ai_operator,
        "replay_round21_development_economics",
        fake_economic_replay,
    )
    replay = ai_operator.replay_round21_development_ai_ablation(
        source_database="source.duckdb",
        terminal_transport_manifest={},
        partition_policy=object(),  # type: ignore[arg-type]
        development_panels=(),
        development_model_artifact={},
        core_publication_manifest_sha256=_sha("publication"),
        selected_population_layer="core",
        expected_development_economic_result=economic,
        cases=cases,
        reports=benchmark.reports,
    )
    result = ai_operator.complete_round21_development_ai_program(
        economic_result=economic,
        benchmark_result=benchmark,
        replay_result=replay,
    )

    assert source_passes == 1
    assert result.candidate_selection.nominated_model is None
    assert result.candidate_selection.qualified_candidate_count == 0
    assert result.target_accessed is False
    assert result.live_trading_authority is False
    assert result.asdict()["candidate_selection"]["nominated_model"] is None
    with pytest.raises(ValueError, match="program result differs"):
        replace(result, ai_edge_claim=True).validated()

    stages: list[str] = []

    def first_pass(**_kwargs):
        stages.append("case-building-source-pass")
        return economic, cases

    def model_preflight(**_kwargs):
        stages.append("model-preflight")
        return tuple(
            (model, digests[model], _sha(f"{model}-metadata")) for model in models
        )

    def candidate_benchmark(**kwargs):
        stages.append("three-model-benchmark")
        assert kwargs["cases"] == cases
        return benchmark

    def second_pass(**kwargs):
        stages.append("matched-economic-source-pass")
        assert kwargs["reports"] == benchmark.reports
        return replay

    monkeypatch.setattr(
        ai_operator,
        "preflight_round21_ai_candidate_models",
        model_preflight,
    )
    monkeypatch.setattr(
        ai_operator,
        "replay_round21_development_economics_with_ai_cases",
        first_pass,
    )
    monkeypatch.setattr(
        ai_operator,
        "benchmark_round21_development_ai_candidates",
        candidate_benchmark,
    )
    monkeypatch.setattr(
        ai_operator,
        "replay_round21_development_ai_ablation",
        second_pass,
    )
    orchestrated = ai_operator.run_round21_development_ai_program(
        source_database="source.duckdb",
        terminal_transport_manifest={},
        partition_policy=object(),  # type: ignore[arg-type]
        development_panels=(),
        development_model_artifact={"artifact_sha256": _sha("model")},
        core_publication_manifest_sha256=_sha("publication"),
        selected_population_layer="core",
        risk_benchmark_evidence_sha256=_sha("risk-benchmark"),
        configs=tuple(PolymarketAIVetoConfig(model=model) for model in models),
        expected_model_digests=digests,
    )

    assert orchestrated == result
    assert stages == [
        "model-preflight",
        "case-building-source-pass",
        "three-model-benchmark",
        "matched-economic-source-pass",
    ]
