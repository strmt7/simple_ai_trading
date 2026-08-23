from __future__ import annotations

import pytest

from simple_ai_trading import statistical_resampling
from simple_ai_trading.ai_model_identity import OllamaModelIdentity
from simple_ai_trading.ai_uplift import (
    AIUpliftPolicy,
    _binomial_upper_tail,
    _matched_period_deltas,
    _model_parameters_b,
    _moving_block_bootstrap,
    assess_ai_uplift,
    normalize_uplift_metrics,
)
from simple_ai_trading.ai_runtime import estimate_model_parameters_b


_DATASET_SHA256 = "d" * 64
_BASELINE_SHA256 = "b" * 64
_AI_SHA256 = "a" * 64
_MODEL_SHA256 = "c" * 64
_PERIOD_MS = 3 * 86_400_000


def _bound(metrics: dict[str, object], evidence_sha256: str) -> dict[str, object]:
    return {
        **metrics,
        "dataset_fingerprint": _DATASET_SHA256,
        "evidence_sha256": evidence_sha256,
    }


def _complete(metrics: dict[str, object], evidence_sha256: str) -> dict[str, object]:
    return _bound(
        {
            "roi_pct": 0.0,
            "profit_factor": 1.5,
            "win_rate": 0.6,
            "liquidation_events": 0,
            "max_consecutive_losses": 2,
            "downside_return_risk_ratio": 1.0,
            **metrics,
        },
        evidence_sha256,
    )


def _matched_periods(deltas: tuple[float, ...]) -> list[dict[str, object]]:
    first = 1_700_000_000_000
    return [
        {
            "scope": "BTCUSDT",
            "period_start_ms": first + index * _PERIOD_MS,
            "period_end_ms": first + (index + 1) * _PERIOD_MS,
            "baseline_return": 0.001,
            "ai_return": 0.001 + delta,
        }
        for index, delta in enumerate(deltas)
    ]


def test_estimate_model_parameters_from_local_model_names() -> None:
    assert estimate_model_parameters_b("qwen2.5:7b") == 7.0
    assert estimate_model_parameters_b("tiny-560m") == 0.56
    assert estimate_model_parameters_b("operator-selected-local-llm") is None


def test_ai_uplift_statistics_scale_to_ninety_days_of_five_minute_periods() -> None:
    period_count = 90 * 24 * 12
    p_value = _binomial_upper_tail(period_count, 14_256)
    bootstrap = _moving_block_bootstrap(
        tuple(0.0001 if index % 3 else -0.00005 for index in range(period_count)),
        samples=2_000,
        confidence=0.95,
        seed_material="f" * 64,
    )

    assert p_value == pytest.approx(1.7712369034957107e-58, rel=1e-9)
    assert bootstrap["samples"] == 2_000
    assert bootstrap["block_length"] == 161
    assert bootstrap["mean_delta_ci_lower"] > 0.0


def test_moving_block_bootstrap_samples_every_valid_remainder_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounds: list[tuple[int, int]] = []

    class DeterministicRandom:
        def randint(self, start: int, end: int) -> int:
            bounds.append((start, end))
            return start

    monkeypatch.setattr(
        statistical_resampling.random,
        "Random",
        lambda _seed: DeterministicRandom(),
    )
    bootstrap = _moving_block_bootstrap(
        tuple(float(value) for value in range(10)),
        samples=200,
        confidence=0.95,
        seed_material="a" * 64,
    )

    assert bootstrap["block_length"] == 3
    assert bounds[:4] == [(0, 7), (0, 7), (0, 7), (0, 9)]
    assert len(bounds) == 800


def test_ai_uplift_accepts_multibillion_holdout_improvement() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "roi_pct": 1.8,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 12,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is True
    assert report.advisory_only is False
    assert report.model_parameters_b == 7.0
    assert report.deltas["realized_pnl"] == 6.0
    assert report.statistical_evidence["accepted"] is True
    assert report.statistical_evidence["sample_count"] == 30
    assert report.statistical_evidence["effective_sample_count"] == 30
    assert report.statistical_evidence["positive_delta_count"] == 30
    assert report.statistical_evidence["negative_delta_count"] == 0
    assert report.statistical_evidence["tie_count"] == 0
    assert report.statistical_evidence["mean_delta_ci_lower"] > 0.0
    assert report.evidence_binding["accepted"] is True
    assert report.schema_version == "ai-uplift-v5"
    assert report.trading_authority is False
    assert report.profitability_claim is False


def test_ai_uplift_rejects_small_or_non_improving_models() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 12.0,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
        },
        {
            "realized_pnl": 11.0,
            "max_drawdown": 0.05,
            "expectancy": 0.8,
            "closed_trades": 3,
        },
        model_name="tiny-560m",
    )

    assert report.accepted is False
    assert report.advisory_only is True
    assert "model_parameters<2.00B" in report.reasons
    assert "ai_pnl_not_above_baseline" in report.reasons
    assert "ai_drawdown_worse_than_baseline" in report.reasons
    assert "ai_closed_trades<5" in report.reasons
    assert "ai_uplift_non_tied_pairs<30" in report.reasons


@pytest.mark.parametrize(
    ("model_name", "asserted_parameters_b", "expected_parameters_b", "reason"),
    [
        ("tiny-560m", 14.0, 0.56, "model_parameter_count_mismatch"),
        (
            "operator-selected-local-llm",
            14.0,
            None,
            "model_parameter_count_not_inferred_from_model_identity",
        ),
    ],
)
def test_ai_uplift_rejects_unverified_parameter_size_assertions(
    model_name: str,
    asserted_parameters_b: float,
    expected_parameters_b: float | None,
    reason: str,
) -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 30,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "roi_pct": 1.8,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 30,
            },
            _AI_SHA256,
        ),
        model_name=model_name,
        model_parameters_b=asserted_parameters_b,
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert report.model_parameters_b == expected_parameters_b
    assert reason in report.reasons


def test_ai_uplift_accepts_exact_attested_custom_model_identity() -> None:
    identity = OllamaModelIdentity(
        canonical_model="private-finance:latest",
        digest=_MODEL_SHA256,
        metadata_sha256="e" * 64,
        parameter_count=8_200_000_000,
        parameter_size="8.2B",
    ).validated()
    baseline = _complete(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 30,
        },
        _BASELINE_SHA256,
    )
    ai = _complete(
        {
            "realized_pnl": 18.0,
            "roi_pct": 1.8,
            "max_drawdown": 0.035,
            "expectancy": 1.2,
            "closed_trades": 30,
        },
        _AI_SHA256,
    )

    accepted = assess_ai_uplift(
        baseline,
        ai,
        model_name="private-finance:latest",
        model_artifact_sha256=_MODEL_SHA256,
        attested_model_identity=identity,
        matched_periods=_matched_periods((0.002,) * 30),
    )
    assertion_mismatch = assess_ai_uplift(
        baseline,
        ai,
        model_name="private-finance:latest",
        model_parameters_b=14.0,
        model_artifact_sha256=_MODEL_SHA256,
        attested_model_identity=identity,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert accepted.accepted is True
    assert accepted.model_parameters_b == pytest.approx(8.2)
    assert "model_parameter_count_not_inferred_from_model_identity" not in (
        accepted.reasons
    )
    assert assertion_mismatch.accepted is False
    assert "model_parameter_count_mismatch" in assertion_mismatch.reasons


@pytest.mark.parametrize(
    ("model_name", "artifact_sha256", "reason"),
    [
        (
            "different-private-model:latest",
            _MODEL_SHA256,
            "attested_model_name_mismatch",
        ),
        ("private-finance:latest", "f" * 64, "attested_model_digest_mismatch"),
    ],
)
def test_ai_uplift_rejects_misbound_attested_model_identity(
    model_name: str,
    artifact_sha256: str,
    reason: str,
) -> None:
    identity = OllamaModelIdentity(
        canonical_model="private-finance:latest",
        digest=_MODEL_SHA256,
        metadata_sha256="e" * 64,
        parameter_count=8_200_000_000,
        parameter_size="8.2B",
    ).validated()
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 30,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "roi_pct": 1.8,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 30,
            },
            _AI_SHA256,
        ),
        model_name=model_name,
        model_artifact_sha256=artifact_sha256,
        attested_model_identity=identity,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert report.model_parameters_b is None
    assert reason in report.reasons


def test_ai_uplift_rejects_structurally_invalid_attested_identity() -> None:
    invalid_identity = OllamaModelIdentity(
        canonical_model="private-finance:latest",
        digest=_MODEL_SHA256,
        metadata_sha256="e" * 64,
        parameter_count=0,
        parameter_size="unknown",
    )
    report = assess_ai_uplift(
        {},
        {},
        model_name="private-finance:latest",
        model_artifact_sha256=_MODEL_SHA256,
        attested_model_identity=invalid_identity,
    )

    assert report.model_parameters_b is None
    assert "attested_model_identity_invalid" in report.reasons


def test_ai_uplift_rejection_reason_order_is_stable() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 12.0,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
        },
        {
            "realized_pnl": 11.0,
            "max_drawdown": 0.05,
            "expectancy": 0.8,
            "closed_trades": 3,
        },
        model_name="tiny-560m",
    )

    assert report.reasons == (
        "ai_uplift_baseline_roi_pct_missing",
        "ai_uplift_baseline_profit_factor_missing",
        "ai_uplift_baseline_win_rate_missing",
        "ai_uplift_baseline_liquidation_events_missing",
        "ai_uplift_baseline_max_consecutive_losses_missing",
        "ai_uplift_baseline_downside_return_risk_ratio_missing",
        "ai_uplift_ai_roi_pct_missing",
        "ai_uplift_ai_profit_factor_missing",
        "ai_uplift_ai_win_rate_missing",
        "ai_uplift_ai_liquidation_events_missing",
        "ai_uplift_ai_max_consecutive_losses_missing",
        "ai_uplift_ai_downside_return_risk_ratio_missing",
        "model_parameters<2.00B",
        "ai_closed_trades<5",
        "ai_uplift_matched_periods_missing",
        "ai_uplift_evaluation_span_days<90",
        "ai_uplift_non_tied_pairs<30",
        "ai_uplift_positive_delta_rate<0.55",
        "ai_uplift_sign_test_p_value>0.0500",
        "ai_uplift_mean_sample_delta<=0",
        "ai_uplift_block_bootstrap_lower_mean_delta<=0",
        "ai_uplift_baseline_dataset_fingerprint_invalid",
        "ai_uplift_ai_dataset_fingerprint_invalid",
        "ai_uplift_baseline_evidence_sha256_invalid",
        "ai_uplift_ai_evidence_sha256_invalid",
        "ai_uplift_model_artifact_sha256_invalid",
        "ai_uplift_paired_samples_sha256_invalid",
        "ai_pnl_not_above_baseline",
        "ai_roi_not_above_baseline",
        "ai_expectancy_not_above_baseline",
        "ai_drawdown_worse_than_baseline",
    )


def test_ai_uplift_rejects_statistically_weak_paired_samples() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 13.0,
                "roi_pct": 1.3,
                "max_drawdown": 0.04,
                "expectancy": 1.0,
                "closed_trades": 10,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.001, -0.001) * 15),
    )

    assert report.accepted is False
    assert report.statistical_evidence["sample_count"] == 30
    assert report.statistical_evidence["effective_sample_count"] == 30
    assert report.statistical_evidence["positive_delta_count"] == 15
    assert report.statistical_evidence["negative_delta_count"] == 15
    assert "ai_uplift_positive_delta_rate<0.55" in report.reasons
    assert "ai_uplift_sign_test_p_value>0.0500" in report.reasons


def test_ai_uplift_sign_test_excludes_unchanged_periods() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "roi_pct": 1.8,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 12,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.0, 0.0, 0.002) * 30),
    )

    assert report.accepted is True
    assert report.statistical_evidence["sample_count"] == 90
    assert report.statistical_evidence["effective_sample_count"] == 30
    assert report.statistical_evidence["tie_count"] == 60
    assert report.statistical_evidence["positive_delta_rate"] == 1.0
    assert report.statistical_evidence["sign_test_p_value"] == 2**-30


def test_ai_uplift_rejects_aggregate_roi_degradation() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 2.0,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "roi_pct": 1.8,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 12,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert "ai_roi_not_above_baseline" in report.reasons


def test_ai_uplift_rejects_mismatched_metric_units_and_normalizes_percentages() -> None:
    baseline = _complete(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
            "win_rate": 0.60,
        },
        _BASELINE_SHA256,
    )
    ai = _complete(
        {
            "realized_pnl": 18.0,
            "roi_pct": 1.8,
            "max_drawdown_pct": 3.5,
            "expectancy": 1.2,
            "closed_trades": 12,
            "win_rate_pct": 65.0,
        },
        _AI_SHA256,
    )
    ai.pop("win_rate")

    report = assess_ai_uplift(
        baseline,
        ai,
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    normalized = normalize_uplift_metrics(ai)
    assert normalized["max_drawdown"] == pytest.approx(0.035)
    assert normalized["win_rate"] == pytest.approx(0.65)
    assert report.accepted is False
    assert "ai_uplift_max_drawdown_source_key_mismatch" in report.reasons
    assert "ai_uplift_win_rate_source_key_mismatch" in report.reasons


def test_ai_uplift_requires_thirty_non_tied_periods() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 18.0,
                "max_drawdown": 0.035,
                "expectancy": 1.2,
                "closed_trades": 12,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.0, 0.002) * 29 + (0.0, 0.0)),
    )

    assert report.accepted is False
    assert report.statistical_evidence["sample_count"] == 60
    assert report.statistical_evidence["effective_sample_count"] == 29
    assert report.statistical_evidence["tie_count"] == 31
    assert "ai_uplift_non_tied_pairs<30" in report.reasons


def test_ai_uplift_rejects_noncontiguous_matched_periods() -> None:
    periods = _matched_periods((0.002,) * 30)
    periods[10]["period_start_ms"] = int(periods[10]["period_start_ms"]) + 1
    periods[10]["period_end_ms"] = int(periods[10]["period_end_ms"]) + 1
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 14.0,
                "roi_pct": 1.4,
                "max_drawdown": 0.035,
                "expectancy": 1.1,
                "closed_trades": 10,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=periods,
    )

    assert report.accepted is False
    assert "ai_uplift_periods_not_contiguous" in report.reasons


def test_ai_uplift_rejects_noninteger_matched_period_timestamp() -> None:
    periods = _matched_periods((0.002,) * 30)
    periods[0]["period_start_ms"] = 1.5

    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 12.0,
                "roi_pct": 1.2,
                "max_drawdown": 0.04,
                "expectancy": 0.9,
                "closed_trades": 10,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 14.0,
                "roi_pct": 1.4,
                "max_drawdown": 0.035,
                "expectancy": 1.1,
                "closed_trades": 10,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=periods,
    )

    assert report.accepted is False
    assert "ai_uplift_period_0_invalid" in report.reasons
    assert "ai_uplift_period_rows_invalid" in report.reasons


def test_ai_uplift_rejects_malformed_matched_period_rows() -> None:
    rows: list[object] = [
        object(),
        {
            "scope": "",
            "period_start_ms": 0,
            "period_end_ms": _PERIOD_MS,
            "baseline_return": 0.0,
            "ai_return": 0.002,
        },
    ]

    deltas, binding, reasons = _matched_period_deltas(rows)  # type: ignore[arg-type]

    assert deltas == ()
    assert binding["sample_count"] == 0
    assert reasons == (
        "ai_uplift_period_0_not_mapping",
        "ai_uplift_period_1_invalid",
        "ai_uplift_period_rows_invalid",
        "ai_uplift_matched_periods_missing",
    )


def test_ai_uplift_rejects_period_scope_and_duration_mismatch() -> None:
    periods = _matched_periods((0.002, 0.002))
    periods[1]["scope"] = "ETHUSDT"
    periods[1]["period_end_ms"] = int(periods[1]["period_end_ms"]) + 1

    _deltas, _binding, reasons = _matched_period_deltas(periods)

    assert reasons == (
        "ai_uplift_period_scope_mismatch",
        "ai_uplift_period_duration_mismatch",
    )


def test_ai_uplift_rejects_invalid_metric_types_counts_and_ranges() -> None:
    baseline = _complete(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
        },
        _BASELINE_SHA256,
    )
    ai = _complete(
        {
            "realized_pnl": True,
            "roi_pct": 1.4,
            "max_drawdown": -0.035,
            "expectancy": 1.1,
            "closed_trades": 10.5,
            "win_rate": 1.1,
        },
        _AI_SHA256,
    )

    report = assess_ai_uplift(
        baseline,
        ai,
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert "ai_uplift_ai_realized_pnl_nonfinite" in report.reasons
    assert "ai_uplift_ai_closed_trades_invalid_count" in report.reasons
    assert "ai_uplift_ai_max_drawdown_invalid_range" in report.reasons
    assert "ai_uplift_ai_win_rate_invalid_range" in report.reasons


def test_ai_uplift_rejects_tail_risk_deterioration() -> None:
    report = assess_ai_uplift(
        _bound(
            {
                "realized_pnl": 20.0,
                "max_drawdown": 0.04,
                "expectancy": 0.8,
                "profit_factor": 1.8,
                "win_rate": 0.62,
                "closed_trades": 20,
                "max_consecutive_losses": 2,
                "downside_return_risk_ratio": 0.70,
                "liquidation_events": 0,
            },
            _BASELINE_SHA256,
        ),
        _bound(
            {
                "realized_pnl": 25.0,
                "max_drawdown": 0.04,
                "expectancy": 1.0,
                "profit_factor": 1.4,
                "win_rate": 0.55,
                "closed_trades": 22,
                "max_consecutive_losses": 4,
                "downside_return_risk_ratio": 0.60,
                "liquidation_events": 1,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert "ai_liquidation_events>0" in report.reasons
    assert "ai_loss_streak_worse_than_baseline" in report.reasons
    assert "ai_profit_factor_worse_than_baseline" in report.reasons
    assert "ai_win_rate_worse_than_baseline" in report.reasons
    assert "ai_downside_return_risk_not_above_baseline" in report.reasons


def test_ai_uplift_policy_can_require_stricter_model_size() -> None:
    report = assess_ai_uplift(
        _complete(
            {
                "realized_pnl": 10.0,
                "max_drawdown": 0.04,
                "expectancy": 0.5,
                "closed_trades": 8,
            },
            _BASELINE_SHA256,
        ),
        _complete(
            {
                "realized_pnl": 12.0,
                "max_drawdown": 0.03,
                "expectancy": 0.7,
                "closed_trades": 8,
            },
            _AI_SHA256,
        ),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
        policy=AIUpliftPolicy(min_model_parameters_b=13.0),
    )

    assert report.accepted is False
    assert "model_parameters<13.00B" in report.reasons


def test_ai_uplift_rejects_index_paired_trade_sequences() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 10.0,
            "max_drawdown": 0.04,
            "expectancy": 0.5,
            "closed_trades": 30,
            "trade_returns": [0.01] * 30,
        },
        {
            "realized_pnl": 12.0,
            "max_drawdown": 0.03,
            "expectancy": 0.7,
            "closed_trades": 30,
            "trade_returns": [0.02] * 30,
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is False
    assert "ai_uplift_unbound_trade_sequence_rejected" in report.reasons


@pytest.mark.parametrize(
    "override",
    [
        {"min_model_parameters_b": 1.0},
        {"min_paired_samples": 29},
        {"max_sign_test_p_value": 0.051},
        {"block_bootstrap_samples": 1_999},
        {"block_bootstrap_confidence": 0.949},
        {"min_evaluation_span_days": 89},
        {"require_evidence_binding": False},
        {"min_positive_delta_rate": 0.54},
        {"min_bootstrap_mean_delta_lower": -0.01},
        {"min_pnl_delta": -0.01},
        {"max_drawdown_delta": 0.01},
        {"min_downside_return_risk_delta": -0.01},
    ],
)
def test_ai_uplift_policy_cannot_weaken_mandatory_floors(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="cannot|mandatory"):
        AIUpliftPolicy(**override)


def test_ai_uplift_policy_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        AIUpliftPolicy(min_pnl_delta=float("nan"))


def test_ai_uplift_rejects_invalid_supplied_parameter_counts() -> None:
    assert _model_parameters_b("", float("nan")) is None
    assert _model_parameters_b("", True) is None  # type: ignore[arg-type]
    assert _model_parameters_b("", object()) is None  # type: ignore[arg-type]


def test_ai_uplift_rejects_missing_source_metrics_even_with_positive_periods() -> None:
    report = assess_ai_uplift(
        _bound({"realized_pnl": 10.0, "closed_trades": 30}, _BASELINE_SHA256),
        _bound({"realized_pnl": 12.0, "closed_trades": 30}, _AI_SHA256),
        model_name="qwen2.5:7b",
        model_artifact_sha256=_MODEL_SHA256,
        matched_periods=_matched_periods((0.002,) * 30),
    )

    assert report.accepted is False
    assert "ai_uplift_baseline_max_drawdown_missing" in report.reasons
    assert "ai_uplift_ai_expectancy_missing" in report.reasons
    assert "ai_uplift_baseline_liquidation_events_missing" in report.reasons
    assert "ai_uplift_ai_downside_return_risk_ratio_missing" in report.reasons
