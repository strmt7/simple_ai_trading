from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from simple_ai_trading import polymarket_mlp as mlp
from simple_ai_trading.polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES
from simple_ai_trading.assets import SUPPORTED_MAJOR_BASE_ASSETS
from simple_ai_trading.polymarket_ridge import (
    POLYMARKET_RIDGE_THRESHOLD_GRID,
    PolymarketPolicyMetrics,
    PolymarketRidgeSplit,
)


_BACKEND_IDENTITY_SHA256 = (
    "a301c1208b5c09a7f81248a068c7ece46daa829fd8b600c31ca857ebfe14f2e8"
)
_MEMBER_SHA256 = {
    4701: "6acdd3b2937496b70e7ace3a74fdd74f0dc24ed2d5b7f66b41d6dc9f60c2c54c",
    4702: "9b0bb6a614635774341790b977df578502274faa7d7681ca6adfa2fd9033f947",
    4703: "9076ecbe6bfd1cfc6dc2aba80d9516bf2012d593e380d675742065eb68a64716",
}
_ENSEMBLE_SHA256 = "da8e3a83be90a57fa4bb85c0b6409ec583b73f19971cb2ae8307fdffc4787690"
_NO_TRADE_REPORT_SHA256 = (
    "f24c8a92186309073c4d07be3c1ce2fbc3012af5a59eda3e77e8b38b35a90f9c"
)
_ADMITTED_REPORT_SHA256 = (
    "9c471198599a852778f6e0500245095a201effda708e969eea1aff82540da05b"
)
_FEATURE_COUNT = len(POLYMARKET_ACTION_FEATURE_NAMES)
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)


def _sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _backend() -> mlp.PolymarketMLPBackendEvidence:
    return mlp.PolymarketMLPBackendEvidence(
        requested="cpu",
        kind="cpu",
        device="cpu",
        vendor="test",
        fallback_reason="",
        torch_version="test",
        torch_directml_version="",
        preflight_objective=1.0,
        preflight_parameter_delta=1.0,
        preflight_seconds=0.1,
        training_seconds=1.0,
        canonical_replay_max_probability_drift=0.0,
    )


def _member(seed: int) -> mlp.PolymarketMLPMember:
    trace = (
        mlp.PolymarketMLPEpoch(
            seed=seed,
            epoch=1,
            training_loss=0.5,
            validation_log_loss=0.4,
        ),
    )
    provisional = mlp.PolymarketMLPMember(
        seed=seed,
        best_epoch=1,
        epochs_ran=1,
        hidden1_weight=(0.0,) * (64 * _FEATURE_COUNT),
        hidden1_bias=(0.0,) * 64,
        hidden2_weight=(0.0,) * (32 * 64),
        hidden2_bias=(0.0,) * 32,
        output_weight=(0.0,) * 32,
        output_bias=0.0,
        trace=trace,
        member_sha256="",
    )
    return replace(
        provisional,
        member_sha256=_sha256(provisional.identity_payload()),
    )


def _ensemble() -> mlp.PolymarketMLPEnsemble:
    provisional = mlp.PolymarketMLPEnsemble(
        dataset_sha256="a" * 64,
        feature_mean=(0.0,) * _FEATURE_COUNT,
        feature_scale=(1.0,) * _FEATURE_COUNT,
        members=tuple(_member(seed) for seed in mlp.POLYMARKET_MLP_SEEDS),
        backend=_backend(),
        reproducibility_max_probability_drift=0.0,
        ensemble_sha256="",
    )
    return replace(
        provisional,
        ensemble_sha256=_sha256(provisional.identity_payload()),
    )


def _wilson_lower_bound(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    z_score = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z_score * z_score / trials
    center = probability + z_score * z_score / (2.0 * trials)
    margin = z_score * math.sqrt(
        probability * (1.0 - probability) / trials
        + z_score * z_score / (4.0 * trials * trials)
    )
    return max(0.0, (center - margin) / denominator)


def _policy_metrics(
    threshold: float,
    *,
    passed: bool,
) -> PolymarketPolicyMetrics:
    if passed:
        return PolymarketPolicyMetrics(
            threshold=threshold,
            attempt_count=30,
            completed_trade_count=30,
            completed_by_asset={asset: 10 for asset in _ASSETS},
            positive_complete_count=30,
            failed_exit_count=0,
            aggregate_stress_utility_quote=3.0,
            pnl_by_asset={asset: 1.0 for asset in _ASSETS},
            median_market_pnl_quote=1.0,
            maximum_realized_drawdown_quote=0.0,
            positive_complete_precision=1.0,
            wilson_lower_bound_95=_wilson_lower_bound(30, 30),
            selected_action_sha256="b" * 64,
            gate_passed=True,
            gate_reasons=(),
        )
    reasons = tuple(
        sorted(
            [
                "aggregate_stress_utility_not_positive",
                "completed_trade_count:0/30",
                *(f"completed_trade_count:{asset}:0/5" for asset in _ASSETS),
                "median_market_pnl_not_positive",
            ]
        )
    )
    return PolymarketPolicyMetrics(
        threshold=threshold,
        attempt_count=0,
        completed_trade_count=0,
        completed_by_asset={asset: 0 for asset in _ASSETS},
        positive_complete_count=0,
        failed_exit_count=0,
        aggregate_stress_utility_quote=0.0,
        pnl_by_asset={asset: 0.0 for asset in _ASSETS},
        median_market_pnl_quote=0.0,
        maximum_realized_drawdown_quote=0.0,
        positive_complete_precision=0.0,
        wilson_lower_bound_95=0.0,
        selected_action_sha256="c" * 64,
        gate_passed=False,
        gate_reasons=reasons,
    )


def _bootstrap(*, sample_count: int = 6) -> mlp.PolymarketMLPBootstrap:
    return mlp.PolymarketMLPBootstrap(
        sample_count=sample_count,
        block_length=max(1, int(round(math.sqrt(sample_count)))),
        resamples=mlp.POLYMARKET_MLP_BOOTSTRAP_SAMPLES,
        mean_delta=1.0,
        lower_95=0.5,
        upper_95=1.5,
        positive_mean_probability=1.0,
        values_sha256="d" * 64,
    )


def _report(*, admitted: bool) -> mlp.PolymarketMLPReport:
    split = PolymarketRidgeSplit(
        train_groups=tuple(range(1, 17)),
        validation_groups=tuple(range(18, 24)),
        test_groups=tuple(range(25, 31)),
        purged_groups=(17, 24),
    )
    trials = tuple(
        _policy_metrics(threshold, passed=threshold == 0.5)
        for threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
    )
    provisional = mlp.PolymarketMLPReport(
        dataset_sha256="a" * 64,
        parent_ridge_report_sha256="e" * 64,
        split=split,
        ensemble=_ensemble(),
        validation_log_loss=0.4,
        ridge_validation_log_loss=0.5,
        validation_log_loss_uplift=_bootstrap(),
        validation_trials=trials,
        validation_stress_utility_uplift_quote=1.0 if admitted else 0.0,
        validation_admission_reasons=(
            () if admitted else ("validation_stress_utility_not_above_ridge",)
        ),
        selected_policy="causal_mlp" if admitted else "no_trade",
        selected_threshold=0.5 if admitted else None,
        test_evaluated=admitted,
        test_log_loss=0.3 if admitted else None,
        test_metrics=_policy_metrics(0.5, passed=True) if admitted else None,
        test_utility_uplift=_bootstrap() if admitted else None,
        test_gate_reasons=(),
        development_passed=admitted,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    )


def test_mlp_validation_preserves_canonical_identities() -> None:
    backend = _backend()
    ensemble = _ensemble()

    assert backend.validated() is backend
    assert _sha256(backend.identity_payload()) == _BACKEND_IDENTITY_SHA256
    assert ensemble.validated() is ensemble
    assert {
        item.seed: item.member_sha256 for item in ensemble.members
    } == _MEMBER_SHA256
    assert ensemble.ensemble_sha256 == _ENSEMBLE_SHA256


@pytest.mark.parametrize(
    ("admitted", "expected_sha256"),
    [
        (False, _NO_TRADE_REPORT_SHA256),
        (True, _ADMITTED_REPORT_SHA256),
    ],
)
def test_mlp_report_validation_preserves_canonical_states(
    admitted: bool,
    expected_sha256: str,
) -> None:
    report = _report(admitted=admitted)

    assert report.validated() is report
    assert report.report_sha256 == expected_sha256


@pytest.mark.parametrize(
    "report",
    [
        replace(_report(admitted=False), dataset_sha256="f" * 64),
        replace(
            _report(admitted=False),
            validation_trials=tuple(
                reversed(_report(admitted=False).validation_trials)
            ),
        ),
        replace(
            _report(admitted=False),
            validation_trials=tuple(
                _policy_metrics(threshold, passed=False)
                for threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
            ),
        ),
        replace(
            _report(admitted=False),
            validation_log_loss_uplift=replace(_bootstrap(), lower_95=0.0),
        ),
        replace(_report(admitted=False), selected_policy="causal_mlp"),
        replace(_report(admitted=False), validation_log_loss=float("nan")),
        replace(
            _report(admitted=False),
            validation_stress_utility_uplift_quote=None,
        ),
        replace(_report(admitted=False), test_log_loss=0.3),
        replace(
            _report(admitted=True),
            test_log_loss=float("nan"),
        ),
        replace(
            _report(admitted=True),
            test_gate_reasons=("duplicate", "duplicate"),
            development_passed=False,
        ),
        replace(_report(admitted=False), report_sha256="f" * 64),
    ],
)
def test_mlp_report_validation_preserves_fail_closed_rejections(
    report: mlp.PolymarketMLPReport,
) -> None:
    with pytest.raises(ValueError, match="Polymarket MLP report is invalid"):
        report.validated()


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (replace(_backend(), requested="directml"), "backend evidence"),
        (replace(_backend(), device=""), "backend evidence"),
        (replace(_backend(), training_seconds=0.0), "backend evidence"),
        (replace(_member(4701), best_epoch=2), "member"),
        (replace(_member(4701), hidden1_bias=()), "member"),
        (
            replace(
                _member(4701),
                trace=(replace(_member(4701).trace[0], seed=4702),),
            ),
            "member",
        ),
        (
            replace(
                _ensemble(),
                feature_scale=(0.0,) * _FEATURE_COUNT,
            ),
            "ensemble",
        ),
        (replace(_ensemble(), members=_ensemble().members[:-1]), "ensemble"),
        (
            replace(
                _ensemble(),
                reproducibility_max_probability_drift=0.1,
            ),
            "ensemble",
        ),
    ],
)
def test_mlp_validation_preserves_fail_closed_rejections(
    evidence: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evidence.validated()  # type: ignore[attr-defined]
