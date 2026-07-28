from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from simple_ai_trading import round74_representation_comparison as subject


class _Policy:
    def __init__(
        self,
        profile: str,
        *,
        objective_bps: float,
        total_net_bps: float,
        maximum_drawdown_bps: float,
        adverse_selection_rate: float,
        mean_run_maximum_adverse_excursion_bps: float,
        identity: int,
        run_payoffs: tuple[float, ...],
    ) -> None:
        metrics = SimpleNamespace(
            total_net_bps=total_net_bps,
            maximum_drawdown_bps=maximum_drawdown_bps,
            adverse_selection_rate=adverse_selection_rate,
            mean_run_maximum_adverse_excursion_bps=(
                mean_run_maximum_adverse_excursion_bps
            ),
        )
        evaluation = SimpleNamespace(
            accepted=True,
            quantile=0.5,
            threshold_score=1.0,
            objective_bps=objective_bps,
            trace=SimpleNamespace(
                metrics=metrics,
                expected_run_ids=tuple(f"{index:032x}" for index in range(1, 7)),
                run_id=tuple(f"{index:032x}" for index in range(1, 7)),
                net_payoff_bps=run_payoffs,
            ),
        )
        self.profile = profile
        self.accepted = True
        self.selected_quantile = 0.5
        self.selected_threshold_score = 1.0
        self.evaluations = (evaluation,)
        self.selection_sha256 = f"{identity:064x}"

    def validate(self) -> None:
        return None


def _policies(
    *,
    conservative_drawdown: float = 2.0,
    conservative_run_payoffs: tuple[float, ...] = (5.0,) * 6,
    identity: int = 1,
):
    return tuple(
        _Policy(
            profile,
            objective_bps=5.0,
            total_net_bps=sum(
                conservative_run_payoffs if profile == "conservative" else (5.0,) * 6
            ),
            maximum_drawdown_bps=(
                conservative_drawdown if profile == "conservative" else 3.0
            ),
            adverse_selection_rate=0.1,
            mean_run_maximum_adverse_excursion_bps=1.0,
            identity=identity + index,
            run_payoffs=(
                conservative_run_payoffs if profile == "conservative" else (5.0,) * 6
            ),
        )
        for index, profile in enumerate(("conservative", "regular", "aggressive"))
    )


def test_representation_proper_loss_gate_requires_every_run_noninferior() -> None:
    promoted = subject.round74_representation_proper_loss_gate(
        (1.0,) * 12,
        (0.9,) * 12,
        minimum_mean_improvement=0.01,
    )
    assert promoted["promoted"] is True
    assert promoted["challenger_win_count"] == 12
    assert promoted["sealed_test_accessed"] is False

    one_bad_run = subject.round74_representation_proper_loss_gate(
        (1.0,) * 12,
        (0.9,) * 11 + (1.1,),
        minimum_mean_improvement=0.01,
    )
    assert one_bad_run["mean_proper_loss_improvement"] > 0.01
    assert one_bad_run["all_paired_runs_noninferior"] is False
    assert one_bad_run["promoted"] is False

    one_run_driven = subject.round74_representation_proper_loss_gate(
        (1.0,) * 12,
        (0.0,) + (0.999989,) * 7 + (1.0,) * 4,
        minimum_mean_improvement=1e-5,
    )
    assert one_run_driven["material_win_majority"] is True
    assert (
        one_run_driven[
            "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement"
        ]
        is False
    )
    assert one_run_driven["promoted"] is False

    with pytest.raises(ValueError, match="proper-loss panel differs"):
        subject.round74_representation_proper_loss_gate(
            (1.0,),
            (0.9,),
            minimum_mean_improvement=0.01,
        )


def test_representation_economic_gate_rejects_conservative_drawdown_increase() -> None:
    baseline = _policies(conservative_drawdown=2.0, identity=1)
    equal = subject.round74_representation_economic_gate(
        baseline,
        _policies(conservative_drawdown=2.0, identity=10),
    )
    assert all(report["noninferior"] is True for report in equal)

    degraded = subject.round74_representation_economic_gate(
        baseline,
        _policies(conservative_drawdown=2.1, identity=20),
    )
    conservative = degraded[0]
    assert conservative["profile"] == "conservative"
    assert conservative["noninferior"] is False
    assert "maximum_drawdown_increased" in conservative["reasons"]
    assert all(report["sealed_test_accessed"] is False for report in degraded)

    mismatched = _policies(identity=30)
    mismatched[0].evaluations[0].trace.expected_run_ids = tuple(
        reversed(mismatched[0].evaluations[0].trace.expected_run_ids)
    )
    with pytest.raises(ValueError, match="economic run identity differs"):
        subject.round74_representation_economic_gate(baseline, mismatched)


def test_representation_economic_gate_rejects_hidden_conservative_run_loss() -> None:
    baseline = _policies(identity=100)
    challenger = _policies(
        conservative_run_payoffs=(4.0, 5.3, 5.3, 5.3, 5.3, 5.3),
        identity=110,
    )

    reports = subject.round74_representation_economic_gate(
        baseline,
        challenger,
    )

    conservative = reports[0]
    assert (
        conservative["challenger_selected_metrics"]["total_net_bps"]
        > (conservative["baseline_selected_metrics"]["total_net_bps"])
    )
    assert conservative["minimum_paired_run_net_delta_bps"] == -1.0
    assert conservative["challenger_worse_run_count"] == 1
    assert conservative["all_paired_runs_net_noninferior"] is False
    assert conservative["noninferior"] is False
    assert "conservative_paired_run_net_payoff_degraded" in conservative["reasons"]


def test_representation_comparison_cannot_overstate_authority() -> None:
    proper = subject.round74_representation_proper_loss_gate(
        (1.0,) * 12,
        (1.0,) * 12,
        minimum_mean_improvement=0.01,
    )
    profiles = subject.round74_representation_economic_gate(
        _policies(identity=30),
        _policies(identity=40),
    )
    comparison = subject.Round74RepresentationComparison(
        matched_preparation_sha256="1" * 64,
        baseline_pretest_policy_sha256="2" * 64,
        challenger_pretest_policy_sha256="3" * 64,
        baseline_development_bundle_sha256="4" * 64,
        challenger_development_bundle_sha256="5" * 64,
        fixed_candidate_id="event_pooling_linear",
        seeds=(7411, 7423, 7433),
        proper_loss_gate=proper,
        profile_economic_gates=profiles,
        selected_representation="per_symbol",
        promoted=False,
        source_module_sha256=subject._module_sha256(),
    )
    comparison.validate()
    payload = comparison.as_dict()
    assert len(payload["comparison_sha256"]) == 64
    assert payload["sealed_test_accessed"] is False
    assert payload["profitability_claim"] is False
    assert payload["live_trading_authority"] is False
    assert (
        subject.Round74RepresentationComparison.from_dict(payload).as_dict() == payload
    )

    overstated = dict(payload)
    overstated["live_trading_authority"] = True
    overstated.pop("comparison_sha256")
    overstated["comparison_sha256"] = subject._canonical_sha256(overstated)
    with pytest.raises(ValueError, match="payload differs"):
        subject.Round74RepresentationComparison.from_dict(overstated)

    with pytest.raises(ValueError, match="comparison differs"):
        replace(
            comparison,
            promoted=True,
            selected_representation="global_cross_asset",
        ).validate()

    forged_economics = deepcopy(payload)
    forged_economics["profile_economic_gates"][0][
        "minimum_paired_run_net_delta_bps"
    ] = -1.0
    forged_economics.pop("comparison_sha256")
    forged_economics["comparison_sha256"] = subject._canonical_sha256(forged_economics)
    with pytest.raises(ValueError, match="paired economics differ"):
        subject.Round74RepresentationComparison.from_dict(forged_economics)


def test_representation_coordinator_fixes_challenger_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, object]] = []
    partition = SimpleNamespace()
    assemblies = {"policy-run": object()}
    inputs = SimpleNamespace(
        partition=partition,
        validate=lambda: calls.append(("inputs", None)),
        target_assembly_by_run_id=lambda: assemblies,
    )
    prepared_by_representation = {
        representation: SimpleNamespace(marker=representation)
        for representation in ("per_symbol", "global_cross_asset")
    }
    prepared = SimpleNamespace(
        preparation_sha256="a" * 64,
        representation=lambda value: prepared_by_representation[value],
    )
    subpartition = SimpleNamespace(policy_selection_run_ids=("policy-run",))
    roles = {
        representation: SimpleNamespace(marker=representation)
        for representation in ("per_symbol", "global_cross_asset")
    }
    monkeypatch.setattr(
        subject,
        "prepare_round74_matched_development_data",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        subject,
        "build_round74_tuning_subpartition",
        lambda _partition: subpartition,
    )
    monkeypatch.setattr(
        subject,
        "split_round74_prepared_tuning_roles",
        lambda selected, **_kwargs: roles[selected.marker],
    )

    artifacts = {}

    def train(selected, selected_roles, *, config, **kwargs):
        representation = selected.marker
        assert selected_roles.marker == representation
        calls.append((representation, config))
        assert kwargs["matched_preparation_sha256"] == "a" * 64
        selected_candidate = (
            "causal_event_tcn"
            if representation == "per_symbol"
            else config.candidate_ids[0]
        )
        artifact = SimpleNamespace(
            pretest_policy=SimpleNamespace(
                policy_path=tmp_path / f"{representation}.json",
                selected_candidate_id=selected_candidate,
            )
        )
        artifacts[representation] = artifact
        return artifact

    monkeypatch.setattr(
        subject,
        "train_calibrate_and_select_round74_development_policy",
        train,
    )
    monkeypatch.setattr(
        subject,
        "load_round74_pretest_policy",
        lambda path: (object(), {"path": str(path)}),
    )
    comparison = SimpleNamespace(
        comparison_sha256="b" * 64,
        as_dict=lambda: {"comparison_sha256": "b" * 64},
    )

    def compare(
        selected,
        *,
        baseline,
        challenger,
        **_kwargs,
    ):
        assert selected is prepared
        assert baseline is artifacts["per_symbol"]
        assert challenger is artifacts["global_cross_asset"]
        return comparison

    monkeypatch.setattr(subject, "build_round74_representation_comparison", compare)
    monkeypatch.setattr(
        subject,
        "load_round74_representation_comparison",
        lambda _path: comparison,
    )

    result = subject.train_and_compare_round74_representations(
        object(),
        inputs,
        output_directory=tmp_path,
        compute_backend="cpu",
    )

    baseline_config = calls[1][1]
    challenger_config = calls[2][1]
    assert baseline_config.architecture_selection_mode == "complexity_gate"
    assert challenger_config.architecture_selection_mode == "fixed"
    assert challenger_config.candidate_ids == ("causal_event_tcn",)
    assert result.baseline is artifacts["per_symbol"]
    assert result.challenger is artifacts["global_cross_asset"]


def test_build_comparison_binds_matched_batches_and_selection_modes() -> None:
    candidate = "causal_event_tcn"
    preparation_sha256 = "a" * 64

    def batches(start: int):
        return tuple(
            SimpleNamespace(batch_sha256=f"{start + index:064x}") for index in range(24)
        )

    per_symbol_batches = batches(100)
    global_batches = batches(200)
    prepared = SimpleNamespace(
        validate=lambda: None,
        preparation_sha256=preparation_sha256,
        tuning=SimpleNamespace(
            per_symbol=per_symbol_batches,
            global_cross_asset=global_batches,
        ),
    )

    def artifact(
        identity: int,
        selected_batches,
        policies,
    ):
        bundle_sha256 = f"{identity:064x}"
        bundle = SimpleNamespace(
            validate=lambda: None,
            bundle_sha256=bundle_sha256,
            model_selection_batch_sha256=tuple(
                batch.batch_sha256 for batch in selected_batches[:12]
            ),
            calibration_batch_sha256=tuple(
                batch.batch_sha256 for batch in selected_batches[12:18]
            ),
            policy_selection_batch_sha256=tuple(
                batch.batch_sha256 for batch in selected_batches[18:]
            ),
            feature_scaler_sha256="e" * 64,
            tuning_subpartition_sha256="f" * 64,
            action_policies=policies,
        )
        return SimpleNamespace(
            bundle_sha256=bundle_sha256,
            bundle=bundle,
            pretest_policy=SimpleNamespace(policy_sha256=f"{identity + 10:064x}"),
        )

    baseline = artifact(1, per_symbol_batches, _policies(identity=50))
    challenger = artifact(2, global_batches, _policies(identity=60))

    def policy(
        artifact_value,
        *,
        representation: str,
        mode: str,
        losses: tuple[float, ...],
    ):
        candidate_ids = (
            list(subject.ROUND74_EVENT_MODEL_CANDIDATES)
            if mode == "complexity_gate"
            else [candidate]
        )
        return {
            "policy_sha256": artifact_value.pretest_policy.policy_sha256,
            "development_data": {
                "window_representation": representation,
                "representative_window_policy_kind": "matched_representation",
                "matched_preparation_sha256": preparation_sha256,
            },
            "training_policy": {
                "seeds": [7411, 7423, 7433],
                "candidate_ids": candidate_ids,
                "architecture_selection_mode": mode,
            },
            "selection": {
                "selected_candidate_id": candidate,
                "architecture_selection_mode": mode,
            },
            "candidate_panel": {
                candidate: {"ensemble_tuning_run_losses": list(losses)}
            },
        }

    comparison = subject.build_round74_representation_comparison(
        prepared,
        baseline=baseline,
        challenger=challenger,
        baseline_policy=policy(
            baseline,
            representation="per_symbol",
            mode="complexity_gate",
            losses=(1.0,) * 12,
        ),
        challenger_policy=policy(
            challenger,
            representation="global_cross_asset",
            mode="fixed",
            losses=(0.9,) * 12,
        ),
        minimum_mean_improvement=0.01,
    )

    assert comparison.promoted is True
    assert comparison.selected_representation == "global_cross_asset"
    assert comparison.fixed_candidate_id == candidate

    wrong_batches = SimpleNamespace(
        validate=lambda: None,
        preparation_sha256=preparation_sha256,
        tuning=SimpleNamespace(
            per_symbol=per_symbol_batches,
            global_cross_asset=global_batches[:-1] + per_symbol_batches[-1:],
        ),
    )
    with pytest.raises(ValueError, match="development binding differs"):
        subject.build_round74_representation_comparison(
            wrong_batches,
            baseline=baseline,
            challenger=challenger,
            baseline_policy=policy(
                baseline,
                representation="per_symbol",
                mode="complexity_gate",
                losses=(1.0,) * 12,
            ),
            challenger_policy=policy(
                challenger,
                representation="global_cross_asset",
                mode="fixed",
                losses=(0.9,) * 12,
            ),
            minimum_mean_improvement=0.01,
        )
