from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round21_operator as operator_module
from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_CORE_FEATURE_NAMES,
    Round21CoreFeatureSnapshot,
)
from simple_ai_trading.polymarket_round21_dataset import (
    POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS,
    Round21OfficialOutcome,
    Round21PartitionPolicy,
)
from simple_ai_trading.polymarket_round21_binance_features import (
    Round21OptionalBinanceFeatures,
)
from simple_ai_trading.polymarket_round21_operator import (
    apply_round21_optional_binance_features,
    assemble_round21_core_development,
    assemble_round21_matched_development,
    build_round21_core_causal_rows,
    evaluate_round21_core_probability_basis,
    fit_round21_core_baseline,
    fit_round21_matched_optional_candidate,
    load_round21_official_outcomes,
)
from simple_ai_trading.polymarket_round21_sidecar_replay import Round21SidecarReplay


START_MS = 1_800_001_200_000
TRANSPORT_SHA = "a" * 64
PUBLICATION_SHA = "b" * 64
RUN_ID = "c" * 32


def _snapshot(*, digit: str, event_start_ms: int) -> Round21CoreFeatureSnapshot:
    return Round21CoreFeatureSnapshot(
        condition_id="0x" + digit * 64,
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        available=True,
        reasons=(),
        structural_probability=0.51,
        market_prior_probability=0.49,
        values=(0.01,) * len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES),
        source_chain_sha256=digit * 64,
        maximum_receipt_ms=event_start_ms + 29_999,
    )


def _transport() -> dict[str, object]:
    return {
        "manifest_sha256": TRANSPORT_SHA,
        "segments": [
            {
                "run_id": RUN_ID,
                "eligible_for_condition_rebuild": True,
            }
        ],
    }


def _development_snapshots() -> tuple[Round21CoreFeatureSnapshot, ...]:
    starts = (
        START_MS,
        START_MS + 300_000,
        START_MS + 1_557_000_000,
        START_MS + 1_557_300_000,
        START_MS + 1_771_200_000,
        START_MS + 1_771_500_000,
    )
    return tuple(
        _snapshot(digit=str(index + 1), event_start_ms=start)
        for index, start in enumerate(starts)
    )


def _outcomes(
    snapshots: tuple[Round21CoreFeatureSnapshot, ...],
) -> tuple[Round21OfficialOutcome, ...]:
    return tuple(
        Round21OfficialOutcome.create(
            condition_id=snapshot.condition_id,
            event_start_ms=snapshot.event_start_ms,
            resolved_up=index % 2 == 0,
            observed_at_ms=snapshot.event_start_ms + 300_000,
            source="fixture_consensus",
            source_payload_sha256=str(index + 1) * 64,
        )
        for index, snapshot in enumerate(snapshots)
    )


def _optional_replay(
    snapshots: tuple[Round21CoreFeatureSnapshot, ...],
) -> Round21SidecarReplay:
    features = tuple(
        Round21OptionalBinanceFeatures(
            decision_time_ms=snapshot.decision_time_ms,
            spot_values=(0.02,)
            * len(operator_module.POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names),
            usdm_values=(0.03,)
            * len(operator_module.POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names),
            spot_available=True,
            usdm_available=True,
            spot_source_chain_sha256="d" * 64,
            usdm_source_chain_sha256="e" * 64,
            spot_maximum_receipt_ms=snapshot.decision_time_ms - 2,
            usdm_maximum_receipt_ms=snapshot.decision_time_ms - 1,
        )
        for snapshot in snapshots
    )
    return Round21SidecarReplay(
        terminal_manifest_sha256="f" * 64,
        eligible_run_ids=("a" * 32,),
        decision_times_ms=tuple(snapshot.decision_time_ms for snapshot in snapshots),
        features=features,
        raw_message_count=2,
        stream_counts={"binance_futures": 1, "binance_spot": 1},
        stream_gap_count=0,
        receipt_chain_sha256="c" * 64,
    ).validated()


def test_core_rows_preserve_exact_values_and_explicit_optional_missingness() -> None:
    snapshot = _snapshot(digit="1", event_start_ms=START_MS)

    row = build_round21_core_causal_rows((snapshot,))[0]

    assert row.core_values == snapshot.values
    assert row.core_source_chain_sha256 == snapshot.source_chain_sha256
    assert row.spot_available is False
    assert row.usdm_available is False
    assert not any(row.spot_values)
    assert not any(row.usdm_values)


def test_core_rows_reject_unavailable_and_duplicate_snapshots() -> None:
    snapshot = _snapshot(digit="1", event_start_ms=START_MS)
    unavailable = replace(
        snapshot,
        available=False,
        reasons=("missing",),
        structural_probability=0.5,
        market_prior_probability=0.5,
        values=(0.0,) * len(snapshot.values),
        source_chain_sha256=operator_module._EMPTY_SHA256,
        maximum_receipt_ms=0,
    )

    with pytest.raises(ValueError, match="snapshot population differs"):
        build_round21_core_causal_rows((unavailable,))
    with pytest.raises(ValueError, match="contains duplicates"):
        build_round21_core_causal_rows((snapshot, snapshot))


def test_optional_join_preserves_core_identity_and_uses_exact_decisions() -> None:
    snapshots = _development_snapshots()
    core = build_round21_core_causal_rows(snapshots)
    replay = _optional_replay(snapshots)

    rows = apply_round21_optional_binance_features(core, replay)

    assert tuple(row.row_sha256 for row in rows) != tuple(
        row.row_sha256 for row in core
    )
    assert tuple(row.core_values for row in rows) == tuple(
        row.core_values for row in core
    )
    assert all(row.spot_available and row.usdm_available for row in rows)
    changed_times = (replay.decision_times_ms[0] + 1,) + replay.decision_times_ms[1:]
    changed = replace(
        replay,
        decision_times_ms=changed_times,
        features=(
            replace(replay.features[0], decision_time_ms=changed_times[0]),
            *replay.features[1:],
        ),
    )
    with pytest.raises(ValueError, match="join population differs"):
        apply_round21_optional_binance_features(core, changed)


def test_official_outcomes_use_only_eligible_consensus_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = "0x" + "1" * 64

    class FakeStore:
        def __init__(self, path: Path, **kwargs: object) -> None:
            assert path == tmp_path / "source.duckdb"
            assert kwargs == {"read_only": True, "memory_limit": "1GB", "threads": 2}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(operator_module, "PolymarketEvidenceStore", FakeStore)
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module,
        "load_official_resolutions",
        lambda _store, *, run_id: (
            (
                SimpleNamespace(
                    condition_id=condition,
                    asset="BTC",
                    winning_outcome="Up",
                    observed_wall_ms=START_MS + 300_000,
                    evidence_sha256="d" * 64,
                ),
            )
            if run_id == RUN_ID
            else ()
        ),
    )

    outcomes = load_round21_official_outcomes(
        source_database=tmp_path / "source.duckdb",
        terminal_transport_manifest=_transport(),
        condition_event_starts={condition: START_MS},
    )

    assert len(outcomes) == 1
    assert outcomes[0].resolved_up is True
    assert outcomes[0].source_payload_sha256 == "d" * 64


def test_official_outcomes_reject_missing_or_duplicate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = "0x" + "1" * 64

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(operator_module, "PolymarketEvidenceStore", FakeStore)
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module, "load_official_resolutions", lambda *_a, **_k: ()
    )
    with pytest.raises(ValueError, match="population is incomplete"):
        load_round21_official_outcomes(
            source_database=tmp_path / "source.duckdb",
            terminal_transport_manifest=_transport(),
            condition_event_starts={condition: START_MS},
        )

    resolution = SimpleNamespace(
        condition_id=condition,
        asset="BTC",
        winning_outcome="Down",
        observed_wall_ms=START_MS + 300_000,
        evidence_sha256="d" * 64,
    )
    monkeypatch.setattr(
        operator_module,
        "load_official_resolutions",
        lambda *_args, **_kwargs: (resolution, resolution),
    )
    with pytest.raises(ValueError, match="population differs"):
        load_round21_official_outcomes(
            source_database=tmp_path / "source.duckdb",
            terminal_transport_manifest=_transport(),
            condition_event_starts={condition: START_MS},
        )


def test_assembly_builds_all_frozen_development_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = _development_snapshots()
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=START_MS,
        campaign_end_ms=START_MS + POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS,
    )
    publication = {
        "manifest_sha256": PUBLICATION_SHA,
        "terminal_transport_manifest_sha256": TRANSPORT_SHA,
        "development_partition": {
            "campaign_start_ms": policy.campaign_start_ms,
            "campaign_end_ms": policy.campaign_end_ms,
            "partition_policy_sha256": policy.policy_sha256,
        },
    }
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module,
        "load_round21_core_development_publication",
        lambda _path: (publication, snapshots),
    )
    monkeypatch.setattr(
        operator_module,
        "load_round21_official_outcomes",
        lambda **_kwargs: _outcomes(snapshots),
    )

    assembly = assemble_round21_core_development(
        publication_directory=tmp_path / "publication",
        source_database=tmp_path / "source.duckdb",
        terminal_transport_manifest=_transport(),
    )

    assert assembly.population_layer == "core"
    assert assembly.outcome_count == 6
    assert assembly.train.role == "train"
    assert assembly.tune_calibration.role == "tune_calibration"
    assert assembly.tune_selection.role == "tune_selection"
    assert not assembly.train.spot_available.any()
    assert not assembly.train.usdm_available.any()


def test_matched_assembly_replays_optional_features_on_same_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = _development_snapshots()
    rows = build_round21_core_causal_rows(snapshots)
    outcomes = _outcomes(snapshots)
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=START_MS,
        campaign_end_ms=START_MS + POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS,
    )
    publication = {
        "manifest_sha256": PUBLICATION_SHA,
        "terminal_transport_manifest_sha256": TRANSPORT_SHA,
    }
    sidecar_terminal = {
        "manifest_sha256": "f" * 64,
        "campaign_start_ms": policy.campaign_start_ms,
        "campaign_end_ms": policy.campaign_end_ms,
    }
    replay = _optional_replay(snapshots)
    monkeypatch.setattr(
        operator_module,
        "_load_round21_core_development_evidence",
        lambda **_kwargs: (publication, policy, rows, outcomes),
    )
    monkeypatch.setattr(
        operator_module,
        "validate_round21_sidecar_terminal_manifest",
        lambda value: value,
    )
    observed: dict[str, object] = {}

    def fake_replay(**kwargs: object) -> Round21SidecarReplay:
        observed.update(kwargs)
        return replay

    monkeypatch.setattr(
        operator_module,
        "replay_round21_optional_binance_features",
        fake_replay,
    )

    assembly = assemble_round21_matched_development(
        publication_directory=tmp_path / "publication",
        source_database=tmp_path / "core.duckdb",
        terminal_transport_manifest=_transport(),
        sidecar_database=tmp_path / "sidecar.duckdb",
        sidecar_terminal_manifest=sidecar_terminal,
    )

    assert assembly.outcome_count == len(outcomes)
    assert assembly.sidecar_raw_message_count == 2
    assert assembly.sidecar_terminal_manifest_sha256 == "f" * 64
    assert assembly.train.spot_available.all()
    assert assembly.tune_calibration.usdm_available.all()
    assert observed["decision_times_ms"] == replay.decision_times_ms
    assert observed["source_database"] == tmp_path / "sidecar.duckdb"


def test_assembly_rejects_transport_drift_and_fit_uses_exact_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = _development_snapshots()
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=START_MS,
        campaign_end_ms=START_MS + POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS,
    )
    publication = {
        "manifest_sha256": PUBLICATION_SHA,
        "terminal_transport_manifest_sha256": "f" * 64,
        "development_partition": {
            "campaign_start_ms": policy.campaign_start_ms,
            "campaign_end_ms": policy.campaign_end_ms,
            "partition_policy_sha256": policy.policy_sha256,
        },
    }
    monkeypatch.setattr(
        operator_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        operator_module,
        "load_round21_core_development_publication",
        lambda _path: (publication, snapshots),
    )
    with pytest.raises(ValueError, match="publication and terminal transport differ"):
        assemble_round21_core_development(
            publication_directory=tmp_path / "publication",
            source_database=tmp_path / "source.duckdb",
            terminal_transport_manifest=_transport(),
        )

    expected = SimpleNamespace(
        train=object(),
        tune_calibration=object(),
        tune_selection=object(),
        publication_manifest_sha256=PUBLICATION_SHA,
        terminal_transport_manifest_sha256=TRANSPORT_SHA,
    )
    monkeypatch.setattr(
        operator_module, "assemble_round21_core_development", lambda **_k: expected
    )
    observed: dict[str, object] = {}

    def fake_fit(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"artifact_sha256": "e" * 64}

    monkeypatch.setattr(operator_module, "fit_round21_development", fake_fit)
    gate_calls: list[dict[str, object]] = []

    def fake_gate(value, **kwargs):
        gate_calls.append({"value": value, **kwargs})
        return value

    monkeypatch.setattr(
        operator_module,
        "_require_accepted_round21_probability_basis",
        fake_gate,
    )
    basis_ablation = {"result_sha256": "8" * 64, "basis_accepted": True}
    artifact = fit_round21_core_baseline(
        publication_directory=tmp_path / "publication",
        source_database=tmp_path / "source.duckdb",
        terminal_transport_manifest=_transport(),
        basis_ablation_result=basis_ablation,
        compute_backend="directml",
    )
    assert artifact["artifact_sha256"] == "e" * 64
    assert observed == {
        "train": expected.train,
        "tune_calibration": expected.tune_calibration,
        "tune_selection": expected.tune_selection,
        "basis_ablation_result": basis_ablation,
        "compute_backend": "directml",
        "feature_layers": ("core",),
    }
    assert gate_calls == [
        {
            "value": basis_ablation,
            "assembly": expected,
            "require_exact_dataset_identity": True,
        }
    ]

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        operator_module,
        "load_round21_probability_basis_ablation_design",
        lambda repository: calls.append(("design", repository)),
    )

    def fake_ablation(**kwargs: object) -> dict[str, object]:
        calls.append(("evaluate", kwargs))
        return {"result_sha256": "9" * 64, "basis_accepted": False}

    monkeypatch.setattr(
        operator_module,
        "evaluate_round21_probability_basis_ablation",
        fake_ablation,
    )
    ablation = evaluate_round21_core_probability_basis(
        repository=tmp_path,
        publication_directory=tmp_path / "publication",
        source_database=tmp_path / "source.duckdb",
        terminal_transport_manifest=_transport(),
    )
    assert ablation == {"result_sha256": "9" * 64, "basis_accepted": False}
    assert calls == [
        ("design", tmp_path),
        (
            "evaluate",
            {
                "train": expected.train,
                "tune_calibration": expected.tune_calibration,
                "tune_selection": expected.tune_selection,
                "publication_manifest_sha256": PUBLICATION_SHA,
                "terminal_transport_manifest_sha256": TRANSPORT_SHA,
            },
        ),
    ]

    observed.clear()
    gate_calls.clear()
    monkeypatch.setattr(
        operator_module,
        "assemble_round21_matched_development",
        lambda **_kwargs: expected,
    )
    matched = fit_round21_matched_optional_candidate(
        publication_directory=tmp_path / "publication",
        source_database=tmp_path / "core.duckdb",
        terminal_transport_manifest=_transport(),
        sidecar_database=tmp_path / "sidecar.duckdb",
        sidecar_terminal_manifest={"manifest_sha256": "f" * 64},
        basis_ablation_result=basis_ablation,
        compute_backend="cpu",
    )
    assert matched["artifact_sha256"] == "e" * 64
    assert observed == {
        "train": expected.train,
        "tune_calibration": expected.tune_calibration,
        "tune_selection": expected.tune_selection,
        "basis_ablation_result": basis_ablation,
        "compute_backend": "cpu",
    }
    assert gate_calls == [
        {
            "value": basis_ablation,
            "assembly": expected,
            "require_exact_dataset_identity": False,
        }
    ]
