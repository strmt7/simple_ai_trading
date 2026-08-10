from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from polymarket_round25_support import (
    small_round25_sequence_collation,
    small_round25_sequence_condition_batch,
)
from simple_ai_trading.compute import resolve_backend, torch_device_for_backend
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    round25_logit,
)
from simple_ai_trading.polymarket_round25_tcn import (
    POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON,
    POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
    Round25CompiledTCN,
    Round25CompiledTCNEnsemble,
    _create_round25_tcn_ensemble_artifact,
    _create_round25_tcn_seed_artifact,
    _model,
    _state_bytes,
    create_round25_tcn_corpus_source,
    fit_round25_tcn_ensemble,
    round25_tcn_loss,
    round25_tcn_parameter_count,
    round25_tcn_train_step,
    validate_round25_tcn_fit_sources,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_tcn_fit_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-tcn-fit-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["truth_state"] == {
        "round25_v2_data_captured": False,
        "tcn_seed_model_fitted": False,
        "tcn_ensemble_fitted": False,
        "selection_evaluated": False,
        "sealed_test_evaluated": False,
        "edge_verified": False,
        "profitability_verified": False,
        "ai_uplift_verified": False,
        "paper_authority": False,
        "live_authority": False,
    }


def test_tcn_directml_host_probe_is_self_hashed_and_source_bound() -> None:
    root = Path(__file__).parents[1]
    path = (
        root
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-tcn-directml-host-probe-2026-08-10.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))
    claimed = evidence.pop("evidence_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["status"] == "runtime_mechanics_verified"
    assert evidence["backend"]["request_satisfied"] is True
    assert evidence["backend"]["accelerated"] is True
    assert evidence["mechanics"]["state_changed"] is True
    assert evidence["mechanics"]["state_roundtrip_byte_identity"] is True
    assert evidence["claims"] == {
        "market_data_used": False,
        "model_fitted": False,
        "predictive_edge_verified": False,
        "profitability_verified": False,
        "ai_uplift_verified": False,
        "paper_authority": False,
        "live_authority": False,
        "order_submitted": False,
    }
    for path_key, hash_key in (
        ("tcn_module_path", "tcn_module_sha256"),
        ("probe_tool_path", "probe_tool_sha256"),
    ):
        source_path = root / evidence["source"][path_key]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == evidence[
            "source"
        ][hash_key]


def test_tcn_forward_shapes_and_parameter_count_are_exact() -> None:
    torch = pytest.importorskip("torch")
    model = _model()
    values = torch.zeros((16, 64, 149), dtype=torch.float32)

    terminal, auxiliary = model(values)

    assert terminal.shape == (16,)
    assert auxiliary.shape == (16, 2)
    assert round25_tcn_parameter_count() == 14_371
    assert sum(parameter.numel() for parameter in model.parameters()) == 14_371


def test_masked_auxiliary_targets_cannot_change_loss() -> None:
    torch = pytest.importorskip("torch")
    terminal = torch.zeros(16, dtype=torch.float32)
    auxiliary = torch.zeros((16, 2), dtype=torch.float32)
    labels = torch.zeros(16, dtype=torch.float32)
    prior = torch.full((16,), 0.5, dtype=torch.float32)
    weights = torch.full((16,), 1.0 / 16.0, dtype=torch.float32)
    mask = torch.zeros((16, 2), dtype=torch.bool)
    zero_targets = torch.zeros((16, 2), dtype=torch.float32)
    changed_targets = torch.full((16, 2), 1_000_000.0, dtype=torch.float32)

    baseline = round25_tcn_loss(
        terminal,
        auxiliary,
        labels,
        prior,
        weights,
        zero_targets,
        mask,
    )
    masked_change = round25_tcn_loss(
        terminal,
        auxiliary,
        labels,
        prior,
        weights,
        changed_targets,
        mask,
    )
    available_change = round25_tcn_loss(
        terminal,
        auxiliary,
        labels,
        prior,
        weights,
        changed_targets,
        torch.ones_like(mask),
    )

    assert float(baseline[0]) == float(masked_change[0])
    assert tuple(float(value) for value in masked_change[2]) == (0.0, 0.0)
    assert float(available_change[0]) > float(masked_change[0])


def test_cpu_training_step_state_roundtrip_and_bound() -> None:
    torch = pytest.importorskip("torch")
    backend = resolve_backend("cpu", require=True)
    device = torch_device_for_backend(backend)
    torch.manual_seed(1729)
    model = _model().to(device)
    parameters = tuple(model.parameters())
    first_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    second_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    collation = small_round25_sequence_collation()
    before = _state_bytes(model)

    losses = round25_tcn_train_step(
        model,
        collation,
        device=device,
        first_moments=first_moments,
        second_moments=second_moments,
        step=1,
    )
    after = _state_bytes(model)

    assert all(np.isfinite(value) for value in (losses[0], losses[1], *losses[2]))
    assert before != after

    artifact = _create_round25_tcn_seed_artifact(
        model=model,
        training_seed=1729,
        train_dataset_sha256=collation.source_dataset_sha256,
        calibration_dataset_sha256="2" * 64,
        train_resolution_authority_sha256=(
            collation.resolution_authority_sha256
        ),
        calibration_resolution_authority_sha256="3" * 64,
        feature_transform_sha256=collation.feature_transform_sha256,
        train_batch_manifest_sha256="4" * 64,
        calibration_batch_manifest_sha256="5" * 64,
        best_epoch=1,
        epochs_run=1,
        calibration_condition_equal_log_loss=0.7,
        calibration_condition_equal_brier_score=0.25,
        backend_requested=backend.requested,
        backend_kind=backend.kind,
        backend_device=backend.device,
        backend_vendor=backend.vendor,
        backend_reason=backend.reason,
        backend_selection=backend.selection,
    )
    compiled = Round25CompiledTCN(artifact, compute_backend="cpu")
    probabilities = compiled.predict_probabilities(
        collation.sequence_values,
        collation.terminal_market_prior,
    )

    assert compiled.artifact_sha256 == artifact.artifact_sha256
    assert len(probabilities) == len(collation.terminal_labels)
    for probability, prior in zip(
        probabilities,
        collation.terminal_market_prior,
        strict=True,
    ):
        assert 0.0 < probability < 1.0
        assert abs(round25_logit(probability) - round25_logit(float(prior))) <= (
            POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND + 1e-6
        )
    assert artifact.architecture_json == POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON
    with pytest.raises(ValueError, match="seed artifact differs"):
        replace(artifact, architecture_json="{}").validated()
    with pytest.raises(ValueError, match="seed artifact differs"):
        replace(artifact, state_sha256="0" * 64).validated()


def _metadata_only_corpus(
    *,
    role: str,
    count: int,
    first_id: int,
    first_event_ms: int,
    dataset_sha256: str,
) -> object:
    condition_ids = tuple(
        "0x" + format(first_id + offset, "064x") for offset in range(count)
    )
    event_start_ms = tuple(first_event_ms + offset * 300_000 for offset in range(count))

    def unavailable_loader(_condition_ids: tuple[str, ...]) -> tuple[object, ...]:
        return ()

    return create_round25_tcn_corpus_source(
        role=role,
        condition_ids=condition_ids,
        event_start_ms=event_start_ms,
        batch_sha256=tuple(
            hashlib.sha256(condition_id.encode("ascii")).hexdigest()
            for condition_id in condition_ids
        ),
        source_dataset_sha256=dataset_sha256,
        resolution_authority_sha256="b" * 64,
        feature_transform_sha256="c" * 64,
        loader=unavailable_loader,
    )


def test_fit_sources_enforce_minimum_disjoint_chronological_corpora() -> None:
    small_train = _metadata_only_corpus(
        role="train",
        count=1,
        first_id=1,
        first_event_ms=1_000,
        dataset_sha256="d" * 64,
    )
    small_calibration = _metadata_only_corpus(
        role="calibration",
        count=1,
        first_id=3_000,
        first_event_ms=1_000_000,
        dataset_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="frozen corpus gates"):
        validate_round25_tcn_fit_sources(small_train, small_calibration)
    with pytest.raises(ValueError, match="corpus source differs"):
        replace(small_train, manifest_sha256="0" * 64).validated()

    train = _metadata_only_corpus(
        role="train",
        count=2_000,
        first_id=1,
        first_event_ms=1_000,
        dataset_sha256="d" * 64,
    )
    calibration = _metadata_only_corpus(
        role="calibration",
        count=400,
        first_id=3_000,
        first_event_ms=700_000_000,
        dataset_sha256="e" * 64,
    )
    assert validate_round25_tcn_fit_sources(train, calibration) == (
        train,
        calibration,
    )
    with pytest.raises(ValueError, match="loader population differs"):
        fit_round25_tcn_ensemble(train, calibration, compute_backend="cpu")


def test_corpus_source_loads_only_manifest_bound_batches() -> None:
    batch = small_round25_sequence_condition_batch()

    def loader(_condition_ids: tuple[str, ...]) -> tuple[object, ...]:
        return (batch,)

    source = create_round25_tcn_corpus_source(
        role=batch.role,
        condition_ids=(batch.condition_id,),
        event_start_ms=(batch.event_start_ms,),
        batch_sha256=(batch.batch_sha256,),
        source_dataset_sha256=batch.source_dataset_sha256,
        resolution_authority_sha256=batch.resolution_authority_sha256,
        feature_transform_sha256=batch.feature_transform_sha256,
        loader=loader,
    )
    collation = source.load_collation((batch.condition_id,))

    assert collation.condition_ids == (batch.condition_id,)
    assert collation.source_batch_sha256 == (batch.batch_sha256,)
    with pytest.raises(ValueError, match="corpus source differs"):
        replace(source, batch_sha256=("0" * 64,))


def test_three_seed_ensemble_is_hash_bound_and_arithmetic_mean() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(25)
    model = _model()
    collation = small_round25_sequence_collation()
    artifacts = tuple(
        _create_round25_tcn_seed_artifact(
            model=model,
            training_seed=seed,
            train_dataset_sha256=collation.source_dataset_sha256,
            calibration_dataset_sha256="2" * 64,
            train_resolution_authority_sha256=(
                collation.resolution_authority_sha256
            ),
            calibration_resolution_authority_sha256="3" * 64,
            feature_transform_sha256=collation.feature_transform_sha256,
            train_batch_manifest_sha256="4" * 64,
            calibration_batch_manifest_sha256="5" * 64,
            best_epoch=1,
            epochs_run=1,
            calibration_condition_equal_log_loss=0.7,
            calibration_condition_equal_brier_score=0.25,
            backend_requested="cpu",
            backend_kind="cpu",
            backend_device="cpu",
            backend_vendor="portable CPU reference",
            backend_reason="",
            backend_selection="deterministic_cpu_reference",
        )
        for seed in (1729, 3253, 7919)
    )
    ensemble = _create_round25_tcn_ensemble_artifact(artifacts)
    runtime = Round25CompiledTCNEnsemble(ensemble, compute_backend="cpu")
    seed_runtime = Round25CompiledTCN(artifacts[0], compute_backend="cpu")

    ensemble_probability = runtime.predict_probabilities(
        collation.sequence_values,
        collation.terminal_market_prior,
    )
    seed_probability = seed_runtime.predict_probabilities(
        collation.sequence_values,
        collation.terminal_market_prior,
    )

    assert ensemble_probability == pytest.approx(seed_probability, abs=1e-12)
    with pytest.raises(ValueError, match="ensemble artifact differs"):
        replace(ensemble, artifact_sha256="0" * 64).validated()
