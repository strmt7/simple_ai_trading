"""Deterministic development-only training for Round 74 event models.

This module deliberately stops at a pretest model policy. It cannot consume a
test batch, calculate trading returns, or grant any execution authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
import warnings

import numpy as np
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
import torch
from torch import nn

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .distributional_tcn_model import ExplicitAdamW
from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
    Round74EventTrainingBatch,
)
from .impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
    Round74EventModelOutput,
    build_round74_event_model,
    round74_event_model_loss,
)
from .storage import write_bytes_atomic


ROUND74_EVENT_TRAINING_SCHEMA_VERSION = "round-074-event-training-v6"
ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION = "round-074-event-pretest-policy-v5"
ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION = (
    "round-074-target-context-panel-v1"
)
ROUND74_EVENT_TRAINING_DEFAULT_SEEDS = (7411, 7423, 7433)
ROUND74_EVENT_TRAINING_LOSS_WEIGHTS = {
    "maximum_adverse_excursion": 0.35,
    "positive_payoff": 0.25,
    "adverse_selection": 0.20,
    "regime_unpredictability": 0.10,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_FILENAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 74 pretest policy has duplicate JSON keys")
        output[key] = value
    return output


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish once through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable Round 74 artifact already exists: {path}")
    write_bytes_atomic(path, payload)


@dataclass(frozen=True)
class Round74EventTrainingConfig:
    """Bounded training and chronological early-stopping policy."""

    candidate_ids: tuple[str, ...] = ROUND74_EVENT_MODEL_CANDIDATES
    seeds: tuple[int, ...] = ROUND74_EVENT_TRAINING_DEFAULT_SEEDS
    maximum_epochs: int = 48
    early_stopping_patience: int = 8
    minimum_tuning_improvement: float = 1e-5
    minibatch_rows: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    minimum_role_rows: int = 1_024

    def validate(self) -> None:
        if (
            self.candidate_ids != ROUND74_EVENT_MODEL_CANDIDATES
            or not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.seeds
            )
            or int(self.maximum_epochs) < 1
            or int(self.early_stopping_patience) < 1
            or int(self.early_stopping_patience) > int(self.maximum_epochs)
            or int(self.minibatch_rows) < 1
            or int(self.minimum_role_rows) < 1
        ):
            raise ValueError("Round 74 event training configuration differs")
        values = (
            float(self.minimum_tuning_improvement),
            float(self.learning_rate),
            float(self.weight_decay),
            float(self.gradient_clip_norm),
        )
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] < 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] <= 0.0
        ):
            raise ValueError("Round 74 event training numeric policy differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_ids": list(self.candidate_ids),
            "seeds": list(self.seeds),
            "maximum_epochs": int(self.maximum_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "minimum_tuning_improvement": float(self.minimum_tuning_improvement),
            "minibatch_rows": int(self.minibatch_rows),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "minimum_role_rows": int(self.minimum_role_rows),
            "training_order": "chronological_no_shuffle",
            "tuning_order": "chronological_no_shuffle",
            "checkpoint_policy": "best_state_in_memory_only",
            "loss_weights": dict(ROUND74_EVENT_TRAINING_LOSS_WEIGHTS),
        }


class Round74EventEnsemble(nn.Module):
    """Equal-weight seed ensemble for one fixed architecture."""

    def __init__(self, candidate_id: str, peer_count: int) -> None:
        super().__init__()
        if candidate_id not in ROUND74_EVENT_MODEL_CANDIDATES or peer_count < 1:
            raise ValueError("Round 74 event ensemble identity differs")
        self.candidate_id = candidate_id
        self.peers = nn.ModuleList(
            build_round74_event_model(candidate_id) for _ in range(int(peer_count))
        )

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        outputs = tuple(peer(values) for peer in self.peers)

        def tensor_mean(name: str) -> torch.Tensor:
            return torch.stack(
                tuple(getattr(output, name) for output in outputs),
                dim=0,
            ).mean(dim=0)

        def predictive_mixture_logit(name: str) -> torch.Tensor:
            peer_logits = torch.stack(
                tuple(getattr(output, name) for output in outputs),
                dim=0,
            )
            probability = torch.sigmoid(peer_logits).mean(dim=0)
            epsilon = torch.finfo(probability.dtype).eps
            bounded = torch.clamp(
                probability,
                min=epsilon,
                max=1.0 - epsilon,
            )
            return torch.log(bounded / (1.0 - bounded))

        output = Round74EventModelOutput(
            payoff_quantiles_bps=tensor_mean("payoff_quantiles_bps"),
            maximum_adverse_excursion_quantiles_bps=tensor_mean(
                "maximum_adverse_excursion_quantiles_bps"
            ),
            positive_payoff_logits=predictive_mixture_logit(
                "positive_payoff_logits"
            ),
            adverse_selection_logits=predictive_mixture_logit(
                "adverse_selection_logits"
            ),
            regime_unpredictability_logits=predictive_mixture_logit(
                "regime_unpredictability_logits"
            ),
        )
        output.validate(int(values.shape[0]))
        return output


@dataclass(frozen=True)
class Round74PretestPolicyArtifact:
    """Verified paths and identity for one immutable development-only policy."""

    policy_sha256: str
    policy_path: Path
    model_sha256: str
    model_path: Path
    selected_candidate_id: str
    tuning_loss: float


@dataclass(frozen=True)
class _DevelopmentIdentity:
    partition_sha256: str
    scaler_sha256: str
    target_context_sha256: tuple[str, ...]
    target_context_panel_sha256: str
    training_batch_sha256: tuple[str, ...]
    tuning_batch_sha256: tuple[str, ...]
    training_rows: int
    tuning_rows: int
    training_first_wall_ns: int
    training_last_wall_ns: int
    tuning_first_wall_ns: int
    tuning_last_wall_ns: int


@dataclass(frozen=True)
class _CandidateFit:
    candidate_id: str
    peer_states: tuple[dict[str, torch.Tensor], ...]
    peer_reports: tuple[dict[str, object], ...]
    ensemble_metrics: dict[str, float]
    ensemble_prediction_sha256: str
    parameter_count_per_peer: int


def _row_key(
    batch: Round74EventTrainingBatch,
    index: int,
) -> tuple[object, ...]:
    return (
        int(batch.decision_wall_ns[index]),
        batch.run_id[index],
        int(batch.decision_monotonic_ns[index]),
        int(batch.endpoint_frame_index[index]),
        int(batch.endpoint_message_index[index]),
        batch.symbol[index],
        int(batch.anchor_index[index]),
    )


def _validate_role_batches(
    batches: Sequence[Round74EventTrainingBatch],
    *,
    required_role: str,
    minimum_rows: int,
) -> tuple[object, ...]:
    if required_role not in {"training", "tuning"} or not batches:
        raise ValueError(f"Round 74 {required_role} batches are missing")
    prior_key: tuple[object, ...] | None = None
    rows = 0
    samples: set[str] = set()
    capture_runs: set[str] = set()
    for batch in batches:
        batch.validate()
        if batch.role != required_role:
            raise ValueError(
                f"Round 74 trainer rejects {batch.role!r} data in "
                f"the {required_role} role"
            )
        batch_runs = set(batch.run_id)
        if len(batch_runs) != 1:
            raise ValueError(
                f"Round 74 {required_role} batch mixes capture runs"
            )
        if len(set(batch.target_context_sha256)) != 1:
            raise ValueError(
                f"Round 74 {required_role} batch mixes target contexts"
            )
        run_id = next(iter(batch_runs))
        if run_id in capture_runs:
            raise ValueError(
                f"Round 74 {required_role} capture run is repeated"
            )
        capture_runs.add(run_id)
        first = _row_key(batch, 0)
        last = _row_key(batch, batch.rows - 1)
        if prior_key is not None and first <= prior_key:
            raise ValueError(f"Round 74 {required_role} batch order regressed")
        prior_key = last
        rows += batch.rows
        for sample in batch.sample_sha256:
            if sample in samples:
                raise ValueError(f"Round 74 {required_role} sample is duplicated")
            samples.add(sample)
    if rows < minimum_rows:
        raise ValueError(f"Round 74 {required_role} rows are below the minimum")
    return (
        rows,
        _row_key(batches[0], 0),
        _row_key(batches[-1], batches[-1].rows - 1),
        frozenset(samples),
    )


def _validate_development_batches(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    minimum_rows: int,
) -> _DevelopmentIdentity:
    training = tuple(training_batches)
    tuning = tuple(tuning_batches)
    training_rows, training_first, training_last, training_samples = (
        _validate_role_batches(
            training,
            required_role="training",
            minimum_rows=minimum_rows,
        )
    )
    tuning_rows, tuning_first, tuning_last, tuning_samples = _validate_role_batches(
        tuning,
        required_role="tuning",
        minimum_rows=minimum_rows,
    )
    if training_samples & tuning_samples:
        raise ValueError("Round 74 training and tuning samples overlap")
    if int(tuning_first[0]) - int(training_last[0]) < (
        ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    ):
        raise ValueError("Round 74 development roles lack the minimum purge")
    all_batches = (*training, *tuning)
    partitions = {batch.partition_sha256 for batch in all_batches}
    scalers = {batch.scaler_sha256 for batch in all_batches}
    contexts = tuple(sorted({
        context for batch in all_batches for context in batch.target_context_sha256
    }))
    if len(partitions) != 1:
        raise ValueError("Round 74 development partition identity differs")
    if len(scalers) != 1:
        raise ValueError("Round 74 development scaler identity differs")
    target_context_panel_sha256 = _canonical_sha256(
        {
            "schema_version": (
                ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
            ),
            "target_context_sha256": list(contexts),
        }
    )
    return _DevelopmentIdentity(
        partition_sha256=next(iter(partitions)),
        scaler_sha256=next(iter(scalers)),
        target_context_sha256=contexts,
        target_context_panel_sha256=target_context_panel_sha256,
        training_batch_sha256=tuple(batch.batch_sha256 for batch in training),
        tuning_batch_sha256=tuple(batch.batch_sha256 for batch in tuning),
        training_rows=int(training_rows),
        tuning_rows=int(tuning_rows),
        training_first_wall_ns=int(training_first[0]),
        training_last_wall_ns=int(training_last[0]),
        tuning_first_wall_ns=int(tuning_first[0]),
        tuning_last_wall_ns=int(tuning_last[0]),
    )


def _iter_minibatches(
    batches: Sequence[Round74EventTrainingBatch],
    maximum_rows: int,
) -> Iterable[tuple[Round74EventTrainingBatch, slice]]:
    for batch in batches:
        for start in range(0, batch.rows, maximum_rows):
            yield batch, slice(start, min(start + maximum_rows, batch.rows))


def _to_device_tensor(
    value: np.ndarray,
    row_slice: slice,
    device: object,
) -> torch.Tensor:
    copied = np.array(value[row_slice], dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(copied).to(device)


def _minibatch_target_counts(
    batch: Round74EventTrainingBatch,
    row_slice: slice,
) -> tuple[int, int, int]:
    start, stop, step = row_slice.indices(batch.rows)
    if step != 1 or stop <= start:
        raise ValueError("Round 74 minibatch slice differs")
    action_weight = int(batch.action_eligibility[row_slice].sum())
    regime_weight = int(
        batch.regime_unpredictability_eligibility[row_slice].sum()
    )
    if (action_weight == 0) != (regime_weight == 0):
        raise ValueError("Round 74 minibatch target eligibility differs")
    return action_weight, regime_weight, stop - start


def _skip_fully_censored_minibatch(
    totals: dict[str, float],
    batch: Round74EventTrainingBatch,
    row_slice: slice,
) -> bool:
    action_weight, regime_weight, rows = _minibatch_target_counts(
        batch,
        row_slice,
    )
    if action_weight > 0 and regime_weight > 0:
        return False
    totals["fully_censored_minibatches"] += 1.0
    totals["fully_censored_rows"] += float(rows)
    return True


def _loss_for_minibatch(
    model: nn.Module,
    batch: Round74EventTrainingBatch,
    row_slice: slice,
    device: object,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int]:
    features = _to_device_tensor(batch.feature_values, row_slice, device)
    payoff = _to_device_tensor(batch.net_payoff_bps, row_slice, device)
    adverse_excursion = _to_device_tensor(
        batch.maximum_adverse_excursion_bps,
        row_slice,
        device,
    )
    adverse = _to_device_tensor(batch.adverse_selection, row_slice, device)
    unpredictable = _to_device_tensor(
        batch.regime_unpredictability,
        row_slice,
        device,
    )
    action_eligibility = _to_device_tensor(
        batch.action_eligibility,
        row_slice,
        device,
    )
    regime_eligibility = _to_device_tensor(
        batch.regime_unpredictability_eligibility,
        row_slice,
        device,
    )
    output = model(features)
    loss, components = round74_event_model_loss(
        output,
        net_payoff_bps=payoff,
        maximum_adverse_excursion_bps=adverse_excursion,
        adverse_selection=adverse,
        regime_unpredictable=unpredictable,
        action_eligibility=action_eligibility,
        regime_unpredictability_eligibility=regime_eligibility,
        maximum_adverse_excursion_weight=(
            ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
        ),
        positive_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"],
        adverse_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"],
        unpredictability_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
            "regime_unpredictability"
        ],
    )
    return (
        loss,
        components,
        int(action_eligibility.detach().sum().cpu()),
        int(regime_eligibility.detach().sum().cpu()),
    )


def _empty_metric_sums() -> dict[str, float]:
    return {
        "payoff_pinball": 0.0,
        "maximum_adverse_excursion_pinball": 0.0,
        "positive_bce": 0.0,
        "adverse_bce": 0.0,
        "unpredictability_bce": 0.0,
        "action_weight": 0.0,
        "regime_weight": 0.0,
        "fully_censored_minibatches": 0.0,
        "fully_censored_rows": 0.0,
    }


def _accumulate_metrics(
    totals: dict[str, float],
    components: Mapping[str, torch.Tensor],
    *,
    action_weight: int,
    regime_weight: int,
) -> None:
    for name in (
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
    ):
        totals[name] += float(components[name].detach().cpu()) * action_weight
    totals["unpredictability_bce"] += (
        float(components["unpredictability_bce"].detach().cpu()) * regime_weight
    )
    totals["action_weight"] += action_weight
    totals["regime_weight"] += regime_weight


def _finalize_metrics(totals: Mapping[str, float]) -> dict[str, float]:
    action_weight = float(totals["action_weight"])
    regime_weight = float(totals["regime_weight"])
    if action_weight <= 0.0 or regime_weight <= 0.0:
        raise ValueError("Round 74 metric aggregation has no eligible targets")
    metrics = {
        name: float(totals[name]) / action_weight
        for name in (
            "payoff_pinball",
            "maximum_adverse_excursion_pinball",
            "positive_bce",
            "adverse_bce",
        )
    }
    metrics["unpredictability_bce"] = (
        float(totals["unpredictability_bce"]) / regime_weight
    )
    metrics["loss"] = (
        metrics["payoff_pinball"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
        * metrics["maximum_adverse_excursion_pinball"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"]
        * metrics["positive_bce"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"]
        * metrics["adverse_bce"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["regime_unpredictability"]
        * metrics["unpredictability_bce"]
    )
    metrics["eligible_action_targets"] = action_weight
    metrics["eligible_regime_targets"] = regime_weight
    metrics["fully_censored_minibatches"] = float(
        totals["fully_censored_minibatches"]
    )
    metrics["fully_censored_rows"] = float(totals["fully_censored_rows"])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Round 74 aggregate metrics are nonfinite")
    return metrics


def _evaluate_model(
    model: nn.Module,
    batches: Sequence[Round74EventTrainingBatch],
    *,
    minibatch_rows: int,
    device: object,
) -> dict[str, float]:
    model.eval()
    totals = _empty_metric_sums()
    per_run_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in batches:
            run_totals = _empty_metric_sums()
            for _selected, row_slice in _iter_minibatches(
                (batch,),
                minibatch_rows,
            ):
                if _skip_fully_censored_minibatch(
                    totals,
                    batch,
                    row_slice,
                ):
                    _skip_fully_censored_minibatch(
                        run_totals,
                        batch,
                        row_slice,
                    )
                    continue
                _loss, components, action_weight, regime_weight = (
                    _loss_for_minibatch(model, batch, row_slice, device)
                )
                _accumulate_metrics(
                    totals,
                    components,
                    action_weight=action_weight,
                    regime_weight=regime_weight,
                )
                _accumulate_metrics(
                    run_totals,
                    components,
                    action_weight=action_weight,
                    regime_weight=regime_weight,
                )
            per_run_metrics.append(_finalize_metrics(run_totals))
    metrics = _finalize_metrics(totals)
    run_losses = tuple(item["loss"] for item in per_run_metrics)
    if not run_losses:
        raise ValueError("Round 74 tuning has no capture runs")
    metrics["run_balanced_loss"] = sum(run_losses) / len(run_losses)
    metrics["worst_run_loss"] = max(run_losses)
    metrics["run_count"] = float(len(run_losses))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Round 74 run-balanced metrics are nonfinite")
    return metrics


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }


def _parameters_are_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.detach()).all())
        for parameter in model.parameters()
    )


def _train_peer(
    candidate_id: str,
    seed: int,
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventTrainingConfig,
    device: object,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_round74_event_model(candidate_id).to(device)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        optimization_totals = _empty_metric_sums()
        for batch, row_slice in _iter_minibatches(
            training_batches,
            config.minibatch_rows,
        ):
            if _skip_fully_censored_minibatch(
                optimization_totals,
                batch,
                row_slice,
            ):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, components, action_weight, regime_weight = _loss_for_minibatch(
                model, batch, row_slice, device
            )
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise RuntimeError("Round 74 model gradient is nonfinite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm,
                foreach=False,
            )
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise RuntimeError("Round 74 model gradient norm is nonfinite")
            optimizer.step()
            _accumulate_metrics(
                optimization_totals,
                components,
                action_weight=action_weight,
                regime_weight=regime_weight,
            )
        if not _parameters_are_finite(model):
            raise RuntimeError("Round 74 model parameters are nonfinite")
        optimization_metrics = _finalize_metrics(optimization_totals)
        tuning_metrics = _evaluate_model(
            model,
            tuning_batches,
            minibatch_rows=config.minibatch_rows,
            device=device,
        )
        tuning_loss = tuning_metrics["run_balanced_loss"]
        improved = (
            best_state is None
            or tuning_loss < best_loss - config.minimum_tuning_improvement
        )
        if improved:
            best_loss = tuning_loss
            best_epoch = epoch
            best_state = _cpu_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "optimization_metrics": optimization_metrics,
                "tuning_metrics": tuning_metrics,
                "improved": improved,
            }
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Round 74 model has no finite early-stop state")
    model.load_state_dict(best_state, strict=True)
    restored_metrics = _evaluate_model(
        model,
        tuning_batches,
        minibatch_rows=config.minibatch_rows,
        device=device,
    )
    if not math.isclose(
        restored_metrics["run_balanced_loss"],
        best_loss,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Round 74 best-state run-balanced reload metric differs"
        )
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_tuning_metrics": restored_metrics,
        "history": history,
    }


def _prediction_sha256(
    model: nn.Module,
    batch: Round74EventTrainingBatch,
    *,
    maximum_rows: int,
    device: object,
) -> str:
    rows = min(batch.rows, maximum_rows)
    values = _to_device_tensor(
        batch.feature_values,
        slice(0, rows),
        device,
    )
    model.eval()
    with torch.no_grad():
        output = model(values)
    digest = hashlib.sha256()
    digest.update(str(rows).encode("ascii"))
    for name in (
        "payoff_quantiles_bps",
        "maximum_adverse_excursion_quantiles_bps",
        "positive_payoff_logits",
        "adverse_selection_logits",
        "regime_unpredictability_logits",
    ):
        array = getattr(output, name).detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(name.encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _ensemble_from_states(
    candidate_id: str,
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: object,
) -> Round74EventEnsemble:
    ensemble = Round74EventEnsemble(candidate_id, len(states))
    for peer, state in zip(ensemble.peers, states, strict=True):
        peer.load_state_dict(dict(state), strict=True)
    return ensemble.to(device)


def _fit_candidate(
    candidate_id: str,
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventTrainingConfig,
    device: object,
) -> _CandidateFit:
    states: list[dict[str, torch.Tensor]] = []
    reports: list[dict[str, object]] = []
    for seed in config.seeds:
        state, report = _train_peer(
            candidate_id,
            seed,
            training_batches,
            tuning_batches,
            config=config,
            device=device,
        )
        states.append(state)
        reports.append(report)
    ensemble = _ensemble_from_states(candidate_id, states, device=device)
    ensemble_metrics = _evaluate_model(
        ensemble,
        tuning_batches,
        minibatch_rows=config.minibatch_rows,
        device=device,
    )
    prediction_sha256 = _prediction_sha256(
        ensemble,
        tuning_batches[0],
        maximum_rows=config.minibatch_rows,
        device=device,
    )
    parameter_count = sum(
        parameter.numel() for parameter in ensemble.peers[0].parameters()
    )
    return _CandidateFit(
        candidate_id=candidate_id,
        peer_states=tuple(states),
        peer_reports=tuple(reports),
        ensemble_metrics=ensemble_metrics,
        ensemble_prediction_sha256=prediction_sha256,
        parameter_count_per_peer=parameter_count,
    )


def _flatten_ensemble_state(
    fit: _CandidateFit,
) -> dict[str, torch.Tensor]:
    return {
        f"peers.{peer_index}.{name}": tensor.contiguous()
        for peer_index, state in enumerate(fit.peer_states)
        for name, tensor in sorted(state.items())
    }


def _load_ensemble_from_bytes(
    payload: bytes,
    *,
    candidate_id: str,
    peer_count: int,
) -> Round74EventEnsemble:
    state = load_safetensors(payload)
    ensemble = Round74EventEnsemble(candidate_id, peer_count)
    incompatible = ensemble.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Round 74 pretest model tensor names differ")
    return ensemble


def _runtime_source_binding() -> dict[str, str]:
    module_directory = Path(__file__).parent
    modules = {
        "sequence": "impact_absorption_event_sequence.py",
        "scaling": "impact_absorption_event_scaling.py",
        "targets": "impact_absorption_event_targets.py",
        "dataset": "impact_absorption_event_dataset.py",
        "model": "impact_absorption_event_model.py",
        "training": "impact_absorption_event_training.py",
        "storage": "storage.py",
    }
    return {
        f"{name}_module_sha256": _sha256_bytes(
            (module_directory / filename).read_bytes()
        )
        for name, filename in modules.items()
    }


def train_and_seal_round74_pretest_policy(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
) -> Round74PretestPolicyArtifact:
    """Train the declared panel and publish one reload-verified pretest policy."""

    selected_config = config or Round74EventTrainingConfig()
    selected_config.validate()
    development = _validate_development_batches(
        training_batches,
        tuning_batches,
        minimum_rows=selected_config.minimum_role_rows,
    )
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_python_random = random.getstate()
    prior_numpy_random = np.random.get_state()
    prior_torch_random = torch.get_rng_state()
    candidate_fits: list[_CandidateFit] = []
    warning_messages: list[str] = []
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for candidate_id in selected_config.candidate_ids:
                candidate_fits.append(
                    _fit_candidate(
                        candidate_id,
                        training_batches,
                        tuning_batches,
                        config=selected_config,
                        device=device,
                    )
                )
            warning_messages.extend(str(item.message) for item in caught)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        random.setstate(prior_python_random)
        np.random.set_state(prior_numpy_random)
        torch.set_rng_state(prior_torch_random)
    fallback_messages = [
        message
        for message in warning_messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    ]
    if fallback_messages:
        raise RuntimeError(
            f"Round 74 event training used CPU fallback: {fallback_messages}"
        )
    winner = min(
        candidate_fits,
        key=lambda fit: (
            fit.ensemble_metrics["run_balanced_loss"],
            fit.ensemble_metrics["worst_run_loss"],
            fit.ensemble_metrics["loss"],
            fit.parameter_count_per_peer,
            fit.candidate_id,
        ),
    )
    model_state = _flatten_ensemble_state(winner)
    # safetensors 0.8 metadata uses map ordering that is not byte-stable.
    # Identity belongs in the hash-bound policy; sorted tensors stay stable.
    model_bytes = save_safetensors(model_state)
    if model_bytes != save_safetensors(model_state):
        raise RuntimeError("Round 74 safetensors serialization is not stable")
    model_sha256 = _sha256_bytes(model_bytes)
    loaded = _load_ensemble_from_bytes(
        model_bytes,
        candidate_id=winner.candidate_id,
        peer_count=len(winner.peer_states),
    ).to(device)
    loaded_state = _cpu_state(loaded)
    if set(loaded_state) != set(model_state) or any(
        not torch.equal(loaded_state[name], model_state[name]) for name in model_state
    ):
        raise RuntimeError("Round 74 safetensors state reload differs")
    reload_prediction_sha256 = _prediction_sha256(
        loaded,
        tuning_batches[0],
        maximum_rows=selected_config.minibatch_rows,
        device=device,
    )
    if reload_prediction_sha256 != winner.ensemble_prediction_sha256:
        raise RuntimeError("Round 74 pretest prediction reload differs")
    model_filename = f"round74-{winner.candidate_id}-{model_sha256}.safetensors"
    policy: dict[str, object] = {
        "schema_version": ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
        "training_schema_version": ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
        "model_schema_version": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
        "source_binding": _runtime_source_binding(),
        "development_data": {
            "partition_sha256": development.partition_sha256,
            "scaler_sha256": development.scaler_sha256,
            "target_context_panel_schema_version": (
                ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
            ),
            "target_context_panel_sha256": (
                development.target_context_panel_sha256
            ),
            "target_context_sha256": list(development.target_context_sha256),
            "target_context_count": len(development.target_context_sha256),
            "training_batch_sha256": list(development.training_batch_sha256),
            "tuning_batch_sha256": list(development.tuning_batch_sha256),
            "training_rows": development.training_rows,
            "tuning_rows": development.tuning_rows,
            "training_first_wall_ns": development.training_first_wall_ns,
            "training_last_wall_ns": development.training_last_wall_ns,
            "tuning_first_wall_ns": development.tuning_first_wall_ns,
            "tuning_last_wall_ns": development.tuning_last_wall_ns,
            "role_transition_gap_ns": (
                development.tuning_first_wall_ns - development.training_last_wall_ns
            ),
            "test_batches_consumed": 0,
            "test_sample_digests_consumed": 0,
        },
        "training_policy": selected_config.as_dict(),
        "ensemble_aggregation": {
            "peer_weights": "equal",
            "payoff_and_adverse_excursion_quantiles": (
                "arithmetic_mean_of_peer_quantiles"
            ),
            "classification_heads": (
                "arithmetic_mean_of_peer_probabilities_then_logit"
            ),
            "mean_peer_logits_permitted": False,
            "probability_calibration_fitted_after_ensemble_aggregation": True,
        },
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": str(device),
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
            "torch_version": str(torch.__version__),
            "torch_directml_version": _package_version("torch-directml"),
            "safetensors_version": _package_version("safetensors"),
            "deterministic_algorithms_requested": True,
            "cross_platform_bitwise_reproducibility_claim": False,
            "cpu_fallback_warning_count": 0,
            "warning_count": len(warning_messages),
        },
        "candidate_panel": {
            fit.candidate_id: {
                "parameter_count_per_peer": fit.parameter_count_per_peer,
                "peer_count": len(fit.peer_states),
                "peer_reports": list(fit.peer_reports),
                "ensemble_tuning_metrics": fit.ensemble_metrics,
                "ensemble_prediction_sha256": (fit.ensemble_prediction_sha256),
            }
            for fit in candidate_fits
        },
        "selection": {
            "criterion": (
                "minimum run-balanced chronological tuning proper loss; "
                "then worst-run loss; then pooled loss; then parameter count; "
                "then candidate id"
            ),
            "selected_candidate_id": winner.candidate_id,
            "selected_tuning_metrics": winner.ensemble_metrics,
            "complexity_promotion_privilege": False,
            "backtest_metric_used_for_selection": False,
        },
        "model_artifact": {
            "filename": model_filename,
            "sha256": model_sha256,
            "byte_count": len(model_bytes),
            "media_type": "application/x-safetensors",
            "header_metadata_omitted_for_byte_stability": True,
            "pickle_permitted": False,
            "state_reload_verified": True,
            "prediction_reload_verified": True,
            "prediction_sha256": reload_prediction_sha256,
        },
        "authority": {
            "development_training_completed": True,
            "chronological_tuning_completed": True,
            "pretest_policy_sealed": True,
            "sealed_test_evaluated": False,
            "representative_market_evidence_claim": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    policy_sha256 = _canonical_sha256(policy)
    policy["policy_sha256"] = policy_sha256
    policy_bytes = _canonical_json_bytes(policy) + b"\n"
    output = Path(output_directory)
    model_path = output / model_filename
    policy_path = output / f"round74-pretest-policy-{policy_sha256}.json"
    _write_immutable_bytes(model_path, model_bytes)
    _write_immutable_bytes(policy_path, policy_bytes)
    verified_model, verified_policy = load_round74_pretest_policy(policy_path)
    if (
        verified_policy["policy_sha256"] != policy_sha256
        or verified_model.candidate_id != winner.candidate_id
    ):
        raise RuntimeError("Round 74 published pretest policy reload differs")
    return Round74PretestPolicyArtifact(
        policy_sha256=policy_sha256,
        policy_path=policy_path,
        model_sha256=model_sha256,
        model_path=model_path,
        selected_candidate_id=winner.candidate_id,
        tuning_loss=float(winner.ensemble_metrics["run_balanced_loss"]),
    )


def load_round74_pretest_policy(
    policy_path: str | Path,
) -> tuple[Round74EventEnsemble, dict[str, object]]:
    """Validate an immutable policy and load its safe tensor artifact on CPU."""

    selected_path = Path(policy_path)
    try:
        policy = json.loads(
            selected_path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 pretest policy could not be read") from exc
    if not isinstance(policy, dict):
        raise ValueError("Round 74 pretest policy root differs")
    claimed = str(policy.pop("policy_sha256", ""))
    if _SHA256.fullmatch(claimed) is None or claimed != _canonical_sha256(policy):
        raise ValueError("Round 74 pretest policy digest differs")
    policy["policy_sha256"] = claimed
    if selected_path.name != f"round74-pretest-policy-{claimed}.json":
        raise ValueError("Round 74 pretest policy filename differs")
    if (
        policy.get("schema_version") != ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
        or policy.get("training_schema_version")
        != ROUND74_EVENT_TRAINING_SCHEMA_VERSION
        or policy.get("model_schema_version") != ROUND74_EVENT_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("Round 74 pretest policy schema differs")
    if set(policy) != {
        "schema_version",
        "training_schema_version",
        "model_schema_version",
        "source_binding",
        "development_data",
        "training_policy",
        "ensemble_aggregation",
        "backend",
        "candidate_panel",
        "selection",
        "model_artifact",
        "authority",
        "policy_sha256",
    }:
        raise ValueError("Round 74 pretest policy top-level contract differs")
    selection = policy.get("selection")
    artifact = policy.get("model_artifact")
    development = policy.get("development_data")
    authority = policy.get("authority")
    training_policy = policy.get("training_policy")
    ensemble_aggregation = policy.get("ensemble_aggregation")
    source_binding = policy.get("source_binding")
    candidate_panel = policy.get("candidate_panel")
    backend = policy.get("backend")
    if not all(
        isinstance(value, Mapping)
        for value in (
            selection,
            artifact,
            development,
            authority,
            training_policy,
            ensemble_aggregation,
            source_binding,
            candidate_panel,
            backend,
        )
    ):
        raise ValueError("Round 74 pretest policy sections differ")
    assert isinstance(selection, Mapping)
    assert isinstance(artifact, Mapping)
    assert isinstance(development, Mapping)
    assert isinstance(authority, Mapping)
    assert isinstance(training_policy, Mapping)
    assert isinstance(ensemble_aggregation, Mapping)
    assert isinstance(source_binding, Mapping)
    assert isinstance(candidate_panel, Mapping)
    assert isinstance(backend, Mapping)
    candidate_id = str(selection.get("selected_candidate_id", ""))
    seeds = training_policy.get("seeds")
    candidate_ids = training_policy.get("candidate_ids")
    filename = str(artifact.get("filename", ""))
    model_sha256 = str(artifact.get("sha256", ""))
    if (
        candidate_id not in ROUND74_EVENT_MODEL_CANDIDATES
        or not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or not isinstance(candidate_ids, list)
        or not candidate_ids
        or any(not isinstance(value, str) for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or any(value not in ROUND74_EVENT_MODEL_CANDIDATES for value in candidate_ids)
        or set(candidate_panel) != set(candidate_ids)
        or candidate_id not in candidate_panel
        or _SAFE_FILENAME.fullmatch(filename) is None
        or _SHA256.fullmatch(model_sha256) is None
        or filename != f"round74-{candidate_id}-{model_sha256}.safetensors"
        or artifact.get("pickle_permitted") is not False
        or artifact.get("state_reload_verified") is not True
        or artifact.get("prediction_reload_verified") is not True
        or development.get("test_batches_consumed") != 0
        or development.get("test_sample_digests_consumed") != 0
    ):
        raise ValueError("Round 74 pretest policy static contract differs")
    if dict(source_binding) != _runtime_source_binding():
        raise ValueError("Round 74 pretest policy source binding differs")
    try:
        reconstructed_config = Round74EventTrainingConfig(
            candidate_ids=tuple(str(value) for value in candidate_ids),
            seeds=tuple(int(value) for value in seeds),
            maximum_epochs=int(training_policy["maximum_epochs"]),
            early_stopping_patience=int(training_policy["early_stopping_patience"]),
            minimum_tuning_improvement=float(
                training_policy["minimum_tuning_improvement"]
            ),
            minibatch_rows=int(training_policy["minibatch_rows"]),
            learning_rate=float(training_policy["learning_rate"]),
            weight_decay=float(training_policy["weight_decay"]),
            gradient_clip_norm=float(training_policy["gradient_clip_norm"]),
            minimum_role_rows=int(training_policy["minimum_role_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 74 pretest training policy differs") from exc
    if reconstructed_config.as_dict() != dict(training_policy):
        raise ValueError("Round 74 pretest training policy differs")
    if dict(ensemble_aggregation) != {
        "peer_weights": "equal",
        "payoff_and_adverse_excursion_quantiles": (
            "arithmetic_mean_of_peer_quantiles"
        ),
        "classification_heads": (
            "arithmetic_mean_of_peer_probabilities_then_logit"
        ),
        "mean_peer_logits_permitted": False,
        "probability_calibration_fitted_after_ensemble_aggregation": True,
    }:
        raise ValueError("Round 74 pretest ensemble aggregation differs")
    if (
        set(backend)
        != {
            "requested",
            "kind",
            "device",
            "vendor",
            "selection",
            "accelerated",
            "torch_version",
            "torch_directml_version",
            "safetensors_version",
            "deterministic_algorithms_requested",
            "cross_platform_bitwise_reproducibility_claim",
            "cpu_fallback_warning_count",
            "warning_count",
        }
        or backend.get("kind") not in {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
        or backend.get("requested")
        not in {"auto", "cpu", "cuda", "rocm", "xpu", "directml", "mps"}
        or (
            backend.get("requested") != "auto"
            and backend.get("requested") != backend.get("kind")
        )
        or backend.get("accelerated") is not (backend.get("kind") != "cpu")
        or any(
            not isinstance(backend.get(name), str) or not str(backend[name]).strip()
            for name in (
                "device",
                "vendor",
                "selection",
                "torch_version",
                "torch_directml_version",
                "safetensors_version",
            )
        )
        or backend.get("deterministic_algorithms_requested") is not True
        or backend.get("cross_platform_bitwise_reproducibility_claim") is not False
        or backend.get("cpu_fallback_warning_count") != 0
        or isinstance(backend.get("warning_count"), bool)
        or not isinstance(backend.get("warning_count"), int)
        or int(backend["warning_count"]) < 0
    ):
        raise ValueError("Round 74 pretest backend identity differs")
    batch_fields = ("training_batch_sha256", "tuning_batch_sha256")
    for field in batch_fields:
        values = development.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or any(_SHA256.fullmatch(str(value)) is None for value in values)
        ):
            raise ValueError("Round 74 pretest data identity differs")
    training_batch_hashes = set(development["training_batch_sha256"])
    tuning_batch_hashes = set(development["tuning_batch_sha256"])
    target_contexts = development.get("target_context_sha256")
    if (
        not isinstance(target_contexts, list)
        or not target_contexts
        or target_contexts != sorted(target_contexts)
        or len(target_contexts) != len(set(target_contexts))
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in target_contexts
        )
    ):
        raise ValueError("Round 74 pretest target-context panel differs")
    target_context_panel_sha256 = _canonical_sha256(
        {
            "schema_version": (
                ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
            ),
            "target_context_sha256": target_contexts,
        }
    )
    if (
        development.get("target_context_panel_schema_version")
        != ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
        or development.get("target_context_panel_sha256")
        != target_context_panel_sha256
        or development.get("target_context_count") != len(target_contexts)
    ):
        raise ValueError("Round 74 pretest target-context panel differs")
    development_times = tuple(
        development.get(name)
        for name in (
            "training_first_wall_ns",
            "training_last_wall_ns",
            "tuning_first_wall_ns",
            "tuning_last_wall_ns",
        )
    )
    if (
        set(development)
        != {
            "partition_sha256",
            "scaler_sha256",
            "target_context_panel_schema_version",
            "target_context_panel_sha256",
            "target_context_sha256",
            "target_context_count",
            "training_batch_sha256",
            "tuning_batch_sha256",
            "training_rows",
            "tuning_rows",
            "training_first_wall_ns",
            "training_last_wall_ns",
            "tuning_first_wall_ns",
            "tuning_last_wall_ns",
            "role_transition_gap_ns",
            "test_batches_consumed",
            "test_sample_digests_consumed",
        }
        or training_batch_hashes & tuning_batch_hashes
        or any(
            _SHA256.fullmatch(str(development.get(name, ""))) is None
            for name in (
                "partition_sha256",
                "scaler_sha256",
            )
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in development_times
        )
        or not (
            0
            < int(development_times[0])
            <= int(development_times[1])
            < int(development_times[2])
            <= int(development_times[3])
        )
        or int(development_times[2]) - int(development_times[1])
        < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
        or development.get("role_transition_gap_ns")
        != int(development_times[2]) - int(development_times[1])
        or any(
            isinstance(development.get(name), bool)
            or not isinstance(development.get(name), int)
            or int(development[name]) < reconstructed_config.minimum_role_rows
            for name in ("training_rows", "tuning_rows")
        )
    ):
        raise ValueError("Round 74 pretest data identity differs")
    expected_metric_names = {
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
        "unpredictability_bce",
        "loss",
        "run_balanced_loss",
        "worst_run_loss",
        "run_count",
        "eligible_action_targets",
        "eligible_regime_targets",
        "fully_censored_minibatches",
        "fully_censored_rows",
    }
    for panel_candidate, raw_report in candidate_panel.items():
        if not isinstance(raw_report, Mapping):
            raise ValueError("Round 74 pretest candidate report differs")
        metrics = raw_report.get("ensemble_tuning_metrics")
        peers = raw_report.get("peer_reports")
        parameter_count = raw_report.get("parameter_count_per_peer")
        if (
            set(raw_report)
            != {
                "parameter_count_per_peer",
                "peer_count",
                "peer_reports",
                "ensemble_tuning_metrics",
                "ensemble_prediction_sha256",
            }
            or isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count <= 0
            or raw_report.get("peer_count") != len(seeds)
            or not isinstance(peers, list)
            or len(peers) != len(seeds)
            or [peer.get("seed") for peer in peers if isinstance(peer, Mapping)]
            != seeds
            or not isinstance(metrics, Mapping)
            or set(metrics) != expected_metric_names
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in metrics.values()
            )
            or float(metrics.get("run_count", 0.0))
            != float(len(tuning_batch_hashes))
            or float(metrics.get("worst_run_loss", math.inf))
            < float(metrics.get("run_balanced_loss", -math.inf))
            or _SHA256.fullmatch(str(raw_report.get("ensemble_prediction_sha256", "")))
            is None
            or panel_candidate not in reconstructed_config.candidate_ids
        ):
            raise ValueError("Round 74 pretest candidate report differs")
    selected_report = candidate_panel[candidate_id]
    assert isinstance(selected_report, Mapping)
    expected_winner = min(
        candidate_panel,
        key=lambda value: (
            float(
                candidate_panel[value]["ensemble_tuning_metrics"][
                    "run_balanced_loss"
                ]
            ),
            float(
                candidate_panel[value]["ensemble_tuning_metrics"][
                    "worst_run_loss"
                ]
            ),
            float(candidate_panel[value]["ensemble_tuning_metrics"]["loss"]),
            int(candidate_panel[value]["parameter_count_per_peer"]),
            str(value),
        ),
    )
    if (
        set(selection)
        != {
            "criterion",
            "selected_candidate_id",
            "selected_tuning_metrics",
            "complexity_promotion_privilege",
            "backtest_metric_used_for_selection",
        }
        or selection.get("criterion")
        != (
            "minimum run-balanced chronological tuning proper loss; "
            "then worst-run loss; then pooled loss; then parameter count; "
            "then candidate id"
        )
        or candidate_id != expected_winner
        or selection.get("complexity_promotion_privilege") is not False
        or selection.get("backtest_metric_used_for_selection") is not False
        or selection.get("selected_tuning_metrics")
        != selected_report.get("ensemble_tuning_metrics")
        or artifact.get("prediction_sha256")
        != selected_report.get("ensemble_prediction_sha256")
        or _SHA256.fullmatch(str(artifact.get("prediction_sha256", ""))) is None
        or artifact.get("header_metadata_omitted_for_byte_stability") is not True
        or artifact.get("media_type") != "application/x-safetensors"
        or isinstance(artifact.get("byte_count"), bool)
        or not isinstance(artifact.get("byte_count"), int)
        or int(artifact["byte_count"]) <= 0
        or set(artifact)
        != {
            "filename",
            "sha256",
            "byte_count",
            "media_type",
            "header_metadata_omitted_for_byte_stability",
            "pickle_permitted",
            "state_reload_verified",
            "prediction_reload_verified",
            "prediction_sha256",
        }
    ):
        raise ValueError("Round 74 pretest selection or artifact differs")
    required_true_authority = {
        "development_training_completed",
        "chronological_tuning_completed",
        "pretest_policy_sealed",
    }
    required_false_authority = {
        "sealed_test_evaluated",
        "representative_market_evidence_claim",
        "financial_edge_tested",
        "profitability_claim",
        "ai_uplift_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    }
    if set(authority) != required_true_authority | required_false_authority or any(
        authority.get(name) is not True for name in required_true_authority
    ):
        raise ValueError("Round 74 pretest policy authority differs")
    for forbidden in (*sorted(required_false_authority),):
        if authority.get(forbidden) is not False:
            raise ValueError("Round 74 pretest policy overstates authority")
    model_path = selected_path.parent / filename
    try:
        model_bytes = model_path.read_bytes()
    except OSError as exc:
        raise ValueError("Round 74 pretest model could not be read") from exc
    if _sha256_bytes(model_bytes) != model_sha256 or len(model_bytes) != artifact.get(
        "byte_count"
    ):
        raise ValueError("Round 74 pretest model artifact differs")
    try:
        model = _load_ensemble_from_bytes(
            model_bytes,
            candidate_id=candidate_id,
            peer_count=len(seeds),
        )
    except Exception as exc:
        raise ValueError("Round 74 pretest model tensors differ") from exc
    model.eval()
    return model, policy


__all__ = [
    "ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION",
    "ROUND74_EVENT_TRAINING_DEFAULT_SEEDS",
    "ROUND74_EVENT_TRAINING_LOSS_WEIGHTS",
    "ROUND74_EVENT_TRAINING_SCHEMA_VERSION",
    "Round74EventEnsemble",
    "Round74EventTrainingConfig",
    "Round74PretestPolicyArtifact",
    "load_round74_pretest_policy",
    "train_and_seal_round74_pretest_policy",
]
