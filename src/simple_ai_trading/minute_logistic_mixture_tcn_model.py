"""Minute-level coherent return-distribution TCN candidates for Round 48."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import (
    EXECUTION_CHARGE_BPS,
    DerivativesHurdleDataset,
)
from .distributional_tcn_model import ExplicitAdamW


ROUND = 48
HORIZONS_MINUTES = (15, 30, 60, 120)
CANDIDATE_COMPONENTS = {
    "single_logistic_tcn": 1,
    "state_mixture_logistic_tcn": 3,
}
SEEDS = (4801, 4802, 4803)
FEATURE_COUNT = 71
HIDDEN_CHANNELS = 96
DEPTHWISE_KERNEL_SIZE = 25
DEPTHWISE_DILATIONS = (1, 2, 4, 8)
RECEPTIVE_FIELD_STEPS = 1 + (DEPTHWISE_KERNEL_SIZE - 1) * sum(
    DEPTHWISE_DILATIONS
)
DECISION_INTERVAL_MINUTES = 5
WINDOW_STEPS = 576
SUPERVISED_STEPS = WINDOW_STEPS - RECEPTIVE_FIELD_STEPS + 1
WINDOW_STRIDE_STEPS = SUPERVISED_STEPS
BATCH_SIZE = 32
PREDICTION_CHUNK_STEPS = 2_048
MAXIMUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
MINIMUM_IMPROVEMENT = 1e-4
NLL_WEIGHT = 1.0
HURDLE_CROSS_ENTROPY_WEIGHT = 0.25
PAIRWISE_RANK_WEIGHT = 0.025
PAIRWISE_RANK_OFFSET_STEPS = 24
LOCATION_LIMIT = 8.0
MINIMUM_SCALE = 0.05
MAXIMUM_SCALE = 6.0
SCALE_CALIBRATION_GRID = np.geomspace(0.5, 2.0, 121, dtype=np.float64)
ANALYSIS_ROLES = ("calibration", "viability")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class RobustFeatureScaler:
    median: np.ndarray
    scaled_iqr: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (
            values.astype(np.float64) - self.median.reshape(1, 1, -1)
        ) / self.scaled_iqr.reshape(1, 1, -1)
        transformed = np.clip(transformed, -12.0, 12.0).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Round 48 standardized features are nonfinite")
        return transformed

    def asdict(self) -> dict[str, object]:
        return {
            "median": self.median.tolist(),
            "scaled_iqr": self.scaled_iqr.tolist(),
        }


@dataclass(frozen=True)
class RobustTargetScaler:
    median_bps: np.ndarray
    scaled_iqr_bps: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(HORIZONS_MINUTES):
            raise ValueError("Round 48 target dimensions are invalid")
        return (
            (values - self.median_bps.reshape(1, 1, -1))
            / self.scaled_iqr_bps.reshape(1, 1, -1)
        ).astype(np.float32)

    def denormalize_locations(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-2] != len(HORIZONS_MINUTES):
            raise ValueError("Round 48 location dimensions are invalid")
        return (
            values * self.scaled_iqr_bps.reshape(1, 1, -1, 1)
            + self.median_bps.reshape(1, 1, -1, 1)
        ).astype(np.float32)

    def denormalize_scales(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-2] != len(HORIZONS_MINUTES):
            raise ValueError("Round 48 scale dimensions are invalid")
        return (
            values * self.scaled_iqr_bps.reshape(1, 1, -1, 1)
        ).astype(np.float32)

    def normalized_thresholds(self, cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
        short = (-cost_bps - self.median_bps) / self.scaled_iqr_bps
        long = (cost_bps - self.median_bps) / self.scaled_iqr_bps
        return short.astype(np.float32), long.astype(np.float32)

    def asdict(self) -> dict[str, object]:
        return {
            "median_bps": self.median_bps.tolist(),
            "scaled_iqr_bps": self.scaled_iqr_bps.tolist(),
        }


@dataclass(frozen=True)
class MinuteTemporalDataset:
    feature_names: tuple[str, ...]
    timestamps_ms: np.ndarray
    features: np.ndarray
    signed_target_bps: np.ndarray
    role_masks: Mapping[str, np.ndarray]
    feature_stream_sha256: str
    target_stream_sha256: str
    dataset_sha256: str
    source_evidence: Mapping[str, object]

    @property
    def timestamps(self) -> int:
        return int(self.timestamps_ms.size)

    @property
    def rows(self) -> int:
        return self.timestamps * len(SYMBOLS)


@dataclass(frozen=True)
class MixtureArtifact:
    candidate_id: str
    components: int
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_composite: float
    best_early_stop_negative_log_likelihood: float
    best_early_stop_hurdle_cross_entropy: float
    best_early_stop_pairwise_rank_loss: float
    optimizer_updates: int
    parameter_count: int
    scale_multiplier: float
    calibration_negative_log_likelihood_before: float
    calibration_negative_log_likelihood_after: float
    backend_kind: str
    backend_device: str
    model_path: str
    model_bytes: int
    model_sha256: str
    prediction_path: str
    prediction_bytes: int
    prediction_sha256: str
    reload_max_abs_weight_error: float
    reload_max_abs_location_error: float
    reload_max_abs_scale_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MixtureForecastBundle:
    candidate_id: str
    components: int
    global_indices: np.ndarray
    seed_weights: np.ndarray
    seed_locations_normalized: np.ndarray
    seed_scales_normalized: np.ndarray
    artifacts: tuple[MixtureArtifact, ...]
    feature_scaler: RobustFeatureScaler
    target_scaler: RobustTargetScaler
    backend_kind: str
    backend_device: str
    training_history: tuple[Mapping[str, object], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stream_array_sha256(
    values: np.ndarray,
    *,
    label: str,
    chunk_rows: int = 32_768,
) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    for start in range(0, values.shape[0], chunk_rows):
        chunk = np.ascontiguousarray(values[start : start + chunk_rows])
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def build_minute_temporal_dataset(
    source: DerivativesHurdleDataset,
) -> MinuteTemporalDataset:
    """Reshape the verified symbol-major Round 38 matrix without copying it."""

    symbol_count = len(SYMBOLS)
    if source.rows % symbol_count:
        raise ValueError("Round 48 source rows do not divide by symbol count")
    timestamps = source.rows // symbol_count
    time_blocks = source.decision_time_ms.reshape(symbol_count, timestamps)
    symbol_blocks = source.symbol_index.reshape(symbol_count, timestamps)
    if not np.all(time_blocks == time_blocks[:1]):
        raise ValueError("Round 48 source timestamps differ across symbols")
    expected_symbols = np.arange(symbol_count, dtype=np.int8).reshape(-1, 1)
    if not np.array_equal(symbol_blocks, np.broadcast_to(expected_symbols, symbol_blocks.shape)):
        raise ValueError("Round 48 source symbol blocks are invalid")

    feature_view = source.feature_view("price_flow_only")
    if feature_view.shape[1] != FEATURE_COUNT:
        raise ValueError("Round 48 source feature count is invalid")
    features = feature_view.reshape(symbol_count, timestamps, FEATURE_COUNT).transpose(
        1, 0, 2
    )
    if not np.isfinite(features).all():
        raise ValueError("Round 48 source features are nonfinite")

    target_blocks: list[np.ndarray] = []
    for horizon in HORIZONS_MINUTES:
        long_net = source.long_net_utility_bps[horizon]
        signed = (long_net + EXECUTION_CHARGE_BPS).reshape(symbol_count, timestamps).T
        target_blocks.append(signed.astype(np.float32, copy=False))
    signed_target = np.stack(target_blocks, axis=-1)
    if not np.isfinite(signed_target).all():
        raise ValueError("Round 48 signed targets are nonfinite")

    role_masks: dict[str, np.ndarray] = {}
    for role in ("training", "early_stop", "calibration", "viability"):
        block = source.role_masks[max(HORIZONS_MINUTES)][role].reshape(
            symbol_count, timestamps
        )
        if not np.all(block == block[:1]):
            raise ValueError(f"Round 48 role mask differs across symbols: {role}")
        role_masks[role] = block[0].copy()
        if not np.any(role_masks[role]):
            raise ValueError(f"Round 48 role mask is empty: {role}")
    if any(
        np.any(role_masks[left] & role_masks[right])
        for left_index, left in enumerate(role_masks)
        for right in tuple(role_masks)[left_index + 1 :]
    ):
        raise ValueError("Round 48 chronological roles overlap")

    timestamps_ms = time_blocks[0].copy()
    feature_sha = _stream_array_sha256(features, label="round48-price-flow-features")
    target_sha = _stream_array_sha256(signed_target, label="round48-signed-targets")
    source_evidence = source.source_evidence.asdict()
    identity = _canonical_sha256(
        {
            "schema": "round-048-minute-temporal-dataset-v1",
            "symbols": list(SYMBOLS),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "timestamps": int(timestamps),
            "first_timestamp_ms": int(timestamps_ms[0]),
            "last_timestamp_ms": int(timestamps_ms[-1]),
            "feature_names": list(source.feature_names[:FEATURE_COUNT]),
            "feature_stream_sha256": feature_sha,
            "target_stream_sha256": target_sha,
            "source_certificate_sha256": source_evidence[
                "source_certificate_sha256"
            ],
        }
    )
    return MinuteTemporalDataset(
        feature_names=tuple(source.feature_names[:FEATURE_COUNT]),
        timestamps_ms=timestamps_ms,
        features=features,
        signed_target_bps=signed_target,
        role_masks=role_masks,
        feature_stream_sha256=feature_sha,
        target_stream_sha256=target_sha,
        dataset_sha256=identity,
        source_evidence=source_evidence,
    )


def fit_robust_feature_scaler(
    dataset: MinuteTemporalDataset,
) -> RobustFeatureScaler:
    selected = dataset.features[dataset.role_masks["training"]].reshape(
        -1, dataset.features.shape[-1]
    )
    quartiles = np.quantile(selected, (0.25, 0.5, 0.75), axis=0)
    median = quartiles[1].astype(np.float64)
    scaled_iqr = np.maximum((quartiles[2] - quartiles[0]) / 1.349, 1e-6)
    if not np.isfinite(median).all() or not np.isfinite(scaled_iqr).all():
        raise ValueError("Round 48 feature scaler is nonfinite")
    return RobustFeatureScaler(median=median, scaled_iqr=scaled_iqr)


def fit_robust_target_scaler(
    dataset: MinuteTemporalDataset,
) -> RobustTargetScaler:
    selected = dataset.signed_target_bps[dataset.role_masks["training"]].reshape(
        -1, len(HORIZONS_MINUTES)
    )
    quartiles = np.quantile(selected, (0.25, 0.5, 0.75), axis=0)
    median = quartiles[1].astype(np.float64)
    scaled_iqr = np.maximum((quartiles[2] - quartiles[0]) / 1.349, 1.0)
    if not np.isfinite(median).all() or not np.isfinite(scaled_iqr).all():
        raise ValueError("Round 48 target scaler is nonfinite")
    return RobustTargetScaler(median_bps=median, scaled_iqr_bps=scaled_iqr)


class PerTimestampLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class LargeKernelCausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (DEPTHWISE_KERNEL_SIZE - 1) * dilation
        self.normalization = PerTimestampLayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=DEPTHWISE_KERNEL_SIZE,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.contract = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        normalized = self.normalization(values)
        temporal = self.depthwise(F.pad(normalized, (self.left_padding, 0)))
        mixed = self.expand(temporal)
        mixed = self.dropout(F.gelu(mixed))
        mixed = self.dropout(self.contract(mixed))
        return residual + mixed


class LogisticMixtureTCN(nn.Module):
    """Causal encoder with an ordered conditional logistic-mixture head."""

    def __init__(
        self,
        *,
        input_channels: int = FEATURE_COUNT,
        components: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if components not in (1, 3):
            raise ValueError("Round 48 supports one or three components")
        self.input_channels = input_channels
        self.components = components
        self.projection = nn.Conv1d(input_channels, HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            LargeKernelCausalBlock(HIDDEN_CHANNELS, dilation, dropout)
            for dilation in DEPTHWISE_DILATIONS
        )
        self.final_normalization = PerTimestampLayerNorm(HIDDEN_CHANNELS)
        self.head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(HORIZONS_MINUTES) * 3 * components,
            kernel_size=1,
        )
        with torch.no_grad():
            if self.head.bias is not None:
                bias = self.head.bias.reshape(
                    len(HORIZONS_MINUTES), 3 * components
                )
                scale_start = 2 * components
                bias[:, scale_start:] = -1.65
                if components == 3:
                    bias[:, components + 1 : scale_start] = -0.5

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 48 model input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.final_normalization(encoded)
        raw = self.head(encoded).reshape(
            values.shape[0],
            len(HORIZONS_MINUTES),
            3 * self.components,
            values.shape[-1],
        )
        weight_logits = raw[:, :, : self.components]
        weights = torch.softmax(weight_logits, dim=2)

        center_index = self.components
        center = raw[:, :, center_index]
        if self.components == 1:
            raw_locations = center.unsqueeze(2)
        else:
            lower_gap = F.softplus(raw[:, :, center_index + 1])
            upper_gap = F.softplus(raw[:, :, center_index + 2])
            raw_locations = torch.stack(
                (center - lower_gap, center, center + upper_gap), dim=2
            )
        locations = LOCATION_LIMIT * torch.tanh(raw_locations / LOCATION_LIMIT)

        scale_start = 2 * self.components
        raw_scales = raw[:, :, scale_start : scale_start + self.components]
        scales = MINIMUM_SCALE + (MAXIMUM_SCALE - MINIMUM_SCALE) * torch.sigmoid(
            raw_scales
        )
        return weights, locations, scales


def logistic_mixture_log_density(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if (
        weights.shape != locations.shape
        or weights.shape != scales.shape
        or weights.ndim != 4
        or targets.shape != weights.shape[:2] + weights.shape[3:]
    ):
        raise ValueError("Round 48 mixture density shapes are invalid")
    z = (targets.unsqueeze(2) - locations) / scales
    absolute_z = torch.abs(z)
    component_log_density = (
        -absolute_z
        - 2.0 * torch.log1p(torch.exp(-absolute_z))
        - torch.log(scales)
    )
    return torch.logsumexp(
        torch.log(torch.clamp(weights, min=1e-7)) + component_log_density,
        dim=2,
    )


def logistic_mixture_cdf(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    if weights.shape != locations.shape or weights.shape != scales.shape:
        raise ValueError("Round 48 mixture CDF component shapes are invalid")
    if thresholds.shape != weights.shape[:2] + weights.shape[3:]:
        raise ValueError("Round 48 mixture CDF threshold shape is invalid")
    component_cdf = torch.sigmoid(
        (thresholds.unsqueeze(2) - locations) / scales
    )
    return torch.sum(weights * component_cdf, dim=2)


def mixture_mean(
    weights: torch.Tensor,
    locations: torch.Tensor,
) -> torch.Tensor:
    if weights.shape != locations.shape:
        raise ValueError("Round 48 mixture mean shapes are invalid")
    return torch.sum(weights * locations, dim=2)


def hurdle_probabilities(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    short_thresholds: torch.Tensor,
    long_thresholds: torch.Tensor,
) -> torch.Tensor:
    short = logistic_mixture_cdf(
        weights, locations, scales, short_thresholds
    )
    long = 1.0 - logistic_mixture_cdf(
        weights, locations, scales, long_thresholds
    )
    flat = torch.clamp(1.0 - short - long, min=1e-7, max=1.0)
    probabilities = torch.stack((short, flat, long), dim=2)
    return probabilities / torch.sum(probabilities, dim=2, keepdim=True)


def hurdle_cross_entropy(
    probabilities: torch.Tensor,
    raw_targets_bps: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim != 4 or raw_targets_bps.shape != (
        probabilities.shape[0],
        probabilities.shape[1],
        probabilities.shape[3],
    ):
        raise ValueError("Round 48 hurdle cross-entropy shapes are invalid")
    labels = torch.ones_like(raw_targets_bps, dtype=torch.long)
    labels = torch.where(
        raw_targets_bps < -EXECUTION_CHARGE_BPS,
        torch.zeros_like(labels),
        labels,
    )
    labels = torch.where(
        raw_targets_bps > EXECUTION_CHARGE_BPS,
        torch.full_like(labels, 2),
        labels,
    )
    selected = torch.gather(probabilities, 2, labels.unsqueeze(2)).squeeze(2)
    return -torch.log(torch.clamp(selected, min=1e-7)).mean()


def pairwise_expected_return_rank_loss(
    predicted_mean: torch.Tensor,
    target: torch.Tensor,
    *,
    offset: int = PAIRWISE_RANK_OFFSET_STEPS,
) -> torch.Tensor:
    if predicted_mean.shape != target.shape or predicted_mean.ndim != 3:
        raise ValueError("Round 48 pairwise rank shapes are invalid")
    if offset <= 0 or target.shape[-1] <= offset:
        raise ValueError("Round 48 pairwise rank offset is invalid")
    predicted_delta = predicted_mean[..., :-offset] - predicted_mean[..., offset:]
    target_delta = target[..., :-offset] - target[..., offset:]
    non_tie = (target_delta != 0.0).to(dtype=predicted_mean.dtype)
    direction = torch.where(
        target_delta > 0.0,
        torch.ones_like(target_delta),
        -torch.ones_like(target_delta),
    )
    numerator = (F.softplus(-direction * predicted_delta) * non_tie).sum()
    return numerator / torch.clamp(non_tie.sum(), min=1.0)


def mixture_objective(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    normalized_targets: torch.Tensor,
    raw_targets_bps: torch.Tensor,
    short_thresholds: torch.Tensor,
    long_thresholds: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights, locations, scales = output
    negative_log_likelihood = -logistic_mixture_log_density(
        weights, locations, scales, normalized_targets
    ).mean()
    probabilities = hurdle_probabilities(
        weights,
        locations,
        scales,
        short_thresholds,
        long_thresholds,
    )
    cross_entropy = hurdle_cross_entropy(probabilities, raw_targets_bps)
    rank_loss = pairwise_expected_return_rank_loss(
        mixture_mean(weights, locations), normalized_targets
    )
    objective = (
        NLL_WEIGHT * negative_log_likelihood
        + HURDLE_CROSS_ENTROPY_WEIGHT * cross_entropy
        + PAIRWISE_RANK_WEIGHT * rank_loss
    )
    return objective, {
        "negative_log_likelihood": negative_log_likelihood,
        "hurdle_cross_entropy": cross_entropy,
        "pairwise_rank_loss": rank_loss,
        "composite": objective,
    }


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    return [
        item
        for item in messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]


def _run_preflight(
    device: object,
    *,
    backend_kind: str,
    backend_device: str,
) -> dict[str, object]:
    generator = np.random.default_rng(SEEDS[0])
    reports: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for candidate_id, components in CANDIDATE_COMPONENTS.items():
        torch.manual_seed(SEEDS[0])
        model = LogisticMixtureTCN(components=components).to(device)
        optimizer = ExplicitAdamW(
            tuple(model.parameters()),
            learning_rate=8e-4,
            weight_decay=1e-4,
        )
        values = torch.from_numpy(
            generator.normal(size=(4, FEATURE_COUNT, 400)).astype(np.float32)
        ).to(device)
        normalized = torch.from_numpy(
            generator.normal(
                size=(4, len(HORIZONS_MINUTES), 400)
            ).astype(np.float32)
        ).to(device)
        raw = 25.0 * normalized
        short = torch.full_like(normalized, -0.5)
        long = torch.full_like(normalized, 0.5)
        before = {
            "depthwise": model.blocks[0].depthwise.weight.detach().cpu().clone(),
            "head": model.head.weight.detach().cpu().clone(),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer.zero_grad(set_to_none=True)
            output = model(values)
            objective, losses = mixture_objective(
                output,
                normalized,
                raw,
                short,
                long,
            )
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tail_targets = torch.from_numpy(
                np.broadcast_to(
                    np.where(np.arange(400) % 2 == 0, -500.0, 500.0),
                    (4, len(HORIZONS_MINUTES), 400),
                )
                .copy()
                .astype(np.float32)
            ).to(device)
            tail_output = model(values)
            tail_loss = -logistic_mixture_log_density(
                *tail_output, tail_targets
            ).mean()
            tail_loss.backward()
            tail_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
        warning_messages.extend(str(item.message) for item in caught)
        changes = {
            "depthwise": float(
                torch.max(
                    torch.abs(
                        model.blocks[0].depthwise.weight.detach().cpu()
                        - before["depthwise"]
                    )
                )
            ),
            "head": float(
                torch.max(
                    torch.abs(model.head.weight.detach().cpu() - before["head"])
                )
            ),
        }
        report = {
            "candidate_id": candidate_id,
            "components": components,
            "objective": float(objective.detach().cpu()),
            "negative_log_likelihood": float(
                losses["negative_log_likelihood"].detach().cpu()
            ),
            "hurdle_cross_entropy": float(
                losses["hurdle_cross_entropy"].detach().cpu()
            ),
            "pairwise_rank_loss": float(
                losses["pairwise_rank_loss"].detach().cpu()
            ),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "extreme_tail_loss": float(tail_loss.detach().cpu()),
            "extreme_tail_gradient_norm": float(
                tail_gradient_norm.detach().cpu()
            ),
            "parameter_changes": changes,
            "ordered_locations": bool(
                components == 1
                or torch.all(
                    output[1][:, :, 1:] >= output[1][:, :, :-1]
                ).cpu()
            ),
            "probability_sum_max_abs_error": float(
                torch.max(torch.abs(torch.sum(output[0], dim=2) - 1.0)).cpu()
            ),
        }
        scalar_values = [
            value
            for value in report.values()
            if isinstance(value, float)
        ] + list(changes.values())
        if (
            not all(math.isfinite(value) for value in scalar_values)
            or min(changes.values()) <= 0.0
            or not report["ordered_locations"]
            or report["probability_sum_max_abs_error"] > 1e-5
        ):
            raise RuntimeError(f"Round 48 preflight failed: {report}")
        reports.append(report)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 48 preflight used CPU fallback: {fallback}")
    return {
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "candidates": reports,
        "warning_count": len(warning_messages),
        "cpu_fallback_warnings": len(fallback),
    }


def directml_mixture_preflight() -> tuple[object, dict[str, object]]:
    try:
        import torch_directml  # type: ignore
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Round 48 DirectML is unavailable") from exc
    device = torch_directml.device()
    report = _run_preflight(
        device,
        backend_kind="directml",
        backend_device=str(device),
    )
    return device, report


def cpu_mixture_preflight() -> tuple[object, dict[str, object]]:
    device = torch.device("cpu")
    report = _run_preflight(
        device,
        backend_kind="cpu",
        backend_device="cpu",
    )
    return device, report


def _contiguous_runs(
    mask: np.ndarray,
    timestamps_ms: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    if mask.ndim != 1 or timestamps_ms.shape != mask.shape:
        raise ValueError("Round 48 contiguous-run inputs are invalid")
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return ()
    expected_delta = DECISION_INTERVAL_MINUTES * MINUTE_MS
    split = np.flatnonzero(
        (np.diff(selected) != 1)
        | (np.diff(timestamps_ms[selected]) != expected_delta)
    )
    starts = np.concatenate(([0], split + 1))
    ends = np.concatenate((split + 1, [selected.size]))
    return tuple(
        (int(selected[start]), int(selected[end - 1]) + 1)
        for start, end in zip(starts, ends, strict=True)
    )


def _source_boundaries(timestamps_ms: np.ndarray) -> np.ndarray:
    expected_delta = DECISION_INTERVAL_MINUTES * MINUTE_MS
    breaks = np.flatnonzero(np.diff(timestamps_ms) != expected_delta) + 1
    return np.concatenate(([0], breaks, [timestamps_ms.size])).astype(np.int64)


def _training_pairs(dataset: MinuteTemporalDataset) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for start, end in _contiguous_runs(
        dataset.role_masks["training"], dataset.timestamps_ms
    ):
        last_start = end - WINDOW_STEPS
        if last_start < start:
            continue
        for window_start in range(start, last_start + 1, WINDOW_STRIDE_STEPS):
            rows.extend(
                (window_start, symbol_index)
                for symbol_index in range(len(SYMBOLS))
            )
    if not rows:
        raise ValueError("Round 48 training role has no complete windows")
    pairs = np.asarray(rows, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError("Round 48 training-pair matrix is invalid")
    return pairs


def _batch_arrays(
    normalized_features: np.ndarray,
    normalized_targets: np.ndarray,
    raw_targets_bps: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    supervised_start = RECEPTIVE_FIELD_STEPS - 1
    for start, symbol_index in pairs:
        stop = int(start) + WINDOW_STEPS
        feature_rows.append(
            normalized_features[int(start) : stop, int(symbol_index)].T
        )
        normalized_rows.append(
            normalized_targets[
                int(start) + supervised_start : stop, int(symbol_index)
            ].T
        )
        raw_rows.append(
            raw_targets_bps[
                int(start) + supervised_start : stop, int(symbol_index)
            ].T
        )
    outputs = (
        np.stack(feature_rows).astype(np.float32, copy=False),
        np.stack(normalized_rows).astype(np.float32, copy=False),
        np.stack(raw_rows).astype(np.float32, copy=False),
    )
    if outputs[0].shape != (pairs.shape[0], FEATURE_COUNT, WINDOW_STEPS):
        raise RuntimeError("Round 48 feature batch shape is invalid")
    expected_target_shape = (
        pairs.shape[0],
        len(HORIZONS_MINUTES),
        SUPERVISED_STEPS,
    )
    if outputs[1].shape != expected_target_shape or outputs[2].shape != expected_target_shape:
        raise RuntimeError("Round 48 target batch shape is invalid")
    return tuple(np.ascontiguousarray(value) for value in outputs)  # type: ignore[return-value]


def _predict_mask(
    model: LogisticMixtureTCN,
    normalized_features: np.ndarray,
    dataset: MinuteTemporalDataset,
    mask: np.ndarray,
    device: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Round 48 prediction mask is empty")
    components = model.components
    shape = (
        indices.size,
        len(SYMBOLS),
        len(HORIZONS_MINUTES),
        components,
    )
    weights = np.full(shape, np.nan, dtype=np.float32)
    locations = np.full(shape, np.nan, dtype=np.float32)
    scales = np.full(shape, np.nan, dtype=np.float32)
    index_positions = np.full(dataset.timestamps, -1, dtype=np.int64)
    index_positions[indices] = np.arange(indices.size, dtype=np.int64)
    boundaries = _source_boundaries(dataset.timestamps_ms)

    model.eval()
    with torch.no_grad():
        for run_start, run_end in _contiguous_runs(mask, dataset.timestamps_ms):
            boundary_position = int(
                np.searchsorted(boundaries, run_start, side="right") - 1
            )
            segment_start = int(boundaries[boundary_position])
            for output_start in range(
                run_start, run_end, PREDICTION_CHUNK_STEPS
            ):
                output_end = min(run_end, output_start + PREDICTION_CHUNK_STEPS)
                input_start = max(
                    segment_start, output_start - RECEPTIVE_FIELD_STEPS + 1
                )
                batch = np.ascontiguousarray(
                    normalized_features[input_start:output_end].transpose(1, 2, 0)
                )
                tensor = torch.from_numpy(batch).to(device)
                output = model(tensor)
                local_start = output_start - input_start
                local_stop = local_start + output_end - output_start
                target_positions = index_positions[output_start:output_end]
                if np.any(target_positions < 0):
                    raise RuntimeError("Round 48 prediction indexing is invalid")
                arrays = [
                    value[..., local_start:local_stop]
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(3, 0, 1, 2)
                    .astype(np.float32, copy=False)
                    for value in output
                ]
                weights[target_positions] = arrays[0]
                locations[target_positions] = arrays[1]
                scales[target_positions] = arrays[2]
    weights_finite = np.isfinite(weights)
    locations_finite = np.isfinite(locations)
    scales_finite = np.isfinite(scales)
    weight_sum_error = (
        float(np.max(np.abs(np.sum(weights, axis=-1) - 1.0)))
        if weights_finite.all()
        else math.inf
    )
    minimum_scale = (
        float(np.min(scales[scales_finite])) if scales_finite.any() else math.nan
    )
    if (
        not weights_finite.all()
        or not locations_finite.all()
        or not scales_finite.all()
        or weight_sum_error > 1e-5
        or minimum_scale <= 0.0
    ):
        diagnostics = {
            "components": components,
            "prediction_rows": int(indices.size),
            "weights_nonfinite": int(weights.size - weights_finite.sum()),
            "locations_nonfinite": int(
                locations.size - locations_finite.sum()
            ),
            "scales_nonfinite": int(scales.size - scales_finite.sum()),
            "weight_sum_max_abs_error": weight_sum_error,
            "minimum_finite_scale": minimum_scale,
        }
        raise RuntimeError(
            "Round 48 prediction output failed integrity checks: "
            f"{diagnostics}"
        )
    if components == 3 and np.any(np.diff(locations, axis=-1) < -1e-6):
        raise RuntimeError("Round 48 prediction component locations crossed")
    return indices, weights, locations, scales


def _numpy_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def numpy_logistic_mixture_log_density(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if (
        weights.shape != locations.shape
        or weights.shape != scales.shape
        or weights.ndim != 4
        or targets.shape != weights.shape[:-1]
    ):
        raise ValueError("Round 48 NumPy density shapes are invalid")
    z = (targets[..., None] - locations) / scales
    absolute_z = np.abs(z)
    component = (
        -absolute_z
        - 2.0 * np.log1p(np.exp(-absolute_z))
        - np.log(scales)
    )
    return _numpy_logsumexp(
        np.log(np.clip(weights, 1e-12, 1.0)) + component,
        axis=-1,
    )


def numpy_logistic_mixture_cdf(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    if weights.shape != locations.shape or weights.shape != scales.shape:
        raise ValueError("Round 48 NumPy CDF shapes are invalid")
    z = np.clip((thresholds[..., None] - locations) / scales, -60.0, 60.0)
    return np.sum(weights / (1.0 + np.exp(-z)), axis=-1)


def numpy_hurdle_probabilities(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    short_thresholds: np.ndarray,
    long_thresholds: np.ndarray,
) -> np.ndarray:
    threshold_shape = (1, 1, len(HORIZONS_MINUTES))
    short = numpy_logistic_mixture_cdf(
        weights,
        locations,
        scales,
        short_thresholds.reshape(threshold_shape),
    )
    long = 1.0 - numpy_logistic_mixture_cdf(
        weights,
        locations,
        scales,
        long_thresholds.reshape(threshold_shape),
    )
    flat = np.clip(1.0 - short - long, 1e-12, 1.0)
    output = np.stack((short, flat, long), axis=-1)
    return output / np.sum(output, axis=-1, keepdims=True)


def _numpy_pairwise_rank_loss(
    predicted_mean: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    timestamps_ms: np.ndarray,
) -> float:
    offset = PAIRWISE_RANK_OFFSET_STEPS
    if predicted_mean.shape != targets.shape or predicted_mean.ndim != 3:
        raise ValueError("Round 48 NumPy pairwise shapes are invalid")
    if predicted_mean.shape[0] <= offset:
        return 0.0
    valid_time = (
        (indices[offset:] - indices[:-offset] == offset)
        & (
            timestamps_ms[indices[offset:]] - timestamps_ms[indices[:-offset]]
            == offset * DECISION_INTERVAL_MINUTES * MINUTE_MS
        )
    )
    predicted_delta = predicted_mean[:-offset] - predicted_mean[offset:]
    target_delta = targets[:-offset] - targets[offset:]
    valid = np.broadcast_to(
        valid_time.reshape(-1, 1, 1), target_delta.shape
    ).copy()
    valid &= target_delta != 0.0
    if not np.any(valid):
        return 0.0
    direction = np.where(target_delta > 0.0, 1.0, -1.0)
    return float(np.mean(np.logaddexp(0.0, -direction[valid] * predicted_delta[valid])))


def _numpy_component_losses(
    dataset: MinuteTemporalDataset,
    normalized_targets: np.ndarray,
    target_scaler: RobustTargetScaler,
    indices: np.ndarray,
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> dict[str, float]:
    targets = normalized_targets[indices]
    raw_targets = dataset.signed_target_bps[indices]
    negative_log_likelihood = float(
        -np.mean(
            numpy_logistic_mixture_log_density(
                weights, locations, scales, targets
            )
        )
    )
    short_thresholds, long_thresholds = target_scaler.normalized_thresholds(
        EXECUTION_CHARGE_BPS
    )
    probabilities = numpy_hurdle_probabilities(
        weights,
        locations,
        scales,
        short_thresholds,
        long_thresholds,
    )
    labels = np.ones(raw_targets.shape, dtype=np.int8)
    labels[raw_targets < -EXECUTION_CHARGE_BPS] = 0
    labels[raw_targets > EXECUTION_CHARGE_BPS] = 2
    selected = np.take_along_axis(probabilities, labels[..., None], axis=-1)[..., 0]
    cross_entropy = float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))
    predicted_mean = np.sum(weights * locations, axis=-1)
    rank_loss = _numpy_pairwise_rank_loss(
        predicted_mean,
        targets,
        indices,
        dataset.timestamps_ms,
    )
    composite = (
        negative_log_likelihood
        + HURDLE_CROSS_ENTROPY_WEIGHT * cross_entropy
        + PAIRWISE_RANK_WEIGHT * rank_loss
    )
    return {
        "negative_log_likelihood": negative_log_likelihood,
        "hurdle_cross_entropy": cross_entropy,
        "pairwise_rank_loss": rank_loss,
        "composite": composite,
    }


def _combine_seed_components(
    seed_outputs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(seed_outputs) != len(SEEDS):
        raise ValueError("Round 48 ensemble requires three seeds")
    weights = np.concatenate(
        [output[0] / len(seed_outputs) for output in seed_outputs], axis=-1
    )
    locations = np.concatenate([output[1] for output in seed_outputs], axis=-1)
    scales = np.concatenate([output[2] for output in seed_outputs], axis=-1)
    if np.max(np.abs(np.sum(weights, axis=-1) - 1.0)) > 1e-6:
        raise RuntimeError("Round 48 ensemble weights are incoherent")
    return weights, locations, scales


def _clone_state(model: LogisticMixtureTCN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _fit_scale_multiplier(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float, float]:
    before = float(
        -np.mean(
            numpy_logistic_mixture_log_density(
                weights, locations, scales, targets
            )
        )
    )
    losses = np.asarray(
        [
            -np.mean(
                numpy_logistic_mixture_log_density(
                    weights,
                    locations,
                    scales * multiplier,
                    targets,
                )
            )
            for multiplier in SCALE_CALIBRATION_GRID
        ],
        dtype=np.float64,
    )
    minimum = float(np.min(losses))
    tied = np.flatnonzero(np.isclose(losses, minimum, rtol=0.0, atol=1e-14))
    selected = min(
        tied.tolist(),
        key=lambda index: (
            abs(float(SCALE_CALIBRATION_GRID[index]) - 1.0),
            float(SCALE_CALIBRATION_GRID[index]),
        ),
    )
    multiplier = float(SCALE_CALIBRATION_GRID[selected])
    after = float(losses[selected])
    if after > before + 1e-12:
        raise RuntimeError("Round 48 scale calibration worsened likelihood")
    return multiplier, before, after


def _save_prediction_artifact(
    path: Path,
    *,
    indices: np.ndarray,
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    scale_multiplier: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_indices=indices,
        weights=weights,
        locations_normalized=locations,
        scales_normalized=scales,
        scale_multiplier=np.asarray(scale_multiplier, dtype=np.float64),
    )


def _train_candidate(
    dataset: MinuteTemporalDataset,
    normalized_features: np.ndarray,
    normalized_targets: np.ndarray,
    feature_scaler: RobustFeatureScaler,
    target_scaler: RobustTargetScaler,
    *,
    candidate_id: str,
    components: int,
    model_dir: Path,
    prediction_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> MixtureForecastBundle:
    peers: list[LogisticMixtureTCN] = []
    optimizers: list[ExplicitAdamW] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        peer = LogisticMixtureTCN(components=components).to(device)
        peers.append(peer)
        optimizers.append(
            ExplicitAdamW(
                tuple(peer.parameters()),
                learning_rate=8e-4,
                weight_decay=1e-4,
            )
        )
    pairs = _training_pairs(dataset)
    generator = np.random.default_rng(SEEDS[0] + components)
    short_thresholds, long_thresholds = target_scaler.normalized_thresholds(
        EXECUTION_CHARGE_BPS
    )
    best_loss = math.inf
    best_epoch = 0
    best_states: list[dict[str, torch.Tensor]] | None = None
    best_losses: dict[str, float] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_run = 0
    history: list[Mapping[str, object]] = []
    warning_messages: list[str] = []

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(pairs.shape[0])
        batch_losses: dict[str, list[float]] = {
            "negative_log_likelihood": [],
            "hurdle_cross_entropy": [],
            "pairwise_rank_loss": [],
            "composite": [],
        }
        for peer in peers:
            peer.train()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                feature_batch, normalized_batch, raw_batch = _batch_arrays(
                    normalized_features,
                    normalized_targets,
                    dataset.signed_target_bps,
                    selected,
                )
                values = torch.from_numpy(feature_batch).to(device)
                normalized_tensor = torch.from_numpy(normalized_batch).to(device)
                raw_tensor = torch.from_numpy(raw_batch).to(device)
                short_tensor = torch.from_numpy(
                    np.broadcast_to(
                        short_thresholds.reshape(1, -1, 1),
                        normalized_batch.shape,
                    ).copy()
                ).to(device)
                long_tensor = torch.from_numpy(
                    np.broadcast_to(
                        long_thresholds.reshape(1, -1, 1),
                        normalized_batch.shape,
                    ).copy()
                ).to(device)
                peer_components: dict[str, list[float]] = {
                    key: [] for key in batch_losses
                }
                for seed, peer, optimizer in zip(
                    SEEDS, peers, optimizers, strict=True
                ):
                    optimizer.zero_grad(set_to_none=True)
                    output = tuple(
                        value[..., -SUPERVISED_STEPS:] for value in peer(values)
                    )
                    objective, losses = mixture_objective(
                        output,
                        normalized_tensor,
                        raw_tensor,
                        short_tensor,
                        long_tensor,
                    )
                    loss_values = {
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    }
                    if not all(
                        math.isfinite(value) for value in loss_values.values()
                    ):
                        raise RuntimeError(
                            "Round 48 produced a nonfinite training loss: "
                            f"candidate={candidate_id}, seed={seed}, "
                            f"epoch={epoch}, batch={offset // BATCH_SIZE}, "
                            f"losses={loss_values}"
                        )
                    objective.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        peer.parameters(), 1.0, foreach=False
                    )
                    gradient_norm_value = float(gradient_norm.detach().cpu())
                    if not math.isfinite(gradient_norm_value):
                        raise RuntimeError(
                            "Round 48 rejected a nonfinite gradient update: "
                            f"candidate={candidate_id}, seed={seed}, "
                            f"epoch={epoch}, batch={offset // BATCH_SIZE}, "
                            f"losses={loss_values}"
                        )
                    optimizer.step()
                    for key in peer_components:
                        peer_components[key].append(loss_values[key])
                optimizer_updates += 1
                for key, values_for_key in peer_components.items():
                    batch_losses[key].append(float(np.mean(values_for_key)))
        warning_messages.extend(str(item.message) for item in caught)

        validation_outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        validation_indices: np.ndarray | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for peer in peers:
                predicted = _predict_mask(
                    peer,
                    normalized_features,
                    dataset,
                    dataset.role_masks["early_stop"],
                    device,
                )
                if validation_indices is None:
                    validation_indices = predicted[0]
                elif not np.array_equal(validation_indices, predicted[0]):
                    raise RuntimeError("Round 48 validation indices differ by seed")
                validation_outputs.append(predicted[1:])
        warning_messages.extend(str(item.message) for item in caught)
        if validation_indices is None:
            raise RuntimeError("Round 48 validation produced no indices")
        ensemble = _combine_seed_components(validation_outputs)
        validation = _numpy_component_losses(
            dataset,
            normalized_targets,
            target_scaler,
            validation_indices,
            *ensemble,
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "components": components,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "training_windows": int(pairs.shape[0]),
            **{
                f"training_{key}": float(np.mean(values_for_key))
                for key, values_for_key in batch_losses.items()
            },
            **{f"early_stop_{key}": value for key, value in validation.items()},
        }
        scalar_values = [
            value for value in row.values() if isinstance(value, float)
        ]
        if not all(math.isfinite(value) for value in scalar_values):
            raise RuntimeError("Round 48 training diagnostics are nonfinite")
        improved = validation["composite"] < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = validation["composite"]
            best_epoch = epoch
            best_states = [_clone_state(peer) for peer in peers]
            best_losses = dict(validation)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_early_stop_composite"] = best_loss
        row["best_epoch"] = best_epoch
        row["stale_epochs"] = stale_epochs
        history.append(row)
        if progress is not None:
            progress("round48_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_states is None or best_losses is None:
        raise RuntimeError("Round 48 training did not produce a best state")
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 48 training used CPU fallback: {fallback}")

    analysis_mask = np.logical_or.reduce(
        [dataset.role_masks[role] for role in ANALYSIS_ROLES]
    )
    probe_mask = np.zeros(dataset.timestamps, dtype=bool)
    probe_indices = np.flatnonzero(dataset.role_masks["calibration"])[:512]
    probe_mask[probe_indices] = True
    artifacts: list[MixtureArtifact] = []
    seed_weights: list[np.ndarray] = []
    seed_locations: list[np.ndarray] = []
    seed_scales: list[np.ndarray] = []
    global_indices: np.ndarray | None = None

    for peer, seed, best_state in zip(peers, SEEDS, best_states, strict=True):
        peer.load_state_dict(best_state)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference_probe = _predict_mask(
                peer,
                normalized_features,
                dataset,
                probe_mask,
                device,
            )
        warning_messages.extend(str(item.message) for item in caught)
        model_path = model_dir / f"round48_{candidate_id}_seed_{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        reloaded = LogisticMixtureTCN(components=components)
        reloaded.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        reloaded = reloaded.to(device)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predicted = _predict_mask(
                reloaded,
                normalized_features,
                dataset,
                analysis_mask,
                device,
            )
        warning_messages.extend(str(item.message) for item in caught)
        indices, weights, locations, scales = predicted
        if global_indices is None:
            global_indices = indices
        elif not np.array_equal(global_indices, indices):
            raise RuntimeError("Round 48 final indices differ by seed")
        probe_positions = np.searchsorted(indices, reference_probe[0])
        if not np.array_equal(indices[probe_positions], reference_probe[0]):
            raise RuntimeError("Round 48 reload probe indexing failed")
        reload_errors = tuple(
            float(np.max(np.abs(reference - observed[probe_positions])))
            for reference, observed in zip(
                reference_probe[1:],
                (weights, locations, scales),
                strict=True,
            )
        )
        if not all(
            math.isfinite(value) and value <= 1e-6 for value in reload_errors
        ):
            raise RuntimeError(f"Round 48 reload errors are {reload_errors}")
        calibration_local = dataset.role_masks["calibration"][indices]
        multiplier, nll_before, nll_after = _fit_scale_multiplier(
            weights[calibration_local],
            locations[calibration_local],
            scales[calibration_local],
            normalized_targets[indices[calibration_local]],
        )
        calibrated_scales = (scales * multiplier).astype(np.float32)
        prediction_path = (
            prediction_dir / f"round48_{candidate_id}_seed_{seed}.npz"
        )
        _save_prediction_artifact(
            prediction_path,
            indices=indices,
            weights=weights,
            locations=locations,
            scales=calibrated_scales,
            scale_multiplier=multiplier,
        )
        artifact = MixtureArtifact(
            candidate_id=candidate_id,
            components=components,
            seed=seed,
            epochs=epochs_run,
            best_epoch=best_epoch,
            best_early_stop_composite=float(best_losses["composite"]),
            best_early_stop_negative_log_likelihood=float(
                best_losses["negative_log_likelihood"]
            ),
            best_early_stop_hurdle_cross_entropy=float(
                best_losses["hurdle_cross_entropy"]
            ),
            best_early_stop_pairwise_rank_loss=float(
                best_losses["pairwise_rank_loss"]
            ),
            optimizer_updates=optimizer_updates,
            parameter_count=sum(value.numel() for value in reloaded.parameters()),
            scale_multiplier=multiplier,
            calibration_negative_log_likelihood_before=nll_before,
            calibration_negative_log_likelihood_after=nll_after,
            backend_kind=backend_kind,
            backend_device=backend_device,
            model_path=str(model_path),
            model_bytes=model_path.stat().st_size,
            model_sha256=_file_sha256(model_path),
            prediction_path=str(prediction_path),
            prediction_bytes=prediction_path.stat().st_size,
            prediction_sha256=_file_sha256(prediction_path),
            reload_max_abs_weight_error=reload_errors[0],
            reload_max_abs_location_error=reload_errors[1],
            reload_max_abs_scale_error=reload_errors[2],
            warning_count=len(warning_messages),
        )
        artifacts.append(artifact)
        seed_weights.append(weights)
        seed_locations.append(locations)
        seed_scales.append(calibrated_scales)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 48 finalization used CPU fallback: {fallback}")
    if global_indices is None:
        raise RuntimeError("Round 48 finalization produced no indices")
    return MixtureForecastBundle(
        candidate_id=candidate_id,
        components=components,
        global_indices=global_indices,
        seed_weights=np.stack(seed_weights).astype(np.float32),
        seed_locations_normalized=np.stack(seed_locations).astype(np.float32),
        seed_scales_normalized=np.stack(seed_scales).astype(np.float32),
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def train_minute_mixture_candidates(
    dataset: MinuteTemporalDataset,
    *,
    model_dir: Path,
    prediction_dir: Path,
    compute_backend: str,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, MixtureForecastBundle], dict[str, object]]:
    if compute_backend == "directml":
        device, preflight = directml_mixture_preflight()
    elif compute_backend == "cpu":
        device, preflight = cpu_mixture_preflight()
    else:
        raise ValueError("Round 48 compute backend must be directml or cpu")
    if progress is not None:
        progress("round48_preflight", {"status": "complete", **preflight})
    feature_scaler = fit_robust_feature_scaler(dataset)
    target_scaler = fit_robust_target_scaler(dataset)
    normalized_features = feature_scaler.transform(dataset.features)
    normalized_targets = target_scaler.normalize(dataset.signed_target_bps)
    if progress is not None:
        progress(
            "round48_scaling",
            {
                "status": "complete",
                "feature_bytes": int(normalized_features.nbytes),
                "target_bytes": int(normalized_targets.nbytes),
                "training_windows": int(_training_pairs(dataset).shape[0]),
            },
        )
    bundles: dict[str, MixtureForecastBundle] = {}
    for candidate_id, components in CANDIDATE_COMPONENTS.items():
        bundles[candidate_id] = _train_candidate(
            dataset,
            normalized_features,
            normalized_targets,
            feature_scaler,
            target_scaler,
            candidate_id=candidate_id,
            components=components,
            model_dir=model_dir,
            prediction_dir=prediction_dir,
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            progress=progress,
        )
    return bundles, preflight


__all__ = [
    "ANALYSIS_ROLES",
    "BATCH_SIZE",
    "CANDIDATE_COMPONENTS",
    "DECISION_INTERVAL_MINUTES",
    "HORIZONS_MINUTES",
    "LogisticMixtureTCN",
    "MAXIMUM_EPOCHS",
    "MinuteTemporalDataset",
    "MixtureArtifact",
    "MixtureForecastBundle",
    "RECEPTIVE_FIELD_STEPS",
    "ROUND",
    "RobustFeatureScaler",
    "RobustTargetScaler",
    "SEEDS",
    "SUPERVISED_STEPS",
    "WINDOW_STEPS",
    "build_minute_temporal_dataset",
    "cpu_mixture_preflight",
    "directml_mixture_preflight",
    "fit_robust_feature_scaler",
    "fit_robust_target_scaler",
    "hurdle_probabilities",
    "logistic_mixture_cdf",
    "logistic_mixture_log_density",
    "mixture_mean",
    "mixture_objective",
    "numpy_hurdle_probabilities",
    "numpy_logistic_mixture_cdf",
    "numpy_logistic_mixture_log_density",
    "pairwise_expected_return_rank_loss",
    "train_minute_mixture_candidates",
]
