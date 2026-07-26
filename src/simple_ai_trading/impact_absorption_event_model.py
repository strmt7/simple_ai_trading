"""Probabilistic candidate models for causal Round 74 event sequences.

The candidate panel intentionally contains both a compact pooling MLP and a
causal dilated TCN. Architecture complexity has no promotion privilege. Model
selection must occur later on sealed, after-cost evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .impact_absorption_event_sequence import ROUND74_EVENT_FEATURE_NAMES


ROUND74_EVENT_MODEL_SCHEMA_VERSION = "round-074-event-payoff-model-v1"
ROUND74_EVENT_MODEL_CANDIDATES = ("event_pooling_mlp", "causal_event_tcn")
ROUND74_EVENT_PAYOFF_SIDES = ("long", "short")
ROUND74_EVENT_PAYOFF_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
ROUND74_EVENT_SEQUENCE_LENGTH = 128
ROUND74_EVENT_HIDDEN_CHANNELS = 64
ROUND74_EVENT_TCN_DILATIONS = (1, 2, 4, 8)
ROUND74_EVENT_TCN_KERNEL_SIZE = 3


@dataclass(frozen=True)
class Round74EventModelOutput:
    """Distributional payoff and risk heads for long and short candidates."""

    payoff_quantiles_bps: torch.Tensor
    positive_payoff_logits: torch.Tensor
    adverse_selection_logits: torch.Tensor
    regime_unpredictability_logit: torch.Tensor

    def validate(self, batch_size: int) -> None:
        expected_quantiles = (
            int(batch_size),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        expected_sides = (int(batch_size), len(ROUND74_EVENT_PAYOFF_SIDES))
        if self.payoff_quantiles_bps.shape != expected_quantiles:
            raise ValueError("Round 74 payoff-quantile output shape differs")
        if self.positive_payoff_logits.shape != expected_sides:
            raise ValueError("Round 74 positive-payoff output shape differs")
        if self.adverse_selection_logits.shape != expected_sides:
            raise ValueError("Round 74 adverse-selection output shape differs")
        if self.regime_unpredictability_logit.shape != (int(batch_size), 1):
            raise ValueError("Round 74 unpredictability output shape differs")
        tensors = (
            self.payoff_quantiles_bps,
            self.positive_payoff_logits,
            self.adverse_selection_logits,
            self.regime_unpredictability_logit,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Round 74 model output contains nonfinite values")
        differences = self.payoff_quantiles_bps[:, :, 1:] - (
            self.payoff_quantiles_bps[:, :, :-1]
        )
        if bool((differences < 0.0).any()):
            raise ValueError("Round 74 payoff quantiles cross")


class _Round74DistributionalHeads(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.payoff = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_SIDES)
            * len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        self.positive = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        self.adverse = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        self.unpredictability = nn.Linear(hidden_channels, 1)
        with torch.no_grad():
            reshaped = self.payoff.bias.reshape(
                len(ROUND74_EVENT_PAYOFF_SIDES),
                len(ROUND74_EVENT_PAYOFF_QUANTILES),
            )
            reshaped[:, 0] = 0.0
            reshaped[:, 1:] = -2.0

    def forward(self, encoded: torch.Tensor) -> Round74EventModelOutput:
        raw = self.payoff(encoded).reshape(
            encoded.shape[0],
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        median = raw[:, :, 0]
        lower_near = F.softplus(raw[:, :, 1])
        lower_far = F.softplus(raw[:, :, 2])
        upper_near = F.softplus(raw[:, :, 3])
        upper_far = F.softplus(raw[:, :, 4])
        q25 = median - lower_near
        q10 = q25 - lower_far
        q75 = median + upper_near
        q90 = q75 + upper_far
        output = Round74EventModelOutput(
            payoff_quantiles_bps=torch.stack(
                (q10, q25, median, q75, q90),
                dim=2,
            ),
            positive_payoff_logits=self.positive(encoded),
            adverse_selection_logits=self.adverse(encoded),
            regime_unpredictability_logit=self.unpredictability(encoded),
        )
        output.validate(encoded.shape[0])
        return output


class Round74EventPoolingMLP(nn.Module):
    """Simple last/mean/dispersion baseline over causal event windows."""

    def __init__(
        self,
        input_features: int = len(ROUND74_EVENT_FEATURE_NAMES),
        hidden_channels: int = ROUND74_EVENT_HIDDEN_CHANNELS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_features * 3, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.heads = _Round74DistributionalHeads(hidden_channels)

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        if values.ndim != 3 or values.shape[2] != self.input_features:
            raise ValueError("Round 74 pooling MLP input dimensions are invalid")
        if values.shape[1] < 2:
            raise ValueError("Round 74 pooling MLP requires multiple events")
        mean = values.mean(dim=1)
        centered = values - mean.unsqueeze(1)
        dispersion = torch.sqrt((centered * centered).mean(dim=1) + 1e-6)
        summary = torch.cat((values[:, -1, :], mean, dispersion), dim=1)
        return self.heads(self.encoder(summary))


class _Round74CausalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (ROUND74_EVENT_TCN_KERNEL_SIZE - 1) * int(dilation)
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=ROUND74_EVENT_TCN_KERNEL_SIZE,
            dilation=int(dilation),
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        encoded = F.pad(values, (self.left_padding, 0))
        encoded = F.gelu(self.temporal(encoded))
        encoded = self.dropout(encoded)
        encoded = self.pointwise(encoded)
        encoded = self.dropout(encoded)
        return F.gelu(encoded + residual)


class Round74CausalEventTCN(nn.Module):
    """Compact causal TCN that retains subsecond event order."""

    def __init__(
        self,
        input_features: int = len(ROUND74_EVENT_FEATURE_NAMES),
        hidden_channels: int = ROUND74_EVENT_HIDDEN_CHANNELS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.projection = nn.Conv1d(
            self.input_features,
            hidden_channels,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            _Round74CausalResidualBlock(
                hidden_channels,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in ROUND74_EVENT_TCN_DILATIONS
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.heads = _Round74DistributionalHeads(hidden_channels)

    def _encode_events(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[2] != self.input_features:
            raise ValueError("Round 74 event TCN input dimensions are invalid")
        if values.shape[1] < 2:
            raise ValueError("Round 74 event TCN requires multiple events")
        encoded = F.gelu(self.projection(values.transpose(1, 2)))
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        encoded = self._encode_events(values)
        return self.heads(self.readout(encoded[:, :, -1]))


def build_round74_event_model(candidate_id: str) -> nn.Module:
    selected = str(candidate_id).strip().lower()
    if selected == "event_pooling_mlp":
        return Round74EventPoolingMLP()
    if selected == "causal_event_tcn":
        return Round74CausalEventTCN()
    raise ValueError("Round 74 event model candidate is unsupported")


def round74_event_model_loss(
    output: Round74EventModelOutput,
    *,
    net_payoff_bps: torch.Tensor,
    adverse_selection: torch.Tensor,
    regime_unpredictable: torch.Tensor,
    positive_weight: float = 0.25,
    adverse_weight: float = 0.20,
    unpredictability_weight: float = 0.10,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Use proper probabilistic losses without optimizing a backtest metric."""

    if net_payoff_bps.ndim != 2:
        raise ValueError("Round 74 net-payoff target dimensions differ")
    batch_size = int(net_payoff_bps.shape[0])
    output.validate(batch_size)
    if net_payoff_bps.shape != (
        batch_size,
        len(ROUND74_EVENT_PAYOFF_SIDES),
    ):
        raise ValueError("Round 74 net-payoff target shape differs")
    if adverse_selection.shape != net_payoff_bps.shape:
        raise ValueError("Round 74 adverse-selection target shape differs")
    if regime_unpredictable.shape != (batch_size, 1):
        raise ValueError("Round 74 unpredictability target shape differs")
    targets = (net_payoff_bps, adverse_selection, regime_unpredictable)
    if not all(bool(torch.isfinite(value).all()) for value in targets):
        raise ValueError("Round 74 event-model targets contain nonfinite values")
    if bool(((adverse_selection < 0.0) | (adverse_selection > 1.0)).any()):
        raise ValueError("Round 74 adverse-selection targets are outside [0, 1]")
    if bool(((regime_unpredictable < 0.0) | (regime_unpredictable > 1.0)).any()):
        raise ValueError("Round 74 unpredictability targets are outside [0, 1]")
    loss_weights = (
        float(positive_weight),
        float(adverse_weight),
        float(unpredictability_weight),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in loss_weights):
        raise ValueError("Round 74 event-model loss weights must be finite and nonnegative")
    quantile_levels = torch.tensor(
        ROUND74_EVENT_PAYOFF_QUANTILES,
        dtype=output.payoff_quantiles_bps.dtype,
        device=output.payoff_quantiles_bps.device,
    ).reshape(1, 1, -1)
    errors = net_payoff_bps.unsqueeze(2) - output.payoff_quantiles_bps
    pinball = torch.maximum(
        quantile_levels * errors,
        (quantile_levels - 1.0) * errors,
    ).mean()
    positive_targets = (net_payoff_bps > 0.0).to(net_payoff_bps.dtype)
    # BCE(logits) = softplus(logits) - target * logits. Keeping this explicit
    # avoids torch-directml's silent CPU fallback through log_sigmoid_forward.
    positive = (
        F.softplus(output.positive_payoff_logits)
        - positive_targets * output.positive_payoff_logits
    ).mean()
    adverse = (
        F.softplus(output.adverse_selection_logits)
        - adverse_selection * output.adverse_selection_logits
    ).mean()
    unpredictability = (
        F.softplus(output.regime_unpredictability_logit)
        - regime_unpredictable * output.regime_unpredictability_logit
    ).mean()
    total = (
        pinball
        + loss_weights[0] * positive
        + loss_weights[1] * adverse
        + loss_weights[2] * unpredictability
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("Round 74 event-model loss is nonfinite")
    return total, {
        "pinball": pinball,
        "positive_bce": positive,
        "adverse_bce": adverse,
        "unpredictability_bce": unpredictability,
    }


def _fallback_messages(messages: list[str]) -> list[str]:
    return [
        message
        for message in messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    ]


def round74_event_model_preflight(
    compute_backend: str = "auto",
) -> tuple[object, dict[str, object]]:
    """Run bounded forward/backward updates for both candidates."""

    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    generator = np.random.default_rng(7401)
    values = torch.from_numpy(
        generator.normal(
            size=(
                4,
                ROUND74_EVENT_SEQUENCE_LENGTH,
                len(ROUND74_EVENT_FEATURE_NAMES),
            )
        ).astype(np.float32)
    ).to(device)
    payoff = torch.from_numpy(
        generator.normal(size=(4, len(ROUND74_EVENT_PAYOFF_SIDES))).astype(
            np.float32
        )
    ).to(device)
    adverse = torch.from_numpy(
        generator.integers(
            0,
            2,
            size=(4, len(ROUND74_EVENT_PAYOFF_SIDES)),
        ).astype(np.float32)
    ).to(device)
    unpredictable = torch.from_numpy(
        generator.integers(0, 2, size=(4, 1)).astype(np.float32)
    ).to(device)
    evidence: dict[str, object] = {
        "schema_version": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
        "backend_requested": backend.requested,
        "backend_kind": backend.kind,
        "backend_device": str(device),
        "backend_vendor": backend.vendor,
        "backend_selection": backend.selection,
        "backend_accelerated": backend.accelerated,
        "torch_version": str(torch.__version__),
        "sequence_length": ROUND74_EVENT_SEQUENCE_LENGTH,
        "feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
        "candidates": {},
    }
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES:
            torch.manual_seed(7401)
            model = build_round74_event_model(candidate_id).to(device)
            first = next(model.parameters())
            before = first.detach().cpu().clone()
            model.zero_grad(set_to_none=True)
            output = model(values)
            loss, components = round74_event_model_loss(
                output,
                net_payoff_bps=payoff,
                adverse_selection=adverse,
                regime_unpredictable=unpredictable,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                foreach=False,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-1e-4)
            change = float(torch.max(torch.abs(first.detach().cpu() - before)))
            component_values = {
                key: float(value.detach().cpu())
                for key, value in components.items()
            }
            if (
                not math.isfinite(float(loss.detach().cpu()))
                or not all(math.isfinite(value) for value in component_values.values())
                or change <= 0.0
            ):
                raise RuntimeError(
                    f"Round 74 {candidate_id} preflight produced invalid evidence"
                )
            evidence["candidates"][candidate_id] = {
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "loss": float(loss.detach().cpu()),
                "components": component_values,
                "parameter_max_abs_change": change,
            }
        messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(messages)
    if fallback:
        raise RuntimeError(
            f"Round 74 event-model preflight used CPU fallback: {fallback}"
        )
    evidence["warning_count"] = len(messages)
    evidence["cpu_fallback_warning_count"] = 0
    evidence["target_source"] = "synthetic_preflight_only"
    evidence["financial_edge_tested"] = False
    evidence["profitability_claim"] = False
    return device, evidence


__all__ = [
    "ROUND74_EVENT_HIDDEN_CHANNELS",
    "ROUND74_EVENT_MODEL_CANDIDATES",
    "ROUND74_EVENT_MODEL_SCHEMA_VERSION",
    "ROUND74_EVENT_PAYOFF_QUANTILES",
    "ROUND74_EVENT_PAYOFF_SIDES",
    "ROUND74_EVENT_SEQUENCE_LENGTH",
    "ROUND74_EVENT_TCN_DILATIONS",
    "ROUND74_EVENT_TCN_KERNEL_SIZE",
    "Round74CausalEventTCN",
    "Round74EventModelOutput",
    "Round74EventPoolingMLP",
    "build_round74_event_model",
    "round74_event_model_loss",
    "round74_event_model_preflight",
]
