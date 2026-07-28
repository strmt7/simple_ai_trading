"""Probabilistic candidate models for causal Round 74 event sequences.

The candidate panel intentionally contains pooled linear and MLP controls, a
causal dilated TCN, and a compact causal attention encoder. Architecture
complexity has no promotion privilege. Model selection must occur later on
sealed, after-cost evidence.
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
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)


ROUND74_EVENT_MODEL_SCHEMA_VERSION = "round-074-event-payoff-model-v6"
ROUND74_EVENT_MODEL_CANDIDATES = (
    "event_pooling_linear",
    "event_pooling_mlp",
    "causal_event_tcn",
    "causal_event_attention",
)
ROUND74_EVENT_HIDDEN_CHANNELS = 64
ROUND74_EVENT_TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
ROUND74_EVENT_TCN_KERNEL_SIZE = 3
ROUND74_EVENT_TCN_RECEPTIVE_FIELD = 1 + (
    (ROUND74_EVENT_TCN_KERNEL_SIZE - 1) * sum(ROUND74_EVENT_TCN_DILATIONS)
)
ROUND74_EVENT_ATTENTION_HIDDEN_CHANNELS = 72
ROUND74_EVENT_ATTENTION_HEADS = 4
ROUND74_EVENT_ATTENTION_LAYERS = 3
ROUND74_EVENT_ATTENTION_EXPANSION = 2


@dataclass(frozen=True)
class Round74EventModelOutput:
    """Multi-horizon distributions and path-risk heads for both trade sides."""

    payoff_quantiles_bps: torch.Tensor
    maximum_adverse_excursion_quantiles_bps: torch.Tensor
    positive_payoff_logits: torch.Tensor
    adverse_selection_logits: torch.Tensor
    regime_unpredictability_logits: torch.Tensor

    def validate(self, batch_size: int) -> None:
        expected_quantiles = (
            int(batch_size),
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        expected_sides = (
            int(batch_size),
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        if self.payoff_quantiles_bps.shape != expected_quantiles:
            raise ValueError("Round 74 payoff-quantile output shape differs")
        if self.maximum_adverse_excursion_quantiles_bps.shape != expected_quantiles:
            raise ValueError("Round 74 adverse-excursion output shape differs")
        if self.positive_payoff_logits.shape != expected_sides:
            raise ValueError("Round 74 positive-payoff output shape differs")
        if self.adverse_selection_logits.shape != expected_sides:
            raise ValueError("Round 74 adverse-selection output shape differs")
        if self.regime_unpredictability_logits.shape != (
            int(batch_size),
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        ):
            raise ValueError("Round 74 unpredictability output shape differs")
        tensors = (
            self.payoff_quantiles_bps,
            self.maximum_adverse_excursion_quantiles_bps,
            self.positive_payoff_logits,
            self.adverse_selection_logits,
            self.regime_unpredictability_logits,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Round 74 model output contains nonfinite values")
        payoff_differences = (
            self.payoff_quantiles_bps[..., 1:] - (self.payoff_quantiles_bps[..., :-1])
        )
        adverse_differences = (
            self.maximum_adverse_excursion_quantiles_bps[..., 1:]
            - self.maximum_adverse_excursion_quantiles_bps[..., :-1]
        )
        if bool((payoff_differences < 0.0).any()) or bool(
            (adverse_differences < 0.0).any()
        ):
            raise ValueError("Round 74 payoff quantiles cross")
        if bool((self.maximum_adverse_excursion_quantiles_bps < 0.0).any()):
            raise ValueError("Round 74 adverse excursion is negative")


class _Round74DistributionalHeads(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.payoff = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES)
            * len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        self.maximum_adverse_excursion = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES)
            * len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        self.positive = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        self.adverse = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        self.unpredictability = nn.Linear(
            hidden_channels,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        )
        with torch.no_grad():
            reshaped = self.payoff.bias.reshape(
                len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
                len(ROUND74_EVENT_PAYOFF_SIDES),
                len(ROUND74_EVENT_PAYOFF_QUANTILES),
            )
            reshaped[:, :, 0] = 0.0
            reshaped[:, :, 1:] = -2.0
            self.maximum_adverse_excursion.bias.fill_(-2.0)

    def forward(self, encoded: torch.Tensor) -> Round74EventModelOutput:
        raw = self.payoff(encoded).reshape(
            encoded.shape[0],
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        median = raw[..., 0]
        lower_near = F.softplus(raw[..., 1])
        lower_far = F.softplus(raw[..., 2])
        upper_near = F.softplus(raw[..., 3])
        upper_far = F.softplus(raw[..., 4])
        q25 = median - lower_near
        q10 = q25 - lower_far
        q75 = median + upper_near
        q90 = q75 + upper_far
        adverse_raw = self.maximum_adverse_excursion(encoded).reshape(
            encoded.shape[0],
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        adverse_q10 = F.softplus(adverse_raw[..., 0])
        adverse_q25 = adverse_q10 + F.softplus(adverse_raw[..., 1])
        adverse_q50 = adverse_q25 + F.softplus(adverse_raw[..., 2])
        adverse_q75 = adverse_q50 + F.softplus(adverse_raw[..., 3])
        adverse_q90 = adverse_q75 + F.softplus(adverse_raw[..., 4])
        expected_sides = (
            encoded.shape[0],
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        output = Round74EventModelOutput(
            payoff_quantiles_bps=torch.stack(
                (q10, q25, median, q75, q90),
                dim=3,
            ),
            maximum_adverse_excursion_quantiles_bps=torch.stack(
                (
                    adverse_q10,
                    adverse_q25,
                    adverse_q50,
                    adverse_q75,
                    adverse_q90,
                ),
                dim=3,
            ),
            positive_payoff_logits=self.positive(encoded).reshape(expected_sides),
            adverse_selection_logits=self.adverse(encoded).reshape(expected_sides),
            regime_unpredictability_logits=self.unpredictability(encoded),
        )
        output.validate(encoded.shape[0])
        return output


def _round74_pooling_summary(
    values: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    if values.ndim != 3 or values.shape[2] != input_features:
        raise ValueError("Round 74 pooled model input dimensions are invalid")
    if values.shape[1] < 2:
        raise ValueError("Round 74 pooled model requires multiple events")
    mean = values.mean(dim=1)
    centered = values - mean.unsqueeze(1)
    dispersion = torch.sqrt((centered * centered).mean(dim=1) + 1e-6)
    return torch.cat((values[:, -1, :], mean, dispersion), dim=1)


class Round74EventPoolingLinear(nn.Module):
    """Low-capacity control over last, mean, and dispersion summaries."""

    def __init__(
        self,
        input_features: int = len(ROUND74_EVENT_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.heads = _Round74DistributionalHeads(self.input_features * 3)

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        return self.heads(
            _round74_pooling_summary(
                values,
                input_features=self.input_features,
            )
        )


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
        summary = _round74_pooling_summary(
            values,
            input_features=self.input_features,
        )
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
        if values.shape[1] > ROUND74_EVENT_TCN_RECEPTIVE_FIELD:
            raise ValueError(
                "Round 74 event TCN input exceeds its causal receptive field"
            )
        encoded = F.gelu(self.projection(values.transpose(1, 2)))
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        encoded = self._encode_events(values)
        return self.heads(self.readout(encoded[:, :, -1]))


class _Round74CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        *,
        attention_heads: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_channels % attention_heads != 0:
            raise ValueError("Round 74 attention head dimensions differ")
        self.attention_heads = int(attention_heads)
        self.head_channels = int(hidden_channels) // self.attention_heads
        self.attention_scale = 1.0 / math.sqrt(float(self.head_channels))
        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.query_key_value = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attention_output = nn.Linear(hidden_channels, hidden_channels)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.feed_forward_norm = nn.LayerNorm(hidden_channels)
        self.feed_forward_input = nn.Linear(
            hidden_channels,
            hidden_channels * int(expansion),
        )
        self.feed_forward_output = nn.Linear(
            hidden_channels * int(expansion),
            hidden_channels,
        )
        self.feed_forward_dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        values: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_channels = values.shape
        normalized = self.attention_norm(values)
        query, key, projected_value = self.query_key_value(normalized).chunk(
            3,
            dim=2,
        )

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size,
                sequence_length,
                self.attention_heads,
                self.head_channels,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        projected_value = split_heads(projected_value)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.attention_scale
        scores = scores.masked_fill(
            causal_mask[:sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(weights, projected_value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            hidden_channels,
        )
        values = values + self.attention_dropout(self.attention_output(attended))
        feed_forward = self.feed_forward_output(
            F.gelu(self.feed_forward_input(self.feed_forward_norm(values)))
        )
        return values + self.feed_forward_dropout(feed_forward)


class Round74CausalEventAttention(nn.Module):
    """Compact pre-norm attention encoder over the causal event window."""

    def __init__(
        self,
        input_features: int = len(ROUND74_EVENT_FEATURE_NAMES),
        hidden_channels: int = ROUND74_EVENT_ATTENTION_HIDDEN_CHANNELS,
        attention_heads: int = ROUND74_EVENT_ATTENTION_HEADS,
        layers: int = ROUND74_EVENT_ATTENTION_LAYERS,
        expansion: int = ROUND74_EVENT_ATTENTION_EXPANSION,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.hidden_channels = int(hidden_channels)
        self.input_projection = nn.Linear(
            self.input_features,
            self.hidden_channels,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(
                1,
                ROUND74_EVENT_SEQUENCE_LENGTH,
                self.hidden_channels,
            )
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            _Round74CausalAttentionBlock(
                self.hidden_channels,
                attention_heads=attention_heads,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(int(layers))
        )
        self.final_norm = nn.LayerNorm(self.hidden_channels)
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.heads = _Round74DistributionalHeads(self.hidden_channels)
        self.register_buffer(
            "_causal_mask",
            torch.triu(
                torch.ones(
                    ROUND74_EVENT_SEQUENCE_LENGTH,
                    ROUND74_EVENT_SEQUENCE_LENGTH,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

    def _encode_events(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[2] != self.input_features:
            raise ValueError("Round 74 event attention input dimensions are invalid")
        if values.shape[1] < 2:
            raise ValueError("Round 74 event attention requires multiple events")
        if values.shape[1] > ROUND74_EVENT_SEQUENCE_LENGTH:
            raise ValueError(
                "Round 74 event attention input exceeds the frozen sequence length"
            )
        sequence_length = int(values.shape[1])
        encoded = self.input_projection(values)
        encoded = encoded + self.position_embedding[:, :sequence_length, :]
        for block in self.blocks:
            encoded = block(
                encoded,
                causal_mask=self._causal_mask,
            )
        return self.final_norm(encoded)

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        encoded = self._encode_events(values)
        return self.heads(self.readout(encoded[:, -1, :]))


def build_round74_event_model(candidate_id: str) -> nn.Module:
    selected = str(candidate_id).strip().lower()
    if selected == "event_pooling_linear":
        return Round74EventPoolingLinear()
    if selected == "event_pooling_mlp":
        return Round74EventPoolingMLP()
    if selected == "causal_event_tcn":
        return Round74CausalEventTCN()
    if selected == "causal_event_attention":
        return Round74CausalEventAttention()
    raise ValueError("Round 74 event model candidate is unsupported")


def _round74_event_model_loss_impl(
    output: Round74EventModelOutput,
    *,
    net_payoff_bps: torch.Tensor,
    maximum_adverse_excursion_bps: torch.Tensor,
    adverse_selection: torch.Tensor,
    regime_unpredictable: torch.Tensor,
    action_eligibility: torch.Tensor | None = None,
    regime_unpredictability_eligibility: torch.Tensor | None = None,
    payoff_loss_scale_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_loss_scale_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_weight: float = 0.35,
    positive_weight: float = 0.25,
    adverse_weight: float = 0.20,
    unpredictability_weight: float = 0.10,
    inputs_validated: bool = False,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Use proper probabilistic losses without optimizing a backtest metric."""

    if net_payoff_bps.ndim != 3:
        raise ValueError("Round 74 net-payoff target dimensions differ")
    batch_size = int(net_payoff_bps.shape[0])
    if not inputs_validated:
        output.validate(batch_size)
    expected_action_shape = (
        batch_size,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    if net_payoff_bps.shape != expected_action_shape:
        raise ValueError("Round 74 net-payoff target shape differs")
    if maximum_adverse_excursion_bps.shape != expected_action_shape:
        raise ValueError("Round 74 adverse-excursion target shape differs")
    if adverse_selection.shape != net_payoff_bps.shape:
        raise ValueError("Round 74 adverse-selection target shape differs")
    if regime_unpredictable.shape != (
        batch_size,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    ):
        raise ValueError("Round 74 unpredictability target shape differs")
    if action_eligibility is None:
        action_eligibility = torch.ones_like(net_payoff_bps)
    if regime_unpredictability_eligibility is None:
        regime_unpredictability_eligibility = (action_eligibility.sum(dim=2) > 0.0).to(
            net_payoff_bps.dtype
        )
    if payoff_loss_scale_bps is None:
        payoff_loss_scale_bps = torch.ones_like(net_payoff_bps)
    if maximum_adverse_excursion_loss_scale_bps is None:
        maximum_adverse_excursion_loss_scale_bps = torch.ones_like(
            maximum_adverse_excursion_bps
        )
    if action_eligibility.shape != expected_action_shape:
        raise ValueError("Round 74 action-eligibility shape differs")
    if regime_unpredictability_eligibility.shape != (
        batch_size,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    ):
        raise ValueError("Round 74 unpredictability-eligibility shape differs")
    if (
        payoff_loss_scale_bps.shape != expected_action_shape
        or maximum_adverse_excursion_loss_scale_bps.shape != expected_action_shape
    ):
        raise ValueError("Round 74 target-loss scale shape differs")
    if not inputs_validated:
        targets = (
            net_payoff_bps,
            maximum_adverse_excursion_bps,
            adverse_selection,
            regime_unpredictable,
            action_eligibility,
            regime_unpredictability_eligibility,
            payoff_loss_scale_bps,
            maximum_adverse_excursion_loss_scale_bps,
        )
        if not all(bool(torch.isfinite(value).all()) for value in targets):
            raise ValueError("Round 74 event-model targets contain nonfinite values")
        if bool((maximum_adverse_excursion_bps < 0.0).any()):
            raise ValueError("Round 74 adverse-excursion target is negative")
        if bool(((adverse_selection < 0.0) | (adverse_selection > 1.0)).any()):
            raise ValueError("Round 74 adverse-selection targets are outside [0, 1]")
        if bool(((regime_unpredictable < 0.0) | (regime_unpredictable > 1.0)).any()):
            raise ValueError("Round 74 unpredictability targets are outside [0, 1]")
        if bool((payoff_loss_scale_bps <= 0.0).any()) or bool(
            (maximum_adverse_excursion_loss_scale_bps <= 0.0).any()
        ):
            raise ValueError("Round 74 target-loss scale is not positive")
        if bool(
            ((action_eligibility != 0.0) & (action_eligibility != 1.0)).any()
        ) or bool(
            (
                (regime_unpredictability_eligibility != 0.0)
                & (regime_unpredictability_eligibility != 1.0)
            ).any()
        ):
            raise ValueError("Round 74 event-model eligibility is not binary")
    action_weight = action_eligibility.sum()
    regime_weight = regime_unpredictability_eligibility.sum()
    if not inputs_validated and (
        float(action_weight.detach().cpu()) <= 0.0
        or float(regime_weight.detach().cpu()) <= 0.0
    ):
        raise ValueError("Round 74 event-model batch has no eligible targets")
    loss_weights = (
        float(maximum_adverse_excursion_weight),
        float(positive_weight),
        float(adverse_weight),
        float(unpredictability_weight),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in loss_weights):
        raise ValueError(
            "Round 74 event-model loss weights must be finite and nonnegative"
        )
    quantile_levels = torch.tensor(
        ROUND74_EVENT_PAYOFF_QUANTILES,
        dtype=output.payoff_quantiles_bps.dtype,
        device=output.payoff_quantiles_bps.device,
    ).reshape(1, 1, 1, -1)
    payoff_errors = net_payoff_bps.unsqueeze(3) - output.payoff_quantiles_bps
    payoff_pinball_values = torch.maximum(
        quantile_levels * payoff_errors,
        (quantile_levels - 1.0) * payoff_errors,
    ) / payoff_loss_scale_bps.unsqueeze(3)
    payoff_pinball = (payoff_pinball_values * action_eligibility.unsqueeze(3)).sum() / (
        action_weight * len(ROUND74_EVENT_PAYOFF_QUANTILES)
    )
    adverse_excursion_errors = maximum_adverse_excursion_bps.unsqueeze(3) - (
        output.maximum_adverse_excursion_quantiles_bps
    )
    adverse_excursion_pinball_values = torch.maximum(
        quantile_levels * adverse_excursion_errors,
        (quantile_levels - 1.0) * adverse_excursion_errors,
    ) / maximum_adverse_excursion_loss_scale_bps.unsqueeze(3)
    adverse_excursion_pinball = (
        adverse_excursion_pinball_values * action_eligibility.unsqueeze(3)
    ).sum() / (action_weight * len(ROUND74_EVENT_PAYOFF_QUANTILES))
    positive_targets = (net_payoff_bps > 0.0).to(net_payoff_bps.dtype)
    # BCE(logits) = softplus(logits) - target * logits. Keeping this explicit
    # avoids torch-directml's silent CPU fallback through log_sigmoid_forward.
    positive_values = (
        F.softplus(output.positive_payoff_logits)
        - positive_targets * output.positive_payoff_logits
    )
    positive = (positive_values * action_eligibility).sum() / action_weight
    adverse_values = (
        F.softplus(output.adverse_selection_logits)
        - adverse_selection * output.adverse_selection_logits
    )
    adverse = (adverse_values * action_eligibility).sum() / action_weight
    unpredictability_values = (
        F.softplus(output.regime_unpredictability_logits)
        - regime_unpredictable * output.regime_unpredictability_logits
    )
    unpredictability = (
        unpredictability_values * regime_unpredictability_eligibility
    ).sum() / regime_weight
    total = (
        payoff_pinball
        + loss_weights[0] * adverse_excursion_pinball
        + loss_weights[1] * positive
        + loss_weights[2] * adverse
        + loss_weights[3] * unpredictability
    )
    if not inputs_validated and not bool(torch.isfinite(total)):
        raise ValueError("Round 74 event-model loss is nonfinite")
    return total, {
        "payoff_pinball": payoff_pinball,
        "maximum_adverse_excursion_pinball": adverse_excursion_pinball,
        "positive_bce": positive,
        "adverse_bce": adverse,
        "unpredictability_bce": unpredictability,
    }


def round74_event_model_loss(
    output: Round74EventModelOutput,
    *,
    net_payoff_bps: torch.Tensor,
    maximum_adverse_excursion_bps: torch.Tensor,
    adverse_selection: torch.Tensor,
    regime_unpredictable: torch.Tensor,
    action_eligibility: torch.Tensor | None = None,
    regime_unpredictability_eligibility: torch.Tensor | None = None,
    payoff_loss_scale_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_loss_scale_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_weight: float = 0.35,
    positive_weight: float = 0.25,
    adverse_weight: float = 0.20,
    unpredictability_weight: float = 0.10,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Validate public inputs and evaluate the proper probabilistic loss."""

    return _round74_event_model_loss_impl(
        output,
        net_payoff_bps=net_payoff_bps,
        maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
        adverse_selection=adverse_selection,
        regime_unpredictable=regime_unpredictable,
        action_eligibility=action_eligibility,
        regime_unpredictability_eligibility=regime_unpredictability_eligibility,
        payoff_loss_scale_bps=payoff_loss_scale_bps,
        maximum_adverse_excursion_loss_scale_bps=(
            maximum_adverse_excursion_loss_scale_bps
        ),
        maximum_adverse_excursion_weight=maximum_adverse_excursion_weight,
        positive_weight=positive_weight,
        adverse_weight=adverse_weight,
        unpredictability_weight=unpredictability_weight,
    )


def _round74_event_model_loss_from_validated_inputs(
    output: Round74EventModelOutput,
    *,
    net_payoff_bps: torch.Tensor,
    maximum_adverse_excursion_bps: torch.Tensor,
    adverse_selection: torch.Tensor,
    regime_unpredictable: torch.Tensor,
    action_eligibility: torch.Tensor,
    regime_unpredictability_eligibility: torch.Tensor,
    payoff_loss_scale_bps: torch.Tensor,
    maximum_adverse_excursion_loss_scale_bps: torch.Tensor,
    maximum_adverse_excursion_weight: float,
    positive_weight: float,
    adverse_weight: float,
    unpredictability_weight: float,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Evaluate one slice after its combined model and dataset validation."""

    return _round74_event_model_loss_impl(
        output,
        net_payoff_bps=net_payoff_bps,
        maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
        adverse_selection=adverse_selection,
        regime_unpredictable=regime_unpredictable,
        action_eligibility=action_eligibility,
        regime_unpredictability_eligibility=regime_unpredictability_eligibility,
        payoff_loss_scale_bps=payoff_loss_scale_bps,
        maximum_adverse_excursion_loss_scale_bps=(
            maximum_adverse_excursion_loss_scale_bps
        ),
        maximum_adverse_excursion_weight=maximum_adverse_excursion_weight,
        positive_weight=positive_weight,
        adverse_weight=adverse_weight,
        unpredictability_weight=unpredictability_weight,
        inputs_validated=True,
    )


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
    """Run bounded forward/backward updates for every declared candidate."""

    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    generator = np.random.default_rng(7401)
    values_array = generator.normal(
        size=(
            4,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        )
    ).astype(np.float32)
    values_array[:, :, :8] = 0.0
    for batch_index in range(values_array.shape[0]):
        for event_index in range(values_array.shape[1]):
            values_array[batch_index, event_index, event_index % 5] = 1.0
            values_array[batch_index, event_index, 5 + batch_index % 3] = 1.0
    values = torch.from_numpy(values_array).to(device)
    action_shape = (
        4,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    payoff = torch.from_numpy(
        generator.normal(size=action_shape).astype(np.float32)
    ).to(device)
    maximum_adverse_excursion = torch.from_numpy(
        np.abs(generator.normal(size=action_shape)).astype(np.float32)
    ).to(device)
    adverse = torch.from_numpy(
        generator.integers(
            0,
            2,
            size=action_shape,
        ).astype(np.float32)
    ).to(device)
    unpredictable = torch.from_numpy(
        generator.integers(
            0,
            2,
            size=(4, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)),
        ).astype(np.float32)
    ).to(device)
    action_eligibility = torch.ones(action_shape, device=device)
    action_eligibility[0, 0, 0] = 0.0
    unpredictability_eligibility = torch.ones(
        (4, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)),
        device=device,
    )
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
        "payoff_horizons_seconds": list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
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
                maximum_adverse_excursion_bps=maximum_adverse_excursion,
                adverse_selection=adverse,
                regime_unpredictable=unpredictable,
                action_eligibility=action_eligibility,
                regime_unpredictability_eligibility=(unpredictability_eligibility),
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
                key: float(value.detach().cpu()) for key, value in components.items()
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
    evidence["masked_action_targets"] = 1
    evidence["financial_edge_tested"] = False
    evidence["profitability_claim"] = False
    return device, evidence


__all__ = [
    "ROUND74_EVENT_HIDDEN_CHANNELS",
    "ROUND74_EVENT_MODEL_CANDIDATES",
    "ROUND74_EVENT_MODEL_SCHEMA_VERSION",
    "ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS",
    "ROUND74_EVENT_PAYOFF_QUANTILES",
    "ROUND74_EVENT_PAYOFF_SIDES",
    "ROUND74_EVENT_SEQUENCE_LENGTH",
    "ROUND74_EVENT_TCN_DILATIONS",
    "ROUND74_EVENT_TCN_KERNEL_SIZE",
    "ROUND74_EVENT_TCN_RECEPTIVE_FIELD",
    "Round74CausalEventTCN",
    "Round74EventModelOutput",
    "Round74EventPoolingLinear",
    "Round74EventPoolingMLP",
    "build_round74_event_model",
    "round74_event_model_loss",
    "round74_event_model_preflight",
]
