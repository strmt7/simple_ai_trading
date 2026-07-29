"""Live causal feature parity for the BTC Polymarket Round 16 shadow."""

from __future__ import annotations

import numpy as np

from .polymarket_historical_shadow import PolymarketBtcFlowBuffer
from .polymarket_round16 import ROUND16_DURATION_MS
from .polymarket_round16_dataset import build_round16_feature_vector


ROUND16_LIVE_LOOKBACK_SECONDS = 150


class PolymarketRound16LiveFeatureBuilder:
    """Build Round 16 features from public feeds without execution authority."""

    trading_authority = False

    def __init__(self, flow: PolymarketBtcFlowBuffer) -> None:
        if not isinstance(flow, PolymarketBtcFlowBuffer):
            raise TypeError("flow must be PolymarketBtcFlowBuffer")
        if flow.retention_seconds < ROUND16_LIVE_LOOKBACK_SECONDS:
            raise ValueError("Round 16 live flow retention is insufficient")
        self.flow = flow

    def feature_vector(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> np.ndarray:
        snapshot = self.flow.causal_flow_snapshot(
            decision_time_ms=int(decision_time_ms),
            observed_at_ms=int(observed_at_ms),
            second_count=ROUND16_LIVE_LOOKBACK_SECONDS,
        )
        second_ms = np.asarray(snapshot["second_ms"], dtype=np.int64)
        if (
            second_ms.shape != (ROUND16_LIVE_LOOKBACK_SECONDS,)
            or np.any(np.diff(second_ms) != 1_000)
        ):
            raise RuntimeError("Round 16 live flow chronology differs")
        return build_round16_feature_vector(
            event_start_ms=int(event_start_ms),
            event_end_ms=int(event_start_ms) + ROUND16_DURATION_MS,
            flow_start_ms=int(second_ms[0]),
            decision_time_ms=int(decision_time_ms),
            flow=snapshot,
        )


__all__ = [
    "ROUND16_LIVE_LOOKBACK_SECONDS",
    "PolymarketRound16LiveFeatureBuilder",
]
