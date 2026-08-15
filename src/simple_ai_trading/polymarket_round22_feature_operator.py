"""Bounded causal-feature materialization for Round 22 development books."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .polymarket_round22_feature_store import Round22FeatureStore
from .polymarket_round22_features import build_round22_condition_features
from .polymarket_round22_ingestion import (
    POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN,
)
from .polymarket_round22_pilot import Round22PilotStore


ProgressCallback = Callable[[str, Mapping[str, object]], None]
_DEVELOPMENT_ROLES = frozenset({"train", "tune_calibration", "tune_selection"})


@dataclass(frozen=True, slots=True)
class Round22FeatureMaterializationResult:
    requested_limit: int
    selection_role: str
    source_condition_count: int
    already_complete_count: int
    committed_count: int
    remaining_materializable_condition_count: int
    committed_condition_ids: tuple[str, ...]
    target_accessed: bool = False
    binance_used: bool = False
    trading_authority: bool = False


def materialize_round22_development_features(
    *,
    pilot_store: Round22PilotStore,
    feature_store: Round22FeatureStore,
    maximum_conditions: int = 1,
    role: str | None = None,
    progress: ProgressCallback | None = None,
) -> Round22FeatureMaterializationResult:
    if feature_store.pilot_store is not pilot_store:
        raise ValueError("Round 22 feature operator store differs")
    if (
        type(maximum_conditions) is not int
        or not 1 <= maximum_conditions <= POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN
    ):
        raise ValueError("Round 22 feature condition limit is outside the bound")
    if pilot_store.target_row_count() != 0:
        raise ValueError("Round 22 feature operator detected target access")
    selection_role = str(role or "all_development").strip().lower()
    if selection_role not in {"all_development", *_DEVELOPMENT_ROLES}:
        raise ValueError("Round 22 feature role is invalid")
    rows = pilot_store.connection.execute(
        """
        SELECT identity.condition_id, identity.slug, identity.role
        FROM feature.market_identity AS identity
        INNER JOIN feature.condition_manifest AS source
          ON source.condition_id = identity.condition_id
        ORDER BY identity.event_start_ms
        """
    ).fetchall()
    development = tuple(
        (str(condition_id), str(slug), str(role))
        for condition_id, slug, role in rows
        if str(role) in _DEVELOPMENT_ROLES
        and (selection_role == "all_development" or str(role) == selection_role)
    )
    completed = feature_store.completed_condition_ids()
    pending = tuple(item for item in development if item[0] not in completed)
    selected = pending[:maximum_conditions]
    committed: list[str] = []
    for index, (condition_id, slug, role) in enumerate(selected, start=1):
        if progress is not None:
            progress(
                "feature_source_audit",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "role": role,
                    "slug": slug,
                },
            )
        up_window, down_window = pilot_store.condition_windows(condition_id)
        result = build_round22_condition_features(
            repository=pilot_store.contract.repository,
            up_window=up_window,
            down_window=down_window,
        )
        if progress is not None:
            progress(
                "feature_grid_built",
                {
                    "available_count": sum(row.available for row in result.rows),
                    "condition_id": condition_id,
                    "row_count": len(result.rows),
                },
            )
        if not feature_store.put_condition(result):
            raise ValueError("Round 22 pending feature condition became non-atomic")
        committed.append(condition_id)
        if progress is not None:
            progress(
                "feature_condition_committed",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "condition_id": condition_id,
                },
            )
    return Round22FeatureMaterializationResult(
        requested_limit=maximum_conditions,
        selection_role=selection_role,
        source_condition_count=len(development),
        already_complete_count=len(
            completed & {condition_id for condition_id, _, _ in development}
        ),
        committed_count=len(committed),
        remaining_materializable_condition_count=len(pending) - len(committed),
        committed_condition_ids=tuple(committed),
    )


__all__ = [
    "Round22FeatureMaterializationResult",
    "materialize_round22_development_features",
]
