from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from simple_ai_trading.polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    Round25ClobFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    POLYMARKET_ROUND25_SELECTION_END_MS,
    POLYMARKET_ROUND25_TRAIN_END_MS,
    Round25OfficialResolution,
    Round25ResolutionAuthority,
    build_round25_development_dataset,
    require_round25_dataset_minimum,
    round25_development_role,
    select_round25_condition_endpoints,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    combine_round25_features,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    Round25TwapFeatureSnapshot,
)


CONDITION_ID = "0x" + "a" * 64
UP_TOKEN = "1" * 77
DOWN_TOKEN = "2" * 77


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _joint_row(
    decision_offset_ms: int,
    *,
    start_ms: int = POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    condition_id: str = CONDITION_ID,
    marker: float = 1.0,
) -> object:
    decision = start_ms + decision_offset_ms
    twap = Round25TwapFeatureSnapshot(
        condition_id=condition_id,
        event_start_ms=start_ms,
        decision_time_ms=decision,
        available=True,
        reasons=(),
        values=(marker,) * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES),
        source_chain_sha256=_digest(f"twap:{condition_id}:{decision}:{marker}"),
        maximum_receipt_ms=decision - 100,
        opening_value_e18=65_000 * 10**18,
        latest_value_e18=65_001 * 10**18,
    )
    clob = Round25ClobFeatureSnapshot(
        condition_id=condition_id,
        event_start_ms=start_ms,
        decision_time_ms=decision,
        available=True,
        reasons=(),
        market_prior_probability=0.51,
        values=(marker + 1.0,) * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES),
        source_chain_sha256=_digest(f"clob:{condition_id}:{decision}:{marker}"),
        maximum_receipt_ms=decision - 50,
    )
    return combine_round25_features(twap, clob)


def _condition_rows(
    *,
    start_ms: int = POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    condition_id: str = CONDITION_ID,
    endpoints_per_phase: int = 4,
) -> tuple[object, ...]:
    output = []
    for phase in range(4):
        for index in range(endpoints_per_phase):
            output.append(
                _joint_row(
                    phase * 75_000 + (index + 1) * 1_000,
                    start_ms=start_ms,
                    condition_id=condition_id,
                    marker=float(index + phase + 1),
                )
            )
    return tuple(output)


def _authority() -> Round25ResolutionAuthority:
    return Round25ResolutionAuthority.create(
        terminal_transport_sha256="a" * 64,
        official_resolution_audit_sha256="b" * 64,
        created_at_ms=POLYMARKET_ROUND25_SELECTION_END_MS + 1,
        official_resolution_semantics_verified=True,
    )


def _resolution(
    authority: Round25ResolutionAuthority,
    *,
    start_ms: int = POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    condition_id: str = CONDITION_ID,
    winning_token_id: str = UP_TOKEN,
) -> Round25OfficialResolution:
    return Round25OfficialResolution.create(
        condition_id=condition_id,
        event_start_ms=start_ms,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        winning_token_id=winning_token_id,
        resolved_at_ms=start_ms + 300_000,
        official_payload_sha256=_digest(f"resolution:{condition_id}"),
        authority=authority,
    )


@pytest.mark.parametrize(
    ("event_start_ms", "role"),
    [
        (POLYMARKET_ROUND25_CAMPAIGN_START_MS, "train"),
        (POLYMARKET_ROUND25_TRAIN_END_MS - 600_000, "train"),
        (POLYMARKET_ROUND25_TRAIN_END_MS - 300_000, "purged"),
        (POLYMARKET_ROUND25_TRAIN_END_MS, "purged"),
        (POLYMARKET_ROUND25_TRAIN_END_MS + 300_000, "calibration"),
        (POLYMARKET_ROUND25_CALIBRATION_END_MS - 300_000, "purged"),
        (POLYMARKET_ROUND25_CALIBRATION_END_MS, "purged"),
        (POLYMARKET_ROUND25_CALIBRATION_END_MS + 300_000, "selection"),
        (POLYMARKET_ROUND25_SELECTION_END_MS - 300_000, "selection"),
    ],
)
def test_campaign_roles_and_boundary_purges_are_exact(
    event_start_ms: int,
    role: str,
) -> None:
    assert round25_development_role(event_start_ms) == role


@pytest.mark.parametrize(
    "event_start_ms",
    [
        True,
        POLYMARKET_ROUND25_CAMPAIGN_START_MS - 300_000,
        POLYMARKET_ROUND25_SELECTION_END_MS,
        POLYMARKET_ROUND25_CAMPAIGN_START_MS + 1,
    ],
)
def test_campaign_role_rejects_outside_or_unaligned_start(event_start_ms: object) -> None:
    with pytest.raises(ValueError):
        round25_development_role(event_start_ms)  # type: ignore[arg-type]


def test_endpoint_selection_is_phase_balanced_order_independent_and_target_free() -> None:
    rows = _condition_rows(endpoints_per_phase=5)
    forward = select_round25_condition_endpoints(rows)
    reverse = select_round25_condition_endpoints(tuple(reversed(rows)))

    assert [row.decision_time_ms for row in forward] == [
        row.decision_time_ms for row in reverse
    ]
    assert len(forward) == POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
    assert [
        sum(
            phase * 75_000
            <= row.decision_time_ms - row.event_start_ms
            < (phase + 1) * 75_000
            for row in forward
        )
        for phase in range(4)
    ] == [4, 4, 4, 4]


def test_endpoint_selection_rejects_underfilled_phase_and_duplicate_decision() -> None:
    underfilled = _condition_rows(endpoints_per_phase=3)
    with pytest.raises(ValueError, match="underfilled"):
        select_round25_condition_endpoints(underfilled)

    rows = _condition_rows()
    with pytest.raises(ValueError, match="duplicated"):
        select_round25_condition_endpoints((*rows, rows[0]))


def test_resolution_authority_is_self_hashed_post_campaign_and_non_authoritative() -> None:
    authority = _authority()

    assert authority.validated() is authority
    assert authority.created_at_ms > POLYMARKET_ROUND25_SELECTION_END_MS
    assert authority.official_resolution_semantics_verified is True
    assert authority.trading_authority is False
    with pytest.raises(ValueError, match="authority"):
        replace(authority, authority_sha256="f" * 64).validated()


def test_resolution_requires_official_winning_token_and_authority() -> None:
    authority = _authority()
    resolution = _resolution(authority)

    assert resolution.validated(authority) is resolution
    assert resolution.target_up is True
    assert _resolution(authority, winning_token_id=DOWN_TOKEN).target_up is False
    with pytest.raises(ValueError, match="resolution"):
        replace(resolution, target_origin="constructed_twap_target").validated(authority)
    with pytest.raises(ValueError, match="resolution"):
        replace(resolution, winning_token_id="3" * 77).validated(authority)


def test_small_dataset_is_valid_but_explicitly_ineligible_for_training() -> None:
    authority = _authority()
    dataset = build_round25_development_dataset(
        role="train",
        snapshots=_condition_rows(),
        resolutions=(_resolution(authority),),
        resolution_authority=authority,
    )

    assert dataset.condition_count == 1
    assert dataset.minimum_condition_count == 2000
    assert dataset.minimum_gate_passed is False
    assert len(dataset.samples) == 16
    assert sum(sample.endpoint_weight for sample in dataset.samples) == 1.0
    assert all(sample.target_up is True for sample in dataset.samples)
    assert dataset.trading_authority is False
    with pytest.raises(ValueError, match="minimum condition gate"):
        require_round25_dataset_minimum(dataset)


def test_dataset_is_deterministic_and_endpoint_choice_does_not_use_target() -> None:
    authority = _authority()
    rows = _condition_rows(endpoints_per_phase=5)
    up = build_round25_development_dataset(
        role="train",
        snapshots=rows,
        resolutions=(_resolution(authority, winning_token_id=UP_TOKEN),),
        resolution_authority=authority,
    )
    down = build_round25_development_dataset(
        role="train",
        snapshots=tuple(reversed(rows)),
        resolutions=(_resolution(authority, winning_token_id=DOWN_TOKEN),),
        resolution_authority=authority,
    )

    assert [sample.decision_time_ms for sample in up.samples] == [
        sample.decision_time_ms for sample in down.samples
    ]
    assert all(sample.target_up is True for sample in up.samples)
    assert all(sample.target_up is False for sample in down.samples)
    assert up.dataset_sha256 != down.dataset_sha256


def test_dataset_rejects_sealed_role_cross_role_and_population_mismatch() -> None:
    authority = _authority()
    rows = _condition_rows()
    resolution = _resolution(authority)

    with pytest.raises(ValueError, match="invalid or sealed"):
        build_round25_development_dataset(
            role="test",
            snapshots=rows,
            resolutions=(resolution,),
            resolution_authority=authority,
        )
    with pytest.raises(ValueError, match="feature role"):
        build_round25_development_dataset(
            role="calibration",
            snapshots=rows,
            resolutions=(resolution,),
            resolution_authority=authority,
        )
    with pytest.raises(ValueError, match="populations differ"):
        build_round25_development_dataset(
            role="train",
            snapshots=rows,
            resolutions=(),
            resolution_authority=authority,
        )


def test_sample_and_dataset_hash_tampering_is_rejected() -> None:
    authority = _authority()
    dataset = build_round25_development_dataset(
        role="train",
        snapshots=_condition_rows(),
        resolutions=(_resolution(authority),),
        resolution_authority=authority,
    )

    with pytest.raises(ValueError, match="sample"):
        replace(dataset.samples[0], target_up=False)
    with pytest.raises(ValueError, match="dataset"):
        replace(dataset, dataset_sha256="f" * 64)
