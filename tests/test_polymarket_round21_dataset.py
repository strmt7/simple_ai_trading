from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round21_dataset import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21CausalFeatureRow,
    Round21FeatureSchema,
    Round21OfficialOutcome,
    Round21PartitionPolicy,
    build_round21_development_panel,
    load_round21_dataset_design,
    validate_round21_dataset_design,
)


DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-causal-dataset-design-v1.json"
)
CAMPAIGN_START_MS = 1_800_000_000_000
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _schema() -> Round21FeatureSchema:
    return Round21FeatureSchema.create(
        core_names=("core.structural_distance", "core.up_spread"),
        spot_names=("spot.return_250ms", "spot.trade_count_250ms"),
        usdm_names=("usdm.return_250ms", "usdm.basis"),
        feature_policy_sha256="f" * 64,
    )


def _policy() -> Round21PartitionPolicy:
    return Round21PartitionPolicy.create(
        campaign_start_ms=CAMPAIGN_START_MS,
        campaign_end_ms=CAMPAIGN_START_MS + 30 * 86_400_000,
    )


def _row(
    condition_number: int,
    *,
    label: bool,
    spot_available: bool = True,
    usdm_available: bool = True,
) -> Round21CausalFeatureRow:
    event_start = CAMPAIGN_START_MS + condition_number * 300_000
    decision = event_start + 150_000
    return Round21CausalFeatureRow.create(
        condition_id="0x" + format(condition_number + 1, "064x"),
        event_start_ms=event_start,
        decision_time_ms=decision,
        structural_probability=0.55 if label else 0.45,
        market_prior_probability=0.53 if label else 0.47,
        core_values=(0.1 if label else -0.1, 0.02),
        spot_values=(
            (0.2 if label else -0.2, 4.0)
            if spot_available
            else (0.0, 0.0)
        ),
        usdm_values=(
            (0.15 if label else -0.15, 0.001)
            if usdm_available
            else (0.0, 0.0)
        ),
        spot_available=spot_available,
        usdm_available=usdm_available,
        feature_schema=_schema(),
        core_source_chain_sha256=_sha(f"core-{condition_number}"),
        spot_source_chain_sha256=(
            _sha(f"spot-{condition_number}") if spot_available else EMPTY_SHA256
        ),
        usdm_source_chain_sha256=(
            _sha(f"usdm-{condition_number}") if usdm_available else EMPTY_SHA256
        ),
        core_maximum_receipt_ms=decision,
        spot_maximum_receipt_ms=decision - 1 if spot_available else 0,
        usdm_maximum_receipt_ms=decision - 2 if usdm_available else 0,
    )


def _outcome(
    condition_number: int,
    *,
    label: bool,
) -> Round21OfficialOutcome:
    event_start = CAMPAIGN_START_MS + condition_number * 300_000
    return Round21OfficialOutcome.create(
        condition_id="0x" + format(condition_number + 1, "064x"),
        event_start_ms=event_start,
        resolved_up=label,
        observed_at_ms=event_start + 300_001,
        source="official-polymarket-resolution",
        source_payload_sha256=_sha(f"outcome-{condition_number}"),
    )


def _rehash_design(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("design_sha256", None)
    body["design_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round21_dataset_design_loads_frozen_role_and_optional_join() -> None:
    design = load_round21_dataset_design(DESIGN_PATH)

    assert design["design_sha256"] == POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
    assert design["partition"]["intervals"][1] == {
        "role": "purge_train_to_tune",
        "start_offset_ms": 1_555_200_000,
        "end_offset_ms": 1_557_000_000,
    }
    assert design["feature_rows"]["forward_or_backward_fill"] is False
    assert design["join"]["optional_rows_may_expand_core_population"] is False
    assert design["targets"]["test_target_access_before_one_use_unlock"] is False


def test_round21_dataset_design_rejects_rehashed_causal_or_authority_drift() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    design["feature_rows"]["forward_or_backward_fill"] = True
    with pytest.raises(ValueError, match="dataset design differs"):
        validate_round21_dataset_design(_rehash_design(design))

    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    design["authority"]["paper_trading_authority"] = True
    with pytest.raises(ValueError, match="dataset design differs"):
        validate_round21_dataset_design(_rehash_design(design))


def test_round21_feature_schema_binds_the_feature_policy() -> None:
    schema = _schema()

    assert schema.feature_policy_sha256 == "f" * 64
    with pytest.raises(ValueError, match="feature policy identity"):
        Round21FeatureSchema.create(
            core_names=schema.core_names,
            spot_names=schema.spot_names,
            usdm_names=schema.usdm_names,
            feature_policy_sha256=hashlib.sha256(b"").hexdigest(),
        )
    with pytest.raises(ValueError, match="feature schema differs"):
        replace(schema, feature_policy_sha256="e" * 64).validated()


def test_round21_partition_assigns_purges_and_keeps_test_sealed() -> None:
    policy = _policy()

    assert policy.role_for_event_start(CAMPAIGN_START_MS) == "train"
    assert (
        policy.role_for_event_start(CAMPAIGN_START_MS + 1_555_200_000)
        == "purge_train_to_tune"
    )
    assert (
        policy.role_for_event_start(CAMPAIGN_START_MS + 1_557_000_000)
        == "tune_calibration"
    )
    assert (
        policy.role_for_event_start(CAMPAIGN_START_MS + 1_771_200_000)
        == "tune_selection"
    )
    assert (
        policy.role_for_event_start(CAMPAIGN_START_MS + 1_989_000_000)
        == "test"
    )
    with pytest.raises(ValueError, match="invalid or sealed"):
        build_round21_development_panel(
            role="test",
            feature_schema=_schema(),
            partition_policy=policy,
            feature_rows=(),
            outcomes=(),
        )


def test_round21_development_panel_preserves_exact_optional_missingness() -> None:
    rows = (
        _row(0, label=False, spot_available=False, usdm_available=False),
        _row(1, label=True),
    )
    outcomes = (
        _outcome(0, label=False),
        _outcome(1, label=True),
    )

    panel = build_round21_development_panel(
        role="train",
        feature_schema=_schema(),
        partition_policy=_policy(),
        feature_rows=tuple(reversed(rows)),
        outcomes=tuple(reversed(outcomes)),
    )
    repeated = build_round21_development_panel(
        role="train",
        feature_schema=_schema(),
        partition_policy=_policy(),
        feature_rows=rows,
        outcomes=outcomes,
    )

    assert panel.dataset_sha256 == repeated.dataset_sha256
    assert panel.target_manifest_sha256 == repeated.target_manifest_sha256
    assert panel.spot_available.tolist() == [False, True]
    assert panel.usdm_available.tolist() == [False, True]
    assert panel.spot_features[0].tolist() == [0.0, 0.0]
    assert panel.usdm_features[0].tolist() == [0.0, 0.0]
    assert panel.labels.tolist() == [0.0, 1.0]


def test_round21_feature_row_rejects_future_or_implicit_optional_receipts() -> None:
    row = _row(0, label=False)
    with pytest.raises(ValueError, match="causal feature row is invalid"):
        Round21CausalFeatureRow.create(
            condition_id=row.condition_id,
            event_start_ms=row.event_start_ms,
            decision_time_ms=row.decision_time_ms,
            structural_probability=row.structural_probability,
            market_prior_probability=row.market_prior_probability,
            core_values=row.core_values,
            spot_values=row.spot_values,
            usdm_values=row.usdm_values,
            spot_available=True,
            usdm_available=True,
            feature_schema=_schema(),
            core_source_chain_sha256=row.core_source_chain_sha256,
            spot_source_chain_sha256=row.spot_source_chain_sha256,
            usdm_source_chain_sha256=row.usdm_source_chain_sha256,
            core_maximum_receipt_ms=row.decision_time_ms + 1,
            spot_maximum_receipt_ms=row.spot_maximum_receipt_ms,
            usdm_maximum_receipt_ms=row.usdm_maximum_receipt_ms,
        )
    with pytest.raises(ValueError, match="causal feature row is invalid"):
        Round21CausalFeatureRow.create(
            condition_id=row.condition_id,
            event_start_ms=row.event_start_ms,
            decision_time_ms=row.decision_time_ms,
            structural_probability=row.structural_probability,
            market_prior_probability=row.market_prior_probability,
            core_values=row.core_values,
            spot_values=(1.0, 0.0),
            usdm_values=(0.0, 0.0),
            spot_available=False,
            usdm_available=False,
            feature_schema=_schema(),
            core_source_chain_sha256=row.core_source_chain_sha256,
            core_maximum_receipt_ms=row.core_maximum_receipt_ms,
        )
    with pytest.raises(ValueError, match="causal feature row is invalid"):
        Round21CausalFeatureRow.create(
            condition_id=row.condition_id,
            event_start_ms=row.event_start_ms,
            decision_time_ms=row.decision_time_ms,
            structural_probability=row.structural_probability,
            market_prior_probability=row.market_prior_probability,
            core_values=row.core_values,
            spot_values=row.spot_values,
            usdm_values=row.usdm_values,
            spot_available=1,  # type: ignore[arg-type]
            usdm_available=True,
            feature_schema=_schema(),
            core_source_chain_sha256=row.core_source_chain_sha256,
            spot_source_chain_sha256=row.spot_source_chain_sha256,
            usdm_source_chain_sha256=row.usdm_source_chain_sha256,
            core_maximum_receipt_ms=row.core_maximum_receipt_ms,
            spot_maximum_receipt_ms=row.spot_maximum_receipt_ms,
            usdm_maximum_receipt_ms=row.usdm_maximum_receipt_ms,
        )


def test_round21_development_panel_rejects_duplicate_tampered_or_mixed_roles() -> None:
    first = _row(0, label=False)
    second = _row(1, label=True)
    outcomes = (_outcome(0, label=False), _outcome(1, label=True))
    with pytest.raises(ValueError, match="rows differ"):
        build_round21_development_panel(
            role="train",
            feature_schema=_schema(),
            partition_policy=_policy(),
            feature_rows=(first, first),
            outcomes=(_outcome(0, label=False),),
        )
    with pytest.raises(ValueError, match="feature row differs"):
        build_round21_development_panel(
            role="train",
            feature_schema=_schema(),
            partition_policy=_policy(),
            feature_rows=(replace(first, row_sha256="f" * 64), second),
            outcomes=outcomes,
        )
    mixed_role = _row(5_190, label=False)
    with pytest.raises(ValueError, match="frozen role"):
        build_round21_development_panel(
            role="train",
            feature_schema=_schema(),
            partition_policy=_policy(),
            feature_rows=(first, second, mixed_role),
            outcomes=(*outcomes, _outcome(5_190, label=False)),
        )
