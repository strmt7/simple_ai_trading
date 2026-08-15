"""Prospective receipt-time validation of the Round 23 lead-lag mechanism."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
    Round21CoreFeatureSnapshot,
)
from .polymarket_round21_corpus import POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21CausalFeatureRow,
)
from .polymarket_round21_sidecar_replay import (
    Round21SidecarReplay,
    replay_round21_optional_binance_features,
)
from .polymarket_round21_sidecar_terminal import (
    validate_round21_sidecar_terminal_manifest,
)
from .polymarket_round23_lead_lag import (
    _Partition,
    _condition_metric,
    _fit_ridge,
    _logit,
    _metrics,
    _rounded,
    _select_penalty,
    _select_scale,
)
from .polymarket_round25_materialization import (
    materialize_round25_round24_core,
    round24_role_for_event_start,
)
from .polymarket_round25_terminal import (
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND24_SPEC_V1_RELATIVE = (
    "docs/model-research/polymarket/round-024-prospective-receipt-lead-lag-spec-v1.json"
)
POLYMARKET_ROUND24_SPEC_V1_SHA256 = (
    "80fca0a6381569b813462873bfa506533f4b547a992bd3717ae176cc3821e65d"
)
POLYMARKET_ROUND24_SPEC_V2_RELATIVE = (
    "docs/model-research/polymarket/round-024-prospective-receipt-lead-lag-spec-v2.json"
)
POLYMARKET_ROUND24_SPEC_V2_SHA256 = (
    "d3923dff38211e07f1b9e65776000676291dd3d9656fd898f3bee66c1fce8f64"
)
POLYMARKET_ROUND24_SPEC_RELATIVE = (
    "docs/model-research/polymarket/round-024-prospective-receipt-lead-lag-spec-v3.json"
)
POLYMARKET_ROUND24_SPEC_SHA256 = (
    "a45dc61c44dd77eefd594609973446ffda81576cd4c9abf05a4e28ab24c7b53b"
)
POLYMARKET_ROUND24_RESULT_SCHEMA_VERSION = (
    "polymarket-round24-prospective-receipt-lead-lag-result-v1"
)
_ROUND23_RESULT_RELATIVE = (
    "docs/model-research/polymarket/round-023-lead-lag-results-v1.json"
)
_ROUND23_RESULT_SHA256 = (
    "c3385d4894ca430a91442d1023bf01f61c650f6d208a4d0f89d11007ee5a11c0"
)
_PUBLICATION_V1_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-024-preregistration-publication-2026-08-03.json"
)
_PUBLICATION_V1_SHA256 = (
    "36554e1a8888c2f994205102472e2ea8c9bce2f9f8618f7a04c7522bebe47b5d"
)
_PUBLICATION_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-024-preregistration-publication-v2-2026-08-10.json"
)
_PUBLICATION_SHA256 = "00b7744d3aa92d919c94d5ac4f6fd713bfb3f7c24a2f983c5f39d54aabe49710"
_ROUND25_PLAN_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-025-twap-core-campaign-plan-publication-2026-08-10.json"
)
_ROUND25_PLAN_SHA256 = (
    "95820964e0692d66de2df3b624d8987f8ce9de699ad0b5fa16a7ccee53627bc2"
)
_ROUND25_DESIGN_SHA256 = (
    "b5c130622514b2b82855f0e1cc011b29a81bf0583e6bd37fa3e4d4b702d6a113"
)
_ROUND25_RESOLUTION_SOURCE = "https://data.chain.link/streams/btc-usd-twap-30s-streams"
_MAXIMUM_ARTIFACT_BYTES = 2 * 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 24 JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 24 JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ValueError(f"Round 24 {name} is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 24 {name} is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Round 24 {name} is not an object")
    return dict(decoded)


def _verified_self_hash(
    value: Mapping[str, object],
    *,
    field: str,
    expected: str,
) -> dict[str, object]:
    selected = dict(value)
    claimed = str(selected.pop(field, "")).strip().lower()
    if claimed != expected or claimed != _canonical_sha256(selected):
        raise ValueError("Round 24 artifact hash differs")
    return {**selected, field: claimed}


def load_round24_receipt_lead_lag_spec(repository: str | Path) -> dict[str, object]:
    root = Path(repository).resolve()
    v1 = _verified_self_hash(
        _load_object(root / POLYMARKET_ROUND24_SPEC_V1_RELATIVE, name="v1 spec"),
        field="specification_sha256",
        expected=POLYMARKET_ROUND24_SPEC_V1_SHA256,
    )
    v2 = _verified_self_hash(
        _load_object(root / POLYMARKET_ROUND24_SPEC_V2_RELATIVE, name="v2 spec"),
        field="specification_sha256",
        expected=POLYMARKET_ROUND24_SPEC_V2_SHA256,
    )
    v3 = _verified_self_hash(
        _load_object(root / POLYMARKET_ROUND24_SPEC_RELATIVE, name="v3 spec"),
        field="specification_sha256",
        expected=POLYMARKET_ROUND24_SPEC_SHA256,
    )
    parents = v1.get("parents")
    changes = v2.get("changes")
    authority = v1.get("authority")
    v3_data = v3.get("data")
    v3_partitions = v3.get("partitions")
    v3_policy = v3.get("partition_policy")
    invariance = v3.get("model_and_gate_invariance")
    knowledge = v3.get("knowledge_at_freeze")
    expected_data = {
        "campaign_end_ms": 1788046800000,
        "campaign_start_ms": 1786406400000,
        "core_campaign_design_sha256": _ROUND25_DESIGN_SHA256,
        "core_campaign_plan_sha256": _ROUND25_PLAN_SHA256,
        "core_capture_repository_commit_oid": (
            "6ed30d73d4732673319fa0fd5bf68ad60ee8189e"
        ),
        "fresh_condition_start_minimum_ms": 1786406400000,
        "future_target_horizon_ms": 1_000,
        "legacy_core_campaign_plan_sha256": (
            "2c1d87577de566bd4934c9678bcbded5bf156b671a413b83fa6d463372db1d71"
        ),
        "legacy_core_data_reused": False,
        "required_decision_alignment_ms": 1_000,
        "resolution_source": _ROUND25_RESOLUTION_SOURCE,
        "sidecar_campaign_plan_sha256": (
            "a0a0508c5e1b5c0d55d66cf98186b0e815e674d9a95c4318f3303337378fa518"
        ),
        "source": (
            "terminal_integrity_qualified_round25_twap_core_and_independent_"
            "round21_public_binance_sidecar"
        ),
        "target": (
            "future_normalized_market_prior_up_logit_minus_current_normalized_"
            "market_prior_up_logit"
        ),
        "target_source": (
            "future_polymarket_feature_row_only_without_official_resolution"
        ),
    }
    expected_partitions = [
        {"role": "train", "start_ms": 1786406400000, "end_ms": 1787441400000},
        {
            "role": "tune_calibration",
            "start_ms": 1787443200000,
            "end_ms": 1787745600000,
        },
        {
            "role": "tune_selection",
            "start_ms": 1787745600000,
            "end_ms": 1788046800000,
        },
    ]
    if (
        v1.get("status") != "frozen_before_future_condition_start_or_data_access"
        or v2.get("status")
        != "frozen_before_future_condition_start_or_data_access_superseding_v1_compute_strategy"
        or v2.get("parent_specification_sha256") != POLYMARKET_ROUND24_SPEC_V1_SHA256
        or v3.get("status")
        != "frozen_before_successor_campaign_start_superseding_failed_core_source_lineage"
        or v3.get("parent_specification_sha256") != POLYMARKET_ROUND24_SPEC_V2_SHA256
        or v2.get("authority") != authority
        or v3.get("authority") != authority
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
        or not isinstance(parents, Mapping)
        or not isinstance(changes, Mapping)
        or v3_data != expected_data
        or v3_partitions != expected_partitions
        or v3_policy
        != {
            "condition_duration_ms": 300_000,
            "future_target_horizon_ms": 1_000,
            "purge_between_train_and_calibration_ms": 1_800_000,
            "role_assignment": (
                "whole_condition_by_event_start_in_explicit_closed_open_intervals"
            ),
            "source_regime_may_cross_roles": False,
        }
        or not isinstance(invariance, Mapping)
        or invariance.get("model_sha256") != _canonical_sha256(v1.get("model"))
        or invariance.get("evaluation_sha256")
        != _canonical_sha256(v1.get("evaluation"))
        or invariance.get("baseline_features_sha256")
        != _canonical_sha256(v1["model"]["baseline_features"])
        or invariance.get("candidate_features_sha256")
        != _canonical_sha256(v1.get("candidate_features"))
        or invariance.get("minimum_population_sha256")
        != _canonical_sha256(v1.get("minimum_population"))
        or invariance.get("round24_v2_compute_strategy_unchanged") is not True
        or not isinstance(knowledge, Mapping)
        or knowledge.get("new_core_capture_started") is not False
        or knowledge.get("new_core_conditions_accessed") is not False
        or knowledge.get("round24_targets_constructed") is not False
        or knowledge.get("performance_metrics_computed") is not False
        or parents.get("round21_core_corpus_design_sha256")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or parents.get("round21_dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or parents.get("round21_feature_policy_sha256")
        != POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
        or parents.get("round21_feature_schema_sha256")
        != POLYMARKET_ROUND21_FEATURE_SCHEMA.schema_sha256
        or parents.get("round23_result_sha256") != _ROUND23_RESULT_SHA256
        or changes.get("condition_fold_count") != 8
        or changes.get("bootstrap_batch_draws_maximum") != 5_000
    ):
        raise ValueError("Round 24 prospective specification differs")
    round23 = _load_object(root / _ROUND23_RESULT_RELATIVE, name="Round 23 result")
    if (
        str(round23.pop("result_sha256", "")) != _ROUND23_RESULT_SHA256
        or _canonical_sha256(round23) != _ROUND23_RESULT_SHA256
        or round23.get("mechanism_gate_passed") is not True
    ):
        raise ValueError("Round 24 parent Round 23 result differs")
    publication_v1 = _verified_self_hash(
        _load_object(root / _PUBLICATION_V1_RELATIVE, name="v1 publication"),
        field="publication_sha256",
        expected=_PUBLICATION_V1_SHA256,
    )
    v1_data = v1.get("data")
    if (
        not isinstance(v1_data, Mapping)
        or publication_v1.get("effective_specification_sha256")
        != POLYMARKET_ROUND24_SPEC_V2_SHA256
        or publication_v1.get("first_eligible_condition_start_ms")
        != v1_data.get("fresh_condition_start_minimum_ms")
        or publication_v1.get("status") != "published_before_first_eligible_condition"
    ):
        raise ValueError("Round 24 v1 preregistration publication differs")
    plan_path = root / _ROUND25_PLAN_RELATIVE
    plan = _load_object(plan_path, name="Round 25 plan publication")
    plan_claim = str(plan.pop("plan_sha256", "")).lower()
    if (
        plan_claim != _ROUND25_PLAN_SHA256
        or plan_claim != _canonical_sha256(plan)
        or plan.get("design_sha256") != _ROUND25_DESIGN_SHA256
        or plan.get("resolution_source") != _ROUND25_RESOLUTION_SOURCE
        or plan.get("scheduled_start_ms") != expected_data["campaign_start_ms"]
        or plan.get("scheduled_end_ms") != expected_data["campaign_end_ms"]
        or plan.get("repository_commit_oid")
        != expected_data["core_capture_repository_commit_oid"]
        or plan.get("source_qualification_sha256")
        != "53e8a8a22c94fef83944a29d82229ae69d0688275d953133a19f30b69dc2487b"
        or any(
            plan.get(field) is not False
            for field in (
                "binance_captured",
                "credentials_used",
                "outcomes_consulted",
                "model_scores_consulted",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 24 Round 25 plan publication differs")
    publication = _verified_self_hash(
        _load_object(root / _PUBLICATION_RELATIVE, name="publication"),
        field="publication_sha256",
        expected=_PUBLICATION_SHA256,
    )
    if (
        publication.get("effective_specification_sha256")
        != POLYMARKET_ROUND24_SPEC_SHA256
        or publication.get("superseded_specification_sha256")
        != POLYMARKET_ROUND24_SPEC_V2_SHA256
        or publication.get("superseded_publication_sha256") != _PUBLICATION_V1_SHA256
        or publication.get("core_campaign_plan_sha256") != _ROUND25_PLAN_SHA256
        or publication.get("core_campaign_design_sha256") != _ROUND25_DESIGN_SHA256
        or publication.get("first_eligible_condition_start_ms")
        != expected_data["fresh_condition_start_minimum_ms"]
        or publication.get("frozen_at_ms") != 1786351110280
        or publication.get("frozen_at_utc") != "2026-08-10T08:38:30.2743215Z"
        or int(publication["frozen_at_ms"])
        >= int(publication["first_eligible_condition_start_ms"])
        or publication.get("new_core_capture_started") is not False
        or publication.get("new_core_data_accessed") is not False
        or publication.get("round24_target_constructed") is not False
        or publication.get("performance_metric_computed") is not False
        or publication.get("status")
        != "published_before_successor_campaign_start_after_legacy_source_regime_failure"
    ):
        raise ValueError("Round 24 preregistration publication differs")
    return {
        **v1,
        "data": dict(v3_data),
        "knowledge_at_freeze": dict(knowledge),
        "partitions": list(v3_partitions),
        "partition_policy": dict(v3_policy),
        "compute_strategy": dict(changes),
        "specification_sha256": POLYMARKET_ROUND24_SPEC_SHA256,
        "superseded_specification_sha256": POLYMARKET_ROUND24_SPEC_V2_SHA256,
        "source_regime_specification_sha256": POLYMARKET_ROUND24_SPEC_SHA256,
        "core_campaign_plan_sha256": _ROUND25_PLAN_SHA256,
        "publication_sha256": _PUBLICATION_SHA256,
    }


def _core_rows(
    snapshots: Sequence[Round21CoreFeatureSnapshot],
) -> tuple[Round21CausalFeatureRow, ...]:
    rows: list[Round21CausalFeatureRow] = []
    keys: set[tuple[str, int]] = set()
    for snapshot in snapshots:
        if not snapshot.available or snapshot.trading_authority:
            raise ValueError("Round 24 core snapshot population differs")
        key = (snapshot.condition_id, snapshot.decision_time_ms)
        if key in keys:
            raise ValueError("Round 24 core snapshot population is duplicated")
        keys.add(key)
        rows.append(
            Round21CausalFeatureRow.create(
                condition_id=snapshot.condition_id,
                event_start_ms=snapshot.event_start_ms,
                decision_time_ms=snapshot.decision_time_ms,
                structural_probability=snapshot.structural_probability,
                market_prior_probability=snapshot.market_prior_probability,
                core_values=snapshot.values,
                spot_values=(0.0,) * len(POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names),
                usdm_values=(0.0,) * len(POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names),
                spot_available=False,
                usdm_available=False,
                feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
                core_source_chain_sha256=snapshot.source_chain_sha256,
                spot_source_chain_sha256=_EMPTY_SHA256,
                usdm_source_chain_sha256=_EMPTY_SHA256,
                core_maximum_receipt_ms=snapshot.maximum_receipt_ms,
            )
        )
    return tuple(
        sorted(rows, key=lambda item: (item.event_start_ms, item.decision_time_ms))
    )


def _join_optional(
    core_rows: Sequence[Round21CausalFeatureRow],
    replay: Round21SidecarReplay,
) -> tuple[Round21CausalFeatureRow, ...]:
    rows = tuple(core_rows)
    selected = replay.validated()
    if (
        len(rows) != len(selected.features)
        or tuple(row.decision_time_ms for row in rows) != selected.decision_times_ms
    ):
        raise ValueError("Round 24 sidecar replay population differs")
    output: list[Round21CausalFeatureRow] = []
    for row, optional in zip(rows, selected.features, strict=True):
        output.append(
            Round21CausalFeatureRow.create(
                condition_id=row.condition_id,
                event_start_ms=row.event_start_ms,
                decision_time_ms=row.decision_time_ms,
                structural_probability=row.structural_probability,
                market_prior_probability=row.market_prior_probability,
                core_values=row.core_values,
                spot_values=optional.spot_values,
                usdm_values=optional.usdm_values,
                spot_available=optional.spot_available,
                usdm_available=optional.usdm_available,
                feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
                core_source_chain_sha256=row.core_source_chain_sha256,
                spot_source_chain_sha256=optional.spot_source_chain_sha256,
                usdm_source_chain_sha256=optional.usdm_source_chain_sha256,
                core_maximum_receipt_ms=row.core_maximum_receipt_ms,
                spot_maximum_receipt_ms=optional.spot_maximum_receipt_ms,
                usdm_maximum_receipt_ms=optional.usdm_maximum_receipt_ms,
            )
        )
    return tuple(output)


def assemble_round24_receipt_rows(
    *,
    repository: str | Path,
    core_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    sidecar_database: str | Path | None,
    sidecar_terminal_manifest: Mapping[str, object] | None,
    observed_at_ms: int | None = None,
) -> tuple[
    dict[str, object],
    tuple[Round21CausalFeatureRow, ...],
    dict[str, object],
]:
    spec = load_round24_receipt_lead_lag_spec(repository)
    data = spec["data"]
    assert isinstance(data, Mapping)
    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    if (
        transport["source_plan_sha256"] != data["core_campaign_plan_sha256"]
        or transport["source_capture_design_sha256"]
        != data["core_campaign_design_sha256"]
        or transport["resolution_source"] != data["resolution_source"]
        or int(transport["campaign_start_ms"]) != int(data["campaign_start_ms"])
        or int(transport["campaign_end_ms"]) != int(data["campaign_end_ms"])
    ):
        raise ValueError("Round 24 terminal evidence boundaries differ")
    snapshots, materialization, receipt_audit = materialize_round25_round24_core(
        database=core_database,
        terminal_transport_manifest=transport,
        partitions=spec["partitions"],
        round24_specification_sha256=str(spec["specification_sha256"]),
        observed_at_ms=observed_at_ms,
    )
    fresh_start = int(data["fresh_condition_start_minimum_ms"])
    core = tuple(
        row
        for row in _core_rows(snapshots)
        if row.event_start_ms >= fresh_start
        and round24_role_for_event_start(row.event_start_ms, spec["partitions"])
        in {"train", "tune_calibration", "tune_selection"}
    )
    if not core:
        raise ValueError("Round 24 fresh core population is unavailable")
    if (sidecar_database is None) is not (sidecar_terminal_manifest is None):
        raise ValueError("Round 24 sidecar inputs must be supplied together")
    if sidecar_database is None or sidecar_terminal_manifest is None:
        return (
            spec,
            core,
            {
                "core_materialization": materialization,
                "terminal_receipt_audit": receipt_audit,
                "optional_sidecar_joined": False,
                "row_count": len(core),
            },
        )
    sidecar_terminal = validate_round21_sidecar_terminal_manifest(
        sidecar_terminal_manifest
    )
    if (
        int(sidecar_terminal["campaign_start_ms"]) != int(data["campaign_start_ms"])
        or int(sidecar_terminal["campaign_end_ms"]) != int(data["campaign_end_ms"])
        or sidecar_terminal["source_plan_sha256"]
        != data["sidecar_campaign_plan_sha256"]
    ):
        raise ValueError("Round 24 independent sidecar boundary differs")
    replay = replay_round21_optional_binance_features(
        source_database=sidecar_database,
        terminal_manifest=sidecar_terminal,
        decision_times_ms=tuple(row.decision_time_ms for row in core),
    )
    joined = _join_optional(core, replay)
    return (
        spec,
        joined,
        {
            "core_materialization": materialization,
            "terminal_receipt_audit": receipt_audit,
            "optional_sidecar_joined": True,
            "sidecar_terminal_manifest_sha256": sidecar_terminal["manifest_sha256"],
            "sidecar_receipt_chain_sha256": replay.receipt_chain_sha256,
            "row_count": len(joined),
        },
    )


def _depth_imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return 0.0 if total <= 0.0 else (bid - ask) / total


def _vectors(
    row: Round21CausalFeatureRow,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    core_index = {
        name: index
        for index, name in enumerate(POLYMARKET_ROUND21_FEATURE_SCHEMA.core_names)
    }
    spot_index = {
        name: index
        for index, name in enumerate(POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names)
    }
    usdm_index = {
        name: index
        for index, name in enumerate(POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names)
    }

    def core(name: str) -> float:
        return float(row.core_values[core_index[name]])

    def spot(name: str) -> float:
        return float(row.spot_values[spot_index[name]])

    def usdm(name: str) -> float:
        return float(row.usdm_values[usdm_index[name]])

    up_depth = _depth_imbalance(
        core("core.up_bid_depth_5"), core("core.up_ask_depth_5")
    )
    down_depth = _depth_imbalance(
        core("core.down_bid_depth_5"),
        core("core.down_ask_depth_5"),
    )
    baseline = (
        _logit(row.market_prior_probability, lower=0.005, upper=0.995),
        core("core.up_spread"),
        core("core.down_spread"),
        up_depth,
        down_depth,
        core("core.up_microprice") + core("core.down_microprice") - 1.0,
        core("core.complement_buy_overround"),
        core("core.complement_sell_underround"),
        abs(core("core.book_receipt_skew_ms")),
        core("core.elapsed_fraction"),
        core("core.up_microprice_log_return_1000ms"),
        core("core.down_microprice_log_return_1000ms"),
        core("core.up_microprice_log_return_5000ms"),
        core("core.down_microprice_log_return_5000ms"),
        core("core.up_microprice_log_return_15000ms"),
        core("core.down_microprice_log_return_15000ms"),
    )
    candidate_additions = (
        *(
            10_000.0 * spot(f"spot.trade_log_return_{window}ms")
            for window in (1_000, 5_000, 15_000)
        ),
        *(
            10_000.0 * usdm(f"usdm.trade_log_return_{window}ms")
            for window in (1_000, 5_000, 15_000)
        ),
        *(
            spot(f"spot.trade_signed_quote_imbalance_{window}ms")
            for window in (1_000, 5_000, 15_000)
        ),
        *(
            usdm(f"usdm.trade_signed_quote_imbalance_{window}ms")
            for window in (1_000, 5_000, 15_000)
        ),
        10_000.0 * usdm("usdm.log_mid_basis"),
        -10_000.0 * usdm("usdm.spot_minus_usdm_mid_log_return_5000ms"),
        spot("spot.book_receipt_age_ms"),
        usdm("usdm.book_receipt_age_ms"),
        usdm("usdm.spot_minus_usdm_book_receipt_skew_ms"),
    )
    if (
        len(baseline) != 16
        or len(candidate_additions) != 17
        or any(not math.isfinite(value) for value in (*baseline, *candidate_additions))
    ):
        raise ValueError("Round 24 feature vector differs")
    return baseline, (*baseline, *candidate_additions)


def _partitions(
    rows: Sequence[Round21CausalFeatureRow],
    spec: Mapping[str, object],
    *,
    roles: frozenset[str],
) -> tuple[dict[str, _Partition], dict[str, object]]:
    intervals = {
        str(item["role"]): (int(item["start_ms"]), int(item["end_ms"]))
        for item in spec["partitions"]
        if isinstance(item, Mapping)
    }
    by_key = {(row.condition_id, row.decision_time_ms): row for row in rows}
    grouped: dict[str, dict[str, list[object]]] = {}
    chain = "0" * 64
    for row in rows:
        role = next(
            (
                name
                for name, (start, end) in intervals.items()
                if start <= row.event_start_ms < end
            ),
            None,
        )
        if (
            role not in roles
            or row.decision_time_ms % 1_000
            or not row.spot_available
            or not row.usdm_available
        ):
            continue
        future = by_key.get((row.condition_id, row.decision_time_ms + 1_000))
        if future is None:
            continue
        baseline, candidate = _vectors(row)
        target = (
            _logit(
                future.market_prior_probability,
                lower=0.005,
                upper=0.995,
            )
            - baseline[0]
        )
        bucket = grouped.setdefault(
            role,
            {"baseline": [], "candidate": [], "conditions": [], "target": []},
        )
        body = {
            "baseline": baseline,
            "candidate": candidate[16:],
            "condition_id": row.condition_id,
            "decision_time_ms": row.decision_time_ms,
            "role": role,
            "target": target,
        }
        chain = hashlib.sha256(
            bytes.fromhex(chain) + bytes.fromhex(_canonical_sha256(body))
        ).hexdigest()
        bucket["baseline"].append(baseline)
        bucket["candidate"].append(candidate)
        bucket["conditions"].append(row.condition_id)
        bucket["target"].append(target)
    minimum = spec["minimum_population"]
    assert isinstance(minimum, Mapping)
    output: dict[str, _Partition] = {}
    counts: dict[str, object] = {}
    for role in sorted(roles):
        bucket = grouped.get(role)
        if bucket is None:
            raise ValueError("Round 24 partition is unavailable")
        partition = _Partition(
            role=role,
            baseline=np.asarray(bucket["baseline"], dtype=np.float64),
            candidate=np.asarray(bucket["candidate"], dtype=np.float64),
            target=np.asarray(bucket["target"], dtype=np.float64),
            conditions=np.asarray(bucket["conditions"], dtype=np.str_),
        )
        condition_count = int(np.unique(partition.conditions).size)
        if (
            partition.baseline.ndim != 2
            or partition.baseline.shape[1] != 16
            or partition.candidate.shape[1] != 33
            or condition_count < int(minimum[f"{role}_conditions"])
            or not np.all(np.isfinite(partition.candidate))
            or not np.all(np.isfinite(partition.target))
        ):
            raise ValueError("Round 24 partition population differs")
        output[role] = partition
        counts[role] = {
            "condition_count": condition_count,
            "row_count": int(partition.target.size),
            "target_changed_row_count": int(np.sum(np.abs(partition.target) >= 1e-6)),
        }
    return output, {
        "dataset_chain_sha256": chain,
        "official_resolution_accessed": False,
        "partitions": counts,
        "receipt_time_causal": True,
    }


def _model_payload(model: object) -> dict[str, object]:
    return {
        "coefficients": [float(value) for value in model.coefficients],
        "mean": [float(value) for value in model.mean],
        "penalty": float(model.penalty),
        "scale": [float(value) for value in model.scale],
    }


def _fold(condition: str, count: int) -> int:
    return int(hashlib.sha256(condition.encode("ascii")).hexdigest()[:16], 16) % count


def _group_cv_scores(
    partition: _Partition,
    features: NDArray[np.float64],
    penalties: Sequence[float],
    *,
    fold_count: int,
) -> dict[float, float]:
    row_folds = np.asarray(
        [_fold(str(condition), fold_count) for condition in partition.conditions],
        dtype=np.int64,
    )
    if set(np.unique(row_folds)) != set(range(fold_count)):
        raise ValueError("Round 24 condition folds are incomplete")
    scores: dict[float, float] = {}
    for penalty in penalties:
        predictions = np.zeros_like(partition.target)
        for fold in range(fold_count):
            held_out = row_folds == fold
            model = _fit_ridge(
                features[~held_out],
                partition.target[~held_out],
                partition.conditions[~held_out],
                penalty=penalty,
            )
            predictions[held_out] = model.predict(features[held_out])
        scores[float(penalty)] = _condition_metric(
            (partition.target - predictions) ** 2,
            partition.conditions,
        )
    return scores


def _bootstrap_condition_means(
    values: NDArray[np.float64],
    *,
    draws: int,
    batch_maximum: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, batch_maximum):
        count = min(batch_maximum, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        output[start : start + count] = np.mean(values[indices], axis=1)
    return output


def run_round24_receipt_lead_lag(
    *,
    spec: Mapping[str, object],
    rows: Sequence[Round21CausalFeatureRow],
    claim_selection: Callable[[Mapping[str, object]], str],
) -> dict[str, object]:
    development, development_dataset = _partitions(
        rows,
        spec,
        roles=frozenset(("train", "tune_calibration")),
    )
    model_spec = spec["model"]
    evaluation = spec["evaluation"]
    compute = spec["compute_strategy"]
    assert isinstance(model_spec, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(compute, Mapping)
    train = development["train"]
    calibration = development["tune_calibration"]
    penalties = tuple(float(value) for value in model_spec["l2_penalty_grid"])
    folds = int(compute["condition_fold_count"])
    baseline_cv = _group_cv_scores(
        train,
        train.baseline,
        penalties,
        fold_count=folds,
    )
    candidate_cv = _group_cv_scores(
        train,
        train.candidate,
        penalties,
        fold_count=folds,
    )
    baseline_penalty = _select_penalty(baseline_cv)
    candidate_penalty = _select_penalty(candidate_cv)
    baseline_model = _fit_ridge(
        train.baseline,
        train.target,
        train.conditions,
        penalty=baseline_penalty,
    )
    candidate_model = _fit_ridge(
        train.candidate,
        train.target,
        train.conditions,
        penalty=candidate_penalty,
    )
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    baseline_scale, baseline_scale_scores = _select_scale(
        calibration,
        baseline_model.predict(calibration.baseline),
        scales,
    )
    candidate_scale, candidate_scale_scores = _select_scale(
        calibration,
        candidate_model.predict(calibration.candidate),
        scales,
    )
    changed = float(evaluation["changed_target_absolute_logit_minimum"])
    calibration_baseline = baseline_model.predict(calibration.baseline) * baseline_scale
    calibration_candidate = (
        candidate_model.predict(calibration.candidate) * candidate_scale
    )
    cal_base_metrics = _metrics(
        calibration, calibration_baseline, changed_minimum=changed
    )
    cal_cand_metrics = _metrics(
        calibration, calibration_candidate, changed_minimum=changed
    )
    claim_body: dict[str, object] = {
        "authority": dict(spec["authority"]),
        "baseline_group_cv_mse": {
            str(key): _rounded(value) for key, value in baseline_cv.items()
        },
        "baseline_model": _model_payload(baseline_model),
        "baseline_scale": baseline_scale,
        "baseline_scale_mse": {
            str(key): _rounded(value) for key, value in baseline_scale_scores.items()
        },
        "candidate_group_cv_mse": {
            str(key): _rounded(value) for key, value in candidate_cv.items()
        },
        "candidate_model": _model_payload(candidate_model),
        "candidate_scale": candidate_scale,
        "candidate_scale_mse": {
            str(key): _rounded(value) for key, value in candidate_scale_scores.items()
        },
        "development_dataset": development_dataset,
        "schema_version": "polymarket-round24-selection-access-claim-v1",
        "specification_sha256": spec["specification_sha256"],
        "status": "models_and_calibration_frozen_before_selection_target_construction",
    }
    claim_body["claim_sha256"] = _canonical_sha256(claim_body)
    claim_sha256 = claim_selection(claim_body)
    if claim_sha256 != claim_body["claim_sha256"]:
        raise ValueError("Round 24 selection claim writer differs")
    selection_partitions, selection_dataset = _partitions(
        rows,
        spec,
        roles=frozenset(("tune_selection",)),
    )
    selection = selection_partitions["tune_selection"]
    selection_baseline = baseline_model.predict(selection.baseline) * baseline_scale
    selection_candidate = candidate_model.predict(selection.candidate) * candidate_scale
    sel_base_metrics = _metrics(selection, selection_baseline, changed_minimum=changed)
    sel_cand_metrics = _metrics(selection, selection_candidate, changed_minimum=changed)
    conditions = np.unique(selection.conditions)
    improvements = np.asarray(
        [
            np.mean(
                (
                    selection.target[selection.conditions == condition]
                    - selection_baseline[selection.conditions == condition]
                )
                ** 2
            )
            - np.mean(
                (
                    selection.target[selection.conditions == condition]
                    - selection_candidate[selection.conditions == condition]
                )
                ** 2
            )
            for condition in conditions
        ],
        dtype=np.float64,
    )
    bootstrap = _bootstrap_condition_means(
        improvements,
        draws=int(evaluation["bootstrap_draws"]),
        batch_maximum=int(compute["bootstrap_batch_draws_maximum"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    probability = float(np.mean(bootstrap > 0.0))
    cal_improvement = (
        cal_base_metrics["condition_equal_mse"]
        - cal_cand_metrics["condition_equal_mse"]
    )
    sel_improvement = (
        sel_base_metrics["condition_equal_mse"]
        - sel_cand_metrics["condition_equal_mse"]
    )
    relative_improvement = sel_improvement / sel_base_metrics["condition_equal_mse"]
    sign_improvement = (
        sel_cand_metrics["changed_row_direction_accuracy"]
        - sel_base_metrics["changed_row_direction_accuracy"]
    )
    positive_fraction = float(np.mean(improvements > 0.0))
    gate = evaluation["mechanism_gate"]
    assert isinstance(gate, Mapping)
    passed = bool(
        min(candidate_cv.values()) < min(baseline_cv.values())
        and cal_improvement > 0.0
        and sel_improvement > 0.0
        and relative_improvement
        >= float(gate["selection_mse_relative_improvement_minimum"])
        and lower > 0.0
        and probability >= float(gate["selection_improvement_probability_minimum"])
        and sel_cand_metrics["changed_row_direction_accuracy"]
        >= float(gate["candidate_changed_row_sign_accuracy_minimum"])
        and sign_improvement
        >= float(gate["candidate_sign_accuracy_improvement_minimum"])
        and positive_fraction >= float(gate["positive_condition_fraction_minimum"])
    )
    result: dict[str, object] = {
        "authority": dict(spec["authority"]),
        "bootstrap": {
            "improvement_probability": _rounded(probability),
            "mean_mse_improvement": _rounded(float(np.mean(bootstrap))),
            "mse_improvement_95_interval": [_rounded(lower), _rounded(upper)],
        },
        "calibration": {
            "baseline_metrics": {
                key: _rounded(value) for key, value in cal_base_metrics.items()
            },
            "baseline_scale": baseline_scale,
            "baseline_scale_mse": {
                str(key): _rounded(value)
                for key, value in baseline_scale_scores.items()
            },
            "candidate_metrics": {
                key: _rounded(value) for key, value in cal_cand_metrics.items()
            },
            "candidate_scale": candidate_scale,
            "candidate_scale_mse": {
                str(key): _rounded(value)
                for key, value in candidate_scale_scores.items()
            },
            "mse_improvement": _rounded(cal_improvement),
        },
        "conclusion": (
            "prospective_receipt_time_lead_lag_mechanism_passed"
            if passed
            else "prospective_receipt_time_lead_lag_mechanism_falsified"
        ),
        "dataset": {
            "development": development_dataset,
            "selection": selection_dataset,
        },
        "mechanism_gate_passed": passed,
        "model": {
            "baseline_group_cv_mse": {
                str(key): _rounded(value) for key, value in baseline_cv.items()
            },
            "baseline_selected_penalty": baseline_penalty,
            "candidate_group_cv_mse": {
                str(key): _rounded(value) for key, value in candidate_cv.items()
            },
            "candidate_selected_penalty": candidate_penalty,
        },
        "parents": {
            "round23_result_sha256": _ROUND23_RESULT_SHA256,
            "selection_access_claim_sha256": claim_sha256,
            "specification_sha256": spec["specification_sha256"],
        },
        "schema_version": POLYMARKET_ROUND24_RESULT_SCHEMA_VERSION,
        "selection": {
            "baseline_metrics": {
                key: _rounded(value) for key, value in sel_base_metrics.items()
            },
            "candidate_metrics": {
                key: _rounded(value) for key, value in sel_cand_metrics.items()
            },
            "changed_row_sign_accuracy_improvement": _rounded(sign_improvement),
            "mse_improvement": _rounded(sel_improvement),
            "mse_relative_improvement": _rounded(relative_improvement),
            "positive_condition_fraction": _rounded(positive_fraction),
            "status": "fresh_prospective_receipt_time_selection",
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "POLYMARKET_ROUND24_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND24_SPEC_RELATIVE",
    "POLYMARKET_ROUND24_SPEC_SHA256",
    "assemble_round24_receipt_rows",
    "load_round24_receipt_lead_lag_spec",
    "run_round24_receipt_lead_lag",
]
