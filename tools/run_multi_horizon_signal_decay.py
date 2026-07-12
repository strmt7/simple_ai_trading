"""Run the hash-bound Round 36 consumed-data signal-decay diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_features import (  # noqa: E402
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
)
from simple_ai_trading.microstructure_signal_decay import (  # noqa: E402
    HorizonPath,
    build_horizon_path,
    chronological_nonoverlapping_mask,
    daily_direction_metrics,
    direction_metrics,
    exact_horizon_rows,
    load_bbo_quotes_asof,
    placebo_summary,
    placebo_weighted_auc_distribution,
    ranked_event_outcomes,
    routed_cost_metrics,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools import run_shared_action_viability as shared_runner  # noqa: E402
from tools.run_three_action_viability import load_round34_design  # noqa: E402


DESIGN_SCHEMA_VERSION = "multi-horizon-signal-decay-design-v1"
BINDING_SCHEMA_VERSION = "round-036-signal-decay-execution-binding-v1"
REPORT_SCHEMA_VERSION = "multi-horizon-signal-decay-report-v1"
_ROUND = 36
_DAY_MS = 86_400_000
_EXPECTED_HORIZONS = (5, 15, 30, 60, 120, 300, 900)
_EXPECTED_SIGNALS = (
    "l1_imbalance",
    "microprice_offset_bps",
    "normalized_ofi",
    "ofi_10s_mean",
    "ofi_60s_mean",
    "ofi_300s_mean",
    "trade_imbalance",
    "trade_imbalance_10s_mean",
    "trade_imbalance_60s_mean",
    "trade_imbalance_300s_mean",
    "signed_pressure_to_opposing_depth_10s",
    "signed_pressure_to_opposing_depth_60s",
    "signed_pressure_to_opposing_depth_300s",
)
_REQUIRED_BOUND_PATHS = frozenset(
    {
        "docs/model-research/action-value/consumed-periods-through-round-033.json",
        "docs/model-research/action-value/consumed-periods-through-round-034.json",
        "docs/model-research/action-value/consumed-periods-through-round-035.json",
        "docs/model-research/action-value/round-031-frozen-chronological-confirmation-design.json",
        "docs/model-research/action-value/round-033-failure-analysis.json",
        "docs/model-research/action-value/round-033-selective-action-design.json",
        "docs/model-research/action-value/round-034-execution-binding.json",
        "docs/model-research/action-value/round-034-failure-analysis.json",
        "docs/model-research/action-value/round-034-three-action-utility-design.json",
        "docs/model-research/action-value/round-035-consumed-direction-screen-design.json",
        "docs/model-research/action-value/round-035-direction-screen-execution-binding.json",
        "docs/model-research/action-value/round-035-failure-analysis.json",
        "docs/model-research/action-value/round-036-multi-horizon-signal-decay-design.json",
        "src/simple_ai_trading/assets.py",
        "src/simple_ai_trading/compute.py",
        "src/simple_ai_trading/lightgbm_backend.py",
        "src/simple_ai_trading/microstructure_action_architecture.py",
        "src/simple_ai_trading/microstructure_action_features.py",
        "src/simple_ai_trading/microstructure_action_policy.py",
        "src/simple_ai_trading/microstructure_architecture.py",
        "src/simple_ai_trading/microstructure_barriers.py",
        "src/simple_ai_trading/microstructure_cache.py",
        "src/simple_ai_trading/microstructure_direction_screen.py",
        "src/simple_ai_trading/microstructure_features.py",
        "src/simple_ai_trading/microstructure_model.py",
        "src/simple_ai_trading/microstructure_outcome_lightgbm.py",
        "src/simple_ai_trading/microstructure_selective_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_selective_action_policy.py",
        "src/simple_ai_trading/microstructure_shared_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_signal_decay.py",
        "src/simple_ai_trading/microstructure_three_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_warehouse.py",
        "src/simple_ai_trading/probability_calibration.py",
        "src/simple_ai_trading/progress_heartbeat.py",
        "src/simple_ai_trading/storage.py",
        "tools/publish_three_action_viability.py",
        "tools/run_consumed_direction_screen.py",
        "tools/run_multi_horizon_signal_decay.py",
        "tools/run_selective_action_viability.py",
        "tools/run_shared_action_viability.py",
        "tools/run_three_action_viability.py",
    }
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_git_oid(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root is invalid")
    return payload


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 36 Git binding command failed") from exc


def _feature_contract_sha256() -> str:
    return _canonical_sha256(
        {
            "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
            "feature_names": MICROSTRUCTURE_FEATURE_NAMES,
        }
    )


def load_signal_decay_design(path: str | Path) -> tuple[dict[str, object], str]:
    """Validate the complete frozen Round 36 diagnostic design."""

    design_path = Path(path).resolve()
    design = _read_object(design_path, label="Round 36 signal-decay design")
    canonical = dict(design)
    claimed = canonical.pop("design_sha256", None)
    if (
        not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or design.get("schema_version") != DESIGN_SCHEMA_VERSION
        or design.get("round") != _ROUND
        or design.get("phase") != "pre_model_consumed_data_diagnostic"
        or design.get("design_revision") != 1
    ):
        raise ValueError("Round 36 signal-decay design hash or identity is invalid")
    predecessor = design.get("predecessor")
    governance = design.get("governance")
    source = design.get("source_contract")
    access = design.get("data_access")
    sampler = design.get("event_sampler")
    future = design.get("future_quote_contract")
    execution = design.get("execution_cost_contract")
    statistics = design.get("statistical_contract")
    regimes = design.get("regime_contract")
    decay = design.get("signal_decay_contract")
    required_report = design.get("required_report")
    decision = design.get("decision_contract")
    resources = design.get("runtime_resources")
    claims = design.get("claims")
    sections = (
        predecessor,
        governance,
        source,
        access,
        sampler,
        future,
        execution,
        statistics,
        regimes,
        decay,
        required_report,
        decision,
        resources,
        claims,
    )
    if any(not isinstance(value, Mapping) for value in sections):
        raise ValueError("Round 36 signal-decay design sections are incomplete")
    research_root = design_path.parent
    failure_path = research_root / str(predecessor["failure_analysis"])
    registry_path = research_root / str(governance["consumed_period_registry"])
    loader_path = research_root / str(source["loader_design"])
    failure = _read_object(failure_path, label="Round 35 failure analysis")
    registry = _read_object(registry_path, label="Round 35 consumed registry")
    loader, loader_sha, _profiles = load_round34_design(loader_path)
    if (
        failure.get("analysis_sha256")
        != predecessor.get("failure_analysis_canonical_sha256")
        or _sha256_file(failure_path) != predecessor.get("failure_analysis_file_sha256")
        or registry.get("registry_sha256")
        != governance.get("consumed_period_registry_canonical_sha256")
        or _sha256_file(registry_path)
        != governance.get("consumed_period_registry_file_sha256")
        or loader_sha != source.get("loader_design_canonical_sha256")
        or _sha256_file(loader_path) != source.get("loader_design_file_sha256")
        or loader.get("round") != 34
    ):
        raise ValueError("Round 36 predecessor, registry, or loader evidence drifted")
    signals = design.get("signals")
    horizons = design.get("horizons_seconds")
    if not isinstance(signals, list) or not isinstance(horizons, list):
        raise ValueError("Round 36 signal or horizon budget is invalid")
    signal_names = tuple(
        str(item.get("name") or "") for item in signals if isinstance(item, Mapping)
    )
    if (
        len(signals) != len(_EXPECTED_SIGNALS)
        or signal_names != _EXPECTED_SIGNALS
        or len(signal_names) != len(set(signal_names))
        or any(name not in MICROSTRUCTURE_FEATURE_NAMES for name in signal_names)
        or any(
            not isinstance(item, Mapping)
            or item.get("positive_orientation") != "higher_future_midquote"
            for item in signals
        )
        or tuple(horizons) != _EXPECTED_HORIZONS
    ):
        raise ValueError("Round 36 frozen signal or horizon contract drifted")
    denied = (
        "untouched_period_access_permitted",
        "model_training_permitted",
        "model_architecture_selection_permitted",
        "signal_sign_reversal_permitted",
        "signal_threshold_search_permitted",
        "signal_combination_weight_search_permitted",
        "horizon_selection_permitted",
        "promotion_permitted",
        "trading_policy_selection_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "oracle_feature_or_runtime_label_use_permitted",
    )
    if (
        governance.get("post_hoc_diagnostic_only") is not True
        or governance.get("all_evaluated_dates_already_consumed") is not True
        or any(governance.get(field) is not False for field in denied)
        or any(value is not False for value in claims.values())
    ):
        raise ValueError("Round 36 governance or claim denial drifted")
    placebo = statistics.get("placebo")
    zero_latency = execution.get("zero_latency_counterfactual")
    if (
        source.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION
        or source.get("feature_count") != len(MICROSTRUCTURE_FEATURE_NAMES)
        or source.get("feature_contract_sha256") != _feature_contract_sha256()
        or source.get("corpus_certificate_sha256")
        != "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
        or source.get("barrier_targets_sha256")
        != "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
        or source.get("cache_key")
        != "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
        or source.get("dataset_rows") != 877_894
        or source.get("event_rows") != 230_941
        or any(
            source.get(field) is not False
            for field in (
                "full_level_two_order_book_claim",
                "queue_position_claim",
                "hidden_liquidity_claim",
            )
        )
        or not isinstance(placebo, Mapping)
        or placebo.get("replicates") != 200
        or placebo.get("seed") != 3601
        or not isinstance(zero_latency, Mapping)
        or zero_latency.get("purpose") != "decompose_historical_latency_drag_only"
        or execution.get("delayed_entry_arrival_ms") != 750
        or execution.get("taker_fee_bps_per_side") != 5.0
        or execution.get("additional_adverse_slippage_bps_per_side") != 1.0
        or statistics.get("ranked_tail_counts") != [100, 500, 1000]
        or statistics.get("orientation_flip_after_observing_results_permitted")
        is not False
        or statistics.get("best_horizon_or_best_signal_promotion_permitted")
        is not False
    ):
        raise ValueError("Round 36 source, execution, or statistical contract drifted")
    if (
        access.get("metric_role") != "calibration"
        or access.get("metric_start") != "2023-06-21"
        or access.get("metric_end") != "2023-06-25"
        or any(
            access.get(field) is not False
            for field in (
                "train_prediction_or_metric_access_permitted",
                "early_stop_prediction_or_metric_access_permitted",
                "policy_prediction_or_metric_access_permitted",
                "development_prediction_or_metric_access_permitted",
                "distant_confirmation_source_materialization_permitted",
                "distant_confirmation_prediction_or_metric_access_permitted",
            )
        )
        or decision.get("this_round_can_create_model_candidate") is not False
        or decision.get("this_round_can_create_trading_authority") is not False
        or resources.get("duplicate_dataset_or_quote_archive_permitted") is not False
        or resources.get("gpu_training_required") is not False
    ):
        raise ValueError("Round 36 access, decision, or resource contract drifted")
    research = design.get("research_basis")
    limitations = design.get("limitations")
    if (
        not isinstance(research, list)
        or len(research) < 8
        or any(
            not isinstance(item, Mapping)
            or not str(item.get("url") or "").startswith("https://")
            or not str(item.get("review_status") or "").strip()
            for item in research
        )
        or not isinstance(limitations, list)
        or len(limitations) < 8
        or not all(value is True for value in required_report.values())
    ):
        raise ValueError("Round 36 research, limitations, or report contract drifted")
    return design, str(claimed)


def load_signal_decay_binding(
    path: str | Path,
    *,
    design_path: str | Path,
    design_sha256: str,
) -> tuple[dict[str, object], str]:
    """Verify the implementation commit, critical blobs, and clean worktree."""

    binding = _read_object(Path(path), label="Round 36 execution binding")
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256", None)
    implementation = binding.get("implementation")
    design = binding.get("design")
    if (
        not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("round") != _ROUND
        or binding.get("worktree_policy") != "clean_including_untracked"
        or not isinstance(implementation, Mapping)
        or not isinstance(design, Mapping)
    ):
        raise ValueError("Round 36 execution binding is invalid")
    commit = str(implementation.get("commit") or "").lower()
    files = implementation.get("files")
    if (
        not _is_git_oid(commit)
        or implementation.get("hash_mode") != "git_blob_sha256_v1"
        or not isinstance(files, list)
    ):
        raise ValueError("Round 36 implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    bound: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("Round 36 bound file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in bound
        ):
            raise ValueError("Round 36 bound path is unsafe or duplicated")
        bound[normalized] = str(item["sha256"])
    if set(bound) != _REQUIRED_BOUND_PATHS:
        missing = sorted(_REQUIRED_BOUND_PATHS - set(bound))
        extra = sorted(set(bound) - _REQUIRED_BOUND_PATHS)
        raise ValueError(
            f"Round 36 bound scope changed: missing={missing} extra={extra}"
        )
    for normalized, expected in bound.items():
        historical = _git_bytes("show", f"{commit}:{normalized}")
        current = _git_bytes("show", f"HEAD:{normalized}")
        if (
            hashlib.sha256(historical).hexdigest() != expected
            or hashlib.sha256(current).hexdigest() != expected
        ):
            raise ValueError(f"Round 36 implementation changed: {normalized}")
    if _git_bytes("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Round 36 execution requires a clean worktree")
    relative_design = Path(design_path).resolve().relative_to(ROOT).as_posix()
    if (
        design.get("path") != relative_design
        or design.get("design_sha256") != design_sha256
        or design.get("file_sha256") != bound.get(relative_design)
    ):
        raise ValueError("Round 36 bound design identity changed")
    return binding, str(claimed)


def _utc_date(day: int) -> str:
    return datetime.fromtimestamp(day * 86_400, tz=timezone.utc).date().isoformat()


def _daily_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [
        float(item["weighted_roc_auc"])
        for item in records
        if item.get("weighted_roc_auc") is not None
    ]
    return {
        "days": len(records),
        "days_with_defined_auc": len(values),
        "days_above_chance": sum(value > 0.5 for value in values),
        "weighted_auc_minimum": min(values) if values else None,
        "weighted_auc_median": float(np.median(values)) if values else None,
        "weighted_auc_standard_deviation": (float(np.std(values)) if values else None),
    }


def _regime_masks(
    path: HorizonPath,
    dataset_features: np.ndarray,
    feature_index: Mapping[str, int],
) -> list[tuple[str, str, np.ndarray]]:
    definitions = (
        (
            "relative_spread",
            "spread_vs_60s_mean",
            (
                ("less_than_0_8", -math.inf, 0.8),
                ("0_8_to_less_than_1_25", 0.8, 1.25),
                ("greater_than_or_equal_to_1_25", 1.25, math.inf),
            ),
        ),
        (
            "relative_short_term_volatility",
            "volatility_10s_vs_300s",
            (
                ("less_than_0_75", -math.inf, 0.75),
                ("0_75_to_less_than_1_5", 0.75, 1.5),
                ("greater_than_or_equal_to_1_5", 1.5, math.inf),
            ),
        ),
        (
            "relative_l1_depth",
            "l1_depth_vs_60s_mean",
            (
                ("less_than_0_75", -math.inf, 0.75),
                ("0_75_to_less_than_1_25", 0.75, 1.25),
                ("greater_than_or_equal_to_1_25", 1.25, math.inf),
            ),
        ),
    )
    output: list[tuple[str, str, np.ndarray]] = []
    for regime, feature, bands in definitions:
        values = np.asarray(
            dataset_features[path.source_indexes, feature_index[feature]],
            dtype=np.float64,
        )
        for label, lower, upper in bands:
            output.append((regime, label, (values >= lower) & (values < upper)))
    return output


def _signal_result(
    *,
    signal_position: int,
    signal_contract: Mapping[str, object],
    path: HorizonPath,
    dataset_features: np.ndarray,
    feature_index: Mapping[str, int],
    requested_tails: Sequence[int],
    placebo_replicates: int,
    placebo_master_seed: int,
    minimum_regime_rows: int,
) -> dict[str, object]:
    name = str(signal_contract["name"])
    values = np.asarray(
        dataset_features[path.source_indexes, feature_index[name]],
        dtype=np.float64,
    )
    nonfinite = int(np.sum(~np.isfinite(values)))
    pooled = direction_metrics(path, values)
    daily = daily_direction_metrics(path, values)
    daily_records = [
        {**item, "utc_date": _utc_date(int(item["utc_day"]))} for item in daily
    ]
    nonoverlap = chronological_nonoverlapping_mask(path)
    seed = int(
        np.random.SeedSequence(
            [placebo_master_seed, path.horizon_seconds, signal_position]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    placebo_values = placebo_weighted_auc_distribution(
        path,
        values,
        replicates=placebo_replicates,
        seed=seed,
    )
    regime_records: list[dict[str, object]] = []
    for regime, band, mask in _regime_masks(
        path,
        dataset_features,
        feature_index,
    ):
        support = int(np.sum(mask))
        record: dict[str, object] = {
            "regime": regime,
            "band": band,
            "support_rows": support,
            "minimum_rows_to_report_metric": minimum_regime_rows,
            "metrics_reported": support >= minimum_regime_rows,
        }
        if support >= minimum_regime_rows:
            record["direction"] = direction_metrics(path, values, row_mask=mask)
            record["cost"] = routed_cost_metrics(path, values, row_mask=mask)
        else:
            record["direction"] = None
            record["cost"] = None
        regime_records.append(record)
    return {
        "signal": name,
        "family": str(signal_contract["family"]),
        "positive_orientation": str(signal_contract["positive_orientation"]),
        "horizon_seconds": path.horizon_seconds,
        "nonfinite_signal_rows": nonfinite,
        "direction": pooled,
        "daily_direction": daily_records,
        "daily_summary": _daily_summary(daily_records),
        "nonoverlapping_robustness": {
            "selected_rows": int(np.sum(nonoverlap)),
            "direction": direction_metrics(path, values, row_mask=nonoverlap),
            "cost": routed_cost_metrics(path, values, row_mask=nonoverlap),
        },
        "cost_decomposition": routed_cost_metrics(path, values),
        "ranked_event_outcomes": ranked_event_outcomes(
            path,
            values,
            requested_counts=requested_tails,
        ),
        "regime_metrics": regime_records,
        "placebo": {
            "derived_seed": seed,
            **placebo_summary(pooled["weighted_roc_auc"], placebo_values),
        },
        "model_candidate": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _half_life(
    signal_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ordered = sorted(signal_results, key=lambda item: int(item["horizon_seconds"]))
    horizons = [int(item["horizon_seconds"]) for item in ordered]
    aucs = [item["direction"]["weighted_roc_auc"] for item in ordered]
    information_coefficients = [
        item["direction"]["spearman_information_coefficient"] for item in ordered
    ]
    output = {
        "horizons_seconds": horizons,
        "weighted_auc": aucs,
        "weighted_auc_minus_chance": [
            float(value) - 0.5 if value is not None else None for value in aucs
        ],
        "spearman_information_coefficient": information_coefficients,
        "half_life_status": "no_measurable_half_life_on_consumed_role",
        "half_life_seconds": None,
        "earliest_peak_horizon_seconds": None,
        "earliest_peak_weighted_auc": None,
    }
    if any(value is None for value in aucs):
        return output
    numeric = np.asarray(aucs, dtype=np.float64)
    peak_position = int(np.argmax(numeric))
    peak_auc = float(numeric[peak_position])
    output["earliest_peak_horizon_seconds"] = horizons[peak_position]
    output["earliest_peak_weighted_auc"] = peak_auc
    daily = ordered[peak_position]["daily_summary"]
    if peak_auc < 0.53 or int(daily["days_above_chance"]) < 4:
        return output
    peak_excess = peak_auc - 0.5
    target = peak_excess / 2.0
    excess = numeric - 0.5
    for position in range(peak_position + 1, len(horizons)):
        segment = excess[peak_position : position + 1]
        if np.any(np.diff(segment) > 0.002):
            return output
        if excess[position] > target:
            continue
        prior_horizon = horizons[position - 1]
        current_horizon = horizons[position]
        prior_value = float(excess[position - 1])
        current_value = float(excess[position])
        if prior_value == current_value:
            half_life = float(current_horizon)
        else:
            fraction = (prior_value - target) / (prior_value - current_value)
            half_life = prior_horizon + fraction * (current_horizon - prior_horizon)
        output["half_life_status"] = "measurable_on_consumed_role_only"
        output["half_life_seconds"] = float(half_life)
        return output
    return output


def _memory_evidence() -> dict[str, object]:
    """Return best-effort process memory evidence without a new dependency."""

    if os.name == "nt":
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return {
                    "source": "windows_process_memory_counters_failed",
                    "windows_error": int(ctypes.get_last_error()),
                }
            return {
                "source": "windows_process_memory_counters",
                "current_working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "current_pagefile_bytes": int(counters.PagefileUsage),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
            }
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            return {
                "source": "windows_process_memory_counters_failed",
                "error_type": type(exc).__name__,
            }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        scale = 1 if sys.platform == "darwin" else 1024
        return {
            "source": "getrusage",
            "peak_working_set_bytes": int(usage.ru_maxrss * scale),
        }
    except (ImportError, OSError, ValueError):
        return {"source": "unavailable"}


def run_diagnostic(
    *,
    design_path: Path,
    binding_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Execute every frozen signal-horizon diagnostic and write one report."""

    design, design_sha = load_signal_decay_design(design_path)
    binding, binding_sha = load_signal_decay_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
    )
    source = design["source_contract"]
    access = design["data_access"]
    resources = design["runtime_resources"]
    statistics = design["statistical_contract"]
    regimes = design["regime_contract"]
    assert isinstance(source, Mapping)
    assert isinstance(access, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(statistics, Mapping)
    assert isinstance(regimes, Mapping)
    loader_path = design_path.parent / str(source["loader_design"])
    loader_design, _loader_sha, _profiles = load_round34_design(loader_path)
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Round 36 output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    report_path = output_root / "report.json"
    started = time.monotonic()
    lock = threading.Lock()
    sequence = 0

    def progress(phase: str, **details: object) -> None:
        nonlocal sequence
        with lock:
            sequence += 1
            payload = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "round": _ROUND,
                "sequence": sequence,
                "phase": phase,
                "run_elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(
                "round36-signal-decay "
                + " ".join(
                    f"{name}={value}"
                    for name, value in payload.items()
                    if name not in {"schema_version", "updated_at_utc"}
                ),
                flush=True,
            )
            write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    try:
        progress("initialize")
        thread_evidence = shared_runner._configure_worker_threads(
            int(resources["maximum_worker_threads"])
        )
        threads = int(thread_evidence["effective_worker_threads"])
        first = shared_runner._parse_date(
            source["certified_cache_materialization_start"],
            label="Round 36 materialization start",
        )
        last = shared_runner._parse_date(
            source["certified_cache_materialization_end"],
            label="Round 36 materialization end",
        )
        corpus = shared_runner._load_corpus(
            name="round36_multi_horizon_signal_decay",
            design=loader_design,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            first=first,
            last=last,
            evaluation_first=first,
            evaluation_last=last,
            memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
            threads=threads,
            heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
            progress=progress,
        )
        if (
            corpus.source_certificate.get("certificate_sha256")
            != source["corpus_certificate_sha256"]
            or corpus.targets_sha256 != source["barrier_targets_sha256"]
            or corpus.cache_key != source["cache_key"]
            or corpus.dataset.rows != source["dataset_rows"]
            or len(corpus.event_indexes) != source["event_rows"]
        ):
            raise ValueError("Round 36 certified corpus identity changed")
        metric_first = shared_runner._parse_date(
            access["metric_start"],
            label="Round 36 metric start",
        )
        metric_last = shared_runner._parse_date(
            access["metric_end"],
            label="Round 36 metric end",
        )
        metric_start_ms, metric_end_ms = shared_runner._utc_day_bounds(
            metric_first,
            metric_last,
        )
        event_indexes = np.flatnonzero(
            np.asarray(corpus.event_mask, dtype=bool)
            & (corpus.dataset.decision_time_ms >= metric_start_ms)
            & (corpus.dataset.decision_time_ms <= metric_end_ms)
        ).astype(np.int64)
        if len(event_indexes) < 1_024:
            raise ValueError("Round 36 calibration event support is insufficient")
        progress("horizon-support-start", event_rows=len(event_indexes))
        horizon_pairs: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}
        quote_times: list[np.ndarray] = []
        for horizon in _EXPECTED_HORIZONS:
            source_indexes, future_indexes, exclusions = exact_horizon_rows(
                corpus.dataset.decision_time_ms,
                event_indexes,
                horizon_seconds=horizon,
            )
            horizon_pairs[horizon] = (source_indexes, future_indexes, exclusions)
            quote_times.extend(
                [
                    corpus.dataset.decision_time_ms[source_indexes],
                    corpus.dataset.decision_time_ms[future_indexes],
                ]
            )
        requested_quote_times = np.unique(np.concatenate(quote_times)).astype(np.int64)
        progress(
            "zero-latency-bbo-query-start",
            requested_timestamps=len(requested_quote_times),
        )
        with MicrostructureWarehouse(
            warehouse_path,
            cache_root=cache_root,
            memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
            threads=threads,
            read_only=True,
        ) as warehouse:
            with progress_heartbeat(
                progress,
                phase="zero-latency-bbo-query-heartbeat",
                interval_seconds=float(resources["heartbeat_interval_seconds"]),
                details={"requested_timestamps": len(requested_quote_times)},
            ):
                zero_quotes = load_bbo_quotes_asof(
                    warehouse,
                    symbol=str(source["symbol"]),
                    arrival_time_ms=requested_quote_times,
                    maximum_quote_age_ms=int(
                        design["future_quote_contract"]["maximum_quote_age_ms"]
                    ),
                )
        progress(
            "zero-latency-bbo-query-complete",
            valid_quotes=int(np.sum(zero_quotes.valid)),
            invalid_quotes=int(np.sum(~zero_quotes.valid)),
        )
        feature_index = {
            name: position for position, name in enumerate(corpus.dataset.feature_names)
        }
        signals = design["signals"]
        assert isinstance(signals, list)
        placebo_contract = statistics["placebo"]
        assert isinstance(placebo_contract, Mapping)
        results: list[dict[str, object]] = []
        horizon_support: list[dict[str, object]] = []
        for horizon_position, horizon in enumerate(_EXPECTED_HORIZONS, start=1):
            source_indexes, _future_indexes, initial_exclusions = horizon_pairs[horizon]
            progress(
                "horizon-analysis-start",
                horizon_seconds=horizon,
                horizon_position=horizon_position,
                horizon_count=len(_EXPECTED_HORIZONS),
                retained_rows=len(source_indexes),
            )
            path = build_horizon_path(
                corpus.dataset,
                event_indexes,
                zero_quotes,
                horizon_seconds=horizon,
            )
            if dict(path.exclusion_counts) != {
                **initial_exclusions,
                **{
                    key: value
                    for key, value in path.exclusion_counts.items()
                    if key not in initial_exclusions
                },
            }:
                raise ValueError("Round 36 horizon support changed during construction")
            horizon_support.append(
                {
                    "horizon_seconds": horizon,
                    "rows": path.rows,
                    "first_decision_time_ms": int(path.decision_time_ms[0]),
                    "last_decision_time_ms": int(path.decision_time_ms[-1]),
                    "exclusion_counts": dict(path.exclusion_counts),
                }
            )
            completed: dict[int, dict[str, object]] = {}
            worker_count = max(1, min(threads, len(signals)))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _signal_result,
                        signal_position=position,
                        signal_contract=signal,
                        path=path,
                        dataset_features=corpus.dataset.features,
                        feature_index=feature_index,
                        requested_tails=statistics["ranked_tail_counts"],
                        placebo_replicates=int(placebo_contract["replicates"]),
                        placebo_master_seed=int(placebo_contract["seed"]),
                        minimum_regime_rows=int(
                            regimes["minimum_rows_to_report_metric"]
                        ),
                    ): position
                    for position, signal in enumerate(signals)
                }
                with progress_heartbeat(
                    progress,
                    phase="horizon-analysis-heartbeat",
                    interval_seconds=float(resources["heartbeat_interval_seconds"]),
                    details={
                        "horizon_seconds": horizon,
                        "signal_count": len(signals),
                    },
                ):
                    for future in as_completed(futures):
                        position = futures[future]
                        completed[position] = future.result()
                        progress(
                            "signal-analysis-complete",
                            horizon_seconds=horizon,
                            signal=str(signals[position]["name"]),
                            completed_signals=len(completed),
                            signal_count=len(signals),
                        )
            if set(completed) != set(range(len(signals))):
                raise ValueError("Round 36 signal analysis is incomplete")
            results.extend(completed[position] for position in range(len(signals)))
            progress(
                "horizon-analysis-complete",
                horizon_seconds=horizon,
                result_cells=len(completed),
            )
            del path, completed
            gc.collect()
        expected_cells = len(_EXPECTED_HORIZONS) * len(_EXPECTED_SIGNALS)
        if len(results) != expected_cells:
            raise ValueError("Round 36 result cell count is incomplete")
        decay_summary = []
        for signal in _EXPECTED_SIGNALS:
            signal_results = [item for item in results if item["signal"] == signal]
            decay_summary.append({"signal": signal, **_half_life(signal_results)})
        maximum_auc = max(
            results,
            key=lambda item: float(item["direction"]["weighted_roc_auc"]),
        )
        maximum_net = max(
            results,
            key=lambda item: float(
                item["cost_decomposition"]["mean_delayed_net_return_bps"]
            ),
        )
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "round": _ROUND,
            "report_canonical_sha256": "",
            "status": "diagnostic_complete_no_authority",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": binding["implementation"]["commit"],
            "source_evidence": {
                "symbol": source["symbol"],
                "market_type": source["market_type"],
                "feature_version": source["feature_version"],
                "corpus_certificate_sha256": corpus.source_certificate[
                    "certificate_sha256"
                ],
                "barrier_targets_sha256": corpus.targets_sha256,
                "cache_key": corpus.cache_key,
                "cache_state": corpus.cache_state,
                "dataset_rows": corpus.dataset.rows,
                "event_rows": len(corpus.event_indexes),
                "metric_event_rows": len(event_indexes),
                "metric_start": access["metric_start"],
                "metric_end": access["metric_end"],
                "input_resolution": source["input_resolution"],
                "bbo_execution_path_resolution_ms": source[
                    "bbo_execution_path_resolution_ms"
                ],
            },
            "stage_access": {
                "certified_source_materialized_through_development": True,
                "certified_barrier_targets_recomputed_for_identity_only": True,
                "calibration_metrics": True,
                "train_prediction_or_metrics": False,
                "early_stop_prediction_or_metrics": False,
                "policy_prediction_or_metrics": False,
                "development_prediction_or_metrics": False,
                "distant_confirmation_source_materialized": False,
                "distant_confirmation_prediction_or_metrics": False,
            },
            "zero_latency_quote_evidence": {
                "requested_timestamps": zero_quotes.rows,
                "valid_timestamps": int(np.sum(zero_quotes.valid)),
                "invalid_timestamps": int(np.sum(~zero_quotes.valid)),
                "first_timestamp_ms": int(zero_quotes.arrival_time_ms[0]),
                "last_timestamp_ms": int(zero_quotes.arrival_time_ms[-1]),
                "counterfactual_only": True,
                "execution_claim": False,
            },
            "horizon_support": horizon_support,
            "signal_horizon_results": results,
            "signal_decay_summary": decay_summary,
            "descriptive_extrema_not_selectable": {
                "maximum_weighted_auc": {
                    "signal": maximum_auc["signal"],
                    "horizon_seconds": maximum_auc["horizon_seconds"],
                    "weighted_roc_auc": maximum_auc["direction"]["weighted_roc_auc"],
                },
                "maximum_all_routed_mean_delayed_net_return_bps": {
                    "signal": maximum_net["signal"],
                    "horizon_seconds": maximum_net["horizon_seconds"],
                    "mean_delayed_net_return_bps": maximum_net["cost_decomposition"][
                        "mean_delayed_net_return_bps"
                    ],
                },
                "horizon_or_signal_selection_permitted": False,
            },
            "completeness": {
                "expected_signal_horizon_cells": expected_cells,
                "reported_signal_horizon_cells": len(results),
                "expected_daily_records": expected_cells * 5,
                "reported_daily_records": sum(
                    len(item["daily_direction"]) for item in results
                ),
                "expected_regime_records": expected_cells * 9,
                "reported_regime_records": sum(
                    len(item["regime_metrics"]) for item in results
                ),
                "placebo_replicates_per_cell": int(placebo_contract["replicates"]),
                "all_cells_complete": True,
            },
            "runtime_evidence": {
                "thread_configuration": thread_evidence,
                "memory": _memory_evidence(),
                "elapsed_seconds": float(time.monotonic() - started),
                "gpu_training_used": False,
                "gpu_training_not_applicable_because_no_model_was_trained": True,
                "persistent_duplicate_dataset_or_quote_archive_created": False,
            },
            "model_candidate": None,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
            "model_trained": False,
        }
        if (
            report["completeness"]["reported_daily_records"]
            != report["completeness"]["expected_daily_records"]
            or report["completeness"]["reported_regime_records"]
            != report["completeness"]["expected_regime_records"]
        ):
            raise ValueError("Round 36 report completeness contract failed")
        canonical = dict(report)
        canonical.pop("report_canonical_sha256")
        report["report_canonical_sha256"] = _canonical_sha256(canonical)
        write_json_atomic(report_path, report, indent=2, sort_keys=True)
        progress(
            "complete",
            report_canonical_sha256=report["report_canonical_sha256"],
            result_cells=len(results),
        )
        return report
    except Exception as exc:
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "round": _ROUND,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_elapsed_seconds": float(time.monotonic() - started),
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
            "model_trained": False,
        }
        write_json_atomic(report_path, failure, indent=2, sort_keys=True)
        progress("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs"
        / "model-research"
        / "action-value"
        / "round-036-multi-horizon-signal-decay-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT
        / "docs"
        / "model-research"
        / "action-value"
        / "round-036-signal-decay-execution-binding.json",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_diagnostic(
        design_path=arguments.design,
        binding_path=arguments.binding,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_root=arguments.output_root,
    )
    summary = {
        "status": report["status"],
        "report_canonical_sha256": report["report_canonical_sha256"],
        "reported_signal_horizon_cells": report["completeness"][
            "reported_signal_horizon_cells"
        ],
        "model_candidate": report["model_candidate"],
        "trading_authority": report["trading_authority"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
