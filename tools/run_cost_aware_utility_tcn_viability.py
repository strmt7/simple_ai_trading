"""Run the hash-bound Round 47 replay-aligned utility TCN screen."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.cost_aware_utility_tcn_model import (  # noqa: E402
    ACTION_PROBABILITY_FLOOR,
    CANDIDATES,
    RANK_WEIGHT,
    SEEDS,
    UtilityForecastBundle,
    action_labels,
    rank_ablation_gate,
    select_utility_trades,
    train_utility_candidates,
    utility_action_diagnostics,
)
from simple_ai_trading.cross_asset_cost_data import SYMBOLS  # noqa: E402
from simple_ai_trading.distributional_tcn_model import (  # noqa: E402
    BASE_ONE_WAY_COST_BPS,
    HORIZONS,
    QUANTILES,
    STRESS_ONE_WAY_COST_BPS,
    role_mask,
)
from simple_ai_trading.joint_distributional_tcn_model import (  # noqa: E402
    BOOTSTRAP_BLOCK_HOURS,
    BOOTSTRAP_SAMPLES,
    FAMILYWISE_LOWER_QUANTILE,
    joint_economic_gate,
    joint_forecast_diagnostics,
    replay_consensus_trades,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_stability_regularized_tcn_viability import (  # noqa: E402
    _artifact_manifest,
    _canonical_json,
    _canonical_sha256,
    _file_sha256,
    _git,
    _iso_timestamp,
    _load_verified_cache,
    _memory_evidence,
    _read_object,
    _role_summaries,
    _version,
    _write_csv,
    _write_hourly_ledger,
    _write_npy,
)


ROUND = 47
DESIGN_SCHEMA = "cost-aware-utility-distributional-tcn-design-v1"
BINDING_SCHEMA = "round-047-cost-aware-utility-tcn-execution-binding-v1"
REPORT_SCHEMA = "cost-aware-utility-distributional-tcn-report-v1"
PREDECESSOR_REPORT_SCHEMA = "stability-regularized-distributional-tcn-report-v1"
SOURCE_SCHEMA = "round-038-derivatives-source-certificate-v1"
DESIGN_CANONICAL_SHA256 = (
    "778019adcbc7a7156f4b87647181e8c3bd44fb1e28837cdf875f1266824d1bba"
)
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
PREDECESSOR_REPORT_CANONICAL_SHA256 = (
    "7cd0bce1e797a77a89a389670677e1a3bce785d2018549ac168c4d753122076b"
)
PREDECESSOR_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
)
CACHE_METADATA_SHA256 = (
    "033480cd3b5669a060f297e7e477c2543a551602834914803bfd1127608d1135"
)


class ProgressWriter:
    def __init__(self, root: Path) -> None:
        self.status_path = root / "status.json"
        self.events_path = root / "progress_events.jsonl"
        self.started = time.perf_counter()
        self.sequence = 0
        self.frozen = False

    def __call__(self, phase: str, detail: Mapping[str, object]) -> None:
        if self.frozen:
            raise RuntimeError("Round 47 progress stream is already frozen")
        self.sequence += 1
        payload = {
            "schema_version": "round-047-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "phase": phase,
            "detail": dict(detail),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        encoded = _canonical_json(payload)
        print(encoded, flush=True)
        with self.events_path.open("a", encoding="ascii", newline="\n") as stream:
            stream.write(encoded + "\n")
        write_json_atomic(self.status_path, payload, indent=2, sort_keys=True)

    def freeze(self, detail: Mapping[str, object]) -> None:
        self("finalization", {"status": "complete", **detail})
        self.frozen = True


def _validate_hashed_object(
    value: Mapping[str, object],
    *,
    field: str,
    schema: str,
    round_number: int,
    label: str,
) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if (
        value.get("schema_version") != schema
        or value.get("round") != round_number
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError(f"{label} identity is invalid")
    return claimed


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 47 design section is missing: {key}")
    return value


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 47 design")
    design_sha = _validate_hashed_object(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        round_number=ROUND,
        label="Round 47 design",
    )
    data = _mapping(design, "data_contract")
    model = _mapping(design, "model_contract")
    objective = _mapping(model, "training_objective")
    calibration = _mapping(design, "probability_calibration_contract")
    policy = _mapping(design, "fixed_policy_contract")
    economics = _mapping(design, "economic_gate")
    governance = _mapping(design, "governance")
    ai = _mapping(design, "ai_contract")
    candidates = model.get("candidates")
    if (
        design_sha != DESIGN_CANONICAL_SHA256
        or data.get("source_certificate_canonical_sha256") != SOURCE_CANONICAL_SHA256
        or data.get("predecessor_dataset_sha256") != PREDECESSOR_DATASET_SHA256
        or data.get("round_45_derived_cache_metadata_sha256") != CACHE_METADATA_SHA256
        or data.get("symbols") != list(SYMBOLS)
        or data.get("forecast_target_horizons_hours") != list(HORIZONS)
        or data.get("forecast_target_quantiles") != list(QUANTILES)
        or not isinstance(candidates, list)
        or [item.get("id") for item in candidates if isinstance(item, Mapping)]
        != list(CANDIDATES)
        or model.get("seeds") != list(SEEDS)
        or objective.get("pairwise_rank_loss_weight_ablation") != RANK_WEIGHT
        or calibration.get("method")
        != "one scalar temperature per candidate and seed, shared across symbols, horizons, and sides"
        or policy.get("action_probability_floor") != ACTION_PROBABILITY_FLOOR
        or policy.get("base_one_way_transition_cost_bps") != BASE_ONE_WAY_COST_BPS
        or policy.get("stress_one_way_transition_cost_bps") != STRESS_ONE_WAY_COST_BPS
        or policy.get("leverage") != 1.0
        or economics.get("bootstrap_replicates") != BOOTSTRAP_SAMPLES
        or economics.get("familywise_circular_block_bootstrap_hours")
        != BOOTSTRAP_BLOCK_HOURS
        or economics.get("one_sided_familywise_lower_quantile")
        != FAMILYWISE_LOWER_QUANTILE
    ):
        raise ValueError("Round 47 implementation and frozen design differ")
    denied = (
        "selection_confirmation_2025_h2_access_permitted",
        "terminal_2026_access_permitted",
        "testnet_or_live_execution_permitted",
        "promotion_permitted",
        "leverage_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "post_outcome_parameter_seed_candidate_or_threshold_selection_permitted",
        "manual_graph_or_result_editing_permitted",
        "profitability_ai_or_ranking_uplift_claim_permitted",
    )
    if any(governance.get(field) is not False for field in denied):
        raise ValueError("Round 47 governance boundary differs from the design")
    if (
        ai.get("benchmark_is_financial_edge_evidence") is not False
        or ai.get("market_features_or_future_outcomes_supplied_to_language_models")
        is not False
        or ai.get("language_model_numerical_forecasts_or_orders_permitted") is not False
        or ai.get("ai_trade_ablation_permitted") is not False
    ):
        raise ValueError("Round 47 AI boundary differs from the design")
    return design, design_sha


def _validate_external_input(
    path: Path,
    binding_entry: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not path.is_file() or binding_entry.get("file_sha256") != _file_sha256(path):
        raise ValueError(f"{label} differs from the execution binding")


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
    predecessor_report_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 47 execution binding")
    binding_sha = _validate_hashed_object(
        binding,
        field="binding_sha256",
        schema=BINDING_SCHEMA,
        round_number=ROUND,
        label="Round 47 execution binding",
    )
    source = binding.get("source_certificate")
    predecessor = binding.get("predecessor_report")
    if (
        binding.get("design_sha256") != design_sha256
        or not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or not isinstance(predecessor, Mapping)
        or predecessor.get("canonical_sha256") != PREDECESSOR_REPORT_CANONICAL_SHA256
    ):
        raise ValueError("Round 47 binding inputs differ")
    _validate_external_input(
        source_certificate_path, source, label="Round 47 source certificate"
    )
    _validate_external_input(
        predecessor_report_path, predecessor, label="Round 47 predecessor report"
    )
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or _git("status", "--porcelain"):
        raise ValueError("Round 47 requires a bound commit and clean worktree")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 47 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 47 binding has no blobs")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 47 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}") != expected
            or _git("rev-parse", f"HEAD:{relative_path}") != expected
        ):
            raise ValueError(f"Round 47 bound blob changed: {relative_path}")
    return binding, binding_sha, implementation_commit


def _write_predictions(
    evidence_root: Path,
    bundle: UtilityForecastBundle,
) -> list[Path]:
    root = evidence_root / "predictions"
    tensors = {
        "seed_quantile_predictions_bps": bundle.seed_predictions_bps,
        "ensemble_quantile_predictions_bps": bundle.ensemble_predictions_bps,
        "seed_utility_predictions_bps": bundle.seed_utility_bps,
        "ensemble_utility_predictions_bps": bundle.ensemble_utility_bps,
        "seed_action_logits": bundle.seed_action_logits,
        "seed_action_probabilities": bundle.seed_action_probabilities,
        "ensemble_action_probabilities": bundle.ensemble_action_probabilities,
    }
    paths: list[Path] = []
    for suffix, values in tensors.items():
        path = root / f"{bundle.candidate_id}_{suffix}.npy"
        _write_npy(path, values)
        paths.append(path)
    return paths


def _trade_rows(
    dataset: object,
    utility_bps: np.ndarray,
    trades: Sequence[object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in trades:
        horizon_index = HORIZONS.index(int(trade.horizon_hours))
        base_realized = (
            int(trade.side)
            * float(
                utility_bps[
                    int(trade.decision_index), int(trade.symbol_index), horizon_index
                ]
            )
            - 2.0 * BASE_ONE_WAY_COST_BPS
        )
        stress_realized = base_realized - 2.0 * (
            STRESS_ONE_WAY_COST_BPS - BASE_ONE_WAY_COST_BPS
        )
        rows.append(
            {
                **trade.asdict(),
                "decision_time_utc": _iso_timestamp(int(trade.decision_time_ms)),
                "base_realized_net_bps": base_realized,
                "stress_realized_net_bps": stress_realized,
            }
        )
    return rows


def _economic_gate(
    *,
    forecast_gate_passed: bool,
    action_gate_passed: bool,
    stress: object,
) -> dict[str, object]:
    gate = joint_economic_gate(
        forecast_gate_passed=forecast_gate_passed and action_gate_passed,
        stress=stress,
    )
    reasons = list(gate["reasons"])
    if not action_gate_passed:
        reasons.append("action_quality_gate_failed")
    if forecast_gate_passed and not action_gate_passed:
        reasons = [item for item in reasons if item != "forecast_gate_failed"]
    return {**gate, "passed": not reasons, "reasons": list(dict.fromkeys(reasons))}


def _role_label_rows(
    dataset: object, utility_bps: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role_name in ("training", "early_stop", "calibration", "evaluation"):
        mask = role_mask(dataset, role_name)
        labels = action_labels(utility_bps[mask])
        for horizon_index, horizon in enumerate(HORIZONS):
            values = utility_bps[mask, :, horizon_index]
            rows.append(
                {
                    "role": role_name,
                    "horizon_hours": horizon,
                    "rows": int(values.size),
                    "short_profitable_fraction": float(
                        np.mean(labels[:, :, horizon_index, 0])
                    ),
                    "long_profitable_fraction": float(
                        np.mean(labels[:, :, horizon_index, 1])
                    ),
                    "no_trade_fraction": float(np.mean(np.abs(values) <= 12.0)),
                    "mean_signed_utility_bps": float(np.mean(values)),
                    "finite": bool(np.isfinite(values).all()),
                }
            )
    return rows


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root)
    started = time.perf_counter()
    design, design_sha = _validate_design(arguments.design.resolve())
    binding, binding_sha, implementation_commit = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        source_certificate_path=arguments.source_certificate.resolve(),
        predecessor_report_path=arguments.predecessor_report.resolve(),
    )
    source = _read_object(
        arguments.source_certificate.resolve(), "Round 47 source certificate"
    )
    source_sha = _validate_hashed_object(
        source,
        field="source_certificate_sha256",
        schema=SOURCE_SCHEMA,
        round_number=38,
        label="Round 47 source certificate",
    )
    predecessor = _read_object(
        arguments.predecessor_report.resolve(), "Round 47 predecessor report"
    )
    predecessor_sha = _validate_hashed_object(
        predecessor,
        field="report_canonical_sha256",
        schema=PREDECESSOR_REPORT_SCHEMA,
        round_number=46,
        label="Round 47 predecessor report",
    )
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
            "source_certificate_canonical_sha256": source_sha,
            "predecessor_report_canonical_sha256": predecessor_sha,
        },
    )
    dataset, cache_manifest, cache_verification = _load_verified_cache(
        arguments.derived_cache.resolve(), binding
    )
    if dataset.dataset_sha256 != PREDECESSOR_DATASET_SHA256:
        raise RuntimeError("Round 47 dataset identity differs from the frozen design")
    roles = _role_summaries(dataset)
    progress(
        "round47_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "features_per_symbol": len(dataset.feature_names),
            "dataset_sha256": dataset.dataset_sha256,
            "cache_copied": False,
        },
    )
    bundles, utility_bps, preflight = train_utility_candidates(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    label_rows = _role_label_rows(dataset, utility_bps)
    forecast_diagnostics: dict[str, dict[str, object]] = {}
    action_diagnostics: dict[str, dict[str, object]] = {}
    forecast_monthly_rows: list[dict[str, object]] = []
    quantile_stability_rows: list[dict[str, object]] = []
    prediction_paths: list[Path] = []
    trade_map: dict[str, tuple[object, ...]] = {}
    replay_map: dict[tuple[str, str], object] = {}
    for candidate_index, candidate_id in enumerate(CANDIDATES):
        bundle = bundles[candidate_id]
        prediction_paths.extend(_write_predictions(evidence_root, bundle))
        monthly, stability, forecast = joint_forecast_diagnostics(
            dataset,
            bundle,  # type: ignore[arg-type]
        )
        forecast_monthly_rows.extend(monthly)
        quantile_stability_rows.extend(stability)
        forecast_diagnostics[candidate_id] = forecast
        action_diagnostics[candidate_id] = utility_action_diagnostics(
            dataset, utility_bps, bundle
        )
        trades = select_utility_trades(dataset, utility_bps, bundle)
        trade_map[candidate_id] = trades
        base = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="base",
            one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
            bootstrap_seed=SEEDS[0] + candidate_index * 1_000,
        )
        stress = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="stress",
            one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
            bootstrap_seed=SEEDS[0] + candidate_index * 1_000 + 100,
        )
        if not (
            tuple(item.trade_id for item in base.trades)
            == tuple(item.trade_id for item in stress.trades)
            and np.array_equal(base.positions, stress.positions)
        ):
            raise RuntimeError("Round 47 stress replay changed the base trade ledger")
        replay_map[(candidate_id, "base")] = base
        replay_map[(candidate_id, "stress")] = stress
    rank_gate = rank_ablation_gate(
        action_diagnostics[CANDIDATES[0]], action_diagnostics[CANDIDATES[1]]
    )
    action_gate_passed = {
        CANDIDATES[0]: bool(action_diagnostics[CANDIDATES[0]]["gate"]["passed"]),
        CANDIDATES[1]: bool(action_diagnostics[CANDIDATES[1]]["gate"]["passed"])
        and bool(rank_gate["passed"]),
    }
    economic_gates: dict[str, dict[str, object]] = {}
    for candidate_id in CANDIDATES:
        economic_gates[candidate_id] = _economic_gate(
            forecast_gate_passed=bool(
                forecast_diagnostics[candidate_id]["gate"]["passed"]
            ),
            action_gate_passed=action_gate_passed[candidate_id],
            stress=replay_map[(candidate_id, "stress")],
        )
        base = replay_map[(candidate_id, "base")]
        stress = replay_map[(candidate_id, "stress")]
        progress(
            "round47_candidate",
            {
                "status": "complete",
                "candidate_id": candidate_id,
                "forecast_gate_passed": forecast_diagnostics[candidate_id]["gate"][
                    "passed"
                ],
                "action_gate_passed": action_gate_passed[candidate_id],
                "economic_gate_passed": economic_gates[candidate_id]["passed"],
                "trades": len(trade_map[candidate_id]),
                "base_total_net_return_fraction": base.metrics[
                    "total_net_return_fraction"
                ],
                "stress_total_net_return_fraction": stress.metrics[
                    "total_net_return_fraction"
                ],
            },
        )

    paths = {
        "forecast_diagnostics": evidence_root / "forecast_diagnostics.csv",
        "horizon_summary": evidence_root / "horizon_summary.csv",
        "symbol_horizon": evidence_root / "symbol_horizon_summary.csv",
        "quantile_stability": evidence_root / "quantile_seed_stability.csv",
        "utility_horizons": evidence_root / "utility_horizon_summary.csv",
        "action_horizons": evidence_root / "action_side_horizon_summary.csv",
        "action_stability": evidence_root / "action_seed_stability.csv",
        "labels": evidence_root / "role_label_prevalence.csv",
        "gates": evidence_root / "quality_gates.json",
        "training": evidence_root / "training_history.json",
        "models": evidence_root / "models.csv",
        "roles": evidence_root / "roles.csv",
        "trades": evidence_root / "trades.csv",
        "replays": evidence_root / "replays.csv",
        "monthly": evidence_root / "monthly_economics.csv",
        "symbols": evidence_root / "symbol_economics.csv",
        "ledger": evidence_root / "hourly_ledger.csv.gz",
        "scalers": evidence_root / "scalers.json",
    }
    _write_csv(paths["forecast_diagnostics"], forecast_monthly_rows)
    _write_csv(
        paths["horizon_summary"],
        (
            row
            for candidate_id in CANDIDATES
            for row in forecast_diagnostics[candidate_id]["horizons"]
        ),
    )
    _write_csv(
        paths["symbol_horizon"],
        (
            row
            for candidate_id in CANDIDATES
            for row in forecast_diagnostics[candidate_id]["symbol_horizons"]
        ),
    )
    _write_csv(paths["quantile_stability"], quantile_stability_rows)
    _write_csv(
        paths["utility_horizons"],
        (
            row
            for candidate_id in CANDIDATES
            for row in action_diagnostics[candidate_id]["utility_horizons"]
        ),
    )
    _write_csv(
        paths["action_horizons"],
        (
            row
            for candidate_id in CANDIDATES
            for row in action_diagnostics[candidate_id]["action_side_horizons"]
        ),
    )
    _write_csv(
        paths["action_stability"],
        (
            row
            for candidate_id in CANDIDATES
            for row in action_diagnostics[candidate_id]["seed_stability"]
        ),
    )
    _write_csv(paths["labels"], label_rows)
    write_json_atomic(
        paths["gates"],
        {
            "forecast": {
                candidate_id: forecast_diagnostics[candidate_id]["gate"]
                for candidate_id in CANDIDATES
            },
            "action": {
                candidate_id: action_diagnostics[candidate_id]["gate"]
                for candidate_id in CANDIDATES
            },
            "rank_ablation": rank_gate,
            "combined_action_pass": action_gate_passed,
            "economic": economic_gates,
        },
        indent=2,
        sort_keys=True,
    )
    write_json_atomic(
        paths["training"],
        {
            "schema_version": "round-047-training-history-v1",
            "round": ROUND,
            "candidates": {
                candidate_id: list(bundles[candidate_id].training_history)
                for candidate_id in CANDIDATES
            },
        },
        indent=2,
        sort_keys=True,
    )
    _write_csv(
        paths["models"],
        (
            artifact.asdict()
            for candidate_id in CANDIDATES
            for artifact in bundles[candidate_id].artifacts
        ),
    )
    _write_csv(paths["roles"], roles)
    trade_rows = [
        row
        for candidate_id in CANDIDATES
        for row in _trade_rows(dataset, utility_bps, trade_map[candidate_id])
    ]
    _write_csv(paths["trades"], trade_rows)
    replays = [
        replay_map[(candidate_id, scenario)]
        for candidate_id in CANDIDATES
        for scenario in ("base", "stress")
    ]
    _write_csv(
        paths["replays"],
        (
            {
                **{
                    key: value
                    for key, value in replay.metrics.items()
                    if key
                    not in {
                        "monthly",
                        "trades_by_symbol",
                        "symbol_net_bps",
                        "bootstrap_mean_hourly_portfolio_bps",
                    }
                },
                "trades_by_symbol": _canonical_json(replay.metrics["trades_by_symbol"]),
                "symbol_net_bps": _canonical_json(replay.metrics["symbol_net_bps"]),
                "bootstrap_mean_hourly_portfolio_bps": _canonical_json(
                    replay.metrics["bootstrap_mean_hourly_portfolio_bps"]
                ),
            }
            for replay in replays
        ),
    )
    _write_csv(
        paths["monthly"],
        (
            {
                "candidate_id": replay.metrics["candidate_id"],
                "scenario": replay.scenario,
                **row,
            }
            for replay in replays
            for row in replay.metrics["monthly"]
        ),
    )
    _write_csv(
        paths["symbols"],
        (
            {
                "candidate_id": replay.metrics["candidate_id"],
                "scenario": replay.scenario,
                "symbol": symbol,
                "trades": replay.metrics["trades_by_symbol"][symbol],
                "net_bps": replay.metrics["symbol_net_bps"][symbol],
            }
            for replay in replays
            for symbol in SYMBOLS
        ),
    )
    ledger_rows = _write_hourly_ledger(paths["ledger"], replays)
    first_bundle = bundles[CANDIDATES[0]]
    write_json_atomic(
        paths["scalers"],
        {
            "schema_version": "round-047-training-scalers-v1",
            "feature_names": list(dataset.feature_names),
            "feature_scaler": first_bundle.feature_scaler.asdict(),
            "distributional_target_scaler": first_bundle.target_scaler.asdict(),
            "replay_aligned_utility_scaler": first_bundle.utility_scaler.asdict(),
            "fit_role": "training",
            "identical_across_candidates": all(
                bundle.feature_scaler.asdict() == first_bundle.feature_scaler.asdict()
                and bundle.target_scaler.asdict() == first_bundle.target_scaler.asdict()
                and bundle.utility_scaler.asdict()
                == first_bundle.utility_scaler.asdict()
                for bundle in bundles.values()
            ),
        },
        indent=2,
        sort_keys=True,
    )
    progress.freeze(
        {
            "candidate_forecast_gate_pass_count": sum(
                bool(forecast_diagnostics[value]["gate"]["passed"])
                for value in CANDIDATES
            ),
            "candidate_action_gate_pass_count": sum(action_gate_passed.values()),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[value]["passed"]) for value in CANDIDATES
            ),
            "ledger_rows": ledger_rows,
        }
    )
    model_paths = [
        Path(artifact.path)
        for candidate_id in CANDIDATES
        for artifact in bundles[candidate_id].artifacts
    ]
    output_paths = (
        *paths.values(),
        progress.events_path,
        progress.status_path,
        *prediction_paths,
        *model_paths,
    )
    candidate_reports: list[dict[str, object]] = []
    for candidate_id in CANDIDATES:
        base = replay_map[(candidate_id, "base")]
        stress = replay_map[(candidate_id, "stress")]
        candidate_reports.append(
            {
                "candidate_id": candidate_id,
                "models": [
                    artifact.asdict() for artifact in bundles[candidate_id].artifacts
                ],
                "forecast_diagnostics": forecast_diagnostics[candidate_id],
                "action_diagnostics": action_diagnostics[candidate_id],
                "combined_action_gate_passed": action_gate_passed[candidate_id],
                "rank_ablation_gate": (
                    rank_gate if candidate_id == CANDIDATES[1] else None
                ),
                "trade_count": len(trade_map[candidate_id]),
                "trade_ledger_sha256": _canonical_sha256(
                    [item.asdict() for item in trade_map[candidate_id]]
                ),
                "exact_replay_aligned_target_identity": True,
                "fixed_ledger_under_stress": True,
                "base": dict(base.metrics),
                "stress": dict(stress.metrics),
                "economic_gate": economic_gates[candidate_id],
                "prediction_files": _artifact_manifest(
                    [
                        path
                        for path in prediction_paths
                        if path.name.startswith(candidate_id + "_")
                    ]
                ),
            }
        )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "design": design,
        "binding": binding,
        "source_lineage": {
            "source_certificate_canonical_sha256": source_sha,
            "source_certificate_file_sha256": _file_sha256(
                arguments.source_certificate.resolve()
            ),
            "predecessor_report_canonical_sha256": predecessor_sha,
            "predecessor_report_file_sha256": _file_sha256(
                arguments.predecessor_report.resolve()
            ),
        },
        "dataset": {
            "dataset_sha256": dataset.dataset_sha256,
            "symbols": list(SYMBOLS),
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count_per_symbol": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "target_horizons_hours": list(HORIZONS),
            "target_quantiles": list(QUANTILES),
            "first_timestamp_utc": _iso_timestamp(int(dataset.timestamps_ms[0])),
            "last_timestamp_utc": _iso_timestamp(int(dataset.timestamps_ms[-1])),
            "roles": roles,
            "role_label_prevalence": label_rows,
            "replay_aligned_utility_target_shape": list(utility_bps.shape),
            "replay_aligned_utility_target_finite_in_all_roles": True,
            "matrix_bytes": int(
                dataset.features.nbytes
                + dataset.hourly_return_bps.nbytes
                + dataset.forward_return_bps.nbytes
                + utility_bps.nbytes
            ),
            "derived_cache_inputs": cache_manifest,
            "cache_verification": cache_verification,
        },
        "compute": {
            "requested_backend": arguments.compute_backend,
            "backend_kind": preflight["backend_kind"],
            "backend_device": preflight["backend_device"],
            "preflight": preflight,
            "torch_version": _version("torch"),
            "torch_directml_version": _version("torch-directml"),
            "numpy_version": _version("numpy"),
            "scipy_version": _version("scipy"),
            "model_artifacts": len(model_paths),
            "all_artifacts_exact_reload": all(
                artifact.reload_max_abs_quantile_error <= 1e-6
                and artifact.reload_max_abs_utility_error <= 1e-6
                and artifact.reload_max_abs_logit_error <= 1e-6
                for candidate_id in CANDIDATES
                for artifact in bundles[candidate_id].artifacts
            ),
            "all_temperature_fits_nonworsening": all(
                artifact.calibration_bce_after
                <= artifact.calibration_bce_before + 1e-12
                for candidate_id in CANDIDATES
                for artifact in bundles[candidate_id].artifacts
            ),
        },
        "rank_ablation_gate": rank_gate,
        "candidates": candidate_reports,
        "ai_evidence": _mapping(design, "ai_contract"),
        "outputs": _artifact_manifest(output_paths),
        "hourly_ledger_rows": ledger_rows,
        "progress_event_count": progress.sequence,
        "claims": {
            "candidate_forecast_gate_pass_count": sum(
                bool(forecast_diagnostics[value]["gate"]["passed"])
                for value in CANDIDATES
            ),
            "candidate_action_gate_pass_count": sum(action_gate_passed.values()),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[value]["passed"]) for value in CANDIDATES
            ),
            "rank_ablation_passed": bool(rank_gate["passed"]),
            "profitability_established": False,
            "ai_improvement_established": False,
            "selection_confirmation_established": False,
            "promotion_authorized": False,
            "testnet_or_live_trading_authorized": False,
            "leverage_authorized": False,
            "reason": "All Round 47 roles are consumed development evidence. No result authorizes trading or establishes future profitability or AI edge.",
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
        },
    }
    if not all(
        math.isfinite(float(candidate[scenario]["total_net_return_fraction"]))
        for candidate in candidate_reports
        for scenario in ("base", "stress")
    ):
        raise RuntimeError("Round 47 report contains nonfinite economics")
    report["report_canonical_sha256"] = _canonical_sha256(report)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-binding.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--predecessor-report", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=("directml", "cpu"), default="directml"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
