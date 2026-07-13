#!/usr/bin/env python3
"""Run the frozen Round 54 sequential distributional action-value prototype."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import gzip
import hashlib
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.cross_asset_cost_data import SYMBOLS  # noqa: E402
from simple_ai_trading.distributional_tcn_model import (  # noqa: E402
    DistributionalDataset,
    HORIZONS,
    role_mask,
)
from simple_ai_trading.sequential_action_value_model import (  # noqa: E402
    DEFAULT_SPEC,
    POLICY_IDS,
    ROUND,
    consensus_policy_actions,
    pairwise_seed_score_spearman,
    replay_consensus_actions,
    train_sequential_q_ensemble,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DESIGN_SCHEMA = (
    "round-054-sequential-distributional-action-value-prototype-design-v1"
)
BINDING_SCHEMA = "round-047-cost-aware-utility-tcn-execution-binding-v1"
EXPECTED_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


class ProgressWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "progress_events.jsonl"
        self.started = time.perf_counter()

    def __call__(self, stage: str, values: Mapping[str, object]) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "stage": stage,
            **dict(values),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(payload) + "\n")
        detail = " ".join(f"{key}={value}" for key, value in values.items())
        print(f"round54 {stage} {detail}".rstrip(), flush=True)


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 54 design")
    canonical = dict(design)
    claimed = canonical.pop("design_sha256", None)
    actual = _canonical_sha256(canonical)
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen_development_only"
        or claimed != actual
    ):
        raise ValueError("Round 54 design identity or status is invalid")
    return design, actual


def _validate_source_binding(
    path: Path,
    derived_cache: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    binding = _read_object(path, "Round 47 source binding")
    if binding.get("schema_version") != BINDING_SCHEMA or binding.get("round") != 47:
        raise ValueError("Round 54 source binding has an invalid schema")
    rows = binding.get("derived_cache")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Round 54 source binding has no derived-cache manifest")
    output: dict[str, Path] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Round 54 source binding cache row is invalid")
        name = str(raw.get("name", ""))
        expected_sha = str(raw.get("sha256", ""))
        expected_bytes = int(raw.get("bytes", -1))
        candidate = (derived_cache / name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Round 54 source cache is missing: {candidate}")
        if candidate.stat().st_size != expected_bytes:
            raise ValueError(f"Round 54 source cache byte count differs: {name}")
        if _file_sha256(candidate) != expected_sha:
            raise ValueError(f"Round 54 source cache hash differs: {name}")
        output[name] = candidate
    required = {
        "timestamps_ms.npy",
        "features.npy",
        "hourly_return_bps.npy",
        "forward_return_bps.npy",
        "metadata.json",
    }
    if set(output) != required:
        raise ValueError("Round 54 source cache file set differs from the frozen set")
    return binding, output


def _array_identity(
    arrays: Sequence[np.ndarray],
    names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _load_dataset(paths: Mapping[str, Path]) -> DistributionalDataset:
    metadata = _read_object(paths["metadata.json"], "Round 54 dataset metadata")
    if metadata.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Round 54 metadata dataset identity differs")
    feature_names = tuple(str(item) for item in metadata.get("feature_names", []))
    if len(feature_names) != 71:
        raise ValueError("Round 54 feature-name count differs from 71")
    timestamps = np.load(paths["timestamps_ms.npy"], allow_pickle=False)
    features = np.load(paths["features.npy"], allow_pickle=False)
    hourly = np.load(paths["hourly_return_bps.npy"], allow_pickle=False)
    forward = np.load(paths["forward_return_bps.npy"], allow_pickle=False)
    if (
        timestamps.shape != (30_647,)
        or features.shape != (30_647, 3, 71)
        or hourly.shape != (30_647, 3)
        or forward.shape != (30_647, 3, len(HORIZONS))
    ):
        raise ValueError("Round 54 source array dimensions differ from the contract")
    identity = _array_identity(
        (timestamps, features, hourly, forward),
        feature_names
        + tuple(f"forward_return_{value}h_bps" for value in HORIZONS),
    )
    if identity != EXPECTED_DATASET_SHA256:
        raise ValueError("Round 54 reconstructed dataset identity differs")
    return DistributionalDataset(
        feature_names=feature_names,
        timestamps_ms=timestamps,
        features=features,
        hourly_return_bps=hourly,
        forward_return_bps=forward,
        dataset_sha256=identity,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _canonical_json(value)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_replay_ledger(path: Path, replay) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        fields = [
            "timestamp_utc",
            *[f"{symbol}_position" for symbol in SYMBOLS],
            *[f"{symbol}_net_bps" for symbol in SYMBOLS],
            "portfolio_net_bps",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, timestamp_ms in enumerate(replay.timestamps_ms):
            payload: dict[str, object] = {
                "timestamp_utc": datetime.fromtimestamp(
                    int(timestamp_ms) / 1_000, tz=UTC
                ).isoformat(),
                "portfolio_net_bps": float(replay.portfolio_return_bps[row]),
            }
            for symbol_index, symbol in enumerate(SYMBOLS):
                payload[f"{symbol}_position"] = int(
                    replay.actions[row, symbol_index]
                )
                payload[f"{symbol}_net_bps"] = float(
                    replay.symbol_net_bps[row, symbol_index]
                )
            writer.writerow(payload)


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _calibration_reasons(
    *,
    stability: float,
    stress: Mapping[str, object],
    artifacts,
) -> list[str]:
    reasons: list[str] = []
    if any(item.early_stop_td_skill <= 0.0 for item in artifacts):
        reasons.append("one_or_more_seeds_failed_positive_early_stop_td_skill")
    if any(item.reload_max_abs_q_error > 1e-6 for item in artifacts):
        reasons.append("one_or_more_seed_reload_errors_exceed_1e_6")
    if stability < 0.5:
        reasons.append("minimum_pairwise_seed_score_spearman_below_0_5")
    if int(stress["closed_trades"]) < 30:
        reasons.append("fewer_than_thirty_closed_trades")
    if int(stress["active_days"]) < 30:
        reasons.append("fewer_than_thirty_active_days")
    if _finite(stress["profit_factor"]) < 1.05:
        reasons.append("stress_profit_factor_below_1_05")
    if _finite(stress["maximum_drawdown_fraction"]) > 0.10:
        reasons.append("stress_drawdown_exceeds_ten_percent")
    if _finite(stress["maximum_single_symbol_fraction_of_absolute_net_pnl"]) > 0.60:
        reasons.append("single_symbol_absolute_net_pnl_fraction_exceeds_sixty_percent")
    bootstrap = stress["bootstrap_mean_hourly_portfolio_bps"]
    if not isinstance(bootstrap, Mapping) or _finite(bootstrap.get("lower_bps")) <= 0.0:
        reasons.append("stress_block_bootstrap_lower_not_positive")
    if int(stress["symbols_with_activity"]) != 3:
        reasons.append("not_all_three_symbols_have_activity")
    return reasons


def _evaluation_reasons(stress: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(stress["closed_trades"]) < 180:
        reasons.append("fewer_than_one_hundred_eighty_closed_trades")
    if int(stress["active_days"]) < 90:
        reasons.append("fewer_than_ninety_active_days")
    if int(stress["positive_months"]) < 6:
        reasons.append("fewer_than_six_positive_months")
    if _finite(stress["profit_factor"]) < 1.05:
        reasons.append("stress_profit_factor_below_1_05")
    if _finite(stress["maximum_drawdown_fraction"]) > 0.10:
        reasons.append("stress_drawdown_exceeds_ten_percent")
    if _finite(stress["maximum_single_symbol_fraction_of_absolute_net_pnl"]) > 0.50:
        reasons.append("single_symbol_absolute_net_pnl_fraction_exceeds_half")
    bootstrap = stress["bootstrap_mean_hourly_portfolio_bps"]
    if not isinstance(bootstrap, Mapping) or _finite(bootstrap.get("lower_bps")) <= 0.0:
        reasons.append("stress_familywise_block_bootstrap_lower_not_positive")
    return reasons


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root)
    design, design_sha = _validate_design(arguments.design.resolve())
    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Round 54 requires a clean committed implementation")
    implementation_commit = _git("rev-parse", "HEAD")
    binding, cache_paths = _validate_source_binding(
        arguments.dataset_binding.resolve(),
        arguments.derived_cache.resolve(),
    )
    progress(
        "source_validation",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "implementation_commit": implementation_commit,
            "cache_files": len(cache_paths),
        },
    )
    dataset = _load_dataset(cache_paths)
    role_rows = [
        {
            "role": role,
            "timestamps": int(np.count_nonzero(role_mask(dataset, role))),
            "rows": int(np.count_nonzero(role_mask(dataset, role))) * len(SYMBOLS),
        }
        for role in ("training", "early_stop", "calibration", "evaluation")
    ]
    _write_csv(evidence_root / "roles.csv", role_rows)
    progress(
        "dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "dataset_sha256": dataset.dataset_sha256,
        },
    )
    ensemble = train_sequential_q_ensemble(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        spec=DEFAULT_SPEC,
        progress=progress,
    )
    prediction_path = evidence_root / "seed_q_bps.npy"
    np.save(prediction_path, ensemble.seed_q_bps, allow_pickle=False)
    _write_csv(
        evidence_root / "models.csv",
        [item.asdict() for item in ensemble.artifacts],
    )
    _write_csv(evidence_root / "training_history.csv", ensemble.training_history)
    progress(
        "model_training",
        {
            "status": "complete",
            "models": len(ensemble.artifacts),
            "prediction_bytes": prediction_path.stat().st_size,
            "reward_scale_bps": ensemble.reward_scale_bps,
        },
    )
    calibration_mask = role_mask(dataset, "calibration")
    calibration_rows: list[dict[str, object]] = []
    calibration_reports: dict[str, object] = {}
    calibration_replays: dict[str, tuple[object, object]] = {}
    for policy_index, policy_id in enumerate(POLICY_IDS):
        stability = pairwise_seed_score_spearman(
            ensemble.seed_q_bps,
            calibration_mask,
            policy_id,
        )
        actions, action_diagnostics = consensus_policy_actions(
            ensemble.seed_q_bps,
            calibration_mask,
            policy_id,
        )
        base = replay_consensus_actions(
            dataset,
            actions,
            calibration_mask,
            policy_id=policy_id,
            role="calibration",
            scenario="base",
            one_way_cost_bps=DEFAULT_SPEC.base_one_way_cost_bps,
            bootstrap_seed=DEFAULT_SPEC.seeds[0] + policy_index * 100,
        )
        stress = replay_consensus_actions(
            dataset,
            actions,
            calibration_mask,
            policy_id=policy_id,
            role="calibration",
            scenario="stress",
            one_way_cost_bps=DEFAULT_SPEC.stress_one_way_cost_bps,
            bootstrap_seed=DEFAULT_SPEC.seeds[0] + policy_index * 100 + 1,
        )
        reasons = _calibration_reasons(
            stability=stability,
            stress=stress.metrics,
            artifacts=ensemble.artifacts,
        )
        gate = {"passed": not reasons, "reasons": reasons}
        calibration_reports[policy_id] = {
            "minimum_pairwise_seed_score_spearman": stability,
            "action_diagnostics": action_diagnostics,
            "base": dict(base.metrics),
            "stress": dict(stress.metrics),
            "gate": gate,
        }
        calibration_replays[policy_id] = (base, stress)
        calibration_rows.append(
            {
                "policy_id": policy_id,
                "minimum_pairwise_seed_score_spearman": stability,
                "base_total_net_return_fraction": base.metrics[
                    "total_net_return_fraction"
                ],
                "stress_total_net_return_fraction": stress.metrics[
                    "total_net_return_fraction"
                ],
                "stress_maximum_drawdown_fraction": stress.metrics[
                    "maximum_drawdown_fraction"
                ],
                "stress_profit_factor": stress.metrics["profit_factor"],
                "stress_closed_trades": stress.metrics["closed_trades"],
                "stress_active_days": stress.metrics["active_days"],
                "stress_bootstrap_lower_mean_hourly_bps": stress.metrics[
                    "bootstrap_mean_hourly_portfolio_bps"
                ]["lower_bps"],
                "gate_passed": not reasons,
                "gate_reasons": reasons,
            }
        )
        _write_replay_ledger(
            evidence_root / f"{policy_id}-calibration-ledger.csv.gz",
            stress,
        )
        progress(
            "calibration_policy",
            {
                "policy_id": policy_id,
                "passed": not reasons,
                "closed_trades": stress.metrics["closed_trades"],
                "stress_return": stress.metrics["total_net_return_fraction"],
                "stress_drawdown": stress.metrics["maximum_drawdown_fraction"],
                "reasons": len(reasons),
            },
        )
    _write_csv(evidence_root / "calibration_policies.csv", calibration_rows)
    passed = [
        policy_id
        for policy_id in POLICY_IDS
        if calibration_reports[policy_id]["gate"]["passed"]
    ]
    selected_policy: str | None = None
    if passed:
        selected_policy = min(
            passed,
            key=lambda policy_id: (
                _finite(
                    calibration_reports[policy_id]["stress"][
                        "maximum_drawdown_fraction"
                    ]
                ),
                -_finite(
                    calibration_reports[policy_id]["stress"][
                        "bootstrap_mean_hourly_portfolio_bps"
                    ]["lower_bps"]
                ),
            ),
        )
    evaluation_report: dict[str, object] | None = None
    evaluation_replays: tuple[object, object] | None = None
    if selected_policy is not None:
        evaluation_mask = role_mask(dataset, "evaluation")
        actions, action_diagnostics = consensus_policy_actions(
            ensemble.seed_q_bps,
            evaluation_mask,
            selected_policy,
        )
        base = replay_consensus_actions(
            dataset,
            actions,
            evaluation_mask,
            policy_id=selected_policy,
            role="evaluation",
            scenario="base",
            one_way_cost_bps=DEFAULT_SPEC.base_one_way_cost_bps,
            bootstrap_seed=DEFAULT_SPEC.seeds[0] + 10_000,
        )
        stress = replay_consensus_actions(
            dataset,
            actions,
            evaluation_mask,
            policy_id=selected_policy,
            role="evaluation",
            scenario="stress",
            one_way_cost_bps=DEFAULT_SPEC.stress_one_way_cost_bps,
            bootstrap_seed=DEFAULT_SPEC.seeds[0] + 10_001,
        )
        reasons = _evaluation_reasons(stress.metrics)
        evaluation_report = {
            "policy_id": selected_policy,
            "action_diagnostics": action_diagnostics,
            "base": dict(base.metrics),
            "stress": dict(stress.metrics),
            "gate": {"passed": not reasons, "reasons": reasons},
        }
        evaluation_replays = (base, stress)
        _write_replay_ledger(evidence_root / "evaluation-ledger.csv.gz", stress)
        progress(
            "evaluation",
            {
                "status": "complete",
                "policy_id": selected_policy,
                "gate_passed": not reasons,
                "closed_trades": stress.metrics["closed_trades"],
                "stress_return": stress.metrics["total_net_return_fraction"],
                "stress_drawdown": stress.metrics["maximum_drawdown_fraction"],
            },
        )
    else:
        progress(
            "evaluation",
            {"status": "withheld", "reason": "no_calibration_policy_passed"},
        )
    if selected_policy is None:
        status = "rejected_before_evaluation"
    elif evaluation_report and evaluation_report["gate"]["passed"]:
        status = "development_evaluation_passed_requires_untouched_confirmation"
    else:
        status = "rejected_after_consumed_evaluation"
    source_artifacts = [
        {
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in sorted(cache_paths.items())
    ]
    evidence_paths = [
        prediction_path,
        evidence_root / "models.csv",
        evidence_root / "training_history.csv",
        evidence_root / "roles.csv",
        evidence_root / "calibration_policies.csv",
        *sorted(evidence_root.glob("*-ledger.csv.gz")),
        *sorted((evidence_root / "models").glob("*.pt")),
    ]
    report: dict[str, object] = {
        "schema_version": "round-054-sequential-action-value-prototype-report-v1",
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "development_only": True,
        "selection_contaminated": True,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "leverage_applied": False,
        "evaluation_replay_performed": evaluation_report is not None,
        "design_sha256": design_sha,
        "implementation_commit": implementation_commit,
        "source_binding_sha256": str(binding.get("binding_sha256", "")),
        "dataset_sha256": dataset.dataset_sha256,
        "source_artifacts": source_artifacts,
        "dataset": {
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "symbols": list(SYMBOLS),
            "roles": role_rows,
        },
        "compute": {
            "backend_kind": ensemble.backend_kind,
            "backend_device": ensemble.backend_device,
            "preflight": dict(ensemble.preflight),
            "reward_scale_bps": ensemble.reward_scale_bps,
        },
        "models": [item.asdict() for item in ensemble.artifacts],
        "calibration": calibration_reports,
        "selected_policy": selected_policy,
        "evaluation": evaluation_report,
        "evidence_artifacts": [_artifact(path) for path in evidence_paths],
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "This prototype uses an already consumed development interval. Even a gate pass "
            "would require a new untouched chronological confirmation and cannot authorize "
            "leverage, testnet orders, live orders, or an AI-uplift claim."
        ),
    }
    del evaluation_replays, calibration_replays, design
    report["report_canonical_sha256"] = _canonical_sha256(report)
    write_json_atomic(
        evidence_root / "report.json",
        report,
        indent=2,
        sort_keys=True,
    )
    write_json_atomic(
        evidence_root / "status.json",
        {
            "round": ROUND,
            "status": status,
            "selected_policy": selected_policy,
            "evaluation_replay_performed": evaluation_report is not None,
            "report_canonical_sha256": report["report_canonical_sha256"],
        },
        indent=2,
        sort_keys=True,
    )
    progress(
        "complete",
        {
            "status": status,
            "report_canonical_sha256": report["report_canonical_sha256"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research
        / "round-054-sequential-distributional-action-value-prototype-design.json",
    )
    parser.add_argument(
        "--dataset-binding",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-binding.json",
    )
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend",
        choices=("directml", "cpu"),
        default="directml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
