"""Diagnose Round 32 calibration rejection without training or later-stage access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    derive_action_scores,
)
from simple_ai_trading.microstructure_architecture import _auc  # noqa: E402
from simple_ai_trading.microstructure_shared_action_lightgbm import (  # noqa: E402
    SharedActionEnsembleBatch,
    load_shared_action_lightgbm_model,
)
from simple_ai_trading.microstructure_shared_action_policy import (  # noqa: E402
    derive_shared_action_scores,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_shared_action_viability import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    _canonical_sha256,
    _is_sha256,
    _load_corpus,
    _parse_date,
    _predict_ensemble,
    _profile_spec,
    _role_indexes,
    _sha256_file,
    _target_positions,
    load_round32_design,
)


DIAGNOSTIC_SCHEMA_VERSION = "round-032-selective-action-diagnostic-v1"
_FALSE_CLAIMS = {
    "trading_authority": False,
    "execution_claim": False,
    "profitability_claim": False,
    "portfolio_claim": False,
    "leverage_applied": False,
}


def _validated_round32_report(path: Path, *, design_sha256: str) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 32 source report is unreadable") from exc
    if not isinstance(report, dict):
        raise ValueError("Round 32 source report must be an object")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 32
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha256
        or not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("stage_access")
        != {
            "calibration_prediction": True,
            "development_prediction": False,
            "distant_confirmation_prediction": False,
            "later_stage_predictions_withheld_until_prior_stage_passes": True,
            "policy_prediction": False,
        }
    ):
        raise ValueError("Round 32 source report identity is invalid")
    if any(report.get(name) is not False for name in _FALSE_CLAIMS):
        raise ValueError("Round 32 source report contains an authority claim")
    return report


def _load_models(
    evidence_root: Path,
    report: Mapping[str, object],
) -> list[object]:
    models: list[object] = []
    for evidence in report["ensemble_models"]:
        if not isinstance(evidence, Mapping):
            raise ValueError("Round 32 model evidence is invalid")
        relative = Path(str(evidence.get("artifact_path") or ""))
        path = evidence_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != int(evidence.get("artifact_bytes") or -1)
            or _sha256_file(path) != evidence.get("artifact_sha256")
        ):
            raise ValueError("Round 32 model artifact differs from its report")
        model = load_shared_action_lightgbm_model(path)
        if (
            model.model_sha256 != evidence.get("model_sha256")
            or model.seed != evidence.get("seed")
            or model.backend_kind != "opencl"
        ):
            raise ValueError("Round 32 model identity differs from its report")
        models.append(model)
    if len(models) != 3:
        raise ValueError("Round 32 ensemble is incomplete")
    return models


def _side_arrays(
    ensemble: SharedActionEnsembleBatch,
    *,
    epistemic_penalty: float,
) -> dict[str, np.ndarray]:
    action = ensemble.action_values
    long_strength = np.asarray(action.long_mean_bps) - epistemic_penalty * np.asarray(
        action.long_epistemic_std_bps
    )
    short_strength = np.asarray(action.short_mean_bps) - epistemic_penalty * np.asarray(
        action.short_epistemic_std_bps
    )
    prefer_long = long_strength >= short_strength

    def choose(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
        return np.where(prefer_long, long_values, short_values)

    return {
        "side": np.where(prefer_long, 1, -1).astype(np.int8),
        "strength": choose(long_strength, short_strength),
        "probability": choose(
            np.asarray(action.long_profitable_probability),
            np.asarray(action.short_profitable_probability),
        ),
        "member_agreement": choose(
            np.asarray(action.long_positive_member_ratio),
            np.asarray(action.short_positive_member_ratio),
        ),
        "epistemic_std": choose(
            np.asarray(action.long_epistemic_std_bps),
            np.asarray(action.short_epistemic_std_bps),
        ),
        "lower_bound": choose(
            np.asarray(action.long_lower_bps),
            np.asarray(action.short_lower_bps),
        ),
    }


def _gate_breakdown(
    ensemble: SharedActionEnsembleBatch,
    raw_profile: Mapping[str, object],
) -> dict[str, object]:
    spec = _profile_spec(raw_profile)
    arrays = _side_arrays(ensemble, epistemic_penalty=spec.epistemic_penalty)
    action_value_positive = arrays["strength"] > 0.0
    probability = arrays["probability"] >= spec.minimum_profitable_probability
    action_agreement = arrays["member_agreement"] >= spec.minimum_member_agreement
    action_uncertainty = arrays["epistemic_std"] <= spec.maximum_epistemic_std_bps
    lower_tail = arrays["lower_bound"] >= spec.minimum_lower_bound_bps
    advantage = np.asarray(ensemble.signed_advantage_mean_bps)
    advantage_std = np.asarray(ensemble.signed_advantage_epistemic_std_bps)
    advantage_side = np.sign(advantage).astype(np.int8)
    direction_matches = arrays["side"] == advantage_side
    direction_strength = (
        np.abs(advantage) - spec.epistemic_penalty * advantage_std
    ) > 0.0
    direction_uncertainty = advantage_std <= spec.maximum_epistemic_std_bps
    direction_agreement_values = np.where(
        arrays["side"] == 1,
        np.asarray(ensemble.advantage_long_member_ratio),
        np.asarray(ensemble.advantage_short_member_ratio),
    )
    direction_agreement = (
        direction_agreement_values >= spec.minimum_member_agreement
    )
    head_consensus = (
        np.asarray(ensemble.side_consensus_member_ratio)
        >= spec.minimum_member_agreement
    )
    masks = {
        "action_epistemic_adjusted_value_positive": action_value_positive,
        "profitable_probability": probability,
        "action_member_agreement": action_agreement,
        "action_uncertainty": action_uncertainty,
        "lower_tail": lower_tail,
        "direction_matches_action": direction_matches,
        "direction_epistemic_adjusted_strength_positive": direction_strength,
        "direction_uncertainty": direction_uncertainty,
        "direction_member_agreement": direction_agreement,
        "cross_head_member_consensus": head_consensus,
    }
    sequential: dict[str, int] = {}
    active = np.ones(ensemble.rows, dtype=bool)
    for name, mask in masks.items():
        active &= mask
        sequential[name] = int(np.sum(active))
    base = derive_action_scores(ensemble.action_values, spec)
    final = derive_shared_action_scores(ensemble, spec)
    if int(np.sum(active)) != int(np.sum(final.eligible)):
        raise ValueError("Round 32 reconstructed gate sequence differs")
    return {
        "profile": spec.profile,
        "rows": ensemble.rows,
        "individual_pass_counts": {
            name: int(np.sum(mask)) for name, mask in masks.items()
        },
        "sequential_pass_counts": sequential,
        "base_action_value_eligible_rows": int(np.sum(base.eligible)),
        "final_shared_action_eligible_rows": int(np.sum(final.eligible)),
    }


def _top_rows(
    *,
    name: str,
    score: np.ndarray,
    side: np.ndarray,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
) -> dict[str, object]:
    selected_actual = np.where(side == 1, long_actual, short_actual)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rows: list[dict[str, object]] = []
    for requested in (100, 500, 1_000, 5_000):
        count = min(requested, len(order))
        chosen = order[:count]
        rows.append(
            {
                "requested_rows": requested,
                "actual_rows": count,
                "mean_selected_stress_net_bps": float(
                    np.mean(selected_actual[chosen])
                ),
                "positive_ratio": float(np.mean(selected_actual[chosen] > 0.0)),
                "long_share": float(np.mean(side[chosen] == 1)),
            }
        )
    return {"ranking": name, "top_rows": rows}


def _opportunity_diagnostics(
    ensemble: SharedActionEnsembleBatch,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
) -> dict[str, object]:
    action = ensemble.action_values
    any_profitable = np.maximum(long_actual, short_actual) > 0.0
    both_profitable = (long_actual > 0.0) & (short_actual > 0.0)
    long_better = long_actual > short_actual
    opportunity_probability = np.maximum(
        np.asarray(action.long_profitable_probability),
        np.asarray(action.short_profitable_probability),
    )
    action_difference = np.asarray(action.long_mean_bps) - np.asarray(
        action.short_mean_bps
    )
    advantage = np.asarray(ensemble.signed_advantage_mean_bps)
    action_side = np.where(action_difference >= 0.0, 1, -1).astype(np.int8)
    advantage_side = np.where(advantage >= 0.0, 1, -1).astype(np.int8)
    action_score = np.maximum(
        np.asarray(action.long_mean_bps)
        - np.asarray(action.long_epistemic_std_bps),
        np.asarray(action.short_mean_bps)
        - np.asarray(action.short_epistemic_std_bps),
    )
    opportunity_rows = np.flatnonzero(any_profitable)
    direction_labels = long_better[opportunity_rows].astype(np.int8)
    return {
        "rows": len(long_actual),
        "any_profitable_rows": int(np.sum(any_profitable)),
        "any_profitable_ratio": float(np.mean(any_profitable)),
        "both_profitable_rows": int(np.sum(both_profitable)),
        "long_only_profitable_rows": int(
            np.sum((long_actual > 0.0) & (short_actual <= 0.0))
        ),
        "short_only_profitable_rows": int(
            np.sum((short_actual > 0.0) & (long_actual <= 0.0))
        ),
        "neither_profitable_rows": int(
            np.sum((long_actual <= 0.0) & (short_actual <= 0.0))
        ),
        "opportunity_probability_auc": _auc(
            any_profitable.astype(np.int8), opportunity_probability
        ),
        "direction_auc_all_rows": _auc(long_better.astype(np.int8), advantage),
        "direction_auc_profitable_opportunity_rows": _auc(
            direction_labels,
            advantage[opportunity_rows],
        ),
        "action_value_direction_auc_profitable_opportunity_rows": _auc(
            direction_labels,
            action_difference[opportunity_rows],
        ),
        "action_advantage_side_agreement": float(
            np.mean(action_side == advantage_side)
        ),
        "rankings": [
            _top_rows(
                name="action_value_risk_adjusted",
                score=action_score,
                side=action_side,
                long_actual=long_actual,
                short_actual=short_actual,
            ),
            _top_rows(
                name="opportunity_probability_with_action_value_side",
                score=opportunity_probability,
                side=action_side,
                long_actual=long_actual,
                short_actual=short_actual,
            ),
            _top_rows(
                name="opportunity_probability_with_advantage_side",
                score=opportunity_probability,
                side=advantage_side,
                long_actual=long_actual,
                short_actual=short_actual,
            ),
        ],
    }


def run_diagnostic(
    *,
    design_path: Path,
    evidence_root: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
) -> dict[str, object]:
    design, design_sha, profiles = load_round32_design(design_path)
    source_report = _validated_round32_report(
        evidence_root / "report.json", design_sha256=design_sha
    )
    status_path = output_path.with_suffix(".status.json")
    progress_lock = threading.Lock()
    sequence = 0
    started = time.monotonic()

    def progress(phase: str, **details: object) -> None:
        nonlocal sequence
        with progress_lock:
            sequence += 1
            value = {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "sequence": sequence,
                "phase": phase,
                "run_elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(
                "round32-diagnostic "
                + " ".join(f"{name}={item}" for name, item in value.items()),
                flush=True,
            )
            write_json_atomic(status_path, value, indent=2, sort_keys=True)

    roles = design["data"]["roles"]
    resources = design["runtime_resources"]
    assert isinstance(roles, Mapping)
    assert isinstance(resources, Mapping)
    training = _load_corpus(
        name="round32_calibration_diagnostic",
        design=design,
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        first=_parse_date(roles["train"]["start"], label="training start"),
        last=_parse_date(roles["development"]["end"], label="training end"),
        evaluation_first=_parse_date(
            roles["train"]["start"], label="evaluation start"
        ),
        evaluation_last=_parse_date(
            roles["development"]["end"], label="evaluation end"
        ),
        memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
        threads=min(int(resources["maximum_worker_threads"]), 16),
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    indexes, role_evidence, _maximum_exit = _role_indexes(
        training,
        roles,
        ("train", "early_stop", "calibration", "policy", "development"),
    )
    models = _load_models(evidence_root, source_report)
    ensemble = _predict_ensemble(
        models,
        training.dataset,
        indexes["calibration"],
        role="calibration_diagnostic",
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    positions = _target_positions(training.targets, ensemble.endpoint_indexes)
    long_actual = np.asarray(
        training.targets.stress_long_net_bps[positions], dtype=np.float64
    )
    short_actual = np.asarray(
        training.targets.stress_short_net_bps[positions], dtype=np.float64
    )
    result: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "round": 32,
        "diagnostic_sha256": "PENDING",
        "source_design_sha256": design_sha,
        "source_report_canonical_sha256": source_report[
            "report_canonical_sha256"
        ],
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_corpus_certificate_sha256": training.source_certificate[
            "certificate_sha256"
        ],
        "source_cache_key": training.cache_key,
        "source_cache_state": training.cache_state,
        "barrier_targets_sha256": training.targets_sha256,
        "calibration_role": role_evidence["calibration"],
        "model_sha256": [model.model_sha256 for model in models],
        "scope": {
            "training_performed": False,
            "thresholds_changed": False,
            "policy_prediction_accessed": False,
            "development_prediction_accessed": False,
            "distant_confirmation_accessed": False,
            "untouched_dates_accessed": False,
        },
        "gate_breakdown": [
            _gate_breakdown(ensemble, profile) for profile in profiles
        ],
        "opportunity_diagnostics": _opportunity_diagnostics(
            ensemble, long_actual, short_actual
        ),
        **_FALSE_CLAIMS,
    }
    canonical = dict(result)
    canonical.pop("diagnostic_sha256")
    result["diagnostic_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    progress(
        "complete",
        diagnostic_sha256=result["diagnostic_sha256"],
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose the rejected Round 32 calibration model."
    )
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-032-shared-action-value-viability-design.json",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_diagnostic(
        design_path=arguments.design,
        evidence_root=arguments.evidence_root,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_path=arguments.output,
    )
    print(
        f"Round 32 diagnostic complete sha256={result['diagnostic_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
