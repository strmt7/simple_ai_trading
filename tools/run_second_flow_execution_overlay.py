"""Run the hash-bound Round 42 one-second execution-timing pilot."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import os
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

from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.second_flow_data import (  # noqa: E402
    SYMBOLS,
    canonical_sha256,
    file_sha256,
    load_verified_second_flow,
    validate_source_certificate,
)
from simple_ai_trading.second_flow_execution_model import (  # noqa: E402
    BASE_CHARGE_BPS,
    DELAYS_SECONDS,
    EXPECTED_NET_GRID,
    FOLDS,
    HORIZON_SECONDS,
    LOWER_QUARTILE_GRID,
    MAXIMUM_ENTRIES_PER_SYMBOL_DAY,
    PRIMARY_MARGIN,
    PROBABILITY_GRID,
    STRESS_CHARGE_BPS,
    TimingDataset,
    build_timing_dataset,
    run_round42_screen,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_causal_meta_label_capacity_ai import (  # noqa: E402
    SOURCE_CERTIFICATE_CANONICAL_SHA256,
    _canonical_json,
    _canonical_sha256,
    _git,
    _memory_evidence,
    _read_object,
)


ROUND = 42
DESIGN_SCHEMA = "second-flow-execution-overlay-design-v1"
BINDING_SCHEMA = "round-042-second-flow-execution-binding-v1"
REPORT_SCHEMA = "second-flow-execution-overlay-report-v1"
ROUND41_REPORT_SCHEMA = "prequential-meta-label-ai-report-v1"
ROUND41_REPORT_CANONICAL_SHA256 = (
    "718d75fdebf278f359a16dea9cf3b8e6606e49a420b9f8dfd2e262e9173522d7"
)


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 42 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 42 design identity is invalid")
    governance = design.get("governance")
    source = design.get("source_contract")
    proposal = design.get("frozen_primary_proposal")
    outcome = design.get("delay_option_and_outcome_contract")
    model = design.get("overlay_model")
    walk = design.get("walk_forward_contract")
    ai = design.get("ai_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (governance, source, proposal, outcome, model, walk, ai)
    ):
        raise ValueError("Round 42 design sections are incomplete")
    for field in (
        "unregistered_hyperparameter_search_permitted",
        "round_41_primary_side_or_probability_change_permitted",
        "future_delay_or_outcome_feature_use_permitted",
        "oracle_delay_use_for_training_selection_or_claims_permitted",
        "maker_execution_assumption_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "historical_data_expansion_before_pilot_gate_permitted",
        "ai_inference_during_pilot_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "leverage_permitted",
        "profitability_portfolio_roi_or_drawdown_claim_permitted",
        "selection_confirmation_or_terminal_2026_access_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 42 governance must deny {field}")
    if (
        source.get("symbols") != list(SYMBOLS)
        or source.get("second_rows_total") != 1_814_400
        or proposal.get("minimum_direction_probability_margin") != PRIMARY_MARGIN
        or proposal.get("holding_horizon_seconds") != HORIZON_SECONDS
        or outcome.get("entry_delay_seconds") != list(DELAYS_SECONDS)
        or outcome.get("base_round_trip_charge_bps") != BASE_CHARGE_BPS
        or outcome.get("stress_round_trip_charge_bps") != STRESS_CHARGE_BPS
        or outcome.get("maximum_entries_per_symbol_per_utc_day")
        != MAXIMUM_ENTRIES_PER_SYMBOL_DAY
        or model.get("models_per_walk_forward_fold") != 3
        or walk.get("threshold_probability_grid") != list(PROBABILITY_GRID)
        or walk.get("threshold_expected_net_bps_grid") != list(EXPECTED_NET_GRID)
        or walk.get("threshold_lower_quartile_bps_grid") != list(LOWER_QUARTILE_GRID)
        or walk.get("threshold_cells_total") != 54
        or len(walk.get("folds", [])) != len(FOLDS)
        or ai.get("pilot_ai_cases") != 0
    ):
        raise ValueError("Round 42 implementation and design contracts differ")
    return design, claimed


def _validate_report_identity(
    path: Path,
    *,
    schema: str,
    round_number: int,
    hash_field: str,
    expected_canonical_sha256: str,
    label: str,
) -> dict[str, object]:
    report = _read_object(path, label)
    canonical = dict(report)
    claimed = str(canonical.pop(hash_field, ""))
    if (
        report.get("schema_version") != schema
        or report.get("round") != round_number
        or claimed != expected_canonical_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError(f"{label} identity is invalid")
    return report


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    second_flow_certificate_path: Path,
    minute_source_certificate_path: Path,
    round41_report_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 42 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 42 execution binding identity is invalid")
    certificates = binding.get("source_certificates")
    predecessor = binding.get("round41_predecessor")
    if not isinstance(certificates, Mapping) or not isinstance(predecessor, Mapping):
        raise ValueError("Round 42 bound dependencies are missing")
    second = certificates.get("second_flow")
    minute = certificates.get("minute_derivatives")
    if not isinstance(second, Mapping) or not isinstance(minute, Mapping):
        raise ValueError("Round 42 source bindings are missing")
    second_certificate = _read_object(
        second_flow_certificate_path, "Round 42 one-second source certificate"
    )
    if (
        second.get("canonical_sha256") != second_certificate.get("certificate_sha256")
        or second.get("file_sha256") != file_sha256(second_flow_certificate_path)
        or minute.get("canonical_sha256") != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or minute.get("file_sha256") != file_sha256(minute_source_certificate_path)
        or predecessor.get("canonical_sha256") != ROUND41_REPORT_CANONICAL_SHA256
        or predecessor.get("file_sha256") != file_sha256(round41_report_path)
    ):
        raise ValueError("Round 42 bound source or predecessor identity drifted")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 42 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 42 execution requires a clean worktree")
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
        raise ValueError("Round 42 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 42 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 42 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}")
            != expected_oid
            or _git("rev-parse", f"HEAD:{relative_path}") != expected_oid
        ):
            raise ValueError(f"Round 42 bound blob changed: {relative_path}")
    return binding, claimed, implementation_commit


def _array_sha256(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"round-042-array-v1")
    digest.update(name.encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _proposal_counts(dataset: TimingDataset) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    proposal_day = dataset.proposal_day
    symbol_index = dataset.proposal_symbol_index
    for day in np.unique(proposal_day):
        mask = proposal_day == day
        evidence.append(
            {
                "utc_day": str(np.datetime64(int(day), "D")),
                "total": int(np.count_nonzero(mask)),
                "by_symbol": {
                    symbol: int(np.count_nonzero(mask & (symbol_index == index)))
                    for index, symbol in enumerate(SYMBOLS)
                },
            }
        )
    return evidence


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.last_write = 0.0
        self.sequence = 0

    def __call__(self, phase: str, detail: Mapping[str, object]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "round-042-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "phase": phase,
            "detail": dict(detail),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        print(_canonical_json(payload), flush=True)
        now = time.monotonic()
        status_changed = str(detail.get("status") or "") in {
            "started",
            "complete",
        }
        if status_changed or now - self.last_write >= 30.0:
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            self.last_write = now


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    _, design_sha = _validate_design(arguments.design.resolve())
    round41_report = _validate_report_identity(
        arguments.round41_report.resolve(),
        schema=ROUND41_REPORT_SCHEMA,
        round_number=41,
        hash_field="report_canonical_sha256",
        expected_canonical_sha256=ROUND41_REPORT_CANONICAL_SHA256,
        label="Round 41 predecessor report",
    )
    _, binding_sha, implementation_commit = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        second_flow_certificate_path=arguments.second_flow_certificate.resolve(),
        minute_source_certificate_path=arguments.minute_source_certificate.resolve(),
        round41_report_path=arguments.round41_report.resolve(),
    )
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
        },
    )
    second_flow, current_second_certificate = load_verified_second_flow(
        arguments.database.resolve()
    )
    supplied_second_certificate = _read_object(
        arguments.second_flow_certificate.resolve(),
        "Round 42 one-second source certificate",
    )
    validate_source_certificate(supplied_second_certificate, current_second_certificate)
    progress(
        "second_flow_source",
        {
            "status": "complete",
            "certificate_sha256": current_second_certificate["certificate_sha256"],
            "rows_total": current_second_certificate["rows_total"],
            "raw_aggregate_trade_rows_total": current_second_certificate[
                "raw_aggregate_trade_rows_total"
            ],
        },
    )
    panel, price_source = load_verified_minute_panel(
        arguments.database.resolve(), progress=progress
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.minute_source_certificate.resolve(),
        progress=progress,
    )
    minute_dataset = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=progress,
    )
    del panel, premium
    timing_dataset = build_timing_dataset(
        minute_dataset,
        second_flow,
        funding,
        round41_report=round41_report,
        round41_evidence_root=arguments.round41_evidence_root.resolve(),
        progress=progress,
    )
    progress(
        "timing_dataset",
        {
            "status": "complete",
            "proposals": timing_dataset.proposals,
            "option_rows": timing_dataset.option_rows,
            "feature_count": len(timing_dataset.feature_names),
            "matrix_bytes": int(timing_dataset.features.nbytes),
        },
    )
    del funding, second_flow
    screen = run_round42_screen(
        timing_dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    status = (
        "pilot_gate_observed_data_expansion_only"
        if screen.pilot_gate_passed
        else "rejected"
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": status,
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "backend": {
            "kind": screen.backend_kind,
            "device": screen.backend_device,
            "gpu_first_requested": arguments.compute_backend != "cpu",
            "python_dependencies": {
                package: metadata.version(package)
                for package in ("lightgbm", "numpy", "scipy")
            },
        },
        "source_evidence": {
            "second_flow": current_second_certificate,
            "minute_and_derivatives": minute_dataset.source_evidence.asdict(),
            "round41_predecessor": {
                "report_path": str(arguments.round41_report.resolve()),
                "report_canonical_sha256": ROUND41_REPORT_CANONICAL_SHA256,
                "report_file_sha256": file_sha256(arguments.round41_report.resolve()),
                "evidence_root": str(arguments.round41_evidence_root.resolve()),
                "primary_artifacts": list(timing_dataset.primary_artifacts),
            },
        },
        "dataset": {
            "proposals": timing_dataset.proposals,
            "option_rows": timing_dataset.option_rows,
            "feature_count": len(timing_dataset.feature_names),
            "feature_names": list(timing_dataset.feature_names),
            "feature_names_sha256": canonical_sha256(
                list(timing_dataset.feature_names)
            ),
            "features_dtype": str(timing_dataset.features.dtype),
            "features_bytes": int(timing_dataset.features.nbytes),
            "features_sha256": _array_sha256("features", timing_dataset.features),
            "base_outcomes_sha256": _array_sha256(
                "base_outcomes", timing_dataset.option_base_net_bps
            ),
            "stress_outcomes_sha256": _array_sha256(
                "stress_outcomes", timing_dataset.option_stress_net_bps
            ),
            "proposal_source_indices_sha256": _array_sha256(
                "proposal_source_indices", timing_dataset.proposal_source_index
            ),
            "proposal_exclusions": dict(timing_dataset.proposal_exclusions),
            "proposal_counts_by_utc_day": _proposal_counts(timing_dataset),
            "entry_delays_seconds": list(DELAYS_SECONDS),
            "horizon_seconds": HORIZON_SECONDS,
            "base_round_trip_charge_bps": BASE_CHARGE_BPS,
            "stress_round_trip_charge_bps": STRESS_CHARGE_BPS,
            "persistent_feature_prediction_or_raw_trade_copy_created": False,
        },
        "folds": list(screen.folds),
        "aggregate": dict(screen.aggregate),
        "model_artifacts": [artifact.asdict() for artifact in screen.model_artifacts],
        "pilot_gate_passed": screen.pilot_gate_passed,
        "pilot_gate_reasons": list(screen.pilot_gate_reasons),
        "ai_ablation": {
            "cases": 0,
            "model_loaded": False,
            "reason": (
                "An 8B language model is not admitted to a seconds-scale execution "
                "loop, and two evaluation days cannot establish AI uplift."
            ),
            "future_authority_if_separately_validated": "asynchronous veto only",
        },
        "selection_contaminated": True,
        "development_only": True,
        "data_expansion_authorized": screen.pilot_gate_passed,
        "selection_confirmation_accessed": False,
        "terminal_2026_accessed": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "roi_claim": False,
        "drawdown_claim": False,
        "leverage_applied": False,
        "ai_uplift_claim": False,
        "runtime_evidence": {
            "elapsed_seconds": time.perf_counter() - started,
            "logical_cpu_count": os.cpu_count(),
            "memory": _memory_evidence(),
        },
        "report_canonical_sha256": "PENDING",
    }
    canonical = dict(report)
    canonical.pop("report_canonical_sha256")
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    progress(
        "report",
        {
            "status": "complete",
            "report_canonical_sha256": report["report_canonical_sha256"],
            "pilot_gate_passed": screen.pilot_gate_passed,
            "pilot_gate_reasons": list(screen.pilot_gate_reasons),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-042-second-flow-execution-overlay-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-042-second-flow-execution-binding.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--second-flow-certificate", type=Path, required=True)
    parser.add_argument("--minute-source-certificate", type=Path, required=True)
    parser.add_argument("--round41-report", type=Path, required=True)
    parser.add_argument("--round41-evidence-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(
        _canonical_json(
            {
                "round": ROUND,
                "status": report["status"],
                "pilot_gate_passed": report["pilot_gate_passed"],
                "pilot_gate_reasons": report["pilot_gate_reasons"],
                "report_canonical_sha256": report["report_canonical_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
