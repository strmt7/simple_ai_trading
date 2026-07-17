"""Run the preregistered Round 62 coarse depth-stress transition screen."""

from __future__ import annotations

import argparse
import calendar
from datetime import UTC, date, datetime, timedelta
import gc
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from simple_ai_trading.depth_stress_evaluation import (
    DEPTH_STRESS_EVALUATION_SYMBOLS,
    evaluate_depth_stress_symbol,
    finalize_depth_stress_gate,
)
from simple_ai_trading.depth_stress_model import orient_depth_stress_descriptors
from simple_ai_trading.depth_stress_screen import (
    DEPTH_STRESS_HORIZONS_SECONDS,
    DepthStressPanel,
    build_depth_stress_examples,
    utc_month_label,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.progress_heartbeat import progress_heartbeat
from simple_ai_trading.storage import write_json_atomic


ROUND = 62
DESIGN_DEFAULT = Path(
    "docs/model-research/action-value/round-062-depth-stress-transition-design.json"
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validated_design(path: Path) -> dict[str, object]:
    design = _read_object(path, "Round 62 design")
    canonical = dict(design)
    design_sha256 = str(canonical.pop("design_sha256", ""))
    if (
        design.get("round") != ROUND
        or design.get("schema_version") != "round-062-depth-stress-transition-design-v1"
        or design_sha256 != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 62 design hash or identity is invalid")
    governance = design.get("governance")
    source = design.get("source_contract")
    evaluation = design.get("evaluation_contract")
    if not all(isinstance(value, Mapping) for value in (governance, source, evaluation)):
        raise ValueError("Round 62 design contracts are missing")
    if (
        tuple(source.get("symbols", ())) != DEPTH_STRESS_EVALUATION_SYMBOLS
        or tuple(design["example_contract"].get("forecast_horizons_seconds", ()))
        != DEPTH_STRESS_HORIZONS_SECONDS
        or governance.get("profitability_claim_permitted") is not False
        or governance.get("trading_authority_permitted") is not False
        or governance.get("ai_evaluation_permitted") is not False
        or evaluation.get("same_frozen_challenger_must_pass_every_symbol_and_horizon")
        is not True
    ):
        raise ValueError("Round 62 frozen governance contract is invalid")
    inventory_path = Path(str(source.get("inventory_path", "")))
    if not inventory_path.is_file():
        raise ValueError("Round 62 official inventory is missing")
    inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    if inventory_sha256 != source.get("inventory_file_sha256"):
        raise ValueError("Round 62 official inventory hash is invalid")
    return design


def _complete_month_contract(design: Mapping[str, object]) -> tuple[np.ndarray, int, int]:
    source = design["source_contract"]
    available = source.get("available_period")
    if not isinstance(available, list) or len(available) != 2:
        raise ValueError("Round 62 available period is invalid")
    first_date = date.fromisoformat(str(available[0]))
    last_date = date.fromisoformat(str(available[1]))
    last_calendar_day = calendar.monthrange(last_date.year, last_date.month)[1]
    if last_date.day < last_calendar_day:
        last_date = last_date.replace(day=1) - timedelta(days=1)
    first_month = np.datetime64(first_date.strftime("%Y-%m"), "M").astype(np.int64)
    last_month = np.datetime64(last_date.strftime("%Y-%m"), "M").astype(np.int64)
    months = np.arange(first_month, last_month + 1, dtype=np.int64)
    required_start_ms = int(
        datetime(first_date.year, first_date.month, 1, tzinfo=UTC).timestamp() * 1_000
    )
    required_end_ms = int(
        (
            datetime(last_date.year, last_date.month, last_date.day, tzinfo=UTC)
            + timedelta(days=1)
        ).timestamp()
        * 1_000
        - 1
    )
    if len(months) < 8:
        raise ValueError("Round 62 has fewer than eight complete months")
    return months, required_start_ms, required_end_ms


def _unmasked_array(value: object, *, dtype: str, label: str) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{label} contains missing warehouse values")
    output = np.asarray(value, dtype=dtype)
    if output.ndim != 1 or not len(output):
        raise ValueError(f"{label} is empty or malformed")
    return output


def _load_panel(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    required_start_ms: int,
    required_end_ms: int,
    source_fingerprint: str,
) -> DepthStressPanel:
    columns = warehouse.connect().execute(
        """
        SELECT timestamp_ms, bid_depth_1, ask_depth_1,
               bid_notional_1, ask_notional_1,
               bid_notional_5, ask_notional_5
        FROM current_book_depth_snapshots
        WHERE symbol = ? AND timestamp_ms BETWEEN ? AND ?
        ORDER BY timestamp_ms
        """,
        [symbol, required_start_ms, required_end_ms],
    ).fetchnumpy()
    timestamps = _unmasked_array(
        columns["timestamp_ms"],
        dtype="<i8",
        label=f"{symbol} depth timestamps",
    )
    values = {
        name: _unmasked_array(
            columns[name],
            dtype="<f8",
            label=f"{symbol} {name}",
        )
        for name in (
            "bid_depth_1",
            "ask_depth_1",
            "bid_notional_1",
            "ask_notional_1",
            "bid_notional_5",
            "ask_notional_5",
        )
    }
    descriptors = orient_depth_stress_descriptors(
        bid_near_depth=values["bid_depth_1"],
        ask_near_depth=values["ask_depth_1"],
        bid_near_notional=values["bid_notional_1"],
        ask_near_notional=values["ask_notional_1"],
        bid_far_notional=values["bid_notional_5"],
        ask_far_notional=values["ask_notional_5"],
    )
    return DepthStressPanel(
        symbol=symbol,
        timestamp_ms=timestamps,
        descriptors=descriptors,
        source_fingerprint=source_fingerprint,
    )


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()
        self.sequence = 0
        self._lock = threading.Lock()

    def __call__(self, event: str, **details: object) -> None:
        with self._lock:
            self.sequence += 1
            payload = {
                "schema_version": "round-062-progress-v1",
                "round": ROUND,
                "sequence": self.sequence,
                "event": event,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
                "details": details,
            }
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            print(_canonical_json(payload), file=sys.stderr, flush=True)


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("duckdb", "lightgbm", "numpy"):
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design_path = Path(arguments.design).resolve()
    warehouse_path = Path(arguments.warehouse).resolve()
    output_path = Path(arguments.output).resolve()
    progress_path = Path(arguments.progress).resolve()
    inventory_path = Path(
        "docs/model-research/action-value/round-062-official-archive-inventory.json"
    ).resolve()
    if len({design_path, inventory_path, warehouse_path, output_path, progress_path}) != 5:
        raise ValueError(
            "design, inventory, warehouse, output, and progress paths must be distinct"
        )
    if (
        not 1 <= int(arguments.threads) <= 32
        or not 32 <= int(arguments.maximum_iterations) <= 2_048
        or not 100 <= int(arguments.permutation_draws) <= 1_000_000
        or not 1.0 <= float(arguments.heartbeat_seconds) <= 300.0
    ):
        raise ValueError("Round 62 runtime limits are invalid")
    design = _validated_design(design_path)
    months, required_start_ms, required_end_ms = _complete_month_contract(design)
    progress = ProgressWriter(progress_path)
    progress(
        "round62_started",
        warehouse=str(warehouse_path),
        output=str(output_path),
        symbols=list(DEPTH_STRESS_EVALUATION_SYMBOLS),
        horizons_seconds=list(DEPTH_STRESS_HORIZONS_SECONDS),
    )
    warehouse = MicrostructureWarehouse(
        warehouse_path,
        memory_limit=str(arguments.memory_limit),
        threads=int(arguments.threads),
        read_only=True,
    )
    symbol_reports: list[dict[str, object]] = []
    source_evidence: list[dict[str, object]] = []
    try:
        for symbol_index, symbol in enumerate(DEPTH_STRESS_EVALUATION_SYMBOLS, start=1):
            with progress_heartbeat(
                progress,
                phase="round62_load_panel",
                interval_seconds=float(arguments.heartbeat_seconds),
                details={
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "symbol_count": len(DEPTH_STRESS_EVALUATION_SYMBOLS),
                },
            ):
                certificate = warehouse.require_corpus_certificate(
                    symbol,
                    required_data_types=("bookDepth",),
                    required_start_ms=required_start_ms,
                    required_end_ms=required_end_ms,
                    require_full_history_inventory=True,
                    allow_official_gap_data_types=("bookDepth",),
                )
                certificate_sha256 = str(certificate["certificate_sha256"])
                panel = _load_panel(
                    warehouse,
                    symbol=symbol,
                    required_start_ms=required_start_ms,
                    required_end_ms=required_end_ms,
                    source_fingerprint=certificate_sha256,
                )
                examples = {
                    horizon: build_depth_stress_examples(panel, horizon_seconds=horizon)
                    for horizon in DEPTH_STRESS_HORIZONS_SECONDS
                }
            progress(
                "round62_panel_loaded",
                symbol=symbol,
                snapshots=len(panel.timestamp_ms),
                panel_sha256=panel.panel_sha256,
                examples={
                    str(horizon): len(examples[horizon].anchor_time_ms)
                    for horizon in DEPTH_STRESS_HORIZONS_SECONDS
                },
            )
            with progress_heartbeat(
                progress,
                phase="round62_evaluate_symbol",
                interval_seconds=float(arguments.heartbeat_seconds),
                details={
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "symbol_count": len(DEPTH_STRESS_EVALUATION_SYMBOLS),
                },
            ):
                symbol_report = evaluate_depth_stress_symbol(
                    panel,
                    examples,
                    eligible_month_ordinals=months,
                    compute_backend=str(arguments.compute_backend),
                    maximum_iterations=int(arguments.maximum_iterations),
                    permutation_draws=int(arguments.permutation_draws),
                    seed=int(arguments.seed),
                    progress=progress,
                )
            symbol_reports.append(symbol_report)
            source_evidence.append(
                {
                    "symbol": symbol,
                    "certificate": certificate,
                    "snapshots": len(panel.timestamp_ms),
                    "first_timestamp_ms": int(panel.timestamp_ms[0]),
                    "last_timestamp_ms": int(panel.timestamp_ms[-1]),
                    "panel_sha256": panel.panel_sha256,
                }
            )
            progress(
                "round62_symbol_completed",
                symbol=symbol,
                symbol_index=symbol_index,
                symbol_count=len(DEPTH_STRESS_EVALUATION_SYMBOLS),
            )
            del panel, examples, symbol_report
            gc.collect()
    finally:
        warehouse.close()
    gate = finalize_depth_stress_gate(
        symbol_reports,
        maximum_q_value=float(design["evaluation_contract"]["maximum_q_value"]),
        minimum_relative_improvement=float(
            design["evaluation_contract"]["minimum_relative_nll_improvement"]
        ),
    )
    public_symbol_reports = [
        {key: value for key, value in report.items() if key != "comparisons"}
        for report in symbol_reports
    ]
    report = {
        "schema_version": "round-062-depth-stress-transition-report-v1",
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design": {
            "path": str(Path(arguments.design).as_posix()),
            "file_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
            "canonical_sha256": design["design_sha256"],
        },
        "runtime": {
            "compute_backend_requested": str(arguments.compute_backend),
            "memory_limit": str(arguments.memory_limit),
            "threads": int(arguments.threads),
            "maximum_iterations": int(arguments.maximum_iterations),
            "permutation_draws": int(arguments.permutation_draws),
            "seed": int(arguments.seed),
            "package_versions": _package_versions(),
        },
        "eligible_period": {
            "first_month": utc_month_label(int(months[0])),
            "last_month": utc_month_label(int(months[-1])),
            "months": len(months),
            "required_start_ms": required_start_ms,
            "required_end_ms": required_end_ms,
        },
        "source_evidence": source_evidence,
        "symbol_reports": public_symbol_reports,
        "gate": gate,
        "profitability_claim": False,
        "trading_authority": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(output_path, report, indent=2, sort_keys=True)
    progress(
        "round62_completed",
        report_sha256=report["report_sha256"],
        gate_passed=gate["passed"],
        decision=gate["decision"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default="data/microstructure.duckdb")
    parser.add_argument("--design", default=str(DESIGN_DEFAULT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--maximum-iterations", type=int, default=256)
    parser.add_argument("--permutation-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        report = run(arguments)
    except Exception as exc:
        progress_path = Path(arguments.progress)
        payload = {
            "schema_version": "round-062-progress-v1",
            "round": ROUND,
            "event": "round62_failed",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "total_elapsed_seconds": round(time.monotonic() - started, 3),
            "details": {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "report_published": False,
            },
        }
        try:
            write_json_atomic(progress_path, payload, indent=2, sort_keys=True)
        except Exception:
            pass
        print(_canonical_json(payload), file=sys.stderr)
        return 1
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
