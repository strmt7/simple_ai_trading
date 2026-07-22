"""Run the frozen Round 72 spot/perpetual predictive-skill screen."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Sequence

from simple_ai_trading.price_discovery_dataset import build_price_discovery_datasets
from simple_ai_trading.price_discovery_evaluation import (
    evaluate_price_discovery_primary,
)
from simple_ai_trading.price_discovery_model import run_price_discovery_models
from simple_ai_trading.progress_heartbeat import progress_heartbeat
from simple_ai_trading.spot_perpetual_corpus import (
    SpotPerpetualCorpusStore,
    load_frozen_round72_contract,
)
from simple_ai_trading.storage import write_json_atomic


DESIGN_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-price-discovery-design.json"
)
INVENTORY_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-inventory.json"
)
IMPLEMENTATION_DEFAULT = Path(
    "docs/model-research/action-value/round-072-price-discovery-implementation.json"
)
WAREHOUSE_DEFAULT = Path("data/microstructure.duckdb")
CORPUS_REPORT_DEFAULT = Path("data/round72-spot-perpetual-corpus-ingestion.json")
OUTPUT_DEFAULT = Path("data/round72-price-discovery-evaluation.json")
METRICS_DEFAULT = Path("data/round72-price-discovery-metrics.csv")
PROGRESS_DEFAULT = Path("data/round72-price-discovery.progress.json")


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
                "schema_version": "round-072-price-discovery-progress-v1",
                "round": 72,
                "sequence": self.sequence,
                "event": event,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
                "details": details,
            }
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            print(_canonical_json(payload), file=sys.stderr, flush=True)


def _load_corpus_report(
    path: Path,
    *,
    inventory_sha256: str,
    warehouse_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 72 corpus report is not an object")
    canonical = dict(value)
    observed_hash = str(canonical.pop("report_sha256", ""))
    certificate = value.get("corpus_certificate")
    reported_warehouse = Path(str(value.get("warehouse_path", ""))).resolve()
    if (
        observed_hash != _canonical_sha256(canonical)
        or value.get("schema_version")
        != "round-072-spot-perpetual-corpus-ingestion-v1"
        or value.get("status") != "complete"
        or value.get("inventory_sha256") != inventory_sha256
        or reported_warehouse != warehouse_path
        or value.get("completed_days") != 69
        or value.get("completed_files") != 414
        or value.get("completed_compressed_bytes") != 5_964_131_852
        or value.get("raw_aggregate_trades_retained") is not False
        or value.get("selected_archives_retained") is not False
        or value.get("profitability_claim") is not False
        or value.get("execution_or_fill_claim") is not False
        or value.get("trading_authority") is not False
        or not isinstance(certificate, dict)
        or certificate.get("inventory_sha256") != inventory_sha256
        or certificate.get("status") != "complete"
        or certificate.get("day_count") != 69
        or certificate.get("source_count") != 414
        or certificate.get("flow_rows") != 17_884_800
    ):
        raise ValueError("Round 72 corpus completion report is invalid")
    return value, dict(certificate)


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("duckdb", "lightgbm", "numba", "numpy", "scipy"):
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Round 72 metrics CSV cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("Round 72 metrics CSV columns differ")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _metrics_rows(report: dict[str, object]) -> list[dict[str, object]]:
    report_sha256 = str(report["report_sha256"])
    rows: list[dict[str, object]] = []
    for raw in report["layer_reports"]:
        layer = dict(raw)
        head = str(layer["head"])
        if head == "binary_direction":
            comparisons = layer["prevalence_comparison"]
            for metric in ("log_loss", "brier_score"):
                value = comparisons[metric]
                rows.append(
                    {
                        "record_type": "layer_vs_uninformed_control",
                        "symbol": layer["symbol"],
                        "horizon_seconds": layer["horizon_seconds"],
                        "feature_layer": layer["feature_layer"],
                        "head": head,
                        "metric": metric,
                        "model_value": value["model"],
                        "baseline_value": value["training_prevalence_baseline"],
                        "relative_improvement": value["relative_improvement"],
                        "q_value": "",
                        "passed": "",
                        "evaluation_report_sha256": report_sha256,
                    }
                )
        else:
            controls = layer["controls"]
            for metric in ("mean_squared_error", "mean_absolute_error"):
                value = controls[metric]
                rows.append(
                    {
                        "record_type": "layer_vs_uninformed_control",
                        "symbol": layer["symbol"],
                        "horizon_seconds": layer["horizon_seconds"],
                        "feature_layer": layer["feature_layer"],
                        "head": head,
                        "metric": metric,
                        "model_value": value["model"],
                        "baseline_value": value["zero_return"],
                        "relative_improvement": value["skill_vs_zero"],
                        "q_value": "",
                        "passed": "",
                        "evaluation_report_sha256": report_sha256,
                    }
                )
    for raw in report["feature_comparisons"]:
        comparison = dict(raw)
        rows.append(
            {
                "record_type": "spot_perpetual_vs_perpetual_only",
                "symbol": comparison["symbol"],
                "horizon_seconds": comparison["horizon_seconds"],
                "feature_layer": "spot_perpetual",
                "head": comparison["head"],
                "metric": comparison["metric"],
                "model_value": comparison["challenger_mean_loss"],
                "baseline_value": comparison["baseline_mean_loss"],
                "relative_improvement": comparison["relative_improvement"],
                "q_value": comparison["q_value"],
                "passed": comparison["passed"],
                "evaluation_report_sha256": report_sha256,
            }
        )
    return rows


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design_path = Path(arguments.design).resolve()
    inventory_path = Path(arguments.inventory).resolve()
    implementation_path = Path(arguments.implementation).resolve()
    warehouse_path = Path(arguments.warehouse).resolve()
    corpus_report_path = Path(arguments.corpus_report).resolve()
    output_path = Path(arguments.output).resolve()
    metrics_path = Path(arguments.metrics).resolve()
    progress_path = Path(arguments.progress).resolve()
    if len(
        {
            design_path,
            inventory_path,
            implementation_path,
            warehouse_path,
            corpus_report_path,
            output_path,
            metrics_path,
            progress_path,
        }
    ) != 8:
        raise ValueError("Round 72 screen paths must be distinct")
    if (
        not 1 <= int(arguments.threads) <= 32
        or not 1.0 <= float(arguments.heartbeat_seconds) <= 300.0
    ):
        raise ValueError("Round 72 screen runtime limits are invalid")
    contract = load_frozen_round72_contract(design_path, inventory_path)
    corpus_report, certificate = _load_corpus_report(
        corpus_report_path,
        inventory_sha256=contract.inventory_sha256,
        warehouse_path=warehouse_path,
    )
    progress = ProgressWriter(progress_path)
    progress(
        "round72_screen_started",
        warehouse=str(warehouse_path),
        corpus_report_sha256=corpus_report["report_sha256"],
        inventory_sha256=contract.inventory_sha256,
        compute_backend=str(arguments.compute_backend),
        memory_limit=str(arguments.memory_limit),
        threads=int(arguments.threads),
        package_versions=_package_versions(),
    )

    def forward(event: str, details: dict[str, object]) -> None:
        progress(event, **dict(details))

    with SpotPerpetualCorpusStore(
        warehouse_path,
        memory_limit=str(arguments.memory_limit),
        threads=int(arguments.threads),
        read_only=True,
    ) as store:
        with progress_heartbeat(
            progress,
            phase="round72_build_development_features",
            interval_seconds=float(arguments.heartbeat_seconds),
            details={
                "development_last_month": "2026-03",
                "terminal_months_excluded": ["2026-04", "2026-05", "2026-06"],
            },
        ):
            bundle = build_price_discovery_datasets(
                store,
                contract,
                implementation_path=implementation_path,
                progress=forward,
            )
    progress(
        "round72_development_features_complete",
        dataset_bundle_sha256=bundle.bundle_sha256,
        rows_by_symbol={value.symbol: value.rows for value in bundle.symbols},
        total_feature_bytes=bundle.total_feature_bytes,
        terminal_holdout_months_excluded=list(bundle.terminal_holdout_months_excluded),
    )
    with progress_heartbeat(
        progress,
        phase="round72_fit_primary_models",
        interval_seconds=float(arguments.heartbeat_seconds),
        details={"expected_models": 216},
    ):
        predictions = run_price_discovery_models(
            bundle,
            implementation_path=implementation_path,
            compute_backend=str(arguments.compute_backend),
            progress=forward,
        )
    progress(
        "round72_primary_models_complete",
        model_run_sha256=predictions.run_sha256,
        models=len(predictions.blocks),
        backend_kind=predictions.backend_kind,
        backend_device=predictions.backend_device,
    )
    with progress_heartbeat(
        progress,
        phase="round72_statistical_gate",
        interval_seconds=float(arguments.heartbeat_seconds),
        details={"permutation_draws": 10_000, "bootstrap_draws": 10_000},
    ):
        report = evaluate_price_discovery_primary(
            predictions,
            implementation_path=implementation_path,
            corpus_certificate=certificate,
            progress=forward,
        )
    write_json_atomic(output_path, report, indent=2, sort_keys=True)
    _write_csv_atomic(metrics_path, _metrics_rows(report))
    progress(
        "round72_screen_completed",
        report_sha256=report["report_sha256"],
        metrics_path=str(metrics_path),
        primary_gate_passed=report["primary_gate_passed"],
        decision=report["decision"],
        profitability_claim=False,
    )
    del predictions, bundle
    gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default=str(DESIGN_DEFAULT))
    parser.add_argument("--inventory", default=str(INVENTORY_DEFAULT))
    parser.add_argument("--implementation", default=str(IMPLEMENTATION_DEFAULT))
    parser.add_argument("--warehouse", default=str(WAREHOUSE_DEFAULT))
    parser.add_argument("--corpus-report", default=str(CORPUS_REPORT_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--metrics", default=str(METRICS_DEFAULT))
    parser.add_argument("--progress", default=str(PROGRESS_DEFAULT))
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        report = run(arguments)
    except KeyboardInterrupt:
        event = "round72_screen_interrupted"
        status = 130
        error = "KeyboardInterrupt"
    except Exception as exc:
        event = "round72_screen_failed"
        status = 1
        error = f"{type(exc).__name__}: {exc}"
    else:
        print(_canonical_json(report))
        return 0 if report["primary_gate_passed"] is True else 2
    payload = {
        "schema_version": "round-072-price-discovery-progress-v1",
        "round": 72,
        "event": event,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "total_elapsed_seconds": round(time.monotonic() - started, 3),
        "details": {"error": error, "evaluation_report_published": False},
    }
    try:
        write_json_atomic(arguments.progress, payload, indent=2, sort_keys=True)
    except Exception:
        pass
    print(_canonical_json(payload), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
