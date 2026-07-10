"""Promote verified foundation benchmark evidence into the fixed latest-docs path."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from simple_ai_trading.foundation_benchmark import (
    FOUNDATION_BENCHMARK_VERSION,
    FOUNDATION_SELECTION_END_EXCLUSIVE_MS,
    FOUNDATION_SELECTION_START_MS,
    FOUNDATION_SYMBOLS,
    ForecastObservation,
)


_PROMOTED_FILES = frozenset(
    {"README.md", "benchmark.svg", "observations.csv", "report.json", "manifest.json"}
)
_HOST_PATH_PATTERN = re.compile(
    r"(?i)(?<![a-z])[a-z]:/|(?<![a-z0-9:])/(?:Users|home|tmp)/|(?<!:)//[^/\s]+/"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_values(value: object, path: str = "report") -> Iterator[tuple[str, object]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_values(child, f"{path}[{index}]")


def _assert_finite_numbers(payload: object) -> None:
    for path, value in _walk_values(payload):
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"foundation report contains a non-finite number at {path}")


def _assert_no_host_paths(payload: object) -> None:
    for path, value in _walk_values(payload):
        if not isinstance(value, str):
            continue
        normalized = value.replace("\\", "/")
        if _HOST_PATH_PATTERN.search(normalized):
            raise RuntimeError(f"promoted report leaks a host-local path at {path}")


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"foundation observation row {row_number} has invalid {field}")


def _validate_observations(path: Path, payload: dict[str, object]) -> None:
    expected_fields = tuple(ForecastObservation.__dataclass_fields__)
    seen: set[tuple[str, int]] = set()
    count = 0
    symbols: set[str] = set()
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("foundation report config is missing")
    start_ms = int(config.get("start_ms", -1))
    end_exclusive_ms = int(config.get("end_exclusive_ms", -1))
    if (
        start_ms < FOUNDATION_SELECTION_START_MS
        or end_exclusive_ms > FOUNDATION_SELECTION_END_EXCLUSIVE_MS
        or end_exclusive_ms <= start_ms
    ):
        raise RuntimeError("foundation report observation window violates the sealed-period contract")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError("foundation observation CSV schema does not match the benchmark contract")
        for row_number, row in enumerate(reader, start=2):
            count += 1
            symbol = str(row["symbol"])
            if symbol not in FOUNDATION_SYMBOLS:
                raise RuntimeError(f"foundation observation row {row_number} has an invalid symbol")
            decision_ms = int(row["decision_ms"])
            if not start_ms <= decision_ms < end_exclusive_ms:
                raise RuntimeError(f"foundation observation row {row_number} is outside the report window")
            key = (symbol, decision_ms)
            if key in seen:
                raise RuntimeError(f"foundation observation row {row_number} duplicates symbol/time")
            seen.add(key)
            symbols.add(symbol)
            parsed_time = datetime.fromisoformat(row["decision_time_utc"].replace("Z", "+00:00"))
            if parsed_time.tzinfo is None or int(parsed_time.astimezone(UTC).timestamp() * 1_000) != decision_ms:
                raise RuntimeError(f"foundation observation row {row_number} has mismatched UTC time")
            numeric = {
                field: float(row[field])
                for field in (
                    "last_close",
                    "predicted_average_return",
                    "actual_average_return",
                    "predicted_final_return",
                    "actual_final_return",
                    "absolute_error",
                    "random_walk_absolute_error",
                )
            }
            if not all(math.isfinite(value) for value in numeric.values()):
                raise RuntimeError(f"foundation observation row {row_number} contains non-finite data")
            if numeric["last_close"] <= 0.0:
                raise RuntimeError(f"foundation observation row {row_number} has a nonpositive close")
            expected_error = abs(
                numeric["predicted_average_return"] - numeric["actual_average_return"]
            )
            expected_baseline = abs(numeric["actual_average_return"])
            if not math.isclose(numeric["absolute_error"], expected_error, abs_tol=1e-15):
                raise RuntimeError(f"foundation observation row {row_number} has inconsistent error")
            if not math.isclose(
                numeric["random_walk_absolute_error"], expected_baseline, abs_tol=1e-15
            ):
                raise RuntimeError(
                    f"foundation observation row {row_number} has inconsistent baseline error"
                )
            direction = _parse_bool(
                row["direction_correct"], field="direction_correct", row_number=row_number
            )
            expected_direction = (
                numeric["predicted_average_return"] != 0.0
                and numeric["actual_average_return"] != 0.0
                and math.copysign(1.0, numeric["predicted_average_return"])
                == math.copysign(1.0, numeric["actual_average_return"])
            )
            if direction != expected_direction:
                raise RuntimeError(f"foundation observation row {row_number} has inconsistent direction")
            if int(row["inference_batch"]) < 1:
                raise RuntimeError(f"foundation observation row {row_number} has an invalid batch")
    if count != int(payload.get("observation_count", -1)):
        raise RuntimeError("foundation observation CSV row count does not match the report")
    if symbols != set(FOUNDATION_SYMBOLS):
        raise RuntimeError(f"foundation observation CSV symbol contract failed: {sorted(symbols)}")


def _validate_chart(path: Path) -> None:
    payload = path.read_bytes()
    if not payload or len(payload) > 5 * 1024 * 1024:
        raise RuntimeError("foundation chart is empty or exceeds the repository size gate")
    text = payload.decode("utf-8")
    if "not P&amp;L" not in text or any(symbol not in text for symbol in FOUNDATION_SYMBOLS):
        raise RuntimeError("foundation chart omits its non-P&L or symbol disclosure")
    lowered = text.lower()
    if "<script" in lowered or "href=" in lowered or "url(" in lowered:
        raise RuntimeError("foundation chart contains active or externally referenced content")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeError("foundation chart is not well-formed SVG") from exc
    if not root.tag.endswith("svg"):
        raise RuntimeError("foundation chart root is not SVG")


def _validated_payload(report_path: Path, observations: Path, chart: Path) -> dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("foundation benchmark report is not an object")
    _assert_finite_numbers(payload)
    if payload.get("version") != FOUNDATION_BENCHMARK_VERSION:
        raise RuntimeError("foundation benchmark version is not promotable")
    status = payload.get("status")
    if status not in {"rejected", "predictive_candidate"}:
        raise RuntimeError("foundation benchmark status is invalid")
    if payload.get("predictive_candidate") is not (status == "predictive_candidate"):
        raise RuntimeError("foundation benchmark candidate status is inconsistent")
    if payload.get("trading_authority") is not False:
        raise RuntimeError("foundation benchmark must explicitly deny trading authority")
    if int(payload.get("observation_count", 0)) < 1:
        raise RuntimeError("foundation benchmark contains no observations")
    if _sha256(observations) != payload.get("observations_sha256"):
        raise RuntimeError("foundation observation CSV hash does not match the report")
    if _sha256(chart) != payload.get("chart_sha256"):
        raise RuntimeError("foundation chart hash does not match the report")
    source_evidence = payload.get("source_evidence")
    if not isinstance(source_evidence, list) or any(
        not isinstance(item, dict) for item in source_evidence
    ):
        raise RuntimeError("foundation source evidence is not a list of objects")
    symbols = tuple(item.get("symbol") for item in source_evidence)
    if symbols != FOUNDATION_SYMBOLS:
        raise RuntimeError(f"foundation evidence symbol contract failed: {symbols}")
    for item in source_evidence:
        if (
            int(item.get("row_count", 0)) != int(item.get("expected_rows", -1))
            or int(item.get("row_count", 0)) <= 0
            or tuple(item.get("sources", ())) != ("binance_public_archive",)
            or int(item.get("minimum_open_time", -1)) != int(item.get("start_ms", -2))
            or int(item.get("maximum_open_time", -1)) + 60_000
            != int(item.get("end_exclusive_ms", -2))
        ):
            raise RuntimeError(f"foundation source evidence failed for {item.get('symbol')}")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict) or "not accessed" not in str(
        evaluation.get("terminal_period", "")
    ):
        raise RuntimeError("foundation report does not preserve the sealed terminal period")
    if evaluation.get("orders_allowed") is not False or evaluation.get("after_cost_trading_evidence") is not False:
        raise RuntimeError("foundation report overstates forecast evidence as trading evidence")
    engine = payload.get("engine")
    source = engine.get("source") if isinstance(engine, dict) else None
    if not isinstance(source, dict) or source.get("verified") is not True:
        raise RuntimeError("foundation report lacks verified executable source evidence")
    inference = payload.get("inference")
    repeatability = inference.get("seeded_repeatability") if isinstance(inference, dict) else None
    if (
        not isinstance(repeatability, dict)
        or repeatability.get("checked") is not True
        or repeatability.get("exact") is not True
        or int(inference.get("in_process_retries", -1)) != 0
    ):
        raise RuntimeError("foundation report lacks exact process-isolated repeatability evidence")
    _validate_observations(observations, payload)
    _validate_chart(chart)
    return payload


def _sanitized_payload(
    payload: dict[str, object],
    *,
    source_report_sha256: str,
) -> dict[str, object]:
    promoted = copy.deepcopy(payload)
    config = promoted.get("config")
    if isinstance(config, dict):
        config["database_path"] = "data/market_data.sqlite"
        config["source_cache_root"] = "<verified-local-foundation-cache>"
    engine = promoted.get("engine")
    if isinstance(engine, dict):
        source = engine.get("source")
        if isinstance(source, dict):
            source["source_root"] = (
                "<verified-local-foundation-cache>/kronos/" + str(source.get("commit", ""))
            )
    promoted["observations_path"] = "observations.csv"
    promoted["chart_path"] = "benchmark.svg"
    promoted["promotion"] = {
        "source_report_sha256": source_report_sha256,
        "source_observations_sha256": payload.get("observations_sha256"),
        "path_sanitization": (
            "local database/cache/output paths replaced; UTF-8 text normalized to LF; "
            "numerical values unchanged"
        ),
        "latest_only_contract": "docs/ai/foundation/latest",
    }
    _assert_no_host_paths(promoted)
    return promoted


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _write_bytes_atomic(path, serialized)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=1_048_576)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_text_lf_atomic(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    _write_bytes_atomic(target, normalized.encode("utf-8"))


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected: list[str] = []
    for entry in output.iterdir():
        if entry.name in _PROMOTED_FILES:
            continue
        is_own_temporary = entry.is_file() and any(
            entry.name.startswith(f".{name}.") and entry.name.endswith(".tmp")
            for name in _PROMOTED_FILES
        )
        if is_own_temporary:
            entry.unlink()
        else:
            unexpected.append(entry.name)
    if unexpected:
        raise RuntimeError(
            "foundation latest directory contains unexpected stale artifacts: "
            + ", ".join(sorted(unexpected))
        )


def verify_promoted_bundle(output: Path) -> dict[str, object]:
    actual = {entry.name for entry in output.iterdir() if entry.is_file()}
    if actual != _PROMOTED_FILES:
        raise RuntimeError(f"foundation promoted file set is invalid: {sorted(actual)}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest.get("files", {})) != _PROMOTED_FILES - {"manifest.json"}:
        raise RuntimeError("foundation promotion manifest schema is invalid")
    for name, expected in manifest["files"].items():
        if _sha256(output / name) != expected:
            raise RuntimeError(f"foundation promoted hash mismatch: {name}")
    return manifest


def _readme(payload: dict[str, object]) -> str:
    overall = payload["metrics"]["overall"]
    calibrated = payload["calibration"]["calibrated_selection_metrics"]["overall"]
    bootstrap = payload["bootstrap"]
    inference = payload["inference"]
    reasons = payload.get("reasons") or []
    first_reason = str(reasons[0]) if reasons else "none"
    return f"""# Latest Financial Foundation Benchmark

This directory contains the latest committed Kronos candidate evidence. It is
real post-pretraining Binance USD-M archive data for BTCUSDT, ETHUSDT, and
SOLUSDT. It is **not** a profitability result and grants no trading authority.

![Kronos benchmark](benchmark.svg)

| Field | Result |
|---|---:|
| Status | `{payload['status']}` |
| Observations | {payload['observation_count']} |
| Raw model MAE | {float(overall['model_mae']):.10f} |
| Random-walk MAE | {float(overall['random_walk_mae']):.10f} |
| Raw MAE improvement | {float(overall['mae_improvement_pct']) * 100.0:.4f}% |
| Raw information coefficient | {float(overall['information_coefficient']):.6f} |
| Raw direction accuracy | {float(overall['direction_accuracy']) * 100.0:.3f}% |
| Calibrated selection MAE | {float(calibrated['model_mae']):.10f} |
| Calibrated random-walk MAE | {float(calibrated['random_walk_mae']):.10f} |
| Calibrated uplift probability | {float(bootstrap['positive_probability']) * 100.0:.3f}% |
| Calibrated 95% CI | [{float(bootstrap['ci_95_low']):.10f}, {float(bootstrap['ci_95_high']):.10f}] |
| Fault worker restarts | {int(inference['worker_restart_count'])} |
| Planned worker rotations | {int(inference['planned_worker_rotation_count'])} |
| First rejection reason | `{first_reason}` |

Files:

- `observations.csv` is the source table for replotting.
- `report.json` records data provenance, immutable model/source hashes, metrics,
  causal calibration, bootstrap evidence, seeded repeatability, and worker
  recovery evidence.
- `benchmark.svg` is generated from the CSV and is not the numerical authority.
- `manifest.json` binds the promoted files by SHA-256.

The benchmark intentionally leaves data from 2026 onward sealed as terminal
evidence. A rejected forecast model must remain advisory/research-only.
"""


def promote(report_path: Path, observations: Path, chart: Path, output: Path) -> None:
    payload = _validated_payload(report_path, observations, chart)
    _prepare_output(output)
    source_report_sha256 = _sha256(report_path)
    promoted = _sanitized_payload(
        payload,
        source_report_sha256=source_report_sha256,
    )
    observations_target = output / "observations.csv"
    chart_target = output / "benchmark.svg"
    report_target = output / "report.json"
    manifest_target = output / "manifest.json"
    manifest_target.unlink(missing_ok=True)
    _copy_text_lf_atomic(observations, observations_target)
    _copy_atomic(chart, chart_target)
    promoted["observations_sha256"] = _sha256(observations_target)
    _write_json_atomic(report_target, promoted)
    _write_bytes_atomic((output / "README.md"), _readme(promoted).encode("utf-8"))
    manifest = {
        "version": 1,
        "files": {
            "README.md": _sha256(output / "README.md"),
            "benchmark.svg": _sha256(chart_target),
            "observations.csv": _sha256(observations_target),
            "report.json": _sha256(report_target),
        },
        "source_report_sha256": source_report_sha256,
        "observation_count": promoted["observation_count"],
        "status": promoted["status"],
        "trading_authority": promoted["trading_authority"],
    }
    _write_json_atomic(manifest_target, manifest)
    verify_promoted_bundle(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--chart", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ai/foundation/latest"),
    )
    args = parser.parse_args(argv)
    try:
        promote(args.report, args.observations, args.chart, args.output)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"foundation benchmark promotion failed: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
