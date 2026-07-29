from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from html import escape
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_model import (  # noqa: E402
    condition_balanced_binary_metrics,
    load_historical_model_panel,
    predict_historical_candidate,
)
from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
    load_historical_screen_contract,
)


DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-screen-v2.json"
)
DEFAULT_DATABASE = ROOT / "data" / "polymarket-round14-historical-screen-v2.duckdb"
DEFAULT_RESEARCH_ROOT = ROOT / "docs" / "model-research" / "polymarket"
PUBLICATION_SCHEMA = "polymarket-round14-historical-publication-v1"
_CONTROL = "#59636f"
_CHALLENGER = "#007f78"
_POSITIVE = "#0f766e"
_NEGATIVE = "#b44545"
_INK = "#17212b"
_GRID = "#d7dde2"
_PAPER = "#ffffff"


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(timestamp_ms: int, *, timespec: str = "seconds") -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC)
        .isoformat(timespec=timespec)
        .replace("+00:00", "Z")
    )


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return stream.getvalue()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_evaluation(
    store: HistoricalScreenStore,
) -> tuple[Mapping[str, object], str]:
    row = (
        store.connect()
        .execute(
            """
        SELECT artifact_json, artifact_sha256
        FROM target.evaluation_manifest WHERE singleton
        """
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 14 evaluation artifact is missing")
    value = json.loads(str(row[0]))
    if (
        not isinstance(value, dict)
        or _canonical_json(value) != str(row[0])
        or _canonical_sha256(value) != str(row[1])
    ):
        raise ValueError("Round 14 evaluation artifact integrity failed")
    return value, str(row[1])


def _candidate_map(
    pretest: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    values = pretest.get("candidates")
    if not isinstance(values, list):
        raise ValueError("Round 14 pretest candidates are missing")
    output: dict[str, Mapping[str, object]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Round 14 candidate payload is malformed")
        identifier = str(value.get("candidate_id") or "")
        if not identifier or identifier in output:
            raise ValueError("Round 14 candidate identity differs")
        body = dict(value)
        claimed = str(body.pop("artifact_sha256", ""))
        if _canonical_sha256(body) != claimed:
            raise ValueError("Round 14 candidate hash differs")
        output[identifier] = value
    if len(output) != 4:
        raise ValueError("Round 14 candidate count differs")
    return output


def _binary_loss(label: np.ndarray, probability: np.ndarray) -> np.ndarray:
    prediction = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    truth = np.asarray(label, dtype=np.float64)
    return -(truth * np.log(prediction) + (1.0 - truth) * np.log1p(-prediction))


def _verify_metrics(
    expected: Mapping[str, object],
    actual: Mapping[str, float],
) -> None:
    if set(expected) != set(actual):
        raise ValueError("Round 14 published metric names differ")
    for key, value in actual.items():
        if not math.isclose(
            float(expected[key]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Round 14 published {key} differs")


def _decision_and_condition_rows(
    store: HistoricalScreenStore,
    evaluation: Mapping[str, object],
    pretest: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    panel = load_historical_model_panel(store, roles=("test",))
    candidates = _candidate_map(pretest)
    control_id = str(evaluation["best_control_id"])
    challenger_id = str(evaluation["best_challenger_id"])
    control = predict_historical_candidate(candidates[control_id], panel.features)
    challenger = predict_historical_candidate(
        candidates[challenger_id],
        panel.features,
    )
    expected_metrics = evaluation.get("candidate_metrics")
    if not isinstance(expected_metrics, Mapping):
        raise ValueError("Round 14 evaluation metrics are missing")
    candidate_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for identifier, candidate in candidates.items():
        prediction = predict_historical_candidate(candidate, panel.features)
        predictions[identifier] = prediction
        metrics = condition_balanced_binary_metrics(panel, prediction)
        stored = expected_metrics.get(identifier)
        if not isinstance(stored, Mapping):
            raise ValueError("Round 14 stored candidate metric is missing")
        _verify_metrics(stored, metrics)
        tune = candidate.get("tune_metrics")
        if not isinstance(tune, Mapping):
            raise ValueError("Round 14 tune metrics are missing")
        model = candidate.get("model")
        calibration = candidate.get("calibration")
        if not isinstance(model, Mapping) or not isinstance(calibration, Mapping):
            raise ValueError("Round 14 candidate model metadata is missing")
        candidate_rows.append(
            {
                "candidate_id": identifier,
                "family": candidate["family"],
                "kind": candidate["kind"],
                "selected_best_control": identifier == control_id,
                "selected_best_challenger": identifier == challenger_id,
                "tune_log_loss": tune["log_loss"],
                "tune_brier_score": tune["brier_score"],
                "tune_balanced_accuracy": tune["balanced_accuracy"],
                "test_log_loss": metrics["log_loss"],
                "test_brier_score": metrics["brier_score"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_calibration_intercept": metrics["calibration_intercept"],
                "test_calibration_slope": metrics["calibration_slope"],
                "test_expected_calibration_error": metrics[
                    "expected_calibration_error"
                ],
                "calibration_retained": calibration["retained"],
                "model_backend": model.get("backend_kind") or "cpu",
                "model_device": model.get("backend_device") or "cpu",
                "best_iteration": model.get("best_iteration"),
            }
        )
    control_loss = _binary_loss(panel.labels, control)
    challenger_loss = _binary_loss(panel.labels, challenger)
    decision_rows: list[dict[str, object]] = []
    for index in range(len(panel.labels)):
        offset = int(
            (panel.decision_time_ms[index] - panel.event_start_ms[index]) // 1_000
        )
        decision_rows.append(
            {
                "condition_id": panel.condition_ids[index],
                "event_start_ms": int(panel.event_start_ms[index]),
                "event_start_utc": _utc(int(panel.event_start_ms[index])),
                "decision_time_ms": int(panel.decision_time_ms[index]),
                "decision_time_utc": _utc(int(panel.decision_time_ms[index])),
                "decision_offset_seconds": offset,
                "winning_outcome": "Up" if panel.labels[index] == 1.0 else "Down",
                "label_up": int(panel.labels[index]),
                "control_id": control_id,
                "control_probability_up": float(control[index]),
                "control_log_loss": float(control_loss[index]),
                "control_brier_score": float(
                    (control[index] - panel.labels[index]) ** 2
                ),
                "challenger_id": challenger_id,
                "challenger_probability_up": float(challenger[index]),
                "challenger_log_loss": float(challenger_loss[index]),
                "challenger_brier_score": float(
                    (challenger[index] - panel.labels[index]) ** 2
                ),
                "paired_log_loss_advantage": float(
                    control_loss[index] - challenger_loss[index]
                ),
            }
        )
    ordered_conditions = tuple(dict.fromkeys(panel.condition_ids.tolist()))
    cumulative = 0.0
    condition_rows: list[dict[str, object]] = []
    for index, condition in enumerate(ordered_conditions, start=1):
        selected = panel.condition_ids == condition
        advantage = float(np.mean(control_loss[selected] - challenger_loss[selected]))
        cumulative += advantage
        first = int(np.flatnonzero(selected)[0])
        condition_rows.append(
            {
                "condition_index": index,
                "condition_id": condition,
                "event_start_ms": int(panel.event_start_ms[first]),
                "event_start_utc": _utc(int(panel.event_start_ms[first])),
                "winning_outcome": ("Up" if panel.labels[first] == 1.0 else "Down"),
                "control_mean_probability_up": float(np.mean(control[selected])),
                "challenger_mean_probability_up": float(np.mean(challenger[selected])),
                "control_mean_log_loss": float(np.mean(control_loss[selected])),
                "challenger_mean_log_loss": float(np.mean(challenger_loss[selected])),
                "paired_mean_log_loss_advantage": advantage,
                "cumulative_log_loss_advantage": cumulative,
            }
        )
    offset_rows: list[dict[str, object]] = []
    for offset in range(30, 241, 30):
        selected = (panel.decision_time_ms - panel.event_start_ms) // 1_000 == offset
        truth = panel.labels[selected]
        for identifier in (control_id, challenger_id):
            probability = predictions[identifier][selected]
            predicted = probability >= 0.5
            positive = truth == 1.0
            negative = ~positive
            offset_rows.append(
                {
                    "decision_offset_seconds": offset,
                    "candidate_id": identifier,
                    "kind": ("control" if identifier == control_id else "challenger"),
                    "conditions": int(np.count_nonzero(selected)),
                    "log_loss": float(np.mean(_binary_loss(truth, probability))),
                    "brier_score": float(np.mean(np.square(probability - truth))),
                    "accuracy": float(np.mean(predicted == positive)),
                    "balanced_accuracy": 0.5
                    * (
                        float(np.mean(predicted[positive]))
                        + float(np.mean(~predicted[negative]))
                    ),
                }
            )
    return decision_rows, condition_rows, candidate_rows, offset_rows


def _svg_open(title: str, description: str, *, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title desc"><title id="title">{escape(title)}</title>'
        f'<desc id="desc">{escape(description)}</desc>'
        f'<rect width="{width}" height="{height}" fill="{_PAPER}"/>'
    )


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: int = 400,
    fill: str = _INK,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>'
    )


def _metric_chart(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    control_id: str,
    challenger_id: str,
) -> str:
    values = {str(row["candidate_id"]): row for row in candidate_rows}
    control = values[control_id]
    challenger = values[challenger_id]
    metrics = (
        ("test_log_loss", "Log loss", 0.75, "lower is better"),
        ("test_brier_score", "Brier score", 0.30, "lower is better"),
        ("test_balanced_accuracy", "Balanced accuracy", 1.0, "higher is better"),
    )
    parts = [
        _svg_open(
            "Round 14 held-out predictive metrics",
            "Best frozen control and challenger on 287 BTC five-minute conditions.",
            width=1200,
            height=560,
        ),
        _text(50, 48, "Round 14 held-out predictive metrics", size=26, weight=700),
        _text(
            50,
            76,
            "2026-06-22 UTC | 287 conditions | 2,296 causal decisions",
            size=15,
            fill="#46515c",
        ),
    ]
    for panel_index, (key, label, maximum, direction) in enumerate(metrics):
        left = 50 + panel_index * 380
        top = 115
        parts.append(
            f'<rect x="{left}" y="{top}" width="340" height="380" '
            f'fill="#f7f9fa" stroke="{_GRID}"/>'
        )
        parts.append(_text(left + 20, top + 36, label, size=18, weight=650))
        parts.append(_text(left + 20, top + 60, direction, size=13, fill="#59636f"))
        for row_index, (name, row, color) in enumerate(
            (
                ("Control", control, _CONTROL),
                ("Challenger", challenger, _CHALLENGER),
            )
        ):
            value = float(row[key])
            y = top + 120 + row_index * 120
            width = 270 * min(1.0, max(0.0, value / maximum))
            parts.append(_text(left + 20, y - 12, name, size=14, weight=600))
            parts.append(
                f'<rect x="{left + 20}" y="{y}" width="270" height="28" '
                f'fill="#e5e9ec"/>'
            )
            parts.append(
                f'<rect x="{left + 20}" y="{y}" width="{width:.2f}" '
                f'height="28" fill="{color}"/>'
            )
            parts.append(
                _text(
                    left + 310,
                    y + 21,
                    f"{value:.6f}",
                    size=14,
                    anchor="end",
                    weight=600,
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _line_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    subtitle: str,
    x_labels: Sequence[tuple[int, str]],
    y_label: str,
) -> str:
    width, height = 1200, 600
    left, right, top, bottom = 95, 45, 105, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y_values = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    y_min = min(0.0, float(np.min(y_values)))
    y_max = max(0.0, float(np.max(y_values)))
    margin = max(1e-6, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin

    def x(value: float) -> float:
        return left + (value - x_values[0]) / (x_values[-1] - x_values[0]) * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        _svg_open(title, subtitle, width=width, height=height),
        _text(50, 46, title, size=26, weight=700),
        _text(50, 74, subtitle, size=15, fill="#46515c"),
    ]
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        coordinate = y(value)
        parts.append(
            f'<line x1="{left}" y1="{coordinate:.2f}" x2="{width - right}" '
            f'y2="{coordinate:.2f}" stroke="{_GRID}"/>'
        )
        parts.append(_text(left - 12, coordinate + 5, f"{value:.3f}", anchor="end"))
    zero = y(0.0)
    parts.append(
        f'<line x1="{left}" y1="{zero:.2f}" x2="{width - right}" '
        f'y2="{zero:.2f}" stroke="{_INK}" stroke-width="1.5"/>'
    )
    points = " ".join(
        f"{x(float(row[x_key])):.2f},{y(float(row[y_key])):.2f}" for row in rows
    )
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{_CHALLENGER}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for value, label in x_labels:
        coordinate = x(float(value))
        parts.append(
            f'<line x1="{coordinate:.2f}" y1="{height - bottom}" '
            f'x2="{coordinate:.2f}" y2="{height - bottom + 8}" stroke="{_INK}"/>'
        )
        parts.append(
            _text(coordinate, height - bottom + 30, label, anchor="middle", size=13)
        )
    parts.append(
        _text(
            28,
            top + plot_height / 2,
            y_label,
            anchor="middle",
            size=14,
            weight=600,
        ).replace(
            "<text ",
            f'<text transform="rotate(-90 28 {top + plot_height / 2:.2f})" ',
            1,
        )
    )
    parts.append("</svg>")
    return "".join(parts)


def _offset_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1200, 570
    left, right, top, bottom = 90, 40, 110, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    by_kind = {
        kind: sorted(
            (row for row in rows if row["kind"] == kind),
            key=lambda row: int(row["decision_offset_seconds"]),
        )
        for kind in ("control", "challenger")
    }
    all_values = [float(row["log_loss"]) for row in rows]
    y_min = min(all_values) - 0.01
    y_max = max(all_values) + 0.01

    def x(offset: int) -> float:
        return left + (offset - 30) / 210 * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        _svg_open(
            "Predictive log loss by decision time",
            "Held-out log loss from 30 through 240 seconds into each market.",
            width=width,
            height=height,
        ),
        _text(50, 46, "Predictive log loss by decision time", size=26, weight=700),
        _text(
            50,
            74,
            "2026-06-22 UTC | each point contains 287 independent conditions",
            size=15,
            fill="#46515c",
        ),
    ]
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        coordinate = y(value)
        parts.append(
            f'<line x1="{left}" y1="{coordinate:.2f}" x2="{width - right}" '
            f'y2="{coordinate:.2f}" stroke="{_GRID}"/>'
        )
        parts.append(_text(left - 10, coordinate + 5, f"{value:.3f}", anchor="end"))
    for kind, color, label in (
        ("control", _CONTROL, "Control"),
        ("challenger", _CHALLENGER, "Challenger"),
    ):
        points = " ".join(
            f"{x(int(row['decision_offset_seconds'])):.2f},"
            f"{y(float(row['log_loss'])):.2f}"
            for row in by_kind[kind]
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round"/>'
        )
        for row in by_kind[kind]:
            parts.append(
                f'<circle cx="{x(int(row["decision_offset_seconds"])):.2f}" '
                f'cy="{y(float(row["log_loss"])):.2f}" r="4" fill="{color}"/>'
            )
        parts.append(
            f'<line x1="{930}" y1="{45 + 24 * (kind == "challenger")}" '
            f'x2="965" y2="{45 + 24 * (kind == "challenger")}" '
            f'stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            _text(975, 50 + 24 * (kind == "challenger"), label, size=14, weight=600)
        )
    for offset in range(30, 241, 30):
        parts.append(
            _text(
                x(offset),
                height - bottom + 30,
                f"{offset}s",
                anchor="middle",
                size=13,
            )
        )
    parts.append(
        _text(
            left + plot_width / 2,
            height - 25,
            "Seconds after five-minute market start",
            anchor="middle",
            size=14,
            weight=600,
        )
    )
    parts.append("</svg>")
    return "".join(parts)


def _progress_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1200, 420
    positions = {
        "failed before fit": 1,
        "no positive score": 1,
        "failed uncertainty gate": 1,
        "invalidated before outcome access": 0,
        "failed prospective capture": 0,
        "passed predictive gates; execution untested": 2,
    }
    labels = ("Invalid / unavailable", "Evaluated and failed", "Predictive pass")
    parts = [
        _svg_open(
            "Polymarket evidence progression",
            "Research status by round. Predictive pass is not profitability.",
            width=width,
            height=height,
        ),
        _text(50, 46, "Polymarket evidence progression", size=26, weight=700),
        _text(
            50,
            74,
            "Status is categorical; Round 14 has no execution or PnL claim",
            size=15,
            fill="#46515c",
        ),
    ]
    left, right, top, bottom = 210, 45, 115, 70
    plot_width = width - left - right
    for index, label in enumerate(labels):
        y = top + (2 - index) * 100
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" '
            f'stroke="{_GRID}"/>'
        )
        parts.append(_text(left - 15, y + 5, label, anchor="end", size=13))
    for index, row in enumerate(rows):
        x = left + index / max(1, len(rows) - 1) * plot_width
        level = positions[str(row["status"])]
        y = top + (2 - level) * 100
        color = _POSITIVE if level == 2 else _NEGATIVE if level == 1 else _CONTROL
        parts.append(f'<circle cx="{x:.2f}" cy="{y}" r="9" fill="{color}"/>')
        parts.append(
            _text(
                x,
                height - bottom + 30,
                f"R{row['round']}",
                anchor="middle",
                size=13,
                weight=600,
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _progress_rows(evaluation: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "round": 9,
            "evidence_type": "historical execution screen",
            "status": "failed before fit",
            "conditions": 141,
            "held_out_log_loss_skill": None,
            "held_out_brier_skill": None,
            "held_out_balanced_accuracy": None,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
        {
            "round": 10,
            "evidence_type": "historical execution screen",
            "status": "no positive score",
            "conditions": 141,
            "held_out_log_loss_skill": None,
            "held_out_brier_skill": None,
            "held_out_balanced_accuracy": None,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
        {
            "round": 11,
            "evidence_type": "historical execution screen",
            "status": "failed uncertainty gate",
            "conditions": 42,
            "held_out_log_loss_skill": None,
            "held_out_brier_skill": None,
            "held_out_balanced_accuracy": None,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
        {
            "round": 12,
            "evidence_type": "retrospective capture",
            "status": "invalidated before outcome access",
            "conditions": None,
            "held_out_log_loss_skill": None,
            "held_out_brier_skill": None,
            "held_out_balanced_accuracy": None,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
        {
            "round": 13,
            "evidence_type": "prospective capture",
            "status": "failed prospective capture",
            "conditions": None,
            "held_out_log_loss_skill": None,
            "held_out_brier_skill": None,
            "held_out_balanced_accuracy": None,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
        {
            "round": 14,
            "evidence_type": "held-out BTC direction screen",
            "status": "passed predictive gates; execution untested",
            "conditions": evaluation["test"]["conditions"],  # type: ignore[index]
            "held_out_log_loss_skill": evaluation["challenger_skill"][  # type: ignore[index]
                "log_loss"
            ],
            "held_out_brier_skill": evaluation["challenger_skill"]["brier"],  # type: ignore[index]
            "held_out_balanced_accuracy": evaluation["candidate_metrics"][  # type: ignore[index]
                evaluation["best_challenger_id"]
            ]["balanced_accuracy"],
            "profitability_claim": False,
            "live_trading_authority": False,
        },
    ]


def _safe_remove(path: Path, root: Path) -> None:
    selected = path.resolve()
    if selected.parent != root.resolve() or not selected.name.startswith(".round14-"):
        raise ValueError("unsafe Round 14 publication cleanup path")
    if selected.exists():
        shutil.rmtree(selected)


def publish(
    *,
    database: Path,
    contract_path: Path,
    research_root: Path,
) -> str:
    contract = load_historical_screen_contract(contract_path)
    with HistoricalScreenStore(database, contract=contract, read_only=True) as store:
        if store.state != "evaluated":
            raise ValueError("Round 14 historical screen is not evaluated")
        evaluation, evaluation_sha = _load_evaluation(store)
        pretest, pretest_sha = store.pretest_artifact()
        if (
            evaluation.get("pretest_artifact_sha256") != pretest_sha
            or evaluation.get("accepted_predictive_edge") is not True
        ):
            raise ValueError("Round 14 publication evidence boundary differs")
        decisions, conditions, candidates, offsets = _decision_and_condition_rows(
            store,
            evaluation,
            pretest,
        )
    root = research_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    legacy_ai_risk = latest / "ai-risk-models-rejected.json"
    legacy_ai_risk_content = (
        legacy_ai_risk.read_text(encoding="utf-8") if legacy_ai_risk.is_file() else None
    )
    staging = root / f".round14-latest-{evaluation_sha[:12]}"
    backup = root / f".round14-backup-{evaluation_sha[:12]}"
    _safe_remove(staging, root)
    _safe_remove(backup, root)
    (staging / "tables").mkdir(parents=True)
    (staging / "charts").mkdir(parents=True)
    progress = _progress_rows(evaluation)
    tables = {
        "round14-candidates.csv": candidates,
        "round14-decisions.csv": decisions,
        "round14-conditions.csv": conditions,
        "round14-decision-offsets.csv": offsets,
        "optimization-progress.csv": progress,
    }
    table_columns = {name: tuple(rows[0]) for name, rows in tables.items() if rows}
    for name, rows in tables.items():
        _write_text(
            staging / "tables" / name,
            _csv_text(table_columns[name], rows),
        )
    control_id = str(evaluation["best_control_id"])
    challenger_id = str(evaluation["best_challenger_id"])
    charts = {
        "round14-held-out-metrics.svg": _metric_chart(
            candidates,
            control_id=control_id,
            challenger_id=challenger_id,
        ),
        "round14-cumulative-log-loss-advantage.svg": _line_chart(
            conditions,
            x_key="event_start_ms",
            y_key="cumulative_log_loss_advantage",
            title="Cumulative held-out log-loss advantage",
            subtitle=(
                "Best-control loss minus frozen-challenger loss | 2026-06-22 UTC | "
                "positive favors challenger"
            ),
            x_labels=[
                (
                    int(conditions[index]["event_start_ms"]),
                    _utc(
                        int(conditions[index]["event_start_ms"]),
                        timespec="minutes",
                    )[11:16],
                )
                for index in (0, 72, 144, 216, 286)
            ],
            y_label="Cumulative paired log-loss advantage",
        ),
        "round14-decision-offset-log-loss.svg": _offset_chart(offsets),
        "optimization-progress.svg": _progress_chart(progress),
    }
    for name, content in charts.items():
        _write_text(staging / "charts" / name, content)
    evaluation_document = {**evaluation, "artifact_sha256": evaluation_sha}
    pretest_document = {**pretest, "artifact_sha256": pretest_sha}
    _write_text(
        staging / "round-014-evaluation.json",
        json.dumps(evaluation_document, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        staging / "round-014-pretest.json",
        json.dumps(pretest_document, indent=2, sort_keys=True) + "\n",
    )
    if legacy_ai_risk_content is not None:
        _write_text(
            staging / "ai-risk-models-rejected.json",
            legacy_ai_risk_content,
        )
    challenger = evaluation["candidate_metrics"][challenger_id]  # type: ignore[index]
    control = evaluation["candidate_metrics"][control_id]  # type: ignore[index]
    readme = f"""# Polymarket model status

> **Beta research software. No paper or live trading authority exists.**

![Held-out predictive metrics](charts/round14-held-out-metrics.svg)

Round 14 tested a frozen BTC five-minute direction model on all 287 eligible
conditions from 2026-06-22 UTC. The shallow Binance-flow LightGBM challenger
recorded log loss `{float(challenger["log_loss"]):.6f}` versus
`{float(control["log_loss"]):.6f}` for the best control, a
`{100 * float(evaluation["challenger_skill"]["log_loss"]):.2f}%` relative skill.
Balanced accuracy was `{float(challenger["balanced_accuracy"]):.4f}` and the
paired 95% block-bootstrap improvement interval was
`[{float(evaluation["paired_condition_block_bootstrap"]["lower_95"]):.5f},
{float(evaluation["paired_condition_block_bootstrap"]["upper_95"]):.5f}]`.

This is **predictive evidence only**. Polymarket spread, queue position, fills,
latency, fees, settlement, redemption, inventory risk, and PnL were not tested,
so it is not a profitability or execution claim.

## Audit

- [Evaluation artifact](round-014-evaluation.json)
- [Sealed pretest artifact](round-014-pretest.json)
- [Candidate metrics](tables/round14-candidates.csv)
- [Every held-out decision](tables/round14-decisions.csv)
- [UTC condition series](tables/round14-conditions.csv)
- [Decision-offset metrics](tables/round14-decision-offsets.csv)
- [Cross-round progression](tables/optimization-progress.csv)
- [AI risk-model rejection record](ai-risk-models-rejected.json)
- [Publication integrity](publication-integrity.json)

Regenerate from the closed local evidence database with
`python tools/publish_polymarket_round14_historical.py`.
"""
    _write_text(staging / "README.md", readme)
    chart_sources = {
        "round14-held-out-metrics.svg": ["round14-candidates.csv"],
        "round14-cumulative-log-loss-advantage.svg": ["round14-conditions.csv"],
        "round14-decision-offset-log-loss.svg": ["round14-decision-offsets.csv"],
        "optimization-progress.svg": ["optimization-progress.csv"],
    }
    artifacts: list[dict[str, object]] = []
    for path in sorted(
        (item for item in staging.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(staging).as_posix(),
    ):
        relative = path.relative_to(staging).as_posix()
        entry: dict[str, object] = {
            "path": f"latest/{relative}",
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if path.suffix == ".csv":
            entry["row_count"] = len(tables[path.name])
        if path.suffix == ".svg":
            entry["source_tables"] = chart_sources[path.name]
        artifacts.append(entry)
    manifest_body = {
        "schema_version": PUBLICATION_SCHEMA,
        "latest_round": 14,
        "source_contract_sha256": contract.contract_sha256,
        "source_dataset_sha256": evaluation["dataset_sha256"],
        "source_pretest_artifact_sha256": pretest_sha,
        "source_evaluation_artifact_sha256": evaluation_sha,
        "test_start_utc": _utc(int(evaluation["test"]["first_event_start_ms"])),  # type: ignore[index]
        "test_end_utc": _utc(int(evaluation["test"]["last_event_start_ms"])),  # type: ignore[index]
        "asset": "BTC",
        "market_variant": "fiveminute",
        "artifacts": artifacts,
        "accepted_predictive_edge": True,
        "profitability_claim": False,
        "paper_authority": False,
        "live_trading_authority": False,
        "manual_chart_edits_permitted": False,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _canonical_sha256(manifest_body),
    }
    _write_text(
        staging / "publication-integrity.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    if latest.exists():
        os.replace(latest, backup)
    try:
        os.replace(staging, latest)
        for entry in artifacts:
            path = root / str(entry["path"])
            if _file_sha256(path) != entry["sha256"]:
                raise ValueError("Round 14 published artifact verification failed")
        _safe_remove(backup, root)
    except BaseException:
        if latest.exists():
            invalid = root / f".round14-invalid-{evaluation_sha[:12]}"
            _safe_remove(invalid, root)
            os.replace(latest, invalid)
        if backup.exists():
            os.replace(backup, latest)
        raise
    _write_text(
        root / "round-014-btc-5m-historical-evaluation-v1.json",
        json.dumps(evaluation_document, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        root / "round-014-btc-5m-historical-pretest-v1.json",
        json.dumps(pretest_document, indent=2, sort_keys=True) + "\n",
    )
    return str(manifest["manifest_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish verified Polymarket Round 14 historical evidence."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    args = parser.parse_args()
    manifest_sha = publish(
        database=args.database,
        contract_path=args.contract,
        research_root=args.research_root,
    )
    print(
        json.dumps(
            {
                "event": "polymarket_round14_historical_published",
                "manifest_sha256": manifest_sha,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
