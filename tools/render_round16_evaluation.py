from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
)
from simple_ai_trading.polymarket_resolution import (  # noqa: E402
    validate_official_resolution,
)
from simple_ai_trading.polymarket_round16 import (  # noqa: E402
    Round16HistoricalContract,
    load_round16_historical_contract,
)
from simple_ai_trading.polymarket_round16_model import (  # noqa: E402
    load_round16_model_panel,
    predict_round16_candidate,
)
from simple_ai_trading.polymarket_round16_targets import (  # noqa: E402, PLC2701
    ROUND16_RESOLUTION_SCHEMA_VERSION,
    _official_market,
)


DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v2.json"
)
DEFAULT_DATABASE = ROOT / "data" / "polymarket-round16-btc-15m-screen-v1.duckdb"
DEFAULT_PRETEST = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "evidence"
    / "round-016-pretest-v2.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "evidence"
    / "round-016-evaluation-v2"
    / "round16-evaluation.json"
)
DEFAULT_PINS = DEFAULT_EVALUATION.parent / "round16-shadow-pins.json"
DEFAULT_OUTPUT = DEFAULT_EVALUATION.parent

_CONTROL = "#2C5AA0"
_RIDGE = "#6B7280"
_CHALLENGER = "#087E8B"
_PASS = "#067647"
_FAIL = "#B42318"
_GRID = "#D0D5DD"
_TEXT = "#17202A"


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


def _report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_artifact(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is unavailable or invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    body = dict(value)
    artifact_sha = str(body.pop("artifact_sha256", ""))
    if len(artifact_sha) != 64 or _canonical_sha256(body) != artifact_sha:
        raise ValueError(f"{name} self-hash differs")
    return value


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _verify_test_resolutions(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
) -> Mapping[str, object]:
    markets = {market.condition_id: market for market in store.markets(roles=("test",))}
    manifest_row = (
        store.connect()
        .execute(
            """
        SELECT manifest_json, manifest_sha256
        FROM target.round16_resolution_manifest
        WHERE singleton
        """
        )
        .fetchone()
    )
    if manifest_row is None:
        raise ValueError("Round 16 target implementation manifest is missing")
    manifest_json, implementation_sha = str(manifest_row[0]), str(manifest_row[1])
    manifest = json.loads(manifest_json)
    if not isinstance(manifest, Mapping):
        raise ValueError("Round 16 target implementation manifest is malformed")
    manifest_body = dict(manifest)
    claimed_manifest_sha = str(manifest_body.pop("manifest_sha256", ""))
    if (
        _canonical_json(manifest) != manifest_json
        or _canonical_sha256(manifest_body) != implementation_sha
        or claimed_manifest_sha != implementation_sha
    ):
        raise ValueError("Round 16 target implementation manifest differs")

    rows = (
        store.connect()
        .execute(
            """
        SELECT condition_id, role, winning_token_id, winning_outcome,
               gamma_payload_json, gamma_payload_sha256,
               clob_payload_json, clob_payload_sha256,
               evidence_sha256, observed_at_ms
        FROM target.official_resolution
        WHERE role = 'test'
        ORDER BY condition_id
        """
        )
        .fetchall()
    )
    if len(markets) != 1_440 or len(rows) != 1_440:
        raise ValueError("Round 16 test resolution coverage differs")

    outcomes = {"Down": 0, "Up": 0}
    identity_evidence: list[Mapping[str, object]] = []
    resolution_evidence: list[Mapping[str, object]] = []
    for row in rows:
        (
            condition_id,
            role,
            winning_token_id,
            winning_outcome,
            gamma_payload_json,
            gamma_payload_sha256,
            clob_payload_json,
            clob_payload_sha256,
            evidence_sha256,
            observed_at_ms,
        ) = row
        market = markets.get(str(condition_id))
        if market is None or str(role) != market.role or market.role != "test":
            raise ValueError("Round 16 test resolution role or identity differs")
        identity = json.loads(market.identity_payload_json)
        if (
            _canonical_json(identity) != market.identity_payload_json
            or hashlib.sha256(market.identity_payload_json.encode("ascii")).hexdigest()
            != market.identity_payload_sha256
        ):
            raise ValueError("Round 16 test identity payload differs")

        gamma_json = str(gamma_payload_json)
        clob_json = str(clob_payload_json)
        gamma = json.loads(gamma_json)
        clob = json.loads(clob_json)
        if (
            _canonical_json(gamma) != gamma_json
            or hashlib.sha256(gamma_json.encode("ascii")).hexdigest()
            != str(gamma_payload_sha256)
            or _canonical_json(clob) != clob_json
            or hashlib.sha256(clob_json.encode("ascii")).hexdigest()
            != str(clob_payload_sha256)
        ):
            raise ValueError("Round 16 terminal payload hash differs")
        winner = validate_official_resolution(
            _official_market(market, gamma),
            clob,
            gamma,
            observed_wall_ms=int(observed_at_ms),
        )
        if winner != (str(winning_token_id), str(winning_outcome)):
            raise ValueError("Round 16 stored terminal winner differs")
        evidence = {
            "schema_version": ROUND16_RESOLUTION_SCHEMA_VERSION,
            "contract_sha256": contract.contract_sha256,
            "condition_id": market.condition_id,
            "role": market.role,
            "winning_token_id": winner[0],
            "winning_outcome": winner[1],
            "target_implementation_sha256": implementation_sha,
            "gamma_payload_sha256": str(gamma_payload_sha256),
            "clob_payload_sha256": str(clob_payload_sha256),
        }
        if _canonical_sha256(evidence) != str(evidence_sha256):
            raise ValueError("Round 16 resolution evidence hash differs")
        if winner[1] not in outcomes:
            raise ValueError("Round 16 stored outcome differs")
        outcomes[winner[1]] += 1
        identity_evidence.append(
            {
                "condition_id": market.condition_id,
                "event_start_ms": market.event_start_ms,
                "identity_payload_sha256": market.identity_payload_sha256,
                "market_id": market.market_id,
            }
        )
        resolution_evidence.append(
            {
                "condition_id": market.condition_id,
                "evidence_sha256": str(evidence_sha256),
            }
        )
    if outcomes != {"Down": 720, "Up": 720}:
        raise ValueError("Round 16 test outcome balance differs")
    return {
        "schema_version": "polymarket-round16-test-resolution-audit-v1",
        "contract_sha256": contract.contract_sha256,
        "condition_count": len(rows),
        "role": "test",
        "outcomes": outcomes,
        "invalid_rows": 0,
        "identity_evidence_sha256": _canonical_sha256(identity_evidence),
        "resolution_evidence_sha256": _canonical_sha256(resolution_evidence),
        "target_implementation_sha256": implementation_sha,
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", color=_GRID, linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(_GRID)
    axis.spines["bottom"].set_color(_GRID)
    axis.tick_params(colors=_TEXT, labelsize=9)
    axis.xaxis.label.set_color(_TEXT)
    axis.yaxis.label.set_color(_TEXT)
    axis.title.set_color(_TEXT)


def _footer(figure: plt.Figure) -> None:
    figure.text(
        0.01,
        0.01,
        (
            "Held-out predictive screen, 2026-07-01 to 2026-07-15 UTC. "
            "No order-book execution, fills, fees, PnL, or trading authority."
        ),
        color="#475467",
        fontsize=8,
    )


def _candidate_metrics(
    evaluation: Mapping[str, object],
) -> tuple[list[dict[str, object]], Mapping[str, Mapping[str, float]]]:
    raw = evaluation.get("candidate_metrics")
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation candidate metrics are missing")
    output: list[dict[str, object]] = []
    parsed: dict[str, Mapping[str, float]] = {}
    for candidate_id, metrics_value in raw.items():
        if not isinstance(metrics_value, Mapping):
            raise ValueError("evaluation candidate metrics are malformed")
        metrics = {
            key: float(metrics_value[key])
            for key in (
                "log_loss",
                "brier_score",
                "accuracy",
                "balanced_accuracy",
                "calibration_intercept",
                "calibration_slope",
                "expected_calibration_error",
            )
        }
        parsed[str(candidate_id)] = metrics
        output.append({"candidate_id": str(candidate_id), **metrics})
    output.sort(key=lambda row: float(row["log_loss"]))
    return output, parsed


def _daily_rows(
    event_start_ms: np.ndarray,
    labels: np.ndarray,
    control_probability: np.ndarray,
    challenger_probability: np.ndarray,
) -> list[dict[str, object]]:
    day_values = np.asarray(
        [
            datetime.fromtimestamp(int(value) / 1_000, tz=UTC).date().isoformat()
            for value in event_start_ms
        ],
        dtype=object,
    )
    rows: list[dict[str, object]] = []
    epsilon = np.finfo(np.float64).eps
    for day in sorted(set(day_values.tolist())):
        selected = day_values == day
        y = labels[selected].astype(np.float64)
        control = np.clip(
            control_probability[selected].astype(np.float64),
            epsilon,
            1 - epsilon,
        )
        challenger = np.clip(
            challenger_probability[selected].astype(np.float64),
            epsilon,
            1 - epsilon,
        )
        control_loss = float(
            np.mean(-(y * np.log(control) + (1 - y) * np.log1p(-control)))
        )
        challenger_loss = float(
            np.mean(-(y * np.log(challenger) + (1 - y) * np.log1p(-challenger)))
        )
        rows.append(
            {
                "utc_day": day,
                "decision_rows": int(np.count_nonzero(selected)),
                "conditions": int(np.count_nonzero(selected) // 14),
                "control_log_loss": control_loss,
                "challenger_log_loss": challenger_loss,
                "challenger_improvement": control_loss - challenger_loss,
            }
        )
    return rows


def _calibration_rows(
    labels: np.ndarray,
    *,
    control_probability: np.ndarray,
    challenger_probability: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for model, probabilities in (
        ("digital_moneyness_control", control_probability),
        ("binance_lightgbm_challenger", challenger_probability),
    ):
        indexes = np.minimum(np.digitize(probabilities, edges[1:-1]), 9)
        for index in range(10):
            selected = indexes == index
            count = int(np.count_nonzero(selected))
            rows.append(
                {
                    "model": model,
                    "bin_lower": float(edges[index]),
                    "bin_upper": float(edges[index + 1]),
                    "decision_rows": count,
                    "mean_probability": (
                        float(np.mean(probabilities[selected])) if count else ""
                    ),
                    "observed_up_rate": (
                        float(np.mean(labels[selected])) if count else ""
                    ),
                }
            )
    return rows


def _render_metrics(
    output: Path,
    metrics: Mapping[str, Mapping[str, float]],
    *,
    control_id: str,
    challenger_id: str,
) -> None:
    ridge_id = "binance_ridge_logistic-l2-0.01"
    candidates = (control_id, ridge_id, challenger_id)
    labels = ("Digital moneyness", "Binance ridge", "Binance LightGBM")
    colors = (_CONTROL, _RIDGE, _CHALLENGER)
    measures = (
        ("log_loss", "Log loss (lower is better)", "{:.4f}"),
        ("brier_score", "Brier score (lower is better)", "{:.4f}"),
        ("balanced_accuracy", "Balanced accuracy (higher is better)", "{:.3%}"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    figure.patch.set_facecolor("white")
    for axis, (metric, title, formatter) in zip(axes, measures, strict=True):
        values = [metrics[candidate][metric] for candidate in candidates]
        bars = axis.bar(labels, values, color=colors, width=0.64)
        _style_axis(axis)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.tick_params(axis="x", rotation=22)
        axis.set_ylim(0, max(values) * 1.18)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                formatter.format(value),
                ha="center",
                va="bottom",
                color=_TEXT,
                fontsize=9,
                fontweight="bold",
            )
    figure.suptitle(
        "Round 16 held-out predictive metrics",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=_TEXT,
    )
    figure.text(
        0.99,
        0.98,
        "REJECTED: balanced-accuracy gate",
        ha="right",
        va="top",
        color=_FAIL,
        fontsize=10,
        fontweight="bold",
    )
    _footer(figure)
    figure.tight_layout(rect=(0, 0.07, 1, 0.9))
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_daily(output: Path, rows: Sequence[Mapping[str, object]]) -> None:
    labels = [str(row["utc_day"])[5:] for row in rows]
    values = [float(row["challenger_improvement"]) for row in rows]
    colors = [_PASS if value >= 0 else _FAIL for value in values]
    figure, axis = plt.subplots(figsize=(12.5, 5.4))
    figure.patch.set_facecolor("white")
    bars = axis.bar(labels, values, color=colors, width=0.72)
    _style_axis(axis)
    axis.axhline(0, color=_TEXT, linewidth=0.9)
    axis.set_ylabel("Control log loss - challenger log loss")
    axis.set_xlabel("UTC day (2026)")
    axis.set_title(
        "Probabilistic improvement was positive on most held-out days",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        1,
        1.01,
        "Positive favors challenger",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=_PASS,
        fontsize=9,
        fontweight="bold",
    )
    max_absolute = max(abs(value) for value in values)
    offset = max_absolute * 0.045
    axis.set_ylim(
        min(values) - (max_absolute * 0.32),
        max(values) + (max_absolute * 0.18),
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            rotation=90,
            color=_TEXT,
            fontsize=7,
        )
    _footer(figure)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_calibration(
    output: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 6.4))
    figure.patch.set_facecolor("white")
    _style_axis(axis)
    axis.plot([0, 1], [0, 1], color=_GRID, linewidth=1.3, label="Ideal")
    for model, label, color, marker in (
        ("digital_moneyness_control", "Digital moneyness", _CONTROL, "o"),
        ("binance_lightgbm_challenger", "Binance LightGBM", _CHALLENGER, "s"),
    ):
        selected = [
            row for row in rows if row["model"] == model and row["decision_rows"]
        ]
        axis.plot(
            [float(row["mean_probability"]) for row in selected],
            [float(row["observed_up_rate"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean predicted probability of Up")
    axis.set_ylabel("Observed Up rate")
    axis.set_title(
        "Held-out reliability by fixed probability bin",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(frameon=False, loc="lower right")
    _footer(figure)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render hash-bound Round 16 held-out evaluation reports."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--pretest", type=Path, default=DEFAULT_PRETEST)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pretest = _load_artifact(args.pretest, name="Round 16 pretest")
    evaluation = _load_artifact(args.evaluation, name="Round 16 evaluation")
    pins = _load_artifact(args.pins, name="Round 16 evaluation pins")
    if (
        evaluation.get("pretest_artifact_sha256") != pins.get("pretest_envelope_sha256")
        or evaluation.get("accepted_predictive_edge") is not False
        or pins.get("evaluation_envelope_sha256") in {None, ""}
        or pins.get("accepted_predictive_edge") is not False
        or pins.get("trading_authority") is not False
    ):
        raise ValueError("Round 16 evaluation linkage or decision differs")
    control_id = str(evaluation["best_control_id"])
    challenger_id = str(evaluation["best_challenger_id"])
    candidates_value = pretest.get("candidates")
    if not isinstance(candidates_value, list):
        raise ValueError("Round 16 pretest candidates are missing")
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in candidates_value
        if isinstance(candidate, Mapping)
    }
    if control_id not in candidates or challenger_id not in candidates:
        raise ValueError("Round 16 selected candidates are missing")

    contract = load_round16_historical_contract(args.contract)
    with HistoricalScreenStore(
        args.database,
        contract=contract.historical,
        read_only=True,
    ) as store:
        if store.state != "evaluated":
            raise ValueError("Round 16 database is not evaluated")
        stored_pretest, pretest_envelope_sha = store.pretest_artifact()
        evaluation_row = (
            store.connect()
            .execute(
                """
            SELECT artifact_json, artifact_sha256
            FROM target.evaluation_manifest
            WHERE singleton
            """
            )
            .fetchone()
        )
        stored_evaluation_json = (
            "" if evaluation_row is None else str(evaluation_row[0])
        )
        stored_evaluation_sha = "" if evaluation_row is None else str(evaluation_row[1])
        if (
            dict(stored_pretest) != dict(pretest)
            or pretest_envelope_sha != pins.get("pretest_envelope_sha256")
            or evaluation_row is None
            or _canonical_json(evaluation) != stored_evaluation_json
            or hashlib.sha256(stored_evaluation_json.encode("ascii")).hexdigest()
            != stored_evaluation_sha
            or json.loads(stored_evaluation_json) != dict(evaluation)
            or stored_evaluation_sha != pins.get("evaluation_envelope_sha256")
        ):
            raise ValueError("Round 16 exported and stored artifacts differ")
        resolution_audit_body = dict(_verify_test_resolutions(store, contract))
        test = load_round16_model_panel(store, contract, roles=("test",))
        if (
            evaluation.get("contract_sha256") != contract.contract_sha256
            or pins.get("contract_sha256") != contract.contract_sha256
            or evaluation.get("dataset_sha256") != test.dataset_sha256
            or pretest.get("dataset_sha256") != test.dataset_sha256
            or pins.get("dataset_sha256") != test.dataset_sha256
        ):
            raise ValueError("Round 16 contract or dataset linkage differs")

    control_probability = predict_round16_candidate(
        candidates[control_id],
        test.features,
    )
    challenger_probability = predict_round16_candidate(
        candidates[challenger_id],
        test.features,
    )
    candidate_rows, candidate_metrics = _candidate_metrics(evaluation)
    daily_rows = _daily_rows(
        test.event_start_ms,
        test.labels,
        control_probability,
        challenger_probability,
    )
    calibration_rows = _calibration_rows(
        test.labels,
        control_probability=control_probability,
        challenger_probability=challenger_probability,
    )
    if len(daily_rows) != 15 or sum(
        int(row["decision_rows"]) for row in daily_rows
    ) != len(test.labels):
        raise ValueError("Round 16 daily report coverage differs")

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    candidate_table = output / "round16-candidate-metrics.csv"
    daily_table = output / "round16-daily-logloss.csv"
    calibration_table = output / "round16-calibration-bins.csv"
    resolution_audit_path = output / "round16-resolution-audit.json"
    metrics_chart = output / "round16-heldout-metrics.png"
    daily_chart = output / "round16-daily-logloss-uplift.png"
    calibration_chart = output / "round16-calibration.png"
    _write_csv(candidate_table, tuple(candidate_rows[0]), candidate_rows)
    _write_csv(daily_table, tuple(daily_rows[0]), daily_rows)
    _write_csv(calibration_table, tuple(calibration_rows[0]), calibration_rows)
    resolution_audit = {
        **resolution_audit_body,
        "artifact_sha256": _canonical_sha256(resolution_audit_body),
    }
    _write_json(resolution_audit_path, resolution_audit)
    _render_metrics(
        metrics_chart,
        candidate_metrics,
        control_id=control_id,
        challenger_id=challenger_id,
    )
    _render_daily(daily_chart, daily_rows)
    _render_calibration(calibration_chart, calibration_rows)

    generated = (
        candidate_table,
        daily_table,
        calibration_table,
        resolution_audit_path,
        metrics_chart,
        daily_chart,
        calibration_chart,
    )
    manifest_body: dict[str, object] = {
        "schema_version": "polymarket-round16-evaluation-report-manifest-v1",
        "contract_sha256": contract.contract_sha256,
        "dataset_sha256": str(evaluation["dataset_sha256"]),
        "pretest_artifact_sha256": str(pretest["artifact_sha256"]),
        "evaluation_artifact_sha256": str(evaluation["artifact_sha256"]),
        "accepted_predictive_edge": False,
        "source_files": {
            _report_path(args.pretest): _file_sha256(args.pretest),
            _report_path(args.evaluation): _file_sha256(args.evaluation),
            _report_path(args.pins): _file_sha256(args.pins),
        },
        "outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in generated
        },
    }
    manifest = {
        **manifest_body,
        "artifact_sha256": _canonical_sha256(manifest_body),
    }
    manifest_path = output / "round16-report-manifest.json"
    _write_json(manifest_path, manifest)
    print(_canonical_json({"output_directory": str(output.resolve()), **manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
