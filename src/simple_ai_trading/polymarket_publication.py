"""Deterministic publication of prospective Polymarket model evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import shutil
from typing import Any


_ARTIFACT_SCHEMA = "polymarket-prospective-model-experiment-v1"
_PREDICTION_SCHEMA = "polymarket-held-out-predictions-v1"
_PUBLICATION_SCHEMA = "polymarket-model-publication-v1"
_ASSETS = ("BTC", "ETH", "SOL")
_POLICIES = ("baseline", "model", "ai")
_COLORS = {
    "background": "#0b1220",
    "panel": "#111c2e",
    "grid": "#26364d",
    "text": "#f8fafc",
    "muted": "#a9b8cc",
    "baseline": "#94a3b8",
    "model": "#22c55e",
    "ai": "#38bdf8",
    "BTC": "#f59e0b",
    "ETH": "#8b5cf6",
    "SOL": "#14b8a6",
    "negative": "#fb7185",
}


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.replace("\r\n", "\n").encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"publication table has no rows: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"publication table columns drifted: {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, stream.getvalue())


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _as_rows(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return [_as_mapping(item, f"{name} row") for item in value]


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _utc(timestamp_ms: object) -> str:
    timestamp = int(timestamp_ms)
    return (
        datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _verify_claims(value: Mapping[str, Any], *, name: str) -> None:
    for key in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if key in value and value[key] is not False:
            raise ValueError(f"{name} makes a forbidden {key} claim")


def _verify_embedded_digest(
    value: Mapping[str, Any],
    digest_key: str,
    *,
    name: str,
) -> None:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    if len(claimed) != 64 or claimed != _canonical_sha256(canonical):
        raise ValueError(f"{name} {digest_key} is invalid")


@dataclass(frozen=True)
class ValidatedPolymarketArtifact:
    payload: Mapping[str, Any]
    artifact_sha256: str
    predictions: tuple[Mapping[str, Any], ...]
    executions: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PolymarketPublicationResult:
    round_number: int
    research_root: str
    source_artifact: str
    artifact_sha256: str
    generated_files: tuple[str, ...]
    manifest_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": _PUBLICATION_SCHEMA,
            "round": self.round_number,
            "research_root": self.research_root,
            "source_artifact": self.source_artifact,
            "artifact_sha256": self.artifact_sha256,
            "generated_files": list(self.generated_files),
            "manifest_sha256": self.manifest_sha256,
        }


def _validate_execution_report(
    report: Mapping[str, Any],
    *,
    conditions: set[str],
    expected_time_group_count: int,
    name: str,
) -> None:
    if report.get("schema_version") != "polymarket-execution-report-v2":
        raise ValueError(f"{name} uses an unsupported execution schema")
    _verify_claims(report, name=name)
    _verify_embedded_digest(report, "report_sha256", name=name)
    trades_value = report.get("trades")
    equity_value = report.get("equity_curve")
    if not isinstance(trades_value, list) or not isinstance(equity_value, list):
        raise ValueError(f"{name} execution ledger is malformed")
    trades = [_as_mapping(row, f"{name} trade") for row in trades_value]
    equity = [_as_mapping(row, f"{name} equity point") for row in equity_value]
    permissions = _as_mapping(
        report.get("market_permissions"),
        f"{name} market permissions",
    )
    delays = _as_mapping(
        report.get("decision_delay_ms_by_condition"),
        f"{name} decision delays",
    )
    if (
        set(permissions) != conditions
        or any(not isinstance(value, bool) for value in permissions.values())
        or set(delays) != conditions
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 300_000
            for value in delays.values()
        )
        or report.get("market_permission_sha256")
        != _canonical_sha256(
            {
                "schema_version": "polymarket-market-permission-v1",
                "permissions": dict(sorted(permissions.items())),
            }
        )
        or report.get("decision_delay_input_sha256")
        != _canonical_sha256(
            {
                "schema_version": "polymarket-decision-delay-input-v1",
                "decision_delay_ms_by_condition": dict(sorted(delays.items())),
            }
        )
    ):
        raise ValueError(f"{name} permission or latency map is invalid")
    config = _as_mapping(report.get("config"), f"{name} execution config")
    submission_latency = int(config["submission_latency_ms"])
    if not 1 <= submission_latency <= 60_000:
        raise ValueError(f"{name} network latency is invalid")
    if (
        int(report.get("evaluated_market_count", -1)) != len(conditions)
        or int(report.get("attempted_order_count", -1)) != len(trades)
        or len(equity) != expected_time_group_count
    ):
        raise ValueError(f"{name} execution coverage is inconsistent")

    filled = [row for row in trades if row.get("execution_state") == "FILLED"]
    settled = [
        row
        for row in filled
        if str(row.get("official_resolution_event_id", ""))
    ]
    wins = [row for row in filled if _decimal(row["realized_pnl_quote"], "PnL") > 0]
    losses = [row for row in filled if _decimal(row["realized_pnl_quote"], "PnL") <= 0]
    if (
        int(report.get("filled_order_count", -1)) != len(filled)
        or len(filled) != len(settled)
        or int(report.get("winning_order_count", -1)) != len(wins)
        or int(report.get("losing_order_count", -1)) != len(losses)
    ):
        raise ValueError(f"{name} contains unsettled or miscounted filled trades")

    deployed = Decimal("0")
    payouts = Decimal("0")
    fees = Decimal("0")
    realized = Decimal("0")
    trade_ids: set[str] = set()
    for trade in trades:
        condition_id = str(trade.get("condition_id", ""))
        trade_id = str(trade.get("trade_id", ""))
        trade_sha256 = str(trade.get("trade_sha256", ""))
        decision_delay = int(trade.get("decision_delay_ms", -1))
        trade_identity = dict(trade)
        trade_identity.pop("trade_sha256", None)
        if (
            str(trade.get("asset", "")) not in _ASSETS
            or condition_id not in conditions
            or len(trade_id) != 64
            or trade_id in trade_ids
            or len(trade_sha256) != 64
            or trade_sha256 != _canonical_sha256(trade_identity)
            or decision_delay != delays[condition_id]
            or int(trade.get("submission_latency_ms", -1)) != submission_latency
            or int(trade.get("effective_latency_ms", -1))
            != decision_delay + submission_latency
        ):
            raise ValueError(f"{name} trade identity or latency binding is invalid")
        trade_ids.add(trade_id)
        quantity = _decimal(trade.get("quantity"), f"{name} trade quantity")
        filled_quantity = _decimal(
            trade.get("filled_quantity"),
            f"{name} filled quantity",
        )
        average = _decimal(
            trade.get("average_fill_price"),
            f"{name} fill price",
        )
        fee = _decimal(trade.get("fee_quote"), f"{name} trade fee")
        payout = _decimal(
            trade.get("gross_payout_quote"),
            f"{name} trade payout",
        )
        pnl = _decimal(
            trade.get("realized_pnl_quote"),
            f"{name} trade PnL",
        )
        if min(quantity, filled_quantity, average, fee, payout) < 0:
            raise ValueError(f"{name} trade contains a negative accounting value")
        if trade.get("execution_state") == "FILLED":
            if filled_quantity != quantity or pnl != payout - average * quantity - fee:
                raise ValueError(f"{name} filled-trade accounting is inconsistent")
            deployed += average * quantity + fee
            payouts += payout
            fees += fee
            realized += pnl
        elif any(value != 0 for value in (filled_quantity, average, fee, payout, pnl)):
            raise ValueError(f"{name} non-fill carries economic credit")

    initial = _decimal(report.get("initial_capital_quote"), f"{name} initial equity")
    final = _decimal(report.get("final_equity_quote"), f"{name} final equity")
    reported_net = _decimal(report.get("net_realized_pnl_quote"), f"{name} net PnL")
    if (
        deployed != _decimal(report.get("gross_deployed_capital_quote"), f"{name} deployed")
        or payouts != _decimal(report.get("gross_payout_quote"), f"{name} payouts")
        or fees != _decimal(report.get("total_fees_quote"), f"{name} fees")
        or realized != reported_net
        or final != initial + reported_net
    ):
        raise ValueError(f"{name} aggregate accounting does not reconcile")

    running = initial
    peak = initial
    maximum_drawdown = Decimal("0")
    maximum_drawdown_fraction = Decimal("0")
    settled_times: list[int] = []
    for point in equity:
        settled_at = int(point.get("settled_at_ms", -1))
        group_pnl = _decimal(point.get("group_realized_pnl_quote"), "group PnL")
        running += group_pnl
        peak = max(peak, running)
        drawdown = peak - running
        drawdown_fraction = drawdown / peak if peak > 0 else Decimal("0")
        if (
            settled_at < 0
            or running != _decimal(point.get("equity_quote"), "equity")
            or peak != _decimal(point.get("peak_equity_quote"), "peak equity")
            or drawdown != _decimal(point.get("drawdown_quote"), "drawdown")
            or drawdown_fraction
            != _decimal(point.get("drawdown_fraction"), "drawdown fraction")
        ):
            raise ValueError(f"{name} equity curve does not reconcile")
        maximum_drawdown = max(maximum_drawdown, drawdown)
        maximum_drawdown_fraction = max(
            maximum_drawdown_fraction,
            drawdown_fraction,
        )
        settled_times.append(settled_at)
    if (
        settled_times != sorted(set(settled_times))
        or running != final
        or maximum_drawdown
        != _decimal(report.get("maximum_drawdown_quote"), "maximum drawdown")
        or maximum_drawdown_fraction
        != _decimal(
            report.get("maximum_drawdown_fraction"),
            "maximum drawdown fraction",
        )
    ):
        raise ValueError(f"{name} equity path summary does not reconcile")


def validate_polymarket_model_artifact(path: str | Path) -> ValidatedPolymarketArtifact:
    """Fail closed unless the experiment and every publication input are coherent."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = _as_mapping(json.loads(raw.decode("utf-8")), "model artifact")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket model artifact is not valid UTF-8 JSON") from exc
    if payload.get("schema_version") != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported Polymarket model artifact schema")
    canonical = dict(payload)
    claimed_artifact_sha256 = str(canonical.pop("artifact_sha256", ""))
    if (
        len(claimed_artifact_sha256) != 64
        or claimed_artifact_sha256 != _canonical_sha256(canonical)
    ):
        raise ValueError("Polymarket model artifact identity is invalid")
    _verify_claims(payload, name="model artifact")

    model = _as_mapping(payload.get("model"), "model")
    probability = _as_mapping(payload.get("probability_report"), "probability report")
    split = _as_mapping(payload.get("split"), "split")
    model_dataset = _as_mapping(payload.get("model_dataset"), "model dataset")
    feature_dataset = _as_mapping(payload.get("feature_dataset"), "feature dataset")
    for name, value, digest_key in (
        ("model", model, "model_sha256"),
        ("probability report", probability, "report_sha256"),
    ):
        _verify_claims(value, name=name)
        _verify_embedded_digest(value, digest_key, name=name)
    if not (
        model.get("source_dataset_sha256") == model_dataset.get("dataset_sha256")
        and model.get("source_split_sha256") == split.get("split_sha256")
        and probability.get("source_dataset_sha256") == model_dataset.get("dataset_sha256")
        and probability.get("source_split_sha256") == split.get("split_sha256")
        and model_dataset.get("source_dataset_sha256") == feature_dataset.get("dataset_sha256")
        and payload.get("run_id") == feature_dataset.get("run_id")
    ):
        raise ValueError("Polymarket model provenance chain is inconsistent")
    if model_dataset.get("training_ready") is not True:
        raise ValueError("Polymarket model dataset was not training-ready")

    evidence = _as_mapping(
        payload.get("held_out_prediction_evidence"),
        "held-out prediction evidence",
    )
    if (
        evidence.get("schema_version") != _PREDICTION_SCHEMA
        or evidence.get("role") != "untouched_chronological_test"
    ):
        raise ValueError("held-out prediction evidence contract is invalid")
    predictions = _as_rows(evidence.get("rows"), "held-out predictions")
    if (
        evidence.get("row_count") != len(predictions)
        or evidence.get("rows_sha256") != _canonical_sha256(predictions)
        or tuple(evidence.get("assets", ())) != _ASSETS
    ):
        raise ValueError("held-out prediction evidence identity is invalid")
    sample_ids: set[str] = set()
    conditions: set[str] = set()
    time_groups: set[int] = set()
    condition_labels: dict[str, bool] = {}
    condition_assets: dict[str, str] = {}
    for row in predictions:
        sample_id = str(row.get("sample_id", ""))
        condition_id = str(row.get("condition_id", ""))
        asset = str(row.get("asset", ""))
        event_start = int(row.get("event_start_ms", -1))
        end_ms = int(row.get("end_ms", -1))
        decision_ms = int(row.get("decision_received_wall_ms", -1))
        horizon = int(row.get("horizon_seconds", -1))
        label = row.get("official_up")
        if (
            len(sample_id) != 64
            or sample_id in sample_ids
            or not condition_id
            or asset not in _ASSETS
            or end_ms != event_start + 300_000
            or decision_ms != end_ms - horizon * 1_000
            or horizon not in {30, 60, 120, 180, 240}
            or not isinstance(label, bool)
            or len(str(row.get("input_provenance_sha256", ""))) != 64
        ):
            raise ValueError("held-out prediction row is malformed")
        baseline = _finite_float(row.get("baseline_up_probability"), "baseline probability")
        model_probability = _finite_float(row.get("model_up_probability"), "model probability")
        weight = _finite_float(row.get("market_weight"), "market weight")
        if not (0.0 < baseline < 1.0 and 0.0 < model_probability < 1.0 and weight > 0.0):
            raise ValueError("held-out probability or market weight is outside its domain")
        if condition_id in condition_labels and condition_labels[condition_id] is not label:
            raise ValueError("held-out market has inconsistent official labels")
        if condition_id in condition_assets and condition_assets[condition_id] != asset:
            raise ValueError("held-out market has inconsistent assets")
        sample_ids.add(sample_id)
        conditions.add(condition_id)
        time_groups.add(event_start)
        condition_labels[condition_id] = label
        condition_assets[condition_id] = asset
    expected_groups = tuple(int(value) for value in split.get("test_group_starts_ms", ()))
    test_metrics = _as_mapping(
        _as_mapping(probability.get("baseline_metrics"), "baseline metrics").get("test"),
        "baseline test metrics",
    )
    if (
        set(expected_groups) != time_groups
        or evidence.get("market_count") != len(conditions)
        or evidence.get("time_group_count") != len(time_groups)
        or int(test_metrics.get("row_count", -1)) != len(predictions)
        or int(test_metrics.get("market_count", -1)) != len(conditions)
        or int(test_metrics.get("time_group_count", -1)) != len(time_groups)
    ):
        raise ValueError("held-out predictions do not match the frozen split")

    executions: dict[str, Mapping[str, Any]] = {}
    for policy, key in (("baseline", "baseline_execution"), ("model", "model_execution")):
        report = _as_mapping(payload.get(key), f"{policy} execution")
        _verify_claims(report, name=f"{policy} execution")
        _verify_embedded_digest(report, "report_sha256", name=f"{policy} execution")
        executions[policy] = report
    ai = _as_mapping(payload.get("ai"), "AI evidence")
    if ai.get("enabled") is True:
        veto = _as_mapping(ai.get("veto_report"), "AI veto report")
        ai_execution = _as_mapping(ai.get("execution"), "AI execution")
        _verify_claims(veto, name="AI veto report")
        _verify_claims(ai_execution, name="AI execution")
        _verify_embedded_digest(veto, "report_sha256", name="AI veto report")
        _verify_embedded_digest(ai_execution, "report_sha256", name="AI execution")
        if veto.get("advisory_only") is not True:
            raise ValueError("AI evidence is not advisory-only")
        executions["ai"] = ai_execution
    elif ai.get("enabled") is not False:
        raise ValueError("AI evidence enabled state is invalid")

    for policy, report in executions.items():
        _validate_execution_report(
            report,
            conditions=conditions,
            expected_time_group_count=len(time_groups),
            name=f"{policy} primary execution",
        )

    sensitivity = _as_mapping(
        payload.get("execution_latency_sensitivity"),
        "execution latency sensitivity",
    )
    _verify_claims(sensitivity, name="execution latency sensitivity")
    latency_values = sensitivity.get("network_latencies_ms")
    if (
        sensitivity.get("schema_version")
        != "polymarket-execution-latency-sensitivity-v1"
        or not isinstance(latency_values, list)
        or not latency_values
        or latency_values != sorted(set(latency_values))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 60_000
            for value in latency_values
        )
        or int(sensitivity.get("primary_network_latency_ms", -1))
        not in latency_values
    ):
        raise ValueError("execution latency sensitivity contract is invalid")
    sensitivity_policies = _as_mapping(
        sensitivity.get("policies"),
        "execution latency policies",
    )
    if set(sensitivity_policies) != set(executions):
        raise ValueError("execution latency sensitivity policy set is invalid")
    primary_latency = int(sensitivity["primary_network_latency_ms"])
    for policy, raw_reports in sensitivity_policies.items():
        reports = _as_mapping(raw_reports, f"{policy} latency reports")
        if set(reports) != {str(value) for value in latency_values}:
            raise ValueError(f"{policy} latency scenario set is incomplete")
        for latency in latency_values:
            scenario = _as_mapping(
                reports[str(latency)],
                f"{policy} {latency}ms execution",
            )
            if int(_as_mapping(scenario["config"], "scenario config")["submission_latency_ms"]) != latency:
                raise ValueError(f"{policy} latency scenario config drifted")
            _validate_execution_report(
                scenario,
                conditions=conditions,
                expected_time_group_count=len(time_groups),
                name=f"{policy} {latency}ms execution",
            )
            if (
                latency == primary_latency
                and scenario.get("report_sha256")
                != executions[str(policy)].get("report_sha256")
            ):
                raise ValueError(f"{policy} primary latency report does not match")
    return ValidatedPolymarketArtifact(
        payload=payload,
        artifact_sha256=claimed_artifact_sha256,
        predictions=tuple(predictions),
        executions=executions,
    )


def _probability_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    report = _as_mapping(payload["probability_report"], "probability report")
    rows: list[dict[str, object]] = []
    for treatment, key in (("market_implied", "baseline_metrics"), ("residual_model", "model_metrics")):
        roles = _as_mapping(report[key], f"{treatment} probability metrics")
        for role in ("train", "validation", "test"):
            metric = _as_mapping(roles[role], f"{treatment} {role} metrics")
            rows.append(
                {
                    "role": role,
                    "treatment": treatment,
                    "rows": metric["row_count"],
                    "markets": metric["market_count"],
                    "time_groups": metric["time_group_count"],
                    "effective_market_weight": metric["effective_market_weight"],
                    "weighted_log_loss": metric["weighted_log_loss"],
                    "weighted_brier_score": metric["weighted_brier_score"],
                    "weighted_calibration_error": metric["weighted_calibration_error"],
                    "weighted_accuracy": metric["weighted_accuracy"],
                    "weighted_sharpness": metric["weighted_sharpness"],
                }
            )
    return rows


def _prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "event_start_utc": _utc(row["event_start_ms"]),
            "end_utc": _utc(row["end_ms"]),
            "decision_received_utc": _utc(row["decision_received_wall_ms"]),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["event_start_ms"]),
                str(item["asset"]),
                -int(item["horizon_seconds"]),
            ),
        )
    ]


def _execution_summary_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    fields = (
        "evaluated_market_count",
        "signal_market_count",
        "attempted_order_count",
        "filled_order_count",
        "winning_order_count",
        "losing_order_count",
        "abstained_market_count",
        "gross_deployed_capital_quote",
        "gross_payout_quote",
        "total_fees_quote",
        "net_realized_pnl_quote",
        "initial_capital_quote",
        "final_equity_quote",
        "return_on_initial_capital",
        "return_on_deployed_capital",
        "maximum_drawdown_quote",
        "maximum_drawdown_fraction",
        "report_sha256",
    )
    return [
        {"policy": policy, **{field: report[field] for field in fields}}
        for policy, report in executions.items()
    ]


def _latency_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    sensitivity = _as_mapping(
        payload["execution_latency_sensitivity"],
        "execution latency sensitivity",
    )
    latencies = [int(value) for value in sensitivity["network_latencies_ms"]]
    policies = _as_mapping(sensitivity["policies"], "latency policies")
    rows: list[dict[str, object]] = []
    for policy, raw_reports in policies.items():
        reports = _as_mapping(raw_reports, f"{policy} latency reports")
        for latency in latencies:
            report = _as_mapping(reports[str(latency)], "latency report")
            delays = [
                int(value)
                for value in _as_mapping(
                    report["decision_delay_ms_by_condition"],
                    "decision delays",
                ).values()
            ]
            trades = report["trades"]
            rows.append(
                {
                    "policy": policy,
                    "network_latency_ms": latency,
                    "mean_decision_delay_ms": (
                        sum(delays) / len(delays) if delays else 0.0
                    ),
                    "maximum_decision_delay_ms": max(delays, default=0),
                    "evaluated_markets": report["evaluated_market_count"],
                    "signals": report["signal_market_count"],
                    "attempted_orders": report["attempted_order_count"],
                    "fills": report["filled_order_count"],
                    "indeterminate_orders": sum(
                        row["execution_state"] == "UNKNOWN" for row in trades
                    ),
                    "net_realized_pnl_quote": report["net_realized_pnl_quote"],
                    "return_on_initial_capital": report[
                        "return_on_initial_capital"
                    ],
                    "maximum_drawdown_fraction": report[
                        "maximum_drawdown_fraction"
                    ],
                    "report_sha256": report["report_sha256"],
                }
            )
    return rows


def _equity_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, report in executions.items():
        for point in report["equity_curve"]:
            rows.append(
                {
                    "policy": policy,
                    "settled_at_ms": point["settled_at_ms"],
                    "settled_at_utc": _utc(point["settled_at_ms"]),
                    "group_realized_pnl_quote": point["group_realized_pnl_quote"],
                    "equity_quote": point["equity_quote"],
                    "peak_equity_quote": point["peak_equity_quote"],
                    "drawdown_quote": point["drawdown_quote"],
                    "drawdown_fraction": point["drawdown_fraction"],
                }
            )
    if not rows:
        raise ValueError("execution reports contain no equity points")
    return rows


def _trade_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields: list[str] | None = None
    for policy, report in executions.items():
        for trade in report["trades"]:
            row = {
                "policy": policy,
                **dict(trade),
                "event_start_utc": _utc(trade["event_start_ms"]),
                "end_utc": _utc(trade["end_ms"]),
                "decision_received_utc": _utc(trade["decision_received_wall_ms"]),
            }
            if fields is None:
                fields = list(row)
            elif set(row) != set(fields):
                raise ValueError("execution trade schemas differ between policies")
            rows.append({key: row[key] for key in fields})
    if rows:
        return rows
    return [
        {
            "policy": "none",
            "execution_state": "NO_PROPOSALS",
            "execution_reason": "no positive after-fee proposal",
        }
    ]


def _per_asset_rows(
    predictions: Sequence[Mapping[str, Any]],
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    market_counts = {
        asset: len(
            {
                str(row["condition_id"])
                for row in predictions
                if row["asset"] == asset
            }
        )
        for asset in _ASSETS
    }
    rows: list[dict[str, object]] = []
    for policy, report in executions.items():
        trades = report["trades"]
        for asset in _ASSETS:
            selected = [row for row in trades if row["asset"] == asset]
            filled = [row for row in selected if row["execution_state"] == "FILLED"]
            pnl = sum(
                (_decimal(row["realized_pnl_quote"], "trade PnL") for row in filled),
                Decimal("0"),
            )
            fees = sum(
                (_decimal(row["fee_quote"], "trade fee") for row in filled),
                Decimal("0"),
            )
            deployed = sum(
                (
                    _decimal(row["average_fill_price"], "fill price")
                    * _decimal(row["filled_quantity"], "fill quantity")
                    for row in filled
                ),
                Decimal("0"),
            )
            rows.append(
                {
                    "policy": policy,
                    "asset": asset,
                    "evaluated_markets": market_counts[asset],
                    "signals": len(selected),
                    "fills": len(filled),
                    "wins": sum(_decimal(row["realized_pnl_quote"], "trade PnL") > 0 for row in filled),
                    "losses": sum(_decimal(row["realized_pnl_quote"], "trade PnL") < 0 for row in filled),
                    "fees_quote": str(fees),
                    "deployed_capital_quote": str(deployed),
                    "net_realized_pnl_quote": str(pnl),
                    "return_on_deployed_capital": str(pnl / deployed if deployed > 0 else Decimal("0")),
                }
            )
    return rows


def _ai_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    ai = _as_mapping(payload["ai"], "AI evidence")
    if ai.get("enabled") is not True:
        return [{"enabled": False, "reason": ai.get("reason", "operator_disabled")}]
    veto = _as_mapping(ai["veto_report"], "AI veto report")
    results = veto.get("results")
    if not isinstance(results, list) or not results:
        return [{"enabled": True, "reason": "no_positive_after_fee_proposals"}]
    rows: list[dict[str, object]] = []
    for result in results:
        decision = _as_mapping(result["decision"], "AI decision")
        rows.append(
            {
                "enabled": True,
                "case_id": result["case_id"],
                "condition_id": result["condition_id"],
                "model": result["model"],
                "latency_seconds": result["latency_seconds"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "valid": decision["valid"],
                "permits_entry": decision["permits_entry"],
                "reason_codes": ";".join(decision["reason_codes"]),
                "summary": decision["summary"],
                "failure_reason": decision["failure_reason"],
                "response_sha256": result["response_sha256"],
            }
        )
    return rows


def _svg_base(title: str, subtitle: str, description: str, *, height: int = 700) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(description)}</desc>",
        f'<rect width="1200" height="{height}" fill="{_COLORS["background"]}"/>',
        f'<text x="64" y="58" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="28" font-weight="700">{escape(title)}</text>',
        f'<text x="64" y="88" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="15">{escape(subtitle)}</text>',
    ]


def _probability_svg(rows: Sequence[Mapping[str, object]], *, start: str, end: str) -> str:
    selected = {
        (str(row["role"]), str(row["treatment"])): row
        for row in rows
        if row["role"] in {"validation", "test"}
    }
    categories = (
        ("Validation log loss", "validation", "weighted_log_loss"),
        ("Test log loss", "test", "weighted_log_loss"),
        ("Test Brier score", "test", "weighted_brier_score"),
    )
    values = [
        _finite_float(selected[(role, treatment)][metric], metric)
        for _label, role, metric in categories
        for treatment in ("market_implied", "residual_model")
    ]
    maximum = max(values) * 1.15 if max(values) > 0 else 1.0
    lines = _svg_base(
        "Proper-score comparison",
        f"Chronological validation and untouched test; held-out window {start} to {end}",
        "Weighted proper probability scores from probability-metrics.csv. Lower is better.",
    )
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    for index in range(5):
        value = maximum * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>')
        lines.append(f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.3f}</text>')
    group_width = width / len(categories)
    for index, (label, role, metric) in enumerate(categories):
        center = left + group_width * (index + 0.5)
        for offset, treatment in ((-48.0, "market_implied"), (48.0, "residual_model")):
            value = _finite_float(selected[(role, treatment)][metric], metric)
            height = (bottom - top) * value / maximum
            x = center + offset - 38.0
            y = bottom - height
            color = _COLORS["baseline" if treatment == "market_implied" else "model"]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="76" height="{height:.1f}" rx="3" fill="{color}"/>')
            lines.append(f'<text x="{x + 38:.1f}" y="{y - 10:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700">{value:.4f}</text>')
        lines.append(f'<text x="{center:.1f}" y="608" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="15">{escape(label)}</text>')
    lines.extend(
        (
            f'<rect x="390" y="645" width="18" height="18" rx="2" fill="{_COLORS["baseline"]}"/><text x="418" y="659" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Market-implied prior</text>',
            f'<rect x="625" y="645" width="18" height="18" rx="2" fill="{_COLORS["model"]}"/><text x="653" y="659" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Bounded residual model</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]], *, start: str, end: str) -> str:
    by_policy: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(
            (int(row["settled_at_ms"]), _finite_float(row["equity_quote"], "equity"))
        )
    for values in by_policy.values():
        values.sort()
    all_points = [point for values in by_policy.values() for point in values]
    min_time = min(point[0] for point in all_points)
    max_time = max(point[0] for point in all_points)
    min_value = min(point[1] for point in all_points)
    max_value = max(point[1] for point in all_points)
    padding = max((max_value - min_value) * 0.15, max(abs(max_value), 1.0) * 0.001)
    low, high = min_value - padding, max_value + padding
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    lines = _svg_base(
        "Held-out settled equity",
        f"Full-depth FOK replay with measured feed arrival and fixed latency; {start} to {end}",
        "Settled equity from equity-curves.csv. Every filled order is tied to an official resolution.",
    )
    for index in range(5):
        value = low + (high - low) * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>')
        lines.append(f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>')
    for policy in _POLICIES:
        values = by_policy.get(policy)
        if not values:
            continue
        points: list[str] = []
        for timestamp, equity in values:
            x = left + width * (timestamp - min_time) / max(1, max_time - min_time)
            y = bottom - (bottom - top) * (equity - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[policy]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    tick_count = 5
    for index in range(tick_count):
        timestamp = min_time + (max_time - min_time) * index // (tick_count - 1)
        x = left + width * index / (tick_count - 1)
        label = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc).strftime("%H:%MZ")
        lines.append(f'<text x="{x:.1f}" y="608" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{label}</text>')
    legend_x = 350
    for index, policy in enumerate(name for name in _POLICIES if name in by_policy):
        x = legend_x + index * 190
        lines.append(f'<line x1="{x}" y1="652" x2="{x + 28}" y2="652" stroke="{_COLORS[policy]}" stroke-width="4"/><text x="{x + 38}" y="657" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{policy.title()}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _asset_svg(rows: Sequence[Mapping[str, object]], *, start: str, end: str) -> str:
    policies = tuple(dict.fromkeys(str(row["policy"]) for row in rows))
    values = [_finite_float(row["net_realized_pnl_quote"], "asset PnL") for row in rows]
    extent = max(max((abs(value) for value in values), default=0.0) * 1.25, 1.0)
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    zero = (top + bottom) / 2
    lines = _svg_base(
        "After-fee realized PnL by asset",
        f"BTC, ETH, and SOL five-minute markets; held-out window {start} to {end}",
        "Net settled quote-currency PnL and fill counts from per-asset-execution.csv. No leverage.",
    )
    for value in (-extent, -extent / 2, 0.0, extent / 2, extent):
        y = zero - (bottom - top) * 0.5 * value / extent
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}" stroke-width="{2 if value == 0 else 1}"/>')
        lines.append(f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>')
    group_width = width / len(_ASSETS)
    bar_width = min(68.0, group_width / (len(policies) + 1))
    for asset_index, asset in enumerate(_ASSETS):
        center = left + group_width * (asset_index + 0.5)
        for policy_index, policy in enumerate(policies):
            row = next(item for item in rows if item["asset"] == asset and item["policy"] == policy)
            value = _finite_float(row["net_realized_pnl_quote"], "asset PnL")
            x = center + (policy_index - (len(policies) - 1) / 2) * (bar_width + 10) - bar_width / 2
            y_value = zero - (bottom - top) * 0.5 * value / extent
            y = min(zero, y_value)
            height = max(abs(y_value - zero), 1.0)
            color = _COLORS[policy]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}"/>')
            label_y = y - 9 if value >= 0 else y + height + 19
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{value:.2f} / {row["fills"]} fills</text>')
        lines.append(f'<text x="{center:.1f}" y="610" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="700">{asset}</text>')
    legend_x = 350
    for index, policy in enumerate(policies):
        x = legend_x + index * 190
        lines.append(f'<rect x="{x}" y="642" width="18" height="18" rx="2" fill="{_COLORS[policy]}"/><text x="{x + 28}" y="657" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{policy.title()}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _latency_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
) -> str:
    policies = tuple(dict.fromkeys(str(row["policy"]) for row in rows))
    latencies = sorted({int(row["network_latency_ms"]) for row in rows})
    values = [
        _finite_float(row["net_realized_pnl_quote"], "latency PnL")
        for row in rows
    ]
    low = min(values + [0.0])
    high = max(values + [0.0])
    padding = max((high - low) * 0.15, 1.0)
    low -= padding
    high += padding
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    lines = _svg_base(
        "Network-latency sensitivity",
        f"Causal full-depth FOK replay; held-out window {start} to {end}",
        "Settled after-fee PnL by predeclared network latency from latency-sensitivity.csv. AI lines also include measured model decision delay.",
    )
    for index in range(5):
        value = low + (high - low) * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>')
        lines.append(f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>')
    x_by_latency = {
        latency: left + width * index / max(1, len(latencies) - 1)
        for index, latency in enumerate(latencies)
    }
    for latency, x in x_by_latency.items():
        lines.append(f'<text x="{x:.1f}" y="608" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{latency} ms</text>')
    for policy in policies:
        policy_rows = {
            int(row["network_latency_ms"]): row
            for row in rows
            if row["policy"] == policy
        }
        points: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for latency in latencies:
            value = _finite_float(
                policy_rows[latency]["net_realized_pnl_quote"],
                "latency PnL",
            )
            x = x_by_latency[latency]
            y = bottom - (bottom - top) * (value - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
            coordinates.append((x, y, value))
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[policy]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y, value in coordinates:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{_COLORS[policy]}"/><text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{value:.2f}</text>')
    legend_x = 350
    for index, policy in enumerate(policies):
        x = legend_x + index * 190
        lines.append(f'<line x1="{x}" y1="652" x2="{x + 28}" y2="652" stroke="{_COLORS[policy]}" stroke-width="4"/><text x="{x + 38}" y="657" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{policy.title()}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _progress_rows(
    artifact: ValidatedPolymarketArtifact,
    prior_round_path: Path | None,
    round_number: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if prior_round_path is not None and prior_round_path.is_file():
        prior = _as_mapping(json.loads(prior_round_path.read_text(encoding="utf-8")), "prior round")
        counts = _as_mapping(_as_mapping(prior.get("dataset"), "prior dataset").get("labeled_market_counts"), "prior market counts")
        rows.append(
            {
                "round": int(prior.get("round", 2)),
                "status": prior.get("status", "pipeline_evidence"),
                "BTC_markets": counts.get("BTC", 0),
                "ETH_markets": counts.get("ETH", 0),
                "SOL_markets": counts.get("SOL", 0),
                "feature_rows": _as_mapping(prior["dataset"], "prior dataset").get("row_count", 0),
                "model_evaluated": False,
                "test_log_loss_delta": "",
                "model_net_realized_pnl_quote": "",
                "profitability_claim": False,
            }
        )
    payload = artifact.payload
    counts = _as_mapping(_as_mapping(payload["model_dataset"], "model dataset")["market_counts"], "market counts")
    report = _as_mapping(payload["probability_report"], "probability report")
    model_execution = artifact.executions["model"]
    rows.append(
        {
            "round": round_number,
            "status": "prospective_model_evaluated",
            "BTC_markets": counts["BTC"],
            "ETH_markets": counts["ETH"],
            "SOL_markets": counts["SOL"],
            "feature_rows": _as_mapping(payload["feature_dataset"], "feature dataset")["row_count"],
            "model_evaluated": True,
            "test_log_loss_delta": report["test_log_loss_delta"],
            "model_net_realized_pnl_quote": model_execution["net_realized_pnl_quote"],
            "profitability_claim": False,
        }
    )
    return rows


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    maximum = max(
        int(row[f"{asset}_markets"])
        for row in rows
        for asset in _ASSETS
    )
    maximum = max(maximum, 30)
    lines = _svg_base(
        "Prospective evidence progression",
        "Resolved market coverage by research round; production gate is 30 markets per asset",
        "Real prospective market counts from research-progress.csv. This is an evidence-coverage chart, not a profitability chart.",
    )
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    gate_y = bottom - (bottom - top) * 30 / maximum
    lines.append(f'<line x1="{left:.1f}" y1="{gate_y:.1f}" x2="{left + width:.1f}" y2="{gate_y:.1f}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="8 7"/>')
    lines.append(f'<text x="{left + width - 5:.1f}" y="{gate_y - 9:.1f}" text-anchor="end" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">30-market research gate</text>')
    group_width = width / len(rows)
    for round_index, row in enumerate(rows):
        center = left + group_width * (round_index + 0.5)
        for asset_index, asset in enumerate(_ASSETS):
            value = int(row[f"{asset}_markets"])
            bar_width = min(76.0, group_width / 5)
            x = center + (asset_index - 1) * (bar_width + 14) - bar_width / 2
            height = (bottom - top) * value / maximum
            y = bottom - height
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{_COLORS[asset]}"/>')
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 9:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700">{value}</text>')
        lines.append(f'<text x="{center:.1f}" y="610" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="700">Round {row["round"]}</text>')
        lines.append(f'<text x="{center:.1f}" y="634" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{escape(str(row["status"]))}</text>')
    for index, asset in enumerate(_ASSETS):
        x = 410 + index * 145
        lines.append(f'<rect x="{x}" y="656" width="18" height="18" rx="2" fill="{_COLORS[asset]}"/><text x="{x + 28}" y="671" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{asset}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _results_markdown(
    artifact: ValidatedPolymarketArtifact,
    round_number: int,
    start: str,
    end: str,
) -> str:
    payload = artifact.payload
    report = _as_mapping(payload["probability_report"], "probability report")
    baseline = artifact.executions["baseline"]
    model = artifact.executions["model"]
    ai = _as_mapping(payload["ai"], "AI evidence")
    ai_line = "AI was disabled for this experiment."
    if ai.get("enabled") is True:
        ai_execution = artifact.executions["ai"]
        uplift = _as_mapping(ai["uplift"], "AI uplift")
        ai_line = (
            f"The veto-only AI path filled {ai_execution['filled_order_count']} orders, "
            f"settled {ai_execution['net_realized_pnl_quote']} quote PnL, and its governed "
            f"uplift gate was **{'accepted' if uplift.get('accepted') else 'not accepted'}**."
        )
    gates = _as_mapping(payload["evidence_gates"], "evidence gates")
    gate_rows = "\n".join(
        f"| `{key}` | `{str(value).lower()}` |"
        for key, value in gates.items()
    )
    return f"""# Polymarket research round {round_number}

This report is generated from one hash-verified prospective artifact. It covers
BTC, ETH, and SOL five-minute markets from `{start}` through `{end}`. It is
research evidence only: no live trading, portfolio, leverage, or profitability
claim is made.

![Proper-score comparison](latest/charts/probability-quality.svg)

The selected candidate was `{report['selected_candidate']}`. Relative to the
market-implied prior, validation log-loss changed by
`{_finite_float(report['validation_log_loss_delta'], 'validation delta'):.8f}`
and untouched-test log-loss changed by
`{_finite_float(report['test_log_loss_delta'], 'test delta'):.8f}`. Lower is
better; the test Brier-score change was
`{_finite_float(report['test_brier_delta'], 'Brier delta'):.8f}`.

![Held-out settled equity](latest/charts/held-out-equity.svg)

The baseline replay filled `{baseline['filled_order_count']}` orders and settled
`{baseline['net_realized_pnl_quote']}` quote PnL. The residual-model replay filled
`{model['filled_order_count']}` orders and settled
`{model['net_realized_pnl_quote']}` quote PnL. These are short prospective
diagnostics after modeled dynamic fees and recorded depth, not evidence of a
durable edge. {ai_line}

![After-fee PnL by asset](latest/charts/per-asset-execution.svg)

![Network-latency sensitivity](latest/charts/latency-sensitivity.svg)

## Evidence gates

| Gate | Result |
|---|---:|
{gate_rows}

## Reproduction data

- [Experiment artifact](round-{round_number:03d}-prospective-model-experiment.json)
- [Held-out prediction rows](latest/tables/held-out-predictions.csv)
- [Probability metrics](latest/tables/probability-metrics.csv)
- [Execution summary](latest/tables/execution-summary.csv)
- [Settled equity curves](latest/tables/equity-curves.csv)
- [Execution ledger](latest/tables/trades.csv)
- [Per-asset execution](latest/tables/per-asset-execution.csv)
- [Latency sensitivity](latest/tables/latency-sensitivity.csv)
- [AI decisions](latest/tables/ai-decisions.csv)
- [Round progression](latest/tables/research-progress.csv)
- [Integrity manifest](latest/publication-integrity.json)
"""


def publish_polymarket_model_artifact(
    artifact_path: str | Path,
    research_root: str | Path,
    *,
    round_number: int = 3,
    prior_round_path: str | Path | None = None,
) -> PolymarketPublicationResult:
    """Publish current charts and their exact source tables from one artifact."""

    if round_number < 1:
        raise ValueError("publication round must be positive")
    source = Path(artifact_path).resolve()
    root = Path(research_root).resolve()
    validated = validate_polymarket_model_artifact(source)
    predictions = _prediction_rows(validated.predictions)
    start = _utc(min(int(row["event_start_ms"]) for row in validated.predictions))
    end = _utc(max(int(row["end_ms"]) for row in validated.predictions))
    probability_rows = _probability_rows(validated.payload)
    summary_rows = _execution_summary_rows(validated.executions)
    latency_rows = _latency_rows(validated.payload)
    equity_rows = _equity_rows(validated.executions)
    trade_rows = _trade_rows(validated.executions)
    per_asset_rows = _per_asset_rows(validated.predictions, validated.executions)
    ai_rows = _ai_rows(validated.payload)
    prior = Path(prior_round_path).resolve() if prior_round_path is not None else None
    progress_rows = _progress_rows(validated, prior, round_number)

    latest = root / "latest"
    charts = latest / "charts"
    tables = latest / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    current_chart_names = {
        "probability-quality.svg",
        "held-out-equity.svg",
        "per-asset-execution.svg",
        "latency-sensitivity.svg",
        "research-progress.svg",
    }
    for old_chart in charts.glob("*.svg"):
        if old_chart.name not in current_chart_names:
            old_chart.unlink()

    source_name = f"round-{round_number:03d}-prospective-model-experiment.json"
    source_target = root / source_name
    if source != source_target:
        source_target.parent.mkdir(parents=True, exist_ok=True)
        temporary = source_target.with_name(f".{source_target.name}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(source_target)

    table_payloads = {
        "held-out-predictions.csv": predictions,
        "probability-metrics.csv": probability_rows,
        "execution-summary.csv": summary_rows,
        "equity-curves.csv": equity_rows,
        "trades.csv": trade_rows,
        "per-asset-execution.csv": per_asset_rows,
        "latency-sensitivity.csv": latency_rows,
        "ai-decisions.csv": ai_rows,
        "research-progress.csv": progress_rows,
    }
    for name, rows in table_payloads.items():
        _write_csv(tables / name, rows)
    _write_text(charts / "probability-quality.svg", _probability_svg(probability_rows, start=start, end=end))
    _write_text(charts / "held-out-equity.svg", _equity_svg(equity_rows, start=start, end=end))
    _write_text(charts / "per-asset-execution.svg", _asset_svg(per_asset_rows, start=start, end=end))
    _write_text(
        charts / "latency-sensitivity.svg",
        _latency_svg(latency_rows, start=start, end=end),
    )
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))

    results_name = f"round-{round_number:03d}-prospective-model-results.md"
    results = _results_markdown(validated, round_number, start, end)
    _write_text(root / results_name, results)
    latest_readme = f"""# Polymarket research round {round_number}

![Held-out settled equity](charts/held-out-equity.svg)

The current publication is generated from prospective BTC/ETH/SOL evidence for
`{start}` through `{end}`. It includes market-implied, bounded residual-model,
and governed AI-veto diagnostics where available. No live-trading or durable
profitability claim is made.

[Read the measured results](../{results_name}) or inspect the
[integrity manifest](publication-integrity.json) and [source tables](tables/).

![Research progression](charts/research-progress.svg)
"""
    _write_text(latest / "README.md", latest_readme)

    generated = [
        source_target,
        root / results_name,
        latest / "README.md",
        *(tables / name for name in sorted(table_payloads)),
        *(charts / name for name in sorted(current_chart_names)),
    ]
    entries = []
    for path in generated:
        relative = path.relative_to(root).as_posix()
        entry: dict[str, object] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if path.suffix == ".csv":
            entry["row_count"] = len(table_payloads[path.name])
            entry["columns"] = list(table_payloads[path.name][0])
        entries.append(entry)
    manifest_body = {
        "schema_version": _PUBLICATION_SCHEMA,
        "round": round_number,
        "source_artifact": source_name,
        "source_artifact_sha256": validated.artifact_sha256,
        "held_out_start_utc": start,
        "held_out_end_utc": end,
        "assets": list(_ASSETS),
        "generated_artifacts": entries,
        "claims": {
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
    }
    manifest_sha256 = _canonical_sha256(manifest_body)
    manifest = {**manifest_body, "manifest_sha256": manifest_sha256}
    _write_json(latest / "publication-integrity.json", manifest)
    return PolymarketPublicationResult(
        round_number=round_number,
        research_root=str(root),
        source_artifact=str(source_target),
        artifact_sha256=validated.artifact_sha256,
        generated_files=tuple(str(path) for path in generated),
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "PolymarketPublicationResult",
    "ValidatedPolymarketArtifact",
    "publish_polymarket_model_artifact",
    "validate_polymarket_model_artifact",
]
