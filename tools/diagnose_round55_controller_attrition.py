"""Diagnose Round 55 controller attrition on already-consumed evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.bounded_alpha_lightgbm import (  # noqa: E402
    ConsensusDecisions,
    build_trade_plan,
    consensus_decisions,
    replay_trade_plan,
)
from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, SYMBOLS  # noqa: E402
from simple_ai_trading.stop_time_payoff_data import (  # noqa: E402
    EVENT_NAMES,
    STOP_EVENT,
    StopTimeSpecification,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 55
DESIGN_SCHEMA = "round-055-controller-attrition-diagnostic-design-v1"
DESIGN_SHA256 = "7f875029049b987338666b7ab41c3204650e7c3df36cf20f885be8eb8d953f42"
REPORT_SCHEMA = "round-055-bounded-alpha-report-v1"
SOURCE_REPORT_FILE_SHA256 = (
    "b556ff0302d230ef620a7bfb11ad49ad35f5d083ebc390d5174657a4f466fda2"
)
SOURCE_REPORT_CANONICAL_SHA256 = (
    "47dc22e987fff9cb508ff09fed2222e80391d18797496d4f2f9e476aee887919"
)
FEATURE_FILE_SHA256 = (
    "7cec94d8a025203652ed9b7db4fba6fd383f3dbd84c7edb5be794e4003471b14"
)
TIMESTAMP_FILE_SHA256 = (
    "b85c541d78c29f34a12b75f42656cf09ae4c6a47dcef5334c4d60d2e90e6591f"
)
METADATA_FILE_SHA256 = (
    "033480cd3b5669a060f297e7e477c2543a551602834914803bfd1127608d1135"
)
OUTPUT_SCHEMA = "round-055-controller-attrition-diagnostic-report-v1"
TREATMENTS = ("baseline_71", "ai_program_augmented")
CONTROLLERS = (
    "all_view_consensus",
    "majority_view_consensus",
    "pooled_nine_consensus",
)
INTERVALS = ("policy_development", "development_holdout")
DAY_MS = 24 * 60 * MINUTE_MS


@dataclass(frozen=True)
class ReplayPayoffView:
    """Exact persisted fields required by the frozen risk replay."""

    timestamps_ms: np.ndarray
    stop_bps: np.ndarray
    long_event_code: np.ndarray
    short_event_code: np.ndarray
    long_exit_time_ms: np.ndarray
    short_exit_time_ms: np.ndarray
    long_net_payoff_bps: np.ndarray
    short_net_payoff_bps: np.ndarray
    specification: StopTimeSpecification

    @property
    def timestamps(self) -> int:
        return int(self.timestamps_ms.size)


@dataclass(frozen=True)
class ControllerState:
    decisions: ConsensusDecisions
    model_actions: np.ndarray
    long_view_votes: np.ndarray
    short_view_votes: np.ndarray
    all_view_positive: np.ndarray
    all_view_positive_preferred: np.ndarray
    all_view_seed_supported: np.ndarray


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-055-controller-attrition-diagnostic-design.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round55-bounded-alpha-20260714-v2\report.json"
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round45-joint-sam-tcn-20260713-v1\derived_dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round55-controller-attrition-20260714-v1"
        ),
    )
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
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


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_canonical(
    value: Mapping[str, object],
    *,
    digest_key: str,
    expected: str,
    label: str,
) -> None:
    canonical = dict(value)
    claimed = canonical.pop(digest_key, None)
    actual = _canonical_sha256(canonical)
    if claimed != expected or actual != expected:
        raise ValueError(f"{label} canonical hash drifted")


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"{label} file hash drifted: {path}")


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _validate_implementation() -> dict[str, str]:
    relative = "tools/diagnose_round55_controller_attrition.py"
    blob = _git("rev-parse", f"HEAD:{relative}")
    working_blob = _git("hash-object", relative)
    if blob != working_blob:
        raise ValueError("diagnostic implementation differs from committed HEAD")
    return {"commit": _git("rev-parse", "HEAD"), "git_blob_oid": blob, "path": relative}


def _validate_sources(
    design_path: Path,
    report_path: Path,
    feature_cache: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    np.ndarray,
    tuple[str, ...],
    ReplayPayoffView,
    dict[str, np.ndarray],
]:
    design = _read_json(design_path, "diagnostic design")
    if design.get("schema_version") != DESIGN_SCHEMA or design.get("round") != ROUND:
        raise ValueError("diagnostic design identity drifted")
    _verify_canonical(
        design,
        digest_key="design_sha256",
        expected=DESIGN_SHA256,
        label="diagnostic design",
    )
    _verify_file(report_path, SOURCE_REPORT_FILE_SHA256, "Round 55 report")
    report = _read_json(report_path, "Round 55 report")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("round") != ROUND:
        raise ValueError("Round 55 report identity drifted")
    _verify_canonical(
        report,
        digest_key="report_sha256",
        expected=SOURCE_REPORT_CANONICAL_SHA256,
        label="Round 55 report",
    )
    source_contract = design["source_contract"]
    if (
        report.get("binding_sha256")
        != source_contract["execution_binding_sha256"]
        or report["data"]["dataset_sha256"]
        != source_contract["source_dataset_sha256"]
        or report["data"]["payoff"]["dataset_sha256"]
        != source_contract["payoff_dataset_sha256"]
        or report["data"]["selection_confirmation_or_terminal_rows_read"] is not False
        or report["data"]["forbidden_existing_rows_loaded"] is not False
    ):
        raise ValueError("Round 55 source boundary drifted")

    metadata_path = feature_cache / "metadata.json"
    features_path = feature_cache / "features.npy"
    timestamps_path = feature_cache / "timestamps_ms.npy"
    _verify_file(metadata_path, METADATA_FILE_SHA256, "feature metadata")
    _verify_file(features_path, FEATURE_FILE_SHA256, "feature matrix")
    _verify_file(timestamps_path, TIMESTAMP_FILE_SHA256, "feature timestamps")
    metadata = _read_json(metadata_path, "feature metadata")
    if (
        metadata.get("dataset_sha256") != source_contract["source_dataset_sha256"]
        or tuple(metadata.get("symbols", ())) != SYMBOLS
    ):
        raise ValueError("feature metadata source contract drifted")
    feature_names = tuple(str(name) for name in metadata["feature_names"])

    payoff_record = report["data"]["payoff"]
    payoff_path = Path(str(payoff_record["path"]))
    _verify_file(payoff_path, str(payoff_record["file_sha256"]), "payoff dataset")
    with np.load(payoff_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    specification = StopTimeSpecification(**payoff_record["specification"])
    specification.validate()
    payoff = ReplayPayoffView(
        timestamps_ms=arrays["timestamps_ms"],
        stop_bps=arrays["stop_bps"],
        long_event_code=arrays["long_event_code"],
        short_event_code=arrays["short_event_code"],
        long_exit_time_ms=arrays["long_exit_time_ms"],
        short_exit_time_ms=arrays["short_exit_time_ms"],
        long_net_payoff_bps=arrays["long_net_payoff_bps"],
        short_net_payoff_bps=arrays["short_net_payoff_bps"],
        specification=specification,
    )
    feature_timestamps = np.load(timestamps_path, mmap_mode="r", allow_pickle=False)
    feature_matrix = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if (
        payoff.timestamps != int(report["data"]["development_timestamps"])
        or feature_matrix.shape[0] < payoff.timestamps
        or feature_matrix.shape[1:] != (len(SYMBOLS), len(feature_names))
        or not np.array_equal(feature_timestamps[: payoff.timestamps], payoff.timestamps_ms)
    ):
        raise ValueError("diagnostic feature/payoff alignment drifted")
    features = np.asarray(feature_matrix[: payoff.timestamps], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("diagnostic feature prefix contains nonfinite values")

    predictions: dict[str, np.ndarray] = {}
    for treatment in TREATMENTS:
        record = report["model"]["artifacts"][treatment]
        expected = str(source_contract[f"{'baseline' if treatment == TREATMENTS[0] else 'ai'}_prediction_file_sha256"])
        prediction_path = Path(str(record["prediction_path"]))
        if str(record["prediction_sha256"]) != expected:
            raise ValueError(f"{treatment} prediction record drifted")
        _verify_file(prediction_path, expected, f"{treatment} prediction")
        value = np.asarray(
            np.load(prediction_path, mmap_mode="r", allow_pickle=False),
            dtype=np.float64,
        )
        if value.shape != (3, 3, payoff.timestamps, len(SYMBOLS), 2):
            raise ValueError(f"{treatment} prediction shape drifted")
        predictions[treatment] = value
    return design, report, features, feature_names, payoff, predictions


def _interval_masks(timestamps_ms: np.ndarray) -> dict[str, np.ndarray]:
    def mask(start: str, end: str) -> np.ndarray:
        start_ms = int(datetime.fromisoformat(start).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end).timestamp() * 1000)
        return (
            (timestamps_ms >= start_ms)
            & (timestamps_ms < end_ms)
            & (timestamps_ms + 61 * MINUTE_MS < end_ms)
        )

    return {
        "policy_development": mask("2024-07-01T00:00:00+00:00", "2024-09-01T00:00:00+00:00"),
        "development_holdout": mask("2024-09-01T00:00:00+00:00", "2024-10-01T00:00:00+00:00"),
    }


def _view_state(predictions: np.ndarray) -> tuple[np.ndarray, ...]:
    medians = np.median(predictions, axis=1)
    seed_positive = np.mean(predictions > 0.0, axis=1)
    long_votes = (
        (medians[..., 0] > 0.0)
        & (medians[..., 0] > medians[..., 1])
        & (seed_positive[..., 0] >= 2.0 / 3.0)
    )
    short_votes = (
        (medians[..., 1] > 0.0)
        & (medians[..., 1] > medians[..., 0])
        & (seed_positive[..., 1] >= 2.0 / 3.0)
    )
    long_positive = np.all(medians[..., 0] > 0.0, axis=0)
    short_positive = np.all(medians[..., 1] > 0.0, axis=0)
    all_view_positive = long_positive | short_positive
    all_view_positive_preferred = (
        long_positive & np.all(medians[..., 0] > medians[..., 1], axis=0)
    ) | (short_positive & np.all(medians[..., 1] > medians[..., 0], axis=0))
    all_view_seed_supported = np.all(long_votes, axis=0) | np.all(short_votes, axis=0)
    return (
        medians,
        seed_positive,
        long_votes,
        short_votes,
        all_view_positive,
        all_view_positive_preferred,
        all_view_seed_supported,
    )


def _market_masks(
    features: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    positions = {name: index for index, name in enumerate(feature_names)}
    liquidity = (
        features[..., positions["target_same_minute_of_week_liquidity_ratio"]]
        >= 0.5
    ) & (features[..., positions["target_quote_volume_vs_1440m_mean"]] >= 0.25)
    volatility = features[..., positions["target_realized_volatility_60m_bps"]] <= (
        2.5 * features[..., positions["target_realized_volatility_1440m_bps"]]
    )
    return liquidity, volatility


def _compose_decisions(
    *,
    model_actions: np.ndarray,
    model_score: np.ndarray,
    model_eligible: np.ndarray,
    liquidity: np.ndarray,
    volatility: np.ndarray,
    medians: np.ndarray,
    seed_positive: np.ndarray,
) -> ConsensusDecisions:
    eligible = model_eligible & liquidity & volatility
    actions = np.where(eligible, model_actions, 0).astype(np.int8)
    score = np.where(actions != 0, model_score, 0.0).astype(np.float64)
    return ConsensusDecisions(
        actions=actions,
        score_bps=score,
        model_eligible=model_eligible,
        liquidity_eligible=liquidity,
        volatility_eligible=volatility,
        view_median_bps=np.moveaxis(medians, 0, 2),
        seed_positive_fraction=np.moveaxis(seed_positive, 0, 2),
    )


def _controller_states(
    predictions: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, ControllerState]:
    (
        medians,
        seed_positive,
        long_vote,
        short_vote,
        all_positive,
        all_preferred,
        all_supported,
    ) = _view_state(predictions)
    long_votes = np.sum(long_vote, axis=0, dtype=np.int8)
    short_votes = np.sum(short_vote, axis=0, dtype=np.int8)
    liquidity, volatility = _market_masks(features, feature_names)

    exact = consensus_decisions(predictions, features, feature_names)
    exact_model_actions = np.zeros(features.shape[:2], dtype=np.int8)
    exact_model_actions[np.all(long_vote, axis=0)] = 1
    exact_model_actions[np.all(short_vote, axis=0)] = -1
    if not np.array_equal(exact.actions, np.where(exact.model_eligible & liquidity & volatility, exact_model_actions, 0)):
        raise RuntimeError("exact controller reconstruction differs")

    majority_long = (long_votes >= 2) & (short_votes < 2)
    majority_short = (short_votes >= 2) & (long_votes < 2)
    majority_actions = np.zeros(features.shape[:2], dtype=np.int8)
    majority_actions[majority_long] = 1
    majority_actions[majority_short] = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        long_score = np.nanmedian(np.where(long_vote, medians[..., 0], np.nan), axis=0)
        short_score = np.nanmedian(np.where(short_vote, medians[..., 1], np.nan), axis=0)
    majority_score = np.zeros(features.shape[:2], dtype=np.float64)
    majority_score[majority_long] = long_score[majority_long]
    majority_score[majority_short] = short_score[majority_short]
    majority_eligible = majority_long | majority_short
    majority = _compose_decisions(
        model_actions=majority_actions,
        model_score=majority_score,
        model_eligible=majority_eligible,
        liquidity=liquidity,
        volatility=volatility,
        medians=medians,
        seed_positive=seed_positive,
    )

    pooled_median = np.median(predictions, axis=(0, 1))
    pooled_positive = np.mean(predictions > 0.0, axis=(0, 1))
    pooled_long = (
        (pooled_median[..., 0] > 0.0)
        & (pooled_median[..., 0] > pooled_median[..., 1])
        & (pooled_positive[..., 0] >= 6.0 / 9.0)
    )
    pooled_short = (
        (pooled_median[..., 1] > 0.0)
        & (pooled_median[..., 1] > pooled_median[..., 0])
        & (pooled_positive[..., 1] >= 6.0 / 9.0)
    )
    pooled_actions = np.zeros(features.shape[:2], dtype=np.int8)
    pooled_actions[pooled_long] = 1
    pooled_actions[pooled_short] = -1
    pooled_score = np.zeros(features.shape[:2], dtype=np.float64)
    pooled_score[pooled_long] = pooled_median[..., 0][pooled_long]
    pooled_score[pooled_short] = pooled_median[..., 1][pooled_short]
    pooled_eligible = pooled_long | pooled_short
    pooled = _compose_decisions(
        model_actions=pooled_actions,
        model_score=pooled_score,
        model_eligible=pooled_eligible,
        liquidity=liquidity,
        volatility=volatility,
        medians=medians,
        seed_positive=seed_positive,
    )

    common = {
        "long_view_votes": long_votes,
        "short_view_votes": short_votes,
        "all_view_positive": all_positive,
        "all_view_positive_preferred": all_preferred,
        "all_view_seed_supported": all_supported,
    }
    return {
        "all_view_consensus": ControllerState(
            decisions=exact, model_actions=exact_model_actions, **common
        ),
        "majority_view_consensus": ControllerState(
            decisions=majority, model_actions=majority_actions, **common
        ),
        "pooled_nine_consensus": ControllerState(
            decisions=pooled, model_actions=pooled_actions, **common
        ),
    }


def _attrition_rows(
    treatment: str,
    controller: str,
    state: ControllerState,
    interval: str,
    interval_mask: np.ndarray,
) -> list[dict[str, object]]:
    decisions = state.decisions
    stages = (
        ("all_rows", np.ones_like(decisions.model_eligible), "denominator"),
        (
            "any_view_vote",
            (state.long_view_votes + state.short_view_votes) > 0,
            "independent_model_stage",
        ),
        (
            "two_view_side_agreement",
            np.maximum(state.long_view_votes, state.short_view_votes) >= 2,
            "independent_model_stage",
        ),
        ("all_view_positive", state.all_view_positive, "exact_rule_stage"),
        (
            "all_view_positive_preferred",
            state.all_view_positive_preferred,
            "exact_rule_stage",
        ),
        (
            "all_view_seed_supported",
            state.all_view_seed_supported,
            "exact_rule_stage",
        ),
        ("controller_model_eligible", decisions.model_eligible, "controller_stage"),
        ("liquidity_eligible", decisions.liquidity_eligible, "market_gate"),
        ("volatility_eligible", decisions.volatility_eligible, "market_gate"),
        (
            "model_plus_liquidity",
            decisions.model_eligible & decisions.liquidity_eligible,
            "controller_stage",
        ),
        (
            "model_plus_volatility",
            decisions.model_eligible & decisions.volatility_eligible,
            "controller_stage",
        ),
        ("market_eligible", decisions.actions != 0, "controller_stage"),
    )
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in [(-1, "ALL"), *enumerate(SYMBOLS)]:
        scope = np.broadcast_to(interval_mask[:, None], decisions.model_eligible.shape)
        if symbol_index >= 0:
            symbol_scope = np.zeros_like(scope)
            symbol_scope[:, symbol_index] = interval_mask
            scope = symbol_scope
        denominator = int(np.count_nonzero(scope))
        for stage, values, relation in stages:
            count = int(np.count_nonzero(scope & values))
            rows.append(
                {
                    "round": ROUND,
                    "treatment": treatment,
                    "interval": interval,
                    "controller": controller,
                    "symbol": symbol,
                    "stage": stage,
                    "relation": relation,
                    "eligible_rows": count,
                    "total_rows": denominator,
                    "eligible_fraction": count / denominator,
                }
            )
        for side, code in (("long", 1), ("short", -1)):
            count = int(np.count_nonzero(scope & (decisions.actions == code)))
            rows.append(
                {
                    "round": ROUND,
                    "treatment": treatment,
                    "interval": interval,
                    "controller": controller,
                    "symbol": symbol,
                    "stage": f"market_eligible_{side}",
                    "relation": "controller_side",
                    "eligible_rows": count,
                    "total_rows": denominator,
                    "eligible_fraction": count / denominator,
                }
            )
    return rows


def _vote_pattern_rows(
    treatment: str,
    state: ControllerState,
    interval: str,
    interval_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for long_votes in range(4):
            for short_votes in range(4):
                selected = (
                    interval_mask
                    & (state.long_view_votes[:, symbol_index] == long_votes)
                    & (state.short_view_votes[:, symbol_index] == short_votes)
                )
                count = int(np.count_nonzero(selected))
                if count:
                    rows.append(
                        {
                            "round": ROUND,
                            "treatment": treatment,
                            "interval": interval,
                            "symbol": symbol,
                            "long_view_votes": long_votes,
                            "short_view_votes": short_votes,
                            "rows": count,
                            "interval_rows": int(np.count_nonzero(interval_mask)),
                            "fraction": count / int(np.count_nonzero(interval_mask)),
                        }
                    )
    return rows


def _score_calibration_rows(
    treatment: str,
    controller: str,
    state: ControllerState,
    interval: str,
    interval_mask: np.ndarray,
    payoff: ReplayPayoffView,
) -> list[dict[str, object]]:
    time_index, symbol_index = np.where(interval_mask[:, None] & (state.decisions.actions != 0))
    if time_index.size == 0:
        return []
    actions = state.decisions.actions[time_index, symbol_index]
    scores = state.decisions.score_bps[time_index, symbol_index]
    actual = np.where(
        actions == 1,
        payoff.long_net_payoff_bps[time_index, symbol_index],
        payoff.short_net_payoff_bps[time_index, symbol_index],
    ).astype(np.float64)
    events = np.where(
        actions == 1,
        payoff.long_event_code[time_index, symbol_index],
        payoff.short_event_code[time_index, symbol_index],
    )
    order = np.lexsort((symbol_index, time_index, scores))
    rows = []
    for quintile, indexes in enumerate(np.array_split(order, 5), start=1):
        if indexes.size == 0:
            continue
        rows.append(
            {
                "round": ROUND,
                "treatment": treatment,
                "interval": interval,
                "controller": controller,
                "score_quintile": quintile,
                "signals": int(indexes.size),
                "minimum_score_bps": float(np.min(scores[indexes])),
                "maximum_score_bps": float(np.max(scores[indexes])),
                "mean_score_bps": float(np.mean(scores[indexes])),
                "mean_realized_stress_payoff_bps": float(np.mean(actual[indexes])),
                "median_realized_stress_payoff_bps": float(np.median(actual[indexes])),
                "positive_payoff_fraction": float(np.mean(actual[indexes] > 0.0)),
                "stop_loss_fraction": float(np.mean(events[indexes] == STOP_EVENT)),
            }
        )
    return rows


def _plan_rows(
    treatment: str,
    controller: str,
    interval: str,
    plan: object,
    payoff: ReplayPayoffView,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trades = []
    for index in range(plan.closed_trades):
        symbol_index = int(plan.symbol_index[index])
        trades.append(
            {
                "round": ROUND,
                "treatment": treatment,
                "interval": interval,
                "controller": controller,
                "decision_time_utc": datetime.fromtimestamp(
                    int(plan.decision_time_ms[index]) / 1000.0, UTC
                ).isoformat(),
                "exit_time_utc": datetime.fromtimestamp(
                    int(plan.exit_time_ms[index]) / 1000.0, UTC
                ).isoformat(),
                "symbol": SYMBOLS[symbol_index],
                "side": "long" if int(plan.side[index]) == 1 else "short",
                "size_fraction": float(plan.size_fraction[index]),
                "score_bps": float(plan.score_bps[index]),
                "stop_bps": float(plan.stop_bps[index]),
                "event": EVENT_NAMES[int(plan.event_code[index])],
                "stress_net_payoff_bps": float(plan.stress_net_payoff_bps[index]),
                "stress_initial_capital_return_fraction": float(
                    plan.size_fraction[index] * plan.stress_net_payoff_bps[index] / 10_000.0
                ),
            }
        )
    symbols = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        selected = plan.symbol_index == symbol_index
        count = int(np.count_nonzero(selected))
        returns = (
            plan.size_fraction[selected]
            * plan.stress_net_payoff_bps[selected]
            / 10_000.0
        )
        symbols.append(
            {
                "round": ROUND,
                "treatment": treatment,
                "interval": interval,
                "controller": controller,
                "symbol": symbol,
                "closed_trades": count,
                "active_days": int(
                    np.unique(plan.decision_time_ms[selected] // DAY_MS).size
                ),
                "total_initial_capital_return_fraction": float(np.sum(returns)),
                "mean_trade_initial_capital_bps": float(np.mean(returns) * 10_000.0)
                if count
                else 0.0,
                "mean_stress_payoff_bps": float(
                    np.mean(plan.stress_net_payoff_bps[selected])
                )
                if count
                else 0.0,
                "positive_trade_fraction": float(np.mean(returns > 0.0))
                if count
                else 0.0,
                "stop_loss_fraction": float(
                    np.mean(plan.event_code[selected] == STOP_EVENT)
                )
                if count
                else 0.0,
                "mean_stop_bps": float(np.mean(plan.stop_bps[selected]))
                if count
                else 0.0,
            }
        )
    del payoff
    return trades, symbols


def _overlap_rows(
    treatment: str,
    states: Mapping[str, ControllerState],
    interval: str,
    interval_mask: np.ndarray,
) -> list[dict[str, object]]:
    control = states[CONTROLLERS[0]].decisions.actions
    scope = interval_mask[:, None]
    control_mask = scope & (control != 0)
    rows = []
    for controller in CONTROLLERS[1:]:
        actions = states[controller].decisions.actions
        candidate_mask = scope & (actions != 0)
        union = control_mask | candidate_mask
        overlap = control_mask & candidate_mask
        rows.append(
            {
                "round": ROUND,
                "treatment": treatment,
                "interval": interval,
                "control": CONTROLLERS[0],
                "diagnostic_controller": controller,
                "control_signals": int(np.count_nonzero(control_mask)),
                "diagnostic_signals": int(np.count_nonzero(candidate_mask)),
                "overlap_signals": int(np.count_nonzero(overlap)),
                "same_side_overlap": int(np.count_nonzero(overlap & (control == actions))),
                "opposite_side_overlap": int(
                    np.count_nonzero(overlap & (control == -actions))
                ),
                "diagnostic_only_signals": int(
                    np.count_nonzero(candidate_mask & ~control_mask)
                ),
                "control_only_signals": int(
                    np.count_nonzero(control_mask & ~candidate_mask)
                ),
                "jaccard": float(np.count_nonzero(overlap) / np.count_nonzero(union))
                if np.any(union)
                else 1.0,
            }
        )
    return rows


def _reconcile_control(
    source_report: Mapping[str, object],
    treatment: str,
    interval: str,
    metrics: Mapping[str, object],
) -> None:
    expected = source_report["treatments"][treatment]["intervals"][interval]["stress"]
    exact_fields = (
        "closed_trades",
        "signals_before_cooldowns",
        "blocked_daily_loss_decisions",
        "blocked_cooldown_decisions",
        "active_days",
    )
    float_fields = (
        "total_return_fraction",
        "maximum_drawdown_fraction",
        "profit_factor",
        "win_rate",
        "mean_trade_initial_capital_bps",
        "mean_hourly_initial_capital_bps",
        "gross_round_trip_turnover_fraction",
        "maximum_position_fraction",
        "maximum_holding_minutes",
    )
    if any(metrics[field] != expected[field] for field in exact_fields):
        raise RuntimeError(f"{treatment} {interval} control counts differ")
    for field in float_fields:
        left = metrics[field]
        right = expected[field]
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"{treatment} {interval} {field} differs")
        elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"{treatment} {interval} {field} differs")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"diagnostic table is empty: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _clean_output(path: Path) -> None:
    resolved = path.resolve()
    if "round55-controller-attrition" not in resolved.name.lower():
        raise ValueError("diagnostic output directory name is not specific enough")
    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def diagnose(
    *,
    design_path: Path,
    report_path: Path,
    feature_cache: Path,
    output_dir: Path,
) -> dict[str, object]:
    implementation = _validate_implementation()
    design, source_report, features, names, payoff, predictions = _validate_sources(
        design_path, report_path, feature_cache
    )
    intervals = _interval_masks(payoff.timestamps_ms)
    expected_rows = {
        name: int(source_report["chronology"][name]["timestamps"])
        for name in INTERVALS
    }
    if any(int(np.count_nonzero(intervals[name])) != expected_rows[name] for name in INTERVALS):
        raise ValueError("diagnostic interval chronology drifted")

    attrition_rows: list[dict[str, object]] = []
    vote_rows: list[dict[str, object]] = []
    economics_rows: list[dict[str, object]] = []
    symbol_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for treatment in TREATMENTS:
        states = _controller_states(predictions[treatment], features, names)
        treatment_summary: dict[str, object] = {}
        for interval in INTERVALS:
            mask = intervals[interval]
            vote_rows.extend(
                _vote_pattern_rows(treatment, states[CONTROLLERS[0]], interval, mask)
            )
            overlap_rows.extend(_overlap_rows(treatment, states, interval, mask))
            interval_summary: dict[str, object] = {}
            for controller in CONTROLLERS:
                state = states[controller]
                attrition_rows.extend(
                    _attrition_rows(treatment, controller, state, interval, mask)
                )
                calibration_rows.extend(
                    _score_calibration_rows(
                        treatment, controller, state, interval, mask, payoff
                    )
                )
                plan = build_trade_plan(payoff, state.decisions, mask)
                replay = replay_trade_plan(
                    payoff,
                    plan,
                    mask,
                    scenario="stress",
                    round_trip_execution_charge_bps=16.0,
                )
                metrics = {
                    "round": ROUND,
                    "treatment": treatment,
                    "interval": interval,
                    "controller": controller,
                    **dict(replay.metrics),
                    "stop_loss_fraction": float(
                        np.mean(plan.event_code == STOP_EVENT)
                    )
                    if plan.closed_trades
                    else 0.0,
                }
                economics_rows.append(metrics)
                trades, symbols = _plan_rows(
                    treatment, controller, interval, plan, payoff
                )
                trade_rows.extend(trades)
                symbol_rows.extend(symbols)
                if controller == CONTROLLERS[0]:
                    _reconcile_control(
                        source_report, treatment, interval, replay.metrics
                    )
                interval_summary[controller] = {
                    "market_eligible_signals": int(
                        np.count_nonzero(mask[:, None] & (state.decisions.actions != 0))
                    ),
                    "closed_trades": plan.closed_trades,
                    "active_days": replay.metrics["active_days"],
                    "stress_return_fraction": replay.metrics["total_return_fraction"],
                    "maximum_drawdown_fraction": replay.metrics[
                        "maximum_drawdown_fraction"
                    ],
                    "profit_factor": replay.metrics["profit_factor"],
                    "stop_loss_fraction": metrics["stop_loss_fraction"],
                }
            treatment_summary[interval] = interval_summary
        summary[treatment] = treatment_summary

    _clean_output(output_dir)
    table_rows = {
        "attrition.csv": attrition_rows,
        "vote-patterns.csv": vote_rows,
        "controller-economics.csv": economics_rows,
        "symbol-economics.csv": symbol_rows,
        "score-calibration.csv": calibration_rows,
        "controller-overlap.csv": overlap_rows,
        "trades.csv": trade_rows,
    }
    for name, rows in table_rows.items():
        _write_csv(output_dir / name, rows)
    artifacts = [
        _artifact(output_dir / name, output_dir) for name in sorted(table_rows)
    ]
    report: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "round": ROUND,
        "diagnostic_design_sha256": DESIGN_SHA256,
        "implementation": implementation,
        "source": {
            "report_file_sha256": SOURCE_REPORT_FILE_SHA256,
            "report_canonical_sha256": SOURCE_REPORT_CANONICAL_SHA256,
            "source_dataset_sha256": source_report["data"]["dataset_sha256"],
            "payoff_dataset_sha256": source_report["data"]["payoff"][
                "dataset_sha256"
            ],
            "selection_confirmation_or_terminal_rows_read": False,
            "forbidden_existing_rows_loaded": False,
            "synthetic_rows": 0,
        },
        "trial_accounting": {
            "controller_rules": len(CONTROLLERS),
            "new_model_refits": 0,
            "threshold_trials": 0,
            "intervals_inspected": len(INTERVALS),
            "treatments_inspected": len(TREATMENTS),
            "selection_contaminated": True,
        },
        "claims": {
            "status": "diagnostic_only",
            "profitability_claim": False,
            "promotion_authority": False,
            "ai_uplift_claim": False,
            "untouched_data_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
        },
        "control_reproduced_exactly": True,
        "summary": summary,
        "artifacts": artifacts,
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(output_dir / "report.json", report, indent=2, sort_keys=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = diagnose(
        design_path=arguments.design.resolve(),
        report_path=arguments.report.resolve(),
        feature_cache=arguments.feature_cache.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": report["round"],
                "status": report["claims"]["status"],
                "report_sha256": report["report_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
