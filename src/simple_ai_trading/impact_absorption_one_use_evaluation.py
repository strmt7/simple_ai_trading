"""Frozen one-use predictive and economic evaluation for Round 73."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable

import duckdb
import numpy as np

from .depth_stress_screen import (
    benjamini_hochberg_q_values,
    paired_blocked_permutation_test,
)
from .impact_absorption_cohort import (
    ROUND73_SHOCK_ANCHOR_TABLE,
    audit_round73_shock_cohort,
)
from .impact_absorption_corpus import ROUND73_CORPUS_RUN_TABLE
from .impact_absorption_evaluation_store import (
    ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
    Round73EvaluationAccessClaim,
    Round73StoredEvaluationResult,
    claim_round73_evaluation_access,
    finalize_interrupted_round73_evaluation,
    load_round73_claimed_pretest,
    load_round73_claimed_symbol_artifacts,
    load_round73_test_prediction,
    persist_round73_evaluation_result,
    persist_round73_test_prediction,
)
from .impact_absorption_grid_store import ROUND73_GRID_VECTOR_TABLE
from .impact_absorption_model_dataset import (
    ROUND73_OBSERVED_STATUS,
    ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
    ROUND73_PRE_ENTRY_ABORT_STATUS,
    ROUND73_PRIMARY_ENTRY_DELAY_MS,
    ROUND73_PRIMARY_HORIZON_MS,
    ROUND73_PRIMARY_REFERENCE_NOTIONAL,
    ROUND73_RIGHT_CENSORED_STATUS,
    classify_round73_operational_outcome,
)
from .impact_absorption_model_features import ROUND73_EVALUATION_CONTRACT_SHA256
from .impact_absorption_model_slice import (
    ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES,
    ROUND73_MODEL_SLICE_FETCH_ROWS,
    ROUND73_MODEL_SLICE_SCHEMA_VERSION,
    iter_round73_staged_symbol_slices,
)
from .impact_absorption_pretest_contract import round73_repository_state
from .impact_absorption_shallow_model import (
    ProgressCallback,
    ROUND73_MODEL_SEED,
    ROUND73_SHALLOW_CANDIDATES,
    decode_round73_prediction_artifact,
    encode_round73_prediction_artifact,
    predict_round73_frozen_symbol_model,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_store_v2 import (
    ROUND73_TARGET_V2_ENTRY_DELAYS_MS,
    ROUND73_TARGET_V2_HORIZONS_MS,
    ROUND73_TARGET_V2_REFERENCE_NOTIONALS,
    _STUDY_ID,
    _stream_hash,
)
from .impact_absorption_target_store_v3 import (
    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
    ROUND73_TARGET_V3_OPTION_TABLE,
    ROUND73_TARGET_V3_ROLE_RUN_TABLE,
    ROUND73_TARGET_V3_TEST_STUDY_TABLE,
    audit_round73_role_targets,
)
from .impact_absorption_targets import ROUND73_TARGET_MAX_STATE_LATENESS_NS
from .price_discovery_evaluation import (
    binary_loss_rows,
    binary_predictive_metrics,
    continuous_loss_rows,
    continuous_predictive_metrics,
)


ROUND73_ONE_USE_EVALUATION_IMPLEMENTATION_VERSION = "round-073-one-use-evaluator-v2"
ROUND73_EVALUATION_PERMUTATION_DRAWS = 10_000
ROUND73_EVALUATION_BOOTSTRAP_DRAWS = 10_000
ROUND73_EVALUATION_MAXIMUM_Q_VALUE = 0.05
ROUND73_EVALUATION_MINIMUM_RELATIVE_PROPER_SCORE_IMPROVEMENT = 0.002
ROUND73_EVALUATION_MINIMUM_CONTINUOUS_MSE_SKILL = 0.001
ROUND73_EVALUATION_MINIMUM_COMPLETED_TRADES = 100
ROUND73_EVALUATION_MINIMUM_DAILY_TRADES = 5
ROUND73_EVALUATION_MINIMUM_INDEPENDENT_SYMBOLS = 2
ROUND73_EVALUATION_MAXIMUM_CONCURRENT_POSITIONS = 3

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_DAY_NS = 86_400_000_000_000
_SCENARIO_REASON_NAMES = (
    "eligible",
    "quantity_filter",
    "entry_state_late",
    "entry_capacity",
    "entry_minimum_notional",
    "funding_boundary",
    "source_run_boundary",
    "path_capacity",
    "exit_state_late",
    "exit_capacity",
)
_REASON_TO_CODE = {reason: code for code, reason in enumerate(_SCENARIO_REASON_NAMES)}
_COMPARISON_STAGES = (
    ("linear_vs_controls", "prevalence_zero", "linear_l1_tape"),
    ("l1_tape_vs_linear", "linear_l1_tape", "l1_tape"),
    ("l2_state_vs_l1_tape", "l1_tape", "l2_state"),
    ("impact_absorption_vs_l2_state", "l2_state", "impact_absorption"),
    ("l2_state_vs_linear", "linear_l1_tape", "l2_state"),
    ("impact_absorption_vs_linear", "linear_l1_tape", "impact_absorption"),
)
_PRIMARY_SCENARIO_KEY = "delay-500ms_horizon-60000ms_notional-1000"
_DELAY_STRESS_SCENARIO_KEY = "delay-1000ms_horizon-60000ms_notional-1000"
_SCENARIO_ROW_ALLOCATION_BYTES = sum(
    np.dtype(dtype).itemsize
    for dtype in (
        "S16",
        "i8",
        "i8",
        "i8",
        "i8",
        "i1",
        "u1",
        "u1",
        "f8",
        "f8",
        "f8",
        "i8",
        "i8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
    )
)

RepositoryStateFunction = Callable[[str | Path], Mapping[str, object]]


def _report_progress(
    callback: ProgressCallback | None,
    event: str,
    details: Mapping[str, object],
) -> None:
    if callback is None:
        return
    try:
        callback(event, details)
    except Exception:  # noqa: BLE001 - diagnostics cannot alter one-use evidence
        return


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _fixed_binary_row(value: np.ndarray, index: int, width: int) -> bytes:
    raw = np.ascontiguousarray(value[index : index + 1]).view(np.uint8)
    if raw.size != width:
        raise ValueError("Round 73 fixed binary row width differs")
    return raw.tobytes()


@dataclass(frozen=True, order=True)
class Round73EvaluationScenario:
    entry_delay_ms: int
    horizon_ms: int
    reference_quote_notional: int

    @property
    def key(self) -> str:
        return (
            f"delay-{self.entry_delay_ms}ms_horizon-{self.horizon_ms}ms_"
            f"notional-{self.reference_quote_notional}"
        )

    @property
    def primary(self) -> bool:
        return self.key == _PRIMARY_SCENARIO_KEY

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_key": self.key,
            "entry_delay_ms": self.entry_delay_ms,
            "holding_horizon_ms": self.horizon_ms,
            "reference_quote_notional": self.reference_quote_notional,
            "primary": self.primary,
        }


ROUND73_EVALUATION_SCENARIOS = tuple(
    Round73EvaluationScenario(int(delay), int(horizon), int(notional))
    for delay in ROUND73_TARGET_V2_ENTRY_DELAYS_MS
    for horizon in ROUND73_TARGET_V2_HORIZONS_MS
    for notional in ROUND73_TARGET_V2_REFERENCE_NOTIONALS
)


@dataclass(frozen=True)
class _ScenarioRows:
    symbol: str
    scenario: Round73EvaluationScenario
    source_rows_sha256: str
    run_id_binary: np.ndarray
    anchor_index: np.ndarray
    anchor_monotonic_ns: np.ndarray
    anchor_wall_ns: np.ndarray
    utc_day_ordinal: np.ndarray
    side_sign: np.ndarray
    outcome_status: np.ndarray
    outcome_reason_code: np.ndarray
    binary_target: np.ndarray
    continuous_target_bps: np.ndarray
    net_payoff_quote: np.ndarray
    actual_entry_wall_ns: np.ndarray
    actual_exit_wall_ns: np.ndarray
    entry_quote_notional: np.ndarray
    exit_quote_notional: np.ndarray
    maximum_adverse_excursion_bps: np.ndarray
    maximum_favorable_excursion_bps: np.ndarray
    maximum_spread_bps: np.ndarray
    minimum_exit_side_capacity_ratio: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.anchor_index.size)

    @property
    def label_mask(self) -> np.ndarray:
        return _readonly(
            np.isin(
                self.outcome_status,
                (ROUND73_OBSERVED_STATUS, ROUND73_PRE_ENTRY_ABORT_STATUS),
            )
        )

    def validate(self) -> None:
        rows = self.rows
        arrays = (
            self.run_id_binary,
            self.anchor_index,
            self.anchor_monotonic_ns,
            self.anchor_wall_ns,
            self.utc_day_ordinal,
            self.side_sign,
            self.outcome_status,
            self.outcome_reason_code,
            self.binary_target,
            self.continuous_target_bps,
            self.net_payoff_quote,
            self.actual_entry_wall_ns,
            self.actual_exit_wall_ns,
            self.entry_quote_notional,
            self.exit_quote_notional,
            self.maximum_adverse_excursion_bps,
            self.maximum_favorable_excursion_bps,
            self.maximum_spread_bps,
            self.minimum_exit_side_capacity_ratio,
        )
        labels = self.label_mask
        observed = self.outcome_status == ROUND73_OBSERVED_STATUS
        if (
            self.symbol not in IMPACT_CAPTURE_SYMBOLS
            or self.scenario not in ROUND73_EVALUATION_SCENARIOS
            or _SHA256.fullmatch(self.source_rows_sha256) is None
            or rows <= 0
            or rows % 2
            or any(array.shape != (rows,) for array in arrays)
            or any(array.flags.writeable for array in arrays)
            or np.any((self.side_sign != 1) & (self.side_sign != -1))
            or np.any(~np.isfinite(self.binary_target[labels]))
            or np.any(~np.isfinite(self.continuous_target_bps[labels]))
            or np.any(np.isfinite(self.binary_target[~labels]))
            or np.any(np.isfinite(self.continuous_target_bps[~labels]))
            or np.any(~np.isfinite(self.net_payoff_quote[observed]))
            or np.any(np.isfinite(self.net_payoff_quote[~observed]))
            or np.any(
                self.actual_entry_wall_ns[observed] < self.anchor_wall_ns[observed]
            )
            or np.any(
                self.actual_exit_wall_ns[observed]
                <= self.actual_entry_wall_ns[observed]
            )
            or np.any(self.entry_quote_notional[observed] <= 0.0)
            or np.any(self.exit_quote_notional[observed] <= 0.0)
            or np.any(~np.isfinite(self.maximum_adverse_excursion_bps[observed]))
            or np.any(~np.isfinite(self.maximum_favorable_excursion_bps[observed]))
            or np.any(~np.isfinite(self.maximum_spread_bps[observed]))
            or np.any(~np.isfinite(self.minimum_exit_side_capacity_ratio[observed]))
            or np.any(
                self.maximum_adverse_excursion_bps[observed]
                > self.continuous_target_bps[observed]
            )
            or np.any(
                self.continuous_target_bps[observed]
                > self.maximum_favorable_excursion_bps[observed]
            )
            or np.any(self.maximum_spread_bps[observed] < 0.0)
            or np.any(self.minimum_exit_side_capacity_ratio[observed] < 1.0)
            or np.any(np.isfinite(self.maximum_adverse_excursion_bps[~observed]))
            or np.any(np.isfinite(self.maximum_favorable_excursion_bps[~observed]))
            or np.any(np.isfinite(self.maximum_spread_bps[~observed]))
            or np.any(np.isfinite(self.minimum_exit_side_capacity_ratio[~observed]))
        ):
            raise ValueError("Round 73 scenario rows are invalid")
        even = slice(0, None, 2)
        odd = slice(1, None, 2)
        if (
            not np.array_equal(self.run_id_binary[even], self.run_id_binary[odd])
            or not np.array_equal(self.anchor_index[even], self.anchor_index[odd])
            or not np.array_equal(
                self.anchor_monotonic_ns[even], self.anchor_monotonic_ns[odd]
            )
            or not np.array_equal(self.anchor_wall_ns[even], self.anchor_wall_ns[odd])
            or np.any(self.side_sign[even] != 1)
            or np.any(self.side_sign[odd] != -1)
        ):
            raise ValueError("Round 73 scenario action pairs differ")


def _model_slice_source_digest(
    *,
    study_id: str,
    symbol: str,
    cohort_manifest_sha256: str,
    target_study_manifest_sha256: str,
    row_count: int,
) -> object:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": ROUND73_MODEL_SLICE_SCHEMA_VERSION,
                "study_id": study_id,
                "role_scope": "test",
                "symbol": symbol,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "target_study_manifest_sha256": target_study_manifest_sha256,
                "row_count": row_count,
            }
        ).encode("ascii")
    )


def _scenario_row_count(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    symbol: str,
    scenario: Round73EvaluationScenario,
) -> int:
    return int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
            JOIN {ROUND73_TARGET_V3_OPTION_TABLE} o
              ON o.study_id = s.study_id AND o.run_id = s.run_id
             AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
            WHERE s.study_id = ? AND s.symbol = ? AND s.role = 'test'
              AND o.entry_delay_ms = ? AND o.horizon_ms = ?
              AND o.reference_quote_notional = ?
            """,
            [
                study_id,
                symbol,
                scenario.entry_delay_ms,
                scenario.horizon_ms,
                scenario.reference_quote_notional,
            ],
        ).fetchone()[0]
    )


def _validate_scenario_memory_budget(
    memory_budget_bytes: int,
    *,
    rows: int,
) -> int:
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
        or rows <= 0
    ):
        raise ValueError("Round 73 scenario memory budget is invalid")
    required = rows * _SCENARIO_ROW_ALLOCATION_BYTES
    if required > memory_budget_bytes:
        raise MemoryError(
            "Round 73 scenario rows exceed the explicit memory budget: "
            f"required={required} budget={memory_budget_bytes}"
        )
    return required


def _load_scenario_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    symbol: str,
    scenario: Round73EvaluationScenario,
    cohort_manifest_sha256: str,
    test_study_manifest_sha256: str,
    memory_budget_bytes: int,
) -> _ScenarioRows:
    row_count = _scenario_row_count(
        connection,
        study_id=study_id,
        symbol=symbol,
        scenario=scenario,
    )
    if row_count <= 0 or row_count % 2:
        raise ValueError("Round 73 scenario target rows are incomplete")
    _validate_scenario_memory_budget(memory_budget_bytes, rows=row_count)
    arrays = {
        "run_id_binary": np.empty(row_count, dtype="S16"),
        "anchor_index": np.empty(row_count, dtype=np.int64),
        "anchor_monotonic_ns": np.empty(row_count, dtype=np.int64),
        "anchor_wall_ns": np.empty(row_count, dtype=np.int64),
        "utc_day_ordinal": np.empty(row_count, dtype=np.int64),
        "side_sign": np.empty(row_count, dtype=np.int8),
        "outcome_status": np.empty(row_count, dtype=np.uint8),
        "outcome_reason_code": np.empty(row_count, dtype=np.uint8),
        "binary_target": np.empty(row_count, dtype=np.float64),
        "continuous_target_bps": np.empty(row_count, dtype=np.float64),
        "net_payoff_quote": np.empty(row_count, dtype=np.float64),
        "actual_entry_wall_ns": np.empty(row_count, dtype=np.int64),
        "actual_exit_wall_ns": np.empty(row_count, dtype=np.int64),
        "entry_quote_notional": np.empty(row_count, dtype=np.float64),
        "exit_quote_notional": np.empty(row_count, dtype=np.float64),
        "maximum_adverse_excursion_bps": np.empty(row_count, dtype=np.float64),
        "maximum_favorable_excursion_bps": np.empty(row_count, dtype=np.float64),
        "maximum_spread_bps": np.empty(row_count, dtype=np.float64),
        "minimum_exit_side_capacity_ratio": np.empty(
            row_count,
            dtype=np.float64,
        ),
    }
    digest = _model_slice_source_digest(
        study_id=study_id,
        symbol=symbol,
        cohort_manifest_sha256=cohort_manifest_sha256,
        target_study_manifest_sha256=test_study_manifest_sha256,
        row_count=row_count,
    )
    cursor = connection.execute(
        f"""
        SELECT s.run_id, s.anchor_index, s.anchor_monotonic_ns,
               s.anchor_wall_ns, s.utc_day, s.selected_anchor_sha256,
               c.coverage_end_wall_ns, v.vector_sha256,
               p.option_sha256, p.selected_anchor_sha256,
               o.side, o.eligible, o.ineligible_reasons_json,
               o.positive_net_payoff, o.net_payoff_bps, o.net_payoff_quote,
               o.actual_entry_monotonic_ns, o.actual_exit_monotonic_ns,
               o.entry_quote_notional, o.exit_quote_notional,
               o.option_sha256, o.selected_anchor_sha256,
               o.maximum_adverse_excursion_bps,
               o.maximum_favorable_excursion_bps,
               o.maximum_spread_bps,
               o.minimum_exit_side_capacity_ratio
        FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
        JOIN {ROUND73_CORPUS_RUN_TABLE} c USING (run_id)
        JOIN {ROUND73_GRID_VECTOR_TABLE} v
          USING (run_id, symbol, anchor_index)
        JOIN {ROUND73_TARGET_V3_OPTION_TABLE} p
          ON p.study_id = s.study_id AND p.run_id = s.run_id
         AND p.symbol = s.symbol AND p.anchor_index = s.anchor_index
         AND p.entry_delay_ms = {ROUND73_PRIMARY_ENTRY_DELAY_MS}
         AND p.horizon_ms = {ROUND73_PRIMARY_HORIZON_MS}
         AND p.reference_quote_notional = {ROUND73_PRIMARY_REFERENCE_NOTIONAL}
        JOIN {ROUND73_TARGET_V3_OPTION_TABLE} o
          ON o.study_id = s.study_id AND o.run_id = s.run_id
         AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
         AND o.side = p.side
        WHERE s.study_id = ? AND s.symbol = ? AND s.role = 'test'
          AND o.entry_delay_ms = ? AND o.horizon_ms = ?
          AND o.reference_quote_notional = ?
        ORDER BY s.anchor_wall_ns, s.run_id, s.anchor_index,
                 CASE o.side WHEN 'long' THEN 0 WHEN 'short' THEN 1 ELSE 2 END
        """,
        [
            study_id,
            symbol,
            scenario.entry_delay_ms,
            scenario.horizon_ms,
            scenario.reference_quote_notional,
        ],
    )
    offset = 0
    while True:
        rows = cursor.fetchmany(ROUND73_MODEL_SLICE_FETCH_ROWS)
        if not rows:
            break
        for row in rows:
            run_id = str(row[0]).strip().lower()
            anchor_index = int(row[1])
            anchor_monotonic_ns = int(row[2])
            anchor_wall_ns = int(row[3])
            selected_anchor_hash = str(row[5]).strip().lower()
            coverage_end_wall_ns = int(row[6])
            vector_hash = str(row[7]).strip().lower()
            primary_option_hash = str(row[8]).strip().lower()
            scenario_option_hash = str(row[20]).strip().lower()
            side = str(row[10])
            if (
                _RUN_ID.fullmatch(run_id) is None
                or _SHA256.fullmatch(selected_anchor_hash) is None
                or str(row[9]).strip().lower() != selected_anchor_hash
                or str(row[21]).strip().lower() != selected_anchor_hash
                or _SHA256.fullmatch(vector_hash) is None
                or _SHA256.fullmatch(primary_option_hash) is None
                or _SHA256.fullmatch(scenario_option_hash) is None
                or side not in {"long", "short"}
            ):
                raise ValueError("Round 73 scenario source identity differs")
            required_complete_wall_ns = (
                anchor_wall_ns
                + scenario.entry_delay_ms * 1_000_000
                + scenario.horizon_ms * 1_000_000
                + 2 * ROUND73_TARGET_MAX_STATE_LATENESS_NS
            )
            outcome = classify_round73_operational_outcome(
                eligible=bool(row[11]),
                ineligible_reasons_json=str(row[12]),
                positive_net_payoff=None if row[13] is None else bool(row[13]),
                net_payoff_bps=None if row[14] is None else float(row[14]),
                deterministically_boundary_censored=(
                    required_complete_wall_ns >= coverage_end_wall_ns
                ),
            )
            reason_code = _REASON_TO_CODE.get(outcome.reason)
            if reason_code is None:
                raise ValueError("Round 73 scenario outcome reason is unknown")
            actual_entry_mono = -1 if row[16] is None else int(row[16])
            actual_exit_mono = -1 if row[17] is None else int(row[17])
            actual_entry_wall = (
                -1
                if actual_entry_mono < 0
                else anchor_wall_ns + actual_entry_mono - anchor_monotonic_ns
            )
            actual_exit_wall = (
                -1
                if actual_exit_mono < 0
                else anchor_wall_ns + actual_exit_mono - anchor_monotonic_ns
            )
            arrays["run_id_binary"][offset] = bytes.fromhex(run_id)
            arrays["anchor_index"][offset] = anchor_index
            arrays["anchor_monotonic_ns"][offset] = anchor_monotonic_ns
            arrays["anchor_wall_ns"][offset] = anchor_wall_ns
            arrays["utc_day_ordinal"][offset] = anchor_wall_ns // _DAY_NS
            arrays["side_sign"][offset] = 1 if side == "long" else -1
            arrays["outcome_status"][offset] = outcome.status
            arrays["outcome_reason_code"][offset] = reason_code
            arrays["binary_target"][offset] = outcome.binary_target
            arrays["continuous_target_bps"][offset] = outcome.continuous_target_bps
            arrays["net_payoff_quote"][offset] = (
                float(row[15]) if outcome.completed_transaction else float("nan")
            )
            arrays["actual_entry_wall_ns"][offset] = actual_entry_wall
            arrays["actual_exit_wall_ns"][offset] = actual_exit_wall
            arrays["entry_quote_notional"][offset] = (
                float(row[18]) if outcome.completed_transaction else float("nan")
            )
            arrays["exit_quote_notional"][offset] = (
                float(row[19]) if outcome.completed_transaction else float("nan")
            )
            arrays["maximum_adverse_excursion_bps"][offset] = (
                float(row[22]) if outcome.completed_transaction else float("nan")
            )
            arrays["maximum_favorable_excursion_bps"][offset] = (
                float(row[23]) if outcome.completed_transaction else float("nan")
            )
            arrays["maximum_spread_bps"][offset] = (
                float(row[24]) if outcome.completed_transaction else float("nan")
            )
            arrays["minimum_exit_side_capacity_ratio"][offset] = (
                float(row[25]) if outcome.completed_transaction else float("nan")
            )
            digest.update(bytes.fromhex(primary_option_hash))
            digest.update(bytes.fromhex(vector_hash))
            offset += 1
    if offset != row_count:
        raise ValueError("Round 73 scenario row count changed during read")
    result = _ScenarioRows(
        symbol=symbol,
        scenario=scenario,
        source_rows_sha256=digest.hexdigest(),
        **{name: _readonly(value) for name, value in arrays.items()},
    )
    result.validate()
    return result


def _decoded_predictions(
    payload: bytes,
    *,
    symbol: str,
    source_rows_sha256: str,
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    decoded = decode_round73_prediction_artifact(payload)
    header = decoded["header"]
    indexes = np.asarray(decoded["row_indexes"], dtype=np.int64)
    if (
        not isinstance(header, Mapping)
        or header.get("symbol") != symbol
        or header.get("role") != "test"
        or header.get("source_rows_sha256") != source_rows_sha256
        or int(header.get("row_count", -1)) != len(indexes)
        or not np.array_equal(indexes, np.arange(len(indexes), dtype=np.int64))
    ):
        raise ValueError("Round 73 test prediction row identity differs")
    predictions = decoded["predictions"]
    if not isinstance(predictions, Mapping):
        raise ValueError("Round 73 decoded predictions are invalid")
    return predictions


def _lower_tail_pinball_loss(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    quantile: float = 0.05,
) -> float:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    error = truth - forecast
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _safe_binary_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Mapping[str, object]:
    if len(target) == 0:
        return {"available": False, "reason": "zero_labeled_rows", "rows": 0}
    metrics = dict(binary_predictive_metrics(target, prediction))
    metrics.update(
        {
            "available": True,
            "single_class": len(np.unique(target)) != 2,
            "accuracy_at_0_5": metrics.pop("accuracy"),
            "Matthews_correlation_coefficient": metrics.pop("MCC"),
        }
    )
    return metrics


def _safe_continuous_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Mapping[str, object]:
    if len(target) == 0:
        return {"available": False, "reason": "zero_labeled_rows", "rows": 0}
    metrics = dict(continuous_predictive_metrics(target, prediction))
    metrics.update(
        {
            "available": True,
            "Spearman_rank_correlation": metrics.pop("Spearman"),
            "lower_tail_pinball_loss_tau_0_05": _lower_tail_pinball_loss(
                target,
                prediction,
            ),
        }
    )
    return metrics


def _fold_improvement_count(
    baseline: np.ndarray,
    challenger: np.ndarray,
    block_ids: np.ndarray,
) -> int:
    unique = np.unique(block_ids)
    if len(unique) < 4:
        return 0
    count = 0
    for fold in np.array_split(unique, 4):
        selected = np.isin(block_ids, fold)
        if np.any(selected) and float(np.mean(challenger[selected])) < float(
            np.mean(baseline[selected])
        ):
            count += 1
    return count


def _chronological_block_ids(run_id_binary: np.ndarray) -> np.ndarray:
    values = np.asarray(run_id_binary)
    if values.ndim != 1:
        raise ValueError("Round 73 run block IDs must be one-dimensional")
    if len(values) == 0:
        return _readonly(np.asarray([], dtype=np.int64))
    _unique, first_indexes, lexicographic_inverse = np.unique(
        values,
        return_index=True,
        return_inverse=True,
    )
    chronological_order = np.argsort(first_indexes, kind="stable")
    remap = np.empty(len(chronological_order), dtype=np.int64)
    remap[chronological_order] = np.arange(
        len(chronological_order),
        dtype=np.int64,
    )
    return _readonly(remap[lexicographic_inverse])


def _paired_block_bootstrap_interval(
    baseline_loss: np.ndarray,
    challenger_loss: np.ndarray,
    block_ids: np.ndarray,
    *,
    seed: int,
) -> Mapping[str, object]:
    unique, inverse = np.unique(block_ids, return_inverse=True)
    baseline_sum = np.bincount(
        inverse,
        weights=baseline_loss,
        minlength=len(unique),
    )
    challenger_sum = np.bincount(
        inverse,
        weights=challenger_loss,
        minlength=len(unique),
    )
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    generator = np.random.default_rng(seed)
    mean_difference = np.empty(
        ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
        dtype=np.float64,
    )
    relative_improvement = np.full_like(mean_difference, np.nan)
    completed = 0
    while completed < ROUND73_EVALUATION_BOOTSTRAP_DRAWS:
        batch = min(512, ROUND73_EVALUATION_BOOTSTRAP_DRAWS - completed)
        indexes = generator.integers(
            0,
            len(unique),
            size=(batch, len(unique)),
            endpoint=False,
        )
        sampled_count = np.sum(count[indexes], axis=1)
        baseline_mean = np.sum(baseline_sum[indexes], axis=1) / sampled_count
        challenger_mean = np.sum(challenger_sum[indexes], axis=1) / sampled_count
        mean_difference[completed : completed + batch] = challenger_mean - baseline_mean
        valid = baseline_mean > 0.0
        batch_improvement = np.full(batch, np.nan, dtype=np.float64)
        batch_improvement[valid] = (
            baseline_mean[valid] - challenger_mean[valid]
        ) / baseline_mean[valid]
        relative_improvement[completed : completed + batch] = batch_improvement
        completed += batch
    difference_interval = np.quantile(mean_difference, (0.025, 0.975))
    finite_improvement = relative_improvement[np.isfinite(relative_improvement)]
    improvement_interval = (
        np.quantile(finite_improvement, (0.025, 0.975))
        if len(finite_improvement)
        else None
    )
    return {
        "draws": ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
        "seed": seed,
        "block_count": len(unique),
        "mean_loss_difference_lower": float(difference_interval[0]),
        "mean_loss_difference_upper": float(difference_interval[1]),
        "relative_improvement_finite_draws": len(finite_improvement),
        "relative_improvement_lower": (
            float(improvement_interval[0]) if improvement_interval is not None else None
        ),
        "relative_improvement_upper": (
            float(improvement_interval[1]) if improvement_interval is not None else None
        ),
    }


def _paired_comparison(
    *,
    stage: str,
    baseline_name: str,
    challenger_name: str,
    target_kind: str,
    loss_name: str,
    baseline_loss: np.ndarray,
    challenger_loss: np.ndarray,
    block_ids: np.ndarray,
    seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {
        "stage": stage,
        "baseline": baseline_name,
        "challenger": challenger_name,
        "target_kind": target_kind,
        "loss": loss_name,
        "rows": len(baseline_loss),
        "blocks": len(np.unique(block_ids)),
        "positive_chronological_folds": _fold_improvement_count(
            baseline_loss,
            challenger_loss,
            block_ids,
        ),
        "permutation_draws": ROUND73_EVALUATION_PERMUTATION_DRAWS,
        "seed": seed,
        "q_value": None,
    }
    if (
        len(baseline_loss) < 30
        or len(np.unique(block_ids)) < 10
        or float(np.mean(baseline_loss)) <= 0.0
    ):
        output.update(
            {
                "available": False,
                "reason": "insufficient_rows_blocks_or_baseline_loss",
                "baseline_mean_loss": (
                    _finite_or_none(np.mean(baseline_loss))
                    if len(baseline_loss)
                    else None
                ),
                "challenger_mean_loss": (
                    _finite_or_none(np.mean(challenger_loss))
                    if len(challenger_loss)
                    else None
                ),
                "relative_improvement": None,
                "one_sided_p_value": None,
                "block_bootstrap_95_percent_interval": None,
            }
        )
        return output
    comparison = paired_blocked_permutation_test(
        baseline_loss,
        challenger_loss,
        block_ids,
        draws=ROUND73_EVALUATION_PERMUTATION_DRAWS,
        seed=seed,
    )
    output.update(asdict(comparison))
    output["block_bootstrap_95_percent_interval"] = _paired_block_bootstrap_interval(
        baseline_loss,
        challenger_loss,
        block_ids,
        seed=seed + 500_000,
    )
    output["available"] = True
    return output


def _predictive_evaluation(
    rows: _ScenarioRows,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    training_prevalence: float,
    seed_offset: int,
) -> tuple[Mapping[str, object], list[dict[str, object]]]:
    mask = rows.label_mask
    truth_binary = rows.binary_target[mask]
    truth_continuous = rows.continuous_target_bps[mask]
    prevalence = np.full(len(truth_binary), training_prevalence, dtype=np.float64)
    zero = np.zeros(len(truth_continuous), dtype=np.float64)
    all_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "prevalence_zero": (prevalence, zero)
    }
    for candidate in ROUND73_SHALLOW_CANDIDATES:
        binary, continuous = predictions[candidate]
        if binary.shape != (rows.rows,) or continuous.shape != (rows.rows,):
            raise ValueError("Round 73 prediction and scenario row counts differ")
        all_predictions[candidate] = (binary[mask], continuous[mask])
    metrics: dict[str, object] = {}
    loss_rows: dict[str, dict[str, np.ndarray]] = {}
    for name, (binary, continuous) in all_predictions.items():
        metrics[name] = {
            "binary": _safe_binary_metrics(truth_binary, binary),
            "continuous": _safe_continuous_metrics(truth_continuous, continuous),
        }
        if len(truth_binary):
            loss_rows[name] = {
                **binary_loss_rows(truth_binary, binary),
                **continuous_loss_rows(truth_continuous, continuous),
            }
        else:
            loss_rows[name] = {}
    block_ids = _chronological_block_ids(rows.run_id_binary[mask])
    comparisons: list[dict[str, object]] = []
    loss_contract = (
        ("binary", "log_loss"),
        ("binary", "brier_score"),
        ("continuous", "mean_squared_error"),
        ("continuous", "mean_absolute_error"),
    )
    for stage_index, (stage, baseline, challenger) in enumerate(_COMPARISON_STAGES):
        for loss_index, (target_kind, loss_name) in enumerate(loss_contract):
            if not len(truth_binary):
                baseline_values = np.asarray([], dtype=np.float64)
                challenger_values = np.asarray([], dtype=np.float64)
            else:
                baseline_values = loss_rows[baseline][loss_name]
                challenger_values = loss_rows[challenger][loss_name]
            comparisons.append(
                _paired_comparison(
                    stage=stage,
                    baseline_name=baseline,
                    challenger_name=challenger,
                    target_kind=target_kind,
                    loss_name=loss_name,
                    baseline_loss=baseline_values,
                    challenger_loss=challenger_values,
                    block_ids=block_ids,
                    seed=(
                        ROUND73_MODEL_SEED + seed_offset + stage_index * 10 + loss_index
                    ),
                )
            )
    status_counts = Counter(
        _SCENARIO_REASON_NAMES[int(code)] for code in rows.outcome_reason_code
    )
    report = {
        "rows": rows.rows,
        "labeled_rows": int(np.count_nonzero(mask)),
        "right_censored_rows": int(
            np.count_nonzero(rows.outcome_status == ROUND73_RIGHT_CENSORED_STATUS)
        ),
        "post_entry_unresolved_rows": int(
            np.count_nonzero(
                rows.outcome_status == ROUND73_POST_ENTRY_UNRESOLVED_STATUS
            )
        ),
        "outcome_reason_counts": dict(sorted(status_counts.items())),
        "models": metrics,
        "comparisons": comparisons,
    }
    return report, comparisons


@dataclass(frozen=True)
class _AttemptEvent:
    symbol: str
    run_id: bytes
    anchor_index: int
    anchor_wall_ns: int
    utc_day_ordinal: int
    side_sign: int
    status: int
    reason_code: int
    net_bps: float
    net_payoff_quote: float
    actual_entry_wall_ns: int
    actual_exit_wall_ns: int
    entry_quote_notional: float
    exit_quote_notional: float
    maximum_adverse_excursion_bps: float
    maximum_favorable_excursion_bps: float
    maximum_spread_bps: float
    minimum_exit_side_capacity_ratio: float

    @property
    def ordering_key(self) -> tuple[int, bytes, int, str]:
        return (
            self.anchor_wall_ns,
            self.run_id,
            self.anchor_index,
            self.symbol,
        )


@dataclass(frozen=True)
class _CompletedTrade:
    symbol: str
    run_id: bytes
    utc_day_ordinal: int
    side_sign: int
    net_bps: float
    net_payoff_quote: float
    entry_wall_ns: int
    exit_wall_ns: int
    turnover_quote: float
    maximum_adverse_excursion_bps: float
    maximum_favorable_excursion_bps: float
    maximum_spread_bps: float
    minimum_exit_side_capacity_ratio: float


@dataclass(frozen=True)
class _ExposureInterval:
    symbol: str
    start_wall_ns: int
    end_wall_ns: int
    unresolved: bool


def _select_signal_events(
    rows: _ScenarioRows,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    selected_candidate: str,
    probability_threshold: float,
    action_enabled: bool,
) -> tuple[list[_AttemptEvent], Mapping[str, object]]:
    if (
        selected_candidate not in ROUND73_SHALLOW_CANDIDATES
        or probability_threshold
        not in {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}
        or not isinstance(action_enabled, bool)
    ):
        raise ValueError("Round 73 frozen action policy is invalid")
    binary, continuous = predictions[selected_candidate]
    if binary.shape != (rows.rows,) or continuous.shape != (rows.rows,):
        raise ValueError("Round 73 action predictions differ from scenario rows")
    events: list[_AttemptEvent] = []
    censored_pairs = 0
    no_signal_pairs = 0
    exact_score_ties = 0
    both_sides_passed = 0
    pair_count = rows.rows // 2
    if not action_enabled:
        censored_pairs = int(
            np.count_nonzero(
                (rows.outcome_status[0::2] == ROUND73_RIGHT_CENSORED_STATUS)
                | (rows.outcome_status[1::2] == ROUND73_RIGHT_CENSORED_STATUS)
            )
        )
        return events, {
            "action_enabled": False,
            "anchor_pairs": pair_count,
            "selected_signals": 0,
            "right_censored_pairs": censored_pairs,
            "no_signal_pairs": pair_count - censored_pairs,
            "both_sides_passed": 0,
            "exact_side_score_ties": 0,
        }
    for pair_index in range(pair_count):
        long_row = pair_index * 2
        short_row = long_row + 1
        if (
            rows.outcome_status[long_row] == ROUND73_RIGHT_CENSORED_STATUS
            or rows.outcome_status[short_row] == ROUND73_RIGHT_CENSORED_STATUS
        ):
            censored_pairs += 1
            continue
        long_pass = bool(
            binary[long_row] >= probability_threshold and continuous[long_row] > 0.0
        )
        short_pass = bool(
            binary[short_row] >= probability_threshold and continuous[short_row] > 0.0
        )
        selected_row = -1
        if long_pass and not short_pass:
            selected_row = long_row
        elif short_pass and not long_pass:
            selected_row = short_row
        elif long_pass and short_pass:
            both_sides_passed += 1
            if continuous[long_row] > continuous[short_row]:
                selected_row = long_row
            elif continuous[short_row] > continuous[long_row]:
                selected_row = short_row
            else:
                exact_score_ties += 1
        if selected_row < 0:
            no_signal_pairs += 1
            continue
        status = int(rows.outcome_status[selected_row])
        if status == ROUND73_RIGHT_CENSORED_STATUS:
            raise ValueError("Round 73 selected signal crossed deterministic censoring")
        events.append(
            _AttemptEvent(
                symbol=rows.symbol,
                run_id=_fixed_binary_row(rows.run_id_binary, selected_row, 16),
                anchor_index=int(rows.anchor_index[selected_row]),
                anchor_wall_ns=int(rows.anchor_wall_ns[selected_row]),
                utc_day_ordinal=int(rows.utc_day_ordinal[selected_row]),
                side_sign=int(rows.side_sign[selected_row]),
                status=status,
                reason_code=int(rows.outcome_reason_code[selected_row]),
                net_bps=float(rows.continuous_target_bps[selected_row]),
                net_payoff_quote=float(rows.net_payoff_quote[selected_row]),
                actual_entry_wall_ns=int(rows.actual_entry_wall_ns[selected_row]),
                actual_exit_wall_ns=int(rows.actual_exit_wall_ns[selected_row]),
                entry_quote_notional=float(rows.entry_quote_notional[selected_row]),
                exit_quote_notional=float(rows.exit_quote_notional[selected_row]),
                maximum_adverse_excursion_bps=float(
                    rows.maximum_adverse_excursion_bps[selected_row]
                ),
                maximum_favorable_excursion_bps=float(
                    rows.maximum_favorable_excursion_bps[selected_row]
                ),
                maximum_spread_bps=float(rows.maximum_spread_bps[selected_row]),
                minimum_exit_side_capacity_ratio=float(
                    rows.minimum_exit_side_capacity_ratio[selected_row]
                ),
            )
        )
    return events, {
        "action_enabled": True,
        "anchor_pairs": pair_count,
        "selected_signals": len(events),
        "right_censored_pairs": censored_pairs,
        "no_signal_pairs": no_signal_pairs,
        "both_sides_passed": both_sides_passed,
        "exact_side_score_ties": exact_score_ties,
    }


def _blocked_trade_bootstrap(
    trades: Sequence[_CompletedTrade],
    *,
    seed: int,
) -> Mapping[str, object]:
    if not trades:
        return {
            "available": False,
            "reason": "zero_completed_trades",
            "blocks": 0,
            "draws": ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
            "expectancy_bps_lower_95": None,
            "profit_factor_lower_95": None,
            "profit_factor_lower_is_infinite": False,
        }
    run_ids = np.asarray([trade.run_id for trade in trades], dtype="S16")
    values = np.asarray([trade.net_bps for trade in trades], dtype=np.float64)
    pnl_quote = np.asarray(
        [trade.net_payoff_quote for trade in trades],
        dtype=np.float64,
    )
    unique, inverse = np.unique(run_ids, return_inverse=True)
    if len(unique) < 10:
        return {
            "available": False,
            "reason": "fewer_than_ten_integrity_segments",
            "blocks": len(unique),
            "draws": ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
            "expectancy_bps_lower_95": None,
            "profit_factor_lower_95": None,
            "profit_factor_lower_is_infinite": False,
        }
    count_by_block = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sum_by_block = np.bincount(
        inverse,
        weights=values,
        minlength=len(unique),
    )
    gain_by_block = np.bincount(
        inverse,
        weights=np.maximum(pnl_quote, 0.0),
        minlength=len(unique),
    )
    loss_by_block = np.bincount(
        inverse,
        weights=np.maximum(-pnl_quote, 0.0),
        minlength=len(unique),
    )
    generator = np.random.default_rng(seed)
    expectancy = np.empty(ROUND73_EVALUATION_BOOTSTRAP_DRAWS, dtype=np.float64)
    profit_factor = np.empty(
        ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
        dtype=np.float64,
    )
    completed = 0
    while completed < ROUND73_EVALUATION_BOOTSTRAP_DRAWS:
        batch = min(512, ROUND73_EVALUATION_BOOTSTRAP_DRAWS - completed)
        indexes = generator.integers(
            0,
            len(unique),
            size=(batch, len(unique)),
            endpoint=False,
        )
        sampled_count = np.sum(count_by_block[indexes], axis=1)
        sampled_sum = np.sum(sum_by_block[indexes], axis=1)
        sampled_gain = np.sum(gain_by_block[indexes], axis=1)
        sampled_loss = np.sum(loss_by_block[indexes], axis=1)
        expectancy[completed : completed + batch] = sampled_sum / sampled_count
        sampled_factor = np.divide(
            sampled_gain,
            sampled_loss,
            out=np.full(batch, np.inf, dtype=np.float64),
            where=sampled_loss > 0.0,
        )
        sampled_factor[(sampled_loss == 0.0) & (sampled_gain == 0.0)] = 0.0
        profit_factor[completed : completed + batch] = sampled_factor
        completed += batch
    lower_expectancy = float(np.quantile(expectancy, 0.05))
    lower_factor = float(np.quantile(profit_factor, 0.05))
    return {
        "available": True,
        "blocks": len(unique),
        "draws": ROUND73_EVALUATION_BOOTSTRAP_DRAWS,
        "seed": seed,
        "expectancy_bps_lower_95": lower_expectancy,
        "profit_factor_lower_95": (
            lower_factor if math.isfinite(lower_factor) else None
        ),
        "profit_factor_lower_is_infinite": math.isinf(lower_factor),
    }


def _drawdown_and_underwater(
    trades: Sequence[_CompletedTrade],
    *,
    study_start_wall_ns: int,
    study_end_wall_ns: int,
    capital_capacity_quote: float,
) -> Mapping[str, object]:
    pnl_by_exit: Counter[int] = Counter()
    for trade in trades:
        pnl_by_exit[trade.exit_wall_ns] += trade.net_payoff_quote
    equity = 0.0
    peak = 0.0
    maximum_drawdown_quote = 0.0
    underwater_ns = 0
    previous_wall_ns = study_start_wall_ns
    underwater = False
    for raw_exit_wall_ns, realized_pnl in sorted(pnl_by_exit.items()):
        exit_wall_ns = min(
            study_end_wall_ns,
            max(study_start_wall_ns, raw_exit_wall_ns),
        )
        if underwater and exit_wall_ns > previous_wall_ns:
            underwater_ns += exit_wall_ns - previous_wall_ns
        equity += realized_pnl
        peak = max(peak, equity)
        maximum_drawdown_quote = max(maximum_drawdown_quote, peak - equity)
        underwater = equity < peak - 1e-12
        previous_wall_ns = max(previous_wall_ns, exit_wall_ns)
    if underwater and study_end_wall_ns > previous_wall_ns:
        underwater_ns += study_end_wall_ns - previous_wall_ns
    duration_ns = study_end_wall_ns - study_start_wall_ns
    return {
        "maximum_drawdown_quote": maximum_drawdown_quote,
        "maximum_drawdown_fraction_of_capital_capacity": (
            maximum_drawdown_quote / capital_capacity_quote
        ),
        "time_under_water_seconds": underwater_ns / 1_000_000_000.0,
        "time_under_water_fraction": underwater_ns / duration_ns,
    }


def _exposure_metrics(
    intervals: Sequence[_ExposureInterval],
    *,
    study_start_wall_ns: int,
    study_end_wall_ns: int,
) -> Mapping[str, object]:
    duration_ns = study_end_wall_ns - study_start_wall_ns
    events: list[tuple[int, int]] = []
    summed_position_ns = 0
    for interval in intervals:
        start = max(study_start_wall_ns, interval.start_wall_ns)
        end = min(study_end_wall_ns, interval.end_wall_ns)
        if end <= start:
            continue
        summed_position_ns += end - start
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    maximum_active = 0
    union_ns = 0
    previous = study_start_wall_ns
    for wall_ns, delta in events:
        if active > 0 and wall_ns > previous:
            union_ns += wall_ns - previous
        active += delta
        if active < 0:
            raise ValueError("Round 73 exposure interval sweep became negative")
        maximum_active = max(maximum_active, active)
        previous = wall_ns
    return {
        "summed_position_exposure_seconds": (summed_position_ns / 1_000_000_000.0),
        "time_weighted_average_concurrent_positions": (
            summed_position_ns / duration_ns
        ),
        "any_position_exposure_fraction": union_ns / duration_ns,
        "maximum_concurrent_positions": maximum_active,
    }


def _concentration_metrics(
    trades: Sequence[_CompletedTrade],
) -> Mapping[str, object]:
    if not trades:
        return {
            "maximum_symbol_turnover_fraction": None,
            "maximum_day_trade_fraction": None,
            "maximum_single_trade_absolute_pnl_fraction": None,
            "turnover_herfindahl": None,
        }
    turnover_by_symbol: Counter[str] = Counter()
    trades_by_day: Counter[int] = Counter()
    for trade in trades:
        turnover_by_symbol[trade.symbol] += trade.turnover_quote
        trades_by_day[trade.utc_day_ordinal] += 1
    total_turnover = sum(turnover_by_symbol.values())
    absolute_pnl = np.asarray(
        [abs(trade.net_payoff_quote) for trade in trades],
        dtype=np.float64,
    )
    total_absolute_pnl = float(np.sum(absolute_pnl))
    turnover_shares = np.asarray(
        [value / total_turnover for value in turnover_by_symbol.values()],
        dtype=np.float64,
    )
    return {
        "maximum_symbol_turnover_fraction": float(np.max(turnover_shares)),
        "maximum_day_trade_fraction": max(trades_by_day.values()) / len(trades),
        "maximum_single_trade_absolute_pnl_fraction": (
            float(np.max(absolute_pnl) / total_absolute_pnl)
            if total_absolute_pnl > 0.0
            else None
        ),
        "turnover_herfindahl": float(np.sum(np.square(turnover_shares))),
    }


def _economic_metrics(
    trades: Sequence[_CompletedTrade],
    intervals: Sequence[_ExposureInterval],
    counters: Mapping[str, object],
    *,
    scenario: Round73EvaluationScenario,
    study_start_wall_ns: int,
    study_end_wall_ns: int,
    capital_position_capacity: int,
    seed: int,
) -> Mapping[str, object]:
    completed = len(trades)
    values = np.asarray([trade.net_bps for trade in trades], dtype=np.float64)
    pnl_quote = np.asarray(
        [trade.net_payoff_quote for trade in trades],
        dtype=np.float64,
    )
    positive = values[values > 0.0]
    negative = values[values < 0.0]
    quote_gains = float(np.sum(pnl_quote[pnl_quote > 0.0]))
    quote_losses = float(-np.sum(pnl_quote[pnl_quote < 0.0]))
    point_profit_factor = quote_gains / quote_losses if quote_losses > 0.0 else None
    point_profit_factor_infinite = quote_losses == 0.0 and quote_gains > 0.0
    capital_capacity_quote = float(
        scenario.reference_quote_notional * capital_position_capacity
    )
    total_pnl_quote = float(np.sum(pnl_quote)) if completed else 0.0
    total_turnover_quote = float(sum(trade.turnover_quote for trade in trades))
    bootstrap = _blocked_trade_bootstrap(trades, seed=seed)
    deployed = int(counters.get("deployed_positions", 0))
    unresolved = int(counters.get("post_entry_unresolved", 0))
    complete_fraction = completed / deployed if deployed else None
    selected_days = set(counters.get("selected_signal_days", ()))
    completed_by_day: Counter[int] = Counter(trade.utc_day_ordinal for trade in trades)
    daily_activity_passed = bool(selected_days) and all(
        completed_by_day[day] >= ROUND73_EVALUATION_MINIMUM_DAILY_TRADES
        for day in selected_days
    )
    lower_expectancy = bootstrap.get("expectancy_bps_lower_95")
    lower_factor = bootstrap.get("profit_factor_lower_95")
    lower_factor_passed = bool(
        bootstrap.get("profit_factor_lower_is_infinite")
        or (isinstance(lower_factor, (int, float)) and float(lower_factor) > 1.0)
    )
    operational_passed = bool(
        deployed > 0 and unresolved == 0 and complete_fraction == 1.0
    )
    economic_passed = bool(
        operational_passed
        and completed >= ROUND73_EVALUATION_MINIMUM_COMPLETED_TRADES
        and daily_activity_passed
        and isinstance(lower_expectancy, (int, float))
        and float(lower_expectancy) > 0.0
        and lower_factor_passed
    )
    if completed:
        quantile = float(np.quantile(values, 0.05))
        expected_shortfall = float(np.mean(values[values <= quantile]))
    else:
        expected_shortfall = None
    report: dict[str, object] = {
        "selected_signals": int(counters.get("selected_signals", 0)),
        "attempted_actions": int(counters.get("attempted_actions", 0)),
        "same_symbol_open_skips": int(counters.get("same_symbol_open_skips", 0)),
        "concurrent_capacity_skips": int(counters.get("concurrent_capacity_skips", 0)),
        "pre_entry_aborts": int(counters.get("pre_entry_aborts", 0)),
        "pre_entry_abort_reasons": dict(
            sorted(dict(counters.get("pre_entry_abort_reasons", {})).items())
        ),
        "deployed_positions": deployed,
        "post_entry_unresolved_risk_count": unresolved,
        "post_entry_unresolved_reasons": dict(
            sorted(dict(counters.get("post_entry_unresolved_reasons", {})).items())
        ),
        "completed_trades": completed,
        "complete_transaction_fraction": complete_fraction,
        "return_and_risk_metrics_cover_every_deployed_position": (operational_passed),
        "net_expectancy_bps": float(np.mean(values)) if completed else None,
        "profit_factor": point_profit_factor,
        "profit_factor_is_infinite": point_profit_factor_infinite,
        "win_rate": float(np.mean(values > 0.0)) if completed else None,
        "mean_win_bps": float(np.mean(positive)) if len(positive) else None,
        "mean_loss_bps": float(np.mean(negative)) if len(negative) else None,
        "expected_shortfall_95_bps": expected_shortfall,
        "maximum_single_position_adverse_excursion_bps": (
            min(trade.maximum_adverse_excursion_bps for trade in trades)
            if completed
            else None
        ),
        "maximum_single_position_favorable_excursion_bps": (
            max(trade.maximum_favorable_excursion_bps for trade in trades)
            if completed
            else None
        ),
        "maximum_observed_spread_bps_during_positions": (
            max(trade.maximum_spread_bps for trade in trades) if completed else None
        ),
        "minimum_observed_exit_side_capacity_ratio": (
            min(trade.minimum_exit_side_capacity_ratio for trade in trades)
            if completed
            else None
        ),
        "net_pnl_quote": total_pnl_quote,
        "capital_capacity_quote": capital_capacity_quote,
        "annualization_free_test_period_return": (
            total_pnl_quote / capital_capacity_quote
        ),
        "turnover_quote": total_turnover_quote,
        "turnover_multiple_of_capital_capacity": (
            total_turnover_quote / capital_capacity_quote
        ),
        "completed_trades_by_utc_day": {
            str(day): completed_by_day[day] for day in sorted(completed_by_day)
        },
        "eligible_condition_days": [str(day) for day in sorted(selected_days)],
        "minimum_daily_trade_gate_passed": daily_activity_passed,
        "bootstrap": bootstrap,
        **_drawdown_and_underwater(
            trades,
            study_start_wall_ns=study_start_wall_ns,
            study_end_wall_ns=study_end_wall_ns,
            capital_capacity_quote=capital_capacity_quote,
        ),
        **_exposure_metrics(
            intervals,
            study_start_wall_ns=study_start_wall_ns,
            study_end_wall_ns=study_end_wall_ns,
        ),
        **_concentration_metrics(trades),
        "maximum_drawdown_measurement": (
            "realized exit PnL aggregated at identical exit timestamps"
        ),
        "intratrade_portfolio_maximum_drawdown_reported": False,
        "intratrade_drawdown_limitation": (
            "real adverse excursions are reported per position, but their exact "
            "timestamps were not frozen and no portfolio path is invented"
        ),
        "operational_gate_passed": operational_passed,
        "economic_gate_passed": economic_passed,
        "annualized_roi_reported": False,
        "sharpe_reported": False,
        "leverage": 1.0,
        "profit_reinvestment": False,
    }
    return report


def _simulate_portfolio(
    events: Sequence[_AttemptEvent],
    *,
    scenario: Round73EvaluationScenario,
    study_start_wall_ns: int,
    study_end_wall_ns: int,
    seed: int,
) -> Mapping[str, object]:
    ordered = sorted(events, key=lambda event: event.ordering_key)
    active: dict[str, int | None] = {}
    trades: list[_CompletedTrade] = []
    intervals: list[_ExposureInterval] = []
    counters: dict[str, object] = {
        "selected_signals": len(ordered),
        "attempted_actions": 0,
        "same_symbol_open_skips": 0,
        "concurrent_capacity_skips": 0,
        "pre_entry_aborts": 0,
        "pre_entry_abort_reasons": Counter(),
        "deployed_positions": 0,
        "post_entry_unresolved": 0,
        "post_entry_unresolved_reasons": Counter(),
        "selected_signal_days": {event.utc_day_ordinal for event in ordered},
    }
    per_symbol_counters: dict[str, dict[str, object]] = {
        symbol: {
            "selected_signals": 0,
            "attempted_actions": 0,
            "same_symbol_open_skips": 0,
            "concurrent_capacity_skips": 0,
            "pre_entry_aborts": 0,
            "pre_entry_abort_reasons": Counter(),
            "deployed_positions": 0,
            "post_entry_unresolved": 0,
            "post_entry_unresolved_reasons": Counter(),
            "selected_signal_days": set(),
        }
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    for event in ordered:
        symbol_counter = per_symbol_counters[event.symbol]
        symbol_counter["selected_signals"] = int(symbol_counter["selected_signals"]) + 1
        symbol_counter["selected_signal_days"].add(event.utc_day_ordinal)
        active = {
            symbol: end
            for symbol, end in active.items()
            if end is None or end > event.anchor_wall_ns
        }
        if event.symbol in active:
            counters["same_symbol_open_skips"] = (
                int(counters["same_symbol_open_skips"]) + 1
            )
            symbol_counter["same_symbol_open_skips"] = (
                int(symbol_counter["same_symbol_open_skips"]) + 1
            )
            continue
        if len(active) >= ROUND73_EVALUATION_MAXIMUM_CONCURRENT_POSITIONS:
            counters["concurrent_capacity_skips"] = (
                int(counters["concurrent_capacity_skips"]) + 1
            )
            symbol_counter["concurrent_capacity_skips"] = (
                int(symbol_counter["concurrent_capacity_skips"]) + 1
            )
            continue
        counters["attempted_actions"] = int(counters["attempted_actions"]) + 1
        symbol_counter["attempted_actions"] = (
            int(symbol_counter["attempted_actions"]) + 1
        )
        reason = _SCENARIO_REASON_NAMES[event.reason_code]
        if event.status == ROUND73_PRE_ENTRY_ABORT_STATUS:
            counters["pre_entry_aborts"] = int(counters["pre_entry_aborts"]) + 1
            counters["pre_entry_abort_reasons"][reason] += 1
            symbol_counter["pre_entry_aborts"] = (
                int(symbol_counter["pre_entry_aborts"]) + 1
            )
            symbol_counter["pre_entry_abort_reasons"][reason] += 1
            continue
        if event.status == ROUND73_POST_ENTRY_UNRESOLVED_STATUS:
            counters["deployed_positions"] = int(counters["deployed_positions"]) + 1
            counters["post_entry_unresolved"] = (
                int(counters["post_entry_unresolved"]) + 1
            )
            counters["post_entry_unresolved_reasons"][reason] += 1
            symbol_counter["deployed_positions"] = (
                int(symbol_counter["deployed_positions"]) + 1
            )
            symbol_counter["post_entry_unresolved"] = (
                int(symbol_counter["post_entry_unresolved"]) + 1
            )
            symbol_counter["post_entry_unresolved_reasons"][reason] += 1
            start = (
                event.actual_entry_wall_ns
                if event.actual_entry_wall_ns >= event.anchor_wall_ns
                else event.anchor_wall_ns + scenario.entry_delay_ms * 1_000_000
            )
            active[event.symbol] = None
            intervals.append(
                _ExposureInterval(
                    symbol=event.symbol,
                    start_wall_ns=start,
                    end_wall_ns=study_end_wall_ns,
                    unresolved=True,
                )
            )
            continue
        if event.status != ROUND73_OBSERVED_STATUS:
            raise ValueError("Round 73 portfolio encountered an unknown status")
        if (
            not math.isfinite(event.net_bps)
            or not math.isfinite(event.net_payoff_quote)
            or event.actual_entry_wall_ns < event.anchor_wall_ns
            or event.actual_exit_wall_ns <= event.actual_entry_wall_ns
            or not math.isfinite(event.entry_quote_notional)
            or not math.isfinite(event.exit_quote_notional)
            or not math.isfinite(event.maximum_adverse_excursion_bps)
            or not math.isfinite(event.maximum_favorable_excursion_bps)
            or not math.isfinite(event.maximum_spread_bps)
            or not math.isfinite(event.minimum_exit_side_capacity_ratio)
            or event.maximum_adverse_excursion_bps > event.net_bps
            or event.net_bps > event.maximum_favorable_excursion_bps
            or event.maximum_spread_bps < 0.0
            or event.minimum_exit_side_capacity_ratio < 1.0
        ):
            raise ValueError("Round 73 complete trade mechanics are invalid")
        counters["deployed_positions"] = int(counters["deployed_positions"]) + 1
        symbol_counter["deployed_positions"] = (
            int(symbol_counter["deployed_positions"]) + 1
        )
        active[event.symbol] = event.actual_exit_wall_ns
        trade = _CompletedTrade(
            symbol=event.symbol,
            run_id=event.run_id,
            utc_day_ordinal=event.utc_day_ordinal,
            side_sign=event.side_sign,
            net_bps=event.net_bps,
            net_payoff_quote=event.net_payoff_quote,
            entry_wall_ns=event.actual_entry_wall_ns,
            exit_wall_ns=event.actual_exit_wall_ns,
            turnover_quote=(event.entry_quote_notional + event.exit_quote_notional),
            maximum_adverse_excursion_bps=(event.maximum_adverse_excursion_bps),
            maximum_favorable_excursion_bps=(event.maximum_favorable_excursion_bps),
            maximum_spread_bps=event.maximum_spread_bps,
            minimum_exit_side_capacity_ratio=(event.minimum_exit_side_capacity_ratio),
        )
        trades.append(trade)
        intervals.append(
            _ExposureInterval(
                symbol=event.symbol,
                start_wall_ns=event.actual_entry_wall_ns,
                end_wall_ns=event.actual_exit_wall_ns,
                unresolved=False,
            )
        )
    combined = _economic_metrics(
        trades,
        intervals,
        counters,
        scenario=scenario,
        study_start_wall_ns=study_start_wall_ns,
        study_end_wall_ns=study_end_wall_ns,
        capital_position_capacity=ROUND73_EVALUATION_MAXIMUM_CONCURRENT_POSITIONS,
        seed=seed,
    )
    per_symbol: dict[str, object] = {}
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        symbol_trades = [trade for trade in trades if trade.symbol == symbol]
        symbol_intervals = [
            interval for interval in intervals if interval.symbol == symbol
        ]
        per_symbol[symbol] = _economic_metrics(
            symbol_trades,
            symbol_intervals,
            per_symbol_counters[symbol],
            scenario=scenario,
            study_start_wall_ns=study_start_wall_ns,
            study_end_wall_ns=study_end_wall_ns,
            capital_position_capacity=1,
            seed=seed + IMPACT_CAPTURE_SYMBOLS.index(symbol) + 1,
        )
    return {
        "combined": combined,
        "by_symbol": per_symbol,
        "transaction_order": ("anchor_wall_ns,run_id,anchor_index,symbol"),
        "maximum_concurrent_positions": (
            ROUND73_EVALUATION_MAXIMUM_CONCURRENT_POSITIONS
        ),
    }


def _apply_multiple_testing(
    comparisons: Sequence[dict[str, object]],
) -> Mapping[str, object]:
    available = [
        comparison
        for comparison in comparisons
        if comparison.get("available") is True
        and isinstance(comparison.get("one_sided_p_value"), (int, float))
    ]
    if available:
        q_values = benjamini_hochberg_q_values(
            [float(comparison["one_sided_p_value"]) for comparison in available]
        )
        for comparison, q_value in zip(available, q_values, strict=True):
            comparison["q_value"] = float(q_value)
    return {
        "method": "Benjamini-Hochberg false discovery rate",
        "family": (
            "all available symbol, scenario, staged-comparison, target, "
            "and primary-loss hypotheses"
        ),
        "total_comparisons": len(comparisons),
        "adjusted_comparisons": len(available),
        "unavailable_comparisons": len(comparisons) - len(available),
        "maximum_q_value": ROUND73_EVALUATION_MAXIMUM_Q_VALUE,
    }


def _comparison_by_stage_and_loss(
    report: Mapping[str, object],
    *,
    stage: str,
    loss: str,
) -> Mapping[str, object] | None:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Sequence):
        return None
    matching = [
        item
        for item in comparisons
        if isinstance(item, Mapping)
        and item.get("stage") == stage
        and item.get("loss") == loss
    ]
    if len(matching) != 1:
        return None
    return matching[0]


def _bootstrap_relative_improvement_lower(
    comparison: Mapping[str, object],
) -> float | None:
    interval = comparison.get("block_bootstrap_95_percent_interval")
    if not isinstance(interval, Mapping):
        return None
    value = interval.get("relative_improvement_lower")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _required_stages(selected_candidate: str) -> tuple[str, ...]:
    if selected_candidate == "linear_l1_tape":
        return ("linear_vs_controls",)
    if selected_candidate == "l1_tape":
        return ("linear_vs_controls", "l1_tape_vs_linear")
    if selected_candidate == "l2_state":
        return (
            "linear_vs_controls",
            "l1_tape_vs_linear",
            "l2_state_vs_l1_tape",
            "l2_state_vs_linear",
        )
    if selected_candidate == "impact_absorption":
        return (
            "linear_vs_controls",
            "l1_tape_vs_linear",
            "l2_state_vs_l1_tape",
            "impact_absorption_vs_l2_state",
            "impact_absorption_vs_linear",
        )
    raise ValueError("Round 73 selected candidate is unknown")


def _symbol_predictive_gate(
    primary_report: Mapping[str, object],
    delay_stress_report: Mapping[str, object],
    *,
    selected_candidate: str,
) -> Mapping[str, object]:
    reasons: list[str] = []
    required = _required_stages(selected_candidate)
    models = primary_report.get("models")
    selected_metrics = (
        models.get(selected_candidate) if isinstance(models, Mapping) else None
    )
    binary_metrics = (
        selected_metrics.get("binary")
        if isinstance(selected_metrics, Mapping)
        else None
    )
    if (
        not isinstance(binary_metrics, Mapping)
        or binary_metrics.get("available") is not True
        or binary_metrics.get("single_class") is True
    ):
        reasons.append(
            "selected_model_binary_test_target_is_unavailable_or_single_class"
        )
    loss_thresholds = {
        "log_loss": ROUND73_EVALUATION_MINIMUM_RELATIVE_PROPER_SCORE_IMPROVEMENT,
        "brier_score": (ROUND73_EVALUATION_MINIMUM_RELATIVE_PROPER_SCORE_IMPROVEMENT),
        "mean_squared_error": ROUND73_EVALUATION_MINIMUM_CONTINUOUS_MSE_SKILL,
        "mean_absolute_error": 0.0,
    }
    evidence: list[dict[str, object]] = []
    for stage in required:
        for loss, minimum_improvement in loss_thresholds.items():
            comparison = _comparison_by_stage_and_loss(
                primary_report,
                stage=stage,
                loss=loss,
            )
            bootstrap_lower = (
                _bootstrap_relative_improvement_lower(comparison)
                if isinstance(comparison, Mapping)
                else None
            )
            passed = bool(
                isinstance(comparison, Mapping)
                and comparison.get("available") is True
                and isinstance(comparison.get("q_value"), (int, float))
                and float(comparison["q_value"]) <= ROUND73_EVALUATION_MAXIMUM_Q_VALUE
                and isinstance(
                    comparison.get("relative_improvement"),
                    (int, float),
                )
                and float(comparison["relative_improvement"]) > minimum_improvement
                and bootstrap_lower is not None
                and bootstrap_lower > minimum_improvement
                and int(comparison.get("positive_chronological_folds", 0)) >= 4
            )
            evidence.append(
                {
                    "scenario": _PRIMARY_SCENARIO_KEY,
                    "stage": stage,
                    "loss": loss,
                    "minimum_relative_improvement": minimum_improvement,
                    "block_bootstrap_relative_improvement_lower": bootstrap_lower,
                    "block_bootstrap_lower_bound_is_a_gate": True,
                    "passed": passed,
                }
            )
            if not passed:
                reasons.append(f"primary:{stage}:{loss}:gate_failed")
            stress = _comparison_by_stage_and_loss(
                delay_stress_report,
                stage=stage,
                loss=loss,
            )
            stress_bootstrap_lower = (
                _bootstrap_relative_improvement_lower(stress)
                if isinstance(stress, Mapping)
                else None
            )
            stress_passed = bool(
                isinstance(stress, Mapping)
                and stress.get("available") is True
                and isinstance(stress.get("relative_improvement"), (int, float))
                and float(stress["relative_improvement"]) > 0.0
                and stress_bootstrap_lower is not None
                and stress_bootstrap_lower > 0.0
            )
            evidence.append(
                {
                    "scenario": _DELAY_STRESS_SCENARIO_KEY,
                    "stage": stage,
                    "loss": loss,
                    "minimum_relative_improvement": 0.0,
                    "block_bootstrap_relative_improvement_lower": (
                        stress_bootstrap_lower
                    ),
                    "block_bootstrap_lower_bound_is_a_gate": True,
                    "q_value_is_not_a_delay_stress_gate": True,
                    "passed": stress_passed,
                }
            )
            if not stress_passed:
                reasons.append(f"delay_stress:{stage}:{loss}:skill_not_positive")
    return {
        "selected_candidate": selected_candidate,
        "required_stages": list(required),
        "passed": not reasons,
        "reasons": reasons,
        "evidence": evidence,
        "accuracy_only_can_pass": False,
        "positive_pnl_only_can_pass": False,
    }


def _independent_symbol_viability_gates(
    *,
    predictive_gates: Mapping[str, object],
    primary_economic_by_symbol: Mapping[str, object],
    enabled_symbols: Sequence[str],
) -> tuple[Mapping[str, object], int]:
    enabled = set(enabled_symbols)
    output: dict[str, object] = {}
    passed_count = 0
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        predictive = predictive_gates.get(symbol)
        economic = primary_economic_by_symbol.get(symbol)
        if symbol not in enabled:
            output[symbol] = {
                "passed": False,
                "reason": "symbol_disabled_before_test",
                "predictive_gate_passed": False,
                "operational_gate_passed": False,
                "economic_gate_passed": False,
            }
            continue
        if not isinstance(predictive, Mapping) or not isinstance(economic, Mapping):
            raise ValueError("Round 73 independent symbol gate evidence is missing")
        predictive_passed = predictive.get("passed") is True
        operational_passed = economic.get("operational_gate_passed") is True
        economic_passed = economic.get("economic_gate_passed") is True
        passed = bool(predictive_passed and operational_passed and economic_passed)
        output[symbol] = {
            "passed": passed,
            "predictive_gate_passed": predictive_passed,
            "operational_gate_passed": operational_passed,
            "economic_gate_passed": economic_passed,
        }
        passed_count += int(passed)
    return output, passed_count


def _claimed_test_seal_function(
    claim: Round73EvaluationAccessClaim,
) -> Callable[..., object]:
    def sealed_test(*_args: object, **kwargs: object) -> object:
        if (
            str(kwargs.get("study_id", "")).strip().lower() != claim.study_id
            or str(kwargs.get("pretest_manifest_sha256", "")).strip().lower()
            != claim.pretest_manifest_sha256
        ):
            raise ValueError("Round 73 claimed test seal request differs")
        return SimpleNamespace(
            test_study_manifest_sha256=claim.test_study_manifest_sha256
        )

    return sealed_test


def _reaudit_claimed_test_targets(
    database: str | Path,
    *,
    claim: Round73EvaluationAccessClaim,
    memory_limit: str,
    threads: int,
    progress_callback: ProgressCallback | None,
) -> int:
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        seal = connection.execute(
            f"SELECT source_run_count, role_run_manifests_sha256, "
            f"manifest_sha256 FROM {ROUND73_TARGET_V3_TEST_STUDY_TABLE} "
            "WHERE study_id = ?",
            [claim.study_id],
        ).fetchone()
        rows = connection.execute(
            f"SELECT run_id, target_manifest_sha256 FROM "
            f"{ROUND73_TARGET_V3_ROLE_RUN_TABLE} "
            "WHERE study_id = ? AND role_scope = 'test' ORDER BY run_id",
            [claim.study_id],
        ).fetchall()
    if seal is None:
        raise ValueError("Round 73 claimed test study seal is missing")
    expected = tuple((str(row[0]), str(row[1])) for row in rows)
    if (
        int(seal[0]) != len(expected)
        or len(expected) == 0
        or str(seal[2]) != claim.test_study_manifest_sha256
        or any(
            _RUN_ID.fullmatch(run_id) is None
            or _SHA256.fullmatch(manifest_hash) is None
            for run_id, manifest_hash in expected
        )
        or _stream_hash([manifest_hash for _, manifest_hash in expected])
        != str(seal[1])
    ):
        raise ValueError("Round 73 claimed test role-manifest aggregate differs")
    cohort = audit_round73_shock_cohort(
        database,
        study_id=claim.study_id,
        deep_source_audit=False,
        memory_limit=memory_limit,
        threads=threads,
    )
    if getattr(cohort, "passed", False) is not True:
        raise ValueError("Round 73 claimed test cohort re-audit failed")

    def reuse_cohort_audit(*_args: object, **_kwargs: object) -> object:
        return cohort

    for index, (run_id, manifest_hash) in enumerate(expected, start=1):
        _report_progress(
            progress_callback,
            "test_exact_wire_reaudit_started",
            {
                "study_id": claim.study_id,
                "run_id": run_id,
                "run_index": index,
                "source_run_count": len(expected),
            },
        )
        audit = audit_round73_role_targets(
            database,
            study_id=claim.study_id,
            run_id=run_id,
            role_scope="test",
            pretest_manifest_sha256=claim.pretest_manifest_sha256,
            memory_limit=memory_limit,
            threads=threads,
            cohort_audit_function=reuse_cohort_audit,
            deep_replay=True,
        )
        if (
            audit.passed is not True
            or audit.deep_replay_performed is not True
            or audit.target_manifest_sha256 != manifest_hash
        ):
            raise ValueError(
                "Round 73 claimed test exact-wire re-audit failed: " + run_id
            )
        _report_progress(
            progress_callback,
            "test_exact_wire_reaudit_completed",
            {
                "study_id": claim.study_id,
                "run_id": run_id,
                "run_index": index,
                "source_run_count": len(expected),
            },
        )
    return len(expected)


def _score_and_persist_test_predictions(
    database: str | Path,
    *,
    claim: Round73EvaluationAccessClaim,
    pretest_identity: Mapping[str, object],
    memory_budget_bytes: int,
    memory_limit: str,
    threads: int,
    progress_callback: ProgressCallback | None,
) -> tuple[
    Mapping[str, Mapping[str, object]],
    Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
]:
    symbol_models = pretest_identity.get("symbol_models")
    if not isinstance(symbol_models, Mapping):
        raise ValueError("Round 73 claimed symbol model manifest is missing")
    enabled = tuple(
        symbol
        for symbol in IMPACT_CAPTURE_SYMBOLS
        if isinstance(symbol_models.get(symbol), Mapping)
        and symbol_models[symbol].get("status") == "enabled"
    )
    if not enabled:
        raise ValueError("Round 73 claimed pretest has no enabled symbols")
    model_reports: dict[str, Mapping[str, object]] = {}
    prediction_by_symbol: dict[
        str,
        Mapping[str, tuple[np.ndarray, np.ndarray]],
    ] = {}
    datasets = iter_round73_staged_symbol_slices(
        database,
        study_id=claim.study_id,
        role_scope="test",
        pretest_manifest_sha256=claim.pretest_manifest_sha256,
        symbols=enabled,
        memory_budget_bytes=memory_budget_bytes,
        memory_limit=memory_limit,
        threads=threads,
        test_seal_function=_claimed_test_seal_function(claim),
    )
    for dataset in datasets:
        _report_progress(
            progress_callback,
            "test_symbol_scoring_started",
            {
                "study_id": claim.study_id,
                "symbol": dataset.symbol,
                "rows": dataset.rows,
            },
        )
        manifest = symbol_models[dataset.symbol]
        if not isinstance(manifest, Mapping):
            raise ValueError("Round 73 enabled symbol manifest is invalid")
        artifact_names = manifest.get("artifact_names")
        if not isinstance(artifact_names, Mapping):
            raise ValueError("Round 73 enabled symbol artifact names are missing")
        artifacts = load_round73_claimed_symbol_artifacts(
            database,
            study_id=claim.study_id,
            symbol=dataset.symbol,
            artifact_names=artifact_names,
            memory_limit=memory_limit,
            threads=threads,
        )
        predictions, model = predict_round73_frozen_symbol_model(
            dataset,
            model_payload=artifacts["model"],
            preprocessor_payload=artifacts["preprocessor"],
        )
        selected_candidate = str(model.get("selected_candidate", ""))
        selected_threshold = model.get("selected_probability_threshold")
        action_enabled = model.get("action_enabled")
        training_prevalence = model.get("training_prevalence")
        if (
            selected_candidate not in ROUND73_SHALLOW_CANDIDATES
            or not isinstance(selected_threshold, (int, float))
            or float(selected_threshold) != float(manifest["probability_threshold"])
            or not isinstance(action_enabled, bool)
            or not isinstance(training_prevalence, (int, float))
            or not 0.0 <= float(training_prevalence) <= 1.0
            or model.get("selected_feature_layer")
            != manifest.get("selected_feature_layer")
        ):
            raise ValueError("Round 73 frozen model policy differs from its manifest")
        row_indexes = np.arange(dataset.rows, dtype=np.int64)
        prediction_payload = encode_round73_prediction_artifact(
            symbol=dataset.symbol,
            role="test",
            source_rows_sha256=dataset.source_rows_sha256,
            row_indexes=row_indexes,
            predictions=predictions,
        )
        decoded = _decoded_predictions(
            prediction_payload,
            symbol=dataset.symbol,
            source_rows_sha256=dataset.source_rows_sha256,
        )
        artifact_hash = persist_round73_test_prediction(
            database,
            study_id=claim.study_id,
            symbol=dataset.symbol,
            source_rows_sha256=dataset.source_rows_sha256,
            payload=prediction_payload,
            memory_limit=memory_limit,
            threads=threads,
        )
        stored_source, stored_hash, stored_payload = load_round73_test_prediction(
            database,
            study_id=claim.study_id,
            symbol=dataset.symbol,
            memory_limit=memory_limit,
            threads=threads,
        )
        if (
            stored_source != dataset.source_rows_sha256
            or stored_hash != artifact_hash
            or stored_payload != prediction_payload
        ):
            raise ValueError("Round 73 persisted prediction identity differs")
        model_reports[dataset.symbol] = {
            "status": "enabled",
            "rows": dataset.rows,
            "source_rows_sha256": dataset.source_rows_sha256,
            "prediction_artifact_sha256": artifact_hash,
            "prediction_artifact_bytes": len(prediction_payload),
            "selected_candidate": selected_candidate,
            "selected_feature_layer": model["selected_feature_layer"],
            "selected_probability_threshold": float(selected_threshold),
            "action_enabled": action_enabled,
            "training_prevalence": float(training_prevalence),
            "model_refit_on_test": False,
            "threshold_changed_on_test": False,
        }
        prediction_by_symbol[dataset.symbol] = decoded
        _report_progress(
            progress_callback,
            "test_symbol_prediction_persisted",
            {
                "study_id": claim.study_id,
                "symbol": dataset.symbol,
                "rows": dataset.rows,
                "prediction_artifact_sha256": artifact_hash,
                "prediction_artifact_bytes": len(prediction_payload),
            },
        )
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        if symbol not in model_reports:
            manifest = symbol_models.get(symbol)
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("status") != "disabled"
            ):
                raise ValueError("Round 73 symbol coverage differs")
            model_reports[symbol] = {
                "status": "disabled",
                "reason": str(manifest.get("reason", "")),
                "model_refit_on_test": False,
            }
    return model_reports, prediction_by_symbol


@dataclass(frozen=True)
class Round73OneUseEvaluationReport:
    status: str
    result: Mapping[str, object]
    stored: Round73StoredEvaluationResult

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "result": dict(self.result),
            "storage": self.stored.as_dict(),
        }


def _evaluate_claimed_round73_study(
    database: str | Path,
    *,
    claim: Round73EvaluationAccessClaim,
    memory_budget_bytes: int,
    memory_limit: str,
    threads: int,
    progress_callback: ProgressCallback | None,
) -> tuple[str, Mapping[str, object]]:
    observed_claim, pretest = load_round73_claimed_pretest(
        database,
        study_id=claim.study_id,
        memory_limit=memory_limit,
        threads=threads,
    )
    if observed_claim != claim:
        raise ValueError("Round 73 claimed evaluation identity changed")
    contracts = pretest.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("staged_holdout_contract_sha256")
        != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or contracts.get("evaluation_contract_sha256")
        != ROUND73_EVALUATION_CONTRACT_SHA256
        or pretest.get("test_target_or_payoff_read_before_publish") is not False
    ):
        raise ValueError("Round 73 pretest contract identity differs")
    test_exact_wire_reaudit_count = _reaudit_claimed_test_targets(
        database,
        claim=claim,
        memory_limit=memory_limit,
        threads=threads,
        progress_callback=progress_callback,
    )
    model_reports, prediction_by_symbol = _score_and_persist_test_predictions(
        database,
        claim=claim,
        pretest_identity=pretest,
        memory_budget_bytes=memory_budget_bytes,
        memory_limit=memory_limit,
        threads=threads,
        progress_callback=progress_callback,
    )
    enabled = tuple(
        symbol
        for symbol in IMPACT_CAPTURE_SYMBOLS
        if model_reports[symbol]["status"] == "enabled"
    )
    cohort_manifest_sha256 = str(pretest.get("cohort_manifest_sha256", ""))
    if _SHA256.fullmatch(cohort_manifest_sha256) is None:
        raise ValueError("Round 73 pretest cohort hash is invalid")
    all_comparisons: list[dict[str, object]] = []
    scenario_reports: list[dict[str, object]] = []
    scenario_by_key: dict[str, dict[str, object]] = {}
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        for scenario_index, scenario in enumerate(ROUND73_EVALUATION_SCENARIOS):
            _report_progress(
                progress_callback,
                "evaluation_scenario_started",
                {
                    "study_id": claim.study_id,
                    "scenario_key": scenario.key,
                    "scenario_index": scenario_index + 1,
                    "scenario_count": len(ROUND73_EVALUATION_SCENARIOS),
                },
            )
            events: list[_AttemptEvent] = []
            symbol_reports: dict[str, object] = {}
            day_ordinals: set[int] = set()
            for symbol_index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS):
                if symbol not in enabled:
                    symbol_reports[symbol] = {
                        "status": "disabled",
                        "reason": model_reports[symbol]["reason"],
                        "target_rows": _scenario_row_count(
                            connection,
                            study_id=claim.study_id,
                            symbol=symbol,
                            scenario=scenario,
                        ),
                    }
                    continue
                rows = _load_scenario_rows(
                    connection,
                    study_id=claim.study_id,
                    symbol=symbol,
                    scenario=scenario,
                    cohort_manifest_sha256=cohort_manifest_sha256,
                    test_study_manifest_sha256=(claim.test_study_manifest_sha256),
                    memory_budget_bytes=memory_budget_bytes,
                )
                expected_source = str(model_reports[symbol]["source_rows_sha256"])
                if rows.source_rows_sha256 != expected_source:
                    raise ValueError(
                        "Round 73 scenario and prediction source rows differ"
                    )
                day_ordinals.update(int(day) for day in np.unique(rows.utc_day_ordinal))
                predictive, comparisons = _predictive_evaluation(
                    rows,
                    prediction_by_symbol[symbol],
                    training_prevalence=float(
                        model_reports[symbol]["training_prevalence"]
                    ),
                    seed_offset=scenario_index * 1_000 + symbol_index * 100,
                )
                all_comparisons.extend(comparisons)
                selected_events, selection = _select_signal_events(
                    rows,
                    prediction_by_symbol[symbol],
                    selected_candidate=str(model_reports[symbol]["selected_candidate"]),
                    probability_threshold=float(
                        model_reports[symbol]["selected_probability_threshold"]
                    ),
                    action_enabled=bool(model_reports[symbol]["action_enabled"]),
                )
                events.extend(selected_events)
                symbol_reports[symbol] = {
                    "status": "enabled",
                    "source_rows_sha256": rows.source_rows_sha256,
                    "selected_candidate": model_reports[symbol]["selected_candidate"],
                    "predictive": predictive,
                    "action_selection": selection,
                }
            if not day_ordinals:
                raise ValueError("Round 73 scenario has no enabled-symbol test days")
            study_start_wall_ns = min(day_ordinals) * _DAY_NS
            study_end_wall_ns = (max(day_ordinals) + 1) * _DAY_NS
            portfolio = _simulate_portfolio(
                events,
                scenario=scenario,
                study_start_wall_ns=study_start_wall_ns,
                study_end_wall_ns=study_end_wall_ns,
                seed=ROUND73_MODEL_SEED + 50_000 + scenario_index * 10,
            )
            report = {
                **scenario.as_dict(),
                "study_start_wall_ns": study_start_wall_ns,
                "study_end_wall_ns": study_end_wall_ns,
                "test_duration_seconds": (study_end_wall_ns - study_start_wall_ns)
                / 1_000_000_000.0,
                "symbols": symbol_reports,
                "portfolio": portfolio,
                "stress_scenario_used_for_selection": False,
            }
            scenario_reports.append(report)
            scenario_by_key[scenario.key] = report
            _report_progress(
                progress_callback,
                "evaluation_scenario_completed",
                {
                    "study_id": claim.study_id,
                    "scenario_key": scenario.key,
                    "scenario_index": scenario_index + 1,
                    "scenario_count": len(ROUND73_EVALUATION_SCENARIOS),
                    "selected_attempts": len(events),
                },
            )
    multiple_testing = _apply_multiple_testing(all_comparisons)
    primary = scenario_by_key.get(_PRIMARY_SCENARIO_KEY)
    delay_stress = scenario_by_key.get(_DELAY_STRESS_SCENARIO_KEY)
    if not isinstance(primary, Mapping) or not isinstance(delay_stress, Mapping):
        raise ValueError("Round 73 required evaluation scenarios are missing")
    primary_symbols = primary.get("symbols")
    delay_symbols = delay_stress.get("symbols")
    if not isinstance(primary_symbols, Mapping) or not isinstance(
        delay_symbols,
        Mapping,
    ):
        raise ValueError("Round 73 required symbol reports are missing")
    symbol_gates: dict[str, object] = {}
    passed_symbols = 0
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        if symbol not in enabled:
            symbol_gates[symbol] = {
                "passed": False,
                "reason": "symbol_disabled_before_test",
            }
            continue
        primary_symbol = primary_symbols[symbol]
        delay_symbol = delay_symbols[symbol]
        if not isinstance(primary_symbol, Mapping) or not isinstance(
            delay_symbol,
            Mapping,
        ):
            raise ValueError("Round 73 symbol gate evidence is invalid")
        primary_predictive = primary_symbol.get("predictive")
        delay_predictive = delay_symbol.get("predictive")
        if not isinstance(primary_predictive, Mapping) or not isinstance(
            delay_predictive,
            Mapping,
        ):
            raise ValueError("Round 73 symbol predictive evidence is missing")
        gate = _symbol_predictive_gate(
            primary_predictive,
            delay_predictive,
            selected_candidate=str(model_reports[symbol]["selected_candidate"]),
        )
        symbol_gates[symbol] = gate
        passed_symbols += gate["passed"] is True
    primary_portfolio = primary.get("portfolio")
    if (
        not isinstance(primary_portfolio, Mapping)
        or not isinstance(
            primary_portfolio.get("combined"),
            Mapping,
        )
        or not isinstance(
            primary_portfolio.get("by_symbol"),
            Mapping,
        )
    ):
        raise ValueError("Round 73 primary portfolio evidence is missing")
    combined = primary_portfolio["combined"]
    independent_symbol_gates, independent_symbols_passed = (
        _independent_symbol_viability_gates(
            predictive_gates=symbol_gates,
            primary_economic_by_symbol=primary_portfolio["by_symbol"],
            enabled_symbols=enabled,
        )
    )
    predictive_gate_passed = (
        passed_symbols >= ROUND73_EVALUATION_MINIMUM_INDEPENDENT_SYMBOLS
    )
    independent_symbol_gate_passed = (
        independent_symbols_passed >= ROUND73_EVALUATION_MINIMUM_INDEPENDENT_SYMBOLS
    )
    operational_gate_passed = combined.get("operational_gate_passed") is True
    economic_gate_passed = combined.get("economic_gate_passed") is True
    viability_passed = bool(
        predictive_gate_passed
        and independent_symbol_gate_passed
        and operational_gate_passed
        and economic_gate_passed
    )
    status = "passed" if viability_passed else "failed"
    result: Mapping[str, object] = {
        "schema_version": ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
        "implementation_version": (ROUND73_ONE_USE_EVALUATION_IMPLEMENTATION_VERSION),
        "study_id": claim.study_id,
        "status": status,
        "staged_holdout_contract_sha256": (ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256),
        "evaluation_contract_sha256": ROUND73_EVALUATION_CONTRACT_SHA256,
        "pretest_manifest_sha256": claim.pretest_manifest_sha256,
        "test_study_manifest_sha256": claim.test_study_manifest_sha256,
        "repository_commit_sha": claim.repository_commit_sha,
        "repository_tree_sha": claim.repository_tree_sha,
        "test_access_claimed_at_wall_ns": claim.claimed_at_wall_ns,
        "test_read_count": 1,
        "test_access_claim_count": 1,
        "test_exact_wire_reaudit_count": test_exact_wire_reaudit_count,
        "test_target_scans_are_within_single_claimed_evaluation": True,
        "models": model_reports,
        "multiple_testing": multiple_testing,
        "predictive_symbol_gates": symbol_gates,
        "predictive_symbols_passed": passed_symbols,
        "independent_symbol_viability_gates": independent_symbol_gates,
        "independent_symbols_passed": independent_symbols_passed,
        "minimum_independent_symbols_required": (
            ROUND73_EVALUATION_MINIMUM_INDEPENDENT_SYMBOLS
        ),
        "predictive_gate_passed": predictive_gate_passed,
        "independent_symbol_gate_passed": independent_symbol_gate_passed,
        "economic_gate_passed": economic_gate_passed,
        "operational_gate_passed": operational_gate_passed,
        "seven_day_viability_gate_passed": viability_passed,
        "scenario_reports": scenario_reports,
        "test_predictions_generated_only_from_pretest_model_bytes": True,
        "model_refit_recalibration_or_threshold_change_on_test": False,
        "stress_scenario_selected_model_or_policy": False,
        "crypto_formal_daily_close": False,
        "listed_venue_calendar_authorizes_crypto_execution": False,
        "annualized_roi_reported": False,
        "durable_predictive_edge_claim": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "paper_testnet_or_live_trading_authority": False,
        "interpretation": (
            "Seven-day viability evidence only; it cannot establish durable "
            "profitability, production readiness, or trading authority."
        ),
    }
    _canonical_json(result)
    return status, result


def evaluate_round73_once(
    database: str | Path,
    *,
    study_id: str,
    pretest_manifest_sha256: str,
    repository_root: str | Path,
    memory_budget_bytes: int = ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES,
    memory_limit: str = "2GB",
    threads: int = 2,
    repository_state_function: RepositoryStateFunction = round73_repository_state,
    progress_callback: ProgressCallback | None = None,
) -> Round73OneUseEvaluationReport:
    """Claim, score, persist, and permanently close the frozen test once."""

    selected_study = str(study_id).strip().lower()
    selected_pretest = str(pretest_manifest_sha256).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 evaluation study ID is invalid")
    if _SHA256.fullmatch(selected_pretest) is None:
        raise ValueError("Round 73 evaluation pretest hash is invalid")
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("Round 73 evaluation memory budget must be positive")
    _report_progress(
        progress_callback,
        "evaluation_access_claim_started",
        {"study_id": selected_study},
    )
    claim = claim_round73_evaluation_access(
        database,
        study_id=selected_study,
        pretest_manifest_sha256=selected_pretest,
        repository_root=repository_root,
        memory_limit=memory_limit,
        threads=threads,
        repository_state_function=repository_state_function,
    )
    _report_progress(
        progress_callback,
        "evaluation_access_claimed",
        {
            "study_id": claim.study_id,
            "pretest_manifest_sha256": claim.pretest_manifest_sha256,
            "test_study_manifest_sha256": claim.test_study_manifest_sha256,
            "test_read_count": 1,
        },
    )
    try:
        status, result = _evaluate_claimed_round73_study(
            database,
            claim=claim,
            memory_budget_bytes=memory_budget_bytes,
            memory_limit=memory_limit,
            threads=threads,
            progress_callback=progress_callback,
        )
        _report_progress(
            progress_callback,
            "evaluation_result_persist_started",
            {"study_id": claim.study_id, "status": status},
        )
        stored = persist_round73_evaluation_result(
            database,
            claim=claim,
            status=status,
            result=result,
            memory_limit=memory_limit,
            threads=threads,
        )
        _report_progress(
            progress_callback,
            "evaluation_result_persisted",
            {
                "study_id": claim.study_id,
                "status": status,
                "result_sha256": stored.result_sha256,
            },
        )
        return Round73OneUseEvaluationReport(
            status=status,
            result=result,
            stored=stored,
        )
    except BaseException as exc:
        reason = f"{exc.__class__.__name__}: {' '.join(str(exc).split())}"
        _report_progress(
            progress_callback,
            "evaluation_interrupted",
            {"study_id": claim.study_id, "reason": reason[:500]},
        )
        try:
            finalize_interrupted_round73_evaluation(
                database,
                study_id=claim.study_id,
                reason=reason,
                memory_limit=memory_limit,
                threads=threads,
            )
        except BaseException as persistence_error:
            exc.add_note(
                "Round 73 interrupted-result persistence also failed: "
                f"{persistence_error.__class__.__name__}: {persistence_error}"
            )
        raise


__all__ = [
    "ROUND73_EVALUATION_BOOTSTRAP_DRAWS",
    "ROUND73_EVALUATION_MAXIMUM_Q_VALUE",
    "ROUND73_EVALUATION_MINIMUM_COMPLETED_TRADES",
    "ROUND73_EVALUATION_MINIMUM_DAILY_TRADES",
    "ROUND73_EVALUATION_MINIMUM_INDEPENDENT_SYMBOLS",
    "ROUND73_EVALUATION_PERMUTATION_DRAWS",
    "ROUND73_EVALUATION_SCENARIOS",
    "ROUND73_ONE_USE_EVALUATION_IMPLEMENTATION_VERSION",
    "Round73EvaluationScenario",
    "Round73OneUseEvaluationReport",
    "evaluate_round73_once",
]
