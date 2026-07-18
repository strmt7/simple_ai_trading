"""Round 11 development-only single-leg Polymarket value experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .paper_execution import PolymarketFeeModel
from .polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES
from .polymarket_features import POLYMARKET_FEATURE_NAMES
from .polymarket_hurdle import polymarket_maximum_entry_loss
from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_ROUND11_CONTRACT_SHA256 = (
    "ced2dfcb058845f3cc430c369b00b0cd493c61b739ff1f0fa3d27b052af1aff4"
)
POLYMARKET_ROUND11_SOURCE_IMPLEMENTATION_SHA256 = (
    "5e75c49312431c3bc33c3ace33f2edf061acd6d4e6fa5c0151c76779e9f528ab"
)
POLYMARKET_ROUND11_DATASET_SCHEMA_VERSION = (
    "polymarket-round11-single-leg-development-dataset-v1"
)
POLYMARKET_ROUND11_MODEL_SCHEMA_VERSION = (
    "polymarket-round11-single-leg-transparent-model-v1"
)
POLYMARKET_ROUND11_REPORT_SCHEMA_VERSION = (
    "polymarket-round11-single-leg-development-report-v1"
)
POLYMARKET_ROUND11_L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
POLYMARKET_ROUND11_MARGIN_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
POLYMARKET_ROUND11_MINIMUM_REMAINING_GRID = (120.0, 90.0, 60.0, 45.0, 30.0)
POLYMARKET_ROUND11_RETRY_NS = 1_000_000_000
POLYMARKET_ROUND11_SCALER_CLIP = 12.0
POLYMARKET_ROUND11_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND11_BOOTSTRAP_SEED = 11_011
POLYMARKET_ROUND11_PINBALL_SMOOTHING = 0.05
POLYMARKET_ROUND11_COST_QUANTILE = 0.90

_ASSETS = ("BTC", "ETH", "SOL")
_OUTCOMES = ("Down", "Up")
_SHA256_LENGTH = 64
_UNKNOWN_ENTRY_REASONS = frozenset(
    {
        "entry_confirmation_enters_excluded_close_window",
        "missing_entry_execution_book",
    }
)
_KNOWN_PRE_SUBMIT_UNAVAILABLE_REASONS = frozenset(
    {
        "entry_enters_excluded_close_window",
        "missing_entry_creation_book",
        "missing_entry_execution_parameters",
        "unsupported_entry_minimum_order_age",
    }
)
_EXECUTION_STATIC_FEATURE_NAMES = (
    "current_best_bid",
    "current_best_ask",
    "current_top_ask_cost_quote",
    "active_tick_size",
    "minimum_order_quantity",
    "fees_enabled",
    "fee_rate",
    "fee_exponent",
    "crypto_taker_delay_enabled",
    "maximum_entry_loss_quote",
)
POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES = tuple(POLYMARKET_ACTION_FEATURE_NAMES)
POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES = (
    *POLYMARKET_ACTION_FEATURE_NAMES,
    *_EXECUTION_STATIC_FEATURE_NAMES,
)

ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _finite_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), 1e-9, 1.0 - 1e-9)


def _logit(value: np.ndarray) -> np.ndarray:
    probability = _finite_probability(value)
    return np.log(probability) - np.log1p(-probability)


@dataclass(frozen=True)
class PolymarketRound11Dataset:
    """Compact paired-action dataset; one row is one point-in-time decision."""

    pipeline_report_sha256: str
    dataset_sha256: str
    condition_ids: tuple[str, ...]
    source_feature_sha256: np.ndarray
    condition_index: np.ndarray
    asset_index: np.ndarray
    event_start_ms: np.ndarray
    decision_monotonic_ns: np.ndarray
    remaining_seconds: np.ndarray
    official_up: np.ndarray
    market_prior_up: np.ndarray
    direction_features: np.ndarray
    execution_features: np.ndarray
    available: np.ndarray
    observable: np.ndarray
    entry_filled: np.ndarray
    unknown_entry: np.ndarray
    entry_cost_quote: np.ndarray
    current_top_ask_cost_quote: np.ndarray
    maximum_entry_loss_quote: np.ndarray
    realized_hold_utility_quote: np.ndarray
    terminal_reason_code: np.ndarray
    terminal_reasons: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.direction_features.shape[0])

    @property
    def groups(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(self.event_start_ms))

    def validated(self) -> PolymarketRound11Dataset:
        rows = self.rows
        paired = (
            self.execution_features,
            self.available,
            self.observable,
            self.entry_filled,
            self.unknown_entry,
            self.entry_cost_quote,
            self.current_top_ask_cost_quote,
            self.maximum_entry_loss_quote,
            self.realized_hold_utility_quote,
            self.terminal_reason_code,
        )
        one_dimensional = (
            self.source_feature_sha256,
            self.condition_index,
            self.asset_index,
            self.event_start_ms,
            self.decision_monotonic_ns,
            self.remaining_seconds,
            self.official_up,
            self.market_prior_up,
        )
        invalid = (
            len(self.pipeline_report_sha256) != _SHA256_LENGTH
            or len(self.dataset_sha256) != _SHA256_LENGTH
            or rows <= 0
            or self.direction_features.shape
            != (rows, len(POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES))
            or any(value.shape != (rows,) for value in one_dimensional)
            or any(value.shape[:2] != (rows, 2) for value in paired)
            or self.execution_features.shape[2]
            != len(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES)
            or not np.all(np.isfinite(self.direction_features))
            or not np.all(np.isfinite(self.execution_features))
            or not np.all(np.isfinite(self.remaining_seconds))
            or not np.all(np.isfinite(self.market_prior_up))
            or np.any(self.market_prior_up <= 0)
            or np.any(self.market_prior_up >= 1)
            or np.any(self.condition_index < 0)
            or np.any(self.condition_index >= len(self.condition_ids))
            or np.any(self.asset_index < 0)
            or np.any(self.asset_index >= len(_ASSETS))
            or np.any(self.observable & ~self.available)
            or np.any(self.entry_filled & ~self.observable)
            or np.any(self.unknown_entry != (self.available & ~self.observable))
            or np.any(self.maximum_entry_loss_quote <= 0)
            or np.any(self.current_top_ask_cost_quote <= 0)
            or np.any(self.current_top_ask_cost_quote > self.maximum_entry_loss_quote)
            or np.any(self.terminal_reason_code < 0)
            or np.any(self.terminal_reason_code >= len(self.terminal_reasons))
        )
        if invalid:
            raise ValueError("Polymarket Round 11 dataset is invalid")
        if np.any(np.isfinite(self.entry_cost_quote) != self.entry_filled) or np.any(
            np.abs(
                self.realized_hold_utility_quote[self.unknown_entry]
                + self.maximum_entry_loss_quote[self.unknown_entry]
            )
            > 1e-9
        ):
            raise ValueError("Polymarket Round 11 action labels are inconsistent")
        inactive = ~self.available | (self.observable & ~self.entry_filled)
        if np.any(np.abs(self.realized_hold_utility_quote[inactive]) > 1e-12):
            raise ValueError("Polymarket Round 11 inactive utility is nonzero")
        return self


@dataclass(frozen=True)
class PolymarketRound11Split:
    train_groups: tuple[int, ...]
    purge_group: int
    validation_groups: tuple[int, ...]
    split_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round11-development-split-v1",
            "train_groups": list(self.train_groups),
            "purge_group": self.purge_group,
            "validation_groups": list(self.validation_groups),
        }

    def validated(self) -> PolymarketRound11Split:
        combined = (*self.train_groups, self.purge_group, *self.validation_groups)
        if (
            len(self.train_groups) != 32
            or len(self.validation_groups) != 14
            or tuple(sorted(combined)) != combined
            or len(set(combined)) != 47
            or self.split_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 11 split is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRound11Scaler:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float
    scaler_sha256: str

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.center):
            raise ValueError("Polymarket Round 11 scaler input is invalid")
        transformed = (values - np.asarray(self.center)) / np.asarray(self.scale)
        return np.clip(transformed, -self.clip, self.clip)

    def identity_payload(self) -> dict[str, object]:
        return {
            "center": list(self.center),
            "scale": list(self.scale),
            "clip": self.clip,
        }


@dataclass(frozen=True)
class PolymarketRound11LinearHead:
    name: str
    objective: str
    l2: float
    coefficients: tuple[float, ...]
    intercept: float
    target_center: float
    target_scale: float
    optimizer_iterations: int
    training_objective: float
    model_sha256: str

    def linear(self, features: np.ndarray) -> np.ndarray:
        return (
            np.asarray(features, dtype=np.float64) @ np.asarray(self.coefficients)
            + self.intercept
        )

    def predict_binary(
        self, features: np.ndarray, *, offset: np.ndarray | None = None
    ) -> np.ndarray:
        logits = self.linear(features)
        if offset is not None:
            logits = logits + np.asarray(offset, dtype=np.float64)
        return expit(logits)

    def predict_value(self, features: np.ndarray) -> np.ndarray:
        return self.target_center + self.target_scale * self.linear(features)

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("model_sha256")
        return payload


@dataclass(frozen=True)
class PolymarketRound11Calibration:
    method: str
    intercept: float
    slope: float
    oof_log_loss: float
    calibration_sha256: str

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return expit(self.intercept + self.slope * _logit(probabilities))

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("calibration_sha256")
        return payload


def _current_top_cost(
    *,
    price: Decimal,
    quantity: Decimal,
    fees_enabled: bool,
    fee_rate: Decimal,
    fee_exponent: int,
) -> Decimal:
    fee = PolymarketFeeModel(
        enabled=fees_enabled,
        rate=fee_rate,
        exponent=fee_exponent,
        taker_only=True,
    )(price, quantity, "taker")
    return price * quantity + fee


def _pipeline_source_dataset_ids(
    store: PolymarketEvidenceStore, pipeline_report_sha256: str
) -> tuple[str, tuple[str, ...], int]:
    selected = str(pipeline_report_sha256).strip().lower()
    if len(selected) != _SHA256_LENGTH:
        raise ValueError("Polymarket Round 11 pipeline digest is invalid")
    connection = store.connect()
    row = connection.execute(
        """
        SELECT run_id, action_dataset_sha256_json, action_count,
               implementation_sha256
        FROM polymarket_action_value_pipeline
        WHERE report_sha256 = ?
        """,
        [selected],
    ).fetchone()
    if row is None:
        raise ValueError("Polymarket Round 11 source pipeline does not exist")
    dataset_ids = tuple(str(value) for value in json.loads(str(row[1])))
    if (
        len(dataset_ids) != 47
        or int(row[2]) != 251_892
        or str(row[3]) != POLYMARKET_ROUND11_SOURCE_IMPLEMENTATION_SHA256
    ):
        raise ValueError("Polymarket Round 11 source pipeline identity differs")
    return str(row[0]), dataset_ids, int(row[2])


def load_round11_development_dataset(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
    progress: ProgressCallback | None = None,
) -> PolymarketRound11Dataset:
    """Load paired real replay actions without mutating the evidence database."""

    emit = progress or (lambda _stage, _payload: None)
    selected_report = str(pipeline_report_sha256).strip().lower()
    run_id, dataset_ids, action_count = _pipeline_source_dataset_ids(
        store, selected_report
    )
    connection = store.connect()
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE round11_source_dataset("
        "dataset_sha256 VARCHAR PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO round11_source_dataset VALUES (?)",
        [(value,) for value in dataset_ids],
    )
    schema_rows = connection.execute(
        """
        SELECT DISTINCT f.feature_names_json
        FROM polymarket_action_value_dataset AS a
        JOIN round11_source_dataset AS selected USING (dataset_sha256)
        JOIN polymarket_feature_dataset AS f
          ON f.dataset_sha256 = a.source_feature_dataset_sha256
        """
    ).fetchall()
    if len(schema_rows) != 1 or tuple(json.loads(str(schema_rows[0][0]))) != tuple(
        POLYMARKET_FEATURE_NAMES
    ):
        raise ValueError("Polymarket Round 11 source feature schema differs")
    condition_rows = connection.execute(
        """
        SELECT DISTINCT m.condition_id
        FROM polymarket_action_value_dataset AS d
        JOIN round11_source_dataset AS selected USING (dataset_sha256)
        JOIN polymarket_market_snapshot AS m
          ON m.run_id = d.source_run_id
        JOIN polymarket_action_value_row AS a
          ON a.dataset_sha256 = d.dataset_sha256
         AND a.condition_id = m.condition_id
        ORDER BY m.condition_id
        """
    ).fetchall()
    condition_ids = tuple(str(row[0]) for row in condition_rows)
    if len(condition_ids) != 141:
        raise ValueError("Polymarket Round 11 condition breadth differs")
    condition_lookup = {
        condition_id: index for index, condition_id in enumerate(condition_ids)
    }

    rows_expected = action_count // 2
    direction_width = len(POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES)
    execution_width = len(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES)
    source_hash = np.empty(rows_expected, dtype="S64")
    condition_index = np.empty(rows_expected, dtype=np.int32)
    asset_index = np.empty(rows_expected, dtype=np.int8)
    event_start_ms = np.empty(rows_expected, dtype=np.int64)
    decision_ns = np.empty(rows_expected, dtype=np.uint64)
    remaining_seconds = np.empty(rows_expected, dtype=np.float64)
    official_up = np.empty(rows_expected, dtype=np.bool_)
    prior_up = np.empty(rows_expected, dtype=np.float64)
    direction_features = np.empty((rows_expected, direction_width), dtype=np.float64)
    execution_features = np.empty((rows_expected, 2, execution_width), dtype=np.float64)
    available = np.empty((rows_expected, 2), dtype=np.bool_)
    observable = np.empty((rows_expected, 2), dtype=np.bool_)
    entry_filled = np.empty((rows_expected, 2), dtype=np.bool_)
    unknown_entry = np.empty((rows_expected, 2), dtype=np.bool_)
    entry_cost = np.full((rows_expected, 2), np.nan, dtype=np.float64)
    current_top_cost = np.empty((rows_expected, 2), dtype=np.float64)
    maximum_loss = np.empty((rows_expected, 2), dtype=np.float64)
    realized_utility = np.empty((rows_expected, 2), dtype=np.float64)
    terminal_code = np.empty((rows_expected, 2), dtype=np.int8)
    terminal_reasons: list[str] = []
    reason_lookup: dict[str, int] = {}
    source_name_index = {
        name: index for index, name in enumerate(POLYMARKET_FEATURE_NAMES)
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": POLYMARKET_ROUND11_DATASET_SCHEMA_VERSION,
                "contract_sha256": POLYMARKET_ROUND11_CONTRACT_SHA256,
                "pipeline_report_sha256": selected_report,
                "direction_feature_names": POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES,
                "execution_feature_names": POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    cursor = connection.execute(
        """
        SELECT f.feature_id, f.row_sha256, f.condition_id, f.asset,
               f.decision_received_monotonic_ns, f.feature_values_json,
               up.action_feature_sha256, up.action_label_sha256,
               up.feature_values_json, up.terminal_reason,
               up.entry_filled, up.entry_cost_quote,
               down.action_feature_sha256, down.action_label_sha256,
               down.feature_values_json, down.terminal_reason,
               down.entry_filled, down.entry_cost_quote,
               m.event_start_ms, m.tick_size, m.minimum_order_size,
               m.fees_enabled, m.fee_rate, m.fee_exponent,
               m.taker_order_delay_enabled,
               resolution.winning_outcome, resolution.evidence_sha256
        FROM polymarket_action_value_dataset AS d
        JOIN round11_source_dataset AS selected USING (dataset_sha256)
        JOIN polymarket_feature_row AS f
          ON f.dataset_id = d.source_feature_dataset_sha256
        JOIN polymarket_action_value_row AS up
          ON up.dataset_sha256 = d.dataset_sha256
         AND up.source_feature_id = f.feature_id AND up.outcome = 'Up'
        JOIN polymarket_action_value_row AS down
          ON down.dataset_sha256 = d.dataset_sha256
         AND down.source_feature_id = f.feature_id AND down.outcome = 'Down'
        JOIN polymarket_market_snapshot AS m
          ON m.run_id = d.source_run_id AND m.condition_id = f.condition_id
        JOIN polymarket_resolution_evidence AS resolution
          ON resolution.run_id = d.source_run_id
         AND resolution.condition_id = f.condition_id
        ORDER BY m.event_start_ms, f.decision_received_monotonic_ns,
                 f.condition_id
        """
    )
    size = 0
    while batch := cursor.fetchmany(8_192):
        for row in batch:
            raw_source = tuple(float(value) for value in json.loads(str(row[5])))
            if len(raw_source) != len(POLYMARKET_FEATURE_NAMES):
                raise ValueError("Polymarket Round 11 source feature width differs")
            up_action = tuple(float(value) for value in json.loads(str(row[8])))
            down_action = tuple(float(value) for value in json.loads(str(row[14])))
            if len(up_action) != direction_width or len(down_action) != direction_width:
                raise ValueError("Polymarket Round 11 action feature width differs")
            up_mid = raw_source[source_name_index["up_midpoint"]]
            down_mid = raw_source[source_name_index["down_midpoint"]]
            midpoint_sum = up_mid + down_mid
            if midpoint_sum <= 0 or not math.isfinite(midpoint_sum):
                raise ValueError("Polymarket Round 11 market prior is invalid")
            prior = min(1.0 - 1e-6, max(1e-6, up_mid / midpoint_sum))
            tick = Decimal(str(row[19]))
            quantity = Decimal(str(row[20]))
            fees_enabled = bool(row[21])
            fee_rate = Decimal(str(row[22]))
            fee_exponent = int(row[23])
            taker_delay = bool(row[24])
            max_loss = polymarket_maximum_entry_loss(
                tick_size=tick,
                quantity=quantity,
                fees_enabled=fees_enabled,
                fee_rate=fee_rate,
                fee_exponent=fee_exponent,
            )
            best_bid = (
                raw_source[source_name_index["down_best_bid"]],
                raw_source[source_name_index["up_best_bid"]],
            )
            best_ask = (
                raw_source[source_name_index["down_best_ask"]],
                raw_source[source_name_index["up_best_ask"]],
            )
            source_feature_id = str(row[0])
            source_hash[size] = source_feature_id.encode("ascii")
            condition_id = str(row[2])
            condition_index[size] = condition_lookup[condition_id]
            asset = str(row[3])
            if asset not in _ASSETS:
                raise ValueError("Polymarket Round 11 source asset differs")
            asset_index[size] = _ASSETS.index(asset)
            event_start_ms[size] = int(row[18])
            decision_ns[size] = int(row[4])
            remaining_seconds[size] = float(up_action[0])
            winning_outcome = str(row[25])
            if winning_outcome not in _OUTCOMES:
                raise ValueError("Polymarket Round 11 official outcome differs")
            official_up[size] = winning_outcome == "Up"
            prior_up[size] = prior
            direction_features[size] = up_action
            action_values = (down_action, up_action)
            reasons = (str(row[15]), str(row[9]))
            filled_values = (bool(row[16]), bool(row[10]))
            cost_values = (row[17], row[11])
            action_hashes = (str(row[12]), str(row[6]))
            label_hashes = (str(row[13]), str(row[7]))
            for outcome_index, outcome in enumerate(_OUTCOMES):
                reason = reasons[outcome_index]
                if reason not in reason_lookup:
                    reason_lookup[reason] = len(terminal_reasons)
                    terminal_reasons.append(reason)
                is_available = reason not in _KNOWN_PRE_SUBMIT_UNAVAILABLE_REASONS
                is_unknown = reason in _UNKNOWN_ENTRY_REASONS
                is_observable = is_available and not is_unknown
                was_filled = filled_values[outcome_index]
                ask = Decimal(str(best_ask[outcome_index]))
                top_cost = _current_top_cost(
                    price=ask,
                    quantity=quantity,
                    fees_enabled=fees_enabled,
                    fee_rate=fee_rate,
                    fee_exponent=fee_exponent,
                )
                static = (
                    best_bid[outcome_index],
                    best_ask[outcome_index],
                    float(top_cost),
                    float(tick),
                    float(quantity),
                    float(fees_enabled),
                    float(fee_rate),
                    float(fee_exponent),
                    float(taker_delay),
                    float(max_loss),
                )
                execution_features[size, outcome_index] = (
                    *action_values[outcome_index],
                    *static,
                )
                available[size, outcome_index] = is_available
                observable[size, outcome_index] = is_observable
                entry_filled[size, outcome_index] = was_filled
                unknown_entry[size, outcome_index] = is_unknown
                current_top_cost[size, outcome_index] = float(top_cost)
                maximum_loss[size, outcome_index] = float(max_loss)
                terminal_code[size, outcome_index] = reason_lookup[reason]
                if is_unknown:
                    realized_utility[size, outcome_index] = -float(max_loss)
                elif not is_available or not was_filled:
                    realized_utility[size, outcome_index] = 0.0
                else:
                    if cost_values[outcome_index] is None:
                        raise ValueError(
                            "Polymarket Round 11 filled entry cost is missing"
                        )
                    exact_cost = float(Decimal(str(cost_values[outcome_index])))
                    entry_cost[size, outcome_index] = exact_cost
                    won = (outcome == "Up") == bool(official_up[size])
                    realized_utility[size, outcome_index] = (
                        float(quantity) - exact_cost if won else -exact_cost
                    )
                digest.update(bytes.fromhex(action_hashes[outcome_index]))
                digest.update(bytes.fromhex(label_hashes[outcome_index]))
            digest.update(bytes.fromhex(str(row[1])))
            digest.update(bytes.fromhex(str(row[26])))
            digest.update(format(max_loss, "f").encode("ascii"))
            size += 1
        emit("load", {"rows": size, "rows_expected": rows_expected})
    if size != rows_expected:
        raise ValueError("Polymarket Round 11 paired row count differs")
    dataset = PolymarketRound11Dataset(
        pipeline_report_sha256=selected_report,
        dataset_sha256=digest.hexdigest(),
        condition_ids=condition_ids,
        source_feature_sha256=np.ascontiguousarray(source_hash[:size]),
        condition_index=np.ascontiguousarray(condition_index[:size]),
        asset_index=np.ascontiguousarray(asset_index[:size]),
        event_start_ms=np.ascontiguousarray(event_start_ms[:size]),
        decision_monotonic_ns=np.ascontiguousarray(decision_ns[:size]),
        remaining_seconds=np.ascontiguousarray(remaining_seconds[:size]),
        official_up=np.ascontiguousarray(official_up[:size]),
        market_prior_up=np.ascontiguousarray(prior_up[:size]),
        direction_features=np.ascontiguousarray(direction_features[:size]),
        execution_features=np.ascontiguousarray(execution_features[:size]),
        available=np.ascontiguousarray(available[:size]),
        observable=np.ascontiguousarray(observable[:size]),
        entry_filled=np.ascontiguousarray(entry_filled[:size]),
        unknown_entry=np.ascontiguousarray(unknown_entry[:size]),
        entry_cost_quote=np.ascontiguousarray(entry_cost[:size]),
        current_top_ask_cost_quote=np.ascontiguousarray(current_top_cost[:size]),
        maximum_entry_loss_quote=np.ascontiguousarray(maximum_loss[:size]),
        realized_hold_utility_quote=np.ascontiguousarray(realized_utility[:size]),
        terminal_reason_code=np.ascontiguousarray(terminal_code[:size]),
        terminal_reasons=tuple(terminal_reasons),
    )
    return dataset.validated()


def build_round11_development_split(
    dataset: PolymarketRound11Dataset,
) -> PolymarketRound11Split:
    groups = dataset.validated().groups
    if len(groups) != 47:
        raise ValueError("Polymarket Round 11 source group breadth differs")
    provisional = PolymarketRound11Split(
        train_groups=groups[:32],
        purge_group=groups[32],
        validation_groups=groups[33:],
        split_sha256="",
    )
    return replace(
        provisional,
        split_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _condition_balanced_weights(
    condition_index: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    conditions = np.asarray(condition_index, dtype=np.int64)
    selected = np.asarray(mask, dtype=np.bool_)
    if (
        conditions.ndim != 1
        or selected.shape != conditions.shape
        or not np.any(selected)
    ):
        raise ValueError("Polymarket Round 11 weighting mask is empty")
    chosen = conditions[selected]
    counts = np.bincount(chosen, minlength=int(np.max(conditions)) + 1)
    weights = 1.0 / counts[chosen]
    weights /= np.sum(weights)
    return weights


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 1
        or mass.shape != data.shape
        or data.size == 0
        or not np.all(np.isfinite(data))
        or not np.all(np.isfinite(mass))
        or np.any(mass < 0)
        or np.sum(mass) <= 0
        or not 0 <= probability <= 1
    ):
        raise ValueError("Polymarket Round 11 weighted quantile input is invalid")
    order = np.argsort(data, kind="mergesort")
    ordered = data[order]
    cumulative = np.cumsum(mass[order])
    target = float(probability) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered[min(index, ordered.size - 1)])


def _fit_scaler(features: np.ndarray, weights: np.ndarray) -> PolymarketRound11Scaler:
    values = np.asarray(features, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or mass.shape != (values.shape[0],)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Polymarket Round 11 scaler input is invalid")
    center = np.empty(values.shape[1], dtype=np.float64)
    scale = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        q25 = _weighted_quantile(values[:, column], mass, 0.25)
        q50 = _weighted_quantile(values[:, column], mass, 0.50)
        q75 = _weighted_quantile(values[:, column], mass, 0.75)
        center[column] = q50
        width = q75 - q25
        scale[column] = width if math.isfinite(width) and width > 1e-12 else 1.0
    payload = {
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
        "clip": POLYMARKET_ROUND11_SCALER_CLIP,
    }
    return PolymarketRound11Scaler(
        center=tuple(payload["center"]),
        scale=tuple(payload["scale"]),
        clip=POLYMARKET_ROUND11_SCALER_CLIP,
        scaler_sha256=_canonical_sha256(payload),
    )


def _weighted_binary_log_loss(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    target = np.asarray(labels, dtype=np.float64)
    prediction = _finite_probability(probabilities)
    mass = np.asarray(weights, dtype=np.float64)
    if target.shape != prediction.shape or mass.shape != target.shape:
        raise ValueError("Polymarket Round 11 binary metric input is invalid")
    losses = -(target * np.log(prediction) + (1.0 - target) * np.log1p(-prediction))
    return float(np.sum(mass * losses) / np.sum(mass))


def _weighted_brier(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    target = np.asarray(labels, dtype=np.float64)
    prediction = np.asarray(probabilities, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    return float(np.sum(mass * (prediction - target) ** 2) / np.sum(mass))


def _make_head(
    *,
    name: str,
    objective: str,
    l2: float,
    coefficients: np.ndarray,
    intercept: float,
    target_center: float,
    target_scale: float,
    optimizer_iterations: int,
    training_objective: float,
) -> PolymarketRound11LinearHead:
    provisional = PolymarketRound11LinearHead(
        name=name,
        objective=objective,
        l2=float(l2),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        target_center=float(target_center),
        target_scale=float(target_scale),
        optimizer_iterations=int(optimizer_iterations),
        training_objective=float(training_objective),
        model_sha256="",
    )
    return replace(
        provisional,
        model_sha256=_canonical_sha256(provisional.identity_payload()),
    )


def _fit_binary_head(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    name: str,
    l2: float,
    offset: np.ndarray | None = None,
) -> PolymarketRound11LinearHead:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    base = np.zeros(y.shape, dtype=np.float64) if offset is None else np.asarray(offset)
    if (
        x.ndim != 2
        or y.shape != (x.shape[0],)
        or mass.shape != y.shape
        or base.shape != y.shape
        or x.shape[0] == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(base))
        or not set(np.unique(y)).issubset({0.0, 1.0})
        or len(np.unique(y)) != 2
        or l2 <= 0
    ):
        raise ValueError(f"Polymarket Round 11 {name} fit input is invalid")
    mass = mass / np.sum(mass)
    prevalence = float(np.sum(mass * y))
    initial_intercept = 0.0
    if offset is None:
        initial_intercept = math.log(prevalence / (1.0 - prevalence))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        logits = base + x @ coefficients + theta[-1]
        losses = np.logaddexp(0.0, logits) - y * logits
        value = float(
            np.sum(mass * losses) + 0.5 * l2 * np.dot(coefficients, coefficients)
        )
        residual = mass * (expit(logits) - y)
        gradient = np.empty_like(theta)
        gradient[:-1] = x.T @ residual + l2 * coefficients
        gradient[-1] = float(np.sum(residual))
        return value, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[-1] = initial_intercept
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 350, "ftol": 1e-11, "gtol": 1e-7},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(
            f"Polymarket Round 11 {name} optimizer failed:{result.message}"
        )
    return _make_head(
        name=name,
        objective="weighted_binary_log_loss",
        l2=l2,
        coefficients=result.x[:-1],
        intercept=float(result.x[-1]),
        target_center=0.0,
        target_scale=1.0,
        optimizer_iterations=int(result.nit),
        training_objective=float(result.fun),
    )


def _weighted_target_scale(
    target: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    center = _weighted_quantile(target, weights, 0.50)
    scale = _weighted_quantile(target, weights, 0.75) - _weighted_quantile(
        target, weights, 0.25
    )
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _fit_value_head(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    name: str,
    objective_name: str,
    l2: float,
) -> PolymarketRound11LinearHead:
    x = np.asarray(features, dtype=np.float64)
    raw_target = np.asarray(target, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 2
        or raw_target.shape != (x.shape[0],)
        or mass.shape != raw_target.shape
        or x.shape[0] == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(raw_target))
        or l2 <= 0
    ):
        raise ValueError(f"Polymarket Round 11 {name} fit input is invalid")
    mass = mass / np.sum(mass)
    center, scale = _weighted_target_scale(raw_target, mass)
    y = (raw_target - center) / scale
    if objective_name == "huber_mean":
        initial_intercept = float(np.sum(mass * y))

        def loss_gradient(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            absolute = np.abs(residual)
            loss = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
            gradient = np.clip(residual, -1.0, 1.0)
            return loss, gradient

    elif objective_name == "pinball_q90":
        initial_intercept = _weighted_quantile(
            y, mass, POLYMARKET_ROUND11_COST_QUANTILE
        )

        def loss_gradient(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            smoothing = POLYMARKET_ROUND11_PINBALL_SMOOTHING
            positive = residual / smoothing
            negative = -residual / smoothing
            quantile = POLYMARKET_ROUND11_COST_QUANTILE
            loss = smoothing * (
                (1.0 - quantile) * np.logaddexp(0.0, positive)
                + quantile * np.logaddexp(0.0, negative)
            )
            gradient = (1.0 - quantile) * expit(positive) - quantile * expit(negative)
            return loss, gradient

    else:
        raise ValueError("Polymarket Round 11 value objective is invalid")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        residual = x @ coefficients + theta[-1] - y
        losses, data_gradient = loss_gradient(residual)
        value = float(
            np.sum(mass * losses) + 0.5 * l2 * np.dot(coefficients, coefficients)
        )
        weighted_gradient = mass * data_gradient
        gradient = np.empty_like(theta)
        gradient[:-1] = x.T @ weighted_gradient + l2 * coefficients
        gradient[-1] = float(np.sum(weighted_gradient))
        return value, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[-1] = initial_intercept
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 350, "ftol": 1e-11, "gtol": 1e-7},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(
            f"Polymarket Round 11 {name} optimizer failed:{result.message}"
        )
    return _make_head(
        name=name,
        objective=objective_name,
        l2=l2,
        coefficients=result.x[:-1],
        intercept=float(result.x[-1]),
        target_center=center,
        target_scale=scale,
        optimizer_iterations=int(result.nit),
        training_objective=float(result.fun),
    )


def _pinball_loss(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    y = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    error = y - estimate
    losses = np.maximum(
        POLYMARKET_ROUND11_COST_QUANTILE * error,
        (POLYMARKET_ROUND11_COST_QUANTILE - 1.0) * error,
    )
    return float(np.sum(mass * losses) / np.sum(mass))


def _huber_loss(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    residual = np.asarray(prediction) - np.asarray(target)
    absolute = np.abs(residual)
    losses = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
    return float(np.sum(np.asarray(weights) * losses) / np.sum(weights))


def _training_folds(
    groups: Sequence[int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    ordered = tuple(int(value) for value in groups)
    if len(ordered) != 32:
        raise ValueError("Polymarket Round 11 training fold source differs")
    return tuple(
        (ordered[:start], ordered[start : start + 4]) for start in (16, 20, 24, 28)
    )


def _calibration_candidates(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[PolymarketRound11Calibration, ...]:
    y = np.asarray(labels, dtype=np.float64)
    logits = _logit(probabilities)
    mass = np.asarray(weights, dtype=np.float64)
    mass /= np.sum(mass)

    candidates: list[PolymarketRound11Calibration] = []

    def add(method: str, intercept: float, slope: float) -> None:
        calibrated = expit(intercept + slope * logits)
        payload = {
            "method": method,
            "intercept": float(intercept),
            "slope": float(slope),
            "oof_log_loss": _weighted_binary_log_loss(y, calibrated, mass),
        }
        candidates.append(
            PolymarketRound11Calibration(
                **payload,
                calibration_sha256=_canonical_sha256(payload),
            )
        )

    add("identity", 0.0, 1.0)

    def temperature_objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        slope = math.exp(float(theta[0]))
        calibrated_logits = slope * logits
        residual = expit(calibrated_logits) - y
        loss = float(
            np.sum(
                mass * (np.logaddexp(0.0, calibrated_logits) - y * calibrated_logits)
            )
        )
        gradient = np.asarray([np.sum(mass * residual * logits) * slope])
        return loss, gradient

    temperature = minimize(
        temperature_objective,
        np.zeros(1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-3.0, 3.0)],
    )
    if temperature.success:
        add("temperature", 0.0, math.exp(float(temperature.x[0])))

    def platt_objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(theta[0])
        slope = math.exp(float(theta[1]))
        calibrated_logits = intercept + slope * logits
        residual = expit(calibrated_logits) - y
        loss = float(
            np.sum(
                mass * (np.logaddexp(0.0, calibrated_logits) - y * calibrated_logits)
            )
        )
        gradient = np.asarray(
            [np.sum(mass * residual), np.sum(mass * residual * logits) * slope]
        )
        return loss, gradient

    platt = minimize(
        platt_objective,
        np.zeros(2),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-5.0, 5.0), (-3.0, 3.0)],
    )
    if platt.success:
        add("platt", float(platt.x[0]), math.exp(float(platt.x[1])))
    return tuple(candidates)


def _fit_direction_model(
    dataset: PolymarketRound11Dataset,
    split: PolymarketRound11Split,
    *,
    progress: ProgressCallback,
) -> tuple[
    PolymarketRound11Scaler,
    PolymarketRound11LinearHead,
    PolymarketRound11Calibration,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    folds = _training_folds(split.train_groups)
    candidate_folds: dict[float, list[dict[str, object]]] = {
        value: [] for value in POLYMARKET_ROUND11_L2_GRID
    }
    oof: dict[float, dict[str, list[np.ndarray]]] = {
        value: {"labels": [], "probabilities": [], "weights": []}
        for value in POLYMARKET_ROUND11_L2_GRID
    }
    for fold_index, (fit_groups, score_groups) in enumerate(folds):
        fit_mask = np.isin(dataset.event_start_ms, fit_groups)
        score_mask = np.isin(dataset.event_start_ms, score_groups)
        fit_weights = _condition_balanced_weights(dataset.condition_index, fit_mask)
        score_weights = _condition_balanced_weights(dataset.condition_index, score_mask)
        scaler = _fit_scaler(dataset.direction_features[fit_mask], fit_weights)
        fit_x = scaler.transform(dataset.direction_features[fit_mask])
        score_x = scaler.transform(dataset.direction_features[score_mask])
        fit_y = dataset.official_up[fit_mask].astype(np.float64)
        score_y = dataset.official_up[score_mask].astype(np.float64)
        fit_offset = _logit(dataset.market_prior_up[fit_mask])
        score_offset = _logit(dataset.market_prior_up[score_mask])
        for l2 in POLYMARKET_ROUND11_L2_GRID:
            head = _fit_binary_head(
                fit_x,
                fit_y,
                fit_weights,
                name="direction_residual",
                l2=l2,
                offset=fit_offset,
            )
            probability = head.predict_binary(score_x, offset=score_offset)
            candidate_folds[l2].append(
                {
                    "fold": fold_index,
                    "fit_group_count": len(fit_groups),
                    "score_group_count": len(score_groups),
                    "score_condition_count": int(
                        np.unique(dataset.condition_index[score_mask]).size
                    ),
                    "log_loss": _weighted_binary_log_loss(
                        score_y, probability, score_weights
                    ),
                    "brier_score": _weighted_brier(score_y, probability, score_weights),
                    "market_prior_log_loss": _weighted_binary_log_loss(
                        score_y, dataset.market_prior_up[score_mask], score_weights
                    ),
                    "model_sha256": head.model_sha256,
                }
            )
            oof[l2]["labels"].append(score_y)
            oof[l2]["probabilities"].append(probability)
            oof[l2]["weights"].append(score_weights)
        progress(
            "direction_cv",
            {"fold": fold_index + 1, "folds": len(folds)},
        )
    candidates: list[dict[str, object]] = []
    for l2 in POLYMARKET_ROUND11_L2_GRID:
        rows = candidate_folds[l2]
        candidates.append(
            {
                "l2": l2,
                "mean_log_loss": float(np.mean([row["log_loss"] for row in rows])),
                "mean_brier_score": float(
                    np.mean([row["brier_score"] for row in rows])
                ),
                "mean_market_prior_log_loss": float(
                    np.mean([row["market_prior_log_loss"] for row in rows])
                ),
                "folds": rows,
            }
        )
    selected_l2 = float(
        min(candidates, key=lambda row: (row["mean_log_loss"], -float(row["l2"])))["l2"]
    )
    selected_oof = oof[selected_l2]
    oof_labels = np.concatenate(selected_oof["labels"])
    oof_probabilities = np.concatenate(selected_oof["probabilities"])
    oof_weights = np.concatenate(selected_oof["weights"])
    calibration_candidates = _calibration_candidates(
        oof_labels, oof_probabilities, oof_weights
    )
    selected_calibration = min(
        calibration_candidates,
        key=lambda value: (
            value.oof_log_loss,
            ("identity", "temperature", "platt").index(value.method),
        ),
    )
    train_mask = np.isin(dataset.event_start_ms, split.train_groups)
    train_weights = _condition_balanced_weights(dataset.condition_index, train_mask)
    scaler = _fit_scaler(dataset.direction_features[train_mask], train_weights)
    head = _fit_binary_head(
        scaler.transform(dataset.direction_features[train_mask]),
        dataset.official_up[train_mask].astype(np.float64),
        train_weights,
        name="direction_residual",
        l2=selected_l2,
        offset=_logit(dataset.market_prior_up[train_mask]),
    )
    progress(
        "direction_final",
        {
            "l2": selected_l2,
            "calibration": selected_calibration.method,
            "model_sha256": head.model_sha256,
        },
    )
    return (
        scaler,
        head,
        selected_calibration,
        candidates,
        [
            {
                **candidate.identity_payload(),
                "calibration_sha256": candidate.calibration_sha256,
            }
            for candidate in calibration_candidates
        ],
    )


def _fit_execution_models(
    dataset: PolymarketRound11Dataset,
    split: PolymarketRound11Split,
    *,
    progress: ProgressCallback,
) -> tuple[
    PolymarketRound11Scaler,
    dict[str, PolymarketRound11LinearHead],
    list[dict[str, object]],
]:
    feature_width = len(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES)
    features = dataset.execution_features.reshape(-1, feature_width)
    conditions = np.repeat(dataset.condition_index, 2)
    groups = np.repeat(dataset.event_start_ms, 2)
    available = dataset.available.reshape(-1)
    observable = dataset.observable.reshape(-1)
    filled = dataset.entry_filled.reshape(-1)
    costs = dataset.entry_cost_quote.reshape(-1)
    folds = _training_folds(split.train_groups)
    objectives = (
        ("entry_observable", "binary", available, observable.astype(np.float64)),
        ("entry_fill", "binary", observable, filled.astype(np.float64)),
        ("entry_cost_mean", "huber_mean", filled, costs),
        ("entry_cost_q90", "pinball_q90", filled, costs),
    )
    candidate_rows: list[dict[str, object]] = []
    selected_l2: dict[str, float] = {}
    for name, objective_name, target_available, target in objectives:
        by_l2: dict[float, list[dict[str, object]]] = {
            value: [] for value in POLYMARKET_ROUND11_L2_GRID
        }
        for fold_index, (fit_groups, score_groups) in enumerate(folds):
            fit_base = np.isin(groups, fit_groups) & target_available
            score_base = np.isin(groups, score_groups) & target_available
            fit_scaler_mask = np.isin(groups, fit_groups) & available
            scaler_weights = _condition_balanced_weights(conditions, fit_scaler_mask)
            scaler = _fit_scaler(features[fit_scaler_mask], scaler_weights)
            fit_weights = _condition_balanced_weights(conditions, fit_base)
            score_weights = _condition_balanced_weights(conditions, score_base)
            fit_x = scaler.transform(features[fit_base])
            score_x = scaler.transform(features[score_base])
            for l2 in POLYMARKET_ROUND11_L2_GRID:
                if objective_name == "binary":
                    head = _fit_binary_head(
                        fit_x,
                        target[fit_base],
                        fit_weights,
                        name=name,
                        l2=l2,
                    )
                    prediction = head.predict_binary(score_x)
                    loss = _weighted_binary_log_loss(
                        target[score_base], prediction, score_weights
                    )
                else:
                    head = _fit_value_head(
                        fit_x,
                        target[fit_base],
                        fit_weights,
                        name=name,
                        objective_name=objective_name,
                        l2=l2,
                    )
                    prediction = head.predict_value(score_x)
                    loss = (
                        _huber_loss(target[score_base], prediction, score_weights)
                        if objective_name == "huber_mean"
                        else _pinball_loss(
                            target[score_base], prediction, score_weights
                        )
                    )
                by_l2[l2].append(
                    {
                        "fold": fold_index,
                        "fit_rows": int(np.count_nonzero(fit_base)),
                        "score_rows": int(np.count_nonzero(score_base)),
                        "validation_loss": loss,
                        "model_sha256": head.model_sha256,
                    }
                )
            progress(
                "execution_cv",
                {
                    "head": name,
                    "fold": fold_index + 1,
                    "folds": len(folds),
                },
            )
        summaries = []
        for l2 in POLYMARKET_ROUND11_L2_GRID:
            summary = {
                "head": name,
                "objective": objective_name,
                "l2": l2,
                "mean_validation_loss": float(
                    np.mean([row["validation_loss"] for row in by_l2[l2]])
                ),
                "folds": by_l2[l2],
            }
            summaries.append(summary)
            candidate_rows.append(summary)
        selected_l2[name] = float(
            min(
                summaries,
                key=lambda row: (row["mean_validation_loss"], -float(row["l2"])),
            )["l2"]
        )
    train_available = np.isin(groups, split.train_groups) & available
    scaler_weights = _condition_balanced_weights(conditions, train_available)
    scaler = _fit_scaler(features[train_available], scaler_weights)
    heads: dict[str, PolymarketRound11LinearHead] = {}
    for name, objective_name, target_available, target in objectives:
        mask = np.isin(groups, split.train_groups) & target_available
        weights = _condition_balanced_weights(conditions, mask)
        transformed = scaler.transform(features[mask])
        if objective_name == "binary":
            head = _fit_binary_head(
                transformed,
                target[mask],
                weights,
                name=name,
                l2=selected_l2[name],
            )
        else:
            head = _fit_value_head(
                transformed,
                target[mask],
                weights,
                name=name,
                objective_name=objective_name,
                l2=selected_l2[name],
            )
        heads[name] = head
        progress(
            "execution_final",
            {"head": name, "l2": selected_l2[name], "model_sha256": head.model_sha256},
        )
    return scaler, heads, candidate_rows


def _reliability_table(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> list[dict[str, object]]:
    y = np.asarray(labels, dtype=np.float64)
    prediction = _finite_probability(probabilities)
    mass = np.asarray(weights, dtype=np.float64)
    bins = np.minimum(9, np.floor(prediction * 10).astype(np.int8))
    result: list[dict[str, object]] = []
    for index in range(10):
        selected = bins == index
        if not np.any(selected):
            result.append(
                {
                    "bin": index,
                    "rows": 0,
                    "condition_weight": 0.0,
                    "mean_probability": None,
                    "observed_frequency": None,
                }
            )
            continue
        selected_mass = mass[selected]
        total = float(np.sum(selected_mass))
        result.append(
            {
                "bin": index,
                "rows": int(np.count_nonzero(selected)),
                "condition_weight": total,
                "mean_probability": float(
                    np.sum(selected_mass * prediction[selected]) / total
                ),
                "observed_frequency": float(
                    np.sum(selected_mass * y[selected]) / total
                ),
            }
        )
    return result


def _bootstrap_group_mean(group_utility: np.ndarray) -> dict[str, float]:
    values = np.asarray(group_utility, dtype=np.float64)
    if values.ndim != 1 or values.size < 10 or not np.all(np.isfinite(values)):
        raise ValueError("Polymarket Round 11 bootstrap input is invalid")
    generator = np.random.default_rng(POLYMARKET_ROUND11_BOOTSTRAP_SEED)
    block_length = 3
    samples = np.empty(POLYMARKET_ROUND11_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(samples.size):
        selected: list[int] = []
        while len(selected) < values.size:
            start = int(generator.integers(0, values.size))
            selected.extend(
                (start + offset) % values.size for offset in range(block_length)
            )
        samples[sample_index] = float(np.mean(values[selected[: values.size]]))
    lower, median, upper = np.quantile(samples, (0.025, 0.50, 0.975))
    return {
        "samples": int(samples.size),
        "block_length_groups": block_length,
        "lower_95_mean_group_utility_quote": float(lower),
        "median_mean_group_utility_quote": float(median),
        "upper_95_mean_group_utility_quote": float(upper),
        "bootstrap_samples_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
    }


def _policy_metrics(
    dataset: PolymarketRound11Dataset,
    split: PolymarketRound11Split,
    *,
    probability_up: np.ndarray,
    observable_probability: np.ndarray,
    fill_probability: np.ndarray,
    upper_entry_cost: np.ndarray,
    margin_quote: float,
    minimum_remaining_seconds: float,
    include_decisions: bool,
) -> dict[str, object]:
    p_up = _finite_probability(probability_up)
    p_observable = _finite_probability(observable_probability)
    p_fill = _finite_probability(fill_probability)
    upper_cost = np.asarray(upper_entry_cost, dtype=np.float64)
    if (
        p_up.shape != (dataset.rows,)
        or p_observable.shape != (dataset.rows, 2)
        or p_fill.shape != (dataset.rows, 2)
        or upper_cost.shape != (dataset.rows, 2)
        or not np.all(np.isfinite(upper_cost))
        or margin_quote not in POLYMARKET_ROUND11_MARGIN_GRID
        or minimum_remaining_seconds not in POLYMARKET_ROUND11_MINIMUM_REMAINING_GRID
    ):
        raise ValueError("Polymarket Round 11 policy input is invalid")
    quantity_index = POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES.index(
        "minimum_order_quantity"
    )
    quantity = dataset.execution_features[:, :, quantity_index]
    p_win = np.column_stack((1.0 - p_up, p_up))
    score = (
        p_observable * p_fill * (quantity * p_win - upper_cost)
        - (1.0 - p_observable) * dataset.maximum_entry_loss_quote
    )
    intrinsic = quantity * p_win - upper_cost
    validation_mask = np.isin(dataset.event_start_ms, split.validation_groups)
    validation_indices = np.flatnonzero(validation_mask)
    validation_conditions = np.unique(dataset.condition_index[validation_mask])
    condition_position = {
        int(condition): index
        for index, condition in enumerate(validation_conditions.tolist())
    }
    state = np.zeros(validation_conditions.size, dtype=np.int8)
    last_attempt_ns = np.zeros(validation_conditions.size, dtype=np.int64)
    utility = np.zeros(validation_conditions.size, dtype=np.float64)
    selected_outcome = np.full(validation_conditions.size, -1, dtype=np.int8)
    selected_row = np.full(validation_conditions.size, -1, dtype=np.int64)
    attempts = 0
    definite_no_fills = 0
    pre_submit_unavailable = 0
    unknowns = 0
    filled = 0
    wins = 0
    lock_seconds = 0.0
    capital_locked = 0.0
    decisions: list[dict[str, object]] = []
    for row_index in validation_indices:
        position = condition_position[int(dataset.condition_index[row_index])]
        if state[position] != 0:
            continue
        remaining = float(dataset.remaining_seconds[row_index])
        if remaining < minimum_remaining_seconds or remaining <= 30.0:
            continue
        decision_time = int(dataset.decision_monotonic_ns[row_index])
        if (
            last_attempt_ns[position] > 0
            and decision_time - int(last_attempt_ns[position])
            < POLYMARKET_ROUND11_RETRY_NS
        ):
            continue
        passes = (score[row_index] > margin_quote) & (
            intrinsic[row_index] > margin_quote
        )
        if not np.any(passes):
            continue
        candidates = np.flatnonzero(passes)
        if candidates.size == 2:
            difference = float(score[row_index, 1] - score[row_index, 0])
            if abs(difference) <= 1e-12:
                continue
            outcome_index = 1 if difference > 0 else 0
        else:
            outcome_index = int(candidates[0])
        last_attempt_ns[position] = decision_time
        reason = dataset.terminal_reasons[
            int(dataset.terminal_reason_code[row_index, outcome_index])
        ]
        base_decision = {
            "condition_id": dataset.condition_ids[
                int(dataset.condition_index[row_index])
            ],
            "asset": _ASSETS[int(dataset.asset_index[row_index])],
            "event_start_ms": int(dataset.event_start_ms[row_index]),
            "decision_monotonic_ns": decision_time,
            "remaining_seconds": remaining,
            "outcome": _OUTCOMES[outcome_index],
            "p_win": float(p_win[row_index, outcome_index]),
            "p_observable": float(p_observable[row_index, outcome_index]),
            "p_fill": float(p_fill[row_index, outcome_index]),
            "upper_entry_cost_quote": float(upper_cost[row_index, outcome_index]),
            "score_quote": float(score[row_index, outcome_index]),
            "terminal_reason": reason,
        }
        if not dataset.available[row_index, outcome_index]:
            pre_submit_unavailable += 1
            if include_decisions:
                decisions.append({**base_decision, "result": "pre_submit_unavailable"})
            continue
        attempts += 1
        if dataset.unknown_entry[row_index, outcome_index]:
            unknowns += 1
            state[position] = 2
            utility[position] = float(
                dataset.realized_hold_utility_quote[row_index, outcome_index]
            )
            selected_outcome[position] = outcome_index
            selected_row[position] = row_index
            if include_decisions:
                decisions.append(
                    {
                        **base_decision,
                        "result": "unknown_blocked",
                        "utility_quote": float(utility[position]),
                    }
                )
            continue
        if not dataset.entry_filled[row_index, outcome_index]:
            definite_no_fills += 1
            if include_decisions:
                decisions.append({**base_decision, "result": "definite_no_fill"})
            continue
        filled += 1
        state[position] = 1
        selected_outcome[position] = outcome_index
        selected_row[position] = row_index
        utility[position] = float(
            dataset.realized_hold_utility_quote[row_index, outcome_index]
        )
        won = bool(dataset.official_up[row_index]) == (outcome_index == 1)
        wins += int(won)
        exact_cost = float(dataset.entry_cost_quote[row_index, outcome_index])
        capital_locked += exact_cost
        lock_seconds += exact_cost * remaining
        if include_decisions:
            decisions.append(
                {
                    **base_decision,
                    "result": "filled_hold_to_resolution",
                    "won": won,
                    "entry_cost_quote": exact_cost,
                    "utility_quote": float(utility[position]),
                }
            )
    condition_asset = np.empty(validation_conditions.size, dtype=np.int8)
    condition_group = np.empty(validation_conditions.size, dtype=np.int64)
    condition_records: list[dict[str, object]] = []
    for condition, position in condition_position.items():
        first = int(
            np.flatnonzero(validation_mask & (dataset.condition_index == condition))[0]
        )
        condition_asset[position] = dataset.asset_index[first]
        condition_group[position] = dataset.event_start_ms[first]
        condition_records.append(
            {
                "condition_id": dataset.condition_ids[condition],
                "asset": _ASSETS[int(condition_asset[position])],
                "event_start_ms": int(condition_group[position]),
                "state": ("abstained", "filled", "unknown_blocked")[
                    int(state[position])
                ],
                "outcome": (
                    None
                    if selected_outcome[position] < 0
                    else _OUTCOMES[int(selected_outcome[position])]
                ),
                "utility_quote": float(utility[position]),
            }
        )
    group_utility = np.asarray(
        [
            float(np.sum(utility[condition_group == group]))
            for group in split.validation_groups
        ],
        dtype=np.float64,
    )
    equity = np.cumsum(group_utility)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdown = running_peak[1:] - equity
    per_asset: dict[str, dict[str, object]] = {}
    for asset_index, asset in enumerate(_ASSETS):
        selected = condition_asset == asset_index
        filled_asset = selected & (state == 1)
        per_asset[asset] = {
            "conditions": int(np.count_nonzero(selected)),
            "filled_conditions": int(np.count_nonzero(filled_asset)),
            "utility_quote": float(np.sum(utility[selected])),
            "mean_condition_utility_quote": float(np.mean(utility[selected])),
        }
    bootstrap = _bootstrap_group_mean(group_utility)
    total_utility = float(np.sum(utility))
    mean_utility = float(np.mean(utility))
    median_utility = float(np.median(utility))
    gate_reasons: list[str] = []
    if filled < 30:
        gate_reasons.append("fewer_than_30_filled_conditions")
    for asset in _ASSETS:
        if int(per_asset[asset]["filled_conditions"]) < 5:
            gate_reasons.append(f"fewer_than_5_filled_{asset}")
        if float(per_asset[asset]["utility_quote"]) <= 0:
            gate_reasons.append(f"nonpositive_utility_{asset}")
    if unknowns != 0:
        gate_reasons.append("selected_unknown_entry")
    if total_utility <= 0:
        gate_reasons.append("nonpositive_total_utility")
    if mean_utility <= 0:
        gate_reasons.append("nonpositive_mean_condition_utility")
    if median_utility <= 0:
        gate_reasons.append("nonpositive_median_condition_utility")
    if float(bootstrap["lower_95_mean_group_utility_quote"]) <= 0:
        gate_reasons.append("nonpositive_bootstrap_lower_mean_group_utility")
    result: dict[str, object] = {
        "margin_quote": margin_quote,
        "minimum_remaining_seconds": minimum_remaining_seconds,
        "validation_groups": len(split.validation_groups),
        "validation_conditions": int(validation_conditions.size),
        "attempts": attempts,
        "pre_submit_unavailable": pre_submit_unavailable,
        "definite_no_fills": definite_no_fills,
        "unknown_entries": unknowns,
        "filled_conditions": filled,
        "wins": wins,
        "losses": filled - wins,
        "win_rate": None if filled == 0 else wins / filled,
        "total_utility_quote": total_utility,
        "mean_condition_utility_quote": mean_utility,
        "median_condition_utility_quote": median_utility,
        "maximum_drawdown_quote": float(np.max(drawdown)) if drawdown.size else 0.0,
        "capital_locked_quote": capital_locked,
        "capital_time_quote_seconds": lock_seconds,
        "per_asset": per_asset,
        "group_utility_quote": [float(value) for value in group_utility],
        "bootstrap": bootstrap,
        "gate_passed_before_market_prior_comparison": not gate_reasons,
        "gate_reasons": gate_reasons,
    }
    if include_decisions:
        result["condition_results"] = sorted(
            condition_records,
            key=lambda value: (value["event_start_ms"], value["asset"]),
        )
        result["decision_results"] = decisions
    return result


def _direction_metrics(
    dataset: PolymarketRound11Dataset,
    mask: np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    weights = _condition_balanced_weights(dataset.condition_index, mask)
    target = dataset.official_up[mask].astype(np.float64)
    prediction = probability[mask]
    prior = dataset.market_prior_up[mask]
    pooled = {
        "rows": int(np.count_nonzero(mask)),
        "conditions": int(np.unique(dataset.condition_index[mask]).size),
        "up_prevalence": float(np.sum(weights * target)),
        "model_log_loss": _weighted_binary_log_loss(target, prediction, weights),
        "market_prior_log_loss": _weighted_binary_log_loss(target, prior, weights),
        "model_brier_score": _weighted_brier(target, prediction, weights),
        "market_prior_brier_score": _weighted_brier(target, prior, weights),
        "reliability": _reliability_table(target, prediction, weights),
    }
    per_asset: dict[str, dict[str, object]] = {}
    for index, asset in enumerate(_ASSETS):
        selected = mask & (dataset.asset_index == index)
        selected_weights = _condition_balanced_weights(
            dataset.condition_index, selected
        )
        selected_target = dataset.official_up[selected].astype(np.float64)
        per_asset[asset] = {
            "rows": int(np.count_nonzero(selected)),
            "conditions": int(np.unique(dataset.condition_index[selected]).size),
            "model_log_loss": _weighted_binary_log_loss(
                selected_target, probability[selected], selected_weights
            ),
            "market_prior_log_loss": _weighted_binary_log_loss(
                selected_target, dataset.market_prior_up[selected], selected_weights
            ),
            "model_brier_score": _weighted_brier(
                selected_target, probability[selected], selected_weights
            ),
            "market_prior_brier_score": _weighted_brier(
                selected_target, dataset.market_prior_up[selected], selected_weights
            ),
        }
    return {"pooled": pooled, "per_asset": per_asset}


def _execution_metrics(
    dataset: PolymarketRound11Dataset,
    split: PolymarketRound11Split,
    *,
    observable_probability: np.ndarray,
    fill_probability: np.ndarray,
    cost_mean: np.ndarray,
    cost_q90: np.ndarray,
) -> dict[str, object]:
    conditions = np.repeat(dataset.condition_index, 2)
    groups = np.repeat(dataset.event_start_ms, 2)
    validation = np.isin(groups, split.validation_groups)
    available = dataset.available.reshape(-1)
    observable = dataset.observable.reshape(-1)
    filled = dataset.entry_filled.reshape(-1)
    exact_cost = dataset.entry_cost_quote.reshape(-1)
    p_observable = observable_probability.reshape(-1)
    p_fill = fill_probability.reshape(-1)
    mean = cost_mean.reshape(-1)
    q90 = cost_q90.reshape(-1)
    observable_mask = validation & available
    fill_mask = validation & observable
    cost_mask = validation & filled
    observable_weights = _condition_balanced_weights(conditions, observable_mask)
    fill_weights = _condition_balanced_weights(conditions, fill_mask)
    cost_weights = _condition_balanced_weights(conditions, cost_mask)
    observed_cost = exact_cost[cost_mask]
    return {
        "observability": {
            "rows": int(np.count_nonzero(observable_mask)),
            "prevalence": float(
                np.sum(observable_weights * observable[observable_mask])
            ),
            "log_loss": _weighted_binary_log_loss(
                observable[observable_mask],
                p_observable[observable_mask],
                observable_weights,
            ),
            "brier_score": _weighted_brier(
                observable[observable_mask],
                p_observable[observable_mask],
                observable_weights,
            ),
        },
        "fill_given_observable": {
            "rows": int(np.count_nonzero(fill_mask)),
            "prevalence": float(np.sum(fill_weights * filled[fill_mask])),
            "log_loss": _weighted_binary_log_loss(
                filled[fill_mask], p_fill[fill_mask], fill_weights
            ),
            "brier_score": _weighted_brier(
                filled[fill_mask], p_fill[fill_mask], fill_weights
            ),
        },
        "entry_cost_given_fill": {
            "rows": int(np.count_nonzero(cost_mask)),
            "weighted_mean_absolute_error_quote": float(
                np.sum(cost_weights * np.abs(mean[cost_mask] - observed_cost))
            ),
            "weighted_q90_pinball_loss_quote": _pinball_loss(
                observed_cost, q90[cost_mask], cost_weights
            ),
            "q90_empirical_coverage": float(
                np.sum(cost_weights * (observed_cost <= q90[cost_mask]))
            ),
        },
    }


def _model_predictions(
    dataset: PolymarketRound11Dataset,
    *,
    direction_scaler: PolymarketRound11Scaler,
    direction_head: PolymarketRound11LinearHead,
    calibration: PolymarketRound11Calibration,
    execution_scaler: PolymarketRound11Scaler,
    execution_heads: Mapping[str, PolymarketRound11LinearHead],
) -> dict[str, np.ndarray]:
    direction_x = direction_scaler.transform(dataset.direction_features)
    raw_direction = direction_head.predict_binary(
        direction_x, offset=_logit(dataset.market_prior_up)
    )
    probability_up = calibration.predict(raw_direction)
    execution_x = execution_scaler.transform(
        dataset.execution_features.reshape(
            -1, len(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES)
        )
    )
    observable = execution_heads["entry_observable"].predict_binary(execution_x)
    fill = execution_heads["entry_fill"].predict_binary(execution_x)
    cost_mean = execution_heads["entry_cost_mean"].predict_value(execution_x)
    cost_q90 = execution_heads["entry_cost_q90"].predict_value(execution_x)
    shape = (dataset.rows, 2)
    maximum = dataset.maximum_entry_loss_quote
    current = dataset.current_top_ask_cost_quote
    cost_mean = np.clip(cost_mean.reshape(shape), 0.0, maximum)
    cost_q90 = np.clip(cost_q90.reshape(shape), 0.0, maximum)
    upper_cost = np.maximum(current, cost_q90)
    predictions = {
        "probability_up": probability_up,
        "observable_probability": observable.reshape(shape),
        "fill_probability": fill.reshape(shape),
        "cost_mean_quote": cost_mean,
        "cost_q90_quote": cost_q90,
        "upper_entry_cost_quote": upper_cost,
    }
    if any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise ValueError("Polymarket Round 11 model prediction is nonfinite")
    return predictions


def _add_market_prior_comparison(
    model: dict[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    result = dict(model)
    reasons = list(result["gate_reasons"])
    model_lower = float(
        result["bootstrap"]["lower_95_mean_group_utility_quote"]  # type: ignore[index]
    )
    baseline_lower = float(
        baseline["bootstrap"]["lower_95_mean_group_utility_quote"]  # type: ignore[index]
    )
    model_total = float(result["total_utility_quote"])
    baseline_total = float(baseline["total_utility_quote"])
    if model_lower <= baseline_lower:
        reasons.append("bootstrap_lower_not_better_than_market_prior_policy")
    if model_total <= baseline_total:
        reasons.append("total_utility_not_better_than_market_prior_policy")
    result["market_prior_comparison"] = {
        "model_minus_prior_total_utility_quote": model_total - baseline_total,
        "model_minus_prior_bootstrap_lower_mean_group_utility_quote": model_lower
        - baseline_lower,
    }
    result["gate_reasons"] = reasons
    result["gate_passed"] = not reasons
    return result


def fit_round11_development(
    dataset: PolymarketRound11Dataset,
    split: PolymarketRound11Split,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit once on development evidence and return hash-bound report/artifact."""

    source = dataset.validated()
    frozen_split = split.validated()
    emit = progress or (lambda _stage, _payload: None)
    (
        direction_scaler,
        direction_head,
        calibration,
        direction_candidates,
        calibration_candidates,
    ) = _fit_direction_model(source, frozen_split, progress=emit)
    execution_scaler, execution_heads, execution_candidates = _fit_execution_models(
        source, frozen_split, progress=emit
    )
    predictions = _model_predictions(
        source,
        direction_scaler=direction_scaler,
        direction_head=direction_head,
        calibration=calibration,
        execution_scaler=execution_scaler,
        execution_heads=execution_heads,
    )
    validation_mask = np.isin(source.event_start_ms, frozen_split.validation_groups)
    direction_metrics = _direction_metrics(
        source,
        validation_mask,
        predictions["probability_up"],
    )
    execution_metrics = _execution_metrics(
        source,
        frozen_split,
        observable_probability=predictions["observable_probability"],
        fill_probability=predictions["fill_probability"],
        cost_mean=predictions["cost_mean_quote"],
        cost_q90=predictions["cost_q90_quote"],
    )
    policy_candidates: list[dict[str, object]] = []
    for minimum_remaining in POLYMARKET_ROUND11_MINIMUM_REMAINING_GRID:
        for margin in POLYMARKET_ROUND11_MARGIN_GRID:
            model = _policy_metrics(
                source,
                frozen_split,
                probability_up=predictions["probability_up"],
                observable_probability=predictions["observable_probability"],
                fill_probability=predictions["fill_probability"],
                upper_entry_cost=predictions["upper_entry_cost_quote"],
                margin_quote=margin,
                minimum_remaining_seconds=minimum_remaining,
                include_decisions=False,
            )
            baseline = _policy_metrics(
                source,
                frozen_split,
                probability_up=source.market_prior_up,
                observable_probability=predictions["observable_probability"],
                fill_probability=predictions["fill_probability"],
                upper_entry_cost=predictions["upper_entry_cost_quote"],
                margin_quote=margin,
                minimum_remaining_seconds=minimum_remaining,
                include_decisions=False,
            )
            compared = _add_market_prior_comparison(model, baseline)
            compared["market_prior_policy"] = baseline
            compared["candidate_sha256"] = _canonical_sha256(compared)
            policy_candidates.append(compared)
        emit(
            "policy_grid",
            {
                "minimum_remaining_seconds": minimum_remaining,
                "candidates_complete": len(policy_candidates),
            },
        )
    passing = [row for row in policy_candidates if row["gate_passed"]]
    selection_pool = passing or policy_candidates
    selected_summary = max(
        selection_pool,
        key=lambda row: (
            float(row["bootstrap"]["lower_95_mean_group_utility_quote"]),  # type: ignore[index]
            float(row["total_utility_quote"]),
            int(row["filled_conditions"]),
            -float(row["maximum_drawdown_quote"]),
            -float(row["margin_quote"]),
            float(row["minimum_remaining_seconds"]),
        ),
    )
    selected_model = _policy_metrics(
        source,
        frozen_split,
        probability_up=predictions["probability_up"],
        observable_probability=predictions["observable_probability"],
        fill_probability=predictions["fill_probability"],
        upper_entry_cost=predictions["upper_entry_cost_quote"],
        margin_quote=float(selected_summary["margin_quote"]),
        minimum_remaining_seconds=float(selected_summary["minimum_remaining_seconds"]),
        include_decisions=True,
    )
    selected_baseline = _policy_metrics(
        source,
        frozen_split,
        probability_up=source.market_prior_up,
        observable_probability=predictions["observable_probability"],
        fill_probability=predictions["fill_probability"],
        upper_entry_cost=predictions["upper_entry_cost_quote"],
        margin_quote=float(selected_summary["margin_quote"]),
        minimum_remaining_seconds=float(selected_summary["minimum_remaining_seconds"]),
        include_decisions=True,
    )
    selected_policy = _add_market_prior_comparison(selected_model, selected_baseline)
    selected_policy["market_prior_policy"] = selected_baseline
    selected_policy["selection_role"] = (
        "frozen_development_candidate"
        if selected_policy["gate_passed"]
        else "diagnostic_best_failed_candidate"
    )
    artifact_without_hash: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND11_MODEL_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND11_CONTRACT_SHA256,
        "dataset_sha256": source.dataset_sha256,
        "split_sha256": frozen_split.split_sha256,
        "direction_feature_names": list(POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES),
        "execution_feature_names": list(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES),
        "direction_scaler": asdict(direction_scaler),
        "direction_head": asdict(direction_head),
        "direction_calibration": asdict(calibration),
        "execution_scaler": asdict(execution_scaler),
        "execution_heads": {
            name: asdict(head) for name, head in sorted(execution_heads.items())
        },
        "selected_policy": {
            "margin_quote": selected_policy["margin_quote"],
            "minimum_remaining_seconds": selected_policy["minimum_remaining_seconds"],
            "retry_ns": POLYMARKET_ROUND11_RETRY_NS,
            "development_gate_passed": selected_policy["gate_passed"],
        },
        "cpu_reference": "numpy_float64_scipy_fit",
        "onnx_exported": False,
    }
    artifact_sha256 = _canonical_sha256(artifact_without_hash)
    artifact = {**artifact_without_hash, "artifact_sha256": artifact_sha256}
    terminal_counts = {
        reason: int(np.count_nonzero(source.terminal_reason_code == index))
        for index, reason in enumerate(source.terminal_reasons)
    }
    event_start = int(np.min(source.event_start_ms))
    event_end = int(np.max(source.event_start_ms)) + 300_000
    report_without_hash: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND11_REPORT_SCHEMA_VERSION,
        "round": 11,
        "contract_sha256": POLYMARKET_ROUND11_CONTRACT_SHA256,
        "source_pipeline_report_sha256": source.pipeline_report_sha256,
        "dataset_sha256": source.dataset_sha256,
        "split_sha256": frozen_split.split_sha256,
        "artifact_sha256": artifact_sha256,
        "evidence_role": "round9_corpus_reused_for_round11_development_only",
        "utc_span": {
            "start": datetime.fromtimestamp(
                event_start / 1000, tz=timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(
                event_end / 1000, tz=timezone.utc
            ).isoformat(),
        },
        "dataset": {
            "paired_decision_rows": source.rows,
            "action_rows": source.rows * 2,
            "event_groups": len(source.groups),
            "conditions": len(source.condition_ids),
            "assets": list(_ASSETS),
            "entry_filled": int(np.count_nonzero(source.entry_filled)),
            "unknown_entry": int(np.count_nonzero(source.unknown_entry)),
            "pre_submit_unavailable": int(np.count_nonzero(~source.available)),
            "terminal_reason_counts": dict(sorted(terminal_counts.items())),
            "independent_unit": "resolved_condition",
        },
        "split": {
            **frozen_split.identity_payload(),
            "split_sha256": frozen_split.split_sha256,
            "train_conditions": int(
                np.unique(
                    source.condition_index[
                        np.isin(source.event_start_ms, frozen_split.train_groups)
                    ]
                ).size
            ),
            "validation_conditions": int(
                np.unique(source.condition_index[validation_mask]).size
            ),
        },
        "direction_candidates": direction_candidates,
        "calibration_candidates": calibration_candidates,
        "selected_direction": {
            "scaler_sha256": direction_scaler.scaler_sha256,
            "model_sha256": direction_head.model_sha256,
            "l2": direction_head.l2,
            "calibration_sha256": calibration.calibration_sha256,
            "calibration_method": calibration.method,
        },
        "direction_validation": direction_metrics,
        "execution_candidates": execution_candidates,
        "selected_execution": {
            "scaler_sha256": execution_scaler.scaler_sha256,
            "heads": {
                name: {"l2": head.l2, "model_sha256": head.model_sha256}
                for name, head in sorted(execution_heads.items())
            },
        },
        "execution_validation": execution_metrics,
        "policy_candidates": policy_candidates,
        "selected_policy": selected_policy,
        "development_passed": bool(selected_policy["gate_passed"]),
        "nonlinear_challenger_authorized": False,
        "ai_treatment_authorized": False,
        "confirmation_capture_authorized": bool(selected_policy["gate_passed"]),
        "settlement_overhead_measured": False,
        "authenticated_lifecycle_proven": False,
        "profitability_claim": False,
        "roi_claim": False,
        "drawdown_claim": False,
        "paper_authority": False,
        "trading_authority": False,
    }
    report_sha256 = _canonical_sha256(report_without_hash)
    report = {**report_without_hash, "report_sha256": report_sha256}
    emit(
        "complete",
        {
            "development_passed": report["development_passed"],
            "report_sha256": report_sha256,
            "artifact_sha256": artifact_sha256,
        },
    )
    return report, artifact


__all__ = [
    "POLYMARKET_ROUND11_CONTRACT_SHA256",
    "POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES",
    "POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES",
    "PolymarketRound11Dataset",
    "PolymarketRound11Split",
    "build_round11_development_split",
    "fit_round11_development",
    "load_round11_development_dataset",
]
