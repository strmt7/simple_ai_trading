"""Round 10 development-only Polymarket observability/utility hurdle baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import re
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES
from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_ROUND10_CONTRACT_SHA256 = (
    "91cdc3064a88e4d6c744a2e2333d98bbd9ad382f3317def21e02b72983964d9d"
)
POLYMARKET_ROUND10_DATASET_SCHEMA_VERSION = (
    "polymarket-round10-development-hurdle-dataset-v1"
)
POLYMARKET_ROUND10_MODEL_SCHEMA_VERSION = (
    "polymarket-round10-transparent-hurdle-baseline-v1"
)
POLYMARKET_ROUND10_REPORT_SCHEMA_VERSION = (
    "polymarket-round10-development-hurdle-report-v1"
)
POLYMARKET_ROUND10_L2_GRID = (0.01, 0.1, 1.0, 10.0)
POLYMARKET_ROUND10_SCORE_MARGIN_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
POLYMARKET_ROUND10_STANDARDIZED_CLIP = 12.0
POLYMARKET_ROUND10_MAX_RELEASE_DELAY_NS = 3_500_000_000
POLYMARKET_ROUND10_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND10_BOOTSTRAP_SEED = 10010
POLYMARKET_ROUND10_DEVELOPMENT_TRAIN_FRACTION = 0.70
POLYMARKET_ROUND10_UTILITY_QUANTILE = 0.10
POLYMARKET_ROUND10_PINBALL_SMOOTHING = 0.05
POLYMARKET_ROUND10_HUBER_DELTA = 1.0

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ASSETS = ("BTC", "ETH", "SOL")
_OUTCOMES = ("Down", "Up")
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
_STATIC_FEATURE_NAMES = (
    "active_tick_size",
    "minimum_order_quantity",
    "fees_enabled",
    "fee_rate",
    "fee_exponent",
    "crypto_taker_delay_enabled",
    "unknown_entry_maximum_loss_quote",
)
POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES = (
    *POLYMARKET_ACTION_FEATURE_NAMES,
    *_STATIC_FEATURE_NAMES,
)
_FEATURE_COUNT = len(POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES)

ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def polymarket_maximum_entry_loss(
    *,
    tick_size: Decimal,
    quantity: Decimal,
    fees_enabled: bool,
    fee_rate: Decimal,
    fee_exponent: int,
) -> Decimal:
    if (
        tick_size <= 0
        or tick_size >= 1
        or quantity <= 0
        or fee_rate < 0
        or fee_rate > 1
        or fee_exponent <= 0
    ):
        raise ValueError("Round 10 maximum-loss inputs are invalid")
    gross = quantity * (Decimal("1") - tick_size)
    fee_bound = Decimal("0")
    if fees_enabled and fee_rate > 0:
        fee_bound = quantity * fee_rate * (Decimal("0.25") ** fee_exponent)
        fee_bound = fee_bound.quantize(Decimal("0.00001"), rounding=ROUND_CEILING)
    result = gross + fee_bound
    if not result.is_finite() or result <= 0:
        raise ValueError("Round 10 maximum entry loss is invalid")
    return result


@dataclass(frozen=True)
class PolymarketHurdleDataset:
    """Compact numeric development dataset with no confirmation authority."""

    pipeline_report_sha256: str
    dataset_sha256: str
    feature_names: tuple[str, ...]
    condition_ids: tuple[str, ...]
    features: np.ndarray
    action_feature_sha256: np.ndarray
    condition_index: np.ndarray
    asset_index: np.ndarray
    outcome_up: np.ndarray
    event_start_ms: np.ndarray
    decision_monotonic_ns: np.ndarray
    release_monotonic_ns: np.ndarray
    official_up: np.ndarray
    observable: np.ndarray
    entry_filled: np.ndarray
    exit_filled: np.ndarray
    complete_net_quote: np.ndarray
    stress_utility_quote: np.ndarray
    maximum_entry_loss_quote: np.ndarray
    unknown_entry: np.ndarray
    failed_exit: np.ndarray
    terminal_reason_code: np.ndarray
    terminal_reasons: tuple[str, ...]
    excluded_pre_submit_count: int

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def groups(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(self.event_start_ms))

    def validated(self) -> PolymarketHurdleDataset:
        rows = self.rows
        one_dimensional = (
            self.action_feature_sha256,
            self.condition_index,
            self.asset_index,
            self.outcome_up,
            self.event_start_ms,
            self.decision_monotonic_ns,
            self.release_monotonic_ns,
            self.official_up,
            self.observable,
            self.entry_filled,
            self.exit_filled,
            self.complete_net_quote,
            self.stress_utility_quote,
            self.maximum_entry_loss_quote,
            self.unknown_entry,
            self.failed_exit,
            self.terminal_reason_code,
        )
        if (
            not _SHA256.fullmatch(self.pipeline_report_sha256)
            or not _SHA256.fullmatch(self.dataset_sha256)
            or self.feature_names != POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES
            or self.features.shape != (rows, _FEATURE_COUNT)
            or rows <= 0
            or any(value.shape != (rows,) for value in one_dimensional)
            or not np.all(np.isfinite(self.features))
            or not np.all(np.isfinite(self.stress_utility_quote))
            or not np.all(np.isfinite(self.maximum_entry_loss_quote))
            or np.any(self.maximum_entry_loss_quote <= 0)
            or np.any(self.condition_index < 0)
            or np.any(self.condition_index >= len(self.condition_ids))
            or np.any(self.asset_index < 0)
            or np.any(self.asset_index >= len(_ASSETS))
            or np.any(self.release_monotonic_ns < self.decision_monotonic_ns)
            or np.any(self.entry_filled & ~self.observable)
            or np.any(self.exit_filled & ~self.entry_filled)
            or np.any(self.unknown_entry != ~self.observable)
            or np.any(self.failed_exit != (self.entry_filled & ~self.exit_filled))
            or np.any(self.terminal_reason_code < 0)
            or np.any(self.terminal_reason_code >= len(self.terminal_reasons))
            or self.excluded_pre_submit_count < 0
        ):
            raise ValueError("Polymarket Round 10 hurdle dataset is invalid")
        complete_mask = self.exit_filled
        if np.any(np.isfinite(self.complete_net_quote) != complete_mask) or np.any(
            np.abs(
                self.stress_utility_quote[self.unknown_entry]
                + self.maximum_entry_loss_quote[self.unknown_entry]
            )
            > 1e-9
        ):
            raise ValueError("Polymarket Round 10 utility labels are inconsistent")
        return self


@dataclass(frozen=True)
class PolymarketHurdleSplit:
    train_groups: tuple[int, ...]
    purge_group: int
    validation_groups: tuple[int, ...]
    split_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round10-development-split-v1",
            "train_groups": list(self.train_groups),
            "purge_group": self.purge_group,
            "validation_groups": list(self.validation_groups),
        }

    def validated(self) -> PolymarketHurdleSplit:
        combined = (*self.train_groups, self.purge_group, *self.validation_groups)
        if (
            len(self.train_groups) < 20
            or len(self.validation_groups) < 10
            or tuple(sorted(combined)) != combined
            or len(set(combined)) != len(combined)
            or self.split_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 10 development split is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRobustScaler:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float
    scaler_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "center": list(self.center),
            "scale": list(self.scale),
            "clip": self.clip,
        }

    def validated(self) -> PolymarketRobustScaler:
        if (
            len(self.center) != _FEATURE_COUNT
            or len(self.scale) != _FEATURE_COUNT
            or not all(math.isfinite(value) for value in self.center)
            or not all(math.isfinite(value) and value > 0 for value in self.scale)
            or not math.isfinite(self.clip)
            or self.clip <= 0
            or self.scaler_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 10 robust scaler is invalid")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != _FEATURE_COUNT:
            raise ValueError("Polymarket Round 10 scaler input is invalid")
        scaled = (values - np.asarray(self.center, dtype=np.float64)) / np.asarray(
            self.scale, dtype=np.float64
        )
        return np.clip(scaled, -self.clip, self.clip)


@dataclass(frozen=True)
class PolymarketLinearHead:
    name: str
    objective: str
    l2: float
    coefficients: tuple[float, ...]
    intercept: float
    target_center: float
    target_scale: float
    optimizer_iterations: int
    training_objective: float
    validation_loss: float
    model_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round10-linear-head-v1",
            "name": self.name,
            "objective": self.objective,
            "l2": self.l2,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "target_center": self.target_center,
            "target_scale": self.target_scale,
            "optimizer_iterations": self.optimizer_iterations,
            "training_objective": self.training_objective,
            "validation_loss": self.validation_loss,
        }

    def validated(self) -> PolymarketLinearHead:
        values = (
            *self.coefficients,
            self.intercept,
            self.target_center,
            self.target_scale,
            self.training_objective,
            self.validation_loss,
        )
        if (
            self.name
            not in {
                "observable",
                "entry_fill",
                "exit_fill",
                "complete_utility_mean",
                "complete_utility_q10",
            }
            or self.objective not in {"binary_log_loss", "huber", "pinball_q10"}
            or self.l2 not in POLYMARKET_ROUND10_L2_GRID
            or len(self.coefficients) != _FEATURE_COUNT
            or not all(math.isfinite(value) for value in values)
            or self.target_scale <= 0
            or self.optimizer_iterations < 0
            or self.model_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 10 linear head is invalid")
        return self

    def predict(self, standardized_features: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != _FEATURE_COUNT:
            raise ValueError("Polymarket Round 10 head input is invalid")
        linear = values @ np.asarray(self.coefficients) + self.intercept
        if self.objective == "binary_log_loss":
            result = expit(linear)
        else:
            result = self.target_center + self.target_scale * linear
        if not np.all(np.isfinite(result)):
            raise ValueError("Polymarket Round 10 head prediction is non-finite")
        return result


@dataclass(frozen=True)
class PolymarketHurdleBaseline:
    scaler: PolymarketRobustScaler
    observable: PolymarketLinearHead
    entry_fill: PolymarketLinearHead
    exit_fill: PolymarketLinearHead
    complete_utility_mean: PolymarketLinearHead
    complete_utility_q10: PolymarketLinearHead
    model_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND10_MODEL_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ROUND10_CONTRACT_SHA256,
            "scaler_sha256": self.scaler.scaler_sha256,
            "head_sha256": {
                "observable": self.observable.model_sha256,
                "entry_fill": self.entry_fill.model_sha256,
                "exit_fill": self.exit_fill.model_sha256,
                "complete_utility_mean": self.complete_utility_mean.model_sha256,
                "complete_utility_q10": self.complete_utility_q10.model_sha256,
            },
        }

    def validated(self) -> PolymarketHurdleBaseline:
        self.scaler.validated()
        for head in (
            self.observable,
            self.entry_fill,
            self.exit_fill,
            self.complete_utility_mean,
            self.complete_utility_q10,
        ):
            head.validated()
        if self.model_sha256 != _canonical_sha256(self.identity_payload()):
            raise ValueError("Polymarket Round 10 hurdle baseline is invalid")
        return self

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        values = self.scaler.transform(features)
        return {
            "observable": self.observable.predict(values),
            "entry_fill": self.entry_fill.predict(values),
            "exit_fill": self.exit_fill.predict(values),
            "complete_utility_mean": self.complete_utility_mean.predict(values),
            "complete_utility_q10": self.complete_utility_q10.predict(values),
        }


@dataclass(frozen=True)
class PolymarketHeadCandidate:
    head: str
    l2: float
    training_rows: int
    validation_rows: int
    training_objective: float
    validation_loss: float
    model_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolymarketBinaryMetrics:
    rows: int
    positives: int
    prevalence: float
    log_loss: float
    brier_score: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolymarketHurdlePolicyMetrics:
    score_margin_quote: float
    attempted_count: int
    complete_count: int
    complete_by_asset: tuple[int, int, int]
    unknown_entry_count: int
    failed_exit_count: int
    aggregate_stress_utility_quote: float
    utility_by_asset: tuple[float, float, float]
    median_market_utility_quote: float
    maximum_realized_drawdown_quote: float
    bootstrap_lower_mean_group_utility_quote: float
    bootstrap_median_mean_group_utility_quote: float
    bootstrap_upper_mean_group_utility_quote: float
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    selected_action_sha256: str

    def asdict(self) -> dict[str, object]:
        value = asdict(self)
        value["complete_by_asset"] = dict(
            zip(_ASSETS, self.complete_by_asset, strict=True)
        )
        value["utility_by_asset"] = dict(
            zip(_ASSETS, self.utility_by_asset, strict=True)
        )
        return value


@dataclass(frozen=True)
class PolymarketHurdleDevelopmentReport:
    dataset_sha256: str
    pipeline_report_sha256: str
    split: PolymarketHurdleSplit
    model: PolymarketHurdleBaseline
    head_candidates: tuple[PolymarketHeadCandidate, ...]
    validation_head_metrics: Mapping[str, PolymarketBinaryMetrics]
    policy_candidates: tuple[PolymarketHurdlePolicyMetrics, ...]
    selected_policy: PolymarketHurdlePolicyMetrics
    development_passed: bool
    nonlinear_challenger_authorized: bool
    confirmation_authorized: bool
    profitability_claim: bool
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND10_REPORT_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ROUND10_CONTRACT_SHA256,
            "dataset_sha256": self.dataset_sha256,
            "pipeline_report_sha256": self.pipeline_report_sha256,
            "split_sha256": self.split.split_sha256,
            "model_sha256": self.model.model_sha256,
            "head_candidates": [value.asdict() for value in self.head_candidates],
            "validation_head_metrics": {
                key: self.validation_head_metrics[key].asdict()
                for key in sorted(self.validation_head_metrics)
            },
            "policy_candidates": [value.asdict() for value in self.policy_candidates],
            "selected_policy": self.selected_policy.asdict(),
            "development_passed": self.development_passed,
            "nonlinear_challenger_authorized": self.nonlinear_challenger_authorized,
            "confirmation_authorized": self.confirmation_authorized,
            "profitability_claim": self.profitability_claim,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> PolymarketHurdleDevelopmentReport:
        self.split.validated()
        self.model.validated()
        if (
            not _SHA256.fullmatch(self.dataset_sha256)
            or not _SHA256.fullmatch(self.pipeline_report_sha256)
            or not self.head_candidates
            or len(self.policy_candidates) != len(POLYMARKET_ROUND10_SCORE_MARGIN_GRID)
            or self.selected_policy not in self.policy_candidates
            or self.development_passed != self.selected_policy.gate_passed
            or self.nonlinear_challenger_authorized != self.development_passed
            or self.confirmation_authorized
            or self.profitability_claim
            or self.report_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 10 development report is invalid")
        return self


def load_round9_hurdle_development_dataset(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
) -> PolymarketHurdleDataset:
    """Map the failed Round 9 action corpus into a Round 10 development role."""

    selected_report = str(pipeline_report_sha256).strip().lower()
    if not _SHA256.fullmatch(selected_report):
        raise ValueError("Polymarket Round 10 pipeline report digest is invalid")
    connection = store.connect()
    pipeline = connection.execute(
        """
        SELECT run_id, action_dataset_sha256_json, action_count,
               implementation_sha256
        FROM polymarket_action_value_pipeline
        WHERE report_sha256 = ?
        """,
        [selected_report],
    ).fetchone()
    if pipeline is None:
        raise ValueError("Polymarket Round 10 source pipeline does not exist")
    run_id = str(pipeline[0])
    dataset_ids = tuple(str(value) for value in json.loads(str(pipeline[1])))
    if len(dataset_ids) != 47 or int(pipeline[2]) != 251_892:
        raise ValueError("Polymarket Round 10 source population is unexpected")
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE round10_source_dataset(dataset_sha256 VARCHAR PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO round10_source_dataset VALUES (?)",
        [(value,) for value in dataset_ids],
    )
    count_row = connection.execute(
        """
        SELECT count(*)
        FROM polymarket_action_value_row AS a
        JOIN round10_source_dataset AS s USING (dataset_sha256)
        """
    ).fetchone()
    if count_row is None or int(count_row[0]) != int(pipeline[2]):
        raise ValueError("Polymarket Round 10 source action rows are incomplete")
    condition_rows = connection.execute(
        """
        SELECT condition_id
        FROM polymarket_market_snapshot
        WHERE run_id = ? AND condition_id IN (
            SELECT DISTINCT a.condition_id
            FROM polymarket_action_value_row AS a
            JOIN round10_source_dataset AS s USING (dataset_sha256)
        )
        ORDER BY condition_id
        """,
        [run_id],
    ).fetchall()
    condition_ids = tuple(str(row[0]) for row in condition_rows)
    if len(condition_ids) != 141:
        raise ValueError("Polymarket Round 10 source condition breadth is invalid")
    condition_index_by_id = {
        condition_id: index for index, condition_id in enumerate(condition_ids)
    }

    capacity = int(pipeline[2])
    features = np.empty((capacity, _FEATURE_COUNT), dtype=np.float64)
    action_hashes = np.empty(capacity, dtype="S64")
    condition_index = np.empty(capacity, dtype=np.int32)
    asset_index = np.empty(capacity, dtype=np.int8)
    outcome_up = np.empty(capacity, dtype=np.bool_)
    event_start_ms = np.empty(capacity, dtype=np.int64)
    decision_ns = np.empty(capacity, dtype=np.uint64)
    release_ns = np.empty(capacity, dtype=np.uint64)
    official_up = np.empty(capacity, dtype=np.bool_)
    observable = np.empty(capacity, dtype=np.bool_)
    entry_filled = np.empty(capacity, dtype=np.bool_)
    exit_filled = np.empty(capacity, dtype=np.bool_)
    complete_net = np.full(capacity, np.nan, dtype=np.float64)
    stress_utility = np.empty(capacity, dtype=np.float64)
    maximum_loss = np.empty(capacity, dtype=np.float64)
    unknown_entry = np.empty(capacity, dtype=np.bool_)
    failed_exit = np.empty(capacity, dtype=np.bool_)
    terminal_code = np.empty(capacity, dtype=np.int8)
    terminal_reasons: list[str] = []
    terminal_code_by_reason: dict[str, int] = {}
    excluded_pre_submit_count = 0
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": POLYMARKET_ROUND10_DATASET_SCHEMA_VERSION,
                "contract_sha256": POLYMARKET_ROUND10_CONTRACT_SHA256,
                "pipeline_report_sha256": selected_report,
                "feature_names": POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES,
                "source_role": "round9_development_only",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    cursor = connection.execute(
        """
        SELECT a.action_feature_sha256, a.action_label_sha256,
               a.condition_id, a.asset, a.outcome,
               a.decision_received_monotonic_ns, a.feature_values_json,
               a.terminal_reason, a.entry_filled, a.exit_filled,
               a.stress_utility_quote, a.net_quote,
               m.event_start_ms, m.tick_size, m.minimum_order_size,
               m.fees_enabled, m.fee_rate, m.fee_exponent,
               m.taker_order_delay_enabled, resolution.winning_outcome
        FROM polymarket_action_value_row AS a
        JOIN round10_source_dataset AS selected USING (dataset_sha256)
        JOIN polymarket_action_value_dataset AS d USING (dataset_sha256)
        JOIN polymarket_market_snapshot AS m
          ON m.run_id = d.source_run_id AND m.condition_id = a.condition_id
        JOIN polymarket_resolution_evidence AS resolution
          ON resolution.run_id = d.source_run_id
         AND resolution.condition_id = a.condition_id
        ORDER BY m.event_start_ms, a.decision_received_monotonic_ns,
                 a.condition_id, a.outcome
        """
    )
    size = 0
    while rows := cursor.fetchmany(8_192):
        for row in rows:
            reason = str(row[7])
            if reason in _KNOWN_PRE_SUBMIT_UNAVAILABLE_REASONS:
                excluded_pre_submit_count += 1
                continue
            if reason not in terminal_code_by_reason:
                terminal_code_by_reason[reason] = len(terminal_reasons)
                terminal_reasons.append(reason)
            raw_features = json.loads(str(row[6]))
            if len(raw_features) != len(POLYMARKET_ACTION_FEATURE_NAMES):
                raise ValueError("Polymarket Round 10 source feature width differs")
            tick = Decimal(str(row[13]))
            quantity = Decimal(str(row[14]))
            fees_enabled = bool(row[15])
            fee_rate = Decimal(str(row[16]))
            fee_exponent = int(row[17])
            taker_delay = bool(row[18])
            max_loss = polymarket_maximum_entry_loss(
                tick_size=tick,
                quantity=quantity,
                fees_enabled=fees_enabled,
                fee_rate=fee_rate,
                fee_exponent=fee_exponent,
            )
            oriented = [float(value) for value in raw_features]
            static = [
                float(tick),
                float(quantity),
                float(fees_enabled),
                float(fee_rate),
                float(fee_exponent),
                float(taker_delay),
                float(max_loss),
            ]
            features[size] = (*oriented, *static)
            action_hash = str(row[0])
            label_hash = str(row[1])
            if not _SHA256.fullmatch(action_hash) or not _SHA256.fullmatch(label_hash):
                raise ValueError("Polymarket Round 10 source row hash is invalid")
            action_hashes[size] = action_hash.encode("ascii")
            condition_id = str(row[2])
            condition_index[size] = condition_index_by_id[condition_id]
            asset = str(row[3])
            outcome = str(row[4])
            if asset not in _ASSETS or outcome not in _OUTCOMES:
                raise ValueError(
                    "Polymarket Round 10 source action identity is invalid"
                )
            asset_index[size] = _ASSETS.index(asset)
            outcome_up[size] = outcome == "Up"
            event_start_ms[size] = int(row[12])
            decision = int(row[5])
            decision_ns[size] = decision
            release_ns[size] = decision + POLYMARKET_ROUND10_MAX_RELEASE_DELAY_NS
            winning_outcome = str(row[19])
            if winning_outcome not in _OUTCOMES:
                raise ValueError("Polymarket Round 10 official outcome is invalid")
            official_up[size] = winning_outcome == "Up"
            is_unknown = reason in _UNKNOWN_ENTRY_REASONS
            is_observable = not is_unknown
            was_entry_filled = bool(row[8])
            was_exit_filled = bool(row[9])
            observable[size] = is_observable
            entry_filled[size] = was_entry_filled
            exit_filled[size] = was_exit_filled
            unknown_entry[size] = is_unknown
            failed_exit[size] = was_entry_filled and not was_exit_filled
            maximum_loss[size] = float(max_loss)
            if was_exit_filled:
                if row[11] is None or str(row[11]) == "":
                    raise ValueError("Polymarket Round 10 complete net is missing")
                complete_net[size] = float(Decimal(str(row[11])))
            if is_unknown:
                stress_utility[size] = -float(max_loss)
            else:
                if row[10] is None or str(row[10]) == "":
                    raise ValueError("Polymarket Round 10 stress utility is missing")
                stress_utility[size] = float(Decimal(str(row[10])))
            terminal_code[size] = terminal_code_by_reason[reason]
            digest.update(bytes.fromhex(action_hash))
            digest.update(bytes.fromhex(label_hash))
            digest.update(format(max_loss, "f").encode("ascii"))
            size += 1
    if size + excluded_pre_submit_count != capacity:
        raise ValueError("Polymarket Round 10 streamed row count differs")

    def trimmed(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value[:size])

    dataset = PolymarketHurdleDataset(
        pipeline_report_sha256=selected_report,
        dataset_sha256=digest.hexdigest(),
        feature_names=POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES,
        condition_ids=condition_ids,
        features=trimmed(features),
        action_feature_sha256=trimmed(action_hashes),
        condition_index=trimmed(condition_index),
        asset_index=trimmed(asset_index),
        outcome_up=trimmed(outcome_up),
        event_start_ms=trimmed(event_start_ms),
        decision_monotonic_ns=trimmed(decision_ns),
        release_monotonic_ns=trimmed(release_ns),
        official_up=trimmed(official_up),
        observable=trimmed(observable),
        entry_filled=trimmed(entry_filled),
        exit_filled=trimmed(exit_filled),
        complete_net_quote=trimmed(complete_net),
        stress_utility_quote=trimmed(stress_utility),
        maximum_entry_loss_quote=trimmed(maximum_loss),
        unknown_entry=trimmed(unknown_entry),
        failed_exit=trimmed(failed_exit),
        terminal_reason_code=trimmed(terminal_code),
        terminal_reasons=tuple(terminal_reasons),
        excluded_pre_submit_count=excluded_pre_submit_count,
    )
    return dataset.validated()


def build_round10_development_split(
    dataset: PolymarketHurdleDataset,
) -> PolymarketHurdleSplit:
    groups = dataset.validated().groups
    train_count = int(
        math.floor(len(groups) * POLYMARKET_ROUND10_DEVELOPMENT_TRAIN_FRACTION)
    )
    if train_count < 20 or len(groups) - train_count - 1 < 10:
        raise ValueError(
            "Polymarket Round 10 development group breadth is insufficient"
        )
    provisional = PolymarketHurdleSplit(
        train_groups=groups[:train_count],
        purge_group=groups[train_count],
        validation_groups=groups[train_count + 1 :],
        split_sha256="",
    )
    split = replace(
        provisional,
        split_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()
    for role, selected_groups in (
        ("train", split.train_groups),
        ("validation", split.validation_groups),
    ):
        role_mask = np.isin(dataset.event_start_ms, selected_groups)
        for asset in range(len(_ASSETS)):
            outcomes = set(
                dataset.official_up[role_mask & (dataset.asset_index == asset)]
            )
            if outcomes != {False, True}:
                raise ValueError(
                    f"Polymarket Round 10 {role} outcome breadth is insufficient for {_ASSETS[asset]}"
                )
    return split


def _fit_scaler(features: np.ndarray) -> PolymarketRobustScaler:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != _FEATURE_COUNT or values.shape[0] == 0:
        raise ValueError("Polymarket Round 10 scaler fit matrix is invalid")
    center = np.empty(_FEATURE_COUNT, dtype=np.float64)
    scale = np.empty(_FEATURE_COUNT, dtype=np.float64)
    for column in range(_FEATURE_COUNT):
        q25, q50, q75 = np.quantile(values[:, column], (0.25, 0.50, 0.75))
        center[column] = q50
        spread = q75 - q25
        scale[column] = (
            spread if math.isfinite(float(spread)) and spread > 1e-12 else 1.0
        )
    provisional = PolymarketRobustScaler(
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        clip=POLYMARKET_ROUND10_STANDARDIZED_CLIP,
        scaler_sha256="",
    )
    return replace(
        provisional,
        scaler_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.float64)
    prediction = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Polymarket Round 10 binary metric input is invalid")
    return float(
        -np.mean(target * np.log(prediction) + (1.0 - target) * np.log1p(-prediction))
    )


def _binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> PolymarketBinaryMetrics:
    target = np.asarray(labels, dtype=np.bool_)
    prediction = np.asarray(probabilities, dtype=np.float64)
    return PolymarketBinaryMetrics(
        rows=int(target.size),
        positives=int(np.count_nonzero(target)),
        prevalence=float(np.mean(target)),
        log_loss=_binary_log_loss(target, prediction),
        brier_score=float(np.mean((prediction - target.astype(np.float64)) ** 2)),
    )


def _head(
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
    validation_loss: float,
) -> PolymarketLinearHead:
    provisional = PolymarketLinearHead(
        name=name,
        objective=objective,
        l2=l2,
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        target_center=float(target_center),
        target_scale=float(target_scale),
        optimizer_iterations=int(optimizer_iterations),
        training_objective=float(training_objective),
        validation_loss=float(validation_loss),
        model_sha256="",
    )
    return replace(
        provisional,
        model_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _fit_binary_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    name: str,
    l2: float,
) -> PolymarketLinearHead:
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64)
    x_validation = np.asarray(validation_features, dtype=np.float64)
    y_validation = np.asarray(validation_labels, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != _FEATURE_COUNT
        or y.shape != (x.shape[0],)
        or x_validation.ndim != 2
        or x_validation.shape[1] != _FEATURE_COUNT
        or y_validation.shape != (x_validation.shape[0],)
        or set(np.unique(y)) != {0.0, 1.0}
        or set(np.unique(y_validation)) != {0.0, 1.0}
    ):
        raise ValueError(f"Polymarket Round 10 {name} binary fit input is invalid")
    prevalence = float(np.mean(y))
    initial_intercept = math.log(prevalence / (1.0 - prevalence))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        intercept = theta[-1]
        logits = x @ coefficients + intercept
        loss = float(
            np.mean(np.logaddexp(0.0, logits) - y * logits)
            + 0.5 * l2 * np.dot(coefficients, coefficients)
        )
        residual = expit(logits) - y
        gradient = np.empty_like(theta)
        gradient[:-1] = x.T @ residual / y.size + l2 * coefficients
        gradient[-1] = float(np.mean(residual))
        return loss, gradient

    initial = np.zeros(_FEATURE_COUNT + 1, dtype=np.float64)
    initial[-1] = initial_intercept
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(
            f"Polymarket Round 10 {name} optimizer failed:{result.message}"
        )
    probabilities = expit(x_validation @ result.x[:-1] + result.x[-1])
    return _head(
        name=name,
        objective="binary_log_loss",
        l2=l2,
        coefficients=result.x[:-1],
        intercept=float(result.x[-1]),
        target_center=0.0,
        target_scale=1.0,
        optimizer_iterations=int(result.nit),
        training_objective=float(result.fun),
        validation_loss=_binary_log_loss(y_validation, probabilities),
    )


def _utility_target_scale(target: np.ndarray) -> tuple[float, float]:
    values = np.asarray(target, dtype=np.float64)
    q25, center, q75 = np.quantile(values, (0.25, 0.50, 0.75))
    scale = float(q75 - q25)
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return float(center), scale


def _fit_utility_head(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    name: str,
    objective_name: str,
    l2: float,
) -> PolymarketLinearHead:
    x = np.asarray(train_features, dtype=np.float64)
    y_raw = np.asarray(train_target, dtype=np.float64)
    x_validation = np.asarray(validation_features, dtype=np.float64)
    y_validation_raw = np.asarray(validation_target, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != _FEATURE_COUNT
        or y_raw.shape != (x.shape[0],)
        or x_validation.ndim != 2
        or x_validation.shape[1] != _FEATURE_COUNT
        or y_validation_raw.shape != (x_validation.shape[0],)
        or x.shape[0] == 0
        or x_validation.shape[0] == 0
        or not np.all(np.isfinite(y_raw))
        or not np.all(np.isfinite(y_validation_raw))
    ):
        raise ValueError(f"Polymarket Round 10 {name} utility fit input is invalid")
    target_center, target_scale = _utility_target_scale(y_raw)
    y = (y_raw - target_center) / target_scale
    y_validation = (y_validation_raw - target_center) / target_scale
    if objective_name == "huber":
        initial_intercept = float(np.mean(y))

        def data_loss_and_gradient(
            residual: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            absolute = np.abs(residual)
            quadratic = absolute <= POLYMARKET_ROUND10_HUBER_DELTA
            loss = np.where(
                quadratic,
                0.5 * residual * residual,
                POLYMARKET_ROUND10_HUBER_DELTA
                * (absolute - 0.5 * POLYMARKET_ROUND10_HUBER_DELTA),
            )
            gradient = np.clip(
                residual,
                -POLYMARKET_ROUND10_HUBER_DELTA,
                POLYMARKET_ROUND10_HUBER_DELTA,
            )
            return loss, gradient

    elif objective_name == "pinball_q10":
        initial_intercept = float(np.quantile(y, POLYMARKET_ROUND10_UTILITY_QUANTILE))

        def data_loss_and_gradient(
            residual: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            smoothing = POLYMARKET_ROUND10_PINBALL_SMOOTHING
            positive = residual / smoothing
            negative = -residual / smoothing
            loss = smoothing * (
                (1.0 - POLYMARKET_ROUND10_UTILITY_QUANTILE)
                * np.logaddexp(0.0, positive)
                + POLYMARKET_ROUND10_UTILITY_QUANTILE * np.logaddexp(0.0, negative)
            )
            gradient = (1.0 - POLYMARKET_ROUND10_UTILITY_QUANTILE) * expit(
                positive
            ) - POLYMARKET_ROUND10_UTILITY_QUANTILE * expit(negative)
            return loss, gradient

    else:
        raise ValueError("Polymarket Round 10 utility objective is invalid")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        intercept = theta[-1]
        residual = x @ coefficients + intercept - y
        losses, data_gradient = data_loss_and_gradient(residual)
        loss = float(np.mean(losses) + 0.5 * l2 * np.dot(coefficients, coefficients))
        gradient = np.empty_like(theta)
        gradient[:-1] = x.T @ data_gradient / y.size + l2 * coefficients
        gradient[-1] = float(np.mean(data_gradient))
        return loss, gradient

    initial = np.zeros(_FEATURE_COUNT + 1, dtype=np.float64)
    initial[-1] = initial_intercept
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(
            f"Polymarket Round 10 {name} optimizer failed:{result.message}"
        )
    validation_residual = x_validation @ result.x[:-1] + result.x[-1] - y_validation
    validation_losses, _ = data_loss_and_gradient(validation_residual)
    return _head(
        name=name,
        objective=objective_name,
        l2=l2,
        coefficients=result.x[:-1],
        intercept=float(result.x[-1]),
        target_center=target_center,
        target_scale=target_scale,
        optimizer_iterations=int(result.nit),
        training_objective=float(result.fun),
        validation_loss=float(np.mean(validation_losses)),
    )


def _select_head(
    *,
    name: str,
    objective: str,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    progress: ProgressCallback | None = None,
) -> tuple[PolymarketLinearHead, tuple[PolymarketHeadCandidate, ...]]:
    models: list[PolymarketLinearHead] = []
    candidates: list[PolymarketHeadCandidate] = []
    for l2 in POLYMARKET_ROUND10_L2_GRID:
        if progress is not None:
            progress(
                "head_candidate_started",
                {
                    "head": name,
                    "objective": objective,
                    "l2": l2,
                    "training_rows": int(train_target.size),
                    "validation_rows": int(validation_target.size),
                },
            )
        if objective == "binary_log_loss":
            model = _fit_binary_head(
                train_features,
                train_target,
                validation_features,
                validation_target,
                name=name,
                l2=l2,
            )
        else:
            model = _fit_utility_head(
                train_features,
                train_target,
                validation_features,
                validation_target,
                name=name,
                objective_name=objective,
                l2=l2,
            )
        models.append(model)
        candidates.append(
            PolymarketHeadCandidate(
                head=name,
                l2=l2,
                training_rows=int(train_target.size),
                validation_rows=int(validation_target.size),
                training_objective=model.training_objective,
                validation_loss=model.validation_loss,
                model_sha256=model.model_sha256,
            )
        )
        if progress is not None:
            progress(
                "head_candidate_complete",
                {
                    "head": name,
                    "objective": objective,
                    "l2": l2,
                    "training_objective": model.training_objective,
                    "validation_loss": model.validation_loss,
                    "optimizer_iterations": model.optimizer_iterations,
                },
            )
    selected = min(models, key=lambda value: (value.validation_loss, -value.l2))
    return selected, tuple(candidates)


def _bootstrap_group_mean(
    utility_by_group: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(utility_by_group, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Polymarket Round 10 bootstrap input is invalid")
    generator = np.random.default_rng(POLYMARKET_ROUND10_BOOTSTRAP_SEED)
    means = np.empty(POLYMARKET_ROUND10_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(POLYMARKET_ROUND10_BOOTSTRAP_SAMPLES):
        sample = generator.integers(0, values.size, size=values.size)
        means[index] = float(np.mean(values[sample]))
    lower, median, upper = np.quantile(means, (0.025, 0.50, 0.975))
    return float(lower), float(median), float(upper)


def _evaluate_policy(
    dataset: PolymarketHurdleDataset,
    validation_mask: np.ndarray,
    score: np.ndarray,
    *,
    score_margin_quote: float,
) -> PolymarketHurdlePolicyMetrics:
    indices = np.flatnonzero(validation_mask)
    if score.shape != (indices.size,) or not np.all(np.isfinite(score)):
        raise ValueError("Polymarket Round 10 policy score is invalid")
    by_decision: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row_index, action_score in zip(indices, score, strict=True):
        by_decision.setdefault(
            (
                int(dataset.condition_index[row_index]),
                int(dataset.decision_monotonic_ns[row_index]),
            ),
            [],
        ).append((int(row_index), float(action_score)))
    release_by_condition: dict[int, int] = {}
    blocked: set[int] = set()
    selected: list[int] = []
    utility_by_asset = np.zeros(len(_ASSETS), dtype=np.float64)
    complete_by_asset = np.zeros(len(_ASSETS), dtype=np.int64)
    utility_by_condition: dict[int, float] = {}
    realized_events: list[tuple[int, float]] = []
    unknown_count = 0
    failed_exit_count = 0
    complete_count = 0
    for (condition, decision), candidates in sorted(
        by_decision.items(), key=lambda value: (value[0][1], value[0][0])
    ):
        if condition in blocked or decision < release_by_condition.get(condition, 0):
            continue
        passing = [value for value in candidates if value[1] > score_margin_quote]
        if not passing:
            continue
        best_score = max(value[1] for value in passing)
        winners = [value for value in passing if value[1] == best_score]
        if len(winners) != 1:
            continue
        row_index = winners[0][0]
        selected.append(row_index)
        utility = float(dataset.stress_utility_quote[row_index])
        asset = int(dataset.asset_index[row_index])
        utility_by_asset[asset] += utility
        utility_by_condition[condition] = (
            utility_by_condition.get(condition, 0.0) + utility
        )
        realized_events.append((int(dataset.release_monotonic_ns[row_index]), utility))
        if bool(dataset.unknown_entry[row_index]):
            unknown_count += 1
            blocked.add(condition)
        elif bool(dataset.failed_exit[row_index]):
            failed_exit_count += 1
            blocked.add(condition)
        else:
            release_by_condition[condition] = int(
                dataset.release_monotonic_ns[row_index]
            )
            if bool(dataset.exit_filled[row_index]):
                complete_count += 1
                complete_by_asset[asset] += 1
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for _, utility in sorted(realized_events):
        equity += utility
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    validation_groups = tuple(
        int(value) for value in np.unique(dataset.event_start_ms[validation_mask])
    )
    group_utility = np.zeros(len(validation_groups), dtype=np.float64)
    group_index = {value: index for index, value in enumerate(validation_groups)}
    for row_index in selected:
        group_utility[group_index[int(dataset.event_start_ms[row_index])]] += float(
            dataset.stress_utility_quote[row_index]
        )
    bootstrap = _bootstrap_group_mean(group_utility)
    aggregate = float(np.sum(utility_by_asset))
    market_values = np.asarray(list(utility_by_condition.values()), dtype=np.float64)
    median_market = float(np.median(market_values)) if market_values.size else 0.0
    reasons: list[str] = []
    if complete_count < 30:
        reasons.append(f"completed_actions_below_30:{complete_count}")
    for asset, count in zip(_ASSETS, complete_by_asset, strict=True):
        if int(count) < 5:
            reasons.append(f"{asset.lower()}_completed_actions_below_5:{int(count)}")
    if aggregate <= 0:
        reasons.append(f"nonpositive_aggregate_stress_utility:{aggregate:.8f}")
    if median_market <= 0:
        reasons.append(f"nonpositive_median_market_utility:{median_market:.8f}")
    for asset, utility in zip(_ASSETS, utility_by_asset, strict=True):
        if float(utility) <= 0:
            reasons.append(f"nonpositive_{asset.lower()}_utility:{float(utility):.8f}")
    if unknown_count:
        reasons.append(f"selected_unknown_entries:{unknown_count}")
    if failed_exit_count:
        reasons.append(f"selected_failed_exits:{failed_exit_count}")
    if bootstrap[0] <= 0:
        reasons.append(f"nonpositive_bootstrap_lower_group_mean:{bootstrap[0]:.8f}")
    selected_hash = _canonical_sha256(
        [dataset.action_feature_sha256[index].decode("ascii") for index in selected]
    )
    return PolymarketHurdlePolicyMetrics(
        score_margin_quote=float(score_margin_quote),
        attempted_count=len(selected),
        complete_count=complete_count,
        complete_by_asset=tuple(int(value) for value in complete_by_asset),
        unknown_entry_count=unknown_count,
        failed_exit_count=failed_exit_count,
        aggregate_stress_utility_quote=aggregate,
        utility_by_asset=tuple(float(value) for value in utility_by_asset),
        median_market_utility_quote=median_market,
        maximum_realized_drawdown_quote=maximum_drawdown,
        bootstrap_lower_mean_group_utility_quote=bootstrap[0],
        bootstrap_median_mean_group_utility_quote=bootstrap[1],
        bootstrap_upper_mean_group_utility_quote=bootstrap[2],
        gate_passed=not reasons,
        gate_reasons=tuple(reasons),
        selected_action_sha256=selected_hash,
    )


def fit_round10_development_hurdle_baseline(
    dataset: PolymarketHurdleDataset,
    *,
    progress: ProgressCallback | None = None,
) -> PolymarketHurdleDevelopmentReport:
    """Fit on Round 9 development roles without creating confirmation evidence."""

    source = dataset.validated()
    split = build_round10_development_split(source)
    train_mask = np.isin(source.event_start_ms, split.train_groups)
    validation_mask = np.isin(source.event_start_ms, split.validation_groups)
    if progress is not None:
        progress(
            "scaler_started",
            {
                "training_rows": int(np.count_nonzero(train_mask)),
                "feature_count": _FEATURE_COUNT,
            },
        )
    scaler = _fit_scaler(source.features[train_mask])
    standardized = scaler.transform(source.features)
    if progress is not None:
        progress("scaler_complete", {"scaler_sha256": scaler.scaler_sha256})
    candidates: list[PolymarketHeadCandidate] = []

    head_specs = (
        (
            "observable",
            "binary_log_loss",
            np.ones(source.rows, dtype=np.bool_),
            source.observable,
        ),
        (
            "entry_fill",
            "binary_log_loss",
            source.observable,
            source.entry_filled,
        ),
        (
            "exit_fill",
            "binary_log_loss",
            source.entry_filled,
            source.exit_filled,
        ),
        (
            "complete_utility_mean",
            "huber",
            source.exit_filled,
            source.complete_net_quote,
        ),
        (
            "complete_utility_q10",
            "pinball_q10",
            source.exit_filled,
            source.complete_net_quote,
        ),
    )
    selected_heads: dict[str, PolymarketLinearHead] = {}
    for name, objective, eligible, target in head_specs:
        train = train_mask & eligible
        validation = validation_mask & eligible
        selected, head_candidates = _select_head(
            name=name,
            objective=objective,
            train_features=standardized[train],
            train_target=target[train],
            validation_features=standardized[validation],
            validation_target=target[validation],
            progress=progress,
        )
        selected_heads[name] = selected
        candidates.extend(head_candidates)
    provisional_model = PolymarketHurdleBaseline(
        scaler=scaler,
        observable=selected_heads["observable"],
        entry_fill=selected_heads["entry_fill"],
        exit_fill=selected_heads["exit_fill"],
        complete_utility_mean=selected_heads["complete_utility_mean"],
        complete_utility_q10=selected_heads["complete_utility_q10"],
        model_sha256="",
    )
    model = replace(
        provisional_model,
        model_sha256=_canonical_sha256(provisional_model.identity_payload()),
    ).validated()
    if progress is not None:
        progress("baseline_model_complete", {"model_sha256": model.model_sha256})
    validation_indices = np.flatnonzero(validation_mask)
    predictions = model.predict(source.features[validation_mask])
    validation_head_metrics = {
        "observable": _binary_metrics(
            source.observable[validation_mask], predictions["observable"]
        ),
        "entry_fill": _binary_metrics(
            source.entry_filled[validation_mask & source.observable],
            model.predict(source.features[validation_mask & source.observable])[
                "entry_fill"
            ],
        ),
        "exit_fill": _binary_metrics(
            source.exit_filled[validation_mask & source.entry_filled],
            model.predict(source.features[validation_mask & source.entry_filled])[
                "exit_fill"
            ],
        ),
    }
    conservative_q10 = np.minimum(
        predictions["complete_utility_q10"], predictions["complete_utility_mean"]
    )
    score = (
        predictions["observable"]
        * predictions["entry_fill"]
        * (
            predictions["exit_fill"] * conservative_q10
            - (1.0 - predictions["exit_fill"])
            * source.maximum_entry_loss_quote[validation_indices]
        )
        - (1.0 - predictions["observable"])
        * source.maximum_entry_loss_quote[validation_indices]
    )
    policy_candidates = tuple(
        _evaluate_policy(
            source,
            validation_mask,
            score,
            score_margin_quote=margin,
        )
        for margin in POLYMARKET_ROUND10_SCORE_MARGIN_GRID
    )
    if progress is not None:
        for candidate in policy_candidates:
            progress(
                "policy_candidate_complete",
                {
                    "score_margin_quote": candidate.score_margin_quote,
                    "attempted_count": candidate.attempted_count,
                    "complete_count": candidate.complete_count,
                    "unknown_entry_count": candidate.unknown_entry_count,
                    "failed_exit_count": candidate.failed_exit_count,
                    "aggregate_stress_utility_quote": (
                        candidate.aggregate_stress_utility_quote
                    ),
                    "gate_passed": candidate.gate_passed,
                },
            )
    passing = [value for value in policy_candidates if value.gate_passed]
    selected_policy = (
        max(
            passing,
            key=lambda value: (
                value.bootstrap_lower_mean_group_utility_quote,
                value.aggregate_stress_utility_quote,
                value.score_margin_quote,
            ),
        )
        if passing
        else max(
            policy_candidates,
            key=lambda value: (
                value.bootstrap_lower_mean_group_utility_quote,
                value.aggregate_stress_utility_quote,
                -value.unknown_entry_count,
                -value.failed_exit_count,
                value.score_margin_quote,
            ),
        )
    )
    provisional_report = PolymarketHurdleDevelopmentReport(
        dataset_sha256=source.dataset_sha256,
        pipeline_report_sha256=source.pipeline_report_sha256,
        split=split,
        model=model,
        head_candidates=tuple(candidates),
        validation_head_metrics=validation_head_metrics,
        policy_candidates=policy_candidates,
        selected_policy=selected_policy,
        development_passed=selected_policy.gate_passed,
        nonlinear_challenger_authorized=selected_policy.gate_passed,
        confirmation_authorized=False,
        profitability_claim=False,
        report_sha256="",
    )
    return replace(
        provisional_report,
        report_sha256=_canonical_sha256(provisional_report.identity_payload()),
    ).validated()


__all__ = [
    "POLYMARKET_ROUND10_CONTRACT_SHA256",
    "polymarket_maximum_entry_loss",
    "POLYMARKET_ROUND10_DEVELOPMENT_FEATURE_NAMES",
    "POLYMARKET_ROUND10_L2_GRID",
    "POLYMARKET_ROUND10_SCORE_MARGIN_GRID",
    "PolymarketBinaryMetrics",
    "PolymarketHeadCandidate",
    "PolymarketHurdleBaseline",
    "PolymarketHurdleDataset",
    "PolymarketHurdleDevelopmentReport",
    "PolymarketHurdlePolicyMetrics",
    "PolymarketHurdleSplit",
    "PolymarketLinearHead",
    "PolymarketRobustScaler",
    "build_round10_development_split",
    "fit_round10_development_hurdle_baseline",
    "load_round9_hurdle_development_dataset",
]
