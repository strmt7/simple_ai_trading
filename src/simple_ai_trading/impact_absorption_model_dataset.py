"""Audited operational target adapter for the Round 73 shallow evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np

from .impact_absorption_cohort import (
    ROUND73_SHOCK_ANCHOR_TABLE,
    ROUND73_SHOCK_STUDY_TABLE,
)
from .impact_absorption_corpus import ROUND73_CORPUS_RUN_TABLE
from .impact_absorption_grid import ROUND73_GRID_FEATURE_NAMES
from .impact_absorption_grid_store import ROUND73_GRID_VECTOR_TABLE
from .impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
    ROUND73_ACTION_SIDES,
    ROUND73_EVALUATION_CONTRACT_SHA256,
    ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
    action_align_round73_features,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_store_v2 import (
    ROUND73_TARGET_V2_OPTION_TABLE,
    ROUND73_TARGET_V2_STUDY_TABLE,
    audit_round73_target_study,
)
from .impact_absorption_target_store_v3 import (
    ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE,
    ROUND73_TARGET_V3_OPTION_TABLE,
    ROUND73_TARGET_V3_TEST_STUDY_TABLE,
    seal_round73_development_targets,
    seal_round73_test_targets,
)
from .impact_absorption_targets import ROUND73_TARGET_MAX_STATE_LATENESS_NS


ROUND73_OPERATIONAL_DATASET_SCHEMA_VERSION = "round-073-operational-dataset-v1"
ROUND73_PRIMARY_ENTRY_DELAY_MS = 500
ROUND73_PRIMARY_HORIZON_MS = 60_000
ROUND73_PRIMARY_REFERENCE_NOTIONAL = 1_000
ROUND73_OPERATIONAL_STATUS_NAMES = (
    "observed_complete_transaction",
    "pre_entry_safety_abort",
    "experimental_right_censoring",
    "post_entry_unresolved_risk",
)
ROUND73_OBSERVED_STATUS = 0
ROUND73_PRE_ENTRY_ABORT_STATUS = 1
ROUND73_RIGHT_CENSORED_STATUS = 2
ROUND73_POST_ENTRY_UNRESOLVED_STATUS = 3

_PRE_ENTRY_REASONS = frozenset(
    {
        "quantity_filter",
        "entry_state_late",
        "entry_capacity",
        "entry_minimum_notional",
        "funding_boundary",
    }
)
_POST_ENTRY_REASONS = frozenset({"path_capacity", "exit_state_late", "exit_capacity"})
_KNOWN_REASONS = _PRE_ENTRY_REASONS | _POST_ENTRY_REASONS | {"coverage_end"}
_ROLE_NAMES = frozenset({"training", "tuning", "test"})
_STUDY_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

TargetStudyAuditFunction = Callable[..., object]
TargetStudySealFunction = Callable[..., object]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _grid_vector_sha256(
    run_id: str,
    symbol: str,
    anchor_index: int,
    values: np.ndarray,
) -> str:
    identity = f"{run_id}:{symbol}:{int(anchor_index)}:".encode("ascii")
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(identity + memoryview(canonical).cast("B")).hexdigest()


def _array_digest(digest: object, value: np.ndarray) -> None:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest.update(dtype.str.encode("ascii"))
    digest.update(int(canonical.ndim).to_bytes(2, "little", signed=False))
    for size in canonical.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _string_sequence_digest(digest: object, values: Sequence[str]) -> None:
    digest.update(_canonical_json(list(values)).encode("ascii"))


@dataclass(frozen=True)
class Round73OperationalOutcome:
    status: int
    reason: str
    binary_target: float
    continuous_target_bps: float
    completed_transaction: bool

    @property
    def label_available(self) -> bool:
        return self.status in {ROUND73_OBSERVED_STATUS, ROUND73_PRE_ENTRY_ABORT_STATUS}


def classify_round73_operational_outcome(
    *,
    eligible: bool,
    ineligible_reasons_json: str,
    positive_net_payoff: bool | None,
    net_payoff_bps: float | None,
    deterministically_boundary_censored: bool,
) -> Round73OperationalOutcome:
    """Map exact target mechanics into the frozen operational target semantics."""

    try:
        parsed = json.loads(str(ineligible_reasons_json))
    except json.JSONDecodeError as exc:
        raise ValueError("Round 73 target reason JSON is invalid") from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise ValueError("Round 73 target reasons must be a string list")
    reasons = tuple(parsed)
    if len(reasons) != len(set(reasons)) or any(
        reason not in _KNOWN_REASONS for reason in reasons
    ):
        raise ValueError("Round 73 target reasons are unknown or duplicated")
    if bool(eligible):
        if (
            reasons
            or positive_net_payoff is None
            or net_payoff_bps is None
            or not math.isfinite(float(net_payoff_bps))
        ):
            raise ValueError("Round 73 eligible operational outcome is incomplete")
    elif (
        len(reasons) != 1
        or positive_net_payoff is not None
        or net_payoff_bps is not None
    ):
        raise ValueError("Round 73 ineligible operational outcome is inconsistent")
    if deterministically_boundary_censored:
        return Round73OperationalOutcome(
            status=ROUND73_RIGHT_CENSORED_STATUS,
            reason="source_run_boundary",
            binary_target=float("nan"),
            continuous_target_bps=float("nan"),
            completed_transaction=False,
        )
    if eligible:
        return Round73OperationalOutcome(
            status=ROUND73_OBSERVED_STATUS,
            reason="eligible",
            binary_target=float(bool(positive_net_payoff)),
            continuous_target_bps=float(net_payoff_bps),
            completed_transaction=True,
        )
    reason = reasons[0]
    if reason == "coverage_end":
        raise ValueError(
            "Round 73 coverage-end target is outside deterministic censoring"
        )
    if reason in _PRE_ENTRY_REASONS:
        return Round73OperationalOutcome(
            status=ROUND73_PRE_ENTRY_ABORT_STATUS,
            reason=reason,
            binary_target=0.0,
            continuous_target_bps=0.0,
            completed_transaction=False,
        )
    return Round73OperationalOutcome(
        status=ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
        reason=reason,
        binary_target=float("nan"),
        continuous_target_bps=float("nan"),
        completed_transaction=False,
    )


@dataclass(frozen=True)
class Round73OperationalDataset:
    schema_version: str
    study_id: str
    cohort_manifest_sha256: str
    target_study_manifest_sha256: str
    feature_names_sha256: str
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    role: tuple[str, ...]
    side: tuple[str, ...]
    outcome_reason: tuple[str, ...]
    option_sha256: tuple[str, ...]
    selected_anchor_sha256: tuple[str, ...]
    anchor_index: np.ndarray
    anchor_wall_ns: np.ndarray
    feature_values: np.ndarray
    outcome_status: np.ndarray
    binary_target: np.ndarray
    continuous_target_bps: np.ndarray
    completed_transaction: np.ndarray
    dataset_sha256: str
    target_result_observed: bool = True
    predictive_edge_claim: bool = False
    profitability_claim: bool = False
    trading_authority: bool = False

    @property
    def rows(self) -> int:
        return len(self.run_id)

    @property
    def model_label_mask(self) -> np.ndarray:
        output = np.isin(
            self.outcome_status,
            (ROUND73_OBSERVED_STATUS, ROUND73_PRE_ENTRY_ABORT_STATUS),
        )
        output.setflags(write=False)
        return output

    def validate(self) -> None:
        rows = self.rows
        strings = (
            self.run_id,
            self.symbol,
            self.role,
            self.side,
            self.outcome_reason,
            self.option_sha256,
            self.selected_anchor_sha256,
        )
        arrays = (
            self.anchor_index,
            self.anchor_wall_ns,
            self.outcome_status,
            self.binary_target,
            self.continuous_target_bps,
            self.completed_transaction,
        )
        if (
            self.schema_version != ROUND73_OPERATIONAL_DATASET_SCHEMA_VERSION
            or _STUDY_ID.fullmatch(self.study_id) is None
            or _SHA256.fullmatch(self.cohort_manifest_sha256) is None
            or _SHA256.fullmatch(self.target_study_manifest_sha256) is None
            or self.feature_names_sha256 != ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256
            or rows == 0
            or any(len(values) != rows for values in strings)
            or any(value.shape != (rows,) for value in arrays)
            or self.feature_values.shape
            != (rows, len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES))
            or self.feature_values.dtype != np.float64
            or any(value.flags.writeable for value in (*arrays, self.feature_values))
            or not np.all(np.isfinite(self.feature_values))
            or any(symbol not in IMPACT_CAPTURE_SYMBOLS for symbol in self.symbol)
            or any(role not in _ROLE_NAMES for role in self.role)
            or any(side not in ROUND73_ACTION_SIDES for side in self.side)
            or any(_SHA256.fullmatch(value) is None for value in self.option_sha256)
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.selected_anchor_sha256
            )
            or np.any(self.anchor_index < 0)
            or np.any(self.anchor_wall_ns <= 0)
            or np.any(
                (self.outcome_status < 0)
                | (self.outcome_status >= len(ROUND73_OPERATIONAL_STATUS_NAMES))
            )
            or any(
                (
                    self.target_result_observed is not True,
                    self.predictive_edge_claim,
                    self.profitability_claim,
                    self.trading_authority,
                )
            )
        ):
            raise ValueError("Round 73 operational dataset contract is invalid")
        label_mask = self.model_label_mask
        if (
            np.any(~np.isfinite(self.binary_target[label_mask]))
            or np.any(~np.isfinite(self.continuous_target_bps[label_mask]))
            or np.any(np.isfinite(self.binary_target[~label_mask]))
            or np.any(np.isfinite(self.continuous_target_bps[~label_mask]))
            or np.any(
                (self.binary_target[label_mask] != 0.0)
                & (self.binary_target[label_mask] != 1.0)
            )
            or np.any(
                self.completed_transaction
                != (self.outcome_status == ROUND73_OBSERVED_STATUS)
            )
        ):
            raise ValueError("Round 73 operational dataset target semantics differ")
        keys = list(
            zip(
                self.anchor_wall_ns.tolist(),
                self.run_id,
                self.symbol,
                self.anchor_index.tolist(),
                self.side,
                strict=True,
            )
        )
        side_order = {"long": 0, "short": 1}
        expected_order = sorted(
            keys,
            key=lambda value: (*value[:4], side_order[value[4]]),
        )
        if keys != expected_order or len(keys) != len(set(keys)):
            raise ValueError("Round 73 operational dataset ordering differs")
        anchor_keys = [value[:4] for value in keys]
        for index in range(0, rows, 2):
            if (
                index + 1 >= rows
                or anchor_keys[index] != anchor_keys[index + 1]
                or self.side[index : index + 2] != ("long", "short")
                or self.role[index] != self.role[index + 1]
                or self.selected_anchor_sha256[index]
                != self.selected_anchor_sha256[index + 1]
            ):
                raise ValueError("Round 73 operational action pair differs")
        if self.dataset_sha256 != _dataset_sha256(self):
            raise ValueError("Round 73 operational dataset hash differs")


def _dataset_sha256(dataset: Round73OperationalDataset) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": dataset.schema_version,
                "study_id": dataset.study_id,
                "cohort_manifest_sha256": dataset.cohort_manifest_sha256,
                "target_study_manifest_sha256": (dataset.target_study_manifest_sha256),
                "feature_names_sha256": dataset.feature_names_sha256,
                "evaluation_contract_sha256": (ROUND73_EVALUATION_CONTRACT_SHA256),
                "feature_contract_sha256": ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
                "primary_entry_delay_ms": ROUND73_PRIMARY_ENTRY_DELAY_MS,
                "primary_horizon_ms": ROUND73_PRIMARY_HORIZON_MS,
                "primary_reference_notional": ROUND73_PRIMARY_REFERENCE_NOTIONAL,
                "target_result_observed": dataset.target_result_observed,
                "predictive_edge_claim": dataset.predictive_edge_claim,
                "profitability_claim": dataset.profitability_claim,
                "trading_authority": dataset.trading_authority,
            }
        ).encode("ascii")
    )
    for values in (
        dataset.run_id,
        dataset.symbol,
        dataset.role,
        dataset.side,
        dataset.outcome_reason,
        dataset.option_sha256,
        dataset.selected_anchor_sha256,
    ):
        _string_sequence_digest(digest, values)
    for array in (
        dataset.anchor_index,
        dataset.anchor_wall_ns,
        dataset.feature_values,
        dataset.outcome_status,
        dataset.binary_target,
        dataset.continuous_target_bps,
        dataset.completed_transaction,
    ):
        _array_digest(digest, array)
    return digest.hexdigest()


def _strict_reason_text(raw: object) -> str:
    text = str(raw)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Round 73 target reason JSON is not a list")
    return text


def _readonly_array(values: Sequence[object], dtype: object) -> np.ndarray:
    output = np.ascontiguousarray(values, dtype=dtype)
    output.setflags(write=False)
    return output


def _operational_dataset_from_rows(
    rows: Sequence[Sequence[object]],
    *,
    selected_study: str,
    cohort_manifest_sha256: str,
    target_study_manifest_sha256: str,
) -> Round73OperationalDataset:
    if not rows or len(rows) % 2:
        raise ValueError("Round 73 primary operational target rows are incomplete")

    run_ids: list[str] = []
    symbols: list[str] = []
    roles: list[str] = []
    sides: list[str] = []
    reasons: list[str] = []
    option_hashes: list[str] = []
    selected_anchor_hashes: list[str] = []
    anchor_indexes: list[int] = []
    anchor_walls: list[int] = []
    features: list[np.ndarray] = []
    statuses: list[int] = []
    binary_targets: list[float] = []
    continuous_targets: list[float] = []
    completed: list[bool] = []

    for row in rows:
        run_id = str(row[0])
        symbol = str(row[1])
        anchor_index = int(row[2])
        anchor_wall_ns = int(row[4])
        role = str(row[6])
        shock_ratio = float(row[7])
        shock_direction = int(row[8])
        shock_share = float(row[9])
        feature_hash = str(row[10])
        selected_anchor_hash = str(row[11])
        coverage_end_wall_ns = int(row[12])
        raw_features = np.asarray(row[13], dtype=np.float64)
        stored_vector_hash = str(row[14])
        side = str(row[15])
        reason_text = _strict_reason_text(row[17])
        option_hash = str(row[20])
        option_anchor_hash = str(row[21])
        cohort_option_hash = str(row[22])
        if (
            symbol not in IMPACT_CAPTURE_SYMBOLS
            or role not in _ROLE_NAMES
            or side not in ROUND73_ACTION_SIDES
            or raw_features.shape != (len(ROUND73_GRID_FEATURE_NAMES),)
            or not np.all(np.isfinite(raw_features))
            or _grid_vector_sha256(run_id, symbol, anchor_index, raw_features)
            != stored_vector_hash
            or stored_vector_hash != feature_hash
            or selected_anchor_hash != option_anchor_hash
            or _SHA256.fullmatch(selected_anchor_hash) is None
            or _SHA256.fullmatch(option_hash) is None
            or _SHA256.fullmatch(cohort_option_hash) is None
        ):
            raise ValueError("Round 73 operational source row identity differs")
        required_complete_wall_ns = (
            anchor_wall_ns
            + ROUND73_PRIMARY_ENTRY_DELAY_MS * 1_000_000
            + ROUND73_PRIMARY_HORIZON_MS * 1_000_000
            + 2 * ROUND73_TARGET_MAX_STATE_LATENESS_NS
        )
        boundary_censored = required_complete_wall_ns >= coverage_end_wall_ns
        outcome = classify_round73_operational_outcome(
            eligible=bool(row[16]),
            ineligible_reasons_json=reason_text,
            positive_net_payoff=(None if row[18] is None else bool(row[18])),
            net_payoff_bps=(None if row[19] is None else float(row[19])),
            deterministically_boundary_censored=boundary_censored,
        )
        aligned = action_align_round73_features(
            raw_features,
            side=side,
            shock_ratio=shock_ratio,
            shock_direction=shock_direction,
            shock_direction_taker_share=shock_share,
        )
        run_ids.append(run_id)
        symbols.append(symbol)
        roles.append(role)
        sides.append(side)
        reasons.append(outcome.reason)
        option_hashes.append(option_hash)
        selected_anchor_hashes.append(selected_anchor_hash)
        anchor_indexes.append(anchor_index)
        anchor_walls.append(anchor_wall_ns)
        features.append(aligned)
        statuses.append(outcome.status)
        binary_targets.append(outcome.binary_target)
        continuous_targets.append(outcome.continuous_target_bps)
        completed.append(outcome.completed_transaction)

    dataset = Round73OperationalDataset(
        schema_version=ROUND73_OPERATIONAL_DATASET_SCHEMA_VERSION,
        study_id=selected_study,
        cohort_manifest_sha256=cohort_manifest_sha256,
        target_study_manifest_sha256=target_study_manifest_sha256,
        feature_names_sha256=ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
        run_id=tuple(run_ids),
        symbol=tuple(symbols),
        role=tuple(roles),
        side=tuple(sides),
        outcome_reason=tuple(reasons),
        option_sha256=tuple(option_hashes),
        selected_anchor_sha256=tuple(selected_anchor_hashes),
        anchor_index=_readonly_array(anchor_indexes, np.int64),
        anchor_wall_ns=_readonly_array(anchor_walls, np.int64),
        feature_values=_readonly_array(features, np.float64),
        outcome_status=_readonly_array(statuses, np.int8),
        binary_target=_readonly_array(binary_targets, np.float64),
        continuous_target_bps=_readonly_array(continuous_targets, np.float64),
        completed_transaction=_readonly_array(completed, np.bool_),
        dataset_sha256="",
    )
    dataset = Round73OperationalDataset(
        **{**dataset.__dict__, "dataset_sha256": _dataset_sha256(dataset)}
    )
    dataset.validate()
    return dataset


def build_round73_operational_dataset(
    database: str | Path,
    *,
    study_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    target_study_audit_function: TargetStudyAuditFunction = (
        audit_round73_target_study
    ),
) -> Round73OperationalDataset:
    """Audit and adapt the sealed primary-scenario target study exactly once."""

    selected_study = str(study_id).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 operational dataset study ID is invalid")
    audit = target_study_audit_function(
        database,
        study_id=selected_study,
        memory_limit=memory_limit,
        threads=threads,
    )
    if getattr(audit, "passed", False) is not True:
        raise ValueError("Round 73 target-study audit did not pass")
    target_study_manifest_sha256 = str(
        getattr(audit, "target_study_manifest_sha256", "")
    )
    if _SHA256.fullmatch(target_study_manifest_sha256) is None:
        raise ValueError("Round 73 target-study audit hash is invalid")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        study_row = connection.execute(
            f"SELECT manifest_sha256 FROM {ROUND73_SHOCK_STUDY_TABLE} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        target_study_row = connection.execute(
            f"SELECT target_study_manifest_sha256 "
            f"FROM {ROUND73_TARGET_V2_STUDY_TABLE} WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        if study_row is None or target_study_row is None:
            raise ValueError("Round 73 sealed study manifests are missing")
        cohort_manifest_sha256 = str(study_row[0])
        if (
            _SHA256.fullmatch(cohort_manifest_sha256) is None
            or str(target_study_row[0]) != target_study_manifest_sha256
        ):
            raise ValueError("Round 73 sealed study manifest identity differs")
        rows = connection.execute(
            f"""
            SELECT s.run_id, s.symbol, s.anchor_index, s.anchor_monotonic_ns,
                   s.anchor_wall_ns, s.utc_day, s.role, s.shock_ratio,
                   s.shock_direction, s.shock_direction_taker_share,
                   s.feature_vector_sha256, s.selected_anchor_sha256,
                   c.coverage_end_wall_ns, v.feature_values, v.vector_sha256,
                   o.side, o.eligible, o.ineligible_reasons_json,
                   o.positive_net_payoff, o.net_payoff_bps,
                   o.option_sha256, o.selected_anchor_sha256,
                   o.cohort_option_sha256
            FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
            JOIN {ROUND73_CORPUS_RUN_TABLE} c USING (run_id)
            JOIN {ROUND73_GRID_VECTOR_TABLE} v
              USING (run_id, symbol, anchor_index)
            JOIN {ROUND73_TARGET_V2_OPTION_TABLE} o
              ON o.study_id = s.study_id AND o.run_id = s.run_id
             AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
            WHERE s.study_id = ? AND o.entry_delay_ms = ?
              AND o.horizon_ms = ? AND o.reference_quote_notional = ?
            ORDER BY s.anchor_wall_ns, s.run_id, s.symbol, s.anchor_index,
                     CASE o.side WHEN 'long' THEN 0 WHEN 'short' THEN 1 ELSE 2 END
            """,
            [
                selected_study,
                ROUND73_PRIMARY_ENTRY_DELAY_MS,
                ROUND73_PRIMARY_HORIZON_MS,
                ROUND73_PRIMARY_REFERENCE_NOTIONAL,
            ],
        ).fetchall()
    return _operational_dataset_from_rows(
        rows,
        selected_study=selected_study,
        cohort_manifest_sha256=cohort_manifest_sha256,
        target_study_manifest_sha256=target_study_manifest_sha256,
    )


def build_round73_staged_operational_dataset(
    database: str | Path,
    *,
    study_id: str,
    role_scope: str,
    pretest_manifest_sha256: str | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    development_seal_function: TargetStudySealFunction = (
        seal_round73_development_targets
    ),
    test_seal_function: TargetStudySealFunction = seal_round73_test_targets,
) -> Round73OperationalDataset:
    """Build one physically staged dataset after its exact target seal passes."""

    selected_study = str(study_id).strip().lower()
    selected_scope = str(role_scope).strip().lower()
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 staged dataset study ID is invalid")
    if selected_scope == "development":
        if pretest_manifest_sha256 is not None:
            raise ValueError(
                "Round 73 development dataset cannot receive a pretest hash"
            )
        seal = development_seal_function(
            database,
            study_id=selected_study,
            memory_limit=memory_limit,
            threads=threads,
        )
        target_study_manifest_sha256 = str(
            getattr(seal, "development_study_manifest_sha256", "")
        )
        seal_table = ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE
        roles = ("training", "tuning")
    elif selected_scope == "test":
        selected_pretest = str(pretest_manifest_sha256 or "").strip().lower()
        if _SHA256.fullmatch(selected_pretest) is None:
            raise ValueError("Round 73 test dataset requires a pretest hash")
        seal = test_seal_function(
            database,
            study_id=selected_study,
            pretest_manifest_sha256=selected_pretest,
            memory_limit=memory_limit,
            threads=threads,
        )
        target_study_manifest_sha256 = str(
            getattr(seal, "test_study_manifest_sha256", "")
        )
        seal_table = ROUND73_TARGET_V3_TEST_STUDY_TABLE
        roles = ("test",)
    else:
        raise ValueError(
            "Round 73 staged dataset role scope must be development or test"
        )
    if _SHA256.fullmatch(target_study_manifest_sha256) is None:
        raise ValueError("Round 73 staged target-study seal hash is invalid")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        study_row = connection.execute(
            f"SELECT manifest_sha256 FROM {ROUND73_SHOCK_STUDY_TABLE} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        target_study_row = connection.execute(
            f"SELECT cohort_manifest_sha256, manifest_sha256 FROM {seal_table} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        if study_row is None or target_study_row is None:
            raise ValueError("Round 73 staged study manifests are missing")
        cohort_manifest_sha256 = str(study_row[0])
        if (
            _SHA256.fullmatch(cohort_manifest_sha256) is None
            or str(target_study_row[0]) != cohort_manifest_sha256
            or str(target_study_row[1]) != target_study_manifest_sha256
        ):
            raise ValueError("Round 73 staged study manifest identity differs")
        rows = connection.execute(
            f"""
            SELECT s.run_id, s.symbol, s.anchor_index, s.anchor_monotonic_ns,
                   s.anchor_wall_ns, s.utc_day, s.role, s.shock_ratio,
                   s.shock_direction, s.shock_direction_taker_share,
                   s.feature_vector_sha256, s.selected_anchor_sha256,
                   c.coverage_end_wall_ns, v.feature_values, v.vector_sha256,
                   o.side, o.eligible, o.ineligible_reasons_json,
                   o.positive_net_payoff, o.net_payoff_bps,
                   o.option_sha256, o.selected_anchor_sha256,
                   o.cohort_option_sha256
            FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
            JOIN {ROUND73_CORPUS_RUN_TABLE} c USING (run_id)
            JOIN {ROUND73_GRID_VECTOR_TABLE} v
              USING (run_id, symbol, anchor_index)
            JOIN {ROUND73_TARGET_V3_OPTION_TABLE} o
              ON o.study_id = s.study_id AND o.run_id = s.run_id
             AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
            WHERE s.study_id = ? AND s.role IN (
                {",".join("?" for _ in roles)}
            ) AND o.entry_delay_ms = ? AND o.horizon_ms = ?
              AND o.reference_quote_notional = ?
            ORDER BY s.anchor_wall_ns, s.run_id, s.symbol, s.anchor_index,
                     CASE o.side WHEN 'long' THEN 0 WHEN 'short' THEN 1 ELSE 2 END
            """,
            [
                selected_study,
                *roles,
                ROUND73_PRIMARY_ENTRY_DELAY_MS,
                ROUND73_PRIMARY_HORIZON_MS,
                ROUND73_PRIMARY_REFERENCE_NOTIONAL,
            ],
        ).fetchall()
    dataset = _operational_dataset_from_rows(
        rows,
        selected_study=selected_study,
        cohort_manifest_sha256=cohort_manifest_sha256,
        target_study_manifest_sha256=target_study_manifest_sha256,
    )
    if set(dataset.role) - set(roles):
        raise ValueError("Round 73 staged dataset crossed its role scope")
    return dataset


__all__ = [
    "ROUND73_OBSERVED_STATUS",
    "ROUND73_OPERATIONAL_DATASET_SCHEMA_VERSION",
    "ROUND73_OPERATIONAL_STATUS_NAMES",
    "ROUND73_POST_ENTRY_UNRESOLVED_STATUS",
    "ROUND73_PRE_ENTRY_ABORT_STATUS",
    "ROUND73_PRIMARY_ENTRY_DELAY_MS",
    "ROUND73_PRIMARY_HORIZON_MS",
    "ROUND73_PRIMARY_REFERENCE_NOTIONAL",
    "ROUND73_RIGHT_CENSORED_STATUS",
    "Round73OperationalDataset",
    "Round73OperationalOutcome",
    "build_round73_operational_dataset",
    "build_round73_staged_operational_dataset",
    "classify_round73_operational_outcome",
]
