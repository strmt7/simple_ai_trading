"""Bounded, symbol-scoped model inputs for the staged Round 73 study."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .impact_absorption_cohort import (
    ROUND73_SHOCK_ANCHOR_TABLE,
    ROUND73_SHOCK_STUDY_TABLE,
)
from .impact_absorption_corpus import ROUND73_CORPUS_RUN_TABLE
from .impact_absorption_grid import ROUND73_GRID_FEATURE_NAMES
from .impact_absorption_grid_store import ROUND73_GRID_VECTOR_TABLE
from .impact_absorption_model_dataset import (
    ROUND73_OBSERVED_STATUS,
    ROUND73_OPERATIONAL_STATUS_NAMES,
    ROUND73_PRE_ENTRY_ABORT_STATUS,
    ROUND73_PRIMARY_ENTRY_DELAY_MS,
    ROUND73_PRIMARY_HORIZON_MS,
    ROUND73_PRIMARY_REFERENCE_NOTIONAL,
    classify_round73_operational_outcome,
)
from .impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
    action_align_round73_feature_batch,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_store_v2 import _STUDY_ID
from .impact_absorption_target_store_v3 import (
    ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE,
    ROUND73_TARGET_V3_OPTION_TABLE,
    ROUND73_TARGET_V3_TEST_STUDY_TABLE,
    seal_round73_development_targets,
    seal_round73_test_targets,
)
from .impact_absorption_targets import ROUND73_TARGET_MAX_STATE_LATENESS_NS


ROUND73_MODEL_SLICE_SCHEMA_VERSION = "round-073-symbol-model-slice-v1"
ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES = 3 * 1024**3
ROUND73_MODEL_SLICE_FETCH_ROWS = 4_096

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_ROLE_NAMES = ("training", "tuning", "test")
_ROLE_TO_CODE = {name: index for index, name in enumerate(_ROLE_NAMES)}
_REASON_NAMES = (
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
_REASON_TO_CODE = {name: index for index, name in enumerate(_REASON_NAMES)}
_ROW_FIXED_BYTES = (
    8  # anchor index
    + 8  # anchor wall time
    + 16  # binary run id
    + 32  # binary option hash
    + 32  # binary selected-anchor hash
    + 4  # role, side, status, reason
    + 8  # binary target
    + 8  # continuous target
    + 1  # completed transaction
)

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


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _binary_digest(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


@dataclass(frozen=True)
class Round73SymbolModelSlice:
    study_id: str
    role_scope: str
    symbol: str
    cohort_manifest_sha256: str
    target_study_manifest_sha256: str
    feature_names_sha256: str
    anchor_index: np.ndarray
    anchor_wall_ns: np.ndarray
    run_id_binary: np.ndarray
    option_sha256_binary: np.ndarray
    selected_anchor_sha256_binary: np.ndarray
    role_code: np.ndarray
    side_sign: np.ndarray
    feature_values: np.ndarray
    outcome_status: np.ndarray
    outcome_reason_code: np.ndarray
    binary_target: np.ndarray
    continuous_target_bps: np.ndarray
    completed_transaction: np.ndarray
    source_rows_sha256: str
    estimated_allocation_bytes: int

    @property
    def rows(self) -> int:
        return int(self.anchor_index.size)

    @property
    def model_label_mask(self) -> np.ndarray:
        return _readonly(
            np.isin(
                self.outcome_status,
                (ROUND73_OBSERVED_STATUS, ROUND73_PRE_ENTRY_ABORT_STATUS),
            )
        )

    def role_mask(self, role: str) -> np.ndarray:
        try:
            code = _ROLE_TO_CODE[str(role)]
        except KeyError as exc:
            raise ValueError("Round 73 model-slice role is invalid") from exc
        return _readonly(self.role_code == code)

    def validate(self) -> None:
        rows = self.rows
        one_dimensional = (
            self.anchor_index,
            self.anchor_wall_ns,
            self.run_id_binary,
            self.option_sha256_binary,
            self.selected_anchor_sha256_binary,
            self.role_code,
            self.side_sign,
            self.outcome_status,
            self.outcome_reason_code,
            self.binary_target,
            self.continuous_target_bps,
            self.completed_transaction,
        )
        expected_roles = (
            {_ROLE_TO_CODE["training"], _ROLE_TO_CODE["tuning"]}
            if self.role_scope == "development"
            else {_ROLE_TO_CODE["test"]}
        )
        label_mask = self.model_label_mask
        if (
            _STUDY_ID.fullmatch(self.study_id) is None
            or self.role_scope not in {"development", "test"}
            or self.symbol not in IMPACT_CAPTURE_SYMBOLS
            or _SHA256.fullmatch(self.cohort_manifest_sha256) is None
            or _SHA256.fullmatch(self.target_study_manifest_sha256) is None
            or self.feature_names_sha256 != ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256
            or rows <= 0
            or any(value.shape != (rows,) for value in one_dimensional)
            or self.feature_values.shape
            != (rows, len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES))
            or self.feature_values.dtype != np.float32
            or any(
                value.flags.writeable
                for value in (*one_dimensional, self.feature_values)
            )
            or not np.all(np.isfinite(self.feature_values))
            or set(np.unique(self.role_code).tolist()) - expected_roles
            or np.any((self.side_sign != 1) & (self.side_sign != -1))
            or np.any(
                (self.outcome_status < 0)
                | (self.outcome_status >= len(ROUND73_OPERATIONAL_STATUS_NAMES))
            )
            or np.any(
                (self.outcome_reason_code < 0)
                | (self.outcome_reason_code >= len(_REASON_NAMES))
            )
            or np.any(~np.isfinite(self.binary_target[label_mask]))
            or np.any(~np.isfinite(self.continuous_target_bps[label_mask]))
            or np.any(np.isfinite(self.binary_target[~label_mask]))
            or np.any(np.isfinite(self.continuous_target_bps[~label_mask]))
            or np.any(
                self.completed_transaction
                != (self.outcome_status == ROUND73_OBSERVED_STATUS)
            )
            or _SHA256.fullmatch(self.source_rows_sha256) is None
            or self.estimated_allocation_bytes
            != sum(value.nbytes for value in (*one_dimensional, self.feature_values))
        ):
            raise ValueError("Round 73 symbol model slice is invalid")
        if rows % 2:
            raise ValueError("Round 73 symbol model slice action pairs are incomplete")
        even = slice(0, None, 2)
        odd = slice(1, None, 2)
        if (
            not np.array_equal(self.anchor_index[even], self.anchor_index[odd])
            or not np.array_equal(self.anchor_wall_ns[even], self.anchor_wall_ns[odd])
            or not np.array_equal(self.run_id_binary[even], self.run_id_binary[odd])
            or not np.array_equal(
                self.selected_anchor_sha256_binary[even],
                self.selected_anchor_sha256_binary[odd],
            )
            or np.any(self.side_sign[even] != 1)
            or np.any(self.side_sign[odd] != -1)
        ):
            raise ValueError("Round 73 symbol model slice action pair differs")

    def identity(self) -> Mapping[str, object]:
        self.validate()
        return {
            "schema_version": ROUND73_MODEL_SLICE_SCHEMA_VERSION,
            "study_id": self.study_id,
            "role_scope": self.role_scope,
            "symbol": self.symbol,
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "target_study_manifest_sha256": self.target_study_manifest_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "rows": self.rows,
            "source_rows_sha256": self.source_rows_sha256,
            "feature_values_sha256": _binary_digest(self.feature_values),
            "binary_target_sha256": _binary_digest(self.binary_target),
            "continuous_target_bps_sha256": _binary_digest(self.continuous_target_bps),
            "estimated_allocation_bytes": self.estimated_allocation_bytes,
        }


def _allocation_bytes(rows: int) -> int:
    return int(rows) * (
        len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES) * np.dtype(np.float32).itemsize
        + _ROW_FIXED_BYTES
    )


def _validate_memory_budget(memory_budget_bytes: int, *, rows: int) -> int:
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("Round 73 model-slice memory budget must be positive")
    required = _allocation_bytes(rows)
    if required > memory_budget_bytes:
        raise MemoryError(
            "Round 73 symbol model slice exceeds its explicit memory budget: "
            f"required={required} budget={memory_budget_bytes}"
        )
    return required


def _scope_contract(
    role_scope: str,
) -> tuple[tuple[str, ...], str, str]:
    selected = str(role_scope).strip().lower()
    if selected == "development":
        return (
            ("training", "tuning"),
            ROUND73_TARGET_V3_DEVELOPMENT_STUDY_TABLE,
            "development_study_manifest_sha256",
        )
    if selected == "test":
        return (
            ("test",),
            ROUND73_TARGET_V3_TEST_STUDY_TABLE,
            "test_study_manifest_sha256",
        )
    raise ValueError("Round 73 model-slice role scope must be development or test")


def _seal_scope(
    database: str | Path,
    *,
    study_id: str,
    role_scope: str,
    pretest_manifest_sha256: str | None,
    memory_limit: str,
    threads: int,
    development_seal_function: TargetStudySealFunction,
    test_seal_function: TargetStudySealFunction,
) -> tuple[tuple[str, ...], str, str]:
    roles, table, attribute = _scope_contract(role_scope)
    if role_scope == "development":
        if pretest_manifest_sha256 is not None:
            raise ValueError("Round 73 development slice cannot receive a pretest hash")
        seal = development_seal_function(
            database,
            study_id=study_id,
            memory_limit=memory_limit,
            threads=threads,
        )
    else:
        selected_pretest = str(pretest_manifest_sha256 or "").strip().lower()
        if _SHA256.fullmatch(selected_pretest) is None:
            raise ValueError("Round 73 test slice requires a pretest hash")
        seal = test_seal_function(
            database,
            study_id=study_id,
            pretest_manifest_sha256=selected_pretest,
            memory_limit=memory_limit,
            threads=threads,
        )
    manifest_sha256 = str(getattr(seal, attribute, ""))
    if _SHA256.fullmatch(manifest_sha256) is None:
        raise ValueError("Round 73 model-slice target seal hash is invalid")
    return roles, table, manifest_sha256


def _allocate(rows: int) -> dict[str, np.ndarray]:
    return {
        "anchor_index": np.empty(rows, dtype=np.int64),
        "anchor_wall_ns": np.empty(rows, dtype=np.int64),
        "run_id_binary": np.empty(rows, dtype="S16"),
        "option_sha256_binary": np.empty(rows, dtype="S32"),
        "selected_anchor_sha256_binary": np.empty(rows, dtype="S32"),
        "role_code": np.empty(rows, dtype=np.uint8),
        "side_sign": np.empty(rows, dtype=np.int8),
        "feature_values": np.empty(
            (rows, len(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)), dtype=np.float32
        ),
        "outcome_status": np.empty(rows, dtype=np.uint8),
        "outcome_reason_code": np.empty(rows, dtype=np.uint8),
        "binary_target": np.empty(rows, dtype=np.float64),
        "continuous_target_bps": np.empty(rows, dtype=np.float64),
        "completed_transaction": np.empty(rows, dtype=np.bool_),
    }


def _decode_hash(value: object, pattern: re.Pattern[str], label: str) -> bytes:
    text = str(value).strip().lower()
    if pattern.fullmatch(text) is None:
        raise ValueError(f"Round 73 model-slice {label} is invalid")
    return bytes.fromhex(text)


def _fill_batch(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Sequence[object]],
    *,
    start: int,
    symbol: str,
    source_digest: object,
) -> int:
    count = len(rows)
    stop = start + count
    raw_features = np.empty((count, len(ROUND73_GRID_FEATURE_NAMES)), dtype=np.float64)
    sides = np.empty(count, dtype=np.int8)
    shock_ratios = np.empty(count, dtype=np.float64)
    shock_directions = np.empty(count, dtype=np.int8)
    shock_shares = np.empty(count, dtype=np.float64)
    for offset, row in enumerate(rows):
        index = start + offset
        run_id = str(row[0]).strip().lower()
        anchor_index = int(row[1])
        anchor_wall_ns = int(row[2])
        role = str(row[3])
        shock_ratios[offset] = float(row[4])
        shock_directions[offset] = int(row[5])
        shock_shares[offset] = float(row[6])
        feature_hash = str(row[7]).strip().lower()
        selected_anchor_hash = str(row[8]).strip().lower()
        coverage_end_wall_ns = int(row[9])
        vector = np.asarray(row[10], dtype=np.float64)
        vector_hash = str(row[11]).strip().lower()
        side = str(row[12])
        sides[offset] = 1 if side == "long" else -1 if side == "short" else 0
        reason_text = str(row[14])
        option_hash = str(row[17]).strip().lower()
        option_anchor_hash = str(row[18]).strip().lower()
        if (
            role not in _ROLE_TO_CODE
            or vector.shape != (len(ROUND73_GRID_FEATURE_NAMES),)
            or not np.all(np.isfinite(vector))
            or _SHA256.fullmatch(feature_hash) is None
            or feature_hash != vector_hash
            or _grid_vector_sha256(run_id, symbol, anchor_index, vector) != vector_hash
            or selected_anchor_hash != option_anchor_hash
            or _SHA256.fullmatch(selected_anchor_hash) is None
            or _SHA256.fullmatch(option_hash) is None
        ):
            raise ValueError("Round 73 model-slice source row identity differs")
        required_complete_wall_ns = (
            anchor_wall_ns
            + ROUND73_PRIMARY_ENTRY_DELAY_MS * 1_000_000
            + ROUND73_PRIMARY_HORIZON_MS * 1_000_000
            + 2 * ROUND73_TARGET_MAX_STATE_LATENESS_NS
        )
        outcome = classify_round73_operational_outcome(
            eligible=bool(row[13]),
            ineligible_reasons_json=reason_text,
            positive_net_payoff=None if row[15] is None else bool(row[15]),
            net_payoff_bps=None if row[16] is None else float(row[16]),
            deterministically_boundary_censored=(
                required_complete_wall_ns >= coverage_end_wall_ns
            ),
        )
        try:
            reason_code = _REASON_TO_CODE[outcome.reason]
        except KeyError as exc:
            raise ValueError("Round 73 model-slice outcome reason is unknown") from exc
        arrays["anchor_index"][index] = anchor_index
        arrays["anchor_wall_ns"][index] = anchor_wall_ns
        arrays["run_id_binary"][index] = _decode_hash(run_id, _RUN_ID, "run ID")
        arrays["option_sha256_binary"][index] = bytes.fromhex(option_hash)
        arrays["selected_anchor_sha256_binary"][index] = bytes.fromhex(
            selected_anchor_hash
        )
        arrays["role_code"][index] = _ROLE_TO_CODE[role]
        arrays["side_sign"][index] = sides[offset]
        arrays["outcome_status"][index] = outcome.status
        arrays["outcome_reason_code"][index] = reason_code
        arrays["binary_target"][index] = outcome.binary_target
        arrays["continuous_target_bps"][index] = outcome.continuous_target_bps
        arrays["completed_transaction"][index] = outcome.completed_transaction
        raw_features[offset] = vector
        source_digest.update(bytes.fromhex(option_hash))
        source_digest.update(bytes.fromhex(vector_hash))
    aligned = action_align_round73_feature_batch(
        raw_features,
        side=sides,
        shock_ratio=shock_ratios,
        shock_direction=shock_directions,
        shock_direction_taker_share=shock_shares,
        dtype=np.float32,
    )
    arrays["feature_values"][start:stop] = aligned
    return stop


def _load_symbol_slice(
    connection,
    *,
    study_id: str,
    role_scope: str,
    roles: tuple[str, ...],
    symbol: str,
    cohort_manifest_sha256: str,
    target_study_manifest_sha256: str,
    memory_budget_bytes: int,
) -> Round73SymbolModelSlice:
    placeholders = ",".join("?" for _ in roles)
    parameters: list[object] = [
        study_id,
        symbol,
        *roles,
        ROUND73_PRIMARY_ENTRY_DELAY_MS,
        ROUND73_PRIMARY_HORIZON_MS,
        ROUND73_PRIMARY_REFERENCE_NOTIONAL,
    ]
    where = (
        "s.study_id = ? AND s.symbol = ? "
        f"AND s.role IN ({placeholders}) AND o.entry_delay_ms = ? "
        "AND o.horizon_ms = ? AND o.reference_quote_notional = ?"
    )
    row_count = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
            JOIN {ROUND73_TARGET_V3_OPTION_TABLE} o
              ON o.study_id = s.study_id AND o.run_id = s.run_id
             AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
            WHERE {where}
            """,
            parameters,
        ).fetchone()[0]
    )
    if row_count <= 0 or row_count % 2:
        raise ValueError("Round 73 symbol model-slice rows are incomplete")
    estimated = _validate_memory_budget(memory_budget_bytes, rows=row_count)
    arrays = _allocate(row_count)
    cursor = connection.execute(
        f"""
        SELECT s.run_id, s.anchor_index, s.anchor_wall_ns, s.role,
               s.shock_ratio, s.shock_direction,
               s.shock_direction_taker_share, s.feature_vector_sha256,
               s.selected_anchor_sha256, c.coverage_end_wall_ns,
               v.feature_values, v.vector_sha256, o.side, o.eligible,
               o.ineligible_reasons_json, o.positive_net_payoff,
               o.net_payoff_bps, o.option_sha256, o.selected_anchor_sha256
        FROM {ROUND73_SHOCK_ANCHOR_TABLE} s
        JOIN {ROUND73_CORPUS_RUN_TABLE} c USING (run_id)
        JOIN {ROUND73_GRID_VECTOR_TABLE} v
          USING (run_id, symbol, anchor_index)
        JOIN {ROUND73_TARGET_V3_OPTION_TABLE} o
          ON o.study_id = s.study_id AND o.run_id = s.run_id
         AND o.symbol = s.symbol AND o.anchor_index = s.anchor_index
        WHERE {where}
        ORDER BY s.anchor_wall_ns, s.run_id, s.anchor_index,
                 CASE o.side WHEN 'long' THEN 0 WHEN 'short' THEN 1 ELSE 2 END
        """,
        parameters,
    )
    source_digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": ROUND73_MODEL_SLICE_SCHEMA_VERSION,
                "study_id": study_id,
                "role_scope": role_scope,
                "symbol": symbol,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "target_study_manifest_sha256": target_study_manifest_sha256,
                "row_count": row_count,
            }
        ).encode("ascii")
    )
    offset = 0
    while True:
        batch = cursor.fetchmany(ROUND73_MODEL_SLICE_FETCH_ROWS)
        if not batch:
            break
        offset = _fill_batch(
            arrays,
            batch,
            start=offset,
            symbol=symbol,
            source_digest=source_digest,
        )
    if offset != row_count:
        raise ValueError("Round 73 model-slice row count changed during read")
    frozen = {name: _readonly(value) for name, value in arrays.items()}
    result = Round73SymbolModelSlice(
        study_id=study_id,
        role_scope=role_scope,
        symbol=symbol,
        cohort_manifest_sha256=cohort_manifest_sha256,
        target_study_manifest_sha256=target_study_manifest_sha256,
        feature_names_sha256=ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
        source_rows_sha256=source_digest.hexdigest(),
        estimated_allocation_bytes=estimated,
        **frozen,
    )
    result.validate()
    return result


def iter_round73_staged_symbol_slices(
    database: str | Path,
    *,
    study_id: str,
    role_scope: str,
    pretest_manifest_sha256: str | None = None,
    symbols: Sequence[str] = IMPACT_CAPTURE_SYMBOLS,
    memory_budget_bytes: int = ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES,
    memory_limit: str = "2GB",
    threads: int = 2,
    development_seal_function: TargetStudySealFunction = (
        seal_round73_development_targets
    ),
    test_seal_function: TargetStudySealFunction = seal_round73_test_targets,
) -> Iterator[Round73SymbolModelSlice]:
    """Seal once, then materialize one bounded symbol matrix at a time."""

    selected_study = str(study_id).strip().lower()
    selected_scope = str(role_scope).strip().lower()
    selected_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if _STUDY_ID.fullmatch(selected_study) is None:
        raise ValueError("Round 73 model-slice study ID is invalid")
    if (
        not selected_symbols
        or len(selected_symbols) != len(set(selected_symbols))
        or any(symbol not in IMPACT_CAPTURE_SYMBOLS for symbol in selected_symbols)
    ):
        raise ValueError("Round 73 model-slice symbols are invalid")
    roles, seal_table, target_manifest_sha256 = _seal_scope(
        database,
        study_id=selected_study,
        role_scope=selected_scope,
        pretest_manifest_sha256=pretest_manifest_sha256,
        memory_limit=memory_limit,
        threads=threads,
        development_seal_function=development_seal_function,
        test_seal_function=test_seal_function,
    )
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
        seal_row = connection.execute(
            f"SELECT cohort_manifest_sha256, manifest_sha256 FROM {seal_table} "
            "WHERE study_id = ?",
            [selected_study],
        ).fetchone()
        if study_row is None or seal_row is None:
            raise ValueError("Round 73 model-slice sealed manifests are missing")
        cohort_manifest_sha256 = str(study_row[0])
        if (
            _SHA256.fullmatch(cohort_manifest_sha256) is None
            or str(seal_row[0]) != cohort_manifest_sha256
            or str(seal_row[1]) != target_manifest_sha256
        ):
            raise ValueError("Round 73 model-slice seal identity differs")
        for symbol in selected_symbols:
            yield _load_symbol_slice(
                connection,
                study_id=selected_study,
                role_scope=selected_scope,
                roles=roles,
                symbol=symbol,
                cohort_manifest_sha256=cohort_manifest_sha256,
                target_study_manifest_sha256=target_manifest_sha256,
                memory_budget_bytes=memory_budget_bytes,
            )


__all__ = [
    "ROUND73_MODEL_SLICE_DEFAULT_MEMORY_BUDGET_BYTES",
    "ROUND73_MODEL_SLICE_FETCH_ROWS",
    "ROUND73_MODEL_SLICE_SCHEMA_VERSION",
    "Round73SymbolModelSlice",
    "iter_round73_staged_symbol_slices",
]
