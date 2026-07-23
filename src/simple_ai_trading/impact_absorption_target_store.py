"""Transactional exact-wire target replay for the Round 73 causal grid."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import time

import duckdb
import numpy as np

from .impact_absorption import (
    L2BookState,
    SynchronizedDepthBook,
    parse_mark_price,
    validate_combined_stream_name,
)
from .impact_absorption_corpus import (
    ROUND73_CORPUS_RUN_TABLE,
    audit_round73_corpus_manifest,
)
from .impact_absorption_grid_store import (
    ROUND73_GRID_ANCHOR_TABLE,
    ROUND73_GRID_MANIFEST_TABLE,
    audit_round73_causal_grid,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
    iter_impact_capture_v9_records,
    load_impact_capture_v9_preflight,
)
from .impact_absorption_targets import (
    ROUND73_TARGET_CONTRACT_SHA256,
    ROUND73_TARGET_ENTRY_DELAYS_MS,
    ROUND73_TARGET_HORIZONS_MS,
    ROUND73_TARGET_LEVELS,
    ROUND73_TARGET_MAX_STATE_LATENESS_NS,
    ROUND73_TARGET_REFERENCE_NOTIONALS,
    ROUND73_TARGET_SCHEMA_VERSION,
    ROUND73_TARGET_SIDES,
    Round73BookWalk,
    Round73MarketQuantityRules,
    round73_target_payoff,
    walk_round73_book,
)


ROUND73_TARGET_OPTION_TABLE = "impact_target_option_v1"
ROUND73_TARGET_MANIFEST_TABLE = "impact_target_run_manifest_v1"
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INSERT_BATCH_SIZE = 32_768
_FETCH_BATCH_SIZE = 8_192
_EXPECTED_OPTIONS_PER_ANCHOR = (
    len(ROUND73_TARGET_ENTRY_DELAYS_MS)
    * len(ROUND73_TARGET_HORIZONS_MS)
    * len(ROUND73_TARGET_REFERENCE_NOTIONALS)
    * len(ROUND73_TARGET_SIDES)
)
_INELIGIBLE_BITS = {
    "quantity_filter": 1 << 0,
    "entry_state_late": 1 << 1,
    "entry_capacity": 1 << 2,
    "entry_minimum_notional": 1 << 3,
    "funding_boundary": 1 << 4,
    "path_capacity": 1 << 5,
    "exit_state_late": 1 << 6,
    "exit_capacity": 1 << 7,
    "coverage_end": 1 << 8,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _strict_json_object(raw_text: str, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key is forbidden in {label}: {key}")
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validated_run_id(value: str) -> str:
    selected = str(value).strip().lower()
    if _RUN_ID.fullmatch(selected) is None:
        raise ValueError("Round 73 target run ID must be 32 lowercase hex characters")
    return selected


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _ineligible_mask(reason: str) -> int:
    try:
        return _INELIGIBLE_BITS[reason]
    except KeyError as exc:
        raise ValueError(f"unsupported Round 73 target reason: {reason}") from exc


@dataclass(frozen=True)
class _Anchor:
    symbol: str
    anchor_index: int
    decision_monotonic_ns: int
    decision_wall_ns: int
    source_max_received_monotonic_ns: int


@dataclass(frozen=True)
class _Decision:
    anchor: _Anchor
    book_received_monotonic_ns: int
    book_state: L2BookState
    quantities: tuple[tuple[float, float | None], ...]


@dataclass(frozen=True)
class _PendingEntry:
    decision: _Decision
    entry_delay_ms: int
    requested_entry_monotonic_ns: int


@dataclass
class _ActivePosition:
    identifier: int
    decision: _Decision
    entry_delay_ms: int
    horizon_ms: int
    reference_quote_notional: float
    side: str
    requested_entry_monotonic_ns: int
    actual_entry_monotonic_ns: int
    requested_exit_monotonic_ns: int
    base_quantity: float
    entry_walk: Round73BookWalk
    entry_update_id: int
    minimum_net_payoff_bps: float
    maximum_net_payoff_bps: float
    maximum_spread_bps: float
    minimum_exit_side_capacity_ratio: float


@dataclass(frozen=True)
class Round73TargetOption:
    run_id: str
    symbol: str
    anchor_index: int
    entry_delay_ms: int
    horizon_ms: int
    reference_quote_notional: float
    side: str
    eligible: bool
    ineligible_reason_mask: int
    ineligible_reasons_json: str
    decision_monotonic_ns: int
    decision_book_received_monotonic_ns: int
    requested_entry_monotonic_ns: int
    actual_entry_monotonic_ns: int | None
    entry_state_lateness_ms: float | None
    requested_exit_monotonic_ns: int | None
    actual_exit_monotonic_ns: int | None
    exit_state_lateness_ms: float | None
    base_quantity: float | None
    decision_mid: float
    entry_average_price: float | None
    entry_quote_notional: float | None
    exit_average_price: float | None
    exit_quote_notional: float | None
    gross_payoff_quote: float | None
    charge_quote: float | None
    net_payoff_quote: float | None
    net_payoff_bps: float | None
    positive_net_payoff: bool | None
    maximum_adverse_excursion_bps: float | None
    maximum_favorable_excursion_bps: float | None
    maximum_spread_bps: float | None
    minimum_exit_side_capacity_ratio: float | None
    entry_update_id: int | None
    exit_update_id: int | None
    option_sha256: str

    @property
    def key(self) -> tuple[str, int, int, int, float, str]:
        return (
            self.symbol,
            self.anchor_index,
            self.entry_delay_ms,
            self.horizon_ms,
            self.reference_quote_notional,
            self.side,
        )

    def values_without_hash(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())[:-1]

    def as_row(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


def _option_hash(values_without_hash: Sequence[object]) -> str:
    return _sha256_text(_canonical_json(list(values_without_hash)))


def _make_option(**values: object) -> Round73TargetOption:
    values_without_hash = tuple(values[name] for name in _OPTION_COLUMNS[:-1])
    return Round73TargetOption(
        **values,  # type: ignore[arg-type]
        option_sha256=_option_hash(values_without_hash),
    )


_OPTION_COLUMNS = tuple(Round73TargetOption.__dataclass_fields__)
_OUTCOME_FIELDS = (
    "exit_average_price",
    "exit_quote_notional",
    "gross_payoff_quote",
    "charge_quote",
    "net_payoff_quote",
    "net_payoff_bps",
    "positive_net_payoff",
    "maximum_adverse_excursion_bps",
    "maximum_favorable_excursion_bps",
    "maximum_spread_bps",
    "minimum_exit_side_capacity_ratio",
    "exit_update_id",
)


def _option_invariant_errors(option: Round73TargetOption) -> tuple[str, ...]:
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    require(option.run_id == option.run_id.strip().lower(), "run_id")
    require(option.symbol in IMPACT_CAPTURE_SYMBOLS, "symbol")
    require(option.anchor_index >= 0, "anchor_index")
    require(option.entry_delay_ms in ROUND73_TARGET_ENTRY_DELAYS_MS, "entry_delay")
    require(option.horizon_ms in ROUND73_TARGET_HORIZONS_MS, "horizon")
    require(
        option.reference_quote_notional in ROUND73_TARGET_REFERENCE_NOTIONALS,
        "reference_notional",
    )
    require(option.side in ROUND73_TARGET_SIDES, "side")
    require(
        option.decision_book_received_monotonic_ns < option.decision_monotonic_ns,
        "decision_book_future",
    )
    require(
        option.requested_entry_monotonic_ns
        == option.decision_monotonic_ns + option.entry_delay_ms * 1_000_000,
        "entry_request_time",
    )
    require(math.isfinite(option.decision_mid) and option.decision_mid > 0, "mid")
    reasons = json.loads(option.ineligible_reasons_json)
    decoded = [
        name for name, bit in _INELIGIBLE_BITS.items() if option.ineligible_reason_mask & bit
    ]
    require(reasons == decoded, "reason_identity")
    require(option.eligible == (option.ineligible_reason_mask == 0), "eligibility")
    require(
        option.option_sha256 == _option_hash(option.values_without_hash()),
        "option_hash",
    )
    if option.actual_entry_monotonic_ns is not None:
        require(
            option.actual_entry_monotonic_ns >= option.requested_entry_monotonic_ns,
            "entry_before_request",
        )
        require(option.entry_state_lateness_ms is not None, "entry_lateness_missing")
        if option.entry_state_lateness_ms is not None:
            expected_entry_lateness = (
                option.actual_entry_monotonic_ns
                - option.requested_entry_monotonic_ns
            ) / 1_000_000.0
            require(
                math.isclose(
                    option.entry_state_lateness_ms,
                    expected_entry_lateness,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "entry_lateness_identity",
            )
    if option.base_quantity is not None:
        require(
            math.isfinite(option.base_quantity) and option.base_quantity > 0,
            "base_quantity",
        )
    if option.entry_average_price is not None or option.entry_quote_notional is not None:
        require(
            option.entry_average_price is not None
            and option.entry_quote_notional is not None
            and option.base_quantity is not None,
            "entry_walk_partial",
        )
        if (
            option.entry_average_price is not None
            and option.entry_quote_notional is not None
            and option.base_quantity is not None
        ):
            require(
                option.entry_average_price > 0.0
                and option.entry_quote_notional > 0.0,
                "entry_walk_nonpositive",
            )
            require(
                math.isclose(
                    option.entry_average_price * option.base_quantity,
                    option.entry_quote_notional,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ),
                "entry_walk_identity",
            )
    if option.eligible:
        required = (
            option.actual_entry_monotonic_ns,
            option.entry_state_lateness_ms,
            option.requested_exit_monotonic_ns,
            option.actual_exit_monotonic_ns,
            option.exit_state_lateness_ms,
            option.base_quantity,
            option.entry_average_price,
            option.entry_quote_notional,
            *(getattr(option, name) for name in _OUTCOME_FIELDS),
        )
        require(all(value is not None for value in required), "eligible_null")
        if all(value is not None for value in required):
            assert option.actual_entry_monotonic_ns is not None
            assert option.requested_exit_monotonic_ns is not None
            assert option.actual_exit_monotonic_ns is not None
            assert option.exit_state_lateness_ms is not None
            assert option.entry_quote_notional is not None
            assert option.exit_quote_notional is not None
            assert option.gross_payoff_quote is not None
            assert option.charge_quote is not None
            assert option.net_payoff_quote is not None
            assert option.net_payoff_bps is not None
            assert option.positive_net_payoff is not None
            assert option.maximum_adverse_excursion_bps is not None
            assert option.maximum_favorable_excursion_bps is not None
            assert option.maximum_spread_bps is not None
            assert option.minimum_exit_side_capacity_ratio is not None
            assert option.entry_average_price is not None
            assert option.exit_average_price is not None
            assert option.base_quantity is not None
            assert option.entry_update_id is not None
            assert option.exit_update_id is not None
            require(
                0.0
                <= option.entry_state_lateness_ms
                <= ROUND73_TARGET_MAX_STATE_LATENESS_NS / 1_000_000.0,
                "eligible_entry_lateness",
            )
            require(
                option.requested_exit_monotonic_ns
                == option.actual_entry_monotonic_ns + option.horizon_ms * 1_000_000,
                "exit_request_time",
            )
            require(
                option.actual_exit_monotonic_ns >= option.requested_exit_monotonic_ns,
                "exit_before_request",
            )
            expected_exit_lateness = (
                option.actual_exit_monotonic_ns - option.requested_exit_monotonic_ns
            ) / 1_000_000.0
            require(
                math.isclose(
                    option.exit_state_lateness_ms,
                    expected_exit_lateness,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "exit_lateness_identity",
            )
            require(
                0.0
                <= option.exit_state_lateness_ms
                <= ROUND73_TARGET_MAX_STATE_LATENESS_NS / 1_000_000.0,
                "eligible_exit_lateness",
            )
            require(
                option.exit_average_price > 0.0
                and option.exit_quote_notional > 0.0,
                "exit_walk_nonpositive",
            )
            require(
                math.isclose(
                    option.exit_average_price * option.base_quantity,
                    option.exit_quote_notional,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ),
                "exit_walk_identity",
            )
            payoff = round73_target_payoff(
                side=option.side,  # type: ignore[arg-type]
                entry_quote_notional=option.entry_quote_notional,
                exit_quote_notional=option.exit_quote_notional,
            )
            require(
                math.isclose(
                    option.gross_payoff_quote,
                    payoff.gross_payoff_quote,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "gross_identity",
            )
            require(
                math.isclose(
                    option.charge_quote,
                    payoff.charge_quote,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "charge_identity",
            )
            require(
                math.isclose(
                    option.net_payoff_quote,
                    payoff.net_payoff_quote,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "net_identity",
            )
            require(
                math.isclose(
                    option.net_payoff_bps,
                    payoff.net_payoff_bps,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "net_bps_identity",
            )
            require(
                option.positive_net_payoff == payoff.positive_net_payoff,
                "binary_identity",
            )
            require(
                option.maximum_adverse_excursion_bps
                <= option.net_payoff_bps
                <= option.maximum_favorable_excursion_bps,
                "excursion_bounds",
            )
            require(option.maximum_spread_bps >= 0.0, "path_spread")
            require(
                option.minimum_exit_side_capacity_ratio >= 1.0,
                "path_capacity_ratio",
            )
            require(
                option.entry_update_id > 0 and option.exit_update_id > 0,
                "update_id",
            )
    else:
        require(option.ineligible_reason_mask > 0, "ineligible_reason")
        require(
            all(getattr(option, name) is None for name in _OUTCOME_FIELDS),
            "ineligible_outcome",
        )
    return tuple(errors)


def _quantity_invariant_errors(
    option: Round73TargetOption,
    rules: Round73MarketQuantityRules,
) -> tuple[str, ...]:
    expected = rules.quantize_reference_quantity(
        reference_quote_notional=option.reference_quote_notional,
        decision_mid=option.decision_mid,
    )
    if expected is None:
        if (
            option.base_quantity is not None
            or option.ineligible_reason_mask != _INELIGIBLE_BITS["quantity_filter"]
        ):
            return ("quantity_filter_identity",)
        return ()
    if (
        option.base_quantity is None
        or not rules.is_step_aligned(option.base_quantity)
        or not math.isclose(
            option.base_quantity,
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        return ("quantity_identity",)
    if option.entry_quote_notional is not None:
        below_minimum = Decimal(str(option.entry_quote_notional)) < rules.minimum_notional
        if below_minimum != (
            option.ineligible_reason_mask
            == _INELIGIBLE_BITS["entry_minimum_notional"]
        ):
            return ("entry_minimum_notional_identity",)
    return ()


@dataclass(frozen=True)
class Round73TargetBuildReport:
    run_id: str
    option_count: int
    eligible_option_count: int
    ineligible_option_count: int
    positive_option_count: int
    reason_counts: dict[str, int]
    per_symbol: dict[str, dict[str, int]]
    first_decision_wall_ns: int
    last_decision_wall_ns: int
    source_grid_manifest_sha256: str
    target_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = ROUND73_TARGET_SCHEMA_VERSION
        payload["contract_sha256"] = ROUND73_TARGET_CONTRACT_SHA256
        payload["target_constructed"] = True
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


@dataclass(frozen=True)
class Round73TargetAudit:
    run_id: str
    passed: bool
    errors: tuple[str, ...]
    option_count: int
    eligible_option_count: int
    positive_option_count: int
    target_manifest_sha256: str
    source_grid_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "round-073-target-audit-v1"
        payload["contract_sha256"] = ROUND73_TARGET_CONTRACT_SHA256
        payload["errors"] = list(self.errors)
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


class _TargetReplay:
    def __init__(
        self,
        *,
        run_id: str,
        anchors: Mapping[str, Sequence[_Anchor]],
        quantity_rules: Mapping[str, Round73MarketQuantityRules],
        run_started_wall_ns: int,
        run_started_monotonic_ns: int,
        coverage_end_monotonic_ns: int,
    ) -> None:
        self.run_id = run_id
        self.anchors = {
            symbol: deque(anchors[symbol]) for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        self.quantity_rules = dict(quantity_rules)
        self.run_started_wall_ns = int(run_started_wall_ns)
        self.run_started_monotonic_ns = int(run_started_monotonic_ns)
        self.coverage_end_monotonic_ns = int(coverage_end_monotonic_ns)
        self.latest_state: dict[str, tuple[int, L2BookState]] = {}
        self.latest_next_funding_ms: dict[str, int] = {}
        self.pending_entries: dict[str, deque[_PendingEntry]] = {
            symbol: deque() for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        self.active: dict[str, dict[int, _ActivePosition]] = {
            symbol: {} for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        self.exit_heap: dict[str, list[tuple[int, int]]] = {
            symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        self.rows: list[Round73TargetOption] = []
        self._next_identifier = 0
        self.scheduled_anchor_count = 0

    def _wall_ns(self, monotonic_ns: int) -> int:
        return self.run_started_wall_ns + (
            int(monotonic_ns) - self.run_started_monotonic_ns
        )

    @staticmethod
    def _walk_for_action(
        state: L2BookState,
        *,
        side: str,
        entering: bool,
        base_quantity: float,
    ) -> Round73BookWalk | None:
        buy = (side == "long") == entering
        levels = state.ask_levels if buy else state.bid_levels
        return walk_round73_book(
            levels,
            base_quantity=base_quantity,
            ascending_prices=buy,
        )

    def _record_ineligible(
        self,
        *,
        decision: _Decision,
        entry_delay_ms: int,
        horizon_ms: int,
        reference_quote_notional: float,
        side: str,
        reason: str,
        requested_entry_monotonic_ns: int,
        actual_entry_monotonic_ns: int | None = None,
        requested_exit_monotonic_ns: int | None = None,
        actual_exit_monotonic_ns: int | None = None,
        base_quantity: float | None = None,
        entry_walk: Round73BookWalk | None = None,
        entry_update_id: int | None = None,
    ) -> None:
        mask = _ineligible_mask(reason)
        entry_lateness = (
            None
            if actual_entry_monotonic_ns is None
            else (actual_entry_monotonic_ns - requested_entry_monotonic_ns)
            / 1_000_000.0
        )
        exit_lateness = (
            None
            if actual_exit_monotonic_ns is None
            or requested_exit_monotonic_ns is None
            else (actual_exit_monotonic_ns - requested_exit_monotonic_ns)
            / 1_000_000.0
        )
        option = _make_option(
            run_id=self.run_id,
            symbol=decision.anchor.symbol,
            anchor_index=decision.anchor.anchor_index,
            entry_delay_ms=int(entry_delay_ms),
            horizon_ms=int(horizon_ms),
            reference_quote_notional=float(reference_quote_notional),
            side=side,
            eligible=False,
            ineligible_reason_mask=mask,
            ineligible_reasons_json=_canonical_json([reason]),
            decision_monotonic_ns=decision.anchor.decision_monotonic_ns,
            decision_book_received_monotonic_ns=(
                decision.book_received_monotonic_ns
            ),
            requested_entry_monotonic_ns=int(requested_entry_monotonic_ns),
            actual_entry_monotonic_ns=actual_entry_monotonic_ns,
            entry_state_lateness_ms=entry_lateness,
            requested_exit_monotonic_ns=requested_exit_monotonic_ns,
            actual_exit_monotonic_ns=actual_exit_monotonic_ns,
            exit_state_lateness_ms=exit_lateness,
            base_quantity=base_quantity,
            decision_mid=decision.book_state.mid,
            entry_average_price=(None if entry_walk is None else entry_walk.average_price),
            entry_quote_notional=(
                None if entry_walk is None else entry_walk.quote_notional
            ),
            exit_average_price=None,
            exit_quote_notional=None,
            gross_payoff_quote=None,
            charge_quote=None,
            net_payoff_quote=None,
            net_payoff_bps=None,
            positive_net_payoff=None,
            maximum_adverse_excursion_bps=None,
            maximum_favorable_excursion_bps=None,
            maximum_spread_bps=None,
            minimum_exit_side_capacity_ratio=None,
            entry_update_id=entry_update_id,
            exit_update_id=None,
        )
        errors = _option_invariant_errors(option)
        if errors:
            raise ValueError(
                "Round 73 ineligible target invariant failed: " + ",".join(errors)
            )
        self.rows.append(option)

    def _schedule_anchor(self, anchor: _Anchor) -> None:
        latest = self.latest_state.get(anchor.symbol)
        if latest is None:
            raise ValueError("Round 73 target decision has no synchronized book")
        book_receipt, state = latest
        if (
            book_receipt >= anchor.decision_monotonic_ns
            or anchor.source_max_received_monotonic_ns >= anchor.decision_monotonic_ns
        ):
            raise ValueError("Round 73 target decision uses a future receipt")
        rules = self.quantity_rules[anchor.symbol]
        quantities = tuple(
            (
                reference,
                rules.quantize_reference_quantity(
                    reference_quote_notional=reference,
                    decision_mid=state.mid,
                ),
            )
            for reference in ROUND73_TARGET_REFERENCE_NOTIONALS
        )
        decision = _Decision(
            anchor=anchor,
            book_received_monotonic_ns=book_receipt,
            book_state=state,
            quantities=quantities,
        )
        for delay_ms in ROUND73_TARGET_ENTRY_DELAYS_MS:
            requested_entry = anchor.decision_monotonic_ns + delay_ms * 1_000_000
            for reference, quantity in quantities:
                if quantity is not None:
                    continue
                for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                    for side in ROUND73_TARGET_SIDES:
                        self._record_ineligible(
                            decision=decision,
                            entry_delay_ms=delay_ms,
                            horizon_ms=horizon_ms,
                            reference_quote_notional=reference,
                            side=side,
                            reason="quantity_filter",
                            requested_entry_monotonic_ns=requested_entry,
                        )
            if any(quantity is not None for _reference, quantity in quantities):
                self.pending_entries[anchor.symbol].append(
                    _PendingEntry(
                        decision=decision,
                        entry_delay_ms=delay_ms,
                        requested_entry_monotonic_ns=requested_entry,
                    )
                )
        self.scheduled_anchor_count += 1

    def before_record(self, received_monotonic_ns: int) -> None:
        timestamp = int(received_monotonic_ns)
        for symbol in IMPACT_CAPTURE_SYMBOLS:
            anchors = self.anchors[symbol]
            while anchors and anchors[0].decision_monotonic_ns <= timestamp:
                self._schedule_anchor(anchors.popleft())

    def observe_mark(self, *, symbol: str, next_funding_time_ms: int) -> None:
        selected = int(next_funding_time_ms)
        if selected <= 0:
            raise ValueError("Round 73 target next funding time is invalid")
        self.latest_next_funding_ms[symbol] = selected

    def _pending_entry_failure(
        self,
        pending: _PendingEntry,
        *,
        reason: str,
        actual_entry_monotonic_ns: int | None = None,
    ) -> None:
        for reference, quantity in pending.decision.quantities:
            if quantity is None:
                continue
            for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                for side in ROUND73_TARGET_SIDES:
                    self._record_ineligible(
                        decision=pending.decision,
                        entry_delay_ms=pending.entry_delay_ms,
                        horizon_ms=horizon_ms,
                        reference_quote_notional=reference,
                        side=side,
                        reason=reason,
                        requested_entry_monotonic_ns=(
                            pending.requested_entry_monotonic_ns
                        ),
                        actual_entry_monotonic_ns=actual_entry_monotonic_ns,
                        base_quantity=quantity,
                    )

    def _start_position(
        self,
        *,
        pending: _PendingEntry,
        state_received_monotonic_ns: int,
        state: L2BookState,
        reference_quote_notional: float,
        base_quantity: float,
        side: str,
        entry_walk: Round73BookWalk,
        horizon_ms: int,
        close_walk: Round73BookWalk,
    ) -> None:
        requested_exit = state_received_monotonic_ns + horizon_ms * 1_000_000
        next_funding_ms = self.latest_next_funding_ms.get(state.symbol)
        if next_funding_ms is None:
            raise ValueError("Round 73 target entry has no causal funding context")
        entry_wall_ns = self._wall_ns(state_received_monotonic_ns)
        latest_exit_wall_ns = self._wall_ns(
            requested_exit + ROUND73_TARGET_MAX_STATE_LATENESS_NS
        )
        if entry_wall_ns < next_funding_ms * 1_000_000 <= latest_exit_wall_ns:
            self._record_ineligible(
                decision=pending.decision,
                entry_delay_ms=pending.entry_delay_ms,
                horizon_ms=horizon_ms,
                reference_quote_notional=reference_quote_notional,
                side=side,
                reason="funding_boundary",
                requested_entry_monotonic_ns=pending.requested_entry_monotonic_ns,
                actual_entry_monotonic_ns=state_received_monotonic_ns,
                requested_exit_monotonic_ns=requested_exit,
                base_quantity=base_quantity,
                entry_walk=entry_walk,
                entry_update_id=state.update_id,
            )
            return
        path_payoff = round73_target_payoff(
            side=side,  # type: ignore[arg-type]
            entry_quote_notional=entry_walk.quote_notional,
            exit_quote_notional=close_walk.quote_notional,
        )
        identifier = self._next_identifier
        self._next_identifier += 1
        position = _ActivePosition(
            identifier=identifier,
            decision=pending.decision,
            entry_delay_ms=pending.entry_delay_ms,
            horizon_ms=horizon_ms,
            reference_quote_notional=reference_quote_notional,
            side=side,
            requested_entry_monotonic_ns=pending.requested_entry_monotonic_ns,
            actual_entry_monotonic_ns=state_received_monotonic_ns,
            requested_exit_monotonic_ns=requested_exit,
            base_quantity=base_quantity,
            entry_walk=entry_walk,
            entry_update_id=state.update_id,
            minimum_net_payoff_bps=path_payoff.net_payoff_bps,
            maximum_net_payoff_bps=path_payoff.net_payoff_bps,
            maximum_spread_bps=state.spread_bps,
            minimum_exit_side_capacity_ratio=close_walk.capacity_ratio,
        )
        self.active[state.symbol][identifier] = position
        heapq.heappush(self.exit_heap[state.symbol], (requested_exit, identifier))

    def _fulfill_entries(
        self,
        *,
        symbol: str,
        received_monotonic_ns: int,
        state: L2BookState,
    ) -> None:
        pending_queue = self.pending_entries[symbol]
        while (
            pending_queue
            and pending_queue[0].requested_entry_monotonic_ns
            <= received_monotonic_ns
        ):
            pending = pending_queue.popleft()
            lateness = (
                received_monotonic_ns - pending.requested_entry_monotonic_ns
            )
            if lateness > ROUND73_TARGET_MAX_STATE_LATENESS_NS:
                self._pending_entry_failure(
                    pending,
                    reason="entry_state_late",
                    actual_entry_monotonic_ns=received_monotonic_ns,
                )
                continue
            rules = self.quantity_rules[symbol]
            for reference, quantity in pending.decision.quantities:
                if quantity is None:
                    continue
                for side in ROUND73_TARGET_SIDES:
                    entry_walk = self._walk_for_action(
                        state,
                        side=side,
                        entering=True,
                        base_quantity=quantity,
                    )
                    if entry_walk is None:
                        for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                            self._record_ineligible(
                                decision=pending.decision,
                                entry_delay_ms=pending.entry_delay_ms,
                                horizon_ms=horizon_ms,
                                reference_quote_notional=reference,
                                side=side,
                                reason="entry_capacity",
                                requested_entry_monotonic_ns=(
                                    pending.requested_entry_monotonic_ns
                                ),
                                actual_entry_monotonic_ns=received_monotonic_ns,
                                base_quantity=quantity,
                            )
                        continue
                    if Decimal(str(entry_walk.quote_notional)) < rules.minimum_notional:
                        for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                            self._record_ineligible(
                                decision=pending.decision,
                                entry_delay_ms=pending.entry_delay_ms,
                                horizon_ms=horizon_ms,
                                reference_quote_notional=reference,
                                side=side,
                                reason="entry_minimum_notional",
                                requested_entry_monotonic_ns=(
                                    pending.requested_entry_monotonic_ns
                                ),
                                actual_entry_monotonic_ns=received_monotonic_ns,
                                base_quantity=quantity,
                                entry_walk=entry_walk,
                                entry_update_id=state.update_id,
                            )
                        continue
                    close_walk = self._walk_for_action(
                        state,
                        side=side,
                        entering=False,
                        base_quantity=quantity,
                    )
                    if close_walk is None:
                        for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                            self._record_ineligible(
                                decision=pending.decision,
                                entry_delay_ms=pending.entry_delay_ms,
                                horizon_ms=horizon_ms,
                                reference_quote_notional=reference,
                                side=side,
                                reason="path_capacity",
                                requested_entry_monotonic_ns=(
                                    pending.requested_entry_monotonic_ns
                                ),
                                actual_entry_monotonic_ns=received_monotonic_ns,
                                base_quantity=quantity,
                                entry_walk=entry_walk,
                                entry_update_id=state.update_id,
                            )
                        continue
                    for horizon_ms in ROUND73_TARGET_HORIZONS_MS:
                        self._start_position(
                            pending=pending,
                            state_received_monotonic_ns=received_monotonic_ns,
                            state=state,
                            reference_quote_notional=reference,
                            base_quantity=quantity,
                            side=side,
                            entry_walk=entry_walk,
                            horizon_ms=horizon_ms,
                            close_walk=close_walk,
                        )

    def _record_position_ineligible(
        self,
        position: _ActivePosition,
        *,
        reason: str,
        actual_exit_monotonic_ns: int | None = None,
    ) -> None:
        self._record_ineligible(
            decision=position.decision,
            entry_delay_ms=position.entry_delay_ms,
            horizon_ms=position.horizon_ms,
            reference_quote_notional=position.reference_quote_notional,
            side=position.side,
            reason=reason,
            requested_entry_monotonic_ns=position.requested_entry_monotonic_ns,
            actual_entry_monotonic_ns=position.actual_entry_monotonic_ns,
            requested_exit_monotonic_ns=position.requested_exit_monotonic_ns,
            actual_exit_monotonic_ns=actual_exit_monotonic_ns,
            base_quantity=position.base_quantity,
            entry_walk=position.entry_walk,
            entry_update_id=position.entry_update_id,
        )

    def _update_active_paths(
        self,
        *,
        symbol: str,
        state: L2BookState,
    ) -> dict[tuple[str, float], Round73BookWalk | None]:
        cache: dict[tuple[str, float], Round73BookWalk | None] = {}
        failed: list[int] = []
        for identifier, position in tuple(self.active[symbol].items()):
            key = (position.side, position.base_quantity)
            if key not in cache:
                cache[key] = self._walk_for_action(
                    state,
                    side=position.side,
                    entering=False,
                    base_quantity=position.base_quantity,
                )
            close_walk = cache[key]
            if close_walk is None:
                self._record_position_ineligible(position, reason="path_capacity")
                failed.append(identifier)
                continue
            payoff = round73_target_payoff(
                side=position.side,  # type: ignore[arg-type]
                entry_quote_notional=position.entry_walk.quote_notional,
                exit_quote_notional=close_walk.quote_notional,
            )
            position.minimum_net_payoff_bps = min(
                position.minimum_net_payoff_bps,
                payoff.net_payoff_bps,
            )
            position.maximum_net_payoff_bps = max(
                position.maximum_net_payoff_bps,
                payoff.net_payoff_bps,
            )
            position.maximum_spread_bps = max(
                position.maximum_spread_bps,
                state.spread_bps,
            )
            position.minimum_exit_side_capacity_ratio = min(
                position.minimum_exit_side_capacity_ratio,
                close_walk.capacity_ratio,
            )
        for identifier in failed:
            self.active[symbol].pop(identifier, None)
        return cache

    def _complete_exits(
        self,
        *,
        symbol: str,
        received_monotonic_ns: int,
        state: L2BookState,
        cache: Mapping[tuple[str, float], Round73BookWalk | None],
    ) -> None:
        heap = self.exit_heap[symbol]
        active = self.active[symbol]
        while heap and heap[0][0] <= received_monotonic_ns:
            _requested_exit, identifier = heapq.heappop(heap)
            position = active.pop(identifier, None)
            if position is None:
                continue
            lateness = received_monotonic_ns - position.requested_exit_monotonic_ns
            if lateness > ROUND73_TARGET_MAX_STATE_LATENESS_NS:
                self._record_position_ineligible(
                    position,
                    reason="exit_state_late",
                    actual_exit_monotonic_ns=received_monotonic_ns,
                )
                continue
            close_walk = cache.get((position.side, position.base_quantity))
            if close_walk is None:
                self._record_position_ineligible(
                    position,
                    reason="exit_capacity",
                    actual_exit_monotonic_ns=received_monotonic_ns,
                )
                continue
            payoff = round73_target_payoff(
                side=position.side,  # type: ignore[arg-type]
                entry_quote_notional=position.entry_walk.quote_notional,
                exit_quote_notional=close_walk.quote_notional,
            )
            option = _make_option(
                run_id=self.run_id,
                symbol=symbol,
                anchor_index=position.decision.anchor.anchor_index,
                entry_delay_ms=position.entry_delay_ms,
                horizon_ms=position.horizon_ms,
                reference_quote_notional=position.reference_quote_notional,
                side=position.side,
                eligible=True,
                ineligible_reason_mask=0,
                ineligible_reasons_json="[]",
                decision_monotonic_ns=(
                    position.decision.anchor.decision_monotonic_ns
                ),
                decision_book_received_monotonic_ns=(
                    position.decision.book_received_monotonic_ns
                ),
                requested_entry_monotonic_ns=(
                    position.requested_entry_monotonic_ns
                ),
                actual_entry_monotonic_ns=position.actual_entry_monotonic_ns,
                entry_state_lateness_ms=(
                    position.actual_entry_monotonic_ns
                    - position.requested_entry_monotonic_ns
                )
                / 1_000_000.0,
                requested_exit_monotonic_ns=position.requested_exit_monotonic_ns,
                actual_exit_monotonic_ns=received_monotonic_ns,
                exit_state_lateness_ms=lateness / 1_000_000.0,
                base_quantity=position.base_quantity,
                decision_mid=position.decision.book_state.mid,
                entry_average_price=position.entry_walk.average_price,
                entry_quote_notional=position.entry_walk.quote_notional,
                exit_average_price=close_walk.average_price,
                exit_quote_notional=close_walk.quote_notional,
                gross_payoff_quote=payoff.gross_payoff_quote,
                charge_quote=payoff.charge_quote,
                net_payoff_quote=payoff.net_payoff_quote,
                net_payoff_bps=payoff.net_payoff_bps,
                positive_net_payoff=payoff.positive_net_payoff,
                maximum_adverse_excursion_bps=(
                    position.minimum_net_payoff_bps
                ),
                maximum_favorable_excursion_bps=(
                    position.maximum_net_payoff_bps
                ),
                maximum_spread_bps=position.maximum_spread_bps,
                minimum_exit_side_capacity_ratio=(
                    position.minimum_exit_side_capacity_ratio
                ),
                entry_update_id=position.entry_update_id,
                exit_update_id=state.update_id,
            )
            errors = _option_invariant_errors(option)
            if errors:
                raise ValueError(
                    "Round 73 eligible target invariant failed: "
                    + ",".join(errors)
                )
            self.rows.append(option)

    def observe_depth(
        self,
        *,
        symbol: str,
        received_monotonic_ns: int,
        state: L2BookState,
    ) -> None:
        timestamp = int(received_monotonic_ns)
        self.latest_state[symbol] = (timestamp, state)
        cache = self._update_active_paths(symbol=symbol, state=state)
        self._complete_exits(
            symbol=symbol,
            received_monotonic_ns=timestamp,
            state=state,
            cache=cache,
        )
        self._fulfill_entries(
            symbol=symbol,
            received_monotonic_ns=timestamp,
            state=state,
        )

    def finish(self) -> list[Round73TargetOption]:
        for symbol in IMPACT_CAPTURE_SYMBOLS:
            while self.anchors[symbol]:
                self._schedule_anchor(self.anchors[symbol].popleft())
            while self.pending_entries[symbol]:
                self._pending_entry_failure(
                    self.pending_entries[symbol].popleft(),
                    reason="coverage_end",
                )
            for position in tuple(self.active[symbol].values()):
                self._record_position_ineligible(position, reason="coverage_end")
            self.active[symbol].clear()
            self.exit_heap[symbol].clear()
        expected = self.scheduled_anchor_count * _EXPECTED_OPTIONS_PER_ANCHOR
        if len(self.rows) != expected:
            raise ValueError(
                "Round 73 target option count differs: "
                f"expected={expected} actual={len(self.rows)}"
            )
        self.rows.sort(key=lambda option: option.key)
        keys = [option.key for option in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Round 73 target option keys are duplicated")
        return self.rows


def _parse_quantity_rules(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> dict[str, Round73MarketQuantityRules]:
    exchange_info: Mapping[str, object] | None = None
    records = iter_impact_capture_v9_records(
        connection,
        run_id=run_id,
    )
    try:
        for _frame_index, _message_index, record in records:
            if record.stream != "binance_futures_rest":
                continue
            root = _strict_json_object(
                record.raw_text,
                "Round 73 exchange information",
            )
            if isinstance(root.get("symbols"), Sequence):
                if exchange_info is not None:
                    raise ValueError(
                        "Round 73 target has duplicate exchange information"
                    )
                exchange_info = root
                break
    finally:
        records.close()
    if exchange_info is None:
        raise ValueError("Round 73 target exchange information is missing")
    raw_symbols = exchange_info.get("symbols")
    if not isinstance(raw_symbols, Sequence) or isinstance(
        raw_symbols, (str, bytes, bytearray)
    ):
        raise ValueError("Round 73 target exchange symbols are missing")
    selected: dict[str, Round73MarketQuantityRules] = {}
    for raw_symbol in raw_symbols:
        if not isinstance(raw_symbol, Mapping):
            continue
        symbol = str(raw_symbol.get("symbol", "")).strip().upper()
        if symbol not in IMPACT_CAPTURE_SYMBOLS:
            continue
        if str(raw_symbol.get("status", "")) != "TRADING":
            raise ValueError(f"Round 73 target symbol is not trading: {symbol}")
        filters = raw_symbol.get("filters")
        if not isinstance(filters, Sequence) or isinstance(
            filters, (str, bytes, bytearray)
        ):
            raise ValueError(f"Round 73 target filters are missing: {symbol}")
        by_type = {
            str(item.get("filterType", "")): item
            for item in filters
            if isinstance(item, Mapping)
        }
        market = by_type.get("MARKET_LOT_SIZE")
        minimum_notional = by_type.get("MIN_NOTIONAL")
        if market is None or minimum_notional is None:
            raise ValueError(f"Round 73 target market filters differ: {symbol}")
        selected[symbol] = Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size=market.get("stepSize"),
            minimum_quantity=market.get("minQty"),
            maximum_quantity=market.get("maxQty"),
            minimum_notional=minimum_notional.get("notional"),
        )
    if tuple(sorted(selected)) != tuple(sorted(IMPACT_CAPTURE_SYMBOLS)):
        raise ValueError("Round 73 target exchange information is incomplete")
    return selected


def _target_rows_v9(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    anchors: Mapping[str, Sequence[_Anchor]],
    quantity_rules: Mapping[str, Round73MarketQuantityRules],
    run_started_wall_ns: int,
    run_started_monotonic_ns: int,
    coverage_start_wall_ns: int,
    coverage_end_wall_ns: int,
    segments: Sequence[tuple[object, ...]],
) -> list[Round73TargetOption]:
    preflight = load_impact_capture_v9_preflight(connection, run_id=run_id)
    if coverage_start_wall_ns < preflight.ready_wall_ns:
        raise ValueError("Round 73 target coverage precedes feature-ready marker")
    coverage_end_mono = run_started_monotonic_ns + (
        coverage_end_wall_ns - run_started_wall_ns
    )
    books = {
        str(symbol): SynchronizedDepthBook(str(symbol), float(tick_size))
        for symbol, _status, tick_size in segments
    }
    snapshots = dict(preflight.snapshot_records)
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        books[symbol].initialize(
            _strict_json_object(
                snapshots[symbol].raw_text,
                "Round 73 target depth snapshot",
            )
        )
    replay = _TargetReplay(
        run_id=run_id,
        anchors=anchors,
        quantity_rules=quantity_rules,
        run_started_wall_ns=run_started_wall_ns,
        run_started_monotonic_ns=run_started_monotonic_ns,
        coverage_end_monotonic_ns=coverage_end_mono,
    )
    observed_snapshots: set[str] = set()
    for frame_index, message_index, record in iter_impact_capture_v9_records(
        connection,
        run_id=run_id,
    ):
        event_mono = int(record.received_monotonic_ns)
        replay.before_record(event_mono)
        if event_mono >= coverage_end_mono:
            break
        if record.stream == "binance_futures_rest":
            for symbol, snapshot in snapshots.items():
                if record == snapshot:
                    observed_snapshots.add(symbol)
                    break
            continue
        raw_text = record.raw_text
        is_depth = "depthUpdate" in raw_text
        is_mark = "markPriceUpdate" in raw_text
        if not is_depth and not is_mark:
            continue
        try:
            root = _strict_json_object(raw_text, "Round 73 target WebSocket event")
        except (json.JSONDecodeError, ValueError):
            if ImpactAbsorptionStore._v9_websocket_event_type(record) == "rejectedWire":
                continue
            raise
        payload = root.get("data")
        if not isinstance(payload, Mapping):
            raise ValueError("Round 73 target WebSocket payload is missing")
        event_type = str(payload.get("e", ""))
        symbol = str(payload.get("s", "")).strip().upper()
        if symbol not in books:
            raise ValueError("Round 73 target WebSocket symbol differs")
        validate_combined_stream_name(
            str(root.get("stream", "")),
            event_type=event_type,
            symbol=symbol,
        )
        if event_type == "depthUpdate":
            book = books[symbol]
            event = book.apply(payload, receive_time_ns=event_mono)
            if event.stale:
                continue
            replay.observe_depth(
                symbol=symbol,
                received_monotonic_ns=event_mono,
                state=book.state(ROUND73_TARGET_LEVELS),
            )
        elif event_type == "markPriceUpdate":
            mark = parse_mark_price(
                payload,
                symbol=symbol,
                receive_time_ns=event_mono,
            )
            replay.observe_mark(
                symbol=symbol,
                next_funding_time_ms=mark.next_funding_time_ms,
            )
    if tuple(sorted(observed_snapshots)) != tuple(sorted(IMPACT_CAPTURE_SYMBOLS)):
        raise ValueError("Round 73 target snapshot replay is incomplete")
    return replay.finish()


def _option_rows_sha256(options: Sequence[Round73TargetOption]) -> str:
    digest = hashlib.sha256()
    for option in options:
        digest.update(option.option_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _summarize_options(
    options: Sequence[Round73TargetOption],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    reason_counts = {reason: 0 for reason in _INELIGIBLE_BITS}
    per_symbol = {
        symbol: {"options": 0, "eligible": 0, "ineligible": 0, "positive": 0}
        for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    for option in options:
        values = per_symbol[option.symbol]
        values["options"] += 1
        if option.eligible:
            values["eligible"] += 1
            values["positive"] += int(bool(option.positive_net_payoff))
        else:
            values["ineligible"] += 1
            for reason, bit in _INELIGIBLE_BITS.items():
                reason_counts[reason] += int(bool(option.ineligible_reason_mask & bit))
    return reason_counts, per_symbol


def _build_identity(
    *,
    run_id: str,
    source_corpus_manifest_sha256: str,
    source_grid_manifest_sha256: str,
    options: Sequence[Round73TargetOption],
    valid_anchor_count: int,
    first_decision_wall_ns: int,
    last_decision_wall_ns: int,
) -> dict[str, object]:
    reason_counts, per_symbol = _summarize_options(options)
    eligible = sum(option.eligible for option in options)
    positive = sum(
        option.eligible and bool(option.positive_net_payoff) for option in options
    )
    return {
        "schema_version": ROUND73_TARGET_SCHEMA_VERSION,
        "contract_sha256": ROUND73_TARGET_CONTRACT_SHA256,
        "run_id": run_id,
        "source_corpus_manifest_sha256": source_corpus_manifest_sha256,
        "source_grid_manifest_sha256": source_grid_manifest_sha256,
        "valid_anchor_count": valid_anchor_count,
        "expected_options_per_valid_anchor": _EXPECTED_OPTIONS_PER_ANCHOR,
        "option_count": len(options),
        "eligible_option_count": eligible,
        "ineligible_option_count": len(options) - eligible,
        "positive_option_count": positive,
        "reason_counts": reason_counts,
        "per_symbol": per_symbol,
        "option_rows_sha256": _option_rows_sha256(options),
        "first_decision_wall_ns": first_decision_wall_ns,
        "last_decision_wall_ns": last_decision_wall_ns,
        "crypto_formal_daily_close": False,
        "target_constructed": True,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }


def _report_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73TargetBuildReport:
    return Round73TargetBuildReport(
        run_id=str(identity["run_id"]),
        option_count=int(identity["option_count"]),
        eligible_option_count=int(identity["eligible_option_count"]),
        ineligible_option_count=int(identity["ineligible_option_count"]),
        positive_option_count=int(identity["positive_option_count"]),
        reason_counts={
            str(key): int(value)
            for key, value in dict(identity["reason_counts"]).items()  # type: ignore[arg-type]
        },
        per_symbol={
            str(symbol): {str(key): int(value) for key, value in dict(counts).items()}
            for symbol, counts in dict(identity["per_symbol"]).items()  # type: ignore[arg-type]
        },
        first_decision_wall_ns=int(identity["first_decision_wall_ns"]),
        last_decision_wall_ns=int(identity["last_decision_wall_ns"]),
        source_grid_manifest_sha256=str(identity["source_grid_manifest_sha256"]),
        target_manifest_sha256=manifest_sha256,
    )


def _create_target_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_OPTION_TABLE} (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            entry_delay_ms USMALLINT NOT NULL,
            horizon_ms USMALLINT NOT NULL,
            reference_quote_notional UINTEGER NOT NULL,
            side VARCHAR NOT NULL,
            eligible BOOLEAN NOT NULL,
            ineligible_reason_mask UINTEGER NOT NULL,
            ineligible_reasons_json VARCHAR NOT NULL,
            decision_monotonic_ns UBIGINT NOT NULL,
            decision_book_received_monotonic_ns UBIGINT NOT NULL,
            requested_entry_monotonic_ns UBIGINT NOT NULL,
            actual_entry_monotonic_ns UBIGINT,
            entry_state_lateness_ms DOUBLE,
            requested_exit_monotonic_ns UBIGINT,
            actual_exit_monotonic_ns UBIGINT,
            exit_state_lateness_ms DOUBLE,
            base_quantity DOUBLE,
            decision_mid DOUBLE NOT NULL,
            entry_average_price DOUBLE,
            entry_quote_notional DOUBLE,
            exit_average_price DOUBLE,
            exit_quote_notional DOUBLE,
            gross_payoff_quote DOUBLE,
            charge_quote DOUBLE,
            net_payoff_quote DOUBLE,
            net_payoff_bps DOUBLE,
            positive_net_payoff BOOLEAN,
            maximum_adverse_excursion_bps DOUBLE,
            maximum_favorable_excursion_bps DOUBLE,
            maximum_spread_bps DOUBLE,
            minimum_exit_side_capacity_ratio DOUBLE,
            entry_update_id UBIGINT,
            exit_update_id UBIGINT,
            option_sha256 VARCHAR NOT NULL CHECK (length(option_sha256) = 64),
            PRIMARY KEY (
                run_id, symbol, anchor_index, entry_delay_ms, horizon_ms,
                reference_quote_notional, side
            )
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_TARGET_MANIFEST_TABLE} (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            source_corpus_manifest_sha256 VARCHAR NOT NULL,
            source_grid_manifest_sha256 VARCHAR NOT NULL,
            target_manifest_json VARCHAR NOT NULL,
            target_manifest_sha256 VARCHAR NOT NULL,
            option_rows_sha256 VARCHAR NOT NULL,
            valid_anchor_count UINTEGER NOT NULL,
            option_count UBIGINT NOT NULL,
            eligible_option_count UBIGINT NOT NULL,
            positive_option_count UBIGINT NOT NULL,
            first_decision_wall_ns UBIGINT NOT NULL,
            last_decision_wall_ns UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(source_corpus_manifest_sha256) = 64),
            CHECK (length(source_grid_manifest_sha256) = 64),
            CHECK (length(target_manifest_sha256) = 64),
            CHECK (length(option_rows_sha256) = 64)
        )
        """
    )


def _assert_target_table_shapes(connection: duckdb.DuckDBPyConnection) -> None:
    expected = {
        ROUND73_TARGET_OPTION_TABLE: _OPTION_COLUMNS,
        ROUND73_TARGET_MANIFEST_TABLE: (
            "run_id",
            "schema_version",
            "contract_sha256",
            "source_corpus_manifest_sha256",
            "source_grid_manifest_sha256",
            "target_manifest_json",
            "target_manifest_sha256",
            "option_rows_sha256",
            "valid_anchor_count",
            "option_count",
            "eligible_option_count",
            "positive_option_count",
            "first_decision_wall_ns",
            "last_decision_wall_ns",
            "recorded_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 target table schema differs: {table}")


def _insert_option_batch(
    connection: duckdb.DuckDBPyConnection,
    options: Sequence[Round73TargetOption],
    *,
    batch_index: int,
) -> None:
    if not options:
        return
    rows = [option.as_row() for option in options]
    views: list[str] = []
    projections: list[str] = []
    try:
        for index, column in enumerate(zip(*rows, strict=True)):
            values = tuple(column)
            view = f"_round73_target_{batch_index}_{index}"
            if all(isinstance(value, bool) for value in values):
                array = np.asarray(values, dtype=np.bool_)
                projection = f"{view}.column0"
            elif all(value is None or isinstance(value, bool) for value in values):
                array = np.asarray(
                    [-1 if value is None else int(value) for value in values],
                    dtype=np.int8,
                )
                projection = (
                    f"CASE WHEN {view}.column0 = -1 THEN NULL "
                    f"ELSE {view}.column0 != 0 END"
                )
            elif all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                array = np.asarray(values, dtype=np.int64)
                projection = f"{view}.column0"
            elif all(
                value is None
                or (isinstance(value, int) and not isinstance(value, bool))
                for value in values
            ):
                array = np.asarray(
                    [-1 if value is None else value for value in values],
                    dtype=np.int64,
                )
                projection = (
                    f"CASE WHEN {view}.column0 = -1 THEN NULL ELSE {view}.column0 END"
                )
            elif all(
                value is None or isinstance(value, (int, float)) for value in values
            ):
                array = np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype=np.float64,
                )
                projection = (
                    f"CASE WHEN isnan({view}.column0) THEN NULL "
                    f"ELSE {view}.column0 END"
                )
            elif all(isinstance(value, str) for value in values):
                array = np.asarray(values, dtype=np.str_)
                projection = f"{view}.column0"
            else:
                raise TypeError("Round 73 target insert column type is unsupported")
            connection.register(view, array)
            views.append(view)
            projections.append(projection)
        connection.execute(
            f"INSERT INTO {ROUND73_TARGET_OPTION_TABLE} SELECT "
            + ", ".join(projections)
            + " FROM "
            + " POSITIONAL JOIN ".join(views)
        )
    finally:
        for view in views:
            connection.unregister(view)


def _option_from_row(row: Sequence[object]) -> Round73TargetOption:
    if len(row) != len(_OPTION_COLUMNS):
        raise ValueError("Round 73 target row width differs")
    raw = dict(zip(_OPTION_COLUMNS, row, strict=True))
    for name in (
        "anchor_index",
        "entry_delay_ms",
        "horizon_ms",
        "ineligible_reason_mask",
        "decision_monotonic_ns",
        "decision_book_received_monotonic_ns",
        "requested_entry_monotonic_ns",
    ):
        raw[name] = int(raw[name])
    for name in (
        "actual_entry_monotonic_ns",
        "requested_exit_monotonic_ns",
        "actual_exit_monotonic_ns",
        "entry_update_id",
        "exit_update_id",
    ):
        raw[name] = None if raw[name] is None else int(raw[name])
    raw["reference_quote_notional"] = float(raw["reference_quote_notional"])
    raw["eligible"] = bool(raw["eligible"])
    raw["positive_net_payoff"] = (
        None if raw["positive_net_payoff"] is None else bool(raw["positive_net_payoff"])
    )
    for name in (
        "entry_state_lateness_ms",
        "exit_state_lateness_ms",
        "base_quantity",
        "decision_mid",
        "entry_average_price",
        "entry_quote_notional",
        "exit_average_price",
        "exit_quote_notional",
        "gross_payoff_quote",
        "charge_quote",
        "net_payoff_quote",
        "net_payoff_bps",
        "maximum_adverse_excursion_bps",
        "maximum_favorable_excursion_bps",
        "maximum_spread_bps",
        "minimum_exit_side_capacity_ratio",
    ):
        raw[name] = None if raw[name] is None else float(raw[name])
    return Round73TargetOption(**raw)  # type: ignore[arg-type]


def build_round73_executable_targets(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73TargetBuildReport:
    """Replay and atomically publish one target set without model selection."""

    selected = _validated_run_id(run_id)
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        existing = None
        if _table_exists(connection, ROUND73_TARGET_MANIFEST_TABLE):
            existing = connection.execute(
                f"SELECT target_manifest_json, target_manifest_sha256 "
                f"FROM {ROUND73_TARGET_MANIFEST_TABLE} WHERE run_id = ?",
                [selected],
            ).fetchone()
    if existing is not None:
        audit = audit_round73_executable_targets(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing target audit failed")
        identity = _strict_json_object(str(existing[0]), "target manifest")
        return _report_from_identity(identity, str(existing[1]))
    corpus_audit = audit_round73_corpus_manifest(
        database,
        run_id=selected,
        memory_limit=memory_limit,
        threads=threads,
    )
    grid_audit = audit_round73_causal_grid(
        database,
        run_id=selected,
        memory_limit=memory_limit,
        threads=threads,
    )
    if not corpus_audit.passed or not grid_audit.passed:
        raise ValueError("Round 73 target source audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        source = connection.execute(
            f"""
            SELECT c.manifest_sha256, c.coverage_start_wall_ns,
                   c.coverage_end_wall_ns, r.schema_version,
                   r.capture_contract_sha256, r.started_wall_ns,
                   r.started_monotonic_ns, g.build_manifest_sha256
            FROM {ROUND73_CORPUS_RUN_TABLE} c
            JOIN impact_capture_run r ON r.run_id = c.run_id
            JOIN {ROUND73_GRID_MANIFEST_TABLE} g ON g.run_id = c.run_id
            WHERE c.run_id = ?
            """,
            [selected],
        ).fetchone()
        if source is None:
            raise ValueError("Round 73 target source identity is missing")
        if (
            str(source[0]) != corpus_audit.manifest_sha256
            or str(source[3]) != IMPACT_CAPTURE_V9_SCHEMA_VERSION
            or str(source[4]) != IMPACT_CAPTURE_V9_CONTRACT_SHA256
            or str(source[7]) != grid_audit.build_manifest_sha256
            or _SHA256.fullmatch(str(source[7])) is None
        ):
            raise ValueError("Round 73 target source identity differs")
        coverage_start_wall_ns = int(source[1])
        coverage_end_wall_ns = int(source[2])
        run_started_wall_ns = int(source[5])
        run_started_monotonic_ns = int(source[6])
        segment_rows = connection.execute(
            "SELECT symbol, status, tick_size FROM impact_capture_segment "
            "WHERE run_id = ? ORDER BY symbol",
            [selected],
        ).fetchall()
        if tuple(str(row[0]) for row in segment_rows) != IMPACT_CAPTURE_SYMBOLS or any(
            str(row[1]) != "valid" for row in segment_rows
        ):
            raise ValueError("Round 73 target source segments are invalid")
        anchor_rows = connection.execute(
            f"""
            SELECT symbol, anchor_index, anchor_monotonic_ns, anchor_wall_ns,
                   source_max_received_monotonic_ns
            FROM {ROUND73_GRID_ANCHOR_TABLE}
            WHERE run_id = ? AND valid
            ORDER BY symbol, anchor_index
            """,
            [selected],
        ).fetchall()
        anchors: dict[str, list[_Anchor]] = {
            symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        for row in anchor_rows:
            anchor = _Anchor(
                symbol=str(row[0]),
                anchor_index=int(row[1]),
                decision_monotonic_ns=int(row[2]),
                decision_wall_ns=int(row[3]),
                source_max_received_monotonic_ns=int(row[4]),
            )
            anchors[anchor.symbol].append(anchor)
        if sum(len(values) for values in anchors.values()) != grid_audit.valid_anchor_count:
            raise ValueError("Round 73 target valid-anchor count differs")
        quantity_rules = _parse_quantity_rules(connection, run_id=selected)
        options = _target_rows_v9(
            connection,
            run_id=selected,
            anchors=anchors,
            quantity_rules=quantity_rules,
            run_started_wall_ns=run_started_wall_ns,
            run_started_monotonic_ns=run_started_monotonic_ns,
            coverage_start_wall_ns=coverage_start_wall_ns,
            coverage_end_wall_ns=coverage_end_wall_ns,
            segments=segment_rows,
        )
    for option in options:
        errors = (
            *_option_invariant_errors(option),
            *_quantity_invariant_errors(option, quantity_rules[option.symbol]),
        )
        if errors:
            raise ValueError(
                "Round 73 target financial invariant failed: "
                f"{option.symbol}:{option.anchor_index}:{','.join(errors)}"
            )
    identity = _build_identity(
        run_id=selected,
        source_corpus_manifest_sha256=corpus_audit.manifest_sha256,
        source_grid_manifest_sha256=grid_audit.build_manifest_sha256,
        options=options,
        valid_anchor_count=grid_audit.valid_anchor_count,
        first_decision_wall_ns=min(anchor.decision_wall_ns for values in anchors.values() for anchor in values),
        last_decision_wall_ns=max(anchor.decision_wall_ns for values in anchors.values() for anchor in values),
    )
    manifest_text = _canonical_json(identity)
    manifest_sha256 = _sha256_text(manifest_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_target_tables(connection)
        _assert_target_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            concurrent = connection.execute(
                f"SELECT target_manifest_sha256 FROM {ROUND73_TARGET_MANIFEST_TABLE} "
                "WHERE run_id = ?",
                [selected],
            ).fetchone()
            if concurrent is not None:
                if str(concurrent[0]) != manifest_sha256:
                    raise ValueError("Round 73 concurrent target build differs")
            else:
                for batch_index, start in enumerate(
                    range(0, len(options), _INSERT_BATCH_SIZE)
                ):
                    _insert_option_batch(
                        connection,
                        options[start : start + _INSERT_BATCH_SIZE],
                        batch_index=batch_index,
                    )
                connection.execute(
                    f"INSERT INTO {ROUND73_TARGET_MANIFEST_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected,
                        ROUND73_TARGET_SCHEMA_VERSION,
                        ROUND73_TARGET_CONTRACT_SHA256,
                        corpus_audit.manifest_sha256,
                        grid_audit.build_manifest_sha256,
                        manifest_text,
                        manifest_sha256,
                        str(identity["option_rows_sha256"]),
                        int(identity["valid_anchor_count"]),
                        int(identity["option_count"]),
                        int(identity["eligible_option_count"]),
                        int(identity["positive_option_count"]),
                        int(identity["first_decision_wall_ns"]),
                        int(identity["last_decision_wall_ns"]),
                        time.time_ns(),
                    ],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _report_from_identity(identity, manifest_sha256)


def audit_round73_executable_targets(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73TargetAudit:
    """Reconcile every target row, source identity, and financial invariant."""

    selected = _validated_run_id(run_id)
    errors: list[str] = []
    option_count = 0
    eligible_count = 0
    positive_count = 0
    manifest_sha256 = ""
    source_grid_manifest_sha256 = ""
    try:
        corpus_audit = audit_round73_corpus_manifest(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
        )
        grid_audit = audit_round73_causal_grid(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
        )
        if not corpus_audit.passed:
            errors.append("source_corpus_manifest_audit_failed")
        if not grid_audit.passed:
            errors.append("source_grid_audit_failed")
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"source:{type(exc).__name__}:{exc}")
        corpus_audit = None
        grid_audit = None
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        if not all(
            _table_exists(connection, table)
            for table in (ROUND73_TARGET_OPTION_TABLE, ROUND73_TARGET_MANIFEST_TABLE)
        ):
            raise ValueError("Round 73 target table set is incomplete")
        _assert_target_table_shapes(connection)
        row = connection.execute(
            f"""
            SELECT schema_version, contract_sha256,
                   source_corpus_manifest_sha256, source_grid_manifest_sha256,
                   target_manifest_json, target_manifest_sha256,
                   option_rows_sha256, valid_anchor_count, option_count,
                   eligible_option_count, positive_option_count,
                   first_decision_wall_ns, last_decision_wall_ns
            FROM {ROUND73_TARGET_MANIFEST_TABLE} WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise ValueError("Round 73 target manifest was not found")
        manifest_sha256 = str(row[5])
        source_grid_manifest_sha256 = str(row[3])
        try:
            manifest_text = str(row[4])
            if (
                str(row[0]) != ROUND73_TARGET_SCHEMA_VERSION
                or str(row[1]) != ROUND73_TARGET_CONTRACT_SHA256
                or corpus_audit is None
                or grid_audit is None
                or str(row[2]) != corpus_audit.manifest_sha256
                or source_grid_manifest_sha256 != grid_audit.build_manifest_sha256
                or _sha256_text(manifest_text) != manifest_sha256
                or _SHA256.fullmatch(manifest_sha256) is None
            ):
                raise ValueError("target manifest identity differs")
            identity = _strict_json_object(manifest_text, "target manifest")
            if (
                identity.get("schema_version") != ROUND73_TARGET_SCHEMA_VERSION
                or identity.get("contract_sha256") != ROUND73_TARGET_CONTRACT_SHA256
                or identity.get("run_id") != selected
                or identity.get("source_corpus_manifest_sha256") != str(row[2])
                or identity.get("source_grid_manifest_sha256") != str(row[3])
                or identity.get("expected_options_per_valid_anchor")
                != _EXPECTED_OPTIONS_PER_ANCHOR
                or identity.get("crypto_formal_daily_close") is not False
                or identity.get("target_constructed") is not True
                or identity.get("model_evaluated") is not False
                or identity.get("profitability_claim") is not False
                or identity.get("trading_authority") is not False
            ):
                raise ValueError("target manifest fields differ")
            quantity_rules = _parse_quantity_rules(connection, run_id=selected)
            query = (
                "SELECT "
                + ", ".join(_OPTION_COLUMNS)
                + f" FROM {ROUND73_TARGET_OPTION_TABLE} WHERE run_id = ? "
                "ORDER BY symbol, anchor_index, entry_delay_ms, horizon_ms, "
                "reference_quote_notional, side"
            )
            cursor = connection.cursor()
            cursor.execute(query, [selected])
            digest = hashlib.sha256()
            reason_counts = {reason: 0 for reason in _INELIGIBLE_BITS}
            per_symbol = {
                symbol: {
                    "options": 0,
                    "eligible": 0,
                    "ineligible": 0,
                    "positive": 0,
                }
                for symbol in IMPACT_CAPTURE_SYMBOLS
            }
            observed_dimensions: list[tuple[int, int, float, str]] = []
            expected_dimensions = sorted(
                (
                    delay,
                    horizon,
                    reference,
                    side,
                )
                for delay in ROUND73_TARGET_ENTRY_DELAYS_MS
                for horizon in ROUND73_TARGET_HORIZONS_MS
                for reference in ROUND73_TARGET_REFERENCE_NOTIONALS
                for side in ROUND73_TARGET_SIDES
            )
            prior_anchor: tuple[str, int] | None = None
            try:
                while batch := cursor.fetchmany(_FETCH_BATCH_SIZE):
                    for raw_row in batch:
                        option = _option_from_row(raw_row)
                        invariant_errors = (
                            *_option_invariant_errors(option),
                            *_quantity_invariant_errors(
                                option,
                                quantity_rules[option.symbol],
                            ),
                        )
                        if invariant_errors:
                            raise ValueError(
                                "target row financial invariants differ: "
                                f"{option.symbol}:{option.anchor_index}:"
                                + ",".join(invariant_errors)
                            )
                        current_anchor = (option.symbol, option.anchor_index)
                        if prior_anchor is not None and current_anchor != prior_anchor:
                            if observed_dimensions != expected_dimensions:
                                raise ValueError("target anchor dimensions differ")
                            observed_dimensions = []
                        prior_anchor = current_anchor
                        observed_dimensions.append(
                            (
                                option.entry_delay_ms,
                                option.horizon_ms,
                                option.reference_quote_notional,
                                option.side,
                            )
                        )
                        digest.update(option.option_sha256.encode("ascii"))
                        digest.update(b"\n")
                        option_count += 1
                        eligible_count += int(option.eligible)
                        positive_count += int(
                            option.eligible and bool(option.positive_net_payoff)
                        )
                        symbol_counts = per_symbol[option.symbol]
                        symbol_counts["options"] += 1
                        if option.eligible:
                            symbol_counts["eligible"] += 1
                            symbol_counts["positive"] += int(
                                bool(option.positive_net_payoff)
                            )
                        else:
                            symbol_counts["ineligible"] += 1
                            for reason, bit in _INELIGIBLE_BITS.items():
                                reason_counts[reason] += int(
                                    bool(option.ineligible_reason_mask & bit)
                                )
                if observed_dimensions != expected_dimensions:
                    raise ValueError("target final anchor dimensions differ")
            finally:
                cursor.close()
            valid_anchor_count = int(row[7])
            expected_option_count = valid_anchor_count * _EXPECTED_OPTIONS_PER_ANCHOR
            invalid_anchor_options = connection.execute(
                f"""
                SELECT count(*)
                FROM {ROUND73_TARGET_OPTION_TABLE} t
                LEFT JOIN {ROUND73_GRID_ANCHOR_TABLE} a
                  ON a.run_id = t.run_id AND a.symbol = t.symbol
                 AND a.anchor_index = t.anchor_index
                WHERE t.run_id = ? AND (a.run_id IS NULL OR NOT a.valid)
                """,
                [selected],
            ).fetchone()[0]
            decision_bounds = connection.execute(
                f"SELECT min(a.anchor_wall_ns), max(a.anchor_wall_ns) "
                f"FROM {ROUND73_GRID_ANCHOR_TABLE} a "
                "WHERE a.run_id = ? AND a.valid",
                [selected],
            ).fetchone()
            if (
                int(invalid_anchor_options) != 0
                or grid_audit.valid_anchor_count != valid_anchor_count
                or option_count != expected_option_count
                or option_count != int(row[8])
                or eligible_count != int(row[9])
                or positive_count != int(row[10])
                or digest.hexdigest() != str(row[6])
                or str(row[6]) != identity.get("option_rows_sha256")
                or option_count != identity.get("option_count")
                or eligible_count != identity.get("eligible_option_count")
                or positive_count != identity.get("positive_option_count")
                or reason_counts != identity.get("reason_counts")
                or per_symbol != identity.get("per_symbol")
                or int(decision_bounds[0]) != int(row[11])
                or int(decision_bounds[1]) != int(row[12])
                or int(row[11]) != identity.get("first_decision_wall_ns")
                or int(row[12]) != identity.get("last_decision_wall_ns")
            ):
                raise ValueError("target aggregate reconciliation differs")
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"target:{type(exc).__name__}:{exc}")
    return Round73TargetAudit(
        run_id=selected,
        passed=not errors,
        errors=tuple(errors),
        option_count=option_count,
        eligible_option_count=eligible_count,
        positive_option_count=positive_count,
        target_manifest_sha256=manifest_sha256,
        source_grid_manifest_sha256=source_grid_manifest_sha256,
    )


__all__ = [
    "ROUND73_TARGET_MANIFEST_TABLE",
    "ROUND73_TARGET_OPTION_TABLE",
    "Round73TargetAudit",
    "Round73TargetBuildReport",
    "Round73TargetOption",
    "audit_round73_executable_targets",
    "build_round73_executable_targets",
]
