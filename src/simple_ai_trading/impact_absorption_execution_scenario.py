"""Source-bound public-mainnet execution scenarios for Round 74 targets.

The scenario is intentionally not a mainnet fill estimate. It retains the
testnet empirical execution tail as an upstream stress input, adds a
distribution-free upper bound for production public-feed receipt delay, and
requires the target engine to walk the exact future production L2 state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_EVIDENCE_SOURCES,
    ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS,
    ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE,
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_latency_evidence_claims,
    round74_slippage_evidence_claims,
)
from .impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
    Round74ExecutionEvidenceBundle,
)


ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION = (
    "round-074-public-mainnet-execution-scenario-v1"
)
ROUND74_PUBLIC_TRANSPORT_SOURCE_SCHEMA_VERSION = (
    "round-074-public-mainnet-transport-source-v1"
)
ROUND74_EXECUTION_AGGREGATE_SCHEMA_VERSION = (
    "round-074-execution-calibration-aggregate-v1"
)
ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID = (
    "round74_public_mainnet_execution_latency_scenario_v1"
)
ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID = (
    "round74_public_mainnet_residual_shortfall_scenario_v1"
)
ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME = "production_feed_stressed"
ROUND74_PUBLIC_EXECUTION_SCENARIO_MAXIMUM_SOURCE_LATENCY_NS = 60_000_000_000
_MAXIMUM_EXECUTION_AGGREGATE_BYTES = 64 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 public execution scenario is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Round 74 public execution {label} digest differs")
    return str(value)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        selected: dict[str, object] = {}
        for key, value in pairs:
            if key in selected:
                raise ValueError(f"duplicate key: {key}")
            selected[key] = value
        return selected

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite value: {value}")

    try:
        parsed = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Round 74 public execution {label} JSON differs") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Round 74 public execution {label} root differs")
    return parsed


def _upper_confidence_quantile(
    values: Sequence[int],
    *,
    quantile: float,
    confidence: float,
) -> int:
    if not values:
        raise ValueError("Round 74 public execution transport sample is empty")
    ordered = sorted(values)
    sample_count = len(ordered)
    cumulative = 0.0
    for below_count in range(sample_count):
        cumulative += (
            math.comb(sample_count, below_count)
            * quantile**below_count
            * (1.0 - quantile) ** (sample_count - below_count)
        )
        if cumulative >= confidence:
            return int(ordered[below_count])
    raise ValueError("Round 74 public execution sample cannot bound the requested tail")


ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT = {
    "schema_version": "round-074-public-mainnet-execution-scenario-contract-v1",
    "execution_environment": "binance_usdm_mainnet",
    "symbols": list(ROUND74_EVENT_TARGET_SYMBOLS),
    "scenario_panel": [
        "testnet_empirical_tail_input",
        ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME,
    ],
    "selected_scenario": ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME,
    "testnet_execution_equivalence_claim": False,
    "testnet_to_mainnet_multiplier_permitted": False,
    "synthetic_mainnet_fill_point_estimate_permitted": False,
    "production_feed_delay_estimator": (
        "distribution-free one-sided 95 percent upper confidence "
        "order statistic for p99"
    ),
    "production_feed_delay_quantile": (ROUND74_EXECUTION_CALIBRATION_QUANTILE),
    "production_feed_delay_confidence": (
        ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE
    ),
    "production_feed_delay_clock_uncertainty": (
        "causal clock-correction plus one-half observed REST clock-probe RTT"
    ),
    "selected_entry_latency": (
        "testnet empirical entry p99 upper confidence tail plus production "
        "public-feed p99 upper confidence receipt delay"
    ),
    "selected_exit_latency": (
        "testnet empirical exit p99 upper confidence tail plus production "
        "public-feed p99 upper confidence receipt delay"
    ),
    "residual_shortfall": (
        "testnet empirical p99 upper confidence residual retained as a "
        "conservative scenario input, not relabeled as a mainnet fill"
    ),
    "entry_and_exit_price_source": (
        "exact run-bound production public L2 book walk after selected delay"
    ),
    "interpolation_permitted": False,
    "invented_liquidity_permitted": False,
    "scenario_tuning_on_sealed_test_permitted": False,
    "orders_submitted": False,
    "trading_authority": False,
}
ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256 = _canonical_sha256(
    ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT
)


def _symbol_integer_mapping(
    value: Sequence[tuple[str, int]],
    *,
    label: str,
    allow_zero: bool,
) -> dict[str, int]:
    selected = dict(value)
    if tuple(symbol for symbol, _item in value) != ROUND74_EVENT_TARGET_SYMBOLS or len(
        selected
    ) != len(value):
        raise ValueError(f"Round 74 public execution {label} universe differs")
    for item in selected.values():
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < (0 if allow_zero else 1)
        ):
            raise ValueError(f"Round 74 public execution {label} value differs")
    return selected


def _symbol_float_mapping(
    value: Sequence[tuple[str, float]],
    *,
    label: str,
) -> dict[str, float]:
    selected = dict(value)
    if (
        tuple(symbol for symbol, _item in value) != ROUND74_EVENT_TARGET_SYMBOLS
        or len(selected) != len(value)
        or any(
            not math.isfinite(float(item))
            or not 0.0
            <= float(item)
            <= ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE
            for item in selected.values()
        )
    ):
        raise ValueError(f"Round 74 public execution {label} panel differs")
    return {symbol: float(item) for symbol, item in selected.items()}


def _validate_testnet_execution_bundle(
    bundle: Round74ExecutionEvidenceBundle,
) -> None:
    reference = float(bundle.reference_quote_notional)
    entries = _symbol_integer_mapping(
        bundle.decision_to_entry_latency_ns_by_symbol,
        label="testnet entry latency",
        allow_zero=False,
    )
    exits = _symbol_integer_mapping(
        bundle.decision_to_exit_latency_ns_by_symbol,
        label="testnet exit latency",
        allow_zero=False,
    )
    slippage = _symbol_float_mapping(
        bundle.additional_slippage_bps_per_side_by_symbol,
        label="testnet residual shortfall",
    )
    latency_evidence = bundle.entry_exit_latency_evidence
    slippage_evidence = bundle.residual_slippage_evidence
    if (
        not math.isfinite(reference)
        or reference <= 0.0
        or any(
            value > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
            for value in (*entries.values(), *exits.values())
        )
        or latency_evidence.environment != "binance_usdm_testnet"
        or slippage_evidence.environment != "binance_usdm_testnet"
        or latency_evidence.kind != "entry_exit_latency"
        or slippage_evidence.kind != "residual_slippage"
        or latency_evidence.source_id
        != ROUND74_EVENT_TARGET_EVIDENCE_SOURCES["entry_exit_latency"]
        or slippage_evidence.source_id
        != ROUND74_EVENT_TARGET_EVIDENCE_SOURCES["residual_slippage"]
        or latency_evidence.record_count != slippage_evidence.record_count
        or latency_evidence.record_count
        < (
            2
            * len(ROUND74_EVENT_TARGET_SYMBOLS)
            * ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
        )
        or not latency_evidence.binds(
            round74_latency_evidence_claims(
                decision_to_entry_latency_ns_by_symbol=entries,
                decision_to_exit_latency_ns_by_symbol=exits,
            )
        )
        or not slippage_evidence.binds(
            round74_slippage_evidence_claims(
                reference_quote_notional=reference,
                additional_slippage_bps_per_side_by_symbol=slippage,
            )
        )
    ):
        raise ValueError("Round 74 public execution testnet calibration differs")


@dataclass(frozen=True)
class Round74PublicTransportSource:
    """Run-bound production public-feed receipt-delay observations."""

    run_id: str
    cohort_binding_sha256: str
    capture_report_sha256: str
    observed_wall_ns: int
    source_payload_sha256: str
    transport_latency_ns_by_symbol: tuple[tuple[str, tuple[int, ...]], ...]
    schema_version: str = ROUND74_PUBLIC_TRANSPORT_SOURCE_SCHEMA_VERSION

    def validate(self) -> None:
        panels = dict(self.transport_latency_ns_by_symbol)
        if (
            self.schema_version != ROUND74_PUBLIC_TRANSPORT_SOURCE_SCHEMA_VERSION
            or len(self.run_id) != 32
            or any(character not in _SHA256_CHARACTERS for character in self.run_id)
            or int(self.observed_wall_ns) <= 0
            or tuple(symbol for symbol, _values in self.transport_latency_ns_by_symbol)
            != ROUND74_EVENT_TARGET_SYMBOLS
            or len(panels) != len(self.transport_latency_ns_by_symbol)
        ):
            raise ValueError("Round 74 public execution transport identity differs")
        _require_sha256(self.cohort_binding_sha256, "cohort binding")
        _require_sha256(self.capture_report_sha256, "capture report")
        _require_sha256(self.source_payload_sha256, "transport source")
        for values in panels.values():
            if len(
                values
            ) < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0
                <= value
                <= ROUND74_PUBLIC_EXECUTION_SCENARIO_MAXIMUM_SOURCE_LATENCY_NS
                for value in values
            ):
                raise ValueError("Round 74 public execution transport sample differs")

    def sample_count_mapping(self) -> dict[str, int]:
        self.validate()
        return {
            symbol: len(values)
            for symbol, values in self.transport_latency_ns_by_symbol
        }

    def upper_tail_mapping(self) -> dict[str, int]:
        self.validate()
        return {
            symbol: _upper_confidence_quantile(
                values,
                quantile=ROUND74_EXECUTION_CALIBRATION_QUANTILE,
                confidence=(ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE),
            )
            for symbol, values in self.transport_latency_ns_by_symbol
        }


@dataclass(frozen=True)
class Round74ExecutionAggregateSource:
    """Validated immutable testnet execution aggregate and file identity."""

    bundle: Round74ExecutionEvidenceBundle
    artifact_sha256: str
    artifact_file_sha256: str
    observed_wall_ns: int
    source_record_count: int

    def validate(self) -> None:
        _validate_testnet_execution_bundle(self.bundle)
        _require_sha256(self.artifact_sha256, "aggregate artifact")
        _require_sha256(
            self.artifact_file_sha256,
            "aggregate artifact file",
        )
        if (
            int(self.observed_wall_ns) <= 0
            or int(self.source_record_count)
            != self.bundle.entry_exit_latency_evidence.record_count
            or self.source_record_count
            != self.bundle.residual_slippage_evidence.record_count
        ):
            raise ValueError("Round 74 public execution aggregate source differs")


@dataclass(frozen=True)
class Round74PublicExecutionScenarioArtifactSource:
    """Validated scenario payload and its immutable file identities."""

    bundle: Round74PublicExecutionScenarioBundle
    artifact_sha256: str
    artifact_file_sha256: str

    def validate(self) -> None:
        self.bundle.validate()
        _require_sha256(self.artifact_sha256, "scenario artifact")
        _require_sha256(
            self.artifact_file_sha256,
            "scenario artifact file",
        )


@dataclass(frozen=True)
class Round74PublicExecutionScenarioBundle:
    """Selected conservative scenario and mainnet-scoped target evidence."""

    run_id: str
    cohort_binding_sha256: str
    capture_report_sha256: str
    transport_source_payload_sha256: str
    testnet_aggregate_artifact_sha256: str
    testnet_aggregate_artifact_file_sha256: str
    testnet_latency_evidence_sha256: str
    testnet_slippage_evidence_sha256: str
    reference_quote_notional: float
    transport_sample_count_by_symbol: tuple[tuple[str, int], ...]
    transport_latency_upper_ns_by_symbol: tuple[tuple[str, int], ...]
    testnet_entry_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    testnet_exit_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    selected_entry_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    selected_exit_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    selected_residual_slippage_bps_by_symbol: tuple[
        tuple[str, float],
        ...,
    ]
    entry_exit_latency_evidence: Round74EventTargetEvidence
    residual_slippage_evidence: Round74EventTargetEvidence
    observed_wall_ns: int
    scenario_contract_sha256: str = ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
    schema_version: str = ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION

    def validate(self) -> None:
        counts = _symbol_integer_mapping(
            self.transport_sample_count_by_symbol,
            label="transport count",
            allow_zero=False,
        )
        transport = _symbol_integer_mapping(
            self.transport_latency_upper_ns_by_symbol,
            label="transport tail",
            allow_zero=True,
        )
        base_entry = _symbol_integer_mapping(
            self.testnet_entry_latency_ns_by_symbol,
            label="testnet entry latency",
            allow_zero=False,
        )
        base_exit = _symbol_integer_mapping(
            self.testnet_exit_latency_ns_by_symbol,
            label="testnet exit latency",
            allow_zero=False,
        )
        selected_entry = _symbol_integer_mapping(
            self.selected_entry_latency_ns_by_symbol,
            label="selected entry latency",
            allow_zero=False,
        )
        selected_exit = _symbol_integer_mapping(
            self.selected_exit_latency_ns_by_symbol,
            label="selected exit latency",
            allow_zero=False,
        )
        slippage = _symbol_float_mapping(
            self.selected_residual_slippage_bps_by_symbol,
            label="selected residual shortfall",
        )
        if (
            self.schema_version != ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION
            or self.scenario_contract_sha256
            != ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
            or len(self.run_id) != 32
            or any(character not in _SHA256_CHARACTERS for character in self.run_id)
            or int(self.observed_wall_ns) <= 0
            or not math.isfinite(float(self.reference_quote_notional))
            or float(self.reference_quote_notional) <= 0.0
            or any(
                count < ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
                for count in counts.values()
            )
            or any(
                selected_entry[symbol] != base_entry[symbol] + transport[symbol]
                or selected_exit[symbol] != base_exit[symbol] + transport[symbol]
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            )
            or any(
                value > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
                for value in (
                    *selected_entry.values(),
                    *selected_exit.values(),
                )
            )
        ):
            raise ValueError("Round 74 public execution scenario policy differs")
        for digest, label in (
            (self.cohort_binding_sha256, "cohort binding"),
            (self.capture_report_sha256, "capture report"),
            (self.transport_source_payload_sha256, "transport source"),
            (
                self.testnet_aggregate_artifact_sha256,
                "aggregate artifact",
            ),
            (
                self.testnet_aggregate_artifact_file_sha256,
                "aggregate artifact file",
            ),
            (
                self.testnet_latency_evidence_sha256,
                "testnet latency evidence",
            ),
            (
                self.testnet_slippage_evidence_sha256,
                "testnet slippage evidence",
            ),
        ):
            _require_sha256(digest, label)
        if (
            self.entry_exit_latency_evidence.environment != "binance_usdm_mainnet"
            or self.residual_slippage_evidence.environment != "binance_usdm_mainnet"
            or self.entry_exit_latency_evidence.source_id
            != ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID
            or self.residual_slippage_evidence.source_id
            != ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID
            or not self.entry_exit_latency_evidence.binds(
                round74_latency_evidence_claims(
                    decision_to_entry_latency_ns_by_symbol=selected_entry,
                    decision_to_exit_latency_ns_by_symbol=selected_exit,
                )
            )
            or not self.residual_slippage_evidence.binds(
                round74_slippage_evidence_claims(
                    reference_quote_notional=(self.reference_quote_notional),
                    additional_slippage_bps_per_side_by_symbol=slippage,
                )
            )
        ):
            raise ValueError("Round 74 public execution scenario evidence differs")

    @property
    def scenario_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def entry_latency_mapping(self) -> dict[str, int]:
        self.validate()
        return dict(self.selected_entry_latency_ns_by_symbol)

    def exit_latency_mapping(self) -> dict[str, int]:
        self.validate()
        return dict(self.selected_exit_latency_ns_by_symbol)

    def slippage_mapping(self) -> dict[str, float]:
        self.validate()
        return dict(self.selected_residual_slippage_bps_by_symbol)

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        base_entry = dict(self.testnet_entry_latency_ns_by_symbol)
        base_exit = dict(self.testnet_exit_latency_ns_by_symbol)
        selected_entry = dict(self.selected_entry_latency_ns_by_symbol)
        selected_exit = dict(self.selected_exit_latency_ns_by_symbol)
        slippage = dict(self.selected_residual_slippage_bps_by_symbol)
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "environment": "binance_usdm_mainnet",
            "run_id": self.run_id,
            "scenario_contract_sha256": self.scenario_contract_sha256,
            "cohort_binding_sha256": self.cohort_binding_sha256,
            "capture_report_sha256": self.capture_report_sha256,
            "observed_wall_ns": self.observed_wall_ns,
            "reference_quote_notional": self.reference_quote_notional,
            "upstream_testnet_calibration": {
                "source_venue": "binance_usdm_testnet",
                "aggregate_artifact_sha256": (self.testnet_aggregate_artifact_sha256),
                "aggregate_artifact_file_sha256": (
                    self.testnet_aggregate_artifact_file_sha256
                ),
                "latency_evidence_sha256": (self.testnet_latency_evidence_sha256),
                "slippage_evidence_sha256": (self.testnet_slippage_evidence_sha256),
                "mainnet_execution_equivalence": False,
                "mainnet_transfer_permitted": False,
            },
            "public_mainnet_transport": {
                "source_payload_sha256": (self.transport_source_payload_sha256),
                "sample_count_by_symbol": dict(self.transport_sample_count_by_symbol),
                "p99_upper_confidence_ns_by_symbol": dict(
                    self.transport_latency_upper_ns_by_symbol
                ),
            },
            "scenario_panel": [
                {
                    "name": "testnet_empirical_tail_input",
                    "entry_latency_ns_by_symbol": base_entry,
                    "exit_latency_ns_by_symbol": base_exit,
                    "residual_slippage_bps_by_symbol": slippage,
                    "mainnet_fill_estimate": False,
                    "selected": False,
                },
                {
                    "name": ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME,
                    "entry_latency_ns_by_symbol": selected_entry,
                    "exit_latency_ns_by_symbol": selected_exit,
                    "residual_slippage_bps_by_symbol": slippage,
                    "exact_future_public_l2_replay_required": True,
                    "mainnet_fill_estimate": False,
                    "selected": True,
                },
            ],
            "selected_scenario": (ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME),
            "entry_exit_latency_evidence": (self.entry_exit_latency_evidence.as_dict()),
            "residual_slippage_evidence": (self.residual_slippage_evidence.as_dict()),
            "authority": {
                "testnet_execution_equivalence": False,
                "mainnet_fill_evidence": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
                "orders_submitted": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        if include_sha256:
            value["scenario_sha256"] = _canonical_sha256(value)
        return value


def load_round74_public_execution_scenario_artifact(
    path: str | Path,
) -> Round74PublicExecutionScenarioArtifactSource:
    """Load one scenario artifact and reconstruct every derived field."""

    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or selected.stat().st_size > _MAXIMUM_EXECUTION_AGGREGATE_BYTES
    ):
        raise ValueError("Round 74 public execution scenario artifact file differs")
    raw = selected.read_bytes()
    payload = _strict_json_object(raw, label="scenario artifact")
    artifact_sha256 = str(payload.pop("artifact_sha256", ""))
    if artifact_sha256 != _canonical_sha256(payload):
        raise ValueError("Round 74 public execution scenario artifact digest differs")
    upstream = payload.get("upstream_testnet_calibration")
    transport = payload.get("public_mainnet_transport")
    panel = payload.get("scenario_panel")
    latency_evidence = payload.get("entry_exit_latency_evidence")
    slippage_evidence = payload.get("residual_slippage_evidence")
    if (
        not isinstance(upstream, Mapping)
        or not isinstance(transport, Mapping)
        or not isinstance(panel, list)
        or len(panel) != 2
        or any(not isinstance(row, Mapping) for row in panel)
        or not isinstance(latency_evidence, Mapping)
        or not isinstance(slippage_evidence, Mapping)
    ):
        raise ValueError("Round 74 public execution scenario artifact payload differs")
    base = panel[0]
    stressed = panel[1]
    counts = transport.get("sample_count_by_symbol")
    tails = transport.get("p99_upper_confidence_ns_by_symbol")
    base_entries = base.get("entry_latency_ns_by_symbol")
    base_exits = base.get("exit_latency_ns_by_symbol")
    selected_entries = stressed.get("entry_latency_ns_by_symbol")
    selected_exits = stressed.get("exit_latency_ns_by_symbol")
    selected_slippage = stressed.get("residual_slippage_bps_by_symbol")
    mappings = (
        counts,
        tails,
        base_entries,
        base_exits,
        selected_entries,
        selected_exits,
        selected_slippage,
    )
    if any(not isinstance(value, Mapping) for value in mappings):
        raise ValueError("Round 74 public execution scenario artifact panel differs")
    assert isinstance(counts, Mapping)
    assert isinstance(tails, Mapping)
    assert isinstance(base_entries, Mapping)
    assert isinstance(base_exits, Mapping)
    assert isinstance(selected_entries, Mapping)
    assert isinstance(selected_exits, Mapping)
    assert isinstance(selected_slippage, Mapping)
    try:
        bundle = Round74PublicExecutionScenarioBundle(
            run_id=str(payload["run_id"]),
            cohort_binding_sha256=str(payload["cohort_binding_sha256"]),
            capture_report_sha256=str(payload["capture_report_sha256"]),
            transport_source_payload_sha256=str(transport["source_payload_sha256"]),
            testnet_aggregate_artifact_sha256=str(
                upstream["aggregate_artifact_sha256"]
            ),
            testnet_aggregate_artifact_file_sha256=str(
                upstream["aggregate_artifact_file_sha256"]
            ),
            testnet_latency_evidence_sha256=str(upstream["latency_evidence_sha256"]),
            testnet_slippage_evidence_sha256=str(upstream["slippage_evidence_sha256"]),
            reference_quote_notional=float(payload["reference_quote_notional"]),
            transport_sample_count_by_symbol=tuple(
                (symbol, int(counts[symbol])) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            transport_latency_upper_ns_by_symbol=tuple(
                (symbol, int(tails[symbol])) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            testnet_entry_latency_ns_by_symbol=tuple(
                (symbol, int(base_entries[symbol]))
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            testnet_exit_latency_ns_by_symbol=tuple(
                (symbol, int(base_exits[symbol]))
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            selected_entry_latency_ns_by_symbol=tuple(
                (symbol, int(selected_entries[symbol]))
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            selected_exit_latency_ns_by_symbol=tuple(
                (symbol, int(selected_exits[symbol]))
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            selected_residual_slippage_bps_by_symbol=tuple(
                (symbol, float(selected_slippage[symbol]))
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
            entry_exit_latency_evidence=(
                Round74EventTargetEvidence.from_dict(latency_evidence)
            ),
            residual_slippage_evidence=(
                Round74EventTargetEvidence.from_dict(slippage_evidence)
            ),
            observed_wall_ns=int(payload["observed_wall_ns"]),
            scenario_contract_sha256=str(payload["scenario_contract_sha256"]),
            schema_version=str(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Round 74 public execution scenario artifact values differ"
        ) from exc
    bundle.validate()
    if bundle.as_dict() != payload:
        raise ValueError("Round 74 public execution scenario artifact policy differs")
    source = Round74PublicExecutionScenarioArtifactSource(
        bundle=bundle,
        artifact_sha256=artifact_sha256,
        artifact_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    source.validate()
    return source


def load_round74_execution_aggregate_source(
    path: str | Path,
) -> Round74ExecutionAggregateSource:
    """Load one immutable aggregate without accepting environment transfer."""

    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or selected.stat().st_size > _MAXIMUM_EXECUTION_AGGREGATE_BYTES
    ):
        raise ValueError("Round 74 public execution aggregate file differs")
    raw = selected.read_bytes()
    payload = _strict_json_object(raw, label="aggregate")
    artifact_sha256 = str(payload.pop("artifact_sha256", ""))
    execution = payload.get("execution_evidence")
    authority = payload.get("authority")
    campaign = payload.get("campaign_plan")
    captures = payload.get("source_capture_artifacts")
    aggregator = payload.get("aggregator_source")
    expected_keys = {
        "schema_version",
        "operation",
        "environment",
        "campaign_plan",
        "source_capture_artifacts",
        "source_capture_artifact_count",
        "source_record_count",
        "observed_wall_ns",
        "execution_evidence",
        "aggregator_source",
        "network_accessed",
        "orders_submitted",
        "credential_material_read",
        "authority",
    }
    if (
        set(payload) != expected_keys
        or artifact_sha256 != _canonical_sha256(payload)
        or payload["schema_version"] != ROUND74_EXECUTION_AGGREGATE_SCHEMA_VERSION
        or payload["operation"] != "aggregate"
        or payload["environment"] != "binance_usdm_testnet"
        or payload["network_accessed"] is not False
        or payload["orders_submitted"] is not False
        or payload["credential_material_read"] is not False
        or not isinstance(execution, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(campaign, Mapping)
        or not isinstance(captures, list)
        or not isinstance(aggregator, Mapping)
        or campaign.get("slot_count") != 900
        or not _is_sha256(campaign.get("plan_sha256"))
        or not _is_sha256(campaign.get("plan_artifact_sha256"))
        or not _is_sha256(campaign.get("plan_artifact_file_sha256"))
        or not str(campaign.get("campaign_id", "")).strip()
        or payload["source_capture_artifact_count"] != 900
        or len(captures) != 900
        or payload["source_record_count"] != 1800
        or aggregator.get("path") != "tools/aggregate_round74_execution_calibration.py"
        or not _is_sha256(aggregator.get("sha256"))
        or authority.get("testnet_execution_calibration") is not True
        or authority.get("mainnet_execution_equivalence") is not False
        or authority.get("mainnet_transfer_permitted") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("live_trading_authority") is not False
    ):
        raise ValueError("Round 74 public execution aggregate contract differs")
    capture_ordinals: list[int] = []
    for capture in captures:
        if (
            not isinstance(capture, Mapping)
            or set(capture)
            != {
                "slot_ordinal",
                "round_trip_id",
                "capture_artifact_sha256",
                "capture_artifact_file_sha256",
                "pair_sha256",
            }
            or isinstance(capture.get("slot_ordinal"), bool)
            or not isinstance(capture.get("slot_ordinal"), int)
            or not str(capture.get("round_trip_id", "")).strip()
            or not _is_sha256(capture.get("capture_artifact_sha256"))
            or not _is_sha256(capture.get("capture_artifact_file_sha256"))
            or not _is_sha256(capture.get("pair_sha256"))
        ):
            raise ValueError("Round 74 public execution aggregate capture differs")
        capture_ordinals.append(int(capture["slot_ordinal"]))
    if capture_ordinals != list(range(900)):
        raise ValueError("Round 74 public execution aggregate capture order differs")
    expected_execution_keys = {
        "reference_quote_notional",
        "decision_to_entry_latency_ns_by_symbol",
        "decision_to_exit_latency_ns_by_symbol",
        "additional_slippage_bps_per_side_by_symbol",
        "entry_exit_latency_evidence",
        "residual_slippage_evidence",
    }
    entry = execution.get("decision_to_entry_latency_ns_by_symbol")
    exit_values = execution.get("decision_to_exit_latency_ns_by_symbol")
    slippage = execution.get("additional_slippage_bps_per_side_by_symbol")
    latency_evidence = execution.get("entry_exit_latency_evidence")
    slippage_evidence = execution.get("residual_slippage_evidence")
    if (
        set(execution) != expected_execution_keys
        or not isinstance(entry, Mapping)
        or not isinstance(exit_values, Mapping)
        or not isinstance(slippage, Mapping)
        or not isinstance(latency_evidence, Mapping)
        or not isinstance(slippage_evidence, Mapping)
    ):
        raise ValueError("Round 74 public execution aggregate evidence differs")
    bundle = Round74ExecutionEvidenceBundle(
        reference_quote_notional=float(execution["reference_quote_notional"]),
        decision_to_entry_latency_ns_by_symbol=tuple(
            (symbol, int(entry[symbol])) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        decision_to_exit_latency_ns_by_symbol=tuple(
            (symbol, int(exit_values[symbol]))
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        additional_slippage_bps_per_side_by_symbol=tuple(
            (symbol, float(slippage[symbol])) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        entry_exit_latency_evidence=Round74EventTargetEvidence.from_dict(
            latency_evidence
        ),
        residual_slippage_evidence=Round74EventTargetEvidence.from_dict(
            slippage_evidence
        ),
    )
    source = Round74ExecutionAggregateSource(
        bundle=bundle,
        artifact_sha256=artifact_sha256,
        artifact_file_sha256=hashlib.sha256(raw).hexdigest(),
        observed_wall_ns=int(payload["observed_wall_ns"]),
        source_record_count=int(payload["source_record_count"]),
    )
    source.validate()
    if (
        source.observed_wall_ns
        != source.bundle.entry_exit_latency_evidence.observed_wall_ns
        or source.observed_wall_ns
        != source.bundle.residual_slippage_evidence.observed_wall_ns
    ):
        raise ValueError("Round 74 public execution aggregate observation time differs")
    return source


def collect_round74_public_transport_source(
    store: object,
    *,
    binding: Round74SegmentedCohortRunBinding,
) -> Round74PublicTransportSource:
    """Collect fresh run-bound depth receipt delays from the indexed store."""

    from .impact_absorption_store import ImpactAbsorptionStore

    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 public execution transport requires an impact store")
    if not store.read_only:
        raise ValueError(
            "Round 74 public execution transport requires a read-only store"
        )
    binding.validate()
    audit = store.audit_run(binding.run_id)
    if not audit.passed:
        raise ValueError("Round 74 public execution transport frame audit failed")
    connection = store.connect()
    report = connection.execute(
        """
        SELECT report_sha256 FROM impact_capture_report WHERE run_id = ?
        """,
        [binding.run_id],
    ).fetchone()
    if report is None or str(report[0]) != binding.report_sha256:
        raise ValueError("Round 74 public execution transport report differs")
    segments = connection.execute(
        """
        SELECT segment_id, symbol, status, clock_offset_ns, clock_rtt_ns
        FROM impact_capture_segment
        WHERE run_id = ? ORDER BY segment_id
        """,
        [binding.run_id],
    ).fetchall()
    by_segment: dict[str, tuple[str, int, int]] = {}
    for segment_id, symbol, status, offset, rtt in segments:
        selected_segment = str(segment_id)
        selected_symbol = str(symbol)
        if (
            selected_segment in by_segment
            or selected_symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or str(status) != "valid"
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(rtt, bool)
            or not isinstance(rtt, int)
            or rtt <= 0
        ):
            raise ValueError("Round 74 public execution transport segment differs")
        by_segment[selected_segment] = (
            selected_symbol,
            int(offset),
            int(rtt),
        )
    if {symbol for symbol, _offset, _rtt in by_segment.values()} != set(
        ROUND74_EVENT_TARGET_SYMBOLS
    ):
        raise ValueError("Round 74 public execution transport segment universe differs")
    rows = connection.execute(
        """
        SELECT e.segment_id, e.symbol, e.frame_index, e.message_index,
               e.received_wall_ns, e.event_time_ms
        FROM impact_event_index e
        JOIN impact_depth_update d
          ON d.run_id = e.run_id
         AND d.frame_index = e.frame_index
         AND d.message_index = e.message_index
        WHERE e.run_id = ?
          AND e.event_type = 'depthUpdate'
          AND e.event_time_ms IS NOT NULL
          AND d.stale = FALSE
          AND e.received_wall_ns >= ?
          AND e.received_wall_ns <= ?
        ORDER BY e.frame_index, e.message_index
        """,
        [
            binding.run_id,
            binding.feature_ready_wall_ns,
            binding.usable_end_wall_ns,
        ],
    ).fetchall()
    latencies = {symbol: [] for symbol in ROUND74_EVENT_TARGET_SYMBOLS}
    hasher = hashlib.sha256()
    header = {
        "schema_version": ROUND74_PUBLIC_TRANSPORT_SOURCE_SCHEMA_VERSION,
        "run_id": binding.run_id,
        "cohort_binding_sha256": binding.binding_sha256,
        "capture_report_sha256": binding.report_sha256,
        "query": (
            "fresh indexed depthUpdate receipts within the admitted usable "
            "epoch, joined to causal segment clock probes"
        ),
    }
    hasher.update((_canonical_json(header) + "\n").encode("ascii"))
    prior_key: tuple[int, int] | None = None
    observed_wall_ns = 0
    for segment_id, symbol, frame_index, message_index, receipt, event_ms in rows:
        segment = by_segment.get(str(segment_id))
        selected_symbol = str(symbol)
        key = (int(frame_index), int(message_index))
        if (
            segment is None
            or segment[0] != selected_symbol
            or prior_key is not None
            and key <= prior_key
        ):
            raise ValueError("Round 74 public execution transport row differs")
        prior_key = key
        offset_ns = segment[1]
        clock_uncertainty_ns = (segment[2] + 1) // 2
        corrected_upper_ns = (
            int(receipt) + offset_ns + clock_uncertainty_ns - int(event_ms) * 1_000_000
        )
        if (
            not 0
            <= corrected_upper_ns
            <= (ROUND74_PUBLIC_EXECUTION_SCENARIO_MAXIMUM_SOURCE_LATENCY_NS)
        ):
            raise ValueError("Round 74 public execution corrected transport differs")
        latencies[selected_symbol].append(corrected_upper_ns)
        observed_wall_ns = max(observed_wall_ns, int(receipt))
        hasher.update(
            (
                _canonical_json(
                    [
                        str(segment_id),
                        selected_symbol,
                        key[0],
                        key[1],
                        int(receipt),
                        int(event_ms),
                        offset_ns,
                        segment[2],
                        corrected_upper_ns,
                    ]
                )
                + "\n"
            ).encode("ascii")
        )
    source = Round74PublicTransportSource(
        run_id=binding.run_id,
        cohort_binding_sha256=binding.binding_sha256,
        capture_report_sha256=binding.report_sha256,
        observed_wall_ns=observed_wall_ns,
        source_payload_sha256=hasher.hexdigest(),
        transport_latency_ns_by_symbol=tuple(
            (symbol, tuple(latencies[symbol]))
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
    )
    source.validate()
    return source


def build_round74_public_execution_scenario(
    *,
    transport_source: Round74PublicTransportSource,
    execution_aggregate: Round74ExecutionAggregateSource,
) -> Round74PublicExecutionScenarioBundle:
    """Build the preregistered conservative public-mainnet scenario."""

    transport_source.validate()
    execution_aggregate.validate()
    calibration = execution_aggregate.bundle
    transport = transport_source.upper_tail_mapping()
    counts = transport_source.sample_count_mapping()
    base_entry = calibration.entry_latency_mapping()
    base_exit = calibration.exit_latency_mapping()
    slippage = calibration.slippage_mapping()
    selected_entry = {
        symbol: base_entry[symbol] + transport[symbol]
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    selected_exit = {
        symbol: base_exit[symbol] + transport[symbol]
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    if any(
        value > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
        for value in (*selected_entry.values(), *selected_exit.values())
    ):
        raise ValueError(
            "Round 74 public execution selected latency exceeds target bound"
        )
    source_payload = {
        "scenario_contract_sha256": (ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256),
        "run_id": transport_source.run_id,
        "cohort_binding_sha256": (transport_source.cohort_binding_sha256),
        "capture_report_sha256": (transport_source.capture_report_sha256),
        "transport_source_payload_sha256": (transport_source.source_payload_sha256),
        "transport_sample_count_by_symbol": counts,
        "transport_latency_upper_ns_by_symbol": transport,
        "testnet_aggregate_artifact_sha256": (execution_aggregate.artifact_sha256),
        "testnet_aggregate_artifact_file_sha256": (
            execution_aggregate.artifact_file_sha256
        ),
        "testnet_latency_evidence_sha256": (
            calibration.entry_exit_latency_evidence.evidence_sha256
        ),
        "testnet_slippage_evidence_sha256": (
            calibration.residual_slippage_evidence.evidence_sha256
        ),
        "selected_entry_latency_ns_by_symbol": selected_entry,
        "selected_exit_latency_ns_by_symbol": selected_exit,
        "selected_residual_slippage_bps_by_symbol": slippage,
    }
    source_payload_sha256 = _canonical_sha256(source_payload)
    record_count = (
        sum(counts.values()) + calibration.entry_exit_latency_evidence.record_count
    )
    observed_wall_ns = max(
        transport_source.observed_wall_ns,
        execution_aggregate.observed_wall_ns,
    )
    latency_evidence = Round74EventTargetEvidence.create(
        kind="entry_exit_latency",
        source_id=ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID,
        environment="binance_usdm_mainnet",
        observed_wall_ns=observed_wall_ns,
        record_count=record_count,
        source_query_or_protocol_sha256=(
            ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
        ),
        source_payload_sha256=source_payload_sha256,
        claims=round74_latency_evidence_claims(
            decision_to_entry_latency_ns_by_symbol=selected_entry,
            decision_to_exit_latency_ns_by_symbol=selected_exit,
        ),
    )
    slippage_evidence = Round74EventTargetEvidence.create(
        kind="residual_slippage",
        source_id=ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID,
        environment="binance_usdm_mainnet",
        observed_wall_ns=observed_wall_ns,
        record_count=record_count,
        source_query_or_protocol_sha256=(
            ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
        ),
        source_payload_sha256=source_payload_sha256,
        claims=round74_slippage_evidence_claims(
            reference_quote_notional=calibration.reference_quote_notional,
            additional_slippage_bps_per_side_by_symbol=slippage,
        ),
    )
    result = Round74PublicExecutionScenarioBundle(
        run_id=transport_source.run_id,
        cohort_binding_sha256=(transport_source.cohort_binding_sha256),
        capture_report_sha256=(transport_source.capture_report_sha256),
        transport_source_payload_sha256=(transport_source.source_payload_sha256),
        testnet_aggregate_artifact_sha256=(execution_aggregate.artifact_sha256),
        testnet_aggregate_artifact_file_sha256=(
            execution_aggregate.artifact_file_sha256
        ),
        testnet_latency_evidence_sha256=(
            calibration.entry_exit_latency_evidence.evidence_sha256
        ),
        testnet_slippage_evidence_sha256=(
            calibration.residual_slippage_evidence.evidence_sha256
        ),
        reference_quote_notional=calibration.reference_quote_notional,
        transport_sample_count_by_symbol=tuple(
            (symbol, counts[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        transport_latency_upper_ns_by_symbol=tuple(
            (symbol, transport[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        testnet_entry_latency_ns_by_symbol=tuple(
            (symbol, base_entry[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        testnet_exit_latency_ns_by_symbol=tuple(
            (symbol, base_exit[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        selected_entry_latency_ns_by_symbol=tuple(
            (symbol, selected_entry[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        selected_exit_latency_ns_by_symbol=tuple(
            (symbol, selected_exit[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        selected_residual_slippage_bps_by_symbol=tuple(
            (symbol, slippage[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
        entry_exit_latency_evidence=latency_evidence,
        residual_slippage_evidence=slippage_evidence,
        observed_wall_ns=observed_wall_ns,
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT",
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256",
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID",
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION",
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME",
    "ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID",
    "ROUND74_PUBLIC_TRANSPORT_SOURCE_SCHEMA_VERSION",
    "Round74ExecutionAggregateSource",
    "Round74PublicExecutionScenarioArtifactSource",
    "Round74PublicExecutionScenarioBundle",
    "Round74PublicTransportSource",
    "build_round74_public_execution_scenario",
    "collect_round74_public_transport_source",
    "load_round74_execution_aggregate_source",
    "load_round74_public_execution_scenario_artifact",
]
