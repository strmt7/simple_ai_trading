"""Frozen label-free execution program for Polymarket Round 13."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import (
    Context,
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    localcontext,
)
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .paper_execution import PolymarketFeeModel
from .polymarket import POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION
from .polymarket_action_value import (
    PolymarketActionFeature,
    PolymarketActionValueDataset,
)
from .polymarket_features import POLYMARKET_FEATURE_NAMES, PolymarketFeatureDataset
from .polymarket_repricing import (
    PolymarketRecordedBook,
    PolymarketRepricingDecision,
    PolymarketRepricingExecutionContext,
)
from .polymarket_round12_reference import (
    PolymarketRound12PrimaryPolicy,
    PolymarketRound12ReferenceModel,
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
    polymarket_round12_reference_implementation_sha256,
)

if TYPE_CHECKING:
    from .polymarket import PolymarketFiveMinuteMarket
    from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_ROUND13_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round13-sealed-confirmation-contract-v1"
)
POLYMARKET_ROUND13_SCENARIO_SCHEMA_VERSION = "polymarket-round13-label-free-scenario-v1"
POLYMARKET_ROUND13_PROGRAM_IMPLEMENTATION_SCHEMA_VERSION = (
    "polymarket-round13-program-implementation-v1"
)
POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE = Decimal("1000")
POLYMARKET_ROUND13_MINIMUM_EXPECTED_EDGE_QUOTE = Decimal("0.02")
POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD = 1e-12
POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM = Decimal("0.01")
POLYMARKET_ROUND13_SETTLEMENT_QUANTUM = Decimal("0.000001")
POLYMARKET_ROUND13_FEE_QUANTUM = Decimal("0.00001")
POLYMARKET_ROUND13_DECIMAL_PRECISION = 50
POLYMARKET_ROUND13_V2_CLIENT_COMMIT = "215fc63a8fd6ec3a10c7edb73997c9772d8686d3"
POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT = "ccc0596074f4dfd62c944fbca4de252893b82b4b"
POLYMARKET_ROUND13_RS_V2_CLIENT_COMMIT = "222143d321eba97d5711a848265eb9aab3bc7ff4"
POLYMARKET_ROUND13_OFFICIAL_CLI_COMMIT = "9b18b5faf5493b945c48ca22efaf9645f0c69ab8"
POLYMARKET_ROUND13_AGENT_SKILLS_COMMIT = "91ee44ae113e958affd20cd505c6e9d9d6100e0b"
POLYMARKET_ROUND13_BOOTSTRAP_SAMPLES = 10_000
POLYMARKET_ROUND13_BOOTSTRAP_SEED = 13_013
POLYMARKET_ROUND13_BOOTSTRAP_BLOCK_GROUPS = 12
POLYMARKET_ROUND13_MINIMUM_SYNCHRONIZED_GROUPS = 160
POLYMARKET_ROUND13_MINIMUM_SIMULATED_FILLED_CONDITIONS = 30
POLYMARKET_ROUND13_MINIMUM_SIMULATED_FILLS_PER_ASSET = 5
POLYMARKET_ROUND13_MINIMUM_NON_TIED_CONTROL_CONDITIONS = 30

_ASSETS = ("BTC", "ETH", "SOL")
_OUTCOMES = ("Up", "Down")
_POLICIES = ("calibrated", "raw_market_prior")
_SHA256_LENGTH = 64
_FEATURE_INDEX = {name: index for index, name in enumerate(POLYMARKET_FEATURE_NAMES)}
_V2_MARKET_BUY_TAKER_DECIMALS = {
    Decimal("0.1"): 3,
    Decimal("0.01"): 4,
    Decimal("0.005"): 5,
    Decimal("0.0025"): 6,
    Decimal("0.001"): 5,
    Decimal("0.0001"): 6,
}
_IMPLEMENTATION_FILES = (
    "polymarket_round13.py",
    "polymarket_round13_capture.py",
    "polymarket_round13_evaluation.py",
    "polymarket_round13_publication.py",
)
_UPSTREAM_ORDER_SOURCE_SHA256 = (
    (
        "Polymarket/py-clob-client-v2",
        POLYMARKET_ROUND13_V2_CLIENT_COMMIT,
        "py_clob_client_v2/order_builder/builder.py",
        "cb5d2499da246ec713840ac0c490b5d7cf80e6bef68ff51e6f43fe8e1f231e82",
    ),
    (
        "Polymarket/py-clob-client-v2",
        POLYMARKET_ROUND13_V2_CLIENT_COMMIT,
        "py_clob_client_v2/client.py",
        "66604f43bf37f8482f3f50674b9f3e3834ff13ef83722c00c69520e37052bd3a",
    ),
    (
        "Polymarket/py-clob-client-v2",
        POLYMARKET_ROUND13_V2_CLIENT_COMMIT,
        "py_clob_client_v2/fees.py",
        "f121783c0492cd2ea3c68dd9e765ae37885b53c5662c02668c22f8f6a64df591",
    ),
    (
        "Polymarket/py-clob-client-v2",
        POLYMARKET_ROUND13_V2_CLIENT_COMMIT,
        "py_clob_client_v2/clob_types.py",
        "1eaab86968594bdaa58c20e1bec33c789149441ee1e4e0164922f361f83a8060",
    ),
    (
        "Polymarket/ctf-exchange-v2",
        POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT,
        "src/exchange/mixins/Trading.sol",
        "dd8d18fca897e664583a93944b379435e6f70e84f4190c39d669b2be62596012",
    ),
    (
        "Polymarket/ctf-exchange-v2",
        POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT,
        "src/exchange/libraries/CalculatorHelper.sol",
        "b8dce122a07f5fe7898747ee61f73a82c4b96bf206b238e98da2ceed97737dc1",
    ),
    (
        "Polymarket/ctf-exchange-v2",
        POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT,
        "src/exchange/mixins/Fees.sol",
        "38af8b9878e55bdddc3669e5771ba995321d7546716be894c851b723cad973fe",
    ),
    (
        "Polymarket/rs-clob-client-v2",
        POLYMARKET_ROUND13_RS_V2_CLIENT_COMMIT,
        "src/clob/order_builder.rs",
        "7071149a34578310c1a6f0c52eac121e75f5eb238a0b8633b10dc7ae8e04af7e",
    ),
    (
        "Polymarket/rs-clob-client-v2",
        POLYMARKET_ROUND13_RS_V2_CLIENT_COMMIT,
        "src/clob/types/mod.rs",
        "1311f0b29b4f013eb60582cb0f716a2bcc30b708816c6763624f694526e9e814",
    ),
    (
        "Polymarket/rs-clob-client-v2",
        POLYMARKET_ROUND13_RS_V2_CLIENT_COMMIT,
        "examples/clob/orders/market_buy.rs",
        "d26d567480c8c9ef2ebd431aa77114b52273267e77171cb2f448aa2cdf8eb9ae",
    ),
    (
        "Polymarket/polymarket-cli",
        POLYMARKET_ROUND13_OFFICIAL_CLI_COMMIT,
        "src/commands/clob.rs",
        "47e2feeaa46cc0ae4582ed601b9d421f84046e1ef054197f0747a739c92bbab3",
    ),
    (
        "Polymarket/agent-skills",
        POLYMARKET_ROUND13_AGENT_SKILLS_COMMIT,
        "order-patterns.md",
        "d70c20d09c98edd0f735f8653079672d332d1b0f385094632fcc62691183d88e",
    ),
)
_ROUND13_DECIMAL_CONTEXT = Context(
    prec=POLYMARKET_ROUND13_DECIMAL_PRECISION,
    rounding=ROUND_HALF_EVEN,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _decimal_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return left + right


def _expected_edge(
    quantity: Decimal,
    probability: float | Decimal,
    cost: Decimal,
) -> Decimal:
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return quantity * Decimal(str(probability)) - cost


def _decimal_absolute_difference(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return abs(left - right)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value: {value}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


def _normalized_source_sha256(path: Path) -> str:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot attest Round 13 source: {path.name}") from exc
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def polymarket_round13_program_implementation_sha256() -> str:
    """Hash every module that can create, score, or publish Round 13 evidence."""

    source_root = Path(__file__).resolve().parent
    digests = {
        filename: _normalized_source_sha256(source_root / filename)
        for filename in _IMPLEMENTATION_FILES
    }
    return _sha256(
        {
            "schema_version": (
                POLYMARKET_ROUND13_PROGRAM_IMPLEMENTATION_SCHEMA_VERSION
            ),
            "critical_module_source_sha256": digests,
        }
    )


def polymarket_round13_upstream_order_semantics() -> dict[str, object]:
    """Return the immutable primary-source basis for the modeled V2 FOK BUY."""

    return {
        "audited_at_utc_date": "2026-07-18",
        "official_documentation": [
            "https://docs.polymarket.com/trading/orders/create",
            "https://docs.polymarket.com/trading/orderbook",
            "https://docs.polymarket.com/trading/fees",
        ],
        "market_buy_side": "quote_amount",
        "market_sell_side": "share_quantity",
        "fok_behavior": "fill_entire_quote_amount_immediately_or_cancel",
        "worst_price_limit": True,
        "book_depth_unit": "shares",
        "buy_fee_unit": "quote_collateral",
        "required_live_clob_protocol_version": (
            POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION
        ),
        "minimum_order_size_documented_unit": "unspecified",
        "minimum_order_size_unit_assumption": False,
        "minimum_order_size_guard": (
            "quote_amount_and_signed_share_quantity_each_at_least_"
            "recorded_numeric_minimum"
        ),
        "utility_share_quantity": "signed_minimum_not_modeled_price_improvement",
        "market_buy_quote_quantum": format(
            POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM,
            "f",
        ),
        "settlement_quantum": format(POLYMARKET_ROUND13_SETTLEMENT_QUANTUM, "f"),
        "fee_quantum": format(POLYMARKET_ROUND13_FEE_QUANTUM, "f"),
        "decimal_precision": POLYMARKET_ROUND13_DECIMAL_PRECISION,
        "decimal_rounding": "ROUND_HALF_EVEN",
        "v2_market_buy_taker_decimals_by_tick": {
            format(tick, "f"): decimals
            for tick, decimals in sorted(_V2_MARKET_BUY_TAKER_DECIMALS.items())
        },
        "source_files": [
            {
                "repository": repository,
                "commit": commit,
                "path": path,
                "sha256": sha256,
            }
            for repository, commit, path, sha256 in _UPSTREAM_ORDER_SOURCE_SHA256
        ],
        "live_order_authority": False,
    }


@dataclass(frozen=True)
class PolymarketRound13Scenario:
    name: str
    submission_latency_ms: int
    fee_multiplier: int
    adverse_ticks: int
    depth_numerator: int
    depth_denominator: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def validated(self) -> PolymarketRound13Scenario:
        expected = {item.name: item for item in polymarket_round13_scenarios()}
        if expected.get(self.name) != self:
            raise ValueError("Polymarket Round 13 scenario is invalid")
        return self


def polymarket_round13_scenarios() -> tuple[PolymarketRound13Scenario, ...]:
    return (
        PolymarketRound13Scenario("primary", 500, 1, 0, 2, 2),
        PolymarketRound13Scenario("latency_750ms", 750, 1, 0, 2, 2),
        PolymarketRound13Scenario("latency_1000ms", 1000, 1, 0, 2, 2),
        PolymarketRound13Scenario("double_taker_fee", 500, 2, 0, 2, 2),
        PolymarketRound13Scenario("one_tick_adverse", 500, 1, 1, 2, 2),
        PolymarketRound13Scenario("half_displayed_depth", 500, 1, 0, 1, 2),
        PolymarketRound13Scenario("combined_stress", 1000, 2, 1, 1, 2),
    )


@dataclass(frozen=True)
class PolymarketRound13EvaluationGates:
    minimum_synchronized_event_groups: int
    minimum_resolved_markets_per_asset: int
    minimum_outcome_classes_per_asset: int
    minimum_simulated_filled_conditions: int
    minimum_simulated_fills_per_asset: int
    minimum_non_tied_treatment_control_conditions: int
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_block_groups: int
    drawdown_capital_fraction: str
    drawdown_median_positive_group_multiple: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def validated(self) -> PolymarketRound13EvaluationGates:
        if self != polymarket_round13_evaluation_gates():
            raise ValueError("Polymarket Round 13 evaluation gates are invalid")
        return self

    @property
    def drawdown_capital_fraction_decimal(self) -> Decimal:
        return Decimal(self.drawdown_capital_fraction)

    @property
    def drawdown_median_positive_group_multiple_decimal(self) -> Decimal:
        return Decimal(self.drawdown_median_positive_group_multiple)


def polymarket_round13_evaluation_gates() -> PolymarketRound13EvaluationGates:
    return PolymarketRound13EvaluationGates(
        minimum_synchronized_event_groups=(
            POLYMARKET_ROUND13_MINIMUM_SYNCHRONIZED_GROUPS
        ),
        minimum_resolved_markets_per_asset=(
            POLYMARKET_ROUND13_MINIMUM_SYNCHRONIZED_GROUPS
        ),
        minimum_outcome_classes_per_asset=2,
        minimum_simulated_filled_conditions=(
            POLYMARKET_ROUND13_MINIMUM_SIMULATED_FILLED_CONDITIONS
        ),
        minimum_simulated_fills_per_asset=(
            POLYMARKET_ROUND13_MINIMUM_SIMULATED_FILLS_PER_ASSET
        ),
        minimum_non_tied_treatment_control_conditions=(
            POLYMARKET_ROUND13_MINIMUM_NON_TIED_CONTROL_CONDITIONS
        ),
        bootstrap_samples=POLYMARKET_ROUND13_BOOTSTRAP_SAMPLES,
        bootstrap_seed=POLYMARKET_ROUND13_BOOTSTRAP_SEED,
        bootstrap_block_groups=POLYMARKET_ROUND13_BOOTSTRAP_BLOCK_GROUPS,
        drawdown_capital_fraction="0.02",
        drawdown_median_positive_group_multiple="2",
    )


@dataclass(frozen=True)
class PolymarketRound13Program:
    contract: Mapping[str, object]
    contract_sha256: str
    model: PolymarketRound12ReferenceModel
    policy: PolymarketRound12PrimaryPolicy
    scenarios: tuple[PolymarketRound13Scenario, ...]
    evaluation_gates: PolymarketRound13EvaluationGates
    confirmation_capital_quote: Decimal

    def validated(self) -> PolymarketRound13Program:
        if (
            not _is_sha256(self.contract_sha256)
            or self.contract.get("contract_sha256") != self.contract_sha256
            or self.confirmation_capital_quote
            != POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE
            or self.scenarios != polymarket_round13_scenarios()
            or self.evaluation_gates != polymarket_round13_evaluation_gates()
        ):
            raise ValueError("Polymarket Round 13 program is invalid")
        self.model.validated()
        self.policy.validated()
        self.evaluation_gates.validated()
        return self


def load_round13_confirmation_contract(path: str | Path) -> PolymarketRound13Program:
    """Load a canonical contract and verify every executable implementation hash."""

    contract_path = Path(path).resolve()
    try:
        decoded = json.loads(
            contract_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Round 13 confirmation contract is invalid JSON") from exc
    contract = dict(_mapping(decoded, name="Round 13 confirmation contract"))
    claimed = contract.pop("contract_sha256", None)
    if (
        not _is_sha256(claimed)
        or _sha256(contract) != claimed
        or contract.get("schema_version") != POLYMARKET_ROUND13_CONTRACT_SCHEMA_VERSION
        or contract.get("round") != 13
        or contract.get("status") != "frozen_before_fresh_capture"
    ):
        raise ValueError("Round 13 confirmation contract identity is invalid")

    predecessor = _mapping(contract.get("predecessor_evidence"), name="predecessor")
    artifact_filename = str(predecessor.get("artifact_filename") or "")
    if not artifact_filename or Path(artifact_filename).name != artifact_filename:
        raise ValueError("Round 13 predecessor artifact path is invalid")
    artifact_path = contract_path.parent / artifact_filename
    try:
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("Round 13 predecessor artifact is unavailable") from exc
    if artifact_sha256 != predecessor.get("artifact_file_sha256"):
        raise ValueError("Round 13 predecessor artifact bytes differ")
    model = load_round12_reference_from_round11_artifact(artifact_path)
    policy = polymarket_round12_primary_policy()

    model_contract = _mapping(contract.get("model_contract"), name="model contract")
    policy_contract = _mapping(contract.get("primary_policy"), name="policy contract")
    if (
        model_contract.get("model_sha256") != model.model_sha256
        or model_contract.get("training_or_refit_performed") is not False
        or policy_contract.get("policy_sha256") != policy.policy_sha256
        or policy_contract.get("minimum_direction_probability")
        != policy.minimum_direction_probability
        or policy_contract.get("minimum_expected_edge_quote")
        != policy.minimum_expected_edge_quote
        or policy_contract.get("numeric_decision_guard")
        != POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD
        or policy_contract.get("minimum_remaining_seconds")
        != policy.minimum_remaining_seconds
        or policy_contract.get("forced_activity") is not False
    ):
        raise ValueError("Round 13 model or policy contract differs")

    expected_scenarios = [item.asdict() for item in polymarket_round13_scenarios()]
    execution = _mapping(contract.get("execution_scenarios"), name="scenarios")
    if execution.get("scenarios") != expected_scenarios:
        raise ValueError("Round 13 execution scenarios differ")
    order_semantics = _mapping(
        contract.get("order_semantics"),
        name="order semantics",
    )
    if dict(order_semantics) != polymarket_round13_upstream_order_semantics():
        raise ValueError("Round 13 upstream order semantics differ")
    scope = _mapping(contract.get("scope"), name="scope")
    risk = _mapping(contract.get("risk_contract"), name="risk contract")
    from .polymarket_round13_capture import (
        POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS,
    )

    capture = _mapping(contract.get("capture_contract"), name="capture contract")
    if (
        scope.get("assets") != list(_ASSETS)
        or scope.get("synthetic_data") is not False
        or scope.get("leverage_applicable") is not False
        or scope.get("forced_activity") is not False
        or capture.get("one_shot") is not True
        or capture.get("fresh_prospective_duration_seconds")
        != POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS
        or Decimal(str(risk.get("confirmation_capital_quote")))
        != POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE
        or risk.get("reinvestment") is not False
        or risk.get("leverage") is not False
        or Decimal(str(risk.get("maximum_group_exposure_quote")))
        != POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE
        or risk.get("no_inferred_capital") is not True
    ):
        raise ValueError("Round 13 scope, capture, or risk contract differs")
    gates_contract = _mapping(
        contract.get("executable_evaluation_gates"), name="evaluation gates"
    )
    gates = polymarket_round13_evaluation_gates()
    if dict(gates_contract) != gates.asdict():
        raise ValueError("Round 13 executable evaluation gates differ")

    from .polymarket_action_pipeline import (
        polymarket_action_pipeline_implementation_sha256,
    )

    implementation = _mapping(contract.get("implementation"), name="implementation")
    if (
        implementation.get("reference_implementation_sha256")
        != polymarket_round12_reference_implementation_sha256()
        or implementation.get("action_pipeline_implementation_sha256")
        != polymarket_action_pipeline_implementation_sha256()
        or implementation.get("round13_program_implementation_sha256")
        != polymarket_round13_program_implementation_sha256()
        or implementation.get("source_hash_normalization") != "utf8_lf_normalized"
        or implementation.get("reference_numeric_semantics")
        != "finite IEEE-754 binary64"
        or implementation.get("economic_decimal_semantics")
        != "local decimal precision 50 with ROUND_HALF_EVEN; caller context ignored"
        or implementation.get("financial_gate_numeric_semantics")
        != (
            "local decimal precision 50 with ROUND_HALF_EVEN through gate "
            "decisions; finite IEEE-754 binary64 only for the frozen bootstrap "
            "and report serialization"
        )
        or implementation.get("runtime_paths_in_model_semantics") is not False
        or implementation.get("environment_variables_in_model_semantics") is not False
        or implementation.get("operating_system_branches_in_model_semantics")
        is not False
    ):
        raise ValueError("Round 13 implementation differs from its contract")

    freshness = _mapping(contract.get("freshness"), name="freshness")
    authority = _mapping(contract.get("authority"), name="authority")
    if (
        freshness.get("capture_started") is not False
        or freshness.get("confirmation_consumed") is not False
        or freshness.get("outcome_labels_consulted") is not False
        or freshness.get("thresholds_changed_after_freeze") is not False
        or freshness.get("one_shot") is not True
        or authority.get("paper_trading") is not False
        or authority.get("live_trading") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("roi_claim") is not False
        or authority.get("drawdown_claim") is not False
        or authority.get("ai_edge_claim") is not False
        or authority.get("authenticated_order_lifecycle_proven") is not False
        or authority.get("owned_balance_reconciliation_proven") is not False
        or authority.get("settlement_overhead_measured") is not False
    ):
        raise ValueError("Round 13 freshness or authority state differs")
    frozen = {**contract, "contract_sha256": claimed}
    return PolymarketRound13Program(
        contract=frozen,
        contract_sha256=str(claimed),
        model=model,
        policy=policy,
        scenarios=polymarket_round13_scenarios(),
        evaluation_gates=gates,
        confirmation_capital_quote=POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE,
    ).validated()


@dataclass(frozen=True)
class PolymarketRound13EntryObservation:
    scenario: str
    action_feature_sha256: str
    condition_id: str
    outcome: str
    decision_event_id: str
    decision_segment_id: str
    decision_monotonic_ns: int
    creation_book_event_id: str
    fok_tick_size: str
    fok_limit_price: str
    order_amount_quote: str
    execution_parameter_sha256: str
    execution_target_monotonic_ns: int | None
    entry_book_event_id: str
    entry_book_segment_id: str
    entry_book_monotonic_ns: int | None
    entry_book_tick_size: str | None
    submission_attempted: bool
    observation_state: str
    entry_modeled_quantity: str | None
    entry_fee_quote: str | None
    entry_cost_quote: str | None
    maximum_entry_loss_quote: str
    reason: str
    source_evidence_sha256: str
    evidence_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("evidence_sha256")
        return {
            "schema_version": "polymarket-round13-entry-observation-v1",
            **payload,
        }

    def validated(self) -> PolymarketRound13EntryObservation:
        amount = Decimal(self.order_amount_quote)
        filled = (
            None
            if self.entry_modeled_quantity is None
            else Decimal(self.entry_modeled_quantity)
        )
        fee = None if self.entry_fee_quote is None else Decimal(self.entry_fee_quote)
        cost = None if self.entry_cost_quote is None else Decimal(self.entry_cost_quote)
        maximum = Decimal(self.maximum_entry_loss_quote)
        tick = Decimal(self.fok_tick_size)
        limit = Decimal(self.fok_limit_price)
        entry_tick = (
            None
            if self.entry_book_tick_size is None
            else Decimal(self.entry_book_tick_size)
        )
        if (
            self.scenario not in {item.name for item in polymarket_round13_scenarios()}
            or not _is_sha256(self.action_feature_sha256)
            or not self.condition_id
            or self.outcome not in _OUTCOMES
            or not self.decision_event_id
            or not _is_sha256(self.decision_segment_id)
            or self.decision_monotonic_ns < 0
            or not self.creation_book_event_id
            or not tick.is_finite()
            or tick <= 0
            or tick >= 1
            or not limit.is_finite()
            or limit <= 0
            or limit >= 1
            or limit % tick != 0
            or not amount.is_finite()
            or amount <= 0
            or amount % POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM != 0
            or (
                self.execution_parameter_sha256
                and not _is_sha256(self.execution_parameter_sha256)
            )
            or (
                self.entry_book_segment_id
                and not _is_sha256(self.entry_book_segment_id)
            )
            or (
                entry_tick is not None
                and (not entry_tick.is_finite() or entry_tick <= 0 or entry_tick >= 1)
            )
            or self.observation_state
            not in {
                "not_submitted",
                "simulated_no_fill",
                "simulated_fill",
                "unknown_after_submit",
            }
            or not self.reason
            or not _is_sha256(self.source_evidence_sha256)
            or not maximum.is_finite()
            or maximum <= 0
            or maximum < amount
            or (filled is not None and (not filled.is_finite() or filled <= 0))
            or (fee is not None and (not fee.is_finite() or fee < 0))
            or (cost is not None and (not cost.is_finite() or cost <= 0))
            or (cost is not None and cost > maximum)
            or (self.observation_state == "simulated_fill")
            != (filled is not None and fee is not None and cost is not None)
            or (cost is not None and (fee is None or cost != _decimal_add(amount, fee)))
            or self.evidence_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 13 entry observation is invalid")
        has_entry_book = bool(self.entry_book_event_id)
        if (
            has_entry_book != bool(self.entry_book_segment_id)
            or has_entry_book != (self.entry_book_monotonic_ns is not None)
            or has_entry_book != (entry_tick is not None)
        ):
            raise ValueError("Round 13 entry-book identity is inconsistent")
        if self.submission_attempted != (self.observation_state != "not_submitted"):
            raise ValueError("Round 13 submission state is inconsistent")
        requires_entry_book = self.observation_state in {
            "simulated_fill",
            "simulated_no_fill",
        }
        if requires_entry_book != has_entry_book and self.reason != (
            "post_submit_tick_size_drift"
        ):
            raise ValueError("Round 13 observed-book state is inconsistent")
        if self.observation_state == "not_submitted":
            if (
                self.execution_target_monotonic_ns is not None
                or self.entry_book_event_id
                or self.entry_book_segment_id
                or self.entry_book_monotonic_ns is not None
                or self.entry_book_tick_size is not None
                or self.reason
                not in {"missing_execution_parameters", "unsupported_minimum_order_age"}
            ):
                raise ValueError("Round 13 non-submission evidence is inconsistent")
        else:
            if (
                self.execution_target_monotonic_ns is None
                or self.execution_target_monotonic_ns < self.decision_monotonic_ns
                or not self.execution_parameter_sha256
            ):
                raise ValueError("Round 13 submitted timing evidence is inconsistent")
        if has_entry_book and (
            self.entry_book_segment_id != self.decision_segment_id
            or self.entry_book_monotonic_ns is None
            or self.execution_target_monotonic_ns is None
            or self.entry_book_monotonic_ns < self.execution_target_monotonic_ns
            or self.entry_book_monotonic_ns
            > self.execution_target_monotonic_ns + 500_000_000
        ):
            raise ValueError("Round 13 entry-book timing evidence is inconsistent")
        if requires_entry_book and entry_tick != tick:
            raise ValueError("Round 13 known entry book changed tick size")
        if self.reason == "post_submit_tick_size_drift" and (
            not has_entry_book or entry_tick == tick
        ):
            raise ValueError("Round 13 tick-drift evidence is inconsistent")
        expected_reasons = {
            "simulated_fill": "stressed_displayed_depth_walk_complete",
            "simulated_no_fill": (
                "insufficient_stressed_displayed_depth_within_fok_limit"
            ),
            "unknown_after_submit": {
                "missing_same_segment_entry_observation",
                "post_submit_tick_size_drift",
            },
        }
        expected_reason = expected_reasons.get(self.observation_state)
        if isinstance(expected_reason, str) and self.reason != expected_reason:
            raise ValueError("Round 13 entry observation reason is inconsistent")
        if isinstance(expected_reason, set) and self.reason not in expected_reason:
            raise ValueError("Round 13 unknown observation reason is inconsistent")
        return self


@dataclass(frozen=True)
class PolymarketRound13CalibrationSnapshot:
    condition_id: str
    asset: str
    event_start_ms: int
    action_feature_up_sha256: str
    action_feature_down_sha256: str
    decision_event_id: str
    decision_monotonic_ns: int
    remaining_seconds: float
    market_prior_up: float
    calibrated_probability_up: float
    snapshot_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("snapshot_sha256")
        return {
            "schema_version": "polymarket-round13-calibration-snapshot-v1",
            **payload,
        }

    def validated(self) -> PolymarketRound13CalibrationSnapshot:
        if (
            not self.condition_id
            or self.asset not in _ASSETS
            or self.event_start_ms < 0
            or not _is_sha256(self.action_feature_up_sha256)
            or not _is_sha256(self.action_feature_down_sha256)
            or not self.decision_event_id
            or self.decision_monotonic_ns < 0
            or not math.isfinite(self.remaining_seconds)
            or self.remaining_seconds < 120.0
            or not 0 < self.market_prior_up < 1
            or not 0 < self.calibrated_probability_up < 1
            or self.snapshot_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 13 calibration snapshot is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRound13Attempt:
    scenario: str
    policy: str
    condition_id: str
    asset: str
    event_start_ms: int
    attempt_index: int
    action_feature_sha256: str
    decision_event_id: str
    decision_monotonic_ns: int
    remaining_seconds: float
    outcome: str
    probability: float
    expected_edge_quote: str
    order_amount_quote: str
    minimum_signed_quantity: str
    creation_modeled_quantity: str
    creation_fee_quote: str
    conservative_entry_cost_quote: str
    observation: PolymarketRound13EntryObservation
    attempt_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("attempt_sha256")
        payload["observation"] = self.observation.identity_payload() | {
            "evidence_sha256": self.observation.evidence_sha256
        }
        return {"schema_version": "polymarket-round13-attempt-v1", **payload}

    def validated(self) -> PolymarketRound13Attempt:
        expected_edge = Decimal(self.expected_edge_quote)
        amount = Decimal(self.order_amount_quote)
        minimum_quantity = Decimal(self.minimum_signed_quantity)
        creation_quantity = Decimal(self.creation_modeled_quantity)
        creation_fee = Decimal(self.creation_fee_quote)
        cost = Decimal(self.conservative_entry_cost_quote)
        maximum_loss = Decimal(self.observation.maximum_entry_loss_quote)
        observed_quantity = (
            None
            if self.observation.entry_modeled_quantity is None
            else Decimal(self.observation.entry_modeled_quantity)
        )
        guarded_edge = _decimal_add(
            POLYMARKET_ROUND13_MINIMUM_EXPECTED_EDGE_QUOTE,
            Decimal(str(POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD)),
        )
        self.observation.validated()
        recomputed_edge = _expected_edge(
            minimum_quantity,
            self.probability,
            cost,
        )
        if (
            self.scenario != self.observation.scenario
            or self.policy not in _POLICIES
            or self.condition_id != self.observation.condition_id
            or self.asset not in _ASSETS
            or self.attempt_index < 0
            or self.action_feature_sha256 != self.observation.action_feature_sha256
            or self.decision_event_id != self.observation.decision_event_id
            or self.decision_monotonic_ns != self.observation.decision_monotonic_ns
            or self.outcome != self.observation.outcome
            or self.order_amount_quote != self.observation.order_amount_quote
            or not math.isfinite(self.remaining_seconds)
            or self.remaining_seconds < 120.0
            or not 0 < self.probability < 1
            or not expected_edge.is_finite()
            or expected_edge <= guarded_edge
            or expected_edge != recomputed_edge
            or not amount.is_finite()
            or amount <= 0
            or amount % POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM != 0
            or not minimum_quantity.is_finite()
            or minimum_quantity <= 0
            or not creation_quantity.is_finite()
            or creation_quantity < minimum_quantity
            or (observed_quantity is not None and observed_quantity < minimum_quantity)
            or not creation_fee.is_finite()
            or creation_fee < 0
            or not cost.is_finite()
            or cost <= 0
            or cost != _decimal_add(amount, creation_fee)
            or maximum_loss
            >= _expected_edge(minimum_quantity, self.probability, guarded_edge)
            or self.attempt_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 13 attempt is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRound13LabelFreeDataset:
    contract_sha256: str
    source_run_id: str
    source_feature_dataset_sha256: str
    source_action_dataset_sha256: str
    model_sha256: str
    policy_sha256: str
    event_start_ms: int
    condition_ids: tuple[str, ...]
    calibration_snapshots: tuple[PolymarketRound13CalibrationSnapshot, ...]
    attempts: tuple[PolymarketRound13Attempt, ...]
    abstention_counts: Mapping[str, int]
    dataset_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND13_SCENARIO_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "source_run_id": self.source_run_id,
            "source_feature_dataset_sha256": self.source_feature_dataset_sha256,
            "source_action_dataset_sha256": self.source_action_dataset_sha256,
            "model_sha256": self.model_sha256,
            "policy_sha256": self.policy_sha256,
            "event_start_ms": self.event_start_ms,
            "condition_ids": list(self.condition_ids),
            "scenario_config": [
                item.asdict() for item in polymarket_round13_scenarios()
            ],
            "calibration_snapshot_sha256": [
                item.snapshot_sha256 for item in self.calibration_snapshots
            ],
            "attempt_sha256": [item.attempt_sha256 for item in self.attempts],
            "abstention_counts": dict(sorted(self.abstention_counts.items())),
        }

    def validated(self) -> PolymarketRound13LabelFreeDataset:
        for item in self.calibration_snapshots:
            item.validated()
        for item in self.attempts:
            item.validated()
        snapshot_by_condition = {
            item.condition_id: item for item in self.calibration_snapshots
        }
        condition_order = {
            condition_id: index for index, condition_id in enumerate(self.condition_ids)
        }
        scenario_order = {
            scenario.name: index
            for index, scenario in enumerate(polymarket_round13_scenarios())
        }
        policy_order = {policy: index for index, policy in enumerate(_POLICIES)}
        expected_attempt_order = tuple(
            sorted(
                self.attempts,
                key=lambda item: (
                    condition_order.get(item.condition_id, len(condition_order)),
                    item.decision_monotonic_ns,
                    scenario_order.get(item.scenario, len(scenario_order)),
                    policy_order.get(item.policy, len(policy_order)),
                ),
            )
        )
        allowed_abstention_reasons = {
            "lifecycle_blocked",
            "retry_cooldown",
            "inadmissible_pair",
            "no_positive_conservative_edge",
            "contradictory_equal_edge",
            "no_safe_fok_limit",
            "insufficient_creation_depth_within_fok_limit",
        }
        abstention_keys_valid = all(
            len(parts := str(key).split("|")) == 3
            and parts[0] in scenario_order
            and parts[1] in policy_order
            and parts[2] in allowed_abstention_reasons
            for key in self.abstention_counts
        )
        attempt_groups: dict[tuple[str, str, str], list[PolymarketRound13Attempt]] = {}
        for attempt in self.attempts:
            attempt_groups.setdefault(
                (attempt.condition_id, attempt.scenario, attempt.policy), []
            ).append(attempt)
        lifecycle_valid = True
        for values in attempt_groups.values():
            ordered = sorted(values, key=lambda item: item.attempt_index)
            indexes = [item.attempt_index for item in ordered]
            timestamps = [item.decision_monotonic_ns for item in ordered]
            terminal_indexes = [
                index
                for index, item in enumerate(ordered)
                if item.observation.observation_state
                in {"simulated_fill", "unknown_after_submit"}
            ]
            if (
                indexes != list(range(len(ordered)))
                or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
                or (terminal_indexes and terminal_indexes != [len(ordered) - 1])
            ):
                lifecycle_valid = False
                break
        maximum_exposure: dict[tuple[str, str], Decimal] = {}
        for attempt in self.attempts:
            state = attempt.observation.observation_state
            if state == "simulated_fill":
                exposure = Decimal(str(attempt.observation.entry_cost_quote))
            elif state == "unknown_after_submit":
                exposure = Decimal(attempt.observation.maximum_entry_loss_quote)
            else:
                exposure = Decimal("0")
            exposure_key = (attempt.scenario, attempt.policy)
            maximum_exposure[exposure_key] = _decimal_add(
                maximum_exposure.get(exposure_key, Decimal("0")),
                exposure,
            )
        if (
            not _is_sha256(self.contract_sha256)
            or not self.source_run_id
            or not _is_sha256(self.source_feature_dataset_sha256)
            or not _is_sha256(self.source_action_dataset_sha256)
            or not _is_sha256(self.model_sha256)
            or not _is_sha256(self.policy_sha256)
            or self.event_start_ms < 0
            or len(self.condition_ids) != 3
            or len(set(self.condition_ids)) != 3
            or len(self.calibration_snapshots) != 3
            or {item.condition_id for item in self.calibration_snapshots}
            != set(self.condition_ids)
            or tuple(snapshot_by_condition[value].asset for value in self.condition_ids)
            != _ASSETS
            or any(
                item.event_start_ms != self.event_start_ms
                for item in self.calibration_snapshots
            )
            or any(
                item.condition_id not in snapshot_by_condition for item in self.attempts
            )
            or any(
                item.asset != snapshot_by_condition[item.condition_id].asset
                for item in self.attempts
            )
            or any(item.event_start_ms != self.event_start_ms for item in self.attempts)
            or len({item.snapshot_sha256 for item in self.calibration_snapshots})
            != len(self.calibration_snapshots)
            or len({item.attempt_sha256 for item in self.attempts})
            != len(self.attempts)
            or expected_attempt_order != self.attempts
            or not lifecycle_valid
            or not abstention_keys_valid
            or any(int(value) < 0 for value in self.abstention_counts.values())
            or any(
                exposure > POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE
                for exposure in maximum_exposure.values()
            )
            or self.dataset_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 13 label-free dataset is invalid")
        return self


def _floor_to_quantum(value: Decimal, quantum: Decimal) -> Decimal:
    if not value.is_finite() or not quantum.is_finite() or quantum <= 0:
        raise ValueError("Round 13 decimal quantization input is invalid")
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return (value / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum


def _ceil_to_quantum(value: Decimal, quantum: Decimal) -> Decimal:
    if not value.is_finite() or not quantum.is_finite() or quantum <= 0:
        raise ValueError("Round 13 decimal quantization input is invalid")
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return (value / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum


def _market_buy_taker_quantum(tick_size: Decimal) -> Decimal:
    try:
        decimals = _V2_MARKET_BUY_TAKER_DECIMALS[tick_size]
    except KeyError as exc:
        raise ValueError(
            "Round 13 tick is unsupported by the pinned V2 client"
        ) from exc
    return Decimal("1").scaleb(-decimals)


@dataclass(frozen=True)
class _Round13FokBuyBound:
    limit_price: Decimal
    amount_quote: Decimal
    minimum_signed_quantity: Decimal
    maximum_entry_loss_quote: Decimal


_Round13TransformedFill = tuple[str, str, str, str]
_Round13WalkResult = tuple[
    Decimal,
    Decimal,
    Decimal,
    tuple[_Round13TransformedFill, ...],
]


def _conservative_fok_buy_bound(
    market: PolymarketFiveMinuteMarket,
    scenario: PolymarketRound13Scenario,
    probability: float,
    minimum_expected_edge_quote: float | Decimal,
    *,
    tick_size: Decimal | None = None,
) -> _Round13FokBuyBound | None:
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return _conservative_fok_buy_bound_fixed(
            market,
            scenario,
            probability,
            minimum_expected_edge_quote,
            tick_size=tick_size,
        )


def _conservative_fok_buy_bound_fixed(
    market: PolymarketFiveMinuteMarket,
    scenario: PolymarketRound13Scenario,
    probability: float,
    minimum_expected_edge_quote: float | Decimal,
    *,
    tick_size: Decimal | None = None,
) -> _Round13FokBuyBound | None:
    """Return one quote-denominated V2 FOK BUY contract with a loss bound."""

    tick = market.tick_size if tick_size is None else tick_size
    if not tick.is_finite() or tick <= 0 or tick >= 1:
        raise ValueError("Round 13 FOK tick size is invalid")
    taker_quantum = _market_buy_taker_quantum(tick)
    rate = market.fee_schedule.rate * scenario.fee_multiplier
    exponent = market.fee_schedule.exponent
    if (
        not market.minimum_order_size.is_finite()
        or market.minimum_order_size <= 0
        or not rate.is_finite()
        or rate < 0
        or rate > 1
        or isinstance(exponent, bool)
        or not isinstance(exponent, int)
        or exponent <= 0
    ):
        raise ValueError("Round 13 market or stressed fee parameters are invalid")
    probability_decimal = Decimal(str(probability))
    edge_floor = Decimal(str(minimum_expected_edge_quote))
    if (
        not probability_decimal.is_finite()
        or probability_decimal <= 0
        or probability_decimal >= 1
        or not edge_floor.is_finite()
        or edge_floor < 0
    ):
        return None
    venue_maximum_steps = int(
        ((Decimal("1") - tick) / tick).to_integral_value(rounding=ROUND_FLOOR)
    )
    probability_maximum_steps = (
        int((probability_decimal / tick).to_integral_value(rounding=ROUND_CEILING)) - 1
    )
    maximum_steps = min(venue_maximum_steps, probability_maximum_steps)
    if maximum_steps < 1:
        return None
    # A positive edge is impossible at a limit >= probability. Search the
    # remaining ticks from highest to lowest and stop at the first safe limit;
    # this remains exact across cent-rounding discontinuities.
    for step in range(maximum_steps, 0, -1):
        limit = tick * step
        share_floor_amount = _ceil_to_quantum(
            market.minimum_order_size * limit,
            POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM,
        )
        quote_floor_amount = _ceil_to_quantum(
            market.minimum_order_size,
            POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM,
        )
        # Official public material names ``min_order_size`` but does not state
        # its unit for a quote-denominated BUY. Satisfy both interpretations.
        amount = max(share_floor_amount, quote_floor_amount)
        minimum_quantity = _floor_to_quantum(amount / limit, taker_quantum)
        maximum_fee = Decimal("0")
        if market.fee_schedule.enabled and rate > 0:
            # For exponent >= 1, fee/quote is globally bounded by ``rate``.
            # One fee quantum per possible displayed price level covers this
            # simulator's adverse per-level rounding.
            maximum_fee = amount * rate + Decimal(step) * POLYMARKET_ROUND13_FEE_QUANTUM
        maximum_loss = amount + maximum_fee
        worst_edge = minimum_quantity * probability_decimal - maximum_loss
        if (
            amount >= market.minimum_order_size
            and minimum_quantity >= market.minimum_order_size
            and maximum_loss <= POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE
            and worst_edge > edge_floor
        ):
            return _Round13FokBuyBound(
                limit_price=limit,
                amount_quote=amount,
                minimum_signed_quantity=minimum_quantity,
                maximum_entry_loss_quote=maximum_loss,
            )
    return None


def _walk_transformed_asks(
    book: PolymarketRecordedBook,
    decision: PolymarketRepricingDecision,
    scenario: PolymarketRound13Scenario,
    *,
    limit_price: Decimal,
    order_amount_quote: Decimal,
) -> _Round13WalkResult | None:
    with localcontext(_ROUND13_DECIMAL_CONTEXT):
        return _walk_transformed_asks_fixed(
            book,
            decision,
            scenario,
            limit_price=limit_price,
            order_amount_quote=order_amount_quote,
        )


def _walk_transformed_asks_fixed(
    book: PolymarketRecordedBook,
    decision: PolymarketRepricingDecision,
    scenario: PolymarketRound13Scenario,
    *,
    limit_price: Decimal,
    order_amount_quote: Decimal,
) -> _Round13WalkResult | None:
    market = book.market
    if (
        not order_amount_quote.is_finite()
        or order_amount_quote <= 0
        or order_amount_quote % POLYMARKET_ROUND13_MARKET_BUY_QUOTE_QUANTUM != 0
    ):
        raise ValueError("Round 13 FOK BUY amount is not an exact quote-cent value")
    limit = min(
        Decimal("1") - decision.creation_book.tick_size,
        limit_price,
    )
    fee = PolymarketFeeModel(
        enabled=market.fee_schedule.enabled,
        rate=market.fee_schedule.rate * scenario.fee_multiplier,
        exponent=market.fee_schedule.exponent,
        taker_only=True,
    )
    remaining_quote = order_amount_quote
    gross_quantity = Decimal("0")
    total_fee = Decimal("0")
    fills: list[_Round13TransformedFill] = []
    depth_scale = Decimal(scenario.depth_numerator) / Decimal(
        scenario.depth_denominator
    )
    for level in book.snapshot.validated().asks:
        price = level.price + decision.creation_book.tick_size * scenario.adverse_ticks
        available = level.quantity * depth_scale
        if price >= 1 or price > limit:
            break
        if available <= 0:
            continue
        available_quote = price * available
        filled_quote = min(remaining_quote, available_quote)
        filled_quantity = filled_quote / price
        fee_quote = fee(price, filled_quantity, "taker")
        gross_quantity += filled_quantity
        total_fee += fee_quote
        fills.append(
            (
                format(price, "f"),
                format(filled_quantity, "f"),
                format(filled_quote, "f"),
                format(fee_quote, "f"),
            )
        )
        remaining_quote -= filled_quote
        if remaining_quote <= 0:
            break
    if remaining_quote > 0:
        return None
    settled_quantity = _floor_to_quantum(
        gross_quantity,
        POLYMARKET_ROUND13_SETTLEMENT_QUANTUM,
    )
    if settled_quantity <= 0:
        raise RuntimeError("Round 13 quote FOK produced no settlement-sized shares")
    return (
        order_amount_quote + total_fee,
        settled_quantity,
        total_fee,
        tuple(fills),
    )


def _entry_observation(
    context: PolymarketRepricingExecutionContext,
    feature: PolymarketActionFeature,
    decision: PolymarketRepricingDecision,
    scenario: PolymarketRound13Scenario,
    fok_limit_price: Decimal,
    order_amount_quote: Decimal,
    maximum_entry_loss_quote: Decimal,
) -> PolymarketRound13EntryObservation:
    market = decision.creation_book.market
    parameter = context.parameter_index.latest_at_or_before(
        market.condition_id,
        decision.received_monotonic_ns + scenario.submission_latency_ms * 1_000_000,
    )
    parameter_sha = "" if parameter is None else parameter.snapshot_sha256
    target_ns: int | None = None
    entry: PolymarketRecordedBook | None = None
    cost: Decimal | None = None
    filled_quantity: Decimal | None = None
    fee_quote: Decimal | None = None
    fills: tuple[tuple[str, str, str, str], ...] = ()
    if parameter is None:
        state = "not_submitted"
        reason = "missing_execution_parameters"
    elif parameter.minimum_order_age_seconds != 0:
        state = "not_submitted"
        reason = "unsupported_minimum_order_age"
    else:
        target_ns = (
            decision.received_monotonic_ns
            + (scenario.submission_latency_ms + parameter.taker_order_delay_ms)
            * 1_000_000
        )
        entry = context.book_index.first_at_or_after(
            token_id=decision.token_id,
            segment_id=decision.segment_id,
            condition_id=decision.condition_id,
            target_monotonic_ns=target_ns,
            maximum_observation_delay_ms=500,
        )
        if entry is None:
            state = "unknown_after_submit"
            reason = "missing_same_segment_entry_observation"
        elif entry.tick_size != decision.creation_book.tick_size:
            state = "unknown_after_submit"
            reason = "post_submit_tick_size_drift"
        else:
            walked = _walk_transformed_asks(
                entry,
                decision,
                scenario,
                limit_price=fok_limit_price,
                order_amount_quote=order_amount_quote,
            )
            if walked is None:
                state = "simulated_no_fill"
                reason = "insufficient_stressed_displayed_depth_within_fok_limit"
            else:
                state = "simulated_fill"
                reason = "stressed_displayed_depth_walk_complete"
                cost, filled_quantity, fee_quote, fills = walked
    payload = {
        "scenario": scenario.name,
        "action_feature_sha256": feature.action_feature_sha256,
        "condition_id": feature.condition_id,
        "outcome": feature.outcome,
        "decision_event_id": decision.event_id,
        "decision_segment_id": decision.segment_id,
        "decision_monotonic_ns": decision.received_monotonic_ns,
        "creation_book_event_id": decision.creation_book.event_id,
        "fok_tick_size": format(decision.creation_book.tick_size, "f"),
        "fok_limit_price": format(fok_limit_price, "f"),
        "order_amount_quote": format(order_amount_quote, "f"),
        "execution_parameter_sha256": parameter_sha,
        "execution_target_monotonic_ns": target_ns,
        "entry_book_event_id": "" if entry is None else entry.event_id,
        "entry_book_segment_id": "" if entry is None else entry.segment_id,
        "entry_book_monotonic_ns": (
            None if entry is None else entry.received_monotonic_ns
        ),
        "entry_book_tick_size": (
            None if entry is None else format(entry.tick_size, "f")
        ),
        "submission_attempted": state != "not_submitted",
        "observation_state": state,
        "entry_modeled_quantity": (
            None if filled_quantity is None else format(filled_quantity, "f")
        ),
        "entry_fee_quote": None if fee_quote is None else format(fee_quote, "f"),
        "entry_cost_quote": None if cost is None else format(cost, "f"),
        "maximum_entry_loss_quote": format(maximum_entry_loss_quote, "f"),
        "reason": reason,
        "source_evidence": {
            "creation_source_sha256": decision.creation_book.snapshot.source_payload_sha256,
            "entry_source_sha256": (
                "" if entry is None else entry.snapshot.source_payload_sha256
            ),
            "transformed_fills": [list(item) for item in fills],
        },
    }
    identity = {
        key: value for key, value in payload.items() if key != "source_evidence"
    }
    identity["source_evidence_sha256"] = _sha256(payload["source_evidence"])
    provisional = PolymarketRound13EntryObservation(
        **identity,
        evidence_sha256="",
    )
    return replace(
        provisional,
        evidence_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _market_prior_up(feature_values: Sequence[float]) -> float:
    up = float(feature_values[_FEATURE_INDEX["up_midpoint"]])
    down = float(feature_values[_FEATURE_INDEX["down_midpoint"]])
    total = up + down
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Round 13 market prior is invalid")
    return min(1.0 - 1e-9, max(1e-9, up / total))


@dataclass(frozen=True)
class _Round13FokCandidate:
    outcome: str
    probability: float
    expected_edge_quote: Decimal
    bound: _Round13FokBuyBound
    creation_modeled_quantity: Decimal
    creation_fee_quote: Decimal
    creation_cost_quote: Decimal


def _select_fok_candidate(
    probability_up: float,
    market: PolymarketFiveMinuteMarket,
    decisions: Mapping[str, PolymarketRepricingDecision],
    scenario: PolymarketRound13Scenario,
    policy: PolymarketRound12PrimaryPolicy,
) -> tuple[_Round13FokCandidate | None, str]:
    probability_floor = (
        policy.minimum_direction_probability + POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD
    )
    edge_floor = _decimal_add(
        Decimal(str(policy.minimum_expected_edge_quote)),
        Decimal(str(POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD)),
    )
    eligible = tuple(
        (outcome, probability)
        for outcome, probability in (
            ("Up", probability_up),
            ("Down", 1.0 - probability_up),
        )
        if probability >= probability_floor
    )
    if not eligible:
        return None, "no_positive_conservative_edge"

    bounded: list[tuple[str, float, _Round13FokBuyBound]] = []
    for outcome, probability in eligible:
        decision = decisions[outcome]
        bound = _conservative_fok_buy_bound(
            market,
            scenario,
            probability,
            edge_floor,
            tick_size=decision.creation_book.tick_size,
        )
        if bound is not None:
            bounded.append((outcome, probability, bound))
    if not bounded:
        return None, "no_safe_fok_limit"

    candidates: dict[str, _Round13FokCandidate] = {}
    executions: dict[str, tuple[Decimal, Decimal] | None] = {
        outcome: None for outcome in _OUTCOMES
    }
    for outcome, probability, bound in bounded:
        decision = decisions[outcome]
        walked = _walk_transformed_asks(
            decision.creation_book,
            decision,
            scenario,
            limit_price=bound.limit_price,
            order_amount_quote=bound.amount_quote,
        )
        if walked is None:
            continue
        cost, filled_quantity, fee_quote, _fills = walked
        if filled_quantity < bound.minimum_signed_quantity:
            raise RuntimeError("Round 13 FOK fill violates its signed minimum quantity")
        edge = _expected_edge(bound.minimum_signed_quantity, probability, cost)
        executions[outcome] = bound.minimum_signed_quantity, cost
        candidates[outcome] = _Round13FokCandidate(
            outcome=outcome,
            probability=probability,
            expected_edge_quote=edge,
            bound=bound,
            creation_modeled_quantity=filled_quantity,
            creation_fee_quote=fee_quote,
            creation_cost_quote=cost,
        )
    if not candidates:
        return None, "insufficient_creation_depth_within_fok_limit"
    outcome, _probability, edge, reason = _select_outcome(
        probability_up,
        executions,
        policy,
    )
    if outcome is None or edge is None:
        return None, reason
    selected = candidates[outcome]
    if selected.expected_edge_quote != edge:
        raise RuntimeError("Round 13 candidate selection edge is inconsistent")
    return selected, "eligible_primary_candidate"


def _select_outcome(
    probability_up: float,
    executions: Mapping[str, tuple[Decimal, Decimal] | None],
    policy: PolymarketRound12PrimaryPolicy,
) -> tuple[str | None, float | None, Decimal | None, str]:
    """Select from already modeled quote-FOK executions (testable pure rule)."""

    candidates: list[tuple[Decimal, str, float]] = []
    probability_floor = (
        policy.minimum_direction_probability + POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD
    )
    edge_floor = _decimal_add(
        Decimal(str(policy.minimum_expected_edge_quote)),
        Decimal(str(POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD)),
    )
    for outcome, probability in (
        ("Up", probability_up),
        ("Down", 1.0 - probability_up),
    ):
        execution = executions[outcome]
        if execution is None:
            continue
        quantity, cost = execution
        edge = _expected_edge(quantity, probability, cost)
        if probability >= probability_floor and edge > edge_floor:
            candidates.append((edge, outcome, probability))
    if not candidates:
        return None, None, None, "no_positive_conservative_edge"
    candidates.sort(reverse=True)
    if len(candidates) > 1 and _decimal_absolute_difference(
        candidates[0][0],
        candidates[1][0],
    ) <= Decimal(str(POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD)):
        return None, None, None, "contradictory_equal_edge"
    edge, outcome, probability = candidates[0]
    return outcome, probability, edge, "eligible_primary_candidate"


def build_round13_label_free_dataset(
    features: PolymarketFeatureDataset,
    actions: PolymarketActionValueDataset,
    context: PolymarketRepricingExecutionContext,
    program: PolymarketRound13Program,
) -> PolymarketRound13LabelFreeDataset:
    """Apply every frozen policy/scenario before official outcomes are consulted."""

    source = features
    source.config.validated()
    for row in source.rows:
        row.validated()
    if any(
        row.official_up is not None or bool(row.resolution_event_id)
        for row in source.rows
    ):
        raise ValueError("Round 13 label-free source contains outcome labels")
    if (
        source.dataset_id != source.dataset_sha256
        or not _is_sha256(source.dataset_sha256)
        or not source.rows
        or any(row.run_id != source.run_id for row in source.rows)
    ):
        raise ValueError("Round 13 source feature dataset identity differs")
    action_source = actions.validated()
    frozen = program.validated()
    if (
        source.run_id != context.run_id
        or action_source.source_run_id != source.run_id
        or action_source.source_feature_dataset_sha256 != source.dataset_sha256
    ):
        raise ValueError("Round 13 label-free source identity differs")
    action_lookup = {
        (item.source_feature_id, item.outcome): item for item in action_source.features
    }
    market_by_condition = context.market_by_condition
    rows_by_condition: dict[str, list[object]] = {}
    for row in source.rows:
        rows_by_condition.setdefault(row.condition_id, []).append(row)
    if len(rows_by_condition) != 3:
        raise ValueError("Round 13 batch must contain one synchronized asset group")
    ordered_conditions = tuple(
        market.condition_id
        for market in sorted(
            (market_by_condition[value] for value in rows_by_condition),
            key=lambda item: _ASSETS.index(item.asset),
        )
    )
    event_starts = {
        market_by_condition[value].event_start_ms for value in ordered_conditions
    }
    if len(event_starts) != 1:
        raise ValueError("Round 13 batch conditions are not synchronized")
    event_start = next(iter(event_starts))
    snapshots: list[PolymarketRound13CalibrationSnapshot] = []
    attempts: list[PolymarketRound13Attempt] = []
    abstentions: Counter[str] = Counter()

    for condition_id in ordered_conditions:
        market = market_by_condition[condition_id]
        rows = sorted(
            rows_by_condition[condition_id],
            key=lambda item: (item.decision_received_monotonic_ns, item.feature_id),
        )
        snapshot_candidate: PolymarketRound13CalibrationSnapshot | None = None
        blocked = {
            (scenario.name, policy_name): False
            for scenario in frozen.scenarios
            for policy_name in _POLICIES
        }
        last_attempt_ns: dict[tuple[str, str], int] = {}
        attempt_indexes: Counter[tuple[str, str]] = Counter()
        for raw_row in rows:
            row = raw_row.validated()
            remaining = float(row.feature_values[_FEATURE_INDEX["remaining_seconds"]])
            action_features = {
                outcome: action_lookup.get((row.feature_id, outcome))
                for outcome in _OUTCOMES
            }
            if any(value is None for value in action_features.values()):
                raise ValueError("Round 13 action feature linkage is incomplete")
            decisions = {
                outcome: context.decision_at(
                    market,
                    event_id=row.decision_event_id,
                    received_wall_ms=row.decision_received_wall_ms,
                    received_monotonic_ns=row.decision_received_monotonic_ns,
                    outcome=outcome,
                    maximum_creation_book_age_ms=500,
                )
                for outcome in _OUTCOMES
            }
            pair_admissible = (
                all(value is not None for value in decisions.values())
                and len({value.segment_id for value in decisions.values() if value})
                == 1
            )
            prior_up = _market_prior_up(row.feature_values)
            calibrated_up, _calibrated_down = frozen.model.predict_pair(prior_up)
            if pair_admissible and remaining >= frozen.policy.minimum_remaining_seconds:
                provisional_snapshot = PolymarketRound13CalibrationSnapshot(
                    condition_id=condition_id,
                    asset=market.asset,
                    event_start_ms=event_start,
                    action_feature_up_sha256=action_features[
                        "Up"
                    ].action_feature_sha256,  # type: ignore[union-attr]
                    action_feature_down_sha256=action_features[
                        "Down"
                    ].action_feature_sha256,  # type: ignore[union-attr]
                    decision_event_id=row.decision_event_id,
                    decision_monotonic_ns=row.decision_received_monotonic_ns,
                    remaining_seconds=remaining,
                    market_prior_up=prior_up,
                    calibrated_probability_up=calibrated_up,
                    snapshot_sha256="",
                )
                snapshot_candidate = replace(
                    provisional_snapshot,
                    snapshot_sha256=_sha256(provisional_snapshot.identity_payload()),
                ).validated()
            if remaining < frozen.policy.minimum_remaining_seconds:
                continue
            if not pair_admissible:
                for scenario in frozen.scenarios:
                    for policy_name in _POLICIES:
                        abstentions[
                            f"{scenario.name}|{policy_name}|inadmissible_pair"
                        ] += 1
                continue
            typed_decisions = {
                outcome: value
                for outcome, value in decisions.items()
                if value is not None
            }
            observation_cache: dict[
                tuple[str, str, str], PolymarketRound13EntryObservation
            ] = {}
            for scenario in frozen.scenarios:
                for policy_name, probability_up in (
                    ("calibrated", calibrated_up),
                    ("raw_market_prior", prior_up),
                ):
                    key = (scenario.name, policy_name)
                    if blocked[key]:
                        abstentions[
                            f"{scenario.name}|{policy_name}|lifecycle_blocked"
                        ] += 1
                        continue
                    previous = last_attempt_ns.get(key)
                    if previous is not None and (
                        row.decision_received_monotonic_ns - previous
                        < frozen.policy.retry_interval_ms * 1_000_000
                    ):
                        abstentions[
                            f"{scenario.name}|{policy_name}|retry_cooldown"
                        ] += 1
                        continue
                    candidate, reason = _select_fok_candidate(
                        probability_up,
                        market,
                        typed_decisions,
                        scenario,
                        frozen.policy,
                    )
                    if candidate is None:
                        abstentions[f"{scenario.name}|{policy_name}|{reason}"] += 1
                        continue
                    outcome = candidate.outcome
                    cache_key = (scenario.name, policy_name, outcome)
                    if cache_key not in observation_cache:
                        observation_cache[cache_key] = _entry_observation(
                            context,
                            action_features[outcome],  # type: ignore[arg-type]
                            typed_decisions[outcome],
                            scenario,
                            candidate.bound.limit_price,
                            candidate.bound.amount_quote,
                            candidate.bound.maximum_entry_loss_quote,
                        )
                    observation = observation_cache[cache_key]
                    attempt_index = attempt_indexes[key]
                    provisional_attempt = PolymarketRound13Attempt(
                        scenario=scenario.name,
                        policy=policy_name,
                        condition_id=condition_id,
                        asset=market.asset,
                        event_start_ms=event_start,
                        attempt_index=attempt_index,
                        action_feature_sha256=action_features[
                            outcome
                        ].action_feature_sha256,  # type: ignore[union-attr]
                        decision_event_id=row.decision_event_id,
                        decision_monotonic_ns=row.decision_received_monotonic_ns,
                        remaining_seconds=remaining,
                        outcome=outcome,
                        probability=candidate.probability,
                        expected_edge_quote=format(
                            candidate.expected_edge_quote,
                            "f",
                        ),
                        order_amount_quote=format(
                            candidate.bound.amount_quote,
                            "f",
                        ),
                        minimum_signed_quantity=format(
                            candidate.bound.minimum_signed_quantity,
                            "f",
                        ),
                        creation_modeled_quantity=format(
                            candidate.creation_modeled_quantity,
                            "f",
                        ),
                        creation_fee_quote=format(
                            candidate.creation_fee_quote,
                            "f",
                        ),
                        conservative_entry_cost_quote=format(
                            candidate.creation_cost_quote,
                            "f",
                        ),
                        observation=observation,
                        attempt_sha256="",
                    )
                    attempts.append(
                        replace(
                            provisional_attempt,
                            attempt_sha256=_sha256(
                                provisional_attempt.identity_payload()
                            ),
                        ).validated()
                    )
                    attempt_indexes[key] += 1
                    last_attempt_ns[key] = row.decision_received_monotonic_ns
                    if observation.observation_state in {
                        "simulated_fill",
                        "unknown_after_submit",
                    }:
                        blocked[key] = True
        if snapshot_candidate is None:
            raise ValueError(
                f"Round 13 condition has no admissible 120-second snapshot: {condition_id}"
            )
        snapshots.append(snapshot_candidate)
    provisional = PolymarketRound13LabelFreeDataset(
        contract_sha256=frozen.contract_sha256,
        source_run_id=source.run_id,
        source_feature_dataset_sha256=source.dataset_sha256,
        source_action_dataset_sha256=action_source.dataset_sha256,
        model_sha256=frozen.model.model_sha256,
        policy_sha256=frozen.policy.policy_sha256,
        event_start_ms=event_start,
        condition_ids=ordered_conditions,
        calibration_snapshots=tuple(snapshots),
        attempts=tuple(attempts),
        abstention_counts=dict(sorted(abstentions.items())),
        dataset_sha256="",
    )
    return replace(
        provisional,
        dataset_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _snapshot_payload(value: PolymarketRound13CalibrationSnapshot) -> dict[str, object]:
    return {**value.identity_payload(), "snapshot_sha256": value.snapshot_sha256}


def _observation_payload(value: PolymarketRound13EntryObservation) -> dict[str, object]:
    return {**value.identity_payload(), "evidence_sha256": value.evidence_sha256}


def _attempt_payload(value: PolymarketRound13Attempt) -> dict[str, object]:
    payload = value.identity_payload()
    payload["observation"] = _observation_payload(value.observation)
    return {**payload, "attempt_sha256": value.attempt_sha256}


def _snapshot_from_payload(value: object) -> PolymarketRound13CalibrationSnapshot:
    payload = dict(_mapping(value, name="Round 13 calibration snapshot"))
    if payload.pop("schema_version", None) != (
        "polymarket-round13-calibration-snapshot-v1"
    ):
        raise ValueError("Round 13 calibration snapshot schema differs")
    try:
        return PolymarketRound13CalibrationSnapshot(**payload).validated()
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 13 calibration snapshot payload is invalid") from exc


def _observation_from_payload(value: object) -> PolymarketRound13EntryObservation:
    payload = dict(_mapping(value, name="Round 13 entry observation"))
    if payload.pop("schema_version", None) != "polymarket-round13-entry-observation-v1":
        raise ValueError("Round 13 entry observation schema differs")
    try:
        return PolymarketRound13EntryObservation(**payload).validated()
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 13 entry observation payload is invalid") from exc


def _attempt_from_payload(value: object) -> PolymarketRound13Attempt:
    payload = dict(_mapping(value, name="Round 13 attempt"))
    if payload.pop("schema_version", None) != "polymarket-round13-attempt-v1":
        raise ValueError("Round 13 attempt schema differs")
    payload["observation"] = _observation_from_payload(payload.get("observation"))
    try:
        return PolymarketRound13Attempt(**payload).validated()
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 13 attempt payload is invalid") from exc


def _strict_json_payload(raw: object, *, name: str) -> object:
    try:
        return json.loads(
            str(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid JSON") from exc


def _ensure_round13_tables(store: PolymarketEvidenceStore) -> None:
    store.connect().execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_round13_scenario_dataset (
            dataset_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            source_run_id VARCHAR NOT NULL,
            source_feature_dataset_sha256 VARCHAR NOT NULL,
            source_action_dataset_sha256 VARCHAR NOT NULL UNIQUE,
            model_sha256 VARCHAR NOT NULL,
            policy_sha256 VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            condition_ids_json VARCHAR NOT NULL,
            calibration_snapshot_count UINTEGER NOT NULL,
            attempt_count UBIGINT NOT NULL,
            abstention_counts_json VARCHAR NOT NULL,
            manifest_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS polymarket_round13_calibration_snapshot (
            dataset_sha256 VARCHAR NOT NULL,
            snapshot_index UINTEGER NOT NULL,
            condition_id VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            snapshot_json VARCHAR NOT NULL,
            snapshot_sha256 VARCHAR NOT NULL,
            PRIMARY KEY(dataset_sha256, snapshot_index),
            UNIQUE(dataset_sha256, condition_id)
        );

        CREATE TABLE IF NOT EXISTS polymarket_round13_attempt (
            dataset_sha256 VARCHAR NOT NULL,
            attempt_row_index UBIGINT NOT NULL,
            scenario VARCHAR NOT NULL,
            policy VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            observation_state VARCHAR NOT NULL,
            attempt_json VARCHAR NOT NULL,
            attempt_sha256 VARCHAR NOT NULL,
            PRIMARY KEY(dataset_sha256, attempt_row_index),
            UNIQUE(dataset_sha256, attempt_sha256)
        );
        """
    )


def materialize_round13_label_free_dataset(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRound13LabelFreeDataset,
) -> str:
    """Persist compact scenario decisions without reading or copying outcome labels."""

    from .duckdb_batch import insert_rows_columnar

    selected = dataset.validated()
    connection = store.connect()
    source = connection.execute(
        """
        SELECT source_run_id, source_feature_dataset_sha256, action_count
        FROM polymarket_action_value_dataset WHERE dataset_sha256 = ?
        """,
        [selected.source_action_dataset_sha256],
    ).fetchone()
    if source is None or tuple(source[:2]) != (
        selected.source_run_id,
        selected.source_feature_dataset_sha256,
    ):
        raise ValueError("Round 13 source action manifest is inconsistent")
    _ensure_round13_tables(store)
    manifest_payload = {
        **selected.identity_payload(),
        "dataset_sha256": selected.dataset_sha256,
    }
    manifest = (
        selected.dataset_sha256,
        POLYMARKET_ROUND13_SCENARIO_SCHEMA_VERSION,
        selected.contract_sha256,
        selected.source_run_id,
        selected.source_feature_dataset_sha256,
        selected.source_action_dataset_sha256,
        selected.model_sha256,
        selected.policy_sha256,
        selected.event_start_ms,
        _canonical_json(list(selected.condition_ids)),
        len(selected.calibration_snapshots),
        len(selected.attempts),
        _canonical_json(dict(sorted(selected.abstention_counts.items()))),
        _canonical_json(manifest_payload),
        selected.dataset_sha256,
    )
    snapshot_rows = [
        (
            selected.dataset_sha256,
            index,
            item.condition_id,
            item.asset,
            item.event_start_ms,
            _canonical_json(_snapshot_payload(item)),
            item.snapshot_sha256,
        )
        for index, item in enumerate(selected.calibration_snapshots)
    ]
    attempt_rows = [
        (
            selected.dataset_sha256,
            index,
            item.scenario,
            item.policy,
            item.condition_id,
            item.asset,
            item.event_start_ms,
            item.observation.observation_state,
            _canonical_json(_attempt_payload(item)),
            item.attempt_sha256,
        )
        for index, item in enumerate(selected.attempts)
    ]
    existing = connection.execute(
        "SELECT * FROM polymarket_round13_scenario_dataset WHERE dataset_sha256 = ?",
        [selected.dataset_sha256],
    ).fetchone()
    if existing is not None:
        if tuple(existing) != manifest:
            raise ValueError("stored Round 13 scenario manifest is inconsistent")
        stored_snapshots = connection.execute(
            """
            SELECT dataset_sha256, snapshot_index, condition_id, asset,
                   event_start_ms, snapshot_json, snapshot_sha256
            FROM polymarket_round13_calibration_snapshot
            WHERE dataset_sha256 = ? ORDER BY snapshot_index
            """,
            [selected.dataset_sha256],
        ).fetchall()
        stored_attempts = connection.execute(
            """
            SELECT dataset_sha256, attempt_row_index, scenario, policy,
                   condition_id, asset, event_start_ms, observation_state,
                   attempt_json, attempt_sha256
            FROM polymarket_round13_attempt
            WHERE dataset_sha256 = ? ORDER BY attempt_row_index
            """,
            [selected.dataset_sha256],
        ).fetchall()
        if [tuple(row) for row in stored_snapshots] != snapshot_rows or [
            tuple(row) for row in stored_attempts
        ] != attempt_rows:
            raise ValueError("stored Round 13 scenario rows are inconsistent")
        return "existing"
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "INSERT INTO polymarket_round13_scenario_dataset VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            manifest,
        )
        insert_rows_columnar(
            connection,
            sql=(
                "INSERT INTO polymarket_round13_calibration_snapshot "
                "SELECT unnest(?), unnest(?), unnest(?), unnest(?), "
                "unnest(?), unnest(?), unnest(?)"
            ),
            rows=snapshot_rows,
            width=7,
        )
        if attempt_rows:
            insert_rows_columnar(
                connection,
                sql=(
                    "INSERT INTO polymarket_round13_attempt SELECT "
                    + ", ".join("unnest(?)" for _ in range(10))
                ),
                rows=attempt_rows,
                width=10,
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return "created"


def load_round13_label_free_dataset(
    store: PolymarketEvidenceStore,
    *,
    source_action_dataset_sha256: str,
) -> PolymarketRound13LabelFreeDataset:
    """Load and re-hash one scenario batch from its source action identity."""

    selected = str(source_action_dataset_sha256).strip().lower()
    if not _is_sha256(selected):
        raise ValueError("Round 13 source action digest is invalid")
    _ensure_round13_tables(store)
    manifest = (
        store.connect()
        .execute(
            """
        SELECT dataset_sha256, contract_sha256, source_run_id,
               source_feature_dataset_sha256, model_sha256, policy_sha256,
               event_start_ms, condition_ids_json, calibration_snapshot_count,
               attempt_count, abstention_counts_json, manifest_json,
               manifest_sha256
        FROM polymarket_round13_scenario_dataset
        WHERE source_action_dataset_sha256 = ?
        """,
            [selected],
        )
        .fetchone()
    )
    if manifest is None:
        raise ValueError("Round 13 scenario manifest is missing")
    dataset_sha256 = str(manifest[0])
    snapshot_rows = (
        store.connect()
        .execute(
            """
        SELECT snapshot_json, snapshot_sha256
        FROM polymarket_round13_calibration_snapshot
        WHERE dataset_sha256 = ? ORDER BY snapshot_index
        """,
            [dataset_sha256],
        )
        .fetchall()
    )
    attempt_rows = (
        store.connect()
        .execute(
            """
        SELECT attempt_json, attempt_sha256
        FROM polymarket_round13_attempt
        WHERE dataset_sha256 = ? ORDER BY attempt_row_index
        """,
            [dataset_sha256],
        )
        .fetchall()
    )
    snapshots = tuple(
        _snapshot_from_payload(_strict_json_payload(row[0], name="snapshot JSON"))
        for row in snapshot_rows
    )
    attempts = tuple(
        _attempt_from_payload(_strict_json_payload(row[0], name="attempt JSON"))
        for row in attempt_rows
    )
    if tuple(item.snapshot_sha256 for item in snapshots) != tuple(
        str(row[1]) for row in snapshot_rows
    ) or tuple(item.attempt_sha256 for item in attempts) != tuple(
        str(row[1]) for row in attempt_rows
    ):
        raise ValueError("Round 13 stored row hashes differ")
    conditions = _strict_json_payload(manifest[7], name="condition IDs")
    abstentions = _strict_json_payload(manifest[10], name="abstention counts")
    if not isinstance(conditions, list) or not isinstance(abstentions, Mapping):
        raise ValueError("Round 13 scenario manifest payload is invalid")
    dataset = PolymarketRound13LabelFreeDataset(
        contract_sha256=str(manifest[1]),
        source_run_id=str(manifest[2]),
        source_feature_dataset_sha256=str(manifest[3]),
        source_action_dataset_sha256=selected,
        model_sha256=str(manifest[4]),
        policy_sha256=str(manifest[5]),
        event_start_ms=int(manifest[6]),
        condition_ids=tuple(str(value) for value in conditions),
        calibration_snapshots=snapshots,
        attempts=attempts,
        abstention_counts={str(key): int(value) for key, value in abstentions.items()},
        dataset_sha256=dataset_sha256,
    ).validated()
    expected_manifest = {
        **dataset.identity_payload(),
        "dataset_sha256": dataset.dataset_sha256,
    }
    stored_manifest = _strict_json_payload(manifest[11], name="scenario manifest")
    if (
        int(manifest[8]) != len(snapshots)
        or int(manifest[9]) != len(attempts)
        or str(manifest[12]) != dataset.dataset_sha256
        or stored_manifest != expected_manifest
        or _canonical_json(stored_manifest) != str(manifest[11])
    ):
        raise ValueError("Round 13 scenario manifest does not revalidate")
    return dataset


__all__ = [
    "POLYMARKET_ROUND13_BOOTSTRAP_BLOCK_GROUPS",
    "POLYMARKET_ROUND13_BOOTSTRAP_SAMPLES",
    "POLYMARKET_ROUND13_BOOTSTRAP_SEED",
    "POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE",
    "POLYMARKET_ROUND13_MINIMUM_EXPECTED_EDGE_QUOTE",
    "POLYMARKET_ROUND13_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND13_SCENARIO_SCHEMA_VERSION",
    "PolymarketRound13Attempt",
    "PolymarketRound13CalibrationSnapshot",
    "PolymarketRound13EntryObservation",
    "PolymarketRound13LabelFreeDataset",
    "PolymarketRound13Program",
    "PolymarketRound13Scenario",
    "build_round13_label_free_dataset",
    "load_round13_confirmation_contract",
    "load_round13_label_free_dataset",
    "materialize_round13_label_free_dataset",
    "polymarket_round13_program_implementation_sha256",
    "polymarket_round13_scenarios",
]
