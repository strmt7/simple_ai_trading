"""Atomic one-use evaluation for the frozen Polymarket Round 13 program."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
import requests
import struct
import time

from .polymarket_action_pipeline import (
    polymarket_action_pipeline_implementation_sha256,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_resolution import (
    PolymarketResolutionFinalizer,
    load_official_resolutions,
)
from .polymarket_round13 import (
    PolymarketRound13Attempt,
    PolymarketRound13EvaluationGates,
    PolymarketRound13LabelFreeDataset,
    PolymarketRound13Program,
    load_round13_label_free_dataset,
)


POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION = "polymarket-round13-sealed-evaluation-v1"
POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION = "polymarket-round13-one-use-claim-v1"

_ASSETS = ("BTC", "ETH", "SOL")
_POLICIES = ("calibrated", "raw_market_prior")
_FINANCIAL_DECIMAL_PRECISION = 50

ProgressCallback = Callable[[str, Mapping[str, object]], None]


class PolymarketRound13ResolutionPending(RuntimeError):
    """The immutable claim remains open while official outcomes are unavailable."""


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
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
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(raw: object, *, name: str) -> object:
    try:
        return json.loads(
            str(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid JSON") from exc


class _SplitMix64:
    """Small specified PRNG; bootstrap identity does not depend on NumPy."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def index(self, size: int) -> int:
        if size <= 0:
            raise ValueError("Round 13 PRNG index size is invalid")
        return self.next_u64() % size


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("Round 13 quantile input is invalid")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _decimal_median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Round 13 decimal median input is empty")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


def _finite_float(value: Decimal) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Round 13 decimal cannot be serialized as binary64")
    return converted


def _bootstrap_group_mean(
    values: Sequence[float],
    *,
    series_name: str,
    gates: PolymarketRound13EvaluationGates,
) -> dict[str, object]:
    data = tuple(float(value) for value in values)
    if len(data) < 10 or any(not math.isfinite(value) for value in data):
        raise ValueError("Round 13 bootstrap input is invalid")
    seed_material = hashlib.sha256(series_name.encode("ascii")).digest()
    frozen_gates = gates.validated()
    seed = frozen_gates.bootstrap_seed ^ int.from_bytes(seed_material[:8], "little")
    generator = _SplitMix64(seed)
    samples: list[float] = []
    block = frozen_gates.bootstrap_block_groups
    for _sample_index in range(frozen_gates.bootstrap_samples):
        selected: list[float] = []
        while len(selected) < len(data):
            start = generator.index(len(data))
            selected.extend(
                data[(start + offset) % len(data)] for offset in range(block)
            )
        samples.append(math.fsum(selected[: len(data)]) / len(data))
    packed = b"".join(struct.pack("<d", value) for value in samples)
    return {
        "samples": len(samples),
        "block_length_groups": block,
        "lower_95_mean_group_utility_quote": _linear_quantile(samples, 0.025),
        "median_mean_group_utility_quote": _linear_quantile(samples, 0.5),
        "upper_95_mean_group_utility_quote": _linear_quantile(samples, 0.975),
        "bootstrap_samples_sha256": hashlib.sha256(packed).hexdigest(),
        "prng": "splitmix64",
        "seed": seed,
    }


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return math.log(clipped) - math.log1p(-clipped)


def _calibration_intercept_slope(
    labels: Sequence[int], probabilities: Sequence[float]
) -> dict[str, object]:
    y = tuple(int(value) for value in labels)
    x = tuple(_logit(value) for value in probabilities)
    if len(y) != len(x) or len(y) < 10 or set(y) != {0, 1}:
        return {"available": False, "intercept": None, "slope": None}
    intercept = math.log(sum(y) / (len(y) - sum(y)))
    slope = 1.0
    converged = False
    for iteration in range(100):
        p = tuple(
            1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, intercept + slope * value))))
            for value in x
        )
        g0 = math.fsum(prediction - target for prediction, target in zip(p, y))
        g1 = math.fsum(
            (prediction - target) * value for prediction, target, value in zip(p, y, x)
        )
        w = tuple(max(1e-12, value * (1.0 - value)) for value in p)
        h00 = math.fsum(w) + 1e-10
        h01 = math.fsum(weight * value for weight, value in zip(w, x))
        h11 = math.fsum(weight * value * value for weight, value in zip(w, x)) + 1e-10
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-18 or not math.isfinite(determinant):
            break
        step0 = (h11 * g0 - h01 * g1) / determinant
        step1 = (-h01 * g0 + h00 * g1) / determinant
        scale = max(1.0, abs(step0) / 2.0, abs(step1) / 2.0)
        intercept -= step0 / scale
        slope -= step1 / scale
        intercept = max(-20.0, min(20.0, intercept))
        slope = max(-20.0, min(20.0, slope))
        if max(abs(step0 / scale), abs(step1 / scale)) < 1e-10:
            converged = True
            break
    return {
        "available": converged,
        "intercept": intercept if converged else None,
        "slope": slope if converged else None,
        "iterations": iteration + 1,
    }


def _proper_scores(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, object]:
    y = tuple(int(value) for value in labels)
    p = tuple(min(1.0 - 1e-12, max(1e-12, float(value))) for value in probabilities)
    if len(y) != len(p) or not y:
        raise ValueError("Round 13 proper-score input is invalid")
    log_loss = math.fsum(
        -(target * math.log(prediction) + (1 - target) * math.log1p(-prediction))
        for target, prediction in zip(y, p)
    ) / len(y)
    brier = math.fsum(
        (prediction - target) ** 2 for target, prediction in zip(y, p)
    ) / len(y)
    bins: list[dict[str, object]] = []
    for index in range(10):
        selected = [
            position
            for position, prediction in enumerate(p)
            if min(9, int(prediction * 10)) == index
        ]
        bins.append(
            {
                "bin": index,
                "lower_probability": index / 10,
                "upper_probability": (index + 1) / 10,
                "count": len(selected),
                "mean_probability": (
                    None
                    if not selected
                    else math.fsum(p[position] for position in selected) / len(selected)
                ),
                "observed_frequency": (
                    None
                    if not selected
                    else math.fsum(y[position] for position in selected) / len(selected)
                ),
            }
        )
    return {
        "count": len(y),
        "log_loss": log_loss,
        "brier_score": brier,
        "calibration": _calibration_intercept_slope(y, p),
        "reliability_bins": bins,
    }


def _ensure_evaluation_tables(store: PolymarketEvidenceStore) -> None:
    store.connect().execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_round13_evaluation_claim (
            contract_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            claim_sha256 VARCHAR NOT NULL UNIQUE,
            run_id VARCHAR NOT NULL,
            pipeline_report_sha256 VARCHAR NOT NULL,
            scenario_dataset_sha256_json VARCHAR NOT NULL,
            opened_at_ms BIGINT NOT NULL,
            status VARCHAR NOT NULL,
            report_sha256 VARCHAR NOT NULL,
            error VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS polymarket_round13_evaluation_report (
            report_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL UNIQUE,
            run_id VARCHAR NOT NULL,
            pipeline_report_sha256 VARCHAR NOT NULL,
            created_at_ms BIGINT NOT NULL,
            report_json VARCHAR NOT NULL
        );
        """
    )


def _load_pipeline_identity(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    pipeline_report_sha256: str,
    program: PolymarketRound13Program,
) -> tuple[str, ...]:
    if not _is_sha256(pipeline_report_sha256):
        raise ValueError("Round 13 pipeline report digest is invalid")
    row = (
        store.connect()
        .execute(
            """
        SELECT run_id, eligibility_sha256, action_dataset_sha256_json,
               implementation_sha256, report_json
        FROM polymarket_action_value_pipeline WHERE report_sha256 = ?
        """,
            [pipeline_report_sha256],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 13 action pipeline report is missing")
    action_ids = _strict_json(row[2], name="Round 13 action dataset IDs")
    report = _strict_json(row[4], name="Round 13 pipeline report")
    if (
        str(row[0]) != run_id
        or str(row[1]) != program.contract_sha256
        or str(row[3]) != polymarket_action_pipeline_implementation_sha256()
        or not isinstance(action_ids, list)
        or not isinstance(report, Mapping)
        or report.get("report_sha256") != pipeline_report_sha256
        or _canonical_json(report) != str(row[4])
    ):
        raise ValueError("Round 13 action pipeline identity differs")
    unhashed_report = dict(report)
    claimed_report_sha256 = unhashed_report.pop("report_sha256", None)
    if (
        claimed_report_sha256 != pipeline_report_sha256
        or _sha256(unhashed_report) != pipeline_report_sha256
        or report.get("continuity_admission_mode") != "action_local"
        or not isinstance(report.get("config"), Mapping)
        or report["config"].get("market_groups_per_batch") != 1  # type: ignore[union-attr]
        or report.get("eligibility_sha256") != program.contract_sha256
    ):
        raise ValueError("Round 13 action pipeline report does not revalidate")
    values = tuple(str(value) for value in action_ids)
    report_batches = report.get("batches")
    if (
        not isinstance(report_batches, list)
        or [
            str(item.get("action_dataset_sha256") or "")
            for item in report_batches
            if isinstance(item, Mapping)
        ]
        != list(values)
        or any(
            not isinstance(item, Mapping)
            or not _is_sha256(item.get("round13_scenario_dataset_sha256"))
            for item in report_batches
        )
    ):
        raise ValueError("Round 13 action pipeline batch evidence differs")
    if (
        len(values) < program.evaluation_gates.minimum_synchronized_event_groups
        or len(set(values)) != len(values)
        or not all(_is_sha256(value) for value in values)
    ):
        raise ValueError("Round 13 action pipeline has insufficient unique groups")
    return values


def _load_label_free_batches(
    store: PolymarketEvidenceStore,
    action_dataset_ids: Sequence[str],
    program: PolymarketRound13Program,
    run_id: str,
    progress: ProgressCallback,
) -> tuple[PolymarketRound13LabelFreeDataset, ...]:
    output: list[PolymarketRound13LabelFreeDataset] = []
    for index, action_id in enumerate(action_dataset_ids, start=1):
        dataset = load_round13_label_free_dataset(
            store,
            source_action_dataset_sha256=action_id,
        )
        if (
            dataset.contract_sha256 != program.contract_sha256
            or dataset.source_run_id != run_id
            or dataset.model_sha256 != program.model.model_sha256
            or dataset.policy_sha256 != program.policy.policy_sha256
            or dataset.source_action_dataset_sha256 != action_id
        ):
            raise ValueError("Round 13 label-free batch program differs")
        output.append(dataset)
        if index % 25 == 0 or index == len(action_dataset_ids):
            progress(
                "label-free-batches",
                {"loaded": index, "total": len(action_dataset_ids)},
            )
    ordered = tuple(sorted(output, key=lambda item: item.event_start_ms))
    condition_ids = [condition for item in ordered for condition in item.condition_ids]
    if (
        len({item.event_start_ms for item in ordered}) != len(ordered)
        or len({item.dataset_sha256 for item in ordered}) != len(ordered)
        or len(set(condition_ids)) != len(condition_ids)
    ):
        raise ValueError("Round 13 label-free event groups are duplicated")
    return ordered


def _open_claim(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    pipeline_report_sha256: str,
    scenario_dataset_ids: Sequence[str],
    program: PolymarketRound13Program,
) -> tuple[int, Mapping[str, object] | None]:
    _ensure_evaluation_tables(store)
    connection = store.connect()
    existing = connection.execute(
        """
        SELECT schema_version, claim_sha256, run_id,
               pipeline_report_sha256, scenario_dataset_sha256_json,
               opened_at_ms, status, report_sha256, error
        FROM polymarket_round13_evaluation_claim
        WHERE contract_sha256 = ?
        """,
        [program.contract_sha256],
    ).fetchone()
    if existing is not None:
        scenarios = _strict_json(existing[4], name="Round 13 stored claim scenarios")
        identity = {
            "schema_version": POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION,
            "contract_sha256": program.contract_sha256,
            "run_id": run_id,
            "pipeline_report_sha256": pipeline_report_sha256,
            "scenario_dataset_sha256": list(scenario_dataset_ids),
            "opened_at_ms": int(existing[5]),
            "state": "opened_before_resolution_query",
            "preexisting_resolution_count": 0,
        }
        status = str(existing[6])
        report_sha256 = str(existing[7] or "")
        error = str(existing[8] or "")
        identity_valid = _sha256(identity) == str(existing[1])
        if (
            not identity_valid
            and status == "failed"
            and error == "preexisting_resolution_evidence_before_one_use_claim"
        ):
            preexisting_count = int(
                connection.execute(
                    "SELECT count(*) FROM polymarket_resolution_evidence "
                    "WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
            )
            failed_identity = {
                **identity,
                "state": "failed_preexisting_resolution_evidence",
                "preexisting_resolution_count": preexisting_count,
            }
            identity_valid = _sha256(failed_identity) == str(existing[1])
        if (
            str(existing[0]) != POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION
            or str(existing[2]) != run_id
            or str(existing[3]) != pipeline_report_sha256
            or scenarios != list(scenario_dataset_ids)
            or _canonical_json(scenarios) != str(existing[4])
            or not identity_valid
        ):
            raise ValueError("Round 13 stored claim identity differs")
        if status == "opened" and not report_sha256 and not error:
            return int(existing[5]), None
        if status == "complete" and _is_sha256(report_sha256) and not error:
            report_row = connection.execute(
                "SELECT report_json FROM polymarket_round13_evaluation_report "
                "WHERE report_sha256 = ?",
                [report_sha256],
            ).fetchone()
            if report_row is None:
                raise ValueError("Round 13 completed claim has no report")
            report = _strict_json(report_row[0], name="Round 13 stored report")
            if not isinstance(report, Mapping):
                raise ValueError("Round 13 stored report is not an object")
            unhashed = dict(report)
            claimed = unhashed.pop("report_sha256", None)
            if (
                claimed != report_sha256
                or _sha256(unhashed) != claimed
                or report.get("contract_sha256") != program.contract_sha256
                or report.get("run_id") != run_id
                or report.get("pipeline_report_sha256") != pipeline_report_sha256
                or report.get("scenario_dataset_sha256") != list(scenario_dataset_ids)
            ):
                raise ValueError("Round 13 completed claim identity differs")
            return 0, report
        raise ValueError(
            f"Round 13 confirmation was already consumed: status={status} error={error}"
        )
    opened_at_ms = time.time_ns() // 1_000_000
    connection.execute("BEGIN TRANSACTION")
    try:
        preexisting_resolution_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_resolution_evidence WHERE run_id = ?",
                [run_id],
            ).fetchone()[0]
        )
        state = (
            "failed_preexisting_resolution_evidence"
            if preexisting_resolution_count
            else "opened_before_resolution_query"
        )
        identity = {
            "schema_version": POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION,
            "contract_sha256": program.contract_sha256,
            "run_id": run_id,
            "pipeline_report_sha256": pipeline_report_sha256,
            "scenario_dataset_sha256": list(scenario_dataset_ids),
            "opened_at_ms": opened_at_ms,
            "state": state,
            "preexisting_resolution_count": preexisting_resolution_count,
        }
        connection.execute(
            "INSERT INTO polymarket_round13_evaluation_claim VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, '', ?)",
            [
                program.contract_sha256,
                POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION,
                _sha256(identity),
                run_id,
                pipeline_report_sha256,
                _canonical_json(list(scenario_dataset_ids)),
                opened_at_ms,
                "failed" if preexisting_resolution_count else "opened",
                (
                    "preexisting_resolution_evidence_before_one_use_claim"
                    if preexisting_resolution_count
                    else ""
                ),
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    if preexisting_resolution_count:
        raise ValueError(
            "Round 13 was consumed by resolution evidence created before its claim"
        )
    return opened_at_ms, None


def _attempt_utility(
    attempt: PolymarketRound13Attempt,
    winning_outcome: str,
) -> Decimal:
    state = attempt.observation.observation_state
    if state == "simulated_fill":
        cost = Decimal(str(attempt.observation.entry_cost_quote))
        quantity = Decimal(str(attempt.minimum_signed_quantity))
        return quantity - cost if attempt.outcome == winning_outcome else -cost
    if state == "unknown_after_submit":
        return -Decimal(attempt.observation.maximum_entry_loss_quote)
    return Decimal("0")


def _policy_metrics(
    batches: Sequence[PolymarketRound13LabelFreeDataset],
    winners: Mapping[str, str],
    *,
    scenario: str,
    policy: str,
    program: PolymarketRound13Program,
) -> dict[str, object]:
    with localcontext(
        Context(prec=_FINANCIAL_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)
    ):
        return _policy_metrics_fixed(
            batches,
            winners,
            scenario=scenario,
            policy=policy,
            program=program,
        )


def _policy_metrics_fixed(
    batches: Sequence[PolymarketRound13LabelFreeDataset],
    winners: Mapping[str, str],
    *,
    scenario: str,
    policy: str,
    program: PolymarketRound13Program,
) -> dict[str, object]:
    frozen = program.validated()
    gates = frozen.evaluation_gates
    zero = Decimal("0")
    allocated_capital = frozen.confirmation_capital_quote
    attempts = tuple(
        attempt
        for batch in batches
        for attempt in batch.attempts
        if attempt.scenario == scenario and attempt.policy == policy
    )
    abstention_prefix = f"{scenario}|{policy}|"
    abstentions_by_reason: Counter[str] = Counter()
    for batch in batches:
        for key, count in batch.abstention_counts.items():
            if key.startswith(abstention_prefix):
                abstentions_by_reason[key.removeprefix(abstention_prefix)] += int(count)
    condition_asset = {
        snapshot.condition_id: snapshot.asset
        for batch in batches
        for snapshot in batch.calibration_snapshots
    }
    condition_group = {
        snapshot.condition_id: snapshot.event_start_ms
        for batch in batches
        for snapshot in batch.calibration_snapshots
    }
    utility_by_condition = {condition: zero for condition in condition_asset}
    simulated_fill_conditions: set[str] = set()
    unknown_conditions: set[str] = set()
    simulated_no_fill_count = 0
    not_submitted_count = 0
    wins = 0
    losses = 0
    capital_deployed = zero
    capital_time_quote_seconds = zero
    turnover = zero
    capital_by_group: dict[int, Decimal] = defaultdict(lambda: Decimal("0"))
    for attempt in attempts:
        state = attempt.observation.observation_state
        utility = _attempt_utility(attempt, winners[attempt.condition_id])
        utility_by_condition[attempt.condition_id] += utility
        if state == "simulated_fill":
            simulated_fill_conditions.add(attempt.condition_id)
            cost = Decimal(str(attempt.observation.entry_cost_quote))
            capital_deployed += cost
            capital_by_group[attempt.event_start_ms] += cost
            turnover += cost
            capital_time_quote_seconds += cost * Decimal(str(attempt.remaining_seconds))
            if attempt.outcome == winners[attempt.condition_id]:
                wins += 1
            else:
                losses += 1
        elif state == "unknown_after_submit":
            unknown_conditions.add(attempt.condition_id)
            maximum = Decimal(attempt.observation.maximum_entry_loss_quote)
            capital_deployed += maximum
            capital_by_group[attempt.event_start_ms] += maximum
            turnover += maximum
            capital_time_quote_seconds += maximum * Decimal(
                str(attempt.remaining_seconds)
            )
        elif state == "simulated_no_fill":
            simulated_no_fill_count += 1
        else:
            not_submitted_count += 1
    group_utility: dict[int, Decimal] = defaultdict(lambda: Decimal("0"))
    asset_utility: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    filled_by_asset: Counter[str] = Counter()
    for condition, utility in utility_by_condition.items():
        group_utility[condition_group[condition]] += utility
        asset_utility[condition_asset[condition]] += utility
        if condition in simulated_fill_conditions:
            filled_by_asset[condition_asset[condition]] += 1
    ordered_groups = tuple(sorted(group_utility))
    group_values = tuple(group_utility[value] for value in ordered_groups)
    equity = zero
    peak = zero
    maximum_drawdown = zero
    equity_rows: list[dict[str, object]] = []
    for group, utility in zip(ordered_groups, group_values):
        equity += utility
        peak = max(peak, equity)
        drawdown = peak - equity
        maximum_drawdown = max(maximum_drawdown, drawdown)
        equity_rows.append(
            {
                "event_start_ms": group,
                "group_utility_quote": _finite_float(utility),
                "cumulative_utility_quote": _finite_float(equity),
                "drawdown_quote": _finite_float(drawdown),
            }
        )
    positive_groups = sorted(value for value in group_values if value > 0)
    median_positive = zero if not positive_groups else _decimal_median(positive_groups)
    drawdown_limit = min(
        gates.drawdown_capital_fraction_decimal * allocated_capital,
        gates.drawdown_median_positive_group_multiple_decimal * median_positive,
    )
    bootstrap = _bootstrap_group_mean(
        tuple(_finite_float(value) for value in group_values),
        series_name=f"{scenario}|{policy}",
        gates=gates,
    )
    condition_values = tuple(utility_by_condition.values())
    filled_condition_values = tuple(
        utility_by_condition[condition]
        for condition in sorted(simulated_fill_conditions)
    )
    total = sum(condition_values, zero)
    maximum_group_exposure = max(capital_by_group.values(), default=zero)
    condition_rows = [
        {
            "condition_id": condition,
            "asset": condition_asset[condition],
            "event_start_ms": condition_group[condition],
            "utility_quote": _finite_float(utility_by_condition[condition]),
        }
        for condition in sorted(
            utility_by_condition,
            key=lambda value: (
                condition_group[value],
                _ASSETS.index(condition_asset[value]),
            ),
        )
    ]
    reasons: list[str] = []
    if len(simulated_fill_conditions) < gates.minimum_simulated_filled_conditions:
        reasons.append("insufficient_simulated_filled_conditions")
    for asset in _ASSETS:
        if filled_by_asset[asset] < gates.minimum_simulated_fills_per_asset:
            reasons.append(f"insufficient_simulated_fills_{asset.lower()}")
    if unknown_conditions:
        reasons.append("selected_unknown_after_submit")
    if total <= 0:
        reasons.append("nonpositive_total_utility")
    if total / Decimal(len(condition_values)) <= 0:
        reasons.append("nonpositive_mean_condition_utility")
    median_simulated_filled_condition_utility = (
        zero
        if not filled_condition_values
        else _decimal_median(filled_condition_values)
    )
    if median_simulated_filled_condition_utility <= 0:
        reasons.append("nonpositive_median_simulated_filled_condition_utility")
    for asset in _ASSETS:
        if asset_utility[asset] <= 0:
            reasons.append(f"nonpositive_{asset.lower()}_utility")
    if float(bootstrap["lower_95_mean_group_utility_quote"]) <= 0:
        reasons.append("nonpositive_bootstrap_lower_mean_group_utility")
    if drawdown_limit <= 0 or maximum_drawdown > drawdown_limit:
        reasons.append("maximum_drawdown_exceeds_frozen_limit")
    if maximum_group_exposure > allocated_capital:
        reasons.append("maximum_group_exposure_exceeds_allocation")
    return {
        "scenario": scenario,
        "policy": policy,
        "condition_count": len(condition_values),
        "attempt_count": len(attempts),
        "simulated_filled_conditions": len(simulated_fill_conditions),
        "simulated_fills_per_asset": {
            asset: filled_by_asset[asset] for asset in _ASSETS
        },
        "simulated_no_fill_attempts": simulated_no_fill_count,
        "not_submitted_attempts": not_submitted_count,
        "unknown_after_submit_conditions": len(unknown_conditions),
        "attempt_states": dict(
            sorted(
                Counter(item.observation.observation_state for item in attempts).items()
            )
        ),
        "abstentions_by_reason": dict(sorted(abstentions_by_reason.items())),
        "wins": wins,
        "losses": losses,
        "total_utility_quote": _finite_float(total),
        "mean_condition_utility_quote": _finite_float(
            total / Decimal(len(condition_values))
        ),
        "median_condition_utility_quote": _finite_float(
            _decimal_median(condition_values)
        ),
        "median_simulated_filled_condition_utility_quote": (
            _finite_float(median_simulated_filled_condition_utility)
        ),
        "per_asset_utility_quote": {
            asset: _finite_float(asset_utility[asset]) for asset in _ASSETS
        },
        "allocated_capital_quote": _finite_float(allocated_capital),
        "maximum_group_entry_exposure_quote": _finite_float(maximum_group_exposure),
        "capital_deployed_quote": _finite_float(capital_deployed),
        "maximum_group_exposure_fraction": _finite_float(
            maximum_group_exposure / allocated_capital
        ),
        "market_horizon_capital_time_quote_seconds": _finite_float(
            capital_time_quote_seconds
        ),
        "turnover_quote": _finite_float(turnover),
        "maximum_drawdown_quote": _finite_float(maximum_drawdown),
        "median_positive_group_profit_quote": _finite_float(median_positive),
        "drawdown_limit_quote": _finite_float(drawdown_limit),
        "bootstrap": bootstrap,
        "equity": equity_rows,
        "per_condition_utility": condition_rows,
        "gate_reasons_without_control": reasons,
        "gate_without_control_passed": not reasons,
    }


def _paired_control_gate(
    treatment: dict[str, object],
    control: Mapping[str, object],
    gates: PolymarketRound13EvaluationGates,
) -> None:
    treatment_equity = treatment["equity"]
    control_equity = control["equity"]
    if not isinstance(treatment_equity, list) or not isinstance(control_equity, list):
        raise ValueError("Round 13 equity evidence is malformed")
    treatment_by_group = {
        int(row["event_start_ms"]): float(row["group_utility_quote"])
        for row in treatment_equity
    }
    control_by_group = {
        int(row["event_start_ms"]): float(row["group_utility_quote"])
        for row in control_equity
    }
    if treatment_by_group.keys() != control_by_group.keys():
        raise ValueError("Round 13 treatment/control groups differ")
    differences = tuple(
        treatment_by_group[group] - control_by_group[group]
        for group in sorted(treatment_by_group)
    )
    bootstrap = _bootstrap_group_mean(
        differences,
        series_name=f"{treatment['scenario']}|treatment-minus-control",
        gates=gates,
    )
    treatment_conditions = treatment.get("per_condition_utility")
    control_conditions = control.get("per_condition_utility")
    if not isinstance(treatment_conditions, list) or not isinstance(
        control_conditions, list
    ):
        raise ValueError("Round 13 treatment/control condition evidence is malformed")
    treatment_by_condition = {
        str(row["condition_id"]): row
        for row in treatment_conditions
        if isinstance(row, Mapping)
    }
    control_by_condition = {
        str(row["condition_id"]): row
        for row in control_conditions
        if isinstance(row, Mapping)
    }
    if (
        len(treatment_by_condition) != len(treatment_conditions)
        or len(control_by_condition) != len(control_conditions)
        or treatment_by_condition.keys() != control_by_condition.keys()
    ):
        raise ValueError("Round 13 treatment/control conditions differ")
    condition_differences = [
        {
            "condition_id": condition,
            "asset": treatment_by_condition[condition]["asset"],
            "event_start_ms": treatment_by_condition[condition]["event_start_ms"],
            "treatment_utility_quote": float(
                treatment_by_condition[condition]["utility_quote"]
            ),
            "control_utility_quote": float(
                control_by_condition[condition]["utility_quote"]
            ),
            "difference_quote": float(
                treatment_by_condition[condition]["utility_quote"]
            )
            - float(control_by_condition[condition]["utility_quote"]),
        }
        for condition in sorted(
            treatment_by_condition,
            key=lambda value: (
                int(treatment_by_condition[value]["event_start_ms"]),
                _ASSETS.index(str(treatment_by_condition[value]["asset"])),
            ),
        )
    ]
    non_tied = sum(
        not math.isclose(
            float(item["difference_quote"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for item in condition_differences
    )
    total_difference = float(treatment["total_utility_quote"]) - float(
        control["total_utility_quote"]
    )
    if not math.isclose(
        total_difference,
        math.fsum(float(item["difference_quote"]) for item in condition_differences),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Round 13 paired condition utility does not reconcile")
    reasons = list(treatment["gate_reasons_without_control"])
    if non_tied < gates.minimum_non_tied_treatment_control_conditions:
        reasons.append("insufficient_non_tied_treatment_control_conditions")
    if total_difference <= 0:
        reasons.append("treatment_total_not_above_raw_prior_control")
    if float(bootstrap["lower_95_mean_group_utility_quote"]) <= 0:
        reasons.append("treatment_control_bootstrap_lower_not_positive")
    treatment["control_comparison"] = {
        "treatment_minus_control_total_utility_quote": total_difference,
        "non_tied_condition_count": non_tied,
        "treatment_minus_control_bootstrap": bootstrap,
        "per_condition": condition_differences,
    }
    treatment["gate_reasons"] = reasons
    treatment["gate_passed"] = not reasons


def _score_report(
    batches: Sequence[PolymarketRound13LabelFreeDataset],
    winners: Mapping[str, str],
) -> dict[str, object]:
    snapshots = tuple(
        snapshot for batch in batches for snapshot in batch.calibration_snapshots
    )
    labels = tuple(1 if winners[item.condition_id] == "Up" else 0 for item in snapshots)

    def scores_for(asset: str | None, *, calibrated: bool) -> dict[str, object]:
        selected = [
            index
            for index, item in enumerate(snapshots)
            if asset is None or item.asset == asset
        ]
        probabilities = [
            (
                snapshots[index].calibrated_probability_up
                if calibrated
                else snapshots[index].market_prior_up
            )
            for index in selected
        ]
        return _proper_scores([labels[index] for index in selected], probabilities)

    calibrated = {
        "pooled": scores_for(None, calibrated=True),
        "per_asset": {asset: scores_for(asset, calibrated=True) for asset in _ASSETS},
    }
    prior = {
        "pooled": scores_for(None, calibrated=False),
        "per_asset": {asset: scores_for(asset, calibrated=False) for asset in _ASSETS},
    }
    return {
        "calibrated": calibrated,
        "raw_market_prior": prior,
        "pooled_difference": {
            "log_loss": float(calibrated["pooled"]["log_loss"])
            - float(prior["pooled"]["log_loss"]),
            "brier_score": float(calibrated["pooled"]["brier_score"])
            - float(prior["pooled"]["brier_score"]),
        },
    }


def evaluate_round13_confirmation(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    pipeline_report_sha256: str,
    program: PolymarketRound13Program,
    progress: ProgressCallback | None = None,
    resolution_wait_seconds: int = 900,
    resolution_poll_interval_seconds: int = 15,
    resolution_finalizer: PolymarketResolutionFinalizer | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, object]:
    """Consume one confirmation exactly once after all label-free checks pass."""

    frozen = program.validated()
    emit = progress or (lambda _phase, _payload: None)
    selected_run = str(run_id or "").strip()
    wait_seconds = int(resolution_wait_seconds)
    poll_seconds = int(resolution_poll_interval_seconds)
    if (
        not selected_run
        or wait_seconds < 0
        or wait_seconds > 3_600
        or poll_seconds < 1
        or poll_seconds > 300
    ):
        raise ValueError("Round 13 evaluation run or wait controls are invalid")
    run = (
        store.connect()
        .execute(
            "SELECT status, error, report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
            [selected_run],
        )
        .fetchone()
    )
    if (
        run is None
        or str(run[0]) != "complete"
        or str(run[1] or "")
        or not _is_sha256(run[2])
    ):
        raise ValueError("Round 13 recorder run is not complete and error-free")
    integrity = store.resume_integrity_errors(selected_run, progress=emit)
    if integrity:
        raise ValueError("Round 13 recorder integrity failed: " + "; ".join(integrity))
    from .polymarket_round13_capture import load_round13_capture_manifest

    capture_manifest = load_round13_capture_manifest(
        store,
        run_id=selected_run,
        program=frozen,
    )
    action_ids = _load_pipeline_identity(
        store,
        run_id=selected_run,
        pipeline_report_sha256=pipeline_report_sha256,
        program=frozen,
    )
    batches = _load_label_free_batches(
        store,
        action_ids,
        frozen,
        selected_run,
        emit,
    )
    scenario_ids = tuple(item.dataset_sha256 for item in batches)
    deadline = time.monotonic() + wait_seconds
    maximum_end_row = (
        store.connect()
        .execute(
            "SELECT max(end_ms) FROM polymarket_market_snapshot WHERE run_id = ?",
            [selected_run],
        )
        .fetchone()
    )
    if maximum_end_row is None or maximum_end_row[0] is None:
        raise ValueError("Round 13 market end-time evidence is missing")
    maximum_end_ms = int(maximum_end_row[0])
    before_claim_wait = max(
        0.0,
        (maximum_end_ms - (time.time_ns() // 1_000_000)) / 1000.0,
    )
    remaining_before_claim = deadline - time.monotonic()
    if before_claim_wait > remaining_before_claim:
        raise ValueError(
            "Round 13 outcomes are not yet eligible; no one-use claim was opened"
        )
    if before_claim_wait > 0:
        emit(
            "waiting-before-claim",
            {
                "seconds": before_claim_wait,
                "maximum_market_end_ms": maximum_end_ms,
            },
        )
        sleep(before_claim_wait)
    if time.time_ns() // 1_000_000 < maximum_end_ms:
        raise ValueError(
            "Round 13 market end has not been observed; no one-use claim was opened"
        )
    opened_at_ms, stored_report = _open_claim(
        store,
        run_id=selected_run,
        pipeline_report_sha256=pipeline_report_sha256,
        scenario_dataset_ids=scenario_ids,
        program=frozen,
    )
    if stored_report is not None:
        return stored_report
    emit("claim-opened", {"opened_at_ms": opened_at_ms, "groups": len(batches)})
    try:
        finalizer = resolution_finalizer or PolymarketResolutionFinalizer(store)
        resolution_poll_count = 0
        while True:
            resolution_poll_count += 1
            try:
                finalization = finalizer.finalize(
                    run_id=selected_run,
                    integrity_prevalidated=True,
                    round13_contract_sha256=frozen.contract_sha256,
                )
            except (requests.RequestException, OSError, TimeoutError) as exc:
                raise PolymarketRound13ResolutionPending(
                    "official resolution query was interrupted; exact claim remains open"
                ) from exc
            emit(
                "resolution-finalization",
                {
                    "poll": resolution_poll_count,
                    "status": finalization.status,
                    "finalized": finalization.finalized_count,
                    "markets": finalization.market_count,
                    "pending": len(finalization.pending_condition_ids),
                },
            )
            if finalization.status == "complete":
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PolymarketRound13ResolutionPending(
                    "official resolutions remain pending; exact claim remains open"
                )
            sleep(min(float(poll_seconds), remaining))
        resolutions = load_official_resolutions(store, run_id=selected_run)
        expected_conditions = {
            condition for batch in batches for condition in batch.condition_ids
        }
        winners = {item.condition_id: item.winning_outcome for item in resolutions}
        if winners.keys() != expected_conditions:
            raise ValueError(
                "Round 13 official resolutions are incomplete or excessive"
            )
        classes_by_asset = {
            asset: {item.winning_outcome for item in resolutions if item.asset == asset}
            for asset in _ASSETS
        }
        resolved_per_asset = {
            asset: sum(item.asset == asset for item in resolutions) for asset in _ASSETS
        }
        gates = frozen.evaluation_gates
        if (
            len(batches) < gates.minimum_synchronized_event_groups
            or any(
                resolved_per_asset[asset] < gates.minimum_resolved_markets_per_asset
                for asset in _ASSETS
            )
            or any(
                len(classes_by_asset[asset]) < gates.minimum_outcome_classes_per_asset
                for asset in _ASSETS
            )
        ):
            raise ValueError("Round 13 independent-group or outcome-class gate failed")
        scores = _score_report(batches, winners)
        scenarios: dict[str, object] = {}
        for scenario in frozen.scenarios:
            treatment = _policy_metrics(
                batches,
                winners,
                scenario=scenario.name,
                policy="calibrated",
                program=frozen,
            )
            control = _policy_metrics(
                batches,
                winners,
                scenario=scenario.name,
                policy="raw_market_prior",
                program=frozen,
            )
            _paired_control_gate(treatment, control, gates)
            scenarios[scenario.name] = {
                "calibrated_treatment": treatment,
                "raw_market_prior_control": control,
                "no_trade_control_total_utility_quote": 0.0,
            }
            emit(
                "scenario-scored",
                {
                    "scenario": scenario.name,
                    "gate_passed": treatment["gate_passed"],
                },
            )
        primary = scenarios["primary"]["calibrated_treatment"]  # type: ignore[index]
        stress_passed = all(
            bool(value["calibrated_treatment"]["gate_passed"])  # type: ignore[index]
            for value in scenarios.values()
        )
        created_at_ms = time.time_ns() // 1_000_000
        report_without_hash: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION,
            "round": 13,
            "contract_sha256": frozen.contract_sha256,
            "run_id": selected_run,
            "run_report_sha256": str(run[2]),
            "capture_manifest_sha256": capture_manifest["manifest_sha256"],
            "pipeline_report_sha256": pipeline_report_sha256,
            "scenario_dataset_sha256": list(scenario_ids),
            "resolution_evidence_sha256": sorted(
                item.evidence_sha256 for item in resolutions
            ),
            "resolution_finalization": {
                **finalization.asdict(),
                "poll_count": resolution_poll_count,
                "integrity_prevalidated": True,
            },
            "opened_before_resolution_query_at_ms": opened_at_ms,
            "created_at_ms": created_at_ms,
            "utc_span_ms": {
                "start": batches[0].event_start_ms,
                "end": batches[-1].event_start_ms + 300_000,
            },
            "data": {
                "independent_synchronized_groups": len(batches),
                "resolved_conditions": len(resolutions),
                "resolved_markets_per_asset": resolved_per_asset,
                "outcome_classes_per_asset": {
                    asset: sorted(classes_by_asset[asset]) for asset in _ASSETS
                },
                "real_data_only": True,
                "labels_opened_after_claim": True,
            },
            "allocated_confirmation_capital_quote": format(
                frozen.confirmation_capital_quote, "f"
            ),
            "execution_scenarios": [scenario.asdict() for scenario in frozen.scenarios],
            "executable_evaluation_gates": gates.asdict(),
            "proper_scores": scores,
            "scenarios": scenarios,
            "primary_gate_passed": bool(primary["gate_passed"]),
            "all_stress_gates_passed": stress_passed,
            "confirmation_passed": bool(primary["gate_passed"] and stress_passed),
            "after_cost_edge_confirmed": bool(primary["gate_passed"] and stress_passed),
            "settlement_overhead_measured": False,
            "authenticated_lifecycle_proven": False,
            "annualized_roi_available": False,
            "profitability_claim": False,
            "paper_authority": False,
            "live_trading_authority": False,
            "ai_edge_claim": False,
        }
        report_sha256 = _sha256(report_without_hash)
        report = {**report_without_hash, "report_sha256": report_sha256}
        connection = store.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "INSERT INTO polymarket_round13_evaluation_report VALUES "
                "(?, ?, ?, ?, ?, ?, ?)",
                [
                    report_sha256,
                    POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION,
                    frozen.contract_sha256,
                    selected_run,
                    pipeline_report_sha256,
                    created_at_ms,
                    _canonical_json(report),
                ],
            )
            connection.execute(
                """
                UPDATE polymarket_round13_evaluation_claim
                SET status = 'complete', report_sha256 = ?, error = ''
                WHERE contract_sha256 = ? AND status = 'opened'
                """,
                [report_sha256, frozen.contract_sha256],
            )
            claim_state = connection.execute(
                """
                SELECT status, report_sha256, error
                FROM polymarket_round13_evaluation_claim
                WHERE contract_sha256 = ?
                """,
                [frozen.contract_sha256],
            ).fetchone()
            if claim_state != ("complete", report_sha256, ""):
                raise ValueError("Round 13 completed claim update did not reconcile")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return report
    except PolymarketRound13ResolutionPending:
        raise
    except Exception as exc:
        store.connect().execute(
            """
            UPDATE polymarket_round13_evaluation_claim
            SET status = 'failed', error = ?
            WHERE contract_sha256 = ? AND status = 'opened'
            """,
            [f"{exc.__class__.__name__}:{exc}", frozen.contract_sha256],
        )
        raise


def load_round13_evaluation_report(
    store: PolymarketEvidenceStore,
    *,
    report_sha256: str,
) -> Mapping[str, object]:
    if not _is_sha256(report_sha256):
        raise ValueError("Round 13 report digest is invalid")
    _ensure_evaluation_tables(store)
    row = (
        store.connect()
        .execute(
            "SELECT report_json FROM polymarket_round13_evaluation_report "
            "WHERE report_sha256 = ?",
            [report_sha256],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 13 report is missing")
    payload = _strict_json(row[0], name="Round 13 report")
    if not isinstance(payload, Mapping):
        raise ValueError("Round 13 report is not an object")
    unhashed = dict(payload)
    claimed = unhashed.pop("report_sha256", None)
    if claimed != report_sha256 or _sha256(unhashed) != report_sha256:
        raise ValueError("Round 13 report hash differs")
    return payload


__all__ = [
    "POLYMARKET_ROUND13_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION",
    "PolymarketRound13ResolutionPending",
    "evaluate_round13_confirmation",
    "load_round13_evaluation_report",
]
