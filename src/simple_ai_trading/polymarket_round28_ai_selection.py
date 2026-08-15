"""Complete candidate accounting and nomination for the Round 28 AI veto."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import hashlib
import json

from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    POLYMARKET_ROUND28_AI_MODEL_IDS,
    validate_round28_ai_contract,
)
from .polymarket_round28_ai_economics import (
    validate_round28_ai_economic_report,
)
from .polymarket_round28_ai_host import (
    Round28AIHostCandidate,
    validate_round28_ai_host_report,
)


POLYMARKET_ROUND28_AI_HOST_FAILURE_SCHEMA_VERSION = (
    "polymarket-round28-ai-host-failure-v1"
)
POLYMARKET_ROUND28_AI_SELECTION_SCHEMA_VERSION = (
    "polymarket-round28-ai-candidate-selection-v1"
)
POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES = 2
_FAILURE_PHASES = frozenset(
    {
        "artifact_download",
        "artifact_verification",
        "runtime_inventory",
        "cold_conformance",
        "warm_conformance",
        "gpu_residency",
        "mandatory_unload",
    }
)
_FAILURE_CODES = frozenset(
    {
        "artifact_hash_or_size_mismatch",
        "artifact_unavailable",
        "conformance_response_invalid",
        "gpu_residency_incomplete",
        "host_latency_limit_exceeded",
        "provider_unavailable",
        "runtime_digest_mismatch",
        "runtime_inventory_invalid",
        "unload_not_observed",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 28 AI {name} SHA-256 differs")
    return selected


def _is_sha256(value: object) -> bool:
    try:
        _sha256(value, name="candidate evidence")
    except ValueError:
        return False
    return True


def build_round28_ai_host_failure(
    *,
    contract: Mapping[str, object],
    model_id: str,
    phase: str,
    error_code: str,
    private_detail_sha256: str,
    observed_at_ms: int,
) -> dict[str, object]:
    """Persist a sanitized terminal host failure without paths or credentials."""

    selected = validate_round28_ai_contract(contract)
    candidates = {
        str(item["model_id"]): item for item in selected["candidate_program"]
    }
    candidate = candidates.get(model_id)
    if (
        candidate is None
        or phase not in _FAILURE_PHASES
        or error_code not in _FAILURE_CODES
        or type(observed_at_ms) is not int
        or observed_at_ms <= 0
    ):
        raise ValueError("Round 28 AI host failure specification differs")
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_HOST_FAILURE_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "model_id": model_id,
        "runtime_model": candidate["runtime_model"],
        "expected_artifact_sha256": candidate["artifact_sha256"],
        "phase": phase,
        "error_code": error_code,
        "private_detail_sha256": _sha256(
            private_detail_sha256,
            name="private failure detail",
        ),
        "observed_at_ms": observed_at_ms,
        "sanitized_no_raw_exception_path_url_or_credential": True,
        "candidate_host_qualified": False,
        "candidate_economically_evaluated": False,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_ai_host_failure(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    selected = validate_round28_ai_contract(contract)
    candidates = {
        str(item["model_id"]): item for item in selected["candidate_program"]
    }
    report = dict(value)
    claimed = _sha256(report.pop("report_sha256", None), name="host failure")
    candidate = candidates.get(str(report.get("model_id", "")))
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version")
        != POLYMARKET_ROUND28_AI_HOST_FAILURE_SCHEMA_VERSION
        or report.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or candidate is None
        or report.get("runtime_model") != candidate["runtime_model"]
        or report.get("expected_artifact_sha256")
        != candidate["artifact_sha256"]
        or report.get("phase") not in _FAILURE_PHASES
        or report.get("error_code") not in _FAILURE_CODES
        or _sha256(
            report.get("private_detail_sha256"),
            name="private failure detail",
        )
        != report.get("private_detail_sha256")
        or type(report.get("observed_at_ms")) is not int
        or int(report["observed_at_ms"]) <= 0
        or report.get("sanitized_no_raw_exception_path_url_or_credential")
        is not True
        or any(
            report.get(field) is not False
            for field in (
                "candidate_host_qualified",
                "candidate_economically_evaluated",
                "edge_claim",
                "profitability_claim",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 AI host failure report differs")
    return {**report, "report_sha256": claimed}


def _decimal(value: object, *, name: str) -> Decimal:
    selected = Decimal(str(value))
    if not selected.is_finite():
        raise ValueError(f"Round 28 AI {name} differs")
    return selected


def _candidate_rank(report: Mapping[str, object]) -> tuple[Decimal, Decimal, Decimal, str]:
    scenarios = report["paired_scenarios"]
    primary = next(item for item in scenarios if item["base_delay_ms"] == 500)
    lower = _decimal(
        primary["paired_condition_bootstrap"]["ci95_lower"],
        name="primary bootstrap lower bound",
    )
    worst_mean = min(
        _decimal(
            item["paired_mean_net_pnl_delta_quote"],
            name="paired mean net PnL",
        )
        for item in scenarios
    )
    worst_drawdown = max(
        _decimal(
            item["maximum_drawdown_delta_fraction"],
            name="drawdown delta",
        )
        for item in scenarios
    )
    return (
        -lower,
        -worst_mean,
        worst_drawdown,
        str(report["candidate"]["model_id"]),
    )


@dataclass(frozen=True, slots=True)
class Round28AICandidateSelection:
    case_panel_sha256: str
    round28_economic_report_sha256: str
    candidate_coverage: tuple[dict[str, object], ...]
    economic_report_sha256: tuple[str, ...]
    nominated_model_id: str | None
    nominated_runtime_digest: str | None
    nominated_report_sha256: str | None
    selection_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_AI_SELECTION_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "case_panel_sha256": self.case_panel_sha256,
            "round28_economic_report_sha256": (
                self.round28_economic_report_sha256
            ),
            "candidate_coverage": [dict(item) for item in self.candidate_coverage],
            "economic_report_sha256": list(self.economic_report_sha256),
            "minimum_economically_evaluated_candidates": (
                POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES
            ),
            "nominated_model_id": self.nominated_model_id,
            "nominated_runtime_digest": self.nominated_runtime_digest,
            "nominated_report_sha256": self.nominated_report_sha256,
            "selection_partition_only": True,
            "sealed_partition_accessed": False,
            "post_selection_retuning_allowed": False,
            "edge_claim": False,
            "profitability_claim": False,
            "orders_submitted": False,
            "trading_authority": False,
        }

    def validated(self) -> "Round28AICandidateSelection":
        coverage_ids = tuple(str(item.get("model_id")) for item in self.candidate_coverage)
        evaluated = tuple(
            item for item in self.candidate_coverage if item.get("status") == "evaluated"
        )
        nominated_absent = self.nominated_model_id is None
        if (
            not all(
                _is_sha256(value)
                for value in (
                    self.case_panel_sha256,
                    self.round28_economic_report_sha256,
                    *self.economic_report_sha256,
                )
            )
            or coverage_ids != POLYMARKET_ROUND28_AI_MODEL_IDS
            or any(
                set(item)
                != {
                    "model_id",
                    "status",
                    "host_evidence_sha256",
                    "runtime_digest",
                    "economic_report_sha256",
                }
                or item.get("status") not in {"evaluated", "host_rejected"}
                or not _is_sha256(item.get("host_evidence_sha256"))
                or (
                    item.get("status") == "evaluated"
                    and (
                        not _is_sha256(item.get("economic_report_sha256"))
                        or not _is_sha256(item.get("runtime_digest"))
                    )
                )
                or (
                    item.get("status") == "host_rejected"
                    and (
                        item.get("economic_report_sha256") is not None
                        or item.get("runtime_digest") is not None
                    )
                )
                for item in self.candidate_coverage
            )
            or len(self.economic_report_sha256) != len(evaluated)
            or tuple(sorted(self.economic_report_sha256))
            != tuple(
                sorted(str(item["economic_report_sha256"]) for item in evaluated)
            )
            or nominated_absent != (self.nominated_runtime_digest is None)
            or nominated_absent != (self.nominated_report_sha256 is None)
            or (
                not nominated_absent
                and (
                    len(evaluated)
                    < POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES
                    or self.nominated_model_id not in coverage_ids
                    or self.nominated_report_sha256 not in self.economic_report_sha256
                    or not _is_sha256(self.nominated_runtime_digest)
                    or not any(
                        item["model_id"] == self.nominated_model_id
                        and item["runtime_digest"]
                        == self.nominated_runtime_digest
                        and item["economic_report_sha256"]
                        == self.nominated_report_sha256
                        for item in evaluated
                    )
                )
            )
            or self.selection_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 28 AI candidate selection differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "selection_sha256": self.selection_sha256,
        }


def select_round28_ai_candidate(
    *,
    contract: Mapping[str, object],
    host_qualification_reports: Sequence[Mapping[str, object]],
    host_failure_reports: Sequence[Mapping[str, object]],
    economic_reports: Sequence[Mapping[str, object]],
    case_panel_sha256: str,
    round28_economic_report_sha256: str,
) -> Round28AICandidateSelection:
    """Account for every preregistered model and nominate at most one."""

    selected_contract = validate_round28_ai_contract(contract)
    expected_case_panel_sha256 = _sha256(case_panel_sha256, name="case panel")
    expected_baseline_sha256 = _sha256(
        round28_economic_report_sha256,
        name="Round 28 economic report",
    )
    hosts: dict[str, tuple[dict[str, object], Round28AIHostCandidate]] = {}
    for raw_report in host_qualification_reports:
        report, candidate = validate_round28_ai_host_report(
            raw_report,
            contract=selected_contract,
        )
        if candidate.model_id in hosts:
            raise ValueError("Round 28 AI host qualification is duplicated")
        hosts[candidate.model_id] = (report, candidate)
    failures: dict[str, dict[str, object]] = {}
    for raw_report in host_failure_reports:
        report = validate_round28_ai_host_failure(
            raw_report,
            contract=selected_contract,
        )
        model_id = str(report["model_id"])
        if model_id in failures:
            raise ValueError("Round 28 AI host failure is duplicated")
        failures[model_id] = report
    if set(hosts) & set(failures):
        raise ValueError("Round 28 AI candidate has conflicting host evidence")
    reports: dict[str, dict[str, object]] = {}
    for raw_report in economic_reports:
        report = validate_round28_ai_economic_report(raw_report)
        candidate = report.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("Round 28 AI economic candidate differs")
        model_id = str(candidate.get("model_id", ""))
        host = hosts.get(model_id)
        if (
            host is None
            or model_id in reports
            or candidate != asdict(host[1])
            or report.get("host_qualification_report_sha256")
            != host[0]["report_sha256"]
        ):
            raise ValueError("Round 28 AI economic host lineage differs")
        reports[model_id] = report
    if set(reports) != set(hosts) or set(hosts) | set(failures) != set(
        POLYMARKET_ROUND28_AI_MODEL_IDS
    ):
        raise ValueError("Round 28 AI candidate coverage is incomplete")
    ordered_reports = tuple(
        reports[model_id]
        for model_id in POLYMARKET_ROUND28_AI_MODEL_IDS
        if model_id in reports
    )
    if ordered_reports and (
        any(
            report.get("case_panel_sha256") != expected_case_panel_sha256
            or report.get("round28_economic_report_sha256")
            != expected_baseline_sha256
            for report in ordered_reports
        )
        or any(report.get("partition_role") != "selection" for report in ordered_reports)
    ):
        raise ValueError("Round 28 AI matched selection population differs")
    qualified = tuple(
        report
        for report in ordered_reports
        if report.get("matched_after_cost_uplift_gate_passed") is True
    )
    nominated = (
        None
        if len(ordered_reports) < POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES
        or not qualified
        else min(qualified, key=_candidate_rank)
    )
    coverage = tuple(
        {
            "model_id": model_id,
            "status": "evaluated" if model_id in reports else "host_rejected",
            "host_evidence_sha256": (
                hosts[model_id][0]["report_sha256"]
                if model_id in hosts
                else failures[model_id]["report_sha256"]
            ),
            "runtime_digest": (
                hosts[model_id][1].runtime_digest if model_id in hosts else None
            ),
            "economic_report_sha256": (
                reports[model_id]["report_sha256"]
                if model_id in reports
                else None
            ),
        }
        for model_id in POLYMARKET_ROUND28_AI_MODEL_IDS
    )
    provisional = Round28AICandidateSelection(
        case_panel_sha256=expected_case_panel_sha256,
        round28_economic_report_sha256=expected_baseline_sha256,
        candidate_coverage=coverage,
        economic_report_sha256=tuple(
            sorted(str(report["report_sha256"]) for report in ordered_reports)
        ),
        nominated_model_id=(
            None if nominated is None else str(nominated["candidate"]["model_id"])
        ),
        nominated_runtime_digest=(
            None
            if nominated is None
            else str(nominated["candidate"]["runtime_digest"])
        ),
        nominated_report_sha256=(
            None if nominated is None else str(nominated["report_sha256"])
        ),
        selection_sha256="",
    )
    return replace(
        provisional,
        selection_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def round28_ai_candidate_selection_from_mapping(
    value: Mapping[str, object],
) -> Round28AICandidateSelection:
    payload = dict(value)
    expected = {
        *Round28AICandidateSelection.__dataclass_fields__,
        "schema_version",
        "ai_contract_sha256",
        "minimum_economically_evaluated_candidates",
        "selection_partition_only",
        "sealed_partition_accessed",
        "post_selection_retuning_allowed",
        "edge_claim",
        "profitability_claim",
        "orders_submitted",
        "trading_authority",
    }
    false_fields = (
        "sealed_partition_accessed",
        "post_selection_retuning_allowed",
        "edge_claim",
        "profitability_claim",
        "orders_submitted",
        "trading_authority",
    )
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_AI_SELECTION_SCHEMA_VERSION
        or payload.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or payload.get("minimum_economically_evaluated_candidates")
        != POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES
        or payload.get("selection_partition_only") is not True
        or any(payload.get(field) is not False for field in false_fields)
        or not isinstance(payload.get("candidate_coverage"), list)
        or not isinstance(payload.get("economic_report_sha256"), list)
    ):
        raise ValueError("Round 28 persisted AI candidate selection differs")
    return Round28AICandidateSelection(
        case_panel_sha256=str(payload["case_panel_sha256"]),
        round28_economic_report_sha256=str(
            payload["round28_economic_report_sha256"]
        ),
        candidate_coverage=tuple(dict(item) for item in payload["candidate_coverage"]),
        economic_report_sha256=tuple(payload["economic_report_sha256"]),
        nominated_model_id=payload["nominated_model_id"],
        nominated_runtime_digest=payload["nominated_runtime_digest"],
        nominated_report_sha256=payload["nominated_report_sha256"],
        selection_sha256=str(payload["selection_sha256"]),
    ).validated()


__all__ = [
    "POLYMARKET_ROUND28_AI_HOST_FAILURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_MINIMUM_EVALUATED_CANDIDATES",
    "POLYMARKET_ROUND28_AI_SELECTION_SCHEMA_VERSION",
    "Round28AICandidateSelection",
    "build_round28_ai_host_failure",
    "round28_ai_candidate_selection_from_mapping",
    "select_round28_ai_candidate",
    "validate_round28_ai_host_failure",
]
