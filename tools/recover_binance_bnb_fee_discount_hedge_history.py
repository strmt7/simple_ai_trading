"""Recover one older BNB funding page and rerun the unchanged public gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
ORIGINAL_TOOL_PATH = ROOT / "tools" / "screen_binance_bnb_fee_discount_hedge.py"
SPEC = importlib.util.spec_from_file_location(
    "screen_binance_bnb_fee_discount_hedge_frozen", ORIGINAL_TOOL_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - installation failure
    raise RuntimeError("cannot import the frozen BNB fee-discount screen")
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)

CONTRACT_PATH = (
    ACTION_VALUE / "binance-bnb-fee-discount-hedge-recovery-contract-v1.json"
)
JOURNAL_PATH = ACTION_VALUE / "binance-bnb-fee-discount-hedge-recovery-journal-v1.json"
RAW_ROOT = ACTION_VALUE / "raw" / "binance-bnb-fee-discount-hedge-recovery-v1"
DEFAULT_OUTPUT = (
    ACTION_VALUE / "binance-bnb-fee-discount-hedge-recovery-v1-2026-08-25.json"
)
SCHEMA_VERSION = "binance-bnb-fee-discount-hedge-recovery-v1"
JOURNAL_SCHEMA_VERSION = "binance-bnb-fee-discount-hedge-recovery-journal-v1"
MAXIMUM_GAP_MS = 12 * 60 * 60 * 1000
REQUEST_SPEC: dict[str, object] = {
    "name": "older_futures_funding_history",
    "method": "GET",
    "url": "https://fapi.binance.com/fapi/v1/fundingRate",
    "parameters": {
        "endTime": 1773302399999,
        "limit": 1000,
        "symbol": "BNBUSDT",
    },
    "raw_filename": "01-older-futures-funding-history.json",
}


def _canonical_json(value: object) -> str:
    return ORIGINAL._canonical_json(value)  # noqa: SLF001


def _sha256(payload: bytes) -> str:
    return ORIGINAL._sha256(payload)  # noqa: SLF001


def _mapping(value: object, *, name: str) -> dict[str, object]:
    return ORIGINAL._mapping(value, name=name)  # noqa: SLF001


def _embedded_hash(payload: Mapping[str, object], *, field: str) -> str:
    return ORIGINAL._embedded_hash(payload, field=field)  # noqa: SLF001


def _display_path(path: Path) -> str:
    return ORIGINAL._display_path(path)  # noqa: SLF001


def _write_hashed_json(path: Path, payload: dict[str, object], *, field: str) -> None:
    payload[field] = _embedded_hash(payload, field=field)
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _load_hashed(path: Path, *, field: str, name: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_bytes()), name=name)
    if payload.get(field) != _embedded_hash(payload, field=field):
        raise ValueError(f"{name} canonical hash mismatch")
    return payload


def _load_sources(
    contract_path: Path = CONTRACT_PATH,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = _load_hashed(
        contract_path, field="result_sha256", name="recovery contract"
    )
    if contract.get("status") != "frozen_before_non_overlapping_recovery_request":
        raise ValueError("recovery contract is not frozen before market access")
    source = _mapping(contract.get("source_binding"), name="source binding")
    paths = {
        "initial_contract": ORIGINAL.CONTRACT_PATH,
        "initial_screen": ORIGINAL.DEFAULT_OUTPUT,
        "initial_journal": ORIGINAL.JOURNAL_PATH,
        "initial_implementation": ORIGINAL_TOOL_PATH,
        "recovery_implementation": Path(__file__),
    }
    for prefix, path in paths.items():
        expected_path = ROOT / str(source[f"{prefix}_path"])
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"{prefix} path mismatch")
        if _sha256(path.read_bytes()) != source[f"{prefix}_file_sha256"]:
            raise ValueError(f"{prefix} file hash mismatch")

    initial_contract = _load_hashed(
        ORIGINAL.CONTRACT_PATH,
        field="result_sha256",
        name="initial contract",
    )
    initial_screen = _load_hashed(
        ORIGINAL.DEFAULT_OUTPUT,
        field="result_sha256",
        name="initial screen",
    )
    initial_journal = _load_hashed(
        ORIGINAL.JOURNAL_PATH,
        field="journal_sha256",
        name="initial journal",
    )
    if initial_contract["result_sha256"] != source["initial_contract_result_sha256"]:
        raise ValueError("initial contract result identity mismatch")
    if initial_screen["result_sha256"] != source["initial_screen_result_sha256"]:
        raise ValueError("initial screen result identity mismatch")
    if initial_journal["journal_sha256"] != source["initial_journal_sha256"]:
        raise ValueError("initial journal identity mismatch")
    if initial_journal.get("status") != "data_complete":
        raise ValueError("initial journal is not data complete")
    if initial_screen.get("failure_reasons") != ["insufficient_complete_inner_months"]:
        raise ValueError("initial screen has an unexpected failure boundary")
    rows = ORIGINAL._funding_rows(  # noqa: SLF001
        initial_screen.get("funding_history_payload")
    )
    if len(rows) != int(source["initial_funding_row_count"]):
        raise ValueError("initial funding row count mismatch")
    expected_end = int(rows[0]["fundingTime"]) - 1
    parameters = _mapping(REQUEST_SPEC["parameters"], name="request parameters")
    if expected_end != parameters["endTime"]:
        raise ValueError("non-overlapping recovery boundary mismatch")
    if contract.get("frozen_request") != REQUEST_SPEC:
        raise ValueError("recovery request sequence mismatch")
    return contract, initial_screen, initial_journal


def _write_journal(path: Path, journal: dict[str, object]) -> None:
    _write_hashed_json(path, journal, field="journal_sha256")


def _create_journal(path: Path, contract: Mapping[str, object]) -> dict[str, object]:
    if path.exists():
        existing = _load_hashed(
            path, field="journal_sha256", name="existing recovery journal"
        )
        raise RuntimeError(
            "one-use recovery journal already exists; rerun is prohibited "
            f"with status {existing.get('status')}"
        )
    journal: dict[str, object] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "contract_result_sha256": contract["result_sha256"],
        "implementation_sha256": _sha256(Path(__file__).read_bytes()),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "status": "active",
        "next_request": None,
        "completed_request_count": 0,
        "responses": [],
    }
    _write_journal(path, journal)
    return journal


def _capture(
    session: requests.Session,
    *,
    journal: dict[str, object],
    journal_path: Path,
    raw_root: Path,
) -> object:
    fingerprint = dict(REQUEST_SPEC)
    fingerprint["parameters"] = dict(
        _mapping(REQUEST_SPEC["parameters"], name="request parameters")
    )
    journal["next_request"] = fingerprint
    _write_journal(journal_path, journal)
    before_ms = time.time_ns() // 1_000_000
    response = session.get(
        str(fingerprint["url"]),
        params=_mapping(fingerprint["parameters"], name="request parameters"),
        timeout=30,
    )
    after_ms = time.time_ns() // 1_000_000
    raw = bytes(response.content)
    raw_path = raw_root / str(fingerprint["raw_filename"])
    write_bytes_atomic(raw_path, raw)
    headers = getattr(response, "headers", {})
    receipt = {
        "status_code": int(response.status_code),
        "payload_bytes": len(raw),
        "payload_sha256": _sha256(raw),
        "raw_path": _display_path(raw_path),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
        "content_type": str(headers.get("Content-Type", "")),
        "retry_after": str(headers.get("Retry-After", "")),
        "response_url": str(response.url),
    }
    journal["responses"] = [{"request": fingerprint, "receipt": receipt}]
    journal["completed_request_count"] = 1
    journal["next_request"] = None
    _write_journal(journal_path, journal)
    if response.status_code == 429:
        raise RuntimeError("Binance rate limit reached; stopped without retry")
    if response.status_code != 200:
        raise RuntimeError(f"Binance returned HTTP {response.status_code}; no retry")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "recovery raw body is not valid JSON; raw body retained"
        ) from exc


def _merge_rows(
    older: list[dict[str, object]], recent: list[dict[str, object]]
) -> tuple[list[dict[str, object]], int]:
    if int(older[-1]["fundingTime"]) >= int(recent[0]["fundingTime"]):
        raise ValueError("older recovery page overlaps the retained page")
    gap_ms = int(recent[0]["fundingTime"]) - int(older[-1]["fundingTime"])
    if gap_ms > MAXIMUM_GAP_MS:
        raise ValueError("recovery-to-retained funding gap exceeds twelve hours")
    merged = [*older, *recent]
    times = [int(row["fundingTime"]) for row in merged]
    if times != sorted(set(times)):
        raise ValueError("merged funding times are not unique and increasing")
    return merged, gap_ms


def _gate_failures(
    *,
    initial_contract: Mapping[str, object],
    evaluation: Mapping[str, object],
    execution: Mapping[str, object],
    scenarios: list[dict[str, object]],
) -> list[str]:
    gate = _mapping(initial_contract["decision_gate"], name="decision gate")
    primary = next(
        row
        for row in scenarios
        if row["scenario"] == gate["primary_non_authoritative_scenario"]
    )
    regimes = _mapping(evaluation["newest_half_regimes"], name="newest regimes")
    required = {
        "direction_down",
        "direction_sideways",
        "direction_up",
        "volatility_high",
        "volatility_regular",
    }
    failures: list[str] = []
    if not bool(execution["top_book_depth_covers_common_quantity"]):
        failures.append("current_minimum_common_quantity_lacks_top_depth")
    if int(evaluation["complete_inner_month_count"]) < int(
        gate["minimum_complete_inner_months"]
    ):
        failures.append("insufficient_complete_inner_months")
    if set(regimes) != required or any(
        int(_mapping(regimes[key], name=key)["count"])
        < int(gate["minimum_rows_per_newest_half_regime"])
        for key in required
    ):
        failures.append("insufficient_newest_half_cross_regime_coverage")
    if ORIGINAL._decimal(  # noqa: SLF001
        primary["worst_monthly_turnover_multiple_to_cover_negative_funding"],
        name="monthly turnover multiple",
    ) > ORIGINAL._decimal(  # noqa: SLF001
        gate["maximum_primary_worst_monthly_turnover_multiple"],
        name="monthly turnover gate",
    ):
        failures.append("primary_worst_monthly_turnover_gate_failed")
    if ORIGINAL._decimal(  # noqa: SLF001
        primary["combined_spread_and_funding_drawdown_turnover_multiple"],
        name="combined turnover multiple",
    ) > ORIGINAL._decimal(  # noqa: SLF001
        gate["maximum_primary_combined_turnover_multiple"],
        name="combined turnover gate",
    ):
        failures.append("primary_combined_turnover_gate_failed")
    return failures


def _terminalize(path: Path, journal: dict[str, object], error: Exception) -> None:
    journal["status"] = "terminal_failure"
    journal["failure"] = {"type": type(error).__name__, "message": str(error)}
    _write_journal(path, journal)


def run(
    *,
    session: requests.Session | None = None,
    contract_path: Path = CONTRACT_PATH,
    journal_path: Path = JOURNAL_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, object]:
    """Run the one-request non-overlapping recovery and unchanged gate."""

    recovery_contract, initial_screen, initial_journal = _load_sources(contract_path)
    initial_contract = _load_hashed(
        ORIGINAL.CONTRACT_PATH,
        field="result_sha256",
        name="initial contract",
    )
    journal = _create_journal(journal_path, recovery_contract)
    started_ms = time.time_ns() // 1_000_000
    try:
        raw_older = _capture(
            session or requests.Session(),
            journal=journal,
            journal_path=journal_path,
            raw_root=raw_root,
        )
        older = ORIGINAL._funding_rows(raw_older)  # noqa: SLF001
        recent = ORIGINAL._funding_rows(  # noqa: SLF001
            initial_screen["funding_history_payload"]
        )
        merged, gap_ms = _merge_rows(older, recent)
        evaluation = ORIGINAL._funding_evaluation(merged)  # noqa: SLF001
        execution = _mapping(
            initial_screen["current_execution"], name="current execution"
        )
        scenarios = ORIGINAL._turnover_scenarios(  # noqa: SLF001
            evaluation=evaluation, execution=execution
        )
        failures = _gate_failures(
            initial_contract=initial_contract,
            evaluation=evaluation,
            execution=execution,
            scenarios=scenarios,
        )
        journal["status"] = "data_complete"
        _write_journal(journal_path, journal)
    except Exception as exc:
        _terminalize(journal_path, journal, exc)
        raise

    finished_ms = time.time_ns() // 1_000_000
    journal_bytes = journal_path.read_bytes()
    qualified = not failures
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "new_request_count": 1,
        "source_binding": {
            "recovery_contract_path": _display_path(contract_path),
            "recovery_contract_result_sha256": recovery_contract["result_sha256"],
            "recovery_implementation_path": _display_path(Path(__file__)),
            "recovery_implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "initial_screen_result_sha256": initial_screen["result_sha256"],
            "initial_journal_sha256": initial_journal["journal_sha256"],
            "recovery_journal_path": _display_path(journal_path),
            "recovery_journal_file_sha256": _sha256(journal_bytes),
            "recovery_journal_sha256": journal["journal_sha256"],
        },
        "history_recovery": {
            "older_row_count": len(older),
            "retained_row_count": len(recent),
            "merged_row_count": len(merged),
            "older_start_time_ms": older[0]["fundingTime"],
            "older_end_time_ms": older[-1]["fundingTime"],
            "retained_start_time_ms": recent[0]["fundingTime"],
            "retained_end_time_ms": recent[-1]["fundingTime"],
            "boundary_gap_ms": gap_ms,
            "overlap_count": 0,
        },
        "current_execution": execution,
        "funding_evaluation": evaluation,
        "turnover_break_even_scenarios": scenarios,
        "merged_funding_history_payload": merged,
        "failure_reasons": failures,
        "verdict": {
            "status": (
                "public_cost_edge_candidate_requires_authenticated_account_commission_turnover_and_realized_fee_evidence"
                if qualified
                else "rejected_recovered_bnb_fee_discount_hedge_prequalification"
            ),
            "qualified_public_prequalification": qualified,
            "accepted_edge": False,
            "profitability_claim": False,
            "credentials_used": False,
            "signed_requests_made": 0,
            "orders_placed": False,
            "trading_authority": False,
        },
    }
    result["result_sha256"] = _embedded_hash(result, field="result_sha256")
    return result


def _failure(error: Exception, *, started_ms: int) -> dict[str, object]:
    finished_ms = time.time_ns() // 1_000_000
    result: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}-terminal-failure-v1",
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "error_type": type(error).__name__,
        "error": str(error),
        "accepted_edge": False,
        "profitability_claim": False,
        "credentials_used": False,
        "signed_requests_made": 0,
        "orders_placed": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _embedded_hash(result, field="result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started_ms = time.time_ns() // 1_000_000
    try:
        result = run()
    except Exception as exc:
        result = _failure(exc, started_ms=started_ms)
        write_bytes_atomic(
            args.output, (_canonical_json(result) + "\n").encode("ascii")
        )
        print(f"terminal_failure={type(exc).__name__}: {exc}")
        print(f"output={args.output}")
        return 1
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(f"new_request_count={result['new_request_count']}")
    print(f"merged_row_count={result['history_recovery']['merged_row_count']}")
    print(
        "qualified_public_prequalification="
        f"{result['verdict']['qualified_public_prequalification']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
