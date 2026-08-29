"""Adjudicate a consumed TradFi funding-history sample shortfall offline."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_polymarket_binance_tradfi_perps_funding_history import (
    align_funding,
    evaluate_history,
    parse_binance_funding,
    parse_binance_klines,
    parse_polymarket_funding,
)
from tools.screen_polymarket_binance_tradfi_perps_funding_prefilter import (
    _canonical_json,
    _canonical_payload_hash,
    _decimal,
    _list,
    _mapping,
    _parse_utc,
    _sha256,
)


def _verified_payload(path: Path, expected_file_hash: str, *, name: str) -> object:
    raw = path.read_bytes()
    if _sha256(raw) != expected_file_hash:
        raise ValueError(f"{name} file hash mismatch")
    return json.loads(raw)


def _load_journal_payloads(
    journal: Mapping[str, object], *, expected_result_hash: str
) -> dict[str, object]:
    if journal.get("status") != "complete":
        raise ValueError("source journal is not complete")
    if journal.get("result_sha256") != expected_result_hash:
        raise ValueError("source journal does not bind the consumed result")
    payloads: dict[str, object] = {}
    for raw in _list(journal.get("requests"), name="journal requests"):
        row = _mapping(raw, name="journal request")
        if row.get("status") != "received" or int(row.get("http_status", 0)) != 200:
            raise ValueError(
                "one or more retained requests did not complete with HTTP 200"
            )
        path = Path(str(row["raw_path"]))
        response = path.read_bytes()
        if _sha256(response) != str(row["payload_sha256"]):
            raise ValueError(f"retained response hash mismatch for {row['label']}")
        payloads[str(row["label"])] = json.loads(response)
    return payloads


def run(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_payload_hash(contract, "result_sha256")
    frozen_at = _parse_utc(contract.get("frozen_at_utc"), name="frozen_at_utc")
    if frozen_at > datetime.now(UTC):
        raise ValueError("frozen_at_utc is in the future")
    if str(contract.get("implementation_sha256") or "") != _sha256(
        Path(__file__).read_bytes()
    ):
        raise ValueError("implementation SHA-256 does not match frozen contract")

    source = _mapping(contract.get("consumed_source"), name="consumed source")
    history_path = Path(str(source["result_path"]))
    history = _mapping(
        _verified_payload(
            history_path, str(source["result_file_sha256"]), name="history result"
        ),
        name="history result",
    )
    if _canonical_payload_hash(history, "result_sha256") != str(
        source["result_sha256"]
    ):
        raise ValueError("consumed history canonical hash mismatch")
    journal = _mapping(
        _verified_payload(
            Path(str(source["journal_path"])),
            str(source["journal_file_sha256"]),
            name="journal",
        ),
        name="journal",
    )
    payloads = _load_journal_payloads(
        journal, expected_result_hash=str(source["result_sha256"])
    )

    economics = _mapping(history.get("economics"), name="economics")
    selected = [
        str(value) for value in _list(history.get("selected_symbols"), name="selected")
    ]
    original_results = _mapping(history.get("symbol_results"), name="symbol results")
    diagnostics: dict[str, object] = {}
    for symbol in selected:
        key = symbol.lower()
        original = _mapping(original_results[symbol], name=f"{symbol} original")
        polymarket_rows = parse_polymarket_funding(
            payloads[f"polymarket-{key}-funding"]
        )
        binance_rows, _ = parse_binance_funding(payloads[f"binance-{key}-funding"])
        kline_rows = parse_binance_klines(payloads[f"binance-{key}-klines"])
        aligned, alignment = align_funding(
            polymarket_rows,
            binance_rows,
            kline_rows,
            orientation=str(original["fixed_orientation"]),
            maximum_timestamp_phase_skew_ms=int(
                economics["maximum_timestamp_phase_skew_ms"]
            ),
        )
        original_evaluation = _mapping(original["result"], name="original evaluation")
        if len(aligned) != int(original_evaluation["aligned_count"]):
            raise ValueError(f"{symbol} retained alignment count changed")
        evaluation = evaluate_history(
            aligned,
            execution_hurdle_bips=_decimal(
                economics["execution_hurdle_bips"], name="execution hurdle"
            ),
            annual_opportunity_hurdle_bips_per_leg=_decimal(
                economics["annual_opportunity_hurdle_bips_per_leg"],
                name="capital hurdle",
            ),
            minimum_aligned_rows=1,
            minimum_regime_rows=int(economics["minimum_regime_rows"]),
        )
        diagnostics[symbol] = {
            "alignment_diagnostics": alignment,
            "economic_role_pass_on_actual_retained_rows": evaluation[
                "economic_role_pass"
            ],
            "fixed_orientation": original["fixed_orientation"],
            "original_frozen_minimum_rows": economics["minimum_aligned_rows"],
            "original_status": original_evaluation["status"],
            "retained_aligned_rows": len(aligned),
            "retained_diagnostic": evaluation,
        }

    all_economically_rejected = all(
        _mapping(value, name="diagnostic").get(
            "economic_role_pass_on_actual_retained_rows"
        )
        is False
        for value in diagnostics.values()
    )
    result: dict[str, object] = {
        "schema_version": (
            "polymarket-binance-tradfi-perps-funding-history-shortfall-adjudication-v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {"path": contract_path.as_posix(), "result_sha256": contract_hash},
        "consumed_source": source,
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "network_requests": 0,
            "orders_transfers_or_account_mutations": 0,
            "paper_or_live_trading_authority": False,
        },
        "diagnostics": diagnostics,
        "adjudication": {
            "accepted_edge": False,
            "candidate_for_books": False,
            "deployment_ready": False,
            "all_symbols_economically_rejected_on_actual_retained_rows": (
                all_economically_rejected
            ),
            "status": (
                "retained_actual_rows_reject_every_symbol"
                if all_economically_rejected
                else "retained_actual_rows_leave_sample_only_survivor"
            ),
        },
        "interpretation_boundary": (
            "This zero-network diagnostic cannot repair the consumed twelve-row "
            "sample gate or promote a candidate. It may only strengthen rejection."
        ),
        "next_action": (
            "request_no_books_and_do_not_repeat_this_family_absent_a_material_"
            "funding_fee_session_or_instrument_change"
            if all_economically_rejected
            else "retain_only_the_frozen_sample_shortfall_and_wait_for_a_"
            "prospectively_distinct_complete_window_before_any_further_access"
        ),
    }
    result_hash = _sha256(_canonical_json(result).encode("ascii"))
    result["result_sha256"] = result_hash
    write_bytes_atomic(
        output_path,
        (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args.contract, args.output)
    print(
        _canonical_json(
            {
                "adjudication": result["adjudication"],
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
