#!/usr/bin/env python3
"""Freeze sealed AI cases behind sanitized access and prompt envelopes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES,
)
from simple_ai_trading.polymarket_round27_economics import Round27EconomicConfig
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
)
from simple_ai_trading.polymarket_round28_ai_cases import (
    materialize_round28_ai_cases,
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    validate_round28_ai_host_report,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    round28_ai_inference_report_from_mapping,
    run_round28_ai_inference,
    validate_round28_ai_inference_report,
)
from simple_ai_trading.polymarket_round28_ai_prompt_envelope import (
    validate_round28_ai_prompt_envelope_report,
)
from simple_ai_trading.polymarket_round28_ai_sealed_access import (
    validate_round28_ai_sealed_access_receipt,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    compose_round28_feature_rows,
    load_round28_book_ticker_overlay,
)


_PREREGISTRATION = Path(
    "docs/model-research/polymarket/"
    "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
)
_RESULT_SCHEMA_VERSION = "polymarket-round28-ai-sealed-case-result-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--round27-feature-store", type=Path, required=True)
    parser.add_argument("--round28-overlay-store", type=Path, required=True)
    parser.add_argument("--sealed-source-database", type=Path, required=True)
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--selection-ai-case-panel", type=Path, required=True)
    parser.add_argument("--ai-selection-claim", type=Path, required=True)
    parser.add_argument("--sealed-access-receipt", type=Path, required=True)
    parser.add_argument("--nominated-host-report", type=Path)
    parser.add_argument("--nominated-prompt-envelope-report", type=Path)
    parser.add_argument("--sealed-case-panel", type=Path, required=True)
    parser.add_argument("--sealed-inference-report", type=Path, required=True)
    parser.add_argument("--sealed-case-result", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 sealed AI case path must not be a symlink")
    return selected.resolve(strict=strict)


def _result(
    *,
    access_receipt_sha256: str,
    ai_selection_sha256: str,
    status: str,
    model_id: str | None,
    prompt_envelope_report_sha256: str | None,
    panel_sha256: str | None,
    inference_report_sha256: str | None,
) -> dict[str, object]:
    if status not in {"no_candidate_nominated", "sealed_inference_frozen"}:
        raise ValueError("Round 28 sealed AI case status differs")
    present = status == "sealed_inference_frozen"
    if present != all(
        value is not None
        for value in (
            model_id,
            prompt_envelope_report_sha256,
            panel_sha256,
            inference_report_sha256,
        )
    ):
        raise ValueError("Round 28 sealed AI case result binding differs")
    body: dict[str, object] = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "sealed_access_receipt_sha256": access_receipt_sha256,
        "ai_selection_sha256": ai_selection_sha256,
        "status": status,
        "nominated_model_id": model_id,
        "prompt_envelope_report_sha256": prompt_envelope_report_sha256,
        "sealed_case_panel_sha256": panel_sha256,
        "sealed_inference_report_sha256": inference_report_sha256,
        "selection_metrics_accessed": False,
        "target_accessed": False,
        "outcome_accessed": False,
        "future_books_accessed": False,
        "pnl_accessed": False,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["result_sha256"] = canonical_sha256(body)
    return body


def _progress(model_id: str):
    def report(completed: int, total: int) -> None:
        if completed == total or completed == 1 or completed % 5 == 0:
            print(
                json.dumps(
                    {
                        "observed_at_ms": time.time_ns() // 1_000_000,
                        "phase": "sealed_target_free_ai_inference",
                        "model_id": model_id,
                        "completed_cases": completed,
                        "total_cases": total,
                        "selection_metrics_accessed": False,
                        "target_accessed": False,
                        "outcome_accessed": False,
                        "orders_submitted": False,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    inputs = {
        "feature": _resolve(repository, args.round27_feature_store, strict=True),
        "overlay": _resolve(repository, args.round28_overlay_store, strict=True),
        "source": _resolve(repository, args.sealed_source_database, strict=True),
        "manifest": _resolve(
            repository,
            args.selection_input_manifest,
            strict=True,
        ),
        "selection": _resolve(repository, args.selection_claim, strict=True),
        "selection_ai_panel": _resolve(
            repository,
            args.selection_ai_case_panel,
            strict=True,
        ),
        "ai_selection": _resolve(
            repository,
            args.ai_selection_claim,
            strict=True,
        ),
        "access": _resolve(
            repository,
            args.sealed_access_receipt,
            strict=True,
        ),
    }
    host_path = (
        None
        if args.nominated_host_report is None
        else _resolve(repository, args.nominated_host_report, strict=True)
    )
    envelope_path = (
        None
        if args.nominated_prompt_envelope_report is None
        else _resolve(
            repository,
            args.nominated_prompt_envelope_report,
            strict=True,
        )
    )
    outputs = {
        "panel": _resolve(repository, args.sealed_case_panel, strict=False),
        "inference": _resolve(
            repository,
            args.sealed_inference_report,
            strict=False,
        ),
        "result": _resolve(repository, args.sealed_case_result, strict=False),
    }
    optional_inputs = tuple(
        path for path in (host_path, envelope_path) if path is not None
    )
    all_paths = (*inputs.values(), *optional_inputs, *outputs.values())
    if (
        any(not path.is_file() for path in inputs.values())
        or any(not path.is_file() for path in optional_inputs)
        or any(
            Path(f"{inputs[key]}.wal").exists()
            for key in ("feature", "overlay", "source")
        )
        or len(set(all_paths)) != len(all_paths)
        or any(path.is_symlink() for path in outputs.values() if path.exists())
    ):
        raise ValueError("Round 28 sealed target-free AI paths differ")

    contract = load_round27_model_contract(repository)
    ai_contract = load_round28_ai_contract(repository)
    preregistration = load_mapping(repository / _PREREGISTRATION)
    manifest = load_mapping(inputs["manifest"])
    selection_claim = load_mapping(inputs["selection"])
    selection_ai_panel = load_mapping(inputs["selection_ai_panel"])
    ai_selection_claim = load_mapping(inputs["ai_selection"])
    receipt, pair, _prior_panel, ai_selection = (
        validate_round28_ai_sealed_access_receipt(
            load_mapping(inputs["access"]),
            contract=contract,
            preregistration=preregistration,
            selection_input_manifest=manifest,
            selection_claim=selection_claim,
            selection_ai_case_panel=selection_ai_panel,
            ai_selection_claim=ai_selection_claim,
        )
    )
    has_nomination = ai_selection.nominated_model_id is not None
    if has_nomination != (host_path is not None and envelope_path is not None):
        raise ValueError("Round 28 sealed AI prompt-envelope nomination differs")
    if not has_nomination:
        if outputs["panel"].exists() or outputs["inference"].exists():
            raise ValueError("Round 28 sealed AI artifacts exist without nomination")
        result = _result(
            access_receipt_sha256=str(receipt["receipt_sha256"]),
            ai_selection_sha256=ai_selection.selection_sha256,
            status="no_candidate_nominated",
            model_id=None,
            prompt_envelope_report_sha256=None,
            panel_sha256=None,
            inference_report_sha256=None,
        )
        if outputs["result"].exists() and load_mapping(outputs["result"]) != result:
            raise ValueError("Round 28 persisted sealed AI case result differs")
        artifact_writer(outputs["result"], "result_sha256")(result)
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
        return 0

    host_report, candidate = validate_round28_ai_host_report(
        load_mapping(host_path),
        contract=ai_contract,
    )
    envelope = validate_round28_ai_prompt_envelope_report(
        load_mapping(envelope_path),
        contract=ai_contract,
        host_qualification_report=host_report,
    )
    envelope_candidate = envelope.get("candidate")
    if (
        not isinstance(envelope_candidate, Mapping)
        or envelope.get("passed") is not True
        or candidate.model_id != ai_selection.nominated_model_id
        or candidate.runtime_digest != ai_selection.nominated_runtime_digest
        or envelope_candidate.get("model_id") != candidate.model_id
        or envelope_candidate.get("runtime_digest") != candidate.runtime_digest
    ):
        raise ValueError("Round 28 sealed AI nominated prompt envelope differs")

    with Round27FeatureStore(inputs["feature"], read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        all_base_rows = tuple(
            sorted(feature_store.load_rows(), key=lambda row: row.decision_time_ms)
        )
        sealed_base_rows = feature_store.load_rows(slot_id="stage1-c")
    overlay_rows, overlay_report = load_round28_book_ticker_overlay(
        inputs["overlay"]
    )
    composed = compose_round28_feature_rows(
        base_rows=all_base_rows,
        overlay_rows=overlay_rows,
        report=overlay_report,
    )
    sealed_decisions = {row.decision_time_ms for row in sealed_base_rows}
    sealed_rows = tuple(
        row for row in composed if row.decision_time_ms in sealed_decisions
    )
    condition_ids = tuple(sorted({row.condition_id for row in sealed_rows}))
    run_ids = {row.run_id for row in sealed_rows}
    if not condition_ids or len(run_ids) != 1:
        raise ValueError("Round 28 sealed AI feature population differs")
    run_id = next(iter(run_ids))
    source_binding_sha256 = canonical_sha256(
        {
            "schema_version": "polymarket-round28-ai-sealed-source-binding-v2",
            "feature_store_audit_sha256": feature_audit["audit_sha256"],
            "overlay_report_sha256": overlay_report["report_sha256"],
            "selection_input_manifest_sha256": manifest["manifest_sha256"],
            "selection_claim_sha256": selection_claim["claim_sha256"],
            "sealed_access_receipt_sha256": receipt["receipt_sha256"],
            "ai_selection_sha256": ai_selection.selection_sha256,
            "condition_population_sha256": canonical_sha256(list(condition_ids)),
            "selection_metrics_accessed": False,
            "target_accessed": False,
            "outcome_accessed": False,
        }
    )
    config = Round27EconomicConfig(
        minimum_executed_trades=(
            POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES
        )
    ).validated()
    if outputs["panel"].exists():
        panel = round28_ai_case_panel_from_mapping(load_mapping(outputs["panel"]))
        if (
            panel.partition_role != "sealed"
            or panel.source_run_id != run_id
            or panel.model_name != pair.augmented_model.model_name
            or panel.model_sha256 != pair.augmented_model.model_sha256
            or panel.selection_claim_sha256 != selection_claim["claim_sha256"]
            or panel.source_audit_sha256 != source_binding_sha256
            or panel.evaluated_condition_ids_sha256
            != canonical_sha256(list(condition_ids))
        ):
            raise ValueError("Round 28 persisted sealed AI panel differs")
    else:
        with PolymarketEvidenceStore(
            inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=run_id,
                condition_ids=condition_ids,
            )
            panel = materialize_round28_ai_cases(
                role="sealed",
                rows=sealed_rows,
                selected_model=pair.augmented_model,
                selection_claim_sha256=str(selection_claim["claim_sha256"]),
                markets=markets,
                source_audit_sha256=source_binding_sha256,
                config=config,
                book_batches=economic_book_batches(
                    source,
                    run_id=run_id,
                    condition_ids=condition_ids,
                    maximum_conditions=(
                        config.maximum_conditions_per_book_batch
                    ),
                ),
            )
        artifact_writer(outputs["panel"], "panel_sha256")(panel.asdict())
    if outputs["inference"].exists():
        inference = round28_ai_inference_report_from_mapping(
            load_mapping(outputs["inference"])
        )
        validate_round28_ai_inference_report(
            inference.asdict(),
            contract=ai_contract,
            host_qualification_report=host_report,
            panel=panel,
        )
    else:
        inference = run_round28_ai_inference(
            panel=panel,
            candidate=candidate,
            contract=ai_contract,
            host_qualification_report=host_report,
            progress=_progress(candidate.model_id),
        )
        artifact_writer(outputs["inference"], "report_sha256")(
            inference.asdict()
        )
    result = _result(
        access_receipt_sha256=str(receipt["receipt_sha256"]),
        ai_selection_sha256=ai_selection.selection_sha256,
        status="sealed_inference_frozen",
        model_id=candidate.model_id,
        prompt_envelope_report_sha256=str(envelope["report_sha256"]),
        panel_sha256=panel.panel_sha256,
        inference_report_sha256=inference.report_sha256,
    )
    if outputs["result"].exists() and load_mapping(outputs["result"]) != result:
        raise ValueError("Round 28 persisted sealed AI case result differs")
    artifact_writer(outputs["result"], "result_sha256")(result)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
