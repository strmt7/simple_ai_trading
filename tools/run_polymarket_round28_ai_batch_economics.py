#!/usr/bin/env python3
"""Evaluate all qualified Round 28 AI vetoes in one matched book scan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import time

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    economic_book_batches,
    load_mapping,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore
from simple_ai_trading.polymarket_round28_ai_batch_economics import (
    evaluate_round28_ai_candidate_batch,
)
from simple_ai_trading.polymarket_round28_ai_cases import (
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_economics import (
    validate_round28_ai_economic_report,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    validate_round28_ai_host_report,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    validate_round28_ai_inference_report,
)
from simple_ai_trading.polymarket_round28_operator import (
    validate_round28_economic_report,
    validate_round28_selection_input_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument(
        "--host-report",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for each qualified candidate, in inference-report order.",
    )
    parser.add_argument(
        "--inference-report",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for each qualified candidate, in host-report order.",
    )
    parser.add_argument("--round27-target-store", type=Path, required=True)
    parser.add_argument("--selection-source-database", type=Path, required=True)
    parser.add_argument("--selection-input-manifest", type=Path, required=True)
    parser.add_argument("--selection-claim", type=Path, required=True)
    parser.add_argument("--baseline-economic-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 17))
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 AI economic path must not be a symlink")
    return selected.resolve(strict=strict)


def _target_role(audit: Mapping[str, object]) -> Mapping[str, object]:
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Round 28 AI target roles differ")
    matches = tuple(
        item
        for item in roles
        if isinstance(item, Mapping) and item.get("role") == "selection"
    )
    if len(matches) != 1 or matches[0].get("finalized") is not True:
        raise ValueError("Round 28 AI selection target role differs")
    return matches[0]


def _slug(model_id: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    if not value:
        raise ValueError("Round 28 AI model identifier differs")
    return value


def _progress(phase: str, **detail: object) -> None:
    print(
        json.dumps(
            {
                "observed_at_ms": time.time_ns() // 1_000_000,
                "phase": phase,
                **detail,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.host_report) != len(args.inference_report):
        raise ValueError("Round 28 AI host and inference counts differ")
    repository = args.repository.resolve(strict=True)
    fixed_inputs = {
        "panel": _resolve(repository, args.case_panel, strict=True),
        "target": _resolve(repository, args.round27_target_store, strict=True),
        "source": _resolve(
            repository,
            args.selection_source_database,
            strict=True,
        ),
        "manifest": _resolve(
            repository,
            args.selection_input_manifest,
            strict=True,
        ),
        "selection": _resolve(repository, args.selection_claim, strict=True),
        "baseline": _resolve(
            repository,
            args.baseline_economic_report,
            strict=True,
        ),
    }
    host_paths = tuple(
        _resolve(repository, path, strict=True) for path in args.host_report
    )
    inference_paths = tuple(
        _resolve(repository, path, strict=True) for path in args.inference_report
    )
    output_directory = _resolve(repository, args.output_directory, strict=False)
    all_inputs = tuple(fixed_inputs.values()) + host_paths + inference_paths
    if (
        any(not path.is_file() for path in all_inputs)
        or len(set(all_inputs)) != len(all_inputs)
        or any(
            Path(f"{fixed_inputs[key]}.wal").exists()
            for key in ("target", "source")
        )
        or output_directory in all_inputs
        or (output_directory.exists() and not output_directory.is_dir())
    ):
        raise ValueError("Round 28 AI batch economic paths differ")

    load_round27_model_contract(repository)
    contract = load_round28_ai_contract(repository)
    panel = round28_ai_case_panel_from_mapping(load_mapping(fixed_inputs["panel"]))
    if panel.partition_role != "selection":
        raise ValueError("Round 28 AI batch requires the selection case panel")
    manifest = validate_round28_selection_input_manifest(
        load_mapping(fixed_inputs["manifest"])
    )
    selection_claim = load_mapping(fixed_inputs["selection"])
    baseline = load_mapping(fixed_inputs["baseline"])
    candidate_evidence: list[tuple[dict[str, object], dict[str, object]]] = []
    model_ids: list[str] = []
    for host_path, inference_path in zip(
        host_paths,
        inference_paths,
        strict=True,
    ):
        host_report = load_mapping(host_path)
        validated_host, candidate = validate_round28_ai_host_report(
            host_report,
            contract=contract,
        )
        inference = validate_round28_ai_inference_report(
            load_mapping(inference_path),
            contract=contract,
            host_qualification_report=validated_host,
            panel=panel,
        )
        candidate_evidence.append((validated_host, inference.asdict()))
        model_ids.append(candidate.model_id)
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Round 28 AI batch candidate is duplicated")
    output_paths = tuple(
        output_directory
        / f"round28-ai-{_slug(model_id)}-selection-economics-v1.json"
        for model_id in model_ids
    )
    if any(path.is_symlink() for path in output_paths if path.exists()):
        raise ValueError("Round 28 AI economic output must not be a symlink")

    with Round27TargetStore(fixed_inputs["target"], read_only=True) as target_store:
        target_audit = target_store.audit()
        role_audit = _target_role(target_audit)
        outcomes = target_store.outcomes_up(roles=("selection",))
    condition_ids = tuple(sorted(outcomes))
    if (
        target_audit["audit_sha256"]
        != manifest["round27_target_store_audit_sha256"]
        or len(condition_ids) != panel.evaluated_condition_count
        or canonical_sha256(list(condition_ids))
        != panel.evaluated_condition_ids_sha256
    ):
        raise ValueError("Round 28 AI target lineage differs")
    resolution_sha256 = str(role_audit["evidence_chain_sha256"])
    baseline = validate_round28_economic_report(
        baseline,
        input_manifest=manifest,
        selection_claim=selection_claim,
        resolution_evidence_sha256=resolution_sha256,
    )

    reports: tuple[dict[str, object], ...]
    reused_existing = bool(
        output_paths and all(path.exists() for path in output_paths)
    )
    if reused_existing:
        restored: list[dict[str, object]] = []
        for model_id, path, (host_report, inference_report) in zip(
            model_ids,
            output_paths,
            candidate_evidence,
            strict=True,
        ):
            report = validate_round28_ai_economic_report(load_mapping(path))
            if (
                report["candidate"]["model_id"] != model_id
                or report["host_qualification_report_sha256"]
                != host_report["report_sha256"]
                or report["inference_report_sha256"]
                != inference_report["report_sha256"]
                or report["case_panel_sha256"] != panel.panel_sha256
                or report["round28_economic_report_sha256"]
                != baseline["report_sha256"]
                or report["resolution_evidence_sha256"] != resolution_sha256
            ):
                raise ValueError("Round 28 persisted AI economics lineage differs")
            restored.append(report)
        reports = tuple(restored)
    else:
        _progress(
            "matched-economic-scan-started",
            candidate_count=len(candidate_evidence),
            condition_count=len(outcomes),
        )
        with PolymarketEvidenceStore(
            fixed_inputs["source"],
            read_only=True,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ) as source:
            markets = PolymarketEvidenceReplay.load_markets(
                source,
                run_id=panel.source_run_id,
                condition_ids=condition_ids,
            )
            reports = evaluate_round28_ai_candidate_batch(
                panel=panel,
                candidate_evidence=candidate_evidence,
                contract=contract,
                round28_economic_report=baseline,
                input_manifest=manifest,
                selection_claim=selection_claim,
                markets=markets,
                outcomes_up=outcomes,
                resolution_evidence_sha256=resolution_sha256,
                book_batches=economic_book_batches(
                    source,
                    run_id=panel.source_run_id,
                    condition_ids=condition_ids,
                    maximum_conditions=int(
                        panel.economic_config["maximum_conditions_per_book_batch"]
                    ),
                ),
            )
        for path, report in zip(output_paths, reports, strict=True):
            artifact_writer(path, "report_sha256")(report)
        _progress(
            "matched-economic-scan-completed",
            candidate_count=len(reports),
            source_scan_count=1,
        )
    print(
        json.dumps(
            {
                "candidate_reports": [
                    {
                        "model_id": report["candidate"]["model_id"],
                        "report_sha256": report["report_sha256"],
                        "matched_after_cost_uplift_gate_passed": report[
                            "matched_after_cost_uplift_gate_passed"
                        ],
                    }
                    for report in reports
                ],
                "source_scan_count": 0 if reused_existing else 1,
                "sealed_partition_accessed": False,
                "edge_claim": False,
                "profitability_claim": False,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
